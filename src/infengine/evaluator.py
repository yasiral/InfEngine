"""M13 productized evaluation workflow (``infengine evaluate``).

An A/B *policy* experiment on one dedicated engine, supporting BOTH:

  * single-control experiments (the default recommendation for causal
    interpretability), and
  * multi-control policy experiments.

It reuses the existing M6B-2/M9C machinery — ``WorkloadSpec``, the closed/open
loop drivers, ``_warmup``, ``_measure_phase``, the telemetry collectors and
metric summarization — and the existing control plane (Control Contract V2 +
If-Match + DesiredState/Reconciler). It never mutates policy outside the
baseline/candidate PATCHes and always restores the captured pre-experiment state.

The report states OBSERVED TRADE-OFF / NO MEASURABLE DIFFERENCE / INCONCLUSIVE.
It never names a winner or recommends a production setting; a human decides.
"""

import json
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .experiment import (
    ControlPlaneClient,
    ExperimentError,
    ExperimentSpec,
    MeasurementSpec,
    RunnerConfig,
    SuccessCriteria,
    WorkloadSpec,
    _default_collector_factory,
    _default_send_request,
    _idle_check,
    _measure_phase,
    _warmup,
)

SCHEMA = "infengine.policy_evaluation/1"

# Comparison metrics (engine-neutral where available; engine-native otherwise).
COMPARISON_METRICS: Tuple[str, ...] = (
    "throughput_req_s", "output_tokens_per_s", "ttft_p95_s", "queue_time_p95_s",
    "e2e_p50_s", "e2e_p95_s", "kv_usage", "requests_running", "requests_waiting",
)
DEFAULT_DECISION_METRICS: Tuple[str, ...] = (
    "ttft_p95_s", "e2e_p50_s", "e2e_p95_s", "queue_time_p95_s", "kv_usage",
    "throughput_req_s", "requests_running", "requests_waiting",
)
NO_DIFFERENCE_THRESHOLD = 0.05  # |relative delta| below this is "no measurable difference"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ----------------------------------------------------------------------- specs
@dataclass
class PolicyExperiment:
    experiment_id: str
    engine: str
    model: str
    baseline: Dict[str, Any]
    candidate: Dict[str, Any]
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    warmup_requests: int = 16
    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)
    criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    repeats: int = 1
    alternate: bool = True
    require_if_match: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "engine": self.engine,
            "model": self.model,
            "baseline": dict(self.baseline),
            "candidate": dict(self.candidate),
            "repeats": self.repeats,
            "alternate": self.alternate,
            "workload": {
                "load_model": self.workload.load_model,
                "clients": self.workload.clients,
                "measurement_s": self.workload.measurement_s,
                "arrival_rate": self.workload.arrival_rate,
                "requests": self.workload.requests,
                "prompt_tokens": self.workload.prompt_tokens,
                "max_new_tokens": self.workload.max_new_tokens,
                "seed": self.workload.seed,
            },
        }


# ----------------------------------------------------------------------- loader
def load_experiment(path: str) -> PolicyExperiment:
    try:
        import yaml  # type: ignore
    except ImportError:  # pragma: no cover - yaml is expected in the pilot env
        yaml = None
    with open(path, "r", encoding="utf-8") as handle:
        text = handle.read()
    if yaml is not None:
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ExperimentError("experiment file must be a mapping")
    exp = data.get("experiment", data)
    if not isinstance(exp, dict):
        raise ExperimentError("experiment must be a mapping")
    workload = exp.get("workload") or {}
    baseline = exp.get("baseline")
    candidate = exp.get("candidate")
    if not isinstance(baseline, dict) or not baseline:
        raise ExperimentError("experiment.baseline must be a non-empty mapping")
    if not isinstance(candidate, dict) or not candidate:
        raise ExperimentError("experiment.candidate must be a non-empty mapping")
    if set(baseline) != set(candidate):
        raise ExperimentError(
            "baseline and candidate must control the same set of controls"
        )
    measurement = exp.get("measurement") or {}
    ws = WorkloadSpec(
        load_model=workload.get("profile", workload.get("load_model", "closed_loop")),
        clients=int(workload.get("offered_concurrency", workload.get("clients", 8))),
        measurement_s=float(workload.get("duration_seconds",
                                        workload.get("measurement_s", 60.0))),
        arrival_rate=float(workload.get("arrival_rate", 10.0)),
        requests=int(workload.get("requests", 600)),
        prompt_tokens=workload.get("prompt_tokens", 128),
        max_new_tokens=int(workload.get("max_new_tokens", 64)),
        seed=int(workload.get("seed", 1)),
    )
    measurement_spec = MeasurementSpec(
        window_s=float(measurement.get("window_s", 10.0)),
        poll_interval_s=float(measurement.get("poll_interval_s", 2.0)),
        min_polls=int(measurement.get("min_polls", 3)),
        min_requests=int(measurement.get("min_requests", 8)),
        decision_metrics=list(measurement.get("decision_metrics", DEFAULT_DECISION_METRICS)),
    )
    return PolicyExperiment(
        experiment_id=str(exp.get("experiment_id", "evaluation")),
        engine=str(exp["engine"]),
        model=str(exp.get("model", "")),
        baseline=dict(baseline),
        candidate=dict(candidate),
        workload=ws,
        warmup_requests=int(exp.get("warmup_requests", 16)),
        measurement=measurement_spec,
        repeats=int(exp.get("repetitions", exp.get("repeats", 1))),
        alternate=bool(exp.get("alternate", True)),
        require_if_match=bool(exp.get("require_if_match", True)),
    )


# ------------------------------------------------------------------- preflight
def _capability_index(control: ControlPlaneClient) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for entry in control.capabilities():
        if entry.get("intent"):
            index[entry["intent"]] = entry
        if entry.get("internal_key"):
            index.setdefault(entry["internal_key"], entry)
    return index


def preflight(exp: PolicyExperiment, control: ControlPlaneClient) -> Dict[str, Any]:
    notes: List[str] = []
    if exp.engine not in ("vllm", "sglang"):
        return {"ok": False, "reason": f"unsupported engine {exp.engine!r}"}
    if exp.repeats < 1:
        return {"ok": False, "reason": "repetitions must be >= 1"}
    try:
        health = control.health()
    except ExperimentError as exc:
        return {"ok": False, "reason": f"control plane unreachable: {exc}"}
    if not health.get("admitting"):
        return {"ok": False, "reason": "engine not admitting"}

    index = _capability_index(control)
    resolved_baseline: Dict[str, Any] = {}
    resolved_candidate: Dict[str, Any] = {}
    errors: List[str] = []
    for policy, target, label in ((exp.baseline, resolved_baseline, "baseline"),
                                  (exp.candidate, resolved_candidate, "candidate")):
        for key, value in policy.items():
            entry = index.get(key)
            if entry is None:
                errors.append(f"{label}: unknown control {key!r}")
                continue
            if not entry.get("supported_on_selected_engine"):
                errors.append(f"{label}: {key!r} unsupported on the selected engine")
                continue
            lo, hi = entry.get("min"), entry.get("max")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if lo is not None and value < lo:
                    errors.append(f"{label}: {key}={value} below min {lo}")
                if hi is not None and value > hi:
                    errors.append(f"{label}: {key}={value} above max {hi}")
            target[entry.get("internal_key") or key] = value
    if errors:
        return {"ok": False, "reason": "invalid candidate policy", "errors": errors}

    config = control.get_config()
    revision = config.get("desired_revision")
    if exp.require_if_match and revision is None:
        return {"ok": False, "reason": "desired_revision not exposed (If-Match required)"}
    desired = dict(config.get("desired") or {})
    # Rollback-target feasibility: merge-only PATCH cannot unset a key the
    # experiment would introduce, so every controlled key must already be present.
    internal_keys = set()
    for key in set(exp.baseline) | set(exp.candidate):
        entry = index.get(key)
        if entry:
            internal_keys.add(entry.get("internal_key") or key)
    missing = [k for k in internal_keys if k not in desired]
    if missing:
        return {"ok": False,
                "reason": f"rollback target not restorable: {missing} absent from desired"}

    notes.append("baseline_established" if _policy_applied(desired, resolved_baseline) is not None else "")
    return {
        "ok": True,
        "revision": revision,
        "pre_desired": desired,
        "baseline": resolved_baseline,
        "candidate": resolved_candidate,
        "internal_keys": sorted(internal_keys),
        "notes": [n for n in notes if n],
    }


def _policy_applied(desired: Dict[str, Any], policy: Dict[str, Any]) -> Optional[bool]:
    return None


# ------------------------------------------------------------------ apply/verify
def apply_policy(control: ControlPlaneClient, policy: Dict[str, Any],
                 if_match: Optional[int]) -> Dict[str, Any]:
    status, payload = control.patch_config(policy, if_match=if_match)
    payload = payload or {}
    verified = status == 200 and payload.get("status") == "CONVERGED"
    return {
        "http": status,
        "status": payload.get("status"),
        "verified": verified,
        "observed": payload.get("observed"),
        "steps": payload.get("steps"),
        "rollback": payload.get("rollback"),
        "revision": payload.get("desired_revision"),
    }


# ---------------------------------------------------------------------- runner
def _measure_with_policy(exp: PolicyExperiment, cfg: RunnerConfig,
                         control: ControlPlaneClient, name: str,
                         policy: Dict[str, Any], if_match: Optional[int],
                         probe_spec: ExperimentSpec) -> Dict[str, Any]:
    phase: Dict[str, Any] = {"name": name, "policy": dict(policy)}
    apply = apply_policy(control, policy, if_match)
    phase["apply"] = {k: apply[k] for k in ("http", "status", "verified",
                                            "rollback", "revision")}
    if not apply["verified"]:
        phase["ok"] = False
        phase["failure_reason"] = f"policy not CONVERGED: {apply['status']}"
        return phase
    observed = apply.get("observed") or {}
    mismatched = {k: {"requested": policy[k], "observed": observed.get(k)}
                  for k in policy if observed.get(k) != policy[k]}
    phase["control_evidence"] = {
        "requested": dict(policy),
        "observed": {k: observed.get(k) for k in policy},
        "revision": apply.get("revision"),
        "mismatched": mismatched,
    }
    if mismatched:
        phase["ok"] = False
        phase["failure_reason"] = f"whole-policy verification failed: {mismatched}"
        return phase
    phase["idle_before"] = _idle_check(cfg, probe_spec)
    if not _warmup(probe_spec, cfg):
        phase["ok"] = False
        phase["failure_reason"] = "warmup failed"
        return phase
    measured = _measure_phase(probe_spec, cfg, None)
    phase["ok"] = bool(measured.get("ok"))
    phase["failure_reason"] = measured.get("reason")
    phase["counts"] = measured.get("counts")
    phase["wall_s"] = measured.get("wall_s")
    phase["metrics"] = measured.get("metrics")
    phase["environment"] = measured.get("environment")
    phase["revision_after"] = apply.get("revision")
    return phase


def _phase_order(repeats: int, alternate: bool) -> List[List[str]]:
    order: List[List[str]] = []
    for i in range(repeats):
        if alternate and i % 2 == 1:
            order.append(["candidate", "baseline"])
        else:
            order.append(["baseline", "candidate"])
    return order


def _metric_value(phase: Dict[str, Any], metric: str) -> Optional[float]:
    entry = (phase.get("metrics") or {}).get(metric) or {}
    value = entry.get("value")
    return float(value) if isinstance(value, (int, float)) else None


def _compare(exp: PolicyExperiment,
             baseline_phases: List[Dict[str, Any]],
             candidate_phases: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], str]:
    comparison: Dict[str, Any] = {}
    any_diff = False
    any_adequate = False
    for metric in exp.measurement.decision_metrics:
        bvals = [v for v in (_metric_value(p, metric) for p in baseline_phases) if v is not None]
        cvals = [v for v in (_metric_value(p, metric) for p in candidate_phases) if v is not None]
        if not bvals or not cvals:
            comparison[metric] = {"available": False}
            continue
        any_adequate = True
        bmed, cmed = statistics.median(bvals), statistics.median(cvals)
        delta = (cmed - bmed) / bmed if bmed else None
        rel = None if delta is None else delta
        measurable = rel is not None and abs(rel) >= NO_DIFFERENCE_THRESHOLD
        if measurable:
            any_diff = True
        comparison[metric] = {
            "available": True,
            "baseline_median": bmed,
            "candidate_median": cmed,
            "absolute_difference": (cmed - bmed) if rel is not None else None,
            "relative_difference": rel,
            "baseline_samples": bvals,
            "candidate_samples": cvals,
            "repeat_variability": {
                "baseline": (max(bvals) - min(bvals)) if len(bvals) >= 2 else 0.0,
                "candidate": (max(cvals) - min(cvals)) if len(cvals) >= 2 else 0.0,
            },
            "measurable_difference": measurable,
        }
    if not any_adequate:
        return comparison, "INCONCLUSIVE"
    if any_diff:
        return comparison, "OBSERVED_TRADE_OFF"
    return comparison, "NO_MEASURABLE_DIFFERENCE"


def run_policy_experiment(exp: PolicyExperiment, control_url: str,
                          engine_url: str, telemetry_url: Optional[str] = None,
                          progress=None) -> Dict[str, Any]:
    progress = progress or (lambda _m: None)
    result: Dict[str, Any] = {
        "schema": SCHEMA,
        "experiment_id": exp.experiment_id,
        "engine": exp.engine,
        "model": exp.model,
        "started_at": _iso_now(),
        "spec": exp.to_dict(),
        "phases": [],
        "run_order": [],
        "control_evidence": {},
        "comparison": {},
        "outcome": "",
        "restore": {},
        "failure_reason": None,
    }
    control = ControlPlaneClient(control_url)
    first_key = next(iter(exp.baseline))
    probe_spec = ExperimentSpec(
        experiment_id=exp.experiment_id, engine=exp.engine, model=exp.model,
        baseline={first_key: exp.baseline[first_key]},
        candidate={first_key: exp.candidate[first_key]},
        workload=exp.workload, warmup_requests=exp.warmup_requests,
        measurement=exp.measurement, criteria=exp.criteria,
        require_if_match=exp.require_if_match,
    )
    cfg = RunnerConfig(control_url=control_url, engine_url=engine_url,
                       telemetry_url=telemetry_url or engine_url)
    cfg.collector_factory = _default_collector_factory(probe_spec)
    cfg.send_request = _default_send_request(engine_url, exp.workload)

    try:
        pre = preflight(exp, control)
    except ExperimentError as exc:
        result["failure_reason"] = f"preflight error: {exc}"
        result["finished_at"] = _iso_now()
        return result
    result["preflight"] = pre
    if not pre.get("ok"):
        result["failure_reason"] = f"preflight failed: {pre.get('reason')}"
        result["details"] = pre
        result["finished_at"] = _iso_now()
        return result

    # Establish the baseline as the known starting state.
    order = _phase_order(exp.repeats, exp.alternate)
    result["run_order"] = order
    baseline_phases: List[Dict[str, Any]] = []
    candidate_phases: List[Dict[str, Any]] = []
    rev = pre["revision"]
    failed = False
    try:
        for cycle_index, cycle in enumerate(order):
            for name in cycle:
                policy = pre["baseline"] if name == "baseline" else pre["candidate"]
                progress(f"[cycle {cycle_index + 1}/{len(order)}] {name} {policy}")
                phase = _measure_with_policy(exp, cfg, control, name, policy, rev,
                                             probe_spec)
                rev = phase.get("revision_after") or rev
                result["phases"].append(phase)
                (baseline_phases if name == "baseline" else candidate_phases).append(phase)
                if not phase.get("ok"):
                    failed = True
                    result["failure_reason"] = (
                        f"{name} phase failed: {phase.get('failure_reason')}")
                    break
            if failed:
                break
    finally:
        # Never leave the candidate active: always attempt a verified restore.
        try:
            restore = apply_policy(control, pre["pre_desired"], rev)
        except Exception as exc:
            restore = {"verified": False, "status": "FAILED", "detail": str(exc)}
        result["restore"] = restore

    result["configuration"] = {
        "engine": exp.engine, "model": exp.model,
        "baseline": pre["baseline"], "candidate": pre["candidate"],
        "repetitions": exp.repeats, "run_order": order,
    }
    if not failed:
        comparison, outcome = _compare(exp, baseline_phases, candidate_phases)
        result["comparison"] = comparison
        result["outcome"] = outcome
    else:
        result["outcome"] = "INCOMPLETE"
    result["notes"] = [
        "Outcome is a statement about the OBSERVED trade-off, not a recommendation.",
        "A human decides which policy is preferable for the deployment.",
        "Evidence is specific to this engine version, model, hardware and workload.",
    ]
    result["finished_at"] = _iso_now()
    return result


# ------------------------------------------------------------------------ HTML
def render_evaluation_html(result: Dict[str, Any]) -> str:
    import html as _html

    def esc(v):
        return _html.escape(str(v))

    rows = []
    for metric, entry in (result.get("comparison") or {}).items():
        if not entry.get("available"):
            rows.append(f"<tr><td>{esc(metric)}</td><td colspan='5'>unavailable</td></tr>")
            continue
        rel = entry.get("relative_difference")
        rows.append(
            "<tr><td>{}</td><td>{:.4g}</td><td>{:.4g}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                esc(metric), entry["baseline_median"], entry["candidate_median"],
                "n/a" if rel is None else f"{rel:+.2%}",
                "yes" if entry.get("measurable_difference") else "no",
                len(entry.get("baseline_samples") or [])))
    phases = "".join(
        f"<tr><td>{esc(p.get('name'))}</td><td>{esc(p.get('apply', {}).get('status'))}</td>"
        f"<td>{esc(p.get('apply', {}).get('verified'))}</td>"
        f"<td>{esc(p.get('ok'))}</td><td>{esc(p.get('failure_reason') or '')}</td></tr>"
        for p in result.get("phases", []))
    restore = result.get("restore") or {}
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>InfEngine policy evaluation</title>
<style>body{{font:13px/1.5 Segoe UI,Arial;background:#0f1115;color:#e6e6e6;margin:0}}
.wrap{{max-width:1000px;margin:0 auto;padding:20px}}
table{{border-collapse:collapse;width:100%;margin:8px 0}}th,td{{padding:5px 8px;border-bottom:1px solid #262b33;text-align:left}}
th{{color:#9aa4b2}}h2{{font-size:14px;margin:18px 0 6px}}code,.meta{{color:#9aa4b2}}</style></head>
<body><div class="wrap">
<h1>InfEngine policy evaluation</h1>
<p class="meta">{esc(result.get('experiment_id'))} · engine {esc(result.get('engine'))} · model {esc(result.get('model'))}</p>
<h2>Outcome: {esc(result.get('outcome'))}</h2>
<p class="meta">OBSERVED trade-off only — not a recommendation. A human decides.</p>
<h2>Configuration</h2><pre>{esc(json.dumps(result.get('configuration'), indent=2))}</pre>
<h2>Comparison</h2>
<table><thead><tr><th>metric</th><th>baseline median</th><th>candidate median</th>
<th>relative</th><th>measurable</th><th>baseline samples</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<h2>Phases</h2>
<table><thead><tr><th>phase</th><th>apply</th><th>verified</th><th>ok</th><th>failure</th></tr></thead>
<tbody>{phases}</tbody></table>
<h2>Restore</h2><pre>{esc(json.dumps(restore, indent=2))}</pre>
<p class="meta">{esc(' '.join(result.get('notes') or []))}</p>
</div></body></html>"""


# ------------------------------------------------------------------------- CLI
def run_evaluation_file(path: str, *, as_json: bool = False,
                        control_url: Optional[str] = None,
                        report_path: Optional[str] = None,
                        engine_url: Optional[str] = None,
                        telemetry_url: Optional[str] = None) -> int:
    import os
    from .policy import PolicyError  # reuse error shape

    try:
        exp = load_experiment(path)
    except (OSError, ExperimentError, PolicyError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=__import__("sys").stderr)
        return 2
    control_url = control_url or os.environ.get("INFENGINE_CONTROL_URL",
                                                "http://127.0.0.1:8000")
    engine_url = os.environ.get("INFENGINE_ENGINE_URL") or engine_url
    if not engine_url:
        print("error: engine URL required (set INFENGINE_ENGINE_URL or --engine-url)",
              file=__import__("sys").stderr)
        return 2
    result = run_policy_experiment(
        exp, control_url=control_url, engine_url=engine_url,
        telemetry_url=telemetry_url or os.environ.get("INFENGINE_TELEMETRY_URL"),
        progress=lambda m: print(m, file=__import__("sys").stderr),
    )
    if as_json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(f"outcome: {result.get('outcome')}")
        if result.get("failure_reason"):
            print(f"failure: {result['failure_reason']}")
        for metric, entry in (result.get("comparison") or {}).items():
            if entry.get("available") and entry.get("relative_difference") is not None:
                print(f"  {metric}: {entry['relative_difference']:+.2%} "
                      f"(baseline {entry['baseline_median']:.4g} -> "
                      f"candidate {entry['candidate_median']:.4g})")
        restore = result.get("restore") or {}
        print(f"restore: {restore.get('status')} (verified={restore.get('verified')})")
    if report_path:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(render_evaluation_html(result))
        print(f"report: {report_path}")
    return 0 if result.get("outcome") not in ("INCOMPLETE", "") else 1
