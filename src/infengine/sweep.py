"""M9C: continuous-batching load sweep (CLI-first, no dashboard server).

Runs a configurable multi-level load sweep against an existing vLLM/SGLang
inference endpoint, reusing the experiment runner's workload drivers, prompt
generation, warmup, timeout/drain accounting, telemetry collectors and metric
summaries. Runtime policy is never changed: a sweep observes.

Each level is measured with the SAME ``_warmup`` / ``_measure_phase`` machinery
the A/B experiment runner uses, then the M9B execution-evidence contract is
attached per run. Results are one machine-readable ``result.json``; the local
HTML report (``infengine.report``) is a pure presentation of that result.

Reference: M9C_CONTINUOUS_BATCHING_LOAD_SWEEP.md.
"""

import argparse
import json
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .experiment import (
    ExperimentSpec,
    MeasurementSpec,
    RollbackSpec,
    RunnerConfig,
    SuccessCriteria,
    WorkloadSpec,
    _default_send_request,
    _measure_phase,
    _warmup,
)
from .telemetry import (
    SglangTelemetryCollector,
    VllmTelemetryCollector,
    build_execution_evidence,
)

SWEEP_SCHEMA = "infengine.load_sweep/1"

# Normalized metrics summarized by the reused experiment summarizer, plus the
# runner-derived throughput (added by ``_measure_phase``) and the native
# engine-specific throughput fingerprint.
SWEEP_DECISION_METRICS: Tuple[str, ...] = (
    "ttft_p95_s",
    "queue_time_p95_s",
    "e2e_p50_s",
    "e2e_p95_s",
    "kv_usage",
    "requests_running",
    "requests_waiting",
    "throughput",
)

# Cumulative counters recorded per run and delta'd within a level (reset-safe:
# a decrease or instance change yields null, never a fabricated delta).
CUMULATIVE_COUNTERS: Tuple[str, ...] = (
    "num_preemptions",
    "num_retracted_requests_total",
    "prompt_tokens",
    "generation_tokens",
    "prompt_tokens_total",
    "generation_tokens_total",
)

# M9C.1: native latency-histogram families exposed per run (vLLM and SGLang both
# advertise decode_time; SGLang returns it as UNAVAILABLE — never synthesized).
HISTOGRAM_KEYS: Dict[str, Tuple[str, ...]] = {
    "vllm": ("ttft", "queue_time", "e2e", "prefill_time", "decode_time"),
    "sglang": ("ttft", "queue_time", "e2e", "prefill_time", "decode_time"),
}
HISTOGRAM_LABELS: Dict[str, str] = {
    "ttft": "TTFT",
    "queue_time": "Queue latency",
    "e2e": "End-to-end",
    "prefill_time": "Prefill (measured)",
    "decode_time": "Decode (measured)",
}

_SUMMARY_METRICS: Tuple[str, ...] = (
    "throughput_req_s",
    "output_tokens_per_s",
    "throughput",
    "ttft_p95_s",
    "queue_time_p95_s",
    "e2e_p95_s",
    "kv_usage",
    "requests_running",
    "requests_waiting",
)


# ---------------------------------------------------------------------- spec
@dataclass
class SweepSpec:
    """Everything that defines one load sweep (engine config is NOT included:
    runtime policy is held fixed and never changed by the sweep)."""

    engine: str                       # "vllm" | "sglang"
    model: str
    load_model: str = "closed_loop"   # "closed_loop" | "open_loop"
    levels: List[float] = field(default_factory=lambda: [1.0, 2.0, 4.0, 8.0])
    # workload
    prompt_tokens: Any = 128          # int, or (min, max)
    max_new_tokens: int = 64
    seed: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    # closed_loop
    measurement_s: float = 20.0
    max_requests: int = 4096
    # open_loop
    requests: int = 400
    # shared
    warmup_requests: int = 16
    repeats: int = 1
    timeout_s: float = 60.0
    drain_budget_s: float = 10.0
    # measurement
    window_s: float = 10.0
    poll_interval_s: float = 2.0
    min_polls: int = 3
    min_requests: int = 8
    stale_fraction_max: float = 0.5
    # saturation interpretation thresholds (report-only; no recommendations)
    plateau_gain_min: float = 0.05
    kv_pressure_threshold: float = 0.9
    repeat_spread_max: float = 0.5


# ---------------------------------------------------------------------- utils
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _http_json(url: str, timeout: float = 5.0) -> Optional[Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return None


def _engine_version(engine: str, base_url: str) -> Dict[str, Any]:
    """Best-effort, read-only engine version discovery (never mutates)."""
    base = base_url.rstrip("/")
    if engine == "sglang":
        # /server_info can be slow right after startup; give it more room.
        info = _http_json(f"{base}/server_info", timeout=10.0)
        if isinstance(info, dict):
            return {"version": info.get("version"), "source": "/server_info"}
        return {"version": None, "source": "/server_info (unreachable)"}
    version = _http_json(f"{base}/version")
    if isinstance(version, dict):
        return {"version": version.get("version"), "source": "/version"}
    return {"version": None, "source": "/version (unreachable)"}


def _host_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    try:  # optional, best-effort
        import torch  # noqa: PLC0415

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:  # pragma: no cover - torch optional
        env["torch"] = None
    return env


def _build_collector(engine: str, url: str, measurement: MeasurementSpec):
    kw = {
        "window_s": measurement.window_s,
        "poll_interval": measurement.poll_interval_s,
    }
    if engine == "sglang":
        return SglangTelemetryCollector(url, **kw)
    return VllmTelemetryCollector(url, **kw)


def _numeric(value: Any) -> Optional[float]:
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _counter_deltas(
    baseline: Any, final: Any
) -> Dict[str, Dict[str, Any]]:
    """Within-level cumulative-counter deltas. Reset/instance-safe: a decrease,
    a missing value, or an instance change yields ``value=None`` (never a
    fabricated delta)."""
    out: Dict[str, Dict[str, Any]] = {}
    if baseline is None or final is None:
        return out
    same_instance = getattr(baseline, "instance_id", None) == getattr(final, "instance_id", None)
    for key in CUMULATIVE_COUNTERS:
        before = _numeric(getattr(baseline, "engine_specific", {}).get(key))
        after = _numeric(getattr(final, "engine_specific", {}).get(key))
        if before is None or after is None:
            out[key] = {"value": None, "reason": "unavailable"}
        elif not same_instance:
            out[key] = {"value": None, "reason": "instance_changed"}
        elif after < before:
            out[key] = {"value": None, "reason": "counter_reset"}
        else:
            out[key] = {"value": after - before, "from": before, "to": after}
    return out


def _generation_counter_key(engine: str) -> str:
    return "generation_tokens_total" if engine == "sglang" else "generation_tokens"


def _output_token_throughput(
    engine: str, deltas: Dict[str, Dict[str, Any]], window_s: float,
) -> Dict[str, Any]:
    """Experiment-period output-token throughput derived from the generation-token
    counter delta over the measurement window.

    This is DISTINCT from the engine-native windowed throughput gauge
    (``throughput``): vLLM exposes no gauge (its native value is itself derived
    from the same counter), while SGLang's ``gen_throughput`` gauge is windowed
    over ``decode_log_interval`` and can read 0 under conditions where the
    cumulative counter still advances. A valid zero delta yields a genuine
    ``0.0``; missing/reset/instance-changed counter evidence yields ``None``
    (unavailable), never a fabricated positive value.
    """
    key = _generation_counter_key(engine)
    entry = deltas.get(key) or {}
    out: Dict[str, Any] = {
        "value": None,
        "dominant_quality": "unavailable",
        "stale_fraction": 0.0,
        "polls_usable": None,
        "sample_count": None,
        "counter": key,
        "counter_delta": entry.get("value"),
        "window_s": round(window_s, 4) if window_s else None,
        "boundary": (
            "generation-token counter delta over [post-warmup poll, final poll] "
            "divided by that window's duration; warmup and post-measurement drain "
            "are excluded"
        ),
        "reason": entry.get("reason"),
    }
    if entry.get("value") is not None and window_s and window_s > 0:
        out["value"] = entry["value"] / window_s
        out["dominant_quality"] = "fresh"
        out["sample_count"] = int(entry["value"])
    return out


def _histogram_unavailable_reason(measurement: Optional[Dict[str, Any]]) -> str:
    if not measurement:
        return "not present in execution evidence for this engine"
    if measurement.get("classification") == "UNAVAILABLE":
        return "not emitted by this engine/version (see boundary note)"
    return ("native histogram window could not be established "
            "(insufficient in-window baseline, reset, or missing/stale scrape)")


def _build_histograms(
    engine: str, snapshot: Any, evidence: Any, min_requests: int,
) -> Dict[str, Any]:
    """Assemble per-run native histogram buckets (windowed) from the final
    snapshot's measurement_meta, enriched with execution-evidence metadata."""
    raw = {}
    if snapshot is not None:
        raw = (snapshot.measurement_meta or {}).get("histograms") or {}
    evidence_dict = evidence.to_dict() if evidence is not None else {}
    request_stage = (evidence_dict.get("request_stage") or {})
    families: Dict[str, Any] = {}
    for key in HISTOGRAM_KEYS.get(engine, ()):
        measurement = request_stage.get(key) or {}
        entry = raw.get(key)
        family: Dict[str, Any] = {
            "key": key,
            "label": HISTOGRAM_LABELS.get(key, key),
            "source": measurement.get("source"),
            "unit": measurement.get("unit") or "seconds",
            "scope": measurement.get("scope"),
            "classification": measurement.get("classification"),
            "boundary_note": measurement.get("boundary_note"),
        }
        if entry is None:
            family["available"] = False
            family["reason"] = _histogram_unavailable_reason(measurement)
        else:
            total = int(entry["total"])
            family.update({
                "available": True,
                "reason": None,
                "le": list(entry["le"]),
                "counts": [int(c) for c in entry["counts"]],
                "total": total,
                "window_s": entry["window_s"],
                "low_sample": total < min_requests,
                "zero_observations": total == 0,
            })
        families[key] = family
    return {"families": families}


# --------------------------------------------------------------- level runner
def _workload(spec: SweepSpec, offered: float) -> WorkloadSpec:
    if spec.load_model == "open_loop":
        return WorkloadSpec(
            load_model="open_loop", arrival_rate=float(offered), requests=int(spec.requests),
            prompt_tokens=spec.prompt_tokens, max_new_tokens=spec.max_new_tokens,
            temperature=spec.temperature, top_p=spec.top_p, seed=spec.seed,
            timeout_s=spec.timeout_s, drain_budget_s=spec.drain_budget_s,
        )
    return WorkloadSpec(
        load_model="closed_loop", clients=int(offered), measurement_s=spec.measurement_s,
        max_requests=int(spec.max_requests), prompt_tokens=spec.prompt_tokens,
        max_new_tokens=spec.max_new_tokens, temperature=spec.temperature,
        top_p=spec.top_p, seed=spec.seed, timeout_s=spec.timeout_s,
        drain_budget_s=spec.drain_budget_s,
    )


def _experiment_spec(spec: SweepSpec, workload: WorkloadSpec, tag: str) -> ExperimentSpec:
    """A single-phase ExperimentSpec used ONLY to reuse the experiment runner's
    warmup/measurement machinery. No control PATCH is ever issued."""
    return ExperimentSpec(
        experiment_id=f"sweep-{spec.engine}-{tag}",
        engine=spec.engine,
        model=spec.model,
        baseline={"runtime_concurrency": 1},
        candidate={"runtime_concurrency": 1},
        workload=workload,
        warmup_requests=spec.warmup_requests,
        measurement=MeasurementSpec(
            window_s=spec.window_s, poll_interval_s=spec.poll_interval_s,
            min_polls=spec.min_polls, min_requests=spec.min_requests,
            stale_fraction_max=spec.stale_fraction_max,
            decision_metrics=list(SWEEP_DECISION_METRICS),
        ),
        criteria=SuccessCriteria(),
        rollback=RollbackSpec(),
        require_if_match=False,
    )


def run_level_once(
    spec: SweepSpec,
    offered: float,
    *,
    engine_url: str,
    telemetry_url: str,
    run_index: int,
    gpu_collector: Any = None,
) -> Dict[str, Any]:
    """One repeat of one load level through the reused experiment machinery."""
    workload = _workload(spec, offered)
    exp = _experiment_spec(spec, workload, f"{offered}-{run_index}")
    collector = _build_collector(spec.engine, telemetry_url, exp.measurement)
    cfg = RunnerConfig(
        control_url=engine_url,  # unused by _measure_phase/_warmup
        engine_url=engine_url,
        telemetry_url=telemetry_url,
        collector_factory=lambda _url: collector,
        send_request=_default_send_request(engine_url, workload),
        gpu_collector=gpu_collector,
    )
    started_at = _iso_now()
    idle_snap = collector.poll_once()
    idle = None
    if idle_snap is not None:
        running = getattr(idle_snap, "requests_running", None)
        waiting = getattr(idle_snap, "requests_waiting", None)
        if running is not None and waiting is not None:
            idle = int(running) == 0 and int(waiting) == 0
    warmup_ok = _warmup(exp, cfg) if spec.warmup_requests > 0 else True
    # Counter baseline is taken AFTER warmup so cumulative-counter deltas cover
    # the measurement phase (plus the tiny gap to the first measurement poll),
    # not warmup traffic. Histogram families live in a per-run collector whose
    # rolling window ends at the final poll; the ramp-in exclusion keeps the
    # summarized windows inside the measurement interval.
    t_base = time.monotonic()
    counter_baseline = collector.poll_once()
    t_measure_call = time.monotonic()
    measured = _measure_phase(exp, cfg, None)
    final = collector.latest()
    # The generation-token counter is observed at the post-warmup baseline poll
    # and the final poll. The last measurement poll runs BEFORE phase-end drain,
    # so drain completions never enter the counter or the histograms.
    wall_s = measured.get("wall_s") or 0.0
    counter_window_s = (t_measure_call - t_base) + wall_s
    counter_deltas = _counter_deltas(counter_baseline, final)
    metrics = dict(measured.get("metrics", {}))
    metrics["output_tokens_per_s"] = _output_token_throughput(
        spec.engine, counter_deltas, counter_window_s
    )
    evidence = build_execution_evidence(final) if final is not None else None
    run: Dict[str, Any] = {
        "run_index": run_index,
        "started_at": started_at,
        "finished_at": _iso_now(),
        "offered_load": offered,
        "idle_before": idle,
        "warmup_ok": warmup_ok,
        "ok": bool(measured.get("ok")) and warmup_ok,
        "failure_reason": measured.get("reason") if not measured.get("ok") else (
            None if warmup_ok else "warmup failed (timeouts/errors/incomplete)"
        ),
        "counts": measured.get("counts", {}),
        "wall_s": measured.get("wall_s"),
        "environment": measured.get("environment", {}),
        "metrics": metrics,
        "gpu": measured.get("gpu", {}),
        "counter_deltas": counter_deltas,
        "histograms": _build_histograms(
            spec.engine, final, evidence, spec.min_requests
        ),
        "measurement_window_s": round(counter_window_s, 4),
        "evidence_windows": {
            "normalized_metrics": (
                "median over telemetry polls in [t_start + window_s, t_end] "
                "(measurement ramp-in excluded)"
            ),
            "execution_evidence": (
                "native request-histogram delta over the per-run collector rolling "
                f"window (window_s={spec.window_s}) ending at the final poll; the "
                "cumulative baseline is taken post-warmup, so warmup observations "
                "are excluded"
            ),
            "output_tokens_per_s": (
                "generation-token counter delta over [post-warmup poll, final poll] "
                f"(~{round(counter_window_s, 3)}s) / that window; warmup and "
                "post-measurement drain excluded"
            ),
            "counter_deltas": (
                "cumulative counters delta'd from the post-warmup poll to the "
                "final poll (warmup and drain excluded)"
            ),
            "throughput_req_s": (
                "measurement-period completions / actual measurement span; "
                "post-measurement drain excluded"
            ),
            "final_poll": (
                "the final measurement poll occurs BEFORE phase-end drain begins, "
                "so drain completions are not in the histograms or counters; it can "
                "include a few milliseconds of post-t1 measurement traffic"
            ),
            "note": (
                "Cumulative/rolling native metrics are NOT fully isolated to a "
                "single load level; the intervals above are the actual "
                "observation windows."
            ),
        },
        "execution_evidence": evidence.to_dict() if evidence is not None else None,
        "snapshot": final.to_dict() if final is not None else None,
    }
    return run


# ---------------------------------------------------------------- aggregation
def _metric_value(run: Dict[str, Any], metric: str) -> Optional[float]:
    entry = (run.get("metrics") or {}).get(metric) or {}
    return _numeric(entry.get("value"))


def _metric_quality(run: Dict[str, Any], metric: str) -> Optional[str]:
    return ((run.get("metrics") or {}).get(metric) or {}).get("dominant_quality")


def _evidence_measurement(run: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    ev = run.get("execution_evidence") or {}
    for section in ("request_stage", "engine_level"):
        entry = (ev.get(section) or {}).get(key)
        if entry is not None:
            return entry
    return None


def _min_int(values: List[Any]) -> Optional[int]:
    nums = [int(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return min(nums) if nums else None


def _aggregate_level(runs: List[Dict[str, Any]], spec: SweepSpec) -> Dict[str, Any]:
    ok_runs = [r for r in runs if r.get("ok")]
    censored = any(
        int(r.get("counts", {}).get("timeout", 0)) > 0
        or int(r.get("counts", {}).get("incomplete", 0)) > 0
        for r in runs
    )
    aggregates: Dict[str, Any] = {}
    for metric in _SUMMARY_METRICS:
        vals = [v for v in (_metric_value(r, metric) for r in ok_runs) if v is not None]
        entries = [(r.get("metrics", {}).get(metric) or {}) for r in ok_runs]
        dominant = next(
            (e.get("dominant_quality") for e in entries if e.get("dominant_quality")),
            None,
        )
        polls = _min_int([e.get("polls_usable") for e in entries])
        samples = _min_int([e.get("sample_count") for e in entries])
        adequate = bool(
            vals
            and dominant not in ("unavailable", "disabled", "no_samples")
            and (polls is None or polls >= spec.min_polls)
            and (samples is None or samples >= spec.min_requests)
        )
        aggregates[metric] = {
            "runs": len(vals),
            "median": statistics.median(vals) if vals else None,
            "min": min(vals) if vals else None,
            "max": max(vals) if vals else None,
            "polls_usable": polls,
            "sample_count": samples,
            "dominant_quality": dominant,
            "adequate": adequate,
        }
    if not ok_runs:
        quality = "inadequate"
    elif len(ok_runs) < len(runs) or censored:
        quality = "mixed"
    else:
        quality = "adequate"
    spread = None
    tp = aggregates["throughput_req_s"]
    if tp["runs"] >= 2 and tp["median"]:
        spread = (tp["max"] - tp["min"]) / tp["median"]
        if spread > spec.repeat_spread_max:
            quality = "mixed"
    limitations: List[str] = []
    if not ok_runs:
        limitations.append("no successful run at this level")
    if censored:
        limitations.append("latency evidence censored by timeouts/incomplete requests")
    if quality != "adequate":
        limitations.append(f"evidence quality={quality}")
    return {"aggregate": aggregates, "evidence_quality": quality,
            "throughput_spread": spread, "censored": censored,
            "limitations": limitations}


# ------------------------------------------------------------- interpretation
def _series(levels: List[Dict[str, Any]], metric: str) -> List[Tuple[float, Optional[float]]]:
    out = []
    for lvl in levels:
        val = (lvl.get("aggregate", {}).get(metric) or {}).get("median")
        out.append((float(lvl["offered_load"]), _numeric(val)))
    return out


def interpret_sweep(
    spec: SweepSpec,
    levels: List[Dict[str, Any]],
    *,
    runtime_config_changed: Optional[bool] = None,
) -> Dict[str, Any]:
    """Deterministic, conservative reading of the observed load range.

    Separates (1) experiment completion, (2) measurement-evidence adequacy,
    (3) observed behaviour, and (4) whether saturation was actually
    demonstrated. Never claims a universal saturation knee and never recommends
    settings. A single-run-per-level sweep (``repeats < 2``) can never yield a
    non-INCONCLUSIVE verdict.
    """
    ordered = sorted(levels, key=lambda lvl: float(lvl["offered_load"]))
    offered = [float(lvl["offered_load"]) for lvl in ordered]
    tp = _series(ordered, "throughput_req_s")
    wait = _series(ordered, "requests_waiting")
    kv = _series(ordered, "kv_usage")
    timeouts = [sum(int(r.get("counts", {}).get("timeout", 0)) for r in lvl["runs"])
                for lvl in ordered]
    ok_levels = sum(1 for lvl in ordered if lvl["evidence_quality"] != "inadequate")

    inadequate = [lvl["offered_load"] for lvl in ordered
                  if lvl["evidence_quality"] != "adequate"]
    censored = [lvl["offered_load"] for lvl in ordered if lvl.get("censored")]
    low_sample: List[float] = []
    for lvl in ordered:
        agg = lvl.get("aggregate", {}).get("ttft_p95_s") or {}
        polls = agg.get("polls_usable")
        samples = agg.get("sample_count")
        if (polls is None or polls < spec.min_polls) or (
            samples is not None and samples < spec.min_requests
        ):
            low_sample.append(lvl["offered_load"])
    repeat_disagreement = any(
        (lvl.get("throughput_spread") or 0) > spec.repeat_spread_max for lvl in ordered
    )
    single_run = spec.repeats < 2

    gains: List[Optional[float]] = []
    for i in range(1, len(tp)):
        prev, cur = tp[i - 1][1], tp[i][1]
        gains.append(None if prev in (None, 0) or cur is None else (cur - prev) / prev)
    valid_gains = [g for g in gains if g is not None]
    non_monotonic = any(
        tp[i][1] is not None and tp[i - 1][1] is not None and tp[i][1] < tp[i - 1][1]
        for i in range(1, len(tp))
    )
    decreasing = any(g is not None and g < 0 for g in gains)

    # A plateau requires a NON-NEGATIVE marginal gain below the threshold.
    plateau_index = None
    for i, gain in enumerate(gains):
        if gain is not None and 0 <= gain < spec.plateau_gain_min and tp[i + 1][1] is not None:
            plateau_index = i + 1
    plateau = None
    if plateau_index is not None:
        plateau = {
            "observed_between_offered_load": [offered[plateau_index - 1], offered[plateau_index]],
            "throughput_req_s": [tp[plateau_index - 1][1], tp[plateau_index][1]],
            "marginal_relative_gain": gains[plateau_index - 1],
            "note": "observed flattening in this range on this hardware/model; not "
                    "a universal saturation knee",
        }

    wait_vals = [v for _, v in wait if v is not None]
    queue_growth = None
    if len(wait_vals) >= 2:
        queue_growth = {
            "requests_waiting_first": wait_vals[0],
            "requests_waiting_last": wait_vals[-1],
            "monotone_increasing": all(b >= a for a, b in zip(wait_vals, wait_vals[1:])),
        }
    kv_vals = [v for _, v in kv if v is not None]
    kv_pressure = None
    if kv_vals:
        kv_pressure = {
            "kv_usage_max": max(kv_vals),
            "threshold": spec.kv_pressure_threshold,
            "exceeded": max(kv_vals) >= spec.kv_pressure_threshold,
        }
    timeout_increase = {
        "timeouts_per_level": timeouts,
        "increased": len(timeouts) >= 2 and timeouts[-1] > timeouts[0],
    }

    blockers: List[str] = []
    if inadequate:
        blockers.append("inadequate measurement evidence at offered load(s): "
                        + ", ".join(str(x) for x in inadequate))
    if censored:
        blockers.append("latency evidence censored by timeouts/incomplete at offered "
                        "load(s): " + ", ".join(str(x) for x in censored))
    if low_sample:
        blockers.append("insufficient sample counts at offered load(s): "
                        + ", ".join(str(x) for x in low_sample))
    if repeat_disagreement:
        blockers.append("repeat runs disagree materially on throughput")
    if non_monotonic:
        blockers.append("throughput is non-monotonic across levels")
    if runtime_config_changed:
        blockers.append("runtime configuration changed during the sweep")
    if single_run:
        blockers.append("single run per level (repeats < 2): saturation cannot be "
                        "demonstrated from one noisy measurement")

    last_gain = valid_gains[-1] if valid_gains else None
    still_rising = last_gain is not None and last_gain >= spec.plateau_gain_min
    saturation_demonstrated = (
        not blockers and plateau is not None and queue_growth is not None
        and (queue_growth["monotone_increasing"] or timeout_increase["increased"]
             or (kv_pressure or {}).get("exceeded"))
    )
    unsaturated_in_range = (
        not blockers and plateau is None and still_rising and queue_growth is None
        and not timeout_increase["increased"] and not (kv_pressure or {}).get("exceeded")
    )
    if saturation_demonstrated:
        verdict = "SATURATION_OBSERVED"
    elif unsaturated_in_range:
        verdict = "UNSATURATED_IN_OBSERVED_RANGE"
    else:
        verdict = "INCONCLUSIVE"

    evidence_quality = (
        "inadequate" if not ok_levels else ("partly_inadequate" if inadequate else "adequate")
    )
    return {
        "observed_load_range": {
            "min": min(offered) if offered else None,
            "max": max(offered) if offered else None,
            "unit": "concurrency" if spec.load_model == "closed_loop"
            else "arrival_rate_req_s",
        },
        "experiment": {
            "completed": all(r.get("ok") for lvl in ordered for r in lvl["runs"]),
            "levels_ok": ok_levels, "levels_total": len(ordered), "repeats": spec.repeats,
        },
        "evidence": {
            "quality": evidence_quality,
            "inadequate_levels": inadequate,
            "censored_levels": censored,
            "low_sample_levels": low_sample,
            "repeat_disagreement": repeat_disagreement,
        },
        "observations": {
            "throughput_plateau": plateau,
            "throughput_non_monotonic": non_monotonic,
            "throughput_decreasing": decreasing,
            "queue_growth": queue_growth,
            "timeout_increase": timeout_increase,
            "kv_pressure": kv_pressure,
            "runtime_config_changed": runtime_config_changed,
        },
        "saturation": {
            "demonstrated": bool(saturation_demonstrated),
            "basis": [] if not saturation_demonstrated else [
                "observed throughput flattening with sustained/consequential queue, "
                "timeout or KV pressure under adequate, repeated evidence"
            ],
            "note": "Saturation is NOT demonstrated unless evidence is adequate and "
                    "repeated; a functional smoke test on small hardware cannot "
                    "establish production saturation limits.",
        },
        "verdict": verdict,
        "evidence_quality": evidence_quality,
        "caveats": blockers,
        "comparability_disclaimer": (
            "Results are specific to this engine version, model, hardware and runtime "
            "configuration. Saturation points are NOT directly comparable across "
            "different hardware, models or runtime settings."
        ),
    }


# ------------------------------------------------------------------ sweep run
def run_sweep(
    spec: SweepSpec,
    *,
    engine_url: str,
    telemetry_url: Optional[str] = None,
    control_url: Optional[str] = None,
    gpu_collector: Any = None,
    progress: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Run all levels/repeats and return the result.json document."""
    if spec.engine not in ("vllm", "sglang"):
        raise ValueError(f"unsupported engine: {spec.engine!r}")
    if spec.repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not spec.levels:
        raise ValueError("at least one load level is required")
    telemetry_url = telemetry_url or engine_url

    control_base = control_url.rstrip("/") if control_url else None
    config_readbacks: List[Dict[str, Any]] = []
    if control_base:
        initial = _http_json(f"{control_base}/runtime/config")
        if initial is not None:
            config_readbacks.append({"at": "start", "config": initial})

    result: Dict[str, Any] = {
        "schema": SWEEP_SCHEMA,
        "generated_at": _iso_now(),
        "engine": {
            "name": spec.engine,
            "version": _engine_version(spec.engine, engine_url),
            "base_url": engine_url,
            "telemetry_url": telemetry_url,
        },
        "environment": _host_environment(),
        "model": spec.model,
        "sweep": {
            "load_model": spec.load_model,
            "levels": list(spec.levels),
            "workload": {
                "prompt_tokens": spec.prompt_tokens,
                "max_new_tokens": spec.max_new_tokens,
                "seed": spec.seed,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
                "warmup_requests": spec.warmup_requests,
                "timeout_s": spec.timeout_s,
                "drain_budget_s": spec.drain_budget_s,
            },
            "measurement": {
                "window_s": spec.window_s,
                "poll_interval_s": spec.poll_interval_s,
                "min_polls": spec.min_polls,
                "min_requests": spec.min_requests,
                "stale_fraction_max": spec.stale_fraction_max,
            },
            "repeats": spec.repeats,
            "runtime_policy_mutation": "none",
            "runtime_policy_stability": "unverified",
        },
        "levels": [],
        "summary": {},
        "limitations": [],
    }

    for index, offered in enumerate(spec.levels):
        runs = []
        for run_index in range(spec.repeats):
            progress(f"[level {index + 1}/{len(spec.levels)} "
                     f"offered={offered} run {run_index + 1}/{spec.repeats}] measuring...")
            run = run_level_once(
                spec, offered, engine_url=engine_url, telemetry_url=telemetry_url,
                run_index=run_index, gpu_collector=gpu_collector,
            )
            runs.append(run)
            counts = run.get("counts", {})
            tp = _metric_value(run, "throughput_req_s")
            ttft = _metric_value(run, "ttft_p95_s")
            progress(
                f"    ok={run['ok']} completed={counts.get('completed')} "
                f"req/s={'-' if tp is None else round(tp, 2)} "
                f"ttft_p95={'-' if ttft is None else round(ttft, 3)} "
                f"timeout={counts.get('timeout')} error={counts.get('error')} "
                f"incomplete={counts.get('incomplete')}"
            )
        level = {
            "index": index,
            "offered_load": offered,
            "offered": {"concurrency": int(offered)} if spec.load_model == "closed_loop"
            else {"arrival_rate": float(offered)},
            "runs": runs,
        }
        level.update(_aggregate_level(runs, spec))
        result["levels"].append(level)
        if control_base:
            after = _http_json(f"{control_base}/runtime/config")
            if after is not None:
                config_readbacks.append({"at": f"after_level_{index}", "config": after})

    runtime_config_changed: Optional[bool] = None
    if len(config_readbacks) >= 2:
        def _state(cfg: Any) -> Any:
            cfg = cfg or {}
            return (cfg.get("desired_revision"), cfg.get("observed"), cfg.get("desired"))
        runtime_config_changed = _state(config_readbacks[0]["config"]) != _state(
            config_readbacks[-1]["config"]
        )
    result["runtime_config"] = {
        "source": control_base,
        "verified": runtime_config_changed is not None,
        "changed": runtime_config_changed,
        "readbacks": config_readbacks,
        "note": (
            "read-only /runtime/config readbacks at level boundaries; the sweep "
            "never mutates policy"
        ) if control_base else (
            "not read (no --control-url); the sweep never mutates policy, but "
            "runtime-configuration stability was NOT independently verified"
        ),
    }
    result["sweep"]["runtime_policy_stability"] = (
        "stable" if runtime_config_changed is False
        else ("changed" if runtime_config_changed else "unverified")
    )
    result["assumptions"] = {
        "runtime_policy_mutation": "none by the sweep",
        "runtime_policy_stability": result["sweep"]["runtime_policy_stability"],
        "sole_traffic": (
            "detection heuristic only (environment.sole_traffic_invariant per run); "
            "external traffic is not proven absent"
        ),
        "histogram_boundaries": (
            "native cumulative/rolling metrics are not fully isolated to a single "
            "load level; see per-run evidence_windows"
        ),
        "latency_distributions": (
            "only p50/p95/p99 quantiles are collected; raw histogram bucket "
            "distributions are not preserved"
        ),
    }
    result["summary"] = interpret_sweep(
        spec, result["levels"], runtime_config_changed=runtime_config_changed
    )
    if result["summary"]["verdict"] == "INCONCLUSIVE":
        result["limitations"].append(
            "the sweep verdict is INCONCLUSIVE: saturation is not demonstrated "
            "(see summary.caveats)"
        )
    return result


# ------------------------------------------------------------------------ CLI
def _parse_levels(raw: str) -> List[float]:
    out: List[float] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            out.append(float(token))
    if not out:
        raise argparse.ArgumentTypeError("levels must be a non-empty comma list")
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        "infengine-sweep",
        description="Continuous-batching load sweep with a local HTML report.",
    )
    ap.add_argument("--engine", choices=("vllm", "sglang"), required=True)
    ap.add_argument("--base-url", required=True,
                    help="inference endpoint (native /v1/completions + /metrics)")
    ap.add_argument("--telemetry-url", default=None,
                    help="telemetry base URL (defaults to --base-url)")
    ap.add_argument("--control-url", default=None,
                    help="optional control plane; read-only runtime config")
    ap.add_argument("--model", default="", help="model name (recorded)")
    ap.add_argument("--load-model", choices=("closed_loop", "open_loop"),
                    default="closed_loop")
    ap.add_argument("--levels", type=_parse_levels, default=[1.0, 2.0, 4.0, 8.0],
                    help="comma list of concurrency (closed) or arrival rate (open)")
    ap.add_argument("--prompt-tokens", default="128",
                    help="int or 'min:max'")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--requests", type=int, default=400, help="open_loop arrivals")
    ap.add_argument("--measurement", type=float, default=20.0, help="closed_loop seconds")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--window", type=float, default=10.0)
    ap.add_argument("--poll-interval", type=float, default=2.0)
    ap.add_argument("--gpu", action="store_true", help="collect nvidia-smi GPU diagnostics")
    ap.add_argument("--out", default="result.json")
    ap.add_argument("--html", default="report.html")
    ap.add_argument("--report-only", metavar="RESULT_JSON", default=None,
                    help="skip running a sweep; render HTML from an existing result")
    return ap


def _prompt_tokens(raw: str) -> Any:
    if ":" in raw:
        lo, hi = raw.split(":", 1)
        return (int(lo), int(hi))
    return int(raw)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    from .report import write_report

    if args.report_only:
        with open(args.report_only, "r", encoding="utf-8") as handle:
            result = json.load(handle)
        write_report(result, args.html)
        print(f"[report] wrote {args.html} from {args.report_only}")
        return 0

    spec = SweepSpec(
        engine=args.engine, model=args.model, load_model=args.load_model,
        levels=list(args.levels), prompt_tokens=_prompt_tokens(args.prompt_tokens),
        max_new_tokens=args.max_new_tokens, seed=args.seed,
        measurement_s=args.measurement, requests=args.requests,
        warmup_requests=args.warmup, repeats=args.repeats, timeout_s=args.timeout,
        window_s=args.window, poll_interval_s=args.poll_interval,
    )
    gpu = None
    if args.gpu:
        from .telemetry import GpuTelemetryCollector

        gpu = GpuTelemetryCollector(collect_processes=True)
    result = run_sweep(
        spec, engine_url=args.base_url, telemetry_url=args.telemetry_url,
        control_url=args.control_url, gpu_collector=gpu,
    )
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    write_report(result, args.html)
    summary = result["summary"]
    obs = summary.get("observations") or {}
    print("\n=== sweep summary ===")
    print(f"engine={result['engine']['name']} load_model={spec.load_model} "
          f"levels={len(result['levels'])} repeats={spec.repeats}")
    print(f"verdict: {summary['verdict']} "
          f"(evidence={(summary.get('evidence') or {}).get('quality')})")
    if obs.get("throughput_plateau"):
        p = obs["throughput_plateau"]
        print(f"observed flattening between offered load {p['observed_between_offered_load']} "
              f"(marginal gain {p['marginal_relative_gain']:+.1%})")
    if obs.get("kv_pressure"):
        print(f"kv_pressure exceeded: {obs['kv_pressure']['exceeded']} "
              f"(max {obs['kv_pressure']['kv_usage_max']})")
    for caveat in (summary.get("caveats") or [])[:4]:
        print(f"  caveat: {caveat}")
    print(f"result: {args.out}")
    print(f"report: {args.html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
