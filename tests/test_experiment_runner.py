"""M6B-2: controlled experiment runner tests.

Exercises the runner against the EXISTING service API (a real uvicorn-served
``create_app`` with a fake engine for the control plane + native
/v1/completions) and a mock out-of-band telemetry source (M5B exposition).
Covers: successful comparison, inadequate telemetry, failed convergence,
concurrent changes, rollback conflict (operator change preserved), timeouts /
incomplete requests (censored evidence), cleanup on exceptions, and the
threshold-based verdict procedure.
"""

import json
import socket
import sys
import threading
import time
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uvicorn  # noqa: E402
from starlette.testclient import TestClient  # noqa: E402

from infengine.control import ApplyResult, InfengineAdapter, TransitionClass  # noqa: E402
from infengine.experiment import (  # noqa: E402
    ClosedLoopWorkloadDriver,
    ControlPlaneClient,
    ControlPlaneError,
    ExperimentSpec,
    MeasurementSpec,
    OpenLoopWorkloadDriver,
    RunnerConfig,
    SuccessCriteria,
    WorkloadSpec,
    WorkloadTimeout,
    _aggregate,
    _build_prompt_pool,
    _compare,
    run_experiment,
)
from infengine.serve import create_app  # noqa: E402
from infengine.telemetry import GpuTelemetryCollector, SglangTelemetryCollector  # noqa: E402

from telemetry_helpers import (  # noqa: E402
    FakeGpuSource,
    SGLANG_LABELS,
    compute_apps_csv,
    gpu_csv,
    histogram_text,
    sglang_loads_exposition,
    telemetry_server,
)
from test_control import make_engine  # noqa: E402


class _FakeTok:
    def encode(self, prompt, add_special_tokens=False):
        n = 1 + (len(prompt) % 6)
        return [(i * 7 + 3) % 118 + 5 for i in range(n)]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(t) for t in ids)


def _make_engine(max_batch=2):
    engine = make_engine(max_batch=max_batch, chunk_size=8, num_blocks=32)
    engine.tokenizer = _FakeTok()
    return engine


def _make_app(engine, state_store=None):
    return create_app(get_engine=lambda: engine, state_store=state_store, drain_timeout=5.0)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def _served(app):
    port = _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not getattr(server, "started", False):
        if time.time() > deadline:
            raise TimeoutError("uvicorn did not start")
        time.sleep(0.01)
    try:
        yield f"http://127.0.0.1:{port}"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def _preseed(control_url, desired):
    c = ControlPlaneClient(control_url, timeout=5.0)
    status, payload = c.patch_config(desired)
    assert status == 200, payload
    return payload.get("desired_revision")


def _spec(**overrides):
    spec = ExperimentSpec(
        experiment_id="test-exp",
        engine="sglang",
        model="fake",
        baseline={"scheduler.max_batch": 2},
        candidate={"scheduler.max_batch": 4},
        workload=WorkloadSpec(
            clients=2, measurement_s=3.0, max_requests=500,
            prompt_tokens=8, max_new_tokens=4, seed=1,
            timeout_s=5.0, drain_budget_s=1.0,
        ),
        warmup_requests=8,
        measurement=MeasurementSpec(
            window_s=1.0, poll_interval_s=0.2, min_polls=2, min_requests=4,
            decision_metrics=["e2e_p50_s", "ttft_p95_s", "kv_usage",
                              "throughput_req_s", "throughput"],
        ),
        criteria=SuccessCriteria(primary="e2e_p50_s"),
    )
    for k, v in overrides.items():
        setattr(spec, k, v)
    return spec


def _latency_exposition(n, scale=1.0):
    ttft = histogram_text(
        "sglang:time_to_first_token_seconds",
        [(0.01, 0), (0.05 * scale, n), (0.1 * scale, n), ("+Inf", n)],
        0.05 * scale * n, n, labels=SGLANG_LABELS, add_help=True,
    )
    e2e = histogram_text(
        "sglang:e2e_request_latency_seconds",
        [(0.01, 0), (0.1 * scale, n), (0.2 * scale, n), ("+Inf", n)],
        0.1 * scale * n, n, labels=SGLANG_LABELS, add_help=True,
    )
    queue = histogram_text(
        "sglang:queue_time_seconds",
        [(0.001, 0), (0.005 * scale, n), (0.01 * scale, n), ("+Inf", n)],
        0.002 * scale * n, n, labels=SGLANG_LABELS, add_help=True,
    )
    return "\n".join([ttft, e2e, queue]) + "\n"


def _observed_value(control_url, intent_key):
    with urllib.request.urlopen(f"{control_url}/runtime/config", timeout=2) as resp:
        return json.loads(resp.read())["observed"].get(intent_key)


def _phase_aware_routes(control_url, intent_key, values):
    step = {"metrics": 0}

    def phase():
        obs = _observed_value(control_url, intent_key)
        return "candidate" if obs == values["candidate"]["value"] else "baseline"

    def loads():
        v = values[phase()]
        return (200, sglang_loads_exposition(
            running=v.get("running", 0), waiting=v.get("waiting", 0),
            gen_throughput=v["gen_throughput"],
            token_usage=v.get("token_usage", 0.4)))

    def metrics():
        v = values[phase()]
        step["metrics"] += 1
        return (200, _latency_exposition(step["metrics"], v.get("latency_scale", 1.0)))

    return {"/v1/loads?format=prometheus": loads, "/metrics": metrics}


def _static_routes(gen_throughput=10.0):
    step = {"metrics": 0}

    def metrics():
        step["metrics"] += 1
        return (200, _latency_exposition(step["metrics"]))

    return {
        "/v1/loads?format=prometheus": lambda: (
            200, sglang_loads_exposition(running=0, waiting=0,
                                         gen_throughput=gen_throughput, token_usage=0.4)),
        "/metrics": metrics,
    }


def _alternate_aware_routes(control_url, intent_key, values, preseed_rev):
    """Phase-aware telemetry for alternate cycles. The runner applies exactly
    one policy PATCH per phase (each bumps desired_revision by 1), so baseline
    PATCHes land at revision offsets 1,4 (leading baseline) and 3,6 (trailing
    baseline) from the pre-seed revision. Observed-value routing alone cannot
    separate a trailing baseline from the next cycle's leading baseline (both
    keep the same observed value with no transition between them)."""
    state = {"metrics": 0}

    def phase():
        cfg = _observed_cfg(control_url)
        obs = cfg["observed"].get(intent_key)
        if obs == values["candidate"]["value"]:
            return "candidate"
        offset = int(cfg.get("desired_revision", preseed_rev) or preseed_rev) - preseed_rev
        return "trail" if offset % 3 == 0 else "baseline"

    def loads():
        v = values[phase()]
        return (200, sglang_loads_exposition(
            running=v.get("running", 0), waiting=0,
            gen_throughput=v["gen_throughput"],
            token_usage=v.get("token_usage", 0.4)))

    def metrics():
        v = values[phase()]
        state["metrics"] += 1
        return (200, _latency_exposition(state["metrics"], v.get("latency_scale", 1.0)))

    return {"/v1/loads?format=prometheus": loads, "/metrics": metrics}


def _phase_send(control_url, intent_key, candidate_value, mode):
    def send(prompt):
        ph = "candidate" if _observed_value(control_url, intent_key) == candidate_value else "baseline"
        m = mode(ph)
        if m == "timeout":
            raise WorkloadTimeout("simulated client timeout")
        if m == "hang":
            time.sleep(10)

    return send


def _runner_cfg(control_url, tel_url, **kw):
    return RunnerConfig(
        control_url=control_url, engine_url=control_url,
        telemetry_url=tel_url, **kw,
    )


# -------------------------------------------------------------- successful A/B
def test_successful_comparison_improved():
    engine = _make_engine(max_batch=2)
    spec = _spec()
    # Explicitly select the native token gauge for the throughput clause so the
    # mocked phase-aware gauge (10 -> 13) drives the verdict deterministically.
    # The default throughput_metric is the runner-derived throughput_req_s.
    spec.criteria.throughput_metric = "throughput"
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        values = {
            "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0,
                         "token_usage": 0.4, "running": 0},
            "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8,
                          "token_usage": 0.5, "running": 0},
        }
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert result.verdict == "IMPROVED", result.to_dict()
    assert result.phases["baseline"]["convergence"]["status"] == "CONVERGED"
    assert result.phases["candidate"]["convergence"]["status"] == "CONVERGED"
    # native token-throughput gauge (engine-specific) is reported as preserved
    # diagnostic telemetry
    assert abs(result.phases["baseline"]["metrics"]["throughput"]["value"] - 10.0) < 1e-6
    assert abs(result.phases["candidate"]["metrics"]["throughput"]["value"] - 13.0) < 1e-6
    # the runner-derived benchmark is measured and reported too
    assert result.phases["baseline"]["metrics"]["throughput_req_s"]["value"] is not None
    assert result.phases["candidate"]["metrics"]["throughput_req_s"]["value"] is not None
    assert result.comparison["deltas"]["throughput"]["relative"] > 0.10
    assert result.comparison["deltas"]["e2e_p50_s"]["relative"] < 0.0  # latency improved
    assert result.comparison["adequacy"]["e2e_p50_s"]["baseline"] == "adequate"
    assert result.comparison["adequacy"]["throughput"]["candidate"] == "adequate"
    assert result.verdict_basis["evidence"]
    # rollback restored the pre-experiment policy
    assert result.rollback["status"] == "CONVERGED"
    assert engine.scheduler.max_batch == 2
    assert result.desired_revision["after_rollback"] > result.desired_revision["after_candidate"]


# -------------------------------------------------- inadequate telemetry
def test_inadequate_telemetry_inconclusive():
    engine = _make_engine(max_batch=2)
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        # /metrics -> 404: SGLang latency is disabled -> primary inadequate
        routes = {"/v1/loads?format=prometheus": lambda: (
            200, sglang_loads_exposition(running=0, waiting=0, gen_throughput=10.0))}
        with telemetry_server(routes) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(_spec(), cfg)

    assert result.verdict == "INCONCLUSIVE", result.to_dict()
    assert result.comparison["adequacy"]["e2e_p50_s"]["baseline"] == "unavailable"
    assert result.comparison["adequacy"]["throughput_req_s"]["baseline"] == "adequate"
    assert result.rollback["status"] == "CONVERGED"
    assert engine.scheduler.max_batch == 2
    assert result.verdict_basis["evidence"]


# ------------------------------------------------------ failed convergence
def test_failed_convergence_candidate_patch_fails(monkeypatch):
    engine = _make_engine(max_batch=2)
    app = _make_app(engine)
    with _served(app) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})

        def failing_apply(self, name, value):
            return ApplyResult(False, TransitionClass.HOT, reason="simulated apply failure")

        monkeypatch.setattr(InfengineAdapter, "apply", failing_apply)
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(_spec(), cfg)

    assert result.verdict == "FAILED", result.to_dict()
    assert "candidate phase failed" in result.failure_reason
    assert result.phases["candidate"]["ok"] is False
    # engine untouched (candidate apply failed) and rollback restored
    assert engine.scheduler.max_batch == 2
    assert result.rollback["status"] in ("CONVERGED",)


# ------------------------------------------------- concurrent change + rollback conflict
def test_concurrent_change_rollback_conflict_preserves_operator_change():
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.workload.measurement_s = 1.5  # keep the test fast
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            box = {}

            def run():
                box["result"] = run_experiment(spec, cfg)

            t = threading.Thread(target=run)
            t.start()
            # Wait until the candidate policy is applied, then PATCH a DIFFERENT
            # key concurrently (operator change) -> bumps the revision mid-phase.
            ext = ControlPlaneClient(control_url, timeout=5.0)
            deadline = time.time() + 30
            while True:
                if _observed_value(control_url, "scheduler.max_batch") == 4:
                    break
                if time.time() > deadline:
                    raise TimeoutError("candidate never applied")
                time.sleep(0.01)
            status, payload = ext.patch_config({"scheduler.chunk_size": 16})
            assert status == 200, payload
            t.join(timeout=60)
            assert not t.is_alive()
            result = box["result"]
            final_cfg = _observed_cfg(control_url)

    assert result.verdict == "FAILED", result.to_dict()
    assert "concurrent change detected" in result.failure_reason
    assert result.phases["candidate"]["environment"]["concurrent_change_detected"] is True
    # rollback conflicted: the runner did NOT overwrite the operator's change
    assert result.rollback["status"] == "CONFLICT"
    assert final_cfg["desired"].get("scheduler.chunk_size") == 16  # preserved
    assert final_cfg["desired"].get("scheduler.max_batch") == 4  # left as-is, not rolled back


def _observed_cfg(control_url):
    with urllib.request.urlopen(f"{control_url}/runtime/config", timeout=2) as resp:
        return json.loads(resp.read())


# -------------------------------------------------------------------- timeouts
def test_client_timeouts_censor_latency_evidence():
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.warmup_requests = 0
    spec.workload.measurement_s = 1.5
    spec.workload.timeout_s = 1.0
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        send = _phase_send(control_url, "scheduler.max_batch", 4,
                           lambda ph: "ok" if ph == "baseline" else "timeout")
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel, send_request=send)
            result = run_experiment(spec, cfg)

    assert result.verdict == "INCONCLUSIVE", result.to_dict()
    assert result.phases["candidate"]["counts"]["timeout"] > 0
    assert result.comparison["adequacy"]["e2e_p50_s"]["candidate"] == "censored_timeouts"
    assert result.rollback["status"] == "CONVERGED"


# ----------------------------------------------------------------- incomplete
def test_incomplete_requests_censor_latency_evidence():
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.warmup_requests = 0
    spec.workload.measurement_s = 1.5
    spec.workload.timeout_s = 1.0
    spec.workload.drain_budget_s = 0.2
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        send = _phase_send(control_url, "scheduler.max_batch", 4,
                           lambda ph: "ok" if ph == "baseline" else "hang")
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel, send_request=send)
            result = run_experiment(spec, cfg)

    assert result.verdict == "INCONCLUSIVE", result.to_dict()
    assert result.phases["candidate"]["counts"]["incomplete"] > 0
    assert result.comparison["adequacy"]["e2e_p50_s"]["candidate"] == "censored_incomplete"
    assert result.rollback["status"] == "CONVERGED"


# ---------------------------------------------------------- cleanup on exception
def test_cleanup_on_exception_attempts_rollback(tmp_path, monkeypatch):
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.workload.measurement_s = 1.5
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        calls = {"n": 0}
        orig = ControlPlaneClient.get_config

        def flaky(self):
            calls["n"] += 1
            if calls["n"] >= 3:  # blows up at the candidate end-of-phase check
                raise ControlPlaneError("control plane gone")
            return orig(self)

        monkeypatch.setattr(ControlPlaneClient, "get_config", flaky)
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert result.verdict == "FAILED"
    assert "INTERNAL_ERROR" in result.failure_reason
    assert result.rollback  # best-effort rollback was attempted and reported


# ------------------------------------------------------------ preflight rejects
def test_preflight_rejects_bad_specs():
    engine = _make_engine(max_batch=2)
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            # measurement window too short for the adequacy constraint
            bad = _spec()
            bad.workload.measurement_s = 0.5
            r = run_experiment(bad, cfg)
            assert r.verdict == "FAILED" and "preflight failed" in r.failure_reason
            # baseline == candidate is a spec error
            same = _spec(candidate={"scheduler.max_batch": 2})
            r = run_experiment(same, cfg)
            assert r.verdict == "FAILED" and "preflight failed" in r.failure_reason
            # criteria throughput clause references a metric not measured
            nomet = _spec()
            nomet.measurement.decision_metrics = ["e2e_p50_s"]
            r = run_experiment(nomet, cfg)
            assert r.verdict == "FAILED"
            assert "throughput clause references" in r.failure_reason
    assert engine.scheduler.max_batch == 2  # zero mutation from failed experiments


def test_preflight_rejects_unrestorable_rollback_target():
    """M6B-2.1: an experiment is rejected BEFORE mutation when its rollback
    target cannot be restored exactly through the merge-only PATCH semantics —
    here the intent key is absent from the (empty) pre-experiment DesiredState,
    so the experiment could never unset it."""
    engine = _make_engine(max_batch=2)
    with _served(_make_app(engine)) as control_url:
        # NO pre-seed: pre_experiment DesiredState is empty
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            r = run_experiment(_spec(), cfg)
            assert r.verdict == "FAILED", r.to_dict()
            assert "rollback target not exactly restorable" in r.failure_reason
    assert engine.scheduler.max_batch == 2  # zero mutation (preflight gate)


# ------------------------------------------ runtime_concurrency alias regression
def test_runner_runtime_concurrency_alias_regression():
    """The operator alias runtime_concurrency -> internal runtime_max_running_seqs
    through the real ControlContract. This pins the M6B-2 live bug: the runner
    must key the observed check on the INTERNAL key while reading the desired
    value from the OPERATOR key, and must never raise a KeyError."""
    from infengine.serve import create_app as _create_app  # noqa: F401
    from test_sglang_adapter import FakeSglangRuntimeClient

    app = create_app(runtime="sglang",
                     runtime_client=FakeSglangRuntimeClient(provisioned_seqs=8,
                                                            provisioned_tokens=16384))
    spec = _spec(engine="sglang",
                 baseline={"runtime_concurrency": 2},
                 candidate={"runtime_concurrency": 4})
    spec.warmup_requests = 0  # the sglang control-plane backend returns 501 for /v1/completions
    spec.workload.measurement_s = 1.5
    with _served(app) as control_url:
        _preseed(control_url, {"runtime_concurrency": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert "INTERNAL_ERROR" not in (result.failure_reason or ""), result.to_dict()
    assert result.phases["baseline"]["convergence"]["status"] == "CONVERGED"
    # the observed check used the INTERNAL key for convergence
    assert result.phases["baseline"]["convergence"]["observed"]["runtime_max_running_seqs"] == 2
    assert result.phases["candidate"]["convergence"]["observed"]["runtime_max_running_seqs"] == 4
    # sglang backend /v1/completions is 501 -> measurement requests error ->
    # latency evidence censored -> INCONCLUSIVE (not a runner crash)
    assert result.verdict == "INCONCLUSIVE"
    assert result.rollback["status"] == "CONVERGED"


# ------------------------------------------------------------- verdict procedure
def _compare_spec():
    spec = _spec()
    spec.measurement.decision_metrics = ["e2e_p50_s", "throughput_req_s", "throughput"]
    return spec


def _phase_payload(e2e, thr_req, completed=100, timeout=0, incomplete=0, native=None):
    payload = {
        "metrics": {
            "e2e_p50_s": {"value": e2e, "dominant_quality": "fresh",
                          "polls_usable": 10, "stale_fraction": 0.0},
            "throughput_req_s": {"value": thr_req, "dominant_quality": "fresh",
                                 "polls_usable": 1, "stale_fraction": 0.0},
        },
        "counts": {"completed": completed, "timeout": timeout, "incomplete": incomplete},
    }
    if native is not None:
        payload["metrics"]["throughput"] = {
            "value": native, "dominant_quality": "fresh",
            "polls_usable": 10, "stale_fraction": 0.0,
        }
    return payload


def test_compare_verdicts_threshold_based():
    spec = _compare_spec()
    # IMPROVED: latency better, benchmark requests/sec >= +10%
    verdict = _compare(spec, _phase_payload(0.10, 10.0), _phase_payload(0.09, 13.0))[1]
    assert verdict == "IMPROVED"
    # DEGRADED: latency worse by >= +10%
    verdict = _compare(spec, _phase_payload(0.10, 10.0), _phase_payload(0.12, 12.0))[1]
    assert verdict == "DEGRADED"
    # NO_MATERIAL_IMPROVEMENT: inside the threshold band (no equivalence claim)
    verdict = _compare(spec, _phase_payload(0.10, 10.0), _phase_payload(0.107, 12.0))[1]
    assert verdict == "NO_MATERIAL_IMPROVEMENT"
    # INCONCLUSIVE: censored evidence in the candidate phase
    verdict = _compare(spec, _phase_payload(0.10, 10.0),
                       _phase_payload(0.09, 13.0, timeout=1))[1]
    assert verdict == "INCONCLUSIVE"
    # INCONCLUSIVE: undefined delta (zero baseline benchmark throughput)
    verdict = _compare(spec, _phase_payload(0.10, 0.0), _phase_payload(0.09, 13.0))[1]
    assert verdict == "INCONCLUSIVE"
    # INCONCLUSIVE: primary disabled in baseline (unavailable)
    bp = _phase_payload(0.10, 10.0)
    bp["metrics"]["e2e_p50_s"] = {"value": None, "dominant_quality": "disabled",
                                  "polls_usable": 0, "stale_fraction": 1.0}
    verdict = _compare(spec, bp, _phase_payload(0.09, 13.0))[1]
    assert verdict == "INCONCLUSIVE"


def test_compare_uses_selected_throughput_metric_not_misled_by_native_gauge():
    """The verdict's throughput clause uses the EXPLICITLY selected metric. With
    the default (runner-derived throughput_req_s), a native token gauge that
    reads 0.0 (the live SGLang zero-gauge scenario) is reported but never gates
    the verdict."""
    spec = _compare_spec()  # throughput_metric defaults to throughput_req_s
    bp = _phase_payload(0.10, 5.0, native=8.0)
    cp = _phase_payload(0.09, 6.5, native=0.0)  # native gauge zero, live anomaly
    comparison, verdict, basis = _compare(spec, bp, cp)
    assert verdict == "IMPROVED", basis
    assert "throughput" not in comparison["adequacy"]  # native gauge NOT gating
    assert comparison["adequacy"]["throughput_req_s"]["candidate"] == "adequate"
    assert cp["metrics"]["throughput"]["value"] == 0.0  # still reported as-is


def test_compare_explicit_native_gauge_selection_is_operator_choice():
    """Explicitly selecting the native gauge for the throughput clause makes it
    gating (the operator's choice) — then a 0.0 gauge drives the verdict."""
    spec = _compare_spec()
    spec.criteria.throughput_metric = "throughput"
    bp = _phase_payload(0.10, 5.0, native=8.0)
    cp = _phase_payload(0.09, 6.5, native=0.0)
    comparison, verdict, _ = _compare(spec, bp, cp)
    assert "throughput" in comparison["adequacy"]
    assert "throughput_req_s" not in comparison["adequacy"]
    assert verdict == "DEGRADED"  # gauge -100% -> degradation criterion


# --------------------------------------------------- zero-gauge runner scenario
def test_zero_native_gauge_not_misleading_end_to_end():
    """Live SGLang finding replayed: the native gen_throughput reads 0.0 in the
    candidate window while requests complete. The verdict must be based on the
    runner-derived benchmark, not the gauge."""
    engine = _make_engine(max_batch=2)
    spec = _spec()  # default throughput_metric = throughput_req_s
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0,
                     "token_usage": 0.4, "running": 0},
        "candidate": {"value": 4, "gen_throughput": 0.0, "latency_scale": 0.9,
                      "token_usage": 0.5, "running": 0},
    }
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert result.phases["candidate"]["metrics"]["throughput"]["value"] == 0.0  # reported
    assert result.phases["baseline"]["metrics"]["throughput"]["value"] > 0.0
    assert result.phases["candidate"]["metrics"]["throughput_req_s"]["value"] is not None
    assert "throughput" not in result.comparison["adequacy"]  # gauge never gates
    assert result.comparison["adequacy"]["throughput_req_s"]["candidate"] == "adequate"
    # a real benchmark-based verdict (not INCONCLUSIVE driven by the gauge)
    assert result.verdict in ("IMPROVED", "DEGRADED", "NO_MATERIAL_IMPROVEMENT")
    assert result.rollback["status"] == "CONVERGED"


# ----------------------------------------------------------- workload driver
def test_driver_post_measurement_drain_not_throughput():
    spec = WorkloadSpec(clients=1, measurement_s=10.0, max_requests=2,
                        prompt_tokens=8, max_new_tokens=4, seed=1,
                        timeout_s=5.0, drain_budget_s=2.0)
    pool = _build_prompt_pool(8, 2, 1)

    def slow_send(prompt):
        time.sleep(0.4)  # completes AFTER the phase-end boundary

    driver = ClosedLoopWorkloadDriver(spec, slow_send, pool)
    driver.start_measurement()
    while driver.in_flight() == 0:
        time.sleep(0.01)
    # phase end NOW: the in-flight request completes at ~0.4s > t1
    driver.stop_measurement(t1=time.monotonic(), drain_budget_s=2.0)
    counts = driver.counts()
    assert counts["started"] == 1
    assert counts["completed"] == 0  # NOT measurement-period throughput
    assert counts["drained"] == 1  # post-measurement drain completion
    assert counts["incomplete"] == 0
    assert counts["timeout"] == 0
    assert driver.benchmark()["value"] is None  # no measurement completions


def test_driver_benchmark_uses_measurement_completions_only():
    """M6B-2.1: the runner-derived benchmark is requests completed in the
    measurement interval / the ACTUAL measurement span. Post-measurement drain
    completions never inflate it."""
    spec = WorkloadSpec(clients=1, timeout_s=5.0, drain_budget_s=1.0, max_requests=3)
    pool = _build_prompt_pool(8, 3, 1)
    sent = {"n": 0}

    def send(prompt):
        sent["n"] += 1
        if sent["n"] == 1:
            time.sleep(0.05)  # measurement-period completion
        else:
            time.sleep(0.5)  # completes after t1 -> drained

    driver = ClosedLoopWorkloadDriver(spec, send, pool)
    driver.start_measurement()
    deadline = time.time() + 5
    while not (driver.counts()["completed"] >= 1 and driver.in_flight() >= 1):
        assert time.time() < deadline, "first completion never happened"
        time.sleep(0.005)
    # first request completed (measurement); the second is now in flight and
    # will complete after t1 -> drained, never counted as throughput
    driver.stop_measurement(t1=time.monotonic() + 0.05, drain_budget_s=1.0)
    counts = driver.counts()
    assert counts["completed"] == 1
    assert counts["drained"] == 1
    b = driver.benchmark()
    assert b["completed"] == 1  # drain completion excluded
    assert b["value"] is not None and b["value"] > 0
    assert abs(b["value"] - 1.0 / b["duration_s"]) < 1e-9


def test_driver_timeout_and_incomplete_counts():
    pool = _build_prompt_pool(8, 100, 1)

    def timeout_send(prompt):
        raise WorkloadTimeout("simulated timeout")

    driver = ClosedLoopWorkloadDriver(WorkloadSpec(clients=2, timeout_s=1.0,
                                                   drain_budget_s=0.2),
                                      timeout_send, pool)
    driver.start_measurement()
    time.sleep(0.3)
    driver.stop_measurement(t1=time.monotonic(), drain_budget_s=0.2)
    counts = driver.counts()
    assert counts["timeout"] > 0
    assert counts["completed"] == 0
    # no measurement completions -> no benchmark value
    assert driver.benchmark()["value"] is None

    # incomplete: requests still in flight after the drain budget
    def hang_send(prompt):
        time.sleep(10)

    pool2 = _build_prompt_pool(8, 100, 1)
    driver2 = ClosedLoopWorkloadDriver(WorkloadSpec(clients=2, timeout_s=1.0,
                                                    drain_budget_s=0.1),
                                       hang_send, pool2)
    driver2.start_measurement()
    while driver2.in_flight() < 2:
        time.sleep(0.01)
    driver2.stop_measurement(t1=time.monotonic(), drain_budget_s=0.1)
    counts2 = driver2.counts()
    assert counts2["incomplete"] == 2
    assert counts2["completed"] == 0


# --------------------------------------------------------- collector wiring
def test_runner_uses_phase_scoped_collectors(monkeypatch):
    engine = _make_engine(max_batch=2)
    made = {"n": 0}
    orig = SglangTelemetryCollector.__init__

    def counting_init(self, *a, **kw):
        made["n"] += 1
        return orig(self, *a, **kw)

    monkeypatch.setattr(SglangTelemetryCollector, "__init__", counting_init)
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            run_experiment(_spec(), cfg)
    # one collector per phase + the preflight probe + the idle check
    assert made["n"] >= 3


def test_runner_collector_wiring_default():
    # the default factory builds the engine-appropriate collector against the
    # telemetry URL (phase-scoped), not the engine URL.
    from infengine.experiment import _default_collector_factory
    spec = _spec()
    factory = _default_collector_factory(spec)
    coll = factory("http://127.0.0.1:1")
    assert isinstance(coll, SglangTelemetryCollector)
    assert coll.window_s == spec.measurement.window_s
    assert coll.poll_interval == spec.measurement.poll_interval_s


# ================================================== M6B-3 repeated experiments
def _repeat_spec(**overrides):
    """A repeated-experiment spec with a comfortable adequacy margin. The
    shortened 1.5 s window leaves only ~2-3 summary polls, which is flaky under
    sequential-test CPU load; 3.0 s (the M6B-2 stable default) with min_polls=3
    gives ~10 nominal summary polls."""
    spec = _spec(**overrides)
    spec.workload.measurement_s = 3.0
    spec.measurement.min_polls = 3
    return spec


def test_repeated_cycles_median_aggregation():
    engine = _make_engine(max_batch=2)
    spec = _repeat_spec()
    spec.repeats = 2
    spec.criteria.throughput_metric = "throughput"  # deterministic mock gauge
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0, "running": 0},
        "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8, "running": 0},
    }
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert len(result.cycles) == 2
    for c in result.cycles:
        assert c["failed"] is False
        assert c["verdict"] == "IMPROVED"
        assert abs(c["deltas"]["throughput"]["relative"] - 0.30) < 0.02
    # M6A §8.4: aggregate = median of per-cycle deltas with spread
    assert result.aggregate["verdict"] == "IMPROVED"
    assert abs(result.aggregate["deltas"]["throughput"]["median"] - 0.30) < 0.02
    assert len(result.aggregate["deltas"]["throughput"]["per_cycle"]) == 2
    assert result.aggregate["deltas"]["throughput"]["min"] <= result.aggregate["deltas"]["throughput"]["max"]
    assert result.aggregate["per_cycle_verdicts"] == ["IMPROVED", "IMPROVED"]
    assert result.verdict == "IMPROVED"
    assert result.rollback["status"] == "CONVERGED"
    assert engine.scheduler.max_batch == 2


def test_alternate_drift_checks_reported_separately():
    engine = _make_engine(max_batch=2)
    spec = _repeat_spec()
    spec.repeats = 2
    spec.alternate = True
    spec.criteria.throughput_metric = "throughput"
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0, "running": 0},
        "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8, "running": 0},
        "trail": {"value": 2, "gen_throughput": 8.0, "latency_scale": 1.0, "running": 0},
    }
    with _served(_make_app(engine)) as control_url:
        r0 = _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_alternate_aware_routes(control_url, "scheduler.max_batch", values, r0)) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert len(result.cycles) == 2
    for c in result.cycles:
        assert "baseline_trail" in c["phases"]
        assert c["drift"] is not None
        # drift (trailing 8 vs leading 10) = -20%, reported separately
        assert abs(c["drift"]["throughput"]["relative"] - (-0.20)) < 0.02
        # the cycle delta uses the LEADING baseline (10 -> 13), never the trail
        assert abs(c["deltas"]["throughput"]["relative"] - 0.30) < 0.02
        assert abs(c["deltas"]["throughput"]["baseline_value"] - 10.0) < 0.02
    assert result.aggregate["drift"]["throughput"]["median"] < -0.1
    assert result.aggregate["verdict"] == "IMPROVED"
    assert result.verdict == "IMPROVED"
    assert result.rollback["status"] == "CONVERGED"


def test_repeated_cycles_inadequate_not_silently_excluded():
    engine = _make_engine(max_batch=2)
    spec = _repeat_spec()
    spec.repeats = 2
    # latency disabled (metrics 404): every cycle inadequate -> strict aggregate
    routes = {"/v1/loads?format=prometheus": lambda: (
        200, sglang_loads_exposition(running=0, waiting=0, gen_throughput=10.0))}
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(routes) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert len(result.cycles) == 2  # both recorded, none silently dropped
    assert result.aggregate["verdict"] == "INCONCLUSIVE"
    assert result.aggregate["inadequate_cycles"] == [0, 1]
    assert result.verdict == "INCONCLUSIVE"
    assert result.rollback["status"] == "CONVERGED"


def test_repeated_cycles_rollback_conflict_preserves_operator_change():
    engine = _make_engine(max_batch=2)
    spec = _repeat_spec()
    spec.repeats = 2
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            box = {}

            def run():
                box["result"] = run_experiment(spec, cfg)

            t = threading.Thread(target=run)
            t.start()
            ext = ControlPlaneClient(control_url, timeout=5.0)
            # wait for the SECOND candidate application (cycle 1), then change a
            # DIFFERENT key concurrently -> revision bump mid-cycle-1.
            last_obs = {"v": None}
            candidates = {"n": 0}
            deadline = time.time() + 30
            while True:
                obs = _observed_value(control_url, "scheduler.max_batch")
                if last_obs["v"] == 2 and obs == 4:
                    candidates["n"] += 1
                last_obs["v"] = obs
                if candidates["n"] >= 2:
                    break
                if time.time() > deadline:
                    raise TimeoutError("second candidate never applied")
                time.sleep(0.01)
            status, payload = ext.patch_config({"scheduler.chunk_size": 16})
            assert status == 200, payload
            t.join(timeout=60)
            assert not t.is_alive()
            result = box["result"]
            final_cfg = _observed_cfg(control_url)

    assert result.cycles[0]["verdict"] != "FAILED"  # cycle 0 completed
    assert result.aggregate["verdict"] == "FAILED"
    assert result.aggregate["failed_cycles"] == [1]
    assert result.rollback["status"] == "CONFLICT"
    assert final_cfg["desired"].get("scheduler.chunk_size") == 16  # preserved
    assert final_cfg["desired"].get("scheduler.max_batch") == 4  # left as-is


def test_repeated_cycles_cleanup_on_exception(monkeypatch):
    engine = _make_engine(max_batch=2)
    spec = _repeat_spec()
    spec.repeats = 2
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        calls = {"n": 0}
        orig = ControlPlaneClient.get_config

        def flaky(self):
            calls["n"] += 1
            if calls["n"] >= 5:  # cycle 1 candidate end-of-phase check
                raise ControlPlaneError("control plane gone")
            return orig(self)

        monkeypatch.setattr(ControlPlaneClient, "get_config", flaky)
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert result.verdict == "FAILED"
    assert "INTERNAL_ERROR" in result.failure_reason
    assert result.rollback  # best-effort rollback attempted and reported


# ------------------------------------------------- M6B-3 aggregation (unit)
def _cycle(index, verdict="IMPROVED", failed=False, reason=None, deltas=None,
           drift=None, basis=None):
    return {
        "index": index, "verdict": verdict, "failed": failed,
        "failure_reason": reason, "basis": basis or [],
        "deltas": deltas or {}, "adequacy": {}, "drift": drift,
    }


def test_aggregate_median_of_cycle_deltas():
    spec = _spec()
    spec.measurement.decision_metrics = ["e2e_p50_s", "throughput_req_s"]
    cycles = [
        _cycle(0, deltas={"e2e_p50_s": {"relative": -0.20},
                          "throughput_req_s": {"relative": 0.30}}),
        _cycle(1, deltas={"e2e_p50_s": {"relative": -0.10},
                          "throughput_req_s": {"relative": 0.10}}),
    ]
    agg = _aggregate(spec, cycles)
    assert agg["verdict"] == "IMPROVED"
    assert abs(agg["deltas"]["e2e_p50_s"]["median"] - (-0.15)) < 1e-9
    assert abs(agg["deltas"]["throughput_req_s"]["median"] - 0.20) < 1e-9
    assert agg["deltas"]["throughput_req_s"]["per_cycle"] == [0.30, 0.10]
    assert agg["deltas"]["throughput_req_s"]["min"] == 0.10
    assert agg["deltas"]["throughput_req_s"]["max"] == 0.30


def test_aggregate_strict_inadequate_not_excluded():
    spec = _spec()
    spec.measurement.decision_metrics = ["e2e_p50_s", "throughput_req_s"]
    cycles = [
        _cycle(0, deltas={"e2e_p50_s": {"relative": -0.20},
                          "throughput_req_s": {"relative": 0.30}}),
        _cycle(1, verdict="INCONCLUSIVE", basis=["e2e_p50_s: baseline=unavailable"]),
    ]
    agg = _aggregate(spec, cycles)
    assert agg["verdict"] == "INCONCLUSIVE"
    assert agg["inadequate_cycles"] == [1]
    assert agg["per_cycle_verdicts"] == ["IMPROVED", "INCONCLUSIVE"]


def test_aggregate_strict_failed_not_excluded():
    spec = _spec()
    spec.measurement.decision_metrics = ["e2e_p50_s", "throughput_req_s"]
    cycles = [
        _cycle(0, deltas={"e2e_p50_s": {"relative": -0.20},
                          "throughput_req_s": {"relative": 0.30}}),
        _cycle(1, failed=True, reason="concurrent change detected"),
    ]
    agg = _aggregate(spec, cycles)
    assert agg["verdict"] == "FAILED"
    assert agg["failed_cycles"] == [1]
    assert "cycle 1 failed" in agg["basis"][0]


def test_aggregate_drift_separate_from_deltas():
    spec = _spec()
    spec.measurement.decision_metrics = ["e2e_p50_s", "throughput_req_s"]
    cycles = [
        _cycle(0, deltas={"e2e_p50_s": {"relative": -0.20},
                          "throughput_req_s": {"relative": 0.30}},
               drift={"e2e_p50_s": {"relative": 0.05},
                      "throughput_req_s": {"relative": -0.10}}),
        _cycle(1, deltas={"e2e_p50_s": {"relative": -0.10},
                          "throughput_req_s": {"relative": 0.10}},
               drift={"e2e_p50_s": {"relative": 0.07}}),
    ]
    agg = _aggregate(spec, cycles)
    assert agg["drift"]["throughput_req_s"]["median"] == -0.10
    assert agg["drift"]["e2e_p50_s"]["per_cycle"] == [0.05, 0.07]
    # aggregate deltas are NOT polluted by the drift values
    assert abs(agg["deltas"]["throughput_req_s"]["median"] - 0.20) < 1e-9


# ================================================== M6B-4 open-loop workload
def test_open_loop_driver_all_arrivals_complete_deterministic():
    pool = _build_prompt_pool(8, 10, 42)
    sent = []

    def send(prompt):
        sent.append(prompt)

    spec = WorkloadSpec(load_model="open_loop", arrival_rate=20.0, requests=6,
                        timeout_s=5.0, seed=42)
    driver = OpenLoopWorkloadDriver(spec, send, pool)
    driver.start_measurement()
    deadline = time.time() + 10
    while not driver.emission_done() and time.time() < deadline:
        time.sleep(0.05)
    driver.stop_measurement(driver.emission_end_ts(), drain_budget_s=2.0)
    counts = driver.counts()
    assert counts["started"] == 6
    assert counts["completed"] == 6
    assert counts["timeout"] == 0 and counts["error"] == 0 and counts["incomplete"] == 0
    # deterministic stream: prompts consumed in seeded pool order
    assert sent == pool[:6]
    assert driver.benchmark()["completed"] == 6


def test_open_loop_driver_timeout_censored():
    def send(prompt):
        raise WorkloadTimeout("simulated client timeout")

    spec = WorkloadSpec(load_model="open_loop", arrival_rate=20.0, requests=5,
                        timeout_s=1.0, seed=7)
    pool = _build_prompt_pool(8, 5, 7)
    driver = OpenLoopWorkloadDriver(spec, send, pool)
    driver.start_measurement()
    deadline = time.time() + 10
    while not driver.emission_done() and time.time() < deadline:
        time.sleep(0.05)
    driver.stop_measurement(driver.emission_end_ts(), drain_budget_s=1.0)
    counts = driver.counts()
    assert counts["started"] == 5
    assert counts["completed"] == 0
    assert counts["timeout"] == 5
    assert driver.benchmark()["value"] is None  # no measurement completions


def test_open_loop_driver_drain_not_throughput():
    def send(prompt):
        time.sleep(0.5)  # completes after t1 -> drained/incomplete

    spec = WorkloadSpec(load_model="open_loop", arrival_rate=50.0, requests=3,
                        timeout_s=5.0, drain_budget_s=0.1, seed=3)
    pool = _build_prompt_pool(8, 3, 3)
    driver = OpenLoopWorkloadDriver(spec, send, pool)
    driver.start_measurement()
    deadline = time.time() + 10
    while not driver.emission_done() and time.time() < deadline:
        time.sleep(0.05)
    t1 = driver.emission_end_ts()
    driver.stop_measurement(t1, drain_budget_s=0.1)
    counts = driver.counts()
    # all arrivals are in flight at t1 (0.5s sends) and not drained within budget
    assert counts["started"] == 3
    assert counts["completed"] == 0  # NOT measurement-period throughput
    assert counts["incomplete"] == 3
    assert driver.benchmark()["value"] is None


def test_open_loop_queue_growth_observed():
    def send(prompt):
        time.sleep(0.3)  # slow "engine": arrivals pile up in flight

    spec = WorkloadSpec(load_model="open_loop", arrival_rate=100.0, requests=30,
                        timeout_s=5.0, drain_budget_s=0.2, seed=1)
    pool = _build_prompt_pool(8, 30, 1)
    driver = OpenLoopWorkloadDriver(spec, send, pool)
    driver.start_measurement()
    deadline = time.time() + 10
    max_inflight = 0
    while not driver.emission_done() and time.time() < deadline:
        max_inflight = max(max_inflight, driver.in_flight())
        time.sleep(0.02)
    assert max_inflight > 5  # arrivals independent of completions -> pile up
    driver.stop_measurement(driver.emission_end_ts(), drain_budget_s=0.2)


def test_open_loop_experiment_run():
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.workload.load_model = "open_loop"
    spec.workload.arrival_rate = 4.0
    spec.workload.requests = 10  # 2.5 s nominal, > needed 1.2 s
    spec.measurement.decision_metrics = ["e2e_p50_s", "requests_waiting",
                                         "throughput_req_s", "throughput"]
    spec.criteria.throughput_metric = "throughput"  # deterministic mock gauge
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0,
                     "running": 0, "waiting": 2},
        "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8,
                      "running": 0, "waiting": 1},
    }
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(spec, cfg)

    assert result.verdict == "IMPROVED", result.to_dict()
    assert result.phases["baseline"]["metrics"]["offered_rate"]["value"] == 4.0
    assert result.phases["candidate"]["metrics"]["offered_rate"]["value"] == 4.0
    # queue depth is measured and reported (diagnostic decision metric)
    assert result.phases["baseline"]["metrics"]["requests_waiting"]["value"] == 2
    assert result.phases["candidate"]["metrics"]["requests_waiting"]["value"] == 1
    assert result.phases["baseline"]["metrics"]["throughput_req_s"]["value"] is not None
    assert result.rollback["status"] == "CONVERGED"
    assert engine.scheduler.max_batch == 2


def test_preflight_rejects_bad_open_loop():
    engine = _make_engine(max_batch=2)
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            bad = _spec()
            bad.workload.load_model = "open_loop"
            bad.workload.arrival_rate = 0.0
            r = run_experiment(bad, cfg)
            assert r.verdict == "FAILED" and "arrival_rate" in r.failure_reason
            short = _spec()
            short.workload.load_model = "open_loop"
            short.workload.arrival_rate = 50.0
            short.workload.requests = 10  # 0.2 s < needed 1.2 s
            r = run_experiment(short, cfg)
            assert r.verdict == "FAILED" and "open_loop duration" in r.failure_reason
    assert engine.scheduler.max_batch == 2  # zero mutation (preflight gate)


# ================================================== M5C GPU telemetry
def _fake_gpu_collector(gpu_out=None, gpu_rc=0, **kw):
    """GPU collector over a fake nvidia-smi source (no GPU required)."""
    source = FakeGpuSource(
        gpu_out=gpu_out if gpu_out is not None else gpu_csv(
            (45, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)),
        apps_out=compute_apps_csv((1234, "python.exe", 512)),
        gpu_rc=gpu_rc,
    )
    kw.setdefault("collect_processes", True)
    kw.setdefault("engine_pid", 1234)
    return GpuTelemetryCollector(runner=source, **kw)


def test_runner_with_gpu_collector_attaches_phase_gpu_summary():
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.criteria.throughput_metric = "throughput"  # deterministic mock gauge
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0, "running": 0},
        "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8, "running": 0},
    }
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel, gpu_collector=_fake_gpu_collector())
            result = run_experiment(spec, cfg)

    assert result.verdict == "IMPROVED", result.to_dict()
    for phase_name in ("baseline", "candidate"):
        gpu = result.phases[phase_name]["gpu"]
        assert gpu["configured"] is True
        assert gpu["available"] is True
        assert gpu["polls"] > 0
        # per-signal median/max over the phase summary window
        assert gpu["signals"]["gpu_util"]["median"] == 45.0
        assert gpu["signals"]["gpu_util"]["max"] == 45.0
        assert gpu["signals"]["sm_clock_mhz"]["median"] == 1800.0
        assert gpu["signals"]["temperature_c"]["max"] == 70.0
        assert gpu["signals"]["memory_total_mb"]["median"] == 4096.0
        # shared-GPU attribution of the engine PID
        assert gpu["engine_process"]["pid"] == 1234
        assert gpu["engine_process"]["used_memory_mb"]["median"] == 512.0
    # serialized through to_dict / cycles
    assert result.cycles[0]["phases"]["baseline"]["gpu"]["configured"] is True
    assert result.to_dict()["phases"]["baseline"]["gpu"]["available"] is True
    assert result.rollback["status"] == "CONVERGED"


def test_runner_without_gpu_collector_reports_not_configured():
    engine = _make_engine(max_batch=2)
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_static_routes()) as tel:
            cfg = _runner_cfg(control_url, tel)
            result = run_experiment(_spec(), cfg)

    assert result.phases["baseline"]["gpu"]["configured"] is False
    assert result.phases["baseline"]["gpu"]["available"] is False
    assert "not configured" in result.phases["baseline"]["gpu"]["unavailable_reason"]
    assert result.phases["baseline"]["gpu"]["signals"] == {}


def test_gpu_metrics_diagnostic_only_never_verdict_gating():
    """GPU evidence is attached to phases but NEVER participates in the
    comparison/adequacy/verdict (the payload used by _compare stays metrics +
    counts only)."""
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.criteria.throughput_metric = "throughput"
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0, "running": 0},
        "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8, "running": 0},
    }
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel, gpu_collector=_fake_gpu_collector())
            result = run_experiment(spec, cfg)

    assert result.verdict == "IMPROVED"  # verdict driven by engine metrics alone
    assert "gpu" not in result.comparison
    assert "gpu" not in result.comparison["adequacy"]
    # payload() (the compare input) never carries gpu evidence
    from infengine.experiment import PhaseResult

    bp = result.phases["baseline"]
    assert "gpu" not in PhaseResult("baseline", metrics=bp["metrics"],
                                    counts=bp["counts"]).payload()
    # GPU never merged into engine metrics
    assert "gpu_util" not in result.phases["baseline"]["metrics"]
    assert result.rollback["status"] == "CONVERGED"


def test_runner_gpu_graceful_absence():
    """No GPU / no nvidia-smi (exit 127): the run proceeds and records
    available=False with a reason; it never fails the phase."""
    engine = _make_engine(max_batch=2)
    spec = _spec()
    spec.criteria.throughput_metric = "throughput"
    values = {
        "baseline": {"value": 2, "gen_throughput": 10.0, "latency_scale": 1.0, "running": 0},
        "candidate": {"value": 4, "gen_throughput": 13.0, "latency_scale": 0.8, "running": 0},
    }
    with _served(_make_app(engine)) as control_url:
        _preseed(control_url, {"scheduler.max_batch": 2})
        with telemetry_server(_phase_aware_routes(control_url, "scheduler.max_batch", values)) as tel:
            cfg = _runner_cfg(control_url, tel,
                              gpu_collector=_fake_gpu_collector(gpu_rc=127))
            result = run_experiment(spec, cfg)

    assert result.verdict == "IMPROVED", result.to_dict()
    gpu = result.phases["baseline"]["gpu"]
    assert gpu["configured"] is True
    assert gpu["available"] is False
    assert "exited 127" in gpu["unavailable_reason"]
    assert gpu["signals"] == {}
    assert result.rollback["status"] == "CONVERGED"