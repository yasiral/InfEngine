"""M9C: continuous-batching load sweep tests.

Canned orchestration tests (deterministic) plus one bounded integration test
that runs a real level through the reused experiment machinery against the
built-in service and a mock telemetry source. No live engine/GPU required.
"""

import json
import sys
from pathlib import Path

from infengine import sweep as sweep_mod
from infengine.experiment import _build_prompt_pool
from infengine.sweep import (
    SweepSpec,
    _aggregate_level,
    _counter_deltas,
    _workload,
    interpret_sweep,
    main,
    run_level_once,
    run_sweep,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from telemetry_helpers import sglang_loads_exposition, telemetry_server  # noqa: E402
from test_experiment_runner import _make_app, _make_engine, _served  # noqa: E402


# --------------------------------------------------------------- synthetic
def _run(offered, *, ok=True, tp=10.0, ttft=0.1, queue=0.01, e2e=0.2, kv=0.2,
         wait=0.0, running=1, native_tp=600.0, prefill=0.05, decode=0.02,
         engine="vllm", counts=None, counter=None, failed_reason=None):
    stage = {
        "prefill_time": {"quantiles": {"p95": prefill}, "classification": "MEASURED",
                         "source": "x", "sample_count": 20, "quality": "fresh"},
        "decode_time": ({"quantiles": {"p95": decode}, "classification": "MEASURED",
                         "source": "x", "sample_count": 20, "quality": "fresh"}
                        if decode is not None else
                        {"quantiles": {}, "classification": "UNAVAILABLE",
                         "source": "", "sample_count": None, "quality": None}),
        "inter_token_latency": {"quantiles": {"p95": 0.02}, "classification": "MEASURED",
                                "source": "x", "sample_count": 20, "quality": "fresh"},
        "time_per_output_token": ({"quantiles": {"p95": 0.03}, "classification": "MEASURED",
                                   "source": "x", "sample_count": 20, "quality": "fresh"}
                                  if engine == "vllm" else
                                  {"quantiles": {}, "classification": "UNAVAILABLE",
                                   "source": "", "sample_count": None, "quality": None}),
    }
    metrics = {
        "throughput_req_s": {"value": tp, "dominant_quality": "fresh", "sample_count": 20,
                             "polls_usable": 6, "stale_fraction": 0.0},
        "throughput": {"value": native_tp, "dominant_quality": "fresh", "sample_count": 20,
                       "polls_usable": 6, "stale_fraction": 0.0},
        "output_tokens_per_s": {"value": native_tp, "dominant_quality": "fresh",
                                "sample_count": int(native_tp), "polls_usable": None,
                                "stale_fraction": 0.0, "window_s": 6.0},
        "ttft_p95_s": {"value": ttft, "dominant_quality": "fresh", "sample_count": 20,
                       "polls_usable": 6, "stale_fraction": 0.0},
        "queue_time_p95_s": {"value": queue, "dominant_quality": "fresh", "sample_count": 20,
                             "polls_usable": 6, "stale_fraction": 0.0},
        "e2e_p95_s": {"value": e2e, "dominant_quality": "fresh", "sample_count": 20,
                      "polls_usable": 6, "stale_fraction": 0.0},
        "kv_usage": {"value": kv, "dominant_quality": "fresh", "sample_count": 20,
                     "polls_usable": 6, "stale_fraction": 0.0},
        "requests_running": {"value": running, "dominant_quality": "fresh",
                             "sample_count": 20, "polls_usable": 6, "stale_fraction": 0.0},
        "requests_waiting": {"value": wait, "dominant_quality": "fresh",
                             "sample_count": 20, "polls_usable": 6, "stale_fraction": 0.0},
    }
    default_counts = {"started": 20, "completed": 18, "drained": 2, "timeout": 0,
                      "error": 0, "incomplete": 0, "in_flight": 0}
    return {
        "run_index": 0, "started_at": "t", "finished_at": "t", "offered_load": offered,
        "idle_before": True, "warmup_ok": True, "ok": ok,
        "failure_reason": failed_reason if not ok else None,
        "counts": counts or default_counts, "wall_s": 20.0, "environment": {},
        "metrics": metrics, "gpu": {"configured": False, "available": False},
        "counter_deltas": counter or {"num_preemptions": {"value": 0}},
        "execution_evidence": {"engine": engine, "request_stage": stage, "engine_level": {}},
        "snapshot": None,
    }


def _level(offered, runs, engine="vllm"):
    lvl = {"index": 0, "offered_load": offered, "runs": runs}
    lvl.update(_aggregate_level(runs, SweepSpec(engine=engine, model="m")))
    return lvl


def _synth_result(engine="vllm", levels=None, repeats=1):
    if levels is None:
        levels = [_level(x, [_run(x, engine=engine, decode=(None if engine == "sglang" else 0.02))])
                  for x in (1, 2, 4, 8)]
    result = {
        "schema": sweep_mod.SWEEP_SCHEMA, "generated_at": "t",
        "engine": {"name": engine, "version": {"version": "vX"}, "base_url": "x",
                   "telemetry_url": "x"},
        "environment": {}, "model": "m",
        "runtime_config": {"note": "fixed"},
        "sweep": {"load_model": "closed_loop", "levels": [lvl["offered_load"] for lvl in levels],
                  "workload": {"seed": 1},
                  "measurement": {"window_s": 2.0, "poll_interval_s": 1.0,
                                  "min_polls": 3, "min_requests": 8,
                                  "stale_fraction_max": 0.5},
                  "repeats": repeats,
                  "runtime_policy_mutation": "none",
                  "runtime_policy_stability": "unverified"},
        "levels": levels, "limitations": [],
    }
    result["summary"] = interpret_sweep(
        SweepSpec(engine=engine, model="m"), levels)
    return result


# --------------------------------------------------------------- orchestration
def test_run_sweep_orchestration(monkeypatch):
    calls = []

    def fake_level(spec, offered, *, engine_url, telemetry_url, run_index, gpu_collector=None):
        calls.append((offered, run_index))
        return _run(offered, tp=float(offered))

    monkeypatch.setattr(sweep_mod, "run_level_once", fake_level)
    spec = SweepSpec(engine="vllm", model="m", levels=[1, 2, 4], repeats=2)
    result = run_sweep(spec, engine_url="http://127.0.0.1:1",
                       telemetry_url="http://127.0.0.1:1", progress=lambda *_: None)
    assert calls == [(1.0, 0), (1.0, 1), (2.0, 0), (2.0, 1), (4.0, 0), (4.0, 1)]
    assert [lvl["offered_load"] for lvl in result["levels"]] == [1.0, 2.0, 4.0]
    assert all(len(lvl["runs"]) == 2 for lvl in result["levels"])
    assert result["schema"] == sweep_mod.SWEEP_SCHEMA
    assert result["sweep"]["runtime_policy_mutation"] == "none"
    assert result["sweep"]["runtime_policy_stability"] == "unverified"
    assert result["runtime_config"]["verified"] is False
    assert "runtime_policy_stability" in result["assumptions"]
    json.dumps(result)  # JSON-serializable
    assert result["summary"]["verdict"] in (
        "SATURATION_OBSERVED", "UNSATURATED_IN_OBSERVED_RANGE", "INCONCLUSIVE")


def test_aggregate_medians_and_quality():
    runs = [_run(2.0, tp=10.0), _run(2.0, tp=14.0), _run(2.0, tp=12.0)]
    agg = _aggregate_level(runs, SweepSpec(engine="vllm", model="m"))["aggregate"]
    assert agg["throughput_req_s"]["median"] == 12.0
    assert agg["throughput_req_s"]["min"] == 10.0 and agg["throughput_req_s"]["max"] == 14.0
    assert agg["throughput_req_s"]["runs"] == 3


def test_failed_run_makes_level_inadequate():
    runs = [_run(2.0, ok=False, failed_reason="foreign traffic")]
    summary = _aggregate_level(runs, SweepSpec(engine="vllm", model="m"))
    assert summary["evidence_quality"] == "inadequate"
    assert summary["aggregate"]["throughput_req_s"]["median"] is None


# ------------------------------------------------------------- repeatability
def test_seed_is_deterministic_and_recorded():
    spec = SweepSpec(engine="vllm", model="m", seed=7, prompt_tokens=16,
                     load_model="closed_loop", levels=[2])
    workload = _workload(spec, 2)
    assert workload.seed == 7
    assert (_build_prompt_pool(workload.prompt_tokens, 4, workload.seed)
            == _build_prompt_pool(workload.prompt_tokens, 4, workload.seed))
    assert (_build_prompt_pool(workload.prompt_tokens, 4, 1)
            != _build_prompt_pool(workload.prompt_tokens, 4, 2))


# --------------------------------------------------------- throughput / drain
def test_throughput_uses_runner_benchmark_and_drain_is_separate():
    # drained completions are recorded but never folded into the benchmark.
    run = _run(4.0, tp=9.5, counts={"started": 20, "completed": 15, "drained": 5,
                                     "timeout": 0, "error": 0, "incomplete": 0})
    assert run["counts"]["drained"] == 5
    assert run["metrics"]["throughput_req_s"]["value"] == 9.5
    agg = _aggregate_level([run], SweepSpec(engine="vllm", model="m"))
    assert agg["aggregate"]["throughput_req_s"]["median"] == 9.5


def test_counter_delta_is_reset_safe():
    class S:
        def __init__(self, i, d):
            self.instance_id = i
            self.engine_specific = d

    base = S("i1", {"num_preemptions": 100})
    assert _counter_deltas(base, S("i1", {"num_preemptions": 105}))["num_preemptions"]["value"] == 5
    assert _counter_deltas(base, S("i1", {"num_preemptions": 3}))["num_preemptions"]["value"] is None
    assert _counter_deltas(base, S("i2", {"num_preemptions": 105}))["num_preemptions"]["value"] is None
    assert _counter_deltas(base, S("i1", {}))["num_preemptions"]["value"] is None


# ----------------------------------------------------- execution-stage semantics
def test_vllm_evidence_has_decode_and_tpot():
    level = _level(2.0, [_run(2.0, engine="vllm", decode=0.03)])
    from infengine.report import build_panels

    panels = {p["id"]: p for p in build_panels(_synth_result("vllm", [level]))}
    names = {s["name"] for s in panels["execution_stage"]["series"]}
    assert "decode (measured)" in names
    assert "time per output token (measured)" in names


def test_sglang_decode_is_unavailable_not_fabricated():
    level = _level(2.0, [_run(2.0, engine="sglang", decode=None)], engine="sglang")
    from infengine.report import build_panels

    panels = {p["id"]: p for p in build_panels(_synth_result("sglang", [level]))}
    names = {s["name"] for s in panels["execution_stage"]["series"]}
    unavailable = {u["name"] for u in panels["execution_stage"]["unavailable"]}
    assert "decode (measured)" not in names
    assert "decode (measured)" in unavailable
    # and no decode point is silently plotted as zero
    for s in panels["execution_stage"]["series"]:
        assert all(pt[1] != 0 for pt in s["points"] if s["name"] == "decode (measured)")


# --------------------------------------------------------- missing / stale
def test_missing_latency_is_unavailable_not_zero():
    level = _level(2.0, [_run(2.0)])
    level["runs"][0]["metrics"]["ttft_p95_s"] = {"value": None, "dominant_quality": "disabled"}
    level.update(_aggregate_level(level["runs"], SweepSpec(engine="vllm", model="m")))
    from infengine.report import build_panels

    panels = {p["id"]: p for p in build_panels(_synth_result("vllm", [level]))}
    ttft = [s for s in panels["latency_tail"]["series"] if s["name"] == "TTFT p95"]
    if ttft:
        assert all(pt[1] is None for pt in ttft[0]["points"])
    assert any(u["name"] == "TTFT p95" for u in panels["latency_tail"]["unavailable"])


# ------------------------------------------------------------- interpretation
def test_inconclusive_when_evidence_inadequate():
    levels = [_level(1.0, [_run(1.0, ok=False, failed_reason="x")]),
              _level(2.0, [_run(2.0)])]
    summary = interpret_sweep(SweepSpec(engine="vllm", model="m"), levels)
    assert summary["verdict"] == "INCONCLUSIVE"
    assert any("inadequate" in c for c in summary["caveats"])


def test_plateau_reported_without_knee_claim():
    levels = [_level(1.0, [_run(1.0, tp=10)]), _level(2.0, [_run(2.0, tp=20)]),
              _level(4.0, [_run(4.0, tp=21)])]
    summary = interpret_sweep(SweepSpec(engine="vllm", model="m", plateau_gain_min=0.1), levels)
    plateau = summary["observations"]["throughput_plateau"]
    assert plateau is not None
    assert "not a universal saturation knee" in plateau["note"]


# ---------------------------------------------------------------------- CLI
def test_cli_report_only(tmp_path, capsys):
    result = _synth_result()
    src = tmp_path / "result.json"
    src.write_text(json.dumps(result), encoding="utf-8")
    out = tmp_path / "report.html"
    rc = main(["--engine", "vllm", "--base-url", "http://x",
               "--report-only", str(src), "--html", str(out)])
    assert rc == 0
    assert out.exists() and "InfEngine load sweep" in out.read_text(encoding="utf-8")


def test_cli_run_writes_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr(sweep_mod, "run_sweep",
                        lambda spec, **kw: _synth_result(spec.engine))
    out = tmp_path / "r.json"
    html = tmp_path / "r.html"
    rc = main(["--engine", "vllm", "--base-url", "http://x", "--levels", "1,2",
               "--measurement", "2", "--warmup", "1", "--repeats", "1",
               "--out", str(out), "--html", str(html)])
    assert rc == 0
    assert json.loads(out.read_text(encoding="utf-8"))["schema"] == sweep_mod.SWEEP_SCHEMA
    assert html.exists()


# --------------------------------------------------------- token throughput
def test_output_token_throughput_from_generation_counter_delta():
    from infengine.sweep import _output_token_throughput

    m = _output_token_throughput("sglang", {"generation_tokens_total": {"value": 900}}, 6.0)
    assert m["value"] == 150.0 and m["dominant_quality"] == "fresh"
    assert m["counter"] == "generation_tokens_total" and m["window_s"] == 6.0
    m2 = _output_token_throughput("vllm", {"generation_tokens": {"value": 300}}, 6.0)
    assert m2["value"] == 50.0 and m2["counter"] == "generation_tokens"


def test_output_token_throughput_zero_vs_unavailable():
    from infengine.sweep import _output_token_throughput

    zero = _output_token_throughput(
        "sglang", {"generation_tokens_total": {"value": 0}}, 6.0)
    assert zero["value"] == 0.0 and zero["dominant_quality"] == "fresh"
    for reason in ("counter_reset", "unavailable", "instance_changed"):
        u = _output_token_throughput(
            "sglang", {"generation_tokens_total": {"value": None, "reason": reason}}, 6.0)
        assert u["value"] is None and u["dominant_quality"] == "unavailable"
    no_window = _output_token_throughput("sglang", {"generation_tokens_total": {"value": 10}}, 0.0)
    assert no_window["value"] is None


def test_sglang_derived_token_throughput_is_positive_and_separate_from_native():
    from infengine.report import build_panels

    panels = {p["id"]: p for p in build_panels(_synth_result("sglang"))}
    derived = panels["output_throughput"]["series"][0]
    native = panels["throughput_tokens"]["series"][0]
    assert any(pt[1] and pt[1] > 0 for pt in derived["points"])
    assert native["name"].startswith("native windowed gauge")


# ------------------------------------------------- interpretation hardening
def test_single_run_is_inconclusive():
    levels = [_level(1.0, [_run(1.0, tp=10)]), _level(2.0, [_run(2.0, tp=20)]),
              _level(4.0, [_run(4.0, tp=40)])]
    s = interpret_sweep(SweepSpec(engine="vllm", model="m", repeats=1), levels)
    assert s["verdict"] == "INCONCLUSIVE"
    assert any("single run" in c for c in s["caveats"])


def test_non_monotonic_throughput_inconclusive():
    levels = [_level(1.0, [_run(1.0, tp=10), _run(1.0, tp=10)]),
              _level(2.0, [_run(2.0, tp=20), _run(2.0, tp=20)]),
              _level(4.0, [_run(4.0, tp=15), _run(4.0, tp=15)])]
    s = interpret_sweep(SweepSpec(engine="vllm", model="m", repeats=2), levels)
    assert s["observations"]["throughput_non_monotonic"] is True
    assert s["verdict"] == "INCONCLUSIVE"


def test_censoring_makes_level_mixed_and_inconclusive():
    censored = _run(2.0, counts={"started": 20, "completed": 5, "drained": 0,
                                 "timeout": 3, "error": 0, "incomplete": 0})
    levels = [_level(1.0, [_run(1.0)]), _level(2.0, [censored])]
    assert levels[1]["censored"] is True
    assert levels[1]["evidence_quality"] == "mixed"
    s = interpret_sweep(SweepSpec(engine="vllm", model="m", repeats=1), levels)
    assert 2.0 in s["evidence"]["censored_levels"]
    assert s["verdict"] == "INCONCLUSIVE"


def test_runtime_config_changed_inconclusive():
    levels = [_level(1.0, [_run(1.0, tp=10), _run(1.0, tp=10)]),
              _level(2.0, [_run(2.0, tp=20), _run(2.0, tp=20)])]
    s = interpret_sweep(SweepSpec(engine="vllm", model="m", repeats=2), levels,
                        runtime_config_changed=True)
    assert s["verdict"] == "INCONCLUSIVE"
    assert any("runtime configuration changed" in c for c in s["caveats"])


# ------------------------------------------------- window-contamination guards
def test_counter_delta_excludes_warmup_and_collector_is_fresh(monkeypatch):
    from infengine.telemetry import RuntimeSnapshot

    def snap(pre):
        return RuntimeSnapshot(
            engine="vllm", instance_id="i1",
            collected_at="2026-01-01T00:00:00+00:00", source="x",
            engine_specific={"num_preemptions": pre},
        )

    made = []

    class FakeCollector:
        def __init__(self):
            self.seq = [snap(100), snap(200), snap(210)]  # idle, post-warmup, final
            self.i = 0
            self._latest = None
        window_s = 1.0
        poll_interval = 0.1

        def poll_once(self):
            self._latest = self.seq[self.i]
            self.i += 1
            return self._latest

        def latest(self):
            return self._latest

    def build(engine, url, measurement):
        c = FakeCollector()
        made.append(c)
        return c

    def fake_measure(exp, cfg, bound):
        cfg.collector_factory("x").poll_once()  # measurement-end poll
        return {"ok": True, "counts": {"timeout": 0, "incomplete": 0},
                "metrics": {}, "environment": {}, "gpu": {}}

    monkeypatch.setattr(sweep_mod, "_build_collector", build)
    monkeypatch.setattr(sweep_mod, "_warmup", lambda exp, cfg: True)
    monkeypatch.setattr(sweep_mod, "_measure_phase", fake_measure)

    spec = SweepSpec(engine="vllm", model="m", load_model="closed_loop",
                     levels=[1.0], warmup_requests=2, window_s=1.0)
    run = run_level_once(spec, 1.0, engine_url="http://x", telemetry_url="http://x",
                         run_index=0)
    # warmup (100 -> 200) is excluded; only 200 -> 210 is the measurement delta.
    assert run["counter_deltas"]["num_preemptions"]["value"] == 10
    assert run["evidence_windows"]["counter_deltas"]
    # A second run builds a FRESH collector (no RollingHistograms reuse).
    run_level_once(spec, 1.0, engine_url="http://x", telemetry_url="http://x",
                   run_index=1)
    assert len(made) == 2 and made[0] is not made[1]


def test_output_token_throughput_window_and_boundaries(monkeypatch):
    from infengine.telemetry import RuntimeSnapshot

    def snap(gen):
        return RuntimeSnapshot(
            engine="sglang", instance_id="i1",
            collected_at="2026-01-01T00:00:00+00:00", source="x",
            engine_specific={"generation_tokens_total": gen},
        )

    class FakeCollector:
        def __init__(self):
            self.seq = [snap(0), snap(100), snap(700)]  # idle, post-warmup, final
            self.i = 0
            self._latest = None
        window_s = 1.0
        poll_interval = 0.1

        def poll_once(self):
            self._latest = self.seq[self.i]
            self.i += 1
            return self._latest

        def latest(self):
            return self._latest

    collector = FakeCollector()
    monkeypatch.setattr(sweep_mod, "_build_collector", lambda e, u, m: collector)
    monkeypatch.setattr(sweep_mod, "_warmup", lambda exp, cfg: True)

    def fake_measure(exp, cfg, bound):
        cfg.collector_factory("x").poll_once()  # final measurement poll
        return {"ok": True, "counts": {"timeout": 0, "incomplete": 0}, "wall_s": 6.0,
                "metrics": {}, "environment": {}, "gpu": {}}

    monkeypatch.setattr(sweep_mod, "_measure_phase", fake_measure)
    spec = SweepSpec(engine="sglang", model="m", levels=[1.0], warmup_requests=1,
                     window_s=1.0)
    run = run_level_once(spec, 1.0, engine_url="http://x", telemetry_url="http://x",
                         run_index=0)
    # warmup (0 -> 100) excluded; measurement delta 100 -> 700 = 600 over ~6s.
    assert run["counter_deltas"]["generation_tokens_total"]["value"] == 600
    assert run["metrics"]["output_tokens_per_s"]["value"] > 0
    assert run["evidence_windows"]["output_tokens_per_s"]
    assert "final_poll" in run["evidence_windows"]


# ------------------------------------------------------------------ integration
def test_run_level_once_integration_builtin_service():
    engine = _make_engine(max_batch=2)
    app = _make_app(engine)
    loads = sglang_loads_exposition(running=0, waiting=0)
    spec = SweepSpec(
        engine="sglang", model="fake", load_model="closed_loop", levels=[1.0],
        prompt_tokens=8, max_new_tokens=4, seed=1, measurement_s=2.0, max_requests=50,
        warmup_requests=2, timeout_s=5.0, drain_budget_s=1.0,
        window_s=0.5, poll_interval_s=0.2, min_polls=2, min_requests=1,
    )
    with _served(app) as control_url:
        with telemetry_server({"/v1/loads?format=prometheus": (200, loads),
                               "/metrics": (404, "not found")}) as tel:
            run = run_level_once(spec, 1.0, engine_url=control_url, telemetry_url=tel,
                                 run_index=0)
    assert run["ok"] is True, run.get("failure_reason")
    assert run["counts"]["started"] > 0
    assert run["metrics"]["throughput_req_s"]["value"] is not None
    # D1 evidence path: normalized running/waiting are collected per level.
    assert "requests_running" in run["metrics"]
    assert "requests_waiting" in run["metrics"]
    assert run["execution_evidence"] is not None
    assert run["snapshot"] is not None
    assert run["evidence_windows"]["execution_evidence"]
