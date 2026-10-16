"""M9C.1: native latency-histogram buckets (collection -> result -> report).

Covers cumulative->windowed conversion, cumulative->non-overlapping conversion,
reset/instance handling, availability (valid zero vs unavailable), stage-label
isolation, warmup/drain exclusion, backward compatibility, and rendering.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infengine.report import (  # noqa: E402
    _bucket_ranges,
    _histograms_html,
    build_histograms,
    render_report,
)
from infengine.sweep import _build_histograms  # noqa: E402
from infengine.telemetry import (  # noqa: E402
    MetricQuality,
    RuntimeSnapshot,
    RollingHistograms,
    SglangTelemetryCollector,
    VllmTelemetryCollector,
    validate_snapshot,
)

from telemetry_helpers import (  # noqa: E402
    sglang_loads_exposition,
    sglang_metrics_exposition,
    telemetry_server,
    vllm_exposition,
)

PRE_A = ([(0.05, 0), (0.5, 10), ("+Inf", 10)], 2.0, 10)
PRE_B = ([(0.05, 5), (0.5, 25), ("+Inf", 25)], 4.0, 25)
OTHER_A = ([(0.05, 0), (0.5, 2), ("+Inf", 2)], 0.4, 2)
OTHER_B = ([(0.05, 1), (0.5, 7), ("+Inf", 7)], 1.4, 7)


def _sequence(events):
    state = {"i": 0}

    def route():
        event = events[state["i"] % len(events)]
        state["i"] += 1
        return event

    return route


# ---------------------------------------------------- RollingHistograms delta
def test_window_buckets_is_windowed_cumulative_delta():
    w = RollingHistograms()
    w.push("f", ts=0.0, buckets=PRE_A[0])
    w.push("f", ts=1.0, buckets=PRE_B[0])
    le, counts, total = w.window_buckets("f", window_s=30.0)
    assert list(counts) == [5, 15, 15]
    assert total == 15
    assert total == w.total_in_window("f", 30.0)


def test_window_buckets_requires_baseline():
    w = RollingHistograms()
    w.push("f", ts=0.0, buckets=PRE_A[0])
    assert w.window_buckets("f", window_s=30.0) is None


def test_window_buckets_drops_pre_reset_history():
    w = RollingHistograms()
    w.push("f", ts=0.0, buckets=PRE_B[0])                       # cumulative 25
    w.push("f", ts=1.0, buckets=[(0.05, 1), (0.5, 3), ("+Inf", 3)])  # reset
    # Only one usable post-reset sample so far -> no valid window.
    assert w.window_buckets("f", window_s=30.0) is None
    w.push("f", ts=2.0, buckets=[(0.05, 2), (0.5, 8), ("+Inf", 8)])
    le, counts, total = w.window_buckets("f", window_s=30.0)
    assert total == 5 and list(counts) == [1, 5, 5]          # 3->8, post-reset only


def test_window_buckets_excludes_out_of_window_baseline():
    w = RollingHistograms(maxlen=10, retention_s=1000.0)
    w.push("f", ts=0.0, buckets=PRE_A[0])
    w.push("f", ts=100.0, buckets=PRE_B[0])
    # newest at 100s; baseline at 0s is outside a 30s window -> unusable
    assert w.window_buckets("f", window_s=30.0) is None


# ---------------------------------------------- cumulative -> non-overlapping
def test_cumulative_to_non_overlapping_buckets():
    buckets = _bucket_ranges([0.1, 0.2, "+Inf"], [2, 5, 7])
    assert [b["count"] for b in buckets] == [2, 3, 2]
    assert [b["cumulative"] for b in buckets] == [2, 5, 7]
    assert sum(b["count"] for b in buckets) == 7
    assert buckets[-1]["hi"] is None and buckets[-1]["lo"] == 0.2


def test_bucket_labels_expose_actual_ranges():
    buckets = _bucket_ranges([0.5, 2.0, "+Inf"], [1, 3, 4])
    assert buckets[0]["label"] == "< 0.5 s"
    assert buckets[1]["label"] == "0.5–2 s"
    assert buckets[2]["label"] == "≥ 2 s"


# ------------------------------------------------------------- collectors
def test_vllm_histogram_buckets_collected_and_consistent():
    a = vllm_exposition(ttft=PRE_A, queue_time=PRE_A, e2e=PRE_A, prefill=PRE_A, decode=PRE_A)
    b = vllm_exposition(ttft=PRE_B, queue_time=PRE_B, e2e=PRE_B, prefill=PRE_B, decode=PRE_B)
    with telemetry_server({"/metrics": _sequence([(200, a), (200, b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    hist = snap.measurement_meta["histograms"]
    family_key = {"ttft": "ttft", "queue_time": "queue_time", "e2e": "e2e",
                  "prefill_time": "request_prefill_time",
                  "decode_time": "request_decode_time"}
    for key, fam_key in family_key.items():
        assert key in hist, key
        assert hist[key]["total"] == 15
        assert hist[key]["counts"] == [5, 15, 15]
        assert hist[key]["le"][-1] == "+Inf"
        # bucket total agrees with the rolling-histogram sample count
        assert hist[key]["total"] == snap.measurement_meta["families"][fam_key]["samples"]


def test_sglang_histograms_no_decode_and_stage_isolation():
    loads = sglang_loads_exposition()
    a = sglang_metrics_exposition(
        ttft=PRE_A, queue_time=PRE_A, e2e=PRE_A,
        per_stage={"prefill_forward": PRE_A, "request_process": OTHER_A},
    )
    b = sglang_metrics_exposition(
        ttft=PRE_B, queue_time=PRE_B, e2e=PRE_B,
        per_stage={"prefill_forward": PRE_B, "request_process": OTHER_B},
    )
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": _sequence([(200, a), (200, b)]),
    }) as base:
        c = SglangTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    hist = snap.measurement_meta["histograms"]
    assert set(hist) == {"ttft", "queue_time", "e2e", "prefill_time"}
    assert "decode_time" not in hist  # never synthesized
    # prefill_time comes from stage=prefill_forward, NOT request_process
    assert hist["prefill_time"]["counts"] == [5, 15, 15]


def test_disabled_metrics_yield_no_histogram_buckets():
    loads = sglang_loads_exposition()
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": (404, "not found"),
    }) as base:
        snap = SglangTelemetryCollector(base, timeout=5.0).poll_once()
    assert "histograms" not in snap.measurement_meta or not snap.measurement_meta["histograms"]
    assert snap.quality_of("ttft_p95_s") is MetricQuality.DISABLED


def test_valid_zero_observations_vs_missing_family():
    # unchanged cumulative histogram -> valid zero-observation window
    body = vllm_exposition(ttft=PRE_A)
    with telemetry_server({"/metrics": _sequence([(200, body), (200, body)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    zero = snap.measurement_meta["histograms"]["ttft"]
    assert zero["total"] == 0 and zero["counts"] == [0, 0, 0]
    # a family that is not emitted at all -> no histogram entry (unavailable)
    assert "decode_time" not in snap.measurement_meta.get("histograms", {})


# ------------------------------------------------- sweep histogram assembly
class _FakeEvidence:
    def to_dict(self):
        return {"request_stage": {
            "ttft": {"classification": "MEASURED", "source": "vllm:time_to_first_token_seconds",
                     "unit": "seconds", "scope": "request_scoped_aggregated",
                     "boundary_note": "arrival -> first token"},
            "decode_time": {"classification": "UNAVAILABLE",
                            "source": "", "unit": "seconds", "scope": "request_scoped_aggregated",
                            "boundary_note": "not emitted"},
        }}


def _snap_with_histograms():
    return RuntimeSnapshot(
        engine="vllm", instance_id="i1",
        collected_at="2026-01-01T00:00:00+00:00", source="x",
        measurement_meta={"histograms": {
            "ttft": {"le": [0.1, "+Inf"], "counts": [2, 5], "total": 5,
                     "unit": "seconds", "window_s": 2.0},
        }},
    )


def test_sweep_histogram_assembly_and_decode_unavailable():
    out = _build_histograms("vllm", _snap_with_histograms(), _FakeEvidence(), min_requests=8)
    fams = out["families"]
    assert fams["ttft"]["available"] is True
    assert fams["ttft"]["total"] == 5 and fams["ttft"]["low_sample"] is True
    assert fams["ttft"]["source"] == "vllm:time_to_first_token_seconds"
    assert fams["decode_time"]["available"] is False
    assert "not emitted" in fams["decode_time"]["reason"]
    # not-collected family (queue_time) -> unavailable, window not established
    assert fams["queue_time"]["available"] is False


# --------------------------------------------------- report histogram build
def _run_with_histograms(level, ttft_counts, decode=None):
    fams = {
        "ttft": {"key": "ttft", "label": "TTFT", "available": True, "reason": None,
                 "source": "vllm:time_to_first_token_seconds", "unit": "seconds",
                 "boundary_note": "arrival -> first token",
                 "le": [0.1, 0.2, "+Inf"], "counts": ttft_counts,
                 "total": ttft_counts[-1], "window_s": 2.0,
                 "low_sample": ttft_counts[-1] < 8, "zero_observations": ttft_counts[-1] == 0},
        "decode_time": (decode or {"key": "decode_time", "label": "Decode (measured)",
                                   "available": False, "reason": "not emitted by this engine/version",
                                   "source": "", "unit": "seconds", "boundary_note": "x"}),
    }
    return {"run_index": 0, "ok": True, "counts": {}, "histograms": {"families": fams}}


def _result_with_histograms(runs):
    level = {"index": 0, "offered_load": 2.0, "runs": runs}
    return {"schema": "infengine.load_sweep/1", "generated_at": "t",
            "engine": {"name": "vllm", "version": {"version": "v0.28.0"}},
            "model": "m", "sweep": {"load_model": "closed_loop",
                                    "measurement": {"min_requests": 8}},
            "levels": [level], "summary": {}, "limitations": []}


def test_report_histograms_non_overlapping_and_selectors():
    result = _result_with_histograms([
        _run_with_histograms(2.0, [2, 5, 7]),
        _run_with_histograms(2.0, [1, 3, 4]),
    ])
    entries = build_histograms(result)
    # per-run + aggregate entries for ttft and decode
    ttft_runs = [e for e in entries if e["metric"] == "ttft" and not e["is_aggregate"]]
    assert len(ttft_runs) == 2
    assert [b["count"] for b in ttft_runs[0]["buckets"]] == [2, 3, 2]
    agg = [e for e in entries if e["metric"] == "ttft" and e["is_aggregate"]][0]
    assert agg["total"] == 11 and agg["buckets"][0]["count"] == 3  # 2+1
    decode = [e for e in entries if e["metric"] == "decode_time"][0]
    assert decode["available"] is False and "not emitted" in decode["reason"]
    html = _histograms_html(entries)
    assert "Latency distribution by load level" in html
    for sel in ("hist-level", "hist-metric", "hist-run"):
        assert sel in html


def test_report_histogram_low_sample_warning_flag():
    entries = build_histograms(_result_with_histograms([_run_with_histograms(2.0, [1, 3, 4])]))
    ttft = [e for e in entries if e["metric"] == "ttft"][0]
    assert ttft["low_sample"] is True


def test_report_missing_histograms_is_explicit():
    # A legacy M9C result without histogram data renders unchanged and states the
    # limitation explicitly (never fabricates a distribution).
    old = {"schema": "infengine.load_sweep/1", "engine": {"name": "vllm"},
           "sweep": {"measurement": {}},
           "levels": [{"index": 0, "offered_load": 1.0, "runs": []}],
           "summary": {}}
    assert build_histograms(old) == []
    assert "Latency distribution by load level" in render_report(old)
    assert "Native histogram distribution unavailable for this result." in render_report(old)


def test_render_report_embeds_histograms_for_sglang():
    result = _result_with_histograms([_run_with_histograms(2.0, [2, 5, 7])])
    result["engine"]["name"] = "sglang"
    html = render_report(result)
    assert "Latency distribution by load level" in html
    assert '"histograms"' in html
    assert "hist-canvas" in html
