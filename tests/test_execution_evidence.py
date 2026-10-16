"""M9B execution-stage evidence tests.

Source-grounded Prometheus fixtures are served for both engines and the
read-only execution-evidence contract is exercised over real collector output.
Covers classification (MEASURED / DERIVED_ESTIMATE / UNAVAILABLE), stage
labelling, sample counts, quantiles, missing/disabled metrics, and the hard
"no inferred phase duration" rule.
"""

import pytest

from infengine.telemetry import (
    MeasurementClassification,
    SglangTelemetryCollector,
    VllmTelemetryCollector,
    build_execution_evidence,
    validate_snapshot,
)

from telemetry_helpers import (
    sglang_loads_exposition,
    sglang_metrics_exposition,
    telemetry_server,
    vllm_exposition,
)

# (le,count), sum, count
PRE_A = ([(0.05, 0), (0.5, 10), ("+Inf", 10)], 2.0, 10)
PRE_B = ([(0.05, 5), (0.5, 25), ("+Inf", 25)], 4.0, 25)
DEC_A = ([(0.10, 0), (1.0, 8), ("+Inf", 8)], 3.0, 8)
DEC_B = ([(0.10, 4), (1.0, 20), ("+Inf", 20)], 6.0, 20)
TTFT_A = ([(0.01, 0), (0.1, 10), ("+Inf", 10)], 0.5, 10)
TTFT_B = ([(0.01, 5), (0.1, 25), ("+Inf", 25)], 1.0, 25)


def _sequence(events):
    state = {"i": 0}

    def route():
        event = events[state["i"] % len(events)]
        state["i"] += 1
        return event

    return route


def _vllm_two_polls(a_overrides=None, b_overrides=None, **common):
    """Two scrapes with monotonically advancing prefill/decode/inference, plus
    optional per-poll overrides (needed so a preserved histogram actually has a
    non-zero window delta)."""
    kwargs_a = dict(ttft=TTFT_A, prefill=PRE_A, decode=DEC_A, inference=PRE_A, tpot=PRE_A)
    kwargs_b = dict(ttft=TTFT_B, prefill=PRE_B, decode=DEC_B, inference=PRE_B, tpot=PRE_B)
    kwargs_a.update(common)
    kwargs_a.update(a_overrides or {})
    kwargs_b.update(common)
    kwargs_b.update(b_overrides or {})
    body_a = vllm_exposition(**kwargs_a)
    body_b = vllm_exposition(**kwargs_b)
    with telemetry_server({"/metrics": _sequence([(200, body_a), (200, body_b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    return snap


# ------------------------------------------------------------------- vLLM
def test_vllm_execution_stage_measured():
    snap = _vllm_two_polls(
        a_overrides={
            "request_tpot": PRE_A,
            "request_prompt_tokens_hist": PRE_A,
            "request_generation_tokens_hist": DEC_A,
        },
        b_overrides={
            "request_tpot": PRE_B,
            "request_prompt_tokens_hist": PRE_B,
            "request_generation_tokens_hist": DEC_B,
        },
        request_success={"stop": 12, "length": 3},
        prompt_tokens_by_source={"local_compute": 100, "local_cache_hit": 40},
    )
    validate_snapshot(snap)
    ev = build_execution_evidence(snap)

    prefill = ev.request_stage["prefill_time"]
    assert prefill.classification == MeasurementClassification.MEASURED.value
    assert prefill.source == "vllm:request_prefill_time_seconds"
    assert prefill.sample_count and prefill.sample_count > 0
    assert prefill.quantiles["p50"] is not None
    assert "SCHEDULED" in prefill.boundary_note

    assert ev.request_stage["decode_time"].classification == "MEASURED"
    assert ev.request_stage["inference_time"].classification == "MEASURED"
    assert ev.request_stage["time_per_output_token"].classification == "MEASURED"
    assert ev.request_stage["inter_token_latency"].classification == "MEASURED"

    # The legacy tpot family is present but explicitly engine-dependent.
    legacy = ev.request_stage["tpot"]
    assert legacy.classification == "MEASURED"
    assert "LEGACY" in legacy.boundary_note

    # Multi-label counter preserved and summed.
    assert ev.engine_level["request_success"].value == 15.0
    assert ev.engine_level["num_preemptions"].classification == "MEASURED"


def test_vllm_single_scrape_is_unavailable_not_zero():
    # One scrape -> no windowed baseline -> every request-stage family must be
    # UNAVAILABLE with no value (never 0).
    body = vllm_exposition(prefill=PRE_A, decode=DEC_A)
    with telemetry_server({"/metrics": (200, body)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    ev = build_execution_evidence(snap)
    for key in ("prefill_time", "decode_time", "inference_time",
                "time_per_output_token", "inter_token_latency"):
        m = ev.request_stage[key]
        assert m.classification == MeasurementClassification.UNAVAILABLE.value
        assert m.value is None and m.quantiles == {}


def test_vllm_empty_registry_all_unavailable():
    with telemetry_server({"/metrics": (200, "# TYPE none gauge\n")}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    ev = build_execution_evidence(snap)
    assert ev.engine_level["requests_running"].classification == "UNAVAILABLE"
    assert ev.request_stage["prefill_time"].classification == "UNAVAILABLE"


def test_no_inferred_prefill_from_ttft_minus_queue():
    # ttft and queue_time are measured, but prefill_time must remain
    # UNAVAILABLE (the contract never computes TTFT - queue_time).
    body_a = vllm_exposition(ttft=TTFT_A, queue_time=PRE_A)
    body_b = vllm_exposition(ttft=TTFT_B, queue_time=PRE_B)
    with telemetry_server({"/metrics": _sequence([(200, body_a), (200, body_b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    ev = build_execution_evidence(snap)
    assert ev.request_stage["ttft"].classification == "MEASURED"
    assert ev.request_stage["queue_time"].classification == "MEASURED"
    assert ev.request_stage["prefill_time"].classification == "UNAVAILABLE"


# ------------------------------------------------------------------ SGLang
def _sglang_metrics_bodies():
    loads = sglang_loads_exposition(
        running=2, waiting=1, token_usage=0.5,
        num_used_tokens=900, num_total_tokens=100000, num_active_tokens=880,
        num_waiting_uncached_tokens=40, total_prefill_uncached_tokens=1234,
        total_prefill_busy_us=5678,
    )
    body_a = sglang_metrics_exposition(
        ttft=TTFT_A, tpot=PRE_A, per_stage={"prefill_forward": PRE_A},
        prompt_tokens_hist=PRE_A, generation_tokens_hist=DEC_A,
        forward_execution_seconds_total=10.0, fwd_occupancy=0.5,
        decode_sum_seq_lens=32.0, num_requests_total=40,
        num_aborted_requests_total=2, kv_used_tokens=900,
    )
    body_b = sglang_metrics_exposition(
        ttft=TTFT_B, tpot=PRE_B, per_stage={"prefill_forward": PRE_B},
        prompt_tokens_hist=PRE_B, generation_tokens_hist=DEC_B,
        forward_execution_seconds_total=20.0, fwd_occupancy=0.7,
        decode_sum_seq_lens=48.0, num_requests_total=80,
        num_aborted_requests_total=3, kv_used_tokens=950,
    )
    return loads, body_a, body_b


def test_sglang_execution_stage_measured_and_decode_unavailable():
    loads, body_a, body_b = _sglang_metrics_bodies()
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": _sequence([(200, body_a), (200, body_b)]),
    }) as base:
        c = SglangTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    ev = build_execution_evidence(snap)

    prefill = ev.request_stage["prefill_time"]
    assert prefill.classification == MeasurementClassification.MEASURED.value
    assert "prefill_forward" in prefill.source
    assert prefill.sample_count and prefill.sample_count > 0

    # SGLang emits no request-scoped decode duration: explicitly unavailable.
    assert ev.request_stage["decode_time"].classification == "UNAVAILABLE"
    assert ev.request_stage["decode_time"].value is None
    assert ev.request_stage["time_per_output_token"].classification == "UNAVAILABLE"

    # Token-weighted inter-token family is measured and distinct.
    itl = ev.request_stage["inter_token_latency"]
    assert itl.classification == "MEASURED"
    assert "token-weighted" in itl.boundary_note

    # Engine-level cumulative evidence preserved as measured.
    assert ev.engine_level["total_prefill_busy_us"].value == 5678.0
    assert ev.engine_level["total_prefill_uncached_tokens"].value == 1234.0
    assert ev.engine_level["decode_sum_seq_lens"].classification == "MEASURED"
    assert ev.engine_level["num_requests_total"].value == 80.0


def test_sglang_default_metrics_disabled_stage_unavailable():
    loads = sglang_loads_exposition(running=1, waiting=0, token_usage=0.2)
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": (404, "not found"),
    }) as base:
        snap = SglangTelemetryCollector(base, timeout=5.0).poll_once()
    ev = build_execution_evidence(snap)
    for key in ("prefill_time", "inter_token_latency", "request_prompt_tokens",
                "request_generation_tokens"):
        assert ev.request_stage[key].classification == "UNAVAILABLE"
    # engine-level gauges remain measured
    assert ev.engine_level["requests_running"].classification == "MEASURED"


# --------------------------------------------------------------- determinism
def test_evidence_is_deterministic_and_serializable():
    snap = _vllm_two_polls()
    a = build_execution_evidence(snap)
    b = build_execution_evidence(snap)
    assert a.to_dict() == b.to_dict()
    assert a.request_stage["prefill_time"].engine_version == "v0.28.0"
    assert isinstance(a.to_dict()["request_stage"], dict)


def test_engine_version_defaults_per_engine():
    loads = sglang_loads_exposition()
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": (404, "not found"),
    }) as base:
        snap = SglangTelemetryCollector(base, timeout=5.0).poll_once()
    ev = build_execution_evidence(snap)
    assert ev.engine_version == "v0.5.18"


def test_unsupported_engine_raises():
    from infengine.telemetry import RuntimeSnapshot

    snap = RuntimeSnapshot(
        engine="ollama", instance_id="x",
        collected_at="2026-09-19T00:00:00+00:00", source="s",
    )
    with pytest.raises(ValueError):
        build_execution_evidence(snap)
