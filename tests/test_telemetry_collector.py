"""M5B collector tests: parsing into RuntimeSnapshot with MetricQuality.

Uses a local HTTP server serving canned exposition (telemetry_helpers) as a
stand-in for the engines' native telemetry surfaces, covering parsing,
histogram-window calculations, counter resets, missing/disabled metrics, stale
data, scrape failures, and recovery.
"""

import socket
import time

import pytest

from infengine.telemetry import (
    MetricQuality,
    SglangTelemetryCollector,
    VllmTelemetryCollector,
    validate_snapshot,
)

from telemetry_helpers import (
    histogram_text,
    sglang_loads_exposition,
    sglang_metrics_exposition,
    telemetry_server,
    vllm_exposition,
)

TTFT_A = ([(0.01, 0), (0.1, 10), (1.0, 20), ("+Inf", 20)], 5.0, 20)
TTFT_B = ([(0.01, 5), (0.1, 25), (1.0, 40), ("+Inf", 40)], 10.0, 40)
TPOT_A = ([(0.01, 0), (0.1, 10), (1.0, 20), ("+Inf", 20)], 4.0, 20)
TPOT_B = ([(0.01, 4), (0.1, 14), (1.0, 24), ("+Inf", 24)], 5.0, 24)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def sequence(events):
    state = {"i": 0}

    def route():
        event = events[state["i"] % len(events)]
        state["i"] += 1
        return event

    return route


# ------------------------------------------------------------------- vLLM
def test_vllm_fresh_snapshot():
    body = vllm_exposition(
        waiting_reasons={"capacity": 1, "deferred": 0},
        ttft=TTFT_A,
    )
    with telemetry_server({"/metrics": (200, body)}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running == 2
    assert snap.requests_waiting == 1
    assert snap.kv_usage == 0.42
    assert snap.engine_specific["num_preemptions"] == 3
    assert snap.engine_specific["generation_tokens"] == 500
    assert snap.engine_specific["num_requests_waiting_by_reason"]["capacity"] == 1
    assert "iteration_tokens_total" in snap.engine_specific
    # One scrape -> no windowed quantile yet.
    assert snap.quality_of("ttft_p95_s") is MetricQuality.NO_SAMPLES
    assert snap.quality_of("requests_running") is MetricQuality.FRESH


def test_vllm_latency_quantiles_over_window():
    body_a = vllm_exposition(ttft=TTFT_A, tpot=TPOT_A)
    body_b = vllm_exposition(ttft=TTFT_B, tpot=TPOT_B)
    with telemetry_server({"/metrics": sequence([(200, body_a), (200, body_b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.ttft_p95_s is not None
    assert snap.tpot_p95_s is not None
    assert snap.ttft_p50_s <= snap.ttft_p95_s <= snap.ttft_p99_s


def test_vllm_no_samples_empty_histograms():
    empty_hist = histogram_text("vllm:time_to_first_token_seconds",
                                [(0.1, 0), ("+Inf", 0)], 0.0, 0)
    body = vllm_exposition() + "\n" + empty_hist + "\n"
    with telemetry_server({"/metrics": (200, body)}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.ttft_p95_s is None
    assert snap.quality_of("ttft_p95_s") is MetricQuality.NO_SAMPLES


def test_vllm_scrape_failure_http_500():
    with telemetry_server({"/metrics": (500, "boom")}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running is None
    assert snap.quality_of("requests_running") is MetricQuality.SCRAPE_FAILED
    assert snap.quality_of("ttft_p95_s") is MetricQuality.SCRAPE_FAILED
    assert snap.engine_specific == {}


def test_vllm_scrape_failure_connection_refused():
    c = VllmTelemetryCollector(f"http://127.0.0.1:{free_port()}", timeout=1.0)
    snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.quality_of("kv_usage") is MetricQuality.SCRAPE_FAILED


def test_vllm_disabled_empty_registry():
    # vLLM --disable-log-stats -> empty /metrics registry.
    with telemetry_server({"/metrics": (200, "# HELP none\n# TYPE none gauge\n")}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running is None
    assert snap.quality_of("requests_running") is MetricQuality.DISABLED
    assert snap.quality_of("ttft_p95_s") is MetricQuality.DISABLED
    assert snap.quality_of("num_preemptions") is MetricQuality.DISABLED


def test_vllm_partial_family_disabled():
    body = vllm_exposition()  # includes all gauges
    body = body.replace("vllm:kv_cache_usage_perc", "vllm:kv_cache_usage_perc_renamed")
    with telemetry_server({"/metrics": (200, body)}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running == 2
    assert snap.kv_usage is None
    assert snap.quality_of("kv_usage") is MetricQuality.DISABLED


def test_vllm_recovery_after_failure():
    ok = vllm_exposition(running=4)
    with telemetry_server({"/metrics": sequence([(500, ""), (200, ok)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        failed = c.poll_once()
        recovered = c.poll_once()
    assert failed.quality_of("requests_running") is MetricQuality.SCRAPE_FAILED
    validate_snapshot(recovered)
    assert recovered.requests_running == 4
    assert recovered.quality_of("requests_running") is MetricQuality.FRESH


def test_vllm_latest_and_background_loop():
    body = vllm_exposition()
    with telemetry_server({"/metrics": (200, body)}) as base:
        c = VllmTelemetryCollector(base, poll_interval=0.05, timeout=5.0)
        assert c.latest() is None
        c.start()
        deadline = time.time() + 3.0
        while c.latest() is None and time.time() < deadline:
            time.sleep(0.02)
        snap = c.latest()
        c.stop()
    assert snap is not None
    assert snap.requests_running == 2


def test_failed_scrape_does_not_leave_stale_fresh_values():
    # A failed scrape must REPLACE the previous fresh snapshot with a
    # SCRAPE_FAILED one — a consumer (e.g. /runtime/telemetry) must never see
    # stale values presented as fresh after a failure.
    ok = vllm_exposition(running=3)
    with telemetry_server({"/metrics": sequence([(200, ok), (500, "")])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()  # fresh
        assert c.latest().requests_running == 3
        c.poll_once()  # scrape fails
        latest = c.latest()
    assert latest.requests_running is None
    assert latest.quality_of("requests_running") is MetricQuality.SCRAPE_FAILED
    assert latest.quality_of("ttft_p95_s") is MetricQuality.SCRAPE_FAILED
    assert latest.engine_specific == {}


# ------------------------------------------------------------------- SGLang
def test_sglang_default_latency_disabled():
    loads = sglang_loads_exposition(running=1, waiting=2, token_usage=0.5)
    with telemetry_server({"/v1/loads?format=prometheus": (200, loads)}) as base:
        # /metrics unregistered -> 404 -> latency disabled.
        c = SglangTelemetryCollector(base, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running == 1
    assert snap.requests_waiting == 2
    assert snap.kv_usage == 0.5
    assert snap.quality_of("ttft_p95_s") is MetricQuality.DISABLED
    assert snap.quality_of("gen_throughput") is MetricQuality.FRESH
    assert snap.engine_specific["max_total_num_tokens"] == 100000
    assert snap.quality_of("kv_used_tokens") is MetricQuality.DISABLED


def test_sglang_metrics_enabled_latency():
    loads = sglang_loads_exposition()
    body_a = histogram_text("sglang:time_to_first_token_seconds", TTFT_A[0], TTFT_A[1], TTFT_A[2],
                            labels='dp_rank="0",is_streaming="false"')
    body_b = histogram_text("sglang:time_to_first_token_seconds", TTFT_B[0], TTFT_B[1], TTFT_B[2],
                            labels='dp_rank="0",is_streaming="false"')
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": sequence([(200, body_a), (200, body_b)]),
    }) as base:
        c = SglangTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.ttft_p95_s is not None
    assert snap.ttft_p50_s <= snap.ttft_p95_s <= snap.ttft_p99_s


def test_sglang_stale_via_timestamp():
    old = time.time() - 60.0
    loads = sglang_loads_exposition(running=1, token_usage=0.7, timestamp=old)
    with telemetry_server({"/v1/loads?format=prometheus": (200, loads)}) as base:
        c = SglangTelemetryCollector(base, staleness_s=10.0, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running == 1  # value retained
    assert snap.quality_of("requests_running") is MetricQuality.STALE
    assert snap.quality_of("kv_usage") is MetricQuality.STALE


def test_sglang_loads_failure_gauges_scrape_failed():
    with telemetry_server({
        "/v1/loads?format=prometheus": (500, "boom"),
        "/metrics": (404, "not found"),
    }) as base:
        c = SglangTelemetryCollector(base, timeout=5.0)
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.requests_running is None
    assert snap.quality_of("requests_running") is MetricQuality.SCRAPE_FAILED
    assert snap.quality_of("ttft_p95_s") is MetricQuality.DISABLED


def test_sglang_connection_refused_all_failed():
    c = SglangTelemetryCollector(f"http://127.0.0.1:{free_port()}", timeout=1.0)
    snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.quality_of("requests_running") is MetricQuality.SCRAPE_FAILED
    assert snap.quality_of("ttft_p95_s") is MetricQuality.SCRAPE_FAILED


def test_sglang_recovery_after_loads_failure():
    loads = sglang_loads_exposition(running=3)
    with telemetry_server({
        "/v1/loads?format=prometheus": sequence([(500, ""), (200, loads)]),
        "/metrics": (404, "not found"),
    }) as base:
        c = SglangTelemetryCollector(base, timeout=5.0)
        failed = c.poll_once()
        recovered = c.poll_once()
    assert failed.quality_of("requests_running") is MetricQuality.SCRAPE_FAILED
    validate_snapshot(recovered)
    assert recovered.requests_running == 3
    assert recovered.quality_of("requests_running") is MetricQuality.FRESH


# --------------------------------------------------- M9B execution-stage fields
def test_vllm_execution_stage_families_and_meta():
    pre_a = ([(0.05, 0), (0.5, 10), ("+Inf", 10)], 2.0, 10)
    pre_b = ([(0.05, 5), (0.5, 25), ("+Inf", 25)], 4.0, 25)
    body_a = vllm_exposition(prefill=pre_a)
    body_b = vllm_exposition(prefill=pre_b)
    with telemetry_server({"/metrics": sequence([(200, body_a), (200, body_b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    assert snap.engine_specific["request_prefill_time"]["p50"] is not None
    assert snap.quality_of("request_prefill_time") is MetricQuality.FRESH
    assert snap.measurement_meta["families"]["request_prefill_time"]["samples"] > 0
    assert snap.measurement_meta["window_s"] == c.window_s


def test_vllm_legacy_tpot_and_engine_family_share_one_window():
    body_a = vllm_exposition(tpot=TPOT_A)
    body_b = vllm_exposition(tpot=TPOT_B)
    with telemetry_server({"/metrics": sequence([(200, body_a), (200, body_b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    # Legacy normalized field and the correctly-named engine family coexist and
    # carry the same windowed value (one native histogram, one push).
    assert snap.tpot_p95_s is not None
    assert snap.engine_specific["inter_token_latency"]["p95"] == pytest.approx(
        snap.tpot_p95_s
    )
    assert snap.quality_of("inter_token_latency") is MetricQuality.FRESH


def test_vllm_disabled_execution_stage_families():
    with telemetry_server({"/metrics": (200, "# TYPE none gauge\n")}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    validate_snapshot(snap)
    assert snap.quality_of("request_prefill_time") is MetricQuality.DISABLED
    assert snap.quality_of("request_success") is MetricQuality.DISABLED
    assert "request_prefill_time" not in snap.engine_specific


def test_sglang_per_stage_and_execution_fields():
    ttft_a = ([(0.1, 0), (1.0, 10), ("+Inf", 10)], 2.0, 10)
    ttft_b = ([(0.1, 5), (1.0, 25), ("+Inf", 25)], 4.0, 25)
    loads = sglang_loads_exposition(token_usage=0.5, total_prefill_busy_us=999,
                                    total_prefill_uncached_tokens=123)
    body_a = sglang_metrics_exposition(
        ttft=ttft_a, per_stage={"prefill_forward": ttft_a},
        num_requests_total=10, decode_sum_seq_lens=16.0, kv_used_tokens=100,
    )
    body_b = sglang_metrics_exposition(
        ttft=ttft_b, per_stage={"prefill_forward": ttft_b},
        num_requests_total=20, decode_sum_seq_lens=24.0, kv_used_tokens=200,
    )
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": sequence([(200, body_a), (200, body_b)]),
    }) as base:
        c = SglangTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    validate_snapshot(snap)
    stage = snap.engine_specific["per_stage_req_latency"]["prefill_forward"]
    assert stage["p50"] is not None
    assert snap.quality_of("per_stage_req_latency") is MetricQuality.FRESH
    assert snap.engine_specific["total_prefill_busy_us"] == 999
    assert snap.engine_specific["total_prefill_uncached_tokens"] == 123
    assert snap.engine_specific["num_requests_total"] == 20
    assert snap.engine_specific["decode_sum_seq_lens"] == 24.0


# --------------------------------------------- M9B.1 vLLM counter name fix
def test_vllm_counters_collected_from_live_total_names():
    body = vllm_exposition(preemptions=3, prompt_tokens=100,
                           prompt_tokens_cached=40, generation_tokens=500,
                           prefix_queries=120, prefix_hits=60)
    with telemetry_server({"/metrics": (200, body)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    validate_snapshot(snap)
    assert snap.engine_specific["num_preemptions"] == 3
    assert snap.engine_specific["prompt_tokens"] == 100
    assert snap.engine_specific["prompt_tokens_cached"] == 40
    assert snap.engine_specific["generation_tokens"] == 500
    assert snap.engine_specific["prefix_cache_queries"] == 120
    assert snap.engine_specific["prefix_cache_hits"] == 60
    assert snap.quality_of("num_preemptions") is MetricQuality.FRESH


def test_vllm_counter_legacy_unsuffixed_fallback():
    # A surface that does NOT append _total still resolves via the explicit,
    # narrow fallback (deterministic precedence; no global suffix rule).
    body = vllm_exposition(preemptions=7, generation_tokens=42, legacy_counters=True)
    with telemetry_server({"/metrics": (200, body)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    validate_snapshot(snap)
    assert snap.engine_specific["num_preemptions"] == 7
    assert snap.engine_specific["generation_tokens"] == 42
    assert snap.quality_of("num_preemptions") is MetricQuality.FRESH


def test_vllm_counter_forms_are_never_summed():
    # Both forms present with different values: the primary suffixed form wins
    # and the unsuffixed form is NOT added.
    base_body = vllm_exposition(preemptions=5)
    extra = (
        "\n# TYPE vllm:num_preemptions counter\n"
        'vllm:num_preemptions{engine="0"} 7\n'
    )
    with telemetry_server({"/metrics": (200, base_body + extra)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    assert snap.engine_specific["num_preemptions"] == 5


def test_vllm_counter_zero_is_available_not_missing():
    body = vllm_exposition(preemptions=0)
    with telemetry_server({"/metrics": (200, body)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    # A genuine zero is present (available), never treated as absent.
    assert snap.engine_specific["num_preemptions"] == 0
    assert snap.quality_of("num_preemptions") is MetricQuality.FRESH


def test_vllm_missing_counter_is_unavailable_not_zero():
    body = "\n".join(
        line for line in vllm_exposition().splitlines()
        if "num_preemptions" not in line
    )
    with telemetry_server({"/metrics": (200, body)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    assert "num_preemptions" not in snap.engine_specific
    assert snap.quality_of("num_preemptions") is MetricQuality.DISABLED


def test_vllm_counter_increase_tracked():
    with telemetry_server({"/metrics": sequence([
        (200, vllm_exposition(preemptions=3)),
        (200, vllm_exposition(preemptions=9)),
    ])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        first = c.poll_once()
        second = c.poll_once()
    assert first.engine_specific["num_preemptions"] == 3
    assert second.engine_specific["num_preemptions"] == 9


def test_vllm_counter_reset_reports_smaller_value_not_negative():
    # A counter reset/restart yields a SMALLER cumulative value; the collector
    # must not wrap, negate or synthesize a delta.
    with telemetry_server({"/metrics": sequence([
        (200, vllm_exposition(preemptions=100, generation_tokens=5000)),
        (200, vllm_exposition(preemptions=4, generation_tokens=20)),
    ])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        after = c.poll_once()
    assert after.engine_specific["num_preemptions"] == 4
    assert after.engine_specific["generation_tokens"] == 20
    assert after.engine_specific["num_preemptions"] >= 0


def test_vllm_labelled_counter_family_preserved_and_aggregatable():
    body = vllm_exposition(request_success={"stop": 12, "length": 3},
                           prompt_tokens_by_source={"local_compute": 100})
    with telemetry_server({"/metrics": (200, body)}) as base:
        snap = VllmTelemetryCollector(base, timeout=5.0).poll_once()
    # finished_reason partitions the outcome space: summing is justified and the
    # per-label breakdown is preserved.
    assert snap.engine_specific["request_success"] == {"stop": 12, "length": 3}
    assert snap.engine_specific["prompt_tokens_by_source"] == {"local_compute": 100}


def test_vllm_preserved_histogram_single_push_no_double_count():
    # Two scrapes with an UNCHANGED cumulative histogram must yield a zero
    # window delta -> no_samples, and the family must not be present. A double
    # push of the same scrape would have produced a bogus non-zero delta.
    body = vllm_exposition(prefill=([(0.05, 0), (0.5, 10), ("+Inf", 10)], 2.0, 10))
    with telemetry_server({"/metrics": (200, body)}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    assert "request_prefill_time" not in snap.engine_specific
    assert snap.quality_of("request_prefill_time") is MetricQuality.NO_SAMPLES
    assert snap.measurement_meta["families"].get("request_prefill_time", {}).get(
        "samples", 0
    ) == 0


def test_vllm_preserved_histogram_sample_count_is_window_delta():
    a = ([(0.05, 0), (0.5, 10), ("+Inf", 10)], 2.0, 10)
    b = ([(0.05, 4), (0.5, 25), ("+Inf", 25)], 5.0, 25)
    with telemetry_server({"/metrics": sequence(
        [(200, vllm_exposition(prefill=a)), (200, vllm_exposition(prefill=b))]
    )}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        snap = c.poll_once()
    # Delta between cumulative counts 10 -> 25 is exactly 15 observations.
    assert snap.measurement_meta["families"]["request_prefill_time"]["samples"] == 15
    assert snap.measurement_meta["families"]["request_prefill_time"]["window_s"] == c.window_s


def test_vllm_preserved_histogram_counter_reset_is_not_misleading():
    before = ([(0.05, 0), (0.5, 25), ("+Inf", 25)], 8.0, 25)
    reset = ([(0.05, 0), (0.5, 3), ("+Inf", 3)], 1.0, 3)     # counter reset
    after = ([(0.05, 2), (0.5, 8), ("+Inf", 8)], 2.0, 8)
    bodies = sequence([
        (200, vllm_exposition(prefill=before)),
        (200, vllm_exposition(prefill=reset)),
        (200, vllm_exposition(prefill=after)),
    ])
    with telemetry_server({"/metrics": bodies}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()               # baseline 25
        s_reset = c.poll_once()     # reset to 3: pre-reset history dropped
        s_after = c.poll_once()     # delta 3 -> 8
    # The reset scrape cannot produce a quantile across the reset.
    assert "request_prefill_time" not in s_reset.engine_specific
    assert s_reset.quality_of("request_prefill_time") is MetricQuality.NO_SAMPLES
    # The next scrape measures only post-reset data (delta 5).
    assert s_after.measurement_meta["families"]["request_prefill_time"]["samples"] == 5
    assert s_after.engine_specific["request_prefill_time"]["p95"] is not None


def test_sglang_execution_stage_disabled_without_metrics_flag():
    loads = sglang_loads_exposition()
    with telemetry_server({
        "/v1/loads?format=prometheus": (200, loads),
        "/metrics": (404, "not found"),
    }) as base:
        snap = SglangTelemetryCollector(base, timeout=5.0).poll_once()
    validate_snapshot(snap)
    assert snap.quality_of("per_stage_req_latency") is MetricQuality.DISABLED
    assert snap.quality_of("inter_token_latency") is MetricQuality.DISABLED
    assert snap.quality_of("request_generation_tokens") is MetricQuality.DISABLED
    # Decode duration is never fabricated from other fields.
    assert "request_decode_time" not in snap.engine_specific