"""M5A contract tests: initial normalized RuntimeSnapshot.

These tests pin the M5A telemetry contract (mirrors
M5A_RUNTIME_TELEMETRY_CONTRACT.md):

  * The current normalized field set as the INITIAL SUBSET (extensible by
    design; additions are deliberate contract changes, not forbidden).
  * The exact engine-specific preserved metric sets (never normalized).
  * Source-name consistency, including the corrected ``queue_time_p99_s``
    SGLang source (``sglang:queue_time_seconds``) and explicit quantile
    arguments on every histogram_quantile expression.
  * Exact Prometheus exposition names, verified against live SGLang v0.5.18
    samples where available.
  * The MetricQuality availability model (fresh/stale/disabled/no_samples/
    scrape_failed) and its validation invariants.
  * Validation invariants: ranges, units, percentile ordering, ISO timestamp.
  * JSON round-trip of the snapshot.

No collector or time-series store is exercised: this module tests the contract
schema and its documented source/availability tables only.
"""

import math

import pytest

from infengine.telemetry import (
    FIELD_META,
    GAUGE_FIELDS,
    LATENCY_FAMILIES,
    LATENCY_FIELDS,
    NORMALIZED_FIELDS,
    PRESERVED_METRICS,
    SCHEMA_VERSION,
    VERIFIED_LIVE_SGLANG_EXPOSITION,
    VERIFIED_LIVE_SGLANG_METRICS_ENABLED,
    EngineKind,
    MetricQuality,
    RuntimeSnapshot,
    SnapshotValidationError,
    exposition_name,
    native_metric,
    validate_snapshot,
)


def make_snapshot(**overrides):
    data = {
        "engine": "vllm",
        "instance_id": "vllm:http://localhost:8000",
        "collected_at": "2026-09-07T12:00:00+00:00",
        "source": "prometheus:/metrics",
    }
    data.update(overrides)
    return RuntimeSnapshot(**data)


# ----------------------------------------------------------- exact contract
def test_normalized_field_set_is_exact_and_smallest():
    assert set(NORMALIZED_FIELDS) == {
        "requests_running",
        "requests_waiting",
        "kv_usage",
        "ttft_p50_s",
        "ttft_p95_s",
        "ttft_p99_s",
        "tpot_p50_s",
        "tpot_p95_s",
        "tpot_p99_s",
        "e2e_p50_s",
        "e2e_p95_s",
        "e2e_p99_s",
        "queue_time_p50_s",
        "queue_time_p95_s",
        "queue_time_p99_s",
    }
    assert set(GAUGE_FIELDS) == {"requests_running", "requests_waiting", "kv_usage"}
    assert set(GAUGE_FIELDS) | set(LATENCY_FIELDS) == set(NORMALIZED_FIELDS)


def test_field_meta_covers_exactly_the_normalized_fields():
    assert set(FIELD_META) == set(NORMALIZED_FIELDS)
    for f in NORMALIZED_FIELDS:
        meta = FIELD_META[f]
        assert meta.name == f
        assert meta.unit in ("count", "ratio", "seconds")
        assert meta.vllm_source and meta.sglang_source


def test_latency_families_reference_real_fields():
    for family, (p50, p95, p99) in LATENCY_FAMILIES.items():
        assert p50.endswith("_p50_s") and p95.endswith("_p95_s") and p99.endswith("_p99_s")
        assert {p50, p95, p99} <= set(NORMALIZED_FIELDS)


def test_preserved_metric_sets_are_exact_per_engine():
    assert set(PRESERVED_METRICS) == {EngineKind.VLLM.value, EngineKind.SGLANG.value}
    vllm_preserved = set(PRESERVED_METRICS["vllm"])
    sglang_preserved = set(PRESERVED_METRICS["sglang"])
    assert "num_preemptions" in vllm_preserved
    assert "generation_tokens" in vllm_preserved
    assert "gen_throughput" in sglang_preserved
    assert "num_retracted_requests_total" in sglang_preserved
    # Engine-specific names must not collide with normalized field names.
    assert vllm_preserved.isdisjoint(NORMALIZED_FIELDS)
    assert sglang_preserved.isdisjoint(NORMALIZED_FIELDS)


def test_default_availability_matrix_matches_documented_behavior():
    # vLLM /metrics is on by default for gauges and latency histograms.
    for f in NORMALIZED_FIELDS:
        assert FIELD_META[f].vllm_default is True
    # SGLang gauges are available by default via /v1/loads; latency percentiles
    # require --enable-metrics.
    for f in GAUGE_FIELDS:
        assert FIELD_META[f].sglang_default is True
    for f in LATENCY_FIELDS:
        assert FIELD_META[f].sglang_default is False
        assert "--enable-metrics" in FIELD_META[f].requires


# ------------------------------------------- source names / derivation rules
def test_queue_time_p99_s_sglang_source_is_corrected():
    # Regression: the SGLang source must be the real sglang:queue_time_seconds,
    # never the vLLM-named request_queue_time_seconds.
    meta = FIELD_META["queue_time_p99_s"]
    assert meta.sglang_source == "histogram_quantile(0.99, sglang:queue_time_seconds)"
    assert meta.vllm_source == "histogram_quantile(0.99, vllm:request_queue_time_seconds)"


def test_all_percentile_sources_carry_explicit_quantile():
    # Every derived source must name its quantile (no bare histogram names).
    for f in LATENCY_FIELDS:
        for source in (FIELD_META[f].vllm_source, FIELD_META[f].sglang_source):
            assert source.startswith("histogram_quantile(")
            assert ", " in source
            quantile = source[len("histogram_quantile("):].split(",", 1)[0].strip()
            assert float(quantile) in (0.5, 0.95, 0.99)


def test_derived_flag_partitions_fields():
    for f in GAUGE_FIELDS:
        assert FIELD_META[f].derived is False
    for f in LATENCY_FIELDS:
        assert FIELD_META[f].derived is True


# ------------------------------------------------------------- exposition names
def test_exposition_name_sanitizes_colon():
    assert exposition_name("vllm:num_requests_running") == "vllm_num_requests_running"
    assert exposition_name("vllm:kv_cache_usage_perc") == "vllm_kv_cache_usage_perc"
    assert exposition_name("sglang:num_running_reqs") == "sglang_num_running_reqs"
    assert exposition_name("sglang:queue_time_seconds") == "sglang_queue_time_seconds"


def test_native_metric_extracts_from_derived_expression():
    assert (
        native_metric("histogram_quantile(0.95, vllm:time_to_first_token_seconds)")
        == "vllm:time_to_first_token_seconds"
    )
    assert native_metric("vllm:num_requests_running") == "vllm:num_requests_running"


def test_field_exposition_names_are_consistent_with_sanitization():
    for f in NORMALIZED_FIELDS:
        meta = FIELD_META[f]
        vllm_expo = meta.vllm_exposition()
        sglang_expo = meta.sglang_exposition()
        assert vllm_expo == exposition_name(native_metric(meta.vllm_source))
        assert sglang_expo == exposition_name(native_metric(meta.sglang_source))
        assert ":" not in vllm_expo and ":" not in sglang_expo
        assert vllm_expo.startswith("vllm_")
        assert sglang_expo.startswith("sglang_")


def test_sglang_exposition_names_match_live_verified_samples():
    # Verified 2026-09-07 against a real SGLang v0.5.18 at localhost:30000 via
    # GET /v1/loads?format=prometheus (see VERIFIED_LIVE_SGLANG_EXPOSITION).
    assert VERIFIED_LIVE_SGLANG_METRICS_ENABLED is False
    assert set(VERIFIED_LIVE_SGLANG_EXPOSITION) == set(GAUGE_FIELDS)
    for f in GAUGE_FIELDS:
        live_name = VERIFIED_LIVE_SGLANG_EXPOSITION[f]
        assert exposition_name(native_metric(FIELD_META[f].sglang_source)) in (
            live_name,
            # requests_waiting: /metrics gauge is sglang_num_queue_reqs, while
            # the always-on /v1/loads surface names the same semantic
            # sglang_num_waiting_reqs (both documented).
            FIELD_META["requests_waiting"].sglang_exposition(),
        )


# ------------------------------------------------------------ quality model
def test_metric_quality_has_exactly_five_distinct_states():
    assert set(MetricQuality) == {
        MetricQuality.FRESH,
        MetricQuality.STALE,
        MetricQuality.DISABLED,
        MetricQuality.NO_SAMPLES,
        MetricQuality.SCRAPE_FAILED,
    }


def test_quality_fresh_and_stale_require_a_value():
    validate_snapshot(make_snapshot(requests_running=1, quality={"requests_running": "fresh"}))
    validate_snapshot(make_snapshot(requests_running=1, quality={"requests_running": "stale"}))
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(quality={"requests_running": "fresh"}))
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(quality={"requests_running": "stale"}))


def test_quality_absent_states_require_no_value():
    for state in ("disabled", "no_samples", "scrape_failed"):
        validate_snapshot(make_snapshot(quality={"ttft_p95_s": state}))
        with pytest.raises(SnapshotValidationError):
            validate_snapshot(
                make_snapshot(ttft_p95_s=0.2, quality={"ttft_p95_s": state})
            )


def test_quality_applies_to_preserved_engine_specific_metrics():
    snap = make_snapshot(
        engine="sglang",
        engine_specific={"gen_throughput": 0.0},
        quality={"gen_throughput": "stale"},
    )
    validate_snapshot(snap)
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(
            make_snapshot(
                engine="sglang",
                engine_specific={"gen_throughput": 0.0},
                quality={"gen_throughput": "disabled"},
            )
        )


def test_quality_rejects_unknown_keys_and_states():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(quality={"bogus_field": "fresh"}))
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(quality={"requests_running": "nope"}))


def test_quality_of_derives_defaults():
    snap = make_snapshot(requests_running=1)
    assert snap.quality_of("requests_running") is MetricQuality.FRESH
    assert snap.quality_of("requests_waiting") is None
    assert (
        make_snapshot(quality={"requests_waiting": "disabled"}).quality_of("requests_waiting")
        is MetricQuality.DISABLED
    )


def test_sglang_latency_disabled_snapshot_is_valid():
    # A realistic SGLang default-config snapshot: gauges fresh, latency
    # disabled (no --enable-metrics), preserved metrics with quality.
    snap = make_snapshot(
        engine="sglang",
        instance_id="sglang:http://localhost:30000",
        source="http:/v1/loads",
        requests_running=0,
        requests_waiting=0,
        kv_usage=0.0,
        engine_specific={"gen_throughput": 0.0, "cache_hit_rate": 0.0},
        quality={
            "ttft_p50_s": "disabled",
            "ttft_p95_s": "disabled",
            "ttft_p99_s": "disabled",
            "tpot_p50_s": "disabled",
            "tpot_p95_s": "disabled",
            "tpot_p99_s": "disabled",
            "e2e_p50_s": "disabled",
            "e2e_p95_s": "disabled",
            "e2e_p99_s": "disabled",
            "queue_time_p50_s": "disabled",
            "queue_time_p95_s": "disabled",
            "queue_time_p99_s": "disabled",
        },
    )
    validate_snapshot(snap)
    assert snap.available() == GAUGE_FIELDS


# ---------------------------------------------------------------- validation
def test_minimal_snapshot_is_valid():
    snap = make_snapshot()
    validate_snapshot(snap)
    assert snap.available() == ()


def test_full_valid_snapshot_passes():
    snap = make_snapshot(
        requests_running=3,
        requests_waiting=2,
        kv_usage=0.42,
        ttft_p50_s=0.05,
        ttft_p95_s=0.08,
        ttft_p99_s=0.12,
        tpot_p50_s=0.01,
        tpot_p95_s=0.02,
        tpot_p99_s=0.03,
        e2e_p50_s=1.0,
        e2e_p95_s=2.5,
        e2e_p99_s=4.0,
        queue_time_p50_s=0.1,
        queue_time_p95_s=0.4,
        queue_time_p99_s=1.0,
        engine_specific={"num_preemptions": 7, "generation_tokens": 1024},
    )
    validate_snapshot(snap)
    assert snap.available() == NORMALIZED_FIELDS


def test_rejects_unknown_engine():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(engine="ollama"))


def test_rejects_negative_count():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(requests_waiting=-1))


def test_rejects_fractional_count():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(requests_running=1.5))


def test_rejects_kv_usage_out_of_range():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(kv_usage=1.5))
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(kv_usage=-0.1))


def test_rejects_non_monotone_percentiles():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(ttft_p50_s=0.5, ttft_p95_s=0.4, ttft_p99_s=0.6))


def test_rejects_non_finite_latency():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(ttft_p95_s=math.inf))
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(tpot_p50_s=math.nan))


def test_allows_partial_latency_families():
    # A partially-populated family is valid when the present quantiles are
    # monotone (p50 omitted is fine; a missing quantile means "no samples").
    validate_snapshot(make_snapshot(ttft_p95_s=0.2))
    validate_snapshot(make_snapshot(ttft_p50_s=0.1, ttft_p95_s=0.2))


def test_rejects_bad_timestamp():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(collected_at="not a timestamp"))


def test_rejects_non_finite_engine_specific_value():
    with pytest.raises(SnapshotValidationError):
        validate_snapshot(make_snapshot(engine_specific={"generation_tokens": math.inf}))


def test_allows_non_numeric_engine_specific_values():
    validate_snapshot(make_snapshot(engine_specific={"notes": "idle"}))


# -------------------------------------------------------------- serialization
def test_to_dict_from_dict_round_trip():
    snap = make_snapshot(
        requests_running=2,
        kv_usage=0.5,
        ttft_p95_s=0.08,
        engine_specific={"num_preemptions": 3, "gen_throughput": 12.5},
        quality={"ttft_p95_s": "fresh", "gen_throughput": "stale"},
    )
    restored = RuntimeSnapshot.from_dict(snap.to_dict())
    assert restored == snap
    assert restored.quality == {"ttft_p95_s": "fresh", "gen_throughput": "stale"}


def test_from_dict_ignores_unknown_normalized_keys():
    data = make_snapshot().to_dict()
    data["normalized"]["requests_running"] = 1
    data["normalized"]["bogus_field"] = 99
    restored = RuntimeSnapshot.from_dict(data)
    assert restored.requests_running == 1
    assert not hasattr(restored, "bogus_field")


def test_normalized_excludes_metadata_and_engine_specific():
    snap = make_snapshot(requests_running=1, engine_specific={"x": 1})
    norm = snap.normalized()
    assert "requests_running" in norm
    assert "engine" not in norm
    assert "engine_specific" not in norm


# ---------------------------------------------------------- extensibility
def test_schema_version_marks_extensible_minor_bump():
    # 1.1 -> 1.2: the additive preserved metric max_running_requests was added
    # to PRESERVED_METRICS["sglang"] in M5B (additive compatible -> minor bump).
    # 1.2 -> 1.3: M9B added engine-specific execution-stage preserved families
    # and the additive measurement_meta sidecar; no normalized field changed.
    assert SCHEMA_VERSION == "1.3"


# ------------------------------------------------- M9B execution-stage contract
def test_execution_stage_preserved_families_exist():
    vllm = PRESERVED_METRICS["vllm"]
    for key in (
        "request_prefill_time",
        "request_decode_time",
        "request_inference_time",
        "inter_token_latency",
        "request_time_per_output_token",
        "request_prompt_tokens",
        "request_generation_tokens",
        "request_success",
        "prompt_tokens_by_source",
    ):
        assert key in vllm
    sglang = PRESERVED_METRICS["sglang"]
    for key in (
        "per_stage_req_latency",
        "inter_token_latency",
        "request_prompt_tokens",
        "request_uncached_prompt_tokens",
        "request_generation_tokens",
        "num_requests_total",
        "num_aborted_requests_total",
        "forward_execution_seconds_total",
        "fwd_occupancy",
        "decode_sum_seq_lens",
        "num_used_tokens",
        "num_total_tokens",
        "num_active_tokens",
        "num_waiting_uncached_tokens",
        "total_prefill_uncached_tokens",
        "total_prefill_busy_us",
    ):
        assert key in sglang


def test_vllm_m9b_counter_sources_use_live_total_suffix():
    # Live-verified on vLLM v0.28.0 (M9B): prometheus_client exposes counters
    # with a '_total' suffix. The M9B-added counters pin the live names.
    assert (
        PRESERVED_METRICS["vllm"]["request_success"].engine_source
        == "vllm:request_success_total"
    )
    assert (
        PRESERVED_METRICS["vllm"]["prompt_tokens_by_source"].engine_source
        == "vllm:prompt_tokens_by_source_total"
    )


def test_vllm_existing_counter_sources_use_live_total_names():
    # M9B.1: live-verified v0.28.0 counter exposition names (Prometheus '_total'
    # suffix). InfEngine-facing field names are unchanged.
    for key, name in (
        ("num_preemptions", "vllm:num_preemptions_total"),
        ("prompt_tokens", "vllm:prompt_tokens_total"),
        ("prompt_tokens_cached", "vllm:prompt_tokens_cached_total"),
        ("generation_tokens", "vllm:generation_tokens_total"),
        ("prefix_cache_queries", "vllm:prefix_cache_queries_total"),
        ("prefix_cache_hits", "vllm:prefix_cache_hits_total"),
    ):
        assert PRESERVED_METRICS["vllm"][key].engine_source == name


def test_execution_stage_sources_are_native_and_engine_prefixed():
    for engine, prefix in (("vllm", "vllm:"), ("sglang", "sglang:")):
        for key, metric in PRESERVED_METRICS[engine].items():
            assert metric.engine_source.startswith(prefix), (engine, key)
            assert metric.unit and metric.semantics


def test_legacy_tpot_fields_are_marked_engine_dependent():
    tx = __import__("infengine.telemetry", fromlist=["LEGACY_ENGINE_DEPENDENT_FIELDS"])
    assert set(tx.LEGACY_ENGINE_DEPENDENT_FIELDS) == {
        "tpot_p50_s", "tpot_p95_s", "tpot_p99_s",
    }
    assert set(tx.LEGACY_ENGINE_DEPENDENT_FIELDS) <= set(NORMALIZED_FIELDS)
    # The legacy tpot semantics text names the engine-dependence explicitly.
    assert "LEGACY" in FIELD_META["tpot_p50_s"].semantics
    assert "MUST NOT" in FIELD_META["tpot_p50_s"].semantics


def test_measurement_meta_is_additive_and_round_trips():
    snap = make_snapshot(
        requests_running=1,
        measurement_meta={
            "window_s": 30.0,
            "staleness_s": 20.0,
            "families": {"ttft": {"samples": 5, "window_s": 30.0}},
        },
    )
    validate_snapshot(snap)
    restored = RuntimeSnapshot.from_dict(snap.to_dict())
    assert restored == snap
    assert restored.measurement_meta["families"]["ttft"]["samples"] == 5


def test_measurement_meta_defaults_empty_for_existing_snapshots():
    data = make_snapshot(requests_running=1).to_dict()
    data.pop("measurement_meta", None)
    restored = RuntimeSnapshot.from_dict(data)
    assert restored.measurement_meta == {}