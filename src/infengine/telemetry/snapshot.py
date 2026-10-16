"""M5A: initial normalized ``RuntimeSnapshot`` contract (out-of-band telemetry).

This module is the engine-neutral data contract for telemetry collected
OUT-OF-BAND (via the engines' native HTTP surfaces — Prometheus ``/metrics``,
SGLang ``/v1/loads``). It deliberately contains no collector, no HTTP client,
and no time-series store: the contract is the schema plus the authoritative,
machine-checkable record of metric semantics, units and availability for
vLLM v0.28.0 and SGLang v0.5.18.

EXTENSIBLE INITIAL SUBSET (not a frozen field universe)
-------------------------------------------------------

``NORMALIZED_FIELDS`` is the *initial* normalized subset, not a permanent exact
set. Future fields are added deliberately through the same contract machinery:

  * A new normalized field requires a :class:`FieldMeta` entry naming BOTH
    engines' native source, unit, exact semantic, and default availability,
    plus an entry in ``NORMALIZED_FIELDS`` (and ``LATENCY_FAMILIES`` when it
    belongs to a latency family). It is then pinned by the contract tests.
  * DERIVED fields (percentiles, rates) are allowed ONLY when the derivation is
    source-verified per engine (cite the native measurement and the formula,
    e.g. ``histogram_quantile(...)``) AND the two engines' native measurements
    are genuinely comparable (same semantic, same unit, same lifecycle point).
    Two different native measurements are NEVER silently equated under one
    field name (e.g. vLLM's derived counter rate and SGLang's windowed
    ``gen_throughput`` gauge stay separate and preserved).
  * Metrics whose native measurements differ in semantic, unit or lifecycle
    point remain engine-specific in :data:`PRESERVED_METRICS` /
    ``engine_specific`` — preserved verbatim, never renamed or merged.
  * Versioning: additive compatible fields bump the minor ``SCHEMA_VERSION``;
    removing or redefining a field bumps the major version.

Normalization rule (M5A):

  * A field is normalized ONLY when the two engines expose the same exact
    semantic with the same unit at the same lifecycle point.
  * Everything else is preserved engine-specific in ``engine_specific``.

The exact per-engine source for every normalized and preserved field is
recorded in :data:`FIELD_META` and :data:`PRESERVED_METRICS` (mirrors
``M5A_RUNTIME_TELEMETRY_CONTRACT.md``; engine source file:line citations live
in the milestone document). Exact Prometheus exposition names are derived by
:func:`exposition_name` / :func:`native_metric`; live-verified SGLang
exposition names are recorded in :data:`VERIFIED_LIVE_SGLANG_EXPOSITION`.

REVISION v1.1
-------------

  * Percentile source names made explicit and consistent: every
    ``histogram_quantile`` expression now carries its quantile argument, and
    the ``queue_time_p99_s`` SGLang source was corrected from the vLLM name
    ``sglang:request_queue_time_seconds`` to the real ``sglang:queue_time_seconds``.
  * Added the explicit :class:`MetricQuality` availability model (fresh /
    stale / disabled / no_samples / scrape_failed) with validation invariants.
  * Added exposition-name derivation and the live-verified SGLang names.
  * Reframed the field set as an extensible initial subset.

REVISION v1.3 (M9B - execution-stage telemetry, ADDITIVE)
---------------------------------------------------------

  * Added engine-specific preserved families for execution-stage evidence
    (vLLM request prefill/decode/inference time, per-step inter-token latency,
    per-request time-per-output-token, request token-count histograms and
    outcome counters; SGLang ``per_stage_req_latency``, token-weighted
    inter-token latency, request token histograms, scheduler/KV evidence and
    outcome counters). Nothing normalized was renamed, redefined or removed.
  * Added ``RuntimeSnapshot.measurement_meta`` -- the additive per-family
    aggregation window and sample-count sidecar consumed by the read-only
    execution-evidence contract (``infengine.telemetry.execution``).
  * Marked the legacy ``tpot_*`` fields as engine-dependent
    (:data:`LEGACY_ENGINE_DEPENDENT_FIELDS`); their values are unchanged.
  * No new normalized field. Minor version bump only.
"""

import copy
import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional, Tuple

SCHEMA_VERSION = "1.3"

# Legacy normalized field names that must be preserved for existing M5/M7
# consumers but whose per-engine native sources are NOT equivalent. The
# ``tpot_*`` fields were historically sourced from each engine's
# ``inter_token_latency_seconds`` metric; vLLM's is a per-engine-step interval
# while SGLang's is a token-weighted per-output-token average (see M9A/M9B).
# They therefore MUST NOT be interpreted as directly comparable cross-engine
# TPOT measurements. The fields are retained (values unchanged) and the
# correctly named engine-specific families live in ``PRESERVED_METRICS``
# (``inter_token_latency``, ``request_time_per_output_token``).
LEGACY_ENGINE_DEPENDENT_FIELDS: Tuple[str, ...] = (
    "tpot_p50_s",
    "tpot_p95_s",
    "tpot_p99_s",
)


class EngineKind(str, Enum):
    """Engines the contract normalizes across (vLLM v0.28.0, SGLang v0.5.18)."""

    VLLM = "vllm"
    SGLANG = "sglang"


class MetricQuality(str, Enum):
    """Explicit availability/quality state of a metric in a snapshot.

    The four non-fresh states are distinct and MUST NOT be conflated:

      * ``fresh``        — value present and current (within the collector's
                           staleness window).
      * ``stale``        — value present but last observed BEFORE the staleness
                           window (e.g. SGLang gauges hold their last value
                           during idle; windowed gauges reset on flush_cache).
      * ``disabled``     — metric not enabled on this engine/surface
                           (SGLang latency without ``--enable-metrics``; vLLM
                           with ``--disable-log-stats``); no value.
      * ``no_samples``   — enabled, but no observations in the window (e.g. an
                           empty latency histogram); no value.
      * ``scrape_failed``— the out-of-band source was unreachable or failed to
                           parse; no value.
    """

    FRESH = "fresh"
    STALE = "stale"
    DISABLED = "disabled"
    NO_SAMPLES = "no_samples"
    SCRAPE_FAILED = "scrape_failed"


class SnapshotValidationError(ValueError):
    """A RuntimeSnapshot violates the normalized contract invariants."""


def native_metric(source: str) -> str:
    """Extract the native Prometheus metric name from a source expression.

    Handles plain names (``vllm:num_requests_running``) and derived
    expressions (``histogram_quantile(0.5, vllm:time_to_first_token_seconds)``).
    """
    expr = source.strip()
    if expr.startswith("histogram_quantile("):
        expr = expr[len("histogram_quantile("):].split(",", 1)[1].strip()
        expr = expr.rstrip(")")
    return expr


def exposition_name(native: str) -> str:
    """Exact Prometheus text exposition name a scraper matches.

    Prometheus sanitizes ``:`` to ``_`` in metric names
    (``vllm:num_requests_running`` -> ``vllm_num_requests_running``).
    """
    return native.replace(":", "_")


# Live out-of-band verification of the exact exposition names (captured
# 2026-09-07 against a real SGLang v0.5.18 runtime at http://localhost:30000):
#
#   GET /v1/loads?format=prometheus  -> 200, gauges named
#       sglang_num_running_reqs, sglang_num_waiting_reqs, sglang_token_usage,
#       sglang_gen_throughput, sglang_cache_hit_rate, sglang_max_total_num_tokens,
#       sglang_max_running_requests, ...  (per-DP-rank label dp_rank="0")
#   GET /metrics                       -> 404  (confirms the running runtime was
#       launched WITHOUT --enable-metrics: SGLang latency histograms are
#       DISABLED by default, exactly as FIELD_META documents).
#
# vLLM v0.28.0: no live runtime was available in this environment, so the vLLM
# exposition names below are source-verified (registration strings in
# vllm/v1/metrics/loggers.py) plus the prometheus_client ":" -> "_" rule, not
# yet confirmed against a live scrape.
VERIFIED_LIVE_SGLANG_EXPOSITION: Dict[str, str] = {
    "requests_running": "sglang_num_running_reqs",
    "requests_waiting": "sglang_num_waiting_reqs",
    "kv_usage": "sglang_token_usage",
}
VERIFIED_LIVE_SGLANG_METRICS_ENABLED: bool = False


@dataclass(frozen=True)
class FieldMeta:
    """Machine-readable record of one normalized snapshot field.

    ``vllm_source`` / ``sglang_source`` name the actual engine metric or the
    derivation (e.g. ``histogram_quantile(0.95, <metric>)``). ``vllm_default`` /
    ``sglang_default`` state whether the field is available on the engine's
    DEFAULT surface (vLLM: Prometheus /metrics, on unless --disable-log-stats;
    SGLang: the always-on /v1/loads surface and/or /metrics when
    --enable-metrics). ``requires`` names the flag/setting needed when not
    available by default.
    """

    name: str
    unit: str
    kind: str
    semantics: str
    vllm_source: str
    sglang_source: str
    vllm_default: bool
    sglang_default: bool
    requires: str = ""

    @property
    def derived(self) -> bool:
        """Whether the field is DERIVED (computed) rather than scraped as-is."""
        return self.kind == "percentile"

    def vllm_exposition(self) -> str:
        return exposition_name(native_metric(self.vllm_source))

    def sglang_exposition(self) -> str:
        return exposition_name(native_metric(self.sglang_source))


@dataclass(frozen=True)
class PreservedMetric:
    """One engine-specific metric that MUST NOT be normalized.

    Preserved metrics keep their engine-native name, unit and exact semantic.
    Normalizing them would be misleading: e.g. vLLM's preemption counter has no
    SGLang equivalent (SGLang retraction is a different mechanism), and SGLang's
    windowed ``gen_throughput`` gauge is not comparable to a rate derived from
    vLLM's ``generation_tokens`` counter.
    """

    name: str
    unit: str
    semantics: str
    engine_source: str
    default_on: bool
    requires: str = ""


# --------------------------------------------------------------------------- fields
NORMALIZED_FIELDS: Tuple[str, ...] = (
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
)

GAUGE_FIELDS: Tuple[str, ...] = ("requests_running", "requests_waiting", "kv_usage")

LATENCY_FIELDS: Tuple[str, ...] = tuple(
    f for f in NORMALIZED_FIELDS if f not in GAUGE_FIELDS
)

# Latency families: base name -> (p50, p95, p99) field names.
LATENCY_FAMILIES: Dict[str, Tuple[str, str, str]] = {
    "ttft": ("ttft_p50_s", "ttft_p95_s", "ttft_p99_s"),
    "tpot": ("tpot_p50_s", "tpot_p95_s", "tpot_p99_s"),
    "e2e": ("e2e_p50_s", "e2e_p95_s", "e2e_p99_s"),
    "queue_time": ("queue_time_p50_s", "queue_time_p95_s", "queue_time_p99_s"),
}

# ------------------------------------------------------------------- field meta
# Authoritative source/availability table for the normalized fields.
#   vLLM   -> vllm:v1/metrics/loggers.py PrometheusStatLogger (gauge/counter/
#             histogram definitions) and vllm:v1/core/kv_cache_manager.py.
#   SGLang -> sglang:srt/observability/metrics_collector.py SchedulerMetrics/
#             TokenizerMetricsCollector and srt/entrypoints/v1_loads.py.
# Latency percentiles are DERIVED by the collector over a window via histogram
# quantiles; they are not scraped as scalars.
FIELD_META: Dict[str, FieldMeta] = {
    "requests_running": FieldMeta(
        name="requests_running",
        unit="count",
        kind="gauge",
        semantics="Instantaneous number of requests admitted into the scheduler "
        "running set (in prefill/decode execution).",
        vllm_source="vllm:num_requests_running",
        sglang_source="sglang:num_running_reqs",
        vllm_default=True,
        sglang_default=True,
        requires=(
            "sglang: /metrics (sglang_num_running_reqs) needs --enable-metrics; "
            "always-on /v1/loads exposes sglang_num_running_reqs"
        ),
    ),
    "requests_waiting": FieldMeta(
        name="requests_waiting",
        unit="count",
        kind="gauge",
        semantics="Instantaneous number of requests queued/waiting for admission "
        "(vLLM: waiting + skipped-waiting; SGLang: waiting queue).",
        vllm_source="vllm:num_requests_waiting",
        sglang_source="sglang:num_queue_reqs",
        vllm_default=True,
        sglang_default=True,
        requires=(
            "sglang: /metrics (sglang_num_queue_reqs) needs --enable-metrics; "
            "always-on /v1/loads exposes the same semantic as sglang_num_waiting_reqs"
        ),
    ),
    "kv_usage": FieldMeta(
        name="kv_usage",
        unit="ratio",
        kind="gauge",
        semantics="Fraction of the engine KV memory pool in use (0..1). vLLM: "
        "KV block-pool usage ratio of allocated GPU blocks. SGLang: max "
        "memory-pool token-usage ratio across full/SWA/Mamba pools. BOTH are "
        "instantaneous pool-utilization ratios; pool composition differs, so "
        "the per-pool breakdown is preserved engine-specific.",
        vllm_source="vllm:kv_cache_usage_perc",
        sglang_source="sglang:token_usage",
        vllm_default=True,
        sglang_default=True,
        requires=(
            "sglang: /metrics (sglang_token_usage) needs --enable-metrics; "
            "always-on /v1/loads exposes sglang_token_usage"
        ),
    ),
    "ttft_p50_s": FieldMeta(
        name="ttft_p50_s",
        unit="seconds",
        kind="percentile",
        semantics="Median time-to-first-token (request arrival -> first token), "
        "histogram quantile over the collection window.",
        vllm_source="histogram_quantile(0.5, vllm:time_to_first_token_seconds)",
        sglang_source="histogram_quantile(0.5, sglang:time_to_first_token_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "ttft_p95_s": FieldMeta(
        name="ttft_p95_s",
        unit="seconds",
        kind="percentile",
        semantics="P95 time-to-first-token over the collection window.",
        vllm_source="histogram_quantile(0.95, vllm:time_to_first_token_seconds)",
        sglang_source="histogram_quantile(0.95, sglang:time_to_first_token_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "ttft_p99_s": FieldMeta(
        name="ttft_p99_s",
        unit="seconds",
        kind="percentile",
        semantics="P99 time-to-first-token over the collection window.",
        vllm_source="histogram_quantile(0.99, vllm:time_to_first_token_seconds)",
        sglang_source="histogram_quantile(0.99, sglang:time_to_first_token_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "tpot_p50_s": FieldMeta(
        name="tpot_p50_s",
        unit="seconds",
        kind="percentile",
        semantics="LEGACY, ENGINE-DEPENDENT (see LEGACY_ENGINE_DEPENDENT_FIELDS): "
        "median of each engine's native inter_token_latency_seconds. vLLM is a "
        "per-engine-step interval; SGLang is a token-weighted per-output-token "
        "average. Values are retained for existing consumers but MUST NOT be "
        "compared across engines as TPOT.",
        vllm_source="histogram_quantile(0.5, vllm:inter_token_latency_seconds)",
        sglang_source="histogram_quantile(0.5, sglang:inter_token_latency_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "tpot_p95_s": FieldMeta(
        name="tpot_p95_s",
        unit="seconds",
        kind="percentile",
        semantics="P95 inter-token latency / time-per-output-token.",
        vllm_source="histogram_quantile(0.95, vllm:inter_token_latency_seconds)",
        sglang_source="histogram_quantile(0.95, sglang:inter_token_latency_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "tpot_p99_s": FieldMeta(
        name="tpot_p99_s",
        unit="seconds",
        kind="percentile",
        semantics="P99 inter-token latency / time-per-output-token.",
        vllm_source="histogram_quantile(0.99, vllm:inter_token_latency_seconds)",
        sglang_source="histogram_quantile(0.99, sglang:inter_token_latency_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "e2e_p50_s": FieldMeta(
        name="e2e_p50_s",
        unit="seconds",
        kind="percentile",
        semantics="Median end-to-end request latency (arrival -> finish).",
        vllm_source="histogram_quantile(0.5, vllm:e2e_request_latency_seconds)",
        sglang_source="histogram_quantile(0.5, sglang:e2e_request_latency_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "e2e_p95_s": FieldMeta(
        name="e2e_p95_s",
        unit="seconds",
        kind="percentile",
        semantics="P95 end-to-end request latency.",
        vllm_source="histogram_quantile(0.95, vllm:e2e_request_latency_seconds)",
        sglang_source="histogram_quantile(0.95, sglang:e2e_request_latency_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "e2e_p99_s": FieldMeta(
        name="e2e_p99_s",
        unit="seconds",
        kind="percentile",
        semantics="P99 end-to-end request latency.",
        vllm_source="histogram_quantile(0.99, vllm:e2e_request_latency_seconds)",
        sglang_source="histogram_quantile(0.99, sglang:e2e_request_latency_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "queue_time_p50_s": FieldMeta(
        name="queue_time_p50_s",
        unit="seconds",
        kind="percentile",
        semantics="Median time spent in the waiting/queueing phase.",
        vllm_source="histogram_quantile(0.5, vllm:request_queue_time_seconds)",
        sglang_source="histogram_quantile(0.5, sglang:queue_time_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "queue_time_p95_s": FieldMeta(
        name="queue_time_p95_s",
        unit="seconds",
        kind="percentile",
        semantics="P95 queueing time.",
        vllm_source="histogram_quantile(0.95, vllm:request_queue_time_seconds)",
        sglang_source="histogram_quantile(0.95, sglang:queue_time_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
    "queue_time_p99_s": FieldMeta(
        name="queue_time_p99_s",
        unit="seconds",
        kind="percentile",
        semantics="P99 queueing time.",
        vllm_source="histogram_quantile(0.99, vllm:request_queue_time_seconds)",
        sglang_source="histogram_quantile(0.99, sglang:queue_time_seconds)",
        vllm_default=True,
        sglang_default=False,
        requires="sglang --enable-metrics",
    ),
}

# ------------------------------------------------------------------ preserved
# Engine-specific metrics that must NOT be normalized. Engine-native names and
# units are preserved verbatim. ``default_on``: vLLM metrics are on by default
# (--disable-log-stats turns ALL of them off); SGLang /metrics needs
# --enable-metrics while /v1/loads is always on (latency histograms and TTFT/
# ITL/e2e are metrics-only).
PRESERVED_METRICS: Dict[str, Dict[str, PreservedMetric]] = {
    EngineKind.VLLM.value: {
        "num_preemptions": PreservedMetric(
            name="num_preemptions",
            unit="count",
            semantics="Cumulative request preemptions (counter). SGLang has no "
            "equivalent: SGLang retraction (num_retracted_requests_total) is a "
            "different mechanism and must not be substituted. Prometheus "
            "exposes the counter with a '_total' suffix (live-verified v0.28.0); "
            "the collector falls back to the unsuffixed name only when the "
            "suffixed family is absent.",
            engine_source="vllm:num_preemptions_total",
            default_on=True,
        ),
        "prompt_tokens": PreservedMetric(
            name="prompt_tokens",
            unit="tokens",
            semantics="Cumulative prefill (prompt) tokens computed. Prometheus "
            "counter '_total' suffix; unsuffixed fallback only if absent.",
            engine_source="vllm:prompt_tokens_total",
            default_on=True,
        ),
        "prompt_tokens_cached": PreservedMetric(
            name="prompt_tokens_cached",
            unit="tokens",
            semantics="Cumulative prompt tokens served from prefix cache "
            "(local + external). Prometheus counter '_total' suffix; unsuffixed "
            "fallback only if absent.",
            engine_source="vllm:prompt_tokens_cached_total",
            default_on=True,
        ),
        "generation_tokens": PreservedMetric(
            name="generation_tokens",
            unit="tokens",
            semantics="Cumulative generated tokens (counter). Token rate must be "
            "DERIVED as rate(generation_tokens) over a window; vLLM exposes no "
            "throughput gauge. Prometheus counter '_total' suffix; unsuffixed "
            "fallback only if absent.",
            engine_source="vllm:generation_tokens_total",
            default_on=True,
        ),
        "prefix_cache_queries": PreservedMetric(
            name="prefix_cache_queries",
            unit="tokens",
            semantics="Cumulative prefix-cache queried tokens (counter). "
            "Prometheus counter '_total' suffix; unsuffixed fallback only if "
            "absent.",
            engine_source="vllm:prefix_cache_queries_total",
            default_on=True,
        ),
        "prefix_cache_hits": PreservedMetric(
            name="prefix_cache_hits",
            unit="tokens",
            semantics="Cumulative prefix-cache cached-token hits (counter). "
            "Hit RATE must be derived as rate(hits)/rate(queries); NOT comparable "
            "to SGLang's cache_hit_rate ratio gauge. Prometheus counter '_total' "
            "suffix; unsuffixed fallback only if absent.",
            engine_source="vllm:prefix_cache_hits_total",
            default_on=True,
        ),
        "iteration_tokens_total": PreservedMetric(
            name="iteration_tokens_total",
            unit="tokens",
            semantics="Tokens per engine step (histogram); step-size "
            "distribution, not comparable across engines.",
            engine_source="vllm:iteration_tokens_total",
            default_on=True,
        ),
        "num_requests_waiting_by_reason": PreservedMetric(
            name="num_requests_waiting_by_reason",
            unit="count",
            semantics="Waiting requests by reason (capacity/deferred); vLLM-only "
            "breakdown of requests_waiting.",
            engine_source="vllm:num_requests_waiting_by_reason",
            default_on=True,
        ),
        # ---- M9B execution-stage evidence (engine-specific, NOT normalized) --
        "request_prefill_time": PreservedMetric(
            name="request_prefill_time",
            unit="seconds (p50/p95/p99 over the collection window)",
            semantics="MEASURED request-scoped PREFILL phase interval: first "
            "SCHEDULED -> first NEW_TOKEN (engine-core timestamps). INCLUDES any "
            "preemptions during prefill; not pure compute.",
            engine_source="vllm:request_prefill_time_seconds",
            default_on=True,
        ),
        "request_decode_time": PreservedMetric(
            name="request_decode_time",
            unit="seconds (p50/p95/p99 over the collection window)",
            semantics="MEASURED request-scoped DECODE phase interval: first "
            "NEW_TOKEN -> last NEW_TOKEN (engine-core timestamps). INCLUDES any "
            "preemptions during decode.",
            engine_source="vllm:request_decode_time_seconds",
            default_on=True,
        ),
        "request_inference_time": PreservedMetric(
            name="request_inference_time",
            unit="seconds (p50/p95/p99 over the collection window)",
            semantics="MEASURED request-scoped RUNNING interval: first SCHEDULED "
            "-> last NEW_TOKEN (engine-core timestamps). INCLUDES preemptions.",
            engine_source="vllm:request_inference_time_seconds",
            default_on=True,
        ),
        "inter_token_latency": PreservedMetric(
            name="inter_token_latency",
            unit="seconds (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-engine-STEP inter-token interval "
            "(engine_core_timestamp delta between consecutive decode outputs). "
            "One sample per step (NOT per generated token); distinct from "
            "request_time_per_output_token. Never equated with SGLang's "
            "token-weighted inter_token_latency.",
            engine_source="vllm:inter_token_latency_seconds",
            default_on=True,
        ),
        "request_time_per_output_token": PreservedMetric(
            name="request_time_per_output_token",
            unit="seconds (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-finished-request average: decode_time / "
            "(num_generation_tokens - 1); one sample per completed request.",
            engine_source="vllm:request_time_per_output_token_seconds",
            default_on=True,
        ),
        "request_prompt_tokens": PreservedMetric(
            name="request_prompt_tokens",
            unit="tokens (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-request prompt-token count distribution.",
            engine_source="vllm:request_prompt_tokens",
            default_on=True,
        ),
        "request_generation_tokens": PreservedMetric(
            name="request_generation_tokens",
            unit="tokens (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-request generation-token count distribution.",
            engine_source="vllm:request_generation_tokens",
            default_on=True,
        ),
        "request_success": PreservedMetric(
            name="request_success",
            unit="count",
            semantics="MEASURED cumulative successful requests by finished_reason "
            "label (per-request outcome counter). Live exposition carries the "
            "Prometheus counter '_total' suffix on v0.28.0.",
            engine_source="vllm:request_success_total",
            default_on=True,
        ),
        "prompt_tokens_by_source": PreservedMetric(
            name="prompt_tokens_by_source",
            unit="tokens",
            semantics="MEASURED cumulative prompt tokens by source "
            "(local_compute/local_cache_hit/external_kv_transfer). Live "
            "exposition carries the counter '_total' suffix on v0.28.0.",
            engine_source="vllm:prompt_tokens_by_source_total",
            default_on=True,
        ),
    },
    EngineKind.SGLANG.value: {
        "gen_throughput": PreservedMetric(
            name="gen_throughput",
            unit="tokens/sec",
            semantics="Windowed generation-token rate gauge, recomputed every "
            "decode_log_interval iterations. NOT comparable to a rate derived "
            "from vLLM's generation_tokens counter.",
            engine_source="sglang:gen_throughput | /v1/loads.gen_throughput",
            default_on=True,
        ),
        "cache_hit_rate": PreservedMetric(
            name="cache_hit_rate",
            unit="ratio",
            semantics="SGLang's own prefix-cache hit ratio gauge. Semantics "
            "differ from vLLM's query/hit token counters; never translated.",
            engine_source="sglang:cache_hit_rate | /v1/loads.cache_hit_rate",
            default_on=True,
        ),
        "token_usage": PreservedMetric(
            name="token_usage",
            unit="ratio",
            semantics="Max memory-pool token-usage ratio (full/SWA/Mamba). Feeds "
            "the normalized kv_usage; the per-pool breakdown is preserved here.",
            engine_source="sglang:token_usage | /v1/loads.token_usage",
            default_on=True,
        ),
        "kv_used_tokens": PreservedMetric(
            name="kv_used_tokens",
            unit="tokens",
            semantics="Active KV slots (tokens). vLLM exposes no equivalent "
            "absolute slot count out-of-band.",
            engine_source="sglang:kv_used_tokens",
            default_on=False,
            requires="--enable-metrics",
        ),
        "kv_available_tokens": PreservedMetric(
            name="kv_available_tokens",
            unit="tokens",
            semantics="Free KV slots (tokens).",
            engine_source="sglang:kv_available_tokens",
            default_on=False,
            requires="--enable-metrics",
        ),
        "kv_evictable_tokens": PreservedMetric(
            name="kv_evictable_tokens",
            unit="tokens",
            semantics="Evictable (radix-cache) KV slots (tokens).",
            engine_source="sglang:kv_evictable_tokens",
            default_on=False,
            requires="--enable-metrics",
        ),
        "prompt_tokens_total": PreservedMetric(
            name="prompt_tokens_total",
            unit="tokens",
            semantics="Cumulative prefill tokens (counter), label is_streaming.",
            engine_source="sglang:prompt_tokens_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "generation_tokens_total": PreservedMetric(
            name="generation_tokens_total",
            unit="tokens",
            semantics="Cumulative generation tokens (counter), label "
            "is_streaming.",
            engine_source="sglang:generation_tokens_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "cached_tokens_total": PreservedMetric(
            name="cached_tokens_total",
            unit="tokens",
            semantics="Cumulative cached tokens by cache_source (counter).",
            engine_source="sglang:cached_tokens_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "num_retracted_requests_total": PreservedMetric(
            name="num_retracted_requests_total",
            unit="count",
            semantics="Cumulative retracted requests (counter). NOT vLLM "
            "preemption; SGLang retraction is a pause/flush mechanism.",
            engine_source="sglang:num_retracted_requests_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "utilization": PreservedMetric(
            name="utilization",
            unit="ratio",
            semantics="Scheduler utilization gauge. DOCUMENTED STUCK AT 0 in "
            "v0.5.18 (no setter; regression #22713) — do not use for decisions.",
            engine_source="sglang:utilization",
            default_on=False,
            requires="--enable-metrics",
        ),
        "max_total_num_tokens": PreservedMetric(
            name="max_total_num_tokens",
            unit="tokens",
            semantics="Provisioned KV-pool capacity in tokens (constant).",
            engine_source="sglang:max_total_num_tokens | /v1/loads.max_total_num_tokens",
            default_on=True,
        ),
        "max_running_requests": PreservedMetric(
            name="max_running_requests",
            unit="count",
            semantics="Provisioned concurrency ceiling in requests (constant). "
            "Added in M5B from the always-on /v1/loads surface.",
            engine_source="sglang:max_running_requests | /v1/loads.max_running_requests",
            default_on=True,
        ),
        "kv_cache_memory_usage_gb": PreservedMetric(
            name="kv_cache_memory_usage_gb",
            unit="GB",
            semantics="Provisioned KV-cache memory in GB (constant).",
            engine_source="sglang:kv_cache_memory_usage_gb | /server_info internal_states",
            default_on=True,
        ),
        "weight_memory_usage_gb": PreservedMetric(
            name="weight_memory_usage_gb",
            unit="GB",
            semantics="Model weights memory in GB (constant).",
            engine_source="sglang:weight_memory_usage_gb | /server_info internal_states",
            default_on=True,
        ),
        # ---- M9B execution-stage evidence (engine-specific, NOT normalized) --
        "per_stage_req_latency": PreservedMetric(
            name="per_stage_req_latency",
            unit="seconds (p50/p95/p99 per stage label)",
            semantics="MEASURED per-stage request intervals labelled by "
            "``stage``. In the supported unified envelope the prefill stage is "
            "``prefill_forward`` (first forward entry -> prefill finished; for "
            "chunked prefill it spans all chunks plus inter-chunk gaps) with "
            "``chunked_prefill`` and ``request_process``. There is NO decode "
            "stage emitted to this histogram (decode_loop is trace-only).",
            engine_source="sglang:per_stage_req_latency_seconds",
            default_on=False,
            requires="--enable-metrics",
        ),
        "inter_token_latency": PreservedMetric(
            name="inter_token_latency",
            unit="seconds (p50/p95/p99 over the collection window)",
            semantics="MEASURED TOKEN-WEIGHTED per-output-token interval at the "
            "tokenizer output layer (bucket weighted by num_new_tokens). This is "
            "a per-token average (TPOT-like), NOT a per-engine-step ITL, and is "
            "not comparable to vLLM's inter_token_latency.",
            engine_source="sglang:inter_token_latency_seconds",
            default_on=False,
            requires="--enable-metrics",
        ),
        "request_prompt_tokens": PreservedMetric(
            name="request_prompt_tokens",
            unit="tokens (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-request prompt-token count distribution.",
            engine_source="sglang:prompt_tokens_histogram",
            default_on=False,
            requires="--enable-metrics",
        ),
        "request_uncached_prompt_tokens": PreservedMetric(
            name="request_uncached_prompt_tokens",
            unit="tokens (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-request uncached prompt-token distribution.",
            engine_source="sglang:uncached_prompt_tokens_histogram",
            default_on=False,
            requires="--enable-metrics",
        ),
        "request_generation_tokens": PreservedMetric(
            name="request_generation_tokens",
            unit="tokens (p50/p95/p99 over the collection window)",
            semantics="MEASURED per-request generation-token count distribution.",
            engine_source="sglang:generation_tokens_histogram",
            default_on=False,
            requires="--enable-metrics",
        ),
        "num_requests_total": PreservedMetric(
            name="num_requests_total",
            unit="count",
            semantics="MEASURED cumulative finished requests (label is_streaming).",
            engine_source="sglang:num_requests_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "num_aborted_requests_total": PreservedMetric(
            name="num_aborted_requests_total",
            unit="count",
            semantics="MEASURED cumulative aborted requests.",
            engine_source="sglang:num_aborted_requests_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "forward_execution_seconds_total": PreservedMetric(
            name="forward_execution_seconds_total",
            unit="seconds",
            semantics="MEASURED cumulative forward-execution time (counter).",
            engine_source="sglang:forward_execution_seconds_total",
            default_on=False,
            requires="--enable-metrics",
        ),
        "fwd_occupancy": PreservedMetric(
            name="fwd_occupancy",
            unit="ratio",
            semantics="MEASURED forward occupancy gauge (scheduler batch "
            "utilisation evidence).",
            engine_source="sglang:fwd_occupancy",
            default_on=False,
            requires="--enable-metrics",
        ),
        "decode_sum_seq_lens": PreservedMetric(
            name="decode_sum_seq_lens",
            unit="tokens",
            semantics="MEASURED sum of running sequence lengths during decode "
            "(engine-level batch-size proxy).",
            engine_source="sglang:decode_sum_seq_lens",
            default_on=False,
            requires="--enable-metrics",
        ),
        "num_used_tokens": PreservedMetric(
            name="num_used_tokens",
            unit="tokens",
            semantics="MEASURED instantaneous KV tokens in use (/v1/loads).",
            engine_source="sglang:num_used_tokens | /v1/loads.num_used_tokens",
            default_on=True,
        ),
        "num_total_tokens": PreservedMetric(
            name="num_total_tokens",
            unit="tokens",
            semantics="MEASURED instantaneous total KV tokens (/v1/loads).",
            engine_source="sglang:num_total_tokens | /v1/loads.num_total_tokens",
            default_on=True,
        ),
        "num_active_tokens": PreservedMetric(
            name="num_active_tokens",
            unit="tokens",
            semantics="MEASURED active KV tokens (total minus tokens awaiting KV "
            "transfer) (/v1/loads).",
            engine_source="sglang:num_active_tokens | /v1/loads.num_active_tokens",
            default_on=True,
        ),
        "num_waiting_uncached_tokens": PreservedMetric(
            name="num_waiting_uncached_tokens",
            unit="tokens",
            semantics="MEASURED waiting-queue uncached tokens (/v1/loads).",
            engine_source=(
                "sglang:num_waiting_uncached_tokens | "
                "/v1/loads.num_waiting_uncached_tokens"
            ),
            default_on=True,
        ),
        "total_prefill_uncached_tokens": PreservedMetric(
            name="total_prefill_uncached_tokens",
            unit="tokens",
            semantics="MEASURED CUMULATIVE prefill uncached tokens (monotonic "
            "engine-level counter, /v1/loads; no reset path).",
            engine_source=(
                "sglang:total_prefill_uncached_tokens | "
                "/v1/loads.total_prefill_uncached_tokens"
            ),
            default_on=True,
        ),
        "total_prefill_busy_us": PreservedMetric(
            name="total_prefill_busy_us",
            unit="microseconds",
            semantics="MEASURED CUMULATIVE prefill busy time (monotonic "
            "engine-level counter, /v1/loads; no reset path).",
            engine_source=(
                "sglang:total_prefill_busy_us | /v1/loads.total_prefill_busy_us"
            ),
            default_on=True,
        ),
    },
}

# Valid quality keys: normalized fields plus the documented preserved metrics
# for the engine (engine_specific may carry extra raw keys, which also stay
# addressable for quality).
def _quality_keys_for(engine: str) -> Tuple[str, ...]:
    return tuple(NORMALIZED_FIELDS) + tuple(PRESERVED_METRICS.get(engine, {}))


# ------------------------------------------------------------------- snapshot
@dataclass(frozen=True)
class RuntimeSnapshot:
    """Initial normalized, engine-neutral telemetry snapshot.

    All normalized fields are OPTIONAL with a defined meaning: the value is
    present only when the collector observed it; ``quality`` records WHY a
    field has no value or is stale (see :class:`MetricQuality`). A missing
    latency quantile therefore means "disabled / no samples / scrape failed",
    NOT zero latency.

    ``quality`` maps a normalized field (or a preserved engine-specific metric
    key) to one of the five :class:`MetricQuality` states. A present value with
    no quality entry defaults to ``fresh``; an absent value with no quality
    entry is unspecified (a conforming collector SHOULD always state why).

    ``engine_specific`` preserves engine-native metrics verbatim (see
    :data:`PRESERVED_METRICS`). A value is normalized ONLY when semantics and
    units genuinely match across engines; everything else stays preserved.
    """

    engine: str
    instance_id: str
    collected_at: str
    source: str
    schema_version: str = SCHEMA_VERSION

    requests_running: Optional[int] = None
    requests_waiting: Optional[int] = None
    kv_usage: Optional[float] = None

    ttft_p50_s: Optional[float] = None
    ttft_p95_s: Optional[float] = None
    ttft_p99_s: Optional[float] = None
    tpot_p50_s: Optional[float] = None
    tpot_p95_s: Optional[float] = None
    tpot_p99_s: Optional[float] = None
    e2e_p50_s: Optional[float] = None
    e2e_p95_s: Optional[float] = None
    e2e_p99_s: Optional[float] = None
    queue_time_p50_s: Optional[float] = None
    queue_time_p95_s: Optional[float] = None
    queue_time_p99_s: Optional[float] = None

    engine_specific: Dict[str, Any] = field(default_factory=dict)
    quality: Dict[str, str] = field(default_factory=dict)
    # M9B additive sidecar: per-family aggregation metadata for the read-only
    # execution-evidence contract. Shape:
    #   {"window_s": float, "staleness_s": float,
    #    "families": {<family_key>: {"samples": int, "window_s": float}}}
    # It is NOT an engine metric and is never normalized; empty when the
    # collector did not record any windowed family (e.g. a failed scrape).
    measurement_meta: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- accessors
    def available(self) -> Tuple[str, ...]:
        """Normalized field names present in this snapshot (non-None)."""
        return tuple(f for f in NORMALIZED_FIELDS if getattr(self, f) is not None)

    def quality_of(self, field_name: str) -> Optional[MetricQuality]:
        """Explicit quality state for a field.

        Returns the recorded state, else ``fresh`` for a present value, else
        ``None`` when the collector did not state why the value is absent.
        """
        if field_name in self.quality:
            return MetricQuality(self.quality[field_name])
        value = (
            getattr(self, field_name, None)
            if field_name in NORMALIZED_FIELDS
            else self.engine_specific.get(field_name)
        )
        if value is not None:
            return MetricQuality.FRESH
        return None

    def normalized(self) -> Dict[str, Any]:
        """Only the normalized fields (no metadata, no engine_specific)."""
        return {f: getattr(self, f) for f in NORMALIZED_FIELDS}

    # ------------------------------------------------------------ serialization
    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "engine": self.engine,
            "instance_id": self.instance_id,
            "collected_at": self.collected_at,
            "source": self.source,
            "schema_version": self.schema_version,
            "normalized": self.normalized(),
            "engine_specific": dict(self.engine_specific),
            "quality": dict(self.quality),
            "measurement_meta": copy.deepcopy(self.measurement_meta),
        }
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeSnapshot":
        normalized = data.get("normalized", {})
        kwargs: Dict[str, Any] = dict(data)
        kwargs.pop("normalized", None)
        kwargs.pop("engine_specific", None)
        kwargs.pop("quality", None)
        kwargs.update({k: v for k, v in normalized.items() if k in NORMALIZED_FIELDS})
        kwargs["engine_specific"] = dict(data.get("engine_specific", {}))
        kwargs["quality"] = dict(data.get("quality", {}))
        kwargs["measurement_meta"] = dict(data.get("measurement_meta", {}))
        return cls(**kwargs)


# ------------------------------------------------------------------ validation
def _is_finite(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _check_quality(snapshot: RuntimeSnapshot) -> None:
    valid_keys = set(_quality_keys_for(snapshot.engine))
    for key, state in snapshot.quality.items():
        if key not in valid_keys:
            raise SnapshotValidationError(
                f"quality key {key!r} is not a normalized field or a documented "
                f"preserved metric for {snapshot.engine!r}"
            )
        try:
            q = MetricQuality(state)
        except ValueError as exc:
            raise SnapshotValidationError(
                f"quality[{key!r}] = {state!r} is not a valid MetricQuality"
            ) from exc
        value = (
            getattr(snapshot, key, None)
            if key in NORMALIZED_FIELDS
            else snapshot.engine_specific.get(key)
        )
        if q in (MetricQuality.FRESH, MetricQuality.STALE):
            if value is None:
                raise SnapshotValidationError(
                    f"quality[{key!r}] = {state!r} requires a present value"
                )
        else:
            if value is not None:
                raise SnapshotValidationError(
                    f"quality[{key!r}] = {state!r} requires the value to be absent"
                )


def validate_snapshot(snapshot: RuntimeSnapshot) -> None:
    """Validate the normalized contract invariants; raise on violation.

    Checks: engine kind, required metadata, field ranges (counts >= 0,
    kv_usage in [0.0, 1.0], latency quantiles >= 0 and finite), monotone
    percentile ordering per latency family (p50 <= p95 <= p99), and the
    quality/availability invariants (fresh/stale require a value;
    disabled/no_samples/scrape_failed require absence). ``None`` is valid for
    values (availability is explained by ``quality``).
    """
    if snapshot.engine not in (EngineKind.VLLM.value, EngineKind.SGLANG.value):
        raise SnapshotValidationError(
            f"engine must be 'vllm' or 'sglang', got {snapshot.engine!r}"
        )
    if not snapshot.instance_id:
        raise SnapshotValidationError("instance_id must be non-empty")
    if not snapshot.collected_at:
        raise SnapshotValidationError("collected_at must be non-empty")
    try:
        datetime.fromisoformat(snapshot.collected_at)
    except ValueError as exc:
        raise SnapshotValidationError(
            f"collected_at must be ISO-8601, got {snapshot.collected_at!r}"
        ) from exc
    if not snapshot.source:
        raise SnapshotValidationError("source must be non-empty")

    for f in GAUGE_FIELDS:
        value = getattr(snapshot, f)
        if value is None:
            continue
        if not _is_finite(value):
            raise SnapshotValidationError(f"{f} must be a finite number")
        if f in ("requests_running", "requests_waiting"):
            if int(value) != value or value < 0:
                raise SnapshotValidationError(f"{f} must be a non-negative integer")
        elif not 0.0 <= value <= 1.0:
            raise SnapshotValidationError(f"{f} must be in [0.0, 1.0]")

    for family, (p50, p95, p99) in LATENCY_FAMILIES.items():
        vals = [
            (p50, getattr(snapshot, p50)),
            (p95, getattr(snapshot, p95)),
            (p99, getattr(snapshot, p99)),
        ]
        for name, value in vals:
            if value is None:
                continue
            if not _is_finite(value) or value < 0:
                raise SnapshotValidationError(
                    f"{name} must be a non-negative finite number"
                )
        ordered = [value for _, value in vals if value is not None]
        if ordered != sorted(ordered):
            raise SnapshotValidationError(
                f"{family} quantiles must satisfy p50 <= p95 <= p99"
            )

    for key, value in snapshot.engine_specific.items():
        if (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and not math.isfinite(value)
        ):
            raise SnapshotValidationError(
                f"engine_specific[{key!r}] contains a non-finite number"
            )

    if not isinstance(snapshot.measurement_meta, dict):
        raise SnapshotValidationError("measurement_meta must be a dict")
    families = snapshot.measurement_meta.get("families")
    if families is not None and not isinstance(families, dict):
        raise SnapshotValidationError("measurement_meta['families'] must be a dict")

    _check_quality(snapshot)