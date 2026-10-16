"""M9B: read-only execution-stage evidence contract.

Assembles the M9A-frozen request-stage and engine-level measurements from an
existing :class:`~infengine.telemetry.snapshot.RuntimeSnapshot` (plus the
additive ``measurement_meta`` sidecar) into a small, explicitly classified
representation reusable by the experiment runner and the future saturation
sweep.

Design guarantees (M9A/M9B):

  * **No second collection path.** Everything is read from a snapshot plus the
    contract's own ``FIELD_META`` / ``PRESERVED_METRICS`` source tables and a
    static scope/classification/boundary registry defined here.
  * **Unavailable stays UNAVAILABLE** (never coerced to zero).
  * **No cross-engine conflation.** ``inter_token_latency`` (vLLM per-step /
    SGLang token-weighted) and ``time_per_output_token`` (vLLM per-request) are
    distinct, correctly named measurements; the legacy normalized ``tpot_*``
    fields are surfaced only as an explicitly engine-dependent legacy family.
  * **No inferred phase durations.** The builder never computes prefill as
    ``TTFT - queue_time`` or decode as ``TPOT * mean_output_tokens``.
  * Request-scoped items are aggregated histograms (quantiles over a window),
    never per-request traces.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from .snapshot import (
    LATENCY_FAMILIES,
    LEGACY_ENGINE_DEPENDENT_FIELDS,
    EngineKind,
    RuntimeSnapshot,
    native_metric,
)

# Pinned engine versions the contract was source-verified against (M9A).
SUPPORTED_ENGINE_VERSIONS: Dict[str, str] = {
    EngineKind.VLLM.value: "v0.28.0",
    EngineKind.SGLANG.value: "v0.5.18",
}

_WINDOW_FALLBACK_S = 30.0


class MeasurementClassification(str, Enum):
    """How a measurement came to exist.

    ``MEASURED``          -- a native engine observation (gauge/counter/histogram).
    ``DERIVED_ESTIMATE``  -- an explicit documented formula over measured inputs.
    ``UNAVAILABLE``       -- not emitted by the engine on the supported version
                             or not enabled in this configuration.
    """

    MEASURED = "MEASURED"
    DERIVED_ESTIMATE = "DERIVED_ESTIMATE"
    UNAVAILABLE = "UNAVAILABLE"


class MeasurementScope(str, Enum):
    REQUEST_SCOPED_AGGREGATED = "request_scoped_aggregated"
    ENGINE_LEVEL = "engine_level"


@dataclass(frozen=True)
class Measurement:
    """One classified execution-stage measurement.

    A present ``value`` (scalar) or non-empty ``quantiles`` (histogram family)
    with a fresh/stale ``quality`` is MEASURED/DERIVED_ESTIMATE; otherwise the
    classification is downgraded to ``UNAVAILABLE`` and no value is carried.
    ``sample_count`` is the number of observations in the aggregation window
    (``None`` when the family is not windowed).
    """

    key: str
    engine: str
    source: str
    engine_version: str
    scope: str
    unit: str
    classification: str
    boundary_note: str
    window_s: Optional[float] = None
    sample_count: Optional[int] = None
    quality: Optional[str] = None
    value: Optional[float] = None
    quantiles: Dict[str, Optional[float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "engine": self.engine,
            "source": self.source,
            "engine_version": self.engine_version,
            "scope": self.scope,
            "unit": self.unit,
            "classification": self.classification,
            "boundary_note": self.boundary_note,
            "window_s": self.window_s,
            "sample_count": self.sample_count,
            "quality": self.quality,
            "value": self.value,
            "quantiles": dict(self.quantiles),
        }


@dataclass(frozen=True)
class ExecutionEvidence:
    """Read-only execution-stage evidence for one engine instance."""

    engine: str
    instance_id: str
    collected_at: str
    engine_version: str
    window_s: Optional[float]
    request_stage: Dict[str, Measurement] = field(default_factory=dict)
    engine_level: Dict[str, Measurement] = field(default_factory=dict)

    def get(self, key: str) -> Optional[Measurement]:
        return self.request_stage.get(key) or self.engine_level.get(key)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "engine": self.engine,
            "instance_id": self.instance_id,
            "collected_at": self.collected_at,
            "engine_version": self.engine_version,
            "window_s": self.window_s,
            "request_stage": {k: v.to_dict() for k, v in self.request_stage.items()},
            "engine_level": {k: v.to_dict() for k, v in self.engine_level.items()},
        }


# --------------------------------------------------------------------- registry
@dataclass(frozen=True)
class _Spec:
    key: str
    scope: str
    unit: str
    base: str  # MeasurementClassification.MEASURED / DERIVED_ESTIMATE
    kind: str  # normalized_family | preserved_quantile | preserved_scalar |
    #            preserved_label_counter | stage | unavailable
    native: str
    boundary: str
    family_key: str = ""
    preserved_key: str = ""
    stage: str = ""


def _fam(family_key: str, source: str, boundary: str) -> _Spec:
    return _Spec(
        key=family_key,
        scope=MeasurementScope.REQUEST_SCOPED_AGGREGATED.value,
        unit="seconds",
        base=MeasurementClassification.MEASURED.value,
        kind="normalized_family",
        native=source,
        boundary=boundary,
        family_key=family_key,
    )


def _preserved(
    key: str, unit: str, source: str, boundary: str,
    preserved_key: Optional[str] = None,
) -> _Spec:
    return _Spec(
        key=key,
        scope=MeasurementScope.REQUEST_SCOPED_AGGREGATED.value,
        unit=unit,
        base=MeasurementClassification.MEASURED.value,
        kind="preserved_quantile",
        native=source,
        boundary=boundary,
        preserved_key=preserved_key or key,
    )


def _scalar(
    key: str, unit: str, source: str, boundary: str, base: str
) -> _Spec:
    return _Spec(
        key=key,
        scope=MeasurementScope.ENGINE_LEVEL.value,
        unit=unit,
        base=base,
        kind="preserved_scalar",
        native=source,
        boundary=boundary,
        preserved_key=key,
    )


def _gauge(key: str, unit: str, source: str, boundary: str) -> _Spec:
    return _Spec(
        key=key,
        scope=MeasurementScope.ENGINE_LEVEL.value,
        unit=unit,
        base=MeasurementClassification.MEASURED.value,
        kind="normalized_gauge",
        native=source,
        boundary=boundary,
    )


def _unavailable(key: str, unit: str, source: str, boundary: str) -> _Spec:
    return _Spec(
        key=key,
        scope=MeasurementScope.REQUEST_SCOPED_AGGREGATED.value,
        unit=unit,
        base=MeasurementClassification.UNAVAILABLE.value,
        kind="unavailable",
        native=source,
        boundary=boundary,
    )


_VLLM_REGISTRY: Tuple[_Spec, ...] = (
    # request-stage (normalized families; sources from FIELD_META)
    _fam("ttft", "vllm:time_to_first_token_seconds",
         "arrival -> first token (frontend wall-clock)."),
    _fam("queue_time", "vllm:request_queue_time_seconds",
         "first QUEUED -> first SCHEDULED (engine-core clock)."),
    _fam("e2e", "vllm:e2e_request_latency_seconds",
         "arrival -> finished (frontend wall-clock)."),
    _fam("tpot", "vllm:inter_token_latency_seconds",
         "LEGACY engine-dependent alias; see inter_token_latency."),
    # correctly named engine-specific request-stage families
    _preserved("inter_token_latency", "seconds", "vllm:inter_token_latency_seconds",
               "per engine step: consecutive decode output iteration delta; "
               "one sample per step (not per token)."),
    _preserved("time_per_output_token", "seconds",
               "vllm:request_time_per_output_token_seconds",
               "per finished request: decode_time / (output_tokens - 1).",
               preserved_key="request_time_per_output_token"),
    _preserved("prefill_time", "seconds", "vllm:request_prefill_time_seconds",
               "first SCHEDULED -> first NEW_TOKEN; includes prefill preemptions.",
               preserved_key="request_prefill_time"),
    _preserved("decode_time", "seconds", "vllm:request_decode_time_seconds",
               "first NEW_TOKEN -> last NEW_TOKEN; includes decode preemptions.",
               preserved_key="request_decode_time"),
    _preserved("inference_time", "seconds", "vllm:request_inference_time_seconds",
               "first SCHEDULED -> last NEW_TOKEN.",
               preserved_key="request_inference_time"),
    _preserved("request_prompt_tokens", "tokens", "vllm:request_prompt_tokens",
               "per-request prompt-token count distribution."),
    _preserved("request_generation_tokens", "tokens",
               "vllm:request_generation_tokens",
               "per-request generation-token count distribution."),
    # engine-level
    _scalar("num_preemptions", "count", "vllm:num_preemptions",
            "cumulative preemptions (counter).",
            MeasurementClassification.MEASURED.value),
    _scalar("iteration_tokens_p95", "tokens/step", "vllm:iteration_tokens_total",
            "windowed p95 tokens per scheduling step (batch-size proxy).",
            MeasurementClassification.DERIVED_ESTIMATE.value),
    _scalar("request_success", "count", "vllm:request_success",
            "cumulative successful requests by finished_reason.",
            MeasurementClassification.MEASURED.value),
)

_SGLANG_REGISTRY: Tuple[_Spec, ...] = (
    _fam("ttft", "sglang:time_to_first_token_seconds",
         "arrival -> first token (requires --enable-metrics)."),
    _fam("queue_time", "sglang:queue_time_seconds",
         "queueing phase (requires --enable-metrics)."),
    _fam("e2e", "sglang:e2e_request_latency_seconds",
         "arrival -> finished (requires --enable-metrics)."),
    _fam("tpot", "sglang:inter_token_latency_seconds",
         "LEGACY engine-dependent alias; SGLang's native family is "
         "token-weighted (see inter_token_latency)."),
    _preserved("inter_token_latency", "seconds", "sglang:inter_token_latency_seconds",
               "token-weighted per-output-token interval at the tokenizer output "
               "layer (TPOT-like), not a per-engine-step ITL."),
    _Spec(
        key="prefill_time",
        scope=MeasurementScope.REQUEST_SCOPED_AGGREGATED.value,
        unit="seconds",
        base=MeasurementClassification.MEASURED.value,
        kind="stage",
        native="sglang:per_stage_req_latency_seconds{stage=prefill_forward}",
        boundary="first forward entry -> prefill finished (includes chunked "
        "prefill inter-chunk gaps).",
        preserved_key="per_stage_req_latency",
        stage="prefill_forward",
    ),
    _unavailable("decode_time", "seconds", "",
                 "SGLang v0.5.18 emits no request-scoped decode-stage histogram "
                 "(decode_loop is trace-only); not synthesized."),
    _unavailable("time_per_output_token", "seconds", "",
                 "SGLang has no request-scoped time-per-output-token histogram; "
                 "use the token-weighted inter_token_latency as a distinct family."),
    _unavailable("inference_time", "seconds", "",
                 "No request-scoped running/inference interval histogram."),
    _preserved("request_prompt_tokens", "tokens", "sglang:prompt_tokens_histogram",
               "per-request prompt-token distribution (--enable-metrics)."),
    _preserved("request_generation_tokens", "tokens",
               "sglang:generation_tokens_histogram",
               "per-request generation-token distribution (--enable-metrics)."),
    # engine-level
    _scalar("num_retracted_requests_total", "count",
            "sglang:num_retracted_requests_total",
            "cumulative retracted requests (NOT preemption).",
            MeasurementClassification.MEASURED.value),
    _scalar("gen_throughput", "tokens/sec", "sglang:gen_throughput",
            "windowed generation-token rate gauge.",
            MeasurementClassification.MEASURED.value),
    _scalar("decode_sum_seq_lens", "tokens", "sglang:decode_sum_seq_lens",
            "engine-level decode batch-size proxy.",
            MeasurementClassification.MEASURED.value),
    _scalar("fwd_occupancy", "ratio", "sglang:fwd_occupancy",
            "forward occupancy gauge.",
            MeasurementClassification.MEASURED.value),
    _scalar("total_prefill_busy_us", "microseconds",
            "sglang:total_prefill_busy_us | /v1/loads.total_prefill_busy_us",
            "cumulative prefill busy time (monotonic).",
            MeasurementClassification.MEASURED.value),
    _scalar("total_prefill_uncached_tokens", "tokens",
            "sglang:total_prefill_uncached_tokens | "
            "/v1/loads.total_prefill_uncached_tokens",
            "cumulative prefill uncached tokens (monotonic).",
            MeasurementClassification.MEASURED.value),
    _scalar("num_requests_total", "count", "sglang:num_requests_total",
            "cumulative finished requests.",
            MeasurementClassification.MEASURED.value),
    _scalar("num_aborted_requests_total", "count",
            "sglang:num_aborted_requests_total",
            "cumulative aborted requests.",
            MeasurementClassification.MEASURED.value),
)

_REGISTRY: Dict[str, Tuple[_Spec, ...]] = {
    EngineKind.VLLM.value: _VLLM_REGISTRY,
    EngineKind.SGLANG.value: _SGLANG_REGISTRY,
}

# Engine-level normalized gauges shared by both engines.
_GAUGE_SPECS: Tuple[_Spec, ...] = (
    _gauge("requests_running", "count", "num_requests_running",
           "running requests (normalized gauge)."),
    _gauge("requests_waiting", "count", "num_requests_waiting",
           "waiting requests (normalized gauge)."),
    _gauge("kv_usage", "ratio", "kv_cache_usage_perc / token_usage",
           "KV pool utilization (normalized gauge)."),
)


def _native_source(spec: _Spec, engine: str) -> str:
    if spec.kind == "normalized_family":
        from .snapshot import FIELD_META

        field = LATENCY_FAMILIES[spec.family_key][0]
        meta = FIELD_META[field]
        source = meta.vllm_source if engine == EngineKind.VLLM.value else meta.sglang_source
        return native_metric(source)
    return spec.native


def _family_meta(snapshot: RuntimeSnapshot, key: str) -> Tuple[Optional[float], Optional[int]]:
    meta = snapshot.measurement_meta or {}
    families = meta.get("families") or {}
    entry = families.get(key)
    if not isinstance(entry, dict):
        return meta.get("window_s"), None
    samples = entry.get("samples")
    window = entry.get("window_s", meta.get("window_s"))
    return window, (int(samples) if isinstance(samples, (int, float)) else None)


def _measurement(
    spec: _Spec,
    snapshot: RuntimeSnapshot,
    engine_version: str,
) -> Measurement:
    engine = snapshot.engine
    source = _native_source(spec, engine)
    window_s, samples = _family_meta(snapshot, spec.preserved_key or spec.family_key)
    quality: Optional[str] = None
    value: Optional[float] = None
    quantiles: Dict[str, Optional[float]] = {}
    classification = spec.base

    if spec.kind == "unavailable":
        return Measurement(
            key=spec.key, engine=engine, source=spec.native,
            engine_version=engine_version, scope=spec.scope, unit=spec.unit,
            classification=MeasurementClassification.UNAVAILABLE.value,
            boundary_note=spec.boundary,
        )

    if spec.kind == "normalized_family":
        for name, field_name in zip(("p50", "p95", "p99"), LATENCY_FAMILIES[spec.family_key]):
            quantiles[name] = getattr(snapshot, field_name, None)
        present = [v for v in quantiles.values() if v is not None]
        if not present:
            classification = MeasurementClassification.UNAVAILABLE.value
        # quality: prefer the recorded state of the p50 field
        q = snapshot.quality.get(LATENCY_FAMILIES[spec.family_key][0])
        quality = q
        if q is None and present:
            quality = "fresh"
    elif spec.kind == "preserved_quantile":
        data = snapshot.engine_specific.get(spec.preserved_key)
        if isinstance(data, dict):
            for name in ("p50", "p95", "p99"):
                q = data.get(name)
                quantiles[name] = q if isinstance(q, (int, float)) else None
            present = [v for v in quantiles.values() if v is not None]
            if not present:
                classification = MeasurementClassification.UNAVAILABLE.value
        else:
            classification = MeasurementClassification.UNAVAILABLE.value
        quality = snapshot.quality.get(spec.preserved_key)
        if quality is None and classification != MeasurementClassification.UNAVAILABLE.value:
            quality = "fresh"
    elif spec.kind == "stage":
        stages = snapshot.engine_specific.get(spec.preserved_key) or {}
        data = stages.get(spec.stage) if isinstance(stages, dict) else None
        if isinstance(data, dict):
            for name in ("p50", "p95", "p99"):
                q = data.get(name)
                quantiles[name] = q if isinstance(q, (int, float)) else None
            present = [v for v in quantiles.values() if v is not None]
            if not present:
                classification = MeasurementClassification.UNAVAILABLE.value
        else:
            classification = MeasurementClassification.UNAVAILABLE.value
        quality = snapshot.quality.get(spec.preserved_key)
        if quality is None and classification != MeasurementClassification.UNAVAILABLE.value:
            quality = "fresh"
        window_s, samples = _family_meta(snapshot, f"{spec.preserved_key}:{spec.stage}")
    elif spec.kind == "normalized_gauge":
        raw = getattr(snapshot, spec.key, None)
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = float(raw)
        else:
            classification = MeasurementClassification.UNAVAILABLE.value
        quality = snapshot.quality.get(spec.key)
        if quality is None and value is not None:
            quality = "fresh"
    elif spec.kind in ("preserved_scalar", "preserved_label_counter"):
        raw = snapshot.engine_specific.get(spec.key)
        if isinstance(raw, dict):
            # A multi-label counter is preserved as {label: cumulative}; the
            # evidence value is the sum of the finite numeric leaves.
            nums = [
                v for v in raw.values()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ]
            value = float(sum(nums)) if nums else None
        elif isinstance(raw, (int, float)) and not isinstance(raw, bool):
            value = float(raw)
        if value is None:
            classification = MeasurementClassification.UNAVAILABLE.value
        quality = snapshot.quality.get(spec.key)
        if quality is None and value is not None:
            quality = "fresh"

    if classification == MeasurementClassification.UNAVAILABLE.value:
        value = None
        quantiles = {}

    return Measurement(
        key=spec.key,
        engine=engine,
        source=source,
        engine_version=engine_version,
        scope=spec.scope,
        unit=spec.unit,
        classification=classification,
        boundary_note=spec.boundary,
        window_s=window_s,
        sample_count=samples,
        quality=quality,
        value=value,
        quantiles=quantiles,
    )


def build_execution_evidence(
    snapshot: RuntimeSnapshot,
    *,
    engine_version: Optional[str] = None,
) -> ExecutionEvidence:
    """Assemble classified execution-stage evidence from a snapshot.

    No I/O, no mutation, deterministic. Unavailable measurements are included
    with ``classification == UNAVAILABLE`` and no value.
    """
    engine = snapshot.engine
    if engine not in _REGISTRY:
        raise ValueError(f"unsupported engine for execution evidence: {engine!r}")
    version = engine_version or SUPPORTED_ENGINE_VERSIONS.get(engine, "unknown")

    request_stage: Dict[str, Measurement] = {}
    for spec in _REGISTRY[engine]:
        request_stage[spec.key] = _measurement(spec, snapshot, version)

    engine_level: Dict[str, Measurement] = {}
    for spec in _GAUGE_SPECS + tuple(
        s for s in _REGISTRY[engine] if s.scope == MeasurementScope.ENGINE_LEVEL.value
    ):
        engine_level[spec.key] = _measurement(spec, snapshot, version)

    meta = snapshot.measurement_meta or {}
    return ExecutionEvidence(
        engine=engine,
        instance_id=snapshot.instance_id,
        collected_at=snapshot.collected_at,
        engine_version=version,
        window_s=meta.get("window_s"),
        request_stage=request_stage,
        engine_level=engine_level,
    )


# The legacy fields are surfaced (unchanged values) but flagged engine-dependent.
LEGACY_FIELDS = LEGACY_ENGINE_DEPENDENT_FIELDS
