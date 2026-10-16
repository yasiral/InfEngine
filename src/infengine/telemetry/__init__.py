"""Out-of-band telemetry for the engine control plane.

M5A scope: the initial normalized ``RuntimeSnapshot`` contract plus the
documented metric semantics/units/availability for vLLM v0.28.0 and SGLang
v0.5.18. M5B adds the source-specific collectors that implement the contract
out-of-band.

The field set is an extensible initial subset, not a frozen universe; extension
rules are documented in ``snapshot``.
"""

from .snapshot import (
    FIELD_META,
    GAUGE_FIELDS,
    LATENCY_FAMILIES,
    LATENCY_FIELDS,
    LEGACY_ENGINE_DEPENDENT_FIELDS,
    NORMALIZED_FIELDS,
    PRESERVED_METRICS,
    SCHEMA_VERSION,
    VERIFIED_LIVE_SGLANG_EXPOSITION,
    VERIFIED_LIVE_SGLANG_METRICS_ENABLED,
    EngineKind,
    FieldMeta,
    MetricQuality,
    PreservedMetric,
    RuntimeSnapshot,
    SnapshotValidationError,
    exposition_name,
    native_metric,
    validate_snapshot,
)
from .exposition import (
    Exposition,
    ExpositionParseError,
    Family,
    HistogramSeries,
    Sample,
    histogram_quantile,
    parse_exposition,
)
from .window import RollingHistograms
from .collectors import (
    SglangTelemetryCollector,
    TelemetryCollector,
    TelemetryError,
    VllmTelemetryCollector,
)
from .gpu import GpuSnapshot, GpuTelemetryCollector
from .execution import (
    SUPPORTED_ENGINE_VERSIONS,
    ExecutionEvidence,
    Measurement,
    MeasurementClassification,
    MeasurementScope,
    build_execution_evidence,
)

__all__ = [
    "EngineKind",
    "FieldMeta",
    "MetricQuality",
    "PreservedMetric",
    "RuntimeSnapshot",
    "SnapshotValidationError",
    "validate_snapshot",
    "exposition_name",
    "native_metric",
    "SCHEMA_VERSION",
    "LEGACY_ENGINE_DEPENDENT_FIELDS",
    "ExecutionEvidence",
    "Measurement",
    "MeasurementClassification",
    "MeasurementScope",
    "SUPPORTED_ENGINE_VERSIONS",
    "build_execution_evidence",
    "NORMALIZED_FIELDS",
    "GAUGE_FIELDS",
    "LATENCY_FIELDS",
    "LATENCY_FAMILIES",
    "FIELD_META",
    "PRESERVED_METRICS",
    "VERIFIED_LIVE_SGLANG_EXPOSITION",
    "VERIFIED_LIVE_SGLANG_METRICS_ENABLED",
    "Exposition",
    "ExpositionParseError",
    "Family",
    "HistogramSeries",
    "Sample",
    "histogram_quantile",
    "parse_exposition",
    "RollingHistograms",
    "TelemetryCollector",
    "TelemetryError",
    "VllmTelemetryCollector",
    "SglangTelemetryCollector",
    "GpuSnapshot",
    "GpuTelemetryCollector",
]