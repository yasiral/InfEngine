"""Allowlist + typed validation for vLLM runtime scheduler-config control.

This is the control-plane-side mirror of the tiny allowlist the vLLM
EngineCore hook enforces (see ``contrib/vllm/v0.28.0_hook.patch``). The
adapter validates at the control-plane boundary (``Capability``/``coerce``);
the vLLM hook re-validates for safety. Both sides must keep the spec identical.

Scope (Milestone 3B): a single HOT parameter.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FieldSpec:
    dtype: type
    min_value: Optional[int] = None
    max_value: Optional[int] = None


RUNTIME_SCHEDULER_CONFIG_SPEC: Dict[str, FieldSpec] = {
    "long_prefill_token_threshold": FieldSpec(int, min_value=0),
}


class RuntimeSchedulerConfigError(ValueError):
    """An invalid runtime scheduler-config override was rejected."""


def validate_override(field: str, value: Any) -> Any:
    """Validate a single override value against the allowlist.

    Raises :class:`RuntimeSchedulerConfigError`; returns the value unchanged.
    """
    spec = RUNTIME_SCHEDULER_CONFIG_SPEC.get(field)
    if spec is None:
        raise RuntimeSchedulerConfigError(
            f"unknown runtime scheduler config field: {field!r}"
        )
    if isinstance(value, bool) or not isinstance(value, spec.dtype):
        raise RuntimeSchedulerConfigError(
            f"{field}: expected {spec.dtype.__name__}, got {type(value).__name__!r}"
        )
    if spec.min_value is not None and value < spec.min_value:
        raise RuntimeSchedulerConfigError(f"{field}: {value} below minimum {spec.min_value}")
    if spec.max_value is not None and value > spec.max_value:
        raise RuntimeSchedulerConfigError(f"{field}: {value} above maximum {spec.max_value}")
    return value


def validate_runtime_max_running_seqs(value: Any, provisioned: int) -> int:
    """Validate the runtime admission policy against the provisioned ceiling.

    The upper bound is boot-time knowledge (``max_num_seqs`` sizes InputBatch,
    worker buffers and CUDA-graph capture), so the engine enforces it; this
    mirrors that check for the control-plane/fake side.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeSchedulerConfigError("runtime_max_running_seqs: expected int")
    if value < 1:
        raise RuntimeSchedulerConfigError("runtime_max_running_seqs: below minimum 1")
    if value > provisioned:
        raise RuntimeSchedulerConfigError(
            f"runtime_max_running_seqs: {value} exceeds provisioned max_num_seqs {provisioned}"
        )
    return value


def validate_runtime_max_scheduled_tokens(value: Any, provisioned: int) -> int:
    """Validate the runtime token-budget policy against the boot-time ceiling."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeSchedulerConfigError("runtime_max_scheduled_tokens: expected int")
    if value < 1:
        raise RuntimeSchedulerConfigError("runtime_max_scheduled_tokens: below minimum 1")
    if value > provisioned:
        raise RuntimeSchedulerConfigError(
            f"runtime_max_scheduled_tokens: {value} exceeds provisioned "
            f"scheduler token budget {provisioned}"
        )
    return value


def validate_runtime_kv_watermark(value: Any) -> float:
    """Validate the runtime admission-headroom policy: 0.0 <= v < 1.0."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RuntimeSchedulerConfigError("runtime_kv_watermark: expected float")
    v = float(value)
    if v < 0.0:
        raise RuntimeSchedulerConfigError("runtime_kv_watermark: below minimum 0.0")
    if v >= 1.0:
        raise RuntimeSchedulerConfigError("runtime_kv_watermark: must be < 1.0")
    return v


def validate_runtime_reserve_full_isl(value: Any) -> bool:
    """Validate the runtime long-request admission-gate policy: a bool."""
    if not isinstance(value, bool):
        raise RuntimeSchedulerConfigError("runtime_reserve_full_isl: expected bool")
    return value


def apply_overrides(target: Any, overrides: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ALL overrides, then apply them atomically in place.

    ``target`` may be a pydantic dataclass (setattr) or a plain dict
    (setitem). Returns the resulting observed values.
    """
    validated = {f: validate_override(f, v) for f, v in overrides.items()}
    for field, value in validated.items():
        if isinstance(target, dict):
            target[field] = value
        else:
            setattr(target, field, value)
    return dict(validated)