"""Strict typed validation for the SGLang runtime knobs.

This is the control-plane-side mirror of the allowlist the SGLang v0.5.18
patch enforces (see ``contrib/sglang/v0.5.18_runtime_prefill_budget.patch``).
The adapter validates at the control-plane boundary (``Capability``/``coerce``);
the patched SGLang ``Scheduler.set_internal_state`` re-validates for safety.
Both sides must keep the rules identical.

Strictness is the point: a value that is not a genuine ``int`` (string, float,
None, bool) is rejected — never coerced (``int(2.5) -> 2`` would silently change
the operator's intent). Bool is rejected explicitly because ``bool`` is a
subclass of ``int``.
"""

from typing import Any, Dict


class SglangControlError(ValueError):
    """An invalid SGLang control value was rejected."""


def validate_concurrency_upper_bound(value: Any, provisioned: int) -> int:
    """Validate the runtime concurrency upper bound against the provisioned
    ceiling (``max_running_requests``). Strict int only."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SglangControlError("runtime_max_running_seqs: expected int")
    if value < 1:
        raise SglangControlError("runtime_max_running_seqs: below minimum 1")
    if value > provisioned:
        raise SglangControlError(
            f"runtime_max_running_seqs: {value} exceeds provisioned "
            f"max_running_requests {provisioned}"
        )
    return value


def validate_prefill_admission_budget(value: Any, provisioned: int) -> int:
    """Validate the runtime prefill admission budget against the boot ceiling
    (``max_prefill_tokens``). Strict int only."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise SglangControlError("runtime_max_prefill_tokens: expected int")
    if value < 1:
        raise SglangControlError("runtime_max_prefill_tokens: below minimum 1")
    if value > provisioned:
        raise SglangControlError(
            f"runtime_max_prefill_tokens: {value} exceeds provisioned "
            f"max_prefill_tokens {provisioned}"
        )
    return value


def apply_policy_record(record: Dict[str, Any], overrides: Dict[str, Any],
                        provisioned: Dict[str, Any]) -> Dict[str, Any]:
    """Validate ALL overrides, then apply them atomically in place to a policy
    record. Returns the resulting record (for fake-transport parity)."""
    for field, value in overrides.items():
        if field == "runtime_max_running_seqs":
            validate_concurrency_upper_bound(value, provisioned["max_running_requests"])
        elif field == "runtime_max_prefill_tokens":
            validate_prefill_admission_budget(value, provisioned["max_prefill_tokens"])
        else:
            raise SglangControlError(f"unknown SGLang policy field: {field!r}")
    for field, value in overrides.items():
        record[field] = int(value)
    return record