"""M7D0: machine-readable policy coverage for the persistent HOT controls.

For each currently exposed persistent HOT control and each engine, this module
reports whether Phase-7 supervision has a policy contract
(``POLICY_SUPPORTED``), whether the control is executable only manually because
telemetry cannot factually support a policy
(``MANUAL_ONLY_INSUFFICIENT_EVIDENCE``), or whether the engine does not expose
the intent at all (``ENGINE_UNSUPPORTED``).

``policy_for`` selects the *actuator-specific* policy for an intent. There is no
single generic policy algorithm — each supported actuator has its own policy.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from infengine.control import INTENT_REGISTRY

from .policy import ConcurrencyPolicy, KvWatermarkPolicy, TokenBudgetPolicy
from .supervisor import INTENT

# Engine-specific token-budget intents. Never aliased to each other: vLLM's
# runtime_max_scheduled_tokens is a STRICT per-step scheduled-token budget; SGLang's
# runtime_max_prefill_tokens is an AGGREGATE prefill input-admission budget whose
# first request per scheduling pass is unbounded. Different semantics, different
# bounds, not interchangeable.
SCHEDULED_TOKENS_INTENT = "runtime_max_scheduled_tokens"
SCHEDULED_TOKENS_KEY = "runtime_max_scheduled_tokens"
SGLANG_PREFILL_INTENT = "runtime_max_prefill_tokens"
SGLANG_PREFILL_KEY = "runtime_max_prefill_tokens"
KV_WATERMARK_INTENT = "runtime_kv_watermark"
KV_WATERMARK_KEY = "runtime_kv_watermark"
LONG_PREFILL_INTENT = "long_prefill_token_threshold"
RESERVE_FULL_ISL_INTENT = "runtime_reserve_full_isl"


class PolicySupport(str, Enum):
    POLICY_SUPPORTED = "POLICY_SUPPORTED"
    POLICY_PENDING_EVALUATION = "POLICY_PENDING_EVALUATION"
    MANUAL_ONLY_INSUFFICIENT_EVIDENCE = "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"
    ENGINE_UNSUPPORTED = "ENGINE_UNSUPPORTED"


@dataclass(frozen=True)
class PolicyCoverageEntry:
    """One (intent, engine) coverage record."""

    intent: str
    internal_key: str
    engine: str
    status: str
    dtype: str
    bounds: str
    semantics: str
    required_evidence: Tuple[str, ...]
    reason: str
    direction_support: Optional[Dict[str, str]] = None
    # M7D3: source-derived effect model for the actuator:
    #   MONOTONIC          increase = MORE_PERMISSIVE, decrease = MORE_CONSERVATIVE
    #   INVERSE_MONOTONIC  increase = MORE_CONSERVATIVE, decrease = MORE_PERMISSIVE
    #   NON_MONOTONIC      no monotonic permissive/conservative orientation
    #   BOOLEAN            boolean policy; explicit per-transition effects
    effect_model: Optional[str] = None
    transition_effects: Optional[Dict[str, str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "internal_key": self.internal_key,
            "engine": self.engine,
            "status": self.status,
            "dtype": self.dtype,
            "bounds": self.bounds,
            "semantics": self.semantics,
            "required_evidence": list(self.required_evidence),
            "reason": self.reason,
            "direction_support": dict(self.direction_support) if self.direction_support else None,
            "effect_model": self.effect_model,
            "transition_effects": dict(self.transition_effects) if self.transition_effects else None,
        }


_S = PolicySupport.POLICY_SUPPORTED
_P = PolicySupport.POLICY_PENDING_EVALUATION
_M = PolicySupport.MANUAL_ONLY_INSUFFICIENT_EVIDENCE
_U = PolicySupport.ENGINE_UNSUPPORTED

_GUARDRAIL_EVIDENCE = ("requests_running", "requests_waiting", "ttft_p95_s", "kv_usage")
_TOKEN_BUDGET_EVIDENCE = (
    "requests_running",
    "requests_waiting",
    "iteration_tokens_p95",
    "ttft_p95_s",
    "kv_usage",
)

# Source-derived transition-effect models (M7D3).
_MONOTONIC = {"increase": "MORE_PERMISSIVE", "decrease": "MORE_CONSERVATIVE"}
_INVERSE = {"increase": "MORE_CONSERVATIVE", "decrease": "MORE_PERMISSIVE"}
_BOOLEAN_ISL = {"false_to_true": "MORE_CONSERVATIVE", "true_to_false": "MORE_PERMISSIVE"}
_NON_MONOTONIC_PREFILL = {
    "note": "0 disables the per-step long-prefill cap; positive values cap it "
            "(larger = larger chunks); no monotonic permissive/conservative orientation",
}

# intent -> {engine: status, dtype, bounds, required_evidence, reason{engine}}
_COVERAGE: Dict[str, Dict[str, Any]] = {
    "runtime_concurrency": {
        "status": {"vllm": _S, "sglang": _S},
        "dtype": "int",
        "bounds": "1..provisioned max_num_seqs (vLLM) / max_running_requests (SGLang)",
        "required_evidence": _GUARDRAIL_EVIDENCE,
        "reason": {
            "vllm": "ConcurrencyPolicy: saturation with waiting demand + passing guardrails.",
            "sglang": "ConcurrencyPolicy: saturation with waiting demand + passing guardrails.",
        },
        "effect_model": "MONOTONIC",
        "transition_effects": _MONOTONIC,
    },
    "runtime_max_scheduled_tokens": {
        "status": {"vllm": _S, "sglang": _U},
        "dtype": "int",
        "bounds": "1..provisioned max_num_scheduled_tokens (boot max_num_batched_tokens)",
        "required_evidence": _TOKEN_BUDGET_EVIDENCE,
        "reason": {
            "vllm": "TokenBudgetPolicy: scale-up requires persistent budget-utilization evidence "
                    "(windowed tokens per scheduling step, iteration_tokens_p95, at/above the "
                    "active budget) while waiting demand exists and guardrails pass; a breach "
                    "yields a bounded decrease. Generic queueing is not attributed.",
            "sglang": "vLLM-only strict per-step scheduled-token budget; SGLang exposes "
                      "runtime_max_prefill_tokens instead (different semantics).",
        },
        "effect_model": "MONOTONIC",
        "transition_effects": _MONOTONIC,
    },
    "runtime_max_prefill_tokens": {
        "status": {"sglang": _M, "vllm": _U},
        "dtype": "int",
        "bounds": "1..provisioned max_prefill_tokens (local scheduler; first request of a "
                  "scheduling pass is unbounded; proven envelope PP=1/DP=1)",
        "required_evidence": (),
        "reason": {
            "sglang": "MANUAL_ONLY: no factual telemetry distinguishes prefill-admission-budget "
                      "throttling from generic capacity queueing (queue depth is not sufficient "
                      "evidence, and the first request per pass is unbounded). Needs a "
                      "budget-specific signal before a policy can be supported.",
            "vllm": "SGLang-only aggregate prefill input-admission budget; NOT vLLM "
                    "runtime_max_scheduled_tokens.",
        },
        "effect_model": "MONOTONIC",
        "transition_effects": _MONOTONIC,
    },
    "runtime_kv_watermark": {
        "status": {"vllm": _S, "sglang": _U},
        "dtype": "float",
        "bounds": "0.0..<1.0 (fraction of KV blocks reserved as admission headroom)",
        "required_evidence": ("kv_usage", "num_preemptions"),
        "reason": {
            "vllm": "KvWatermarkPolicy (asymmetric): conservative increase (MORE_CONSERVATIVE) "
                    "supported on factual KV usage + preemption pressure; permissive decrease "
                    "(MORE_PERMISSIVE) is MANUAL_ONLY — vLLM exposes no watermark-binding / "
                    "admission-block signal, and generic queueing is not substituted.",
            "sglang": "vLLM-only control.",
        },
        "direction_support": {"vllm": {
            "MORE_CONSERVATIVE": "POLICY_SUPPORTED",
            "MORE_PERMISSIVE": "MANUAL_ONLY_INSUFFICIENT_EVIDENCE",
        }},
        "effect_model": "INVERSE_MONOTONIC",
        "transition_effects": _INVERSE,
    },
    "long_prefill_token_threshold": {
        "status": {"vllm": _M, "sglang": _U},
        "dtype": "int",
        "bounds": "0..max_model_len (0 disables the per-step long-prefill cap)",
        "required_evidence": (),
        "reason": {
            "vllm": "MANUAL_ONLY: NON_MONOTONIC (0 disables the cap; positive values cap "
                    "per-step long-prefill tokens, larger = larger chunks) and vLLM exposes "
                    "no factual long-prefill classification/chunk/decision telemetry; no "
                    "queue-depth attribution.",
            "sglang": "vLLM-only control.",
        },
        "effect_model": "NON_MONOTONIC",
        "transition_effects": _NON_MONOTONIC_PREFILL,
    },
    "runtime_reserve_full_isl": {
        "status": {"vllm": _M, "sglang": _U},
        "dtype": "bool",
        "bounds": "bool",
        "required_evidence": (),
        "reason": {
            "vllm": "MANUAL_ONLY: boolean admission gate (True = full sequence must fit, "
                    "MORE_CONSERVATIVE; False = first chunk only, MORE_PERMISSIVE). No "
                    "factual 'blocked because full-ISL does not fit' telemetry exists "
                    "(waiting reasons are only capacity/deferred); neither transition is "
                    "supportable and preemptions are not full-ISL-specific.",
            "sglang": "vLLM-only control.",
        },
        "effect_model": "BOOLEAN",
        "transition_effects": _BOOLEAN_ISL,
    },
}


def policy_coverage(engine: Optional[str] = None) -> List[Dict[str, Any]]:
    """Machine-readable coverage for the persistent HOT controls.

    Returns one entry per (intent, engine); filter with ``engine`` (``"vllm"`` /
    ``"sglang"``) when a single engine's view is wanted.
    """
    entries: List[Dict[str, Any]] = []
    for intent, spec in _COVERAGE.items():
        registry = INTENT_REGISTRY.get(intent)
        internal_key = registry.internal_key if registry is not None else intent
        semantics = registry.semantic if registry is not None else ""
        for eng, status in spec["status"].items():
            if engine is not None and eng != engine:
                continue
            entries.append(PolicyCoverageEntry(
                intent=intent,
                internal_key=internal_key,
                engine=eng,
                status=status.value,
                dtype=spec["dtype"],
                bounds=spec["bounds"],
                semantics=semantics or "",
                required_evidence=tuple(spec["required_evidence"]),
                reason=spec["reason"].get(eng, ""),
                direction_support=(spec.get("direction_support") or {}).get(eng),
                effect_model=spec.get("effect_model"),
                transition_effects=spec.get("transition_effects"),
            ).to_dict())
    return entries


def policy_for(intent: str, config: Any) -> Optional[Any]:
    """Return the actuator-specific policy for ``intent``, or ``None``.

    Only intents whose coverage is ``POLICY_SUPPORTED`` return a policy; there is
    no single generic algorithm. Engine-specific intents are never aliased
    (``runtime_max_scheduled_tokens`` and ``runtime_max_prefill_tokens`` are
    separate; the latter is deliberately unsupported).
    """
    if intent == INTENT:
        return ConcurrencyPolicy(config)
    if intent == SCHEDULED_TOKENS_INTENT:
        return TokenBudgetPolicy(config)
    if intent == KV_WATERMARK_INTENT:
        return KvWatermarkPolicy(config)
    return None
