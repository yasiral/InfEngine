"""Control Contract V2: engine-neutral operator intent resolution and validation.

Placed ABOVE the existing ``DesiredState -> Reconciler -> Engine Adapter`` spine.
It does NOT replace the adapters (no generic adapter with engine-specific
branches) and does NOT redesign ``DesiredState``, ``StateStore``, ``Reconciler``,
``TransitionPlan`` or ``TransitionClass``.

Responsibilities:

  * **Intent resolution** — translate operator intents (a common intent alias
    such as ``runtime_concurrency``, or an existing V1 internal key) into the
    existing internal DesiredState key for the selected engine.
  * **Capability discovery** — expose the UNION of vLLM + SGLang capabilities
    with their exact semantic, scope, bounds, transition and quiescence
    metadata; engine-specific capabilities stay first-class and unsupported
    ones remain discoverable but not executable.
  * **Semantic validation** — reject unsupported intents and cross-engine
    semantic requests BEFORE any engine mutation (zero engine mutation).

Preservation invariants (M4B-C):

  * V1 API requests and persisted DesiredState keys continue to work verbatim:
    ``runtime_max_running_seqs``, ``runtime_max_scheduled_tokens``,
    ``long_prefill_token_threshold``, ``runtime_kv_watermark``,
    ``runtime_reserve_full_isl`` are never renamed.
  * The V2 alias ``runtime_concurrency`` resolves AT THE OPERATOR/API BOUNDARY
    to the existing internal key ``runtime_max_running_seqs``; it is never
    stored in DesiredState.
  * Token budgets are NOT normalized across engines. ``runtime_work_budget`` is
    a discovery family only, never an executable intent. A strict total
    scheduled-token budget against SGLang is REJECTED, not translated to the
    weaker prefill-admission budget.
  * Every vLLM/SGLang capability remains individually addressable.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .capabilities import Capability, ValidationError


class IntentResolution(str, Enum):
    """Per-intent resolution outcome (checked before any mutation)."""

    RESOLVED = "RESOLVED"          # valid for the selected engine -> reconcile
    UNSUPPORTED = "UNSUPPORTED"    # known but not supported on the selected engine
    REJECTED = "REJECTED"          # unknown key / invalid value / semantic mismatch


@dataclass(frozen=True)
class IntentSpec:
    """Product-level description of an operator intent (engine-independent)."""

    intent: str                        # operator-facing key
    internal_key: str                  # existing DesiredState/adapter key
    engines: tuple                     # engines that implement it (""vllm"", ""sglang"")
    semantic: str                      # exact semantic identity (never merged)
    family: Optional[str] = None       # discovery-only grouping (e.g. "work_budget")
    operation: bool = False            # one-shot operation, not persistent policy
    supported: bool = True             # False => advertised but never executable
    description: str = ""


# --------------------------------------------------------------------------- registry
# Single authoritative, repository-derived union of operator intents. This is the
# machine-readable form of docs/control-contract-v2-preservation-matrix.md.
INTENT_REGISTRY: Dict[str, IntentSpec] = {
    # Common intent: only where semantics genuinely match (CONCURRENCY_UPPER_BOUND).
    "runtime_concurrency": IntentSpec(
        intent="runtime_concurrency",
        internal_key="runtime_max_running_seqs",
        engines=("vllm", "sglang"),
        semantic="CONCURRENCY_UPPER_BOUND",
        family="concurrency",
        description=(
            "Common concurrency intent. Maps to vLLM runtime_max_running_seqs and "
            "SGLang runtime_max_running_seqs (pp_max_micro_batch_size). A "
            "CONCURRENCY_UPPER_BOUND, not an exact N-request guarantee; engine "
            "scope/bounds are preserved."
        ),
    ),
    # vLLM capabilities ------------------------------------------------
    "runtime_max_running_seqs": IntentSpec(
        intent="runtime_max_running_seqs",
        internal_key="runtime_max_running_seqs",
        engines=("vllm", "sglang"),
        semantic="CONCURRENCY_UPPER_BOUND",
        family="concurrency",
    ),
    "runtime_max_scheduled_tokens": IntentSpec(
        intent="runtime_max_scheduled_tokens",
        internal_key="runtime_max_scheduled_tokens",
        engines=("vllm",),
        semantic="STRICT_SCHEDULED_TOKEN_BUDGET",
        family="work_budget",
        description=(
            "vLLM-only strict per-step total scheduled-token budget. NOT the "
            "SGLang prefill-admission budget; never translated to it."
        ),
    ),
    "long_prefill_token_threshold": IntentSpec(
        intent="long_prefill_token_threshold",
        internal_key="long_prefill_token_threshold",
        engines=("vllm",),
        semantic="LONG_PREFILL_TOKEN_THRESHOLD",
        family="prefill",
    ),
    "runtime_kv_watermark": IntentSpec(
        intent="runtime_kv_watermark",
        internal_key="runtime_kv_watermark",
        engines=("vllm",),
        semantic="KV_ADMISSION_WATERMARK",
        family="kv",
    ),
    "runtime_reserve_full_isl": IntentSpec(
        intent="runtime_reserve_full_isl",
        internal_key="runtime_reserve_full_isl",
        engines=("vllm",),
        semantic="RESERVE_FULL_ISL",
        family="kv",
    ),
    "reset_prefix_cache": IntentSpec(
        intent="reset_prefix_cache",
        internal_key="reset_prefix_cache",
        engines=("vllm",),
        semantic="PREFIX_CACHE_RESET",
        family="cache_ops",
        operation=True,
        description="vLLM one-shot prefix-cache reset (DRAIN, WAIT_INFLIGHT).",
    ),
    # SGLang capabilities ----------------------------------------------
    "runtime_max_prefill_tokens": IntentSpec(
        intent="runtime_max_prefill_tokens",
        internal_key="runtime_max_prefill_tokens",
        engines=("sglang",),
        semantic="AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED",
        family="work_budget",
        description=(
            "SGLang-only aggregate prefill input-token admission budget; the "
            "first request of a scheduling pass is unbounded by it. NOT a strict "
            "per-request cap, NOT a total scheduler-token budget, NOT vLLM's "
            "runtime_max_scheduled_tokens."
        ),
    ),
    "engine_cache_flush": IntentSpec(
        intent="engine_cache_flush",
        internal_key="engine_cache_flush",
        engines=("sglang",),
        semantic="ENGINE_CACHE_FLUSH",
        family="cache_ops",
        operation=True,
        description=(
            "SGLang one-shot flush_cache() (broader than a prefix reset; REQUIRE_IDLE)."
        ),
    ),
    "prefix_cache_reset": IntentSpec(
        intent="prefix_cache_reset",
        internal_key="prefix_cache_reset",
        engines=("sglang",),
        semantic="PREFIX_CACHE_RESET",
        family="cache_ops",
        operation=True,
        supported=False,
        description="Advertised but unsupported on SGLang; use engine_cache_flush.",
    ),
}


@dataclass
class ResolvedIntent:
    """Outcome of resolving one operator intent against a specific engine adapter."""

    intent: str
    status: IntentResolution
    internal_key: Optional[str] = None
    value: Any = None
    semantic: Optional[str] = None
    reason: Optional[str] = None
    operation: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "status": self.status.value,
            "internal_key": self.internal_key,
            "value": self.value,
            "semantic": self.semantic,
            "operation": self.operation,
            "reason": self.reason,
        }


@dataclass
class ResolveResult:
    """Result of resolving an operator request. ``valid`` gates mutation."""

    intents: List[ResolvedIntent] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return all(i.status is IntentResolution.RESOLVED for i in self.intents)

    @property
    def resolved(self) -> Dict[str, Any]:
        """Internal DesiredState keys -> values, for the existing Reconciler."""
        return {
            i.internal_key: i.value
            for i in self.intents
            if i.status is IntentResolution.RESOLVED
        }

    def rejected(self) -> List[ResolvedIntent]:
        return [i for i in self.intents if i.status is not IntentResolution.RESOLVED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "intents": [i.to_dict() for i in self.intents],
            "resolved": self.resolved,
        }


class ControlContract:
    """Semantic resolution / validation layer over the existing control plane.

    A contract is bound to one selected engine (``"builtin"`` / ``"vllm"`` /
    ``"sglang"``) and resolves operator requests against that engine's adapter.
    It never mutates the engine itself: it only decides which existing internal
    keys may be reconciled, then hands them to the existing ``Reconciler``.
    """

    def __init__(self, engine: str):
        self.engine = engine

    # ------------------------------------------------------------- discovery
    def intent_names(self) -> List[str]:
        """The full union of operator intents (both engines + common intent)."""
        return sorted(INTENT_REGISTRY)

    def discover(self, adapter) -> List[Dict[str, Any]]:
        """Return the union of capabilities annotated for the current engine.

        Each entry carries enough for an operator/future controller:
        canonical/operator intent, existing internal key, engine(s), supported
        status, exact semantic, scope, bounds, transition, persistent-vs-operation,
        and quiescence metadata (from the live ``Capability`` where supported).
        Unsupported / other-engine capabilities stay discoverable but are not
        executable.
        """
        out: List[Dict[str, Any]] = []
        for name in self.intent_names():
            spec = INTENT_REGISTRY[name]
            cap = adapter.capability(spec.internal_key) if adapter is not None else None
            supported = bool(cap is not None and cap.supported)
            entry = {
                "intent": spec.intent,
                "internal_key": spec.internal_key,
                "engine": list(spec.engines),
                "supported": supported,
                "supported_on_selected_engine": self.engine in spec.engines and supported,
                "semantic": spec.semantic,
                "family": spec.family,
                "operation": spec.operation,
                "description": spec.description,
                "scope": cap.scope if cap is not None else None,
                "dtype": cap.dtype if cap is not None else None,
                "min": cap.min_value if cap is not None else None,
                "max": cap.max_value if cap is not None else None,
                "transition": cap.transition.value if cap is not None else None,
                "quiescence_strategy": (
                    cap.quiescence_strategy.value if cap is not None and cap.quiescence_strategy else None
                ),
                "quiescence_requirement": (
                    cap.quiescence_requirement if cap is not None else None
                ),
                "requires_quiescence": (
                    bool(cap.requires_quiescence) if cap is not None else None
                ),
            }
            out.append(entry)
        return out

    # -------------------------------------------------------------- resolution
    def resolve(self, adapter, request: Dict[str, Any]) -> ResolveResult:
        """Resolve + validate an operator request against ``adapter``.

        Returns a :class:`ResolveResult`. If ``result.valid`` is False the caller
        MUST NOT reconcile/mutate (zero engine mutation). Resolved internal keys
        are fed to the existing ``Reconciler`` as-is (persisted keys unchanged).
        """
        intents: List[ResolvedIntent] = []
        for key, value in request.items():
            intents.append(self._resolve_one(adapter, key, value))
        # M13-5A: reject duplicate/conflicting aliases. Two operator keys that
        # resolve to the SAME internal DesiredState key (e.g. the alias
        # ``runtime_concurrency`` and its native ``runtime_max_running_seqs``)
        # must never silently overwrite one another. Any collision invalidates
        # the whole request before mutation.
        by_internal: Dict[str, ResolvedIntent] = {}
        conflicts: Dict[str, str] = {}
        for intent in intents:
            if intent.status is not IntentResolution.RESOLVED or intent.internal_key is None:
                continue
            first = by_internal.get(intent.internal_key)
            if first is None:
                by_internal[intent.internal_key] = intent
            elif first.value != intent.value:
                conflicts[intent.internal_key] = (
                    f"conflicting aliases for {intent.internal_key!r}: "
                    f"{first.intent!r}={first.value!r} vs {intent.intent!r}={intent.value!r}"
                )
        if conflicts:
            for intent in intents:
                if intent.internal_key in conflicts and intent.status is IntentResolution.RESOLVED:
                    intent.status = IntentResolution.REJECTED
                    intent.reason = conflicts[intent.internal_key]
        return ResolveResult(intents)

    def _resolve_one(self, adapter, key: str, value: Any) -> ResolvedIntent:
        spec = INTENT_REGISTRY.get(key)
        cap = adapter.capability(key)

        # Existing internal key present on this adapter (V1 path).
        if cap is not None:
            if not cap.supported:
                return ResolvedIntent(
                    key, IntentResolution.UNSUPPORTED, internal_key=key, value=value,
                    semantic=cap.semantic, operation=cap.operation,
                    reason=f"{key!r} is advertised but unsupported on {self.engine}",
                )
            try:
                normalized = adapter.coerce(key, value)
            except ValidationError as exc:
                return ResolvedIntent(
                    key, IntentResolution.REJECTED, internal_key=key, value=value,
                    semantic=cap.semantic, operation=cap.operation, reason=str(exc),
                )
            return ResolvedIntent(
                key, IntentResolution.RESOLVED, internal_key=key, value=normalized,
                semantic=cap.semantic, operation=cap.operation,
            )

        # Unknown to this adapter but in the registry.
        if spec is not None:
            internal = adapter.capability(spec.internal_key)
            if internal is None:
                return ResolvedIntent(
                    key, IntentResolution.UNSUPPORTED, internal_key=spec.internal_key,
                    value=value, semantic=spec.semantic, operation=spec.operation,
                    reason=(
                        f"{key!r} is a {','.join(spec.engines)}-specific capability "
                        f"({spec.semantic}); not supported on {self.engine}"
                    ),
                )
            if not internal.supported:
                return ResolvedIntent(
                    key, IntentResolution.UNSUPPORTED, internal_key=spec.internal_key,
                    value=value, semantic=spec.semantic, operation=spec.operation,
                    reason=f"{key!r} is advertised but unsupported on {self.engine}",
                )
            try:
                normalized = adapter.coerce(spec.internal_key, value)
            except ValidationError as exc:
                return ResolvedIntent(
                    key, IntentResolution.REJECTED, internal_key=spec.internal_key,
                    value=value, semantic=spec.semantic, operation=spec.operation,
                    reason=str(exc),
                )
            return ResolvedIntent(
                key, IntentResolution.RESOLVED, internal_key=spec.internal_key,
                value=normalized, semantic=spec.semantic, operation=spec.operation,
            )

        # Completely unknown intent.
        return ResolvedIntent(
            key, IntentResolution.REJECTED, value=value,
            reason=f"unknown intent {key!r}",
        )