"""M7A: minimal deterministic supervisory controller (OBSERVE_ONLY).

The supervisor wraps the pure :class:`~infengine.supervisor.policy.ConcurrencyPolicy`
with the read-only control-plane state and the supervisory safety gates:

  * **capability bounds** — the proposal is checked against the Engine's
    advertised ``min``/``max`` (defensive; the policy already respects them);
  * **cooldown** — a new proposal is suppressed while ``cooldown_s`` has not
    elapsed since the last emitted proposal;
  * **conflict / safety** — HOLD when desired != observed (state not converged)
    or when ``desired_revision`` changed since the previous observation.

OBSERVE_ONLY is the only mode implemented. The supervisor reads state through
``control.capabilities()`` and ``control.get_config()`` only; it never calls the
mutation path (``PATCH /runtime/config`` / Reconciler). When approved mutation is
added in a later milestone it must go through the existing Control
Contract/DesiredState/Reconciler — there is no second mutation path here.

Only ``runtime_concurrency`` (internal key ``runtime_max_running_seqs``) is
modelled; no LLM/ML/RL, no generic diagnosis, no fleet/routing/dashboard, no
admission layer, no additional actuator.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, Sequence, Tuple

from ..telemetry import RuntimeSnapshot
from .policy import (
    ActuatorValue,
    CapabilityBounds,
    ConcurrencyPolicy,
    DecisionKind,
    SupervisorConfig,
)
from .signals import build_signal_window

INTENT = "runtime_concurrency"
INTERNAL_KEY = "runtime_max_running_seqs"
_PROPOSAL_KINDS = (
    DecisionKind.PROPOSE_UP,
    DecisionKind.PROPOSE_DOWN,
    DecisionKind.PROPOSE_CHANGE,
)


class SupervisorMode(str, Enum):
    """The only implemented mode mutates nothing.

    A future approved-mutation mode must reuse the existing Control
    Contract/DesiredState/Reconciler; it is intentionally absent here.
    """

    OBSERVE_ONLY = "OBSERVE_ONLY"


class ControlReadView(Protocol):
    """Read-only control-plane surface the supervisor depends on.

    Satisfied by the existing ``ControlPlaneClient`` (``experiment.py``): it
    exposes ``capabilities()`` and ``get_config()``. The supervisor never calls
    the client's mutation methods.
    """

    def capabilities(self) -> List[Dict[str, Any]]:
        ...

    def get_config(self) -> Dict[str, Any]:
        ...


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DecisionRecord:
    """The complete, auditable record of one supervisory observation.

    Carries the exact telemetry window, the numeric evidence, guardrail
    outcomes, capability bounds, current value, proposed value, transition type
    and the reasons for HOLD / PROPOSE. ``mutation_applied`` is always False in
    OBSERVE_ONLY.
    """

    mode: str
    intent: str
    internal_key: str
    decided_at: str
    decision: str
    proposed_value: Optional[ActuatorValue]
    current_value: Optional[ActuatorValue]
    capability_bounds: Optional[Dict[str, Any]]
    transition_type: Optional[str]
    guardrails: List[Dict[str, Any]]
    evidence: Dict[str, Any]
    window: Dict[str, Any]
    reasons: List[str]
    overrides: List[str]
    safety: Dict[str, Any]
    desired_revision: Optional[int]
    policy: Dict[str, Any]
    effect: Optional[str] = None
    mutation_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "intent": self.intent,
            "internal_key": self.internal_key,
            "decided_at": self.decided_at,
            "decision": self.decision,
            "proposed_value": self.proposed_value,
            "effect": self.effect,
            "current_value": self.current_value,
            "capability_bounds": self.capability_bounds,
            "transition_type": self.transition_type,
            "guardrails": self.guardrails,
            "evidence": self.evidence,
            "window": self.window,
            "reasons": self.reasons,
            "overrides": self.overrides,
            "safety": self.safety,
            "desired_revision": self.desired_revision,
            "policy": self.policy,
            "mutation_applied": self.mutation_applied,
        }


class Supervisor:
    """Deterministic, OBSERVE_ONLY supervisory controller for one actuator intent.

    Defaults to the common ``runtime_concurrency`` intent/policy; pass
    ``intent``/``internal_key``/``policy`` to supervise another actuator (e.g. a
    token budget). The lifecycle (observe -> propose -> human approval ->
    If-Match/Reconciler -> convergence -> assessment) is shared; each actuator has
    its own policy algorithm.
    """

    def __init__(
        self,
        config: SupervisorConfig,
        control: ControlReadView,
        *,
        policy: Optional[Any] = None,
        intent: str = INTENT,
        internal_key: str = INTERNAL_KEY,
        mode: SupervisorMode = SupervisorMode.OBSERVE_ONLY,
    ):
        if mode is not SupervisorMode.OBSERVE_ONLY:
            raise ValueError(
                "only OBSERVE_ONLY is implemented; mutation must reuse the "
                "existing Control Contract/DesiredState/Reconciler (M7A scope)"
            )
        self.config = config
        self.control = control
        self.mode = mode
        self.intent = intent
        self.internal_key = internal_key
        self._policy = policy or ConcurrencyPolicy(config)
        self._last_propose_at: Optional[float] = None
        self._last_revision: Optional[int] = None

    # ------------------------------------------------------------- control view
    def _capability(self) -> Optional[Dict[str, Any]]:
        try:
            entries = self.control.capabilities()
        except Exception:
            return None
        for entry in entries or []:
            if (
                entry.get("intent") == self.intent
                or entry.get("internal_key") == self.internal_key
            ):
                return entry
        return None

    # --------------------------------------------------------------- observation
    def observe(
        self,
        samples: Sequence[Tuple[float, RuntimeSnapshot]],
        *,
        now: Optional[float] = None,
    ) -> DecisionRecord:
        """Evaluate one observation; never mutates DesiredState or the engine."""
        now = time.monotonic() if now is None else now
        config = self.config

        config_payload = self.control.get_config()
        observed = config_payload.get("observed") or {}
        desired = config_payload.get("desired") or {}
        revision = config_payload.get("desired_revision")

        cap = self._capability()
        current = observed.get(self.internal_key)
        if current is None:
            current = desired.get(self.internal_key)

        bounds: Optional[CapabilityBounds] = None
        if cap is not None and cap.get("min") is not None and cap.get("max") is not None:
            bounds = CapabilityBounds(cap["min"], cap["max"])

        signals = build_signal_window(
            samples, now=now, window_s=config.window_s,
            metrics=config.effective_metrics(),
        )
        evaluation = self._policy.evaluate(
            signals, current_value=current, bounds=bounds
        )

        safety: Dict[str, Any] = {
            "capability_found": cap is not None,
            "converged": None,
            "concurrent_change_detected": False,
            "cooldown_active": False,
            "bounds_enforced": True,
        }
        overrides: List[str] = []

        # Safety: desired/observed convergence.
        desired_value = desired.get(self.internal_key)
        if desired_value is not None and current is not None:
            safety["converged"] = desired_value == current
            if not safety["converged"]:
                overrides.append(
                    f"state not converged: desired={desired_value} != observed={current}"
                )

        # Safety: a concurrent DesiredState change since the previous observation.
        if (
            self._last_revision is not None
            and revision is not None
            and revision != self._last_revision
        ):
            safety["concurrent_change_detected"] = True
            overrides.append(
                f"desired_revision changed {self._last_revision} -> {revision} "
                "since previous observation"
            )
        if revision is not None:
            self._last_revision = revision

        # Cooldown: gate a fresh proposal.
        if evaluation.kind in _PROPOSAL_KINDS:
            if self._last_propose_at is not None:
                elapsed = now - self._last_propose_at
                if elapsed < config.cooldown_s:
                    remaining = config.cooldown_s - elapsed
                    safety["cooldown_active"] = True
                    overrides.append(
                        f"cooldown active: {remaining:.1f}s of {config.cooldown_s}s "
                        "remaining since last proposal"
                    )

        # Bounds enforcement (defensive; the policy already clamps).
        proposed = evaluation.proposed_value
        if proposed is not None and bounds is not None:
            if proposed < bounds.min_value or proposed > bounds.max_value:
                safety["bounds_enforced"] = False
                overrides.append(
                    f"proposed {proposed} outside capability bounds "
                    f"[{bounds.min_value}, {bounds.max_value}]"
                )

        decision = evaluation.kind
        if overrides:
            decision = DecisionKind.HOLD
            proposed = None
        elif decision in _PROPOSAL_KINDS:
            self._last_propose_at = now

        reasons = list(overrides) + list(evaluation.reasons)
        effect = None
        if decision in _PROPOSAL_KINDS:
            effect = evaluation.effect.value if evaluation.effect else None

        return DecisionRecord(
            mode=self.mode.value,
            intent=self.intent,
            internal_key=self.internal_key,
            decided_at=_iso_now(),
            decision=decision.value,
            proposed_value=proposed,
            current_value=current,
            capability_bounds=bounds.to_dict() if bounds else None,
            transition_type=cap.get("transition") if cap else None,
            guardrails=[g.to_dict() for g in evaluation.guardrails],
            evidence=dict(evaluation.evidence),
            window=signals.to_dict(),
            reasons=reasons,
            overrides=overrides,
            safety=safety,
            desired_revision=revision,
            policy=evaluation.to_dict(),
            effect=effect,
            mutation_applied=False,
        )
