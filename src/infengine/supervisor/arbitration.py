"""M7E: deterministic multi-actuator arbitration (human-governed).

M7E is the final Phase-7 milestone. It arbitrates **only among actuator
transitions that are already ``POLICY_SUPPORTED``** in the machine-readable
coverage matrix. It adds no mutation path: a selected candidate is handed to the
existing M7B human registration/approval machinery, and the existing M7D1
engine-wide causal-isolation gate remains the final enforcement barrier.

Design rules (all deterministic and inspectable):

  * **One evidence epoch.** Every eligible policy is evaluated against the same
    engine/instance, the same telemetry ``samples`` and ``now``, the same desired
    revision and the same capability snapshot (captured once via
    :class:`_EpochControl`). Each policy owns its own HOLD/proposal decision;
    M7E never reinterprets raw telemetry or manufactures a proposal.
  * **First priority: policy effect.** MORE_CONSERVATIVE proposals outrank
    MORE_PERMISSIVE proposals, so a factual safety/guardrail response is addressed
    before a capacity-expansion proposal. There are no weighted scores, opaque
    utility functions, throughput optimization, or inferred bottleneck ranking.
  * **Second priority: configured actuator order.** Same-effect ties are resolved
    by a configured, visible, stable actuator order — never by pretending
    telemetry proves one actuator is the root cause. The default order for vLLM
    is ``runtime_kv_watermark → runtime_concurrency → runtime_max_scheduled_tokens``
    (the KV conservative proposal carries direct KV/preemption evidence; the
    others are broader safety responses). It is configurable/overridable and the
    effective order is recorded.
  * **One change at a time.** While any application or post-change assessment is
    active, ``gate()`` returns a reason and the outcome is ``DEFERRED`` with zero
    mutation. Candidates are recomputed from fresh evidence on every call; losing
    candidates are never stored or reused.
  * **Manual-only is never eligible.** Eligibility comes solely from the coverage
    matrix; ``PROPOSE_CHANGE``, effect metadata, or HOT mutability alone never
    make a manual-only transition eligible. A supported direction (e.g. KV
    MORE_CONSERVATIVE) is required; KV MORE_PERMISSIVE is never a candidate.

The :class:`ArbitrationRecord` is an audit artifact, not a new DesiredState.
"""

import time
from dataclasses import dataclass, replace
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .approval import ApplicationStatus
from .coverage import (
    KV_WATERMARK_INTENT,
    SCHEDULED_TOKENS_INTENT,
    policy_coverage,
    policy_for,
)
from .policy import ActuatorValue, PolicyEffect, SupervisorConfig
from .supervisor import INTENT, DecisionRecord, Supervisor

# A candidate is a real proposal from a policy (never a HOLD).
_PROPOSAL_DECISIONS = frozenset({"PROPOSE_UP", "PROPOSE_DOWN", "PROPOSE_CHANGE"})
# Only directional effects can be arbitrated. NON_MONOTONIC / boolean changes are
# manual-only (M7D3) and are never eligible.
_ARBITRATION_EFFECTS = (
    PolicyEffect.MORE_CONSERVATIVE.value,
    PolicyEffect.MORE_PERMISSIVE.value,
)
_EFFECT_ORDER = {
    PolicyEffect.MORE_CONSERVATIVE.value: 0,
    PolicyEffect.MORE_PERMISSIVE.value: 1,
}

# Terminal applications that release the engine-wide slot: no assessment follows.
_SLOT_RELEASING = frozenset({
    ApplicationStatus.REJECTED.value,
    ApplicationStatus.STALE.value,
    ApplicationStatus.APPLY_FAILED.value,
})

# Deterministic default actuator priority per engine. Same-effect ties resolve by
# this order; a conservative KV proposal outranks the broader safety responses.
DEFAULT_ACTUATOR_PRIORITY: Dict[str, Tuple[str, ...]] = {
    "vllm": (KV_WATERMARK_INTENT, INTENT, SCHEDULED_TOKENS_INTENT),
    "sglang": (INTENT,),
}


class ArbitrationOutcome(str, Enum):
    """Explicit arbitration result (no mutation is performed by M7E)."""

    NO_CANDIDATE = "NO_CANDIDATE"
    SELECTED = "SELECTED"
    DEFERRED = "DEFERRED"


class PolicyChangeSlot:
    """Engine-wide, one-change-at-a-time lifecycle slot (M7E invariant).

    The slot owns the engine-wide policy-change lifecycle and is consulted by
    every arbitration. It is **fail closed**: if its state cannot be determined
    (no assessment manager, or the provider raises) it reports the slot as busy,
    so arbitration returns ``DEFERRED`` rather than selecting another proposal.

    Lifecycle:

      * :meth:`claim` — called when a selected proposal is **registered** into
        M7B (the slot begins busy here, before any mutation or assessment).
      * :meth:`note_application` — called with each M7B application outcome:
        ``REJECTED`` / ``STALE`` / ``APPLY_FAILED`` release the slot (no
        assessment follows); ``DEFERRED`` keeps it busy (the proposal is
        preserved); ``CONVERGED`` keeps it busy until the M7C assessment for that
        application becomes terminal or is abandoned.
      * :meth:`blocking_reason` — ``None`` when free, else why the slot is busy.

    A proposal that is still pending human decision, an application that is
    non-terminal, and any active M7C assessment all block the slot.
    """

    def __init__(self, assessments: Optional[Any] = None):
        self._assessments = assessments
        self._pending_id: Optional[str] = None
        self._awaiting: Optional[Any] = None

    def claim(self, decision_id: str) -> None:
        """Occupy the slot: a selected proposal was registered into M7B."""
        self._pending_id = decision_id
        self._awaiting = None

    def note_application(self, application: Any) -> None:
        """Update the slot from a terminal/non-terminal M7B application outcome."""
        status = getattr(application, "status", None)
        applied = bool(getattr(application, "applied", False))
        if status == ApplicationStatus.CONVERGED.value and applied:
            # Mutation happened: stay busy until the M7C assessment terminates.
            self._pending_id = None
            self._awaiting = application
        elif status in _SLOT_RELEASING:
            # Rejected / stale / apply-failed: no assessment follows -> release.
            self._pending_id = None
            self._awaiting = None
        elif status == ApplicationStatus.DEFERRED.value:
            # Non-terminal: the proposal is preserved; the slot stays occupied.
            return
        # Unknown status: leave state unchanged (fail closed).

    def release(self) -> None:
        """Explicitly free the slot (e.g. an abandoned assessment)."""
        self._pending_id = None
        self._awaiting = None

    def blocking_reason(self) -> Optional[str]:
        """``None`` when the engine-wide slot is free, else why it is busy."""
        if self._assessments is None:
            return ("lifecycle slot state unavailable (no M7C assessment "
                    "manager configured); fail closed")
        try:
            assessment_reason = self._assessments.blocking_reason()
        except Exception as exc:  # provider failed -> cannot determine state
            return f"lifecycle slot state unavailable: {exc}; fail closed"

        if self._pending_id is not None:
            return (f"engine-wide slot busy: selected proposal {self._pending_id} "
                    "registered and pending human decision")
        if assessment_reason:
            return f"engine-wide slot busy: {assessment_reason}"
        if self._awaiting is not None:
            try:
                assessment = self._assessments.assessment_for(self._awaiting.intent)
            except Exception as exc:
                return f"lifecycle slot state unavailable: {exc}; fail closed"
            if assessment is None:
                return ("engine-wide slot busy: CONVERGED application "
                        f"{self._awaiting.decision_id} awaiting M7C assessment")
            if assessment.active:
                return ("engine-wide slot busy: post-change assessment active for "
                        f"{self._awaiting.intent}")
            # Terminal/abandoned assessment for the applied change -> release.
            self._awaiting = None
        return None

    def state(self) -> Dict[str, Any]:
        """Auditable slot state (for the arbitration evidence epoch)."""
        return {
            "pending_decision_id": self._pending_id,
            "awaiting_assessment": (
                getattr(self._awaiting, "decision_id", None)
                if self._awaiting is not None else None
            ),
            "assessment_manager": self._assessments is not None,
        }


@dataclass(frozen=True)
class ArbitrationConfig:
    """Arbitration configuration: engine identity + visible actuator priority.

    ``priority`` is an explicit override of the configured stable actuator order.
    When ``None`` the deterministic :data:`DEFAULT_ACTUATOR_PRIORITY` for the
    engine is used. Any eligible intent not named in the order is appended in a
    deterministic lexical order, and the effective order is recorded — so the
    order is always visible and reproducible.
    """

    engine: str = "vllm"
    priority: Optional[Tuple[str, ...]] = None

    def effective_priority(self, intents: Sequence[str]) -> Tuple[str, ...]:
        base = (
            self.priority
            if self.priority is not None
            else DEFAULT_ACTUATOR_PRIORITY.get(self.engine, ())
        )
        ordered: List[str] = []
        for intent in base:
            if intent in intents and intent not in ordered:
                ordered.append(intent)
        for intent in sorted(i for i in intents if i not in ordered):
            ordered.append(intent)
        return tuple(ordered)


@dataclass(frozen=True)
class ArbitrationActuator:
    """One actuator bound to its own policy and :class:`SupervisorConfig`.

    The policy is actuator-specific; there is no generic algorithm. When
    ``policy`` is ``None`` the coverage-selected policy (``policy_for``) is used.
    The config carries the objective, window, sample floor and guardrails for
    that actuator.
    """

    intent: str
    internal_key: str
    config: SupervisorConfig
    policy: Optional[Any] = None


@dataclass(frozen=True)
class ActuatorEvaluation:
    """Per-actuator arbitration diagnostic (candidate or not)."""

    intent: str
    internal_key: str
    support_status: str
    considered: bool
    decision: str
    effect: Optional[str]
    direction_supported: bool
    current_value: Optional[ActuatorValue]
    proposed_value: Optional[ActuatorValue]
    transition_type: Optional[str]
    rank: int
    is_candidate: bool
    selected: bool
    lost_reason: Optional[str]
    reasons: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "internal_key": self.internal_key,
            "support_status": self.support_status,
            "considered": self.considered,
            "decision": self.decision,
            "effect": self.effect,
            "direction_supported": self.direction_supported,
            "current_value": self.current_value,
            "proposed_value": self.proposed_value,
            "transition_type": self.transition_type,
            "rank": self.rank,
            "is_candidate": self.is_candidate,
            "selected": self.selected,
            "lost_reason": self.lost_reason,
            "reasons": list(self.reasons),
        }


@dataclass
class ArbitrationRecord:
    """Deterministic, auditable arbitration result (not a DesiredState)."""

    arbitration_id: str
    outcome: str
    engine: str
    instance_id: Optional[str]
    desired_revision: Optional[int]
    evidence: Dict[str, Any]
    priority: Tuple[str, ...]
    considered: Tuple[str, ...]
    evaluations: Tuple[ActuatorEvaluation, ...]
    candidates: Tuple[str, ...]
    selected: Optional[ActuatorEvaluation]
    selected_record: Optional[DecisionRecord]
    reasons: Tuple[str, ...]
    mutation_applied: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "arbitration_id": self.arbitration_id,
            "outcome": self.outcome,
            "engine": self.engine,
            "instance_id": self.instance_id,
            "desired_revision": self.desired_revision,
            "evidence": dict(self.evidence),
            "priority": list(self.priority),
            "considered": list(self.considered),
            "evaluations": [e.to_dict() for e in self.evaluations],
            "candidates": list(self.candidates),
            "selected": self.selected.to_dict() if self.selected else None,
            "selected_decision": (
                self.selected_record.to_dict() if self.selected_record else None
            ),
            "reasons": list(self.reasons),
            "mutation_applied": self.mutation_applied,
        }


def combine_gates(
    *gates: Optional[Callable[[], Optional[str]]],
) -> Callable[[], Optional[str]]:
    """Compose arbitration gates; the first non-``None`` reason wins.

    Typical composition: the M7C assessment gate plus a check for a pending M7B
    application (the engine-wide policy-change slot).
    """

    def _combined() -> Optional[str]:
        for gate in gates:
            if gate is None:
                continue
            reason = gate()
            if reason:
                return reason
        return None

    return _combined


class _EpochControl:
    """Freeze the capability + config snapshot for one evidence epoch.

    All policies in one arbitration call read this exact snapshot, so they share
    the same desired revision and capability bounds (determinism + epoch
    binding). It is read-only: it exposes no mutation path.
    """

    def __init__(self, control: Any):
        self._capabilities = list(control.capabilities() or [])
        self._config = control.get_config() or {}

    def capabilities(self) -> List[Dict[str, Any]]:
        return list(self._capabilities)

    def get_config(self) -> Dict[str, Any]:
        return self._config


class MultiActuatorArbitrator:
    """Deterministic multi-actuator arbitrator over supported policies."""

    def __init__(
        self,
        config: ArbitrationConfig,
        control: Any,
        actuators: Sequence[ArbitrationActuator],
        *,
        slot: Optional[PolicyChangeSlot],
        gate: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.config = config
        self.control = control
        self.actuators = tuple(actuators)
        # The engine-wide lifecycle slot is a required invariant: normal M7E
        # construction cannot omit it. A ``None`` slot fails closed (DEFERRED).
        self.slot = slot
        self.gate = gate

    # ------------------------------------------------------------------ arbitrate
    def arbitrate(
        self,
        samples: Sequence[Tuple[float, Any]],
        *,
        now: Optional[float] = None,
    ) -> ArbitrationRecord:
        """Evaluate every eligible policy and select at most one candidate.

        Reads only the control plane (through ``_EpochControl``); never mutates.
        """
        now = time.monotonic() if now is None else now
        engine = self.config.engine
        epoch = _EpochControl(self.control)
        coverage = {e["intent"]: e for e in policy_coverage(engine)}
        priority = self.config.effective_priority([a.intent for a in self.actuators])
        by_intent = {a.intent: a for a in self.actuators}

        config_payload = epoch.get_config() or {}
        desired_revision = config_payload.get("desired_revision")
        instance_id = samples[-1][1].instance_id if samples else None
        slot_reason = self._slot_reason()
        gate_reason = self._gate_reason()
        deferred_reason = slot_reason or gate_reason

        evaluations: List[ActuatorEvaluation] = []
        records: Dict[str, DecisionRecord] = {}

        for rank, intent in enumerate(priority):
            spec = by_intent[intent]
            entry = coverage.get(intent)
            status = entry["status"] if entry else "ENGINE_UNSUPPORTED"
            if status != "POLICY_SUPPORTED":
                evaluations.append(self._ineligible(
                    spec, rank, status,
                    f"not eligible: {status} is not POLICY_SUPPORTED",
                ))
                continue
            policy = spec.policy or policy_for(intent, spec.config)
            if policy is None:
                evaluations.append(self._ineligible(
                    spec, rank, status,
                    "not eligible: no policy wired for a supported intent",
                ))
                continue

            supervisor = Supervisor(
                spec.config, epoch, policy=policy, intent=intent,
                internal_key=spec.internal_key,
            )
            record = supervisor.observe(samples, now=now)
            records[intent] = record
            effect = record.effect
            direction_supported = self._direction_supported(entry, effect)
            is_candidate = (
                record.decision in _PROPOSAL_DECISIONS
                and record.proposed_value is not None
                and record.current_value is not None
                and direction_supported
            )
            lost_reason = None
            if record.decision in _PROPOSAL_DECISIONS and not is_candidate:
                if effect not in _ARBITRATION_EFFECTS:
                    lost_reason = (
                        f"not eligible: effect {effect} is not a supported "
                        "arbitration effect (manual-only)"
                    )
                else:
                    direction = (entry.get("direction_support") or {}).get(
                        effect, "not supported"
                    )
                    lost_reason = (
                        f"not eligible: direction {effect} is {direction} "
                        "(not POLICY_SUPPORTED)"
                    )
            evaluations.append(ActuatorEvaluation(
                intent=intent,
                internal_key=spec.internal_key,
                support_status=status,
                considered=True,
                decision=record.decision,
                effect=effect,
                direction_supported=direction_supported,
                current_value=record.current_value,
                proposed_value=record.proposed_value,
                transition_type=record.transition_type,
                rank=rank,
                is_candidate=is_candidate,
                selected=False,
                lost_reason=lost_reason,
                reasons=tuple(record.reasons),
            ))

        candidate_evals = [e for e in evaluations if e.is_candidate]
        reasons: List[str] = []
        selected: Optional[ActuatorEvaluation] = None
        selected_record: Optional[DecisionRecord] = None

        record_revisions = {r.desired_revision for r in records.values()}
        epoch_inconsistent = len(record_revisions) > 1 or (
            desired_revision is not None
            and any(
                rev is not None and rev != desired_revision
                for rev in record_revisions
            )
        )

        if deferred_reason:
            outcome = ArbitrationOutcome.DEFERRED
            reasons.append(f"deferred: {deferred_reason}")
        elif epoch_inconsistent:
            outcome = ArbitrationOutcome.DEFERRED
            reasons.append(
                "deferred: inconsistent evidence epoch (desired_revision differs "
                "across actuator evaluations)"
            )
        elif not candidate_evals:
            outcome = ArbitrationOutcome.NO_CANDIDATE
            reasons.append(
                "no candidate: every considered supported actuator returned HOLD"
            )
        else:
            outcome = ArbitrationOutcome.SELECTED
            ordered = sorted(
                candidate_evals,
                key=lambda e: (_EFFECT_ORDER[e.effect], e.rank, e.intent,
                               e.internal_key),
            )
            selected = ordered[0]
            selected_record = records[selected.intent]
            reasons.append(
                f"selected {selected.intent} ({selected.decision} "
                f"{selected.current_value}->{selected.proposed_value}, "
                f"effect={selected.effect}) by effect priority then configured "
                "actuator order"
            )

        final_evals = tuple(
            self._finalize_eval(e, selected) for e in evaluations
        )
        selected_final = next((e for e in final_evals if e.selected), None)
        candidates = tuple(
            e.intent for e in final_evals if e.is_candidate
        )
        considered = tuple(e.intent for e in final_evals if e.considered)

        evidence = {
            "now": now,
            "engine": engine,
            "instance_id": instance_id,
            "desired_revision": desired_revision,
            "sample_count": len(samples),
            "first_sample_ts": samples[0][0] if samples else None,
            "last_sample_ts": samples[-1][0] if samples else None,
            "window_s": {a.intent: a.config.window_s for a in self.actuators},
            "lifecycle_slot": self._slot_state(),
            "lifecycle_slot_reason": slot_reason,
            "capability_snapshot": {
                (c.get("intent") or c.get("internal_key")): {
                    "min": c.get("min"),
                    "max": c.get("max"),
                    "transition": c.get("transition"),
                    "supported_on_selected_engine": c.get(
                        "supported_on_selected_engine"
                    ),
                }
                for c in epoch.capabilities()
            },
        }

        return ArbitrationRecord(
            arbitration_id=self._arbitration_id(
                engine, desired_revision, now, priority, candidate_evals, selected,
            ),
            outcome=outcome.value,
            engine=engine,
            instance_id=instance_id,
            desired_revision=desired_revision,
            evidence=evidence,
            priority=priority,
            considered=considered,
            evaluations=final_evals,
            candidates=candidates,
            selected=selected_final,
            selected_record=selected_record if selected_final else None,
            reasons=tuple(reasons),
            mutation_applied=False,
        )

    # -------------------------------------------------------------------- helpers
    def _slot_reason(self) -> Optional[str]:
        """Engine-wide lifecycle-slot gate (fail closed if undeterminable)."""
        if self.slot is None:
            return "lifecycle slot unavailable (fail closed)"
        try:
            return self.slot.blocking_reason()
        except Exception as exc:
            return f"lifecycle slot state unavailable: {exc} (fail closed)"

    def _slot_state(self) -> Optional[Dict[str, Any]]:
        if self.slot is None:
            return None
        try:
            return self.slot.state()
        except Exception as exc:
            return {"unavailable": str(exc)}

    def register_selected(
        self,
        application: Any,
        arbitration: ArbitrationRecord,
        *,
        now: Optional[float] = None,
    ) -> Optional[str]:
        """Register a selected candidate into M7B and claim the engine slot.

        The slot begins busy **when the selected proposal is registered** (before
        any human decision, mutation or assessment). Returns the M7B decision id,
        or ``None`` if the arbitration selected nothing.
        """
        record = arbitration.selected_record
        if record is None:
            return None
        did = application.register(record, now=now)
        if did is not None and self.slot is not None:
            self.slot.claim(did)
        return did

    def note_application(self, application: Any) -> None:
        """Forward an M7B application outcome to the engine-wide slot."""
        if self.slot is not None:
            self.slot.note_application(application)

    def _gate_reason(self) -> Optional[str]:
        if self.gate is None:
            return None
        try:
            return self.gate()
        except Exception as exc:  # fail closed
            return f"arbitration gate error: {exc}"

    @staticmethod
    def _direction_supported(
        entry: Dict[str, Any], effect: Optional[str]
    ) -> bool:
        if effect not in _ARBITRATION_EFFECTS:
            return False
        direction = entry.get("direction_support")
        if direction is None:
            return entry.get("status") == "POLICY_SUPPORTED"
        return direction.get(effect) == "POLICY_SUPPORTED"

    @staticmethod
    def _ineligible(
        spec: ArbitrationActuator, rank: int, status: str, reason: str
    ) -> ActuatorEvaluation:
        return ActuatorEvaluation(
            intent=spec.intent,
            internal_key=spec.internal_key,
            support_status=status,
            considered=False,
            decision="NOT_EVALUATED",
            effect=None,
            direction_supported=False,
            current_value=None,
            proposed_value=None,
            transition_type=None,
            rank=rank,
            is_candidate=False,
            selected=False,
            lost_reason=reason,
            reasons=(),
        )

    @staticmethod
    def _finalize_eval(
        evaluation: ActuatorEvaluation,
        selected: Optional[ActuatorEvaluation],
    ) -> ActuatorEvaluation:
        if not evaluation.is_candidate or selected is None:
            return evaluation
        if (
            evaluation.intent == selected.intent
            and evaluation.internal_key == selected.internal_key
        ):
            return replace(evaluation, selected=True)
        return replace(
            evaluation,
            lost_reason=MultiActuatorArbitrator._lost_reason(evaluation, selected),
        )

    @staticmethod
    def _lost_reason(
        candidate: ActuatorEvaluation, selected: ActuatorEvaluation
    ) -> str:
        if _EFFECT_ORDER[candidate.effect] != _EFFECT_ORDER[selected.effect]:
            return (
                f"lost effect priority: {candidate.effect} ranks below selected "
                f"{selected.effect}"
            )
        if candidate.rank != selected.rank:
            return (
                f"lost configured actuator order: rank {candidate.rank} is after "
                f"selected rank {selected.rank} (same effect {candidate.effect})"
            )
        return (
            "lost deterministic tie-break: same effect and rank; "
            f"{candidate.intent}/{candidate.internal_key} sorts after the selected "
            "actuator"
        )

    @staticmethod
    def _arbitration_id(
        engine: str,
        desired_revision: Optional[int],
        now: float,
        priority: Tuple[str, ...],
        candidate_evals: Sequence[ActuatorEvaluation],
        selected: Optional[ActuatorEvaluation],
    ) -> str:
        material = "|".join([
            "M7E",
            engine,
            str(desired_revision),
            f"{now:.6f}",
            ",".join(priority),
            ",".join(
                f"{e.intent}:{e.decision}:{e.current_value}:{e.proposed_value}:{e.effect}"
                for e in candidate_evals
            ),
            f"{selected.intent}:{selected.proposed_value}" if selected else "NONE",
        ])
        return sha256(material.encode("utf-8")).hexdigest()[:16]
