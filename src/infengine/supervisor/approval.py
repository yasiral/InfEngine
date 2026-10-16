"""M7B: human-approved application of an M7A PROPOSE decision.

M7A is frozen: its :class:`~infengine.supervisor.supervisor.Supervisor` stays
OBSERVE_ONLY and its policy semantics are unchanged. M7B adds the *only* path
by which a proposed ``runtime_concurrency`` change may reach the engine — an
explicit human ``APPROVE`` operation — and nothing else:

  * **No autonomous mutation.** :meth:`SupervisorApplication.register` records a
    PROPOSE decision for review; only :meth:`~SupervisorApplication.approve`
    applies it, and only after full revalidation.
  * **Binding.** An approval references the exact proposal/decision ID and binds
    to that proposal's recorded ``desired_revision``, ``current_value`` and
    ``proposed_value`` (optionally re-asserted in the request).
  * **Fail closed.** Before any mutation the proposal is revalidated against the
    live control state: desired revision unchanged, observed/current policy still
    matches, capability/bounds still permit the proposed value, the proposal has
    not expired, and the control state is converged. Any stale/conflicting
    approval (including an ``If-Match`` conflict at apply time) is recorded
    ``STALE`` with zero engine mutation.
  * **Evidence revalidation.** Immediately before mutation, a FRESH M7A
    evaluation over current telemetry must independently reproduce the exact
    proposal (same intent, direction, current value, proposed value, with the
    required guardrails satisfied). If it returns HOLD/opposite/different target,
    insufficient telemetry, an unavailable required guardrail, or otherwise no
    longer supports the proposal, the result is ``STALE`` ("no longer eligible")
    with zero mutation. M7A policy semantics are not modified.
  * **Pending lifecycle.** A registered actionable proposal is NOT cleared by a
    later HOLD decision (M7A intentionally emits HOLD during its proposal
    cooldown). It remains pending until APPROVE/REJECT, TTL expiry, a terminal
    application outcome, or explicit refresh. A fresh actionable proposal reuses
    the existing decision ID only when it is the same request (same intent/key,
    direction, current and proposed value, and ``desired_revision``) AND the
    pending proposal is still within TTL; otherwise the old decision ID is
    retired (``STALE``: ``EXPIRED`` or "binding changed") with zero mutation and
    the fresh proposal becomes the new pending decision. Repeated equivalent
    observations therefore keep one decision ID, while a newer revision or an
    aged-out proposal gets a fresh ID — no duplicate mutation opportunity and no
    erasing of a proposal before the human acts.
  * **Single-use / idempotent.** A decision is applied at most once. Once a
    decision reaches a terminal disposition (``CONVERGED`` / ``APPLY_FAILED`` /
    ``STALE`` / ``REJECTED``), any later approval/rejection for that same
    ``decision_id`` performs zero mutation (a duplicate APPROVE, an APPROVE
    after REJECT, or two competing approvals cause at most one PATCH). The
    ``If-Match`` check remains the final race barrier.
  * **One mutation path.** Application uses only the existing Control Contract →
    conditional ``If-Match`` → DesiredState → Reconciler → EngineAdapter path
    (``control.patch_config(intent, if_match=revision)``); there is no second
    mutation mechanism.
  * **Operational cooldown** begins only after an approved change is successfully
    VERIFIED/CONVERGED. A human ``REJECT`` performs zero mutation and does not
    start the cooldown; neither do ``STALE`` / ``APPLY_FAILED`` outcomes.

Records: an :class:`ApplicationRecord` is produced for every approve/reject and
carries the terminal status (``CONVERGED`` / ``APPLY_FAILED`` / ``STALE`` /
``REJECTED``), the ``stages`` (including ``APPROVED`` when the approval passed
revalidation), revisions before/after, requested/observed values, and the
cooldown boundary.

Scope: ``runtime_concurrency`` only. No auto-apply, no additional actuator, no
post-change KEEP/REVERT controller (M7C), no fleet/routing/dashboard/admission
layer, no LLM/ML/RL.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

from .policy import ActuatorValue, DecisionKind
from .supervisor import INTERNAL_KEY, INTENT, DecisionRecord

_PROPOSE_KINDS = (
    DecisionKind.PROPOSE_UP.value,
    DecisionKind.PROPOSE_DOWN.value,
    DecisionKind.PROPOSE_CHANGE.value,
)


class ApprovalControlView(Protocol):
    """Control-plane surface M7B needs (read + the existing conditional PATCH).

    Satisfied by ``ControlPlaneClient`` (``experiment.py``): ``capabilities()``,
    ``get_config()`` and ``patch_config(desired, if_match)``. Application uses
    exclusively ``patch_config`` (the existing Control Contract → If-Match →
    Reconciler path).
    """

    def capabilities(self) -> List[Dict[str, Any]]:
        ...

    def get_config(self) -> Dict[str, Any]:
        ...

    def patch_config(
        self, desired: Dict[str, Any], if_match: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        ...


class FreshEvidenceProvider(Protocol):
    """Supplies a FRESH M7A evaluation from current fresh telemetry.

    Called immediately before an approved mutation. It must run the frozen M7A
    supervisor/policy over the current telemetry window (typically a fresh
    ``Supervisor`` instance, so M7A's proposal-emission cooldown does not mask a
    still-valid proposal) and return the resulting ``DecisionRecord`` — or
    ``None`` when telemetry is insufficient. M7B never applies a proposal unless
    this fresh evaluation independently reproduces it.
    """

    def __call__(self) -> Optional[DecisionRecord]:
        ...


class ApplicationStatus(str, Enum):
    """Lifecycle outcomes of a human-approved application attempt.

    ``DEFERRED`` is **non-terminal**: the proposal was blocked by a temporary gate
    (e.g. an active M7C post-change assessment) and remains pending for a later
    approval. ``STALE`` is terminal and reserved for genuinely invalid/obsolete
    proposals.
    """

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    STALE = "STALE"
    APPLY_FAILED = "APPLY_FAILED"
    CONVERGED = "CONVERGED"
    DEFERRED = "DEFERRED"


@dataclass
class ApplicationConfig:
    """All M7B horizons/thresholds (configuration, no hidden constants)."""

    proposal_ttl_s: float = 300.0
    operational_cooldown_s: float = 60.0


@dataclass
class ApprovalRequest:
    """A human approval/rejection bound to one exact proposal decision ID."""

    decision_id: str
    expected_desired_revision: Optional[int] = None
    expected_current_value: Optional[ActuatorValue] = None
    expected_proposed_value: Optional[ActuatorValue] = None
    approver: Optional[str] = None


@dataclass
class ApplicationRecord:
    """Complete result of one approve/reject against a ``DecisionRecord``."""

    decision_id: str
    intent: str
    internal_key: str
    status: str
    stages: List[str]
    proposal_decision: str
    proposal_decided_at: str
    decided_at: str
    current_value: Optional[ActuatorValue]
    requested_value: Optional[ActuatorValue]
    observed_value: Optional[ActuatorValue]
    desired_revision_before: Optional[int]
    desired_revision_after: Optional[int]
    reasons: List[str]
    mutation_attempted: bool
    applied: bool
    cooldown_until: Optional[float]
    approver: Optional[str] = None
    # M7C: the fresh M7A evaluation performed immediately before the approved
    # mutation (the immutable pre-change baseline), captured verbatim.
    baseline_evidence: Optional[Dict[str, Any]] = None
    # M7D2: the actuator-specific policy effect of the applied change
    # (MORE_PERMISSIVE / MORE_CONSERVATIVE), preserved for assessment.
    policy_effect: Optional[str] = None
    # M13-9: multi-control extension (None for the single-control lifecycle, so
    # existing records are byte-for-byte unchanged).
    policy: Optional[Dict[str, Any]] = None
    normalized_policy: Optional[Dict[str, Any]] = None
    current_state: Optional[Dict[str, Any]] = None
    observed_state: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "intent": self.intent,
            "internal_key": self.internal_key,
            "status": self.status,
            "stages": list(self.stages),
            "proposal_decision": self.proposal_decision,
            "proposal_decided_at": self.proposal_decided_at,
            "decided_at": self.decided_at,
            "current_value": self.current_value,
            "requested_value": self.requested_value,
            "observed_value": self.observed_value,
            "desired_revision_before": self.desired_revision_before,
            "desired_revision_after": self.desired_revision_after,
            "reasons": list(self.reasons),
            "mutation_attempted": self.mutation_attempted,
            "applied": self.applied,
            "cooldown_until": self.cooldown_until,
            "approver": self.approver,
            "baseline_evidence": self.baseline_evidence,
            "policy_effect": self.policy_effect,
            "policy": self.policy,
            "normalized_policy": self.normalized_policy,
            "current_state": self.current_state,
            "observed_state": self.observed_state,
        }


def decision_id(record: DecisionRecord) -> str:
    """Deterministic ID binding an approval to one exact proposal decision."""
    material = "|".join(
        [
            record.intent,
            record.internal_key,
            record.decision,
            str(record.current_value),
            str(record.proposed_value),
            str(record.desired_revision),
            record.decided_at,
        ]
    )
    return sha256(material.encode("utf-8")).hexdigest()[:16]


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _epoch(iso: str) -> float:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


class SupervisorApplication:
    """Explicit, human-approved application of M7A ``runtime_concurrency`` proposals."""

    def __init__(
        self,
        config: ApplicationConfig,
        control: ApprovalControlView,
        *,
        fresh_evidence: Optional[FreshEvidenceProvider] = None,
        application_gate: Optional[Callable[[], Optional[str]]] = None,
        intent: str = INTENT,
        internal_key: str = INTERNAL_KEY,
    ):
        self.config = config
        self.control = control
        self.fresh_evidence = fresh_evidence
        # M7C: optional gate; when it returns a reason, approval is blocked with
        # zero mutation and the pending proposal is preserved (e.g. while a
        # post-change assessment is active).
        self.application_gate = application_gate
        self.intent = intent
        self.internal_key = internal_key
        self._pending: Optional[DecisionRecord] = None
        self._pending_id: Optional[str] = None
        self._cooldown_until: Optional[float] = None
        self._history: List[ApplicationRecord] = []
        # decision_id -> terminal ApplicationRecord (single-use enforcement)
        self._terminal: Dict[str, ApplicationRecord] = {}

    # ------------------------------------------------------------------ review
    def pending_id(self) -> Optional[str]:
        return self._pending_id

    def history(self) -> List[ApplicationRecord]:
        return list(self._history)

    def operational_cooldown_until(self) -> Optional[float]:
        return self._cooldown_until

    def in_operational_cooldown(self, now: Optional[float] = None) -> bool:
        if self._cooldown_until is None:
            return False
        now = time.time() if now is None else now
        return now < self._cooldown_until

    def register(
        self, record: DecisionRecord, *, now: Optional[float] = None
    ) -> Optional[str]:
        """Offer an observation to the pending-proposal lifecycle.

        A HOLD decision (or any non-actionable/non-``runtime_concurrency``
        record) is ignored: it never clears an existing pending proposal. An
        actionable PROPOSE decision:

          * is ignored if its decision ID is already terminal (single-use);
          * **reuses the existing pending decision ID** only when it is the same
            request — same intent/internal key, direction, current and proposed
            value, **same ``desired_revision``**, and the pending proposal is
            still within ``proposal_ttl_s``;
          * otherwise the old pending decision is **retired** (``STALE``:
            ``EXPIRED`` when past TTL, else "binding changed") with zero
            mutation, and the fresh proposal becomes the new pending decision.

        The pending decision remains available until APPROVE/REJECT, TTL expiry,
        a terminal application outcome, or supersession. Returns the actionable
        decision ID, or ``None``. Registration never mutates the engine.
        """
        now = time.time() if now is None else now

        actionable = (
            record.decision in _PROPOSE_KINDS
            and not record.mutation_applied
            and record.intent == self.intent
            and record.internal_key == self.internal_key
            and record.proposed_value is not None
            and record.current_value is not None
        )
        if not actionable:
            # Normal HOLD (e.g. M7A proposal cooldown) or non-actionable record:
            # leave any pending proposal untouched.
            return None

        did = decision_id(record)
        if did in self._terminal:
            return None  # single-use: terminal decision cannot be re-registered

        if self._pending is not None:
            expired = (
                now - _epoch(self._pending.decided_at)
            ) > self.config.proposal_ttl_s
            if expired:
                self._retire(
                    self._pending,
                    "pending proposal EXPIRED (TTL exceeded); refreshing with a "
                    "fresh decision",
                )
            elif self._equivalent(record, self._pending):
                return self._pending_id
            else:
                self._retire(
                    self._pending,
                    "pending proposal superseded (binding changed) by newer "
                    f"actionable proposal {did} "
                    f"({record.decision} {record.current_value}->{record.proposed_value}, "
                    f"revision {self._pending.desired_revision}->{record.desired_revision})",
                )

        self._pending = record
        self._pending_id = did
        return did

    @staticmethod
    def _equivalent(a: DecisionRecord, b: DecisionRecord) -> bool:
        """Whether two proposals are the same actionable request.

        Requires the same intent/internal key, direction, current and proposed
        value, and the same ``desired_revision`` (a proposal valid under a newer
        revision is NOT the same request).
        """
        return (
            a.intent == b.intent
            and a.internal_key == b.internal_key
            and a.decision == b.decision
            and a.current_value == b.current_value
            and a.proposed_value == b.proposed_value
            and a.desired_revision == b.desired_revision
        )

    def _retire(self, record: DecisionRecord, reason: str) -> ApplicationRecord:
        """Retire a decision ID (TTL expiry or supersession) with zero mutation."""
        did = decision_id(record)
        application = ApplicationRecord(
            decision_id=did,
            intent=record.intent,
            internal_key=record.internal_key,
            status=ApplicationStatus.STALE.value,
            stages=["STALE"],
            proposal_decision=record.decision,
            proposal_decided_at=record.decided_at,
            decided_at=_iso_now(),
            current_value=record.current_value,
            requested_value=record.proposed_value,
            observed_value=None,
            desired_revision_before=record.desired_revision,
            desired_revision_after=record.desired_revision,
            reasons=[f"{reason}; single-use: zero mutation"],
            mutation_attempted=False,
            applied=False,
            cooldown_until=self._cooldown_until,
            approver=None,
            policy_effect=record.effect,
        )
        self._terminal[did] = application
        self._history.append(application)
        return application

    # ------------------------------------------------------------------ reject
    def reject(
        self, request: ApprovalRequest, *, now: Optional[float] = None
    ) -> ApplicationRecord:
        """Human rejection: zero mutation, no operational cooldown."""
        now = time.time() if now is None else now
        if request.decision_id and request.decision_id in self._terminal:
            return self._replay(request, self._terminal[request.decision_id], now)
        if self._pending is None or request.decision_id != self._pending_id:
            return self._finalize(
                record=None, status=ApplicationStatus.STALE, stages=["STALE"],
                reasons=[
                    "no pending proposal matches decision id "
                    f"{request.decision_id!r}; rejection not applied"
                ],
                mutation_attempted=False, applied=False, observed=None,
                cooldown_until=self._cooldown_until, now=now,
                approver=request.approver, before=None, after=None,
            )
        pending = self._pending
        return self._finalize(
            record=pending, status=ApplicationStatus.REJECTED, stages=["REJECTED"],
            reasons=[f"human rejected proposal {self._pending_id}"],
            mutation_attempted=False, applied=False, observed=None,
            cooldown_until=self._cooldown_until, now=now,
            approver=request.approver,
            before=pending.desired_revision, after=pending.desired_revision,
        )

    # ----------------------------------------------------------------- approve
    def approve(
        self, request: ApprovalRequest, *, now: Optional[float] = None
    ) -> ApplicationRecord:
        """Apply a proposed change, but only after full revalidation.

        Returns an :class:`ApplicationRecord`; the engine is mutated only on the
        single successful path (revalidation passes → conditional PATCH →
        verified convergence). Any failure is recorded with zero mutation.
        """
        now = time.time() if now is None else now
        if request.decision_id and request.decision_id in self._terminal:
            return self._replay(request, self._terminal[request.decision_id], now)
        if self._pending is None or request.decision_id != self._pending_id:
            return self._finalize(
                record=None, status=ApplicationStatus.STALE, stages=["STALE"],
                reasons=[
                    "no pending proposal matches decision id "
                    f"{request.decision_id!r}"
                ],
                mutation_attempted=False, applied=False, observed=None,
                cooldown_until=self._cooldown_until, now=now,
                approver=request.approver, before=None, after=None,
            )
        pending = self._pending

        # --- assessment gate (M7C): mutation waits while an assessment is active
        if self.application_gate is not None:
            try:
                gate_reason = self.application_gate()
            except Exception as exc:
                gate_reason = f"application gate error: {exc}"
            if gate_reason:
                return self._blocked(
                    pending, f"application blocked: {gate_reason}", now,
                    request.approver,
                )

        # --- binding: request must agree with the recorded proposal ---------
        binding = self._binding_reasons(request, pending)
        if binding:
            return self._stale(pending, binding, now, request.approver)

        # --- expiry --------------------------------------------------------
        age = now - _epoch(pending.decided_at)
        if age > self.config.proposal_ttl_s:
            return self._stale(
                pending,
                [f"proposal expired: age={age:.1f}s > ttl={self.config.proposal_ttl_s}s"],
                now, request.approver,
            )

        # --- live revalidation against the control plane -------------------
        try:
            config = self.control.get_config()
        except Exception as exc:  # unreachable control plane -> fail closed
            return self._stale(pending, [f"control read failed: {exc}"], now,
                               request.approver)
        observed = config.get("observed") or {}
        desired = config.get("desired") or {}
        revision = config.get("desired_revision")

        capacity = self._capability()
        cap_reasons = self._capability_reasons(capacity, pending)
        if cap_reasons:
            return self._stale(pending, cap_reasons, now, request.approver)

        if (
            pending.desired_revision is not None
            and revision is not None
            and revision != pending.desired_revision
        ):
            return self._stale(
                pending,
                [f"desired revision changed: {pending.desired_revision} -> {revision}"],
                now, request.approver, after=revision,
            )

        live_current = observed.get(self.internal_key)
        if live_current is None:
            live_current = desired.get(self.internal_key)
        if live_current is None or live_current != pending.current_value:
            return self._stale(
                pending,
                [
                    "observed policy changed: expected current="
                    f"{pending.current_value}, live={live_current}"
                ],
                now, request.approver, after=revision,
            )

        desired_value = desired.get(self.internal_key)
        if desired_value is not None and desired_value != live_current:
            return self._stale(
                pending,
                [f"control state not converged: desired={desired_value} != observed={live_current}"],
                now, request.approver, after=revision,
            )

        # --- fresh evidence revalidation, immediately before mutation ------
        fresh, fresh_reasons = self._fresh_evidence(pending)
        if fresh_reasons:
            return self._stale(pending, fresh_reasons, now, request.approver,
                               after=revision)
        baseline = fresh.to_dict() if fresh is not None else None

        # --- apply through the existing conditional PATCH path -------------
        # (Control Contract -> If-Match -> DesiredState -> Reconciler -> Adapter)
        try:
            status_code, payload = self.control.patch_config(
                {pending.intent: pending.proposed_value},
                if_match=pending.desired_revision,
            )
        except Exception as exc:
            return self._apply_failed(pending, [f"apply raised: {exc}"], now,
                                      revision, None, request.approver,
                                      baseline=baseline)
        payload = payload or {}
        revision_after = payload.get("desired_revision", revision)

        if status_code == 409:
            # If-Match conflict: a concurrent change; zero engine mutation.
            return self._stale(
                pending,
                ["If-Match conflict at apply time: desired revision changed concurrently"],
                now, request.approver, after=revision_after,
                mutation_attempted=True,
            )
        if status_code != 200 or payload.get("status") != "CONVERGED":
            return self._apply_failed(
                pending,
                [f"apply failed: HTTP {status_code}, status={payload.get('status')!r}"],
                now, revision_after, payload.get("observed"), request.approver,
                baseline=baseline,
            )

        observed_after = (payload.get("observed") or {}).get(self.internal_key)
        if observed_after is None or observed_after != pending.proposed_value:
            return self._apply_failed(
                pending,
                [
                    "apply not verified: observed="
                    f"{observed_after} != proposed={pending.proposed_value}"
                ],
                now, revision_after, observed_after, request.approver,
                baseline=baseline,
            )

        # Verified convergence: only now does operational cooldown begin.
        cooldown_until = now + self.config.operational_cooldown_s
        self._cooldown_until = cooldown_until
        return self._finalize(
            record=pending, status=ApplicationStatus.CONVERGED,
            stages=["APPROVED", "CONVERGED"],
            reasons=[
                "approved and verified: desired == observed =="
                f" {pending.proposed_value}"
            ],
            mutation_attempted=True, applied=True, observed=observed_after,
            cooldown_until=cooldown_until, now=now, approver=request.approver,
            before=pending.desired_revision, after=revision_after,
            baseline=baseline,
        )

    # ---------------------------------------------------------------- helpers
    def _fresh_evidence(
        self, pending: DecisionRecord
    ) -> Tuple[Optional[DecisionRecord], List[str]]:
        """Obtain the fresh M7A evaluation immediately before mutation.

        Returns ``(fresh_record, reasons)``; empty ``reasons`` means the fresh
        evaluation reproduced the exact proposal with its required guardrails
        satisfied. The fresh record is returned so the caller can persist it as
        the M7C immutable pre-change baseline.
        """
        if self.fresh_evidence is None:
            return None, [
                "no fresh evidence provider configured; refusing to apply without "
                "fresh policy revalidation"
            ]
        try:
            fresh = self.fresh_evidence()
        except Exception as exc:
            return None, [f"fresh evaluation failed: {exc}"]
        return fresh, self._fresh_supports(fresh, pending)

    def _fresh_supports(
        self, fresh: Optional[DecisionRecord], pending: DecisionRecord
    ) -> List[str]:
        if fresh is None:
            return ["no longer eligible: no fresh evaluation available"]
        reasons: List[str] = []
        if fresh.intent != pending.intent or fresh.internal_key != pending.internal_key:
            reasons.append("no longer eligible: intent/internal key changed")
        if fresh.decision != pending.decision:
            reasons.append(
                f"no longer eligible: direction changed "
                f"({fresh.decision} != {pending.decision})"
            )
        if fresh.current_value != pending.current_value:
            reasons.append(
                f"no longer eligible: current value changed "
                f"({fresh.current_value} != {pending.current_value})"
            )
        if fresh.proposed_value != pending.proposed_value:
            reasons.append(
                f"no longer eligible: proposed value changed "
                f"({fresh.proposed_value} != {pending.proposed_value})"
            )
        enabled = [g for g in fresh.guardrails if g.get("enabled")]
        if pending.decision == "PROPOSE_UP":
            for guardrail in enabled:
                if guardrail.get("required_for_scale_up") and (
                    guardrail.get("available") is not True
                    or guardrail.get("passed") is not True
                ):
                    reasons.append(
                        "no longer eligible: required guardrail "
                        f"{guardrail.get('metric')} not satisfied "
                        f"(available={guardrail.get('available')}, "
                        f"passed={guardrail.get('passed')})"
                    )
        elif pending.decision == "PROPOSE_DOWN":
            if not any(g.get("passed") is False for g in fresh.guardrails):
                reasons.append(
                    "no longer eligible: guardrail breach no longer present"
                )
        return reasons

    def _replay(
        self, request: ApprovalRequest, prior: ApplicationRecord, now: float
    ) -> ApplicationRecord:
        """Zero-mutation response for a decision that is already terminal."""
        application = ApplicationRecord(
            decision_id=request.decision_id,
            intent=prior.intent,
            internal_key=prior.internal_key,
            status=ApplicationStatus.STALE.value,
            stages=["STALE"],
            proposal_decision=prior.proposal_decision,
            proposal_decided_at=prior.proposal_decided_at,
            decided_at=_iso_now(),
            current_value=prior.current_value,
            requested_value=prior.requested_value,
            observed_value=prior.observed_value,
            desired_revision_before=prior.desired_revision_before,
            desired_revision_after=prior.desired_revision_after,
            reasons=[
                f"decision {request.decision_id} already terminal "
                f"({prior.status}); single-use: zero mutation"
            ],
            mutation_attempted=False,
            applied=False,
            cooldown_until=self._cooldown_until,
            approver=request.approver,
            policy_effect=prior.policy_effect,
        )
        self._history.append(application)
        return application

    def _blocked(
        self, pending: DecisionRecord, reason: str, now: float,
        approver: Optional[str],
    ) -> ApplicationRecord:
        """Zero-mutation, non-terminal deferral (e.g. the M7C assessment gate).

        The pending proposal is preserved (not consumed) and the decision is not
        marked terminal, so it remains approvable once the gate lifts. The
        disposition is ``DEFERRED`` (never ``STALE``, which is terminal).
        """
        application = ApplicationRecord(
            decision_id=self._pending_id or "",
            intent=pending.intent,
            internal_key=pending.internal_key,
            status=ApplicationStatus.DEFERRED.value,
            stages=["DEFERRED"],
            proposal_decision=pending.decision,
            proposal_decided_at=pending.decided_at,
            decided_at=_iso_now(),
            current_value=pending.current_value,
            requested_value=pending.proposed_value,
            observed_value=None,
            desired_revision_before=pending.desired_revision,
            desired_revision_after=pending.desired_revision,
            reasons=[reason + "; zero mutation"],
            mutation_attempted=False,
            applied=False,
            cooldown_until=self._cooldown_until,
            approver=approver,
            policy_effect=pending.effect,
        )
        self._history.append(application)
        return application

    def _binding_reasons(
        self, request: ApprovalRequest, pending: DecisionRecord
    ) -> List[str]:
        reasons: List[str] = []
        if (
            request.expected_desired_revision is not None
            and request.expected_desired_revision != pending.desired_revision
        ):
            reasons.append(
                f"approval binding mismatch: expected revision "
                f"{request.expected_desired_revision} != recorded {pending.desired_revision}"
            )
        if (
            request.expected_current_value is not None
            and pending.current_value is not None
            and request.expected_current_value != pending.current_value
        ):
            reasons.append(
                f"approval binding mismatch: expected current "
                f"{request.expected_current_value} != recorded {pending.current_value}"
            )
        if (
            request.expected_proposed_value is not None
            and pending.proposed_value is not None
            and request.expected_proposed_value != pending.proposed_value
        ):
            reasons.append(
                f"approval binding mismatch: expected proposed "
                f"{request.expected_proposed_value} != recorded {pending.proposed_value}"
            )
        return reasons

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

    def _capability_reasons(
        self, capacity: Optional[Dict[str, Any]], pending: DecisionRecord
    ) -> List[str]:
        reasons: List[str] = []
        if capacity is None:
            return ["capability unavailable for runtime_concurrency"]
        if capacity.get("operation"):
            reasons.append("capability is now a one-shot operation")
        if not capacity.get("supported_on_selected_engine", False):
            reasons.append("capability no longer supported on the selected engine")
        if capacity.get("transition") != "HOT":
            reasons.append(
                f"capability transition is now {capacity.get('transition')!r}, not HOT"
            )
        lo = capacity.get("min")
        hi = capacity.get("max")
        if isinstance(pending.proposed_value, bool):
            # Boolean actuators have no numeric bounds; do not coerce to a number.
            pass
        elif lo is None or hi is None:
            reasons.append("capability bounds unavailable")
        elif pending.proposed_value is not None and not (
            lo <= pending.proposed_value <= hi
        ):
            reasons.append(
                f"proposed value {pending.proposed_value} outside capability bounds "
                f"[{lo}, {hi}]"
            )
        return reasons

    def _stale(
        self, pending: DecisionRecord, reasons: List[str], now: float,
        approver: Optional[str], after: Optional[int] = None,
        mutation_attempted: bool = False,
    ) -> ApplicationRecord:
        return self._finalize(
            record=pending, status=ApplicationStatus.STALE, stages=["STALE"],
            reasons=reasons, mutation_attempted=mutation_attempted, applied=False,
            observed=None, cooldown_until=self._cooldown_until, now=now,
            approver=approver, before=pending.desired_revision,
            after=after if after is not None else pending.desired_revision,
        )

    def _apply_failed(
        self, pending: DecisionRecord, reasons: List[str], now: float,
        revision_after: Optional[int], observed: Optional[Any],
        approver: Optional[str],
        baseline: Optional[Dict[str, Any]] = None,
    ) -> ApplicationRecord:
        return self._finalize(
            record=pending, status=ApplicationStatus.APPLY_FAILED,
            stages=["APPROVED", "APPLY_FAILED"], reasons=reasons,
            mutation_attempted=True, applied=False,
            observed=observed,
            cooldown_until=self._cooldown_until, now=now, approver=approver,
            before=pending.desired_revision, after=revision_after,
            baseline=baseline,
        )

    def _finalize(
        self, *, record: Optional[DecisionRecord], status: ApplicationStatus,
        stages: List[str], reasons: List[str], mutation_attempted: bool,
        applied: bool, observed: Optional[int], cooldown_until: Optional[float],
        now: float, approver: Optional[str], before: Optional[int],
        after: Optional[int], baseline: Optional[Dict[str, Any]] = None,
    ) -> ApplicationRecord:
        if record is None:
            # No pending proposal: record the attempt with empty identifiers.
            did = ""
            intent = self.intent
            internal_key = self.internal_key
            proposal_decision = ""
            proposal_decided_at = ""
            current_value = None
            requested_value = None
        else:
            did = self._pending_id or ""
            intent = record.intent
            internal_key = record.internal_key
            proposal_decision = record.decision
            proposal_decided_at = record.decided_at
            current_value = record.current_value
            requested_value = record.proposed_value
        application = ApplicationRecord(
            decision_id=did,
            intent=intent,
            internal_key=internal_key,
            status=status.value,
            stages=list(stages),
            proposal_decision=proposal_decision,
            proposal_decided_at=proposal_decided_at,
            decided_at=_iso_now(),
            current_value=current_value,
            requested_value=requested_value,
            observed_value=observed,
            desired_revision_before=before,
            desired_revision_after=after,
            reasons=list(reasons),
            mutation_attempted=mutation_attempted,
            applied=applied,
            cooldown_until=cooldown_until,
            approver=approver,
            baseline_evidence=baseline,
            policy_effect=(record.effect if record is not None else None),
        )
        self._pending = None
        self._pending_id = None
        if did:
            self._terminal[did] = application
        self._history.append(application)
        return application


# ===========================================================================
# M13-9: multi-control policy proposals — an ADDITIVE extension of the SAME M7
# proposal/approval lifecycle. It reuses ApplicationConfig / ApplicationStatus /
# ApplicationRecord / the single-use + TTL + If-Match + fresh-evidence + gate
# semantics above; it does NOT introduce a second approval architecture.
# ===========================================================================
import json as _json  # noqa: E402  (kept local to the extension)

# A fresh-evidence provider for a policy proposal returns an observed-state
# snapshot ``{internal_key: value}`` immediately before mutation (or None when
# evidence is unavailable). It must confirm the captured pre-state is still
# current.
PolicyEvidenceProvider = Callable[[], Optional[Dict[str, Any]]]


@dataclass
class PolicyProposal:
    """A human-authored candidate policy (one or many controls).

    ``policy`` is keyed by the operator intent (or internal key) the control
    plane accepts; ``register`` resolves it to internal keys and captures the
    verified pre-state.
    """

    policy: Dict[str, Any]
    evidence: Optional[Dict[str, Any]] = None


@dataclass
class _PendingPolicy:
    proposal_id: str
    policy: Dict[str, Any]
    normalized_policy: Dict[str, Any]
    current_state: Dict[str, Any]
    desired_revision: Optional[int]
    created_at: str
    evidence: Optional[Dict[str, Any]] = None


def policy_id(policy: Dict[str, Any], revision: Optional[int], created_at: str) -> str:
    """Deterministic ID covering the WHOLE policy, the revision and creation
    time. Any change to any control/value yields a different ID."""
    material = "|".join([
        _json.dumps(policy, sort_keys=True, default=str),
        str(revision),
        created_at,
    ])
    return sha256(material.encode("utf-8")).hexdigest()[:16]


@dataclass
class PolicyApprovalRequest:
    """A human approval/rejection bound to one exact policy proposal."""

    decision_id: str
    expected_desired_revision: Optional[int] = None
    expected_policy: Optional[Dict[str, Any]] = None
    approver: Optional[str] = None


class SupervisorPolicyApplication:
    """Multi-control specialization of :class:`SupervisorApplication`.

    Same lifecycle (register -> human approve/reject -> fresh-evidence +
    live-state revalidation -> conditional If-Match PATCH -> whole-policy
    verification -> single-use terminal record). Application uses the existing
    multi-control policy execution path (``control.patch_config(policy, if_match)``).
    """

    def __init__(
        self,
        config: ApplicationConfig,
        control: ApprovalControlView,
        *,
        fresh_evidence: Optional[PolicyEvidenceProvider] = None,
        application_gate: Optional[Callable[[], Optional[str]]] = None,
        require_fresh_evidence: bool = True,
    ):
        self.config = config
        self.control = control
        self.fresh_evidence = fresh_evidence
        self.application_gate = application_gate
        self.require_fresh_evidence = require_fresh_evidence
        self._pending: Optional[_PendingPolicy] = None
        self._pending_id: Optional[str] = None
        self._cooldown_until: Optional[float] = None
        self._history: List[ApplicationRecord] = []
        self._terminal: Dict[str, ApplicationRecord] = {}
        self._last_reason: Optional[str] = None

    # ------------------------------------------------------------------ review
    def pending_id(self) -> Optional[str]:
        return self._pending_id

    def history(self) -> List[ApplicationRecord]:
        return list(self._history)

    def last_reason(self) -> Optional[str]:
        return self._last_reason

    def register(self, proposal: PolicyProposal, *, now: Optional[float] = None) -> Optional[str]:
        """Resolve, validate and register a candidate policy. Returns the
        proposal ID, or None when the proposal is not registrable. Never
        mutates the engine."""
        now = time.time() if now is None else now
        created = _iso_now()
        try:
            caps = self._capability_index()
            config = self.control.get_config()
        except Exception as exc:
            self._last_reason = f"control read failed: {exc}"
            return None
        normalized: Dict[str, Any] = {}
        errors: List[str] = []
        for key, value in proposal.policy.items():
            entry = caps.get(key)
            if entry is None or not entry.get("supported_on_selected_engine"):
                errors.append(f"{key}: unknown/unsupported on the selected engine")
                continue
            if entry.get("operation"):
                errors.append(f"{key}: one-shot operations are not policy controls")
                continue
            if entry.get("transition") != "HOT":
                errors.append(f"{key}: transition {entry.get('transition')} is not HOT")
                continue
            lo, hi = entry.get("min"), entry.get("max")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                if lo is not None and value < lo:
                    errors.append(f"{key}: {value} below minimum {lo}")
                if hi is not None and value > hi:
                    errors.append(f"{key}: {value} above maximum {hi}")
            normalized[entry.get("internal_key") or key] = value
        if errors or not normalized:
            self._last_reason = "; ".join(errors) or "empty policy"
            return None
        observed = config.get("observed") or {}
        revision = config.get("desired_revision")
        pending = _PendingPolicy(
            proposal_id=policy_id(proposal.policy, revision, created),
            policy=dict(proposal.policy),
            normalized_policy=normalized,
            current_state={k: observed.get(k) for k in normalized},
            desired_revision=revision,
            created_at=created,
            evidence=proposal.evidence,
        )
        if self._pending is not None:
            expired = (now - _epoch(self._pending.created_at)) > self.config.proposal_ttl_s
            if expired:
                self._retire(self._pending, "pending proposal EXPIRED (TTL exceeded)")
            elif self._equivalent(pending, self._pending):
                return self._pending_id
            else:
                self._retire(self._pending, "pending proposal superseded (policy changed)")
        self._pending = pending
        self._pending_id = pending.proposal_id
        self._last_reason = None
        return pending.proposal_id

    @staticmethod
    def _equivalent(a: _PendingPolicy, b: _PendingPolicy) -> bool:
        return a.policy == b.policy and a.desired_revision == b.desired_revision

    # ------------------------------------------------------------------ reject
    def reject(self, request: PolicyApprovalRequest, *, now: Optional[float] = None) -> ApplicationRecord:
        now = time.time() if now is None else now
        if request.decision_id and request.decision_id in self._terminal:
            return self._replay(request, self._terminal[request.decision_id], now)
        if self._pending is None or request.decision_id != self._pending_id:
            return self._finalize(
                pending=None, status=ApplicationStatus.STALE, stages=["STALE"],
                reasons=[f"no pending policy matches decision id {request.decision_id!r}"],
                mutation_attempted=False, applied=False, observed_state=None,
                cooldown_until=self._cooldown_until, now=now, approver=request.approver,
                before=None, after=None)
        pending = self._pending
        return self._finalize(
            pending=pending, status=ApplicationStatus.REJECTED, stages=["REJECTED"],
            reasons=[f"human rejected policy {self._pending_id}"],
            mutation_attempted=False, applied=False, observed_state=None,
            cooldown_until=self._cooldown_until, now=now, approver=request.approver,
            before=pending.desired_revision, after=pending.desired_revision)

    # ----------------------------------------------------------------- approve
    def approve(self, request: PolicyApprovalRequest, *, now: Optional[float] = None) -> ApplicationRecord:
        now = time.time() if now is None else now
        if request.decision_id and request.decision_id in self._terminal:
            return self._replay(request, self._terminal[request.decision_id], now)
        if self._pending is None or request.decision_id != self._pending_id:
            return self._finalize(
                pending=None, status=ApplicationStatus.STALE, stages=["STALE"],
                reasons=[f"no pending policy matches decision id {request.decision_id!r}"],
                mutation_attempted=False, applied=False, observed_state=None,
                cooldown_until=self._cooldown_until, now=now, approver=request.approver,
                before=None, after=None)
        pending = self._pending

        if self.application_gate is not None:
            try:
                gate_reason = self.application_gate()
            except Exception as exc:
                gate_reason = f"application gate error: {exc}"
            if gate_reason:
                return self._blocked(pending, f"application blocked: {gate_reason}", now,
                                     request.approver)

        binding = self._binding_reasons(request, pending)
        if binding:
            return self._stale(pending, binding, now, request.approver)

        age = now - _epoch(pending.created_at)
        if age > self.config.proposal_ttl_s:
            return self._stale(
                pending,
                [f"proposal expired: age={age:.1f}s > ttl={self.config.proposal_ttl_s}s"],
                now, request.approver)

        try:
            config = self.control.get_config()
        except Exception as exc:
            return self._stale(pending, [f"control read failed: {exc}"], now, request.approver)
        observed = config.get("observed") or {}
        desired = config.get("desired") or {}
        revision = config.get("desired_revision")

        cap_reasons = self._capability_reasons(pending)
        if cap_reasons:
            return self._stale(pending, cap_reasons, now, request.approver)
        if (pending.desired_revision is not None and revision is not None
                and revision != pending.desired_revision):
            return self._stale(
                pending,
                [f"desired revision changed: {pending.desired_revision} -> {revision}"],
                now, request.approver, after=revision)
        for key, value in pending.normalized_policy.items():
            live = observed.get(key)
            if live is None:
                live = desired.get(key)
            if live != pending.current_state.get(key):
                return self._stale(
                    pending,
                    [f"observed policy changed for {key}: expected "
                     f"{pending.current_state.get(key)}, live={live}"],
                    now, request.approver, after=revision)
            desired_value = desired.get(key)
            if desired_value is not None and desired_value != live:
                return self._stale(
                    pending,
                    [f"control state not converged: desired {desired_value} != observed {live}"],
                    now, request.approver, after=revision)

        fresh, fresh_reasons = self._fresh_evidence(pending)
        if fresh_reasons:
            return self._stale(pending, fresh_reasons, now, request.approver, after=revision)

        try:
            status_code, payload = self.control.patch_config(
                pending.normalized_policy, if_match=pending.desired_revision)
        except Exception as exc:
            return self._apply_failed(pending, [f"apply raised: {exc}"], now, revision,
                                      None, request.approver, fresh)
        payload = payload or {}
        revision_after = payload.get("desired_revision", revision)

        if status_code == 409:
            return self._stale(
                pending,
                ["If-Match conflict at apply time: desired revision changed concurrently"],
                now, request.approver, after=revision_after, mutation_attempted=True)
        if status_code != 200 or payload.get("status") != "CONVERGED":
            return self._apply_failed(
                pending,
                [f"apply failed: HTTP {status_code}, status={payload.get('status')!r}"],
                now, revision_after, payload.get("observed"), request.approver, fresh)

        observed_after = payload.get("observed") or {}
        mismatched = {
            k: observed_after.get(k)
            for k, v in pending.normalized_policy.items()
            if observed_after.get(k) != v
        }
        if mismatched:
            # A partial / unverified apply is NEVER an approved success.
            return self._apply_failed(
                pending,
                [f"whole-policy verification failed: requested {pending.normalized_policy} "
                 f"but observed {mismatched}"],
                now, revision_after, observed_after, request.approver, fresh)

        cooldown_until = now + self.config.operational_cooldown_s
        self._cooldown_until = cooldown_until
        return self._finalize(
            pending=pending, status=ApplicationStatus.CONVERGED,
            stages=["APPROVED", "CONVERGED"],
            reasons=[f"approved and verified whole policy: {pending.normalized_policy}"],
            mutation_attempted=True, applied=True, observed_state=observed_after,
            cooldown_until=cooldown_until, now=now, approver=request.approver,
            before=pending.desired_revision, after=revision_after)

    # ---------------------------------------------------------------- helpers
    def _capability_index(self) -> Dict[str, Dict[str, Any]]:
        index: Dict[str, Dict[str, Any]] = {}
        for entry in self.control.capabilities() or []:
            if entry.get("intent"):
                index[entry["intent"]] = entry
            if entry.get("internal_key"):
                index.setdefault(entry["internal_key"], entry)
        return index

    def _capability_reasons(self, pending: _PendingPolicy) -> List[str]:
        try:
            caps = self._capability_index()
        except Exception as exc:
            return [f"capability read failed: {exc}"]
        reasons: List[str] = []
        for key, value in pending.normalized_policy.items():
            entry = caps.get(key)
            if entry is None:
                reasons.append(f"{key}: capability unavailable")
                continue
            if entry.get("operation"):
                reasons.append(f"{key}: capability is now a one-shot operation")
            if not entry.get("supported_on_selected_engine"):
                reasons.append(f"{key}: no longer supported on the selected engine")
            if entry.get("transition") != "HOT":
                reasons.append(f"{key}: transition is now {entry.get('transition')!r}, not HOT")
            lo, hi = entry.get("min"), entry.get("max")
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)) and (lo is not None or hi is not None):
                if lo is not None and value < lo:
                    reasons.append(f"{key}: {value} outside bounds [{lo}, {hi}]")
                if hi is not None and value > hi:
                    reasons.append(f"{key}: {value} outside bounds [{lo}, {hi}]")
        return reasons

    def _fresh_evidence(
        self, pending: _PendingPolicy
    ) -> Tuple[Optional[Dict[str, Any]], List[str]]:
        if self.fresh_evidence is None:
            if self.require_fresh_evidence:
                return None, ["no fresh evidence provider configured; refusing to apply "
                              "without fresh revalidation"]
            return None, []
        try:
            snapshot = self.fresh_evidence()
        except Exception as exc:
            return None, [f"fresh evaluation failed: {exc}"]
        if snapshot is None:
            return None, ["no longer eligible: no fresh evidence available"]
        if not isinstance(snapshot, dict):
            return None, ["fresh evidence type not understood"]
        reasons: List[str] = []
        for key, expected in pending.current_state.items():
            if snapshot.get(key) != expected:
                reasons.append(
                    f"no longer eligible: fresh {key}={snapshot.get(key)} != captured {expected}")
        return snapshot, reasons

    def _binding_reasons(
        self, request: PolicyApprovalRequest, pending: _PendingPolicy
    ) -> List[str]:
        reasons: List[str] = []
        if (request.expected_desired_revision is not None
                and request.expected_desired_revision != pending.desired_revision):
            reasons.append(
                "approval binding mismatch: expected revision "
                f"{request.expected_desired_revision} != recorded {pending.desired_revision}")
        if request.expected_policy is not None and request.expected_policy != pending.policy:
            reasons.append(
                "approval binding mismatch: expected policy differs from the proposed "
                f"policy ({request.expected_policy} != {pending.policy})")
        return reasons

    # ------------------------------------------------------------- terminal forms
    def _retire(self, pending: _PendingPolicy, reason: str) -> ApplicationRecord:
        record = self._finalize(
            pending=pending, status=ApplicationStatus.STALE, stages=["STALE"],
            reasons=[f"{reason}; single-use: zero mutation"],
            mutation_attempted=False, applied=False, observed_state=None,
            cooldown_until=self._cooldown_until, now=time.time(), approver=None,
            before=pending.desired_revision, after=pending.desired_revision)
        self._pending = None
        self._pending_id = None
        return record

    def _replay(self, request: PolicyApprovalRequest, prior: ApplicationRecord,
                now: float) -> ApplicationRecord:
        application = ApplicationRecord(
            decision_id=request.decision_id,
            intent=prior.intent, internal_key=prior.internal_key,
            status=ApplicationStatus.STALE.value, stages=["STALE"],
            proposal_decision=prior.proposal_decision,
            proposal_decided_at=prior.proposal_decided_at,
            decided_at=_iso_now(),
            current_value=prior.current_value, requested_value=prior.requested_value,
            observed_value=prior.observed_value,
            desired_revision_before=prior.desired_revision_before,
            desired_revision_after=prior.desired_revision_after,
            reasons=[f"decision {request.decision_id} already terminal "
                     f"({prior.status}); single-use: zero mutation"],
            mutation_attempted=False, applied=False,
            cooldown_until=self._cooldown_until, approver=request.approver,
            policy=prior.policy, normalized_policy=prior.normalized_policy,
            current_state=prior.current_state, observed_state=prior.observed_state,
        )
        self._history.append(application)
        return application

    def _blocked(self, pending: _PendingPolicy, reason: str, now: float,
                 approver: Optional[str]) -> ApplicationRecord:
        return self._finalize(
            pending=pending, status=ApplicationStatus.DEFERRED, stages=["DEFERRED"],
            reasons=[reason + "; zero mutation"], mutation_attempted=False, applied=False,
            observed_state=None, cooldown_until=self._cooldown_until, now=now,
            approver=approver, before=pending.desired_revision,
            after=pending.desired_revision, retain_pending=True)

    def _stale(self, pending: _PendingPolicy, reasons: List[str], now: float,
               approver: Optional[str], after: Optional[int] = None,
               mutation_attempted: bool = False) -> ApplicationRecord:
        return self._finalize(
            pending=pending, status=ApplicationStatus.STALE, stages=["STALE"],
            reasons=reasons, mutation_attempted=mutation_attempted, applied=False,
            observed_state=None, cooldown_until=self._cooldown_until, now=now,
            approver=approver, before=pending.desired_revision,
            after=after if after is not None else pending.desired_revision)

    def _apply_failed(self, pending: _PendingPolicy, reasons: List[str], now: float,
                      revision_after: Optional[int], observed_state: Optional[Dict[str, Any]],
                      approver: Optional[str],
                      fresh: Optional[Dict[str, Any]] = None) -> ApplicationRecord:
        return self._finalize(
            pending=pending, status=ApplicationStatus.APPLY_FAILED,
            stages=["APPROVED", "APPLY_FAILED"], reasons=reasons,
            mutation_attempted=True, applied=False, observed_state=observed_state,
            cooldown_until=self._cooldown_until, now=now, approver=approver,
            before=pending.desired_revision, after=revision_after)

    def _finalize(self, *, pending: Optional[_PendingPolicy], status: ApplicationStatus,
                  stages: List[str], reasons: List[str], mutation_attempted: bool,
                  applied: bool, observed_state: Optional[Dict[str, Any]],
                  cooldown_until: Optional[float], now: float,
                  approver: Optional[str], before: Optional[int],
                  after: Optional[int], retain_pending: bool = False) -> ApplicationRecord:
        if pending is not None:
            did = pending.proposal_id
            policy = dict(pending.policy)
            normalized = dict(pending.normalized_policy)
            current_state = dict(pending.current_state)
            intent = next(iter(policy)) if len(policy) == 1 else "<policy>"
            internal_key = ",".join(sorted(normalized))
        else:
            did = self._pending_id or ""
            policy = normalized = current_state = None
            intent = "<policy>"
            internal_key = ""
        application = ApplicationRecord(
            decision_id=did, intent=intent, internal_key=internal_key,
            status=status.value, stages=list(stages),
            proposal_decision="POLICY",
            proposal_decided_at=(pending.created_at if pending is not None else ""),
            decided_at=_iso_now(),
            current_value=None, requested_value=None, observed_value=None,
            desired_revision_before=before, desired_revision_after=after,
            reasons=list(reasons), mutation_attempted=mutation_attempted, applied=applied,
            cooldown_until=cooldown_until, approver=approver,
            policy=policy, normalized_policy=normalized, current_state=current_state,
            observed_state=observed_state,
        )
        if not retain_pending:
            self._pending = None
            self._pending_id = None
        if did:
            self._terminal[did] = application
        self._history.append(application)
        return application
