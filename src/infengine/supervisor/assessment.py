"""M7C: deterministic post-change supervisory assessment (runtime_concurrency only).

After an M7B application reaches verified ``CONVERGED``, M7C creates an
assessment context bound to that exact application (its decision ID, previous
value, applied value and resulting desired revision) and to the **fresh M7A
evaluation performed immediately before the approved mutation** — the immutable
pre-change baseline. The older original proposal window is deliberately NOT used
as the baseline.

M7C never mutates the engine. It produces exactly one of:

  * ``KEEP``           — sufficient fresh post-change evidence and all configured
                         guardrails pass;
  * ``PROPOSE_REVERT`` — (UP changes) a guardrail that was passing in the
                         pre-change baseline is breached after stabilization; the
                         recommended target is the exact previous value, and it
                         performs zero mutation;
  * ``INCONCLUSIVE``   — insufficient/stale/unavailable required evidence, an
                         external revision/current-value/convergence change, a
                         baseline binding mismatch, a stabilization cooldown not
                         yet elapsed, or (DOWN changes) guardrails still breached
                         with no automatic upward-revert logic in V1.

The M7B operational cooldown is the stabilization period: no conclusion is drawn
before it elapses. Baseline/post numeric evidence and deltas are recorded for
inspection only — they are never a scoring system.

While an assessment is active for ``runtime_concurrency``, the M7B application
gate blocks another application to that intent (observation may continue),
preserving the evaluation boundary; the gate lifts when the assessment is
terminal or explicitly abandoned. A later human-approved revert must be
independently supported by a fresh M7A policy evaluation of the exact reverse
transition and applied through the existing M7B / If-Match / Reconciler path.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .approval import ApplicationRecord
from .policy import ActuatorValue, GuardrailResult, SupervisorConfig
from .signals import SignalWindow, build_signal_window
from .supervisor import INTENT, ControlReadView, DecisionRecord


class AssessmentOutcome(str, Enum):
    """The only three post-change assessments M7C V1 can produce."""

    KEEP = "KEEP"
    PROPOSE_REVERT = "PROPOSE_REVERT"
    INCONCLUSIVE = "INCONCLUSIVE"


@dataclass
class AssessmentConfig:
    """M7C-specific knobs (the M7A policy config is passed separately).

    ``max_evidence_wait_s`` bounds how long, **after the stabilization cooldown
    ends**, the assessment may wait for sufficient post-change evidence. The
    deadline is ``application.cooldown_until + max_evidence_wait_s`` — the
    stabilization period does NOT consume the evidence wait. A temporary
    inability to judge (shortage of fresh samples, a required metric not yet
    published, a transient telemetry/control read failure) keeps the assessment
    non-terminal (WAITING) until this deadline; only then does it become terminal
    ``INCONCLUSIVE``.
    """

    max_evidence_wait_s: float = 120.0


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class AssessmentRecord:
    """A post-change assessment result (terminal or a pre-cooldown wait)."""

    application_id: str
    intent: str
    internal_key: str
    outcome: str
    terminal: bool
    previous_value: Optional[ActuatorValue]
    applied_value: Optional[ActuatorValue]
    desired_revision: Optional[int]
    direction: Optional[str]
    recommended_value: Optional[ActuatorValue]
    revert_direction: Optional[str]
    policy_effect: Optional[str]
    baseline: Optional[Dict[str, Any]]
    baseline_guardrails: List[Dict[str, Any]]
    post: Dict[str, Any]
    post_guardrails: List[Dict[str, Any]]
    numeric_deltas: Dict[str, Any]
    reasons: List[str]
    assessed_at: str
    mutation_applied: bool = False
    waiting: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "application_id": self.application_id,
            "intent": self.intent,
            "internal_key": self.internal_key,
            "outcome": self.outcome,
            "terminal": self.terminal,
            "waiting": self.waiting,
            "previous_value": self.previous_value,
            "applied_value": self.applied_value,
            "desired_revision": self.desired_revision,
            "direction": self.direction,
            "recommended_value": self.recommended_value,
            "revert_direction": self.revert_direction,
            "policy_effect": self.policy_effect,
            "baseline": self.baseline,
            "baseline_guardrails": list(self.baseline_guardrails),
            "post": self.post,
            "post_guardrails": list(self.post_guardrails),
            "numeric_deltas": self.numeric_deltas,
            "reasons": list(self.reasons),
            "assessed_at": self.assessed_at,
            "mutation_applied": self.mutation_applied,
        }


def _guardrails_to_dicts(guardrails: List[GuardrailResult]) -> List[Dict[str, Any]]:
    return [g.to_dict() for g in guardrails]


class PostChangeAssessment:
    """One assessment context bound to one CONVERGED M7B application."""

    def __init__(
        self,
        application: ApplicationRecord,
        baseline: Optional[Dict[str, Any]],
        config: SupervisorConfig,
        control: ControlReadView,
        *,
        max_evidence_wait_s: float = 120.0,
        started_at: Optional[float] = None,
    ):
        self.application = application
        self.config = config
        self.control = control
        self._baseline = baseline
        self.max_evidence_wait_s = max_evidence_wait_s
        self.started_at = time.time() if started_at is None else started_at
        # The evidence timeout begins when stabilization ENDS, not at assessment
        # start: the M7B operational cooldown must not consume the evidence wait.
        stabilization_end = (
            application.cooldown_until
            if application.cooldown_until is not None
            else self.started_at
        )
        self.deadline = stabilization_end + max_evidence_wait_s
        self.application_id = application.decision_id
        self.intent = application.intent
        self.internal_key = application.internal_key
        # M7D2: assess from the actuator-specific policy effect, NOT numeric
        # UP/DOWN (for the KV watermark, an increase is MORE_CONSERVATIVE).
        self.policy_effect = application.policy_effect
        self.previous_value = application.current_value
        self.applied_value = application.requested_value
        self.desired_revision = application.desired_revision_after
        self.cooldown_until = application.cooldown_until
        if isinstance(self.previous_value, bool) or isinstance(self.applied_value, bool):
            # No implicit numeric UP/DOWN semantics for bool actuators.
            self.direction = None
        elif self.previous_value is not None and self.applied_value is not None:
            if self.applied_value > self.previous_value:
                self.direction = "UP"
            elif self.applied_value < self.previous_value:
                self.direction = "DOWN"
            else:
                self.direction = "NONE"
        else:
            self.direction = None
        self._record: Optional[AssessmentRecord] = None
        self._abandoned: Optional[str] = None

    # ------------------------------------------------------------------ state
    @property
    def active(self) -> bool:
        return self._record is None and self._abandoned is None

    @property
    def terminal(self) -> bool:
        return self._record is not None and self._record.terminal

    def record(self) -> Optional[AssessmentRecord]:
        return self._record

    def abandon(self, reason: str) -> AssessmentRecord:
        """Explicitly abandon the assessment (lifts the gate, zero mutation)."""
        self._abandoned = reason
        record = self._build(
            outcome=AssessmentOutcome.INCONCLUSIVE, terminal=True,
            reasons=[f"assessment explicitly abandoned: {reason}"],
            post={}, post_guardrails=[], deltas={},
        )
        self._record = record
        return record

    # --------------------------------------------------------------- baseline
    def _baseline_binding_reasons(self) -> List[str]:
        baseline = self._baseline
        if baseline is None:
            return ["no pre-change baseline persisted; cannot assess"]
        reasons: List[str] = []
        if baseline.get("intent") != self.intent or baseline.get("internal_key") != self.internal_key:
            reasons.append("baseline binding mismatch: intent/internal key")
        if baseline.get("current_value") != self.previous_value:
            reasons.append(
                "baseline binding mismatch: baseline current="
                f"{baseline.get('current_value')} != application previous={self.previous_value}"
            )
        if baseline.get("proposed_value") != self.applied_value:
            reasons.append(
                "baseline binding mismatch: baseline proposed="
                f"{baseline.get('proposed_value')} != application applied={self.applied_value}"
            )
        return reasons

    # ------------------------------------------------------------ assessment
    def assess(self, samples, *, now: Optional[float] = None) -> AssessmentRecord:
        """Assess the post-change window. Never mutates the engine.

        Before the stabilization cooldown elapses this returns a non-terminal
        INCONCLUSIVE "wait" record (the assessment stays active). Once a terminal
        outcome is reached it is frozen and returned on later calls.
        """
        now = time.time() if now is None else now
        if self._record is not None and self._record.terminal:
            return self._record
        if self._abandoned is not None:
            return self._record  # type: ignore[return-value]

        if self.cooldown_until is not None and now < self.cooldown_until:
            remaining = self.cooldown_until - now
            return self._waiting(
                [f"stabilization cooldown not elapsed ({remaining:.1f}s remaining)"]
            )

        binding = self._baseline_binding_reasons()
        if binding:
            return self._terminal_inconclusive(binding, now)

        try:
            config = self.control.get_config()
        except Exception as exc:  # transient read failure -> wait until deadline
            return self._temporary_or_terminal(
                [f"temporary control read failure: {exc}"], now
            )
        observed = config.get("observed") or {}
        desired = config.get("desired") or {}
        revision = config.get("desired_revision")

        if (
            self.desired_revision is not None
            and revision is not None
            and revision != self.desired_revision
        ):
            return self._terminal_inconclusive(
                [f"external desired revision change during assessment: "
                 f"{self.desired_revision} -> {revision}"],
                now,
            )

        live_current = observed.get(self.internal_key)
        if live_current is None:
            live_current = desired.get(self.internal_key)
        if live_current is None or int(live_current) != int(self.applied_value):
            return self._terminal_inconclusive(
                ["observed current value changed during assessment: expected "
                 f"{self.applied_value}, live {live_current}"],
                now,
            )
        desired_value = desired.get(self.internal_key)
        if desired_value is not None and int(desired_value) != int(live_current):
            return self._terminal_inconclusive(
                [f"control state not converged during assessment: desired="
                 f"{desired_value} != observed={live_current}"],
                now,
            )

        window = build_signal_window(
            samples, now=now, window_s=self.config.window_s,
            metrics=self.config.effective_metrics(),
        )
        insufficient = self._sufficiency_reasons(window)
        if insufficient:
            return self._temporary_or_terminal(insufficient, now, window=window)

        post_guardrails = self._evaluate_guardrails(window)
        baseline_guardrails = {
            g.get("metric"): g for g in ((self._baseline or {}).get("guardrails") or [])
        }
        breaches = [g for g in post_guardrails if g.passed is False]
        deltas = self._numeric_deltas(window)

        if self._is_permissive_change():
            new_breaches = [
                g for g in breaches
                if baseline_guardrails.get(g.metric, {}).get("passed") is True
            ]
            if new_breaches:
                return self._build(
                    outcome=AssessmentOutcome.PROPOSE_REVERT, terminal=True,
                    reasons=[
                        "guardrail was passing pre-change but breached post-change: "
                        f"{g.metric} max={g.observed} > {g.comparator} limit={g.limit} "
                        f"(recommend revert {self.applied_value} -> {self.previous_value})"
                        for g in new_breaches
                    ],
                    post=window, post_guardrails=post_guardrails, deltas=deltas,
                )
            if breaches:
                return self._build(
                    outcome=AssessmentOutcome.INCONCLUSIVE, terminal=True,
                    reasons=[
                        "guardrail(s) breached post-change but not newly (already "
                        "failing pre-change; attribution unclear): "
                        + ", ".join(g.metric for g in breaches)
                    ],
                    post=window, post_guardrails=post_guardrails, deltas=deltas,
                )
            return self._build(
                outcome=AssessmentOutcome.KEEP, terminal=True,
                reasons=[
                    "post-change evidence sufficient and all configured guardrails pass"
                ],
                post=window, post_guardrails=post_guardrails, deltas=deltas,
            )

        if self._is_conservative_change():
            if breaches:
                return self._build(
                    outcome=AssessmentOutcome.INCONCLUSIVE, terminal=True,
                    reasons=[
                        "guardrail(s) still breached after a MORE_CONSERVATIVE change; "
                        "V1 has no automatic permissive revert: "
                        + ", ".join(g.metric for g in breaches)
                    ],
                    post=window, post_guardrails=post_guardrails, deltas=deltas,
                )
            return self._build(
                outcome=AssessmentOutcome.KEEP, terminal=True,
                reasons=[
                    "recovery verified: all configured guardrails pass after the "
                    "MORE_CONSERVATIVE change"
                ],
                post=window, post_guardrails=post_guardrails, deltas=deltas,
            )

        # NON_MONOTONIC / boolean / unknown effects have no V1 directional
        # assessment semantics -> terminal INCONCLUSIVE (never forced numeric).
        return self._build(
            outcome=AssessmentOutcome.INCONCLUSIVE, terminal=True,
            reasons=[
                "no directional policy effect to assess "
                f"(effect={self.policy_effect}, direction={self.direction})"
            ],
            post=window, post_guardrails=post_guardrails, deltas=deltas,
        )

    # --------------------------------------------------------------- revert
    def supports_revert(self, fresh: Optional[DecisionRecord]) -> bool:
        """Whether a fresh M7A evaluation independently supports the exact reverse.

        The precondition for a later human-approved revert. M7C itself never
        mutates; the revert is applied through the existing M7B path.
        """
        if self._record is None or self._record.outcome != AssessmentOutcome.PROPOSE_REVERT.value:
            return False
        if fresh is None:
            return False
        return (
            fresh.intent == self.intent
            and fresh.internal_key == self.internal_key
            and fresh.decision == self._record.revert_direction
            and fresh.current_value == self.applied_value
            and fresh.proposed_value == self._record.recommended_value
        )

    # -------------------------------------------------------------- helpers
    def _sufficiency_reasons(self, window: SignalWindow) -> List[str]:
        reasons: List[str] = []
        if window.snapshots < self.config.min_samples:
            reasons.append(
                f"insufficient post-change telemetry: snapshots={window.snapshots} "
                f"< min_samples={self.config.min_samples}"
            )
        for name in ("requests_running", "requests_waiting"):
            aggregate = window.get(name)
            have = aggregate.samples if aggregate else 0
            if have < self.config.min_samples:
                reasons.append(
                    f"insufficient post-change telemetry: {name} fresh samples={have} "
                    f"< min_samples={self.config.min_samples}"
                )
        for guardrail in self.config.guardrails.enabled_guardrails():
            if not guardrail.required_for_scale_up:
                continue
            aggregate = window.get(guardrail.metric)
            if aggregate is None or aggregate.samples == 0:
                reasons.append(
                    f"required post-change evidence unavailable: {guardrail.metric} "
                    "fresh samples=0"
                )
        return reasons

    def _evaluate_guardrails(self, window: SignalWindow) -> List[GuardrailResult]:
        results: List[GuardrailResult] = []
        for guardrail in self.config.guardrails.enabled_guardrails():
            aggregate = window.get(guardrail.metric)
            available = aggregate is not None and aggregate.samples > 0
            observed = aggregate.maximum if available else None
            passed = (observed <= guardrail.max_value) if available else None
            results.append(GuardrailResult(
                name=guardrail.name, metric=guardrail.metric, comparator="<=",
                limit=guardrail.max_value, observed=observed, available=available,
                passed=passed, enabled=True,
                required_for_scale_up=guardrail.required_for_scale_up,
            ))
        return results

    def _numeric_deltas(self, window: SignalWindow) -> Dict[str, Any]:
        """Baseline/post numeric evidence deltas (for inspection only)."""
        baseline_metrics = ((self._baseline or {}).get("window") or {}).get("metrics", {})
        deltas: Dict[str, Any] = {}
        for metric, aggregate in sorted(window.metrics.items()):
            base = baseline_metrics.get(metric) or {}
            base_median = base.get("median")
            post_median = aggregate.median
            if base_median is not None and post_median is not None:
                deltas[metric] = {
                    "baseline_median": base_median,
                    "post_median": post_median,
                    "abs_delta": post_median - base_median,
                }
        return deltas

    def _is_permissive_change(self) -> bool:
        """Whether the applied change was MORE_PERMISSIVE.

        Uses the actuator policy effect when present; otherwise falls back to the
        numeric direction (legacy: an increase is more permissive).
        """
        if self.policy_effect is not None:
            return self.policy_effect == "MORE_PERMISSIVE"
        return self.direction == "UP"

    def _is_conservative_change(self) -> bool:
        if self.policy_effect is not None:
            return self.policy_effect == "MORE_CONSERVATIVE"
        return self.direction == "DOWN"

    def _revert_direction(self) -> Optional[str]:
        if self.direction == "UP":
            return "PROPOSE_DOWN"
        if self.direction == "DOWN":
            return "PROPOSE_UP"
        return None

    def _waiting(self, reasons: List[str]) -> AssessmentRecord:
        """A non-terminal WAITING record (assessment/gate stay active)."""
        return self._build(
            outcome=AssessmentOutcome.INCONCLUSIVE, terminal=False,
            waiting=True, reasons=["waiting: " + r for r in reasons],
            post={}, post_guardrails=[], deltas={},
        )

    def _temporary_or_terminal(
        self, reasons: List[str], now: float,
        window: Optional[SignalWindow] = None,
    ) -> AssessmentRecord:
        """Wait for evidence until the deadline, then terminally INCONCLUSIVE."""
        if now < self.deadline:
            return self._waiting(reasons)
        return self._terminal_inconclusive(
            reasons + [
                f"post-change evidence wait ({self.max_evidence_wait_s:.0f}s after "
                "stabilization) expired without sufficient evidence"
            ],
            now, window=window,
        )

    def _terminal_inconclusive(
        self, reasons: List[str], now: float,
        window: Optional[SignalWindow] = None,
    ) -> AssessmentRecord:
        return self._build(
            outcome=AssessmentOutcome.INCONCLUSIVE, terminal=True, reasons=reasons,
            post=(window or {}), post_guardrails=(
                self._evaluate_guardrails(window) if window is not None else []
            ),
            deltas=(self._numeric_deltas(window) if window is not None else {}),
        )

    def _build(
        self, *, outcome: AssessmentOutcome, terminal: bool, reasons: List[str],
        post: Any, post_guardrails: List[GuardrailResult],
        deltas: Dict[str, Any], waiting: bool = False,
    ) -> AssessmentRecord:
        is_revert = outcome == AssessmentOutcome.PROPOSE_REVERT
        record = AssessmentRecord(
            application_id=self.application_id,
            intent=self.intent,
            internal_key=self.internal_key,
            outcome=outcome.value,
            terminal=terminal,
            waiting=waiting,
            previous_value=self.previous_value,
            applied_value=self.applied_value,
            desired_revision=self.desired_revision,
            direction=self.direction,
            recommended_value=self.previous_value if is_revert else None,
            revert_direction=self._revert_direction() if is_revert else None,
            policy_effect=self.policy_effect,
            baseline=self._baseline,
            baseline_guardrails=((self._baseline or {}).get("guardrails") or []),
            post=(post.to_dict() if isinstance(post, SignalWindow) else (post or {})),
            post_guardrails=_guardrails_to_dicts(post_guardrails),
            numeric_deltas=deltas,
            reasons=list(reasons),
            assessed_at=_iso_now(),
            mutation_applied=False,
        )
        if terminal:
            self._record = record
        return record


class AssessmentManager:
    """Tracks the active post-change assessment per intent and gates mutation."""

    def __init__(self):
        self._assessments: Dict[str, PostChangeAssessment] = {}

    def start(
        self,
        application: ApplicationRecord,
        *,
        config: SupervisorConfig,
        control: ControlReadView,
        assessment_config: Optional[AssessmentConfig] = None,
        now: Optional[float] = None,
    ) -> PostChangeAssessment:
        """Create an assessment bound to a CONVERGED M7B application."""
        if application.status != "CONVERGED" or not application.applied:
            raise ValueError(
                "assessment requires a CONVERGED, applied M7B application"
            )
        assessment_config = assessment_config or AssessmentConfig()
        assessment = PostChangeAssessment(
            application, application.baseline_evidence, config, control,
            max_evidence_wait_s=assessment_config.max_evidence_wait_s, started_at=now,
        )
        self._assessments[application.intent] = assessment
        return assessment

    def assessment_for(self, intent: str = INTENT) -> Optional[PostChangeAssessment]:
        return self._assessments.get(intent)

    def active_for(self, intent: str = INTENT) -> Optional[PostChangeAssessment]:
        assessment = self._assessments.get(intent)
        if assessment is not None and assessment.active:
            return assessment
        return None

    def blocking_reason(self, intent: Optional[str] = None) -> Optional[str]:
        """M7B application-gate callable.

        With no ``intent`` (recommended for M7D1) this enforces **engine-wide
        causal isolation**: while ANY post-change assessment is active on the
        engine, every other actuator's application is deferred. Pass an explicit
        ``intent`` only for per-intent introspection.
        """
        if intent is None:
            active = [a for a in self._assessments.values() if a.active]
            if not active:
                return None
            current = active[0]
            return (
                f"post-change assessment active for {current.intent} "
                f"(application {current.application_id}); engine-wide causal "
                "isolation: mutation waits until it terminates or is abandoned"
            )
        active_intent = self.active_for(intent)
        if active_intent is None:
            return None
        return (
            f"post-change assessment active for {intent} "
            f"(application {active_intent.application_id}); mutation waits until it "
            "terminates or is abandoned"
        )

    def assess(self, samples, *, intent: str = INTENT, now: Optional[float] = None):
        assessment = self._assessments.get(intent)
        if assessment is None:
            return None
        return assessment.assess(samples, now=now)

    def abandon(self, reason: str, *, intent: str = INTENT) -> Optional[AssessmentRecord]:
        assessment = self._assessments.get(intent)
        if assessment is None:
            return None
        return assessment.abandon(reason)

    def supports_revert(self, fresh: Optional[DecisionRecord], *,
                        intent: str = INTENT) -> bool:
        assessment = self._assessments.get(intent)
        return bool(assessment and assessment.supports_revert(fresh))
