"""M7A: deterministic supervisory policy controller V1 (OBSERVE_ONLY).

The smallest deterministic supervisory controller, scoped to
``runtime_concurrency`` only:

  * :func:`build_signal_window` / :class:`SignalWindow` — windowed, factual,
    numeric signals over existing M5B/M5C telemetry (no labels/diagnoses);
  * :class:`SupervisorConfig` / :class:`ConcurrencyPolicy` — configurable
    objective + guardrails and the deterministic HOLD / PROPOSE_UP /
    PROPOSE_DOWN evaluation;
  * :class:`DecisionRecord` / :class:`Supervisor` — the auditable decision and
    the OBSERVE_ONLY enforcement of capability bounds, cooldown and
    conflict/safety.

The supervisor never mutates DesiredState or the engine. No LLM/ML/RL, no
generic diagnosis, no fleet/routing/dashboard, no admission layer, no additional
actuator.

M7B (this package, ``approval``) adds the explicit **human-approval** path for an
M7A PROPOSE decision. It changes no M7A policy semantics: only an explicit
``APPROVE`` may apply a proposal, and only through the existing Control
Contract → If-Match → Reconciler path. See :class:`SupervisorApplication`.

M7C (this package, ``assessment``) adds deterministic **post-change
assessment**: after a CONVERGED application it produces ``KEEP`` /
``PROPOSE_REVERT`` / ``INCONCLUSIVE`` from a fresh post-stabilization window
against the persisted pre-change baseline, never mutating the engine, and gates
further applications while an assessment is active. See
:class:`AssessmentManager`.

M7E (this package, ``arbitration``) adds deterministic **multi-actuator
arbitration** over the already-POLICY_SUPPORTED policies: one shared evidence
epoch, effect priority (MORE_CONSERVATIVE before MORE_PERMISSIVE) then a
configured visible actuator order, ``NO_CANDIDATE`` / ``SELECTED`` / ``DEFERRED``,
zero mutation, and an auditable :class:`ArbitrationRecord`. A selected candidate
is handed to the unchanged M7B human approval path. The engine-wide
one-change-at-a-time lifecycle is a fail-closed invariant owned by
:class:`PolicyChangeSlot` (a pending/registered proposal, a non-terminal
application, or an active M7C assessment all force ``DEFERRED``), so normal M7E
construction cannot silently omit it. See :class:`MultiActuatorArbitrator`.
"""

from .signals import (
    CUMULATIVE_COUNTER_METRICS,
    MetricAggregate,
    SignalWindow,
    build_signal_window,
)
from .policy import (
    ActuatorValue,
    CapabilityBounds,
    ConcurrencyObjective,
    ConcurrencyPolicy,
    DecisionKind,
    Guardrail,
    GuardrailResult,
    Guardrails,
    KvWatermarkObjective,
    KvWatermarkPolicy,
    PolicyEffect,
    PolicyEvaluation,
    SupervisorConfig,
    TokenBudgetObjective,
    TokenBudgetPolicy,
)
from .supervisor import (
    INTERNAL_KEY,
    INTENT,
    ControlReadView,
    DecisionRecord,
    Supervisor,
    SupervisorMode,
)
from .approval import (
    ApplicationConfig,
    ApplicationRecord,
    ApplicationStatus,
    ApprovalControlView,
    ApprovalRequest,
    FreshEvidenceProvider,
    SupervisorApplication,
    decision_id,
)
from .assessment import (
    AssessmentConfig,
    AssessmentManager,
    AssessmentOutcome,
    AssessmentRecord,
    PostChangeAssessment,
)
from .coverage import (
    KV_WATERMARK_INTENT,
    KV_WATERMARK_KEY,
    LONG_PREFILL_INTENT,
    RESERVE_FULL_ISL_INTENT,
    SGLANG_PREFILL_INTENT,
    SGLANG_PREFILL_KEY,
    SCHEDULED_TOKENS_INTENT,
    SCHEDULED_TOKENS_KEY,
    PolicyCoverageEntry,
    PolicySupport,
    policy_coverage,
    policy_for,
)
from .arbitration import (
    DEFAULT_ACTUATOR_PRIORITY,
    ActuatorEvaluation,
    ArbitrationActuator,
    ArbitrationConfig,
    ArbitrationOutcome,
    ArbitrationRecord,
    MultiActuatorArbitrator,
    PolicyChangeSlot,
    combine_gates,
)

__all__ = [
    "ActuatorValue",
    "MetricAggregate",
    "SignalWindow",
    "build_signal_window",
    "CUMULATIVE_COUNTER_METRICS",
    "CapabilityBounds",
    "ConcurrencyObjective",
    "ConcurrencyPolicy",
    "DecisionKind",
    "Guardrail",
    "GuardrailResult",
    "Guardrails",
    "PolicyEffect",
    "PolicyEvaluation",
    "SupervisorConfig",
    "TokenBudgetObjective",
    "TokenBudgetPolicy",
    "KvWatermarkObjective",
    "KvWatermarkPolicy",
    "ControlReadView",
    "DecisionRecord",
    "Supervisor",
    "SupervisorMode",
    "INTENT",
    "INTERNAL_KEY",
    "ApplicationConfig",
    "ApplicationRecord",
    "ApplicationStatus",
    "ApprovalControlView",
    "ApprovalRequest",
    "FreshEvidenceProvider",
    "SupervisorApplication",
    "decision_id",
    "AssessmentConfig",
    "AssessmentManager",
    "AssessmentOutcome",
    "AssessmentRecord",
    "PostChangeAssessment",
    "SCHEDULED_TOKENS_INTENT",
    "SCHEDULED_TOKENS_KEY",
    "SGLANG_PREFILL_INTENT",
    "SGLANG_PREFILL_KEY",
    "KV_WATERMARK_INTENT",
    "KV_WATERMARK_KEY",
    "LONG_PREFILL_INTENT",
    "RESERVE_FULL_ISL_INTENT",
    "PolicyCoverageEntry",
    "PolicySupport",
    "policy_coverage",
    "policy_for",
    "DEFAULT_ACTUATOR_PRIORITY",
    "ActuatorEvaluation",
    "ArbitrationActuator",
    "ArbitrationConfig",
    "ArbitrationOutcome",
    "ArbitrationRecord",
    "MultiActuatorArbitrator",
    "PolicyChangeSlot",
    "combine_gates",
]
