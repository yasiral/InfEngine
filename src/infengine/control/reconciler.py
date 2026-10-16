from dataclasses import dataclass, field
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional

from .adapter import InfengineAdapter
from .capabilities import TransitionClass, ValidationError
from .state import StateStore


class ReconcileStatus(str, Enum):
    """Overall reconciliation status.

    CONVERGED:    all accepted desired state == observed state.
    PROGRESSING:  valid desired state remains unconverged and an executable
                  transition is pending/in progress. In this synchronous slice
                  it is transient (reserved for deferred/background execution).
    BLOCKED:      valid desired state remains unconverged and cannot currently
                  proceed (e.g. DRAIN waiting for quiescence timed out).
    FAILED:       execution or verification failed, or an input was invalid.
    CONFLICT:     conditional-PATCH guard (M6B-1): the request carried an
                  If-Match/expected desired_revision that does not match the
                  current revision. Zero mutation — nothing persisted, nothing
                  executed. The caller maps this to HTTP 409.
    """

    CONVERGED = "CONVERGED"
    PROGRESSING = "PROGRESSING"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CONFLICT = "CONFLICT"


class StepStatus(str, Enum):
    """Per-parameter transition status."""

    NOOP = "NOOP"                # desired already equals observed
    APPLIED = "APPLIED"          # mutation applied (verification deferred)
    VERIFIED = "VERIFIED"        # applied and observed == desired
    BLOCKED_BUSY = "BLOCKED_BUSY"  # executable but cannot proceed right now
    REJECTED = "REJECTED"        # invalid input / unknown / unsupported
    FAILED = "FAILED"            # execution or verification failed


@dataclass
class PlannedTransition:
    """One entry of the TransitionPlan (built before any execution)."""

    param: str
    transition: TransitionClass
    desired: Any
    observed_before: Any
    status: str  # "NOOP" | "EXECUTE" | "REJECTED"
    reason: Optional[str] = None
    operation: bool = False  # one-shot operation, not a persistent policy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param": self.param,
            "transition": self.transition.value,
            "desired": self.desired,
            "observed_before": self.observed_before,
            "status": self.status,
            "reason": self.reason,
            "operation": self.operation,
        }


@dataclass
class TransitionPlan:
    """Full transition plan: Desired + Observed + Capabilities -> transitions.

    Constructed and validated before anything is executed, so future adapters
    can introduce ordering constraints and dependencies.
    """

    transitions: List[PlannedTransition] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not any(t.status == "REJECTED" for t in self.transitions)

    @property
    def executable(self) -> List[PlannedTransition]:
        return [t for t in self.transitions if t.status == "EXECUTE"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "transitions": [t.to_dict() for t in self.transitions],
        }


@dataclass
class TransitionStep:
    """Execution result for one parameter."""

    param: str
    transition: TransitionClass
    desired: Any
    observed_before: Any
    observed_after: Any
    status: StepStatus
    reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "param": self.param,
            "transition": self.transition.value,
            "desired": self.desired,
            "observed_before": self.observed_before,
            "observed_after": self.observed_after,
            "status": self.status.value,
            "reason": self.reason,
        }


@dataclass
class ReconcileResult:
    plan: TransitionPlan
    steps: List[TransitionStep]
    status: ReconcileStatus
    identity: Dict[str, Any]
    # M13-5D: verified rollback evidence when a coordinated HOT apply partially
    # failed. None when no rollback was required. Never used to hide a failure:
    # ``status`` stays FAILED while this records whether the captured pre-state
    # was restored and verified.
    rollback: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan": self.plan.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
            "status": self.status.value,
            "identity": self.identity,
            "rollback": self.rollback,
        }


class Reconciler:
    """Desired state -> TransitionPlan -> validate -> execute -> verify.

    Desired State is *accepted intent*: every valid requested value is persisted
    even when it cannot currently be applied (e.g. a DRAIN blocked by live
    requests). Invalid/unknown values are rejected and never enter Desired
    State. Physical convergence is per-parameter; Desired State is never
    rewritten to hide unconverged intent.

    M6B-1: every accepted intent is persisted by ``StateStore.save(merged)``,
    which increments the store's ``desired_revision`` (each atomic intent
    acceptance, including a value-identical re-persist). The optional
    ``expected_revision`` argument is the conditional-PATCH guard (If-Match);
    it is compared inside the single-flight ``_reconcile_lock`` before any
    persist or mutation, and a mismatch returns ``ReconcileStatus.CONFLICT``
    with zero mutation.
    """

    def __init__(
        self,
        adapter: InfengineAdapter,
        state_store: Optional[StateStore] = None,
        drain_timeout: float = 30.0,
    ):
        self.adapter = adapter
        self.store = state_store or StateStore(None)
        self.drain_timeout = drain_timeout
        self.last: Optional[ReconcileResult] = None
        self._persist_lock = Lock()
        # Authoritative single-flight boundary for reconciliation. The ENTIRE
        # transaction (plan -> persist -> execute -> verify -> publish) runs
        # under this lock, so direct callers from any thread are serialized too
        # (not just FastAPI PATCHes). Never used for normal inference. The
        # served path additionally uses a single-worker executor to keep this
        # blocking transaction off the event loop; both are intentional.
        self._reconcile_lock = Lock()

    def desired(self) -> Dict[str, Any]:
        return dict(self.store.desired)

    def plan_for(self, desired: Dict[str, Any]) -> TransitionPlan:
        """Build and validate the TransitionPlan without executing anything."""
        transitions: List[PlannedTransition] = []
        for name, value in desired.items():
            cap = self.adapter.capability(name)
            if cap is None:
                transitions.append(PlannedTransition(
                    param=name, transition=TransitionClass.RESTART, desired=value,
                    observed_before=None, status="REJECTED",
                    reason="unknown parameter for this adapter",
                ))
                continue
            try:
                normalized = self.adapter.coerce(name, value)
            except ValidationError as exc:
                transitions.append(PlannedTransition(
                    param=name, transition=cap.transition, desired=value,
                    observed_before=None, status="REJECTED", reason=str(exc),
                ))
                continue
            try:
                before = self.adapter.get(name)
            except KeyError:
                before = None
            if before == normalized:
                transitions.append(PlannedTransition(
                    param=name, transition=cap.transition, desired=normalized,
                    observed_before=before, status="NOOP", operation=cap.operation,
                ))
            elif cap.transition not in (TransitionClass.HOT, TransitionClass.DRAIN):
                transitions.append(PlannedTransition(
                    param=name, transition=cap.transition, desired=normalized,
                    observed_before=before, status="REJECTED",
                    reason=f"{cap.transition.value} transition not supported in this slice",
                    operation=cap.operation,
                ))
            else:
                transitions.append(PlannedTransition(
                    param=name, transition=cap.transition, desired=normalized,
                    observed_before=before, status="EXECUTE", operation=cap.operation,
                ))
        return TransitionPlan(transitions)

    def reconcile(
        self,
        desired: Optional[Dict[str, Any]] = None,
        *,
        drain_timeout: Optional[float] = None,
        expected_revision: Optional[int] = None,
    ) -> ReconcileResult:
        with self._reconcile_lock:
            # M6B-1 conditional-PATCH guard: the If-Match/expected revision is
            # compared ATOMICALLY with the mutation, inside the reconciliation
            # critical section — never as a separate check-then-write at the
            # API layer. A mismatch aborts the entire transaction with zero
            # mutation: nothing is persisted, nothing is executed, and the
            # previous ReconcileResult (self.last) is left untouched. This runs
            # BEFORE plan validation so a stale writer gets a deterministic
            # CONFLICT regardless of its (possibly stale) content.
            if expected_revision is not None and expected_revision != self.store.revision:
                return ReconcileResult(
                    plan=TransitionPlan(),
                    steps=[],
                    status=ReconcileStatus.CONFLICT,
                    identity=dict(self.adapter.identity()),
                )
            target = dict(self.store.load() if desired is None else desired)
            plan = self.plan_for(target)

            # Atomic intent acceptance: an invalid/unknown entry rejects the ENTIRE
            # request. Nothing is persisted and nothing is mutated, so a client that
            # receives FAILED/422 cannot discover partial side effects.
            if not plan.valid:
                result = ReconcileResult(
                    plan=plan,
                    steps=self._static_steps(plan),
                    status=ReconcileStatus.FAILED,
                    identity=dict(self.adapter.identity()),
                )
                self.last = result
                return result

# Accept valid intent atomically, then reconcile physically. A valid but
            # currently blocked DRAIN is persisted as Desired State even though it
            # cannot converge this cycle. One-shot operations are NOT persisted
            # (they are transient requests, not persistent policy).
            accepted = {t.param: t.desired for t in plan.transitions
                        if t.status != "REJECTED" and not t.operation}
            with self._persist_lock:
                merged = dict(self.store.desired)
                merged.update(accepted)
                self.store.save(merged)

            steps = self._execute(plan, self.drain_timeout if drain_timeout is None else drain_timeout)
            status = self._overall(steps)
            # M13-5D: a HOT apply that failed leaves a possibly partial engine
            # mutation. Restore the captured pre-state (verified) and record the
            # outcome. DRAIN BLOCKED is NOT rolled back: it is accepted intent
            # awaiting quiescence (preserved M10/M11/M12 semantics).
            rollback = None
            if self._needs_hot_rollback(status, steps):
                rollback = self._rollback_hot(steps)
            result = ReconcileResult(
                plan=plan,
                steps=steps,
                status=status,
                identity=dict(self.adapter.identity()),
                rollback=rollback,
            )
            # Atomic publish: `result` is fully constructed before the reference is
            # assigned, so /runtime/status can never observe a partially-mutated
            # reconciliation result.
            self.last = result
            return result

    def _static_steps(self, plan: TransitionPlan) -> List[TransitionStep]:
        """Steps for non-executed plan entries (NOOP / REJECTED)."""
        steps: List[TransitionStep] = []
        for t in plan.transitions:
            if t.status == "NOOP":
                steps.append(TransitionStep(
                    param=t.param, transition=t.transition, desired=t.desired,
                    observed_before=t.observed_before, observed_after=t.observed_before,
                    status=StepStatus.NOOP,
                ))
            elif t.status == "REJECTED":
                steps.append(TransitionStep(
                    param=t.param, transition=t.transition, desired=t.desired,
                    observed_before=t.observed_before, observed_after=t.observed_before,
                    status=StepStatus.REJECTED, reason=t.reason,
                ))
        return steps

    # ------------------------------------------------------------- execution
    def _execute(self, plan: TransitionPlan, drain_timeout: float) -> List[TransitionStep]:
        steps = self._static_steps(plan)
        hot = [t for t in plan.transitions
               if t.status == "EXECUTE" and t.transition is TransitionClass.HOT]
        drain = [t for t in plan.transitions
                 if t.status == "EXECUTE" and t.transition is TransitionClass.DRAIN]
        steps.extend(self._exec_hot(hot))
        if drain:
            steps.extend(self._exec_drain_cycle(drain, drain_timeout))
        return steps

    def _exec_hot(self, hot: List[PlannedTransition]) -> List[TransitionStep]:
        """Apply all HOT transitions. Adapters that can coordinate a multi-field
        engine request (``APPLY_SEMANTICS`` batch-capable) receive ONE
        ``apply_many`` call; others keep the historical per-parameter ``apply``
        (which preserves single-flight/apply-spy semantics)."""
        if not hot:
            return []
        if getattr(self.adapter, "SUPPORTS_BATCH_APPLY", False) and len(hot) > 0:
            overrides = {t.param: t.desired for t in hot}
            res = self.adapter.apply_many(overrides)
            return [self._verify_apply(t, res) for t in hot]
        return [self._exec_one(t) for t in hot]

    def _exec_one(self, t: PlannedTransition) -> TransitionStep:
        return self._verify_apply(t, self.adapter.apply(t.param, t.desired))

    def _verify_apply(self, t: PlannedTransition, res) -> TransitionStep:
        if t.operation:
            # one-shot operation: verification comes from the engine observable,
            # not get()==desired (there is no persistent desired value)
            verified = bool(res.applied and self.adapter.verify(t.param))
            return TransitionStep(t.param, t.transition, t.desired, t.observed_before,
                                  int(verified), StepStatus.VERIFIED if verified else StepStatus.FAILED,
                                  None if verified else (res.reason or "operation verification failed"))
        after = self.adapter.get(t.param) if res.applied else t.observed_before
        if res.applied and after == t.desired:
            return TransitionStep(t.param, t.transition, t.desired, t.observed_before,
                                  after, StepStatus.VERIFIED)
        if res.applied:
            return TransitionStep(t.param, t.transition, t.desired, t.observed_before,
                                  after, StepStatus.FAILED,
                                  reason=f"verification mismatch: observed {after} != desired {t.desired}")
        # Apply reported failure: read the CURRENT observed value (best-effort) so
        # a coordinated batch that partially mutated the engine is detected and
        # can be restored, rather than assumed unchanged.
        try:
            after = self.adapter.get(t.param)
        except Exception:
            after = t.observed_before
        return TransitionStep(t.param, t.transition, t.desired, t.observed_before,
                              after, StepStatus.FAILED, reason=res.reason)

    # ------------------------------------------------------------- rollback (M13-5D)
    @staticmethod
    def _needs_hot_rollback(status: ReconcileStatus, steps: List[TransitionStep]) -> bool:
        """Rollback only a genuinely failed HOT apply. A DRAIN BLOCKED is
        accepted intent awaiting quiescence (preserved M10-12 behavior) and must
        NOT be rolled back."""
        if status is not ReconcileStatus.FAILED:
            return False
        return any(
            s.transition is TransitionClass.HOT and s.status is StepStatus.FAILED
            for s in steps
        )

    def _rollback_hot(self, steps: List[TransitionStep]) -> Dict[str, Any]:
        """Restore the captured pre-state for HOT parameters, then verify.

        Only parameters whose observed value actually changed are re-applied.
        Returns a deterministic, machine-readable record. A rollback that cannot
        restore+verify every changed parameter reports ``ROLLBACK_FAILED`` (never
        hidden)."""
        changed = [
            s for s in steps
            if s.transition is TransitionClass.HOT
            and s.status in (StepStatus.VERIFIED, StepStatus.FAILED)
            and s.observed_after != s.observed_before
        ]
        if not changed:
            return {"status": "ROLLBACK_SUCCEEDED", "restored": [], "failed": [],
                    "detail": "no HOT parameter observed changed"}
        try:
            if getattr(self.adapter, "SUPPORTS_BATCH_APPLY", False):
                res = self.adapter.apply_many({s.param: s.observed_before for s in changed})
                if not res.applied:
                    return {"status": "ROLLBACK_FAILED", "restored": [],
                            "failed": [s.param for s in changed],
                            "detail": res.reason or "batch rollback apply failed"}
            else:
                for s in changed:
                    res = self.adapter.apply(s.param, s.observed_before)
                    if not res.applied:
                        return {"status": "ROLLBACK_FAILED",
                                "restored": [], "failed": [c.param for c in changed],
                                "detail": res.reason or f"rollback apply failed for {s.param}"}
            restored, failed = [], []
            for s in changed:
                try:
                    now = self.adapter.get(s.param)
                except Exception:
                    now = None
                if now == s.observed_before:
                    restored.append(s.param)
                else:
                    failed.append({"param": s.param, "expected": s.observed_before,
                                   "observed": now})
        except Exception as exc:
            return {"status": "ROLLBACK_FAILED", "restored": [], "failed": [
                s.param for s in changed], "detail": f"{type(exc).__name__}: {exc}"}
        return {
            "status": "ROLLBACK_SUCCEEDED" if not failed else "ROLLBACK_FAILED",
            "restored": restored,
            "failed": failed,
        }

    def apply_semantics(self) -> str:
        """Truthful engine/control-plane apply guarantee for this adapter."""
        return getattr(self.adapter, "APPLY_SEMANTICS", "COORDINATED_NON_ATOMIC")

    def _exec_drain_cycle(self, drain: List[PlannedTransition],
                          drain_timeout: float) -> List[TransitionStep]:
        d = self.adapter.drain(drain_timeout)
        if not d.drained:
            return [TransitionStep(t.param, t.transition, t.desired, t.observed_before,
                                   t.observed_before, StepStatus.BLOCKED_BUSY, d.reason)
                    for t in drain]
        steps: List[TransitionStep] = []
        resume_error: Optional[Exception] = None
        try:
            steps = [self._exec_one(t) for t in drain]
        finally:
            try:
                # always reopen admission after a drain cycle, success or failure,
                # so admission is never left disabled after an exception.
                self.adapter.resume()
            except Exception as exc:
                # Lifecycle restoration failed. Even a VERIFIED operation cannot
                # be reported CONVERGED because the engine may remain paused.
                # Preserve BOTH facts: the primary operation outcome (in the
                # reason) and the cleanup error. Overall becomes FAILED.
                resume_error = exc
        if resume_error is not None:
            return [
                TransitionStep(
                    param=t.param, transition=t.transition, desired=t.desired,
                    observed_before=t.observed_before, observed_after=t.observed_after,
                    status=StepStatus.FAILED,
                    reason=(
                        f"drain lifecycle resume failed (engine may remain paused): "
                        f"{resume_error}; primary op outcome: {t.status.value}"
                    ),
                )
                for t in steps
            ]
        return steps

    # ------------------------------------------------------------ verification
    def _overall(self, steps: List[TransitionStep]) -> ReconcileStatus:
        if any(s.status in (StepStatus.REJECTED, StepStatus.FAILED) for s in steps):
            return ReconcileStatus.FAILED
        if any(s.status is StepStatus.BLOCKED_BUSY for s in steps):
            return ReconcileStatus.BLOCKED
        if all(s.observed_after == s.desired for s in steps):
            return ReconcileStatus.CONVERGED
        return ReconcileStatus.PROGRESSING