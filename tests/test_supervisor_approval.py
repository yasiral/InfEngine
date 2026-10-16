"""M7B: human-approved supervisory application tests.

Deterministic coverage of the explicit approval path over an M7A PROPOSE
DecisionRecord: successful (fresh-supported) approval/apply, rejection, stale
revision, changed current value, expired proposal, bounds/capability change,
Reconciler failure, If-Match conflict, verified convergence, operational
cooldown beginning only after successful convergence, fresh-evidence
revalidation (no-longer-eligible), and single-use/idempotent decisions.
"""

import time
from datetime import datetime, timezone

from infengine.supervisor import (
    INTERNAL_KEY,
    INTENT,
    ApplicationConfig,
    ApprovalRequest,
    ConcurrencyObjective,
    DecisionRecord,
    Guardrail,
    Guardrails,
    Supervisor,
    SupervisorApplication,
    SupervisorConfig,
    decision_id,
)
from infengine.telemetry import RuntimeSnapshot

FIXED_DECIDED = "2026-09-08T00:00:00+00:00"
FIXED_EPOCH = datetime.fromisoformat(FIXED_DECIDED).timestamp()


def _guardrails(decision="PROPOSE_UP", latency_available=True, latency_passed=True,
                kv_passed=True):
    if decision == "PROPOSE_DOWN":
        return [
            {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": False},
            {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": True},
        ]
    return [
        {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
         "required_for_scale_up": True,
         "available": latency_available,
         "passed": (latency_passed if latency_available else None)},
        {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": kv_passed},
    ]


def _record(decision="PROPOSE_UP", current=4, proposed=5, revision=1,
            decided_at=FIXED_DECIDED, guardrails=None, intent=INTENT,
            internal_key=INTERNAL_KEY):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=intent, internal_key=internal_key,
        decided_at=decided_at, decision=decision, proposed_value=proposed,
        current_value=current, capability_bounds={"min": 1, "max": 8},
        transition_type="HOT",
        guardrails=(guardrails if guardrails is not None else _guardrails(decision)),
        evidence={}, window={}, reasons=[], overrides=[], safety={},
        desired_revision=revision, policy={}, mutation_applied=False,
    )


def _fresh(**overrides):
    """A fresh evaluation that reproduces the default proposal."""
    return lambda: _record(**overrides)


class FakeApprovalControl:
    """Read + conditional-PATCH control view for M7B tests."""

    def __init__(self, *, current=4, desired=None, revision=1, lo=1, hi=8,
                 transition="HOT", capability=True, supported=True, operation=False,
                 patch=None):
        self.current = current
        self.desired = dict(desired) if desired is not None else {INTERNAL_KEY: current}
        self.revision = revision
        self.lo, self.hi, self.transition = lo, hi, transition
        self.capability = capability
        self.supported = supported
        self.operation = operation
        self.patch = patch
        self.patch_calls = []
        self.applied_values = []

    def capabilities(self):
        if not self.capability:
            return []
        return [{
            "intent": INTENT, "internal_key": INTERNAL_KEY, "min": self.lo,
            "max": self.hi, "transition": self.transition,
            "supported_on_selected_engine": self.supported,
            "operation": self.operation,
        }]

    def get_config(self):
        return {
            "observed": {INTERNAL_KEY: self.current},
            "desired": dict(self.desired),
            "desired_revision": self.revision,
        }

    def patch_config(self, desired, if_match=None):
        self.patch_calls.append((dict(desired), if_match))
        if self.patch is None:
            raise AssertionError("unexpected mutation path call")
        code, payload = self.patch(dict(desired), if_match)
        if code == 200 and payload.get("status") == "CONVERGED":
            value = (payload.get("observed") or {}).get(INTERNAL_KEY)
            if value is not None:
                self.current = value
                self.desired = {INTERNAL_KEY: value}
                self.applied_values.append(value)
            self.revision = payload.get("desired_revision", self.revision)
        return code, payload


def _ok(value=5, revision=2):
    def patch(desired, if_match):
        return (200, {"status": "CONVERGED",
                      "observed": {INTERNAL_KEY: value},
                      "desired": {INTERNAL_KEY: value},
                      "desired_revision": revision})
    return patch


class _App(SupervisorApplication):
    """Test wrapper: default register()'s clock into the fixed-time window so
    pending TTL is deterministic (age ~1s for the fixed proposal)."""

    def register(self, record, *, now=FIXED_EPOCH + 1.0):
        return super().register(record, now=now)


def _app(control, fresh=None, **cfg):
    provider = _fresh() if fresh is None else fresh
    return _App(ApplicationConfig(**cfg), control, fresh_evidence=provider)


# --------------------------------------------------------------- happy path
def test_successful_fresh_supported_approval_converges():
    control = FakeApprovalControl(current=4, revision=1, patch=_ok(5, 2))
    app = _app(control, proposal_ttl_s=300.0, operational_cooldown_s=60.0)
    record = _record()
    did = app.register(record)
    assert did == decision_id(record)

    now = FIXED_EPOCH + 1.0
    out = app.approve(ApprovalRequest(
        did, expected_desired_revision=1, expected_current_value=4,
        expected_proposed_value=5, approver="alice"), now=now)

    assert out.status == "CONVERGED"
    assert out.stages == ["APPROVED", "CONVERGED"]
    assert out.requested_value == 5 and out.observed_value == 5
    assert out.desired_revision_before == 1 and out.desired_revision_after == 2
    assert out.mutation_attempted is True and out.applied is True
    assert control.applied_values == [5]
    assert control.patch_calls[0][1] == 1  # If-Match bound to recorded revision
    assert out.cooldown_until == now + 60.0
    assert app.in_operational_cooldown(now) is True
    assert app.in_operational_cooldown(now + 60.0) is False


def test_registration_never_mutates():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    assert app.register(_record()) is not None
    assert control.patch_calls == []


# ------------------------------------------------------------------ rejection
def test_explicit_rejection_zero_mutation_no_cooldown():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record())
    now = FIXED_EPOCH + 1.0
    out = app.reject(ApprovalRequest(did, approver="bob"), now=now)

    assert out.status == "REJECTED" and out.stages == ["REJECTED"]
    assert control.patch_calls == [] and control.applied_values == []
    assert out.mutation_attempted is False and out.applied is False
    assert app.operational_cooldown_until() is None
    assert app.in_operational_cooldown(now) is False


# ------------------------------------------------------------- fresh evidence
def test_fresh_hold_no_longer_eligible_zero_mutation():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, fresh=lambda: _record(decision="HOLD", proposed=None))
    did = app.register(_record())
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("no longer eligible" in r and "direction changed" in r for r in out.reasons)
    assert control.patch_calls == []


def test_fresh_opposite_direction_no_longer_eligible():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, fresh=lambda: _record(decision="PROPOSE_DOWN",
                                              current=4, proposed=3))
    did = app.register(_record(decision="PROPOSE_UP", current=4, proposed=5))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("direction changed" in r for r in out.reasons)
    assert control.patch_calls == []


def test_fresh_different_target_no_longer_eligible():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, fresh=lambda: _record(current=4, proposed=6))
    did = app.register(_record(current=4, proposed=5))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("proposed value changed" in r for r in out.reasons)
    assert control.patch_calls == []


def test_fresh_insufficient_telemetry_no_longer_eligible():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, fresh=lambda: None)
    did = app.register(_record())
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("no fresh evaluation available" in r for r in out.reasons)
    assert control.patch_calls == []


def test_fresh_required_guardrail_unavailable_no_longer_eligible():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, fresh=lambda: _record(
        guardrails=_guardrails("PROPOSE_UP", latency_available=False)))
    did = app.register(_record())
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("required guardrail" in r and "not satisfied" in r for r in out.reasons)
    assert control.patch_calls == []


def test_missing_fresh_provider_fails_closed():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = SupervisorApplication(ApplicationConfig(), control)  # no provider
    did = app.register(_record())
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("no fresh evidence provider" in r for r in out.reasons)
    assert control.patch_calls == []


# ----------------------------------------------------- single-use / idempotent
def test_duplicate_approve_at_most_one_patch():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record())
    first = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    second = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 2.0)
    assert first.status == "CONVERGED"
    assert second.status == "STALE"
    assert any("already terminal" in r for r in second.reasons)
    assert second.mutation_attempted is False
    assert len(control.patch_calls) == 1
    assert control.applied_values == [5]


def test_approve_after_reject_zero_mutation():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record())
    rejected = app.reject(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert rejected.status == "REJECTED"
    approval = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 2.0)
    assert approval.status == "STALE"
    assert any("already terminal" in r for r in approval.reasons)
    assert control.patch_calls == []


def test_two_competing_approvals_at_most_one_patch():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app1 = _app(control)
    app2 = _app(control)
    record = _record()
    did1 = app1.register(record)
    did2 = app2.register(record)
    assert did1 == did2
    first = app1.approve(ApprovalRequest(did1), now=FIXED_EPOCH + 1.0)
    second = app2.approve(ApprovalRequest(did2), now=FIXED_EPOCH + 1.0)
    assert first.status == "CONVERGED"
    assert second.status == "STALE"  # live current already moved to 5
    assert len(control.patch_calls) == 1


def test_terminal_decision_cannot_be_reregistered():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    record = _record()
    did = app.register(record)
    app.reject(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert app.register(record) is None  # consumed


# ---------------------------------------------------------- pending lifecycle
def test_cooldown_hold_preserves_pending_proposal_then_approves():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record())  # PROPOSE_UP 4->5
    assert did is not None

    # M7A emits HOLD during its proposal cooldown; it must NOT clear the proposal
    assert app.register(_record(decision="HOLD", proposed=None)) is None
    assert app.pending_id() == did

    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "CONVERGED"
    assert len(control.patch_calls) == 1


def test_preserved_proposal_stale_when_fresh_becomes_hold():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, fresh=lambda: _record(decision="HOLD", proposed=None))
    did = app.register(_record())
    app.register(_record(decision="HOLD", proposed=None))  # cooldown HOLD
    assert app.pending_id() == did

    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("no longer eligible" in r for r in out.reasons)
    assert control.patch_calls == []


def test_newer_conflicting_proposal_supersedes_old():
    control = FakeApprovalControl(
        current=4,
        patch=lambda d, m: (200, {"status": "CONVERGED",
                                  "observed": {INTERNAL_KEY: 6},
                                  "desired_revision": 2}))
    app = _app(control, fresh=lambda: _record(proposed=6))
    old_id = app.register(_record(proposed=5))   # UP 4->5
    new_id = app.register(_record(proposed=6))   # UP 4->6 (conflicting target)
    assert new_id != old_id
    assert app.pending_id() == new_id

    # the superseded decision ID can no longer mutate
    out_old = app.approve(ApprovalRequest(old_id), now=FIXED_EPOCH + 1.0)
    assert out_old.status == "STALE"
    assert any("already terminal" in r for r in out_old.reasons)
    assert control.patch_calls == []

    out_new = app.approve(ApprovalRequest(new_id), now=FIXED_EPOCH + 1.0)
    assert out_new.status == "CONVERGED"
    assert control.applied_values == [6]


def test_opposite_direction_proposal_supersedes():
    control = FakeApprovalControl(
        current=4,
        patch=lambda d, m: (200, {"status": "CONVERGED",
                                  "observed": {INTERNAL_KEY: 3},
                                  "desired_revision": 2}))
    app = _app(control, fresh=lambda: _record(decision="PROPOSE_DOWN",
                                              current=4, proposed=3))
    old_id = app.register(_record(decision="PROPOSE_UP", current=4, proposed=5))
    new_id = app.register(_record(decision="PROPOSE_DOWN", current=4, proposed=3))
    assert new_id != old_id
    out_old = app.approve(ApprovalRequest(old_id), now=FIXED_EPOCH + 1.0)
    assert out_old.status == "STALE" and control.patch_calls == []
    out_new = app.approve(ApprovalRequest(new_id), now=FIXED_EPOCH + 1.0)
    assert out_new.status == "CONVERGED"
    assert control.applied_values == [3]


def test_repeated_equivalent_proposal_no_duplicate_opportunity():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did1 = app.register(_record(decided_at="2026-09-08T00:00:00+00:00"))
    did2 = app.register(_record(decided_at="2026-09-08T00:00:05+00:00"))
    assert did2 == did1  # same decision ID: no duplicate opportunity
    assert app.pending_id() == did1

    out = app.approve(ApprovalRequest(did1), now=FIXED_EPOCH + 1.0)
    assert out.status == "CONVERGED"
    assert len(control.patch_calls) == 1


def test_hold_with_no_pending_is_a_noop():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    assert app.register(_record(decision="HOLD", proposed=None)) is None
    assert app.pending_id() is None
    assert control.patch_calls == []


def test_same_proposal_same_revision_within_ttl_reuses_id():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did1 = app.register(_record(revision=1, decided_at="2026-09-08T00:00:00+00:00"))
    did2 = app.register(_record(revision=1, decided_at="2026-09-08T00:00:05+00:00"))
    assert did2 == did1
    assert app.pending_id() == did1


def test_same_proposal_newer_revision_retires_old_and_registers_fresh():
    control = FakeApprovalControl(current=4, revision=1, patch=_ok(5, 2))
    app = _app(control)
    old_id = app.register(_record(revision=1))
    control.revision = 2  # operator advanced the desired revision
    later = datetime.fromtimestamp(FIXED_EPOCH + 5, tz=timezone.utc).isoformat()
    new_id = app.register(_record(revision=2, decided_at=later))
    assert new_id != old_id
    assert app.pending_id() == new_id

    # the retired (binding-changed) ID can never mutate
    out_old = app.approve(ApprovalRequest(old_id), now=FIXED_EPOCH + 6)
    assert out_old.status == "STALE"
    assert any("already terminal" in r for r in out_old.reasons)
    assert control.patch_calls == []

    # the fresh replacement proposal applies normally, exactly one PATCH
    out_new = app.approve(ApprovalRequest(new_id), now=FIXED_EPOCH + 6)
    assert out_new.status == "CONVERGED"
    assert len(control.patch_calls) == 1


def test_same_proposal_after_ttl_retires_old_and_registers_fresh():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, proposal_ttl_s=30.0)
    old_id = app.register(_record())  # register clock +1s; age 1 < 30
    later = datetime.fromtimestamp(FIXED_EPOCH + 32, tz=timezone.utc).isoformat()
    new_id = app.register(_record(decided_at=later), now=FIXED_EPOCH + 32)  # age 32 > 30
    assert new_id != old_id
    assert app.pending_id() == new_id

    # the expired ID can never mutate
    out_old = app.approve(ApprovalRequest(old_id), now=FIXED_EPOCH + 33)
    assert out_old.status == "STALE"
    assert control.patch_calls == []

    # the fresh replacement proposal applies normally, exactly one PATCH
    out_new = app.approve(ApprovalRequest(new_id), now=FIXED_EPOCH + 33)
    assert out_new.status == "CONVERGED"
    assert len(control.patch_calls) == 1


# --------------------------------------------------------------------- stale
def test_stale_revision_fails_closed():
    control = FakeApprovalControl(current=4, revision=2, patch=_ok(5, 3))
    app = _app(control)
    did = app.register(_record(revision=1))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("desired revision changed" in r for r in out.reasons)
    assert control.patch_calls == [] and control.applied_values == []


def test_changed_current_value_fails_closed():
    control = FakeApprovalControl(current=5, patch=_ok(6, 2))
    app = _app(control)
    did = app.register(_record(current=4, proposed=5))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("observed policy changed" in r for r in out.reasons)
    assert control.patch_calls == []


def test_expired_proposal_fails_closed():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control, proposal_ttl_s=30.0)
    did = app.register(_record())
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 31.0)
    assert out.status == "STALE"
    assert any("expired" in r for r in out.reasons)
    assert control.patch_calls == []


def test_bounds_change_fails_closed():
    control = FakeApprovalControl(current=4, hi=4, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record(proposed=5))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("outside capability bounds" in r for r in out.reasons)
    assert control.patch_calls == []


def test_missing_capability_fails_closed():
    control = FakeApprovalControl(capability=False, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record())
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("capability unavailable" in r for r in out.reasons)
    assert control.patch_calls == []


def test_not_converged_fails_closed():
    control = FakeApprovalControl(current=4, desired={INTERNAL_KEY: 3}, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record(current=4))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("not converged" in r for r in out.reasons)
    assert control.patch_calls == []


def test_unknown_decision_id_fails_closed():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    app.register(_record())
    out = app.approve(ApprovalRequest("deadbeef"), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert control.patch_calls == []


def test_binding_mismatch_fails_closed():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    did = app.register(_record(proposed=5))
    out = app.approve(ApprovalRequest(did, expected_proposed_value=6),
                      now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert any("binding mismatch" in r for r in out.reasons)
    assert control.patch_calls == []


def test_hold_decision_cannot_be_registered():
    control = FakeApprovalControl(current=4, patch=_ok(5, 2))
    app = _app(control)
    assert app.register(_record(decision="HOLD", proposed=None)) is None
    out = app.approve(ApprovalRequest("anything"), now=FIXED_EPOCH + 1.0)
    assert out.status == "STALE"
    assert control.patch_calls == []


# --------------------------------------------------------------- apply faults
def test_reconciler_failure_records_apply_failed():
    control = FakeApprovalControl(
        current=4, patch=lambda d, m: (422, {"status": "FAILED"}))
    app = _app(control)
    did = app.register(_record())
    now = FIXED_EPOCH + 1.0
    out = app.approve(ApprovalRequest(did), now=now)
    assert out.status == "APPLY_FAILED"
    assert out.stages == ["APPROVED", "APPLY_FAILED"]
    assert out.mutation_attempted is True and out.applied is False
    assert control.applied_values == []
    assert app.operational_cooldown_until() is None


def test_if_match_conflict_fails_closed():
    control = FakeApprovalControl(
        current=4, patch=lambda d, m: (409, {"status": "CONFLICT"}))
    app = _app(control)
    did = app.register(_record())
    now = FIXED_EPOCH + 1.0
    out = app.approve(ApprovalRequest(did), now=now)
    assert out.status == "STALE"
    assert any("If-Match conflict" in r for r in out.reasons)
    assert control.applied_values == []  # zero engine mutation
    assert app.operational_cooldown_until() is None


def test_apply_not_verified_is_apply_failed():
    control = FakeApprovalControl(
        current=4,
        patch=lambda d, m: (200, {"status": "CONVERGED",
                                  "observed": {INTERNAL_KEY: 4},
                                  "desired_revision": 2}))
    app = _app(control)
    did = app.register(_record(proposed=5))
    out = app.approve(ApprovalRequest(did), now=FIXED_EPOCH + 1.0)
    assert out.status == "APPLY_FAILED"
    assert any("not verified" in r for r in out.reasons)
    assert out.applied is False
    assert app.operational_cooldown_until() is None


# --------------------------------------------------------------- cooldown rule
def test_cooldown_begins_only_after_successful_convergence():
    now = FIXED_EPOCH + 1.0

    c1 = FakeApprovalControl(current=4, patch=_ok(5, 2))
    a1 = _app(c1)
    a1.reject(ApprovalRequest(a1.register(_record())), now=now)
    assert a1.operational_cooldown_until() is None

    c2 = FakeApprovalControl(current=4, revision=9, patch=_ok(5, 2))
    a2 = _app(c2)
    a2.approve(ApprovalRequest(a2.register(_record())), now=now)
    assert a2.operational_cooldown_until() is None

    c3 = FakeApprovalControl(current=4, patch=lambda d, m: (422, {"status": "FAILED"}))
    a3 = _app(c3)
    a3.approve(ApprovalRequest(a3.register(_record())), now=now)
    assert a3.operational_cooldown_until() is None

    c4 = FakeApprovalControl(current=4, patch=_ok(5, 2))
    a4 = _app(c4, operational_cooldown_s=45.0)
    out = a4.approve(ApprovalRequest(a4.register(_record())), now=now)
    assert out.status == "CONVERGED"
    assert a4.operational_cooldown_until() == now + 45.0
    assert a4.in_operational_cooldown(now + 44.0) is True
    assert a4.in_operational_cooldown(now + 45.0) is False


# --------------------------------------------------------------- determinism
def test_identical_approve_inputs_are_deterministic():
    now = FIXED_EPOCH + 1.0
    outs = []
    for _ in range(2):
        control = FakeApprovalControl(current=4, patch=_ok(5, 2))
        app = _app(control, operational_cooldown_s=60.0)
        did = app.register(_record())
        outs.append(app.approve(ApprovalRequest(did, approver="x"), now=now).to_dict())
    a, b = outs
    a.pop("decided_at")
    b.pop("decided_at")
    assert a == b


# ------------------------------------------------------- M7A integration test
def _snap(running, waiting, kv=0.3, ttft=0.1):
    return RuntimeSnapshot(
        engine="vllm", instance_id="test", collected_at=FIXED_DECIDED,
        source="test", requests_running=running, requests_waiting=waiting,
        kv_usage=kv, ttft_p95_s=ttft,
    )


def test_m7a_proposal_then_human_approval_end_to_end():
    control = FakeApprovalControl(
        current=4, desired={INTERNAL_KEY: 4}, revision=1,
        patch=lambda d, m: (200, {"status": "CONVERGED",
                                  "observed": {INTERNAL_KEY: 5},
                                  "desired_revision": 2}))
    m7a = SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        objective=ConcurrencyObjective(waiting_high=1.0, waiting_low=0.0),
        guardrails=Guardrails(
            latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=60.0,
                              enabled=True, required_for_scale_up=True),
            kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=0.9,
                               enabled=True, required_for_scale_up=True),
        ),
    )
    supervisor = Supervisor(m7a, control)
    now = 1000.0
    samples = [(now - 2, _snap(4, 5)), (now - 1, _snap(4, 6)), (now, _snap(4, 7))]
    record = supervisor.observe(samples, now=now)
    assert record.decision == "PROPOSE_UP" and record.proposed_value == 5

    # fresh evidence: a fresh M7A supervisor over the same fresh telemetry
    fresh = lambda: Supervisor(m7a, control).observe(samples, now=now)  # noqa: E731
    app = SupervisorApplication(ApplicationConfig(), control, fresh_evidence=fresh)
    did = app.register(record)
    out = app.approve(ApprovalRequest(did, approver="operator"), now=time.time())
    assert out.status == "CONVERGED"
    assert out.requested_value == 5 and out.observed_value == 5
    assert control.current == 5
