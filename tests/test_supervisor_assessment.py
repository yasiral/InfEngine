"""M7C: deterministic post-change assessment tests.

Covers KEEP, post-UP guardrail breach -> PROPOSE_REVERT (zero mutation),
INCONCLUSIVE for insufficient/stale evidence, external revision / current-value /
convergence change, baseline binding mismatch, cooldown-not-elapsed, DOWN
recovery KEEP / unresolved INCONCLUSIVE, the exact-reverse revert precondition,
no mutation, and the assessment gate blocking a second application.
"""

import pytest

from infengine.supervisor import (
    INTERNAL_KEY,
    INTENT,
    ApplicationConfig,
    ApplicationRecord,
    ApprovalRequest,
    AssessmentConfig,
    AssessmentManager,
    ConcurrencyObjective,
    DecisionRecord,
    Guardrail,
    Guardrails,
    SupervisorApplication,
    SupervisorConfig,
)
from infengine.telemetry import RuntimeSnapshot

FIXED = "2026-09-08T00:00:00+00:00"
NOW = 1_000_000.0


def _snap(running, waiting, kv=0.2, ttft=0.1, quality=None):
    return RuntimeSnapshot(
        engine="sglang", instance_id="test", collected_at=FIXED, source="test",
        requests_running=running, requests_waiting=waiting, kv_usage=kv,
        ttft_p95_s=ttft, quality=quality or {},
    )


def _post(now, *, running=5, waiting=8, kv=0.2, ttft=0.1, n=3):
    return [(now - (n - 1 - i) * 1.0, _snap(running, waiting, kv=kv, ttft=ttft))
            for i in range(n)]


def _config(*, window_s=10.0, min_samples=3, latency_required=True,
            latency_max=1.0, kv_max=0.9):
    return SupervisorConfig(
        window_s=window_s, min_samples=min_samples, cooldown_s=60.0,
        objective=ConcurrencyObjective(waiting_high=1.0, waiting_low=0.0),
        guardrails=Guardrails(
            latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=latency_max,
                              enabled=True, required_for_scale_up=latency_required),
            kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=kv_max,
                               enabled=True, required_for_scale_up=True),
        ),
    )


def _baseline(*, previous=4, applied=5, decision="PROPOSE_UP", revision=1,
              latency_passed=True, kv_passed=True, latency_available=True):
    return {
        "intent": INTENT, "internal_key": INTERNAL_KEY, "decision": decision,
        "current_value": previous, "proposed_value": applied,
        "desired_revision": revision,
        "guardrails": [
            {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
             "required_for_scale_up": True, "available": latency_available,
             "passed": (latency_passed if latency_available else None)},
            {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": kv_passed},
        ],
        "window": {"metrics": {
            "requests_running": {"median": previous},
            "requests_waiting": {"median": 8.0},
            "kv_usage": {"median": 0.2},
        }},
    }


def _converged(*, previous=4, applied=5, revision_before=1, revision_after=2,
               cooldown_until=None, baseline="default", decision="PROPOSE_UP"):
    if baseline == "default":
        baseline = _baseline(previous=previous, applied=applied,
                             decision=decision, revision=revision_before)
    return ApplicationRecord(
        decision_id="app-1", intent=INTENT, internal_key=INTERNAL_KEY,
        status="CONVERGED", stages=["APPROVED", "CONVERGED"],
        proposal_decision=decision, proposal_decided_at=FIXED, decided_at=FIXED,
        current_value=previous, requested_value=applied, observed_value=applied,
        desired_revision_before=revision_before, desired_revision_after=revision_after,
        reasons=["approved and verified"], mutation_attempted=True, applied=True,
        cooldown_until=(NOW - 1.0 if cooldown_until is None else cooldown_until),
        approver="human", baseline_evidence=baseline,
    )


class FakeAssessmentControl:
    """Read-only control view; ``patch_config`` raises if ever called."""

    def __init__(self, *, current=5, desired=None, revision=2):
        self.current = current
        self.desired = dict(desired) if desired is not None else {INTERNAL_KEY: current}
        self.revision = revision
        self.patch_calls = 0

    def capabilities(self):
        return [{"intent": INTENT, "internal_key": INTERNAL_KEY, "min": 1, "max": 8,
                 "transition": "HOT", "supported_on_selected_engine": True}]

    def get_config(self):
        return {"observed": {INTERNAL_KEY: self.current}, "desired": dict(self.desired),
                "desired_revision": self.revision}

    def patch_config(self, *args, **kwargs):
        self.patch_calls += 1
        raise AssertionError("M7C must not mutate")


def _assess(application, control, samples, *, config=None, assessment_config=None,
            now=NOW):
    manager = AssessmentManager()
    assessment = manager.start(application, config=(config or _config()), control=control,
                               assessment_config=assessment_config, now=now)
    record = assessment.assess(samples, now=now)
    return manager, assessment, record


# ----------------------------------------------------------------------- KEEP
def test_keep_when_all_guardrails_pass():
    control = FakeAssessmentControl(current=5, revision=2)
    _, assessment, record = _assess(_converged(), control, _post(NOW))
    assert record.outcome == "KEEP"
    assert record.terminal is True
    assert record.mutation_applied is False
    assert assessment.terminal is True
    assert control.patch_calls == 0


def test_numeric_deltas_recorded_for_inspection():
    control = FakeAssessmentControl(current=5, revision=2)
    _, _, record = _assess(_converged(), control, _post(NOW))
    assert "requests_running" in record.numeric_deltas
    delta = record.numeric_deltas["requests_running"]
    assert set(delta) == {"baseline_median", "post_median", "abs_delta"}
    assert delta["baseline_median"] == 4 and delta["post_median"] == 5


# ----------------------------------------------------- breach -> PROPOSE_REVERT
def test_post_up_guardrail_breach_proposes_revert():
    control = FakeAssessmentControl(current=5, revision=2)
    _, assessment, record = _assess(
        _converged(), control, _post(NOW, ttft=5.0), config=_config(latency_max=1.0))
    assert record.outcome == "PROPOSE_REVERT"
    assert record.recommended_value == 4  # exact previous value
    assert record.revert_direction == "PROPOSE_DOWN"
    assert record.mutation_applied is False
    assert control.patch_calls == 0
    assert any("passing pre-change but breached post-change" in r for r in record.reasons)


def test_pre_existing_breach_is_not_attributed_inconclusive():
    control = FakeAssessmentControl(current=5, revision=2)
    baseline = _baseline(latency_passed=False)  # already failing pre-change
    _, _, record = _assess(_converged(baseline=baseline), control,
                           _post(NOW, ttft=5.0), config=_config(latency_max=1.0))
    assert record.outcome == "INCONCLUSIVE"
    assert any("not newly" in r for r in record.reasons)


# ------------------------------------------------------------- INCONCLUSIVE
def test_post_cooldown_insufficient_evidence_waits_nonterminal():
    control = FakeAssessmentControl(current=5, revision=2)
    manager, assessment, record = _assess(_converged(), control, _post(NOW, n=1))
    assert record.outcome == "INCONCLUSIVE"
    assert record.terminal is False and record.waiting is True
    assert any("waiting:" in r and "insufficient post-change telemetry" in r
               for r in record.reasons)
    assert manager.active_for(INTENT) is assessment  # gate stays active


def test_evidence_never_arrives_terminal_only_at_deadline():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(_converged(), config=_config(), control=control,
                               assessment_config=AssessmentConfig(max_evidence_wait_s=30.0),
                               now=NOW)
    early = assessment.assess(_post(NOW, n=1), now=NOW + 5.0)
    assert early.terminal is False and early.waiting is True
    assert manager.active_for(INTENT) is assessment

    late = assessment.assess(_post(NOW + 35, n=1), now=NOW + 35.0)
    assert late.outcome == "INCONCLUSIVE"
    assert late.terminal is True
    assert any("evidence wait" in r and "expired" in r for r in late.reasons)
    assert manager.active_for(INTENT) is None  # gate lifts at the deadline


def test_later_sufficient_evidence_after_waiting_keeps():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(_converged(), config=_config(), control=control,
                               assessment_config=AssessmentConfig(max_evidence_wait_s=30.0),
                               now=NOW)
    waiting = assessment.assess(_post(NOW, n=1), now=NOW + 1.0)
    assert waiting.waiting is True and waiting.terminal is False
    keeper = assessment.assess(_post(NOW + 5.0), now=NOW + 5.0)
    assert keeper.outcome == "KEEP" and keeper.terminal is True


def test_later_guardrail_breach_after_waiting_proposes_revert():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(_converged(), config=_config(latency_max=1.0),
                               control=control,
                               assessment_config=AssessmentConfig(max_evidence_wait_s=30.0),
                               now=NOW)
    waiting = assessment.assess(_post(NOW, n=1), now=NOW + 1.0)
    assert waiting.waiting is True
    revert = assessment.assess(_post(NOW + 5.0, ttft=5.0), now=NOW + 5.0)
    assert revert.outcome == "PROPOSE_REVERT"
    assert revert.recommended_value == 4
    assert revert.mutation_applied is False
    assert control.patch_calls == 0


def test_stale_telemetry_waits_then_terminal_at_deadline():
    control = FakeAssessmentControl(current=5, revision=2)
    quality = {"requests_running": "stale", "requests_waiting": "stale",
               "kv_usage": "stale", "ttft_p95_s": "stale"}

    def stale(now):
        return [(now - 2, _snap(5, 8, quality=quality)),
                (now - 1, _snap(5, 8, quality=quality)),
                (now, _snap(5, 8, quality=quality))]

    manager = AssessmentManager()
    assessment = manager.start(_converged(), config=_config(), control=control,
                               assessment_config=AssessmentConfig(max_evidence_wait_s=30.0),
                               now=NOW)
    early = assessment.assess(stale(NOW), now=NOW + 5.0)
    assert early.waiting is True and early.terminal is False
    late = assessment.assess(stale(NOW + 35), now=NOW + 35.0)
    assert late.terminal is True and late.outcome == "INCONCLUSIVE"


def test_cooldown_longer_than_evidence_timeout_gets_full_wait_after_cooldown():
    # cooldown 100s, evidence wait 30s -> deadline is +130 (full 30s AFTER cooldown)
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(
        _converged(cooldown_until=NOW + 100.0), config=_config(), control=control,
        assessment_config=AssessmentConfig(max_evidence_wait_s=30.0), now=NOW)

    during = assessment.assess(_post(NOW + 99, n=1), now=NOW + 99.0)
    assert during.waiting is True and during.terminal is False

    just_after = assessment.assess(_post(NOW + 101, n=1), now=NOW + 101.0)
    assert just_after.waiting is True and just_after.terminal is False

    at_deadline = assessment.assess(_post(NOW + 130, n=1), now=NOW + 130.0)
    assert at_deadline.terminal is True and at_deadline.outcome == "INCONCLUSIVE"


def test_no_terminal_missing_evidence_during_stabilization():
    # Even if the post-cooldown evidence wait would already have elapsed, the
    # assessment cannot terminate for missing evidence while stabilization runs.
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(
        _converged(cooldown_until=NOW + 50.0), config=_config(), control=control,
        assessment_config=AssessmentConfig(max_evidence_wait_s=10.0), now=NOW)

    dense = assessment.assess(_post(NOW + 40, n=1), now=NOW + 40.0)
    assert dense.terminal is False and dense.waiting is True
    assert any("cooldown not elapsed" in r for r in dense.reasons)
    assert manager.active_for(INTENT) is assessment

    # stabilization ends at +50; the evidence wait then runs to +60
    after = assessment.assess(_post(NOW + 55, n=1), now=NOW + 55.0)
    assert after.terminal is False and after.waiting is True
    assert assessment.assess(_post(NOW + 60, n=1), now=NOW + 60.0).terminal is True


def test_immediately_after_cooldown_insufficient_evidence_is_waiting():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(
        _converged(cooldown_until=NOW + 10.0), config=_config(), control=control,
        assessment_config=AssessmentConfig(max_evidence_wait_s=30.0), now=NOW)
    record = assessment.assess(_post(NOW + 11, n=1), now=NOW + 11.0)
    assert record.outcome == "INCONCLUSIVE"
    assert record.terminal is False and record.waiting is True
    assert any("insufficient post-change telemetry" in r for r in record.reasons)


def test_external_revision_change_inconclusive():
    control = FakeAssessmentControl(current=5, revision=9)
    _, _, record = _assess(_converged(), control, _post(NOW))
    assert record.outcome == "INCONCLUSIVE"
    assert any("external desired revision change" in r for r in record.reasons)


def test_current_value_change_inconclusive():
    control = FakeAssessmentControl(current=6, revision=2)
    _, _, record = _assess(_converged(), control, _post(NOW))
    assert record.outcome == "INCONCLUSIVE"
    assert any("observed current value changed" in r for r in record.reasons)


def test_not_converged_inconclusive():
    control = FakeAssessmentControl(current=5, desired={INTERNAL_KEY: 6}, revision=2)
    _, _, record = _assess(_converged(), control, _post(NOW))
    assert record.outcome == "INCONCLUSIVE"
    assert any("not converged" in r for r in record.reasons)


def test_baseline_binding_mismatch_inconclusive():
    control = FakeAssessmentControl(current=5, revision=2)
    bad = _baseline(previous=4, applied=6)  # proposed != applied
    _, _, record = _assess(_converged(baseline=bad), control, _post(NOW))
    assert record.outcome == "INCONCLUSIVE"
    assert any("baseline binding mismatch" in r for r in record.reasons)


def test_missing_baseline_inconclusive():
    control = FakeAssessmentControl(current=5, revision=2)
    application = _converged()
    application.baseline_evidence = None
    _, _, record = _assess(application, control, _post(NOW))
    assert record.outcome == "INCONCLUSIVE"
    assert any("no pre-change baseline" in r for r in record.reasons)


def test_cooldown_not_elapsed_then_keep():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(_converged(cooldown_until=NOW + 10.0),
                               config=_config(), control=control, now=NOW)

    record = assessment.assess(_post(NOW), now=NOW)
    assert record.outcome == "INCONCLUSIVE"
    assert record.terminal is False and record.waiting is True
    assert any("cooldown not elapsed" in r for r in record.reasons)
    assert manager.active_for(INTENT) is assessment  # gate still on

    after = assessment.assess(_post(NOW + 11.0), now=NOW + 11.0)
    assert after.outcome == "KEEP"
    assert manager.active_for(INTENT) is None  # terminal: gate lifts


# ---------------------------------------------------------------------- DOWN
def test_down_change_recovery_keep():
    control = FakeAssessmentControl(current=4, revision=2)
    application = _converged(previous=5, applied=4, decision="PROPOSE_DOWN")
    _, _, record = _assess(application, control, _post(NOW, running=4, waiting=2))
    assert record.outcome == "KEEP"
    assert record.direction == "DOWN"


def test_down_change_unresolved_inconclusive():
    control = FakeAssessmentControl(current=4, revision=2)
    application = _converged(previous=5, applied=4, decision="PROPOSE_DOWN")
    _, _, record = _assess(application, control, _post(NOW, ttft=5.0, running=4),
                           config=_config(latency_max=1.0))
    assert record.outcome == "INCONCLUSIVE"
    assert any("no automatic permissive revert" in r for r in record.reasons)


# --------------------------------------------------------------- no mutation
def test_assessment_never_mutates():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(_converged(), config=_config(), control=control)
    record = assessment.assess(_post(NOW), now=NOW)
    assert control.patch_calls == 0
    assert record.mutation_applied is False
    assert record.to_dict()["mutation_applied"] is False


# ------------------------------------------------------------ revert precondition
def _decision(*, decision="PROPOSE_DOWN", current=5, proposed=4, revision=2):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=INTENT, internal_key=INTERNAL_KEY,
        decided_at=FIXED, decision=decision, proposed_value=proposed,
        current_value=current, capability_bounds={"min": 1, "max": 8},
        transition_type="HOT", guardrails=[], evidence={}, window={}, reasons=[],
        overrides=[], safety={}, desired_revision=revision, policy={},
        mutation_applied=False,
    )


def test_supports_revert_requires_exact_reverse():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(_converged(), config=_config(latency_max=1.0),
                               control=control)
    record = assessment.assess(_post(NOW, ttft=5.0), now=NOW)
    assert record.outcome == "PROPOSE_REVERT"
    assert assessment.supports_revert(_decision(decision="PROPOSE_DOWN", current=5,
                                                proposed=4, revision=2)) is True
    assert assessment.supports_revert(_decision(decision="PROPOSE_UP", current=5,
                                                proposed=4, revision=2)) is False
    assert assessment.supports_revert(_decision(decision="PROPOSE_DOWN", current=5,
                                                proposed=3, revision=2)) is False
    assert assessment.supports_revert(None) is False
    assert manager.supports_revert(_decision(decision="PROPOSE_DOWN", current=5,
                                             proposed=4, revision=2)) is True


# ---------------------------------------------------------------- assessment gate
class FakeApprovalControl:
    def __init__(self, *, current=5, desired=None, revision=2, lo=1, hi=8, patch=None):
        self.current = current
        self.desired = dict(desired) if desired is not None else {INTERNAL_KEY: current}
        self.revision = revision
        self.lo, self.hi = lo, hi
        self.patch = patch
        self.patch_calls = []

    def capabilities(self):
        return [{"intent": INTENT, "internal_key": INTERNAL_KEY, "min": self.lo,
                 "max": self.hi, "transition": "HOT",
                 "supported_on_selected_engine": True}]

    def get_config(self):
        return {"observed": {INTERNAL_KEY: self.current},
                "desired": dict(self.desired), "desired_revision": self.revision}

    def patch_config(self, desired, if_match=None):
        self.patch_calls.append((dict(desired), if_match))
        code, payload = self.patch(dict(desired), if_match)
        if code == 200 and payload.get("status") == "CONVERGED":
            value = (payload.get("observed") or {}).get(INTERNAL_KEY)
            self.current = value
            self.desired = {INTERNAL_KEY: value}
            self.revision = payload.get("desired_revision", self.revision)
        return code, payload


def _proposal(*, decision="PROPOSE_UP", current=5, proposed=6, revision=2):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=INTENT, internal_key=INTERNAL_KEY,
        decided_at=FIXED, decision=decision, proposed_value=proposed,
        current_value=current, capability_bounds={"min": 1, "max": 8},
        transition_type="HOT",
        guardrails=[
            {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": True},
            {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": True},
        ],
        evidence={}, window={}, reasons=[], overrides=[], safety={},
        desired_revision=revision, policy={}, mutation_applied=False,
    )


def test_assessment_gate_defers_second_application():
    manager = AssessmentManager()
    assessment_control = FakeAssessmentControl(current=5, revision=2)
    manager.start(_converged(), config=_config(), control=assessment_control, now=NOW)

    approval_control = FakeApprovalControl(
        current=5, revision=2,
        patch=lambda d, m: (200, {"status": "CONVERGED",
                                  "observed": {INTERNAL_KEY: 6},
                                  "desired_revision": 3}))
    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0, operational_cooldown_s=60.0),
        approval_control,
        fresh_evidence=lambda: _proposal(),
        application_gate=manager.blocking_reason,
    )
    did = application.register(_proposal())
    assert did is not None

    blocked = application.approve(ApprovalRequest(did), now=NOW)
    assert blocked.status == "DEFERRED"                 # non-terminal, not STALE
    assert blocked.mutation_attempted is False and blocked.applied is False
    assert any("application blocked" in r for r in blocked.reasons)
    assert approval_control.patch_calls == []           # zero mutation
    assert application.pending_id() == did              # proposal preserved
    assert manager.blocking_reason() is not None

    # terminate the assessment -> gate lifts -> the preserved proposal applies
    manager.assess(_post(NOW), now=NOW)                 # KEEP (terminal)
    assert manager.blocking_reason() is None
    allowed = application.approve(ApprovalRequest(did), now=NOW)
    assert allowed.status == "CONVERGED"
    assert len(approval_control.patch_calls) == 1


def test_true_m7b_stale_remains_terminal():
    approval_control = FakeApprovalControl(
        current=5, revision=9,  # live revision no longer matches the proposal
        patch=lambda d, m: (200, {"status": "CONVERGED",
                                  "observed": {INTERNAL_KEY: 6},
                                  "desired_revision": 10}))
    application = SupervisorApplication(
        ApplicationConfig(), approval_control,
        fresh_evidence=lambda: _proposal(revision=2))
    did = application.register(_proposal(revision=2))

    out = application.approve(ApprovalRequest(did), now=NOW)
    assert out.status == "STALE"                        # genuine invalidation
    assert out.mutation_attempted is False
    assert approval_control.patch_calls == []
    # STALE is terminal: a repeat is a replay, not a re-attempt
    again = application.approve(ApprovalRequest(did), now=NOW)
    assert again.status == "STALE"
    assert any("already terminal" in r for r in again.reasons)


def test_abandon_lifts_gate():
    control = FakeAssessmentControl(current=5, revision=2)
    manager = AssessmentManager()
    manager.start(_converged(), config=_config(), control=control)
    assert manager.blocking_reason() is not None
    record = manager.abandon("operator cancelled the assessment")
    assert record.outcome == "INCONCLUSIVE"
    assert any("abandoned" in r for r in record.reasons)
    assert manager.blocking_reason() is None


def test_start_requires_converged_application():
    control = FakeAssessmentControl(current=5, revision=2)
    application = _converged()
    application.status = "APPLY_FAILED"
    application.applied = False
    with pytest.raises(ValueError):
        AssessmentManager().start(application, config=_config(), control=control)


# --------------------------------------------------------------- determinism
def test_identical_assessment_inputs_deterministic():
    r1 = _assess(_converged(), FakeAssessmentControl(current=5, revision=2), _post(NOW))[2]
    r2 = _assess(_converged(), FakeAssessmentControl(current=5, revision=2), _post(NOW))[2]
    a, b = r1.to_dict(), r2.to_dict()
    a.pop("assessed_at")
    b.pop("assessed_at")
    assert a == b
