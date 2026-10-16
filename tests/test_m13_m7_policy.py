"""M13 item 1: multi-control M7 policy proposal/approval tests.

Deterministic; no live engine. The policy application reuses the existing M7
lifecycle (single-use, TTL, If-Match, fresh-evidence, gate) and the existing
multi-control execution path.
"""

import time

from infengine.control import ReconcileStatus  # noqa: F401  (sanity import)
from infengine.supervisor.approval import (
    ApplicationConfig,
    ApplicationStatus,
    PolicyApprovalRequest,
    PolicyProposal,
    SupervisorPolicyApplication,
)


class FakePolicyControl:
    def __init__(self):
        self.current = {
            "runtime_max_running_seqs": 16,
            "runtime_max_scheduled_tokens": 4096,
        }
        self.revision = 1
        self.patches = []
        self.fail_mode = None  # None | "partial" | "failed" | "conflict"

    def capabilities(self):
        return [
            {"intent": "runtime_concurrency",
             "internal_key": "runtime_max_running_seqs",
             "supported_on_selected_engine": True, "transition": "HOT",
             "min": 1, "max": 64},
            {"intent": "runtime_max_scheduled_tokens",
             "internal_key": "runtime_max_scheduled_tokens",
             "supported_on_selected_engine": True, "transition": "HOT",
             "min": 1, "max": 8192},
        ]

    def get_config(self):
        return {"desired": dict(self.current), "observed": dict(self.current),
                "desired_revision": self.revision}

    def patch_config(self, desired, if_match=None):
        self.patches.append((dict(desired), if_match))
        if self.fail_mode == "conflict":
            return 409, {"status": "CONFLICT"}
        if self.fail_mode == "failed":
            return 200, {"status": "FAILED", "observed": dict(self.current),
                         "desired_revision": self.revision}
        for key, value in desired.items():
            self.current[key] = value
        self.revision += 1
        observed = dict(self.current)
        if self.fail_mode == "partial":
            observed = dict(self.current)
            observed["runtime_max_scheduled_tokens"] = 999
        return 200, {"status": "CONVERGED", "observed": observed,
                     "desired_revision": self.revision}


POLICY = {"runtime_concurrency": 8, "runtime_max_scheduled_tokens": 1024}


def _app(control, **kwargs):
    return SupervisorPolicyApplication(
        ApplicationConfig(proposal_ttl_s=300.0),
        control,
        fresh_evidence=lambda: dict(control.current),
        **kwargs,
    )


def test_two_control_proposal_approved_and_converged():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    assert pid is not None
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY),
        approver="yasir"))
    assert outcome.status == ApplicationStatus.CONVERGED.value
    assert outcome.applied is True
    assert control.current == {"runtime_max_running_seqs": 8,
                               "runtime_max_scheduled_tokens": 1024}
    # ONE multi-control policy request, bound to the expected revision
    assert control.patches == [({
        "runtime_max_running_seqs": 8, "runtime_max_scheduled_tokens": 1024}, 1)]
    assert outcome.normalized_policy == {
        "runtime_max_running_seqs": 8, "runtime_max_scheduled_tokens": 1024}


def test_changed_value_after_approval_rejected():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    wrong = dict(POLICY)
    wrong["runtime_concurrency"] = 4
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=wrong, approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert control.patches == []
    assert "binding mismatch" in " ".join(outcome.reasons)


def test_added_control_after_approval_rejected():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    extra = dict(POLICY)
    extra["runtime_kv_watermark"] = 0.1
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=extra, approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert control.patches == []


def test_removed_control_after_approval_rejected():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1,
        expected_policy={"runtime_concurrency": 8}, approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert control.patches == []


def test_live_revision_change_rejected():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    control.revision = 2  # a concurrent change landed after registration
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY),
        approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert control.patches == []
    assert "revision changed" in " ".join(outcome.reasons)


def test_binding_revision_mismatch_rejected():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=99, expected_policy=dict(POLICY),
        approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert control.patches == []


def test_expired_approval_rejected():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    outcome = app.approve(
        PolicyApprovalRequest(pid, expected_desired_revision=1,
                              expected_policy=dict(POLICY), approver="yasir"),
        now=time.time() + 10_000)
    assert outcome.status == ApplicationStatus.STALE.value
    assert "expired" in " ".join(outcome.reasons)
    assert control.patches == []


def test_single_use_enforced():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    first = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY), approver="a"))
    second = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY), approver="b"))
    assert first.status == ApplicationStatus.CONVERGED.value
    assert second.status == ApplicationStatus.STALE.value
    assert len(control.patches) == 1  # exactly one mutation


def test_partial_apply_is_not_approved_success():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    control.fail_mode = "partial"
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY), approver="yasir"))
    assert outcome.status == ApplicationStatus.APPLY_FAILED.value
    assert outcome.applied is False
    assert "whole-policy verification failed" in " ".join(outcome.reasons)


def test_failed_apply_is_not_success():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    control.fail_mode = "failed"
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY), approver="yasir"))
    assert outcome.status == ApplicationStatus.APPLY_FAILED.value


def test_if_match_conflict_at_apply_is_stale_zero_extra_mutation():
    control = FakePolicyControl()
    app = _app(control)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    control.fail_mode = "conflict"
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY), approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert outcome.mutation_attempted is True
    assert control.current["runtime_max_running_seqs"] == 16


def test_missing_fresh_evidence_refuses_to_apply():
    control = FakePolicyControl()
    app = SupervisorPolicyApplication(
        ApplicationConfig(), control, fresh_evidence=None, require_fresh_evidence=True)
    pid = app.register(PolicyProposal(policy=dict(POLICY)))
    outcome = app.approve(PolicyApprovalRequest(
        pid, expected_desired_revision=1, expected_policy=dict(POLICY), approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert "fresh" in " ".join(outcome.reasons)
    assert control.patches == []


def test_unknown_control_not_registrable():
    control = FakePolicyControl()
    app = _app(control)
    assert app.register(PolicyProposal(policy={"nope": 1})) is None
    assert app.last_reason() and "unknown" in app.last_reason()


def test_superseding_policy_retires_previous_proposal():
    control = FakePolicyControl()
    app = _app(control)
    first = app.register(PolicyProposal(policy={"runtime_concurrency": 8}))
    second = app.register(PolicyProposal(policy={"runtime_concurrency": 4}))
    assert second is not None and second != first
    # the old decision id is now terminal (STALE)
    outcome = app.approve(PolicyApprovalRequest(
        first, expected_desired_revision=1,
        expected_policy={"runtime_concurrency": 8}, approver="yasir"))
    assert outcome.status == ApplicationStatus.STALE.value
    assert control.patches == []
