"""M7D3: long_prefill_token_threshold + runtime_reserve_full_isl evaluation.

Both remaining vLLM persistent HOT controls are source-inspected and found
MANUAL_ONLY_INSUFFICIENT_EVIDENCE (no factual actuator-specific telemetry). The
source-derived effect models are machine-readable: long_prefill_token_threshold is
NON_MONOTONIC (0 disables the per-step cap); runtime_reserve_full_isl is BOOLEAN
(False->True = MORE_CONSERVATIVE, True->False = MORE_PERMISSIVE). Also covers the
boolean binding path (no numeric coercion) and effect-aware M7C assessment
(NON_MONOTONIC / boolean -> INCONCLUSIVE, never forced numeric).
"""

from infengine.supervisor import (
    INTENT,
    KV_WATERMARK_INTENT,
    LONG_PREFILL_INTENT,
    RESERVE_FULL_ISL_INTENT,
    SCHEDULED_TOKENS_INTENT,
    ApplicationConfig,
    ApplicationRecord,
    ApprovalRequest,
    AssessmentManager,
    ConcurrencyObjective,
    DecisionKind,
    DecisionRecord,
    Guardrail,
    Guardrails,
    PolicyEffect,
    SupervisorApplication,
    SupervisorConfig,
    policy_coverage,
    policy_for,
)
from infengine.telemetry import RuntimeSnapshot

FIXED = "2026-09-08T00:00:00+00:00"
NOW = 1_000_000.0


def _snap(running=4, waiting=6, kv=0.2, ttft=0.1):
    return RuntimeSnapshot(engine="vllm", instance_id="test", collected_at=FIXED,
                           source="test", requests_running=running,
                           requests_waiting=waiting, kv_usage=kv, ttft_p95_s=ttft)


def _win(now, *snaps):
    n = len(snaps)
    return [(now - (n - 1 - i) * 1.0, s) for i, s in enumerate(snaps)]


def _config():
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        objective=ConcurrencyObjective(waiting_high=1.0, waiting_low=0.0),
        guardrails=Guardrails(
            latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=1.0,
                              enabled=True, required_for_scale_up=True),
            kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=0.95,
                               enabled=True, required_for_scale_up=True),
        ),
    )


def _passing_guardrails():
    return [
        {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
        {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
    ]


class FakeControl:
    def __init__(self, *, intent, internal_key, current, lo=None, hi=None,
                 revision=1, patch=None):
        self.intent = intent
        self.internal_key = internal_key
        self.current = current
        self.lo, self.hi = lo, hi
        self.revision = revision
        self.patch = patch
        self.patch_calls = []

    def capabilities(self):
        return [{"intent": self.intent, "internal_key": self.internal_key,
                 "min": self.lo, "max": self.hi, "transition": "HOT",
                 "supported_on_selected_engine": True}]

    def get_config(self):
        return {"observed": {self.internal_key: self.current},
                "desired": {self.internal_key: self.current},
                "desired_revision": self.revision}

    def patch_config(self, desired, if_match=None):
        self.patch_calls.append((dict(desired), if_match))
        if self.patch is None:
            raise AssertionError("unexpected mutation path call")
        code, payload = self.patch(dict(desired), if_match)
        if code == 200 and payload.get("status") == "CONVERGED":
            value = (payload.get("observed") or {}).get(self.internal_key)
            self.current = value
            self.revision = payload.get("desired_revision", self.revision)
        return code, payload


def _bool_patch(value, revision=2):
    def patch(desired, if_match):
        key = next(iter(desired))
        return (200, {"status": "CONVERGED", "observed": {key: value},
                      "desired_revision": revision})
    return patch


def _bool_record(*, current, proposed, decision="PROPOSE_CHANGE", revision=1):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=RESERVE_FULL_ISL_INTENT,
        internal_key=RESERVE_FULL_ISL_INTENT, decided_at=FIXED, decision=decision,
        proposed_value=proposed, current_value=current, capability_bounds=None,
        transition_type="HOT", guardrails=_passing_guardrails(), evidence={},
        window={}, reasons=[], overrides=[], safety={}, desired_revision=revision,
        policy={}, effect=None, mutation_applied=False)


def _converged(*, intent, internal_key, previous, applied, effect, decision,
               revision_after=2):
    baseline = {"intent": intent, "internal_key": internal_key, "decision": decision,
                "current_value": previous, "proposed_value": applied,
                "desired_revision": 1, "effect": effect,
                "guardrails": _passing_guardrails(), "window": {"metrics": {}}}
    return ApplicationRecord(
        decision_id="m7d3-1", intent=intent, internal_key=internal_key,
        status="CONVERGED", stages=["APPROVED", "CONVERGED"],
        proposal_decision=decision, proposal_decided_at=FIXED, decided_at=FIXED,
        current_value=previous, requested_value=applied, observed_value=applied,
        desired_revision_before=1, desired_revision_after=revision_after, reasons=[],
        mutation_attempted=True, applied=True, cooldown_until=NOW - 1.0,
        approver="human", baseline_evidence=baseline, policy_effect=effect)


# ------------------------------------------------------ coverage findings
def _entries():
    return {(e["intent"], e["engine"]): e for e in policy_coverage()}


def test_remaining_controls_manual_only_and_unsupported():
    entries = _entries()
    for intent in (LONG_PREFILL_INTENT, RESERVE_FULL_ISL_INTENT):
        entry = entries[(intent, "vllm")]
        assert entry["status"] == "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"
        assert policy_for(intent, _config()) is None  # no policy -> no queue-only attribution


def test_long_prefill_effect_model_non_monotonic():
    entry = _entries()[(LONG_PREFILL_INTENT, "vllm")]
    assert entry["effect_model"] == "NON_MONOTONIC"
    assert entry["transition_effects"] and "note" in entry["transition_effects"]
    assert "0 disables" in entry["transition_effects"]["note"]
    assert "no factual long-prefill" in entry["reason"]


def test_reserve_full_isl_boolean_effect_model():
    entry = _entries()[(RESERVE_FULL_ISL_INTENT, "vllm")]
    assert entry["effect_model"] == "BOOLEAN"
    assert entry["transition_effects"] == {
        "false_to_true": "MORE_CONSERVATIVE",
        "true_to_false": "MORE_PERMISSIVE",
    }
    assert "blocked because full-ISL does not fit" in entry["reason"]


def test_monotonic_and_inverse_effect_models_source_derived():
    entries = _entries()
    assert entries[(INTENT, "vllm")]["effect_model"] == "MONOTONIC"
    assert entries[(INTENT, "vllm")]["transition_effects"]["increase"] == "MORE_PERMISSIVE"
    assert entries[(SCHEDULED_TOKENS_INTENT, "vllm")]["effect_model"] == "MONOTONIC"
    assert entries[(KV_WATERMARK_INTENT, "vllm")]["effect_model"] == "INVERSE_MONOTONIC"
    assert entries[(KV_WATERMARK_INTENT, "vllm")]["transition_effects"]["increase"] == \
        "MORE_CONSERVATIVE"


def test_effect_and_decision_kind_enums_have_non_monotonic_and_change():
    assert PolicyEffect.NON_MONOTONIC.value == "NON_MONOTONIC"
    assert DecisionKind.PROPOSE_CHANGE.value == "PROPOSE_CHANGE"


# ------------------------------------------------------ boolean binding
def test_boolean_binding_without_numeric_coercion():
    control = FakeControl(intent=RESERVE_FULL_ISL_INTENT,
                          internal_key=RESERVE_FULL_ISL_INTENT, current=True,
                          lo=None, hi=None, revision=1, patch=_bool_patch(False, 2))
    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0, operational_cooldown_s=60.0), control,
        intent=RESERVE_FULL_ISL_INTENT, internal_key=RESERVE_FULL_ISL_INTENT,
        fresh_evidence=lambda: _bool_record(current=True, proposed=False))
    did = application.register(_bool_record(current=True, proposed=False))
    assert did is not None
    out = application.approve(
        ApprovalRequest(did, expected_current_value=True, expected_proposed_value=False),
        now=NOW)
    assert out.status == "CONVERGED"
    assert out.requested_value is False and isinstance(out.requested_value, bool)
    assert out.observed_value is False and isinstance(out.observed_value, bool)
    assert control.current is False


def test_boolean_reject_zero_mutation():
    control = FakeControl(intent=RESERVE_FULL_ISL_INTENT,
                          internal_key=RESERVE_FULL_ISL_INTENT, current=True,
                          lo=None, hi=None, patch=None)
    application = SupervisorApplication(
        ApplicationConfig(), control, intent=RESERVE_FULL_ISL_INTENT,
        internal_key=RESERVE_FULL_ISL_INTENT,
        fresh_evidence=lambda: _bool_record(current=True, proposed=False))
    did = application.register(_bool_record(current=True, proposed=False))
    out = application.reject(ApprovalRequest(did), now=NOW)
    assert out.status == "REJECTED"
    assert control.patch_calls == []


# ------------------------------------------------------ effect-aware assessment
def test_boolean_change_assessment_has_no_numeric_direction():
    application = _converged(intent=RESERVE_FULL_ISL_INTENT,
                             internal_key=RESERVE_FULL_ISL_INTENT,
                             previous=True, applied=False, effect=None,
                             decision="PROPOSE_CHANGE")
    control = FakeControl(intent=RESERVE_FULL_ISL_INTENT,
                          internal_key=RESERVE_FULL_ISL_INTENT, current=False,
                          lo=None, hi=None, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(application, config=_config(), control=control, now=NOW)
    record = assessment.assess(_win(NOW, _snap(), _snap(), _snap()), now=NOW)
    assert record.outcome == "INCONCLUSIVE"
    assert record.direction is None               # boolean -> no numeric UP/DOWN
    assert record.policy_effect is None
    assert any("no directional policy effect" in r for r in record.reasons)


def test_non_monotonic_change_assessment_inconclusive():
    application = _converged(intent=LONG_PREFILL_INTENT,
                             internal_key=LONG_PREFILL_INTENT,
                             previous=128, applied=64, effect="NON_MONOTONIC",
                             decision="PROPOSE_UP")
    control = FakeControl(intent=LONG_PREFILL_INTENT,
                          internal_key=LONG_PREFILL_INTENT, current=64,
                          lo=0, hi=512, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(application, config=_config(), control=control, now=NOW)
    record = assessment.assess(_win(NOW, _snap(), _snap(), _snap()), now=NOW)
    assert record.outcome == "INCONCLUSIVE"
    assert record.policy_effect == "NON_MONOTONIC"
    assert any("no directional policy effect" in r for r in record.reasons)
    assert record.recommended_value is None       # never an automatic revert
