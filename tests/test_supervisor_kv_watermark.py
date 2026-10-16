"""M7D2: vLLM KV-watermark supervisory policy (transition-effect + asymmetric).

Covers: explicit policy-effect mapping (concurrency/token increase = MORE_PERMISSIVE;
watermark increase = MORE_CONSERVATIVE); watermark conservative increase on factual
KV/preemption pressure; no decrease on generic queueing; insufficient pressure
evidence -> HOLD; float values preserved exactly through proposal/approval/binding;
fresh-evidence revalidation; M7C assessment driven by policy effect (not numeric
UP/DOWN); coverage direction support; engine-wide gate.
"""

from infengine.supervisor import (
    INTERNAL_KEY,
    INTENT,
    KV_WATERMARK_INTENT,
    KV_WATERMARK_KEY,
    ApplicationConfig,
    ApplicationRecord,
    ApprovalRequest,
    AssessmentManager,
    ConcurrencyObjective,
    ConcurrencyPolicy,
    DecisionKind,
    DecisionRecord,
    Guardrail,
    Guardrails,
    KvWatermarkObjective,
    KvWatermarkPolicy,
    PolicyEffect,
    Supervisor,
    SupervisorApplication,
    SupervisorConfig,
    TokenBudgetObjective,
    TokenBudgetPolicy,
    policy_coverage,
    policy_for,
)
from infengine.telemetry import RuntimeSnapshot

FIXED = "2026-09-08T00:00:00+00:00"
NOW = 1_000_000.0


def _snap(running=4, waiting=6, kv=0.2, ttft=0.1, preemptions=None, iteration=None):
    kwargs = dict(engine="vllm", instance_id="test", collected_at=FIXED, source="test",
                  requests_running=running, requests_waiting=waiting, kv_usage=kv,
                  ttft_p95_s=ttft)
    engine_specific = {}
    if preemptions is not None:
        engine_specific["num_preemptions"] = preemptions
    if iteration is not None:
        engine_specific["iteration_tokens_p95"] = iteration
    if engine_specific:
        kwargs["engine_specific"] = engine_specific
    return RuntimeSnapshot(**kwargs)


def _win(now, *snaps):
    n = len(snaps)
    return [(now - (n - 1 - i) * 1.0, s) for i, s in enumerate(snaps)]


def _guardrails(latency_max=1.0, kv_max=0.95):
    return Guardrails(
        latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=latency_max,
                          enabled=True, required_for_scale_up=True),
        kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=kv_max,
                           enabled=True, required_for_scale_up=True),
    )


def _kv_config(*, step=0.05, kv_threshold=0.9, preemptions_min=1.0):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        signal_metrics=("requests_running", "requests_waiting", "kv_usage", "num_preemptions"),
        objective=KvWatermarkObjective(kv_pressure_threshold=kv_threshold,
                                       preemptions_min=preemptions_min, step_up=step),
        guardrails=_guardrails(),
    )


def _kv_supervisor(control, config=None):
    cfg = config or _kv_config()
    return Supervisor(cfg, control, policy=KvWatermarkPolicy(cfg),
                      intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY)


class FakeControl:
    def __init__(self, *, intent, internal_key, current, lo, hi, revision=1,
                 desired=None, patch=None, supported=True, transition="HOT"):
        self.intent = intent
        self.internal_key = internal_key
        self.current = current
        self.lo, self.hi = lo, hi
        self.revision = revision
        self.desired = dict(desired) if desired is not None else {internal_key: current}
        self.patch = patch
        self.supported = supported
        self.transition = transition
        self.patch_calls = []

    def capabilities(self):
        return [{"intent": self.intent, "internal_key": self.internal_key,
                 "min": self.lo, "max": self.hi, "transition": self.transition,
                 "supported_on_selected_engine": self.supported}]

    def get_config(self):
        return {"observed": {self.internal_key: self.current},
                "desired": dict(self.desired), "desired_revision": self.revision}

    def patch_config(self, desired, if_match=None):
        self.patch_calls.append((dict(desired), if_match))
        if self.patch is None:
            raise AssertionError("unexpected mutation path call")
        code, payload = self.patch(dict(desired), if_match)
        if code == 200 and payload.get("status") == "CONVERGED":
            value = (payload.get("observed") or {}).get(self.internal_key)
            self.current = value
            self.desired = {self.internal_key: value}
            self.revision = payload.get("desired_revision", self.revision)
        return code, payload


def _ok_for(key, value, revision=2):
    def patch(desired, if_match):
        return (200, {"status": "CONVERGED", "observed": {key: value},
                      "desired_revision": revision})
    return patch


def _passing_guardrails():
    return [
        {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
        {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
    ]


def _record(*, current, proposed, decision="PROPOSE_UP", revision=1, effect=None,
            intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=intent, internal_key=internal_key,
        decided_at=FIXED, decision=decision, proposed_value=proposed,
        current_value=current, capability_bounds={"min": 0.0, "max": 1.0},
        transition_type="HOT", guardrails=_passing_guardrails(), evidence={},
        window={}, reasons=[], overrides=[], safety={}, desired_revision=revision,
        policy={}, effect=effect, mutation_applied=False,
    )


# ------------------------------------------------------ transition-effect map
def test_effect_mapping_orientation():
    conc = ConcurrencyPolicy(SupervisorConfig(objective=ConcurrencyObjective()))
    token = TokenBudgetPolicy(SupervisorConfig(objective=TokenBudgetObjective()))
    kv = KvWatermarkPolicy(_kv_config())
    # increase = MORE_PERMISSIVE
    for policy in (conc, token):
        assert policy.effect_for(DecisionKind.PROPOSE_UP) is PolicyEffect.MORE_PERMISSIVE
        assert policy.effect_for(DecisionKind.PROPOSE_DOWN) is PolicyEffect.MORE_CONSERVATIVE
    # watermark: increase = MORE_CONSERVATIVE, decrease = MORE_PERMISSIVE
    assert kv.effect_for(DecisionKind.PROPOSE_UP) is PolicyEffect.MORE_CONSERVATIVE
    assert kv.effect_for(DecisionKind.PROPOSE_DOWN) is PolicyEffect.MORE_PERMISSIVE


# ------------------------------------------------------ watermark policy
def test_watermark_conservative_increase_on_kv_pressure():
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=0.2, lo=0.0, hi=1.0)
    record = _kv_supervisor(control).observe(
        _win(NOW,
             _snap(kv=0.95, preemptions=10),
             _snap(kv=0.95, preemptions=12),
             _snap(kv=0.96, preemptions=15)),
        now=NOW)
    assert record.decision == "PROPOSE_UP"
    assert record.effect == "MORE_CONSERVATIVE"
    assert record.proposed_value == round(0.2 + 0.05, 4)  # 0.25, float preserved
    assert record.evidence["preemptions_in_window"] == 5


def test_watermark_no_decrease_on_generic_queueing():
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=0.5, lo=0.0, hi=1.0)
    record = _kv_supervisor(control).observe(
        _win(NOW,
             _snap(waiting=50, kv=0.2, preemptions=5),
             _snap(waiting=50, kv=0.2, preemptions=5),
             _snap(waiting=50, kv=0.2, preemptions=5)),
        now=NOW)
    assert record.decision == "HOLD"
    assert record.proposed_value is None
    assert any("permissive decrease unsupported" in r for r in record.reasons)


def test_watermark_holds_when_pressure_evidence_unavailable():
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=0.5, lo=0.0, hi=1.0)
    record = _kv_supervisor(control).observe(  # no num_preemptions on the snapshots
        _win(NOW, _snap(kv=0.95), _snap(kv=0.95), _snap(kv=0.96)), now=NOW)
    assert record.decision == "HOLD"
    assert any("num_preemptions" in r and "insufficient telemetry" in r
               for r in record.reasons)


def test_watermark_at_cap_holds():
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=0.9, lo=0.0, hi=1.0)
    record = _kv_supervisor(control).observe(
        _win(NOW,
             _snap(kv=0.99, preemptions=10),
             _snap(kv=0.99, preemptions=12),
             _snap(kv=0.99, preemptions=15)),
        now=NOW)
    assert record.decision == "HOLD"
    assert any("max watermark cap" in r for r in record.reasons)


# --------------------------------------- float preservation + approval + binding
def test_watermark_float_preserved_through_proposal_approval():
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=0.2, lo=0.0, hi=1.0, revision=1,
                          patch=_ok_for(KV_WATERMARK_KEY, 0.25, 2))
    record = _record(current=0.2, proposed=0.25, effect="MORE_CONSERVATIVE")
    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0, operational_cooldown_s=60.0), control,
        intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
        fresh_evidence=lambda: _record(current=0.2, proposed=0.25,
                                       effect="MORE_CONSERVATIVE"))
    did = application.register(record)
    out = application.approve(ApprovalRequest(did, expected_current_value=0.2,
                                              expected_proposed_value=0.25), now=NOW)
    assert out.status == "CONVERGED"
    assert out.requested_value == 0.25 and out.observed_value == 0.25
    assert isinstance(out.requested_value, float)
    assert out.policy_effect == "MORE_CONSERVATIVE"
    assert control.current == 0.25


def test_watermark_fresh_evidence_revalidated_before_mutation():
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=0.2, lo=0.0, hi=1.0, revision=1,
                          patch=_ok_for(KV_WATERMARK_KEY, 0.25, 2))
    application = SupervisorApplication(
        ApplicationConfig(), control, intent=KV_WATERMARK_INTENT,
        internal_key=KV_WATERMARK_KEY,
        fresh_evidence=lambda: _record(current=0.2, proposed=0.3,
                                       effect="MORE_CONSERVATIVE"))
    application.register(_record(current=0.2, proposed=0.25, effect="MORE_CONSERVATIVE"))
    out = application.approve(ApprovalRequest(application.pending_id()), now=NOW)
    assert out.status == "STALE"
    assert any("no longer eligible" in r for r in out.reasons)
    assert control.patch_calls == []


# --------------------------------------- assessment uses policy effect
def _converged_kv(previous, applied, effect, *, revision_after=2, latency_passed=True):
    baseline = {
        "intent": KV_WATERMARK_INTENT, "internal_key": KV_WATERMARK_KEY,
        "decision": "PROPOSE_UP" if effect == "MORE_CONSERVATIVE" else "PROPOSE_DOWN",
        "current_value": previous, "proposed_value": applied, "desired_revision": 1,
        "effect": effect,
        "guardrails": [
            {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": latency_passed},
            {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": True},
        ],
        "window": {"metrics": {}},
    }
    return ApplicationRecord(
        decision_id="kv-1", intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
        status="CONVERGED", stages=["APPROVED", "CONVERGED"],
        proposal_decision="PROPOSE_UP", proposal_decided_at=FIXED, decided_at=FIXED,
        current_value=previous, requested_value=applied, observed_value=applied,
        desired_revision_before=1, desired_revision_after=revision_after, reasons=[],
        mutation_attempted=True, applied=True, cooldown_until=NOW - 1.0,
        approver="human", baseline_evidence=baseline, policy_effect=effect)


def _assess_kv(application, *, control_value, latency_max=1.0, samples=None):
    control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                          current=control_value, lo=0.0, hi=1.0, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(application, config=_kv_config(), control=control, now=NOW)
    return assessment.assess(samples or _win(NOW, _snap(), _snap(), _snap()), now=NOW)


def test_assessment_conservative_change_does_not_revert_permissively():
    # numeric INCREASE 0.2 -> 0.25 but effect MORE_CONSERVATIVE: a breached
    # guardrail must NOT produce a permissive revert.
    application = _converged_kv(0.2, 0.25, "MORE_CONSERVATIVE")
    record = _assess_kv(application, control_value=0.25,
                        samples=_win(NOW, _snap(ttft=5.0), _snap(ttft=5.0), _snap(ttft=5.0)))
    assert record.outcome == "INCONCLUSIVE"
    assert record.recommended_value is None
    assert any("MORE_CONSERVATIVE" in r for r in record.reasons)
    # passing guardrails -> recovery/KEEP
    keeper = _assess_kv(application, control_value=0.25)
    assert keeper.outcome == "KEEP"


def test_assessment_permissive_change_may_revert():
    # numeric DECREASE 0.25 -> 0.2 with effect MORE_PERMISSIVE: a newly breached
    # baseline-passing guardrail -> PROPOSE_REVERT to the exact previous value.
    application = _converged_kv(0.25, 0.2, "MORE_PERMISSIVE")
    record = _assess_kv(application, control_value=0.2,
                        samples=_win(NOW, _snap(ttft=5.0), _snap(ttft=5.0), _snap(ttft=5.0)))
    assert record.outcome == "PROPOSE_REVERT"
    assert record.recommended_value == 0.25
    assert record.revert_direction == "PROPOSE_UP"
    assert record.mutation_applied is False


# ------------------------------------------------------ coverage
def test_watermark_coverage_direction_support():
    entries = {(e["intent"], e["engine"]): e for e in policy_coverage()}
    kv = entries[(KV_WATERMARK_INTENT, "vllm")]
    assert kv["status"] == "POLICY_SUPPORTED"
    assert kv["direction_support"] == {
        "MORE_CONSERVATIVE": "POLICY_SUPPORTED",
        "MORE_PERMISSIVE": "MANUAL_ONLY_INSUFFICIENT_EVIDENCE",
    }
    assert "kv_usage" in kv["required_evidence"] and "num_preemptions" in kv["required_evidence"]
    assert isinstance(policy_for(KV_WATERMARK_INTENT, _kv_config()), KvWatermarkPolicy)
    # long-prefill / reserve-full-ISL are manual-only (M7D3)
    assert entries[("long_prefill_token_threshold", "vllm")]["status"] == \
        "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"
    assert entries[("runtime_reserve_full_isl", "vllm")]["status"] == \
        "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"


# ------------------------------------------------------ engine-wide gate
def test_engine_wide_gate_defers_concurrency_while_watermark_assessed():
    manager = AssessmentManager()
    kv_control = FakeControl(intent=KV_WATERMARK_INTENT, internal_key=KV_WATERMARK_KEY,
                             current=0.25, lo=0.0, hi=1.0, revision=2)
    manager.start(_converged_kv(0.2, 0.25, "MORE_CONSERVATIVE"),
                  config=_kv_config(), control=kv_control, now=NOW)

    conc_control = FakeControl(intent=INTENT, internal_key=INTERNAL_KEY, current=4,
                               lo=1, hi=8, revision=2, patch=_ok_for(INTERNAL_KEY, 5, 3))
    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0), conc_control, intent=INTENT,
        internal_key=INTERNAL_KEY,
        fresh_evidence=lambda: _record(intent=INTENT, internal_key=INTERNAL_KEY,
                                       current=4, proposed=5, revision=2),
        application_gate=manager.blocking_reason)
    did = application.register(_record(intent=INTENT, internal_key=INTERNAL_KEY,
                                       current=4, proposed=5, revision=2))
    blocked = application.approve(ApprovalRequest(did), now=NOW)
    assert blocked.status == "DEFERRED"
    assert conc_control.patch_calls == []

    manager.abandon("done", intent=KV_WATERMARK_INTENT)
    allowed = application.approve(ApprovalRequest(did), now=NOW)
    assert allowed.status == "CONVERGED"
    assert len(conc_control.patch_calls) == 1
