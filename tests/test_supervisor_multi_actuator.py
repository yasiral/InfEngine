"""M7D0/D1: generalized supervisory lifecycle + token-budget policies.

Proves: concurrency behavior is unchanged after generalization; the vLLM
token-budget policy requires factual budget-utilization evidence (not generic
queueing); human approval uses the existing If-Match path; fresh evidence is
revalidated before mutation; post-change assessment works for the token actuator;
the coverage matrix (incl. POLICY_PENDING_EVALUATION) is correct; SGLang prefill
does not falsely claim support; engine-specific semantics are not aliased; and the
engine-wide causal-isolation gate defers every other actuator while any
assessment is active (both directions).
"""

from infengine.supervisor import (
    INTERNAL_KEY,
    INTENT,
    SGLANG_PREFILL_INTENT,
    SCHEDULED_TOKENS_INTENT,
    SCHEDULED_TOKENS_KEY,
    ApplicationConfig,
    ApplicationRecord,
    ApprovalRequest,
    AssessmentConfig,
    AssessmentManager,
    ConcurrencyObjective,
    ConcurrencyPolicy,
    DecisionRecord,
    Guardrail,
    Guardrails,
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
TOKEN_MIN, TOKEN_MAX = 1, 8192


def _snap(running, waiting, kv=0.2, ttft=0.1, iteration=None, quality=None):
    kwargs = dict(engine="vllm", instance_id="test", collected_at=FIXED, source="test",
                  requests_running=running, requests_waiting=waiting, kv_usage=kv,
                  ttft_p95_s=ttft, quality=quality or {})
    if iteration is not None:
        kwargs["engine_specific"] = {"iteration_tokens_p95": iteration}
    return RuntimeSnapshot(**kwargs)


def _win(now, *snaps):
    n = len(snaps)
    return [(now - (n - 1 - i) * 1.0, s) for i, s in enumerate(snaps)]


def _guardrails(latency_max=1.0, latency_required=True, kv_max=0.9):
    return Guardrails(
        latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=latency_max,
                          enabled=True, required_for_scale_up=latency_required),
        kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=kv_max,
                           enabled=True, required_for_scale_up=True),
    )


def _config(*, latency_max=1.0, latency_required=True):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        objective=ConcurrencyObjective(waiting_high=1.0, waiting_low=0.0),
        guardrails=_guardrails(latency_max=latency_max, latency_required=latency_required),
    )


def _token_config(*, step=256, threshold=0.9, latency_max=1.0):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        signal_metrics=("requests_running", "requests_waiting", "iteration_tokens_p95"),
        objective=TokenBudgetObjective(budget_utilization_threshold=threshold,
                                       waiting_high=1.0, step_up=step, step_down=step),
        guardrails=_guardrails(latency_max=latency_max),
    )


def _passing_guardrails():
    return [
        {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
        {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
    ]


def _record(*, intent, internal_key, decision="PROPOSE_UP", current, proposed,
            revision=1, guardrails=None):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=intent, internal_key=internal_key,
        decided_at=FIXED, decision=decision, proposed_value=proposed,
        current_value=current, capability_bounds={"min": TOKEN_MIN, "max": TOKEN_MAX},
        transition_type="HOT",
        guardrails=(guardrails if guardrails is not None else _passing_guardrails()),
        evidence={}, window={}, reasons=[], overrides=[], safety={},
        desired_revision=revision, policy={}, mutation_applied=False,
    )


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


# -------------------------------------------- concurrency unchanged
def test_concurrency_defaults_and_behavior_unchanged():
    control = FakeControl(intent=INTENT, internal_key=INTERNAL_KEY, current=2, lo=1, hi=8)
    supervisor = Supervisor(_config(), control)
    assert supervisor.intent == INTENT and supervisor.internal_key == INTERNAL_KEY
    assert isinstance(supervisor._policy, ConcurrencyPolicy)
    assert isinstance(policy_for(INTENT, _config()), ConcurrencyPolicy)

    record = supervisor.observe(_win(NOW, _snap(2, 5), _snap(2, 6), _snap(2, 7)), now=NOW)
    assert record.intent == INTENT and record.internal_key == INTERNAL_KEY
    assert record.decision == "PROPOSE_UP" and record.proposed_value == 3


# -------------------------------------------- vLLM token budget (budget utilization)
def _token_supervisor(control, config=None):
    cfg = config or _token_config()
    return Supervisor(cfg, control, policy=TokenBudgetPolicy(cfg),
                      intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY)


def test_token_budget_proposes_up_only_when_budget_reached_with_demand():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=256, lo=TOKEN_MIN, hi=TOKEN_MAX)
    record = _token_supervisor(control).observe(
        _win(NOW, _snap(4, 5, iteration=256), _snap(4, 6, iteration=256),
             _snap(4, 7, iteration=256)), now=NOW)
    assert record.intent == SCHEDULED_TOKENS_INTENT
    assert record.decision == "PROPOSE_UP" and record.proposed_value == 512
    assert record.evidence["iteration_tokens_p95_median"] == 256


def test_token_budget_holds_when_budget_not_reached():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=256, lo=TOKEN_MIN, hi=TOKEN_MAX)
    # persistent waiting demand BUT steps well below the budget -> generic queueing
    # is NOT attributed to the token budget
    record = _token_supervisor(control).observe(
        _win(NOW, _snap(4, 5, iteration=64), _snap(4, 6, iteration=64),
             _snap(4, 7, iteration=64)), now=NOW)
    assert record.decision == "HOLD"
    assert any("within operating band" in r for r in record.reasons)


def test_token_budget_holds_when_utilization_evidence_unavailable():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=256, lo=TOKEN_MIN, hi=TOKEN_MAX)
    record = _token_supervisor(control).observe(
        _win(NOW, _snap(4, 5), _snap(4, 6), _snap(4, 7)), now=NOW)
    assert record.decision == "HOLD"
    assert any("budget-utilization evidence unavailable" in r for r in record.reasons)


def test_token_budget_bounds_hold_at_max_and_min():
    control_max = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                              current=TOKEN_MAX, lo=TOKEN_MIN, hi=TOKEN_MAX)
    at_max = _token_supervisor(control_max).observe(
        _win(NOW, _snap(4, 5, iteration=TOKEN_MAX), _snap(4, 6, iteration=TOKEN_MAX),
             _snap(4, 7, iteration=TOKEN_MAX)), now=NOW)
    assert at_max.decision == "HOLD"
    assert any("max bound" in r for r in at_max.reasons)

    control_min = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                              current=TOKEN_MIN, lo=TOKEN_MIN, hi=TOKEN_MAX)
    at_min = _token_supervisor(control_min).observe(
        _win(NOW, _snap(4, 5, ttft=5.0), _snap(4, 6, ttft=5.0), _snap(4, 7, ttft=5.0)), now=NOW)
    assert at_min.decision == "HOLD"
    assert any("min bound" in r for r in at_min.reasons)


def test_token_budget_guardrail_breach_proposes_bounded_down():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=256, lo=TOKEN_MIN, hi=TOKEN_MAX)
    record = _token_supervisor(control).observe(
        _win(NOW, _snap(4, 5, ttft=5.0), _snap(4, 6, ttft=5.0), _snap(4, 7, ttft=5.0)), now=NOW)
    assert record.decision == "PROPOSE_DOWN"
    assert record.proposed_value == max(TOKEN_MIN, 256 - 256)
    assert "guardrail breach" in record.reasons[0]


def test_token_budget_approval_uses_if_match_and_converges():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=256, lo=TOKEN_MIN, hi=TOKEN_MAX, revision=1,
                          patch=_ok_for(SCHEDULED_TOKENS_KEY, 512, 2))
    record = _record(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                     current=256, proposed=512, revision=1)
    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0, operational_cooldown_s=60.0), control,
        intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
        fresh_evidence=lambda: _record(intent=SCHEDULED_TOKENS_INTENT,
                                       internal_key=SCHEDULED_TOKENS_KEY,
                                       current=256, proposed=512, revision=1))
    did = application.register(record)
    out = application.approve(ApprovalRequest(did), now=NOW)
    assert out.status == "CONVERGED"
    assert control.patch_calls[0][1] == 1
    assert control.current == 512


def test_token_budget_fresh_evidence_revalidated_before_mutation():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=256, lo=TOKEN_MIN, hi=TOKEN_MAX, revision=1,
                          patch=_ok_for(SCHEDULED_TOKENS_KEY, 512, 2))
    application = SupervisorApplication(
        ApplicationConfig(), control, intent=SCHEDULED_TOKENS_INTENT,
        internal_key=SCHEDULED_TOKENS_KEY,
        fresh_evidence=lambda: _record(intent=SCHEDULED_TOKENS_INTENT,
                                       internal_key=SCHEDULED_TOKENS_KEY,
                                       current=256, proposed=1024, revision=1))
    application.register(_record(intent=SCHEDULED_TOKENS_INTENT,
                                 internal_key=SCHEDULED_TOKENS_KEY,
                                 current=256, proposed=512, revision=1))
    out = application.approve(ApprovalRequest(application.pending_id()), now=NOW)
    assert out.status == "STALE"
    assert any("no longer eligible" in r for r in out.reasons)
    assert control.patch_calls == []


# -------------------------------------------- post-change assessment (token budget)
def _converged_tokens(previous=256, applied=512, revision_after=2, cooldown_until=None,
                      latency_passed=True):
    baseline = {
        "intent": SCHEDULED_TOKENS_INTENT, "internal_key": SCHEDULED_TOKENS_KEY,
        "decision": "PROPOSE_UP", "current_value": previous, "proposed_value": applied,
        "desired_revision": 1,
        "guardrails": [
            {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": latency_passed},
            {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
             "required_for_scale_up": True, "available": True, "passed": True},
        ],
        "window": {"metrics": {}},
    }
    return ApplicationRecord(
        decision_id="tok-1", intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
        status="CONVERGED", stages=["APPROVED", "CONVERGED"],
        proposal_decision="PROPOSE_UP", proposal_decided_at=FIXED, decided_at=FIXED,
        current_value=previous, requested_value=applied, observed_value=applied,
        desired_revision_before=1, desired_revision_after=revision_after,
        reasons=[], mutation_attempted=True, applied=True,
        cooldown_until=(NOW - 1.0 if cooldown_until is None else cooldown_until),
        approver="human", baseline_evidence=baseline)


def _assess_tokens(samples, *, control_value=512, revision=2, latency_max=1.0):
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=control_value, lo=TOKEN_MIN, hi=TOKEN_MAX, revision=revision)
    manager = AssessmentManager()
    assessment = manager.start(
        _converged_tokens(), config=_config(latency_max=latency_max), control=control, now=NOW)
    return assessment, assessment.assess(samples, now=NOW)


def test_token_budget_post_change_keep():
    _, record = _assess_tokens(_win(NOW, _snap(4, 5), _snap(4, 6), _snap(4, 7)))
    assert record.outcome == "KEEP"
    assert record.intent == SCHEDULED_TOKENS_INTENT
    assert record.mutation_applied is False


def test_token_budget_post_change_guardrail_breach_proposes_revert():
    _, record = _assess_tokens(
        _win(NOW, _snap(4, 5, ttft=5.0), _snap(4, 6, ttft=5.0), _snap(4, 7, ttft=5.0)))
    assert record.outcome == "PROPOSE_REVERT"
    assert record.recommended_value == 256
    assert record.revert_direction == "PROPOSE_DOWN"
    assert record.mutation_applied is False


def test_token_budget_post_change_insufficient_then_terminal_at_deadline():
    control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                          current=512, lo=TOKEN_MIN, hi=TOKEN_MAX, revision=2)
    manager = AssessmentManager()
    assessment = manager.start(
        _converged_tokens(), config=_config(), control=control,
        assessment_config=AssessmentConfig(max_evidence_wait_s=30.0), now=NOW)
    waiting = assessment.assess(_win(NOW + 1, _snap(4, 5)), now=NOW + 1.0)
    assert waiting.waiting is True and waiting.terminal is False
    terminal = assessment.assess(_win(NOW + 35, _snap(4, 5)), now=NOW + 35.0)
    assert terminal.terminal is True and terminal.outcome == "INCONCLUSIVE"


# -------------------------------------------- coverage matrix
def test_coverage_matrix_states():
    entries = {(e["intent"], e["engine"]): e for e in policy_coverage()}
    assert len(entries) == 12

    assert entries[("runtime_concurrency", "vllm")]["status"] == "POLICY_SUPPORTED"
    assert entries[("runtime_concurrency", "sglang")]["status"] == "POLICY_SUPPORTED"
    assert entries[(SCHEDULED_TOKENS_INTENT, "vllm")]["status"] == "POLICY_SUPPORTED"
    assert entries[(SCHEDULED_TOKENS_INTENT, "sglang")]["status"] == "ENGINE_UNSUPPORTED"
    assert "iteration_tokens_p95" in entries[(SCHEDULED_TOKENS_INTENT, "vllm")]["required_evidence"]
    assert entries[(SGLANG_PREFILL_INTENT, "sglang")]["status"] == \
        "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"
    assert entries[(SGLANG_PREFILL_INTENT, "vllm")]["status"] == "ENGINE_UNSUPPORTED"
    kv = entries[("runtime_kv_watermark", "vllm")]
    assert kv["status"] == "POLICY_SUPPORTED"
    assert kv["direction_support"] == {
        "MORE_CONSERVATIVE": "POLICY_SUPPORTED",
        "MORE_PERMISSIVE": "MANUAL_ONLY_INSUFFICIENT_EVIDENCE",
    }
    for intent in ("long_prefill_token_threshold", "runtime_reserve_full_isl"):
        assert entries[(intent, "vllm")]["status"] == "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"
    for entry in entries.values():
        assert entry["engine"] and entry["bounds"] and entry["dtype"]
        assert entry["semantics"] and entry["reason"]
        assert "required_evidence" in entry


def test_sglang_prefill_not_policy_supported_and_not_aliased():
    assert policy_for(SGLANG_PREFILL_INTENT, _config()) is None
    assert isinstance(policy_for(SCHEDULED_TOKENS_INTENT, _config()), TokenBudgetPolicy)
    entries = {(e["intent"], e["engine"]): e for e in policy_coverage()}
    sched = entries[(SCHEDULED_TOKENS_INTENT, "vllm")]
    prefill = entries[(SGLANG_PREFILL_INTENT, "sglang")]
    assert sched["internal_key"] != prefill["internal_key"]
    assert sched["semantics"] != prefill["semantics"]
    assert "STRICT" in sched["semantics"] or "SCHEDULED" in sched["semantics"]
    assert "PREFILL" in prefill["semantics"]


def test_policy_coverage_filters_by_engine():
    vllm = policy_coverage("vllm")
    sglang = policy_coverage("sglang")
    assert len(vllm) == 6 and len(sglang) == 6
    assert all(e["engine"] == "vllm" for e in vllm)
    assert all(e["engine"] == "sglang" for e in sglang)


# -------------------------------------------- engine-wide causal isolation gate
def _converged_concurrency(previous=4, applied=5, revision_after=2, cooldown_until=None):
    baseline = {"intent": INTENT, "internal_key": INTERNAL_KEY, "decision": "PROPOSE_UP",
                "current_value": previous, "proposed_value": applied, "desired_revision": 1,
                "guardrails": _passing_guardrails(), "window": {"metrics": {}}}
    return ApplicationRecord(
        decision_id="conc-1", intent=INTENT, internal_key=INTERNAL_KEY,
        status="CONVERGED", stages=["APPROVED", "CONVERGED"],
        proposal_decision="PROPOSE_UP", proposal_decided_at=FIXED, decided_at=FIXED,
        current_value=previous, requested_value=applied, observed_value=applied,
        desired_revision_before=1, desired_revision_after=revision_after, reasons=[],
        mutation_attempted=True, applied=True,
        cooldown_until=(NOW - 1.0 if cooldown_until is None else cooldown_until),
        approver="human", baseline_evidence=baseline)


def test_engine_wide_gate_defers_token_while_concurrency_assessed():
    manager = AssessmentManager()
    conc_control = FakeControl(intent=INTENT, internal_key=INTERNAL_KEY, current=5, lo=1, hi=8,
                               revision=2)
    manager.start(_converged_concurrency(), config=_config(), control=conc_control, now=NOW)

    token_control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                                current=256, lo=TOKEN_MIN, hi=TOKEN_MAX, revision=2,
                                patch=_ok_for(SCHEDULED_TOKENS_KEY, 512, 3))
    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0), token_control, intent=SCHEDULED_TOKENS_INTENT,
        internal_key=SCHEDULED_TOKENS_KEY,
        fresh_evidence=lambda: _record(intent=SCHEDULED_TOKENS_INTENT,
                                       internal_key=SCHEDULED_TOKENS_KEY,
                                       current=256, proposed=512, revision=2),
        application_gate=manager.blocking_reason)  # engine-wide (no intent)
    did = application.register(_record(intent=SCHEDULED_TOKENS_INTENT,
                                       internal_key=SCHEDULED_TOKENS_KEY,
                                       current=256, proposed=512, revision=2))
    blocked = application.approve(ApprovalRequest(did), now=NOW)
    assert blocked.status == "DEFERRED"
    assert token_control.patch_calls == []
    assert application.pending_id() == did

    manager.assess(_win(NOW, _snap(5, 5), _snap(5, 6), _snap(5, 7)), intent=INTENT, now=NOW)
    assert manager.blocking_reason() is None
    allowed = application.approve(ApprovalRequest(did), now=NOW)
    assert allowed.status == "CONVERGED"
    assert len(token_control.patch_calls) == 1


def test_engine_wide_gate_defers_concurrency_while_token_assessed():
    manager = AssessmentManager()
    token_control = FakeControl(intent=SCHEDULED_TOKENS_INTENT, internal_key=SCHEDULED_TOKENS_KEY,
                                current=512, lo=TOKEN_MIN, hi=TOKEN_MAX, revision=2)
    manager.start(_converged_tokens(), config=_token_config(), control=token_control, now=NOW)

    conc_control = FakeControl(intent=INTENT, internal_key=INTERNAL_KEY, current=4, lo=1, hi=8,
                               revision=2, patch=_ok_for(INTERNAL_KEY, 5, 3))
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

    manager.abandon("test complete", intent=SCHEDULED_TOKENS_INTENT)
    allowed = application.approve(ApprovalRequest(did), now=NOW)
    assert allowed.status == "CONVERGED"
    assert len(conc_control.patch_calls) == 1
