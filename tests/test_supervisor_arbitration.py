"""M7E: deterministic multi-actuator arbitration tests.

Proves the required safety cases: NO_CANDIDATE / SELECTED / DEFERRED; effect
priority (MORE_CONSERVATIVE before MORE_PERMISSIVE); same-effect ties resolved by
the configured, visible actuator order (and overridable); manual-only actuators
and KV MORE_PERMISSIVE never enter the candidate set; determinism; evidence-epoch
/ desired-revision binding; the engine-wide gate defers with zero mutation and
losing candidates are not reused after it lifts; M7B fresh revalidation can reject
an M7E-selected proposal; and exactly one PATCH is issued for the selected
candidate.
"""

from infengine.supervisor import (
    INTERNAL_KEY,
    INTENT,
    KV_WATERMARK_INTENT,
    KV_WATERMARK_KEY,
    RESERVE_FULL_ISL_INTENT,
    SCHEDULED_TOKENS_INTENT,
    SCHEDULED_TOKENS_KEY,
    ApplicationConfig,
    ApplicationRecord,
    ApprovalRequest,
    ArbitrationActuator,
    ArbitrationConfig,
    AssessmentManager,
    ConcurrencyObjective,
    DecisionKind,
    DecisionRecord,
    Guardrail,
    Guardrails,
    KvWatermarkObjective,
    MultiActuatorArbitrator,
    PolicyEffect,
    PolicyEvaluation,
    PolicyChangeSlot,
    SupervisorApplication,
    SupervisorConfig,
    TokenBudgetObjective,
    combine_gates,
)
from infengine.telemetry import RuntimeSnapshot

FIXED = "2026-09-08T00:00:00+00:00"
NOW = 1_000_000.0
DEFAULT_OBSERVED = {INTERNAL_KEY: 4, SCHEDULED_TOKENS_KEY: 256, KV_WATERMARK_KEY: 0.1}
_UNSET = object()


# ----------------------------------------------------------------- snapshots
def _snap(running, waiting, kv=0.2, ttft=0.1, iteration=None, preemptions=None):
    engine_specific = {}
    if iteration is not None:
        engine_specific["iteration_tokens_p95"] = iteration
    if preemptions is not None:
        engine_specific["num_preemptions"] = preemptions
    return RuntimeSnapshot(
        engine="vllm", instance_id="arb", collected_at=FIXED, source="test",
        requests_running=running, requests_waiting=waiting, kv_usage=kv,
        ttft_p95_s=ttft, engine_specific=engine_specific,
    )


def _win(now, *snaps):
    n = len(snaps)
    return [(now - (n - 1 - i) * 1.0, s) for i, s in enumerate(snaps)]


def _demand_snaps(*, ttft=0.1, kv=0.2, iteration=256, preemptions=None, waiting=6):
    return _win(
        NOW,
        _snap(4, waiting, kv=kv, ttft=ttft, iteration=iteration, preemptions=preemptions),
        _snap(4, waiting, kv=kv, ttft=ttft, iteration=iteration,
              preemptions=(preemptions + 1 if preemptions is not None else None)),
        _snap(4, waiting, kv=kv, ttft=ttft, iteration=iteration,
              preemptions=(preemptions + 2 if preemptions is not None else None)),
    )


def _idle_snaps():
    return _win(
        NOW, _snap(4, 0, iteration=64), _snap(4, 0, iteration=64),
        _snap(4, 0, iteration=64),
    )


# ----------------------------------------------------------------- configs
def _guardrails(*, latency_max=1.0, kv_max=0.99):
    return Guardrails(
        latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=latency_max,
                          enabled=True, required_for_scale_up=True),
        kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=kv_max,
                           enabled=True, required_for_scale_up=True),
    )


def _concurrency_config(*, latency_max=1.0):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        objective=ConcurrencyObjective(waiting_high=1.0, waiting_low=0.0),
        guardrails=_guardrails(latency_max=latency_max),
    )


def _token_config(*, latency_max=1.0):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        signal_metrics=("requests_running", "requests_waiting", "iteration_tokens_p95"),
        objective=TokenBudgetObjective(budget_utilization_threshold=0.9,
                                       waiting_high=1.0, step_up=256, step_down=256),
        guardrails=_guardrails(latency_max=latency_max),
    )


def _kv_config(*, latency_max=1.0):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        signal_metrics=("requests_running", "requests_waiting", "kv_usage",
                        "num_preemptions"),
        objective=KvWatermarkObjective(kv_pressure_threshold=0.9, preemptions_min=1.0,
                                       step_up=0.05, max_watermark=0.9),
        guardrails=_guardrails(latency_max=latency_max),
    )


def _specs(*, latency_max=1.0, include=("kv", "concurrency", "tokens")):
    specs = []
    if "kv" in include:
        specs.append(ArbitrationActuator(KV_WATERMARK_INTENT, KV_WATERMARK_KEY,
                                         _kv_config(latency_max=latency_max)))
    if "concurrency" in include:
        specs.append(ArbitrationActuator(INTENT, INTERNAL_KEY,
                                         _concurrency_config(latency_max=latency_max)))
    if "tokens" in include:
        specs.append(ArbitrationActuator(SCHEDULED_TOKENS_INTENT, SCHEDULED_TOKENS_KEY,
                                         _token_config(latency_max=latency_max)))
    return specs


# ----------------------------------------------------------------- control
def _caps():
    return [
        {"intent": INTENT, "internal_key": INTERNAL_KEY, "min": 1, "max": 8,
         "transition": "HOT", "supported_on_selected_engine": True},
        {"intent": SCHEDULED_TOKENS_INTENT, "internal_key": SCHEDULED_TOKENS_KEY,
         "min": 1, "max": 8192, "transition": "HOT",
         "supported_on_selected_engine": True},
        {"intent": KV_WATERMARK_INTENT, "internal_key": KV_WATERMARK_KEY, "min": 0.0,
         "max": 1.0, "transition": "HOT", "supported_on_selected_engine": True},
    ]


class EngineControl:
    """Engine-wide read view with the existing conditional PATCH (for M7B)."""

    def __init__(self, observed=None, *, revision=1, desired=None, caps=None,
                 patch=None):
        self._observed = dict(observed if observed is not None else DEFAULT_OBSERVED)
        self._desired = dict(desired) if desired is not None else dict(self._observed)
        self._revision = revision
        self._caps = caps if caps is not None else _caps()
        self.patch = patch
        self.patch_calls = []

    def capabilities(self):
        return list(self._caps)

    def get_config(self):
        return {"observed": dict(self._observed), "desired": dict(self._desired),
                "desired_revision": self._revision}

    def patch_config(self, desired, if_match=None):
        self.patch_calls.append((dict(desired), if_match))
        if self.patch is None:
            raise AssertionError("unexpected mutation path call")
        code, payload = self.patch(dict(desired), if_match)
        if code == 200 and payload.get("status") == "CONVERGED":
            self._observed.update(payload.get("observed") or {})
            self._desired.update(payload.get("observed") or {})
            self._revision = payload.get("desired_revision", self._revision)
        return code, payload


def _ok_for(key, value, revision=2):
    def patch(desired, if_match):
        return (200, {"status": "CONVERGED", "observed": {key: value},
                      "desired_revision": revision})
    return patch


def _arbitrate(control, samples, *, gate=None, priority=None, now=NOW, specs=None,
               slot=_UNSET):
    if slot is _UNSET:
        slot = PolicyChangeSlot(AssessmentManager())
    return _make(control, slot=slot, gate=gate, priority=priority, specs=specs
                 ).arbitrate(samples, now=now)


def _make(control, *, slot, gate=None, priority=None, specs=None, engine="vllm"):
    return MultiActuatorArbitrator(
        ArbitrationConfig(engine=engine, priority=priority), control,
        specs if specs is not None else _specs(), slot=slot, gate=gate)


# ----------------------------------------------------------------- records
def _passing_guardrails():
    return [
        {"name": "latency", "metric": "ttft_p95_s", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
        {"name": "kv_usage", "metric": "kv_usage", "enabled": True,
         "required_for_scale_up": True, "available": True, "passed": True},
    ]


def _record(*, intent=INTENT, internal_key=INTERNAL_KEY, decision="PROPOSE_UP",
            current, proposed, revision=1, effect=None):
    return DecisionRecord(
        mode="OBSERVE_ONLY", intent=intent, internal_key=internal_key,
        decided_at=FIXED, decision=decision, proposed_value=proposed,
        current_value=current, capability_bounds={"min": 1, "max": 8192},
        transition_type="HOT", guardrails=_passing_guardrails(), evidence={},
        window={}, reasons=[], overrides=[], safety={}, desired_revision=revision,
        policy={}, effect=effect, mutation_applied=False)


def _converged_concurrency(previous=4, applied=5, revision_after=2):
    baseline = {
        "intent": INTENT, "internal_key": INTERNAL_KEY, "decision": "PROPOSE_UP",
        "current_value": previous, "proposed_value": applied, "desired_revision": 1,
        "effect": "MORE_PERMISSIVE", "guardrails": _passing_guardrails(),
        "window": {"metrics": {}},
    }
    return ApplicationRecord(
        decision_id="conc-arb-1", intent=INTENT, internal_key=INTERNAL_KEY,
        status="CONVERGED", stages=["APPROVED", "CONVERGED"],
        proposal_decision="PROPOSE_UP", proposal_decided_at=FIXED, decided_at=FIXED,
        current_value=previous, requested_value=applied, observed_value=applied,
        desired_revision_before=1, desired_revision_after=revision_after, reasons=[],
        mutation_attempted=True, applied=True, cooldown_until=NOW - 1.0,
        approver="human", baseline_evidence=baseline, policy_effect="MORE_PERMISSIVE")


class _FakePolicy:
    def __init__(self, evaluation):
        self.evaluation = evaluation
        self.calls = 0

    def evaluate(self, signals, *, current_value, bounds):
        self.calls += 1
        return self.evaluation


# ============================================================ candidate generation
def test_no_proposals_yields_no_candidate():
    control = EngineControl()
    record = _arbitrate(control, _idle_snaps())
    assert record.outcome == "NO_CANDIDATE"
    assert record.selected is None and record.selected_record is None
    assert record.candidates == ()
    assert {e.intent for e in record.evaluations} == {
        KV_WATERMARK_INTENT, INTENT, SCHEDULED_TOKENS_INTENT}
    assert all(e.decision == "HOLD" for e in record.evaluations)
    assert record.mutation_applied is False


def test_exactly_one_eligible_proposal_selected():
    control = EngineControl()
    record = _arbitrate(control, _demand_snaps(), specs=_specs(include=("concurrency",)))
    assert record.outcome == "SELECTED"
    assert record.considered == (INTENT,)
    assert record.candidates == (INTENT,)
    assert record.selected is not None and record.selected.intent == INTENT
    assert record.selected.decision == "PROPOSE_UP"
    assert record.selected.effect == "MORE_PERMISSIVE"
    assert record.selected_record is not None
    assert record.selected_record.proposed_value == 5


# ============================================================ effect priority
def test_conservative_outranks_permissive():
    control = EngineControl()
    record = _arbitrate(control, _demand_snaps(kv=0.95, preemptions=0))
    effects = {e.intent: e.effect for e in record.evaluations if e.is_candidate}
    assert effects[KV_WATERMARK_INTENT] == "MORE_CONSERVATIVE"
    assert effects[INTENT] == "MORE_PERMISSIVE"
    assert effects[SCHEDULED_TOKENS_INTENT] == "MORE_PERMISSIVE"
    assert record.selected is not None
    assert record.selected.intent == KV_WATERMARK_INTENT
    # every permissive candidate explains it lost on effect priority
    for evaluation in record.evaluations:
        if evaluation.is_candidate and evaluation.intent != KV_WATERMARK_INTENT:
            assert "lost effect priority" in evaluation.lost_reason


def test_two_conservative_resolved_by_configured_priority():
    control = EngineControl()
    record = _arbitrate(
        control, _demand_snaps(ttft=5.0),
        specs=_specs(latency_max=0.5, include=("concurrency", "tokens")))
    candidates = {e.intent: e for e in record.evaluations if e.is_candidate}
    assert set(candidates) == {INTENT, SCHEDULED_TOKENS_INTENT}
    assert all(e.effect == "MORE_CONSERVATIVE" for e in candidates.values())
    assert record.priority == (INTENT, SCHEDULED_TOKENS_INTENT)
    assert record.selected is not None and record.selected.intent == INTENT
    loser = candidates[SCHEDULED_TOKENS_INTENT]
    assert "lost configured actuator order" in loser.lost_reason


def test_two_permissive_resolved_by_configured_priority():
    control = EngineControl()
    record = _arbitrate(control, _demand_snaps(),
                        specs=_specs(include=("concurrency", "tokens")))
    candidates = {e.intent: e for e in record.evaluations if e.is_candidate}
    assert set(candidates) == {INTENT, SCHEDULED_TOKENS_INTENT}
    assert all(e.effect == "MORE_PERMISSIVE" for e in candidates.values())
    assert record.priority == (INTENT, SCHEDULED_TOKENS_INTENT)
    assert record.selected is not None and record.selected.intent == INTENT
    assert "lost configured actuator order" in \
        candidates[SCHEDULED_TOKENS_INTENT].lost_reason


def test_configured_priority_override_is_visible_and_effective():
    control = EngineControl()
    override = (SCHEDULED_TOKENS_INTENT, INTENT)
    record = _arbitrate(control, _demand_snaps(), priority=override,
                        specs=_specs(include=("concurrency", "tokens")))
    assert record.priority == override  # visible in the record
    assert record.selected is not None and record.selected.intent == SCHEDULED_TOKENS_INTENT
    concurrency = next(e for e in record.evaluations if e.intent == INTENT)
    assert "lost configured actuator order" in concurrency.lost_reason


# ============================================================ ineligibility
def test_manual_only_never_enters_candidate_set():
    fake = _FakePolicy(PolicyEvaluation(
        DecisionKind.PROPOSE_CHANGE, True, ["pretend proposal"], [], {},
        effect=PolicyEffect.NON_MONOTONIC))
    specs = list(_specs()) + [ArbitrationActuator(
        RESERVE_FULL_ISL_INTENT, RESERVE_FULL_ISL_INTENT,
        _concurrency_config(), policy=fake)]
    record = _arbitrate(EngineControl(), _demand_snaps(), specs=specs)
    evaluation = next(e for e in record.evaluations
                      if e.intent == RESERVE_FULL_ISL_INTENT)
    assert evaluation.support_status == "MANUAL_ONLY_INSUFFICIENT_EVIDENCE"
    assert evaluation.considered is False
    assert evaluation.is_candidate is False
    assert fake.calls == 0  # never even evaluated (coverage gate)
    assert RESERVE_FULL_ISL_INTENT not in record.candidates


def test_kv_permissive_never_enters_candidate_set():
    fake = _FakePolicy(PolicyEvaluation(
        DecisionKind.PROPOSE_DOWN, 0.05, ["pretend permissive decrease"], [], {},
        effect=PolicyEffect.MORE_PERMISSIVE))
    specs = [ArbitrationActuator(KV_WATERMARK_INTENT, KV_WATERMARK_KEY,
                                 _kv_config(), policy=fake)]
    record = _arbitrate(EngineControl(), _demand_snaps(), specs=specs)
    evaluation = record.evaluations[0]
    assert evaluation.support_status == "POLICY_SUPPORTED"
    assert evaluation.direction_supported is False
    assert evaluation.is_candidate is False
    assert "not POLICY_SUPPORTED" in evaluation.lost_reason
    assert record.outcome == "NO_CANDIDATE" and record.candidates == ()


# ============================================================ determinism / epoch
def test_same_inputs_and_config_produce_identical_result():
    first = _arbitrate(EngineControl(revision=1), _demand_snaps())
    second = _arbitrate(EngineControl(revision=1), _demand_snaps())
    assert first.arbitration_id == second.arbitration_id
    assert first.outcome == second.outcome
    assert first.priority == second.priority
    assert first.candidates == second.candidates
    assert [e.to_dict() for e in first.evaluations] == \
        [e.to_dict() for e in second.evaluations]
    assert first.selected is not None and second.selected is not None
    assert first.selected.to_dict() == second.selected.to_dict()


def test_desired_revision_and_time_bind_the_evidence_epoch():
    base = _arbitrate(EngineControl(revision=1), _demand_snaps())
    newer = _arbitrate(EngineControl(revision=2), _demand_snaps())
    later = _arbitrate(EngineControl(revision=1), _demand_snaps(), now=NOW + 1.0)
    assert base.desired_revision == 1 and base.evidence["desired_revision"] == 1
    assert newer.desired_revision == 2
    assert base.arbitration_id != newer.arbitration_id  # revision-bound
    assert base.arbitration_id != later.arbitration_id  # time-bound
    assert base.evidence["sample_count"] == 3
    assert base.evidence["first_sample_ts"] == NOW - 2.0
    assert base.evidence["last_sample_ts"] == NOW


# ============================================================ DEFERRED lifecycle
def test_active_assessment_defers_and_recomputes_after():
    control = EngineControl({INTERNAL_KEY: 5, SCHEDULED_TOKENS_KEY: 256,
                             KV_WATERMARK_KEY: 0.1}, revision=2)
    manager = AssessmentManager()
    manager.start(_converged_concurrency(), config=_concurrency_config(),
                  control=control, now=NOW)
    assert manager.blocking_reason() is not None
    slot = PolicyChangeSlot(manager)

    deferred = _arbitrate(control, _demand_snaps(), slot=slot)
    assert deferred.outcome == "DEFERRED"
    assert deferred.selected is None and deferred.selected_record is None
    assert deferred.candidates != ()  # diagnostics still evaluated
    assert control.patch_calls == []  # zero mutation
    assert deferred.mutation_applied is False

    manager.assess(_win(NOW, _snap(5, 5), _snap(5, 6), _snap(5, 7)), intent=INTENT,
                   now=NOW)
    assert manager.blocking_reason() is None

    # old losing candidates are NOT reused: an idle epoch now yields NO_CANDIDATE
    reused = _arbitrate(control, _idle_snaps(), slot=slot)
    assert reused.outcome == "NO_CANDIDATE" and reused.candidates == ()
    assert reused.selected is None


def test_combine_gates_returns_first_reason():
    combined = combine_gates(lambda: None, lambda: "second active", lambda: "third")
    assert combined() == "second active"
    assert combine_gates(None, lambda: None)() is None


# ============================================================ M7B handoff
def test_m7b_fresh_revalidation_can_reject_selected():
    control = EngineControl(patch=_ok_for(INTERNAL_KEY, 5, 2))
    record = _arbitrate(control, _demand_snaps(),
                        specs=_specs(include=("concurrency", "tokens")))
    assert record.selected is not None and record.selected.intent == INTENT
    selected = record.selected_record

    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0), control, intent=INTENT,
        internal_key=INTERNAL_KEY,
        fresh_evidence=lambda: _record(current=4, proposed=6, revision=1,
                                       effect="MORE_PERMISSIVE"))
    did = application.register(selected)
    out = application.approve(ApprovalRequest(did), now=NOW)
    assert out.status == "STALE"
    assert any("no longer eligible" in r for r in out.reasons)
    assert control.patch_calls == []  # zero mutation


def test_selected_candidate_issues_exactly_one_patch():
    control = EngineControl(patch=_ok_for(INTERNAL_KEY, 5, 2))
    record = _arbitrate(control, _demand_snaps(),
                        specs=_specs(include=("concurrency", "tokens")))
    assert len(record.candidates) == 2  # several candidates originally existed
    assert record.selected is not None and record.selected.intent == INTENT

    application = SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0), control, intent=INTENT,
        internal_key=INTERNAL_KEY,
        fresh_evidence=lambda: _record(current=4, proposed=5, revision=1,
                                       effect="MORE_PERMISSIVE"))
    did = application.register(record.selected_record)
    out = application.approve(ApprovalRequest(did), now=NOW)
    assert out.status == "CONVERGED"
    assert len(control.patch_calls) == 1
    assert control.get_config()["observed"][INTERNAL_KEY] == 5


# ============================================================ lifecycle slot
def _concurrency_application(control, *, refresh=None):
    return SupervisorApplication(
        ApplicationConfig(proposal_ttl_s=300.0, operational_cooldown_s=0.0), control,
        intent=INTENT, internal_key=INTERNAL_KEY,
        fresh_evidence=refresh or (lambda: _record(current=2, proposed=3, revision=1,
                                                   effect="MORE_PERMISSIVE")))


def _slot_setup(*, patch=None):
    control = EngineControl({INTERNAL_KEY: 2, SCHEDULED_TOKENS_KEY: 256,
                             KV_WATERMARK_KEY: 0.1}, revision=1, patch=patch)
    manager = AssessmentManager()
    slot = PolicyChangeSlot(manager)
    arb = _make(control, slot=slot, specs=_specs(include=("concurrency",)))
    return control, manager, slot, arb


def test_registered_pending_proposal_defers_next_arbitration():
    control, _, slot, arb = _slot_setup()
    first = arb.arbitrate(_demand_snaps(), now=NOW)
    assert first.outcome == "SELECTED"
    application = _concurrency_application(control)
    did = arb.register_selected(application, first)
    assert did is not None
    assert slot.blocking_reason() is not None

    second = arb.arbitrate(_demand_snaps(), now=NOW)
    assert second.outcome == "DEFERRED" and second.selected is None
    assert second.candidates != ()  # diagnostics only
    assert control.patch_calls == []  # zero mutation
    assert "pending human decision" in second.evidence["lifecycle_slot_reason"]


def test_rejected_application_releases_slot():
    control, _, slot, arb = _slot_setup()
    first = arb.arbitrate(_demand_snaps(), now=NOW)
    application = _concurrency_application(control)
    did = arb.register_selected(application, first)
    out = application.reject(ApprovalRequest(did), now=NOW)
    assert out.status == "REJECTED"
    arb.note_application(out)
    assert slot.blocking_reason() is None
    assert arb.arbitrate(_demand_snaps(), now=NOW).outcome == "SELECTED"


def test_stale_application_releases_slot():
    control, _, slot, arb = _slot_setup(patch=_ok_for(INTERNAL_KEY, 3, 2))
    first = arb.arbitrate(_demand_snaps(), now=NOW)
    application = _concurrency_application(control)
    did = arb.register_selected(application, first)
    out = application.approve(ApprovalRequest(did, expected_current_value=999), now=NOW)
    assert out.status == "STALE"
    assert control.patch_calls == []
    arb.note_application(out)
    assert slot.blocking_reason() is None
    assert arb.arbitrate(_demand_snaps(), now=NOW).outcome == "SELECTED"


def test_converged_keeps_slot_blocked_through_assessment():
    control, manager, slot, arb = _slot_setup(patch=_ok_for(INTERNAL_KEY, 3, 2))
    first = arb.arbitrate(_demand_snaps(), now=NOW)
    application = _concurrency_application(control)
    did = arb.register_selected(application, first)
    out = application.approve(ApprovalRequest(did), now=NOW)
    assert out.status == "CONVERGED"
    assert len(control.patch_calls) == 1
    arb.note_application(out)

    # CONVERGED, before M7C start: awaiting assessment -> still blocked
    assert slot.blocking_reason() is not None
    assert arb.arbitrate(_demand_snaps(), now=NOW).outcome == "DEFERRED"

    # M7C active -> still blocked
    manager.start(out, config=_concurrency_config(), control=control, now=NOW)
    assert slot.blocking_reason() is not None
    assert arb.arbitrate(_demand_snaps(), now=NOW).outcome == "DEFERRED"
    assert len(control.patch_calls) == 1  # no second mutation


def test_terminal_assessment_releases_slot():
    control, manager, slot, arb = _slot_setup(patch=_ok_for(INTERNAL_KEY, 3, 2))
    first = arb.arbitrate(_demand_snaps(), now=NOW)
    application = _concurrency_application(control)
    did = arb.register_selected(application, first)
    out = application.approve(ApprovalRequest(did), now=NOW)
    arb.note_application(out)
    manager.start(out, config=_concurrency_config(), control=control, now=NOW)

    manager.assess(_win(NOW, _snap(4, 5), _snap(4, 6), _snap(4, 7)), intent=INTENT,
                   now=NOW)
    assessment = manager.assessment_for(INTENT)
    assert assessment is not None and assessment.active is False
    assert slot.blocking_reason() is None
    released = arb.arbitrate(_demand_snaps(), now=NOW)
    assert released.outcome == "SELECTED"


def test_unavailable_slot_state_fails_closed():
    control = EngineControl({INTERNAL_KEY: 2, SCHEDULED_TOKENS_KEY: 256,
                             KV_WATERMARK_KEY: 0.1}, revision=1)
    specs = _specs(include=("concurrency",))

    explicit_none = _make(control, slot=None, specs=specs).arbitrate(
        _demand_snaps(), now=NOW)
    assert explicit_none.outcome == "DEFERRED" and explicit_none.selected is None
    assert "fail closed" in explicit_none.evidence["lifecycle_slot_reason"]

    no_manager = _make(control, slot=PolicyChangeSlot(None), specs=specs).arbitrate(
        _demand_snaps(), now=NOW)
    assert no_manager.outcome == "DEFERRED" and no_manager.selected is None
    assert "fail closed" in no_manager.evidence["lifecycle_slot_reason"]

    class _RaisingSlot:
        def blocking_reason(self):
            raise RuntimeError("slot provider down")

    raising = _make(control, slot=_RaisingSlot(), specs=specs).arbitrate(
        _demand_snaps(), now=NOW)
    assert raising.outcome == "DEFERRED" and raising.selected is None
    assert "fail closed" in raising.evidence["lifecycle_slot_reason"]
    assert control.patch_calls == []


def test_register_selected_without_selection_is_noop():
    control, _, slot, arb = _slot_setup()
    empty = _arbitrate(control, _idle_snaps(), specs=_specs(include=("concurrency",)))
    application = _concurrency_application(control)
    assert arb.register_selected(application, empty) is None
    assert slot.blocking_reason() is None
