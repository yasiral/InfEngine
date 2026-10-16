"""M7A: deterministic supervisory controller tests (OBSERVE_ONLY).

All inputs are synthetic telemetry snapshots and a fake read-only control view,
so every decision is fully deterministic. Covers: saturation with waiting
demand, no demand (HOLD), latency guardrail failure, KV guardrail failure,
required-vs-non-required/unavailable guardrails, stale-sample exclusion from
decision evidence, cooldown, capability bounds, insufficient telemetry,
identical-input determinism, observe-only non-mutation, convergence/conflict
safety, and the factual fresh-only window.
"""

import json

from infengine.supervisor import (
    INTERNAL_KEY,
    INTENT,
    CapabilityBounds,
    ConcurrencyPolicy,
    Supervisor,
    SupervisorConfig,
    build_signal_window,
)
from infengine.telemetry import RuntimeSnapshot

ISO = "2026-09-08T00:00:00+00:00"


def _snap(running=None, waiting=None, kv=None, ttft=None, quality=None):
    kwargs = {
        "engine": "sglang",
        "instance_id": "test",
        "collected_at": ISO,
        "source": "test",
        "quality": quality or {},
    }
    if running is not None:
        kwargs["requests_running"] = running
    if waiting is not None:
        kwargs["requests_waiting"] = waiting
    if kv is not None:
        kwargs["kv_usage"] = kv
    if ttft is not None:
        kwargs["ttft_p95_s"] = ttft
    return RuntimeSnapshot(**kwargs)


def _win(now, *snaps):
    n = len(snaps)
    return [(now - (n - 1 - i) * 1.0, s) for i, s in enumerate(snaps)]


def _sat(now):
    """Fresh saturation-with-waiting-demand window with passing guardrails."""
    return _win(
        now,
        _snap(2, 5, kv=0.3, ttft=0.1),
        _snap(2, 6, kv=0.3, ttft=0.1),
        _snap(2, 7, kv=0.3, ttft=0.1),
    )


def _config(**kw):
    cfg = SupervisorConfig(window_s=10.0, min_samples=3, cooldown_s=60.0)
    for key, value in kw.items():
        setattr(cfg, key, value)
    return cfg


class FakeControl:
    """Read-only control view; ``patch_config`` raises if ever called."""

    def __init__(self, *, current=2, desired=None, revision=1, lo=1, hi=8,
                 transition="HOT", capability=True):
        self.current = current
        self.desired = dict(desired) if desired is not None else {INTERNAL_KEY: current}
        self.revision = revision
        self.lo, self.hi, self.transition, self.capability = lo, hi, transition, capability
        self.mutations = 0

    def capabilities(self):
        if not self.capability:
            return []
        return [{
            "intent": INTENT,
            "internal_key": INTERNAL_KEY,
            "min": self.lo,
            "max": self.hi,
            "transition": self.transition,
            "supported_on_selected_engine": True,
        }]

    def get_config(self):
        return {
            "observed": {INTERNAL_KEY: self.current},
            "desired": dict(self.desired),
            "desired_revision": self.revision,
        }

    def patch_config(self, *args, **kwargs):
        self.mutations += 1
        raise AssertionError("OBSERVE_ONLY must never call the mutation path")


# -------------------------------------------------------- required decision cases
def test_saturation_with_waiting_demand_proposes_up():
    supervisor = Supervisor(_config(), FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(_sat(now), now=now)
    assert rec.decision == "PROPOSE_UP"
    assert rec.proposed_value == 3
    assert rec.current_value == 2
    assert rec.capability_bounds == {"min": 1, "max": 8}
    assert rec.transition_type == "HOT"
    assert rec.mutation_applied is False
    assert any("saturation with waiting demand" in r for r in rec.reasons)
    assert rec.evidence["requests_waiting_median"] == 6.0
    latency = next(g for g in rec.guardrails if g["name"] == "latency")
    assert latency["passed"] is True and latency["available"] is True


def test_no_demand_holds_not_down():
    """An idle admission ceiling consumes no provisioned capacity: no demand
    alone must NOT reduce concurrency."""
    supervisor = Supervisor(_config(), FakeControl(current=4))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(1, 0, kv=0.1, ttft=0.1),
             _snap(1, 0, kv=0.1, ttft=0.1),
             _snap(0, 0, kv=0.1, ttft=0.1)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert rec.proposed_value is None
    assert any("no waiting demand" in r for r in rec.reasons)


def test_latency_guardrail_failure_proposes_down():
    config = _config()
    config.guardrails.latency.max_value = 0.5
    supervisor = Supervisor(config, FakeControl(current=4))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(4, 1, kv=0.5, ttft=0.9),
             _snap(4, 1, kv=0.5, ttft=0.95),
             _snap(4, 1, kv=0.5, ttft=1.0)),
        now=now,
    )
    assert rec.decision == "PROPOSE_DOWN"
    assert rec.proposed_value == 3
    assert any("guardrail breach" in r and "ttft_p95_s" in r for r in rec.reasons)
    latency = next(g for g in rec.guardrails if g["name"] == "latency")
    assert latency["passed"] is False
    assert latency["observed"] == 1.0


def test_kv_guardrail_failure_proposes_down():
    supervisor = Supervisor(_config(), FakeControl(current=4))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(4, 1, kv=0.95, ttft=0.1),
             _snap(4, 1, kv=0.98, ttft=0.1),
             _snap(4, 1, kv=0.97, ttft=0.1)),
        now=now,
    )
    assert rec.decision == "PROPOSE_DOWN"
    assert rec.proposed_value == 3
    kv = next(g for g in rec.guardrails if g["name"] == "kv_usage")
    assert kv["passed"] is False


def test_known_guardrail_breach_proposes_bounded_down_to_min():
    supervisor = Supervisor(_config(), FakeControl(current=2, lo=1, hi=8))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(2, 0, kv=0.95, ttft=0.1),
             _snap(2, 0, kv=0.96, ttft=0.1),
             _snap(1, 0, kv=0.97, ttft=0.1)),
        now=now,
    )
    assert rec.decision == "PROPOSE_DOWN"
    assert rec.proposed_value == 1  # bounded at the capability minimum


def test_guardrail_breach_at_min_bound_holds():
    supervisor = Supervisor(_config(), FakeControl(current=1, lo=1, hi=8))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(1, 0, kv=0.95, ttft=0.1),
             _snap(1, 0, kv=0.96, ttft=0.1),
             _snap(1, 0, kv=0.97, ttft=0.1)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert any("at min bound" in r for r in rec.reasons)


# ------------------------------------------------ fail-closed / guardrail config
def test_required_latency_unavailable_holds_on_otherwise_eligible_up():
    """Latency is required for scale-up (default) but its telemetry is absent:
    fail closed — HOLD, never treat unavailable as PASS."""
    supervisor = Supervisor(_config(), FakeControl(current=2))
    now = 1000.0
    # saturation + waiting demand, kv available, but no ttft samples at all
    rec = supervisor.observe(
        _win(now,
             _snap(2, 5, kv=0.3),
             _snap(2, 6, kv=0.3),
             _snap(2, 7, kv=0.3)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert any("required guardrail" in r for r in rec.reasons)
    latency = next(g for g in rec.guardrails if g["name"] == "latency")
    assert latency["available"] is False
    assert latency["passed"] is None
    assert latency["required_for_scale_up"] is True


def test_non_required_unavailable_guardrail_does_not_block_up():
    config = _config()
    config.guardrails.latency.required_for_scale_up = False
    supervisor = Supervisor(config, FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(2, 5, kv=0.3),
             _snap(2, 6, kv=0.3),
             _snap(2, 7, kv=0.3)),
        now=now,
    )
    assert rec.decision == "PROPOSE_UP"
    latency = next(g for g in rec.guardrails if g["name"] == "latency")
    assert latency["available"] is False  # unavailable recorded
    assert latency["required_for_scale_up"] is False


def test_disabled_guardrail_is_not_evaluated():
    """A deliberately disabled guardrail neither blocks scale-up nor triggers a
    down-proposal, even when its metric would breach if enabled."""
    config = _config()
    config.guardrails.latency.enabled = False
    assert config.guardrails.latency.metric not in config.effective_metrics()
    supervisor = Supervisor(config, FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(2, 5, kv=0.3, ttft=99.0),
             _snap(2, 6, kv=0.3, ttft=99.0),
             _snap(2, 7, kv=0.3, ttft=99.0)),
        now=now,
    )
    assert rec.decision == "PROPOSE_UP"
    latency = next(g for g in rec.guardrails if g["name"] == "latency")
    assert latency["enabled"] is False
    assert latency["passed"] is None


# ---------------------------------------------------- fresh-vs-stale semantics
def test_stale_samples_cannot_satisfy_proposal_evidence():
    """All demand samples stale -> zero fresh samples -> insufficient telemetry."""
    quality = {"requests_running": "stale", "requests_waiting": "stale"}
    supervisor = Supervisor(_config(), FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(2, 5, kv=0.3, ttft=0.1, quality=quality),
             _snap(2, 6, kv=0.3, ttft=0.1, quality=quality),
             _snap(2, 7, kv=0.3, ttft=0.1, quality=quality)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert any("fresh samples=0" in r for r in rec.reasons)
    running = rec.window["metrics"]["requests_running"]
    assert running["samples"] == 0
    assert running["stale"] == 3  # recorded diagnostically, unused


def test_stale_latency_does_not_satisfy_required_guardrail():
    """Fresh demand is present, but the required latency guardrail has only
    stale samples -> unavailable -> fail closed HOLD."""
    quality = {"ttft_p95_s": "stale"}
    supervisor = Supervisor(_config(), FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(2, 5, kv=0.3, ttft=0.1, quality=quality),
             _snap(2, 6, kv=0.3, ttft=0.1, quality=quality),
             _snap(2, 7, kv=0.3, ttft=0.1, quality=quality)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert any("required guardrail" in r for r in rec.reasons)
    latency = next(g for g in rec.guardrails if g["name"] == "latency")
    assert latency["available"] is False
    assert latency["passed"] is None


def test_signal_window_uses_fresh_only_and_records_stale():
    now = 1000.0
    samples = [
        (now - 2.0, _snap(2, 5, kv=0.4)),
        (now - 1.0, _snap(2, 7, kv=0.6)),
        (now, _snap(4, 9, kv=0.5, quality={"requests_running": "stale"})),
    ]
    window = build_signal_window(
        samples, now=now, window_s=10.0,
        metrics=("requests_running", "requests_waiting", "kv_usage"),
    )
    running = window.get("requests_running")
    # fresh-only decision stats: the stale [4] is excluded
    assert running.samples == 2
    assert running.stale == 1
    assert running.median == 2 and running.minimum == 2 and running.maximum == 2
    assert running.stale_maximum == 4
    assert window.get("requests_waiting").samples == 3  # all fresh


# ----------------------------------------------------------- safety / boundary
def test_cooldown_blocks_then_allows_after_elapsed():
    supervisor = Supervisor(_config(cooldown_s=60.0), FakeControl(current=2))
    now = 1000.0

    first = supervisor.observe(_sat(now), now=now)
    assert first.decision == "PROPOSE_UP"

    during = supervisor.observe(_sat(now + 10.0), now=now + 10.0)
    assert during.decision == "HOLD"
    assert during.proposed_value is None
    assert during.safety["cooldown_active"] is True
    assert any("cooldown active" in r for r in during.reasons)

    after = supervisor.observe(_sat(now + 61.0), now=now + 61.0)
    assert after.decision == "PROPOSE_UP"
    assert after.safety["cooldown_active"] is False


def test_bounds_hold_at_max_on_saturation():
    supervisor = Supervisor(_config(), FakeControl(current=8, lo=1, hi=8))
    now = 1000.0
    rec = supervisor.observe(_win(now, _snap(8, 5), _snap(8, 6), _snap(8, 7)), now=now)
    assert rec.decision == "HOLD"
    assert rec.proposed_value is None
    assert any("max bound" in r for r in rec.reasons)


def test_bounds_min_no_demand_holds():
    supervisor = Supervisor(_config(), FakeControl(current=1, lo=1, hi=8))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(0, 0, kv=0.1, ttft=0.1),
             _snap(0, 0, kv=0.1, ttft=0.1),
             _snap(0, 0, kv=0.1, ttft=0.1)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert any("no waiting demand" in r for r in rec.reasons)


def test_insufficient_telemetry_holds():
    supervisor = Supervisor(_config(min_samples=3), FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(_win(now, _snap(2, 5)), now=now)
    assert rec.decision == "HOLD"
    assert rec.proposed_value is None
    assert any("insufficient telemetry" in r for r in rec.reasons)
    assert rec.evidence["requests_waiting_samples"] == 1


def test_identical_inputs_produce_identical_decisions():
    cfg = _config()
    now = 1000.0
    samples = _sat(now)
    one = Supervisor(cfg, FakeControl(current=2)).observe(samples, now=now)
    two = Supervisor(cfg, FakeControl(current=2)).observe(samples, now=now)
    a, b = one.to_dict(), two.to_dict()
    a.pop("decided_at")
    b.pop("decided_at")
    assert a == b

    bounds = CapabilityBounds(1, 8)
    window = build_signal_window(samples, now=now, window_s=10.0,
                                 metrics=cfg.effective_metrics())
    policy = ConcurrencyPolicy(cfg)
    assert policy.evaluate(window, current_value=2, bounds=bounds) == \
        policy.evaluate(window, current_value=2, bounds=bounds)


def test_observe_only_never_mutates():
    control = FakeControl(current=2)
    supervisor = Supervisor(_config(), control)
    now = 1000.0
    for i in range(5):
        t = now + i * 100.0
        rec = supervisor.observe(_sat(t), now=t)
        assert rec.mutation_applied is False
    assert control.mutations == 0


def test_non_converged_state_holds():
    control = FakeControl(current=2, desired={INTERNAL_KEY: 4})
    supervisor = Supervisor(_config(), control)
    now = 1000.0
    rec = supervisor.observe(_sat(now), now=now)
    assert rec.decision == "HOLD"
    assert rec.safety["converged"] is False
    assert any("not converged" in r for r in rec.reasons)


def test_concurrent_revision_change_holds():
    control = FakeControl(current=2, revision=1)
    supervisor = Supervisor(_config(), control)
    now = 1000.0
    first = supervisor.observe(_sat(now), now=now)
    assert first.decision == "PROPOSE_UP"

    control.revision = 2  # a concurrent DesiredState change
    second = supervisor.observe(_sat(now + 100.0), now=now + 100.0)
    assert second.decision == "HOLD"
    assert second.safety["concurrent_change_detected"] is True
    assert any("desired_revision changed" in r for r in second.reasons)


def test_missing_capability_holds_without_bounds():
    supervisor = Supervisor(_config(), FakeControl(capability=False))
    now = 1000.0
    rec = supervisor.observe(_sat(now), now=now)
    assert rec.decision == "HOLD"
    assert rec.capability_bounds is None
    assert rec.safety["capability_found"] is False
    assert any("bounds unknown" in r for r in rec.reasons)


def test_within_operating_band_holds():
    supervisor = Supervisor(_config(), FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(
        _win(now,
             _snap(2, 0, kv=0.1, ttft=0.1),
             _snap(2, 0, kv=0.1, ttft=0.1),
             _snap(2, 0, kv=0.1, ttft=0.1)),
        now=now,
    )
    assert rec.decision == "HOLD"
    assert any("within operating band" in r for r in rec.reasons)


# -------------------------------------------------------- factual signal window
def test_signal_window_aggregates_are_numeric_and_factual():
    now = 1000.0
    samples = _win(now, _snap(2, 5, kv=0.4), _snap(2, 7, kv=0.6), _snap(4, 9, kv=0.5))
    window = build_signal_window(
        samples, now=now, window_s=10.0,
        metrics=("requests_running", "requests_waiting", "kv_usage"),
    )
    assert window.snapshots == 3
    running = window.get("requests_running")
    assert (running.samples, running.median, running.minimum, running.maximum) == (3, 2, 2, 4)
    assert window.get("requests_waiting").median == 7
    assert window.get("kv_usage").maximum == 0.6
    assert window.get("kv_usage").last == 0.5


def test_samples_outside_horizon_are_excluded():
    now = 1000.0
    samples = [
        (now - 100.0, _snap(9, 9, kv=0.99)),  # outside window_s=10
        (now - 1.0, _snap(2, 5, kv=0.4)),
        (now, _snap(2, 6, kv=0.5)),
    ]
    window = build_signal_window(
        samples, now=now, window_s=10.0, metrics=("requests_running", "kv_usage"))
    assert window.snapshots == 2
    assert window.get("requests_running").maximum == 2
    assert window.get("kv_usage").maximum == 0.5


def test_metrics_without_values_are_absent_not_zero():
    now = 1000.0
    window = build_signal_window(
        [(now, _snap(2, 1))], now=now, window_s=10.0, metrics=("kv_usage",))
    kv = window.get("kv_usage")
    assert kv.samples == 0
    assert kv.median is None and kv.maximum is None and kv.last is None


def test_disabled_metric_is_excluded():
    now = 1000.0
    snap = _snap(2, 1, quality={"ttft_p95_s": "disabled"})
    window = build_signal_window(
        [(now, snap)], now=now, window_s=10.0, metrics=("ttft_p95_s",))
    assert window.get("ttft_p95_s").samples == 0


def test_decision_record_is_json_serializable():
    supervisor = Supervisor(_config(), FakeControl(current=2))
    now = 1000.0
    rec = supervisor.observe(_sat(now), now=now)
    payload = rec.to_dict()
    json.dumps(payload)  # must not raise
    assert payload["mode"] == "OBSERVE_ONLY"
    assert payload["intent"] == INTENT
    assert payload["window"]["window_s"] == 10.0
