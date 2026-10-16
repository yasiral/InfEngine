"""Control-plane adapter #3 tests: SglangAdapter + Reconciler over a fake transport.

M4B-B.1 correctness hardening:

  * LIVE readback of runtime_max_prefill_tokens (no shadow state): get() /
    observed_state() read the actual scheduler value from the /server_info
    mirror; divergence from an out-of-band mutation is detected and reconciled.
  * ENGINE_CACHE_FLUSH quiescence = REQUIRE_IDLE: a busy engine -> BLOCKED with
    NO pause/retract side effects; an idle engine -> VERIFIED.
  * Resume failure fails the DRAIN lifecycle (overall FAILED, both facts
    preserved) even when the primary operation VERIFIED.
  * Unsupported operations (prefix_cache_reset) are REJECTED at planning and
    never enter the DRAIN lifecycle (zero pause/flush/resume).
"""

import pytest

from infengine.control import (
    QuiescenceStrategy,
    Reconciler,
    ReconcileStatus,
    StepStatus,
    TransitionClass,
)
from infengine.sglang import SglangAdapter, SglangRuntimeClient, SglangRuntimeError, validation


class FakeSglangRuntimeClient(SglangRuntimeClient):
    """In-memory transport fake mirroring the SGLang v0.5.18 HTTP surface.

    ``_server_info`` is the /server_info mirror: top-level provisioned fields
    plus ``internal_states[0]`` carrying the live ``pp_max_micro_batch_size``
    and ``runtime_max_prefill_tokens``. ``get_policy`` refreshes from it (live
    read); ``set_policy`` applies to it; out-of-band divergence is simulated by
    mutating ``_server_info["internal_states"][0]`` directly.
    """

    def __init__(self, *, fail_set=False, diverge=False, fail_drain=False,
                 fail_resume=False, fail_flush=False, fail_load=False,
                 flush_ok=True, load=None, identity=None,
                 provisioned_seqs=8, provisioned_tokens=16384):
        self.provisioned_seqs = provisioned_seqs
        self.provisioned_tokens = provisioned_tokens
        self.fail_set = fail_set
        self.diverge = diverge
        self.fail_drain = fail_drain
        self.fail_resume = fail_resume
        self.fail_flush = fail_flush
        self.fail_load = fail_load
        self.flush_ok = flush_ok
        self._server_info = {
            "max_running_requests": provisioned_seqs,
            "max_prefill_tokens": provisioned_tokens,
            "pp_size": 1,
            "dp_size": 1,
            "internal_states": [{
                "pp_max_micro_batch_size": provisioned_seqs,
                "runtime_max_prefill_tokens": provisioned_tokens,
            }],
        }
        self.load = dict(load or {"num_reqs": 0, "num_waiting_reqs": 0,
                                  "num_tokens": 0, "num_pending_tokens": 0})
        self.set_calls = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.flush_calls = 0
        self._last_flush_ok = False
        self._identity = dict(identity or {
            "engine_instance_id": "sglang-fake-0001",
            "base_url": "http://localhost:30000",
            "control_pid": 9001,
            "runtime_pids": {"scheduler": 9002},
        })

    # ---- SglangRuntimeClient interface ----------------------------------
    def provisioned(self):
        return {
            "max_running_requests": self._server_info["max_running_requests"],
            "max_prefill_tokens": self._server_info["max_prefill_tokens"],
            "pp_size": self._server_info["pp_size"],
            "dp_size": self._server_info["dp_size"],
        }

    def refresh(self):
        sched = self._server_info["internal_states"][0]
        self._policy = {
            "runtime_max_running_seqs": int(sched["pp_max_micro_batch_size"]),
            "runtime_max_prefill_tokens": int(sched.get(
                "runtime_max_prefill_tokens", self._server_info["max_prefill_tokens"])),
        }

    def get_policy(self, name):
        self.refresh()
        return self._policy.get(name)

    def set_policy(self, name, value):
        self.set_calls.append({name: value})
        if self.fail_set:
            raise SglangRuntimeError("transport failure")
        if self.diverge:
            return True  # transport accepted but runtime diverges (no record)
        sched = self._server_info["internal_states"][0]
        if name == "runtime_max_running_seqs":
            validation.validate_concurrency_upper_bound(value, self.provisioned_seqs)
            sched["pp_max_micro_batch_size"] = int(value)
            return True
        if name == "runtime_max_prefill_tokens":
            validation.validate_prefill_admission_budget(value, self.provisioned_tokens)
            sched["runtime_max_prefill_tokens"] = int(value)
            return True
        return False

    def get_load(self):
        if self.fail_load:
            raise SglangRuntimeError("load transport failure")
        return dict(self.load)

    def pause_generation(self, mode="retract"):
        self.pause_calls += 1
        if self.fail_drain:
            raise SglangRuntimeError("pause failed")

    def continue_generation(self, torch_empty_cache=False):
        self.resume_calls += 1
        if self.fail_resume:
            raise SglangRuntimeError("resume failed")

    def flush_cache(self):
        self.flush_calls += 1
        if self.fail_flush:
            raise SglangRuntimeError("flush failed")
        ok = self.flush_ok
        self._last_flush_ok = ok
        return ok

    def flush_cache_ok(self):
        return self._last_flush_ok

    def identity(self):
        return dict(self._identity)


def make_reconciler(**kwargs):
    client = FakeSglangRuntimeClient(**kwargs)
    adapter = SglangAdapter(client)
    return client, adapter, Reconciler(adapter)


# ------------------------------------------------------------------ observation
def test_observe_returns_runtime_values():
    client, adapter, _ = make_reconciler()
    assert adapter.get("runtime_max_running_seqs") == 8
    assert adapter.get("runtime_max_prefill_tokens") == 16384
    assert adapter.get("engine_cache_flush") == 0  # operation: idle, no synthetic state


def test_capabilities_and_runtime_derived_constraints():
    client, adapter, _ = make_reconciler()
    assert set(adapter.capabilities()) == {
        "runtime_max_running_seqs", "runtime_max_prefill_tokens",
        "engine_cache_flush", "prefix_cache_reset",
    }
    con = adapter.capability("runtime_max_running_seqs")
    assert con.transition is TransitionClass.HOT
    assert con.min_value == 1
    assert con.max_value == 8  # provisioned max_running_requests
    assert con.semantic == "CONCURRENCY_UPPER_BOUND"
    assert con.scope == "PER_DP_RANK"
    pref = adapter.capability("runtime_max_prefill_tokens")
    assert pref.max_value == 16384
    assert pref.semantic == "AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED"
    assert pref.scope == "LOCAL_SCHEDULER"


def test_engine_cache_flush_operation_metadata():
    _, adapter, _ = make_reconciler()
    op = adapter.capability("engine_cache_flush")
    assert op.transition is TransitionClass.DRAIN
    assert op.operation is True
    assert op.requires_quiescence is True
    # REQUIRE_IDLE, NOT RETRACT_INFLIGHT (retract does not make a busy engine
    # flushable).
    assert op.quiescence_requirement == "REQUIRE_IDLE"
    assert op.quiescence_strategy is None
    assert op.semantic == "ENGINE_CACHE_FLUSH"
    # PREFIX_CACHE_RESET advertised but unsupported
    pr = adapter.capability("prefix_cache_reset")
    assert pr.supported is False
    assert pr.semantic == "PREFIX_CACHE_RESET"


# ------------------------------------------------------ policies (HOT) reconcile
def test_hot_policies_verified_and_persisted():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_running_seqs": 3, "runtime_max_prefill_tokens": 512})
    by = {s.param: s for s in r.steps}
    assert by["runtime_max_running_seqs"].status is StepStatus.VERIFIED
    assert by["runtime_max_prefill_tokens"].status is StepStatus.VERIFIED
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get("runtime_max_running_seqs") == 3
    assert adapter.get("runtime_max_prefill_tokens") == 512
    # persistent policies ARE stored in DesiredState
    assert reconciler.desired()["runtime_max_running_seqs"] == 3
    assert reconciler.desired()["runtime_max_prefill_tokens"] == 512


def test_hot_lower_then_raise_within_ceiling():
    client, adapter, reconciler = make_reconciler()
    reconciler.reconcile({"runtime_max_running_seqs": 1})
    assert adapter.get("runtime_max_running_seqs") == 1
    reconciler.reconcile({"runtime_max_running_seqs": 6})
    assert adapter.get("runtime_max_running_seqs") == 6
    assert reconciler.last.status is ReconcileStatus.CONVERGED


def test_concurrency_above_provisioned_rejected_before_apply():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_running_seqs": 20})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert client.set_calls == []
    assert adapter.get("runtime_max_running_seqs") == 8  # unchanged


def test_prefill_above_provisioned_rejected_before_apply():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_prefill_tokens": 20000})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert client.set_calls == []
    assert adapter.get("runtime_max_prefill_tokens") == 16384  # boot ceiling unchanged


def test_below_minimum_rejected():
    client, _, reconciler = make_reconciler()
    for bad in (0, -1):
        r = reconciler.reconcile({"runtime_max_running_seqs": bad})
        assert r.steps[0].status is StepStatus.REJECTED
        assert client.set_calls == []


# ------------------------------------------------ strict type validation (M4B-B)
@pytest.mark.parametrize("bad", ["abc", 2.5, None, True, False])
def test_concurrency_invalid_type_rejected(bad):
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_running_seqs": bad})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert client.set_calls == []
    assert adapter.get("runtime_max_running_seqs") == 8


@pytest.mark.parametrize("bad", ["abc", 2.5, None, True, False])
def test_prefill_invalid_type_rejected(bad):
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_prefill_tokens": bad})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert client.set_calls == []
    assert adapter.get("runtime_max_prefill_tokens") == 16384


def test_float_not_silently_truncated():
    client, _, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_running_seqs": 2.5})
    assert r.steps[0].status is StepStatus.REJECTED
    assert "expected int" in (r.steps[0].reason or "")
    assert client.set_calls == []


# ------------------------------------------------- Fix #1: live readback + divergence
def test_prefill_readback_is_live_not_shadowed():
    client, adapter, _ = make_reconciler()
    reconciler = Reconciler(adapter)
    reconciler.reconcile({"runtime_max_prefill_tokens": 512})
    assert adapter.get("runtime_max_prefill_tokens") == 512
    # out-of-band direct mutation to another legal value (simulates scheduler
    # restart / external mutation / divergence)
    client._server_info["internal_states"][0]["runtime_max_prefill_tokens"] = 256
    # live readback MUST observe the divergence
    assert adapter.get("runtime_max_prefill_tokens") == 256


def test_divergence_detected_and_reconverged():
    client, adapter, reconciler = make_reconciler()
    reconciler.reconcile({"runtime_max_prefill_tokens": 512})
    assert adapter.get("runtime_max_prefill_tokens") == 512
    # out-of-band mutation to another legal value
    client._server_info["internal_states"][0]["runtime_max_prefill_tokens"] = 256
    assert adapter.get("runtime_max_prefill_tokens") == 256
    # reconciliation detects divergence (EXECUTE, not NOOP) and restores desired
    r = reconciler.reconcile({"runtime_max_prefill_tokens": 512})
    assert r.steps[0].status is StepStatus.VERIFIED
    assert r.status is ReconcileStatus.CONVERGED
    assert r.steps[0].observed_before == 256
    assert adapter.get("runtime_max_prefill_tokens") == 512


def test_engine_side_rejection_maps_to_failed():
    # runtime rejects an above-ceiling value at set time (mimics [false])
    client, adapter, reconciler = make_reconciler()
    client.set_policy = lambda name, value: False
    r = reconciler.reconcile({"runtime_max_running_seqs": 4})
    assert r.steps[0].status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert adapter.get("runtime_max_running_seqs") == 8  # unchanged (live)


def test_transport_failure_failed():
    client, adapter, reconciler = make_reconciler(fail_set=True)
    r = reconciler.reconcile({"runtime_max_prefill_tokens": 128})
    assert r.steps[0].status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert adapter.get("runtime_max_prefill_tokens") == 16384


# ------------------------------------------------------ operations (DRAIN)
def test_engine_cache_flush_operation_verified_when_idle():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"engine_cache_flush": 1})
    step = r.steps[0]
    assert step.transition is TransitionClass.DRAIN
    assert step.status is StepStatus.VERIFIED
    assert r.status is ReconcileStatus.CONVERGED
    # REQUIRE_IDLE: drain does NOT pause (engine idle); flush + resume run
    assert client.pause_calls == 0
    assert client.flush_calls == 1
    assert client.resume_calls == 1
    # operation is NOT persisted into DesiredState
    assert "engine_cache_flush" not in reconciler.desired()
    assert adapter.get("engine_cache_flush") == 0


def test_busy_engine_cache_flush_blocked_without_retract_side_effects():
    # Fix #2: a busy engine -> BLOCKED with NO pause/retract -> waiting -> resume
    client, adapter, reconciler = make_reconciler(
        load={"num_reqs": 2, "num_waiting_reqs": 1, "num_tokens": 0, "num_pending_tokens": 0}
    )
    r = reconciler.reconcile({"engine_cache_flush": 1})
    assert r.steps[0].status is StepStatus.BLOCKED_BUSY
    assert r.status is ReconcileStatus.BLOCKED
    assert client.pause_calls == 0
    assert client.flush_calls == 0
    assert client.resume_calls == 0
    assert "engine_cache_flush" not in reconciler.desired()


def test_operation_not_persisted_and_runnable_again():
    client, adapter, reconciler = make_reconciler()
    reconciler.reconcile({"engine_cache_flush": 1})
    assert "engine_cache_flush" not in reconciler.desired()
    r2 = reconciler.reconcile({"engine_cache_flush": 1})
    assert r2.steps[0].status is StepStatus.VERIFIED
    assert client.flush_calls == 2


# Fix #4: unsupported operations rejected before DRAIN
def test_prefix_cache_reset_rejected_at_planning_no_transport():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"prefix_cache_reset": 1})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    # unsupported op NEVER enters the DRAIN lifecycle
    assert client.pause_calls == 0
    assert client.flush_calls == 0
    assert client.resume_calls == 0
    assert client.set_calls == []


def test_flush_operation_verify_fails_if_not_confirmed():
    client, adapter, reconciler = make_reconciler()

    def noop_flush():
        client.flush_calls += 1
        return True  # HTTP 200 but confirmation observable stays False

    client.flush_cache = noop_flush
    client._last_flush_ok = False
    r = reconciler.reconcile({"engine_cache_flush": 1})
    assert r.steps[0].status is StepStatus.FAILED  # verify() did not confirm
    assert client.resume_calls == 1  # resume still guaranteed


def test_load_failure_blocked():
    client, adapter, reconciler = make_reconciler(fail_load=True)
    r = reconciler.reconcile({"engine_cache_flush": 1})
    assert r.steps[0].status is StepStatus.BLOCKED_BUSY
    assert r.status is ReconcileStatus.BLOCKED
    assert client.pause_calls == 0 and client.flush_calls == 0 and client.resume_calls == 0


def test_post_drain_flush_failure_still_resumes():
    client, _, reconciler = make_reconciler(fail_flush=True)
    r = reconciler.reconcile({"engine_cache_flush": 1})
    assert r.steps[0].status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert client.pause_calls == 0
    assert client.resume_calls == 1  # resume guaranteed in the finally


# ------------------------------------------- Fix #3: resume failure fails lifecycle
def test_operation_succeeds_resume_fails_overall_failed():
    client, adapter, reconciler = make_reconciler(fail_resume=True, flush_ok=True)
    r = reconciler.reconcile({"engine_cache_flush": 1})
    step = r.steps[0]
    # primary operation was verified but resume (cleanup) failed -> overall FAILED
    assert step.status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert "resume failed" in (step.reason or "")
    assert "primary op outcome: VERIFIED" in (step.reason or "")
    assert "may remain paused" in (step.reason or "")
    assert client.resume_calls == 1
    assert adapter._resume_error is not None


def test_operation_fails_resume_fails():
    client, adapter, reconciler = make_reconciler(fail_resume=True, flush_ok=False)
    r = reconciler.reconcile({"engine_cache_flush": 1})
    step = r.steps[0]
    assert step.status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert "resume failed" in (step.reason or "")
    assert "primary op outcome: FAILED" in (step.reason or "")
    assert client.resume_calls == 1


def test_verification_fails_resume_fails():
    client, adapter, reconciler = make_reconciler(fail_resume=True)

    def noop_flush():
        client.flush_calls += 1
        return True

    client.flush_cache = noop_flush
    client._last_flush_ok = False  # verify() will not confirm
    r = reconciler.reconcile({"engine_cache_flush": 1})
    step = r.steps[0]
    assert step.status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert "resume failed" in (step.reason or "")
    assert client.resume_calls == 1


# ------------------------------------------------------ identity
def test_identity_preserved_across_changes():
    client, adapter, reconciler = make_reconciler()
    before = adapter.identity()
    reconciler.reconcile({"runtime_max_running_seqs": 2})
    reconciler.reconcile({"engine_cache_flush": 1})
    assert adapter.identity() == before


# ------------------------------------------------------ validation (hook semantics)
def test_validate_concurrency_upper_bound():
    assert validation.validate_concurrency_upper_bound(2, 8) == 2
    with pytest.raises(validation.SglangControlError, match="exceeds"):
        validation.validate_concurrency_upper_bound(9, 8)
    with pytest.raises(validation.SglangControlError, match="minimum"):
        validation.validate_concurrency_upper_bound(0, 8)
    for bad in ("x", 2.5, None, True):
        with pytest.raises(validation.SglangControlError, match="expected int"):
            validation.validate_concurrency_upper_bound(bad, 8)


def test_validate_prefill_admission_budget():
    assert validation.validate_prefill_admission_budget(512, 16384) == 512
    with pytest.raises(validation.SglangControlError, match="exceeds"):
        validation.validate_prefill_admission_budget(20000, 16384)
    with pytest.raises(validation.SglangControlError, match="minimum"):
        validation.validate_prefill_admission_budget(0, 16384)
    for bad in ("x", 2.5, None, True):
        with pytest.raises(validation.SglangControlError, match="expected int"):
            validation.validate_prefill_admission_budget(bad, 16384)