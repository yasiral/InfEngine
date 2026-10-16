"""Control-plane adapter #2 tests: VllmAdapter + Reconciler over a fake transport.

M3D: persistent policies (HOT) vs one-shot operations (DRAIN); runtime-derived
capability max constraints; operation verification via engine observables.
"""

import pytest

from infengine.control import (
    Reconciler,
    ReconcileStatus,
    StepStatus,
    TransitionClass,
)
from infengine.vllm import VllmAdapter, VllmRuntimeClient, VllmRuntimeError, validation


class FakeVllmRuntimeClient(VllmRuntimeClient):
    """In-memory transport fake mimicking the vLLM EngineCore runtime-policy hook."""

    def __init__(self, config=None, *, fail_on_set=False, diverge=False,
                 provisioned_seqs=4, provisioned_tokens=512, fail_drain=False,
                 fail_reset=False, prefix_entries=3, identity=None):
        default = {
            "num_gpu_blocks": 512,
            "max_num_seqs": provisioned_seqs,
            "max_num_batched_tokens": provisioned_tokens,
            "max_num_scheduled_tokens": provisioned_tokens,
            "long_prefill_token_threshold": 0,
            "runtime_max_running_seqs": provisioned_seqs,
            "runtime_max_scheduled_tokens": provisioned_tokens,
            "runtime_kv_watermark": 0.0,
            "runtime_reserve_full_isl": True,
        }
        cfg = dict(default)
        cfg.update(config or {})
        self.config = cfg
        self.provisioned_seqs = provisioned_seqs
        self.provisioned_tokens = provisioned_tokens
        self.fail_on_set = fail_on_set
        self.diverge = diverge
        self.fail_drain = fail_drain
        self.fail_reset = fail_reset
        self.prefix_entries = prefix_entries
        self.set_calls = []
        self.pause_calls = 0
        self.resume_calls = 0
        self.reset_calls = 0
        self._identity = dict(identity or {
            "engine_core_pid": 4242, "worker_pids": [4243],
            "engine_instance_id": "vllm-v1-fake-0001", "started_at": 1234567890.0,
        })

    def get_runtime_scheduler_config(self):
        return {
            "policies": {
                "long_prefill_token_threshold": self.config["long_prefill_token_threshold"],
                "runtime_max_running_seqs": self.config["runtime_max_running_seqs"],
                "runtime_max_scheduled_tokens": self.config["runtime_max_scheduled_tokens"],
                "runtime_kv_watermark": self.config["runtime_kv_watermark"],
                "runtime_reserve_full_isl": self.config["runtime_reserve_full_isl"],
            },
            "provisioned": {
                "max_num_seqs": self.config["max_num_seqs"],
                "max_num_batched_tokens": self.config["max_num_batched_tokens"],
                "max_num_scheduled_tokens": self.config["max_num_scheduled_tokens"],
                "num_gpu_blocks": self.config["num_gpu_blocks"],
            },
        }

    def set_runtime_scheduler_config(self, overrides):
        self.set_calls.append(dict(overrides))
        if self.fail_on_set:
            raise VllmRuntimeError("transport failure")
        if self.diverge:
            return self.get_runtime_scheduler_config()
        for field, value in overrides.items():
            if field == "runtime_max_running_seqs":
                validation.validate_runtime_max_running_seqs(value, self.provisioned_seqs)
                self.config["runtime_max_running_seqs"] = value
            elif field == "runtime_max_scheduled_tokens":
                validation.validate_runtime_max_scheduled_tokens(value, self.provisioned_tokens)
                self.config["runtime_max_scheduled_tokens"] = value
            elif field == "runtime_kv_watermark":
                validation.validate_runtime_kv_watermark(value)
                self.config["runtime_kv_watermark"] = value
                self.config["watermark_blocks"] = int(value * self.config["num_gpu_blocks"])
            elif field == "runtime_reserve_full_isl":
                validation.validate_runtime_reserve_full_isl(value)
                self.config["runtime_reserve_full_isl"] = value
            else:
                validation.apply_overrides(self.config, {field: value})
        return self.get_runtime_scheduler_config()

    def pause_generation(self, mode="wait", clear_cache=False):
        self.pause_calls += 1
        if self.fail_drain:
            raise VllmRuntimeError("pause failed")

    def resume_generation(self):
        self.resume_calls += 1

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        if self.fail_reset:
            raise VllmRuntimeError("reset failed")
        self.reset_calls += 1
        self.prefix_entries = 0
        return True

    def get_runtime_cache_stats(self):
        return {"prefix_cache_entries": self.prefix_entries,
                "num_gpu_blocks": self.config["num_gpu_blocks"],
                "watermark_blocks": int(self.config["runtime_kv_watermark"] * self.config["num_gpu_blocks"])}

    def identity(self):
        return dict(self._identity)


def make_reconciler(config=None, **kwargs):
    client = FakeVllmRuntimeClient(config=config, **kwargs)
    adapter = VllmAdapter(client)
    return client, adapter, Reconciler(adapter)


# ------------------------------------------------------------------ observation
def test_observe_returns_runtime_values():
    _, adapter, _ = make_reconciler()
    assert adapter.get("long_prefill_token_threshold") == 0
    assert adapter.get("runtime_max_running_seqs") == 4
    assert adapter.get("runtime_max_scheduled_tokens") == 512
    assert adapter.get("reset_prefix_cache") == 0  # operation: idle, no synthetic state


def test_capabilities_and_runtime_derived_constraints():
    _, adapter, _ = make_reconciler()
    assert set(adapter.capabilities()) == {
        "long_prefill_token_threshold", "runtime_max_running_seqs",
        "runtime_max_scheduled_tokens", "runtime_kv_watermark",
        "runtime_reserve_full_isl", "reset_prefix_cache",
    }
    assert adapter.capability("runtime_max_running_seqs").transition is TransitionClass.HOT
    assert adapter.capability("runtime_max_running_seqs").min_value == 1
    assert adapter.capability("runtime_max_running_seqs").max_value == 4  # provisioned
    assert adapter.capability("runtime_max_scheduled_tokens").max_value == 512
    op = adapter.capability("reset_prefix_cache")
    assert op.transition is TransitionClass.DRAIN
    assert op.operation is True


# ------------------------------------------------------ policies (HOT) reconcile
def test_hot_policy_verified_and_persisted():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_running_seqs": 2, "runtime_max_scheduled_tokens": 256})
    by = {s.param: s for s in r.steps}
    assert by["runtime_max_running_seqs"].status is StepStatus.VERIFIED
    assert by["runtime_max_scheduled_tokens"].status is StepStatus.VERIFIED
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get("runtime_max_running_seqs") == 2
    assert adapter.get("runtime_max_scheduled_tokens") == 256
    # persistent policies ARE stored in DesiredState
    assert reconciler.desired()["runtime_max_running_seqs"] == 2
    assert reconciler.desired()["runtime_max_scheduled_tokens"] == 256


def test_noop_when_policy_already_converged():
    client, _, reconciler = make_reconciler(config={"runtime_max_running_seqs": 2})
    r = reconciler.reconcile({"runtime_max_running_seqs": 2})
    assert r.steps[0].status is StepStatus.NOOP
    assert client.set_calls == []


def test_policy_above_runtime_derived_max_rejected_before_apply():
    client, adapter, reconciler = make_reconciler(provisioned_seqs=4)
    r = reconciler.reconcile({"runtime_max_running_seqs": 8})
    # runtime-derived max (provisioned) is reported by the adapter -> planning
    # rejects BEFORE calling the engine (no set call)
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert client.set_calls == []
    assert adapter.get("runtime_max_running_seqs") == 4


def test_scheduled_tokens_above_provisioned_rejected_before_apply():
    client, adapter, reconciler = make_reconciler(provisioned_tokens=512)
    r = reconciler.reconcile({"runtime_max_scheduled_tokens": 1024})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert client.set_calls == []
    assert adapter.get("runtime_max_scheduled_tokens") == 512


def test_below_minimum_rejected():
    client, _, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_max_running_seqs": 0})
    assert r.steps[0].status is StepStatus.REJECTED
    assert client.set_calls == []


def test_transport_failure_failed():
    client, adapter, reconciler = make_reconciler(fail_on_set=True)
    r = reconciler.reconcile({"runtime_max_scheduled_tokens": 128})
    assert r.steps[0].status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert adapter.get("runtime_max_scheduled_tokens") == 512


def test_setter_succeeds_but_getter_diverges_failed():
    client, adapter, reconciler = make_reconciler(diverge=True)
    r = reconciler.reconcile({"runtime_max_running_seqs": 2})
    assert r.steps[0].status is StepStatus.FAILED
    assert "verification mismatch" in (r.steps[0].reason or "")
    assert r.status is ReconcileStatus.FAILED


# ------------------------------------------------------ operations (DRAIN)
def test_reset_prefix_cache_operation_verified_via_observable():
    client, adapter, reconciler = make_reconciler(prefix_entries=3)
    r = reconciler.reconcile({"reset_prefix_cache": 1})
    step = r.steps[0]
    assert step.transition is TransitionClass.DRAIN
    assert step.status is StepStatus.VERIFIED
    assert r.status is ReconcileStatus.CONVERGED
    assert client.pause_calls == 1 and client.reset_calls == 1 and client.resume_calls == 1
    # engine observable confirms the cache was actually cleared
    assert client.get_runtime_cache_stats()["prefix_cache_entries"] == 0
    # operation is NOT persisted into DesiredState
    assert "reset_prefix_cache" not in reconciler.desired()
    assert adapter.get("reset_prefix_cache") == 0


def test_operation_not_persisted_and_runnable_again():
    client, adapter, reconciler = make_reconciler()
    reconciler.reconcile({"reset_prefix_cache": 1})
    assert "reset_prefix_cache" not in reconciler.desired()
    # no synthetic monotonic counter: a second request runs the op again
    client.prefix_entries = 5
    r2 = reconciler.reconcile({"reset_prefix_cache": 1})
    assert r2.steps[0].status is StepStatus.VERIFIED
    assert client.reset_calls == 2


def test_operation_verify_fails_if_cache_not_cleared():
    client, adapter, reconciler = make_reconciler(prefix_entries=5)
    # simulate a reset that did not actually clear the cache
    client.reset_calls = 0

    def noop_reset(*a, **k):
        client.reset_calls += 1
        return True  # returns success but cache entries unchanged

    client.reset_prefix_cache = noop_reset
    r = reconciler.reconcile({"reset_prefix_cache": 1})
    assert r.steps[0].status is StepStatus.FAILED  # engine observable did not confirm
    assert client.resume_calls == 1  # resume still guaranteed


def test_drain_failure_blocked_no_resume():
    client, _, reconciler = make_reconciler(fail_drain=True)
    r = reconciler.reconcile({"reset_prefix_cache": 1})
    assert r.steps[0].status is StepStatus.BLOCKED_BUSY
    assert r.status is ReconcileStatus.BLOCKED
    assert client.resume_calls == 0


def test_post_drain_reset_failure_still_resumes():
    client, _, reconciler = make_reconciler(fail_reset=True)
    r = reconciler.reconcile({"reset_prefix_cache": 1})
    assert r.steps[0].status is StepStatus.FAILED
    assert r.status is ReconcileStatus.FAILED
    assert client.pause_calls == 1
    assert client.resume_calls == 1  # resume guaranteed in the finally


# ------------------------------------------------------ identity
def test_identity_preserved_across_changes():
    client, adapter, reconciler = make_reconciler()
    before = adapter.identity()
    reconciler.reconcile({"runtime_max_running_seqs": 2})
    reconciler.reconcile({"reset_prefix_cache": 1})
    assert adapter.identity() == before


# ------------------------------------------------------ validation (hook semantics)
def test_validate_runtime_scheduled_tokens():
    assert validation.validate_runtime_max_scheduled_tokens(256, 512) == 256
    with pytest.raises(validation.RuntimeSchedulerConfigError, match="exceeds"):
        validation.validate_runtime_max_scheduled_tokens(1024, 512)
    with pytest.raises(validation.RuntimeSchedulerConfigError):
        validation.validate_runtime_max_scheduled_tokens("x", 512)
    with pytest.raises(validation.RuntimeSchedulerConfigError, match="minimum"):
        validation.validate_runtime_max_scheduled_tokens(0, 512)


def test_validate_override_rejects_unknown_field():
    with pytest.raises(validation.RuntimeSchedulerConfigError, match="unknown"):
        validation.validate_override("max_num_seqs", 8)


# ------------------------------------------------ M3F: runtime_kv_watermark + reserve_full_isl
def test_runtime_kv_watermark_hot_verified():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_kv_watermark": 0.2})
    step = r.steps[0]
    assert step.status is StepStatus.VERIFIED
    assert step.transition is TransitionClass.HOT
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get("runtime_kv_watermark") == 0.2
    # physical capacity unchanged; headroom derived
    stats = client.get_runtime_cache_stats()
    assert stats["num_gpu_blocks"] == 512
    assert stats["watermark_blocks"] == int(0.2 * 512)  # 102
    assert reconciler.desired()["runtime_kv_watermark"] == 0.2


def test_runtime_kv_watermark_bounds_rejected():
    client, adapter, reconciler = make_reconciler()
    for bad in (-0.1, 1.0):
        r = reconciler.reconcile({"runtime_kv_watermark": bad})
        assert r.steps[0].status is StepStatus.REJECTED
        assert r.status is ReconcileStatus.FAILED
        assert client.set_calls == []  # planning rejects before RPC
    assert adapter.get("runtime_kv_watermark") == 0.0


def test_runtime_kv_watermark_non_numeric_rejected():
    client, _, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_kv_watermark": "abc"})
    assert r.steps[0].status is StepStatus.REJECTED
    assert client.set_calls == []


def test_runtime_reserve_full_isl_hot_verified():
    client, adapter, reconciler = make_reconciler()
    r = reconciler.reconcile({"runtime_reserve_full_isl": False})
    step = r.steps[0]
    assert step.status is StepStatus.VERIFIED
    assert step.transition is TransitionClass.HOT
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get("runtime_reserve_full_isl") is False
    assert client.set_calls == [{"runtime_reserve_full_isl": False}]


def test_runtime_reserve_full_isl_non_bool_rejected():
    client, adapter, reconciler = make_reconciler()
    for bad in (1, "true", None):
        r = reconciler.reconcile({"runtime_reserve_full_isl": bad})
        assert r.steps[0].status is StepStatus.REJECTED
        assert client.set_calls == []
    assert adapter.get("runtime_reserve_full_isl") is True


def test_five_policy_mixed_reconcile():
    client, adapter, reconciler = make_reconciler()
    desired = {
        "long_prefill_token_threshold": 128,
        "runtime_max_running_seqs": 2,
        "runtime_max_scheduled_tokens": 256,
        "runtime_kv_watermark": 0.1,
        "runtime_reserve_full_isl": False,
    }
    r = reconciler.reconcile(desired)
    assert r.status is ReconcileStatus.CONVERGED
    by = {s.param: s for s in r.steps}
    for p, d in desired.items():
        assert by[p].status is StepStatus.VERIFIED, p
        assert by[p].transition is TransitionClass.HOT, p
        assert adapter.get(p) == d, p
    # all five policies persisted
    assert reconciler.desired() == desired
    # the one-shot operation is NOT in DesiredState
    reconciler.reconcile({"reset_prefix_cache": 1})
    assert "reset_prefix_cache" not in reconciler.desired()