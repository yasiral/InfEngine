import json
import os
import threading
import time

import pytest
import torch

from infengine.config import ModelConfig
from infengine.model import LlamaModel, precompute_freqs
from infengine.engine import LLMEngine, AdmissionError
from infengine.control import (
    Capability,
    InfengineAdapter,
    Reconciler,
    ReconcileStatus,
    StateStore,
    StateStoreError,
    StepStatus,
    TransitionClass,
)


def make_engine(max_batch=2, chunk_size=8, block_size=4, num_blocks=32):
    torch.manual_seed(1)
    cfg = ModelConfig(hidden_size=32, num_layers=2, num_heads=4, num_kv_heads=2,
                      vocab_size=128, intermediate_size=64, max_seq_len=64)
    device = "cpu"
    model = LlamaModel(cfg).to(dtype=torch.float32, device=device)
    cos, sin = precompute_freqs(cfg.head_dim, 64, 10000.0, torch.float32, device)
    return LLMEngine(model, None, device, torch.float32, cos=cos, sin=sin,
                     block_size=block_size, num_blocks=num_blocks,
                     max_batch=max_batch, chunk_size=chunk_size)


PROMPT = [5, 8, 23, 12, 7, 3, 9, 10]
PROMPT_B = [40, 41, 42, 43, 44, 45, 46, 47]


def _wait_until(pred, timeout=5.0):
    deadline = time.monotonic() + timeout
    while not pred():
        if time.monotonic() >= deadline:
            raise TimeoutError("condition not met within timeout")
        time.sleep(0.005)


# ---------------------------------------------------------------- HOT semantics
def test_hot_reconcile_max_batch():
    engine = make_engine(max_batch=2, chunk_size=8)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    assert adapter.get("scheduler.max_batch") == 2
    result = reconciler.reconcile({"scheduler.max_batch": 4})

    assert adapter.get("scheduler.max_batch") == 4
    assert adapter.engine is engine  # same process, no restart
    step = result.steps[0]
    assert step.status is StepStatus.VERIFIED
    assert step.transition is TransitionClass.HOT
    assert step.observed_before == 2
    assert step.observed_after == 4
    assert result.status is ReconcileStatus.CONVERGED
    assert result.identity["worker_pid"] == os.getpid()
    assert "engine_instance_id" in result.identity


def test_hot_reconcile_chunk_size():
    engine = make_engine(chunk_size=8)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    result = reconciler.reconcile({"scheduler.chunk_size": 16})
    assert adapter.get("scheduler.chunk_size") == 16
    assert result.steps[0].status is StepStatus.VERIFIED
    assert result.steps[0].transition is TransitionClass.HOT
    assert result.status is ReconcileStatus.CONVERGED


def test_hot_mutation_live_between_requests():
    """Mutating max_batch while a generator is running must not corrupt output."""
    engine = make_engine(max_batch=1, chunk_size=8)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    list(engine.generate(PROMPT, max_new_tokens=6, temperature=0.0))  # prime cache
    ref_b = list(make_engine(max_batch=1).generate(PROMPT_B, max_new_tokens=6,
                                                   temperature=0.0))
    results = {}

    def run():
        results["out"] = list(engine.generate(PROMPT_B, max_new_tokens=6, temperature=0.0))

    t = threading.Thread(target=run)
    t.start()
    reconciler.reconcile({"scheduler.max_batch": 3})  # mutate while running
    t.join()

    assert results["out"] == ref_b
    assert adapter.get("scheduler.max_batch") == 3
    assert engine.metrics["requests"] == 2


def test_hot_preserves_identity():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    before = adapter.identity()
    result = reconciler.reconcile({"scheduler.max_batch": 4})
    after = adapter.identity()

    assert before == after
    assert result.identity == after
    assert before["worker_pid"] == os.getpid()
    assert "engine_started_at" in before


# ---------------------------------------------------------------- desired state
def test_desired_persists_when_drain_blocked():
    engine = make_engine(max_batch=2, num_blocks=32)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=0.2)

    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)  # request live -> DRAIN cannot quiesce

    result = reconciler.reconcile({"scheduler.max_batch": 8, "kv_store.num_blocks": 512})
    by = {s.param: s for s in result.steps}

    assert by["scheduler.max_batch"].status is StepStatus.VERIFIED
    assert by["kv_store.num_blocks"].status is StepStatus.BLOCKED_BUSY
    assert result.status is ReconcileStatus.BLOCKED
    assert adapter.get("scheduler.max_batch") == 8
    assert adapter.get("kv_store.num_blocks") == 32  # observed unchanged

    # intent persists even though it could not be applied
    assert reconciler.desired()["kv_store.num_blocks"] == 512
    assert reconciler.desired()["scheduler.max_batch"] == 8
    list(gen)


def test_desired_observed_remain_distinct_after_partial():
    engine = make_engine(max_batch=2, num_blocks=32)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=0.2)

    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)
    reconciler.reconcile({"kv_store.num_blocks": 512})
    list(gen)

    desired = reconciler.desired()
    observed = adapter.observed_state()
    assert desired["kv_store.num_blocks"] == 512
    assert observed["kv_store.num_blocks"] == 32
    assert desired != observed


def test_mixed_valid_invalid_patch_rejected_atomically(tmp_path):
    """A PATCH containing any invalid entry rejects the ENTIRE intent update:
    no persistence, no engine mutation, no adapter apply calls."""
    engine = make_engine(max_batch=2, chunk_size=8)
    adapter = InfengineAdapter(engine)
    store = StateStore(tmp_path / "state.json")
    reconciler = Reconciler(adapter, store)
    before_desired = dict(store.desired)
    before_observed = adapter.observed_state()

    calls = []
    orig_apply = adapter.apply

    def spy(name, value):
        calls.append((name, value))
        return orig_apply(name, value)

    adapter.apply = spy

    r = reconciler.reconcile({"scheduler.max_batch": 8, "scheduler.chunk_size": "not-an-int"})
    assert r.status is ReconcileStatus.FAILED
    assert r.steps[0].status is StepStatus.REJECTED
    assert calls == []  # no adapter apply calls
    assert store.desired == before_desired  # zero persistence
    assert adapter.observed_state() == before_observed  # zero engine mutation
    assert not (tmp_path / "state.json").exists()
    assert "scheduler.max_batch" not in reconciler.desired()


def test_out_of_range_value_rejected():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    r = reconciler.reconcile({"scheduler.max_batch": 0})
    step = r.steps[0]
    assert step.status is StepStatus.REJECTED
    assert "below minimum" in step.reason
    assert r.status is ReconcileStatus.FAILED
    assert adapter.get("scheduler.max_batch") == 2
    assert "scheduler.max_batch" not in reconciler.desired()


def test_unknown_param_rejected_never_applied():
    engine = make_engine()
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    before = adapter.observed_state()

    r = reconciler.reconcile({"model.foo": 1})
    assert r.steps[0].status is StepStatus.REJECTED
    assert "unknown" in r.steps[0].reason
    assert r.status is ReconcileStatus.FAILED
    assert adapter.observed_state() == before
    assert "model.foo" not in reconciler.desired()


def test_typed_normalization_comparable():
    engine = make_engine(max_batch=4)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    # engine already at 4; a string "4" must normalize to int and become NOOP
    r = reconciler.reconcile({"scheduler.max_batch": "4"})
    assert r.steps[0].status is StepStatus.NOOP
    assert r.steps[0].desired == 4
    assert r.status is ReconcileStatus.CONVERGED
    assert reconciler.desired()["scheduler.max_batch"] == 4


# ---------------------------------------------------------------- transition plan
def test_plan_constructed_before_execution():
    engine = make_engine(max_batch=2, num_blocks=32)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    plan = reconciler.plan_for({
        "scheduler.max_batch": 8,
        "kv_store.num_blocks": 64,
        "nope": 1,
    })
    by = {t.param: t for t in plan.transitions}

    assert by["scheduler.max_batch"].status == "EXECUTE"
    assert by["scheduler.max_batch"].desired == 8
    assert by["scheduler.max_batch"].observed_before == 2
    assert by["kv_store.num_blocks"].transition is TransitionClass.DRAIN
    assert by["kv_store.num_blocks"].status == "EXECUTE"
    assert by["nope"].status == "REJECTED"
    assert plan.valid is False

    # nothing executed by planning alone
    assert adapter.get("scheduler.max_batch") == 2


def test_plan_for_typed_normalization():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    plan = reconciler.plan_for({"scheduler.max_batch": "8"})
    assert plan.transitions[0].desired == 8


# ---------------------------------------------------------------- overall status
def test_overall_status_converged():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    assert reconciler.reconcile({"scheduler.max_batch": 2}).status is ReconcileStatus.CONVERGED
    r = reconciler.reconcile({"scheduler.max_batch": 4})
    assert r.status is ReconcileStatus.CONVERGED
    assert all(s.status in (StepStatus.NOOP, StepStatus.VERIFIED) for s in r.steps)


def test_overall_status_failed():
    engine = make_engine()
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    assert reconciler.reconcile({"nope": 1}).status is ReconcileStatus.FAILED
    assert reconciler.reconcile({"scheduler.max_batch": "x"}).status is ReconcileStatus.FAILED


def test_overall_status_blocked():
    engine = make_engine(max_batch=2, num_blocks=32)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=0.2)
    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)
    assert reconciler.reconcile({"kv_store.num_blocks": 64}).status is ReconcileStatus.BLOCKED
    list(gen)


def test_restart_transition_rejected_never_applied():
    engine = make_engine()
    adapter = InfengineAdapter(engine)
    adapter._capabilities["runtime.engine_id"] = Capability(
        name="runtime.engine_id", transition=TransitionClass.RESTART,
        description="not in this slice", dtype="str", choices=("alpha", "beta"),
    )
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({"runtime.engine_id": "alpha"})
    step = r.steps[0]
    assert step.status is StepStatus.REJECTED
    assert "RESTART" in step.reason
    assert r.status is ReconcileStatus.FAILED


def test_repeated_reconcile_after_convergence_noop():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    reconciler.reconcile({"scheduler.max_batch": 8})
    r = reconciler.reconcile({"scheduler.max_batch": 8})
    assert r.steps[0].status is StepStatus.NOOP
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get("scheduler.max_batch") == 8


# ---------------------------------------------------------------- drain
def test_drain_and_wait_lifecycle():
    engine = make_engine(max_batch=4, chunk_size=8, num_blocks=128)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=5.0)
    start = threading.Event()
    release = threading.Event()
    results = {}

    def in_flight():
        gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
        next(gen)  # request live in scheduler
        start.set()
        release.wait()  # hold the request open until released
        list(gen)

    t = threading.Thread(target=in_flight)
    t.start()
    assert start.wait(5)
    assert len(engine.scheduler.requests) == 1

    def recon():
        results["r"] = reconciler.reconcile({"kv_store.num_blocks": 64})

    rt = threading.Thread(target=recon)
    rt.start()

    # drain begins: admission closed
    _wait_until(lambda: not adapter.admitting)
    assert adapter.admitting is False
    assert adapter.drain_active is True

    release.set()  # let the in-flight request finish; drain-and-wait proceeds
    rt.join(timeout=10)
    assert not rt.is_alive()

    r = results["r"]
    assert r.status is ReconcileStatus.CONVERGED
    assert r.steps[0].status is StepStatus.VERIFIED
    assert r.steps[0].transition is TransitionClass.DRAIN
    assert adapter.get("kv_store.num_blocks") == 64
    assert adapter.admitting is True  # admission resumed
    assert adapter.drain_active is False
    assert adapter.engine is engine
    t.join(timeout=5)


def test_drain_rebuilds_kv_pools():
    engine = make_engine(block_size=4, num_blocks=32)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    old_store = engine.store

    result = reconciler.reconcile({"kv_store.block_size": 8})
    assert result.steps[0].status is StepStatus.VERIFIED
    assert result.steps[0].transition is TransitionClass.DRAIN
    assert adapter.get("kv_store.block_size") == 8
    assert engine.store is not old_store  # pools rebuilt
    assert result.status is ReconcileStatus.CONVERGED

    out = list(engine.generate([5, 8, 23, 12, 7, 3], max_new_tokens=4, temperature=0.0))
    assert len(out) >= 1  # engine still serves after reinit


def test_drain_timeout_leaves_engine_safe():
    engine = make_engine(max_batch=4, chunk_size=8, num_blocks=128)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=0.2)

    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)  # request held open; drain cannot quiesce

    r = reconciler.reconcile({"kv_store.num_blocks": 64})
    step = r.steps[0]
    assert step.status is StepStatus.BLOCKED_BUSY
    assert "timed out" in step.reason
    assert r.status is ReconcileStatus.BLOCKED
    assert adapter.get("kv_store.num_blocks") == 128  # not applied
    assert adapter.admitting is True   # admission reopened (safe state)
    assert adapter.drain_active is False
    assert len(engine.scheduler.requests) == 1  # in-flight request untouched
    list(gen)
    assert reconciler.desired()["kv_store.num_blocks"] == 64  # intent preserved


def test_concurrent_drain_ownership_single_owner():
    """The adapter-level drain ownership guard: two simultaneous drain attempts
    yield exactly one owner; the second is deterministically blocked without
    altering drain state. (Reconciler-level concurrency is already serialized by
    single-flight, so this tests the adapter boundary directly.)"""
    engine = make_engine(max_batch=4, chunk_size=8, num_blocks=128)
    adapter = InfengineAdapter(engine)
    start = threading.Event()
    release = threading.Event()
    results = {}

    def in_flight():
        gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
        next(gen)
        start.set()
        release.wait()
        list(gen)

    t = threading.Thread(target=in_flight)
    t.start()
    assert start.wait(5)

    def first_drain():
        results["r1"] = adapter.drain(5.0)

    rt = threading.Thread(target=first_drain)
    rt.start()
    _wait_until(lambda: adapter.drain_active)

    # second drain attempt while a drain lifecycle is active -> blocked, no state change
    r2 = adapter.drain(5.0)
    assert r2.drained is False
    assert r2.owned is False
    assert "already in progress" in r2.reason
    assert adapter.drain_active is True  # first still owns
    assert adapter.admitting is False    # admission unchanged by the loser

    release.set()
    rt.join(timeout=10)
    assert not rt.is_alive()
    r1 = results["r1"]
    assert r1.drained is True
    assert r1.owned is True
    adapter.resume()
    assert adapter.drain_active is False
    assert adapter.admitting is True
    t.join(timeout=5)


def test_mixed_hot_and_drain_statuses():
    engine = make_engine(max_batch=2, num_blocks=32, chunk_size=8)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=0.2)

    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)  # busy
    r = reconciler.reconcile({"scheduler.max_batch": 8, "kv_store.num_blocks": 64})
    by = {s.param: s for s in r.steps}
    assert by["scheduler.max_batch"].status is StepStatus.VERIFIED
    assert by["scheduler.max_batch"].transition is TransitionClass.HOT
    assert by["kv_store.num_blocks"].status is StepStatus.BLOCKED_BUSY
    assert r.status is ReconcileStatus.BLOCKED
    assert adapter.get("scheduler.max_batch") == 8
    assert adapter.get("kv_store.num_blocks") == 32
    list(gen)


def test_mixed_hot_and_drain_success():
    engine = make_engine(max_batch=2, num_blocks=32, chunk_size=8)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({"scheduler.max_batch": 8, "kv_store.num_blocks": 64})
    by = {s.param: s for s in r.steps}
    assert by["scheduler.max_batch"].status is StepStatus.VERIFIED
    assert by["kv_store.num_blocks"].status is StepStatus.VERIFIED
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get("scheduler.max_batch") == 8
    assert adapter.get("kv_store.num_blocks") == 64


# ---------------------------------------------------------------- persistence
def test_desired_state_persisted_and_reapplied(tmp_path):
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    store = StateStore(tmp_path / "state.json")
    reconciler = Reconciler(adapter, store)

    reconciler.reconcile({"scheduler.max_batch": 6})
    assert (tmp_path / "state.json").exists()
    assert json.loads((tmp_path / "state.json").read_text())["scheduler.max_batch"] == 6

    engine2 = make_engine(max_batch=2)
    adapter2 = InfengineAdapter(engine2)
    reconciler2 = Reconciler(adapter2, StateStore(tmp_path / "state.json"))
    reconciler2.reconcile()
    assert adapter2.get("scheduler.max_batch") == 6


def test_capabilities_are_adapter_reported():
    engine = make_engine()
    adapter = InfengineAdapter(engine)
    caps = adapter.capabilities()
    assert set(caps) == {"scheduler.max_batch", "scheduler.chunk_size",
                         "kv_store.block_size", "kv_store.num_blocks"}
    assert caps["scheduler.max_batch"].transition is TransitionClass.HOT
    assert caps["kv_store.block_size"].transition is TransitionClass.DRAIN
    assert caps["scheduler.max_batch"].dtype == "int"
    assert caps["scheduler.max_batch"].min_value == 1
    for cap in caps.values():
        assert cap.supported


# ---------------------------------------------------------------- HTTP endpoints
class _StubTok:
    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(t) for t in ids)


def test_runtime_endpoints():
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from infengine.serve import create_app

    engine = make_engine(max_batch=2)
    engine.tokenizer = _StubTok()
    app = create_app(get_engine=lambda: engine, drain_timeout=0.2)
    client = TestClient(app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    r = client.get("/runtime/config")
    assert r.json()["observed"]["scheduler.max_batch"] == 2
    assert "identity" in r.json()

    r = client.patch("/runtime/config", json={"scheduler.max_batch": 5})
    assert r.status_code == 200
    assert r.json()["status"] == "CONVERGED"
    assert r.json()["steps"][0]["status"] == "VERIFIED"
    assert r.json()["steps"][0]["transition"] == "HOT"
    assert engine.scheduler.max_batch == 5
    ident = r.json()["identity"]

    r = client.patch("/runtime/config", json={"model.foo": 1})
    assert r.status_code == 422
    assert r.json()["status"] == "FAILED"

    r = client.patch("/runtime/config", json={"scheduler.max_batch": "not-an-int"})
    assert r.status_code == 422
    assert r.json()["status"] == "FAILED"

    r = client.patch("/runtime/config", json={"kv_store.num_blocks": 48})
    assert r.status_code == 200
    assert r.json()["steps"][0]["transition"] == "DRAIN"
    assert r.json()["steps"][0]["status"] == "VERIFIED"
    assert engine.store.k_pool[0].shape[0] == 48

    r = client.get("/runtime/status")
    body = r.json()
    assert body["admitting"] is True
    assert body["draining"] is False
    assert body["identity"]["worker_pid"] == ident["worker_pid"]
    assert len(body["capabilities"]) == 4
    assert body["capabilities"][0]["dtype"] == "int"
    assert body["last_reconcile"]["status"] == "CONVERGED"

    r = client.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": 4})
    assert r.status_code == 200


# ---------------------------------------------------------------- M2.5: engine admission
def test_engine_rejects_when_not_admitting():
    engine = make_engine()
    engine._admitting = False
    with pytest.raises(AdmissionError):
        list(engine.generate(PROMPT, max_new_tokens=4, temperature=0.0))
    assert engine.scheduler.requests == {}
    assert engine.store.seqs == {}
    assert engine.store.used_blocks == 0


def test_direct_generate_rejected_while_draining():
    """Direct engine.generate() bypassing the serve layer must be rejected at
    the engine boundary while a DRAIN owns the engine."""
    engine = make_engine(max_batch=4, chunk_size=8, num_blocks=128)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=5.0)
    start = threading.Event()
    release = threading.Event()

    def in_flight():
        gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
        next(gen)
        start.set()
        release.wait()
        list(gen)

    t = threading.Thread(target=in_flight)
    t.start()
    assert start.wait(5)

    results = {}

    def recon():
        results["r"] = reconciler.reconcile({"kv_store.num_blocks": 64})

    rt = threading.Thread(target=recon)
    rt.start()
    _wait_until(lambda: adapter.drain_active)
    assert adapter.admitting is False

    with pytest.raises(AdmissionError):
        list(engine.generate(PROMPT_B, max_new_tokens=4, temperature=0.0))
    assert len(engine.scheduler.requests) == 1  # only the pre-existing request

    release.set()
    rt.join(timeout=10)
    assert not rt.is_alive()
    t.join(timeout=5)
    assert adapter.admitting is True
    assert results["r"].steps[0].status is StepStatus.VERIFIED


def test_request_enqueue_never_adopted_into_discarded_store():
    """A request racing a drain is either admitted before admission closes (and
    DRAIN waits for it) or rejected after close; it is never adopted into a KV
    store that DRAIN then discards."""
    engine = make_engine(max_batch=4, chunk_size=8, num_blocks=128)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, drain_timeout=5.0)
    release = threading.Event()
    results = {}

    # Admitted before drain closes: DRAIN must wait for it and it must survive.
    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)

    def recon():
        results["r"] = reconciler.reconcile({"kv_store.num_blocks": 64})

    rt = threading.Thread(target=recon)
    rt.start()
    _wait_until(lambda: adapter.drain_active)

    # attempts after admission closed are rejected; none enters scheduler/KV
    for _ in range(3):
        with pytest.raises(AdmissionError):
            list(engine.generate([(10 + i) % 90 for i in range(8)],
                                 max_new_tokens=4, temperature=0.0))
    assert len(engine.scheduler.requests) == 1  # the admitted request only

    list(gen)  # the admitted request finishes; drain quiesces
    rt.join(timeout=10)
    assert not rt.is_alive()
    assert results["r"].status is ReconcileStatus.CONVERGED
    assert adapter.get("kv_store.num_blocks") == 64
    assert adapter.admitting is True


# ---------------------------------------------------------------- M2.5: StateStore
def test_state_save_roundtrip_atomic(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"scheduler.max_batch": 8, "kv_store.num_blocks": 512})
    assert (tmp_path / "state.json").exists()
    assert not (tmp_path / "state.json.tmp").exists()  # temp cleaned up
    data = json.loads((tmp_path / "state.json").read_text())
    assert data == {"scheduler.max_batch": 8, "kv_store.num_blocks": 512}


def test_state_corruption_is_observable(tmp_path):
    p = tmp_path / "state.json"
    p.write_text("{ this is not valid json")
    store = StateStore(p)
    with pytest.raises(StateStoreError):
        store.load()
    # desired must not be silently reset to empty intent
    assert store.desired == {}


def test_state_save_cleanup_on_failure(tmp_path, monkeypatch):
    import os

    p = tmp_path / "state.json"
    store = StateStore(p)

    def boom(src, dst):
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.save({"a": 1})
    assert not (tmp_path / "state.json.tmp").exists()  # temp cleaned up
    assert not p.exists()  # target never left torn


# ---------------------------------------------------------------- M2.5: event loop
def test_drain_keeps_event_loop_responsive():
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from infengine.serve import create_app

    engine = make_engine(max_batch=4, chunk_size=8, num_blocks=128)
    engine.tokenizer = _StubTok()
    app = create_app(get_engine=lambda: engine, drain_timeout=5.0)
    patch_client = TestClient(app)
    probe_client = TestClient(app)

    gen = engine.generate(PROMPT, max_new_tokens=8, temperature=0.0)
    next(gen)  # in-flight request -> the drain will have to wait

    result = {}

    def do_patch():
        result["r"] = patch_client.patch("/runtime/config", json={"kv_store.num_blocks": 64})

    pt = threading.Thread(target=do_patch)
    pt.start()

    _wait_until(lambda: not probe_client.get("/health").json()["admitting"])
    # while the drain waits, the event loop stays responsive
    r = probe_client.get("/health")
    assert r.status_code == 200 and r.json()["admitting"] is False
    r = probe_client.get("/runtime/status")
    assert r.status_code == 200 and r.json()["draining"] is True

    list(gen)  # finish the in-flight request; drain proceeds
    pt.join(timeout=10)
    assert not pt.is_alive()
    assert result["r"].json()["status"] == "CONVERGED"
    assert probe_client.get("/health").json()["admitting"] is True


# ---------------------------------------------------------------- M2.6: single-flight reconciliation
def test_reconciles_serialize_single_flight(monkeypatch):
    """Two concurrent PATCHes must serialize: B cannot begin its transaction
    until A completes. Proven with a counter (no two applies overlap) and an
    event order (B's apply strictly after A's apply finished)."""
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from infengine.serve import create_app

    engine = make_engine(max_batch=2)
    engine.tokenizer = _StubTok()
    app = create_app(get_engine=lambda: engine, drain_timeout=5.0)
    client_a = TestClient(app)
    client_b = TestClient(app)

    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    events = []
    orig_apply = InfengineAdapter.apply

    def blocking_apply(self, name, value):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            events.append(("apply", value))
        try:
            if not entered.is_set():
                entered.set()
                release.wait(10)  # hold A's transaction open
            res = orig_apply(self, name, value)
            if value == 4:
                with lock:
                    events.append(("done", value))
            return res
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(InfengineAdapter, "apply", blocking_apply)

    results = {}

    def do_a():
        results["a"] = client_a.patch("/runtime/config", json={"scheduler.max_batch": 4})

    def do_b():
        results["b"] = client_b.patch("/runtime/config", json={"scheduler.max_batch": 8})

    ta = threading.Thread(target=do_a)
    ta.start()
    assert entered.wait(10)  # A is provably mid-transaction, blocked in apply

    tb = threading.Thread(target=do_b)
    tb.start()

    # While A is held open, B must not enter apply: the invariant is constant.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with lock:
            assert events == [("apply", 4)], f"reconciles overlapped: {events}"
        time.sleep(0.005)

    release.set()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()

    # Serialization: B applied strictly after A's apply completed, never in parallel.
    assert max_active == 1
    assert events == [("apply", 4), ("done", 4), ("apply", 8)]

    # Ordering: A accepted first (4), then B (8) deterministically wins.
    assert results["a"].json()["status"] == "CONVERGED"
    assert results["a"].json()["steps"][0]["observed_after"] == 4
    assert results["b"].json()["status"] == "CONVERGED"
    assert results["b"].json()["steps"][0]["observed_after"] == 8
    cfg = client_a.get("/runtime/config").json()
    assert cfg["desired"]["scheduler.max_batch"] == 8
    assert cfg["observed"]["scheduler.max_batch"] == 8
    assert engine.scheduler.max_batch == 8


def test_reconcile_ordering_deterministic():
    """Sequential PATCHes A(4) then B(8) leave final state = B, with each
    result reflecting the transition it actually executed."""
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from infengine.serve import create_app

    engine = make_engine(max_batch=2)
    engine.tokenizer = _StubTok()
    app = create_app(get_engine=lambda: engine, drain_timeout=5.0)
    client = TestClient(app)

    ra = client.patch("/runtime/config", json={"scheduler.max_batch": 4})
    assert ra.status_code == 200
    assert ra.json()["steps"][0]["observed_before"] == 2
    assert ra.json()["steps"][0]["observed_after"] == 4

    rb = client.patch("/runtime/config", json={"scheduler.max_batch": 8})
    assert rb.status_code == 200
    assert rb.json()["steps"][0]["observed_before"] == 4
    assert rb.json()["steps"][0]["observed_after"] == 8

    cfg = client.get("/runtime/config").json()
    assert cfg["desired"]["scheduler.max_batch"] == 8
    assert cfg["observed"]["scheduler.max_batch"] == 8
    assert engine.scheduler.max_batch == 8


def test_serving_responsive_while_reconcile_waiting(monkeypatch):
    """While reconcile A runs and reconcile B waits, /health and /runtime/status
    stay responsive and normal inference is not serialized behind the reconcile
    executor."""
    fastapi = pytest.importorskip("fastapi")
    from starlette.testclient import TestClient
    from infengine.serve import create_app

    engine = make_engine(max_batch=2)
    engine.tokenizer = _StubTok()
    app = create_app(get_engine=lambda: engine, drain_timeout=5.0)
    client_a = TestClient(app)   # runs reconcile A
    client_b = TestClient(app)   # runs reconcile B (queued behind A)
    probe = TestClient(app)      # probes endpoints

    entered = threading.Event()
    release = threading.Event()
    orig_apply = InfengineAdapter.apply

    def blocking_apply(self, name, value):
        if not entered.is_set():
            entered.set()
            release.wait(10)
        return orig_apply(self, name, value)

    monkeypatch.setattr(InfengineAdapter, "apply", blocking_apply)

    results = {}

    def do_a():
        results["a"] = client_a.patch("/runtime/config", json={"scheduler.max_batch": 4})

    def do_b():
        results["b"] = client_b.patch("/runtime/config", json={"scheduler.max_batch": 8})

    ta = threading.Thread(target=do_a)
    ta.start()
    assert entered.wait(10)  # A is mid-transaction
    tb = threading.Thread(target=do_b)
    tb.start()

    # endpoints stay responsive while A runs and B waits
    for _ in range(10):
        assert probe.get("/health").status_code == 200
        assert probe.get("/runtime/status").status_code == 200
    # normal inference is not serialized behind the reconciliation executor
    r = probe.post("/v1/completions", json={"prompt": PROMPT, "max_tokens": 4})
    assert r.status_code == 200
    # B's PATCH is still pending (queued behind A)
    assert results.get("b") is None

    release.set()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()
    assert results["b"].json()["status"] == "CONVERGED"
    assert engine.scheduler.max_batch == 8


def test_reconciler_direct_calls_serialize():
    """The Reconciler itself is the authoritative single-flight boundary:
    two direct (non-FastAPI) concurrent reconcile() calls serialize, so B cannot
    enter plan/persist/apply while A owns reconciliation."""
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)

    entered = threading.Event()
    release = threading.Event()
    lock = threading.Lock()
    active = 0
    max_active = 0
    plan_entries = []
    apply_events = []
    orig_apply = adapter.apply
    orig_plan = reconciler.plan_for

    def spy_apply(name, value):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            apply_events.append(("apply", value))
        try:
            if not entered.is_set():
                entered.set()
                release.wait(10)  # hold A's transaction open inside apply
            res = orig_apply(name, value)
            if value == 4:
                with lock:
                    apply_events.append(("done", value))
            return res
        finally:
            with lock:
                active -= 1

    def spy_plan(desired):
        with lock:
            plan_entries.append(("plan", desired.get("scheduler.max_batch")))
        return orig_plan(desired)

    adapter.apply = spy_apply
    reconciler.plan_for = spy_plan

    results = {}

    def run(value):
        results[value] = reconciler.reconcile({"scheduler.max_batch": value})

    ta = threading.Thread(target=run, args=(4,))
    tb = threading.Thread(target=run, args=(8,))
    ta.start()
    assert entered.wait(10)  # A owns reconciliation, blocked inside apply

    tb.start()
    # While A holds the reconcile lock, B must not even reach planning.
    deadline = time.monotonic() + 1.0
    while time.monotonic() < deadline:
        with lock:
            assert plan_entries == [("plan", 4)], f"B leaked into plan: {plan_entries}"
            assert apply_events == [("apply", 4)], f"B leaked into apply: {apply_events}"
        time.sleep(0.005)

    release.set()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()

    assert max_active == 1  # never more than one active reconcile/apply
    assert plan_entries == [("plan", 4), ("plan", 8)]  # B began after A
    assert apply_events == [("apply", 4), ("done", 4), ("apply", 8)]

    assert results[4].steps[0].status is StepStatus.VERIFIED
    assert results[4].steps[0].observed_after == 4
    assert results[8].steps[0].status is StepStatus.VERIFIED
    assert results[8].steps[0].observed_after == 8
    assert adapter.get("scheduler.max_batch") == 8
    assert reconciler.desired()["scheduler.max_batch"] == 8