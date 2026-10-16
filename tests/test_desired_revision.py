"""M6B-1: atomic desired-state revision + conditional PATCH.

Pins the design from M6A r3 §10.3:

  * ``desired_revision`` is persistent SIBLING metadata (``{path}.revision``);
    the desired-state JSON file remains exactly the desired key/value dict
    (DesiredState key model unchanged).
  * ``save()`` increments the revision on every atomic intent acceptance,
    including a value-identical re-persist; rejected/conflicting requests and
    reads never bump it.
  * ``Reconciler.reconcile(..., expected_revision=...)`` compares the revision
    ATOMICALLY with the mutation inside the single-flight critical section; a
    mismatch returns ``CONFLICT`` with zero mutation (no persistence, no engine
    mutation, ``self.last`` untouched).
  * ``PATCH /runtime/config`` accepts an optional ``If-Match`` header: matching
    -> 200, stale -> 409 CONFLICT, malformed -> 422, absent -> backward
    compatible. ``GET /runtime/config`` returns ``desired_revision``.

Covers concurrent PATCH races, stale revisions, value-identical accepted
PATCHes, rejected requests, persistence failures, restart persistence, and
backward compatibility.
"""

import json
import os
import sys
import threading
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient  # noqa: E402

from infengine.control import (  # noqa: E402
    InfengineAdapter,
    Reconciler,
    ReconcileStatus,
    StateStore,
    StateStoreError,
    StepStatus,
)
from infengine.serve import create_app  # noqa: E402

from test_control import make_engine  # noqa: E402


# ------------------------------------------------------------------ StateStore
def test_revision_defaults_zero(tmp_path):
    assert StateStore(None).revision == 0
    assert StateStore(tmp_path / "state.json").revision == 0


def test_save_increments_revision_and_keeps_desired_file_format(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"scheduler.max_batch": 8, "kv_store.num_blocks": 512})
    assert store.revision == 1
    # desired-state file is EXACTLY the desired dict (key model unchanged)
    data = json.loads((tmp_path / "state.json").read_text())
    assert data == {"scheduler.max_batch": 8, "kv_store.num_blocks": 512}
    # revision lives in the sibling metadata file only
    meta = json.loads((tmp_path / "state.json.revision").read_text())
    assert meta == {"desired_revision": 1}
    store.save({"scheduler.max_batch": 9})
    assert store.revision == 2
    assert store.desired == {"scheduler.max_batch": 9}


def test_revision_persists_across_restart(tmp_path):
    p = tmp_path / "state.json"
    s1 = StateStore(p)
    s1.save({"scheduler.max_batch": 6})
    assert s1.revision == 1
    s2 = StateStore(p)  # eager load restores revision + desired
    assert s2.revision == 1
    assert s2.desired == {"scheduler.max_batch": 6}


def test_load_restores_revision_and_desired(tmp_path):
    p = tmp_path / "state.json"
    s1 = StateStore(p)
    s1.save({"a": 1})
    s2 = StateStore(p)
    s2.load()
    assert s2.revision == 1
    assert s2.desired == {"a": 1}


def test_legacy_store_without_revision_file_loads_revision_zero(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"scheduler.max_batch": 8}))
    store = StateStore(p)
    assert store.desired == {"scheduler.max_batch": 8}
    assert store.revision == 0


def test_reserved_revision_key_never_enters_desired(tmp_path):
    store = StateStore(tmp_path / "state.json")
    store.save({"a": 1, "desired_revision": 99})
    assert store.desired == {"a": 1}
    assert store.revision == 1  # bumped by the accepted save, never 99
    assert json.loads((tmp_path / "state.json").read_text()) == {"a": 1}


def test_in_memory_store_tracks_revision():
    store = StateStore(None)
    assert store.revision == 0
    store.save({"a": 1})
    assert store.revision == 1
    assert store.desired == {"a": 1}


def test_persistence_failure_no_increment_and_no_partial_files(tmp_path, monkeypatch):
    p = tmp_path / "state.json"
    store = StateStore(p)

    def boom(src, dst):
        raise OSError("simulated crash during replace")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError):
        store.save({"a": 1})
    assert store.revision == 0
    assert store.desired == {}
    assert not p.exists()
    assert not (tmp_path / "state.json.tmp").exists()
    assert not (tmp_path / "state.json.revision.tmp").exists()

    monkeypatch.undo()
    store.save({"a": 1})
    assert store.revision == 1
    assert json.loads(p.read_text()) == {"a": 1}


def test_second_write_failure_revision_metadata_ahead_documented(tmp_path, monkeypatch):
    """A crash between the two writes leaves the revision metadata one ahead of
    the desired file; in-memory state stays consistent and the next save
    converges both."""
    p = tmp_path / "state.json"
    store = StateStore(p)
    calls = {"n": 0}
    orig = StateStore._write_json

    def fail_second(self, path, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("desired-state write failed")
        return orig(self, path, data)

    monkeypatch.setattr(StateStore, "_write_json", fail_second)
    with pytest.raises(OSError):
        store.save({"a": 1})
    assert store.revision == 0  # in-memory uncommitted
    assert store.desired == {}
    assert (tmp_path / "state.json.revision").exists()  # metadata written first
    assert not p.exists()  # desired file untouched

    monkeypatch.undo()
    store.save({"a": 1})
    assert store.revision == 1
    assert json.loads(p.read_text()) == {"a": 1}
    assert json.loads((tmp_path / "state.json.revision").read_text()) == {
        "desired_revision": 1
    }


def test_corrupt_revision_file_raises(tmp_path):
    p = tmp_path / "state.json"
    p.write_text(json.dumps({"a": 1}))
    (tmp_path / "state.json.revision").write_text("{ not json")
    store = StateStore(p)
    with pytest.raises(StateStoreError):
        store.load()


def test_crash_window_end_to_end_recovery(tmp_path, monkeypatch):
    """Durability verification of the documented two-file crash window.

    Simulates a crash AFTER the revision metadata file is durably persisted but
    BEFORE the desired-state file is replaced, then reconstructs the store and
    verifies: (1) the revision never decreases, (2) stale If-Match requests are
    rejected with zero mutation, and (3) a subsequent successful write restores
    a consistent state/revision pair.

    NOTE: the compare-and-persist is atomic with respect to concurrent requests
    under the Reconciler lock; the two-file DURABLE commit is NOT crash-atomic.
    This test pins the recovery behavior of the non-crash-atomic window.
    """
    p = tmp_path / "state.json"
    store = StateStore(p)
    store.save({"scheduler.max_batch": 4})  # baseline accepted at revision 1
    assert store.revision == 1
    assert json.loads(p.read_text()) == {"scheduler.max_batch": 4}

    # Crash: the revision metadata write succeeds; the desired-state replace
    # never runs.
    calls = {"n": 0}
    orig = StateStore._write_json

    def crash_after_revision(self, path, data):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("crash before desired-state replace")
        return orig(self, path, data)

    monkeypatch.setattr(StateStore, "_write_json", crash_after_revision)
    with pytest.raises(OSError):
        store.save({"scheduler.max_batch": 8})
    monkeypatch.undo()

    # Reconstruct StateStore from disk (restart after the crash).
    store2 = StateStore(p)
    assert store2.desired == {"scheduler.max_batch": 4}  # old accepted content
    assert store2.revision == 2  # metadata one ahead (documented phantom bump)

    # (1) revision never decreases across the crash + reconstruction.
    assert store2.revision == store.revision + 1  # recovered == durable metadata
    store3 = StateStore(p)
    assert store3.revision == 2  # repeated reconstruction never lowers it

    # (2) stale If-Match requests are rejected with zero mutation.
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter, store2)
    r = reconciler.reconcile({"scheduler.max_batch": 8}, expected_revision=1)  # stale
    assert r.status is ReconcileStatus.CONFLICT
    assert store2.revision == 2  # unchanged
    assert engine.scheduler.max_batch == 2

    # (3) a subsequent successful write restores a consistent pair.
    r2 = reconciler.reconcile({"scheduler.max_batch": 8}, expected_revision=2)
    assert r2.status is ReconcileStatus.CONVERGED
    assert store2.revision == 3
    assert json.loads(p.read_text()) == {"scheduler.max_batch": 8}
    assert json.loads((tmp_path / "state.json.revision").read_text()) == {
        "desired_revision": 3
    }


# ------------------------------------------------------------------ Reconciler
def test_matching_expected_revision_converges_and_bumps():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    assert reconciler.store.revision == 0
    r = reconciler.reconcile({"scheduler.max_batch": 4}, expected_revision=0)
    assert r.status is ReconcileStatus.CONVERGED
    assert r.steps[0].status is StepStatus.VERIFIED
    assert reconciler.store.revision == 1


def test_stale_expected_revision_conflict_zero_mutation():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    r1 = reconciler.reconcile({"scheduler.max_batch": 4})
    assert r1.status is ReconcileStatus.CONVERGED
    assert reconciler.store.revision == 1

    before_desired = dict(reconciler.desired())
    before_observed = adapter.observed_state()
    before_last = reconciler.last
    calls = []
    orig_apply = adapter.apply

    def spy(name, value):
        calls.append(name)
        return orig_apply(name, value)

    adapter.apply = spy

    r = reconciler.reconcile({"scheduler.max_batch": 8}, expected_revision=0)  # stale
    assert r.status is ReconcileStatus.CONFLICT
    assert r.steps == []
    assert calls == []  # zero engine mutation
    assert reconciler.desired() == before_desired  # zero persistence
    assert reconciler.store.revision == 1  # revision unchanged
    assert adapter.observed_state() == before_observed
    assert reconciler.last is before_last  # last result untouched


def test_value_identical_accepted_patch_bumps_revision():
    engine = make_engine(max_batch=4)  # engine already at desired value
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({"scheduler.max_batch": 4})
    assert r.steps[0].status is StepStatus.NOOP
    assert r.status is ReconcileStatus.CONVERGED
    assert reconciler.store.revision == 1  # accepted intent re-persisted -> bump
    r2 = reconciler.reconcile({"scheduler.max_batch": 4}, expected_revision=1)
    assert r2.status is ReconcileStatus.CONVERGED
    assert reconciler.store.revision == 2


def test_rejected_request_does_not_bump_revision():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({"scheduler.max_batch": 0})
    assert r.status is ReconcileStatus.FAILED
    assert reconciler.store.revision == 0
    assert reconciler.desired() == {}
    r2 = reconciler.reconcile({"unknown.thing": 1})
    assert r2.status is ReconcileStatus.FAILED
    assert reconciler.store.revision == 0
    assert reconciler.desired() == {}


def test_conflict_precedes_content_validation():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    reconciler.reconcile({"scheduler.max_batch": 4})  # revision -> 1
    # stale revision AND invalid content: the guard fires first (deterministic)
    r = reconciler.reconcile({"scheduler.max_batch": 0}, expected_revision=0)
    assert r.status is ReconcileStatus.CONFLICT
    assert reconciler.store.revision == 1
    assert adapter.get("scheduler.max_batch") == 4


def test_bare_reconcile_reapplies_persisted_intent_and_bumps(tmp_path):
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    store = StateStore(tmp_path / "state.json")
    reconciler = Reconciler(adapter, store)
    reconciler.reconcile({"scheduler.max_batch": 6})
    assert store.revision == 1

    engine2 = make_engine(max_batch=2)
    adapter2 = InfengineAdapter(engine2)
    reconciler2 = Reconciler(adapter2, StateStore(tmp_path / "state.json"))
    assert reconciler2.store.revision == 1  # restored before any reconcile
    r = reconciler2.reconcile()
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter2.get("scheduler.max_batch") == 6
    assert reconciler2.store.revision == 2  # re-acceptance re-persists


def test_reconcile_persistence_failure_propagates_no_partial_state(tmp_path, monkeypatch):
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    store = StateStore(tmp_path / "state.json")
    reconciler = Reconciler(adapter, store)

    def boom(*a, **k):
        raise StateStoreError("disk full")

    monkeypatch.setattr(store, "save", boom)
    with pytest.raises(StateStoreError):
        reconciler.reconcile({"scheduler.max_batch": 8})
    assert store.revision == 0
    assert store.desired == {}
    assert adapter.get("scheduler.max_batch") == 2  # no engine mutation
    assert reconciler.last is None


def test_concurrent_reconcile_same_expected_revision_one_conflict():
    engine = make_engine(max_batch=2)
    adapter = InfengineAdapter(engine)
    reconciler = Reconciler(adapter)
    results = {}

    def run(value):
        results[value] = reconciler.reconcile(
            {"scheduler.max_batch": value}, expected_revision=0
        )

    ta = threading.Thread(target=run, args=(4,))
    tb = threading.Thread(target=run, args=(8,))
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()

    # Exactly one thread converges, the other conflicts; the atomic guard makes
    # this deterministic no matter which thread wins the lock first.
    statuses = sorted((results[4].status, results[8].status),
                      key=lambda s: s.value)
    assert statuses == [ReconcileStatus.CONFLICT, ReconcileStatus.CONVERGED]
    assert reconciler.store.revision == 1
    winner = reconciler.desired()["scheduler.max_batch"]
    assert winner in (4, 8)
    assert results[winner].status is ReconcileStatus.CONVERGED
    loser = 8 if winner == 4 else 4
    assert results[loser].status is ReconcileStatus.CONFLICT
    assert adapter.get("scheduler.max_batch") == winner


# ---------------------------------------------------------------- serve layer
def _app():
    engine = make_engine(max_batch=2)
    return engine, create_app(get_engine=lambda: engine, drain_timeout=5.0)


def test_get_runtime_config_reports_desired_revision():
    engine, app = _app()
    client = TestClient(app)
    assert client.get("/runtime/config").json()["desired_revision"] == 0
    client.patch("/runtime/config", json={"scheduler.max_batch": 4})
    assert client.get("/runtime/config").json()["desired_revision"] == 1


def test_patch_matching_if_match_returns_200_with_revision():
    engine, app = _app()
    client = TestClient(app)
    r = client.patch("/runtime/config", json={"scheduler.max_batch": 4},
                     headers={"If-Match": "0"})
    assert r.status_code == 200
    assert r.json()["status"] == "CONVERGED"
    assert r.json()["desired_revision"] == 1
    assert engine.scheduler.max_batch == 4


def test_patch_stale_if_match_returns_409_zero_mutation():
    engine, app = _app()
    client = TestClient(app)
    r0 = client.patch("/runtime/config", json={"scheduler.max_batch": 4})
    assert r0.status_code == 200
    assert r0.json()["desired_revision"] == 1

    r = client.patch("/runtime/config", json={"scheduler.max_batch": 8},
                     headers={"If-Match": "0"})  # stale
    assert r.status_code == 409
    body = r.json()
    assert body["status"] == "CONFLICT"
    assert body["desired_revision"] == 1
    assert body["desired"]["scheduler.max_batch"] == 4
    assert body["observed"]["scheduler.max_batch"] == 4
    assert engine.scheduler.max_batch == 4  # zero engine mutation
    assert client.get("/runtime/config").json()["desired_revision"] == 1


def test_patch_without_if_match_backward_compatible():
    engine, app = _app()
    client = TestClient(app)
    r = client.patch("/runtime/config", json={"scheduler.max_batch": 4})
    assert r.status_code == 200
    assert r.json()["status"] == "CONVERGED"
    assert r.json()["desired_revision"] == 1
    assert engine.scheduler.max_batch == 4


def test_patch_invalid_if_match_returns_422():
    engine, app = _app()
    client = TestClient(app)
    r = client.patch("/runtime/config", json={"scheduler.max_batch": 4},
                     headers={"If-Match": "abc"})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "FAILED"
    assert "If-Match" in body["reason"]
    assert body["desired_revision"] == 0
    assert engine.scheduler.max_batch == 2
    assert client.get("/runtime/config").json()["desired_revision"] == 0


def test_invalid_intent_422_includes_revision_no_bump():
    engine, app = _app()
    client = TestClient(app)
    r = client.patch("/runtime/config", json={"model.nope": 1})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "FAILED"
    assert body["desired_revision"] == 0
    assert engine.scheduler.max_batch == 2
    assert client.get("/runtime/config").json()["desired_revision"] == 0


def test_concurrent_patches_same_if_match_one_conflict():
    engine, app = _app()
    client_pre = TestClient(app)
    assert client_pre.get("/runtime/config").json()["desired_revision"] == 0
    client_a = TestClient(app)
    client_b = TestClient(app)
    results = {}

    def do(client, value):
        results[value] = client.patch(
            "/runtime/config", json={"scheduler.max_batch": value},
            headers={"If-Match": "0"},
        )

    ta = threading.Thread(target=do, args=(client_a, 4))
    tb = threading.Thread(target=do, args=(client_b, 8))
    ta.start()
    tb.start()
    ta.join(timeout=10)
    tb.join(timeout=10)
    assert not ta.is_alive() and not tb.is_alive()

    codes = sorted((results[4].status_code, results[8].status_code))
    assert codes == [200, 409]
    winner = next(v for v in (4, 8) if results[v].status_code == 200)
    assert engine.scheduler.max_batch == winner
    cfg = client_pre.get("/runtime/config").json()
    assert cfg["desired_revision"] == 1
    assert cfg["desired"]["scheduler.max_batch"] == winner
    assert cfg["observed"]["scheduler.max_batch"] == winner


def test_restart_persistence_via_serve(tmp_path):
    p = tmp_path / "state.json"
    engine1 = make_engine(max_batch=2)
    app1 = create_app(get_engine=lambda: engine1, state_store=StateStore(p),
                      drain_timeout=5.0)
    c1 = TestClient(app1)
    r = c1.patch("/runtime/config", json={"scheduler.max_batch": 6})
    assert r.status_code == 200
    assert r.json()["desired_revision"] == 1

    engine2 = make_engine(max_batch=2)
    app2 = create_app(get_engine=lambda: engine2, state_store=StateStore(p),
                      drain_timeout=5.0)
    c2 = TestClient(app2)
    cfg = c2.get("/runtime/config").json()
    assert cfg["desired_revision"] == 1  # restored after restart
    assert cfg["desired"]["scheduler.max_batch"] == 6