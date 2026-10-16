"""Live SGLang v0.5.18 integration test (M4B-B.1) — REQUIRES GPU + patched SGLang.

Skipped unless ``SGLANG_INTEGRATION=1`` and ``SGLANG_BASE_URL`` is set (e.g.
``http://localhost:30000``). Proves the private control plane drives a real
patched SGLang v0.5.18 runtime through ``SglangAdapter``/``HttpSglangRuntimeClient``
over the native HTTP admin surface — not direct curl.

Prerequisites (see contrib/sglang/README.md):

  * A running patched ``lmsysorg/sglang:v0.5.18`` server (runtime_max_prefill_tokens
    applied via the canonical patch; PP=1, DP=1).
  * The server reachable at ``SGLANG_BASE_URL``.

M4B-B.1 additions:

  * Real runtime_max_prefill_tokens readback via /server_info (no shadow state).
  * Divergence detection + reconvergence after an out-of-band direct mutation.
  * Busy ENGINE_CACHE_FLUSH -> BLOCKED with no retract side effects.
  * Idle ENGINE_CACHE_FLUSH -> VERIFIED.
"""

import json
import os
import threading
import time
import urllib.request

import pytest

from infengine.control import Reconciler, ReconcileStatus, StepStatus
from infengine.sglang import SglangAdapter, HttpSglangRuntimeClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("SGLANG_INTEGRATION") != "1",
                       reason="set SGLANG_INTEGRATION=1 and SGLANG_BASE_URL to a patched SGLang v0.5.18 runtime"),
]


def _base_url():
    return os.environ.get("SGLANG_BASE_URL", "http://localhost:30000")


def _make():
    client = HttpSglangRuntimeClient(_base_url(), timeout=15.0)
    adapter = SglangAdapter(client)
    return client, adapter, Reconciler(adapter)


def _out_of_band_set(server_args):
    """Direct HTTP mutation bypassing the control plane (simulates external /
    out-of-band divergence)."""
    body = json.dumps({"server_args": server_args}).encode("utf-8")
    req = urllib.request.Request(
        _base_url() + "/set_internal_state", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _generate_in_background(text, max_new_tokens):
    body = json.dumps({
        "text": text,
        "sampling_params": {"max_new_tokens": max_new_tokens,
                            "temperature": 0.7, "ignore_eos": True},
    }).encode("utf-8")
    req = urllib.request.Request(
        _base_url() + "/generate", data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    out = {}

    def _run():
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                out["body"] = resp.read().decode("utf-8")
        except Exception as exc:  # pragma: no cover
            out["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    return t, out


def _wait_running(client, timeout=10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        load = client.get_load()
        running = int(load.get("num_reqs", 0)) - int(load.get("num_waiting_reqs", 0))
        if running > 0:
            return True
        time.sleep(0.2)
    return False


def _force_reconcile(reconciler, adapter, name, target):
    """Force a real apply of ``target`` regardless of the current runtime value
    (integration tests share the live runtime and may start already-converged,
    which would otherwise be a NOOP). Moves to a distinct VALID baseline first
    when needed, then reconciles to ``target`` and asserts a real VERIFIED step."""
    cap = adapter.capability(name)
    maxv = cap.max_value if cap.max_value is not None else target + 1
    baseline = target + 1 if target < maxv else max(target - 1, 1)
    if baseline != target:
        r = reconciler.reconcile({name: baseline})
        assert r.steps[0].status is StepStatus.VERIFIED, r.to_dict()
    r = reconciler.reconcile({name: target})
    assert r.steps[0].status is StepStatus.VERIFIED, r.to_dict()
    assert r.status is ReconcileStatus.CONVERGED
    assert adapter.get(name) == target
    return r


def test_hot_concurrency_converges_and_preserves_identity():
    client, adapter, reconciler = _make()
    prov = client.provisioned()
    ident_before = adapter.identity()

    _force_reconcile(reconciler, adapter, "runtime_max_running_seqs", 1)
    assert adapter.get("runtime_max_running_seqs") == 1

    _force_reconcile(reconciler, adapter, "runtime_max_running_seqs", 3)
    assert adapter.get("runtime_max_running_seqs") == 3

    assert prov.get("max_running_requests", 0) >= 3
    assert adapter.identity()["engine_instance_id"] == ident_before["engine_instance_id"]
    assert adapter.identity()["runtime_pids"] == ident_before["runtime_pids"]


def test_hot_prefill_budget_converges_with_real_readback():
    client, adapter, reconciler = _make()
    prov = client.provisioned()

    _force_reconcile(reconciler, adapter, "runtime_max_prefill_tokens", 512)
    # real readback: the scheduler value, not shadow state
    assert adapter.get("runtime_max_prefill_tokens") == 512
    assert prov.get("max_prefill_tokens", 0) >= 512


def test_prefill_divergence_detected_and_reconverged():
    client, adapter, reconciler = _make()

    # establish a known baseline (reconcile may be NOOP if already 512)
    reconciler.reconcile({"runtime_max_prefill_tokens": 512})
    assert adapter.get("runtime_max_prefill_tokens") == 512

    # out-of-band direct mutation to another LEGAL value (bypassing the plane)
    resp = _out_of_band_set({"runtime_max_prefill_tokens": 256})
    assert resp == [True], f"out-of-band mutation rejected: {resp}"

    # control plane MUST observe the divergence (live readback)
    assert adapter.get("runtime_max_prefill_tokens") == 256

    # reconciliation detects divergence and restores desired
    r2 = reconciler.reconcile({"runtime_max_prefill_tokens": 512})
    assert r2.steps[0].status is StepStatus.VERIFIED, r2.to_dict()
    assert r2.status is ReconcileStatus.CONVERGED
    assert r2.steps[0].observed_before == 256
    assert adapter.get("runtime_max_prefill_tokens") == 512


def test_concurrency_above_provisioned_rejected():
    client, adapter, reconciler = _make()
    prov = client.provisioned()
    ceiling = prov.get("max_running_requests", 0)
    if ceiling <= 0:
        pytest.skip("runtime did not report provisioned max_running_requests")

    r = reconciler.reconcile({"runtime_max_running_seqs": ceiling + 1})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED
    assert adapter.get("runtime_max_running_seqs") != ceiling + 1


def test_engine_cache_flush_idle_verified():
    client, adapter, reconciler = _make()
    # ensure the engine is idle first (previous requests must have completed)
    load = client.get_load()
    assert int(load.get("num_reqs", 0)) - int(load.get("num_waiting_reqs", 0)) == 0
    assert int(load.get("num_waiting_reqs", 0)) == 0

    r = reconciler.reconcile({"engine_cache_flush": 1})
    step = r.steps[0]
    assert step.status is StepStatus.VERIFIED, r.to_dict()
    assert r.status is ReconcileStatus.CONVERGED
    assert "engine_cache_flush" not in reconciler.desired()


def test_engine_cache_flush_busy_blocked_no_retract_side_effects():
    client, adapter, reconciler = _make()

    # start a long generation and wait until it is RUNNING
    t, out = _generate_in_background(
        "Write a very long story about a robot learning to paint. " * 5,
        max_new_tokens=800)
    assert _wait_running(client), "test request never entered running state"

    # busy engine: ENGINE_CACHE_FLUSH must be BLOCKED, not retracted
    r = reconciler.reconcile({"engine_cache_flush": 1})
    assert r.steps[0].status is StepStatus.BLOCKED_BUSY, r.to_dict()
    assert r.status is ReconcileStatus.BLOCKED

    # let the generation finish so the runtime is clean for later tests
    t.join(timeout=60)
    assert "error" not in out, out.get("error")


def test_invalid_type_rejected_live():
    client, adapter, reconciler = _make()
    r = reconciler.reconcile({"runtime_max_prefill_tokens": 2.5})
    assert r.steps[0].status is StepStatus.REJECTED
    assert r.status is ReconcileStatus.FAILED