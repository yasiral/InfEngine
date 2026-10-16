"""Live vLLM V1 integration test (Milestone 3B) — REQUIRES GPU + patched vLLM.

Skipped unless ``VLLM_INTEGRATION=1`` and vLLM is importable. This mirrors the
container experiment in contrib/vllm/experiment.py: observe -> reconcile ->
observe on a live EngineCore, with process continuity, through the real
transport (patched LLMEngine / SyncMPClient -> EngineCore UTILITY RPC).
"""

import os
import subprocess

import pytest

from infengine.control import Reconciler, ReconcileStatus, StepStatus
from infengine.vllm import VllmAdapter, VllmRuntimeClient

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(os.environ.get("VLLM_INTEGRATION") != "1",
                       reason="set VLLM_INTEGRATION=1 and provide a patched vLLM runtime"),
]


class EngineClientRuntimeClient(VllmRuntimeClient):
    def __init__(self, llm_engine):
        self._e = llm_engine

    def get_runtime_scheduler_config(self):
        return self._e.get_runtime_scheduler_config()

    def set_runtime_scheduler_config(self, overrides):
        return self._e.set_runtime_scheduler_config(overrides)

    def pause_generation(self, mode="wait", clear_cache=False):
        return self._e.pause_generation(mode=mode, clear_cache=clear_cache)

    def resume_generation(self):
        return self._e.resume_generation()

    def reset_prefix_cache(self, reset_running_requests=False, reset_connector=False):
        return self._e.reset_prefix_cache(reset_running_requests, reset_connector)

    def get_runtime_cache_stats(self):
        return self._e.get_runtime_cache_stats()

    def identity(self):
        out = subprocess.check_output(["ps", "-eo", "pid=,args="], text=True)
        vllm_pids = sorted(int(l.split()[0]) for l in out.splitlines()
                           if "vllm" in l or "spawn_main" in l)
        return {"api_pid": os.getpid(), "engine_pids": vllm_pids}


def test_live_hot_mutation_preserves_process_identity():
    # WSL2 GPU passthrough requires explicitly enabling pinned memory (vLLM
    # gates UVA behind this; without it EngineCore fails with "UVA is not
    # available").
    if os.environ.get("VLLM_WSL2_ENABLE_PIN_MEMORY") != "1" and _in_wsl():
        pytest.skip("set VLLM_WSL2_ENABLE_PIN_MEMORY=1 under WSL2")

    vllm = pytest.importorskip("vllm")
    from vllm import LLM

    llm = LLM(model="facebook/opt-125m", dtype="float16", max_model_len=512,
              max_num_seqs=1, enforce_eager=True, gpu_memory_utilization=0.8)
    engine = llm.llm_engine
    assert hasattr(engine, "set_runtime_scheduler_config"), (
        "vLLM hook patch not applied: LLMEngine.set_runtime_scheduler_config missing"
    )

    adapter = VllmAdapter(EngineClientRuntimeClient(engine))
    reconciler = Reconciler(adapter)

    ident_before = adapter.identity()
    before = adapter.get("long_prefill_token_threshold")

    result = reconciler.reconcile({"long_prefill_token_threshold": 128})

    after = adapter.get("long_prefill_token_threshold")
    ident_after = adapter.identity()

    assert result.steps[0].status is StepStatus.VERIFIED
    assert result.status is ReconcileStatus.CONVERGED
    assert before != 128 and after == 128
    assert ident_after["engine_pids"] == ident_before["engine_pids"], (
        "EngineCore/worker processes must survive the HOT change"
    )

    del llm


def _in_wsl():
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False
