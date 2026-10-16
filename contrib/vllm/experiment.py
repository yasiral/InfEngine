#!/usr/bin/env python3
"""Milestone 3F live experiment: final vLLM Control Contract V1 (5 policies +
reset_prefix_cache operation) on real vLLM v0.28.0.
"""
import os
import subprocess
import threading
import time

import torch
from vllm import LLM, SamplingParams

from infengine.control import Reconciler, ReconcileStatus, StepStatus
from infengine.vllm import VllmAdapter, VllmRuntimeClient


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
        return {"api_pid": os.getpid(), "engine_pids": engine_pids(process_table())}


def process_table():
    out = subprocess.check_output(["ps", "-eo", "pid=,ppid=,args="], text=True)
    rows = []
    for line in out.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) == 3:
            rows.append({"pid": int(parts[0]), "ppid": int(parts[1]), "cmd": parts[2][:80]})
    return rows


def engine_pids(rows):
    return sorted(r["pid"] for r in rows
                  if "vllm" in r["cmd"] or "spawn_main" in r["cmd"] or "EngineCore" in r["cmd"])


def main():
    print("=== ENVIRONMENT ===", flush=True)
    print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
          torch.cuda.get_device_name(), torch.cuda.get_device_capability(), flush=True)

    print("=== LOAD MODEL (opt-125m, small KV to make admission proofs meaningful) ===", flush=True)
    llm = LLM(model="facebook/opt-125m", dtype="float16", max_model_len=512,
              max_num_seqs=4, enforce_eager=True, gpu_memory_utilization=0.15,
              enable_logging_iteration_details=True, disable_log_stats=False)
    engine = llm.llm_engine
    assert hasattr(engine, "set_runtime_scheduler_config"), "hook not applied"
    assert hasattr(engine, "get_runtime_cache_stats"), "cache-stats hook not applied"

    adapter = VllmAdapter(EngineClientRuntimeClient(engine))
    reconciler = Reconciler(adapter)
    engines_before = engine_pids(process_table())
    print("=== IDENTITY BEFORE === engine pids:", engines_before, flush=True)

    tok = llm.get_tokenizer()
    seed = tok.encode("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
                      "kilo lima mike november oscar papa quebec romeo sierra tango "
                      "uniform victor whiskey xray yankee zulu one two three four five. ",
                      add_special_tokens=False)
    lp_a = (seed * 60)[:400]
    lp_b = (tok.encode("zebra xray walrus violin umbrella tiger snake rabbit queen "
                       "penguin owl newt monkey lion kangaroo jaguar iguana horse "
                       "giraffe fox eagle dog cat bat ant bee cow deer fish goat. ",
                       add_special_tokens=False) * 60)[:400]

    # ---------------- FIVE-POLICY MIXED RECONCILE ----------------
    print("=== FIVE-POLICY MIXED RECONCILE ===", flush=True)
    desired = {
        "long_prefill_token_threshold": 128,
        "runtime_max_running_seqs": 2,
        "runtime_max_scheduled_tokens": 256,
        "runtime_kv_watermark": 0.2,
        "runtime_reserve_full_isl": False,
    }
    r = reconciler.reconcile(desired)
    for s in r.steps:
        print(" ", s.param, "->", s.status.value, "(", s.transition.value, ")", flush=True)
    print("overall:", r.status.value, flush=True)
    observed = {k: adapter.get(k) for k in desired}
    print("observed:", observed, flush=True)
    print("desired persisted:", reconciler.desired(), flush=True)
    stats = engine.get_runtime_cache_stats()
    print("cache stats: num_gpu_blocks=", stats["num_gpu_blocks"],
          "watermark_blocks=", stats["watermark_blocks"], flush=True)
    num_blocks_before = stats["num_gpu_blocks"]

    # ---------------- runtime_reserve_full_isl behavioral ----------------
    print("=== HOT #5 behavioral: reserve_full_isl (long prompt, first chunk fits / full doesn't) ===", flush=True)
    # shrink the token budget so a long prompt must chunk
    reconciler.reconcile({"runtime_max_scheduled_tokens": 64})
    reconciler.reconcile({"runtime_kv_watermark": 0.0})  # remove watermark headroom interference
    # fill the KV cache with running requests to reduce free below the full prompt size
    # fill the KV cache with long-running requests (hold blocks in decode) so free < full prompt
    fillers = [llm.generate({"prompt_token_ids": (seed * i)[:400]},
                            sampling_params=SamplingParams(max_tokens=256, temperature=0.0))
               for i in (5, 6, 7)]
    stats = engine.get_runtime_cache_stats()
    print("after fill: num_gpu_blocks=", stats["num_gpu_blocks"],
          "watermark_blocks=", stats["watermark_blocks"], flush=True)
    print("scheduler stats (Running/Waiting) around admission decision:", flush=True)

    reconciler.reconcile({"runtime_reserve_full_isl": True})
    print("reserve=True; submitting long-prompt candidate (full 400-token seq must fit KV)", flush=True)
    cand_true = llm.generate({"prompt_token_ids": lp_a},
                             sampling_params=SamplingParams(max_tokens=8, temperature=0.0))
    print("  candidate@reserve=True completed:", bool(cand_true), flush=True)

    reconciler.reconcile({"runtime_reserve_full_isl": False})
    print("reserve=False; submitting long-prompt candidate (only first chunk must fit)", flush=True)
    cand_false = llm.generate({"prompt_token_ids": lp_b},
                              sampling_params=SamplingParams(max_tokens=8, temperature=0.0))
    print("  candidate@reserve=False completed:", bool(cand_false), flush=True)

    # ---------------- runtime_kv_watermark behavioral ----------------
    print("=== HOT #4 behavioral: runtime_kv_watermark (admission headroom) ===", flush=True)
    reconciler.reconcile({"runtime_kv_watermark": 0.0})
    stats0 = engine.get_runtime_cache_stats()
    print("watermark=0.0 -> watermark_blocks=", stats0["watermark_blocks"],
          "num_gpu_blocks=", stats0["num_gpu_blocks"], flush=True)
    reconciler.reconcile({"runtime_kv_watermark": 0.25})
    stats1 = engine.get_runtime_cache_stats()
    print("watermark=0.25 -> watermark_blocks=", stats1["watermark_blocks"],
          "num_gpu_blocks=", stats1["num_gpu_blocks"],
          "| physical capacity unchanged:", stats1["num_gpu_blocks"] == num_blocks_before, flush=True)

    # ---------------- negative paths ----------------
    print("=== negative paths (planning rejects before RPC) ===", flush=True)
    for bad in [{"runtime_kv_watermark": -0.1}, {"runtime_kv_watermark": 1.0},
                {"runtime_kv_watermark": "x"}, {"runtime_reserve_full_isl": 1},
                {"runtime_reserve_full_isl": "true"}]:
        rr = reconciler.reconcile(bad)
        print(" ", list(bad)[0], "=", list(bad.values())[0], "->", rr.steps[0].status.value,
              "|", (rr.steps[0].reason or "")[:60], flush=True)

    # ---------------- DRAIN operation regression ----------------
    print("=== DRAIN operation regression: reset_prefix_cache ===", flush=True)
    _ = llm.generate({"prompt_token_ids": lp_a}, sampling_params=SamplingParams(max_tokens=8))
    _ = llm.generate({"prompt_token_ids": lp_a}, sampling_params=SamplingParams(max_tokens=8))
    before = engine.get_runtime_cache_stats()["prefix_cache_entries"]
    rd = reconciler.reconcile({"reset_prefix_cache": 1})
    after = engine.get_runtime_cache_stats()["prefix_cache_entries"]
    print("DRAIN:", rd.steps[0].status.value, "| entries", before, "->", after,
          "| cleared:", before > 0 and after == 0,
          "| not in DesiredState:", "reset_prefix_cache" not in reconciler.desired(), flush=True)

    # ---------------- identity ----------------
    engines_after = engine_pids(process_table())
    print("=== IDENTITY AFTER === engine pids:", engines_after, flush=True)
    print("EngineCore/worker PIDs unchanged:", engines_before == engines_after, flush=True)
    print("=== SUMMARY ===", flush=True)
    print("observed policies:", {k: adapter.get(k) for k in desired}, flush=True)
    del llm


if __name__ == "__main__":
    main()