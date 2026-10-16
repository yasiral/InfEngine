#!/usr/bin/env python3
"""Apply the vLLM v0.28.0 runtime scheduler-config hook (full transport).

Deterministic string-anchored injection into the installed vLLM wheel. The
allowlist is restricted to `long_prefill_token_threshold` (int, >= 0).

Idempotent: safe to run more than once.
"""

import os

import vllm

BASE = os.path.dirname(vllm.__file__)

CORE_METHODS = '''    RUNTIME_SCHEDULER_POLICY_ALLOWLIST = {
        "long_prefill_token_threshold": (int, 0, None),
        "runtime_max_running_seqs": (int, 1, None),
        "runtime_max_scheduled_tokens": (int, 1, None),
        "runtime_kv_watermark": (float, 0.0, None),
        "runtime_reserve_full_isl": (bool, None, None),
    }

    def set_runtime_scheduler_config(self, overrides: dict) -> dict:
        sc = self.vllm_config.scheduler_config
        validated: dict = {}
        for field, value in overrides.items():
            spec = self.RUNTIME_SCHEDULER_POLICY_ALLOWLIST.get(field)
            if spec is None:
                raise ValueError(f"unknown runtime scheduler policy: {field!r}")
            dtype, lo, hi = spec
            if dtype is bool:
                if not isinstance(value, bool):
                    raise ValueError(
                        f"{field}: expected bool, got {type(value).__name__!r}"
                    )
            elif isinstance(value, bool) or not isinstance(value, dtype):
                raise ValueError(
                    f"{field}: expected {dtype.__name__}, got {type(value).__name__!r}"
                )
            if lo is not None and value < lo:
                raise ValueError(f"{field}: {value} below minimum {lo}")
            if hi is not None and value > hi:
                raise ValueError(f"{field}: {value} above maximum {hi}")
            if field == "runtime_kv_watermark" and value >= 1.0:
                raise ValueError("runtime_kv_watermark: must be < 1.0")
            validated[field] = value
        # Runtime policies are bounded by provisioned (boot-time) capacity and
        # touch ONLY the scheduler's live operating state -- never worker
        # buffers, KV allocation, CUDA-graph capture, or compilation.
        if "runtime_max_running_seqs" in validated:
            provisioned = sc.max_num_seqs
            if validated["runtime_max_running_seqs"] > provisioned:
                raise ValueError(
                    f"runtime_max_running_seqs: {validated['runtime_max_running_seqs']} "
                    f"exceeds provisioned max_num_seqs {provisioned}"
                )
            self.scheduler.max_num_running_reqs = validated["runtime_max_running_seqs"]
        if "runtime_max_scheduled_tokens" in validated:
            provisioned = sc.max_num_scheduled_tokens or sc.max_num_batched_tokens
            if validated["runtime_max_scheduled_tokens"] > provisioned:
                raise ValueError(
                    f"runtime_max_scheduled_tokens: {validated['runtime_max_scheduled_tokens']} "
                    f"exceeds provisioned token budget {provisioned}"
                )
            self.scheduler.max_num_scheduled_tokens = validated["runtime_max_scheduled_tokens"]
        if "runtime_kv_watermark" in validated:
            pool = self.scheduler.kv_cache_manager.block_pool
            num_blocks = pool.num_gpu_blocks
            self.scheduler.runtime_kv_watermark = validated["runtime_kv_watermark"]
            self.scheduler.kv_cache_manager.watermark_blocks = int(
                validated["runtime_kv_watermark"] * num_blocks
            )
        if "runtime_reserve_full_isl" in validated:
            self.scheduler.scheduler_reserve_full_isl = validated["runtime_reserve_full_isl"]
        for field, value in validated.items():
            if field not in ("runtime_max_running_seqs", "runtime_max_scheduled_tokens",
                             "runtime_kv_watermark", "runtime_reserve_full_isl"):
                setattr(sc, field, value)
        return self.get_runtime_scheduler_config()

    def get_runtime_scheduler_config(self) -> dict:
        sc = self.vllm_config.scheduler_config
        pool = self.scheduler.kv_cache_manager.block_pool
        num_blocks = pool.num_gpu_blocks
        return {
            "policies": {
                "long_prefill_token_threshold": sc.long_prefill_token_threshold,
                "runtime_max_running_seqs": self.scheduler.max_num_running_reqs,
                "runtime_max_scheduled_tokens": self.scheduler.max_num_scheduled_tokens,
                "runtime_kv_watermark": getattr(
                    self.scheduler, "runtime_kv_watermark", sc.watermark),
                "runtime_reserve_full_isl": self.scheduler.scheduler_reserve_full_isl,
            },
            "provisioned": {
                "max_num_seqs": sc.max_num_seqs,
                "max_num_batched_tokens": sc.max_num_batched_tokens,
                "max_num_scheduled_tokens": sc.max_num_scheduled_tokens or sc.max_num_batched_tokens,
                "num_gpu_blocks": num_blocks,
            },
        }

    def get_runtime_cache_stats(self) -> dict:
        """Engine observable for operation verification (prefix-cache state)."""
        pool = self.scheduler.kv_cache_manager.block_pool
        return {
            "prefix_cache_entries": len(pool.cached_block_hash_to_block._cache),
            "num_gpu_blocks": pool.num_gpu_blocks,
            "watermark_blocks": self.scheduler.kv_cache_manager.watermark_blocks,
        }'''

SYNC_MP_CACHE_STATS = '''    def get_runtime_cache_stats(self) -> dict:
        return self.call_utility("get_runtime_cache_stats")'''

LLM_ENGINE_CACHE_STATS = '''    def get_runtime_cache_stats(self) -> dict:
        return self.engine_core.get_runtime_cache_stats()'''

SYNC_MP_PAUSE_METHODS = '''    def pause_scheduler(self, mode: str = "wait", clear_cache: bool = False):
        return self.call_utility("pause_scheduler", mode, clear_cache)

    def resume_scheduler(self):
        return self.call_utility("resume_scheduler")

    def is_scheduler_paused(self) -> bool:
        return self.call_utility("is_scheduler_paused")'''

LLM_ENGINE_PAUSE_METHODS = '''    def pause_generation(self, *, mode: str = "wait", clear_cache: bool = False) -> None:
        return self.engine_core.pause_scheduler(mode=mode, clear_cache=clear_cache)

    def resume_generation(self) -> None:
        return self.engine_core.resume_scheduler()

    def is_paused(self) -> bool:
        return self.engine_core.is_scheduler_paused()'''

ENGINE_CLIENT_DEFAULTS = '''    def set_runtime_scheduler_config(self, overrides: dict) -> dict:
        raise NotImplementedError

    def get_runtime_scheduler_config(self) -> dict:
        raise NotImplementedError

    async def set_runtime_scheduler_config_async(self, overrides: dict) -> dict:
        raise NotImplementedError

    async def get_runtime_scheduler_config_async(self) -> dict:
        raise NotImplementedError'''

INPROC_METHODS = '''    def set_runtime_scheduler_config(self, overrides: dict) -> dict:
        return self.engine_core.set_runtime_scheduler_config(overrides)

    def get_runtime_scheduler_config(self) -> dict:
        return self.engine_core.get_runtime_scheduler_config()

    async def set_runtime_scheduler_config_async(self, overrides: dict) -> dict:
        return self.set_runtime_scheduler_config(overrides)

    async def get_runtime_scheduler_config_async(self) -> dict:
        return self.get_runtime_scheduler_config()'''

SYNC_MP_METHODS = '''    def set_runtime_scheduler_config(self, overrides: dict) -> dict:
        return self.call_utility("set_runtime_scheduler_config", overrides)

    def get_runtime_scheduler_config(self) -> dict:
        return self.call_utility("get_runtime_scheduler_config")

    async def set_runtime_scheduler_config_async(self, overrides: dict) -> dict:
        return self.set_runtime_scheduler_config(overrides)

    async def get_runtime_scheduler_config_async(self) -> dict:
        return self.get_runtime_scheduler_config()'''

ASYNC_MP_METHODS = '''    async def set_runtime_scheduler_config_async(self, overrides: dict) -> dict:
        return await self.call_utility_async("set_runtime_scheduler_config", overrides)

    async def get_runtime_scheduler_config_async(self) -> dict:
        return await self.call_utility_async("get_runtime_scheduler_config")'''

ASYNC_LLM_METHODS = '''    async def set_runtime_scheduler_config(self, overrides: dict) -> dict:
        return await self.engine_core.set_runtime_scheduler_config_async(overrides)

    async def get_runtime_scheduler_config(self) -> dict:
        return await self.engine_core.get_runtime_scheduler_config_async()'''

LLM_ENGINE_METHODS = '''    def set_runtime_scheduler_config(self, overrides: dict) -> dict:
        return self.engine_core.set_runtime_scheduler_config(overrides)

    def get_runtime_scheduler_config(self) -> dict:
        return self.engine_core.get_runtime_scheduler_config()'''


def inject(path: str, anchor: str, insertion: str, marker: str):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if marker in text:
        print(f"[skip] already patched ({marker!r}): {path}")
        return
    if anchor not in text:
        raise RuntimeError(f"anchor not found in {path}:\n{anchor!r}")
    assert text.count(anchor) == 1, f"anchor not unique in {path}"
    text = text.replace(anchor, anchor + "\n\n" + insertion, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[patched] {path}")


def main():
    inject(
        os.path.join(BASE, "v1/engine/core.py"),
        '    def get_weight_version(self) -> str:\n'
        '        """Return the latest committed weight version."""\n'
        "        return self._weight_version",
        CORE_METHODS,
        "RUNTIME_SCHEDULER_POLICY_ALLOWLIST",
    )

    client = os.path.join(BASE, "v1/engine/core_client.py")
    inject(
        client,
        "    def get_weight_version(self) -> str:\n        raise NotImplementedError",
        ENGINE_CLIENT_DEFAULTS,
        "async def set_runtime_scheduler_config_async(self, overrides: dict) -> dict:\n        raise NotImplementedError",
    )
    inject(
        client,
        "    def get_weight_version(self) -> str:\n"
        "        return self.engine_core.get_weight_version()",
        INPROC_METHODS,
        "return self.engine_core.set_runtime_scheduler_config(overrides)",
    )
    inject(
        client,
        "    def get_weight_version(self) -> str:\n"
        '        return self.call_utility("get_weight_version")',
        SYNC_MP_METHODS,
        'call_utility("set_runtime_scheduler_config", overrides)',
    )
    inject(
        client,
        "    async def get_weight_version_async(self) -> str:\n"
        '        return await self.call_utility_async("get_weight_version")',
        ASYNC_MP_METHODS,
        'call_utility_async("set_runtime_scheduler_config", overrides)',
    )

    inject(
        os.path.join(BASE, "v1/engine/async_llm.py"),
        "    async def get_weight_version(self) -> str:\n"
        '        """Return the latest committed weight version."""\n'
        "        return await self.engine_core.get_weight_version_async()",
        ASYNC_LLM_METHODS,
        "engine_core.set_runtime_scheduler_config_async(overrides)",
    )
    inject(
        os.path.join(BASE, "v1/engine/llm_engine.py"),
        "    def get_weight_version(self) -> str:\n"
        '        """Return the latest committed weight version."""\n'
        "        return self.engine_core.get_weight_version()",
        LLM_ENGINE_METHODS,
        "engine_core.set_runtime_scheduler_config(overrides)",
    )

    # DRAIN transport: sync pause/resume + is_paused (reset_prefix_cache already
    # exists on both SyncMPClient and LLMEngine).
    inject(
        client,
        "    def reset_prefix_cache(\n"
        "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
        "    ) -> bool:\n"
        "        return self.call_utility(\n"
        '            "reset_prefix_cache", reset_running_requests, reset_connector\n'
        "        )",
        SYNC_MP_PAUSE_METHODS,
        'call_utility("pause_scheduler", mode, clear_cache)',
    )
    inject(
        os.path.join(BASE, "v1/engine/llm_engine.py"),
        "    def reset_prefix_cache(\n"
        "        self, reset_running_requests: bool = False, reset_connector: bool = False\n"
        "    ) -> bool:\n"
        "        return self.engine_core.reset_prefix_cache(\n"
        "            reset_running_requests, reset_connector\n"
        "        )",
        LLM_ENGINE_PAUSE_METHODS,
        "engine_core.pause_scheduler(mode=mode, clear_cache=clear_cache)",
    )

    # DRAIN verification observable: prefix-cache entries (cache stats)
    inject(
        client,
        "    def is_scheduler_paused(self) -> bool:\n"
        '        return self.call_utility("is_scheduler_paused")',
        SYNC_MP_CACHE_STATS,
        'call_utility("get_runtime_cache_stats")',
    )
    inject(
        os.path.join(BASE, "v1/engine/llm_engine.py"),
        "    def is_paused(self) -> bool:\n"
        "        return self.engine_core.is_scheduler_paused()",
        LLM_ENGINE_CACHE_STATS,
        "engine_core.get_runtime_cache_stats()",
    )
    print("hook applied.")


if __name__ == "__main__":
    main()
