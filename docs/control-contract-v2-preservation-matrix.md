# Control Contract V2 — Capability Preservation Matrix

Repository-derived preservation baseline for **M4B-C**. Every capability and
operation already implemented by the vLLM v0.28.0 and SGLang v0.5.18 adapters is
recorded below **before** any V2 change. The V2 operator contract must expose the
**UNION** of these capabilities (common intent only where semantics genuinely
match; engine-specific capabilities first-class), with no capability removed,
hidden, weakened, renamed incompatibly, or silently re-mapped to a different
semantic.

Source of truth: the live adapters and their transports
(`src/infengine/vllm/adapter.py`, `src/infengine/sglang/adapter.py`,
`src/infengine/vllm/runtime_client.py`, `src/infengine/sglang/runtime_client.py`,
`src/infengine/control/capabilities.py`, `src/infengine/control/reconciler.py`)
plus the M4B-B.1 report and the DeepSeek V4 Pro review (authoritative where
historical notes disagree).

---

## 1. Engine-agnostic control-plane machinery (preserved, NOT modified)

| Concern | Mechanism | Existing location | Notes |
|---------|-----------|-------------------|-------|
| Desired State | `StateStore` JSON persistence (atomic temp-file + fsync + `os.replace`) | `control/state.py` | Kept; no key migration in this milestone. |
| Acceptance | `Reconciler.reconcile` atomic intent acceptance (all-or-nothing; invalid/unknown entry rejects the WHOLE request with zero mutation) | `control/reconciler.py` | Preserved. |
| Reconciliation single-flight | `_reconcile_lock` + single-worker serve executor | `control/reconciler.py`, `serve.py` | Preserved. |
| Transition plan/class | `TransitionPlan`, `PlannedTransition`, `TransitionClass` (`HOT`/`DRAIN`/`RESTART`/`REDEPLOY`) | `control/reconciler.py`, `control/capabilities.py` | No new variants. |
| Persistent policy vs one-shot operation | `Capability.operation`; operations never persisted into Desired State | `control/capabilities.py` | Preserved. |
| Quiescence metadata | `QuiescenceStrategy` (`WAIT_INFLIGHT`/`RETRACT_INFLIGHT`), `quiescence_requirement`, `requires_quiescence` | `control/capabilities.py` | Metadata-only; Reconciler never branches on it. |
| Runtime-derived bounds | `Capability.min_value`/`max_value` from boot provisioned ceilings | each adapter | Preserved. |
| Real observed-state verification | `adapter.get()`/`observed_state()` live readback; `get()==desired` or operation `verify()` | adapters | Preserved. |
| Process identity | `adapter.identity()` (PIDs, instance id) stable across mutations | adapters | Preserved. |
| Drain failure cleanup | `_exec_drain_cycle` always `resume()`; resume failure fails overall lifecycle | `control/reconciler.py` | Preserved (M4B-B.1 fix #3). |
| Unsupported-op rejection before mutation | `adapter.coerce` raises on `supported=False` at planning; never enters DRAIN | adapters | Preserved (M4B-B.1 fix #4). |

---

## 2. vLLM v0.28.0 capabilities

| # | Internal key | Engine mechanism | Persistent / One-shot | Supported | Exact semantic | Scope | Runtime-derived bounds | Transition | Quiescence requirement/strategy | Verification | Existing API exposure | Existing tests |
|---|--------------|------------------|-----------------------|-----------|----------------|-------|------------------------|------------|----------------------------------|--------------|------------------------|----------------|
| 1 | `runtime_max_running_seqs` | EngineCore runtime scheduler-config allowlist (`set_runtime_scheduler_config` → scheduler admission gate) | Persistent | Yes | `CONCURRENCY_UPPER_BOUND` | engine-local admission gate | int, min 1, max provisioned `max_num_seqs` | HOT | — | `get_runtime_scheduler_config()["policies"][name] == desired` | `/runtime/config` PATCH/GET | `test_vllm_adapter.py`, `test_vllm_integration.py` |
| 2 | `runtime_max_scheduled_tokens` | EngineCore runtime scheduler token budget (per-step) | Persistent | Yes | `STRICT_SCHEDULED_TOKEN_BUDGET` (strict per-step total scheduled-token budget) | engine-local scheduler step | int, min 1, max provisioned `max_num_scheduled_tokens` | HOT | — | live readback `== desired` | `/runtime/config` | `test_vllm_adapter.py` |
| 3 | `long_prefill_token_threshold` | EngineCore-only scheduler policy (read live each `Scheduler.schedule()`) | Persistent | Yes | `LONG_PREFILL_TOKEN_THRESHOLD` (max prefill tokens for a single long prompt per step) | engine-local prefill | int, min 0 | HOT | — | live readback `== desired` | `/runtime/config` | `test_vllm_adapter.py`, `test_vllm_integration.py` |
| 4 | `runtime_kv_watermark` | Runtime KV admission headroom: `watermark_blocks = int(v * num_gpu_blocks)`; physical capacity unchanged | Persistent | Yes | `KV_ADMISSION_WATERMARK` (fractional [0.0, 1.0) headroom reserve) | engine-local KV admission | float, 0.0 ≤ v < 1.0 | HOT | — | live readback `== desired` | `/runtime/config` | `test_vllm_adapter.py` |
| 5 | `runtime_reserve_full_isl` | Runtime long-request admission gate (full ISL vs first-chunk) | Persistent | Yes | `RESERVE_FULL_ISL` (bool: admit only when FULL input seq fits KV) | engine-local admission | bool | HOT | — | live readback `== desired` | `/runtime/config` | `test_vllm_adapter.py` |
| 6 | `reset_prefix_cache` | `pause_generation(mode=wait)` → quiescence → reset → resume (DRAIN lifecycle) | One-shot operation | Yes | `PREFIX_CACHE_RESET` | engine-local prefix cache | int (sentinel 1) | DRAIN | `WAIT_INFLIGHT` / `DRAIN_TO_IDLE` / `requires_quiescence=True` | `get_runtime_cache_stats()["prefix_cache_entries"] == 0` | `/runtime/config` PATCH (operation) | `test_vllm_adapter.py` |

---

## 3. SGLang v0.5.18 capabilities

| # | Internal key | Engine mechanism | Persistent / One-shot | Supported | Exact semantic | Scope | Runtime-derived bounds | Transition | Quiescence requirement/strategy | Verification | Existing API exposure | Existing tests |
|---|--------------|------------------|-----------------------|-----------|----------------|-------|------------------------|------------|----------------------------------|--------------|------------------------|----------------|
| 7 | `runtime_max_running_seqs` | `pp_max_micro_batch_size` (native `set_internal_state`; admission `min(pp_max_micro_batch_size - running_bs, req_to_token_pool.available_size())`) | Persistent | Yes | `CONCURRENCY_UPPER_BOUND` (NOT exact N-request; NOT fleet-wide) | `PER_DP_RANK` / local scheduler | int, min 1, max provisioned `max_running_requests` | HOT | — | LIVE readback `get_policy()` from `/server_info` `== desired` | `/runtime/config` | `test_sglang_adapter.py`, `test_sglang_integration.py` |
| 8 | `runtime_max_prefill_tokens` | `runtime_max_prefill_tokens` scheduler-local field (canonical patch; `PrefillAdder` admission budget) | Persistent | Yes | `AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED` (NOT strict per-request cap, NOT total scheduler-token budget, NOT vLLM `runtime_max_scheduled_tokens`) | `LOCAL_SCHEDULER` | int, min 1, max provisioned `max_prefill_tokens` | HOT | — | LIVE readback via `/server_info -> internal_states[0]` `== desired` (no shadow state) | `/runtime/config` | `test_sglang_adapter.py`, `test_sglang_integration.py` |
| 9 | `engine_cache_flush` | `flush_cache()` (HTTP 200=flushed / 400=not idle) | One-shot operation | Yes | `ENGINE_CACHE_FLUSH` (broader than prefix reset: radix cache, req→token, KV allocator, grammar, metrics) | engine (scheduler) | int (sentinel 1) | DRAIN | `REQUIRE_IDLE` (engine must ALREADY be idle; busy → BLOCKED, **no** retract side effects); no `quiescence_strategy` | `flush_cache_ok()` (HTTP 200 confirmation) | `/runtime/config` PATCH (operation) | `test_sglang_adapter.py`, `test_sglang_integration.py` |
| 10 | `prefix_cache_reset` | none (must NOT be conflated with `flush_cache()`) | One-shot operation | **No** (advertised, rejected at planning) | `PREFIX_CACHE_RESET` | — | int (sentinel 1) | DRAIN | `REQUIRE_IDLE` / `requires_quiescence=True` | — | discoverable, never executable | `test_sglang_adapter.py` |

---

## 4. Proven SGLang operating envelope (must not be implied broader)

SGLang capabilities are proven within: **SGLang v0.5.18 (patched), PP=1, DP=1,
text/token input, no speculative decoding, no disaggregated prefill/decode**.
V2 must not imply global fleet concurrency or exact N-request execution, nor
suggest these capabilities extend beyond this envelope.

---

## 5. Token-budget family — DISTINCT semantics (never merged)

The two token budgets are **not** equivalent and must not be silently translated
into each other:

| Key | Engine | Exact semantic | What it is NOT |
|-----|--------|----------------|----------------|
| `runtime_max_scheduled_tokens` | vLLM | strict per-step total scheduled-token budget | not the SGLang prefill-admission budget |
| `runtime_max_prefill_tokens` | SGLang | aggregate prefill input-token admission budget; first request of a scheduling pass unbounded | not a strict per-request cap, not a total scheduler-token budget, not vLLM's strict budget |

They may share a **discovery/UI family** (`work_budget`) for grouping, but must
retain distinct exact semantic identities. A request for a strict total
scheduled-token budget against SGLang must be **rejected**, not translated into
the weaker prefill-admission budget.

---

## 6. V2 union capability set (target, used by the preservation regression test)

Exposed operator intents (union across engines, with the common intent added):

1. `runtime_concurrency` → common intent → `runtime_max_running_seqs`
2. `runtime_max_running_seqs` (vLLM + SGLang)
3. `runtime_max_scheduled_tokens` (vLLM)
4. `long_prefill_token_threshold` (vLLM)
5. `runtime_kv_watermark` (vLLM)
6. `runtime_reserve_full_isl` (vLLM)
7. `reset_prefix_cache` (vLLM, operation)
8. `runtime_max_prefill_tokens` (SGLang)
9. `engine_cache_flush` (SGLang, operation)
10. `prefix_cache_reset` (SGLang, unsupported)

Persisted Desired State keys are **unchanged**: the V2 alias `runtime_concurrency`
resolves at the operator/API boundary to the existing internal key
`runtime_max_running_seqs`; it is never stored as `runtime_concurrency`. All five
vLLM V1 keys (`runtime_max_running_seqs`, `runtime_max_scheduled_tokens`,
`long_prefill_token_threshold`, `runtime_kv_watermark`,
`runtime_reserve_full_isl`) continue to work verbatim.