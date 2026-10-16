# InfEngine Control Reference

Every supported control, its native engine mapping, semantics, transition class,
bounds, verification source, expected areas of effect, conditions where it is
inert, engine differences and known limitations.

`infengine explain <control>` renders the same knowledge plus the live current
value and bounds. Nothing here is an LLM call; it is deterministic metadata.

---

## vLLM

### runtime_concurrency
- **Native mapping:** EngineCore `scheduler.max_num_running_reqs`.
- **Semantics:** `CONCURRENCY_UPPER_BOUND` — a cap on concurrently running
  requests, not an exact-N guarantee.
- **Transition:** HOT.
- **Bounds:** `[1, provisioned max_num_seqs]`.
- **Verification:** engine readback of the live running bound equals the request.
- **Expected effects:** throughput and queueing pressure.
- **Inert when:** offered load is below the bound, or KV capacity is the limit.
- **Limitations:** bounded by the boot-time `max_num_seqs`; does not resize
  worker buffers/KV/CUDA graphs.

### long_prefill_token_threshold
- **Native mapping:** `scheduler_config.long_prefill_token_threshold`.
- **Semantics:** `LONG_PREFILL_TOKEN_THRESHOLD`.
- **Transition:** HOT.
- **Bounds:** `[0, ∞)`.
- **Verification:** readback equals the request.
- **Expected effects:** how a single long prompt's prefill is chunked per step.
- **Inert when:** prompts are short relative to the threshold.
- **Limitations:** vLLM only.

### runtime_max_scheduled_tokens
- **Native mapping:** EngineCore `scheduler.max_num_scheduled_tokens`.
- **Semantics:** `STRICT_SCHEDULED_TOKEN_BUDGET` (per step, total).
- **Transition:** HOT.
- **Bounds:** `[1, provisioned max_num_scheduled_tokens]`.
- **Verification:** readback equals the request.
- **Expected effects:** TTFT, admission and throughput; may raise per-step cost.
- **Inert when:** the workload never approaches the per-step budget.
- **Limitations:** **vLLM only**. NOT the SGLang prefill-admission budget and
  never translated to it. Unlike SGLang's control, it is a strict total budget.

### runtime_kv_watermark
- **Native mapping:** `scheduler.kv_cache_manager.watermark_blocks`
  (`int(v * num_gpu_blocks)`).
- **Semantics:** `KV_ADMISSION_WATERMARK`.
- **Transition:** HOT.
- **Bounds:** `[0.0, 1.0)` (must be `< 1.0`).
- **Verification:** readback equals the request.
- **Expected effects:** reservation of KV headroom; can reduce preemptions.
- **Inert when:** KV usage never approaches the watermark.
- **Limitations:** physical KV capacity is unchanged.

### runtime_reserve_full_isl
- **Native mapping:** `scheduler.scheduler_reserve_full_isl` (bool).
- **Semantics:** `RESERVE_FULL_ISL`.
- **Transition:** HOT.
- **Bounds:** true/false.
- **Verification:** readback equals the request.
- **Expected effects:** whether long requests are admitted only when their full
  input fits KV, or chunked-prefill admitted.
- **Inert when:** all inputs fit KV regardless.
- **Limitations:** affects future admissions only.

---

## SGLang

### runtime_concurrency
- **Native mapping:** `pp_max_micro_batch_size` (per DP rank).
- **Semantics:** `CONCURRENCY_UPPER_BOUND`, scope `PER_DP_RANK`.
- **Transition:** HOT.
- **Bounds:** `[1, provisioned max_running_requests]`.
- **Verification:** scheduler readback via `/server_info → internal_states`.
- **Expected effects:** concurrency and queueing.
- **Inert when:** load is below the bound.
- **Limitations:** admission is also constrained by available request-token pool
  slots; a per-rank bound is not a cluster-wide guarantee.

### runtime_max_prefill_tokens
- **Native mapping:** scheduler-local `runtime_max_prefill_tokens` (canonical
  patch; exposed through `/server_info → internal_states`).
- **Semantics:**
  `AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED`.
- **Transition:** HOT.
- **Bounds:** `[1, provisioned max_prefill_tokens]`.
- **Verification:** scheduler readback equals the request.
- **Expected effects:** aggregate prefill work per scheduling pass; TTFT/queue.
- **Inert when:** one request dominates each pass (the first request is
  unbounded by this control).
- **Limitations:** **SGLang only**; NOT a strict per-request cap, NOT a total
  scheduler-token budget, and NOT equivalent to vLLM's
  `runtime_max_scheduled_tokens`.

---

## One-shot operations (not persistent policy)

- **vLLM `reset_prefix_cache`** — DRAIN lifecycle
  (`pause_generation(mode=wait)` → quiescence → reset → resume). Not persisted in
  desired state. Not exposed over the HTTP control transport.
- **SGLang `engine_cache_flush`** — `flush_cache()`; requires an already-idle
  engine (`REQUIRE_IDLE`). A busy engine is `BLOCKED`, with no retract side
  effects. `prefix_cache_reset` is advertised but unsupported on SGLang.

## Apply semantics

Multi-control policies are validated as a whole before any mutation. The
effective guarantee is reported per deployment:

| Engine | `apply_semantics` | Meaning |
|---|---|---|
| vLLM | `CONTROL_PLANE_TRANSACTIONAL` | One coordinated request; verify all; verified rollback on failure |
| SGLang | `COORDINATED_NON_ATOMIC` | Sequential per-field requests; verify all; best-effort rollback on failure |
| builtin | `COORDINATED_NON_ATOMIC` | Sequential in-process applies; best-effort rollback on failure |

No engine is claimed to be `ENGINE_ATOMIC`.
