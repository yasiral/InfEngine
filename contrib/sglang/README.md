# SGLang v0.5.18 runtime prefill admission budget (Milestone 4B-B)

Live proof that the control plane can mutate two genuine SGLang v0.5.18 runtime
scheduler knobs over the native HTTP admin surface, without restarting the
Scheduler or GPU workers.

## What is here

- `v0.5.18_runtime_prefill_budget.patch` — the **canonical, upstream-quality
  SGLang change** (generated from the actual pristine-vs-patched diff; verified
  to apply cleanly against the pristine v0.5.18 tag, commit `71de97b`). Adds the
  scheduler-local `runtime_max_prefill_tokens` runtime knob with **strict**
  type validation.

  The patch implements:
  - `Scheduler.runtime_max_prefill_tokens` (a scheduler-local runtime field),
    initialized from the boot provisioned `max_prefill_tokens`.
  - `set_internal_state` allowlist entry + **strict integer validation**
    (rejects `str` / `float` / `None` / `bool` before any comparison — no
    `TypeError` crash, no silent `int(2.5) -> 2` coercion; rejects below 1 and
    above the boot ceiling).
  - `PrefillAdder` receives `runtime_max_prefill_tokens` instead of the boot
    `max_prefill_tokens`.
  - `Scheduler.get_internal_state` exposes `runtime_max_prefill_tokens`, so the
    control plane can READ the ACTUAL scheduler value through
    `/server_info -> internal_states[rank]` (no shadow state).

  The runtime field is scheduler-local: it never reaches worker provisioning,
  KV allocation, CUDA-graph sizing, or request-pool sizing. The boot
  `max_prefill_tokens` stays the immutable provisioned ceiling.

Apply the canonical patch to a pristine v0.5.18 checkout:

```bash
git clone --depth 1 --branch v0.5.18 https://github.com/sgl-project/sglang
cd sglang
git apply /path/to/contrib/sglang/v0.5.18_runtime_prefill_budget.patch
```

## Exact semantics (source-backed)

`runtime_max_prefill_tokens` is the **aggregate prefill input-token admission
budget** consumed by `PrefillAdder.rem_input_tokens`:

- the FIRST request of a scheduling pass is unbounded by it
  (`schedule_policy.add_one_req` gates only when `can_run_list` is non-empty);
- it is NOT a strict per-request token cap;
- it is NOT a total scheduler-token budget (decode tokens / `max_new_tokens`
  are tracked separately in `rem_total_tokens`);
- it is NOT an exact equivalent of vLLM's `runtime_max_scheduled_tokens`.

## Runtime concurrency knob (no patch needed)

`pp_max_micro_batch_size` is natively runtime-settable via
`set_internal_state`. It is a **concurrency upper bound** scoped per DP rank /
local scheduler: admission is `min(pp_max_micro_batch_size - running_bs,
req_to_token_pool.available_size())`. Bounded by the boot provisioned
`max_running_requests`; HOT at PP=1.

## Environment / hardware notes

- GPU: Quadro T2000 with Max-Q (SM 7.5, 4 GiB). SGLang auto-falls back to fp16
  below SM80 and uses `torch_native` attention on Turing (FlashInfer JIT fails
  on SM75 for small head dims); `--cuda-graph-backend-decode disabled` avoids
  CUDA-graph capture.
- Supported envelope (M4B-B): SGLang v0.5.18, PP=1, DP=1, text/token input, no
  speculative decoding, no disaggregated prefill/decode.

## ENGINE_CACHE_FLUSH quiescence

`engine_cache_flush` (SGLang `flush_cache()`) requires the engine to ALREADY be
idle. SGLang's `flush_cache()` refuses to run while running OR waiting requests
exist, and `pause_generation(mode="retract")` does NOT make a busy engine
flushable (it parks requests in the waiting queue). The adapter models this as
`quiescence_requirement = REQUIRE_IDLE`: a busy engine is BLOCKED with no retract
side effects; an idle engine is flushed directly.