> This document retains the original single-control integration example. The public product supports five verified HOT settings; use the root README and `docs/DEPLOYMENT.md` for the current multi-control, authenticated pilot path.

# vLLM v0.28.0 runtime scheduler-config control (Milestone 3B)

Live proof that the control plane can mutate one genuine vLLM V1 engine-level
scheduler policy (`long_prefill_token_threshold`) at runtime, over transport,
without restarting EngineCore or GPU workers.

## What is here

- `v0.28.0_runtime_scheduler_policy.patch` — the **canonical, upstream-quality vLLM
  change** (generated from the actual pristine-vs-patched diff; verified to apply
  cleanly and reproduce the patched files exactly). Implements the runtime
  scheduler-policy API: `EngineCore.set_runtime_scheduler_config` /
  `get_runtime_scheduler_config` (allowlisted: `long_prefill_token_threshold`,
  `runtime_max_running_seqs`, `runtime_max_scheduled_tokens`, `runtime_kv_watermark`,
  `runtime_reserve_full_isl`, bounded by
  provisioned boot capacity), `get_runtime_cache_stats` (prefix-cache observable),
  plus the full client transport and `LLMEngine`/`AsyncLLM` exposure.
- `apply_hook.py` — reproducible local application mechanism for the integration
  image; it injects the same changes into the installed wheel. Retained for the
  Docker integration environment; the `.patch` is the canonical implementation.
- `Dockerfile` — isolated vLLM v0.28.0 runtime image with the hook applied and
  the control-plane package installed.
- `experiment.py` — live M3D experiment.

Apply the canonical patch to a vLLM v0.28.0 checkout:

```bash
git clone --depth 1 --branch v0.28.0 https://github.com/vllm-project/vllm
cd vllm
git apply /path/to/contrib/vllm/v0.28.0_runtime_scheduler_policy.patch
```

## Build

```bash
# from the repository root (build context = repo root)
docker build -f contrib/vllm/Dockerfile -t infengine-vllm .
```

## Run the live experiment

```bash
docker run --rm --gpus all --ipc=host \
  -v "$(pwd)/hf_cache:/root/.cache/huggingface" \
  infengine-vllm python3 /workspace/experiment.py
```

The image bakes in `VLLM_WSL2_ENABLE_PIN_MEMORY=1`, which is **required under
WSL2** (vLLM gates UVA/pinned memory behind it; without it the engine core
fails with `RuntimeError: UVA is not available`).

## Environment / hardware notes

- GPU: Quadro T2000 with Max-Q (SM 7.5, 4 GiB) — within vLLM's documented
  minimum (compute capability >= 7.5).
- On SM 7.5 vLLM auto-falls back where needed (e.g. FlashInfer top-p/top-k
  sampler unavailable -> falls back). `enforce_eager=True` avoids CUDA-graph
  capture on Turing.
- `enable_prefix_caching=True` is the V1 default, so the behavioral proof uses
  **two distinct prompts** (an identical second prompt would hit the prefix
  cache and mask the threshold effect).

## What the experiment proves

1. `observe() -> 0`, reconcile `{long_prefill_token_threshold: 128}` ->
   `VERIFIED/CONVERGED`, observe -> 128; reconcile `{...: 32}` -> 32.
2. EngineCore process (and single-GPU worker, which runs inside it) PID is
   unchanged before/after (`engine pids: [84] == [84]`).
3. Scheduler behavioral consumption: a 400-token prompt prefills in chunks of
   128 (128/128/128/16) at threshold 128, and in chunks of 32 (32x12 + 16) at
   threshold 32, per the EngineCore iteration-details log.
