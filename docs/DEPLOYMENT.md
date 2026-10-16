# InfEngine Deployment

## Supported deployment path (vLLM)

The pilot path is a versioned container that bakes in the exact supported vLLM
version, the InfEngine runtime hook and the control listener, so the operator
never runs `apply_hook.py` by hand.

```bash
# from the repository root (build context = repo root)
docker build -f contrib/vllm/Dockerfile.infengine \
  -t infengine-vllm:0.1.0-vllm0.28.0 .
```

Conceptual tag: `infengine-vllm:<infengine-version>-vllm0.28.0`.

Run it (engine + control listener in one process; publish the control listener
to host loopback only):

```bash
docker run -d --name infengine_vllm --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 -p 127.0.0.1:9001:9001 \
  -e INFENGINE_CONTROL_PORT=9001 \
  -e INFENGINE_CONTROL_TOKEN=<strong-token> \
  -v infengine_hf:/root/.cache/huggingface \
  infengine-vllm:0.1.0-vllm0.28.0 \
  --model Qwen/Qwen2.5-0.5B-Instruct --port 8000 \
  --enforce-eager --max-model-len 2048 \
  --gpu-memory-utilization 0.55 --max-num-seqs 64
```

(`--enforce-eager` and a lower `--gpu-memory-utilization` are required on a
4 GiB Turing GPU; vLLM fails closed with an explicit message if the requested
fraction exceeds free memory.)

Then run the InfEngine control plane on the host (use a DIFFERENT port from the
engine's 8000):

```bash
export INFENGINE_VLLM_CONTROL_URL=http://127.0.0.1:9001
export INFENGINE_VLLM_CONTROL_TOKEN=<strong-token>
export INFENGINE_AUTH_TOKENS='{"<operator-token>":{"actor":"you","role":"OPERATOR"}}'
infengine serve --runtime vllm \
  --vllm-control-url http://127.0.0.1:9001 \
  --state-file ./desired.json --audit-file ./audit.db \
  --host 127.0.0.1 --port 8010
```

The development `contrib/vllm/Dockerfile` (which runs `apply_hook.py` and a dev
script) is retained for development only.

## SGLang

SGLang keeps its own native deployment semantics; do not force vLLM packaging
assumptions onto it. Build the patched image (bakes in the
`runtime_max_prefill_tokens` scheduler knob):

```bash
docker build -f contrib/sglang/Dockerfile.infengine \
  -t infengine-sglang:0.1.0-sglang0.5.18 .
docker run -d --name infengine_sglang --gpus all --ipc=host \
  -p 127.0.0.1:30000:30000 -v infengine_hf:/root/.cache/huggingface \
  infengine-sglang:0.1.0-sglang0.5.18 \
  python3 -m sglang.launch_server --model-path Qwen/Qwen2.5-0.5B-Instruct \
    --host 0.0.0.0 --port 30000 --mem-fraction-static 0.8 \
    --max-running-requests 64 --attention-backend torch_native --disable-cuda-graph
```

Then point InfEngine at it:

```bash
export SGLANG_BASE_URL=http://127.0.0.1:30000
export INFENGINE_AUTH_TOKENS='{"<operator-token>":{"actor":"you","role":"OPERATOR"}}'
infengine serve --runtime sglang --runtime-url http://127.0.0.1:30000 \
  --audit-file ./audit.db --host 127.0.0.1 --port 8020
```

Notes for a 4 GiB GPU: lower `--mem-fraction-static` if another process holds
GPU memory; the engine fails closed with an explicit minimum if it is too low.
SGLang latency histograms require `--enable-metrics`; without it, scheduler
gauges come from the always-on `/get_load` and driver-derived throughput is
still available.

## Ports

| Port | Service | Exposure |
|---|---|---|
| 8000 | vLLM OpenAI API | as needed |
| 9001 | vLLM InfEngine control listener | **loopback only** |
| 8010 | InfEngine control plane (vLLM) | loopback or authenticated |
| 8020 | InfEngine control plane (SGLang) | loopback or authenticated |
| 30000 | SGLang admin/inference | as needed |

## Security

- The vLLM control listener requires `INFENGINE_CONTROL_TOKEN` (env/secret).
- The InfEngine control plane authenticates mutations with bearer tokens and a
  deterministic role model: `VIEWER`, `OPERATOR`, `APPROVER`.
- Configure tokens out-of-band:

  ```bash
  export INFENGINE_AUTH_TOKENS='{"<token>":{"actor":"name","role":"OPERATOR"}}'
  # or
  export INFENGINE_AUTH_FILE=/run/secrets/infengine-tokens.json
  ```

- **Actor identity is derived from the authenticated bearer token.** The
  `INFENGINE_ACTOR` environment variable / `X-InfEngine-Actor` header is
  advisory client metadata ONLY and is ignored when authentication is enabled;
  it is recorded only when authentication is disabled (local development).
- With authentication disabled, the service refuses to bind a non-loopback host
  unless `--insecure` is passed, and prints a prominent warning.
- Configured secrets are excluded from normal product reports/history and
  `doctor` output. Environment values remain visible to privileged host users;
  use a secret manager and avoid credentials in shell command arguments.

## Version requirements and compatibility

- vLLM **0.28.0** with `v0.28.0_runtime_scheduler_policy.patch` (or the product
  image). Other versions are not claimed.
- SGLang **0.5.18** with `v0.5.18_runtime_prefill_budget.patch`.
- On an unsupported/unreachable engine the control plane **fails closed**:
  `infengine serve` refuses to start and `infengine doctor` reports not ready.
  There is no silent fallback to another backend.

## Upgrade considerations

- Pin the exact image tag; the compatibility matrix lists tested combinations.
- `desired.json` and `audit.db` are independently versioned; keep `audit.db`
  for restore history across restarts.
- Re-run `infengine doctor` after any engine/image change.

## Local vs remote

- Local development: `--host 127.0.0.1`, authentication optional.
- Remote/pilot: always enable authentication and terminate TLS in front of the
  control plane; expose the vLLM control listener only on loopback.
