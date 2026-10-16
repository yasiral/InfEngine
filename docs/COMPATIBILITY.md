# InfEngine Compatibility

Truthful matrix. Only tested combinations are claimed. "LIVE VERIFIED" means the
control was exercised against a real engine and its raw readback matched
(verified on the M11/M12 local A100/T2000 environment; re-verify per deployment
with `infengine doctor` and `infengine evaluate`).

| Engine | Version | Status | Supported HOT controls | Transport | Live verified |
|---|---|---|---|---|---|
| vLLM | 0.28.0 (patched) | supported | `runtime_concurrency`, `long_prefill_token_threshold`, `runtime_max_scheduled_tokens`, `runtime_kv_watermark`, `runtime_reserve_full_isl` | HTTP control listener (`/infengine/control/*`), token-authenticated | Yes (M11/M12) |
| SGLang | 0.5.18 (patched) | supported | `runtime_concurrency`, `runtime_max_prefill_tokens` | native admin HTTP (`/set_internal_state`, `/server_info`, `/get_load`) | Yes (M11) |
| built-in LLMEngine | in-repo | development | `scheduler.max_batch`, `scheduler.chunk_size`, `kv_store.block_size`, `kv_store.num_blocks` | in-process | Yes (unit/integration) |

## One-shot operations

| Engine | Operation | Status | Requirement |
|---|---|---|---|
| vLLM | `reset_prefix_cache` | supported (in-process transport only) | DRAIN_TO_IDLE |
| SGLang | `engine_cache_flush` | supported | REQUIRE_IDLE |
| SGLang | `prefix_cache_reset` | advertised, unsupported | — |

## Not claimed

- Any vLLM version other than 0.28.0.
- Any SGLang version other than 0.5.18.
- Cross-engine semantic equivalence of token budgets.
- Engine atomicity of multi-control apply (see `apply_semantics`).

## Apply semantics by engine

| Engine | `apply_semantics` | Live confirmed |
|---|---|---|
| vLLM | `CONTROL_PLANE_TRANSACTIONAL` | yes (0.28.0) |
| SGLang | `COORDINATED_NON_ATOMIC` | yes (0.5.18) |
| built-in | `COORDINATED_NON_ATOMIC` | unit/integration |

`infengine doctor` and `infengine inspect` surface the live value.

## Product images (built and validated locally)

| Image tag | Base | Contents |
|---|---|---|
| `infengine-vllm:0.1.0-vllm0.28.0` | `vllm/vllm-openai:v0.28.0` | baked-in scheduler-config hook + control listener |
| `infengine-sglang:0.1.0-sglang0.5.18` | `lmsysorg/sglang:v0.5.18` | `runtime_max_prefill_tokens` scheduler patch |

## Live acceptance performed (local T2000, 4 GiB)

- vLLM 0.28.0: 2-control policy (`runtime_concurrency`,
  `runtime_max_scheduled_tokens`) and 5-control policy applied in one operation,
  raw readback verified, inference served, EngineCore PID unchanged, restore
  verified.
- SGLang 0.5.18: 2-control policy (`runtime_concurrency`,
  `runtime_max_prefill_tokens`) applied, raw `internal_states` readback verified,
  inference served, restore verified.
- Evaluator: one single-control and one two-control (interleaved) experiment via
  `infengine evaluate`; restore verified; HTML reports generated.

## Human approval

- Single-control M7A observe→approve: unchanged.
- Multi-control policy proposal/approval (`SupervisorPolicyApplication`): the
  same M7 lifecycle extended to a whole policy map (single-use, TTL, If-Match,
  fresh-evidence, whole-policy verification).
