# Runtime control verification and evidence levels

The tables below summarize native control mappings, documented functional acceptance and the limits of the available performance evidence.

## Native mappings and acceptance

| Operator name | Native mapping | Contract and acceptance |
|---|---|---|
| vLLM `runtime_concurrency` | Internal alias `runtime_max_running_seqs`; `scheduler.max_num_running_reqs` | HOT integer 1 through provisioned `max_num_seqs`; individual local 8→4→8 acceptance; A100 A/B and factorial |
| vLLM `runtime_max_scheduled_tokens` | `scheduler.max_num_scheduled_tokens` | HOT integer 1 through provisioned scheduled-token ceiling; fallback provisioning field `max_num_batched_tokens`; local 2048→1024→2048; A100 A/B and factorial |
| vLLM `long_prefill_token_threshold` | `scheduler_config.long_prefill_token_threshold` | HOT nonnegative integer; positive per-prompt prefill-step restriction; zero disables that restriction; local 0→128→0; one A100 A/B comparison |
| vLLM `runtime_kv_watermark` | Fraction retained in scheduler state; `scheduler.kv_cache_manager.watermark_blocks = int(fraction * num_gpu_blocks)` | HOT float in `[0.0,1.0)`; upper bound exclusive despite display metadata showing max 1.0; local 0→0.1→0; no A100 performance characterization |
| vLLM `runtime_reserve_full_isl` | `scheduler.scheduler_reserve_full_isl` | HOT Boolean; full-input versus first-chunk admission reservation; future admissions only; local true→false→true; no A100 performance characterization |
| SGLang `runtime_concurrency` | Internal alias `runtime_max_running_seqs`; `pp_max_micro_batch_size` | HOT integer 1 through provisioned `max_running_requests` for proven PP=1; native general pipeline limit is `max_running_requests // pp_size`; local 8→4→8; no A100 performance study |
| SGLang `runtime_max_prefill_tokens` | Scheduler-local field passed to `PrefillAdder` and exposed through `internal_states` | HOT integer 1 through provisioned `max_prefill_tokens`; aggregate prefill input admission with first-request exception; local 16384→8192→16384; no A100 performance study |

Each local individual acceptance includes native application/readback, successful inference and verified restore. These establish functional behavior, not a performance gain. The later two-control acceptance separately checks product apply/verify/restore workflows. No engine-atomic guarantee is established.

## Implementation references

- [Control reference](../../CONTROL_REFERENCE.md): supported settings, bounds and transport-specific operations.
- [Operator contract](../../../src/infengine/control/contract.py): intent registry and alias resolution.
- [vLLM adapter](../../../src/infengine/vllm/adapter.py) and [native integration](../../../contrib/vllm/): runtime limits, native application and readback.
- [SGLang adapter](../../../src/infengine/sglang/adapter.py) and [native integration](../../../contrib/sglang/): per-rank admission, prefill semantics and idle-only cache flush.

## Selection rationale and limitations

The initial controls focus on bounded scheduling and admission policies with native readback. Useful selection criteria include operational relevance, observability, testability and avoiding reprovisioning. These criteria describe the design rationale; they are not a measured comparison of every possible control.

Tensor parallelism, context capacity, memory-pool resizing, graph provisioning and distributed reconfiguration have no validated HOT path in this integration. SGLang chunk size was discussed as a possible extension, not shipped as an InfEngine HOT setting. These scope boundaries do not prove that every excluded setting is universally impossible to change at runtime.

Cache operations are one-shot, with separate transition and quiescence requirements. vLLM's reset is unavailable through the tested HTTP control transport. SGLang engine cache flush requires existing idleness and returns BLOCKED if busy; the separate `prefix_cache_reset` intent is unsupported. Do not infer cache-operation acceptance from the seven individual HOT tests.

The vLLM adapter's numeric coercion and native hook's strict type checks are different layers. Use correctly typed policy values; not every interface rejects every numeric string. The watermark's exclusive upper bound is enforced in the adapter and native hook.

Implementation, live verification, measured performance effect and workload-specific benefit are four distinct evidence levels. The initial seven controls do not constitute an architectural ceiling, but no additional actuator is claimed implemented or supported.
