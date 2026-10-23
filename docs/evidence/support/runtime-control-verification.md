# Runtime control verification and evidence levels

Checked against the supplied InfEngine source on 16 October 2026. Paths below are relative to the InfEngine area, `areas/50_inference-engines`. Source and preserved run records were read only. No live inference or GPU experiment was run for this integration.

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

## Source references

- `docs/CONTROL_REFERENCE.md`: seven HOT settings, bounds, native mappings, expected effects, inactive conditions and transport-specific operations.
- `notes/M4A_SGLANG_CAPABILITY_MAPPING.md`: historical recommendation to reuse native concurrency and add the smallest prefill ceiling; pipeline caveats; discussion of chunk size and missing cross-engine equivalents. Its proposed retraction/cache sequence is a historical proposal, not current adapter support.
- `docs/M4B_C_CONTROL_CONTRACT_V2.md`, especially sections 3–6: aliases, semantic normalization, distinct token budgets, persistent settings versus operations.
- `docs/M11_RUNTIME_CONTROL_ACCEPTANCE.md`, phases 1–5 and limitations: every HOT setting individually accepted; invalid-type/range and stale-revision cases; DRAIN/cache tests excluded; local PP=1/DP=1 scope.
- `docs/M13_PILOT_READY_INFENGINE.md`, sections 2–4 and 11: whole-policy prevalidation, native multi-control acceptance, compensation and non-atomic boundaries.
- `src/infengine/control/contract.py`: actual operator intent registry and alias resolution.
- `src/infengine/vllm/adapter.py`: capability declarations, live bounds, prefill-step description, type coercion and supported-operation filtering.
- `contrib/vllm/v0.28.0_runtime_scheduler_policy.patch`: native allowlist and exact scheduler/config writes; strict native types; provisioning checks inside the mutating phase.
- `src/infengine/sglang/adapter.py`, `runtime_client.py`, `validation.py`: strict integer validation, per-rank semantics, native update and readback paths, idle-only cache flush.
- `contrib/sglang/v0.5.18_runtime_prefill_budget.patch`: added scheduler field, live readback, setter validation and prefill-adder input.

## Selection rationale and limitations

The source explicitly documents scheduler-only vLLM mutation and the smallest initial SGLang hooks. The article's broader criteria—useful operational question, bounded native mutation, observability, testability and avoiding reprovisioning—are engineering synthesis of these records, not an invented historical decision matrix.

Tensor parallelism, context capacity, memory-pool resizing, graph provisioning and distributed reconfiguration have no validated HOT path in this integration. SGLang chunk size was discussed as a possible extension, not shipped as an InfEngine HOT setting. These scope boundaries do not prove that every excluded setting is universally impossible to change at runtime.

Cache operations are one-shot, with separate transition and quiescence requirements. vLLM's reset is unavailable through the tested HTTP control transport. SGLang engine cache flush requires existing idleness and returns BLOCKED if busy; the separate `prefix_cache_reset` intent is unsupported. Do not infer cache-operation acceptance from the seven individual HOT tests.

The vLLM adapter's numeric coercion and native hook's strict type checks are different layers. The article recommends correctly typed policy values and does not claim that every interface rejects every numeric string. The watermark's exclusive upper bound is enforced in the adapter and native hook.

Implementation, live verification, measured performance effect and workload-specific benefit are four distinct evidence levels. The initial seven controls do not constitute an architectural ceiling, but no additional actuator is claimed implemented or supported.
