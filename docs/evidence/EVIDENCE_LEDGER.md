# Evidence ledger

This ledger maps the experimental claims to their supporting measurements. Source paths and hashes are recorded in [PROVENANCE.json](PROVENANCE.json). Implementation, native acceptance, measured effect and operational benefit are different levels of evidence.

| Claim | Evidence | What is established / limit |
|---|---|---|
| Five vLLM and two SGLang HOT controls | [Control reference](../CONTROL_REFERENCE.md), [functional extracts](support/runtime-control-verification.md) | Documented native functional acceptance on pinned patched versions. No claim of A100 performance for all controls. |
| Live concurrency change and restoration | [Smoke capture](inputs/m12_a100_validation/deployment/smoke/smoke.json) | Native 64→16→64, verified convergence, completion payloads, API PID 3695 unchanged. No separate EngineCore PID or explicit HTTP status line. |
| Initial pilot load response | [Pilot CSV](data/pilot_profiles.csv), [captures](inputs/m12_a100_validation/pilot/) | Fifteen median points; two repeats and 15-second windows. Does not establish saturation. |
| Initial A/B comparison was confounded | [Cache-order CSV](data/cache_confound.csv), per-arm `cache_counters.json` | Eighteen before/after token counter deltas confirm asymmetric cache warmth across nine pairs. Not clean performance effects. |
| Selected controls change performance | [48 arm records](data/ab_per_arm.csv), matching compact results | Workload-specific throughput and latency changes. Three pairs per condition; limited loads/model/hardware. |
| Paired A/B ratios need correction | [Reconciliation](support/ab-ratio-reconciliation.json) | Recomputed median of three B/A ratios; original summaries preserved. High-load concurrency 0.736535, not inherited 0.709. |
| Concurrency and token budget interact | [12 per-block runs](data/2x2_per_block.csv), [summary](data/2x2_summary.json) | +0.505588 req/s descriptive interaction. 17.19% of grand mean; 18.57% of A. `interaction_pct` uses A. |
| Lower budget changes TTFT | Same factorial captures | Means of block p95 estimates ~19–19.5 vs 39 seconds, not pooled percentiles. |
| Native latency-window distributions | [Compact pair-1 captures](inputs/m12_a100_validation/runtime_concurrency/), [figure manifest](FIGURE_MANIFEST.md) | Adjacent differences of cumulative buckets; final 5-second windows and small samples. |
| SGLang multi-control evaluation | [Evaluator extract](support/sglang-evaluator-results.md), [policy](support/sglang-evaluation-policy.yaml) | Functional evaluation/restoration context. Missing metrics or `NO_MEASURABLE_DIFFERENCE` do not prove equivalence. |
| Architecture and control workflows | [Nine diagrams](DIAGRAMS.md) | Design/engineering explanations; diagrams are not additional benchmark observations. |

## Provenance and scope

The original measured InfEngine revision is **unknown**. Engine/model versions, workload definitions and captured configuration are available, but this is not a bit-for-bit reconstruction of the original serving environment. The current software release's 619 local passing tests are a separate historical validation result, not new GPU evidence.

Measurement data and experimental figures are preserved as historical artifacts. Analysis checks numeric measurements against captures; it does not endorse every sentence in older embedded metadata, such as claims of a "real" interaction or percentage labels with an unstated denominator. Prefer the explicitly defined estimands in the case study and reconciliation.
