# InfEngine source and evidence index

This index records local source paths relative to the supplied InfEngine area, not links to an assumed public checkout. Historical commit IDs come from engineering records; current source HEAD is not asserted. Source inspection and captured acceptance are separate from new live execution. No engine test or GPU experiment was rerun during editorial preparation.

## Engineering records and source paths

* `publication/publication/infengine/01_ENGINEERING_HISTORY.md`: M1–M13 provenance; M1–M3 reconstructed, M8 design-only.
* `publication/publication/infengine/02_EVIDENCE_LEDGER.md`, `05_FACT_CHECK.md`: claim boundaries and checks.
* `src/infengine/control/{state,reconciler,contract,adapter}.py`: desired state, conditional reconciliation, capability and application semantics.
* `src/infengine/supervisor/approval.py`: single-control approval and `SupervisorPolicyApplication` whole-policy specialization.
* `src/infengine/telemetry/collectors.py`, `src/infengine/evaluator.py`, `src/infengine/experiment.py`: quality-aware collectors, temporary evaluation and measurement/accounting.
* `src/infengine/{security,audit,unified}.py`: bearer roles, persistent audit and unified CLI.
* `contrib/vllm/` and `contrib/sglang/`: canonical runtime patches, native control listener and product images.
* `src/infengine/sglang/runtime_client.py`: runtime load client `/get_load`; normalized collector separately uses `/v1/loads?format=prometheus` and `/metrics`.
* `docs/`: milestone records. M11 historical reference `d176f0a`; M12/M13 evidence does not establish current Git HEAD.
* `experiments/m12_a100_validation/`: raw factorial records, pressure pilots and longer L96 confirmation.

## Newly supplied evidence used

`New folder/m12_a100_validation.zip` contains 717 entries supporting A/B, pilots, token-cache fractions, smoke and histogram checks; no raw factorial records. `m12_blog_evidence.zip` retains older tables and experiment descriptions. `M12_A100_WORKLOAD_VALIDATION.md` and `PURISTAN_INFENGINE_A100_TECHNICAL_CASE_STUDY.md` explain experimental progression. M13 functional acceptance uses `M13_PILOT_READY_INFENGINE.md`, `m13_live_acceptance.zip` and its preserved YAML/HTML. `CONTROL_REFERENCE.md`, `COMPATIBILITY.md`, `control-contract-v2-preservation-matrix.md`, `DEPLOYMENT.md`, `QUICKSTART.md`, `RECIPES.md`, `CLI_HELP_TRANSCRIPT.md`, `E2E_OPERATOR_ACCEPTANCE.md` and `INFENGINE_OPERATOR_INTEGRATION.md` ground mappings, deployment and workflow history. Supplementary pitch diagrams were reviewed for framing, not treated as performance evidence.

## Corrections and interpretation boundaries

1. Concurrency cache pair 1 is 0.0313/0.9033; 0.0335/0.8342 belongs to pair 3.
2. Native `le` buckets are cumulative. Disjoint bins are adjacent differences. Concurrency baseline TTFT occupies `(1,2.5]`, not `(2,2.5]`; candidate `(10,20]`. Token E2E windows occupy `(20,30]` in both arms.
3. Absolute factorial interaction is +0.5055881335 requests/s. 17.2% uses the grand mean; 18.5737% uses A. Paired A/B ratios are not ratios of marginal medians.
4. No-error/no-timeout is not zero incomplete work. Final factorial has 162–192 measured completions per run, 62–65 boundary-incomplete and 31–34 drain completions. Native timing sample populations differ.
5. Short L96 load-selection results have unavailable throughput. A separate longer confirmation motivates adequate windows. Missing values are not zero.
6. A100 smoke PID3695 identifies API server; local M13 PID1112 identifies EngineCore.
7. Whole-policy prevalidation and verification do not imply engine atomicity. Compensation is attempted/verified where possible; accepted desired intent can remain after physical rollback.
8. vLLM total-step budget and SGLang first-request-unbounded prefill admission are distinct. Native transport limits remain explicit.
9. M13 evaluator's missing performance metrics preclude equivalence claims. Its original state is restored, not necessarily the experimental baseline.
10. Claim-map confidence is not formal significance; no universal optimum, production certification, autonomous apply/revert or unimplemented M8 performance claim.

## Hashes and staged-copy provenance

The nine data files remain byte-identical to the supplied publication assets. SVGs are rebuilt editorial copies, with corrected histograms, readable labels and technical titles. Mermaid content is directly embedded in the QMD, with staged corrected sources for inspection. Original source/evidence hashes were checked before and after editorial work.

| Asset | SHA-256 of staged file | Treatment |
|---|---|---|
| `data/2x2_per_block.csv` | `ea590bca7acbd0978af68e4e70cd58fd48d82d47a96f8823836d0057c05c8798` | Original bytes preserved |
| `data/2x2_summary.csv` | `e0bea5b001900d627a4711e7f7b84085e6660ff576f3d9ca47ff06fd624ecc8d` | Original bytes preserved |
| `data/2x2_summary.json` | `8b7717a06563427c5dbe1663c6b6e3a5b6c018c1cb8630cb3498509c07289a03` | Original bytes preserved |
| `data/ab_medians.csv` | `1c27fb373aae022f39274196eb7b13ad364ac001aaf51c6670e8cb40c62b2b54` | Original bytes preserved |
| `data/ab_per_arm.csv` | `54c48fc3d928e8ae77d29e94d1fef52ecc32c3d87cb33ed703b4c5ec2807b680` | Original bytes preserved |
| `data/cache_confound.csv` | `7bc5619c4cde8ba7623f3e63722fd1905770f90029adccd8c05956209a29bd61` | Original bytes preserved |
| `data/claims_evidence_map.md` | `cf71e8000f06daf6327325d6fdf8c38a68c35bdb293c9d9083642f6f9e3ad8d8` | Original bytes preserved |
| `data/pilot_prefill_m12_1.json` | `49594bc7bec37d85fbef0cc907103e1475b7bc2ba4b53010930d9608a55c353b` | Original bytes preserved |
| `data/pilot_profiles.csv` | `bbd5e4331fd54efe6c4b1cebf1391119ee7cf802cd9071689581b797f538e494` | Original bytes preserved |
| `figures/fig1_throughput_vs_concurrency.svg` | `7d0a1d58e51e720a896e443e1b712bf4cca16dbc9cdfcbcf39ef2fff4e2bc742` | Editorial corrected copy |
| `figures/fig2_ttft_p95_vs_concurrency.svg` | `34e27232e77da3433f468af58674f7a98a879f90153b74a9634b7237b666db11` | Editorial corrected copy |
| `figures/fig3_token_budget_high_medium.svg` | `37373370906f6d0bbaca1fbdedc13b22852c1288382dacbc9e95dd24ed44b84e` | Editorial corrected copy |
| `figures/fig4_2x2_interaction.svg` | `8ae7d33cd03c774aea1976104098109c18dc3558063e3d20b0a6c24b2ae7bb15` | Editorial corrected copy |
| `figures/fig5_latency_histograms.svg` | `879ba61f1aac93bb5450432bdcc42203ca8ad92fa421ca11c7fdb26bd58db6d5` | Editorial corrected copy |
| `figures/fig6_scheduler_running_waiting.svg` | `b1ed71632f1f3130c5678a1a3611a507b97fa4dc3aa8c0a4c5a8f508cd194e87` | Editorial corrected copy |
| `figures/fig7_cache_contamination.svg` | `e6376f35f222f4f7cb0cda0fa7de3bd44816fe014526cf247f146b9c5cbba3e6` | Editorial corrected copy |
| `figures/fig8_live_control_readback_pid.svg` | `93f8a9c2250ac488fec05cbea47cb9e11d71f6e5ee00261e8c5601cc1fbfa02c` | Editorial corrected copy |
| `diagrams/d01_system_architecture.mmd` | `e0aecf133784836bb71c2e333fd587a7cc597515a56581cdc7fce07dc3eaeeb2` | Editorial corrected copy |
| `diagrams/d02_reconciliation_sequence.mmd` | `230daac20ec0b57079e7a69b08354544de3e23a7e807a01a5eb68dd76fb26c27` | Editorial corrected copy |
| `diagrams/d03_vllm_control_path.mmd` | `2e4ee94bbc07218c6557e7cde9aac74d98799347cba1a0b7f0a3a763ca3c33d3` | Editorial corrected copy |
| `diagrams/d04_sglang_control_path.mmd` | `f2291e67cc0f2960db205fb0b5e6d662378777eaf2175c6e4799b4ad2a2dccd9` | Editorial corrected copy |
| `diagrams/d05_capability_semantics.mmd` | `55d01854a3b3739c45521c98aab15d407c6979bb1170511fc25f86996a493b4e` | Editorial corrected copy |
| `diagrams/d06_approval_lifecycle.mmd` | `34470d4282b3694146989558cd01def8d1182e66ef4f8463fb72e2801e1e052b` | Editorial corrected copy |
| `diagrams/d07_multi_control_failure_recovery.mmd` | `424d374e3202f0d92896c07f88cce7b523f6e65411763b036444490ab502db1d` | Editorial corrected copy |
| `diagrams/d08_evaluation_telemetry_lifecycle.mmd` | `752ca9864360ba7f18f12109301a3540dba0ec63c0c7ef4f5d0248fa320aa3d8` | Editorial corrected copy |
| `diagrams/d09_timeline.mmd` | `615c865365c6530a810b71256bb9665e1b8baff3d824f20fca81d5d436433db2` | Editorial corrected copy |
| `diagrams/d10_operator_workflow.mmd` | `14010a112fd604eebebad310f362c846fb9d2a65306401cc98815542466b6d14` | Editorial corrected copy |
