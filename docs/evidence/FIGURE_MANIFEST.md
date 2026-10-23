# Experimental figure gallery and manifest

The eight SVGs below preserve the published article assets. Reproduction scripts create separate renderings with corrected technical headings and narrower smoke wording. All performance figures concern the same A100/model/vLLM study; each experiment has its own cache, load and window settings.

## 1. Pilot request throughput across offered concurrency

![Pilot request throughput across offered concurrency](figures/fig1_throughput_vs_concurrency.svg)

**Source:** pilot_profiles.csv. **Interpretation:** Two-repeat medians, 15-second windows, cache enabled; no proven saturation knee.

## 2. Pilot TTFT p95 across offered concurrency

![Pilot TTFT p95 across offered concurrency](figures/fig2_ttft_p95_vs_concurrency.svg)

**Source:** pilot_profiles.csv. **Interpretation:** Same initial pilot; quantile summaries, not pooled percentiles.

## 3. Token-budget throughput and TTFT trade-offs

![Token-budget throughput and TTFT trade-offs](figures/fig3_token_budget_high_medium.svg)

**Source:** ab_medians.csv; ab_per_arm.csv. **Interpretation:** Marginal medians of three arms per condition; paired-ratio estimates are a separate statistic.

## 4. Concurrency and token-budget factorial interaction

![Concurrency and token-budget factorial interaction](figures/fig4_2x2_interaction.svg)

**Source:** 2x2_per_block.csv; 2x2_summary.json. **Interpretation:** Three interleaved blocks, offered concurrency 96, 60 seconds/cell, cache disabled; means of block quantile estimates.

## 5. Native latency histogram windows and bucket resolution

![Native latency histogram windows and bucket resolution](figures/fig5_latency_histograms.svg)

**Source:** pair-1 result.json histogram families. **Interpretation:** Each arm's own final five-second window; cumulative buckets must be differenced; TTFT n=41/15 and E2E n=16/15.

## 6. Scheduler running and waiting request counts

![Scheduler running and waiting request counts](figures/fig6_scheduler_running_waiting.svg)

**Source:** ab_medians.csv; ab_per_arm.csv. **Interpretation:** Medians of available window summaries; no per-request tracing or causal proof.

## 7. Prefix-cache contamination across paired arms

![Prefix-cache contamination across paired arms](figures/fig7_cache_contamination.svg)

**Source:** cache_confound.csv; native cache counter extracts. **Interpretation:** Nine ordered pairs, 18 native token counter deltas; this is a confound, not a speedup.

## 8. Live concurrency control, readback and restoration

![Live concurrency control, readback and restoration](figures/fig8_live_control_readback_pid.svg)

**Source:** deployment/smoke/smoke.json. **Interpretation:** Functional readback/restore, unchanged API PID. Historical HTTP 200 label is reported; no explicit status line in transcript.
