# A100 runtime-control case study

InfEngine provides an inspect–apply–verify–evaluate–restore workflow for bounded native scheduler controls. Its purpose is to make policy changes observable and reviewable. It does not know the optimal configuration for every workload.

## Experimental setup

The historical M12/M12.1 measurements used one NVIDIA A100 80 GB PCIe, Qwen2.5-7B-Instruct in BF16 and patched vLLM 0.28.0. CUDA graphs stayed enabled. Initial provisioning used `max_num_seqs=64`; M12.1 restarted with a provisioned ceiling of 256 to separate runtime admission limits from that startup bound. Runtime concurrency remains a ceiling, not an exact number of active requests. The token budget is per scheduler step and includes prefill and decode tokens; it is not output length.

Initial workload profiles were interactive (128 input / 256 maximum output tokens), prefill (2048 / 64), and mixed (128 or 2048 / 256). Five offered closed-loop concurrency levels, two repeats and 15-second measurement windows produced the 15 initial pilot points. These short pilots selected load regions; they did not prove GPU saturation. Later pressure pilots tested higher offered loads and retained incomplete/drained work and evidence-window qualifications.

## Native control verification

The smoke capture records native concurrency **64 → 16 → 64**, successful completion payloads while limited and after restoration, and unchanged API-server PID **3695**. Apply and restore both report `CONVERGED` with verified observed values. This establishes live native control and API-server continuity for the recorded sequence. It does not independently identify the separate EngineCore process or establish a performance improvement.

The transcript has no explicit HTTP status line. The HTTP 200 label retained in the historical published artwork is a reported engineering observation; successful completion payloads are the directly recoverable evidence. The regenerated figure uses that narrower wording.

## Prefix-cache contamination in the initial A/B experiment

The initial cache-enabled pass compared three controls in three paired runs each. Counter deltas show first-arm token hit fractions around **0.031–0.059**, while second arms reached **0.510–0.903**. AB/BA ordering did not remove this carry-over: the second phase inherited a warmer cache. Those runs document a confound and must not be used as clean policy speedup estimates. Later A/B and factorial experiments disabled prefix caching. Native token hit fractions are token-weighted, not percentages of requests with a cache hit.

## Throughput and latency trade-offs with cache disabled

The clean A/B experiments used three pairs, 30-second measurement windows, fixed seeds within pairs and alternating arm order. At higher offered load, reducing the total token budget from 8192 to 1024 produced a median paired request-throughput ratio of **1.359641**, while worsening first-token latency. At medium offered load the corresponding ratio was **0.865958**. Reducing concurrency produced ratios of **0.736535** at higher load and **0.421683** at medium load. Changing the long-prefill threshold from 0 to 1024 produced **1.069601** in its tested condition.

These ratios are medians of three within-pair B/A ratios. A ratio of the two marginal arm medians is a different statistic. Neither statistic describes every workload. The unchanged publication summaries contain inherited discrepancies; use [the reconciliation](support/ab-ratio-reconciliation.json) for pair-level values. Scheduler running/waiting gauges help describe each operating point but do not independently prove an internal causal mechanism.

## M12.1 factorial experiment

The 2×2 experiment tested runtime concurrency 64/32 and total scheduled-token budget 8192/1024 with offered concurrency 96. It used three interleaved blocks and 60-second measurement windows per cell, with prefix caching disabled and a provisioned sequence ceiling of 256.

| Cell | Tokens per step | Concurrency ceiling | Mean completed requests/s | Mean block TTFT p95, seconds |
|---|---:|---:|---:|---:|
| A | 8192 | 64 | 2.722061 | 19.3214 |
| B | 8192 | 32 | 3.222064 | 19.5000 |
| C | 1024 | 64 | 2.916190 | 39.0000 |
| D | 1024 | 32 | 2.910604 | 39.0000 |

At 8192 tokens, reducing concurrency added **0.500002 requests/s**. At 1024 tokens, it changed throughput by **−0.005586 requests/s**. The difference of these effects, `(B−A)−(D−C)`, is **+0.505588 requests/s**. That is about **17.19% of the grand cell mean**, or **18.57% of cell A**. Those denominators must not be interchanged; `interaction_pct` in the publication JSON uses cell A.

The result shows an interaction under this experimental condition: the throughput effect of admission depends on the token budget. The lower token budget also doubled the mean of block-level TTFT p95 estimates. Those values are **not pooled request percentiles**. Three blocks support a descriptive comparison; no confidence interval or universal tuning recommendation is implied. These measurements do not compare InfEngine with an alternative controller.

## Telemetry resolution and unfinished work

Native latency histogram buckets are cumulative counts; subtract adjacent buckets before plotting disjoint masses. The illustrated pair-1 windows cover each arm's own final five seconds. TTFT counts are 41 versus 15; E2E counts are 16 versus 15. Small samples and coarse bucket boundaries limit interpretation. They are not full-run per-request distributions. The captures retain completion, incomplete, timeout, error and drain counts so throughput is not interpreted without workload accounting.

## Functional coverage and remaining limitations

Five vLLM and two SGLang HOT controls have documented functional acceptance. The A100 performance work covers selected vLLM controls: concurrency, scheduled-token budget and long-prefill threshold. It does not measure A100 benefits for the KV watermark, full-input reservation, or SGLang controls. The M13 extracts support multi-control evaluation and restoration reasoning; unavailable telemetry cannot demonstrate performance equivalence.

vLLM's coordinated multi-field operation and SGLang's sequential updates are not engine-atomic transactions. Restoration changes policy; it cannot undo completed requests or cache history. Functional readback, measurable effect and benefit for an operational objective remain separate evidence levels. See [control semantics](../CONTROL_REFERENCE.md), [the evidence ledger](EVIDENCE_LEDGER.md) and [the full article](https://puristan.com/posts/infengine-runtime-control/) for the engineering narrative and claim boundaries.
