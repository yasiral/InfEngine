# Claims → evidence map

Every claim below is traceable to a captured artifact. "Confidence" is the
author's assessment given sample sizes and repeatability, not a statistical test.
M12.1 2×2 entries are filled in by Part F (see `tables/2x2_summary.csv`).

| # | claim | exact numbers | source runs | source artifacts | confidence | caveat |
|---|---|---|---|---|---|---|
| C1 | InfEngine changed live vLLM scheduler policy with no engine restart | `runtime_max_running_reqs` 64→16→64, status CONVERGED, `api_pid=3695` unchanged; raw readback matches | M12 smoke | `deployment/smoke/smoke.txt` | HIGH (functional) | single engine instance |
| C2 | At HIGH mixed offered load, lowering `runtime_concurrency` 64→16 cut throughput and inflated latency tails | req/s 4.12→2.92 (**−29%**); TTFT p95 4.23→19.5 s (**×4.6**); queue p95 2.66→14.43 s (**×5.4**); running 56→16, waiting 0→48 | `runtime_concurrency` clean, 3 pairs | `runtime_concurrency/pair{1,2,3}_*/` | HIGH (3/3 pairs same sign; per-pair req/s B/A 0.671/0.737/0.901) | not saturated; per-arm completed 97–128 |
| C3 | At MEDIUM mixed load the concurrency trade is larger | req/s 2.97→1.25 (**−58%**); TTFT p95 2.38→9.88 s (**×4.2**); queue p95 0.3→9.75 s | `runtime_concurrency_medium`, 3 pairs | `runtime_concurrency_medium/pair{1,2,3}_*/` | HIGH (ratios 0.422/0.422/0.411) | B=4 arm completed 36/arm (sample-limited) |
| C4 | At HIGH prefill load, lowering `runtime_max_scheduled_tokens` 8192→1024 raised throughput and cut E2E but worsened TTFT | req/s 1.93→2.63 (**+36%**); E2E p95 29.39→19.63 s (**−33%**); TTFT p95 8.81→19.5 s (**×2.2 worse**); running 59→20, waiting 0→44 | `runtime_max_scheduled_tokens` clean, 3 pairs | `runtime_max_scheduled_tokens/pair{1,2,3}_*/` | HIGH (ratios 1.365/1.360/1.324) | token budget also moved running (coupling; see 2×2) |
| C5 | At MEDIUM prefill load the same change flipped emphasis | req/s 3.16→2.74 (**−13%**); TTFT p95 2.41→0.74 s (**×0.31**); E2E p95 8.0→9.75 s (worse) | `runtime_max_scheduled_tokens_medium`, 3 pairs | `runtime_max_scheduled_tokens_medium/pair{1,2,3}_*/` | HIGH (ratios ≈0.866) | per-arm completed 82–95 |
| C6 | `long_prefill_token_threshold` 0→1024 produced no measurable effect | req/s +7%, tok/s −1%, TTFT p95 +4%, E2E p95 1.00×, queue 1.00×, running 59=59 | `long_prefill_token_threshold` clean, 3 pairs | `long_prefill_token_threshold/pair{1,2,3}_*/` | HIGH (null, ratios ≈1.0) | NULL because trigger not exercised |
| C7 | Prefix-cache warmth confounded the first pass and was corrected | second-arm hit fraction 0.60–0.90 vs first-arm 0.03–0.06; one pair reversed | cache-ON pass | `*_cacheON/pair*/metrics_{before,after}.txt` | HIGH (methodological) | retained as evidence |
| C8 | KV-pressure controls were never exercised | KV usage ≤ ~20%, `num_preemptions=0` at all loads | pilot + all A/B | `pilot/*/result.json`, `*/pair*/result.json` | HIGH | cannot make any claim for `runtime_kv_watermark` / `runtime_reserve_full_isl` |
| C9 | InfEngine's control path is token-gated and loopback-only | control endpoint returns 401 without token; not published publicly | smoke | `deployment/smoke/smoke.txt` | HIGH | — |
| C10 | 2×2: at fixed concurrency, lowering `runtime_max_scheduled_tokens` 8192→1024 doubles TTFT p95, at both concurrency 64 and 32 | TTFT p95 19.3→39.0 s (conc64) and 19.5→39.0 s (conc32) — **×2.0 both**; TTFT insensitive to concurrency | M12.1 `twobytwo/`, 3 blocks | `twobytwo/`, `final/2x2_summary.json`, `figures/fig4_2x2_interaction.svg` | HIGH (repeat CV <1%) | single offered load L=96, prefill-bound |
| C11 | 2×2: the token-budget throughput effect interacts with concurrency (sign flip) | token effect +7.1% at conc64, −9.7% at conc32; interaction **+17.2%** on req/s; main token effect −2.0%, main concurrency effect +8.8% | M12.1 `twobytwo/`, 3 blocks | `final/2x2_table.csv`, `final/2x2_summary.json` | MEDIUM-HIGH (repeat spread 0.1–2.0%; effect far above noise) | magnitude specific to L=96; not extrapolated |
| C12 | M12.1 reached a more meaningful prefill pressure region than M12 HIGH | raised max-num-seqs 64→256; L=96 gives waiting 42→204, TTFT 16–19 s, KV≤31%, still ~110 completions/arm at 60 s; still 0 preemptions | M12.1 pilot | `pilot/prefill_m12_1_result.json`, `pilot/confirm96/result.json` | HIGH | no preemptions/OOM; not a full saturation knee |

## Not claimed (unsupported)

- No GPU-saturation knee, no production "best" configuration, no automatic policy.
- No causal mechanism beyond the measured scheduler/telemetry evidence.
- No cross-engine (SGLang) or cross-GPU generalization.
