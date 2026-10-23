# Evidence release verification

Evidence version: **evidence-v1**. Review date: **2026-10-23**. Software version remains 0.1.0. This is an analysis and documentation release; no GPU experiment was rerun.

| Check | Evidence |
|---|---|
| Publication coverage | Eight SVGs, ten Mermaid sources and nine data files; nine data files retain their original SHA-256 hashes. |
| Native measurement agreement | All 48 A/B arms checked for throughput, token rate, latency, completions, running/waiting/KV gauges and execution evidence. |
| Pilot agreement | Fifteen initial pilot aggregate points and M12.1 pressure-pilot measurement projections checked. |
| Cache confound | Eighteen native before/after counter fractions checked against publication precision and nine ordered pairs. |
| Factorial reproduction | Twelve raw cell records reproduce per-block CSV and full-precision summary JSON within absolute tolerance 1e-10. Interaction +0.505588 requests/s. |
| Ratio reconciliation | Forty metric/experiment paired-ratio summaries reproduced from individual arms. Original inherited values retained separately. |
| Histogram accounting | Four native windows checked for nonnegative disjoint bucket counts and preserved sample totals. |
| Smoke evidence | Native 64→16→64, verified convergence, successful completion payloads and unchanged API PID; no explicit HTTP status captured. |
| Archive integrity | Every compact/full record mapping checked by SHA-256 and recursive measurement comparison; archive includes its own per-file manifest. |
| Independent source comparison | 94,364 numeric leaves checked across 466 original/sanitized JSON records: zero measurement changes. Thirteen numeric metadata leaves removed under private account/cost fields. |
| Cross-platform byte preservation | Evidence files have Git text normalization disabled so Windows/Linux checkouts retain the bytes covered by the checksum manifest. |
| Reproducibility | Python 3.11 standard-library analysis and SVG generation; outputs separate from preserved inputs. |

The [GitHub workflow](../../.github/workflows/evidence.yml) repeats the evidence checks and runs the CPU test suite with live integrations and model downloads disabled. Check the [Actions results](https://github.com/yasiral/InfEngine/actions/workflows/evidence.yml) for current execution status. The existing software [release verification](../RELEASE_VERIFICATION.md) records the earlier 619 passing local tests; that count is not presented here as a new GPU result.

The original engineering source, experimental data and Puristan article are read-only inputs to this release. The complete existing README is preserved, with an appended evidence section. Its diagrams remain unchanged. The only other existing repository file changed is `.gitattributes`, with one rule added to preserve evidence bytes. Source-hash and protected-file comparisons are part of the release preparation. Independent review found no remaining critical or important issues after the line-ending correction.

Private account/pod/SSH information, GPU identifiers, credentials, logs, databases, source archives and unreviewed free-text transcripts are excluded. Full sanitized JSON captures retain numerical measurements and relevant engine configuration; metadata removals and JSON extraction are recorded in the archive's `REDACTIONS.json`. Sanitization and checksum verification do not establish that the historical experiment was methodologically perfect. See the [ledger](EVIDENCE_LEDGER.md) for confounds, missing measured source revision and interpretation limits.
