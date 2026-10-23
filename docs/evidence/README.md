# InfEngine experimental evidence

This release brings the measurements behind the [InfEngine engineering article](https://puristan.com/posts/infengine-runtime-control/) into the source repository. It includes all **eight experimental SVG figures, ten architectural Mermaid sources and nine publication data files**, together with compact measurement captures, analysis scripts and provenance.

The experiments concern **patched vLLM 0.28.0, Qwen2.5-7B-Instruct BF16 on one A100 80 GB PCIe**. They establish workload-specific effects of selected runtime controls. They do not establish a general speedup, an automatic optimal policy, SGLang A100 performance, or production readiness. All seven documented controls have functional acceptance; performance coverage is narrower.

## Start here

| Material | Purpose |
|---|---|
| [Technical case study](CASE_STUDY.md) | Experimental design, quantitative results and interpretation |
| [Evidence ledger](EVIDENCE_LEDGER.md) | What each claim establishes and its limitations |
| [Figure gallery and manifest](FIGURE_MANIFEST.md) | All eight figures with measurement context and sources |
| [Architecture diagrams](DIAGRAMS.md) | Ten diagrams rendered from the accompanying Mermaid sources |
| [Reproduction instructions](REPRODUCE.md) | Offline analysis, checksums and optional full archive verification |
| [Verification report](VERIFICATION.md) | Checks performed for this evidence release |
| [Provenance](PROVENANCE.json) | Original source hashes and compact-to-archive mappings |
| [Release verification](../RELEASE_VERIFICATION.md) | Existing software release checks, separate from historical GPU evidence |

## Measurements and corrections

The [nine publication datasets](data/) are preserved without edits. [Compact captures](inputs/) retain the measurement fields required for offline verification: 48 A/B arms, 12 factorial cell runs, initial and pressure pilots, native cache counters and the live-control smoke sequence. The [sanitized full-measurement archive](https://github.com/yasiral/InfEngine/releases/tag/evidence-v1) provides broader JSON records and original capture configuration without deployment credentials or account identifiers.

**Use recomputed paired ratios when interpreting A/B effects.** Some inherited ratios in `ab_medians.csv` disagree with the individual paired measurements. [The reconciliation](support/ab-ratio-reconciliation.json) preserves the archived value and gives all three B/A ratios and their median. In the high-load concurrency experiment the archived request-throughput ratio is 0.709; the paired median is **0.736535** (about −26.3%). The original dataset remains available for provenance.

The original measured InfEngine commit was not established. `PROVENANCE.json` records that field as unknown; the current public source revision is not presented as the measured revision. Sanitization removes private environment metadata, not numerical performance values. These are historical captures, not newly repeated GPU experiments.

## Reproduce without a GPU

From the repository root with Python 3.11:

```bash
python -B docs/evidence/scripts/verify_and_analyze.py
python -B docs/evidence/scripts/make_figures.py
```

Outputs go to ignored `build/evidence/`. No serving engine, model download, API credential or third-party Python package is needed. [REPRODUCE.md](REPRODUCE.md) distinguishes recomputing an analysis from rerunning an experiment on a GPU.
