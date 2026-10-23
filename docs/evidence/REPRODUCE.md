# Reproducing the evidence analysis

## Offline analysis

Use Python 3.11 from a clone of this repository. Only the standard library is required. Run from the repository root:

```bash
python -B docs/evidence/scripts/verify_and_analyze.py
python -B docs/evidence/scripts/make_figures.py
```

The first command checks every file in `CHECKSUMS.sha256`, validates compact capture hashes, verifies published tables against measurement fields, recalculates factorial summaries and 40 paired-ratio medians, checks native cache fractions, differences cumulative histograms and verifies smoke readback/restore. A discrepancy raises an error and exits unsuccessfully. Outputs go to ignored `build/evidence/`:

- `analysis.json`: verification results, block-level factorial values, corrected ratios and disjoint histogram counts.
- `2x2_per_block.csv` and `2x2_summary.json`: recomputed factorial measurements and effects.
- `ab_medians_recomputed.csv`: marginal arm medians recalculated from captures.
- `figures/`: eight deterministically regenerated SVGs.

The regenerated figures use corrected technical headings, explicit window context and narrower smoke wording. They are analytical reproductions, not byte-identical copies of the historical published artwork in `figures/`. Original publication datasets are not overwritten. Rounded archival CSV summaries and inherited ratio columns remain preserved; the full-precision factorial JSON and per-arm measurements are the numerical references. The smoke transcript preserves successful inference payloads, not an explicit HTTP status line.

## Full sanitized capture archive

Download `infengine-measurements-v1.zip` and its `.sha256` companion from the [evidence-v1 release](https://github.com/yasiral/InfEngine/releases/tag/evidence-v1). Check the ZIP's SHA-256 against that companion before extraction. The archive also has a per-file `CHECKSUMS.sha256` and `REDACTIONS.json`.

On PowerShell:

```powershell
Get-FileHash .\infengine-measurements-v1.zip -Algorithm SHA256
Expand-Archive .\infengine-measurements-v1.zip -DestinationPath .uild\captures
python -B docs/evidence/scripts/verify_and_analyze.py --raw-dir build/captures/infengine-measurements-v1
```

On Linux/macOS:

```bash
sha256sum -c infengine-measurements-v1.zip.sha256
unzip infengine-measurements-v1.zip -d build/captures
python -B docs/evidence/scripts/verify_and_analyze.py --raw-dir build/captures/infengine-measurements-v1
```

The optional archive check verifies full-record hashes and compares every compact projection recursively with its sanitized full record. Preserve the repository and release tag together: `PROVENANCE.json` binds their measurement files by SHA-256. Checksums detect changes; they are not a signed attestation.

## Capture environment and GPU reruns

See [environment.sanitized.json](inputs/environment.sanitized.json) and the full archive's `twobytwo/` design, engine identity, engine configuration and GPU configuration records. Initial M12 provisioning and M12.1 provisioning differ; use the configuration for the experiment being examined. Loopback addresses and API PIDs are retained as technical context; account, pod, SSH and GPU identity metadata are omitted.

Offline reproduction reruns analysis of captured measurements. A fresh GPU experiment requires the pinned patched engine, model, matching hardware/provisioning and workload tooling, and control over the serving deployment. The measured InfEngine commit is not established, so an exact historical runtime rerun cannot be guaranteed. This release does not claim a new GPU run, reproduce private deployment infrastructure, or provide credentials. Use the [deployment guide](../DEPLOYMENT.md) for current functional setup; do not treat it as proof of an identical historical environment.
