# Public release verification

Prepared 17 October 2026 from the supplied InfEngine area and `InfEngine_github_package.zip`. The public release uses a separate staging directory and new Git history; the original engineering workspace is read only.

## Fresh local tests

On Windows with Python 3.11, the CPU-only local suite completed:

**619 passed, 9 deselected, 2 dependency deprecation warnings, 217.35 seconds.**

```bash
python -m pytest -q -m "not integration"
```

The release run used `CUDA_VISIBLE_DEVICES` empty, `VLLM_INTEGRATION=0`, `SGLANG_INTEGRATION=0`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1`. The nine live GPU integration tests were deliberately excluded, not reported as passed. Local HTTP fixtures ran outside the filesystem/network sandbox. An earlier sandboxed test attempt was stopped; the result above is the subsequent completed run.

Warnings came from installed `websockets.legacy` and Uvicorn's legacy WebSocket protocol integration. They were not hidden or converted to test successes.

## Release contents and source preservation

- The initial 121 implementation/test/integration/example/documentation files selected from the supplied ZIP match the current source area byte for byte.
- All implementation Python files remain unchanged. Release edits concern README, architecture/setup instructions, metadata, ignore/package manifests, license/attribution and patch modification headers.
- The release excludes original experimental environments/logs, article/review archives, private notes, local credentials/configuration, cached environments, model weights and previous Git history.
- A pre/post hash inventory covers 1,232 original engineering files; the original workspace is not modified.
- The public root history will contain only the reviewed release tree. It does not import the engineering monorepo's historical commits.

## License and dependency scope

The project owner selected Apache-2.0 for InfEngine. License texts were checked at the pinned upstream tags, vLLM 0.28.0 and SGLang 0.5.18, and retained with attributions. Patch headers identify modified upstream material. See `THIRD_PARTY_NOTICES.md` and `NOTICE` for exact scope.

This source release does not bundle upstream engine binaries, container base-image contents, installed dependencies or model weights. Those retain their own licenses; the source-release review does not certify all contents of an image an operator later downloads.

## Packaging and upstream-source checks

The 0.1.0 wheel built offline with no dependency resolution and installed into an isolated directory. Its installed unified CLI displayed help successfully, all six declared console entry points were present, and the wheel included the project license, NOTICE, third-party notices and both upstream license texts.

The canonical vLLM hook applied successfully to source files fetched from v0.28.0, applied a second time without duplicate changes, and left syntactically valid Python. The SGLang patch passed `git apply --check` against the v0.5.18 scheduler source. This check initially exposed Windows CRLF patch context and an embedded BOM; the release patches now use UTF-8 without BOM and LF, enforced by `.gitattributes`. No original engineering files were changed. These source-level checks do not replace a container build or live GPU acceptance.

## Validation boundaries

The control tables, CLI flags, native mappings and Docker COPY targets were checked against included implementation and reference documentation. The README distinguishes functional acceptance from performance evidence and engine-transaction boundaries from engine atomicity.

No Docker image build, live vLLM/SGLang acceptance, model download, new GPU benchmark or cloud GPU provisioning was performed for this release. Historical T2000 acceptance and A100 experiments are attributed as preserved engineering evidence, not reruns. Installation/build and final content-scan receipts are retained by the publisher alongside the release commit.
