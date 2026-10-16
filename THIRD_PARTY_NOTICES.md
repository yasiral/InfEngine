# Third-party code and redistribution scope

The public release includes InfEngine implementation and integration patches, not complete vendored upstream engines, downloaded Python dependencies, images or model weights.

| Component | Verified upstream reference | Included material and terms |
|---|---|---|
| vLLM | [`v0.28.0/LICENSE`](https://github.com/vllm-project/vllm/blob/v0.28.0/LICENSE) | Patch context, runtime-hook injection/adaptation and integration listener. Apache-2.0 text is retained in `third_party/vllm-LICENSE`; modified paths and purpose are identified in the patch headers and NOTICE. |
| SGLang | [`v0.5.18/LICENSE`](https://github.com/sgl-project/sglang/blob/v0.5.18/LICENSE) | Scheduler patch and integration definition. Apache-2.0 text, including SGLang Team attribution, is retained in `third_party/sglang-LICENSE`. |

The `NOTICE` path at each pinned repository root returned HTTP 404 during release inspection; no upstream root NOTICE was found there. The license texts and source attributions that are present are retained. This observation does not assert that every dependency inside an upstream container lacks additional notices.

The canonical vLLM patch modifies utility APIs in `vllm/v1/engine/{async_llm,core,core_client,llm_engine}.py`; the SGLang patch modifies `python/sglang/srt/managers/scheduler.py`. `contrib/vllm/apply_hook.py` applies the corresponding vLLM scheduler integration to the pinned runtime. Patch files themselves record changed files and added/removed lines; release headers explicitly identify them as modified upstream material.

Container base images and dependencies are retrieved by the operator under their respective licenses. Their binaries are not redistributed by this source release. A Dockerfile is not a blanket license audit of every image component. Model weights, including the example Qwen models, are obtained separately and have their own applicable terms. The InfEngine license does not replace those terms.

The InfEngine project license was selected as Apache-2.0 by the project owner for this release. Existing upstream rights and notices remain separate and unchanged.
