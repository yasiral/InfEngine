FROM vllm/vllm-openai:v0.28.0

# --- 1. Apply the vLLM runtime scheduler-config hook (full transport) ---
COPY contrib/vllm/apply_hook.py /workspace/apply_hook.py
RUN python3 /workspace/apply_hook.py

# --- 2. Install the control-plane package (control + vllm subpackages only; no torch dep) ---
RUN mkdir -p /workspace/cp/infengine
COPY src/infengine/control /workspace/cp/infengine/control
COPY src/infengine/vllm /workspace/cp/infengine/vllm
RUN touch /workspace/cp/infengine/__init__.py

# --- 3. The live experiment script ---
COPY contrib/vllm/experiment.py /workspace/experiment.py

ENV PYTHONPATH=/workspace/cp:/workspace
# WSL2 GPU passthrough requires explicitly enabling pinned memory (vLLM gates
# UVA behind this; kernel >= 4.19.121). Required for the live experiment.
ENV VLLM_WSL2_ENABLE_PIN_MEMORY=1
WORKDIR /workspace
