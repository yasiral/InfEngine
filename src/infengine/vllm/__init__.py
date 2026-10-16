"""vLLM adapter #2 for the dynamic inference control plane.

Milestone 3B: prove the existing DesiredState -> Reconciler -> Adapter spine can
control ONE genuine vLLM V1 engine-level scheduler policy
(``long_prefill_token_threshold``) at runtime, over transport, without
restarting EngineCore or GPU workers.

The package is split so the "vLLM contribution" (the EngineCore runtime hook,
see ``contrib/vllm/``) stays cleanly separated from "our product" (policy,
DesiredState, reconciliation, capability classification, verification) in this
package.
"""

from .runtime_client import (
    ALL_OPERATIONS,
    HTTP_HOT_OPERATIONS,
    HttpVllmRuntimeClient,
    VllmRuntimeClient,
    VllmRuntimeError,
)
from .adapter import VllmAdapter
from . import validation

__all__ = [
    "ALL_OPERATIONS",
    "HTTP_HOT_OPERATIONS",
    "HttpVllmRuntimeClient",
    "VllmRuntimeClient",
    "VllmRuntimeError",
    "VllmAdapter",
    "validation",
]