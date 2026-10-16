"""SGLang adapter #3 for the dynamic inference control plane.

Milestone 4B-B: prove the existing DesiredState -> Reconciler -> Adapter spine
can control a real SGLang v0.5.18 runtime over its native HTTP admin surface
(HTTP -> TokenizerManager -> ZMQ/msgpack -> Scheduler), without restarting the
Scheduler or GPU workers.

Supported envelope (M4B-B): SGLang v0.5.18, PP=1, DP=1, text/token input, no
speculative decoding, no disaggregated prefill/decode.

The package is split so the "SGLang contribution" (the runtime hook patch, see
``contrib/sglang/``) stays cleanly separated from "our product" (policy,
DesiredState, reconciliation, capability classification, verification) here.
"""

from .runtime_client import (
    HttpSglangRuntimeClient,
    SglangRuntimeClient,
    SglangRuntimeError,
)
from .adapter import SglangAdapter
from . import validation

__all__ = [
    "HttpSglangRuntimeClient",
    "SglangRuntimeClient",
    "SglangRuntimeError",
    "SglangAdapter",
    "validation",
]