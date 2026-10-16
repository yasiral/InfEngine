#!/usr/bin/env python3
"""Deterministic InfEngine/vLLM compatibility gate.

Exit 0 when the installed vLLM version is a supported one AND the runtime
scheduler-config hook is present; exit non-zero otherwise, with an actionable
message. Run inside the product image (``python3 /workspace/compat_check.py``)
or from the control-plane host against a reachable control listener.

Unsupported versions fail CLOSED: InfEngine never silently falls back to another
backend and never claims a version it has not tested.
"""

import json
import os
import sys

SUPPORTED_VLLM = {"0.28.0"}


def check_installed() -> int:
    try:
        import vllm
    except ImportError:
        print("compat_check: vLLM is not installed", file=sys.stderr)
        return 1
    version = getattr(vllm, "__version__", "unknown")
    print(f"compat_check: vLLM {version} (supported: {sorted(SUPPORTED_VLLM)})")
    if version not in SUPPORTED_VLLM:
        print(
            f"compat_check: UNSUPPORTED vLLM {version}; expected one of "
            f"{sorted(SUPPORTED_VLLM)}. Failing closed.",
            file=sys.stderr,
        )
        return 1
    # The hook injects EngineCore.set_runtime_scheduler_config.
    try:
        from vllm.v1.engine.core import EngineCore
    except Exception as exc:  # pragma: no cover - import shape varies
        print(f"compat_check: cannot import EngineCore: {exc}", file=sys.stderr)
        return 1
    if not hasattr(EngineCore, "set_runtime_scheduler_config"):
        print(
            "compat_check: runtime scheduler-config hook is NOT applied; rebuild the "
            "product image (the hook is baked in).",
            file=sys.stderr,
        )
        return 1
    print("compat_check: supported version and hook present.")
    return 0


def check_remote(base_url: str, token: str = "") -> int:
    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        base_url.rstrip("/") + "/infengine/control/identity",
        headers={"X-InfEngine-Token": token} if token else {},
    )
    try:
        with urllib.request.urlopen(req, timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception as exc:
        print(f"compat_check: control listener unreachable: {exc}", file=sys.stderr)
        return 1
    version = payload.get("version")
    print(f"compat_check: control listener reports vLLM {version}")
    if version not in SUPPORTED_VLLM:
        print(f"compat_check: UNSUPPORTED vLLM {version}; failing closed.",
              file=sys.stderr)
        return 1
    return 0


def main() -> int:
    base = os.environ.get("INFENGINE_VLLM_CONTROL_URL")
    if base:
        return check_remote(base, os.environ.get("INFENGINE_VLLM_CONTROL_TOKEN", ""))
    return check_installed()


if __name__ == "__main__":
    raise SystemExit(main())
