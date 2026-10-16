"""Transport boundary between the control plane and a remote vLLM V1 runtime.

vLLM is NOT in the same Python object graph as the control-plane process:
EngineCore runs in a background process and GPU workers run under the executor.
The adapter must therefore mutate/observe engine state through a transport,
not direct attribute access.

Two transports are provided:

* ``VllmRuntimeClient`` — the abstract boundary. Concrete in-process transports
  (see ``contrib/vllm/experiment.py``) implement every operation, so
  ``supported_operations()`` defaults to the full set.
* ``HttpVllmRuntimeClient`` — talks to the narrow, opt-in control listener hosted
  by the patched vLLM API-server process (``contrib/vllm/infengine_control.py``).
  That listener exposes ONLY HOT runtime-scheduler config read/apply + identity;
  DRAIN / ``reset_prefix_cache`` / cache stats are intentionally not exposed, so
  this client reports only the operations it actually implements (no permissive
  default that would advertise unsupported operations).
"""

import abc
import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Set

# Every operation named by this ABC. An in-process transport implements all.
ALL_OPERATIONS: Set[str] = {
    "get_runtime_scheduler_config",
    "set_runtime_scheduler_config",
    "identity",
    "pause_generation",
    "resume_generation",
    "reset_prefix_cache",
    "get_runtime_cache_stats",
}

# The HOT runtime-scheduler policies exposed by the HTTP control listener.
HTTP_HOT_OPERATIONS: Set[str] = {
    "get_runtime_scheduler_config",
    "set_runtime_scheduler_config",
    "identity",
}


class VllmRuntimeError(RuntimeError):
    """A vLLM runtime transport/control operation failed."""


class VllmRuntimeClient(abc.ABC):
    """Minimal read/write surface to a remote vLLM V1 runtime."""

    def supported_operations(self) -> Set[str]:
        """Operations this transport actually implements.

        In-process transports implement the full set; an HTTP transport MUST
        override this to report only what its listener exposes, so the adapter
        never advertises an operation the transport cannot perform.
        """
        return set(ALL_OPERATIONS)

    @abc.abstractmethod
    def get_runtime_scheduler_config(self) -> Dict[str, Any]:
        """Return the live EngineCore scheduler-config values."""

    @abc.abstractmethod
    def set_runtime_scheduler_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a validated scheduler-config override; return resulting values."""

    @abc.abstractmethod
    def identity(self) -> Dict[str, Any]:
        """Return externally observable runtime identity (PIDs, instance id)."""

    # ---- DRAIN-class operations (3C) -------------------------------------
    @abc.abstractmethod
    def pause_generation(self, mode: str = "wait", clear_cache: bool = False) -> None:
        """Stop admitting new requests; mode='wait' drains in-flight work."""

    @abc.abstractmethod
    def resume_generation(self) -> None:
        """Resume admission after pause_generation."""

    @abc.abstractmethod
    def reset_prefix_cache(self, reset_running_requests: bool = False,
                           reset_connector: bool = False) -> bool:
        """Reset the prefix cache after quiescence."""

    @abc.abstractmethod
    def get_runtime_cache_stats(self) -> Dict[str, Any]:
        """Engine observable for DRAIN/operation verification (e.g. prefix-cache
        entries before/after a reset)."""


class HttpVllmRuntimeClient(VllmRuntimeClient):
    """Concrete vLLM runtime client over the private HTTP control listener.

    The listener (``contrib/vllm/infengine_control.py``) forwards to the SAME
    ``AsyncLLM``/EngineCore that serves inference. Only HOT scheduler-config
    read/apply and identity are exposed; the DRAIN/cache operations raise
    ``VllmRuntimeError`` and are excluded from ``supported_operations()``.
    """

    def __init__(
        self,
        base_url: str,
        *,
        token: Optional[str] = None,
        timeout: float = 10.0,
        instance_id: Optional[str] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.token = token if token is not None else os.environ.get(
            "INFENGINE_VLLM_CONTROL_TOKEN"
        )
        self.timeout = timeout
        self._instance_id = instance_id or f"vllm:{self.base_url}"

    # ------------------------------------------------------------ transport
    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None
    ) -> Any:
        url = f"{self.base_url}{path}"
        data = json.dumps(body).encode("utf-8") if body is not None else None
        headers = {"Content-Type": "application/json"} if body is not None else {}
        if self.token:
            headers["X-InfEngine-Token"] = self.token
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            raise VllmRuntimeError(
                f"{method} {path} returned HTTP {exc.code}: {raw[:200]}"
            ) from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise VllmRuntimeError(f"{method} {path} failed: {exc}") from exc

    # ------------------------------------------------------------ supported
    def supported_operations(self) -> Set[str]:
        return set(HTTP_HOT_OPERATIONS)

    # ------------------------------------------------------------ read/write
    def get_runtime_scheduler_config(self) -> Dict[str, Any]:
        return self._request("GET", "/infengine/control/config")

    def set_runtime_scheduler_config(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/infengine/control/config", overrides)

    def identity(self) -> Dict[str, Any]:
        payload = self._request("GET", "/infengine/control/identity")
        payload.setdefault("engine_instance_id", self._instance_id)
        payload.setdefault("base_url", self.base_url)
        return payload

    # ------------------------------------------------------- out of scope
    def pause_generation(self, mode: str = "wait", clear_cache: bool = False) -> None:
        raise VllmRuntimeError(
            "pause_generation is not supported by the HTTP control transport"
        )

    def resume_generation(self) -> None:
        raise VllmRuntimeError(
            "resume_generation is not supported by the HTTP control transport"
        )

    def reset_prefix_cache(self, reset_running_requests: bool = False,
                           reset_connector: bool = False) -> bool:
        raise VllmRuntimeError(
            "reset_prefix_cache is not supported by the HTTP control transport"
        )

    def get_runtime_cache_stats(self) -> Dict[str, Any]:
        raise VllmRuntimeError(
            "get_runtime_cache_stats is not supported by the HTTP control transport"
        )
