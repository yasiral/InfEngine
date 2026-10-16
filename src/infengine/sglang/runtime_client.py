"""Transport boundary between the control plane and a remote SGLang v0.5.18 runtime.

SGLang is NOT in the same Python object graph as the control-plane process: it
runs as a separate server (typically in Docker) behind an HTTP admin surface.
The adapter mutates/observes engine state through that HTTP surface, not direct
attribute access.

Reused native SGLang transport (no new RPC protocol):

    HTTP endpoint -> TokenizerManager -> ZMQ/msgpack -> Scheduler

Endpoints used (verified against v0.5.18 source and live M4B-A evidence):

  POST /set_internal_state     -> [true] / [false] per scheduler rank
  POST /pause_generation       -> {"message": ..., "status": "ok"}
  POST /continue_generation    -> {"message": ..., "status": "ok"}
  POST /flush_cache            -> 200 "Cache flushed..." | 400 failure message
  GET  /server_info            -> ServerArgs + internal_states (live config)
  GET  /get_load               -> [{"dp_rank", "num_reqs", "num_waiting_reqs", ...}]

Observation note: v0.5.18 has no ``/get_internal_state`` HTTP endpoint (the
M4B-A experiment got a 404 for it). The live scheduler-config observation
surface is ``/server_info`` -> ``internal_states[rank]`` (resolved server args
including the runtime-overridden ``pp_max_micro_batch_size`` and the provisioned
``max_running_requests`` / ``max_prefill_tokens``). The canonical patch adds the
scheduler-local ``runtime_max_prefill_tokens`` to ``Scheduler.get_internal_state``,
so it too is exposed through ``/server_info -> internal_states``. All policy
readback therefore reads the ACTUAL scheduler value (no shadow state).

``HttpSglangRuntimeClient`` is deliberately tiny and replaceable: any
transport (in-process, sidecar, upstream control API) can implement
``SglangRuntimeClient`` without changing ``Reconciler`` or ``SglangAdapter``.
"""

import abc
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, Optional


class SglangRuntimeError(RuntimeError):
    """An SGLang runtime transport/control operation failed."""


class SglangRuntimeClient(abc.ABC):
    """Minimal read/write surface to a remote SGLang v0.5.18 runtime.

    ``provisioned`` holds the boot-time ceilings used by planning:
    ``max_running_requests`` and ``max_prefill_tokens``. A concrete client
    discovers them from the runtime when possible (``HttpSglangRuntimeClient``
    reads them from ``/server_info``) and may also accept them at construction.
    """

    @abc.abstractmethod
    def provisioned(self) -> Dict[str, Any]:
        """Return boot-time provisioned ceilings: max_running_requests,
        max_prefill_tokens (and any other planning-relevant constants)."""

    @abc.abstractmethod
    def set_policy(self, name: str, value: int) -> bool:
        """Apply a validated runtime policy via set_internal_state.

        Returns True only when the Scheduler confirmed the mutation
        (``[true]``). The client records the confirmed value for readback.
        """

    @abc.abstractmethod
    def get_policy(self, name: str) -> Any:
        """Return the current value of a runtime policy.

        ``runtime_max_running_seqs`` is read live from the runtime when
        available; ``runtime_max_prefill_tokens`` is the last scheduler-confirmed
        value (no v0.5.18 endpoint exposes the scheduler-local field).
        """

    @abc.abstractmethod
    def refresh(self) -> None:
        """Re-read live runtime state (bounds + runtime-overridden policies)."""

    @abc.abstractmethod
    def get_load(self) -> Dict[str, Any]:
        """Return the first DP rank's load snapshot from /get_load."""

    @abc.abstractmethod
    def pause_generation(self, mode: str = "retract") -> None:
        """Stop admitting new requests; mode='retract' releases in-flight KV."""

    @abc.abstractmethod
    def continue_generation(self, torch_empty_cache: bool = False) -> None:
        """Resume admission after pause_generation."""

    @abc.abstractmethod
    def flush_cache(self) -> bool:
        """Flush the engine cache after quiescence. True only on HTTP 200."""

    @abc.abstractmethod
    def flush_cache_ok(self) -> bool:
        """Whether the most recent flush_cache() was confirmed (HTTP 200)."""

    @abc.abstractmethod
    def identity(self) -> Dict[str, Any]:
        """Return externally observable runtime identity (instance, PIDs)."""


class HttpSglangRuntimeClient(SglangRuntimeClient):
    """Concrete SGLang runtime client over the native HTTP admin surface.

    Uses only the Python standard library (``urllib.request``); no new
    dependency is introduced.
    """

    def __init__(
        self,
        base_url: str,
        *,
        provisioned: Optional[Dict[str, Any]] = None,
        timeout: float = 10.0,
        instance_id: Optional[str] = None,
        remote_pids: Optional[Dict[str, Any]] = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._instance_id = instance_id or f"sglang:{self.base_url}"
        self._remote_pids = dict(remote_pids or {})
        self._control_pid = os.getpid()
        # Provisioned ceilings: discovered from /server_info when the runtime is
        # reachable, else the caller-supplied values (never hard-coded product
        # constants).
        self._provisioned: Dict[str, Any] = dict(provisioned or {})
        # Runtime policy record (control-plane authoritative view).
        self._policy: Dict[str, Any] = {}
        self._discovered = False
        self._last_flush_ok: Optional[bool] = None
        try:
            self.refresh()
        except SglangRuntimeError:
            # Runtime may be unreachable at construction (e.g. unit tests or
            # startup ordering); caller-supplied provisioned values remain.
            if not self._provisioned:
                raise

    # -------------------------------------------------------------- transport
    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
                 timeout: Optional[float] = None) -> Any:
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                raw = resp.read()
                return resp.status, raw
        except urllib.error.HTTPError as exc:
            return exc.code, exc.read()
        except (urllib.error.URLError, OSError) as exc:
            raise SglangRuntimeError(f"SGLang request {method} {path} failed: {exc}") from exc

    def _json(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        status, raw = self._request(method, path, body)
        if not (200 <= status < 300):
            raise SglangRuntimeError(
                f"SGLang {method} {path} returned HTTP {status}: "
                f"{raw.decode('utf-8', errors='replace')[:300]}"
            )
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise SglangRuntimeError(f"SGLang {path} returned non-JSON body") from exc

    # -------------------------------------------------------------- discovery
    def refresh(self) -> None:
        """Read /server_info: provisioned bounds + LIVE scheduler policies.

        This is the authoritative observation surface. ``internal_states[0]``
        carries the resolved scheduler config, which now includes the
        scheduler-local ``runtime_max_prefill_tokens`` (exposed by the canonical
        patch) and the runtime-overridden ``pp_max_micro_batch_size``.
        """
        try:
            info = self._json("GET", "/server_info")
        except SglangRuntimeError:
            self._discovered = False
            raise
        # Provisioned boot ceilings.
        self._provisioned.setdefault("max_running_requests", info.get("max_running_requests"))
        self._provisioned.setdefault("max_prefill_tokens", info.get("max_prefill_tokens"))
        self._provisioned.setdefault("pp_size", info.get("pp_size", 1))
        self._provisioned.setdefault("dp_size", info.get("dp_size", 1))
        # Live resolved scheduler config (includes runtime overrides).
        internal = info.get("internal_states")
        if isinstance(internal, list) and internal and isinstance(internal[0], dict):
            sched = internal[0]
            if "pp_max_micro_batch_size" in sched and sched.get("pp_max_micro_batch_size") is not None:
                self._policy["runtime_max_running_seqs"] = int(sched["pp_max_micro_batch_size"])
            if ("runtime_max_prefill_tokens" in sched
                    and sched.get("runtime_max_prefill_tokens") is not None):
                self._policy["runtime_max_prefill_tokens"] = int(sched["runtime_max_prefill_tokens"])
        if "max_running_requests" in self._provisioned and self._provisioned["max_running_requests"]:
            self._policy.setdefault(
                "runtime_max_running_seqs",
                int(self._provisioned["max_running_requests"] // max(int(self._provisioned.get("pp_size", 1)), 1)),
            )
        if "max_prefill_tokens" in self._provisioned and self._provisioned["max_prefill_tokens"]:
            self._policy.setdefault(
                "runtime_max_prefill_tokens",
                int(self._provisioned["max_prefill_tokens"]),
            )
        self._discovered = True

    def provisioned(self) -> Dict[str, Any]:
        if not self._discovered:
            try:
                self.refresh()
            except SglangRuntimeError:
                pass
        return dict(self._provisioned)

    # ---------------------------------------------------------------- policies
    def set_policy(self, name: str, value: int) -> bool:
        field = {
            "runtime_max_running_seqs": "pp_max_micro_batch_size",
            "runtime_max_prefill_tokens": "runtime_max_prefill_tokens",
        }.get(name)
        if field is None:
            raise SglangRuntimeError(f"no SGLang policy mapping for {name!r}")
        # /set_internal_state returns a JSON array of bools (one per rank).
        payload = self._json("POST", "/set_internal_state", {"server_args": {field: value}})
        ok = self._parse_updated(payload)
        # NOTE: observation is NOT shadowed here. get_policy() reads the live
        # value via refresh(); the setter confirmation only signals acceptance.
        return ok

    @staticmethod
    def _parse_updated(payload: Any) -> bool:
        """Robustly parse the set_internal_state confirmation.

        Live v0.5.18 returns ``[true]``/``[false]`` (msgspec list of bools per
        rank); tolerate ``{"updated": true}`` and ``[{"updated": true}]`` too.
        """
        if isinstance(payload, list):
            if not payload:
                return False
            first = payload[0]
            if isinstance(first, bool):
                return all(isinstance(x, bool) and x for x in payload)
            if isinstance(first, dict):
                return all(bool(x.get("updated")) for x in payload if isinstance(x, dict))
            return False
        if isinstance(payload, dict):
            return bool(payload.get("updated"))
        return False

    def get_policy(self, name: str) -> Any:
        """LIVE read of a runtime policy from /server_info (authoritative)."""
        self.refresh()
        return self._policy.get(name)

    # -------------------------------------------------------------- lifecycle
    def get_load(self) -> Dict[str, Any]:
        payload = self._json("GET", "/get_load")
        if isinstance(payload, list) and payload:
            return dict(payload[0])
        return {"num_reqs": 0, "num_waiting_reqs": 0, "num_tokens": 0}

    def pause_generation(self, mode: str = "retract") -> None:
        self._json("POST", "/pause_generation", {"mode": mode})

    def continue_generation(self, torch_empty_cache: bool = False) -> None:
        self._json("POST", "/continue_generation", {"torch_empty_cache": torch_empty_cache})

    def flush_cache(self) -> bool:
        """POST /flush_cache; True only when the scheduler actually flushed (200)."""
        status, raw = self._request("POST", "/flush_cache", {})
        ok = 200 <= status < 300
        self._last_flush_ok = ok
        return ok

    def flush_cache_ok(self) -> bool:
        return bool(self._last_flush_ok)

    # ---------------------------------------------------------------- identity
    def identity(self) -> Dict[str, Any]:
        return {
            "engine_instance_id": self._instance_id,
            "base_url": self.base_url,
            "control_pid": self._control_pid,
            "runtime_pids": dict(self._remote_pids),
        }