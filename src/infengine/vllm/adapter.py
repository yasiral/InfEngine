"""VllmAdapter: engine adapter #2, over transport.

Implements the same responsibilities ``InfengineAdapter`` does, but through a
``VllmRuntimeClient`` transport rather than direct in-process object access.

M3D model:

  Persistent policies (HOT, converge through DesiredState/Reconciler):
    long_prefill_token_threshold        prefill behavior
    runtime_max_running_seqs            concurrency (<= provisioned max_num_seqs)
    runtime_max_scheduled_tokens        scheduler token budget (<= provisioned)

  One-shot operations (DRAIN, transient requests, NOT Desired State):
    reset_prefix_cache                  via pause_generation(wait) lifecycle

Capabilities report runtime-derived max constraints from the actual runtime
(provisioned boot ceilings), so planning rejects impossible values before the
EngineCore is called; EngineCore retains final validation.
"""

from dataclasses import replace
from typing import Any, Dict, Optional

from infengine.control import (
    ApplyResult,
    Capability,
    DrainResult,
    QuiescenceStrategy,
    TransitionClass,
    ValidationError,
)

from . import validation
from .runtime_client import VllmRuntimeClient


class VllmAdapter:
    # M13-5B: all HOT scheduler fields are sent in ONE set_runtime_scheduler_config
    # request, EngineCore validates the complete patch before mutating the live
    # scheduler, and InfEngine verifies every field and restores the captured
    # pre-state on failure -> CONTROL_PLANE_TRANSACTIONAL (never ENGINE_ATOMIC).
    SUPPORTS_BATCH_APPLY = True
    APPLY_SEMANTICS = "CONTROL_PLANE_TRANSACTIONAL"

    def __init__(self, client: VllmRuntimeClient):
        self._client = client
        runtime = client.get_runtime_scheduler_config()
        prov = runtime["provisioned"]
        self._capabilities = {
            "long_prefill_token_threshold": Capability(
                name="long_prefill_token_threshold",
                transition=TransitionClass.HOT,
                description=(
                    "Max prefill tokens scheduled for a single long prompt per "
                    "step (EngineCore-only scheduler policy, read live each "
                    "Scheduler.schedule())."
                ),
                dtype="int",
                min_value=0,
            ),
            "runtime_max_running_seqs": Capability(
                name="runtime_max_running_seqs",
                transition=TransitionClass.HOT,
                description=(
                    "Runtime admission policy: max concurrently running requests. "
                    "Bounded by the provisioned (boot-time) max_num_seqs; only "
                    "changes the scheduler admission gate."
                ),
                dtype="int",
                min_value=1,
                max_value=prov["max_num_seqs"],
            ),
            "runtime_max_scheduled_tokens": Capability(
                name="runtime_max_scheduled_tokens",
                transition=TransitionClass.HOT,
                description=(
                    "Runtime scheduler token budget (per-step). Bounded by the "
                    "boot-time ceiling; only changes the scheduler token budget, "
                    "never worker buffers/KV/CUDA-graph sizing."
                ),
                dtype="int",
                min_value=1,
                max_value=prov["max_num_scheduled_tokens"],
            ),
            "runtime_kv_watermark": Capability(
                name="runtime_kv_watermark",
                transition=TransitionClass.HOT,
                description=(
                    "Runtime KV admission headroom in [0.0, 1.0): reserves "
                    "watermark_blocks = int(v * num_gpu_blocks) as headroom for "
                    "waiting/preempted admissions. Physical KV capacity is "
                    "unchanged."
                ),
                dtype="float",
                min_value=0.0,
                max_value=1.0,
            ),
            "runtime_reserve_full_isl": Capability(
                name="runtime_reserve_full_isl",
                transition=TransitionClass.HOT,
                description=(
                    "Runtime long-request admission gate: if True, a request is "
                    "only admitted when its FULL input sequence fits the KV cache; "
                    "if False, only the first chunk must fit (chunked prefill). "
                    "Affects future admissions only."
                ),
                dtype="bool",
            ),
            "reset_prefix_cache": Capability(
                name="reset_prefix_cache",
                transition=TransitionClass.DRAIN,
                description=(
                    "One-shot operation: reset the prefix cache. Runs inside the "
                    "DRAIN lifecycle (pause_generation(mode='wait') -> quiescence "
                    "-> reset -> resume). NOT persistent Desired State."
                ),
                dtype="int",
                min_value=1,
                operation=True,
                quiescence_strategy=QuiescenceStrategy.WAIT_INFLIGHT,
                quiescence_requirement="DRAIN_TO_IDLE",
                semantic="PREFIX_CACHE_RESET",
                requires_quiescence=True,
            ),
        }
        # Transport-scoped support: report a capability as supported ONLY when
        # the transport actually implements the operations it needs. In-process
        # transports implement everything (unchanged); the HTTP control transport
        # excludes DRAIN/reset_prefix_cache/cache stats, so that capability is
        # advertised as unsupported rather than failing after admission.
        ops = client.supported_operations()
        hot_ok = {"get_runtime_scheduler_config", "set_runtime_scheduler_config"} <= ops
        reset_ok = {"reset_prefix_cache", "get_runtime_cache_stats"} <= ops
        self._capabilities = {
            name: replace(
                cap,
                supported=(reset_ok if name == "reset_prefix_cache" else hot_ok),
            )
            for name, cap in self._capabilities.items()
        }

    # ------------------------------------------------------------- capabilities
    def capabilities(self) -> Dict[str, Capability]:
        return dict(self._capabilities)

    def capability(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def coerce(self, name: str, value: Any) -> Any:
        cap = self._capabilities.get(name)
        if cap is None:
            raise ValidationError(f"unknown parameter {name!r}")
        if cap.dtype == "bool":
            if not isinstance(value, bool):
                raise ValidationError(f"{name}: expected bool, got {type(value).__name__!r}")
            return value
        if isinstance(value, bool):
            raise ValidationError(f"{name}: expected {cap.dtype}, got bool")
        if cap.dtype == "float":
            try:
                v = float(value)
            except (TypeError, ValueError):
                raise ValidationError(f"{name}: expected float, got {type(value).__name__!r}")
        else:
            try:
                v = int(value)
            except (TypeError, ValueError):
                raise ValidationError(f"{name}: expected int, got {type(value).__name__!r}")
        if cap.min_value is not None and v < cap.min_value:
            raise ValidationError(f"{name}: {v} below minimum {cap.min_value}")
        if cap.max_value is not None and v > cap.max_value:
            raise ValidationError(f"{name}: {v} above maximum {cap.max_value}")
        if name == "runtime_kv_watermark" and v >= 1.0:
            raise ValidationError(f"{name}: must be < 1.0")
        return v

    # ---------------------------------------------------------------- observed
    def get(self, name: str) -> Any:
        if name == "reset_prefix_cache":
            return 0  # one-shot operation: no persistent observed state (idle)
        if name not in self._capabilities:
            raise KeyError(f"unknown parameter: {name}")
        return self._client.get_runtime_scheduler_config()["policies"][name]

    def observed_state(self) -> Dict[str, Any]:
        return {name: self.get(name) for name in self._capabilities}

    # ---------------------------------------------------------------- apply
    def apply(self, name: str, value: Any) -> ApplyResult:
        cap = self.capability(name)
        if cap is None:
            return ApplyResult(False, TransitionClass.HOT, reason=f"unknown parameter {name!r}")
        try:
            if name == "reset_prefix_cache":
                self._client.reset_prefix_cache()
            else:
                self._client.set_runtime_scheduler_config({name: value})
        except Exception as exc:
            return ApplyResult(False, cap.transition, reason=str(exc))
        return ApplyResult(True, cap.transition)

    def apply_many(self, overrides: Dict[str, Any]) -> ApplyResult:
        """Apply several HOT scheduler policies in ONE coordinated engine
        request. The engine validates the complete patch first (types) and only
        then mutates; InfEngine verifies each field afterwards and restores the
        captured pre-state on failure. One-shot operations are never batched."""
        if any(name == "reset_prefix_cache" for name in overrides):
            return ApplyResult(False, TransitionClass.DRAIN,
                               reason="one-shot operations are not batch-applied")
        try:
            self._client.set_runtime_scheduler_config(dict(overrides))
        except Exception as exc:
            return ApplyResult(False, TransitionClass.HOT, reason=str(exc))
        return ApplyResult(True, TransitionClass.HOT)

    # ------------------------------------------------------------- verify (operations)
    def verify(self, name: str) -> bool:
        """Engine-observable verification for one-shot operations."""
        if name == "reset_prefix_cache":
            return self._client.get_runtime_cache_stats()["prefix_cache_entries"] == 0
        return True

    # ---------------------------------------------------------------- identity
    def identity(self) -> Dict[str, Any]:
        return self._client.identity()

    # ---------------------------------------------------------------- drain
    def drain(self, timeout: float = 30.0) -> DrainResult:
        """Stop admitting and wait for quiescence (pause_generation mode=wait)."""
        try:
            self._client.pause_generation(mode="wait", clear_cache=False)
        except Exception as exc:
            return DrainResult(False, owned=True, reason=str(exc))
        return DrainResult(True, owned=True)

    def resume(self) -> None:
        # A resume failure must propagate so the Reconciler can fail the DRAIN
        # lifecycle (the engine may remain paused). Do NOT swallow it here.
        self._client.resume_generation()

    @property
    def drain_active(self) -> bool:
        return False

    @property
    def admitting(self) -> bool:
        # The vLLM adapter never owns admission: drain is an explicit engine
        # operation (pause_generation) and drain_active reflects it, so admission
        # is the inverse. On the HTTP transport drain is unsupported and this
        # stays True.
        return not self.drain_active