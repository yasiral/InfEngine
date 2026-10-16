"""SglangAdapter: engine adapter #3, over the native SGLang v0.5.18 transport.

Implements the same responsibilities ``VllmAdapter`` does, but through a
``SglangRuntimeClient`` talking to the SGLang HTTP admin surface
(HTTP -> TokenizerManager -> ZMQ/msgpack -> Scheduler). No new RPC protocol.

M4B-B model (proven operating envelope: SGLang v0.5.18, PP=1, DP=1, text/token
input, no speculative decoding, no disaggregated modes):

  Persistent policies (HOT, converge through DesiredState/Reconciler):
    runtime_max_running_seqs         pp_max_micro_batch_size
                                     semantic = CONCURRENCY_UPPER_BOUND
                                     scope = PER_DP_RANK / LOCAL_SCHEDULER
    runtime_max_prefill_tokens       runtime_max_prefill_tokens (patched)
                                     semantic =
                                       AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET
                                       FIRST_REQUEST_UNBOUNDED
                                     scope = LOCAL_SCHEDULER

  One-shot operations (DRAIN, transient requests, NOT Desired State):
    engine_cache_flush               flush_cache()
                                     quiescence_requirement = REQUIRE_IDLE
                                     (engine must ALREADY be idle; a busy engine
                                     -> BLOCKED, no retract side effects)
    prefix_cache_reset               UNSUPPORTED (rejected at planning; never
                                     enters the DRAIN lifecycle)

Semantic notes (source-backed, DeepSeek V4 Pro review):

  * ``pp_max_micro_batch_size`` gates admission as
      min(pp_max_micro_batch_size - running_bs, req_to_token_pool.available_size())
    so it is a concurrency UPPER BOUND, not an exact guarantee, and is scoped
    per DP rank / local scheduler.

  * ``runtime_max_prefill_tokens`` is the aggregate prefill input-token
    admission budget; the first request of a scheduling pass is unbounded by it
    (schedule_policy.add_one_req gates only when can_run_list is non-empty). It
    is NOT a strict per-request cap, NOT a total scheduler-token budget, NOT an
    exact equivalent of vLLM's runtime_max_scheduled_tokens. The canonical patch
    exposes it through ``/server_info -> internal_states`` so readback is the
    ACTUAL scheduler value (no shadow state).

  * ENGINE_CACHE_FLUSH requires an IDLE engine: SGLang's flush_cache() refuses
    to run while running OR waiting requests exist, and pause_generation(
    mode="retract") does NOT transform a busy engine into a flushable state (it
    parks requests in the waiting queue). Hence REQUIRE_IDLE, NOT
    RETRACT_INFLIGHT. SGLang retract remains available as a strategy where a
    drain genuinely transforms in-flight work; it is not attached to
    ENGINE_CACHE_FLUSH.
"""

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
from .runtime_client import SglangRuntimeClient


class SglangAdapter:
    """Control-plane adapter for a remote SGLang v0.5.18 runtime."""

    # M13-5B: the SGLang admin transport applies one scheduler field per
    # set_internal_state request, so a multi-control policy is COORDINATED
    # (InfEngine validates all first and restores the captured pre-state on
    # failure) but NOT engine-atomic.
    SUPPORTS_BATCH_APPLY = False
    APPLY_SEMANTICS = "COORDINATED_NON_ATOMIC"

    def __init__(self, client: SglangRuntimeClient):
        self._client = client
        self._resume_error: Optional[str] = None
        prov = client.provisioned()
        if not prov.get("max_running_requests"):
            raise ValidationError(
                "SGLang adapter requires provisioned max_running_requests "
                "(discovered from /server_info or supplied at construction)"
            )
        if not prov.get("max_prefill_tokens"):
            raise ValidationError(
                "SGLang adapter requires provisioned max_prefill_tokens "
                "(discovered from /server_info or supplied at construction)"
            )
        self._capabilities = {
            "runtime_max_running_seqs": Capability(
                name="runtime_max_running_seqs",
                transition=TransitionClass.HOT,
                description=(
                    "Runtime concurrency upper bound, mapped to SGLang "
                    "pp_max_micro_batch_size (PP=1). Admission is constrained by "
                    "min(pp_max_micro_batch_size - running_bs, "
                    "req_to_token_pool.available_size()), so this is an UPPER "
                    "BOUND, not an exact concurrency guarantee, and is scoped per "
                    "DP rank / local scheduler. Never resizes request pools, KV "
                    "pools, buffers or CUDA graphs."
                ),
                dtype="int",
                min_value=1,
                max_value=int(prov["max_running_requests"]),
                semantic="CONCURRENCY_UPPER_BOUND",
                scope="PER_DP_RANK",
            ),
            "runtime_max_prefill_tokens": Capability(
                name="runtime_max_prefill_tokens",
                transition=TransitionClass.HOT,
                description=(
                    "Runtime aggregate prefill input-token admission budget, "
                    "mapped to SGLang runtime_max_prefill_tokens (patched, "
                    "scheduler-local). The first request of a scheduling pass is "
                    "unbounded by it. NOT a strict per-request cap, NOT a total "
                    "scheduler-token budget, NOT a decode budget. Bounded by the "
                    "boot provisioned max_prefill_tokens; never alters worker "
                    "provisioning, KV allocation, CUDA-graph or request-pool "
                    "sizing."
                ),
                dtype="int",
                min_value=1,
                max_value=int(prov["max_prefill_tokens"]),
                semantic=(
                    "AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_"
                    "FIRST_REQUEST_UNBOUNDED"
                ),
                scope="LOCAL_SCHEDULER",
            ),
            "engine_cache_flush": Capability(
                name="engine_cache_flush",
                transition=TransitionClass.DRAIN,
                description=(
                    "One-shot operation: SGLang flush_cache(). Broader than a "
                    "prefix-cache reset: clears radix cache, request->token "
                    "state, KV allocator state, grammar manager, metrics, and "
                    "draft/spec cache where applicable. REQUIRES the engine to "
                    "ALREADY be idle: flush_cache() returns HTTP 400 whenever "
                    "running OR waiting requests exist, and SGLang's "
                    "pause_generation(mode='retract') does NOT make a busy "
                    "engine flushable (it parks requests in the waiting queue). "
                    "A busy engine -> BLOCKED, no retract side effects. NOT "
                    "persistent Desired State."
                ),
                dtype="int",
                min_value=1,
                operation=True,
                # RETRACT_INFLIGHT is NOT the strategy here: retract does not
                # transform a busy engine into a flushable state. The honest
                # precondition is REQUIRE_IDLE.
                quiescence_requirement="REQUIRE_IDLE",
                semantic="ENGINE_CACHE_FLUSH",
                requires_quiescence=True,
            ),
            "prefix_cache_reset": Capability(
                name="prefix_cache_reset",
                transition=TransitionClass.DRAIN,
                description=(
                    "Unsupported on SGLang v0.5.18. The native flush_cache() is a "
                    "much broader engine cache flush and must NOT be conflated "
                    "with a narrow prefix-cache reset. Use engine_cache_flush."
                ),
                dtype="int",
                min_value=1,
                supported=False,
                operation=True,
                quiescence_requirement="REQUIRE_IDLE",
                semantic="PREFIX_CACHE_RESET",
                requires_quiescence=True,
            ),
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
        if not cap.supported:
            # Reject unsupported capabilities at planning time so they NEVER
            # enter the DRAIN lifecycle (no pause/flush/resume side effects).
            raise ValidationError(
                f"{name}: capability not supported by the SGLang adapter"
            )
        if isinstance(value, bool):
            raise ValidationError(f"{name}: expected int, got bool")
        if cap.dtype == "int":
            # STRICT: accept genuine int only. Reject str/float/None. Never
            # coerce (int(2.5) -> 2 would silently change operator intent).
            if not isinstance(value, int):
                raise ValidationError(
                    f"{name}: expected int, got {type(value).__name__!r}"
                )
            v = value
        else:
            raise ValidationError(f"{name}: unsupported dtype {cap.dtype!r}")
        if cap.min_value is not None and v < cap.min_value:
            raise ValidationError(f"{name}: {v} below minimum {cap.min_value}")
        if cap.max_value is not None and v > cap.max_value:
            raise ValidationError(f"{name}: {v} above maximum {cap.max_value}")
        return v

    # ---------------------------------------------------------------- observed
    def get(self, name: str) -> Any:
        # one-shot operations: no persistent observed state (idle)
        if name in ("engine_cache_flush", "prefix_cache_reset"):
            return 0
        if name not in self._capabilities:
            raise KeyError(f"unknown parameter: {name}")
        # LIVE read: get_policy() refreshes from /server_info (authoritative).
        return self._client.get_policy(name)

    def observed_state(self) -> Dict[str, Any]:
        return {name: self.get(name) for name in self._capabilities}

    # ---------------------------------------------------------------- apply
    def apply(self, name: str, value: Any) -> ApplyResult:
        cap = self.capability(name)
        if cap is None:
            return ApplyResult(False, TransitionClass.HOT, reason=f"unknown parameter {name!r}")
        if not cap.supported:
            return ApplyResult(
                False, cap.transition,
                reason="PREFIX_CACHE_RESET is not supported by the SGLang adapter; use engine_cache_flush",
            )
        try:
            if name == "runtime_max_running_seqs":
                ok = self._client.set_policy("runtime_max_running_seqs", int(value))
                return ApplyResult(ok, cap.transition,
                                   None if ok else "SGLang set_internal_state returned false")
            if name == "runtime_max_prefill_tokens":
                ok = self._client.set_policy("runtime_max_prefill_tokens", int(value))
                return ApplyResult(ok, cap.transition,
                                   None if ok else "SGLang set_internal_state returned false")
            if name == "engine_cache_flush":
                ok = self._client.flush_cache()
                return ApplyResult(ok, cap.transition,
                                   None if ok else "SGLang flush_cache returned HTTP 400 (not quiescent)")
        except Exception as exc:
            return ApplyResult(False, cap.transition, reason=str(exc))
        return ApplyResult(False, cap.transition, reason=f"no setter for {name!r}")

    def apply_many(self, overrides: Dict[str, Any]) -> ApplyResult:
        """Apply several HOT policies sequentially (the SGLang admin transport
        has one field per request). A failure aborts the remaining fields and is
        returned so the Reconciler can restore the captured pre-state."""
        for name, value in overrides.items():
            res = self.apply(name, value)
            if not res.applied:
                return ApplyResult(False, res.transition, reason=res.reason)
        return ApplyResult(True, TransitionClass.HOT)

    # ------------------------------------------------------------- verify (operations)
    def verify(self, name: str) -> bool:
        """Engine-observable verification for one-shot operations."""
        if name == "engine_cache_flush":
            # HTTP 200 from /flush_cache is the scheduler's own confirmation that
            # it flushed (is_fully_idle was true and the reset ran).
            return bool(self._client.flush_cache_ok())
        return True

    # ---------------------------------------------------------------- identity
    def identity(self) -> Dict[str, Any]:
        return self._client.identity()

    # ---------------------------------------------------------------- drain
    def drain(self, timeout: float = 30.0) -> DrainResult:
        """ENGINE_CACHE_FLUSH quiescence precondition: the engine must ALREADY
        be idle (REQUIRE_IDLE).

        SGLang's ``flush_cache()`` refuses to run while running OR waiting
        requests exist, and ``pause_generation(mode="retract")`` does NOT make a
        busy engine flushable — it merely parks requests in the waiting queue,
        keeping the engine non-quiescent. So a busy engine is BLOCKED here
        WITHOUT any pause/retract side effect (no pointless retract ->
        waiting -> resume cycle). ``timeout`` is retained for interface
        compatibility but is not used to wait: this operation does not transform
        a busy engine into a usable state.
        """
        try:
            load = self._client.get_load()
        except Exception as exc:
            return DrainResult(False, owned=True, reason=str(exc))
        running = int(load.get("num_reqs", 0)) - int(load.get("num_waiting_reqs", 0))
        waiting = int(load.get("num_waiting_reqs", 0))
        if running == 0 and waiting == 0:
            return DrainResult(True, owned=True)
        return DrainResult(
            False, owned=True,
            reason=(
                f"engine not idle (running={running}, waiting={waiting}); "
                "ENGINE_CACHE_FLUSH requires an idle engine"
            ),
        )

    def resume(self) -> None:
        """Resume admission after a drain cycle.

        A failure propagates so the Reconciler can fail the DRAIN lifecycle
        (the engine may remain paused). ``_resume_error`` is recorded for
        operator observability before re-raising.
        """
        try:
            self._client.continue_generation()
            self._resume_error = None
        except Exception as exc:
            self._resume_error = str(exc)
            raise

    @property
    def admitting(self) -> bool:
        """The remote SGLang server owns admission; the control plane holds no
        separate in-process admission gate. Reported True for the serve layer's
        /health and admission fast-path. Engine-idleness for operations is
        handled by the DRAIN (REQUIRE_IDLE) precondition, not this flag."""
        return True

    @property
    def drain_active(self) -> bool:
        return False