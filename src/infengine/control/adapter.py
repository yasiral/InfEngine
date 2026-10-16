import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .capabilities import Capability, TransitionClass, ValidationError


@dataclass
class ApplyResult:
    applied: bool
    transition: TransitionClass
    reason: Optional[str] = None


@dataclass
class DrainResult:
    drained: bool
    owned: bool = True
    reason: Optional[str] = None


class InfengineAdapter:
    """Engine adapter for the built-in ``infengine.LLMEngine``.

    Owns the capabilities of *this* engine and the mechanics of applying each
    transition class. The reconciler never applies a change this adapter has
    not classified and made possible.
    """

    # M13-5B: the built-in engine applies HOT policies one at a time under the
    # engine lock; there is no multi-field engine request to coordinate.
    SUPPORTS_BATCH_APPLY = False
    APPLY_SEMANTICS = "COORDINATED_NON_ATOMIC"

    def __init__(self, engine):
        self.engine = engine
        self._drained = False
        self._drain_active = False
        # Protects acquisition/release of drain ownership. Never held while
        # waiting for quiescence, so a drain can be observed as in-progress.
        self._drain_lock = threading.Lock()
        self._capabilities = {
            "scheduler.max_batch": Capability(
                name="scheduler.max_batch",
                transition=TransitionClass.HOT,
                description=(
                    "Max sequences batched per scheduler step. Read live each "
                    "iteration in Scheduler.prefill_group/decode_group; safe to "
                    "change between iterations."
                ),
                dtype="int",
                min_value=1,
            ),
            "scheduler.chunk_size": Capability(
                name="scheduler.chunk_size",
                transition=TransitionClass.HOT,
                description=(
                    "Max prompt tokens prefilled per step. Read live in "
                    "LLMEngine._do_prefill; safe to change between iterations."
                ),
                dtype="int",
                min_value=1,
            ),
            "kv_store.block_size": Capability(
                name="kv_store.block_size",
                transition=TransitionClass.DRAIN,
                description=(
                    "KV block size; shapes the pooled tensors at init. Requires "
                    "drain + pool reinit (PagedKVStore.__init__)."
                ),
                dtype="int",
                min_value=1,
            ),
            "kv_store.num_blocks": Capability(
                name="kv_store.num_blocks",
                transition=TransitionClass.DRAIN,
                description=(
                    "Number of KV blocks; bounds pool capacity. Requires drain "
                    "+ pool reinit."
                ),
                dtype="int",
                min_value=1,
            ),
        }
        # runtime identity: stable while this adapter is bound to the engine
        self._pid = os.getpid()
        self._instance_id = f"pid{os.getpid()}:engine{id(engine)}"
        self._started_at = time.time()

    # ------------------------------------------------------------------ identity
    def identity(self) -> Dict[str, Any]:
        return {
            "worker_pid": self._pid,
            "engine_instance_id": self._instance_id,
            "engine_started_at": round(self._started_at, 3),
        }

    # ------------------------------------------------------------- capabilities
    def capabilities(self) -> Dict[str, Capability]:
        return dict(self._capabilities)

    def capability(self, name: str) -> Optional[Capability]:
        return self._capabilities.get(name)

    def coerce(self, name: str, value: Any) -> Any:
        """Validate + normalize a desired-state value before it is accepted.

        Raises ValidationError on invalid input; never mutates anything.
        """
        cap = self._capabilities.get(name)
        if cap is None:
            raise ValidationError(f"unknown parameter {name!r}")
        if cap.dtype == "int":
            if isinstance(value, bool):
                raise ValidationError(f"{name}: expected int, got bool")
            try:
                v = int(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"{name}: expected int, got {type(value).__name__!r}"
                )
            if cap.min_value is not None and v < cap.min_value:
                raise ValidationError(f"{name}: {v} below minimum {cap.min_value}")
            if cap.max_value is not None and v > cap.max_value:
                raise ValidationError(f"{name}: {v} above maximum {cap.max_value}")
            return v
        if cap.dtype == "float":
            if isinstance(value, bool):
                raise ValidationError(f"{name}: expected float, got bool")
            try:
                return float(value)
            except (TypeError, ValueError):
                raise ValidationError(
                    f"{name}: expected float, got {type(value).__name__!r}"
                )
        if cap.dtype == "str":
            if not isinstance(value, str):
                raise ValidationError(f"{name}: expected str, got {type(value).__name__!r}")
            if cap.choices is not None and value not in cap.choices:
                raise ValidationError(f"{name}: {value!r} not in {cap.choices}")
            return value
        raise ValidationError(f"{name}: unsupported dtype {cap.dtype!r}")

    # ---------------------------------------------------------------- observed
    def get(self, name: str) -> Any:
        if name == "scheduler.max_batch":
            return self.engine.scheduler.max_batch
        if name == "scheduler.chunk_size":
            return self.engine.scheduler.chunk_size
        if name == "kv_store.block_size":
            return self.engine.store.block_size
        if name == "kv_store.num_blocks":
            return self.engine.store.k_pool[0].shape[0] if self.engine.store.k_pool else 0
        raise KeyError(f"unknown parameter: {name}")

    def observed_state(self) -> Dict[str, Any]:
        return {name: self.get(name) for name in self._capabilities}

    def can_apply(self, name: str) -> bool:
        cap = self.capability(name)
        return bool(cap and cap.supported)

    def verify(self, name: str) -> bool:
        """Engine-observable verification for one-shot operations.

        This adapter has no operations; always True (policies verify via get())."""
        return True

    @property
    def admitting(self) -> bool:
        # Single authoritative admission truth lives on the engine; the
        # serve layer reads it here for fast 503s.
        return self.engine.admitting

    @property
    def drain_active(self) -> bool:
        return self._drain_active

    # ---------------------------------------------------------------- apply
    def apply(self, name: str, value: Any) -> ApplyResult:
        cap = self.capability(name)
        if cap is None:
            return ApplyResult(False, TransitionClass.HOT, reason=f"unknown parameter {name!r}")
        if not cap.supported:
            return ApplyResult(False, cap.transition, reason="capability not supported by adapter")
        if cap.transition is TransitionClass.DRAIN and not self._drained:
            return ApplyResult(False, cap.transition, reason="engine not drained; drain() first")
        if cap.transition is TransitionClass.RESTART:
            return ApplyResult(False, cap.transition, reason="worker restart required; not supported in this slice")
        if cap.transition is TransitionClass.REDEPLOY:
            return ApplyResult(False, cap.transition, reason="redeploy required; not supported in this slice")

        if name == "scheduler.max_batch":
            with self.engine._lock:
                self.engine.scheduler.max_batch = int(value)
        elif name == "scheduler.chunk_size":
            with self.engine._lock:
                self.engine.scheduler.chunk_size = int(value)
        elif name in ("kv_store.block_size", "kv_store.num_blocks"):
            bs = int(value) if name == "kv_store.block_size" else None
            nb = int(value) if name == "kv_store.num_blocks" else None
            ok = self.engine.reconfigure_kv(block_size=bs, num_blocks=nb)
            if not ok:
                return ApplyResult(False, cap.transition, reason="engine busy; drain pending")
        else:
            return ApplyResult(False, cap.transition, reason=f"no setter for {name!r}")
        return ApplyResult(True, cap.transition)

    def apply_many(self, overrides: Dict[str, Any]) -> ApplyResult:
        """Apply several HOT policies. The built-in engine applies them
        sequentially (each under the engine lock); a failure is returned, and
        the Reconciler restores the captured pre-state for the HOT parameters
        that actually changed."""
        transition = TransitionClass.HOT
        for name, value in overrides.items():
            res = self.apply(name, value)
            if not res.applied:
                return ApplyResult(False, res.transition, reason=res.reason)
            transition = res.transition
        return ApplyResult(True, transition)

    # ---------------------------------------------------------------- drain
    def drain(self, timeout: float = 30.0) -> DrainResult:
        """Drain-and-wait lifecycle: stop admitting, wait for quiescence.

        Ownership acquisition is atomic (under ``_drain_lock``): exactly one
        concurrent caller may own a drain; every other caller deterministically
        receives ``owned=False`` without altering drain state. Admission is
        closed on the engine (the single authoritative gate) under the engine
        lock. The wait loop holds no lock.

        On timeout admission is reopened and nothing is mutated (defined safe
        state). Returns ``drained=True`` once the scheduler is empty and reinit
        is safe; admission stays closed until :meth:`resume`.
        """
        with self._drain_lock:
            if self._drain_active:
                return DrainResult(False, owned=False, reason="drain already in progress")
            self._drain_active = True
        with self.engine._lock:
            self.engine._admitting = False
        try:
            deadline = time.monotonic() + timeout
            while True:
                with self.engine._lock:
                    busy = bool(self.engine.scheduler.requests)
                if not busy:
                    break
                if time.monotonic() >= deadline:
                    with self.engine._lock:
                        self.engine._admitting = True
                    with self._drain_lock:
                        self._drain_active = False
                    return DrainResult(
                        False,
                        owned=True,
                        reason=(
                            "timed out waiting for quiescence "
                            f"({len(self.engine.scheduler.requests)} in-flight)"
                        ),
                    )
                time.sleep(0.005)
            self._drained = True
            return DrainResult(True, owned=True)
        except BaseException:
            with self.engine._lock:
                self.engine._admitting = True
            with self._drain_lock:
                self._drain_active = False
            raise

    def resume(self) -> None:
        with self.engine._lock:
            self.engine._admitting = True
        self._drained = False
        with self._drain_lock:
            self._drain_active = False