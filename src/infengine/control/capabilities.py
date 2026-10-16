from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class TransitionClass(str, Enum):
    """How a parameter may safely change on a running engine.

    HOT: apply immediately.
    DRAIN: stop admitting, drain in-flight work, reconfigure, resume.
    RESTART: worker/process must restart.
    REDEPLOY: hardware/model/topology changes require redeployment.
    """

    HOT = "HOT"
    DRAIN = "DRAIN"
    RESTART = "RESTART"
    REDEPLOY = "REDEPLOY"


class QuiescenceStrategy(str, Enum):
    """How a DRAIN-class transition handles in-flight work.

    This is metadata only: engine mechanics stay adapter-owned. The Reconciler
    uses it for observability / transition reporting / expected-duration
    accounting; it never branches on it to perform engine-specific procedures.

    WAIT_INFLIGHT:     stop admitting, hold KV, let in-flight work finish,
                       reconfigure, resume (vLLM pause_generation mode=wait).
    RETRACT_INFLIGHT:  stop admitting, release KV of in-flight requests, keep
                       logical output token ids, recompute that KV on resume
                       (SGLang pause_generation mode=retract).
    """

    WAIT_INFLIGHT = "WAIT_INFLIGHT"
    RETRACT_INFLIGHT = "RETRACT_INFLIGHT"


class ValidationError(ValueError):
    """A desired-state value failed typed validation."""


@dataclass(frozen=True)
class Capability:
    """A parameter a given engine adapter can mutate, and how.

    Capabilities are reported *by an adapter for a specific engine*; they are
    not universal facts about a parameter across every inference engine.

    ``dtype`` drives typed validation/normalization before a value enters
    Desired State: ``int`` / ``float`` are coerced, ``str`` is checked against
    ``choices``. ``min_value`` / ``max_value`` bound numeric values.

    The trailing fields are engine-capability metadata (M4B-B). They are
    additive and carry no Reconciler behavior: the Reconciler reads them for
    reporting/planning metadata but never branches engine-specific logic on
    them.
    """

    name: str
    transition: TransitionClass
    description: str
    dtype: str = "int"
    min_value: Any = None
    max_value: Any = None
    choices: Optional[tuple] = None
    supported: bool = True
    operation: bool = False  # one-shot operation (e.g. cache reset), not a persistent policy
    # M4B-B metadata ------------------------------------------------
    # How a DRAIN transition quiesces in-flight work (None for non-DRAIN caps).
    # WAIT_INFLIGHT: hold KV and let in-flight work finish (vLLM).
    # RETRACT_INFLIGHT: release in-flight KV, keep logical output, recompute on
    #   resume (SGLang pause_generation mode=retract). Only set where a drain
    #   actually transforms in-flight work into a usable state.
    quiescence_strategy: Optional[QuiescenceStrategy] = None
    # Quiescence precondition the operation requires of the engine. Smallest
    # honest vocabulary:
    #   None          -> no particular precondition
    #   "REQUIRE_IDLE"-> the engine must ALREADY be idle; a busy engine cannot be
    #                    transformed into a usable state (e.g. SGLang flush_cache
    #                    refuses to run while running OR waiting requests exist,
    #                    and retract merely parks requests in the waiting queue).
    #   "DRAIN_TO_IDLE"-> in-flight work may be drained (waited out) to reach
    #                    quiescence (e.g. vLLM reset_prefix_cache).
    quiescence_requirement: Optional[str] = None
    # Machine-readable semantic tag of the control (e.g. CONCURRENCY_UPPER_BOUND,
    # AGGREGATE_PREFILL_INPUT_ADMISSION_BUDGET_FIRST_REQUEST_UNBOUNDED).
    semantic: Optional[str] = None
    # Scope of the control (e.g. PER_DP_RANK / LOCAL_SCHEDULER).
    scope: Optional[str] = None
    # One-shot operations that require the engine to reach quiescence first.
    requires_quiescence: bool = False