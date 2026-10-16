"""M7A: deterministic concurrency policy (HOLD / PROPOSE_UP / PROPOSE_DOWN).

The policy is the *decision* half of the supervisory controller: given a
factual :class:`~infengine.supervisor.signals.SignalWindow`, the current
``runtime_concurrency`` value and the engine's capability bounds, it returns a
bounded proposal. It performs **no I/O and no mutation** — it is a pure,
deterministic function of its inputs (identical inputs -> identical decision).

Decision procedure (exhaustive, ordered; every branch states the numeric facts):

  1. **Insufficient telemetry** — the configured demand metrics must each have
     at least ``min_samples`` FRESH samples in the window, else HOLD.
  2. **Unknown current value / bounds** — HOLD (cannot make a bounded proposal).
  3. **Guardrail breach** — if an ENABLED guardrail's fresh window MAXIMUM
     exceeds its configured limit, PROPOSE_DOWN one step (HOLD at the lower
     bound). This is the only source of PROPOSE_DOWN: no demand alone never
     reduces concurrency (an idle admission ceiling consumes no capacity).
  4. **Saturation with waiting demand** — running at the current value (within
     ``running_bound_slack``) and waiting above ``waiting_high`` -> a scale-up
     proposal is *otherwise eligible*; it is emitted only if every guardrail
     configured ``required_for_scale_up`` has FRESH, passing telemetry (fail
     closed). At the upper bound, or with a required guardrail unavailable or
     breaching, HOLD.
  5. Otherwise — HOLD (covers no-demand and within-band).

Every threshold/horizon/step is a field of :class:`SupervisorConfig`; there are
no hidden constants. Only ``runtime_concurrency`` (internal key
``runtime_max_running_seqs``) is modelled here. Decision evidence is FRESH-only;
stale samples are recorded but never used.
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from .signals import SignalWindow

# Explicit supported actuator-value model for the lifecycle (M7D1). Policies are
# responsible for valid transitions; a bool value carries no numeric UP/DOWN
# semantics of its own.
ActuatorValue = Union[int, float, bool]


class PolicyEffect(str, Enum):
    """The actuator-specific effect of a proposed change (M7D2).

    NOT derived from numeric UP/DOWN: for some actuators an increase is more
    PERMISSIVE, for others more CONSERVATIVE. Each policy declares its
    orientation.
    """

    MORE_PERMISSIVE = "MORE_PERMISSIVE"
    MORE_CONSERVATIVE = "MORE_CONSERVATIVE"
    NON_MONOTONIC = "NON_MONOTONIC"


class DecisionKind(str, Enum):
    """The outcomes a policy can produce.

    ``PROPOSE_CHANGE`` is for actuators with no numeric increase/decrease
    semantics (booleans / non-monotonic controls); it carries no implicit
    MORE_PERMISSIVE/MORE_CONSERVATIVE orientation.
    """

    HOLD = "HOLD"
    PROPOSE_UP = "PROPOSE_UP"
    PROPOSE_DOWN = "PROPOSE_DOWN"
    PROPOSE_CHANGE = "PROPOSE_CHANGE"


@dataclass(frozen=True)
class CapabilityBounds:
    """Inclusive bounds of the controlled capability (int/float/bool)."""

    min_value: ActuatorValue
    max_value: ActuatorValue

    def to_dict(self) -> Dict[str, Any]:
        return {"min": self.min_value, "max": self.max_value}


@dataclass
class Guardrail:
    """One safety guardrail.

    ``metric`` names a snapshot metric whose FRESH window MAXIMUM is compared
    against ``max_value``. ``enabled`` false removes it from evaluation
    entirely. ``required_for_scale_up`` makes the guardrail fail closed: if it
    is enabled but its fresh telemetry is unavailable/insufficient, a scale-up
    proposal becomes HOLD. A guardrail that is enabled and breached still
    produces a bounded PROPOSE_DOWN regardless of ``required_for_scale_up``.
    """

    name: str
    metric: str
    max_value: float
    enabled: bool = True
    required_for_scale_up: bool = False


@dataclass
class Guardrails:
    """Operating guardrails relevant to concurrency (explicit + configurable)."""

    latency: Guardrail = field(default_factory=lambda: Guardrail(
        name="latency", metric="ttft_p95_s", max_value=0.5,
        enabled=True, required_for_scale_up=True,
    ))
    kv_usage: Guardrail = field(default_factory=lambda: Guardrail(
        name="kv_usage", metric="kv_usage", max_value=0.9,
        enabled=True, required_for_scale_up=True,
    ))

    def all_guardrails(self) -> Tuple[Guardrail, ...]:
        return (self.latency, self.kv_usage)

    def enabled_guardrails(self) -> Tuple[Guardrail, ...]:
        return tuple(g for g in self.all_guardrails() if g.enabled)


@dataclass
class ConcurrencyObjective:
    """Configurable operating objective thresholds for concurrency.

    The objective is: keep the current concurrency unless waiting demand
    justifies a bounded increase, or a guardrail breach justifies a bounded
    decrease. ``waiting_high`` is the waiting backlog above which demand is
    present; ``waiting_low`` the backlog at/below which demand is considered
    absent (informational; absent demand never decreases concurrency);
    ``running_bound_slack`` relaxes the "running at the bound" test;
    ``step_up`` / ``step_down`` bound each proposal to one step.
    """

    waiting_high: float = 2.0
    waiting_low: float = 0.0
    running_bound_slack: int = 0
    step_up: int = 1
    step_down: int = 1


@dataclass
class TokenBudgetObjective:
    """Configurable operating objective for a token-budget actuator.

    Used by :class:`TokenBudgetPolicy` (e.g. vLLM ``runtime_max_scheduled_tokens``).
    Generic queueing is NOT attribution: scale-up requires **persistent evidence
    that the token budget is actually being reached** — the windowed tokens per
    scheduling step (``iteration_tokens_p95``) at/above
    ``budget_utilization_threshold`` of the active budget — while waiting demand
    exists and all required guardrails are available+passing. A guardrail breach
    yields a bounded decrease. ``step_up`` / ``step_down`` bound each proposal to
    one step. No throughput score is used.
    """

    budget_utilization_threshold: float = 0.9
    waiting_high: float = 2.0
    step_up: int = 256
    step_down: int = 256


@dataclass
class KvWatermarkObjective:
    """Operating objective for the vLLM KV watermark (float [0, 1)).

    Asymmetric by evidence: a **conservative** increase (reserve more KV headroom)
    is supported on factual KV/preemption safety pressure; a **permissive**
    decrease requires factual evidence that the watermark itself is constraining
    admission, which vLLM does not expose, so the permissive direction is
    MANUAL_ONLY (the policy never decreases; generic queueing is never used).
    """

    kv_pressure_threshold: float = 0.9   # kv_usage window max at/above -> pressure
    preemptions_min: float = 1.0         # window preemption counter delta at/above
    step_up: float = 0.05                # conservative step (increase)
    max_watermark: float = 0.9           # strictly < 1.0 cap (config)


@dataclass
class SupervisorConfig:
    """All horizons, thresholds, guardrails and steps for the supervisor.

    Nothing in the policy is a hidden constant: the window, sample floor and
    cooldown live here alongside the objective and guardrails.
    """

    window_s: float = 30.0
    min_samples: int = 3
    cooldown_s: float = 60.0
    signal_metrics: Tuple[str, ...] = ("requests_running", "requests_waiting")
    # Policy-specific objective: a ``ConcurrencyObjective`` for the concurrency
    # policy, a ``TokenBudgetObjective`` for the token-budget policy. The
    # algorithm is never generic; each policy reads its own objective type.
    objective: Any = field(default_factory=ConcurrencyObjective)
    guardrails: Guardrails = field(default_factory=Guardrails)

    def effective_metrics(self) -> Tuple[str, ...]:
        """Base signal metrics plus every enabled guardrail metric."""
        metrics = list(self.signal_metrics)
        for guardrail in self.guardrails.enabled_guardrails():
            if guardrail.metric not in metrics:
                metrics.append(guardrail.metric)
        return tuple(dict.fromkeys(metrics))


@dataclass(frozen=True)
class GuardrailResult:
    """Outcome of one guardrail over the window.

    ``available`` false means no FRESH sample was present; ``passed`` is then
    ``None`` (never treated as PASS). ``enabled`` / ``required_for_scale_up``
    are preserved so the record shows exactly how the guardrail was configured.
    """

    name: str
    metric: str
    comparator: str
    limit: float
    observed: Optional[float]
    available: bool
    passed: Optional[bool]
    enabled: bool
    required_for_scale_up: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "comparator": self.comparator,
            "limit": self.limit,
            "observed": self.observed,
            "available": self.available,
            "passed": self.passed,
            "enabled": self.enabled,
            "required_for_scale_up": self.required_for_scale_up,
        }


@dataclass(frozen=True)
class PolicyEvaluation:
    """Raw, deterministic policy output for one window (pre-supervisor gates)."""

    kind: DecisionKind
    proposed_value: Optional[ActuatorValue]
    reasons: List[str]
    guardrails: List[GuardrailResult]
    evidence: Dict[str, Any]
    effect: Optional[PolicyEffect] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind.value,
            "proposed_value": self.proposed_value,
            "effect": self.effect.value if self.effect else None,
            "reasons": list(self.reasons),
            "guardrails": [g.to_dict() for g in self.guardrails],
            "evidence": dict(self.evidence),
        }


def _effect_of(kind: DecisionKind, increase_is_permissive: bool) -> Optional[PolicyEffect]:
    if kind is DecisionKind.PROPOSE_UP:
        return (PolicyEffect.MORE_PERMISSIVE if increase_is_permissive
                else PolicyEffect.MORE_CONSERVATIVE)
    if kind is DecisionKind.PROPOSE_DOWN:
        return (PolicyEffect.MORE_CONSERVATIVE if increase_is_permissive
                else PolicyEffect.MORE_PERMISSIVE)
    return None


class _ActuatorPolicy:
    """Base for numeric actuator policies; attaches the actuator-specific effect.

    ``increase_is_permissive`` declares the orientation:
      * True  (concurrency, token budget): increase = MORE_PERMISSIVE.
      * False (runtime_kv_watermark): increase = MORE_CONSERVATIVE.
    The effect is NEVER derived from numeric UP/DOWN generically.
    """

    increase_is_permissive: bool = True

    def effect_for(self, kind: DecisionKind) -> Optional[PolicyEffect]:
        return _effect_of(kind, self.increase_is_permissive)

    def evaluate(self, signals: SignalWindow, *, current_value, bounds) -> PolicyEvaluation:
        result = self._evaluate(signals, current_value=current_value, bounds=bounds)
        return replace(result, effect=self.effect_for(result.kind))

    def _evaluate(self, signals: SignalWindow, *, current_value, bounds) -> PolicyEvaluation:
        raise NotImplementedError  # pragma: no cover

    def _guardrail_results(self, signals: SignalWindow) -> List[GuardrailResult]:
        results: List[GuardrailResult] = []
        for guardrail in self.config.guardrails.all_guardrails():
            if not guardrail.enabled:
                results.append(GuardrailResult(
                    name=guardrail.name, metric=guardrail.metric, comparator="<=",
                    limit=guardrail.max_value, observed=None, available=False,
                    passed=None, enabled=False,
                    required_for_scale_up=guardrail.required_for_scale_up,
                ))
                continue
            aggregate = signals.get(guardrail.metric)
            observed = aggregate.maximum if aggregate is not None and aggregate.samples > 0 else None
            available = observed is not None
            passed = (observed <= guardrail.max_value) if available else None
            results.append(GuardrailResult(
                name=guardrail.name, metric=guardrail.metric, comparator="<=",
                limit=guardrail.max_value, observed=observed, available=available,
                passed=passed, enabled=True,
                required_for_scale_up=guardrail.required_for_scale_up,
            ))
        return results


class ConcurrencyPolicy(_ActuatorPolicy):
    """Deterministic HOLD / PROPOSE_UP / PROPOSE_DOWN policy for concurrency."""

    def __init__(self, config: SupervisorConfig):
        self.config = config

    # --------------------------------------------------------------- evaluate
    def _evaluate(
        self,
        signals: SignalWindow,
        *,
        current_value: Optional[ActuatorValue],
        bounds: Optional[CapabilityBounds],
    ) -> PolicyEvaluation:
        config = self.config
        obj = config.objective

        running = signals.get("requests_running")
        waiting = signals.get("requests_waiting")
        kv = signals.get("kv_usage")
        latency = signals.get(config.guardrails.latency.metric)
        guardrails = self._guardrail_results(signals)
        evidence: Dict[str, Any] = {
            "requests_running_median": running.median if running else None,
            "requests_running_samples": running.samples if running else 0,
            "requests_waiting_median": waiting.median if waiting else None,
            "requests_waiting_samples": waiting.samples if waiting else 0,
            "kv_usage_max": kv.maximum if kv else None,
            "latency_metric": config.guardrails.latency.metric,
            "latency_max_observed": latency.maximum if latency else None,
            "current_value": current_value,
            "bounds_min": bounds.min_value if bounds else None,
            "bounds_max": bounds.max_value if bounds else None,
        }

        # 1. Telemetry sufficiency for the demand metrics (FRESH-only samples).
        for name in ("requests_running", "requests_waiting"):
            agg = signals.get(name)
            have = agg.samples if agg else 0
            if have < config.min_samples:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [f"insufficient telemetry: {name} fresh samples={have} "
                     f"< min_samples={config.min_samples}"],
                    guardrails, evidence,
                )

        # 2. Current value and bounds must be known to propose a bounded step.
        if current_value is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                ["current runtime_concurrency value unknown"], guardrails, evidence,
            )
        if bounds is None or bounds.min_value is None or bounds.max_value is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                ["capability bounds unknown"], guardrails, evidence,
            )

        # 3. Guardrail breach -> bounded PROPOSE_DOWN (safe direction).
        breaches = [g for g in guardrails if g.enabled and g.passed is False]
        if breaches:
            breach_reasons = [
                f"guardrail breach: {g.metric} fresh max={g.observed} > {g.comparator} "
                f"limit={g.limit}"
                for g in breaches
            ]
            if current_value > bounds.min_value:
                proposed = max(bounds.min_value, current_value - obj.step_down)
                return PolicyEvaluation(
                    DecisionKind.PROPOSE_DOWN, proposed,
                    breach_reasons + [f"propose down {current_value} -> {proposed}"],
                    guardrails, evidence,
                )
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                breach_reasons + [f"at min bound {bounds.min_value}; cannot decrease"],
                guardrails, evidence,
            )

        running_median = running.median
        waiting_median = waiting.median

        # 4. Saturation with waiting demand (scale-up, fail closed on required guardrails).
        at_bound = running_median is not None and (
            running_median >= (current_value - obj.running_bound_slack)
        )
        waiting_demand = waiting_median is not None and waiting_median > obj.waiting_high
        if at_bound and waiting_demand:
            if current_value >= bounds.max_value:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [
                        f"saturation with waiting demand at max bound {bounds.max_value}: "
                        f"running median={running_median}, waiting median={waiting_median}"
                    ],
                    guardrails, evidence,
                )
            required_missing = [
                g for g in guardrails
                if g.enabled and g.required_for_scale_up and not g.available
            ]
            if required_missing:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [
                        "scale-up blocked: required guardrail(s) unavailable: "
                        + ", ".join(f"{g.metric}(fresh samples=0)" for g in required_missing)
                    ] + [
                        f"saturation with waiting demand: running median={running_median} "
                        f">= current={current_value} - slack={obj.running_bound_slack}; "
                        f"waiting median={waiting_median} > waiting_high={obj.waiting_high}"
                    ],
                    guardrails, evidence,
                )
            proposed = min(bounds.max_value, current_value + obj.step_up)
            return PolicyEvaluation(
                DecisionKind.PROPOSE_UP, proposed,
                [
                    f"saturation with waiting demand: running median={running_median} "
                    f">= current={current_value} - slack={obj.running_bound_slack}; "
                    f"waiting median={waiting_median} > waiting_high={obj.waiting_high}",
                    f"propose up {current_value} -> {proposed}",
                ],
                guardrails, evidence,
            )

        # 5. Otherwise HOLD (no demand and within-band both land here).
        no_demand = waiting_median is not None and waiting_median <= obj.waiting_low
        underused = running_median is not None and running_median < current_value
        if no_demand and underused:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                [
                    f"no waiting demand: waiting median={waiting_median} "
                    f"<= waiting_low={obj.waiting_low}, running median={running_median} "
                    f"< current={current_value}; an idle admission ceiling does not "
                    "consume provisioned capacity and is not decreased on demand "
                    "absence alone"
                ],
                guardrails, evidence,
            )
        return PolicyEvaluation(
            DecisionKind.HOLD, None,
            [
                f"within operating band: running median={running_median}, "
                f"waiting median={waiting_median}, current={current_value}, "
                f"bounds=[{bounds.min_value},{bounds.max_value}]"
            ],
            guardrails, evidence,
        )


class TokenBudgetPolicy(_ActuatorPolicy):
    """Deterministic HOLD / PROPOSE_UP / PROPOSE_DOWN policy for a token budget.

    Actuator-specific (NOT a generic algorithm): the token budget's own bound is
    not directly observable in telemetry, so scale-up eligibility is demand-based
    — persistent schedulable demand (waiting present in every fresh sample and a
    median above ``waiting_high``) with required guardrails available+passing —
    while a guardrail breach yields a bounded decrease. No throughput score.
    """

    def __init__(self, config: SupervisorConfig):
        self.config = config

    # --------------------------------------------------------------- evaluate
    def _evaluate(
        self,
        signals: SignalWindow,
        *,
        current_value: Optional[ActuatorValue],
        bounds: Optional[CapabilityBounds],
    ) -> PolicyEvaluation:
        config = self.config
        obj = config.objective

        running = signals.get("requests_running")
        waiting = signals.get("requests_waiting")
        kv = signals.get("kv_usage")
        latency = signals.get(config.guardrails.latency.metric)
        iteration = signals.get("iteration_tokens_p95")
        guardrails = self._guardrail_results(signals)
        evidence: Dict[str, Any] = {
            "requests_running_median": running.median if running else None,
            "requests_running_samples": running.samples if running else 0,
            "requests_waiting_median": waiting.median if waiting else None,
            "requests_waiting_samples": waiting.samples if waiting else 0,
            "iteration_tokens_p95_median": iteration.median if iteration else None,
            "iteration_tokens_p95_samples": iteration.samples if iteration else 0,
            "kv_usage_max": kv.maximum if kv else None,
            "latency_metric": config.guardrails.latency.metric,
            "latency_max_observed": latency.maximum if latency else None,
            "current_value": current_value,
            "bounds_min": bounds.min_value if bounds else None,
            "bounds_max": bounds.max_value if bounds else None,
        }

        # 1. Telemetry sufficiency for the demand metrics (FRESH-only samples).
        for name in ("requests_running", "requests_waiting"):
            agg = signals.get(name)
            have = agg.samples if agg else 0
            if have < config.min_samples:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [f"insufficient telemetry: {name} fresh samples={have} "
                     f"< min_samples={config.min_samples}"],
                    guardrails, evidence,
                )

        # 2. Current value and bounds must be known to propose a bounded step.
        if current_value is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                ["current token-budget value unknown"], guardrails, evidence,
            )
        if bounds is None or bounds.min_value is None or bounds.max_value is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                ["capability bounds unknown"], guardrails, evidence,
            )

        # 3. Guardrail breach -> bounded PROPOSE_DOWN (safe direction).
        breaches = [g for g in guardrails if g.enabled and g.passed is False]
        if breaches:
            breach_reasons = [
                f"guardrail breach: {g.metric} fresh max={g.observed} > {g.comparator} "
                f"limit={g.limit}"
                for g in breaches
            ]
            if current_value > bounds.min_value:
                proposed = max(bounds.min_value, current_value - obj.step_down)
                return PolicyEvaluation(
                    DecisionKind.PROPOSE_DOWN, proposed,
                    breach_reasons + [f"propose down {current_value} -> {proposed}"],
                    guardrails, evidence,
                )
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                breach_reasons + [f"at min bound {bounds.min_value}; cannot decrease"],
                guardrails, evidence,
            )

        running_median = running.median
        waiting_median = waiting.median
        budget_p95 = iteration.median if iteration is not None else None

        # 4. Scale-up requires persistent evidence that the token budget is being
        #    reached. Generic queueing is NOT attributed to this actuator.
        if iteration is None or iteration.samples == 0 or budget_p95 is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                [
                    "scheduled-token budget-utilization evidence unavailable "
                    "(iteration_tokens_p95 has no fresh samples); generic queueing "
                    "is not attributed to the token budget"
                ],
                guardrails, evidence,
            )

        threshold = current_value * obj.budget_utilization_threshold
        budget_reached = budget_p95 >= threshold
        waiting_demand = waiting_median is not None and waiting_median > obj.waiting_high
        active = running_median is not None and running_median > 0

        if budget_reached and waiting_demand and active:
            if current_value >= bounds.max_value:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [
                        f"budget utilization at max bound {bounds.max_value}: "
                        f"iteration_tokens_p95 median={budget_p95} >= "
                        f"{obj.budget_utilization_threshold:.0%} of budget; "
                        f"waiting median={waiting_median}"
                    ],
                    guardrails, evidence,
                )
            required_missing = [
                g for g in guardrails
                if g.enabled and g.required_for_scale_up and not g.available
            ]
            if required_missing:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [
                        "scale-up blocked: required guardrail(s) unavailable: "
                        + ", ".join(f"{g.metric}(fresh samples=0)" for g in required_missing)
                    ] + [
                        f"budget-constraint evidence: iteration_tokens_p95 median="
                        f"{budget_p95} >= {obj.budget_utilization_threshold:.0%} of "
                        f"budget {current_value}; waiting median={waiting_median} > "
                        f"waiting_high={obj.waiting_high}; running median={running_median} > 0"
                    ],
                    guardrails, evidence,
                )
            proposed = min(bounds.max_value, current_value + obj.step_up)
            return PolicyEvaluation(
                DecisionKind.PROPOSE_UP, proposed,
                [
                    f"budget-constraint evidence: iteration_tokens_p95 median="
                    f"{budget_p95} >= {obj.budget_utilization_threshold:.0%} of "
                    f"budget {current_value}; waiting median={waiting_median} > "
                    f"waiting_high={obj.waiting_high}; running median={running_median} > 0",
                    f"propose up {current_value} -> {proposed}",
                ],
                guardrails, evidence,
            )

        return PolicyEvaluation(
            DecisionKind.HOLD, None,
            [
                f"within operating band: running median={running_median}, "
                f"waiting median={waiting_median}, iteration_tokens_p95 median="
                f"{budget_p95}, current={current_value}, "
                f"bounds=[{bounds.min_value},{bounds.max_value}]"
            ],
            guardrails, evidence,
        )


class KvWatermarkPolicy(_ActuatorPolicy):
    """Asymmetric KV-watermark policy (conservative increase only).

    The watermark is float [0,1) KV-admission headroom. A larger watermark is
    MORE_CONSERVATIVE (more reserve); a smaller one is MORE_PERMISSIVE. V1
    supports only the conservative direction: increase on factual KV/preemption
    safety pressure (high ``kv_usage`` AND preemptions occurring in the window)
    while below the cap. The permissive decrease would require factual evidence
    that the watermark is constraining admission (e.g. a watermark/admission-block
    counter), which vLLM does not expose; it is MANUAL_ONLY and never proposed
    here (generic queueing is not substituted).
    """

    increase_is_permissive = False

    def __init__(self, config: SupervisorConfig):
        self.config = config

    def _evaluate(
        self,
        signals: SignalWindow,
        *,
        current_value: Optional[ActuatorValue],
        bounds: Optional[CapabilityBounds],
    ) -> PolicyEvaluation:
        config = self.config
        obj = config.objective
        kv = signals.get("kv_usage")
        preemptions = signals.get("num_preemptions")
        latency = signals.get(config.guardrails.latency.metric)
        guardrails = self._guardrail_results(signals)
        preempt_delta = None
        if (preemptions is not None and preemptions.maximum is not None
                and preemptions.minimum is not None):
            preempt_delta = preemptions.maximum - preemptions.minimum
        evidence: Dict[str, Any] = {
            "kv_usage_max": kv.maximum if kv else None,
            "kv_usage_samples": kv.samples if kv else 0,
            "preemptions_last": preemptions.last if preemptions else None,
            "preemptions_in_window": preempt_delta,
            "preemptions_samples": preemptions.samples if preemptions else 0,
            "latency_metric": config.guardrails.latency.metric,
            "latency_max_observed": latency.maximum if latency else None,
            "current_value": current_value,
            "bounds_min": bounds.min_value if bounds else None,
            "bounds_max": bounds.max_value if bounds else None,
        }

        # 1. Sufficiency: KV usage and the preemption counter are both required.
        for name in ("kv_usage", "num_preemptions"):
            aggregate = signals.get(name)
            have = aggregate.samples if aggregate else 0
            if have < config.min_samples:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [f"insufficient telemetry: {name} fresh samples={have} "
                     f"< min_samples={config.min_samples}"],
                    guardrails, evidence,
                )
        if current_value is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                ["current KV watermark value unknown"], guardrails, evidence,
            )
        if bounds is None or bounds.min_value is None or bounds.max_value is None:
            return PolicyEvaluation(
                DecisionKind.HOLD, None,
                ["capability bounds unknown"], guardrails, evidence,
            )

        # 2. Conservative increase on factual KV/preemption safety pressure.
        kv_pressure = kv.maximum is not None and kv.maximum >= obj.kv_pressure_threshold
        preemption_pressure = preempt_delta is not None and preempt_delta >= obj.preemptions_min
        if kv_pressure and preemption_pressure:
            cap = min(obj.max_watermark, float(bounds.max_value), 0.999)
            if float(current_value) >= cap:
                return PolicyEvaluation(
                    DecisionKind.HOLD, None,
                    [
                        f"KV/preemption pressure at max watermark cap {cap}: "
                        f"kv_usage max={kv.maximum}, preemptions_in_window={preempt_delta}"
                    ],
                    guardrails, evidence,
                )
            proposed = round(min(cap, float(current_value) + obj.step_up), 4)
            return PolicyEvaluation(
                DecisionKind.PROPOSE_UP, proposed,
                [
                    f"KV/preemption safety pressure: kv_usage max={kv.maximum} >= "
                    f"kv_pressure_threshold={obj.kv_pressure_threshold}; "
                    f"preemptions_in_window={preempt_delta} >= "
                    f"preemptions_min={obj.preemptions_min}",
                    f"propose conservative increase {current_value} -> {proposed}",
                ],
                guardrails, evidence,
            )

        # 3. Permissive decrease is MANUAL_ONLY: no factual watermark-binding
        #    evidence is exposed, and generic queueing is not substituted.
        return PolicyEvaluation(
            DecisionKind.HOLD, None,
            [
                f"within operating band: kv_usage max={kv.maximum}, "
                f"preemptions_in_window={preempt_delta}, current={current_value}; "
                "permissive decrease unsupported (no factual watermark-binding "
                "evidence)"
            ],
            guardrails, evidence,
        )
