"""M7A: windowed, factual telemetry signal builder for the supervisory controller.

Turns a sequence of existing M5B/M5C telemetry snapshots into one **numeric,
windowed** :class:`SignalWindow`. Every derived value is a plain aggregate
(median / min / max / last / sample counts) computed over the snapshots whose
timestamp falls inside a configured horizon.

Design constraints (M7A):

  * **Factual and numeric only** — no inferred labels, no diagnoses, no
    "bottleneck"/"pressure" classifications. The controller reasons about the
    numbers recorded here.
  * **Fresh-only decision evidence** — the decision statistics (median / min /
    max / last and the ``samples`` count) are computed from FRESH snapshots
    only. STALE samples are recorded diagnostically (count + their own
    aggregates) but never satisfy ``min_samples``, saturation, waiting-demand or
    guardrail evidence.
  * The window horizon is configuration (``window_s``), never a hidden constant.
  * Availability is recorded explicitly; metrics the engine reports as disabled
    / missing are simply absent, never coerced to zero.
"""

import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..telemetry import MetricQuality, RuntimeSnapshot

# Metrics the snapshot stores as normalized top-level fields. Anything else is
# an engine-specific key in ``RuntimeSnapshot.engine_specific``.
from ..telemetry.snapshot import NORMALIZED_FIELDS


def _value_of(snapshot: RuntimeSnapshot, metric: str) -> Optional[float]:
    """Raw value for a metric, or ``None`` when absent (never coerced)."""
    if metric in NORMALIZED_FIELDS:
        value = getattr(snapshot, metric, None)
    else:
        value = snapshot.engine_specific.get(metric)
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric != numeric or numeric in (float("inf"), float("-inf")):
        return None
    return numeric


@dataclass(frozen=True)
class MetricAggregate:
    """Numeric aggregate of one metric over the window.

    ``samples`` / ``median`` / ``minimum`` / ``maximum`` / ``last`` are the
    FRESH-only decision evidence. ``stale*`` fields record the stale samples
    diagnostically and are never used for any decision.
    """

    metric: str
    samples: int
    stale: int
    median: Optional[float]
    minimum: Optional[float]
    maximum: Optional[float]
    last: Optional[float]
    stale_median: Optional[float]
    stale_minimum: Optional[float]
    stale_maximum: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric": self.metric,
            "samples": self.samples,
            "stale": self.stale,
            "median": self.median,
            "min": self.minimum,
            "max": self.maximum,
            "last": self.last,
            "stale_median": self.stale_median,
            "stale_min": self.stale_minimum,
            "stale_max": self.stale_maximum,
        }


@dataclass(frozen=True)
class SignalWindow:
    """The exact telemetry window one decision was evaluated over."""

    window_s: float
    start_ts: float
    end_ts: float
    snapshots: int
    metrics: Dict[str, MetricAggregate] = field(default_factory=dict)

    def get(self, metric: str) -> Optional[MetricAggregate]:
        return self.metrics.get(metric)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_s": self.window_s,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "snapshots": self.snapshots,
            "metrics": {k: v.to_dict() for k, v in sorted(self.metrics.items())},
        }


def _aggregate(metric: str, fresh: List[float], stale: List[float]) -> MetricAggregate:
    if fresh:
        median = statistics.median(fresh)
        minimum, maximum, last = min(fresh), max(fresh), fresh[-1]
    else:
        median = minimum = maximum = last = None
    if stale:
        stale_median, stale_minimum, stale_maximum = (
            statistics.median(stale), min(stale), max(stale),
        )
    else:
        stale_median = stale_minimum = stale_maximum = None
    return MetricAggregate(
        metric=metric,
        samples=len(fresh),
        stale=len(stale),
        median=median,
        minimum=minimum,
        maximum=maximum,
        last=last,
        stale_median=stale_median,
        stale_minimum=stale_minimum,
        stale_maximum=stale_maximum,
    )


# Cumulative engine counters: a decrease (counter reset / engine restart) or a
# change of the source engine instance is a discontinuity. Windowed aggregates
# for these metrics are computed over the LATEST contiguous counter epoch only,
# so pre-reset and post-reset observations are never combined (M9B.2). Derived
# scalars (e.g. ``iteration_tokens_p95``) and gauges are NOT in this set.
CUMULATIVE_COUNTER_METRICS: Tuple[str, ...] = (
    "num_preemptions",
    "num_retracted_requests_total",
    "prompt_tokens",
    "prompt_tokens_cached",
    "generation_tokens",
    "prefix_cache_queries",
    "prefix_cache_hits",
)


def _latest_counter_segment(
    sequence: List[Optional[Tuple[float, str]]],
) -> List[float]:
    """Values of the latest contiguous counter epoch.

    ``sequence`` is chronological -- one entry per in-window observation -- where
    an entry is ``(value, instance_id)`` for a FRESH observation or ``None`` for
    an **unavailable observation** (missing / disabled / no_samples /
    scrape_failed / stale). A ``None`` gap means the interval was not observed, so
    continuity MUST break there: the counter cannot be bridged across it (a
    reset may have happened). A value decrease or an ``instance_id`` change also
    starts a new epoch. Everything before the last discontinuity (gap, reset or
    instance change) is discarded. The returned list may be shorter than
    ``sequence`` (or empty); callers must not treat a short/empty segment as a
    valid count.
    """
    start = 0
    prev_value: Optional[float] = None
    prev_instance: Optional[str] = None
    for index, item in enumerate(sequence):
        if item is None:
            # Unobserved interval: nothing after this may be combined with
            # anything before it.
            start = index + 1
            prev_value = None
            prev_instance = None
            continue
        value, instance = item
        if prev_value is not None and (
            instance != prev_instance or value < prev_value
        ):
            start = index
        prev_value = value
        prev_instance = instance
    return [
        item[0]
        for item in sequence[start:]
        if item is not None
    ]


def build_signal_window(
    samples: Sequence[Tuple[float, RuntimeSnapshot]],
    *,
    now: float,
    window_s: float,
    metrics: Sequence[str],
    counter_metrics: Optional[Sequence[str]] = None,
) -> SignalWindow:
    """Aggregate ``samples`` in ``[now - window_s, now]`` into a SignalWindow.

    ``samples`` are ``(timestamp, RuntimeSnapshot)`` pairs; timestamps are the
    same monotonic clock used for ``now``. Samples are sorted internally, so a
    caller may pass any order. Only FRESH samples feed the decision statistics.

    Cumulative counters listed in ``counter_metrics`` (default
    :data:`CUMULATIVE_COUNTER_METRICS`) are reset-safe: their decision
    statistics are computed over the latest contiguous counter epoch only
    (decrease or instance change = discontinuity), and ``samples`` reflects that
    segment so existing window-adequacy gates fail closed on insufficient
    post-reset evidence. Gauges, derived scalars and stale/missing values are
    unaffected and are never coerced to zero.
    """
    counters = set(
        CUMULATIVE_COUNTER_METRICS if counter_metrics is None else counter_metrics
    )
    lo = now - window_s
    in_window = sorted(
        ((ts, snap) for (ts, snap) in samples if lo <= ts <= now),
        key=lambda item: item[0],
    )
    aggregates: Dict[str, MetricAggregate] = {}
    for metric in metrics:
        is_counter = metric in counters
        # For counters, one chronological entry per in-window snapshot: a FRESH
        # (value, instance) or None for an unavailable observation (a gap that
        # breaks continuity). Non-counters keep a plain fresh-values list.
        sequence: List[Optional[Tuple[float, str]]] = []
        fresh_values: List[float] = []
        stale: List[float] = []
        for _, snap in in_window:
            value = _value_of(snap, metric)
            quality = snap.quality_of(metric)
            if value is not None and quality is MetricQuality.FRESH:
                if is_counter:
                    sequence.append((value, getattr(snap, "instance_id", "")))
                else:
                    fresh_values.append(value)
                continue
            # Unavailable / stale observation for this metric.
            if is_counter:
                sequence.append(None)
            if value is not None and quality is MetricQuality.STALE:
                stale.append(value)
        if is_counter:
            fresh_values = _latest_counter_segment(sequence)
        aggregates[metric] = _aggregate(metric, fresh_values, stale)
    return SignalWindow(
        window_s=window_s,
        start_ts=lo,
        end_ts=now,
        snapshots=len(in_window),
        metrics=aggregates,
    )
