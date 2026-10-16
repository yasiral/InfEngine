"""Bounded in-memory rolling window for windowed histogram quantiles (M5B).

Latency percentiles cannot be computed from a single scrape: Prometheus
histograms are cumulative counters that only advance when a request finishes.
The collector keeps a SMALL BOUNDED in-memory history of per-scrape cumulative
bucket counts per family and computes quantiles over a time window as the DELTA
between the newest and oldest in-window samples (mirroring PromQL
``rate()`` + ``histogram_quantile()``).

No database is used. History is bounded by both count (``maxlen``) and age
(``retention_s``) and is reset when a counter reset or bucket-layout change is
detected (e.g. SGLang ``flush_cache()`` / engine restart).

Counter reset handling: if any newer cumulative bucket count is smaller than an
older one, the distribution was reset between them; pre-reset history is
dropped so a quantile is never computed across a reset.
"""

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from .exposition import histogram_quantile


class RollingHistograms:
    """Bounded per-family window of cumulative histogram samples."""

    def __init__(self, maxlen: int = 120, retention_s: float = 90.0):
        self.maxlen = maxlen
        self.retention_s = retention_s
        self._series: Dict[str, deque] = {}

    def _le_tuple(self, buckets) -> Tuple[float, ...]:
        return tuple(le for le, _ in buckets)

    def _count_tuple(self, buckets) -> Tuple[float, ...]:
        return tuple(count for _, count in buckets)

    def push(self, family: str, ts: float, buckets) -> None:
        """Record one cumulative-bucket snapshot at time ``ts``."""
        q = self._series.setdefault(family, deque(maxlen=self.maxlen))
        if not buckets:
            return
        if q:
            previous = q[-1][1]
            if self._le_tuple(buckets) != self._le_tuple(previous):
                q.clear()
            elif any(n < o for o, n in zip(self._count_tuple(previous), self._count_tuple(buckets))):
                q.clear()
        q.append((ts, tuple(buckets)))
        self._prune(q, ts)

    def _prune(self, q: deque, now: float) -> None:
        while q and now - q[0][0] > self.retention_s:
            q.popleft()

    def _delta(self, family: str, window_s: float) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...], float]]:
        """(le, delta_counts, delta_total) newest minus oldest in-window sample.

        Returns ``None`` when fewer than two usable in-window samples exist
        (including when every candidate is separated from the newest sample by
        a counter reset).
        """
        q = self._series.get(family)
        if not q or len(q) < 2:
            return None
        newest = q[-1]
        now = newest[0]
        newest_counts = self._count_tuple(newest[1])
        for index in range(len(q) - 1):
            candidate = q[index]
            if now - candidate[0] > window_s:
                continue
            older_counts = self._count_tuple(candidate[1])
            if any(n < o for o, n in zip(older_counts, newest_counts)):
                continue
            delta = tuple(n - o for o, n in zip(older_counts, newest_counts))
            return self._le_tuple(newest[1]), delta, delta[-1]
        return None

    def total_in_window(self, family: str, window_s: float) -> Optional[float]:
        """Total observations (delta count) in the window, or None if unusable."""
        result = self._delta(family, window_s)
        return result[2] if result is not None else None

    def window_buckets(
        self, family: str, window_s: float
    ) -> Optional[Tuple[Tuple[float, ...], Tuple[float, ...], float]]:
        """Windowed cumulative bucket delta -- the SAME source used by quantile().

        Returns ``(le_bounds, delta_cumulative_counts, total)`` where the counts
        are cumulative *within the window* (observations ``<= le``), or ``None``
        when no valid baseline exists (too few in-window samples, or a counter
        reset with no usable post-reset baseline). A returned tuple with
        ``total == 0`` is a valid zero-observation window; ``None`` is
        unavailable. No second collection path: this is the exact delta the
        quantile calculation uses.
        """
        return self._delta(family, window_s)

    def quantile(self, family: str, q: float, window_s: float) -> Optional[float]:
        """Windowed quantile (0..1) for ``family`` over the last ``window_s``."""
        result = self._delta(family, window_s)
        if result is None:
            return None
        le, delta_counts, total = result
        if total <= 0 or not math.isfinite(total):
            return None
        buckets = tuple(zip(le, delta_counts))
        return histogram_quantile(q, buckets)

    def quantiles(self, family: str, quantiles, window_s: float) -> Dict[str, Optional[float]]:
        return {key: self.quantile(family, q, window_s) for key, q in quantiles.items()}

    def reset(self, family: str) -> None:
        """Drop all history for a family."""
        self._series.pop(family, None)

    def families(self) -> List[str]:
        return list(self._series)