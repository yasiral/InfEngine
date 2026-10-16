"""Out-of-band telemetry collectors for vLLM v0.28.0 and SGLang v0.5.18 (M5B).

Each collector scrapes the engine's NATIVE HTTP telemetry surfaces (no
in-process access, no adapter involvement), keeps a small bounded in-memory
history for windowed histogram quantiles, and produces a
:class:`~infengine.telemetry.snapshot.RuntimeSnapshot` with ``MetricQuality``
populated per the M5A v1.1 contract:

  * ``fresh``        — value from a successful scrape (within the staleness window)
  * ``stale``        — value retained but the engine's published data is older
                       than the staleness window (SGLang ``sglang_timestamp``)
  * ``disabled``     — surface/metric not enabled (SGLang ``/metrics`` -> 404
                       without ``--enable-metrics``; vLLM ``--disable-log-stats``
                       yields an empty registry)
  * ``no_samples``   — enabled but no observations in the window (empty
                       histogram delta)
  * ``scrape_failed``— surface unreachable / HTTP error / parse failure

Collection is independent of the control adapters and the Reconciler.
Inference traffic is never proxied: the collector only issues ``GET`` requests
to the engine's own telemetry endpoints.
"""

import abc
import math
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .exposition import Exposition, ExpositionParseError, HistogramSeries, parse_exposition
from .snapshot import (
    LATENCY_FAMILIES,
    NORMALIZED_FIELDS,
    PRESERVED_METRICS,
    EngineKind,
    MetricQuality,
    RuntimeSnapshot,
)
from .window import RollingHistograms

_QUANTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99))
_LATENCY_FIELD_NAMES = {family: LATENCY_FAMILIES[family] for family in LATENCY_FAMILIES}


class TelemetryError(RuntimeError):
    """A telemetry source could not be reached or queried."""


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_epoch(iso: str) -> float:
    dt = datetime.fromisoformat(iso)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _http_get(url: str, timeout: float) -> Tuple[int, str]:
    """GET a URL; raise :class:`TelemetryError` on connect-level failure."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise TelemetryError(f"GET {url} failed: {exc}") from exc


def _first_value(expo: Exposition, name: str, match: Optional[Dict[str, str]] = None) -> Optional[float]:
    sample = expo.find(name, match)
    if sample is None or not math.isfinite(sample.value):
        return None
    return sample.value


def _first_value_with_fallback(
    expo: Exposition, primary: str, fallback: Optional[str] = None
) -> Optional[float]:
    """Exact-name lookup with a single, explicit fallback.

    Precedence is strict (primary first); the two forms are never summed. Used
    only for the vLLM counters whose Prometheus exposition name gained a
    ``_total`` suffix, so older/unsuffixed surfaces still resolve.
    """
    value = _first_value(expo, primary)
    if value is None and fallback is not None:
        value = _first_value(expo, fallback)
    return value


def _sum_values(expo: Exposition, name: str) -> Optional[float]:
    values = [v for v in expo.values(name) if math.isfinite(v)]
    return sum(values) if values else None


def _label_counter(
    expo: Exposition, name: str, label: str
) -> Optional[Dict[str, int]]:
    """Sum a multi-label counter into ``{label_value: cumulative_count}``.

    Tolerates the Prometheus counter ``_total`` suffix in either direction
    (``prometheus_client`` appends ``_total`` to counters; vLLM v0.28.0 emits
    ``..._total`` live). Returns ``None`` when the family is truly absent, so
    the caller records an explicit availability state instead of a fabricated
    empty dict.
    """
    fam = expo.family(name)
    if fam is None and not name.endswith("_total"):
        fam = expo.family(name + "_total")
    if fam is None and name.endswith("_total"):
        fam = expo.family(name[: -len("_total")])
    if fam is None:
        return None
    counts: Dict[str, int] = {}
    for sample in fam.samples:
        key = sample.label(label) or "unknown"
        if not math.isfinite(sample.value):
            continue
        counts[key] = counts.get(key, 0) + _to_int(sample.value)
    return counts or None


def _safe_parse(body: str) -> Optional[Exposition]:
    try:
        return parse_exposition(body)
    except ExpositionParseError:
        return None


def _merge_series(series: List[HistogramSeries]):
    """Combine histogram series (same family, different label values) by summing
    cumulative bucket counts — valid because all series measure the same
    quantity (e.g. latency seconds, regardless of ``is_streaming``/``dp_rank``)."""
    if not series:
        return ()
    bounds = tuple(le for le, _ in series[0].buckets)
    counts = tuple(
        sum(s.buckets[i][1] for s in series if len(s.buckets) == len(bounds))
        for i in range(len(bounds))
    )
    return tuple(zip(bounds, counts))


def _series_sum(series: List[HistogramSeries]) -> float:
    return float(sum(s.sum for s in series))


def _to_int(value: float) -> int:
    return int(round(value))


class TelemetryCollector(abc.ABC):
    """Shared poll-loop, bounded window, and snapshot state for one engine.

    A collector is created for one engine instance and polls its native
    telemetry surface(s) on an interval. ``latest()`` returns the most recent
    :class:`RuntimeSnapshot` (``None`` before the first poll). ``poll_once()``
    performs a single synchronous collection (used by tests and the poll loop).
    """

    engine: str = ""
    source_desc: str = ""

    def __init__(
        self,
        base_url: str,
        *,
        poll_interval: float = 5.0,
        window_s: float = 30.0,
        staleness_s: float = 20.0,
        timeout: float = 5.0,
        max_history: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.poll_interval = poll_interval
        self.window_s = window_s
        self.staleness_s = staleness_s
        self.timeout = timeout
        self.window = RollingHistograms(
            maxlen=max_history, retention_s=max(2.0 * window_s, 60.0)
        )
        self._latest: Optional[RuntimeSnapshot] = None
        self._lock = threading.RLock()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    @property
    def instance_id(self) -> str:
        return f"{self.engine}:{self.base_url}"

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run, name=f"infengine-telemetry-{self.engine}", daemon=True
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lock:
            self._stop.set()
            thread = self._thread
            self._thread = None
        if thread is not None:
            thread.join(timeout=self.poll_interval + 2.0)

    def _run(self) -> None:
        while True:
            try:
                self.poll_once()
            except Exception:
                # _collect() maps all reachability/parse failures to quality
                # states; this is a safety net so the poll loop never dies.
                pass
            if self._stop.wait(self.poll_interval):
                return

    def latest(self) -> Optional[RuntimeSnapshot]:
        with self._lock:
            return self._latest

    def poll_once(self) -> RuntimeSnapshot:
        snap = self._collect()
        with self._lock:
            self._latest = snap
        return snap

    @abc.abstractmethod
    def _collect(self) -> RuntimeSnapshot:
        ...

    # ------------------------------------------------------- snapshot helpers
    def _empty_snapshot(self, collected_at: str, quality: Dict[str, str]) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            engine=self.engine,
            instance_id=self.instance_id,
            collected_at=collected_at,
            source=self.source_desc,
            quality=quality,
        )

    def _failure_snapshot(self, error: str) -> RuntimeSnapshot:
        """Every field sourced from this engine is SCRAPE_FAILED (no values)."""
        quality = {f: MetricQuality.SCRAPE_FAILED.value for f in NORMALIZED_FIELDS}
        for key in PRESERVED_METRICS.get(self.engine, {}):
            quality[key] = MetricQuality.SCRAPE_FAILED.value
        return self._empty_snapshot(_iso_now(), quality)

    # ------------------------------------------------------- evidence helpers
    def _new_meta(self) -> Dict[str, Any]:
        """Fresh M9B measurement sidecar for one snapshot."""
        return {
            "window_s": self.window_s,
            "staleness_s": self.staleness_s,
            "families": {},
        }

    def _record_family_meta(
        self, meta: Optional[Dict[str, Any]], key: str, samples: float,
    ) -> None:
        if meta is None:
            return
        meta.setdefault("families", {})[key] = {
            "samples": int(samples),
            "window_s": self.window_s,
        }

    def _record_histogram_buckets(
        self,
        meta: Optional[Dict[str, Any]],
        out_key: str,
        window_family: str,
        unit: str = "seconds",
    ) -> None:
        """Preserve the windowed native histogram buckets for one family.

        Uses ``RollingHistograms.window_buckets`` -- the SAME delta the quantile
        calculation consumes (one push per scrape, no second collection path).
        The stored counts are cumulative *within the window*; the report converts
        them to non-overlapping buckets. A valid zero-observation window is
        stored (``total == 0``); when no valid baseline exists nothing is stored
        (the caller treats the absence as UNAVAILABLE with an explicit reason).
        ``+Inf`` bounds are encoded as the string ``"+Inf"`` so the artifact is
        strict JSON.
        """
        if meta is None:
            return
        delta = self.window.window_buckets(window_family, self.window_s)
        if delta is None:
            return
        le, counts, total = delta
        meta.setdefault("histograms", {})[out_key] = {
            "le": [("+Inf" if math.isinf(float(b)) else float(b)) for b in le],
            "counts": [int(c) for c in counts],
            "total": int(total),
            "unit": unit,
            "window_s": self.window_s,
        }

    def _record_latency(
        self,
        expo: Exposition,
        family_key: str,
        metric_name: str,
        now: float,
        quality: Dict[str, str],
        kwargs: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
        engine_specific: Optional[Dict[str, Any]] = None,
        preserved_key: Optional[str] = None,
    ) -> Tuple[Dict[str, float], int]:
        """Window a latency histogram; fill the p50/p95/p99 snapshot fields.

        Returns ``(quantiles, samples)`` with ``quantiles`` mapping
        ``p50``/``p95``/``p99`` to present values (seconds) and ``samples`` the
        windowed observation count. When ``preserved_key`` is supplied the same
        single window push additionally records the correctly-named
        engine-specific family (used where a normalized legacy field and a
        preserved field share one native histogram, e.g. ``tpot`` /
        ``inter_token_latency``) -- never a second push.
        """
        fields = _LATENCY_FIELD_NAMES[family_key]
        series = expo.histogram_series(metric_name)
        if not series:
            quality.update({f: MetricQuality.DISABLED.value for f in fields})
            if preserved_key is not None:
                quality[preserved_key] = MetricQuality.DISABLED.value
            return {}, 0
        self.window.push(metric_name, now, _merge_series(series))
        total = self.window.total_in_window(metric_name, self.window_s)
        if total is None or total <= 0:
            quality.update({f: MetricQuality.NO_SAMPLES.value for f in fields})
            if preserved_key is not None:
                quality[preserved_key] = MetricQuality.NO_SAMPLES.value
            return {}, 0
        quantiles: Dict[str, float] = {}
        for field, key, q in zip(fields, ("p50", "p95", "p99"), (0.5, 0.95, 0.99)):
            value = self.window.quantile(metric_name, q, self.window_s)
            if value is None:
                quality[field] = MetricQuality.NO_SAMPLES.value
            else:
                kwargs[field] = value
                quantiles[key] = value
        self._record_family_meta(meta, family_key, total)
        if preserved_key is not None and engine_specific is not None:
            if quantiles:
                engine_specific[preserved_key] = dict(quantiles)
            self._record_family_meta(meta, preserved_key, total)
        return quantiles, int(total)

    def _record_preserved_histogram(
        self,
        expo: Exposition,
        key: str,
        metric_name: str,
        now: float,
        quality: Dict[str, str],
        engine_specific: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Window an engine-specific histogram; store its windowed quantiles.

        The family is preserved verbatim under ``engine_specific[key]`` (never
        normalized). Unavailability is explicit: no family -> ``disabled``;
        enabled but no window samples -> ``no_samples``; otherwise a quantile
        dict is stored.
        """
        series = expo.histogram_series(metric_name)
        if not series:
            quality[key] = MetricQuality.DISABLED.value
            return
        self.window.push(metric_name, now, _merge_series(series))
        total = self.window.total_in_window(metric_name, self.window_s)
        if total is None or total <= 0:
            quality[key] = MetricQuality.NO_SAMPLES.value
            return
        quantiles: Dict[str, float] = {}
        for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
            value = self.window.quantile(metric_name, q, self.window_s)
            if value is not None:
                quantiles[name] = value
        if not quantiles:
            quality[key] = MetricQuality.NO_SAMPLES.value
            return
        engine_specific[key] = quantiles
        self._record_family_meta(meta, key, total)

    def _record_preserved_stage_histogram(
        self,
        expo: Exposition,
        key: str,
        metric_name: str,
        now: float,
        quality: Dict[str, str],
        engine_specific: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Window a stage-labelled histogram into ``{stage: quantiles}``.

        Series are grouped by their ``stage`` label (DP ranks merged within a
        stage) before windowing, so each stage keeps its own rolling window.
        """
        series = expo.histogram_series(metric_name)
        if not series:
            quality[key] = MetricQuality.DISABLED.value
            return
        by_stage: Dict[str, List[HistogramSeries]] = {}
        for item in series:
            stage = dict(item.labels).get("stage")
            if stage is None:
                continue
            by_stage.setdefault(stage, []).append(item)
        if not by_stage:
            quality[key] = MetricQuality.DISABLED.value
            return
        out: Dict[str, Dict[str, float]] = {}
        for stage, group in by_stage.items():
            family = f"{metric_name}::{stage}"
            self.window.push(family, now, _merge_series(group))
            total = self.window.total_in_window(family, self.window_s)
            if total is None or total <= 0:
                continue
            quantiles: Dict[str, float] = {}
            for name, q in (("p50", 0.5), ("p95", 0.95), ("p99", 0.99)):
                value = self.window.quantile(family, q, self.window_s)
                if value is not None:
                    quantiles[name] = value
            if quantiles:
                out[stage] = quantiles
                self._record_family_meta(meta, f"{key}:{stage}", total)
        if not out:
            quality[key] = MetricQuality.NO_SAMPLES.value
            return
        engine_specific[key] = out


class VllmTelemetryCollector(TelemetryCollector):
    """Collector for vLLM v0.28.0 ``GET /metrics`` (Prometheus text, default on)."""

    engine = EngineKind.VLLM.value
    source_desc = "prometheus:/metrics"

    GAUGES = {
        "requests_running": ("vllm:num_requests_running", "int"),
        "requests_waiting": ("vllm:num_requests_waiting", "int"),
        "kv_usage": ("vllm:kv_cache_usage_perc", "float"),
    }
    LATENCY_METRICS = {
        "ttft": "vllm:time_to_first_token_seconds",
        "tpot": "vllm:inter_token_latency_seconds",
        "e2e": "vllm:e2e_request_latency_seconds",
        "queue_time": "vllm:request_queue_time_seconds",
    }
    # Prometheus counter names carry a '_total' suffix (live-verified on the
    # patched vLLM v0.28.0 endpoint). The InfEngine-facing keys are unchanged.
    PRESERVED = {
        "num_preemptions": ("vllm:num_preemptions_total", "int"),
        "prompt_tokens": ("vllm:prompt_tokens_total", "int"),
        "prompt_tokens_cached": ("vllm:prompt_tokens_cached_total", "int"),
        "generation_tokens": ("vllm:generation_tokens_total", "int"),
        "prefix_cache_queries": ("vllm:prefix_cache_queries_total", "int"),
        "prefix_cache_hits": ("vllm:prefix_cache_hits_total", "int"),
        "num_requests_waiting_by_reason": ("vllm:num_requests_waiting_by_reason", "reason_gauge"),
        "iteration_tokens_total": ("vllm:iteration_tokens_total", "histogram"),
    }
    # Narrow, explicit exact-name fallback: only these counters may resolve
    # against the unsuffixed name, and only when the suffixed family is absent.
    # Deterministic precedence; the two forms are NEVER summed. Not a global
    # "_total" rule.
    COUNTER_FALLBACKS = {
        "vllm:num_preemptions_total": "vllm:num_preemptions",
        "vllm:prompt_tokens_total": "vllm:prompt_tokens",
        "vllm:prompt_tokens_cached_total": "vllm:prompt_tokens_cached",
        "vllm:generation_tokens_total": "vllm:generation_tokens",
        "vllm:prefix_cache_queries_total": "vllm:prefix_cache_queries",
        "vllm:prefix_cache_hits_total": "vllm:prefix_cache_hits",
    }
    # M9B: engine-specific request-scoped histograms preserved verbatim.
    PRESERVED_LATENCY = {
        "request_prefill_time": "vllm:request_prefill_time_seconds",
        "request_decode_time": "vllm:request_decode_time_seconds",
        "request_inference_time": "vllm:request_inference_time_seconds",
        "request_time_per_output_token": "vllm:request_time_per_output_token_seconds",
        "request_prompt_tokens": "vllm:request_prompt_tokens",
        "request_generation_tokens": "vllm:request_generation_tokens",
    }
    # M9C.1: normalized latency families whose native buckets are preserved, and
    # the mapping from preserved histogram key -> result-facing histogram key.
    HISTOGRAM_NORMALIZED = ("ttft", "queue_time", "e2e")
    HISTOGRAM_FAMILIES = {
        "request_prefill_time": "prefill_time",
        "request_decode_time": "decode_time",
    }
    # M9B: multi-label counters preserved verbatim (family, label). Live
    # v0.28.0 exposition carries the Prometheus counter '_total' suffix.
    PRESERVED_LABEL_COUNTERS = {
        "request_success": ("vllm:request_success_total", "finished_reason"),
        "prompt_tokens_by_source": ("vllm:prompt_tokens_by_source_total", "source"),
    }

    def __init__(self, base_url: str, *, metrics_path: str = "/metrics", **kwargs):
        super().__init__(base_url, **kwargs)
        self.metrics_url = f"{self.base_url}{metrics_path}"

    # ---------------------------------------------------------------- polling
    def _collect(self) -> RuntimeSnapshot:
        try:
            status, body = _http_get(self.metrics_url, self.timeout)
        except TelemetryError as exc:
            return self._failure_snapshot(str(exc))
        if status != 200:
            return self._failure_snapshot(f"HTTP {status}")
        expo = _safe_parse(body)
        if expo is None:
            return self._failure_snapshot("unparseable /metrics body")
        return self._build(expo, _iso_now())

    def _build(self, expo: Exposition, collected_at: str) -> RuntimeSnapshot:
        now = _to_epoch(collected_at)
        kwargs: Dict[str, Any] = {}
        quality: Dict[str, str] = {}
        for field, (fam_name, kind) in self.GAUGES.items():
            value = _first_value(expo, fam_name)
            if value is None:
                quality[field] = MetricQuality.DISABLED.value
            elif kind == "int":
                kwargs[field] = _to_int(value)
            else:
                kwargs[field] = value
        meta = self._new_meta()
        engine_specific: Dict[str, Any] = {}
        for family_key, metric_name in self.LATENCY_METRICS.items():
            # The legacy normalized tpot_* and the preserved inter_token_latency
            # share one native histogram: one push, both views (M9B).
            self._record_latency(
                expo, family_key, metric_name, now, quality, kwargs,
                meta=meta,
                engine_specific=engine_specific if family_key == "tpot" else None,
                preserved_key="inter_token_latency" if family_key == "tpot" else None,
            )
            if family_key in self.HISTOGRAM_NORMALIZED:
                self._record_histogram_buckets(meta, family_key, metric_name)
        for key, metric_name in self.PRESERVED_LATENCY.items():
            self._record_preserved_histogram(
                expo, key, metric_name, now, quality, engine_specific, meta
            )
            out_key = self.HISTOGRAM_FAMILIES.get(key)
            if out_key is not None:
                self._record_histogram_buckets(meta, out_key, metric_name)
        for key, (fam_name, label) in self.PRESERVED_LABEL_COUNTERS.items():
            counts = _label_counter(expo, fam_name, label)
            if counts is None:
                quality[key] = MetricQuality.DISABLED.value
            else:
                engine_specific[key] = counts
        for key, (fam_name, kind) in self.PRESERVED.items():
            if kind == "reason_gauge":
                fam = expo.family(fam_name)
                if fam is None:
                    quality[key] = MetricQuality.DISABLED.value
                    continue
                engine_specific[key] = {
                    (s.label("reason") or "unknown"): _to_int(s.value)
                    for s in fam.samples
                }
            elif kind == "histogram":
                series = expo.histogram_series(fam_name)
                if not series:
                    quality[key] = MetricQuality.DISABLED.value
                    continue
                merged = _merge_series(series)
                engine_specific[key] = {
                    "count": _to_int(merged[-1][1]),
                    "sum": _series_sum(series),
                }
                if key == "iteration_tokens_total":
                    # Factual budget-utilization surface (M7D1): tokens actually
                    # processed per scheduling step (windowed p95), directly
                    # comparable to the active runtime_max_scheduled_tokens
                    # budget. No generic bottleneck inference.
                    self.window.push(fam_name, now, merged)
                    if (self.window.total_in_window(fam_name, self.window_s) or 0) > 0:
                        p95 = self.window.quantile(fam_name, 0.95, self.window_s)
                        if p95 is not None:
                            engine_specific["iteration_tokens_p95"] = p95
            else:
                value = _first_value_with_fallback(
                    expo, fam_name, self.COUNTER_FALLBACKS.get(fam_name)
                )
                if value is None:
                    quality[key] = MetricQuality.DISABLED.value
                else:
                    engine_specific[key] = _to_int(value) if kind == "int" else value
        return RuntimeSnapshot(
            engine=self.engine,
            instance_id=self.instance_id,
            collected_at=collected_at,
            source=self.source_desc,
            engine_specific=engine_specific,
            quality=quality,
            measurement_meta=meta,
            **kwargs,
        )


class SglangTelemetryCollector(TelemetryCollector):
    """Collector for SGLang v0.5.18.

    Two native surfaces:

      * ``GET /v1/loads?format=prometheus`` (always on, no flag) — the three
        gauges plus the always-on preserved gauges; staleness is derived from
        the ``sglang_timestamp`` sample (epoch seconds).
      * ``GET /metrics`` (404 without ``--enable-metrics``) — latency
        histograms plus the metrics-only preserved metrics. A 404 is reported
        as ``disabled`` (config signal), not ``scrape_failed``.
    """

    engine = EngineKind.SGLANG.value
    source_desc = "sglang:/v1/loads + sglang:/metrics"

    LOAD_GAUGES = {
        "requests_running": ("sglang_num_running_reqs", "int"),
        "requests_waiting": ("sglang_num_waiting_reqs", "int"),
        "kv_usage": ("sglang_token_usage", "float"),
    }
    LOAD_TIMESTAMP = "sglang_timestamp"
    LOAD_PRESERVED = {
        "gen_throughput": ("sglang_gen_throughput", "float"),
        "cache_hit_rate": ("sglang_cache_hit_rate", "float"),
        "token_usage": ("sglang_token_usage", "float"),
        "max_total_num_tokens": ("sglang_max_total_num_tokens", "int"),
        "max_running_requests": ("sglang_max_running_requests", "int"),
        "utilization": ("sglang_utilization", "float"),
        "kv_cache_memory_usage_gb": ("sglang_memory_kv_cache_gb", "float"),
        "weight_memory_usage_gb": ("sglang_memory_weight_gb", "float"),
        # M9B engine-level /v1/loads evidence (flat scalars, default on).
        "num_used_tokens": ("sglang_num_used_tokens", "int"),
        "num_total_tokens": ("sglang_num_total_tokens", "int"),
        "num_active_tokens": ("sglang_num_active_tokens", "int"),
        "num_waiting_uncached_tokens": ("sglang_num_waiting_uncached_tokens", "int"),
        "total_prefill_uncached_tokens": ("sglang_total_prefill_uncached_tokens", "int"),
        "total_prefill_busy_us": ("sglang_total_prefill_busy_us", "int"),
    }
    LATENCY_METRICS = {
        "ttft": "sglang:time_to_first_token_seconds",
        "tpot": "sglang:inter_token_latency_seconds",
        "e2e": "sglang:e2e_request_latency_seconds",
        "queue_time": "sglang:queue_time_seconds",
    }
    # M9B: metrics-only engine-specific histograms (``--enable-metrics``).
    PRESERVED_LATENCY = {
        "request_prompt_tokens": "sglang:prompt_tokens_histogram",
        "request_uncached_prompt_tokens": "sglang:uncached_prompt_tokens_histogram",
        "request_generation_tokens": "sglang:generation_tokens_histogram",
    }
    # M9B: stage-labelled histogram -> engine_specific["<key>"][stage].
    PRESERVED_STAGE_LATENCY = {
        "per_stage_req_latency": "sglang:per_stage_req_latency_seconds",
    }
    # M9C.1: normalized latency families preserved from native buckets, and the
    # request-stage -> result-facing histogram key (SGLang has no request-scoped
    # decode duration, so only prefill_forward is mapped).
    HISTOGRAM_NORMALIZED = ("ttft", "queue_time", "e2e")
    HISTOGRAM_STAGE_FAMILIES = {"prefill_forward": "prefill_time"}
    METRICS_PRESERVED = {
        "kv_used_tokens": ("sglang:kv_used_tokens", "int"),
        "kv_available_tokens": ("sglang:kv_available_tokens", "int"),
        "kv_evictable_tokens": ("sglang:kv_evictable_tokens", "int"),
        "prompt_tokens_total": ("sglang:prompt_tokens_total", "sum_int"),
        "generation_tokens_total": ("sglang:generation_tokens_total", "sum_int"),
        "cached_tokens_total": ("sglang:cached_tokens_total", "sum_int"),
        "num_retracted_requests_total": ("sglang:num_retracted_requests_total", "int"),
        # M9B scheduler/outcome evidence (metrics-only).
        "forward_execution_seconds_total": ("sglang:forward_execution_seconds_total", "sum_float"),
        "fwd_occupancy": ("sglang:fwd_occupancy", "float"),
        "decode_sum_seq_lens": ("sglang:decode_sum_seq_lens", "float"),
        "num_requests_total": ("sglang:num_requests_total", "sum_int"),
        "num_aborted_requests_total": ("sglang:num_aborted_requests_total", "sum_int"),
    }

    def __init__(
        self,
        base_url: str,
        *,
        loads_path: str = "/v1/loads",
        metrics_path: str = "/metrics",
        query_metrics: bool = True,
        **kwargs,
    ):
        super().__init__(base_url, **kwargs)
        self.loads_url = f"{self.base_url}{loads_path}?format=prometheus"
        self.metrics_url = f"{self.base_url}{metrics_path}"
        self.query_metrics = query_metrics

    # ---------------------------------------------------------------- polling
    def _collect(self) -> RuntimeSnapshot:
        loads_expo: Optional[Exposition] = None
        loads_error: Optional[str] = None
        try:
            status, body = _http_get(self.loads_url, self.timeout)
        except TelemetryError as exc:
            loads_error = str(exc)
        else:
            if status != 200:
                loads_error = f"HTTP {status}"
            else:
                loads_expo = _safe_parse(body)
                if loads_expo is None:
                    loads_error = "unparseable /v1/loads body"

        metrics_expo: Optional[Exposition] = None
        metrics_error: Optional[str] = None
        metrics_disabled = False
        if self.query_metrics:
            try:
                mstatus, mbody = _http_get(self.metrics_url, self.timeout)
            except TelemetryError as exc:
                metrics_error = str(exc)
            else:
                if mstatus == 404:
                    metrics_disabled = True
                elif mstatus != 200:
                    metrics_error = f"HTTP {mstatus}"
                else:
                    metrics_expo = _safe_parse(mbody)
                    if metrics_expo is None:
                        metrics_error = "unparseable /metrics body"
        return self._build(
            loads_expo,
            loads_error,
            metrics_expo,
            metrics_disabled,
            metrics_error,
            _iso_now(),
        )

    def _build(
        self,
        loads_expo: Optional[Exposition],
        loads_error: Optional[str],
        metrics_expo: Optional[Exposition],
        metrics_disabled: bool,
        metrics_error: Optional[str],
        collected_at: str,
    ) -> RuntimeSnapshot:
        now = _to_epoch(collected_at)
        kwargs: Dict[str, Any] = {}
        quality: Dict[str, str] = {}
        engine_specific: Dict[str, Any] = {}
        meta = self._new_meta()

        observed = now
        if loads_expo is not None:
            ts = _first_value(loads_expo, self.LOAD_TIMESTAMP)
            if ts is not None:
                observed = ts

        def stale() -> bool:
            return observed is not None and (now - observed) > self.staleness_s

        # Gauges from /v1/loads.
        for field, (fam_name, kind) in self.LOAD_GAUGES.items():
            if loads_expo is None:
                quality[field] = MetricQuality.SCRAPE_FAILED.value
                continue
            value = _first_value(loads_expo, fam_name)
            if value is None:
                quality[field] = MetricQuality.DISABLED.value
            else:
                kwargs[field] = _to_int(value) if kind == "int" else value
                if stale():
                    quality[field] = MetricQuality.STALE.value

        # Always-on preserved gauges from /v1/loads.
        for key, (fam_name, kind) in self.LOAD_PRESERVED.items():
            if loads_expo is None:
                quality[key] = MetricQuality.SCRAPE_FAILED.value
                continue
            value = _first_value(loads_expo, fam_name)
            if value is None:
                quality[key] = MetricQuality.DISABLED.value
            else:
                engine_specific[key] = _to_int(value) if kind == "int" else value
                if stale():
                    quality[key] = MetricQuality.STALE.value

        # Latency histograms from /metrics.
        if metrics_disabled:
            for fields in _LATENCY_FIELD_NAMES.values():
                quality.update({f: MetricQuality.DISABLED.value for f in fields})
            quality["inter_token_latency"] = MetricQuality.DISABLED.value
        elif metrics_expo is None:
            for fields in _LATENCY_FIELD_NAMES.values():
                quality.update({f: MetricQuality.SCRAPE_FAILED.value for f in fields})
            quality["inter_token_latency"] = MetricQuality.SCRAPE_FAILED.value
        else:
            for family_key, metric_name in self.LATENCY_METRICS.items():
                # Legacy normalized tpot_* and preserved inter_token_latency
                # share one native histogram: one push, both views (M9B). The
                # SGLang family is TOKEN-WEIGHTED (per-token), unlike vLLM's
                # per-step family, and is never equated with it.
                self._record_latency(
                    metrics_expo, family_key, metric_name, now, quality, kwargs,
                    meta=meta,
                    engine_specific=engine_specific if family_key == "tpot" else None,
                    preserved_key="inter_token_latency" if family_key == "tpot" else None,
                )
                if family_key in self.HISTOGRAM_NORMALIZED:
                    self._record_histogram_buckets(meta, family_key, metric_name)
            for key, metric_name in self.PRESERVED_LATENCY.items():
                self._record_preserved_histogram(
                    metrics_expo, key, metric_name, now, quality, engine_specific, meta
                )
            for key, metric_name in self.PRESERVED_STAGE_LATENCY.items():
                self._record_preserved_stage_histogram(
                    metrics_expo, key, metric_name, now, quality, engine_specific, meta
                )
                for stage, out_key in self.HISTOGRAM_STAGE_FAMILIES.items():
                    self._record_histogram_buckets(
                        meta, out_key, f"{metric_name}::{stage}"
                    )

        # Explicit availability for preserved /metrics families when the surface
        # is disabled or unreachable (never fabricated values).
        if metrics_disabled or metrics_expo is None:
            state = (
                MetricQuality.DISABLED.value
                if metrics_disabled
                else MetricQuality.SCRAPE_FAILED.value
            )
            for key in list(self.PRESERVED_LATENCY) + list(self.PRESERVED_STAGE_LATENCY):
                quality.setdefault(key, state)

        # Metrics-only preserved scalars from /metrics.
        for key, (fam_name, kind) in self.METRICS_PRESERVED.items():
            if metrics_disabled:
                quality[key] = MetricQuality.DISABLED.value
                continue
            if metrics_expo is None:
                quality[key] = MetricQuality.SCRAPE_FAILED.value
                continue
            if kind in ("sum_int", "sum_float"):
                value = _sum_values(metrics_expo, fam_name)
            else:
                value = _first_value(metrics_expo, fam_name)
            if value is None:
                quality[key] = MetricQuality.DISABLED.value
            elif kind in ("sum_int", "int"):
                engine_specific[key] = _to_int(value)
            else:
                engine_specific[key] = value

        return RuntimeSnapshot(
            engine=self.engine,
            instance_id=self.instance_id,
            collected_at=collected_at,
            source=self.source_desc,
            engine_specific=engine_specific,
            quality=quality,
            measurement_meta=meta,
            **kwargs,
        )