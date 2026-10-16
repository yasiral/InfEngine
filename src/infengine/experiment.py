"""M6B-2: client-side controlled experiment runner (M6A r3 contract).

A single-instance A/B runner that compares two runtime policies on one
dedicated engine through the EXISTING service API:

  * control plane: ``PATCH /runtime/config`` (Control Contract V2 + Reconciler +
    M6B-1 ``If-Match``), ``GET /runtime/config``, ``GET /runtime/health``,
    ``GET /runtime/capabilities``, ``GET /runtime/status``;
  * measurement: phase-scoped out-of-band telemetry collectors (M5B) constructed
    client-side, scraping the engine's native surfaces;
  * workload: closed-loop requests to the engine's native ``/v1/completions``.

The runner performs no engine mutation of its own beyond the policy PATCHes and
never proxies inference. It implements the M6A threshold-based verdict procedure
(IMPROVED / DEGRADED / NO_MATERIAL_IMPROVEMENT / INCONCLUSIVE / FAILED) with a
mandatory ``verdict_basis`` and always attempts conditional (``If-Match``)
rollback to the pre-experiment DesiredState. Configuration convergence (control
fact) and performance improvement (telemetry fact) are recorded separately.

Not implemented (per M6A direction): autotuning, routing, fleet orchestration,
dashboards, a database, or a new server-side experiment service. ``repeats > 1``
and ``load_model == "open_loop"`` are out of M6B-2 scope (single closed-loop run).

M5C (this file's GPU integration): a non-engine GPU telemetry surface
(``telemetry.gpu``) is polled in the same measurement loop and attached to each
phase as DIAGNOSTIC evidence (``phases[].gpu``: per-phase median/max per signal).
GPU metrics never gate the verdict and are never normalized into the engine
``RuntimeSnapshot``.
"""

import json
import math
import os
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

from .telemetry import (
    GpuTelemetryCollector,
    MetricQuality,
    SglangTelemetryCollector,
    TelemetryCollector,
    VllmTelemetryCollector,
)

# ---------------------------------------------------------------------- errors
class ExperimentError(RuntimeError):
    """Base error for the experiment runner."""


class ControlPlaneError(ExperimentError):
    """The control-plane service could not be reached or queried."""


class WorkloadTimeout(ExperimentError):
    """A workload request exceeded the client-side timeout (censored evidence)."""


class WorkloadError(ExperimentError):
    """A workload request failed (HTTP error / connection error / transport)."""


# Tolerance for the sole-traffic invariant (M6A §11 detection heuristic): the
# engine snapshot and the driver's in-flight read are taken a few ms apart, and
# completions in that window can transiently make running+waiting exceed
# in_flight (most visible under open-loop bursty arrivals). A foreign flood
# exceeds this by far.
_INVARIANT_TOLERANCE = 4


# ----------------------------------------------------------------------- specs
@dataclass
class WorkloadSpec:
    """Offered load (M6A §4). Exactly one load model is used:

    * ``closed_loop`` (M6A §4.1): fixed offered concurrency (``clients``
      workers, arrivals gated on completions), bounded by ``measurement_s``.
    * ``open_loop`` (M6A §4.2): fixed arrival rate (``arrival_rate`` requests/s,
      seeded Poisson inter-arrival, arrivals independent of completions) for
      ``requests`` arrivals; ``requests / arrival_rate`` is the nominal
      measurement duration.

    Shared: ``timeout_s`` (client-side per-request timeout; a timeout is
    abandoned and censors latency) and ``drain_budget_s`` (bounded phase-end
    drain, <= timeout_s per M6A §7.2)."""

    load_model: str = "closed_loop"
    # closed_loop fields
    clients: int = 8
    measurement_s: float = 180.0
    max_requests: int = 4096
    # open_loop fields
    arrival_rate: float = 10.0
    requests: int = 600
    # shared fields
    prompt_tokens: Any = 256  # int, or (min, max) sampled from the seed
    max_new_tokens: int = 64
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 1
    timeout_s: float = 60.0
    drain_budget_s: float = 10.0


@dataclass
class MeasurementSpec:
    """Measurement window + evidence-adequacy thresholds (M6A §7–8)."""

    window_s: float = 60.0
    poll_interval_s: float = 5.0
    min_polls: int = 6
    min_requests: int = 128
    stale_fraction_max: float = 0.5
    decision_metrics: List[str] = field(
        default_factory=lambda: [
            "e2e_p50_s",
            "ttft_p95_s",
            "kv_usage",
            "throughput_req_s",
            "throughput",
        ]
    )


@dataclass
class SuccessCriteria:
    """Threshold-based verdict criteria (M6A §8). Deliberately opaque inputs,
    never constants; all verdicts are threshold comparisons, not statistical
    tests.

    ``throughput_metric`` names the metric whose delta feeds the throughput
    clause of the verdict. The default is ``throughput_req_s`` — the
    runner-derived benchmark (requests completed in the measurement interval /
    actual measurement duration). Engine-native token-throughput gauges
    (``throughput`` = SGLang ``gen_throughput`` / vLLM derived token rate) are
    preserved as diagnostic telemetry and are only gating when explicitly
    selected here (M6B-2.1). Requests/sec and tokens/sec are never equated.
    """

    primary: str = "e2e_p50_s"  # lower-is-better latency metric
    throughput_metric: str = "throughput_req_s"
    improvement: Dict[str, float] = field(
        default_factory=lambda: {
            "throughput_min_relative": 0.10,
            "latency_max_relative": 0.05,
        }
    )
    degradation: Dict[str, float] = field(
        default_factory=lambda: {
            "latency_degrade_relative": 0.10,
            "throughput_drop_relative": -0.10,
        }
    )


@dataclass
class RollbackSpec:
    """Conditional rollback to the pre-experiment DesiredState (M6A §9)."""

    target: str = "pre_experiment"
    verify: bool = True
    conditional: bool = True


@dataclass
class ExperimentSpec:
    """One A/B experiment on one dedicated engine instance (M6A §3)."""

    experiment_id: str
    engine: str  # "vllm" | "sglang"
    model: str
    baseline: Dict[str, Any]
    candidate: Dict[str, Any]
    workload: WorkloadSpec = field(default_factory=WorkloadSpec)
    warmup_requests: int = 64
    measurement: MeasurementSpec = field(default_factory=MeasurementSpec)
    criteria: SuccessCriteria = field(default_factory=SuccessCriteria)
    rollback: RollbackSpec = field(default_factory=RollbackSpec)
    require_if_match: bool = True
    repeats: int = 1
    alternate: bool = False


@dataclass
class RunnerConfig:
    """Runtime wiring for one run (client-side; no server changes)."""

    control_url: str
    engine_url: str
    telemetry_url: Optional[str] = None  # defaults to engine_url
    collector_factory: Optional[Callable[[str], TelemetryCollector]] = None
    send_request: Optional[Callable[[str], None]] = None
    http_timeout: float = 10.0
    # M5C: optional non-engine GPU telemetry (diagnostic only). When set, the
    # measurement loop polls it at the same cadence and attaches the per-phase
    # summary to the result; it never gates the verdict.
    gpu_collector: Optional[GpuTelemetryCollector] = None


# ---------------------------------------------------------------------- result
@dataclass
class PhaseResult:
    name: str
    ok: bool = True
    failure_reason: Optional[str] = None
    convergence: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    counts: Dict[str, int] = field(default_factory=dict)
    wall_s: float = 0.0
    metrics: Dict[str, Any] = field(default_factory=dict)
    rev_after_patch: Optional[int] = None
    # M5C: per-phase GPU diagnostic summary (never verdict-gating). Empty when
    # no gpu_collector was configured.
    gpu: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "convergence": self.convergence,
            "environment": self.environment,
            "counts": self.counts,
            "wall_s": round(self.wall_s, 3),
            "metrics": self.metrics,
            "rev_after_patch": self.rev_after_patch,
            "gpu": self.gpu,
            "ok": self.ok,
            "failure_reason": self.failure_reason,
        }

    def payload(self) -> Dict[str, Any]:
        return {"metrics": self.metrics, "counts": self.counts}


@dataclass
class ExperimentResult:
    """JSON-serializable result (M6A §13). ``verdict_basis`` is mandatory so a
    consumer can always tell which fact (convergence/evidence/environment) the
    verdict rests on. For ``repeats > 1``, ``cycles`` holds every cycle's
    measurements/deltas/adequacy/verdict and ``aggregate`` holds the M6A §8.4
    strict aggregation (median-of-cycle deltas, spread, drift, and any failed /
    inadequate cycles). ``phases`` is a convenience view of the LAST cycle's
    phases."""

    experiment_id: str
    engine: str
    model: str
    spec: Dict[str, Any]
    started_at: str
    finished_at: str = ""
    desired_revision: Dict[str, Any] = field(default_factory=dict)
    phases: Dict[str, Any] = field(default_factory=dict)
    cycles: List[Dict[str, Any]] = field(default_factory=list)
    aggregate: Dict[str, Any] = field(default_factory=dict)
    comparison: Dict[str, Any] = field(default_factory=dict)
    verdict: str = ""
    verdict_basis: Dict[str, Any] = field(default_factory=dict)
    failure_reason: Optional[str] = None
    rollback: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "engine": self.engine,
            "model": self.model,
            "spec": self.spec,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "desired_revision": self.desired_revision,
            "phases": self.phases,
            "cycles": self.cycles,
            "aggregate": self.aggregate,
            "comparison": self.comparison,
            "verdict": self.verdict,
            "verdict_basis": self.verdict_basis,
            "failure_reason": self.failure_reason,
            "rollback": self.rollback,
        }


# ---------------------------------------------------------------------- utils
def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _spec_dict(spec: ExperimentSpec) -> Dict[str, Any]:
    return {
        "experiment_id": spec.experiment_id,
        "engine": spec.engine,
        "model": spec.model,
        "baseline": dict(spec.baseline),
        "candidate": dict(spec.candidate),
        "warmup_requests": spec.warmup_requests,
        "require_if_match": spec.require_if_match,
        "repeats": spec.repeats,
        "alternate": spec.alternate,
        "workload": {
            "load_model": spec.workload.load_model,
            "clients": spec.workload.clients,
            "measurement_s": spec.workload.measurement_s,
            "max_requests": spec.workload.max_requests,
            "arrival_rate": spec.workload.arrival_rate,
            "requests": spec.workload.requests,
            "prompt_tokens": spec.workload.prompt_tokens,
            "max_new_tokens": spec.workload.max_new_tokens,
            "temperature": spec.workload.temperature,
            "top_p": spec.workload.top_p,
            "seed": spec.workload.seed,
            "timeout_s": spec.workload.timeout_s,
            "drain_budget_s": spec.workload.drain_budget_s,
        },
        "measurement": {
            "window_s": spec.measurement.window_s,
            "poll_interval_s": spec.measurement.poll_interval_s,
            "min_polls": spec.measurement.min_polls,
            "min_requests": spec.measurement.min_requests,
            "stale_fraction_max": spec.measurement.stale_fraction_max,
            "decision_metrics": list(spec.measurement.decision_metrics),
        },
        "criteria": {
            "primary": spec.criteria.primary,
            "throughput_metric": spec.criteria.throughput_metric,
            "improvement": dict(spec.criteria.improvement),
            "degradation": dict(spec.criteria.degradation),
        },
        "rollback": {
            "target": spec.rollback.target,
            "verify": spec.rollback.verify,
            "conditional": spec.rollback.conditional,
        },
    }


_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel",
    "india", "juliet", "kilo", "lima", "mike", "november", "oscar", "papa",
    "quebec", "romeo", "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]


def _build_prompt_pool(prompt_tokens: Any, size: int, seed: int) -> List[str]:
    """Deterministic request stream: the same seed reproduces the same prompts
    (count, order, lengths) in every phase."""
    rng = random.Random(seed)
    if isinstance(prompt_tokens, (tuple, list)):
        lo, hi = int(prompt_tokens[0]), int(prompt_tokens[1])
    else:
        lo = hi = int(prompt_tokens)
    pool = []
    for _ in range(max(0, int(size))):
        n = rng.randint(lo, hi) if hi > lo else lo
        pool.append(" ".join(rng.choice(_WORDS) for _ in range(n)))
    return pool


# ------------------------------------------------------------ default transport
def _default_send_request(engine_url: str, spec: WorkloadSpec) -> Callable[[str], None]:
    """POST ``{prompt}`` to the engine's native /v1/completions with the spec's
    sampling args. A client-side ``timeout_s`` is enforced by the socket
    timeout; a timeout raises WorkloadTimeout (censored evidence)."""
    url = f"{engine_url.rstrip('/')}/v1/completions"

    def send(prompt: str) -> None:
        body = json.dumps(
            {
                "prompt": prompt,
                "max_tokens": spec.max_new_tokens,
                "temperature": spec.temperature,
                "top_p": spec.top_p,
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=spec.timeout_s) as resp:
                if resp.status != 200:
                    raise WorkloadError(f"completion HTTP {resp.status}")
                resp.read()
        except urllib.error.HTTPError as exc:
            raise WorkloadError(f"completion HTTP {exc.code}") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, TimeoutError) or "timed out" in str(exc.reason):
                raise WorkloadTimeout(str(exc)) from exc
            raise WorkloadError(str(exc)) from exc
        except TimeoutError as exc:
            raise WorkloadTimeout(str(exc)) from exc
        except OSError as exc:
            raise WorkloadError(str(exc)) from exc

    return send


# -------------------------------------------------------------- control plane
class ControlPlaneClient:
    """Thin HTTP client for the existing service API (client-side, urllib)."""

    def __init__(self, base_url: str, timeout: float = 10.0,
                 token: Optional[str] = None, actor: Optional[str] = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.token = token if token is not None else os.environ.get("INFENGINE_TOKEN")
        self.actor = actor if actor is not None else os.environ.get("INFENGINE_ACTOR")

    def _auth_headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if self.actor:
            headers["X-InfEngine-Actor"] = self.actor
        return headers

    def _request(
        self, method: str, path: str, body: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> Tuple[int, Dict[str, Any]]:
        url = f"{self.base_url}{path}"
        data = None
        hdrs = dict(self._auth_headers())
        hdrs.update(headers or {})
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            hdrs["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
                return resp.status, json.loads(raw) if raw else {}
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
            try:
                return exc.code, json.loads(raw) if raw else {}
            except json.JSONDecodeError:
                return exc.code, {"detail": raw}
        except (urllib.error.URLError, OSError) as exc:
            raise ControlPlaneError(f"{method} {url} failed: {exc}") from exc

    def info(self) -> Dict[str, Any]:
        status, payload = self._request("GET", "/runtime/info")
        if status == 404:
            return {}
        if status != 200:
            raise ControlPlaneError(f"/runtime/info returned HTTP {status}")
        return payload

    def history(self, limit: int = 50, offset: int = 0) -> Tuple[int, Dict[str, Any]]:
        return self._request("GET", f"/runtime/history?limit={int(limit)}&offset={int(offset)}")

    def history_one(self, change_id: str) -> Tuple[int, Dict[str, Any]]:
        return self._request("GET", f"/runtime/history/{change_id}")

    def health(self) -> Dict[str, Any]:
        status, payload = self._request("GET", "/health")
        if status != 200:
            raise ControlPlaneError(f"/health returned HTTP {status}")
        return payload

    def capabilities(self) -> List[Dict[str, Any]]:
        status, payload = self._request("GET", "/runtime/capabilities")
        if status != 200:
            raise ControlPlaneError(f"/runtime/capabilities returned HTTP {status}")
        return payload if isinstance(payload, list) else []

    def status(self) -> Dict[str, Any]:
        status, payload = self._request("GET", "/runtime/status")
        if status != 200:
            raise ControlPlaneError(f"/runtime/status returned HTTP {status}")
        return payload

    def get_config(self) -> Dict[str, Any]:
        status, payload = self._request("GET", "/runtime/config")
        if status != 200:
            raise ControlPlaneError(f"/runtime/config returned HTTP {status}")
        return payload

    def patch_config(
        self, desired: Dict[str, Any], if_match: Optional[int] = None
    ) -> Tuple[int, Dict[str, Any]]:
        headers = {"If-Match": str(if_match)} if if_match is not None else {}
        return self._request("PATCH", "/runtime/config", body=desired, headers=headers)


# ------------------------------------------------------------- workload driver
class ClosedLoopWorkloadDriver:
    """Fixed-offered-concurrency workload (M6A §4.1) with request-outcome
    accounting (M6A §4.4): completed / timeout / error / incomplete, plus a
    post-measurement drain that is NOT counted as measurement throughput."""

    def __init__(self, spec: WorkloadSpec, send: Callable[[str], None], pool: List[str]):
        self.spec = spec
        self._send = send
        self.pool = pool
        self._lock = threading.Lock()
        self._next_index = 0
        self._started = 0
        self._completed = 0
        self._drained = 0
        self._timeout = 0
        self._error = 0
        self._incomplete = 0
        self._inflight_ids = set()
        self._abandoned = set()
        self._t1: Optional[float] = None
        self._stop = threading.Event()
        self._workers: List[threading.Thread] = []
        # benchmark throughput (M6B-2.1): requests completed in the measurement
        # interval / the ACTUAL measurement span (first start -> last
        # measurement-period completion). Post-measurement drain completions
        # never enter it.
        self._first_start_ts: Optional[float] = None
        self._last_completion_ts: Optional[float] = None

    def start_measurement(self) -> None:
        for _ in range(max(1, self.spec.clients)):
            t = threading.Thread(target=self._run, daemon=True,
                                 name="infengine-experiment-client")
            self._workers.append(t)
            t.start()

    def _next_prompt(self) -> Optional[int]:
        with self._lock:
            if self._stop.is_set():
                return None
            if self._next_index >= len(self.pool):
                return None
            idx = self._next_index
            self._next_index += 1
            return idx

    def _run(self) -> None:
        while True:
            idx = self._next_prompt()
            if idx is None:
                return
            rid = object()
            with self._lock:
                self._started += 1
                self._inflight_ids.add(rid)
                if self._first_start_ts is None:
                    self._first_start_ts = time.monotonic()
            try:
                self._send(self.pool[idx])
                outcome = "completed"
            except WorkloadTimeout:
                outcome = "timeout"
            except WorkloadError:
                outcome = "error"
            tc = time.monotonic()
            with self._lock:
                self._inflight_ids.discard(rid)
                if rid in self._abandoned:
                    self._abandoned.discard(rid)
                    return  # already counted incomplete at phase end
                if outcome == "completed":
                    if self._t1 is None or tc <= self._t1:
                        self._completed += 1
                        self._last_completion_ts = tc
                    else:
                        self._drained += 1
                elif outcome == "timeout":
                    self._timeout += 1
                else:
                    self._error += 1

    def stop_measurement(self, t1: Optional[float], drain_budget_s: float) -> None:
        """Stop issuing at ``t1`` and drain in-flight requests within the bounded
        budget. Requests still in flight after the budget are recorded
        ``incomplete`` (censored evidence, never counted as throughput)."""
        with self._lock:
            self._t1 = t1
            self._stop.set()
        deadline = time.monotonic() + drain_budget_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._inflight_ids:
                    break
            time.sleep(0.01)
        with self._lock:
            self._abandoned.update(self._inflight_ids)
            self._incomplete += len(self._inflight_ids)
        for t in self._workers:
            t.join(timeout=self.spec.timeout_s + 2.0)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._inflight_ids)

    def benchmark(self) -> Dict[str, Any]:
        """Runner-derived benchmark throughput (M6B-2.1): requests completed in
        the measurement interval divided by the ACTUAL measurement span
        (first request start -> last measurement-period completion). Drain
        completions are excluded by construction. ``value`` is None when there
        is no usable span (no measurement-period completions)."""
        with self._lock:
            completed = self._completed
            first = self._first_start_ts
            last = self._last_completion_ts
        if completed <= 0 or first is None or last is None or last <= first:
            return {"value": None, "completed": completed, "duration_s": 0.0}
        duration = last - first
        return {
            "value": completed / duration,
            "completed": completed,
            "duration_s": duration,
        }

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "started": self._started,
                "completed": self._completed,
                "drained": self._drained,
                "timeout": self._timeout,
                "error": self._error,
                "incomplete": self._incomplete,
                "in_flight": len(self._inflight_ids),
            }


class OpenLoopWorkloadDriver:
    """Fixed-arrival-rate workload (M6A §4.2).

    A seeded timer emits exactly ``requests`` arrivals at ``arrival_rate``
    requests/sec (Poisson inter-arrival, seeded), INDEPENDENT of completions.
    Each arrival is sent and awaited with a client-side ``timeout_s``; a request
    that does not complete in time is abandoned and recorded ``timeout``
    (censored evidence). If the engine cannot keep up, in-flight arrivals pile
    up (bounded by ``timeout_s``) and the engine-side queue grows — a
    legitimate observed outcome.

    Outcome accounting, the bounded phase-end drain (never counted as
    measurement throughput), and the runner-derived completion-rate benchmark
    match ``ClosedLoopWorkloadDriver``.
    """

    def __init__(self, spec: WorkloadSpec, send: Callable[[str], None], pool: List[str]):
        self.spec = spec
        self._send = send
        self.pool = pool
        self._lock = threading.Lock()
        self._next_index = 0
        self._started = 0
        self._completed = 0
        self._drained = 0
        self._timeout = 0
        self._error = 0
        self._incomplete = 0
        self._inflight: Dict[int, bool] = {}
        self._abandoned: set = set()
        self._t1: Optional[float] = None
        self._stop = threading.Event()
        self._workers: List[threading.Thread] = []
        self._arrival_thread: Optional[threading.Thread] = None
        self._first_start_ts: Optional[float] = None
        self._last_completion_ts: Optional[float] = None
        self._emission_end_ts: Optional[float] = None

    def start_measurement(self) -> None:
        self._arrival_thread = threading.Thread(target=self._emit, daemon=True,
                                                name="infengine-experiment-arrivals")
        self._arrival_thread.start()

    def _next_prompt(self) -> Optional[int]:
        with self._lock:
            if self._next_index >= len(self.pool):
                return None
            idx = self._next_index
            self._next_index += 1
            return idx

    def _emit(self) -> None:
        rng = random.Random(self.spec.seed)
        for _ in range(max(0, int(self.spec.requests))):
            if self._stop.is_set():
                break
            idx = self._next_prompt()
            if idx is None:
                break
            t = threading.Thread(target=self._send_one, args=(idx,), daemon=True,
                                 name="infengine-experiment-openloop")
            with self._lock:
                self._started += 1
                self._inflight[idx] = False
                if self._first_start_ts is None:
                    self._first_start_ts = time.monotonic()
                self._workers.append(t)
            t.start()
            # seeded Poisson inter-arrival (mean 1 / arrival_rate)
            self._stop.wait(rng.expovariate(self.spec.arrival_rate))
        with self._lock:
            self._emission_end_ts = time.monotonic()

    def _send_one(self, idx: int) -> None:
        try:
            self._send(self.pool[idx])
            outcome = "completed"
        except WorkloadTimeout:
            outcome = "timeout"
        except WorkloadError:
            outcome = "error"
        tc = time.monotonic()
        with self._lock:
            if idx in self._abandoned:
                self._abandoned.discard(idx)
                self._inflight.pop(idx, None)
                return  # already counted incomplete at phase end
            self._inflight.pop(idx, None)
            if outcome == "completed":
                if self._t1 is None or tc <= self._t1:
                    self._completed += 1
                    self._last_completion_ts = tc
                else:
                    self._drained += 1
            elif outcome == "timeout":
                self._timeout += 1
            else:
                self._error += 1

    def emission_done(self) -> bool:
        with self._lock:
            return self._emission_end_ts is not None

    def emission_end_ts(self) -> Optional[float]:
        with self._lock:
            return self._emission_end_ts

    def stop_measurement(self, t1: Optional[float], drain_budget_s: float) -> None:
        """Stop emitting at ``t1`` and drain in-flight arrivals within the bounded
        budget. Requests still in flight after the budget are ``incomplete``
        (censored evidence, never counted as throughput)."""
        with self._lock:
            self._t1 = t1
            self._stop.set()
        if self._arrival_thread is not None:
            self._arrival_thread.join(timeout=self.spec.timeout_s + 2.0)
        deadline = time.monotonic() + drain_budget_s
        while time.monotonic() < deadline:
            with self._lock:
                if not self._inflight:
                    break
            time.sleep(0.01)
        with self._lock:
            self._abandoned.update(self._inflight.keys())
            self._incomplete += len(self._inflight)
        with self._lock:
            workers = list(self._workers)
        for t in workers:
            t.join(timeout=self.spec.timeout_s + 2.0)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._inflight)

    def benchmark(self) -> Dict[str, Any]:
        """Runner-derived completion rate: measurement-period completions /
        the ACTUAL measurement span (first arrival start -> last measurement
        completion). Offered rate is ``spec.arrival_rate``."""
        with self._lock:
            completed = self._completed
            first = self._first_start_ts
            last = self._last_completion_ts
        if completed <= 0 or first is None or last is None or last <= first:
            return {"value": None, "completed": completed, "duration_s": 0.0}
        duration = last - first
        return {
            "value": completed / duration,
            "completed": completed,
            "duration_s": duration,
        }

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {
                "started": self._started,
                "completed": self._completed,
                "drained": self._drained,
                "timeout": self._timeout,
                "error": self._error,
                "incomplete": self._incomplete,
                "in_flight": len(self._inflight),
            }


# ------------------------------------------------------------------- telemetry
def _default_collector_factory(spec: ExperimentSpec) -> Callable[[str], TelemetryCollector]:
    def factory(url: str) -> TelemetryCollector:
        kw = {
            "window_s": spec.measurement.window_s,
            "poll_interval": spec.measurement.poll_interval_s,
        }
        if spec.engine == "sglang":
            return SglangTelemetryCollector(url, **kw)
        return VllmTelemetryCollector(url, **kw)

    return factory


_LATENCY_SUFFIXES = ("_p50_s", "_p95_s", "_p99_s")


def _is_latency(metric: str) -> bool:
    return metric.endswith(_LATENCY_SUFFIXES)


def _metric_value(snapshot: Any, metric: str, engine: str) -> Any:
    if metric == "throughput":
        if engine == "sglang":
            return snapshot.engine_specific.get("gen_throughput")
        if engine == "vllm":
            return snapshot.engine_specific.get("generation_tokens")
        return None
    return getattr(snapshot, metric, None)


def _metric_quality(snapshot: Any, metric: str, engine: str) -> Optional[MetricQuality]:
    if metric == "throughput":
        key = "gen_throughput" if engine == "sglang" else "generation_tokens"
        return snapshot.quality_of(key)
    return snapshot.quality_of(metric)


def _metric_summary(
    in_window: List[Tuple[float, Any, int]], metric: str, engine: str,
    counts: Dict[str, int], spec: ExperimentSpec,
) -> Dict[str, Any]:
    """Per-metric windowed summary: median of fresh per-snapshot values, spread,
    dominant quality, stale fraction, polls usable, and the phase's completed
    request count (the ``min_requests`` gate)."""
    fresh: List[float] = []
    stale: List[float] = []
    other: List[Tuple[MetricQuality, float]] = []
    for _, snapshot, _ in in_window:
        q = _metric_quality(snapshot, metric, engine)
        v = _metric_value(snapshot, metric, engine)
        if v is None:
            continue
        if q is MetricQuality.FRESH:
            fresh.append(v)
        elif q is MetricQuality.STALE:
            stale.append(v)
        else:
            other.append((q, v))
    polls_usable = len(fresh)
    stale_fraction = len(stale) / max(1, len(fresh) + len(stale))
    if fresh:
        dominant = "fresh"
        value = statistics.median(fresh)
        spread = [min(fresh), max(fresh)]
    elif stale:
        dominant = "stale"
        value = statistics.median(stale)
        spread = [min(stale), max(stale)]
    else:
        dominant = other[0][0].value if other else "unavailable"
        value = None
        spread = None
    return {
        "value": value,
        "spread": spread,
        "dominant_quality": dominant,
        "stale_fraction": round(stale_fraction, 4),
        "polls_usable": polls_usable,
        "sample_count": counts.get("completed", 0),
    }


def _throughput_summary(
    in_window: List[Tuple[float, Any, int]], engine: str,
    counts: Dict[str, int],
) -> Dict[str, Any]:
    """Throughput is engine-specific, never normalized (M5A §4): SGLang reads
    the windowed ``gen_throughput`` gauge (median over fresh polls); vLLM
    derives ``rate(generation_tokens)`` across the phase window from the
    cumulative counter."""
    key = "gen_throughput" if engine == "sglang" else "generation_tokens"
    fresh: List[Tuple[float, float]] = []
    stale: int = 0
    for wall, snapshot, _ in in_window:
        q = _metric_quality(snapshot, "throughput", engine)
        v = _metric_value(snapshot, "throughput", engine)
        if v is None:
            continue
        if q is MetricQuality.FRESH:
            fresh.append((wall, float(v)))
        elif q is MetricQuality.STALE:
            stale += 1
    stale_fraction = stale / max(1, len(fresh) + stale)
    if engine == "sglang":
        vals = [v for _, v in fresh]
        value = statistics.median(vals) if vals else None
        spread = [min(vals), max(vals)] if vals else None
    else:
        ordered = sorted(fresh, key=lambda t: t[0])
        value = None
        spread = None
        if len(ordered) >= 2:
            t0, g0 = ordered[0]
            t1, g1 = ordered[-1]
            dt = t1 - t0
            value = (g1 - g0) / dt if dt > 0 else 0.0
    dominant = "fresh" if fresh else ("stale" if stale else "unavailable")
    return {
        "value": value,
        "spread": spread,
        "dominant_quality": dominant,
        "stale_fraction": round(stale_fraction, 4),
        "polls_usable": len(fresh),
        "sample_count": counts.get("completed", 0),
    }


def _summarize(
    spec: ExperimentSpec,
    records: List[Tuple[float, Any, int]],
    summarize_start: float,
    t1: float,
    counts: Dict[str, int],
) -> Dict[str, Any]:
    """Phase summary over ``[summarize_start, t1]``. Ramp-in (the first
    ``window_s`` of measurement) is excluded so latency values carry a full
    window of measurement traffic (M6A §7.2)."""
    in_window = [
        (w, s, f) for (w, s, f) in records
        if w >= summarize_start and w <= t1
    ]
    metrics: Dict[str, Any] = {}
    for m in spec.measurement.decision_metrics:
        if m == "throughput":
            metrics[m] = _throughput_summary(in_window, spec.engine, counts)
        else:
            metrics[m] = _metric_summary(in_window, m, spec.engine, counts, spec)
    return metrics


# M5C: per-phase GPU signals summarized (diagnostic only, never verdict-gating).
_GPU_SIGNAL_FIELDS = (
    "gpu_util",
    "sm_clock_mhz",
    "memory_used_mb",
    "memory_free_mb",
    "memory_total_mb",
    "power_w",
    "power_limit_w",
    "temperature_c",
)


def _gpu_summary(records, lo: float, hi: float) -> Dict[str, Any]:
    """Per-phase GPU diagnostic summary over polls in ``[lo, hi]``: per-signal
    median/max (and polls usable), availability, and the optional engine-PID
    attribution when process collection was enabled. Diagnostic only — never
    feeds the verdict."""
    in_window = [(w, s) for (w, s) in records if lo <= w <= hi]
    available = any(s.available for _, s in in_window)
    reason: Optional[str] = None
    if not available:
        reason = next((s.unavailable_reason for _, s in in_window if not s.available), None)
    signals: Dict[str, Any] = {}
    engine_process: Optional[Dict[str, Any]] = None
    if available:
        for field in _GPU_SIGNAL_FIELDS:
            vals = [
                getattr(s, field)
                for _, s in in_window
                if s.available and getattr(s, field) is not None
            ]
            if vals:
                signals[field] = {
                    "median": round(statistics.median(vals), 4),
                    "max": round(max(vals), 4),
                    "polls_usable": len(vals),
                }
        ep_present = [s.engine_process for _, s in in_window if s.engine_process is not None]
        if ep_present:
            mem_vals = [
                p["used_memory_mb"] for p in ep_present
                if p.get("used_memory_mb") is not None
            ]
            util_vals = [
                p["sm_util_pct"] for p in ep_present
                if p.get("sm_util_pct") is not None
            ]
            engine_process = {
                "pid": ep_present[0]["pid"],
                "polls_present": len(ep_present),
                "used_memory_mb": (
                    {"median": round(statistics.median(mem_vals), 4),
                     "max": round(max(mem_vals), 4)} if mem_vals else None
                ),
                "sm_util_pct": (
                    {"median": round(statistics.median(util_vals), 4),
                     "max": round(max(util_vals), 4)} if util_vals else None
                ),
            }
    return {
        "configured": True,
        "available": available,
        "unavailable_reason": reason,
        "polls": len(in_window),
        "signals": signals,
        "engine_process": engine_process,
    }


# ------------------------------------------------------------------ adequacy
def _adequacy_reason(payload: Dict[str, Any], metric: str, spec: ExperimentSpec) -> str:
    counts = payload.get("counts") or {}
    sm = (payload.get("metrics") or {}).get(metric) or {}
    if metric == "throughput_req_s":
        # runner-derived benchmark: no scrape polls / staleness / dominant
        # quality; it is the driver's real completed-requests rate.
        if sm.get("value") is None:
            return "unavailable"
        if counts.get("completed", 0) < spec.measurement.min_requests:
            return "low_sample"
        return "adequate"
    if _is_latency(metric):
        if counts.get("timeout", 0) > 0:
            return "censored_timeouts"
        if counts.get("incomplete", 0) > 0:
            return "censored_incomplete"
        if counts.get("error", 0) > 0:
            return "censored_error"
    if sm.get("dominant_quality") not in ("fresh", "stale"):
        return "unavailable"  # disabled / no_samples / scrape_failed dominant
    if sm.get("polls_usable", 0) < spec.measurement.min_polls:
        return "insufficient_polls"
    if sm.get("stale_fraction", 1.0) > spec.measurement.stale_fraction_max:
        return "stale"
    if counts.get("completed", 0) < spec.measurement.min_requests:
        return "low_sample"
    return "adequate"


def _referenced_metrics(spec: ExperimentSpec) -> set:
    """Decision metrics referenced by the criteria (M6A §8.2): the primary
    latency metric plus the explicitly selected throughput metric when the
    criteria carry throughput clauses and it is being measured."""
    A = {spec.criteria.primary}
    thr_metric = spec.criteria.throughput_metric
    if thr_metric in spec.measurement.decision_metrics and (
        "throughput_min_relative" in spec.criteria.improvement
        or "throughput_drop_relative" in spec.criteria.degradation
    ):
        A.add(thr_metric)
    return A


def _apply_thresholds(spec: ExperimentSpec, rels: Dict[str, float]) -> Tuple[str, List[str]]:
    """Threshold-based verdict on a dict of relative deltas (M6A §8.2).
    Deterministic, exhaustive, mutually exclusive; never a statistical claim.
    ``rels`` must contain the primary metric and, when the criteria reference
    it, the selected throughput metric."""
    if spec.criteria.primary not in rels:
        return "INCONCLUSIVE", [f"primary {spec.criteria.primary} delta undefined"]
    dlat = rels[spec.criteria.primary]
    thr_metric = spec.criteria.throughput_metric
    thr = rels.get(thr_metric)
    lat_max = spec.criteria.improvement.get("latency_max_relative")
    thr_min = spec.criteria.improvement.get("throughput_min_relative")
    lat_deg = spec.criteria.degradation.get("latency_degrade_relative")
    thr_drop = spec.criteria.degradation.get("throughput_drop_relative")
    improved = dlat <= lat_max and (thr is None or thr >= thr_min)
    degraded = dlat >= lat_deg or (thr is not None and thr <= thr_drop)
    verdict = "IMPROVED" if improved else ("DEGRADED" if degraded else "NO_MATERIAL_IMPROVEMENT")
    basis = [
        f"primary {spec.criteria.primary} delta={dlat:+.2%} "
        f"(improve <= +{lat_max:.0%}, degrade >= +{lat_deg:.0%})"
    ]
    if thr is not None:
        basis.append(
            f"throughput {thr_metric} delta={thr:+.2%} "
            f"(improve >= +{thr_min:.0%}, degrade <= +{thr_drop:.0%})"
        )
    return verdict, basis


def _compare(
    spec: ExperimentSpec, bp: Dict[str, Any], cp: Dict[str, Any],
) -> Tuple[Dict[str, Any], str, List[str]]:
    """Single-cycle threshold-based comparison (M6A §8.1/§8.2): per-metric
    adequacy + relative deltas + verdict. The throughput clause uses the
    EXPLICITLY selected ``criteria.throughput_metric`` (default
    ``throughput_req_s``), so an unavailable or unreliable native token gauge
    cannot mislead the verdict (M6B-2.1)."""
    A = _referenced_metrics(spec)
    adequacy: Dict[str, Any] = {}
    basis: List[str] = []
    ok = True
    for m in sorted(A):
        rb = _adequacy_reason(bp, m, spec)
        rc = _adequacy_reason(cp, m, spec)
        adequacy[m] = {"baseline": rb, "candidate": rc}
        if rb != "adequate" or rc != "adequate":
            ok = False
            basis.append(f"{m}: baseline={rb}, candidate={rc}")
    deltas: Dict[str, Any] = {}
    if ok:
        for m in sorted(A):
            bv = (bp.get("metrics") or {}).get(m, {}).get("value")
            cv = (cp.get("metrics") or {}).get(m, {}).get("value")
            if bv is None or cv is None or bv == 0:
                ok = False
                basis.append(f"{m}: delta undefined (baseline={bv}, candidate={cv})")
                break
            deltas[m] = {
                "relative": (cv - bv) / bv,
                "baseline_value": bv,
                "candidate_value": cv,
            }
    if not ok:
        return {"deltas": deltas, "adequacy": adequacy}, "INCONCLUSIVE", basis
    rels = {m: d["relative"] for m, d in deltas.items()}
    verdict, threshold_basis = _apply_thresholds(spec, rels)
    return {"deltas": deltas, "adequacy": adequacy}, verdict, basis + threshold_basis


# ------------------------------------------------------- repeats / aggregation
def _drift(spec: ExperimentSpec, bp: Dict[str, Any], tp: Dict[str, Any]) -> Dict[str, Any]:
    """Per-cycle drift check (M6A §10.3): trailing baseline vs leading baseline
    for every decision metric. Reported SEPARATELY; never pooled into the
    baseline/candidate comparison."""
    drift: Dict[str, Any] = {}
    for m in spec.measurement.decision_metrics:
        bv = (bp.get("metrics") or {}).get(m, {}).get("value")
        tv = (tp.get("metrics") or {}).get(m, {}).get("value")
        if bv is None or tv is None or bv == 0:
            continue
        drift[m] = {
            "relative": (tv - bv) / bv,
            "leading_baseline": bv,
            "trailing_baseline": tv,
        }
    return drift


def _run_cycle(
    spec: ExperimentSpec, cfg: RunnerConfig, control: ControlPlaneClient,
    pre: Dict[str, Any], rev_before: Optional[int], index: int,
) -> Tuple[Dict[str, Any], Optional[int]]:
    """One repeated cycle (M6A §10.3): baseline -> candidate -> [trailing
    baseline if ``alternate``], each with explicit phase boundaries and the same
    seeded workload. Returns (record, rev_after). A control/environment failure
    in any phase marks the cycle failed; an inadequate cycle still carries its
    measurements and is NOT silently excluded from the aggregate."""
    intent_key = pre["intent_key"]
    record: Dict[str, Any] = {
        "index": index,
        "phases": {},
        "rev_after_phase": {"baseline": None, "candidate": None, "baseline_trail": None},
        "verdict": "FAILED",
        "basis": [],
        "failed": False,
        "failure_reason": None,
        "deltas": {},
        "adequacy": {},
        "drift": None,
    }
    baseline = _run_phase(spec, cfg, control, "baseline", spec.baseline, intent_key, rev_before)
    record["phases"]["baseline"] = baseline.to_dict()
    record["rev_after_phase"]["baseline"] = baseline.rev_after_patch
    rev_after = baseline.rev_after_patch if baseline.rev_after_patch is not None else rev_before
    if not baseline.ok:
        record["failed"] = True
        record["failure_reason"] = f"baseline phase failed: {baseline.failure_reason}"
        record["basis"] = [record["failure_reason"]]
        return record, rev_after

    cphase = _run_phase(spec, cfg, control, "candidate", spec.candidate, intent_key,
                        baseline.rev_after_patch)
    record["phases"]["candidate"] = cphase.to_dict()
    record["rev_after_phase"]["candidate"] = cphase.rev_after_patch
    rev_after = cphase.rev_after_patch if cphase.rev_after_patch is not None else rev_after
    if not cphase.ok:
        record["failed"] = True
        record["failure_reason"] = f"candidate phase failed: {cphase.failure_reason}"
        record["basis"] = [record["failure_reason"]]
        return record, rev_after

    trail = None
    if spec.alternate:
        trail = _run_phase(spec, cfg, control, "baseline_trail", spec.baseline, intent_key,
                           cphase.rev_after_patch)
        record["phases"]["baseline_trail"] = trail.to_dict()
        record["rev_after_phase"]["baseline_trail"] = trail.rev_after_patch
        rev_after = trail.rev_after_patch if trail.rev_after_patch is not None else rev_after
        if not trail.ok:
            record["failed"] = True
            record["failure_reason"] = f"baseline_trail phase failed: {trail.failure_reason}"
            record["basis"] = [record["failure_reason"]]
            return record, rev_after

    comparison, verdict, basis = _compare(spec, baseline.payload(), cphase.payload())
    record["verdict"] = verdict
    record["basis"] = basis
    record["deltas"] = comparison.get("deltas", {})
    record["adequacy"] = comparison.get("adequacy", {})
    if trail is not None:
        record["drift"] = _drift(spec, baseline.payload(), trail.payload())
    return record, rev_after


def _aggregate_drift(cycles: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    drift_cycles = [c for c in cycles if c.get("drift")]
    if not drift_cycles:
        return None
    keys = set()
    for c in drift_cycles:
        keys.update(c["drift"].keys())
    drift: Dict[str, Any] = {}
    for m in sorted(keys):
        vals = [c["drift"][m]["relative"] for c in drift_cycles if m in c["drift"]]
        if vals:
            drift[m] = {
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "per_cycle": vals,
            }
    return drift


def _aggregate(spec: ExperimentSpec, cycles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """M6A §8.4 strict aggregation across ALL cycles. A failed cycle makes the
    aggregate FAILED; any inadequate cycle makes it INCONCLUSIVE (neither is
    silently excluded). Otherwise the aggregate delta per metric is the MEDIAN
    of the per-cycle deltas with spread (min/max/per_cycle), and the verdict is
    the threshold procedure applied to the median deltas. Drift is aggregated
    separately."""
    per_cycle_verdicts = [c["verdict"] for c in cycles]
    failed = [c for c in cycles if c["failed"]]
    if failed:
        return {
            "verdict": "FAILED",
            "basis": [f"cycle {c['index']} failed: {c['failure_reason']}" for c in failed],
            "failed_cycles": [c["index"] for c in failed],
            "inadequate_cycles": [],
            "per_cycle_verdicts": per_cycle_verdicts,
            "deltas": {},
            "drift": None,
        }
    inadequate = [c for c in cycles if c["verdict"] == "INCONCLUSIVE"]
    if inadequate:
        return {
            "verdict": "INCONCLUSIVE",
            "basis": [
                f"cycle {c['index']}: " + " | ".join(c["basis"][:2])
                for c in inadequate
            ],
            "failed_cycles": [],
            "inadequate_cycles": [c["index"] for c in inadequate],
            "per_cycle_verdicts": per_cycle_verdicts,
            "deltas": {},
            "drift": None,
        }
    A = _referenced_metrics(spec)
    deltas: Dict[str, Any] = {}
    for m in sorted(A):
        vals = [c["deltas"][m]["relative"] for c in cycles if m in c.get("deltas", {})]
        if not vals:
            continue
        deltas[m] = {
            "median": statistics.median(vals),
            "min": min(vals),
            "max": max(vals),
            "per_cycle": vals,
        }
    rels = {m: v["median"] for m, v in deltas.items()}
    if not rels or spec.criteria.primary not in rels:
        return {
            "verdict": "INCONCLUSIVE",
            "basis": ["no aggregate deltas available"],
            "failed_cycles": [],
            "inadequate_cycles": [],
            "per_cycle_verdicts": per_cycle_verdicts,
            "deltas": deltas,
            "drift": _aggregate_drift(cycles),
        }
    verdict, basis = _apply_thresholds(spec, rels)
    return {
        "verdict": verdict,
        "basis": basis,
        "failed_cycles": [],
        "inadequate_cycles": [],
        "per_cycle_verdicts": per_cycle_verdicts,
        "deltas": deltas,
        "drift": _aggregate_drift(cycles),
    }


# ---------------------------------------------------------------- preflight
def _resolve_intent_key(control: ControlPlaneClient, intent: str) -> Optional[str]:
    try:
        for e in control.capabilities():
            if e.get("intent") == intent:
                if e.get("supported_on_selected_engine"):
                    return e.get("internal_key")
                return None
    except ControlPlaneError:
        pass
    try:
        for c in (control.status().get("capabilities") or []):
            if c.get("name") == intent and c.get("supported"):
                return intent
    except ControlPlaneError:
        pass
    return None


def _preflight(
    spec: ExperimentSpec, cfg: RunnerConfig, control: ControlPlaneClient,
) -> Dict[str, Any]:
    notes: List[str] = []
    try:
        h = control.health()
        if not h.get("admitting"):
            return {"ok": False, "reason": "engine not admitting", "notes": notes}
    except ControlPlaneError as exc:
        return {"ok": False, "reason": f"control plane unreachable: {exc}", "notes": notes}

    if len(spec.baseline) != 1 or len(spec.candidate) != 1:
        return {"ok": False, "reason": "baseline/candidate must be a single shared intent"}
    intent = next(iter(spec.baseline))
    if intent not in spec.candidate:
        return {"ok": False, "reason": "baseline and candidate intents differ"}
    if spec.candidate[intent] == spec.baseline[intent]:
        return {"ok": False, "reason": "baseline and candidate must differ"}

    internal_key = _resolve_intent_key(control, intent)
    if internal_key is None:
        return {"ok": False, "reason": f"intent {intent} not supported on selected engine"}

    cfg0 = control.get_config()
    pre_desired = dict(cfg0.get("desired") or {})
    revision = cfg0.get("desired_revision")
    if spec.require_if_match and revision is None:
        return {"ok": False, "reason": "desired_revision not exposed (M6B-1 extension missing); require_if_match"}

    # M6B-2.1: rollback-target feasibility. The control plane's PATCH is
    # merge-only (it can add/overwrite keys but never remove them), so the
    # experiment can restore the pre-experiment DesiredState exactly ONLY if
    # every key the experiment introduces already exists in it. The experiment
    # mutates exactly the resolved internal key, so it must be present in
    # pre_desired. Reject BEFORE any mutation — an experiment that cannot
    # return to its baseline must not run.
    if internal_key not in pre_desired:
        return {
            "ok": False,
            "reason": (
                f"rollback target not exactly restorable: intent internal key "
                f"{internal_key!r} is absent from the pre-experiment DesiredState; "
                "the merge-only PATCH semantics cannot unset a key the experiment "
                "would introduce"
            ),
        }

    # M6B-2.1: the verdict's throughput clause references an explicit metric;
    # fail closed if that metric is not being measured.
    if ("throughput_min_relative" in spec.criteria.improvement
            or "throughput_drop_relative" in spec.criteria.degradation) \
            and spec.criteria.throughput_metric not in spec.measurement.decision_metrics:
        return {
            "ok": False,
            "reason": (
                f"criteria throughput clause references {spec.criteria.throughput_metric!r} "
                "but it is not in decision_metrics"
            ),
        }

    ws = spec.workload
    needed = spec.measurement.window_s + (spec.measurement.min_polls - 1) * spec.measurement.poll_interval_s
    if ws.load_model == "open_loop":
        if ws.arrival_rate <= 0 or int(ws.requests) < 1:
            return {"ok": False, "reason": "open_loop requires arrival_rate > 0 and requests >= 1"}
        nominal_s = int(ws.requests) / max(ws.arrival_rate, 1e-9)
        if nominal_s < needed:
            return {
                "ok": False,
                "reason": (
                    f"open_loop duration requests/arrival_rate ({nominal_s:.2f}s) "
                    f"< window_s + (min_polls-1)*poll_interval ({needed})"
                ),
            }
    elif ws.load_model == "closed_loop":
        if ws.measurement_s < needed:
            return {
                "ok": False,
                "reason": (
                    f"measurement_s {ws.measurement_s} < window_s + (min_polls-1)*poll_interval "
                    f"({needed})"
                ),
            }
    else:
        return {"ok": False, "reason": f"unsupported load_model {ws.load_model!r}"}
    if spec.engine not in ("sglang", "vllm"):
        return {"ok": False, "reason": f"unsupported engine {spec.engine}"}
    if int(spec.repeats) < 1:
        return {"ok": False, "reason": "repeats must be >= 1"}
    if not isinstance(spec.alternate, bool):
        return {"ok": False, "reason": "alternate must be a bool"}
    if spec.rollback.conditional and not spec.require_if_match:
        return {"ok": False, "reason": "conditional rollback requires require_if_match"}

    try:
        probe = cfg.collector_factory(cfg.telemetry_url or cfg.engine_url)
        snap = probe.poll_once()
        for m in spec.measurement.decision_metrics:
            q = _metric_quality(snap, m, spec.engine)
            notes.append(f"{m} availability={q.value if q else 'absent'}")
    except Exception as exc:
        notes.append(f"availability probe failed: {exc}")

    try:
        status, _ = control._request("GET", "/runtime/telemetry")
        notes.append(f"telemetry_service_http={status}")
    except Exception as exc:
        notes.append(f"telemetry_service_unreachable: {exc}")

    return {
        "ok": True,
        "intent": intent,
        "intent_key": internal_key,
        "pre_desired": pre_desired,
        "revision": revision,
        "notes": notes,
    }


# ------------------------------------------------------------------ phases
def _idle_check(cfg: RunnerConfig, spec: ExperimentSpec) -> Optional[bool]:
    """Detection heuristic: engine must be idle before warmup. Returns True when
    both gauges read 0, False when non-idle, None when gauges are unavailable
    (cannot verify — recorded, not a failure)."""
    try:
        coll = cfg.collector_factory(cfg.telemetry_url or cfg.engine_url)
        snap = coll.poll_once()
    except Exception:
        return None
    running = getattr(snap, "requests_running", None)
    waiting = getattr(snap, "requests_waiting", None)
    if running is None or waiting is None:
        return None
    return int(running) == 0 and int(waiting) == 0


def _warmup(spec: ExperimentSpec, cfg: RunnerConfig) -> bool:
    if spec.warmup_requests <= 0:
        return True
    pool = _build_prompt_pool(spec.workload.prompt_tokens, spec.warmup_requests, spec.workload.seed)
    driver = ClosedLoopWorkloadDriver(spec.workload, cfg.send_request, pool)
    driver.start_measurement()
    driver.stop_measurement(t1=None, drain_budget_s=spec.workload.timeout_s + 5.0)
    counts = driver.counts()
    return counts["timeout"] == 0 and counts["error"] == 0 and counts["incomplete"] == 0


def _measure_phase(
    spec: ExperimentSpec, cfg: RunnerConfig, applied_bound: Any,
) -> Dict[str, Any]:
    """Measurement interval [t_start, t1] with per-poll snapshots, the
    sole-traffic invariant, admission-bound check, and a bounded phase-end
    drain. Post-measurement drain completions are recorded separately and never
    counted as measurement throughput. The driver is chosen by ``load_model``:
    closed-loop fixed concurrency, or open-loop fixed arrival rate."""
    collector = cfg.collector_factory(cfg.telemetry_url or cfg.engine_url)
    ws = spec.workload
    if ws.load_model == "open_loop":
        pool = _build_prompt_pool(ws.prompt_tokens, ws.requests, ws.seed)
        driver: Any = OpenLoopWorkloadDriver(ws, cfg.send_request, pool)
        nominal_s = ws.requests / max(ws.arrival_rate, 1e-9)
    else:
        pool = _build_prompt_pool(ws.prompt_tokens, ws.max_requests, ws.seed)
        driver = ClosedLoopWorkloadDriver(ws, cfg.send_request, pool)
        nominal_s = ws.measurement_s
    t_start = time.monotonic()
    driver.start_measurement()
    records: List[Tuple[float, Any, int]] = []
    gpu_records: List[Tuple[float, Any]] = []
    env: Dict[str, Any] = {
        "sole_traffic_invariant": True,
        "foreign_traffic_detected": False,
        "admission_bound_violation": False,
    }
    failure: Optional[str] = None
    t1 = t_start + nominal_s
    try:
        while True:
            wall = time.monotonic()
            snap = collector.poll_once()
            records.append((wall, snap, driver.in_flight()))
            if cfg.gpu_collector is not None:
                # M5C: the GPU surface is polled in the same loop (same cadence)
                # as the engine telemetry; it is diagnostic-only evidence.
                gpu_records.append((wall, cfg.gpu_collector.poll_once()))
            running = getattr(snap, "requests_running", None)
            waiting = getattr(snap, "requests_waiting", None)
            inflight = driver.in_flight()
            # M6B-4: the sole-traffic invariant is a DETECTION heuristic (M6A
            # §11). The engine can hold, at most, the requests the driver has
            # sent and not yet completed (``in_flight``) PLUS requests the
            # driver has ABANDONED on a client timeout (``timeouts``) — SGLang
            # keeps processing an abandoned request, so its running/waiting
            # counts still include it. ``running + waiting`` (an engine
            # snapshot) and the driver counters are read a few ms apart, so a
            # small race tolerance absorbs completions between the two reads.
            # A foreign flood exceeds ``in_flight + timeouts + tolerance`` by
            # far.
            timeouts = driver.counts().get("timeout", 0)
            if running is not None and waiting is not None and running + waiting > inflight + timeouts + _INVARIANT_TOLERANCE:
                env["foreign_traffic_detected"] = True
                env["sole_traffic_invariant"] = False
                failure = (
                    f"foreign traffic detected (running+waiting={running + waiting} > "
                    f"in_flight={inflight} + timeouts={timeouts})"
                )
                break
            if running is not None and applied_bound is not None and running > applied_bound:
                env["admission_bound_violation"] = True
                failure = f"admission bound violated (running={running} > bound={applied_bound})"
                break
            if ws.load_model == "open_loop":
                if driver.emission_done():
                    t1 = driver.emission_end_ts() or t1
                    break
                # safety cap so a stuck emitter cannot hang the phase forever
                if time.monotonic() > t_start + nominal_s + 60.0:
                    t1 = time.monotonic()
                    break
            elif time.monotonic() >= t1:
                break
            remaining = t1 - time.monotonic()
            time.sleep(min(spec.measurement.poll_interval_s, max(0.05, remaining)))
    finally:
        driver.stop_measurement(t1, ws.drain_budget_s)
    counts = driver.counts()
    wall_s = t1 - t_start
    if failure is not None:
        return {"ok": False, "reason": failure, "environment": env,
                "counts": counts, "wall_s": wall_s, "metrics": {},
                "gpu": _gpu_summary(gpu_records, t_start, t1)}
    metrics = _summarize(spec, records, t_start + spec.measurement.window_s, t1, counts)
    bench = driver.benchmark()
    metrics["throughput_req_s"] = {
        "value": bench["value"],
        "spread": None,
        "dominant_quality": "fresh" if bench["value"] is not None else "unavailable",
        "stale_fraction": 0.0,
        "polls_usable": bench["completed"],
        "sample_count": counts.get("completed", 0),
        "duration_s": round(bench["duration_s"], 4),
    }
    if ws.load_model == "open_loop":
        # offered arrival rate (fixed by the spec) + completion rate (measured);
        # the gap is the engine's ability to keep up.
        metrics["offered_rate"] = {
            "value": ws.arrival_rate,
            "dominant_quality": "fresh",
            "stale_fraction": 0.0,
            "polls_usable": 1,
            "sample_count": counts.get("started", 0),
            "unit": "requests/s",
        }
    if cfg.gpu_collector is not None:
        # GPU summary uses the SAME summary window as the engine metrics
        # ([summarize_start, t1]) for cross-phase comparability.
        gpu = _gpu_summary(gpu_records, t_start + spec.measurement.window_s, t1)
    else:
        gpu = {
            "configured": False,
            "available": False,
            "unavailable_reason": "gpu_collector not configured",
            "polls": 0,
            "signals": {},
            "engine_process": None,
        }
    return {"ok": True, "environment": env, "counts": counts,
            "wall_s": wall_s, "metrics": metrics, "gpu": gpu}


def _run_phase(
    spec: ExperimentSpec, cfg: RunnerConfig, control: ControlPlaneClient,
    name: str, desired: Dict[str, Any], intent_key: str, rev_before: Optional[int],
) -> PhaseResult:
    phase = PhaseResult(name)
    # ``desired`` is keyed by the OPERATOR intent (e.g. "runtime_concurrency");
    # ``intent_key`` is the resolved internal DesiredState key (e.g.
    # "runtime_max_running_seqs"). The observed state uses internal keys.
    intent = next(iter(desired))
    desired_value = desired[intent]
    status_code, payload = control.patch_config(desired, if_match=rev_before)
    phase.rev_after_patch = payload.get("desired_revision") if isinstance(payload, dict) else None
    if status_code != 200:
        phase.ok = False
        phase.failure_reason = (
            f"policy PATCH failed (HTTP {status_code}): {payload.get('status')}"
        )
        phase.convergence = {
            "status": payload.get("status"), "steps": payload.get("steps", []),
            "desired": desired,
        }
        return phase
    if payload.get("status") != "CONVERGED":
        phase.ok = False
        phase.failure_reason = f"policy not CONVERGED: {payload.get('status')}"
        phase.convergence = {"status": payload.get("status"), "desired": desired}
        return phase
    observed = payload.get("observed") or {}
    if observed.get(intent_key) != desired_value:
        phase.ok = False
        phase.failure_reason = (
            f"observed {intent_key}={observed.get(intent_key)} != desired {desired_value}"
        )
        phase.convergence = {"status": payload.get("status"), "desired": desired, "observed": observed}
        return phase
    phase.convergence = {"status": "CONVERGED", "desired": dict(desired), "observed": dict(observed)}

    phase.environment["idle_before"] = _idle_check(cfg, spec)
    if not _warmup(spec, cfg):
        phase.ok = False
        phase.failure_reason = "warmup failed (engine not admitting / timeouts / errors)"
        return phase

    measured = _measure_phase(spec, cfg, desired_value)
    phase.environment.update(measured["environment"])
    phase.counts = measured["counts"]
    phase.wall_s = measured["wall_s"]
    phase.metrics = measured["metrics"]
    phase.gpu = measured.get("gpu", {})
    if not measured["ok"]:
        phase.ok = False
        phase.failure_reason = measured["reason"]
        return phase

    cfg_now = control.get_config()
    if cfg_now.get("desired_revision") != phase.rev_after_patch:
        phase.ok = False
        phase.failure_reason = "concurrent change detected (desired_revision changed during phase)"
        phase.environment["concurrent_change_detected"] = True
        return phase
    obs_now = cfg_now.get("observed") or {}
    if obs_now.get(intent_key) != desired_value:
        phase.ok = False
        phase.failure_reason = "observed diverged from desired at end of phase"
        phase.environment["concurrent_change_detected"] = True
        return phase
    phase.environment["concurrent_change_detected"] = False
    return phase


def _rollback(
    spec: ExperimentSpec, cfg: RunnerConfig, control: ControlPlaneClient,
    pre_desired: Dict[str, Any], rev_if_match: Optional[int],
) -> Dict[str, Any]:
    """Conditional rollback (M6A §9): never overwrites a concurrent change
    (409 -> CONFLICT); verifies observed matches the pre-experiment target for
    every target key."""
    target = dict(pre_desired or {})
    try:
        status_code, payload = control.patch_config(target, if_match=rev_if_match)
    except Exception as exc:
        return {"status": "FAILED", "target": target, "error": str(exc)}
    outcome: Dict[str, Any] = {
        "target": target,
        "revision": payload.get("desired_revision") if isinstance(payload, dict) else None,
    }
    if status_code == 409:
        outcome["status"] = "CONFLICT"
        outcome["current_desired"] = (payload or {}).get("desired")
        outcome["observed"] = (payload or {}).get("observed")
        return outcome
    if status_code != 200 or (payload or {}).get("status") != "CONVERGED":
        outcome["status"] = "NOT_CONVERGED"
        outcome["detail"] = (payload or {}).get("status")
        return outcome
    try:
        cfg_now = control.get_config()
        observed = cfg_now.get("observed") or {}
    except Exception as exc:
        outcome["status"] = "NOT_CONVERGED"
        outcome["detail"] = f"verify failed: {exc}"
        return outcome
    bad = [k for k, v in target.items() if observed.get(k) != v]
    if bad:
        outcome["status"] = "NOT_CONVERGED"
        outcome["detail"] = f"observed mismatch for {bad}"
        outcome["observed"] = observed
        return outcome
    outcome["status"] = "CONVERGED"
    outcome["observed"] = observed
    return outcome


# ------------------------------------------------------------------- runner
def run_experiment(spec: ExperimentSpec, cfg: RunnerConfig) -> ExperimentResult:
    """Execute one A/B experiment (single or repeated cycles, optional alternate
    trailing-baseline drift checks) and return a JSON-serializable result. Never
    raises for expected failures; unexpected exceptions are captured, a
    best-effort conditional rollback is attempted, and a FAILED result is
    returned.

    M6B-3 (M6A §8.4/§10.3): with ``repeats > 1`` every cycle uses the same
    seeded workload and explicit phase boundaries; per-cycle measurements,
    adequacy, deltas and verdicts are recorded in ``cycles``; the aggregate
    applies the strict rule (any failed cycle -> FAILED, any inadequate cycle ->
    INCONCLUSIVE, otherwise MEDIAN of per-cycle deltas). Drift checks
    (``alternate``) are reported separately in ``aggregate.drift`` and never
    pooled into the baseline/candidate comparison."""
    if cfg.telemetry_url is None:
        cfg.telemetry_url = cfg.engine_url
    if cfg.collector_factory is None:
        cfg.collector_factory = _default_collector_factory(spec)
    if cfg.send_request is None:
        cfg.send_request = _default_send_request(cfg.engine_url, spec.workload)

    result = ExperimentResult(
        experiment_id=spec.experiment_id,
        engine=spec.engine,
        model=spec.model,
        spec=_spec_dict(spec),
        started_at=_iso_now(),
    )
    control = ControlPlaneClient(cfg.control_url, timeout=cfg.http_timeout)
    rev: Dict[str, Any] = {"preflight": None, "after_baseline": None,
                           "after_candidate": None, "after_rollback": None}
    pre: Dict[str, Any] = {}
    rev_before: Optional[int] = None

    def fail(reason: str, basis_key: str) -> ExperimentResult:
        result.verdict = "FAILED"
        result.failure_reason = reason
        result.verdict_basis.setdefault(basis_key, []).append(reason)
        result.finished_at = _iso_now()
        return result

    try:
        pre = _preflight(spec, cfg, control)
        rev["preflight"] = pre.get("revision")
        result.desired_revision.update(rev)
        result.verdict_basis["environment"] = pre.get("notes", [])
        if not pre.get("ok"):
            return fail(f"preflight failed: {pre.get('reason')}", "convergence")
        rev_before = pre["revision"]

        cycles: List[Dict[str, Any]] = []
        last_rev: Optional[int] = rev_before
        for index in range(int(spec.repeats)):
            cycle, rev_after = _run_cycle(spec, cfg, control, pre, last_rev, index)
            cycles.append(cycle)
            last_rev = rev_after if rev_after is not None else last_rev
            if cycle["rev_after_phase"] is not None:
                rev["after_baseline"] = cycle["rev_after_phase"].get("baseline")
                rev["after_candidate"] = cycle["rev_after_phase"].get("candidate")
            if cycle["failed"]:
                # a failed cycle means the environment/control gate broke;
                # abort remaining cycles and roll back (M6A §8.4 strict).
                break

        agg = _aggregate(spec, cycles)

        rollback = _rollback(spec, cfg, control, pre["pre_desired"], last_rev)
        result.rollback = rollback
        rev["after_rollback"] = rollback.get("revision")
        result.desired_revision.update(rev)

        final_verdict = agg["verdict"]
        final_basis = list(agg["basis"])
        if rollback.get("status") in ("NOT_CONVERGED", "CONFLICT"):
            final_verdict = "FAILED"
            final_basis.append(f"rollback {rollback.get('status')}")
            result.failure_reason = f"rollback {rollback.get('status')}"

        result.cycles = cycles
        result.aggregate = agg
        result.phases = dict(cycles[-1]["phases"]) if cycles else {}
        if len(cycles) == 1 and agg["verdict"] in (
            "IMPROVED", "DEGRADED", "NO_MATERIAL_IMPROVEMENT"
        ):
            # single adequate cycle: comparison keeps the M6B-2 shape
            result.comparison = {"deltas": cycles[0]["deltas"], "adequacy": cycles[0]["adequacy"]}
        else:
            result.comparison = {
                "deltas": agg.get("deltas") or {},
                "adequacy": cycles[-1].get("adequacy", {}) if cycles else {},
                "inadequate_cycles": agg.get("inadequate_cycles", []),
                "failed_cycles": agg.get("failed_cycles", []),
                "per_cycle_verdicts": agg.get("per_cycle_verdicts", []),
            }
        result.verdict = final_verdict
        result.verdict_basis["cycles"] = agg.get("per_cycle_verdicts", [])
        result.verdict_basis["evidence"] = final_basis
        if agg["verdict"] == "FAILED":
            result.failure_reason = "aggregate FAILED: " + "; ".join(agg["basis"])
        result.finished_at = _iso_now()
        return result
    except Exception as exc:  # cleanup on unexpected exceptions: rollback best-effort
        result.failure_reason = f"INTERNAL_ERROR: {exc}"
        result.verdict = "FAILED"
        result.verdict_basis["convergence"] = [result.failure_reason]
        # roll back from the LAST KNOWN mutation revision (never the preflight
        # revision once a mutation has applied), so the If-Match is fresh.
        last_known = (
            rev.get("after_candidate") or rev.get("after_baseline") or rev_before
        )
        try:
            result.rollback = _rollback(spec, cfg, control, pre.get("pre_desired") or {},
                                        last_known)
            rev["after_rollback"] = result.rollback.get("revision")
        except Exception as rexc:
            result.rollback = {"status": "FAILED", "error": str(rexc)}
        result.desired_revision.update(rev)
        result.finished_at = _iso_now()
        return result


# ------------------------------------------------------------------------ CLI
def main() -> None:
    import argparse

    ap = argparse.ArgumentParser("infengine-experiment")
    ap.add_argument("--experiment-id", required=True)
    ap.add_argument("--engine", choices=["vllm", "sglang"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--intent", required=True)
    ap.add_argument("--baseline", type=int, required=True)
    ap.add_argument("--candidate", type=int, required=True)
    ap.add_argument("--control-url", required=True)
    ap.add_argument("--engine-url", required=True)
    ap.add_argument("--telemetry-url", default=None)
    ap.add_argument("--clients", type=int, default=8)
    ap.add_argument("--measurement-s", type=float, default=90.0)
    ap.add_argument("--max-requests", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--poll-interval-s", type=float, default=5.0)
    ap.add_argument("--min-polls", type=int, default=6)
    ap.add_argument("--min-requests", type=int, default=128)
    ap.add_argument("--timeout-s", type=float, default=60.0)
    ap.add_argument("--drain-budget-s", type=float, default=10.0)
    ap.add_argument("--primary", default="e2e_p50_s")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--alternate", action="store_true")
    ap.add_argument("--load-model", choices=["closed_loop", "open_loop"], default="closed_loop")
    ap.add_argument("--arrival-rate", type=float, default=10.0)
    ap.add_argument("--requests", type=int, default=600)
    ap.add_argument("--gpu-telemetry", action="store_true",
                    help="collect diagnostic nvidia-smi GPU telemetry (M5C)")
    ap.add_argument("--gpu-engine-pid", type=int, default=None,
                    help="engine PID for shared-GPU attribution (enables per-process collection)")
    args = ap.parse_args()

    spec = ExperimentSpec(
        experiment_id=args.experiment_id,
        engine=args.engine,
        model=args.model,
        baseline={args.intent: args.baseline},
        candidate={args.intent: args.candidate},
        workload=WorkloadSpec(
            load_model=args.load_model,
            clients=args.clients, measurement_s=args.measurement_s,
            max_requests=args.max_requests,
            arrival_rate=args.arrival_rate, requests=args.requests,
            prompt_tokens=args.prompt_tokens, max_new_tokens=args.max_new_tokens,
            seed=args.seed, timeout_s=args.timeout_s, drain_budget_s=args.drain_budget_s,
        ),
        warmup_requests=args.warmup,
        measurement=MeasurementSpec(
            window_s=args.window_s, poll_interval_s=args.poll_interval_s,
            min_polls=args.min_polls, min_requests=args.min_requests,
        ),
        criteria=SuccessCriteria(primary=args.primary),
        repeats=args.repeats,
        alternate=args.alternate,
    )
    cfg = RunnerConfig(
        control_url=args.control_url,
        engine_url=args.engine_url,
        telemetry_url=args.telemetry_url,
        gpu_collector=(
            GpuTelemetryCollector(
                engine_pid=args.gpu_engine_pid,
                collect_processes=args.gpu_engine_pid is not None,
            )
            if args.gpu_telemetry
            else None
        ),
    )
    result = run_experiment(spec, cfg)
    print(json.dumps(result.to_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()