"""M5C: non-engine GPU telemetry surface (client-side, nvidia-smi subprocess).

A separate, diagnostic-only telemetry surface OUTSIDE the engine. The engine's
``RuntimeSnapshot`` is normalized engine telemetry; this module is the GPU
hardware surface (DCGM-style signals) that no engine metric can provide. It is
deliberately NOT normalized into ``RuntimeSnapshot`` and never feeds the
experiment verdict by default.

Source: ``nvidia-smi --query-gpu=...`` subprocess (zero dependency, already
proven in the M6B-3.1 benchmark-stability investigation, which measured util /
SM clock / power / temperature while exonerating the GPU on a CPU-bound
runtime). The same cadence as the engine telemetry is used: the experiment
runner polls this collector in the same ``_measure_phase`` loop.

Signals per poll (:class:`GpuSnapshot`):

  * ``gpu_util``            — GPU utilization (percent)
  * ``sm_clock_mhz``        — SM clock (MHz)
  * ``memory_used_mb`` / ``memory_free_mb`` / ``memory_total_mb`` — GPU memory
  * ``power_w`` / ``power_limit_w`` — power draw and its cap
  * ``temperature_c``
  * ``processes`` (optional) — per-process compute apps (pid / name /
    used_memory_mb, plus per-process SM% when ``process_util``), used for
    shared-GPU attribution of the engine's PID.

Graceful absence: when there is no GPU or no ``nvidia-smi`` on PATH, a poll
records ``available=False`` with an ``unavailable_reason`` (never raises). Tests
inject a fake GPU source via ``runner``.

Production path (documented, NOT implemented here — M5C boundary): DCGM ->
Prometheus -> control plane. ``nvidia-ml-py`` (pynvml) is a richer zero-free
alternative for per-process utilization when subprocess ``pmon`` parsing proves
insufficient.

Boundaries (M5C): GPU metrics are DIAGNOSTIC ONLY — never verdict-gating by
default, never merged into ``RuntimeSnapshot``, no Prometheus/DCGM deployment,
no dashboard, no fleet, no recommendation layer.
"""

import statistics
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple

# nvidia-smi CSV field order for the --query-gpu below. Keep the two in sync.
_GPU_QUERY = (
    "utilization.gpu,clocks.sm,memory.used,memory.free,memory.total,"
    "power.draw,power.limit,temperature.gpu"
)
_GPU_FIELD_NAMES = (
    "gpu_util",
    "sm_clock_mhz",
    "memory_used_mb",
    "memory_free_mb",
    "memory_total_mb",
    "power_w",
    "power_limit_w",
    "temperature_c",
)

_COMPUTE_APPS_QUERY = "pid,process_name,used_memory"


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_number(raw: str) -> Optional[float]:
    """Parse one nvidia-smi CSV field; ``[N/A]`` / ``N/A`` / empty -> None."""
    text = raw.strip().strip("[]").strip()
    if not text or text.upper() in ("N/A", "NAN", "<UNKNOWN>"):
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if not (value == value) or value in (float("inf"), float("-inf")):  # not NaN
        return None
    return value


def _to_int(raw: str) -> Optional[int]:
    value = _to_number(raw)
    return int(value) if value is not None else None


def _split_csv(line: str) -> List[str]:
    """Split one CSV line, honoring double-quoted fields. nvidia-smi quotes
    fields that contain commas (e.g. a process name / path), so a plain
    ``split(",")`` would corrupt them."""
    fields: List[str] = []
    cur: List[str] = []
    in_quotes = False
    for ch in line:
        if ch == '"':
            in_quotes = not in_quotes
        elif ch == "," and not in_quotes:
            fields.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    fields.append("".join(cur).strip())
    return fields


def _mean(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return statistics.mean(vals) if vals else None


def _sum_non_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def _max_non_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return max(vals) if vals else None


def _default_run(argv: List[str], timeout: float) -> Tuple[int, str]:
    """Run ``nvidia-smi``; map absence/timeout to non-zero exit codes so the
    collector can record ``available=False`` instead of raising."""
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode, proc.stdout
    except FileNotFoundError:
        return 127, ""  # nvidia-smi not on PATH
    except subprocess.TimeoutExpired:
        return 124, ""  # command timed out


@dataclass
class GpuSnapshot:
    """One GPU poll. ``available=False`` means no GPU / nvidia-smi (recorded,
    never an error). All signals are Optional (a field may be ``N/A`` on some
    hardware/driver); ``per_gpu`` preserves per-device detail for multi-GPU."""

    available: bool
    collected_at: str
    unavailable_reason: Optional[str] = None
    gpu_count: int = 0
    gpu_util: Optional[float] = None
    sm_clock_mhz: Optional[float] = None
    memory_used_mb: Optional[float] = None
    memory_free_mb: Optional[float] = None
    memory_total_mb: Optional[float] = None
    power_w: Optional[float] = None
    power_limit_w: Optional[float] = None
    temperature_c: Optional[float] = None
    per_gpu: List[Dict[str, Any]] = field(default_factory=list)
    # Optional shared-GPU attribution (engine PID) when collect_processes.
    processes: Optional[List[Dict[str, Any]]] = None
    engine_process: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self.available,
            "collected_at": self.collected_at,
            "unavailable_reason": self.unavailable_reason,
            "gpu_count": self.gpu_count,
            "gpu_util": self.gpu_util,
            "sm_clock_mhz": self.sm_clock_mhz,
            "memory_used_mb": self.memory_used_mb,
            "memory_free_mb": self.memory_free_mb,
            "memory_total_mb": self.memory_total_mb,
            "power_w": self.power_w,
            "power_limit_w": self.power_limit_w,
            "temperature_c": self.temperature_c,
            "per_gpu": list(self.per_gpu),
            "processes": list(self.processes) if self.processes is not None else None,
            "engine_process": self.engine_process,
        }


class GpuTelemetryCollector:
    """Client-side GPU telemetry collector (nvidia-smi subprocess).

    ``poll_once()`` runs the query and returns a :class:`GpuSnapshot`; the
    experiment runner calls it in the same ``_measure_phase`` loop as the engine
    collector (same poll cadence). ``latest()`` returns the last snapshot.

    ``runner`` is injectable for tests: ``runner(argv) -> (returncode, stdout)``.
    The default runner executes ``argv`` as a subprocess with ``command_timeout``.
    """

    def __init__(
        self,
        *,
        nvidia_smi: str = "nvidia-smi",
        command_timeout: float = 5.0,
        engine_pid: Optional[int] = None,
        collect_processes: bool = False,
        process_util: bool = False,
        runner: Optional[Callable[[List[str]], Tuple[int, str]]] = None,
    ):
        self.nvidia_smi = nvidia_smi
        self.command_timeout = command_timeout
        self.engine_pid = engine_pid
        self.collect_processes = collect_processes
        self.process_util = process_util
        self._run = runner or (lambda argv: _default_run(argv, command_timeout))
        self._last: Optional[GpuSnapshot] = None

    # ------------------------------------------------------------- commands
    def gpu_query_argv(self) -> List[str]:
        return [
            self.nvidia_smi,
            f"--query-gpu={_GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ]

    def compute_apps_argv(self) -> List[str]:
        return [
            self.nvidia_smi,
            f"--query-compute-apps={_COMPUTE_APPS_QUERY}",
            "--format=csv,noheader,nounits",
        ]

    def pmon_argv(self) -> List[str]:
        return [self.nvidia_smi, "pmon", "-c", "1"]

    # --------------------------------------------------------------- polling
    def poll_once(self) -> GpuSnapshot:
        snap = self._collect()
        self._last = snap
        return snap

    def latest(self) -> Optional[GpuSnapshot]:
        return self._last

    # -------------------------------------------------------------- collecting
    def _collect(self) -> GpuSnapshot:
        collected_at = _iso_now()
        rc, out = self._run(self.gpu_query_argv())
        if rc != 0:
            return GpuSnapshot(
                available=False, collected_at=collected_at,
                unavailable_reason=f"nvidia-smi exited {rc}",
            )
        rows = [ln for ln in out.splitlines() if ln.strip()]
        if not rows:
            return GpuSnapshot(
                available=False, collected_at=collected_at,
                unavailable_reason="nvidia-smi returned no GPU rows",
            )
        gpus: List[Dict[str, Any]] = []
        for row in rows:
            fields = [f.strip() for f in row.split(",")]
            if len(fields) < len(_GPU_FIELD_NAMES):
                continue
            gpus.append(
                {
                    name: _to_number(raw)
                    for name, raw in zip(_GPU_FIELD_NAMES, fields[: len(_GPU_FIELD_NAMES)])
                }
            )
        if not gpus:
            return GpuSnapshot(
                available=False, collected_at=collected_at,
                unavailable_reason="unparseable nvidia-smi output",
            )
        for idx, gpu in enumerate(gpus):
            gpu["index"] = idx
        processes: Optional[List[Dict[str, Any]]] = None
        engine_process: Optional[Dict[str, Any]] = None
        if self.collect_processes:
            processes, engine_process = self._collect_processes()
        return GpuSnapshot(
            available=True,
            collected_at=collected_at,
            gpu_count=len(gpus),
            gpu_util=_mean([g["gpu_util"] for g in gpus]),
            sm_clock_mhz=_mean([g["sm_clock_mhz"] for g in gpus]),
            memory_used_mb=_sum_non_none([g["memory_used_mb"] for g in gpus]),
            memory_free_mb=_sum_non_none([g["memory_free_mb"] for g in gpus]),
            memory_total_mb=_sum_non_none([g["memory_total_mb"] for g in gpus]),
            power_w=_mean([g["power_w"] for g in gpus]),
            power_limit_w=_mean([g["power_limit_w"] for g in gpus]),
            # max: the thermal-relevant device drives throttling
            temperature_c=_max_non_none([g["temperature_c"] for g in gpus]),
            per_gpu=gpus,
            processes=processes,
            engine_process=engine_process,
        )

    def _collect_processes(self) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Per-process compute-app attribution (shared-GPU): used memory from
        ``--query-compute-apps``; per-process SM% from ``pmon -c 1`` when
        ``process_util`` is set (nvidia-smi's CSV query exposes no per-process
        utilization)."""
        processes: List[Dict[str, Any]] = []
        rc, out = self._run(self.compute_apps_argv())
        if rc == 0:
            for ln in out.splitlines():
                ln = ln.strip()
                if not ln:
                    continue
                parts = _split_csv(ln)
                if len(parts) < 3:
                    continue
                pid = _to_int(parts[0])
                if pid is None:
                    continue
                processes.append(
                    {
                        "pid": pid,
                        "name": parts[1],
                        "used_memory_mb": _to_number(parts[-1]),
                    }
                )
        if self.process_util:
            sm_by_pid = self._collect_pmon_sm()
            for proc in processes:
                sm = sm_by_pid.get(proc["pid"])
                if sm is not None:
                    proc["sm_util_pct"] = sm
        engine_process = None
        if self.engine_pid is not None:
            for proc in processes:
                if proc["pid"] == self.engine_pid:
                    engine_process = proc
                    break
        return processes, engine_process

    def _collect_pmon_sm(self) -> Dict[int, float]:
        """Parse ``nvidia-smi pmon -c 1`` (per-process SM/MEM utilization)."""
        sm_by_pid: Dict[int, float] = {}
        rc, out = self._run(self.pmon_argv())
        if rc != 0:
            return sm_by_pid
        for ln in out.splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 4:
                continue
            try:
                pid = int(parts[1])
                sm = float(parts[3])
            except ValueError:
                continue
            sm_by_pid[pid] = sm  # last row wins (per-GPU); same PID on one device
        return sm_by_pid