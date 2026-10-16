"""M5C: non-engine GPU telemetry (nvidia-smi) tests.

The collector's nvidia-smi subprocess is replaced by a FAKE GPU source
(``telemetry_helpers.FakeGpuSource``) so no GPU / driver is required. Covers:
CSV parsing, N/A handling, graceful absence (no nvidia-smi / no GPU), multi-GPU
aggregation, per-process (engine PID) attribution, pmon SM% parsing, and
serialization.
"""

import json

from infengine.telemetry import GpuTelemetryCollector

from telemetry_helpers import FakeGpuSource, compute_apps_csv, gpu_csv, pmon_text


def _collector(**kw):
    kw.setdefault("runner", FakeGpuSource())
    return GpuTelemetryCollector(**kw)


# ------------------------------------------------------------------- parsing
def test_single_gpu_full_parse():
    source = FakeGpuSource(gpu_out=gpu_csv((42, 1800, 2048, 2048, 4096, 25.5, 45.0, 70)))
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.available is True
    assert snap.unavailable_reason is None
    assert snap.gpu_count == 1
    assert snap.gpu_util == 42.0
    assert snap.sm_clock_mhz == 1800.0
    assert snap.memory_used_mb == 2048.0
    assert snap.memory_free_mb == 2048.0
    assert snap.memory_total_mb == 4096.0
    assert snap.power_w == 25.5
    assert snap.power_limit_w == 45.0
    assert snap.temperature_c == 70.0
    assert len(snap.per_gpu) == 1
    assert snap.per_gpu[0]["index"] == 0


def test_na_fields_become_none():
    # power.limit and clocks.sm are commonly N/A / [N/A] on some drivers
    source = FakeGpuSource(gpu_out=gpu_csv((0, "[N/A]", 3705, 231, 4096, 2.94, "N/A", 65)))
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.available is True
    assert snap.sm_clock_mhz is None
    assert snap.power_limit_w is None
    assert snap.gpu_util == 0.0
    assert snap.temperature_c == 65.0
    assert snap.power_w == 2.94


def test_missing_nvidia_smi_recorded_unavailable():
    # FileNotFoundError is mapped to exit 127 by the default runner; a fake
    # source simulates it. Nothing raises, the poll records the absence.
    source = FakeGpuSource(gpu_rc=127)
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.available is False
    assert snap.unavailable_reason == "nvidia-smi exited 127"
    assert snap.gpu_count == 0
    assert snap.to_dict()["available"] is False


def test_nonzero_exit_recorded_unavailable():
    source = FakeGpuSource(gpu_rc=1)
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.available is False
    assert "exited 1" in snap.unavailable_reason


def test_empty_output_recorded_unavailable():
    snap = GpuTelemetryCollector(runner=FakeGpuSource(gpu_out="")).poll_once()
    assert snap.available is False
    assert "no GPU rows" in snap.unavailable_reason


def test_garbage_output_recorded_unavailable():
    source = FakeGpuSource(gpu_out="not,a,valid,row\n")
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.available is False
    assert "unparseable" in snap.unavailable_reason


# -------------------------------------------------------------- multi-GPU
def test_multi_gpu_aggregation():
    source = FakeGpuSource(gpu_out=gpu_csv(
        (50, 1800, 1000, 3000, 4000, 20.0, 45.0, 60),
        (30, 1700, 1500, 2500, 4000, 15.0, 45.0, 75),
    ))
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.gpu_count == 2
    # util / clock / power / limit = mean; temperature = max; memory = sum
    assert snap.gpu_util == 40.0
    assert snap.sm_clock_mhz == 1750.0
    assert snap.power_w == 17.5
    assert snap.memory_used_mb == 2500.0
    assert snap.memory_free_mb == 5500.0
    assert snap.memory_total_mb == 8000.0
    assert snap.temperature_c == 75.0
    assert len(snap.per_gpu) == 2
    assert snap.per_gpu[0]["index"] == 0 and snap.per_gpu[1]["index"] == 1


# ----------------------------------------------------- per-process attribution
def test_process_attribution_engine_pid_memory():
    source = FakeGpuSource(
        gpu_out=gpu_csv((60, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)),
        apps_out=compute_apps_csv((1234, "python.exe", 512), (5678, "other", 128)),
    )
    coll = GpuTelemetryCollector(runner=source, collect_processes=True, engine_pid=1234)
    snap = coll.poll_once()
    assert snap.processes is not None and len(snap.processes) == 2
    assert snap.engine_process is not None
    assert snap.engine_process["pid"] == 1234
    assert snap.engine_process["name"] == "python.exe"
    assert snap.engine_process["used_memory_mb"] == 512.0


def test_process_attribution_engine_pid_absent():
    source = FakeGpuSource(
        gpu_out=gpu_csv((60, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)),
        apps_out=compute_apps_csv((5678, "other", 128)),
    )
    coll = GpuTelemetryCollector(runner=source, collect_processes=True, engine_pid=1234)
    snap = coll.poll_once()
    assert snap.processes and len(snap.processes) == 1
    assert snap.engine_process is None  # engine PID not on the GPU


def test_process_name_with_commas_parsed():
    # process_name may contain commas (a path); only pid (first) and
    # used_memory (last) are positional.
    source = FakeGpuSource(
        gpu_out=gpu_csv((60, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)),
        apps_out=compute_apps_csv(("999", "C:\\Tools, Inc\\python.exe", 256)),
    )
    coll = GpuTelemetryCollector(runner=source, collect_processes=True, engine_pid=999)
    snap = coll.poll_once()
    assert snap.engine_process["pid"] == 999
    assert snap.engine_process["used_memory_mb"] == 256.0
    assert "Tools, Inc" in snap.engine_process["name"]


def test_pmon_per_process_sm_util():
    source = FakeGpuSource(
        gpu_out=gpu_csv((60, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)),
        apps_out=compute_apps_csv((1234, "python.exe", 512), (5678, "other", 128)),
        pmon_out=pmon_text([
            (0, 1234, "C", 43, 61, 0, 0, 1024, 0, 8192),
            (0, 5678, "C", 12, 30, 0, 0, 512, 0, 4096),
        ]),
    )
    coll = GpuTelemetryCollector(runner=source, collect_processes=True,
                                 process_util=True, engine_pid=1234)
    snap = coll.poll_once()
    assert snap.engine_process["sm_util_pct"] == 43.0
    other = next(p for p in snap.processes if p["pid"] == 5678)
    assert other["sm_util_pct"] == 12.0


def test_pmon_not_run_when_process_util_disabled():
    source = FakeGpuSource(
        gpu_out=gpu_csv((60, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)),
        apps_out=compute_apps_csv((1234, "python.exe", 512)),
    )
    coll = GpuTelemetryCollector(runner=source, collect_processes=True, engine_pid=1234)
    snap = coll.poll_once()
    assert "sm_util_pct" not in snap.engine_process
    assert snap.engine_process["used_memory_mb"] == 512.0


def test_processes_not_collected_by_default():
    source = FakeGpuSource(gpu_out=gpu_csv((60, 1800, 2048, 2048, 4096, 25.0, 45.0, 70)))
    snap = GpuTelemetryCollector(runner=source).poll_once()
    assert snap.processes is None
    assert snap.engine_process is None


# ------------------------------------------------------------------ misc
def test_poll_once_updates_latest():
    source = FakeGpuSource(gpu_out=gpu_csv((42, 1800, 2048, 2048, 4096, 25.5, 45.0, 70)))
    coll = GpuTelemetryCollector(runner=source)
    assert coll.latest() is None
    snap = coll.poll_once()
    assert coll.latest() is snap
    assert snap.to_dict()["gpu_util"] == 42.0


def test_snapshot_to_dict_json_serializable():
    source = FakeGpuSource(
        gpu_out=gpu_csv((42, 1800, 2048, 2048, 4096, 25.5, 45.0, 70)),
        apps_out=compute_apps_csv((1234, "python.exe", 512)),
    )
    coll = GpuTelemetryCollector(runner=source, collect_processes=True, engine_pid=1234)
    data = coll.poll_once().to_dict()
    json.dumps(data)  # must not raise
    assert data["available"] is True
    assert data["processes"][0]["pid"] == 1234


def test_finite_values_only():
    source = FakeGpuSource(gpu_out=gpu_csv((50, "nan", 1000, 3000, 4000, "inf", 45.0, 60)))
    snap = GpuTelemetryCollector(runner=source).poll_once()
    # non-finite values are rejected by the parser (None), never leaked as inf/nan
    assert snap.sm_clock_mhz is None
    assert snap.power_w is None
    assert snap.gpu_util == 50.0


def test_argv_shapes():
    coll = _collector(nvidia_smi="/usr/bin/nvidia-smi")
    g = coll.gpu_query_argv()
    assert g[0] == "/usr/bin/nvidia-smi"
    assert "--query-gpu=utilization.gpu,clocks.sm,memory.used,memory.free,memory.total,power.draw,power.limit,temperature.gpu" in g
    assert "--format=csv,noheader,nounits" in g
    assert "--query-compute-apps=pid,process_name,used_memory" in coll.compute_apps_argv()
    assert coll.pmon_argv() == ["/usr/bin/nvidia-smi", "pmon", "-c", "1"]