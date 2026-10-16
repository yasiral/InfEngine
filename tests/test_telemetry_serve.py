"""M5B service tests: expose the latest snapshot through the existing service.

Verifies the new ``GET /runtime/telemetry`` endpoint and that the existing
control endpoints are unchanged when telemetry is configured. Telemetry is
independent of the control adapters/Reconciler; inference is never proxied.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from starlette.testclient import TestClient  # noqa: E402

from infengine.serve import create_app  # noqa: E402
from infengine.telemetry import SglangTelemetryCollector, VllmTelemetryCollector  # noqa: E402

from test_sglang_adapter import FakeSglangRuntimeClient  # noqa: E402
from telemetry_helpers import sglang_loads_exposition, telemetry_server, vllm_exposition  # noqa: E402


def _sequence(events):
    state = {"i": 0}

    def route():
        event = events[state["i"] % len(events)]
        state["i"] += 1
        return event

    return route


def test_telemetry_endpoint_404_without_telemetry():
    app = create_app(runtime="sglang", runtime_client=FakeSglangRuntimeClient())
    with TestClient(app) as c:
        assert c.get("/runtime/telemetry").status_code == 404


def test_telemetry_endpoint_returns_latest_snapshot():
    loads = sglang_loads_exposition(running=1, waiting=2)
    with telemetry_server({"/v1/loads?format=prometheus": (200, loads)}) as base:
        coll = SglangTelemetryCollector(base, timeout=5.0)
        coll.poll_once()
        app = create_app(runtime="sglang", runtime_client=FakeSglangRuntimeClient(), telemetry=coll)
        with TestClient(app) as c:
            r = c.get("/runtime/telemetry")
    assert r.status_code == 200
    data = r.json()
    assert data["engine"] == "sglang"
    assert data["normalized"]["requests_running"] == 1
    assert data["normalized"]["requests_waiting"] == 2
    assert data["quality"]["ttft_p95_s"] == "disabled"


def test_existing_control_endpoints_unchanged_with_telemetry():
    loads = sglang_loads_exposition()
    with telemetry_server({"/v1/loads?format=prometheus": (200, loads)}) as base:
        coll = SglangTelemetryCollector(base, timeout=5.0)
        coll.poll_once()
        app = create_app(runtime="sglang", runtime_client=FakeSglangRuntimeClient(), telemetry=coll)
        with TestClient(app) as c:
            assert c.get("/health").status_code == 200
            assert c.get("/runtime/config").status_code == 200
            assert c.get("/runtime/capabilities").status_code == 200


def test_telemetry_url_constructs_vllm_collector():
    body = vllm_exposition(running=5)
    with telemetry_server({"/metrics": (200, body)}) as base:
        app = create_app(runtime="builtin", get_engine=lambda: None,
                         telemetry_url=base, telemetry_engine="vllm")
        with TestClient(app) as c:
            deadline = time.time() + 5.0
            data = None
            while time.time() < deadline:
                r = c.get("/runtime/telemetry")
                if r.status_code == 200:
                    data = r.json()
                    break
                time.sleep(0.05)
    assert data is not None
    assert data["engine"] == "vllm"
    assert data["normalized"]["requests_running"] == 5


def test_telemetry_url_constructs_sglang_collector_for_sglang_backend():
    loads = sglang_loads_exposition(running=7)
    with telemetry_server({"/v1/loads?format=prometheus": (200, loads)}) as base:
        app = create_app(runtime="sglang", runtime_client=FakeSglangRuntimeClient(),
                         telemetry_url=base)
        with TestClient(app) as c:
            deadline = time.time() + 5.0
            data = None
            while time.time() < deadline:
                r = c.get("/runtime/telemetry")
                if r.status_code == 200:
                    data = r.json()
                    break
                time.sleep(0.05)
    assert data is not None
    assert data["engine"] == "sglang"
    assert data["normalized"]["requests_running"] == 7


def test_runtime_telemetry_never_serves_stale_fresh_after_failure():
    # A successful poll followed by a failed scrape: /runtime/telemetry must
    # surface the failure (scrape_failed, no value), not the old fresh value.
    ok = vllm_exposition(running=3)
    with telemetry_server({"/metrics": _sequence([(200, ok), (500, "")])}) as base:
        coll = VllmTelemetryCollector(base, timeout=5.0)
        coll.poll_once()  # fresh
        coll.poll_once()  # scrape fails -> latest() is SCRAPE_FAILED
        app = create_app(runtime="sglang", runtime_client=FakeSglangRuntimeClient(), telemetry=coll)
        # No lifespan/background thread here (plain TestClient) so latest() is
        # exactly the failure snapshot the manual polls produced.
        data = TestClient(app).get("/runtime/telemetry").json()
    assert data["normalized"]["requests_running"] is None
    assert data["quality"]["requests_running"] == "scrape_failed"