"""Operator-integration tests: explicit vLLM backend, transport-scoped
capabilities, manual operator commands, and the single-invocation review flow.

These are deterministic (no live engine, no GPU): the vLLM adapter runs over the
same in-memory fake transport used by its own unit tests, and the operator CLI
runs over a fake control view. Live acceptance on real engines is separate.
"""

import http.server
import json
import sys
import threading
from pathlib import Path

import pytest
from starlette.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_vllm_adapter import FakeVllmRuntimeClient  # noqa: E402

from infengine import operator  # noqa: E402
from infengine.serve import create_app, validate_runtime  # noqa: E402
from infengine.telemetry import RuntimeSnapshot  # noqa: E402
from infengine.vllm import (  # noqa: E402
    HTTP_HOT_OPERATIONS,
    HttpVllmRuntimeClient,
    VllmRuntimeError,
)


class HttpSubsetVllmRuntimeClient(FakeVllmRuntimeClient):
    """In-memory fake whose transport implements only the HTTP control subset."""

    def supported_operations(self):
        return set(HTTP_HOT_OPERATIONS)


# --------------------------------------------------------------- backend choice
def test_unknown_backend_is_rejected_no_fallback():
    with pytest.raises(ValueError):
        create_app(runtime="nope")


def test_vllm_backend_uses_vllm_contract():
    app = create_app(runtime="vllm", runtime_client=FakeVllmRuntimeClient())
    c = TestClient(app)
    caps = {e["intent"]: e for e in c.get("/runtime/capabilities").json()}
    # In-process fake implements every operation -> DRAIN capability supported.
    assert caps["reset_prefix_cache"]["supported_on_selected_engine"] is True
    assert caps["engine_cache_flush"]["supported_on_selected_engine"] is False
    status = c.get("/runtime/status").json()
    assert status["identity"]["engine_instance_id"] == "vllm-v1-fake-0001"


def test_http_transport_marks_drain_unsupported_and_rejects():
    client = HttpSubsetVllmRuntimeClient()
    app = create_app(runtime="vllm", runtime_client=client)
    c = TestClient(app)
    caps = {e["intent"]: e for e in c.get("/runtime/capabilities").json()}
    assert caps["reset_prefix_cache"]["supported_on_selected_engine"] is False
    assert caps["runtime_concurrency"]["supported_on_selected_engine"] is True

    resp = c.patch("/runtime/config", json={"reset_prefix_cache": True})
    assert resp.status_code == 422
    assert client.reset_calls == 0  # zero mutation


def test_vllm_completions_are_control_plane_only():
    app = create_app(runtime="vllm", runtime_client=FakeVllmRuntimeClient())
    resp = TestClient(app).post("/v1/completions", json={"prompt": "hi"})
    assert resp.status_code == 501


# --------------------------------------------------------------- manual commands
def test_manual_set_and_restore_need_no_m7():
    client = FakeVllmRuntimeClient()
    app = create_app(runtime="vllm", runtime_client=client)
    c = TestClient(app)

    up = c.patch("/runtime/config", json={"runtime_max_running_seqs": 2})
    assert up.status_code == 200 and up.json()["status"] == "CONVERGED"
    assert up.json()["observed"]["runtime_max_running_seqs"] == 2
    assert up.json()["desired_revision"] == 1

    back = c.patch("/runtime/config", json={"runtime_max_running_seqs": 4})
    assert back.status_code == 200 and back.json()["status"] == "CONVERGED"
    assert back.json()["observed"]["runtime_max_running_seqs"] == 4


def test_invalid_value_rejected_zero_mutation():
    client = FakeVllmRuntimeClient(provisioned_seqs=4)
    app = create_app(runtime="vllm", runtime_client=client)
    c = TestClient(app)
    resp = c.patch("/runtime/config", json={"runtime_max_running_seqs": 999})
    assert resp.status_code == 422
    assert c.get("/runtime/config").json()["desired_revision"] == 0


def test_revision_conflict_is_409():
    app = create_app(runtime="vllm", runtime_client=FakeVllmRuntimeClient())
    c = TestClient(app)
    resp = c.patch("/runtime/config", json={"runtime_max_running_seqs": 2},
                   headers={"If-Match": "99"})
    assert resp.status_code == 409


# --------------------------------------------------------- startup validation
def test_validate_runtime_unreachable_is_hard_failure():
    with pytest.raises(SystemExit):
        validate_runtime("vllm", vllm_control_url="http://127.0.0.1:1")


def test_validate_runtime_builtin_is_noop():
    assert validate_runtime("builtin") is None


# ------------------------------------------------------------- HTTP vllm client
class _Handler(http.server.BaseHTTPRequestHandler):
    captured = {}
    status = 200

    def _respond(self):
        type(self).captured["token"] = self.headers.get("X-InfEngine-Token")
        if type(self).status != 200:
            body = json.dumps({"detail": "unauthorized"}).encode()
            self.send_response(type(self).status)
        else:
            body = json.dumps({"policies": {}, "provisioned": {}}).encode()
            self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    do_GET = _respond
    do_POST = _respond

    def log_message(self, *args):
        pass


def _serve(handler):
    httpd = http.server.HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def test_http_client_sends_token_and_parses():
    _Handler.status = 200
    httpd = _serve(_Handler)
    try:
        client = HttpVllmRuntimeClient(
            f"http://127.0.0.1:{httpd.server_address[1]}", token="sekret"
        )
        assert client.get_runtime_scheduler_config() == {"policies": {}, "provisioned": {}}
        assert _Handler.captured["token"] == "sekret"
    finally:
        httpd.shutdown()


def test_http_client_rejects_and_never_leaks_token():
    _Handler.status = 401
    httpd = _serve(_Handler)
    try:
        client = HttpVllmRuntimeClient(
            f"http://127.0.0.1:{httpd.server_address[1]}", token="sekret"
        )
        with pytest.raises(VllmRuntimeError) as exc:
            client.get_runtime_scheduler_config()
        assert "sekret" not in str(exc.value)
        for op in ("pause_generation", "resume_generation",
                   "reset_prefix_cache", "get_runtime_cache_stats"):
            with pytest.raises(VllmRuntimeError):
                getattr(client, op)()
    finally:
        httpd.shutdown()


# --------------------------------------------------------------- operator CLI
class FakeOperatorControl:
    """Minimal ApprovalControlView + telemetry for the operator CLI."""

    def __init__(self, current=4, maximum=8):
        self.current = current
        self.maximum = maximum
        self.revision = 1
        self.patched = []

    def capabilities(self):
        return [{
            "intent": "runtime_concurrency",
            "internal_key": "runtime_max_running_seqs",
            "min": 1, "max": self.maximum, "transition": "HOT",
            "supported_on_selected_engine": True,
        }]

    def get_config(self):
        return {
            "observed": {"runtime_max_running_seqs": self.current},
            "desired": {"runtime_max_running_seqs": self.current},
            "desired_revision": self.revision,
        }

    def patch_config(self, desired, if_match=None):
        self.patched.append((dict(desired), if_match))
        # The service resolves the operator alias; the fake accepts either key.
        self.current = desired.get(
            "runtime_max_running_seqs", desired.get("runtime_concurrency")
        )
        self.revision += 1
        return 200, {
            "status": "CONVERGED",
            "observed": {"runtime_max_running_seqs": self.current},
            "desired_revision": self.revision,
        }

    def _request(self, method, path, body=None, headers=None):
        snap = RuntimeSnapshot(
            engine="vllm", instance_id="t", collected_at="2020-01-01T00:00:00Z",
            source="t", requests_running=self.current, requests_waiting=10,
            kv_usage=0.3, ttft_p95_s=0.1,
        )
        return 200, snap.to_dict()


def _review_args(**over):
    base = dict(samples=3, interval=0.0, window_s=30.0, min_samples=3,
                cooldown_s=60.0, ttl=300.0, approver="tester", yes=True,
                latency_max=0.5, kv_max=0.9, latency_guardrail=True,
                kv_guardrail=True, required_guardrails=True)
    base.update(over)
    return type("A", (), base)


def test_operator_review_single_invocation_applies_proposal():
    control = FakeOperatorControl(current=4, maximum=8)
    rc = operator.cmd_review(control, _review_args())
    assert rc == 0
    assert control.patched == [({"runtime_concurrency": 5}, 1)]
    assert control.current == 5


def test_operator_review_declines_without_confirmation(monkeypatch):
    control = FakeOperatorControl(current=4, maximum=8)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    rc = operator.cmd_review(control, _review_args(yes=False))
    assert rc == 0
    assert control.patched == []
    assert control.current == 4


def test_operator_manual_set_calls_patch_directly():
    control = FakeOperatorControl(current=4, maximum=8)
    args = type("A", (), {"intent": "runtime_concurrency", "value": "3",
                          "if_match": None})
    assert operator.cmd_set(control, args) == 0
    assert control.patched == [({"runtime_concurrency": 3}, None)]
