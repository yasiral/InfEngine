"""M13 product tests: coordinated multi-control apply, rollback, alias
conflicts, auth/actor, durable audit history, policy files, the unified CLI and
the multi-control evaluator.

Deterministic and local: fakes and the built-in engine only; no GPU.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_control import make_engine, _StubTok  # noqa: E402
from test_vllm_adapter import FakeVllmRuntimeClient  # noqa: E402

from infengine.audit import AuditStore  # noqa: E402
from infengine.control import Reconciler, ReconcileStatus, StepStatus  # noqa: E402
from infengine.policy import PolicyError, load_policy, validate_policy  # noqa: E402
from infengine.security import Action, AuthConfig, AuthError, Role  # noqa: E402
from infengine.vllm import VllmAdapter, VllmRuntimeError  # noqa: E402


# --------------------------------------------------------- coordinated apply
class PartialVllmClient(FakeVllmRuntimeClient):
    """Applies fields until ``fail_field`` mutates, then raises mid-batch."""

    def __init__(self, fail_field, **kwargs):
        super().__init__(**kwargs)
        self.fail_field = fail_field

    def set_runtime_scheduler_config(self, overrides):
        for field, value in overrides.items():
            if field == self.fail_field:
                raise VllmRuntimeError("simulated mid-batch failure")
            self.config[field] = value
        return self.get_runtime_scheduler_config()


def test_vllm_multi_control_applied_in_one_request():
    client = FakeVllmRuntimeClient()
    adapter = VllmAdapter(client)
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({
        "runtime_max_running_seqs": 2,
        "runtime_max_scheduled_tokens": 256,
        "runtime_kv_watermark": 0.1,
    })
    assert r.status is ReconcileStatus.CONVERGED
    # ONE coordinated engine request for all three fields.
    assert client.set_calls == [{
        "runtime_max_running_seqs": 2,
        "runtime_max_scheduled_tokens": 256,
        "runtime_kv_watermark": 0.1,
    }]
    assert adapter.get("runtime_kv_watermark") == 0.1
    assert VllmAdapter.APPLY_SEMANTICS == "CONTROL_PLANE_TRANSACTIONAL"
    assert reconciler.apply_semantics() == "CONTROL_PLANE_TRANSACTIONAL"


def test_vllm_partial_batch_failure_is_rolled_back():
    client = PartialVllmClient(fail_field="runtime_max_scheduled_tokens")
    adapter = VllmAdapter(client)
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({
        "runtime_max_running_seqs": 2,
        "runtime_max_scheduled_tokens": 256,
    })
    assert r.status is ReconcileStatus.FAILED
    # the first field mutated before the failure, and was restored to pre-state
    assert adapter.get("runtime_max_running_seqs") == 4
    assert r.rollback is not None
    assert r.rollback["status"] == "ROLLBACK_SUCCEEDED"
    assert "runtime_max_running_seqs" in r.rollback["restored"]


def test_vllm_one_field_divergence_prevents_converged():
    client = FakeVllmRuntimeClient(diverge=False)
    adapter = VllmAdapter(client)
    # force a divergence by monkeypatching get() for one field only
    real_get = adapter.get

    def flaky_get(name):
        if name == "runtime_max_scheduled_tokens":
            return 999
        return real_get(name)

    adapter.get = flaky_get
    reconciler = Reconciler(adapter)
    r = reconciler.reconcile({
        "runtime_max_running_seqs": 2, "runtime_max_scheduled_tokens": 256})
    assert r.status is ReconcileStatus.FAILED
    by = {s.param: s for s in r.steps}
    assert by["runtime_max_running_seqs"].status is StepStatus.VERIFIED
    assert by["runtime_max_scheduled_tokens"].status is StepStatus.FAILED


def test_sglang_multi_control_sequential_semantics():
    from infengine.sglang import SglangAdapter
    assert SglangAdapter.SUPPORTS_BATCH_APPLY is False
    assert SglangAdapter.APPLY_SEMANTICS == "COORDINATED_NON_ATOMIC"


# --------------------------------------------------------- alias conflict
def test_conflicting_aliases_rejected_zero_mutation():
    engine = make_engine(max_batch=2)
    from infengine.serve import create_app
    from starlette.testclient import TestClient

    app = create_app(get_engine=lambda: engine)
    c = TestClient(app)
    r = c.patch("/runtime/config", json={
        "runtime_concurrency": 3, "runtime_max_running_seqs": 5})
    # builtin engine does not support runtime_concurrency; the point is that the
    # request is rejected as a whole with zero mutation.
    assert r.status_code == 422
    assert engine.scheduler.max_batch == 2


def test_contract_conflict_detection_unit():
    from infengine.control import ControlContract

    adapter = VllmAdapter(FakeVllmRuntimeClient())
    contract = ControlContract("vllm")  # registry maps runtime_concurrency alias
    # Same internal key via alias + native key with different values.
    resolved = contract.resolve(adapter, {
        "runtime_concurrency": 2, "runtime_max_running_seqs": 3})
    assert resolved.valid is False
    assert any("conflicting aliases" in (i.reason or "") for i in resolved.intents)


# --------------------------------------------------------- auth + actor + audit
def _secured_app(email=""):
    from infengine.serve import create_app
    engine = make_engine(max_batch=2)
    engine.tokenizer = _StubTok()
    auth = AuthConfig({
        "tok-view": {"actor": "viewer", "role": "VIEWER"},
        "tok-op": {"actor": "operator", "role": "OPERATOR"},
    })
    audit = AuditStore(":memory:")
    app = create_app(get_engine=lambda: engine, auth=auth, audit_store=audit)
    return app, audit


def test_auth_required_and_role_enforced():
    from starlette.testclient import TestClient
    app, _ = _secured_app()
    c = TestClient(app)
    # no token
    assert c.patch("/runtime/config", json={"scheduler.max_batch": 4}).status_code == 401
    # viewer cannot mutate
    r = c.patch("/runtime/config", json={"scheduler.max_batch": 4},
                headers={"Authorization": "Bearer tok-view"})
    assert r.status_code == 403
    # operator can
    r = c.patch("/runtime/config", json={"scheduler.max_batch": 4},
                headers={"Authorization": "Bearer tok-op"})
    assert r.status_code == 200
    assert r.json()["status"] == "CONVERGED"
    # history requires auth; viewer can read
    assert c.get("/runtime/history").status_code == 401
    h = c.get("/runtime/history", headers={"Authorization": "Bearer tok-view"})
    assert h.status_code == 200
    records = h.json()["history"]
    assert records and records[0]["actor"] == "operator"
    assert records[0]["result"] == "CONVERGED"


def test_audit_captures_pre_state_for_restore():
    from starlette.testclient import TestClient
    app, audit = _secured_app()
    c = TestClient(app)
    c.patch("/runtime/config", json={"scheduler.max_batch": 4},
            headers={"Authorization": "Bearer tok-op"})
    rec = audit.latest()
    assert rec is not None
    assert rec.pre_state == {"scheduler.max_batch": 2}
    assert rec.requested == {"scheduler.max_batch": 4}
    assert rec.revision_after == 1


def test_audit_records_validation_failure():
    from starlette.testclient import TestClient
    app, audit = _secured_app()
    c = TestClient(app)
    c.patch("/runtime/config", json={"scheduler.max_batch": "bad"},
            headers={"Authorization": "Bearer tok-op"})
    rec = audit.latest()
    assert rec.result == "VALIDATION_FAILED"


def test_actor_cannot_be_spoofed_when_authenticated():
    """The audit actor is derived from the bearer token, not the client-supplied
    actor header. A spoofed X-InfEngine-Actor must be ignored."""
    from starlette.testclient import TestClient
    app, audit = _secured_app()
    c = TestClient(app)
    r = c.patch("/runtime/config", json={"scheduler.max_batch": 4},
                headers={"Authorization": "Bearer tok-op",
                         "X-InfEngine-Actor": "spoofed-admin"})
    assert r.status_code == 200
    rec = audit.latest()
    assert rec.actor == "operator"          # token identity, not "spoofed-admin"
    assert rec.role == "OPERATOR"


def test_actor_header_ignored_even_with_viewer_token():
    from starlette.testclient import TestClient
    app, audit = _secured_app()
    c = TestClient(app)
    r = c.patch("/runtime/config", json={"scheduler.max_batch": 4},
                headers={"Authorization": "Bearer tok-view",
                         "X-InfEngine-Actor": "operator"})
    # viewer cannot mutate regardless of the spoofed actor header
    assert r.status_code == 403
    assert audit.latest() is None


def test_actor_header_is_advisory_when_auth_disabled():
    """Documented behavior: with authentication disabled the actor header is
    advisory client metadata (no authenticated identity exists)."""
    from starlette.testclient import TestClient
    from infengine.serve import create_app

    engine = make_engine(max_batch=2)
    engine.tokenizer = _StubTok()
    audit = AuditStore(":memory:")
    app = create_app(get_engine=lambda: engine, auth=AuthConfig({}), audit_store=audit)
    c = TestClient(app)
    r = c.patch("/runtime/config", json={"scheduler.max_batch": 4},
                headers={"X-InfEngine-Actor": "local-dev"})
    assert r.status_code == 200
    assert audit.latest().actor == "local-dev"



# --------------------------------------------------------- security unit
def test_auth_config_fail_closed_on_bad_config():
    with pytest.raises(AuthError):
        AuthConfig({"t": {"actor": "a", "role": "SUPERUSER"}})
    with pytest.raises(AuthError):
        AuthConfig({"t": {"role": "VIEWER"}})


def test_auth_role_model():
    auth = AuthConfig({"t": {"actor": "a", "role": "VIEWER"}})
    ident = auth.authenticate({"authorization": "Bearer t"})
    assert ident.role is Role.VIEWER
    auth.require(ident, Action.READ)
    with pytest.raises(AuthError):
        auth.require(ident, Action.MUTATE)


# --------------------------------------------------------- policy files
def test_policy_flat_yaml(tmp_path):
    p = tmp_path / "policy.yaml"
    p.write_text(
        "# a multi-control policy\n"
        "runtime_concurrency: 32\n"
        "runtime_max_scheduled_tokens: 1024\n"
        "runtime_kv_watermark: 0.1\n"
    )
    policy = load_policy(str(p))
    assert policy == {"runtime_concurrency": 32,
                      "runtime_max_scheduled_tokens": 1024,
                      "runtime_kv_watermark": 0.1}


def test_policy_json_and_wrapper(tmp_path):
    p = tmp_path / "policy.json"
    p.write_text(json.dumps({"policy": {"runtime_concurrency": 8}}))
    assert load_policy(str(p)) == {"runtime_concurrency": 8}


def test_policy_rejects_nested(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("runtime:\n  foo: 1\n")
    with pytest.raises(PolicyError):
        load_policy(str(p))


def test_policy_tolerates_utf8_bom(tmp_path):
    # Windows editors/PowerShell write a UTF-8 BOM; the loader must strip it.
    p = tmp_path / "bom.yaml"
    p.write_bytes(b"\xef\xbb\xbfruntime_concurrency: 8\n")
    assert load_policy(str(p)) == {"runtime_concurrency": 8}


def test_validate_policy_detects_unknown_and_conflict():
    caps = [
        {"intent": "runtime_concurrency", "internal_key": "runtime_max_running_seqs",
         "supported_on_selected_engine": True, "min": 1, "max": 64},
        {"intent": "runtime_max_running_seqs", "internal_key": "runtime_max_running_seqs",
         "supported_on_selected_engine": True, "min": 1, "max": 64},
    ]
    resolved, errors = validate_policy({"runtime_concurrency": 32}, caps)
    assert not errors and resolved == {"runtime_concurrency": 32}
    _, errors = validate_policy({"nope": 1}, caps)
    assert errors and errors[0]["control"] == "nope"
    _, errors = validate_policy(
        {"runtime_concurrency": 32, "runtime_max_running_seqs": 8}, caps)
    assert errors and "conflicting alias" in errors[0]["error"]


# --------------------------------------------------------- evaluator
def test_phase_order_interleaves():
    from infengine.evaluator import _phase_order
    assert _phase_order(3, True) == [
        ["baseline", "candidate"], ["candidate", "baseline"],
        ["baseline", "candidate"]]
    assert _phase_order(2, False) == [
        ["baseline", "candidate"], ["baseline", "candidate"]]


def test_load_experiment_multi_control(tmp_path):
    from infengine.evaluator import load_experiment
    p = tmp_path / "exp.yaml"
    p.write_text(
        "experiment:\n"
        "  engine: vllm\n"
        "  model: Qwen2.5-0.5B-Instruct\n"
        "  workload:\n"
        "    profile: closed_loop\n"
        "    offered_concurrency: 64\n"
        "    duration_seconds: 30\n"
        "  baseline:\n"
        "    runtime_concurrency: 64\n"
        "    runtime_max_scheduled_tokens: 8192\n"
        "  candidate:\n"
        "    runtime_concurrency: 32\n"
        "    runtime_max_scheduled_tokens: 1024\n"
        "  repetitions: 3\n"
    )
    exp = load_experiment(str(p))
    assert exp.baseline == {"runtime_concurrency": 64,
                            "runtime_max_scheduled_tokens": 8192}
    assert exp.candidate["runtime_concurrency"] == 32
    assert exp.repeats == 3
    assert exp.workload.clients == 64


def test_evaluator_preflight_rejects_unknown_and_bounds():
    from infengine.evaluator import PolicyExperiment, preflight, WorkloadSpec
    from infengine.experiment import MeasurementSpec

    class FakeControl:
        def health(self):
            return {"status": "ok", "admitting": True}

        def capabilities(self):
            return [{"intent": "runtime_concurrency",
                     "internal_key": "runtime_max_running_seqs",
                     "supported_on_selected_engine": True, "min": 1, "max": 8}]

        def get_config(self):
            return {"desired": {"runtime_max_running_seqs": 4},
                    "desired_revision": 1}

    exp = PolicyExperiment(
        experiment_id="e", engine="vllm", model="m",
        baseline={"runtime_concurrency": 4}, candidate={"runtime_concurrency": 99},
        workload=WorkloadSpec(), measurement=MeasurementSpec())
    out = preflight(exp, FakeControl())
    assert out["ok"] is False
    assert out["errors"]


def test_compare_no_measurable_difference():
    from infengine.evaluator import PolicyExperiment, _compare, WorkloadSpec
    from infengine.experiment import MeasurementSpec

    exp = PolicyExperiment(
        experiment_id="e", engine="vllm", model="m",
        baseline={"x": 1}, candidate={"x": 2}, workload=WorkloadSpec(),
        measurement=MeasurementSpec(decision_metrics=["throughput_req_s"]))
    base = [{"metrics": {"throughput_req_s": {"value": 100.0}}}]
    cand = [{"metrics": {"throughput_req_s": {"value": 101.0}}}]
    comparison, outcome = _compare(exp, base, cand)
    assert outcome == "NO_MEASURABLE_DIFFERENCE"
    base = [{"metrics": {"throughput_req_s": {"value": 100.0}}}]
    cand = [{"metrics": {"throughput_req_s": {"value": 150.0}}}]
    _, outcome = _compare(exp, base, cand)
    assert outcome == "OBSERVED_TRADE_OFF"


# --------------------------------------------------------- unified CLI
def test_unified_version_runs(capsys):
    from infengine import unified
    assert unified.main(["version"]) == 0
    out = capsys.readouterr().out
    assert "0.1.0" in out


def test_unified_apply_dry_run(monkeypatch, tmp_path, capsys):
    from infengine import unified

    class FakeClient:
        def capabilities(self):
            return [{"intent": "runtime_concurrency",
                     "internal_key": "runtime_max_running_seqs",
                     "supported_on_selected_engine": True, "min": 1, "max": 64}]

    monkeypatch.setattr(unified, "_client", lambda args: FakeClient())
    p = tmp_path / "policy.yaml"
    p.write_text("runtime_concurrency: 32\n")
    assert unified.main(["apply", str(p), "--dry-run"]) == 0
    assert "dry_run" in capsys.readouterr().out


# ------------------------------------------------- HTTP rollback via audit path
def test_http_partial_apply_rolls_back_and_records():
    from infengine.serve import create_app
    from starlette.testclient import TestClient

    client = PartialVllmClient(fail_field="runtime_max_scheduled_tokens")
    audit = AuditStore(":memory:")
    app = create_app(runtime="vllm", runtime_client=client, audit_store=audit)
    c = TestClient(app)
    r = c.patch("/runtime/config", json={
        "runtime_max_running_seqs": 2, "runtime_max_scheduled_tokens": 256})
    assert r.status_code == 422
    body = r.json()
    assert body["status"] == "FAILED"
    assert body["rollback"]["status"] == "ROLLBACK_SUCCEEDED"
    assert client.config["runtime_max_running_seqs"] == 4  # restored
    rec = audit.latest()
    assert rec.result == "FAILED"
    assert rec.rollback["status"] == "ROLLBACK_SUCCEEDED"
    assert rec.pre_state["runtime_max_running_seqs"] == 4


def test_http_stale_revision_zero_mutation():
    from infengine.serve import create_app
    from starlette.testclient import TestClient

    client = FakeVllmRuntimeClient()
    app = create_app(runtime="vllm", runtime_client=client)
    c = TestClient(app)
    r = c.patch("/runtime/config", json={"runtime_max_running_seqs": 2},
                headers={"If-Match": "99"})
    assert r.status_code == 409
    assert client.set_calls == []
    assert client.config["runtime_max_running_seqs"] == 4


# ------------------------------------------------- evaluator orchestration
class _FakeEvalControl:
    def __init__(self):
        self.current = {"runtime_max_running_seqs": 16,
                        "runtime_max_scheduled_tokens": 4096}
        self.revision = 0
        self.applied = []

    def health(self):
        return {"status": "ok", "admitting": True}

    def capabilities(self):
        return [
            {"intent": "runtime_concurrency",
             "internal_key": "runtime_max_running_seqs",
             "supported_on_selected_engine": True, "min": 1, "max": 64},
            {"intent": "runtime_max_scheduled_tokens",
             "internal_key": "runtime_max_scheduled_tokens",
             "supported_on_selected_engine": True, "min": 1, "max": 8192},
        ]

    def get_config(self):
        return {"desired": dict(self.current), "desired_revision": self.revision,
                "observed": dict(self.current)}

    def patch_config(self, desired, if_match=None):
        self.applied.append(dict(desired))
        for k, v in desired.items():
            self.current[k] = v
        self.revision += 1
        return 200, {"status": "CONVERGED", "observed": dict(self.current),
                     "desired_revision": self.revision,
                     "steps": [{"param": k, "desired": v, "observed_after": v,
                                "status": "VERIFIED"} for k, v in desired.items()]}


def test_evaluator_multi_control_orchestration(monkeypatch):
    from infengine import evaluator as ev

    fake = _FakeEvalControl()
    monkeypatch.setattr(ev, "ControlPlaneClient", lambda url, **kw: fake)
    monkeypatch.setattr(ev, "_idle_check", lambda cfg, spec: True)
    monkeypatch.setattr(ev, "_warmup", lambda spec, cfg: True)

    def fake_measure(spec, cfg, bound):
        conc = fake.current["runtime_max_running_seqs"]
        return {
            "ok": True,
            "counts": {"completed": 20},
            "wall_s": 1.0,
            "environment": {},
            "metrics": {
                "throughput_req_s": {"value": 100.0 if conc >= 16 else 60.0},
                "ttft_p95_s": {"value": 0.1 if conc >= 16 else 0.3},
            },
        }

    monkeypatch.setattr(ev, "_measure_phase", fake_measure)
    exp = ev.PolicyExperiment(
        experiment_id="multi", engine="vllm", model="m",
        baseline={"runtime_concurrency": 16, "runtime_max_scheduled_tokens": 4096},
        candidate={"runtime_concurrency": 8, "runtime_max_scheduled_tokens": 1024},
        measurement=ev.MeasurementSpec(window_s=1, poll_interval_s=0.1, min_polls=1,
                                       min_requests=1,
                                       decision_metrics=["throughput_req_s", "ttft_p95_s"]),
        repeats=3, alternate=True)
    result = ev.run_policy_experiment(exp, control_url="http://x",
                                      engine_url="http://y")
    assert result["outcome"] == "OBSERVED_TRADE_OFF"
    # interleaved A/B ordering recorded
    assert result["run_order"][1] == ["candidate", "baseline"]
    assert all(p["ok"] for p in result["phases"])
    # multi-control applied and verified per phase
    assert all(len(p["control_evidence"]["requested"]) == 2 for p in result["phases"])
    # original pre-experiment state restored and verified
    assert result["restore"]["verified"] is True
    assert fake.current == {"runtime_max_running_seqs": 16,
                            "runtime_max_scheduled_tokens": 4096}


def test_evaluator_candidate_failure_restores(monkeypatch):
    from infengine import evaluator as ev

    class FailControl(_FakeEvalControl):
        def patch_config(self, desired, if_match=None):
            self.applied.append(dict(desired))
            if desired.get("runtime_max_running_seqs") == 8:
                return 200, {"status": "FAILED", "observed": dict(self.current),
                             "desired_revision": self.revision}
            for k, v in desired.items():
                self.current[k] = v
            self.revision += 1
            return 200, {"status": "CONVERGED", "observed": dict(self.current),
                         "desired_revision": self.revision}

    fake = FailControl()
    monkeypatch.setattr(ev, "ControlPlaneClient", lambda url, **kw: fake)
    monkeypatch.setattr(ev, "_idle_check", lambda cfg, spec: True)
    monkeypatch.setattr(ev, "_warmup", lambda spec, cfg: True)
    monkeypatch.setattr(ev, "_measure_phase",
                        lambda spec, cfg, bound: {"ok": True, "counts": {"completed": 5},
                                                  "wall_s": 1.0, "environment": {},
                                                  "metrics": {}})
    exp = ev.PolicyExperiment(
        experiment_id="fail", engine="vllm", model="m",
        baseline={"runtime_concurrency": 16, "runtime_max_scheduled_tokens": 4096},
        candidate={"runtime_concurrency": 8, "runtime_max_scheduled_tokens": 1024},
        measurement=ev.MeasurementSpec(decision_metrics=["throughput_req_s"]),
        repeats=1)
    result = ev.run_policy_experiment(exp, control_url="http://x", engine_url="http://y")
    assert result["outcome"] == "INCOMPLETE"
    assert result["restore"].get("verified") is True
    assert fake.current == {"runtime_max_running_seqs": 16,
                            "runtime_max_scheduled_tokens": 4096}

