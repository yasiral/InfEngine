"""M13 round 3: unified CLI multi-control policy review (``infengine review``).

Deterministic: a fake control plane (no live engine) is injected in place of the
HTTP client. Covers 2-control approval, rejected mutated/invalid policy,
expired/stale policy, history/audit entry, restore after approved apply, and
single-control compatibility.
"""

import json

from infengine import unified


class FakeReviewControl:
    def __init__(self):
        self.current = {
            "runtime_max_running_seqs": 16,
            "runtime_max_scheduled_tokens": 4096,
        }
        self.revision = 1
        self.patches = []
        self.history_records = []

    def capabilities(self):
        return [
            {"intent": "runtime_concurrency",
             "internal_key": "runtime_max_running_seqs",
             "supported_on_selected_engine": True, "transition": "HOT",
             "min": 1, "max": 64},
            {"intent": "runtime_max_scheduled_tokens",
             "internal_key": "runtime_max_scheduled_tokens",
             "supported_on_selected_engine": True, "transition": "HOT",
             "min": 1, "max": 8192},
        ]

    def get_config(self):
        return {"desired": dict(self.current), "observed": dict(self.current),
                "desired_revision": self.revision}

    def patch_config(self, desired, if_match=None):
        self.patches.append((dict(desired), if_match))
        if if_match is not None and if_match != self.revision:
            return 409, {"status": "CONFLICT", "desired_revision": self.revision}
        pre = {k: self.current.get(k) for k in desired}
        for k, v in desired.items():
            self.current[k] = v
        before = self.revision
        self.revision += 1
        record = {
            "change_id": f"c{before}",
            "created_at": "2020-01-01T00:00:00Z",
            "actor": "yasir", "role": "OPERATOR",
            "requested": dict(desired), "pre_state": pre,
            "observed": {k: self.current[k] for k in desired},
            "result": "CONVERGED",
            "revision_before": before, "revision_after": self.revision,
        }
        self.history_records.append(record)
        return 200, {
            "status": "CONVERGED", "observed": dict(self.current),
            "desired_revision": self.revision,
            "steps": [{"param": k, "desired": v, "observed_after": v,
                       "status": "VERIFIED"} for k, v in desired.items()],
        }

    def history(self, limit=50, offset=0):
        return 200, {"history": list(reversed(self.history_records))[:limit],
                     "persisted": True}

    def history_one(self, change_id):
        for record in self.history_records:
            if record["change_id"] == change_id:
                return 200, record
        return 404, {"detail": "no such change"}


def _patch_client(monkeypatch, control):
    monkeypatch.setattr(unified, "_client", lambda args: control)


def _write_policy(tmp_path, body):
    p = tmp_path / "policy.yaml"
    p.write_text(body, encoding="utf-8")
    return str(p)


POLICY = "runtime_concurrency: 8\nruntime_max_scheduled_tokens: 1024\n"


def test_review_policy_two_control_approved(monkeypatch, tmp_path, capsys):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, POLICY)
    rc = unified.main(["review", path, "--yes"])
    assert rc == 0
    assert control.current == {"runtime_max_running_seqs": 8,
                               "runtime_max_scheduled_tokens": 1024}
    assert control.patches == [({
        "runtime_max_running_seqs": 8, "runtime_max_scheduled_tokens": 1024}, 1)]
    assert "CONVERGED" in capsys.readouterr().out


def test_review_policy_rejects_unknown_control(monkeypatch, tmp_path, capsys):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, "runtime_concurrency: 8\nnot_a_control: 1\n")
    rc = unified.main(["review", path, "--yes"])
    assert rc == 2
    assert control.patches == []  # zero mutation
    assert "not registrable" in capsys.readouterr().err


def test_review_policy_rejects_out_of_bounds(monkeypatch, tmp_path):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, "runtime_concurrency: 999\n")
    rc = unified.main(["review", path, "--yes"])
    assert rc == 2
    assert control.patches == []


def test_review_policy_expired_is_stale(monkeypatch, tmp_path):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, POLICY)
    # A non-positive TTL expires the proposal before approval.
    rc = unified.main(["review", path, "--yes", "--ttl", "-1"])
    assert rc == 1
    assert control.patches == []  # zero mutation


def test_review_policy_declined_without_confirmation(monkeypatch, tmp_path):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    path = _write_policy(tmp_path, POLICY)
    rc = unified.main(["review", path])
    assert rc == 0
    assert control.patches == []


def test_review_policy_history_and_restore(monkeypatch, tmp_path, capsys):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, POLICY)
    assert unified.main(["review", path, "--yes"]) == 0
    # audit/history entry exists
    assert unified.main(["history", "--json"]) == 0
    out = capsys.readouterr().out
    assert "c1" in out
    assert "runtime_max_running_seqs" in out
    # restore returns to the captured pre-state and verifies
    assert unified.main(["restore", "--json"]) == 0
    assert control.current == {"runtime_max_running_seqs": 16,
                               "runtime_max_scheduled_tokens": 4096}
    assert len(control.patches) == 2


def test_review_without_policy_delegates_to_single_control(monkeypatch):
    called = {}

    def fake_cmd_review(control, args):
        called["args"] = args
        return 7

    import infengine.operator as operator
    monkeypatch.setattr(operator, "cmd_review", fake_cmd_review)
    _patch_client(monkeypatch, FakeReviewControl())
    rc = unified.main(["review", "--yes"])
    assert rc == 7
    assert "args" in called


def test_review_help_documents_policy_file():
    parser = unified.build_parser()
    help_text = parser.format_help()
    assert "review" in help_text


def test_review_policy_alias_flag(monkeypatch, tmp_path):
    control = FakeReviewControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, POLICY)
    assert unified.main(["review", "--policy", path, "--yes"]) == 0
    assert control.current["runtime_max_running_seqs"] == 8


class MutatingControl(FakeReviewControl):
    """Simulates a concurrent operator change after the proposal is registered."""

    def __init__(self):
        super().__init__()
        self._gets = 0

    def get_config(self):
        self._gets += 1
        if self._gets == 3:  # register(1), display(2), fresh-evidence(3)
            self.current["runtime_max_running_seqs"] = 20
            self.revision = 2
        return super().get_config()


def test_review_policy_rejects_concurrent_value_change(monkeypatch, tmp_path):
    control = MutatingControl()
    _patch_client(monkeypatch, control)
    path = _write_policy(tmp_path, POLICY)
    rc = unified.main(["review", path, "--yes"])
    assert rc == 1
    assert control.patches == []  # stale -> zero mutation

