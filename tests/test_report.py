"""M9C: local HTML report tests (pure panel/point derivation + self-containment)."""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infengine.report import build_panels, render_report, write_report  # noqa: E402
from test_sweep import _synth_result  # noqa: E402


def test_build_panels_has_all_required_charts():
    panels = {p["id"]: p for p in build_panels(_synth_result("vllm"))}
    for pid in ("throughput_req", "throughput_tokens", "latency_tail",
                "execution_stage", "concurrency", "scheduler", "kv", "pressure",
                "outcomes"):
        assert pid in panels
    # A-series throughput, B latency points present.
    assert panels["throughput_req"]["series"][0]["points"]
    assert panels["latency_tail"]["series"]


def test_unavailable_series_are_not_zeroed():
    result = _synth_result("sglang")  # decode is unavailable for SGLang
    panels = {p["id"]: p for p in build_panels(result)}
    stage = panels["execution_stage"]
    assert any(u["name"] == "decode (measured)" for u in stage["unavailable"])
    assert all(s["name"] != "decode (measured)" for s in stage["series"])


def test_outcome_zeros_are_plotted_not_marked_unavailable():
    # Zero failures are measured data and must be plotted (not hidden as
    # "unavailable"); only genuinely missing series become unavailable.
    panels = {p["id"]: p for p in build_panels(_synth_result("vllm"))}
    outcomes = panels["outcomes"]
    names = {s["name"] for s in outcomes["series"]}
    assert {"timeout", "error", "incomplete"} <= names
    for s in outcomes["series"]:
        assert any(pt[1] == 0 for pt in s["points"])
    assert outcomes["unavailable"] == []


def test_running_waiting_panel_has_data():
    panels = {p["id"]: p for p in build_panels(_synth_result("vllm"))}
    names = {s["name"] for s in panels["concurrency"]["series"]}
    assert "running (median)" in names and "waiting (median)" in names


def test_low_sample_latency_points_are_flagged():
    from infengine.sweep import SweepSpec, _aggregate_level

    run = __import__("test_sweep")._run(2.0)
    run["metrics"]["ttft_p95_s"]["sample_count"] = 1
    run["metrics"]["ttft_p95_s"]["polls_usable"] = 1
    lvl = {"index": 0, "offered_load": 2.0, "runs": [run]}
    lvl.update(_aggregate_level([run], SweepSpec(engine="vllm", model="m")))
    assert lvl["aggregate"]["ttft_p95_s"]["adequate"] is False
    panels = {p["id"]: p for p in build_panels(_synth_result("vllm", [lvl]))}
    ttft = [s for s in panels["latency_tail"]["series"] if s["name"] == "TTFT p95"][0]
    assert ttft["low_sample"][0] is True


def test_missing_ttft_is_a_gap_with_reason():
    from infengine.sweep import SweepSpec, _aggregate_level
    from test_sweep import _level, _run

    ok = _level(1.0, [_run(1.0)])
    missing_run = _run(4.0)
    missing_run["metrics"]["ttft_p95_s"] = {"value": None, "dominant_quality": "disabled",
                                            "sample_count": None, "polls_usable": None}
    missing = {"index": 1, "offered_load": 4.0, "runs": [missing_run]}
    missing.update(_aggregate_level([missing_run], SweepSpec(engine="vllm", model="m")))
    panels = {p["id"]: p for p in build_panels(_synth_result("vllm", [ok, missing]))}
    ttft = [s for s in panels["latency_tail"]["series"] if s["name"] == "TTFT p95"][0]
    ys = [pt[1] for pt in ttft["points"]]
    assert ys[0] is not None and ys[1] is None  # genuine gap for level 4, not zero
    notes = " ".join(panels["latency_tail"]["notes"])
    assert "unavailable" in notes and "gap" in notes


def test_levels_table_shows_completed_and_drained():
    from infengine.report import _levels_table_html
    from test_sweep import _level, _run

    run = _run(2.0, counts={"started": 20, "completed": 15, "drained": 5,
                            "timeout": 0, "error": 0, "incomplete": 0})
    table = _levels_table_html(_synth_result("vllm", [_level(2.0, [run])]))
    assert "completed" in table and "drained" in table
    assert "phase-end drain" in table
    assert "<td>15</td>" in table and "<td>5</td>" in table


def test_engine_specific_notes_present():
    vllm = {p["id"]: p for p in build_panels(_synth_result("vllm"))}
    notes = " ".join(vllm["execution_stage"]["notes"])
    assert "SGLang does NOT" in notes or "not synthesized" in notes


def test_report_is_self_contained():
    html = render_report(_synth_result("vllm"))
    # No external resources: no <script src>, no <link href>, no http(s) URLs in
    # resource positions. (Data may contain a base_url string; that is inert.)
    assert "<script src" not in html.lower()
    assert "<link" not in html.lower()
    for host in ("cdn.jsdelivr", "unpkg.com", "cdnjs", "googleapis", "http://", "https://"):
        assert host not in html.lower(), host
    assert "<canvas" in html
    # Embedded JSON payload with the same result.
    match = re.search(r'<script id="sweep-data" type="application/json">(.*?)</script>',
                      html, re.DOTALL)
    assert match, "embedded sweep-data script missing"
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert payload["result"]["schema"] == _synth_result("vllm")["schema"]
    assert payload["panels"]


def test_write_report(tmp_path):
    out = tmp_path / "report.html"
    write_report(_synth_result("vllm"), str(out))
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "InfEngine continuous-batching load sweep" in text
