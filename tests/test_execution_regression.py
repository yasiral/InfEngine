"""M9B regression coverage for existing M5/M7 consumers.

The additive execution-stage families must not disturb the M5 snapshot
contract/serialization or the M7 supervisory signal path, and the legacy
``tpot_*`` fields must keep their historical values.
"""

from infengine.supervisor.signals import build_signal_window
from infengine.telemetry import (
    VllmTelemetryCollector,
    RuntimeSnapshot,
    validate_snapshot,
)

from telemetry_helpers import vllm_exposition, telemetry_server


def _sequence(events):
    state = {"i": 0}

    def route():
        event = events[state["i"] % len(events)]
        state["i"] += 1
        return event

    return route


PRE_A = ([(0.05, 0), (0.5, 10), ("+Inf", 10)], 2.0, 10)
PRE_B = ([(0.05, 5), (0.5, 25), ("+Inf", 25)], 4.0, 25)
TTFT_A = ([(0.01, 0), (0.1, 10), ("+Inf", 10)], 0.5, 10)
TTFT_B = ([(0.01, 5), (0.1, 25), ("+Inf", 25)], 1.0, 25)


def _two_poll_snapshot():
    a = vllm_exposition(ttft=TTFT_A, tpot=PRE_A, prefill=PRE_A)
    b = vllm_exposition(ttft=TTFT_B, tpot=PRE_B, prefill=PRE_B)
    with telemetry_server({"/metrics": _sequence([(200, a), (200, b)])}) as base:
        c = VllmTelemetryCollector(base, timeout=5.0)
        c.poll_once()
        return c.poll_once()


def test_dict_valued_engine_specific_round_trips():
    snap = _two_poll_snapshot()
    validate_snapshot(snap)
    assert isinstance(snap.engine_specific["request_prefill_time"], dict)
    restored = RuntimeSnapshot.from_dict(snap.to_dict())
    assert restored == snap
    assert restored.engine_specific["request_prefill_time"]["p95"] is not None


def test_legacy_tpot_is_unchanged_and_still_populates():
    snap = _two_poll_snapshot()
    assert snap.tpot_p50_s is not None
    assert snap.tpot_p95_s is not None
    # Historical behavior: the field is sourced from the native ITL histogram.
    assert "inter_token_latency_seconds" in __import__(
        "infengine.telemetry", fromlist=["FIELD_META"]
    ).FIELD_META["tpot_p95_s"].vllm_source


def test_m7_signal_window_accepts_new_families_without_crashing():
    snap = _two_poll_snapshot()
    samples = [(0.0, snap), (1.0, snap)]
    window = build_signal_window(
        samples,
        now=2.0,
        window_s=10.0,
        metrics=(
            "requests_running",
            "requests_waiting",
            "kv_usage",
            "iteration_tokens_p95",
            "inter_token_latency",   # dict-valued -> ignored, never coerced
            "request_prefill_time",  # dict-valued -> ignored
        ),
    )
    assert window.snapshots == 2
    assert window.get("requests_running").samples == 2
    # A dict-valued preserved family is not a numeric signal: no samples, no
    # crash, and never coerced to zero.
    assert window.get("inter_token_latency").samples == 0
    assert window.get("inter_token_latency").median is None
