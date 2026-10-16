"""M5B rolling-window tests: bounded history + windowed histogram quantiles."""

import pytest

from infengine.telemetry import RollingHistograms

# (le, count) tuples; +Inf bucket last.
TTFT_A = ((0.01, 0), (0.1, 10), (1.0, 20), ("+Inf", 20))
TTFT_B = ((0.01, 5), (0.1, 25), (1.0, 40), ("+Inf", 40))


def test_quantile_requires_two_samples():
    w = RollingHistograms()
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    assert w.quantile("ttft", 0.95, window_s=30.0) is None


def test_quantile_computed_from_window_delta():
    w = RollingHistograms()
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=1.0, buckets=TTFT_B)
    # Delta over window: counts become ((0.01,5),(0.1,15),(1.0,20),(inf,20)).
    # total=20; q50 target=10 -> second bucket [0.01,0.1]: 0.01+0.09*(10-5)/(15-5)
    q50 = w.quantile("ttft", 0.5, window_s=30.0)
    assert q50 == pytest.approx(0.01 + 0.09 * (10 - 5) / (15 - 5))
    assert w.total_in_window("ttft", window_s=30.0) == 20


def test_counter_reset_drops_pre_reset_history():
    w = RollingHistograms()
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=1.0, buckets=TTFT_B)
    # A reset: cumulative counts drop below the previous sample.
    w.push("ttft", ts=2.0, buckets=((0.01, 0), (0.1, 3), (1.0, 4), ("+Inf", 4)))
    # Only one sample remains after the reset -> no quantile.
    assert w.quantile("ttft", 0.95, window_s=30.0) is None
    # A second post-reset sample yields a quantile from post-reset data only.
    w.push("ttft", ts=3.0, buckets=((0.01, 2), (0.1, 5), (1.0, 8), ("+Inf", 8)))
    q = w.quantile("ttft", 0.95, window_s=30.0)
    assert q is not None


def test_bucket_layout_change_resets():
    w = RollingHistograms()
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=1.0, buckets=((0.05, 1), (1.0, 2), ("+Inf", 2)))
    assert w.quantile("ttft", 0.95, window_s=30.0) is None


def test_retention_prunes_old_samples():
    w = RollingHistograms(retention_s=10.0)
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=1.0, buckets=TTFT_B)
    # Now push at t=20 (beyond retention); the window only sees samples at t=20
    # plus any within retention. t=0/1 are pruned.
    w.push("ttft", ts=20.0, buckets=TTFT_B)
    assert w.quantile("ttft", 0.95, window_s=30.0) is None  # only one sample in retention


def test_bounded_history_length():
    w = RollingHistograms(maxlen=3)
    for t in range(5):
        w.push("ttft", ts=float(t), buckets=TTFT_A)
    q = w._series["ttft"]
    assert len(q) <= 3


def test_quantiles_map():
    w = RollingHistograms()
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=1.0, buckets=TTFT_B)
    out = w.quantiles("ttft", {"p50": 0.5, "p95": 0.95, "p99": 0.99}, window_s=30.0)
    assert out["p50"] <= out["p95"] <= out["p99"]


def test_no_samples_when_window_delta_zero():
    w = RollingHistograms()
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=1.0, buckets=TTFT_A)  # no change
    assert w.total_in_window("ttft", window_s=30.0) == 0
    assert w.quantile("ttft", 0.95, window_s=30.0) is None


def test_empty_window_when_no_in_window_baseline():
    # Two samples exist but the gap exceeds the window, so there is no valid
    # in-window baseline -> no quantile (NO_SAMPLES at the collector level).
    w = RollingHistograms(maxlen=10, retention_s=1000.0)
    w.push("ttft", ts=0.0, buckets=TTFT_A)
    w.push("ttft", ts=100.0, buckets=TTFT_B)
    assert w.quantile("ttft", 0.95, window_s=30.0) is None
    assert w.total_in_window("ttft", window_s=30.0) is None


def test_low_sample_count_still_yields_quantile():
    # A window delta of just 2 observations still yields a quantile (only a
    # zero/insufficient delta is reported as NO_SAMPLES upstream).
    w = RollingHistograms()
    empty = ((0.01, 0), (0.1, 0), (1.0, 0), ("+Inf", 0))
    two = ((0.01, 0), (0.1, 2), (1.0, 2), ("+Inf", 2))
    w.push("ttft", ts=0.0, buckets=empty)
    w.push("ttft", ts=1.0, buckets=two)
    assert w.total_in_window("ttft", window_s=30.0) == 2
    assert w.quantile("ttft", 0.5, window_s=30.0) is not None
