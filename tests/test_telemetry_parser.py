"""M5B parser tests: Prometheus text exposition parsing."""

import math

import pytest

from infengine.telemetry import (
    ExpositionParseError,
    histogram_quantile,
    parse_exposition,
)

from telemetry_helpers import histogram_text, sglang_loads_exposition, vllm_exposition


# ------------------------------------------------------------------ basics
def test_parses_plain_gauge():
    expo = parse_exposition("# TYPE foo gauge\nfoo 3.5\n")
    assert expo.find("foo").value == 3.5
    assert expo.family("foo").type == "gauge"


def test_parses_labels_and_escaping():
    text = '# TYPE foo gauge\nfoo{bar="a\\"b",baz="x\\\\y",engine="0"} 1\n'
    expo = parse_exposition(text)
    sample = expo.find("foo")
    assert sample.label("bar") == 'a"b'
    assert sample.label("baz") == "x\\y"
    assert sample.label("engine") == "0"


def test_parses_inf_and_nan_values():
    expo = parse_exposition("# TYPE foo gauge\nfoo +Inf\nbar -Inf\nbaz NaN\n")
    assert expo.find("foo").value == math.inf
    assert expo.find("bar").value == -math.inf
    assert math.isnan(expo.find("baz").value)


def test_parses_timestamp_ignored_for_values():
    expo = parse_exposition("# TYPE foo gauge\nfoo 1.5 123456\n")
    assert expo.find("foo").value == 1.5
    assert expo.find("foo").timestamp == 123456.0


def test_parses_colon_namespaced_families():
    expo = parse_exposition("# TYPE vllm:num_requests_running gauge\n"
                            "vllm:num_requests_running{engine=\"0\"} 2\n")
    assert expo.find("vllm:num_requests_running").value == 2


def test_malformed_line_raises():
    with pytest.raises(ExpositionParseError):
        parse_exposition("foo{engine 1\n")
    with pytest.raises(ExpositionParseError):
        parse_exposition("foo 1.2.3\n")


# ---------------------------------------------------------------- histogram
def test_parses_histogram_series():
    text = histogram_text("m", [(0.1, 2), (1.0, 5), ("+Inf", 9)], 7.5, 9)
    expo = parse_exposition(text)
    series = expo.histogram_series("m")
    assert len(series) == 1
    s = series[0]
    assert s.count == 9
    assert s.sum == 7.5
    assert [le for le, _ in s.buckets] == [0.1, 1.0, math.inf]
    assert [c for _, c in s.buckets] == [2, 5, 9]


def test_histogram_absent_family_returns_empty():
    expo = parse_exposition("# TYPE x gauge\nx 1\n")
    assert expo.histogram_series("m") == []


def test_histogram_quantile_interpolation():
    buckets = ((0.1, 0), (1.0, 100), ("+Inf", 100))
    # Rank 50 falls in the second bucket [0.1, 1.0]; linear interpolation.
    assert histogram_quantile(0.5, buckets) == pytest.approx(0.55)
    assert histogram_quantile(0.9, buckets) == pytest.approx(0.91)
    assert histogram_quantile(0.99, buckets) == pytest.approx(0.991)


def test_histogram_quantile_empty_distribution():
    assert histogram_quantile(0.5, []) is None
    assert histogram_quantile(0.5, ((0.1, 0), ("+Inf", 0))) is None


def test_histogram_quantile_overflow_uses_lower_bound():
    # All observations beyond 1.0 -> the +Inf bucket is the first to contain the
    # rank; returns the lower bound (1.0).
    buckets = ((1.0, 0), ("+Inf", 1))
    assert histogram_quantile(1.0, buckets) == 1.0


# ------------------------------------------------------------ engine fixtures
def test_parses_vllm_exposition():
    expo = parse_exposition(vllm_exposition(ttft=([(0.1, 5), ("+Inf", 10)], 2.0, 10)))
    assert expo.find("vllm:num_requests_running").value == 2
    assert expo.find("vllm:kv_cache_usage_perc").value == 0.42
    assert len(expo.histogram_series("vllm:time_to_first_token_seconds")) == 1
    assert expo.histogram_series("vllm:time_to_first_token_seconds")[0].count == 10


def test_parses_sglang_loads_exposition():
    expo = parse_exposition(sglang_loads_exposition(running=1, waiting=2, token_usage=0.5))
    assert expo.find("sglang_num_running_reqs", {"dp_rank": "0"}).value == 1
    assert expo.find("sglang_num_waiting_reqs").value == 2
    assert expo.find("sglang_token_usage").value == 0.5
    assert expo.find("sglang_timestamp").value > 0