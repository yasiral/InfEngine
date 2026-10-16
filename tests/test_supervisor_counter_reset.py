"""M9B.2: reset-safe cumulative-counter (preemption) evidence.

Synthetic counter sequences only (no live engine restart). Proves that
``build_signal_window`` computes cumulative-counter statistics over the latest
contiguous counter epoch, so a counter reset/restart or an engine-instance change
cannot manufacture a false preemption increase, without changing the KV policy.
"""

from infengine.supervisor import (
    CUMULATIVE_COUNTER_METRICS,
    CapabilityBounds,
    Guardrail,
    Guardrails,
    KvWatermarkObjective,
    KvWatermarkPolicy,
    SupervisorConfig,
    build_signal_window,
)
from infengine.telemetry import RuntimeSnapshot

FIXED = "2026-09-08T00:00:00+00:00"
NOW = 1_000_000.0


def _snap(preemptions=None, instance="i1", kv=0.2, quality=None):
    kwargs = dict(
        engine="vllm", instance_id=instance, collected_at=FIXED, source="test",
        requests_running=4, requests_waiting=6, kv_usage=kv, ttft_p95_s=0.1,
        quality=quality or {},
    )
    if preemptions is not None:
        kwargs["engine_specific"] = {"num_preemptions": preemptions}
    return RuntimeSnapshot(**kwargs)


def _win(now, *snaps):
    n = len(snaps)
    return [(now - (n - 1 - i) * 1.0, s) for i, s in enumerate(snaps)]


def _pwin(*obs, kv=0.2, metrics=("num_preemptions",)):
    """Build a window from observations; each item is ``value`` or
    ``(value, instance_id)``."""
    snaps = []
    for item in obs:
        if isinstance(item, tuple):
            snaps.append(_snap(preemptions=item[0], instance=item[1], kv=kv))
        else:
            snaps.append(_snap(preemptions=item, kv=kv))
    return build_signal_window(
        _win(NOW, *snaps), now=NOW, window_s=10.0, metrics=metrics
    )


def _kv_config(**objective_kw):
    return SupervisorConfig(
        window_s=10.0, min_samples=3, cooldown_s=60.0,
        signal_metrics=("requests_running", "requests_waiting", "kv_usage", "num_preemptions"),
        objective=KvWatermarkObjective(**objective_kw),
        guardrails=Guardrails(
            latency=Guardrail(name="latency", metric="ttft_p95_s", max_value=1.0,
                              enabled=True, required_for_scale_up=True),
            kv_usage=Guardrail(name="kv_usage", metric="kv_usage", max_value=0.95,
                               enabled=True, required_for_scale_up=True),
        ),
    )


def _evaluate(window, current=0.5):
    policy = KvWatermarkPolicy(_kv_config())
    return policy.evaluate(window, current_value=current,
                           bounds=CapabilityBounds(0.0, 1.0))


# ------------------------------------------------------- core segment behaviour
def test_increase_is_max_minus_min():
    agg = _pwin(100, 102, 105).get("num_preemptions")
    assert agg.samples == 3
    assert agg.minimum == 100 and agg.maximum == 105
    assert agg.maximum - agg.minimum == 5


def test_unchanged_counter_is_valid_zero_increase():
    agg = _pwin(100, 100, 100).get("num_preemptions")
    assert agg.samples == 3
    assert agg.minimum == 100 and agg.maximum == 100
    assert agg.maximum - agg.minimum == 0


def test_reset_does_not_produce_false_increase():
    # 105 -> 0 -> 1: never 105.
    agg = _pwin(105, 0, 1).get("num_preemptions")
    assert agg.maximum != 105
    assert agg.minimum == 0 and agg.maximum == 1
    assert agg.maximum - agg.minimum == 1
    assert agg.samples == 2  # only the post-reset segment


def test_reset_does_not_combine_epochs():
    # 100 -> 105 -> 0 -> 2: only [0, 2] is used.
    agg = _pwin(100, 105, 0, 2).get("num_preemptions")
    assert agg.minimum == 0 and agg.maximum == 2
    assert agg.maximum - agg.minimum == 2
    assert agg.samples == 2
    assert agg.median == 1.0  # median over the valid post-reset segment only


# --------------------------------------------------------- missing / stale
def test_missing_counter_is_never_zero():
    w = build_signal_window(
        _win(NOW, _snap(), _snap(), _snap()), now=NOW, window_s=10.0,
        metrics=("num_preemptions",),
    )
    agg = w.get("num_preemptions")
    assert agg.samples == 0
    assert agg.minimum is None and agg.maximum is None and agg.median is None


def test_stale_counter_is_never_zero_and_not_fresh_evidence():
    snap = _snap(preemptions=7, quality={"num_preemptions": "stale"})
    w = build_signal_window(
        _win(NOW, snap, _snap(), _snap()), now=NOW, window_s=10.0,
        metrics=("num_preemptions",),
    )
    agg = w.get("num_preemptions")
    # Not fresh, so no decision evidence; recorded diagnostically only.
    assert agg.samples == 0
    assert agg.minimum is None and agg.maximum is None
    assert agg.stale == 1 and agg.stale_maximum == 7.0


# ------------------------------------------------------- engine-instance change
def test_instance_change_breaks_counter_continuity():
    agg = _pwin((10, "i1"), (12, "i1"), (13, "i2")).get("num_preemptions")
    # Only the new-instance epoch is valid; no cross-instance combination.
    assert agg.samples == 1
    assert agg.minimum == 13 and agg.maximum == 13


def test_instance_change_then_contiguous_growth_uses_new_epoch():
    agg = _pwin((10, "i1"), (12, "i1"), (3, "i2"), (4, "i2")).get("num_preemptions")
    assert agg.samples == 2
    assert agg.minimum == 3 and agg.maximum == 4
    assert agg.maximum - agg.minimum == 1


# --------------------------------------------------------------- KV policy
def test_insufficient_post_reset_evidence_holds():
    # 100 -> 105 -> 0 : post-reset segment has 1 sample < min_samples=3.
    window = _pwin(100, 105, 0, kv=0.95,
                   metrics=("kv_usage", "num_preemptions", "ttft_p95_s"))
    evaluation = _evaluate(window)
    assert evaluation.kind.value == "HOLD"
    assert any("num_preemptions" in r and "insufficient telemetry" in r
               for r in evaluation.reasons)


def test_sufficient_post_reset_evidence_uses_only_segment():
    # 100 -> 105 -> 0 -> 2 -> 3 : valid segment [0, 2, 3], delta 3 (never 105).
    window = _pwin(100, 105, 0, 2, 3, kv=0.95,
                   metrics=("kv_usage", "num_preemptions", "ttft_p95_s"))
    evaluation = _evaluate(window)
    assert evaluation.evidence["preemptions_in_window"] == 3
    assert evaluation.kind.value == "PROPOSE_UP"


def test_reset_with_still_insufficient_segment_holds_not_spikes():
    # 50 -> 60 -> 0 : would be a spurious 60 under max-min; holds instead.
    window = _pwin(50, 60, 0, kv=0.95,
                   metrics=("kv_usage", "num_preemptions", "ttft_p95_s"))
    evaluation = _evaluate(window)
    assert evaluation.kind.value == "HOLD"


# --------------------------------------- unavailable observation = hard break
def test_unavailable_observation_breaks_counter_continuity():
    # 100 (FRESH), unavailable (SCRAPE_FAILED), 105 (FRESH) must NOT be a 5
    # preemption increase: the unobserved interval breaks the segment.
    gap = _snap(quality={"num_preemptions": "scrape_failed"})  # no value
    snaps = [_snap(preemptions=100), gap, _snap(preemptions=105)]
    w = build_signal_window(_win(NOW, *snaps), now=NOW, window_s=10.0,
                            metrics=("num_preemptions",))
    agg = w.get("num_preemptions")
    assert agg.samples == 1
    assert agg.minimum == 105 and agg.maximum == 105
    assert agg.maximum - agg.minimum != 5


def test_missing_observation_between_fresh_values_breaks_continuity():
    snaps = [_snap(preemptions=100), _snap(), _snap(preemptions=105)]
    w = build_signal_window(_win(NOW, *snaps), now=NOW, window_s=10.0,
                            metrics=("num_preemptions",))
    agg = w.get("num_preemptions")
    assert agg.samples == 1
    assert agg.minimum == 105 and agg.maximum == 105


def test_unavailable_gap_then_contiguous_growth_uses_only_post_gap():
    gap = _snap(quality={"num_preemptions": "scrape_failed"})
    snaps = [
        _snap(preemptions=100), _snap(preemptions=102), gap,
        _snap(preemptions=105), _snap(preemptions=107),
    ]
    w = build_signal_window(_win(NOW, *snaps), now=NOW, window_s=10.0,
                            metrics=("num_preemptions",))
    agg = w.get("num_preemptions")
    assert agg.samples == 2
    assert agg.minimum == 105 and agg.maximum == 107
    assert agg.maximum - agg.minimum == 2


def test_unavailable_observation_holds_kv_policy():
    gap = _snap(quality={"num_preemptions": "scrape_failed"}, kv=0.95)
    snaps = [_snap(preemptions=100, kv=0.95), gap, _snap(preemptions=105, kv=0.95)]
    w = build_signal_window(_win(NOW, *snaps), now=NOW, window_s=10.0,
                            metrics=("kv_usage", "num_preemptions", "ttft_p95_s"))
    evaluation = _evaluate(w)
    # Only one valid post-gap sample < min_samples=3 -> fail closed.
    assert evaluation.kind.value == "HOLD"


# ---------------------------------------------------------- narrow scope
def test_counter_set_is_declared_and_narrow():
    assert "num_preemptions" in CUMULATIVE_COUNTER_METRICS
    # Derived scalars are NOT cumulative counters.
    assert "iteration_tokens_p95" not in CUMULATIVE_COUNTER_METRICS


def test_non_counter_metric_is_not_segmented():
    # A derived scalar that happens to decrease must keep all fresh samples
    # (only cumulative counters get reset segmentation).
    snaps = [
        _snap(preemptions=None), _snap(preemptions=None), _snap(preemptions=None),
    ]
    # put an unrelated decreasing value into each snapshot's engine_specific
    for snap, value in zip(snaps, (10, 5, 3)):
        snap.engine_specific["iteration_tokens_p95"] = value
    w = build_signal_window(
        _win(NOW, *snaps), now=NOW, window_s=10.0,
        metrics=("iteration_tokens_p95",),
    )
    agg = w.get("iteration_tokens_p95")
    assert agg.samples == 3
    assert agg.minimum == 3 and agg.maximum == 10
