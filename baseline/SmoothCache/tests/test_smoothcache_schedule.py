"""Unit tests for SmoothCache schedule generation (pure Python, no torch)."""

from hyvideo.smoothcache.schedule import (
    build_schedule,
    estimate_component_macs,
    rel_l1_loss,
    schedule_mac_reduction,
    schedule_skip_stats,
)


def _approx(value, eps=1e-9):
    """Plain-python approximate check (pytest-free)."""

    def check(actual):
        if isinstance(value, (list, tuple)):
            assert len(actual) == len(value)
            for a, b in zip(actual, value):
                assert abs(a - b) <= eps, (a, b)
            return True
        return abs(actual - value) <= eps

    return check


def test_all_errors_above_alpha_never_caches():
    curves = {"d_attn": [[0.5] * 50, [0.5] * 50, [0.5] * 50]}
    schedule = build_schedule(curves, alpha=0.1, num_steps=50, lookahead=3)
    assert schedule["d_attn"] == [1] * 50


def test_low_errors_cache_with_lookahead3_pattern():
    curves = {"d_attn": [[0.0] * 50, [0.0] * 50, [0.0] * 50]}
    schedule = build_schedule(curves, alpha=0.1, num_steps=50, lookahead=3)["d_attn"]
    # compute at 0, skip 1..3, compute at 4, skip 5..7, ...
    assert schedule[0] == 1
    assert schedule[1:4] == [0, 0, 0]
    assert schedule[4] == 1
    assert schedule[5:8] == [0, 0, 0]
    assert schedule[49] == 1  # last step always computed


def test_lag1_row_falls_back_to_single_skip():
    # Only the lag-1 row is below alpha; the largest-lag rows are too noisy.
    curves = {
        "d_mlp": [
            [0.0] * 50,  # lag 1: very similar
            [1.0] * 50,  # lag 2: too different
            [1.0] * 50,  # lag 3: too different
        ]
    }
    schedule = build_schedule(curves, alpha=0.1, num_steps=10, lookahead=3)["d_mlp"]
    assert schedule == [1, 0, 1, 0, 1, 0, 1, 0, 1, 1]


def test_largest_lag_has_priority():
    # lag 3 passes at step 0 -> skip 3 steps even though lag 1 also passes.
    curves = {"x": [[0.0] * 10, [0.0] * 10, [0.0] * 10]}
    schedule = build_schedule(curves, alpha=0.5, num_steps=10, lookahead=3)["x"]
    assert schedule[0] == 1
    assert schedule[1:4] == [0, 0, 0]
    assert schedule[4] == 1


def test_first_and_last_steps_forced_compute():
    curves = {"x": [[0.0] * 50, [0.0] * 50, [0.0] * 50]}
    schedule = build_schedule(curves, alpha=0.5, num_steps=50, lookahead=3)["x"]
    assert schedule[0] == 1
    assert schedule[-1] == 1


def test_short_rows_fall_back_to_compute():
    curves = {"x": [[0.0] * 10, [0.0] * 10, [0.0] * 10]}
    schedule = build_schedule(curves, alpha=0.5, num_steps=20, lookahead=3)["x"]
    assert schedule[:12] == [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]
    assert all(v == 1 for v in schedule[12:])


def test_missing_component_never_caches():
    schedule = build_schedule({}, alpha=0.5, num_steps=50, lookahead=3)
    assert schedule == {}


def test_skip_stats():
    schedule = {"a": [1, 0, 0, 1], "b": [1, 1, 1, 1]}
    stats = schedule_skip_stats(schedule)
    assert stats["a"]["skip"] == 2
    assert stats["a"]["skip_fraction"] == 0.5
    assert stats["b"]["skip"] == 0


def test_mac_reduction_bounds():
    curves = {
        "d_attn": [[0.2] * 50, [0.2] * 50, [0.2] * 50],
        "s_attn_mlp": [[0.2] * 50, [0.2] * 50, [0.2] * 50],
    }
    macs = {"d_attn": 100.0, "s_attn_mlp": 50.0}
    zero = schedule_mac_reduction(
        build_schedule(curves, alpha=0.0, num_steps=50, lookahead=3),
        macs,
        num_double_blocks=1,
        num_single_blocks=1,
    )
    aggressive = schedule_mac_reduction(
        build_schedule(curves, alpha=1.0, num_steps=50, lookahead=3),
        macs,
        num_double_blocks=1,
        num_single_blocks=1,
    )
    assert zero["reduction_fraction"] == 0.0
    assert 0.0 < aggressive["reduction_fraction"] < 1.0
    assert aggressive["reduction_fraction"] > zero["reduction_fraction"]


def test_estimate_component_macs_shapes():
    macs = estimate_component_macs(
        hidden_size=3072, mlp_hidden=12288, img_tokens=1000, txt_tokens=256
    )
    assert set(macs) == {"d_attn", "d_img_mlp", "d_txt_mlp", "s_attn_mlp"}
    assert macs["d_img_mlp"] > 0 and macs["d_txt_mlp"] > 0


def test_rel_l1_loss_plain_sequences():
    assert _approx(0.5)(rel_l1_loss([1.0, 1.0], [2.0, 2.0]))
    assert rel_l1_loss([1.0], [1.0]) == 0.0
    # Tuple output: average of |1-3|/|3| and |2-2|/|2| = (2/3 + 0) / 2.
    assert _approx(1.0 / 3.0)(rel_l1_loss(([1.0], [2.0]), ([3.0], [2.0])))


def test_rel_l1_loss_torch():
    try:
        import torch
    except ImportError:
        return  # torch unavailable in this environment
    prev = torch.tensor([1.0, 2.0, 3.0])
    cur = torch.tensor([2.0, 2.0, 3.0])
    assert _approx(1.0 / 7.0)(rel_l1_loss(prev, cur))
