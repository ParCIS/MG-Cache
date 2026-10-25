"""Unit tests for the SmoothCache runtime controller and calibrator.

These tests do not require torch: fake outputs are plain iterables with an
optional ``detach``, and the relative-L1 helper falls back to plain sums.
"""

from hyvideo.smoothcache.calibration import CalibrationCollector, SmoothCacheCalibrator
from hyvideo.smoothcache.runtime import SmoothCacheRuntime


def _approx(value, eps=1e-9):
    def check(actual):
        if isinstance(value, (list, tuple)):
            assert len(actual) == len(value)
            for a, b in zip(actual, value):
                assert abs(a - b) <= eps, (a, b)
            return True
        return abs(actual - value) <= eps

    return check


class FakeOutput:
    def __init__(self, value):
        self.value = value

    def detach(self):
        return self

    def clone(self):
        return self

    def __iter__(self):
        return iter([self.value])

    def __repr__(self):
        return f"FakeOutput({self.value})"


def _runtime(schedule):
    return SmoothCacheRuntime(schedule)


def test_runtime_reuses_cached_output_on_skip():
    calls = []

    def compute():
        calls.append(1)
        return FakeOutput(len(calls))

    rt = _runtime({"d_attn": [1, 0, 1]})
    rt.begin_step(0)
    first = rt.branch("d_attn", 0, compute)
    rt.end_step()
    rt.begin_step(1)
    second = rt.branch("d_attn", 0, compute)
    rt.end_step()
    rt.begin_step(2)
    third = rt.branch("d_attn", 0, compute)
    rt.end_step()

    assert first.value == 1
    assert second is first  # cached object returned untouched
    assert third.value == 2
    assert len(calls) == 2


def test_runtime_per_block_cache_is_independent():
    computed = {"a": 0, "b": 0}

    def make(key):
        def compute():
            computed[key] += 1
            return FakeOutput(computed[key])

        return compute

    rt = _runtime({"d_attn": [1, 0]})
    rt.begin_step(0)
    a0 = rt.branch("d_attn", 0, make("a"))
    b0 = rt.branch("d_attn", 1, make("b"))
    rt.end_step()
    rt.begin_step(1)
    a1 = rt.branch("d_attn", 0, make("a"))
    b1 = rt.branch("d_attn", 1, make("b"))
    rt.end_step()

    assert a1 is a0 and b1 is b0
    assert a1.value == 1 and b1.value == 1


def test_runtime_missing_component_always_computes():
    calls = []
    rt = _runtime({"d_attn": [1, 0]})
    for step in range(2):
        rt.begin_step(step)
        rt.branch("s_attn_mlp", 0, lambda: calls.append(step) or FakeOutput(step))
        rt.end_step()
    assert len(calls) == 2


def test_runtime_step_modes_and_counts():
    rt = _runtime({"d_attn": [1, 0, 1], "d_img_mlp": [1, 1, 0]})
    for step in range(3):
        rt.begin_step(step)
        rt.branch("d_attn", 0, lambda: FakeOutput(step))
        rt.branch("d_img_mlp", 0, lambda: FakeOutput(step))
        rt.end_step()
    assert len(rt.step_modes) == 3
    assert rt.step_modes[1]["d_attn"] == 0
    assert rt.step_modes[2]["d_img_mlp"] == 0
    assert rt.mode_counts() == {"d_attn": 1, "d_img_mlp": 1}


def test_calibrator_records_lag_errors():
    cal = SmoothCacheCalibrator(lookahead=3)
    for step, value in enumerate([1.0, 2.0, 3.0, 4.0]):
        cal.begin_step(step)
        cal.branch("d_attn", 0, lambda v=value: [v])
    rows = cal.per_component_errors()["d_attn"][0]
    # k=1 errors: |1-2|/2, |2-3|/3, |3-4|/4
    assert _approx([0.5, 1.0 / 3.0, 0.25])(rows[0])
    # k=2 errors: |1-3|/3, |2-4|/4
    assert _approx([2.0 / 3.0, 0.5])(rows[1])
    # k=3 errors: |1-4|/4
    assert _approx([0.75])(rows[2])


def test_collector_averages_across_videos():
    collector = CalibrationCollector(lookahead=1)
    collector.add_video({"d_attn": {0: [[0.2, 0.4]], 1: [[0.3, 0.5]]}})
    collector.add_video({"d_attn": {0: [[0.4, 0.6]], 1: [[0.5, 0.7]]}})
    curves = collector.average_curves()
    # Averaged over blocks and videos per timestep index.
    assert _approx([0.35, 0.55])(curves["d_attn"][0])


def test_calibrator_finish_video_resets_state():
    cal = SmoothCacheCalibrator(lookahead=1)
    cal.begin_step(0)
    cal.branch("d_attn", 0, lambda: [1.0])
    rows1 = cal.finish_video()
    rows2 = cal.per_component_errors()
    assert rows1["d_attn"] == {0: [[]]}
    assert rows2 == {}


def test_calibrator_pipeline_interface():
    """The calibrator must expose the same hook surface the pipeline calls."""
    cal = SmoothCacheCalibrator(lookahead=3)
    cal.begin_step(0)
    assert cal.end_step() == {}
    assert callable(cal.branch)
    assert callable(cal.finish_run)
    cal.finish_run()


def test_runtime_finish_run_releases_cache():
    rt = SmoothCacheRuntime({"d_attn": [1, 0]})
    rt.begin_step(0)
    first = rt.branch("d_attn", 0, lambda: FakeOutput(1))
    rt.end_step()
    assert len(rt.cache) == 1
    rt.finish_run()
    assert rt.cache == {}
    # Cached value no longer available -> next step recomputes.
    rt.begin_step(1)
    second = rt.branch("d_attn", 0, lambda: FakeOutput(2))
    rt.end_step()
    assert second.value == 2
