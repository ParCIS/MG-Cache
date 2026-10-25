"""Logic tests for the MixCache controller (no torch / GPU required).

The controller only uses a small subset of torch tensor operations, which this
test emulates with a minimal stub so the control flow (warm-up triggering,
N scaling, hybrid mode decision, penalty, logging) can be verified anywhere,
including machines without torch installed.

Run from the repo root:
    python tests/test_mixcache_controller.py
"""

import itertools
import json
import os
import sys
import types


current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)


class T:
    """Minimal tensor stub: supports detach/float/abs/sum/sub/add/mul/div."""

    def __init__(self, val, batch=1):
        self.val = float(val)
        self.batch = batch

    def detach(self):
        return self

    def float(self):
        return self

    def abs(self):
        return T(abs(self.val), self.batch)

    def sum(self):
        return T(self.val, self.batch)

    def item(self):
        return self.val

    def __int__(self):
        return int(self.val)

    def __index__(self):
        return int(self.val)

    def __float__(self):
        return self.val

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            return T(self.val - other, self.batch)
        return T(self.val - other.val, self.batch)

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return T(self.val + other, self.batch)
        return T(self.val + other.val, self.batch)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return T(self.val * other, self.batch)
        return T(self.val * other.val, self.batch)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return T(self.val / other, self.batch)
        return T(self.val / other.val, self.batch)

    def __getitem__(self, item):
        # Simulate cond-slice [1:] and full-slice [:].
        return T(self.val, self.batch)

    def __repr__(self):
        return f"T({self.val}, b={self.batch})"


torch_stub = types.ModuleType("torch")
torch_stub.Tensor = T
sys.modules["torch"] = torch_stub

from hyvideo.mixcache.config import MixCacheConfig
from hyvideo.mixcache.controller import MixCacheController


def make_outputs(step, batch=2):
    """Noise outputs whose consecutive difference shrinks over time."""
    value = 1.0 - step * 0.012
    # uncond and cond are close (cfg redundancy is the strongest), step-level
    # drift is moderate, block-level drift is the largest.
    return T(value, batch), T(value * 1.002, batch), T(value, batch)


def make_block_outputs(step, batch=2):
    return {
        10: (T(0.9 - step * 0.008, batch), T(0.8 - step * 0.008, batch)),
        30: T(0.7 - step * 0.008, batch),
        50: T(0.6 - step * 0.008, batch),
    }


def run(do_cfg=True, schedule="effi", profiling=False):
    cfg = MixCacheConfig(schedule=schedule, profiling=profiling)
    ctl = MixCacheController(cfg, do_cfg=do_cfg, num_blocks=60, num_double_blocks=20)
    ctl.begin(50)

    for i in range(50):
        ctl.step_begin(i, T(1000 - i * 20))
        mode = ctl.mode
        if mode == "full":
            final, uncond, cond = make_outputs(i)
            if do_cfg:
                ctl.update_full(
                    final,
                    uncond_raw=uncond,
                    cond_raw=cond,
                    block_outputs=make_block_outputs(i),
                )
            else:
                ctl.update_full(final, block_outputs=make_block_outputs(i))
        else:
            if mode.startswith("block_"):
                assert ctl.get_block_cache(mode) is not None
            ctl.update_after_cache(mode, block_outputs=make_block_outputs(i))

    result = ctl.result()
    assert result["denoise_seconds"] is not None
    modes = [e["mode"] for e in result["steps"]]
    assert len(modes) == 50
    assert modes[0] == "full", "first step must be full"
    assert any(m != "full" for m in modes), "no cache steps triggered"
    assert result["cache_enabled_from_step"] is not None
    assert result["cache_enabled_from_step"] < 20, "warm-up too long"
    json.dumps(result)  # must be JSON serializable
    assert sum(result["mode_counts"].values()) == 50
    return result, modes


def test_schedules_and_cfg():
    for schedule in ("acc", "effi"):
        result, modes = run(do_cfg=True, schedule=schedule)
        n_values = [e["N"] for e in result["steps"] if e["N"] is not None]
        assert all(2 <= n <= 5 for n in n_values)
        assert any(m == "cfg" for m in modes), "cfg cache never selected with CFG on"
        assert any(m.startswith("block_") for m in modes), "block cache never selected"

    result, modes = run(do_cfg=False, schedule="effi")
    assert "cfg" not in modes, "cfg cache must be disabled without CFG"
    assert any(m.startswith("block_") for m in modes)
    assert any(m == "step" for m in modes)


def test_penalty_changes_decision():
    cfg = MixCacheConfig(schedule="effi")
    ctl = MixCacheController(cfg, do_cfg=True, num_blocks=60, num_double_blocks=20)
    ctl.begin(50)
    ctl.cache_enabled = True
    ctl.d_step = 0.5
    ctl.d_cfg = 0.2

    ctl.last_mode = None
    assert ctl._decide_mode() == "step", "0.5*0.1=0.05 must beat 0.2*1.0=0.2"
    ctl.last_mode = "step"
    assert ctl._decide_mode() == "cfg", "penalty 5x must flip step -> cfg"
    ctl.last_mode = "cfg"
    assert ctl._decide_mode() == "step", "penalty 5x must flip cfg -> step"


def test_runs_bounded_by_interval():
    result, modes = run(do_cfg=True, schedule="effi")
    run_len = max(len(list(g)) for _, g in itertools.groupby(modes))
    max_n = max(e["N"] for e in result["steps"])
    assert run_len <= max_n, f"identical-mode run {run_len} exceeds interval {max_n}"


def test_profiling_always_full():
    cfg = MixCacheConfig(profiling=True)
    ctl = MixCacheController(cfg, do_cfg=True, num_blocks=60, num_double_blocks=20)
    ctl.begin(10)
    for i in range(10):
        ctl.step_begin(i, T(1000 - i * 20))
        assert ctl.mode == "full"
        final, uncond, cond = make_outputs(i)
        ctl.update_full(
            final,
            uncond_raw=uncond,
            cond_raw=cond,
            block_outputs=make_block_outputs(i),
        )
    result = ctl.result()
    assert len(result["profile_trace"]) == 10


if __name__ == "__main__":
    test_schedules_and_cfg()
    test_penalty_changes_decision()
    test_runs_bounded_by_interval()
    test_profiling_always_full()
    print("mixcache logic tests passed")
