"""SmoothCache adaptation for HunyuanVideo.

Training-free, error-guided layer caching for the HunyuanVideo DiT backbone,
ported from the official SmoothCache implementation (Roblox, CVPR 2025
workshop, https://github.com/Roblox/SmoothCache).

The package contains:

- config.py      : serialisable experiment configuration.
- schedule.py    : error-curve -> static 0/1 schedule generation and stats.
- calibration.py : per-step layer-output recorder used during a full
                   (non-cached) calibration generation pass.
- runtime.py     : runtime controller that reuses cached layer outputs
                   according to a static schedule.
"""

from .config import SmoothCacheConfig
from .schedule import (
    build_schedule,
    estimate_component_macs,
    load_schedule,
    rel_l1_loss,
    save_schedule,
    schedule_mac_reduction,
    schedule_skip_stats,
)
from .calibration import CalibrationCollector, SmoothCacheCalibrator, plot_error_curves
from .runtime import SmoothCacheNoop, SmoothCacheRuntime

__all__ = [
    "SmoothCacheConfig",
    "SmoothCacheCalibrator",
    "CalibrationCollector",
    "SmoothCacheRuntime",
    "SmoothCacheNoop",
    "build_schedule",
    "schedule_mac_reduction",
    "schedule_skip_stats",
    "estimate_component_macs",
    "rel_l1_loss",
    "load_schedule",
    "save_schedule",
    "plot_error_curves",
]

COMPONENTS = ("d_attn", "d_img_mlp", "d_txt_mlp", "s_attn_mlp")
