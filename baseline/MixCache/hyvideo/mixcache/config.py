"""Configuration for the MixCache cache controller.

The defaults follow the MixCache paper (arXiv:2508.12691) for HunyuanVideo
wherever the paper publishes concrete numbers:

    theta  = 0.1, delta1 = 0.05, delta2 = 0.1
    cache interval schedules: acc = (2, 3, 4), effi = (3, 4, 5)
    penalty coefficient        = 5

The per-granularity *impact* values I^step / I^cfg / I^block are not published
as numbers in the paper (only as curves).  The defaults below are heuristic
values consistent with the paper's observations (cfg impact is roughly one
order of magnitude larger than step/block impact, and block impact grows
towards the end of the diffusion process).  Every value can be overridden with
a JSON config file (see MIXCACHE.md for the schema).
"""

import json
from dataclasses import dataclass, field
from typing import List, Optional, Union


DEFAULT_BLOCK_INDICES = [10, 30, 50]


@dataclass
class MixCacheConfig:
    theta: float = 0.1
    delta1: float = 0.05
    delta2: float = 0.1
    schedule: Union[str, List[int]] = "effi"  # "acc" -> (2, 3, 4), "effi" -> (3, 4, 5)
    penalty: float = 5.0
    block_indices: List[int] = field(default_factory=lambda: list(DEFAULT_BLOCK_INDICES))
    enable_cfg_cache: bool = True
    # Heuristic impact values (see module docstring).
    impact_step: float = 0.1
    impact_cfg: float = 1.0
    impact_block_base: float = 0.05
    impact_block_scale: float = 0.45
    # Small preference for earlier blocks; 0 disables the index term.
    impact_block_index_weight: float = 0.2
    # Profiling mode performs full computation on every step while recording
    # the redundancy traces that the offline profiling script needs.
    profiling: bool = False

    @property
    def n_schedule(self):
        if self.schedule == "acc":
            return (2, 3, 4)
        if self.schedule == "effi":
            return (3, 4, 5)
        if isinstance(self.schedule, (list, tuple)) and len(self.schedule) == 3:
            values = self.schedule
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in values):
                raise ValueError(
                    f"schedule must be a list of 3 integers, got {self.schedule!r}"
                )
            n = (int(values[0]), int(values[1]), int(values[2]))
            if not (1 <= n[0] < n[1] < n[2]):
                raise ValueError(
                    f"schedule must be 3 strictly increasing integers >= 1, got {list(values)!r}"
                )
            return n
        raise ValueError(
            f"Unknown MixCache schedule: {self.schedule!r} "
            "(expected 'acc', 'effi', or a list of 3 strictly increasing integers)"
        )

    @property
    def n_initial(self):
        # Middle value of the schedule (accuracy/effciency trade-off).
        return self.n_schedule[1]

    def impact_block(self, block_index: int, progress: float) -> float:
        """Heuristic time- and index-dependent block-level impact value.

        progress is the denoising progress in [0, 1] (0 = first step).  Earlier
        blocks are slightly preferred (lower impact), and impact grows as the
        diffusion process advances.
        """
        base = self.impact_block_base + self.impact_block_scale * progress
        index_factor = 1.0 - self.impact_block_index_weight * (1.0 - block_index / 60.0)
        return max(base * index_factor, 1e-6)

    @classmethod
    def from_args(cls, args) -> "MixCacheConfig":
        cfg = cls()
        if getattr(args, "mixcache_mode", None):
            cfg.schedule = args.mixcache_mode
        if getattr(args, "mixcache_block_indices", None):
            cfg.block_indices = [int(i) for i in args.mixcache_block_indices]
        if getattr(args, "mixcache_profiling", False):
            cfg.profiling = True
        config_path = getattr(args, "mixcache_config", None)
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for key, value in data.items():
                if not hasattr(cfg, key):
                    raise ValueError(f"Unknown MixCache config key: {key}")
                setattr(cfg, key, value)
        return cfg

    def to_dict(self) -> dict:
        return {
            "theta": self.theta,
            "delta1": self.delta1,
            "delta2": self.delta2,
            "schedule": self.schedule,
            "penalty": self.penalty,
            "block_indices": list(self.block_indices),
            "enable_cfg_cache": self.enable_cfg_cache,
            "impact_step": self.impact_step,
            "impact_cfg": self.impact_cfg,
            "impact_block_base": self.impact_block_base,
            "impact_block_scale": self.impact_block_scale,
            "impact_block_index_weight": self.impact_block_index_weight,
        }
