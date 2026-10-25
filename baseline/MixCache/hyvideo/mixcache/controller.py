"""Runtime cache controller for MixCache.

Implements the three pieces of the MixCache paper (arXiv:2508.12691):

1. context-aware cache triggering (warm-up until D^step < theta);
2. adaptive cache interval N scaling based on the relative L1 distance between
   consecutive full computations (delta1 / delta2);
3. adaptive hybrid cache decision: for every cache-enabled step choose the
   granularity (step / cfg / block_i) that minimizes P = D * I, with a penalty
   applied to the granularity used at the previous step.

The controller itself is framework-agnostic: it stores cached tensors and
produces decisions; the pipeline drives it.
"""

import time
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().item())
    return float(value)


def _rel_l1(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> float:
    a = a.detach().float()
    b = b.detach().float()
    return float((a - b).abs().sum() / (b.abs().sum() + eps))


def _rel_l1_multi(pairs: List[Tuple[torch.Tensor, torch.Tensor]], eps: float = 1e-8) -> float:
    num = 0.0
    den = 0.0
    for a, b in pairs:
        a = a.detach().float()
        b = b.detach().float()
        num += float((a - b).abs().sum())
        den += float(b.abs().sum()) + eps
    return num / den


class MixCacheController:
    """State machine that decides, for every denoising step, whether to run a
    full transformer forward or to reuse cached computation at step / cfg /
    block granularity."""

    def __init__(self, config, do_cfg: bool, num_blocks: int, num_double_blocks: int):
        self.cfg = config
        self.do_cfg = bool(do_cfg and config.enable_cfg_cache)
        self.num_blocks = int(num_blocks)
        self.num_double_blocks = int(num_double_blocks)

        # Runtime state (reset by begin()).
        self.cache_enabled = False
        self.N = config.n_initial
        self.cnt = 0
        self.mode = "full"
        self.last_mode: Optional[str] = None
        self.step_idx = 0
        self.timestep = 0
        self.total_steps = 1

        # Cached tensors.
        self.cached_output: Optional[torch.Tensor] = None  # final (CFG-combined) noise pred
        self.cached_text_output: Optional[torch.Tensor] = None  # conditional noise pred
        self.delta_cfg: Optional[torch.Tensor] = None      # uncond - cond from last full step
        self.block_outputs_cache: Dict[int, object] = {}   # flattened block idx -> cond slice

        # Redundancy / impact metrics.
        self.d_step: Optional[float] = None
        self.d_cfg: Optional[float] = None
        self.d_full: Optional[float] = None
        self.d_block: Dict[int, float] = {}

        # Logging / profiling state.
        self.steps_log: List[dict] = []
        self.profile_trace: List[dict] = []
        self._pending: Optional[dict] = None
        self._t0 = 0.0
        self.denoise_seconds: Optional[float] = None
        self.mode_counts: Dict[str, int] = {}
        self.cache_enabled_from_step: Optional[int] = None

    # ------------------------------------------------------------------ #
    # Public API used by the pipeline.
    # ------------------------------------------------------------------ #

    @property
    def return_block_indices(self) -> List[int]:
        return list(self.cfg.block_indices)

    def begin(self, total_steps: int):
        self.total_steps = max(1, int(total_steps))
        self.cache_enabled = False
        self.N = self.cfg.n_initial
        self.cnt = 0
        self.mode = "full"
        self.last_mode = None
        self.cached_output = None
        self.cached_text_output = None
        self.delta_cfg = None
        self.block_outputs_cache = {}
        self.d_step = None
        self.d_cfg = None
        self.d_full = None
        self.d_block = {}
        self.steps_log = []
        self.profile_trace = []
        self._pending = None
        self.cache_enabled_from_step = None
        self._t0 = time.perf_counter()

    def step_begin(self, step_idx: int, timestep):
        """Decide what to do on this step.  Must be called before the
        transformer forward of every denoising step."""
        self.step_idx = int(step_idx)
        self.timestep = int(timestep)

        if self.cfg.profiling:
            # Profiling mode always performs full computation so that the
            # redundancy traces cover every denoising step.
            self.mode = "full"
            self._open_log()
            return

        if not self.cache_enabled:
            # Context-aware triggering: leave warm-up once consecutive full
            # computations are similar enough (D^step < theta).
            if self.d_step is not None and self.d_step < self.cfg.theta:
                self.cache_enabled = True
                self.cache_enabled_from_step = self.step_idx
                self.N = self.cfg.n_initial
                self.cnt = 0
            else:
                self.mode = "full"
                self._open_log()
                return

        # Cache-enabled phase: full computation every N steps.
        self.cnt = (self.cnt + 1) % self.N
        if self.cnt == 0:
            self.mode = "full"
        else:
            self.mode = self._decide_mode()
            if self.mode.startswith("block_") and self.get_block_cache(self.mode) is None:
                # The required prefix is not cached (safety fallback).
                self.mode = "cfg" if self.do_cfg else "full"
        self._open_log()

    def update_full(
        self,
        final_noise_pred: torch.Tensor,
        uncond_raw: Optional[torch.Tensor] = None,
        cond_raw: Optional[torch.Tensor] = None,
        block_outputs: Optional[Dict[int, object]] = None,
    ):
        """Called after a full transformer forward (warm-up or interval
        boundary).  Updates redundancy metrics, cached tensors and N."""
        prev_output = self.cached_output
        self.cached_output = final_noise_pred.detach()

        if prev_output is not None:
            self.d_step = _rel_l1(final_noise_pred, prev_output)
            if self.cache_enabled:
                self.d_full = self.d_step
                self._scale_n()

        if self.do_cfg and uncond_raw is not None and cond_raw is not None:
            self.d_cfg = _rel_l1(uncond_raw, cond_raw)
            self.delta_cfg = (uncond_raw - cond_raw).detach()
            self.cached_text_output = cond_raw.detach()

        if block_outputs is not None:
            for idx, output in block_outputs.items():
                idx = int(idx)
                cond_out = self._cond_slice(output)
                prev_out = self.block_outputs_cache.get(idx)
                if prev_out is not None:
                    self.d_block[idx] = _rel_l1_multi(self._to_pairs(cond_out, prev_out))
                self.block_outputs_cache[idx] = cond_out

        if self.cfg.profiling:
            self.profile_trace.append(
                {
                    "step": self.step_idx,
                    "timestep": self.timestep,
                    "d_step": _to_float(self.d_step),
                    "d_cfg": _to_float(self.d_cfg),
                    "d_block": {str(i): _to_float(v) for i, v in self.d_block.items()},
                }
            )

        # A full computation refreshes every cache; no granularity was reused
        # this step, so the decision penalty is cleared.
        self.last_mode = None
        self._close_log()

    def get_block_cache(self, mode: str) -> Optional[dict]:
        """Return the cached prefix needed by the transformer for a
        block_<i> cache step, or None if unavailable."""
        if not mode.startswith("block_"):
            return None
        block_index = int(mode.split("_", 1)[1])
        cached = self.block_outputs_cache.get(block_index)
        if cached is None:
            return None
        if block_index < self.num_double_blocks:
            # Cache point inside the double-block range.
            img, txt = cached
            return {"block_index": block_index, "img": img, "txt": txt, "x": None}
        return {"block_index": block_index, "img": None, "txt": None, "x": cached}

    def update_after_cache(self, mode: str, block_outputs: Optional[Dict[int, object]] = None):
        """Called after a cache-mode step (step / cfg / block_i)."""
        self.last_mode = mode
        if block_outputs is not None:
            for idx, output in block_outputs.items():
                idx = int(idx)
                if idx in self.cfg.block_indices:
                    self.block_outputs_cache[idx] = self._cond_slice(output)
        self.mode = mode
        self._close_log()

    def finish(self):
        self.denoise_seconds = time.perf_counter() - self._t0
        self.mode_counts = dict(Counter(entry["mode"] for entry in self.steps_log))

    def result(self) -> dict:
        self.finish()
        return {
            "config": self.cfg.to_dict(),
            "denoise_seconds": self.denoise_seconds,
            "cache_enabled_from_step": self.cache_enabled_from_step,
            "warmup_steps": self.cache_enabled_from_step if self.cache_enabled_from_step is not None else self.total_steps,
            "steps": self.steps_log,
            "mode_counts": self.mode_counts,
            "profile_trace": self.profile_trace if self.cfg.profiling else None,
        }

    # ------------------------------------------------------------------ #
    # Internals.
    # ------------------------------------------------------------------ #

    def _decide_mode(self) -> str:
        progress = self.step_idx / max(1, self.total_steps - 1)
        candidates = [("step", self.d_step)]
        if self.do_cfg:
            candidates.append(("cfg", self.d_cfg))
        for block_index in self.cfg.block_indices:
            candidates.append((f"block_{block_index}", self.d_block.get(block_index)))

        best_mode = "step"
        best_p = None
        for mode, distance in candidates:
            if distance is None:
                continue
            p = distance * self._impact(mode, progress)
            if mode == self.last_mode:
                p *= self.cfg.penalty
            if best_p is None or p < best_p:
                best_p = p
                best_mode = mode
        return best_mode

    def _impact(self, mode: str, progress: float) -> float:
        if mode == "step":
            return self.cfg.impact_step
        if mode == "cfg":
            return self.cfg.impact_cfg
        block_index = int(mode.split("_", 1)[1])
        return self.cfg.impact_block(block_index, progress)

    def _scale_n(self):
        n_min, _, n_max = self.cfg.n_schedule
        if self.d_full is None:
            return
        if self.d_full < self.cfg.delta1:
            self.N = min(n_max, self.N + 1)
        elif self.d_full >= self.cfg.delta2:
            self.N = max(n_min, self.N - 1)

    def _cond_slice(self, output):
        if isinstance(output, tuple):
            if self.do_cfg:
                return tuple(o[1:].detach() for o in output)
            return tuple(o.detach() for o in output)
        if self.do_cfg:
            return output[1:].detach()
        return output.detach()

    @staticmethod
    def _to_pairs(current, previous):
        if isinstance(current, tuple):
            return list(zip(current, previous))
        return [(current, previous)]

    def _open_log(self):
        self._pending = {
            "step": self.step_idx,
            "timestep": self.timestep,
            "mode": self.mode,
            "block_index": (
                int(self.mode.split("_", 1)[1]) if self.mode.startswith("block_") else None
            ),
            "d_step": _to_float(self.d_step),
            "d_cfg": _to_float(self.d_cfg),
            "d_block": {str(i): _to_float(v) for i, v in self.d_block.items()},
            "N": self.N,
            "full_computation": self.mode == "full",
        }

    def _close_log(self):
        if self._pending is not None:
            self.steps_log.append(self._pending)
            self._pending = None
