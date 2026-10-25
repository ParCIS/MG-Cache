"""Runtime controllers for SmoothCache.

The runtime receives the static 0/1 schedule and is handed to the model the
same way as the calibrator: blocks call ``branch(component, block_id,
compute_fn)``. On a skipped timestep the runtime returns the last computed
output of that block's layer, otherwise it computes and stores the new output.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple


def _detached(output):
    """Detach a tensor (or tuple of tensors) before it is cached."""
    if isinstance(output, (tuple, list)):
        return tuple(_detached(o) for o in output)
    if hasattr(output, "detach"):
        return output.detach()
    return output


class SmoothCacheRuntime:
    """Applies a static SmoothCache schedule during one generation run."""

    def __init__(
        self,
        schedule: Dict[str, List[int]],
        components: Optional[List[str]] = None,
        log_steps: bool = True,
    ):
        self.schedule = schedule
        self.components = list(components) if components else list(schedule.keys())
        self.log_steps = log_steps
        self.cache: Dict[Tuple[str, int], object] = {}
        self.step = 0
        self.step_modes: List[Dict[str, int]] = []
        self._current_modes: Dict[str, int] = {}

    def reset(self) -> None:
        self.cache.clear()
        self.step = 0
        self.step_modes = []
        self._current_modes = {}

    def finish_run(self) -> None:
        """Release cached layer outputs after the denoising loop.

        The cache is only needed during denoising; dropping it before the VAE
        decode frees tens of GB on high-resolution runs.
        """
        self.cache.clear()

    def begin_step(self, step_index: int) -> None:
        self.step = int(step_index)
        self._current_modes = {}

    def should_compute(self, component: str) -> bool:
        values = self.schedule.get(component)
        if not values:
            return True
        if self.step >= len(values):
            return True
        return values[self.step] != 0

    def branch(self, component: str, block_id: int, compute_fn: Callable):
        """Return the cached output on a skipped step, else compute + cache."""
        if not self.should_compute(component):
            key = (component, int(block_id))
            if key in self.cache:
                self._current_modes[component] = 0
                return self.cache[key]
        output = compute_fn()
        self.cache[(component, int(block_id))] = _detached(output)
        self._current_modes[component] = 1
        return output

    def end_step(self) -> Dict[str, int]:
        """Record the current step's per-component compute/skip modes."""
        if self.log_steps:
            self.step_modes.append(
                {"step": self.step, **{c: self._current_modes.get(c, 1) for c in self.components}}
            )
        return self._current_modes

    def mode_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for entry in self.step_modes:
            for component, mode in entry.items():
                if component == "step":
                    continue
                counts[component] = counts.get(component, 0) + (1 if mode == 0 else 0)
        return counts


class SmoothCacheNoop:
    """No-op controller used to run the plain (non-cached) reference path."""

    def begin_step(self, step_index: int) -> None:
        pass

    def finish_run(self) -> None:
        pass

    def branch(self, component: str, block_id: int, compute_fn: Callable):
        return compute_fn()

    def end_step(self) -> Dict[str, int]:
        return {}
