"""Calibration recorder and collector for SmoothCache.

The calibrator is handed to the model like the runtime controller: blocks call
``branch(component, block_id, compute_fn)`` and the calibrator records the
pre-residual layer outputs, computes relative-L1 errors at every cache gap
(1..lookahead) and accumulates them per timestep. A :class:`CalibrationCollector`
aggregates the rows across calibration videos, which is what
:func:`hyvideo.smoothcache.schedule.build_schedule` consumes.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .schedule import rel_l1_loss


def _detached(output):
    """Return a detached copy of a tensor (or tuple of tensors)."""
    if isinstance(output, (tuple, list)):
        return tuple(_detached(o) for o in output)
    if hasattr(output, "detach"):
        return output.detach().clone() if hasattr(output, "clone") else output.detach()
    return output


class SmoothCacheCalibrator:
    """Per-video recorder. Create one instance per generated calibration video."""

    def __init__(self, lookahead: int = 3):
        if lookahead <= 0:
            raise ValueError("lookahead must be positive.")
        self.lookahead = lookahead
        self.step = 0
        # (component, block_id) -> rolling list of past outputs (0 = 1 step ago).
        self._prev_outputs: Dict[Tuple[str, int], List] = {}
        # (component, block_id, lag) -> per-block error values per timestep.
        self._errors: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
        self._components = set()
        self._block_ids: Dict[str, set] = defaultdict(set)
        self._reset()

    def _reset(self) -> None:
        self._prev_outputs.clear()
        self._errors.clear()
        self._components.clear()
        self._block_ids.clear()

    def begin_step(self, step_index: int) -> None:
        """Called by the pipeline at the start of denoising step ``step_index``."""
        self.step = int(step_index)

    def branch(self, component: str, block_id: int, compute_fn):
        """Run ``compute_fn``, record its output and the lag errors.

        The returned value is the untouched computed output so the model
        forward proceeds exactly as without caching.
        """
        output = compute_fn()
        key = (component, int(block_id))
        self._components.add(component)
        self._block_ids[component].add(int(block_id))
        prevs = self._prev_outputs.get(key)
        if prevs is None:
            prevs = []
            self._prev_outputs[key] = prevs

        # prevs[lag] holds the output from ``lag + 1`` timesteps ago.
        for lag in range(self.lookahead):
            if lag < len(prevs) and prevs[lag] is not None:
                error = rel_l1_loss(prevs[lag], output)
                self._errors[(component, int(block_id), lag)].append(error)

        prevs.insert(0, _detached(output))
        if len(prevs) > self.lookahead:
            prevs.pop()
        return output

    def end_step(self) -> dict:
        """Pipeline hook. Calibration does not record per-step compute/skip
        modes (the run is always full-computation), so this is a no-op."""
        return {}

    def finish_run(self) -> None:
        """Release the rolling per-block output buffers after the denoising
        loop. The per-step lag errors are already accumulated in ``_errors``,
        so dropping the past outputs frees tens of GB before VAE decode."""
        self._prev_outputs.clear()

    def per_component_errors(self) -> Dict[str, Dict[int, List[List[float]]]]:
        """Return ``component -> {block_id: rows}`` with per-block error rows.

        ``rows[lag]`` holds the errors for that block, and the error at
        ``rows[lag][i]`` compares the layer output at timestep ``i + lag + 1``
        with the output at timestep ``i``. Keeping blocks separate is
        required so the collector can align the per-timestep averages across
        blocks and videos.
        """
        out: Dict[str, Dict[int, List[List[float]]]] = {
            component: {
                block_id: [[] for _ in range(self.lookahead)]
                for block_id in self._block_ids.get(component, set())
            }
            for component in self._components
        }
        for (component, block_id, lag), values in self._errors.items():
            out[component][block_id][lag] = list(values)
        return out

    def finish_video(self) -> Dict[str, Dict[int, List[List[float]]]]:
        """Return and clear the per-component error rows for this video."""
        rows = self.per_component_errors()
        self._reset()
        return rows


class CalibrationCollector:
    """Aggregates per-timestep error rows across calibration videos/blocks."""

    def __init__(self, lookahead: int = 3):
        self.lookahead = lookahead
        # (component, lag, index) -> [sum, count]
        self._acc: Dict[Tuple[str, int, int], List[float]] = defaultdict(
            lambda: [0.0, 0.0]
        )
        self._num_videos = 0

    def add_video(self, rows: Dict[str, Dict[int, List[List[float]]]]) -> None:
        """Merge one calibration video's error rows."""
        for component, lag_rows in rows.items():
            for block_id, block_rows in lag_rows.items():
                for lag, values in enumerate(block_rows):
                    for idx, value in enumerate(values):
                        acc = self._acc[(component, lag, idx)]
                        acc[0] += float(value)
                        acc[1] += 1.0
        self._num_videos += 1

    @property
    def num_videos(self) -> int:
        return self._num_videos

    def average_curves(self) -> Dict[str, List[List[float]]]:
        """Return ``component -> rows`` averaged over blocks and videos."""
        # Accumulate per (component, lag) index-aligned averages.
        per_component: Dict[str, List[Dict[int, float]]] = defaultdict(lambda: [
            {} for _ in range(self.lookahead)
        ])
        for (component, lag, idx), (total, count) in sorted(self._acc.items()):
            per_component[component][lag][idx] = total / count if count else 0.0

        return {
            component: [
                [value for _, value in sorted(rows[lag].items())]
                for lag in range(self.lookahead)
            ]
            for component, rows in per_component.items()
        }

    def save_curves(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.average_curves(), f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_curves(path: str) -> Dict[str, List[List[float]]]:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)


def plot_error_curves(
    curves: Dict[str, List[List[float]]],
    out_path: str,
    title: str = "SmoothCache calibration error curves",
) -> Optional[str]:
    """Plot per-component error rows (paper Fig.2 style). matplotlib optional."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - environment without matplotlib
        return None

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    for ax, (component, rows) in zip(axes, curves.items()):
        for lag, values in enumerate(rows):
            if values:
                ax.plot(values, label=f"k={lag + 1}")
        ax.set_title(component)
        ax.set_xlabel("timestep")
        ax.set_ylabel("rel L1 error")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
