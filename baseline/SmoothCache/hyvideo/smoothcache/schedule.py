"""Static schedule generation for SmoothCache.

This is a faithful port of the schedule construction from the official
SmoothCache repository (``CalibrationHelper._scan_nrows_sublists``): error
rows are scanned in reverse priority order (largest cache gap first) and a
timestep is computed (1) whenever the representation error against the
previous output is below ``alpha``, otherwise the following steps reuse the
cached output (0).
"""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Sequence, Tuple


def rel_l1_loss(prev_output, cur_output) -> float:
    """Relative L1 error ``||prev - cur||_1 / ||cur||_1`` (official port).

    Accepts tensors or 1-D numeric sequences. Tensors are computed in float32
    and reduced over all elements. Tuples of tensors (e.g. the fused
    image+text attention branch) are averaged element-wise.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - torch always present at runtime
        torch = None

    def _one(prev, cur) -> float:
        if torch is not None and isinstance(prev, torch.Tensor):
            prev = prev.detach().float()
            cur = cur.detach().float()
            numerator = float(torch.norm(prev - cur, p=1))
            denominator = float(torch.norm(cur, p=1))
        else:
            # Plain numeric sequences (unit tests / CPU debug fallback).
            flat_prev = [float(v) for v in prev]
            flat_cur = [float(v) for v in cur]
            if len(flat_prev) != len(flat_cur):
                raise ValueError("Outputs must have the same number of elements.")
            numerator = sum(abs(a - b) for a, b in zip(flat_prev, flat_cur))
            denominator = sum(abs(b) for b in flat_cur)
        return numerator / denominator if denominator else 0.0

    def _is_tuple_of_outputs(x) -> bool:
        """True for (tensor, tensor) tuples or nested sequences of outputs."""
        if not isinstance(x, (tuple, list)) or len(x) == 0:
            return False
        first = x[0]
        if torch is not None and isinstance(first, torch.Tensor):
            return True
        return isinstance(first, (tuple, list))

    if _is_tuple_of_outputs(prev_output):
        if len(prev_output) != len(cur_output):
            raise ValueError("Tuple outputs must have matching length.")
        values = [_one(p, c) for p, c in zip(prev_output, cur_output)]
        return sum(values) / len(values)
    return _one(prev_output, cur_output)


def build_schedule(
    error_curves: Dict[str, List[List[float]]],
    alpha: float,
    num_steps: int,
    lookahead: int = 3,
) -> Dict[str, List[int]]:
    """Turn averaged calibration error rows into a static 0/1 schedule.

    Args:
        error_curves: mapping ``component -> rows`` where ``rows[lag]`` is the
            per-timestep average error at gap ``lag + 1`` (row 0 = gap 1,
            row 2 = gap 3 for the default lookahead of 3).
        alpha: error threshold. Lower alpha caches less.
        num_steps: length of the returned schedule (number of denoising steps).
        lookahead: number of cached-gap rows considered (k = 1..lookahead).

    Returns:
        Mapping ``component -> list[int]`` with ``1`` = compute, ``0`` = reuse
        the cached output. The first and last timesteps are always computed.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if lookahead <= 0:
        raise ValueError("lookahead must be positive.")

    schedules: Dict[str, List[int]] = {}
    for component, rows in error_curves.items():
        if not rows:
            # No calibration data for this component: never cache.
            schedules[component] = [1] * num_steps
            continue
        schedules[component] = _scan_rows(rows, alpha, num_steps)
    return schedules


def _scan_rows(rows: Sequence[Sequence[float]], alpha: float, num_steps: int) -> List[int]:
    """Official ``_scan_nrows_sublists`` logic for one component."""
    schedule: List[Optional[int]] = [None] * num_steps
    i = 0
    while i < num_steps:
        used = False
        # Highest-priority rows (largest cache gap) are scanned first.
        for row_idx in range(len(rows) - 1, -1, -1):
            row = rows[row_idx]
            if i >= len(row):
                continue
            if row[i] <= alpha:
                schedule[i] = 1
                num_skips = row_idx + 1
                for s in range(1, num_skips + 1):
                    if i + s < num_steps:
                        schedule[i + s] = 0
                i += num_skips + 1
                used = True
                break
        if not used:
            schedule[i] = 1
            i += 1

    if num_steps > 0:
        schedule[0] = 1
        schedule[-1] = 1
    return [1 if x is None else x for x in schedule]


def schedule_skip_stats(
    schedule: Dict[str, List[int]],
) -> Dict[str, Dict[str, float]]:
    """Per-component skip statistics (counts and fractions)."""
    stats: Dict[str, Dict[str, float]] = {}
    for component, values in schedule.items():
        total = len(values)
        skips = sum(1 for v in values if v == 0)
        stats[component] = {
            "steps": total,
            "compute": total - skips,
            "skip": skips,
            "skip_fraction": skips / total if total else 0.0,
        }
    return stats


def estimate_component_macs(
    hidden_size: int,
    mlp_hidden: int,
    img_tokens: int,
    txt_tokens: int,
) -> Dict[str, float]:
    """Approximate per-step MACs of each cacheable component.

    Weights are parameter-count based (linear layers dominate DiT compute);
    attention is approximated by ``2 * seq^2 * hidden`` per fused attention
    call. This mirrors the "theoretical compute" accounting used by the
    MixCache reproduction and is deliberately approximate.
    """
    h = hidden_size
    m = mlp_hidden
    seq = img_tokens + txt_tokens
    weights = {
        # qkv (2 streams) + 2 projections, times tokens, plus fused attention.
        "d_attn": (8.0 * h * h) * seq + 2.0 * seq * seq * h,
        "d_img_mlp": (2.0 * h * m) * img_tokens,
        "d_txt_mlp": (2.0 * h * m) * txt_tokens,
        # linear1 (3h + m) + linear2 (h + m) over the fused sequence.
        "s_attn_mlp": (4.0 * h * h + 2.0 * h * m) * seq + 2.0 * seq * seq * h,
    }
    return weights


def schedule_mac_reduction(
    schedule: Dict[str, List[int]],
    component_macs: Dict[str, float],
    num_double_blocks: int,
    num_single_blocks: int,
) -> Dict[str, float]:
    """Estimate the transformer-level MAC reduction of a schedule.

    ``component_macs`` holds per-block MACs for one block of each type
    (d_* for double blocks, s_* for single blocks). The returned dictionary
    contains absolute/computed MAC estimates and the reduction fraction.
    """
    per_step_compute = {c: 0.0 for c in component_macs}
    per_step_total = {c: 0.0 for c in component_macs}
    for component, values in schedule.items():
        if component not in component_macs:
            continue
        blocks = (
            num_double_blocks
            if component.startswith("d_")
            else num_single_blocks
            if component.startswith("s_")
            else 1
        )
        per_step_total[component] = component_macs[component] * blocks * len(values)
        per_step_compute[component] = (
            component_macs[component]
            * blocks
            * sum(1 for v in values if v == 1)
        )
    total = sum(per_step_total.values())
    computed = sum(per_step_compute.values())
    return {
        "total_macs": total,
        "computed_macs": computed,
        "saved_macs": total - computed,
        "reduction_fraction": (total - computed) / total if total else 0.0,
        "per_component": {
            c: {
                "total_macs": per_step_total[c],
                "computed_macs": per_step_compute[c],
                "reduction_fraction": (
                    (per_step_total[c] - per_step_compute[c]) / per_step_total[c]
                    if per_step_total[c]
                    else 0.0
                ),
            }
            for c in per_step_total
        },
    }


def save_schedule(schedule: Dict[str, List[int]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(schedule, f, ensure_ascii=False, indent=2)


def load_schedule(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
