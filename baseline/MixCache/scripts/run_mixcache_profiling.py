"""Optional offline profiling for MixCache hyper-parameters.

Runs full (uncached) generation on a subset of prompts while recording the
per-step redundancy distances D^step / D^cfg / D^block, then writes a JSON
config that can be passed to the runtime via --mixcache-config.

Usage (on the server, from the repo root):

    python scripts/run_mixcache_profiling.py \
        --model-base ckpts \
        --prompt-json prompts/mixcache_50_prompts.json \
        --profiling-prompts 10 \
        --profiling-output mixcache_profile.json
"""

import json
import os
import sys
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _infer_model_base_env():
    """Derive MODEL_BASE from --model-base before hyvideo is imported, so the
    VAE / text-encoder paths (resolved from MODEL_BASE at import time) point
    at the same checkpoint root."""
    if os.environ.get("MODEL_BASE"):
        return
    argv = sys.argv[1:]
    try:
        idx = argv.index("--model-base")
        os.environ["MODEL_BASE"] = argv[idx + 1]
    except (ValueError, IndexError):
        pass


_infer_model_base_env()

import numpy as np  # noqa: E402
from loguru import logger  # noqa: E402

from hyvideo.config import parse_args  # noqa: E402
from hyvideo.inference import HunyuanVideoSampler  # noqa: E402


VIDEO_HEIGHT = 544
VIDEO_WIDTH = 960
VIDEO_LENGTH = 129
INFER_STEPS = 50
SEED = 42


def _normalize_prompts(data):
    """Accept a JSON array of strings or an array of objects carrying a
    'prompt' / 'prompt_en' field (e.g. VBench-style info files)."""
    if not isinstance(data, list) or not data:
        raise ValueError("--prompt-json must contain a non-empty JSON array.")
    if all(isinstance(p, str) for p in data):
        return [p for p in data if p.strip()]
    if all(isinstance(p, dict) and ("prompt" in p or "prompt_en" in p) for p in data):
        return [
            (p.get("prompt") or p.get("prompt_en") or "").strip()
            for p in data
            if (p.get("prompt") or p.get("prompt_en") or "").strip()
        ]
    raise ValueError(
        "--prompt-json must be a JSON array of strings, or an array of objects "
        "with a 'prompt' / 'prompt_en' field."
    )


def _means(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def main():
    args = parse_args()
    if args.prompt_json is None:
        raise ValueError("--prompt-json is required for profiling.")
    args.mixcache = True
    args.mixcache_profiling = True
    # Default is the official single-forward path (embedded guidance only);
    # pass --cfg-scale 6.0 to profile with explicit CFG instead.
    args.embedded_cfg_scale = 6.0
    args.video_size = [VIDEO_HEIGHT, VIDEO_WIDTH]
    args.video_length = VIDEO_LENGTH
    args.infer_steps = INFER_STEPS
    args.seed = SEED
    args.batch_size = 1

    with open(args.prompt_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompts = _normalize_prompts(prompts)
    prompts = prompts[: max(1, int(args.profiling_prompts))]
    logger.info(f"Profiling on {len(prompts)} prompts (full computation only).")

    sampler = HunyuanVideoSampler.from_pretrained(Path(args.model_base), args=args)

    traces = []
    for idx, prompt in enumerate(prompts):
        logger.info(f"[{idx + 1}/{len(prompts)}] Profiling: {prompt[:80]}")
        outputs = sampler.predict(
            prompt=prompt,
            height=VIDEO_HEIGHT,
            width=VIDEO_WIDTH,
            video_length=VIDEO_LENGTH,
            seed=SEED,
            infer_steps=INFER_STEPS,
            guidance_scale=args.cfg_scale,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_cfg_scale,
            batch_size=1,
        )
        mixcache_log = outputs.get("mixcache_log") or {}
        traces.extend(mixcache_log.get("profile_trace") or [])
        logger.info(f"  trace steps: {len(mixcache_log.get('profile_trace') or [])}")

    if not traces:
        raise RuntimeError("No profile traces collected - check --mixcache-profiling.")

    d_step = [t.get("d_step") for t in traces]
    d_cfg = [t.get("d_cfg") for t in traces]
    d_step_clean = [v for v in d_step if v is not None]

    # theta: median D^step over the stable (later) part of the trace.
    stable = d_step_clean[-max(1, int(len(d_step_clean) * 0.4)) :]
    theta = float(np.median(stable)) if stable else 0.1
    delta1 = theta * 0.5
    delta2 = theta

    # Impact values: the measured D of a granularity *is* the per-step error
    # caused by applying that cache, so its average is a natural estimate of I.
    impact_step = _means(d_step) or 0.1
    impact_cfg = _means(d_cfg) or 1.0

    # Block-level impact is time-dependent: use the first/last third of the
    # trace to fit the heuristic base/scale used by MixCacheConfig.
    block_traces = {}
    for t in traces:
        for key, value in (t.get("d_block") or {}).items():
            block_traces.setdefault(key, []).append(value)
    impact_block_base = 0.05
    impact_block_scale = 0.45
    if block_traces:
        early, late = [], []
        for key, values in block_traces.items():
            values = [v for v in values if v is not None]
            if not values:
                continue
            third = max(1, len(values) // 3)
            early.append(float(np.mean(values[:third])))
            late.append(float(np.mean(values[-third:])))
        if early and late:
            impact_block_base = min(early)
            impact_block_scale = max(late) - min(early)

    profile = {
        "theta": round(theta, 6),
        "delta1": round(delta1, 6),
        "delta2": round(delta2, 6),
        "impact_step": round(impact_step, 6),
        "impact_cfg": round(impact_cfg, 6),
        "impact_block_base": round(impact_block_base, 6),
        "impact_block_scale": round(impact_block_scale, 6),
    }

    output_path = Path(args.profiling_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info(f"Profiling config written to {output_path.resolve()}")
    logger.info(json.dumps(profile, indent=2))


if __name__ == "__main__":
    main()
