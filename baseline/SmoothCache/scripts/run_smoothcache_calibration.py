"""SmoothCache calibration for HunyuanVideo.

Runs full-computation generations on a set of calibration prompts, records the
per-step relative-L1 error curves of every cacheable layer type, averages them
across prompts, and writes:

    <calibration-output>   averaged error curves JSON
    <calibration-plot>     error-curve figure (matplotlib, optional)
    alpha scan table       stdout: skip fraction / estimated MAC reduction
                           for a range of alpha values

Usage (on the server, from the repo root):

    python scripts/run_smoothcache_calibration.py \
        --model-base ckpts \
        --prompt-json prompts/smoothcache_calibration_10_prompts.json \
        --calibration-output smoothcache_calibration.json \
        --calibration-plot smoothcache_calibration.png \
        --alpha-scan 0.02,0.04,0.06,0.08,0.12,0.16,0.20

Calibration videos are generated with the same sampling configuration as the
acceptance run (544x960 / 129 frames / 50 steps / official embedded CFG).
"""

import json
import os
import sys
import time
from pathlib import Path

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _infer_model_base_env():
    """Derive MODEL_BASE from --model-base before hyvideo is imported."""
    if os.environ.get("MODEL_BASE"):
        return
    argv = sys.argv[1:]
    try:
        idx = argv.index("--model-base")
        os.environ["MODEL_BASE"] = argv[idx + 1]
    except (ValueError, IndexError):
        pass


_infer_model_base_env()

from loguru import logger  # noqa: E402

from hyvideo.config import parse_args  # noqa: E402
from hyvideo.inference import HunyuanVideoSampler  # noqa: E402
from hyvideo.smoothcache import (  # noqa: E402
    CalibrationCollector,
    SmoothCacheCalibrator,
    build_schedule,
    estimate_component_macs,
    plot_error_curves,
    schedule_mac_reduction,
    schedule_skip_stats,
)


VIDEO_HEIGHT = 544
VIDEO_WIDTH = 960
VIDEO_LENGTH = 129
INFER_STEPS = 50
SEED = 42


def _normalize_prompts(data):
    """Accept a JSON array of strings or of objects with prompt/prompt_en."""
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


def _alpha_list(raw: str):
    return [float(x) for x in raw.split(",") if x.strip()]


def main():
    args = parse_args()
    if args.prompt_json is None:
        raise ValueError("--prompt-json is required for calibration.")

    with open(args.prompt_json, "r", encoding="utf-8") as f:
        prompts = _normalize_prompts(json.load(f))
    logger.info(f"Loaded {len(prompts)} calibration prompts from {args.prompt_json}")

    # Force the acceptance sampling configuration (full-computation reference).
    args.cfg_scale = 1.0
    args.embedded_cfg_scale = 6.0
    args.flow_shift = 7.0
    args.flow_reverse = True
    calib_h, calib_w = _calibration_size(args)
    args.video_size = [calib_h, calib_w]
    args.video_length = args.calibration_video_length
    args.infer_steps = INFER_STEPS
    args.seed = SEED
    args.batch_size = 1

    sampler = HunyuanVideoSampler.from_pretrained(Path(args.model_base), args=args)
    args = sampler.args

    collector = CalibrationCollector(lookahead=args.smoothcache_lookahead)
    for idx, prompt in enumerate(prompts):
        calibrator = SmoothCacheCalibrator(lookahead=args.smoothcache_lookahead)
        t0 = time.perf_counter()
        logger.info(f"[{idx + 1}/{len(prompts)}] Calibration generation: {prompt[:80]}")
        sampler.predict(
            prompt=prompt,
            height=calib_h,
            width=calib_w,
            video_length=args.calibration_video_length,
            seed=SEED,
            infer_steps=INFER_STEPS,
            guidance_scale=args.cfg_scale,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_cfg_scale,
            batch_size=1,
            smooth_cache=calibrator,
        )
        import torch

        torch.cuda.empty_cache()
        collector.add_video(calibrator.finish_video())
        logger.info(
            f"Calibration video {idx + 1} done in {time.perf_counter() - t0:.1f}s"
        )

    curves = collector.average_curves()
    out_path = Path(args.calibration_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(curves, f, ensure_ascii=False, indent=2)
    logger.info(f"Averaged error curves written to {out_path}")

    if args.calibration_plot:
        plot_path = plot_error_curves(
            curves, args.calibration_plot, title=f"SmoothCache calibration ({len(prompts)} videos)"
        )
        if plot_path:
            logger.info(f"Error-curve plot written to {plot_path}")
        else:
            logger.warning("matplotlib unavailable; skipping error-curve plot.")

    # ---- offline alpha sweep ----------------------------------------------
    num_double = len(sampler.model.double_blocks)
    num_single = len(sampler.model.single_blocks)
    img_tokens, txt_tokens = _token_counts(sampler)
    component_macs = estimate_component_macs(
        hidden_size=sampler.model.hidden_size,
        mlp_hidden=int(sampler.model.hidden_size * sampler.model.mlp_width_ratio),
        img_tokens=img_tokens,
        txt_tokens=txt_tokens,
    )
    logger.info("Alpha sweep (skip fraction per component):")
    for alpha in _alpha_list(args.alpha_scan):
        schedule = build_schedule(
            curves, alpha, num_steps=INFER_STEPS, lookahead=args.smoothcache_lookahead
        )
        stats = schedule_skip_stats(schedule)
        reduction = schedule_mac_reduction(
            schedule, component_macs, num_double_blocks=num_double, num_single_blocks=num_single
        )
        summary = {a: round(stats[a]["skip_fraction"], 3) for a in stats}
        logger.info(
            f"  alpha={alpha:.3f}: skip={summary} "
            f"mac_reduction={reduction['reduction_fraction']:.3f}"
        )


def _token_counts(sampler):
    """Approximate image/text token counts for the MAC estimate."""
    vae = sampler.args.vae
    if "884" in vae:
        t_latent = (VIDEO_LENGTH - 1) // 4 + 1
    elif "888" in vae:
        t_latent = (VIDEO_LENGTH - 1) // 8 + 1
    else:
        t_latent = VIDEO_LENGTH
    h_latent, w_latent = VIDEO_HEIGHT // 8, VIDEO_WIDTH // 8
    patch_size = sampler.model.patch_size
    if isinstance(patch_size, int):
        pt = ph = pw = patch_size
    else:
        pt, ph, pw = patch_size
    img_tokens = (t_latent // pt) * (h_latent // ph) * (w_latent // pw)
    txt_tokens = int(getattr(sampler.args, "text_len", 256))
    return img_tokens, txt_tokens


def _calibration_size(args):
    """Resolve the low-resolution calibration video size from CLI args."""
    size = list(args.calibration_video_size)
    if len(size) == 1:
        size = [size[0], size[0]]
    if len(size) != 2:
        raise ValueError(
            "--calibration-video-size must be 'height width' (one or two ints)."
        )
    return int(size[0]), int(size[1])


if __name__ == "__main__":
    main()
