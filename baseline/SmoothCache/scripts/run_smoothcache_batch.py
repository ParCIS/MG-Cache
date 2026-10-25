"""Batch video generation for HunyuanVideo with SmoothCache.

Modes:

* no-cache (no --smoothcache): official reference generation (used for the
  equivalence gate and as a plain baseline).
* cached: pass --smoothcache with either
    --smoothcache-schedule <schedule.json>          (static 0/1 schedule), or
    --smoothcache-calibration-json <curves.json>
    --smoothcache-alpha <alpha>                     (build from calibration).

Each prompt produces one 544x960 / 129-frame / 50-step video with seed 42,
a per-video JSON log (per-step per-component compute/skip modes + timing) and
a summary CSV. Existing outputs are skipped so interrupted runs can resume.

Usage (on the server, from the repo root):

    python scripts/run_smoothcache_batch.py \
        --model-base ckpts \
        --prompt-json VBench_small_info.json \
        --smoothcache \
        --smoothcache-calibration-json smoothcache_calibration.json \
        --smoothcache-alpha 0.08 \
        --save-path results/smoothcache_alpha008 \
        --smoothcache-log-dir results/smoothcache_alpha008/logs
"""

import csv
import json
import os
import re
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
    SmoothCacheRuntime,
    build_schedule,
    estimate_component_macs,
    load_schedule,
    schedule_mac_reduction,
    schedule_skip_stats,
)
from hyvideo.utils.file_utils import save_videos_grid  # noqa: E402


VIDEO_HEIGHT = 544
VIDEO_WIDTH = 960
VIDEO_LENGTH = 129
INFER_STEPS = 50
SEED = 42


def _sanitize_filename(text, max_len=100):
    """Make a prompt safe for use as a file name.

    Only path-unsafe characters are replaced; spaces inside the prompt are
    preserved (e.g. "In a still frame" -> "In a still frame").
    """
    text = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", text)
    return text.strip()[:max_len]


def _video_name(args, idx, prompt):
    if getattr(args, "video_name_format", "prompt") == "index":
        return f"{idx:03d}_seed{SEED}.mp4"
    return f"{_sanitize_filename(prompt)}-0.mp4"


def _batch_video_size(args):
    """Resolve batch video size (height width) from CLI args."""
    size = [int(v) for v in args.batch_video_size]
    if len(size) == 1:
        size = [size[0], size[0]]
    if len(size) != 2:
        raise ValueError("--batch-video-size must be 'height width' (one or two ints).")
    return size


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


def _resolve_schedule(args):
    """Return (schedule, source_description) from CLI arguments."""
    if args.smoothcache_schedule:
        schedule = load_schedule(args.smoothcache_schedule)
        return schedule, f"schedule={args.smoothcache_schedule}"
    if args.smoothcache_calibration_json:
        with open(args.smoothcache_calibration_json, "r", encoding="utf-8") as f:
            curves = json.load(f)
        schedule = build_schedule(
            curves,
            alpha=args.smoothcache_alpha,
            num_steps=args.infer_steps,
            lookahead=args.smoothcache_lookahead,
        )
        return schedule, (
            f"calibration={args.smoothcache_calibration_json} "
            f"alpha={args.smoothcache_alpha}"
        )
    raise ValueError(
        "--smoothcache requires --smoothcache-schedule or "
        "--smoothcache-calibration-json (with --smoothcache-alpha)."
    )


def main():
    args = parse_args()
    if args.prompt_json is None:
        raise ValueError("--prompt-json is required for batch generation.")

    with open(args.prompt_json, "r", encoding="utf-8") as f:
        prompts = _normalize_prompts(json.load(f))
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_json}")

    args.cfg_scale = 1.0  # official embedded-CFG single-forward path
    args.embedded_cfg_scale = 6.0
    args.flow_shift = 7.0
    args.flow_reverse = True
    batch_h, batch_w = _batch_video_size(args)
    args.video_size = [batch_h, batch_w]
    args.video_length = args.batch_video_length
    args.infer_steps = INFER_STEPS
    args.seed = SEED
    args.batch_size = 1

    schedule = None
    schedule_source = "no-cache (official reference)"
    if args.smoothcache:
        schedule, schedule_source = _resolve_schedule(args)
        logger.info(f"SmoothCache enabled: {schedule_source}")
        logger.info(
            "Skip fractions: "
            + json.dumps(schedule_skip_stats(schedule), ensure_ascii=False)
        )

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.smoothcache_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sampler = HunyuanVideoSampler.from_pretrained(Path(args.model_base), args=args)
    args = sampler.args

    if schedule is not None:
        component_macs = estimate_component_macs(
            hidden_size=sampler.model.hidden_size,
            mlp_hidden=int(sampler.model.hidden_size * sampler.model.mlp_width_ratio),
            img_tokens=_img_tokens(sampler, batch_h, batch_w, args.batch_video_length),
            txt_tokens=int(getattr(sampler.args, "text_len", 256)),
        )
        mac_reduction = schedule_mac_reduction(
            schedule,
            component_macs,
            num_double_blocks=len(sampler.model.double_blocks),
            num_single_blocks=len(sampler.model.single_blocks),
        )
        logger.info(
            f"Estimated transformer MAC reduction: "
            f"{mac_reduction['reduction_fraction']:.3f}"
        )
    else:
        mac_reduction = None

    summary_rows = []
    for local_idx, prompt in enumerate(prompts):
        idx = local_idx + args.index_offset
        video_name = _video_name(args, idx, prompt)
        video_path = save_path / video_name
        if video_path.exists():
            logger.info(f"Skip existing {video_path.name}")
            continue

        logger.info(f"[{idx + 1}/{len(prompts) + args.index_offset}] Generating: {prompt[:80]}")
        runtime = SmoothCacheRuntime(schedule) if schedule is not None else None
        total_t0 = time.perf_counter()
        outputs = sampler.predict(
            prompt=prompt,
            height=batch_h,
            width=batch_w,
            video_length=args.batch_video_length,
            seed=SEED,
            infer_steps=INFER_STEPS,
            guidance_scale=args.cfg_scale,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_cfg_scale,
            batch_size=1,
            smooth_cache=runtime,
            offload_text_encoder=args.offload_text_encoder,
        )
        total_seconds = time.perf_counter() - total_t0

        sample = outputs["samples"][0].unsqueeze(0)
        save_videos_grid(sample, str(video_path), fps=24)

        smooth_cache_log = outputs.get("smooth_cache_log") or {}
        per_video_log = {
            "index": idx,
            "prompt": prompt,
            "seed": int(outputs["seeds"][0]),
            "schedule_source": schedule_source,
            "total_seconds": round(total_seconds, 3),
            "denoise_seconds": round(smooth_cache_log.get("denoise_seconds") or 0.0, 3),
            "mode_counts": smooth_cache_log.get("mode_counts") or {},
            "step_modes": smooth_cache_log.get("step_modes") or [],
        }
        log_path = log_dir / f"{idx:03d}.json"
        log_path.write_text(
            json.dumps(per_video_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            f"Saved {video_name} ({total_seconds:.1f}s total, "
            f"{per_video_log['denoise_seconds']:.1f}s denoise)"
        )

        mode_counts = per_video_log["mode_counts"]
        row = {
            "index": idx,
            "seed": per_video_log["seed"],
            "total_seconds": round(total_seconds, 3),
            "denoise_seconds": per_video_log["denoise_seconds"],
        }
        for component in ("d_attn", "d_img_mlp", "d_txt_mlp", "s_attn_mlp"):
            row[f"skip_{component}"] = mode_counts.get(component, 0)
        row["mac_reduction_fraction"] = (
            round(mac_reduction["reduction_fraction"], 4) if mac_reduction else ""
        )
        summary_rows.append(row)

    if args.index_offset > 0:
        summary_name = f"summary_{args.index_offset:03d}.csv"
    else:
        summary_name = "summary.csv"
    if summary_rows:
        summary_path = log_dir / summary_name
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary written to {summary_path}")

    logger.info(f"Done. Videos: {save_path.resolve()}, logs: {log_dir.resolve()}")


def _img_tokens(sampler, height, width, video_length):
    vae = sampler.args.vae
    if "884" in vae:
        t_latent = (video_length - 1) // 4 + 1
    elif "888" in vae:
        t_latent = (video_length - 1) // 8 + 1
    else:
        t_latent = video_length
    h_latent, w_latent = height // 8, width // 8
    patch_size = sampler.model.patch_size
    if isinstance(patch_size, int):
        pt = ph = pw = patch_size
    else:
        pt, ph, pw = patch_size
    return (t_latent // pt) * (h_latent // ph) * (w_latent // pw)


if __name__ == "__main__":
    main()
