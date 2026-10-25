"""Batch video generation for HunyuanVideo with MixCache enabled.

Usage (on the server, from the repo root):

    python scripts/run_mixcache_batch.py \
        --model-base ckpts \
        --prompt-json prompts/mixcache_50_prompts.json \
        --save-path results/mixcache_effi \
        --mixcache-log-dir results/mixcache_effi/logs

The script generates one 544x960 / 129-frame / 50-step video per prompt with
seed 42, writes a per-video JSON log (per-step cache mode + timing) and a
summary CSV.  See MIXCACHE.md for details.
"""

import csv
import json
import os
import sys
import time
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

from loguru import logger  # noqa: E402

from hyvideo.config import parse_args  # noqa: E402
from hyvideo.inference import HunyuanVideoSampler  # noqa: E402
from hyvideo.utils.file_utils import save_videos_grid  # noqa: E402


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


def _safe_prompt(prompt: str) -> str:
    """Sanitize a prompt for use in a file name."""
    return prompt.strip().replace("/", "_").replace("\\", "_")


def main():
    args = parse_args()
    if not args.mixcache:
        logger.warning("--mixcache is not set; enabling MixCache for batch generation.")
        args.mixcache = True
    if args.prompt_json is None:
        raise ValueError("--prompt-json is required for batch generation.")

    with open(args.prompt_json, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    prompts = _normalize_prompts(prompts)
    logger.info(f"Loaded {len(prompts)} prompts from {args.prompt_json}")

    # Default: official single-forward path (embedded guidance only), matching
    # the released cfgdistill weights and the baseline run.sh timing.  Only
    # step / block-level caching applies in this mode.  To enable explicit CFG
    # (which also activates cfg-level caching), pass --cfg-scale 6.0.
    args.embedded_cfg_scale = 6.0
    # Resolution / frame count / steps come from the CLI
    # (--video-size --video-length --infer-steps).
    args.video_size = list(args.video_size)
    args.video_length = int(args.video_length)
    args.infer_steps = int(args.infer_steps)
    args.seed = SEED
    args.batch_size = 1

    save_path = Path(args.save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.mixcache_log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    sampler = HunyuanVideoSampler.from_pretrained(Path(args.model_base), args=args)
    args = sampler.args

    summary_rows = []
    for local_idx, prompt in enumerate(prompts):
        idx = local_idx + args.index_offset
        logger.info(
            f"[{idx + 1}/{len(prompts) + args.index_offset}] Generating: {prompt[:80]}"
        )
        total_t0 = time.perf_counter()
        outputs = sampler.predict(
            prompt=prompt,
            height=args.video_size[0],
            width=args.video_size[1],
            video_length=args.video_length,
            seed=SEED,
            infer_steps=args.infer_steps,
            guidance_scale=args.cfg_scale,
            flow_shift=args.flow_shift,
            embedded_guidance_scale=args.embedded_cfg_scale,
            batch_size=1,
        )
        total_seconds = time.perf_counter() - total_t0

        sample = outputs["samples"][0].unsqueeze(0)
        seed_used = int(outputs["seeds"][0])
        video_name = f"{_safe_prompt(prompt)}-0.mp4"
        save_videos_grid(sample, str(save_path / video_name), fps=24)

        mixcache_log = outputs.get("mixcache_log") or {}
        mixcache_log["index"] = idx
        mixcache_log["prompt"] = prompt
        mixcache_log["seed"] = seed_used
        mixcache_log["total_seconds"] = total_seconds
        per_video_log = log_dir / f"{idx:03d}.json"
        per_video_log.write_text(
            json.dumps(mixcache_log, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            f"Saved {video_name} ({total_seconds:.1f}s total, "
            f"{mixcache_log.get('denoise_seconds', 0):.1f}s denoise)"
        )

        mode_counts = mixcache_log.get("mode_counts") or {}
        summary_rows.append(
            {
                "index": idx,
                "seed": seed_used,
                "total_seconds": round(total_seconds, 3),
                "denoise_seconds": round(mixcache_log.get("denoise_seconds") or 0.0, 3),
                "warmup_steps": mixcache_log.get("warmup_steps"),
                "count_full": mode_counts.get("full", 0),
                "count_step": mode_counts.get("step", 0),
                "count_cfg": mode_counts.get("cfg", 0),
                "count_block": sum(
                    v for k, v in mode_counts.items() if k.startswith("block_")
                ),
            }
        )

    if args.index_offset > 0:
        summary_name = f"summary_{args.index_offset:03d}.csv"
    else:
        summary_name = "summary.csv"
    summary_path = log_dir / summary_name
    if summary_rows:
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        logger.info(f"Summary written to {summary_path}")

    logger.info(f"Done. Videos: {save_path.resolve()}, logs: {log_dir.resolve()}")


if __name__ == "__main__":
    main()
