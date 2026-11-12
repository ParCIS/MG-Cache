import json
import math
import os
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

import imageio.v2 as imageio
import numpy as np
import psutil
import torch

import comfy.model_management as model_management
import folder_paths
from comfy.utils import ProgressBar

from .metrics import compare_frames


STEPS = 50
FPS = 24
LAYERS = 60
VIDEO_TEXT_TOKENS = 256
IMAGE_TEXT_TOKENS = 315
HIDDEN_SIZE = 3072
EMBEDDED_GUIDANCE = 6.0
FLOW_SHIFT = 7.0
HYUNYUAN_ROOT = Path(__file__).resolve().parents[1] / "MG-Cache" / "HunyuanVideo"


def validate_settings(width, height, frames, cfg_scale, alpha, beta):
    if width < 16 or height < 16 or width % 16 or height % 16:
        raise ValueError("Width and height must be positive multiples of 16.")
    if frames < 1 or (frames - 1) % 4:
        raise ValueError("Frame count must be 4n + 1 (1, 5, 9, …).")
    if not math.isfinite(cfg_scale) or cfg_scale < 1:
        raise ValueError("CFG scale must be finite and at least 1.")
    if any(not math.isfinite(v) or v < 0 for v in (alpha, beta)):
        raise ValueError("Alpha and beta must be finite and nonnegative.")

    text_tokens = IMAGE_TEXT_TOKENS if frames == 1 else VIDEO_TEXT_TOKENS
    tokens = ((frames - 1) // 4 + 1) * (width // 16) * (height // 16) + text_tokens
    cfg_batch = 2 if cfg_scale > 1 else 1
    return cfg_batch * tokens * HIDDEN_SIZE * LAYERS * 2 * 2


@lru_cache(maxsize=1)
def get_sampler():
    checkpoint_dir = Path(os.environ.get("MODEL_BASE", "")).expanduser()
    if not checkpoint_dir.is_absolute() or not checkpoint_dir.is_dir():
        raise RuntimeError("Set MODEL_BASE to the absolute HunyuanVideo checkpoint directory before starting ComfyUI.")
    if not HYUNYUAN_ROOT.is_dir():
        raise RuntimeError(f"Missing MG-Cache HunyuanVideo source at {HYUNYUAN_ROOT}.")

    if str(HYUNYUAN_ROOT) not in sys.path:
        sys.path.insert(0, str(HYUNYUAN_ROOT))
    from hyvideo.config import parse_args
    from hyvideo.inference import HunyuanVideoSampler

    weight = checkpoint_dir / "hunyuan-video-t2v-720p" / "transformers" / "mp_rank_00_model_states.pt"
    args = parse_args(argv=[
        "--model-base", str(checkpoint_dir), "--dit-weight", str(weight),
        "--flow-reverse", "--vae-tiling",
    ])
    model_management.unload_all_models()
    model_management.soft_empty_cache()
    return HunyuanVideoSampler.from_pretrained(checkpoint_dir, args, device=torch.device("cuda"))


def decoded_frames(result):
    samples = result["samples"][0]
    if samples.ndim == 3:
        samples = samples.unsqueeze(1)
    return samples.permute(1, 2, 3, 0).numpy()


def write_video(path, frames):
    with imageio.get_writer(str(path), fps=FPS, codec="libx264", macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(np.rint(frame * 255).astype(np.uint8))


class HunyuanVideoMGCache:
    CATEGORY = "MG-Cache"
    DESCRIPTION = "Generate with HunyuanVideo in 50 denoising steps. Optionally run a full-compute baseline on the same GPU."
    FUNCTION = "generate"
    RETURN_TYPES = ()
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Used when cfg_scale > 1. Blank uses HunyuanVideo's default negative prompt."}),
            "seed": ("INT", {"default": 42, "min": 0, "max": 2**53 - 1, "control_after_generate": True}),
            "width": ("INT", {"default": 640, "min": 16, "max": 2048, "step": 16}),
            "height": ("INT", {"default": 480, "min": 16, "max": 2048, "step": 16}),
            "frames": ("INT", {"default": 65, "min": 1, "max": 257, "step": 4}),
            "cfg_scale": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 10.0, "step": 0.1}),
            "alpha": ("FLOAT", {"default": 0.03, "min": 0.0, "max": 1.0, "step": 0.001}),
            "beta": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
            "compare": ("BOOLEAN", {"default": False, "label_on": "启用对照", "label_off": "仅 MG-Cache"}),
        }}

    @classmethod
    def IS_CHANGED(cls, **inputs):
        # Each execution measures fresh timings, including repeated identical prompts.
        return float("nan")

    def generate(self, prompt, negative_prompt, seed, width, height, frames, cfg_scale, alpha, beta, compare):
        cache_bytes = validate_settings(width, height, frames, cfg_scale, alpha, beta)
        available = psutil.virtual_memory().available
        if cache_bytes >= available:
            raise ValueError(f"Layer cache requires {cache_bytes / 2**30:.2f} GiB of pinned RAM. "
                             f"Available host RAM is {available / 2**30:.2f} GiB.")
        if not torch.cuda.is_available():
            raise RuntimeError("The HunyuanVideo node requires a CUDA GPU.")

        sampler = get_sampler()
        progress = ProgressBar(STEPS * (2 if compare else 1))

        def on_step(_pipeline, _index, _timestep, _tensors):
            model_management.throw_exception_if_processing_interrupted()
            progress.update(1)
            return {}

        sampling = {
            "prompt": prompt, "negative_prompt": negative_prompt, "seed": seed,
            "height": height, "width": width, "video_length": frames,
            "infer_steps": STEPS, "guidance_scale": cfg_scale,
            "flow_shift": FLOW_SHIFT, "embedded_guidance_scale": EMBEDDED_GUIDANCE,
        }
        # predict() recreates a device-local generator and scheduler for each call.
        baseline = sampler.predict(**sampling, cache_enabled=False, profile=True,
                                   callback_on_step_end=on_step) if compare else None
        cached = sampler.predict(**sampling, cache_alpha=alpha, cache_beta=beta, profile=True,
                                 callback_on_step_end=on_step)

        report = {
            "config": {**sampling, "alpha": alpha, "beta": beta, "compare": compare, "fps": FPS,
                       "effective_negative_prompt": cached["negative_prompts"][0]},
            "estimated_layer_cache_bytes": cache_bytes,
            "layer_count": LAYERS,
            "mg_cache": cached["run_stats"],
        }
        cached_frames = decoded_frames(cached)
        baseline_frames = None
        if baseline is not None:
            baseline_stats = baseline["run_stats"]
            if baseline_stats["initial_latents_sha256"] != report["mg_cache"]["initial_latents_sha256"]:
                raise RuntimeError("Baseline and MG-Cache did not start from the same latents.")
            baseline_frames = decoded_frames(baseline)
            report["baseline"] = baseline_stats
            report["speedup"] = baseline_stats["denoise_seconds"] / report["mg_cache"]["denoise_seconds"]
            report["quality"] = compare_frames(baseline_frames, cached_frames)

        output_dir = Path(folder_paths.get_output_directory()) / "mg_cache"
        output_dir.mkdir(parents=True, exist_ok=True)
        run_dir = output_dir / uuid4().hex
        subfolder = f"mg_cache/{run_dir.name}"
        report["files"] = {"mg_cache": "mg_cache.mp4", "report": "report.json"}
        with tempfile.TemporaryDirectory(dir=output_dir) as temporary:
            staging = Path(temporary)
            write_video(staging / "mg_cache.mp4", cached_frames)
            if baseline_frames is not None:
                write_video(staging / "baseline.mp4", baseline_frames)
                report["files"]["baseline"] = "baseline.mp4"
            (staging / "report.json").write_text(
                json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8",
            )
            staging.rename(run_dir)

        return {"ui": {"mg_cache_report": [{**report, "subfolder": subfolder}]}}
