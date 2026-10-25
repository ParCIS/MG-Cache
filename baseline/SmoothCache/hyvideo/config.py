import argparse
from .constants import *
import re
from .modules.models import HUNYUAN_VIDEO_CONFIG


def parse_args(namespace=None):
    parser = argparse.ArgumentParser(description="HunyuanVideo inference script")

    parser = add_network_args(parser)
    parser = add_extra_models_args(parser)
    parser = add_denoise_schedule_args(parser)
    parser = add_inference_args(parser)
    parser = add_parallel_args(parser)
    parser = add_smoothcache_args(parser)

    args = parser.parse_args(namespace=namespace)
    args = sanity_check_args(args)

    return args


def add_network_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="HunyuanVideo network args")

    # Main model
    group.add_argument(
        "--model",
        type=str,
        choices=list(HUNYUAN_VIDEO_CONFIG.keys()),
        default="HYVideo-T/2-cfgdistill",
    )
    group.add_argument(
        "--latent-channels",
        type=str,
        default=16,
        help="Number of latent channels of DiT. If None, it will be determined by `vae`. If provided, "
        "it still needs to match the latent channels of the VAE model.",
    )
    group.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=PRECISIONS,
        help="Precision mode. Options: fp32, fp16, bf16. Applied to the backbone model and optimizer.",
    )

    # RoPE
    group.add_argument(
        "--rope-theta", type=int, default=256, help="Theta used in RoPE."
    )
    return parser


def add_extra_models_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(
        title="Extra models args, including vae, text encoders and tokenizers)"
    )

    # - VAE
    group.add_argument(
        "--vae",
        type=str,
        default="884-16c-hy",
        choices=list(VAE_PATH),
        help="Name of the VAE model.",
    )
    group.add_argument(
        "--vae-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the VAE model.",
    )
    group.add_argument(
        "--vae-tiling",
        action="store_true",
        help="Enable tiling for the VAE model to save GPU memory.",
    )
    group.set_defaults(vae_tiling=True)

    group.add_argument(
        "--text-encoder",
        type=str,
        default="llm",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the text encoder model.",
    )
    group.add_argument(
        "--text-states-dim",
        type=int,
        default=4096,
        help="Dimension of the text encoder hidden states.",
    )
    group.add_argument(
        "--text-len", type=int, default=256, help="Maximum length of the text input."
    )
    group.add_argument(
        "--tokenizer",
        type=str,
        default="llm",
        choices=list(TOKENIZER_PATH),
        help="Name of the tokenizer model.",
    )
    group.add_argument(
        "--prompt-template",
        type=str,
        default="dit-llm-encode",
        choices=PROMPT_TEMPLATE,
        help="Image prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--prompt-template-video",
        type=str,
        default="dit-llm-encode-video",
        choices=PROMPT_TEMPLATE,
        help="Video prompt template for the decoder-only text encoder model.",
    )
    group.add_argument(
        "--hidden-state-skip-layer",
        type=int,
        default=2,
        help="Skip layer for hidden states.",
    )
    group.add_argument(
        "--apply-final-norm",
        action="store_true",
        help="Apply final normalization to the used text encoder hidden states.",
    )

    # - CLIP
    group.add_argument(
        "--text-encoder-2",
        type=str,
        default="clipL",
        choices=list(TEXT_ENCODER_PATH),
        help="Name of the second text encoder model.",
    )
    group.add_argument(
        "--text-encoder-precision-2",
        type=str,
        default="fp16",
        choices=PRECISIONS,
        help="Precision mode for the second text encoder model.",
    )
    group.add_argument(
        "--text-states-dim-2",
        type=int,
        default=768,
        help="Dimension of the second text encoder hidden states.",
    )
    group.add_argument(
        "--tokenizer-2",
        type=str,
        default="clipL",
        choices=list(TOKENIZER_PATH),
        help="Name of the second tokenizer model.",
    )
    group.add_argument(
        "--text-len-2",
        type=int,
        default=77,
        help="Maximum length of the second text input.",
    )

    return parser


def add_denoise_schedule_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Denoise schedule args")

    group.add_argument(
        "--denoise-type",
        type=str,
        default="flow",
        help="Denoise type for noised inputs.",
    )

    # Flow Matching
    group.add_argument(
        "--flow-shift",
        type=float,
        default=7.0,
        help="Shift factor for flow matching schedulers.",
    )
    group.add_argument(
        "--flow-reverse",
        action="store_true",
        help="If reverse, learning/sampling from t=1 -> t=0.",
    )
    group.add_argument(
        "--flow-solver",
        type=str,
        default="euler",
        help="Solver for flow matching.",
    )
    group.add_argument(
        "--use-linear-quadratic-schedule",
        action="store_true",
        help="Use linear quadratic schedule for flow matching."
        "Following MovieGen (https://ai.meta.com/static-resource/movie-gen-research-paper)",
    )
    group.add_argument(
        "--linear-schedule-end",
        type=int,
        default=25,
        help="End step for linear quadratic schedule for flow matching.",
    )

    return parser


def add_inference_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Inference args")

    # ======================== Model loads ========================
    group.add_argument(
        "--model-base",
        type=str,
        default="ckpts",
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--dit-weight",
        type=str,
        default="ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
        help="Path to the HunyuanVideo model. If None, search the model in the args.model_root."
        "1. If it is a file, load the model directly."
        "2. If it is a directory, search the model in the directory. Support two types of models: "
        "1) named `pytorch_model_*.pt`"
        "2) named `*_model_states.pt`, where * can be `mp_rank_00`.",
    )
    group.add_argument(
        "--model-resolution",
        type=str,
        default="540p",
        choices=["540p", "720p"],
        help="Root path of all the models, including t2v models and extra models.",
    )
    group.add_argument(
        "--load-key",
        type=str,
        default="module",
        help="Key to load the model states. 'module' for the main model, 'ema' for the EMA model.",
    )
    group.add_argument(
        "--use-cpu-offload",
        action="store_true",
        help="Use CPU offload for the model load.",
    )

    # ======================== Inference general setting ========================
    group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference and evaluation.",
    )
    group.add_argument(
        "--infer-steps",
        type=int,
        default=50,
        help="Number of denoising steps for inference.",
    )
    group.add_argument(
        "--disable-autocast",
        action="store_true",
        help="Disable autocast for denoising loop and vae decoding in pipeline sampling.",
    )
    group.add_argument(
        "--save-path",
        type=str,
        default="./results",
        help="Path to save the generated samples.",
    )
    group.add_argument(
        "--save-path-suffix",
        type=str,
        default="",
        help="Suffix for the directory of saved samples.",
    )
    group.add_argument(
        "--name-suffix",
        type=str,
        default="",
        help="Suffix for the names of saved samples.",
    )
    group.add_argument(
        "--num-videos",
        type=int,
        default=1,
        help="Number of videos to generate for each prompt.",
    )
    # ---sample size---
    group.add_argument(
        "--video-size",
        type=int,
        nargs="+",
        default=(720, 1280),
        help="Video size for training. If a single value is provided, it will be used for both height "
        "and width. If two values are provided, they will be used for height and width "
        "respectively.",
    )
    group.add_argument(
        "--video-length",
        type=int,
        default=129,
        help="How many frames to sample from a video. if using 3d vae, the number should be 4n+1",
    )
    # --- prompt ---
    group.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Prompt for sampling during evaluation.",
    )
    group.add_argument(
        "--seed-type",
        type=str,
        default="auto",
        choices=["file", "random", "fixed", "auto"],
        help="Seed type for evaluation. If file, use the seed from the CSV file. If random, generate a "
        "random seed. If fixed, use the fixed seed given by `--seed`. If auto, `csv` will use the "
        "seed column if available, otherwise use the fixed `seed` value. `prompt` will use the "
        "fixed `seed` value.",
    )
    group.add_argument("--seed", type=int, default=None, help="Seed for evaluation.")

    # Classifier-Free Guidance
    group.add_argument(
        "--neg-prompt", type=str, default=None, help="Negative prompt for sampling."
    )
    group.add_argument(
        "--cfg-scale", type=float, default=1.0, help="Classifier free guidance scale."
    )
    group.add_argument(
        "--embedded-cfg-scale",
        type=float,
        default=6.0,
        help="Embeded classifier free guidance scale.",
    )

    group.add_argument(
        "--use-fp8",
        action="store_true",
        help="Enable use fp8 for inference acceleration."
    )

    group.add_argument(
        "--reproduce",
        action="store_true",
        help="Enable reproducibility by setting random seeds and deterministic algorithms.",
    )

    return parser


def add_parallel_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="Parallel args")

    # ======================== Model loads ========================
    group.add_argument(
        "--ulysses-degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )
    group.add_argument(
        "--ring-degree",
        type=int,
        default=1,
        help="Ulysses degree.",
    )

    return parser


def add_smoothcache_args(parser: argparse.ArgumentParser):
    group = parser.add_argument_group(title="SmoothCache args")

    group.add_argument(
        "--smoothcache",
        action="store_true",
        help="Enable SmoothCache layer caching for video DiT inference.",
    )
    group.add_argument(
        "--smoothcache-schedule",
        type=str,
        default=None,
        help="Path to a static SmoothCache schedule JSON "
        "({component: [0/1 per step]}). Takes precedence over --smoothcache-alpha.",
    )
    group.add_argument(
        "--smoothcache-alpha",
        type=float,
        default=0.05,
        help="Error threshold used to build the SmoothCache schedule from the "
        "calibration error curves (lower = fewer cached steps).",
    )
    group.add_argument(
        "--smoothcache-lookahead",
        type=int,
        default=3,
        help="Calibration lookahead k in {1, ..., lookahead} (paper default 3).",
    )
    group.add_argument(
        "--smoothcache-calibration-json",
        type=str,
        default=None,
        help="Path to averaged calibration error curves JSON, used to build "
        "schedules offline (scripts/run_smoothcache_batch.py / calibration).",
    )
    group.add_argument(
        "--smoothcache-log-dir",
        type=str,
        default="smoothcache_logs",
        help="Directory for per-video SmoothCache step logs and summary CSV.",
    )
    group.add_argument(
        "--smoothcache-calibrate",
        action="store_true",
        help="Calibration mode: run full-compute generations and record layer "
        "output errors (scripts/run_smoothcache_calibration.py).",
    )
    group.add_argument(
        "--offload-text-encoder",
        action="store_true",
        help="Move the LLM/CLIP text encoders to CPU after prompt encoding "
        "and restore them on the next call. Frees ~16GB during denoising, "
        "required for SmoothCache runs at 544x960 on an 80GB GPU.",
    )
    group.add_argument(
        "--calibration-output",
        type=str,
        default="smoothcache_calibration.json",
        help="Output JSON path for averaged calibration error curves.",
    )
    group.add_argument(
        "--calibration-plot",
        type=str,
        default=None,
        help="Optional PNG path for the calibration error-curve plot.",
    )
    group.add_argument(
        "--calibration-video-size",
        type=int,
        nargs="+",
        default=[272, 480],
        help="Low-resolution calibration video size (height width). The "
        "calibrator keeps lookahead copies of every cached layer output in "
        "memory, so reducing resolution cuts peak memory quadratically. "
        "Default 272 480 (one quarter of the acceptance 544x960 tokens).",
    )
    group.add_argument(
        "--calibration-video-length",
        type=int,
        default=129,
        help="Number of frames for calibration videos (default matches the "
        "acceptance run; lower it if GPU memory is still tight).",
    )
    group.add_argument(
        "--alpha-scan",
        type=str,
        default="0.02,0.04,0.06,0.08,0.12,0.16,0.20",
        help="Comma-separated alpha values for the offline alpha sweep.",
    )
    group.add_argument(
        "--prompt-json",
        type=str,
        default=None,
        help="Path to a JSON file containing a list of prompt strings for "
        "batch generation / calibration.",
    )
    group.add_argument(
        "--index-offset",
        type=int,
        default=0,
        help="Global index offset for batch generation: video / log files are "
        "named by (shard index + offset), so parallel shards keep unique, "
        "order-preserving names.",
    )
    group.add_argument(
        "--video-name-format",
        type=str,
        default="prompt",
        choices=["prompt", "index"],
        help="Video file naming: 'prompt' = {sanitized_prompt}-0.mp4 "
        "(default), 'index' = {global_index:03d}_seed{seed}.mp4.",
    )
    group.add_argument(
        "--batch-video-size",
        type=int,
        nargs="+",
        default=[544, 960],
        help="Height width for batch generation (scripts/run_smoothcache_batch.py), "
        "default 544 960.",
    )
    group.add_argument(
        "--batch-video-length",
        type=int,
        default=129,
        help="Number of frames for batch generation, default 129.",
    )
    return parser


def sanity_check_args(args):
    # VAE channels
    vae_pattern = r"\d{2,3}-\d{1,2}c-\w+"
    if not re.match(vae_pattern, args.vae):
        raise ValueError(
            f"Invalid VAE model: {args.vae}. Must be in the format of '{vae_pattern}'."
        )
    vae_channels = int(args.vae.split("-")[1][:-1])
    if args.latent_channels is None:
        args.latent_channels = vae_channels
    if vae_channels != args.latent_channels:
        raise ValueError(
            f"Latent channels ({args.latent_channels}) must match the VAE channels ({vae_channels})."
        )
    return args
