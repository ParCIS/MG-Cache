# HunyuanVideo baselines

This directory contains the HunyuanVideo baseline implementations used in our
MG-Cache study. FORA, MixCache and SmoothCache are our implementations based on
[official HunyuanVideo](https://github.com/Tencent-Hunyuan/HunyuanVideo), each with a
complete inference codebase. TeaCache and TaylorSeer use the official implementations
linked below; their source is hosted in the respective repositories.

| Method | Code | Paper setting |
| --- | --- | --- |
| FORA | [FORA/](FORA/) | N=3 |
| MixCache | [MixCache/](MixCache/) | Efficient configuration; N=6 variant |
| SmoothCache | [SmoothCache/](SmoothCache/) | Threshold delta=0.16 |
| TeaCache | [Official code](https://github.com/ali-vilab/TeaCache/tree/main/TeaCache4HunyuanVideo) | delta=0.4 at 544 x 960; delta=0.15 at 480 x 640 |
| TaylorSeer | [Official code](https://github.com/Shenyi-Z/TaylorSeer/tree/main/TaylorSeer-HunyuanVideo) | N=3, O=1 at 480 x 640; OOM at 544 x 960 |

## Setup

Use a Linux/CUDA environment with FlashAttention v2. Follow the HunyuanVideo
[installation guide](https://github.com/Tencent-Hunyuan/HunyuanVideo#dependencies-and-installation)
and [checkpoint download instructions](https://github.com/Tencent-Hunyuan/HunyuanVideo/blob/main/ckpts/README.md).
The three included baselines share the same Python requirements. From this
repository's root, install them and set the shared checkpoint paths:

```bash
python -m pip install -r baseline/FORA/requirements.txt
export MODEL_BASE=/absolute/path/to/ckpts
export DIT_WEIGHT="$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt"
```

The checkpoint directory must contain:

```text
ckpts/
  hunyuan-video-t2v-720p/
    transformers/mp_rank_00_model_states.pt
    vae/
  text_encoder/
  text_encoder_2/
```

Set `MODEL_BASE` before starting Python so the VAE and text encoders resolve to the
same checkpoint directory. Run the following commands from this repository's root.
Each example runs one video at a time on one GPU, with 50 denoising steps, seed 42,
embedded guidance 6.0 and `cfg_scale=1.0`.

## FORA

FORA refreshes cached attention every three steps (N=3).

```bash
(
cd baseline/FORA
python sample_video.py \
    --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --prompt "A cat walks on the grass, realistic style." \
    --seed 42 --cfg-scale 1.0 --embedded-cfg-scale 6.0 \
    --flow-shift 7.0 --flow-reverse \
    --save-path results/fora
)
```

For 480 x 640 / 65 frames, use `--video-size 480 640 --video-length 65`.
Keep batch size 1 and `cfg_scale=1.0`, as required by the attention cache buffers.

## MixCache

The example uses the efficient configuration with adaptive intervals `(3, 4, 5)`.
Batch prompts are read from the included `VBench_small_info.json`; a JSON array of
prompt strings is also accepted.

```bash
(
cd baseline/MixCache
python scripts/run_mixcache_batch.py \
    --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
    --prompt-json VBench_small_info.json \
    --mixcache --mixcache-config configs/mixcache_effi_2x.json \
    --video-size 544 960 --video-length 129 --infer-steps 50 \
    --cfg-scale 1.0 --flow-shift 7.0 --flow-reverse \
    --save-path results/mixcache \
    --mixcache-log-dir results/mixcache/logs
)
```

For 480 x 640 / 65 frames, use `--video-size 480 640 --video-length 65`.
Additional configurations are in [MixCache/configs/](MixCache/configs/).
`mixcache_effi_high3.json` changes thresholds as well as intervals; it should not be
interpreted as changing only N to 6. The example uses embedded guidance, so only
step and block caching apply.

## SmoothCache

First generate calibration curves using the included calibration prompts. The
272 x 480 calibration resolution reduces memory use. A calibration JSON is not
bundled; use the original experimental artifact if exact paper reproduction is needed.

```bash
(
cd baseline/SmoothCache
python scripts/run_smoothcache_calibration.py \
    --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
    --prompt-json prompts/smoothcache_calibration_10_prompts.json \
    --calibration-video-size 272 480 --calibration-video-length 129 \
    --calibration-output smoothcache_calibration.json
)
```

Then run with the paper's threshold, delta=0.16:

```bash
(
cd baseline/SmoothCache
python scripts/run_smoothcache_batch.py \
    --model-base "$MODEL_BASE" --dit-weight "$DIT_WEIGHT" \
    --prompt-json VBench_small_info.json \
    --smoothcache --smoothcache-calibration-json smoothcache_calibration.json \
    --smoothcache-alpha 0.16 --offload-text-encoder \
    --batch-video-size 544 960 --batch-video-length 129 \
    --save-path results/smoothcache \
    --smoothcache-log-dir results/smoothcache/logs
)
```

For 480 x 640 / 65 frames, use `--batch-video-size 480 640 --batch-video-length 65`.
Use a separate output directory for each configuration; existing videos are skipped.

## Acknowledgements and licenses

We thank the HunyuanVideo and baseline authors for their work. HunyuanVideo source
is copyright (C) 2024 THL A29 Limited, a Tencent company, and is distributed under
the Tencent Hunyuan Community License Agreement. Each implementation retains its
upstream license and dependency notices:
[FORA license](FORA/LICENSE.txt) / [Notice](FORA/Notice),
[MixCache license](MixCache/LICENSE.txt) / [Notice](MixCache/Notice), and
[SmoothCache license](SmoothCache/LICENSE.txt) / [Notice](SmoothCache/Notice).

SmoothCache scheduling is adapted from
[Roblox/SmoothCache](https://github.com/Roblox/SmoothCache), distributed under
[Apache 2.0](https://github.com/Roblox/SmoothCache/blob/main/LICENSE).
The Apache 2.0 terms are included in [SmoothCache/Notice](SmoothCache/Notice).
