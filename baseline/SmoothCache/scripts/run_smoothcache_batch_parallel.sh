#!/usr/bin/env bash
# Run independent single-GPU workers. Activate the inference environment first.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export MODEL_BASE="${MODEL_BASE:-$PWD/ckpts}"
export DIT_WEIGHT="${DIT_WEIGHT:-$MODEL_BASE/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export BASELINE_SKIP="${SKIP:-0}"
export BASELINE_GPUS="${GPU_IDS:-0 1 2 3 4 5 6 7}"
export BASELINE_OUTPUT="${SAVE_PATH:-results/smoothcache_alpha_016}"
export ALPHA="${ALPHA:-0.16}"
export CALIBRATION_JSON="${CALIBRATION_JSON:-smoothcache_calibration.json}"
python - <<'PY'
import json
import os
import subprocess
import sys
import time
from pathlib import Path

repo = Path.cwd()
skip = int(os.environ["BASELINE_SKIP"])
gpus = os.environ["BASELINE_GPUS"].replace(",", " ").split()
if skip < 0 or not gpus or len(gpus) != len(set(gpus)):
    raise SystemExit("SKIP must be non-negative and GPU_IDS must be non-empty and unique")
with open(os.environ.get("PROMPT_JSON", "VBench_small_info.json"), encoding="utf-8") as f:
    data = json.load(f)
prompts = [p if isinstance(p, str) else p.get("prompt") or p.get("prompt_en", "") for p in data]
prompts = prompts[skip:]
if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
    raise SystemExit("No valid prompts left; check PROMPT_JSON and SKIP")
output = Path(os.environ["BASELINE_OUTPUT"]).resolve()
log_dir = output / "logs"
shard_dir = log_dir / "shards"
shard_dir.mkdir(parents=True, exist_ok=True)
base = [
    sys.executable, "scripts/run_smoothcache_batch.py",
    "--model-base", os.environ["MODEL_BASE"],
    "--dit-weight", os.environ["DIT_WEIGHT"],
    "--save-path", str(output),
    "--smoothcache", "--smoothcache-alpha", os.environ["ALPHA"],
    "--smoothcache-calibration-json", os.environ["CALIBRATION_JSON"],
    "--batch-video-size", *os.environ.get("VIDEO_SIZE", "544 960").split(),
    "--batch-video-length", os.environ.get("VIDEO_LENGTH", "129"),
    "--offload-text-encoder", "--smoothcache-log-dir", str(log_dir),
]
env = os.environ.copy()
env["PYTHONPATH"] = str(repo) + os.pathsep + env.get("PYTHONPATH", "")
workers = []
start = 0
for position, gpu in enumerate(gpus):
    count = len(prompts) // len(gpus) + (position < len(prompts) % len(gpus))
    if not count:
        continue
    shard = shard_dir / f"shard_{position}.json"
    shard.write_text(json.dumps(prompts[start:start + count], ensure_ascii=False, indent=2), encoding="utf-8")
    command = base + ["--prompt-json", str(shard), "--index-offset", str(skip + start)]
    env["CUDA_VISIBLE_DEVICES"] = gpu
    with (log_dir / f"worker_{position}.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(command, cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
    workers.append(process)
    print(f"gpu={gpu} pid={process.pid} start={skip + start} count={count}", flush=True)
    start += count
    if start < len(prompts):
        time.sleep(float(os.environ.get("LAUNCH_DELAY", "45")))
exit_codes = [worker.wait() for worker in workers]
if any(exit_codes):
    raise SystemExit(f"Worker failures: {exit_codes}; see {log_dir}")
PY
