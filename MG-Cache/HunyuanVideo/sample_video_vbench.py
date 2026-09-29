import os
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import numpy as np

import json


def main():
    args = parse_args()
    print(args)
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args

    # Start sampling
    # TODO: batch inference check
    with open(args.vbench_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    start_i = args.index_start
    end_i = args.index_end

    latency = []
    for i in range(start_i, end_i):
        prompt = data[i]["prompt_en"]
        for seed_offset in range(args.num_per_prompt):
            cur_save_path = f"{save_path}/{prompt}-{seed_offset}.mp4"
            if os.path.exists(cur_save_path):
                continue
            outputs = hunyuan_video_sampler.predict(
                prompt=prompt, 
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=(args.seed + seed_offset),
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale
            )
            samples = outputs['samples']
            latency.append(outputs["latency"])
            
            # Save samples
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                for i, sample in enumerate(samples):
                    sample = samples[i].unsqueeze(0)
                    save_videos_grid(sample, cur_save_path, fps=24)
                    logger.info(f'Sample save to: {cur_save_path}')

    print(f'avg latency: {np.mean(latency):.4f}')

if __name__ == "__main__":
    main()
