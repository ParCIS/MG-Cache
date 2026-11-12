import math

import numpy as np
from skimage.metrics import structural_similarity


def compare_frames(reference, candidate):
    """Compare pairs of decoded FHWC RGB frames in [0, 1]."""
    if reference.shape != candidate.shape or len(reference) == 0:
        raise ValueError("Baseline and MG-Cache frame shapes must match.")

    psnr = []
    ssim = []
    for baseline, cached in zip(reference, candidate):
        mse = np.mean(np.square(baseline - cached), dtype=np.float64)
        psnr.append("inf" if mse == 0 else -10 * math.log10(mse))
        ssim.append(float(structural_similarity(
            baseline, cached, data_range=1.0, channel_axis=-1,
            gaussian_weights=True, sigma=1.5, use_sample_covariance=False,
        )))

    return {
        "psnr": {"mean": "inf" if "inf" in psnr else math.fsum(psnr) / len(psnr), "frames": psnr},
        "ssim": {"mean": math.fsum(ssim) / len(ssim), "frames": ssim},
    }
