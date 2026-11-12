import json
from functools import lru_cache
from pathlib import Path

import torch


CALIBRATION_DIR = Path(__file__).resolve().parents[1] / "calibration_data"


@lru_cache(maxsize=None)
def load_calibration(filename):
    path = CALIBRATION_DIR / filename
    if path.suffix == ".json":
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    return torch.load(path, map_location="cpu", weights_only=True)


class CacheRun:
    """Tensor buffers and error accumulators belonging to one denoising loop."""

    def __init__(self, skip_steps, num_layers, offload, device):
        self.skip_steps = frozenset(skip_steps)
        self.num_layers = num_layers
        self.offload = offload
        self.errors = [0.0] * num_layers
        self.buffers = []
        self.prefetch_stream = None
        self.timestep_output = None
        calibration = load_calibration("linear_fit_step.pth")
        self.fits = {
            index: {name: torch.as_tensor(value).to(device) for name, value in calibration[f"step_{index}"].items()}
            for index in self.skip_steps
        }
        self.scores = load_calibration("layer_scores.pth")

    def policy(self, index):
        if index in self.skip_steps:
            return "step"
        if index < 2:
            return "blank"
        return "layer" if self.buffers else "warmup"

    def allocate(self, img, txt):
        shape = (img.shape[0], img.shape[1] + txt.shape[1], img.shape[2])
        device = "cpu" if self.offload else img.device
        pin = self.offload and img.is_cuda
        self.buffers = [
            tuple(
                torch.empty(shape, dtype=img.dtype, device=device, pin_memory=pin)
                for _ in range(2)
            )
            for _ in range(self.num_layers)
        ]
        if pin:
            self.prefetch_stream = torch.cuda.Stream(device=img.device)

    def layer_inputs(self, device):
        """Prefetch the next layer while the current layer runs."""
        stream = self.prefetch_stream
        if stream is None:
            yield from self.buffers
            return

        compute = torch.cuda.current_stream(device)
        # Host buffers may still be receiving asynchronous writes from the previous step.
        stream.wait_stream(compute)
        with torch.cuda.stream(stream):
            current = tuple(t.to(device, non_blocking=True) for t in self.buffers[0])
        for next_buffers in self.buffers[1:]:
            compute.wait_stream(stream)
            with torch.cuda.stream(stream):
                following = tuple(t.to(device, non_blocking=True) for t in next_buffers)
            for tensor in current:
                tensor.record_stream(compute)
            yield current
            current = following
        compute.wait_stream(stream)
        for tensor in current:
            tensor.record_stream(compute)
        yield current

    def update_layer(self, index, layer_input, attention, error):
        for buffer, update in zip(self.buffers[index], (layer_input, attention)):
            if update is not None:
                buffer.copy_(update, non_blocking=True)
        self.errors[index] = error
