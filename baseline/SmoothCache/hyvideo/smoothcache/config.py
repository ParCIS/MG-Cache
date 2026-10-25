"""Serialisable configuration for the SmoothCache HunyuanVideo adaptation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


DEFAULT_COMPONENTS = ("d_attn", "d_img_mlp", "d_txt_mlp", "s_attn_mlp")


@dataclass
class SmoothCacheConfig:
    """Experiment configuration for one SmoothCache run.

    ``num_steps`` is the number of denoising steps (schedule length),
    ``lookahead`` is the calibration lookahead (k = 1..lookahead, paper uses
    ``k in {1, 2, 3}``), ``alpha`` is the error threshold controlling how
    aggressively layers are cached, and ``components`` lists the layer types
    that participate in caching.
    """

    num_steps: int = 50
    lookahead: int = 3
    alpha: float = 0.05
    components: List[str] = field(default_factory=lambda: list(DEFAULT_COMPONENTS))
    # Optional schedule that overrides alpha-based generation. When both are
    # present, the explicit schedule wins.
    schedule: Optional[Dict[str, List[int]]] = None
    # Metadata (informational only).
    prompt_json: Optional[str] = None
    save_path: Optional[str] = None
    log_dir: Optional[str] = None

    @classmethod
    def from_json(cls, path: str) -> "SmoothCacheConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        known = {k for k in asdict(cls())}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
