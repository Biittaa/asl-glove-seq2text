"""Small, boring utilities shared across train.py / infer.py."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(preferred: str) -> torch.device:
    if preferred == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA requested but not available, falling back to CPU.")
        return torch.device("cpu")
    return torch.device(preferred)


def save_checkpoint(path: str | Path, model: torch.nn.Module, optimizer, epoch: int, extra: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            **extra,
        },
        path,
    )


def load_checkpoint(path: str | Path, model: torch.nn.Module, optimizer=None, map_location: str = "cpu") -> dict[str, Any]:
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model_state"])
    if optimizer is not None and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    return ckpt


class WarmupInverseSqrtLR:
    """Linear warmup followed by inverse-sqrt decay (Transformer-style schedule)."""

    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.step_num = 0

    def step(self) -> None:
        self.step_num += 1
        scale = self.d_model ** (-0.5) * min(
            self.step_num ** (-0.5), self.step_num * self.warmup_steps ** (-1.5)
        )
        for group in self.optimizer.param_groups:
            group["lr"] = scale
