from __future__ import annotations

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from .config import TrainingArguments


def get_scheduler(optimizer: Optimizer, args: TrainingArguments, total_steps: int) -> LambdaLR:
    warmup = args.warmup_steps

    def _warmup_scale(step: int) -> float:
        return step / max(1, warmup) if step < warmup else 1.0

    match args.scheduler:
        case "constant":
            lr_lambda = _warmup_scale

        case "linear":
            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return step / max(1, warmup)
                return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

        case "cosine":
            def lr_lambda(step: int) -> float:
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, total_steps - warmup)
                return 0.5 * (1.0 + math.cos(math.pi * progress))

        case _:
            raise ValueError(f"Unknown scheduler '{args.scheduler}'. Choose: cosine | linear | constant")

    return LambdaLR(optimizer, lr_lambda)
