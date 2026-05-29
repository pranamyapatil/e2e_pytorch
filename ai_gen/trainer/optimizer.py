from __future__ import annotations

import torch
from torch.optim import Optimizer

from .config import TrainingArguments

# Parameters containing these substrings are excluded from weight decay.
_NO_DECAY = {"bias", "LayerNorm.weight", "layer_norm.weight"}


def get_optimizer(named_params: list[tuple[str, torch.nn.Parameter]], args: TrainingArguments) -> Optimizer:
    param_groups = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in _NO_DECAY)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in _NO_DECAY)],
            "weight_decay": 0.0,
        },
    ]

    match args.optimizer:
        case "adamw":
            return torch.optim.AdamW(param_groups, lr=args.learning_rate)
        case "adam":
            return torch.optim.Adam(param_groups, lr=args.learning_rate)
        case "sgd":
            return torch.optim.SGD(param_groups, lr=args.learning_rate, momentum=0.9)
        case _:
            raise ValueError(f"Unknown optimizer '{args.optimizer}'. Choose: adamw | adam | sgd")
