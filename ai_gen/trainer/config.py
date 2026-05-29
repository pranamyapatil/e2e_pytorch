from __future__ import annotations

from dataclasses import dataclass

from common.config import BaseConfig


@dataclass
class TrainingArguments(BaseConfig):
    output_dir: str = "outputs"

    # loop
    num_epochs: int = 10
    batch_size: int = 32
    eval_batch_size: int = 32
    gradient_accumulation_steps: int = 1

    # optimisation
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"     # adamw | adam | sgd
    scheduler: str = "cosine"   # cosine | linear | constant
    warmup_steps: int = 0

    # logging / checkpointing (in epochs unless noted)
    log_every: int = 50          # steps
    eval_every: int = 1
    save_every: int = 1

    # hardware
    device: str = "auto"         # auto | cpu | cuda | mps
    fp16: bool = False
    num_workers: int = 4
    seed: int = 42
