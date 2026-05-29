"""
Entry point for training.

Usage:
    python train.py --config configs/text/transformer_base.json
    python train.py --config configs/audio/conformer.json --resume outputs/checkpoint-epoch-0005

Config file schema (JSON):
    {
        "model_type":    "transformer",        # registered name in Registry("model")
        "dataset_type":  "text_lm",            # registered name in Registry("dataset")
        "do_eval":       true,
        "model_config":  { ... },              # passed to model.config_class.from_dict()
        "dataset_config": { ... },             # passed as kwargs to Dataset(split=..., **dataset_config)
        "training_args": { ... }               # passed to TrainingArguments.from_dict()
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from common.registry import Registry
from trainer import BaseTrainer, TrainingArguments

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a model from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to experiment JSON config file.")
    parser.add_argument("--resume", default=None, help="Path to a checkpoint directory to resume from.")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    training_args = TrainingArguments.from_dict(cfg.get("training_args", {}))
    output_dir = Path(training_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Persist the full experiment config alongside checkpoints for reproducibility.
    shutil.copy(args.config, output_dir / "experiment_config.json")

    model_type = cfg["model_type"]
    dataset_type = cfg["dataset_type"]

    model_cls = Registry.get("model", model_type)
    model_config = model_cls.config_class.from_dict(cfg.get("model_config", {}))

    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        model = model_cls.load_checkpoint(args.resume, model_config)
    else:
        model = model_cls(model_config)

    logger.info(f"Model: {model_type} | {model.num_parameters():,} trainable parameters")

    dataset_cls = Registry.get("dataset", dataset_type)
    dataset_kwargs = cfg.get("dataset_config", {})
    train_dataset = dataset_cls(split="train", **dataset_kwargs)
    eval_dataset = dataset_cls(split="eval", **dataset_kwargs) if cfg.get("do_eval", True) else None

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
