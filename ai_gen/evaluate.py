"""
Entry point for evaluation.

Usage:
    # Evaluate the latest checkpoint using the experiment config saved during training:
    python evaluate.py --checkpoint outputs/checkpoint-epoch-0010

    # Evaluate on a specific split with a custom config:
    python evaluate.py --checkpoint outputs/checkpoint-epoch-0010 --config configs/text/transformer_base.json --split test

The script looks for experiment_config.json in the checkpoint's parent directory
(written there automatically by train.py). Pass --config to override.
"""

from __future__ import annotations

import argparse
import json
import logging
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
    parser = argparse.ArgumentParser(description="Evaluate a saved model checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Experiment config JSON. Defaults to experiment_config.json in the checkpoint's parent.",
    )
    parser.add_argument("--split", default="test", help="Dataset split to evaluate on (default: test).")
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)

    # Resolve experiment config: explicit flag > saved alongside training outputs.
    config_path = Path(args.config) if args.config else checkpoint_path.parent / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Experiment config not found at {config_path}. "
            "Pass --config explicitly or point --checkpoint to a directory produced by train.py."
        )

    with open(config_path) as f:
        cfg = json.load(f)

    model_type = cfg["model_type"]
    dataset_type = cfg["dataset_type"]

    model_cls = Registry.get("model", model_type)
    model = model_cls.load_checkpoint(checkpoint_path)
    logger.info(f"Loaded {model_type} from {checkpoint_path} | {model.num_parameters():,} parameters")

    dataset_cls = Registry.get("dataset", dataset_type)
    dataset_kwargs = cfg.get("dataset_config", {})
    eval_dataset = dataset_cls(split=args.split, **dataset_kwargs)

    training_args = TrainingArguments.from_dict(cfg.get("training_args", {}))

    trainer = BaseTrainer(
        model=model,
        args=training_args,
        eval_dataset=eval_dataset,
    )

    logger.info(f"Evaluating on split='{args.split}' ...")
    metrics = trainer.evaluate()

    logger.info(f"Results [{args.split}]:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
