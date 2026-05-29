from __future__ import annotations

import logging
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from common.model import BaseModel
from .config import TrainingArguments
from .optimizer import get_optimizer
from .scheduler import get_scheduler

logger = logging.getLogger(__name__)


class BaseTrainer:
    """
    Domain-agnostic training loop.

    Expects model.forward(batch) to return a dict with at least {"loss": tensor}.
    Override compute_metrics() in a subclass to add domain-specific eval metrics
    (e.g. WER for audio, BLEU for text) on top of the default eval_loss.
    """

    def __init__(
        self,
        model: BaseModel,
        args: TrainingArguments,
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | None = None,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.device = self._resolve_device(args.device)
        self.model = self.model.to(self.device)
        self._set_seed(args.seed)

        self.optimizer = get_optimizer(list(model.named_parameters()), args)

        total_steps = self._compute_total_steps()
        self.scheduler = get_scheduler(self.optimizer, args, total_steps)

        # fp16 only on CUDA
        self.scaler = (
            torch.cuda.amp.GradScaler()
            if args.fp16 and self.device.type == "cuda"
            else None
        )
        self.global_step = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(self):
        if self.train_dataset is None:
            raise RuntimeError("train_dataset is required to call train().")

        loader = self._make_loader(self.train_dataset, self.args.batch_size, shuffle=True)
        logger.info(
            f"Training | device={self.device} | params={self.model.num_parameters():,} | "
            f"epochs={self.args.num_epochs} | steps/epoch={len(loader)}"
        )

        for epoch in range(1, self.args.num_epochs + 1):
            train_loss = self._train_epoch(loader, epoch)
            logger.info(f"[epoch {epoch}] train_loss={train_loss:.4f}")

            if self.eval_dataset and epoch % self.args.eval_every == 0:
                metrics = self.evaluate()
                logger.info(f"[epoch {epoch}] eval → " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

            if epoch % self.args.save_every == 0:
                self._save_checkpoint(epoch)

    def evaluate(self) -> dict[str, float]:
        if self.eval_dataset is None:
            raise RuntimeError("eval_dataset is required to call evaluate().")

        loader = self._make_loader(self.eval_dataset, self.args.eval_batch_size, shuffle=False)
        self.model.eval()
        total_loss, n = 0.0, 0
        all_outputs: list[dict] = []

        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                outputs = self.model(batch)
                total_loss += outputs["loss"].item()
                all_outputs.append(outputs)
                n += 1

        metrics = {"eval_loss": total_loss / max(n, 1)}
        metrics.update(self.compute_metrics(all_outputs))
        return metrics

    def compute_metrics(self, outputs: list[dict]) -> dict[str, float]:
        """Override in a domain-specific subclass to add WER, BLEU, accuracy, etc."""
        return {}

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        self.model.train()
        total_loss, n = 0.0, 0
        self.optimizer.zero_grad()

        for step, batch in enumerate(loader):
            batch = self._to_device(batch)
            loss = self._forward(batch) / self.args.gradient_accumulation_steps
            self._backward(loss)

            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                if self.args.max_grad_norm > 0:
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1

                if self.global_step % self.args.log_every == 0:
                    lr = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"  step={self.global_step} | loss={total_loss / max(n, 1):.4f} | lr={lr:.3e}"
                    )

            total_loss += loss.item() * self.args.gradient_accumulation_steps
            n += 1

        return total_loss / max(n, 1)

    def _forward(self, batch: dict) -> torch.Tensor:
        if self.scaler:
            with torch.cuda.amp.autocast():
                return self.model(batch)["loss"]
        return self.model(batch)["loss"]

    def _backward(self, loss: torch.Tensor):
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _save_checkpoint(self, epoch: int):
        path = Path(self.args.output_dir) / f"checkpoint-epoch-{epoch:04d}"
        self.model.save_checkpoint(path)
        self.args.to_json(path / "training_args.json")
        logger.info(f"Checkpoint saved → {path}")

    def _make_loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        collate_fn = getattr(dataset, "collate_fn", None)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.args.num_workers,
            pin_memory=(self.device.type == "cuda"),
            collate_fn=collate_fn,
        )

    def _to_device(self, batch):
        if isinstance(batch, dict):
            return {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        if isinstance(batch, torch.Tensor):
            return batch.to(self.device)
        return batch

    def _compute_total_steps(self) -> int:
        if self.train_dataset is None:
            return 0
        steps_per_epoch = len(self.train_dataset) // self.args.batch_size
        effective_steps = steps_per_epoch // max(1, self.args.gradient_accumulation_steps)
        return effective_steps * self.args.num_epochs

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
