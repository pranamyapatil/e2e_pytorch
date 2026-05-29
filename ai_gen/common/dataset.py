from __future__ import annotations

from abc import ABC
from typing import Callable

from torch.utils.data import Dataset


class BaseDataset(Dataset, ABC):
    """
    Extends torch.utils.data.Dataset with a collate_fn hook.
    BaseTrainer._make_loader picks this up automatically — no manual
    wiring needed when subclasses define a custom collator.
    """

    @property
    def collate_fn(self) -> Callable | None:
        """Return a collator callable, or None to use PyTorch's default."""
        return None
