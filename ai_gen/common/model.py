from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from .config import BaseConfig


class BaseModel(nn.Module, ABC):
    """
    Base class for all models. Ties an nn.Module to a typed config and provides
    checkpoint save/load. Subclasses must set config_class and implement forward().

    Convention: forward(batch: dict) -> dict must return at least {"loss": tensor}
    during training. Add any extra outputs (logits, hidden_states, etc.) as needed.
    """

    config_class: type[BaseConfig] = BaseConfig

    def __init__(self, config: BaseConfig):
        super().__init__()
        self.config = config

    @abstractmethod
    def forward(self, batch: dict) -> dict:
        ...

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, path: str | Path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        self.config.to_json(path / "config.json")

    @classmethod
    def load_checkpoint(cls, path: str | Path, config: BaseConfig | None = None) -> BaseModel:
        path = Path(path)
        if config is None:
            config = cls.config_class.from_json(path / "config.json")
        model = cls(config)
        state = torch.load(path / "model.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model
