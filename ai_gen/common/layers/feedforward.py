from __future__ import annotations

import torch
import torch.nn as nn

_ACTIVATIONS = {
    "relu":  nn.ReLU,
    "gelu":  nn.GELU,
    "swish": nn.SiLU,   # SiLU == Swish
    "silu":  nn.SiLU,
}


class FeedForward(nn.Module):
    """
    Two-layer position-wise feed-forward network used inside Transformer and Conformer blocks.

    Architecture:  Linear → Activation → Dropout → Linear → Dropout
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.0,
        activation: str = "gelu",
    ):
        super().__init__()
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. Choose: {list(_ACTIVATIONS)}")

        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            _ACTIVATIONS[activation](),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
