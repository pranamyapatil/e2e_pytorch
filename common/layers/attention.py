from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Scaled dot-product multi-head attention.

    Supports self-attention and cross-attention via separate query/key/value inputs.
    Masks follow PyTorch convention:
      - attn_mask:         additive (T, S) or (B*H, T, S) float mask (−inf blocks a position)
      - key_padding_mask:  (B, S) bool mask, True = pad position to ignore
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        key_padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (B, T, d_model)
            key:   (B, S, d_model)
            value: (B, S, d_model)
        Returns:
            output:       (B, T, d_model)
            attn_weights: (B, num_heads, T, S)
        """
        B, T, _ = query.shape
        S = key.size(1)

        Q = self.q_proj(query).view(B, T, self.num_heads, self.d_k).transpose(1, 2)
        K = self.k_proj(key).view(B, S, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_proj(value).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B, H, T, S)

        if attn_mask is not None:
            scores = scores + attn_mask

        if key_padding_mask is not None:
            # Expand to (B, 1, 1, S) so it broadcasts over heads and query positions
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)                         # (B, H, T, d_k)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.out_proj(out), attn_weights
