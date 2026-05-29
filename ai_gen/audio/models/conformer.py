"""
Conformer encoder for speech recognition (Gulati et al., 2020).

Architecture — each Conformer block:
    x  →  ½ FFN  →  MHSA  →  Convolution  →  ½ FFN  →  LayerNorm  →  output

Full model:
    Linear input projection
    → N × ConformerBlock
    → CTC linear head  →  log-softmax  →  CTC loss

Registered as: Registry("model", "conformer")

────────────────────────────────────────────────────────────
Model API
────────────────────────────────────────────────────────────
forward(batch)                  → {"loss", "log_probs", "output_lengths"}   training
encode(features, lengths)       → (encoded, output_lengths)                 encoder pass
greedy_decode(log_probs, lens)  → list[list[int]]                           CTC collapse
generate(features, lengths)     → list[list[int]]                           full inference
────────────────────────────────────────────────────────────
Inherited from BaseModel
────────────────────────────────────────────────────────────
save_checkpoint(path)
load_checkpoint(path, config)
num_parameters()                → int
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import BaseConfig
from common.model import BaseModel
from common.registry import Registry
from common.layers import MultiHeadAttention, FeedForward


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class ConformerConfig(BaseConfig):
    input_dim: int = 80          # mel filterbank dimension
    d_model: int = 256
    num_heads: int = 4
    num_layers: int = 16
    d_ff: int = 2048
    conv_kernel_size: int = 31   # must be odd; controls local context in conv module
    dropout: float = 0.1
    vocab_size: int = 32         # output units (characters, BPE, or phonemes)
    blank_id: int = 0            # CTC blank token id


# ─────────────────────────────────────────────
# Sub-modules
# ─────────────────────────────────────────────

class ConvolutionModule(nn.Module):
    """
    Conformer convolution sub-module.

    pointwise (gating) → depthwise conv → BatchNorm → Swish → pointwise

    GLU gating doubles the channel width before the depthwise conv and
    halves it back, allowing the network to selectively pass information.
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        assert kernel_size % 2 == 1, "conv_kernel_size must be odd (same-length output)"
        padding = kernel_size // 2

        self.norm = nn.LayerNorm(d_model)
        self.pointwise_expand = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_project = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.activation = nn.SiLU()   # Swish
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = self.norm(x).transpose(1, 2)             # (B, d_model, T)
        x = F.glu(self.pointwise_expand(x), dim=1)  # (B, d_model, T)  GLU halves channels
        x = self.depthwise(x)
        x = self.activation(self.batch_norm(x))
        x = self.pointwise_project(x).transpose(1, 2)  # (B, T, d_model)
        return self.dropout(x)


class FeedForwardModule(nn.Module):
    """Pre-norm Swish FFN used at both ends of a Conformer block (scaled ½ at the call site)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation="swish")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(self.norm(x))


class ConformerBlock(nn.Module):
    """
    Single Conformer block.

    x → ½ FFN₁ → MHSA → Conv → ½ FFN₂ → LayerNorm
    Each sub-module output is added to the residual stream.
    """

    def __init__(self, config: ConformerConfig):
        super().__init__()
        self.ffn1 = FeedForwardModule(config.d_model, config.d_ff, config.dropout)
        self.attn_norm = nn.LayerNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.conv = ConvolutionModule(config.d_model, config.conv_kernel_size, config.dropout)
        self.ffn2 = FeedForwardModule(config.d_model, config.d_ff, config.dropout)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + 0.5 * self.ffn1(x)

        normed = self.attn_norm(x)
        attn_out, _ = self.self_attn(normed, normed, normed, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        x = x + self.conv(x)
        x = x + 0.5 * self.ffn2(x)
        return self.final_norm(x)


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────

@Registry.register("model", "conformer")
class ConformerModel(BaseModel):
    """
    Conformer encoder + CTC head for automatic speech recognition.

    batch contract (forward):
        features        (B, T, input_dim)  log mel filterbanks
        feature_lengths (B,)               real frame counts (before padding)
        labels          (B, L)             token ids (padded)
        label_lengths   (B,)               real label lengths
    """

    config_class = ConformerConfig

    def __init__(self, config: ConformerConfig):
        super().__init__(config)
        self.input_projection = nn.Linear(config.input_dim, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList([ConformerBlock(config) for _ in range(config.num_layers)])
        self.ctc_head = nn.Linear(config.d_model, config.vocab_size)

    # ── Public API ──────────────────────────────

    def forward(self, batch: dict) -> dict:
        features = batch["features"]
        feature_lengths = batch["feature_lengths"]
        labels = batch["labels"]
        label_lengths = batch["label_lengths"]

        encoded, output_lengths = self.encode(features, feature_lengths)
        log_probs = F.log_softmax(self.ctc_head(encoded), dim=-1)   # (B, T, vocab)

        # CTC loss expects input as (T, B, vocab)
        loss = F.ctc_loss(
            log_probs.permute(1, 0, 2),
            labels,
            output_lengths,
            label_lengths,
            blank=self.config.blank_id,
            reduction="mean",
            zero_infinity=True,  # guards against -inf gradients on very short sequences
        )
        return {"loss": loss, "log_probs": log_probs, "output_lengths": output_lengths}

    def encode(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode raw acoustic features into contextual representations.

        Args:
            features: (B, T, input_dim) — log mel filterbanks
            lengths:  (B,)             — number of real frames per utterance
        Returns:
            encoded:        (B, T, d_model)
            output_lengths: (B,)  unchanged here; update if subsampling is added
        """
        x = self.dropout(self.input_projection(features))
        key_padding_mask = self._lengths_to_padding_mask(lengths, x.size(1))
        for block in self.encoder_layers:
            x = block(x, key_padding_mask)
        return x, lengths

    def greedy_decode(
        self,
        log_probs: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[list[int]]:
        """
        CTC greedy decode: argmax per frame → collapse repeats → remove blanks.

        Args:
            log_probs: (B, T, vocab)
            lengths:   (B,) real frame counts
        Returns:
            List of token id lists, one per batch item (variable length).
        """
        best_paths = log_probs.argmax(dim=-1)  # (B, T)
        results: list[list[int]] = []

        for b in range(best_paths.size(0)):
            seq = best_paths[b, : lengths[b]].tolist()
            # Collapse consecutive repeats, then strip blank
            collapsed = [
                t for i, t in enumerate(seq)
                if t != self.config.blank_id and (i == 0 or t != seq[i - 1])
            ]
            results.append(collapsed)

        return results

    def generate(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> list[list[int]]:
        """
        Full inference pipeline: encode → CTC greedy decode.

        Args:
            features: (B, T, input_dim)
            lengths:  (B,) real frame counts
        Returns:
            List of decoded token id lists, one per utterance.
        """
        self.eval()
        with torch.no_grad():
            encoded, output_lengths = self.encode(features, lengths)
            log_probs = F.log_softmax(self.ctc_head(encoded), dim=-1)
        return self.greedy_decode(log_probs, output_lengths)

    # ── Helpers ─────────────────────────────────

    @staticmethod
    def _lengths_to_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        """Returns (B, T) bool mask — True where position is padding."""
        idxs = torch.arange(max_len, device=lengths.device).unsqueeze(0)
        return idxs >= lengths.unsqueeze(1)
