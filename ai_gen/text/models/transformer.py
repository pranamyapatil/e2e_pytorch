"""
Seq2Seq Transformer for text (machine translation, summarisation, etc.)

Architecture:
    Embedding + SinusoidalPE
    → N × EncoderLayer  (self-attn + FFN, pre-norm)
    → M × DecoderLayer  (masked self-attn + cross-attn + FFN, pre-norm)
    → Linear projection → vocab logits

Registered as: Registry("model", "seq2seq_transformer")

────────────────────────────────────────────────────────────
Model API
────────────────────────────────────────────────────────────
forward(batch)          → {"loss", "logits"}          training
encode(src, mask)       → memory                      encoder-only pass
decode(tgt, memory, …)  → logits                      one decoder pass (teacher-forced)
generate(src, …)        → token_ids                   autoregressive greedy decode
────────────────────────────────────────────────────────────
Inherited from BaseModel
────────────────────────────────────────────────────────────
save_checkpoint(path)
load_checkpoint(path, config)
num_parameters()        → int
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.config import BaseConfig
from common.model import BaseModel
from common.registry import Registry
from common.layers import MultiHeadAttention, FeedForward, SinusoidalPositionalEncoding


# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

@dataclass
class TransformerConfig(BaseConfig):
    vocab_size: int = 32_000
    d_model: int = 512
    num_heads: int = 8
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    pad_id: int = 0


# ─────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────

class EncoderLayer(nn.Module):
    """Self-attention → FFN with pre-norm residuals."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Pre-norm self-attention
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = residual + self.dropout(attn_out)

        # Pre-norm FFN
        x = x + self.ffn(self.norm2(x))
        return x


class DecoderLayer(nn.Module):
    """Masked self-attention → cross-attention → FFN with pre-norm residuals."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.self_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.cross_attn = MultiHeadAttention(config.d_model, config.num_heads, config.dropout)
        self.ffn = FeedForward(config.d_model, config.d_ff, config.dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention (causal)
        residual = x
        x = self.norm1(x)
        self_out, _ = self.self_attn(x, x, x, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask)
        x = residual + self.dropout(self_out)

        # Cross-attention over encoder memory
        residual = x
        x = self.norm2(x)
        cross_out, _ = self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
        x = residual + self.dropout(cross_out)

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ─────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────

@Registry.register("model", "seq2seq_transformer")
class Seq2SeqTransformer(BaseModel):
    """
    Encoder-decoder Transformer for sequence-to-sequence tasks.

    batch contract (forward):
        input_ids             (B, S)   source token ids
        attention_mask        (B, S)   1 = real token, 0 = pad  [optional]
        decoder_input_ids     (B, T)   target token ids (BOS-prefixed, teacher-forced)
        decoder_attention_mask(B, T)   [optional]
        labels                (B, T)   target token ids shifted left (for CE loss)
    """

    config_class = TransformerConfig

    def __init__(self, config: TransformerConfig):
        super().__init__(config)
        self.src_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.tgt_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_id)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, config.max_seq_len, config.dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.num_decoder_layers)])
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self._init_weights()

    # ── Public API ──────────────────────────────

    def forward(self, batch: dict) -> dict:
        src = batch["input_ids"]
        tgt = batch["decoder_input_ids"]
        labels = batch["labels"]

        src_pad_mask = self._padding_mask(batch.get("attention_mask"), src.device)
        tgt_pad_mask = self._padding_mask(batch.get("decoder_attention_mask"), tgt.device)
        tgt_causal_mask = self._causal_mask(tgt.size(1), tgt.device)

        memory = self.encode(src, src_pad_mask)
        logits = self.decode(tgt, memory, tgt_causal_mask, tgt_pad_mask, src_pad_mask)

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            labels.reshape(-1),
            ignore_index=self.config.pad_id,
        )
        return {"loss": loss, "logits": logits}

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run the encoder stack.

        Args:
            src:                  (B, S) token ids
            src_key_padding_mask: (B, S) bool, True = pad position
        Returns:
            memory: (B, S, d_model)
        """
        x = self.pos_enc(self.src_embedding(src) * (self.config.d_model ** 0.5))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask)
        return x

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        tgt_key_padding_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        One teacher-forced decoder pass.

        Args:
            tgt:    (B, T) target token ids
            memory: (B, S, d_model) from encode()
        Returns:
            logits: (B, T, vocab_size)
        """
        x = self.pos_enc(self.tgt_embedding(tgt) * (self.config.d_model ** 0.5))
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask, tgt_key_padding_mask, memory_key_padding_mask)
        return self.output_projection(self.output_norm(x))

    def generate(
        self,
        src: torch.Tensor,
        max_length: int = 128,
        bos_id: int = 1,
        eos_id: int = 2,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Greedy autoregressive decoding.

        Args:
            src:    (B, S) source token ids
            max_length: maximum tokens to generate
            bos_id / eos_id: boundary token ids
        Returns:
            generated: (B, T) token ids including BOS, up to and including EOS
        """
        self.eval()
        with torch.no_grad():
            B = src.size(0)
            memory = self.encode(src, src_key_padding_mask)
            generated = src.new_full((B, 1), bos_id)
            done = src.new_zeros(B, dtype=torch.bool)

            for _ in range(max_length):
                tgt_mask = self._causal_mask(generated.size(1), src.device)
                logits = self.decode(generated, memory, tgt_mask)        # (B, T, vocab)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (B, 1)
                next_token[done] = eos_id
                generated = torch.cat([generated, next_token], dim=1)
                done |= next_token.squeeze(1).eq(eos_id)
                if done.all():
                    break

        return generated

    # ── Helpers ─────────────────────────────────

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(size: int, device: torch.device) -> torch.Tensor:
        """Additive upper-triangular mask that blocks future positions."""
        mask = torch.triu(torch.ones(size, size, device=device), diagonal=1)
        return mask.masked_fill(mask.bool(), float("-inf"))

    @staticmethod
    def _padding_mask(attention_mask: torch.Tensor | None, device: torch.device) -> torch.Tensor | None:
        """Convert 1/0 attention_mask → bool padding mask (True = ignore)."""
        if attention_mask is None:
            return None
        return attention_mask.eq(0).to(device)
