"""
Character-level tokenizer.

Intentionally simple — every character is a token, vocab is built from raw text.
Good for learning because the encoding is fully transparent and requires no
external library. Swap for a BPE tokenizer once the architecture is understood.

Special tokens always occupy fixed ids:
    0 → <pad>   1 → <bos>   2 → <eos>   3 → <unk>
"""

from __future__ import annotations

import json
from pathlib import Path


class CharTokenizer:

    SPECIAL = ["<pad>", "<bos>", "<eos>", "<unk>"]

    def __init__(self, vocab: dict[str, int]):
        self.vocab = vocab
        self.inv_vocab: dict[int, str] = {v: k for k, v in vocab.items()}

    # ── Construction ────────────────────────────────────────────────────

    @classmethod
    def build(cls, texts: list[str]) -> CharTokenizer:
        """Build a vocab from a flat list of strings."""
        chars = sorted({c for text in texts for c in text})
        vocab: dict[str, int] = {tok: i for i, tok in enumerate(cls.SPECIAL)}
        for c in chars:
            if c not in vocab:
                vocab[c] = len(vocab)
        return cls(vocab)

    def save(self, path: str | Path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))

    # ── Encoding / decoding ─────────────────────────────────────────────

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        ids = [self.vocab.get(c, self.unk_id) for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        special_set = set(self.SPECIAL)
        tokens = [self.inv_vocab.get(i, "<unk>") for i in ids]
        if skip_special:
            tokens = [t for t in tokens if t not in special_set]
        return "".join(tokens)

    def batch_encode(self, texts: list[str], **kwargs) -> list[list[int]]:
        return [self.encode(t, **kwargs) for t in texts]

    # ── Properties ──────────────────────────────────────────────────────

    @property
    def pad_id(self) -> int: return self.vocab["<pad>"]
    @property
    def bos_id(self) -> int: return self.vocab["<bos>"]
    @property
    def eos_id(self) -> int: return self.vocab["<eos>"]
    @property
    def unk_id(self) -> int: return self.vocab["<unk>"]
    @property
    def vocab_size(self) -> int: return len(self.vocab)

    def __len__(self) -> int: return self.vocab_size
    def __repr__(self) -> str: return f"CharTokenizer(vocab_size={self.vocab_size})"
