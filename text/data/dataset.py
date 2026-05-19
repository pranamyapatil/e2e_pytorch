"""
Text seq2seq dataset.

Supported file formats
──────────────────────
TSV  (default)  one pair per line:  source<TAB>target
JSON            list of objects:    [{"src": "...", "tgt": "..."}, ...]

Directory layout expected by the split resolver:
    <data_path>/train.tsv   (or .json)
    <data_path>/eval.tsv
    <data_path>/test.tsv

Alternatively pass a single file path directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from common.dataset import BaseDataset
from common.registry import Registry
from .collator import Seq2SeqCollator
from .tokenizer import CharTokenizer


@Registry.register("dataset", "seq2seq")
class Seq2SeqDataset(BaseDataset):
    """
    Parallel text pair dataset for seq2seq tasks (translation, summarisation).

    dataset_config keys
    ───────────────────
    data_path      str   path to directory with {split}.tsv/.json, or a single file
    tokenizer_path str   path to a saved CharTokenizer JSON
    max_src_len    int   (default 256) clips source sequence
    max_tgt_len    int   (default 256) clips target sequence

    __getitem__ returns
    ───────────────────
    input_ids          (S,)  source token ids
    decoder_input_ids  (T,)  BOS + target[:-1]  (teacher-forced decoder input)
    labels             (T,)  target[1:] + EOS   (cross-entropy targets)

    Padding is handled by Seq2SeqCollator (returned via collate_fn).
    """

    def __init__(
        self,
        split: str,
        data_path: str,
        tokenizer_path: str,
        max_src_len: int = 256,
        max_tgt_len: int = 256,
    ):
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        self.pairs = self._resolve_and_load(Path(data_path), split)

    # ── Data loading ────────────────────────────────────────────────────

    def _resolve_and_load(self, base: Path, split: str) -> list[tuple[str, str]]:
        candidates = [
            base / f"{split}.tsv",
            base / f"{split}.json",
            base,           # single-file path passed directly
        ]
        for path in candidates:
            if path.exists():
                return self._parse(path)
        raise FileNotFoundError(
            f"No data file found for split='{split}' under {base}. "
            f"Expected one of: {[str(c) for c in candidates[:2]]}"
        )

    def _parse(self, path: Path) -> list[tuple[str, str]]:
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            return [(r["src"], r["tgt"]) for r in records]
        # Default: TSV
        pairs: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) == 2:
                    pairs.append((parts[0], parts[1]))
        return pairs

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        src_text, tgt_text = self.pairs[idx]
        tok = self.tokenizer

        src_ids = tok.encode(src_text)[: self.max_src_len]

        # Full target with BOS and EOS, then split into decoder_input / labels
        tgt_ids = tok.encode(tgt_text, add_bos=True, add_eos=True)[: self.max_tgt_len + 1]
        decoder_input = tgt_ids[:-1]   # BOS + tgt (without last)
        labels = tgt_ids[1:]           # tgt + EOS (without BOS)

        return {
            "input_ids":         torch.tensor(src_ids,      dtype=torch.long),
            "decoder_input_ids": torch.tensor(decoder_input, dtype=torch.long),
            "labels":            torch.tensor(labels,        dtype=torch.long),
        }

    @property
    def collate_fn(self) -> Seq2SeqCollator:
        return Seq2SeqCollator(pad_id=self.tokenizer.pad_id)
