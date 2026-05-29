"""
Audio dataset for ASR / speech tasks.

Manifest format
───────────────
CSV  (default)   header row required:  audio_path,transcript
JSON             list of objects:      [{"audio_path": "...", "transcript": "..."}, ...]

Directory layout:
    <data_path>/train.csv   (or .json)
    <data_path>/eval.csv
    <data_path>/test.csv

Alternatively pass a single manifest file path directly.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import torch
import torchaudio

from common.dataset import BaseDataset
from common.registry import Registry
from text.data.tokenizer import CharTokenizer  # shared tokenizer
from .collator import AudioCollator
from .processor import LogMelConfig, LogMelExtractor


@Registry.register("dataset", "audio_ctc")
class AudioDataset(BaseDataset):
    """
    Loads (audio_path, transcript) pairs from a manifest file, extracts log mel
    features on the fly, and encodes transcripts with a CharTokenizer.

    dataset_config keys
    ───────────────────
    data_path         str    manifest directory or single file
    tokenizer_path    str    path to saved CharTokenizer JSON
    processor_config  dict   (optional) overrides for LogMelConfig fields
    max_duration_s    float  (default 30.0) samples longer than this are skipped

    __getitem__ returns
    ───────────────────
    features        (T, n_mels)   log mel filterbank features
    feature_length  int           real frame count (before batch padding)
    labels          (L,)          token ids
    label_length    int           real label length
    """

    def __init__(
        self,
        split: str,
        data_path: str,
        tokenizer_path: str,
        processor_config: dict | None = None,
        max_duration_s: float = 30.0,
    ):
        self.tokenizer = CharTokenizer.load(tokenizer_path)
        self.processor = LogMelExtractor(
            LogMelConfig.from_dict(processor_config) if processor_config else LogMelConfig()
        )
        sr = self.processor.config.sample_rate
        hop = self.processor.config.hop_length
        self.max_frames = int(max_duration_s * sr / hop)
        self.samples = self._resolve_and_load(Path(data_path), split)

    # ── Data loading ────────────────────────────────────────────────────

    def _resolve_and_load(self, base: Path, split: str) -> list[tuple[str, str]]:
        candidates = [
            base / f"{split}.csv",
            base / f"{split}.json",
            base,
        ]
        for path in candidates:
            if path.exists():
                return self._parse(path)
        raise FileNotFoundError(
            f"No manifest found for split='{split}' under {base}. "
            f"Expected one of: {[str(c) for c in candidates[:2]]}"
        )

    def _parse(self, path: Path) -> list[tuple[str, str]]:
        if path.suffix == ".json":
            with open(path, encoding="utf-8") as f:
                records = json.load(f)
            return [(r["audio_path"], r["transcript"]) for r in records]
        # Default: CSV
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return [(row["audio_path"], row["transcript"]) for row in reader]

    # ── Dataset interface ────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        audio_path, transcript = self.samples[idx]

        waveform, sample_rate = torchaudio.load(audio_path)
        features = self.processor(waveform, sample_rate)   # (T, n_mels)
        features = features[: self.max_frames]             # clip to max duration

        label_ids = self.tokenizer.encode(transcript)

        return {
            "features":       features,
            "feature_length": features.size(0),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
            "label_length":   len(label_ids),
        }

    @property
    def collate_fn(self) -> AudioCollator:
        return AudioCollator()
