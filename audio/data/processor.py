"""
Acoustic feature extraction: raw waveform → log mel filterbanks.

LogMelExtractor is the default processor. It wraps torchaudio.transforms
and handles resampling, mono conversion, log compression, and per-utterance
normalisation in one callable.

Output shape: (T_frames, n_mels)
    T_frames ≈ ceil(num_samples / hop_length)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torchaudio

from common.config import BaseConfig


@dataclass
class LogMelConfig(BaseConfig):
    sample_rate: int = 16_000
    n_fft: int = 512
    win_length: int = 400        # 25 ms at 16 kHz
    hop_length: int = 160        # 10 ms at 16 kHz — standard ASR frame shift
    n_mels: int = 80
    f_min: float = 0.0
    f_max: float = 8_000.0
    normalize: bool = True       # per-utterance zero-mean / unit-variance


class LogMelExtractor:
    """
    Converts a raw audio waveform to log mel filterbank features.

    Usage:
        extractor = LogMelExtractor(LogMelConfig())
        features = extractor(waveform, sample_rate)  # (T, n_mels)
    """

    def __init__(self, config: LogMelConfig | None = None):
        self.config = config or LogMelConfig()
        self._mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            win_length=self.config.win_length,
            hop_length=self.config.hop_length,
            n_mels=self.config.n_mels,
            f_min=self.config.f_min,
            f_max=self.config.f_max,
        )

    def __call__(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """
        Args:
            waveform:    (C, T) or (T,) raw audio tensor
            sample_rate: source sample rate (resampled if != config.sample_rate)
        Returns:
            log_mel: (T_frames, n_mels)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)                         # (1, T)

        if sample_rate != self.config.sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, orig_freq=sample_rate, new_freq=self.config.sample_rate
            )

        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)            # stereo → mono

        mel = self._mel(waveform)                                    # (1, n_mels, T)
        log_mel = torch.log(mel.clamp(min=1e-9)).squeeze(0).T        # (T, n_mels)

        if self.config.normalize:
            mean = log_mel.mean()
            std = log_mel.std().clamp(min=1e-9)
            log_mel = (log_mel - mean) / std

        return log_mel

    def __repr__(self) -> str:
        c = self.config
        return (
            f"LogMelExtractor(sr={c.sample_rate}, n_mels={c.n_mels}, "
            f"hop={c.hop_length}, win={c.win_length})"
        )
