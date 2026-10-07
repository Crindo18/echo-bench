"""Shared acoustic front-end.

Per the ECHO spec (Section 4), a single log-Mel filterbank front-end with
identical window/hop/mel-bin settings is held constant across the three
supervised encoders (Conformer, E-Branchformer, Zipformer).

DPWavLM is the documented exception: as an SSL model its "front-end" is its
own inherited convolutional feature extractor, which is part of the *origin*
variable under study. It therefore consumes raw waveform, not log-Mel. The
harness normalises the four models at the two points the spec actually
matches them on: the embedding interface into the Prototypical head, and the
INT8-ONNX budget / runtime. See README section "The front-end caveat".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torchaudio


class LogMelFrontend(nn.Module):
    """log-Mel filterbank + per-utterance CMVN. 16 kHz in, [B, T, n_mels] out."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,        # 25 ms window
        hop_length: int = 160,   # 10 ms hop
        win_length: int = 400,
        n_mels: int = 80,
        f_min: float = 20.0,
        f_max: float | None = 7600.0,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=2.0,
            center=True,
        )

    @torch.no_grad()
    def _lengths_to_frames(self, wav_lengths: torch.Tensor) -> torch.Tensor:
        # center=True padding => n_frames = 1 + floor(L / hop)
        return torch.div(wav_lengths, self.hop_length, rounding_mode="floor") + 1

    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor | None = None):
        """wav: [B, num_samples]  ->  feats: [B, T, n_mels], feat_lengths: [B]."""
        mel = self.melspec(wav)                      # [B, n_mels, T]
        feats = torch.log(mel.clamp_min(1e-10))      # log compression
        feats = feats.transpose(1, 2)                # [B, T, n_mels]
        T = feats.size(1)

        if wav_lengths is None:
            feat_lengths = torch.full(
                (feats.size(0),), T, dtype=torch.long, device=feats.device
            )
        else:
            feat_lengths = self._lengths_to_frames(wav_lengths).clamp_max(T)

        # Per-utterance mean/var normalisation (CMVN), computed over VALID frames
        # only. Collate zero-pads each batch to its longest clip; log-Mel of that
        # padding is a large constant (~log 1e-10), so including it would make a
        # short clip's normalisation depend on its batch's max length (and thus on
        # batch size). Mask the padding so every utterance normalises the same way
        # regardless of its neighbours.
        mask = (torch.arange(T, device=feats.device)[None, :]
                < feat_lengths[:, None]).unsqueeze(-1).to(feats.dtype)   # [B, T, 1]
        n = mask.sum(dim=1, keepdim=True).clamp_min(1.0)                 # [B, 1, 1]
        mean = (feats * mask).sum(dim=1, keepdim=True) / n
        var = (((feats - mean) ** 2) * mask).sum(dim=1, keepdim=True) / n
        std = var.sqrt().clamp_min(1e-5)
        feats = (feats - mean) / std
        feats = feats * mask                          # zero padded frames post-norm
        return feats, feat_lengths


class SpecAugment(nn.Module):
    """Frequency + time masking. Same augmentation policy applies to all models
    (spec Section 6, "Data hygiene"). Active in train() mode only."""

    def __init__(
        self,
        freq_mask_param: int = 27,
        time_mask_param: int = 100,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
        max_time_frac: float = 0.2,
    ):
        super().__init__()
        self.n_freq_masks = n_freq_masks
        self.n_time_masks = n_time_masks
        self.max_time_frac = max_time_frac
        self.freq = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self.time = torchaudio.transforms.TimeMasking(
            time_mask_param, p=max_time_frac
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return feats
        x = feats.transpose(1, 2)  # [B, n_mels, T] for torchaudio maskers
        for _ in range(self.n_freq_masks):
            x = self.freq(x)
        for _ in range(self.n_time_masks):
            x = self.time(x)
        return x.transpose(1, 2)
