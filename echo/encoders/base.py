"""Common encoder interface + shared components.

Every encoder in the benchmark implements the same contract so the harness can
swap the encoder and hold everything else fixed (spec Section 3). The contract:

    forward(feats, feat_lengths) -> (hidden[B, T', D], out_lengths[B])

where `feats` is [B, T, n_mels] from the shared front-end (supervised models)
or raw waveform (DPWavLM, which overrides `accepts_waveform`). `D` is the
encoder output dim, exposed as `.out_dim`, and is projected to a shared
embedding dim by the Prototypical head so all four are comparable there.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


def count_parameters(module: nn.Module, trainable_only: bool = True) -> int:
    return sum(
        p.numel() for p in module.parameters() if (p.requires_grad or not trainable_only)
    )


def make_pad_mask(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    """Return a [B, T] boolean mask that is True at PADDED positions."""
    if max_len is None:
        max_len = int(lengths.max().item())
    idx = torch.arange(max_len, device=lengths.device).unsqueeze(0)  # [1, T]
    return idx >= lengths.unsqueeze(1)                                # [B, T]


class EncoderBase(nn.Module, ABC):
    """Abstract acoustic encoder producing frame-level embeddings."""

    #: DPWavLM sets this True: it consumes raw 16 kHz waveform, not log-Mel.
    accepts_waveform: bool = False

    @property
    @abstractmethod
    def out_dim(self) -> int:
        ...

    @abstractmethod
    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        """feats: [B, T, n_mels] (or [B, num_samples] if accepts_waveform).
        Returns (hidden[B, T', out_dim], out_lengths[B])."""
        ...

class Conv2dSubsampling(nn.Module):
    """4x time downsampling via two stride-2 convolutions, projecting the
    log-Mel front-end to `d_model`. Shared by the supervised encoders so the
    subsampling stem is not a confound between architectures (spec Section 6)."""

    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(d_model, d_model, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        # after two stride-2 convs with padding=1, freq dim -> ceil(n_mels/4)
        freq_out = (((n_mels + 1) // 2) + 1) // 2
        self.out = nn.Linear(d_model * freq_out, d_model)
        self.dropout = nn.Dropout(dropout)

    def _sub_len(self, lengths: torch.Tensor) -> torch.Tensor:
        # padding=1, kernel=3, stride=2 -> floor((L + 2*1 - 3)/2) + 1 = floor((L-1)/2)+1
        for _ in range(2):
            lengths = torch.div(lengths - 1, 2, rounding_mode="floor") + 1
        return lengths

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        x = feats.unsqueeze(1)                       # [B, 1, T, n_mels]
        x = self.conv(x)                             # [B, C, T', F']
        b, c, t, f = x.shape
        x = x.transpose(1, 2).contiguous().view(b, t, c * f)
        x = self.dropout(self.out(x))                # [B, T', d_model]
        return x, self._sub_len(lengths).clamp_max(t)
