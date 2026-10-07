"""Task heads attached on top of the swappable encoder.

  * CTCHead      — character CTC for supervised pretrain/finetune (learns the
                   acoustic representation on LibriSpeech, adapts it on TORGO).
  * EmbeddingProjector — maps encoder frames to the SHARED embedding dim the
                   Prototypical head consumes, so all four encoders are
                   compared at an identical embedding interface (spec Section 4).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pool(hidden: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Masked mean over time -> [B, D] utterance embedding."""
    mask = (torch.arange(hidden.size(1), device=hidden.device)[None, :]
            < lengths[:, None]).unsqueeze(-1).float()
    summed = (hidden * mask).sum(dim=1)
    return summed / mask.sum(dim=1).clamp_min(1.0)


class CTCHead(nn.Module):
    def __init__(self, in_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, vocab_size)
        self.ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, hidden, hid_lengths, targets=None, target_lengths=None):
        logits = self.proj(hidden)                       # [B, T, V]
        if targets is None:
            return logits, None
        log_probs = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, V]
        loss = self.ctc(log_probs, targets, hid_lengths, target_lengths)
        return logits, loss


class EmbeddingProjector(nn.Module):
    """Frozen-encoder frames -> single L2-normalised utterance embedding at the
    shared embedding dim, for the Prototypical Network."""

    def __init__(self, in_dim: int, embed_dim: int = 256, normalize: bool = True):
        super().__init__()
        self.proj = nn.Linear(in_dim, embed_dim)
        self.normalize = normalize

    def forward(self, hidden, lengths):
        return self.from_pooled(mean_pool(hidden, lengths))

    def from_pooled(self, pooled: torch.Tensor) -> torch.Tensor:
        """Project an already-pooled [B, in_dim] utterance vector to the
        embedding space. Used by episodic meta-training, which caches the frozen
        encoder's pooled features once and then only trains this linear head."""
        emb = self.proj(pooled)
        if self.normalize:
            emb = F.normalize(emb, dim=-1)
        return emb
