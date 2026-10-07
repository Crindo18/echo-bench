"""Shared training harness.

The whole point of the benchmark is that only the encoder changes (spec
Section 3). ``AcousticModel`` encodes that invariant: it wires the shared
log-Mel front-end (or raw-waveform path for DPWavLM), the swappable encoder,
and a task head into one module, so pretrain / finetune / eval all treat the
four models identically.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from ..features import LogMelFrontend, SpecAugment
from ..heads import CTCHead, EmbeddingProjector
from ..encoders import EncoderBase


# --------------------------------------------------------------------------- #
# Reproducibility (spec Section 6: multi-seed for convergence/variance)
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# The single swappable-encoder model
# --------------------------------------------------------------------------- #
class AcousticModel(nn.Module):
    """front-end (shared) -> encoder (swapped) -> {CTC head | embedding}.

    ``encoder.accepts_waveform`` decides whether the log-Mel front-end runs.
    For DPWavLM the front-end is bypassed and the raw waveform goes straight
    into the SSL conv extractor — the declared 'origin' front-end exception
    (spec Section 4).
    """

    def __init__(
        self,
        encoder: EncoderBase,
        vocab_size: int,
        n_mels: int = 80,
        embed_dim: int = 256,
        use_specaugment: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.accepts_waveform = encoder.accepts_waveform
        self.frontend = None if self.accepts_waveform else LogMelFrontend(n_mels=n_mels)
        self.specaug = SpecAugment() if (use_specaugment and not self.accepts_waveform) else None
        self.ctc_head = CTCHead(encoder.out_dim, vocab_size)
        self.projector = EmbeddingProjector(encoder.out_dim, embed_dim)

    # -- feature path ------------------------------------------------------- #
    def encode(self, wav: torch.Tensor, wav_lengths: torch.Tensor):
        if self.accepts_waveform:
            return self.encoder(wav, wav_lengths)
        feats, feat_lengths = self.frontend(wav, wav_lengths)
        if self.specaug is not None and self.training:
            feats = self.specaug(feats)
        return self.encoder(feats, feat_lengths)

    # -- CTC (pretrain / finetune) ------------------------------------------ #
    def forward_ctc(self, wav, wav_lengths, targets=None, target_lengths=None):
        hidden, out_lengths = self.encode(wav, wav_lengths)
        return self.ctc_head(hidden, out_lengths, targets, target_lengths)

    # -- embedding (few-shot stage) ----------------------------------------- #
    @torch.no_grad()
    def embed(self, wav, wav_lengths):
        hidden, out_lengths = self.encode(wav, wav_lengths)
        return self.projector(hidden, out_lengths)

    def freeze_encoder(self) -> None:
        for p in self.encoder.parameters():
            p.requires_grad = False
        if self.frontend is not None:
            for p in self.frontend.parameters():
                p.requires_grad = False


# --------------------------------------------------------------------------- #
# Checkpoint helpers
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str, model: nn.Module, meta: Optional[Dict] = None,
                    optimizer=None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {"model": model.state_dict(), "meta": meta or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(path: str, model: nn.Module, map_location="cpu",
                    strict: bool = True) -> Dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model"], strict=strict)
    return ckpt.get("meta", {})


def build_optimizer(model: nn.Module, encoder_name: str, lr: float = None):
    """Pick the optimizer each encoder is designed to train with.

    Zipformer uses **ScaledAdam** (its authors' optimizer — plain Adam trains it
    poorly); everything else uses AdamW. Returns (optimizer, is_scaled) so the
    caller can set a sensible default LR (ScaledAdam runs at a much higher LR
    than AdamW). Training-only: the choice never affects the deployed model.
    """
    if encoder_name == "zipformer":
        from .scaled_adam import ScaledAdam
        return ScaledAdam(model.parameters(), lr=lr or 0.04,
                          betas=(0.9, 0.98), weight_decay=1e-4), True
    return torch.optim.AdamW(model.parameters(), lr=lr or 3e-4,
                             weight_decay=1e-2, betas=(0.9, 0.98)), False


def load_init_weights(path: str, model: "AcousticModel", map_location="cpu") -> Dict:
    """Load a *stage-1* checkpoint into an ``AcousticModel`` for finetuning.

    Handles both checkpoint layouts the pipeline produces:
      * a full ``AcousticModel`` checkpoint (keys ``encoder.*``, ``ctc_head.*``,
        ``projector.*``) — e.g. the supervised pretrain output; and
      * a bare **encoder-only** checkpoint (keys ``feature_extractor.*`` /
        ``layers.*`` with no ``encoder.`` prefix) — e.g. the DPWavLM
        distill+prune output, which saves the raw student.

    It matches keys by name+shape, loads them into the right submodule, and
    **raises** if nothing loaded (so a prefix/architecture mismatch can never
    silently leave the encoder at random init — the failure mode we are guarding
    against). Returns the checkpoint meta.
    """
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    sd = ckpt["model"]
    full_keys = set(model.state_dict().keys())
    if any(k in full_keys for k in sd):                    # full AcousticModel ckpt
        loaded, _ = model.load_state_dict(sd, strict=False)
        n_loaded = len(full_keys) - len(loaded)
        print(f"[load_init] full checkpoint: loaded {n_loaded}/{len(full_keys)} tensors")
        return ckpt.get("meta", {})

    # encoder-only checkpoint -> load into model.encoder
    enc_sd = model.encoder.state_dict()
    stripped = {(k[len("encoder."):] if k.startswith("encoder.") else k): v
                for k, v in sd.items()}
    matched = {k: v for k, v in stripped.items()
               if k in enc_sd and enc_sd[k].shape == v.shape}
    if not matched:
        raise RuntimeError(
            f"load_init_weights: no tensors in {path} matched the model. "
            f"Checkpoint keys look like {list(sd)[:2]}; encoder expects "
            f"{list(enc_sd)[:2]}. Architecture/config mismatch?")
    model.encoder.load_state_dict(matched, strict=False)
    print(f"[load_init] encoder-only checkpoint: loaded {len(matched)}/{len(enc_sd)} "
          f"encoder tensors")
    return ckpt.get("meta", {})


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.total += float(val) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)
