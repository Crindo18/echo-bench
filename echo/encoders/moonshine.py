"""Moonshine encoder — a 5th benchmark model (spec Section 2, "build small").

Moonshine (Jeffries et al., 2024, Useful Sensors) is an edge-optimised ASR
encoder-decoder. We take only its **encoder** as a candidate first ML stage.
Its distinctive design choices, all reproduced here:

  * **Raw waveform in, no Mel front-end.** A 3-conv stem (strides 64/3/2)
    compresses 16 kHz audio by 384x to a ~41.7 Hz frame sequence — Moonshine
    does no hand-engineered feature extraction. This makes Moonshine, like
    DPWavLM, an "origin" exception to the benchmark's shared log-Mel front-end
    (spec Section 4): it consumes waveform and its stem *is* the front-end.
  * **Rotary Position Embeddings (RoPE)** at every encoder layer (partial, 0.9
    of the head dim), rather than absolute positions — this is what lets
    Moonshine encode variable-length audio without padding.
  * Pre-norm Transformer blocks, GELU MLP, bias-free projections and norms.

Faithful PyTorch reimplementation (not the HF checkpoint) so all five models
share one swappable interface. Moonshine-tiny's encoder is only ~7.7M params, so
— exactly as with the other "build small" encoders — we size it up to the
matched ~15M budget rather than porting a released checkpoint.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import EncoderBase, make_pad_mask


# ----------------------------------------------------------------------------- #
# Rotary position embedding (partial)
# ----------------------------------------------------------------------------- #
class RotaryEmbedding(nn.Module):
    """Precomputes cos/sin tables for the rotary dims; partial RoPE rotates the
    first ``dim`` (even) channels of each head and passes the rest through."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        assert dim % 2 == 0, "rotary dim must be even"
        self.dim = dim
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device):
        pos = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(pos, self.inv_freq.to(device))     # [T, dim/2]
        emb = torch.cat([freqs, freqs], dim=-1)                 # [T, dim]
        return emb.cos(), emb.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(q, k, cos, sin):
    # q, k: [B, h, T, rot]; cos, sin: [T, rot]
    cos = cos[None, None]
    sin = sin[None, None]
    q = q * cos + _rotate_half(q) * sin
    k = k * cos + _rotate_half(k) * sin
    return q, k


# ----------------------------------------------------------------------------- #
# Transformer block
# ----------------------------------------------------------------------------- #
class MoonshineAttention(nn.Module):
    def __init__(self, d_model, n_heads, rotary_dim, dropout):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.rot = rotary_dim
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, pad_mask):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.d_k).transpose(1, 2)   # [B,h,T,dk]
        k = self.k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        # partial RoPE: rotate the first `rot` dims, pass the rest unchanged
        q_rot, q_pass = q[..., : self.rot], q[..., self.rot:]
        k_rot, k_pass = k[..., : self.rot], k[..., self.rot:]
        q_rot, k_rot = _apply_rope(q_rot, k_rot, cos, sin)
        q = torch.cat([q_rot, q_pass], dim=-1)
        k = torch.cat([k_rot, k_pass], dim=-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2),
                                        float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.o(ctx)


class MoonshineMLP(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.gelu(self.fc1(x))))


class MoonshineEncoderLayer(nn.Module):
    """Pre-norm block: x += attn(LN(x)); x += mlp(LN(x)). LayerNorm is bias-free,
    matching Moonshine."""

    def __init__(self, d_model, n_heads, d_ff, rotary_dim, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, bias=False)
        self.attn = MoonshineAttention(d_model, n_heads, rotary_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model, bias=False)
        self.mlp = MoonshineMLP(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cos, sin, pad_mask):
        x = x + self.dropout(self.attn(self.norm1(x), cos, sin, pad_mask))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


# ----------------------------------------------------------------------------- #
# Conv stem (front-end) + encoder
# ----------------------------------------------------------------------------- #
class MoonshinePreprocessor(nn.Module):
    """3-conv stem, raw waveform -> [B, T, d_model] at ~41.7 Hz (384x stride)."""

    # (kernel, stride) per conv, from the Moonshine paper
    CONV = [(127, 64), (7, 3), (3, 2)]

    def __init__(self, d_model: int):
        super().__init__()
        self.conv1 = nn.Conv1d(1, d_model, 127, stride=64, bias=False)
        self.groupnorm = nn.GroupNorm(1, d_model, eps=1e-5)
        self.conv2 = nn.Conv1d(d_model, 2 * d_model, 7, stride=3)
        self.conv3 = nn.Conv1d(2 * d_model, d_model, 3, stride=2)

    def _out_len(self, lengths):
        for kernel, stride in self.CONV:
            lengths = torch.div(lengths - kernel, stride,
                                rounding_mode="floor") + 1
        return lengths.clamp_min(0)

    def forward(self, wav, wav_lengths):
        x = wav.unsqueeze(1)                      # [B, 1, samples]
        x = torch.tanh(self.conv1(x))
        x = self.groupnorm(x)
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))
        x = x.transpose(1, 2)                     # [B, T, d_model]
        return x, self._out_len(wav_lengths)


class MoonshineEncoder(EncoderBase):
    """Moonshine-style raw-waveform encoder, sized to the ~15M matched budget."""

    accepts_waveform = True

    def __init__(
        self,
        d_model: int = 384,
        n_heads: int = 8,
        n_layers: int = 8,
        d_ff: int = 1120,          # sized to the matched ~14.6M budget
        partial_rotary_factor: float = 0.9,
        rope_theta: float = 10000.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._out_dim = d_model
        head_dim = d_model // n_heads
        # rotary dim = even part of (head_dim * partial_rotary_factor)
        rotary_dim = 2 * int(head_dim * partial_rotary_factor / 2)
        self.preprocessor = MoonshinePreprocessor(d_model)
        self.rotary = RotaryEmbedding(rotary_dim, rope_theta)
        self.layers = nn.ModuleList(
            [MoonshineEncoderLayer(d_model, n_heads, d_ff, rotary_dim, dropout)
             for _ in range(n_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model, bias=False)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor):
        x, lengths = self.preprocessor(wav, wav_lengths)
        cos, sin = self.rotary(x.size(1), x.device)
        pad_mask = make_pad_mask(lengths, x.size(1))    # [B, T] True at pads
        for layer in self.layers:
            x = layer(x, cos, sin, pad_mask)
        x = self.final_norm(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x, lengths
