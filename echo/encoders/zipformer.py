"""Zipformer encoder — the EFFICIENT RECENT ENCODER (spec Section 2).

Downsampling U-Net hybrid (Yao et al., 2024). Most compute happens at
downsampled frame rates, giving roughly half the FLOPs/memory of a
Branchformer-class encoder at matched quality. The block reuses one set of
attention weights across two attention applications (a key Zipformer trick).

Faithful PyTorch reimplementation including the training machinery Zipformer
relies on: it trains with **ScaledAdam** (`echo/train/scaled_adam.py`) and each
block carries the **Balancer** and **Whitener** regularisers
(`echo/encoders/regularizers.py`), alongside BiasNorm and the SwooshR
activation. The regularisers and optimizer are training-only — identity at
inference, zero added parameters or footprint — so the deployed model is pure
standard ops (Linear/Conv1d/matmul) that quantise and export to INT8 ONNX
cleanly on ARM, the deployability property that put Zipformer in the set over
Branchformer-class encoders (spec Section 2).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .base import Conv2dSubsampling, EncoderBase, make_pad_mask
from .regularizers import Balancer, Whitener


def swoosh_r(x: torch.Tensor) -> torch.Tensor:
    # log1p(exp(x-1)) is softplus(x-1); computing exp(x-1) directly overflows to
    # +inf for x > ~88 in fp32 (-> inf -> NaN that cascades through the net). F.softplus
    # is the numerically stable form (returns x for large x), so this is identical
    # for normal values but can't overflow. This was the Zipformer sudden-NaN cause.
    return F.softplus(x - 1.0) - 0.08 * x - 0.313261687


class SwooshR(nn.Module):
    def forward(self, x):
        return swoosh_r(x)


class BiasNorm(nn.Module):
    """Zipformer BiasNorm: RMS-normalise after subtracting a learned channel
    bias, then rescale by a learned positive scale. Retains a length cue that
    plain LayerNorm discards."""

    def __init__(self, d_model: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.log_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x - self.bias).pow(2).mean(dim=-1, keepdim=True).add(1e-6).sqrt()
        return x / rms * torch.exp(self.log_scale)


class FeedForwardSwoosh(nn.Module):
    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.act = SwooshR()
        self.drop = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.lin2(self.drop(self.act(self.lin1(x))))


class RelPositionEmbedding(nn.Module):
    def __init__(self, d_model, max_len=2000):
        super().__init__()
        self.d_model = d_model
        self._pe = None
        self._build(max_len)

    def _build(self, length):
        pe = torch.zeros(2 * length - 1, self.d_model)
        pos = torch.arange(length - 1, -length, -1.0).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2).float()
            * (-math.log(10000.0) / self.d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self._pe = pe.unsqueeze(0)

    def forward(self, t, device):
        if (
            self._pe is None
            or self._pe.size(1) < 2 * t - 1
            or self._pe.device != device
        ):
            self._build(max(t, 2))
            self._pe = self._pe.to(device)
        center = self._pe.size(1) // 2
        return self._pe[:, center - t + 1 : center + t]


class AttentionWeights(nn.Module):
    """Computes shared relative-position attention weights [B, h, T, T] once."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(dropout)

    def _rel_shift(self, x):
        b, h, t1, t2 = x.shape
        zero = x.new_zeros(b, h, t1, 1)
        x = torch.cat([zero, x], dim=-1).view(b, h, t2 + 1, t1)
        return x[:, :, 1:].view(b, h, t1, t2)[:, :, :, : t2 // 2 + 1]

    def forward(self, x, pos_emb, pad_mask):
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        ac = torch.matmul(q_u, k.transpose(-2, -1))
        bd = self._rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))
        scores = (ac + bd) / math.sqrt(self.d_k)
        if pad_mask is not None:
            scores = scores.masked_fill(
                pad_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )
        return self.dropout(torch.softmax(scores, dim=-1))


class SelfAttention(nn.Module):
    """Applies pre-computed attention weights to a value projection."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.h = n_heads
        self.d_k = d_model // n_heads
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_weights):
        b, t, _ = x.shape
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        out = torch.matmul(attn_weights, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.dropout(self.linear_out(out))


class NonlinAttention(nn.Module):
    """Reuses the shared weights on a gated non-linear value path."""

    def __init__(self, d_model, dropout):
        super().__init__()
        self.proj_in = nn.Linear(d_model, 3 * d_model // 2)
        self.proj_out = nn.Linear(d_model // 2, d_model)
        self.act = nn.Tanh()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_weights):
        b, t, _ = x.shape
        y = self.proj_in(x)
        a, gate, val = y.split(x.size(-1) // 2, dim=-1)
        val = self.act(a) * val  # gated value, [B, T, D/2]
        head = attn_weights[:, :1]  # use first head's weights
        val = torch.matmul(head.squeeze(1), val)  # [B, T, D/2]
        val = val * torch.sigmoid(gate)
        return self.dropout(self.proj_out(val))


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel_size, dropout):
        super().__init__()
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size,
            padding=(kernel_size - 1) // 2,
            groups=d_model,
        )
        self.act = SwooshR()
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        x = x.transpose(1, 2)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = nn.functional.glu(self.pointwise1(x), dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.act(self.depthwise(x))
        x = self.pointwise2(x)
        return self.dropout(x).transpose(1, 2)


class ZipformerBlock(nn.Module):
    """FFN -> attn-weights(once) -> {SelfAttn, NonlinAttn} -> Conv -> SelfAttn
    -> FFN. Two attention applications share one weight computation."""

    def __init__(self, d_model, d_ff, n_heads, kernel_size, dropout):
        super().__init__()
        self.ffn1 = FeedForwardSwoosh(d_model, d_ff, dropout)
        self.attn_weights = AttentionWeights(d_model, n_heads, dropout)
        self.self_attn1 = SelfAttention(d_model, n_heads, dropout)
        self.nonlin_attn = NonlinAttention(d_model, dropout)
        self.conv = ConvModule(d_model, kernel_size, dropout)
        self.self_attn2 = SelfAttention(d_model, n_heads, dropout)
        self.ffn2 = FeedForwardSwoosh(d_model, d_ff, dropout)
        self.norm = BiasNorm(d_model)
        # training-only regularisers (identity at inference, 0 params/footprint)
        self.balancer = Balancer(d_model, channel_dim=-1)
        self.whitener = Whitener(channel_dim=-1)

    def forward(self, x, pos_emb, pad_mask):
        x = x + self.ffn1(x)
        weights = self.attn_weights(x, pos_emb, pad_mask)
        x = x + self.self_attn1(x, weights)
        x = x + self.nonlin_attn(x, weights)
        x = x + self.conv(x, pad_mask)
        x = x + self.self_attn2(x, weights)
        x = x + self.ffn2(x)
        x = self.balancer(x)  # keep per-channel mean/rms in range (train)
        x = self.norm(x)
        return self.whitener(x)  # discourage feature collapse (train)


class SimpleDownsample(nn.Module):
    """Downsample time by `factor` with a learned weighted average per group."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor
        self.weight = nn.Parameter(torch.ones(factor) / factor)

    def forward(self, x, lengths):
        b, t, d = x.shape
        pad = (self.factor - t % self.factor) % self.factor
        if pad:
            x = torch.cat([x, x.new_zeros(b, pad, d)], dim=1)
        x = x.view(b, x.size(1) // self.factor, self.factor, d)
        w = torch.softmax(self.weight, dim=0).view(1, 1, self.factor, 1)
        x = (x * w).sum(dim=2)
        lengths = torch.div(
            lengths + self.factor - 1, self.factor, rounding_mode="floor"
        )
        return x, lengths.clamp_max(x.size(1))


class SimpleUpsample(nn.Module):
    """Upsample time by repeating each frame `factor` times, cropped to target."""

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, x, target_len):
        x = x.repeat_interleave(self.factor, dim=1)
        return x[:, :target_len]


class ZipformerEncoder(EncoderBase):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        d_ff: int = 768,
        n_heads: int = 4,
        kernel_size: int = 15,
        downsample_factors=(1, 2, 4, 2, 1),
        blocks_per_stack=(1, 2, 2, 2, 1),  # matched ~15M budget (spec Section 4)
        dropout: float = 0.1,
    ):
        super().__init__()
        assert len(downsample_factors) == len(blocks_per_stack)
        self._out_dim = d_model
        self.subsampling = Conv2dSubsampling(n_mels, d_model, dropout)
        self.pos_enc = RelPositionEmbedding(d_model)
        self.downsamplers = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.stacks = nn.ModuleList()
        self.factors = downsample_factors
        for factor, n_blocks in zip(downsample_factors, blocks_per_stack):
            self.downsamplers.append(SimpleDownsample(factor) if factor > 1 else None)
            self.upsamplers.append(SimpleUpsample(factor) if factor > 1 else None)
            self.stacks.append(
                nn.ModuleList(
                    [
                        ZipformerBlock(d_model, d_ff, n_heads, kernel_size, dropout)
                        for _ in range(n_blocks)
                    ]
                )
            )
        self.final_norm = BiasNorm(d_model)

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x, lengths = self.subsampling(feats, feat_lengths)
        base_len = x.size(1)
        for i, stack in enumerate(self.stacks):
            residual, res_len = x, x.size(1)
            down = self.downsamplers[i]
            if down is not None:
                x, lengths = down(x, lengths)
            pad_mask = make_pad_mask(lengths, x.size(1))
            pos_emb = self.pos_enc(x.size(1), x.device)
            for block in stack:
                x = block(x, pos_emb, pad_mask)
            up = self.upsamplers[i]
            if up is not None:
                x = up(x, res_len)
                lengths = torch.div(
                    (lengths * self.factors[i]).clamp_max(res_len),
                    1,
                    rounding_mode="floor",
                )
            x = x + residual  # U-Net skip / bypass
            lengths = lengths.clamp_max(base_len)
        pad_mask = make_pad_mask(lengths, x.size(1))
        x = self.final_norm(x).masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x, lengths
