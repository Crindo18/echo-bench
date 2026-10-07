"""E-Branchformer encoder — the PROPOSED BASELINE (spec Section 2).

Parallel-branch attention||convolution hybrid (Kim et al., 2023). A global
branch (relative-position MHSA) and a local branch (convolutional gating MLP,
cgMLP) run in parallel and are fused by a depthwise-conv merge module.
Every other model in the benchmark is measured against this one.

Faithful PyTorch reimplementation, built small to the ~15M matched budget.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Conv2dSubsampling, EncoderBase, make_pad_mask
from .modules import (
    FeedForward,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)


class ConvolutionalSpatialGatingUnit(nn.Module):
    """CSGU: split channels, gate one half by a depthwise conv of the other."""

    def __init__(self, d_ff: int, kernel_size: int, dropout: float):
        super().__init__()
        assert d_ff % 2 == 0
        half = d_ff // 2
        self.norm = nn.LayerNorm(half)
        self.conv = nn.Conv1d(
            half, half, kernel_size,
            padding=(kernel_size - 1) // 2, groups=half,
        )
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.conv.weight, std=1e-6)
        nn.init.ones_(self.conv.bias)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None):
        a, b = x.chunk(2, dim=-1)                 # each [B, T, half]
        b = self.norm(b).transpose(1, 2)          # [B, half, T]
        if pad_mask is not None:
            b = b.masked_fill(pad_mask.unsqueeze(1), 0.0)
        b = self.conv(b).transpose(1, 2)          # [B, T, half]
        return self.dropout(a * b)


class cgMLP(nn.Module):
    """Convolutional gating MLP — the local branch."""

    def __init__(self, d_model: int, d_ff: int, kernel_size: int, dropout: float):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU())
        self.csgu = ConvolutionalSpatialGatingUnit(d_ff, kernel_size, dropout)
        self.proj_out = nn.Linear(d_ff // 2, d_model)

    def forward(self, x, pad_mask):
        x = self.proj_in(x)
        x = self.csgu(x, pad_mask)
        return self.proj_out(x)


class EBranchformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, cgmlp_ff, n_heads, kernel_size,
                 merge_kernel, dropout):
        super().__init__()
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.norm_ffn1 = nn.LayerNorm(d_model)

        self.norm_attn = nn.LayerNorm(d_model)
        self.attn = RelPositionMultiHeadAttention(d_model, n_heads, dropout)

        self.norm_cgmlp = nn.LayerNorm(d_model)
        self.cgmlp = cgMLP(d_model, cgmlp_ff, kernel_size, dropout)

        # Merge: depthwise conv over concatenated branches, then linear back.
        self.merge_conv = nn.Conv1d(
            2 * d_model, 2 * d_model, merge_kernel,
            padding=(merge_kernel - 1) // 2, groups=2 * d_model,
        )
        self.merge_proj = nn.Linear(2 * d_model, d_model)

        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, attn_mask, pad_mask):
        x = x + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))

        x_global = self.attn(self.norm_attn(x), pos_emb, attn_mask)
        x_local = self.cgmlp(self.norm_cgmlp(x), pad_mask)

        merged = torch.cat([x_global, x_local], dim=-1)        # [B, T, 2D]
        merged = merged.transpose(1, 2)
        if pad_mask is not None:
            merged = merged.masked_fill(pad_mask.unsqueeze(1), 0.0)
        merged = self.merge_conv(merged).transpose(1, 2)
        x = x + self.dropout(self.merge_proj(merged))

        x = x + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
        return self.norm_final(x)


class EBranchformerEncoder(EncoderBase):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        d_ff: int = 1024,
        cgmlp_ff: int = 1536,
        n_heads: int = 4,
        n_layers: int = 6,   # matched ~15M budget (spec Section 4)
        kernel_size: int = 31,
        merge_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._out_dim = d_model
        self.subsampling = Conv2dSubsampling(n_mels, d_model, dropout)
        self.pos_enc = RelPositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList(
            [
                EBranchformerBlock(
                    d_model, d_ff, cgmlp_ff, n_heads, kernel_size,
                    merge_kernel, dropout,
                )
                for _ in range(n_layers)
            ]
        )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x, lengths = self.subsampling(feats, feat_lengths)
        pad_mask = make_pad_mask(lengths, x.size(1))
        pos_emb = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, pos_emb, pad_mask, pad_mask)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x, lengths
