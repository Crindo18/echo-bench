"""Conformer encoder — the scientific CONTROL (spec Section 2).

Sequential attention->convolution hybrid (Gulati et al., 2020). This is the
predecessor E-Branchformer claims to beat; keeping it on ECHO's own data,
task and hardware is what keeps that claim honest. Built small to the ~15M
matched budget rather than borrowed from a published checkpoint.

Faithful PyTorch reimplementation (not the ESPnet recipe verbatim) so all four
models live in one swappable harness. Block = ½FFN -> MHSA -> Conv -> ½FFN,
with a final LayerNorm, per the Conformer paper.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import Conv2dSubsampling, EncoderBase, make_pad_mask
from .modules import (
    ConvolutionModule,
    FeedForward,
    RelPositionalEncoding,
    RelPositionMultiHeadAttention,
)


class ConformerBlock(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, kernel_size, dropout):
        super().__init__()
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.norm_ffn1 = nn.LayerNorm(d_model)
        self.self_attn = RelPositionMultiHeadAttention(d_model, n_heads, dropout)
        self.norm_attn = nn.LayerNorm(d_model)
        self.conv = ConvolutionModule(d_model, kernel_size, dropout)
        self.norm_conv = nn.LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.norm_ffn2 = nn.LayerNorm(d_model)
        self.norm_final = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, attn_mask, pad_mask):
        x = x + 0.5 * self.dropout(self.ffn1(self.norm_ffn1(x)))
        x = x + self.dropout(self.self_attn(self.norm_attn(x), pos_emb, attn_mask))
        x = x + self.dropout(self.conv(self.norm_conv(x), pad_mask))
        x = x + 0.5 * self.dropout(self.ffn2(self.norm_ffn2(x)))
        return self.norm_final(x)


class ConformerEncoder(EncoderBase):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        d_ff: int = 1024,
        n_heads: int = 4,
        n_layers: int = 8,
        kernel_size: int = 15,
        dropout: float = 0.1,
    ):
        super().__init__()
        self._out_dim = d_model
        self.subsampling = Conv2dSubsampling(n_mels, d_model, dropout)
        self.pos_enc = RelPositionalEncoding(d_model, dropout)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(d_model, d_ff, n_heads, kernel_size, dropout)
                for _ in range(n_layers)
            ]
        )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        x, lengths = self.subsampling(feats, feat_lengths)
        pad_mask = make_pad_mask(lengths, x.size(1))     # [B, T'] True at pads
        pos_emb = self.pos_enc(x)
        for block in self.blocks:
            x = block(x, pos_emb, pad_mask, pad_mask)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        return x, lengths
