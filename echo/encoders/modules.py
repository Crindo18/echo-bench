"""Reusable building blocks shared by the ESPnet-family encoders
(Conformer, E-Branchformer): relative-position multi-head self-attention,
macaron-style feed-forward, and the depthwise-separable convolution module."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RelPositionalEncoding(nn.Module):
    """Transformer-XL style relative positional encoding (as used by ESPnet
    Conformer). Produces the positional term consumed by RelPositionAttention."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(d_model)
        self._pe: torch.Tensor | None = None
        self._build(max_len)

    def _build(self, length: int):
        pe_pos = torch.zeros(length, self.d_model)
        pe_neg = torch.zeros(length, self.d_model)
        position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / self.d_model)
        )
        pe_pos[:, 0::2] = torch.sin(position * div)
        pe_pos[:, 1::2] = torch.cos(position * div)
        pe_neg[:, 0::2] = torch.sin(-1 * position * div)
        pe_neg[:, 1::2] = torch.cos(-1 * position * div)
        # order: [+ (L-1) ... +1, 0, -1 ... -(L-1)]  -> length 2L-1
        pe_pos = torch.flip(pe_pos, [0]).unsqueeze(0)
        pe_neg = pe_neg[1:].unsqueeze(0)
        self._pe = torch.cat([pe_pos, pe_neg], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, D]. Returns positional embedding [1, 2T-1, D]."""
        t = x.size(1)
        if self._pe is None or self._pe.size(1) < 2 * t - 1 or self._pe.device != x.device:
            self._build(t)
            self._pe = self._pe.to(x.device)
        center = self._pe.size(1) // 2
        pos = self._pe[:, center - t + 1 : center + t]
        return self.dropout(pos)


class RelPositionMultiHeadAttention(nn.Module):
    """Multi-head self-attention with relative positional encoding."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.h = n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(self.h, self.d_k))
        nn.init.xavier_uniform_(self.pos_bias_u)
        nn.init.xavier_uniform_(self.pos_bias_v)
        self.dropout = nn.Dropout(dropout)

    def _rel_shift(self, x: torch.Tensor) -> torch.Tensor:
        b, h, t1, t2 = x.shape
        zero = x.new_zeros(b, h, t1, 1)
        x = torch.cat([zero, x], dim=-1)
        x = x.view(b, h, t2 + 1, t1)
        x = x[:, :, 1:].view_as(torch.empty(b, h, t1, t2))
        return x[:, :, :, : t2 // 2 + 1]

    def forward(self, x: torch.Tensor, pos_emb: torch.Tensor, mask: torch.Tensor | None):
        b, t, _ = x.shape
        q = self.linear_q(x).view(b, t, self.h, self.d_k)
        k = self.linear_k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(b, t, self.h, self.d_k).transpose(1, 2)

        n_pos = pos_emb.size(1)
        p = self.linear_pos(pos_emb).view(1, n_pos, self.h, self.d_k).transpose(1, 2)

        q_u = (q + self.pos_bias_u).transpose(1, 2)   # [B, h, T, d_k]
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        ac = torch.matmul(q_u, k.transpose(-2, -1))   # [B, h, T, T]
        bd = torch.matmul(q_v, p.transpose(-2, -1))   # [B, h, T, 2T-1]
        bd = self._rel_shift(bd)
        scores = (ac + bd) / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, -1)
        return self.linear_out(out)


class FeedForward(nn.Module):
    """Position-wise FFN (used macaron-style: two half-step FFNs per block)."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ConvolutionModule(nn.Module):
    """Conformer convolution module: pointwise -> GLU -> depthwise -> BN ->
    SiLU -> pointwise. Uses masked padding so pads never leak into the conv."""

    def __init__(self, d_model: int, kernel_size: int = 15, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0
        self.pointwise1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(
            d_model, d_model, kernel_size,
            padding=(kernel_size - 1) // 2, groups=d_model,
        )
        self.norm = nn.BatchNorm1d(d_model)
        self.pointwise2 = nn.Conv1d(d_model, d_model, 1)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor | None):
        x = x.transpose(1, 2)                         # [B, D, T]
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.pointwise1(x)
        x = nn.functional.glu(x, dim=1)
        if pad_mask is not None:
            x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x = self.depthwise(x)
        x = self.activation(self.norm(x))
        x = self.pointwise2(x)
        x = self.dropout(x)
        return x.transpose(1, 2)
