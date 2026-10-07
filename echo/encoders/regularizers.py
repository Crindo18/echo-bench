"""Training-only activation regularizers used by Zipformer (icefall).

Both are **identity in the forward pass and only touch gradients in the backward
pass**, so they add no parameters, no compute at inference, and nothing to the
exported/quantized model — they simply steer training away from two failure modes
Zipformer is prone to:

  * **Balancer** keeps each channel's activation mean and RMS inside a sane range,
    preventing dead channels (stuck at zero) and saturated ones — Zipformer's
    aggressive per-block scaling makes these common without it.
  * **Whitener** discourages the channels from collapsing onto a low-rank subspace
    (all carrying the same information) by penalising large off-diagonal feature
    covariance when it exceeds a limit.

Each is skipped entirely in eval mode, so they cost nothing at inference. The
implementations reproduce the icefall regularizers' *mechanism* (a bounded,
scaled gradient nudge applied only when a statistic leaves its target range)
rather than their exact constants.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class _BalancerFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, channel_dim, min_mean, max_mean, min_rms, max_rms, grad_scale):
        ctx.save_for_backward(x)
        ctx.cfg = (channel_dim, min_mean, max_mean, min_rms, max_rms, grad_scale)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        cdim, min_mean, max_mean, min_rms, max_rms, gs = ctx.cfg
        # reduce over every dim except the channel dim
        dims = [d for d in range(x.dim()) if d != (cdim % x.dim())]
        mean = x.mean(dim=dims, keepdim=True)
        rms = (x * x).mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-10)
        # push mean toward [min_mean, max_mean] and rms toward [min_rms, max_rms].
        # gradient on x: positive where we want to DECREASE x (so it is added to
        # the outgoing grad, which descends). Bounded to +-1 per element.
        mean_pen = (mean > max_mean).float() - (mean < min_mean).float()
        rms_pen = ((rms > max_rms).float() - (rms < min_rms).float()) * (x / rms)
        correction = gs * (mean_pen + rms_pen)
        return grad_out + correction, None, None, None, None, None, None


class Balancer(nn.Module):
    def __init__(self, num_channels: int, channel_dim: int = -1,
                 min_mean: float = -0.8, max_mean: float = 0.8,
                 min_rms: float = 0.2, max_rms: float = 4.0,
                 grad_scale: float = 0.02):
        super().__init__()
        self.channel_dim = channel_dim
        self.cfg = (min_mean, max_mean, min_rms, max_rms, grad_scale)

    def forward(self, x):
        if not self.training or not x.requires_grad:
            return x
        return _BalancerFn.apply(x, self.channel_dim, *self.cfg)


class _WhitenFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, channel_dim, limit, grad_scale):
        ctx.save_for_backward(x)
        ctx.cfg = (channel_dim, limit, grad_scale)
        return x

    @staticmethod
    def backward(ctx, grad_out):
        (x,) = ctx.saved_tensors
        cdim, limit, gs = ctx.cfg
        cdim = cdim % x.dim()
        # flatten to [N, C]
        xt = x.transpose(cdim, -1).reshape(-1, x.size(cdim))
        N, C = xt.shape
        if N < 2:
            return grad_out, None, None, None
        xc = xt - xt.mean(dim=0, keepdim=True)
        cov = (xc.t() @ xc) / (N - 1)                      # [C, C]
        diag = torch.diagonal(cov)
        # "whitening metric": off-diagonal energy vs on-diagonal energy
        off = cov - torch.diag(diag)
        metric = (off.pow(2).sum() / (diag.pow(2).sum() + 1e-10)).sqrt()
        if float(metric) <= limit:
            return grad_out, None, None, None
        # gradient of off-diagonal covariance energy w.r.t. x, pushing toward
        # decorrelated (whitened) features; scaled and applied only when over limit
        grad_cov = 2.0 * (xc @ off) / (N - 1)              # [N, C]
        grad_cov = grad_cov.reshape_as(x.transpose(cdim, -1)).transpose(cdim, -1)
        return grad_out + gs * grad_cov, None, None, None


class Whitener(nn.Module):
    def __init__(self, channel_dim: int = -1, limit: float = 5.0,
                 grad_scale: float = 0.01):
        super().__init__()
        self.channel_dim = channel_dim
        self.limit = limit
        self.grad_scale = grad_scale

    def forward(self, x):
        if not self.training or not x.requires_grad:
            return x
        return _WhitenFn.apply(x, self.channel_dim, self.limit, self.grad_scale)
