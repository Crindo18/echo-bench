"""ScaledAdam — the optimizer Zipformer is designed to train with.

Zipformer (Yao et al., 2024) does not train well with plain Adam; its authors use
**ScaledAdam** (icefall). The essential idea, reproduced here: for a matrix-shaped
parameter, scale the Adam update by the parameter's own RMS, so the *relative*
step size is the same regardless of the parameter's magnitude. That scale
invariance is what lets Zipformer's many differently-scaled blocks train stably.
A per-step gradient-norm clip (from a running estimate of recent grad norms)
suppresses the occasional exploding update.

This is training-only machinery: it changes how weights are learned, not the model
— nothing here ships to the device or affects the INT8 footprint. Scalar/1-D
parameters (norm weights, biases, BiasNorm scales) use ordinary Adam; only
matrix parameters get the RMS scaling, exactly as in icefall.
"""
from __future__ import annotations

import torch
from torch.optim import Optimizer


class ScaledAdam(Optimizer):
    def __init__(self, params, lr=0.04, betas=(0.9, 0.98), eps=1e-8,
                 clipping_scale=2.0, param_min_rms=1e-5, param_max_rms=3.0,
                 weight_decay=0.0):
        if not 0.0 <= lr:
            raise ValueError(f"invalid lr: {lr}")
        defaults = dict(lr=lr, betas=betas, eps=eps, clipping_scale=clipping_scale,
                        param_min_rms=param_min_rms, param_max_rms=param_max_rms,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def _clip_scale(self, group) -> float:
        """A running-median style gradient-norm clip: keep a decaying history of
        batch grad norms and scale down a step whose norm is an outlier."""
        clip = group["clipping_scale"]
        if clip is None:
            return 1.0
        tot = 0.0
        for p in group["params"]:
            if p.grad is not None:
                tot += float(p.grad.norm()) ** 2
        tot = tot ** 0.5
        st = self.state.setdefault("_global", {})
        hist = st.setdefault("grad_norm_ema", tot if tot > 0 else 1.0)
        # exponential moving average of the grad norm
        st["grad_norm_ema"] = 0.98 * hist + 0.02 * tot
        threshold = clip * st["grad_norm_ema"]
        return min(1.0, threshold / (tot + 1e-20)) if tot > threshold else 1.0

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, eps = group["lr"], group["eps"]
            wd = group["weight_decay"]
            pmin, pmax = group["param_min_rms"], group["param_max_rms"]
            grad_scale = self._clip_scale(group)

            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("ScaledAdam does not support sparse gradients")
                g = g.mul(grad_scale)
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                bc1 = 1 - beta1 ** t
                bc2 = 1 - beta2 ** t
                denom = (exp_avg_sq / bc2).sqrt_().add_(eps)
                update = (exp_avg / bc1) / denom          # standard Adam direction

                if wd:                                    # decoupled weight decay
                    p.mul_(1 - lr * wd)
                if p.dim() >= 2 and p.numel() > 1:
                    # matrix parameter: step size scales with its RMS
                    # (scale-invariant), clamped to [param_min_rms, param_max_rms]
                    rms = float((p.norm() / (p.numel() ** 0.5)).clamp(pmin, pmax))
                    p.add_(update, alpha=-lr * rms)
                else:                                     # scalars / 1-D: plain Adam
                    p.add_(update, alpha=-lr)
        return loss
