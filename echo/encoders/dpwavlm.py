"""DPWavLM — the COMPRESSED FOUNDATION MODEL (spec Section 2, 5).

The "shrink-big" entry: WavLM Base+ (~95M) compressed via joint distillation
and structured pruning (the DPHuBERT recipe) down to the ~15M edge budget.
This tests whether inherited SSL pretraining beats building small. It is the
one calculated gamble in the set, isolated behind a Go/No-Go gate (Section 5).

Design of this module:
  * `WavLMTeacher` wraps torchaudio's real pretrained WAVLM_BASE_PLUS and
    produces frozen per-layer hidden-state targets for distillation. It is the
    *origin* variable — the pretraining the student inherits.
  * `DPWavLMStudent` is a WavLM-style backbone with FIRST-CLASS structured
    pruning gates (HardConcrete / L0) on attention heads, FFN units, and whole
    layers, so the DPHuBERT recipe runs cleanly and the pruned model can be
    *materialised* into a compact static encoder that exports to INT8 ONNX.
  * `init_student_from_teacher` performs the DPHuBERT "student = teacher"
    initialisation: it structurally copies the teacher's conv extractor, feature
    projection, and every transformer layer's attention/FFN/LayerNorm weights
    into the full-size student, so the student genuinely starts as WavLM Base+.
    The one inherited-capacity gap is WavLM's gated relative-position bias, which
    this student does not model (it uses a positional conv instead); that part of
    the teacher is not transferable and is learned during distillation. Coverage
    is measured and the init fails loud if the installed torchaudio's key names
    don't match (rather than silently producing a random student).

Consumes raw 16 kHz waveform (its own conv feature extractor is its inherited
front-end — the documented origin exception to the shared log-Mel front-end,
spec Section 4). The compression procedure and gate live in
`echo/train/distill_prune.py`.
"""
from __future__ import annotations

import math
import warnings

import torch
import torch.nn as nn

from .base import EncoderBase, make_pad_mask

# WavLM Base+ conv feature extractor: (out_ch, kernel, stride) per layer.
WAVLM_CONV_CONFIG = [(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2


def wavlm_feat_lengths(lengths, conv_config=WAVLM_CONV_CONFIG):
    """Valid output frames after the (no-padding) conv feature extractor:
    L <- floor((L - kernel) / stride) + 1 for each conv layer."""
    import torch
    out = lengths.clone().to(torch.long)
    for _, kernel, stride in conv_config:
        out = torch.div(out - kernel, stride, rounding_mode="floor") + 1
    return out.clamp_min(0)


# ----------------------------------------------------------------------------- #
# HardConcrete gate (Louizos et al., 2018) — the L0 machinery used by DPHuBERT.
# ----------------------------------------------------------------------------- #
class HardConcreteGate(nn.Module):
    """A vector of differentiable binary gates with an expected-L0 penalty.

    During training gates are sampled and stretched into [0,1]; at eval they are
    deterministic. `expected_l0()` returns the expected number of OPEN gates,
    which the sparsity Lagrangian multiplies by params-per-unit to hit budget."""

    def __init__(self, n_gates: int, init_mean: float = 0.98,
                 beta: float = 0.83, gamma: float = -0.1, zeta: float = 1.1):
        super().__init__()
        self.beta, self.gamma, self.zeta = beta, gamma, zeta
        # init log_alpha so gates start almost fully open (keep teacher capacity)
        p = min(max(init_mean, 1e-3), 1 - 1e-3)
        self.log_alpha = nn.Parameter(
            torch.full((n_gates,), math.log(p / (1 - p)))
        )

    def _open_prob(self) -> torch.Tensor:
        # P(gate > 0) under the hard-concrete stretch
        return torch.sigmoid(
            self.log_alpha - self.beta * math.log(-self.gamma / self.zeta)
        )

    def expected_l0(self) -> torch.Tensor:
        return self._open_prob().sum()

    def forward(self) -> torch.Tensor:
        if self.training:
            u = torch.rand_like(self.log_alpha).clamp(1e-6, 1 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + self.log_alpha)
                              / self.beta)
        else:
            s = torch.sigmoid(self.log_alpha)
        s = s * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0.0, 1.0)

    @torch.no_grad()
    def keep_mask(self, threshold: float = 0.0) -> torch.Tensor:
        s = torch.sigmoid(self.log_alpha) * (self.zeta - self.gamma) + self.gamma
        return s.clamp(0.0, 1.0) > threshold


# ----------------------------------------------------------------------------- #
# Backbone components
# ----------------------------------------------------------------------------- #
class ConvFeatureExtractor(nn.Module):
    """WavLM-style 7-layer conv stack: raw waveform -> [B, T, 512] @ 20 ms."""

    def __init__(self, conv_config=WAVLM_CONV_CONFIG, dropout: float = 0.0):
        super().__init__()
        layers, in_ch = [], 1
        for out_ch, kernel, stride in conv_config:
            layers.append(nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False))
            layers.append(nn.GroupNorm(out_ch, out_ch))  # layer_norm mode
            layers.append(nn.GELU())
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.conv_config = conv_config

    def _out_len(self, lengths):
        for _, kernel, stride in self.conv_config:
            lengths = torch.div(lengths - kernel, stride, rounding_mode="floor") + 1
        return lengths

    def forward(self, wav, lengths):
        x = self.conv(wav.unsqueeze(1)).transpose(1, 2)   # [B, T, 512]
        return x, self._out_len(lengths).clamp_min(0)


class GatedMHSA(nn.Module):
    """Multi-head self-attention with a per-head prune gate.

    `head_dim` is explicit so pruning heads shrinks the q/k/v/out projections
    (inner_dim = n_heads * head_dim) rather than silently growing head_dim."""

    def __init__(self, d_model, n_heads, head_dim=None, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.d_k = head_dim if head_dim is not None else d_model // n_heads
        inner = self.h * self.d_k
        self.q = nn.Linear(d_model, inner)
        self.k = nn.Linear(d_model, inner)
        self.v = nn.Linear(d_model, inner)
        self.out = nn.Linear(inner, d_model)
        self.head_gate = HardConcreteGate(n_heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        b, t, _ = x.shape
        q = self.q(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        k = self.k(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        v = self.v(x).view(b, t, self.h, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx = torch.matmul(attn, v)                       # [B, h, T, d_k]
        gate = self.head_gate().view(1, self.h, 1, 1)     # gate per head
        ctx = ctx * gate
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, -1)
        return self.out(ctx)


class GatedFFN(nn.Module):
    """Position-wise FFN with a per-intermediate-unit prune gate."""

    def __init__(self, d_model, d_ff, dropout):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.unit_gate = HardConcreteGate(d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.act(self.fc1(x))
        h = h * self.unit_gate().view(1, 1, -1)           # gate intermediate units
        return self.fc2(self.dropout(h))


class PrunableEncoderLayer(nn.Module):
    """WavLM post-norm layer with attention/FFN/layer prune gates."""

    def __init__(self, d_model, n_heads, d_ff, dropout, head_dim=None):
        super().__init__()
        self.attn = GatedMHSA(d_model, n_heads, head_dim, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.ffn = GatedFFN(d_model, d_ff, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.layer_gate = HardConcreteGate(1)             # can drop whole layer
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pad_mask):
        g = self.layer_gate()                             # scalar in [0,1]
        x = self.attn_norm(x + g * self.dropout(self.attn(x, pad_mask)))
        x = self.ffn_norm(x + g * self.dropout(self.ffn(x)))
        return x


class DPWavLMStudent(EncoderBase):
    """WavLM-style student encoder with structured pruning gates.

    Built at Base+ dims by default; `distill_prune` learns the gates and
    `materialize()` produces a compact static student at the target budget.
    """

    accepts_waveform = True

    def __init__(
        self,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        d_ff: int = 3072,
        conv_config=WAVLM_CONV_CONFIG,
        dropout: float = 0.0,
        layers_cfg=None,
        head_dim: int | None = None,
    ):
        super().__init__()
        self._out_dim = d_model
        self.feature_extractor = ConvFeatureExtractor(conv_config, dropout)
        self.feature_proj = nn.Linear(conv_config[-1][0], d_model)
        self.feature_norm = nn.LayerNorm(conv_config[-1][0])
        self.pos_conv = nn.Conv1d(d_model, d_model, 128, padding=64, groups=16)
        # ``layers_cfg`` lets a *materialised* (pruned) student have per-layer
        # head/unit counts. When absent, all layers are the uniform Base+ dims.
        # ``head_dim`` fixes the attention head size independent of ``n_heads``
        # (materialised layers keep 64-d heads but fewer of them); None -> d/nh.
        if layers_cfg is None:
            layers_cfg = [dict(n_heads=n_heads, d_ff=d_ff) for _ in range(n_layers)]
        self.layers = nn.ModuleList(
            [PrunableEncoderLayer(d_model, int(lc["n_heads"]), int(lc["d_ff"]),
                                  dropout, head_dim=head_dim)
             for lc in layers_cfg]
        )
        self.cfg = dict(
            d_model=d_model, n_layers=len(layers_cfg), n_heads=n_heads,
            d_ff=d_ff, conv_config=[tuple(c) for c in conv_config],
            layers_cfg=[dict(n_heads=int(lc["n_heads"]), d_ff=int(lc["d_ff"]))
                        for lc in layers_cfg],
            head_dim=head_dim,
        )

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor,
                return_hidden: bool = False):
        x, lengths = self.feature_extractor(wav, wav_lengths)
        x = self.feature_proj(self.feature_norm(x))
        pc = self.pos_conv(x.transpose(1, 2))[:, :, : x.size(1)]
        x = x + pc.transpose(1, 2)
        pad_mask = make_pad_mask(lengths, x.size(1))
        hiddens = []
        for layer in self.layers:
            x = layer(x, pad_mask)
            hiddens.append(x)
        x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
        if return_hidden:
            return x, lengths, hiddens
        return x, lengths

    # ---- pruning-gate utilities -------------------------------------------- #
    def gate_parameters(self):
        return [p for n, p in self.named_parameters() if "log_alpha" in n]

    def main_parameters(self):
        return [p for n, p in self.named_parameters() if "log_alpha" not in n]

    def expected_num_params(self) -> torch.Tensor:
        """Differentiable estimate of remaining params given current gates.
        Drives the sparsity Lagrangian toward the ≤15M budget."""
        d = self.cfg["d_model"]
        dk = d // self.cfg["n_heads"]
        total = x = torch.zeros((), device=self.layers[0].attn.q.weight.device)
        # fixed (un-gated) params: extractor, proj, pos_conv, norms
        fixed = sum(p.numel() for n, p in self.named_parameters()
                    if not any(s in n for s in ("layers.",)))
        total = total + fixed
        for layer in self.layers:
            e_layer = layer.layer_gate.expected_l0()                # ~1 or ~0
            e_heads = layer.attn.head_gate.expected_l0()
            e_units = layer.ffn.unit_gate.expected_l0()
            # attention params scale with kept heads: q,k,v,out ~ 4 * d * (heads*dk)
            attn_p = 4 * d * (e_heads * dk) + 4 * (e_heads * dk)
            # ffn params scale with kept units: fc1 (d*u) + fc2 (u*d) + biases
            ffn_p = 2 * d * e_units + e_units + d
            # layernorms counted as fixed-ish
            total = total + e_layer * (attn_p + ffn_p)
        return total

    @torch.no_grad()
    def materialize(self, threshold: float = 0.0) -> "DPWavLMStudent":
        """Slice weights by the learned gates into a compact static student
        (no gates). This is the model that enters the benchmark if the gate
        passes (spec Section 5)."""
        d = self.cfg["d_model"]
        dk = d // self.cfg["n_heads"]

        kept_layers = [i for i, l in enumerate(self.layers)
                       if bool(l.layer_gate.keep_mask(threshold).item())]
        # per kept layer, decide heads / ffn units to keep
        head_keep, ffn_keep = {}, {}
        for i in kept_layers:
            hm = self.layers[i].attn.head_gate.keep_mask(threshold)
            um = self.layers[i].ffn.unit_gate.keep_mask(threshold)
            # guarantee at least one head / unit
            if hm.sum() == 0:
                hm[0] = True
            if um.sum() == 0:
                um[0] = True
            head_keep[i] = hm
            ffn_keep[i] = um

        # Per-layer pruned dims. A materialised layer keeps full 64-d heads but
        # fewer of them (and fewer FFN units); widths differ per layer, so we
        # record them in ``layers_cfg`` and build the student from that — this is
        # what makes the saved ``cfg`` reconstruct the exact architecture on
        # reload (finetune / quantize / eval all rebuild from cfg).
        layers_cfg = [dict(n_heads=int(head_keep[i].sum().item()),
                           d_ff=int(ffn_keep[i].sum().item()))
                      for i in kept_layers]
        student = DPWavLMStudent(
            d_model=d, n_heads=self.cfg["n_heads"], d_ff=self.cfg["d_ff"],
            conv_config=self.cfg["conv_config"], dropout=0.0,
            layers_cfg=layers_cfg, head_dim=dk,
        )
        # copy the shared (fixed) params verbatim
        student.feature_extractor.load_state_dict(self.feature_extractor.state_dict())
        student.feature_proj.load_state_dict(self.feature_proj.state_dict())
        student.feature_norm.load_state_dict(self.feature_norm.state_dict())
        student.pos_conv.load_state_dict(self.pos_conv.state_dict())

        for new_i, old_i in enumerate(kept_layers):
            old = self.layers[old_i]
            new = student.layers[new_i]
            hm, um = head_keep[old_i], ffn_keep[old_i]
            head_idx = torch.arange(self.cfg["n_heads"])[hm]
            dim_idx = torch.cat([torch.arange(h * dk, h * dk + dk) for h in head_idx])
            # attention: slice q/k/v out-dims and out in-dims by kept head dims
            for name in ("q", "k", "v"):
                src, dst = getattr(old.attn, name), getattr(new.attn, name)
                dst.weight.copy_(src.weight[dim_idx])
                dst.bias.copy_(src.bias[dim_idx])
            new.attn.out.weight.copy_(old.attn.out.weight[:, dim_idx])
            new.attn.out.bias.copy_(old.attn.out.bias)
            # ffn: slice intermediate units
            new.ffn.fc1.weight.copy_(old.ffn.fc1.weight[um])
            new.ffn.fc1.bias.copy_(old.ffn.fc1.bias[um])
            new.ffn.fc2.weight.copy_(old.ffn.fc2.weight[:, um])
            new.ffn.fc2.bias.copy_(old.ffn.fc2.bias)
            new.attn_norm.load_state_dict(old.attn_norm.state_dict())
            new.ffn_norm.load_state_dict(old.ffn_norm.state_dict())
            # open all gates in the materialised model (static, no pruning)
            for g in (new.attn.head_gate, new.ffn.unit_gate, new.layer_gate):
                nn.init.constant_(g.log_alpha, 4.0)
        return student


class WavLMTeacher(nn.Module):
    """Frozen torchaudio WavLM Base+ producing per-layer hidden-state targets.

    Uses real pretrained weights when available (the SSL *origin*). If the
    weights cannot be downloaded (offline / restricted network), it falls back
    to the WavLM Base+ *architecture* with random init and warns — enough to
    exercise the full pipeline, but not a real teacher. In production, ensure
    the pretrained checkpoint downloads."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchaudio
        self.pretrained = pretrained
        model = None
        if pretrained:
            try:
                model = torchaudio.pipelines.WAVLM_BASE_PLUS.get_model()
            except Exception as e:  # download blocked, etc.
                warnings.warn(
                    f"Could not load pretrained WavLM Base+ ({e}). Falling back "
                    "to randomly-initialised architecture — NOT a real teacher."
                )
        if model is None:
            model = torchaudio.models.wavlm_model(
                extractor_mode="layer_norm",
                extractor_conv_layer_config=WAVLM_CONV_CONFIG,
                extractor_conv_bias=False,
                encoder_embed_dim=768, encoder_projection_dropout=0.0,
                encoder_pos_conv_kernel=128, encoder_pos_conv_groups=16,
                encoder_num_layers=12, encoder_num_heads=12,
                encoder_num_buckets=320, encoder_max_distance=800,
                encoder_attention_dropout=0.0, encoder_ff_interm_features=3072,
                encoder_ff_interm_dropout=0.0, encoder_dropout=0.0,
                encoder_layer_norm_first=False, encoder_layer_drop=0.0,
                aux_num_out=None,
            )
            self.pretrained = False
        self.model = model
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def hidden_states(self, wav: torch.Tensor, lengths: torch.Tensor | None = None):
        # NOTE: torchaudio's WavLM (gated relative-position) attention asserts
        # attention_mask is None, so we must NOT pass lengths into
        # extract_features (that would build a mask and crash). We run the full
        # (padded) batch unmasked and instead return the valid frame lengths so
        # the distillation loss can mask padded frames itself.
        feats, _ = self.model.extract_features(wav, None)
        out_len = wavlm_feat_lengths(lengths) if lengths is not None else None
        return feats, out_len  # list[Tensor[B,T,768]], lengths|None


def _copy_linear(dst: nn.Module, weight, bias) -> int:
    """Copy weight (and bias if present) into a Linear/Conv, shape-checked.
    Returns numel copied (0 if shapes don't match)."""
    if weight is None or dst.weight.shape != weight.shape:
        return 0
    dst.weight.data.copy_(weight)
    n = dst.weight.numel()
    if bias is not None and getattr(dst, "bias", None) is not None \
            and dst.bias.shape == bias.shape:
        dst.bias.data.copy_(bias)
        n += dst.bias.numel()
    return n


def init_student_from_teacher(
    student: DPWavLMStudent, teacher: WavLMTeacher,
    strict: bool = True, min_coverage: float = 0.70,
) -> dict:
    """DPHuBERT "student = teacher" init: structural weight copy from torchaudio
    WavLM into the student backbone, so the student *starts as the teacher*
    (minus WavLM's gated relative-position bias, which this student does not
    model) and the "inherits SSL pretraining" property is real rather than
    distillation-only.

    Called on the FULL-size student (Base+ dims), before any gate is learned, so
    every copied tensor is the same shape in teacher and student -- these are
    plain state-dict copies, not sliced ones. Structural pruning happens later in
    ``materialize()``.

    What is copied: the 7 conv feature-extractor layers, the feature projection
    and its LayerNorm, and per transformer layer the q/k/v/out attention
    projections, both FFN linears, and both LayerNorms. Skipped (no student home,
    or version-fragile): WavLM's relative-position-bias params, and the
    weight-normed positional conv (reconstructing weight_norm across torchaudio
    versions is unreliable -- it is left to training).

    Robustness: teacher tensors are located by substring role within each
    ``layers.{i}`` block rather than by exact hardcoded paths, so it tolerates
    prefix differences across torchaudio versions. Coverage (student body params
    initialised from the teacher / total) is measured; if it falls below
    ``min_coverage`` the key names on THIS torchaudio version don't match the
    expected WavLM layout, and with ``strict=True`` the function RAISES rather
    than silently degrading to a mostly-random student (the failure mode the old
    conv-only init hid). Pass ``strict=False`` to downgrade to a warning.

    Returns a coverage report dict.
    """
    import re

    t_sd = teacher.model.state_dict()
    report = {"copied_params": 0, "components": {}}

    def _teacher_body_numel() -> int:
        # student params that inheritance is *expected* to cover (everything
        # under the backbone except the prune gates and the skipped pos_conv).
        tot = 0
        for n, p in student.named_parameters():
            if "log_alpha" in n or n.startswith("pos_conv"):
                continue
            tot += p.numel()
        return tot

    # -- conv feature extractor: match Conv1d weights by order ---------------- #
    t_conv = [k for k in t_sd if "feature_extractor" in k and k.endswith("weight")
              and t_sd[k].dim() == 3]
    s_conv = [m for m in student.feature_extractor.conv if isinstance(m, nn.Conv1d)]
    conv_n = 0
    for m, k in zip(s_conv, t_conv):
        if m.weight.shape == t_sd[k].shape:
            m.weight.data.copy_(t_sd[k]); conv_n += m.weight.numel()
    report["components"]["conv"] = conv_n

    # -- feature projection (512->768) + its LayerNorm (512) ----------------- #
    def _first(*needles, exclude=()):
        for k, v in t_sd.items():
            if all(nd in k for nd in needles) and not any(e in k for e in exclude):
                return v
        return None

    proj_w = _first("feature_projection", "projection", "weight")
    proj_b = _first("feature_projection", "projection", "bias")
    fn_w = _first("feature_projection", "layer_norm", "weight")
    fn_b = _first("feature_projection", "layer_norm", "bias")
    report["components"]["feature_proj"] = _copy_linear(student.feature_proj, proj_w, proj_b)
    fn_n = 0
    if fn_w is not None and student.feature_norm.weight.shape == fn_w.shape:
        student.feature_norm.weight.data.copy_(fn_w)
        student.feature_norm.bias.data.copy_(fn_b)
        fn_n = student.feature_norm.weight.numel() + student.feature_norm.bias.numel()
    report["components"]["feature_norm"] = fn_n

    # -- transformer layers: q/k/v/out, fc1/fc2, attn-LN, ffn-LN ------------- #
    # NB: exclude the conv feature extractor first. Its keys are
    # "feature_extractor.conv_layers.{i}." which CONTAINS the substring
    # "layers.{i}.", so a bare `layers\.(\d+)\.` search would wrongly bucket the
    # conv layer_norm ([512]) into a transformer layer and shadow that layer's
    # real attention LayerNorm ([768]) in _pick — silently leaving it random.
    layer_re = re.compile(r"layers\.(\d+)\.")
    t_layers: dict[int, dict] = {}
    for k, v in t_sd.items():
        if "feature_extractor" in k or "conv_layers" in k:
            continue
        m = layer_re.search(k)
        if m:
            t_layers.setdefault(int(m.group(1)), {})[k] = v

    def _pick(kv, *needles, exclude=()):
        for k, v in kv.items():
            if all(nd in k for nd in needles) and not any(e in k for e in exclude):
                return v
        return None

    body_n = 0
    n_layers_copied = 0
    for i, layer in enumerate(student.layers):
        kv = t_layers.get(i)
        if not kv:
            continue
        before = body_n
        # attention projections (WavLM uses q_proj/k_proj/v_proj/out_proj)
        for role, dst in (("q_proj", layer.attn.q), ("k_proj", layer.attn.k),
                          ("v_proj", layer.attn.v), ("out_proj", layer.attn.out)):
            w = _pick(kv, role, "weight")
            b = _pick(kv, role, "bias")
            body_n += _copy_linear(dst, w, b)
        # feed-forward (intermediate_dense = fc1, output_dense = fc2)
        body_n += _copy_linear(layer.ffn.fc1,
                               _pick(kv, "intermediate_dense", "weight"),
                               _pick(kv, "intermediate_dense", "bias"))
        body_n += _copy_linear(layer.ffn.fc2,
                               _pick(kv, "output_dense", "weight"),
                               _pick(kv, "output_dense", "bias"))
        # layernorms: attn LN = "layer_norm" (not final); ffn LN = "final_layer_norm"
        aw = _pick(kv, "layer_norm", "weight", exclude=("final", "feed_forward"))
        ab = _pick(kv, "layer_norm", "bias", exclude=("final", "feed_forward"))
        if aw is not None and layer.attn_norm.weight.shape == aw.shape:
            layer.attn_norm.weight.data.copy_(aw)
            layer.attn_norm.bias.data.copy_(ab)
            body_n += layer.attn_norm.weight.numel() + layer.attn_norm.bias.numel()
        fw = _pick(kv, "final_layer_norm", "weight")
        fb = _pick(kv, "final_layer_norm", "bias")
        if fw is not None and layer.ffn_norm.weight.shape == fw.shape:
            layer.ffn_norm.weight.data.copy_(fw)
            layer.ffn_norm.bias.data.copy_(fb)
            body_n += layer.ffn_norm.weight.numel() + layer.ffn_norm.bias.numel()
        if body_n > before:
            n_layers_copied += 1
    report["components"]["transformer_layers"] = body_n
    report["n_layers_copied"] = n_layers_copied
    report["n_layers_total"] = len(student.layers)

    total = (conv_n + report["components"]["feature_proj"] + fn_n + body_n)
    report["copied_params"] = total
    expected = max(1, _teacher_body_numel())
    coverage = total / expected
    report["coverage"] = coverage

    msg = (f"init_student_from_teacher: inherited {coverage:.1%} of the student "
           f"backbone from the teacher ({total/1e6:.1f}M params; "
           f"{n_layers_copied}/{len(student.layers)} transformer layers). "
           "Skipped: WavLM relative-position bias (no student home) and the "
           "weight-normed positional conv (left to training).")
    if coverage < min_coverage:
        detail = (
            f"\nLOW COVERAGE ({coverage:.1%} < {min_coverage:.0%}): the WavLM "
            "encoder-layer key names on your installed torchaudio version do not "
            "match the expected q_proj/k_proj/v_proj/out_proj / intermediate_dense "
            "/ output_dense / final_layer_norm layout, so the student would be "
            "mostly RANDOM despite 'inheriting' from the teacher. Inspect "
            "`teacher.model.state_dict().keys()` and adjust the role substrings "
            "in init_student_from_teacher for your version."
        )
        if strict:
            raise RuntimeError(
                "init_student_from_teacher aborted: " + msg + detail +
                "\nPass strict=False (or --allow-random-teacher upstream) to "
                "proceed with a partially-random student anyway."
            )
        warnings.warn(msg + detail)
    else:
        # a light note so a healthy run still shows what was inherited
        print(f"[dpwavlm] {msg}")
    return report
