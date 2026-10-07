"""Encoder registry. `build_encoder(name, **cfg)` returns an EncoderBase so the
harness can swap the single manipulated variable (spec Section 3)."""
from __future__ import annotations

from functools import partial

from .base import EncoderBase, count_parameters
from .conformer import ConformerEncoder
from .ebranchformer import EBranchformerEncoder
from .ebranchformer_espnet import ESPnetEBranchformerEncoder
from .zipformer import ZipformerEncoder
from .moonshine import MoonshineEncoder
from .moonshine_official import MoonshineOfficialEncoder
from .dpwavlm import DPWavLMStudent, WavLMTeacher, init_student_from_teacher

# `moonshine_tiny` is this repo's Moonshine at the official tiny geometry
# (d=288, 6 layers) — ~7.68M, matching the released moonshine-tiny encoder for a
# like-for-like own-vs-official comparison. `moonshine` (default) is sized up to
# the matched ~14.6M budget.
_moonshine_tiny = partial(MoonshineEncoder, d_model=288, n_layers=6, n_heads=8,
                          d_ff=1152)

ENCODERS = {
    "conformer": ConformerEncoder,
    "ebranchformer": EBranchformerEncoder,               # this repo's reimplementation
    "ebranchformer_espnet": ESPnetEBranchformerEncoder,  # upstream ESPnet (needs espnet)
    "zipformer": ZipformerEncoder,
    "moonshine": MoonshineEncoder,                       # reimpl @ matched budget (~14.6M)
    "moonshine_tiny": _moonshine_tiny,                   # reimpl @ official tiny geom (~7.68M)
    "moonshine_official": MoonshineOfficialEncoder,      # released HF model (needs transformers)
    "dpwavlm": DPWavLMStudent,
}


def build_encoder(name: str, **cfg) -> EncoderBase:
    name = name.lower()
    if name not in ENCODERS:
        raise KeyError(f"unknown encoder '{name}'. options: {list(ENCODERS)}")
    return ENCODERS[name](**cfg)


__all__ = [
    "EncoderBase", "build_encoder", "count_parameters", "ENCODERS",
    "ConformerEncoder", "EBranchformerEncoder", "ESPnetEBranchformerEncoder",
    "ZipformerEncoder", "MoonshineEncoder", "MoonshineOfficialEncoder",
    "DPWavLMStudent", "WavLMTeacher", "init_student_from_teacher",
]
