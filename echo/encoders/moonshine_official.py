"""Official Moonshine (HuggingFace ``transformers``) behind ``EncoderBase``.

This wraps the *released* Moonshine encoder (Useful Sensors, via
``transformers.MoonshineForConditionalGeneration``) so the real, pretrained model
can be benchmarked in this harness alongside everything else — the deployment
answer ("which real model do we ship") and the reimplementation-validation study
("does our budget-sized reimplementation track the official one") both become one
command. It mirrors what echo-train's ``embed.py`` does, behind the same
interface as the other encoders.

Moonshine is raw-waveform (``accepts_waveform = True``): its own convolutional
front-end reads the audio, so no log-Mel step. Output frames are ~41.7 Hz (384x
downsampling); we compute valid frame lengths with the published conv geometry so
the downstream masked pooling is exact.

**Optional dependency** (``transformers``), imported lazily — the rest of the
project works without it. Install with ``pip install -e ".[moonshine]"``.

Strict loading (same discipline as echo-train): a pretrained load refuses to
proceed if any encoder tensor is missing / reshaped / unexpected, so embeddings
can never be silently half-random. ``pretrained=False`` gives the untrained
"floor" baseline (the official architecture with random weights).
"""
from __future__ import annotations

import torch

from .base import EncoderBase

# Moonshine conv feature-extractor geometry (kernel, stride), no padding.
_MOONSHINE_CONV = [(127, 64), (7, 3), (3, 2)]
_ENC_PREFIX = "model.encoder."


def _moonshine_out_len(lengths: torch.Tensor) -> torch.Tensor:
    out = lengths.clone().to(torch.long)
    for kernel, stride in _MOONSHINE_CONV:
        out = torch.div(out - kernel, stride, rounding_mode="floor") + 1
    return out.clamp_min(0)


class MoonshineOfficialEncoder(EncoderBase):
    """The released Moonshine encoder (default: ``moonshine-tiny``, ~7.7M)."""

    accepts_waveform = True

    def __init__(self, source: str = "UsefulSensors/moonshine-tiny",
                 pretrained: bool = True, sample_rate: int = 16000):
        super().__init__()
        try:
            from transformers import (AutoConfig, AutoFeatureExtractor,
                                      MoonshineForConditionalGeneration)
            from transformers.utils import logging as hf_logging
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "The official Moonshine adapter needs `transformers`, which is "
                "not installed. Install it with:  pip install -e \".[moonshine]\"  "
                "(or: pip install 'transformers>=5.17'). The other encoders work "
                "without it."
            ) from e
        hf_logging.set_verbosity_error()

        folder = self._fetch(source)
        cfg = AutoConfig.from_pretrained(folder)
        if getattr(cfg, "model_type", None) != "moonshine":
            raise ValueError(
                f"{source}: model_type={getattr(cfg,'model_type',None)!r}; only "
                "the original 'moonshine' (not the streaming variant) is supported.")

        if pretrained:
            model, info = MoonshineForConditionalGeneration.from_pretrained(
                folder, output_loading_info=True)
            self._assert_encoder_fully_loaded(info)  # strict: no half-random weights
        else:
            torch.manual_seed(0)                      # same untrained floor each run
            model = MoonshineForConditionalGeneration(cfg)

        self.encoder = model.get_encoder().eval()
        try:
            self.extractor = AutoFeatureExtractor.from_pretrained(folder)
            sample_rate = int(getattr(self.extractor, "sampling_rate", sample_rate))
        except Exception:
            self.extractor = None
        self.sample_rate = sample_rate
        self._out_dim = int(cfg.hidden_size)
        self.pretrained = pretrained

    # ---------------------------------------------------------------- #
    @staticmethod
    def _fetch(source: str):
        from pathlib import Path
        p = Path(source).expanduser()
        if p.is_dir():
            return p
        from huggingface_hub import snapshot_download
        return snapshot_download(source,
                                 allow_patterns=["*.json", "*.safetensors"])

    @staticmethod
    def _assert_encoder_fully_loaded(info: dict) -> None:
        mism = [m[0] if isinstance(m, (tuple, list)) else m
                for m in info.get("mismatched_keys", [])]
        problems = {
            "missing": [k for k in info.get("missing_keys", [])
                        if k.startswith(_ENC_PREFIX)],
            "reshaped": [k for k in mism if str(k).startswith(_ENC_PREFIX)],
            "unexpected": [k for k in info.get("unexpected_keys", [])
                           if k.startswith(_ENC_PREFIX)],
        }
        if any(problems.values()):
            counts = ", ".join(f"{len(v)} {k}" for k, v in problems.items() if v)
            example = next(k for v in problems.values() for k in v)
            raise RuntimeError(
                f"Official Moonshine did not load cleanly ({counts} encoder "
                f"tensors, e.g. {example}); its embeddings would be partly random. "
                "This usually means a transformers version mismatch — pin a "
                "compatible transformers, or use pretrained=False for the floor.")

    @property
    def out_dim(self) -> int:
        return self._out_dim

    # ---------------------------------------------------------------- #
    def forward(self, wav: torch.Tensor, wav_lengths: torch.Tensor):
        # wav: [B, samples]. Apply the model's own feature extractor when present
        # (matches the pretrained input distribution), else pass the waveform.
        input_values = self._prepare(wav, wav_lengths)
        out = self.encoder(input_values)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        return hidden, _moonshine_out_len(wav_lengths).clamp_max(hidden.size(1))

    def _prepare(self, wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        if self.extractor is None:
            return wav
        arrs = [wav[i, : int(lengths[i])].detach().cpu().numpy()
                for i in range(wav.size(0))]
        feat = self.extractor(arrs, sampling_rate=self.sample_rate,
                              padding=True, return_tensors="pt")
        key = self.extractor.model_input_names[0]
        return feat[key].to(wav.device)
