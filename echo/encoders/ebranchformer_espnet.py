"""ESPnet E-Branchformer wrapped in the benchmark's ``EncoderBase`` interface.

This adapter lets the *community-maintained ESPnet* E-Branchformer (the exact
implementation used by the echo-bench thesis harness, config ``ebf12m``) drop
into this project's matched, swap-only-the-encoder harness — and optionally load
ESPnet's pretrained LibriSpeech checkpoint (e.g.
``pyf98/librispeech_100_e_branchformer``). Use it when you want your headline
E-Branchformer baseline to be the trusted upstream implementation rather than
this repo's from-scratch reimplementation (``ebranchformer.py``), while keeping
everything downstream (front-end, Prototypical head, CV, scorecard) identical.

**ESPnet is an optional dependency.** It is imported lazily inside ``__init__``,
so the rest of the project (and the other four encoders) work without it. Install
it only if you use this encoder:  ``pip install -e ".[espnet]"``  (or
``pip install espnet``).

Interface mapping (both are feature-in / frame-out, ESPnet subsamples 4x itself
via its ``input_layer``):

    EncoderBase.forward(feats[B,T,n_mels], lengths) -> (hidden[B,T',D], out_lens)
    ESPnet:      encoder(xs_pad[B,T,input_size], ilens) -> (xs_pad, olens, None)

**Front-end caveat for pretrained weights.** ESPnet models are trained with their
own feature normalisation (global MVN). This project's shared log-Mel front-end
(`echo/features.py`) applies utterance CMVN, which is close but not identical. For
*random-init* training that is irrelevant (the encoder learns whatever it is fed).
For *pretrained* weights to transfer well, feed ESPnet-matched features — either
disable this repo's CMVN or apply ESPnet's stored `normalize` stats. This is
flagged in the README and left to the user's ESPnet version.
"""
from __future__ import annotations

import torch

from .base import EncoderBase


class ESPnetEBranchformerEncoder(EncoderBase):
    """ESPnet ``EBranchformerEncoder`` behind ``EncoderBase``.

    Defaults mirror ESPnet's 12-block E-Branchformer (~13M). Bump
    ``cgmlp_linear_units`` / ``linear_units`` to size it to the matched ~15M
    budget used by the other encoders (see ``count_parameters`` after building).
    """

    accepts_waveform = False

    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 256,
        n_heads: int = 4,
        num_blocks: int = 12,
        cgmlp_linear_units: int = 1024,
        cgmlp_conv_kernel: int = 31,
        linear_units: int = 1024,
        merge_conv_kernel: int = 3,
        use_ffn: bool = True,
        macaron_ffn: bool = True,
        dropout: float = 0.1,
        input_layer: str = "conv2d",
        **espnet_kwargs,
    ):
        super().__init__()
        try:
            from espnet2.asr.encoder.e_branchformer_encoder import (
                EBranchformerEncoder,
            )
        except ImportError as e:  # pragma: no cover - depends on optional dep
            raise ImportError(
                "The ESPnet E-Branchformer adapter needs ESPnet, which is not "
                "installed. Install it with:  pip install -e \".[espnet]\"  "
                "(or: pip install espnet). The other encoders work without it."
            ) from e

        self.encoder = EBranchformerEncoder(
            input_size=n_mels,
            output_size=d_model,
            attention_heads=n_heads,
            num_blocks=num_blocks,
            cgmlp_linear_units=cgmlp_linear_units,
            cgmlp_conv_kernel=cgmlp_conv_kernel,
            linear_units=linear_units,
            merge_conv_kernel=merge_conv_kernel,
            use_ffn=use_ffn,
            macaron_ffn=macaron_ffn,
            dropout_rate=dropout,
            input_layer=input_layer,
            **espnet_kwargs,
        )
        self._out_dim = d_model

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, feats: torch.Tensor, feat_lengths: torch.Tensor):
        # ESPnet returns (encoder_out, encoder_out_lens, prev_states|None)
        hidden, out_lengths, *_ = self.encoder(feats, feat_lengths)
        return hidden, out_lengths

    # ------------------------------------------------------------------ #
    # Optional: load pretrained ESPnet encoder weights
    # ------------------------------------------------------------------ #
    def load_pretrained(self, source: str, map_location: str = "cpu") -> dict:
        """Load ESPnet encoder weights from a checkpoint into this adapter.

        ``source`` may be:
          * a local path to an ESPnet ASR ``*.pth`` checkpoint (full model or a
            bare encoder state dict), or
          * a Hugging Face repo id (e.g. ``"pyf98/librispeech_100_e_branchformer"``)
            — requires ``huggingface_hub``; the ``.pth`` is downloaded first.

        Only tensors under the ``encoder.*`` namespace (or a bare encoder state
        dict) that match this adapter's parameter shapes are loaded; the rest
        (decoder, ctc, frontend) are ignored. Returns a small report so silent
        partial loads are visible. **The encoder config here must match the
        checkpoint's** (dims/blocks), or most tensors will be skipped — size this
        adapter to the pretrained model's config before calling.
        """
        path = source
        if not source.endswith((".pth", ".pt")):
            # treat as a HF repo id and fetch the model checkpoint
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as e:  # pragma: no cover
                raise ImportError(
                    "Loading by Hugging Face id needs huggingface_hub: "
                    "pip install huggingface_hub"
                ) from e
            # ESPnet ASR checkpoints are commonly stored as 'valid.*.pth' /
            # 'exp/.../valid.acc.best.pth'; the filename varies per model card,
            # so pass an explicit local .pth for full control if this guess fails.
            path = hf_hub_download(repo_id=source, filename="exp/asr_train/"
                                   "valid.acc.ave.pth")

        ckpt = torch.load(path, map_location=map_location)
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
        # strip a leading 'encoder.' if the checkpoint is a full ASR model
        enc_sd = self.encoder.state_dict()
        remapped = {}
        for k, v in state.items():
            key = k[len("encoder."):] if k.startswith("encoder.") else k
            if key in enc_sd and enc_sd[key].shape == v.shape:
                remapped[key] = v
        missing, unexpected = self.encoder.load_state_dict(remapped, strict=False)
        report = dict(loaded=len(remapped), total=len(enc_sd),
                      missing=len(missing), skipped=len(state) - len(remapped))
        if report["loaded"] == 0:
            raise RuntimeError(
                f"load_pretrained loaded 0/{report['total']} tensors from {source}. "
                "The adapter's config almost certainly doesn't match the "
                "checkpoint (dims/blocks) — size it to the pretrained model first.")
        print(f"[espnet-ebf] loaded {report['loaded']}/{report['total']} encoder "
              f"tensors from {source} (skipped {report['skipped']} non-encoder)")
        return report
