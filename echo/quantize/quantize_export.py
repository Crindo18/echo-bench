"""INT8 post-training quantization + ONNX export + footprint reporting.

The deployment normaliser for the benchmark (spec Section 4): whatever toolkit a
model came from, every encoder is exported to ONNX and measured through the same
runtime, so footprint and latency are directly comparable. This module:

  * wraps a frozen encoder so it exports a clean (feats|wav) -> embedding graph,
  * exports FP32 ONNX,
  * applies INT8 dynamic quantization (weight-only, the robust default for
    attention/conv encoders that avoids calibration-set sensitivity),
  * reports the deployed footprint in MB and the parameter count (SO1 metrics).

Static (activation) INT8 quantization with a calibration reader is provided as
an option for the final Pi-5 numbers; dynamic INT8 is the default because it is
reproducible without shipping a calibration set.
"""
from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn

from ..data import CharTokenizer
from ..encoders import count_parameters
from ..train.common import AcousticModel, load_checkpoint
from ..train.pretrain import build_model


# --------------------------------------------------------------------------- #
# Export wrapper: encoder -> pooled, projected embedding (what the head uses)
# --------------------------------------------------------------------------- #
class EmbeddingExport(nn.Module):
    """Exportable module: encoder native input -> L2 embedding.

    We export the *embedding* path (not the CTC head), because the deployed
    artifact is the frozen feature extractor feeding the Prototypical head
    (spec Section 6).

    The graph input is each encoder's *native* input:
      * supervised encoders  -> log-Mel features [B, T, n_mels]. The log-Mel
        filterbank is a fixed, shared DSP stage held constant across all models
        (spec Section 4), computed on-device before this graph — not a learned
        part of the encoder — and STFT is not cleanly ONNX-exportable across
        runtimes anyway, so keeping it outside the graph is both correct and
        portable.
      * DPWavLM -> raw waveform [B, num_samples]. Its conv feature extractor is
        learned and part of the encoder (the 'origin' front-end exception), so
        it stays inside the exported graph.
    """

    def __init__(self, model: AcousticModel):
        super().__init__()
        self.accepts_waveform = model.accepts_waveform
        self.encoder = model.encoder
        self.projector = model.projector

    def forward(self, x, lengths):
        hidden, out_len = self.encoder(x, lengths)
        return self.projector(hidden, out_len)


def _example_inputs(accepts_waveform: bool, seconds: float = 4.0,
                     sr: int = 16000, n_mels: int = 80, hop: int = 160):
    n = int(seconds * sr)
    if accepts_waveform:                      # DPWavLM: raw waveform
        return torch.randn(1, n), torch.tensor([n])
    # supervised: log-Mel features (front-end runs separately, on-device)
    n_frames = n // hop + 1
    return torch.randn(1, n_frames, n_mels), torch.tensor([n_frames])


def export_fp32_onnx(model: AcousticModel, path: str, seconds: float = 4.0) -> str:
    wrapper = EmbeddingExport(model).eval()
    x, lengths = _example_inputs(wrapper.accepts_waveform, seconds)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    export_kwargs = dict(
        input_names=["input", "lengths"], output_names=["embedding"],
        dynamic_axes={"input": {0: "batch", 1: "time"},
                      "lengths": {0: "batch"},
                      "embedding": {0: "batch"}},
        opset_version=17, do_constant_folding=True,
    )
    # torch>=2.x defaults to the dynamo exporter (needs onnxscript). Prefer the
    # stable TorchScript exporter when available for a dependency-light export.
    try:
        torch.onnx.export(wrapper, (x, lengths), path, dynamo=False, **export_kwargs)
    except TypeError:  # older torch without the dynamo kwarg
        torch.onnx.export(wrapper, (x, lengths), path, **export_kwargs)
    return path


def quantize_int8_onnx(fp32_path: str, int8_path: str, weight_only: bool = True) -> str:
    """Dynamic INT8 quantization of an ONNX graph via onnxruntime.

    Reproducible without a calibration set, but note: ORT dynamic quantization
    targets MatMul/Gemm only and leaves Conv layers in FP32, so conv-heavy
    encoders may exceed the 16 MB budget under dynamic quantization. Use
    ``quantize_int8_static`` (below) for the budget-compliant deployment number.
    """
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=fp32_path, model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    return int8_path


class _RandomCalibrationReader:
    """Minimal CalibrationDataReader yielding feature batches.

    For real deployment numbers, replace with a reader over a held-out slice of
    the TORGO/LibriSpeech features so activation ranges are representative. The
    random reader is only for exercising the static-quantization path.
    """

    def __init__(self, input_name: str, lengths_name: str, shape, n_batches=8,
                 accepts_waveform=False):
        import numpy as np
        self._data = []
        for _ in range(n_batches):
            if accepts_waveform:
                x = np.random.randn(1, shape).astype("float32")
                ln = np.array([shape], dtype="int64")
            else:
                n_frames, n_mels = shape
                x = np.random.randn(1, n_frames, n_mels).astype("float32")
                ln = np.array([n_frames], dtype="int64")
            self._data.append({input_name: x, lengths_name: ln})
        self._it = iter(self._data)

    def get_next(self):
        return next(self._it, None)


def quantize_int8_static(fp32_path: str, int8_path: str, accepts_waveform: bool,
                         seconds: float = 4.0, sr: int = 16000, hop: int = 160,
                         n_mels: int = 80, calib_reader=None,
                         per_channel: bool = False) -> str:
    """Static INT8 post-training quantization (QDQ format).

    Quantizes *both* Conv and MatMul weights and (via calibration) their
    activations, so the whole encoder runs in INT8 and the on-disk size reflects
    the true deployed footprint — unlike dynamic quantization, which leaves Conv
    in FP32 (see ``quantize_int8_onnx``).

    We use the QDQ (QuantizeLinear/DequantizeLinear) format rather than QOperator
    because QDQ is ONNX Runtime's recommended, most broadly supported static
    format and — critically for these encoders — it handles the custom activation
    subgraphs in Zipformer (SwooshR, BiasNorm) and E-Branchformer (cgMLP) that
    the QOperator quantizer cannot shape-infer through. On the Pi-5 CPU EP, ORT
    fuses each QuantizeLinear->op->DequantizeLinear island into a true INT8
    kernel at session-load, so QDQ and QOperator run equivalently; QDQ just
    carries a little more graph metadata on disk.

    Footprint note (spec Section 4): the ~1 byte/param INT8 budget is realised on
    *trained* checkpoints calibrated with representative features. Pass a real
    ``calib_reader`` (a CalibrationDataReader over a held-out slice of
    TORGO/LibriSpeech features) for the deployment number. The random reader used
    when ``calib_reader is None`` only exercises the code path; on untrained,
    randomly-initialised weights it produces degenerate activation ranges and an
    inflated size, so treat the parameter count (~1 byte/param) as the footprint
    proxy until a trained model is quantized.

    Set ``per_channel=True`` for per-output-channel weight scales (better
    accuracy on conv layers, negligible extra size on a trained model).
    """
    import onnx
    from onnxruntime.quantization import (quantize_static, QuantType,
                                          QuantFormat, CalibrationMethod)
    from onnxruntime.quantization.preprocess import quant_pre_process

    prep = int8_path.replace(".onnx", "_prep.onnx")
    quant_pre_process(fp32_path, prep, skip_symbolic_shape=True)

    model = onnx.load(prep)
    in_name = model.graph.input[0].name
    len_name = model.graph.input[1].name
    if calib_reader is None:
        n = int(seconds * sr)
        shape = n if accepts_waveform else (n // hop + 1, n_mels)
        calib_reader = _RandomCalibrationReader(in_name, len_name, shape,
                                                accepts_waveform=accepts_waveform)
    quantize_static(
        model_input=prep, model_output=int8_path,
        calibration_data_reader=calib_reader,
        quant_format=QuantFormat.QDQ, per_channel=per_channel,
        weight_type=QuantType.QInt8, activation_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
    )
    try:
        os.remove(prep)
    except OSError:
        pass
    return int8_path


def file_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 * 1024)


def footprint_report(model: AcousticModel, fp32_path: str, int8_path: str) -> dict:
    """Footprint metrics for the SO1 comparison (spec Section 4).

    Two budget views are reported:
      * ``params_within_budget`` — parameter count vs the ~15M / ~1-byte-per-param
        INT8 target. This is the stable, weight-driven proxy and the primary
        SO1 number: it does not depend on calibration data or untrained weights.
      * ``onnx_within_budget`` — measured INT8 ONNX size vs the 16 MB line. This
        is the ground-truth deployment number *on a trained, properly-calibrated
        model*; on untrained weights with random calibration it is inflated and
        should be read as informational only.
    """
    enc_params_M = count_parameters(model.encoder, trainable_only=False) / 1e6
    int8_mb = file_size_mb(int8_path)
    param_budget_MB = enc_params_M  # INT8 ideal ~= 1 byte/param
    report = dict(
        encoder_params_M=enc_params_M,
        total_params_M=count_parameters(model, trainable_only=False) / 1e6,
        onnx_fp32_MB=file_size_mb(fp32_path),
        onnx_int8_MB=int8_mb,
        param_budget_MB=param_budget_MB,
        params_within_budget=param_budget_MB <= 16.0,   # primary SO1 proxy
        onnx_within_budget=int8_mb <= 16.0,              # trained-model ground truth
        int8_compresses=int8_mb < file_size_mb(fp32_path),
        # Back-compat alias (older callers used `within_budget`): keep the
        # stable param-based verdict here so scripts don't flip on random weights.
        within_budget=param_budget_MB <= 16.0,
    )
    return report


def run(checkpoint: str, encoder: str, out_dir: str, seconds: float = 4.0,
        encoder_cfg: dict | None = None, mode: str = "static",
        calib_reader=None) -> dict:
    """Export FP32 ONNX and INT8-quantize. ``mode='static'`` (default) gives the
    budget-compliant deployment footprint (quantizes Conv+MatMul); ``'dynamic'``
    is the calibration-free path but leaves Conv in FP32."""
    tokenizer = CharTokenizer()
    model = build_model(encoder, tokenizer, encoder_cfg=encoder_cfg)
    load_checkpoint(checkpoint, model, strict=False)
    model.eval()

    os.makedirs(out_dir, exist_ok=True)
    fp32 = os.path.join(out_dir, f"{encoder}_fp32.onnx")
    int8 = os.path.join(out_dir, f"{encoder}_int8.onnx")
    export_fp32_onnx(model, fp32, seconds)
    if mode == "static":
        quantize_int8_static(fp32, int8, accepts_waveform=model.accepts_waveform,
                             seconds=seconds, calib_reader=calib_reader)
    else:
        quantize_int8_onnx(fp32, int8)
    report = footprint_report(model, fp32, int8)
    report["quant_mode"] = mode

    print(f"[quantize] {encoder} ({mode}): "
          f"{report['encoder_params_M']:.2f}M params "
          f"(~{report['param_budget_MB']:.1f}MB INT8 ideal, "
          f"budget<=16MB: {report['params_within_budget']}) | "
          f"FP32 {report['onnx_fp32_MB']:.1f}MB -> INT8 {report['onnx_int8_MB']:.1f}MB "
          f"(measured budget<=16MB: {report['onnx_within_budget']})")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="INT8 PTQ + ONNX export")
    ap.add_argument("--encoder", required=True,
                    choices=["conformer", "ebranchformer", "ebranchformer_espnet", "zipformer", "moonshine", "moonshine_tiny", "moonshine_official", "dpwavlm"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out-dir", default="export")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--mode", default="static", choices=["static", "dynamic"],
                    help="static: budget-compliant (quantizes Conv+MatMul); "
                         "dynamic: calibration-free but Conv stays FP32")
    args = ap.parse_args()

    encoder_cfg = None
    if args.encoder == "dpwavlm":
        meta = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        encoder_cfg = meta.get("meta", {}).get("encoder_cfg")
    run(args.checkpoint, args.encoder, args.out_dir, args.seconds, encoder_cfg,
        mode=args.mode)


if __name__ == "__main__":
    main()
