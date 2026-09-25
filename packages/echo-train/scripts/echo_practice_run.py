"""
echo_practice_run.py - the ECHO-Bench M1 loop in one file, for learning.

What it does (each step prints what it found):
  1. Builds a randomly-initialised E-Branchformer encoder + mean pooling.
     Random weights are fine here: file size and speed depend on the model's
     SHAPE, not on what it has learned.
  2. Counts parameters (in INT8, 1 parameter is about 1 byte on disk).
  3. Exports the model to ONNX, the portable format the Raspberry Pi will run.
  4. Parity check: PyTorch and ONNX must agree at many input lengths,
     including lengths the export never saw (blueprint gate G2a).
  5. Makes three INT8 versions (numbers stored in 1 byte instead of 4).
  6. Compares file sizes (gate G1: INT8 file <= 16 MB), shows the biggest
     things inside the file, and checks every INT8 version still runs.
  7. Times all versions with ONNX Runtime and reports p50 / p95 latency.

How to "refine" with it: change a value in CONFIG, run again, compare.
Tested with Python 3.12, torch 2.11.0, espnet 202610.post2, onnx 1.23.0,
onnxruntime 1.30.0. Takes a few minutes. A line saying "Failed to import
Flash Attention" is harmless (that is a GPU-only speed-up).

Run it from inside its uv project:
    uv run python echo_practice_run.py
"""

import logging
import os
import time
import warnings

# Exporting prints many "TracerWarning"/"DeprecationWarning" lines and the
# quantizers print shape-inference and "Expected bias" chatter. All expected
# for this model, so hide warnings and keep real errors.
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
import onnx
import onnxruntime as ort
import torch
from espnet2.asr.encoder.e_branchformer_encoder import EBranchformerEncoder
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ort.set_default_logger_severity(3)  # only show ONNX Runtime errors

# ---------------------------------------------------------------------------
# CONFIG - the knobs. Change these, rerun, compare.
# ---------------------------------------------------------------------------
CONFIG = dict(
    input_size=80,            # mel bins per 10 ms frame (must match the frontend)
    output_size=256,          # model width; parameters grow roughly with width squared
    attention_heads=4,
    num_blocks=8,             # model depth; size and latency grow roughly linearly
    cgmlp_linear_units=1024,  # width of the convolution branch
    cgmlp_conv_kernel=31,
    use_ffn=True,             # ESPnet's default is False; the blueprint's ~13M estimate assumes True
    macaron_ffn=True,         # (same)
    linear_units=512,         # feed-forward width (ignored when use_ffn=False)
    merge_conv_kernel=3,
    input_layer="conv2d",     # shrinks time 4x before the blocks
    max_pos_emb_len=5000,     # ESPnet default. EXERCISE: change to 500 and rerun (500 still covers ~20 s of audio)
)
ORT_THREADS = 4                     # the blueprint's desktop proxy uses 4 threads
PARITY_SECONDS = [1, 2, 3, 5, 8, 10]
LATENCY_SECONDS = [1, 3, 8]
WARMUP, ITERS = 10, 50              # the real harness uses 20 and 300
OUT_DIR = "practice_artifacts"
FRAMES_PER_SECOND = 100             # 10 ms hop -> 100 feature frames per second of audio
SIZE_BUDGET_MB = 16.0               # gate G1


class EmbeddingNet(torch.nn.Module):
    """Encoder + masked mean pooling -> one embedding (a list of numbers) per utterance."""

    def __init__(self, encoder: torch.nn.Module):
        super().__init__()
        self.encoder = encoder

    def forward(self, feats: torch.Tensor, feats_lens: torch.Tensor) -> torch.Tensor:
        hs, hlens, _ = self.encoder(feats, feats_lens)            # hs: [batch, frames, width]
        frame_idx = torch.arange(hs.size(1), device=hs.device)
        mask = (frame_idx[None, :] < hlens[:, None]).to(hs.dtype)  # 1 = real frame, 0 = padding
        summed = (hs * mask.unsqueeze(-1)).sum(dim=1)
        return summed / mask.sum(dim=1, keepdim=True).clamp(min=1.0)


def fake_features(seconds: float, seed: int) -> dict:
    """Random 'log-mel' input of the right SHAPE. Enough for size, speed and export tests."""
    frames = int(seconds * FRAMES_PER_SECOND)
    rng = np.random.default_rng(seed)
    return {
        "feats": rng.standard_normal((1, frames, CONFIG["input_size"]), dtype=np.float32),
        "feats_lens": np.array([frames], dtype=np.int64),
    }


class FakeCalibrationReader(CalibrationDataReader):
    """Static quantization measures typical value ranges on sample inputs.
    This stand-in uses random features. The real one must use audio from
    TRAINING speakers only (blueprint leakage rule L3)."""

    def __init__(self, n: int = 16):
        lengths = np.random.default_rng(1).uniform(1, 8, n)
        self._items = iter([fake_features(s, seed=1000 + i) for i, s in enumerate(lengths)])

    def get_next(self):
        return next(self._items, None)


def mb(path: str) -> float:
    return os.path.getsize(path) / 1e6


def largest_tensors(path: str, n: int = 3) -> list:
    """What is taking up space inside an ONNX file (weights and stored constants)."""
    model = onnx.load(path)
    found = [(numpy_helper.to_array(t), t.name) for t in model.graph.initializer]
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    found.append((numpy_helper.to_array(attr.t), f"constant from node {node.name}"))
    return sorted(found, key=lambda item: item[0].nbytes, reverse=True)[:n]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    fp32_path = os.path.join(OUT_DIR, "model_fp32.onnx")
    prep_path = os.path.join(OUT_DIR, "model_fp32_prep.onnx")

    # 1. Build ---------------------------------------------------------------
    torch.manual_seed(0)
    encoder = EBranchformerEncoder(**CONFIG, use_flash_attn=False)
    model = EmbeddingNet(encoder).eval()  # eval() switches off training-only behaviour (dropout)

    # 2. Count parameters ----------------------------------------------------
    def n_params(m: torch.nn.Module) -> int:
        return sum(p.numel() for p in m.parameters())

    total = n_params(model)
    print("\n[1-2] Parameters")
    print(f"  input subsampling : {n_params(encoder.embed) / 1e6:6.2f} M")
    print(f"  one block         : {n_params(encoder.encoders[0]) / 1e6:6.2f} M  x {CONFIG['num_blocks']} blocks")
    print(f"  TOTAL             : {total / 1e6:6.2f} M  (~{total / 1e6:.1f} MB if every weight were INT8)")

    # 3. Export to ONNX (traced with a 3-second example) ------------------------
    example = fake_features(3, seed=0)
    torch.onnx.export(
        model,
        (torch.from_numpy(example["feats"]), torch.from_numpy(example["feats_lens"])),
        fp32_path,
        input_names=["feats", "feats_lens"],
        output_names=["embedding"],
        dynamic_axes={"feats": {0: "B", 1: "T"}, "feats_lens": {0: "B"}, "embedding": {0: "B"}},
        opset_version=17,
        dynamo=False,  # the classic TorchScript exporter; record this choice in the model card
    )

    # 4. Parity: PyTorch vs ONNX at many lengths (gate G2a: max |diff| <= 1e-4)
    print("\n[4] Export parity, PyTorch FP32 vs ONNX FP32 (the export only ever saw 3 s)")
    fp32_sess = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    worst = 0.0
    for sec in PARITY_SECONDS:
        inputs = fake_features(sec, seed=100 + sec)
        with torch.no_grad():
            ref = model(*(torch.from_numpy(v) for v in inputs.values())).numpy()
        diff = float(np.abs(ref - fp32_sess.run(None, inputs)[0]).max())
        worst = max(worst, diff)
        print(f"  {sec:>2} s : max |diff| = {diff:.2e}  {'PASS' if diff <= 1e-4 else 'FAIL'}")

    # 5. Three INT8 versions -------------------------------------------------------
    # Preprocess first (same as `python -m onnxruntime.quantization.preprocess`).
    # On this model the default settings stop with "Incomplete symbolic shape
    # inference", and auto_merge=True gets past that but produces an INT8 model
    # that crashes at run time. skip_symbolic_shape=True is the setting that works.
    quant_pre_process(fp32_path, prep_path, skip_symbolic_shape=True)
    variants = {
        "INT8 dynamic (all ops)": os.path.join(OUT_DIR, "model_int8_dynamic_all.onnx"),
        "INT8 dynamic (MatMul only)": os.path.join(OUT_DIR, "model_int8_dynamic_matmul.onnx"),
        "INT8 static QDQ": os.path.join(OUT_DIR, "model_int8_static_qdq.onnx"),
    }
    paths = list(variants.values())
    quantize_dynamic(prep_path, paths[0], weight_type=QuantType.QInt8, per_channel=True)
    quantize_dynamic(prep_path, paths[1], weight_type=QuantType.QInt8, per_channel=True,
                     op_types_to_quantize=["MatMul"])
    quantize_static(prep_path, paths[2], FakeCalibrationReader(), quant_format=QuantFormat.QDQ,
                    per_channel=True, activation_type=QuantType.QUInt8, weight_type=QuantType.QInt8,
                    calibrate_method=CalibrationMethod.MinMax)

    # 6. Sizes, what fills the file, and does every INT8 version still run? ----
    print(f"\n[5-6] File sizes (gate G1: INT8 <= {SIZE_BUDGET_MB:.0f} MB)")
    print(f"  {'FP32':28s} {mb(fp32_path):6.2f} MB")
    for name, path in variants.items():
        verdict = "PASS" if mb(path) <= SIZE_BUDGET_MB else "FAIL"
        int8_sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        lowest = min(
            cosine(fp32_sess.run(None, x)[0][0], int8_sess.run(None, x)[0][0])
            for x in (fake_features(s, seed=200 + s) for s in PARITY_SECONDS)
        )
        print(f"  {name:28s} {mb(path):6.2f} MB  G1 {verdict}   runs at all lengths, lowest cosine vs FP32 {lowest:.4f}")
    print("  (cosine here is only a 'did it break?' check; real fidelity needs a trained model and real audio)")
    print(f"  Biggest things inside '{os.path.basename(paths[0])}':")
    for arr, name in largest_tensors(paths[0]):
        print(f"    {arr.nbytes / 1e6:5.2f} MB  {str(arr.dtype):8s} {list(arr.shape)}  {name[:50]}")

    # 7. Latency with ONNX Runtime ---------------------------------------------
    print(f"\n[7] Encoder latency, {ORT_THREADS} threads, {ITERS} timed runs after {WARMUP} warm-ups")
    for name, path in [("FP32", fp32_path), *variants.items()]:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = ORT_THREADS
        sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        cells = []
        for sec in LATENCY_SECONDS:
            inputs = fake_features(sec, seed=sec)
            for _ in range(WARMUP):
                sess.run(None, inputs)
            samples_ms = []
            for _ in range(ITERS):
                t0 = time.perf_counter_ns()
                sess.run(None, inputs)
                samples_ms.append((time.perf_counter_ns() - t0) / 1e6)
            p50, p95 = np.percentile(samples_ms, [50, 95])
            cells.append(f"{sec} s: {p50:6.1f} / {p95:6.1f}")
        print(f"  {name:28s} " + "   ".join(cells) + "   (p50 / p95 ms)")

    print("\nDesktop latency is only a proxy; official speed numbers come from the Pi (blueprint P2).")
    print(f"Worst export-parity diff {worst:.2e} -> gate G2a {'PASS' if worst <= 1e-4 else 'FAIL'}\n")


if __name__ == "__main__":
    main()
