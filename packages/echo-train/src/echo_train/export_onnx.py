"""ONNX FP32 export and the multi-length parity test (blueprint section 6.3, gate G2a)."""

import warnings
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from echo_train.build_model import EmbeddingNet
from echo_train.config import ModelConfig

FRAMES_PER_SECOND = 100  # 10 ms hop


def synthetic_features(seconds: float, input_size: int, seed: int) -> dict[str, np.ndarray]:
    """Random 'log-mel' input of the right shape. Enough for export, size and speed tests."""
    frames = int(round(seconds * FRAMES_PER_SECOND))
    rng = np.random.default_rng(seed)
    return {
        "feats": rng.standard_normal((1, frames, input_size), dtype=np.float32),
        "feats_lens": np.array([frames], dtype=np.int64),
    }


def export_fp32(model: EmbeddingNet, config: ModelConfig, path: Path) -> None:
    """Export with dynamic batch (B) and time (T) axes, traced on one example length."""
    example = synthetic_features(config.export.trace_length_s, config.input_size, seed=0)
    with warnings.catch_warnings():  # tracing prints many harmless TracerWarnings
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model,
            (torch.from_numpy(example["feats"]), torch.from_numpy(example["feats_lens"])),
            str(path),
            input_names=["feats", "feats_lens"],
            output_names=["embedding"],
            dynamic_axes={"feats": {0: "B", 1: "T"}, "feats_lens": {0: "B"}, "embedding": {0: "B"}},
            opset_version=config.export.opset,
            dynamo=False,  # the TorchScript exporter; recorded in the model card
        )
    onnx.checker.check_model(str(path))


def check_parity(model: EmbeddingNet, config: ModelConfig, onnx_path: Path) -> dict[float, float]:
    """Max |PyTorch - ONNX| of the embedding at every parity length."""
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    results: dict[float, float] = {}
    for seconds in config.export.parity_lengths_s:
        inputs = synthetic_features(seconds, config.input_size, seed=100 + int(seconds * 10))
        with torch.no_grad():
            reference = model(*(torch.from_numpy(v) for v in inputs.values())).numpy()
        output = session.run(None, inputs)[0]
        results[seconds] = float(np.abs(reference - output).max())
    return results
