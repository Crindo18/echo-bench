"""INT8 versions of an FP32 ONNX model (blueprint section 6.3).

Preprocessing note (found in M1): with default settings, ONNX Runtime's
preprocessing stops with "Incomplete symbolic shape inference" on this model,
and auto_merge=True gets past that but produces an INT8 model that crashes at
run time. skip_symbolic_shape=True is the setting that works.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)
from onnxruntime.quantization.registry import QDQRegistry
from onnxruntime.quantization.shape_inference import quant_pre_process

from echo_train.config import ModelConfig
from echo_train.export_onnx import synthetic_features

PREPROCESS = {"skip_symbolic_shape": True}
VARIANT_OF = {"dynamic": "onnx_int8_dynamic", "static_qdq": "onnx_int8_static"}

# ONNX Runtime's histogram calibrators (Entropy, Percentile) stack every clip's
# activations into one array, so they crash when clips differ in length (M1
# finding). They get equal-length clips; MinMax handles mixed lengths.
HISTOGRAM_METHODS = ("Entropy", "Percentile")
HISTOGRAM_FIXED_LENGTH_S = 3.0

# ESPnet masks attention with a constant of -3.4e38 (the float32 minimum) through
# Where nodes. The histogram calibrators overflow on it, and a mask gains nothing
# from quantization, so every static method skips Where. Keeping the op list the
# same for MinMax, Entropy and Percentile keeps their comparison fair (M1 finding).
STATIC_EXCLUDED_OP_TYPES = ["Where"]
STATIC_OP_TYPES = sorted(op for op in QDQRegistry if op not in STATIC_EXCLUDED_OP_TYPES)


class SyntheticCalibrationReader(CalibrationDataReader):  # type: ignore[misc]
    """Random features of realistic lengths. Only valid for random-weight models (M1).
    From M4, calibration must use features from TRAINING speakers only (rule L3),
    cropped to a fixed length for the histogram methods."""

    def __init__(self, config: ModelConfig):
        calibration = config.quantization.calibration
        if calibration.method in HISTOGRAM_METHODS:
            lengths = np.full(calibration.n_utts, HISTOGRAM_FIXED_LENGTH_S)
        else:
            lengths = np.random.default_rng(config.seed + 1).uniform(1, 8, calibration.n_utts)
        self._items = iter(
            synthetic_features(s, config.input_size, seed=1000 + i) for i, s in enumerate(lengths)
        )

    def get_next(self) -> dict[str, np.ndarray] | None:
        return next(self._items, None)


def preprocess(fp32_path: Path, out_path: Path) -> None:
    """Same as `python -m onnxruntime.quantization.preprocess`, with the setting that works here."""
    quant_logger = logging.getLogger("onnxruntime.tools.symbolic_shape_infer")
    quant_logger.setLevel(logging.ERROR)
    quant_pre_process(str(fp32_path), str(out_path), **PREPROCESS)


def quantize(prep_path: Path, out_path: Path, method: str, config: ModelConfig) -> dict[str, Any]:
    """Write one INT8 version and return its model-card 'quantization' block."""
    q = config.quantization
    info: dict[str, Any] = {
        "method": method,
        "weight_type": "int8",
        "per_channel": q.per_channel,
        "preprocess": PREPROCESS,
        "onnxruntime": ort.__version__,
    }
    root = logging.getLogger()
    previous = root.level
    root.setLevel(logging.ERROR)  # hide "Expected bias ... to be an initializer" chatter
    try:
        if method == "dynamic":
            quantize_dynamic(
                str(prep_path),
                str(out_path),
                weight_type=QuantType.QInt8,
                per_channel=q.per_channel,
            )
        elif method == "static_qdq":
            quantize_static(
                str(prep_path),
                str(out_path),
                SyntheticCalibrationReader(config),
                quant_format=QuantFormat.QDQ,
                per_channel=q.per_channel,
                activation_type=QuantType.QUInt8,
                weight_type=QuantType.QInt8,
                calibrate_method=getattr(CalibrationMethod, q.calibration.method),
                op_types_to_quantize=STATIC_OP_TYPES,
            )
            info["activation_type"] = "uint8"
            info["op_types_excluded"] = STATIC_EXCLUDED_OP_TYPES
            info["calibration"] = q.calibration.model_dump()
            if q.calibration.method in HISTOGRAM_METHODS:
                info["calibration"]["fixed_length_s"] = HISTOGRAM_FIXED_LENGTH_S
        else:
            raise ValueError(f"unknown quantization method {method!r}")
    finally:
        root.setLevel(previous)
    return info


def int8_sanity(fp32_path: Path, int8_path: Path, config: ModelConfig) -> float:
    """Run the INT8 model at every parity length; return the lowest cosine vs FP32.
    With random weights this only proves the graph still runs; real fidelity is M4."""
    fp32 = ort.InferenceSession(str(fp32_path), providers=["CPUExecutionProvider"])
    int8 = ort.InferenceSession(str(int8_path), providers=["CPUExecutionProvider"])
    lowest = 1.0
    for seconds in config.export.parity_lengths_s:
        inputs = synthetic_features(seconds, config.input_size, seed=200 + int(seconds * 10))
        a = fp32.run(None, inputs)[0][0]
        b = int8.run(None, inputs)[0][0]
        lowest = min(lowest, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b))))
    return lowest
