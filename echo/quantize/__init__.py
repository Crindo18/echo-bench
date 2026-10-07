"""INT8 quantization + ONNX export (the deployment normaliser, spec Section 4)."""
from .quantize_export import (
    EmbeddingExport, export_fp32_onnx, quantize_int8_onnx, quantize_int8_static,
    footprint_report, file_size_mb, run,
)

__all__ = [
    "EmbeddingExport", "export_fp32_onnx", "quantize_int8_onnx",
    "quantize_int8_static", "footprint_report", "file_size_mb", "run",
]
