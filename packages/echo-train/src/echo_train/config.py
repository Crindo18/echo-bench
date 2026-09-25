"""The model recipe file, configs/models/*.yaml (blueprint section 4.4)."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EncoderConfig(_Strict):
    output_size: int = 256
    attention_heads: int = 4
    num_blocks: int = 8
    cgmlp_linear_units: int = 1024
    cgmlp_conv_kernel: int = 31
    use_ffn: bool = True
    macaron_ffn: bool = True
    linear_units: int = 512
    merge_conv_kernel: int = 3
    input_layer: str = "conv2d"
    max_pos_emb_len: int = 500


class EmbeddingConfig(_Strict):
    pooling: Literal["masked_mean"] = "masked_mean"
    l2_normalized: bool = False


class ExportConfig(_Strict):
    opset: int = 17
    trace_length_s: float = 3.0
    parity_lengths_s: list[float] = Field(default_factory=lambda: [1, 2, 3, 5, 8, 10])
    parity_tolerance: float = 1e-4


class CalibrationConfig(_Strict):
    n_utts: int = 16
    method: Literal["MinMax", "Entropy", "Percentile"] = "MinMax"
    source: str = "synthetic"


class QuantizationConfig(_Strict):
    methods: list[Literal["dynamic", "static_qdq"]] = Field(
        default_factory=lambda: ["dynamic", "static_qdq"]
    )
    per_channel: bool = True
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)


class ModelConfig(_Strict):
    name: str
    seed: int = 0
    input_size: int = 80
    encoder: EncoderConfig = Field(default_factory=EncoderConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    quantization: QuantizationConfig = Field(default_factory=QuantizationConfig)


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
