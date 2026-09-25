"""The model card: the contract between echo-train and everything downstream (blueprint section 4.4).

Every model folder, artifacts/models/<name>/<sha8>/, holds the model file plus a
model_card.json describing it. The registry validates the card against this
schema and refuses the folder if any file's SHA-256 doesn't match the card.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

CARD_FILENAME = "model_card.json"
Variant = Literal[
    "torch_fp32", "onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static", "onnx_int8_mixed"
]
INT8_VARIANTS = ("onnx_int8_dynamic", "onnx_int8_static", "onnx_int8_mixed")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")  # a typo in a card is an error, not a silent extra


class IOSpec(_Strict):
    inputs: dict[str, list[str | int]]
    outputs: dict[str, list[str | int]]


class EmbeddingSpec(_Strict):
    pooling: Literal["masked_mean"] = "masked_mean"
    dim: int = Field(gt=0)
    l2_normalized: bool = False


class ExportInfo(_Strict):
    torch: str
    exporter: Literal["torchscript", "dynamo"]
    opset: int
    dynamic_axes: list[str]
    trace_length_s: float
    parity_lengths_s: list[float]
    parity_max_abs: float  # worst PyTorch-vs-ONNX difference over parity_lengths_s (gate G2a)
    parity_tolerance: float = 1e-4

    @property
    def parity_passed(self) -> bool:
        return self.parity_max_abs <= self.parity_tolerance


class CalibrationInfo(_Strict):
    n_utts: int = Field(gt=0)
    source: str  # 'synthetic' (M1 only), or e.g. 'fold0:train': TRAINING speakers only (rule L3)
    method: Literal["MinMax", "Entropy", "Percentile"]
    fixed_length_s: float | None = None  # Entropy/Percentile need equal-length clips (M1 finding)


class QuantizationInfo(_Strict):
    method: Literal["dynamic", "static_qdq", "mixed"]
    weight_type: Literal["int8", "uint8"] = "int8"
    activation_type: Literal["int8", "uint8"] | None = None  # static and mixed only
    per_channel: bool = True
    calibration: CalibrationInfo | None = None
    nodes_excluded: list[str] = Field(default_factory=list)
    op_types_excluded: list[str] = Field(default_factory=list)
    preprocess: dict[str, Any] = Field(default_factory=dict)
    onnxruntime: str


class TrainingInfo(_Strict):
    cv_plan: str | None = None
    fold: int | None = None
    corpora: list[str] = Field(default_factory=list)
    git_commit: str | None = None


class ModelCard(_Strict):
    schema_version: Literal[1] = 1
    name: str
    variant: Variant
    weights_state: Literal["random_init", "trained"]
    parent_sha256: str | None = None  # an INT8 model points at its FP32 source
    seed: int | None = None
    param_count: int | None = None
    io: IOSpec
    frontend: dict[str, Any] | None = None  # copied from the training recipe (M4)
    encoder: dict[str, Any]
    embedding: EmbeddingSpec
    export: ExportInfo | None = None
    quantization: QuantizationInfo | None = None
    training: TrainingInfo | None = None
    created_at: str
    files_sha256: dict[str, str]

    @model_validator(mode="after")
    def _check_consistency(self) -> ModelCard:
        if self.variant.startswith("onnx") and self.export is None:
            raise ValueError("ONNX variants need an 'export' block")
        if self.variant in INT8_VARIANTS and self.quantization is None:
            raise ValueError("INT8 variants need a 'quantization' block")
        if self.variant == "onnx_int8_static" and (
            self.quantization is None or self.quantization.calibration is None
        ):
            raise ValueError("static quantization needs a 'calibration' block")
        if self.weights_state == "trained" and self.frontend is None:
            raise ValueError(
                "trained models need a 'frontend' block copied from the training recipe (section 4.4)"
            )
        if self.artifact_file not in self.files_sha256:
            raise ValueError(f"files_sha256 must include {self.artifact_file}")
        return self

    @property
    def artifact_file(self) -> str:
        return "model.pt" if self.variant == "torch_fp32" else "model.onnx"

    @property
    def artifact_sha256(self) -> str:
        return self.files_sha256[self.artifact_file]

    @property
    def ref(self) -> str:
        """How configs refer to this model, for example 'ebf-12m-rand:onnx_int8_static'."""
        return f"{self.name}:{self.variant}"

    def to_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=2) + "\n"


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_card(folder: str | Path) -> ModelCard:
    text = (Path(folder) / CARD_FILENAME).read_text(encoding="utf-8")
    return ModelCard.model_validate_json(text)


def verify_files(card: ModelCard, folder: str | Path) -> None:
    """Raise ValueError if a file listed in the card is missing or its SHA-256 differs."""
    for name, expected in card.files_sha256.items():
        path = Path(folder) / name
        if not path.is_file():
            raise ValueError(f"{path} is listed in the model card but missing")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(
                f"{path}: SHA-256 {actual[:12]} does not match the card ({expected[:12]})"
            )
