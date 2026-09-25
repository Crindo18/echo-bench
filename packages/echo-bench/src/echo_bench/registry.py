"""Model registry: register, resolve and inspect artifacts (blueprint sections 4.1 and 4.4, M1).

Registering checks every file against its model card's SHA-256, records the
artifact in the database, and evaluates the gates that depend only on the file:
G1 (INT8 size) and G2a (export parity, taken from the FP32 card).
"""

import collections
import json
import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import onnx
from onnx import numpy_helper
from sqlalchemy import select
from sqlalchemy.orm import Session

from echo_bench.db.models import GateResult, ModelArtifact
from echo_bench.gates import Gate, record_gate
from echo_bench.inputs import synthetic_inputs
from echo_core.model_card import (
    CARD_FILENAME,
    INT8_VARIANTS,
    ModelCard,
    load_card,
    sha256_file,
    verify_files,
)

QUANTIZED_OPS = {
    "MatMulInteger",
    "ConvInteger",
    "QLinearMatMul",
    "QLinearConv",
    "DynamicQuantizeMatMul",
    "MatMulIntegerToFloat",
    "QGemm",
}
WEIGHT_OPS = {"MatMul", "Conv", "Gemm"}


class RegistryError(Exception):
    pass


@dataclass
class Registration:
    folder: Path
    artifact: ModelArtifact
    created: bool
    gates: list[GateResult] = field(default_factory=list)


def find_artifact_folders(paths: list[Path]) -> list[Path]:
    """Each path is an artifact folder, or a folder searched for model_card.json files."""
    folders: list[Path] = []
    for path in paths:
        if (path / CARD_FILENAME).is_file():
            folders.append(path)
        elif path.is_dir():
            folders.extend(sorted(p.parent for p in path.rglob(CARD_FILENAME)))
        else:
            raise RegistryError(f"{path} is not a folder")
    if not folders:
        raise RegistryError(f"no {CARD_FILENAME} found under {', '.join(map(str, paths))}")
    return list(dict.fromkeys(folders))


def _relative(path: Path) -> str:
    try:
        return Path(os.path.relpath(path)).as_posix()
    except ValueError:  # another drive on Windows
        return path.resolve().as_posix()


def _by_sha(session: Session, sha256: str) -> ModelArtifact | None:
    return session.scalar(select(ModelArtifact).where(ModelArtifact.sha256 == sha256))


def register_folder(session: Session, folder: Path, gates: dict[str, Gate]) -> Registration:
    card = load_card(folder)
    try:
        verify_files(card, folder)  # validity rule 3: files must match the card
    except ValueError as error:
        raise RegistryError(str(error)) from error
    existing = _by_sha(session, card.artifact_sha256)
    if existing is not None:
        return Registration(folder, existing, created=False)

    parent_id = None
    if card.parent_sha256:
        parent = _by_sha(session, card.parent_sha256)
        if parent is None:
            raise RegistryError(f"{folder}: register its parent ({card.parent_sha256[:8]}) first")
        parent_id = parent.id

    model_path = folder / card.artifact_file
    opset = None
    if model_path.suffix == ".onnx":
        graph = onnx.load(str(model_path), load_external_data=False)
        opset = max(
            (o.version for o in graph.opset_import if o.domain in ("", "ai.onnx")), default=None
        )
    artifact = ModelArtifact(
        name=card.name,
        variant=card.variant,
        parent_id=parent_id,
        weights_state=card.weights_state,
        file_path=_relative(model_path),
        sha256=card.artifact_sha256,
        size_bytes=model_path.stat().st_size,
        param_count=card.param_count,
        embedding_dim=card.embedding.dim,
        opset=opset,
        model_card_json=card.model_dump_json(),
    )
    session.add(artifact)
    session.flush()

    results = []
    if card.variant in INT8_VARIANTS and "G1_footprint" in gates:
        size_mb = artifact.size_bytes / 1e6
        results.append(
            record_gate(session, gates["G1_footprint"], size_mb, model_artifact_id=artifact.id)
        )
    if card.variant == "onnx_fp32" and card.export is not None and "G2a_export_parity" in gates:
        parity = card.export.parity_max_abs
        results.append(
            record_gate(session, gates["G2a_export_parity"], parity, model_artifact_id=artifact.id)
        )
    return Registration(folder, artifact, created=True, gates=results)


def register_paths(
    session: Session, paths: list[Path], gates: dict[str, Gate]
) -> list[Registration]:
    """Register every artifact found, parents before children."""
    pending = {folder: load_card(folder) for folder in find_artifact_folders(paths)}
    pending_shas = {card.artifact_sha256 for card in pending.values()}
    registrations = []
    while pending:
        ready = [
            folder
            for folder, card in pending.items()
            if card.parent_sha256 is None or card.parent_sha256 not in pending_shas
        ]
        if not ready:
            raise RegistryError("circular parent links between model cards")
        for folder in ready:
            registrations.append(register_folder(session, folder, gates))
            pending_shas.discard(pending.pop(folder).artifact_sha256)
    return registrations


def resolve(session: Session, ref: str) -> ModelArtifact:
    """'name:variant' (newest registration wins) or a SHA-256 prefix of at least 6 characters."""
    if ":" in ref:
        name, variant = ref.split(":", 1)
        query = (
            select(ModelArtifact)
            .where(ModelArtifact.name == name, ModelArtifact.variant == variant)
            .order_by(ModelArtifact.created_at.desc())
        )
        found = session.scalar(query)
    else:
        if len(ref) < 6:
            raise RegistryError(f"{ref!r}: SHA-256 prefixes need at least 6 characters")
        matches = session.scalars(
            select(ModelArtifact).where(ModelArtifact.sha256.startswith(ref.lower()))
        ).all()
        if len(matches) > 1:
            raise RegistryError(f"{ref!r} matches {len(matches)} models; use a longer prefix")
        found = matches[0] if matches else None
    if found is None:
        raise RegistryError(
            f"no registered model matches {ref!r}; run `echo-bench model register` first"
        )
    return found


def verify_artifact_file(artifact: ModelArtifact) -> None:
    """Validity rule 3: the file on disk must still be the file that was registered."""
    path = Path(artifact.file_path)
    if not path.is_file():
        raise RegistryError(f"{path} is missing (copy the artifacts to this machine first)")
    if sha256_file(path) != artifact.sha256:
        raise RegistryError(f"{path} changed since it was registered (SHA-256 mismatch)")


def card_of(artifact: ModelArtifact) -> ModelCard:
    return ModelCard.model_validate_json(artifact.model_card_json)


def _stored_arrays(graph: onnx.GraphProto) -> dict[str, Any]:
    arrays = {t.name: numpy_helper.to_array(t) for t in graph.initializer}
    for node in graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value":
                    arrays[node.output[0]] = numpy_helper.to_array(attr.t)
    return arrays


def inspect_onnx(path: Path) -> dict[str, Any]:
    """Op histogram, quantization coverage, storage by data type, largest tensors, inputs/outputs."""
    model = onnx.load(str(path))
    graph = model.graph
    arrays = _stored_arrays(graph)
    producer = {out: node for node in graph.node for out in node.output}

    def weight_kind(name: str) -> str | None:
        """'int8' or 'float' if this input is a stored weight, else None (an activation)."""
        if name in arrays:
            return "int8" if arrays[name].dtype.itemsize == 1 else "float"
        node = producer.get(name)
        if node is not None and node.op_type == "DequantizeLinear" and node.input[0] in arrays:
            return "int8"
        return None

    total: collections.Counter[str] = collections.Counter()
    quantized: collections.Counter[str] = collections.Counter()
    for node in graph.node:
        family = "Conv" if "Conv" in node.op_type else "MatMul"
        if node.op_type in QUANTIZED_OPS:
            if any(name in arrays for name in node.input):
                total[family] += 1
                quantized[family] += 1
        elif node.op_type in WEIGHT_OPS:
            kinds = [weight_kind(name) for name in node.input[:2]]
            if any(kinds):
                total[family] += 1
                quantized[family] += "int8" in kinds

    bytes_by_dtype: collections.Counter[str] = collections.Counter()
    for array in arrays.values():
        bytes_by_dtype[str(array.dtype)] += array.nbytes
    largest = sorted(arrays.items(), key=lambda item: item[1].nbytes, reverse=True)[:5]

    def shape(value_info: onnx.ValueInfoProto) -> list[str | int]:
        dims = value_info.type.tensor_type.shape.dim
        return [d.dim_param or d.dim_value for d in dims]

    return {
        "opset": max(
            (o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), default=None
        ),
        "inputs": {vi.name: shape(vi) for vi in graph.input if vi.name not in arrays},
        "outputs": {vi.name: shape(vi) for vi in graph.output},
        "op_histogram": collections.Counter(node.op_type for node in graph.node).most_common(),
        "coverage": {family: (quantized[family], total[family]) for family in ("MatMul", "Conv")},
        "bytes_by_dtype": dict(bytes_by_dtype.most_common()),
        "largest": [(name, str(a.dtype), list(a.shape), a.nbytes) for name, a in largest],
    }


# Operators that do the multiply-accumulate work, and which input holds the weights.
_MATMUL_OPS = {"MatMul": 1, "MatMulInteger": 1, "QLinearMatMul": 3, "Gemm": 1}
_CONV_OPS = {"Conv": 1, "ConvInteger": 1, "QLinearConv": 3}


def _shapes(entries: list[dict[str, list[int]]]) -> list[list[int]]:
    return [list(next(iter(entry.values()))) for entry in entries]


def count_macs(path: Path, card: ModelCard, seconds: float) -> tuple[int, dict[str, int]]:
    """Multiply-accumulates for one utterance of this length, from the shapes ONNX Runtime
    actually ran (profiling, graph optimizations off). Compute is a property of the model,
    so the count is the same on every machine: a desktop-measurable predictor of Pi speed.
    Returns the total and a breakdown by module (block numbers replaced by *)."""
    import onnxruntime as ort

    with tempfile.TemporaryDirectory() as tmp:
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.enable_profiling = True
        options.profile_file_prefix = os.path.join(tmp, "profile")
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        session.run(None, synthetic_inputs(card, seconds))
        events = json.loads(Path(session.end_profiling()).read_text())

    groups: collections.Counter[str] = collections.Counter()
    for event in events:
        args = event.get("args", {})
        op = args.get("op_name")
        if event.get("cat") != "Node" or "input_type_shape" not in args:
            continue
        if op not in _MATMUL_OPS and op not in _CONV_OPS:
            continue
        inputs, output = _shapes(args["input_type_shape"]), _shapes(args["output_type_shape"])[0]
        if op in _CONV_OPS:
            macs = math.prod(output) * math.prod(inputs[_CONV_OPS[op]][1:])
        elif op == "Gemm":
            macs = math.prod(output) * (math.prod(inputs[0]) // output[0])
        else:
            macs = math.prod(output) * inputs[0][-1]
        node = event["name"].removesuffix("_kernel_time").strip("/")
        groups[re.sub(r"\d+", "*", "/".join(node.split("/")[:3]))] += macs
    return sum(groups.values()), dict(groups.most_common())
