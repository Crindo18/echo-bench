"""Model registry (M1): registration, gates, lookup, tamper refusal and inspection."""

from pathlib import Path

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from echo_bench.db.models import GateResult
from echo_bench.gates import load_gates
from echo_bench.registry import (
    RegistryError,
    count_macs,
    inspect_onnx,
    register_paths,
    resolve,
    verify_artifact_file,
)
from echo_core.model_card import load_card


def test_register_records_gates_once(session: Session, tiny_artifacts: dict[str, Path]) -> None:
    gates = load_gates()
    first = register_paths(session, [tiny_artifacts["models"]], gates)
    session.commit()
    by_variant = {r.artifact.variant: r for r in first}
    assert all(r.created for r in first)
    fp32, int8 = by_variant["onnx_fp32"].artifact, by_variant["onnx_int8_dynamic"].artifact
    assert int8.parent_id == fp32.id  # the INT8 file points at the FP32 file it came from
    assert int8.file_path.startswith("artifacts/models/tiny-rand/")

    rows = session.scalars(select(GateResult)).all()
    assert {(r.gate, r.passed) for r in rows} == {("G1_footprint", 1), ("G2a_export_parity", 1)}

    again = register_paths(session, [tiny_artifacts["models"]], gates)
    session.commit()
    assert not any(r.created for r in again)
    assert len(session.scalars(select(GateResult)).all()) == 2  # no duplicate gate rows


def test_resolve_by_name_or_hash(session: Session, tiny_artifacts: dict[str, Path]) -> None:
    register_paths(session, [tiny_artifacts["models"]], load_gates())
    int8 = resolve(session, "tiny-rand:onnx_int8_dynamic")
    assert resolve(session, int8.sha256[:8]).id == int8.id
    with pytest.raises(RegistryError, match="at least 6"):
        resolve(session, "abc")
    with pytest.raises(RegistryError, match="no registered model"):
        resolve(session, "tiny-rand:onnx_int8_static")


def test_changed_files_are_refused(session: Session, tiny_artifacts: dict[str, Path]) -> None:
    register_paths(session, [tiny_artifacts["fp32"]], load_gates())
    verify_artifact_file(resolve(session, "tiny-rand:onnx_fp32"))
    with (tiny_artifacts["fp32"] / "model.onnx").open("ab") as handle:
        handle.write(b"\0")
    with pytest.raises(RegistryError, match="changed since it was registered"):
        verify_artifact_file(resolve(session, "tiny-rand:onnx_fp32"))
    with pytest.raises(RegistryError, match="does not match"):
        register_paths(session, [tiny_artifacts["fp32"]], load_gates())


def test_children_need_their_parent(session: Session, tiny_artifacts: dict[str, Path]) -> None:
    with pytest.raises(RegistryError, match="parent"):
        register_paths(session, [tiny_artifacts["int8"]], load_gates())


def test_inspect_reports_quantization_coverage(tiny_artifacts: dict[str, Path]) -> None:
    fp32 = inspect_onnx(tiny_artifacts["fp32"] / "model.onnx")
    int8 = inspect_onnx(tiny_artifacts["int8"] / "model.onnx")
    assert fp32["coverage"]["MatMul"] == (0, 1)
    assert int8["coverage"]["MatMul"] == (1, 1)
    assert fp32["inputs"] == {"feats": ["B", "T", 80], "feats_lens": ["B"]}
    assert "int8" in int8["bytes_by_dtype"]


def test_compute_count_matches_the_arithmetic(tiny_artifacts: dict[str, Path]) -> None:
    # The tiny model averages over time, then multiplies [1, 80] by [80, 16]: 80 x 16 MACs,
    # whatever the length, and the same for the INT8 version (MatMulInteger).
    for key in ("fp32", "int8"):
        folder = tiny_artifacts[key]
        for seconds in (1, 4):
            total, groups = count_macs(folder / "model.onnx", load_card(folder), seconds)
            assert total == 80 * 16
            assert sum(groups.values()) == total
