"""Shared test fixtures."""

import shutil
from collections.abc import Callable, Iterator
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from echo_bench.db.migrate import upgrade_to_head
from echo_bench.db.session import make_engine, make_session
from echo_core.model_card import CARD_FILENAME, ModelCard, sha256_file


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A fresh results database, created through the real migrations."""
    path = tmp_path / "bench_test.db"
    upgrade_to_head(path)
    return path


@pytest.fixture
def engine(db_path: Path) -> Iterator[Engine]:
    engine = make_engine(db_path)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with make_session(engine) as session:
        yield session


# ------------------------------------------------------------- M1 fixtures# A tiny stand-in for the encoder (mean over time, then one matrix multiply)
# with the same inputs and outputs as the real model. It builds in milliseconds,
# so the registry and perf tests don't need PyTorch.

REPO_ROOT = Path(__file__).resolve().parents[1]
TINY_DIM = 16


def build_tiny_onnx(path: Path, n_mels: int = 80, dim: int = TINY_DIM) -> None:
    weight = np.random.default_rng(0).standard_normal((n_mels, dim)).astype(np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("ReduceMean", ["feats"], ["pooled"], axes=[1], keepdims=0),
            helper.make_node("MatMul", ["pooled", "weight"], ["embedding"]),
        ],
        "tiny",
        [
            helper.make_tensor_value_info("feats", TensorProto.FLOAT, ["B", "T", n_mels]),
            helper.make_tensor_value_info("feats_lens", TensorProto.INT64, ["B"]),
        ],
        [helper.make_tensor_value_info("embedding", TensorProto.FLOAT, ["B", dim])],
        [numpy_helper.from_array(weight, "weight")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, str(path))


def tiny_card_fields(variant: str, **extra: object) -> dict[str, object]:
    return {
        "name": "tiny-rand",
        "variant": variant,
        "weights_state": "random_init",
        "seed": 0,
        "param_count": 80 * TINY_DIM,
        "io": {
            "inputs": {"feats": ["B", "T", 80], "feats_lens": ["B"]},
            "outputs": {"embedding": ["B", TINY_DIM]},
        },
        "encoder": {"impl": "tests.tiny", "input_size": 80},
        "embedding": {"dim": TINY_DIM},
        "export": {
            "torch": "none",
            "exporter": "torchscript",
            "opset": 17,
            "dynamic_axes": ["B", "T"],
            "trace_length_s": 3,
            "parity_lengths_s": [1, 3],
            "parity_max_abs": 2e-7,
        },
        "created_at": "2026-09-25T00:00:00.000Z",
    } | extra


def write_artifact(models_dir: Path, onnx_path: Path, fields: dict[str, object]) -> Path:
    sha = sha256_file(onnx_path)
    folder = models_dir / sha[:8]
    folder.mkdir(parents=True)
    shutil.move(str(onnx_path), folder / "model.onnx")
    card = ModelCard.model_validate(fields | {"files_sha256": {"model.onnx": sha}})
    (folder / CARD_FILENAME).write_text(card.to_json())
    return folder


@pytest.fixture
def tiny_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    """artifacts/models/tiny-rand/<sha8>/ for an FP32 model and its dynamic INT8 version,
    inside a temporary working folder that also holds configs/gates.yaml."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    monkeypatch.chdir(tmp_path)
    (tmp_path / "configs").mkdir()
    shutil.copy(REPO_ROOT / "configs" / "gates.yaml", tmp_path / "configs" / "gates.yaml")
    models_dir = tmp_path / "artifacts" / "models" / "tiny-rand"
    models_dir.mkdir(parents=True)

    build_tiny_onnx(tmp_path / "fp32.onnx")
    quantize_dynamic(
        str(tmp_path / "fp32.onnx"), str(tmp_path / "int8.onnx"), weight_type=QuantType.QInt8
    )
    fp32 = write_artifact(models_dir, tmp_path / "fp32.onnx", tiny_card_fields("onnx_fp32"))
    fp32_sha = sha256_file(fp32 / "model.onnx")
    quantization = {"method": "dynamic", "onnxruntime": "test"}
    int8 = write_artifact(
        models_dir,
        tmp_path / "int8.onnx",
        tiny_card_fields("onnx_int8_dynamic", parent_sha256=fp32_sha, quantization=quantization),
    )
    return {"root": tmp_path, "models": models_dir, "fp32": fp32, "int8": int8}


@pytest.fixture
def card_fields() -> Callable[..., dict[str, object]]:
    """Valid model-card fields for the tiny model (tests add files_sha256 or overrides)."""
    return tiny_card_fields


# ------------------------------------------------------------- M3 fixtures
# Tiny fake corpora with the real TORGO and UASpeech layouts and file names.


def _wav(path: Path, seconds: float = 0.2, rate: int = 16000) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    t = np.arange(int(seconds * rate)) / rate
    sf.write(str(path), (0.1 * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate)


@pytest.fixture
def torgo_root(tmp_path: Path) -> Path:
    """F01 (severe) with 4 sessions of 'yes', M03 (mild), and two controls."""
    root = tmp_path / "TORGO"
    prompts = {
        "0001": "yes",
        "0002": "No.",
        "0003": "[relax your mouth in its normal position]",
        "0004": "input/images/cat.jpg",
        "0005": "The quick brown fox",
    }
    for group, speaker, sessions in (
        ("F", "F01", 4),
        ("M", "M03", 1),
        ("FC", "FC01", 1),
        ("MC", "MC01", 1),
    ):
        for n in range(1, sessions + 1):
            session = root / group / speaker / f"Session{n}"
            for pid, prompt in prompts.items():
                (session / "prompts").mkdir(parents=True, exist_ok=True)
                (session / "prompts" / f"{pid}.txt").write_text(prompt + "\n")
                _wav(session / "wav_arrayMic" / f"{pid}.wav")
                if not (speaker == "M03" and pid == "0002"):  # one prompt only on the array mic
                    _wav(session / "wav_headMic" / f"{pid}.wav")
    (root / "M" / "M03" / "Session1" / "wav_headMic" / "0005.wav").write_bytes(b"not audio")
    return root


@pytest.fixture
def uaspeech_root(tmp_path: Path) -> Path:
    root = tmp_path / "UASpeech" / "audio"
    words = ["D1", "LA", "C1", "CW1"]
    for speaker in ("F02", "M04", "M08", "CF02", "CM01"):
        for block, uncommon in (("B1", "UW1"), ("B2", "UW101"), ("B3", "UW201")):
            for word in [*words, uncommon]:
                mics = ["M2", "M8"] if (speaker == "M08" and word == "LA") else ["M5", "M2"]
                for mic in mics:
                    _wav(root / speaker / f"{speaker}_{block}_{word}_{mic}.wav")
        _wav(root / speaker / f"{speaker}_B1_D1_M1.wav")  # sync-tone mic, always skipped
    (root / "README.txt").write_text("not audio")
    return root


@pytest.fixture
def speaker_tables() -> dict[str, Path]:
    return {
        "torgo": REPO_ROOT / "configs" / "datasets" / "torgo_speakers.csv",
        "uaspeech": REPO_ROOT / "configs" / "datasets" / "uaspeech_speakers.csv",
    }
