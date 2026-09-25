"""Artifact folders: artifacts/models/<name>/<sha8>/ with model.onnx + model_card.json
(blueprint section 4.1).

A folder is named after the first 8 hex characters of its model file's SHA-256 and
is never changed once written. Rebuilding identical weights finds the existing
folder and leaves it alone.
"""

import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from echo_core.model_card import CARD_FILENAME, ModelCard, load_card, sha256_file
from echo_train.build_model import ENCODER_IMPL
from echo_train.config import ModelConfig


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def base_card_fields(
    config: ModelConfig, name: str, param_count: int, export_info: dict[str, Any]
) -> dict[str, Any]:
    """Card fields shared by every variant built from one recipe."""
    return {
        "name": name,
        "weights_state": "random_init",
        "seed": config.seed,
        "param_count": param_count,
        "io": {
            "inputs": {"feats": ["B", "T", config.input_size], "feats_lens": ["B"]},
            "outputs": {"embedding": ["B", config.encoder.output_size]},
        },
        "frontend": None,  # random weights have no training recipe yet (filled in M4)
        "encoder": {"impl": ENCODER_IMPL, "input_size": config.input_size}
        | config.encoder.model_dump(),
        "embedding": {
            "pooling": config.embedding.pooling,
            "dim": config.encoder.output_size,
            "l2_normalized": config.embedding.l2_normalized,
        },
        "export": export_info,
    }


def write_artifact(
    out_root: Path, model_path: Path, card_fields: dict[str, Any]
) -> tuple[Path, bool]:
    """Move model_path into out_root/<sha8>/ next to its card. Returns (folder, newly_created)."""
    sha = sha256_file(model_path)
    folder = out_root / sha[:8]
    if (folder / CARD_FILENAME).exists():
        if load_card(folder).artifact_sha256 != sha:
            raise RuntimeError(f"{folder} already holds a different model")
        model_path.unlink()
        return folder, False
    card = ModelCard.model_validate(
        card_fields | {"created_at": utc_now(), "files_sha256": {"model.onnx": sha}}
    )
    folder.mkdir(parents=True, exist_ok=True)
    shutil.move(str(model_path), folder / card.artifact_file)
    (folder / CARD_FILENAME).write_text(card.to_json(), encoding="utf-8")
    return folder, True
