"""echo-train end to end on a tiny recipe, so the test takes seconds, not minutes.

Run inside packages/echo-train:  uv run pytest
"""

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from echo_core.model_card import load_card, verify_files
from echo_train.cli import app

TINY = {
    "name": "tiny",
    "seed": 0,
    "input_size": 80,
    "encoder": {
        "output_size": 64,
        "attention_heads": 2,
        "num_blocks": 1,
        "cgmlp_linear_units": 128,
        "linear_units": 64,
        "max_pos_emb_len": 300,
    },
    "export": {"parity_lengths_s": [1, 3, 5]},
    "quantization": {"methods": ["dynamic", "static_qdq"], "calibration": {"n_utts": 4}},
}


def test_build_random_then_quantize(tmp_path: Path) -> None:
    recipe = tmp_path / "tiny.yaml"
    recipe.write_text(yaml.safe_dump(TINY))
    out = tmp_path / "artifacts" / "tiny-rand"
    runner = CliRunner()

    result = runner.invoke(app, ["build-random", "--config", str(recipe), "--out", str(out)])
    assert result.exit_code == 0, result.output
    cards = {load_card(p.parent).variant: p.parent for p in out.glob("*/model_card.json")}
    assert set(cards) == {"onnx_fp32", "onnx_int8_dynamic", "onnx_int8_static"}
    for folder in cards.values():
        verify_files(load_card(folder), folder)
    fp32 = load_card(cards["onnx_fp32"])
    assert fp32.export is not None and fp32.export.parity_passed
    assert load_card(cards["onnx_int8_static"]).parent_sha256 == fp32.artifact_sha256
    last_build = json.loads((out / "last_build.json").read_text())
    assert last_build["artifacts"]["onnx_fp32"] == fp32.artifact_sha256

    # Rebuilding identical weights leaves the existing folders untouched.
    again = runner.invoke(app, ["build-random", "--config", str(recipe), "--out", str(out)])
    assert again.exit_code == 0, again.output
    assert len(list(out.glob("*/model_card.json"))) == 3

    # Another calibration method makes one more static artifact.
    result = runner.invoke(
        app,
        [
            "quantize",
            "--fp32",
            str(cards["onnx_fp32"]),
            "--method",
            "static_qdq",
            "--calibration",
            "Percentile",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(list(out.glob("*/model_card.json"))) == 4
