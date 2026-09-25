"""The model card schema enforces the section 4.4 contract."""

from collections.abc import Callable
from pathlib import Path

import pytest
from pydantic import ValidationError

from echo_core.model_card import ModelCard, sha256_file, verify_files

HASHES = {"files_sha256": {"model.onnx": "0" * 64}}
Fields = Callable[..., dict[str, object]]


def test_valid_card_round_trips(card_fields: Fields) -> None:
    card = ModelCard.model_validate(card_fields("onnx_fp32") | HASHES)
    assert ModelCard.model_validate_json(card.to_json()) == card
    assert card.ref == "tiny-rand:onnx_fp32"
    assert card.export is not None and card.export.parity_passed


def test_int8_needs_a_quantization_block(card_fields: Fields) -> None:
    with pytest.raises(ValidationError, match="quantization"):
        ModelCard.model_validate(card_fields("onnx_int8_dynamic") | HASHES)


def test_static_int8_needs_calibration(card_fields: Fields) -> None:
    fields = card_fields(
        "onnx_int8_static", quantization={"method": "static_qdq", "onnxruntime": "x"}
    )
    with pytest.raises(ValidationError, match="calibration"):
        ModelCard.model_validate(fields | HASHES)


def test_trained_models_need_their_frontend(card_fields: Fields) -> None:
    with pytest.raises(ValidationError, match="frontend"):
        ModelCard.model_validate(card_fields("onnx_fp32", weights_state="trained") | HASHES)


def test_unknown_fields_are_rejected(card_fields: Fields) -> None:
    with pytest.raises(ValidationError):
        ModelCard.model_validate(card_fields("onnx_fp32", colour="blue") | HASHES)


def test_verify_files_detects_a_changed_file(tmp_path: Path, card_fields: Fields) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"weights")
    card = ModelCard.model_validate(
        card_fields("onnx_fp32") | {"files_sha256": {"model.onnx": sha256_file(model)}}
    )
    verify_files(card, tmp_path)
    model.write_bytes(b"weights, edited")
    with pytest.raises(ValueError, match="does not match"):
        verify_files(card, tmp_path)
