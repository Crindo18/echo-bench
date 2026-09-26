"""echo-train embed on a tiny ESPnet model built from a generated config (no downloads)."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
import yaml
from typer.testing import CliRunner

from echo_train.cli import app
from echo_train.embed import EmbedError, find_checkpoint, load_model


def _tiny_espnet_model(folder: Path) -> Path:
    from espnet2.tasks.asr import ASRTask

    config = ASRTask.get_default_config()
    config.update(
        token_list=["<blank>", "<unk>", "a", "b", "<sos/eos>"],
        frontend="default",
        frontend_conf={"n_fft": 512, "win_length": 400, "hop_length": 160, "fs": "16k"},
        normalize="utterance_mvn",
        normalize_conf={},
        encoder="e_branchformer",
        encoder_conf={
            "output_size": 32,
            "attention_heads": 2,
            "num_blocks": 1,
            "cgmlp_linear_units": 64,
            "linear_units": 32,
        },
        decoder="transformer",
        decoder_conf={"attention_heads": 2, "linear_units": 32, "num_blocks": 1},
        model_conf={"ctc_weight": 0.3},
    )
    run = folder / "exp" / "asr_train_tiny"
    run.mkdir(parents=True)
    (run / "config.yaml").write_text(yaml.safe_dump(config))
    model, _ = ASRTask.build_model_from_file(run / "config.yaml", None, "cpu")
    torch.save(model.state_dict(), run / "valid.acc.ave.pth")
    return folder


def _corpus(folder: Path) -> Path:
    records = []
    rng = np.random.default_rng(0)
    for i, (speaker, word) in enumerate(
        [("F01", "yes"), ("F01", "no"), ("M03", "yes"), ("M03", "no")]
    ):
        path = folder / "audio" / f"{i}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(path), (0.05 * rng.standard_normal(8000 + 4000 * i)).astype(np.float32), 16000)
        records.append(
            {
                "speaker": speaker,
                "label_key": word,
                "label_text": word,
                "category": "proxy:word",
                "recording_group_id": f"g{i}",
                "session": "Session1",
                "channel": "headMic",
                "is_primary_channel": True,
                "audio_path": f"audio/{i}.wav",
                "audio_sha256": "0" * 64,
                "sample_rate": 16000,
                "duration_ms": 500,
            }
        )
    tiny = folder / "audio" / "tiny.wav"  # 40 ms: too short for the conv2d input stage
    sf.write(str(tiny), np.zeros(640, dtype=np.float32), 16000)
    records.append(records[0] | {"audio_path": "audio/tiny.wav", "recording_group_id": "g-tiny"})
    manifest = folder / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records))
    return manifest


def test_embed_writes_one_vector_per_utterance(tmp_path: Path) -> None:
    model_dir = _tiny_espnet_model(tmp_path / "model")
    manifest = _corpus(tmp_path / "corpus")
    out = tmp_path / "emb.npz"
    result = CliRunner().invoke(
        app,
        [
            "embed",
            "--manifest",
            str(manifest),
            "--root",
            str(tmp_path / "corpus"),
            "--out",
            str(out),
            "--model",
            str(model_dir),
        ],
    )
    assert result.exit_code == 0, result.output
    data = np.load(out)
    assert data["embeddings"].shape == (4, 32) and np.isfinite(data["embeddings"]).all()
    assert list(data["speaker"]) == ["F01", "F01", "M03", "M03"]
    meta = json.loads(str(data["metadata"]))
    assert meta["weights_state"] == "trained" and meta["encoder"] == "e_branchformer"
    assert (
        meta["skipped_too_short"] == ["audio/tiny.wav"] and "Skipped 1 recording" in result.output
    )


def test_trained_and_random_weights_differ(tmp_path: Path) -> None:
    checkpoint = find_checkpoint(_tiny_espnet_model(tmp_path / "model"))
    trained, info, fs = load_model(checkpoint)
    untrained, _, _ = load_model(checkpoint, random_weights=True)
    assert info["encoder_tensors_loaded"] > 0 and fs == 16000
    saved = torch.load(checkpoint.weights, weights_only=True)
    key = next(k for k in saved if k.startswith("encoder."))
    assert torch.equal(trained.state_dict()[key], saved[key])


def test_older_espnet_names_are_mapped(tmp_path: Path) -> None:
    """2022 ESPnet saved the input stage's output layer as encoder.embed.out.0.*"""
    checkpoint = find_checkpoint(_tiny_espnet_model(tmp_path / "model"))
    state = torch.load(checkpoint.weights, weights_only=True)
    original = state["encoder.embed.out.weight"].clone()
    for suffix in ("weight", "bias"):
        state[f"encoder.embed.out.0.{suffix}"] = state.pop(f"encoder.embed.out.{suffix}")
    torch.save(state, checkpoint.weights)
    model, info, _ = load_model(checkpoint)
    assert info["renamed_tensors"] == {
        "encoder.embed.out.0.weight": "encoder.embed.out.weight",
        "encoder.embed.out.0.bias": "encoder.embed.out.bias",
    }
    assert torch.equal(model.state_dict()["encoder.embed.out.weight"], original)


def test_a_rename_with_another_shape_is_refused(tmp_path: Path) -> None:
    checkpoint = find_checkpoint(_tiny_espnet_model(tmp_path / "model"))
    state = torch.load(checkpoint.weights, weights_only=True)
    state["encoder.embed.out.0.weight"] = torch.zeros(3, 3)
    del state["encoder.embed.out.weight"]
    torch.save(state, checkpoint.weights)
    with pytest.raises(EmbedError, match="Missing: encoder.embed.out.weight"):
        load_model(checkpoint)


def test_a_checkpoint_that_does_not_fit_is_refused(tmp_path: Path) -> None:
    checkpoint = find_checkpoint(_tiny_espnet_model(tmp_path / "model"))
    state = torch.load(checkpoint.weights, weights_only=True)
    del state[next(k for k in state if k.startswith("encoder."))]
    torch.save(state, checkpoint.weights)
    with pytest.raises(EmbedError, match="partly random"):
        load_model(checkpoint)
