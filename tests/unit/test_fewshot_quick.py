"""The quick few-shot check on synthetic embeddings with known structure."""

import json
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from echo_bench.cli import app
from echo_bench.fewshot.embedding_cache import load_embeddings
from echo_bench.fewshot.quick import evaluate, summarize

SPEAKERS = {
    "F01": "severe",
    "M01": "severe",
    "F04": "mild",
    "M03": "mild",
    "FC01": "control",
    "MC01": "control",
}


def _write(path: Path, *, structured: bool, source: str) -> Path:
    """12 classes x 4 recordings per speaker. Structured: class centre + a distortion that is
    specific to each speaker AND word (like dysarthric speech) + small recording noise."""
    rng = np.random.default_rng(1)
    centres = rng.standard_normal((12, 32))
    rows, meta = [], []
    for speaker in SPEAKERS:
        for c in range(12):
            distortion = rng.standard_normal(32) * 1.5
            for rep in range(4):
                vector = (
                    centres[c] + distortion + rng.standard_normal(32) * 0.15
                    if structured
                    else rng.standard_normal(32)
                )
                rows.append(vector)
                meta.append((speaker, f"w{c}", f"{speaker}:{c}:{rep}", f"Session{rep + 1}"))
    np.savez(
        path,
        embeddings=np.array(rows, dtype=np.float32),
        speaker=np.array([m[0] for m in meta]),
        label_key=np.array([m[1] for m in meta]),
        recording_group_id=np.array([m[2] for m in meta]),
        session=np.array([m[3] for m in meta]),
        audio_path=np.array([f"{m[2]}.wav" for m in meta]),
        category=np.array(["proxy:word"] * len(meta)),
        metadata=np.array(
            json.dumps(
                {"model_source": source, "weights_state": "trained", "manifest": "torgo-v1.jsonl"}
            )
        ),
    )
    return path


def test_structured_embeddings_separate_and_personal_prototypes_help(tmp_path: Path) -> None:
    es = load_embeddings(_write(tmp_path / "good.npz", structured=True, source="hf:good"))
    rows = summarize(evaluate(es, SPEAKERS, ks=[1, 3], ways=10, episodes=5), ["SI", "K=1", "K=3"])
    overall = rows[-1][2]
    assert rows[-1][0] == "all dysarthric"
    assert (
        overall["K=3"] > 0.95 and overall["K=1"] > 0.9
    )  # speaker offsets cancel with own prototypes
    assert (
        overall["SI"] < 0.9 < overall["K=3"]
    )  # other speakers' prototypes miss personal distortions


def test_random_embeddings_are_near_chance(tmp_path: Path) -> None:
    es = load_embeddings(_write(tmp_path / "noise.npz", structured=False, source="hf:noise"))
    rows = summarize(evaluate(es, SPEAKERS, ks=[3], ways=10, episodes=10), ["SI", "K=3"])
    assert rows[-1][2]["K=3"] < 0.3  # 10-way chance is 10%


def test_cli_compares_encoders(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    good = _write(tmp_path / "good.npz", structured=True, source="hf:good")
    noise = _write(tmp_path / "noise.npz", structured=False, source="hf:noise")
    speakers = tmp_path / "speakers.csv"
    speakers.write_text(
        "code,cohort,severity_tier,intelligibility_pct,sex\n"
        + "".join(
            f"{c},{'control' if t == 'control' else 'dysarthric'},{'' if t == 'control' else t},,\n"
            for c, t in SPEAKERS.items()
        )
    )
    result = CliRunner().invoke(
        app,
        [
            "fewshot",
            "quick",
            "--embeddings",
            str(good),
            "--embeddings",
            str(noise),
            "--speakers",
            str(speakers),
            "--k",
            "1",
            "--k",
            "3",
            "--episodes",
            "3",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "side by side" in result.output and "Chance is about 10%" in result.output
