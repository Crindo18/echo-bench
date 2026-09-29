"""The quick few-shot check on synthetic embeddings with known structure."""

import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from echo_bench.cli import app
from echo_bench.fewshot.embedding_cache import EmbeddingSet, load_embeddings
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


def test_several_files_are_compared_on_the_recordings_they_share(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    full = _write(tmp_path / "full.npz", structured=True, source="hf:full")
    data = dict(np.load(full))
    keep = np.arange(len(data["embeddings"])) != 0  # this encoder skipped one recording
    np.savez(tmp_path / "short.npz", **{k: (v[keep] if v.ndim else v) for k, v in data.items()})
    result = CliRunner().invoke(
        app,
        [
            "fewshot",
            "quick",
            "--embeddings",
            str(full),
            "--embeddings",
            str(tmp_path / "short.npz"),
            "--k",
            "1",
            "--episodes",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Comparing on the 287 recordings" in result.output  # 6 speakers x 12 classes x 4 - 1


# ---------------------------------------------------------------- word pool (SI vs K)


def _uaspeech_like() -> EmbeddingSet:
    """UASpeech's layout in miniature: 8 words said 3 times, 12 words said once, 4 speakers."""
    rng = np.random.default_rng(7)
    labels, speakers = [], []
    for s in range(4):
        for w in range(8):
            labels += [f"CW{w}"] * 3
            speakers += [f"S{s}"] * 3
        for w in range(12):
            labels.append(f"B1_UW{w}")
            speakers.append(f"S{s}")
    n = len(labels)
    return EmbeddingSet(
        name="toy",
        path=Path("toy.npz"),
        embeddings=rng.standard_normal((n, 16)).astype(np.float32),
        speaker=np.array(speakers),
        label=np.array(labels),
        group=np.array([f"g{i}" for i in range(n)]),
        session=np.array(["B1"] * n),
        metadata={},
    )


def test_matched_pool_uses_the_same_words_for_si_and_k() -> None:
    es = _uaspeech_like()
    groups = {f"S{s}": "low" for s in range(4)}
    matched = evaluate(es, groups, ks=[1, 2], episodes=3, pool="matched")
    assert all(r.pool_size == {"SI": 8, "K=1": 8, "K=2": 8} for r in matched)


def test_pool_all_keeps_the_old_behaviour_and_its_mismatch() -> None:
    es = _uaspeech_like()
    groups = {f"S{s}": "low" for s in range(4)}
    old = evaluate(es, groups, ks=[1, 2], episodes=3, pool="all")
    # SI also draws the 12 words said once; K can't use them.
    assert all(r.pool_size == {"SI": 20, "K=1": 8, "K=2": 8} for r in old)


def test_unknown_pool_is_refused() -> None:
    with pytest.raises(ValueError):
        evaluate(_uaspeech_like(), {}, ks=[1], pool="some")  # type: ignore[arg-type]


def test_cli_writes_paired_table_and_per_speaker_csv(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    good = _write(tmp_path / "good.npz", structured=True, source="hf:good")
    noise = _write(tmp_path / "noise.npz", structured=False, source="hf:noise")
    out = tmp_path / "reports" / "per_speaker.csv"
    result = CliRunner().invoke(
        app,
        [
            "fewshot",
            "quick",
            "--embeddings",
            str(noise),
            "--embeddings",
            str(good),
            "--k",
            "1",
            "--episodes",
            "3",
            "--csv",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Paired by speaker" in result.output and "Holm" in result.output
    lines = out.read_text().splitlines()
    assert lines[0] == "encoder,speaker,group,pool,SI,K=1"
    assert len(lines) == 1 + 2 * len(SPEAKERS)
