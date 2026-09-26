"""M3 end to end through the CLI: ingest -> CV plan -> validate -> coverage -> summary."""

import json
import sqlite3
from pathlib import Path

import yaml
from typer.testing import CliRunner

from echo_bench.cli import app
from echo_bench.data.coverage import coverage
from echo_bench.db.session import make_engine, make_session

runner = CliRunner()


def _ingest_both(
    tmp_path: Path, torgo_root: Path, uaspeech_root: Path, tables: dict[str, Path]
) -> str:
    db = str(tmp_path / "results" / "bench_test.db")
    assert runner.invoke(app, ["db", "init", "--db", db]).exit_code == 0
    for corpus, root in (("torgo", torgo_root), ("uaspeech", uaspeech_root)):
        args = [
            "data",
            "ingest",
            corpus,
            "--root",
            str(root),
            "--speakers",
            str(tables[corpus]),
            "--db",
            db,
        ]
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    return db


def _plan(tmp_path: Path, db: str, **overrides: object) -> None:
    config = tmp_path / "plan.yaml"
    body = {"name": "sgkf3-test", "k": 3, "datasets": ["torgo", "uaspeech"]} | overrides
    config.write_text(yaml.safe_dump(body))
    result = runner.invoke(app, ["cv", "plan", "--config", str(config), "--db", db])
    assert result.exit_code == 0, result.output


def test_ingest_plan_validate_summarize(
    tmp_path: Path,
    monkeypatch,
    torgo_root: Path,
    uaspeech_root: Path,
    speaker_tables: dict[str, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    db = _ingest_both(tmp_path, torgo_root, uaspeech_root, speaker_tables)

    again = runner.invoke(
        app,
        [
            "data",
            "ingest",
            "torgo",
            "--root",
            str(torgo_root),
            "--speakers",
            str(speaker_tables["torgo"]),
            "--db",
            db,
        ],
    )
    assert again.exit_code == 0 and "already loaded" in again.output  # idempotent

    listing = runner.invoke(app, ["data", "list", "--db", db])
    assert listing.exit_code == 0 and "2 dataset version(s)" in listing.output

    _plan(tmp_path, db)
    validation = runner.invoke(app, ["data", "validate", "--db", db])
    assert validation.exit_code == 0, validation.output
    assert "No leakage found (1 CV plan(s) checked" in validation.output

    con = sqlite3.connect(db)
    speakers = con.execute("SELECT COUNT(*) FROM speaker").fetchone()[0]
    rows = con.execute("SELECT fold_index, speaker_id, role FROM fold_assignment").fetchall()
    tests = [s for _, s, role in rows if role == "test"]
    assert len(tests) == speakers == len(set(tests))  # every speaker is tested exactly once
    for fold in (0, 1, 2):
        roles = {role for f, _, role in rows if f == fold}
        assert "test" in roles and "train" in roles
    assert json.loads(con.execute("SELECT dataset_ids_json FROM cv_plan").fetchone()[0])
    con.close()

    shown = runner.invoke(app, ["data", "coverage", "--db", db])
    assert shown.exit_code == 0 and "Feasible K" in shown.output
    engine = make_engine(db)
    with make_session(engine) as session:
        f01 = next(row for row in coverage(session, "torgo") if row.speaker == "F01")
    engine.dispose()
    assert f01.feasible == {3: 3, 5: 0, 10: 0}  # 3 prompts x 4 sessions: K=3 feasible, K=5/10 not

    summary = runner.invoke(app, ["cv", "summary", "--db", db, "--out-dir", "reports"])
    assert summary.exit_code == 0, summary.output
    assert Path("reports/cv_sgkf3-test_summary.md").read_text().startswith("| dataset:group |")


def test_validate_catches_planted_leaks(
    tmp_path: Path,
    monkeypatch,
    torgo_root: Path,
    uaspeech_root: Path,
    speaker_tables: dict[str, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    db = _ingest_both(tmp_path, torgo_root, uaspeech_root, speaker_tables)
    _plan(tmp_path, db)
    con = sqlite3.connect(db)
    # L2: the same physical recording attributed to a second speaker
    (utt_id, group) = con.execute("SELECT id, recording_group_id FROM utterance LIMIT 1").fetchone()
    other = con.execute(
        "SELECT id FROM speaker WHERE id != (SELECT speaker_id FROM utterance WHERE id = ?) LIMIT 1",
        (utt_id,),
    ).fetchone()[0]
    con.execute(
        "INSERT INTO utterance SELECT 'leak', dataset_id, ?, label_id, recording_group_id, session, channel, 0, "
        "'leak.wav', audio_sha256, sample_rate, duration_ms, recorded_at FROM utterance WHERE id = ?",
        (other, utt_id),
    )
    # L1: a speaker dropped from one fold
    con.execute(
        "DELETE FROM fold_assignment WHERE rowid = (SELECT MIN(rowid) FROM fold_assignment)"
    )
    con.commit()
    con.close()
    result = runner.invoke(app, ["data", "validate", "--db", db])
    assert result.exit_code == 1
    assert "L2" in result.output and group.split(":")[0] in result.output
    assert "speakers assigned" in result.output


def test_plans_are_immutable(
    tmp_path: Path,
    monkeypatch,
    torgo_root: Path,
    uaspeech_root: Path,
    speaker_tables: dict[str, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    db = _ingest_both(tmp_path, torgo_root, uaspeech_root, speaker_tables)
    _plan(tmp_path, db)
    config = tmp_path / "plan.yaml"
    result = runner.invoke(app, ["cv", "plan", "--config", str(config), "--db", db])
    assert result.exit_code == 1 and "immutable" in result.output


def test_two_folds_with_a_validation_fold_is_refused(
    tmp_path: Path,
    monkeypatch,
    torgo_root: Path,
    uaspeech_root: Path,
    speaker_tables: dict[str, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    db = _ingest_both(tmp_path, torgo_root, uaspeech_root, speaker_tables)
    config = tmp_path / "bad.yaml"
    config.write_text(yaml.safe_dump({"name": "k2", "k": 2, "datasets": ["torgo"]}))
    result = runner.invoke(app, ["cv", "plan", "--config", str(config), "--db", db])
    assert result.exit_code != 0


def test_validate_warns_about_recordings_too_short_to_be_speech(
    tmp_path: Path,
    monkeypatch,
    torgo_root: Path,
    uaspeech_root: Path,
    speaker_tables: dict[str, Path],
) -> None:
    monkeypatch.chdir(tmp_path)
    db = _ingest_both(tmp_path, torgo_root, uaspeech_root, speaker_tables)
    con = sqlite3.connect(db)
    con.execute(
        "UPDATE utterance SET duration_ms = 50 WHERE rowid = (SELECT MIN(rowid) FROM utterance)"
    )
    con.commit()
    con.close()
    result = runner.invoke(app, ["data", "validate", "--db", db])
    assert result.exit_code == 0  # a warning, not an error
    assert "shorter than 0.1 s" in result.output
