"""The M0 commands work end to end (blueprint section 7.2, M0 exit criteria)."""

from pathlib import Path

from typer.testing import CliRunner

from echo_bench.cli import app

runner = CliRunner()


def test_db_init_then_doctor_save(tmp_path: Path) -> None:
    db = str(tmp_path / "bench_cli.db")

    result = runner.invoke(app, ["db", "init", "--db", db])
    assert result.exit_code == 0, result.output
    assert "Database ready" in result.output

    result = runner.invoke(app, ["doctor", "--db", db, "--save"])
    assert result.exit_code == 0, result.output
    assert "Hardware" in result.output and "Saved to" in result.output

    # Running init again on an existing database is safe.
    assert runner.invoke(app, ["db", "init", "--db", db]).exit_code == 0


def test_db_upgrade_needs_an_existing_database(tmp_path: Path) -> None:
    result = runner.invoke(app, ["db", "upgrade", "--db", str(tmp_path / "missing.db")])
    assert result.exit_code == 1
