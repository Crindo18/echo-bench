"""The M1 slice through the CLI: register -> perf -> report (blueprint section 7.2, M1)."""

from pathlib import Path

import yaml
from sqlalchemy import func, select
from typer.testing import CliRunner

from echo_bench.cli import app
from echo_bench.db.models import BenchmarkRun, Campaign, LatencySample, ResourceSample, RunMetric
from echo_bench.db.session import make_engine, make_session

runner = CliRunner()

TINY_PERF = {
    "kind": "perf",
    "models": ["tiny-rand:onnx_fp32", "tiny-rand:onnx_int8_dynamic"],
    "input_durations_s": [0.5, 1],
    "warmup_iters": 2,
    "measure_iters": 6,
    "block_size": 3,
    "cooldown_below_c": None,
    "pause_between_blocks_s": 0.05,
    "bootstrap_resamples": 200,
    "ort": {"intra_op_threads": [1]},
    "monitor": {"sample_hz": 50},
}


def _setup(root: Path) -> tuple[str, Path]:
    db = str(root / "results" / "bench_test.db")
    assert runner.invoke(app, ["db", "init", "--db", db]).exit_code == 0
    result = runner.invoke(app, ["model", "register", "artifacts/models", "--db", db])
    assert result.exit_code == 0, result.output
    config = root / "perf_tiny.yaml"
    config.write_text(yaml.safe_dump(TINY_PERF))
    return db, config


def test_perf_then_report(tiny_artifacts: dict[str, Path]) -> None:
    db, config = _setup(tiny_artifacts["root"])
    result = runner.invoke(app, ["perf", "--config", str(config), "--db", db])
    assert result.exit_code == 0, result.output

    engine = make_engine(db)
    with make_session(engine) as session:
        campaign = session.scalars(select(Campaign)).one()
        assert (campaign.kind, campaign.status) == ("perf", "completed")
        runs = session.scalars(select(BenchmarkRun)).all()
        assert len(runs) == 2 and all(run.status == "completed" for run in runs)
        for run in runs:
            count = (
                select(func.count())
                .select_from(LatencySample)
                .where(LatencySample.run_id == run.id)
            )
            assert (
                session.scalar(count) == 1 + (2 + 6) * 2
            )  # session load + 2 lengths x (warm-up + measured)
            cold = select(func.count()).where(
                LatencySample.run_id == run.id, LatencySample.is_cold == 1
            )
            assert session.scalar(cold) == 2  # the session load and the first inference
            names = {
                (m.name, m.scope)
                for m in session.scalars(select(RunMetric).where(RunMetric.run_id == run.id))
            }
            assert {("latency_p95_ms", "encoder@0.5s"), ("latency_p95_ms", "encoder@1s")} <= names
            assert ("session_load_ms", "overall") in names
        assert session.scalar(select(func.count()).select_from(ResourceSample)) > 0
    engine.dispose()

    result = runner.invoke(app, ["report", "perf", "--latest", "--db", db, "--out-dir", "reports"])
    assert result.exit_code == 0, result.output
    assert "Encoder latency" in result.output
    assert len(list(Path("reports").glob("perf_*_latency_vs_length.png"))) == 1


def test_perf_refuses_a_changed_model_file(tiny_artifacts: dict[str, Path]) -> None:
    db, config = _setup(tiny_artifacts["root"])
    with (tiny_artifacts["fp32"] / "model.onnx").open("ab") as handle:
        handle.write(b"\0")
    result = runner.invoke(app, ["perf", "--config", str(config), "--db", db])
    assert result.exit_code == 1
    assert "changed since it was registered" in result.output


def test_bad_config_is_rejected_before_anything_runs(tiny_artifacts: dict[str, Path]) -> None:
    db, config = _setup(tiny_artifacts["root"])
    config.write_text(yaml.safe_dump(TINY_PERF | {"measure_itters": 5}))
    result = runner.invoke(app, ["perf", "--config", str(config), "--db", db])
    assert result.exit_code == 1
