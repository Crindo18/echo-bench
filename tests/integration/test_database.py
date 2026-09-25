"""The results database: schema, safety rules and migrations (blueprint sections 4.1 and 4.3)."""

import json
from pathlib import Path

import pytest
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import Engine, func, inspect, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from echo_bench.db.migrate import current_revision, downgrade_to, head_revision, upgrade_to_head
from echo_bench.db.models import (
    Base,
    BenchmarkRun,
    Campaign,
    LatencySample,
    ModelArtifact,
    SoftwareEnv,
)
from echo_bench.db.repository import get_or_create_hardware_profile, get_or_create_software_env
from echo_bench.env.fingerprint import collect_hardware, collect_software

M1_TABLES = {
    "schema_meta",
    "hardware_profile",
    "software_env",
    "model_artifact",
    "campaign",
    "benchmark_run",
    "latency_sample",
    "resource_sample",
    "run_metric",
    "gate_result",
}


def _make_run(session: Session) -> BenchmarkRun:
    hw = get_or_create_hardware_profile(session, collect_hardware())
    sw = get_or_create_software_env(session, collect_software())
    campaign = Campaign(
        name="test",
        kind="perf",
        config_json="{}",
        config_sha256="0" * 64,
        hardware_profile_id=hw.id,
        software_env_id=sw.id,
        status="running",
    )
    session.add(campaign)
    session.flush()
    run = BenchmarkRun(campaign_id=campaign.id, params_json="{}", seed=42, status="running")
    session.add(run)
    session.flush()
    return run


def test_migrations_create_every_m1_table(engine: Engine, db_path: Path) -> None:
    assert M1_TABLES <= set(inspect(engine).get_table_names())
    assert current_revision(db_path) == head_revision()


def test_connection_settings(engine: Engine) -> None:
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).scalar() == 1
        assert connection.execute(text("PRAGMA journal_mode")).scalar() == "wal"


def test_migrations_match_the_models(engine: Engine) -> None:
    """If this fails, models.py changed without a new migration (run alembic --autogenerate)."""
    with engine.connect() as connection:
        differences = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    assert differences == []


def test_invalid_json_is_rejected(session: Session) -> None:
    fields = collect_software()
    fields["packages_json"] = "this is not json"
    session.add(SoftwareEnv(**fields))
    with pytest.raises(IntegrityError):
        session.flush()


def test_booleans_must_be_0_or_1(session: Session) -> None:
    artifact = ModelArtifact(
        name="ebf-12m-rand",
        variant="onnx_fp32",
        weights_state="random_init",
        file_path="artifacts/models/x/model.onnx",
        sha256="a" * 64,
        size_bytes=1,
        embedding_dim=256,
        model_card_json=json.dumps({"schema_version": 1}),
        promoted=2,
    )
    session.add(artifact)
    with pytest.raises(IntegrityError):
        session.flush()


def test_unknown_stage_is_rejected(session: Session) -> None:
    run = _make_run(session)
    session.add(
        LatencySample(run_id=run.id, iteration=0, stage="warp_drive", latency_us=5, t_offset_ms=0)
    )
    with pytest.raises(IntegrityError):
        session.flush()


def test_deleting_a_campaign_deletes_its_runs_and_samples(session: Session) -> None:
    run = _make_run(session)
    for i in range(3):
        session.add(
            LatencySample(
                run_id=run.id, iteration=i, stage="encoder", latency_us=1000, t_offset_ms=i
            )
        )
    session.commit()
    session.delete(session.get(Campaign, run.campaign_id))
    session.commit()
    assert session.scalar(select(func.count()).select_from(LatencySample)) == 0
    assert session.scalar(select(func.count()).select_from(BenchmarkRun)) == 0


def test_fingerprints_are_stored_once(session: Session) -> None:
    first = get_or_create_hardware_profile(session, collect_hardware())
    again = get_or_create_hardware_profile(session, collect_hardware())
    assert first.id == again.id


def test_downgrade_and_upgrade_again(db_path: Path) -> None:
    downgrade_to(db_path, "base")
    assert current_revision(db_path) is None
    upgrade_to_head(db_path)
    assert current_revision(db_path) == head_revision()
