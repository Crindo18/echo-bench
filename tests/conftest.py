"""Shared test fixtures."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from echo_bench.db.migrate import upgrade_to_head
from echo_bench.db.session import make_engine, make_session


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """A fresh results database, created through the real migrations."""
    path = tmp_path / "bench_test.db"
    upgrade_to_head(path)
    return path


@pytest.fixture
def engine(db_path: Path) -> Iterator[Engine]:
    engine = make_engine(db_path)
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine: Engine) -> Iterator[Session]:
    with make_session(engine) as session:
        yield session
