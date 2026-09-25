"""Creating and upgrading the database schema with Alembic.

Alembic keeps a numbered history of schema changes in db/migrations/versions/.
upgrade_to_head() applies whatever steps a database is missing, so the same
call creates a new database or brings an old one up to date.
"""

from __future__ import annotations

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from echo_bench.db.session import make_engine

MIGRATIONS_DIR = Path(__file__).resolve().parent / "migrations"


def alembic_config(db_path: str | Path) -> Config:
    config = Config()
    config.set_main_option("script_location", str(MIGRATIONS_DIR).replace("%", "%%"))
    config.set_main_option("echo_db_path", str(db_path).replace("%", "%%"))
    return config


def upgrade_to_head(db_path: str | Path) -> None:
    command.upgrade(alembic_config(db_path), "head")


def downgrade_to(db_path: str | Path, revision: str) -> None:
    command.downgrade(alembic_config(db_path), revision)


def head_revision() -> str | None:
    return ScriptDirectory(str(MIGRATIONS_DIR)).get_current_head()


def current_revision(db_path: str | Path) -> str | None:
    """The schema revision of a database, or None if it doesn't exist or is empty."""
    if not Path(db_path).exists():
        return None
    engine = make_engine(db_path)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()
