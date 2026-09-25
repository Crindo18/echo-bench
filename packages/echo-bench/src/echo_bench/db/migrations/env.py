"""Alembic environment: how schema migrations connect to a results database."""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import text

from echo_bench.db.models import Base
from echo_bench.db.session import make_engine

config = context.config
if config.config_file_name is not None:  # only when run through `uv run alembic ...`
    fileConfig(config.config_file_name)


def _db_path() -> str:
    # `uv run alembic -x db=results/scratch.db ...` overrides the path in alembic.ini.
    path = context.get_x_argument(as_dictionary=True).get("db") or config.get_main_option(
        "echo_db_path"
    )
    if not path:
        raise RuntimeError("No database path. Pass -x db=<file> or use `echo-bench db upgrade`.")
    return path


def run_migrations_online() -> None:
    # Foreign keys stay OFF while migrating; see make_engine() for why.
    engine = make_engine(_db_path(), foreign_keys=False)
    try:
        with engine.connect() as connection:
            context.configure(
                connection=connection,
                target_metadata=Base.metadata,
                render_as_batch=True,  # lets future migrations change SQLite tables safely
            )
            with context.begin_transaction():
                context.run_migrations()
            broken = connection.execute(text("PRAGMA foreign_key_check")).fetchall()
            if broken:
                raise RuntimeError(f"Migration left broken foreign keys: {broken[:5]}")
    finally:
        engine.dispose()


if context.is_offline_mode():
    raise SystemExit("ECHO-Bench does not use offline (SQL script) migrations.")
run_migrations_online()
