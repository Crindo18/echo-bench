"""Opening the results database (blueprint section 4.1 and ADR-2).

Each machine writes its own SQLite file, results/bench_<hostname>.db, and the
desktop merges them later (M2). Every connection turns on foreign keys, WAL
journaling and synchronous=NORMAL.
"""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker


def default_db_path() -> Path:
    return Path("results") / f"bench_{socket.gethostname()}.db"


def make_engine(db_path: str | Path, *, foreign_keys: bool = True) -> Engine:
    """Create an engine for one SQLite file, creating its folder if needed.

    foreign_keys=False is only for migrations. With foreign keys on, SQLite's
    DROP TABLE (which Alembic uses to change a table) silently deletes the rows
    of child tables through ON DELETE CASCADE.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(f"sqlite:///{path}")

    @event.listens_for(engine, "connect")
    def _set_pragmas(dbapi_connection: Any, _connection_record: Any) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute(f"PRAGMA foreign_keys={'ON' if foreign_keys else 'OFF'}")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()

    return engine


def make_session(engine: Engine) -> Session:
    return sessionmaker(engine, expire_on_commit=False)()
