"""The engine, and where the database file lives.

The path comes from POCUS_DATA rather than being computed from the source tree, because the
container runs as a non-root user for whom /app is deliberately not writable. A database opened
next to serve.py works in development and fails on the first write in the image -- the kind of
difference that only appears after deployment. POCUS_DATA names a directory that is a volume in
Docker and defaults to ./data locally.

Two SQLite pragmas are set on every connection, and both matter:

  foreign_keys=ON   SQLite does NOT enforce foreign keys by default. Without this the
                    ON DELETE CASCADE declared in models.py is decorative, and deleting a
                    doctor would leave their patients behind as unreachable rows that still
                    hold clinical data.
  journal_mode=WAL  readers do not block the writer. The interface polls while an assessment
                    is being written, and the default rollback journal makes that a lock
                    contention rather than a read.
"""
from __future__ import annotations

import os
from pathlib import Path

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session as OrmSession, sessionmaker

from .models import Base

ROOT = Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    d = Path(os.environ.get("POCUS_DATA") or (ROOT / "data"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def db_path() -> Path:
    return data_dir() / "pocus.db"


_engine = None
_Session: sessionmaker[OrmSession] | None = None


def engine():
    global _engine, _Session
    if _engine is None:
        # check_same_thread=False because uvicorn serves requests from a thread pool and the
        # connection is handed between them; the pool below, not SQLite, does the serialising.
        _engine = create_engine(f"sqlite:///{db_path()}",
                                connect_args={"check_same_thread": False},
                                future=True)

        @event.listens_for(_engine, "connect")
        def _pragmas(dbapi_connection, _record):
            cur = dbapi_connection.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()

        Base.metadata.create_all(_engine)
        _Session = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
    return _engine


def session() -> OrmSession:
    engine()
    assert _Session is not None
    return _Session()


def reset_for_tests(path: Path | None = None) -> None:
    """Point the engine at a fresh database. Only the tests call this.

    Without it every test would share one file and the ownership tests -- which are about what
    one doctor can see of another's data -- would depend on the order they ran in.
    """
    global _engine, _Session
    if path is not None:
        os.environ["POCUS_DATA"] = str(path)
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _Session = None
    engine()
