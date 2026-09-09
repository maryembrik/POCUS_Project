"""Look inside the application database.

    python tools/show_db.py              tables, row counts, and the columns of each
    python tools/show_db.py doctors      the rows of one table

Read-only: it opens the file, prints, and closes. Nothing here writes.

Password hashes are truncated in the output rather than printed whole. They are not secrets in
the way a password is -- that is the entire point of storing a hash -- but a full scrypt digest
pasted into a chat or a screenshot is still material nobody needs to see, and a tool that makes
it easy to paste is a tool that gets it pasted.
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.auth import db  # noqa: E402

MASK = {"password_hash", "token"}


def main(argv: list[str]) -> int:
    path = db.db_path()
    if not path.exists():
        print(f"no database at {path}")
        print("it is created on the first sign-in, or by "
              "`python -m src.auth.accounts add ...`")
        return 1

    # read-only URI, so this cannot corrupt the file it is inspecting even by accident
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    print(f"database: {path}  ({path.stat().st_size / 1024:.0f} KB)\n")

    tables = [r[0] for r in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' "
        "ORDER BY name")]

    if len(argv) > 1:
        want = argv[1]
        if want not in tables:
            print(f"no table {want!r}; there is {', '.join(tables)}")
            return 1
        rows = con.execute(f"SELECT * FROM {want} LIMIT 50").fetchall()  # noqa: S608
        if not rows:
            print(f"{want}: empty")
            return 0
        for r in rows:
            print(f"--- {want}")
            for k in r.keys():
                v = r[k]
                if k in MASK and v:
                    v = f"{str(v)[:24]}...  ({len(str(v))} chars, not shown in full)"
                elif isinstance(v, str) and len(v) > 90:
                    v = v[:90] + f"...  ({len(v)} chars)"
                print(f"    {k:18s} {v}")
        return 0

    for t in tables:
        n = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]      # noqa: S608
        cols = [f"{c[1]}" for c in con.execute(f"PRAGMA table_info({t})")]
        print(f"  {t:14s} {n:4d} row(s)")
        print(f"                 {', '.join(cols)}\n")
    print("one table's rows:  python tools/show_db.py <table>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
