"""Create and manage doctor accounts, from the command line only.

    python -m src.auth.accounts add --username brikmariem --name "Dr Mariem Brik"
    python -m src.auth.accounts list
    python -m src.auth.accounts passwd --username brikmariem

There is no sign-up endpoint, and that is the point. An account in a clinical system is issued,
not claimed: a public "create account" form would let anyone who reaches the URL start storing
patient data under a name of their choosing. Keeping account creation here means no
unauthenticated request can ever create one, because no code path exists that would.

The password is read with getpass -- never taken as an argument -- so it does not land in the
shell history, in `ps` output, or in a terminal transcript. `--password-stdin` is provided for
scripting, which is the one case where a prompt cannot be used.
"""
from __future__ import annotations

import argparse
import getpass
import os
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from sqlalchemy import select                                          # noqa: E402

from src.auth import db                                                # noqa: E402
from src.auth.models import Doctor                                     # noqa: E402
from src.auth.passwords import hash_password                           # noqa: E402

MIN_LENGTH = 8


def _read_password(from_stdin: bool) -> str:
    if from_stdin:
        return sys.stdin.readline().rstrip("\n")
    first = getpass.getpass("password: ")
    if getpass.getpass("repeat:   ") != first:
        sys.exit("the two passwords differ; nothing was written")
    return first


def _check(password: str) -> None:
    # A floor, not a policy. Complexity rules push people towards Passw0rd! and this is a
    # prototype whose threat model is a demonstration, not a hospital network -- so the honest
    # thing is a minimum length and a warning, rather than a rule that pretends to more.
    if len(password) < MIN_LENGTH:
        sys.exit(f"password must be at least {MIN_LENGTH} characters")


def add(username: str, name: str, email: str | None, password: str) -> Doctor:
    _check(password)
    with db.session() as s:
        username = username.strip().lower()
        if s.scalar(select(Doctor).where(Doctor.username == username)):
            sys.exit(f"a doctor with username {username!r} already exists")
        d = Doctor(username=username, name=name, email=email,
                   password_hash=hash_password(password))
        s.add(d)
        s.commit()
        return d


def ensure_demo() -> tuple[str, str] | None:
    """Seed one demo account on first run, if POCUS_DEMO_PASSWORD says to.

    Returns (username, password) when an account was created, so the caller can print it once.
    Absent the variable nothing is seeded: an image that ships with a known password reachable
    from the network is a back door, however clearly it is labelled a demo. The password is
    never written to the repository -- only to the database, hashed.
    """
    password = os.environ.get("POCUS_DEMO_PASSWORD")
    if not password:
        return None
    username = os.environ.get("POCUS_DEMO_USERNAME", "demo")
    with db.session() as s:
        if s.scalar(select(Doctor).where(Doctor.username == username)):
            return None
    add(username, os.environ.get("POCUS_DEMO_NAME", "Demo clinician"), None, password)
    return username, password


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m src.auth.accounts",
                                description="Manage doctor accounts.")
    sub = p.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="create a doctor account")
    a.add_argument("--username", required=True)
    a.add_argument("--name", required=True)
    a.add_argument("--email")
    a.add_argument("--password-stdin", action="store_true",
                   help="read the password from stdin instead of prompting")

    sub.add_parser("list", help="list accounts (never shows a password or its hash)")

    w = sub.add_parser("passwd", help="change a password")
    w.add_argument("--username", required=True)
    w.add_argument("--password-stdin", action="store_true")

    g = sub.add_parser("generate", help="create an account with a random password, printed once")
    g.add_argument("--username", required=True)
    g.add_argument("--name", required=True)

    args = p.parse_args(argv)

    if args.cmd == "add":
        d = add(args.username, args.name, args.email, _read_password(args.password_stdin))
        print(f"created {d.username} ({d.name})  id={d.id}")

    elif args.cmd == "generate":
        password = secrets.token_urlsafe(12)
        d = add(args.username, args.name, None, password)
        print(f"created {d.username} ({d.name})  id={d.id}")
        print(f"password: {password}      <- shown once, not stored anywhere in plaintext")

    elif args.cmd == "list":
        with db.session() as s:
            rows = s.scalars(select(Doctor).order_by(Doctor.created_at)).all()
        if not rows:
            print("no accounts yet")
        for d in rows:
            print(f"  {d.username:20s} {d.name:28s} {d.created_at:%Y-%m-%d}")

    elif args.cmd == "passwd":
        password = _read_password(args.password_stdin)
        _check(password)
        with db.session() as s:
            d = s.scalar(select(Doctor).where(Doctor.username == args.username.strip().lower()))
            if d is None:
                sys.exit(f"no doctor with username {args.username!r}")
            d.password_hash = hash_password(password)
            s.commit()
        print(f"password changed for {args.username}")

    print(f"database: {db.db_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
