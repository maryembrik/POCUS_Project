"""Signing in, signing out, and deciding who is asking.

Authentication is deliberately boring here. The only judgements worth stating:

  * A failed login says "username or password is incorrect" whichever was wrong. Saying which
    turns the form into an oracle for valid usernames.
  * A login attempt against an unknown username still runs a scrypt verification, against a
    dummy hash. Returning early would make a missing account measurably faster to reject than
    a wrong password, which is the same oracle by a slower route.
  * Sessions are rows, so revoking one is a DELETE. See the note in models.Session.
"""
from __future__ import annotations

import secrets
from datetime import timedelta

from sqlalchemy import delete, select

from . import db
from .models import Doctor, Session, utcnow
from .passwords import hash_password, verify_password

SESSION_HOURS = 12
COOKIE = "pocus_session"

# Verified against when no such user exists, purely so that the failure takes the same time as
# a real one. Built once at import: hashing here costs the same ~100 ms as any other scrypt call
# and must not be paid on every failed login.
_DUMMY_HASH = hash_password(secrets.token_urlsafe(16))


def login(username: str, password: str) -> str | None:
    """Return a new session token, or None. Never says which half was wrong."""
    with db.session() as s:
        doctor = s.scalar(select(Doctor).where(Doctor.username == username.strip().lower()))
        stored = doctor.password_hash if doctor else _DUMMY_HASH
        ok = verify_password(password, stored)
        if not doctor or not ok:
            return None

        token = secrets.token_urlsafe(32)
        s.add(Session(token=token, doctor_id=doctor.id,
                      expires_at=utcnow() + timedelta(hours=SESSION_HOURS)))
        s.commit()
        return token


def logout(token: str | None) -> None:
    if not token:
        return
    with db.session() as s:
        s.execute(delete(Session).where(Session.token == token))
        s.commit()


def doctor_for(token: str | None) -> Doctor | None:
    """The doctor this token belongs to, or None if it is absent, unknown or expired."""
    if not token:
        return None
    with db.session() as s:
        row = s.get(Session, token)
        if row is None:
            return None
        # Compared in Python rather than in SQL because SQLite has no native timezone-aware
        # comparison, and a naive one here would be an hour wrong twice a year.
        expires = row.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=utcnow().tzinfo)
        if expires <= utcnow():
            s.execute(delete(Session).where(Session.token == token))
            s.commit()
            return None
        return s.get(Doctor, row.doctor_id)


def purge_expired() -> int:
    """Expired rows are dead weight and a session table that only grows is a slow leak."""
    with db.session() as s:
        n = s.execute(delete(Session).where(Session.expires_at <= utcnow())).rowcount
        s.commit()
        return n or 0
