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

import os
import re
import secrets
from datetime import timedelta

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

from . import db
from .models import Doctor, Session, utcnow
from .passwords import hash_password, verify_password

SESSION_HOURS = 12
COOKIE = "pocus_session"
# A floor, not a policy. Complexity rules push people towards Passw0rd!; length is the property
# that actually costs an attacker anything.
MIN_PASSWORD = 8

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


def register(username: str, name: str, password: str,
             email: str | None = None) -> tuple[str | None, str | None]:
    """Create an account and sign it in. Returns (token, error); exactly one is None.

    Registration is open by default and can be closed with POCUS_OPEN_REGISTRATION=0. The
    switch exists because "anyone who reaches this URL may create an account and store patient
    data" is the correct behaviour for a demonstration and the wrong one for a deployment, and
    which of those a given instance is cannot be decided here.

    Unlike login, this endpoint cannot avoid disclosing whether a username is taken -- it has
    to refuse the second one. That is inherent to sign-up rather than a flaw in this
    implementation, and it is the reason login is careful about it: an attacker who can
    enumerate usernames here still learns nothing about which of them have which passwords.
    """
    if os.environ.get("POCUS_OPEN_REGISTRATION", "1") == "0":
        return None, "registration is closed on this instance"

    username = (username or "").strip().lower()
    name = (name or "").strip()
    if not username or not name:
        return None, "a username and a name are required"
    if len(username) < 3 or not re.fullmatch(r"[a-z0-9._-]+", username):
        return None, "the username may use letters, digits, dot, dash and underscore only"
    if len(password or "") < MIN_PASSWORD:
        return None, f"the password must be at least {MIN_PASSWORD} characters"

    with db.session() as s:
        if s.scalar(select(Doctor).where(Doctor.username == username)):
            return None, "that username is already taken"
        doctor = Doctor(username=username, name=name, email=(email or None),
                        password_hash=hash_password(password))
        s.add(doctor)
        try:
            s.commit()
        except IntegrityError:
            # Two sign-ups racing for the same username. The unique index is what actually
            # decides it; the check above only makes the common case a tidy message.
            s.rollback()
            return None, "that username is already taken"

    token = login(username, password)
    return token, None if token else "the account was created but could not be signed in"


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
