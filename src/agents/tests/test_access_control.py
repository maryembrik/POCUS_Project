"""Whose data is this? Every test here is an attempt to reach a patient that is not yours.

The clinical tests in this package ask whether the system reasons correctly. These ask a
different question, and one that only became askable when the application grew accounts: can a
signed-in doctor reach another doctor's patients, and can someone who is not signed in reach
any at all.

They are written as attacks rather than as confirmations. A test that logs in and reads its own
patient proves that the feature works; it proves nothing about whether it is safe. Each test
below performs the request an attacker would perform -- a guessed identifier, a stolen but
revoked cookie, a forged patient id in a write -- and asserts on what came back.

No pytest fixtures: run_benchmark.py calls every test here as a plain zero-argument callable.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import serve  # noqa: E402
from src.auth import accounts, passwords, sessions  # noqa: E402

from .helpers import prop, temp_database  # noqa: E402

ACCESS = "Access control"

_A: TestClient | None = None
_B: TestClient | None = None


def _doctors() -> tuple[TestClient, TestClient]:
    """Two signed-in doctors, built once. A is the victim; B is the attacker."""
    global _A, _B
    if _A is None or _B is None:
        temp_database()
        for user in ("alice", "mallory"):
            try:
                accounts.add(user, f"Dr {user.title()}", None, f"{user}-password-1")
            except SystemExit:
                pass                          # already created by an earlier test in this run
        _A, _B = TestClient(serve.app), TestClient(serve.app)
        for client, user in ((_A, "alice"), (_B, "mallory")):
            r = client.post("/api/login",
                            json={"username": user, "password": f"{user}-password-1"})
            assert r.status_code == 200, f"{user} could not sign in: {r.text}"
    return _A, _B


def _a_patient_of_alice(a: TestClient) -> tuple[str, str]:
    """One patient and one examination belonging to Alice. Returns (patient id, record id)."""
    p = a.post("/api/patients", json={"name": "Alice's patient"}).json()["patient"]
    a.post("/api/analyse", json={"name": "Alice's patient", "age": 71, "sex": "F",
                                 "complaint": "shortness of breath", "organ": "Lung",
                                 "patientId": p["id"],
                                 "findings": {}, "vitals": {}, "labs": {}})
    records = a.get("/api/bootstrap").json()["records"]
    assert records, "Alice's own encounter did not reach her record list"
    return p["id"], records[-1]["id"]


# ------------------------------------------------------------------ nobody is signed in ----
@prop(ACCESS)
def test_every_clinical_endpoint_refuses_an_unauthenticated_request():
    """The list is explicit on purpose.

    A dependency added to most handlers and forgotten on one is the ordinary way this fails,
    and the forgotten one is invisible in review because the file is full of neighbours that
    have it. Naming every route here means adding an endpoint without a doctor makes a test
    fail rather than making a hole.
    """
    anon = TestClient(serve.app)
    body = {"name": "x", "age": 50, "sex": "F", "complaint": "x", "organ": "Lung",
            "findings": {}, "vitals": {}, "labs": {}}
    attempts = [
        ("get", "/api/bootstrap", None), ("get", "/api/view", None),
        ("get", "/api/record?id=1", None), ("get", "/api/patients", None),
        ("get", "/api/preset?key=septic", None),
        ("get", "/api/patients/anything/examinations", None),
        ("post", "/api/analyse", body), ("post", "/api/triage/suggest", body),
        ("post", "/api/ask", {"question": "what is wrong?"}),
        ("post", "/api/patients", {"name": "x"}),
        ("post", "/api/upload", {"images": [], "organ": "Lung"}),
        ("post", "/api/record/attach", {"id": "1", "images": []}),
    ]
    leaked = []
    for method, url, payload in attempts:
        r = getattr(anon, method)(url, json=payload) if payload is not None \
            else getattr(anon, method)(url)
        if r.status_code != 401:
            leaked.append(f"{method.upper()} {url} -> {r.status_code}")
    assert not leaked, "reachable without signing in: " + "; ".join(leaked)


@prop(ACCESS)
def test_the_application_page_redirects_to_the_login_form():
    """Both the route and the file behind it.

    The page is a file under web/, which is also served by a StaticFiles mount -- so gating the
    route while leaving the file addressable by name would have left the interface reachable at
    /pocus-copilot.dc.html. That is the version of this bug that actually shipped in other
    projects, and it is why the second assertion exists.
    """
    anon = TestClient(serve.app)
    for url in ("/", "/fr", "/pocus-copilot.dc.html"):
        r = anon.get(url, follow_redirects=False)
        assert r.status_code == 303, f"{url} did not redirect: {r.status_code}"
        assert r.headers["location"] == "/login"
    assert anon.get("/login").status_code == 200, "the login form itself must stay reachable"


# ------------------------------------------------------------- one doctor, another's data --
@prop(ACCESS)
def test_a_doctor_cannot_open_another_doctors_record():
    a, b = _doctors()
    _, record_id = _a_patient_of_alice(a)

    assert a.get(f"/api/record?id={record_id}").status_code == 200, \
        "Alice cannot open her own record"
    r = b.get(f"/api/record?id={record_id}")
    assert r.status_code == 404, f"Mallory read Alice's record: {r.status_code}"
    # 404 and not 403: a 403 would confirm that this identifier names a real record, which is
    # the fact the attacker was trying to establish.
    assert "Alice" not in r.text


@prop(ACCESS)
def test_a_doctor_cannot_open_another_doctors_patient_history():
    a, b = _doctors()
    patient_id, _ = _a_patient_of_alice(a)

    assert a.get(f"/api/patients/{patient_id}/examinations").status_code == 200
    assert b.get(f"/api/patients/{patient_id}/examinations").status_code == 404


@prop(ACCESS)
def test_a_doctor_cannot_file_an_assessment_against_another_doctors_patient():
    """The write direction, which is the one that gets forgotten.

    Reading someone else's record is the obvious attack and the obvious defence. Writing INTO
    their record is quieter and worse: it puts a clinical assessment of one patient into
    another patient's history, where a clinician will later read it as theirs.
    """
    a, b = _doctors()
    patient_id, _ = _a_patient_of_alice(a)

    r = b.post("/api/analyse", json={"name": "Mallory's forgery", "age": 30, "sex": "M",
                                     "complaint": "test", "organ": "Lung",
                                     "patientId": patient_id,
                                     "findings": {}, "vitals": {}, "labs": {}})
    assert r.status_code == 404, f"Mallory filed against Alice's patient: {r.status_code}"

    history = a.get(f"/api/patients/{patient_id}/examinations").json()["examinations"]
    assert not any("forgery" in (e.get("complaint") or "") for e in history), \
        "a forged assessment reached Alice's patient history"


@prop(ACCESS)
def test_one_doctors_patient_list_never_contains_anothers():
    a, b = _doctors()
    _a_patient_of_alice(a)
    mine = {p["name"] for p in b.get("/api/patients").json()["patients"]}
    assert "Alice's patient" not in mine


@prop(ACCESS)
def test_the_current_encounter_is_not_shared_between_doctors():
    """The bug that made this whole change urgent.

    `_last` was one module-level dict holding "the encounter being looked at". With two
    clinicians signed in, one analysing a patient changed what the other's next screen refresh
    showed, and /api/ask answered questions about the wrong patient -- with nothing on screen
    to suggest anything was wrong. It would simply have been someone else's case.
    """
    a, b = _doctors()
    a.post("/api/analyse", json={"name": "Alice's current case", "age": 64, "sex": "F",
                                 "complaint": "pleuritic chest pain", "organ": "Lung",
                                 "findings": {}, "vitals": {}, "labs": {}})
    view = b.get("/api/view").json()
    assert "Alice" not in str(view), "Alice's encounter appeared on Mallory's screen"

    answer = b.post("/api/ask", json={"question": "what is the presenting complaint?"}).json()
    assert "pleuritic" not in answer.get("answer", "").lower(), \
        "the assistant answered Mallory from Alice's encounter"


# ------------------------------------------------------------------- passwords and sessions -
@prop(ACCESS)
def test_a_password_is_never_stored_and_never_returned():
    """Neither in the database nor in any response.

    The hash is checked for the password as a substring rather than for equality with it: a
    hash that merely CONTAINED the password would also be a disclosure, and equality would not
    catch that.
    """
    from sqlalchemy import select

    from src.auth import db
    from src.auth.models import Doctor

    temp_database()
    secret = "a-very-specific-password-9713"
    try:
        accounts.add("hashcheck", "Dr Hash", None, secret)
    except SystemExit:
        pass
    with db.session() as s:
        stored = s.scalar(select(Doctor.password_hash).where(Doctor.username == "hashcheck"))
    assert stored and secret not in stored, "the password is recoverable from its own hash"
    assert stored.startswith("scrypt$")
    assert passwords.verify_password(secret, stored)
    assert not passwords.verify_password(secret + "x", stored)

    client = TestClient(serve.app)
    client.post("/api/login", json={"username": "hashcheck", "password": secret})
    assert secret not in client.get("/api/me").text
    assert "password" not in client.get("/api/me").text.lower()


@prop(ACCESS)
def test_a_wrong_password_and_an_unknown_user_are_indistinguishable():
    """Otherwise the form enumerates who works here."""
    temp_database()
    try:
        accounts.add("realdoctor", "Dr Real", None, "real-password-1")
    except SystemExit:
        pass
    client = TestClient(serve.app)
    wrong = client.post("/api/login", json={"username": "realdoctor", "password": "nope"})
    absent = client.post("/api/login", json={"username": "ghost", "password": "nope"})
    assert wrong.status_code == absent.status_code == 401
    assert wrong.json() == absent.json(), "the two failures are told apart by their message"


@prop(ACCESS)
def test_logging_out_revokes_the_session_rather_than_forgetting_it():
    """A stolen cookie must stop working, not merely stop being sent.

    This is the difference between a server-side session and a signed token, and the reason
    this project stores sessions in a table. The check replays the token from a NEW client,
    which is exactly what someone holding a copied cookie would do.
    """
    temp_database()
    try:
        accounts.add("logoutdoc", "Dr Logout", None, "logout-password-1")
    except SystemExit:
        pass
    client = TestClient(serve.app)
    client.post("/api/login", json={"username": "logoutdoc", "password": "logout-password-1"})
    token = client.cookies.get(sessions.COOKIE)
    assert token, "no session cookie was set"
    assert client.get("/api/me").status_code == 200

    client.post("/api/logout")

    thief = TestClient(serve.app)
    thief.cookies.set(sessions.COOKIE, token)
    assert thief.get("/api/me").status_code == 401, \
        "the token still works after logout: the session was forgotten, not revoked"


@prop(ACCESS)
def test_a_forged_or_expired_session_cookie_is_refused():
    from datetime import timedelta

    from src.auth import db
    from src.auth.models import Session, utcnow

    temp_database()
    forged = TestClient(serve.app)
    forged.cookies.set(sessions.COOKIE, "not-a-real-token-just-a-guess")
    assert forged.get("/api/me").status_code == 401

    try:
        accounts.add("expdoc", "Dr Expired", None, "expired-password-1")
    except SystemExit:
        pass
    token = sessions.login("expdoc", "expired-password-1")
    with db.session() as s:
        row = s.get(Session, token)
        row.expires_at = utcnow() - timedelta(minutes=1)
        s.commit()
    stale = TestClient(serve.app)
    stale.cookies.set(sessions.COOKIE, token)
    assert stale.get("/api/me").status_code == 401, "an expired session still authenticates"


@prop(ACCESS)
def test_the_session_cookie_is_not_readable_by_script():
    """HttpOnly, so an injected script cannot read the session out of document.cookie."""
    temp_database()
    try:
        accounts.add("cookiedoc", "Dr Cookie", None, "cookie-password-1")
    except SystemExit:
        pass
    client = TestClient(serve.app)
    r = client.post("/api/login", json={"username": "cookiedoc", "password": "cookie-password-1"})
    header = r.headers.get("set-cookie", "")
    assert "httponly" in header.lower(), f"session cookie is script-readable: {header}"
    assert "samesite=lax" in header.lower().replace(" ", ""), \
        f"session cookie has no SameSite protection: {header}"


# -------------------------------------------------------------------------- registration ----
# Doctors create their own accounts. The risk that comes with that is stated in serve.py; what
# these tests establish is the part that must hold regardless: registering adds a person to the
# system and adds nothing to what any person can see.
@prop(ACCESS)
def test_a_newly_registered_doctor_sees_nobody_elses_patients():
    """The property that makes open registration survivable.

    If sign-up were a way to reach existing data, it would be a way in for anyone. A new
    account starts empty, and stays empty until its owner records something.
    """
    a, _ = _doctors()
    _a_patient_of_alice(a)

    fresh = TestClient(serve.app)
    r = fresh.post("/api/register", json={"username": "newcomer", "name": "Dr Newcomer",
                                          "password": "newcomer-password-1"})
    assert r.status_code == 200, f"registration failed: {r.text}"

    assert fresh.get("/api/patients").json()["patients"] == []
    assert fresh.get("/api/bootstrap").json()["records"] == []
    counts = fresh.get("/api/me").json()
    assert counts["patients"] == 0 and counts["examinations"] == 0


@prop(ACCESS)
def test_registration_signs_the_new_doctor_in_and_stores_only_a_hash():
    from sqlalchemy import select

    from src.auth import db
    from src.auth.models import Doctor

    temp_database()
    client = TestClient(serve.app)
    secret = "register-password-4471"
    r = client.post("/api/register", json={"username": "regcheck", "name": "Dr Reg",
                                           "password": secret})
    assert r.status_code == 200
    assert client.cookies.get(sessions.COOKIE), "registration did not sign the doctor in"
    assert client.get("/api/me").json()["name"] == "Dr Reg"

    with db.session() as s:
        stored = s.scalar(select(Doctor.password_hash).where(Doctor.username == "regcheck"))
    assert stored.startswith("scrypt$") and secret not in stored


@prop(ACCESS)
def test_registration_refuses_a_duplicate_username_and_a_short_password():
    temp_database()
    first = TestClient(serve.app)
    assert first.post("/api/register", json={"username": "taken", "name": "Dr First",
                                             "password": "first-password-1"}).status_code == 200

    second = TestClient(serve.app)
    dup = second.post("/api/register", json={"username": "taken", "name": "Dr Second",
                                             "password": "second-password-1"})
    assert dup.status_code == 400, "a second account took an existing username"

    short = second.post("/api/register", json={"username": "shorty", "name": "Dr Short",
                                               "password": "abc"})
    assert short.status_code == 400, "an account was created with a three-character password"

    # And the refusals must not have signed anybody in.
    assert second.get("/api/me").status_code == 401


@prop(ACCESS)
def test_registration_cannot_be_used_to_take_over_an_existing_account():
    """Re-registering a username must not overwrite the password on the existing account.

    This is the attack the duplicate check actually prevents, and asserting the 400 alone would
    not catch a version that returned 400 after having already written the new hash.
    """
    temp_database()
    victim = TestClient(serve.app)
    victim.post("/api/register", json={"username": "victim", "name": "Dr Victim",
                                       "password": "victim-password-1"})

    attacker = TestClient(serve.app)
    attacker.post("/api/register", json={"username": "victim", "name": "Dr Attacker",
                                         "password": "attacker-password-1"})

    assert sessions.login("victim", "attacker-password-1") is None, \
        "re-registering overwrote the existing account's password"
    assert sessions.login("victim", "victim-password-1") is not None, \
        "the original password stopped working"


@prop(ACCESS)
def test_registration_can_be_closed_for_a_deployment():
    """POCUS_OPEN_REGISTRATION=0 is what a real deployment would set."""
    import os

    temp_database()
    client = TestClient(serve.app)
    os.environ["POCUS_OPEN_REGISTRATION"] = "0"
    try:
        r = client.post("/api/register", json={"username": "blocked", "name": "Dr Blocked",
                                               "password": "blocked-password-1"})
        assert r.status_code == 400
        assert sessions.login("blocked", "blocked-password-1") is None, \
            "the account was created even though registration was closed"
    finally:
        os.environ.pop("POCUS_OPEN_REGISTRATION", None)
