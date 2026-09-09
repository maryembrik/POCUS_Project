"""Every query a request can reach, and every one of them scoped to a doctor.

The single rule this module exists to enforce: a doctor's id is a REQUIRED argument of every
function here, and it appears in the WHERE clause of every statement. There is deliberately no
`get_patient(patient_id)` for a caller to reach for and forget to guard -- the unguarded version
does not exist, so it cannot be called by accident.

A row belonging to someone else is reported as absent, not as forbidden. Returning 403 for a
patient that exists and 404 for one that does not tells an unauthorised caller which ids are
real, which is a disclosure in itself. Both are None here, and the endpoint turns both into 404.
"""
from __future__ import annotations

from typing import Any

from sqlalchemy import func, select

from . import db
from .models import Examination, Patient


# ---------------------------------------------------------------------------- patients ----
def list_patients(doctor_id: str) -> list[dict[str, Any]]:
    with db.session() as s:
        rows = s.execute(
            select(Patient,
                   func.count(Examination.id).label("n"),
                   func.max(Examination.created_at).label("last"))
            .outerjoin(Examination, Examination.patient_id == Patient.id)
            .where(Patient.doctor_id == doctor_id)
            .group_by(Patient.id)
            .order_by(func.coalesce(func.max(Examination.created_at),
                                    Patient.created_at).desc())).all()

        out = []
        for p, n, last in rows:
            # The most recent complaint, so a row in the patient list says what this person was
            # last seen ABOUT. A list of names and dates makes the clinician open each card to
            # find out which one they mean.
            recent = s.scalar(
                select(Examination.complaint)
                .where(Examination.patient_id == p.id)
                .order_by(Examination.created_at.desc()).limit(1)) if n else None
            out.append({"id": p.id, "reference": p.reference, "name": p.name,
                        "dateOfBirth": p.date_of_birth, "sex": p.sex,
                        "examinations": n, "lastSeen": last.isoformat() if last else None,
                        "lastComplaint": recent})
        return out


def create_patient(doctor_id: str, *, name: str, reference: str | None = None,
                   date_of_birth: str | None = None, sex: str | None = None) -> dict[str, Any]:
    with db.session() as s:
        if not reference:
            # Per doctor, so one clinician's numbering never depends on another's.
            n = s.scalar(select(func.count(Patient.id))
                         .where(Patient.doctor_id == doctor_id)) or 0
            reference = f"P-{n + 1:03d}"
        p = Patient(doctor_id=doctor_id, name=name.strip() or "Unnamed patient",
                    reference=reference, date_of_birth=date_of_birth, sex=sex)
        s.add(p)
        s.commit()
        return {"id": p.id, "reference": p.reference, "name": p.name,
                "dateOfBirth": p.date_of_birth, "sex": p.sex,
                "examinations": 0, "lastSeen": None}


def find_or_create_patient(doctor_id: str, *, name: str, age: int | None = None,
                           sex: str | None = None) -> dict[str, Any] | None:
    """The patient this encounter is about, created on first sight.

    The clinician already types a name, an age and a sex into the intake form. Asking them to
    ALSO pick a patient record from a list, before the system will think about the case, is
    paperwork the tool imposes on itself -- and in an emergency department it is paperwork at
    exactly the wrong moment. So the record is made from what was already entered.

    Matched on name, age and sex together rather than on name alone: two patients called the
    same thing would otherwise share one history, which is a worse failure than two records for
    one person. It is still a heuristic, and the honest description of it is that this is a
    prototype's substitute for a hospital's patient index, not a replacement for one.

    Returns None for a blank name -- an unnamed encounter is not filed under a patient rather
    than being filed under a patient called "Unnamed patient", which would collect every
    unnamed case from every session into one person's record.
    """
    name = (name or "").strip()
    if not name or name.lower() in ("unnamed patient", "no patient"):
        return None

    with db.session() as s:
        q = select(Patient).where(Patient.doctor_id == doctor_id, Patient.name == name)
        for p in s.scalars(q).all():
            if (age is None or p.date_of_birth is None or str(age) == p.date_of_birth) \
                    and (sex is None or p.sex is None or sex == p.sex):
                return {"id": p.id, "reference": p.reference, "name": p.name,
                        "dateOfBirth": p.date_of_birth, "sex": p.sex}
    # date_of_birth carries the AGE as entered, because the form collects an age and not a date
    # of birth. Storing an age in a column named for a birth date would be a lie the next reader
    # has to discover, so the interface reports it as an age and this note says why.
    return create_patient(doctor_id, name=name, date_of_birth=str(age) if age else None,
                          sex=sex)


def get_patient(doctor_id: str, patient_id: str) -> dict[str, Any] | None:
    with db.session() as s:
        p = s.scalar(select(Patient).where(Patient.id == patient_id,
                                           Patient.doctor_id == doctor_id))
        if p is None:
            return None
        return {"id": p.id, "reference": p.reference, "name": p.name,
                "dateOfBirth": p.date_of_birth, "sex": p.sex}


# ------------------------------------------------------------------------ examinations ----
def save_examination(doctor_id: str, patient_id: str | None, *, encounter: dict,
                     assessment: dict, studies: list[dict],
                     content_sha256: str | None = None) -> dict[str, Any] | None:
    """Store one assessment whole. Returns None if the patient is not this doctor's.

    The ownership check is a query, not a trust: a patient_id arriving from the browser is
    checked against this doctor before anything is written, so a forged id writes nothing
    rather than filing a real assessment under someone else's patient.
    """
    with db.session() as s:
        if patient_id is not None:
            owned = s.scalar(select(Patient.id).where(Patient.id == patient_id,
                                                      Patient.doctor_id == doctor_id))
            if owned is None:
                return None

        support = (assessment.get("support") or {})
        e = Examination(
            patient_id=patient_id, doctor_id=doctor_id,
            encounter_id=(assessment.get("report") or {}).get("encounter_id"),
            severity=(support.get("severity") or {}).get("severity"),
            alerts=len(support.get("alerts") or []),
            organ=encounter.get("organ"),
            complaint=encounter.get("complaint"),
            content_sha256=content_sha256,
            encounter_json=encounter, assessment_json=assessment, studies_json=studies)
        s.add(e)
        s.commit()
        return {"id": e.id, "createdAt": e.created_at.isoformat()}


def list_examinations(doctor_id: str, patient_id: str | None = None) -> list[dict[str, Any]]:
    """The doctor's examinations, newest first, optionally for one patient.

    Deliberately does not load the JSON blobs: this draws a list, and a list that loads every
    stored assessment to show a row of headlines gets slower with every case ever recorded.
    """
    with db.session() as s:
        q = (select(Examination, Patient)
             .outerjoin(Patient, Patient.id == Examination.patient_id)
             .where(Examination.doctor_id == doctor_id))
        if patient_id is not None:
            q = q.where(Examination.patient_id == patient_id)
        rows = s.execute(q.order_by(Examination.created_at.desc())).all()
        return [{"id": e.id, "patientId": e.patient_id,
                 "name": p.name if p else "Unnamed patient",
                 "reference": p.reference if p else None,
                 "at": e.created_at.isoformat(), "severity": e.severity,
                 "alerts": e.alerts, "organ": e.organ, "complaint": e.complaint,
                 "encounterId": e.encounter_id} for e, p in rows]


def load_examination(doctor_id: str, exam_id: str) -> dict[str, Any] | None:
    """The whole stored assessment, so re-opening shows what was shown at the time."""
    with db.session() as s:
        e = s.scalar(select(Examination).where(Examination.id == exam_id,
                                               Examination.doctor_id == doctor_id))
        if e is None:
            return None
        return {"id": e.id, "patientId": e.patient_id,
                "encounter": e.encounter_json, "assessment": e.assessment_json,
                "studies": e.studies_json, "at": e.created_at.isoformat()}


def attach_studies(doctor_id: str, exam_id: str, studies: list[dict]) -> bool:
    """File further studies against a stored examination without re-running the assessment.

    The reason is the one already given at /api/record/attach: the differential, alerts and
    severity were reached without this image, and quietly re-deriving them would replace what
    the clinician saw with something else under the same encounter identifier.
    """
    with db.session() as s:
        e = s.scalar(select(Examination).where(Examination.id == exam_id,
                                               Examination.doctor_id == doctor_id))
        if e is None:
            return False
        have = {st.get("id") for st in (e.studies_json or [])}
        e.studies_json = list(e.studies_json or []) + [st for st in studies
                                                       if st.get("id") not in have]
        enc = dict(e.encounter_json or {})
        enc["images"] = [st["id"] for st in e.studies_json]
        e.encounter_json = enc
        s.commit()
        return True


def counts(doctor_id: str) -> dict[str, int]:
    with db.session() as s:
        return {
            "patients": s.scalar(select(func.count(Patient.id))
                                 .where(Patient.doctor_id == doctor_id)) or 0,
            "examinations": s.scalar(select(func.count(Examination.id))
                                     .where(Examination.doctor_id == doctor_id)) or 0,
        }
