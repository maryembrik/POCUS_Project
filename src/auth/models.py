"""The database: doctors, their patients, and the examinations belonging to both.

Four tables. The shape is dictated by one rule -- a doctor may only ever reach their own
patients -- and that rule is enforced in the WHERE clause of every query rather than by a check
after the row has been loaded. `doctor_id` therefore appears on `examinations` as well as on
`patients`, denormalised on purpose: it lets an examination be authorised in one comparison
without a join, and a join that can be forgotten is an authorisation that can be forgotten.

Identifiers are `secrets.token_urlsafe`, not integers. The interface used `str(len(records)+1)`,
which was harmless while every user saw everything and becomes an invitation the moment records
have owners: /api/record?id=2 is a guess anyone can make. An unguessable id is not the
protection -- the ownership check is -- but it removes the cheapest attack on it.

Times are stored as timezone-aware UTC. A clinical record whose timestamps depend on the
server's local zone cannot be compared across deployments.
"""
from __future__ import annotations

import secrets
from datetime import datetime, timezone

from sqlalchemy import (JSON, DateTime, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def new_id() -> str:
    return secrets.token_urlsafe(12)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Doctor(Base):
    __tablename__ = "doctors"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    # Never the password. The column name says what it holds so that a reader of the schema
    # cannot mistake it, and nothing in this project ever writes a plaintext password anywhere.
    password_hash: Mapped[str] = mapped_column(String(255))
    name: Mapped[str] = mapped_column(String(120))
    email: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    patients: Mapped[list[Patient]] = relationship(back_populates="doctor",
                                                   cascade="all, delete-orphan")


class Patient(Base):
    __tablename__ = "patients"
    # A reference is unique per doctor, not globally: two clinicians may each have a "P-001",
    # and forcing one namespace on them would make the second doctor's numbering depend on the
    # first's.
    __table_args__ = (UniqueConstraint("doctor_id", "reference", name="uq_patient_ref"),)

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.id", ondelete="CASCADE"),
                                           index=True)
    reference: Mapped[str] = mapped_column(String(64))
    name: Mapped[str] = mapped_column(String(120))
    date_of_birth: Mapped[str | None] = mapped_column(String(10), nullable=True)
    sex: Mapped[str | None] = mapped_column(String(1), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    doctor: Mapped[Doctor] = relationship(back_populates="patients")
    examinations: Mapped[list[Examination]] = relationship(
        back_populates="patient", cascade="all, delete-orphan",
        order_by="Examination.created_at.desc()")


class Examination(Base):
    """One assessment, stored whole rather than as a summary.

    The interface re-opens a past encounter and must show what was shown at the time -- its
    findings, alerts, differential, timeline and report. Keeping only headline columns would
    mean re-deriving the rest on open, and a record that reconstructs itself is a record that
    can differ from the one the clinician acted on. So the computed state travels in the JSON
    columns, and the indexed columns beside them exist only so the patient list can be drawn
    without loading every blob.

    `content_sha256` is the digest archive_report() computes over the canonical report with the
    generation timestamp removed. It is the join to the on-disk archive: the row points at the
    archived report instead of being a second copy that can drift from it.
    """
    __tablename__ = "examinations"

    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=new_id)
    # Nullable, because an assessment can precede the patient record. A clinician working a
    # resuscitation enters what they have and files it against a patient afterwards; requiring
    # the record first would make the tool demand paperwork before it would think. The
    # examination is still owned -- doctor_id below is not nullable -- so an unfiled assessment
    # belongs to exactly one person and is reachable only by them.
    patient_id: Mapped[str | None] = mapped_column(
        ForeignKey("patients.id", ondelete="CASCADE"), index=True, nullable=True)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.id", ondelete="CASCADE"),
                                           index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow,
                                                 index=True)

    encounter_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    alerts: Mapped[int] = mapped_column(Integer, default=0)
    organ: Mapped[str | None] = mapped_column(String(32), nullable=True)
    complaint: Mapped[str | None] = mapped_column(Text, nullable=True)
    content_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)

    encounter_json: Mapped[dict] = mapped_column(JSON, default=dict)
    assessment_json: Mapped[dict] = mapped_column(JSON, default=dict)
    studies_json: Mapped[list] = mapped_column(JSON, default=list)

    patient: Mapped[Patient] = relationship(back_populates="examinations")


class Session(Base):
    """Server-side sessions, so that logging out actually ends the session.

    A signed cookie carrying the doctor's id would authenticate just as well and could not be
    revoked: the token stays valid until it expires no matter what the server does, so "log out"
    would mean "please forget this locally". On a shared clinical workstation that is the wrong
    behaviour, and it is the reason this table exists rather than a stateless token.
    """
    __tablename__ = "sessions"

    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    doctor_id: Mapped[str] = mapped_column(ForeignKey("doctors.id", ondelete="CASCADE"),
                                           index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


Index("ix_exam_doctor_created", Examination.doctor_id, Examination.created_at.desc())
