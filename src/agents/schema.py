"""Shared output contract for the POCUS-Emergency agents.

Why this module exists
----------------------
The Ultrasound Agent routes by organ, and each organ module was developed separately -- cardiac
emits measurements, gallbladder emits a clinical group, lung emits several simultaneous findings.
Three shapes is exactly what the architecture forbids ("the same output schema"), so the shape is
defined once here and imported by every notebook.

Why files rather than direct calls
----------------------------------
The organ models and the reasoning LLM cannot co-exist in one Colab session, so the agents run at
different times in different processes. Communication therefore goes through an *encounter bundle*
on disk: Triage and Ultrasound write their own sections independently (they run in parallel and
neither depends on the other), and the Clinical Reasoning Agent reads the assembled bundle.

Stdlib only, so it imports identically in Colab and locally.
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "1.0"

ORGANS = {"heart", "lung", "gallbladder", "vascular", "fast"}
STATUSES = {"ok", "not_supported", "failed"}
URGENCIES = {"low", "medium", "high"}


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------
def make_finding(label: str, confidence: float, group: str | None = None) -> dict:
    """One detected finding.

    `confidence` must be a *calibrated* probability. Raw softmax or raw agreement fractions are
    not comparable across organs, and the reasoning agent fuses them as if they were -- so the
    calibration step in each organ notebook is what makes this field meaningful, not decorative.
    """
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return {"label": str(label), "group": group, "confidence": round(float(confidence), 3)}


def make_report(
    organ: str,
    findings: Iterable[dict] = (),
    *,
    not_detected: Iterable[dict] = (),
    status: str = "ok",
    measurements: dict | None = None,
    quality: dict | None = None,
    reliability: dict | None = None,
    explanation: dict | None = None,
    model: str | None = None,
) -> dict:
    """An Ultrasound Agent report for one organ.

    `findings` is always a list, even when the organ yields a single label: lung is genuinely
    multi-label (B-lines and an effusion co-occur), and a schema that assumed one finding would
    force that module into a shape it does not fit.

    `not_detected` carries the findings the module screened for and did *not* see. It is a
    separate field rather than an absence from `findings`, because three states must stay
    distinguishable downstream:

        detected        -- the module looked and found it
        not_detected    -- the module looked and did not find it
        neither list    -- the module never assessed it at all

    Collapsing the second into the third loses real information; collapsing it into a negative
    result invents one. Only a module that screens a fixed label set can populate it.

    Unsupported organs return status='not_supported' with both lists empty -- same shape, so
    the reasoning agent never needs a special case for them.
    """
    if organ not in ORGANS:
        raise ValueError(f"unknown organ {organ!r}; expected one of {sorted(ORGANS)}")
    if status not in STATUSES:
        raise ValueError(f"unknown status {status!r}; expected one of {sorted(STATUSES)}")

    return {
        "schema_version": SCHEMA_VERSION,
        "organ": organ,
        "status": status,
        "findings": [dict(f) for f in findings],
        "not_detected": [dict(f) for f in not_detected],
        "measurements": dict(measurements or {}),
        "quality": dict(quality or {}),
        "reliability": dict(reliability or {}),
        "explanation": dict(explanation or {}),
        "model": model,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def make_triage(urgency: str, confidence: float, *, features: dict | None = None,
                model: str | None = None) -> dict:
    """Triage Agent output. Urgency lives here and nowhere else -- the Ultrasound Agent reports
    findings only, and the escalation decision belongs to the Clinical Reasoning Agent."""
    if urgency not in URGENCIES:
        raise ValueError(f"urgency must be one of {sorted(URGENCIES)}, got {urgency!r}")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    return {
        "schema_version": SCHEMA_VERSION,
        "urgency": urgency,
        "confidence": round(float(confidence), 3),
        "features": dict(features or {}),
        "model": model,
        "produced_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------------------
def validate_report(report: dict) -> list[str]:
    """Return a list of problems; empty means valid.

    Validation happens on write so a malformed finding cannot reach the reasoning agent silently
    and be rationalised by an LLM that has no way to tell it is malformed.
    """
    errs: list[str] = []
    for key in ("schema_version", "organ", "status", "findings"):
        if key not in report:
            errs.append(f"missing key: {key}")
    if errs:
        return errs

    if report["organ"] not in ORGANS:
        errs.append(f"unknown organ: {report['organ']!r}")
    if report["status"] not in STATUSES:
        errs.append(f"unknown status: {report['status']!r}")
    if not isinstance(report["findings"], list):
        errs.append("findings must be a list")
        return errs

    negatives = report.get("not_detected") or []
    if not isinstance(negatives, list):
        errs.append("not_detected must be a list")
        negatives = []

    for field, items in (("finding", report["findings"]), ("not_detected", negatives)):
        for i, f in enumerate(items):
            if "label" not in f or "confidence" not in f:
                errs.append(f"{field}[{i}] missing label/confidence")
                continue
            c = f["confidence"]
            if not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
                errs.append(f"{field}[{i}] confidence out of range: {c!r}")

    # A screening module that fires on nothing is a legitimate, informative result -- but it
    # must say what it screened. 'ok' with both lists empty asserts an assessment happened
    # while recording nothing about it, which the reasoning layer cannot use or verify.
    if report["status"] == "ok" and not report["findings"] and not negatives:
        errs.append("status is 'ok' but both findings and not_detected are empty")
    if report["status"] == "not_supported" and (report["findings"] or negatives):
        errs.append("status is 'not_supported' but findings/not_detected is non-empty")

    labels = {str(f.get("label")) for f in report["findings"]}
    for f in negatives:
        if str(f.get("label")) in labels:
            errs.append(f"label {f.get('label')!r} appears as both detected and not detected")

    # A declared ceiling that the emitted confidence exceeds means either the calibrator was not
    # applied or the ceiling is stale. Either way the number reaching the reasoning agent is not
    # what it claims to be, so fail loudly rather than render it.
    ceiling = (report.get("reliability") or {}).get("confidence_ceiling")
    if isinstance(ceiling, (int, float)):
        for i, f in enumerate(report["findings"]):
            c = f.get("confidence")
            if isinstance(c, (int, float)) and c > ceiling + 1e-9:
                errs.append(f"finding[{i}] confidence {c} exceeds declared ceiling {ceiling}")
    return errs


# --------------------------------------------------------------------------------------
# Encounter bundle
# --------------------------------------------------------------------------------------
def _bundle_path(encounter_id: str, root: str | Path) -> Path:
    return Path(root) / "encounters" / f"{encounter_id}.json"


def _atomic_write(path: Path, payload: dict) -> None:
    """Write via a temp file + replace so a crash mid-write cannot leave a half-written bundle
    that later parses as valid JSON with missing sections."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def load_encounter(encounter_id: str, root: str | Path) -> dict:
    p = _bundle_path(encounter_id, root)
    if not p.exists():
        return {"encounter_id": encounter_id, "schema_version": SCHEMA_VERSION,
                "triage": None, "ultrasound": {}, "clinical": {}}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def write_ultrasound(encounter_id: str, report: dict, root: str | Path) -> Path:
    """Add or replace one organ's section. Organs are written independently, so two organ
    notebooks can contribute to the same encounter from different sessions."""
    errs = validate_report(report)
    if errs:
        raise ValueError("invalid report: " + "; ".join(errs))
    bundle = load_encounter(encounter_id, root)
    bundle["ultrasound"][report["organ"]] = report
    p = _bundle_path(encounter_id, root)
    _atomic_write(p, bundle)
    return p


def write_triage(encounter_id: str, triage: dict, root: str | Path) -> Path:
    bundle = load_encounter(encounter_id, root)
    bundle["triage"] = triage
    p = _bundle_path(encounter_id, root)
    _atomic_write(p, bundle)
    return p


def write_clinical(encounter_id: str, clinical: dict, root: str | Path) -> Path:
    """Age, sex, symptoms, labs -- whatever the Clinical Reasoning Agent should also weigh."""
    bundle = load_encounter(encounter_id, root)
    bundle["clinical"] = dict(clinical)
    p = _bundle_path(encounter_id, root)
    _atomic_write(p, bundle)
    return p


# --------------------------------------------------------------------------------------
# Rendering for the reasoning LLM
# --------------------------------------------------------------------------------------
# Quality keys whose *absence* is the problem: 'lv_detected': True is the healthy state, so
# surfacing it as a flag would tell the reasoning agent the opposite of what it means. Anything
# not listed here is treated as a warning when True ('fragmented', 'low_confidence', ...).
WARN_IF_FALSE = {"lv_detected", "organ_visible", "in_plane", "calibrated"}


def quality_warnings(quality: dict) -> list[str]:
    """Only the entries a clinician should be warned about, in a stable order."""
    out: list[str] = []
    for k in sorted(quality):
        v = quality[k]
        if k in WARN_IF_FALSE:
            if v is False:
                out.append(f"{k}=False")
        elif v is True:
            out.append(k)
    return out



def render_for_prompt(bundle: dict) -> str:
    """Deterministic text rendering of an encounter.

    The LLM is given this, not raw JSON: the same bundle always produces byte-identical text, so
    the reasoning step is reproducible and auditable. Confidence ceilings are spelled out inline
    because a bare '0.74' reads as hesitancy unless the reader knows it is that model's maximum.
    """
    lines: list[str] = [f"ENCOUNTER {bundle.get('encounter_id', '?')}"]

    clin = bundle.get("clinical") or {}
    if clin:
        lines.append("\nPATIENT")
        for k in sorted(clin):
            lines.append(f"  {k}: {clin[k]}")

    tri = bundle.get("triage")
    lines.append("\nTRIAGE AGENT")
    if tri:
        lines.append(f"  urgency: {tri['urgency']} (confidence {tri['confidence']:.2f})")
    else:
        lines.append("  not available")

    lines.append("\nULTRASOUND AGENT")
    us = bundle.get("ultrasound") or {}
    if not us:
        lines.append("  no organs assessed")
    for organ in sorted(us):
        rep = us[organ]
        if rep["status"] != "ok":
            lines.append(f"  {organ}: {rep['status']}")
            continue
        lines.append(f"  {organ}:")
        for f in rep["findings"]:
            grp = f" [{f['group']}]" if f.get("group") else ""
            lines.append(f"    - {f['label']}{grp} (confidence {f['confidence']:.2f})")
        for k, v in sorted((rep.get("measurements") or {}).items()):
            lines.append(f"    measurement: {k} = {v}")

        rel = rep.get("reliability") or {}
        ceiling = rel.get("confidence_ceiling")
        if ceiling is not None and ceiling < 1.0:
            lines.append(f"    NOTE: this model's confidence is capped at {ceiling:.2f} after "
                         f"calibration; treat that as its maximum, not as uncertainty.")
        if rel.get("scope"):
            lines.append(f"    scope: {rel['scope']}")
        flags = quality_warnings(rep.get("quality") or {})
        if flags:
            lines.append(f"    quality warnings: {', '.join(flags)}")

    conflicts = detect_conflicts(bundle)
    if conflicts:
        lines.append("\nCONFLICTS")
        for c in conflicts:
            lines.append(f"  - {c}")

    lines.append("\nLIMITS")
    lines.append("  A 'normal' read means none of that model's trained classes fired. It does not")
    lines.append("  rule out pathology the model was never trained to recognise.")
    return "\n".join(lines)


def detect_conflicts(bundle: dict, low_conf: float = 0.5) -> list[str]:
    """Surface disagreements the reasoning agent should not have to infer from prose.

    The architecture escalates to simulation when confidence is low or when Triage and Ultrasound
    disagree -- so those two conditions are computed here rather than left to the LLM to notice.
    """
    out: list[str] = []
    tri = bundle.get("triage")
    us = bundle.get("ultrasound") or {}

    for organ, rep in sorted(us.items()):
        if rep.get("status") != "ok":
            continue
        for f in rep.get("findings", []):
            if f["confidence"] < low_conf:
                out.append(f"low confidence: {organ} / {f['label']} at {f['confidence']:.2f}")

    if tri and tri.get("urgency") == "low":
        for organ, rep in sorted(us.items()):
            if rep.get("status") != "ok":
                continue
            for f in rep.get("findings", []):
                grp = (f.get("group") or "").lower()
                lbl = f["label"].lower()
                serious = any(w in grp or w in lbl for w in
                              ("severe", "acute inflammation", "perforation", "carcinoma"))
                if serious and f["confidence"] >= low_conf:
                    out.append(
                        f"triage says low urgency but {organ} reports '{f['label']}' "
                        f"at {f['confidence']:.2f}")
    return out
