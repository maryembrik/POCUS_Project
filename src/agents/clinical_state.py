"""Clinical State Builder -- Phase 3.

Pure Python. No model, no LLM. It takes what the upstream agents produced and assembles one
standardised object that every later module reads.

Two principles drive the design:

**Absent is not normal.** A troponin that was never drawn must never look like a troponin of zero.
Every state therefore carries an explicit `missing` list, and the renderer prints it. Without this
a language model will confidently exclude NSTEMI on the strength of a test nobody ordered.

**Confidence means one thing.** Each upstream agent calibrated its own score, so the numbers are
comparable as probabilities -- but only where calibration actually happened. Findings that arrived
uncalibrated, or that their own module flagged as thin evidence, are carried with that mark
attached rather than silently averaged in with the rest.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from . import schema as S

STATE_VERSION = "1.0"

# Reference ranges for the labs an emergency workup actually turns on. `high_is_abnormal` records
# which direction matters clinically: a lactate of 0.4 is unremarkable, a lactate of 6 is not.
LAB_REFERENCE: dict[str, dict[str, Any]] = {
    "troponin":   {"unit": "ng/L",     "normal_max": 14,   "high_is_abnormal": True,
                   "relevance": "myocardial injury"},
    "bnp":        {"unit": "pg/mL",    "normal_max": 100,  "high_is_abnormal": True,
                   "relevance": "ventricular wall stress / heart failure"},
    "d_dimer":    {"unit": "ng/mL",    "normal_max": 500,  "high_is_abnormal": True,
                   "relevance": "thrombosis; high sensitivity, low specificity"},
    "lactate":    {"unit": "mmol/L",   "normal_max": 2.0,  "high_is_abnormal": True,
                   "relevance": "tissue hypoperfusion"},
    "crp":        {"unit": "mg/L",     "normal_max": 5,    "high_is_abnormal": True,
                   "relevance": "inflammation"},
    "wbc":        {"unit": "10^9/L",   "normal_min": 4.0, "normal_max": 11.0,
                   "high_is_abnormal": True, "relevance": "infection / inflammation"},
    "creatinine": {"unit": "umol/L",   "normal_max": 110,  "high_is_abnormal": True,
                   "relevance": "renal function"},
    "ph":         {"unit": "",         "normal_min": 7.35, "normal_max": 7.45,
                   "high_is_abnormal": False, "relevance": "acid-base status"},
}

VITAL_REFERENCE: dict[str, dict[str, Any]] = {
    "hr":   {"unit": "bpm",  "normal_min": 60,  "normal_max": 100},
    "sbp":  {"unit": "mmHg", "normal_min": 90,  "normal_max": 140},
    "rr":   {"unit": "/min", "normal_min": 12,  "normal_max": 20},
    "spo2": {"unit": "%",    "normal_min": 94,  "normal_max": 100},
    "temp": {"unit": "C",    "normal_min": 36.0, "normal_max": 38.0},
}

# The Triage Agent emits the column names of its source data (o2sat, pulse, bpsys, respr),
# which are not the names used above. Matching on the canonical name alone silently routed
# every measured vital into `missing.vitals` -- reporting data as absent when it was present,
# which is the same class of error as treating absent data as normal, in the other direction.
VITAL_ALIASES: dict[str, tuple[str, ...]] = {
    "hr":   ("hr", "pulse", "heart_rate", "heartrate"),
    "sbp":  ("sbp", "bpsys", "systolic", "bp_systolic", "sysbp"),
    "rr":   ("rr", "respr", "resp_rate", "respiratory_rate"),
    "spo2": ("spo2", "o2sat", "oxygen_saturation", "sao2"),
    "temp": ("temp", "temp_c", "temperature"),
}

# Fahrenheit is converted, never aliased: 98.6 read against a 36--38 range is "high", which
# would flag a normothermic patient as febrile.
FAHRENHEIT_KEYS = ("temp_f", "temperature_f", "tempf")


def _lookup_vital(raw: dict[str, Any], name: str) -> float | None:
    """Find a vital under any of its accepted names, converting units where needed."""
    lower = {str(k).lower(): v for k, v in raw.items()}
    for alias in VITAL_ALIASES.get(name, (name,)):
        if lower.get(alias) is not None:
            return float(lower[alias])
    if name == "temp":
        for key in FAHRENHEIT_KEYS:
            if lower.get(key) is not None:
                return (float(lower[key]) - 32.0) * 5.0 / 9.0
    return None


# Labs whose absence materially changes what can be concluded. Missing CRP is a nuisance; missing
# troponin means acute coronary syndrome cannot be argued either way.
KEY_LABS = ("troponin", "lactate")

# Confidence below this is treated as weak regardless of how the finding was produced.
WEAK_CONFIDENCE = 0.40


def _evidence_grade(calibrated: bool, low_evidence: bool, confidence: float) -> str:
    """One word the reasoning layer can weigh a finding by.

    strong        calibrated probability from a module with an adequate evidence base
    limited       calibrated, but thin support -- few positives, or a low score
    experimental  uncalibrated: the number is not a probability and is not comparable
                  with the others, so it must not be averaged in alongside them
    """
    if not calibrated:
        return "experimental"
    if low_evidence or confidence < WEAK_CONFIDENCE:
        return "limited"
    return "strong"


def _case_quality(state: dict) -> dict[str, Any]:
    """How much the reasoning layer should trust this case as a whole.

    Deliberately a small set of named reasons rather than a score: 'POOR because troponin is
    absent and triage disagrees with imaging' is actionable, whereas '0.42' is not.
    """
    reasons: list[str] = []

    missing_key = [l for l in KEY_LABS if l in state["missing"]["labs"]]
    if missing_key:
        reasons.append(f"key lab(s) not obtained: {', '.join(missing_key)}")

    if state["imaging"]["organs_uncalibrated"]:
        reasons.append("uncalibrated model(s): "
                       + ", ".join(state["imaging"]["organs_uncalibrated"]))

    weak = [f for f in state["imaging"]["findings"]
            if f["detected"] and f["evidence"] != "strong"]
    if weak:
        reasons.append(f"{len(weak)} detected finding(s) resting on limited or experimental evidence")

    if state["conflicts"]:
        reasons.append(f"{len(state['conflicts'])} unresolved conflict(s) between agents")

    if not any(f["detected"] for f in state["imaging"]["findings"]):
        reasons.append("no positive imaging finding")

    if state.get("triage") is None:
        reasons.append("no triage assessment")

    # Conflicts weigh double: two agents disagreeing is a stronger warning than one gap in data.
    severity = len(reasons) + len(state["conflicts"])
    grade = "GOOD" if severity == 0 else ("MODERATE" if severity <= 2 else "POOR")
    return {"grade": grade, "reasons": reasons}


def _interpret(name: str, value: float, ref: dict[str, Any]) -> dict[str, Any]:
    """Attach an explicit high/low/normal flag rather than leaving a bare number.

    The flag is derived here, in code, so the reasoning layer is not left to infer reference
    ranges from memory -- which is exactly the sort of thing a language model gets subtly wrong.
    """
    lo, hi = ref.get("normal_min"), ref.get("normal_max")
    flag = "normal"
    if hi is not None and value > hi:
        flag = "high"
    elif lo is not None and value < lo:
        flag = "low"
    out = {"value": value, "unit": ref.get("unit", ""), "flag": flag}
    if flag != "normal":
        if lo is not None and hi is not None:
            out["reference"] = f"{lo}-{hi}"
        elif hi is not None:
            out["reference"] = f"<={hi}"
        if ref.get("relevance"):
            out["relevance"] = ref["relevance"]
    return out


def build_clinical_state(bundle: dict, *, labs: dict | None = None,
                         history: dict | None = None) -> dict:
    """Assemble one structured state from everything upstream produced.

    `bundle` is an encounter written by schema.write_triage / write_ultrasound.
    `labs` and `history` are whatever is available -- both may be empty, and what is absent is
    recorded as absent rather than defaulted.
    """
    labs = dict(labs or {})
    history = dict(history or {})
    demographics = dict(bundle.get("clinical") or {})
    demographics.update({k: v for k, v in history.items()
                         if k in ("age", "sex", "chief_complaint", "symptoms")})

    # ---- imaging findings, flattened across organs -------------------------------------
    findings: list[dict] = []
    not_assessed: list[str] = []
    uncalibrated: list[str] = []

    for organ in sorted((bundle.get("ultrasound") or {})):
        rep = bundle["ultrasound"][organ]
        if rep.get("status") != "ok":
            not_assessed.append(organ)
            continue

        rel = rep.get("reliability") or {}
        if not rel.get("confidence_calibrated", False):
            uncalibrated.append(organ)

        # A module can mark thin evidence two ways: per finding, or once in reliability as a list
        # of names. Checking only the first silently drops the flag for modules that use the
        # second -- lung reports pleural_effusion that way.
        # Both sides must be normalised the same way. Normalising only the reliability list
        # meant "pleural_effusion" was compared against {"pleural effusion"} and never
        # matched, so the flag silently did nothing for every underscored label -- which is
        # every label the modules actually emit.
        def _norm(s: object) -> str:
            return str(s).replace("_", " ").strip().lower()

        thin = {_norm(x) for x in (rel.get("unreliable_findings") or [])}

        def _low_evidence(f: dict) -> bool:
            return bool(f.get("unreliable", False)) or _norm(f["label"]) in thin

        cal = bool(rel.get("confidence_calibrated", False))

        for f in rep.get("findings", []):
            low = _low_evidence(f)
            findings.append({
                "organ": organ,
                "label": f["label"],
                "group": f.get("group"),
                "confidence": f["confidence"],
                "calibrated": cal,
                "low_evidence": low,
                "evidence": _evidence_grade(cal, low, f["confidence"]),
            })

        for f in rep.get("not_detected", []):
            low = _low_evidence(f)
            findings.append({
                "organ": organ, "label": f["label"], "group": f.get("group"),
                "confidence": f["confidence"], "calibrated": cal,
                "low_evidence": low,
                "evidence": _evidence_grade(cal, low, f["confidence"]),
                "detected": False,
            })

    for f in findings:
        f.setdefault("detected", True)

    # ---- labs and vitals ---------------------------------------------------------------
    lab_out, missing_labs = {}, []
    for name, ref in LAB_REFERENCE.items():
        if name in labs and labs[name] is not None:
            lab_out[name] = _interpret(name, float(labs[name]), ref)
        else:
            missing_labs.append(name)

    tri = bundle.get("triage") or {}
    raw_vitals = {**(tri.get("features") or {}), **{k: v for k, v in labs.items()
                                                    if k in VITAL_REFERENCE}}
    vitals_out, missing_vitals = {}, []
    for name, ref in VITAL_REFERENCE.items():
        value = _lookup_vital(raw_vitals, name)
        if value is not None:
            vitals_out[name] = _interpret(name, value, ref)
        else:
            missing_vitals.append(name)

    # ---- what the imaging agents structurally cannot exclude ---------------------------
    # Not the same as "not measured": these are findings outside the models' label sets, so no
    # amount of confident output from them constitutes a negative result.
    out_of_scope: list[str] = []
    for organ in sorted((bundle.get("ultrasound") or {})):
        rel = (bundle["ultrasound"][organ].get("reliability") or {})
        if rel.get("has_normal_class") is False:
            out_of_scope.append(
                f"{organ}: model has no healthy class -- a finding is a choice among "
                f"pathologies and never excludes disease")
        if organ == "lung" and rel.get("modelled_findings"):
            if "pneumothorax" not in rel["modelled_findings"]:
                out_of_scope.append("lung: pneumothorax is not modelled and cannot be excluded")

    state = {
        "state_version": STATE_VERSION,
        "encounter_id": bundle.get("encounter_id"),
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "demographics": demographics,
        "triage": {"urgency": tri.get("urgency"), "confidence": tri.get("confidence")} if tri else None,
        "imaging": {
            "findings": sorted(findings, key=lambda f: (not f["detected"], -f["confidence"])),
            "organs_not_assessed": not_assessed,
            "organs_uncalibrated": uncalibrated,
            "out_of_scope": out_of_scope,
        },
        "labs": lab_out,
        "vitals": vitals_out,
        "missing": {"labs": missing_labs, "vitals": missing_vitals},
        "conflicts": S.detect_conflicts(bundle),
    }
    # Depends on everything above, so it is computed last from the assembled state rather than
    # from the raw inputs.
    state["case_quality"] = _case_quality(state)
    return state


def validate_state(state: dict) -> list[str]:
    """Structural problems that would mislead the reasoning layer if they reached it."""
    errs: list[str] = []
    for key in ("state_version", "imaging", "labs", "missing"):
        if key not in state:
            errs.append(f"missing key: {key}")
    if errs:
        return errs

    for i, f in enumerate(state["imaging"]["findings"]):
        c = f.get("confidence")
        if not isinstance(c, (int, float)) or not 0.0 <= c <= 1.0:
            errs.append(f"finding[{i}] confidence out of range: {c!r}")
        if "detected" not in f:
            errs.append(f"finding[{i}] missing detected flag")

    overlap = set(state["labs"]) & set(state["missing"]["labs"])
    if overlap:
        errs.append(f"labs both present and missing: {sorted(overlap)}")
    return errs


def render_state(state: dict) -> str:
    """Deterministic text for the reasoning LLM.

    Text rather than raw JSON so the same state always produces byte-identical input, and so the
    caveats -- absent tests, uncalibrated confidence, findings outside a model's scope -- are
    stated in prose the model cannot skim past as easily as a false-valued flag.
    """
    q = state.get("case_quality") or {}
    L: list[str] = [f"CLINICAL STATE  {state.get('encounter_id', '?')}"]

    # First, because it governs how much weight anything below deserves.
    L.append(f"\nCASE QUALITY: {q.get('grade', '?')}")
    for r in q.get("reasons", []):
        L.append(f"  - {r}")
    if q.get("grade") == "POOR":
        L.append("  Treat every conclusion below as provisional and say so explicitly.")

    d = state.get("demographics") or {}
    if d:
        L.append("\nPATIENT")
        for k in sorted(d):
            L.append(f"  {k}: {d[k]}")

    t = state.get("triage")
    L.append("\nTRIAGE")
    L.append(f"  urgency: {t['urgency']} (confidence {t['confidence']:.2f})" if t
             else "  not available")

    L.append("\nIMAGING FINDINGS")
    det = [f for f in state["imaging"]["findings"] if f["detected"]]
    neg = [f for f in state["imaging"]["findings"] if not f["detected"]]
    if not det:
        L.append("  none detected")
    for f in det:
        L.append(f"  {f['organ']}: {f['label']} ({f['confidence']:.2f})  "
                 f"evidence={f['evidence']}")
    for f in neg:
        L.append(f"  {f['organ']}: {f['label']} NOT detected ({f['confidence']:.2f})  "
                 f"evidence={f['evidence']}")
    if det or neg:
        L.append("  (strong = calibrated with an adequate evidence base; limited = thin support;")
        L.append("   experimental = UNCALIBRATED, not a probability, not comparable with the rest)")

    for line in state["imaging"]["out_of_scope"]:
        L.append(f"  LIMIT: {line}")
    if state["imaging"]["organs_not_assessed"]:
        L.append(f"  not assessed: {', '.join(state['imaging']['organs_not_assessed'])}")

    L.append("\nVITALS")
    v = state.get("vitals") or {}
    if not v:
        L.append("  none recorded")
    for k in sorted(v):
        e = v[k]
        flag = '' if e['flag'] == 'normal' else f"  <-- {e['flag'].upper()}"
        L.append(f"  {k}: {e['value']} {e['unit']}{flag}")

    L.append("\nLABS")
    lb = state.get("labs") or {}
    if not lb:
        L.append("  none resulted")
    for k in sorted(lb):
        e = lb[k]
        extra = ''
        if e["flag"] != "normal":
            extra = f"  <-- {e['flag'].upper()} (ref {e.get('reference', '?')}; {e.get('relevance', '')})"
        L.append(f"  {k}: {e['value']} {e['unit']}{extra}")

    miss = state.get("missing") or {}
    if miss.get("labs") or miss.get("vitals"):
        L.append("\nNOT MEASURED -- absent, NOT normal")
        if miss.get("labs"):
            L.append(f"  labs: {', '.join(miss['labs'])}")
        if miss.get("vitals"):
            L.append(f"  vitals: {', '.join(miss['vitals'])}")
        L.append("  Do not treat any of these as excluded. They were not obtained.")

    if state.get("conflicts"):
        L.append("\nCONFLICTS")
        for c in state["conflicts"]:
            L.append(f"  - {c}")

    return "\n".join(L)
