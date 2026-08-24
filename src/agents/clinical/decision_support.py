"""Deterministic decision support: severity, alerts, examinations, scenario, protocol gate.

Everything in this module is computed from the structured clinical state. No part of it is
produced by, influenced by, or overridable by the language model, and the module imports
nothing that can reach one. That is the point: "is this patient critical?" is a question a
quantised 8B model should not be answering, and it does not need to -- the state already holds
the values the answer depends on.

The division this file maintains:

    deterministic here   severity, critical alerts, which examinations are missing, which
                         scenario applies, and whether a protocol exists to draw on
    model elsewhere      the differential, its ranking, and the explanation

`escalation_decision` in reasoning.py remains the safety-critical routing decision and is not
recomputed here. Severity summarises the case for a reader; escalation decides what happens
next. They are related but not the same, and conflating them would let a presentation concern
alter a safety decision.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

THRESHOLDS_PATH = Path(__file__).with_name("thresholds.json")


def load_thresholds(path: str | Path | None = None) -> dict[str, Any]:
    return json.loads(Path(path or THRESHOLDS_PATH).read_text(encoding="utf8"))


# ---------------------------------------------------------------------------------------
# Scenario routing
# ---------------------------------------------------------------------------------------
# The four priority presentations. Routing decides which modalities the case *should* have
# had, so that a modality never assessed can be reported as a gap rather than passing
# unnoticed. It does not select a model, and it never suppresses a finding from an organ
# outside the scenario -- an unexpected finding is still a finding.
# Cues are matched as whole words, so every form is written out rather than left as a stem.
# Stems fail in both directions and both failures were observed: "breathless" does not match
# "breathlessness" under a word boundary, and without one "stab" matches "stable". The same
# note appears on VITAL_ALIASES in reasoning.py for the same reason.
SCENARIOS: dict[str, dict[str, Any]] = {
    "acute_dyspnoea": {
        "label": "Acute dyspnoea",
        "cues": ("dyspnoea", "dyspnea", "dyspnoeic", "dyspneic",
                 "breathless", "breathlessness", "short of breath", "sob",
                 "orthopnoea", "orthopnea", "respiratory", "hypoxia", "hypoxic",
                 "wheeze", "wheezing", "desaturation"),
        "expected": ("lung", "heart"),
    },
    "undifferentiated_shock": {
        "label": "Undifferentiated shock",
        "cues": ("shock", "shocked", "hypotension", "hypotensive", "collapse",
                 "collapsed", "sepsis", "septic", "cold peripheries", "clammy",
                 "poorly perfused"),
        "expected": ("heart", "lung"),
    },
    "trauma": {
        "label": "Trauma (FAST / E-FAST)",
        "cues": ("trauma", "traumatic", "injury", "injuries", "injured",
                 "fall", "falls", "fell", "road traffic", "rta", "assault",
                 "assaulted", "stabbing", "stabbed", "blunt", "penetrating"),
        "expected": ("fast", "lung"),
    },
    "cardiorespiratory_arrest": {
        "label": "Cardiorespiratory arrest",
        "cues": ("arrest", "cpr", "resuscitation", "resuscitated", "unresponsive",
                 "pulseless", "peri-arrest", "periarrest", "rosc"),
        "expected": ("heart", "lung"),
    },
}


def _mentions(text: str, cue: str) -> bool:
    """Whole-word match, not substring.

    Substring matching routed "a stable rash" to trauma, because "stable" contains "stab".
    The same class of error was found and fixed once already in the grounding check, where
    substring matching accused a paraphrase of fabrication. Cues are short clinical words and
    several are substrings of ordinary ones, so the boundary is required rather than tidy.
    """
    return re.search(rf"\b{re.escape(cue)}\b", text) is not None


def route_scenario(state: dict) -> dict[str, Any]:
    """Which priority presentation this encounter matches.

    Matched from the presenting complaint first, because that is what a clinician actually
    routes on, and from physiology second. A case matching nothing is reported as
    `unclassified` rather than forced into the nearest scenario: routing exists to say which
    modalities were expected, and inventing an expectation would manufacture a false gap.
    """
    complaint = str((state.get("demographics") or {}).get("chief_complaint", "")).lower()
    matched: list[str] = []
    for key, spec in SCENARIOS.items():
        if any(_mentions(complaint, cue) for cue in spec["cues"]):
            matched.append(key)

    # Physiological fallback, used only when the complaint matched nothing.
    if not matched:
        v = state.get("vitals") or {}
        sbp = (v.get("sbp") or {}).get("value")
        spo2 = (v.get("spo2") or {}).get("value")
        if sbp is not None and sbp < 90:
            matched.append("undifferentiated_shock")
        elif spo2 is not None and spo2 < 94:
            matched.append("acute_dyspnoea")

    if not matched:
        return {"scenario": None, "label": "Unclassified", "expected": (),
                "matched_on": "nothing -- no scenario cue in the complaint or the vitals",
                "not_assessed": []}

    key = matched[0]
    spec = SCENARIOS[key]
    assessed = {f["organ"] for f in state["imaging"]["findings"]}
    never = list(state["imaging"].get("organs_not_assessed") or [])
    absent = [o for o in spec["expected"] if o not in assessed and o not in never]

    return {
        "scenario": key,
        "label": spec["label"],
        "expected": spec["expected"],
        "matched_on": "presenting complaint" if any(
            _mentions(complaint, c) for c in spec["cues"]) else "physiology",
        "also_matched": matched[1:] or None,
        # Expected by the scenario and not present in the record at all: neither assessed
        # nor explicitly recorded as unassessed.
        "not_assessed": sorted(set(never) | set(absent)),
    }


# ---------------------------------------------------------------------------------------
# Critical alerts
# ---------------------------------------------------------------------------------------
def critical_alerts(state: dict, escalation: dict | None = None,
                    scenario: dict | None = None,
                    thresholds: dict | None = None) -> list[dict[str, Any]]:
    """Deterministic alerts from measured values and the structure of the record.

    Two kinds, and the second is the one a simpler system omits. Physiological alerts come
    from a value crossing a configured bound. Structural alerts come from the state of the
    record itself -- agents disagreeing, a key test never obtained, a modality the scenario
    expected and nobody performed. A patient can be in danger because of what was not done,
    and that is not visible in any vital sign.

    Every alert names the threshold it fired on, so a reader can check it against
    thresholds.json rather than trusting the label.
    """
    T = thresholds or load_thresholds()
    alerts: list[dict[str, Any]] = []

    def add(severity: str, kind: str, message: str, **extra) -> None:
        alerts.append({"severity": severity, "type": kind, "message": message, **extra})

    def check(name: str, entry: dict, spec: dict, source: str) -> None:
        val = entry.get("value")
        if val is None:
            return
        unit = spec.get("unit", "")
        for bound, comparison in (("critical_below", "below"), ("warning_below", "below"),
                                  ("critical_above", "above"), ("warning_above", "above")):
            limit = spec.get(bound)
            if limit is None:
                continue
            crossed = val < limit if comparison == "below" else val > limit
            if not crossed:
                continue
            level = "CRITICAL" if bound.startswith("critical") else "WARNING"
            # Direction-specific first. A measurement has bounds on both sides and they are
            # opposite conditions: a heart rate crossing the LOW bound is a bradycardia, and
            # labelling it with the high bound's name announced the opposite of what happened.
            # The numeric message below was always right; the label is what is read first.
            label = spec.get(f"{bound}_label") or spec.get(f"{level.lower()}_label", name)
            add(level, label.upper().replace(" ", "_"),
                f"{name} {val} {unit} is {comparison} the configured "
                f"{level.lower()} bound of {limit} {unit}",
                measurement=name, value=val, threshold=limit, source=source)
            return           # one alert per measurement, the most severe crossed

    for name, entry in (state.get("vitals") or {}).items():
        spec = T["vitals"].get(name)
        if spec:
            check(name, entry, spec, "vitals")
    for name, entry in (state.get("labs") or {}).items():
        spec = T["labs"].get(name)
        if spec:
            check(name, entry, spec, "labs")

    # ---- structural alerts ------------------------------------------------------------
    for c in (state.get("conflicts") or []):
        add("CRITICAL", "AGENT_CONFLICT",
            f"the agents disagree and the system cannot determine which is correct: {c}")

    grade = (state.get("case_quality") or {}).get("grade")
    if grade == "POOR":
        add("WARNING", "POOR_CASE_QUALITY",
            "the record is too incomplete or too inconsistent for its conclusions to carry "
            "their usual weight")

    detected = [f for f in state["imaging"]["findings"] if f["detected"]]
    key_missing = [l for l in ("troponin", "lactate") if l in state["missing"]["labs"]]
    if detected and key_missing:
        add("CRITICAL", "POSITIVE_IMAGING_KEY_LABS_ABSENT",
            f"a positive imaging finding with {', '.join(key_missing)} never measured",
            missing=key_missing)

    never = list(state["imaging"].get("organs_not_assessed") or [])
    expected_gap = list((scenario or {}).get("not_assessed") or [])
    gaps = sorted(set(never) | set(expected_gap))
    if gaps:
        add("CRITICAL", "MODALITY_NOT_ASSESSED",
            f"expected but never assessed: {', '.join(gaps)} -- a gap in the record, not a "
            f"negative result", modalities=gaps)

    order = {"CRITICAL": 0, "WARNING": 1}
    return sorted(alerts, key=lambda a: order.get(a["severity"], 2))


# ---------------------------------------------------------------------------------------
# Unified severity
# ---------------------------------------------------------------------------------------
def severity_level(state: dict, escalation: dict, alerts: list[dict] | None = None,
                   thresholds: dict | None = None) -> dict[str, Any]:
    """One severity for the whole encounter, derived rather than judged.

    Deliberately monotone: any single reason to be worried raises the level, and nothing
    lowers it. A triage tier of low does not reduce the severity of a case carrying a
    critical alert -- that combination is exactly the conflict the system exists to surface,
    and averaging the two would bury it.

    Severity is a summary for a reader. It does not feed back into `escalation_decision`,
    which is computed independently and stays the routing authority.
    """
    alerts = alerts if alerts is not None else critical_alerts(state, escalation,
                                                               thresholds=thresholds)
    triage = state.get("triage") or {}
    tier = str(triage.get("urgency", "")).lower()
    critical = [a for a in alerts if a["severity"] == "CRITICAL"]
    warnings = [a for a in alerts if a["severity"] == "WARNING"]
    detected = [f for f in state["imaging"]["findings"] if f["detected"]]

    reasons: list[str] = []
    level = "LOW"

    if tier == "high":
        level, _ = "HIGH", reasons.append("triage assessed the patient as high urgency")
    if escalation.get("escalate"):
        level = "HIGH"
        reasons.append(f"the escalation policy fired "
                       f"({len(escalation.get('triggers') or [])} trigger(s))")
    if critical:
        level = "HIGH"
        reasons.append(f"{len(critical)} critical alert(s): "
                       f"{', '.join(a['type'] for a in critical)}")

    if level != "HIGH":
        if tier == "medium":
            level, _ = "MODERATE", reasons.append("triage assessed the patient as medium urgency")
        if detected:
            level = "MODERATE"
            reasons.append(f"{len(detected)} positive imaging finding(s) without an "
                           f"escalation trigger")
        if warnings:
            level = "MODERATE"
            reasons.append(f"{len(warnings)} warning-level alert(s)")

    if not reasons:
        reasons.append("no escalation trigger, no critical value, and no positive finding")

    return {
        "severity": level,
        "reasons": reasons,
        "triage_urgency": tier.upper() or None,
        "triage_confidence": triage.get("confidence"),
        "escalation": bool(escalation.get("escalate")),
        "route": escalation.get("route"),
        "critical_alerts": len(critical),
        "warnings": len(warnings),
        # Stated explicitly because it is the property that makes the number safe to show.
        "derivation": "deterministic from the clinical state; not produced by the language "
                      "model and not overridable by it",
    }


# ---------------------------------------------------------------------------------------
# Recommended additional examinations
# ---------------------------------------------------------------------------------------
# Which absent test most bears on which diagnosis. Reused from the confidence guard so that
# the recommendation and the confidence check cannot disagree about what is decisive.
from .reasoning import DECISIVE_TESTS  # noqa: E402


def recommended_examinations(state: dict, escalation: dict,
                             differential: dict | None = None,
                             scenario: dict | None = None) -> list[dict[str, Any]]:
    """What to obtain next, ranked, with the reason each is on the list.

    Built from the record rather than from the model's prose. An examination reaches this
    list because a test was never performed, because an escalation trigger names it, or
    because a modality the scenario expected is missing -- all facts about the state.

    The differential, when available, is used only to raise priority: a missing D-dimer
    matters more when pulmonary embolism is on the list than when it is not. It cannot add an
    examination that the record does not already show to be absent, so a hallucinated
    diagnosis cannot cause a hallucinated recommendation.

    The system recommends an examination. It does not order one, and nothing here is phrased
    as an instruction.
    """
    diagnoses = [str(d.get("diagnosis", "")).lower()
                 for d in ((differential or {}).get("differential") or [])]

    def decisive_for_any(test: str) -> str | None:
        for keys, tests in DECISIVE_TESTS.items():
            if test in tests and any(k in dx for dx in diagnoses for k in keys):
                return next(k for k in keys if any(k in dx for dx in diagnoses))
        return None

    recs: list[dict[str, Any]] = []
    seen: set[str] = set()

    def add(exam: str, reason: str, priority: str, **extra) -> None:
        if exam.lower() in seen:
            return
        seen.add(exam.lower())
        recs.append({"exam": exam, "reason": reason, "priority": priority, **extra})

    # 1. A modality the scenario expected and nobody performed. Highest priority: an
    #    unperformed scan is usually more informative than any further blood test.
    for organ in ((scenario or {}).get("not_assessed") or []):
        add(f"{organ} ultrasound",
            "expected for this presentation and never assessed; an unperformed scan is "
            "missing information, not a negative result",
            "HIGH", kind="imaging")

    # 2. Key laboratory values named by an escalation trigger.
    for trig in (escalation.get("triggers") or []):
        if "key lab(s) absent" in trig:
            for test in trig.split(":")[-1].split(","):
                t = test.strip()
                if t:
                    add(t, "named by an escalation trigger as a key value that was never "
                           "measured", "HIGH", kind="laboratory")

    # 3. Absent tests decisive for a diagnosis under consideration.
    for test in state["missing"]["labs"]:
        key = decisive_for_any(test)
        if key:
            add(test, f"absent, and the most confirmatory test for '{key}' which is on the "
                      f"differential", "HIGH", kind="laboratory")

    # 4. Everything else that was never measured, at lower priority.
    for test in state["missing"]["labs"]:
        add(test, "never measured; absent rather than normal", "MODERATE", kind="laboratory")
    for vital in state["missing"]["vitals"]:
        add(vital, "never recorded; absent rather than normal", "MODERATE", kind="vital sign")

    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    return sorted(recs, key=lambda r: order.get(r["priority"], 3))


# ---------------------------------------------------------------------------------------
# Therapeutic considerations -- protocol-constrained
# ---------------------------------------------------------------------------------------
def therapeutic_considerations(state: dict, scenario: dict | None = None,
                               retrieved: list[dict] | None = None,
                               corpus: dict | None = None) -> dict[str, Any]:
    """Considerations drawn from an approved protocol, or nothing.

    This layer generates no treatment. It surfaces text from a sourced protocol in the
    retrieval corpus, attributed to it, and only when a protocol for the routed scenario is
    actually present. If none is, it returns an empty list and says why -- the same discipline
    the corpus applies to placeholder passages, and for the same reason: a suggestion that
    looks protocol-backed while resting on a model's training data is worse than no suggestion,
    because a reader cannot tell the difference.

    The distinction the wording preserves is between a consideration and an order. This system
    is decision support; `check_scope_of_advice` in reasoning.py rejects the model's output
    when it instructs treatment, and nothing here may do what the model is forbidden from
    doing.
    """
    key = (scenario or {}).get("scenario")

    # Scenario-matched, not merely protocol-shaped. A protocol's topic encodes the
    # presentation it covers, and the initial management of undifferentiated shock is not
    # advice about a trauma case; surfacing it there would be worse than surfacing nothing,
    # because it would arrive with a real citation attached. An unclassified encounter matches
    # no protocol at all.
    def covers(topic: str) -> bool:
        if not topic.startswith("protocol_"):
            return False
        if topic == "protocol_general":
            return True
        return bool(key) and topic.startswith(f"protocol_{key}")

    # Looked up by topic rather than retrieved by similarity. The scenario is already known,
    # so the protocol for it is an exact selection, and ranking would only introduce a way for
    # the right protocol to be missed: measured on this corpus, a shock presentation retrieves
    # finding-vocabulary passages and ranks the management-vocabulary protocol nowhere. A
    # therapeutic layer that silently falls back to nothing because a similarity score came in
    # low is worse than one that looks the protocol up.
    #
    # `retrieved` still takes precedence when supplied, so a caller can pass an explicit set.
    if retrieved is None:
        from .retrieval import load_corpus
        source = (corpus or load_corpus())["passages"]
    else:
        source = retrieved

    protocols = [h for h in source
                 if h.get("status") == "sourced" and covers(str(h.get("topic", "")))]

    if not protocols:
        return {
            "considerations": [],
            "status": "no approved protocol available",
            "note": (
                "No sourced protocol passage was retrieved for this presentation, so no "
                "therapeutic consideration is offered. The mechanism is in place; the corpus "
                "does not yet contain a protocol source. Suggestions are never generated from "
                "the language model's own knowledge, because a reader could not distinguish "
                "those from protocol-backed ones."),
            "scenario": key,
        }

    considerations = [{
        "consideration": p["text"],
        "basis": p["source"],
        "passage": p["id"],
        "status": "CONSIDER",
        "disclaimer": "Requires clinician verification. This is a consideration drawn from a "
                      "protocol, not an instruction, and the decision remains the clinician's.",
    } for p in protocols]

    return {"considerations": considerations, "status": "protocol-backed", "scenario": key}


# ---------------------------------------------------------------------------------------
def decision_support(state: dict, escalation: dict, differential: dict | None = None,
                     retrieved: list[dict] | None = None,
                     thresholds: dict | None = None) -> dict[str, Any]:
    """The complete Module 4 object: severity, alerts, examinations, therapeutics, scenario."""
    T = thresholds or load_thresholds()
    scenario = route_scenario(state)
    alerts = critical_alerts(state, escalation, scenario, T)
    return {
        "scenario": scenario,
        "severity": severity_level(state, escalation, alerts, T),
        "alerts": alerts,
        "additional_examinations": recommended_examinations(state, escalation, differential,
                                                            scenario),
        # retrieved is deliberately not forwarded: the protocol is selected by
        # topic, not ranked among the evidence passages.
        "therapeutic": therapeutic_considerations(state, scenario),
        "thresholds_version": T.get("version"),
    }
