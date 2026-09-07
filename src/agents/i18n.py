"""French rendering of the computed assessment.

A French-speaking emergency physician cannot act on an English alert, so leaving the clinical
output in English was not a defensible compromise -- it made the safety information the system
exists to surface unreadable to the people it is for.

What this module is NOT is a translator running over prose. Every string here is composed from
the SAME structured fields the English string was composed from: an alert carries the
measurement, the value and the threshold it crossed, so the French sentence is generated from
those numbers rather than from the English sentence. Where the source is prose from a closed
set -- the seven escalation triggers, the four examination reasons -- the English form is
matched exactly and its parameters extracted, and `untranslated()` reports anything that did
not match. A test walks every trigger and reason the pipeline can emit and fails if one has no
French rendering, so a new clinical message cannot reach a French screen in English by
accident. Silence is the failure mode this whole project is built to avoid.

The English record stays canonical. It is what the tests assert, what the regression archives
under a content hash, and what the report is built from. French is a rendering of the same
computed state, never a second derivation of it -- if the two could disagree, the record would
have two versions and no way to say which was shown.

Two things are deliberately NOT translated, and the interface says so in French:

  - Corpus passages. They are verbatim quotations from published papers, reproduced with their
    attribution. A translated quotation is no longer a quotation, and it would carry a citation
    to a source that never said it.
  - Model finding labels ("b lines", "consolidation"). They name the class the network was
    trained on, and the label is the identifier of a model output, not a phrase to be reworded.
    The French text around them explains what they are.
"""
from __future__ import annotations

import re
from typing import Any

FR = "fr"
_UNTRANSLATED: list[str] = []


def _miss(text: str) -> str:
    """Record a string that reached a French screen without a French rendering."""
    if text not in _UNTRANSLATED:
        _UNTRANSLATED.append(text)
    return text


def untranslated() -> list[str]:
    """Everything seen that had no French form. Empty is the property under test."""
    return list(_UNTRANSLATED)


def reset_untranslated() -> None:
    _UNTRANSLATED.clear()


# ---------------------------------------------------------------------------- vocabulary
SEVERITY = {"HIGH": "ÉLEVÉE", "MODERATE": "MODÉRÉE", "LOW": "FAIBLE"}
PRIORITY = {"HIGH": "HAUTE", "MODERATE": "MOYENNE", "LOW": "BASSE"}
LEVEL = {"CRITICAL": "CRITIQUE", "WARNING": "AVERTISSEMENT"}
FLAG = {"high": "élevé", "low": "bas", "normal": "normal", "not measured": "non mesuré"}
GRADE = {"GOOD": "BONNE", "ADEQUATE": "ACCEPTABLE", "POOR": "MAUVAISE"}

ORGAN = {"lung": "poumon", "heart": "cœur", "gallbladder": "vésicule biliaire",
         "fast": "FAST", "vascular": "vasculaire"}

# The measurements, by their canonical key. These are what a French chart calls them.
MEASURE = {
    "hr": "fréquence cardiaque", "sbp": "pression artérielle systolique",
    "rr": "fréquence respiratoire", "spo2": "saturation en oxygène",
    "temp": "température", "troponin": "troponine", "bnp": "BNP",
    "d_dimer": "D-dimères", "lactate": "lactate", "crp": "CRP",
    "wbc": "leucocytes", "creatinine": "créatinine", "ph": "pH",
}

# Alert type keys. These are the direction-specific labels from thresholds.json, upper-cased
# with underscores -- stable machine identifiers rather than prose. Every label in that file
# appears here, and a test walks the file to prove it: a threshold given a new label would
# otherwise show a French clinician an English word in the place a doctor reads first.
ALERT_TYPE = {
    "AGENT_CONFLICT": "DÉSACCORD ENTRE AGENTS",
    "POOR_CASE_QUALITY": "QUALITÉ DU DOSSIER INSUFFISANTE",
    "POSITIVE_IMAGING_KEY_LABS_ABSENT": "IMAGERIE POSITIVE, BILAN CLÉ ABSENT",
    "MODALITY_NOT_ASSESSED": "MODALITÉ NON RÉALISÉE",
    "TACHYCARDIA": "TACHYCARDIE", "SEVERE_TACHYCARDIA": "TACHYCARDIE SÉVÈRE",
    "BRADYCARDIA": "BRADYCARDIE", "SEVERE_BRADYCARDIA": "BRADYCARDIE SÉVÈRE",
    "HYPOTENSION": "HYPOTENSION", "LOW_BLOOD_PRESSURE": "PRESSION ARTÉRIELLE BASSE",
    "HYPERTENSION": "HYPERTENSION", "SEVERE_HYPERTENSION": "HYPERTENSION SÉVÈRE",
    "TACHYPNOEA": "TACHYPNÉE", "SEVERE_TACHYPNOEA": "TACHYPNÉE SÉVÈRE",
    "BRADYPNOEA": "BRADYPNÉE",
    "HYPOXAEMIA": "HYPOXÉMIE", "SEVERE_HYPOXAEMIA": "HYPOXÉMIE SÉVÈRE",
    "FEVER": "FIÈVRE", "HIGH_FEVER": "FIÈVRE ÉLEVÉE", "HYPOTHERMIA": "HYPOTHERMIE",
    "ACIDAEMIA": "ACIDÉMIE", "SIGNIFICANT_ACIDAEMIA": "ACIDÉMIE SIGNIFICATIVE",
    "SIGNIFICANT_ALKALAEMIA": "ALCALÉMIE SIGNIFICATIVE",
    "RAISED_TROPONIN": "TROPONINE ÉLEVÉE",
    "MARKEDLY_RAISED_TROPONIN": "TROPONINE FRANCHEMENT ÉLEVÉE",
    "RAISED_LACTATE": "LACTATE ÉLEVÉ",
    "MARKED_HYPERLACTATAEMIA": "HYPERLACTATÉMIE MARQUÉE",
    "RAISED_BNP": "BNP ÉLEVÉ", "RAISED_D_DIMER": "D-DIMÈRES ÉLEVÉS",
    "RAISED_CRP": "CRP ÉLEVÉE", "RAISED_WBC": "LEUCOCYTOSE",
    "LOW_WBC": "LEUCOPÉNIE", "RAISED_CREATININE": "CRÉATININE ÉLEVÉE",
}

SCENARIO_LABEL = {
    "Acute dyspnoea": "Dyspnée aiguë",
    "Undifferentiated shock": "État de choc indifférencié",
    "Trauma (FAST / E-FAST)": "Traumatisme (FAST / E-FAST)",
    "Cardiorespiratory arrest": "Arrêt cardiorespiratoire",
    "Unclassified": "Non classé",
}


def measure(name: str) -> str:
    return MEASURE.get(name, name)


def organ(name: str) -> str:
    return ORGAN.get(str(name).lower(), name)


def organs(names: list[str]) -> str:
    return ", ".join(organ(o) for o in names)


def severity(level: str) -> str:
    return SEVERITY.get(level, level)


def alert_type(kind: str) -> str:
    # Records a miss like everything else. It did not, and two labels -- SEVERE_HYPOXAEMIA and
    # LOW_BLOOD_PRESSURE -- passed through in English while the coverage test reported full
    # coverage. A check that exempts one field is a check that certifies the gap.
    if kind in ALERT_TYPE:
        return ALERT_TYPE[kind]
    return _miss(kind)


# -------------------------------------------------------------------------------- alerts
def alert(a: dict[str, Any]) -> str:
    """Compose the alert in French from the fields it fired on.

    A physiological alert carries the measurement, its value and the bound it crossed, so this
    is a second rendering of the same numbers -- not a translation of the English sentence.
    """
    kind = a.get("type", "")
    # No `level` here. It was computed and never used -- and its fallback returned the English
    # severity without recording a miss, which is the exact shape of the bug fixed in
    # alert_type() above. A dead line that would have failed silently once someone used it.

    if "measurement" in a and "threshold" in a:
        name = measure(a["measurement"])
        unit = _unit_from(a.get("message", ""), a["measurement"])
        below = " below " in f" {a.get('message', '')} "
        sense = "en dessous de" if below else "au-dessus de"
        seuil = "critique" if a.get("severity") == "CRITICAL" else "d'avertissement"
        return (f"{name} à {_num(a['value'])}{unit} — {sense} la borne {seuil} configurée "
                f"de {_num(a['threshold'])}{unit}")

    if kind == "MODALITY_NOT_ASSESSED":
        return (f"attendu mais jamais réalisé : {organs(a.get('modalities') or [])} — une "
                f"lacune du dossier, pas un résultat négatif")

    if kind == "POSITIVE_IMAGING_KEY_LABS_ABSENT":
        return ("imagerie positive alors que " + ", ".join(measure(m) for m in
                                                           (a.get("missing") or []))
                + " n'a jamais été mesuré")

    if kind == "POOR_CASE_QUALITY":
        return ("le dossier est trop incomplet ou trop incohérent pour que ses conclusions "
                "aient leur poids habituel")

    if kind == "AGENT_CONFLICT":
        detail = str(a.get("message", "")).split(":", 1)[-1].strip()
        return ("les agents sont en désaccord et le système ne peut pas déterminer lequel a "
                f"raison : {conflict(detail)}")

    return _miss(a.get("message", ""))


def _num(x: Any) -> str:
    """French writes the decimal comma. 38.6 is read as 386 by a reader who does not expect
    the point, and a temperature is exactly where that matters."""
    try:
        s = f"{float(x):g}"
    except (TypeError, ValueError):
        return str(x)
    return s.replace(".", ",")


def _unit_from(message: str, name: str) -> str:
    m = re.search(rf"{re.escape(name)}\s+[-\d.,]+\s+(\S+)", message)
    unit = m.group(1) if m else ""
    if unit in ("is", ""):
        return ""
    # French puts a space before % and °C; the space is added once, here, not also by the
    # caller -- "88  %" is a typography bug a clinician reads as a broken number.
    return " " + {"C": "°C", "%": "%"}.get(unit, unit)


# ------------------------------------------------------------------------------ conflicts
def conflict(text: str) -> str:
    t = text.strip()
    urgency = {"low": "faible", "medium": "moyenne", "high": "élevée"}
    m = re.match(r"triage says (\w+) urgency but (\w+) reports '(.+)' at ([\d.]+)$", t)
    if m:
        return (f"le triage indique une urgence {urgency.get(m.group(1), m.group(1))} alors "
                f"que l'examen {organ(m.group(2))} rapporte « {m.group(3)} » à "
                f"{_num(m.group(4))}")
    return _miss(t)


# -------------------------------------------------------------------------------- triggers
# The seven forms escalation_decision can emit. Matched exactly and re-composed; anything that
# does not match is recorded by untranslated() and caught by a test rather than shown silently.
def trigger(t: str) -> str:
    s = t.strip()

    m = re.match(r"agents disagree \((\d+) conflict\(s\)\)$", s)
    if m:
        return f"les agents sont en désaccord ({m.group(1)} conflit(s))"

    if s == "case quality POOR":
        return "qualité du dossier MAUVAISE"

    m = re.match(r"high-risk finding on non-strong evidence: (.+) \((\w+), ([\d.]+)\)$", s)
    if m:
        return (f"signe à haut risque sur un niveau de preuve non solide : {m.group(1)} "
                f"({m.group(2)}, {_num(m.group(3))})")

    m = re.match(r"high-risk finding below decisive confidence: (.+) \(([\d.]+)\)$", s)
    if m:
        return (f"signe à haut risque en dessous du seuil de confiance décisif : "
                f"{m.group(1)} ({_num(m.group(2))})")

    m = re.match(r"positive imaging with key lab\(s\) absent: (.+)$", s)
    if m:
        return ("imagerie positive avec bilan clé absent : "
                + ", ".join(measure(x.strip()) for x in m.group(1).split(",")))

    if s == "no positive finding, but the models cannot exclude disease":
        return "aucun signe positif, mais les modèles ne peuvent pas exclure une pathologie"

    m = re.match(r"organ\(s\) requested but not assessed: (.+)$", s)
    if m:
        return ("organe(s) demandé(s) mais non exploré(s) : "
                + organs([x.strip() for x in m.group(1).split(",")]))

    return _miss(s)


# ------------------------------------------------------------------------------ exams
# "échographie cœur" is not what a French clinician writes; the modality takes an adjective.
EXAM_ADJ = {"heart": "cardiaque", "lung": "pulmonaire", "gallbladder": "des voies biliaires",
            "fast": "FAST", "vascular": "vasculaire"}


def exam_name(name: str) -> str:
    m = re.match(r"(\w+) ultrasound$", str(name).strip())
    if m:
        key = m.group(1).lower()
        return f"échographie {EXAM_ADJ.get(key, organ(key))}"
    return measure(str(name).strip())


def exam_reason(reason: str) -> str:
    s = reason.strip()
    if s.startswith("expected for this presentation and never assessed"):
        return ("attendu pour cette présentation et jamais réalisé ; un examen non effectué "
                "est une information manquante, pas un résultat négatif")
    if s.startswith("named by an escalation trigger"):
        return ("désigné par un déclencheur d'escalade comme une valeur clé jamais mesurée")
    m = re.match(r"absent, and the most confirmatory test for '(.+)' which is on the "
                 r"differential$", s)
    if m:
        return (f"absent, et l'examen le plus confirmatoire pour « {m.group(1)} », qui figure "
                f"au diagnostic différentiel")
    if s == "never measured; absent rather than normal":
        return "jamais mesuré ; absent plutôt que normal"
    if s == "never recorded; absent rather than normal":
        return "jamais relevé ; absent plutôt que normal"
    return _miss(s)


# ------------------------------------------------------------------------------- limits
def limit(text: str) -> str:
    s = text.strip()
    m = re.match(r"(\w+): model has no healthy class -- .*", s)
    if m:
        return (f"{organ(m.group(1))} : le modèle n'a pas de classe « sain » — un résultat "
                f"est un choix parmi des pathologies et n'exclut jamais une maladie")
    m = re.match(r"(\w+): pneumothorax is not modelled and cannot be excluded$", s)
    if m:
        return (f"{organ(m.group(1))} : le pneumothorax n'est pas modélisé et ne peut pas "
                f"être exclu")
    m = re.match(r"(\w+): requested but never assessed$", s)
    if m:
        return f"{organ(m.group(1))} : demandé mais jamais réalisé"
    return _miss(s)


def conclusion(report: dict[str, Any]) -> str:
    """The conclusion paragraph, assembled in French by the SAME rules as the English.

    This is the line a busy reader acts on. It is built by fixed rules from the report rather
    than generated, precisely so that it cannot say something the rest of the report does not
    -- and the French is built by those same rules from those same fields, for the same
    reason. Translating the finished English sentence would put a second author between the
    record and the clinician.

    Model finding labels are left as they are. "b lines" is the name of the class the network
    was trained on, not a phrase; the sentence around it carries the meaning.
    """
    sev = report["decision_support"]["severity"]
    esc = report["safety"]["escalation"] or {}
    p = report["pocus"]
    parts: list[str] = []

    parts.append(f"Présentation : {scenario(report['context'].get('scenario', 'Unclassified'))}"
                 f", évaluée à une sévérité {severity(sev['severity'])}.")

    if p["detected"]:
        found = ", ".join(f"{d['finding']} ({organ(d['organ'])}, {_num(d['confidence'])})"
                          for d in p["detected"])
        parts.append(f"L'échographie au lit du patient a détecté : {found}.")
    elif p["screened_not_detected"]:
        parts.append(f"L'échographie a recherché {len(p['screened_not_detected'])} signe(s) "
                     f"et n'en a détecté aucun ; cela n'exclut pas une pathologie que les "
                     f"modèles ne représentent pas.")
    else:
        parts.append("Aucun signe échographique n'est enregistré.")

    if p["not_assessed"]:
        parts.append(f"NON EXPLORÉ : {organs(p['not_assessed'])} — attendu pour cette "
                     f"présentation et jamais examiné.")

    crit = [a for a in report["decision_support"]["alerts"] if a["severity"] == "CRITICAL"]
    if crit:
        parts.append(f"{len(crit)} alerte(s) critique(s) : "
                     + "; ".join(alert(a) for a in crit) + ".")

    if report["reasoning"]["withheld"]:
        parts.append("Le diagnostic différentiel a été RETENU : le raisonnement généré n'a "
                     "pas passé la validation contre le dossier.")
    elif report["reasoning"]["differential"]:
        top = report["reasoning"]["differential"][0]
        band = {"high": "élevée", "moderate": "modérée", "low": "faible"}
        parts.append(f"Hypothèse principale : {top.get('diagnosis')} (vraisemblance "
                     f"{band.get(str(top.get('likelihood')).lower(), top.get('likelihood'))}).")

    if esc.get("escalate"):
        parts.append(f"ESCALADE — {len(esc.get('triggers') or [])} déclencheur(s) ; "
                     f"orientation {esc.get('route')}.")
    else:
        parts.append("Aucun déclencheur d'escalade ; le cas peut être traité directement.")

    high = [r for r in report["decision_support"]["additional_examinations"]
            if r["priority"] == "HIGH"]
    if high:
        parts.append("Le plus informatif ensuite : "
                     + ", ".join(exam_name(r["exam"]) for r in high[:3]) + ".")

    return " ".join(parts)


def severity_reason(text: str) -> str:
    """The six forms `severity_level` can give for why a case is graded as it is.

    Matched exactly and re-composed, like the triggers, with `untranslated()` reporting
    anything new. These appear in the answer to "pourquoi cette sévérité ?", which is the
    question a clinician asks when they do not yet believe the number.
    """
    s = text.strip()

    if s == "triage assessed the patient as high urgency":
        return "le triage a classé le patient en urgence élevée"
    if s == "triage assessed the patient as medium urgency":
        return "le triage a classé le patient en urgence moyenne"
    if s == "no escalation trigger, no critical value, and no positive finding":
        return "aucun déclencheur d'escalade, aucune valeur critique et aucun signe positif"

    m = re.match(r"the escalation policy fired \((\d+) trigger\(s\)\)$", s)
    if m:
        return f"la politique d'escalade s'est activée ({m.group(1)} déclencheur(s))"

    m = re.match(r"(\d+) critical alert\(s\): (.+)$", s)
    if m:
        kinds = ", ".join(alert_type(k.strip()) for k in m.group(2).split(","))
        return f"{m.group(1)} alerte(s) critique(s) : {kinds}"

    m = re.match(r"(\d+) positive imaging finding\(s\) without an? (.+)$", s)
    if m:
        return (f"{m.group(1)} signe(s) d'imagerie positif(s) sans {m.group(2)}"
                .replace("escalation trigger", "déclencheur d'escalade"))

    m = re.match(r"(\d+) warning-level alert\(s\)$", s)
    if m:
        return f"{m.group(1)} alerte(s) de niveau avertissement"

    return _miss(s)


def scenario(label: str) -> str:
    # Not `.get(label, _miss(label))`: Python evaluates a default eagerly, so that recorded
    # every scenario as untranslated even when it had a French form. The coverage test caught
    # it, which is the reason the coverage test exists.
    if label in SCENARIO_LABEL:
        return SCENARIO_LABEL[label]
    return _miss(label)
