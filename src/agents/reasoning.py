"""Clinical Reasoning Agent -- safety layer, prompt contract, output validation.

Deliberately contains no model. The escalation decision and the checks on what the LLM returns
are deterministic Python, so they behave identically whether the model behind them is
HuatuoGPT-o1-8B, an API model, or nothing at all.

The division of labour:

    rules  ->  when to escalate, what counts as absent, what the models cannot exclude
    LLM    ->  reason over the evidence and explain it

Asking a language model "patient has X, what should we do?" puts the safety-critical decision
inside the least predictable component. Here the model proposes a differential and its reasoning;
the escalation call is computed from the structured state before the model is ever consulted, and
the model's output is checked against that state afterwards.
"""

from __future__ import annotations

import json
import re
from typing import Any

# ---------------------------------------------------------------------------------------
# Escalation policy -- computed from the state, never delegated to the model
# ---------------------------------------------------------------------------------------
HIGH_RISK_TERMS = (
    "severe", "perforation", "gangrenous", "carcinoma", "pneumothorax",
    "tamponade", "dissection", "free fluid",
)


def escalation_decision(state: dict) -> dict[str, Any]:
    """Decide direct answer vs escalate to simulation, before the LLM runs.

    The architecture escalates when the case is ambiguous or high-stakes. Each trigger below is
    a condition the structured state already knows about, so none of it depends on the model
    having noticed anything.
    """
    triggers: list[str] = []
    q = state.get("case_quality") or {}
    findings = state["imaging"]["findings"]
    detected = [f for f in findings if f["detected"]]

    if state.get("conflicts"):
        triggers.append(f"agents disagree ({len(state['conflicts'])} conflict(s))")

    if q.get("grade") == "POOR":
        triggers.append("case quality POOR")

    for f in detected:
        if any(t in f["label"].lower() for t in HIGH_RISK_TERMS):
            if f["evidence"] != "strong":
                triggers.append(
                    f"high-risk finding on non-strong evidence: {f['label']} "
                    f"({f['evidence']}, {f['confidence']:.2f})")
            elif f["confidence"] < 0.80:
                triggers.append(
                    f"high-risk finding below decisive confidence: {f['label']} "
                    f"({f['confidence']:.2f})")

    missing_key = [l for l in ("troponin", "lactate") if l in state["missing"]["labs"]]
    if missing_key and detected:
        triggers.append(f"positive imaging with key lab(s) absent: {', '.join(missing_key)}")

    # A high-risk finding the imaging cannot exclude is not the same as its absence.
    if state["imaging"]["out_of_scope"] and not detected:
        triggers.append("no positive finding, but the models cannot exclude disease")

    # An organ that was requested and never assessed is not a negative result either.
    #
    # Deliberately NOT gated on the absence of other findings, unlike the scope trigger above.
    # An earlier version was, and it produced the wrong answer on a case with chest pain where
    # the lung found B-lines and the heart was never scanned: the positive lung result silenced
    # the unexamined heart, and the system answered directly on the strength of imaging that
    # never happened. A finding in one organ says nothing about an organ nobody looked at, and
    # a positive elsewhere raises the stakes of the gap rather than closing it.
    not_assessed = state["imaging"]["organs_not_assessed"]
    if not_assessed:
        triggers.append(f"organ(s) requested but not assessed: {', '.join(not_assessed)}")

    return {
        "escalate": bool(triggers),
        "triggers": triggers,
        "route": "simulation" if triggers else "direct",
    }


# ---------------------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------------------
# Prompt v2. v1 kept the safety rules only, and the model obeyed them: it fabricated nothing.
# What it did badly was reason -- it cited one finding out of four available, merged two
# diagnoses into a single entry, and rated a specific diagnosis "high" on one sign with every
# key lab absent. Rules 3 to 7 below target those failures specifically. The safety rules are
# unchanged, so any difference between v1 and v2 is attributable to the reasoning instructions.
SYSTEM_PROMPT = """You are a clinical reasoning assistant supporting an emergency physician.

You receive a STRUCTURED CLINICAL STATE assembled from a triage model, ultrasound models, vitals
and laboratory results. You never see images. Reason only over what you are given.

HARD RULES. These are not style preferences.

1. Use ONLY findings, vitals and labs that appear in the state. Never introduce a test, finding
   or measurement that is not listed.

2. Anything under "NOT MEASURED" is ABSENT, not normal. You may not use an absent test to argue
   for or against any diagnosis. Put it in missing_information instead.

3. ONE DIAGNOSIS PER ENTRY. Never write "A or B" in a diagnosis field. If two conditions are
   both plausible they are two entries, ranked, each with its own evidence. Merging them is not
   a differential -- it is a refusal to rank, and ranking is the entire point.

4. FINDING CONFIDENCE IS NOT DIAGNOSTIC CONFIDENCE. "b lines (0.86)" means the imaging model is
   confident that the sign is present. It does NOT mean any diagnosis is 86% likely. A single
   strong sign with the key laboratory values absent supports "moderate" at best, never "high".
   Reserve "high" for cases where several independent lines of evidence agree.

5. ACCOUNT FOR EVERY ABNORMAL VALUE. Before answering, go through the state line by line: each
   abnormal vital, each detected finding, each NOT-detected finding, each LIMIT. Every value
   marked HIGH or LOW must either appear in the "supporting" or "contradicting" array of at
   least one diagnosis, or be named in "uncertainty" as not bearing on the assessment. Silence
   is not permitted: a reader cannot tell an abnormal value you considered and dismissed from
   one you failed to notice. Citing one finding when four were available is an incomplete
   answer, even if the one you cited is right.

6. NOT-DETECTED IS EVIDENCE; NOT-ASSESSED IS NOT. A finding the model screened for and did not
   see may appear in "contradicting". A finding never assessed may not appear anywhere except
   missing_information -- and it MUST appear there. If the state lists an organ as "not
   assessed", naming that scan is usually the most useful next step available, so it belongs
   in missing_information ahead of any laboratory test.

7. RESPECT THE STATED LIMITS. Each LIMIT line bounds what you may conclude:
     - "no healthy class" means a finding never establishes that the patient is healthy;
     - a finding that is "not modelled" cannot be excluded, whatever else you see.
   Restate the limits bearing on each diagnosis in that entry's "limitations" array.

8. Respect the evidence grade on each finding. "experimental" means the number is uncalibrated
   and is not a probability; treat it as weak support and say so.

9. You are decision support, not a diagnosis. Recommend next steps; do not instruct treatment as
   if the decision were yours.

FORMAT. missing_information takes ONE test per array element: write ["troponin", "lactate"],
never ["troponin, lactate"]. Rank the differential most likely first.

Return ONLY valid JSON matching this shape, with no prose outside it:

{
  "differential": [
    {"diagnosis": "a single named condition",
     "likelihood": "high|moderate|low",
     "supporting": ["every finding, vital or lab FROM THE STATE that argues for this"],
     "contradicting": ["evidence FROM THE STATE that argues against this"],
     "limitations": ["what the models cannot exclude, for this diagnosis"]}
  ],
  "missing_information": ["one test per element, most decisive first"],
  "uncertainty": "one sentence on what most limits confidence here",
  "recommended_next_step": "the single most useful next action"
}"""


def build_prompt(state_text: str, retrieved: list[dict] | None = None) -> dict[str, str]:
    """System + user prompt. `retrieved` is the RAG payload; absent for now, and its absence is
    stated rather than hidden so the model does not imply guideline grounding it never had."""
    parts = [state_text]

    if retrieved:
        parts.append("\n\nRETRIEVED EVIDENCE")
        for i, doc in enumerate(retrieved, 1):
            parts.append(f"[{i}] {doc.get('source', 'unknown')}: {doc['text']}")
        parts.append("\nCite passages as [1], [2] where you use them.")
    else:
        parts.append("\n\nNO RETRIEVED EVIDENCE — reason from the state alone and do not claim "
                     "guideline support.")

    return {"system": SYSTEM_PROMPT, "user": "\n".join(parts)}


# ---------------------------------------------------------------------------------------
# Output validation -- the hallucination guard
# ---------------------------------------------------------------------------------------
def _norm(s: str) -> str:
    """Underscores and hyphens are punctuation, not meaning: the state stores `d_dimer` and a
    model writes `D-dimer`. Comparing the raw strings makes the two different terms."""
    return re.sub(r"[_\-]+", " ", str(s).lower()).strip()


_TOKEN = re.compile(r"[a-z0-9.]+")


def _tokens(s: str) -> set[str]:
    return {t.strip(".") for t in _TOKEN.findall(_norm(s))} - {""}


def _expand(name: str) -> set[str]:
    """A canonical vital name plus the ways a model actually writes it."""
    return {_norm(name)} | {_norm(a) for a in VITAL_ALIASES.get(name, ())}


def _vocabulary(state: dict) -> list[set[str]]:
    """Every term the model may cite, as a set of tokens per term.

    Matching is by token subset, not substring, and the reason is a false accusation this
    guard actually made: the state said `severe dysfunction` and the model wrote
    `severe heart dysfunction`. Substring matching called that a fabrication and withheld a
    correct answer on the conflict case -- the very case the architecture exists for. A guard
    that blocks correct answers gets switched off, so it must tolerate paraphrase.

    Subset matching keeps it honest in the other direction too. A term like `low urgency`
    requires BOTH tokens, so an unrelated sentence containing the word `low` no longer passes
    as grounded.
    """
    terms: list[set[str]] = []

    def add(s) -> None:
        t = _tokens(s)
        if t:
            terms.append(t)

    for f in state["imaging"]["findings"]:
        add(f["label"])
        if f.get("group"):
            add(f["group"])
    for k in state.get("labs", {}):
        add(k)
    for k in state.get("vitals", {}):
        for alias in _expand(k):
            add(alias)
    t = state.get("triage")
    if t:
        add("triage")
        add("urgency")
        if t.get("urgency"):
            add(f"{t['urgency']} urgency")
    return terms


def _grounded(citation: str, terms: list[set[str]]) -> bool:
    cit = _tokens(citation)
    return any(term <= cit for term in terms)


def parse_llm_json(raw: str) -> dict:
    """Extract the JSON object from a model reply that may be wrapped in prose or fences."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fence:
        text = fence.group(1)
    else:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in model output")
        text = text[start:end + 1]
    return json.loads(text)


def validate_llm_output(out: dict, state: dict) -> list[str]:
    """Check the model's answer against the state it was given.

    The important check is the last one: any cited term absent from the state is a fabrication,
    and a fabricated lab in a differential is exactly the failure that makes a system like this
    unsafe. It is caught here rather than trusted away.
    """
    errs: list[str] = []

    for key in ("differential", "missing_information", "uncertainty", "recommended_next_step"):
        if key not in out:
            errs.append(f"missing key: {key}")
    if "differential" not in out:
        return errs
    if not isinstance(out["differential"], list) or not out["differential"]:
        errs.append("differential must be a non-empty list")
        return errs

    terms = _vocabulary(state)
    absent: list[set[str]] = []
    for l in state["missing"]["labs"]:
        absent.append(_tokens(l))
    for v in state["missing"]["vitals"]:
        for alias in _expand(v):
            if _tokens(alias):
                absent.append(_tokens(alias))

    for i, d in enumerate(out["differential"]):
        if "diagnosis" not in d:
            errs.append(f"differential[{i}] has no diagnosis")
            continue
        if d.get("likelihood") not in ("high", "moderate", "low"):
            errs.append(f"differential[{i}] likelihood invalid: {d.get('likelihood')!r}")

        for field in ("supporting", "contradicting"):
            for cited in d.get(field, []):
                cit = _tokens(cited)
                if any(a <= cit for a in absent):
                    errs.append(
                        f"differential[{i}].{field} cites '{cited}', which was NOT MEASURED — "
                        "an absent test cannot support or contradict anything")
                elif not _grounded(cited, terms):
                    errs.append(
                        f"differential[{i}].{field} cites '{cited}', which does not appear "
                        "anywhere in the clinical state (possible fabrication)")
    return errs


# ---------------------------------------------------------------------------------------
# Revisable checks -- output that is grounded but poorly calibrated
#
# These are separated from validate_llm_output deliberately. A fabricated laboratory value is
# fatal: the answer is withheld. A likelihood the evidence does not support is a judgement the
# model can be asked to reconsider, and asking is better than silently rewriting its answer --
# a downgrade applied by Python would be presented to the clinician as the model's reasoning
# when it is not.
# ---------------------------------------------------------------------------------------

# Which absent test would most undermine a "high" rating for which diagnosis. Deliberately
# diagnosis-aware: forbidding "high" whenever ANY lab is missing would be too blunt, since a
# missing D-dimer says nothing about the confidence owed to a pneumothorax.
DECISIVE_TESTS: dict[tuple[str, ...], tuple[str, ...]] = {
    ("pulmonary oedema", "pulmonary edema", "heart failure", "cardiogenic",
     "congestion", "interstitial syndrome"): ("bnp", "troponin"),
    ("pulmonary embolism", "embolism", "thromboemb"): ("d_dimer",),
    ("myocardial", "coronary", "infarction", "ischaem", "ischem",
     "acute coronary"): ("troponin",),
    ("sepsis", "septic", "hypoperfusion", "shock"): ("lactate",),
    ("renal", "kidney", "aki"): ("creatinine",),
    ("pneumonia", "infection", "consolidation"): ("wbc", "crp"),
}

# How the model is likely to name each vital in prose. The measured value is also accepted,
# because "118 bpm" cites the heart rate as surely as the word "heart rate" does.
#
# Whole words only, not stems. Matching is by token, so a stem like "tachycard" never equals
# the token "tachycardia" and the alias silently stops working -- write out the forms.
VITAL_ALIASES: dict[str, tuple[str, ...]] = {
    "hr": ("hr", "heart rate", "pulse", "tachycardia", "tachycardic", "bradycardia"),
    "rr": ("rr", "respiratory rate", "respiration", "tachypnoea", "tachypnea",
           "tachypnoeic"),
    "sbp": ("sbp", "blood pressure", "systolic", "hypotension", "hypotensive"),
    "spo2": ("spo2", "o2", "oxygen", "saturation", "hypoxia", "hypoxic", "hypoxaemia",
             "hypoxemia", "desaturation"),
    "temp": ("temp", "temperature", "fever", "pyrexia", "febrile", "afebrile"),
}


def _decisive_for(diagnosis: str) -> set[str]:
    d = diagnosis.lower()
    out: set[str] = set()
    for keys, tests in DECISIVE_TESTS.items():
        if any(k in d for k in keys):
            out |= set(tests)
    return out


def check_confidence(out: dict, state: dict) -> list[str]:
    """A 'high' likelihood whose most confirmatory test was never obtained."""
    errs: list[str] = []
    missing = {str(l).lower() for l in state["missing"]["labs"]}
    for i, d in enumerate(out.get("differential", []) or []):
        if str(d.get("likelihood", "")).lower() != "high":
            continue
        absent = sorted(_decisive_for(str(d.get("diagnosis", ""))) & missing)
        if absent:
            errs.append(
                f"differential[{i}] {d.get('diagnosis')!r} is rated 'high' while "
                f"{', '.join(absent)} was never measured -- the test that would most confirm "
                f"it is absent, so the evidence supports 'moderate' at best")
    return errs


def check_evidence_coverage(out: dict, state: dict) -> list[str]:
    """Abnormal values present in the state that the answer used nowhere.

    Targets the observed failure of citing one finding when several were available. An
    abnormal value may legitimately be irrelevant -- but that has to be said, not left
    silent, or the reader cannot tell whether it was considered or overlooked.

    `missing_information` is excluded from the search on purpose. A model asked to account
    for SpO2 once satisfied the check by listing "spo2 (90.0, low)" as missing information --
    while SpO2 had in fact been measured. Scanning the whole object let a factual
    contradiction count as evidence of use.
    """
    considered = json.dumps({k: v for k, v in out.items()
                             if k != "missing_information"}).lower()
    errs: list[str] = []
    unused: list[str] = []

    for name, v in (state.get("vitals") or {}).items():
        if v.get("flag") not in ("high", "low"):
            continue
        aliases = VITAL_ALIASES.get(name, (name,))
        value = f"{v.get('value')}".rstrip("0").rstrip(".")
        if any(a in considered for a in aliases) or (value and value in considered):
            continue
        unused.append(f"{name} ({v.get('value')}, {v.get('flag')})")

    if unused:
        errs.append(
            f"abnormal value(s) not used and not explained: {', '.join(unused)} -- cite each "
            f"in a differential entry or say in 'uncertainty' why it does not bear on the "
            f"assessment")
    return errs


def check_missing_information_accuracy(out: dict, state: dict) -> list[str]:
    """A measured value listed as missing.

    Separated from coverage because it is a different kind of error: not an omission but a
    contradiction of the record, which misleads a reader about what was actually done.
    """
    errs: list[str] = []
    measured = {**(state.get("labs") or {}), **(state.get("vitals") or {})}
    for item in out.get("missing_information", []) or []:
        toks = _tokens(item)
        for name in measured:
            if _tokens(name) <= toks or any(_tokens(a) <= toks
                                            for a in VITAL_ALIASES.get(name, ())):
                errs.append(
                    f"missing_information lists {item!r}, but {name} WAS measured "
                    f"({measured[name].get('value')}) -- it is evidence, not a gap")
                break
    return errs


# Qualifiers that assert a direction. Applying one to a value the state flags as normal
# misreports the record, which is a different failure from inventing a value outright.
_QUALIFIERS = {
    "high": ("elevated", "raised", "increased", "high"),
    "low": ("reduced", "decreased", "depressed", "low"),
}


def check_value_qualifiers(out: dict, state: dict) -> list[str]:
    """A measured value described in a direction the state contradicts.

    Observed: troponin 5.0 against a reference of <=14 was cited as "elevated troponin".
    The value is real and the citation is grounded, so neither the fabrication check nor the
    absence check sees anything wrong -- but the reasoning rests on a misreading.
    """
    errs: list[str] = []
    measured = {**(state.get("labs") or {}), **(state.get("vitals") or {})}

    for i, d in enumerate(out.get("differential", []) or []):
        for field in ("supporting", "contradicting"):
            for cited in d.get(field, []) or []:
                toks = _tokens(cited)
                for name, entry in measured.items():
                    names = {name} | set(VITAL_ALIASES.get(name, ()))
                    if not any(_tokens(n) <= toks for n in names):
                        continue
                    flag = entry.get("flag", "normal")
                    for direction, words in _QUALIFIERS.items():
                        if direction == flag:
                            continue
                        if any(w in toks for w in words):
                            errs.append(
                                f"differential[{i}].{field} calls {name} "
                                f"{[w for w in words if w in toks][0]!r}, but the state "
                                f"records it as {flag} ({entry.get('value')})")
                            break
    return errs


# Verbs that instruct treatment rather than recommend a next step. The system is decision
# support: it may say which test would help, not which drug to give.
_TREATMENT_VERBS = ("initiate", "administer", "prescribe", "start ", "give ", "commence")
_TREATMENT_NOUNS = ("diuretic", "ace inhibitor", "antibiotic", "anticoagul", "thromboly",
                    "heparin", "furosemide", "aspirin", "morphine", "beta blocker")


def check_scope_of_advice(out: dict, state: dict) -> list[str]:
    """A recommendation that prescribes treatment instead of proposing an investigation."""
    text = _norm(out.get("recommended_next_step", ""))
    if not text:
        return []
    if any(n in text for n in _TREATMENT_NOUNS) or any(v in text for v in _TREATMENT_VERBS):
        if any(n in text for n in _TREATMENT_NOUNS):
            return [f"recommended_next_step instructs treatment ({out['recommended_next_step']!r}) "
                    f"-- this system recommends investigations, not therapy"]
    return []


# ---------------------------------------------------------------------------------------
# Which findings bear on which diagnoses
#
# Deliberately small, and deliberately not an expert system. It encodes only relationships
# that follow from what the imaging modules actually measure, and its purpose is to stop one
# specific misuse: treating the ABSENCE of an unrelated finding as evidence against a
# diagnosis. Observed directly -- "pleural thickening NOT detected" was offered as evidence
# against pulmonary embolism, which it is not.
#
# A finding not listed here is not judged. Silence means "no opinion", never "unrelated".
# ---------------------------------------------------------------------------------------
_DX = {
    "oedema": ("pulmonary oedema", "pulmonary edema", "heart failure", "cardiogenic",
               "congestion", "interstitial syndrome"),
    "pneumonia": ("pneumonia", "infection", "consolidation"),
    "pneumothorax": ("pneumothorax",),
    "embolism": ("pulmonary embolism", "embolism", "thromboemb"),
}

# finding -> {diagnosis group: role}. "supports" means presence argues for; "excludes" means
# presence argues against. Absence is handled separately below.
EVIDENCE_MAP: dict[str, dict[str, str]] = {
    "b lines": {"oedema": "supports", "pneumothorax": "excludes"},
    "consolidation": {"pneumonia": "supports"},
    "pleural effusion": {"oedema": "supports", "pneumonia": "supports"},
    "pleural thickening": {},          # bears on none of the diagnoses modelled here
}


def _dx_group(diagnosis: str) -> str | None:
    d = _norm(diagnosis)
    for group, keys in _DX.items():
        if any(k in d for k in keys):
            return group
    return None


def _finding_in(citation: str) -> str | None:
    cit = _tokens(citation)
    for name in EVIDENCE_MAP:
        if _tokens(name) <= cit:
            return name
    return None


def check_evidence_relationships(out: dict, state: dict) -> list[str]:
    """Findings cited against a diagnosis they have no bearing on.

    The model understands the output format better than it understands what each ultrasound
    sign means, which is the expected failure mode for a small local model. This check does
    not attempt to supply the medical reasoning -- it only refuses the clearest misuses.
    """
    errs: list[str] = []
    negatives = {_norm(f["label"]) for f in state["imaging"]["findings"]
                 if not f.get("detected", True)}

    for i, d in enumerate(out.get("differential", []) or []):
        group = _dx_group(str(d.get("diagnosis", "")))
        if group is None:
            continue                                   # unknown diagnosis: no opinion
        for cited in d.get("contradicting", []) or []:
            name = _finding_in(cited)
            if name is None:
                continue                               # unknown finding: no opinion
            roles = EVIDENCE_MAP[name]
            if group in roles:
                continue                               # a relationship exists
            if _norm(name) in negatives:
                errs.append(
                    f"differential[{i}] cites {cited!r} as evidence against "
                    f"{d.get('diagnosis')!r}, but the absence of {name} does not bear on that "
                    f"diagnosis -- a finding the model did not look for, or that is unrelated, "
                    f"is not an argument against it")
    return errs


# Phrases that assert more than decision support can. "to rule out X" claims a test will
# settle the question; the system proposes investigations, it does not adjudicate them.
_OVERCLAIM = ("rule out", "rules out", "ruled out", "confirm the diagnosis", "confirms the",
              "definitively", "exclude the possibility")


def check_recommendation_wording(out: dict, state: dict) -> list[str]:
    text = _norm(out.get("recommended_next_step", ""))
    hits = [p for p in _OVERCLAIM if p in text]
    if hits:
        return [f"recommended_next_step claims to {hits[0]!r} -- state what the test would "
                f"inform, not what it would settle"]
    return []


_CITATION = re.compile(r"\[(\d+)\]")


def check_citations(out: dict, retrieved: list[dict] | None) -> list[str]:
    """Citations must point at passages that were actually retrieved.

    Retrieval is a second route by which evidence enters the prompt, and it deserves the
    same suspicion as the model's own claims. `[2]` where only one passage was retrieved is
    a fabrication of the same kind as an invented laboratory value -- and a more persuasive
    one, because a bracketed number reads as provenance.
    """
    errs: list[str] = []
    hits = retrieved or []
    text = json.dumps(out)
    cited = {int(n) for n in _CITATION.findall(text)}

    for n in sorted(cited):
        if not 1 <= n <= len(hits):
            errs.append(
                f"answer cites [{n}], but {len(hits)} passage(s) were retrieved -- a "
                f"citation must point at retrieved evidence")

    # Claiming guideline support with nothing retrieved is the same failure without the
    # bracket. The prompt states plainly when retrieval returned nothing.
    if not hits:
        low = _norm(text)
        for phrase in ("according to the guideline", "guidelines recommend",
                       "as per the protocol", "the literature shows",
                       "evidence supports", "studies show"):
            if phrase in low:
                errs.append(
                    f"answer claims support from external evidence ({phrase!r}) while no "
                    f"passage was retrieved")
                break
    return errs


def check_unassessed_reported(out: dict, state: dict) -> list[str]:
    """An organ that was never scanned must appear in missing_information.

    Observed: the heart was not assessed, the model offered "Cardiac Event" as a diagnosis,
    and its missing_information listed laboratory tests only. That list is what a clinician
    reads to decide what to order next, so the single most useful action -- scan the heart --
    was the one item absent from it. The escalation policy already knows; the answer the
    clinician sees did not say.
    """
    not_assessed = state["imaging"].get("organs_not_assessed") or []
    if not not_assessed:
        return []
    listed = _norm(" ".join(str(i) for i in (out.get("missing_information") or [])))
    absent = [o for o in not_assessed if _norm(o) not in listed]
    if absent:
        return [f"organ(s) never assessed and not named in missing_information: "
                f"{', '.join(absent)} -- an unperformed scan is missing information, and the "
                f"list is what determines what gets ordered next"]
    return []


def check_atomicity(out: dict, state: dict) -> list[str]:
    """One test per element. A comma-joined string is a list of one to everything downstream."""
    bad = [i for i in (out.get("missing_information") or [])
           if isinstance(i, str) and ("," in i or " and " in i.lower())]
    if bad:
        return [f"missing_information contains combined entries {bad!r} -- one test per "
                f"array element"]
    return []


REVISION_PROMPT = """Your previous answer was rejected by the evidence checker.

{complaints}

Revise the answer to address every point above. Change only what the complaints require;
keep the rest of your reasoning. Return the same JSON shape and nothing else."""


def build_revision_prompt(previous: str, complaints: list[str]) -> str:
    bullets = "\n".join(f"  - {c}" for c in complaints)
    return (REVISION_PROMPT.format(complaints=bullets)
            + "\n\nYour previous answer:\n" + previous)


def _revisable(parsed: dict, state: dict,
               retrieved: list[dict] | None = None) -> tuple[list[str], list[str]]:
    """The two tiers of revisable fault: (unsound, untidy)."""
    unsound = (check_citations(parsed, retrieved)
               + check_confidence(parsed, state)
               + check_value_qualifiers(parsed, state)
               + check_missing_information_accuracy(parsed, state)
               + check_unassessed_reported(parsed, state)
               + check_evidence_relationships(parsed, state)
               + check_scope_of_advice(parsed, state))
    untidy = (check_evidence_coverage(parsed, state)
              + check_atomicity(parsed, state)
              + check_recommendation_wording(parsed, state))
    return unsound, untidy


def reason(state: dict, llm_fn=None, retrieved: list[dict] | None = None,
           max_revisions: int = 1) -> dict:
    """Full pass: escalation, prompt, model call, validation, at most one revision.

    `max_revisions` defaults to 1 on measurement, not preference. Across the five benchmark
    cases the first revision round fixed 3 of 4 complaints and the second fixed 0 of 3, so a
    second round costs a model call per case and buys nothing with this model.
    """
    """Full pass: escalation, prompt, model call, validation.

    `llm_fn(system, user) -> str`. With no model supplied this still returns the escalation
    decision and the prompt, which is what makes the safety layer testable on its own.
    """
    esc = escalation_decision(state)
    prompt = build_prompt_from_state(state, retrieved)

    from .retrieval import is_grounded, retrieval_note
    result: dict[str, Any] = {
        "encounter_id": state.get("encounter_id"),
        "retrieval": {"passages": len(retrieved or []),
                      "grounded": is_grounded(retrieved or []),
                      "note": retrieval_note(retrieved or [])},
        "case_quality": (state.get("case_quality") or {}).get("grade"),
        "escalation": esc,
        "prompt": prompt,
        "differential": None,
        "validation_errors": None,
    }
    if llm_fn is None:
        result["note"] = "no model supplied — escalation and prompt only"
        return result

    # A backend can fail for reasons that have nothing to do with the patient: the weights do
    # not load, the process runs out of memory, the request times out. The escalation decision
    # above was computed before the model ran and does not depend on it, so letting the
    # exception propagate would discard the one part of this output that is guaranteed
    # correct -- at exactly the moment a clinician is waiting for it.
    def _ask(system: str, user: str) -> tuple[str | None, list[str] | None]:
        try:
            return llm_fn(system, user), None
        except Exception as exc:                                    # noqa: BLE001
            return None, [f"model backend failed: {type(exc).__name__}: {exc}"]

    raw, backend_err = _ask(prompt["system"], prompt["user"])
    if backend_err:
        result["validation_errors"] = backend_err
        result["differential_withheld"] = True
        return result

    revisions: list[dict[str, Any]] = []

    for attempt in range(max_revisions + 1):
        try:
            parsed = parse_llm_json(raw)
        except (ValueError, json.JSONDecodeError) as e:
            result["validation_errors"] = [f"unparseable model output: {e}"]
            result["raw_output"] = raw
            result["differential_withheld"] = True
            result["revisions"] = revisions
            return result

        # Fatal: the answer cites something that is not in the state. No revision is offered,
        # because the failure is not a matter of degree.
        fatal = validate_llm_output(parsed, state)
        if fatal:
            # The outcome is already decided, but the remaining checks still run. Returning on
            # the first class of fault reported one fabrication while a misread value in the
            # same answer went unmentioned -- fine for the decision, useless for a benchmark
            # that is supposed to say what went wrong.
            result["differential"] = parsed
            result["validation_errors"] = fatal
            _u, _t = _revisable(parsed, state, retrieved)
            result["also_found"] = (_u + _t) or None
            result["differential_withheld"] = True
            result["revisions"] = revisions
            return result

        # Revisable failures, in two tiers. Both are sent back for revision; they differ only
        # in what happens if the model does not fix them.
        #
        #   unsound  -- the reasoning is corrupted: a misread value, a test claimed missing
        #               that was measured, an unsupported confidence, evidence cited against a
        #               diagnosis it has no bearing on, or treatment instructions. Withheld.
        #   untidy   -- the reasoning stands but the presentation is incomplete: an uncited
        #               abnormal value, a comma-joined list. Delivered with the warning
        #               attached.
        #
        # The split exists because withholding is expensive. An earlier version suppressed a
        # correct differential because the model joined a list with commas, and a guard that
        # blocks correct answers is one a clinician learns to switch off.
        unsound, untidy = _revisable(parsed, state, retrieved)
        soft = unsound + untidy

        if not soft:
            result["differential"] = parsed
            result["validation_errors"] = None
            result["revisions"] = revisions
            return result

        if attempt == max_revisions:
            # The model was told what was wrong and did not fix it. Python does not rewrite
            # the answer -- a likelihood edited in post would be read as the model's judgement
            # when it is not.
            result["differential"] = parsed
            result["revisions"] = revisions
            if unsound:
                result["validation_errors"] = unsound
                result["warnings"] = untidy or None
                result["differential_withheld"] = True
            else:
                result["validation_errors"] = None
                result["warnings"] = untidy
            return result

        # Send ONE complaint, not all of them. Measured on the same case with the same seed:
        # a request carrying a single complaint was fixed, and a request carrying three was
        # fixed in no respect at all -- the model changed an unrelated likelihood instead.
        # Bundling was meant to save a model call; it cost the revision entirely.
        #
        # Unsound faults come first because they are the ones that decide whether the answer
        # can be shown. Untidy faults left unfixed become warnings, so losing a round on them
        # costs nothing.
        focus = [(unsound or untidy)[0]]
        revisions.append({"attempt": attempt + 1, "complaints": focus,
                          "also_found": [c for c in soft if c not in focus] or None})
        raw, backend_err = _ask(prompt["system"], build_revision_prompt(raw, focus))
        if backend_err:
            result["validation_errors"] = backend_err
            result["differential_withheld"] = True
            result["revisions"] = revisions
            return result
    return result


def build_prompt_from_state(state: dict, retrieved: list[dict] | None = None) -> dict[str, str]:
    from .clinical_state import render_state
    return build_prompt(render_state(state), retrieved)
