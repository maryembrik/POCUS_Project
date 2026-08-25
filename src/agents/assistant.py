"""The clinical assistant's question router.

This answers a typed question about one encounter. It is NOT a language model and does not
compose prose: it reads the question for what it is asking about and replies with what the
computed assessment actually holds. Every sentence it returns is quoted or assembled from the
clinical state, the escalation triggers, the alerts, the retrieved corpus or the validated
differential.

That boundary is the point. A fluent paragraph generated here would be a clinical opinion with
no evidence behind it, produced after the validator has finished and therefore checked by
nothing -- the same failure the validator exists to prevent, committed by the presentation
layer instead of by the model. When a question cannot be answered from the record, it says so
and names what it can answer instead.

It lives here rather than in an interface because there are two interfaces, and a question
answered differently depending on which front end asked it would be indefensible.
"""
from __future__ import annotations

import re
from typing import Any

from .clinical.clinical_state import LAB_REFERENCE, VITAL_REFERENCE, build_evidence

# The ways a clinician actually writes each measurement, beyond its canonical key.
ALIASES: dict[str, set[str]] = {
    "spo2": {"oxygen", "saturation", "sats", "o2", "spo2", "sat"},
    "hr": {"heart rate", "pulse", "hr"},
    "sbp": {"blood pressure", "systolic", "sbp", "bp"},
    "rr": {"respiratory rate", "breathing rate", "rr"},
    "temp": {"temperature", "temp"},
    "d_dimer": {"d dimer", "d-dimer", "ddimer"},
    "bnp": {"bnp", "natriuretic"},
    "wbc": {"wbc", "white cells", "white count"},
    "crp": {"crp", "c reactive protein"},
    "ph": {"ph", "acid base"},
    "troponin": {"troponin"},
    "lactate": {"lactate"},
    "creatinine": {"creatinine"},
}

CAPABILITIES = (
    "I can answer, from what was actually computed: any named vital or laboratory value and "
    "whether it was measured at all; what is missing; the alerts and the bound each crossed; "
    "the severity and why; the escalation triggers; what POCUS saw and what it cannot exclude; "
    "the differential and the evidence cited for and against each entry; the retrieved "
    "sources; the scenario routing; case quality; calibration and thresholds; and the stage "
    "timings."
)


def _measurement(key: str, question: str) -> bool:
    words = ALIASES.get(key, set()) | {key, key.replace("_", " ")}
    return any(f" {w} " in question or f" {w}?" in question or f" {w}," in question
               or question.endswith(f" {w}") for w in words)


def answer(question: str, a: dict[str, Any]) -> str:
    """Answer `question` from the analysis dict produced by the pipeline.

    `a` carries: state, esc, support, hits, result, report, marks, origin.
    """
    q = " " + str(question).lower().strip() + " "
    state, sup, esc = a["state"], a["support"], a["esc"]
    rep, hits = a["report"], a["hits"]
    diff = (a["result"].get("differential") or {}).get("differential") or []
    gone = state["missing"]["labs"] + state["missing"]["vitals"]
    det = [f for f in state["imaging"]["findings"] if f["detected"]]

    def has(*words: str) -> bool:
        return any(w in q for w in words)

    # ---- a specific measurement, by any of the names a clinician uses ----------------
    for key in list(LAB_REFERENCE) + list(VITAL_REFERENCE):
        if not _measurement(key, q):
            continue
        vitals = key in VITAL_REFERENCE
        ref = VITAL_REFERENCE[key] if vitals else LAB_REFERENCE[key]
        entry = (state.get("vitals") if vitals else state.get("labs") or {}).get(key)
        if entry:
            lo, hi = ref.get("normal_min", "—"), ref.get("normal_max", "—")
            eid = next((e["id"] for e in build_evidence(state)
                        if key in e["text"].lower()), None)
            return (f"{key} is {entry['value']:g} {entry['unit']} — flagged {entry['flag']}. "
                    f"Reference {lo}–{hi} {ref.get('unit', '')}".strip()
                    + (f". Citable as {eid}." if eid else "."))
        return (f"{key} was NOT measured for this patient. That is not the same as normal: it "
                f"has no evidence identifier, so nothing in the assessment can argue for or "
                f"against a diagnosis using it. It appears in the missing-information list, "
                f"and in the recommended examinations if it bears on this presentation.")

    # ---- intents ---------------------------------------------------------------------
    if has("treat", "therapy", "therapeutic", "manage", "give ", "drug", "dose",
           "medication", "prescri", "fluid", "antibiotic"):
        th = sup["therapeutic"]
        if th["considerations"]:
            return ("A protocol-backed consideration is available: "
                    + "; ".join(f"{c['consideration']} (basis: {c['basis']}, {c['passage']})"
                                for c in th["considerations"])
                    + ". This is decision support, not a treatment instruction.")
        return (f"No therapeutic recommendation for this case. {th['status']}. Therapeutic "
                f"suggestions are gated behind a sourced, citable protocol; with none "
                f"retrieved the system produces nothing rather than drawing on the language "
                f"model's own training knowledge. It is decision support and does not instruct "
                f"treatment.")

    if has("prognos", "survive", "going to be ok", "going to be okay", "will she", "will he",
           "will they", "outcome", "die", "mortality", "chance of", "how likely is he",
           "how likely is she", "recover"):
        # A prognosis is a prediction about a patient's future, and nothing in this system
        # produces one. Every figure it holds describes the record as it stands: what was
        # measured, what was seen, what is absent, how urgent the fixed rules judge it. Reaching
        # into the corpus for something adjacent would dress a keyword match as a forecast.
        return (f"This system does not predict outcomes, and I will not imply one. It was never "
                f"validated against what happened to any patient — no figure in it describes a "
                f"future.\n\nWhat it does say about this encounter, now: severity "
                f"{sup['severity']['severity']}, {len(sup['alerts'])} alert(s), "
                f"{'escalation required' if esc['escalate'] else 'no escalation'}"
                + (f", and {len(gone)} value(s) never measured" if gone else "")
                + ". The clinician makes the prognosis; this is decision support.")

    if has("suggest", "recommend", "next", "investigate", "order", "which test", "what test",
           "what should i do", "work up", "workup", "plan"):
        r = sup["additional_examinations"]
        return ("Recommended, in priority order: "
                + "; ".join(f"{x['exam']} ({x['priority']}) — {x['reason']}" for x in r[:5])
                ) if r else "The record is complete for this presentation."

    if has("miss", "absent", "not measured", "unavailable", "don't have", "do not have"):
        return ("Not measured for this patient: " + ", ".join(gone) + ". These have no "
                "evidence identifier, so the reasoning layer cannot cite them for or against "
                "any diagnosis. Absent is not normal.") if gone else \
            "Every value in the reference set was measured."

    if has("why", "reason", "because", "justif", "how did you"):
        t = esc["triggers"]
        return (f"Severity is {sup['severity']['severity']} because: "
                + "; ".join(sup["severity"]["reasons"] or ["no reason recorded"])
                + ". Escalation triggers that fired: " + ("; ".join(t) if t else "none")
                + ". All of this is computed by fixed rules before the model runs.")

    if has("challenge", "wrong", "sure", "certain", "limitation", "limit"):
        lim = state["imaging"]["out_of_scope"]
        return ("What most limits this assessment: "
                + ("; ".join(lim) if lim else "no scope limit was declared")
                + (". Absent tests: " + ", ".join(gone) if gone else "")
                + f". Case quality is graded {(state.get('case_quality') or {}).get('grade')}."
                + (" The agents disagree: " + "; ".join(state["conflicts"])
                   if state.get("conflicts") else ""))

    if has("alert", "critical", "danger", "urgent", "red flag", "worry"):
        al = sup["alerts"]
        return ("Alerts, each naming the bound it crossed: "
                + "; ".join(f"[{x['severity']}] {x['message']}" for x in al)
                + f". Thresholds v{sup['thresholds_version']}.") if al else \
            ("No alert fired. No measured value crossed a configured bound and the record "
             "shows no structural gap.")

    if has("severity", "priority", "escalat", "how bad", "how serious", "admit", "icu",
           "dispos"):
        base = answer("why", a)
        if has("admit", "icu", "dispos"):
            base += (" Disposition is outside what this system decides: it produces a "
                     "severity, alerts and recommended examinations, not a bed decision.")
        return base

    if has("pocus", "ultrasound", "scan", "image", "b-line", "b line", "finding", "detect",
           "see", "saw"):
        return ("POCUS detected: "
                + ("; ".join(f"{f['label']} at {f['confidence']:.2f}" for f in det)
                   if det else "no finding above the tuned threshold")
                + ". Screened and not seen: "
                + (", ".join(f["label"] for f in state["imaging"]["findings"]
                             if not f["detected"]) or "none")
                + ". " + ("; ".join(state["imaging"]["out_of_scope"]) or ""))

    if has("differential", "diagnos", "cause", "what could", "what might", "possib"):
        if not diff:
            return ("No differential was produced for this encounter. "
                    + ("The model backend failed and the answer was withheld; the severity and "
                       "alerts beside it were computed before the model ran."
                       if a["origin"] == "failed" else
                       "Generating one needs a 4.9 GB model on a GPU, which is not loaded "
                       "here."))
        return ("Ranked: " + "; ".join(f"{i}. {d.get('diagnosis')} ({d.get('likelihood')})"
                                       for i, d in enumerate(diff, 1))
                + ". Offered for consideration — this is not a diagnosis.")

    if has("support", "evidence for", "argue for", "in favour", "in favor"):
        if not diff:
            return (f"No differential was produced, so there is nothing ranked to support. The "
                    f"deterministic layer did run: severity {sup['severity']['severity']}, "
                    f"{len(sup['alerts'])} alert(s).")
        d = diff[0]
        return (f"{d.get('diagnosis')} is ranked first. Evidence cited for it, by identifier: "
                + "; ".join(f"{i} — {t}" for i, t in zip(d.get("supporting_ids") or [],
                                                         d.get("supporting") or []))
                + ". Those identifiers come from the enumerated list built from this record, "
                  "so nothing outside the record can be cited.")

    if has("against", "exclude", "rule out", "contradict"):
        lim = state["imaging"]["out_of_scope"]
        against = "; ".join(
            f"{d.get('diagnosis')}: " + "; ".join(d.get("contradicting") or ["—"])
            for d in diff) or "nothing was cited against any entry"
        return (f"Cited against: {against}. What the imaging cannot exclude: "
                + ("; ".join(lim) if lim else "no scope limit was declared")
                + ". A finding the module does not model cannot be ruled out by it, whatever "
                  "else the scan shows.")

    if has("summar", "overview", "tell me about", "who is", "recap"):
        return rep["conclusion"]

    if has("confiden", "calibrat", "reliab", "trust", "accurate", "threshold", "cutoff"):
        bits = [f"Alert thresholds are v{sup['thresholds_version']}, configuration for a "
                f"prototype rather than a validated scoring system"]
        for lim in state["imaging"]["out_of_scope"]:
            bits.append(lim)
        bits.append("none of this has been validated against patient outcomes")
        return ". ".join(bits) + "."

    if has("conflict", "disagree"):
        c = state.get("conflicts") or []
        return ("The agents disagree: " + "; ".join(c) + ". The system surfaces the "
                "disagreement rather than resolving it.") if c else \
            "The agents do not disagree on this encounter."

    if has("protocol", "guideline", "source", "reference", "citation", "corpus", "retriev",
           "paper", "evidence base"):
        if not hits:
            return ("No passage cleared the 0.10 relevance floor for this encounter, so the "
                    "answer is not marked guideline-grounded. Below the floor the system "
                    "reports nothing rather than the best of a bad set.")
        return "Retrieved: " + "; ".join(
            f"[{h['n']}] {h['topic']} ({h['score']:.2f}) — {h['source']}" for h in hits)

    if has("scenario", "route", "routed", "pathway"):
        return (f"Routed as {sup['scenario']['label']}. Routing selects which modalities are "
                f"expected and which protocol topic is looked up; it is matched on the "
                f"presenting complaint by fixed cues, not by a model.")

    if has("quality", "complete", "enough"):
        cq = state.get("case_quality") or {}
        return (f"Case quality is graded {cq.get('grade')}. "
                + ("Reasons: " + "; ".join(cq.get("reasons") or [])
                   if cq.get("reasons") else "No quality concern was recorded."))

    if has("evidence", "cite", "identifier"):
        ev = build_evidence(state)
        used = {i for d in diff for i in (d.get("supporting_ids") or [])}
        return (f"{len(ev)} citable fact(s), {len(used)} cited by the model: "
                + "; ".join(f"{e['id']} {e['text']}" for e in ev))

    if has("how long", "how fast", "time", "latency", "speed"):
        return ("Stage timings for this encounter: "
                + "; ".join(f"{m['title']} +{m['t'] * 1000:.0f} ms" for m in a["marks"]))

    if has("vital", "observation"):
        v = state.get("vitals") or {}
        return ("Vitals recorded: " + "; ".join(
            f"{k} {e['value']:g} {e['unit']} ({e['flag']})" for k, e in v.items())
            + (f". Never measured: {', '.join(state['missing']['vitals'])}."
               if state["missing"]["vitals"] else "")) if v else \
            "No vital sign was recorded for this patient."

    if has("lab", "blood", "biolog"):
        lb = state.get("labs") or {}
        return ("Laboratory values: " + "; ".join(
            f"{k} {e['value']:g} {e['unit']} ({e['flag']})" for k, e in lb.items())
            + (f". Never measured: {', '.join(state['missing']['labs'])}."
               if state["missing"]["labs"] else "")) if lb else \
            ("No laboratory value was recorded. Not measured is not normal: "
             + ", ".join(state["missing"]["labs"]) + " all have no identifier.")

    # ---- a diagnosis named in the question --------------------------------------------
    for d in diff:
        nm = str(d.get("diagnosis", "")).lower()
        if nm and (nm in q or any(w in q for w in nm.split() if len(w) > 5)):
            return (f"{d.get('diagnosis')} — {d.get('likelihood')} likelihood. Cited for it: "
                    + "; ".join(d.get("supporting") or ["—"])
                    + ". Against: " + "; ".join(d.get("contradicting") or ["—"])
                    + ". Limits: " + "; ".join(d.get("limitations") or ["—"]) + ".")

    # ---- last resort: match the question against the record itself ---------------------
    # Rather than refuse on a phrasing the intents above did not anticipate, the words of the
    # question are matched against the evidence, the alerts, the triggers and the retrieved
    # passages. If anything in the record bears on what was asked, it is returned. This still
    # composes nothing -- every line is quoted from the encounter.
    stop = {"what", "the", "is", "a", "an", "of", "for", "and", "or", "to", "in", "on", "do",
            "does", "did", "you", "i", "me", "my", "this", "that", "with", "about", "can",
            "should", "would", "could", "are", "was", "were", "be", "it", "his", "her",
            "patient", "any", "how", "why", "when", "which", "who", "tell", "show", "give"}
    words = {w for w in re.findall(r"[a-z]{3,}", q) if w not in stop}
    if words:
        pool: list[tuple[int, str]] = []
        for e in build_evidence(state):
            hitn = sum(1 for w in words if w in e["text"].lower())
            if hitn:
                pool.append((hitn, f"{e['id']} — {e['text']}"))
        for x in sup["alerts"]:
            hitn = sum(1 for w in words if w in x["message"].lower())
            if hitn:
                pool.append((hitn, f"[{x['severity']}] {x['message']}"))
        for tg in esc["triggers"]:
            hitn = sum(1 for w in words if w in tg.lower())
            if hitn:
                pool.append((hitn, f"escalation trigger: {tg}"))
        for x in sup["additional_examinations"]:
            hitn = sum(1 for w in words if w in (x["exam"] + " " + x["reason"]).lower())
            if hitn:
                pool.append((hitn, f"recommended: {x['exam']} ({x['priority']}) — "
                                   f"{x['reason']}"))
        for h in hits:
            hitn = sum(1 for w in words if w in (h["topic"] + " " + h["text"]).lower())
            if hitn:
                pool.append((hitn, f"[{h['n']}] {h['topic']} — {h['text']} ({h['source']})"))
        for x in state["imaging"]["out_of_scope"]:
            hitn = sum(1 for w in words if w in x.lower())
            if hitn:
                pool.append((hitn, f"declared limit: {x}"))
        gonehit = [g for g in gone if any(w in g.lower() for w in words)]
        for g in gonehit:
            pool.append((2, f"{g} was never measured — absent, not normal, and it carries no "
                            f"evidence identifier"))

        if pool:
            pool.sort(key=lambda p: -p[0])
            seen, lines = set(), []
            for _, text in pool:
                if text not in seen:
                    seen.add(text)
                    lines.append(text)
                if len(lines) >= 6:
                    break
            return ("From this encounter's record, the parts that bear on what you asked:\n\n"
                    + "\n".join("• " + x for x in lines)
                    + "\n\nThat is what the record holds on it. I have not added anything to "
                      "it.")

    return ("Nothing in this encounter's record bears on that, and I will not compose a "
            "clinical opinion with no evidence behind it — that is the failure this system is "
            "built to prevent.\n\n" + CAPABILITIES
            + "\n\nAsk in those terms and you will get the record, not a guess.")
