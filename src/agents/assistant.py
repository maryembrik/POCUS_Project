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


def has_encounter(a: dict[str, Any]) -> bool:
    return bool(a) and bool(a.get("state"))


# Asking what a sign MEANS is a different question from asking what this patient HAS, and the
# scaffolding is all that separates them in the wording. Stripped before retrieval so the query
# is the clinical term and not the politeness around it.
# Written without apostrophes because the question is stripped of them before matching: a
# French clinician types "c'est quoi" or "c est quoi" indifferently, and the second form
# missing the corpus while the first found it would be an accident of punctuation.
_ASK = ("what is a", "what is an", "what is the", "what is", "what are the", "what are",
        "what does a", "what does the", "what does", "whats a", "whats the", "whats",
        "explain the", "explain", "define the", "define",
        "tell me about the", "tell me about", "meaning of the", "meaning of",
        "c est quoi le", "c est quoi la", "c est quoi", "cest quoi le", "cest quoi",
        "qu est ce que le", "qu est ce que la", "qu est ce que", "quest ce que",
        "signification de la", "signification de")
_TAIL = (" mean", " means", " meaning", " look like", " on ultrasound", " in pocus",
         " exactly", " please", " sign", " signs")

_RETRIEVER: Any = None


def _corpus_retriever() -> Any:
    """Built once. The corpus is 32 short passages; the vectoriser is not worth rebuilding."""
    global _RETRIEVER
    if _RETRIEVER is None:
        from .clinical.retrieval import Retriever
        _RETRIEVER = Retriever()
    return _RETRIEVER


def knowledge_answer(question: str, a: dict[str, Any] | None = None) -> str | None:
    """Answer a REFERENCE question by quoting the corpus, or return None to keep routing.

    This is the one place the assistant answers something that is not about the patient in
    front of it, and the separation is the whole safety argument. A definition is general: it
    is quoted verbatim from a sourced passage, attributed, and labelled as reference rather
    than as a statement about this encounter. Nothing is paraphrased, because a paraphrase of
    a guideline is an unattributable claim wearing a citation.

    The relevance floor decides. A question whose term matches nothing in the corpus returns
    None and falls through to the rest of the router rather than being answered from a passage
    that merely shares a stopword -- the same rule the reasoning agent's retrieval obeys, and
    for the same reason: text that does not bear on the question is worse than no text,
    because the answer then LOOKS grounded.

    Unsourced passages are never quoted. The corpus ships placeholders so the pipeline can be
    tested before real passages exist, and one appearing here would be a fabricated reference.
    """
    q = " " + re.sub(r"[?!.,'’-]", " ", str(question).lower()).strip() + " "
    q = re.sub(r"\s+", " ", q)
    term = None
    for lead in _ASK:
        m = re.search(rf"\b{re.escape(lead)}\b\s+(.+)", q)
        if m:
            term = m.group(1)
            break
    if term is None:
        return None

    for tail in _TAIL:
        term = term.replace(tail, " ")
    term = re.sub(r"\s+", " ", term).strip(" -")
    # One or two words is a sign; a whole sentence is a question about the patient that
    # happened to open with "what is". Retrieval on it would match on incidental words.
    if not term or len(term.split()) > 4:
        return None

    hits = [h for h in _corpus_retriever().retrieve(term, k=3) if h["status"] == "sourced"]
    if not hits:
        return None

    lines = [f"[{h['n']}] {h['topic'].replace('_', ' ')} — {h['text']}\n    Source: "
             f"{h['source']}" for h in hits]
    out = (f"Reference on “{term}”, quoted from the corpus. This is general clinical "
           f"reference, NOT a statement about this patient — nothing below was computed "
           f"from this encounter:\n\n" + "\n\n".join(lines))

    # If the encounter also speaks to the term, say what it holds -- in its own paragraph,
    # after the reference and labelled as this patient. Someone asking what a sign means while
    # looking at a patient who has it wants both, and the one thing that must not happen is
    # the two arriving as a single undifferentiated claim.
    if a and has_encounter(a):
        st = a["state"]
        here = [f for f in st["imaging"]["findings"]
                if term in f["label"].lower() or f["label"].lower() in term]
        limits = [x for x in (st["imaging"].get("out_of_scope") or []) if term in x.lower()]
        own = []
        for f in here:
            own.append(f"{f['label']} was {'DETECTED' if f['detected'] else 'screened and NOT '
                       'detected'} at {f['confidence']:.2f}")
        own.extend(limits)
        if own:
            out += ("\n\nIn THIS encounter, computed rather than quoted:\n"
                    + "\n".join("• " + x for x in own))
    return out


def next_step_answer(a: dict[str, Any]) -> str:
    """What to do next, composed from the computed assessment and nothing else.

    "What should I do now?" is the question a clinician actually asks, and answering it with a
    semicolon-separated list of examinations was correct and close to useless. The answer is
    assembled here in the order the question is really asked in: where the patient stands, what
    is not known, what to obtain next, what the models cannot settle whatever is obtained, and
    when to stop reading the screen.

    Every line is drawn from a computed field. Nothing is composed about the patient that the
    pipeline did not derive: no diagnosis, no drug, no dose. An ACTION is not a diagnosis --
    "complete the observations" can be said safely where "this is heart failure" cannot -- and
    that distinction is the whole reason this can be answered without a model.

    Note which context this uses. The retrieval query deliberately EXCLUDES absent tests,
    because querying on what was never measured retrieves literature about it and invites
    reasoning from a gap. Acting is the opposite case: what is missing, what escalated and what
    the models cannot exclude are exactly what determines the next move, so they are central
    here. Retrieval context and action context are not the same context.
    """
    state, sup, esc = a["state"], a["support"], a["esc"]
    det = [f for f in state["imaging"]["findings"] if f["detected"]]
    gone = state["missing"]["labs"] + state["missing"]["vitals"]
    never = state["imaging"].get("organs_not_assessed") or []
    out = state["imaging"].get("out_of_scope") or []
    blocks: list[str] = []

    # 1. Where the patient stands.
    sev = sup["severity"]["severity"]
    line = (f"Assessed at {sev} severity with {len(sup['alerts'])} alert(s). ")
    if det:
        line += ("POCUS detected " + ", ".join(f"{f['label']} ({f['confidence']:.2f})"
                                               for f in det)
                 + ". A finding's confidence is the model's confidence that the SIGN is "
                   "present; it is not the likelihood of any diagnosis, and these require "
                   "clinical correlation.")
    else:
        line += ("No POCUS finding reached threshold. That is the absence of the findings "
                 "these modules screen for, not the absence of pathology.")
    blocks.append("WHERE THIS STANDS\n" + line)

    # 2. What is not known. Never assessed comes first: a scan that did not happen is a bigger
    #    hole than a lab that did not result, and it is the one most often read as negative.
    unknown = []
    if never:
        unknown.append("Never assessed: " + ", ".join(never)
                       + " — a gap in the record, not a negative result.")
    if gone:
        unknown.append(f"Never measured ({len(gone)}): " + ", ".join(gone)
                       + " — absent, not normal. These carry no evidence identifier, so "
                         "nothing can be cited for or against a diagnosis from them.")
    blocks.append("WHAT IS NOT KNOWN\n" + ("\n".join(unknown) if unknown
                  else "Every value in the reference set was measured and every expected "
                       "view was obtained."))

    # 3. What to obtain, in the order the support module ranked it.
    r = sup["additional_examinations"]
    blocks.append("WHAT TO OBTAIN NEXT\n"
                  + ("\n".join(f"{i}. {x['exam']} ({x['priority']}) — {x['reason']}"
                               for i, x in enumerate(r[:6], 1)) if r else
                     "Nothing further is recommended for this presentation; the record is "
                     "complete."))

    # 4. What obtaining it still will not settle.
    if out:
        blocks.append("WHAT THIS CANNOT SETTLE\n"
                      + "\n".join("• " + x for x in out)
                      + "\nThese bound the conclusion whatever else is obtained.")

    # 5. When to stop reading the screen.
    if esc["escalate"]:
        blocks.append("ESCALATION\nEscalation is required. Triggered by:\n"
                      + "\n".join("• " + t for t in esc["triggers"])
                      + "\nThe triggers are computed before the reasoning model runs and are "
                        "unaffected by whether it ran, so this stands even if no differential "
                        "was produced.")
    else:
        blocks.append("ESCALATION\nNo escalation trigger fired. That is a statement about "
                      "this record, not about the patient: a deteriorating patient is an "
                      "emergency pathway decision, not a screen-reading one.")

    # 6. Therapeutics, only where a protocol backs them.
    th = sup["therapeutic"]
    if th.get("considerations"):
        blocks.append("THERAPEUTIC CONSIDERATIONS\n"
                      + "\n".join(f"• {c['consideration']} (basis: {c['basis']}, "
                                  f"{c['passage']})" for c in th["considerations"][:3])
                      + "\nEach requires clinician verification.")
    else:
        blocks.append(f"THERAPEUTIC CONSIDERATIONS\nNone offered — {th['status']}. Treatment "
                      f"is never generated from the model's own knowledge, because a reader "
                      f"could not tell that from a protocol-backed one.")

    # 7. Sources, only if something actually cleared the relevance floor.
    hits = a.get("hits") or []
    if hits:
        blocks.append("RETRIEVED FOR THIS PRESENTATION\n"
                      + "\n".join(f"[{h['n']}] {h['topic']} — {h['source']}" for h in hits[:4]))

    return "\n\n".join(blocks)


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

    def word(*tokens: str) -> bool:
        """Whole-word match. `hi` must not fire on `troponin high`."""
        return any(f" {t} " in q for t in tokens)

    # ---- conversation, before anything clinical -------------------------------------
    # A greeting is not a clinical question, and answering one with a refusal about
    # fabricating clinical opinions reads as a broken assistant. It is also the first thing
    # anyone types.
    if word("hi", "hello", "hey", "yo", "salut", "bonjour", "coucou") or \
            has("good morning", "good afternoon", "good evening"):
        if not has_encounter(a):
            return ("Hello. No encounter has been analysed yet, so I have nothing to read from "
                    "— open Patient workup, enter what you have and analyse the case, and I "
                    "can answer from it.")
        d = a["state"].get("demographics") or {}
        return (f"Hello. I have encounter {rep['encounter_id']}: {d.get('age', '—')} · "
                f"{d.get('sex', '—')}, "
                f"{d.get('chief_complaint') or 'no complaint given'}, assessed at "
                f"{sup['severity']['severity']} severity with {len(sup['alerts'])} alert(s).\n\n"
                f"Ask me anything about it — a value, what is missing, why the severity is what "
                f"it is, what the scan saw, or what to do next.")

    # "ok" and "okay" are deliberately NOT here: "is she going to be ok" is a prognosis
    # question, and treating it as an acknowledgement would skip the boundary below.
    if word("thanks", "thank", "thx", "merci", "cool", "bye", "goodbye"):
        return "Noted. Ask whenever you want something from the record."

    if has("who are you", "what are you", "what can you do", "what do you do", "help me",
           " help ", "capabilities", "how do you work", "what is this"):
        return ("I am the clinical assistant for this encounter. I do not generate clinical "
                "prose: I read your question and return what the computed assessment actually "
                "holds, so every line I give you is quoted from the record rather than "
                "composed.\n\n" + CAPABILITIES)

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

    # ---- a reference question about a sign or protocol --------------------------------
    # After the measurement lookup and before everything else. "What is the troponin?" asks
    # this patient's value and is answered above; "what are B-lines?" asks what the sign means,
    # and answering it with this encounter's b-line confidence replies to a question about
    # medicine with a fact about one patient. The corpus decides: a term it does not cover
    # returns None and the encounter intents below get their turn.
    know = knowledge_answer(question, a)
    if know:
        return know

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

    # "what shoud i do now??" reached the catch-all: the intent was matched as the literal
    # phrase "what should i do", and one missing letter defeated it. A clinician asking the
    # most obvious question of all should not have to spell it correctly, so the intent is
    # matched on the words that carry it -- some form of "do" together with "now" or "next" --
    # rather than on a phrase they have to reproduce exactly.
    if has("suggest", "recommend", "next", "investigate", "order", "which test", "what test",
           "what should i do", "work up", "workup", "plan", "what now", "assess next",
           "que faire", "prochaine", "maintenant") or \
            (has(" do ", " doing ", " shoud ", " should ") and has("now", "next", "then")):
        return next_step_answer(a)

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

    # Nothing in the record matched the words. That is a reason to say what the record DOES
    # hold, not to recite a menu: a list of capabilities answers a question nobody asked, and
    # reads as a broken assistant at the exact moment someone is trying to use it. The
    # standing summary is real -- every number below is computed -- and it is usually what was
    # wanted anyway. The refusal to invent stands; it just no longer arrives empty-handed.
    lines = [f"Severity {sup['severity']['severity']}, {len(sup['alerts'])} alert(s), "
             f"{'escalation required' if esc['escalate'] else 'no escalation trigger'}."]
    if det:
        lines.append("POCUS: " + ", ".join(f"{f['label']} {f['confidence']:.2f}" for f in det))
    else:
        lines.append("POCUS: no finding above threshold.")
    if gone:
        lines.append(f"{len(gone)} value(s) never measured: " + ", ".join(gone[:6])
                     + ("…" if len(gone) > 6 else "") + " — absent, not normal.")
    r = sup["additional_examinations"]
    if r:
        lines.append("Recommended next: "
                     + "; ".join(f"{x['exam']} ({x['priority']})" for x in r[:3]))
    return ("I could not match that to anything specific in the record, so rather than write "
            "something plausible around the gap, here is where this encounter stands:\n\n"
            + "\n".join("• " + x for x in lines)
            + "\n\nAsk for a value, what is missing, why the severity is what it is, what the "
              "scan saw, or the evidence for a diagnosis, and I will read it out exactly.")
