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

from . import i18n
from .clinical.clinical_state import (
    LAB_REFERENCE, VITAL_REFERENCE, build_evidence)

# The ways a clinician actually writes each measurement, beyond its canonical key. Both
# languages in one set, because a question is matched before the interface language is
# consulted: a French doctor typing "troponin" and an English one typing "troponine" are both
# asking for the same value, and "quelle est la troponine ?" reaching the fallback instead of
# the number was the plainest possible way to look broken.
ALIASES: dict[str, set[str]] = {
    "spo2": {"oxygen", "saturation", "sats", "o2", "spo2", "sat", "oxygène", "oxygene",
             "saturation en oxygène"},
    "hr": {"heart rate", "pulse", "hr", "fréquence cardiaque", "frequence cardiaque",
           "pouls", "fc"},
    "sbp": {"blood pressure", "systolic", "sbp", "bp", "pression artérielle", "systolique",
            "tension", "pas", "pression"},
    "rr": {"respiratory rate", "breathing rate", "rr", "fréquence respiratoire",
           "frequence respiratoire", "fr"},
    "temp": {"temperature", "temp", "température", "fièvre", "fievre"},
    "d_dimer": {"d dimer", "d-dimer", "ddimer", "d dimères", "d-dimères", "dimères",
                "d dimeres"},
    "bnp": {"bnp", "natriuretic", "natriurétique"},
    "wbc": {"wbc", "white cells", "white count", "leucocytes", "globules blancs", "gb"},
    "crp": {"crp", "c reactive protein", "protéine c réactive"},
    "ph": {"ph", "acid base", "acido"},
    "troponin": {"troponin", "troponine"},
    "lactate": {"lactate", "lactates", "lactatémie"},
    "creatinine": {"creatinine", "créatinine", "creat"},
}

CAPABILITIES = (
    "I can answer, from what was actually computed: any named vital or laboratory value and "
    "whether it was measured at all; what is missing; the alerts and the bound each crossed; "
    "the severity and why; the escalation triggers; what POCUS saw and what it cannot exclude; "
    "the differential and the evidence cited for and against each entry; the retrieved "
    "sources; the scenario routing; case quality; calibration and thresholds; and the stage "
    "timings."
)


CAPABILITIES_FR = (
    "Je peux répondre, à partir de ce qui a réellement été calculé : n'importe quelle "
    "constante ou valeur biologique nommée, et si elle a seulement été mesurée ; ce qui "
    "manque ; les alertes et la borne franchie par chacune ; la sévérité et pourquoi ; les "
    "déclencheurs d'escalade ; ce que l'échographie a vu et ce qu'elle ne peut pas exclure ; "
    "le diagnostic différentiel et les éléments cités pour et contre chaque hypothèse ; les "
    "sources retrouvées ; l'orientation par scénario ; la qualité du dossier ; la calibration "
    "et les seuils ; et les durées de chaque étape."
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


def knowledge_answer(question: str, a: dict[str, Any] | None = None,
                     lang: str = "en") -> str | None:
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

    FR = lang == "fr"
    src = "Source : " if FR else "Source: "
    lines = [f"[{h['n']}] {h['topic'].replace('_', ' ')} — {h['text']}\n    {src}"
             f"{h['source']}" for h in hits]
    # The passages themselves stay in the language they were published in. A translated
    # quotation is no longer a quotation, and it would carry a citation to a source that never
    # said it; the frame around them says so in the reader's language.
    head = (f"Référence sur « {term} », citée mot pour mot du corpus, en anglais : ce sont "
            f"des extraits publiés, et les traduire en ferait des citations que leurs auteurs "
            f"n'ont pas écrites. Il s'agit d'une référence clinique générale, PAS d'une "
            f"affirmation sur ce patient — rien de ce qui suit n'a été calculé à partir de "
            f"cette prise en charge :\n\n" if FR else
            f"Reference on “{term}”, quoted from the corpus. This is general clinical "
            f"reference, NOT a statement about this patient — nothing below was computed "
            f"from this encounter:\n\n")
    out = head + "\n\n".join(lines)

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
            if FR:
                verb = "DÉTECTÉ" if f["detected"] else "recherché et NON détecté"
                own.append(f"{f['label']} : {verb} à {f['confidence']:.2f}")
            else:
                verb = "DETECTED" if f["detected"] else "screened and NOT detected"
                own.append(f"{f['label']} was {verb} at {f['confidence']:.2f}")
        own.extend(i18n.limit(x) if FR else x for x in limits)
        if own:
            out += (("\n\nDans CETTE prise en charge, calculé et non cité :\n" if FR
                     else "\n\nIn THIS encounter, computed rather than quoted:\n")
                    + "\n".join("• " + x for x in own))
    return out


def next_step_answer(a: dict[str, Any], lang: str = "en") -> str:
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
    FR = lang == "fr"

    def t(en: str, fr: str) -> str:
        return fr if FR else en

    det = [f for f in state["imaging"]["findings"] if f["detected"]]
    gone = state["missing"]["labs"] + state["missing"]["vitals"]
    never = state["imaging"].get("organs_not_assessed") or []
    out = state["imaging"].get("out_of_scope") or []
    blocks: list[str] = []

    # 1. Where the patient stands.
    sev = i18n.severity(sup["severity"]["severity"]) if FR else sup["severity"]["severity"]
    line = t(f"Assessed at {sev} severity with {len(sup['alerts'])} alert(s). ",
             f"Évalué à une sévérité {sev} avec {len(sup['alerts'])} alerte(s). ")
    if det:
        found = ", ".join(f"{f['label']} ({f['confidence']:.2f})" for f in det)
        line += t("POCUS detected " + found + ". A finding's confidence is the model's "
                  "confidence that the SIGN is present; it is not the likelihood of any "
                  "diagnosis, and these require clinical correlation.",
                  "L'échographie a détecté " + found + ". La confiance d'un signe est la "
                  "confiance du modèle dans la PRÉSENCE du signe ; ce n'est pas la "
                  "vraisemblance d'un diagnostic, et cela demande une corrélation clinique.")
    else:
        line += t("No POCUS finding reached threshold. That is the absence of the findings "
                  "these modules screen for, not the absence of pathology.",
                  "Aucun signe échographique n'a atteint le seuil. C'est l'absence des signes "
                  "que ces modules recherchent, pas l'absence de pathologie.")
    blocks.append(t("WHERE THIS STANDS\n", "OÙ EN EST CE PATIENT\n") + line)

    # 2. What is not known. Never assessed comes first: a scan that did not happen is a bigger
    #    hole than a lab that did not result, and it is the one most often read as negative.
    unknown = []
    if never:
        nv = i18n.organs(never) if FR else ", ".join(never)
        unknown.append(t("Never assessed: " + nv
                         + " — a gap in the record, not a negative result.",
                         "Jamais exploré : " + nv
                         + " — une lacune du dossier, pas un résultat négatif."))
    if gone:
        gv = ", ".join(i18n.measure(g) for g in gone) if FR else ", ".join(gone)
        unknown.append(t(f"Never measured ({len(gone)}): " + gv
                         + " — absent, not normal. These carry no evidence identifier, so "
                           "nothing can be cited for or against a diagnosis from them.",
                         f"Jamais mesuré ({len(gone)}) : " + gv
                         + " — absent, pas normal. Ces valeurs n'ont aucun identifiant de "
                           "preuve : rien ne peut être cité pour ou contre un diagnostic à "
                           "partir d'elles."))
    blocks.append(t("WHAT IS NOT KNOWN\n", "CE QUI N'EST PAS CONNU\n")
                  + ("\n".join(unknown) if unknown
                     else t("Every value in the reference set was measured and every expected "
                            "view was obtained.",
                            "Toutes les valeurs de référence ont été mesurées et toutes les "
                            "coupes attendues obtenues.")))

    # 3. What to obtain, in the order the support module ranked it.
    r = sup["additional_examinations"]
    rows = "\n".join(
        f"{i}. " + (i18n.exam_name(x["exam"]) if FR else x["exam"]) + " ("
        + (i18n.PRIORITY.get(x["priority"], x["priority"]) if FR else x["priority"]) + ") — "
        + (i18n.exam_reason(x["reason"]) if FR else x["reason"])
        for i, x in enumerate(r[:6], 1))
    blocks.append(t("WHAT TO OBTAIN NEXT\n", "CE QU'IL FAUT OBTENIR ENSUITE\n")
                  + (rows if r else
                     t("Nothing further is recommended for this presentation; the record is "
                       "complete.",
                       "Aucun examen supplémentaire n'est recommandé pour cette présentation ; "
                       "le dossier est complet.")))

    # 4. What obtaining it still will not settle.
    if out:
        lines = "\n".join("• " + (i18n.limit(x) if FR else x) for x in out)
        blocks.append(t("WHAT THIS CANNOT SETTLE\n", "CE QUE CECI NE PEUT PAS TRANCHER\n")
                      + lines
                      + t("\nThese bound the conclusion whatever else is obtained.",
                          "\nCeci borne la conclusion quoi que l'on obtienne par ailleurs."))

    # 5. When to stop reading the screen.
    if esc["escalate"]:
        trg = "\n".join("• " + (i18n.trigger(x) if FR else x) for x in esc["triggers"])
        blocks.append(t("ESCALATION\nEscalation is required. Triggered by:\n",
                        "ESCALADE\nUne escalade est requise. Déclenchée par :\n") + trg
                      + t("\nThe triggers are computed before the reasoning model runs and "
                          "are unaffected by whether it ran, so this stands even if no "
                          "differential was produced.",
                          "\nLes déclencheurs sont calculés avant l'exécution du modèle de "
                          "raisonnement et ne dépendent pas de lui : ceci tient même si aucun "
                          "différentiel n'a été produit."))
    else:
        blocks.append(t("ESCALATION\nNo escalation trigger fired. That is a statement about "
                        "this record, not about the patient: a deteriorating patient is an "
                        "emergency pathway decision, not a screen-reading one.",
                        "ESCALADE\nAucun déclencheur d'escalade ne s'est activé. C'est un "
                        "constat sur ce dossier, pas sur le patient : un patient qui se "
                        "dégrade relève d'une décision de filière d'urgence, pas de la "
                        "lecture d'un écran."))

    # 6. Therapeutics, only where a protocol backs them.
    th = sup["therapeutic"]
    if th.get("considerations"):
        body = "\n".join(f"• {c['consideration']} ({c['basis']}, {c['passage']})"
                         for c in th["considerations"][:3])
        blocks.append(t("THERAPEUTIC CONSIDERATIONS\n", "CONSIDÉRATIONS THÉRAPEUTIQUES\n")
                      + body
                      + t("\nEach requires clinician verification.",
                          "\nChacune requiert une vérification par le clinicien."))
    else:
        blocks.append(t(f"THERAPEUTIC CONSIDERATIONS\nNone offered — {th['status']}. "
                        f"Treatment is never generated from the model's own knowledge, "
                        f"because a reader could not tell that from a protocol-backed one.",
                        "CONSIDÉRATIONS THÉRAPEUTIQUES\nAucune proposée : aucun protocole "
                        "approuvé n'est disponible. Un traitement n'est jamais généré à "
                        "partir des connaissances propres du modèle, car un lecteur ne "
                        "pourrait pas le distinguer d'une proposition appuyée sur un "
                        "protocole."))

    # 7. Sources, only if something actually cleared the relevance floor.
    hits = a.get("hits") or []
    if hits:
        blocks.append(t("RETRIEVED FOR THIS PRESENTATION\n",
                        "RETROUVÉ POUR CETTE PRÉSENTATION\n")
                      + "\n".join(f"[{h['n']}] {h['topic']} — {h['source']}"
                                  for h in hits[:4]))

    return "\n\n".join(blocks)


def answer(question: str, a: dict[str, Any], lang: str = "en") -> str:
    """Answer `question` from the analysis dict produced by the pipeline.

    `a` carries: state, esc, support, hits, result, report, marks, origin.

    Both languages are routed by the same rules and answered from the same fields. The cues
    matter as much as the text: a French clinician types "quelles valeurs manquent", and an
    assistant that only recognises "missing" would answer every French question with its
    fallback while appearing to work. Each intent therefore carries its French cues beside its
    English ones, and the cues are matched regardless of the interface language -- a French
    doctor who types an English word, or the reverse, is understood either way.
    """
    q = " " + str(question).lower().strip() + " "
    FR = lang == "fr"

    def t(en: str, fr: str) -> str:
        return fr if FR else en
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
    if word("hi", "hello", "hey", "yo", "salut", "bonjour", "coucou", "bonsoir") or \
            has("good morning", "good afternoon", "good evening"):
        if not has_encounter(a):
            return t("Hello. No encounter has been analysed yet, so I have nothing to read "
                     "from — open Patient workup, enter what you have and analyse the case, "
                     "and I can answer from it.",
                     "Bonjour. Aucune prise en charge n'a encore été analysée, je n'ai donc "
                     "rien à lire — ouvrez « Saisie du patient », renseignez ce dont vous "
                     "disposez et analysez le cas, et je pourrai répondre à partir de là.")
        d = a["state"].get("demographics") or {}
        sev_txt = i18n.severity(sup["severity"]["severity"]) if FR \
            else sup["severity"]["severity"]
        cc = d.get("chief_complaint") or t("no complaint given", "aucun motif renseigné")
        return t(f"Hello. I have encounter {rep['encounter_id']}: {d.get('age', '—')} · "
                 f"{d.get('sex', '—')}, {cc}, assessed at {sev_txt} severity with "
                 f"{len(sup['alerts'])} alert(s).\n\nAsk me anything about it — a value, what "
                 f"is missing, why the severity is what it is, what the scan saw, or what to "
                 f"do next.",
                 f"Bonjour. J'ai la prise en charge {rep['encounter_id']} : "
                 f"{d.get('age', '—')} · {d.get('sex', '—')}, {cc}, évaluée à une sévérité "
                 f"{sev_txt} avec {len(sup['alerts'])} alerte(s).\n\nPosez-moi n'importe "
                 f"quelle question dessus — une valeur, ce qui manque, pourquoi cette "
                 f"sévérité, ce que l'échographie a vu, ou quoi faire ensuite.")

    # "ok" and "okay" are deliberately NOT here: "is she going to be ok" is a prognosis
    # question, and treating it as an acknowledgement would skip the boundary below.
    if word("thanks", "thank", "thx", "merci", "cool", "bye", "goodbye", "au revoir"):
        return t("Noted. Ask whenever you want something from the record.",
                 "Noté. Demandez-moi dès que vous voulez quelque chose du dossier.")

    if has("who are you", "what are you", "what can you do", "what do you do", "help me",
           " help ", "capabilities", "how do you work", "what is this",
           "qui es-tu", "qui êtes-vous", "que sais-tu", "que peux-tu", "aide", "aidez",
           "comment ça marche", "comment fonctionnes"):
        return t("I am the clinical assistant for this encounter. I do not generate clinical "
                 "prose: I read your question and return what the computed assessment "
                 "actually holds, so every line I give you is quoted from the record rather "
                 "than composed.\n\n" + CAPABILITIES,
                 "Je suis l'assistant clinique de cette prise en charge. Je ne génère pas de "
                 "texte clinique : je lis votre question et je renvoie ce que l'évaluation "
                 "calculée contient réellement, de sorte que chaque ligne que je vous donne "
                 "est citée du dossier et non composée.\n\n" + CAPABILITIES_FR)

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
            name = i18n.measure(key) if FR else key
            flag = i18n.FLAG.get(entry["flag"], entry["flag"]) if FR else entry["flag"]
            return (t(f"{name} is {entry['value']:g} {entry['unit']} — flagged {flag}. "
                      f"Reference {lo}–{hi} {ref.get('unit', '')}".strip(),
                      f"{name} : {entry['value']:g} {entry['unit']} — signalé {flag}. "
                      f"Référence {lo}–{hi} {ref.get('unit', '')}".strip())
                    + (t(f". Citable as {eid}.", f". Citable sous {eid}.") if eid else "."))
        name = i18n.measure(key) if FR else key
        return t(f"{key} was NOT measured for this patient. That is not the same as normal: it "
                 f"has no evidence identifier, so nothing in the assessment can argue for or "
                 f"against a diagnosis using it. It appears in the missing-information list, "
                 f"and in the recommended examinations if it bears on this presentation.",
                 f"{name} n'a PAS été mesuré pour ce patient. Ce n'est pas la même chose que "
                 f"normal : cette valeur n'a aucun identifiant de preuve, donc rien dans "
                 f"l'évaluation ne peut argumenter pour ou contre un diagnostic à partir "
                 f"d'elle. Elle figure dans la liste des informations manquantes, et dans les "
                 f"examens recommandés si elle concerne cette présentation.")

    # ---- a reference question about a sign or protocol --------------------------------
    # After the measurement lookup and before everything else. "What is the troponin?" asks
    # this patient's value and is answered above; "what are B-lines?" asks what the sign means,
    # and answering it with this encounter's b-line confidence replies to a question about
    # medicine with a fact about one patient. The corpus decides: a term it does not cover
    # returns None and the encounter intents below get their turn.
    know = knowledge_answer(question, a, lang)
    if know:
        return know

    # ---- intents ---------------------------------------------------------------------
    if has("treat", "therapy", "therapeutic", "manage", "give ", "drug", "dose",
           "medication", "prescri", "fluid", "antibiotic",
           "traite", "traitement", "thérapeut", "prise en charge", "médicament",
           "posologie", "prescri", "perfusion", "antibio", "que donner"):
        th = sup["therapeutic"]
        if th["considerations"]:
            body = "; ".join(f"{c['consideration']} ({c['basis']}, {c['passage']})"
                             for c in th["considerations"])
            return t("A protocol-backed consideration is available: " + body
                     + ". This is decision support, not a treatment instruction.",
                     "Une considération appuyée par un protocole est disponible : " + body
                     + ". Ceci est une aide à la décision, pas une prescription.")
        return t(f"No therapeutic recommendation for this case. {th['status']}. Therapeutic "
                 f"suggestions are gated behind a sourced, citable protocol; with none "
                 f"retrieved the system produces nothing rather than drawing on the language "
                 f"model's own training knowledge. It is decision support and does not "
                 f"instruct treatment.",
                 "Aucune recommandation thérapeutique pour ce cas : aucun protocole approuvé "
                 "n'est disponible. Les suggestions thérapeutiques sont conditionnées à un "
                 "protocole sourcé et citable ; à défaut, le système ne produit rien plutôt "
                 "que de puiser dans les connaissances propres du modèle de langage. C'est "
                 "une aide à la décision, et cela ne prescrit aucun traitement.")

    if has("prognos", "survive", "going to be ok", "going to be okay", "will she", "will he",
           "will they", "outcome", "die", "mortality", "chance of", "how likely is he",
           "how likely is she", "recover",
           "pronostic", "survie", "survivre", "va-t-elle", "va-t-il", "s en sortir",
           "mourir", "décès", "mortalité", "guérir", "issue"):
        # A prognosis is a prediction about a patient's future, and nothing in this system
        # produces one. Every figure it holds describes the record as it stands: what was
        # measured, what was seen, what is absent, how urgent the fixed rules judge it. Reaching
        # into the corpus for something adjacent would dress a keyword match as a forecast.
        sev_txt = i18n.severity(sup["severity"]["severity"]) if FR \
            else sup["severity"]["severity"]
        return t(f"This system does not predict outcomes, and I will not imply one. It was "
                 f"never validated against what happened to any patient — no figure in it "
                 f"describes a future.\n\nWhat it does say about this encounter, now: "
                 f"severity {sev_txt}, {len(sup['alerts'])} alert(s), "
                 f"{'escalation required' if esc['escalate'] else 'no escalation'}"
                 + (f", and {len(gone)} value(s) never measured" if gone else "")
                 + ". The clinician makes the prognosis; this is decision support.",
                 f"Ce système ne prédit pas l'évolution d'un patient, et je ne vais pas le "
                 f"laisser entendre. Il n'a jamais été validé contre le devenir réel d'aucun "
                 f"patient — aucun chiffre qu'il contient ne décrit un avenir.\n\nCe qu'il "
                 f"dit de cette prise en charge, maintenant : sévérité {sev_txt}, "
                 f"{len(sup['alerts'])} alerte(s), "
                 f"{'escalade requise' if esc['escalate'] else 'pas d escalade'}"
                 + (f", et {len(gone)} valeur(s) jamais mesurée(s)" if gone else "")
                 + ". Le pronostic revient au clinicien ; ceci est une aide à la décision.")

    # "what shoud i do now??" reached the catch-all: the intent was matched as the literal
    # phrase "what should i do", and one missing letter defeated it. A clinician asking the
    # most obvious question of all should not have to spell it correctly, so the intent is
    # matched on the words that carry it -- some form of "do" together with "now" or "next" --
    # rather than on a phrase they have to reproduce exactly.
    if has("suggest", "recommend", "next", "investigate", "order", "which test", "what test",
           "what should i do", "work up", "workup", "plan", "what now", "assess next",
           "que faire", "prochaine", "maintenant") or \
            (has(" do ", " doing ", " shoud ", " should ") and has("now", "next", "then")):
        return next_step_answer(a, lang)

    if has("miss", "absent", "not measured", "unavailable", "don't have", "do not have",
           "manque", "manquant", "non mesur", "pas mesur", "indisponible", "absente"):
        names = ", ".join(i18n.measure(g) for g in gone) if FR else ", ".join(gone)
        return (t("Not measured for this patient: " + names + ". These have no evidence "
                  "identifier, so the reasoning layer cannot cite them for or against any "
                  "diagnosis. Absent is not normal.",
                  "Jamais mesuré pour ce patient : " + names + ". Ces valeurs n'ont aucun "
                  "identifiant de preuve : la couche de raisonnement ne peut donc les citer "
                  "ni pour ni contre un diagnostic. Absent n'est pas normal.")) if gone else \
            t("Every value in the reference set was measured.",
              "Toutes les valeurs de référence ont été mesurées.")

    if has("why", "reason", "because", "justif", "how did you",
           "pourquoi", "raison", "parce que", "justif", "comment avez"):
        trg = esc["triggers"]
        trg_txt = ("; ".join(i18n.trigger(x) for x in trg) if FR else "; ".join(trg)) \
            if trg else t("none", "aucun")
        sev_txt = i18n.severity(sup["severity"]["severity"]) if FR \
            else sup["severity"]["severity"]
        rs = sup["severity"]["reasons"] or ["—"]
        reasons = "; ".join(i18n.severity_reason(x) for x in rs) if FR else "; ".join(rs)
        return t(f"Severity is {sev_txt} because: {reasons}. Escalation triggers that fired: "
                 f"{trg_txt}. All of this is computed by fixed rules before the model runs.",
                 f"La sévérité est {sev_txt} parce que : {reasons}. Déclencheurs d'escalade "
                 f"activés : {trg_txt}. Tout ceci est calculé par des règles fixes avant "
                 f"l'exécution du modèle.")

    if has("challenge", "wrong", "sure", "certain", "limitation", "limit",
           "conteste", "contester", "faux", "sûr", "limite", "limitation"):
        lim = state["imaging"]["out_of_scope"]
        lim_txt = ("; ".join(i18n.limit(x) for x in lim) if FR else "; ".join(lim)) if lim \
            else t("no scope limit was declared", "aucune limite de portée n'a été déclarée")
        absent = ", ".join(i18n.measure(g) for g in gone) if FR else ", ".join(gone)
        grade = (state.get("case_quality") or {}).get("grade")
        grade_txt = i18n.GRADE.get(grade, grade) if FR else grade
        conf = state.get("conflicts") or []
        conf_txt = ("; ".join(i18n.conflict(c) for c in conf) if FR else "; ".join(conf))
        return (t(f"What most limits this assessment: {lim_txt}",
                  f"Ce qui limite le plus cette évaluation : {lim_txt}")
                + (t(". Absent tests: ", ". Examens absents : ") + absent if gone else "")
                + t(f". Case quality is graded {grade_txt}.",
                    f". La qualité du dossier est jugée {grade_txt}.")
                + ((t(" The agents disagree: ", " Les agents sont en désaccord : ") + conf_txt)
                   if conf else ""))

    if has("alert", "critical", "danger", "urgent", "red flag", "worry",
           "alerte", "critique", "danger", "urgence", "signal d alarme", "inqui"):
        al = sup["alerts"]
        body = "; ".join(
            (f"[{i18n.LEVEL.get(x['severity'], x['severity'])}] {i18n.alert(x)}" if FR
             else f"[{x['severity']}] {x['message']}") for x in al)
        return (t(f"Alerts, each naming the bound it crossed: {body}. "
                  f"Thresholds v{sup['thresholds_version']}.",
                  f"Alertes, chacune nommant la borne franchie : {body}. "
                  f"Seuils v{sup['thresholds_version']}.")) if al else \
            t("No alert fired. No measured value crossed a configured bound and the record "
              "shows no structural gap.",
              "Aucune alerte déclenchée. Aucune valeur mesurée n'a franchi une borne "
              "configurée et le dossier ne présente aucune lacune structurelle.")

    if has("severity", "priority", "escalat", "how bad", "how serious", "admit", "icu",
           "dispos",
           "sévérit", "gravit", "priorit", "escalade", "hospitalis", "réanimation",
           "orientation"):
        base = answer("why", a, lang)
        if has("admit", "icu", "dispos", "hospitalis", "réanimation", "orientation"):
            base += t(" Disposition is outside what this system decides: it produces a "
                      "severity, alerts and recommended examinations, not a bed decision.",
                      " L'orientation ne relève pas de ce système : il produit une sévérité, "
                      "des alertes et des examens recommandés, pas une décision "
                      "d'hospitalisation.")
        return base

    if has("pocus", "ultrasound", "scan", "image", "b-line", "b line", "finding", "detect",
           "see", "saw",
           "échographie", "echographie", "écho", "coupe", "ligne b", "signe",
           "détect", "detect", "vu", "montre"):
        found = "; ".join(f"{f['label']} ({f['confidence']:.2f})" for f in det) if det \
            else t("no finding above the tuned threshold",
                   "aucun signe au-dessus du seuil calibré")
        neg = ", ".join(f["label"] for f in state["imaging"]["findings"]
                        if not f["detected"]) or t("none", "aucun")
        lim = state["imaging"]["out_of_scope"]
        lim_txt = ("; ".join(i18n.limit(x) for x in lim) if FR else "; ".join(lim))
        return (t(f"POCUS detected: {found}. Screened and not seen: {neg}. ",
                  f"L'échographie a détecté : {found}. Recherché et non vu : {neg}. ")
                + lim_txt)

    if has("differential", "diagnos", "cause", "what could", "what might", "possib",
           "différentiel", "differentiel", "diagnostic", "hypothèse", "hypothese",
           "qu est-ce que ça pourrait", "possibilit"):
        if not diff:
            why = (t("The model backend failed and the answer was withheld; the severity and "
                     "alerts beside it were computed before the model ran.",
                     "Le moteur du modèle a échoué et la réponse a été retenue ; la sévérité "
                     "et les alertes ont été calculées avant son exécution.")
                   if a["origin"] == "failed" else
                   t("Generating one needs a 4.9 GB model on a GPU, which is not loaded here.",
                     "En générer un nécessite un modèle de 4,9 Go sur GPU, non chargé ici."))
            return t("No differential was produced for this encounter. ",
                     "Aucun diagnostic différentiel n'a été produit pour cette prise en "
                     "charge. ") + why
        band = {"high": "élevée", "moderate": "modérée", "low": "faible"}
        ranked = "; ".join(
            f"{i}. {d.get('diagnosis')} ("
            + (band.get(str(d.get('likelihood')).lower(), str(d.get('likelihood'))) if FR
               else str(d.get('likelihood'))) + ")"
            for i, d in enumerate(diff, 1))
        return t(f"Ranked: {ranked}. Offered for consideration — this is not a diagnosis.",
                 f"Classé : {ranked}. Proposé à titre indicatif — ceci n'est pas un "
                 f"diagnostic.")

    if has("support", "evidence for", "argue for", "in favour", "in favor",
           "en faveur", "soutien", "soutiennent", "étaye", "etaye", "arguments pour"):
        sev_txt = i18n.severity(sup["severity"]["severity"]) if FR \
            else sup["severity"]["severity"]
        if not diff:
            return t(f"No differential was produced, so there is nothing ranked to support. "
                     f"The deterministic layer did run: severity {sev_txt}, "
                     f"{len(sup['alerts'])} alert(s).",
                     f"Aucun différentiel n'a été produit : il n'y a donc rien de classé à "
                     f"étayer. La couche déterministe s'est bien exécutée : sévérité "
                     f"{sev_txt}, {len(sup['alerts'])} alerte(s).")
        d = diff[0]
        cited = "; ".join(f"{i} — {x}" for i, x in zip(d.get("supporting_ids") or [],
                                                       d.get("supporting") or []))
        return t(f"{d.get('diagnosis')} is ranked first. Evidence cited for it, by identifier: "
                 f"{cited}. Those identifiers come from the enumerated list built from this "
                 f"record, so nothing outside the record can be cited.",
                 f"{d.get('diagnosis')} est classé en premier. Éléments cités en sa faveur, "
                 f"par identifiant : {cited}. Ces identifiants proviennent de la liste "
                 f"énumérée construite à partir de ce dossier : rien d'extérieur au dossier "
                 f"ne peut donc être cité.")

    if has("against", "exclude", "rule out", "contradict",
           "en défaveur", "contre", "exclure", "éliminer", "eliminer", "contredit"):
        lim = state["imaging"]["out_of_scope"]
        against = "; ".join(
            f"{d.get('diagnosis')} : " + "; ".join(d.get("contradicting") or ["—"])
            for d in diff) or t("nothing was cited against any entry",
                                "rien n'a été cité contre une hypothèse")
        lim_txt = ("; ".join(i18n.limit(x) for x in lim) if FR else "; ".join(lim)) if lim \
            else t("no scope limit was declared", "aucune limite de portée n'a été déclarée")
        return t(f"Cited against: {against}. What the imaging cannot exclude: {lim_txt}. "
                 f"A finding the module does not model cannot be ruled out by it, whatever "
                 f"else the scan shows.",
                 f"Cité en défaveur : {against}. Ce que l'imagerie ne peut pas exclure : "
                 f"{lim_txt}. Un signe que le module ne modélise pas ne peut pas être écarté "
                 f"par lui, quoi que montre l'examen par ailleurs.")

    if has("summar", "overview", "tell me about", "who is", "recap",
           "résum", "resum", "synthèse", "synthese", "aperçu", "qui est"):
        return i18n.conclusion(rep) if FR else rep["conclusion"]

    if has("confiden", "calibrat", "reliab", "trust", "accurate", "threshold", "cutoff",
           "confiance", "calibr", "fiab", "précis", "seuil", "borne"):
        bits = [t(f"Alert thresholds are v{sup['thresholds_version']}, configuration for a "
                    f"prototype rather than a validated scoring system",
                    f"Les seuils d'alerte sont en v{sup['thresholds_version']} : une "
                    f"configuration de prototype, pas un score validé")]
        for lim in state["imaging"]["out_of_scope"]:
            bits.append(i18n.limit(lim) if FR else lim)
        bits.append(t("none of this has been validated against patient outcomes",
                      "rien de tout ceci n'a été validé contre le devenir réel des patients"))
        return ". ".join(bits) + "."

    if has("conflict", "disagree", "conflit", "désaccord", "desaccord",
           "contradiction"):
        c = state.get("conflicts") or []
        body = "; ".join(i18n.conflict(x) for x in c) if FR else "; ".join(c)
        return (t("The agents disagree: " + body + ". The system surfaces the disagreement "
                  "rather than resolving it.",
                  "Les agents sont en désaccord : " + body + ". Le système expose le "
                  "désaccord plutôt que de le trancher.")) if c else \
            t("The agents do not disagree on this encounter.",
              "Les agents ne sont pas en désaccord sur cette prise en charge.")

    if has("protocol", "guideline", "source", "reference", "citation", "corpus", "retriev",
           "paper", "evidence base",
           "protocole", "recommandation", "référence", "reference", "citation",
           "article", "littérature"):
        if not hits:
            return t("No passage cleared the 0.10 relevance floor for this encounter, so the "
                     "answer is not marked guideline-grounded. Below the floor the system "
                     "reports nothing rather than the best of a bad set.",
                     "Aucun passage n'a franchi le seuil de pertinence de 0,10 pour cette "
                     "prise en charge : la réponse n'est donc pas marquée comme appuyée sur "
                     "des recommandations. Sous le seuil, le système ne rapporte rien plutôt "
                     "que le moins mauvais d'un mauvais lot.")
        body = "; ".join(f"[{h['n']}] {h['topic']} ({h['score']:.2f}) — {h['source']}"
                         for h in hits)
        return t("Retrieved: ", "Passages retrouvés : ") + body

    if has("scenario", "route", "routed", "pathway",
           "scénario", "scenario", "orienté", "filière", "parcours"):
        lbl = i18n.scenario(sup["scenario"]["label"]) if FR else sup["scenario"]["label"]
        return t(f"Routed as {lbl}. Routing selects which modalities are expected and which "
                 f"protocol topic is looked up; it is matched on the presenting complaint by "
                 f"fixed cues, not by a model.",
                 f"Orienté comme : {lbl}. L'orientation détermine quelles modalités sont "
                 f"attendues et quel thème de protocole est consulté ; elle est établie à "
                 f"partir du motif de consultation par des indices fixes, pas par un modèle.")

    if has("quality", "complete", "enough",
           "qualité", "qualite", "complet", "suffis"):
        cq = state.get("case_quality") or {}
        grade = i18n.GRADE.get(cq.get("grade"), cq.get("grade")) if FR else cq.get("grade")
        return (t(f"Case quality is graded {grade}. ",
                  f"La qualité du dossier est jugée {grade}. ")
                + (t("Reasons: ", "Motifs : ") + "; ".join(cq.get("reasons") or [])
                   if cq.get("reasons")
                   else t("No quality concern was recorded.",
                          "Aucun problème de qualité n'a été relevé.")))

    if has("evidence", "cite", "identifier",
           "preuve", "élément", "element", "cité", "identifiant"):
        ev = build_evidence(state)
        used = {i for d in diff for i in (d.get("supporting_ids") or [])}
        body = "; ".join(f"{e['id']} {e['text']}" for e in ev)
        return t(f"{len(ev)} citable fact(s), {len(used)} cited by the model: {body}",
                 f"{len(ev)} fait(s) citable(s), dont {len(used)} cité(s) par le modèle : "
                 f"{body}")

    if has("how long", "how fast", "time", "latency", "speed",
           "combien de temps", "durée", "duree", "latence", "vitesse", "rapide"):
        body = "; ".join(f"{m['title']} +{m['t'] * 1000:.0f} ms" for m in a["marks"])
        return t("Stage timings for this encounter: " + body,
                 "Durées des étapes pour cette prise en charge : " + body)

    if has("vital", "observation", "constante", "paramètre", "parametre"):
        v = state.get("vitals") or {}
        body = "; ".join(
            f"{i18n.measure(k) if FR else k} {e['value']:g} {e['unit']} "
            f"({i18n.FLAG.get(e['flag'], e['flag']) if FR else e['flag']})"
            for k, e in v.items())
        gonev = state["missing"]["vitals"]
        gonev_txt = ", ".join(i18n.measure(x) for x in gonev) if FR else ", ".join(gonev)
        return (t("Vitals recorded: " + body, "Constantes relevées : " + body)
                + (t(f". Never measured: {gonev_txt}.",
                     f". Jamais mesuré : {gonev_txt}.") if gonev else "")) if v else \
            t("No vital sign was recorded for this patient.",
              "Aucune constante n'a été relevée pour ce patient.")

    if has("lab", "blood", "biolog", "biologie", "sang", "analyse", "bilan"):
        lb = state.get("labs") or {}
        body = "; ".join(
            f"{i18n.measure(k) if FR else k} {e['value']:g} {e['unit']} "
            f"({i18n.FLAG.get(e['flag'], e['flag']) if FR else e['flag']})"
            for k, e in lb.items())
        gonel = state["missing"]["labs"]
        gonel_txt = ", ".join(i18n.measure(x) for x in gonel) if FR else ", ".join(gonel)
        return (t("Laboratory values: " + body, "Valeurs biologiques : " + body)
                + (t(f". Never measured: {gonel_txt}.",
                     f". Jamais mesuré : {gonel_txt}.") if gonel else "")) if lb else \
            t("No laboratory value was recorded. Not measured is not normal: "
              + gonel_txt + " all have no identifier.",
              "Aucune valeur biologique n'a été relevée. Non mesuré n'est pas normal : "
              + gonel_txt + " n'ont aucun identifiant de preuve.")

    # ---- a diagnosis named in the question --------------------------------------------
    for d in diff:
        nm = str(d.get("diagnosis", "")).lower()
        if nm and (nm in q or any(w in q for w in nm.split() if len(w) > 5)):
            band = {"high": "élevée", "moderate": "modérée", "low": "faible"}
            lk = band.get(str(d.get("likelihood")).lower(), str(d.get("likelihood"))) if FR \
                else d.get("likelihood")
            forx = "; ".join(d.get("supporting") or ["—"])
            agx = "; ".join(d.get("contradicting") or ["—"])
            limx = "; ".join(d.get("limitations") or ["—"])
            return t(f"{d.get('diagnosis')} — {lk} likelihood. Cited for it: {forx}. "
                     f"Against: {agx}. Limits: {limx}.",
                     f"{d.get('diagnosis')} — vraisemblance {lk}. Cité en sa faveur : {forx}. "
                     f"En défaveur : {agx}. Limites : {limx}.")

    # ---- last resort: match the question against the record itself ---------------------
    # Rather than refuse on a phrasing the intents above did not anticipate, the words of the
    # question are matched against the evidence, the alerts, the triggers and the retrieved
    # passages. If anything in the record bears on what was asked, it is returned. This still
    # composes nothing -- every line is quoted from the encounter.
    # French stopwords sit beside the English ones. Without them "quelles sont les valeurs"
    # contributed "quelles", "sont" and "les" as search terms, and the fallback matched the
    # record on the grammar of the question rather than on its content.
    stop = {"what", "the", "is", "a", "an", "of", "for", "and", "or", "to", "in", "on", "do",
            "does", "did", "you", "i", "me", "my", "this", "that", "with", "about", "can",
            "should", "would", "could", "are", "was", "were", "be", "it", "his", "her",
            "patient", "any", "how", "why", "when", "which", "who", "tell", "show", "give",
            "que", "quoi", "qui", "quel", "quelle", "quels", "quelles", "est", "sont", "les",
            "des", "une", "pour", "avec", "dans", "sur", "par", "pas", "plus", "moi", "mon",
            "ma", "mes", "ce", "cet", "cette", "ces", "son", "sa", "ses", "elle", "il",
            "nous", "vous", "dis", "dites", "montre", "donne", "faire", "fait", "comment",
            "pourquoi", "combien", "quand", "chez", "aux", "du", "de", "la", "le"}
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
                pool.append((hitn, f"[{i18n.LEVEL.get(x['severity'], x['severity'])}] "
                                   f"{i18n.alert(x)}" if FR
                                   else f"[{x['severity']}] {x['message']}"))
        for tg in esc["triggers"]:
            hitn = sum(1 for w in words if w in tg.lower())
            if hitn:
                pool.append((hitn, t("escalation trigger: ", "déclencheur d'escalade : ")
                             + (i18n.trigger(tg) if FR else tg)))
        for x in sup["additional_examinations"]:
            hitn = sum(1 for w in words if w in (x["exam"] + " " + x["reason"]).lower())
            if hitn:
                ex = i18n.exam_name(x["exam"]) if FR else x["exam"]
                pr = i18n.PRIORITY.get(x["priority"], x["priority"]) if FR else x["priority"]
                rs = i18n.exam_reason(x["reason"]) if FR else x["reason"]
                pool.append((hitn, t("recommended: ", "recommandé : ")
                             + f"{ex} ({pr}) — {rs}"))
        for h in hits:
            hitn = sum(1 for w in words if w in (h["topic"] + " " + h["text"]).lower())
            if hitn:
                pool.append((hitn, f"[{h['n']}] {h['topic']} — {h['text']} ({h['source']})"))
        for x in state["imaging"]["out_of_scope"]:
            hitn = sum(1 for w in words if w in x.lower())
            if hitn:
                pool.append((hitn, t("declared limit: ", "limite déclarée : ")
                             + (i18n.limit(x) if FR else x)))
        gonehit = [g for g in gone if any(w in g.lower() for w in words)]
        for g in gonehit:
            gname = i18n.measure(g) if FR else g
            pool.append((2, t(f"{gname} was never measured — absent, not normal, and it "
                              f"carries no evidence identifier",
                              f"{gname} n'a jamais été mesuré — absent, pas normal, et sans "
                              f"identifiant de preuve")))

        if pool:
            pool.sort(key=lambda p: -p[0])
            seen, lines = set(), []
            for _, text in pool:
                if text not in seen:
                    seen.add(text)
                    lines.append(text)
                if len(lines) >= 6:
                    break
            body = "\n".join("• " + x for x in lines)
            return t("From this encounter's record, the parts that bear on what you asked:"
                     "\n\n" + body + "\n\nThat is what the record holds on it. I have not "
                     "added anything to it.",
                     "D'après le dossier de cette prise en charge, voici ce qui concerne "
                     "votre question :\n\n" + body + "\n\nC'est ce que le dossier contient "
                     "à ce sujet. Je n'y ai rien ajouté.")

    # Nothing in the record matched the words. That is a reason to say what the record DOES
    # hold, not to recite a menu: a list of capabilities answers a question nobody asked, and
    # reads as a broken assistant at the exact moment someone is trying to use it. The
    # standing summary is real -- every number below is computed -- and it is usually what was
    # wanted anyway. The refusal to invent stands; it just no longer arrives empty-handed.
    sev_txt = i18n.severity(sup["severity"]["severity"]) if FR \
        else sup["severity"]["severity"]
    esc_txt = (t("escalation required", "escalade requise") if esc["escalate"]
               else t("no escalation trigger", "aucun déclencheur d'escalade"))
    lines = [t(f"Severity {sev_txt}, {len(sup['alerts'])} alert(s), {esc_txt}.",
               f"Sévérité {sev_txt}, {len(sup['alerts'])} alerte(s), {esc_txt}.")]
    if det:
        lines.append(t("POCUS: ", "POCUS : ")
                     + ", ".join(f"{f['label']} {f['confidence']:.2f}" for f in det))
    else:
        lines.append(t("POCUS: no finding above threshold.",
                       "POCUS : aucun signe au-dessus du seuil."))
    if gone:
        gv = ", ".join((i18n.measure(g) if FR else g) for g in gone[:6])
        lines.append(t(f"{len(gone)} value(s) never measured: " + gv
                       + ("…" if len(gone) > 6 else "") + " — absent, not normal.",
                       f"{len(gone)} valeur(s) jamais mesurée(s) : " + gv
                       + ("…" if len(gone) > 6 else "") + " — absent, pas normal."))
    r = sup["additional_examinations"]
    if r:
        recs = "; ".join(
            (i18n.exam_name(x["exam"]) if FR else x["exam"]) + " ("
            + (i18n.PRIORITY.get(x["priority"], x["priority"]) if FR else x["priority"]) + ")"
            for x in r[:3])
        lines.append(t("Recommended next: ", "Recommandé ensuite : ") + recs)
    body = "\n".join("• " + x for x in lines)
    return t("I could not match that to anything specific in the record, so rather than write "
             "something plausible around the gap, here is where this encounter stands:\n\n"
             + body
             + "\n\nAsk for a value, what is missing, why the severity is what it is, what "
               "the scan saw, or the evidence for a diagnosis, and I will read it out exactly.",
             "Je n'ai pas pu rattacher cela à quelque chose de précis dans le dossier ; plutôt "
             "que d'écrire quelque chose de plausible autour du vide, voici où en est cette "
             "prise en charge :\n\n" + body
             + "\n\nDemandez-moi une valeur, ce qui manque, pourquoi cette sévérité, ce que "
               "l'échographie a vu, ou les éléments en faveur d'un diagnostic, et je vous les "
               "lirai exactement.")
