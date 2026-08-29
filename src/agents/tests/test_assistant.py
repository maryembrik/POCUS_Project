"""The clinical assistant's router, and the boundary the ACTION answer must not cross.

The assistant is not a language model. It reads a question for what it is asking about and
replies from the computed assessment, so what is worth testing is not fluency but the two
things a reader cannot check for themselves: that a question a clinician would actually type
reaches the right answer, and that answering "what should I do now?" never turns into stating
a diagnosis or prescribing.

The second is the reason the ACTION layer can exist without a model at all. "Complete the
observations" is safe to say from a record; "this is heart failure" is not.
"""
from src.agents import schema as S
from src.agents.assistant import answer, knowledge_answer, next_step_answer
from src.agents.clinical.clinical_state import build_clinical_state
from src.agents.clinical.decision_support import decision_support
from src.agents.clinical.reasoning import escalation_decision, reason
from src.agents.clinical.report import build_report
from .helpers import prop, ESCALATION, MISSING_NOT_NORMAL, SCOPE

ROUTER = "Assistant routing"
KNOWLEDGE = "Assistant knowledge boundary"
ACTION = "Assistant action boundary"


def _analysis(*, b_lines: float | None = 0.86, labs: dict | None = None,
              vitals: dict | None = None, unassessed: tuple[str, ...] = ("heart",)) -> dict:
    """The dict the pipeline hands the assistant, built the way serve.py builds it."""
    findings = [S.make_finding("b lines", b_lines)] if b_lines is not None else []
    reports = {"lung": S.make_report(
        "lung", findings,
        not_detected=[] if findings else [S.make_finding("b lines", 0.04)],
        # modelled_findings is what makes the pneumothorax limit derivable: the state adds
        # the line because the label set does NOT contain it, not because a string says so.
        reliability={"confidence_calibrated": True, "has_normal_class": False,
                     "modelled_findings": ["b_lines", "consolidation", "pleural_effusion",
                                           "pleural_thickening"],
                     "scope": "pneumothorax is NOT modelled and cannot be excluded"})}
    for organ in unassessed:
        reports[organ] = S.make_report(organ, [], status="not_supported",
                                       reliability={"scope": "requested but never assessed"})
    state = build_clinical_state(
        {"encounter_id": "ENC-TEST",
         "triage": S.make_triage("high", 0.8, features=vitals or {}),
         "ultrasound": reports,
         "clinical": {"age": 71, "sex": "F", "chief_complaint": "acute breathlessness"}},
        labs=labs or {})
    esc = escalation_decision(state)
    result = reason(state, llm_fn=None)
    sup = decision_support(state, esc, (result.get("differential") or {}).get("differential"))
    result["decision_support"] = sup
    return {"state": state, "esc": esc, "support": sup, "hits": [], "result": result,
            "report": build_report(state, result, sup), "marks": [], "origin": "not_generated"}


# --------------------------------------------------------------------------- routing
@prop(ROUTER)
def test_next_step_survives_a_typo():
    """One missing letter used to drop the commonest question into the catch-all."""
    a = _analysis()
    for q in ("what shoud i do now??", "what should I do now?", "what do i do next",
              "what now?", "what should i assess next"):
        out = answer(q, a)
        assert "WHAT TO OBTAIN NEXT" in out, f"{q!r} did not reach the action answer"


@prop(ROUTER)
def test_greeting_is_not_answered_with_a_refusal():
    assert answer("hi", _analysis()).startswith("Hello")


# ---------------------------------------------------------------------- action content
@prop(MISSING_NOT_NORMAL)
def test_action_names_what_was_never_measured_as_absent():
    out = next_step_answer(_analysis(labs={"troponin": 62.0}))
    assert "Never measured" in out
    assert "absent, not normal" in out
    assert "bnp" in out


@prop(SCOPE)
def test_action_carries_the_limit_the_model_cannot_settle():
    out = next_step_answer(_analysis())
    assert "WHAT THIS CANNOT SETTLE" in out
    assert "pneumothorax" in out.lower()


@prop(ESCALATION)
def test_action_states_escalation_and_its_triggers():
    out = next_step_answer(_analysis())
    assert "ESCALATION" in out
    assert ("Escalation is required" in out) or ("No escalation trigger fired" in out)


@prop(SCOPE)
def test_action_does_not_call_a_negative_scan_a_well_patient():
    """The lung module has no healthy class; nothing detected is not nothing wrong."""
    out = next_step_answer(_analysis(b_lines=None))
    assert "not the absence of pathology" in out


@prop(ROUTER)
def test_action_never_assessed_is_reported_before_a_missing_lab():
    """A scan that did not happen is the hole most often read as a negative result."""
    out = next_step_answer(_analysis(labs={}))
    assert out.index("Never assessed") < out.index("Never measured")


# --------------------------------------------------------------------- the boundary
@prop(ACTION)
def test_action_answer_never_states_a_diagnosis_or_a_drug():
    """An ACTION is not a diagnosis. This is what lets it be answered without a model.

    The assistant may say what to obtain and when to escalate. It may not conclude what the
    patient has, or what to give them -- there is no evidence identifier behind either, and
    the sentence would be produced after the validator has finished and checked by nothing.
    """
    out = next_step_answer(_analysis(labs={"troponin": 62.0, "bnp": 890.0})).lower()
    for claim in ("the patient has", "diagnosis is", "this is heart failure",
                  "consistent with heart failure", "start ", "administer", "give ",
                  "prescribe", "mg", "diagnosed"):
        assert claim not in out, f"action answer stated {claim!r}"


@prop(ACTION)
def test_no_therapeutic_consideration_without_a_sourced_protocol():
    out = next_step_answer(_analysis())
    assert "THERAPEUTIC CONSIDERATIONS" in out
    assert "no approved protocol available" in out


@prop(ACTION)
def test_action_cites_no_source_when_nothing_was_retrieved():
    """A retrieved-evidence heading with nothing under it implies grounding it never had."""
    assert "RETRIEVED FOR THIS PRESENTATION" not in next_step_answer(_analysis())


# ------------------------------------------------------------------------- knowledge
@prop(KNOWLEDGE)
def test_definition_is_answered_from_the_corpus_not_the_encounter():
    """"What are B-lines?" asks what the sign means, not what this patient's confidence was."""
    out = answer("what are b lines?", _analysis())
    assert "quoted from the corpus" in out
    assert "NOT a statement about this patient" in out
    assert "Source:" in out


@prop(KNOWLEDGE)
def test_a_patient_value_is_never_answered_with_a_textbook():
    """The dangerous direction. "What is the troponin?" is about this patient."""
    out = answer("what is the troponin?", _analysis(labs={"troponin": 62.0}))
    assert "62" in out
    assert "quoted from the corpus" not in out


@prop(KNOWLEDGE)
def test_reference_and_encounter_are_separated_not_merged():
    out = answer("what are b lines?", _analysis())
    assert out.index("quoted from the corpus") < out.index("In THIS encounter")
    assert "computed rather than quoted" in out


@prop(KNOWLEDGE)
def test_a_term_the_corpus_does_not_cover_is_not_answered_from_it():
    """Nothing clears the relevance floor, so it must fall through rather than reach."""
    assert knowledge_answer("what is aspirin dosing") is None
    assert knowledge_answer("what is the capital of France") is None


@prop(KNOWLEDGE)
def test_french_phrasing_without_the_apostrophe_still_finds_it():
    """A clinician types "c'est quoi" or "c est quoi"; punctuation must not decide."""
    for q in ("c'est quoi le pneumothorax", "c est quoi le pneumothorax"):
        out = knowledge_answer(q)
        assert out and "quoted from the corpus" in out, q


# -------------------------------------------------------------------------- French
FRENCH = "Assistant French routing"


@prop(FRENCH)
def test_a_french_question_reaches_the_intent_not_the_fallback():
    """The cues, not the answers, are what decide whether French works at all.

    An assistant that recognises only English keywords answers every French question with its
    fallback while appearing to work — the worst kind of broken, because it still replies.
    """
    a = _analysis(labs={"troponin": 62.0})
    for q, must in (
            ("quelles valeurs manquent ?", "Jamais mesuré"),
            ("pourquoi cette sévérité ?", "La sévérité est"),
            ("quelles sont les alertes ?", "Alertes"),
            ("que dois-je faire maintenant ?", "CE QU'IL FAUT OBTENIR ENSUITE"),
            ("quelle est la troponine ?", "troponine"),
            ("qu'a vu l'échographie ?", "L'échographie a détecté"),
            ("va-t-elle survivre ?", "ne prédit pas"),
            ("bonjour", "Bonjour")):
        out = answer(q, a, "fr")
        assert must in out, f"{q!r} -> {out[:110]!r}"


@prop(FRENCH)
def test_the_french_assistant_leaves_nothing_untranslated():
    """Every branch a French question can reach must render fully in French."""
    from src.agents import i18n

    i18n.reset_untranslated()
    a = _analysis(labs={"troponin": 62.0})
    for q in ("quelles valeurs manquent ?", "pourquoi cette sévérité ?",
              "quelles sont les alertes ?", "que dois-je faire maintenant ?",
              "quelles sont les limites ?", "résume ce patient", "quelle est la qualité ?",
              "quels sont les conflits ?", "quel scénario ?", "les constantes ?",
              "la biologie ?", "le différentiel ?", "quelles sources ?"):
        answer(q, a, "fr")
    assert i18n.untranslated() == [], i18n.untranslated()


@prop(FRENCH)
def test_the_prognosis_boundary_holds_in_french():
    """The refusal must survive translation. A boundary that only exists in one language is
    not a boundary."""
    out = answer("va-t-elle s en sortir ?", _analysis(), "fr").lower()
    assert "ne prédit pas" in out
    assert "pronostic revient au clinicien" in out


@prop(FRENCH)
def test_a_quoted_passage_is_not_translated():
    """A translated quotation is no longer a quotation, and would carry a citation to a source
    that never said it. The frame is French; the passage stays as published."""
    out = knowledge_answer("c est quoi le pneumothorax", None, "fr")
    assert out is not None
    assert "citée mot pour mot" in out
    assert "Source :" in out


@prop(KNOWLEDGE)
def test_a_whole_sentence_is_not_treated_as_a_term():
    """"What is the most likely diagnosis for this breathless patient" is not a definition."""
    assert knowledge_answer("what is the most likely diagnosis for this patient") is None
