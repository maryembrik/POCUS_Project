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
from src.agents.assistant import answer, next_step_answer
from src.agents.clinical.clinical_state import build_clinical_state
from src.agents.clinical.decision_support import decision_support
from src.agents.clinical.reasoning import escalation_decision, reason
from src.agents.clinical.report import build_report
from .helpers import prop, ESCALATION, MISSING_NOT_NORMAL, SCOPE

ROUTER = "Assistant routing"
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
