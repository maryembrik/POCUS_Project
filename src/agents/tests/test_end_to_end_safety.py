"""End-to-end safety: three encounters carried from agent output to escalation decision.

These are the worked scenarios of the report's integration chapter, run as tests. Each has a
clinically motivated expected behaviour, and the point is that the behaviour follows from
the structured state rather than from anything a model noticed.
"""
import json

from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state, render_state
from src.agents.clinical.reasoning import escalation_decision, reason
from .helpers import (CASE_QUALITY, CONFLICT, ESCALATION, MISSING_NOT_NORMAL,
                      prop, bundle, heart_report, lung_report, llm_output)


# ------------------------------------------------------------------ scenario 1: concordant
def _concordant():
    rep = S.make_report("heart", [S.make_finding("normal_function", 0.93)],
                        reliability={"confidence_calibrated": True,
                                     "has_normal_class": True})
    return build_clinical_state(
        bundle(triage=S.make_triage("low", 0.91, features={"o2sat": 98, "pulse": 74}),
               ultrasound={"heart": rep}, encounter_id="E2E-CONCORDANT"),
        labs={"troponin": 5.0, "lactate": 1.0})


@prop(ESCALATION)
def test_concordant_case_is_answered_directly():
    d = escalation_decision(_concordant())
    assert d["escalate"] is False, d["triggers"]


@prop(CASE_QUALITY)
def test_concordant_case_is_graded_good():
    assert _concordant()["case_quality"]["grade"] == "GOOD", \
        _concordant()["case_quality"]


# ------------------------------------------------------------------ scenario 2: conflicting
def _conflicting():
    return build_clinical_state(
        bundle(triage=S.make_triage("low", 0.88, features={"o2sat": 96, "pulse": 82}),
               ultrasound={"heart": heart_report(label="severe_dysfunction",
                                                 confidence=0.74)},
               encounter_id="E2E-CONFLICT"),
        labs={"troponin": 5.0, "lactate": 1.0})


@prop(CONFLICT)
def test_conflicting_case_surfaces_the_disagreement():
    st = _conflicting()
    assert st["conflicts"], "the disagreement must be recorded, not resolved"


@prop(ESCALATION)
def test_conflicting_case_escalates_rather_than_picking_a_side():
    d = escalation_decision(_conflicting())
    assert d["escalate"] is True
    assert d["route"] == "simulation"


# ------------------------------------------------------------------ scenario 3: missing data
def _missing():
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79, features={"o2sat": 90, "pulse": 118}),
               ultrasound={"lung": lung_report(b_lines=0.86)},
               encounter_id="E2E-MISSING"),
        labs={})


@prop(MISSING_NOT_NORMAL)
def test_missing_case_records_the_absent_labs():
    st = _missing()
    assert st["missing"]["labs"], st["missing"]


@prop(ESCALATION)
def test_missing_case_escalates_on_the_absent_key_lab():
    d = escalation_decision(_missing())
    assert d["escalate"] is True
    assert any("key lab" in t for t in d["triggers"]), d["triggers"]


@prop(MISSING_NOT_NORMAL)
def test_missing_case_bars_the_model_from_citing_the_absent_lab():
    st = _missing()
    bad = json.dumps(llm_output(["b_lines", "troponin negative"]))
    out = reason(st, llm_fn=lambda s, u: bad)
    assert out["differential_withheld"] is True, out


# ------------------------------------------------------------------ the prompt itself
@prop(MISSING_NOT_NORMAL)
def test_prompt_states_when_no_evidence_was_retrieved():
    """With RAG absent, the prompt must say so rather than omitting the fact, or the model
    may imply guideline grounding it never had."""
    out = reason(_missing(), llm_fn=None, retrieved=None)
    assert "NO RETRIEVED EVIDENCE" in out["prompt"]["user"]


@prop(MISSING_NOT_NORMAL)
def test_rendered_state_carries_the_model_limits_into_the_prompt():
    text = render_state(_missing())
    assert "NOT MEASURED" in text
    assert "LIMITS" in text or "cannot" in text.lower()
