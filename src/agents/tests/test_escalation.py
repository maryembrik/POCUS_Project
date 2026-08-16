"""Escalation policy: computed from the state before the model runs.

Every trigger below is a condition the structured state already knows about, so none of it
depends on a language model having noticed anything. That is the property being tested.
"""
from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state
from src.agents.clinical.reasoning import escalation_decision
from .helpers import ESCALATION, prop, bundle, heart_report, lung_report


def _state(**kw):
    return build_clinical_state(bundle(**{k: v for k, v in kw.items()
                                          if k in ("triage", "ultrasound", "clinical")}),
                                labs=kw.get("labs"))


@prop(ESCALATION)
def test_agent_disagreement_escalates():
    st = _state(triage=S.make_triage("low", 0.85),
                ultrasound={"heart": heart_report(confidence=0.71)},
                labs={"troponin": 5.0, "lactate": 1.0})
    d = escalation_decision(st)
    assert d["escalate"] is True
    assert d["route"] == "simulation"
    assert any("disagree" in t for t in d["triggers"]), d["triggers"]


@prop(ESCALATION)
def test_positive_imaging_with_key_lab_absent_escalates():
    """A positive finding without the laboratory value that would confirm or refute it is
    precisely the case a clinician should be asked to look at."""
    st = _state(triage=S.make_triage("high", 0.9),
                ultrasound={"lung": lung_report(b_lines=0.88)},
                labs={})
    d = escalation_decision(st)
    assert d["escalate"] is True
    assert any("key lab" in t for t in d["triggers"]), d["triggers"]


@prop(ESCALATION)
def test_high_risk_finding_on_weak_evidence_escalates():
    st = _state(triage=S.make_triage("high", 0.9),
                ultrasound={"heart": heart_report(label="severe_dysfunction",
                                                  confidence=0.62,
                                                  calibrated=False)},
                labs={"troponin": 5.0, "lactate": 1.0})
    d = escalation_decision(st)
    assert d["escalate"] is True
    assert any("high-risk" in t for t in d["triggers"]), d["triggers"]


@prop(ESCALATION)
def test_organ_never_assessed_prevents_a_direct_answer():
    """A scan that did not happen is not a negative scan. Without this the system answers
    directly on the strength of imaging it never performed."""
    rep = S.make_report("lung", [], status="not_supported")
    st = _state(triage=S.make_triage("high", 0.9),
                ultrasound={"lung": rep},
                labs={"troponin": 5.0, "lactate": 1.0})
    d = escalation_decision(st)
    assert d["escalate"] is True, d
    assert any("not assessed" in t for t in d["triggers"]), d["triggers"]


@prop(ESCALATION)
def test_no_finding_but_models_cannot_exclude_disease_escalates():
    """A negative read from a module with no healthy class is not a negative result."""
    rep = S.make_report("lung", [S.make_finding("b_lines", 0.05)],
                        reliability={"confidence_calibrated": True,
                                     "has_normal_class": False,
                                     "modelled_findings": ["b_lines"]})
    rep["findings"] = []
    rep["status"] = "ok"
    st = _state(triage=S.make_triage("high", 0.9),
                ultrasound={"lung": rep},
                labs={"troponin": 5.0, "lactate": 1.0})
    d = escalation_decision(st)
    assert d["escalate"] is True, d
    assert any("cannot exclude" in t for t in d["triggers"]), d["triggers"]


@prop(ESCALATION)
def test_poor_case_quality_escalates():
    st = _state(triage=None,
                ultrasound={"lung": lung_report(b_lines=0.3, calibrated=False)},
                labs={})
    d = escalation_decision(st)
    assert d["escalate"] is True
    assert any("POOR" in t for t in d["triggers"]), d["triggers"]


@prop(ESCALATION)
def test_a_clean_concordant_case_is_answered_directly():
    """The policy must be able to say no. A system that escalates everything provides no
    decision support at all."""
    rep = S.make_report(
        "heart",
        [S.make_finding("normal_function", 0.93)],
        reliability={"confidence_calibrated": True, "has_normal_class": True},
    )
    st = _state(triage=S.make_triage("low", 0.91),
                ultrasound={"heart": rep},
                labs={"troponin": 5.0, "lactate": 1.0})
    d = escalation_decision(st)
    assert d["escalate"] is False, d["triggers"]
    assert d["route"] == "direct"


@prop(ESCALATION)
def test_escalation_does_not_depend_on_a_model_being_present():
    """The decision is computed from the state alone; reason() must return it with no
    llm_fn supplied."""
    from src.agents.clinical.reasoning import reason
    st = _state(triage=S.make_triage("low", 0.85),
                ultrasound={"heart": heart_report(confidence=0.71)},
                labs={})
    out = reason(st, llm_fn=None)
    assert out["escalation"]["escalate"] is True
    assert out["differential"] is None
    assert "prompt" in out
