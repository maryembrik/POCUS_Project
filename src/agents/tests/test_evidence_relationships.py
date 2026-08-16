"""Evidence relationships and the unsound/untidy split.

Both come from the same observation: the model handles the output format better than it
handles what each ultrasound sign means. Neither check tries to supply medical reasoning --
the first refuses the clearest misuses, the second decides what a residual fault costs.
"""
import json

from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from src.agents.llm import ScriptedBackend
from src.agents.reasoning import (check_evidence_relationships, check_recommendation_wording,
                                  reason)
from .helpers import prop, bundle

RELATIONSHIP = "Evidence relationships"
SEVERITY = "Failure severity"
SCOPE_ADVICE = "Advice scope"


def _lung_state():
    rep = S.make_report(
        "lung", [S.make_finding("b lines", 0.86)],
        not_detected=[S.make_finding("consolidation", 0.09),
                      S.make_finding("pleural effusion", 0.04),
                      S.make_finding("pleural thickening", 0.11)],
        reliability={"confidence_calibrated": True, "has_normal_class": False,
                     "modelled_findings": ["b_lines", "consolidation",
                                           "pleural_effusion", "pleural_thickening"]})
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79,
                                    features={"o2sat": 90, "pulse": 118, "respr": 24}),
               ultrasound={"lung": rep}), labs={})


def _entry(diagnosis, contradicting=(), likelihood="low"):
    return {"differential": [{"diagnosis": diagnosis, "likelihood": likelihood,
                              "supporting": ["b lines (0.86)", "hr 118", "rr 24",
                                             "spo2 90"],
                              "contradicting": list(contradicting), "limitations": []}],
            "missing_information": ["troponin"], "uncertainty": "u",
            "recommended_next_step": "obtain a troponin"}


# ------------------------------------------------------------------ the observed misuse
@prop(RELATIONSHIP)
def test_absent_pleural_thickening_does_not_argue_against_embolism():
    """Observed: 'pleural thickening NOT detected' offered as evidence against pulmonary
    embolism. The lung module not seeing pleural thickening says nothing about a PE."""
    errs = check_evidence_relationships(
        _entry("Pulmonary Embolism",
               contradicting=["lung: pleural thickening NOT detected (0.11)"]),
        _lung_state())
    assert errs, "an unrelated absent finding must not count as contradicting evidence"
    assert "does not bear" in errs[0]


@prop(RELATIONSHIP)
def test_a_genuine_relationship_is_left_alone():
    """B-lines argue against pneumothorax -- they require pleural apposition. That is a real
    relationship and must not be flagged."""
    assert check_evidence_relationships(
        _entry("Pneumothorax", contradicting=["b lines (0.86)"]), _lung_state()) == []


@prop(RELATIONSHIP)
def test_absent_consolidation_may_argue_against_pneumonia():
    """A finding the module screened for, and that the diagnosis would predict, is a
    legitimate argument."""
    assert check_evidence_relationships(
        _entry("Pneumonia", contradicting=["consolidation NOT detected (0.09)"]),
        _lung_state()) == []


@prop(RELATIONSHIP)
def test_an_unknown_diagnosis_is_not_judged():
    """The map is small on purpose. Silence means no opinion, never 'unrelated'."""
    assert check_evidence_relationships(
        _entry("Aortic dissection",
               contradicting=["lung: pleural thickening NOT detected (0.11)"]),
        _lung_state()) == []


@prop(RELATIONSHIP)
def test_supporting_evidence_is_not_second_guessed():
    """Only contradicting claims are checked. Deciding what may SUPPORT a diagnosis is the
    reasoning this layer deliberately does not attempt."""
    assert check_evidence_relationships(
        _entry("Pulmonary Embolism"), _lung_state()) == []


# ------------------------------------------------------------------ recommendation wording
@prop(SCOPE_ADVICE)
def test_claiming_a_test_will_rule_something_out_is_flagged():
    st = _lung_state()
    out = _entry("Pulmonary Edema")
    out["recommended_next_step"] = "Obtain troponin and lactate to rule out a heart attack"
    assert check_recommendation_wording(out, st)


@prop(SCOPE_ADVICE)
def test_stating_what_a_test_would_inform_is_accepted():
    st = _lung_state()
    out = _entry("Pulmonary Edema")
    out["recommended_next_step"] = ("Obtain troponin and lactate, and reassess alongside the "
                                    "abnormal oxygen saturation")
    assert check_recommendation_wording(out, st) == []


# ------------------------------------------------------------------ severity split
@prop(SEVERITY)
def test_an_untidy_answer_is_delivered_with_a_warning():
    """A correct differential must not be suppressed because a list was joined with commas.
    An earlier version did exactly that to the concordant case."""
    st = _lung_state()
    out = _entry("Pulmonary Edema", likelihood="moderate")
    out["missing_information"] = ["troponin, lactate"]          # untidy only
    raw = json.dumps(out)
    res = reason(st, llm_fn=ScriptedBackend(raw, raw), max_revisions=1)
    assert res.get("differential_withheld") is not True, res["validation_errors"]
    assert res["differential"] is not None
    assert res["warnings"], "the fault must still be reported"


@prop(SEVERITY)
def test_an_unsound_answer_is_still_withheld():
    """A misread value corrupts the reasoning and is not a presentation problem."""
    st = build_clinical_state(
        bundle(triage=S.make_triage("low", 0.88, features={"pulse": 82}),
               ultrasound={"heart": S.make_report(
                   "heart", [S.make_finding("severe dysfunction", 0.74)],
                   reliability={"confidence_calibrated": True,
                                "has_normal_class": True})}),
        labs={"troponin": 5.0, "lactate": 1.1})
    out = {"differential": [{"diagnosis": "Myocardial infarction", "likelihood": "moderate",
                             "supporting": ["high troponin level (5.0 ng/L)"],
                             "contradicting": [], "limitations": []}],
           "missing_information": ["bnp"], "uncertainty": "u",
           "recommended_next_step": "obtain a BNP"}
    raw = json.dumps(out)
    res = reason(st, llm_fn=ScriptedBackend(raw, raw), max_revisions=1)
    assert res["differential_withheld"] is True
    assert any("troponin" in e for e in res["validation_errors"])


@prop(SEVERITY)
def test_a_revision_request_carries_one_complaint_not_all_of_them():
    """Measured on a real case: one complaint was fixed, three were fixed in no respect at
    all. The unsound fault is sent, because it is the one deciding whether the answer can be
    shown; the untidy fault becomes a warning either way."""
    st = _lung_state()
    out = _entry("Pulmonary Embolism",
                 contradicting=["lung: pleural thickening NOT detected (0.11)"])
    out["missing_information"] = ["troponin, lactate"]
    backend = ScriptedBackend(json.dumps(out))
    res = reason(st, llm_fn=backend, max_revisions=1)
    _, revision = backend.calls[1]
    assert "does not bear" in revision, "the unsound fault must be the one sent"
    assert "one test per" not in revision, "the untidy fault must not crowd the request"
    # Nothing is lost: everything found is still recorded.
    assert res["revisions"][0]["also_found"], res["revisions"]
