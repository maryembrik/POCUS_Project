"""Calibration and evidence coverage: grounded answers that are still not defensible.

These checks are separated from the hallucination guard on purpose. A fabricated laboratory
value is fatal and the answer is withheld. A likelihood the evidence does not support is a
judgement the model can be asked to reconsider -- and asking is better than rewriting, since
a downgrade applied by Python would be read as the model's reasoning when it is not.
"""
import json

from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from src.agents.llm import ScriptedBackend
from src.agents.reasoning import (check_confidence, check_evidence_coverage, reason,
                                  _decisive_for)
from .helpers import prop, bundle, lung_report

CALIBRATION = "Confidence calibration"
COVERAGE = "Evidence coverage"


def _state(labs=None, o2sat=90):
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79,
                                    features={"o2sat": o2sat, "pulse": 118, "respr": 24}),
               ultrasound={"lung": lung_report(b_lines=0.86)}),
        labs=labs if labs is not None else {})


def _answer(diagnosis, likelihood, supporting, uncertainty="limited evidence"):
    return {"differential": [{"diagnosis": diagnosis, "likelihood": likelihood,
                              "supporting": supporting, "contradicting": [],
                              "limitations": []}],
            "missing_information": ["troponin"],
            "uncertainty": uncertainty,
            "recommended_next_step": "obtain troponin"}


# ------------------------------------------------------------------ diagnosis awareness
@prop(CALIBRATION)
def test_decisive_tests_are_diagnosis_specific():
    """Blanket rules would be too blunt: a missing D-dimer says nothing about the confidence
    owed to a pneumothorax."""
    assert "d_dimer" in _decisive_for("Pulmonary embolism")
    assert "troponin" in _decisive_for("Acute coronary syndrome")
    assert "bnp" in _decisive_for("Pulmonary oedema")
    assert _decisive_for("Pneumothorax") == set()


@prop(CALIBRATION)
def test_high_confidence_with_the_decisive_test_absent_is_flagged():
    """The exact failure observed with prompt v2: pulmonary oedema rated 'high' with BNP and
    troponin never drawn."""
    errs = check_confidence(_answer("Pulmonary Edema", "high", ["b lines"]), _state())
    assert errs, "high on an unconfirmed diagnosis must be challenged"
    assert "moderate" in errs[0]


@prop(CALIBRATION)
def test_moderate_on_the_same_evidence_is_accepted():
    assert check_confidence(_answer("Pulmonary Edema", "moderate", ["b lines"]),
                            _state()) == []


@prop(CALIBRATION)
def test_high_is_accepted_once_the_decisive_test_is_available():
    """The rule is about absent evidence, not about forbidding confidence."""
    st = _state(labs={"bnp": 890.0, "troponin": 62.0, "lactate": 1.1})
    assert check_confidence(_answer("Pulmonary Edema", "high", ["b lines", "bnp"]),
                            st) == []


@prop(CALIBRATION)
def test_an_unrelated_missing_lab_does_not_block_high():
    """Creatinine being absent is irrelevant to a pneumothorax, and must not suppress a
    confident answer about one."""
    assert check_confidence(_answer("Pneumothorax", "high", ["b lines"]), _state()) == []


# ------------------------------------------------------------------ evidence coverage
@prop(COVERAGE)
def test_an_unused_abnormal_vital_is_flagged():
    """Prompt v2 cited heart rate and respiratory rate but silently dropped SpO2 90."""
    ans = _answer("Pulmonary Edema", "moderate",
                  ["b lines (0.86)", "heart rate 118", "respiratory rate 24"])
    errs = check_evidence_coverage(ans, _state())
    assert errs, "an unused abnormal value must be challenged"
    assert "spo2" in errs[0]


@prop(COVERAGE)
def test_citing_the_value_counts_as_using_it():
    """'90%' cites the saturation as surely as the word 'saturation' does."""
    ans = _answer("Pulmonary Edema", "moderate",
                  ["b lines", "heart rate 118", "respiratory rate 24", "oxygen 90%"])
    assert check_evidence_coverage(ans, _state()) == []


@prop(COVERAGE)
def test_dismissing_a_value_in_uncertainty_counts_as_accounting_for_it():
    """An abnormal value may legitimately be irrelevant -- but that has to be said."""
    ans = _answer("Pulmonary Edema", "moderate",
                  ["b lines", "heart rate 118", "respiratory rate 24"],
                  uncertainty="hypoxia at 90% is consistent with several causes and does "
                              "not discriminate between them")
    assert check_evidence_coverage(ans, _state()) == []


@prop(COVERAGE)
def test_normal_values_need_no_mention():
    """Only HIGH and LOW values require accounting; demanding every normal value be discussed
    would make the requirement noise."""
    st = _state(o2sat=98)
    ans = _answer("Pulmonary Edema", "moderate",
                  ["b lines", "heart rate 118", "respiratory rate 24"])
    assert check_evidence_coverage(ans, st) == []


# ------------------------------------------------------------------ the revision loop
@prop(CALIBRATION)
def test_a_revisable_answer_is_sent_back_before_being_rejected():
    st = _state()
    bad = json.dumps(_answer("Pulmonary Edema", "high",
                             ["b lines", "heart rate 118", "respiratory rate 24",
                              "oxygen 90%"]))
    good = json.dumps(_answer("Pulmonary Edema", "moderate",
                              ["b lines", "heart rate 118", "respiratory rate 24",
                               "oxygen 90%"]))
    backend = ScriptedBackend(bad, good)
    out = reason(st, llm_fn=backend, max_revisions=1)
    assert len(backend.calls) == 2, "the model should have been asked to revise"
    assert out["validation_errors"] is None, out["validation_errors"]
    assert out["differential"]["differential"][0]["likelihood"] == "moderate"
    assert out["revisions"], "the revision should be recorded"


@prop(CALIBRATION)
def test_the_revision_request_names_the_specific_complaint():
    st = _state()
    bad = json.dumps(_answer("Pulmonary Edema", "high", ["b lines", "hr 118", "rr 24",
                                                         "spo2 90"]))
    backend = ScriptedBackend(bad)
    reason(st, llm_fn=backend, max_revisions=1)
    _, revision_user = backend.calls[1]
    assert "rejected" in revision_user.lower()
    assert "moderate" in revision_user


@prop(CALIBRATION)
def test_python_never_rewrites_the_models_answer():
    """If the model will not revise, the differential is withheld and the reason reported. A
    likelihood edited in post would be presented to a clinician as the model's judgement."""
    st = _state()
    bad = json.dumps(_answer("Pulmonary Edema", "high", ["b lines", "hr 118", "rr 24",
                                                         "spo2 90"]))
    out = reason(st, llm_fn=ScriptedBackend(bad, bad), max_revisions=1)
    assert out["differential_withheld"] is True
    assert out["differential"]["differential"][0]["likelihood"] == "high", \
        "the model's answer must be reported unmodified"
    assert out["validation_errors"]


@prop(CALIBRATION)
def test_a_fabrication_is_not_offered_a_revision():
    """Grounding failures are fatal, not a matter of degree: no retry is spent on them."""
    st = _state()
    fabricated = json.dumps(_answer("Acute coronary syndrome", "moderate",
                                    ["elevated troponin"]))
    backend = ScriptedBackend(fabricated)
    out = reason(st, llm_fn=backend, max_revisions=1)
    assert len(backend.calls) == 1, "a fabrication must not trigger a revision round"
    assert out["differential_withheld"] is True
