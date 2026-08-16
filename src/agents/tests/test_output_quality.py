"""Failures found in real GPU runs: paraphrase, misread values, treatment advice, atomicity.

Every case here reproduces something HuatuoGPT-o1 actually produced. They are separated from
the hallucination guard because none of them is a fabrication -- the evidence cited is real.
What is wrong is how it was matched, read, or acted on.
"""
from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from src.agents.reasoning import (check_atomicity, check_evidence_coverage,
                                  check_missing_information_accuracy,
                                  check_scope_of_advice, check_value_qualifiers,
                                  validate_llm_output)
from .helpers import HALLUCINATION, prop, bundle, heart_report, lung_report

QUALIFIER = "Value-reading consistency"
SCOPE_ADVICE = "Advice scope"
COVERAGE = "Evidence coverage"


def _conflict_state():
    return build_clinical_state(
        bundle(triage=S.make_triage("low", 0.88,
                                    features={"pulse": 82, "bpsys": 128, "o2sat": 96}),
               ultrasound={"heart": heart_report(label="severe dysfunction",
                                                 confidence=0.74)}),
        labs={"troponin": 5.0, "lactate": 1.1})


def _answer(**kw):
    base = {"differential": [{"diagnosis": "Acute coronary syndrome", "likelihood": "low",
                              "supporting": [], "contradicting": [], "limitations": []}],
            "missing_information": ["bnp"], "uncertainty": "u",
            "recommended_next_step": "obtain a BNP"}
    for k, v in kw.items():
        if k in ("supporting", "contradicting"):
            base["differential"][0][k] = v
        else:
            base[k] = v
    return base


# ------------------------------------------------------------------ paraphrase tolerance
@prop(HALLUCINATION)
def test_a_paraphrased_finding_is_not_a_fabrication():
    """The state said 'severe dysfunction'; the model wrote 'severe heart dysfunction' and the
    guard withheld the whole answer. A guard that blocks correct answers gets switched off."""
    st = _conflict_state()
    assert validate_llm_output(_answer(supporting=["severe heart dysfunction (0.74)"]),
                               st) == []


@prop(HALLUCINATION)
def test_citing_the_triage_assessment_is_not_a_fabrication():
    """'low urgency (confidence 0.88)' is the triage output, restated."""
    st = _conflict_state()
    assert validate_llm_output(_answer(contradicting=["low urgency (confidence 0.88)"]),
                               st) == []


@prop(HALLUCINATION)
def test_paraphrase_tolerance_does_not_admit_an_invented_finding():
    """Subset matching must not become no matching."""
    st = _conflict_state()
    errs = validate_llm_output(_answer(supporting=["ST elevation on ECG"]), st)
    assert errs, "an invented finding must still be caught"


@prop(HALLUCINATION)
def test_a_single_shared_word_does_not_ground_a_citation():
    """'low urgency' requires both tokens; an unrelated sentence containing 'low' must not
    pass as grounded."""
    st = _conflict_state()
    errs = validate_llm_output(_answer(supporting=["low suspicion of aortic dissection"]), st)
    assert errs, "one shared adjective is not grounding"


# ------------------------------------------------------------------ misread values
@prop(QUALIFIER)
def test_calling_a_normal_value_elevated_is_flagged():
    """Observed: troponin 5.0 against a reference of <=14 cited as 'elevated troponin'. The
    value is real, so no fabrication check sees it -- but the reasoning rests on a misreading."""
    st = _conflict_state()
    assert st["labs"]["troponin"]["flag"] == "normal", st["labs"]["troponin"]
    errs = check_value_qualifiers(_answer(supporting=["elevated troponin (5.0 ng/L)"]), st)
    assert errs, "a normal value described as elevated must be challenged"
    assert "troponin" in errs[0]


@prop(QUALIFIER)
def test_calling_a_high_value_elevated_is_accepted():
    st = build_clinical_state(
        bundle(ultrasound={"lung": lung_report()}), labs={"troponin": 340.0})
    assert check_value_qualifiers(_answer(supporting=["elevated troponin"]), st) == []


@prop(QUALIFIER)
def test_naming_a_value_without_a_direction_is_accepted():
    st = _conflict_state()
    assert check_value_qualifiers(_answer(supporting=["troponin 5.0 ng/L"]), st) == []


# ------------------------------------------------------------------ coverage loophole
@prop(COVERAGE)
def test_a_measured_value_listed_as_missing_is_flagged():
    """Observed: asked to account for SpO2, the model listed 'spo2 (90.0, low)' under
    missing_information -- while SpO2 had been measured."""
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79,
                                    features={"o2sat": 90, "pulse": 118, "respr": 24}),
               ultrasound={"lung": lung_report()}), labs={})
    out = _answer(supporting=["b lines", "hr 118", "rr 24"],
                  missing_information=["spo2 (90.0, low)"])
    # Listing a measured value as missing is a contradiction of the record, not an omission,
    # so it is checked separately from coverage -- and it withholds where coverage warns.
    errs = check_missing_information_accuracy(out, st)
    assert any("WAS measured" in e for e in errs), errs


@prop(COVERAGE)
def test_listing_a_genuinely_absent_test_is_accepted():
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79,
                                    features={"o2sat": 90, "pulse": 118, "respr": 24}),
               ultrasound={"lung": lung_report()}), labs={})
    out = _answer(supporting=["b lines", "hr 118", "rr 24", "spo2 90"],
                  missing_information=["troponin"])
    assert check_evidence_coverage(out, st) == []


# ------------------------------------------------------------------ scope of advice
@prop(SCOPE_ADVICE)
def test_prescribing_treatment_is_flagged():
    """Observed: 'Initiate treatment for heart failure, including diuretics or ACE
    inhibitors'. The system recommends investigations, not therapy."""
    st = _conflict_state()
    out = _answer(recommended_next_step="Initiate treatment for heart failure, including "
                                        "diuretics or ACE inhibitors")
    errs = check_scope_of_advice(out, st)
    assert errs, "treatment instructions must be challenged"


@prop(SCOPE_ADVICE)
def test_recommending_an_investigation_is_accepted():
    st = _conflict_state()
    for step in ("Obtain a BNP and repeat the troponin at 3 hours",
                 "Arrange a formal echocardiogram",
                 "Escalate to the on-call physician for review"):
        assert check_scope_of_advice(_answer(recommended_next_step=step), st) == [], step


# ------------------------------------------------------------------ atomicity
@prop(COVERAGE)
def test_a_comma_joined_missing_information_list_is_flagged():
    """Observed twice: 'd-dimer, crp, wbc, creatinine, ph, temp' as a single element."""
    st = _conflict_state()
    out = _answer(missing_information=["d-dimer, crp, wbc, creatinine, ph, temp"])
    assert check_atomicity(out, st)


@prop(COVERAGE)
def test_atomic_entries_are_accepted():
    st = _conflict_state()
    assert check_atomicity(_answer(missing_information=["d_dimer", "crp"]), st) == []
