"""An unperformed scan is missing information, and must be named as such.

The escalation policy already knows an organ was never assessed. This checks that the answer
the clinician reads says so too -- `missing_information` is what determines what gets ordered
next, and an unperformed scan is usually the most useful item on it.
"""
import json

from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state
from src.agents.clinical.llm import ScriptedBackend
from src.agents.clinical.reasoning import check_unassessed_reported, reason
from .helpers import cite, prop, bundle, lung_report

UNASSESSED = "Unassessed-organ reporting"


def _state():
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.82,
                                    features={"o2sat": 93, "pulse": 104, "respr": 22}),
               ultrasound={"heart": S.make_report("heart", [], status="not_supported"),
                           "lung": lung_report(b_lines=0.78)}),
        labs={"troponin": 9.0, "lactate": 1.4})


def _answer(missing_information, state=None):
    """Free-text evidence by default: most tests here call check_unassessed_reported directly,
    which reads missing_information and never looks at what was cited. Pass `state` for the
    tests that go through reason(), where citations must be evidence identifiers."""
    supporting = cite(state, "troponin") if state is not None else ["troponin 9.0"]
    return {"differential": [{"diagnosis": "Cardiac event", "likelihood": "low",
                              "supporting": supporting, "contradicting": [],
                              "limitations": []}],
            "missing_information": missing_information,
            "uncertainty": "u",
            "recommended_next_step": "obtain a BNP"}


@prop(UNASSESSED)
def test_omitting_the_unassessed_organ_is_flagged():
    """Observed: the heart was never scanned, the model offered 'Cardiac Event' as a
    diagnosis, and missing_information listed laboratory tests only."""
    errs = check_unassessed_reported(_answer(["bnp", "d_dimer"]), _state())
    assert errs, "an unperformed scan must be named"
    assert "heart" in errs[0]


@prop(UNASSESSED)
def test_naming_the_unassessed_organ_passes():
    assert check_unassessed_reported(
        _answer(["heart ultrasound", "bnp"]), _state()) == []


@prop(UNASSESSED)
def test_no_complaint_when_every_organ_was_assessed():
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.8, features={"o2sat": 93}),
               ultrasound={"lung": lung_report()}), labs={})
    assert check_unassessed_reported(_answer(["troponin"]), st) == []


@prop(UNASSESSED)
def test_it_is_unsound_so_an_unfixed_answer_is_withheld():
    """A gap in the workup presented as a complete assessment is not a presentation
    problem."""
    st = _state()
    raw = json.dumps(_answer(["bnp"], state=st))
    out = reason(st, llm_fn=ScriptedBackend(raw, raw), max_revisions=1)
    assert out["differential_withheld"] is True
    assert any("never assessed" in e for e in out["validation_errors"]), \
        out["validation_errors"]


@prop(UNASSESSED)
def test_every_fault_is_reported_even_when_the_answer_is_already_doomed():
    """Returning on the first class of fault reported one fabrication while a misread value
    in the same answer went unmentioned. Fine for the decision, useless for a benchmark."""
    st = _state()
    bad = json.dumps({
        "differential": [{"diagnosis": "Myocardial infarction", "likelihood": "moderate",
                          "supporting": ["elevated troponin (9.0 ng/L)",
                                         "ST elevation on ECG"],
                          "contradicting": [], "limitations": []}],
        "missing_information": ["bnp"], "uncertainty": "u",
        "recommended_next_step": "obtain a BNP"})
    out = reason(st, llm_fn=ScriptedBackend(bad), max_revisions=1)
    assert out["differential_withheld"] is True
    # "ST elevation on ECG" is now rejected one step earlier, as prose where an identifier was
    # required, rather than as an ungrounded citation. Either way the answer is refused; what
    # this test is about is that refusing it does not stop the other faults being reported.
    assert any("not an evidence identifier" in e for e in out["validation_errors"]), \
        out["validation_errors"]
    assert out["also_found"], "the other faults must still be listed"
    joined = " ".join(out["also_found"])
    assert "troponin" in joined     # the misread value
    assert "heart" in joined        # the unassessed organ
