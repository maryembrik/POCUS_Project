"""Evidence relationships and the unsound/untidy split.

Both come from the same observation: the model handles the output format better than it
handles what each ultrasound sign means. Neither check tries to supply medical reasoning --
the first refuses the clearest misuses, the second decides what a residual fault costs.
"""
import json

from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state
from src.agents.clinical.llm import ScriptedBackend
from src.agents.clinical.reasoning import (check_evidence_relationships, check_recommendation_wording,
                                  reason)
from .helpers import cite, prop, bundle

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
    out["differential"][0]["supporting"] = cite(st, "b lines", "hr", "rr", "spo2")
    out["missing_information"] = ["troponin, lactate"]          # untidy only
    raw = json.dumps(out)
    res = reason(st, llm_fn=ScriptedBackend(raw, raw), max_revisions=1)
    assert res.get("differential_withheld") is not True, res["validation_errors"]
    assert res["differential"] is not None
    assert res["warnings"], "the fault must still be reported"


@prop(SEVERITY)
def test_a_misread_value_can_no_longer_be_expressed_at_all():
    """This case used to be the value-qualifier check's headline: the model wrote "high
    troponin level (5.0 ng/L)" for a troponin the state recorded as normal, and the misread was
    caught after the fact.

    Enumerated evidence removes the opportunity. The qualifier travels with the identifier --
    "troponin: 5.0 ng/L -- NORMAL" -- so citing it cannot relabel it, and writing the
    qualifier in prose is rejected as not being an identifier. The fault is now unreachable
    rather than merely detected, which is why this test asserts the rejection instead of the
    old error message.
    """
    st = build_clinical_state(
        bundle(triage=S.make_triage("low", 0.88, features={"pulse": 82}),
               ultrasound={"heart": S.make_report(
                   "heart", [S.make_finding("severe dysfunction", 0.74)],
                   reliability={"confidence_calibrated": True,
                                "has_normal_class": True})}),
        labs={"troponin": 5.0, "lactate": 1.1})

    # The misread, attempted in prose.
    prose = json.dumps({
        "differential": [{"diagnosis": "Myocardial infarction", "likelihood": "moderate",
                          "supporting": ["high troponin level (5.0 ng/L)"],
                          "contradicting": [], "limitations": []}],
        "missing_information": ["bnp"], "uncertainty": "u",
        "recommended_next_step": "obtain a BNP"})
    res = reason(st, llm_fn=ScriptedBackend(prose, prose), max_revisions=1)
    assert res["differential_withheld"] is True
    assert any("not an evidence identifier" in e for e in res["validation_errors"]), \
        res["validation_errors"]

    # Cited properly instead, the value reaches the reader carrying its own qualifier.
    ids = json.dumps({
        "differential": [{"diagnosis": "Myocardial infarction", "likelihood": "moderate",
                          "supporting": cite(st, "troponin", "severe dysfunction"),
                          "contradicting": [], "limitations": []}],
        "missing_information": ["bnp"], "uncertainty": "u",
        "recommended_next_step": "obtain a BNP"})
    ok = reason(st, llm_fn=ScriptedBackend(ids, ids), max_revisions=1)
    cited = " ".join(ok["differential"]["differential"][0]["supporting"])
    assert "NORMAL" in cited, cited
    assert "high troponin" not in cited.lower()


def _two_fault_answer(evidence_ids, *, joined: bool, overclaim: bool) -> str:
    """An otherwise sound answer carrying two INDEPENDENT untidy faults.

    Independent on purpose: a comma-joined missing_information and an overclaiming next step
    have nothing to do with each other, so fixing one cannot fix the other and the number of
    rounds required is exactly two.
    """
    return json.dumps({
        "differential": [{"diagnosis": "Pulmonary oedema", "likelihood": "moderate",
                          "supporting": list(evidence_ids), "contradicting": [],
                          "limitations": []}],
        "missing_information": ["troponin, lactate"] if joined else ["troponin", "lactate"],
        "uncertainty": "u",
        "recommended_next_step": (
            "obtain a troponin to rule out myocardial infarction" if overclaim
            else "obtain a troponin as additional information relevant to assessing "
                 "myocardial injury")})


@prop(SEVERITY)
def test_two_independent_faults_need_two_revision_rounds():
    """The case that justifies max_revisions=2.

    One complaint is sent per round, so two independent faults cannot clear in one round
    however cooperative the model is. This was worth testing rather than assuming: the reason
    for raising the cap was that identifier-era complaints are actionable, and an untested
    architecture claim is not one worth making in a report.
    """
    st = _lung_state()
    ev = cite(st, "b lines", "hr", "rr", "spo2")
    backend = ScriptedBackend(
        _two_fault_answer(ev, joined=True, overclaim=True),      # both faults
        _two_fault_answer(ev, joined=False, overclaim=True),     # one fixed
        _two_fault_answer(ev, joined=False, overclaim=False))    # both fixed
    res = reason(st, llm_fn=backend, max_revisions=2)

    assert len(backend.calls) == 3, "one initial call and two revision rounds"
    assert len(res["revisions"]) == 2
    assert all(len(r["complaints"]) == 1 for r in res["revisions"]), res["revisions"]
    assert res.get("differential_withheld") is not True
    assert not (res.get("warnings") or []), res.get("warnings")


@prop(SEVERITY)
def test_one_round_is_not_enough_for_two_faults():
    """The other half of the same measurement: with the old cap the second fault survives and
    is delivered as a warning. The answer is still shown -- these are untidy faults -- but the
    report would carry a complaint the model was never given the chance to fix."""
    st = _lung_state()
    ev = cite(st, "b lines", "hr", "rr", "spo2")
    backend = ScriptedBackend(
        _two_fault_answer(ev, joined=True, overclaim=True),
        _two_fault_answer(ev, joined=False, overclaim=True))
    res = reason(st, llm_fn=backend, max_revisions=1)

    assert len(backend.calls) == 2
    assert res.get("differential_withheld") is not True
    assert res["warnings"], "the unfixed fault must still be reported"


@prop(SEVERITY)
def test_a_revision_request_carries_one_complaint_not_all_of_them():
    """Measured on a real case: one complaint was fixed, three were fixed in no respect at
    all. The unsound fault is sent, because it is the one deciding whether the answer can be
    shown; the untidy fault becomes a warning either way."""
    st = _lung_state()
    out = _entry("Pulmonary Embolism")
    out["differential"][0]["supporting"] = cite(st, "b lines", "hr", "rr", "spo2")
    # A screened negative that has nothing to do with pulmonary embolism. Enumeration does not
    # prevent this one: pleural thickening was genuinely assessed, so it has an identifier, and
    # whether it bears on the diagnosis is a judgement no list of identifiers can settle.
    out["differential"][0]["contradicting"] = cite(st, "pleural thickening")
    out["missing_information"] = ["troponin, lactate"]
    backend = ScriptedBackend(json.dumps(out))
    res = reason(st, llm_fn=backend, max_revisions=1)
    _, revision = backend.calls[1]
    assert "does not bear" in revision, "the unsound fault must be the one sent"
    assert "one test per" not in revision, "the untidy fault must not crowd the request"
    # Nothing is lost: everything found is still recorded.
    assert res["revisions"][0]["also_found"], res["revisions"]
