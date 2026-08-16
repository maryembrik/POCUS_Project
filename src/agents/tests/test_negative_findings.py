"""Screened-and-negative must stay distinguishable from never-assessed.

Three states, three meanings:

    detected      the module looked and found it
    not_detected  the module looked and did not find it
    absent        the module never assessed it

Collapsing the middle state into the third throws away a real observation. Collapsing it
into a negative result invents one. The lung module screens four findings and typically
fires on one or two, so this is the ordinary case, not an edge case.
"""
from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state, render_state
from .helpers import MISSING_NOT_NORMAL, SCHEMA_REJECTION, prop, bundle


def _screened_report() -> dict:
    """A lung scan that looked for four findings and saw one."""
    return S.make_report(
        "lung",
        [S.make_finding("b_lines", 0.86)],
        not_detected=[S.make_finding("consolidation", 0.09),
                      S.make_finding("pleural_effusion", 0.04),
                      S.make_finding("pleural_thickening", 0.11)],
        reliability={"confidence_calibrated": True,
                     "has_normal_class": False,
                     "modelled_findings": ["b_lines", "consolidation",
                                           "pleural_effusion", "pleural_thickening"]},
    )


@prop(MISSING_NOT_NORMAL)
def test_a_module_can_report_what_it_screened_and_did_not_find():
    rep = _screened_report()
    assert S.validate_report(rep) == []
    assert len(rep["not_detected"]) == 3


@prop(MISSING_NOT_NORMAL)
def test_negatives_reach_the_clinical_state_marked_as_not_detected():
    st = build_clinical_state(bundle(ultrasound={"lung": _screened_report()}))
    by_label = {f["label"]: f for f in st["imaging"]["findings"]}
    assert by_label["b_lines"]["detected"] is True
    assert by_label["consolidation"]["detected"] is False
    assert by_label["pleural_effusion"]["detected"] is False


@prop(MISSING_NOT_NORMAL)
def test_a_screened_negative_is_not_the_same_as_an_unassessed_organ():
    screened = build_clinical_state(bundle(ultrasound={"lung": _screened_report()}))
    unassessed = build_clinical_state(bundle(ultrasound={}))
    assert any(f["label"] == "consolidation" for f in screened["imaging"]["findings"])
    assert unassessed["imaging"]["findings"] == []


@prop(MISSING_NOT_NORMAL)
def test_an_all_negative_scan_is_expressible():
    """A scan where nothing fires is a real, informative result. Before not_detected existed
    it could not be represented at all: status 'ok' with no findings was rejected."""
    rep = S.make_report(
        "lung", [],
        not_detected=[S.make_finding("b_lines", 0.05),
                      S.make_finding("consolidation", 0.03)],
        reliability={"confidence_calibrated": True, "has_normal_class": False},
    )
    assert S.validate_report(rep) == []


@prop(SCHEMA_REJECTION)
def test_ok_with_nothing_screened_and_nothing_found_is_still_rejected():
    """The relaxation must not become a hole: claiming an assessment happened while
    recording nothing about it remains invalid."""
    rep = S.make_report("lung", [], not_detected=[], status="ok")
    assert S.validate_report(rep)


@prop(SCHEMA_REJECTION)
def test_a_label_cannot_be_both_detected_and_not_detected():
    rep = S.make_report(
        "lung",
        [S.make_finding("b_lines", 0.86)],
        not_detected=[S.make_finding("b_lines", 0.04)],
        reliability={"confidence_calibrated": True},
    )
    errs = S.validate_report(rep)
    assert any("both detected and not detected" in e for e in errs), errs


@prop(SCHEMA_REJECTION)
def test_not_supported_cannot_carry_negatives_either():
    rep = S.make_report("vascular", [], status="not_supported")
    rep["not_detected"] = [S.make_finding("dvt", 0.02)]
    assert S.validate_report(rep)


@prop(SCHEMA_REJECTION)
def test_negative_findings_are_range_checked_like_positive_ones():
    rep = _screened_report()
    rep["not_detected"][0]["confidence"] = 1.7
    assert any("not_detected" in e for e in S.validate_report(rep))


@prop(MISSING_NOT_NORMAL)
def test_an_all_negative_scan_does_not_count_as_a_positive_finding():
    """Regression guard. Before `not_detected` existed, the lung module represented "nothing
    above threshold" with a placeholder entry in `findings`. Everything in that list is marked
    detected=True downstream, so the ABSENCE of findings was read as a positive finding: it
    suppressed the "no positive finding" escalation trigger and counted as weak evidence in
    the case-quality grade. The absence of findings escalated less readily than their
    presence, which is backwards."""
    from src.agents.reasoning import escalation_decision
    rep = S.make_report(
        "lung", [],
        not_detected=[S.make_finding("b lines", 0.04),
                      S.make_finding("consolidation", 0.03)],
        reliability={"confidence_calibrated": True, "has_normal_class": False,
                     "modelled_findings": ["b_lines", "consolidation"]},
    )
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.9), ultrasound={"lung": rep}),
        labs={"troponin": 5.0, "lactate": 1.0})

    assert not any(f["detected"] for f in st["imaging"]["findings"]), \
        "an all-negative scan must contain no detected finding"
    d = escalation_decision(st)
    assert d["escalate"] is True, d
    assert any("cannot exclude" in t for t in d["triggers"]), d["triggers"]


@prop(MISSING_NOT_NORMAL)
def test_screened_negatives_survive_into_the_rendered_prompt():
    """If the negatives never reach the text the model reads, recording them changes
    nothing."""
    st = build_clinical_state(bundle(ultrasound={"lung": _screened_report()}))
    text = render_state(st).lower()
    assert "consolidation" in text, text
