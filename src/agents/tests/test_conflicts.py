"""Conflict detection: disagreement between agents must be surfaced, not resolved.

The system does not have the information to decide which agent is wrong, so the correct
behaviour is to make the disagreement visible and escalate -- never to average it away or
defer to whichever model is more confident.
"""
from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from .helpers import CONFLICT, prop, bundle, heart_report, lung_report


@prop(CONFLICT)
def test_low_triage_with_severe_imaging_finding_is_a_conflict():
    """The central case for this project: the nurse's assessment and the scan disagree."""
    b = bundle(triage=S.make_triage("low", 0.85),
               ultrasound={"heart": heart_report(label="severe_dysfunction",
                                                 confidence=0.71)})
    conflicts = S.detect_conflicts(b)
    assert conflicts, "low triage against a severe finding must be flagged"
    assert any("low urgency" in c for c in conflicts), conflicts


@prop(CONFLICT)
def test_high_triage_with_severe_finding_is_not_a_conflict():
    """Agreement must not be reported as disagreement, or the flag becomes noise."""
    b = bundle(triage=S.make_triage("high", 0.85),
               ultrasound={"heart": heart_report(label="severe_dysfunction",
                                                 confidence=0.71)})
    assert not any("low urgency" in c for c in S.detect_conflicts(b))


@prop(CONFLICT)
def test_low_confidence_finding_is_flagged():
    b = bundle(ultrasound={"lung": lung_report(b_lines=0.31)})
    assert any("low confidence" in c for c in S.detect_conflicts(b))


@prop(CONFLICT)
def test_a_severe_finding_below_the_confidence_floor_is_not_a_triage_conflict():
    """A finding the module barely believes should not be escalated as if it contradicted
    triage; it is flagged as low confidence instead."""
    b = bundle(triage=S.make_triage("low", 0.85),
               ultrasound={"heart": heart_report(label="severe_dysfunction",
                                                 confidence=0.20)})
    conflicts = S.detect_conflicts(b)
    assert not any("low urgency" in c for c in conflicts), conflicts
    assert any("low confidence" in c for c in conflicts), conflicts


@prop(CONFLICT)
def test_serious_group_label_also_triggers_the_conflict():
    """Severity may be carried by the group rather than the label, e.g. a gallbladder class
    grouped under acute inflammation."""
    rep = S.make_report("gallbladder",
                        [S.make_finding("cholecystitis", 0.68, group="acute inflammation")],
                        reliability={"confidence_calibrated": True,
                                     "has_normal_class": False})
    b = bundle(triage=S.make_triage("low", 0.9), ultrasound={"gallbladder": rep})
    assert any("low urgency" in c for c in S.detect_conflicts(b)), S.detect_conflicts(b)


@prop(CONFLICT)
def test_conflicts_reach_the_clinical_state():
    """Detection is worthless if the state does not carry it forward."""
    b = bundle(triage=S.make_triage("low", 0.85),
               ultrasound={"heart": heart_report(confidence=0.71)})
    st = build_clinical_state(b)
    assert st["conflicts"], st


@prop(CONFLICT)
def test_conflicts_weigh_on_case_quality():
    """Two agents disagreeing is a stronger warning than one gap in the record, and the
    grade should reflect that."""
    b = bundle(triage=S.make_triage("low", 0.85),
               ultrasound={"heart": heart_report(confidence=0.71)})
    st = build_clinical_state(b)
    assert st["case_quality"]["grade"] in ("MODERATE", "POOR")
    assert any("conflict" in r for r in st["case_quality"]["reasons"]), st["case_quality"]


@prop(CONFLICT)
def test_no_conflict_when_agents_agree_and_confidence_is_adequate():
    b = bundle(triage=S.make_triage("high", 0.9),
               ultrasound={"lung": lung_report(b_lines=0.88)})
    assert S.detect_conflicts(b) == []
