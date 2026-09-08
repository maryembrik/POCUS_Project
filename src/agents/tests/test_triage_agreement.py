"""Human--AI disagreement: detected, recorded, and deliberately inert.

The system carries three kinds of uncertainty and this is the third. The imaging modules are
uncertain about a finding. The missing-data account is uncertain about what was ever measured.
This is uncertainty about the ASSESSMENT: two independent judgements of one patient that did
not coincide.

What makes it worth testing is mostly what it must NOT do. It must not decide who was right,
it must not change the tier, and it must not escalate -- and none of those failures would
announce itself. A disagreement signal that quietly raised the urgency would look like a more
cautious system rather than like a model overriding a clinician.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents import schema as S  # noqa: E402
from src.agents.clinical.clinical_state import build_clinical_state  # noqa: E402
from src.agents.clinical.decision_support import decision_support  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision  # noqa: E402
from src.agents.clinical.report import build_report  # noqa: E402

from .helpers import bundle, lung_report, prop  # noqa: E402

AGREEMENT = "Human-AI assessment agreement"


def _state(final: str, suggested: str | None = "high", **over):
    b = bundle(triage=S.make_triage(final, 0.8, model="clinician"),
               ultrasound={"lung": lung_report()})
    if suggested is not None:
        b["triage_suggestion"] = {"urgency": suggested,
                                  "probabilities": {"low": 0.1, "medium": 0.2, "high": 0.7},
                                  "model": "triage_deployed"}
    b.update(over)
    return build_clinical_state(b)


@prop(AGREEMENT)
def test_agreement_is_recorded_when_the_two_assessments_match():
    a = _state("high", "high")["triage_agreement"]
    assert a["agree"] is True
    assert a["direction"] == "same"
    assert a["steps"] == 0


@prop(AGREEMENT)
def test_the_direction_of_a_disagreement_is_recorded_because_the_two_are_not_equivalent():
    """A clinician grading ABOVE the suggestion is being more cautious than the model.

    Grading BELOW it is the direction in which a missed deterioration would sit. The system
    records which happened; it draws no conclusion from it.
    """
    lower = _state("low", "high")["triage_agreement"]
    assert lower["agree"] is False
    assert lower["direction"] == "clinician_lower"
    assert lower["steps"] == 2

    higher = _state("high", "low")["triage_agreement"]
    assert higher["direction"] == "clinician_higher"
    assert higher["steps"] == 2


@prop(AGREEMENT)
def test_the_tier_that_reaches_the_reasoning_layer_is_the_clinicians():
    """The signal is an observation about the assessment, not an input to it.

    If the recorded tier ever became the model's, the clinician's control would be advisory --
    the reverse of the design -- and every downstream rule that reads the tier, including the
    conflict detector, would be reading a number the clinician did not choose.
    """
    for final, suggested in (("low", "high"), ("high", "low"), ("medium", "high")):
        state = _state(final, suggested)
        assert state["triage"]["urgency"] == final, (
            f"clinician set {final!r} and the state carries "
            f"{state['triage']['urgency']!r}")


@prop(AGREEMENT)
def test_a_disagreement_does_not_escalate():
    """Escalation is a judgement about the patient, from evidence about the patient.

    A disagreement between two assessors is a fact about the assessment. Wiring it to the
    escalation policy would let a model that graded higher than the clinician raise the urgency
    of the case indirectly -- authority the design gives the clinician and not the model.

    Compared against the identical encounter with no suggestion at all, so the assertion is
    that the signal changes nothing rather than that some particular trigger list appears.
    """
    agreeing = escalation_decision(_state("high", "high"))
    differing = escalation_decision(_state("high", "low"))
    absent = escalation_decision(_state("high", None))

    assert differing["triggers"] == absent["triggers"]
    assert agreeing["triggers"] == absent["triggers"]
    assert differing["escalate"] == absent["escalate"]


@prop(AGREEMENT)
def test_nothing_records_either_party_as_wrong():
    """There is no ground truth here, so any field naming a loser would be asserting one."""
    a = _state("low", "high")["triage_agreement"]
    blob = " ".join(f"{k} {v}" for k, v in a.items()).lower()
    for word in ("wrong", "incorrect", "error", "override", "correct"):
        assert word not in blob, f"the agreement record contains {word!r}: {a}"


@prop(AGREEMENT)
def test_the_signal_is_absent_rather_than_invented_when_no_model_proposed_a_tier():
    """The deployed model may not be installed. That is not a disagreement."""
    assert _state("high", None)["triage_agreement"] is None


@prop(AGREEMENT)
def test_an_unknown_tier_produces_no_signal_rather_than_a_wrong_one():
    """A tier outside the three-way scale cannot be ordered against another one.

    Returning a direction here would mean inventing a comparison; the honest answer is that
    there is nothing to compare.
    """
    assert _state("high", "urgent")["triage_agreement"] is None


@prop(AGREEMENT)
def test_the_archived_report_carries_the_comparison():
    """The record is where this belongs -- not the screen a clinician reads while assessing.

    Someone reviewing the encounter afterwards, or comparing the two assessments across many
    encounters, needs it; the clinician in front of the patient does not.
    """
    state = _state("low", "high")
    # The real decision-support object rather than a hand-built stand-in. A fixture that
    # imitates it has to be updated whenever the report reads a new key, and it fails as a
    # KeyError in the test rather than as a defect in the code.
    escalation = escalation_decision(state)
    support = decision_support(state, escalation)
    result = {"differential": [], "missing_information": [], "uncertainty": "",
              "recommended_next_step": "", "escalation": escalation}
    rep = build_report(state, result, support)

    agreement = rep["triage"]["agreement"]
    assert agreement["suggested"] == "high"
    assert agreement["final"] == "low"
    assert agreement["direction"] == "clinician_lower"
    assert agreement["model"] == "triage_deployed"
