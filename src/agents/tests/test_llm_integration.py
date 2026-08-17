"""LLM integration: the model is never trusted, only checked.

Nothing here needs the 4.6 GB model. The point is that the pipeline's behaviour is
determined by the safety layer, so a scripted backend exercises the same paths a real one
would -- including the paths a real model reaches only occasionally.
"""
import json

from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state, build_evidence
from src.agents.clinical.llm import FailingBackend, ScriptedBackend, is_available
from src.agents.clinical.reasoning import reason
from .helpers import (HALLUCINATION, MISSING_NOT_NORMAL, cite, prop, bundle, llm_output,
                      lung_report)

LLM_ROBUSTNESS = "LLM failure containment"


def _state(labs=None):
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79, features={"o2sat": 90, "pulse": 118}),
               ultrasound={"lung": lung_report(b_lines=0.86)}),
        labs=labs if labs is not None else {})


# ------------------------------------------------------------------ the happy path
@prop(HALLUCINATION)
def test_a_well_formed_grounded_answer_is_returned():
    """Grounded AND complete: since the evidence-coverage check was added, an answer must
    also account for the abnormal vitals, so the fixture cites them."""
    st = _state(labs={"troponin": 340.0, "lactate": 1.2})
    answer = llm_output(cite(st, "b_lines", "troponin", "hr", "spo2"),
                        missing_information=["d_dimer"])
    backend = ScriptedBackend(json.dumps(answer))
    out = reason(st, llm_fn=backend)
    assert out["validation_errors"] is None, out["validation_errors"]
    assert out["differential"] is not None
    assert out.get("differential_withheld") is not True
    # The identifiers are resolved before anything downstream reads the answer, so the
    # clinician sees the fact and the audit trail keeps what the model actually cited.
    entry = out["differential"]["differential"][0]
    assert entry["supporting_ids"] == answer["differential"][0]["supporting"]
    assert any("b_lines" in s for s in entry["supporting"]), entry["supporting"]
    assert any("HIGH" in s for s in entry["supporting"]), entry["supporting"]


@prop(HALLUCINATION)
def test_the_model_receives_the_state_not_the_raw_bundle():
    """What the model sees is the rendered state, with its absences and limits. If the raw
    numbers reached it instead, none of the upstream guarantees would apply."""
    st = _state()
    backend = ScriptedBackend(json.dumps(llm_output(cite(st, "b_lines"))))
    reason(st, llm_fn=backend)
    system, user = backend.calls[0]
    assert "NOT MEASURED" in user
    assert "HARD RULES" in system
    # The evidence block travels with the state rather than replacing it: the state carries the
    # absences and the limits, which bound the conclusion but are not citable facts.
    assert "AVAILABLE EVIDENCE" in user
    assert user.index("CLINICAL STATE") < user.index("AVAILABLE EVIDENCE")


# ------------------------------------------------------------------ failure containment
@prop(LLM_ROBUSTNESS)
def test_a_backend_that_raises_does_not_crash_the_pipeline():
    """A model that fails to load, times out or runs out of memory must degrade to a
    withheld differential. An exception reaching the caller in a clinical setting means the
    screen goes blank at the moment a decision is being made."""
    st = _state()
    out = reason(st, llm_fn=FailingBackend("out of memory"))
    assert out["differential"] is None
    assert out["validation_errors"], out
    assert out["escalation"] is not None, "the escalation decision must survive model failure"


@prop(LLM_ROBUSTNESS)
def test_prose_instead_of_json_degrades_safely():
    st = _state()
    out = reason(st, llm_fn=ScriptedBackend("I'm sorry, I can't help with that."))
    assert out["differential"] is None
    assert out["validation_errors"]


@prop(LLM_ROBUSTNESS)
def test_truncated_json_degrades_safely():
    """A reasoning model that exhausts its token budget mid-object is a configuration
    failure, not a clinical one, and must not be presented as an answer."""
    st = _state()
    truncated = '{"differential": [{"diagnosis": "Pulmonary oedema", "supp'
    out = reason(st, llm_fn=ScriptedBackend(truncated))
    assert out["differential"] is None
    assert out["validation_errors"]


@prop(LLM_ROBUSTNESS)
def test_chain_of_thought_before_the_json_is_tolerated():
    """HuatuoGPT-o1 reasons in prose before answering. The JSON must still be recovered, or
    the model's principal strength becomes a parse failure."""
    st = _state(labs={"troponin": 340.0, "lactate": 1.2})
    reply = ("## Thinking\nThe patient has B-lines and a raised troponin, so cardiogenic "
             "pulmonary oedema is likely.\n\n## Final Response\n"
             + json.dumps(llm_output(cite(st, "b_lines", "troponin", "hr", "spo2"),
                                     missing_information=["d_dimer"])))
    out = reason(st, llm_fn=ScriptedBackend(reply))
    # Accepted, not merely parsed. `differential` is populated even for a rejected answer, so
    # asserting it alone would pass on a reply the pipeline had refused.
    assert out["validation_errors"] is None, out["validation_errors"]
    assert out.get("differential_withheld") is not True


@prop(LLM_ROBUSTNESS)
def test_the_escalation_decision_is_identical_with_and_without_a_model():
    """The safety decision must not depend on the model at all -- that is the whole design.
    Same state, three different backends, one escalation decision."""
    st = _state()
    a = reason(st, llm_fn=None)["escalation"]
    b = reason(st, llm_fn=ScriptedBackend(json.dumps(llm_output(cite(st, "b_lines")))))["escalation"]
    c = reason(st, llm_fn=FailingBackend())["escalation"]
    assert a == b == c, (a, b, c)


# ------------------------------------------------------------------ grounding
@prop(MISSING_NOT_NORMAL)
def test_a_fabricated_lab_is_caught_even_in_a_fluent_answer():
    """The dangerous case is not a malformed reply but a fluent, plausible one that cites a
    test nobody ordered.

    This state resulted no labs, so there is no troponin identifier to cite. The fabrication is
    now refused for being prose rather than for naming an unmeasured test -- an earlier and
    blunter rejection, but the same outcome, and one the model cannot talk its way past.
    """
    st = _state(labs={})
    assert not any(e.get("label") == "troponin"
                   for e in build_evidence(st)), "fixture must not offer a troponin identifier"
    fluent = json.dumps(llm_output(
        ["E1", "markedly elevated troponin"],
        diagnosis="Acute coronary syndrome", likelihood="high"))
    out = reason(st, llm_fn=ScriptedBackend(fluent))
    assert out["differential_withheld"] is True
    assert any("not an evidence identifier" in e for e in out["validation_errors"]), \
        out["validation_errors"]


@prop(MISSING_NOT_NORMAL)
def test_availability_is_reported_rather_than_assumed():
    ok, msg = is_available()
    assert isinstance(ok, bool)
    assert msg, "an unavailable backend must say why"
