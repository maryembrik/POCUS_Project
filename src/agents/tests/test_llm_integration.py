"""LLM integration: the model is never trusted, only checked.

Nothing here needs the 4.6 GB model. The point is that the pipeline's behaviour is
determined by the safety layer, so a scripted backend exercises the same paths a real one
would -- including the paths a real model reaches only occasionally.
"""
import json

from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from src.agents.llm import FailingBackend, ScriptedBackend, is_available
from src.agents.reasoning import reason
from .helpers import (HALLUCINATION, MISSING_NOT_NORMAL, prop, bundle, llm_output,
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
    answer = llm_output(["b_lines", "troponin", "heart rate 118",
                         "oxygen saturation 90%"],
                        missing_information=["d_dimer"])
    backend = ScriptedBackend(json.dumps(answer))
    out = reason(st, llm_fn=backend)
    assert out["validation_errors"] is None, out["validation_errors"]
    assert out["differential"] is not None
    assert out.get("differential_withheld") is not True


@prop(HALLUCINATION)
def test_the_model_receives_the_state_not_the_raw_bundle():
    """What the model sees is the rendered state, with its absences and limits. If the raw
    numbers reached it instead, none of the upstream guarantees would apply."""
    st = _state()
    backend = ScriptedBackend(json.dumps(llm_output(["b_lines"])))
    reason(st, llm_fn=backend)
    system, user = backend.calls[0]
    assert "NOT MEASURED" in user
    assert "ONLY findings" in system or "Use ONLY" in system


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
             + json.dumps(llm_output(["b_lines", "troponin"])))
    out = reason(st, llm_fn=ScriptedBackend(reply))
    assert out["differential"] is not None, out["validation_errors"]


@prop(LLM_ROBUSTNESS)
def test_the_escalation_decision_is_identical_with_and_without_a_model():
    """The safety decision must not depend on the model at all -- that is the whole design.
    Same state, three different backends, one escalation decision."""
    st = _state()
    a = reason(st, llm_fn=None)["escalation"]
    b = reason(st, llm_fn=ScriptedBackend(json.dumps(llm_output(["b_lines"]))))["escalation"]
    c = reason(st, llm_fn=FailingBackend())["escalation"]
    assert a == b == c, (a, b, c)


# ------------------------------------------------------------------ grounding
@prop(MISSING_NOT_NORMAL)
def test_a_fabricated_lab_is_caught_even_in_a_fluent_answer():
    """The dangerous case is not a malformed reply but a fluent, plausible one that cites a
    test nobody ordered."""
    st = _state(labs={})
    fluent = json.dumps(llm_output(
        ["b_lines", "markedly elevated troponin"],
        diagnosis="Acute coronary syndrome", likelihood="high"))
    out = reason(st, llm_fn=ScriptedBackend(fluent))
    assert out["differential_withheld"] is True
    assert any("NOT MEASURED" in e for e in out["validation_errors"])


@prop(MISSING_NOT_NORMAL)
def test_availability_is_reported_rather_than_assumed():
    ok, msg = is_available()
    assert isinstance(ok, bool)
    assert msg, "an unavailable backend must say why"
