"""The hallucination guard: the single most safety-critical function in the system.

A differential citing a laboratory value that was never drawn is worse than no differential,
because it is indistinguishable from a well-founded one. These tests check that such an
answer is rejected rather than shown with a caveat.
"""
from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from src.agents.reasoning import validate_llm_output, parse_llm_json, reason
from .helpers import HALLUCINATION, prop, bundle, llm_output, lung_report


def _state(labs=None):
    return build_clinical_state(
        bundle(triage=S.make_triage("low", 0.66, features={"pulse": 96, "o2sat": 95}),
               ultrasound={"lung": lung_report(b_lines=0.81)}),
        labs=labs if labs is not None else {})


# ------------------------------------------------------------------ the core guarantee
@prop(HALLUCINATION)
def test_citing_an_unmeasured_lab_is_rejected():
    st = _state(labs={})
    assert "troponin" in st["missing"]["labs"], "precondition: troponin absent"
    errs = validate_llm_output(llm_output(["b_lines", "elevated troponin"]), st)
    assert errs, "citing an absent lab must be an error"
    assert any("NOT MEASURED" in e for e in errs), errs


@prop(HALLUCINATION)
def test_citing_a_finding_no_module_produced_is_rejected():
    st = _state()
    errs = validate_llm_output(llm_output(["dense consolidation on chest X-ray"]), st)
    assert errs, "a fabricated finding must be an error"
    assert any("fabrication" in e for e in errs), errs


@prop(HALLUCINATION)
def test_citing_only_what_is_in_the_state_passes():
    """The guard must not reject honest answers, or it will be switched off."""
    st = _state()
    assert validate_llm_output(llm_output(["b_lines"]), st) == []


@prop(HALLUCINATION)
def test_natural_phrasing_of_a_vital_is_not_a_fabrication():
    """The state stores 'hr'; a model writes 'heart rate 118 bpm'. Demanding the canonical
    abbreviation accuses an honest answer of inventing data."""
    st = _state()
    for phrasing in ("heart rate 118 bpm", "pulse 118", "tachycardia at 118",
                     "oxygen saturation 90%", "SpO2 90"):
        assert validate_llm_output(llm_output([phrasing]), st) == [], phrasing


@prop(HALLUCINATION)
def test_the_urgency_word_alone_does_not_ground_a_citation():
    """The triage urgency is the word 'high'. Admitting it to the vocabulary let any sentence
    containing 'high' pass the fabrication check -- which is grounding by coincidence."""
    st = _state()
    errs = validate_llm_output(llm_output(["high suspicion of aortic dissection"]), st)
    assert errs, "a fabricated finding must not be rescued by the word 'high'"


@prop(HALLUCINATION)
def test_punctuation_does_not_hide_an_absent_test():
    """The state stores 'd_dimer'; a model writes 'D-dimer'. Comparing raw strings would let
    an absent test through on a hyphen."""
    st = _state(labs={})
    assert "d_dimer" in st["missing"]["labs"]
    errs = validate_llm_output(llm_output(["b_lines", "normal D-dimer"]), st)
    assert any("NOT MEASURED" in e for e in errs), errs


@prop(HALLUCINATION)
def test_a_measured_lab_may_be_cited():
    st = _state(labs={"troponin": 340.0})
    assert validate_llm_output(llm_output(["b_lines", "troponin"]), st) == []


@prop(HALLUCINATION)
def test_contradicting_evidence_is_checked_as_well_as_supporting():
    """A fabrication used to argue against a diagnosis is as dangerous as one used for it."""
    st = _state()
    out = llm_output(["b_lines"], contradicting=["normal d-dimer"])
    assert validate_llm_output(out, st), "contradicting field must be validated too"


# ------------------------------------------------------------------ structural checks
@prop(HALLUCINATION)
def test_missing_required_key_is_rejected():
    st = _state()
    out = llm_output(["b_lines"])
    del out["recommended_next_step"]
    assert any("recommended_next_step" in e for e in validate_llm_output(out, st))


@prop(HALLUCINATION)
def test_empty_differential_is_rejected():
    st = _state()
    out = llm_output(["b_lines"])
    out["differential"] = []
    assert validate_llm_output(out, st)


@prop(HALLUCINATION)
def test_invalid_likelihood_value_is_rejected():
    st = _state()
    out = llm_output(["b_lines"], likelihood="very high")
    assert any("likelihood" in e for e in validate_llm_output(out, st))


# ------------------------------------------------------------------ parsing
@prop(HALLUCINATION)
def test_json_wrapped_in_code_fences_is_parsed():
    raw = '```json\n{"differential": [], "uncertainty": "x"}\n```'
    assert parse_llm_json(raw)["uncertainty"] == "x"


@prop(HALLUCINATION)
def test_json_surrounded_by_prose_is_parsed():
    raw = 'Here is my answer:\n{"differential": [], "uncertainty": "x"}\nHope that helps.'
    assert parse_llm_json(raw)["uncertainty"] == "x"


@prop(HALLUCINATION)
def test_unparseable_output_does_not_crash_the_pipeline():
    """A model that returns prose must degrade to a withheld differential, not an exception
    reaching the caller."""
    st = _state()
    out = reason(st, llm_fn=lambda s, u: "I cannot answer that.")
    assert out["differential"] is None
    assert out["validation_errors"], out


# ------------------------------------------------------------------ end of the guard
@prop(HALLUCINATION)
def test_a_hallucinated_differential_is_withheld_not_annotated():
    """The design choice under test: a differential citing tests that were never run is
    withheld entirely rather than displayed beside a warning, because a warning beside a
    plausible differential is routinely ignored."""
    st = _state(labs={})
    import json
    bad = json.dumps(llm_output(["b_lines", "elevated troponin"]))
    out = reason(st, llm_fn=lambda s, u: bad)
    assert out["validation_errors"]
    assert out.get("differential_withheld") is True, out
