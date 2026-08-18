"""Enumerated evidence: the model cites identifiers, so it cannot invent evidence.

Free-text evidence was the interface for the first version, and it was the interface that
failed. On the five benchmark cases the model wrote "stable vitals" for a patient whose vitals
were never in the state, called a troponin of 5.0 "elevated" against a state that printed it as
normal, and -- once retrieval arrived -- lifted sentences out of the corpus into
`contradicting` as though they were observations about the patient.

Every one of those was caught after the fact. These tests are about the ones that can no
longer be written down: there is no identifier for a fact the state does not hold, none for an
absent test, and none for a sentence out of a guideline.

What enumeration does NOT fix is equally worth stating, and `test_a_real_finding_can_still_be
_cited_against_the_wrong_diagnosis` says so: whether a genuinely assessed finding bears on a
given diagnosis is a judgement, and no list of identifiers settles it.
"""
import json

from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state, build_evidence, \
    render_evidence
from src.agents.clinical.llm import ScriptedBackend
from src.agents.clinical.reasoning import check_evidence_ids, reason, resolve_evidence
from .helpers import HALLUCINATION, MISSING_NOT_NORMAL, cite, prop, bundle, lung_report

EVIDENCE = "Enumerated evidence"


def _state(labs=None):
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79,
                                    features={"o2sat": 90, "pulse": 118, "respr": 24}),
               ultrasound={"lung": lung_report(b_lines=0.86)}),
        labs=labs if labs is not None else {"troponin": 5.0})


def _answer(supporting, contradicting=()):
    return {"differential": [{"diagnosis": "Pulmonary oedema", "likelihood": "moderate",
                              "supporting": list(supporting),
                              "contradicting": list(contradicting), "limitations": []}],
            "missing_information": ["d_dimer"], "uncertainty": "u",
            "recommended_next_step": "obtain a d-dimer"}


# ------------------------------------------------------------------ what can be cited
@prop(EVIDENCE)
def test_every_citable_fact_gets_an_identifier():
    ev = build_evidence(_state())
    labels = {e["label"] for e in ev}
    assert "b_lines" in labels
    assert "hr" in labels and "spo2" in labels
    assert "troponin" in labels
    assert [e["id"] for e in ev] == [f"E{i}" for i in range(1, len(ev) + 1)]


@prop(MISSING_NOT_NORMAL)
def test_an_absent_test_has_no_identifier():
    """The central structural guarantee. "An absent test cannot support or contradict
    anything" stops being a rule the model is asked to obey and becomes something it has no
    way to say."""
    st = _state(labs={})                       # nothing resulted
    assert st["missing"]["labs"], "fixture must have absent labs to be meaningful"
    labels = {e["label"] for e in build_evidence(st)}
    for absent in st["missing"]["labs"]:
        assert absent not in labels, f"{absent} was never measured but has an identifier"


@prop(EVIDENCE)
def test_the_qualifier_travels_with_the_identifier():
    """The troponin misread, closed at the source: the model cites E-something and the word
    NORMAL comes with it, so there is no step at which it can be relabelled."""
    ev = build_evidence(_state(labs={"troponin": 5.0}))
    trop = next(e for e in ev if e["label"] == "troponin")
    assert "NORMAL" in trop["text"]
    assert "5.0" in trop["text"]


@prop(EVIDENCE)
def test_identifiers_are_stable_for_the_same_state():
    a = build_evidence(_state())
    b = build_evidence(_state())
    assert [(e["id"], e["text"]) for e in a] == [(e["id"], e["text"]) for e in b]


@prop(EVIDENCE)
def test_screened_negatives_are_citable_but_unassessed_organs_are_not():
    rep = S.make_report(
        "lung", [S.make_finding("b lines", 0.86)],
        not_detected=[S.make_finding("consolidation", 0.09)],
        reliability={"confidence_calibrated": True, "has_normal_class": False})
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.8),
               ultrasound={"lung": rep,
                           "heart": S.make_report("heart", [], status="not_supported")}),
        labs={})
    labels = {e["label"] for e in build_evidence(st)}
    assert "consolidation" in labels, "a screened negative is an observation and is citable"
    assert "heart" not in labels, "an organ never assessed must not be citable"


# ------------------------------------------------------------------ rejection
@prop(HALLUCINATION)
def test_prose_in_an_evidence_array_is_rejected():
    st = _state()
    out = reason(st, llm_fn=ScriptedBackend(json.dumps(_answer(["stable vitals"]))))
    assert out["differential_withheld"] is True
    assert any("not an evidence identifier" in e for e in out["validation_errors"])


@prop(HALLUCINATION)
def test_an_identifier_that_does_not_exist_is_rejected():
    st = _state()
    out = reason(st, llm_fn=ScriptedBackend(json.dumps(_answer(["E999"]))))
    assert out["differential_withheld"] is True
    assert any("does not exist" in e for e in out["validation_errors"])


@prop(HALLUCINATION)
def test_a_retrieved_passage_cannot_be_cited_as_a_patient_finding():
    """The failure retrieval introduced. On the A/B run the model put 'lung sliding (low
    specificity)' -- a sentence out of corpus unit L09 -- into `contradicting`, and did the
    same with a sentence from C12. Neither is an observation about the patient."""
    st = _state()
    quoted = "absence of sonographic signs of RV overload or dysfunction"
    out = reason(st, llm_fn=ScriptedBackend(
        json.dumps(_answer(cite(st, "b_lines"), contradicting=[quoted]))))
    assert out["differential_withheld"] is True
    assert any("not an evidence identifier" in e for e in out["validation_errors"])


@prop(HALLUCINATION)
def test_a_fabrication_is_still_not_offered_a_revision():
    """Grounding failures stay fatal. A revision is for a judgement the model can reconsider,
    not for invention."""
    st = _state()
    backend = ScriptedBackend(json.dumps(_answer(["invented finding"])))
    reason(st, llm_fn=backend, max_revisions=2)
    assert len(backend.calls) == 1


# ------------------------------------------------------------------ resolution
@prop(EVIDENCE)
def test_identifiers_are_resolved_for_the_reader_and_kept_for_the_audit():
    st = _state()
    ids = cite(st, "b_lines", "hr")
    out = reason(st, llm_fn=ScriptedBackend(json.dumps(_answer(ids))))
    entry = out["differential"]["differential"][0]
    assert entry["supporting_ids"] == ids, "what the model cited must survive"
    assert all(s.startswith("E") is False for s in entry["supporting"]), entry["supporting"]
    assert any("b_lines" in s for s in entry["supporting"])


@prop(EVIDENCE)
def test_resolve_is_not_reached_with_an_unknown_identifier():
    """resolve_evidence drops what it cannot resolve, so calling it on an unchecked answer
    would make a fabricated identifier disappear rather than be reported. The pipeline checks
    first; this pins the ordering that makes that safe."""
    ev = build_evidence(_state())
    answer = _answer(["E1", "E999"])
    assert check_evidence_ids(answer, ev), "the check must object before anything resolves"
    resolve_evidence(answer, ev)
    assert answer["differential"][0]["supporting"] == [ev[0]["text"]]
    assert "E999" in answer["differential"][0]["supporting_ids"], \
        "the identifiers actually cited must still be recoverable"


# ------------------------------------------------------------------ the limit of the idea
@prop(EVIDENCE)
def test_a_real_finding_can_still_be_cited_against_the_wrong_diagnosis():
    """Enumeration bounds what may be cited, not what the citation is used to argue. A
    screened negative is a real observation with a real identifier, and citing it against a
    diagnosis it has no bearing on is still a reasoning error -- caught downstream by
    check_evidence_relationships, not here."""
    rep = S.make_report(
        "lung", [S.make_finding("b lines", 0.86)],
        not_detected=[S.make_finding("pleural thickening", 0.11)],
        reliability={"confidence_calibrated": True, "has_normal_class": False})
    st = build_clinical_state(
        bundle(triage=S.make_triage("high", 0.8, features={"o2sat": 90}),
               ultrasound={"lung": rep}), labs={})
    ev = build_evidence(st)
    answer = _answer(cite(st, "b lines"), contradicting=cite(st, "pleural thickening"))
    assert check_evidence_ids(answer, ev) == [], "the identifiers are all real"


@prop(EVIDENCE)
def test_the_evidence_block_renders_every_identifier():
    text = render_evidence(build_evidence(_state()))
    assert "AVAILABLE EVIDENCE" in text
    for e in build_evidence(_state()):
        assert f"[{e['id']}]" in text


# ------------------------------------------------------------------ deterministic output control
NORMALISATION = "Deterministic output control"


@prop(NORMALISATION)
def test_a_comma_joined_missing_information_is_split_not_revised():
    """Formatting is Python's job; clinical judgement is not.

    Splitting ["troponin, bnp"] changes no claim -- it parses a list the model formatted
    wrongly. The rule that Python never rewrites the model's answer is about judgement: a
    likelihood edited in post would reach a clinician as the model's reasoning when it is
    not. That is why the overclaiming wording of a recommendation is still sent back for
    revision rather than reworded here.

    Measured before it was automated: the model was given the corrected array to copy and did
    not copy it, across two revision rounds on three separate cases.
    """
    from src.agents.clinical.reasoning import normalize_missing_information
    st = _state(labs={})
    answer = _answer(cite(st, "b_lines"))
    answer["missing_information"] = ["troponin, bnp, d_dimer"]

    notes = normalize_missing_information(answer, st)
    assert answer["missing_information"] == ["troponin", "bnp", "d_dimer"]
    assert notes and "split" in notes[0]


@prop(NORMALISATION)
def test_normalisation_is_recorded_not_silent():
    """A silent correction would hide how often correction is needed, which is the thing worth
    reporting."""
    st = _state(labs={})
    answer = _answer(cite(st, "b_lines"))
    answer["missing_information"] = ["troponin, bnp"]
    out = reason(st, llm_fn=ScriptedBackend(json.dumps(answer)))
    assert out.get("normalizations"), out.keys()
    assert any("split" in n for n in out["normalizations"])


@prop(NORMALISATION)
def test_a_well_formed_list_is_left_alone():
    from src.agents.clinical.reasoning import normalize_missing_information
    st = _state(labs={})
    answer = _answer(cite(st, "b_lines"))
    answer["missing_information"] = ["troponin", "bnp"]
    assert normalize_missing_information(answer, st) == []
    assert answer["missing_information"] == ["troponin", "bnp"]


@prop(NORMALISATION)
def test_overclaiming_wording_is_revised_not_rewritten():
    """The boundary of the rule above. Python corrects a list; it does not reword a clinical
    recommendation, because the wording IS the claim."""
    st = _state()
    answer = _answer(cite(st, "b_lines", "hr", "rr", "spo2"))
    answer["recommended_next_step"] = "obtain a d-dimer to rule out pulmonary embolism"
    out = reason(st, llm_fn=ScriptedBackend(json.dumps(answer)), max_revisions=1)
    assert out["differential"]["recommended_next_step"] == \
        "obtain a d-dimer to rule out pulmonary embolism", "the wording must be untouched"
    assert out["warnings"], "and reported instead"


@prop(NORMALISATION)
def test_every_abnormal_value_reaches_the_reader_whether_or_not_the_model_cited_it():
    """The coverage fault survived two attempts to make the model account for every abnormal
    value. So the model stops being responsible for it: which values are abnormal is a fact
    about the state, and listing them is a display guarantee that belongs in code.

    This does not make the model more thorough, and check_evidence_coverage still says when it
    is not. It removes the consequence of the omission for the reader.
    """
    from src.agents.clinical.clinical_state import evidence_considered
    st = _state()
    only_one = cite(st, "b_lines")
    out = reason(st, llm_fn=ScriptedBackend(json.dumps(_answer(only_one))), max_revisions=0)

    rows = out["evidence_considered"]
    abnormal = {e["id"] for e in build_evidence(st)
                if e.get("flag") in ("high", "low") or e.get("detected")}
    assert {r["id"] for r in rows} == abnormal, "every abnormal value must be listed"
    assert any(r["used"] for r in rows), "the cited one is marked used"
    assert any(not r["used"] for r in rows), "the uncited ones are listed but unmarked"
    # and the omission is still reported rather than hidden by the display guarantee
    assert out.get("warnings"), out


@prop(NORMALISATION)
def test_a_group_prefix_is_dropped_not_glued_to_the_first_test():
    """Observed on the benchmark. The model wrote "labs: bnp, d_dimer, crp" as one string,
    and splitting on commas alone produced "labs: bnp" -- a test name that does not exist.
    The prefix labels the group, not the first test in it."""
    from src.agents.clinical.reasoning import normalize_missing_information
    st = _state(labs={})
    answer = _answer(cite(st, "b_lines"))
    answer["missing_information"] = ["labs: bnp, d_dimer", "heart: troponin, lactate"]
    normalize_missing_information(answer, st)
    assert answer["missing_information"] == ["bnp", "d_dimer", "troponin", "lactate"], \
        answer["missing_information"]


@prop(NORMALISATION)
def test_a_name_containing_a_colon_is_not_eaten():
    """Only known group words are stripped. Removing anything before a colon would take a
    real name with one in it."""
    from src.agents.clinical.reasoning import normalize_missing_information
    st = _state(labs={})
    answer = _answer(cite(st, "b_lines"))
    answer["missing_information"] = ["CT: chest with contrast, d_dimer"]
    normalize_missing_information(answer, st)
    assert answer["missing_information"] == ["CT: chest with contrast", "d_dimer"]
