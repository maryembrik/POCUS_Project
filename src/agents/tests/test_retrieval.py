"""Retrieval: relevance floor, citation checking, and placeholder honesty.

Retrieval is a second route by which evidence reaches the prompt, so it is checked as
closely as the model's own output. An irrelevant passage is worse than none, because the
answer then looks grounded.
"""
import json

from src.agents import schema as S
from src.agents.clinical_state import build_clinical_state
from src.agents.llm import ScriptedBackend
from src.agents.reasoning import check_citations, reason
from src.agents.retrieval import (Retriever, build_query, corpus_status, is_grounded,
                                  load_corpus, retrieval_note)
from .helpers import prop, bundle, lung_report

RETRIEVAL = "Retrieval grounding"


def _state():
    return build_clinical_state(
        bundle(triage=S.make_triage("high", 0.79,
                                    features={"o2sat": 90, "pulse": 118, "respr": 24}),
               ultrasound={"lung": lung_report(b_lines=0.86)},
               clinical={"age": 74, "chief_complaint": "acute breathlessness"}),
        labs={})


# ------------------------------------------------------------------ corpus
@prop(RETRIEVAL)
def test_corpus_loads_and_every_passage_declares_its_source():
    data = load_corpus()
    assert data["passages"]
    for p in data["passages"]:
        assert p["source"], p["id"]
        assert p["status"] in ("placeholder", "sourced"), p


@prop(RETRIEVAL)
def test_placeholder_passages_are_counted_not_hidden():
    st = corpus_status(load_corpus())
    assert sum(st.values()) == len(load_corpus()["passages"])


# ------------------------------------------------------------------ query construction
@prop(RETRIEVAL)
def test_the_query_is_built_from_what_is_present_not_what_is_absent():
    """Querying on absent tests retrieves text about them and invites the model to reason
    from evidence that was never obtained."""
    q = build_query(_state()).lower()
    assert "breathlessness" in q
    assert "b lines" in q or "b_lines" in q
    assert "troponin" not in q, q          # absent, must not steer retrieval


@prop(RETRIEVAL)
def test_abnormal_vitals_reach_the_query():
    q = build_query(_state()).lower()
    assert "spo2" in q or "hr" in q, q


# ------------------------------------------------------------------ relevance floor
@prop(RETRIEVAL)
def test_a_relevant_query_retrieves_something():
    r = Retriever()
    hits = r.retrieve("B-lines on lung ultrasound in acute breathlessness")
    assert hits, "a query squarely on-corpus must retrieve"
    assert all(h["score"] >= r.floor for h in hits)


@prop(RETRIEVAL)
def test_an_off_topic_query_retrieves_nothing():
    """Reporting an irrelevant hit as a hit is the failure that matters: the answer then
    looks grounded while resting on text that does not bear on the case."""
    r = Retriever()
    assert r.retrieve("quarterly revenue forecast for a logistics company") == []


@prop(RETRIEVAL)
def test_hits_are_numbered_for_citation():
    hits = Retriever().retrieve("pleural effusion")
    assert [h["n"] for h in hits] == list(range(1, len(hits) + 1))


# ------------------------------------------------------------------ placeholder honesty
@prop(RETRIEVAL)
def test_placeholder_hits_are_not_grounding():
    hits = Retriever().retrieve("B-lines")
    assert hits and all(h["status"] == "placeholder" for h in hits)
    assert is_grounded(hits) is False
    assert "PLACEHOLDER" in retrieval_note(hits)


@prop(RETRIEVAL)
def test_no_retrieval_is_reported_as_such():
    assert is_grounded([]) is False
    assert "no relevant passage" in retrieval_note([])


# ------------------------------------------------------------------ citation checking
def _answer(supporting, next_step="obtain a troponin"):
    return {"differential": [{"diagnosis": "Pulmonary oedema", "likelihood": "moderate",
                              "supporting": supporting, "contradicting": [],
                              "limitations": []}],
            "missing_information": ["troponin"], "uncertainty": "u",
            "recommended_next_step": next_step}


@prop(RETRIEVAL)
def test_citing_a_passage_that_was_retrieved_is_accepted():
    hits = Retriever().retrieve("B-lines")
    assert check_citations(_answer(["b lines [1]"]), hits) == []


@prop(RETRIEVAL)
def test_citing_beyond_what_was_retrieved_is_rejected():
    """A bracketed number reads as provenance, which makes an invented one more persuasive
    than an invented laboratory value, not less."""
    hits = Retriever().retrieve("B-lines")[:1]
    errs = check_citations(_answer(["b lines [3]"]), hits)
    assert errs and "[3]" in errs[0]


@prop(RETRIEVAL)
def test_citing_anything_when_nothing_was_retrieved_is_rejected():
    errs = check_citations(_answer(["b lines [1]"]), [])
    assert errs


@prop(RETRIEVAL)
def test_claiming_guideline_support_without_retrieval_is_rejected():
    """The same failure without the bracket."""
    errs = check_citations(
        _answer(["b lines"], next_step="Guidelines recommend a chest X-ray"), [])
    assert errs and "external evidence" in errs[0]


@prop(RETRIEVAL)
def test_an_ordinary_answer_without_citations_is_accepted():
    assert check_citations(_answer(["b lines"]), []) == []


# ------------------------------------------------------------------ end to end
@prop(RETRIEVAL)
def test_reason_records_what_grounding_may_be_claimed():
    hits = Retriever().for_state(_state())
    out = reason(_state(), llm_fn=None, retrieved=hits)
    assert out["retrieval"]["passages"] == len(hits)
    assert out["retrieval"]["grounded"] is False      # corpus is still placeholders
    assert "PLACEHOLDER" in out["retrieval"]["note"]


@prop(RETRIEVAL)
def test_retrieved_passages_reach_the_prompt():
    hits = Retriever().for_state(_state())
    out = reason(_state(), llm_fn=None, retrieved=hits)
    user = out["prompt"]["user"]
    assert "RETRIEVED EVIDENCE" in user
    assert "[1]" in user


@prop(RETRIEVAL)
def test_a_fabricated_citation_withholds_the_differential():
    hits = Retriever().for_state(_state())[:1]
    bad = json.dumps(_answer(["b lines [4]"]))
    out = reason(_state(), llm_fn=ScriptedBackend(bad, bad), retrieved=hits,
                 max_revisions=1)
    assert out["differential_withheld"] is True
