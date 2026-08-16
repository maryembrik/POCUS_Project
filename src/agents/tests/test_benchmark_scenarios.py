"""The five benchmark scenarios, checked without a model.

Each case exists to exercise one behaviour of the safety layer. What the language model
later says about them is a separate question -- these tests fix what the deterministic part
must do, so a change in the model can never quietly change the benchmark's meaning.
"""
from src.agents.reasoning import escalation_decision
from src.agents.run_case import SCENARIOS, build
from .helpers import prop

BENCH = "Benchmark scenarios"


@prop(BENCH)
def test_every_scenario_builds_a_valid_state():
    from src.agents.clinical_state import validate_state
    for name in SCENARIOS:
        st = build(name)
        assert validate_state(st) == [], f"{name}: {validate_state(st)}"


@prop(BENCH)
def test_concordant_is_the_only_case_answered_directly():
    """The policy must be able to say no. If every case escalates, the escalation carries no
    information and the system provides no decision support."""
    direct = [n for n in SCENARIOS if not escalation_decision(build(n))["escalate"]]
    assert direct == ["concordant"], direct


@prop(BENCH)
def test_missing_escalates_on_the_absent_key_lab():
    d = escalation_decision(build("missing"))
    assert any("key lab" in t for t in d["triggers"]), d["triggers"]


@prop(BENCH)
def test_conflict_escalates_on_disagreement():
    d = escalation_decision(build("conflict"))
    assert any("disagree" in t for t in d["triggers"]), d["triggers"]


@prop(BENCH)
def test_reassuring_still_escalates_because_nothing_can_be_excluded():
    """The hardest of the five. Everything is negative, the vitals are normal and the key
    labs resulted -- and the system must STILL not declare the patient well, because the
    lung module has no healthy class. A negative read is the absence of the pathologies it
    knows, not the absence of pathology."""
    st = build("reassuring")
    assert not any(f["detected"] for f in st["imaging"]["findings"])
    assert st["missing"]["labs"] != [], "some labs are absent even here"
    d = escalation_decision(st)
    assert d["escalate"] is True, d
    assert any("cannot exclude" in t for t in d["triggers"]), d["triggers"]


@prop(BENCH)
def test_reassuring_records_the_negatives_rather_than_silence():
    """The distinction that makes the case meaningful: four findings screened and not seen,
    which is different from four findings never looked for."""
    st = build("reassuring")
    labels = {f["label"] for f in st["imaging"]["findings"]}
    assert len(labels) == 4, labels
    assert all(f["detected"] is False for f in st["imaging"]["findings"])


@prop(BENCH)
def test_not_assessed_escalates_because_the_scan_did_not_happen():
    st = build("not_assessed")
    assert "heart" in st["imaging"]["organs_not_assessed"]
    d = escalation_decision(st)
    assert d["escalate"] is True, d
    assert any("not assessed" in t for t in d["triggers"]), d["triggers"]


@prop(BENCH)
def test_an_unassessed_organ_contributes_no_findings():
    """It must not appear as a negative result anywhere."""
    st = build("not_assessed")
    assert not any(f["organ"] == "heart" for f in st["imaging"]["findings"])


@prop(BENCH)
def test_the_five_cases_exercise_five_different_trigger_sets():
    """If two cases fire the same triggers, one of them is not earning its runtime."""
    sets = {name: tuple(sorted(
        t.split(":")[0].split("(")[0].strip()
        for t in escalation_decision(build(name))["triggers"]))
        for name in SCENARIOS}
    assert len(set(sets.values())) == len(SCENARIOS), sets
