"""Module 4 and 5: severity, alerts, examinations, scenario routing, and the report.

Everything under test here is deterministic. The point of the module is that "is this patient
critical?" is answered from the structured state rather than by asking a language model, so
these tests never construct one -- if any of this behaviour depended on a model, they would
fail to run rather than pass by luck.

The properties that matter most are the ones about what CANNOT happen: severity is never
lowered by a reassuring triage tier, a modality nobody examined is never silently omitted from
the report, and a therapeutic consideration is never produced without a sourced protocol behind
it.
"""
import json
import tempfile
from pathlib import Path

from src.agents import schema as S
from src.agents.clinical.clinical_state import build_clinical_state
from src.agents.clinical.decision_support import (
    critical_alerts, decision_support, load_thresholds, recommended_examinations,
    route_scenario, severity_level, therapeutic_considerations)
from src.agents.clinical.llm import ScriptedBackend
from src.agents.clinical.reasoning import escalation_decision, reason
from src.agents.clinical.report import (
    archive_report, build_report, render_report, standardised_conclusion)
from .helpers import cite, prop, bundle, lung_report

SEVERITY = "Severity and alerts"
EXAMS = "Examination recommendations"
SCENARIO = "Scenario routing"
REPORTING = "Automated reporting"


def _state(*, tier="high", tconf=0.79, labs=None, vitals=None, ultrasound=None,
           complaint="acute breathlessness"):
    return build_clinical_state(
        bundle(triage=S.make_triage(tier, tconf, features=vitals or {}),
               ultrasound=ultrasound if ultrasound is not None
               else {"lung": lung_report(b_lines=0.86)},
               clinical={"age": 71, "sex": "F", "chief_complaint": complaint}),
        labs=labs if labs is not None else {"troponin": 9.0, "lactate": 1.2})


def _support(state):
    esc = escalation_decision(state)
    return esc, decision_support(state, esc)


# ------------------------------------------------------------------ thresholds
@prop(SEVERITY)
def test_every_threshold_lives_in_the_config_file():
    """Cutoffs scattered through code cannot be audited or cited in a report. This pins that
    they are all in one file with a version and a stated provenance."""
    T = load_thresholds()
    assert T["version"] and T["provenance"]
    for name in ("spo2", "sbp", "hr", "rr", "temp"):
        assert name in T["vitals"], name
    for name in ("lactate", "troponin", "ph"):
        assert name in T["labs"], name


@prop(SEVERITY)
def test_an_alert_names_the_threshold_it_fired_on():
    """A reader must be able to check the alert against the config rather than trust the
    label."""
    st = _state(vitals={"o2sat": 85, "pulse": 118, "respr": 24})
    alerts = critical_alerts(st, escalation_decision(st))
    hypox = [a for a in alerts if a["type"] == "SEVERE_HYPOXAEMIA"]
    assert hypox, [a["type"] for a in alerts]
    assert hypox[0]["threshold"] == load_thresholds()["vitals"]["spo2"]["critical_below"]
    assert hypox[0]["value"] == 85


# ------------------------------------------------------------------ alerts
@prop(SEVERITY)
def test_physiological_alerts_fire_from_measured_values():
    st = _state(vitals={"o2sat": 85, "pulse": 140, "respr": 34})
    types = {a["type"] for a in critical_alerts(st, escalation_decision(st))}
    assert "SEVERE_HYPOXAEMIA" in types
    assert "SEVERE_TACHYCARDIA" in types
    assert "SEVERE_TACHYPNOEA" in types


@prop(SEVERITY)
def test_a_normal_patient_raises_no_physiological_alert():
    st = _state(tier="low", vitals={"o2sat": 98, "pulse": 72, "respr": 14},
                ultrasound={"lung": lung_report(b_lines=None)},
                labs={"troponin": 4.0, "lactate": 1.0})
    physio = [a for a in critical_alerts(st, escalation_decision(st))
              if a.get("source") in ("vitals", "labs")]
    assert physio == [], physio


@prop(SEVERITY)
def test_structural_alerts_fire_on_the_record_not_the_patient():
    """A patient can be in danger because of what was not done, and no vital sign shows it."""
    st = _state(labs={})                       # every key lab absent, positive imaging present
    types = {a["type"] for a in critical_alerts(st, escalation_decision(st))}
    assert "POSITIVE_IMAGING_KEY_LABS_ABSENT" in types, types


@prop(SEVERITY)
def test_an_unassessed_modality_is_a_critical_alert():
    st = _state(ultrasound={"lung": lung_report(b_lines=0.8),
                            "heart": S.make_report("heart", [], status="not_supported")})
    alerts = critical_alerts(st, escalation_decision(st), route_scenario(st))
    gap = [a for a in alerts if a["type"] == "MODALITY_NOT_ASSESSED"]
    assert gap, [a["type"] for a in alerts]
    assert "heart" in gap[0]["modalities"]


@prop(SEVERITY)
def test_an_agent_conflict_is_a_critical_alert():
    st = build_clinical_state(
        bundle(triage=S.make_triage("low", 0.88),
               ultrasound={"heart": S.make_report(
                   "heart", [S.make_finding("severe dysfunction", 0.74)],
                   reliability={"confidence_calibrated": True, "has_normal_class": True})}),
        labs={"troponin": 5.0, "lactate": 1.1})
    types = {a["type"] for a in critical_alerts(st, escalation_decision(st))}
    assert "AGENT_CONFLICT" in types, types


# ------------------------------------------------------------------ severity
@prop(SEVERITY)
def test_severity_is_high_when_escalation_fires():
    st = _state(labs={})
    esc, sup = _support(st)
    assert esc["escalate"] is True
    assert sup["severity"]["severity"] == "HIGH"


@prop(SEVERITY)
def test_a_reassuring_triage_tier_never_lowers_severity():
    """The property the whole module rests on. A low triage tier beside a critical alert is
    precisely the disagreement the system exists to surface; averaging the two would bury it,
    and this is the case where a naive implementation would report LOW."""
    st = _state(tier="low", tconf=0.9, vitals={"o2sat": 84})
    esc, sup = _support(st)
    assert sup["severity"]["triage_urgency"] == "LOW"
    assert sup["severity"]["severity"] == "HIGH", sup["severity"]
    assert any("critical alert" in r for r in sup["severity"]["reasons"])


@prop(SEVERITY)
def test_severity_is_low_only_when_nothing_is_wrong():
    st = _state(tier="low", vitals={"o2sat": 98, "pulse": 70, "respr": 14},
                ultrasound={"lung": lung_report(b_lines=None)},
                labs={"troponin": 4.0, "lactate": 1.0})
    esc, sup = _support(st)
    if not esc["escalate"]:
        assert sup["severity"]["severity"] == "LOW", sup["severity"]


@prop(SEVERITY)
def test_severity_declares_that_it_is_not_model_derived():
    st = _state()
    _, sup = _support(st)
    assert "not produced by the language model" in sup["severity"]["derivation"]


@prop(SEVERITY)
def test_decision_support_survives_a_model_that_never_ran():
    """Severity and alerts must not depend on the model having answered."""
    st = _state()
    out = reason(st, llm_fn=None)
    assert out["decision_support"]["severity"]["severity"]
    assert out["decision_support"]["alerts"] is not None


# ------------------------------------------------------------------ scenario routing
@prop(SCENARIO)
def test_each_priority_presentation_routes():
    for complaint, expected in (("acute breathlessness", "acute_dyspnoea"),
                                ("hypotension and collapse", "undifferentiated_shock"),
                                ("blunt trauma after a fall", "trauma"),
                                ("found unresponsive, CPR in progress",
                                 "cardiorespiratory_arrest")):
        r = route_scenario(_state(complaint=complaint))
        assert r["scenario"] == expected, (complaint, r)


@prop(SCENARIO)
def test_an_unmatched_complaint_is_unclassified_rather_than_forced():
    """Routing exists to say which modalities were expected. Forcing a case into the nearest
    scenario would manufacture a gap that nobody actually left."""
    r = route_scenario(_state(complaint="routine follow-up of a stable rash",
                              vitals={"o2sat": 98, "sbp": 120}))
    assert r["scenario"] is None
    assert r["not_assessed"] == []


@prop(SCENARIO)
def test_routing_names_the_modality_the_scenario_expected_and_nobody_performed():
    st = _state(complaint="blunt trauma after a fall")       # expects fast + lung
    r = route_scenario(st)
    assert r["scenario"] == "trauma"
    assert "fast" in r["not_assessed"], r


# ------------------------------------------------------------------ examinations
@prop(EXAMS)
def test_absent_key_labs_are_recommended_at_high_priority():
    st = _state(labs={})
    esc, sup = _support(st)
    recs = {r["exam"].lower(): r for r in sup["additional_examinations"]}
    assert "troponin" in recs and recs["troponin"]["priority"] == "HIGH"


@prop(EXAMS)
def test_an_unperformed_scan_outranks_a_blood_test():
    st = _state(complaint="blunt trauma after a fall", labs={})
    esc, sup = _support(st)
    recs = sup["additional_examinations"]
    assert recs, recs
    assert recs[0]["kind"] == "imaging", recs[0]


@prop(EXAMS)
def test_the_differential_can_raise_priority_but_cannot_invent_an_examination():
    """A hallucinated diagnosis must not be able to cause a hallucinated recommendation."""
    st = _state(labs={"troponin": 9.0, "lactate": 1.2})       # nothing absent but the rest
    esc = escalation_decision(st)
    invented = {"differential": [{"diagnosis": "Pulmonary Embolism", "likelihood": "high"}]}
    recs = recommended_examinations(st, esc, invented, route_scenario(st))
    names = {r["exam"].lower() for r in recs}
    assert "troponin" not in names, "a measured test must never be recommended as missing"
    for r in recs:
        assert r["exam"] in st["missing"]["labs"] + st["missing"]["vitals"] \
            or r["kind"] == "imaging", r


@prop(EXAMS)
def test_recommendations_are_not_orders():
    st = _state(labs={})
    _, sup = _support(st)
    for r in sup["additional_examinations"]:
        assert not r["reason"].lower().startswith(("give", "administer", "start", "order"))


# ------------------------------------------------------------------ therapeutics
@prop(EXAMS)
def test_no_protocol_means_no_therapeutic_suggestion():
    """The mechanism exists; the corpus contains no sourced protocol yet. Returning nothing and
    saying why is the same discipline the corpus applies to placeholders, and for the same
    reason: a suggestion resting on the model's training data is indistinguishable, to a
    reader, from one backed by a protocol."""
    st = _state()
    out = therapeutic_considerations(st, route_scenario(st), retrieved=[])
    assert out["considerations"] == []
    assert "no approved protocol" in out["status"]


@prop(EXAMS)
def test_a_sourced_protocol_passage_is_surfaced_with_its_citation():
    st = _state()
    passage = {"id": "P01", "topic": "protocol_acute_dyspnoea", "status": "sourced",
               "source": "Example ED protocol, 2024", "text": "Consider X per protocol.",
               "n": 1, "score": 0.4}
    out = therapeutic_considerations(st, route_scenario(st), retrieved=[passage])
    assert len(out["considerations"]) == 1
    c = out["considerations"][0]
    assert c["basis"] == "Example ED protocol, 2024" and c["passage"] == "P01"
    assert c["status"] == "CONSIDER" and "clinician" in c["disclaimer"].lower()


@prop(EXAMS)
def test_an_unsourced_protocol_passage_is_not_surfaced():
    st = _state()
    placeholder = {"id": "P02", "topic": "protocol_shock", "status": "placeholder",
                   "source": "REPLACE", "text": "PASTE", "n": 1, "score": 0.4}
    assert therapeutic_considerations(st, route_scenario(st),
                                      retrieved=[placeholder])["considerations"] == []


# ------------------------------------------------------------------ reporting
def _full_report(state):
    esc = escalation_decision(state)
    ids = cite(state, "b_lines")
    answer = {"differential": [{"diagnosis": "Pulmonary oedema", "likelihood": "moderate",
                                "supporting": ids, "contradicting": [], "limitations": []}],
              "missing_information": ["d_dimer"], "uncertainty": "u",
              "recommended_next_step": "obtain a d-dimer"}
    out = reason(state, llm_fn=ScriptedBackend(json.dumps(answer)), max_revisions=0)
    return out, build_report(state, out, out["decision_support"])


@prop(REPORTING)
def test_the_report_contains_every_required_section():
    state = _state()
    _, rep = _full_report(state)
    for key in ("context", "triage", "pocus", "quantitative_measures", "values",
                "reasoning", "decision_support", "safety", "conclusion"):
        assert key in rep, key
    assert rep["safety"]["disclaimer"]


@prop(REPORTING)
def test_a_modality_never_assessed_gets_its_own_section_not_an_omission():
    """Safety-relevant rather than cosmetic. Omitting it leaves a reader to infer absence from
    silence, and silence is what a normal result looks like."""
    state = _state(ultrasound={"lung": lung_report(b_lines=0.8),
                               "heart": S.make_report("heart", [], status="not_supported")})
    _, rep = _full_report(state)
    assert "heart" in rep["pocus"]["not_assessed"]
    text = render_report(rep)
    assert "NOT ASSESSED" in text
    assert "NOT a negative result" in text


@prop(REPORTING)
def test_a_withheld_differential_is_rendered_as_withheld():
    """A report that simply lacks a differential is indistinguishable from one where the model
    was never asked."""
    state = _state()
    bad = {"differential": [{"diagnosis": "X", "likelihood": "moderate",
                             "supporting": ["stable vitals"], "contradicting": [],
                             "limitations": []}],
           "missing_information": ["d_dimer"], "uncertainty": "u",
           "recommended_next_step": "obtain a d-dimer"}
    out = reason(state, llm_fn=ScriptedBackend(json.dumps(bad)))
    rep = build_report(state, out, out["decision_support"])
    assert rep["reasoning"]["withheld"] is True
    assert rep["reasoning"]["withheld_reason"]
    assert "WITHHELD" in render_report(rep)


@prop(REPORTING)
def test_the_conclusion_agrees_with_the_report_it_summarises():
    """Written by template rather than generated, so that the line a busy reader acts on cannot
    say something different from the body."""
    state = _state(labs={})
    _, rep = _full_report(state)
    concl = standardised_conclusion(rep)
    assert rep["decision_support"]["severity"]["severity"] in concl
    if rep["safety"]["escalation"]["escalate"]:
        assert "ESCALATED" in concl
    if rep["pocus"]["not_assessed"]:
        assert "NOT ASSESSED" in concl


@prop(REPORTING)
def test_rendering_is_deterministic():
    state = _state()
    _, rep = _full_report(state)
    assert render_report(rep) == render_report(rep)


@prop(REPORTING)
def test_archiving_writes_both_formats_and_a_content_hash():
    state = _state()
    _, rep = _full_report(state)
    with tempfile.TemporaryDirectory() as tmp:
        paths = archive_report(rep, tmp)
        assert Path(paths["json"]).exists() and Path(paths["txt"]).exists()
        assert len(paths["sha256"]) == 64
        stored = json.loads(Path(paths["json"]).read_text(encoding="utf8"))
        assert stored["content_sha256"] == paths["sha256"]


@prop(REPORTING)
def test_the_hash_ignores_the_timestamp_so_a_rerun_is_distinguishable_from_a_change():
    state = _state()
    _, a = _full_report(state)
    _, b = _full_report(state)
    b["generated_at"] = "2099-01-01T00:00:00+00:00"
    with tempfile.TemporaryDirectory() as t1, tempfile.TemporaryDirectory() as t2:
        assert archive_report(a, t1)["sha256"] == archive_report(b, t2)["sha256"]


@prop(EXAMS)
def test_a_protocol_only_surfaces_for_the_scenario_it_covers():
    """Scenario-matched, not merely protocol-shaped. The initial management of undifferentiated
    shock is not advice about a trauma case, and surfacing it there would be worse than
    surfacing nothing because it would arrive with a real citation attached."""
    shock_protocol = {"id": "P31", "topic": "protocol_undifferentiated_shock",
                      "status": "sourced", "source": "Hasanin et al. 2024 (CC BY 4.0)",
                      "text": "Maintain ABCs, INfuse, INvestigate, Ultrasound, Treat, "
                              "Stabilize.", "n": 1, "score": 0.4}

    shock = _state(complaint="hypotension and collapse")
    out = therapeutic_considerations(shock, route_scenario(shock), [shock_protocol])
    assert len(out["considerations"]) == 1, out

    trauma = _state(complaint="blunt trauma after a fall")
    out = therapeutic_considerations(trauma, route_scenario(trauma), [shock_protocol])
    assert out["considerations"] == [], "a shock protocol must not surface on a trauma case"


@prop(EXAMS)
def test_an_unclassified_encounter_matches_no_protocol():
    st = _state(complaint="routine review", vitals={"o2sat": 98, "sbp": 120})
    p = {"id": "P31", "topic": "protocol_undifferentiated_shock", "status": "sourced",
         "source": "s", "text": "t", "n": 1, "score": 0.4}
    assert route_scenario(st)["scenario"] is None
    assert therapeutic_considerations(st, route_scenario(st), [p])["considerations"] == []


@prop(EXAMS)
def test_the_shock_protocol_is_reachable_from_the_live_corpus():
    """End to end rather than with a hand-built passage: the units really are in the corpus,
    really are retrieved for a shock presentation, and really do produce a cited
    consideration."""
    st = _state(complaint="undifferentiated shock with hypotension",
                vitals={"sbp": 82, "pulse": 124})
    out = therapeutic_considerations(st, route_scenario(st))
    assert out["status"] == "protocol-backed", out["status"]
    assert any("Hasanin" in c["basis"] for c in out["considerations"]), out
