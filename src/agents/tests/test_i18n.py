"""French rendering, and the property that makes it safe to show a clinician.

The risk in translating a clinical system is not a clumsy phrase. It is a message that reaches
a French screen still in English, or worse, silently empty -- a doctor who cannot read the
alert is a doctor who is not warned. So the property under test is COVERAGE: every alert,
trigger, examination reason and declared limit the pipeline can emit must have a French form,
and `untranslated()` must be empty after the whole benchmark has been rendered.

The second property is that French is a rendering of the same computed numbers, not a second
derivation: the value and the threshold in the French alert are the ones the alert fired on.
"""
import re

from src.agents import i18n
from src.agents.clinical.decision_support import decision_support
from src.agents.clinical.reasoning import escalation_decision
from src.agents.clinical.run_case import SCENARIOS, build
from .helpers import prop

I18N = "French rendering coverage"


def _all_supports():
    for name in SCENARIOS:
        state = build(name)
        esc = escalation_decision(state)
        yield name, state, esc, decision_support(state, esc, None)


@prop(I18N)
def test_every_message_the_benchmark_emits_has_a_french_form():
    """The coverage property. A new clinical message cannot reach French in English."""
    i18n.reset_untranslated()
    for _name, state, esc, sup in _all_supports():
        for a in sup["alerts"]:
            assert i18n.alert(a), "empty French alert"
        for t in esc["triggers"]:
            i18n.trigger(t)
        for x in sup["additional_examinations"]:
            i18n.exam_name(x["exam"])
            i18n.exam_reason(x["reason"])
        for lim in state["imaging"]["out_of_scope"]:
            i18n.limit(lim)
        i18n.scenario(sup["scenario"]["label"])
    assert i18n.untranslated() == [], (
        "these reached a French screen with no French rendering: "
        + "; ".join(i18n.untranslated()))


@prop(I18N)
def test_every_label_in_thresholds_json_has_a_french_form():
    """Walks the config, not the benchmark.

    Two labels reached a French screen in English while the coverage test above reported full
    coverage, because the benchmark's five encounters never crossed those particular bounds.
    A threshold file is edited far more often than this module, so the file itself is the
    thing to check: any label a clinician could ever see must be translated, whether or not a
    test case happens to trigger it.
    """
    import json
    from pathlib import Path

    path = Path("src/agents/clinical/thresholds.json")
    cfg = json.loads(path.read_text(encoding="utf8"))
    missing = []
    for group in ("vitals", "labs"):
        for spec in cfg[group].values():
            for field, label in spec.items():
                if not field.endswith("label"):
                    continue
                key = str(label).upper().replace(" ", "_")
                if key not in i18n.ALERT_TYPE:
                    missing.append(f"{label} -> {key}")
    assert not missing, "thresholds.json labels with no French form: " + "; ".join(missing)


@prop(I18N)
def test_the_french_alert_carries_the_numbers_the_alert_fired_on():
    """Rendered from the structured fields, not translated from the English sentence."""
    for _name, _state, _esc, sup in _all_supports():
        for a in sup["alerts"]:
            if "measurement" not in a:
                continue
            fr = i18n.alert(a)
            for num in (a["value"], a["threshold"]):
                assert f"{float(num):g}".replace(".", ",") in fr, (
                    f"{num} missing from {fr!r}")


@prop(I18N)
def test_a_decimal_is_written_with_a_comma():
    """38.6 read by someone not expecting a point is 386, and a temperature is where that
    matters. French writes the decimal comma."""
    fr = i18n.alert({"severity": "WARNING", "type": "FEVER", "measurement": "temp",
                     "value": 38.6, "threshold": 38.0,
                     "message": "temp 38.6 C is above the configured warning bound of 38.0 C"})
    assert "38,6" in fr, fr
    assert "38.6" not in fr, "a decimal point survived into the French text"


@prop(I18N)
def test_direction_is_preserved():
    """A bradycardia rendered as a tachycardia would be worse than no translation at all."""
    low = i18n.alert({"severity": "CRITICAL", "type": "BRADYCARDIA", "measurement": "hr",
                      "value": 38.0, "threshold": 40.0,
                      "message": "hr 38.0 bpm is below the configured critical bound of 40.0 bpm"})
    high = i18n.alert({"severity": "WARNING", "type": "TACHYCARDIA", "measurement": "hr",
                       "value": 122.0, "threshold": 100.0,
                       "message": "hr 122.0 bpm is above the configured warning bound of 100.0 bpm"})
    assert "en dessous de" in low and "au-dessus de" not in low
    assert "au-dessus de" in high and "en dessous de" not in high


@prop(I18N)
def test_a_gap_in_the_record_is_not_rendered_as_a_negative_result():
    """The sentence the whole project exists to keep true, in the other language."""
    fr = i18n.alert({"severity": "CRITICAL", "type": "MODALITY_NOT_ASSESSED",
                     "modalities": ["heart", "lung"],
                     "message": "expected but never assessed: heart, lung"})
    assert "jamais réalisé" in fr
    assert "pas un résultat négatif" in fr
    assert "cœur" in fr and "poumon" in fr


@prop(I18N)
def test_an_unknown_message_is_reported_rather_than_silently_passed():
    """The failure mode must be loud. An unrecognised trigger is recorded, not hidden."""
    i18n.reset_untranslated()
    out = i18n.trigger("some trigger form that does not exist yet")
    assert out == "some trigger form that does not exist yet"
    assert i18n.untranslated() == ["some trigger form that does not exist yet"]
    i18n.reset_untranslated()
