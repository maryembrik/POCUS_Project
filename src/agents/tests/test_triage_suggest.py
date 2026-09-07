"""The urgency suggestion: that it maps the form to the model without silently inverting it.

The mapping between an intake form and a model's feature columns is where this kind of system
goes wrong quietly. Every failure mode below is one this code actually had, or one a single
edit could reintroduce, and none of them raises an exception -- they produce a confident number
that is wrong in a direction nobody checks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.triage import suggest as sg  # noqa: E402

from .helpers import prop  # noqa: E402

TRIAGE = "Triage suggestion mapping"

pytestmark = pytest.mark.skipif(not sg.available(),
                                reason="deployed triage artefacts not present")


CRITICAL = {"age": 81, "sex": "F", "arrival": "ambulance",
            "vitals": {"hr": 126, "sbp": 82, "dbp": 48, "rr": 28, "spo2": 88,
                       "temp": 38.9, "pain": 6}}
WELL = {"age": 24, "sex": "M", "arrival": "walk-in",
        "vitals": {"hr": 72, "sbp": 122, "dbp": 78, "rr": 14, "spo2": 99,
                   "temp": 36.8, "pain": 2}}


@prop(TRIAGE)
def test_a_critically_unwell_patient_is_not_graded_below_a_well_one():
    """The inversion test, and the reason this file exists.

    The first version of suggest() read the probability vector against the clinical order
    low/medium/high. The label encoder sorts its classes, so the columns are actually
    high/low/medium -- and a hypotensive, hypoxic, tachycardic 81-year-old arriving by
    ambulance came back LOW urgency at 0.87. Nothing raised: the shapes matched, the
    probabilities summed to one, and every field was populated.

    Asserting the ORDER between two cases rather than either label on its own means this test
    survives retraining. The model may grade the well patient low or medium as it improves;
    it must never grade them above the critical one.
    """
    order = ["low", "medium", "high"]
    critical = sg.suggest(CRITICAL)
    well = sg.suggest(WELL)
    assert order.index(critical["tier"]) > order.index(well["tier"]), (
        f"critical patient graded {critical['tier']!r}, well patient {well['tier']!r}")


@prop(TRIAGE)
def test_the_probability_keys_are_the_tiers_and_they_sum_to_one():
    s = sg.suggest(CRITICAL)
    assert set(s["probabilities"]) == {"low", "medium", "high"}
    assert abs(sum(s["probabilities"].values()) - 1.0) < 0.01
    # The reported tier is the one the probabilities support, unless the safety rule moved it.
    assert s["tier"] in ("low", "medium", "high")


@prop(TRIAGE)
def test_temperature_is_converted_and_not_merely_renamed():
    """The form collects Celsius; the model was fitted on Fahrenheit.

    37.0 passed straight through reads as profound hypothermia and makes the fever flag wrong
    for every patient. This fails loudly if the conversion is ever dropped, because 38.9 C is
    102.0 F and only one of those crosses the 100.4 fever threshold the training used.
    """
    row = sg.features_from_encounter(CRITICAL)
    assert row["temp_f"] == pytest.approx(38.9 * 9 / 5 + 32, abs=0.01)
    assert row["fever_flag"] == 1.0

    afebrile = sg.features_from_encounter(WELL)
    assert afebrile["temp_f"] == pytest.approx(36.8 * 9 / 5 + 32, abs=0.01)
    assert afebrile["fever_flag"] == 0.0


@prop(TRIAGE)
def test_a_vital_that_was_not_taken_stays_missing():
    """Not imputed. A substituted population mean is a measurement nobody made.

    XGBoost handles missing natively and was fitted on data that is 92.7% missing vitals, so
    the honest representation is also the one the model expects.
    """
    import numpy as np
    row = sg.features_from_encounter({"age": 60, "sex": "F", "arrival": "walk-in",
                                      "vitals": {"hr": 90}})
    assert row["pulse"] == 90.0
    for absent in ("bpsys", "bpdias", "respr", "o2sat", "temp_f", "pain_scale"):
        assert np.isnan(row[absent]), f"{absent} was filled in when nobody measured it"


@prop(TRIAGE)
def test_derived_features_match_the_training_definitions():
    """These are recomputed here rather than imported, so they can drift from training.

    A shock index computed one way at training and another at prediction is a feature the model
    has never seen, wearing the name of one it has.
    """
    row = sg.features_from_encounter(CRITICAL)
    assert row["shock_index"] == pytest.approx(126 / 82, abs=1e-6)
    assert row["pulse_pressure"] == pytest.approx(82 - 48, abs=1e-6)
    assert row["map_pressure"] == pytest.approx(48 + (82 - 48) / 3, abs=1e-6)
    assert row["tachypnea_flag"] == 1.0        # 28 >= 22
    assert row["hypotension_flag"] == 1.0      # 82 <= 100
    assert row["qsofa_partial"] == 2.0
    assert row["hypoxia_flag"] == 1.0          # 88 < 92
    assert row["elderly_flag"] == 1.0          # 81 >= 65
    assert row["pediatric_flag"] == 0.0


@prop(TRIAGE)
def test_arrival_by_ambulance_is_recorded_as_the_flag_the_model_was_fitted_with():
    assert sg.features_from_encounter(CRITICAL)["ambulance_flag"] == 1.0
    assert sg.features_from_encounter(WELL)["ambulance_flag"] == 0.0


@prop(TRIAGE)
def test_missing_observations_are_returned_as_keys_for_the_interface_to_word():
    """Counted as observations, and returned as keys rather than as sentences.

    Two separate points. Counting observations means the screen can say "based on 2 of 7
    observations, not recorded: oxygen saturation" -- a fact a clinician acts on by taking the
    third -- instead of "input completeness 29%", which is a property of a feature vector.

    Returning keys means the French page renders "saturation en oxygène". An English phrase
    built here would arrive on a French screen already in the wrong language, which is a bug
    this project has shipped once.
    """
    entered, missing = sg.entered_count({"vitals": {"hr": 90, "sbp": 120}})
    assert entered == 2
    assert missing == ["dbp", "rr", "spo2", "temp", "pain"]
    assert all(m in sg.OBSERVATIONS for m in missing)
    # No prose, in either language.
    assert not any(" " in m for m in missing)
