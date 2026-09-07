"""Turn what a clinician typed into an urgency suggestion.

This is the deployed triage model's only entry point from the application. It exists as its own
module because the mapping between an intake form and a model's feature columns is exactly the
kind of code that goes quietly wrong: the two vocabularies were built years and datasets apart,
and every mismatch fails silently as a missing value rather than as an error.

Three of those mismatches are handled here explicitly:

  temperature   the form collects Celsius; the model was fitted on Fahrenheit. Passing 37.0
                where 98.6 was expected does not raise -- it reads as profound hypothermia, and
                the model's fever flag would be wrong for every patient
  names         the form says hr/sbp/dbp/rr/spo2/temp/pain; the model says
                pulse/bpsys/bpdias/respr/o2sat/temp_f/pain_scale
  arrival       a walk-in/ambulance choice becomes the 0/1 flag the model was fitted with

What this module does NOT do is fill anything in. A vital the clinician did not enter stays
missing, because XGBoost handles missing natively and was trained on data that was mostly
missing -- 92.7% of the training rows carry no vitals at all. Substituting a population mean
would invent a measurement and hide the fact that nobody took it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "models"
ARTIFACT = MODEL_DIR / "triage_deployed_calibrated.joblib"
METRICS = MODEL_DIR / "triage_deployed_metrics.json"

TIER_ORDER = ["low", "medium", "high"]
SAFETY_THRESHOLD = 0.20

# form field -> model column. Temperature is absent deliberately; it needs a conversion, not a
# rename, and putting it here would make the two look interchangeable.
VITAL_TO_COLUMN = {
    "hr": "pulse",
    "sbp": "bpsys",
    "dbp": "bpdias",
    "rr": "respr",
    "spo2": "o2sat",
    "pain": "pain_scale",
}

_MODEL: Any = None
_META: dict[str, Any] | None = None


def available() -> bool:
    return ARTIFACT.exists() and METRICS.exists()


def _load():
    global _MODEL, _META
    if _MODEL is None:
        import joblib
        _MODEL = joblib.load(ARTIFACT)
        _META = json.loads(METRICS.read_text(encoding="utf8"))
    return _MODEL, _META


def features_from_encounter(enc: dict[str, Any]) -> dict[str, Any]:
    """The model's feature row, built from the intake fields and nothing else."""
    import numpy as np

    v = enc.get("vitals") or {}
    row: dict[str, Any] = {c: np.nan for c in _load()[1]["features"]}

    age = enc.get("age")
    row["age"] = float(age) if age is not None else np.nan
    row["sex"] = "female" if str(enc.get("sex", "F")).upper().startswith("F") else "male"
    row["ambulance_flag"] = 1.0 if enc.get("arrival") == "ambulance" else 0.0

    for form_key, column in VITAL_TO_COLUMN.items():
        value = v.get(form_key)
        if value is not None:
            row[column] = float(value)

    if v.get("temp") is not None:
        row["temp_f"] = float(v["temp"]) * 9.0 / 5.0 + 32.0

    # Derived exactly as load_and_engineer() derives them for training. Recomputing rather than
    # importing keeps this free of the manifest, but it means the two definitions must agree --
    # a test asserts they do, on the same inputs.
    pulse, bpsys, bpdias = row["pulse"], row["bpsys"], row["bpdias"]
    respr, o2sat, temp_f = row["respr"], row["o2sat"], row["temp_f"]
    pain = row["pain_scale"]

    def known(x) -> bool:
        return not (x is None or (isinstance(x, float) and np.isnan(x)))

    if known(pulse) and known(bpsys) and bpsys:
        row["shock_index"] = pulse / bpsys
    if known(bpsys) and known(bpdias):
        row["pulse_pressure"] = bpsys - bpdias
        row["map_pressure"] = bpdias + (bpsys - bpdias) / 3
    row["tachypnea_flag"] = float(respr >= 22) if known(respr) else np.nan
    row["hypotension_flag"] = float(bpsys <= 100) if known(bpsys) else np.nan
    row["qsofa_partial"] = ((row["tachypnea_flag"] if known(row["tachypnea_flag"]) else 0.0)
                            + (row["hypotension_flag"] if known(row["hypotension_flag"])
                               else 0.0))
    # These three mirror training, where the comparison is made on a column that may be NaN and
    # NaN >= x is False -- so an unmeasured temperature becomes "not febrile" rather than
    # unknown. Reproduced rather than corrected: the model was fitted against this behaviour,
    # and a prediction-time definition that disagrees with the training-time one is a bug even
    # when the prediction-time one is better.
    row["fever_flag"] = float(known(temp_f) and temp_f >= 100.4)
    row["severe_pain_flag"] = float(known(pain) and pain >= 7)
    row["elderly_flag"] = float(known(row["age"]) and row["age"] >= 65)
    row["pediatric_flag"] = float(known(row["age"]) and row["age"] < 18)
    row["hypoxia_flag"] = float(o2sat < 92) if known(o2sat) else np.nan
    return row


OBSERVATIONS = ("hr", "sbp", "dbp", "rr", "spo2", "temp", "pain")


def entered_count(enc: dict[str, Any]) -> tuple[int, list[str]]:
    """How many intake observations were taken, and the KEYS of those that were not.

    Keys, not sentences: the interface renders them in the reader's language, the way severity
    is already handled. Returning "oxygen saturation" here would put an English phrase on a
    French screen, which is a bug this project has shipped once already.

    Counted as observations rather than as a share of the feature vector. What a clinician can
    act on is that no blood pressure was recorded -- not that input completeness is 56%.
    """
    v = enc.get("vitals") or {}
    missing = [k for k in OBSERVATIONS if v.get(k) is None]
    return len(OBSERVATIONS) - len(missing), missing


def suggest(enc: dict[str, Any]) -> dict[str, Any] | None:
    """An urgency suggestion for this encounter, or None if the model is not installed.

    The returned tier already carries the safety rule the model was evaluated with: never
    'low' while the probability of 'high' is still meaningful. Reporting a raw argmax here and
    applying the rule elsewhere would mean the number shown and the number measured came from
    two different decision procedures.
    """
    if not available():
        return None

    import pandas as pd

    model, meta = _load()
    row = pd.DataFrame([features_from_encounter(enc)])[meta["features"]]
    row["sex"] = pd.Categorical(row["sex"], categories=meta["sex_categories"])

    probs = model.predict_proba(row)[0]
    # Read against the order the encoder produced, taken from the artefact rather than assumed.
    # LabelEncoder sorts its classes, so the columns are ['high', 'low', 'medium'] while the
    # clinically natural order is low/medium/high. Indexing with the natural order inverts the
    # result silently -- the first version of this function graded a hypotensive, hypoxic,
    # tachycardic patient arriving by ambulance as LOW urgency at 0.87 confidence, and nothing
    # raised. The mapping is now data, not a literal.
    classes = meta["classes"]
    named = {str(c): float(probs[k]) for k, c in enumerate(classes)}

    tier = max(named, key=named.get)
    if tier == "low" and named["high"] > SAFETY_THRESHOLD:
        tier = "medium"

    entered, missing = entered_count(enc)
    return {
        "tier": tier,
        "probabilities": {k: round(v, 3) for k, v in named.items()},
        "observations_entered": entered,
        "observations_missing": missing,
        "model": "triage_deployed",
        "macro_f1": meta.get("macro_f1"),
    }
