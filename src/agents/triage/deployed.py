"""Retrain the urgency classifier on the features the clinician-facing intake actually collects.

    python -m src.agents.triage.deployed

Why this exists. The original model is trained on the full manifest and reports macro F1 0.666.
Restricting it at prediction time to what the intake screen can supply -- age, sex and a handful
of vitals -- costs 18.5 accuracy points and collapses it onto two classes: on the held-out split
it stops predicting "low" entirely. A model starved of the features it was fitted with is not
the model whose accuracy was reported, and deploying it that way would put a number in the
report that no deployed prediction could ever achieve.

So the feature space is made the same in all three places:

    training features  ==  interface fields  ==  deployment inputs

What is deliberately excluded, and why:

    rfv1..rfv5   reason-for-visit codes assigned by hospital registration. Recovering them from
                 a free-text complaint is its own classification problem, and a code guessed
                 from prose is not the code the model was fitted on
    source       which dataset a row came from. A property of the corpus, not of the patient;
                 a deployed encounter belongs to no source and the column would be constant
    year         the same, and the temporal split already accounts for time

What is added relative to the earlier measurement: diastolic blood pressure (the same cuff
reading as the systolic already collected, and it restores pulse pressure and MAP), arrival mode
and pain score. All three are routine at triage.

The protocol is copied from tier.main() deliberately -- same temporal split, same class
weighting, same calibration slice carved out of TRAIN, same safety threshold -- so that the two
models differ in their features and in nothing else. Artefacts are written under new names; the
reference model is left untouched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.triage.tier import (  # noqa: E402
    MODEL_DIR, SAFETY_THRESHOLD, TIER_ORDER, expected_calibration_error, load_and_engineer,
    make_sample_weights)

# What the intake screen collects, plus everything derivable from it. Ordered as the form is.
COLLECTED = ["age", "sex", "pulse", "bpsys", "bpdias", "respr", "o2sat", "temp_f",
             "pain_scale", "ambulance_flag"]
DERIVED = ["shock_index", "pulse_pressure", "map_pressure", "qsofa_partial",
           "tachypnea_flag", "hypotension_flag", "fever_flag", "severe_pain_flag",
           "hypoxia_flag", "elderly_flag", "pediatric_flag"]
DEPLOYED_FEATURES = COLLECTED + DERIVED
CATEGORICAL = ["sex"]


def dangerous_miss(y_true, y_pred, classes) -> float:
    """True 'high' graded 'low'. The error that matters, reported separately from accuracy."""
    hi, lo = list(classes).index("high"), list(classes).index("low")
    truly_high = y_true == hi
    return float((y_pred[truly_high] == lo).mean()) if truly_high.any() else 0.0


def apply_safety_threshold(probs, classes):
    """Never output 'low' while P(high) is still meaningful -- bump to 'medium' instead."""
    hi, lo, med = (list(classes).index(c) for c in ("high", "low", "medium"))
    pred = probs.argmax(axis=1)
    pred[(pred == lo) & (probs[:, hi] > SAFETY_THRESHOLD)] = med
    return pred


def main() -> int:
    sys.stdout.reconfigure(encoding="utf8")
    df = load_and_engineer()
    train_df, test_df = df[df["split"] == "train"], df[df["split"] == "test"]

    encoder = LabelEncoder().fit(TIER_ORDER)
    classes = list(encoder.classes_)
    y_train_full = encoder.transform(train_df["tier"])
    y_test = encoder.transform(test_df["tier"])
    high_idx = classes.index("high")

    print(f"train {len(train_df):,}   test {len(test_df):,}")
    print(f"features: {len(DEPLOYED_FEATURES)}  ({len(COLLECTED)} collected, "
          f"{len(DERIVED)} derived)\n")

    fit_idx, calib_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.15, random_state=42, stratify=y_train_full)
    X_fit = train_df.iloc[fit_idx][DEPLOYED_FEATURES]
    y_fit = y_train_full[fit_idx]
    X_calib = train_df.iloc[calib_idx][DEPLOYED_FEATURES]
    y_calib = y_train_full[calib_idx]
    X_test = test_df[DEPLOYED_FEATURES]

    weight_per_class = make_sample_weights(y_fit)
    sample_weight = np.array([weight_per_class[k] * (1.4 if k == high_idx else 1.0)
                              for k in y_fit])

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, tree_method="hist", enable_categorical=True,
        eval_metric="mlogloss", n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42)
    model.fit(X_fit, y_fit, sample_weight=sample_weight)

    ece_before = expected_calibration_error(model.predict_proba(X_test), y_test)
    calibrated = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated.fit(X_calib, y_calib)

    probs = calibrated.predict_proba(X_test)
    ece_after = expected_calibration_error(probs, y_test)

    # Reported both ways. argmax is what the metric conventions assume; the thresholded
    # prediction is what the deployed system actually emits, and they are not the same model.
    for label, pred in (("argmax", probs.argmax(axis=1)),
                        ("with safety threshold", apply_safety_threshold(probs, classes))):
        print(f"===== {label} " + "=" * (62 - len(label)))
        print(classification_report(y_test, pred, target_names=classes, digits=3,
                                    zero_division=0))
        mix = {c: f"{(pred == i).mean() * 100:.1f}%" for i, c in enumerate(classes)}
        print(f"  macro F1            {f1_score(y_test, pred, average='macro'):.4f}")
        print(f"  accuracy            {(pred == y_test).mean():.4f}")
        print(f"  true-high-as-low    {dangerous_miss(y_test, pred, classes) * 100:.2f}%")
        print(f"  prediction mix      {mix}")
        print(f"  confusion (rows=true {classes}):\n{confusion_matrix(y_test, pred)}\n")

    pred = apply_safety_threshold(probs, classes)
    truth = {c: f"{(y_test == i).mean() * 100:.1f}%" for i, c in enumerate(classes)}
    print(f"  actual truth mix    {truth}")
    print(f"  ECE                 {ece_before:.4f} raw -> {ece_after:.4f} calibrated")

    model.save_model(MODEL_DIR / "triage_deployed_xgb.json")
    joblib.dump(calibrated, MODEL_DIR / "triage_deployed_calibrated.joblib")
    metrics = {
        "features": DEPLOYED_FEATURES,
        "collected_from_intake": COLLECTED,
        # The exact category values `sex` was fitted with. Saved because prediction has to
        # rebuild the same dtype, and the alternative is loading the 164,700-row manifest on
        # every request to read two strings off it.
        "sex_categories": [str(c) for c in train_df["sex"].cat.categories],
        # The order the label encoder actually produced, which is ALPHABETICAL and therefore
        # ['high', 'low', 'medium'] -- not TIER_ORDER's clinical ['low', 'medium', 'high'].
        # Recorded because reading the probability vector against the wrong order inverts the
        # prediction without raising anything: a septic patient comes back 'low' at 0.87.
        "classes": [str(c) for c in encoder.classes_],
        "excluded": ["rfv1", "rfv2", "rfv3", "rfv4", "rfv5", "source", "year"],
        "macro_f1": float(f1_score(y_test, pred, average="macro")),
        "accuracy": float((pred == y_test).mean()),
        "dangerous_miss": dangerous_miss(y_test, pred, classes),
        "ece_raw": ece_before,
        "ece_calibrated": ece_after,
        "safety_threshold": SAFETY_THRESHOLD,
        "tier_order": TIER_ORDER,
        "report": classification_report(y_test, pred, target_names=classes, output_dict=True,
                                        zero_division=0),
        "confusion": confusion_matrix(y_test, pred).tolist(),
    }
    with open(MODEL_DIR / "triage_deployed_metrics.json", "w", encoding="utf8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nwrote {MODEL_DIR / 'triage_deployed_calibrated.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
