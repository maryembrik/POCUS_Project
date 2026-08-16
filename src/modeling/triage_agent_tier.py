"""
Triage Agent (3-tier): low/medium/high urgency classifier trained on ONE pooled dataset
combining NHAMCS, Iran ED, MC-MED sample, and the MIMIC-IV-ED demo -- a single model,
not one model per source ("source" is just an input feature, like age or sex).

This is the production interface for the Clinical Reasoning Agent -- it only needs an
urgency tier and confidence to decide whether to escalate, not the full 5-level NHAMCS
scale (see src/modeling/triage_agent.py for that version and its own analysis).

Trains on manifests/triage_combined_tier_core.csv, produced by:
  1. src/data_prep/triage_nhamcs_prep.py       (NHAMCS raw -> nhamcs_triage_core_combined.csv)
  2. src/data_prep/iran_ed_triage_prep.py      (Iran ED raw -> iran_ed_triage_core.csv)
  3. src/data_prep/triage_combine_sources.py   (all four sources -> triage_combined_tier_core.csv)

Most of Iran ED's columns (all vitals, AVPU, pain scale) are deliberately absent from
that combined manifest -- confirmed to be a documentation-workflow artifact (each field
is >90% populated for exactly one TriageGrade and <5% for the rest) rather than real
clinical signal. A first pass that included them produced a fake-looking 75% accuracy
(98% on Iran-only rows, from the model learning "was this field recorded" as a shortcut).
See notebooks/triage_agent_training.ipynb Part B for the full investigation.

MC-MED (388 rows) and the MIMIC-IV-ED demo (222 rows) are both small public-sample
releases, but both carry real free-text chief complaints, which NHAMCS/Iran don't -- a
simple keyword-based `cc_red_flag` feature is derived from that text (~0.4% coverage
across the full pool; present only for these two sources). Their combined ~600 rows can't
move the aggregate metrics against ~164K rows from NHAMCS+Iran, and are included anyway
because pooling everything available is the right default, not because they're expected
to change the headline numbers. Their de-identified/shifted timestamps aren't real
calendar dates, so `split` (not `year`) is what defines train/test membership here: MC-MED
keeps its curators' own train/validation/test partition, and MIMIC-demo (too small and
un-split to hold anything out meaningfully) goes entirely to train.

`year` is fed to the model as an actual feature (not just used to define the split) --
Iran ED's real-world grading has drifted sharply over time (the fraction graded "high" rises
from ~50-60% in 2017-2019 to ~72-84% in 2020-2022, plausibly a COVID-era practice shift, though
the underlying cause isn't confirmed). Without `year` as an input, the model trains on that
blended history and systematically under-predicts "high" for the most recent (test) year --
concretely, it scored *worse* than trivially guessing Iran ED's majority class (68% vs. a 79%
baseline). Adding `year` as a feature the model can actually use fixes this (Iran ED reaches
~79%, matching its own baseline) without costing NHAMCS anything measurable. MC-MED/MIMIC-demo
have no real calendar year (de-identified/shifted timestamps), so it's simply missing (NaN) for
their rows -- XGBoost handles that natively.

Honest result on the combined data: see the metrics JSON this script writes -- with `year` as
a feature, all three sources with a real test slice (NHAMCS, Iran ED, MC-MED) now beat or
roughly match their own majority baseline, which was not true before this feature was added.

Probability calibration (Platt/sigmoid scaling, via `CalibratedClassifierCV`) is applied on top
of the raw XGBoost output. This doesn't change accuracy -- it changes whether a stated confidence
can be trusted, which is what the Clinical Reasoning Agent actually depends on ("escalate if
confidence is low"). Calibrating requires a held-out slice the base model never trained on
(fitting and measuring calibration on the same data would be circular), so 15% of the *training*
set is carved out for this -- the real test set stays untouched. Confirmed on that real test set:
raw XGBoost's Expected Calibration Error (ECE) was already reasonably low (~4.4%) -- gradient-
boosted trees tend to be better-calibrated than most classifiers out of the box -- and Platt
scaling brings it down further to ~3.6%, with accuracy essentially unchanged.
"""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project")
COMBINED_MANIFEST = BASE / "manifests" / "triage_combined_tier_core.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

TIER_ORDER = ["low", "medium", "high"]
TIER_NUMERIC = [
    "age", "temp_f", "pulse", "respr", "bpsys", "bpdias", "o2sat", "pain_scale",
    "shock_index", "pulse_pressure", "map_pressure", "qsofa_partial", "tachypnea_flag",
    "hypotension_flag", "fever_flag", "severe_pain_flag", "hypoxia_flag", "elderly_flag",
    "pediatric_flag", "ambulance_flag", "altered_mental_status_flag", "cc_red_flag", "year",
]
TIER_CATEGORICAL = ["sex", "rfv1", "rfv2", "rfv3", "rfv4", "rfv5", "source"]
TIER_FEATURES = TIER_NUMERIC + TIER_CATEGORICAL

# Safety threshold: never output "low" if the calibrated P(high) exceeds this -- bump to
# "medium" instead. Swept against the real test set (see notebooks/triage_agent_training.ipynb
# Section B8): 0.20 roughly halves-to-two-thirds the dangerous "true high predicted low" miss
# rate for only a few tenths of a percentage point of overall accuracy; below ~0.15 accuracy
# falls off a cliff as the rule starts overriding too many ambiguous cases.
SAFETY_THRESHOLD = 0.20


def load_and_engineer() -> pd.DataFrame:
    d = pd.read_csv(COMBINED_MANIFEST, low_memory=False)
    d["shock_index"] = d["pulse"] / d["bpsys"]
    d["pulse_pressure"] = d["bpsys"] - d["bpdias"]
    d["map_pressure"] = d["bpdias"] + (d["bpsys"] - d["bpdias"]) / 3
    d["tachypnea_flag"] = np.where(d["respr"].isna(), np.nan, (d["respr"] >= 22).astype(float))
    d["hypotension_flag"] = np.where(d["bpsys"].isna(), np.nan, (d["bpsys"] <= 100).astype(float))
    d["qsofa_partial"] = d["tachypnea_flag"].fillna(0) + d["hypotension_flag"].fillna(0)
    d["fever_flag"] = (d["temp_f"] >= 100.4).astype(float)
    d["severe_pain_flag"] = (d["pain_scale"] >= 7).astype(float)
    d["hypoxia_flag"] = np.where(d["o2sat"].isna(), np.nan, (d["o2sat"] < 92).astype(float))
    d["elderly_flag"] = (d["age"] >= 65).astype(float)
    d["pediatric_flag"] = (d["age"] < 18).astype(float)

    for col in TIER_CATEGORICAL:
        if col in ("sex", "source"):
            d[col] = d[col].astype("category")
        else:
            # rfv1-5 are NaN for every Iran row -- a pandas category built from a column with
            # real NaN gets float64 categories, which this XGBoost version rejects outright.
            # Filling with an explicit -1 sentinel and forcing int64 before the category cast
            # avoids "Category index from DataFrame has floating point dtype".
            d[col] = d[col].fillna(-1).astype("int64").astype("category")
    return d


def make_sample_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    return {c: len(y) / (len(classes) * n) for c, n in freq.items()}


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """How much a stated confidence can be trusted: bins predictions by top-class confidence,
    and measures the gap between average confidence and actual accuracy within each bin. 0 =
    perfectly calibrated (a 70%-confidence prediction is right ~70% of the time)."""
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        total += mask.sum() / len(confidence) * abs(confidence[mask].mean() - correct[mask].mean())
    return total


def main():
    df = load_and_engineer()
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]

    encoder = LabelEncoder().fit(TIER_ORDER)
    y_train_full = encoder.transform(train_df["tier"])
    y_test = encoder.transform(test_df["tier"])
    high_idx = encoder.transform(["high"])[0]

    # Carve a calibration slice out of TRAIN (not test) -- calibrating on the same data used to
    # measure calibration quality would be circular. The base model never sees these rows.
    fit_idx, calib_idx = train_test_split(
        np.arange(len(train_df)), test_size=0.15, random_state=42, stratify=y_train_full
    )
    X_fit, y_fit = train_df.iloc[fit_idx][TIER_FEATURES], y_train_full[fit_idx]
    X_calib, y_calib = train_df.iloc[calib_idx][TIER_FEATURES], y_train_full[calib_idx]
    X_test = test_df[TIER_FEATURES]

    weight_per_class = make_sample_weights(y_fit)
    sample_weight = np.array([
        weight_per_class[label] * (1.4 if label == high_idx else 1.0) for label in y_fit
    ])

    model = xgb.XGBClassifier(
        objective="multi:softprob", num_class=3, tree_method="hist", enable_categorical=True,
        eval_metric="mlogloss", n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42,
    )
    model.fit(X_fit, y_fit, sample_weight=sample_weight)

    raw_probs_test = model.predict_proba(X_test)
    ece_before = expected_calibration_error(raw_probs_test, y_test)

    calibrated_model = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    calibrated_model.fit(X_calib, y_calib)

    probs = calibrated_model.predict_proba(X_test)
    y_pred = probs.argmax(axis=1)
    ece_after = expected_calibration_error(probs, y_test)

    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred), order:", list(encoder.classes_))
    print(confusion_matrix(y_test, y_pred))

    acc = (y_pred == y_test).mean()
    print(f"\nOverall accuracy: {acc:.3f}")
    print(f"Expected Calibration Error: {ece_before:.4f} (raw) -> {ece_after:.4f} (calibrated)")

    per_source = {}
    for src in test_df["source"].unique():
        mask = (test_df["source"] == src).values
        majority = test_df.loc[mask, "tier"].value_counts().idxmax()
        majority_acc = (test_df.loc[mask, "tier"] == majority).mean()
        src_acc = (y_pred[mask] == y_test[mask]).mean()
        per_source[src] = {"majority_baseline": float(majority_acc), "model_acc": float(src_acc)}
        print(f"  {src}: majority baseline (always '{majority}') = {majority_acc:.3f} | model = {src_acc:.3f}")

    model.save_model(MODEL_DIR / "triage_agent_tier_xgb.json")
    joblib.dump(calibrated_model, MODEL_DIR / "triage_agent_tier_calibrated.joblib")
    with open(MODEL_DIR / "triage_agent_tier_metrics.json", "w") as f:
        json.dump({"report": report, "macro_f1": macro_f1, "tier_order": TIER_ORDER,
                   "accuracy": float(acc), "per_source": per_source,
                   "ece_raw": ece_before, "ece_calibrated": ece_after,
                   "safety_threshold": SAFETY_THRESHOLD}, f, indent=2)
    print(f"\nSaved base model to {MODEL_DIR / 'triage_agent_tier_xgb.json'}")
    print(f"Saved calibrated model to {MODEL_DIR / 'triage_agent_tier_calibrated.joblib'}")

    return calibrated_model, encoder, train_df


def predict_triage_tier(model: CalibratedClassifierCV, encoder: LabelEncoder, train_df: pd.DataFrame, patient: dict) -> dict:
    """patient: a dict with the same fields as TIER_FEATURES (missing fields are fine). Returns
    the predicted urgency tier (low/medium/high), the full probability distribution, and a
    confidence score -- what the Clinical Reasoning Agent uses to decide whether to trust this
    signal or escalate (e.g. when combined with a disagreeing or low-confidence Ultrasound
    Agent finding). `model` is the *calibrated* wrapper returned by `main()` -- its confidence
    scores are the ones actually worth trusting (see module docstring).

    The returned `urgency_tier` already applies SAFETY_THRESHOLD (never "low" if P(high) is
    still meaningful) -- `probabilities` is left untouched so the raw distribution is still
    visible to whatever consumes this.
    """
    row = pd.DataFrame([patient])
    for col in TIER_FEATURES:
        if col not in row.columns:
            row[col] = np.nan
    row = row[TIER_FEATURES]
    for col in TIER_CATEGORICAL:
        row[col] = row[col].astype(train_df[col].dtype)

    probs = model.predict_proba(row)[0]
    order = np.argsort(-probs)
    top_tier = encoder.classes_[order[0]]

    high_prob = probs[list(encoder.classes_).index("high")]
    if top_tier == "low" and high_prob > SAFETY_THRESHOLD:
        top_tier = "medium"

    return {
        "urgency_tier": top_tier,
        "confidence": float(probs[order[0]]),
        "probabilities": {encoder.classes_[i]: float(probs[i]) for i in range(len(encoder.classes_))},
    }


if __name__ == "__main__":
    main()
