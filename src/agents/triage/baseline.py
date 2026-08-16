"""
Triage Agent: gradient-boosted-tree urgency classifier on NHAMCS triage_core.

Train/test split is by year (2021 train -> 2022 test), not a random shuffle:
this mirrors real deployment (model built on past data, evaluated on the next
year) and avoids overstating performance the way a shuffled split can.

Feature selection deliberately excludes DIAG1-4, WAITTIME, LOV, and BOARDED —
these are only known *after* triage happens (final diagnosis, time spent,
whether the patient got boarded), so including them would leak future
information into a model meant to predict urgency *at* triage time.

See notebooks/triage_agent_training.ipynb for the full development history,
including what was tried and reverted (RFV module grouping, a 2x high-acuity
weight boost) and the honest limitations of the resulting model.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

BASE = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project")
MANIFEST = BASE / "manifests" / "nhamcs_triage_core_combined.csv"
MODEL_DIR = BASE / "models"
MODEL_DIR.mkdir(exist_ok=True)

NON_TRIAGE_TIME_LABELS = ["esa_no_nursing_triage", "no_triage_reported"]
CLASS_ORDER = ["nonurgent", "semi_urgent", "urgent", "emergent", "immediate"]
ZERO_AS_MISSING = ["PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"]  # 0 isn't a physiologically real reading

NUMERIC_FEATURES = [
    "AGE", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE",
    "shock_index", "pulse_pressure", "map_pressure", "qsofa_partial",
    "tachypnea_flag", "hypotension_flag", "fever_flag", "severe_pain_flag",
    "hypoxia_flag", "elderly_flag", "pediatric_flag",
]
CATEGORICAL_FEATURES = ["SEX", "ARREMS", "AMBTRANSFER", "RFV1", "RFV2", "RFV3", "RFV4", "RFV5"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def load_and_engineer() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    df = df[df["IMMEDR_label"].notna() & ~df["IMMEDR_label"].isin(NON_TRIAGE_TIME_LABELS)].copy()

    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    # Composite vital-sign features -- raw vitals barely differ across acuity levels on their
    # own; combining them the way clinicians reason about shock physiology carries more signal.
    df["shock_index"] = df["PULSE"] / df["BPSYS"]
    df["pulse_pressure"] = df["BPSYS"] - df["BPDIAS"]
    df["map_pressure"] = df["BPDIAS"] + (df["BPSYS"] - df["BPDIAS"]) / 3

    # Partial qSOFA (2 of 3 -- altered mental status isn't in this data, so this is not the full
    # clinical score).
    df["tachypnea_flag"] = np.where(df["RESPR"].isna(), np.nan, (df["RESPR"] >= 22).astype(float))
    df["hypotension_flag"] = np.where(df["BPSYS"].isna(), np.nan, (df["BPSYS"] <= 100).astype(float))
    df["qsofa_partial"] = df["tachypnea_flag"].fillna(0) + df["hypotension_flag"].fillna(0)

    df["fever_flag"] = (df["TEMPF"] >= 100.4).astype(float)
    df["severe_pain_flag"] = (df["PAINSCALE"] >= 7).astype(float)
    df["hypoxia_flag"] = np.where(df["POPCT"].isna(), np.nan, (df["POPCT"] < 92).astype(float))
    df["elderly_flag"] = (df["AGE"] >= 65).astype(float)
    df["pediatric_flag"] = (df["AGE"] < 18).astype(float)

    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")

    return df


def make_sample_weights(y: np.ndarray) -> dict:
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    return {c: len(y) / (len(classes) * n) for c, n in freq.items()}


def main():
    df = load_and_engineer()
    train_df = df[df["year"] == 2021]
    test_df = df[df["year"] == 2022]

    encoder = LabelEncoder().fit(CLASS_ORDER)
    y_train = encoder.transform(train_df["IMMEDR_label"])
    y_test = encoder.transform(test_df["IMMEDR_label"])
    high_acuity_idx = {encoder.transform(["immediate"])[0], encoder.transform(["emergent"])[0]}

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    weight_per_class = make_sample_weights(y_train)
    # Asymmetric boost: under-predicting "immediate"/"emergent" (a false negative on a critical
    # patient) is clinically worse than the reverse. A 2x boost was tried and reverted -- it
    # tanked overall accuracy (56% -> 39%) for little extra safety gain. 1.4x is the moderate
    # point that helps both high-acuity recalls without wrecking the rest.
    sample_weight = np.array([
        weight_per_class[label] * (1.4 if label in high_acuity_idx else 1.0)
        for label in y_train
    ])

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_ORDER),
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mlogloss",
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=sample_weight)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred), order:", list(encoder.classes_))
    print(confusion_matrix(y_test, y_pred))

    majority_class = test_df["IMMEDR_label"].value_counts().idxmax()
    majority_acc = (test_df["IMMEDR_label"] == majority_class).mean()
    acc = (y_pred == y_test).mean()
    print(f"\nMajority-class baseline (always predict '{majority_class}'): {majority_acc:.3f} accuracy")
    print(f"Model accuracy: {acc:.3f}")

    high_acuity_recall = {c: report[c]["recall"] for c in ["immediate", "emergent"]}
    print(f"\nHigh-acuity recall (the safety-critical numbers): {high_acuity_recall}")

    model.save_model(MODEL_DIR / "triage_agent_xgb.json")
    with open(MODEL_DIR / "triage_agent_metrics.json", "w") as f:
        json.dump({"report": report, "macro_f1": macro_f1, "class_order": list(encoder.classes_),
                   "majority_baseline_acc": float(majority_acc), "model_acc": float(acc)}, f, indent=2)
    print(f"\nSaved model to {MODEL_DIR / 'triage_agent_xgb.json'}")

    return model, encoder, X_train


def predict_triage(model: xgb.XGBClassifier, encoder: LabelEncoder, X_train: pd.DataFrame, patient: dict) -> dict:
    """patient: a dict with the same fields as FEATURES (missing fields are fine, XGBoost
    handles them as missing). Returns the predicted urgency level, the full probability
    distribution across all 5 levels, and a confidence score (the top class's probability) --
    what the Clinical Reasoning Agent needs to decide whether to trust this signal or escalate.
    """
    row = pd.DataFrame([patient])
    for col in FEATURES:
        if col not in row.columns:
            row[col] = np.nan
    row = row[FEATURES]
    for col in CATEGORICAL_FEATURES:
        # Reuse the exact categorical dtype the model was trained on -- a freshly inferred
        # category dtype on a single-row DataFrame produces codes XGBoost's predictor rejects.
        row[col] = row[col].astype(X_train[col].dtype)

    probs = model.predict_proba(row)[0]
    order = np.argsort(-probs)
    return {
        "urgency_level": encoder.classes_[order[0]],
        "confidence": float(probs[order[0]]),
        "probabilities": {encoder.classes_[i]: float(probs[i]) for i in range(len(encoder.classes_))},
    }


if __name__ == "__main__":
    main()
