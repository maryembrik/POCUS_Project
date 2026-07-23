"""
Triage Agent: gradient-boosted-tree urgency classifier on NHAMCS triage_core.

Train/test split is by year (2021 train -> 2022 test), not a random shuffle:
this mirrors real deployment (model built on past data, evaluated on the next
year) and avoids overstating performance the way a shuffled split can.

Feature selection deliberately excludes DIAG1-4, WAITTIME, LOV, and BOARDED —
these are only known *after* triage happens (final diagnosis, time spent,
whether the patient got boarded), so including them would leak future
information into a model meant to predict urgency *at* triage time.
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

DROP_AS_LEAKAGE = ["DIAG1", "DIAG2", "DIAG3", "DIAG4", "WAITTIME", "LOV", "BOARDED"]
NON_TRIAGE_TIME_LABELS = ["esa_no_nursing_triage", "no_triage_reported"]
NUMERIC_FEATURES = ["AGE", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE"]
CATEGORICAL_FEATURES = ["SEX", "ARREMS", "AMBTRANSFER", "RFV1", "RFV2", "RFV3", "RFV4", "RFV5"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
CLASS_ORDER = ["nonurgent", "semi_urgent", "urgent", "emergent", "immediate"]


def load_data() -> pd.DataFrame:
    df = pd.read_csv(MANIFEST)
    df = df[df["IMMEDR_label"].notna() & ~df["IMMEDR_label"].isin(NON_TRIAGE_TIME_LABELS)]
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype("category")
    return df


def make_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    freq = dict(zip(classes, counts))
    weight_per_class = {c: len(y) / (len(classes) * n) for c, n in freq.items()}
    return np.array([weight_per_class[label] for label in y])


def main():
    df = load_data()
    train_df = df[df["year"] == 2021]
    test_df = df[df["year"] == 2022]

    encoder = LabelEncoder().fit(CLASS_ORDER)
    y_train = encoder.transform(train_df["IMMEDR_label"])
    y_test = encoder.transform(test_df["IMMEDR_label"])

    X_train = train_df[FEATURES]
    X_test = test_df[FEATURES]

    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=len(CLASS_ORDER),
        tree_method="hist",
        enable_categorical=True,
        eval_metric="mlogloss",
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train, sample_weight=make_sample_weights(y_train))

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print(classification_report(y_test, y_pred, target_names=encoder.classes_))
    print(f"Macro F1: {macro_f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred), order:", list(encoder.classes_))
    print(confusion_matrix(y_test, y_pred))

    high_acuity_recall = {c: report[c]["recall"] for c in ["immediate", "emergent"]}
    print(f"\nHigh-acuity recall (the safety-critical numbers): {high_acuity_recall}")

    model.save_model(MODEL_DIR / "triage_agent_xgb.json")
    with open(MODEL_DIR / "triage_agent_metrics.json", "w") as f:
        json.dump({"report": report, "macro_f1": macro_f1, "class_order": list(encoder.classes_)}, f, indent=2)
    print(f"\nSaved model to {MODEL_DIR / 'triage_agent_xgb.json'}")


if __name__ == "__main__":
    main()
