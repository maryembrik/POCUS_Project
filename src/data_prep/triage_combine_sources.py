"""
Combine every locally available triage dataset into one harmonized, 3-tier
(low/medium/high) training manifest -- NHAMCS, Iran ED, MC-MED sample, and the
MIMIC-IV-ED demo. One pooled training run on all of it, not a separate model per source
("source" is just one input feature the shared model sees, the same way age or sex is).

Why 3 tiers, not NHAMCS's native 5 levels: the Clinical Reasoning Agent only needs an
urgency tier to decide whether to escalate, not a 5-way distinction, and 3 tiers is the
common ground triage instruments from different countries/systems can actually agree on.

Why most of Iran ED's columns are excluded despite being cleaned in iran_ed_triage_prep.py:
every vital-sign field (and AVPU) there is >90% populated for exactly one TriageGrade and
<5% populated for the rest -- confirmed to be a documentation-workflow artifact of that
specific hospital's extract, not physiological missingness (see iran_ed_triage_prep.py's
docstring). A first pass that included them produced 75% accuracy that was 98% on
Iran-only rows and 62% on NHAMCS-only rows -- the model was learning "was this field
recorded" as a shortcut proxy for the label. Only age/sex/arrival-mode survive from Iran.

MC-MED and MIMIC-IV-ED are both public-demo/sample releases (388 and 222 rows respectively
-- the full, credentialed versions have far more), but both carry something neither NHAMCS
nor Iran ED has: real free-text chief complaints. They're folded into the same pool rather
than excluded, with an honest caveat: at ~600 rows combined against ~164K from NHAMCS+Iran,
they cannot move the aggregate metrics on their own -- they're included because "use
everything available in one pool" is the right default, not because they're expected to
change the headline numbers.

Timestamps in both MC-MED and MIMIC are synthetically shifted per patient for
de-identification (a MIMIC/PhysioNet-standard practice) -- not real calendar dates, so they
can't feed the same year-based train/test split NHAMCS and Iran ED use. Instead every source
gets an explicit `split` column here: NHAMCS/Iran ED use their real year (train < 2022, test
== 2022); MC-MED keeps the curators' own train/validation/test partition (train.parquet ->
train, validation+test.parquet -> test); MIMIC-demo (too small and un-split to hold anything
out meaningfully) goes entirely to train.
"""
import glob
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project")
MANIFEST_DIR = BASE / "manifests"

NHAMCS_MANIFEST = MANIFEST_DIR / "nhamcs_triage_core_combined.csv"
IRAN_MANIFEST = MANIFEST_DIR / "iran_ed_triage_core.csv"
MCMED_DIR = BASE / "Triage_MCMED_sample"
MIMIC_ED_ZIP = BASE / "Triage_MIMIC" / "mimic-iv-ed-demo-2.2.zip"
MIMIC_HOSP_ZIP = BASE / "Triage_MIMIC" / "mimic-iv-clinical-database-demo-2.2.zip"
OUT_MANIFEST = MANIFEST_DIR / "triage_combined_tier_core.csv"

NON_TRIAGE_TIME_LABELS = ["esa_no_nursing_triage", "no_triage_reported"]
ZERO_AS_MISSING = ["PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT"]
IMMEDR_TO_TIER = {"nonurgent": "low", "semi_urgent": "low", "urgent": "medium",
                  "emergent": "high", "immediate": "high"}
CC_RED_FLAG_TERMS = ["chest pain", "sob", "short of breath", "shortness of breath",
                     "unresponsive", "seizure", "stroke", "cva", "arrest", "hemorrhage",
                     "bleeding", "overdose", "anaphyla", "stemi", "intubat"]

COLUMNS = ["age", "sex", "pulse", "respr", "bpsys", "bpdias", "o2sat", "pain_scale",
           "temp_f", "ambulance_flag", "rfv1", "rfv2", "rfv3", "rfv4", "rfv5",
           "altered_mental_status_flag", "cc_text", "cc_red_flag",
           "tier", "year", "source", "split"]


def _cc_red_flag(text: pd.Series) -> pd.Series:
    lower = text.str.lower()
    flagged = pd.Series(False, index=text.index)
    for term in CC_RED_FLAG_TERMS:
        flagged |= lower.str.contains(term, na=False)
    return np.where(text.notna(), flagged.astype(float), np.nan)


def load_nhamcs() -> pd.DataFrame:
    df = pd.read_csv(NHAMCS_MANIFEST)
    df = df[df["IMMEDR_label"].notna() & ~df["IMMEDR_label"].isin(NON_TRIAGE_TIME_LABELS)].copy()
    for col in ZERO_AS_MISSING:
        df[col] = df[col].replace(0, np.nan)

    out = pd.DataFrame({
        "age": df["AGE"], "sex": df["SEX"].map({1: "male", 2: "female"}),
        "pulse": df["PULSE"], "respr": df["RESPR"], "bpsys": df["BPSYS"], "bpdias": df["BPDIAS"],
        "o2sat": df["POPCT"], "pain_scale": df["PAINSCALE"], "temp_f": df["TEMPF"],
        "ambulance_flag": (df["ARREMS"] == 1).astype(float),
        "rfv1": df["RFV1"], "rfv2": df["RFV2"], "rfv3": df["RFV3"], "rfv4": df["RFV4"], "rfv5": df["RFV5"],
        "altered_mental_status_flag": np.nan, "cc_text": np.nan, "cc_red_flag": np.nan,
        "tier": df["IMMEDR_label"].map(IMMEDR_TO_TIER),
        "year": df["year"], "source": "nhamcs",
    })
    out["split"] = np.where(out["year"] < 2022, "train", "test")
    return out


def load_iran() -> pd.DataFrame:
    df = pd.read_csv(IRAN_MANIFEST)
    out = pd.DataFrame({
        "age": df["age"], "sex": df["sex"],
        "pulse": np.nan, "respr": np.nan, "bpsys": np.nan, "bpdias": np.nan,
        "o2sat": np.nan, "pain_scale": np.nan, "temp_f": np.nan,  # nulled: workflow-selective, not real signal (see module docstring)
        "ambulance_flag": df["ambulance_flag"],
        "rfv1": np.nan, "rfv2": np.nan, "rfv3": np.nan, "rfv4": np.nan, "rfv5": np.nan,
        "altered_mental_status_flag": np.nan, "cc_text": np.nan, "cc_red_flag": np.nan,  # nulled for the same reason
        "tier": df["tier"], "year": df["year"], "source": "iran_ed",
    })
    out["split"] = np.where(out["year"] < 2022, "train", "test")
    return out


def load_mcmed() -> pd.DataFrame:
    tier_map = {"1-Resuscitation": "high", "2-Emergent": "high", "3-Urgent": "medium",
                "4-Semi-Urgent": "low", "5-Non-Urgent": "low"}
    cols = ["Age", "Gender", "Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2",
            "Triage_SBP", "Triage_DBP", "Triage_acuity", "CC", "Means_of_arrival"]
    frames = []
    for name, split in [("train", "train"), ("validation", "test"), ("test", "test")]:
        d = pd.read_parquet(MCMED_DIR / f"{name}.parquet", columns=cols)
        for c in ["Triage_Temp", "Triage_HR", "Triage_RR", "Triage_SpO2", "Triage_SBP", "Triage_DBP"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        out = pd.DataFrame({
            "age": d["Age"], "sex": d["Gender"].map({"M": "male", "F": "female"}),
            "pulse": d["Triage_HR"], "respr": d["Triage_RR"], "bpsys": d["Triage_SBP"],
            "bpdias": d["Triage_DBP"], "o2sat": d["Triage_SpO2"], "pain_scale": np.nan,
            "temp_f": d["Triage_Temp"] * 9 / 5 + 32,  # MC-MED temp is Celsius
            "ambulance_flag": (d["Means_of_arrival"] == "EMS").astype(float),
            "rfv1": np.nan, "rfv2": np.nan, "rfv3": np.nan, "rfv4": np.nan, "rfv5": np.nan,
            "altered_mental_status_flag": np.nan,
            "cc_text": d["CC"], "cc_red_flag": _cc_red_flag(d["CC"]),
            "tier": d["Triage_acuity"].map(tier_map),
            "year": np.nan, "source": "mcmed", "split": split,
        })
        frames.append(out)
    return pd.concat(frames, ignore_index=True)


def load_mimic() -> pd.DataFrame:
    with zipfile.ZipFile(MIMIC_ED_ZIP) as z:
        with z.open("mimic-iv-ed-demo-2.2/ed/triage.csv.gz") as f:
            triage = pd.read_csv(f, compression="gzip")
        with z.open("mimic-iv-ed-demo-2.2/ed/edstays.csv.gz") as f:
            edstays = pd.read_csv(f, compression="gzip")
    with zipfile.ZipFile(MIMIC_HOSP_ZIP) as z:
        with z.open("mimic-iv-clinical-database-demo-2.2/hosp/patients.csv.gz") as f:
            patients = pd.read_csv(f, compression="gzip")

    d = triage.merge(edstays[["stay_id", "gender", "arrival_transport"]], on="stay_id", how="left")
    d = d.merge(patients[["subject_id", "anchor_age"]], on="subject_id", how="left")
    d["pain"] = pd.to_numeric(d["pain"], errors="coerce")
    # A handful of `temperature` values are <50 -- physiologically impossible in Fahrenheit,
    # almost certainly a Celsius mis-entry in the source data (real body temp cannot be 36.5F).
    temp_f = np.where(d["temperature"] < 50, d["temperature"] * 9 / 5 + 32, d["temperature"])
    tier_map = {1.0: "high", 2.0: "high", 3.0: "medium", 4.0: "low", 5.0: "low"}

    out = pd.DataFrame({
        "age": d["anchor_age"], "sex": d["gender"].map({"M": "male", "F": "female"}),
        "pulse": d["heartrate"], "respr": d["resprate"], "bpsys": d["sbp"], "bpdias": d["dbp"],
        "o2sat": d["o2sat"], "pain_scale": d["pain"], "temp_f": temp_f,
        "ambulance_flag": (d["arrival_transport"] == "AMBULANCE").astype(float),
        "rfv1": np.nan, "rfv2": np.nan, "rfv3": np.nan, "rfv4": np.nan, "rfv5": np.nan,
        "altered_mental_status_flag": np.nan,
        "cc_text": d["chiefcomplaint"], "cc_red_flag": _cc_red_flag(d["chiefcomplaint"]),
        "tier": d["acuity"].map(tier_map),
        "year": np.nan, "source": "mimic_demo", "split": "train",
    })
    return out


def main():
    nhamcs = load_nhamcs()
    iran = load_iran()
    mcmed = load_mcmed()
    mimic = load_mimic()
    print(f"NHAMCS: {len(nhamcs)} rows")
    print(f"Iran ED: {len(iran)} rows")
    print(f"MC-MED sample: {len(mcmed)} rows")
    print(f"MIMIC-IV-ED demo: {len(mimic)} rows")

    combined = pd.concat([nhamcs[COLUMNS], iran[COLUMNS], mcmed[COLUMNS], mimic[COLUMNS]], ignore_index=True)
    combined = combined[combined["tier"].notna()].copy()

    combined.to_csv(OUT_MANIFEST, index=False)
    print(f"\nCombined 3-tier manifest -> {OUT_MANIFEST} ({len(combined)} rows)")
    print(combined["source"].value_counts())
    print(combined["tier"].value_counts())
    print("\nTrain/test split by source:")
    print(pd.crosstab(combined["source"], combined["split"]))
    print("\nTier breakdown by source:")
    print(pd.crosstab(combined["source"], combined["tier"], normalize="index"))
    print(f"\ncc_text non-null: {combined['cc_text'].notna().sum()} rows ({combined['cc_text'].notna().mean():.2%})")


if __name__ == "__main__":
    main()
