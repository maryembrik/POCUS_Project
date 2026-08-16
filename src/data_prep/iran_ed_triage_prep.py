"""
Parse the Iran ED (Isfahan teaching hospital) open-access triage dataset into a clean,
persisted manifest -- mirrors triage_nhamcs_prep.py's role for NHAMCS: raw file in,
documented cleaning applied, one reusable CSV out.

Source: "An open-access dataset of emergency department admissions at a large teaching
hospital in Iran" (Data in Brief / ScienceDirect, 2024), 143,582 ED visits, 2017-2022.

Cleaning applied here, and why:
- Exact-duplicate rows (236 found -- same patient/vitals/grade repeated verbatim) are
  dropped. These look like an export artifact, not two genuine visits: real repeat visits
  would differ in at least admission time/vitals, and these don't.
- TriageGrade's direction (1=most urgent or 1=least urgent) isn't stated in the extracted
  documentation, and the source paper is paywalled. Verified empirically instead of assumed:
  mean age drops monotonically 58->34 from grade 1 to grade 5, and critical-status
  assessments concentrate almost entirely in grades 1-2 -- both consistent with grade 1 =
  most urgent (ESI-style), the direction used here.
- Every vital-sign field (and AVPU) is >90% populated for exactly one TriageGrade and <5%
  populated for the rest -- a documentation-workflow artifact of this specific hospital's
  extract, not physiological missingness (real EDs take vitals on resuscitation patients
  too). Left as-is here (this script only cleans/harmonizes column-level values); the
  decision to null these fields out before training is made downstream in
  triage_combine_sources.py, next to the evidence that motivates it, not hidden here.
- No 0-as-sentinel problem exists in the raw vitals (unlike NHAMCS) -- verified min values
  are all physiologically plausible (BP systolic min 37, pulse min 47, etc.), so no
  replacement was needed.
"""
import glob
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project")
OUT_DIR = BASE / "manifests"

TRIAGE_GRADE_TO_TIER = {1: "high", 2: "high", 3: "medium", 4: "low", 5: "low"}


def load_raw() -> pd.DataFrame:
    csv_path = glob.glob(str(BASE / "Triage_Iran_ED" / "extracted" / "*" / "ED_triage.csv"))[0]
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates()
    print(f"Dropped {before - len(df)} exact-duplicate rows ({before} -> {len(df)})")

    df = df[df["TriageGrade"].notna()].copy()
    df["tier"] = df["TriageGrade"].map(TRIAGE_GRADE_TO_TIER)

    out = pd.DataFrame({
        "age": df["age"],
        "sex": df["gender"].map({"Male": "male", "Female": "female"}),
        "pulse": df["PulseRate"],
        "respr": df["RespiratoryRate"],
        "bpsys": df["BlooddpressurSystol"],
        "bpdias": df["BlooddpressurDiastol"],
        "o2sat": df["O2Saturation"],
        "pain_scale": df["PainGrade"],
        "temp_f": df["Temperature"] * 9 / 5 + 32,  # source unit is Celsius
        "ambulance_flag": (df["kindref"] == 3).astype(float),
        "chief_complaint_icd10": df["ChiefComplaint"],
        "avpu": df["AVPU"],
        "triage_grade": df["TriageGrade"],
        "tier": df["tier"],
        "year": df["admission_year"],
        "month": df["admission_month"],
    })
    return out


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = load_raw()
    print(f"Raw: {len(raw)} rows")
    core = clean(raw)

    out_path = OUT_DIR / "iran_ed_triage_core.csv"
    core.to_csv(out_path, index=False)
    print(f"\nCleaned Iran ED triage-core -> {out_path} ({len(core)} rows)")
    print("\nTier distribution:")
    print(core["tier"].value_counts())
    print("\nVital completeness by triage_grade (documenting the workflow-selective missingness,\n"
          "not fixing it here -- the fix is a modeling-time decision made in triage_combine_sources.py):")
    for c in ["pulse", "o2sat", "avpu"]:
        print(f"  {c}:")
        print(f"    {core.groupby('triage_grade')[c].apply(lambda s: s.notna().mean()).to_dict()}")


if __name__ == "__main__":
    main()
