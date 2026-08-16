"""
Build a clean, wide per-case biological data table from VitalDB.

vitaldb_lab_results.csv is long-format (caseid, dt, name, result) with ~145
readings per case on average across 34 lab test types (confirms the registry
note: lactate present, but NO troponin / NO D-dimer anywhere in VitalDB).
Pivots to one row per case (median value per lab, since most fusion-prototyping
work needs a single representative value per test, not the full time series),
then joins with the per-case clinical/demographic table.

Reminder (per the datasets registry): VitalDB is 100% surgical/anesthesia context,
zero ED/chief-complaint fields -- use only as generic vitals+labs fusion
code-practice data for the Clinical Reasoning Agent, not an ED case stand-in.
"""
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\Biological_VitalDB")
OUT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\manifests\vitaldb_biological_manifest.csv")

CLINICAL_COLS = [
    "caseid", "age", "sex", "height", "weight", "bmi", "asa", "emop",
    "department", "optype", "dx", "icu_days", "death_inhosp",
]


def build_manifest():
    clin = pd.read_csv(BASE / "vitaldb_clinical_information.csv", usecols=lambda c: c in CLINICAL_COLS)
    lab = pd.read_csv(BASE / "vitaldb_lab_results.csv")

    print(f"Clinical table: {len(clin)} cases")
    print(f"Lab results (long format): {len(lab)} readings across {lab['name'].nunique()} test types")

    pivot = lab.pivot_table(index="caseid", columns="name", values="result", aggfunc="median")
    pivot.columns = [f"lab_{c}_median" for c in pivot.columns]
    pivot = pivot.reset_index()

    manifest = clin.merge(pivot, on="caseid", how="left")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)

    print(f"\nMerged manifest: {len(manifest)} cases x {len(manifest.columns)} columns -> {OUT}")
    print("\nLab coverage (% of cases with at least one reading):")
    for c in [c for c in manifest.columns if c.startswith("lab_")]:
        pct = manifest[c].notna().mean() * 100
        print(f"  {c}: {pct:.1f}%")
    print("\nConfirmed absent from VitalDB (checked against target biological fields): troponin, D-dimer")
    print("In-hospital mortality rate:", manifest["death_inhosp"].mean())


if __name__ == "__main__":
    build_manifest()
