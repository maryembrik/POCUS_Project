"""
Parse the NHAMCS raw fixed-width ED data (2021 + 2022) into a clean CSV, using the
Stata .dct dictionaries pulled from the CDC to get column positions (previously
this raw ASCII file was unparseable without them).

Full dictionaries have 900+ variables (drug codes RX1-RX30, cause-of-injury codes,
etc.) — most are irrelevant to the Triage Agent, so this script parses the full
fixed-width file generically, then also writes out a curated "triage_core" subset
with just the fields relevant to an ED triage model: demographics, arrival mode,
vitals, pain scale, reason-for-visit codes, and — critically — IMMEDR, which is
NHAMCS's real "immediacy with which patient should be seen" field assigned by the
actual triage nurse. That's the ground-truth label an ESI-style triage classifier
needs, not something we have to simulate.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

BASE = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\Triage_NHAMCS")
OUT_DIR = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\manifests")

# IMMEDR codes per official NHAMCS 2022 documentation (doc22-ed-508.pdf, page with [IMMEDR] entry):
# -9=Blank, -8=Unknown, 0="No triage reported but ESA does conduct nursing triage",
# 1=Immediate, 2=Emergent, 3=Urgent, 4=Semi-urgent, 5=Nonurgent, 7=ESA does not conduct nursing triage
IMMEDR_LABELS = {
    -9: "blank",
    -8: "unknown",
    0: "no_triage_reported",
    1: "immediate",
    2: "emergent",
    3: "urgent",
    4: "semi_urgent",
    5: "nonurgent",
    7: "esa_no_nursing_triage",
}

# Sentinel/missing codes per field, from the official documentation. -9 = Blank is universal;
# a few fields have their own extra sentinels (Doppler readings, "Unknown" for pain scale).
SENTINEL_MAP = {
    "TEMPF": [-9],
    "PULSE": [-9, 998],       # 998 = Dopp/Doppler, not a real heart rate value
    "RESPR": [-9],
    "BPSYS": [-9],
    "BPDIAS": [-9, 998],      # 998 = Palp/Dop/Doppler
    "POPCT": [-9],
    "PAINSCALE": [-9, -8],
    "IMMEDR": [-9, -8],       # keep 0 and 7 as real (meaningful) categories, not missing
}

TRIAGE_CORE_COLS = [
    "VMONTH", "VDAYR", "AGE", "AGER", "SEX", "ARREMS", "AMBTRANSFER",
    "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT",
    "PAINSCALE", "IMMEDR",
    "RFV1", "RFV2", "RFV3", "RFV4", "RFV5",
    "DIAG1", "DIAG2", "DIAG3", "DIAG4",
    "WAITTIME", "LOV", "BOARDED",
]


def parse_dct(dct_path: Path) -> list[tuple[str, int, int]]:
    """Returns list of (varname, start_col, end_col), 1-indexed inclusive, from a Stata infix dictionary."""
    spec = []
    line_re = re.compile(r"^\s*(str\d*|byte|int|long|float|double)\s+(\S+)\s+(\d+)(?:-(\d+))?\s*$")
    for line in dct_path.read_text().splitlines():
        m = line_re.match(line)
        if not m:
            continue
        _, varname, start, end = m.groups()
        start = int(start)
        end = int(end) if end else start
        spec.append((varname, start, end))
    return spec


def read_fixed_width(raw_path: Path, spec: list[tuple[str, int, int]]) -> pd.DataFrame:
    colspecs = [(s - 1, e) for _, s, e in spec]  # pandas is 0-indexed, end-exclusive -> matches (s-1, e)
    names = [name for name, _, _ in spec]
    # RFV*3D duplicate the same column ranges as RFV* (3-digit truncated version) -- pandas allows
    # duplicate column names during read then we dedupe by keeping first occurrence.
    df = pd.read_fwf(raw_path, colspecs=colspecs, names=names, dtype=str)
    df = df.loc[:, ~df.columns.duplicated()]
    return df


def process_year(year: int):
    dct_path = BASE / f"eddict{year}.dct"
    zip_path = BASE / f"ed{year}.zip"
    spec = parse_dct(dct_path)
    print(f"{year}: parsed {len(spec)} variables from dictionary")

    with zipfile.ZipFile(zip_path) as z:
        raw_name = [n for n in z.namelist() if not n.endswith("/")][0]
        with z.open(raw_name) as f:
            import io
            raw_bytes = f.read()
    raw_tmp = OUT_DIR / f"_tmp_ed{year}_raw.txt"
    raw_tmp.write_bytes(raw_bytes)

    df = read_fixed_width(raw_tmp, spec)
    raw_tmp.unlink()

    full_out = OUT_DIR / f"nhamcs_{year}_full.csv"
    df.to_csv(full_out, index=False)
    print(f"{year}: {len(df)} visits, {len(df.columns)} columns -> {full_out}")

    core_cols = [c for c in TRIAGE_CORE_COLS if c in df.columns]
    core = df[core_cols].copy()
    for c in ["AGE", "TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE", "IMMEDR", "WAITTIME", "LOV"]:
        if c in core.columns:
            core[c] = pd.to_numeric(core[c], errors="coerce")

    # Apply official sentinel-value legend (see SENTINEL_MAP) before this data is used for anything.
    for c, sentinels in SENTINEL_MAP.items():
        if c in core.columns:
            core[c] = core[c].replace(sentinels, pd.NA)

    # TEMPF has an implied decimal between the 3rd and 4th digit (e.g. 982 -> 98.2F)
    if "TEMPF" in core.columns:
        core["TEMPF"] = core["TEMPF"] / 10

    if "IMMEDR" in core.columns:
        core["IMMEDR_label"] = core["IMMEDR"].map(IMMEDR_LABELS)
    core["year"] = year

    core_out = OUT_DIR / f"nhamcs_{year}_triage_core.csv"
    core.to_csv(core_out, index=False)
    print(f"{year}: triage-core subset ({len(core_cols)} cols) -> {core_out}")
    return df, core


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cores = []
    for year in [2021, 2022]:
        _, core = process_year(year)
        cores.append(core)

    combined = pd.concat(cores, ignore_index=True)
    combined_out = OUT_DIR / "nhamcs_triage_core_combined.csv"
    combined.to_csv(combined_out, index=False)

    print(f"\nCombined 2021+2022 triage-core: {len(combined)} visits -> {combined_out}")
    print("\nIMMEDR (real triage acuity assigned by actual triage nurses -- the label to train on):")
    print(combined["IMMEDR_label"].value_counts(dropna=False))
    print("\nVitals completeness AFTER sentinel-value cleanup (non-null %%):")
    for c in ["TEMPF", "PULSE", "RESPR", "BPSYS", "BPDIAS", "POPCT", "PAINSCALE"]:
        if c in combined.columns:
            pct = combined[c].notna().mean() * 100
            print(f"  {c}: {pct:.1f}%")
    print("\nTEMPF range check (should be ~89.6-105.6F):")
    print(combined["TEMPF"].describe())


if __name__ == "__main__":
    main()
