"""
Build a clean manifest for the Cardiac (CAMUS) dataset.

Each patient is stored as its own zip (patientXXXX.zip) containing 2CH and 4CH
apical view NIfTI volumes (ED/ES frames + half-sequence + ground-truth segmentations)
plus an Info_2CH.cfg / Info_4CH.cfg with clinical metadata (ED/ES frame index,
sex, age, ejection fraction, image quality, frame rate).

Reads the .cfg files directly from inside each zip (no extraction needed for the
manifest step) and produces one row per patient with both views' metadata.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\Cardiac\CAMUS_nifti")
OUT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\manifests\cardiac_manifest.csv")


def parse_cfg(text: str) -> dict:
    out = {}
    for line in text.strip().splitlines():
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip()] = val.strip()
    return out


def build_manifest():
    rows = []
    zips = sorted(ROOT.glob("patient*.zip"))
    for zpath in zips:
        pid = zpath.stem  # e.g. patient0001
        row = {"patient_id": pid, "zip_path": str(zpath.relative_to(ROOT.parent.parent))}
        try:
            with zipfile.ZipFile(zpath) as z:
                names = z.namelist()
                for view in ["2CH", "4CH"]:
                    cfg_name = f"{pid}/Info_{view}.cfg"
                    if cfg_name in names:
                        cfg = parse_cfg(z.read(cfg_name).decode())
                        row[f"{view}_ED_frame"] = cfg.get("ED")
                        row[f"{view}_ES_frame"] = cfg.get("ES")
                        row[f"{view}_nb_frames"] = cfg.get("NbFrame")
                        row[f"{view}_image_quality"] = cfg.get("ImageQuality")
                        row[f"{view}_ejection_fraction"] = cfg.get("EF")
                        row[f"{view}_frame_rate"] = cfg.get("FrameRate")
                    else:
                        row[f"{view}_ED_frame"] = None
                # sex/age are duplicated identically across both cfgs, just take once
                cfg_any = parse_cfg(z.read(f"{pid}/Info_2CH.cfg").decode()) if f"{pid}/Info_2CH.cfg" in names else {}
                row["sex"] = cfg_any.get("Sex")
                row["age"] = cfg_any.get("Age")
                row["has_2CH"] = any(f"{pid}_2CH_ED.nii.gz" in n for n in names)
                row["has_4CH"] = any(f"{pid}_4CH_ED.nii.gz" in n for n in names)
                row["n_files_in_zip"] = len(names)
        except zipfile.BadZipFile:
            row["error"] = "bad_zip"
        rows.append(row)

    manifest = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)

    print(f"Total patients: {len(manifest)}")
    print("Bad/unreadable zips:", manifest.get("error", pd.Series(dtype=object)).notna().sum() if "error" in manifest else 0)
    print("\nSex distribution:")
    print(manifest["sex"].value_counts(dropna=False))
    print("\nAge stats:")
    print(pd.to_numeric(manifest["age"], errors="coerce").describe())
    print("\n2CH image quality:")
    print(manifest["2CH_image_quality"].value_counts(dropna=False))
    print("\n4CH image quality:")
    print(manifest["4CH_image_quality"].value_counts(dropna=False))
    print("\nEjection fraction (4CH) stats:")
    print(pd.to_numeric(manifest["4CH_ejection_fraction"], errors="coerce").describe())
    print(f"\nManifest written to: {OUT}")


if __name__ == "__main__":
    build_manifest()
