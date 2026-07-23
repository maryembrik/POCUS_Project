"""
Build a clean manifest for the Abdominal (UIdataGB gallbladder) dataset.

Each of the 9 disease classes is its own zip (e.g. "1Gallstones.zip"), containing
a flat folder of images named "<patient_letter+number> (<image_number>).jpg"
(e.g. "a1 (2).jpg" = patient a1, image 2). Reads filenames directly from each zip
without extracting.
"""
import re
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\Abdominal_Gallbladder")
OUT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\manifests\gallbladder_manifest.csv")

FNAME_RE = re.compile(r"^([a-zA-Z]+\d+)\s*\((\d+)\)\.(jpg|jpeg|png)$", re.IGNORECASE)


def build_manifest():
    rows = []
    zips = sorted(ROOT.glob("*.zip"))
    for zpath in zips:
        class_name = zpath.stem  # e.g. "1Gallstones"
        class_label = re.sub(r"^\d+", "", class_name).strip()
        with zipfile.ZipFile(zpath) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                fname = Path(info.filename).name
                m = FNAME_RE.match(fname)
                patient_id = m.group(1) if m else None
                image_num = m.group(2) if m else None
                rows.append({
                    "zip_path": str(zpath.relative_to(ROOT.parent)),
                    "internal_path": info.filename,
                    "filename": fname,
                    "class_raw": class_name,
                    "class": class_label,
                    "patient_id": patient_id,
                    "image_num": image_num,
                    "file_size_bytes": info.file_size,
                    "source": "UIdataGB (Mendeley)",
                })

    manifest = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)

    print(f"Total images: {len(manifest)}")
    print("Unparsed filenames (no patient_id):", manifest["patient_id"].isna().sum())
    print("\nBy class:")
    print(manifest["class"].value_counts())
    print("\nUnique patients per class:")
    print(manifest.groupby("class")["patient_id"].nunique())
    print(f"\nManifest written to: {OUT}")


if __name__ == "__main__":
    build_manifest()
