"""
Build a clean manifest for the Pulmonary (POCUS, Born et al.) dataset.

Walks the extracted covid19_pocus_ultrasound-master data folder and produces one row
per file: filepath, media_type, probe_type, class, source, plus clinical findings
comments where the bundled dataset_metadata.csv has a matching entry.

Class prefixes (from filename, case-insensitive):
  Cov  -> covid19
  Pneu -> bacterial_pneumonia
  Reg  -> healthy
  Vir  -> viral_pneumonia
"""
import re
from pathlib import Path

import pandas as pd

ROOT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\Pulmonary\POCUS_extracted\covid19_pocus_ultrasound-master\data")
OUT = Path(r"C:\Users\HUAWEI\Documents\POCUS-Project\manifests\pulmonary_manifest.csv")

CLASS_MAP = {
    "cov": "covid19",
    "pneu": "bacterial_pneumonia",
    "reg": "healthy",
    "vir": "viral_pneumonia",
}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".gif", ".mpeg", ".mpg"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}

PREFIX_RE = re.compile(r"^([A-Za-z]+)[-_]")


def classify(filename: str) -> str | None:
    m = PREFIX_RE.match(filename)
    if not m:
        return None
    return CLASS_MAP.get(m.group(1).lower())


def load_metadata_lookup() -> dict:
    """filename (lowercased, basename only) -> dict of clinical/technical fields."""
    meta_path = ROOT / "dataset_metadata.csv"
    df = pd.read_csv(meta_path)
    lookup = {}
    for _, row in df.iterrows():
        fname = str(row.get("Filename", "")).strip()
        if not fname or fname.lower() == "nan":
            continue
        key = fname.lower()
        excluded = "not_to_use" in str(row.get("Current location", "")).lower() or \
                   "not used" in str(row.get("Current location", "")).lower()
        md_comments = " | ".join(
            str(row[c]) for c in ["Comments from web site", "Comments first medical doctor (MD1)", "MD2", "MD3"]
            if pd.notna(row.get(c)) and str(row.get(c)).strip()
        )
        lookup[key] = {
            "excluded_by_curators": excluded,
            "framerate": row.get("Framerate"),
            "resolution": row.get("Resolution"),
            "length_frames": row.get("Length (frames)"),
            "clinical_findings": md_comments,
        }
    return lookup


KEYWORD_FLAGS = {
    "finding_b_lines": [r"b[\s-]?lines?", r"blines?"],
    "finding_consolidation": [r"consolidat"],
    "finding_pleural_effusion": [r"effu[cs]ion"],
    "finding_pleural_thickening": [r"thicken", r"irregular"],
    "finding_pneumothorax": [r"pneumothorax", r"lung point", r"absent lung sliding", r"barcode"],
    "finding_normal": [r"\bnormal\b", r"\bhealthy\b"],
    "flag_do_not_use": [r"do not use", r"don't use", r"not to use", r"not use"],
    "flag_off_target_organ": [r"\bliver\b", r"\bkidney\b", r"\bspleen\b"],
}


def extract_findings_flags(text: str) -> dict:
    text = (text or "").lower()
    return {col: bool(re.search("|".join(patterns), text)) for col, patterns in KEYWORD_FLAGS.items()}


def build_manifest():
    meta_lookup = load_metadata_lookup()
    rows = []
    for media_dir, media_type, exts in [
        ("pocus_videos", "video", VIDEO_EXTS),
        ("pocus_images", "image", IMAGE_EXTS),
    ]:
        for probe in ["convex", "linear"]:
            folder = ROOT / media_dir / probe
            if not folder.exists():
                continue
            for f in sorted(folder.iterdir()):
                if not f.is_file() or f.suffix.lower() not in exts:
                    continue
                cls = classify(f.name)
                meta = meta_lookup.get(f.name.lower(), {})
                findings_text = meta.get("clinical_findings", "")
                row = {
                    "filepath": str(f.relative_to(ROOT.parent.parent)),
                    "filename": f.name,
                    "media_type": media_type,
                    "probe_type": probe,
                    "class": cls,
                    "excluded_by_curators": meta.get("excluded_by_curators", False),
                    "framerate": meta.get("framerate"),
                    "resolution": meta.get("resolution"),
                    "length_frames": meta.get("length_frames"),
                    "clinical_findings": findings_text,
                    "source": "POCUS (Born et al.)",
                }
                row.update(extract_findings_flags(findings_text))
                rows.append(row)

    manifest = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(OUT, index=False)

    print(f"Total files: {len(manifest)}")
    print(f"Unclassified (no matching prefix): {manifest['class'].isna().sum()}")
    print("\nBy class:")
    print(manifest["class"].value_counts(dropna=False))
    print("\nBy media_type x probe_type:")
    print(manifest.groupby(["media_type", "probe_type"]).size())
    print("\nFlagged as excluded by original curators:", manifest["excluded_by_curators"].sum())
    print("\nRows with clinical findings text:", (manifest["clinical_findings"] != "").sum())
    print("\nDerived finding flags (from MD free text):")
    for col in KEYWORD_FLAGS:
        print(f"  {col}: {manifest[col].sum()}")
    print(f"\nManifest written to: {OUT}")


if __name__ == "__main__":
    build_manifest()
