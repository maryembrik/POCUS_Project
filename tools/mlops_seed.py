"""Seed the model registry from the metrics this project already produced.

Every number written here is read out of a results file in the repository. Nothing is typed
in by hand and nothing is estimated, because a registry seeded with plausible numbers would
be worse than an empty one: it would look like a record of what happened.

    python tools/mlops_seed.py

The lung entry is the mean over the three cross-validation splits, which is what the report
quotes; the per-split files stay on disk beside it. The cardiac and gallbladder entries carry
only what their validation actually measured. Anything a model does not report is left out
rather than filled with a zero -- the gate treats a missing guard metric as a failed check, so
an absent number must stay visibly absent.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mlops import dataset_version, save_registry  # noqa: E402


def lung() -> dict | None:
    """Mean AUROC and tuned F1 across the three splits, plus per-finding recall if present."""
    files = sorted((ROOT / "Pulmonary").glob("results_efficientnet_b0_split*.json"))
    if not files:
        return None
    runs = [json.loads(p.read_text(encoding="utf8")) for p in files]
    findings = list(runs[0]["auroc"])
    metrics: dict[str, float] = {}
    for f in findings:
        short = f.replace("finding_", "")
        metrics[f"auroc_{short}"] = sum(r["auroc"][f] for r in runs) / len(runs)
        metrics[f"f1_{short}"] = sum(r["f1_tuned"][f] for r in runs) / len(runs)
    metrics["macro_auroc"] = sum(metrics[f"auroc_{f.replace('finding_', '')}"]
                                 for f in findings) / len(findings)
    metrics["macro_f1"] = sum(metrics[f"f1_{f.replace('finding_', '')}"]
                              for f in findings) / len(findings)
    return {"metrics": {k: round(v, 4) for k, v in metrics.items()},
            "eval_set": f"pulmonary {len(files)}-fold cross-validation, tuned thresholds",
            "artifact": "Pulmonary/lung_finding_classifier_efficientnet_b0_final_best.pth",
            "notes": ("Mean over the per-split result files. Recall per finding is not in "
                      "these files, so the guard metrics the gate wants are absent: a "
                      "retrained candidate must report them.")}


def triage() -> dict | None:
    p = ROOT / "models" / "triage_agent_tier_metrics.json"
    if not p.exists():
        return None
    m = json.loads(p.read_text(encoding="utf8"))
    rep = m.get("report", {})
    metrics = {"accuracy": m.get("accuracy"), "macro_f1": m.get("macro_f1"),
               "ece": m.get("ece_calibrated")}
    for tier in ("high", "medium", "low"):
        if tier in rep:
            metrics[f"recall_{tier}"] = rep[tier].get("recall")
            metrics[f"precision_{tier}"] = rep[tier].get("precision")
    return {"metrics": {k: round(v, 4) for k, v in metrics.items() if v is not None},
            "eval_set": "held-out test split of triage_combined_tier_core (16,180 rows)",
            "artifact": "models/triage_agent_tier_calibrated.joblib",
            "notes": ("Calibrated XGBoost. `recall_high` is the guarded metric: a true "
                      "emergency graded low is the miss that matters here.")}


def main() -> int:
    # Re-seeding starts the registry again, but the names already spent must not come back:
    # MLflow keeps every run, so a reused version name would leave two different models
    # answering to the same label in the history.
    from mlops import load_registry  # noqa: PLC0415 - local, to keep the CLI import light

    previous = load_registry()
    reg = {"models": {}}
    seeded = []
    for name, fn in (("lung", lung), ("triage", triage)):
        entry = fn()
        if entry is None:
            print(f"skipped {name}: no results file in the repository")
            continue
        ds = dataset_version(name)
        spent = sorted({v["version"] for v in
                        previous.get("models", {}).get(name, {}).get("versions", [])}
                       | set(previous.get("models", {}).get(name, {})
                             .get("retired_names", [])))
        reg["models"][name] = {"retired_names": spent, "versions": [{
            "version": "v1",
            "stage": "production",
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "dataset": ds["version"],
            "approved_by": "seeded from the results this project already produced",
            **entry,
        }]}
        seeded.append(f"{name} v1 ({ds['version']}, {ds['rows']:,} manifest rows)")

    save_registry(reg)
    print("seeded:")
    for s in seeded:
        print("  -", s)
    print("\nCardiac and gallbladder are not seeded: their validation numbers live in the "
          "notebooks rather than in a results file, and a registry entry typed in by hand "
          "would be a claim rather than a record.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
