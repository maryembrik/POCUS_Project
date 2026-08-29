"""A small MLOps lifecycle for POCUS-Emergency: version, evaluate, gate, promote.

    python tools/mlops.py status
    python tools/mlops.py dataset lung
    python tools/mlops.py register lung --metrics results.json --dataset lung@a1b2c3
    python tools/mlops.py gate lung
    python tools/mlops.py promote lung v2 --approve "who approved, and why"

WHY THIS EXISTS, in one sentence a supervisor can be given: the datasets behind the initial
models do not cover the whole clinical scope, so the system is built to accept validated data
later -- versioned, retrained, evaluated against fixed criteria, and promoted only by a human.

WHAT IT DELIBERATELY IS NOT. There is no automatic retraining on clinical use, and no path
from an uploaded study to a production model without a person in it. A doctor contributes
DATA; a model reaches patients only when someone approves it by name. Everything here is
designed so that the dangerous version of this idea is not reachable by accident: `promote`
refuses without an explicit approver, and the gate refuses on safety regression even when the
headline metric improves.

THE GATE IS THE POINT. "Is the new model better?" is the wrong question for an emergency
system, because a model can gain accuracy by getting cautious cases right while missing more
sick ones. The rules below therefore check what a single accuracy number hides: recall on the
classes where a miss is dangerous, per-class floors, and calibration -- because this pipeline
escalates on low confidence, so a model whose stated confidence stops meaning anything breaks
the escalation logic even if it classifies better.

Storage is a JSON file and content hashes rather than DVC. The datasets here are tens of
gigabytes, already untracked, and not redistributable, so DVC without a remote would add a
tool without adding reproducibility; hashing the MANIFEST -- the list of rows actually used --
records exactly which data produced which model, which is the property that matters. If the
data ever becomes shareable, `dvc add` on the manifests slots in underneath this unchanged.

MLflow is used when it is installed and skipped when it is not, so the pipeline runs on a
bare checkout and gains a UI (`mlflow ui`) when someone wants one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models" / "registry.json"

# Which manifests define each model's training data. Hashing these gives a dataset version
# that changes exactly when the rows change.
DATASETS: dict[str, list[str]] = {
    "lung": ["manifests/pulmonary_manifest.csv"],
    "cardiac": ["manifests/cardiac_manifest.csv"],
    "gallbladder": ["manifests/gallbladder_manifest.csv"],
    "triage": ["manifests/triage_combined_tier_core.csv"],
}

# ─────────────────────────────────────────────────────────────────── promotion rules
#
# `key` is the headline metric. `guard` names the metrics where a REGRESSION is a safety
# regression rather than a trade-off, because in an emergency setting the expensive error is
# the missed sick patient, not the over-called well one.
RULES: dict[str, dict[str, Any]] = {
    "lung": {
        "key": "macro_f1",
        "tolerance": 0.0,          # the headline may not regress at all
        "guard": ["recall_b_lines", "recall_consolidation", "recall_pleural_effusion"],
        "guard_tolerance": 0.02,   # a guarded metric may drop by at most 2 points
        "floor": {"macro_f1": 0.45},
    },
    "cardiac": {
        "key": "mae_ef",
        "lower_is_better": True,
        "tolerance": 0.0,
        "guard": ["recall_severe_dysfunction"],
        "guard_tolerance": 0.02,
        "floor": {},
    },
    "gallbladder": {
        "key": "accuracy",
        "tolerance": 0.0,
        "guard": ["recall_carcinoma", "recall_cholecystitis"],
        "guard_tolerance": 0.02,
        "floor": {},
    },
    "triage": {
        "key": "macro_f1",
        "tolerance": 0.0,
        "guard": ["recall_high"],          # a true emergency graded low is the dangerous miss
        "guard_tolerance": 0.01,
        "floor": {"accuracy": 0.60},
        "calibration": ("ece", 0.10),      # stated confidence must stay usable
    },
}


# ─────────────────────────────────────────────────────────────────────── dataset version
def dataset_version(model: str) -> dict[str, Any]:
    """A content hash of the manifests that define this model's training rows.

    Missing manifests are reported rather than skipped: a dataset version computed from half
    the files would be a different dataset wearing the same name.
    """
    files, digest, missing = [], hashlib.sha256(), []
    for rel in DATASETS.get(model, []):
        p = ROOT / rel
        if not p.exists():
            missing.append(rel)
            continue
        raw = p.read_bytes()
        h = hashlib.sha256(raw).hexdigest()
        digest.update(h.encode())
        files.append({"path": rel, "sha256": h[:16], "bytes": len(raw),
                      "rows": max(raw.count(b"\n") - 1, 0)})
    return {"version": f"{model}@{digest.hexdigest()[:12]}" if files else None,
            "files": files, "missing": missing,
            "rows": sum(f["rows"] for f in files)}


# ─────────────────────────────────────────────────────────────────────────── registry
def load_registry() -> dict[str, Any]:
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text(encoding="utf8"))
    return {"models": {}}


def save_registry(reg: dict[str, Any]) -> None:
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf8")


def production(reg: dict[str, Any], model: str) -> dict[str, Any] | None:
    for v in reg["models"].get(model, {}).get("versions", []):
        if v.get("stage") == "production":
            return v
    return None


def candidate(reg: dict[str, Any], model: str) -> dict[str, Any] | None:
    cands = [v for v in reg["models"].get(model, {}).get("versions", [])
             if v.get("stage") == "candidate"]
    return cands[-1] if cands else None


# ────────────────────────────────────────────────────────────────────────────── gate
def gate(model: str, cand: dict[str, Any], prod: dict[str, Any] | None) -> dict[str, Any]:
    """Decide whether a candidate may be OFFERED for approval. Never promotes anything.

    Returns every check with its own verdict rather than one boolean, because "rejected" is
    not useful to the person who has to fix it, and because a reviewer should be able to see
    which check a passing model passed narrowly.
    """
    rules = RULES.get(model, {"key": "accuracy", "tolerance": 0.0, "guard": [],
                              "guard_tolerance": 0.02, "floor": {}})
    cm, pm = cand.get("metrics", {}), (prod or {}).get("metrics", {})
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # 1. Comparable at all. A candidate scored on a different benchmark is not a comparison.
    if prod is not None:
        same = cand.get("eval_set") == prod.get("eval_set")
        add("same evaluation set", same,
            f"candidate {cand.get('eval_set')!r} vs production {prod.get('eval_set')!r}"
            + ("" if same else " — these numbers cannot be compared"))

    # 2. The dataset that produced it is recorded.
    add("dataset version recorded", bool(cand.get("dataset")),
        cand.get("dataset") or "no dataset version — cannot reproduce this model")

    # 3. Headline metric.
    key, lower = rules["key"], rules.get("lower_is_better", False)
    if key in cm:
        if prod is None or key not in pm:
            add(f"headline {key}", True, f"{cm[key]:.4f} (no production model to beat)")
        else:
            delta = cm[key] - pm[key]
            ok = (-delta if lower else delta) >= -rules["tolerance"]
            add(f"headline {key}", ok,
                f"{pm[key]:.4f} -> {cm[key]:.4f} ({delta:+.4f})")
    else:
        add(f"headline {key}", False, f"candidate does not report {key}")

    # 4. The guarded metrics. This is the check a plain accuracy comparison does not make:
    #    a model may improve overall and still miss more of the patients who are actually ill.
    for g in rules.get("guard", []):
        if g not in cm:
            add(f"guard {g}", False, "not reported — cannot confirm it did not regress")
            continue
        if prod is None or g not in pm:
            add(f"guard {g}", True, f"{cm[g]:.4f} (no production baseline)")
            continue
        delta = cm[g] - pm[g]
        ok = delta >= -rules["guard_tolerance"]
        add(f"guard {g}", ok, f"{pm[g]:.4f} -> {cm[g]:.4f} ({delta:+.4f}; "
                              f"allowed -{rules['guard_tolerance']:.2f})")

    # 5. Absolute floors.
    for name, bound in (rules.get("floor") or {}).items():
        if name in cm:
            add(f"floor {name} >= {bound}", cm[name] >= bound, f"{cm[name]:.4f}")

    # 6. Calibration. The pipeline escalates on low confidence, so a model whose confidence
    #    stops meaning anything breaks the escalation logic even while classifying better.
    cal = rules.get("calibration")
    if cal:
        name, bound = cal
        if name in cm:
            add(f"calibration {name} <= {bound}", cm[name] <= bound, f"{cm[name]:.4f}")

    passed = all(c["pass"] for c in checks)
    return {"model": model, "decision": "CANDIDATE" if passed else "REJECTED",
            "checks": checks,
            "note": ("Passing the gate does NOT deploy this model. It may be offered for "
                     "approval." if passed else
                     "Rejected. A model that fails any check above is not offered for "
                     "approval, whatever its headline number says.")}


# ─────────────────────────────────────────────────────────────────────── mlflow (optional)
def log_to_mlflow(model: str, version: str, entry: dict[str, Any]) -> str:
    """Log the run to MLflow when it is available, and never let it break the pipeline.

    The registry JSON is the record; MLflow is the nice-to-have on top. A tracking backend
    that is missing, unmigrated or misconfigured must not be able to lose a training run that
    completed, so the exception is caught and reported rather than raised. MLflow 3 refuses a
    plain directory store outright, which would otherwise have failed the registration of a
    model that had trained perfectly well.

    SQLite rather than a server: no daemon to run, and `mlflow ui --backend-store-uri
    sqlite:///models/mlflow.db` opens it when somebody wants to look at the history.
    """
    try:
        import mlflow

        db = (ROOT / "models" / "mlflow.db").as_posix()
        mlflow.set_tracking_uri(f"sqlite:///{db}")
        mlflow.set_experiment(f"pocus-{model}")
        with mlflow.start_run(run_name=version):
            mlflow.log_params({"dataset": entry.get("dataset"), "version": version,
                               "eval_set": entry.get("eval_set")})
            mlflow.log_metrics({k: float(v) for k, v in entry.get("metrics", {}).items()
                                if isinstance(v, (int, float))})
        return "logged to mlflow (sqlite:///models/mlflow.db)"
    except ImportError:
        return "mlflow not installed -- the run is recorded in registry.json regardless"
    except Exception as e:                      # noqa: BLE001 -- reported, never fatal
        return (f"mlflow logging skipped ({type(e).__name__}) -- the run is recorded in "
                f"registry.json regardless")


# ────────────────────────────────────────────────────────────────────────── commands
def cmd_dataset(args) -> int:
    d = dataset_version(args.model)
    print(json.dumps(d, indent=2))
    return 0 if d["version"] else 1


def cmd_register(args) -> int:
    metrics = json.loads(Path(args.metrics).read_text(encoding="utf8"))
    if args.metrics_key:
        for part in args.metrics_key.split("."):
            metrics = metrics[part]
    reg = load_registry()
    node = reg["models"].setdefault(args.model, {"versions": []})
    # Never reuse a version name. Numbering by list length meant that re-seeding the registry
    # started again at v1, so MLflow -- which is an append-only log and was right to keep both
    # -- ended up holding two different runs both called v3, with different metrics. An audit
    # trail whose names are ambiguous is not an audit trail. The counter only ever goes up.
    used = {v["version"] for v in node["versions"]} | set(node.get("retired_names", []))
    if args.version:
        version = args.version
        if version in used:
            print(f"REFUSED: {args.model} already has a version named {version!r}")
            return 1
    else:
        n = len(node["versions"]) + 1
        while f"v{n}" in used:
            n += 1
        version = f"v{n}"
    entry = {
        "version": version,
        "stage": "candidate",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": args.dataset or dataset_version(args.model)["version"],
        "eval_set": args.eval_set,
        "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
        "artifact": args.artifact,
        "notes": args.notes or "",
    }
    node["versions"].append(entry)
    save_registry(reg)
    print(f"registered {args.model} {version} as CANDIDATE")
    print(" ", log_to_mlflow(args.model, version, entry))
    return 0


def cmd_gate(args) -> int:
    reg = load_registry()
    cand = candidate(reg, args.model)
    if cand is None:
        print(f"no candidate for {args.model}")
        return 1
    result = gate(args.model, cand, production(reg, args.model))
    print(f"{args.model}  candidate {cand['version']}  ->  {result['decision']}\n")
    for c in result["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['check']:<38} {c['detail']}")
    print("\n" + result["note"])
    return 0 if result["decision"] == "CANDIDATE" else 1


def cmd_promote(args) -> int:
    """Promotion is the human gate. It refuses without a named approver, on purpose."""
    if not args.approve:
        print("REFUSED: --approve \"name, and why\" is required.\n"
              "A model reaches patients when a person decides it should, not when a metric "
              "improves. The registry records who decided.")
        return 1
    reg = load_registry()
    node = reg["models"].get(args.model)
    if not node:
        print(f"unknown model {args.model!r}")
        return 1
    target = next((v for v in node["versions"] if v["version"] == args.version), None)
    if target is None:
        print(f"unknown version {args.version!r}")
        return 1

    verdict = gate(args.model, target, production(reg, args.model))
    if verdict["decision"] != "CANDIDATE" and not args.override:
        print(f"REFUSED: {args.version} does not pass the gate. Run "
              f"`gate {args.model}` to see which check failed, or pass --override with a "
              f"reason recorded in --approve.")
        return 1

    for v in node["versions"]:
        if v.get("stage") == "production":
            v["stage"] = "archived"
    target["stage"] = "production"
    target["approved_by"] = args.approve
    target["approved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if args.override:
        target["override"] = True
    save_registry(reg)
    print(f"{args.model} {args.version} -> PRODUCTION, approved by: {args.approve}")
    return 0


def cmd_status(args) -> int:
    reg = load_registry()
    if not reg["models"]:
        print("registry is empty — run `python tools/mlops.py seed` first")
        return 0
    for model, node in reg["models"].items():
        print(f"\n{model}")
        for v in node["versions"]:
            # Plain ASCII: the Windows console defaults to cp1252 and a filled
            # circle here aborts the command that is meant to report the state.
            mark = {"production": "*", "candidate": "o", "archived": "-"}.get(v["stage"], " ")
            key = RULES.get(model, {}).get("key", "accuracy")
            val = v["metrics"].get(key)
            print(f"  {mark} {v['version']:<5} {v['stage']:<11} "
                  f"{key}={val if val is None else round(val, 4)}  "
                  f"dataset={v.get('dataset')}")
            if v.get("approved_by"):
                print(f"      approved by {v['approved_by']}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dataset", help="content hash of a model's training manifests")
    d.add_argument("model")
    d.set_defaults(fn=cmd_dataset)

    r = sub.add_parser("register", help="record a training run as a candidate")
    r.add_argument("model")
    r.add_argument("--metrics", required=True, help="JSON file of metrics")
    r.add_argument("--metrics-key", help="dotted path into that JSON, e.g. report.high")
    r.add_argument("--dataset")
    r.add_argument("--eval-set", required=True, help="what it was scored on")
    r.add_argument("--artifact")
    r.add_argument("--version")
    r.add_argument("--notes")
    r.set_defaults(fn=cmd_register)

    g = sub.add_parser("gate", help="check the latest candidate against production")
    g.add_argument("model")
    g.set_defaults(fn=cmd_gate)

    pr = sub.add_parser("promote", help="human approval; refuses without an approver")
    pr.add_argument("model")
    pr.add_argument("version")
    pr.add_argument("--approve", help="who approved, and why")
    pr.add_argument("--override", action="store_true",
                    help="promote despite a failed gate; recorded in the registry")
    pr.set_defaults(fn=cmd_promote)

    s = sub.add_parser("status", help="show the registry")
    s.set_defaults(fn=cmd_status)

    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
