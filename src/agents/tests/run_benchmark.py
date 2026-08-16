"""Safety benchmark for the Clinical Reasoning Agent.

Runs every tagged test and reports results grouped by the safety property each exercises,
rather than as one undifferentiated pass count. "Conflict detection 8/8" is a claim about
the system; "47 passed" is not.

    python -m src.agents.tests.run_benchmark

Writes results to models/safety_benchmark.json for the report to cite.
"""
from __future__ import annotations

import importlib
import json
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODULES = [
    "src.agents.tests.test_schema",
    "src.agents.tests.test_clinical_state",
    "src.agents.tests.test_conflicts",
    "src.agents.tests.test_escalation",
    "src.agents.tests.test_hallucination_guard",
    "src.agents.tests.test_negative_findings",
    "src.agents.tests.test_llm_integration",
    "src.agents.tests.test_confidence_guard",
    "src.agents.tests.test_output_quality",
    "src.agents.tests.test_evidence_relationships",
    "src.agents.tests.test_benchmark_scenarios",
    "src.agents.tests.test_unassessed_reporting",
    "src.agents.tests.test_retrieval",
    "src.agents.tests.test_end_to_end_safety",
]


def run() -> dict:
    by_prop: dict[str, list[dict]] = defaultdict(list)
    failures: list[dict] = []

    for mod_name in MODULES:
        mod = importlib.import_module(mod_name)
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            prop = getattr(fn, "safety_property", "Untagged")
            record = {"test": f"{mod_name.split('.')[-1]}::{name}", "passed": True}
            try:
                fn()
            except Exception as exc:                      # noqa: BLE001
                record["passed"] = False
                record["error"] = f"{type(exc).__name__}: {exc}"
                failures.append({**record, "property": prop,
                                 "traceback": traceback.format_exc()})
            by_prop[prop].append(record)

    return {"by_property": dict(by_prop), "failures": failures}


def report(results: dict) -> dict:
    by_prop = results["by_property"]
    width = max(len(p) for p in by_prop) + 2

    print("=" * (width + 22))
    print("CLINICAL REASONING AGENT -- SAFETY BENCHMARK")
    print("=" * (width + 22))
    print(f"{'Safety property':<{width}} {'Tests':>6} {'Passed':>7}")
    print("-" * (width + 22))

    total = passed_total = 0
    summary = {}
    for prop in sorted(by_prop):
        recs = by_prop[prop]
        n, p = len(recs), sum(r["passed"] for r in recs)
        total += n
        passed_total += p
        summary[prop] = {"tests": n, "passed": p}
        mark = "" if p == n else "   <-- FAILING"
        print(f"{prop:<{width}} {n:>6} {p:>7}{mark}")

    print("-" * (width + 22))
    print(f"{'TOTAL':<{width}} {total:>6} {passed_total:>7}"
          f"   ({passed_total / total * 100:.1f}%)" if total else "no tests found")
    print()

    if results["failures"]:
        print("FAILURES")
        print("=" * (width + 22))
        for f in results["failures"]:
            print(f"\n[{f['property']}] {f['test']}")
            print(f"  {f['error']}")
    else:
        print("No failures.")

    return {"summary": summary, "total": total, "passed": passed_total}


if __name__ == "__main__":
    results = run()
    agg = report(results)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_tests": agg["total"],
        "total_passed": agg["passed"],
        "by_property": agg["summary"],
        "failures": [{k: v for k, v in f.items() if k != "traceback"}
                     for f in results["failures"]],
    }
    dest = ROOT / "models" / "safety_benchmark.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(f"\nwritten: {dest}")
    sys.exit(0 if agg["passed"] == agg["total"] else 1)
