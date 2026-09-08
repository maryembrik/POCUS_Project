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
import unittest
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _modules() -> list[str]:
    """Every test_*.py beside this file, discovered rather than listed.

    This used to be a hand-maintained list, which failed the way hand-maintained lists do: a
    new file of fourteen tests was added and the suite reported the same total as before, all
    passing. A test that is never run is worse than no test, because the number at the bottom
    says it was.
    """
    return [f"src.agents.tests.{p.stem}"
            for p in sorted(Path(__file__).resolve().parent.glob("test_*.py"))]


MODULES = _modules()


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
            # A test that could not run is not a test that passed. Some tests need an artefact
            # that is not in the repository -- the deployed triage model is 6 MB and rebuilt by
            # a command -- and in a clean checkout they have nothing to exercise. Counting them
            # as passing would make this table report full coverage of a property nothing
            # checked, which is the failure mode this benchmark exists to prevent elsewhere.
            except unittest.SkipTest as exc:
                record["passed"] = False
                record["skipped"] = True
                record["error"] = str(exc)
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

    total = passed_total = skipped_total = 0
    summary = {}
    for prop in sorted(by_prop):
        recs = by_prop[prop]
        n = len(recs)
        p = sum(r["passed"] for r in recs)
        s = sum(bool(r.get("skipped")) for r in recs)
        total += n
        passed_total += p
        skipped_total += s
        summary[prop] = {"tests": n, "passed": p, "skipped": s}
        mark = ("" if p + s == n else "   <-- FAILING") + (
            f"   ({s} skipped)" if s else "")
        print(f"{prop:<{width}} {n:>6} {p:>7}{mark}")

    print("-" * (width + 22))
    ran = total - skipped_total
    print(f"{'TOTAL':<{width}} {total:>6} {passed_total:>7}"
          f"   ({passed_total / ran * 100:.1f}% of {ran} run)" if ran else "no tests found")
    if skipped_total:
        # Named, not folded into the percentage. A skipped test is a property this run did not
        # check, and the number is only useful if a reader can see it.
        print(f"{skipped_total} test(s) skipped: an artefact they need is not in this checkout.")
    print()

    if results["failures"]:
        print("FAILURES")
        print("=" * (width + 22))
        for f in results["failures"]:
            print(f"\n[{f['property']}] {f['test']}")
            print(f"  {f['error']}")
    else:
        print("No failures.")

    return {"summary": summary, "total": total, "passed": passed_total,
            "skipped": skipped_total}


if __name__ == "__main__":
    results = run()
    agg = report(results)
    out = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "total_tests": agg["total"],
        "total_passed": agg["passed"],
        # Recorded so that a reader of this file can tell a run that checked everything from
        # one that could not, which the pass count alone does not distinguish.
        "total_skipped": agg["skipped"],
        "by_property": agg["summary"],
        "failures": [{k: v for k, v in f.items() if k != "traceback"}
                     for f in results["failures"]],
    }
    dest = ROOT / "models" / "safety_benchmark.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2), encoding="utf8")
    print(f"\nwritten: {dest}")
    # Non-zero on a FAILURE, not on a skip. Comparing passed against total treated a test that
    # could not run as one that broke, which turned every clean checkout red -- and a build
    # that is red for a reason nobody can fix is one people learn to ignore.
    sys.exit(1 if results["failures"] else 0)
