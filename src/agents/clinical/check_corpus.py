"""Report the state of the retrieval corpus.

    python -m src.agents.clinical.check_corpus

Deliberately a report rather than a test. Placeholders are the expected state while the
corpus is being written, so failing on them would mean a red suite for weeks. What this
catches is the thing that would quietly break the system: a unit marked 'sourced' that is
still carrying placeholder text or no citation, which would let an answer claim guideline
grounding it does not have.
"""
from __future__ import annotations

import sys
from collections import Counter

from .retrieval import Retriever, load_corpus

MIN_SOURCED_CHARS = 120          # two or three sentences; below this it is a stub
PLACEHOLDER_MARKERS = ("REPLACE", "PLACEHOLDER", "TODO")


def main() -> int:
    data = load_corpus()
    passages = data["passages"]
    status = Counter(p["status"] for p in passages)

    print("=" * 74)
    print(f"CORPUS  {data['corpus_version']}   {len(passages)} units")
    print("=" * 74)
    for k, n in sorted(status.items()):
        bar = "#" * round(40 * n / len(passages))
        print(f"  {k:<12} {n:>3}  {bar}")

    # ---- the failures that matter -------------------------------------------------
    problems: list[str] = []
    for p in passages:
        if p["status"] != "sourced":
            continue
        if any(m in p["source"].upper() for m in PLACEHOLDER_MARKERS):
            problems.append(f"{p['id']} is marked sourced but its citation is a placeholder")
        if any(m in p["text"].upper() for m in PLACEHOLDER_MARKERS):
            problems.append(f"{p['id']} is marked sourced but its text is a placeholder")
        if len(p["text"]) < MIN_SOURCED_CHARS:
            problems.append(f"{p['id']} is marked sourced but its text is only "
                            f"{len(p['text'])} characters -- too short to be a passage")

    ids = [p["id"] for p in passages]
    dupes = [i for i, n in Counter(ids).items() if n > 1]
    problems += [f"duplicate id: {i}" for i in dupes]

    print()
    if problems:
        print("PROBLEMS")
        for x in problems:
            print("  -", x)
    else:
        print("No unit claims a source it does not have.")

    # ---- does retrieval actually reach each group? --------------------------------
    print()
    print("Retrieval reachability (does a plausible query find each group?)")
    r = Retriever()
    probes = {
        "lung":     "B-lines lung sliding consolidation pleural effusion",
        "cardiac":  "left ventricular function pericardial effusion tamponade",
        "FAST":     "free fluid trauma right upper quadrant",
        "shock":    "undifferentiated shock classification",
        "bounds":   "what a negative study does not establish",
    }
    for name, q in probes.items():
        hits = r.retrieve(q, k=3)
        got = ", ".join(h["id"] for h in hits) or "NOTHING -- this group is unreachable"
        print(f"  {name:<9} -> {got}")

    remaining = status.get("placeholder", 0)
    print()
    if remaining:
        print(f"{remaining} unit(s) still to source. Until they are, reason() marks any "
              f"answer resting on them as NOT guideline-grounded.")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
