"""Run one encounter end to end through the Clinical Reasoning Agent, with a real model.

    python -m src.agents.clinical.run_case              # the missing-data scenario
    python -m src.agents.clinical.run_case --scenario conflict
    python -m src.agents.clinical.run_case --dry-run    # no model; prompt and escalation only

Prints every stage, because the point of the architecture is that the stages are separable:
the escalation decision is computed before the model runs, and the model's answer is checked
against the state afterwards. Both are shown so either can be inspected independently.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .. import schema as S
from .clinical_state import build_clinical_state, render_state
from .llm import DEFAULT_CTX, DEFAULT_MAX_TOKENS, is_available
from .reasoning import reason

ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------------------
# Scenarios -- the worked cases from the report's integration chapter
# ---------------------------------------------------------------------------------------
def _lung(b_lines: float = 0.86) -> dict:
    return S.make_report(
        "lung",
        [S.make_finding("b lines", b_lines)],
        not_detected=[S.make_finding("consolidation", 0.09),
                      S.make_finding("pleural effusion", 0.04),
                      S.make_finding("pleural thickening", 0.11)],
        reliability={"confidence_calibrated": True, "has_normal_class": False,
                     "modelled_findings": ["b_lines", "consolidation",
                                           "pleural_effusion", "pleural_thickening"],
                     "unreliable_findings": ["pleural_effusion"],
                     "scope": "187 clips / 165 cases; pneumothorax NOT modelled"},
        model="effnetb0_multilabel")


def _heart(label: str = "severe dysfunction", conf: float = 0.74) -> dict:
    return S.make_report(
        "heart", [S.make_finding(label, conf)],
        measurements={"ejection_fraction": 28.0},
        reliability={"confidence_calibrated": True, "has_normal_class": True,
                     "scope": "CAMUS-trained; degraded on external domains"},
        model="unet_effb0")


SCENARIOS = {
    # Positive imaging, key laboratory values never drawn.
    "missing": dict(
        triage=S.make_triage("high", 0.79, features={"o2sat": 90, "pulse": 118,
                                                     "bpsys": 104, "respr": 24}),
        ultrasound={"lung": _lung()},
        clinical={"age": 74, "sex": "F", "chief_complaint": "acute breathlessness"},
        labs={},
    ),
    # Triage and imaging disagree.
    "conflict": dict(
        triage=S.make_triage("low", 0.88, features={"o2sat": 96, "pulse": 82,
                                                    "bpsys": 128}),
        ultrasound={"heart": _heart()},
        clinical={"age": 61, "sex": "M", "chief_complaint": "mild chest discomfort"},
        labs={"troponin": 5.0, "lactate": 1.1},
    ),
    # Everything agrees and the record is complete.
    "concordant": dict(
        triage=S.make_triage("high", 0.91, features={"o2sat": 88, "pulse": 122,
                                                     "bpsys": 96, "respr": 28}),
        ultrasound={"lung": _lung(0.92)},
        clinical={"age": 79, "sex": "M", "chief_complaint": "orthopnoea"},
        labs={"troponin": 62.0, "lactate": 2.6, "bnp": 890.0},
    ),
    # Nothing fires. The scan happened, all four findings were screened, none was seen, the
    # vitals are normal and the key labs resulted. The system must still not declare the
    # patient well: the lung module has no healthy class, so a negative read is the absence
    # of the pathologies it knows, not the absence of pathology.
    "reassuring": dict(
        triage=S.make_triage("low", 0.86, features={"o2sat": 98, "pulse": 76,
                                                    "bpsys": 124, "respr": 16}),
        ultrasound={"lung": S.make_report(
            "lung", [],
            not_detected=[S.make_finding("b lines", 0.06),
                          S.make_finding("consolidation", 0.04),
                          S.make_finding("pleural effusion", 0.03),
                          S.make_finding("pleural thickening", 0.08)],
            reliability={"confidence_calibrated": True, "has_normal_class": False,
                         "modelled_findings": ["b_lines", "consolidation",
                                               "pleural_effusion", "pleural_thickening"],
                         "unreliable_findings": ["pleural_effusion"],
                         "scope": "187 clips / 165 cases; pneumothorax NOT modelled"},
            model="effnetb0_multilabel")},
        clinical={"age": 44, "sex": "F", "chief_complaint": "mild breathlessness"},
        labs={"troponin": 4.0, "lactate": 1.0},
    ),
    # The organ that matters was never scanned. A scan that did not happen is not a negative
    # scan, and the system must not answer as though the question had been asked.
    "not_assessed": dict(
        triage=S.make_triage("high", 0.82, features={"o2sat": 93, "pulse": 104,
                                                     "bpsys": 112, "respr": 22}),
        ultrasound={"heart": S.make_report("heart", [], status="not_supported"),
                    "lung": _lung(0.78)},
        clinical={"age": 68, "sex": "M", "chief_complaint": "chest pain and breathlessness"},
        labs={"troponin": 9.0, "lactate": 1.4},
    ),
}


def build(scenario: str) -> dict:
    spec = SCENARIOS[scenario]
    bundle = {"encounter_id": f"DEMO-{scenario.upper()}",
              "triage": spec["triage"], "ultrasound": spec["ultrasound"],
              "clinical": spec["clinical"]}
    for organ, rep in spec["ultrasound"].items():
        errs = S.validate_report(rep)
        assert not errs, f"{organ}: {errs}"
    return build_clinical_state(bundle, labs=spec["labs"])


# ---------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="missing", choices=sorted(SCENARIOS))
    ap.add_argument("--dry-run", action="store_true",
                    help="skip the model; show the state, escalation and prompt only")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    ap.add_argument("--n-ctx", type=int, default=DEFAULT_CTX)
    ap.add_argument("--chat-format", default=None,
                    help="override if the GGUF carries no chat template, e.g. llama-3")
    ap.add_argument("--n-gpu-layers", type=int, default=0,
                    help="-1 offloads every layer to the GPU; 0 is CPU-only")
    ap.add_argument("--max-revisions", type=int, default=1,
                    help="how many times a revisable answer may be sent back")
    ap.add_argument("--retrieval", default="none", choices=("none", "tfidf", "dense"),
                    help="retrieval backend; 'none' reproduces the frozen pre-RAG baseline")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--save", default=None, help="write the full result to this JSON file")
    args = ap.parse_args()

    state = build(args.scenario)

    print("=" * 78)
    print(f"SCENARIO: {args.scenario}   encounter {state['encounter_id']}")
    print("=" * 78)
    print(render_state(state))

    llm_fn = None
    if not args.dry_run:
        ok, msg = is_available()
        print(f"\nmodel: {msg}")
        if not ok:
            print("running dry instead.")
        else:
            from .llm import LlamaCppBackend
            kw = {"n_ctx": args.n_ctx, "max_tokens": args.max_tokens,
                  "n_gpu_layers": args.n_gpu_layers}
            if args.chat_format:
                kw["chat_format"] = args.chat_format
            t0 = time.time()
            llm_fn = LlamaCppBackend(**kw)
            print(f"loaded in {time.time() - t0:.1f}s "
                  f"(n_gpu_layers={args.n_gpu_layers})")

    hits = None
    if args.retrieval != "none":
        from .retrieval import Retriever, retrieval_note
        hits = Retriever(backend=args.retrieval).for_state(state, k=args.top_k)
        print(f"\nretrieval ({args.retrieval}): {retrieval_note(hits)}")
        for h in hits:
            print(f"  [{h['n']}] {h['id']} {h['topic']}  score={h['score']}")

    t0 = time.time()
    out = reason(state, llm_fn=llm_fn, retrieved=hits,
                 max_revisions=args.max_revisions)
    elapsed = time.time() - t0

    print("\n" + "=" * 78)
    print("ESCALATION  (computed before the model ran)")
    print("=" * 78)
    esc = out["escalation"]
    print(f"escalate: {esc['escalate']}   route: {esc['route']}")
    for t in esc["triggers"]:
        print("  -", t)

    print("\n" + "=" * 78)
    print(f"MODEL OUTPUT   ({elapsed:.1f}s)")
    print("=" * 78)
    if llm_fn is None:
        print("(dry run -- prompt built, no model consulted)")
    elif out["validation_errors"]:
        print("DIFFERENTIAL WITHHELD. Validation errors:")
        for e in out["validation_errors"]:
            print("  *", e)
        if "raw_output" in out:
            print("\nraw model output:\n", out["raw_output"][:2000])
    else:
        print(json.dumps(out["differential"], indent=2))

    if args.save:
        Path(args.save).write_text(json.dumps(out, indent=2, default=str), encoding="utf8")
        print(f"\nwritten: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
