"""End-to-end regression over the whole pipeline.

    python tools/run_regression.py

Different in kind from the unit suite. Those tests exercise one property each against a
fixture; this drives a complete encounter from bundle to archived report, for every benchmark
scenario and every priority presentation, and checks the invariants that only exist once the
parts are assembled.

Runs with no GPU and no model weights: the reasoning step uses a scripted backend and a
deliberately failing one, because what is under test here is the pipeline's behaviour, not the
model's answer. Whether the model reasons well is measured in
models/clinical_reasoning_v4_final/ and is a separate question.

Writes models/regression/latest.json so a freeze can record that this passed.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.agents import schema as S  # noqa: E402
from src.agents.clinical.clinical_state import (  # noqa: E402
    build_clinical_state, build_evidence)
from src.agents.clinical.decision_support import decision_support  # noqa: E402
from src.agents.clinical.llm import FailingBackend, ScriptedBackend  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision, reason  # noqa: E402
from src.agents.clinical.retrieval import Retriever, load_corpus  # noqa: E402
from src.agents.clinical.report import archive_report, build_report, render_report  # noqa: E402
from src.agents.clinical.run_case import SCENARIOS, build  # noqa: E402

failures: list[str] = []
checks = 0


def check(condition: bool, label: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(label)


# ---------------------------------------------------------------------------------------
# The priority presentations, built here rather than taken from run_case, because run_case's
# five scenarios exercise reasoning failures while these exercise routing.
# ---------------------------------------------------------------------------------------
def _presentation(complaint: str, *, tier="high", vitals=None, labs=None, organs=("lung",)):
    reports = {}
    for organ in organs:
        if organ == "lung":
            reports["lung"] = S.make_report(
                "lung", [S.make_finding("b lines", 0.81)],
                not_detected=[S.make_finding("consolidation", 0.08)],
                reliability={"confidence_calibrated": True, "has_normal_class": False,
                             "scope": "pneumothorax is NOT modelled and cannot be excluded"})
        else:
            reports[organ] = S.make_report(organ, [], status="not_supported")
    return build_clinical_state(
        {"encounter_id": f"REG-{complaint[:12].replace(' ', '-').upper()}",
         "triage": S.make_triage(tier, 0.8, features=vitals or {}),
         "ultrasound": reports,
         "clinical": {"age": 64, "sex": "M", "chief_complaint": complaint}},
        labs=labs if labs is not None else {})


PRESENTATIONS = {
    "acute_dyspnoea": _presentation("acute breathlessness at rest",
                                    vitals={"o2sat": 89, "pulse": 118, "respr": 28}),
    "undifferentiated_shock": _presentation("hypotension and collapse", tier="high",
                                            vitals={"sbp": 78, "pulse": 132}),
    "trauma": _presentation("blunt trauma after a fall from height",
                            vitals={"sbp": 96, "pulse": 122}),
    "cardiorespiratory_arrest": _presentation("found unresponsive, CPR in progress",
                                              vitals={"sbp": 60, "pulse": 38}),
}


def _benign_control():
    """A control case that SHOULD come out low.

    Every other case here was designed to be difficult, and the first run of this regression
    returned HIGH severity for all nine of them. That is the correct answer for that sample --
    they are hard cases -- but a suite containing only hard cases cannot distinguish a working
    severity rule from one that returns HIGH unconditionally. This case is the discriminator:
    normal vitals, normal laboratory values, a lung scan that screened four findings and saw
    none, and a module that does have a healthy class so nothing is left unexcludable.
    """
    rep = S.make_report(
        "lung", [],
        not_detected=[S.make_finding(n, 0.05) for n in
                      ("b lines", "consolidation", "pleural effusion", "pleural thickening")],
        reliability={"confidence_calibrated": True, "has_normal_class": True})
    return build_clinical_state(
        {"encounter_id": "REG-BENIGN-CONTROL",
         "triage": S.make_triage("low", 0.9, features={"pulse": 72, "respr": 14, "o2sat": 98,
                                                       "sbp": 118, "temp": 36.8}),
         "ultrasound": {"lung": rep},
         "clinical": {"age": 34, "sex": "F", "chief_complaint": "mild cough, feeling well"}},
        labs={"troponin": 4.0, "lactate": 1.0, "bnp": 30.0, "d_dimer": 200.0,
              "crp": 2.0, "wbc": 6.0, "creatinine": 70.0, "ph": 7.40})


def _scripted_answer(state) -> str:
    """A well-formed answer citing one real identifier, so the pipeline reaches its delivery
    path rather than being withheld for a reason unrelated to what is under test."""
    ev = build_evidence(state)
    first = ev[0]["id"] if ev else "E1"
    return json.dumps({
        "differential": [{"diagnosis": "Pulmonary oedema", "likelihood": "moderate",
                          "supporting": [first], "contradicting": [], "limitations": []}],
        "missing_information": ["d_dimer"], "uncertainty": "limited evidence",
        "recommended_next_step": "obtain a d-dimer as additional information"})


def main() -> int:
    print("=" * 78)
    print("END-TO-END REGRESSION")
    print("=" * 78)

    # ---- corpus -------------------------------------------------------------------------
    corpus = load_corpus()
    sourced = sum(p["status"] == "sourced" for p in corpus["passages"])
    check(sourced == len(corpus["passages"]),
          f"corpus has {len(corpus['passages']) - sourced} unsourced unit(s)")
    print(f"\ncorpus {corpus['corpus_version']}: {sourced}/{len(corpus['passages'])} sourced")

    cases = {**{f"benchmark:{k}": build(k) for k in SCENARIOS},
             **{f"presentation:{k}": v for k, v in PRESENTATIONS.items()},
             "control:benign": _benign_control()}

    print(f"\n{'case':<40}{'sev':<10}{'esc':<6}{'alerts':<8}{'report'}")
    print("-" * 78)

    with tempfile.TemporaryDirectory() as archive:
        for name, state in cases.items():
            esc = escalation_decision(state)
            support = decision_support(state, esc)
            sev = support["severity"]

            # --- invariant 1: an absent test never becomes citable evidence ---------------
            labels = {e.get("label") for e in build_evidence(state)}
            for absent in state["missing"]["labs"]:
                check(absent not in labels, f"{name}: absent test '{absent}' has an identifier")

            # --- invariant 2: severity is never LOW beside a critical alert ---------------
            crit = [a for a in support["alerts"] if a["severity"] == "CRITICAL"]
            check(not (crit and sev["severity"] == "LOW"),
                  f"{name}: severity LOW with {len(crit)} critical alert(s)")

            # --- invariant 3: escalation is identical across backends ---------------------
            no_model = reason(state, llm_fn=None)
            scripted = reason(state, llm_fn=ScriptedBackend(_scripted_answer(state)))
            broken = reason(state, llm_fn=FailingBackend("out of memory"))
            check(no_model["escalation"] == esc == scripted["escalation"] == broken["escalation"],
                  f"{name}: escalation differs across backends")

            # --- invariant 4: a failed model still yields severity and alerts -------------
            check(broken["decision_support"]["severity"]["severity"] == sev["severity"],
                  f"{name}: severity lost when the model failed")
            check(broken.get("differential_withheld") is True,
                  f"{name}: a failed backend did not withhold the differential")

            # --- invariant 5: the report renders, archives and hashes ---------------------
            report = build_report(state, scripted, scripted["decision_support"])
            text = render_report(report)
            check(bool(text) and "Not a diagnostic device" in text,
                  f"{name}: report missing its disclaimer")
            check(render_report(report) == text, f"{name}: rendering is not deterministic")

            gaps = report["pocus"]["not_assessed"]
            check(not gaps or "NOT ASSESSED" in text,
                  f"{name}: unassessed modality absent from the rendered report")

            paths = archive_report(report, archive, text=text)
            check(Path(paths["json"]).exists() and len(paths["sha256"]) == 64,
                  f"{name}: archiving failed")

            # --- invariant 6: a withheld differential is visible as withheld --------------
            if scripted.get("differential_withheld"):
                check("WITHHELD" in text, f"{name}: withheld differential not marked in report")

            # The control exists to prove the severity rule discriminates. If it comes out
            # HIGH, either the rule is broken or the case stopped being benign, and both are
            # worth failing over.
            if name == "control:benign":
                check(sev["severity"] == "LOW",
                      f"{name}: benign control graded {sev['severity']}, not LOW")
                check(not esc["escalate"], f"{name}: benign control escalated")
                check(not support["alerts"], f"{name}: benign control raised an alert")

            print(f"{name:<40}{sev['severity']:<10}"
                  f"{'yes' if esc['escalate'] else 'no':<6}"
                  f"{len(support['alerts']):<8}"
                  f"{paths['sha256'][:12]}")

    # ---- retrieval ----------------------------------------------------------------------
    r = Retriever()
    covered = sum(1 for k in SCENARIOS if r.for_state(build(k)))
    print(f"\nretrieval reaches {covered}/{len(SCENARIOS)} benchmark scenarios")
    check(r.retrieve("quarterly revenue forecast for a logistics company") == [],
          "an off-topic query returned a hit")

    # ---- result -------------------------------------------------------------------------
    print("\n" + "=" * 78)
    if failures:
        print(f"REGRESSION FAILED -- {len(failures)} of {checks} checks")
        for f in failures:
            print("  -", f)
    else:
        print(f"REGRESSION PASSED -- {checks} checks over {len(cases)} cases")
    print("=" * 78)

    out = ROOT / "models" / "regression"
    out.mkdir(parents=True, exist_ok=True)
    (out / "latest.json").write_text(json.dumps({
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cases": sorted(cases),
        "checks": checks,
        "failures": failures,
        "passed": not failures,
        "corpus_version": corpus["corpus_version"],
        "retrieval_coverage": f"{covered}/{len(SCENARIOS)}",
    }, indent=2), encoding="utf8")
    print(f"written: {out / 'latest.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
