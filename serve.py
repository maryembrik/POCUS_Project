"""Serve the POCUS Copilot design, driven by the real pipeline.

    python serve.py            →  http://localhost:8000

This is the design's own markup — `web/pocus-copilot.dc.html`, exported from the Claude Design
project and rendered by its own runtime — with every value it displays supplied by this
project's pipeline instead of by the constants the mockup shipped with.

The division is worth stating because it is the whole point of the exercise:

    the DESIGN owns    layout, palette, typography, motion, the twelve screens
    the PIPELINE owns  every number, finding, alert, threshold and sentence on them

Nothing clinical is hard-coded in the page. The mock roster the design shipped with (a named
patient, a visit history, a chat with pre-written replies, a bar chart of invented likelihoods)
is replaced wholesale by `/api/*` responses computed here. Where the system has nothing real to
put in a slot, the slot says so rather than keeping the mockup's placeholder.

No multipart dependency: images arrive as base64 in JSON, so this runs on a bare FastAPI.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.agents import schema as S  # noqa: E402
from src.agents.assistant import answer as assistant_answer  # noqa: E402
from src.agents.clinical.clinical_state import (  # noqa: E402
    LAB_REFERENCE, VITAL_REFERENCE, build_clinical_state, build_evidence)
from src.agents.clinical.decision_support import decision_support  # noqa: E402
from src.agents.clinical.llm import FailingBackend  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision, reason  # noqa: E402
from src.agents.clinical.report import build_report, render_report  # noqa: E402
from src.agents.clinical.retrieval import Retriever  # noqa: E402
from src.agents.clinical.run_case import SCENARIOS, build as build_scenario  # noqa: E402
from src.agents.ultrasound.agent import (  # noqa: E402
    GB_CLASSES, GB_GROUP, LUNG_FINDINGS, finding_caption, module_status, ultrasound_agent)

WEB = ROOT / "web"
FROZEN = ROOT / "models" / "clinical_reasoning_v4_final" / "results.json"
BENCH = ROOT / "models" / "safety_benchmark.json"

app = FastAPI(title="POCUS Copilot")
_retriever: Retriever | None = None
_records: list[dict[str, Any]] = []          # session-scoped, exactly like the Streamlit app


def retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf8")) if FROZEN.exists() else {}


# ═══════════════════════════════════════════════════════════════ benchmark encounters
_VMAP = {"o2sat": "spo2", "pulse": "hr", "bpsys": "sbp", "respr": "rr", "temp": "temp"}
ORGANS_LOWER = ("lung", "heart", "gallbladder")


def _preset(key: str, title: str, tag: str) -> dict:
    sp = SCENARIOS[key]
    tri, us = sp["triage"], sp["ultrasound"]
    scanned = {o: r for o, r in us.items() if r["status"] == "ok"}
    organ = next((o.title() for o in ORGANS_LOWER if o in scanned), "Lung")
    rep = scanned.get(organ.lower(), {})

    def k(label: str) -> str:
        u = label.replace(" ", "_")
        return u if u in LUNG_FINDINGS else label

    findings: dict[str, float] = {}
    for f in rep.get("findings", []):
        findings[k(f["label"])] = f["confidence"]
    for f in rep.get("not_detected", []):
        findings[k(f["label"])] = -f["confidence"]
    return dict(name=title, age=sp["clinical"]["age"], sex=sp["clinical"]["sex"],
                complaint=sp["clinical"]["chief_complaint"], tier=tri["urgency"],
                tconf=float(tri["confidence"]), organ=organ, findings=findings,
                vitals={_VMAP.get(a, a): float(b)
                        for a, b in (tri.get("features") or {}).items()},
                labs=dict(sp["labs"]), key=key, tag=tag,
                unassessed=[o for o, r in us.items() if r["status"] != "ok"])


CASES = {
    "missing": _preset("missing", "Case A", "Critical"),
    "concordant": _preset("concordant", "Case B", "Critical"),
    "conflict": _preset("conflict", "Case C", "Review"),
    "reassuring": _preset("reassuring", "Case D", "Review"),
    "not_assessed": _preset("not_assessed", "Case E", "Critical"),
}


# ═══════════════════════════════════════════════════════════════════════ pipeline
def build_state(enc: dict):
    if enc.get("frozen_key") and enc.get("report") is None:
        return build_scenario(enc["frozen_key"])

    reports, organ = {}, enc.get("organ", "Lung")
    if enc.get("report") is not None:
        reports[enc["report"]["organ"]] = enc["report"]
    elif organ == "Lung":
        det = [S.make_finding(a.replace("_", " "), b)
               for a, b in enc["findings"].items() if b > 0]
        neg = [S.make_finding(a.replace("_", " "), -b)
               for a, b in enc["findings"].items() if b <= 0]
        reports["lung"] = S.make_report(
            "lung", det, not_detected=neg, status="ok" if (det or neg) else "not_supported",
            quality={"no_finding_above_threshold": not det},
            reliability={"confidence_calibrated": True, "has_normal_class": False,
                         "modelled_findings": LUNG_FINDINGS,
                         "scope": "pneumothorax is NOT modelled and cannot be excluded"})
    elif organ == "Heart":
        sd = enc["findings"].get("severe dysfunction", -0.2)
        reports["heart"] = S.make_report(
            "heart", [S.make_finding("severe dysfunction", sd)] if sd > 0 else [],
            not_detected=[] if sd > 0 else [S.make_finding("severe dysfunction", -sd)],
            reliability={"confidence_calibrated": True, "has_normal_class": True,
                         "scope": "CAMUS-like 4CH stills; EF is an area proxy"})
    elif organ == "Gallbladder":
        det = [S.make_finding(a, b, group=GB_GROUP.get(a, ""))
               for a, b in enc["findings"].items() if b > 0]
        reports["gallbladder"] = S.make_report(
            "gallbladder", det,
            reliability={"confidence_calibrated": True, "has_normal_class": False,
                         "modelled_findings": GB_CLASSES,
                         "scope": "teaching-atlas stills; no healthy class exists"})

    for gap in enc.get("not_assessed", []):
        reports.setdefault(gap, S.make_report(
            gap, [], status="not_supported",
            reliability={"scope": "requested but never assessed"}))

    return build_clinical_state(
        {"encounter_id": enc.get("id", "ENC-LIVE"),
         "triage": S.make_triage(enc.get("tier", "medium"), enc.get("tconf", 0.7),
                                 features=enc.get("vitals") or {}),
         "ultrasound": reports,
         "clinical": {"age": enc.get("age"), "sex": enc.get("sex"),
                      "chief_complaint": enc.get("complaint")}},
        labs=enc.get("labs") or {})


def analyse(enc: dict, broken: bool = False) -> dict:
    marks, t0 = [], time.perf_counter()

    def mark(title, detail, hl=False):
        marks.append({"t": time.perf_counter() - t0, "title": title, "detail": detail,
                      "hl": hl})

    mark("Patient registered",
         f"{enc.get('age')} · {enc.get('sex')} · {enc.get('complaint') or 'no complaint'}")
    state = build_state(enc)
    ev = build_evidence(state)
    mark("Clinical state assembled",
         f"{len(ev)} citable fact(s) · "
         f"{len(state['missing']['labs']) + len(state['missing']['vitals'])} absent · "
         f"{len(state.get('conflicts') or [])} conflict(s)")

    esc = escalation_decision(state)
    mark("Escalation evaluated",
         f"{len(esc['triggers'])} trigger(s) → "
         f"{'escalate' if esc['escalate'] else 'answer directly'}", hl=True)

    support = decision_support(state, esc)
    mark("Decision support computed",
         f"severity {support['severity']['severity']} · {len(support['alerts'])} alert(s) · "
         f"{support['scenario']['label']}", hl=True)

    hits = retriever().for_state(state)
    mark("Evidence retrieved", f"{len(hits)} passage(s) above the 0.10 relevance floor")

    if broken:
        result = reason(state, llm_fn=FailingBackend("out of memory"), max_revisions=1)
        origin = "failed"
        mark("Model backend failed", "differential withheld — escalation unaffected", hl=True)
    else:
        rec = (_frozen().get("none") or {}).get(enc.get("frozen_key") or "")
        if rec:
            result, origin = dict(rec, decision_support=support), "recorded"
            mark("Differential loaded",
                 f"{len(rec.get('differential', {}).get('differential') or [])} entries "
                 f"recorded from the GPU run", hl=True)
        else:
            result = reason(state, llm_fn=None)
            result["decision_support"] = support
            origin = "not_generated"
            mark("Reasoning not run", "no GPU in this deployment")

    report = build_report(state, result, support)
    mark("Report assembled", f"encounter {report['encounter_id']}")
    return dict(state=state, esc=esc, support=support, hits=hits, result=result,
                origin=origin, report=report, marks=marks)


# ═══════════════════════════════════════════════════════════ shaping for the design
def _view(enc: dict, a: dict) -> dict:
    """Everything the design's template binds, computed rather than invented."""
    state, sup, esc = a["state"], a["support"], a["esc"]
    diff = (a["result"].get("differential") or {}).get("differential") or []
    ev = build_evidence(state)
    cited = {i for d in diff for i in (d.get("supporting_ids") or [])}
    det = [f for f in state["imaging"]["findings"] if f["detected"]]

    vitals = [{"key": k, "label": k, "unit": VITAL_REFERENCE[k]["unit"],
               "value": f"{state['vitals'][k]['value']:g}"
                        if (state.get("vitals") or {}).get(k) else None,
               "flag": (state.get("vitals") or {}).get(k, {}).get("flag")}
              for k in VITAL_REFERENCE]

    labs = []
    for k, ref in LAB_REFERENCE.items():
        e = (state.get("labs") or {}).get(k)
        rng = (f"&lt;{ref['normal_max']:g}" if "normal_min" not in ref
               else f"{ref['normal_min']:g}–{ref['normal_max']:g}")
        labs.append({"name": k, "result": f"{e['value']:g} {ref['unit']}" if e else None,
                     "ref": rng, "flag": e["flag"] if e else None})

    # The design's bar chart is a supporting-vs-against comparison. These are COUNTS of cited
    # evidence identifiers, not probabilities: the model emits a likelihood band and never a
    # percentage, and painting one here would be the fabrication the system exists to prevent.
    bars = [{"label": str(d.get("diagnosis"))[:26],
             "sup": len(d.get("supporting_ids") or []),
             "ag": len(d.get("contradicting") or [])} for d in diff]

    return {
        "hasEncounter": True,
        "patient": {"name": enc.get("name") or "Unnamed patient", "age": enc.get("age"),
                    "sex": enc.get("sex"), "complaint": enc.get("complaint") or "—",
                    "organ": enc.get("organ"), "id": a["report"]["encounter_id"]},
        "severity": sup["severity"]["severity"],
        "severityReasons": sup["severity"]["reasons"],
        "scenario": sup["scenario"]["label"],
        "thresholdsVersion": str(sup["thresholds_version"]),
        "escalate": esc["escalate"],
        "triggers": esc["triggers"],
        "conclusion": a["report"]["conclusion"],
        "vitals": vitals,
        "labs": labs,
        "findings": [{"label": f["label"], "caption": finding_caption(f, f["organ"]),
                      "conf": round(f["confidence"], 2), "detected": f["detected"]}
                     for f in state["imaging"]["findings"]],
        "notAssessed": state["imaging"]["organs_not_assessed"],
        "outOfScope": state["imaging"]["out_of_scope"],
        "missing": state["missing"]["labs"] + state["missing"]["vitals"],
        "conflicts": state.get("conflicts") or [],
        "caseQuality": (state.get("case_quality") or {}).get("grade"),
        "alerts": [{"severity": x["severity"], "type": x["type"].replace("_", " ").title(),
                    "message": x["message"]} for x in sup["alerts"]],
        "exams": sup["additional_examinations"],
        "therapeutic": sup["therapeutic"],
        "differential": [{"diagnosis": d.get("diagnosis"), "likelihood": d.get("likelihood"),
                          "supporting": d.get("supporting") or [],
                          "supportingIds": d.get("supporting_ids") or [],
                          "contradicting": d.get("contradicting") or [],
                          "limitations": d.get("limitations") or []} for d in diff],
        "differentialOrigin": a["origin"],
        "withheld": bool(a["result"].get("differential_withheld")),
        "validationErrors": a["result"].get("validation_errors") or [],
        "warnings": a["result"].get("warnings") or [],
        "bars": bars,
        "evidence": [{"id": e["id"], "text": e["text"], "cited": e["id"] in cited}
                     for e in ev],
        "hits": [{"n": h["n"], "id": h["id"], "topic": h["topic"], "score": h["score"],
                  "source": h["source"], "text": h["text"]} for h in a["hits"]],
        "timeline": [{"time": f"+{m['t'] * 1000:.0f} ms", "title": m["title"],
                      "detail": m["detail"], "hl": m["hl"]} for m in a["marks"]],
        "reportText": render_report(a["report"]),
        "generatedAt": a["report"]["generated_at"][:16].replace("T", " "),
        "topFinding": (f"{det[0]['label']} {det[0]['confidence']:.2f}" if det
                       else "no finding above threshold"),
    }


def _empty_view() -> dict:
    return {"hasEncounter": False,
            "message": "No encounter has been analysed yet. Enter a patient on New "
                       "assessment; every screen here reads a computed encounter and there "
                       "is not one yet."}


# ═══════════════════════════════════════════════════════════════════════ endpoints
class Encounter(BaseModel):
    name: str = ""
    age: int = 60
    sex: str = "F"
    complaint: str = ""
    tier: str = "medium"
    tconf: float = 0.7
    organ: str = "Lung"
    findings: dict[str, float] = {}
    vitals: dict[str, float] = {}
    labs: dict[str, float] = {}
    preset: str | None = None
    broken: bool = False
    reportJson: dict[str, Any] | None = None


class Upload(BaseModel):
    organ: str
    image: str                      # base64, with or without a data: prefix
    image2: str | None = None       # end-systole frame, cardiac only


class Ask(BaseModel):
    question: str


_last: dict[str, Any] = {}


@app.get("/api/bootstrap")
def bootstrap() -> JSONResponse:
    bench = json.loads(BENCH.read_text(encoding="utf8")) if BENCH.exists() else {}
    roster = []
    for key, c in CASES.items():
        enc = dict(c, frozen_key=key, report=None, not_assessed=list(c["unassessed"]))
        a = analyse(enc)
        roster.append({"key": key, "name": c["name"], "age": c["age"], "sex": c["sex"],
                       "complaint": c["complaint"], "tag": c["tag"],
                       "severity": a["support"]["severity"]["severity"],
                       "alerts": len(a["support"]["alerts"])})
    return JSONResponse({
        "roster": roster,
        "modules": module_status(),
        "tests": bench.get("total_passed"),
        "vitalKeys": [{"key": k, "unit": v["unit"], "min": v["normal_min"],
                       "max": v["normal_max"]} for k, v in VITAL_REFERENCE.items()],
        "labKeys": [{"key": k, "unit": v.get("unit", ""), "max": v.get("normal_max")}
                    for k, v in LAB_REFERENCE.items()],
        "lungFindings": LUNG_FINDINGS,
        "gbClasses": GB_CLASSES,
        "organs": ["Lung", "Heart", "Gallbladder", "Not performed"],
        "records": [{k: r[k] for k in ("name", "at", "severity", "alerts", "organ",
                                       "encounterId", "findings")} for r in _records],
    })


@app.post("/api/analyse")
def api_analyse(e: Encounter) -> JSONResponse:
    enc = e.model_dump()
    preset = enc.pop("preset", None)
    broken = enc.pop("broken", False)
    enc["report"] = enc.pop("reportJson", None)
    enc["not_assessed"] = []

    # A recorded differential belongs to one specific record. It is attached only when the
    # encounter still IS that record: untouched preset, no image analysed here.
    same = False
    if preset in CASES and enc["report"] is None:
        p = CASES[preset]
        same = all(enc[f] == p[f] for f in ("age", "sex", "complaint", "tier", "tconf",
                                            "organ", "vitals", "labs", "findings"))
        if same:
            enc["not_assessed"] = list(p["unassessed"])
    enc["frozen_key"] = preset if same else None
    enc["id"] = f"DEMO-{preset.upper()}" if same else "ENC-LIVE"

    a = analyse(enc, broken)
    _last.clear()
    _last.update(enc=enc, a=a)
    _records.append({
        "name": enc.get("name") or "Unnamed patient", "at": datetime.now().strftime("%H:%M"),
        "severity": a["support"]["severity"]["severity"],
        "alerts": len(a["support"]["alerts"]), "organ": enc.get("organ"),
        "encounterId": a["report"]["encounter_id"],
        "findings": [f["label"] for f in a["state"]["imaging"]["findings"] if f["detected"]],
    })
    return JSONResponse(_view(enc, a))


@app.post("/api/upload")
def api_upload(u: Upload) -> JSONResponse:
    """Run the real perception module on an uploaded study."""
    import numpy as np
    from PIL import Image

    def grey(b64: str):
        raw = base64.b64decode(b64.split(",", 1)[-1])
        return np.array(Image.open(io.BytesIO(raw)).convert("L"))

    organ = u.organ.lower()
    t0 = time.time()
    if organ == "heart":
        if not u.image2:
            rep = S.make_report(
                "heart", [], status="failed",
                reliability={"scope": "ejection fraction needs two frames, end-diastole and "
                                      "end-systole; one still cannot produce it"})
        else:
            rep = ultrasound_agent("heart", ed=grey(u.image), es=grey(u.image2))
    else:
        rep = ultrasound_agent(organ, image=grey(u.image))

    errs = S.validate_report(rep)
    return JSONResponse({
        "report": rep,
        "seconds": round(time.time() - t0, 1),
        "schemaErrors": errs,
        "rows": [{"label": f["label"], "caption": finding_caption(f, rep["organ"]),
                  "conf": round(f["confidence"], 2), "detected": True}
                 for f in rep["findings"]]
        + [{"label": f["label"], "caption": finding_caption(f, rep["organ"]),
            "conf": round(f["confidence"], 2), "detected": False}
           for f in rep.get("not_detected", [])],
    })


@app.get("/api/preset")
def api_preset(key: str, broken: bool = False) -> JSONResponse:
    """Analyse a benchmark encounter as the canonical record, not as a partial form.

    Posting only the patient's name and letting the rest default produced an encounter with
    one citable fact and thirteen absent values, which is a different record from the scenario
    it was named after -- so its recorded differential was correctly refused. The wizard fills
    from here instead.
    """
    if key not in CASES:
        return JSONResponse({"error": f"unknown encounter {key!r}"}, status_code=404)
    c = CASES[key]
    enc = dict(c, frozen_key=key, report=None, not_assessed=list(c["unassessed"]),
               id=f"DEMO-{key.upper()}")
    a = analyse(enc, broken)
    _last.clear()
    _last.update(enc=enc, a=a)
    _records.append({
        "name": c["name"], "at": datetime.now().strftime("%H:%M"),
        "severity": a["support"]["severity"]["severity"],
        "alerts": len(a["support"]["alerts"]), "organ": c["organ"],
        "encounterId": a["report"]["encounter_id"],
        "findings": [f["label"] for f in a["state"]["imaging"]["findings"] if f["detected"]],
    })
    return JSONResponse(_view(enc, a))


@app.post("/api/ask")
def api_ask(a: Ask) -> JSONResponse:
    if not _last:
        return JSONResponse({"answer": "No encounter has been analysed yet, so there is "
                                       "nothing to answer from. Analyse a patient first."})
    return JSONResponse({"answer": assistant_answer(a.question, _last["a"])})


@app.get("/api/view")
def api_view() -> JSONResponse:
    if not _last:
        return JSONResponse(_empty_view())
    return JSONResponse(_view(_last["enc"], _last["a"]))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "pocus-copilot.dc.html")


app.mount("/", StaticFiles(directory=str(WEB)), name="web")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8501))
    # Plain ASCII: the Windows console defaults to cp1252 and an arrow here aborts startup.
    print(f"POCUS-Emergency running at http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
