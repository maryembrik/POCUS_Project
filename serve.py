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
        "images": _studies_of(enc),
        "reportText": render_report(a["report"]),
        "generatedAt": a["report"]["generated_at"][:16].replace("T", " "),
        "topFinding": (f"{det[0]['label']} {det[0]['confidence']:.2f}" if det
                       else "no finding above threshold"),
    }


def _empty_view() -> dict:
    return {"hasEncounter": False, "images": [],
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
    # Identifiers of studies /api/upload has already read, so that re-opening the patient shows
    # the images the assessment was made from. A record that lists findings but cannot produce
    # the scan they came from is not a record of the examination. Ids rather than pixels: the
    # reading belongs to the image the MODULE saw, and passing it back through the browser
    # would let the two drift apart.
    images: list[str] = []


class Upload(BaseModel):
    organ: str
    image: str                       # base64, with or without a data: prefix
    image2: str | None = None        # end-systole frame, cardiac only
    images: list[str] | None = None  # several studies, or several frames of one clip
    asClip: bool = False             # frames of ONE acquisition, aggregated by median


class Attach(BaseModel):
    id: str
    images: list[str]


class Ask(BaseModel):
    question: str


_last: dict[str, Any] = {}
_studies: dict[str, dict[str, Any]] = {}     # study id -> the stored copy and its reading


def _store_study(b64: str, organ: str, rep: dict, zone: str) -> str:
    """Keep a study beside the reading the module made of it.

    Stored at 320 px as a JPEG rather than as the uploaded file. The record grid shows it at
    178 px, a session that holds twenty studies would otherwise carry twenty full-size images
    through every bootstrap, and the pixels the module actually read were the 224 px tensor it
    was resized to -- so this is a copy for the record, not the study itself, and it is labelled
    that way on screen.
    """
    from PIL import Image

    raw = base64.b64decode(b64.split(",", 1)[-1])
    im = Image.open(io.BytesIO(raw)).convert("L")
    im.thumbnail((320, 320))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=72)

    det = rep.get("findings") or []
    sid = f"IMG-{len(_studies) + 1:03d}"
    _studies[sid] = {
        "id": sid,
        "src": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
        "organ": organ.title(),
        "zone": zone,
        # What the module said about THIS image, not about the encounter it was folded into.
        # A clinician scrolling the history has to be able to tell which scan carried the
        # finding, and an unread or failed study says so rather than showing nothing.
        "finding": (f"{det[0]['label']} {det[0]['confidence']:.2f}" if det
                    else "no finding above threshold" if rep.get("status") == "ok"
                    else f"not read — {rep.get('status', 'unknown')}"),
        "time": datetime.now().strftime("%H:%M"),
    }
    return sid


def _studies_of(enc: dict) -> list[dict[str, Any]]:
    return [_studies[i] for i in (enc.get("images") or []) if i in _studies]


def _remember(enc: dict, a: dict) -> None:
    """Keep the whole encounter, not a summary of it.

    The patient record is a list a clinician opens, so re-opening one has to restore the case
    it actually produced -- its findings, alerts, differential, timeline and report. Storing
    only a headline would mean re-deriving the rest, and a record that reconstructs itself is
    a record that can differ from what was shown at the time.
    """
    rid = str(len(_records) + 1)
    _records.append({
        "id": rid,
        "name": enc.get("name") or "Unnamed patient",
        "age": enc.get("age"), "sex": enc.get("sex"),
        "complaint": enc.get("complaint") or "no complaint given",
        "at": datetime.now().strftime("%H:%M"),
        "severity": a["support"]["severity"]["severity"],
        "alerts": len(a["support"]["alerts"]), "organ": enc.get("organ"),
        "encounterId": a["report"]["encounter_id"],
        "findings": [f["label"] for f in a["state"]["imaging"]["findings"] if f["detected"]],
        "images": [s["id"] for s in _studies_of(enc)],
        "_enc": enc, "_a": a,
    })


@app.get("/api/bootstrap")
def bootstrap() -> JSONResponse:
    # The five benchmark encounters are no longer sent. They are fixtures the test suite runs,
    # and listing them under "Recent assessments" beside a tile reading "0 analysed this
    # session" presented five test cases as five patients waiting to be seen. Every screen now
    # lists only encounters analysed here. They were also re-analysed on EVERY bootstrap --
    # five full pipeline runs per page load and after every assessment -- for a table nobody
    # could act on. `/api/preset` still serves them by key for the recorded-differential path.
    bench = json.loads(BENCH.read_text(encoding="utf8")) if BENCH.exists() else {}
    return JSONResponse({
        "modules": module_status(),
        "tests": bench.get("total_passed"),
        "vitalKeys": [{"key": k, "unit": v["unit"], "min": v["normal_min"],
                       "max": v["normal_max"]} for k, v in VITAL_REFERENCE.items()],
        # Both bounds. Sending only the upper one let the interface call a pH of 6 "normal"
        # because it sits under the maximum, when it is profoundly acidaemic.
        "labKeys": [{"key": k, "unit": v.get("unit", ""), "max": v.get("normal_max"),
                     "min": v.get("normal_min")} for k, v in LAB_REFERENCE.items()],
        "lungFindings": LUNG_FINDINGS,
        "gbClasses": GB_CLASSES,
        "organs": ["Lung", "Heart", "Gallbladder", "Not performed"],
        "records": [dict({k: r[k] for k in ("id", "name", "age", "sex", "complaint", "at",
                                            "severity", "alerts", "organ", "encounterId",
                                            "findings")},
                         images=_studies_of(r["_enc"])) for r in _records],
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
    _remember(enc, a)
    return JSONResponse(_view(enc, a))


@app.post("/api/upload")
def api_upload(u: Upload) -> JSONResponse:
    """Run the real perception module on an uploaded study."""
    import numpy as np
    from PIL import Image

    def grey(b64: str):
        raw = base64.b64decode(b64.split(",", 1)[-1])
        return np.array(Image.open(io.BytesIO(raw)).convert("L"))

    def rows(rep: dict) -> list[dict[str, Any]]:
        return [{"label": f["label"], "caption": finding_caption(f, rep["organ"]),
                 "conf": round(f["confidence"], 2), "detected": True}
                for f in rep["findings"]] + \
               [{"label": f["label"], "caption": finding_caption(f, rep["organ"]),
                 "conf": round(f["confidence"], 2), "detected": False}
                for f in rep.get("not_detected", [])]

    organ = u.organ.lower()
    imgs = u.images or [u.image]
    t0 = time.time()

    if organ == "heart":
        # EF is a comparison between two frames of ONE heart, so two images are two phases of
        # the same acquisition rather than two studies.
        second = u.image2 or (imgs[1] if len(imgs) > 1 else None)
        if not second:
            rep = S.make_report(
                "heart", [], status="failed",
                reliability={"scope": "ejection fraction needs two frames, end-diastole and "
                                      "end-systole; one still cannot produce it"})
        else:
            rep = ultrasound_agent("heart", ed=grey(imgs[0]), es=grey(second))
        per = []
        # Two phases of one acquisition, so they are labelled as phases rather than numbered
        # like separate studies. image2 need not be in `imgs`.
        shots = [(imgs[0], "end-diastole")] + ([(second, "end-systole")] if second else [])
    elif u.asClip and organ == "lung":
        # Frames of one acquisition. The lung module reduces them by MEDIAN, because repeated
        # looks at the same anatomy are not independent evidence and a mean lets one blurred
        # frame carry the result.
        rep = ultrasound_agent("lung", frames=[grey(x) for x in imgs])
        per = []
        shots = [(x, f"frame {i + 1} of {len(imgs)}") for i, x in enumerate(imgs)]
    else:
        # SEPARATE studies. Each image is read on its own and reported on its own; nothing is
        # aggregated across them, because images of different patients are not a clip. The
        # encounter takes the first, since the clinical state holds one report per organ.
        per = [ultrasound_agent(organ, image=grey(x)) for x in imgs]
        rep = per[0]
        shots = [(x, f"study {i + 1} of {len(imgs)}" if len(imgs) > 1 else organ.title())
                 for i, x in enumerate(imgs)]

    # Kept before anything else can go wrong with the encounter. A study that was read is part
    # of the patient's history whether or not the doctor goes on to finish the assessment, and
    # each image is stored against the report of THAT image -- not against the encounter's, so
    # a second scan showing nothing cannot inherit the first one's finding.
    stored = [_store_study(b, organ, per[i] if len(per) > 1 else rep, zone)
              for i, (b, zone) in enumerate(shots)]

    return JSONResponse({
        "report": rep,
        "stored": stored,
        "seconds": round(time.time() - t0, 1),
        "schemaErrors": S.validate_report(rep),
        "rows": rows(rep),
        "count": len(imgs),
        "asClip": bool(u.asClip),
        # One block per uploaded file when they were read as separate studies.
        "perImage": [{"index": i + 1, "status": r["status"], "rows": rows(r)}
                     for i, r in enumerate(per)] if len(per) > 1 else [],
        "note": ("Read as one clip: {n} frames of a single acquisition, reduced by median."
                 .format(n=len(imgs)) if u.asClip and len(imgs) > 1 else
                 "Read as {n} separate studies. The encounter carries the first; the rest are "
                 "reported beside it. Nothing is averaged across them, because images of "
                 "different patients are not a clip.".format(n=len(imgs)) if len(per) > 1 else
                 "One image, one study. A figure showing several scans side by side is still "
                 "one image to the module: it is resized whole, so the reading belongs to none "
                 "of the panels. Upload them as separate files."),
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
    _remember(enc, a)
    return JSONResponse(_view(enc, a))


@app.get("/api/record")
def api_record(id: str) -> JSONResponse:
    """Re-open a patient from the list, exactly as their encounter was assessed."""
    for r in _records:
        if r["id"] == id:
            _last.clear()
            _last.update(enc=r["_enc"], a=r["_a"])
            return JSONResponse(_view(r["_enc"], r["_a"]))
    return JSONResponse({"hasEncounter": False,
                         "message": f"no patient {id!r} in this session"}, status_code=404)


@app.post("/api/record/attach")
def api_attach(a: Attach) -> JSONResponse:
    """File a study against a patient already in the list.

    It does NOT re-run the assessment. The differential, alerts and severity on the record were
    reached without this image, and quietly re-deriving them would replace what was shown to the
    clinician at the time with something else under the same encounter identifier. The screen
    says so beside the control.
    """
    for r in _records:
        if r["id"] == a.id:
            ids = [i for i in a.images if i in _studies]
            r["_enc"]["images"] = (r["_enc"].get("images") or []) + ids
            r["images"] = list(r["_enc"]["images"])
            return JSONResponse(_view(r["_enc"], r["_a"]))
    return JSONResponse({"hasEncounter": False,
                         "message": f"no patient {a.id!r} in this session"}, status_code=404)


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
