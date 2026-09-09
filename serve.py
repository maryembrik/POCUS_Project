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
import collections
import contextlib
import io
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from fastapi import Depends, FastAPI, HTTPException, Request, Response  # noqa: E402
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from src.auth import accounts, sessions, store  # noqa: E402
from src.auth.models import Doctor  # noqa: E402

from src.agents import i18n, schema as S  # noqa: E402
from src.agents.assistant import answer as assistant_answer  # noqa: E402
from src.agents.clinical.clinical_state import (  # noqa: E402
    LAB_REFERENCE, VITAL_REFERENCE, build_clinical_state, build_evidence)
from src.agents.clinical.decision_support import decision_support, load_thresholds  # noqa: E402
from src.agents.clinical.llm import FailingBackend  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision, reason  # noqa: E402
from src.agents.clinical.report import build_report, render_report  # noqa: E402
from src.agents.clinical.retrieval import Retriever  # noqa: E402
from src.agents.clinical.run_case import SCENARIOS, build as build_scenario  # noqa: E402
from src.agents.triage.suggest import OBSERVATIONS as triage_observations  # noqa: E402
from src.agents.triage.suggest import suggest as triage_suggest  # noqa: E402
from src.agents.ultrasound.agent import (  # noqa: E402
    GB_CLASSES, GB_GROUP, LUNG_FINDINGS, NOT_MODELLED, finding_caption, lung_reliability,
    module_status, ultrasound_agent)

WEB = ROOT / "web"
FROZEN = ROOT / "models" / "clinical_reasoning_v4_final" / "results.json"
BENCH = ROOT / "models" / "safety_benchmark.json"

@contextlib.asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Seed the demo account named by POCUS_DEMO_PASSWORD, if there is one, once at startup.

    Absent that variable nothing is created. An image that ships with a known password
    reachable from the network is a back door however clearly it is labelled a demo, so the
    password is supplied at run time and never lives in the repository.
    """
    made = accounts.ensure_demo()
    if made:
        # The username only. Logging the password would put it in exactly the place the
        # monitoring section of this file exists to keep secret material out of.
        logging.getLogger("pocus.access").info("seeded demo account %r", made[0])
    yield


app = FastAPI(title="POCUS Copilot", lifespan=_lifespan)
_retriever: Retriever | None = None
_STARTED = time.time()                       # for /api/health; set once, at import


class Workspace:
    """One doctor's working set: their records, their studies, their current encounter.

    These three were module globals, which was correct while the application had exactly one
    user and became a patient-data leak the moment it had accounts. `_last` in particular was a
    single dict holding "the encounter being looked at" -- so with two clinicians signed in, one
    analysing a patient would change what the other's next screen refresh showed, and
    /api/ask would answer questions about the wrong patient. Nothing in the interface would
    have looked wrong; it would simply have been someone else's case.

    Keyed by doctor id below. The database is the durable record; this is the per-doctor cache
    the screens read, hydrated from it on first use after signing in.
    """

    def __init__(self, doctor_id: str) -> None:
        self.doctor_id = doctor_id
        self.records: list[dict[str, Any]] = []
        self.studies: dict[str, dict[str, Any]] = {}
        self.last: dict[str, Any] = {}
        self.hydrated = False

    def hydrate(self) -> Workspace:
        """Load this doctor's stored examinations once per process, newest last.

        Without this, signing in after a restart would show an empty patient list beside a
        database that holds every case -- the screens would report no history where history
        exists, which is worse than an error because it looks like an answer.
        """
        if self.hydrated:
            return self
        self.hydrated = True
        for row in reversed(store.list_examinations(self.doctor_id)):
            full = store.load_examination(self.doctor_id, row["id"])
            if not full or not full.get("assessment"):
                continue
            for st in full.get("studies") or []:
                self.studies[st["id"]] = st
            self.records.append(_record_row(full["id"], full["encounter"], full["assessment"],
                                            self, at=row["at"][11:16],
                                            patient_id=full.get("patientId")))
        return self


_workspaces: dict[str, Workspace] = {}


def _ws(doctor: Doctor) -> Workspace:
    ws = _workspaces.get(doctor.id)
    if ws is None:
        ws = _workspaces[doctor.id] = Workspace(doctor.id)
    return ws.hydrate()


def current_doctor(request: Request) -> Doctor:
    """Who is asking. 401 when the answer is nobody.

    Every endpoint that can reach clinical data depends on this, so there is no route that
    reads a record without a doctor bound to the request. /api/health and /api/metrics are the
    two deliberate exceptions: the container's HEALTHCHECK runs without credentials, and
    neither reports anything about a patient.
    """
    doctor = sessions.doctor_for(request.cookies.get(sessions.COOKIE))
    if doctor is None:
        raise HTTPException(status_code=401, detail="sign in required")
    return doctor


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
def _suggestion_for(enc: dict) -> dict[str, Any] | None:
    """The deployed model's proposal for this encounter, or None if it is not installed.

    Shaped for the clinical state rather than for the screen: `urgency` to match the key the
    triage section uses, so the two can be compared without either side knowing where the
    other came from.
    """
    s = triage_suggest(enc)
    if s is None:
        return None
    return {"urgency": s["tier"], "probabilities": s["probabilities"], "model": s["model"]}


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
            # The module's own declaration, not a second copy of it written from memory. A
            # finding entered by hand and the same finding produced from an image must carry
            # the same account of what the model can support.
            reliability=lung_reliability())
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
         # model="clinician" and not a model name, because no model produced this. The tier is
         # the assessment the clinician entered on the intake screen. The trained triage
         # classifier exists and is evaluated in the report, but it is deliberately not wired
         # here: it needs arrival mode and coded reason-for-visit categories this intake does
         # not collect. Recording the provenance in the encounter itself means a reader of the
         # archived record can tell the two apart without knowing which build produced it.
         "triage": S.make_triage(enc.get("tier", "medium"), enc.get("tconf", 0.7),
                                 features=enc.get("vitals") or {},
                                 model="clinician"),
         # The model's proposal, kept beside the clinician's tier and never in place of it.
         # Recomputed here from the encounter rather than taken from the request: a record the
         # caller can write is not a record, and this one exists to be reviewed afterwards.
         "triage_suggestion": _suggestion_for(enc),
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

    # Recomputed here rather than accepted from the request. The suggestion is part of the
    # record of how this encounter was graded, and a record the client can write is not a
    # record -- a caller could report agreement that never happened.
    #
    # The tier that reached the reasoning layer is the clinician's either way. What this adds
    # is whether a model proposed the same thing, which is only auditable if it is stored at
    # the moment the assessment was made.
    audit = None
    suggestion = triage_suggest(enc)
    if suggestion is not None:
        final = enc.get("tier", "medium")
        audit = {
            "suggested": suggestion["tier"],
            "final": final,
            "probabilities": suggestion["probabilities"],
            "model": suggestion["model"],
            "observations_entered": suggestion["observations_entered"],
            "observations_missing": suggestion["observations_missing"],
            # Neither party is recorded as correct. The system detects that two assessments
            # differ; it has no ground truth with which to say which one was right, and a field
            # called "ai_wrong" would assert exactly that.
            "agreement": "agreed" if suggestion["tier"] == final else "clinician_differs",
        }
        if audit["agreement"] == "clinician_differs":
            mark("Urgency differs from the suggestion",
                 f"suggested {suggestion['tier']}, recorded {final}")

    return dict(state=state, esc=esc, support=support, hits=hits, result=result,
                origin=origin, report=report, marks=marks, triage_audit=audit)


# ═══════════════════════════════════════════════════════════ shaping for the design
def _organs_examined(state: dict) -> set[str]:
    """Organs a module actually read, which is where an unmodelled finding is worth naming.

    Only these: telling a clinician that pneumothorax was not assessed on an organ nobody
    scanned adds a row that says nothing, and the organ's own absence is already reported.
    """
    return {f["organ"] for f in state["imaging"]["findings"]}


def _view(enc: dict, a: dict, lang: str = "en", ws: Workspace | None = None) -> dict:
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

    # French is rendered from the SAME computed fields, never from the English sentence, so a
    # French screen cannot state something the English record does not. The English strings
    # remain what the tests assert and what the archived report is built from.
    fr = lang == "fr"

    return {
        "hasEncounter": True,
        "lang": lang,
        "patient": {"name": enc.get("name") or ("Patient sans nom" if fr
                                                else "Unnamed patient"),
                    "age": enc.get("age"),
                    "sex": enc.get("sex"), "complaint": enc.get("complaint") or "—",
                    "organ": (i18n.organ(enc.get("organ", "")).title() if fr
                              else enc.get("organ")),
                    "id": a["report"]["encounter_id"]},
        "severity": i18n.severity(sup["severity"]["severity"]) if fr
                    else sup["severity"]["severity"],
        "severityKey": sup["severity"]["severity"],
        "severityReasons": sup["severity"]["reasons"],
        "scenario": i18n.scenario(sup["scenario"]["label"]) if fr
                    else sup["scenario"]["label"],
        "thresholdsVersion": str(sup["thresholds_version"]),
        "escalate": esc["escalate"],
        "triggers": [i18n.trigger(t) for t in esc["triggers"]] if fr else esc["triggers"],
        "conclusion": i18n.conclusion(a["report"]) if fr else a["report"]["conclusion"],
        "vitals": vitals,
        "labs": labs,
        # Four states, not two. "Detected" and "not detected" leave a reader to assume that
        # anything unmentioned was checked and clear, and on this system two different things
        # hide in that assumption: a finding the model reports but cannot support (pleural
        # effusion rests on 16 positive cases), and a finding it never models at all
        # (pneumothorax). Both are shown as their own state so the list a clinician reads is
        # the whole picture rather than the part the model happened to be able to answer.
        "findings": [{"label": f["label"], "caption": finding_caption(f, f["organ"]),
                      "conf": round(f["confidence"], 2), "detected": f["detected"],
                      "state": ("unreliable" if f["detected"] and f["low_evidence"]
                                else "detected" if f["detected"]
                                else "screened_negative")}
                     for f in state["imaging"]["findings"]]
        # The classes a single-label module ranked below the one it reported. Shown so a reader
        # can tell a diagnosis that was weighed and rejected from one that was never in the
        # label set -- but as their own state, because they are not independent negatives.
        + [{"label": f["label"], "caption": f["organ"],
            "conf": round(f["confidence"], 2), "detected": False, "state": "alternative"}
           for f in state["imaging"].get("alternatives", [])]
        + [{"label": label, "caption": organ, "conf": None, "detected": False,
            "state": "not_assessed"}
           for organ in sorted(_organs_examined(state))
           for label in NOT_MODELLED.get(organ, [])],
        "notAssessed": ([i18n.organ(o) for o in state["imaging"]["organs_not_assessed"]] if fr
                        else state["imaging"]["organs_not_assessed"]),
        "outOfScope": ([i18n.limit(x) for x in state["imaging"]["out_of_scope"]] if fr
                       else state["imaging"]["out_of_scope"]),
        "missing": ([i18n.measure(m) for m in
                     state["missing"]["labs"] + state["missing"]["vitals"]] if fr
                    else state["missing"]["labs"] + state["missing"]["vitals"]),
        "conflicts": ([i18n.conflict(c) for c in (state.get("conflicts") or [])] if fr
                      else state.get("conflicts") or []),
        "caseQuality": (i18n.GRADE.get((state.get("case_quality") or {}).get("grade"))
                        if fr else (state.get("case_quality") or {}).get("grade")),
        "alerts": [{"severity": i18n.LEVEL.get(x["severity"], x["severity"]) if fr
                                else x["severity"],
                    # The key stays English so the interface can compare on it. A CRITICAL
                    # alert must render as critical whatever language it is displayed in.
                    "severityKey": x["severity"],
                    "type": i18n.alert_type(x["type"]) if fr
                            else x["type"].replace("_", " ").title(),
                    "message": i18n.alert(x) if fr else x["message"]}
                   for x in sup["alerts"]],
        "exams": ([{"exam": i18n.exam_name(x["exam"]),
                    "priority": i18n.PRIORITY.get(x["priority"], x["priority"]),
                    "reason": i18n.exam_reason(x["reason"])}
                   for x in sup["additional_examinations"]] if fr
                  else sup["additional_examinations"]),
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
        "images": _studies_of(ws, enc) if ws else [],
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
    # Walk-in or ambulance. Not a vital sign, so it does not belong in `vitals`, but it is one
    # of the features the deployed triage model is fitted on and one of the strongest: how a
    # patient arrived carries information no measurement at the bedside repeats.
    arrival: str = "walk-in"
    organ: str = "Lung"
    findings: dict[str, float] = {}
    vitals: dict[str, float] = {}
    labs: dict[str, float] = {}
    # The patient this assessment is filed against, or None for one not yet filed. It MUST be
    # declared here: pydantic drops fields a model does not declare, silently and by default,
    # so while this line was missing the browser's patientId never arrived. The ownership check
    # in api_analyse read None every time and passed, and every examination was stored unfiled
    # -- a security control that could not fire and a feature that did not work, from one
    # absent field and no error anywhere.
    patientId: str | None = None
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


_NO_STORE = {"Cache-Control": "no-store, must-revalidate", "Pragma": "no-cache"}


def _store_study(ws: Workspace, b64: str, organ: str, rep: dict, zone: str) -> str:
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
    sid = f"IMG-{len(ws.studies) + 1:03d}"
    ws.studies[sid] = {
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


def _studies_of(ws: Workspace, enc: dict) -> list[dict[str, Any]]:
    return [ws.studies[i] for i in (enc.get("images") or []) if i in ws.studies]


def _record_row(rid: str, enc: dict, a: dict, ws: Workspace, *, at: str | None = None,
                patient_id: str | None = None) -> dict[str, Any]:
    """Keep the whole encounter, not a summary of it.

    The patient record is a list a clinician opens, so re-opening one has to restore the case
    it actually produced -- its findings, alerts, differential, timeline and report. Storing
    only a headline would mean re-deriving the rest, and a record that reconstructs itself is
    a record that can differ from what was shown at the time.
    """
    return {
        "id": rid,
        "patientId": patient_id,
        "name": enc.get("name") or "Unnamed patient",
        "age": enc.get("age"), "sex": enc.get("sex"),
        "complaint": enc.get("complaint") or "no complaint given",
        "at": at or datetime.now().strftime("%H:%M"),
        "severity": a["support"]["severity"]["severity"],
        "alerts": len(a["support"]["alerts"]), "organ": enc.get("organ"),
        "encounterId": a["report"]["encounter_id"],
        "findings": [f["label"] for f in a["state"]["imaging"]["findings"] if f["detected"]],
        "images": [s["id"] for s in _studies_of(ws, enc)],
        "_enc": enc, "_a": a,
    }


def _remember(ws: Workspace, enc: dict, a: dict, patient_id: str | None) -> str | None:
    """Write the assessment to the database, then cache it for this doctor's screens.

    The database first, deliberately. If the write fails the clinician must not be shown a
    record that looks stored and is not -- so the row id the store returns becomes the record
    id, and the identifier on screen is the identifier in the database rather than a position
    in a list. The old `str(len(records) + 1)` was guessable, which mattered not at all while
    every user saw everything and matters entirely now that records have owners.
    """
    # No patient chosen means one is made from what the clinician already typed, rather than
    # the examination being stored unfiled. The intake form collects a name, an age and a sex;
    # requiring the doctor to ALSO pick a record from a list before the system will look at the
    # case is paperwork imposed at the worst possible moment.
    if patient_id is None:
        found = store.find_or_create_patient(ws.doctor_id, name=enc.get("name"),
                                             age=enc.get("age"), sex=enc.get("sex"))
        patient_id = found["id"] if found else None

    saved = store.save_examination(ws.doctor_id, patient_id, encounter=enc, assessment=a,
                                   studies=_studies_of(ws, enc))
    if saved is None:                      # the patient is not this doctor's; nothing is filed
        return None
    ws.records.append(_record_row(saved["id"], enc, a, ws, patient_id=patient_id))
    return saved["id"]


@app.get("/api/bootstrap")
def bootstrap(lang: str = "en", doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    # The five benchmark encounters are no longer sent. They are fixtures the test suite runs,
    # and listing them under "Recent assessments" beside a tile reading "0 analysed this
    # session" presented five test cases as five patients waiting to be seen. Every screen now
    # lists only encounters analysed here. They were also re-analysed on EVERY bootstrap --
    # five full pipeline runs per page load and after every assessment -- for a table nobody
    # could act on. `/api/preset` still serves them by key for the recorded-differential path.
    bench = json.loads(BENCH.read_text(encoding="utf8")) if BENCH.exists() else {}
    ws = _ws(doctor)
    return JSONResponse({
        "modules": module_status(),
        "tests": bench.get("total_passed"),
        # Who is signed in, so the interface can greet them and offer a way out. The patient
        # list below is this doctor's and nobody else's -- it is read from their workspace,
        # which is keyed by their id.
        "doctor": {"name": doctor.name, "username": doctor.username},
        "patients": store.list_patients(doctor.id),
        "vitalKeys": [{"key": k, "unit": v["unit"], "min": v["normal_min"],
                       "max": v["normal_max"]} for k, v in VITAL_REFERENCE.items()],
        # Both bounds. Sending only the upper one let the interface call a pH of 6 "normal"
        # because it sits under the maximum, when it is profoundly acidaemic.
        "labKeys": [{"key": k, "unit": v.get("unit", ""), "max": v.get("normal_max"),
                     "min": v.get("normal_min")} for k, v in LAB_REFERENCE.items()],
        "lungFindings": LUNG_FINDINGS,
        "gbClasses": GB_CLASSES,
        "organs": ["Lung", "Heart", "Gallbladder", "Not performed"],
        # `severityKey` is the English key the interface compares on; `severity` is what it
        # displays. Keeping both means a French page can count high-severity patients without
        # matching on a translated word, which is the kind of comparison that breaks silently
        # in one language and not the other.
        "records": [dict({k: r[k] for k in ("id", "name", "age", "sex", "complaint", "at",
                                            "alerts", "organ", "encounterId", "findings")},
                         severityKey=r["severity"],
                         severity=(i18n.severity(r["severity"]) if lang == "fr"
                                   else r["severity"]),
                         organ=(i18n.organ(r["organ"] or "").title() if lang == "fr"
                                else r["organ"]),
                         images=_studies_of(ws, r["_enc"])) for r in ws.records],
    })


@app.post("/api/analyse")
def api_analyse(e: Encounter, lang: str = "en",
                doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    ws = _ws(doctor)
    enc = e.model_dump()
    patient_id = enc.pop("patientId", None)
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

    # A patient id that is not this doctor's files nothing, and is reported as a missing
    # patient exactly as an id that does not exist would be. Answering "forbidden" would
    # confirm the id is real, which is the disclosure the 404 avoids.
    if patient_id is not None and store.get_patient(doctor.id, patient_id) is None:
        raise HTTPException(status_code=404, detail="no such patient")

    a = analyse(enc, broken)
    ws.last = {"enc": enc, "a": a}
    _remember(ws, enc, a, patient_id)
    return JSONResponse(_view(enc, a, lang, ws))


@app.post("/api/upload")
def api_upload(u: Upload, doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    """Run the real perception module on an uploaded study."""
    ws = _ws(doctor)
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
    stored = [_store_study(ws, b, organ, per[i] if len(per) > 1 else rep, zone)
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
def api_preset(key: str, broken: bool = False, lang: str = "en",
               doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    """Analyse a benchmark encounter as the canonical record, not as a partial form.

    Posting only the patient's name and letting the rest default produced an encounter with
    one citable fact and thirteen absent values, which is a different record from the scenario
    it was named after -- so its recorded differential was correctly refused. The wizard fills
    from here instead.
    """
    if key not in CASES:
        return JSONResponse({"error": f"unknown encounter {key!r}"}, status_code=404)
    ws = _ws(doctor)
    c = CASES[key]
    enc = dict(c, frozen_key=key, report=None, not_assessed=list(c["unassessed"]),
               id=f"DEMO-{key.upper()}")
    a = analyse(enc, broken)
    ws.last = {"enc": enc, "a": a}
    _remember(ws, enc, a, None)
    return JSONResponse(_view(enc, a, lang, ws))


@app.get("/api/record")
def api_record(id: str, lang: str = "en",
               doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    """Re-open a patient from the list, exactly as their encounter was assessed.

    The workspace is searched first and the database second, so a record stored before the last
    restart opens the same way as one assessed a minute ago. Both lookups are this doctor's
    only: the list comprehension runs over their workspace, and load_examination takes their id
    as an argument it cannot be called without.
    """
    ws = _ws(doctor)
    for r in ws.records:
        if r["id"] == id:
            ws.last = {"enc": r["_enc"], "a": r["_a"]}
            return JSONResponse(_view(r["_enc"], r["_a"], lang, ws))

    full = store.load_examination(doctor.id, id)
    if full and full.get("assessment"):
        for st in full.get("studies") or []:
            ws.studies.setdefault(st["id"], st)
        ws.last = {"enc": full["encounter"], "a": full["assessment"]}
        return JSONResponse(_view(full["encounter"], full["assessment"], lang, ws))

    # Someone else's record and a record that never existed give the same answer. A 403 here
    # would tell an unauthorised caller which identifiers are real.
    return JSONResponse({"hasEncounter": False,
                         "message": f"no patient {id!r} in this session"}, status_code=404)


@app.post("/api/record/attach")
def api_attach(a: Attach, lang: str = "en",
               doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    """File a study against a patient already in the list.

    It does NOT re-run the assessment. The differential, alerts and severity on the record were
    reached without this image, and quietly re-deriving them would replace what was shown to the
    clinician at the time with something else under the same encounter identifier. The screen
    says so beside the control.
    """
    ws = _ws(doctor)
    for r in ws.records:
        if r["id"] == a.id:
            ids = [i for i in a.images if i in ws.studies]
            r["_enc"]["images"] = (r["_enc"].get("images") or []) + ids
            r["images"] = list(r["_enc"]["images"])
            # Written through, or the attachment survives only until the process restarts --
            # and a study that is on screen but not in the record is worse than one that was
            # never filed, because the clinician has been told it is there.
            store.attach_studies(doctor.id, a.id, [ws.studies[i] for i in ids])
            return JSONResponse(_view(r["_enc"], r["_a"], lang, ws))
    return JSONResponse({"hasEncounter": False,
                         "message": f"no patient {a.id!r} in this session"}, status_code=404)


@app.post("/api/ask")
def api_ask(a: Ask, lang: str = "en",
            doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    ws = _ws(doctor)
    if not ws.last:
        return JSONResponse({"answer":
            "Aucune prise en charge n'a encore été analysée : il n'y a rien à partir de quoi "
            "répondre. Analysez d'abord un patient." if lang == "fr" else
            "No encounter has been analysed yet, so there is nothing to answer from. "
            "Analyse a patient first."})
    return JSONResponse({"answer": assistant_answer(a.question, ws.last["a"], lang)})


@app.get("/api/view")
def api_view(lang: str = "en", doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    ws = _ws(doctor)
    if not ws.last:
        return JSONResponse(_empty_view())
    return JSONResponse(_view(ws.last["enc"], ws.last["a"], lang, ws))


@app.get("/api/health")
def api_health() -> JSONResponse:
    """Whether this instance can actually do its job, not merely whether the socket is open.

    A container that answers 200 while its perception modules failed to load is worse than one
    that is plainly down: it takes traffic and reports nothing found, which on this system is
    indistinguishable to a reader from an examination that found nothing. So the check reports
    each module's real state -- weights on disk, calibrator loaded -- and degrades the overall
    status when any of them cannot run.

    `degraded` rather than a failure code when a module is missing: the deterministic layer
    (escalation, conflicts, the missing-data account) does not need the models and remains
    correct without them, so an instance in that state is still worth routing to. It is the
    orchestrator's business to decide what to do about it, and it can only decide from a
    report that distinguishes the cases.
    """
    modules = module_status()
    ready = all(m["runs"] for m in modules.values())
    return JSONResponse({
        "status": "ok" if ready else "degraded",
        "modules": modules,
        # The safety layer is what must behave identically wherever this runs, so its version
        # is reported beside the model states rather than left to be inferred from an image tag.
        "thresholds_version": load_thresholds().get("version"),
        "schema_version": S.SCHEMA_VERSION,
        # Summed across every signed-in doctor, not broken down by one. This endpoint is
        # unauthenticated because the container's HEALTHCHECK calls it, so it may report how
        # much work the process is holding and nothing about whose work it is.
        "encounters_held": sum(len(w.records) for w in _workspaces.values()),
        "uptime_seconds": round(time.time() - _STARTED, 1),
    }, headers=_NO_STORE)


@app.post("/api/triage/suggest")
def api_triage_suggest(enc: Encounter, lang: str = "en",
                       doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    """An urgency suggestion for what has been entered so far.

    Returns a tier and the observations still outstanding, in clinical words. The probability
    vector, the model name and its macro F1 are computed and recorded with the encounter, but
    they are not what a clinician needs at intake -- a screen that reports "0.646 medium,
    triage_deployed v1, macro F1 0.567" asks the reader to audit a classifier when their job is
    to see a patient.
    """
    s = triage_suggest(enc.model_dump())
    if s is None:
        return JSONResponse({"available": False})
    return JSONResponse({
        "available": True,
        # The tier key, not a translated word. The interface already holds both languages for
        # these three and renders whichever it is showing.
        "tier": s["tier"],
        "entered": s["observations_entered"],
        "of": len(triage_observations),
        "missing": s["observations_missing"],
        # Computed, recorded with the encounter, and deliberately not put on the intake screen.
        "_audit": {"probabilities": s["probabilities"], "model": s["model"]},
    }, headers=_NO_STORE)


@app.get("/api/metrics")
def api_metrics() -> JSONResponse:
    """Traffic, latency and failures for this instance, since it started.

    In-process counters, deliberately: they reset when the container does, they are not shared
    between replicas, and they are not a substitute for a metrics backend. What they give is
    the thing a reviewer actually needs -- that the service measures its own behaviour and can
    report it -- without adding a database and a scrape target to a system whose deployment
    story is one container.

    Declared above the StaticFiles mount, and that is not a stylistic choice: routes match in
    registration order and the mount answers "/", so anything declared after it is
    unreachable. This endpoint returned 404 until it was moved here.
    """
    paths = {k: {"requests": v["n"],
                 "errors": v["errors"],
                 "mean_ms": round(v["total_ms"] / v["n"], 1) if v["n"] else 0.0,
                 "max_ms": round(v["max_ms"], 1)}
             for k, v in sorted(_METRICS["by_path"].items())}
    return JSONResponse({
        "uptime_seconds": round(time.time() - _STARTED, 1),
        "requests": _METRICS["requests"],
        "server_errors": _METRICS["errors"],
        "client_errors": _METRICS["client_errors"],
        "by_route": paths,
    }, headers=_NO_STORE)


# ═════════════════════════════════════════════════════════════════════ accounts and access
#
# A doctor may create their own account. That is a deliberate choice with a real cost, stated
# plainly: an open sign-up form means anyone who reaches this URL can create an account and
# begin storing patient data under a name of their own choosing. It is the right behaviour for
# a system being demonstrated and the wrong one for a hospital deployment, and which of those
# an instance is cannot be settled in this file -- so POCUS_OPEN_REGISTRATION=0 closes it, and
# `python -m src.auth.accounts` continues to issue accounts from the command line.
#
# What registration does NOT do is widen anyone's reach. A new account starts empty and every
# query in store.py takes a doctor id it cannot be called without, so adding a person to the
# system adds nothing to what any person can see.


class Credentials(BaseModel):
    username: str
    password: str


class Registration(BaseModel):
    username: str
    password: str
    name: str
    email: str | None = None


class NewPatient(BaseModel):
    name: str
    reference: str | None = None
    dateOfBirth: str | None = None
    sex: str | None = None


@app.post("/api/login")
def api_login(c: Credentials, response: Response) -> JSONResponse:
    token = sessions.login(c.username, c.password)
    if token is None:
        # One message for both failures. "No such user" and "wrong password" told apart turn
        # this form into a way to enumerate who works here.
        return JSONResponse({"error": "username or password is incorrect"}, status_code=401)
    doctor = sessions.doctor_for(token)
    assert doctor is not None
    out = JSONResponse({"doctor": {"name": doctor.name, "username": doctor.username}})
    # httponly     script cannot read it, so an injected script cannot exfiltrate the session
    # samesite     the cookie does not ride along on a request another site caused
    # secure       only when actually behind TLS: setting it on plain http makes the cookie
    #              silently undeliverable, which presents as "login does nothing"
    out.set_cookie(sessions.COOKIE, token, httponly=True, samesite="lax",
                   secure=os.environ.get("POCUS_HTTPS") == "1",
                   max_age=sessions.SESSION_HOURS * 3600, path="/")
    return out


@app.post("/api/register")
def api_register(r: Registration, response: Response) -> JSONResponse:
    """Create an account and sign it in.

    A new doctor starts with nothing: no patients, no examinations, and no way to reach
    anyone else's. That is not a property of this endpoint but of every query in store.py,
    each of which takes a doctor id it cannot be called without -- so registration adds a
    person to the system without widening what any person can see.
    """
    token, error = sessions.register(r.username, r.name, r.password, r.email)
    if token is None:
        return JSONResponse({"error": error}, status_code=400)
    doctor = sessions.doctor_for(token)
    assert doctor is not None
    out = JSONResponse({"doctor": {"name": doctor.name, "username": doctor.username}})
    out.set_cookie(sessions.COOKIE, token, httponly=True, samesite="lax",
                   secure=os.environ.get("POCUS_HTTPS") == "1",
                   max_age=sessions.SESSION_HOURS * 3600, path="/")
    return out


@app.post("/api/logout")
def api_logout(request: Request) -> JSONResponse:
    # The row is deleted, so the token is dead for everyone holding it -- not merely forgotten
    # by this browser. On a shared workstation that difference is the whole point.
    sessions.logout(request.cookies.get(sessions.COOKIE))
    out = JSONResponse({"ok": True})
    out.delete_cookie(sessions.COOKIE, path="/")
    return out


@app.get("/api/me")
def api_me(doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    return JSONResponse({"name": doctor.name, "username": doctor.username,
                         **store.counts(doctor.id)}, headers=_NO_STORE)


@app.get("/api/patients")
def api_patients(doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    return JSONResponse({"patients": store.list_patients(doctor.id)}, headers=_NO_STORE)


@app.post("/api/patients")
def api_patient_create(p: NewPatient,
                       doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    return JSONResponse({"patient": store.create_patient(
        doctor.id, name=p.name, reference=p.reference,
        date_of_birth=p.dateOfBirth, sex=p.sex)})


@app.get("/api/patients/{patient_id}/examinations")
def api_patient_history(patient_id: str,
                        doctor: Doctor = Depends(current_doctor)) -> JSONResponse:
    """One patient's history. 404 covers both "not yours" and "does not exist"."""
    patient = store.get_patient(doctor.id, patient_id)
    if patient is None:
        raise HTTPException(status_code=404, detail="no such patient")
    return JSONResponse({"patient": patient,
                         "examinations": store.list_examinations(doctor.id, patient_id)},
                        headers=_NO_STORE)


@app.get("/login")
def login_page() -> FileResponse:
    return FileResponse(WEB / "login.html", headers=_NO_STORE)


def _page(request: Request, filename: str) -> Response:
    """The application page, or the login page if nobody is signed in.

    A redirect rather than a 401, because this is a browser navigation: the person typing the
    address should land on a form, not on a JSON error. The API endpoints behind it still
    answer 401, since a fetch has no use for a login page.
    """
    if sessions.doctor_for(request.cookies.get(sessions.COOKIE)) is None:
        return RedirectResponse("/login", status_code=303)
    return FileResponse(WEB / filename, headers=_NO_STORE)


@app.get("/")
def index(request: Request) -> Response:
    return _page(request, "pocus-copilot.dc.html")


@app.get("/fr")
def index_fr(request: Request) -> Response:
    """The same interface, built in French at bind time rather than translated at runtime.

    A page per language: the language is settled before any script runs, and both pages are
    checked by the same balance and coverage checks. The clinical text inside it is rendered
    by src/agents/i18n.py from the same computed fields as the English, so a French screen
    cannot state something the English record does not.
    """
    return _page(request, "pocus-copilot.fr.dc.html")


app.mount("/", StaticFiles(directory=str(WEB)), name="web")


# The mount above serves everything under web/ by name, and the application page is a file in
# web/ -- so /pocus-copilot.dc.html reached the interface without passing the check on "/".
# The routes were gated and the file behind them was not, which is the oldest way there is to
# leave a door open. Gating the extension covers the built pages and the pristine export in
# web/design/ alike; login.html is not one of them and stays reachable.
#
# The page carries no patient data of its own -- every value on it arrives from /api/*, which
# answers 401 -- so this is defence in depth rather than the only lock. It is still worth
# having: an interface that renders and then fills with errors invites someone to conclude the
# application is broken rather than that they are not signed in.
@app.middleware("http")
async def _gate_static_pages(request, call_next):
    if request.url.path.endswith(".dc.html"):
        if sessions.doctor_for(request.cookies.get(sessions.COOKIE)) is None:
            return RedirectResponse("/login", status_code=303)
    return await call_next(request)


# The interface is ONE generated file -- the binder inlines the logic into the page -- so a
# cached copy is a cached application. Three times now a change was made, the server was
# restarted, and the browser kept running the previous build: uploads that no longer filed
# their images, counters reading the old bindings, an empty screen that had already been
# fixed. Each looked like a fresh bug in the code that had just been written, and one of them
# was reported as one. Revalidation headers were not enough, because the page is re-issued on
# every bind and a browser is entitled to reuse a fresh-looking response without asking.
# It is a few hundred kilobytes off local disk; correctness is worth more than the round trip.
@app.middleware("http")
async def _no_cache(request, call_next):
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers.update(_NO_STORE)
    return response


# ═════════════════════════════════════════════════════════════════════════ monitoring
#
# What a running instance says about itself, and the one rule that shapes all of it: the log
# records that a request happened and how it went, never what it contained.
#
# That rule is not decoration. The bodies passing through here are a patient's presenting
# complaint, vital signs, laboratory values and ultrasound images. A log line is copied to
# disk, shipped to whatever collects it, and read by people who were never in the consultation
# -- so a logger that echoed request bodies would quietly turn every deployment into an
# unregulated second medical record. Method, path, status and duration answer the operational
# questions; nothing clinical is needed to answer them.
_LOG = logging.getLogger("pocus.access")

_METRICS: dict[str, Any] = {
    "requests": 0,
    "errors": 0,                                 # 5xx: the service failed, not the caller
    "client_errors": 0,                          # 4xx: the caller asked for something wrong
    "by_path": collections.defaultdict(
        lambda: {"n": 0, "errors": 0, "total_ms": 0.0, "max_ms": 0.0}),
}


@app.middleware("http")
async def _observe(request, call_next):
    started = time.perf_counter()
    status = 500                                 # if the handler raises, that is what happened
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        ms = (time.perf_counter() - started) * 1000.0

        # Keyed by ROUTE, not by URL. Recording the literal path would make every encounter
        # identifier its own metric, so the series grows without bound and no two requests
        # ever aggregate -- and identifiers of encounters are exactly the thing not to
        # accumulate in a metrics table.
        route = request.scope.get("route")
        key = getattr(route, "path", None) or ("/static" if not
                                               request.url.path.startswith("/api/") else "/other")

        _METRICS["requests"] += 1
        if status >= 500:
            _METRICS["errors"] += 1
        elif status >= 400:
            _METRICS["client_errors"] += 1

        p = _METRICS["by_path"][key]
        p["n"] += 1
        p["total_ms"] += ms
        p["max_ms"] = max(p["max_ms"], ms)
        if status >= 500:
            p["errors"] += 1

        # WARNING for a server failure so it separates from ordinary traffic in a log search.
        _LOG.log(logging.WARNING if status >= 500 else logging.INFO,
                 "%s %s -> %d in %.1fms", request.method, key, status, ms)




if __name__ == "__main__":
    import uvicorn

    # Configured here rather than at import, so that importing serve.py -- which every API test
    # does -- does not reconfigure the root logger of whatever imported it. A library that
    # rearranges logging on import is a library that makes another program's diagnostics
    # disappear.
    #
    # Plain stream to stdout: in a container that IS the log, collected by the runtime. Writing
    # to a file inside a container would put the diagnostics on the one filesystem that is
    # discarded when the container stops.
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)-7s %(name)s  %(message)s",
        stream=sys.stdout,
    )

    port = int(os.environ.get("PORT", 8501))

    # Loopback by default, and the default is the safe one: a development server on a laptop
    # should not be reachable from the rest of the network, least of all one that holds patient
    # data entered during a demonstration.
    #
    # Inside a container that default is wrong rather than merely cautious. 127.0.0.1 there is
    # the *container's* loopback, so the process answers itself and nothing else: `docker run
    # -p 8600:8501` publishes a port that connects to a socket no one is listening on, and the
    # container looks healthy from within while being unreachable from outside. The Dockerfile
    # sets HOST=0.0.0.0 explicitly, which keeps the choice visible in the image rather than
    # hidden in a default that differs by environment.
    host = os.environ.get("HOST", "127.0.0.1")
    # Plain ASCII: the Windows console defaults to cp1252 and an arrow here aborts startup.
    print(f"POCUS-Emergency running at http://localhost:{port}  (bound to {host})")
    uvicorn.run(app, host=host, port=port, log_level="warning")
