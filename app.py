"""POCUS-Emergency — Clinical Copilot.

    streamlit run app.py

Implements the "POCUS Copilot" design (claude.ai/design project 80d3b96f) — its palette,
typography, dark navy chrome, lime accent, motion system and all twelve screens — over the real
pipeline. The visual design is taken as given. What is bound to it is real:

    LIVE      lung inference on the fold checkpoints in this repository, on CPU. The clinician
              drops in a scan and the module reports its own findings; nothing about what the
              image shows is typed in.
    LIVE      clinical state, conflicts, absences, evidence identifiers, escalation, severity,
              alerts, recommended examinations, scenario routing, retrieval, timings, report.
    LIVE      the failure demonstration — a backend that genuinely raises.
    RECORDED  the differential, and only for an unmodified benchmark encounter. Generating one
              needs a 4.9 GB model on a GPU, so an edited or custom encounter says so rather
              than showing a recording that no longer matches the record on screen.
    ABSENT    cardiac and gallbladder weights. Those organs return `not_supported` in the
              ordinary report format rather than a guess.

Three places where the design's placeholder CONTENT could not be reproduced literally, because
copying it would have meant fabricating clinical material rather than styling it:

  * The roster ("Sarah Martin", "Patient B/C/D", "12 assessments today", "7,198 conversations",
    "23 suggestions · 8 anomalies") is mock data. The same tables, cards and counters are
    rendered in the same style from the five benchmark encounters and whatever has actually
    been analysed this session.
  * The assistant screen ships a dictionary of pre-written replies. A chat that answers a typed
    clinical question with a canned paragraph is a fabricated capability, and this system has no
    conversational model. The screen is built as designed; every answer is assembled from the
    computed assessment, and a free-typed question is told plainly what can be answered from.
  * The patient record implies a database — an MRN, four visits, seven stored studies,
    measurements trended across dates. This system has no persistence whatsoever. The screen is
    built as designed and scoped to the current session, and says so at the top: what it shows
    is what this run of the application has actually done.

Everything else — colours, radii, gradients, the navy chrome, the lime accent, the motion, the
twelve screens, the wizard, the nav — is the design as delivered.
"""
from __future__ import annotations

import html
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agents import schema as S  # noqa: E402
from src.agents.clinical.clinical_state import (  # noqa: E402
    LAB_REFERENCE, VITAL_REFERENCE, build_clinical_state, build_evidence)
from src.agents.clinical.decision_support import decision_support  # noqa: E402
from src.agents.clinical.llm import FailingBackend  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision, reason  # noqa: E402
from src.agents.clinical.report import build_report, render_report  # noqa: E402
from src.agents.clinical.retrieval import Retriever  # noqa: E402
from src.agents.clinical.run_case import SCENARIOS, build as build_scenario  # noqa: E402
from src.agents.ultrasound.agent import (  # noqa: E402
    GB_CLASSES, GB_GROUP, GB_LOW_CONFIDENCE, GB_SCOPE, LUNG_FINDINGS)
from src.agents.ultrasound.agent import available as organs_available  # noqa: E402
from src.agents.ultrasound.agent import module_status  # noqa: E402
from src.agents.ultrasound.agent import ultrasound_agent  # noqa: E402

FROZEN = ROOT / "models" / "clinical_reasoning_v4_final" / "results.json"

st.set_page_config(page_title="POCUS-Emergency · Clinical Copilot", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

# ══════════════════════════════════════════════════════ design tokens (as delivered)
BG, CARD, TINT, TINT2 = "#EFEEFB", "#FFFFFF", "#FCFBFF", "#FAFAFE"
INK, MUTED, FAINT, GHOST = "#1B1A3A", "#6A6785", "#8A87A8", "#9C99B8"
BORDER, BORDER2, HOVER, RULE = "#E4E2F8", "#DEDCF4", "#B9B5EC", "#F1EDFF"
NAVY, IRIS = "#2E2A78", "#5B54D6"
V700, V500, V300 = "#5B3CC4", "#7C5CFC", "#8E88E8"
LIME, LIME_D, LIME_L = "#C6F24E", "#5A7A0F", "#F0FADB"
LIME_M, LIME_XL, LIME_XXL = "#D6F58C", "#E8F9BE", "#F2FCDD"
RED, RED_D, RED_L = "#E5484D", "#C13238", "#FDECEC"
AMB, AMB_D, AMB_L = "#F5A623", "#9A6207", "#FFF3E0"
GRN = "#22A06B"
HERO = f"linear-gradient(135deg,{V700} 0%,{V500} 55%,{V300} 100%)"
SHADOW = "0 6px 20px rgba(91,60,196,.05)"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

html,body,.stApp,[class*="css"]{{background:{BG}!important;color:{INK}!important;
  font-family:'Plus Jakarta Sans',system-ui,sans-serif!important;-webkit-font-smoothing:antialiased}}
*{{box-sizing:border-box}} ::selection{{background:#DEDCF4}}
h1,h2,h3,h4,h5,p,div,span,label,li,td,th{{font-family:'Plus Jakarta Sans',system-ui,sans-serif!important}}
/* Streamlit chrome. Only the toolbar is hidden; the HEADER ITSELF keeps its natural height,
   because `stExpandSidebarButton` -- the control that reopens a collapsed sidebar -- lives
   inside it. Hiding the header removed the only way back, and zeroing its height clipped the
   button to nothing, which looked identical. Collapse the sidebar once and the navigation was
   unreachable without a reload. The button is given the design's own styling so it reads as
   part of the page rather than as leftover framework chrome. */
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{{display:none!important}}
header[data-testid="stHeader"]{{background:transparent!important;box-shadow:none!important}}
[data-testid="stExpandSidebarButton"]{{
  display:flex!important;visibility:visible!important;opacity:1!important;z-index:1000!important}}
[data-testid="stExpandSidebarButton"] button,
[data-testid="stSidebarCollapseButton"] button{{
  background:{CARD}!important;border:1px solid {BORDER}!important;border-radius:10px!important;
  color:{NAVY}!important;box-shadow:0 4px 14px rgba(46,42,120,.16)!important;
  width:38px!important;height:38px!important}}
.block-container{{padding:8px 40px 72px!important;max-width:1500px}}

/* ── motion (as delivered) ───────────────────────────────────────────── */
@keyframes riseIn{{from{{opacity:0;transform:translateY(14px)}}to{{opacity:1;transform:none}}}}
@keyframes growUp{{from{{transform:scaleY(0)}}to{{transform:scaleY(1)}}}}
@keyframes pulseDot{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.45;transform:scale(.82)}}}}
@keyframes softPulse{{0%,100%{{box-shadow:0 0 0 0 rgba(229,72,77,.34)}}70%{{box-shadow:0 0 0 12px rgba(229,72,77,0)}}}}
@keyframes drift{{0%,100%{{transform:translateY(0)}}50%{{transform:translateY(-8px)}}}}
.rise{{animation:riseIn .5s cubic-bezier(.2,.7,.3,1) both}}
.rise-2{{animation:riseIn .5s .07s cubic-bezier(.2,.7,.3,1) both}}
.rise-3{{animation:riseIn .5s .14s cubic-bezier(.2,.7,.3,1) both}}
.rise-4{{animation:riseIn .5s .21s cubic-bezier(.2,.7,.3,1) both}}
.gbar{{transform-origin:bottom;animation:growUp .7s cubic-bezier(.2,.8,.25,1) both}}
.livedot{{animation:pulseDot 1.7s ease-in-out infinite;display:inline-block}}
.alertpulse{{animation:softPulse 2.2s ease-out infinite}}
.drift{{animation:drift 5.5s ease-in-out infinite}}
.lift{{transition:transform .2s cubic-bezier(.2,.7,.3,1),box-shadow .2s ease}}
.lift:hover{{transform:translateY(-3px);box-shadow:0 12px 28px rgba(46,42,120,.13)}}
@media (prefers-reduced-motion:reduce){{*{{animation:none!important;transition:none!important}}}}

/* ── sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"]{{background:{CARD}!important;border-right:1px solid {BORDER};
  width:272px!important;min-width:272px!important}}
section[data-testid="stSidebar"] > div{{padding:22px 16px}}
section[data-testid="stSidebar"] .stButton>button{{
  width:100%;text-align:left;justify-content:flex-start;appearance:none;border:0;
  background:transparent;color:{MUTED};font-weight:500;font-size:14.5px;
  padding:11px 14px;border-radius:12px;box-shadow:none;transition:background .12s}}
section[data-testid="stSidebar"] .stButton>button:hover{{background:{BG};color:{NAVY}}}
section[data-testid="stSidebar"] .stButton>button[kind="primary"]{{
  background:{BORDER};color:{NAVY};font-weight:700;box-shadow:inset 3px 0 0 {IRIS}}}
section[data-testid="stSidebar"] .stButton>button p{{font-size:14.5px!important;margin:0}}

/* ── buttons ─────────────────────────────────────────────────────────── */
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{{
  border-radius:12px;font-weight:700;font-size:14.5px;padding:11px 20px;
  border:1px solid {BORDER2};background:{CARD};color:{MUTED};box-shadow:none}}
.stButton>button:hover,.stDownloadButton>button:hover{{border-color:{HOVER};color:{V700}}}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"],
.stDownloadButton>button[kind="primary"]{{
  background:{IRIS}!important;border:0!important;color:#fff!important}}
.stButton>button[kind="primary"]:hover{{filter:brightness(1.08)}}

/* ── inputs ──────────────────────────────────────────────────────────── */
.stTextInput input,.stNumberInput input,.stTextArea textarea,
.stSelectbox div[data-baseweb="select"]>div{{
  border:1px solid {BORDER2}!important;border-radius:11px!important;background:{TINT}!important;
  font-size:14.5px!important;color:{INK}!important}}
.stTextInput input:focus,.stNumberInput input:focus,.stTextArea textarea:focus{{
  border-color:{HOVER}!important;box-shadow:0 0 0 3px rgba(91,84,214,.12)!important}}
.stTextInput label,.stNumberInput label,.stTextArea label,.stSelectbox label,
.stFileUploader label,.stCheckbox label,.stSlider label,.stRadio label{{
  font-size:13px!important;font-weight:600!important;color:{MUTED}!important}}
.stCheckbox input{{accent-color:{V700}}}
[data-testid="stFileUploaderDropzone"]{{background:{TINT2};border:1px dashed #DCD4F7;
  border-radius:18px;padding:22px}}
.stSlider [data-baseweb="slider"] div[role="slider"]{{background:{IRIS}!important}}
div[data-testid="stExpander"]{{border:1px solid {BORDER}!important;border-radius:18px!important;
  background:{CARD}!important;box-shadow:{SHADOW}}}
hr{{border-color:{RULE}!important;margin:6px 0!important}}

/* ── design components ───────────────────────────────────────────────── */
.hdr{{display:flex;align-items:center;justify-content:space-between;gap:24px;
  padding:20px 30px;background:{NAVY};border-radius:16px;margin-bottom:24px}}
.search{{flex:1;max-width:420px;display:flex;align-items:center;justify-content:space-between;
  gap:10px;background:rgba(255,255,255,.13);border:1px solid rgba(255,255,255,.22);
  border-radius:999px;padding:11px 18px;color:#C9C6EE;font-size:14px}}
.pill-live{{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,.14);
  color:{LIME};border-radius:999px;padding:7px 13px;font-size:13px;font-weight:700}}
.pill-lime{{background:{LIME};color:{INK};border-radius:999px;padding:9px 18px;
  font-size:14px;font-weight:700}}
.hero{{background:{HERO};border-radius:24px;padding:38px 40px;color:#fff;display:flex;
  justify-content:space-between;gap:40px;flex-wrap:wrap;
  box-shadow:0 18px 44px rgba(91,60,196,.28);margin-bottom:26px}}
.hero h1{{margin:12px 0 10px;font-size:36px;line-height:1.12;letter-spacing:-.02em;
  font-weight:800;color:#fff!important}}
.eyebrow{{font-size:13px;letter-spacing:.1em;text-transform:uppercase;opacity:.82;font-weight:700}}
.stat{{background:rgba(255,255,255,.14);border-radius:16px;padding:16px 18px}}
.stat b{{font-size:28px;font-weight:800;display:block}}
.stat span{{font-size:12.5px;opacity:.88}}
.card{{background:{CARD};border:1px solid {BORDER};border-radius:20px;padding:26px 28px;
  box-shadow:{SHADOW};margin-bottom:20px}}
.card10{{background:{CARD};border:1px solid {BORDER};border-radius:10px;padding:26px 30px;
  margin-bottom:20px}}
.kicker{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:{V500};font-weight:800}}
.step-eyebrow{{font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;color:{V500};
  font-weight:700}}
.h1{{margin:10px 0 8px;font-size:32px;font-weight:800;letter-spacing:-.02em}}
.h2{{margin:0 0 4px;font-size:17px;font-weight:800}}
.lede{{margin:0;color:{MUTED};font-size:15.5px;max-width:70ch}}
.sub{{margin:0 0 18px;font-size:13.5px;color:{MUTED}}}
.tag{{border-radius:999px;padding:5px 12px;font-size:12.5px;font-weight:700;
  display:inline-block;margin:0 6px 6px 0}}
.t-red{{background:{RED_L};color:{RED_D}}} .t-amb{{background:{AMB_L};color:{AMB_D}}}
.t-lime{{background:{LIME_L};color:{LIME_D}}} .t-vio{{background:{BORDER};color:{NAVY}}}
.t-lav{{background:{BG};color:{V700}}}
.vt{{background:{CARD};border:1px solid {BORDER};border-radius:18px;padding:20px 22px;
  box-shadow:{SHADOW}}}
.vt .lbl{{font-size:13.5px;color:{MUTED};font-weight:600}}
.vt .val{{font-size:32px;font-weight:800;letter-spacing:-.02em}}
.vt .un{{font-size:14px;color:{FAINT}}}
.vt-none{{background:{TINT2};border:1px dashed #DCD4F7;border-radius:18px;padding:20px 22px}}
table.dt{{width:100%;border-collapse:collapse;font-size:14.5px}}
table.dt th{{text-align:left;color:{FAINT};font-size:12.5px;letter-spacing:.06em;
  text-transform:uppercase;padding:0 0 12px;font-weight:700}}
table.dt td{{padding:13px 0;border-top:1px solid {RULE}}}
.band{{border-radius:18px;padding:20px 24px;margin-bottom:20px}}
.band-red{{background:#FFF1F1;border:1px solid #F6CFCF;border-left:5px solid {RED}}}
.band-amb{{background:#FFF8EC;border:1px solid #F7DFB4;border-left:5px solid {AMB}}}
.band-vio{{background:{CARD};border:1px solid {BORDER2};border-left:5px solid {V500}}}
.band-lime{{background:{LIME_L};border:1px solid #D5E8A8;border-left:5px solid {LIME_D}}}
.row{{display:flex;justify-content:space-between;align-items:center;gap:14px;
  padding:15px 0;border-bottom:1px solid {RULE}}}
.row:last-child{{border-bottom:0}}
.note{{background:{BG};border-radius:14px;padding:14px 16px;font-size:13.5px;color:#463A6B}}
.dx{{border:1px solid {BORDER2};border-radius:16px;padding:20px 22px;background:{TINT};
  margin-bottom:16px}}
.dx-sec{{border:1px solid {BORDER};border-radius:16px;padding:20px 22px;margin-bottom:16px}}
.pbar{{height:8px;border-radius:6px;background:{BORDER};overflow:hidden;margin-top:14px}}
.pbar>i{{display:block;height:100%;background:{IRIS}}}
.bub-a{{max-width:78%;background:#F7F5FF;border:1px solid {BORDER};color:{INK};
  padding:14px 18px;border-radius:16px 16px 16px 4px;font-size:14.5px;line-height:1.6}}
.bub-d{{max-width:74%;background:{IRIS};color:#fff;padding:14px 18px;
  border-radius:16px 16px 4px 16px;font-size:14.5px;line-height:1.55}}
.tl{{display:grid;grid-template-columns:82px 22px 1fr;gap:14px;align-items:start;
  padding-bottom:22px}}
.scan{{background:#0E1220;border-radius:20px;min-height:230px;display:flex;
  flex-direction:column;justify-content:space-between;padding:18px 20px;color:#C7CBDA;
  font-size:12.5px}}
.rep{{background:{CARD};border:1px solid {BORDER};border-radius:22px;padding:44px 52px;
  box-shadow:0 10px 34px rgba(91,60,196,.09)}}
.grid4{{display:grid;grid-template-columns:repeat(auto-fit,minmax(212px,1fr));gap:18px;
  margin-bottom:22px}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:22px}}
.axis{{display:flex;flex-direction:column;justify-content:space-between;font-size:12.5px;
  color:{GHOST};font-weight:600;height:280px;text-align:right}}
.plot{{height:280px;display:flex;align-items:flex-end;gap:12px;border-left:1px solid {BORDER};
  border-bottom:1px solid {BORDER};padding:0 6px 0 14px}}
.dxbox{{border:3px solid {IRIS};padding:20px 14px;text-align:center;font-size:12.5px;
  font-weight:800;letter-spacing:.1em;text-transform:uppercase}}
.big{{font-size:32px;font-weight:800;letter-spacing:-.03em}}
</style>
""", unsafe_allow_html=True)

E = html.escape

# ══════════════════════════════════════════════════════════ benchmark encounters
BLANK = dict(name="", age=60, sex="F", complaint="", tier="medium", tconf=0.70,
             organ="Lung", findings={}, vitals={}, labs={}, key=None, unassessed=[],
             tag=("t-lav", "New"))

_VMAP = {"o2sat": "spo2", "pulse": "hr", "bpsys": "sbp", "respr": "rr", "temp": "temp"}
ORGANS_LOWER = ("lung", "heart", "gallbladder")
# The three perception modules the project trained, plus the honest fourth option. `schema`
# also admits "vascular" and "fast" as organs, and the design offers tabs for both, but no
# module was built for either — offering them would advertise a capability that does not exist.
ORGANS = ["Lung", "Heart", "Gallbladder", "Not performed"]


def _preset(key: str, title: str, tag: tuple[str, str]) -> dict:
    """A wizard preset derived from the canonical scenario rather than re-typed beside it.

    Hand-copying these was a mistake worth recording. The presets drifted from the scenarios
    they were named after — a different age, a sex, one missing blood pressure — and the
    encounter on screen was no longer the encounter the recorded differential had been
    generated from. Deriving them makes the drift unrepresentable.
    """
    sp = SCENARIOS[key]
    tri, us = sp["triage"], sp["ultrasound"]
    scanned = {o: r for o, r in us.items() if r["status"] == "ok"}
    organ = next((o.title() for o in ORGANS_LOWER if o in scanned), "Lung")
    rep = scanned.get(organ.lower(), {})

    def _k(label: str) -> str:
        u = label.replace(" ", "_")
        return u if u in LUNG_FINDINGS else label

    findings: dict[str, float] = {}
    for f in rep.get("findings", []):
        findings[_k(f["label"])] = f["confidence"]
    for f in rep.get("not_detected", []):
        findings[_k(f["label"])] = -f["confidence"]
    return dict(
        name=title, age=sp["clinical"]["age"], sex=sp["clinical"]["sex"],
        complaint=sp["clinical"]["chief_complaint"], tier=tri["urgency"],
        tconf=float(tri["confidence"]), organ=organ, findings=findings,
        vitals={_VMAP.get(k, k): float(v) for k, v in (tri.get("features") or {}).items()},
        labs=dict(sp["labs"]), key=key, tag=tag,
        unassessed=[o for o, r in us.items() if r["status"] != "ok"])


CASES = {
    "Blank — new patient": dict(BLANK),
    "Acute dyspnoea, key labs missing": _preset("missing", "Case A", ("t-red", "Critical")),
    "Evidence agrees, record complete": _preset("concordant", "Case B", ("t-red", "Critical")),
    "Triage and imaging disagree": _preset("conflict", "Case C", ("t-amb", "Review")),
    "Everything negative": _preset("reassuring", "Case D", ("t-amb", "Review")),
    "Organ never scanned": _preset("not_assessed", "Case E", ("t-red", "Critical")),
}

SCREENS = [("home", "Home", "⌂", ""), ("intake", "New assessment", "＋", ""),
           ("clinical", "Clinical data", "❤", ""), ("pocus", "POCUS", "◉", ""),
           ("diagnosis", "Diagnosis", "✳", ""), ("record", "Patient record", "▣", ""),
           ("assessment", "Assessment", "✦", "AI"), ("alerts", "Alerts", "⚠", ""),
           ("assistant", "Clinical assistant", "✧", ""), ("timeline", "Timeline", "◷", ""),
           ("report", "Report", "▤", ""), ("history", "History", "⟲", "")]


@st.cache_resource
def retriever() -> Retriever:
    return Retriever()


@st.cache_data
def frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}


@st.cache_data
def bench() -> dict:
    """Test totals read from the artefact the suite writes, never typed into the page.

    A hard-coded count is a claim about the repository that drifts the moment a test is added,
    and it drifted within a day of being written.
    """
    p = ROOT / "models" / "safety_benchmark.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


# ══════════════════════════════════════════════════════════════════ pipeline
def build_state(enc: dict):
    """The clinical state for an encounter.

    An untouched benchmark encounter is built by `run_case.build` — the same function the
    regression and the recorded GPU run used — so the differential shown beside it was
    genuinely computed from this record. Reconstructing an equivalent-looking bundle here would
    not be the same thing: the reliability text alone differs, and that text reaches the prompt.
    """
    if enc.get("frozen_key") and enc.get("report") is None:
        return build_scenario(enc["frozen_key"])

    reports, organ = {}, enc["organ"]
    if enc.get("report") is not None:
        reports[enc["report"]["organ"]] = enc["report"]
    elif organ == "Lung":
        det = [S.make_finding(k.replace("_", " "), v)
               for k, v in enc["findings"].items() if v > 0]
        neg = [S.make_finding(k.replace("_", " "), -v)
               for k, v in enc["findings"].items() if v <= 0]
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
        # Single-label, unlike the lung module: one class is reported, carrying the clinical
        # group the escalation policy reads severity from. There is no `not_detected` list —
        # the four classes that lost the argmax were not screened out, they simply were not the
        # winner, and reporting them as negatives would be a different claim.
        det = [S.make_finding(k, v, group=GB_GROUP.get(k, ""))
               for k, v in enc["findings"].items() if v > 0]
        reports["gallbladder"] = S.make_report(
            "gallbladder", det,
            quality={"low_confidence": bool(det and
                                            det[0]["confidence"] < GB_LOW_CONFIDENCE)},
            reliability={"confidence_calibrated": True, "has_normal_class": False,
                         "modelled_findings": GB_CLASSES, "scope": GB_SCOPE})

    for gap in enc.get("not_assessed", []):
        reports.setdefault(gap, S.make_report(
            gap, [], status="not_supported",
            reliability={"scope": "requested but never assessed"}))

    return build_clinical_state(
        {"encounter_id": enc.get("id", "ENC-LIVE"),
         "triage": S.make_triage(enc["tier"], enc["tconf"], features=enc["vitals"]),
         "ultrasound": reports,
         "clinical": {"age": enc["age"], "sex": enc["sex"],
                      "chief_complaint": enc["complaint"]}},
        labs=enc["labs"])


def analyse(enc: dict, break_model: bool = False):
    """Full pass, with each stage timed. The timings are what the Timeline screen shows."""
    marks, t0 = [], time.perf_counter()

    def mark(title, detail, hl=False):
        marks.append({"t": time.perf_counter() - t0, "title": title, "detail": detail,
                      "hl": hl})

    mark("Patient registered", f"{enc['age']} · {enc['sex']} · "
                               f"{enc['complaint'] or 'no complaint given'}")
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
         f"severity {support['severity']['severity']} · "
         f"{len(support['alerts'])} alert(s) · {support['scenario']['label']}", hl=True)

    hits = retriever().for_state(state)
    mark("Evidence retrieved", f"{len(hits)} passage(s) above the 0.10 relevance floor")

    if break_model:
        result = reason(state, llm_fn=FailingBackend("out of memory"), max_revisions=1)
        origin = "failed"
        mark("Model backend failed", "differential withheld — escalation unaffected", hl=True)
    else:
        rec = (frozen().get("none") or {}).get(enc.get("frozen_key") or "")
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
                origin=origin, report=report, marks=marks,
                started=datetime.now().strftime("%H:%M"))


@st.cache_data(show_spinner=False)
def case_summary(key: str) -> dict:
    enc = dict(CASES[key], frozen_key=CASES[key]["key"], report=None,
               not_assessed=list(CASES[key]["unassessed"]))
    st_ = build_state(enc)
    esc = escalation_decision(st_)
    sup = decision_support(st_, esc)
    return {"severity": sup["severity"]["severity"], "alerts": len(sup["alerts"]),
            "name": enc["name"], "age": enc["age"], "sex": enc["sex"],
            "complaint": enc["complaint"], "tag": CASES[key]["tag"], "label": key}


ROSTER = [case_summary(k) for k in CASES if CASES[k]["key"]]

# ══════════════════════════════════════════════════════════════════ state
st.session_state.setdefault("screen", "home")
st.session_state.setdefault("enc", None)
st.session_state.setdefault("records", [])
st.session_state.setdefault("rec_tab", "Images")
st.session_state.setdefault("msgs", [{"r": "a", "t": "I have the clinical information for "
                                                     "this encounter. What would you like to "
                                                     "know?"}])
st.session_state.setdefault("form", dict(BLANK, weight="", history="", meds="", allergies="",
                                         symptoms=[]))


def go(s: str) -> None:
    st.session_state["screen"] = s


def seed(key: str, value) -> None:
    """Re-seed a widget key Streamlit discarded because its screen was not rendered.

    Streamlit garbage-collects the session-state entry for any keyed widget that a run does not
    instantiate. Across a four-step wizard that is silent data loss: entering the patient on
    step 1 and walking to step 3 dropped the name, age, complaint and triage tier, and the
    encounter was submitted with the blank defaults. It surfaced as a benchmark encounter no
    longer matching its own preset, so its recorded differential was correctly refused — the
    guard worked, and it was guarding against a bug of mine.

    The durable copy lives in `form`, which is a plain dict and is never widget-backed.
    """
    if key not in st.session_state:
        st.session_state[key] = value


enc = st.session_state["enc"]
AV = organs_available()
MODULES = module_status()

# Run the pipeline BEFORE the sidebar is drawn, so the severity chip and the alert badge show
# this encounter's numbers rather than the previous rerun's. The failure toggle is therefore
# read from session state here and only rendered further down.
A = analyse(enc, st.session_state.get("broken", False)) if enc else None
if A:
    st.session_state["_sev"] = A["support"]["severity"]["severity"]
    st.session_state["_nalerts"] = len(A["support"]["alerts"])

# ══════════════════════════════════════════════════════════════════ sidebar
with st.sidebar:
    st.markdown(
        f"<div style='display:flex;align-items:center;gap:12px;padding:0 8px 18px'>"
        f"<div style='width:40px;height:40px;border-radius:12px;background:{IRIS};display:flex;"
        f"align-items:center;justify-content:center;color:#fff;font-weight:800;font-size:17px'>"
        f"P</div><div><div style='font-weight:800;font-size:15.5px;letter-spacing:-.01em'>"
        f"POCUS-Emergency</div><div style='font-size:12px;color:{FAINT}'>Clinical Copilot</div>"
        f"</div></div>", unsafe_allow_html=True)

    for sid, label, icon, badge in SCREENS:
        # The alerts badge carries the count from the last analysis rather than a fixed number,
        # so the nav cannot advertise alerts this encounter does not have.
        b = str(st.session_state.get("_nalerts", 0)) if sid == "alerts" and enc else badge
        b = "" if b == "0" else b
        st.button(f"{icon} {label}" + (f" · {b}" if b else ""),
                  key=f"nav_{sid}", use_container_width=True,
                  type="primary" if st.session_state["screen"] == sid else "secondary",
                  on_click=go, args=(sid,))

    if enc:
        sev = st.session_state.get("_sev", "—")
        cls = {"HIGH": ("t-red", "High priority"), "MODERATE": ("t-amb", "Moderate priority"),
               "LOW": ("t-lime", "Low priority")}.get(sev, ("t-lav", "Not analysed"))
        st.markdown(
            f"<div style='margin-top:22px;background:{BG};border-radius:16px;padding:16px 18px'>"
            f"<div style='font-size:12px;letter-spacing:.09em;text-transform:uppercase;"
            f"color:{V500};font-weight:700'>Current patient</div>"
            f"<div style='margin-top:8px;font-weight:700;font-size:15px'>"
            f"{E(enc['name'] or 'Unnamed patient')}</div>"
            f"<div style='font-size:13px;color:{MUTED}'>{enc['age']} · {E(enc['sex'])} · "
            f"{E(enc['complaint'] or '—')}</div>"
            f"<div style='margin-top:12px'><span class='tag {cls[0]}'>● {cls[1]}</span></div>"
            f"</div>", unsafe_allow_html=True)

    st.markdown(
        f"<div style='margin-top:18px;padding:0 8px'><div style='font-size:12px;"
        f"letter-spacing:.09em;text-transform:uppercase;color:{V500};font-weight:700'>"
        f"Perception modules</div>" +
        "".join(f"<div style='font-size:12.5px;color:{MUTED};margin-top:6px'>"
                f"{'●' if s['runs'] else '○'} {k} — {E(s['reason'])}</div>"
                for k, s in MODULES.items()) + "</div>", unsafe_allow_html=True)

    st.toggle("Simulate a model failure", key="broken",
              help="Runs the pipeline for real with a backend that raises on every call. "
                   "The escalation surviving is executed, not narrated.")
    st.caption("**Not a diagnostic device.** Synthetic cases, no clinical ground truth, never "
               "validated against patient outcomes.")

# ══════════════════════════════════════════════════════════════════ header
st.markdown(
    f"<div class='hdr'><div class='search'>"
    f"<span style='display:flex;gap:10px;align-items:center'><span>⌕</span>"
    f"<span>Search patients, findings, reports…</span></span>"
    f"<span style='color:#B4ACC8'>⚙</span></div>"
    f"<div style='display:flex;align-items:center;gap:14px'>"
    f"<div class='pill-live'><span class='livedot'>●</span> Assistant ready</div>"
    f"<span class='pill-lime'>＋ New assessment</span></div></div>",
    unsafe_allow_html=True)

SCREEN = st.session_state["screen"]


def need_encounter() -> bool:
    if enc:
        return False
    st.markdown("<div class='card'><div class='h2'>No encounter loaded</div>"
                "<p class='sub'>Open <b>New assessment</b>, enter a patient and analyse the "
                "case.</p></div>", unsafe_allow_html=True)
    st.button("＋ New assessment", type="primary", on_click=go, args=("intake",))
    return True


# ═══════════════════════════════════════════════════════════════ 1 · HOME
if SCREEN == "home":
    crit = sum(1 for r in ROSTER if r["severity"] == "HIGH")
    st.markdown(
        f"<div class='hero rise'><div style='max-width:560px'>"
        f"<div class='eyebrow'>Your clinical copilot</div>"
        f"<h1>POCUS-Emergency</h1>"
        f"<p style='margin:0;font-size:16px;line-height:1.55;opacity:.92'>Start a new clinical "
        f"assessment using clinical information, vital signs, laboratory results and POCUS "
        f"findings. The assistant highlights what needs attention and what information is "
        f"missing.</p></div>"
        f"<div style='display:grid;grid-template-columns:repeat(2,minmax(120px,1fr));gap:14px;"
        f"align-content:start;min-width:280px'>"
        f"<div class='stat'><b>{len(ROSTER)}</b><span>Benchmark encounters</span></div>"
        f"<div class='stat'><b>{len(st.session_state['records'])}</b>"
        f"<span>Analysed this session</span></div>"
        f"<div class='stat'><b>{crit}</b><span>High priority</span></div>"
        f"<div class='stat'><b>{bench().get('total_passed', '—')}</b>"
        f"<span>Safety tests passing</span></div>"
        f"</div></div>", unsafe_allow_html=True)

    q = st.columns(4, gap="medium")
    for c, (ic, ttl, txt, cta, dest) in zip(q, [
            ("🩺", "Start an assessment", "Enter a new patient and analyse the case.",
             "Start →", "intake"),
            ("🧠", "Review a case", "Continue the assessment for this encounter.",
             "Open case →", "assessment"),
            ("💬", "Ask the assistant", "Question the computed assessment.",
             "Open assistant →", "assistant"),
            ("📄", "Generate a report", "Create a standardised clinical summary.",
             "Reports →", "report")]):
        with c:
            st.markdown(
                f"<div class='card lift rise-3' style='margin-bottom:10px;min-height:186px'>"
                f"<span style='width:40px;height:40px;border-radius:12px;background:{BG};"
                f"display:flex;align-items:center;justify-content:center;font-size:19px'>{ic}"
                f"</span><div style='font-weight:700;font-size:16.5px;margin-top:12px'>{ttl}"
                f"</div><div style='font-size:13.5px;color:{MUTED};margin-top:6px'>{txt}</div>"
                f"</div>", unsafe_allow_html=True)
            st.button(cta, key=f"qa_{dest}", use_container_width=True, on_click=go,
                      args=(dest,))

    rows = "".join(
        f"<tr><td style='font-weight:600'>{E(r['name'])}</td><td>{r['age']}</td>"
        f"<td>{E(r['complaint'])}</td>"
        f"<td><span class='tag {r['tag'][0]}'>{r['tag'][1]}</span></td>"
        f"<td style='text-align:right;color:{MUTED}'>{r['alerts']} alert(s)</td></tr>"
        for r in ROSTER)
    st.markdown(
        f"<div class='card rise-4'><h2 class='h2' style='font-size:19px'>Benchmark "
        f"encounters</h2>"
        f"<p class='sub'>Synthetic encounters with no clinical ground truth. Severity and alert "
        f"counts are computed live by the deterministic layer.</p>"
        f"<table class='dt'><thead><tr><th>Patient</th><th>Age</th><th>Reason</th>"
        f"<th>Status</th><th style='text-align:right'>Alerts</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>", unsafe_allow_html=True)
    st.button("View all →", on_click=go, args=("history",))

# ═══════════════════════════════════════════════════════════════ 2 · INTAKE
elif SCREEN == "intake":
    st.markdown("<div class='step-eyebrow'>Step 1 of 4</div>"
                "<h1 class='h1'>New clinical assessment</h1>"
                "<p class='lede'>Only age, sex, chief complaint and the vital signs you have "
                "are required. Anything left blank is recorded as not provided — never as "
                "normal.</p>", unsafe_allow_html=True)
    st.write("")

    preset = st.selectbox("Start from a benchmark encounter", list(CASES), key="_preset_sel")
    P = CASES[preset]

    if st.session_state.get("_preset") != preset:
        for f in LUNG_FINDINGS:
            st.session_state[f"f_{f}"] = f in P["findings"]
            st.session_state[f"c_{f}"] = float(P["findings"].get(f, 0.05))
        st.session_state["f_sd"] = "severe dysfunction" in P["findings"]
        st.session_state["c_sd"] = float(P["findings"].get("severe dysfunction", 0.20))
        for k in VITAL_REFERENCE:
            st.session_state[f"v_{k}"] = k in P["vitals"]
            st.session_state[f"vv_{k}"] = float(P["vitals"].get(
                k, (VITAL_REFERENCE[k]["normal_min"] + VITAL_REFERENCE[k]["normal_max"]) / 2))
        for k, ref in LAB_REFERENCE.items():
            st.session_state[f"l_{k}"] = k in P["labs"]
            st.session_state[f"ll_{k}"] = float(P["labs"].get(k, ref.get("normal_max", 1)))
        st.session_state["_name"] = P["name"]
        st.session_state["_age"] = int(P["age"])
        st.session_state["_sex"] = P["sex"]
        st.session_state["_complaint"] = P["complaint"]
        st.session_state["_tier"] = P["tier"]
        st.session_state["_tconf"] = float(P["tconf"])
        st.session_state["_organ"] = P["organ"]
        st.session_state["_preset"] = preset
        st.session_state["form"] = dict(
            P, weight="", history="", meds="", allergies="", symptoms=[],
            vitals=dict(P["vitals"]), labs=dict(P["labs"]), findings=dict(P["findings"]))
        st.rerun()

    F = st.session_state["form"]
    for k, v in (("_name", F["name"]), ("_age", int(F["age"])), ("_sex", F["sex"]),
                 ("_complaint", F["complaint"]), ("_tier", F["tier"]),
                 ("_tconf", float(F["tconf"])), ("_organ", F["organ"]),
                 ("_weight", F["weight"]), ("_hist", F["history"]), ("_meds", F["meds"]),
                 ("_allerg", F["allergies"])):
        seed(k, v)

    st.markdown("<div class='card'><h2 class='h2'>Patient</h2></div>", unsafe_allow_html=True)
    c = st.columns(4, gap="medium")
    name = c[0].text_input("Patient name", key="_name", placeholder="Optional")
    age = c[1].number_input("Age *", 0, 120, key="_age")
    # Stored as the canonical single letter the clinical state uses, displayed in full, so an
    # untouched benchmark encounter stays identical to the record it was recorded from.
    sex = c[2].selectbox("Sex *", ["F", "M"], key="_sex",
                         format_func=lambda s: {"F": "Female", "M": "Male"}[s])
    c[3].text_input("Weight", key="_weight", placeholder="kg")
    complaint = st.text_input("Chief complaint *", key="_complaint",
                              placeholder="e.g. acute breathlessness")
    st.markdown("".join(f"<span class='tag t-lav'>{t}</span>" for t in
                        ["Chest pain", "Dyspnea", "Shock", "Trauma", "Altered consciousness"]),
                unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='card'><h2 class='h2'>Clinical context</h2></div>",
                unsafe_allow_html=True)
    cc = st.columns(2, gap="large")
    syms = []
    with cc[0]:
        for s_ in ["Dyspnea", "Fatigue", "Fever", "Chest pain", "Cough"]:
            seed(f"sym_{s_}", s_ in F["symptoms"])
            if st.checkbox(s_, key=f"sym_{s_}"):
                syms.append(s_)
    with cc[1]:
        hist = st.text_area("Medical history", key="_hist",
                            placeholder="Hypertension, prior heart failure…", height=78)
        meds = st.text_input("Current medications", key="_meds",
                             placeholder="Add medications…")
        allg = st.text_input("Known allergies", key="_allerg", placeholder="Add allergies…")

    st.write("")
    st.markdown("<div class='card'><h2 class='h2'>Triage</h2></div>", unsafe_allow_html=True)
    tc = st.columns(2, gap="large")
    tier = tc[0].select_slider("Assessed urgency", ["low", "medium", "high"], key="_tier")
    tconf = tc[1].slider("Triage confidence", 0.5, 1.0, step=0.01, key="_tconf")

    F.update(name=name, age=int(age), sex=sex, complaint=complaint, tier=tier,
             tconf=float(tconf), weight=st.session_state.get("_weight", ""),
             history=hist, meds=meds, allergies=allg, symptoms=syms)

    st.write("")
    nav = st.columns([1, 1, 6])
    nav[0].button("← Home", on_click=go, args=("home",), use_container_width=True)
    nav[1].button("Continue →", type="primary", on_click=go, args=("clinical",),
                  use_container_width=True)

# ═══════════════════════════════════════════════════════════ 3 · CLINICAL
elif SCREEN == "clinical":
    F = st.session_state["form"]
    st.markdown(f"<div class='step-eyebrow'>Step 2 of 4</div>"
                f"<h1 class='h1'>Clinical measurements</h1>"
                f"<p class='lede'>{E(F['name'] or 'Unnamed patient')} · {F['age']} · "
                f"{E(F['sex'])} · {E(F['complaint'] or 'no complaint given')}</p>",
                unsafe_allow_html=True)
    st.write("")

    for k, ref in VITAL_REFERENCE.items():
        seed(f"v_{k}", k in F["vitals"])
        seed(f"vv_{k}", float(F["vitals"].get(k, ref["normal_max"])))
    for k, ref in LAB_REFERENCE.items():
        seed(f"l_{k}", k in F["labs"])
        seed(f"ll_{k}", float(F["labs"].get(k, ref.get("normal_max", 1))))

    # The design puts the vital cards above the inputs. The inputs execute first so the cards
    # read this run's values, then are drawn into a slot reserved above them.
    cards_slot = st.container()
    with st.expander("Record vital signs",
                     expanded=not any(st.session_state[f"v_{k}"] for k in VITAL_REFERENCE)):
        for k, ref in VITAL_REFERENCE.items():
            a, b = st.columns([1, 2])
            a.checkbox(f"{k} ({ref['unit']})", key=f"v_{k}")
            b.number_input(" ", key=f"vv_{k}", label_visibility="collapsed", step=1.0)

    cards, vitals = [], {}
    for k, ref in VITAL_REFERENCE.items():
        on, val = st.session_state[f"v_{k}"], st.session_state[f"vv_{k}"]
        if on:
            vitals[k] = val
            lo, hi = ref["normal_min"], ref["normal_max"]
            flag, tone, top = ("Normal", "t-lime", GRN) if lo <= val <= hi else (
                ("High", "t-red", RED) if val > hi else ("Low", "t-red", RED))
            if k == "rr" and val > hi:
                flag, tone, top = "High", "t-amb", AMB
            cards.append(
                f"<div class='vt' style='border-top:4px solid {top}'>"
                f"<div class='lbl'>{k}</div>"
                f"<div style='display:flex;align-items:baseline;gap:7px;margin-top:8px'>"
                f"<span class='val'>{val:g}</span><span class='un'>{ref['unit']}</span></div>"
                f"<div style='margin-top:10px'><span class='tag {tone}'>{flag}</span></div>"
                f"</div>")
        else:
            cards.append(
                f"<div class='vt-none'><div class='lbl'>{k}</div>"
                f"<div style='margin-top:8px;font-size:20px;font-weight:700;color:{GHOST}'>"
                f"Not measured</div><div style='margin-top:10px;font-size:12.5px;"
                f"color:{FAINT}'>Not the same as normal</div></div>")
    with cards_slot:
        st.markdown(f"<div class='grid4 rise'>{''.join(cards)}</div>", unsafe_allow_html=True)

    labs_slot = st.container()
    with st.expander("Record laboratory results",
                     expanded=not any(st.session_state[f"l_{k}"] for k in LAB_REFERENCE)):
        for k, ref in LAB_REFERENCE.items():
            a, b = st.columns([1, 2])
            a.checkbox(f"{k} ({ref['unit']})" if ref["unit"] else k, key=f"l_{k}")
            b.number_input(" ", key=f"ll_{k}", label_visibility="collapsed", step=1.0)

    lab_rows, labs = [], {}
    for k, ref in LAB_REFERENCE.items():
        on, val = st.session_state[f"l_{k}"], st.session_state[f"ll_{k}"]
        ref_txt = (f"&lt;{ref['normal_max']:g}" if "normal_min" not in ref
                   else f"{ref['normal_min']:g}–{ref['normal_max']:g}")
        if on:
            labs[k] = val
            hi = val > ref.get("normal_max", 1e9)
            lo = val < ref.get("normal_min", -1e9)
            tone, txt = ("t-red", "High") if hi else (("t-red", "Low") if lo
                                                      else ("t-lime", "Normal"))
            lab_rows.append(
                f"<tr><td style='font-weight:600'>{k}</td><td>{val:g} {ref['unit']}</td>"
                f"<td style='color:{FAINT}'>{ref_txt}</td>"
                f"<td style='text-align:right'><span class='tag {tone}'>{txt}</span></td></tr>")
        else:
            lab_rows.append(
                f"<tr><td style='font-weight:600;color:{GHOST}'>{k}</td>"
                f"<td style='color:{GHOST}'>—</td><td style='color:{GHOST}'>—</td>"
                f"<td style='text-align:right'><span class='tag t-lav'>Not measured</span>"
                f"</td></tr>")

    nmiss = sum(1 for k in LAB_REFERENCE if not st.session_state[f"l_{k}"])
    warn = (f"<div class='band band-amb' style='margin-top:20px'>"
            f"<div style='display:flex;gap:12px'><span style='font-size:16px'>⚠</span>"
            f"<div style='font-size:14px;color:#7A5A16'><strong style='color:#6A4C0C'>"
            f"{nmiss} investigation(s) have not been performed.</strong> An unmeasured test "
            f"does not mean a normal result.</div></div></div>") if nmiss else ""
    with labs_slot:
        st.markdown(
            f"<div class='card rise-2'><h2 class='h2'>Laboratory results</h2>"
            f"<table class='dt' style='margin-top:14px'><thead><tr><th>Test</th>"
            f"<th>Result</th><th>Reference</th><th style='text-align:right'>Status</th>"
            f"</tr></thead><tbody>{''.join(lab_rows)}</tbody></table>{warn}</div>",
            unsafe_allow_html=True)

    F["vitals"], F["labs"] = vitals, labs
    st.write("")
    nav = st.columns([1, 1, 6])
    nav[0].button("← Back", on_click=go, args=("intake",), use_container_width=True)
    nav[1].button("Continue →", type="primary", on_click=go, args=("pocus",),
                  use_container_width=True)

# ═══════════════════════════════════════════════════════════════ 4 · POCUS
elif SCREEN == "pocus":
    F = st.session_state["form"]
    seed("_organ", F["organ"])
    for f in LUNG_FINDINGS:
        seed(f"f_{f}", F["findings"].get(f, 0) > 0)
        seed(f"c_{f}", abs(float(F["findings"].get(f, 0.05))) or 0.05)
    seed("f_sd", F["findings"].get("severe dysfunction", 0) > 0)
    seed("c_sd", abs(float(F["findings"].get("severe dysfunction", 0.20))) or 0.20)
    _gb = next((k for k, v in F["findings"].items() if k in GB_CLASSES and v > 0), None)
    seed("gb_cls", _gb or GB_CLASSES[0])
    seed("gb_conf", float(F["findings"].get(_gb, 0.45)) if _gb else 0.45)

    st.markdown("<div class='step-eyebrow'>Step 3 of 4</div>"
                "<h1 class='h1'>POCUS examination</h1>", unsafe_allow_html=True)
    organ = st.session_state["_organ"]
    st.markdown("".join(
        f"<span class='tag' style='background:{V700};color:#fff;padding:8px 16px;"
        f"font-size:13.5px'>{o}</span>" if o == organ else
        f"<span class='tag' style='background:#fff;border:1px solid {BORDER2};color:{MUTED};"
        f"padding:8px 16px;font-size:13.5px;font-weight:600'>{o}</span>"
        for o in ORGANS), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2, gap="large")
    analysed = None

    with left:
        organ = st.selectbox("Organ examined", ORGANS, key="_organ")
        IMG_T = ["png", "jpg", "jpeg", "bmp"]
        key = organ.lower()
        runs = MODULES.get(key, {}).get("runs", False)

        def _grey(f):
            from PIL import Image
            import numpy as np
            f.seek(0)
            return np.array(Image.open(f).convert("L"))

        def _run(cache_key: str, fn):
            """Analyse once per uploaded file, not once per rerun."""
            if st.session_state.get("_img") != cache_key:
                t0 = time.time()
                with st.spinner("The module is analysing the image…"):
                    st.session_state["_rep"] = fn()
                st.session_state["_img"] = cache_key
                st.session_state["_secs"] = time.time() - t0
            return st.session_state.get("_rep")

        if organ == "Not performed":
            st.markdown(f"<div class='scan'><div style='display:flex;"
                        f"justify-content:space-between'><span>No examination</span>"
                        f"<span>—</span></div><div style='text-align:center;color:#8A90A6'>"
                        f"No POCUS was performed for this encounter</div>"
                        f"<div>&nbsp;</div></div>", unsafe_allow_html=True)
        elif not runs:
            st.warning(f"The {key} module cannot run here — {MODULES[key]['reason']}. "
                       f"Record below what the module reported.")
        elif organ == "Heart":
            # EF is a comparison between two frames, so this organ takes two uploads. A single
            # still cannot produce one, and passing the same frame twice would compute a
            # fractional change of zero and report a normal ventricle.
            st.caption("Ejection fraction is derived from two frames. A single still cannot "
                       "produce one.")
            h1, h2 = st.columns(2)
            ed = h1.file_uploader("End-diastole", type=IMG_T, key="up_ed")
            es = h2.file_uploader("End-systole", type=IMG_T, key="up_es")
            if ed and es:
                h1.image(ed, use_container_width=True)
                h2.image(es, use_container_width=True)
                analysed = _run(f"heart:{ed.name}:{es.name}",
                                lambda: ultrasound_agent("heart", ed=_grey(ed),
                                                         es=_grey(es)))
                ed.seek(0)
                st.session_state["_imgbytes"] = ed.read()
            elif ed or es:
                st.info("Both frames are needed. With only one the module reports a failed "
                        "measurement rather than a number.")
        else:
            up = st.file_uploader(
                f"Add {key} examination — the model reads it", type=IMG_T, key="up_one")
            if up is not None:
                st.image(up, use_container_width=True)
                analysed = _run(f"{key}:{up.name}",
                                lambda: ultrasound_agent(key, image=_grey(up)))
                up.seek(0)
                st.session_state["_imgbytes"] = up.read()

        if analysed is None and organ != "Not performed":
            st.markdown(f"<div class='scan'><div style='display:flex;"
                        f"justify-content:space-between'><span>{organ}</span>"
                        f"<span>No clip</span></div>"
                        f"<div style='text-align:center;color:#8A90A6'>"
                        f"Drop an image above and the module reads it</div>"
                        f"<div>&nbsp;</div></div>", unsafe_allow_html=True)

    with right:
        if analysed is not None:
            rows = "".join(
                f"<div class='row'><div><div style='font-weight:700;font-size:15px'>"
                f"{E(f['label'])}</div><div style='font-size:13px;color:{MUTED}'>"
                f"{'unreliable — weak training signal' if f.get('unreliable') else 'lung'}"
                f"</div></div><div style='text-align:right'>"
                f"<span class='tag t-vio'>Detected</span>"
                f"<div style='margin-top:5px;font-size:12.5px;color:{FAINT}'>"
                f"{f['confidence']:.2f}</div></div></div>"
                for f in analysed["findings"]) + "".join(
                f"<div class='row'><div style='font-weight:600;font-size:15px'>"
                f"{E(f['label'])}</div><div style='color:{MUTED};font-size:13.5px'>"
                f"Not detected · {f['confidence']:.2f}</div></div>"
                for f in analysed["not_detected"])
            if analysed["organ"] == "lung":
                rows += (f"<div class='row'><div style='font-weight:600;font-size:15px;"
                         f"color:{GHOST}'>pneumothorax</div>"
                         f"<div style='color:{GHOST};font-size:13.5px'>Not assessed"
                         f"</div></div>")
            q, rel = analysed["quality"], analysed["reliability"]
            if analysed["status"] != "ok":
                rows = (f"<div class='row' style='color:{RED_D}'>Measurement failed — "
                        f"{E(str(rel.get('scope', '')))}</div>")

            # Each module declares its own limits, so the note is assembled from the report
            # rather than written for one organ.
            note = []
            if rel.get("has_normal_class") is False:
                note.append("This examination cannot establish that the organ is normal, and "
                            "it does not assess every condition.")
            if q.get("thresholds"):
                note.append("Operating points " + ", ".join(
                    f"{k} {v:.2f}" for k, v in q["thresholds"].items())
                    + f" — the values the training run tuned, read back from "
                      f"{q.get('thresholds_source')}.")
            note.append("Confidence is calibrated" if rel.get("confidence_calibrated")
                        else "Confidence is a raw sigmoid output, not a calibrated probability")
            if rel.get("confidence_ceiling") and rel["confidence_ceiling"] < 1.0:
                note.append(f"the calibrator ceilings confidence at "
                            f"{rel['confidence_ceiling']:.2f}, so even unanimous agreement "
                            f"does not read as certainty")
            if rel.get("ece") is not None:
                note.append(f"ECE {rel['ece']:.3f}")
            if q.get("fine_grained"):
                note.append(f"finest-grained class: {q['fine_grained']}")
            if rel.get("scope") and analysed["status"] == "ok":
                note.append(str(rel["scope"]))

            meas = ""
            if analysed.get("measurements"):
                meas = "".join(
                    f"<div class='row'><div style='font-weight:600;font-size:15px'>"
                    f"{E(k.replace('_', ' '))}</div>"
                    f"<div style='font-size:15px;font-weight:700'>{v}</div></div>"
                    for k, v in analysed["measurements"].items())

            st.markdown(
                f"<div class='card rise'><h2 class='h2'>Ultrasound findings</h2>"
                f"<p class='sub'>Read from the uploaded examination by "
                f"{E(str(analysed.get('model') or 'the module'))} in "
                f"{st.session_state.get('_secs', 0):.1f}s on CPU. Nothing here was typed in."
                f"</p>{rows}{meas}"
                f"<div class='note' style='margin-top:18px'>{E('. '.join(note))}.</div></div>",
                unsafe_allow_html=True)
        else:
            st.markdown("<div class='card'><h2 class='h2'>Ultrasound findings</h2>"
                        "<p class='sub'>No image analysed. Record what the module reported — "
                        "unticked means <b>screened for and not seen</b>, which is itself an "
                        "observation.</p></div>", unsafe_allow_html=True)
            if organ == "Lung":
                for f in LUNG_FINDINGS:
                    a, b = st.columns([1, 2])
                    a.checkbox(f.replace("_", " "), key=f"f_{f}")
                    b.slider(" ", 0.0, 1.0, step=0.01, key=f"c_{f}",
                             label_visibility="collapsed")
            elif organ == "Heart":
                a, b = st.columns([1, 2])
                a.checkbox("severe dysfunction", key="f_sd")
                b.slider(" ", 0.0, 1.0, step=0.01, key="c_sd", label_visibility="collapsed")
            elif organ == "Gallbladder":
                st.caption("Single-label: one of five classes is reported, never a set.")
                st.selectbox("Reported class", GB_CLASSES, key="gb_cls")
                st.slider("Confidence", 0.0, 1.0, step=0.01, key="gb_conf")
                _c = st.session_state["gb_conf"]
                st.markdown(
                    f"<div class='note'>Clinical group: "
                    f"<b>{E(GB_GROUP[st.session_state['gb_cls']])}</b>. "
                    + (f"Below {GB_LOW_CONFIDENCE:g} the module's own notebook marks the read "
                       f"low-confidence rather than presenting it as a call — this one is "
                       f"<b>{_c:.2f}</b>. " if _c < GB_LOW_CONFIDENCE else "")
                    + f"{E(GB_SCOPE)}.</div>", unsafe_allow_html=True)

    st.write("")
    nav = st.columns([1, 1.4, 5])
    nav[0].button("← Back", on_click=go, args=("clinical",), use_container_width=True)
    if nav[1].button("Analyse patient →", type="primary", use_container_width=True):
        findings = {}
        if analysed is None:
            if organ == "Lung":
                for f in LUNG_FINDINGS:
                    c_ = st.session_state[f"c_{f}"]
                    findings[f] = c_ if st.session_state[f"f_{f}"] else -c_
            elif organ == "Heart":
                c_ = st.session_state["c_sd"]
                findings["severe dysfunction"] = c_ if st.session_state["f_sd"] else -c_
            elif organ == "Gallbladder":
                findings = {st.session_state["gb_cls"]: st.session_state["gb_conf"]}
        F["organ"], F["findings"] = organ, findings
        P = CASES.get(st.session_state.get("_preset") or "", BLANK)
        sub = dict(name=F["name"], age=int(F["age"]), sex=F["sex"],
                   complaint=F["complaint"], tier=F["tier"], tconf=float(F["tconf"]),
                   organ=organ, findings=findings, vitals=dict(F["vitals"]),
                   labs=dict(F["labs"]), report=analysed,
                   not_assessed=list(F.get("unassessed") or []))
        # A recorded differential belongs to a specific record. If the clinician has edited any
        # field, or the image was analysed here, the recording no longer describes what is on
        # screen — so it is dropped rather than shown beside a record it does not match.
        same = bool(analysed is None and P.get("key") and all(
            sub[f] == P[f] for f in ("age", "sex", "complaint", "tier", "tconf", "organ",
                                     "vitals", "labs", "findings")))
        sub["frozen_key"] = P["key"] if same else None
        sub["id"] = f"DEMO-{P['key'].upper()}" if same else "ENC-LIVE"
        st.session_state["enc"] = sub

        # The patient record is scoped to this session: what it shows is what this run of the
        # application has actually done. There is no database behind it.
        a2 = analyse(sub, st.session_state.get("broken", False))
        st.session_state["records"].append({
            "name": sub["name"] or "Unnamed patient", "age": sub["age"], "sex": sub["sex"],
            "complaint": sub["complaint"], "at": datetime.now().strftime("%H:%M"),
            "organ": organ, "severity": a2["support"]["severity"]["severity"],
            "alerts": len(a2["support"]["alerts"]),
            "encounter_id": a2["report"]["encounter_id"],
            "vitals": dict(sub["vitals"]), "labs": dict(sub["labs"]),
            "findings": [f["label"] for f in a2["state"]["imaging"]["findings"]
                         if f["detected"]],
            "image": st.session_state.get("_imgbytes") if analysed is not None else None,
            "report_text": render_report(a2["report"]),
        })
        st.session_state["ran"] = len(st.session_state["records"])
        st.session_state["msgs"] = st.session_state["msgs"][:1]
        go("assessment")
        st.rerun()

# ═══════════════════════════════════════════════════════════ 5 · DIAGNOSIS
elif SCREEN == "diagnosis":
    if not need_encounter():
        state, sup = A["state"], A["support"]
        diff = (A["result"].get("differential") or {}).get("differential") or []
        ev = build_evidence(state)
        cited = {i for d in diff for i in (d.get("supporting_ids") or [])}
        det = [f for f in state["imaging"]["findings"] if f["detected"]]

        top = st.columns([1.55, 1], gap="large")
        with top[0]:
            st.markdown(
                f"<div class='rise' style='background:{IRIS};border-radius:10px;"
                f"padding:30px 34px;color:#fff'>"
                f"<h1 style='margin:0;font-size:29px;font-weight:800;letter-spacing:-.02em;"
                f"color:#fff!important'>{E(enc['name'] or 'Unnamed patient')}</h1>"
                f"<p style='margin:10px 0 0;font-size:15px;opacity:.9'>{enc['age']} · "
                f"{E(enc['sex'])} · {E(enc['complaint'] or 'no complaint given')}</p>"
                f"<div style='margin-top:22px;display:flex;gap:26px;flex-wrap:wrap;"
                f"font-size:12.5px;font-weight:700;letter-spacing:.09em;"
                f"text-transform:uppercase'>"
                f"<span>✦ {len(sup['additional_examinations'])} suggestions</span>"
                f"<span>⚠ {len(sup['alerts'])} alerts</span>"
                f"<span>◇ {len(state['missing']['labs']) + len(state['missing']['vitals'])} "
                f"absent</span></div></div>", unsafe_allow_html=True)
        with top[1]:
            st.markdown(
                f"<div class='rise-2' style='background:{LIME};border-radius:10px;"
                f"padding:26px 30px;color:{INK}'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center'>"
                f"<span style='font-weight:700;font-size:15.5px'>▤ Evidence considered</span>"
                f"<span style='opacity:.7'>⋮</span></div>"
                f"<div style='margin-top:24px;display:flex;align-items:baseline;gap:9px'>"
                f"<span style='font-size:36px;font-weight:800;letter-spacing:-.03em'>"
                f"{len(cited)}/{len(ev)}</span>"
                f"<span style='font-size:14px;opacity:.9'>identifiers cited</span></div>"
                f"<div style='margin-top:18px;display:flex;gap:7px'>"
                f"<span style='flex:{max(len(cited), 1)};height:6px;background:{NAVY};"
                f"border-radius:3px'></span>"
                f"<span style='flex:{max(len(ev) - len(cited), 1)};height:6px;"
                f"background:rgba(46,42,120,.28);border-radius:3px'></span></div></div>",
                unsafe_allow_html=True)

        st.write("")
        # Supporting vs against, per differential entry. Both counts come from the validated
        # answer's own evidence arrays; nothing here is a model-reported probability, because
        # the model does not emit one.
        if diff:
            mx = max([max(len(d.get("supporting_ids") or []),
                          len(d.get("contradicting") or [])) for d in diff] + [1])
            bars = ""
            for d in diff:
                ns = len(d.get("supporting_ids") or [])
                na = len(d.get("contradicting") or [])
                bars += (
                    f"<div style='flex:1;display:flex;flex-direction:column;"
                    f"align-items:center;justify-content:flex-end;height:100%;gap:8px'>"
                    f"<div style='display:flex;gap:5px;align-items:flex-end;height:100%;"
                    f"width:100%;justify-content:center'>"
                    f"<div class='gbar' style='width:26px;height:{ns / mx * 100:.0f}%;"
                    f"background:{IRIS}' title='{ns} supporting'></div>"
                    f"<div class='gbar' style='width:26px;height:{na / mx * 100:.0f}%;"
                    f"background:{NAVY}' title='{na} against'></div></div>"
                    f"<div style='font-size:11.5px;color:{GHOST};font-weight:600;"
                    f"text-align:center;line-height:1.25'>{E(str(d.get('diagnosis'))[:26])}"
                    f"</div></div>")
            ticks = "".join(f"<span>{v}</span>" for v in
                            (mx, int(mx * .8), int(mx * .6), int(mx * .4), int(mx * .2), 0))
            st.markdown(
                f"<div class='card10 rise-2'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;"
                f"gap:20px;flex-wrap:wrap'>"
                f"<h2 style='margin:0;font-size:21px;font-weight:800;letter-spacing:-.015em'>"
                f"▦ Differential diagnosis — supporting vs against</h2>"
                f"<div style='display:flex;gap:16px;font-size:12.5px;font-weight:700;"
                f"letter-spacing:.07em;text-transform:uppercase'>"
                f"<span style='display:flex;gap:8px;align-items:center'>"
                f"<span style='width:11px;height:11px;background:{IRIS}'></span>Supporting</span>"
                f"<span style='display:flex;gap:8px;align-items:center'>"
                f"<span style='width:11px;height:11px;background:{NAVY}'></span>Against</span>"
                f"</div></div>"
                f"<div style='margin-top:26px;display:grid;grid-template-columns:38px 1fr;"
                f"gap:12px'><div class='axis'>{ticks}</div>"
                f"<div class='plot'>{bars}</div></div>"
                f"<p style='margin:20px 0 0;font-size:13.5px;color:{MUTED}'>Bars count the "
                f"evidence identifiers the validated answer cited for and against each "
                f"possibility. They are not probabilities — the model does not emit one, and "
                f"inventing a percentage here would be the fabrication this system exists to "
                f"prevent.</p></div>", unsafe_allow_html=True)
        else:
            why = ("The model backend failed and the answer was withheld. The escalation "
                   "decision beside it was computed before the model ran, and is unaffected."
                   ) if A["origin"] == "failed" else (
                   "Generating one requires a 4.9 GB language model on a GPU, which is not "
                   "loaded in this deployment.")
            st.markdown(
                f"<div class='card10 rise-2'>"
                f"<h2 style='margin:0;font-size:21px;font-weight:800'>▦ Differential diagnosis "
                f"— supporting vs against</h2>"
                f"<p style='margin:16px 0 0;font-size:15px;color:{MUTED}'>No differential was "
                f"produced for this encounter, so there is nothing to chart. {why}</p></div>",
                unsafe_allow_html=True)

        cards = st.columns(4, gap="medium")
        with cards[0]:
            lead = diff[0] if diff else None
            st.markdown(
                f"<div class='card10 lift rise-3' style='min-height:250px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='font-weight:700;font-size:15px;color:{NAVY}'>⌁ Top match</span>"
                f"<span style='color:#C2BED0'>⋮</span></div>"
                f"<div class='dxbox' style='margin-top:18px'>"
                f"{E(str(lead.get('diagnosis')) if lead else 'none produced')}</div>"
                f"<div style='margin-top:18px;display:flex;align-items:baseline;gap:6px'>"
                f"<span class='big'>{len(lead.get('supporting_ids') or []) if lead else 0}"
                f"</span><span style='font-size:15px;color:{MUTED}'>cited</span></div>"
                f"<div style='font-size:12.5px;color:{FAINT};margin-top:4px'>"
                f"{E(str(lead.get('likelihood')) if lead else '—')} likelihood</div></div>",
                unsafe_allow_html=True)
        with cards[1]:
            vt = state.get("vitals") or {}
            mx2 = 1.0
            spark = ""
            for k, v in vt.items():
                ref = VITAL_REFERENCE[k]
                span = max(ref["normal_max"] - ref["normal_min"], 1)
                dev = min(abs(v["value"] - (ref["normal_min"] + ref["normal_max"]) / 2)
                          / span, 1.6) / 1.6
                col = LIME if v["flag"] == "normal" else (LIME_M if dev < .6 else LIME_XL)
                spark += (f"<span style='flex:1;height:{max(dev * 100, 8):.0f}%;"
                          f"background:{col}' title='{k}'></span>")
            worst = max(vt.items(), key=lambda kv: kv[1]["flag"] != "normal",
                        default=(None, None))
            st.markdown(
                f"<div class='card10 lift rise-3' style='min-height:250px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='font-weight:700;font-size:15px;color:{LIME_D}'>✛ Vitals</span>"
                f"<span style='color:#C2BED0'>⋮</span></div>"
                f"<div style='margin-top:20px;height:92px;display:flex;align-items:flex-end;"
                f"gap:7px'>{spark or '<span style=\"color:#9C99B8\">none recorded</span>'}"
                f"</div>"
                f"<div style='margin-top:18px;display:flex;align-items:baseline;gap:6px'>"
                f"<span class='big'>{len(vt)}</span>"
                f"<span style='font-size:15px;color:{MUTED}'>recorded</span></div>"
                f"<div style='font-size:12.5px;color:{FAINT};margin-top:4px'>"
                f"{sum(1 for v in vt.values() if v['flag'] != 'normal')} outside range · "
                f"{len(state['missing']['vitals'])} never measured</div></div>",
                unsafe_allow_html=True)
        with cards[2]:
            cells = ""
            for e in ev[:28]:
                if e["id"] in cited:
                    bg = IRIS
                elif "HIGH" in e["text"] or "LOW" in e["text"]:
                    bg = RED_L
                else:
                    bg = BORDER
                cells += f"<span style='aspect-ratio:1;border-radius:3px;background:{bg}'></span>"
            st.markdown(
                f"<div class='card10 lift rise-3' style='min-height:250px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='font-weight:700;font-size:15px;color:{RED}'>〜 Evidence map"
                f"</span><span style='color:#C2BED0'>⋮</span></div>"
                f"<div style='margin-top:20px;display:grid;"
                f"grid-template-columns:repeat(7,1fr);gap:5px'>{cells}</div>"
                f"<div style='margin-top:18px;display:flex;align-items:baseline;gap:6px'>"
                f"<span class='big'>{len(ev)}</span>"
                f"<span style='font-size:15px;color:{MUTED}'>citable</span></div>"
                f"<div style='font-size:12.5px;color:{FAINT};margin-top:4px'>"
                f"filled = cited by the model</div></div>", unsafe_allow_html=True)
        with cards[3]:
            st.markdown(
                f"<div class='rise-3' style='background:#E9E8FB;border-radius:10px;"
                f"padding:24px;display:flex;flex-direction:column;align-items:center;"
                f"justify-content:center;gap:12px;color:{NAVY};min-height:250px'>"
                f"<span style='font-size:26px'>＋</span>"
                f"<span style='font-size:12.5px;font-weight:800;letter-spacing:.1em;"
                f"text-transform:uppercase;text-align:center'>Add measurement</span></div>",
                unsafe_allow_html=True)
            st.button("Open clinical data", key="dx_add", use_container_width=True,
                      on_click=go, args=("clinical",))

# ══════════════════════════════════════════════════════════ 6 · PATIENT RECORD
elif SCREEN == "record":
    recs = st.session_state["records"]
    who = (enc["name"] or "Unnamed patient") if enc else None
    mine = [r for r in recs if who is None or r["name"] == who]
    subtitle = (f"{enc['age']} · {E(enc['sex'])} · "
                f"{E(enc['complaint'] or 'no complaint given')}") if enc else \
        "Analyse an encounter to populate this record"

    st.markdown(
        f"<div class='band band-vio rise'><div style='font-size:12.5px;letter-spacing:.11em;"
        f"text-transform:uppercase;color:{V700};font-weight:800'>Session-scoped</div>"
        f"<p style='margin:8px 0 0;font-size:15px'>This system has no database and no "
        f"persistence between runs. What follows is what <b>this run of the application</b> "
        f"has actually done — encounters analysed, images the module read, reports generated. "
        f"Closing the app discards it.</p></div>", unsafe_allow_html=True)

    st.markdown(
        f"<div class='card10' style='display:flex;justify-content:space-between;gap:26px;"
        f"flex-wrap:wrap;align-items:center'>"
        f"<div style='display:flex;gap:18px;align-items:center'>"
        f"<div style='width:62px;height:62px;border-radius:50%;background:{BORDER};"
        f"display:flex;align-items:center;justify-content:center;font-weight:800;"
        f"font-size:20px;color:{V700}'>"
        f"{E((who or '—')[:2].upper())}</div><div>"
        f"<h1 style='margin:0;font-size:27px;font-weight:800;letter-spacing:-.02em'>"
        f"{E(who or 'No patient loaded')}</h1>"
        f"<div style='margin-top:5px;font-size:14px;color:{MUTED}'>{subtitle}</div>"
        f"</div></div>"
        f"<div style='display:flex;gap:28px;flex-wrap:wrap;font-size:13.5px'>"
        f"<div><div style='color:{MUTED}'>Encounters</div>"
        f"<div style='font-weight:800;font-size:18px;margin-top:3px'>{len(mine)}</div></div>"
        f"<div><div style='color:{MUTED}'>POCUS images read</div>"
        f"<div style='font-weight:800;font-size:18px;margin-top:3px'>"
        f"{sum(1 for r in mine if r['image'])}</div></div>"
        f"<div><div style='color:{MUTED}'>Reports</div>"
        f"<div style='font-weight:800;font-size:18px;margin-top:3px'>{len(mine)}</div></div>"
        f"<div><div style='color:{MUTED}'>Last</div>"
        f"<div style='font-weight:800;font-size:18px;margin-top:3px'>"
        f"{E(mine[-1]['at']) if mine else '—'}</div></div></div></div>",
        unsafe_allow_html=True)

    tabs = st.columns([1, 1, 1, 1, 4])
    for i, t in enumerate(["Images", "Data", "Visits", "Reports"]):
        if tabs[i].button(t, key=f"rt_{t}", use_container_width=True,
                          type="primary" if st.session_state["rec_tab"] == t else "secondary"):
            st.session_state["rec_tab"] = t
            st.rerun()

    TAB = st.session_state["rec_tab"]
    if not mine:
        st.markdown("<div class='card10'><p style='margin:0;font-size:15px'>Nothing recorded "
                    "yet this session. Run an assessment and it appears here.</p></div>",
                    unsafe_allow_html=True)

    elif TAB == "Images":
        st.markdown(f"<div class='card10'><div style='display:flex;"
                    f"justify-content:space-between;align-items:baseline;flex-wrap:wrap'>"
                    f"<h2 style='margin:0;font-size:18px;font-weight:800'>Stored POCUS images"
                    f"</h2><span style='font-size:13.5px;color:{MUTED}'>"
                    f"{sum(1 for r in mine if r['image'])} image(s) the module actually read"
                    f"</span></div></div>", unsafe_allow_html=True)
        shots = [r for r in mine if r["image"]]
        if not shots:
            st.markdown(f"<div class='card10'><p style='margin:0;color:{MUTED}'>No image was "
                        f"uploaded this session. Only the lung module can read one — cardiac "
                        f"and gallbladder weights are not in this deployment.</p></div>",
                        unsafe_allow_html=True)
        cols = st.columns(4, gap="medium")
        for i, r in enumerate(shots):
            with cols[i % 4]:
                st.image(r["image"], use_container_width=True)
                st.markdown(f"<div style='font-size:12.5px;color:{MUTED};line-height:1.35;"
                            f"margin-bottom:14px'><span style='font-weight:700;color:{INK}'>"
                            f"{E(', '.join(r['findings']) or 'no finding above threshold')}"
                            f"</span><br>{E(r['organ'])} · {E(r['at'])}</div>",
                            unsafe_allow_html=True)

    elif TAB == "Data":
        cols_ = [r["at"] for r in mine]
        head = "".join(f"<th>{E(c)}</th>" for c in cols_)
        body = ""
        for k, ref in VITAL_REFERENCE.items():
            cells = ""
            for r in mine:
                v = r["vitals"].get(k)
                cells += (f"<td style='color:{MUTED}'>{v:g}</td>" if v is not None
                          else f"<td style='color:{GHOST}'>—</td>")
            body += f"<tr><td style='font-weight:600'>{k} ({ref['unit']})</td>{cells}</tr>"
        for k, ref in LAB_REFERENCE.items():
            cells = ""
            for r in mine:
                v = r["labs"].get(k)
                cells += (f"<td style='color:{MUTED}'>{v:g}</td>" if v is not None
                          else f"<td style='color:{GHOST}'>—</td>")
            body += f"<tr><td style='font-weight:600'>{k}</td>{cells}</tr>"
        st.markdown(
            f"<div class='card10'><h2 style='margin:0 0 18px;font-size:18px;font-weight:800'>"
            f"Stored measurements — all encounters this session</h2>"
            f"<div style='overflow:auto'><table class='dt' style='min-width:560px'>"
            f"<thead><tr><th>Measurement</th>{head}</tr></thead><tbody>{body}</tbody></table>"
            f"</div><p style='margin:18px 0 0;font-size:13.5px;color:{MUTED}'>Blank cells mean "
            f"the measurement was not taken at that encounter. Not measured does not mean "
            f"normal.</p></div>", unsafe_allow_html=True)

    elif TAB == "Visits":
        items = ""
        for r in mine:
            tone = {"HIGH": ("t-red", "Critical"), "MODERATE": ("t-amb", "Review"),
                    "LOW": ("t-lime", "Completed")}[r["severity"]]
            items += (
                f"<div style='border:1px solid {BORDER};border-left:4px solid {IRIS};"
                f"border-radius:8px;padding:18px 22px;display:flex;justify-content:"
                f"space-between;gap:20px;flex-wrap:wrap;align-items:center;margin-bottom:14px'>"
                f"<div><div style='font-weight:800;font-size:16px'>{E(r['at'])} — "
                f"{E(r['complaint'] or 'no complaint given')}</div>"
                f"<div style='margin-top:5px;font-size:13.5px;color:{MUTED}'>"
                f"{E(r['encounter_id'])} · {E(r['organ'])} · {r['alerts']} alert(s) · "
                f"{E(', '.join(r['findings']) or 'no positive finding')}</div></div>"
                f"<span class='tag {tone[0]}'>{tone[1]}</span></div>")
        st.markdown(f"<div class='card10'><h2 style='margin:0 0 18px;font-size:18px;"
                    f"font-weight:800'>Encounter history</h2>{items}</div>",
                    unsafe_allow_html=True)

    elif TAB == "Reports":
        cols = st.columns(3, gap="medium")
        for i, r in enumerate(mine):
            with cols[i % 3]:
                st.markdown(
                    f"<div class='card10 lift' style='min-height:150px'>"
                    f"<div style='font-weight:800;font-size:16px'>{E(r['encounter_id'])}</div>"
                    f"<div style='margin-top:6px;font-size:13.5px;color:{MUTED}'>"
                    f"{E(r['at'])} · {r['severity']} severity · {r['alerts']} alert(s)</div>"
                    f"</div>", unsafe_allow_html=True)
                st.download_button("Download", r["report_text"], key=f"dl{i}",
                                   file_name=f"{r['encounter_id']}_{i}.txt",
                                   use_container_width=True)

# ═══════════════════════════════════════════════════════════ 7 · ASSESSMENT
elif SCREEN == "assessment":
    if not need_encounter():
        state, esc, sup = A["state"], A["esc"], A["support"]
        sev, origin, rep = sup["severity"], A["origin"], A["report"]
        top = st.columns([4, 1.3])
        top[0].markdown(f"<h1 class='h1' style='margin-top:0'>Clinical assessment</h1>"
                        f"<p class='lede'>{E(enc['name'] or 'Unnamed patient')} · "
                        f"{enc['age']} · {E(enc['sex'])} · "
                        f"{E(enc['complaint'] or 'no complaint given')}</p>",
                        unsafe_allow_html=True)
        top[1].button("Generate clinical report", type="primary", on_click=go,
                      args=("report",), use_container_width=True)
        st.write("")

        tone = {"HIGH": ("band-red alertpulse", RED_D, "High clinical priority"),
                "MODERATE": ("band-amb", AMB_D, "Moderate clinical priority"),
                "LOW": ("band-lime", LIME_D, "Low clinical priority")}[sev["severity"]]
        st.markdown(
            f"<div class='band {tone[0]} rise'>"
            f"<div style='font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;"
            f"color:{tone[1]};font-weight:800'>{tone[2]}</div>"
            f"<p style='margin:8px 0 0;font-size:16px;max-width:78ch'>{E(rep['conclusion'])}"
            f"</p><div style='margin-top:10px;font-size:13.5px;color:{MUTED}'>"
            f"Routed as <b>{E(sup['scenario']['label'])}</b> · thresholds "
            f"v{E(str(sup['thresholds_version']))} · computed before the model runs, and "
            f"nothing the model says can lower it.</div></div>", unsafe_allow_html=True)
        st.button("Review alerts →", on_click=go, args=("alerts",))
        st.write("")

        L, R = st.columns([1, 1.25], gap="large")

        with L:
            vit = "".join(
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span>{k} {v['value']:g} {v['unit']}</span>"
                f"<span style='color:{RED_D if v['flag'] != 'normal' else LIME_D};"
                f"font-weight:700'>{v['flag'].title()}</span></div>"
                for k, v in (state.get("vitals") or {}).items())
            lab = "".join(
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span>{k} {v['value']:g}</span>"
                f"<span style='color:{RED_D if v['flag'] != 'normal' else LIME_D};"
                f"font-weight:700'>{v['flag'].title()}</span></div>"
                for k, v in (state.get("labs") or {}).items()) + "".join(
                f"<div style='display:flex;justify-content:space-between;color:{GHOST}'>"
                f"<span>{k}</span><span>Not available</span></div>"
                for k in state["missing"]["labs"])
            det = [f for f in state["imaging"]["findings"] if f["detected"]]
            pocus = "; ".join(f"{f['label']} ({f['confidence']:.2f})" for f in det) or \
                    "no finding above threshold"
            gaps = "".join(
                f"<div style='color:{RED_D};font-weight:700'>{o} — not assessed</div>"
                for o in state["imaging"]["organs_not_assessed"])
            st.markdown(
                f"<div class='card rise-2'><h2 class='h2'>What the assistant sees</h2>"
                f"<p class='sub'>The information available for this patient.</p>"
                f"<div class='kicker'>Symptoms</div>"
                f"<div style='margin:8px 0 18px;font-size:14.5px'>"
                f"{E(enc['complaint'] or '—')}</div>"
                f"<div class='kicker'>Vital signs</div>"
                f"<div style='margin:10px 0 18px;display:flex;flex-direction:column;gap:9px;"
                f"font-size:14.5px'>{vit or '<i>none recorded</i>'}</div>"
                f"<div class='kicker'>POCUS</div>"
                f"<div style='margin:8px 0 18px;font-size:14.5px'>{E(pocus)}{gaps}</div>"
                f"<div class='kicker'>Laboratory</div>"
                f"<div style='margin:10px 0 0;display:flex;flex-direction:column;gap:9px;"
                f"font-size:14.5px'>{lab or '<i>none recorded</i>'}</div></div>",
                unsafe_allow_html=True)

            lim = "".join(f"<li>{E(x)}</li>" for x in state["imaging"]["out_of_scope"])
            if lim:
                st.markdown(f"<div class='card'><h2 class='h2'>What the models cannot "
                            f"exclude</h2><ul style='margin:12px 0 0;padding-left:18px;"
                            f"font-size:14px;color:{MUTED}'>{lim}</ul></div>",
                            unsafe_allow_html=True)

        with R:
            st.markdown("<div class='card' style='padding-bottom:8px'>"
                        "<h2 class='h2'>Differential assessment</h2>"
                        "<p class='sub'>Offered for consideration. This is not a diagnosis."
                        "</p></div>", unsafe_allow_html=True)
            if origin == "failed":
                st.markdown(
                    f"<div class='band band-red'><div style='font-size:12.5px;"
                    f"letter-spacing:.11em;text-transform:uppercase;color:{RED_D};"
                    f"font-weight:800'>Differential withheld</div>"
                    f"<p style='margin:8px 0 0;font-size:15px'>"
                    f"{E('; '.join(A['result']['validation_errors'] or []))}</p></div>"
                    f"<div class='band band-lime'><div style='font-size:12.5px;"
                    f"letter-spacing:.11em;text-transform:uppercase;color:{LIME_D};"
                    f"font-weight:800'>The safety decision survived</div>"
                    f"<p style='margin:8px 0 0;font-size:15px'><b>{sev['severity']} severity, "
                    f"{'escalation required' if esc['escalate'] else 'no escalation'}</b> — "
                    f"computed from the record before the model was consulted, so it does not "
                    f"depend on the model running at all.</p></div>", unsafe_allow_html=True)
            elif origin == "not_generated":
                st.markdown(
                    f"<div class='band band-vio'><div style='font-size:12.5px;"
                    f"letter-spacing:.11em;text-transform:uppercase;color:{V700};"
                    f"font-weight:800'>No differential generated</div>"
                    f"<p style='margin:8px 0 0;font-size:15px'>Producing one requires a 4.9 GB "
                    f"language model on a GPU, which is not loaded in this deployment. "
                    f"Everything else on this screen was computed on this machine. Load a "
                    f"benchmark encounter unedited to see a differential recorded from the GPU "
                    f"run.</p></div>", unsafe_allow_html=True)
            else:
                diff = (A["result"].get("differential") or {}).get("differential") or []
                if A["result"].get("differential_withheld"):
                    st.markdown(
                        f"<div class='band band-red'><div style='font-size:12.5px;"
                        f"letter-spacing:.11em;text-transform:uppercase;color:{RED_D};"
                        f"font-weight:800'>Withheld — did not pass validation</div>"
                        f"<p style='margin:8px 0 0;font-size:15px'>"
                        f"{E('; '.join(A['result'].get('validation_errors') or []))}</p></div>",
                        unsafe_allow_html=True)
                for i, dx in enumerate(diff, 1):
                    lk = str(dx.get("likelihood", "")).lower()
                    pct = {"high": 84, "moderate": 62, "low": 34}.get(lk, 50)
                    sup_l = "".join(f"<li>{E(t)}</li>" for t in (dx.get("supporting") or []))
                    lim_l = "".join(f"<li>{E(t)}</li>" for t in (dx.get("limitations") or []))
                    con_l = "".join(f"<li>{E(t)}</li>" for t in (dx.get("contradicting") or []))
                    ids = ", ".join(dx.get("supporting_ids") or []) or "—"
                    lead = i == 1
                    st.markdown(
                        f"<div class='{'dx' if lead else 'dx-sec'} rise-2'>"
                        f"<div style='display:flex;justify-content:space-between;"
                        f"align-items:baseline;gap:16px;flex-wrap:wrap'><div>"
                        f"<span style='font-size:12.5px;color:{FAINT};font-weight:700'>"
                        f"{i:02d}{' · Most likely' if lead else ''}</span>"
                        f"<div style='font-size:{22 if lead else 19}px;font-weight:"
                        f"{800 if lead else 700};margin-top:4px'>{E(str(dx.get('diagnosis')))}"
                        f"</div></div>"
                        f"<span class='tag t-vio' style='padding:6px 14px;font-size:13px'>"
                        f"{lk} likelihood</span></div>"
                        f"<div class='pbar'><i style='width:{pct}%'></i></div>"
                        f"<div style='margin-top:16px' class='grid2'>"
                        f"<div><div style='font-size:12px;letter-spacing:.09em;"
                        f"text-transform:uppercase;color:{LIME_D};font-weight:800'>"
                        f"Why it is considered</div>"
                        f"<ul style='margin:10px 0 0;padding-left:18px;font-size:14px'>"
                        f"{sup_l or '<li>—</li>'}</ul>"
                        f"<div style='margin-top:8px;font-size:12.5px;color:{FAINT}'>"
                        f"cited as {E(ids)}</div></div>"
                        f"<div><div style='font-size:12px;letter-spacing:.09em;"
                        f"text-transform:uppercase;color:{AMB_D};font-weight:800'>"
                        f"What limits confidence</div>"
                        f"<ul style='margin:10px 0 0;padding-left:18px;font-size:14px;"
                        f"color:{MUTED}'>{lim_l}{con_l or ''}</ul></div></div></div>",
                        unsafe_allow_html=True)
                if A["result"].get("warnings"):
                    st.markdown(
                        f"<div class='band band-amb'><div style='font-size:12.5px;"
                        f"letter-spacing:.11em;text-transform:uppercase;color:{AMB_D};"
                        f"font-weight:800'>Delivered with warnings</div>" +
                        "".join(f"<div style='margin-top:6px;font-size:14.5px'>{E(w)}</div>"
                                for w in A["result"]["warnings"]) + "</div>",
                        unsafe_allow_html=True)
                bq = st.columns([1, 1.4, 4])
                bq[0].button("Why?", on_click=lambda: (
                    st.session_state["msgs"].extend([
                        {"r": "d", "t": "Why is the leading entry ranked first?"},
                        {"r": "a", "t": "ASK:why"}]), go("assistant")))
                bq[1].button("🔍 Challenge this assessment", on_click=lambda: (
                    st.session_state["msgs"].extend([
                        {"r": "d", "t": "Challenge this assessment."},
                        {"r": "a", "t": "ASK:challenge"}]), go("assistant")))

            gone = state["missing"]["labs"] + state["missing"]["vitals"]
            items = "".join(
                f"<div style='display:flex;gap:14px;padding:14px 16px;background:#FFF8EC;"
                f"border-radius:14px;margin-bottom:12px'><span style='color:#E08A00;"
                f"font-size:15px'>●</span><div><div style='font-weight:700;font-size:15px'>"
                f"{E(r['exam'])}</div><div style='font-size:13.5px;color:{MUTED}'>"
                f"{E(r['reason'])}</div></div></div>"
                for r in sup["additional_examinations"][:6]) or \
                "<div class='note'>The record is complete for this presentation.</div>"
            st.markdown(
                f"<div class='card'><h2 class='h2'>What would help clarify this case?</h2>"
                f"<p class='sub'>These were not performed. Their absence does not mean they "
                f"are normal.</p>{items}"
                f"<div style='margin-top:20px;padding-top:18px;border-top:1px solid {RULE}'>"
                f"<div class='kicker'>Suggested next step</div>"
                f"<p style='margin:8px 0 0;font-size:18px;font-weight:700;line-height:1.35'>"
                f"{E(sup['additional_examinations'][0]['exam'] if sup['additional_examinations'] else 'Physician review of the current findings.')}"
                f"</p><div style='margin-top:8px;font-size:13.5px;color:{MUTED}'>The system "
                f"recommends examinations; it does not order them, and they do not replace "
                f"clinician judgement.</div></div></div>", unsafe_allow_html=True)

            if gone:
                st.markdown(
                    f"<div class='card'><h2 class='h2'>Not measured — and not normal</h2>"
                    f"<p class='sub'>These have no evidence identifier, so the reasoning layer "
                    f"cannot cite them for or against any diagnosis.</p>" +
                    "".join(f"<span class='tag t-lav'>{E(g)}</span>" for g in gone) + "</div>",
                    unsafe_allow_html=True)

            th = sup["therapeutic"]
            body = "".join(
                f"<div style='margin-bottom:12px'><b>CONSIDER</b> — {E(c['consideration'])}"
                f"<div style='font-size:13.5px;color:{MUTED};margin-top:5px'>"
                f"Basis: {E(c['basis'])} [{E(c['passage'])}]<br>{E(c['disclaimer'])}</div></div>"
                for c in th["considerations"]) or (
                f"<p style='margin:0;font-size:14.5px'><b>No protocol-backed therapeutic "
                f"recommendation for this case.</b></p><div style='font-size:13.5px;"
                f"color:{MUTED};margin-top:6px'>{E(th['status'])}"
                f"{'. ' + E(th['note']) if th.get('note') else ''}</div>")
            st.markdown(f"<div class='card'><h2 class='h2'>Therapeutic considerations</h2>"
                        f"<p class='sub'>Gated behind a sourced, citable protocol. With none "
                        f"retrieved, nothing is suggested.</p>{body}</div>",
                        unsafe_allow_html=True)

        st.markdown(
            f"<div style='background:{IRIS};border-radius:20px;padding:26px 30px;color:#fff;"
            f"display:flex;justify-content:space-between;gap:26px;flex-wrap:wrap;"
            f"align-items:center'><div style='max-width:62ch'>"
            f"<div style='font-weight:800;font-size:18px'>✨ Ask the clinical assistant</div>"
            f"<p style='margin:8px 0 0;font-size:14.5px;opacity:.92'>Question this case in "
            f"plain language — what supports the leading possibility, what argues against it, "
            f"what is still missing.</p></div></div>", unsafe_allow_html=True)
        st.button("Open assistant", on_click=go, args=("assistant",))

# ═══════════════════════════════════════════════════════════════ 8 · ALERTS
elif SCREEN == "alerts":
    if not need_encounter():
        sup = A["support"]
        alerts = sup["alerts"]
        crit = [a for a in alerts if a["severity"] == "CRITICAL"]
        warn = [a for a in alerts if a["severity"] != "CRITICAL"]
        gone = A["state"]["missing"]["labs"] + A["state"]["missing"]["vitals"]

        st.markdown(f"<h1 class='h1' style='margin-top:0'>Clinical alerts</h1>"
                    f"<p class='lede'>{len(crit)} critical · {len(warn)} warning · from "
                    f"thresholds v{E(str(sup['thresholds_version']))}, each naming the bound "
                    f"it crossed.</p>", unsafe_allow_html=True)
        st.write("")

        if not alerts:
            st.markdown(f"<div class='band band-lime rise'><div style='font-size:12.5px;"
                        f"letter-spacing:.11em;text-transform:uppercase;color:{LIME_D};"
                        f"font-weight:800'>No alert</div><p style='margin:8px 0 0;"
                        f"font-size:15px'>No measured value crossed a configured bound and the "
                        f"record shows no structural gap.</p></div>", unsafe_allow_html=True)

        for a in crit:
            st.markdown(
                f"<div class='band alertpulse rise' style='background:{CARD};"
                f"border:1px solid #F6CFCF;border-left:5px solid {RED}'>"
                f"<div style='font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;"
                f"color:{RED_D};font-weight:800'>Immediate attention</div>"
                f"<h2 style='margin:10px 0 6px;font-size:21px;font-weight:800'>"
                f"{E(a['type'].replace('_', ' ').title())}</h2>"
                f"<p style='margin:0;font-size:15px;color:{MUTED}'>{E(a['message'])}</p></div>",
                unsafe_allow_html=True)

        for a in warn:
            st.markdown(
                f"<div class='band rise-2' style='background:{CARD};border:1px solid #F7DFB4;"
                f"border-left:5px solid {AMB}'>"
                f"<div style='font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;"
                f"color:{AMB_D};font-weight:800'>Important</div>"
                f"<h2 style='margin:10px 0 6px;font-size:21px;font-weight:800'>"
                f"{E(a['type'].replace('_', ' ').title())}</h2>"
                f"<p style='margin:0;font-size:15px;color:{MUTED}'>{E(a['message'])}</p></div>",
                unsafe_allow_html=True)

        if gone:
            st.markdown(
                f"<div class='band band-vio rise-3'>"
                f"<div style='font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;"
                f"color:{V700};font-weight:800'>Information missing</div>"
                f"<h2 style='margin:10px 0 10px;font-size:21px;font-weight:800'>Important "
                f"investigations have not been performed</h2>" +
                "".join(f"<span class='tag t-lav' style='padding:6px 14px;font-size:13.5px'>"
                        f"{E(g)}</span>" for g in gone) +
                f"<p style='margin:14px 0 0;font-size:15px;color:{MUTED}'>These tests have not "
                f"been performed. They should not be interpreted as normal.</p></div>",
                unsafe_allow_html=True)

        st.markdown(f"<div class='card'><h2 class='h2'>Escalation triggers</h2>"
                    f"<p class='sub'>Evaluated before the model runs.</p>" +
                    ("".join(f"<div class='row'>{E(t)}</div>" for t in A["esc"]["triggers"])
                     or "<div class='row'>No trigger fired — the case may be answered "
                        "directly.</div>") + "</div>", unsafe_allow_html=True)
        st.button("View patient →", type="primary", on_click=go, args=("assessment",))

# ═══════════════════════════════════════════════════════════ 9 · ASSISTANT
elif SCREEN == "assistant":
    if not need_encounter():
        state, sup, rep = A["state"], A["support"], A["report"]
        diff = (A["result"].get("differential") or {}).get("differential") or []
        gone = state["missing"]["labs"] + state["missing"]["vitals"]
        det = [f for f in state["imaging"]["findings"] if f["detected"]]

        def answer(key: str) -> str:
            """Assembled from the computed assessment. Nothing here is generated prose."""
            if key == "support":
                if not diff:
                    return ("No differential was produced for this encounter, so there is "
                            "nothing ranked to support. The deterministic layer did run: "
                            f"severity {sup['severity']['severity']}, "
                            f"{len(sup['alerts'])} alert(s).")
                d = diff[0]
                return (f"{d.get('diagnosis')} is ranked first. The evidence cited for it, by "
                        f"identifier, is: "
                        + "; ".join(f"{i} — {t}" for i, t in
                                    zip(d.get("supporting_ids") or [],
                                        d.get("supporting") or []))
                        + ". Those identifiers come from the enumerated evidence list built "
                          "from this record, so nothing outside the record can be cited.")
            if key == "missing":
                return ("Not measured for this patient: " + ", ".join(gone) +
                        ". These have no evidence identifier, so the reasoning layer cannot "
                        "cite them for or against any diagnosis. Absent is not normal."
                        ) if gone else "Every value in the reference set was measured."
            if key == "summary":
                return rep["conclusion"]
            if key == "next":
                r = sup["additional_examinations"]
                return ("Recommended, in priority order: " +
                        "; ".join(f"{x['exam']} ({x['priority']}) — {x['reason']}"
                                  for x in r[:4])) if r else \
                    "The record is complete for this presentation."
            if key == "why":
                t = A["esc"]["triggers"]
                return (f"Severity is {sup['severity']['severity']} because: "
                        + "; ".join(sup["severity"]["reasons"] or ["no reason recorded"])
                        + ". The escalation triggers that fired were: "
                        + ("; ".join(t) if t else "none")
                        + ". All of this is computed by fixed rules before the model runs.")
            if key == "challenge":
                lim = state["imaging"]["out_of_scope"]
                return ("What most limits this assessment: "
                        + ("; ".join(lim) if lim else "no scope limit was declared")
                        + (". Absent tests: " + ", ".join(gone) if gone else "")
                        + f". Case quality is graded "
                          f"{(state.get('case_quality') or {}).get('grade')}."
                        + (" The agents disagree: " + "; ".join(state["conflicts"])
                           if state.get("conflicts") else ""))
            if key == "findings":
                return ("POCUS detected: "
                        + ("; ".join(f"{f['label']} at {f['confidence']:.2f}" for f in det)
                           if det else "no finding above the tuned threshold")
                        + ". Screened and not seen: "
                        + (", ".join(f["label"] for f in state["imaging"]["findings"]
                                     if not f["detected"]) or "none")
                        + ". "
                        + ("; ".join(state["imaging"]["out_of_scope"]) or ""))
            return answer_free(key)

        def answer_free(q: str) -> str:
            """Answer an arbitrary typed question from the computed assessment.

            This is a router over the record, not a language model. It reads the question for
            what it is asking about and replies with what this encounter actually holds. That
            boundary is the whole point: a fluent paragraph composed here would be a clinical
            opinion with no evidence behind it, which is the failure the rest of the system is
            built to prevent. Everything below is quoted from the state, the escalation
            triggers, the alerts, the retrieved corpus or the validated differential.
            """
            ql = " " + q.lower().strip() + " "

            def has(*words) -> bool:
                return any(w in ql for w in words)

            # ---- a specific measurement by name -------------------------------------
            for k in list(LAB_REFERENCE) + list(VITAL_REFERENCE):
                aliases = {k, k.replace("_", " "), k.replace("_", "-")}
                if k == "spo2":
                    aliases |= {"oxygen", "saturation", "sats", "o2"}
                if k == "hr":
                    aliases |= {"heart rate", "pulse"}
                if k == "sbp":
                    aliases |= {"blood pressure", "systolic"}
                if k == "rr":
                    aliases |= {"respiratory rate", "breathing rate"}
                if k == "temp":
                    aliases |= {"temperature", "fever"}
                if not any(f" {a} " in ql or f" {a}?" in ql or f" {a}," in ql
                           for a in aliases):
                    continue
                src = state.get("labs") or {}
                ref = LAB_REFERENCE.get(k)
                if k in VITAL_REFERENCE:
                    src, ref = state.get("vitals") or {}, VITAL_REFERENCE[k]
                e = src.get(k)
                if e:
                    bounds = (f"reference {ref.get('normal_min', '—')}–"
                              f"{ref.get('normal_max', '—')} {ref.get('unit', '')}").strip()
                    eid = next((x["id"] for x in build_evidence(state)
                                if k in x["text"].lower()), None)
                    return (f"{k} is {e['value']:g} {e['unit']} — flagged {e['flag']}. "
                            f"{bounds}."
                            + (f" It is citable as {eid}." if eid else ""))
                return (f"{k} was NOT measured for this patient. That is not the same as "
                        f"normal: it has no evidence identifier, so nothing in the assessment "
                        f"can argue for or against a diagnosis using it. It appears in the "
                        f"missing-information list and, if it matters for this presentation, "
                        f"in the recommended examinations.")

            # ---- intents --------------------------------------------------------------
            if has("treat", "therapy", "therapeutic", "manage", "give ", "drug", "dose",
                   "medication", "prescri", "fluid", "antibiotic"):
                th = sup["therapeutic"]
                if th["considerations"]:
                    return ("A protocol-backed consideration is available for this "
                            "presentation: "
                            + "; ".join(f"{c['consideration']} (basis: {c['basis']}, "
                                        f"{c['passage']})" for c in th["considerations"])
                            + ". This is decision support, not a treatment instruction, and "
                              "the decision remains yours.")
                return (f"No therapeutic recommendation for this case. {th['status']}. "
                        "Therapeutic suggestions are gated behind a sourced, citable protocol "
                        "in the corpus; with none retrieved the system produces nothing rather "
                        "than drawing on the language model's own training knowledge. It is "
                        "decision support and does not instruct treatment.")
            if has("suggest", "recommend", "next", "investigate", "order", "which test",
                   "what test", "what should i do", "what do you suggest", "work up",
                   "workup", "plan"):
                return answer("next")
            if has("miss", "absent", "not measured", "unavailable", "don't have",
                   "do not have"):
                return answer("missing")
            if has("why", "reason", "because", "how did you", "justif"):
                return answer("why")
            if has("challenge", "wrong", "disagree with you", "sure", "certain",
                   "limitation", "limit"):
                return answer("challenge")
            if has("alert", "critical", "danger", "urgent", "red flag", "worry"):
                al = sup["alerts"]
                if not al:
                    return ("No alert fired. No measured value crossed a configured bound and "
                            "the record shows no structural gap.")
                return ("Alerts, each naming the bound it crossed: "
                        + "; ".join(f"[{a['severity']}] {a['message']}" for a in al)
                        + f". Thresholds v{sup['thresholds_version']}.")
            if has("severity", "priority", "escalat", "how bad", "how serious", "admit",
                   "icu", "dispos"):
                base = answer("why")
                if has("admit", "icu", "dispos"):
                    base += (" Disposition is outside what this system decides: it produces a "
                             "severity, alerts and recommended examinations, not a bed "
                             "decision.")
                return base
            if has("pocus", "ultrasound", "scan", "image", "b-line", "b line", "finding",
                   "detect", "see"):
                return answer("findings")
            if has("differential", "diagnos", "cause", "what could", "what might",
                   "possib"):
                if not diff:
                    return ("No differential was produced for this encounter. "
                            + ("The model backend failed and the answer was withheld; the "
                               "severity and alerts beside it were computed before the model "
                               "ran." if A["origin"] == "failed" else
                               "Generating one needs a 4.9 GB model on a GPU, which is not "
                               "loaded here. Load a benchmark encounter unedited to see one "
                               "recorded from the GPU run."))
                return ("Ranked: " + "; ".join(
                    f"{i}. {d.get('diagnosis')} ({d.get('likelihood')})"
                    for i, d in enumerate(diff, 1))
                    + ". Offered for consideration — this is not a diagnosis.")
            if has("support", "evidence for", "argue for", "in favour", "in favor"):
                return answer("support")
            if has("against", "exclude", "rule out", "contradict"):
                lim = state["imaging"]["out_of_scope"]
                against = "; ".join(
                    f"{d.get('diagnosis')}: " + "; ".join(d.get("contradicting") or ["—"])
                    for d in diff) or "nothing was cited against any entry"
                return (f"Cited against: {against}. What the imaging cannot exclude: "
                        + ("; ".join(lim) if lim else "no scope limit was declared")
                        + ". A finding the module does not model cannot be ruled out by it, "
                          "whatever else the scan shows.")
            if has("summar", "overview", "tell me about", "who is", "recap"):
                return answer("summary")
            if has("confiden", "calibrat", "reliab", "trust", "accurate", "threshold",
                   "cutoff"):
                rel = ((enc.get("report") or {}).get("reliability") or {})
                bits = []
                if rel:
                    bits.append("calibrated" if rel.get("confidence_calibrated")
                                else "confidence is a RAW sigmoid output, not a calibrated "
                                     "probability — the decision boundary is tuned, the number "
                                     "is not")
                    if rel.get("scope"):
                        bits.append(f"scope: {rel['scope']}")
                bits.append(f"alert thresholds are v{sup['thresholds_version']} and are "
                            "configuration for a prototype, not a validated scoring system")
                bits.append("none of this has been validated against patient outcomes")
                return " · ".join(bits).capitalize() + "."
            if has("conflict", "disagree"):
                c = state.get("conflicts") or []
                return ("The agents disagree: " + "; ".join(c) +
                        ". The system surfaces the disagreement rather than resolving it."
                        ) if c else "The agents do not disagree on this encounter."
            if has("protocol", "guideline", "source", "reference", "citation", "corpus",
                   "retriev", "paper"):
                if not A["hits"]:
                    return ("No passage cleared the 0.10 relevance floor for this encounter, "
                            "so the answer is not marked guideline-grounded. Below the floor "
                            "the system reports nothing rather than the best of a bad set.")
                return "Retrieved: " + "; ".join(
                    f"[{h['n']}] {h['topic']} ({h['score']:.2f}) — {h['source']}"
                    for h in A["hits"])
            if has("scenario", "route", "routed", "pathway"):
                return (f"Routed as {sup['scenario']['label']}. Routing selects which "
                        f"modalities are expected and which protocol topic is looked up; it is "
                        f"matched on the presenting complaint by fixed cues, not by a model.")
            if has("quality", "complete", "enough"):
                q_ = state.get("case_quality") or {}
                return (f"Case quality is graded {q_.get('grade')}. "
                        + ("Reasons: " + "; ".join(q_.get("reasons") or []) if q_.get("reasons")
                           else "No quality concern was recorded."))
            if has("evidence", "cite", "identifier"):
                ev = build_evidence(state)
                used = {i for d in diff for i in (d.get("supporting_ids") or [])}
                return (f"{len(ev)} citable fact(s), {len(used)} cited by the model: "
                        + "; ".join(f"{e['id']} {e['text']}" for e in ev))
            if has("how long", "how fast", "time", "latency", "speed"):
                return ("Stage timings for this encounter: "
                        + "; ".join(f"{m['title']} +{m['t'] * 1000:.0f} ms"
                                    for m in A["marks"]))
            if has("vital", "observation"):
                v = state.get("vitals") or {}
                return ("Vitals recorded: " + "; ".join(
                    f"{k} {e['value']:g} {e['unit']} ({e['flag']})" for k, e in v.items())
                    + (f". Never measured: {', '.join(state['missing']['vitals'])}."
                       if state["missing"]["vitals"] else "")) if v else \
                    "No vital sign was recorded for this patient."
            if has("lab", "blood", "biolog"):
                lb = state.get("labs") or {}
                return ("Laboratory values: " + "; ".join(
                    f"{k} {e['value']:g} {e['unit']} ({e['flag']})" for k, e in lb.items())
                    + (f". Never measured: {', '.join(state['missing']['labs'])}."
                       if state["missing"]["labs"] else "")) if lb else \
                    ("No laboratory value was recorded. Not measured is not normal: "
                     + ", ".join(state["missing"]["labs"]) + " all have no identifier.")

            # ---- a diagnosis named in the question -----------------------------------
            for d in diff:
                nm = str(d.get("diagnosis", "")).lower()
                if nm and (nm in ql or any(w in ql for w in nm.split() if len(w) > 5)):
                    return (f"{d.get('diagnosis')} — {d.get('likelihood')} likelihood. "
                            f"Cited for it: "
                            + "; ".join(d.get("supporting") or ["—"])
                            + ". Against: " + "; ".join(d.get("contradicting") or ["—"])
                            + ". Limits: " + "; ".join(d.get("limitations") or ["—"]) + ".")

            # ---- nothing matched -----------------------------------------------------
            return (
                "I could not map that to anything in this encounter's record, and I will not "
                "compose a clinical opinion that has no evidence behind it — that is the "
                "failure this system is built to prevent.\n\n"
                "I can answer, from what was actually computed: any named vital or laboratory "
                "value and whether it was measured at all; what is missing; the alerts and the "
                "bound each crossed; the severity and why; the escalation triggers; what POCUS "
                "saw and what it cannot exclude; the differential and the evidence cited for "
                "and against each entry; the retrieved sources; the scenario routing; case "
                "quality; calibration and thresholds; and the stage timings.\n\n"
                "Ask in those terms and you will get the record, not a guess.")

        C, Sd = st.columns([1.9, 1], gap="large")
        with C:
            st.markdown(
                f"<div class='card' style='padding:0;overflow:hidden'>"
                f"<div style='padding:20px 26px;border-bottom:1px solid {RULE};display:flex;"
                f"justify-content:space-between;align-items:center;gap:20px;flex-wrap:wrap'>"
                f"<div style='display:flex;gap:13px;align-items:center'>"
                f"<span style='width:40px;height:40px;border-radius:12px;background:{IRIS};"
                f"color:#fff;display:flex;align-items:center;justify-content:center;"
                f"font-size:18px'>✨</span><div>"
                f"<div style='font-weight:800;font-size:16px'>Clinical assistant</div>"
                f"<div style='font-size:13px;color:{MUTED}'>Answers assembled from the "
                f"computed assessment.</div></div></div>"
                f"<div class='tag t-lime'>● {enc['age']}{enc['sex'][0]} — "
                f"{E(enc['complaint'] or 'no complaint')}</div></div>"
                f"<div style='padding:24px 26px;display:flex;flex-direction:column;gap:16px'>"
                + "".join(
                    f"<div style='display:flex;justify-content:"
                    f"{'flex-end' if m['r'] == 'd' else 'flex-start'}'>"
                    f"<div class='{'bub-d' if m['r'] == 'd' else 'bub-a'}'>"
                    f"{E(answer(m['t'][4:]) if m['t'].startswith('ASK:') else m['t'])}"
                    f"</div></div>" for m in st.session_state["msgs"])
                + "</div></div>", unsafe_allow_html=True)

            st.caption("Suggestions")
            sg = [("What findings support the leading entry?", "support"),
                  ("What information is missing?", "missing"),
                  ("What did POCUS actually see?", "findings"),
                  ("Summarise this patient.", "summary"),
                  ("What should I investigate next?", "next"),
                  ("Challenge this assessment.", "challenge")]
            cols = st.columns(3)
            for i, (label, key) in enumerate(sg):
                if cols[i % 3].button(label, key=f"sg{i}", use_container_width=True):
                    st.session_state["msgs"] += [{"r": "d", "t": label},
                                                 {"r": "a", "t": f"ASK:{key}"}]
                    st.rerun()

            with st.form("ask", clear_on_submit=True):
                fc = st.columns([5, 1])
                q = fc[0].text_input("Ask", label_visibility="collapsed",
                                     placeholder="Ask about this patient…")
                if fc[1].form_submit_button("➤", type="primary", use_container_width=True) \
                        and q.strip():
                    # The typed question is answered from this encounter's record. It is
                    # stored resolved rather than as a key, because the answer belongs to the
                    # question that was asked.
                    st.session_state["msgs"] += [
                        {"r": "d", "t": q.strip()},
                        {"r": "a", "t": answer_free(q.strip())}]
                    st.rerun()

        with Sd:
            st.markdown(
                f"<div class='card'><div class='kicker'>Patient context</div>"
                f"<p style='margin:10px 0 0;font-size:13.5px;color:{MUTED}'>The assistant "
                f"keeps everything entered during this encounter, so you don't need to repeat "
                f"the patient information.</p>"
                f"<div style='margin-top:16px;display:flex;flex-direction:column;gap:10px;"
                f"font-size:13.5px'>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:{MUTED}'>Vitals</span><span style='font-weight:700'>"
                f"{len(state.get('vitals') or {})} recorded</span></div>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:{MUTED}'>Laboratory</span><span style='font-weight:700'>"
                f"{len(state.get('labs') or {})} of {len(LAB_REFERENCE)}</span></div>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:{MUTED}'>POCUS</span><span style='font-weight:700'>"
                f"{E(enc['organ'])}</span></div>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:{MUTED}'>Evidence</span><span style='font-weight:700'>"
                f"{len(build_evidence(state))} citable</span></div>"
                f"<div style='display:flex;justify-content:space-between'>"
                f"<span style='color:{MUTED}'>Assessment</span><span style='font-weight:700'>"
                f"{A['started']}</span></div></div></div>"
                f"<div style='background:{BG};border-radius:20px;padding:22px 24px'>"
                f"<div style='font-weight:800;font-size:15px'>Reminder</div>"
                f"<p style='margin:8px 0 0;font-size:13.5px;color:#463A6B'>The assistant "
                f"provides clinical decision support. It does not make a diagnosis, and "
                f"responsibility for the decision rests with you.</p></div>",
                unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════ 10 · TIMELINE
elif SCREEN == "timeline":
    if not need_encounter():
        st.markdown(f"<h1 class='h1' style='margin-top:0'>Patient clinical timeline</h1>"
                    f"<p class='lede'>{E(enc['name'] or 'Unnamed patient')} · started "
                    f"{A['started']} · elapsed times are the real cost of each stage on this "
                    f"machine.</p>", unsafe_allow_html=True)
        st.write("")
        items = "".join(
            f"<li class='tl'><span style='font-size:13.5px;color:{FAINT};font-weight:700;"
            f"padding-top:2px'>+{m['t'] * 1000:.0f} ms</span>"
            f"<span style='width:13px;height:13px;border-radius:50%;margin-top:5px;"
            f"background:{IRIS if m['hl'] else '#fff'};border:3px solid "
            f"{'#D9CDFF' if m['hl'] else BORDER2};display:block'></span><div>"
            f"<div style='font-weight:{800 if m['hl'] else 600};font-size:15.5px;"
            f"color:{NAVY if m['hl'] else INK}'>{E(m['title'])}</div>"
            f"<div style='font-size:13.5px;color:{MUTED};margin-top:3px'>{E(m['detail'])}"
            f"</div></div></li>" for m in A["marks"])
        st.markdown(f"<div class='card rise' style='padding:30px 34px'>"
                    f"<ol style='list-style:none;margin:0;padding:0'>{items}</ol></div>",
                    unsafe_allow_html=True)

        st.markdown(
            f"<div class='band band-vio'>"
            f"<div style='font-size:12.5px;letter-spacing:.11em;text-transform:uppercase;"
            f"color:{V700};font-weight:800'>Order matters</div>"
            f"<p style='margin:10px 0 0;font-size:15.5px'>Escalation and decision support are "
            f"stamped above <b>before</b> retrieval and the model. That ordering is the safety "
            f"property: the model can fail, and the severity on this encounter is already "
            f"computed and already attached.</p></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════ 11 · REPORT
elif SCREEN == "report":
    if not need_encounter():
        state, sup, rep = A["state"], A["support"], A["report"]
        txt = render_report(rep)
        top = st.columns([3, 1])
        top[0].markdown("<h1 class='h1' style='margin-top:0'>Clinical assessment report</h1>",
                        unsafe_allow_html=True)
        top[1].download_button("📄 Export report", txt, type="primary",
                               use_container_width=True,
                               file_name=f"{rep['encounter_id']}_report.txt")

        vit = "".join(f"<div>{k} {v['value']:g} {v['unit']} — {v['flag']}</div>"
                      for k, v in (state.get("vitals") or {}).items()) or "<div>none</div>"
        lab = "".join(f"<div>{k} {v['value']:g} {v['unit']} — {v['flag']}</div>"
                      for k, v in (state.get("labs") or {}).items()) or "<div>none</div>"
        det = [f for f in state["imaging"]["findings"] if f["detected"]]
        pocus = "".join(f"<div>{E(enc['organ'])} ultrasound: {E(f['label'])} detected "
                        f"({f['confidence']:.2f})</div>" for f in det) or \
            f"<div>{E(enc['organ'])} ultrasound: no finding above threshold</div>"
        diff = (A["result"].get("differential") or {}).get("differential") or []
        dxl = "".join(f"<li>{E(str(d.get('diagnosis')))} — {E(str(d.get('likelihood')))} "
                      f"likelihood</li>" for d in diff) or \
            "<li>No differential generated for this encounter.</li>"
        gone = state["missing"]["labs"] + state["missing"]["vitals"]
        lims = "".join(f"<li>{E(x)}</li>" for x in state["imaging"]["out_of_scope"])
        if gone:
            lims += (f"<li>{E(', '.join(gone))} were not obtained at the time of "
                     f"assessment</li>")
        lims += "<li>Not measured does not mean normal</li>"
        nxt = (sup["additional_examinations"][0]["exam"]
               if sup["additional_examinations"]
               else "Physician review of the current findings.")

        st.markdown(
            f"<div class='rep rise'>"
            f"<header style='display:flex;justify-content:space-between;gap:24px;"
            f"padding-bottom:20px;border-bottom:2px solid {BORDER};flex-wrap:wrap'>"
            f"<div><div style='font-weight:800;font-size:20px'>Clinical assessment</div>"
            f"<div style='font-size:13.5px;color:{MUTED};margin-top:4px'>Emergency Department "
            f"· {E(rep['generated_at'][:16].replace('T', ' '))}</div></div>"
            f"<div style='text-align:right;font-size:13.5px;color:{MUTED}'>"
            f"{E(enc['name'] or 'Unnamed patient')} · {enc['age']} · {E(enc['sex'])}<br>"
            f"Encounter {E(rep['encounter_id'])}</div></header>"
            f"<section style='margin-top:26px'><div class='kicker'>Chief complaint</div>"
            f"<p style='margin:8px 0 0;font-size:16px'>{E(enc['complaint'] or '—')}</p></section>"
            f"<section style='margin-top:24px'><div class='kicker'>Clinical summary</div>"
            f"<p style='margin:8px 0 0;font-size:16px;line-height:1.6'>{E(rep['conclusion'])}"
            f"</p></section>"
            f"<section style='margin-top:24px'><div class='kicker'>Clinical findings</div>"
            f"<div style='margin-top:12px' class='grid2'>"
            f"<div><div style='font-weight:700;margin-bottom:8px'>Vital signs</div>"
            f"<div style='display:flex;flex-direction:column;gap:6px;font-size:15.5px'>{vit}"
            f"</div></div>"
            f"<div><div style='font-weight:700;margin-bottom:8px'>POCUS &amp; laboratory</div>"
            f"<div style='display:flex;flex-direction:column;gap:6px;font-size:15.5px'>"
            f"{pocus}{lab}</div></div></div></section>"
            f"<section style='margin-top:24px'><div class='kicker'>Differential</div>"
            f"<ol style='margin:10px 0 0;padding-left:22px;font-size:16px'>{dxl}</ol></section>"
            f"<section style='margin-top:24px;background:#FFF8EC;border-radius:14px;"
            f"padding:18px 22px'><div style='font-size:12px;letter-spacing:.11em;"
            f"text-transform:uppercase;color:{AMB_D};font-weight:800'>Important limitations"
            f"</div><ul style='margin:10px 0 0;padding-left:20px;font-size:15.5px;"
            f"color:#6A4C0C'>{lims}</ul></section>"
            f"<section style='margin-top:24px'><div class='kicker'>Recommended next step</div>"
            f"<p style='margin:8px 0 0;font-size:18px;font-weight:700;line-height:1.4'>"
            f"{E(nxt)}</p></section>"
            f"<footer style='margin-top:32px;padding-top:18px;border-top:1px solid {BORDER};"
            f"font-size:13.5px;color:{FAINT}'>AI-assisted clinical assessment. Requires "
            f"physician review. This report is clinical decision support and does not "
            f"constitute a diagnosis. Not a diagnostic device; never validated against patient "
            f"outcomes.</footer></div>", unsafe_allow_html=True)

        with st.expander("Full archived report"):
            st.code(txt, language=None)

# ══════════════════════════════════════════════════════════════ 12 · HISTORY
elif SCREEN == "history":
    st.markdown(f"<h1 class='h1' style='margin-top:0'>Benchmark encounters</h1>"
                f"<p class='lede'>{len(ROSTER)} synthetic encounters · "
                f"{len(st.session_state['records'])} analysed this session. No clinical ground "
                f"truth — these exercise the safety layer, not diagnostic accuracy.</p>",
                unsafe_allow_html=True)
    st.write("")
    cols = st.columns(3, gap="medium")
    for i, r in enumerate(ROSTER):
        top = {"HIGH": RED, "MODERATE": AMB, "LOW": GRN}[r["severity"]]
        with cols[i % 3]:
            st.markdown(
                f"<div class='card lift rise-2' style='border-top:4px solid {top};"
                f"margin-bottom:10px'>"
                f"<span class='tag {r['tag'][0]}'>{r['tag'][1]}</span>"
                f"<div style='font-weight:800;font-size:17px;margin-top:6px'>"
                f"{E(r['name'])} · {r['age']}{r['sex'][0]}</div>"
                f"<div style='font-size:14px;color:{MUTED};margin-top:4px'>"
                f"{E(r['complaint'])}</div>"
                f"<div style='font-size:13px;color:{FAINT};margin-top:8px'>"
                f"{r['severity']} severity · {r['alerts']} alert(s)</div></div>",
                unsafe_allow_html=True)
            if st.button("Open encounter", key=f"hx{i}", use_container_width=True):
                st.session_state["_preset"] = None
                st.session_state["_preset_sel"] = r["label"]
                go("intake")
                st.rerun()
