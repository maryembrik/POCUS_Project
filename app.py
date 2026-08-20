"""POCUS Emergency Assistant -- multimodal AI-assisted clinical decision support.

    streamlit run app.py

The interface follows the Broadsheet design: editorial typography, a paper ground, and colour
used for meaning rather than decoration. Where the design's illustrative copy described
behaviour this system does not have, the behaviour won and the copy was corrected -- a screen
claiming a validator that "rewrites the output" or "caps confidence" would misdescribe the one
safety property the project rests on. This system delivers with warnings or withholds, and
`test_python_never_rewrites_the_models_answer` holds that line.

A clinician enters a patient, not a configuration. The assistant selects the pipeline from the
presentation, and for the lung it analyses the uploaded image itself rather than asking what
the image showed.

What runs here and what does not, stated on screen wherever it matters:

    LIVE      lung inference, on the fold checkpoints in this repository, on CPU, in a few
              seconds per image. The clinician uploads a scan and the module reports its own
              findings; nothing about the findings is typed in.

    LIVE      the clinical state, conflicts, what is missing, evidence identifiers, the
              escalation decision, severity, alerts, recommended examinations, scenario
              routing, retrieval, and the report. Deterministic Python, no GPU.

    LIVE      the failure demonstration. Simulating a model failure calls reason() with a
              backend that raises; the escalation surviving is executed, not mocked.

    RECORDED  the differential, and only for an unmodified example encounter. Generating one
              needs a 4.9 GB model on a GPU, so an edited or custom encounter says so rather
              than presenting a recording that no longer matches the record on screen.

    ABSENT    cardiac and gallbladder weights are not in this deployment. Those organs return
              `not supported` in the ordinary report format rather than a guess.
"""
from __future__ import annotations

import json
import os
import sys
import time
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
from src.agents.ultrasound.agent import LUNG_FINDINGS  # noqa: E402
from src.agents.ultrasound.agent import available as organs_available  # noqa: E402
from src.agents.ultrasound.agent import ultrasound_agent  # noqa: E402

FROZEN = ROOT / "models" / "clinical_reasoning_v4_final" / "results.json"

# Written out per measurement rather than derived from the reference range: scaling a range
# gave a pH slider from 0 to 44.7 and an SpO2 that reached 160%.
VITAL_RANGE = {"hr": (30.0, 200.0, 1.0), "sbp": (50.0, 220.0, 1.0), "rr": (6.0, 45.0, 1.0),
               "spo2": (70.0, 100.0, 1.0), "temp": (34.0, 42.0, 0.1)}
LAB_RANGE = {"troponin": (0.0, 500.0, 1.0), "bnp": (0.0, 2000.0, 10.0),
             "d_dimer": (0.0, 5000.0, 50.0), "lactate": (0.0, 12.0, 0.1),
             "crp": (0.0, 300.0, 1.0), "wbc": (0.0, 30.0, 0.1),
             "creatinine": (0.0, 600.0, 5.0), "ph": (6.8, 7.8, 0.01)}

BLANK = dict(age=60, sex="Female", complaint="", tier="medium", tconf=0.70, organ="Lung",
             findings={}, vitals={}, labs={}, key=None, blurb="", tags=[])

PRESETS = {
    "Blank — enter a new patient": dict(BLANK, letter="—", title="Blank"),
    "Acute dyspnoea, key labs missing": dict(
        letter="A", title="Acute dyspnoea", age=74, sex="Female",
        complaint="acute breathlessness", tier="high", tconf=0.79, organ="Lung",
        findings={"b_lines": 0.86}, vitals={"hr": 118.0, "rr": 24.0, "spo2": 90.0}, labs={},
        key="missing",
        blurb="Positive lung POCUS with the key laboratory values unavailable. The absent "
              "tests get no evidence identifier, so nothing can be argued from them.",
        tags=[("accent-2", "Escalation HIGH"), ("outline", "B-lines 0.86")]),
    "Evidence agrees, record complete": dict(
        letter="B", title="Evidence agrees", age=71, sex="Female", complaint="orthopnoea",
        tier="high", tconf=0.88, organ="Lung", findings={"b_lines": 0.92},
        vitals={"hr": 122.0, "rr": 28.0, "spo2": 88.0},
        labs={"troponin": 62.0, "bnp": 890.0, "lactate": 2.6}, key="concordant",
        blurb="Imaging, vitals and laboratory values agree and the record is complete. The "
              "case the safety layer should not obstruct.",
        tags=[("accent", "Answered directly"), ("outline", "B-lines 0.92")]),
    "Triage and imaging disagree": dict(
        letter="C", title="Conflicting evidence", age=68, sex="Male",
        complaint="mild chest discomfort", tier="low", tconf=0.88, organ="Heart",
        findings={"severe dysfunction": 0.74},
        vitals={"hr": 82.0, "sbp": 128.0, "spo2": 96.0},
        labs={"troponin": 5.0, "lactate": 1.1}, key="conflict",
        blurb="Triage says low urgency; the scan reports severe dysfunction. The disagreement "
              "is surfaced, not resolved by picking the more confident agent.",
        tags=[("accent-2", "Agent conflict")]),
    "Everything negative": dict(
        letter="D", title="Reassuring case", age=44, sex="Female",
        complaint="mild breathlessness", tier="low", tconf=0.80, organ="Lung", findings={},
        vitals={"hr": 78.0, "rr": 16.0, "spo2": 97.0},
        labs={"troponin": 4.0, "lactate": 1.0}, key="reassuring",
        blurb="Normal vitals and every screened finding negative — which still escalates, "
              "because the module has no healthy class and cannot exclude pneumothorax.",
        tags=[("neutral", "All findings negative")]),
    "Undifferentiated shock": dict(
        letter="E", title="Undifferentiated shock", age=61, sex="Male",
        complaint="hypotension and collapse", tier="high", tconf=0.85, organ="Heart",
        findings={"severe dysfunction": 0.68},
        vitals={"hr": 132.0, "sbp": 78.0, "spo2": 92.0}, labs={"lactate": 5.2}, key=None,
        blurb="The one presentation with a sourced protocol in the corpus, so the only one "
              "that can carry a therapeutic consideration.",
        tags=[("accent-2", "Protocol available")]),
}

st.set_page_config(page_title="POCUS Emergency Assistant", page_icon="🩺", layout="wide")

# ────────────────────────────────────────────────────────── Broadsheet design tokens
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');
:root{
  --bg:#f3f2f2; --surface:#eae9e9; --text:#201e1d;
  --accent:#0088b0; --accent-100:#e9f8ff; --accent-600:#1186ac; --accent-700:#006786;
  --accent-800:#004961;
  --accent-2:#d6006c; --accent-2-100:#fff1f4; --accent-2-600:#d82071; --accent-2-700:#aa0b56;
  --accent-2-800:#790e3d;
  --n-100:#f8f4f4; --n-200:#eae7e7; --n-300:#d7d3d3; --n-400:#bab6b6;
  --n-600:#7d7979; --n-700:#605d5d; --n-800:#444141;
  --divider:rgba(32,30,29,.16);
  --serif:"Source Serif 4",Georgia,"Times New Roman",serif;
  --shadow-sm:0 1px 2px rgba(45,43,43,.14);
  --shadow-md:0 3px 10px rgba(45,43,43,.16);
}
html,body,.stApp,[class*="css"]{background:var(--bg)!important;color:var(--text)!important;
  font-family:var(--serif)!important;}
.block-container{padding-top:1.2rem;max-width:1560px;}
h1,h2,h3,h4,h5{font-family:var(--serif)!important;font-weight:600!important;
  letter-spacing:-.015em;color:var(--text)!important;}
p,label,li,div[data-testid="stMarkdownContainer"]{font-family:var(--serif)!important;}

/* masthead */
.masthead{display:flex;align-items:flex-end;justify-content:space-between;gap:36px;
  padding-bottom:12px;border-bottom:3px solid var(--text);margin-bottom:14px;}
.masthead h1{margin:0;font-size:26px;line-height:1.15;}
.ready{font-size:12px;letter-spacing:.11em;text-transform:uppercase;color:var(--accent-700);}
.sub{font-size:13.5px;color:var(--n-700);margin-top:3px;}
.modebox{text-align:right;font-size:12.5px;color:var(--n-700);white-space:nowrap;}
.modebox b{font-family:var(--serif);color:var(--text);font-size:14.5px;}

/* editorial furniture */
.kicker{font-size:11.5px;letter-spacing:.14em;text-transform:uppercase;color:var(--n-700);
  margin-bottom:8px;}
.kicker-2{color:var(--accent-2-700);}
.kicker-1{color:var(--accent-700);}
.display{font-family:var(--serif);font-weight:600;font-size:28px;line-height:1.1;}
.display-lg{font-family:var(--serif);font-weight:600;font-size:42px;line-height:1;}
.lead{font-size:15.5px;line-height:1.6;max-width:68ch;color:var(--n-800);}
.paper{background:var(--surface);border-radius:2px;padding:20px 22px;
  box-shadow:var(--shadow-sm);margin-bottom:12px;}
.band{border-left:4px solid;border-radius:2px;padding:18px 22px;margin:4px 0 16px;}
.band-crit{background:var(--accent-2-100);border-color:var(--accent-2-600);}
.band-warn{background:var(--n-100);border-color:var(--n-600);}
.band-ok{background:var(--accent-100);border-color:var(--accent-600);}
.rule{background:var(--n-200);border-radius:2px;padding:13px 15px;
  font-family:ui-monospace,Consolas,Menlo,monospace;font-size:12.5px;line-height:1.75;}
.tag{display:inline-block;font-size:11px;padding:3px 9px;border-radius:1.5px;
  margin:0 6px 6px 0;}
.t-accent{background:var(--accent-100);color:var(--accent-800);}
.t-accent-2{background:var(--accent-2-100);color:var(--accent-2-800);}
.t-neutral{background:var(--n-100);color:var(--n-800);}
.t-outline{border:1px solid var(--accent);color:var(--accent-700);}
.eid{font-family:ui-monospace,Consolas,Menlo,monospace;font-size:12.5px;
  color:var(--accent-700);}
.muted{color:var(--n-700);font-size:13.5px;line-height:1.55;}
.row{padding:6px 0;border-bottom:1px solid var(--divider);font-size:14.5px;}
.row-gone{color:var(--n-600);}
.step{display:grid;grid-template-columns:120px 1fr;gap:24px;padding:18px 0;
  border-top:1px solid var(--divider);}
.stepno{font-family:ui-monospace,Consolas,Menlo,monospace;font-size:12px;color:var(--n-700);}
.stepttl{font-family:var(--serif);font-weight:600;font-size:18px;}
.wf{font-size:14.5px;padding:2px 0;}
.wf-todo{color:var(--n-600);}
.wf .mk{color:var(--accent-700);margin-right:8px;}
hr{border-color:var(--divider)!important;}

/* streamlit chrome */
.stTabs [data-baseweb="tab-list"]{gap:24px;border-bottom:3px solid var(--text);}
.stTabs [data-baseweb="tab"]{font-family:var(--serif);font-weight:600;font-size:15px;
  color:var(--n-700);padding:0 0 10px;background:transparent;}
.stTabs [aria-selected="true"]{color:var(--text)!important;
  border-bottom:3px solid var(--accent-2-600);margin-bottom:-3px;}
.stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{
  font-family:var(--serif)!important;font-weight:600;border-radius:2px;
  border:1px solid var(--n-400);background:var(--n-100);color:var(--text);}
.stButton>button[kind="primary"],.stFormSubmitButton>button[kind="primary"]{
  background:var(--accent-2-600);border-color:var(--accent-2-600);color:#fff;}
section[data-testid="stSidebar"]{background:var(--surface);}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}


def tags(items) -> str:
    return "".join(f"<span class='tag t-{c}'>{t}</span>" for c, t in items)


def band(kind: str, title: str, lines) -> None:
    body = "".join(f"<div style='margin-top:7px;font-size:15px'>{x}</div>" for x in lines)
    st.markdown(f"<div class='band band-{kind}'><span class='kicker "
                f"{'kicker-2' if kind == 'crit' else 'kicker-1' if kind == 'ok' else ''}' "
                f"style='margin:0'>{title}</span>{body}</div>", unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════ analysis
def build_state(enc: dict):
    """Assemble the clinical state from the encounter.

    A module report supplied by the perception agent is used as it stands. Where an organ was
    selected but never scanned, it is recorded as requested-and-not-assessed -- a gap in the
    record, which is not the same as a negative result and must not read as one.
    """
    reports = {}
    organ = enc["organ"]
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


def analyse(enc: dict, break_model: bool):
    state = build_state(enc)
    esc = escalation_decision(state)
    support = decision_support(state, esc)
    hits = Retriever().for_state(state)

    if break_model:
        return (state, esc, support, hits,
                reason(state, llm_fn=FailingBackend("out of memory"), max_revisions=1),
                "failed")

    rec = (load_frozen().get("none") or {}).get(enc.get("frozen_key") or "")
    if rec:
        return state, esc, support, hits, dict(rec, decision_support=support), "recorded"

    result = reason(state, llm_fn=None)          # state + escalation, no generation
    result["decision_support"] = support
    return state, esc, support, hits, result, "not_generated"


# ═════════════════════════════════════════════════════════════════════ sidebar
st.sidebar.markdown("### 🩺 POCUS Emergency Assistant")
st.sidebar.caption("Multimodal AI-assisted clinical decision support for emergency ultrasound.")
st.sidebar.divider()
break_model = st.sidebar.toggle(
    "Simulate a model failure", value=False,
    help="Runs the pipeline for real with a backend that raises on every call. The escalation "
         "surviving is executed, not narrated.")
st.sidebar.divider()
_avail = organs_available()
st.sidebar.markdown("**Perception modules**")
for _o, _ok in _avail.items():
    st.sidebar.markdown(f"<div class='muted'>{'●' if _ok else '○'} {_o} — "
                        f"{'model loaded' if _ok else 'weights absent'}</div>",
                        unsafe_allow_html=True)
st.sidebar.divider()
st.sidebar.caption("**Not a diagnostic device.** Synthetic cases, no clinical ground truth, "
                   "never validated against patient outcomes.")

enc = st.session_state.get("encounter")

# ═════════════════════════════════════════════════════════════════════ masthead
if enc:
    state, esc, support, hits, result, origin = analyse(enc, break_model)
    sev = support["severity"]
    report = build_report(state, result, support)
    report_text = render_report(report)
    _line = (f"Encounter {report['encounter_id']} · Emergency Department · "
             f"{enc['age']} y/o {enc['sex'].lower()}")
    _mode = ("Recorded differential" if origin == "recorded" else
             "Failure demonstration" if origin == "failed" else "Computed on this machine")
else:
    state = esc = support = hits = result = report = None
    origin, report_text = "none", ""
    _line = "No encounter loaded"
    _mode = "Awaiting intake"

st.markdown(f"""
<div class='masthead'>
  <div>
    <h1>POCUS&#8202;—&#8202;Emergency Clinical Decision Support
      <span class='ready'>&nbsp;&nbsp;● System ready</span></h1>
    <div class='sub'>{_line}</div>
  </div>
  <div class='modebox'><b>{_mode}</b><div>{'lung inference runs here on CPU'
      if _avail['lung'] else 'no perception weights in this deployment'}</div></div>
</div>
""", unsafe_allow_html=True)

T = st.tabs(["Encounter", "Assessment", "Reasoning chain", "Evidence & sources",
             "Clinical report", "Architecture"])

# ═════════════════════════════════════════════════════════════ 1. Encounter
with T[0]:
    st.markdown("<div class='kicker kicker-2'>Intake</div>"
                "<div class='display' style='font-size:34px'>New encounter</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='lead'>Enter the patient and, for the lung, upload the scan. The "
                "module analyses the image and reports its own findings — nothing about what "
                "the image shows is typed in. The assistant selects the pipeline from the "
                "presentation; the agents are not configured by hand.</p>",
                unsafe_allow_html=True)

    preset_name = st.selectbox("Start from an example encounter", list(PRESETS),
                               key="_preset_sel")
    P = PRESETS[preset_name]

    if P["blurb"]:
        st.markdown(f"<div class='paper' style='border-top:3px solid var(--accent-2-600)'>"
                    f"<div class='kicker'>Case {P['letter']}</div>"
                    f"<div class='display' style='font-size:21px'>{P['title']}</div>"
                    f"<p style='font-size:14.5px;margin:9px 0 10px'>{P['blurb']}</p>"
                    f"{tags(P['tags'])}</div>", unsafe_allow_html=True)

    # On a preset change the values are WRITTEN into session state and the widgets below read
    # them, rather than the widgets being given a `value=` default. Deleting a widget's key was
    # tried first and does not reset it: Streamlit re-registers the widget with the value the
    # browser still holds, so choosing a preset left every field exactly as it was.
    if st.session_state.get("_preset") != preset_name:
        for f in LUNG_FINDINGS:
            st.session_state[f"f_{f}"] = f in P["findings"]
            st.session_state[f"c_{f}"] = float(P["findings"].get(f, 0.05))
        st.session_state["f_sd"] = "severe dysfunction" in P["findings"]
        st.session_state["c_sd"] = float(P["findings"].get("severe dysfunction", 0.20))
        for k, ref in VITAL_REFERENCE.items():
            lo, hi, _ = VITAL_RANGE[k]
            st.session_state[f"v_{k}"] = k in P["vitals"]
            st.session_state[f"vv_{k}"] = min(max(float(P["vitals"].get(
                k, (ref["normal_min"] + ref["normal_max"]) / 2)), lo), hi)
        for k, ref in LAB_REFERENCE.items():
            lo, hi, _ = LAB_RANGE[k]
            st.session_state[f"l_{k}"] = k in P["labs"]
            st.session_state[f"ll_{k}"] = min(max(float(P["labs"].get(
                k, ref.get("normal_max", 1) * 0.5)), lo), hi)
        st.session_state["_age"] = int(P["age"])
        st.session_state["_sex"] = P["sex"]
        st.session_state["_complaint"] = P["complaint"]
        st.session_state["_tier"] = P["tier"]
        st.session_state["_tconf"] = float(P["tconf"])
        st.session_state["_organ"] = P["organ"]
        st.session_state["_preset"] = preset_name
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)
    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("<div class='kicker'>Patient</div>", unsafe_allow_html=True)
        a, b = st.columns(2)
        age = a.number_input("Age", 0, 120, key="_age")
        sex = b.selectbox("Sex", ["Female", "Male"], key="_sex")
        complaint = st.text_input("Chief complaint", key="_complaint",
                                  placeholder="e.g. acute breathlessness")

        st.markdown("<div class='kicker' style='margin-top:18px'>Triage</div>",
                    unsafe_allow_html=True)
        tier = st.select_slider("Assessed urgency", ["low", "medium", "high"], key="_tier")
        tconf = st.slider("Triage confidence", 0.5, 1.0, step=0.01, key="_tconf")

        st.markdown("<div class='kicker' style='margin-top:18px'>Vitals</div>",
                    unsafe_allow_html=True)
        st.caption("Untick a measurement to record it as **never measured** — which is not the "
                   "same as normal.")
        vitals = {}
        for k, ref in VITAL_REFERENCE.items():
            c1, c2 = st.columns([1, 2])
            on = c1.checkbox(f"{k} ({ref['unit']})", key=f"v_{k}")
            lo, hi, step = VITAL_RANGE[k]
            val = c2.slider(" ", lo, hi, step=step, key=f"vv_{k}",
                            label_visibility="collapsed")
            if on:
                vitals[k] = val

    with right:
        st.markdown("<div class='kicker'>Ultrasound</div>", unsafe_allow_html=True)
        organ = st.selectbox("Organ examined", ["Lung", "Heart", "Not performed"],
                             key="_organ")

        up = None
        analysed = None
        if organ == "Lung":
            up = st.file_uploader("Upload the lung scan — the model reads it",
                                  type=["png", "jpg", "jpeg", "bmp"])
            if up is not None and _avail["lung"]:
                st.image(up, width=230)
                if st.session_state.get("_img_name") != up.name:
                    from PIL import Image
                    import numpy as np
                    t0 = time.time()
                    with st.spinner("Analysing the image…"):
                        rep = ultrasound_agent(
                            "lung", image=np.array(Image.open(up).convert("L")))
                    st.session_state["_img_name"] = up.name
                    st.session_state["_img_rep"] = rep
                    st.session_state["_img_secs"] = time.time() - t0
                analysed = st.session_state.get("_img_rep")
            elif up is not None:
                st.image(up, width=230)
                st.warning("No lung checkpoint in this deployment, so the image cannot be "
                           "analysed. Enter below what the module reported.")

        if analysed is not None:
            secs = st.session_state.get("_img_secs", 0.0)
            st.markdown(f"<span class='tag t-accent'>ANALYSED BY THE MODEL</span> "
                        f"<span class='muted'>{analysed['model']} · {secs:.1f}s on CPU</span>",
                        unsafe_allow_html=True)
            for f in analysed["findings"]:
                st.markdown(f"<div class='row'>● <b>{f['label']}</b> "
                            f"<span class='muted'>· {f['confidence']:.2f}</span></div>",
                            unsafe_allow_html=True)
            for f in analysed["not_detected"]:
                st.markdown(f"<div class='row row-gone'>○ {f['label']} — screened for and not "
                            f"seen ({f['confidence']:.2f})</div>", unsafe_allow_html=True)
            q = analysed["quality"]
            st.caption(
                "Operating points " +
                ", ".join(f"{k.replace('_', ' ')} {v:.2f}"
                          for k, v in q.get("thresholds", {}).items()) +
                f" — the values the training run tuned, read back from "
                f"`{q.get('thresholds_source')}`.")
            if not analysed["reliability"].get("confidence_calibrated"):
                st.caption("Confidence is a raw sigmoid output, not a calibrated probability. "
                           "The Platt calibrators were never exported from the training "
                           "notebook, so the decision boundary is tuned but the number is "
                           "not — and the report says so rather than implying a probability.")
            findings = {}
        else:
            findings = {}
            if organ == "Lung":
                st.caption("No image analysed. Enter what the module reported — unticked means "
                           "**screened and not seen**, which is itself an observation.")
                for f in LUNG_FINDINGS:
                    c1, c2 = st.columns([1, 2])
                    on = c1.checkbox(f.replace("_", " "), key=f"f_{f}")
                    conf = c2.slider(" ", 0.0, 1.0, step=0.01, key=f"c_{f}",
                                     label_visibility="collapsed")
                    findings[f] = conf if on else -conf
            elif organ == "Heart":
                st.caption("Cardiac weights are not in this deployment, so the module cannot "
                           "read an image here. Enter what it reported.")
                c1, c2 = st.columns([1, 2])
                on = c1.checkbox("severe dysfunction", key="f_sd")
                conf = c2.slider(" ", 0.0, 1.0, step=0.01, key="c_sd",
                                 label_visibility="collapsed")
                findings["severe dysfunction"] = conf if on else -conf

        st.markdown("<div class='kicker' style='margin-top:18px'>Laboratory results</div>",
                    unsafe_allow_html=True)
        st.caption("Untick to record as **not measured**.")
        labs = {}
        for k, ref in LAB_REFERENCE.items():
            c1, c2 = st.columns([1, 2])
            unit = f" ({ref['unit']})" if ref.get("unit") else ""
            on = c1.checkbox(f"{k}{unit}", key=f"l_{k}")
            lo, hi, step = LAB_RANGE[k]
            val = c2.slider(" ", lo, hi, step=step, key=f"ll_{k}",
                            label_visibility="collapsed")
            if on:
                labs[k] = val

    st.markdown("<hr>", unsafe_allow_html=True)
    if st.button("Analyse this encounter", type="primary", use_container_width=True):
        submitted = dict(age=int(age), sex=sex, complaint=complaint, tier=tier,
                         tconf=float(tconf), organ=organ, findings=findings, vitals=vitals,
                         labs=labs, report=analysed, not_assessed=[])
        # A recorded differential belongs to a specific record. If the clinician has edited any
        # field, or the image was analysed here, the recording no longer describes what is on
        # screen -- so it is dropped rather than shown beside a record it does not match.
        same = (analysed is None and P["key"] and all(
            submitted[f] == P[f] for f in
            ("age", "sex", "complaint", "tier", "tconf", "organ", "vitals", "labs")) and
            {k: v for k, v in findings.items() if v > 0} == P["findings"])
        submitted["frozen_key"] = P["key"] if same else None
        submitted["id"] = f"ENC-{P['letter']}" if same else "ENC-LIVE"
        st.session_state["encounter"] = submitted
        st.rerun()

    if enc:
        st.caption("An encounter is loaded — open **Assessment** to read it.")

# ═════════════════════════════════════════════════════════════ 2. Assessment
with T[1]:
    if not enc:
        st.info("No encounter yet. Open **Encounter**, enter a patient, and select "
                "**Analyse this encounter**.")
    else:
        side, main = st.columns([1, 3.2], gap="large")

        with side:
            st.markdown(f"<div class='kicker'>Patient</div>"
                        f"<div class='display' style='font-size:25px'>{enc['age']} y/o "
                        f"{enc['sex'].lower()}</div>"
                        f"<div style='margin-top:6px;font-size:14.5px'>"
                        f"{enc['complaint'] or 'no complaint given'}</div>",
                        unsafe_allow_html=True)

            tone = "var(--accent-2-700)" if enc["tier"] == "high" else "var(--accent-700)"
            st.markdown(f"<div class='kicker' style='margin-top:22px'>Triage</div>"
                        f"<span class='display' style='color:{tone}'>"
                        f"{enc['tier'].upper()}</span>"
                        f"<span class='muted'>&nbsp;&nbsp;{enc['tconf']:.2f}</span>"
                        f"<div style='margin-top:7px;height:5px;background:var(--n-300);"
                        f"border-radius:1px;overflow:hidden'>"
                        f"<div style='width:{enc['tconf']*100:.0f}%;height:100%;"
                        f"background:{tone}'></div></div>", unsafe_allow_html=True)

            st.markdown("<div class='kicker' style='margin-top:22px'>Workflow</div>",
                        unsafe_allow_html=True)
            for label, ok in [
                    ("Triage", True),
                    (f"{enc['organ']} POCUS", enc["organ"] != "Not performed"),
                    ("Image analysed by the model", enc.get("report") is not None),
                    ("Clinical state", True), ("Escalation", True),
                    ("Retrieval", bool(hits)),
                    ("Reasoning", origin in ("recorded", "failed")),
                    ("Safety validation", origin in ("recorded", "failed")),
                    ("Decision support", True), ("Report", True)]:
                st.markdown(f"<div class='wf{'' if ok else ' wf-todo'}'>"
                            f"<span class='mk'>{'✓' if ok else '○'}</span>{label}</div>",
                            unsafe_allow_html=True)
            st.caption("A stage marked ○ did not run, and the assessment says where that "
                       "leaves the conclusion.")

            st.markdown("<div class='kicker' style='margin-top:22px'>Vitals</div>",
                        unsafe_allow_html=True)
            rows = ""
            for k in VITAL_REFERENCE:
                e = (state.get("vitals") or {}).get(k)
                if e:
                    col = ("var(--accent-2-700);font-weight:600" if e["flag"] != "normal"
                           else "inherit")
                    rows += (f"<tr><td style='padding:3px 0'>{k}</td><td style='padding:3px 0;"
                             f"text-align:right;color:{col}'>{e['value']:g}</td></tr>")
                else:
                    rows += (f"<tr><td style='padding:3px 0;color:var(--n-600)'>{k}</td>"
                             f"<td style='padding:3px 0;text-align:right;color:var(--n-600)'>"
                             f"not measured</td></tr>")
            st.markdown(f"<table style='width:100%;font-size:13.5px'>{rows}</table>",
                        unsafe_allow_html=True)

        with main:
            ncrit = sum(1 for a in support["alerts"] if a["severity"] == "CRITICAL")
            if support["alerts"]:
                band("crit", f"⚠ {ncrit} critical · {len(support['alerts']) - ncrit} warning",
                     [f"<b>{a['message']}</b>" for a in support["alerts"]])
            else:
                band("ok", "No alert",
                     ["No measured value crossed a configured bound and the record shows no "
                      "structural gap."])

            a, b = st.columns([1.05, 1], gap="large")
            with a:
                st.markdown("<div class='kicker'>Severity &amp; escalation</div>",
                            unsafe_allow_html=True)
                col = ("var(--accent-2-700)" if sev["severity"] == "HIGH" else
                       "var(--n-800)" if sev["severity"] == "MODERATE" else
                       "var(--accent-700)")
                st.markdown(
                    f"<span class='display-lg' style='color:{col}'>{sev['severity']}</span>"
                    f"<span class='muted'>&nbsp;&nbsp;"
                    f"{'escalation required' if esc['escalate'] else 'no escalation'} · "
                    f"deterministic, not model-generated</span>", unsafe_allow_html=True)
                st.markdown(f"<div class='muted' style='margin-top:6px'>routed as "
                            f"<b>{support['scenario']['label']}</b> · thresholds "
                            f"v{support['thresholds_version']}</div>", unsafe_allow_html=True)
                trig = "<br>".join(f"• {t}" for t in esc["triggers"]) or \
                       "no trigger fired — the case may be answered directly"
                st.markdown(f"<div class='rule' style='margin-top:12px'>"
                            f"<div class='kicker' style='margin-bottom:7px'>Triggers "
                            f"evaluated before the model runs</div>{trig}</div>",
                            unsafe_allow_html=True)

            with b:
                st.markdown(f"<div class='kicker'>POCUS findings — "
                            f"{enc['organ'].lower()}</div>", unsafe_allow_html=True)
                any_row = False
                for f in state["imaging"]["findings"]:
                    any_row = True
                    if f["detected"]:
                        st.markdown(f"<div class='row'>● <b>{f['label']}</b> "
                                    f"<span class='muted'>· {f['organ']} · "
                                    f"{f['confidence']:.2f} · evidence {f['evidence']}</span>"
                                    f"</div>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"<div class='row row-gone'>○ {f['label']} — screened for "
                                    f"and not seen ({f['confidence']:.2f})</div>",
                                    unsafe_allow_html=True)
                for org in state["imaging"]["organs_not_assessed"] or []:
                    any_row = True
                    st.markdown(f"<div class='row' style='color:var(--accent-2-700)'>"
                                f"⚠ <b>{org} — not assessed.</b> A gap in the record, not a "
                                f"negative result.</div>", unsafe_allow_html=True)
                if not any_row:
                    st.markdown("<div class='row row-gone'>no ultrasound performed</div>",
                                unsafe_allow_html=True)
                for lim in state["imaging"]["out_of_scope"]:
                    st.markdown(f"<div class='muted' style='margin-top:8px'>Model limitation — "
                                f"{lim}</div>", unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("<div class='kicker'>Assistant summary</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='paper' style='font-size:16px;line-height:1.6'>"
                        f"{report['conclusion']}</div>", unsafe_allow_html=True)
            st.caption("Assembled from the record by fixed rules rather than written by the "
                       "language model, so it cannot say anything the rest of the assessment "
                       "does not.")

            st.markdown("<hr>", unsafe_allow_html=True)
            c, d = st.columns([1.05, 1], gap="large")

            with c:
                st.markdown("<div class='kicker'>Differential — decision support</div>",
                            unsafe_allow_html=True)
                if origin == "failed":
                    band("crit", "Differential withheld",
                         result["validation_errors"] or ["the model failed"])
                    st.markdown(f"<div class='band band-ok'>The safety decision survived: "
                                f"<b>{sev['severity']} severity, "
                                f"{'escalation required' if esc['escalate'] else 'no escalation'}"
                                f"</b>. It is computed from the record before the model is "
                                f"consulted, so it does not depend on the model running at "
                                f"all.</div>", unsafe_allow_html=True)
                elif origin == "not_generated":
                    st.info("**No differential generated for this encounter.** Producing one "
                            "requires a 4.9 GB language model on a GPU, which is not loaded "
                            "here. Everything else on this page was computed on this machine. "
                            "Load an example encounter unedited to see a differential recorded "
                            "from the GPU run.")
                else:
                    secs = result.get("_seconds")
                    st.markdown(f"<span class='tag t-neutral'>RECORDED</span>"
                                f"<span class='muted'>HuatuoGPT-o1-8B, Colab T4"
                                f"{f', {secs:.0f}s' if secs else ''}, temperature 0</span>",
                                unsafe_allow_html=True)
                    diff = result.get("differential") or {}
                    if result.get("differential_withheld"):
                        band("crit", "Differential withheld — did not pass validation",
                             result.get("validation_errors") or [])
                    for i, dx in enumerate(diff.get("differential") or [], 1):
                        lk = str(dx.get("likelihood", "")).lower()
                        cls = ("t-accent-2" if lk == "high" else
                               "t-accent" if lk == "moderate" else "t-outline")
                        st.markdown(f"<div style='margin-top:14px'><span class='display' "
                                    f"style='font-size:20px'>{i}. {dx.get('diagnosis')}</span>"
                                    f"&nbsp;<span class='tag {cls}'>{lk}</span></div>",
                                    unsafe_allow_html=True)
                        for eid, t in zip(dx.get("supporting_ids") or [],
                                          dx.get("supporting") or []):
                            st.markdown(f"<div class='row'>supports "
                                        f"<span class='eid'>{eid}</span> {t}</div>",
                                        unsafe_allow_html=True)
                        for t in (dx.get("contradicting") or []):
                            st.markdown(f"<div class='row row-gone'>against {t}</div>",
                                        unsafe_allow_html=True)
                        for t in (dx.get("limitations") or []):
                            st.caption(f"limitation — {t}")
                    if diff.get("uncertainty"):
                        st.caption(f"uncertainty — {diff['uncertainty']}")
                    if result.get("warnings"):
                        st.markdown("<div class='kicker' style='margin-top:16px'>Warnings "
                                    "raised, answer delivered anyway</div>",
                                    unsafe_allow_html=True)
                        for w in result["warnings"]:
                            st.markdown(f"<div class='row'>{w}</div>", unsafe_allow_html=True)

            with d:
                st.markdown("<div class='kicker'>Missing information</div>",
                            unsafe_allow_html=True)
                gone = state["missing"]["labs"] + state["missing"]["vitals"]
                if not gone:
                    st.markdown("<div class='band band-ok'>Every value in the reference set "
                                "was measured.</div>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='tag t-accent-2'>NOT MEASURED ≠ NORMAL</span>"
                                "<span class='muted'>These have no evidence identifier, so "
                                "the reasoning layer cannot cite them for or against any "
                                "diagnosis.</span>", unsafe_allow_html=True)
                    st.markdown(tags([("neutral", g) for g in gone]), unsafe_allow_html=True)

                st.markdown("<div class='kicker' style='margin-top:20px'>Suggested next "
                            "steps</div>", unsafe_allow_html=True)
                recs = support["additional_examinations"]
                if not recs:
                    st.markdown("<div class='row'>The record is complete for this "
                                "presentation.</div>", unsafe_allow_html=True)
                for i, r in enumerate(recs[:8], 1):
                    st.markdown(f"<div class='row'><b>{i}. {r['exam']}</b> "
                                f"<span class='tag t-neutral'>{r['priority']}</span><br>"
                                f"<span class='muted'>{r['reason']}</span></div>",
                                unsafe_allow_html=True)
                st.caption("The system recommends examinations; it does not order them, and "
                           "they do not replace clinician judgement.")

                st.markdown("<div class='kicker' style='margin-top:20px'>Therapeutic "
                            "considerations</div>", unsafe_allow_html=True)
                th = support["therapeutic"]
                if not th["considerations"]:
                    st.markdown(f"<div class='paper'><b>No protocol-backed therapeutic "
                                f"recommendation for this case.</b><div class='muted' "
                                f"style='margin-top:6px'>{th['status']}"
                                f"{'. ' + th['note'] if th.get('note') else ''}</div></div>",
                                unsafe_allow_html=True)
                for c_ in th["considerations"]:
                    st.markdown(f"<div class='paper'><b>CONSIDER</b> — "
                                f"{c_['consideration']}<div class='muted' "
                                f"style='margin-top:6px'>Basis: {c_['basis']} "
                                f"[{c_['passage']}]<br>{c_['disclaimer']}</div></div>",
                                unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════ 3. Reasoning chain
with T[2]:
    if not enc:
        st.info("No encounter yet.")
    else:
        st.markdown("<div class='kicker kicker-2'>Traceability</div>"
                    "<div class='display' style='font-size:34px'>Reasoning chain</div>",
                    unsafe_allow_html=True)
        st.markdown("<p class='lead'>Every step is recorded with its inputs and outputs. The "
                    "language model contributes one of them and is bounded by the validator "
                    "that follows it.</p>", unsafe_allow_html=True)

        det = [f for f in state["imaging"]["findings"] if f["detected"]]
        ev = build_evidence(state)
        img = enc.get("report")
        st.markdown("".join(
            f"<div class='step'><div class='stepno'>{no}</div><div>"
            f"<div class='stepttl'>{ttl}</div>"
            f"<div class='muted' style='margin-top:5px'>{body}</div></div></div>"
            for no, ttl, body in [
                ("01 · TRIAGE", f"Triage {enc['tconf']:.2f} → {enc['tier'].upper()}",
                 f"Inputs: age {enc['age']}, complaint “{enc['complaint'] or '—'}”, " +
                 (", ".join(f"{k} {v:g}" for k, v in enc["vitals"].items()) or
                  "no vital recorded") + ". Structured data only — the triage agent never "
                 "sees an image."),
                ("02 · POCUS",
                 (f"{det[0]['label']} detected, {det[0]['confidence']:.2f}" if det else
                  "no finding above threshold"),
                 (f"{img['model']} read the uploaded image on this machine. " if img else
                  "Findings as reported by the module. ") +
                 f"{len(state['imaging']['findings'])} finding(s) screened. " +
                 ("; ".join(state["imaging"]["out_of_scope"]) or "no scope limit declared")),
                ("03 · STATE", "Clinical state assembled",
                 f"{len(ev)} citable fact(s), "
                 f"{len(state['missing']['labs']) + len(state['missing']['vitals'])} absent, "
                 f"{len(state.get('conflicts') or [])} conflict(s). Absences are recorded "
                 f"explicitly and never imputed."),
                ("04 · ESCALATION",
                 f"{len(esc['triggers'])} trigger(s) → "
                 f"{'ESCALATE' if esc['escalate'] else 'ANSWER DIRECTLY'}",
                 "Deterministic, and evaluated before any model output is read. Nothing the "
                 "model says can lower it."),
                ("05 · RETRIEVAL", f"{len(hits)} passage(s) retrieved",
                 "Query built from the clinical state rather than from free text. Relevance "
                 "floor 0.10 — below it no hit is reported, rather than the best of a bad "
                 "set."),
                ("06 · REASONING",
                 {"recorded": "Differential drafted", "failed": "Model backend failed",
                  "not_generated": "Not run — no GPU in this deployment"}[origin],
                 "HuatuoGPT-o1-8B, temperature 0, fixed seed. Evidence may be cited only by "
                 "identifier from the enumerated list, so a fact the record does not hold has "
                 "no way to be named."),
                ("07 · VALIDATION",
                 ("Answer withheld" if (origin == "failed" or
                                        result.get("differential_withheld"))
                  else "Answer delivered" if origin == "recorded" else "Not reached"),
                 "The validator delivers with warnings or withholds. It does not rewrite the "
                 "model's answer: a likelihood edited in post would reach a clinician as the "
                 "model's judgement when it is not."),
                ("08 · DECISION SUPPORT",
                 f"Severity {sev['severity']} · {len(support['alerts'])} alert(s)",
                 "Computed from the state rather than from the answer, and attached even when "
                 "the model never ran."),
                ("09 · REPORT", "Conclusion assembled and archived",
                 "Templated from the record, not generated, and archived under a content hash "
                 "that excludes the timestamp so an identical assessment hashes identically."),
            ]), unsafe_allow_html=True)

        if state.get("conflicts"):
            st.markdown("<div class='kicker kicker-2' style='margin-top:22px'>Agent "
                        "disagreement</div>", unsafe_allow_html=True)
            for c_ in state["conflicts"]:
                st.markdown(f"<div class='row'>{c_}</div>", unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════ 4. Evidence & sources
with T[3]:
    if not enc:
        st.info("No encounter yet.")
    else:
        st.markdown("<div class='kicker kicker-2'>Provenance</div>"
                    "<div class='display' style='font-size:34px'>Evidence &amp; sources</div>",
                    unsafe_allow_html=True)
        st.markdown("<p class='lead'>Structured evidence comes from the patient record and the "
                    "perception models. Retrieved passages come from the sourced corpus and "
                    "are shown with the score that admitted them.</p>", unsafe_allow_html=True)

        e1, e2 = st.columns([1, 1.1], gap="large")
        with e1:
            st.markdown("<div class='kicker'>Structured evidence — what may be cited</div>",
                        unsafe_allow_html=True)
            cited = {i for dx in ((result.get("differential") or {}).get("differential") or [])
                     for i in (dx.get("supporting_ids") or [])}
            for e in build_evidence(state):
                mark = "●" if e["id"] in cited else "○"
                st.markdown(f"<div class='row'><span class='eid'>{mark} {e['id']}</span> "
                            f"{e['text']}</div>", unsafe_allow_html=True)
            for g in state["missing"]["labs"] + state["missing"]["vitals"]:
                st.markdown(f"<div class='row row-gone'><span class='eid' "
                            f"style='color:var(--n-600)'>—&nbsp;&nbsp;</span>{g} — never "
                            f"measured, so it has no identifier and cannot be cited</div>",
                            unsafe_allow_html=True)
            st.caption("A filled marker means the model cited that identifier. The list itself "
                       "is built from the record, so every abnormal value appears whether the "
                       "model reasoned from it or not.")

        with e2:
            st.markdown(f"<div class='kicker'>Retrieved passages — {len(hits)}</div>",
                        unsafe_allow_html=True)
            if not hits:
                st.markdown("<div class='paper'>No passage cleared the relevance floor of "
                            "0.10. Reasoning was performed from the structured state alone, "
                            "and the answer is not marked guideline-grounded.</div>",
                            unsafe_allow_html=True)
            for h in hits:
                st.markdown(
                    f"<div class='paper'><div style='display:flex;justify-content:space-between;"
                    f"font-size:11.5px;letter-spacing:.11em;text-transform:uppercase;"
                    f"color:var(--n-700)'><span>[{h['n']}] {h['id']} · {h['topic']}</span>"
                    f"<span>{h['score']:.2f}</span></div>"
                    f"<p style='margin:9px 0 0;font-size:15px'>{h['text']}</p>"
                    f"<div class='muted' style='margin-top:9px'>{h['source']}</div></div>",
                    unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════ 5. Clinical report
with T[4]:
    if not enc:
        st.info("No encounter yet.")
    else:
        h1, h2 = st.columns([3, 1])
        h1.markdown("<div class='kicker kicker-2'>Output</div>"
                    "<div class='display' style='font-size:34px'>Clinical report</div>",
                    unsafe_allow_html=True)
        with h2:
            st.download_button("Export report", report_text, use_container_width=True,
                               file_name=f"{report['encounter_id']}_report.txt")
        st.markdown(
            f"<div class='paper' style='padding:40px 46px;box-shadow:var(--shadow-md)'>"
            f"<div style='display:flex;justify-content:space-between;align-items:flex-end;"
            f"padding-bottom:14px;border-bottom:3px solid var(--text)'>"
            f"<div><div class='display' style='font-size:21px'>Decision support summary</div>"
            f"<div class='muted'>Encounter {report['encounter_id']} · "
            f"{report['generated_at'][:16].replace('T', ' ')}</div></div>"
            f"<div class='muted' style='text-align:right'>Emergency Department<br>"
            f"severity {sev['severity']}</div></div>"
            f"<p style='margin-top:20px;font-size:16px;line-height:1.65'>"
            f"{report['conclusion']}</p></div>", unsafe_allow_html=True)
        st.caption("Not a diagnosis and not a discharge document. The conclusion is templated "
                   "from the record so it cannot assert anything the assessment does not.")
        with st.expander("Full report"):
            st.code(report_text, language=None)

# ═════════════════════════════════════════════════════════════ 6. Architecture
with T[5]:
    st.markdown("<div class='kicker kicker-2'>How it is put together</div>"
                "<div class='display' style='font-size:34px'>Architecture</div>",
                unsafe_allow_html=True)
    st.markdown("<p class='lead'>The language model sits inside the pipeline, not at its "
                "centre. Escalation is decided before it runs and cannot be overridden by "
                "it.</p>", unsafe_allow_html=True)

    def block(title, sub, tone="var(--surface)", accent=None):
        border = f"border-left:4px solid {accent};" if accent else ""
        return (f"<div style='background:{tone};border-radius:2px;padding:14px 22px;"
                f"text-align:center;{border}box-shadow:var(--shadow-sm)'>"
                f"<div class='display' style='font-size:17px'>{title}</div>"
                f"<div class='muted'>{sub}</div></div>")

    arrow = "<div style='height:22px;width:1px;background:var(--n-400);margin:0 auto'></div>"
    top = st.columns(3, gap="medium")
    top[0].markdown(block("Triage Agent", "urgency from tabular data, never an image"),
                    unsafe_allow_html=True)
    top[1].markdown(block("POCUS Agents", "lung · cardiac · gallbladder"),
                    unsafe_allow_html=True)
    top[2].markdown(block("Labs &amp; vitals", "values present and values absent"),
                    unsafe_allow_html=True)
    for t, s_, tone, acc in [
            ("Clinical state", "one structured object · findings, values, conflicts, and what "
             "is MISSING", "var(--accent-100)", None),
            ("Escalation engine", "seven rules, computed BEFORE the model runs",
             "var(--accent-2-100)", "var(--accent-2-600)"),
            ("Retrieval", "TF-IDF over 32 sourced units, with a relevance floor",
             "var(--surface)", None),
            ("HuatuoGPT-o1-8B", "cites enumerated evidence identifiers only",
             "var(--surface)", None),
            ("Validator", "deliver with warnings · or withhold · never rewrite",
             "var(--accent-100)", None),
            ("Decision support", "severity · alerts · examinations · therapeutics",
             "var(--surface)", None)]:
        st.markdown(arrow + block(t, s_, tone, acc), unsafe_allow_html=True)
    st.markdown(arrow, unsafe_allow_html=True)
    out = st.columns(3, gap="medium")
    for c_, t in zip(out, ["Differential", "Critical alerts", "Automated report"]):
        c_.markdown(f"<div style='border-top:3px solid var(--text);padding:12px;"
                    f"text-align:center' class='display'>{t}</div>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    a, b = st.columns(2, gap="large")
    a.markdown("<div class='kicker'>Perception is separate from reasoning</div>"
               "<p style='font-size:15px'>The ultrasound agent reports what it sees and never "
               "assigns urgency — it cannot see the vitals, laboratory results or history "
               "that the decision requires.</p>", unsafe_allow_html=True)
    b.markdown("<div class='kicker'>Rules are separate from the model</div>"
               "<p style='font-size:15px'>Escalation, severity, alerts and what is missing are "
               "computed deterministically before the language model is consulted, and its "
               "output is checked against the same record afterwards.</p>",
               unsafe_allow_html=True)

    st.markdown("<div class='kicker' style='margin-top:14px'>What the model can and cannot "
                "say</div>", unsafe_allow_html=True)
    st.markdown(
        "- Evidence is cited by **identifier**, drawn from an enumerated list of facts the "
        "record holds. There is no identifier for an invented observation, none for a test "
        "never performed, and none for a sentence out of the reference corpus.\n"
        "- A value's qualifier travels with its identifier, so a normal troponin cannot be "
        "relabelled as elevated on the way through.\n"
        "- Therapeutic suggestions are gated behind a citable protocol. With none retrieved "
        "the assistant produces no recommendation, rather than drawing on the model's own "
        "training knowledge.")

    st.markdown("<hr>", unsafe_allow_html=True)
    v = st.columns(4)
    v[0].metric("Safety tests", "224", "25 properties")
    v[1].metric("End-to-end checks", "150", "10 cases")
    v[2].metric("Corpus units", "32", "all sourced")
    v[3].metric("Perception under test", "lung", "real inference")
    st.caption("These are software tests on synthetic cases. They establish that the safety "
               "mechanisms behave as specified and say nothing about diagnostic accuracy. "
               "Clinical validation against expert ground truth is outside the scope of this "
               "work.")
