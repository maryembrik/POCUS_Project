"""POCUS-Emergency -- the safety layer, live.

    streamlit run app.py

Most decision-support demos replay a fixed case. This one lets the examiner build the patient
and watch the safety layer respond, because the safety layer is the part of this system that
can afford to be interactive: escalation, conflict detection and the record of what is missing
are deterministic Python computed from the structured state, with no model and no GPU, in
milliseconds.

What is live and what is recorded is stated on screen throughout:

    LIVE      the clinical state, conflicts, what is missing, the evidence identifiers, and
              the escalation decision. Recomputed on every interaction.

    LIVE      the failure demonstration. "Break the model" genuinely calls reason() with a
              backend that raises; the escalation surviving is executed, not mocked.

    RECORDED  the differential, for the five benchmark cases only. Producing one needs a
              4.9 GB model on a GPU, so those are the answers from the frozen run in
              models/clinical_reasoning_v4_final/, shown with their measured latency.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import streamlit as st

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.agents import schema as S  # noqa: E402
from src.agents.clinical.clinical_state import (  # noqa: E402
    LAB_REFERENCE, VITAL_REFERENCE, build_clinical_state, build_evidence)
from src.agents.clinical.llm import FailingBackend  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision, reason  # noqa: E402
from src.agents.clinical.retrieval import Retriever  # noqa: E402

FROZEN = ROOT / "models" / "clinical_reasoning_v4_final" / "results.json"
LUNG_FINDINGS = ["b_lines", "consolidation", "pleural_effusion", "pleural_thickening"]

# Physiological slider bounds, written out rather than derived from the reference range.
# Scaling the reference bounds by a constant gave pH from 0 to 44.7 and SpO2 up to 160% --
# arithmetically consistent and clinically absurd.  (min, max, step)
VITAL_RANGE = {
    "hr":   (30.0, 200.0, 1.0),
    "sbp":  (50.0, 220.0, 1.0),
    "rr":   (6.0, 45.0, 1.0),
    "spo2": (70.0, 100.0, 1.0),
    "temp": (34.0, 42.0, 0.1),
}
LAB_RANGE = {
    "troponin":   (0.0, 500.0, 1.0),
    "bnp":        (0.0, 2000.0, 10.0),
    "d_dimer":    (0.0, 5000.0, 50.0),
    "lactate":    (0.0, 12.0, 0.1),
    "crp":        (0.0, 300.0, 1.0),
    "wbc":        (0.0, 30.0, 0.1),
    "creatinine": (0.0, 600.0, 5.0),
    "ph":         (6.8, 7.8, 0.01),
}

st.set_page_config(page_title="POCUS-Emergency", page_icon="🫁", layout="wide")

st.markdown("""
<style>
  .block-container { padding-top: 1.6rem; max-width: 1400px; }
  .banner { padding: .95rem 1.2rem; border-radius: 8px; margin: .2rem 0 .8rem 0;
            border-left: 7px solid; }
  .esc  { background: rgba(217,83,79,.13);  border-color:#d9534f; }
  .dir  { background: rgba(76,154,99,.13);  border-color:#4c9a63; }
  .hold { background: rgba(214,168,60,.13); border-color:#d6a83c; }
  .chip { display:inline-block; padding:.12rem .55rem; border-radius:5px;
          font-size:.72rem; letter-spacing:.04em; margin-right:.4rem; }
  .live { background: rgba(76,154,99,.22); color:#5fbf85; }
  .rec  { background: rgba(140,150,170,.18); color:#98a3b5; }
  .row  { padding:.42rem .7rem; border-radius:6px; margin-bottom:.3rem;
          border:1px solid rgba(140,150,170,.22); font-size:.9rem; }
  .gone { opacity:.45; text-decoration: line-through; }
  .muted{ color:#8b95a5; font-size:.85rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}


PRESETS = {
    "Custom": None,
    "Concordant — evidence agrees": dict(
        tier="high", tconf=0.88, organs=["lung"], complaint="orthopnoea",
        findings={"b_lines": 0.92}, vitals={"hr": 122.0, "rr": 28.0, "spo2": 88.0},
        labs={"troponin": 62.0, "bnp": 890.0, "lactate": 2.6}),
    "Conflict — agents disagree": dict(
        tier="low", tconf=0.88, organs=["heart"], complaint="mild chest discomfort",
        findings={"severe dysfunction": 0.74},
        vitals={"hr": 82.0, "sbp": 128.0, "spo2": 96.0},
        labs={"troponin": 5.0, "lactate": 1.1}),
    "Missing data — key labs absent": dict(
        tier="high", tconf=0.79, organs=["lung"], complaint="acute breathlessness",
        findings={"b_lines": 0.86}, vitals={"hr": 118.0, "rr": 24.0, "spo2": 90.0},
        labs={}),
    "Reassuring — everything negative": dict(
        tier="low", tconf=0.80, organs=["lung"], complaint="mild breathlessness",
        findings={}, vitals={"hr": 78.0, "rr": 16.0, "spo2": 97.0},
        labs={"troponin": 4.0, "lactate": 1.0}),
    "Not assessed — organ never scanned": dict(
        tier="high", tconf=0.82, organs=["lung", "heart"],
        complaint="chest pain and breathlessness",
        findings={"b_lines": 0.78}, vitals={"hr": 104.0, "rr": 22.0, "spo2": 93.0},
        labs={"troponin": 9.0, "lactate": 1.4}, unscanned=["heart"]),
}
SCENARIO_KEY = {
    "Concordant — evidence agrees": "concordant",
    "Conflict — agents disagree": "conflict",
    "Missing data — key labs absent": "missing",
    "Reassuring — everything negative": "reassuring",
    "Not assessed — organ never scanned": "not_assessed",
}

# =======================================================================  sidebar
st.sidebar.markdown("## POCUS-Emergency")
st.sidebar.caption("Build a patient. Watch the safety layer respond.")

preset_name = st.sidebar.selectbox("Start from", list(PRESETS))

# Widgets below are keyed, and a keyed widget's stored value wins over the `value=` default.
# Without this, choosing a preset changed nothing after the first render: the sliders and
# checkboxes kept whatever the previous preset had left in session state, and the escalation
# shown no longer matched the case named in the selector.
if st.session_state.get("_preset") != preset_name:
    for k in [k for k in st.session_state
              if k.split("_")[0] in {"f", "c", "v", "vv", "l", "ll"}]:
        del st.session_state[k]
    st.session_state["_preset"] = preset_name

P = PRESETS[preset_name] or PRESETS["Missing data — key labs absent"]

st.sidebar.divider()
st.sidebar.markdown("**Triage Agent**")
tier = st.sidebar.select_slider("Urgency tier", ["low", "medium", "high"],
                                value=P["tier"])
tconf = st.sidebar.slider("Triage confidence", 0.5, 1.0, float(P["tconf"]), 0.01)

st.sidebar.divider()
st.sidebar.markdown("**Ultrasound Agent**")
organs = st.sidebar.multiselect("Organs requested", ["lung", "heart", "gallbladder"],
                                default=P["organs"])
unscanned = st.sidebar.multiselect(
    "…of which never actually scanned", organs, default=P.get("unscanned", []),
    help="An organ requested but not assessed is a gap in the record, not a negative result.")

st.sidebar.divider()
st.sidebar.markdown("**Break it**")
break_model = st.sidebar.toggle("Break the model", value=False,
                                help="Runs the pipeline for real with a backend that raises.")

st.sidebar.divider()
st.sidebar.caption("**Not a diagnostic device.** Synthetic cases, no clinical ground truth, "
                   "never validated against outcomes.")

# =======================================================================  patient editor
st.markdown("## The safety layer, live")
st.markdown("<span class='chip live'>LIVE</span> everything below recomputes on every change, "
            "with no model and no GPU", unsafe_allow_html=True)

edit, derived = st.columns([1, 1.15], gap="large")

with edit:
    st.markdown("#### Patient")
    complaint = st.text_input("Presenting complaint", P["complaint"])

    st.markdown("**Imaging findings**")
    findings: dict[str, float] = {}
    if "lung" in organs and "lung" not in unscanned:
        for f in LUNG_FINDINGS:
            c1, c2 = st.columns([1, 2])
            on = c1.checkbox(f.replace("_", " "), value=f in P["findings"], key=f"f_{f}")
            conf = c2.slider(" ", 0.0, 1.0, float(P["findings"].get(f, 0.05)), 0.01,
                             key=f"c_{f}", label_visibility="collapsed")
            findings[f] = conf if on else -conf          # negative = screened, not detected
    if "heart" in organs and "heart" not in unscanned:
        c1, c2 = st.columns([1, 2])
        on = c1.checkbox("severe dysfunction",
                         value="severe dysfunction" in P["findings"], key="f_sd")
        conf = c2.slider(" ", 0.0, 1.0,
                         float(P["findings"].get("severe dysfunction", 0.20)), 0.01,
                         key="c_sd", label_visibility="collapsed")
        findings["severe dysfunction"] = conf if on else -conf

    st.markdown("**Vitals** — untick to make it *never measured*")
    vitals: dict[str, float] = {}
    for k, ref in VITAL_REFERENCE.items():
        c1, c2 = st.columns([1, 2])
        on = c1.checkbox(f"{k}  ({ref['unit']})", value=k in P["vitals"], key=f"v_{k}")
        lo, hi, step = VITAL_RANGE[k]
        default = float(P["vitals"].get(k, (ref["normal_min"] + ref["normal_max"]) / 2))
        val = c2.slider(" ", lo, hi, min(max(default, lo), hi), step,
                        key=f"vv_{k}", label_visibility="collapsed")
        if on:
            vitals[k] = val

    st.markdown("**Labs** — untick to make it *never measured*")
    labs: dict[str, float] = {}
    for k, ref in LAB_REFERENCE.items():
        c1, c2 = st.columns([1, 2])
        unit = f"  ({ref['unit']})" if ref.get("unit") else ""
        on = c1.checkbox(f"{k}{unit}", value=k in P["labs"], key=f"l_{k}")
        lo, hi, step = LAB_RANGE[k]
        default = float(P["labs"].get(k, ref.get("normal_max", 1) * 0.5))
        val = c2.slider(" ", lo, hi, min(max(default, lo), hi), step,
                        key=f"ll_{k}", label_visibility="collapsed")
        if on:
            labs[k] = val

# =======================================================================  build state, live
reports: dict[str, dict] = {}
if "lung" in organs:
    if "lung" in unscanned:
        reports["lung"] = S.make_report("lung", [], status="not_supported",
                                        reliability={"scope": "requested, never assessed"})
    else:
        det = [S.make_finding(k.replace("_", " "), v) for k, v in findings.items()
               if k in LUNG_FINDINGS and v > 0]
        neg = [S.make_finding(k.replace("_", " "), -v) for k, v in findings.items()
               if k in LUNG_FINDINGS and v <= 0]
        reports["lung"] = S.make_report(
            "lung", det, not_detected=neg, status="ok" if (det or neg) else "failed",
            reliability={"confidence_calibrated": True, "has_normal_class": False,
                         "modelled_findings": LUNG_FINDINGS,
                         "scope": "pneumothorax is NOT modelled and cannot be excluded"})
if "heart" in organs:
    if "heart" in unscanned:
        reports["heart"] = S.make_report("heart", [], status="not_supported",
                                         reliability={"scope": "requested, never assessed"})
    else:
        sd = findings.get("severe dysfunction", -0.2)
        reports["heart"] = S.make_report(
            "heart", [S.make_finding("severe dysfunction", sd)] if sd > 0 else [],
            not_detected=[] if sd > 0 else [S.make_finding("severe dysfunction", -sd)],
            reliability={"confidence_calibrated": True, "has_normal_class": True,
                         "scope": "CAMUS-like 4CH stills; EF is an area proxy"})
if "gallbladder" in organs:
    reports["gallbladder"] = S.make_report(
        "gallbladder", [S.make_finding("cholelithiasis", 0.55, group="Cholelithiasis")],
        reliability={"confidence_calibrated": True, "has_normal_class": False,
                     "scope": "no healthy class exists in the training data"})

bundle = {"encounter_id": "LIVE-001",
          "triage": S.make_triage(tier, tconf, features=vitals),
          "ultrasound": reports,
          "clinical": {"chief_complaint": complaint}}
state = build_clinical_state(bundle, labs=labs)
# vitals reach the state through triage features; the builder flags them against the reference
esc = escalation_decision(state)
evidence = build_evidence(state)

# =======================================================================  derived, live
with derived:
    st.markdown("#### What the system derives")

    if esc["escalate"]:
        trig = "".join(f"<div style='margin-top:.35rem'>• {t}</div>" for t in esc["triggers"])
        st.markdown(f"<div class='banner esc'><b>ESCALATE — route: {esc['route']}</b>{trig}"
                    "</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='banner dir'><b>DIRECT — no escalation trigger</b>"
                    "<div style='margin-top:.35rem'>• evidence agrees, record complete, "
                    "no organ left unassessed</div></div>", unsafe_allow_html=True)

    q = (state.get("case_quality") or {}).get("grade", "?")
    m = st.columns(4)
    m[0].metric("Case quality", q)
    m[1].metric("Citable facts", len(evidence))
    m[2].metric("Never measured",
                len(state["missing"]["labs"]) + len(state["missing"]["vitals"]))
    m[3].metric("Conflicts", len(state.get("conflicts") or []))

    if state.get("conflicts"):
        for c in state["conflicts"]:
            st.warning(c)

    st.markdown("**Evidence the model may cite**")
    st.caption("Each fact gets an identifier. The model cites identifiers and nothing else, "
               "so it cannot invent an observation the state does not hold.")
    for e in evidence:
        st.markdown(f"<div class='row'><code>{e['id']}</code> &nbsp; {e['text']}</div>",
                    unsafe_allow_html=True)

    gone = state["missing"]["labs"] + state["missing"]["vitals"]
    if gone:
        st.markdown("**Never measured — no identifier exists**")
        st.caption("Absent is not normal. With no identifier, an absent test cannot be cited "
                   "for or against any diagnosis. The constraint is structural, not a rule the "
                   "model is asked to follow.")
        for g in gone:
            st.markdown(f"<div class='row gone'>{g}</div>", unsafe_allow_html=True)

    if state["imaging"]["organs_not_assessed"]:
        st.error("Never assessed: " + ", ".join(state["imaging"]["organs_not_assessed"])
                 + " — a gap in the record, not a negative result.")

st.divider()

# =======================================================================  try to break it
st.markdown("#### Try to break it")
st.caption("Each button changes the patient. Watch the escalation banner above.")
b = st.columns(4)
if b[0].button("Hide the troponin", use_container_width=True):
    st.session_state["l_troponin"] = False
    st.rerun()
if b[1].button("Make the agents disagree", use_container_width=True):
    st.session_state["f_sd"] = True
    st.session_state["c_sd"] = 0.74
    st.rerun()
if b[2].button("Turn every finding negative", use_container_width=True):
    for f in LUNG_FINDINGS:
        st.session_state[f"f_{f}"] = False
    st.session_state["f_sd"] = False
    st.rerun()
if b[3].button("Reset", use_container_width=True):
    for k in list(st.session_state):
        del st.session_state[k]
    st.rerun()

st.divider()

# =======================================================================  reasoning
st.markdown("#### Clinical reasoning")

if break_model:
    out = reason(state, llm_fn=FailingBackend("out of memory"), max_revisions=1)
    st.markdown("<span class='chip live'>LIVE</span> executed now, backend raising on "
                "every call", unsafe_allow_html=True)
    errs = "".join(f"<div style='margin-top:.35rem'>• {e}</div>"
                   for e in (out["validation_errors"] or []))
    st.markdown(f"<div class='banner hold'><b>DIFFERENTIAL WITHHELD</b>{errs}</div>",
                unsafe_allow_html=True)
    same = out["escalation"] == esc
    st.success(
        f"Escalation decision **{'unchanged' if same else 'CHANGED — investigate'}**: "
        f"{'ESCALATE' if out['escalation']['escalate'] else 'DIRECT'}, "
        f"{len(out['escalation']['triggers'])} trigger(s).\n\n"
        "It is computed from the structured state before the model is consulted, so it does "
        "not depend on the model running at all. The safety-critical output survives the "
        "failure of the least predictable component.")

else:
    key = SCENARIO_KEY.get(preset_name)
    rec = (load_frozen().get("none") or {}).get(key) if key else None
    if not rec:
        st.info("The differential needs a 4.9 GB model on a GPU, so it is shown only for the "
                "five benchmark presets. Everything above is computed live from the patient "
                "you built — pick a preset, or toggle **Break the model** to see the failure "
                "path executed for real.")
    else:
        secs = rec.get("_seconds")
        st.markdown(f"<span class='chip rec'>RECORDED</span> HuatuoGPT-o1-8B, Colab T4, "
                    f"{secs:.0f}s, temperature 0 &nbsp; "
                    f"<span class='muted'>(the state and escalation above are still live; "
                    f"edits change them but not this recorded answer)</span>",
                    unsafe_allow_html=True)

        diff = rec.get("differential") or {}
        if rec.get("differential_withheld"):
            errs = "".join(f"<div style='margin-top:.35rem'>• {e}</div>"
                           for e in (rec.get("validation_errors") or []))
            st.markdown(f"<div class='banner hold'><b>DIFFERENTIAL WITHHELD</b>{errs}</div>",
                        unsafe_allow_html=True)

        for i, d in enumerate(diff.get("differential") or []):
            st.markdown(f"**{i + 1}. {d.get('diagnosis', '?')}** &nbsp;"
                        f"<span class='muted'>likelihood {d.get('likelihood', '?')}</span>",
                        unsafe_allow_html=True)
            for eid, text in zip(d.get("supporting_ids") or [],
                                 d.get("supporting") or []):
                st.markdown(f"<div class='row'><code>{eid}</code> &nbsp; {text}</div>",
                            unsafe_allow_html=True)
            for text in (d.get("contradicting") or []):
                st.markdown(f"<div class='row'>against: {text}</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**What would most change this assessment**")
            for x in (diff.get("missing_information") or []):
                st.markdown(f"- {x}")
            if rec.get("normalizations"):
                st.caption("Corrected in Python: " + "; ".join(rec["normalizations"]))
        with c2:
            st.markdown("**Recommended next step**")
            st.write(diff.get("recommended_next_step", "—"))
            st.caption("Decision support: the system recommends investigations, it does not "
                       "instruct treatment.")

        considered = rec.get("evidence_considered") or []
        if considered:
            st.markdown("**Clinical evidence considered**")
            st.caption("Built from the state, not from the model's answer — every abnormal "
                       "value appears here whether the model reasoned from it or not.")
            for r in considered:
                mark = "✱" if r["used"] else "&nbsp;&nbsp;"
                cls = "row" if r["used"] else "row gone"
                st.markdown(f"<div class='{cls}'>{mark} <code>{r['id']}</code> {r['text']}"
                            "</div>", unsafe_allow_html=True)
            n = sum(1 for r in considered if r["used"])
            st.caption(f"✱ cited by the model ({n} of {len(considered)}). The rest were "
                       "recorded and shown regardless.")

        if rec.get("warnings"):
            with st.expander(f"{len(rec['warnings'])} warning(s) — delivered anyway"):
                for w in rec["warnings"]:
                    st.markdown(f"- {w}")

st.divider()
st.caption("POCUS-Emergency — internship prototype. The safety layer is verified by 180 tests. "
           "The clinical value of the system is unknown and was never measured. Not for "
           "clinical use.")
