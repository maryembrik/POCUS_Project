"""POCUS Emergency Assistant -- multimodal AI-assisted clinical decision support.

    streamlit run app.py

Three pages: enter an encounter, read the assessment, and see the architecture that produced
it. The intake is clinical rather than technical -- a clinician enters a patient, not a
configuration -- and the assistant selects the pipeline from the presentation.

What runs here and what does not, stated on screen wherever it matters:

    LIVE      the clinical state, conflicts, what is missing, evidence identifiers, the
              escalation decision, severity, alerts, recommended examinations, scenario
              routing, retrieval, and the report. Deterministic Python, no GPU.

    LIVE      the failure demonstration. Simulating a model failure calls reason() with a
              backend that raises; the escalation surviving is executed, not mocked.

    RECORDED  the differential, and only for the example encounters. Generating one needs a
              4.9 GB model on a GPU, so a custom encounter says so rather than inventing one.

    NOT RUN   uploaded images. The perception models run on GPU in the training notebooks and
              are not loaded here. An upload is displayed, never analysed, and the interface
              says so at the point of upload rather than in a footnote.
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
    LAB_REFERENCE, VITAL_REFERENCE, build_clinical_state)
from src.agents.clinical.decision_support import decision_support  # noqa: E402
from src.agents.clinical.llm import FailingBackend  # noqa: E402
from src.agents.clinical.reasoning import escalation_decision, reason  # noqa: E402
from src.agents.clinical.report import build_report, render_report  # noqa: E402
from src.agents.clinical.retrieval import Retriever  # noqa: E402

FROZEN = ROOT / "models" / "clinical_reasoning_v4_final" / "results.json"
LUNG_FINDINGS = ["b_lines", "consolidation", "pleural_effusion", "pleural_thickening"]

VITAL_RANGE = {"hr": (30.0, 200.0, 1.0), "sbp": (50.0, 220.0, 1.0), "rr": (6.0, 45.0, 1.0),
               "spo2": (70.0, 100.0, 1.0), "temp": (34.0, 42.0, 0.1)}
LAB_RANGE = {"troponin": (0.0, 500.0, 1.0), "bnp": (0.0, 2000.0, 10.0),
             "d_dimer": (0.0, 5000.0, 50.0), "lactate": (0.0, 12.0, 0.1),
             "crp": (0.0, 300.0, 1.0), "wbc": (0.0, 30.0, 0.1),
             "creatinine": (0.0, 600.0, 5.0), "ph": (6.8, 7.8, 0.01)}

PRESETS = {
    "Blank": None,
    "Acute dyspnoea, key labs missing": dict(
        age=74, sex="Female", complaint="acute breathlessness", tier="high", tconf=0.79,
        organ="Lung", findings={"b_lines": 0.86},
        vitals={"hr": 118.0, "rr": 24.0, "spo2": 90.0}, labs={}, key="missing"),
    "Evidence agrees, record complete": dict(
        age=71, sex="Female", complaint="orthopnoea", tier="high", tconf=0.88,
        organ="Lung", findings={"b_lines": 0.92},
        vitals={"hr": 122.0, "rr": 28.0, "spo2": 88.0},
        labs={"troponin": 62.0, "bnp": 890.0, "lactate": 2.6}, key="concordant"),
    "Triage and imaging disagree": dict(
        age=68, sex="Male", complaint="mild chest discomfort", tier="low", tconf=0.88,
        organ="Heart", findings={"severe dysfunction": 0.74},
        vitals={"hr": 82.0, "sbp": 128.0, "spo2": 96.0},
        labs={"troponin": 5.0, "lactate": 1.1}, key="conflict"),
    "Everything negative": dict(
        age=44, sex="Female", complaint="mild breathlessness", tier="low", tconf=0.80,
        organ="Lung", findings={}, vitals={"hr": 78.0, "rr": 16.0, "spo2": 97.0},
        labs={"troponin": 4.0, "lactate": 1.0}, key="reassuring"),
    "Undifferentiated shock": dict(
        age=61, sex="Male", complaint="hypotension and collapse", tier="high", tconf=0.85,
        organ="Heart", findings={"severe dysfunction": 0.68},
        vitals={"hr": 132.0, "sbp": 78.0, "spo2": 92.0}, labs={"lactate": 5.2}, key=None),
}

st.set_page_config(page_title="POCUS Emergency Assistant", page_icon="🩺", layout="wide")

st.markdown("""
<style>
  /* Hospital software, not consumer AI: white ground, dark navy text, restrained blue.
     Amber means warning and red means critical, so colour carries clinical meaning. */
  html, body, [class*="css"] { font-family: Inter, "Segoe UI", system-ui, sans-serif; }
  .block-container { padding-top: 1rem; max-width: 1400px; }

  .topbar { background:#1f4e79; color:#fff; padding:.7rem 1.1rem; border-radius:6px;
            display:flex; justify-content:space-between; align-items:center;
            margin-bottom:.9rem; }
  .topbar .t { font-weight:600; letter-spacing:.01em; }
  .topbar .s { font-size:.82rem; opacity:.92; }

  .card { border:1px solid #dfe4ec; border-radius:6px; background:#fff;
          padding:.85rem 1rem; margin-bottom:.6rem; }
  .band { border-left:5px solid; border-radius:5px; padding:.85rem 1.05rem;
          margin:.25rem 0 .8rem 0; }
  .crit { background:#fdf1f0; border-color:#c0392b; }
  .warn { background:#fdf7ea; border-color:#c98a12; }
  .ok   { background:#f0f7f2; border-color:#2e7d4f; }
  .info { background:#f1f5fa; border-color:#1f4e79; }

  .kv { font-size:.86rem; color:#5a6b83; }
  .big { font-size:1.65rem; font-weight:600; color:#12263f; line-height:1.1; }
  .chip { display:inline-block; padding:.1rem .5rem; border-radius:3px;
          font-size:.7rem; font-weight:600; letter-spacing:.05em; margin-right:.35rem; }
  .live { background:#e3f0e8; color:#2e7d4f; }
  .rec  { background:#e9edf3; color:#48607e; }
  .off  { background:#fbe9e7; color:#b03a2e; }
  .row  { padding:.4rem .65rem; border-radius:4px; margin-bottom:.28rem;
          border:1px solid #e4e8ef; background:#fbfcfd; font-size:.88rem; }
  .gone { color:#8a95a5; background:#f7f8fa; }
  .muted{ color:#6b7a90; font-size:.84rem; }
  .step { padding:.25rem 0; font-size:.88rem; color:#2e7d4f; }
  .step.pending { color:#9aa5b4; }
  .sumbox { background:#f1f5fa; border:1px solid #cfdcea; border-left:5px solid #1f4e79;
            border-radius:5px; padding:.95rem 1.15rem; font-size:1rem; line-height:1.62;
            color:#12263f; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_frozen() -> dict:
    return json.loads(FROZEN.read_text(encoding="utf-8")) if FROZEN.exists() else {}


def banner(kind: str, title: str, lines) -> None:
    body = "".join(f"<div style='margin-top:.3rem'>• {x}</div>" for x in lines)
    st.markdown(f"<div class='band {kind}'><b>{title}</b>{body}</div>",
                unsafe_allow_html=True)


# =========================================================================== navigation
st.sidebar.markdown("## 🩺 POCUS Emergency Assistant")
st.sidebar.caption("Multimodal AI-assisted clinical decision support for emergency ultrasound")
_pages = ["New encounter", "Assessment", "System architecture"]
page = st.sidebar.radio("Navigation", _pages, label_visibility="collapsed",
                        index=_pages.index(st.session_state.pop("_goto", "New encounter")))
st.sidebar.divider()
# Demo mode guarantees a defence works without a GPU; live mode runs everything that can be
# run here. The distinction is stated rather than hidden, because which one is active changes
# what the differential on screen actually is.
mode = st.sidebar.radio(
    "Mode", ["Demo — precomputed encounters", "Live — this machine"],
    help="Demo replays differentials recorded from the GPU run, so the pipeline is "
         "reproducible in a presentation. Live computes everything this machine can and "
         "says plainly when a differential would need a GPU.")
DEMO = mode.startswith("Demo")

st.sidebar.divider()
st.sidebar.caption("**Not a diagnostic device.** Synthetic cases, no clinical ground truth, "
                   "never validated against patient outcomes.")


# =========================================================================== page 1
def page_intake() -> None:
    st.markdown("## New encounter")
    st.caption("Enter the patient. The assistant selects the pipeline from the presentation "
               "— the agents are not configured by hand.")

    # Keyed, so the choice survives navigating to the assessment and back.
    preset_name = st.selectbox("Start from an example", list(PRESETS), key="_preset_sel")
    P = PRESETS[preset_name] or dict(age=60, sex="Female", complaint="", tier="medium",
                                     tconf=0.7, organ="Lung", findings={}, vitals={},
                                     labs={}, key=None)

    # On a preset change the values are WRITTEN into session state and the widgets below read
    # them, rather than the widgets being given a `value=` default. Deleting a widget's key
    # was tried first and does not reset it: Streamlit re-registers the widget with the value
    # the browser still holds, so choosing a preset left every field exactly as it was.
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

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown("#### Patient")
        a, b = st.columns(2)
        age = a.number_input("Age", 0, 120, key="_age")
        sex = b.selectbox("Sex", ["Female", "Male"], key="_sex")
        complaint = st.text_input("Chief complaint", key="_complaint",
                                  placeholder="e.g. acute breathlessness")

        st.markdown("#### Triage")
        tier = st.select_slider("Assessed urgency", ["low", "medium", "high"],
                                key="_tier")
        tconf = st.slider("Triage confidence", 0.5, 1.0, step=0.01, key="_tconf")

        st.markdown("#### Vitals")
        st.caption("Untick a measurement to record it as **never measured** — which is not "
                   "the same as normal.")
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
        st.markdown("#### Ultrasound")
        organ = st.selectbox("Organ examined", ["Lung", "Heart", "Not performed"],
                             key="_organ")

        up = st.file_uploader("Ultrasound image (optional)",
                              type=["png", "jpg", "jpeg", "bmp"])
        if up is not None:
            st.image(up, width=240)
            st.markdown("<span class='chip off'>NOT ANALYSED</span> The perception models run "
                        "on GPU in the training notebooks and are not loaded in this "
                        "deployment. The image is shown only — enter below what the module "
                        "reported.", unsafe_allow_html=True)

        findings = {}
        if organ == "Lung":
            st.caption("Findings the lung module screened for. Unticked means **screened and "
                       "not seen**, which is itself an observation.")
            for f in LUNG_FINDINGS:
                c1, c2 = st.columns([1, 2])
                on = c1.checkbox(f.replace("_", " "), key=f"f_{f}")
                conf = c2.slider(" ", 0.0, 1.0, step=0.01, key=f"c_{f}",
                                 label_visibility="collapsed")
                findings[f] = conf if on else -conf
        elif organ == "Heart":
            c1, c2 = st.columns([1, 2])
            on = c1.checkbox("severe dysfunction", key="f_sd")
            conf = c2.slider(" ", 0.0, 1.0, step=0.01, key="c_sd",
                             label_visibility="collapsed")
            findings["severe dysfunction"] = conf if on else -conf

        st.markdown("#### Laboratory results")
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

    st.divider()
    if st.button("Analyze case", type="primary", use_container_width=True):
        st.session_state["encounter"] = dict(
            age=age, sex=sex, complaint=complaint, tier=tier, tconf=tconf, organ=organ,
            findings=findings, vitals=vitals, labs=labs, frozen_key=P.get("key"))
        st.session_state["_goto"] = "Assessment"
        st.rerun()


# =========================================================================== analysis
def analyse(enc: dict, break_model: bool):
    reports = {}
    if enc["organ"] == "Lung":
        det = [S.make_finding(k.replace("_", " "), v)
               for k, v in enc["findings"].items() if v > 0]
        neg = [S.make_finding(k.replace("_", " "), -v)
               for k, v in enc["findings"].items() if v <= 0]
        reports["lung"] = S.make_report(
            "lung", det, not_detected=neg, status="ok" if (det or neg) else "failed",
            reliability={"confidence_calibrated": True, "has_normal_class": False,
                         "modelled_findings": LUNG_FINDINGS,
                         "scope": "pneumothorax is NOT modelled and cannot be excluded"})
    elif enc["organ"] == "Heart":
        sd = enc["findings"].get("severe dysfunction", -0.2)
        reports["heart"] = S.make_report(
            "heart", [S.make_finding("severe dysfunction", sd)] if sd > 0 else [],
            not_detected=[] if sd > 0 else [S.make_finding("severe dysfunction", -sd)],
            reliability={"confidence_calibrated": True, "has_normal_class": True,
                         "scope": "CAMUS-like 4CH stills; EF is an area proxy"})

    state = build_clinical_state(
        {"encounter_id": "ENC-LIVE",
         "triage": S.make_triage(enc["tier"], enc["tconf"], features=enc["vitals"]),
         "ultrasound": reports,
         "clinical": {"age": enc["age"], "sex": enc["sex"],
                      "chief_complaint": enc["complaint"]}},
        labs=enc["labs"])

    esc = escalation_decision(state)
    support = decision_support(state, esc)
    hits = Retriever().for_state(state)

    if break_model:
        result, origin = reason(state, llm_fn=FailingBackend("out of memory"),
                                max_revisions=1), "failed"
    else:
        rec = (load_frozen().get("none") or {}).get(enc.get("frozen_key") or "")
        if rec:
            result, origin = dict(rec, decision_support=support), "recorded"
        else:
            result = reason(state, llm_fn=None)       # state + escalation only
            result["decision_support"] = support
            origin = "not_generated"
    return state, esc, support, hits, result, origin


# =========================================================================== page 2
def page_assessment() -> None:
    enc = st.session_state.get("encounter")
    if not enc:
        st.info("No encounter yet. Open **New encounter**, enter a patient, and select "
                "**Analyze case**.")
        return

    break_model = st.sidebar.toggle(
        "Simulate a model failure", value=False,
        help="Runs the pipeline for real with a backend that raises on every call.")

    state, esc, support, hits, result, origin = analyse(enc, break_model)
    sev = support["severity"]
    report = build_report(state, result, support)
    text = render_report(report)

    st.markdown(
        f"<div class='topbar'><div class='t'>POCUS-Emergency Clinical Decision Support</div>"
        f"<div class='s'>Encounter {report['encounter_id']} &nbsp;·&nbsp; Emergency "
        f"Department &nbsp;·&nbsp; ● {'DEMO' if DEMO else 'LIVE'} MODE</div></div>",
        unsafe_allow_html=True)

    side, main = st.columns([1, 3.4], gap="large")

    with side:
        st.markdown("##### Patient")
        st.markdown(f"<div class='big'>{enc['age']} y</div>"
                    f"<div class='kv'>{enc['sex']}</div>"
                    f"<div style='margin-top:.5rem'>{enc['complaint'] or '—'}</div>",
                    unsafe_allow_html=True)
        st.markdown("##### Triage")
        st.markdown(f"<div class='big'>{enc['tier'].upper()}</div>"
                    f"<div class='kv'>confidence {enc['tconf']:.2f}</div>",
                    unsafe_allow_html=True)

        st.markdown("##### Workflow")
        done = [
            ("Triage", True),
            (f"{enc['organ']} POCUS", enc["organ"] != "Not performed"),
            ("Clinical state", True),
            ("Escalation", True),
            ("Retrieval", bool(hits)),
            ("Reasoning", origin in ("recorded", "failed")),
            ("Decision support", True),
            ("Report", True),
        ]
        for label, ok in done:
            st.markdown(f"<div class='step{'' if ok else ' pending'}'>"
                        f"{'✓' if ok else '○'} {label}</div>", unsafe_allow_html=True)
        st.caption("A stage marked ○ did not run, and the assessment says where that leaves "
                   "the conclusion.")

    with main:
        st.markdown("### Case assessment")
        tone = {"HIGH": "crit", "MODERATE": "warn", "LOW": "ok"}[sev["severity"]]
        head = f"{sev['severity']} SEVERITY"
        if esc["escalate"]:
            head += "   —   ESCALATION REQUIRED"
        banner(tone, head,
               [f"{enc['age']}y {enc['sex'].lower()} — {enc['complaint'] or 'no complaint given'}",
                f"routed as: {support['scenario']['label']}"] + sev["reasons"])

        st.markdown("#### Assistant summary")
        st.markdown(f"<div class='sumbox'>{report['conclusion']}</div>", unsafe_allow_html=True)
        st.caption("Assembled from the record by fixed rules rather than written by the language "
                   "model, so it cannot say anything the rest of the assessment does not.")

        st.markdown("#### Critical alerts")
        ncrit = sum(1 for a in support["alerts"] if a["severity"] == "CRITICAL")
        st.markdown(f"<span class='chip live'>DETERMINISTIC</span> {ncrit} critical, "
                    f"{len(support['alerts']) - ncrit} warning — each names the bound it crossed, "
                    f"from thresholds v{support['thresholds_version']}", unsafe_allow_html=True)
        if not support["alerts"]:
            st.success("No alert. No measured value crossed a configured bound, and the record "
                       "shows no structural gap.")
        for a in support["alerts"]:
            (st.error if a["severity"] == "CRITICAL" else st.warning)(
                f"**{a['type'].replace('_', ' ')}** — {a['message']}")

    st.divider()
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("#### Ultrasound interpretation")
        det = [f for f in state["imaging"]["findings"] if f["detected"]]
        neg = [f for f in state["imaging"]["findings"] if not f["detected"]]
        for f in det:
            st.markdown(f"<div class='row'>✓ <b>{f['label']}</b> &nbsp;<span class='muted'>"
                        f"{f['organ']} · confidence {f['confidence']:.2f} · evidence "
                        f"{f['evidence']}</span></div>", unsafe_allow_html=True)
        if not det:
            st.markdown("<div class='row muted'>no finding detected</div>",
                        unsafe_allow_html=True)
        for f in neg:
            st.markdown(f"<div class='row gone'>○ {f['label']} <span class='muted'>— screened "
                        f"for and not seen ({f['confidence']:.2f})</span></div>",
                        unsafe_allow_html=True)
        for organ in state["imaging"]["organs_not_assessed"] or []:
            st.error(f"**{organ} — NOT ASSESSED.** A gap in the record, not a negative result.")
        for lim in state["imaging"]["out_of_scope"]:
            st.warning(f"Model limitation — {lim}")

    with c2:
        st.markdown("#### Missing information")
        gone = state["missing"]["labs"] + state["missing"]["vitals"]
        if not gone:
            st.success("Every value in the reference set was measured.")
        else:
            st.markdown("<span class='chip off'>NOT MEASURED ≠ NORMAL</span> These have no "
                        "evidence identifier, so the reasoning layer cannot cite them for or "
                        "against any diagnosis.", unsafe_allow_html=True)
            for g in gone:
                st.markdown(f"<div class='row gone'>⚠ {g}</div>", unsafe_allow_html=True)

    st.divider()
    st.markdown("#### Clinical reasoning")

    if origin == "failed":
        st.markdown("<span class='chip live'>LIVE</span> executed now with a failing backend",
                    unsafe_allow_html=True)
        banner("warn", "DIFFERENTIAL WITHHELD", result["validation_errors"] or [])
        st.success(f"The safety decision survived: **{sev['severity']} severity, "
                   f"{'escalation required' if esc['escalate'] else 'no escalation'}**. It is "
                   "computed from the record before the model is consulted, so it does not "
                   "depend on the model running at all.")
    elif origin == "not_generated":
        st.info("**No differential generated for this encounter.** Producing one requires a "
                "4.9 GB language model on a GPU, which is not loaded in this deployment. "
                "Everything else on this page is computed here from the record. Choose one of "
                "the example encounters to see a differential recorded from the GPU run.")
    else:
        secs = result.get("_seconds")
        st.markdown("<span class='chip rec'>RECORDED</span> HuatuoGPT-o1-8B, Colab T4"
                    + (f", {secs:.0f}s" if secs else "") + ", temperature 0",
                    unsafe_allow_html=True)
        diff = result.get("differential") or {}
        if result.get("differential_withheld"):
            banner("warn", "DIFFERENTIAL WITHHELD — did not pass validation",
                   result.get("validation_errors") or [])
        for i, d in enumerate(diff.get("differential") or [], 1):
            st.markdown(f"**{i} — {d.get('diagnosis')}** &nbsp;<span class='muted'>likelihood "
                        f"{d.get('likelihood')}</span>", unsafe_allow_html=True)
            for eid, t in zip(d.get("supporting_ids") or [], d.get("supporting") or []):
                st.markdown(f"<div class='row'>supports &nbsp;<code>{eid}</code> {t}</div>",
                            unsafe_allow_html=True)
            for t in (d.get("contradicting") or []):
                st.markdown(f"<div class='row'>against &nbsp;{t}</div>", unsafe_allow_html=True)
            for t in (d.get("limitations") or []):
                st.caption(f"limitation — {t}")
        if diff.get("uncertainty"):
            st.caption(f"uncertainty — {diff['uncertainty']}")

    considered = result.get("evidence_considered") or []
    if considered:
        used = sum(1 for r in considered if r["used"])
        with st.expander(f"Clinical evidence considered — {used} of {len(considered)} cited "
                         f"by the model"):
            st.caption("Built from the record, not from the model's answer: every abnormal "
                       "value appears whether the model reasoned from it or not.")
            for r in considered:
                st.markdown(f"<div class='{'row' if r['used'] else 'row gone'}'>"
                            f"{'✱' if r['used'] else '&nbsp;&nbsp;'} <code>{r['id']}</code> "
                            f"{r['text']}</div>", unsafe_allow_html=True)

    st.divider()
    c3, c4 = st.columns(2, gap="large")

    with c3:
        st.markdown("#### Suggested next steps")
        recs = support["additional_examinations"]
        if not recs:
            st.success("The record is complete for this presentation.")
        for i, r in enumerate(recs[:8], 1):
            st.markdown(f"**{i}. {r['exam']}** &nbsp;<span class='muted'>[{r['priority']}]"
                        f"</span><br><span class='muted'>{r['reason']}</span>",
                        unsafe_allow_html=True)
        st.caption("Decision-support suggestions. The system recommends examinations; it does "
                   "not order them, and they do not replace clinician judgement.")

    with c4:
        st.markdown("#### Therapeutic considerations")
        th = support["therapeutic"]
        if not th["considerations"]:
            st.info(f"**No protocol-backed therapeutic recommendation for this case.**\n\n"
                    f"Reason: {th['status']}.")
            if th.get("note"):
                st.caption(th["note"])
        for c_ in th["considerations"]:
            st.markdown(f"**CONSIDER** — {c_['consideration']}")
            st.caption(f"Basis: {c_['basis']} [{c_['passage']}]")
            st.caption(c_["disclaimer"])

    st.divider()
    with st.expander(f"📚 Evidence retrieved — {len(hits)} passage(s)"):
        if not hits:
            st.info("No sufficiently relevant passage was found. Reasoning was performed from "
                    "the structured clinical state only, and the answer is not marked as "
                    "guideline-grounded.")
        for h in hits:
            st.markdown(f"**[{h['n']}] {h['topic']}** &nbsp;<span class='muted'>relevance "
                        f"{h['score']:.2f} · {h['id']}</span>", unsafe_allow_html=True)
            st.caption(h["source"])
            st.markdown(f"<div class='row'>{h['text']}</div>", unsafe_allow_html=True)

    st.markdown("#### 📄 Clinical report")
    r1, r2 = st.columns([3, 1])
    with r1:
        rows = ["clinical state recorded", "alerts recorded",
                "reasoning recorded" if origin != "not_generated"
                else "reasoning NOT generated (no GPU) — recorded as such",
                "evidence sources recorded", "escalation decision recorded"]
        st.markdown("".join(f"<div>✓ {x}</div>" for x in rows), unsafe_allow_html=True)
    with r2:
        st.download_button("Export report", text, use_container_width=True,
                           file_name=f"{report['encounter_id']}_report.txt")
    with st.expander("View full report"):
        st.code(text, language=None)


# =========================================================================== page 3
def page_architecture() -> None:
    st.markdown("## System architecture")
    st.caption("Every stage below runs in this deployment except the language model, which "
               "needs a GPU, and the perception models, which run in the training notebooks.")
    st.code("""
                    Patient / Clinician
                            |
                            v
                     Input interface
                            |
                +-----------+-----------+
                v                       v
           Triage Agent            POCUS Agents
        urgency from tabular    +------+------+
        data, never an image    v      v      v
                |              Lung  Cardiac  Gallbladder
                |               |      |      |
                +---------------+------+------+
                                       |
                                       v
                              CLINICAL STATE
                    findings . values . what is MISSING
                          conflicts . case quality
                                       |
            +--------------------------+--------------------------+
            v                          v                          v
      ESCALATION               SCENARIO ROUTER            CRITICAL ALERTS
      7 rules, computed        dyspnoea / shock /         physiological and
      BEFORE the model runs    trauma / arrest            structural
            |                            |                        |
            |                            v                        |
            |                           RAG                       |
            |              TF-IDF with a relevance floor          |
            |                            |                        |
            |                            v                        |
            |                    HuatuoGPT-o1-8B                  |
            |            cites enumerated evidence only           |
            |                            |                        |
            |                            v                        |
            |                       VALIDATOR                     |
            |             deliver . revise . withhold             |
            |                            |                        |
            +--------------------------+-+------------------------+
                                       v
                              DECISION SUPPORT
              severity . alerts . examinations . therapeutics
                                       |
                                       v
                             AUTOMATED REPORT
                     conclusion . archived with a hash
""", language=None)

    st.markdown("#### The two separations that organise it")
    a, b = st.columns(2, gap="large")
    a.markdown("**Perception is separate from reasoning.** The ultrasound agent reports what "
               "it sees and never assigns urgency — it cannot see the vitals, laboratory "
               "results or history that the decision requires.")
    b.markdown("**Rules are separate from the model.** Escalation, severity, alerts and what "
               "is missing are computed deterministically before the language model is "
               "consulted, and the model's output is checked against the same record "
               "afterwards.")

    st.markdown("#### What the model can and cannot say")
    st.markdown(
        "- Evidence is cited by **identifier**, drawn from an enumerated list of facts the "
        "record holds. There is no identifier for an invented observation, none for a test "
        "never performed, and none for a sentence out of the reference corpus.\n"
        "- A value's qualifier travels with its identifier, so a normal troponin cannot be "
        "relabelled as elevated on the way through.\n"
        "- Therapeutic suggestions are gated behind a citable protocol. With none retrieved "
        "the assistant produces no recommendation, rather than drawing on the model's own "
        "training knowledge.")

    st.markdown("#### Verification")
    v1, v2, v3 = st.columns(3)
    v1.metric("Safety tests", "212", "24 properties")
    v2.metric("End-to-end checks", "150", "10 cases")
    v3.metric("Corpus units", "32", "all sourced")
    st.caption("These are software tests on synthetic cases. They establish that the safety "
               "mechanisms behave as specified and say nothing about diagnostic accuracy. "
               "Clinical validation against expert ground truth is outside the scope of this "
               "work.")


# =========================================================================== router
{"New encounter": page_intake,
 "Assessment": page_assessment,
 "System architecture": page_architecture}[page]()
