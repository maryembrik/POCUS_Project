# POCUS-Emergency: Multimodal AI Assistant for Emergency Ultrasound Diagnosis & Decision Support
### Full project context & handoff document — compiled for continued development (e.g. with Claude Code)

**Author:** Maryem Brik — AI Research Intern, RFID & Sensors / ESPRIT, Tunis, Tunisia
**Internship duration:** started Jul 1, 2026 (3-month internship; core build cycle compressed to ~7.5 weeks)
**Document compiled:** Jul 9, 2026

---

## 1. Project Overview

An AI assistant that analyzes point-of-care ultrasound (POCUS) images together with clinical, physiological, and biological data to support emergency physicians — with an agentic escalation path (Digital Twin → RL Agent → VR Simulation) for ambiguous or high-stakes cases.

The original vision (from the source specification document, `POCUS_EchoetAI-en.docx`) is a multi-year, multi-organ, multicentric clinical AI program. This project scopes that vision down to a defensible 3-month internship deliverable: **one organ system built deep (pulmonary), the rest architected as extensible stubs.**

---

## 2. Business Objective (BO) & Data Science Objective (DSO)

**Problem statement:** How to integrate multimodal AI into POCUS to improve diagnostic accuracy, assist clinicians, and strengthen emergency care quality?

**Business Objective (BO):**
Reduce time-to-diagnosis and diagnostic error rate for critically ill emergency patients by giving POCUS operators an AI co-pilot that interprets images, cross-checks clinical data, and — for ambiguous or high-stakes cases — lets the physician rehearse the decision in simulation before committing.

**Data Science Objective (DSO):**
- A computer vision model that classifies/detects defined pathologies from POCUS clips, with a calibrated confidence score.
- An explainability layer (Grad-CAM) tying model attention to anatomical regions clinicians can verify.
- An LLM + RAG layer that turns image findings + clinical/bio data into a ranked differential diagnosis with grounded reasoning.
- An escalation policy — owned by the Clinical Reasoning Agent, not a separate orchestrator — deciding, from confidence + severity + data completeness, whether a case is straightforward or needs simulation.
- A simulation loop (Digital Twin → RL Agent → VR) letting the physician explore "what if I do X vs Y" before the real decision.

---

## 3. Scope & Key Decisions (chronological decision log)

| Date | Category | Decision | Rationale |
|---|---|---|---|
| 2026-07-03 | Scope | Deep-build pulmonary ultrasound; architect (not build) cardiac/FAST/vascular as stubs | Strongest public data availability (Butterfly Network, POCOVID-Net/ICLUS-style datasets) and a clean, well-studied pathology set. A 3-month solo internship can't support real depth across 4 organ systems — one deep + three architected stubs preserves the multi-organ ambition without shallow, non-functional prototypes. |
| 2026-07-03 | Data | Data strategy: transfer learning on mixed public + real hospital data | Pretrain/validate CV backbone on public lung ultrasound datasets for volume and pathology diversity, then fine-tune and clinically validate on the real, de-identified hospital subset. Gives a real domain-adaptation story for the report. |
| 2026-07-03 | Tech stack | VR default: web-based WebXR/3D mockup, not a headset build | Buildable solo in the available time, no hardware procurement or VR lab dependency, still demonstrates spatial outcome exploration. Upgrades to a real headset session later almost for free since WebXR supports both. |
| 2026-07-03 | Architecture | Agent architecture: functional pipeline (Triage → Ultrasound → Clinical Reasoning) instead of organ-specific specialist agents | Matches real ED workflow order (triage happens before/alongside the scan). Avoids duplicating diagnostic reasoning logic four times across organ agents. Organ specialization now lives inside the Ultrasound Agent as internal routing (lung trained, others stubbed) instead of as separate top-level agents. |
| 2026-07-03 | Architecture | Escalation decision owned by Clinical Reasoning Agent — no separate orchestrator agent | The Clinical Reasoning Agent already fuses triage urgency + ultrasound findings + clinical/bio data, so the escalate/don't-escalate call is a natural extra output of that same agent rather than a fourth agent. |
| 2026-07-03 | Tech stack | Compute budget assumption: modest single-GPU / Colab-tier | No confirmed access to a GPU cluster or cloud credits yet. Defaulting to lightweight architectures (e.g. EfficientNet/MobileNet-based). **Open question — revisit once actual compute is confirmed.** |
| 2026-07-03 | Data | Real hospital data volume not yet confirmed | **Open question** — need to confirm how many real, de-identified cases are actually accessible (dozens vs hundreds), and their format/quality, before finalizing the fine-tuning plan. |
| 2026-07-03 | Architecture | Triage Agent assesses patient urgency, not data routing — clarified | Corrected a scope misunderstanding: Triage reads vitals/chief complaint to produce an urgency level, exactly like an ER nurse's first-pass read. It does not decide what data feeds the Ultrasound Agent. This is what makes running Triage and Ultrasound in parallel valid. |
| 2026-07-03 | Architecture | Triage Agent and Ultrasound Agent execute in parallel | No data dependency between them (vitals/complaint vs. image). Both feed into the Clinical Reasoning Agent, which is the only agent that needs both outputs before it can act. |
| 2026-07-03 | Architecture | Digital Twin scoped as a single bounded module: doctor proposes treatment → twin simulates → AI predicts outcome → doctor decides | Keeps the most ambitious/riskiest component from becoming the whole project. Explicitly a decision-support loop around one patient's current state, not a full physiological simulation platform. |
| 2026-07-03 | Architecture | RL Agent trains only inside the Digital Twin sandbox, never on real patients | The digital twin is the RL agent's entire training environment. Its output is a suggested treatment sequence for the doctor — never an autonomous action on a real patient. Explicitly a research-quality proof of concept. |
| 2026-07-03 | Architecture | VR Simulation is the interface layer over Digital Twin + RL, not a separate reasoning engine | VR's job is letting the doctor see and step through outcomes the Digital Twin already computed — it doesn't add new intelligence. |
| 2026-07-05 | Scope | Adopt CRISP-DM as the project's data science methodology | Matches the BO/DSO vocabulary already used. Leaner than TDSP (which adds team-role/repo conventions more relevant to multi-person teams). Widely recognized, so a supervisor/reviewer understands the structure immediately. |
| 2026-07-06 | Scope | Compress project timeline to ~7.5 weeks across 4 phases, with report running in parallel | Business Understanding + Data (2 wks) → Modeling (1.5 wks) → Deployment (2 wks) → Testing (2 wks). A tighter working cycle within the 3-month internship, leaving remaining time as buffer. Report written in parallel throughout, not as a final phase. |

---

## 4. System Architecture (current, corrected version)

### 4.1 Three functional agents — Triage and Ultrasound run in parallel

```
[Vitals + chief complaint]        [POCUS image/video]
         │                                │
         ▼                                ▼
   TRIAGE AGENT                  ULTRASOUND AGENT
 (determines urgency)         (detects findings, NOT urgency)
         │                                │
         └───────────────┬────────────────┘
                          ▼
              CLINICAL REASONING AGENT
     (fuses triage + ultrasound + clinical/bio data
      via LLM + RAG → ranked diagnosis + escalation call)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
  Return recommendation        Escalate to simulation
  (straightforward case)                │
                                         ▼
                                  DIGITAL TWIN
                          (simulates patient outcomes)
                                         │
                                         ▼
                                    RL AGENT
                        (learns/suggests decision policy,
                         sandboxed — never touches real patients)
                                         │
                                         ▼
                                 VR SIMULATION
                     (WebXR/3D mockup — doctor explores outcomes)
                                         │
                                         ▼
                          DOCTOR'S FINAL DECISION
                          (informed by AI throughout)
```

**1. Triage Agent** — reads the *patient*, not the image. Input: chief complaint, vitals (HR, BP, SpO2, RR), age, red-flag symptoms. Output: an urgency/severity level. Can start as a rules-based/calibrated classifier (e.g. simplified ESI/qSOFA-style scoring) — doesn't need deep learning for the MVP.

**2. Ultrasound Agent** — reads the *image*, not the chart. Input: POCUS image/video. Internally routes by organ:
- **Lung → fully trained this internship** (CV model + Grad-CAM)
- **Cardiac / FAST-E-FAST / Vascular → defined stub interfaces**, same input/output contract, model not yet trained — same schema so future work just implements the interface.
Output: detected findings + confidence — **not** an urgency verdict.

*Triage and Ultrasound run in parallel: different inputs, neither depends on the other's output.*

**3. Clinical Reasoning Agent** — the fusion step. Plain-language description: the senior doctor who takes the nurse's rapid read (Triage) and the imaging report (Ultrasound), lines them up against clinical/biological data (age, sex, symptoms, history, blood gas, lactates, troponins, D-dimer, NFS/CBC), and writes the actual diagnostic story via LLM + RAG (grounded in a curated emergency medicine knowledge base). Outputs a ranked differential diagnosis *and* the escalation call: if confidence is low, or Triage's urgency and Ultrasound's findings disagree (e.g. reassuring vitals but a large pneumothorax on screen), escalate to simulation instead of returning directly.

### 4.2 Escalation path — Digital Twin is one bounded module, not the project

```
Doctor proposes a treatment
        │
        ▼
DIGITAL TWIN simulates it
(lightweight virtual copy of THIS patient's current state —
 not full organ physiology, just key parameters like
 SpO2/BP/lactate trend evolving under 2-3 candidate interventions)
        │
        ▼
AI predicts the expected outcome
        │
        ▼
Doctor reviews and makes the real decision
```

- **RL Agent** trains entirely inside the Digital Twin sandbox — never touches a real patient. Runs many simulated treatment sequences to learn which lead to good outcomes; its trained policy becomes a *suggestion generator* for the doctor. Explicitly research-quality proof of concept, not bedside-ready.
- **VR Simulation** is the interface layer on top of Digital Twin + RL — not a separate reasoning engine. Lets the doctor see and step through simulated outcome branches spatially. Default: web-based 3D/WebXR mockup (no headset required to demo); upgradeable to a real headset session later since WebXR supports both.

### 4.3 Multi-organ extensibility

Only Pulmonary is deep-built this internship. Cardiac, Abdominal, and Vascular exist as **defined stub interfaces** inside the Ultrasound Agent's internal router — same input/output schema as the lung model, no trained model behind them yet. This is a deliberate scope decision (see Section 3), not an oversight.

---

## 5. Data Science Methodology — CRISP-DM

Chosen over alternatives (e.g. Microsoft's TDSP) because it matches the BO/DSO vocabulary already used, is leaner for a solo intern, and is the most widely recognized framework in data science.

| Phase | Status (as of Jul 9, 2026) | What it means here |
|---|---|---|
| 1. Business Understanding | ✅ Done | BO/DSO, pulmonary-first scope decision |
| 2. Data Understanding | 🔵 In progress | Datasets Registry work — finding sources, verifying contents, flagging gaps |
| 3. Data Preparation | ⏳ Next | Cleaning, labeling, manifest-building, harmonizing POCUS + ICLUS + BEDLUS |
| 4. Modeling | ⏳ Upcoming | CV model, Triage model, LLM/RAG, RL agent |
| 5. Evaluation | ⏳ Upcoming | Sensitivity, specificity, analysis time, concordance with expert-labeled cases |
| 6. Deployment | ⏳ Upcoming | Full pipeline integration, VR/WebXR mockup, final demo |

CRISP-DM is a **cycle**, not a strict line — a modeling result can send you back to Data Preparation, or even Business Understanding. The weekly Task Tracker (Notion) is the agile/sprint execution layer running on top of these phases.

---

## 6. Datasets & Resources Registry (full detail)

### 6.1 In hand and verified (7)

| Name | Category | What it offers |
|---|---|---|
| **POCUS Dataset (Born et al.)** [`jannisborn/covid19_ultrasound`](https://github.com/jannisborn/covid19_ultrasound) | Ultrasound - Pulmonary | 261 recordings (202 videos + 59 images), 216 patients: COVID-19, healthy, bacterial pneumonia, viral pneumonia. Expert pattern annotations (B-lines, consolidations). **Primary lung training set.** Confirmed real videos/images inside `covid19_pocus_ultrasound-master.zip`, organized in `data/pocus_videos/{convex,linear}/` and `data/pocus_images/{convex,linear}/`, class encoded in filename prefix (`Cov-`/`Reg-`/`Pneu-`/`Vir-`). Ignore the bundled `pocovidscreen` web app. |
| **USCL pretrained backbone (US-4)** [`983632847/USCL`](https://github.com/983632847/USCL) | Pretrained Model | MIT license. ResNet-18 pretrained on ultrasound-domain data (not ImageNet) — scored 94.19% fine-tuning accuracy on POCUS vs ~84% from ImageNet pretraining. US-4 pretraining data covers only **lung + liver** (not "4 organs" as loosely first described) via 4 sub-datasets: Butterfly (lung, public), COVID19-LUSMS (lung, private), Liver Fibrosis (liver, private), CLUST (liver, public tracking challenge). Classifier head trained/evaluated on the original 3-class POCUS problem (COVID-19 / bacterial pneumonia / healthy) — **use as a feature-extractor backbone**, not a finished classifier for our 4 target pathologies. Code obtained; `best_model.pth` weights + 5-fold POCUS split still need separate download from the Google Drive links in the README. |
| **NHAMCS** [cdc.gov/nchs/nhamcs](https://www.cdc.gov/nchs/nhamcs/) | Triage / Vitals | Real US ED visit survey data, CDC/NCHS. No registration. Demographics, reason-for-visit, ICD diagnoses, disposition. Survey-sampled (not one continuous linked record); vitals coverage varies by year. **Raw file is fixed-width ASCII — needs the CDC's SAS/format dictionary to parse into columns** (not yet obtained). |
| **MIMIC-IV-ED Demo (+ Clinical Database Demo)** [physionet.org](https://physionet.org/content/mimic-iv-ed-demo/) | Triage / Vitals | 100-patient open subset, zero registration, same schema as full MIMIC-IV/MIMIC-IV-ED (excludes free-text notes). Confirmed tables: `triage`, `vitalsign`, `edstays`, `diagnosis`, `medrecon`, `pyxis` (ED demo) + `labevents`, `admissions`, `diagnoses_icd`, `patients`, `prescriptions` (hospital demo). Build/debug the Triage Agent pipeline now; swap in full dataset later with no code changes. |
| **MTSamples** [mtsamples.com](https://www.mtsamples.com) | Clinical Notes | ~5,000 free-text medical transcription reports, 40 specialties, includes Emergency Room Reports. CC0 public domain. Confirmed columns: `description, medical_specialty, sample_name, transcription, keywords`. Good for prototyping history/exam-text extraction — not linked to the same patients as vitals/labs data. |
| **Auto-US** (reference only) [`Bean-Young/Auto-US`](https://github.com/Bean-Young/Auto-US) | Reference Architecture | Classification/CNN-Transformer code marked "coming soon" — not usable. **BUT** the `OurAgent` folder has a real, directly usable LLM prompt template for the Clinical Reasoning Agent: feeds model result + chief complaint + physical exam + extra info → preliminary diagnosis + justification + recommended follow-up exams. Tested with DeepSeek-R1-7B (open-weight, self-hostable). Includes worked case examples (`cases.docx`) and a weighted evaluation formula (expert score + non-expert score + METEOR text-similarity) worth adapting for our own eval metrics. |
| **CVA-Net (BUV breast ultrasound)** (reference only) [`jhl-Det/CVA-Net`](https://github.com/jhl-Det/CVA-Net) | Ultrasound - Other organ | MICCAI 2022. "Clip-level and Video-level feature Aggregated Network" — a bespoke video object detector for breast lesions (tracks a lesion box across neighboring frames + video-level benign/malignant classification), not a generic backbone. Full code + pretrained weights, but BUV dataset itself not bundled. Non-commercial license. Outside pulmonary-first scope and outside the original 4-organ list — useful only as an architecture reference for video-based detection (different problem shape than our classification-based organs: detection/tracking vs. whole-frame classification). |

### 6.2 Identified but not yet fully in hand (7)

| Name | Category | Why it's not in hand | What it would add |
|---|---|---|---|
| **ICLUS-DB** [iclus-db.imedlab.org](https://www.iclus-db.imedlab.org) | Ultrasound - Pulmonary | Needs a request | 277 videos, 58,924 frames, 35 patients. Standardized 4-level severity scoring (0=healthy to 3=severe) — closest thing to a clinical gold standard for validating model confidence against real severity. |
| **COVIDx-US** [`nrc-cnrc/COVID-US`](https://github.com/nrc-cnrc/COVID-US) | Ultrasound - Pulmonary | Repo does **not** host actual data — only masks + metadata + a Selenium scraper notebook (`create_COVIDxUS.ipynb`) that pulls videos live from external sources (Butterfly, GrepMed, POCUS Atlas, LITFL, Radiopaedia, CoreUltrasound). Must run the notebook; some 2022-era sources may be stale. | 242 videos, 29,651 images — pathology diversity beyond COVID-era labels (pneumonia, other lung disease, normal). |
| **BEDLUS (Boston ED Lung Ultrasound)** [Harvard Dataverse](https://dataverse.harvard.edu) | Ultrasound - Pulmonary | Gated behind a Harvard Dataverse access request — **reported as hard to obtain.** | 1,419 videos, 188,670 frames, 113 **real ED patients** (not a COVID cohort) — most clinically relevant pulmonary source. Labeled for B-lines, 15,755 individual B-lines annotated on 10,371 frames. Companion repo [`RTLucassen/B-line_detection`](https://github.com/RTLucassen/B-line_detection) (IEEE JBHI 2023) has a **fully open training pipeline** (preprocessing + dataset generation + clip/frame/pixel-level network training) usable as a reference architecture even without the gated data/pretrained weights. |
| **Mendeley Gallbladder Dataset (UIdataGB)** [data.mendeley.com](https://data.mendeley.com/datasets/r6h24d2d3y/1) | Ultrasound - Abdominal | CC BY 4.0, fully open — **user reports downloaded but upload failed (connection issue); not yet verified.** | Real ultrasound images, 9 gallbladder disease classes. Not FAST/trauma-style (free-fluid detection) but real data for the currently-empty Abdominal stub. |
| **PhysioNet Challenge 2019 (Sepsis)** [physionet.org](https://physionet.org/content/challenge-2019/) | Biological / Labs | Open, no DUA — just hasn't been downloaded yet | 40,336 ICU patients, hourly: 8 vitals + 26 labs (**includes lactate and troponin**, plus blood-gas components) + demographics. ICU not ED, but ideal for prototyping the Clinical Reasoning Agent's biological-data fusion logic. |
| **VitalDB** [vitaldb.net](https://vitaldb.net) | Biological / Labs | Codebook (parameter list) verified in hand; **actual per-patient case data reported downloaded but upload failed (connection issue), not yet verified.** | 6,388 surgical patients. Confirmed via official codebook: full CBC, complete blood gas panel, lactate, coagulation panel — but **NO troponin, NO D-dimer** at all, and **100% surgical/anesthesia context** (zero ED/chief-complaint fields). Use only as generic vitals+labs fusion code-practice data, not an ED case stand-in. |
| **MIMIC-IV-ED (full)** + **MIMIC-IV-Note** [physionet.org](https://physionet.org/content/mimic-iv-ed/) | Triage/Vitals + Clinical Notes | **Credentialed** — needs CITI training + signed data use agreement, ~1-2 week turnaround each, separate applications. Start both now. | Full ~425,000 ED stays (ESI acuity, chief complaint, vitals, demographics) + linked free-text discharge/physician notes. |

### 6.3 Reference-only / proof-of-concept (1)

| Name | Category | Role |
|---|---|---|
| **LUCPD** (phantom pneumothorax dataset) | Ultrasound - Pulmonary | Simulator/phantom-based, not real patients. Dedicated pleural-line/A-line/B-line classes. **Only lead found for pneumothorax** — one of the 4 target pathologies with almost no public real-patient data. Proof-of-concept only. |

### 6.4 Real gaps that no amount of downloading fixes

- **Pneumothorax and pleural effusion** — two of the four target pathologies — have essentially no public real-patient dataset behind them. Build/validate B-lines/interstitial-syndrome/pneumonia first on real data; treat pneumothorax as a stretch goal unless real hospital data materializes.
- **Cardiac and Vascular organs** — zero datasets identified at all (not just "not downloaded" — not seriously searched). Remain pure architecture stubs.

---

## 7. Project Timeline (compressed cycle: ~7.5 weeks, Jul 1 – Aug 21, 2026)

Internship actually started **July 1, 2026** (not July 6 — corrected after initial miscalculation). Four sequential phases; the internship report runs in parallel across the entire cycle, not as a separate final phase.

| Phase | Duration | Dates | Focus |
|---|---|---|---|
| **Business Understanding + Data** | 2 weeks | Jul 1 – Jul 14 | BO/DSO document, agent interface contracts, literature review, data audit + collection + cleaning/preprocessing across pulmonary sources (POCUS, ICLUS, BEDLUS, COVIDx-US) and non-imaging sources (MIMIC-IV-ED demo, NHAMCS, MTSamples) |
| **Modeling** | 1.5 weeks | Jul 15 – Jul 24 | Lung CV model on the USCL pretrained backbone + Grad-CAM; Triage Agent; LLM + RAG Clinical Reasoning Agent v1; stub interfaces for cardiac/FAST/vascular |
| **Deployment** | 2 weeks | Jul 25 – Aug 7 | Escalation logic, toy Digital Twin + RL agent proof-of-concept (acute dyspnoea differential), WebXR/3D outcome mockup, full pipeline integration |
| **Testing** | 2 weeks | Aug 8 – Aug 21 | Evaluation metrics (sensitivity, specificity, analysis time, concordance), end-to-end testing + bug fixing, scale-up roadmap |
| **Internship report** | Parallel, whole cycle | Jul 1 – Aug 21 | Intro/BO-DSO section (end of Phase 1) → data & methodology section (end of Phase 2) → architecture & results section (end of Phase 3) → final compilation + defense prep (end of Phase 4) |

**Known risk flagged:** Modeling at 1.5 weeks is tight for CV model + Triage Agent + LLM/RAG + stub interfaces all at once — most likely phase to run over. Deployment/Testing have more schedule cushion if it slips.

---

## 8. Current Status Summary (as of Jul 9, 2026)

**Ready to build on right now:**
- Pulmonary CV training data (POCUS dataset) — verified, real videos+images in hand
- USCL pretrained backbone code — in hand, weights need one more download step
- Triage Agent prototyping data — MIMIC-IV-ED Demo + NHAMCS (needs format dictionary) in hand
- Clinical Reasoning Agent prompt design reference — Auto-US template + DeepSeek-R1-7B precedent
- Clinical notes prototyping data — MTSamples in hand

**Blocked or pending (not code-blocking, can work around):**
- Full MIMIC-IV-ED + MIMIC-IV-Note — credentialing in progress (start applications ASAP)
- BEDLUS — Harvard Dataverse access reported hard; use companion GitHub repo's training pipeline as a workaround reference in the meantime
- ICLUS-DB — access request not yet submitted
- COVIDx-US — scraper notebook not yet run
- Mendeley Gallbladder + VitalDB case data — reported downloaded by user, upload verification pending (connection issues)

**Genuinely open, unresolved:**
- Real hospital data volume — unknown, unconfirmed
- Compute budget — assumed Colab-tier/single-GPU until proven otherwise
- Pneumothorax/pleural effusion real-patient data — does not meaningfully exist publicly
- Cardiac/Vascular datasets — not identified at all

---

## 9. Notes for continued development (e.g. with Claude Code)

- **First real coding task**: build a data manifest script that walks the in-hand POCUS dataset (`data/pocus_videos/`, `data/pocus_images/`) and produces a clean table of `filepath → class → probe_type (convex/linear) → source`. This is the foundation both the CV model and the eventual multi-source harmonization (adding ICLUS/COVIDx-US/BEDLUS later) will build on.
- **CV model**: start from the USCL ResNet-18 backbone (once weights are pulled), replace its 3-class head, fine-tune on the 4 target pathologies from the manifest above. Attach Grad-CAM after a working baseline, not before.
- **Triage Agent**: can be a simple rules-based/calibrated classifier on the MIMIC-IV-ED Demo schema (vitals + chief complaint + ESI acuity) — no deep learning needed for the MVP.
- **Clinical Reasoning Agent**: adapt the Auto-US prompt template structure (model result + chief complaint + physical exam + extra info → diagnosis + justification + follow-ups) as the starting prompt design; RAG knowledge base still needs to be curated (not yet started).
- **Digital Twin / RL / VR**: not yet started at all — scoped for the Deployment phase (Jul 25 – Aug 7). Keep the loop bounded exactly as specified in Section 4.2 — resist scope creep here specifically, it's the highest risk of overrun.
- **Everything above is also tracked live in Notion** (Project Charter, Decision Log, Datasets Registry, Task Tracker with phase-based board view) — this document is a point-in-time snapshot for portability, Notion remains the source of truth going forward.
