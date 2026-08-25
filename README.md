# POCUS-Emergency

Intelligent multimodal assistant for clinical ultrasound and decision support in emergencies.

Point-of-care ultrasound is fast and available at the bedside, but operator-dependent, and a
single imaging finding rarely determines a decision on its own. This project builds the layer
above the individual model: several perception models cooperating with structured clinical
data, under an explicit account of what each of them can and cannot support.

> **This is not a diagnostic device and has not been validated against clinical outcomes.**
> The benchmarks here are controlled software tests on synthetic cases. They measure whether
> the safety layer behaves as specified; they say nothing about diagnostic accuracy.

---

## Architecture

```
        PATIENT
           │
     ┌─────┴─────┐
     ▼           ▼
  TRIAGE     ULTRASOUND        "how urgent?"        "what does the scan show?"
   AGENT       AGENT
     │           │
     └─────┬─────┘
           ▼
   CLINICAL STATE BUILDER   ◄── vitals · labs · history
           │                     records what is present AND what is absent
           ▼
     SAFETY LAYER           escalation computed here, before any model runs
           │
           ▼
      RAG  →  LLM           retrieval grounds it; the model explains
           │
           ▼
      VALIDATOR             checks the answer against the state
           │
     ┌─────┴─────┐
     ▼           ▼
 DIFFERENTIAL  ESCALATION
```

Two separations organise everything:

- **Perception is separate from reasoning.** The Ultrasound Agent reports findings and never
  assigns urgency — it cannot see the vitals, laboratory results or history that decision
  requires.
- **Rules are separate from the model.** Which tests are absent, what the models cannot
  exclude, whether the agents disagree, whether to escalate — all computed deterministically
  *before* the language model is consulted, and the model's output is checked against the
  same state afterwards.

---

## Layout

```
src/agents/
  schema.py              the contract all three agents write through
  triage/                Triage Agent — urgency from structured data, no imaging
  ultrasound/            Ultrasound Agent — organ routing (see its README)
    agent.py               real per-organ inference, importable and under test
  clinical/              Clinical Reasoning Agent
    clinical_state.py      structured state: findings, labs, vitals, and what is MISSING
    reasoning.py           escalation policy, prompt, and every validator
    llm.py                 HuatuoGPT-o1-8B via llama.cpp, plus test backends
    retrieval.py           TF-IDF retrieval with a relevance floor and citation checking
    decision_support.py    severity, alerts, examinations, scenario routing
    report.py              automated clinical report and archiving
    thresholds.json        every alert cutoff, versioned and auditable
    corpus/                32 knowledge units incl. one sourced protocol
    run_case.py            five benchmark scenarios, runnable end to end
  tests/                 236 tests, grouped by the safety property each exercises

src/data_prep/           per-source manifest builders
notebooks/               training and inference notebooks, one per organ
models/                  metrics, calibration artefacts, and the frozen baseline
_docs/report/latex/      internship report
```

---

## Running it

```bash
python -m src.agents.tests.run_benchmark          # 236 safety tests, no model needed
python -m src.agents.clinical.run_case --dry-run  # state + escalation, no model
python tools/run_regression.py                    # 150 end-to-end checks
run.bat                                           # clinician-facing assistant (Windows)
```

On Windows, `streamlit run app.py` usually fails: Anaconda's `Scripts` directory is not on
PATH, and `python` on PATH often resolves to the Windows Store stub, which runs nothing.
`run.bat` finds an interpreter that actually has streamlit installed and sets
`KMP_DUPLICATE_LIB_OK`, which this environment needs because torch and MKL each link their own
OpenMP runtime. The equivalent by hand is
`C:\path\to\anaconda3\python.exe -m streamlit run app.py`.

All three perception modules run on CPU in the app: the clinician uploads a study and the module
reports its own findings. The lung takes one image and reports four independent findings; the
gallbladder takes one and reports exactly one of five classes; the cardiac module takes **two**
frames, end-diastole and end-systole, because an ejection fraction is a comparison between them
and no single still can produce one.

`notebooks/clinical_reasoning_gpu.ipynb` runs all five scenarios on a Colab T4 in about five
minutes. The same workload on CPU takes roughly 23 minutes **per case**.

---

## What the safety layer enforces

| property | tests |
|---|---:|
| Hallucination rejection | 25 |
| Retrieval grounding | 19 |
| Absent is not normal | 19 |
| Malformed output rejection | 16 |
| Severity and alerts | 12 |
| Escalation policy | 11 |
| Perception contract | 11 |
| Examination recommendations | 10 |
| Benchmark scenarios | 9 |
| Confidence calibration | 9 |
| Conflict detection | 9 |
| Enumerated evidence | 8 |
| Evidence coverage | 8 |
| Automated reporting | 7 |
| Deterministic output control | 7 |
| Case-quality grading | 6 |
| Evidence relationships | 5 |
| Failure severity | 5 |
| LLM failure containment | 5 |
| Model-scope propagation | 5 |
| Reference-range detection | 5 |
| Unassessed-organ reporting | 5 |
| Advice scope | 4 |
| Scenario routing | 3 |
| Value-reading consistency | 3 |

Every check was written against a failure the system actually produced. Among them: a normal
troponin described as elevated, invented pathology for a patient with entirely normal
findings, an unrelated negative cited as evidence against a diagnosis, treatment instructions
from a system that is decision support only, and an unexamined organ omitted from the list of
what is missing.

Several were found by the tests before the model ever met them — `not_detected` had a reader
and no writer, so negative findings vanished; a positive lung finding silenced an unexamined
heart in the escalation policy; substring matching accused a paraphrase of fabrication while
letting any sentence containing the word "high" pass as grounded.

The perception tests found one immediately. `lung_calibration.json` was never exported from the
training notebook, and both the notebook and the extracted agent fell back to a threshold of
0.5 for every finding. This model's outputs sit in roughly 0.2–0.8, so at 0.5 it fired on
almost nothing and every scan came back all-negative — a silent failure that reads downstream
as a screened-and-clear study. The tuned operating points (0.30 / 0.20 / 0.35 / 0.45) were on
disk the whole time in the per-split results, and the agent now reads them. On 120 B-line
positives from LUS-BALD, a dataset this classifier never saw, b-lines score 0.81 on average and
fire on 100% of them; consolidation fires on 0% of anything at a threshold of 0.5.

---

## Results

Per-module results are in `models/`. Two states of the reasoning agent are frozen with hashes
of the prompt and sources: `clinical_reasoning_pre_rag/` (before retrieval) and
`clinical_reasoning_v4_final/` (the final experimental version).

The finding worth reporting is what changed between them. With evidence written as free text
the model fabricated — "stable vitals" for a patient with none, a normal troponin described as
elevated, corpus sentences cited as patient findings — and the safety layer correctly refused
3 of 5 answers. Constraining the interface so that evidence is cited by identifier from an
enumerated list took unsupported-output errors from 6 to 0 and withholding from 3/5 to 0/5.
The two remaining format faults were then moved out of the model entirely: a comma-joined list
is normalised in Python, and every abnormal value reaches the clinician whether or not the
model cited it.

> On this five-case benchmark retrieval did not demonstrate a measurable improvement over the
> no-retrieval condition, and one additional error occurred in the retrieval condition. Its
> contribution to reasoning quality remains inconclusive.

None of this is clinical validation. The benchmark has no ground truth, the cases are
synthetic, and it measures whether the safety mechanisms behave as specified — not diagnostic
accuracy. `models/clinical_reasoning_v4_final/VERSION.json` records what the runs establish
and, separately and at greater length, what they do not.
