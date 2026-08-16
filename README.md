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
  clinical/              Clinical Reasoning Agent
    clinical_state.py      structured state: findings, labs, vitals, and what is MISSING
    reasoning.py           escalation policy, prompt, and every validator
    llm.py                 HuatuoGPT-o1-8B via llama.cpp, plus test backends
    retrieval.py           TF-IDF retrieval with a relevance floor and citation checking
    corpus/                30 knowledge units, clinical knowledge only
    run_case.py            five benchmark scenarios, runnable end to end
  tests/                 156 tests, grouped by the safety property each exercises

src/data_prep/           per-source manifest builders
notebooks/               training and inference notebooks, one per organ
models/                  metrics, calibration artefacts, and the frozen baseline
_docs/report/latex/      internship report
```

---

## Running it

```bash
python -m src.agents.tests.run_benchmark          # 156 safety tests, no model needed
python -m src.agents.clinical.run_case --dry-run  # state + escalation, no model
python -m src.agents.clinical.run_case --scenario conflict --n-gpu-layers -1
```

`notebooks/clinical_reasoning_gpu.ipynb` runs all five scenarios on a Colab T4 in about five
minutes. The same workload on CPU takes roughly 23 minutes **per case**.

---

## What the safety layer enforces

| property | tests |
|---|---:|
| Absent is not normal | 16 |
| Hallucination rejection | 21 |
| Retrieval grounding | 17 |
| Malformed output rejection | 15 |
| Escalation policy | 11 |
| Conflict detection | 9 |
| Confidence calibration | 9 |
| Evidence coverage | 8 |
| Benchmark scenarios | 9 |
| Case-quality grading | 6 |
| Model-scope propagation | 5 |
| Reference-range detection | 5 |
| Evidence relationships | 5 |
| Unassessed-organ reporting | 5 |
| LLM failure containment | 5 |
| Advice scope | 4 |
| Value-reading consistency | 3 |
| Failure severity | 3 |

Every check was written against a failure the system actually produced. Among them: a normal
troponin described as elevated, invented pathology for a patient with entirely normal
findings, an unrelated negative cited as evidence against a diagnosis, treatment instructions
from a system that is decision support only, and an unexamined organ omitted from the list of
what is missing.

Several were found by the tests before the model ever met them — `not_detected` had a reader
and no writer, so negative findings vanished; a positive lung finding silenced an unexamined
heart in the escalation policy; substring matching accused a paraphrase of fabrication while
letting any sentence containing the word "high" pass as grounded.

---

## Results

Per-module results are in `models/`. The pre-retrieval baseline for the reasoning agent is
frozen in `models/clinical_reasoning_pre_rag/`, with hashes of the prompt and sources so that
a later comparison against retrieval is against that exact state rather than a re-tuned one.

On five synthetic cases the safety layer withheld output on every case containing a grounding
or reading error, and answered directly on the one case where the evidence agreed and the
record was complete. The model's clinical reasoning remained limited — see `baseline.md` for
what that run establishes and, more importantly, what it does not.
