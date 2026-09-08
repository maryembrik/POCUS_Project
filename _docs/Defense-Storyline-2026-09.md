# Oral Defense — Storyline (September 2026)

Supersedes the July talking points, which describe an earlier plan (gallbladder unbuilt, cardiac
trained from scratch, pneumothorax via optical flow). Where the two disagree, this document is
current. Every number below is traceable to `models/registry.json`,
`models/safety_benchmark.json`, or a named section of the report.

---

## 1. The opening claim

Say this, and do not overstate it:

> POCUS-Emergency is a multimodal clinical **decision-support** prototype for emergency
> point-of-care ultrasound. It combines patient information and ultrasound findings, states
> explicitly what each model knows and does not assess, and leaves the physician as the
> decision-maker.

The word that carries the defense is **decision-support**. Every difficult question below is
answerable because the system was built to that claim rather than to "the AI diagnoses."

**The sentence to have ready if pressed on contribution:** the engineering contribution is not
a model, it is a *representation* — a clinical state in which absent evidence, weak evidence,
and screened-negative evidence are different things, and in which the system cannot cite
evidence it does not have.

---

## 2. The spine: CRISP-DM

The report's chapters *are* the phase names. Walk the jury down them in order; do not
improvise a different structure on the day.

| Phase | What to say in one sentence |
|---|---|
| Business Understanding | Emergency POCUS is operator-dependent and time-critical; the target is support at the bedside, not autonomy. |
| Data Understanding | Four sources, chosen against explicit criteria (case identity recoverable, licence permits derived work, acquisition resembles point-of-care). |
| Data Preparation | Manifest-driven, split by **case**, never by frame — and the disposition fields were *removed* because they leak the outcome into the input. |
| Modeling | Two parallel branches — triage on tabular intake, ultrasound on three organs — plus a clinical reasoning layer above them. |
| Evaluation | Reported per finding, not as one headline number, with the rejections included. |
| Deployment | Technical deployment performed and demonstrated; clinical deployment explicitly not claimed. |

**If asked why CRISP-DM:** because the two phases most projects skip — Data Understanding and
Evaluation — are the two that produced this project's most important results (the case-vs-frame
correction, and the rejected experiments).

---

## 3. The architecture in one breath

Two branches feed one reasoning layer:

- **Triage Agent** → patient and clinical information (vitals, age, arrival mode)
- **Ultrasound Agent** → lung, cardiac, gallbladder
- **Clinical Reasoning** → receives both, combines their evidence, retrieves medical knowledge,
  produces decision support

The detail worth stating unprompted: **escalation is computed before the language model runs**,
from the structured clinical state. If the model fails, is slow, or is absent — as it is in this
deployment, which has no GPU — the alerts are still correct. The LLM is the explanation layer,
never the safety layer.

---

## 4. The five states — the single most defensible idea

Learn these five. Most of the hard questions reduce to them.

| State | Meaning |
|---|---|
| `detected` | positive finding |
| `unreliable` | detected, but the evidence is weak |
| `screened_negative` | independently looked for and not found |
| `not_assessed` | no model covers this — **absence of a model is not evidence of absence** |
| `alternative` | considered and rejected, for mutually exclusive classes |

Two consequences to state:

1. **Evidence is enumerated** (E1…En). The reasoning layer can only cite items that exist, so a
   fabricated citation is not filtered out after the fact — it is unrepresentable.
2. **Gallbladder `alternatives` are deliberately excluded from the clinical state.** A
   considered-and-rejected class is not evidence, so the model is not given the chance to cite
   one as though it were.

---

## 5. Numbers to know cold

| | value | where |
|---|---|---|
| Safety benchmark | **322/322** across 36 safety properties | `models/safety_benchmark.json` |
| Scenario regression | 150 checks over 10 cases | `tools/run_regression.py` |
| Lung, macro AUROC | 0.727 (production v1) | registry, `lung` |
| Lung, pleural effusion positives | 24 cases total, 16 in training, **≈3** in a calibration split | Tab. `lung-calibration-budget` |
| Cardiac LV Dice | 0.928 in-domain → **0.534** external | §cardiac-module |
| Cardiac EF MAE | 12.2 pp → **6.9 pp** after volume correction | Tab. `cardiac-rejected` |
| Triage reference | accuracy 0.680, recall(high) 0.784 | registry, `triage` |
| Triage deployed | accuracy 0.630, macro F1 0.567, recall(high) 0.773, **dangerous miss 1.36%** | registry, `triage_deployed` |

The one comparison to have at your fingertips: the deployed triage model is **less accurate**
than the reference (0.630 vs 0.680) and has a **lower dangerous-miss rate** (1.36% vs 1.55%).
That is the whole argument for why it is the one at the bedside.

---

## 6. The hard questions

Each answer is one or two sentences. Do not elaborate unless asked.

**Why didn't you calibrate the lung module?**
Insufficient independent positives. A held-out calibration split leaves about three positive
effusion cases; a Platt scaling fitted on three points is a curve through noise. The module is
left uncalibrated and says so everywhere the number appears — including `/api/health`, which
reports it as *ready, uncalibrated*.

**Why are there two triage models?**
Different feature spaces. The reference model is fitted on everything the source data contains;
the deployed one only on the 21 features a bedside intake form can actually supply. Training
features equal interface fields equal deployment inputs, so its accuracy is the accuracy a
deployed prediction can reach.

**Why did the MLOps gate reject the deployed model?**
Because it correctly detected a regression against the reference lineage — the deployed model
is less accurate. That is a true finding, so the gate was not weakened; the model was given its
own lineage instead, where it is judged against models that share its constraints.

**Why doesn't the system report pneumothorax as negative?**
Because no model was trained for it. Absence of a model is not evidence of absence. It is
reported as `not_assessed`, with a dash where a confidence would otherwise be.

**Why can the reasoning layer not cite gallbladder alternatives?**
A considered-but-rejected class is not evidence. Including it would let the system support a
conclusion with something it had already ruled out.

**Why does the doctor confirm the triage tier?**
The system provides decision support; the physician retains clinical responsibility. The model
suggests, the clinician decides, and a disagreement is *recorded* — the system states that the
two assessments conflict and that it cannot determine which is correct, rather than choosing.

**Why is deployment in the report with no hospital validation?**
Technical deployment and clinical deployment are different claims. The first is demonstrated;
the second is not attempted, and the report says which claims that removes.

**How do you know the cardiac EF number near a decision boundary?**
The sensitivity zone is derived from the module's own measured error — 6.9 pp MAE — not from an
invented ±5%. Near a cutoff the system says the value could fall on either side.

---

## 7. Volunteer the limitations before you are asked

This is the part that makes the work look engineered rather than assembled. Each limitation is
evidence-based, and each has a number or a reason attached:

- Lung calibration — insufficient independent positives (≈3)
- Pneumothorax — no trained model
- Cardiac EF — engineering sensitivity zone around boundaries, from measured MAE
- Gallbladder — mutually exclusive alternatives, kept out of the evidence
- Triage — separate reference and deployed models, with the gate's rejection intact
- Hospital deployment — not performed
- Prospective clinical validation — not performed
- Adaptive questioning — intentionally not implemented

**The rejected experiments are an asset, not a weakness.** Seven well-motivated changes were
tested and rejected on measurement rather than preference, and each is reported with its
outcome. A method that records only the hypotheses that survived gives a reader no way to
judge it.

---

## 8. What not to claim

- Not "the AI diagnoses" — it supports a decision.
- Not "validated" — it is verified, on retrospective data.
- Not that the lung confidence is calibrated.
- Not that the XGBoost reference model is what runs at the bedside.
- Not that the system was used on patients.

---

## 9. Honest caveat worth stating directly

The ceiling on the imaging modules is dataset size, not tuning. 165 lung case groups and 24
positive effusion cases will not support a better estimate no matter how the model is
configured. The next real lever is more data, and the report says so rather than presenting
further hyperparameter search as progress.
