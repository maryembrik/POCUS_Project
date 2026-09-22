# Demo voiceover — recording of 22 Sept, 4 min 45 s

Written for a **sped-up** video: section-level narration, not click-by-click. Roughly 380
words, about 2½ minutes spoken — so it sits comfortably over a shortened cut.

Timestamps are from the original recording. If you speed the video 1.5×, divide them by 1.5.

---

### 0:00 — Creating an account

> "POCUS-Emergency is a clinical decision-support prototype for emergency point-of-care
> ultrasound. Each physician has their own account."

*(when the red line appears)*

> "The username is already taken, and it says so. It will not quietly overwrite an existing
> account."

---

### 0:30 — The intake

> "I enter what I actually have: age, sex, presenting complaint, vital signs, laboratory
> values."

*(on the amber note)*

> "And it counts what I have left blank. Ten values not entered — each recorded as **not
> measured**, never as normal. An empty cell can be read as a normal result. 'Not measured'
> cannot."

---

### 1:20 — The disagreement  ← **the most important moment**

> "I grade this patient **Low**.
>
> The system suggests **High** — based on five of seven observations, and it names the two it
> does not have.
>
> It does not overrule me, and it does not go quiet. It records that we disagree, and carries
> that disagreement into the assessment."

---

### 2:10 — Reading the scan

> "The lung study is read by the model for that organ."

*(on the four findings)*

> "Before it has read, all four findings say **not read yet** — not 'negative'. The system
> does not report on a scan it has not seen."

---

### 2:45 — The assessment

> "Everything on this screen is computed by the pipeline. Where it has nothing real to put in
> a slot, the slot says so."

---

### 3:20 — Alerts

> "Twelve alerts. Hypotension. Severe hypoxaemia. Markedly raised troponin — each naming the
> value and the threshold it crossed.
>
> These are computed before any language model runs. If the model fails or is absent, the
> alerts are still correct."

---

### 4:05 — The patient record

> "The patient's record: two encounters, one POCUS study, twelve alerts, graded high.
>
> The scan is stored with the reading the model made of it — pleural effusion, 0.90 — and it
> survives a restart. This is a stored record, not a session."

---

### Closing — say to camera

> "What it can do: assemble the evidence, grade urgency, read the scan, and state what is
> missing.
>
> What it cannot: exclude a pneumothorax, or replace a formal echocardiogram.
>
> What it never does: make the clinical decision. That stays with the physician."

---

## If you cut for time

Keep, in this order of importance:

1. **1:20 — the disagreement.** Low against High. Nothing else in the demo shows the idea as
   clearly.
2. **0:55 — the blank-value count.** Absent is not normal, stated by the product itself.
3. **2:10 — "not read yet".**
4. **4:05 — the stored record.**

Cut first: the account creation, and the slow parts of typing the labs.
