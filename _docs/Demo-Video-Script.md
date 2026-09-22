# POCUS-Emergency — Demo Video Script

A walkthrough of the running platform, roughly **6–7 minutes**. Each section gives what to
click, what to say, and what to point at on screen.

Two rules that make the difference between a demo and a sales pitch:

- **Never say the system "found" or "diagnosed" anything.** It *detected*, it *reports*, it
  *suggests*. The physician decides.
- **When the screen says it does not know something, stop and read it out.** That is the
  feature. Skipping past it is skipping the point of the project.

**Before you record:** sign out, so the video starts at the login screen. Have one ultrasound
image on the desktop ready to drag in. Run at 1440×900 or wider — the assessment screen has
three columns and they collapse on a narrow window.

---

## 1 · Sign in  (0:00 – 0:30)

**Do:** land on the login page, type the username and password, sign in.

> "POCUS-Emergency is a clinical decision-support prototype for emergency point-of-care
> ultrasound. Every physician has their own account — the patients you see are yours, and
> nobody else's."

**Point at:** the tagline under the form — *Clinical decision support. The clinician decides.*

> "That line is on the login screen on purpose. It is the claim the whole system is built to."

---

## 2 · The home screen  (0:30 – 1:15)

**Do:** let the home screen load. Don't click anything yet.

> "This is the clinician's workspace. Along the left: patient workup, the patient record,
> the assessment, alerts, the report."

**Point at:** the four counters.

> "Patients this session, studies read, high-priority cases — and the number of automated
> safety tests currently passing. That last one is on the clinician's screen deliberately:
> the system reports its own verification state rather than asking to be trusted."

**Point at:** bottom-left — your name and the three module dots.

> "Signed in as myself, and the three perception modules reporting that they loaded."

---

## 3 · A new assessment  (1:15 – 2:30)

**Do:** click **New patient assessment**. Fill in the intake form as you talk — name, age,
sex, complaint. Enter a few vitals and **deliberately leave several blank.**

> "I enter what I actually have. Notice I am leaving most of the laboratory values empty —
> that matters in a moment."

**Do:** set the urgency tier.

> "I grade the urgency myself. The system will offer a suggestion, but the tier that counts
> is the one the clinician sets."

**Point at:** the triage suggestion appearing as you type.

> "It updates as I enter observations, and it tells me how many of the seven it has —
> 'based on 2 of 7 observations'. Not a percentage. A count I can act on by taking the third."

---

## 4 · Reading a POCUS study  (2:30 – 3:30)

**Do:** upload the ultrasound image. Wait for it to be read.

> "The study is read by the module for that organ — here the gallbladder — and filed against
> this patient as it is acquired."

**Point at:** the returned finding and its confidence.

> "One finding, with a confidence. The classes it considered and ruled out are not passed on
> as evidence — a rejected class is not a finding."

**Do:** click **Analyse**.

---

## 5 · The assessment — the important screen  (3:30 – 5:00)

This is the centre of the demo. Slow down here.

**Point at:** the POCUS block and read the four states aloud.

> "Every finding sits in one of four states. **Detected.** **Detected with weak evidence.**
> **Screened and not detected** — the model looked and did not see it. And **not assessed** —
> no model covers this at all."

**Point at:** the pneumothorax row with the dash where a confidence would be.

> "Pneumothorax. No model was trained for it, so the system says *not assessed*. It does not
> say negative, and it never will. Absence of a model is not evidence of absence — and that
> distinction is enforced by the data structure, not by a convention someone has to remember."

**Point at:** the laboratory column.

> "Every value I left blank reads *not measured* rather than showing an empty cell. A blank
> can be misread as normal. 'Not measured' cannot."

**Point at:** the right-hand panel — what would clarify the case.

> "And it names what would help, with the reason each item matters — not a list of test names,
> but why this particular case needs them."

**Point at:** the differential panel.

> "This deployment has no GPU and does not load the language model, and the panel says so.
> It reports that no differential was generated and that everything else on screen was still
> computed. The mock-up this replaced carried a confident diagnosis with a percentage beside
> it."

---

## 6 · Alerts and disagreement  (5:00 – 5:45)

**Do:** open **Alerts**.

> "The alerts are computed from the structured state before any language model runs. If the
> model fails, is slow, or is absent — as it is here — the alerts are still correct."

**Point at:** the human–AI disagreement alert, if the case produced one.

> "When my assessment and a module's disagree, the system says so — and says it cannot
> determine which of us is right. It does not quietly pick one. That is a prompt to look
> again, not a verdict."

---

## 7 · The report  (5:45 – 6:30)

**Do:** open **Report**.

> "The report is assembled from the computed state, not written by a language model. Every
> sentence traces to a finding the pipeline produced."

**Do:** click **Save to record**.

**Point at:** the confirmation with the fingerprint.

> "Filed to the patient's record, with a content fingerprint. Save the same unchanged
> encounter again and you get the same fingerprint — so a genuine change is distinguishable
> from a re-run."

**Do:** click **Print** and show the print preview, then cancel.

> "And it prints as a clinical document — just the report, not the application around it."

---

## 8 · The patient record  (6:30 – 7:00)

**Do:** open **Patient record**.

> "Each physician's own patients. Open one and you get their history — every examination,
> with the images and the measurements as they were at the time."

**Do:** open the patient, show the examination list.

> "This survives a restart. It is a stored record, not a session."

**Do:** click **Français**.

> "And the whole interface runs in French, composed from the same computed fields as the
> English — so a French screen cannot state something the English record does not."

---

## 9 · Closing  (7:00 – 7:30)

Say this to camera, not over a screen:

> "What it can do: assemble the available evidence, grade urgency, read three organ systems,
> and state plainly what is missing.
>
> What it cannot: exclude a pneumothorax, confirm a normal gallbladder, or replace an
> echocardiographic measurement.
>
> And what it never does: make the clinical decision. That remains the physician's."

---

## If something goes wrong on camera

| | |
|---|---|
| A module shows "weights absent" | Say it out loud — it is the health check doing its job. Carry on; the others still work. |
| The assistant answers "no differential was generated" | That is correct behaviour on a machine with no GPU, not a failure. Read it as designed. |
| An upload is slow | The image is being read by a real model, not looked up. Say so rather than filling silence. |

**Do not** re-record to hide a *not assessed* or a *not measured*. Those are the demo.
