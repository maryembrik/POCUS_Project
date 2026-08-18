# Actionable complaints and a second revision round — no measurable benefit

Third run of the same five scenarios. Two changes from
[v2](findings.md): every remaining complaint was rewritten to carry its own correction, and
`max_revisions` went 1 → 2.

The hypothesis was that the three surviving faults persisted because the complaints stated a
rule without saying what to write instead. Each now names the exact fix:

- `abnormal value(s) cited nowhere: E5 (hr: 122.0 bpm -- HIGH) -- add each identifier to
  'supporting' or 'contradicting'`
- `one test per array element. Write it as ["bnp", "d_dimer", "crp", "wbc", "creatinine"]`
- `write 'obtain X as additional information relevant to assessing Y' rather than 'obtain X to
  rule out Y'`

## Result

| | v2 (one round, rule-only complaints) | v3 (two rounds, corrections given) |
|---|---|---|
| errors, no retrieval | 0 | **0** |
| errors, with retrieval | 1 | **1** |
| warnings, no retrieval | 7 | **6** |
| warnings, with retrieval | 7 | **7** |
| revision rounds used | 4 | **7** |
| runtime, no retrieval | 290 s | **383 s** |
| runtime, with retrieval | 312 s | **400 s** |

Withheld 0/5 without retrieval and 1/5 with, unchanged. 173/173 safety tests. Escalation
identical across arms. Two identical calls give identical output.

**The extra round is reached and used** — `missing`, `concordant` and `not_assessed` all ran
two revisions in both arms — and the answer comes back with the same faults it went in with.
The model was told, in the second round, exactly which identifier to add, and did not add it.

## What this establishes

Rewriting the complaints did not work. That is the finding, and it is worth reporting as one:
with a quantised 8B model at temperature 0, making a validator's complaint mechanically
explicit did not change compliance on three of five cases. The failure is not that the model
misunderstood the instruction.

The cost is 32% more runtime for three extra model calls per arm.

## A flaw in this experiment

Complaint wording and `max_revisions` were changed in the same run. The one improvement --
`conflict` went from one warning to zero -- cannot be attributed to either. Separating them
would need two more runs, and on a five-case benchmark the difference is one warning, which is
not worth the GPU time.

## Consequence

`max_revisions` returns to **1** as the default. The parameter still accepts 2 and the suite
proves two independent faults clear in exactly two rounds
(`test_two_independent_faults_need_two_revision_rounds`), so the capability is real and
tested. It is simply not worth its cost with this model, which is what the measurement says
rather than what the architecture diagram would prefer.

The rewritten complaints stay. They cost nothing, they are strictly more informative to a
human reading the output, and one of them fixed a real defect found while rewriting:
`check_evidence_coverage` walked `state["vitals"]` only, so an abnormal **lab** left uncited
was never flagged at all.

One imperfection recorded rather than tidied: where the model wrote
`["heart: bnp, d_dimer, ...", "vitals: temp"]`, the suggested correction splits mechanically on
commas and proposes `"heart: bnp"` as an element. Faithful to the input, but not a test name.

## Where this leaves the agent

Errors 0, withheld 0/5 without retrieval. The three remaining faults are *warnings* on
delivered answers: an abnormal value not cited, a comma-joined list, and an overclaiming next
step. None of them corrupts the reasoning; all are reported to the reader.

Two attempts have now been made to remove them -- constraining the interface, which worked for
everything it targeted, and sharpening the complaints, which did not. The next attempt would be
prompt engineering against a specific model, which was explicitly ruled out of scope.

**Freeze here.**
