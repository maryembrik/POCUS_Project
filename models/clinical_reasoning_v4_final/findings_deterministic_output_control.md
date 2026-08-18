# Deterministic output control — the final measurement

Fourth and last run of the same five scenarios. The two format faults were moved out of the
model: `missing_information` is normalised in Python, and every abnormal value now reaches the
reader through `evidence_considered` regardless of what the model cited.

## The four runs

| run | arm | errors | warnings | withheld | runtime |
|---|---|---:|---:|---:|---:|
| v1 free text | none | 6 | — | 3/5 | 290 s |
| v1 free text | tfidf | 5 | — | 4/5 | 312 s |
| v2 enumerated evidence | none | **0** | 7 | **0/5** | 290 s |
| v2 enumerated evidence | tfidf | 1 | 7 | 1/5 | 312 s |
| v3 sharper complaints, 2 rounds | none | 0 | 6 | 0/5 | 383 s |
| v3 sharper complaints, 2 rounds | tfidf | 1 | 7 | 1/5 | 400 s |
| **v4 deterministic control** | none | **0** | **5** | **0/5** | **282 s** |
| **v4 deterministic control** | tfidf | **1** | **6** | **1/5** | **330 s** |

Escalation identical across arms on all five. Two identical calls give identical output.
180/180 safety tests.

## Against the stop criteria

| # | criterion | result |
|---|---|---|
| 1 | fabricated evidence → 0 | **met** |
| 2 | unknown evidence identifiers → 0 | **met** |
| 3 | reference-range mistakes → 0 | **met** |
| 4 | abnormal values ignored → 0/5 | **met for the reader, not for the model.** See below |
| 5 | malformed `missing_information` → 0 | **met.** Zero warnings; normalised in Python on 3 occasions and each one recorded |
| 6 | forbidden "rule out" language → 0 | **not met.** Present on 2 of 5 cases without retrieval, 3 of 5 with |
| 7 | useful-output rate better than 20% | **met.** 100% without retrieval, 80% with |
| 8 | no regression in the safety tests | **met.** 158 → 180 |

Seven of eight, with one qualified.

## Criterion 4, stated precisely

The model still fails to cite every abnormal value on three of five cases. That did not
change, and two attempts to change it failed — a prompt rule, then a complaint naming the
exact identifier to add, across two revision rounds.

What changed is who is responsible. `evidence_considered` is built from the state, so the
clinician sees every abnormal value with a mark showing whether the model reasoned from it:

```
CLINICAL EVIDENCE CONSIDERED
  * [E1] lung: b lines DETECTED (0.92, strong)
    [E5] hr: 122.0 bpm -- HIGH
    [E6] rr: 28.0 /min -- HIGH
    [E8] spo2: 88.0 % -- LOW
    ...
```

On `concordant` the model cited 4 of 7 abnormal values. All 7 reach the reader.

So the honest claim is **not** "the model now accounts for every abnormal value". It is that
an abnormal value can no longer be lost between the state and the clinician, and that
`check_evidence_coverage` still reports when the model's own reasoning is incomplete.

## Criterion 6, why it stays

`recommended_next_step` is prose, and the wording is the claim. Splitting a list changes
nothing; rewording a recommendation changes what the system asserts, so Python does not do it
and the fault is reported instead. The remaining options are prompt engineering against this
specific model, which failed in v3 and was ruled out of scope, or templating the field, which
would remove the clinical rationale that makes it useful.

Reported as a limitation.

## A defect this run exposed

Normalisation split `"labs: bnp, d_dimer, crp"` into `["labs: bnp", "d_dimer", ...]` — the
prefix labels the group, not the first test, so `"labs: bnp"` named a test that does not
exist. Fixed by stripping a known group prefix from the first element, with a companion test
that a real name containing a colon (`"CT: chest with contrast"`) is not eaten. This is
exactly the risk flagged when the splitter was written; it materialised on the first real run.

## Retrieval, final position

Unchanged across all four runs: **on this five-case benchmark, retrieval did not demonstrate a
measurable benefit over the no-retrieval condition, and one additional error occurred in the
retrieval condition** (`not_assessed`, the unexamined heart omitted from
`missing_information`).

That is the whole claim. It does not support "RAG makes the system worse", and five synthetic
cases could not support it if it did.

## Verdict

Freeze. Errors 0, withheld 0/5 without retrieval, 180/180 tests, and every fault that a
deterministic layer could own has been moved into one. What remains is one phrasing preference
on a delivered answer, and a model that reasons from some of the evidence rather than all of
it — both reported to the reader rather than hidden from them.
