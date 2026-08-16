# Retrieval A/B — what adding RAG changed

Five scenarios, run twice: `retrieved=None` (reproducing the frozen pre-RAG baseline) and
TF-IDF retrieval over the 30-unit corpus at version `1.0-all-units-sourced`. Colab T4,
HuatuoGPT-o1-8B Q4_K_M, temperature 0, fixed seed, `max_revisions=1`.

## Validity

| check | result |
|---|---|
| no-retrieval arm reproduces the frozen baseline | 5/5 `differential` and `escalation` MATCH |
| escalation identical across arms | 5/5 OK (it is computed before the model runs) |
| two identical calls give identical output | yes |

Differences below are therefore attributable to retrieval.

## Result

| scenario | hits | errors | warnings | withheld |
|---|---|---|---|---|
| missing | L09, B26 | 0 → **1** | 1 → 0 | **F → T** |
| conflict | C12 | 1 → 1 | 1 → 0 | T → T |
| concordant | — | 0 → 0 | 3 → 3 | F → F |
| reassuring | — | 1 → 1 | 0 → 0 | T → T |
| not_assessed | L01, L05, L06 | 4 → **2** | 2 → 1 | T → T |

Totals: 6 errors → 5. Retrieval fired on 3 of 5 cases.

The aggregate is close to meaningless. Retrieval did not reduce a single class of error
across the board; it removed some faults and created others.

## What retrieval fixed

- **`conflict`: the troponin misread is gone.** The model had called a troponin of 5.0
  "elevated" when the state printed it as normal. This was predicted *not* to change,
  on the reasoning that the reference range was already in the prompt and was ignored, so
  supplying more text should not help. That prediction was wrong.
- **`not_assessed`: the unassessed heart is now named in `missing_information`**, and one
  of three unrelated-negative-as-contradicting errors is gone.

## What retrieval broke — a new failure mode

In 2 of the 3 cases where retrieval fired, the model took a sentence from the retrieved
passage and placed it in `contradicting` as though it were an observation about this
patient:

- `missing` cited `lung sliding (low specificity)` — from **L09**, whose text reads
  "lung sliding when used alone has a low specificity".
- `conflict` cited `absence of sonographic signs of RV overload or dysfunction` — verbatim
  from **C12** (Ha & Toh).

Neither is a finding about these patients. The model attributed the corpus's general
statements to the specific case.

This is the risk `retrieval.py` names in the abstract — "handing a small model plausible
text that does not bear on the case creates a new surface for ungrounded claims" — now
measured rather than asserted. **Giving a small model reference material creates a new way
to fabricate.**

Every instance was caught by `check_citations` as `possible fabrication` and the
differential was withheld. Retrieval opened a hole; the validators closed it. That is the
argument for computing the safety layer independently of the model, and it is stronger
evidence for the architecture than a clean improvement would have been.

## Limits on all of the above

- **Two of five cases carry no information.** `concordant` and `reassuring` retrieved
  nothing, so their arms are identical by construction. `reassuring` is the all-normal
  patient: a normal patient produces a thin query, and `build_query` deliberately excludes
  absent tests. That is a coverage gap in the query builder, not a null result.
- **`not_assessed`'s improvement is partly an artefact.** The model changed its diagnosis
  from Pulmonary Embolism to Diffuse Interstitial Syndrome. It is not the same answer with
  fewer faults; it is a different answer that happens to have fewer.
- **F20 does not retrieve for its own question.** Recorded before this run, not after:
  TF-IDF ranks F18 above it because F18 repeats "FAST negative" in a short passage while
  F20 is long and says "false-negative" and "sensitivity". The corpus's most
  safety-relevant passage is currently the hardest to reach.
- **Five synthetic cases, no clinical ground truth.** This measures whether the validators'
  verdicts changed. It says nothing about diagnostic accuracy.

## What this suggests next

1. A validator specifically for retrieved-text-cited-as-finding. `check_citations` catches
   it today as generic fabrication, which is correct but uninformative — the message says
   "does not appear in the clinical state" when the more useful message is "this sentence
   is from passage C12, not from this patient".
2. Fix `build_query` coverage so an all-normal patient still retrieves.
3. Re-run with `query_hints` indexing to make F20 reachable, as a separate, documented
   experiment — measuring a retriever tuned after seeing the problem is a weaker claim and
   should not be folded into this one.
