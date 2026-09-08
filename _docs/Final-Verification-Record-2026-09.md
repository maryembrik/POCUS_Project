# Final Verification Record — September 2026

What was verified before the implementation freeze, how, and what was found wrong along the
way. Implementation frozen at `f59ac6a`, tagged `implementation-freeze`. Report tooling was
corrected after the freeze; nothing under `src/` or `models/` changed.

---

## Verified

| | how | result |
|---|---|---|
| Safety benchmark | `python -m src.agents.tests.run_benchmark` | 322/322, 36 properties |
| Scenario regression | `python tools/run_regression.py` | 150 checks over 10 cases |
| Lint / code security / dependency security | `python tools/checks.py` | 6/6, no advisories |
| Dependencies install clean | `docker build --no-cache --pull --target base` | exit 0 |
| Runtime imports at pinned versions | `python -c "import torch, fastapi, ..."` in the image | torch 2.13.0+cpu, fastapi 0.141.1, sklearn 1.5.2, cv2 4.13.0, pillow 12.3.0, smp 0.5.0 |
| Application starts | `docker run` + `/api/health` | healthy; lung, heart, gallbladder all loaded |
| Lung declares itself uncalibrated | `/api/health` | *ready, uncalibrated* |
| Container is not root | `docker exec … id` | `uid=1000(pocus)` |
| Both languages served | `curl /` and `/fr` | 200 / 200 |
| LaTeX structure | `check_latex.py` | no problems; 17 citations, 35 glossary entries |
| Floats and labels | ad-hoc pass | 146 labels, 48 floats, no duplicates, no dangling refs |
| Report zip is self-sufficient | extracted to an empty directory, checked in place | 27 files, no problems |

## Not verified, and why

**The compiled PDF.** No LaTeX installation on the development machine — no `pdflatex`,
`xelatex` or `latexmk`. Everything checkable without a compiler has been checked; overfull
boxes, float whitespace and page breaks are visible only in the rendered document and are
inspected in Overleaf.

**The GitHub Actions run.** `gh` is not authenticated in the working environment. The local
cold build runs the identical command; the Actions tab is the record.

---

## Two claims that were wrong before they were right

Recorded because a verification record that lists only the checks that passed is the same
failure this project spent the summer designing against.

### 1. A Docker build that proved nothing

The first `docker build --target base` exited 0 with **every layer CACHED**. Exit 0 there
means the file parses and the target resolves — not that the base image is still pullable,
that the pins still resolve, or that they install on a clean machine, which is what the check
exists to prove. Re-run with `--no-cache --pull`; that is the run in the table above.

### 2. A zip verification that never opened the zip

The zip was extracted to a clean directory and `check_latex.py` run inside it. It reported the
zip complete. It had not read a single file in it:

```python
ROOT = Path(r'C:\Users\HUAWEI\Documents\POCUS-Project\_docs\report\latex')   # was
ROOT = Path(__file__).resolve().parent                                       # now
```

The path was hardcoded, so the checker read the working copy wherever it was run from. A check
that cannot be pointed at a different copy cannot verify a copy.

**How it surfaced:** not by review. A `\chapter` guard was added to the checker and tested by
injecting a `\chapter` into a copy — and it did not fire. The fault injection found the
hardcoded path; the path explained the false zip result. Both were then re-run correctly.

### The defect that prompted the guard

`appendices.tex` used `\chapter{}` in an `article`-class document, where `\chapter` is
undefined. An undefined control sequence stops the compile rather than degrading, so the
document would have failed at that line — while every existing check passed, because the
braces balance, the label resolves and nothing is missing. Fixed to `\subsection` with
`\subsubsection` beneath, matching the file's own `\Alph{subsection}` scheme.

---

## Worth saying out loud at the defense

Asked *"how do you know your checks actually check anything?"*, the answer is that one of them
did not, and it was caught by injecting the fault it was supposed to catch and watching it stay
silent. Every property in the safety benchmark is asserted the same way — by a test that has
been seen to fail.
