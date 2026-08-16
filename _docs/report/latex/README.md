# POCUS-Emergency — internship report (LaTeX)

Self-contained. No custom class file is required: standard `article` plus packages that
ship with TeX Live and Overleaf.

```
main.tex             title page, front matter, includes every chapter
preamble.tex         packages, page style, the `keystatement` environment, \rmq
glossary.tex         acronyms (front matter) and technical terms (back matter)
bibfile.bib          references
appendices.tex       appendices A–F
chapters/            one file per chapter, 01–09
figures/             generated figures
logos/               institutional logos
check_latex.py       static checks, no TeX installation needed
```

## Compiling

Overleaf: set the compiler to **pdfLaTeX** and the bibliography tool to **Biber**
(Menu → Settings). Locally:

```bash
latexmk -pdf main.tex
```

Without `latexmk`: `pdflatex` → `biber` → `pdflatex` → `pdflatex`. The glossary needs no
extra tool — `\printnoidxglossary` resolves during the normal passes, which is why it was
chosen over `makeglossaries`.

## Structure

| Chapter | Status |
|---|---|
| 1. Introduction | Written |
| 2. Project Environment and Presentation | Written, except §2.1 and parts of §2.2 |
| 3. Preliminary Study and State of the Art | Skeleton |
| 4. Datasets and Data Preparation | Skeleton |
| 5. Perception Agents | Skeleton |
| 6. Clinical Reasoning Agent | Skeleton |
| 7. System Integration and Prototype | Skeleton |
| 8. Results and Discussion | Skeleton |
| 9. Conclusion and Perspectives | Skeleton |

Skeleton chapters carry their full section structure plus a `\rmq{...}` note per section
describing what belongs there. They are already included in `main.tex`, so the table of
contents shows the complete plan from the first compile.

## Acronyms and glossary

`\newacronym` gives the correct academic behaviour automatically: the **first** `\gls{key}`
in the text renders *Point-of-Care Ultrasound (POCUS)* and every later one renders *POCUS*.
Order of first use therefore matters — check it after reordering any text.

Acronyms print in the front matter as the List of Acronyms; technical terms print at the
back as the Glossary.

## Before submitting

Everything still to be filled in is marked `\rmq{...}` and **renders in red**, so no
unresolved item can survive a read-through of the compiled PDF. Search for `\rmq` to list
them.

Currently outstanding beyond the skeleton chapters:

- title page: internship dates, academic year, both supervisors, host organisation
- `logos/esprit.png` — the school logo. `main.tex` guards it with `\IfFileExists`, so the
  project compiles without it and prints a red placeholder. `logos/host.png` is optional
  and prints nothing if absent
- acknowledgments
- §2.1 Host Organization, and the supervision arrangement in §2.2
- `bibfile.bib` — verify every entry against the publisher record. Volume, issue, page and
  arXiv fields are the ones most easily got wrong; the HuatuoGPT-o1 identifier is
  explicitly unverified and flagged in red in the bibliography itself
- §3.4.3 — the "Auto-US" system could not be matched to a confirmed publication. Find the
  exact reference or drop the subsection

## Managing compile time

Overleaf's free plan caps the time of a **single compile run**. It is not a monthly quota —
you can compile as often as you like; only one long build fails. A full build runs
pdfLaTeX → Biber → pdfLaTeX → pdfLaTeX, so everything is paid for three times, and the
report will get slower as chapters and figures are added.

Four levers, cheapest first:

1. **Fast [draft] mode.** The dropdown beside the green Recompile button. Reuses the
   auxiliary files and runs a single pass. Use it for ordinary edits; press it two or three
   times when cross-references need to settle. This is the day-to-day setting.

2. **`\includeonly`.** In `main.tex`, uncomment the `\includeonly{...}` line and list only
   the chapter being written. The rest keep their page numbers and cross-references from
   the last full build, so `\ref` and the table of contents stay correct. Chapters are
   pulled in with `\include` rather than `\input` specifically to make this work. This is
   the main tool once the report is long.

3. **`\setkeys{Gin}{draft}`.** Uncomment in `main.tex` to replace every figure with an empty
   frame of the same size. Layout is preserved; image processing is skipped. Useful when
   editing text in a chapter that carries many figures.

4. **Keep figures modest.** Export plots at 150–200 dpi, not 600. A 30-figure report at
   600 dpi will time out on any plan setting.

If a **full** build eventually exceeds the limit — likely near submission, with every
chapter and all figures present — the reliable answer is to compile locally. Install
MiKTeX (free, Windows), then:

```bash
latexmk -pdf main.tex
```

No time limit, and it is usually faster than Overleaf. Keep Overleaf for writing and use a
local build for the final PDF.

## Checking without compiling

```bash
python check_latex.py
```

Catches undefined citation, glossary and reference keys, unbalanced braces, missing
`\input` targets, missing graphics, and `\glspl` on a term whose plural would be wrong.
Static only — not a substitute for a real compile.

## Figures

`figures/fig_architecture.png` is generated by `scratchpad/make_arch_diagram.py`
(matplotlib). Edit and re-run the script rather than the PNG.
