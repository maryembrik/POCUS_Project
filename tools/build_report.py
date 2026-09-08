"""Compile the report and report what the compile complained about.

    python tools/build_report.py

Runs the full chain -- pdflatex, biber, pdflatex, pdflatex -- because the cross-references,
the table of contents and the bibliography each need a pass that can see the previous one's
output. One pass produces a PDF with "??" everywhere and no bibliography.

The point of this script is not the PDF; latexmk would do that. It is the SUMMARY: a LaTeX
run prints thousands of lines and buries the handful that matter, so the ones worth a person's
attention are pulled out and grouped -- undefined references, missing citations, and the
overfull boxes that are the visible symptom of text running into the margin.

Output goes to _docs/report/build/ , which is gitignored: a 3 MB PDF regenerated on every run
does not belong in the history.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LATEX = ROOT / "_docs" / "report" / "latex"
BUILD = ROOT / "_docs" / "report" / "build"

# Overfull boxes under this are invisible on the page. TeX reports a 0.5pt overrun with the
# same urgency as a 40pt one, and a list where everything is flagged gets read as noise.
VISIBLE_PT = 5.0


def _tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        sys.exit(f"{name} is not on PATH -- is the TeX installation finished?")
    return found


def _run(cmd: list[str], label: str) -> str:
    print(f"  {label} ...", flush=True)
    proc = subprocess.run(cmd, cwd=BUILD, capture_output=True, text=True,
                          encoding="utf8", errors="replace")
    # pdflatex exits non-zero on an error it recovered from, so the log is the authority on
    # whether a PDF came out, not the exit code.
    return proc.stdout + proc.stderr


def main() -> int:
    pdflatex, biber = _tool("pdflatex"), _tool("biber")

    if BUILD.exists():
        shutil.rmtree(BUILD)
    shutil.copytree(LATEX, BUILD)
    # The checker is not part of the document and its .py would only clutter the build dir.
    (BUILD / "check_latex.py").unlink(missing_ok=True)

    flags = ["-interaction=nonstopmode", "-halt-on-error", "-file-line-error", "main.tex"]
    log = _run([pdflatex] + flags, "pdflatex (1/3)")
    if not (BUILD / "main.aux").exists():
        print("\nthe first pass produced no .aux -- the errors above are fatal:\n")
        print("\n".join(l for l in log.splitlines() if re.search(r"^\S+\.tex:\d+:|^! ", l)))
        return 1

    _run([biber, "main"], "biber")
    _run([pdflatex] + flags, "pdflatex (2/3)")
    _run([pdflatex] + flags, "pdflatex (3/3)")

    pdf = BUILD / "main.pdf"
    if not pdf.exists():
        print("\nno PDF was produced")
        return 1

    text = (BUILD / "main.log").read_text(encoding="utf8", errors="replace")

    pages = re.findall(r"Output written on main\.pdf \((\d+) page", text)
    print(f"\n  PDF: {pdf}  ({pdf.stat().st_size / 1024:.0f} KB"
          + (f", {pages[-1]} pages)" if pages else ")"))

    # --- what a person should look at -------------------------------------------------
    undefined_ref = sorted(set(re.findall(r"Reference `([^']+)' on page", text)))
    undefined_cit = sorted(set(re.findall(r"Citation `([^']+)' on page", text)))

    boxes: dict[float, list[str]] = defaultdict(list)
    for m in re.finditer(r"Overfull \\hbox \(([\d.]+)pt too wide\).*?lines (\d+)--(\d+)", text):
        pt = float(m.group(1))
        if pt >= VISIBLE_PT:
            boxes[pt].append(f"lines {m.group(2)}--{m.group(3)}")

    print()
    if undefined_ref:
        print(f"  UNDEFINED REFERENCES ({len(undefined_ref)}): {', '.join(undefined_ref)}")
    if undefined_cit:
        print(f"  UNDEFINED CITATIONS ({len(undefined_cit)}): {', '.join(undefined_cit)}")
    if not undefined_ref and not undefined_cit:
        print("  no undefined references or citations")

    if boxes:
        print(f"\n  OVERFULL BOXES over {VISIBLE_PT:.0f}pt "
              f"({sum(len(v) for v in boxes.values())}), worst first:")
        for pt in sorted(boxes, reverse=True)[:15]:
            for where in boxes[pt]:
                print(f"    {pt:7.1f}pt   {where}")
    else:
        print(f"\n  no overfull box wider than {VISIBLE_PT:.0f}pt")

    return 0


if __name__ == "__main__":
    sys.exit(main())
