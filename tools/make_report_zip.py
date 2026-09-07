"""Build the zip uploaded to an online LaTeX compiler.

    python tools/make_report_zip.py

Everything the document needs to compile away from this machine, and nothing else. It is
regenerated rather than edited by hand because the previous one was assembled once and then
drifted: chapters were added, a chapter was split in two, three figures were extracted from the
notebooks, and the zip still held the state of the project several weeks earlier. A stale zip
of a report is a quiet failure -- it compiles, it produces a PDF, and the PDF is the wrong
document.

The check that matters runs first: check_latex.py must pass, because a zip that cannot compile
is worse than no zip when the person compiling it is on a website with a five-minute budget.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
LATEX = ROOT / "_docs" / "report" / "latex"
OUT = ROOT / "_docs" / "report" / "POCUS-report-latex.zip"

# Directories whose whole contents ship. figures/ and logos/ are images the document
# \includegraphics; chapters/ is the body.
TREES = ["chapters", "figures", "logos"]
# Files at the top level of latex/. check_latex.py is deliberately NOT among them: it is a
# development tool, it is not \input by anything, and shipping it would invite someone to run
# a script with a hard-coded local path on a machine that does not have one.
FILES = ["main.tex", "preamble.tex", "glossary.tex", "appendices.tex", "bibfile.bib"]


def main() -> int:
    print("running the static checks first ...")
    proc = subprocess.run([sys.executable, str(LATEX / "check_latex.py")],
                          capture_output=True, text=True)
    tail = [ln for ln in (proc.stdout + proc.stderr).strip().split("\n") if ln.strip()][-6:]
    print("\n".join("  " + ln for ln in tail))
    if "no problems found" not in proc.stdout:
        print("\nREFUSING to build: the document has problems that would waste a compile.")
        return 1

    members: list[tuple[pathlib.Path, str]] = []
    for name in FILES:
        p = LATEX / name
        if not p.exists():
            print(f"MISSING {name}")
            return 1
        members.append((p, name))

    for tree in TREES:
        d = LATEX / tree
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if p.is_file() and not p.name.startswith("."):
                members.append((p, str(p.relative_to(LATEX)).replace("\\", "/")))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for path, arc in members:
            z.write(path, arc)

    print(f"\nwrote {OUT}")
    print(f"  {len(members)} entries, {OUT.stat().st_size / 1024:.0f} KB")
    for _, arc in sorted(members, key=lambda m: m[1]):
        print(f"    {arc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
