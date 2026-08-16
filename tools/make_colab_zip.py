"""Build the zip uploaded to Colab.

    python tools/make_colab_zip.py

Regenerate this after ANY change under src/ before running a notebook. A stale zip is not
a loud failure: the notebook imports successfully and produces plausible results from old
code, which once looked like a finding rather than a mistake. The notebook guards against
it by asserting the corpus version, but the guard only helps if the version was bumped.

Contents:
  src/**                                  the packages under test
  models/clinical_reasoning_pre_rag/*     the frozen baseline, so the no-retrieval arm can
                                          be checked against it inside the notebook
"""
from __future__ import annotations

import json
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "pocus_agents_colab.zip"

BASELINE = [
    "models/clinical_reasoning_pre_rag/results.json",
    "models/clinical_reasoning_pre_rag/metrics.json",
    "models/clinical_reasoning_pre_rag/VERSION.json",
]


def main() -> int:
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in (ROOT / "src").rglob("*"):
            if p.is_dir() or "__pycache__" in p.parts:
                continue
            z.write(p, p.relative_to(ROOT).as_posix())
            n += 1
        for extra in BASELINE:
            f = ROOT / extra
            if f.exists():
                z.write(f, extra)
                n += 1
            else:
                print(f"  warning: {extra} missing -- section 9 of the A/B notebook "
                      f"cannot check for confounding without it")

    corpus = json.loads(
        (ROOT / "src/agents/clinical/corpus/pocus_corpus.json").read_text(encoding="utf-8"))
    sourced = sum(p["status"] == "sourced" for p in corpus["passages"])

    print(f"{OUT.name}: {n} files, {OUT.stat().st_size / 1e6:.2f} MB")
    print(f"corpus {corpus['corpus_version']}: {sourced}/{len(corpus['passages'])} sourced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
