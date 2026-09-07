"""Every automated check, in one command.

    python tools/checks.py            run all of them
    python tools/checks.py lint       run one

The point of a single entry is that the checks are run as a set. Run individually they get run
selectively, and the one that would have failed is the one that was skipped -- which is how a
NameError in an endpoint survived in this repository until a lint pass was finally run over it.

Each check reports what it found rather than only whether it passed, because the counts are
what make a report defensible: "bandit: 0 high, 0 medium, 2 low (subprocess in tools/, no
shell)" is a statement someone can check. "Security: OK" is not.

Exit code is the number of checks that failed, so CI fails on any of them.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# KMP_DUPLICATE_LIB_OK: torch and MKL each link their own OpenMP runtime, and without this the
# interpreter aborts at import on this machine. Set for the child, not the parent.
ENV_NOTE = {"KMP_DUPLICATE_LIB_OK": "TRUE"}

PY = sys.executable

CHECKS: dict[str, tuple[str, list[str]]] = {
    "tests": (
        "pytest -- the full suite, including the HTTP surface",
        [PY, "-m", "pytest", "src/agents/tests", "-q"],
    ),
    "benchmark": (
        "safety benchmark -- the same tests grouped by the property each exercises",
        [PY, "-m", "src.agents.tests.run_benchmark"],
    ),
    "regression": (
        "regression -- end-to-end checks over the frozen cases",
        [PY, "tools/run_regression.py"],
    ),
    "lint": (
        "ruff -- undefined names and unused code, not style opinions",
        [PY, "-m", "ruff", "check", "serve.py", "src/", "tools/"],
    ),
    "code-security": (
        "bandit -- unsafe patterns in code we wrote (gate at medium and above)",
        # -ll gates at MEDIUM. The remaining lows are the subprocess calls in tools/ -- an
        # argument list with no shell, in files the container image excludes -- and gating on
        # them would mean a permanently red check that everyone learns to ignore, which is
        # worse than not running it. They stay visible in the ungated run:
        #     python -m bandit -r serve.py src/ tools/ -c bandit.yaml
        [PY, "-m", "bandit", "-r", "serve.py", "src/", "tools/", "-c", "bandit.yaml",
         "-ll", "-q"],
    ),
    "dependency-security": (
        "pip-audit -- known advisories against the pinned runtime dependencies",
        [PY, "-m", "pip_audit", "-r", "requirements.txt", "--progress-spinner", "off"],
    ),
}


def run(name: str) -> bool:
    title, cmd = CHECKS[name]
    print(f"\n{'=' * 86}\n{name}  --  {title}\n{'=' * 86}")
    started = time.time()

    import os
    env = dict(os.environ, **ENV_NOTE)
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)

    out = (proc.stdout or "") + (proc.stderr or "")
    # The tail carries the verdict for every one of these tools; the body is detail that
    # belongs in the tool's own invocation when something has actually failed.
    tail = [ln for ln in out.strip().split("\n") if ln.strip()][-12:]
    print("\n".join(tail))

    ok = proc.returncode == 0
    print(f"\n  -> {'PASS' if ok else 'FAIL'}  ({time.time() - started:.1f}s)")
    return ok


def main() -> int:
    wanted = sys.argv[1:] or list(CHECKS)
    unknown = [w for w in wanted if w not in CHECKS]
    if unknown:
        print(f"unknown check(s): {', '.join(unknown)}")
        print(f"available: {', '.join(CHECKS)}")
        return 2

    results = {name: run(name) for name in wanted}

    print(f"\n{'=' * 86}\nSUMMARY\n{'=' * 86}")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    failed = [n for n, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    if failed:
        print(f"failed: {', '.join(failed)}")
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
