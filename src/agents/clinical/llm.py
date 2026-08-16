"""Local language-model backends for the Clinical Reasoning Agent.

Everything here produces the same thing: a callable ``llm_fn(system, user) -> str`` that
``reasoning.reason`` accepts. Nothing in this module decides anything clinical. The
escalation decision is computed before the model runs and the model's output is checked
against the structured state afterwards, so a backend is interchangeable and a bad one
degrades to a withheld differential rather than to a wrong answer.

Three backends:

    LlamaCppBackend   HuatuoGPT-o1-8B (or any GGUF) through llama-cpp-python, on CPU
    ScriptedBackend   returns canned replies; used to test the wiring without weights
    FailingBackend    raises or returns junk; used to test that failure degrades safely

Determinism is deliberate. Temperature is 0 and the seed is fixed, because two runs on one
patient must produce the same output -- a clinician cannot audit a recommendation that
changes when the button is pressed twice.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol

ROOT = Path(__file__).resolve().parents[3]

# Set POCUS_LLM_PATH to override. The default is where the model already lives in this repo.
DEFAULT_MODEL = ROOT / "Reference_ClinicalReasoning" / "HuatuoGPT-o1-8B-GGUF" / \
                "HuatuoGPT-o1-8B-Q4_K_M.gguf"

# HuatuoGPT-o1 is a reasoning model: it writes a chain of thought before its answer. The
# budget has to cover that as well as the JSON, or the reply is truncated mid-object and the
# parse fails for a reason that looks like a model error but is a configuration error.
DEFAULT_CTX = 4096
DEFAULT_MAX_TOKENS = 1024


class LLMBackend(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


# ---------------------------------------------------------------------------------------
# Real backend
# ---------------------------------------------------------------------------------------
class LlamaCppBackend:
    """HuatuoGPT-o1-8B through llama-cpp-python.

    The model is loaded once and reused; loading a 4.6 GB GGUF per request would dominate
    the runtime entirely.
    """

    def __init__(self, model_path: str | Path | None = None, *,
                 n_ctx: int = DEFAULT_CTX, n_threads: int | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS, seed: int = 0,
                 n_gpu_layers: int = 0, chat_format: str | None = None,
                 verbose: bool = False):
        """`n_gpu_layers=-1` offloads every layer to the GPU; 0 keeps it on CPU.

        The default is 0 because a CPU-only llama-cpp build silently ignores the argument,
        and a default of -1 would make a CPU run look like a misconfigured GPU run. On a
        CUDA build, -1 is the difference between ~20 minutes and well under a minute.
        """
        try:
            from llama_cpp import Llama
        except ImportError as exc:                                  # pragma: no cover
            raise ImportError(
                "llama-cpp-python is not installed.\n"
                "    pip install llama-cpp-python\n"
                "The model file itself is already present; only the runtime is missing."
            ) from exc

        path = Path(model_path or os.environ.get("POCUS_LLM_PATH") or DEFAULT_MODEL)
        if not path.exists():
            raise FileNotFoundError(
                f"GGUF model not found at {path}.\n"
                "Set POCUS_LLM_PATH or pass model_path explicitly.")

        self.path = path
        self.max_tokens = max_tokens
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        kw = {}
        if chat_format:
            kw["chat_format"] = chat_format
        self._llm = Llama(
            model_path=str(path),
            n_ctx=n_ctx,
            n_threads=n_threads or os.cpu_count() or 4,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            verbose=verbose,
            **kw,
        )

    def __call__(self, system: str, user: str) -> str:
        # Clear the key/value cache before every request. Without this the same case run twice
        # against one loaded model gives different answers: llama.cpp carries decoding state
        # between calls, so the second request is conditioned on the first. Observed directly
        # -- one run produced a two-entry differential, an identical rerun produced one entry.
        # A clinical output that changes when the button is pressed twice cannot be audited,
        # and a benchmark built on it measures call order as much as reasoning.
        self._llm.reset()
        out = self._llm.create_chat_completion(
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.0,      # reproducibility is a clinical requirement, not a preference
            top_p=1.0,
            seed=self.seed,
            max_tokens=self.max_tokens,
        )
        return out["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------------------
# Test backends -- these are what make the pipeline testable without a 4.6 GB download
# ---------------------------------------------------------------------------------------
class ScriptedBackend:
    """Returns prepared replies in order, then repeats the last one.

    Used to drive the pipeline through cases a real model produces only occasionally --
    a fabricated laboratory value, a truncated object, prose instead of JSON.
    """

    def __init__(self, *replies: str):
        if not replies:
            raise ValueError("ScriptedBackend needs at least one reply")
        self.replies = list(replies)
        self.calls: list[tuple[str, str]] = []

    def __call__(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        i = min(len(self.calls) - 1, len(self.replies) - 1)
        return self.replies[i]


class FailingBackend:
    """Raises, to check that a backend failure degrades to a withheld differential rather
    than to an exception reaching the caller."""

    def __init__(self, message: str = "backend unavailable"):
        self.message = message

    def __call__(self, system: str, user: str) -> str:
        raise RuntimeError(self.message)


# ---------------------------------------------------------------------------------------
def load_backend(kind: str = "llama", **kw) -> Callable[[str, str], str]:
    """Factory used by scripts and notebooks so the choice of backend is one string."""
    if kind == "llama":
        return LlamaCppBackend(**kw)
    if kind == "scripted":
        return ScriptedBackend(*kw.get("replies", ['{"differential": []}']))
    if kind == "failing":
        return FailingBackend(**kw)
    raise ValueError(f"unknown backend {kind!r}; expected llama, scripted or failing")


def is_available(model_path: str | Path | None = None) -> tuple[bool, str]:
    """Whether a real model can be loaded, and why not if it cannot.

    Reported rather than assumed, so a notebook can degrade honestly instead of failing
    halfway through a run.
    """
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        return False, "llama-cpp-python not installed (pip install llama-cpp-python)"
    path = Path(model_path or os.environ.get("POCUS_LLM_PATH") or DEFAULT_MODEL)
    if not path.exists():
        return False, f"model file not found at {path}"
    return True, f"ready: {path.name} ({path.stat().st_size / 1e9:.1f} GB)"
