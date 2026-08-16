"""Retrieval for the Clinical Reasoning Agent.

Retrieval is a second place evidence can enter the prompt, so it is treated with the same
suspicion as the model's own output. Three properties matter more than retrieval quality:

    1. An irrelevant hit must be reported as no hit. Handing a small model plausible text
       that does not bear on the case creates a new surface for ungrounded claims -- worse
       than retrieving nothing, because the answer then *looks* grounded.
    2. A citation must be checkable. `[2]` in an answer has to correspond to a passage that
       was actually retrieved, or it is a fabrication of a different kind.
    3. Placeholder text can never pass as evidence. The corpus ships with unsourced
       placeholders so the pipeline is testable before real passages exist; any answer
       resting on them is marked as not guideline-grounded.

Default backend is TF-IDF: for a corpus of a few dozen passages it is competitive, needs no
model download, and its scores are inspectable. A dense backend is available for comparison.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_CORPUS = Path(__file__).parent / "corpus" / "pocus_corpus.json"

# Below this cosine similarity a hit is treated as no hit. Set from the corpus, not from
# taste: with a few dozen short passages, anything under ~0.10 is usually a stopword match.
RELEVANCE_FLOOR = 0.10
DEFAULT_K = 4


def load_corpus(path: str | Path | None = None) -> dict[str, Any]:
    data = json.loads(Path(path or DEFAULT_CORPUS).read_text(encoding="utf8"))
    passages = data.get("passages", [])
    if not passages:
        raise ValueError("corpus contains no passages")
    for p in passages:
        for key in ("id", "topic", "text", "source", "status"):
            if key not in p:
                raise ValueError(f"passage {p.get('id', '?')} missing {key!r}")
    return data


def corpus_status(data: dict[str, Any]) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in data["passages"]:
        out[p["status"]] = out.get(p["status"], 0) + 1
    return out


# ---------------------------------------------------------------------------------------
def build_query(state: dict) -> str:
    """Turn a clinical state into a retrieval query.

    Uses the presenting complaint, the detected findings and the abnormal values -- the
    things a clinician would look up. Deliberately excludes absent tests: querying on what
    was NOT measured retrieves text about it and invites the model to reason from it.
    """
    parts: list[str] = []
    cc = (state.get("demographics") or {}).get("chief_complaint")
    if cc:
        parts.append(str(cc))
    for f in state["imaging"]["findings"]:
        if f.get("detected"):
            parts.append(f["label"])
    for name, v in (state.get("vitals") or {}).items():
        if v.get("flag") in ("high", "low"):
            parts.append(f"{v['flag']} {name}")
    for name, v in (state.get("labs") or {}).items():
        if v.get("flag") in ("high", "low"):
            parts.append(f"{v['flag']} {name}")
    for s in state["imaging"].get("out_of_scope", []):
        parts.append(s.split(":")[-1])
    return " ".join(parts) or "emergency point of care ultrasound"


# ---------------------------------------------------------------------------------------
class Retriever:
    """TF-IDF retrieval over the corpus. `backend='dense'` uses sentence-transformers."""

    def __init__(self, corpus: dict[str, Any] | None = None, *, backend: str = "tfidf",
                 floor: float = RELEVANCE_FLOOR, model_name: str = "all-MiniLM-L6-v2"):
        self.corpus = corpus or load_corpus()
        self.passages = self.corpus["passages"]
        self.floor = floor
        self.backend = backend
        texts = [f"{p['topic']} {p['text']}" for p in self.passages]

        if backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self._matrix = self._vec.fit_transform(texts)
        elif backend == "dense":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self._matrix = self._model.encode(texts, normalize_embeddings=True)
        else:
            raise ValueError(f"unknown backend {backend!r}; expected tfidf or dense")

    def retrieve(self, query: str, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        """Top passages above the relevance floor. Empty list means nothing relevant, which
        the prompt states explicitly rather than passing off as an absence of need."""
        import numpy as np

        if self.backend == "tfidf":
            from sklearn.metrics.pairwise import cosine_similarity
            scores = cosine_similarity(self._vec.transform([query]), self._matrix)[0]
        else:
            q = self._model.encode([query], normalize_embeddings=True)
            scores = (self._matrix @ q[0])

        order = np.argsort(scores)[::-1][:k]
        hits = []
        for rank, i in enumerate(order, start=1):
            if scores[i] < self.floor:
                continue
            p = self.passages[int(i)]
            hits.append({"n": len(hits) + 1, "id": p["id"], "topic": p["topic"],
                         "source": p["source"], "text": p["text"],
                         "status": p["status"], "score": round(float(scores[i]), 4)})
        return hits

    def for_state(self, state: dict, k: int = DEFAULT_K) -> list[dict[str, Any]]:
        return self.retrieve(build_query(state), k=k)


# ---------------------------------------------------------------------------------------
def retrieval_note(hits: list[dict[str, Any]]) -> str:
    """One line describing what grounding the answer may claim."""
    if not hits:
        return "no relevant passage retrieved"
    placeholders = [h["id"] for h in hits if h["status"] != "sourced"]
    if placeholders:
        return (f"{len(hits)} passage(s) retrieved, of which {len(placeholders)} are "
                f"PLACEHOLDER ({', '.join(placeholders)}) -- not guideline-grounded")
    return f"{len(hits)} sourced passage(s) retrieved"


def is_grounded(hits: list[dict[str, Any]]) -> bool:
    """Whether an answer resting on these hits may claim grounding in real evidence."""
    return bool(hits) and all(h["status"] == "sourced" for h in hits)
