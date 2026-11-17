"""Optional Word2Vec semantic retrieval baseline (Gensim)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from gensim.models import Word2Vec

from newsmind.preprocess import preprocess
from newsmind.retrieve import Hit


@dataclass
class Word2VecRetriever:
    model: Word2Vec
    doc_vectors: np.ndarray
    metas: list[dict[str, Any]]

    @classmethod
    def fit(
        cls,
        texts: list[str],
        metas: list[dict[str, Any]],
        vector_size: int = 100,
    ) -> Word2VecRetriever:
        tokenized = [preprocess(text, lemmatize=False) for text in texts]
        model = Word2Vec(
            sentences=tokenized,
            vector_size=vector_size,
            window=5,
            min_count=2,
            workers=2,
            seed=42,
        )
        vectors = np.vstack([_doc_vector(model, toks, vector_size) for toks in tokenized]).astype(
            np.float32
        )
        norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8
        vectors = vectors / norms
        return cls(model=model, doc_vectors=vectors, metas=metas)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        q = _doc_vector(
            self.model,
            preprocess(query, lemmatize=False),
            self.doc_vectors.shape[1],
        )
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = self.doc_vectors @ q
        k = min(top_k, len(scores))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [Hit(int(i), float(scores[i]), self.metas[int(i)]) for i in idxs]


def _doc_vector(model: Word2Vec, tokens: list[str], dim: int) -> np.ndarray:
    vecs = [model.wv[tok] for tok in tokens if tok in model.wv]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
