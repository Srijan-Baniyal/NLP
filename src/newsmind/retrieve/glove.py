"""GloVe embedding retrieval baseline (Gensim KeyedVectors)."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from gensim.downloader import load as gensim_load
from gensim.models import KeyedVectors

from newsmind.preprocess import preprocess
from newsmind.retrieve import Hit


@lru_cache(maxsize=1)
def load_glove(name: str = "glove-wiki-gigaword-100") -> KeyedVectors:
    return gensim_load(name)


@dataclass
class GloVeRetriever:
    vectors: KeyedVectors
    doc_vectors: np.ndarray
    metas: list[dict[str, Any]]

    @classmethod
    def fit(
        cls,
        texts: list[str],
        metas: list[dict[str, Any]],
        glove_name: str = "glove-wiki-gigaword-100",
    ) -> GloVeRetriever:
        keyed = load_glove(glove_name)
        dim = int(keyed.vector_size)
        matrix = np.vstack([_doc_vector(keyed, preprocess(t, lemmatize=False), dim) for t in texts])
        norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
        return cls(vectors=keyed, doc_vectors=(matrix / norms).astype(np.float32), metas=metas)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        dim = self.doc_vectors.shape[1]
        q = _doc_vector(self.vectors, preprocess(query, lemmatize=False), dim)
        q = q / (np.linalg.norm(q) + 1e-8)
        scores = self.doc_vectors @ q
        k = min(top_k, len(scores))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [Hit(int(i), float(scores[i]), self.metas[int(i)]) for i in idxs]


def _doc_vector(keyed: KeyedVectors, tokens: list[str], dim: int) -> np.ndarray:
    vecs = [keyed[tok] for tok in tokens if tok in keyed]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(np.asarray(vecs, dtype=np.float32), axis=0)
