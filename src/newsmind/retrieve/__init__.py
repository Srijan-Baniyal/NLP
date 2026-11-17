"""Classical and dense retrieval backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from newsmind.preprocess import preprocess


@dataclass(slots=True)
class Hit:
    index: int
    score: float
    meta: dict[str, Any]


class TfidfRetriever:
    def __init__(self, texts: list[str], metas: list[dict[str, Any]]) -> None:
        self.metas = metas
        self.vectorizer = TfidfVectorizer(
            max_features=50_000,
            ngram_range=(1, 2),
            preprocessor=lambda t: " ".join(preprocess(t, lemmatize=False)),
            token_pattern=r"(?u)\b\w+\b",
        )
        self.matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        q = self.vectorizer.transform([query])
        scores = cosine_similarity(q, self.matrix).ravel()
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [Hit(int(i), float(scores[i]), self.metas[int(i)]) for i in idxs]


class BM25Retriever:
    def __init__(self, texts: list[str], metas: list[dict[str, Any]]) -> None:
        self.metas = metas
        self.corpus_tokens = [preprocess(text, lemmatize=False) for text in texts]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        tokens = preprocess(query, lemmatize=False)
        scores = np.asarray(self.bm25.get_scores(tokens), dtype=np.float64)
        if len(scores) == 0:
            return []
        k = min(top_k, len(scores))
        idxs = np.argpartition(-scores, k - 1)[:k]
        idxs = idxs[np.argsort(-scores[idxs])]
        return [Hit(int(i), float(scores[i]), self.metas[int(i)]) for i in idxs]


class DenseRetriever:
    def __init__(self, texts: list[str], metas: list[dict[str, Any]], model_name: str) -> None:
        import faiss
        from sentence_transformers import SentenceTransformer

        self.metas = metas
        self.model = SentenceTransformer(model_name)
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self.embeddings = np.asarray(embeddings, dtype=np.float32)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def search(self, query: str, top_k: int = 10) -> list[Hit]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(np.asarray(q, dtype=np.float32), top_k)
        hits: list[Hit] = []
        for score, idx in zip(scores[0], idxs[0], strict=True):
            if int(idx) < 0:
                continue
            hits.append(Hit(int(idx), float(score), self.metas[int(idx)]))
        return hits


def _minmax(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    vals = np.asarray(list(scores.values()), dtype=np.float64)
    lo, hi = float(vals.min()), float(vals.max())
    if hi - lo < 1e-12:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


class HybridRetriever:
    """Reciprocal-style fusion of BM25 + dense with alpha blending."""

    def __init__(
        self,
        bm25: BM25Retriever,
        dense: DenseRetriever,
        *,
        alpha: float = 0.55,
    ) -> None:
        self.bm25 = bm25
        self.dense = dense
        self.alpha = alpha
        self.metas = dense.metas

    def search(self, query: str, top_k: int = 20) -> list[Hit]:
        sparse = self.bm25.search(query, top_k=top_k * 2)
        dens = self.dense.search(query, top_k=top_k * 2)
        return self.fuse(sparse, dens, top_k=top_k)

    def fuse(self, sparse: list[Hit], dens: list[Hit], top_k: int = 20) -> list[Hit]:
        """Blend already-retrieved sparse and dense hits without searching again."""
        sparse_map = _minmax({h.index: h.score for h in sparse})
        dense_map = _minmax({h.index: h.score for h in dens})
        keys = set(sparse_map) | set(dense_map)
        fused: list[Hit] = []
        for idx in keys:
            score = self.alpha * dense_map.get(idx, 0.0) + (1.0 - self.alpha) * sparse_map.get(
                idx, 0.0
            )
            fused.append(Hit(idx, score, self.metas[idx]))
        fused.sort(key=lambda h: h.score, reverse=True)
        return fused[:top_k]


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        if not hits:
            return []
        pairs = [(query, str(h.meta.get("text", ""))) for h in hits]
        scores = self.model.predict(pairs)
        ranked = sorted(
            (Hit(h.index, float(score), h.meta) for h, score in zip(hits, scores, strict=True)),
            key=lambda h: h.score,
            reverse=True,
        )
        return ranked[:top_k]
