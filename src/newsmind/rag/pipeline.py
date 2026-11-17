"""Hybrid RAG orchestration: retrieve → rerank → generate with citations."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from newsmind.config import (
    DENSE_MODEL,
    GENERATOR_MODEL,
    HYBRID_ALPHA,
    RERANK_MODEL,
    RERANK_TOP_K,
    RETRIEVE_TOP_K,
)
from newsmind.rag.chunking import chunk_article
from newsmind.rag.generate import answer_with_citations, format_context
from newsmind.retrieve import (
    BM25Retriever,
    CrossEncoderReranker,
    DenseRetriever,
    Hit,
    HybridRetriever,
)


@dataclass
class RAGSystem:
    hybrid: HybridRetriever
    reranker: CrossEncoderReranker
    generator_model: str = GENERATOR_MODEL
    dense_model: str = DENSE_MODEL
    rerank_model: str = RERANK_MODEL
    alpha: float = HYBRID_ALPHA

    def ask(
        self,
        question: str,
        *,
        retrieve_k: int = RETRIEVE_TOP_K,
        rerank_k: int = RERANK_TOP_K,
        use_rerank: bool = True,
    ) -> dict[str, object]:
        return self.ask_traced(
            question,
            retrieve_k=retrieve_k,
            rerank_k=rerank_k,
            use_rerank=use_rerank,
        )

    def ask_traced(
        self,
        question: str,
        *,
        retrieve_k: int = RETRIEVE_TOP_K,
        rerank_k: int = RERANK_TOP_K,
        use_rerank: bool = True,
        article_hint: str | None = None,
    ) -> dict[str, Any]:
        """Run RAG and return answer + a detailed step-by-step pipeline trace."""
        query = question.strip()
        if article_hint:
            query = f"{question.strip()}\n\nFocus on this article context:\n{article_hint[:1200]}"

        steps: list[dict[str, Any]] = []
        steps.append(
            {
                "step": 1,
                "name": "Query intake",
                "detail": (
                    "The user question is normalized and optionally grounded with the "
                    "opened article as extra context so retrieval prefers related passages."
                ),
                "inputs": {"question": question, "used_article_hint": bool(article_hint)},
                "outputs": {"effective_query_preview": query[:240]},
            }
        )

        # BM25 cost grows with query length, so it only sees the question; the
        # article hint is used by the dense retriever, which handles long text cheaply.
        sparse_hits = self.hybrid.bm25.search(question.strip(), top_k=retrieve_k * 2)
        dense_model = getattr(self, "dense_model", DENSE_MODEL)
        rerank_model = getattr(self, "rerank_model", RERANK_MODEL)
        alpha = float(getattr(self, "alpha", HYBRID_ALPHA))
        steps.append(
            {
                "step": 2,
                "name": "Sparse retrieval (BM25)",
                "detail": (
                    "Classical lexical matching over lemmatized/stopword-filtered tokens. "
                    "Good at exact keyword overlap (names, places, numbers)."
                ),
                "inputs": {"top_k": retrieve_k * 2},
                "outputs": {
                    "hits": [
                        {
                            "rank": i,
                            "score": round(h.score, 4),
                            "outlet": h.meta.get("outlet", ""),
                            "title": str(h.meta.get("title", ""))[:80],
                            "snippet": str(h.meta.get("text", ""))[:160],
                        }
                        for i, h in enumerate(sparse_hits[:8], start=1)
                    ]
                },
            }
        )

        dense_hits = self.hybrid.dense.search(query, top_k=retrieve_k * 2)
        steps.append(
            {
                "step": 3,
                "name": f"Dense retrieval ({dense_model})",
                "detail": (
                    "Sentence embeddings (MiniLM) stored in a FAISS inner-product index. "
                    "Captures semantic similarity even when wording differs."
                ),
                "inputs": {"top_k": retrieve_k * 2, "model": dense_model},
                "outputs": {
                    "hits": [
                        {
                            "rank": i,
                            "score": round(h.score, 4),
                            "outlet": h.meta.get("outlet", ""),
                            "title": str(h.meta.get("title", ""))[:80],
                            "snippet": str(h.meta.get("text", ""))[:160],
                        }
                        for i, h in enumerate(dense_hits[:8], start=1)
                    ]
                },
            }
        )

        fused = self.hybrid.fuse(sparse_hits, dense_hits, top_k=retrieve_k)
        steps.append(
            {
                "step": 4,
                "name": "Hybrid fusion",
                "detail": (
                    f"Min-max normalize BM25 and dense scores, then blend with "
                    f"alpha={alpha:.2f} on dense and {1 - alpha:.2f} on BM25."
                ),
                "inputs": {"alpha_dense": alpha, "alpha_bm25": round(1 - alpha, 2)},
                "outputs": {
                    "hits": [
                        {
                            "rank": i,
                            "score": round(h.score, 4),
                            "outlet": h.meta.get("outlet", ""),
                            "title": str(h.meta.get("title", ""))[:80],
                            "snippet": str(h.meta.get("text", ""))[:160],
                        }
                        for i, h in enumerate(fused[:10], start=1)
                    ]
                },
            }
        )

        if use_rerank and fused:
            final_hits = self.reranker.rerank(question.strip(), fused, top_k=rerank_k)
            steps.append(
                {
                    "step": 5,
                    "name": f"Cross-encoder rerank ({rerank_model})",
                    "detail": (
                        "A cross-encoder scores (question, passage) pairs jointly and "
                        "keeps the top passages most likely to answer the question."
                    ),
                    "inputs": {"model": rerank_model, "candidates": len(fused)},
                    "outputs": {
                        "hits": [
                            {
                                "rank": i,
                                "score": round(h.score, 4),
                                "outlet": h.meta.get("outlet", ""),
                                "title": str(h.meta.get("title", ""))[:80],
                                "snippet": str(h.meta.get("text", ""))[:160],
                            }
                            for i, h in enumerate(final_hits, start=1)
                        ]
                    },
                }
            )
        else:
            final_hits = fused[:rerank_k]
            steps.append(
                {
                    "step": 5,
                    "name": "Rerank skipped",
                    "detail": "Using the top hybrid-fused passages without a cross-encoder.",
                    "inputs": {},
                    "outputs": {"kept": len(final_hits)},
                }
            )

        context = format_context(final_hits)
        result = answer_with_citations(question, final_hits, self.generator_model)
        steps.append(
            {
                "step": 6,
                "name": f"Grounded generation ({self.generator_model})",
                "detail": (
                    "FLAN-T5 generates an answer constrained to the retrieved evidence. "
                    "If the model returns a degenerate reply, an extractive citation "
                    "fallback is used instead."
                ),
                "inputs": {
                    "model": self.generator_model,
                    "context_chars": len(context),
                    "passages": len(final_hits),
                },
                "outputs": {"answer": result["answer"], "citations": result["citations"]},
            }
        )
        result["pipeline"] = steps
        return result

    def ask_article(
        self,
        question: str,
        article: dict[str, Any],
        *,
        top_k: int = 3,
    ) -> dict[str, Any]:
        """Answer using only passages from the opened article (no corpus search)."""

        chunks = chunk_article(
            article_id=str(article.get("article_id") or "open"),
            corpus=str(article.get("corpus") or ""),
            outlet=str(article.get("outlet") or ""),
            title=str(article.get("title") or ""),
            text=str(article.get("text") or ""),
            url=str(article.get("url") or ""),
            size=500,
            overlap=0,
        )
        hits = [Hit(index=i, score=0.0, meta=asdict(c)) for i, c in enumerate(chunks)]
        final_hits = self.reranker.rerank(question.strip(), hits, top_k=top_k)
        context = format_context(final_hits)
        result = answer_with_citations(question, final_hits, self.generator_model)
        result["pipeline"] = [
            {
                "step": 1,
                "name": "Chunk the open article",
                "detail": "Split only the opened article into short sentence-aware passages.",
                "outputs": {"passages": len(chunks)},
            },
            {
                "step": 2,
                "name": f"Cross-encoder rerank ({self.rerank_model})",
                "detail": "Score each (question, passage) pair and keep the best ones.",
                "outputs": {
                    "hits": [
                        {
                            "rank": i,
                            "score": round(h.score, 4),
                            "snippet": str(h.meta.get("text", ""))[:160],
                        }
                        for i, h in enumerate(final_hits, start=1)
                    ]
                },
            },
            {
                "step": 3,
                "name": f"Grounded generation ({self.generator_model})",
                "detail": "FLAN-T5 answers using only the selected passages of this article.",
                "inputs": {"model": self.generator_model, "context_chars": len(context)},
                "outputs": {"answer": result["answer"]},
            },
        ]
        return result

    def retrieve_only(self, question: str, top_k: int = RETRIEVE_TOP_K) -> list[Hit]:
        return self.hybrid.search(question, top_k=top_k)


def build_rag(
    texts: list[str],
    metas: list[dict],
    *,
    dense_model: str,
    rerank_model: str,
    alpha: float = HYBRID_ALPHA,
) -> RAGSystem:
    bm25 = BM25Retriever(texts, metas)
    dense = DenseRetriever(texts, metas, dense_model)
    hybrid = HybridRetriever(bm25, dense, alpha=alpha)
    reranker = CrossEncoderReranker(rerank_model)
    return RAGSystem(
        hybrid=hybrid,
        reranker=reranker,
        dense_model=dense_model,
        rerank_model=rerank_model,
        alpha=alpha,
    )
