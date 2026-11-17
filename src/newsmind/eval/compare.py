"""Comparative evaluation helpers for classical vs modern NLP paths."""

from __future__ import annotations

from typing import Any

from newsmind.eval import precision_at_k, recall_at_k, rouge_scores
from newsmind.preprocess.syntax import analyze_syntax
from newsmind.retrieve import BM25Retriever, Hit, TfidfRetriever
from newsmind.retrieve.glove import GloVeRetriever
from newsmind.retrieve.word2vec import Word2VecRetriever
from newsmind.summarize import SUMMARIZER_MODES, summarize_traced


def compare_retrievers(
    query: str,
    texts: list[str],
    metas: list[dict[str, Any]],
    *,
    top_k: int = 5,
    relevant_ids: set[str] | None = None,
) -> dict[str, Any]:
    """Run TF-IDF, BM25, Word2Vec, and GloVe side-by-side."""
    syntax = analyze_syntax(query)
    tfidf = TfidfRetriever(texts, metas)
    bm25 = BM25Retriever(texts, metas)
    w2v = Word2VecRetriever.fit(texts, metas)
    try:
        glove = GloVeRetriever.fit(texts, metas)
        glove_hits = glove.search(query, top_k=top_k)
    except Exception as exc:  # noqa: BLE001
        glove_hits = []
        glove_error = str(exc)
    else:
        glove_error = ""

    results = {
        "TF-IDF": tfidf.search(query, top_k=top_k),
        "BM25": bm25.search(query, top_k=top_k),
        "Word2Vec": w2v.search(query, top_k=top_k),
        "GloVe": glove_hits,
    }
    table: list[dict[str, Any]] = []
    for name, hits in results.items():
        row: dict[str, Any] = {
            "method": name,
            "top1_title": str(hits[0].meta.get("title", ""))[:80] if hits else "",
            "top1_score": round(hits[0].score, 4) if hits else 0.0,
        }
        if relevant_ids:
            row["precision@k"] = round(precision_at_k(hits, relevant_ids, top_k), 4)
            row["recall@k"] = round(recall_at_k(hits, relevant_ids, top_k), 4)
        table.append(row)

    return {
        "syntax": {
            "tokens": syntax.tokens[:30],
            "pos_tags": syntax.pos_tags[:30],
            "noun_phrases": syntax.noun_phrases,
        },
        "table": table,
        "hits": {name: _hits_preview(hits) for name, hits in results.items()},
        "glove_error": glove_error,
    }


def compare_summarizers(
    text: str,
    reference: str,
    *,
    modes: tuple[str, ...] | None = None,
    bart_model: str,
) -> list[dict[str, Any]]:
    modes = modes or SUMMARIZER_MODES
    rows: list[dict[str, Any]] = []
    for mode in modes:
        # Skip heavy abstractive models in bulk unless explicitly requested later;
        # caller chooses modes.
        traced = summarize_traced(text, mode=mode, model_name=bart_model)
        summary = str(traced.get("summary") or "")
        row: dict[str, Any] = {"mode": mode, "summary": summary[:400]}
        if reference.strip():
            row.update(rouge_scores(summary, reference))
        rows.append(row)
    return rows


def _hits_preview(hits: list[Hit]) -> list[dict[str, Any]]:
    return [
        {
            "rank": i,
            "score": round(h.score, 4),
            "outlet": h.meta.get("outlet", ""),
            "title": str(h.meta.get("title", ""))[:100],
            "snippet": str(h.meta.get("text", ""))[:180],
        }
        for i, h in enumerate(hits, start=1)
    ]
