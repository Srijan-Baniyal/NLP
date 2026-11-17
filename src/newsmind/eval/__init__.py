"""Evaluation helpers: ROUGE, retrieval@k, simple EM/F1."""

from __future__ import annotations

import re
from collections import Counter

from rouge_score import rouge_scorer

from newsmind.retrieve import Hit


def rouge_scores(prediction: str, reference: str) -> dict[str, float]:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, prediction)
    return {name: float(val.fmeasure) for name, val in scores.items()}


def _normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, reference: str) -> float:
    return 1.0 if _normalize(prediction) == _normalize(reference) else 0.0


def token_f1(prediction: str, reference: str) -> float:
    pred = _normalize(prediction).split()
    ref = _normalize(reference).split()
    if not pred and not ref:
        return 1.0
    if not pred or not ref:
        return 0.0
    common = Counter(pred) & Counter(ref)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred)
    recall = num_same / len(ref)
    return 2 * precision * recall / (precision + recall)


def precision_at_k(hits: list[Hit], relevant_ids: set[str], k: int) -> float:
    top = hits[:k]
    if not top:
        return 0.0
    hits_ids = {str(h.meta.get("article_id") or h.meta.get("chunk_id") or "") for h in top}
    return len(hits_ids & relevant_ids) / k


def recall_at_k(hits: list[Hit], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return 0.0
    top = hits[:k]
    hits_ids = {str(h.meta.get("article_id") or h.meta.get("chunk_id") or "") for h in top}
    return len(hits_ids & relevant_ids) / len(relevant_ids)
