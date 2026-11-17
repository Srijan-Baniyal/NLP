"""Grounded generation for RAG answers."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import torch

from newsmind.retrieve import Hit


@lru_cache(maxsize=1)
def _seq2seq(model_name: str) -> tuple[Any, Any]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def _generate_text(prompt: str, model_name: str, max_new_tokens: int = 220) -> str:
    tokenizer, model = _seq2seq(model_name)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            min_new_tokens=20,
            num_beams=4,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def format_context(hits: list[Hit], max_chars: int = 1800) -> str:
    parts: list[str] = []
    used = 0
    for i, hit in enumerate(hits, start=1):
        title = str(hit.meta.get("title") or "")
        outlet = str(hit.meta.get("outlet") or "")
        text = str(hit.meta.get("text") or "")
        snippet = text[:450]
        block = f"[{i}] {outlet}: {title}. {snippet}"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def extractive_fallback(hits: list[Hit]) -> str:
    lines: list[str] = []
    for i, hit in enumerate(hits[:3], start=1):
        outlet = str(hit.meta.get("outlet") or "source")
        title = str(hit.meta.get("title") or "").strip()
        text = str(hit.meta.get("text") or "").strip()
        piece = title or text[:220]
        lines.append(f"[{i}] ({outlet}) {piece}")
    return "Based on the retrieved evidence:\n" + "\n".join(lines)


def generate_answer(question: str, hits: list[Hit], model_name: str) -> str:
    if not hits:
        return "I could not find supporting evidence in the indexed corpus."
    context = format_context(hits)
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n"
        "Write a short answer using only the context. Include citation markers like [1]."
    )
    try:
        answer = _generate_text(prompt, model_name, max_new_tokens=160)
    except Exception:  # noqa: BLE001
        return extractive_fallback(hits)
    if len(answer) < 24 or answer.strip() in {"[1].", "[1]", "Yes.", "No."}:
        return extractive_fallback(hits)
    return answer


def answer_with_citations(question: str, hits: list[Hit], model_name: str) -> dict[str, object]:
    answer = generate_answer(question, hits, model_name)
    citations = [
        {
            "rank": i,
            "score": round(h.score, 4),
            "title": h.meta.get("title", ""),
            "outlet": h.meta.get("outlet", ""),
            "url": h.meta.get("url", ""),
            "snippet": str(h.meta.get("text", ""))[:280],
            "article_id": h.meta.get("article_id", ""),
            "chunk_id": h.meta.get("chunk_id", ""),
        }
        for i, h in enumerate(hits, start=1)
    ]
    return {"answer": answer, "citations": citations, "context": format_context(hits)}
