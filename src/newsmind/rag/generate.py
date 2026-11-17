"""Grounded generation for RAG answers."""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

import torch

from newsmind.rag.chunking import split_sentences
from newsmind.retrieve import Hit

_STOP = frozenset(
    "a an the is are was were be been of in on at to for and or with by from as that this "
    "it its what who whom which when where why how did does do has have had about key facts "
    "article story tell me".split()
)


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
            num_beams=2,
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


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _complete_sentences(text: str) -> str:
    """Drop a trailing cut-off fragment so the answer ends on a full sentence."""
    text = text.strip()
    if not text or text[-1] in ".!?\"'”":
        return text
    cut = max(text.rfind(". "), text.rfind("! "), text.rfind("? "))
    return text[: cut + 1].strip() if cut > 20 else text.rstrip(" ,;:") + "."


def _is_copy(answer: str, context: str, n: int = 6) -> bool:
    """True when the answer is mostly a verbatim slice of the retrieved context."""
    words = _words(answer)
    if len(words) < n:
        return False
    ctx = " ".join(_words(context))
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    return sum(g in ctx for g in grams) / len(grams) > 0.6


def clean_answer(answer: str, context: str = "") -> str:
    """Strip echoed context headers, collapse whitespace, end on a full sentence."""
    answer = re.sub(r"^\s*\[\d+\]\s*[^:\n]{0,60}:\s*(?:\.\s*)?", "", answer)
    answer = re.sub(r"\s+", " ", answer).strip().lstrip(". ")
    return _complete_sentences(answer)


def extractive_fallback(hits: list[Hit], question: str = "") -> str:
    """Pick the few most question-relevant complete sentences, each with its citation."""
    q_terms = {w for w in _words(question) if w not in _STOP}
    scored: list[tuple[float, int, int, str]] = []
    for i, hit in enumerate(hits[:5], start=1):
        text = str(hit.meta.get("text") or "")
        for j, sent in enumerate(split_sentences(text[:1500])):
            sent = sent.strip()
            if len(sent) < 40:
                continue
            overlap = len(q_terms & set(_words(sent)))
            scored.append((overlap - 0.05 * j - 0.1 * i, i, j, sent))
    if not scored:
        return "I could not find a clear answer in the retrieved evidence."
    relevant = [t for t in scored if t[0] > 0]
    top = sorted(relevant or scored, reverse=True)[:3]
    top.sort(key=lambda t: (t[1], t[2]))
    return " ".join(f"{_complete_sentences(s)} [{i}]" for _, i, _, s in top)


def generate_answer(question: str, hits: list[Hit], model_name: str) -> str:
    if not hits:
        return "I could not find supporting evidence in the indexed corpus."
    context = format_context(hits)
    prompt = (
        f"Context: {context}\n\n"
        f"Question: {question}\n"
        "Answer in one to three complete sentences, in your own words, using only the "
        "context. Do not copy the context. End with citation markers like [1]."
    )
    try:
        answer = clean_answer(_generate_text(prompt, model_name, max_new_tokens=160), context)
    except Exception:  # noqa: BLE001
        return extractive_fallback(hits, question)
    if (
        len(answer) < 24
        or answer.strip() in {"[1].", "[1]", "Yes.", "No."}
        or _is_copy(answer, context)
    ):
        return extractive_fallback(hits, question)
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
