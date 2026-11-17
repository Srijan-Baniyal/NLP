"""Extractive and abstractive summarization with pipeline traces."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.text_rank import TextRankSummarizer

from newsmind.config import PEGASUS_MODEL, T5_MODEL
from newsmind.preprocess import ensure_nltk, preprocess
from newsmind.rag.chunking import split_sentences

SUMMARIZER_MODES = (
    "Extractive TF-IDF",
    "Extractive TextRank",
    "Abstractive BART",
    "Abstractive T5",
    "Abstractive Pegasus",
)


def extractive_tfidf(text: str, sentence_count: int = 3) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    if len(sentences) <= sentence_count:
        return " ".join(sentences)
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(sentences)
    scores = np.asarray(matrix.sum(axis=1)).ravel()
    top = sorted(np.argsort(-scores)[:sentence_count])
    return " ".join(sentences[i] for i in top)


def extractive_textrank(text: str, sentence_count: int = 3) -> str:
    ensure_nltk()
    if not text.strip():
        return ""
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


def _device() -> str:
    # MPS avoids the single-thread OpenMP limit set in app.py (needed for FAISS).
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# Per-model generation settings: short, fast, and no repetition.
_GEN_KWARGS: dict[str, dict[str, Any]] = {
    PEGASUS_MODEL: {"max_new_tokens": 64, "min_new_tokens": 12},
    T5_MODEL: {"max_new_tokens": 100, "min_new_tokens": 25},
}
_DEFAULT_GEN = {"max_new_tokens": 110, "min_new_tokens": 30}


@lru_cache(maxsize=4)
def _seq2seq(model_name: str) -> tuple[Any, Any, str]:
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model.eval()
    device = _device()
    try:
        model.to(device)
    except Exception:
        device = "cpu"
        model.to(device)
    return tokenizer, model, device


def preload_models(model_names: tuple[str, ...]) -> None:
    """Load models and run a tiny warm-up so the first user click is fast."""
    for name in model_names:
        try:
            abstractive_seq2seq("Warm up sentence for the model. " * 5, name)
        except Exception:
            pass


def abstractive_seq2seq(
    text: str,
    model_name: str,
    *,
    max_chars: int = 2000,
    prefix: str = "",
) -> str:
    if not text.strip():
        return ""
    tokenizer, model, device = _seq2seq(model_name)
    clipped = (prefix + text[:max_chars]).strip()
    inputs = tokenizer(clipped, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    gen = _GEN_KWARGS.get(model_name, _DEFAULT_GEN)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            **gen,
            num_beams=2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(output[0], skip_special_tokens=True).strip()


def abstractive_bart(text: str, model_name: str, max_chars: int = 3000) -> str:
    return abstractive_seq2seq(text, model_name, max_chars=max_chars)


def model_for_mode(mode: str, bart_model: str) -> str:
    if mode == "Abstractive T5":
        return T5_MODEL
    if mode == "Abstractive Pegasus":
        return PEGASUS_MODEL
    return bart_model


def summarize_traced(
    text: str,
    *,
    mode: str,
    model_name: str,
    sentence_count: int = 3,
) -> dict[str, Any]:
    """Return summary plus a detailed explanation of how it was produced."""
    sentences = split_sentences(text)
    tokens = preprocess(text[:4000], lemmatize=True)
    steps: list[dict[str, Any]] = [
        {
            "step": 1,
            "name": "Article intake",
            "detail": "Load the opened article body and measure basic length stats.",
            "outputs": {
                "chars": len(text),
                "sentences": len(sentences),
                "preview": text[:220],
            },
        },
        {
            "step": 2,
            "name": "Preprocessing",
            "detail": (
                "Tokenize with NLTK, drop English stopwords, and lemmatize with WordNet "
                "POS tags so later classical scorers see normalized terms."
            ),
            "outputs": {
                "token_count": len(tokens),
                "sample_tokens": tokens[:25],
            },
        },
    ]

    if mode == "Extractive TF-IDF":
        if not sentences:
            summary = ""
            ranked: list[dict[str, Any]] = []
        elif len(sentences) <= sentence_count:
            summary = " ".join(sentences)
            ranked = [{"rank": i + 1, "score": 1.0, "sentence": s} for i, s in enumerate(sentences)]
        else:
            vectorizer = TfidfVectorizer(stop_words="english")
            matrix = vectorizer.fit_transform(sentences)
            scores = np.asarray(matrix.sum(axis=1)).ravel()
            order = list(np.argsort(-scores))
            top = sorted(order[:sentence_count])
            summary = " ".join(sentences[i] for i in top)
            ranked = [
                {
                    "rank": rank,
                    "score": round(float(scores[i]), 4),
                    "sentence": sentences[i],
                }
                for rank, i in enumerate(order[: min(8, len(order))], start=1)
            ]
        steps.append(
            {
                "step": 3,
                "name": "Extractive TF-IDF scoring",
                "detail": (
                    "Build a sentence-level TF-IDF matrix, score each sentence by the sum "
                    "of its term weights, then keep the top sentences in original order."
                ),
                "outputs": {"selected": sentence_count, "ranked_sentences": ranked},
            }
        )
        steps.append(
            {
                "step": 4,
                "name": "Assemble summary",
                "detail": "Concatenate the selected sentences into a readable extractive summary.",
                "outputs": {"summary": summary},
            }
        )
        return {"summary": summary, "mode": mode, "pipeline": steps}

    if mode == "Extractive TextRank":
        summary = extractive_textrank(text, sentence_count=sentence_count)
        steps.append(
            {
                "step": 3,
                "name": "TextRank graph ranking",
                "detail": (
                    "Build a sentence similarity graph (via Sumy TextRank) and rank "
                    "sentences by centrality, similar to PageRank on text."
                ),
                "outputs": {"selected": sentence_count},
            }
        )
        steps.append(
            {
                "step": 4,
                "name": "Assemble summary",
                "detail": "Return the highest-ranked sentences as the extractive summary.",
                "outputs": {"summary": summary},
            }
        )
        return {"summary": summary, "mode": mode, "pipeline": steps}

    resolved = model_for_mode(mode, model_name)
    prefix = "summarize: " if mode == "Abstractive T5" else ""
    summary = abstractive_seq2seq(text, resolved, prefix=prefix)
    steps.append(
        {
            "step": 3,
            "name": f"Abstractive generation ({resolved})",
            "detail": (
                "A pretrained seq2seq transformer rewrites the article into new sentences "
                "(paraphrase), instead of copying source sentences."
            ),
            "inputs": {"model": resolved, "mode": mode},
            "outputs": {"summary": summary},
        }
    )
    steps.append(
        {
            "step": 4,
            "name": "Post-check",
            "detail": (
                "Return the generated abstractive summary for display and optional ROUGE eval."
            ),
            "outputs": {"summary_chars": len(summary)},
        }
    )
    return {"summary": summary, "mode": mode, "pipeline": steps, "model": resolved}
