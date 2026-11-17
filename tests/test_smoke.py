"""Smoke tests for preprocessing and evaluation helpers."""

from __future__ import annotations

from newsmind.eval import exact_match, rouge_scores, token_f1
from newsmind.preprocess import preprocess, tokenize
from newsmind.rag.chunking import chunk_article
from newsmind.summarize import extractive_tfidf


def test_tokenize_and_preprocess() -> None:
    tokens = tokenize("The elections in India were closely watched.")
    assert "elections" in tokens
    lemmas = preprocess("The cats are running quickly.")
    assert "cat" in lemmas or "cats" in lemmas


def test_chunking() -> None:
    text = " ".join([f"Sentence number {i} about world news." for i in range(40)])
    chunks = chunk_article(
        article_id="a1",
        corpus="test",
        outlet="Test",
        title="T",
        text=text,
        url="",
        size=120,
        overlap=20,
    )
    assert len(chunks) >= 2
    assert chunks[0].chunk_id


def test_extractive_summary() -> None:
    text = (
        "Markets rose sharply on Monday after strong earnings. "
        "Investors cheered the results across technology stocks. "
        "Analysts said the rally may continue into next week. "
        "Weather in coastal cities remained mild overnight."
    )
    summary = extractive_tfidf(text, sentence_count=2)
    assert len(summary.split(".")) >= 1
    assert "Markets" in summary or "Investors" in summary or "Analysts" in summary


def test_eval_metrics() -> None:
    assert exact_match("Hello World", "hello world") == 1.0
    assert token_f1("the cat sat", "the cat") > 0.5
    scores = rouge_scores("the cat sat on the mat", "the cat sat on a mat")
    assert scores["rouge1"] > 0.5


def test_syntax_noun_phrases() -> None:
    from newsmind.preprocess.syntax import analyze_syntax

    analysis = analyze_syntax("The Indian Express reported election results in Delhi.")
    assert analysis.tokens
    assert analysis.pos_tags
    joined = " ".join(analysis.noun_phrases).lower()
    assert "election" in joined or "delhi" in joined or "express" in joined or "result" in joined
