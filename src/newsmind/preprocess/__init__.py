"""Tokenization, stopword removal, and lemmatization."""

from __future__ import annotations

import re
from functools import lru_cache

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?", re.IGNORECASE)


_NLTK_RESOURCES: tuple[tuple[str, str], ...] = (
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
)


@lru_cache(maxsize=1)
def ensure_nltk() -> None:
    """Download missing NLTK data once per process (cached; this is called per token)."""
    for path, package in _NLTK_RESOURCES:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


@lru_cache(maxsize=1)
def _stopwords() -> set[str]:
    ensure_nltk()
    return set(stopwords.words("english"))


@lru_cache(maxsize=1)
def _lemmatizer() -> WordNetLemmatizer:
    ensure_nltk()
    return WordNetLemmatizer()


def _wn_pos(tag: str) -> str:
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def tokenize(text: str) -> list[str]:
    ensure_nltk()
    return [tok.lower() for tok in word_tokenize(text) if _TOKEN_RE.fullmatch(tok)]


def preprocess(text: str, *, lemmatize: bool = True) -> list[str]:
    tokens = tokenize(text)
    stops = _stopwords()
    tokens = [tok for tok in tokens if tok not in stops and len(tok) > 1]
    if not lemmatize or not tokens:
        return tokens
    tagged = pos_tag(tokens)
    lemma = _lemmatizer()
    return [lemma.lemmatize(tok, _wn_pos(tag)) for tok, tag in tagged]


def preprocess_join(text: str, *, lemmatize: bool = True) -> str:
    return " ".join(preprocess(text, lemmatize=lemmatize))
