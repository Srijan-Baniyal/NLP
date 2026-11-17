"""Sentence-aware chunking for RAG passages."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    article_id: str
    corpus: str
    outlet: str
    title: str
    text: str
    url: str
    chunk_index: int


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [part.strip() for part in parts if part.strip()]


def chunk_article(
    *,
    article_id: str,
    corpus: str,
    outlet: str,
    title: str,
    text: str,
    url: str,
    size: int = 900,
    overlap: int = 120,
) -> list[Chunk]:
    body = text.strip()
    if not body:
        return []
    sentences = split_sentences(body)
    if not sentences:
        sentences = [body]

    chunks: list[Chunk] = []
    buf: list[str] = []
    buf_len = 0
    index = 0

    def flush() -> None:
        nonlocal buf, buf_len, index
        if not buf:
            return
        piece = " ".join(buf).strip()
        chunks.append(
            Chunk(
                chunk_id=_stable_id(article_id, str(index), piece[:64]),
                article_id=article_id,
                corpus=corpus,
                outlet=outlet,
                title=title,
                text=piece,
                url=url,
                chunk_index=index,
            )
        )
        index += 1
        if overlap > 0 and piece:
            keep = piece[-overlap:]
            buf = [keep]
            buf_len = len(keep)
        else:
            buf = []
            buf_len = 0

    for sent in sentences:
        sent_len = len(sent) + (1 if buf else 0)
        if buf and buf_len + sent_len > size:
            flush()
        buf.append(sent)
        buf_len += sent_len
        if buf_len >= size:
            flush()
    flush()
    return chunks
