"""Shared article schema for every NewsMind corpus source."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

COLUMNS: tuple[str, ...] = (
    "article_id",
    "corpus",
    "outlet",
    "title",
    "text",
    "summary",
    "url",
    "published_at",
    "split",
    "topic",
)


@dataclass(slots=True)
class ArticleRow:
    article_id: str
    corpus: str
    outlet: str
    title: str
    text: str
    summary: str
    url: str
    published_at: str
    split: str
    topic: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
