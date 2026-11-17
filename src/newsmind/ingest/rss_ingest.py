"""Ingest official RSS headlines and blurbs into a parquet snapshot."""

from __future__ import annotations

import argparse
import hashlib
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any

import feedparser
import pandas as pd

from newsmind.ingest.feeds import FEEDS, Feed
from newsmind.schema import COLUMNS, ArticleRow

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw"
_TAG_RE = re.compile(r"<[^>]+>")


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def _strip_html(value: str) -> str:
    return _TAG_RE.sub(" ", value).replace("\xa0", " ").strip()


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _entry_published(entry: feedparser.FeedParserDict) -> str:
    for key in ("published", "updated", "created"):
        raw = entry.get(key)
        if not raw:
            continue
        try:
            return parsedate_to_datetime(raw).astimezone(UTC).isoformat()
        except (TypeError, ValueError, OverflowError, IndexError):
            return _as_str(raw)
    return ""


def _entry_text(entry: feedparser.FeedParserDict) -> str:
    """Prefer the longest publisher-provided feed body (never fetch article HTML)."""
    candidates: list[str] = []
    summary = _as_str(entry.get("summary"))
    if summary:
        candidates.append(_strip_html(summary))
    content = entry.get("content")
    if isinstance(content, list):
        for block in content:
            value = _strip_html(_as_str(block.get("value")))
            if value:
                candidates.append(value)
    # Some feeds expose content:encoded via feedparser as content list already;
    # also check common alternate keys.
    for key in ("content_encoded", "description"):
        extra = _strip_html(_as_str(entry.get(key)))
        if extra:
            candidates.append(extra)
    if not candidates:
        return ""
    return max(candidates, key=len)


def fetch_feed(feed: Feed) -> list[ArticleRow]:
    parsed = feedparser.parse(feed.url)
    if getattr(parsed, "bozo", False) and not parsed.entries:
        print(f"Skip dead feed {feed.outlet}/{feed.section}: {feed.url}")
        return []

    rows: list[ArticleRow] = []
    for entry in parsed.entries:
        title = _strip_html(_as_str(entry.get("title")))
        link = _as_str(entry.get("link"))
        text = _entry_text(entry)
        if not title and not text:
            continue
        rows.append(
            ArticleRow(
                article_id=_stable_id("rss", feed.outlet, feed.section, link or title),
                corpus="rss_live",
                outlet=feed.outlet,
                title=title,
                text=text,
                summary=text,
                url=link,
                published_at=_entry_published(entry),
                split="live",
                topic=feed.section,
            )
        )
    print(f"{feed.outlet}/{feed.section}: {len(rows)} items")
    return rows


def ingest_rss(out_dir: Path) -> Path:
    rows: list[ArticleRow] = []
    seen: set[str] = set()
    for feed in FEEDS:
        for row in fetch_feed(feed):
            if row.article_id in seen:
                continue
            seen.add(row.article_id)
            rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "rss_live.parquet"
    frame = pd.DataFrame([row.to_dict() for row in rows], columns=list(COLUMNS))
    frame.to_parquet(path, index=False)
    stamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    print(f"Wrote {len(frame)} RSS rows ({stamp}) -> {path}")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest official RSS feeds to parquet.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for parquet outputs (default: data/raw).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ingest_rss(args.out_dir)


if __name__ == "__main__":
    main()
