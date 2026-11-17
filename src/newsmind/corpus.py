"""Corpus loading helpers for the Streamlit app."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from newsmind.config import RAW_DIR
from newsmind.schema import COLUMNS

META_COLUMNS = ["corpus", "outlet", "title", "summary", "url", "published_at", "topic", "split"]


def corpus_row_counts(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in sorted(raw_dir.glob("*.parquet")):
        meta = pq.ParquetFile(path).metadata
        rows.append({"file": path.name, "rows": meta.num_rows, "bytes": path.stat().st_size})
    return pd.DataFrame(rows)


def total_rows(raw_dir: Path = RAW_DIR) -> int:
    return int(corpus_row_counts(raw_dir)["rows"].sum()) if raw_dir.exists() else 0


def load_metadata(raw_dir: Path = RAW_DIR, max_rows_per_file: int | None = 200_000) -> pd.DataFrame:
    paths = sorted(raw_dir.glob("*.parquet"))
    if not paths:
        return pd.DataFrame(columns=META_COLUMNS)
    frames: list[pd.DataFrame] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in META_COLUMNS if c in available]
        if max_rows_per_file is None:
            frames.append(pd.read_parquet(path, columns=cols))
            continue
        # Cap huge headline dumps so the UI stays responsive.
        pf = pq.ParquetFile(path)
        if pf.metadata.num_rows <= max_rows_per_file:
            frames.append(pd.read_parquet(path, columns=cols))
        else:
            batch = next(pf.iter_batches(batch_size=max_rows_per_file, columns=cols))
            frames.append(batch.to_pandas())
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=META_COLUMNS)


def load_article_by_id(article_id: str, raw_dir: Path = RAW_DIR) -> pd.Series | None:
    for path in sorted(raw_dir.glob("*.parquet")):
        cols = list(COLUMNS)
        available = set(pq.ParquetFile(path).schema_arrow.names)
        use_cols = [c for c in cols if c in available]
        df = pd.read_parquet(path, columns=use_cols)
        hit = df[df["article_id"] == article_id]
        if not hit.empty:
            return hit.iloc[0]
    return None


def sample_full_text(raw_dir: Path = RAW_DIR, n: int = 20) -> pd.DataFrame:
    preferred = ["cnn_dailymail.parquet", "xsum.parquet", "cc_news.parquet", "rss_live.parquet"]
    for name in preferred:
        path = raw_dir / name
        if not path.exists():
            continue
        df = pd.read_parquet(path).head(n)
        if "text" in df.columns and df["text"].astype(str).str.len().mean() > 100:
            return df
    return pd.DataFrame(columns=list(COLUMNS))


FEED_FILES = (
    "outlet_archive.parquet",
    "rss_live.parquet",
    "cnn_dailymail.parquet",
    "xsum.parquet",
    "cc_news.parquet",
    "ag_news.parquet",
    "huffpost_categories.parquet",
)


def load_news_feed(
    raw_dir: Path = RAW_DIR,
    *,
    per_file: int = 80,
    min_text_len: int = 40,
) -> pd.DataFrame:
    """Load a readable mixed feed of full-ish articles for the UI."""
    frames: list[pd.DataFrame] = []
    for name in FEED_FILES:
        path = raw_dir / name
        if not path.exists():
            continue
        available = set(pq.ParquetFile(path).schema_arrow.names)
        cols = [c for c in COLUMNS if c in available]
        pf = pq.ParquetFile(path)
        batch = next(pf.iter_batches(batch_size=per_file, columns=cols))
        frame = batch.to_pandas()
        if "text" not in frame.columns:
            continue
        frame = frame[frame["text"].astype(str).str.len() >= min_text_len]
        if frame.empty:
            continue
        frames.append(frame.head(per_file))
    if not frames:
        return pd.DataFrame(columns=list(COLUMNS))
    feed = pd.concat(frames, ignore_index=True)
    # Prefer rows that look like real headlines.
    if "title" in feed.columns:
        feed = feed.sort_values(
            by=["title"],
            key=lambda s: s.astype(str).str.len(),
            ascending=False,
        )
    return feed.reset_index(drop=True)
