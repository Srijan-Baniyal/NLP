"""Build a working-set hybrid index from local parquet corpora."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from newsmind.config import (
    CHUNK_OVERLAP_CHARS,
    CHUNK_SIZE_CHARS,
    DENSE_MODEL,
    INDEX_DIR,
    RAW_DIR,
    RERANK_MODEL,
    WORKING_SET_SIZE,
)
from newsmind.rag.chunking import chunk_article
from newsmind.rag.pipeline import build_rag

FULL_TEXT_CORPORA = ("cnn_dailymail", "xsum", "cc_news", "multi_news", "ag_news", "rss_live")


def load_working_articles(raw_dir: Path, limit: int) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    remaining = limit
    preferred = [
        "rss_live.parquet",
        "cnn_dailymail.parquet",
        "xsum.parquet",
        "multi_news.parquet",
        "cc_news.parquet",
        "ag_news.parquet",
        "huffpost_categories.parquet",
    ]
    paths = [raw_dir / name for name in preferred if (raw_dir / name).exists()]
    paths += [p for p in sorted(raw_dir.glob("*.parquet")) if p not in paths]
    for path in paths:
        if remaining <= 0:
            break
        pf = pq.ParquetFile(path)
        n = min(remaining, pf.metadata.num_rows)
        # Read in batches until we have enough rows with non-empty text.
        collected: list[pd.DataFrame] = []
        got = 0
        for batch in pf.iter_batches(batch_size=5_000, columns=None):
            frame = batch.to_pandas()
            if "text" not in frame.columns:
                continue
            frame = frame[frame["text"].astype(str).str.len() > 80]
            if frame.empty:
                continue
            take = frame.head(n - got)
            collected.append(take)
            got += len(take)
            if got >= n:
                break
        if collected:
            part = pd.concat(collected, ignore_index=True)
            frames.append(part)
            remaining -= len(part)
            print(f"Loaded {len(part):,} from {path.name}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).head(limit)


def articles_to_chunks(df: pd.DataFrame) -> tuple[list[str], list[dict[str, Any]]]:
    texts: list[str] = []
    metas: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        chunks = chunk_article(
            article_id=str(getattr(row, "article_id", "")),
            corpus=str(getattr(row, "corpus", "")),
            outlet=str(getattr(row, "outlet", "")),
            title=str(getattr(row, "title", "")),
            text=str(getattr(row, "text", "")),
            url=str(getattr(row, "url", "")),
            size=CHUNK_SIZE_CHARS,
            overlap=CHUNK_OVERLAP_CHARS,
        )
        for chunk in chunks:
            texts.append(chunk.text)
            metas.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "article_id": chunk.article_id,
                    "corpus": chunk.corpus,
                    "outlet": chunk.outlet,
                    "title": chunk.title,
                    "text": chunk.text,
                    "url": chunk.url,
                    "chunk_index": chunk.chunk_index,
                }
            )
    return texts, metas


def build_index(raw_dir: Path, index_dir: Path, limit: int) -> Path:
    index_dir.mkdir(parents=True, exist_ok=True)
    df = load_working_articles(raw_dir, limit)
    if df.empty:
        raise SystemExit(f"No articles found under {raw_dir}")
    texts, metas = articles_to_chunks(df)
    print(f"Indexing {len(texts):,} chunks from {len(df):,} articles…")
    rag = build_rag(texts, metas, dense_model=DENSE_MODEL, rerank_model=RERANK_MODEL)
    out = index_dir / "hybrid_rag.pkl"
    with out.open("wb") as fh:
        pickle.dump(
            {
                "texts": texts,
                "metas": metas,
                "dense_model": DENSE_MODEL,
                "rerank_model": RERANK_MODEL,
                "article_count": len(df),
                "chunk_count": len(texts),
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    # Persist a lightweight metadata sidecar for the UI.
    meta_path = index_dir / "index_meta.json"
    meta_path.write_text(
        json.dumps(
            {
                "article_count": len(df),
                "chunk_count": len(texts),
                "dense_model": DENSE_MODEL,
                "rerank_model": RERANK_MODEL,
                "corpora": sorted(df["corpus"].astype(str).unique().tolist()),
            },
            indent=2,
        )
    )
    # Keep the live RAG object in a second pickle for faster app load.
    live = index_dir / "rag_system.pkl"
    with live.open("wb") as fh:
        pickle.dump(rag, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Wrote {out} and {live}")
    return live


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build NewsMind hybrid RAG index.")
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument("--limit", type=int, default=WORKING_SET_SIZE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_index(args.raw_dir, args.index_dir, args.limit)


if __name__ == "__main__":
    main()
