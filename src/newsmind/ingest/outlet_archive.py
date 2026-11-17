"""Build a legal archive of named newspapers from already-licensed CC-News.

This does NOT scrape NYT/WSJ/etc. paywalled pages. It only filters publicly
distributed Common Crawl news extracts already stored in cc_news.parquet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from newsmind.config import RAW_DIR
from newsmind.schema import COLUMNS

# Domain fragments → display outlet name
OUTLET_DOMAINS: dict[str, str] = {
    "nytimes.com": "New York Times",
    "wsj.com": "Wall Street Journal",
    "washingtontimes.com": "Washington Times",
    "hindustantimes.com": "Hindustan Times",
    "timesofindia.indiatimes.com": "Times of India",
    "indiatimes.com": "Times of India",
    "indianexpress.com": "Indian Express",
    "cnn.com": "CNN",
    "bbc.co.uk": "BBC",
    "bbc.com": "BBC",
    "theguardian.com": "The Guardian",
    "thehindu.com": "The Hindu",
}


def _match_outlet(domain: str, url: str) -> str | None:
    hay = f"{domain} {url}".lower()
    for needle, name in OUTLET_DOMAINS.items():
        if needle in hay:
            return name
    return None


def extract_named_outlets(cc_path: Path, out_path: Path, limit_per_outlet: int = 5_000) -> Path:
    if not cc_path.exists():
        raise SystemExit(f"Missing {cc_path}. Run `uv run newsmind-hf --corpora cc_news` first.")

    cols = [c for c in COLUMNS if c in set(pq.ParquetFile(cc_path).schema_arrow.names)]
    # Also need domain if present in raw cc_news before remap — our parquet uses outlet.
    pf = pq.ParquetFile(cc_path)
    buckets: dict[str, list[dict]] = {name: [] for name in set(OUTLET_DOMAINS.values())}
    filled = {name: 0 for name in buckets}

    for batch in pf.iter_batches(batch_size=10_000, columns=cols):
        frame = batch.to_pandas()
        if all(v >= limit_per_outlet for v in filled.values()):
            break
        for row in frame.itertuples(index=False):
            outlet_raw = str(getattr(row, "outlet", "") or "")
            url = str(getattr(row, "url", "") or "")
            matched = _match_outlet(outlet_raw, url)
            if matched is None or filled[matched] >= limit_per_outlet:
                continue
            text = str(getattr(row, "text", "") or "")
            if len(text) < 120:
                continue
            buckets[matched].append(
                {
                    "article_id": str(getattr(row, "article_id", "")),
                    "corpus": "outlet_archive",
                    "outlet": matched,
                    "title": str(getattr(row, "title", "") or ""),
                    "text": text,
                    "summary": str(getattr(row, "summary", "") or ""),
                    "url": url,
                    "published_at": str(getattr(row, "published_at", "") or ""),
                    "split": "archive",
                    "topic": "",
                }
            )
            filled[matched] += 1

    rows = [row for items in buckets.values() for row in items]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=list(COLUMNS)).to_parquet(out_path, index=False)
    print(f"Wrote {len(rows):,} outlet-archive rows -> {out_path}")
    for name, count in sorted(filled.items(), key=lambda x: -x[1]):
        if count:
            print(f"  {name}: {count:,}")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter CC-News for named newspapers (no live scraping)."
    )
    parser.add_argument("--cc-path", type=Path, default=RAW_DIR / "cc_news.parquet")
    parser.add_argument("--out", type=Path, default=RAW_DIR / "outlet_archive.parquet")
    parser.add_argument("--limit-per-outlet", type=int, default=5_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extract_named_outlets(args.cc_path, args.out, args.limit_per_outlet)


if __name__ == "__main__":
    main()
