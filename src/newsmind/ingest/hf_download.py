"""Download Hugging Face news corpora and write standardized parquet files."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset

from newsmind.schema import COLUMNS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_DIR = PROJECT_ROOT / "data" / "raw"
CHUNK_SIZE = 10_000


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _take(dataset: Dataset, limit: int | None) -> Dataset:
    if limit is None or limit <= 0:
        return dataset
    return dataset.select(range(min(limit, len(dataset))))


def _write_dataset_chunked(dataset: Dataset, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    chunks: list[pd.DataFrame] = []
    for start in range(0, len(dataset), CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, len(dataset))
        batch = dataset.select(range(start, end)).to_pandas()
        chunks.append(batch[list(COLUMNS)])
    frame = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame(columns=list(COLUMNS))
    frame.to_parquet(path, index=False)
    return len(frame)


def ingest_cnn_dailymail(out_dir: Path, limit: int | None) -> Path:
    loaded = load_dataset("abisee/cnn_dailymail", "3.0.0")
    assert isinstance(loaded, DatasetDict)
    frames: list[pd.DataFrame] = []
    for split_key, split in loaded.items():
        split_name = str(split_key)
        split = _take(split, limit)

        def map_row(
            example: dict[str, Any],
            idx: int,
            *,
            _split: str = split_name,
        ) -> dict[str, str]:
            example_id = _as_str(example.get("id")) or str(idx)
            return {
                "article_id": _stable_id("cnn_dailymail", _split, example_id),
                "corpus": "cnn_dailymail",
                "outlet": "CNN/DailyMail",
                "title": "",
                "text": _as_str(example.get("article")),
                "summary": _as_str(example.get("highlights")),
                "url": "",
                "published_at": "",
                "split": _split,
                "topic": "",
            }

        mapped = split.map(map_row, with_indices=True, remove_columns=split.column_names)
        frames.append(mapped.to_pandas()[list(COLUMNS)])

    path = out_dir / "cnn_dailymail.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(COLUMNS))
    frame.to_parquet(path, index=False)
    print(f"Wrote {len(frame)} rows -> {path}")
    return path


def ingest_xsum(out_dir: Path, limit: int | None) -> Path:
    loaded = load_dataset("EdinburghNLP/xsum")
    assert isinstance(loaded, DatasetDict)
    frames: list[pd.DataFrame] = []
    for split_key, split in loaded.items():
        split_name = str(split_key)
        split = _take(split, limit)

        def map_row(
            example: dict[str, Any],
            idx: int,
            *,
            _split: str = split_name,
        ) -> dict[str, str]:
            example_id = _as_str(example.get("id")) or str(idx)
            return {
                "article_id": _stable_id("xsum", _split, example_id),
                "corpus": "xsum",
                "outlet": "BBC",
                "title": "",
                "text": _as_str(example.get("document")),
                "summary": _as_str(example.get("summary")),
                "url": "",
                "published_at": "",
                "split": _split,
                "topic": "",
            }

        mapped = split.map(map_row, with_indices=True, remove_columns=split.column_names)
        frames.append(mapped.to_pandas()[list(COLUMNS)])

    path = out_dir / "xsum.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(COLUMNS))
    frame.to_parquet(path, index=False)
    print(f"Wrote {len(frame)} rows -> {path}")
    return path


def ingest_cc_news(out_dir: Path, limit: int | None) -> Path:
    split = load_dataset("vblagoje/cc_news", split="train")
    assert isinstance(split, Dataset)
    split = _take(split, limit)

    def map_row(example: dict[str, Any], idx: int) -> dict[str, str]:
        title = _as_str(example.get("title"))
        url = _as_str(example.get("url"))
        return {
            "article_id": _stable_id("cc_news", url or title or str(idx)),
            "corpus": "cc_news",
            "outlet": _as_str(example.get("domain")) or "CC-News",
            "title": title,
            "text": _as_str(example.get("text")),
            "summary": _as_str(example.get("description")),
            "url": url,
            "published_at": _as_str(example.get("date")),
            "split": "train",
            "topic": "",
        }

    mapped = split.map(map_row, with_indices=True, remove_columns=split.column_names)
    path = out_dir / "cc_news.parquet"
    count = _write_dataset_chunked(mapped, path)
    print(f"Wrote {count} rows -> {path}")
    return path


def ingest_huffpost(out_dir: Path, limit: int | None) -> Path:
    split = load_dataset("heegyu/news-category-dataset", split="train")
    assert isinstance(split, Dataset)
    split = _take(split, limit)

    def map_row(example: dict[str, Any], idx: int) -> dict[str, str]:
        headline = _as_str(example.get("headline"))
        short = _as_str(example.get("short_description"))
        link = _as_str(example.get("link"))
        return {
            "article_id": _stable_id("huffpost", link or headline or str(idx)),
            "corpus": "huffpost_categories",
            "outlet": _as_str(example.get("authors")) or "HuffPost",
            "title": headline,
            "text": short,
            "summary": short,
            "url": link,
            "published_at": _as_str(example.get("date")),
            "split": "train",
            "topic": _as_str(example.get("category")),
        }

    mapped = split.map(map_row, with_indices=True, remove_columns=split.column_names)
    path = out_dir / "huffpost_categories.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = mapped.to_pandas()[list(COLUMNS)]
    frame.to_parquet(path, index=False)
    print(f"Wrote {len(frame)} rows -> {path}")
    return path


CORPUS_LOADERS = {
    "cnn_dailymail": ingest_cnn_dailymail,
    "xsum": ingest_xsum,
    "cc_news": ingest_cc_news,
    "huffpost": ingest_huffpost,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Hugging Face news corpora to parquet.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for parquet outputs (default: data/raw).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional per-split row cap for quick smoke tests.",
    )
    parser.add_argument(
        "--corpora",
        nargs="+",
        choices=[*CORPUS_LOADERS.keys(), "all"],
        default=["all"],
        help="Which corpora to download.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected = list(CORPUS_LOADERS.keys()) if "all" in args.corpora else args.corpora
    for name in selected:
        CORPUS_LOADERS[name](args.out_dir, args.limit)


if __name__ == "__main__":
    main()
