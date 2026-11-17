"""Stream large public news sets until the local corpus reaches 10M rows."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset

from newsmind.config import RAW_DIR, TARGET_CORPUS_ROWS
from newsmind.schema import COLUMNS

BATCH_ROWS = 25_000


def _stable_id(*parts: str) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()  # noqa: S324
    return digest[:16]


def _as_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def count_existing_rows(raw_dir: Path) -> int:
    total = 0
    for path in raw_dir.glob("*.parquet"):
        try:
            total += pq.ParquetFile(path).metadata.num_rows
        except Exception:  # noqa: BLE001
            continue
    return total


def _flush(
    rows: list[dict[str, str]],
    writer: pq.ParquetWriter | None,
    path: Path,
) -> pq.ParquetWriter:
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    table = pa.Table.from_pandas(frame, preserve_index=False)
    if writer is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(path, table.schema, compression="zstd")
    writer.write_table(table)
    return writer


def stream_gdelt(out_path: Path, needed: int) -> int:
    """GDELT news headlines (~18M). Stream until `needed` rows are written."""
    if needed <= 0:
        return 0
    print(f"Streaming olm/gdelt-news-headlines for {needed:,} more rows…", flush=True)
    ds = load_dataset("olm/gdelt-news-headlines", split="train", streaming=True)
    rows: list[dict[str, str]] = []
    written = 0
    writer: pq.ParquetWriter | None = None
    try:
        for example in ds:
            title = _as_str(example.get("title"))
            url = _as_str(example.get("url"))
            outlet = _as_str(example.get("outlet_name")) or _as_str(example.get("domain"))
            published = _as_str(example.get("article_date"))
            if not title and not url:
                continue
            rows.append(
                {
                    "article_id": _stable_id("gdelt", url or title),
                    "corpus": "gdelt_headlines",
                    "outlet": outlet or "GDELT",
                    "title": title,
                    "text": title,
                    "summary": title,
                    "url": url,
                    "published_at": published,
                    "split": "train",
                    "topic": "",
                }
            )
            if len(rows) >= BATCH_ROWS:
                writer = _flush(rows, writer, out_path)
                written += len(rows)
                print(f"  gdelt: {written:,}/{needed:,}", flush=True)
                rows = []
                if written >= needed:
                    break
        if rows and written < needed:
            take = rows[: needed - written]
            writer = _flush(take, writer, out_path)
            written += len(take)
    finally:
        if writer is not None:
            writer.close()
    print(f"Wrote {written:,} GDELT rows -> {out_path}", flush=True)
    return written


def stream_ag_news(out_path: Path) -> int:
    loaded = load_dataset("fancyzhx/ag_news")
    label_names = ["World", "Sports", "Business", "Sci/Tech"]
    rows: list[dict[str, str]] = []
    for split_name, split in loaded.items():
        for idx, example in enumerate(split):
            text = _as_str(example.get("text"))
            label = int(example.get("label", 0))
            topic = label_names[label] if 0 <= label < len(label_names) else str(label)
            rows.append(
                {
                    "article_id": _stable_id("ag_news", str(split_name), str(idx)),
                    "corpus": "ag_news",
                    "outlet": "AG News",
                    "title": text[:120],
                    "text": text,
                    "summary": "",
                    "url": "",
                    "published_at": "",
                    "split": str(split_name),
                    "topic": topic,
                }
            )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(f"Wrote {len(frame):,} AG News rows -> {out_path}", flush=True)
    return len(frame)


def stream_multi_news(out_path: Path) -> int:
    loaded = load_dataset("alexfabbri/multi_news")
    rows: list[dict[str, str]] = []
    for split_name, split in loaded.items():
        for idx, example in enumerate(split):
            document = _as_str(example.get("document"))
            summary = _as_str(example.get("summary"))
            rows.append(
                {
                    "article_id": _stable_id("multi_news", str(split_name), str(idx)),
                    "corpus": "multi_news",
                    "outlet": "Multi-News",
                    "title": summary[:120],
                    "text": document,
                    "summary": summary,
                    "url": "",
                    "published_at": "",
                    "split": str(split_name),
                    "topic": "",
                }
            )
    frame = pd.DataFrame(rows, columns=list(COLUMNS))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_path, index=False)
    print(f"Wrote {len(frame):,} Multi-News rows -> {out_path}", flush=True)
    return len(frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale NewsMind corpus toward 10M rows.")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--target", type=int, default=TARGET_CORPUS_ROWS)
    parser.add_argument(
        "--skip-extras",
        action="store_true",
        help="Only stream GDELT filler; skip AG News / Multi-News.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        reconfigure = getattr(sys.stdout, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    except Exception:  # noqa: BLE001
        pass

    if not args.skip_extras:
        if not (out_dir / "ag_news.parquet").exists():
            try:
                stream_ag_news(out_dir / "ag_news.parquet")
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping AG News ({exc})", flush=True)
        if not (out_dir / "multi_news.parquet").exists():
            try:
                stream_multi_news(out_dir / "multi_news.parquet")
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping Multi-News ({exc})", flush=True)

    gdelt_path = out_dir / "gdelt_headlines.parquet"
    existing = 0
    for path in out_dir.glob("*.parquet"):
        if path.name == gdelt_path.name:
            continue
        try:
            existing += pq.ParquetFile(path).metadata.num_rows
        except Exception:  # noqa: BLE001
            continue
    print(f"Existing non-GDELT rows: {existing:,}", flush=True)
    needed = max(0, args.target - existing)

    if gdelt_path.exists():
        try:
            current_gdelt = pq.ParquetFile(gdelt_path).metadata.num_rows
            print(f"Existing GDELT rows: {current_gdelt:,}", flush=True)
            if existing + current_gdelt >= args.target:
                print(
                    f"Already at/above target ({existing + current_gdelt:,} >= {args.target:,}).",
                    flush=True,
                )
                return
            print("Removing partial GDELT file to re-stream to target.", flush=True)
            gdelt_path.unlink()
            needed = max(0, args.target - existing)
        except Exception:  # noqa: BLE001
            print("Removing incomplete GDELT parquet.", flush=True)
            gdelt_path.unlink(missing_ok=True)
            needed = max(0, args.target - existing)

    if needed == 0:
        print(f"Already at/above target ({existing:,} >= {args.target:,}).", flush=True)
        return

    written = stream_gdelt(gdelt_path, needed)
    final = count_existing_rows(out_dir)
    print(f"Done. Added {written:,}. Total rows now: {final:,}", flush=True)


if __name__ == "__main__":
    main()
