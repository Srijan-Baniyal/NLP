"""Project-wide paths and model defaults."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"
CACHE_DIR = DATA_DIR / "cache"

TARGET_CORPUS_ROWS = 10_000_000
WORKING_SET_SIZE = 100_000
CHUNK_SIZE_CHARS = 900
CHUNK_OVERLAP_CHARS = 120

DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
T5_MODEL = "t5-small"
PEGASUS_MODEL = "google/pegasus-xsum"
GENERATOR_MODEL = "google/flan-t5-base"
GLOVE_MODEL = "glove-wiki-gigaword-100"

HYBRID_ALPHA = 0.55  # weight on dense scores after min-max norm
RETRIEVE_TOP_K = 20
RERANK_TOP_K = 5
