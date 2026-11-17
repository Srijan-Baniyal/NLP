"""NewsMind: retrieval-augmented summarization and QA for news."""

from __future__ import annotations


def main() -> None:
    print("NewsMind commands:")
    print("  uv run newsmind-hf          # CNN/DM, XSum, CC-News, HuffPost")
    print("  uv run newsmind-rss         # live RSS headlines / feed bodies")
    print("  uv run newsmind-scale       # grow corpus toward 10M rows")
    print("  uv run newsmind-outlets     # named papers from CC-News (no scrape)")
    print("  uv run newsmind-index       # build hybrid RAG index")
    print("  uv run streamlit run src/newsmind/app.py")


if __name__ == "__main__":
    main()
