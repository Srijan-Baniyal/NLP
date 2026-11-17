"""NewsMind Streamlit demo: read news, open article, summarize, ask, compare, explain."""

from __future__ import annotations

import os

# FAISS and PyTorch each bundle an OpenMP runtime; on macOS they deadlock or abort
# when both run multi-threaded. Must be set before either library is imported.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from newsmind.config import (
    INDEX_DIR,
    RAW_DIR,
    SUMMARIZER_MODEL,
    TARGET_CORPUS_ROWS,
)
from newsmind.corpus import corpus_row_counts, load_news_feed, sample_full_text
from newsmind.eval import exact_match, rouge_scores, token_f1
from newsmind.eval.compare import compare_retrievers, compare_summarizers
from newsmind.preprocess.syntax import analyze_syntax
from newsmind.summarize import SUMMARIZER_MODES, summarize_traced
from newsmind.topics import fit_lda


@st.cache_resource
def _start_preload() -> bool:
    """Warm all abstractive models in the background once per server process."""
    import threading

    from newsmind.config import PEGASUS_MODEL, T5_MODEL
    from newsmind.summarize import preload_models

    threading.Thread(
        target=preload_models,
        args=((T5_MODEL, SUMMARIZER_MODEL, PEGASUS_MODEL),),
        daemon=True,
    ).start()
    return True


_start_preload()


@st.cache_data(show_spinner="Loading corpus stats…")
def cached_counts(raw_dir: str) -> pd.DataFrame:
    return corpus_row_counts(Path(raw_dir))


@st.cache_data(show_spinner="Loading news feed…")
def cached_feed(raw_dir: str, per_file: int) -> pd.DataFrame:
    return load_news_feed(Path(raw_dir), per_file=per_file)


@st.cache_resource(show_spinner="Loading RAG system…")
def load_rag(index_path: str):
    path = Path(index_path)
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


def _headline(row: pd.Series) -> str:
    title = str(row.get("title") or "").strip()
    if title:
        return title
    text = str(row.get("text") or "").strip()
    return text[:110] + ("…" if len(text) > 110 else "")


def _blurb(row: pd.Series, n: int = 180) -> str:
    summary = str(row.get("summary") or "").strip()
    if summary:
        return summary[:n] + ("…" if len(summary) > n else "")
    text = str(row.get("text") or "").strip()
    return text[:n] + ("…" if len(text) > n else "")


def render_pipeline(steps: list[dict[str, Any]], title: str = "How NewsMind produced this") -> None:
    st.markdown(f"### {title}")
    st.caption("Step-by-step breakdown of classical NLP and modern RAG stages.")
    for step in steps:
        expanded = step.get("step") in {1, 6}
        label = f"Step {step.get('step')}: {step.get('name')}"
        with st.expander(label, expanded=expanded):
            st.write(step.get("detail", ""))
            if step.get("inputs"):
                st.markdown("**Inputs**")
                st.json(step["inputs"])
            outputs = step.get("outputs")
            if not outputs:
                continue
            st.markdown("**Outputs**")
            hits = outputs.get("hits") if isinstance(outputs, dict) else None
            if isinstance(hits, list):
                st.dataframe(pd.DataFrame(hits), width="stretch", hide_index=True)
                leftover = {k: v for k, v in outputs.items() if k != "hits"}
                if leftover:
                    st.json(leftover)
            elif isinstance(outputs, dict) and "ranked_sentences" in outputs:
                ranked = outputs.get("ranked_sentences") or []
                if ranked:
                    st.dataframe(pd.DataFrame(ranked), width="stretch", hide_index=True)
                leftover = {k: v for k, v in outputs.items() if k != "ranked_sentences"}
                if leftover:
                    st.json(leftover)
            else:
                st.json(outputs)


def page_overview() -> None:
    st.header("Corpus overview")
    if not RAW_DIR.exists():
        st.warning("No `data/raw` yet. Run ingest commands first.")
        return
    counts = cached_counts(str(RAW_DIR))
    total = int(counts["rows"].sum()) if not counts.empty else 0
    c1, c2, c3 = st.columns(3)
    c1.metric("Total rows", f"{total:,}")
    c2.metric("Target", f"{TARGET_CORPUS_ROWS:,}")
    c3.metric("Coverage", f"{min(100.0, 100 * total / TARGET_CORPUS_ROWS):.1f}%")
    st.dataframe(counts, width="stretch", hide_index=True)
    st.info(
        "Named newspapers (NYT, WSJ, HT, TOI, Indian Express, …) come from "
        "**CC-News research extracts** and **official RSS headlines/blurbs** — "
        "NewsMind does not scrape paywalled article HTML."
    )


def page_read_news() -> None:
    st.header("Read the news")
    st.caption("Browse stories, open one, then summarize or ask questions about it.")

    if "selected_idx" not in st.session_state:
        st.session_state.selected_idx = None
    if "summary_result" not in st.session_state:
        st.session_state.summary_result = None
    if "qa_result" not in st.session_state:
        st.session_state.qa_result = None

    per_file = st.sidebar.slider("Stories per source", 20, 150, 60, key="feed_per_file")
    feed = cached_feed(str(RAW_DIR), per_file)
    if feed.empty:
        st.info("No readable articles found under data/raw.")
        return

    corpora = sorted(feed["corpus"].dropna().astype(str).unique().tolist())
    outlets = sorted(feed["outlet"].dropna().astype(str).unique().tolist())
    preferred = [
        "New York Times",
        "Wall Street Journal",
        "Washington Times",
        "Hindustan Times",
        "Times of India",
        "Indian Express",
        "CNN",
        "BBC",
        "The Guardian",
        "The Hindu",
        "CNN/DailyMail",
    ]
    default_outlets = [o for o in preferred if o in outlets] or outlets[: min(25, len(outlets))]
    selected_corpora = st.multiselect("Corpus", corpora, default=corpora)
    selected_outlets = st.multiselect("Outlet", outlets, default=default_outlets)
    query = st.text_input("Search titles / blurbs", "")

    filtered = feed[
        feed["corpus"].astype(str).isin(selected_corpora)
        & feed["outlet"].astype(str).isin(selected_outlets)
    ].copy()
    if query.strip():
        q = query.strip().lower()
        mask = (
            filtered["title"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["summary"].astype(str).str.lower().str.contains(q, na=False)
            | filtered["text"].astype(str).str.lower().str.contains(q, na=False)
        )
        filtered = filtered[mask]

    st.metric("Stories shown", f"{len(filtered):,}")

    if st.session_state.selected_idx is not None:
        idx = int(st.session_state.selected_idx)
        if idx < 0 or idx >= len(feed):
            st.session_state.selected_idx = None
        else:
            row = feed.iloc[idx]
            if st.button("← Back to all news", key="back_feed"):
                st.session_state.selected_idx = None
                st.session_state.summary_result = None
                st.session_state.qa_result = None
                st.rerun()
            _render_article(row)
            return

    if filtered.empty:
        st.warning("No stories match these filters.")
        return

    show_n = st.slider("How many to list", 5, min(80, len(filtered)), min(20, len(filtered)))
    listing = filtered.head(show_n)
    for _, row in listing.iterrows():
        abs_matches = feed.index[(feed["article_id"] == row["article_id"])].tolist()
        abs_idx = int(abs_matches[0]) if abs_matches else int(row.name)
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                st.markdown(f"**{_headline(row)}**")
                published = str(row.get("published_at") or "")
                caption = f"{row.get('outlet', '')} · {row.get('corpus', '')}"
                if published:
                    caption += f" · {published}"
                st.caption(caption)
                st.write(_blurb(row))
            with right:
                if st.button("Open", key=f"open_{abs_idx}"):
                    st.session_state.selected_idx = abs_idx
                    st.session_state.summary_result = None
                    st.session_state.qa_result = None
                    st.rerun()


def _render_article(row: pd.Series) -> None:
    title = _headline(row)
    st.subheader(title)
    meta_bits = [
        str(row.get("outlet") or ""),
        str(row.get("corpus") or ""),
        str(row.get("topic") or ""),
        str(row.get("published_at") or ""),
    ]
    st.caption(" · ".join(bit for bit in meta_bits if bit and bit != "nan"))
    url = str(row.get("url") or "").strip()
    if url:
        st.markdown(f"[Original link]({url})")

    body = str(row.get("text") or "")
    reference = str(row.get("summary") or "").strip()
    st.markdown("#### Full article")
    st.write(body)

    with st.expander("Syntactic analysis (POS + noun phrases)"):
        syntax = analyze_syntax(body[:2000])
        st.write("Noun phrases:", ", ".join(syntax.noun_phrases[:20]) or "(none)")
        st.dataframe(
            pd.DataFrame(syntax.pos_tags[:40], columns=["token", "POS"]),
            width="stretch",
            hide_index=True,
        )

    sum_tab, ask_tab = st.tabs(["Summarization panel", "Question asking panel"])

    with sum_tab:
        mode = st.selectbox("Summarizer", list(SUMMARIZER_MODES), key="sum_mode")
        if st.button("Generate summary", type="primary", key="do_summary"):
            with st.spinner("Running summarization pipeline…"):
                result = summarize_traced(body, mode=mode, model_name=SUMMARIZER_MODEL)
                if reference:
                    result["rouge"] = rouge_scores(str(result["summary"]), reference)
                st.session_state.summary_result = result

        result = st.session_state.summary_result
        if result:
            st.markdown("#### Summary")
            st.success(result["summary"])
            if result.get("rouge"):
                st.markdown("#### ROUGE vs reference highlights")
                st.json(result["rouge"])
            render_pipeline(result["pipeline"], title="How the summary was obtained")

    with ask_tab:
        rag = load_rag(str(INDEX_DIR / "rag_system.pkl"))
        if rag is None:
            st.warning("No RAG index yet. Run `uv run newsmind-index --limit 5000`.")
        else:
            question = st.text_input(
                "Ask a question about this story (or the wider corpus)",
                "What are the key facts in this article?",
                key="article_question",
            )
            use_article = st.checkbox("Answer only from this article", value=True)
            use_rerank = st.checkbox("Use cross-encoder rerank", value=True)
            if st.button("Ask NewsMind", type="primary", key="do_ask") and question.strip():
                with st.spinner("Running hybrid RAG pipeline…"):
                    if use_article:
                        qa = rag.ask_article(
                            question,
                            {
                                "article_id": row.get("article_id"),
                                "corpus": row.get("corpus"),
                                "outlet": row.get("outlet"),
                                "title": title,
                                "text": body,
                                "url": url,
                            },
                        )
                    else:
                        qa = rag.ask_traced(question, use_rerank=use_rerank)
                    st.session_state.qa_result = qa

            qa = st.session_state.qa_result
            if qa:
                st.markdown("#### Answer")
                st.info(str(qa.get("answer", "")))
                st.markdown("#### Citations")
                cites = qa.get("citations") or []
                if cites:
                    st.dataframe(pd.DataFrame(cites), width="stretch", hide_index=True)
                render_pipeline(
                    list(qa.get("pipeline") or []),
                    title="How the answer was obtained",
                )
                with st.expander("Raw grounded context sent to the generator"):
                    st.text(str(qa.get("context", "")))
                with st.expander("Full pipeline JSON"):
                    st.code(json.dumps(qa.get("pipeline"), indent=2)[:12000])


def page_compare() -> None:
    st.header("Comparative NLP lab")
    st.caption(
        "Side-by-side classical vs embedding retrieval, and extractive vs abstractive "
        "summarization — matching the synopsis evaluation plan."
    )
    sample = sample_full_text(RAW_DIR, n=40)
    if sample.empty:
        st.info("Need full-text articles first.")
        return

    texts = sample["text"].astype(str).tolist()
    metas = sample.to_dict(orient="records")
    query = st.text_input("Retrieval query", "election results and economy")
    top_k = st.slider("Top-k", 3, 10, 5)

    if st.button("Compare retrievers", type="primary"):
        with st.spinner("TF-IDF / BM25 / Word2Vec / GloVe…"):
            result = compare_retrievers(query, texts, metas, top_k=top_k)
        st.subheader("Syntax for the query")
        st.json(result["syntax"])
        st.subheader("Leaderboard")
        st.dataframe(pd.DataFrame(result["table"]), width="stretch", hide_index=True)
        if result.get("glove_error"):
            st.warning(f"GloVe unavailable: {result['glove_error']}")
        for name, hits in result["hits"].items():
            with st.expander(f"{name} hits"):
                st.dataframe(pd.DataFrame(hits), width="stretch", hide_index=True)

    st.divider()
    article_labels = [f"{i}: {_headline(sample.iloc[i])[:70]}" for i in range(min(15, len(sample)))]
    choice = st.selectbox("Article for summarizer bake-off", article_labels)
    idx = int(choice.split(":", 1)[0])
    row = sample.iloc[idx]
    body = str(row["text"])
    reference = str(row.get("summary") or "")
    modes = st.multiselect(
        "Summarizers to run",
        list(SUMMARIZER_MODES),
        default=["Extractive TF-IDF", "Extractive TextRank", "Abstractive BART"],
    )
    if st.button("Compare summarizers") and modes:
        with st.spinner("Running selected summarizers (abstractive models download on first use)…"):
            rows = compare_summarizers(
                body,
                reference,
                modes=tuple(modes),
                bart_model=SUMMARIZER_MODEL,
            )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


def page_topics() -> None:
    st.header("LDA topic explorer")
    sample = sample_full_text(RAW_DIR, n=int(st.slider("Articles", 50, 500, 200)))
    if sample.empty:
        st.info("Need full-text articles for topic modelling.")
        return
    n_topics = st.slider("Topics", 3, 15, 8)
    if st.button("Fit LDA", type="primary"):
        with st.spinner("Fitting LDA…"):
            result = fit_lda(sample["text"].astype(str).tolist(), num_topics=n_topics)
        if not result.topics:
            st.warning("LDA produced no topics.")
            return
        for topic in result.topics:
            st.subheader(f"Topic {topic.topic_id}: {topic.label}")
            st.write(", ".join(f"{w} ({s:.3f})" for w, s in topic.words))
            with st.expander("How LDA produced this topic"):
                st.write(
                    "Documents are bag-of-words after stopword removal and lemmatization. "
                    "Latent Dirichlet Allocation estimates topic-word and document-topic "
                    "distributions; each topic below shows its top words by probability."
                )
                st.dataframe(
                    pd.DataFrame(topic.words, columns=["word", "weight"]),
                    width="stretch",
                    hide_index=True,
                )


def page_eval() -> None:
    st.header("Quick evaluation sandbox")
    pred = st.text_area("Prediction", "The company reported strong quarterly earnings.")
    ref = st.text_area("Reference", "The firm posted strong earnings for the quarter.")
    if st.button("Score"):
        st.write(
            {
                "EM": exact_match(pred, ref),
                "F1": token_f1(pred, ref),
                **rouge_scores(pred, ref),
            }
        )


def main() -> None:
    st.set_page_config(page_title="NewsMind", layout="wide")
    st.title("NewsMind")
    st.caption(
        "Read news · summarize (BART/T5/Pegasus) · hybrid RAG · compare classical vs modern NLP."
    )
    page = st.sidebar.radio(
        "Page",
        ["Read news", "Overview", "Compare methods", "Topics (LDA)", "Eval"],
        index=0,
    )
    if page == "Read news":
        page_read_news()
    elif page == "Overview":
        page_overview()
    elif page == "Compare methods":
        page_compare()
    elif page == "Topics (LDA)":
        page_topics()
    else:
        page_eval()


if __name__ == "__main__":
    main()
