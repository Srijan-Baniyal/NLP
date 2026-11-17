"""Draw every diagram used in the NewsMind report as a PNG (docs/diagrams/)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

OUT = Path(__file__).parent / "diagrams"
OUT.mkdir(exist_ok=True)

BLUE, TEAL, ORANGE = "#2563eb", "#0d9488", "#ea580c"
PURPLE, GREY, GREEN = "#7c3aed", "#475569", "#16a34a"
RED, LIGHT, PINK = "#dc2626", "#e2e8f0", "#db2777"
INK = "#1e293b"


def canvas(w: float, h: float):
    """w x h in 'drawing units' (10 units = 1 inch)."""
    fig, ax = plt.subplots(figsize=(w / 10, h / 10))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc=BLUE, tc="white", fs=10, bold=False, ec=None, alpha=1.0):
    ax.add_patch(
        FancyBboxPatch(
            (cx - w / 2, cy - h / 2),
            w,
            h,
            boxstyle="round,pad=0.3,rounding_size=1.8",
            fc=fc,
            ec=ec or fc,
            lw=1.5,
            alpha=alpha,
        )
    )
    ax.text(
        cx, cy, text, ha="center", va="center", color=tc, fontsize=fs,
        fontweight="bold" if bold else "normal", linespacing=1.3,
    )


def arrow(ax, p1, p2, color=GREY, lw=1.8, style="-|>", ls="-"):
    ax.annotate(
        "", xy=p2, xytext=p1,
        arrowprops=dict(arrowstyle=style, lw=lw, color=color, linestyle=ls, shrinkA=0, shrinkB=0),
    )


def title(ax, w, h, text):
    ax.text(w / 2, h - 3, text, ha="center", va="center", fontsize=14, fontweight="bold", color=INK)


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1 ------------------------------------------------------------ big picture
def big_picture():
    W, H = 130, 80
    fig, ax = canvas(W, H)
    title(ax, W, H, "NewsMind at a glance")
    srcs = [("Hugging Face\ndatasets", 58), ("Live RSS\nfeeds", 46), ("CC-News\narchive", 34), ("GDELT\nheadlines", 22)]
    for t, y in srcs:
        box(ax, 14, y, 22, 9, t, fc=GREY, fs=9)
        arrow(ax, (25.5, y), (33.5, 40))
    ax.text(14, 67, "1. Collect", ha="center", fontsize=11, fontweight="bold", color=INK)
    box(ax, 44, 40, 20, 24, "One unified\ncorpus\n(Parquet files)", fc=BLUE, bold=True)
    ax.text(44, 56, "2. Store", ha="center", fontsize=11, fontweight="bold", color=INK)
    arrow(ax, (54.5, 40), (62.5, 40))
    box(ax, 75, 40, 24, 24, "Clean text\nSplit into chunks\nBuild search\nindex", fc=TEAL, bold=True)
    ax.text(75, 56, "3. Prepare", ha="center", fontsize=11, fontweight="bold", color=INK)
    outs = [("Summaries\n(short version)", 58, ORANGE), ("Cited answers\n(ask anything)", 40, PURPLE), ("Topics\n(what is trending)", 22, PINK)]
    for t, y, c in outs:
        arrow(ax, (87.5, 40), (99.5, y))
        box(ax, 112, y, 24, 11, t, fc=c, fs=10, bold=True)
    ax.text(112, 67, "4. Deliver", ha="center", fontsize=11, fontweight="bold", color=INK)
    box(ax, 65, 6, 110, 8, "Everything is served through a Streamlit web app", fc=LIGHT, tc=INK, fs=11, bold=True)
    save(fig, "01_big_picture")


# 2 ------------------------------------------------------------ data sizes
def data_sizes():
    data = {
        "gdelt_headlines": 1504, "cc_news": 1120, "cnn_dailymail": 816, "xsum": 333,
        "huffpost_categories": 58, "ag_news": 31, "outlet_archive": 24, "rss_live": 0.85,
    }
    fig, ax = plt.subplots(figsize=(9, 4.6))
    names = list(data)[::-1]
    vals = [data[n] for n in names]
    cols = plt.cm.Blues([0.35 + 0.65 * v / 1504 for v in vals])
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v + 15, i, f"{v:,.0f} MB" if v >= 1 else "0.85 MB", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, 1750)
    ax.set_xlabel("File size on disk (MB)")
    ax.set_title("What is inside data/raw/ (about 3.9 GB in total)", fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "02_data_sizes")


# 3 ------------------------------------------------------------ schema
def schema():
    W, H = 130, 62
    fig, ax = canvas(W, H)
    title(ax, W, H, "Every article, whatever its source, becomes the same 10-field record")
    fields = [
        ("article_id", "unique ID"), ("corpus", "which dataset"), ("outlet", "who published"),
        ("title", "headline"), ("text", "body / blurb"), ("summary", "reference summary"),
        ("url", "original link"), ("published_at", "when"), ("split", "train/val/test"), ("topic", "category"),
    ]
    cols = [BLUE, BLUE, BLUE, TEAL, TEAL, ORANGE, GREY, GREY, PURPLE, PINK]
    for i, ((f, d), c) in enumerate(zip(fields, cols, strict=True)):
        r, k = divmod(i, 5)
        x, y = 13 + k * 26, 38 - r * 20
        box(ax, x, y, 23, 14, f"{f}\n{d}", fc=c, fs=9, bold=False)
    ax.text(W / 2, 6, "Defined once in schema.py (ArticleRow), so the rest of the code never needs to care where an article came from.",
            ha="center", fontsize=9, color=GREY, style="italic")
    save(fig, "03_schema")


# 4 ------------------------------------------------------------ preprocessing
def preprocessing():
    W, H = 130, 100
    fig, ax = canvas(W, H)
    title(ax, W, H, "Text preprocessing - from a sentence to clean features (illustrative)")
    rows = [
        ("Raw text", ["The stocks were climbing"], BLUE),
        ("Tokenize", ["The", "stocks", "were", "climbing"], TEAL),
        ("Drop stop-words", ["stocks", "climbing"], ORANGE),
        ("Lemmatize", ["stock", "climb"], PURPLE),
        ("POS tags", ["The  DET", "stocks  NOUN", "were  AUX", "climbing  VERB"], PINK),
    ]
    ys = [82, 66, 50, 34, 18]
    for (label, items, c), y in zip(rows, ys, strict=True):
        ax.text(3, y, label, fontsize=11, fontweight="bold", color=INK, va="center")
        if len(items) == 1:
            box(ax, 78, y, 52, 9, items[0], fc=c, fs=11)
        else:
            n = len(items)
            wd = min(24, 90 / n - 2)
            start = 78 - (n - 1) * (wd + 2) / 2
            for i, it in enumerate(items):
                box(ax, start + i * (wd + 2), y, wd, 9, it, fc=c, fs=10)
    for a, b in zip(ys, ys[1:], strict=False):
        if b != 18:
            arrow(ax, (78, a - 5.5), (78, b + 5.5))
    ax.text(78, 6, 'Noun phrase found: "The stocks"  (used for keywords and better matching)',
            ha="center", fontsize=9.5, color=GREY, style="italic")
    save(fig, "04_preprocessing")


# 5 ------------------------------------------------------------ chunking
def chunking():
    W, H = 130, 62
    fig, ax = canvas(W, H)
    title(ax, W, H, "Chunking - cut long articles into overlapping passages")
    ax.add_patch(Rectangle((10, 40), 110, 8, fc=LIGHT, ec=GREY, lw=1.5))
    ax.text(65, 44, "One long news article", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    L, step, ov = 40, 35.2, 4.8 * 110 / 100 * 1.0
    starts = [10, 10 + 36 * 1.1 - 4.8 * 1.1, 10 + 2 * (36 - 4.8) * 1.1]
    size = 36 * 1.1
    cols = [BLUE, TEAL, ORANGE]
    ys = [30, 20, 10]
    for i, (s, c, y) in enumerate(zip(starts, cols, ys, strict=True)):
        ax.add_patch(Rectangle((s, y - 3.5), size, 7, fc=c, ec="white", lw=1.5))
        ax.text(s + size / 2, y, f"Chunk {i + 1}  (~900 chars)", ha="center", va="center", color="white", fontsize=10)
        arrow(ax, (s + size / 2, 40), (s + size / 2, y + 4), color=c, lw=1, ls="--")
    for i in range(2):
        s = starts[i + 1]
        ax.add_patch(Rectangle((s, 6), 4.8 * 1.1, 28, fc="none", ec=RED, lw=1.5, ls="--"))
    ax.text(starts[1] + 2.6, 2.5, "120-char overlap", ha="center", fontsize=9, color=RED)
    ax.text(124, 20, "Overlap keeps a\nsentence from being\ncut in half and\nlost between chunks", fontsize=8.5, color=GREY, va="center", ha="right", style="italic")
    save(fig, "05_chunking")


# 6 ------------------------------------------------------------ retrieval evolution
def retrieval_evolution():
    W, H = 140, 64
    fig, ax = canvas(W, H)
    title(ax, W, H, "Five ways NewsMind can find relevant text - from simple to smart")
    items = [
        ("TF-IDF", "Counts words,\nrare words\nweigh more", GREY),
        ("BM25", "Smarter counting:\nlength-aware,\nsaturating", BLUE),
        ("Word2Vec\nGloVe", "Each word is a\nvector; similar\nwords sit close", TEAL),
        ("MiniLM\n+ FAISS", "Whole sentences\nbecome vectors;\nfast search", PURPLE),
        ("Hybrid +\nRerank", "Blend BM25 and\nMiniLM, then\nre-judge top hits", ORANGE),
    ]
    for i, (n, d, c) in enumerate(items):
        x = 14 + i * 28
        box(ax, x, 40, 22, 14, n, fc=c, fs=11, bold=True)
        ax.text(x, 22, d, ha="center", va="center", fontsize=9, color=INK)
        if i < 4:
            arrow(ax, (x + 11.8, 40), (x + 16.2, 40))
    arrow(ax, (6, 9), (134, 9), color=INK, lw=2.4)
    ax.text(6, 13, "keyword matching", fontsize=9, color=GREY)
    ax.text(134, 13, "understands meaning", fontsize=9, color=GREY, ha="right")
    save(fig, "06_retrieval_evolution")


# 7 ------------------------------------------------------------ sparse vs dense
def sparse_vs_dense():
    W, H = 130, 74
    fig, ax = canvas(W, H)
    title(ax, W, H, "Why use both? Keyword search and meaning search fail in different ways")
    ax.text(65, 62, 'Question: "Who won the election?"', ha="center", fontsize=11, fontweight="bold", color=INK)
    box(ax, 34, 38, 52, 38, "", fc=LIGHT, tc=INK, ec=BLUE)
    ax.text(34, 51, "Keyword search (BM25)", ha="center", fontsize=11, fontweight="bold", color=BLUE)
    ax.text(34, 40, 'Finds: "...won the election by 3 points"\nMisses: "...claimed victory at the polls"\n\nGreat for names, places, numbers',
            ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.4)
    box(ax, 96, 38, 52, 38, "", fc=LIGHT, tc=INK, ec=PURPLE)
    ax.text(96, 51, "Meaning search (embeddings)", ha="center", fontsize=11, fontweight="bold", color=PURPLE)
    ax.text(96, 40, 'Finds: "...claimed victory at the polls"\nMay drift: related but off-topic stories\n\nGreat for paraphrases',
            ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.4)
    arrow(ax, (34, 18), (58, 9))
    arrow(ax, (96, 18), (72, 9))
    box(ax, 65, 7, 28, 8, "Combine both", fc=ORANGE, fs=11, bold=True)
    save(fig, "07_sparse_vs_dense")


# 8 ------------------------------------------------------------ hybrid scoring
def hybrid_scoring():
    import numpy as np

    labels = ["Passage A", "Passage B", "Passage C"]
    bm25 = np.array([1.0, 0.5, 0.2])
    dense = np.array([0.4, 1.0, 0.6])
    fused = 0.45 * bm25 + 0.55 * dense
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    ax.bar(x - 0.27, bm25, 0.27, label="BM25 (normalized)", color=BLUE)
    ax.bar(x, dense, 0.27, label="Dense (normalized)", color=PURPLE)
    ax.bar(x + 0.27, fused, 0.27, label="Fused = 0.45 BM25 + 0.55 Dense", color=ORANGE)
    for i, f in enumerate(fused):
        ax.text(i + 0.27, f + 0.02, f"{f:.2f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score (0-1)")
    ax.set_title("Hybrid fusion with made-up numbers: B wins even though A is the best keyword match",
                 fontsize=10.5, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "08_hybrid_scoring")


# 9 ------------------------------------------------------------ funnel
def funnel():
    W, H = 140, 70
    fig, ax = canvas(W, H)
    title(ax, W, H, "The retrieval funnel: many candidates in, a few strong passages out")
    box(ax, 12, 35, 18, 12, "Your\nquestion", fc=GREY, bold=True)
    box(ax, 40, 50, 22, 12, "BM25\ntop 40", fc=BLUE, bold=True)
    box(ax, 40, 20, 22, 12, "MiniLM + FAISS\ntop 40", fc=PURPLE, bold=True, fs=9)
    arrow(ax, (21.5, 38), (28.5, 48))
    arrow(ax, (21.5, 32), (28.5, 22))
    box(ax, 68, 35, 20, 16, "Hybrid\nfusion\ntop 20", fc=TEAL, bold=True)
    arrow(ax, (51.5, 48), (58, 40))
    arrow(ax, (51.5, 22), (58, 30))
    box(ax, 95, 35, 20, 16, "Cross-encoder\nrerank\ntop 5", fc=ORANGE, bold=True, fs=9.5)
    arrow(ax, (78.5, 35), (84.5, 35))
    box(ax, 124, 35, 22, 16, "FLAN-T5\nanswer +\ncitations", fc=GREEN, bold=True)
    arrow(ax, (105.5, 35), (112.5, 35))
    ax.text(70, 6, "Defaults from config.py: RETRIEVE_TOP_K = 20 (each retriever fetches 2x), RERANK_TOP_K = 5, HYBRID_ALPHA = 0.55",
            ha="center", fontsize=8.5, color=GREY, style="italic")
    save(fig, "09_funnel")


# 10 ----------------------------------------------------------- rag steps
def rag_steps():
    W, H = 120, 150
    fig, ax = canvas(W, H)
    title(ax, W, H, "The six steps behind every answer")
    steps = [
        ("Query intake", "Clean the question; optionally add the opened article as context.", GREY),
        ("Sparse retrieval", "BM25 keyword search over lemmatized tokens (top 40).", BLUE),
        ("Dense retrieval", "MiniLM embeddings searched in FAISS (top 40).", PURPLE),
        ("Hybrid fusion", "Normalize both score sets, blend 0.55 dense + 0.45 BM25 (top 20).", TEAL),
        ("Cross-encoder rerank", "Read (question, passage) together and keep the best 5.", ORANGE),
        ("Grounded generation", "FLAN-T5 writes the answer only from those passages, with citations.", GREEN),
    ]
    for i, (n, d, c) in enumerate(steps):
        y = H - 20 - i * 23
        ax.add_patch(Circle((12, y), 6, fc=c, ec=c))
        ax.text(12, y, str(i + 1), ha="center", va="center", color="white", fontsize=14, fontweight="bold")
        box(ax, 68, y, 82, 15, "", fc=LIGHT, ec=c, tc=INK)
        ax.text(30, y + 3.2, n, fontsize=12, fontweight="bold", color=c, va="center")
        ax.text(30, y - 3.5, d, fontsize=9, color=INK, va="center")
        if i < 5:
            arrow(ax, (12, y - 6.2), (12, y - 16.8), color=GREY)
    save(fig, "10_rag_steps")


# 11 ----------------------------------------------------------- sequence
def sequence():
    W, H = 140, 100
    fig, ax = canvas(W, H)
    title(ax, W, H, "What happens when you click 'Ask'")
    actors = [("You", 14, GREY), ("Streamlit\napp", 40, BLUE), ("Retrievers\nBM25 + FAISS", 68, PURPLE),
              ("Cross-\nencoder", 97, ORANGE), ("FLAN-T5", 124, GREEN)]
    for n, x, c in actors:
        box(ax, x, 86, 20, 10, n, fc=c, fs=9, bold=True)
        ax.plot([x, x], [80, 4], color=c, lw=1, ls="--", alpha=0.6)
    msgs = [
        (14, 40, 72, "Ask a question"),
        (40, 68, 64, "search(query)"),
        (68, 40, 56, "40 + 40 hits -> fused top 20"),
        (40, 97, 48, "rerank top 20"),
        (97, 40, 40, "best 5 passages"),
        (40, 124, 32, "prompt = question + 5 passages"),
        (124, 40, 24, "answer + citations"),
        (40, 14, 16, "Show answer and sources"),
    ]
    for a, b, y, t in msgs:
        arrow(ax, (a, y), (b, y), color=INK)
        ax.text((a + b) / 2, y + 2.2, t, ha="center", fontsize=8.5, color=INK)
    save(fig, "11_sequence")


# 12 ----------------------------------------------------------- summarization lanes
def summarization():
    W, H = 140, 96
    fig, ax = canvas(W, H)
    title(ax, W, H, "Two ways to summarize")
    # extractive
    ax.text(3, 80, "Extractive: pick the best existing sentences", fontsize=11, fontweight="bold", color=BLUE)
    for i in range(6):
        picked = i in (1, 4)
        box(ax, 12 + i * 12.5, 66, 11, 8, f"S{i + 1}", fc=BLUE if picked else LIGHT, tc="white" if picked else INK, fs=10, bold=picked)
    arrow(ax, (50, 60), (50, 50))
    box(ax, 50, 44, 40, 9, "S2 + S5, copied word-for-word", fc=BLUE, fs=10)
    ax.text(100, 66, "Methods:\nTF-IDF sentence scores\nTextRank (graph ranking)", fontsize=9.5, color=INK, va="center")
    ax.text(100, 44, "Faithful, but can read choppy.", fontsize=9.5, color=GREY, style="italic", va="center")
    ax.plot([3, 137], [34, 34], color=LIGHT, lw=2)
    # abstractive
    ax.text(3, 28, "Abstractive: write a brand-new summary", fontsize=11, fontweight="bold", color=ORANGE)
    box(ax, 16, 14, 24, 11, "Full article", fc=LIGHT, tc=INK)
    arrow(ax, (28.5, 14), (38, 14))
    box(ax, 56, 14, 34, 14, "Seq2seq model\nBART | T5 | Pegasus", fc=ORANGE, bold=True, fs=9.5)
    arrow(ax, (73.5, 14), (83, 14))
    box(ax, 100, 14, 30, 11, "New sentences", fc=ORANGE)
    ax.text(125, 14, "Fluent, but\ncan invent facts.", fontsize=9.5, color=GREY, style="italic", va="center")
    save(fig, "12_summarization")


# 13 ----------------------------------------------------------- tradeoff quadrant
def tradeoff():
    fig, ax = plt.subplots(figsize=(8, 5.4))
    pts = {
        "TF-IDF": (3.0, 9.3, BLUE), "TextRank": (3.6, 9.0, BLUE),
        "BART": (7.6, 7.0, ORANGE), "T5-small": (5.6, 6.0, ORANGE), "Pegasus": (8.6, 5.2, ORANGE),
    }
    for n, (x, y, c) in pts.items():
        ax.scatter(x, y, s=260, color=c, zorder=3)
        ax.annotate(n, (x, y), xytext=(10, 10), textcoords="offset points", fontsize=10, fontweight="bold", color=INK)
    ax.axhline(5, color=LIGHT, zorder=1)
    ax.axvline(5, color=LIGHT, zorder=1)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel("Fluency  (reads like a human)  ->")
    ax.set_ylabel("Faithfulness  (sticks to the source)  ->")
    ax.set_title("Summarizer trade-off (qualitative sketch, not measured)", fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "13_tradeoff")


# 14 ----------------------------------------------------------- package map
def package_map():
    W, H = 140, 92
    fig, ax = canvas(W, H)
    title(ax, W, H, "How the code is organized (src/newsmind/)")
    layers = [
        ("Data layer", BLUE, ["config.py", "schema.py", "corpus.py", "ingest/\n hf, rss, outlets, scale"]),
        ("NLP layer", TEAL, ["preprocess/\n tokens, POS, phrases", "retrieve/\n tfidf, bm25, dense,\n w2v, glove, rerank", "summarize/", "topics/ (LDA)"]),
        ("Application layer", ORANGE, ["rag/\n chunking, generate,\n pipeline", "index/build.py", "eval/\n metrics, compare", "app.py (Streamlit)"]),
    ]
    for j, (name, c, mods) in enumerate(layers):
        x = 24 + j * 46
        box(ax, x, 76, 38, 9, name, fc=c, bold=True, fs=12)
        for i, m in enumerate(mods):
            box(ax, x, 62 - i * 16, 38, 12.5, m, fc=LIGHT, tc=INK, ec=c, fs=8.5)
        if j < 2:
            arrow(ax, (x + 19.5, 76), (x + 26.5, 76))
    save(fig, "14_package_map")


# 15 ----------------------------------------------------------- UI map
def ui_map():
    W, H = 140, 80
    fig, ax = canvas(W, H)
    title(ax, W, H, "The Streamlit app: four pages")
    box(ax, 70, 64, 50, 10, "streamlit run src/newsmind/app.py", fc=GREY, bold=True, fs=10)
    pages = [
        ("Read news", "Browse articles,\nopen one", BLUE, 18),
        ("Compare methods", "Side-by-side\nsummarizers /\nretrievers", TEAL, 50),
        ("Topics", "LDA themes\nand top words", PURPLE, 90),
        ("Eval", "ROUGE, EM/F1,\nP@k, R@k", PINK, 122),
    ]
    for n, d, c, x in pages:
        arrow(ax, (70, 58.5), (x, 50.5))
        box(ax, x, 38, 28, 20, f"{n}\n\n{d}", fc=c, fs=9)
    box(ax, 8, 8, 14, 8, "Summarize", fc=ORANGE, fs=9)
    box(ax, 28, 8, 14, 8, "Ask (RAG)", fc=GREEN, fs=9)
    arrow(ax, (14, 27), (10, 12.5), lw=1.3)
    arrow(ax, (22, 27), (26, 12.5), lw=1.3)
    save(fig, "15_ui_map")


# 16 ----------------------------------------------------------- scale
def scale():
    import json

    meta = json.loads((OUT.parent.parent / "data" / "index" / "index_meta.json").read_text())
    names = [
        "Target corpus\n(rows)",
        "Working set\n(rows)",
        "Chunks in\ncurrent index",
        "Articles in\ncurrent index",
    ]
    vals = [10_000_000, 100_000, meta["chunk_count"], meta["article_count"]]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    bars = ax.bar(names, vals, color=[BLUE, TEAL, ORANGE, PURPLE])
    ax.set_yscale("log")
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.2, f"{v:,}", ha="center", fontweight="bold", color=INK)
    ax.set_ylim(1000, 5e7)
    ax.set_title("Scale of the corpus vs. what is indexed today (log scale)", fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    save(fig, "16_scale")


# 17 ----------------------------------------------------------- eval map
def eval_map():
    W, H = 140, 72
    fig, ax = canvas(W, H)
    title(ax, W, H, "How we judge quality")
    cols = [
        ("Summaries", ORANGE, "ROUGE-1 / 2 / L", "overlap with the\nreference summary"),
        ("Answers", GREEN, "Exact Match + F1", "does the answer match\nthe gold answer?"),
        ("Retrieval", PURPLE, "Precision@k\nRecall@k", "are the right passages\nnear the top?"),
    ]
    for i, (h, c, m, d) in enumerate(cols):
        x = 24 + i * 46
        box(ax, x, 54, 38, 10, h, fc=c, bold=True, fs=12)
        arrow(ax, (x, 48.5), (x, 40.5))
        box(ax, x, 33, 38, 12, m, fc=LIGHT, tc=INK, ec=c, bold=True, fs=10)
        arrow(ax, (x, 26.5), (x, 19.5))
        ax.text(x, 13, d, ha="center", va="center", fontsize=9.5, color=INK)
    save(fig, "17_eval_map")


# 18 ----------------------------------------------------------- dev workflow
def dev_workflow():
    W, H = 140, 40
    fig, ax = canvas(W, H)
    title(ax, W, H, "Quality gate before every commit")
    items = [("ruff check", "lint", BLUE), ("ruff format", "style", TEAL), ("ty check", "types", PURPLE), ("pytest", "tests", GREEN)]
    for i, (n, d, c) in enumerate(items):
        x = 18 + i * 34
        box(ax, x, 16, 24, 12, f"{n}\n({d})", fc=c, bold=True, fs=10)
        if i < 3:
            arrow(ax, (x + 13, 16), (x + 21, 16))
    save(fig, "18_dev_workflow")


if __name__ == "__main__":
    for fn in (big_picture, data_sizes, schema, preprocessing, chunking, retrieval_evolution,
               sparse_vs_dense, hybrid_scoring, funnel, rag_steps, sequence, summarization,
               tradeoff, package_map, ui_map, scale, eval_map, dev_workflow):
        fn()
    print(f"Wrote {len(list(OUT.glob('*.png')))} diagrams to {OUT}")
