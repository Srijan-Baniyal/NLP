"""Generate the NewsMind project report (Word) with diagrams."""

import json
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
META = json.loads((HERE.parent / "data" / "index" / "index_meta.json").read_text())
N_ART = f"{META['article_count']:,}"
N_CHUNK = f"{META['chunk_count']:,}"
CORPORA = ", ".join(META["corpora"])
IMG = HERE / "diagrams"
OUT = HERE / "NewsMind_Project_Report.docx"

doc = Document()
for s in doc.sections:
    s.left_margin = s.right_margin = Inches(1)
    s.top_margin = s.bottom_margin = Inches(0.9)

normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(11.5)
normal.paragraph_format.space_after = Pt(8)
normal.paragraph_format.line_spacing = 1.15
for name, size, color in (("Heading 1", 20, "1E3A8A"), ("Heading 2", 15, "2563EB"), ("Heading 3", 12.5, "0D9488")):
    st = doc.styles[name]
    st.font.name = "Calibri"
    st.font.size = Pt(size)
    st.font.bold = True
    st.font.color.rgb = RGBColor.from_string(color)

fig_no = 0


# ------------------------------------------------------------------ helpers
def shade(cell, hex_fill: str) -> None:
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), hex_fill)
    tcPr.append(shd)


def para(text: str, italic: bool = False, align=None) -> None:
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic = italic
    if align:
        p.alignment = align


def rich(parts: list[tuple[str, bool]]) -> None:
    p = doc.add_paragraph()
    for text, bold in parts:
        p.add_run(text).bold = bold


def bullets(items: list[str]) -> None:
    for item in items:
        if ": " in item and item.index(": ") < 40:
            head, tail = item.split(": ", 1)
            p = doc.add_paragraph(style="List Bullet")
            p.add_run(head + ": ").bold = True
            p.add_run(tail)
        else:
            doc.add_paragraph(item, style="List Bullet")


def numbered(items: list[str]) -> None:
    for item in items:
        doc.add_paragraph(item, style="List Number")


def callout(label: str, text: str, fill: str = "EFF6FF") -> None:
    t = doc.add_table(rows=1, cols=1)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = t.rows[0].cells[0]
    shade(c, fill)
    p = c.paragraphs[0]
    p.add_run(label + "  ").bold = True
    p.add_run(text)
    doc.add_paragraph()


def table(header: list[str], rows: list[list[str]], widths: list[float] | None = None) -> None:
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(header):
        c = t.rows[0].cells[i]
        c.text = ""
        r = c.paragraphs[0].add_run(h)
        r.bold = True
        r.font.color.rgb = RGBColor(255, 255, 255)
        shade(c, "1E3A8A")
    for n, row in enumerate(rows):
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = v
            if n % 2:
                shade(cells[i], "F1F5F9")
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Inches(w)
    doc.add_paragraph()


def figure(file: str, caption: str, width: float = 6.4) -> None:
    global fig_no
    fig_no += 1
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(str(IMG / file), width=Inches(width))
    p.paragraph_format.keep_with_next = True
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = cap.add_run(f"Figure {fig_no}. {caption}")
    r.italic = True
    r.font.size = Pt(10)
    r.font.color.rgb = RGBColor(0x47, 0x55, 0x69)


def field(paragraph, instr: str, placeholder: str = "") -> None:
    run = paragraph.add_run()
    for kind, txt in (("begin", None), (None, instr), ("separate", None), ("text", placeholder), ("end", None)):
        if kind in ("begin", "separate", "end"):
            el = OxmlElement("w:fldChar")
            el.set(qn("w:fldCharType"), kind)
        elif kind == "text":
            el = OxmlElement("w:t")
            el.text = txt
        else:
            el = OxmlElement("w:instrText")
            el.set(qn("xml:space"), "preserve")
            el.text = txt
        run._r.append(el)


# ------------------------------------------------------------------ footer
fp = doc.sections[0].footer.paragraphs[0]
fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
fp.add_run("NewsMind - Project Report   |   Page ")
field(fp, "PAGE", "1")

# ------------------------------------------------------------------ cover
for _ in range(5):
    doc.add_paragraph()
t = doc.add_paragraph()
t.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = t.add_run("NewsMind")
r.bold = True
r.font.size = Pt(54)
r.font.color.rgb = RGBColor.from_string("1E3A8A")
s = doc.add_paragraph()
s.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = s.add_run("Read less. Understand more.\nSummaries and cited answers from a huge news corpus.")
r.font.size = Pt(17)
r.font.color.rgb = RGBColor.from_string("2563EB")
doc.add_paragraph()
doc.add_picture(str(IMG / "01_big_picture.png"), width=Inches(6.2))
doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
doc.add_paragraph()
m = doc.add_paragraph()
m.alignment = WD_ALIGN_PARAGRAPH.CENTER
m.add_run("Project-Based Learning - Fundamentals of NLP\n").bold = True
m.add_run("Author: Srijan Baniyal   |   Version 0.1.0")
doc.add_page_break()

# ------------------------------------------------------------------ TOC
doc.add_heading("Contents", level=1)
tp = doc.add_paragraph()
field(tp, 'TOC \\o "1-2" \\h \\z \\u', "Right-click here and choose 'Update Field' to build the table of contents.")
doc.add_page_break()

# ================================================================== 1
doc.add_heading("1. The short version", level=1)
para(
    "Every day, thousands of news stories are published. Nobody can read them all, and "
    "asking a chatbot about the news is risky because chatbots sometimes make things up."
)
para(
    "NewsMind is our answer to both problems. It is a system that reads a very large "
    "collection of news articles and then does three useful things for you:"
)
bullets(
    [
        "Summarize: shrink any article into a few sentences, using five different techniques you can compare.",
        "Answer: reply to your questions in plain English and show exactly which articles the answer came from.",
        "Discover: reveal the main themes hiding in thousands of headlines.",
    ]
)
callout(
    "In one sentence:",
    "NewsMind is a search engine, a summarizer and a careful research assistant for news, "
    "built to show the main ideas of a Fundamentals of NLP course in one working product.",
)

doc.add_heading("1.1 An analogy to keep in mind", level=2)
para(
    "Imagine a giant library with millions of newspaper clippings. NewsMind plays three "
    "roles inside it. A librarian quickly pulls the dozen clippings most relevant to your "
    "question. A strict editor re-reads those dozen and keeps only the best five. Finally, "
    "a writer composes an answer using only those five clippings, and footnotes every claim. "
    "That is Retrieval-Augmented Generation (RAG) in everyday terms."
)

doc.add_heading("1.2 Why build it?", level=2)
table(
    ["The problem", "How NewsMind helps"],
    [
        ["Too much news to read", "Short summaries, in the style you prefer."],
        ["AI chatbots can invent facts", "Answers are built only from retrieved articles and come with citations."],
        ["Keyword search misses paraphrases", "A hybrid of keyword and meaning-based search."],
        ["Hard to see the big picture", "Topic modelling surfaces the main themes."],
        ["Scraping news sites is legally murky", "Only RSS feeds and research archives are used."],
    ],
    [2.4, 4.1],
)

doc.add_heading("1.3 Goals", level=2)
bullets(
    [
        "Build a large, multi-source news corpus with one consistent format.",
        "Implement classical and neural retrieval, and compare them honestly.",
        "Offer both extractive and abstractive summarization side by side.",
        "Answer questions with citations through a hybrid RAG pipeline.",
        "Make everything explorable in a friendly web app.",
        "Stay on the right side of publisher rights.",
    ]
)

# ================================================================== 2
doc.add_heading("2. Playing fair: the legal and ethical approach", level=1)
para(
    "A news project is only as trustworthy as the way it collects its data. NewsMind does "
    "not scrape or store full web pages from The New York Times, The Wall Street Journal, "
    "The Washington Times, Hindustan Times, The Times of India or The Indian Express."
)
para("Those outlets appear in the corpus only through two legitimate doors:")
numbered(
    [
        "Official RSS feeds: headlines and the short blurbs publishers choose to share (newsmind-rss).",
        "The CC-News research archive: a public academic dataset, filtered by outlet into outlet_archive.parquet (newsmind-outlets).",
    ]
)
callout("Why it matters:", "No paywalls are bypassed, no terms of service are broken, and the project can be shared and reproduced openly.", "ECFDF5")

# ================================================================== 3
doc.add_heading("3. The data", level=1)
para(
    "NewsMind's raw material is eight Parquet files in data/raw/. Together they weigh "
    "about 3.9 GB. Each file comes from a different place and is shaped differently, so "
    "the first job is to translate them all into one common language."
)
figure("02_data_sizes.png", "The eight raw data files, by size on disk.", 6.0)
table(
    ["File", "What it contains", "Why we use it"],
    [
        ["cnn_dailymail", "News articles with human-written highlights", "Training/testing summarization; ROUGE references"],
        ["xsum", "BBC articles with one-sentence summaries", "Tests very short abstractive summaries"],
        ["cc_news", "Common Crawl news articles", "Large, diverse real-world text"],
        ["gdelt_headlines", "Huge headline collection", "Reaches the 10-million-row scale target"],
        ["ag_news", "News snippets with topic labels", "Topic classification and LDA checks"],
        ["huffpost_categories", "HuffPost headlines with categories", "More labelled topics"],
        ["outlet_archive", "Named-outlet articles from CC-News", "NYT, WSJ, TOI and others - legally"],
        ["rss_live", "Today's headlines from official feeds", "Keeps the corpus fresh"],
    ],
    [1.5, 2.5, 2.5],
)

doc.add_heading("3.1 One format to rule them all", level=2)
para(
    "Whatever the source, every article is stored as the same ten-field record, defined "
    "once in schema.py. This is a small decision with a big payoff: the retrieval, "
    "summarization and app code never need to ask 'where did this article come from?'"
)
figure("03_schema.png", "The unified article record (ArticleRow).", 6.4)
figure(
    "16_scale.png",
    f"The corpus target is 10 million rows; the working set is 100,000; the index holds "
    f"{N_ART} articles in {N_CHUNK} chunks.",
    5.6,
)

# ================================================================== 4
doc.add_heading("4. How the system fits together", level=1)
para(
    "Think of NewsMind as an assembly line. Articles come in at one end; summaries, "
    "answers and topic maps come out of the other. Each stage has one clear job."
)
figure("01_big_picture.png", "The four stages: collect, store, prepare, deliver.", 6.4)
doc.add_heading("4.1 Where everything lives in the code", level=2)
figure("14_package_map.png", "The code is split into three layers. Data feeds NLP, which feeds the application.", 6.2)
table(
    ["Module", "Plain-English job"],
    [
        ["config", "One place for paths, model names and tuning knobs."],
        ["schema, corpus", "Define and load the common article record."],
        ["ingest", "Download datasets, read RSS feeds, filter outlets, scale to 10M rows."],
        ["preprocess", "Clean text: tokens, lemmas, parts of speech, noun phrases."],
        ["retrieve", "All the ways of finding relevant text, plus the reranker."],
        ["summarize", "Five summarizers, extractive and abstractive."],
        ["topics", "LDA topic discovery."],
        ["rag", "Chunking, answer generation and the end-to-end pipeline."],
        ["index", "Builds and saves the search index to disk."],
        ["eval", "Measures how good everything is."],
        ["app", "The Streamlit web interface."],
    ],
    [1.6, 4.9],
)

# ================================================================== 5
doc.add_heading("5. Getting text ready for the machine", level=1)
para(
    "Computers do not read; they count. Before any smart algorithm can touch a news "
    "article, the text must be cleaned and broken into pieces that are easy to count and compare."
)
doc.add_heading("5.1 Preprocessing", level=2)
para(
    "NewsMind uses NLTK to split text into words (tokenization), throw away filler words "
    "like 'the' and 'were' (stop-word removal), and reduce words to their dictionary form "
    "(lemmatization), so that 'stocks' and 'stock' count as the same thing. A separate "
    "syntax module tags each word's part of speech and pulls out noun phrases."
)
figure("04_preprocessing.png", "A tiny worked example of the preprocessing steps (illustrative).", 5.8)
callout("Why lemmatize?", "Without it, a search for 'climbing prices' would not match an article about 'prices climbed'.")

doc.add_heading("5.2 Chunking", level=2)
para(
    "Long articles are awkward: a single article may cover several subjects, and AI models "
    "can only read so much at once. So each article is cut into passages of about 900 "
    "characters. Neighbouring passages share a 120-character overlap so that no sentence "
    "is stranded at the edge of a cut."
)
figure("05_chunking.png", f"Overlapping chunks. The current index turns {N_ART} articles into {N_CHUNK} chunks.", 6.2)

# ================================================================== 6
doc.add_heading("6. Finding the right passages", level=1)
para(
    "Search is the heart of NewsMind, because a great answer built on the wrong passages "
    "is still a wrong answer. The project implements the whole history of text search, "
    "from word counting to neural embeddings, so each can be compared with the others."
)
figure("06_retrieval_evolution.png", "From simple keyword counting to meaning-aware search.", 6.5)
table(
    ["Method", "How it thinks", "Strength", "Weakness"],
    [
        ["TF-IDF", "Rare words matter more than common ones.", "Simple, fast, transparent.", "Ignores word order and meaning."],
        ["BM25", "TF-IDF refined for document length.", "Excellent for names, numbers, places.", "Misses paraphrases."],
        ["Word2Vec / GloVe", "Words are points in space; similar words are near.", "Captures word similarity.", "One vector per word, no context."],
        ["MiniLM + FAISS", "Whole sentences become vectors.", "Understands meaning, paraphrases.", "Can drift off-topic."],
        ["Hybrid", "Blend BM25 and MiniLM scores.", "Best of both worlds.", "Needs a tuned weight."],
        ["Cross-encoder", "Reads question and passage together.", "Most accurate judge.", "Too slow to run on everything."],
    ],
    [1.2, 2.0, 1.8, 1.5],
)

doc.add_heading("6.1 Why we need both kinds of search", level=2)
figure("07_sparse_vs_dense.png", "Keyword and meaning search fail in different ways, so we combine them.", 6.2)

doc.add_heading("6.2 How scores are blended", level=2)
para(
    "BM25 and the embedding model produce scores on different scales, so each list is "
    "first squeezed into a 0-1 range (min-max normalization). Then the two are mixed: "
    "55% weight on meaning (dense) and 45% on keywords (BM25). That 0.55 lives in "
    "config.py as HYBRID_ALPHA and is easy to change."
)
figure("08_hybrid_scoring.png", "Example with made-up scores: a passage that is a strong semantic match can beat the best keyword match.", 5.8)

doc.add_heading("6.3 The funnel", level=2)
para(
    "Searching is a funnel. Cast a wide net cheaply, then use progressively more careful "
    "(and more expensive) judges on fewer candidates."
)
figure("09_funnel.png", "40 + 40 candidates become 20 fused candidates, then the best 5.", 6.5)

# ================================================================== 7
doc.add_heading("7. Question answering with receipts", level=1)
para(
    "When you ask NewsMind a question, the RAGSystem walks through six steps. Every step "
    "records what it received and what it produced, so the app can show you the entire "
    "reasoning trail, not just the final answer."
)
figure("10_rag_steps.png", "The six steps of the RAG pipeline.", 5.0)
figure("11_sequence.png", "Who talks to whom when you press the Ask button.", 6.0)
callout(
    "Safety net:",
    "If the language model returns an empty or degenerate reply, NewsMind falls back to "
    "quoting the retrieved passages directly, still with citations. You never get a blank answer.",
    "FFF7ED",
)
doc.add_heading("7.1 The models behind the curtain", level=2)
table(
    ["Job", "Model", "Size / note"],
    [
        ["Sentence embeddings", "sentence-transformers/all-MiniLM-L6-v2", "Small and fast; good all-rounder"],
        ["Reranking", "cross-encoder/ms-marco-MiniLM-L-6-v2", "Trained on web search relevance"],
        ["Answer generation", "google/flan-t5-base", "Instruction-tuned; beam search with 4 beams"],
        ["Abstractive summary", "facebook/bart-large-cnn", "News-trained summarizer"],
        ["Abstractive summary", "t5-small", "Light and quick"],
        ["Abstractive summary", "google/pegasus-xsum", "One-sentence summaries"],
        ["Word vectors", "glove-wiki-gigaword-100", "Pretrained 100-dimension GloVe"],
    ],
    [1.6, 2.9, 2.0],
)

# ================================================================== 8
doc.add_heading("8. Summaries", level=1)
para(
    "A summary can be made in two very different ways, and NewsMind offers both so you "
    "can see the difference with your own eyes."
)
figure("12_summarization.png", "Extractive summarizers select sentences; abstractive ones write new ones.", 6.2)
table(
    ["Mode", "What it does"],
    [
        ["Extractive TF-IDF", "Scores each sentence by how many distinctive words it holds and keeps the top 3, in original order."],
        ["Extractive TextRank", "Builds a graph where sentences vote for each other (like PageRank) and keeps the most central."],
        ["Abstractive BART", "A transformer fine-tuned on CNN/DailyMail that writes a fluent multi-sentence summary."],
        ["Abstractive T5", "A compact text-to-text model prompted to summarize."],
        ["Abstractive Pegasus", "Trained on BBC XSum to produce a single, punchy sentence."],
    ],
    [1.7, 4.8],
)
figure("13_tradeoff.png", "The classic trade-off: extractive is safer, abstractive is smoother. (Qualitative sketch.)", 5.2)
callout(
    "Rule of thumb:",
    "Need to quote or verify? Use extractive. Need something pleasant to read? Use abstractive, "
    "but double-check the facts.",
)

# ================================================================== 9
doc.add_heading("9. Topic discovery", level=1)
para(
    "What are all these articles about? Latent Dirichlet Allocation (LDA), implemented with "
    "Gensim, answers that without any labels. It assumes each article is a mix of a few "
    "hidden topics and each topic is a cluster of words that tend to appear together, for "
    "example {election, vote, party, minister} for politics or {match, league, goal, coach} "
    "for sport. The Topics page displays the top words of each discovered theme."
)

# ================================================================== 10
doc.add_heading("10. How do we know it works?", level=1)
para(
    "Good intentions are not results. NewsMind measures itself with standard NLP metrics, "
    "and the eval module can run the different methods head to head."
)
figure("17_eval_map.png", "Which metric judges which feature.", 6.2)
table(
    ["Metric", "In plain words"],
    [
        ["ROUGE-1 / 2 / L", "How many words, word pairs and longest sequences overlap with the human-written summary."],
        ["Exact Match (EM)", "Did the answer match the gold answer exactly?"],
        ["F1", "Partial credit: how much of the answer's words overlap with the gold answer."],
        ["Precision@k", "Of the top k passages returned, what share were relevant?"],
        ["Recall@k", "Of all relevant passages, what share appeared in the top k?"],
    ],
    [1.7, 4.8],
)

# ================================================================== 11
doc.add_heading("11. The app", level=1)
para("Everything above is wrapped in a Streamlit web app so that you can click instead of code.")
figure("15_ui_map.png", "The four pages of the app.", 6.2)
bullets(
    [
        "Read news: browse articles, open one, then summarize it or ask questions about it.",
        "Compare methods: see the different summarizers or retrievers side by side.",
        "Topics: explore the LDA themes.",
        "Eval: inspect the quality metrics.",
    ]
)

# ================================================================== 12
doc.add_heading("12. Where things stand today", level=1)
para("The search index saved in data/index/ currently looks like this:")
table(
    ["Property", "Value"],
    [
        ["Articles indexed", N_ART],
        ["Chunks (searchable passages)", N_CHUNK],
        ["Source corpora in the index", CORPORA],
        ["Embedding model", "all-MiniLM-L6-v2"],
        ["Reranker", "ms-marco-MiniLM-L-6-v2"],
    ],
    [3.0, 3.5],
)
para(
    "This is the full 100,000-article working set (WORKING_SET_SIZE) built with "
    "'uv run newsmind-index'. The raw corpus on disk is far larger (about 3.9 GB), and "
    "--limit can be raised to index more."
)

# ================================================================== 13
doc.add_heading("13. Try it yourself", level=1)
doc.add_heading("Step 1 - Install", level=3)
para("uv sync")
doc.add_heading("Step 2 - Collect data", level=3)
para("uv run newsmind-hf\nuv run newsmind-rss\nuv run newsmind-scale\nuv run newsmind-outlets")
doc.add_heading("Step 3 - Build the index", level=3)
para("uv run newsmind-index --limit 10000      (quick)\nuv run newsmind-index                          (everything)")
doc.add_heading("Step 4 - Launch the app", level=3)
para("uv run streamlit run src/newsmind/app.py")
doc.add_heading("Step 5 - Check quality", level=3)
figure("18_dev_workflow.png", "The four checks that guard the code.", 5.8)
para("uv run ruff check && uv run ruff format --check && uv run ty check && uv run pytest -q")

# ================================================================== 14
doc.add_heading("14. Built with", level=1)
table(
    ["Area", "Tools"],
    [
        ["Language and tooling", "Python 3.12+, uv, ruff, ty, pytest"],
        ["Data handling", "pandas, pyarrow, datasets, feedparser"],
        ["Classic NLP", "NLTK, scikit-learn, rank-bm25, gensim, sumy, networkx"],
        ["Neural NLP", "PyTorch, transformers, sentence-transformers, accelerate"],
        ["Vector search", "faiss-cpu"],
        ["Evaluation", "rouge-score"],
        ["Interface", "Streamlit"],
    ],
    [2.0, 4.5],
)

# ================================================================== 15
doc.add_heading("15. An honest look at strengths and limits", level=1)
doc.add_heading("What works well", level=3)
bullets(
    [
        "Hybrid search: finds both exact names and loosely-worded ideas.",
        "Reranking: a careful second opinion before any text reaches the generator.",
        "Transparency: citations plus a step-by-step trace for every answer.",
        "Clean data practices: no paywall scraping.",
    ]
)
doc.add_heading("What to be careful about", level=3)
bullets(
    [
        "Small generator: FLAN-T5-base can be shallow on questions that span many articles.",
        f"Index scope: only the {N_ART}-article working set is searchable, not the full corpus.",
        "Hallucination risk: abstractive summaries can still contain errors.",
        "Input limits: the models read roughly 1,024 tokens at a time.",
        "CPU-only search: the current FAISS build is not meant for 10 million vectors without tuning.",
    ]
)
doc.add_heading("Where it could go next", level=3)
bullets(
    [
        "Index the full corpus with an approximate FAISS index (IVF or HNSW).",
        "Swap in a larger instruction-tuned model for answers.",
        "Support Indian languages for the regional outlets.",
        "Add automatic fact-consistency checks for summaries.",
        "Rank fresh news higher with time-aware retrieval.",
    ]
)

# ================================================================== 16
doc.add_heading("16. Quick questions", level=1)
faq = [
    ("Is NewsMind a chatbot?", "Not really. A chatbot answers from memory. NewsMind answers from articles it just retrieved and shows them to you."),
    ("Why two kinds of search?", "Keyword search is precise but literal; meaning search is flexible but fuzzy. Together they cover each other's blind spots."),
    ("Why rerank at all?", "The cross-encoder is too slow to read millions of passages but very accurate on a handful, so it only judges the finalists."),
    ("Does it use live news?", "Yes, through official RSS feeds (rss_live), alongside the archived datasets."),
    ("Can I change the models?", "Yes. All model names and search settings live in config.py."),
]
for q, a in faq:
    rich([(q + "  ", True), (a, False)])

# ================================================================== 17
doc.add_heading("17. Glossary", level=1)
table(
    ["Term", "Meaning"],
    [
        ["Corpus", "A large collection of text documents."],
        ["Token", "A word or word piece, the basic unit of text."],
        ["Lemma", "The dictionary form of a word (climbing -> climb)."],
        ["Stop-word", "A very common word such as 'the' that carries little meaning."],
        ["TF-IDF / BM25", "Scoring formulas that rate how relevant a document is to a query based on word counts."],
        ["Embedding", "A list of numbers that represents the meaning of a word or sentence."],
        ["FAISS", "A library for very fast nearest-neighbour search over embeddings."],
        ["Cross-encoder", "A model that reads a question and a passage together to judge relevance."],
        ["RAG", "Retrieval-Augmented Generation: look things up first, then write the answer."],
        ["Extractive / Abstractive", "Copy sentences / write new ones."],
        ["LDA", "A method that discovers hidden topics from word patterns."],
        ["ROUGE", "A metric comparing a machine summary with a human one."],
    ],
    [1.8, 4.7],
)

# ================================================================== 18
doc.add_heading("18. Closing thoughts", level=1)
para(
    "NewsMind shows how the ideas of a first NLP course, from tokenization to transformers, "
    "combine into something genuinely useful: a tool that helps people read faster, ask "
    "better questions and trust the answers they get. It is also built with the habits of "
    "real engineering: a shared schema, one config file, automated checks and a respect "
    "for the people who write the news."
)

doc.save(OUT)
print(f"Saved {OUT} ({fig_no} figures)")
