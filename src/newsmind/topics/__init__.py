"""LDA topic modelling for corpus browsing."""

from __future__ import annotations

from dataclasses import dataclass

from gensim import corpora
from gensim.models import LdaModel

from newsmind.preprocess import preprocess


@dataclass(slots=True)
class Topic:
    topic_id: int
    words: list[tuple[str, float]]
    label: str


@dataclass(slots=True)
class TopicModelResult:
    topics: list[Topic]
    doc_topics: list[int]


def fit_lda(texts: list[str], num_topics: int = 8, passes: int = 8) -> TopicModelResult:
    tokenized = [preprocess(text) for text in texts]
    tokenized = [toks for toks in tokenized if toks]
    if not tokenized:
        return TopicModelResult(topics=[], doc_topics=[])
    dictionary = corpora.Dictionary(tokenized)
    dictionary.filter_extremes(no_below=2, no_above=0.6)
    corpus = [dictionary.doc2bow(toks) for toks in tokenized]
    if not dictionary or not corpus:
        return TopicModelResult(topics=[], doc_topics=[])
    model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=min(num_topics, max(2, len(dictionary) // 5)),
        passes=passes,
        random_state=42,
    )
    topics: list[Topic] = []
    for tid in range(model.num_topics):
        words = model.show_topic(tid, topn=8)
        label = ", ".join(w for w, _ in words[:4])
        topics.append(Topic(topic_id=tid, words=[(w, float(s)) for w, s in words], label=label))

    doc_topics: list[int] = []
    for bow in corpus:
        dist = model.get_document_topics(bow)
        doc_topics.append(max(dist, key=lambda x: x[1])[0] if dist else 0)
    return TopicModelResult(topics=topics, doc_topics=doc_topics)
