"""POS tagging and noun-phrase extraction for query understanding."""

from __future__ import annotations

from dataclasses import dataclass

from nltk import pos_tag
from nltk.chunk import RegexpParser
from nltk.tree import Tree

from newsmind.preprocess import ensure_nltk, tokenize


@dataclass(slots=True)
class SyntaxAnalysis:
    tokens: list[str]
    pos_tags: list[tuple[str, str]]
    noun_phrases: list[str]


_NP_GRAMMAR = r"""
NP: {<DT>?<JJ.*>*<NN.*>+}
    {<NNP>+}
"""


def analyze_syntax(text: str) -> SyntaxAnalysis:
    ensure_nltk()
    tokens = tokenize(text)
    if not tokens:
        return SyntaxAnalysis(tokens=[], pos_tags=[], noun_phrases=[])
    tagged = pos_tag(tokens)
    parser = RegexpParser(_NP_GRAMMAR)
    tree = parser.parse(tagged)
    phrases: list[str] = []
    for subtree in tree:
        if isinstance(subtree, Tree) and subtree.label() == "NP":
            phrases.append(" ".join(word for word, _tag in subtree.leaves()))
    return SyntaxAnalysis(tokens=tokens, pos_tags=tagged, noun_phrases=phrases)


def expand_query_with_noun_phrases(query: str) -> str:
    """Append extracted noun phrases to strengthen classical retrieval."""
    analysis = analyze_syntax(query)
    if not analysis.noun_phrases:
        return query
    extra = " ".join(analysis.noun_phrases)
    return f"{query} {extra}"
