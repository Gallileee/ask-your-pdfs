"""Unit tests for the BM25 keyword ranking. No model or vector store needed."""

from rag.store import BM25, tokenize


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What pooling strategy does SBERT use by default?") == ["pooling", "strategy", "sbert", "default"]


def test_document_with_the_rare_term_ranks_first():
    docs = [
        "SBERT is trained on NLI data.",
        "The default configuration is MEAN pooling for SBERT.",
        "BERT uses the CLS token.",
    ]
    scores = BM25(docs).scores("default pooling")
    assert max(range(len(docs)), key=scores.__getitem__) == 1


def test_unrelated_documents_score_zero():
    scores = BM25(["apples and pears", "cars and trucks"]).scores("quantum physics")
    assert scores == [0, 0]


def test_common_terms_weigh_less_than_rare_ones():
    docs = ["model model model rare"] + ["model text"] * 20
    bm25 = BM25(docs)
    assert bm25.idf["rare"] > bm25.idf["model"]
