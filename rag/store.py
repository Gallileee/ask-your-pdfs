"""Embedding model + Chroma vector store, and top-k search over them.

Three search modes (config.SEARCH_MODE): semantic search with embeddings ("vector"),
semantic + keyword (BM25) rankings merged with reciprocal rank fusion ("hybrid"), and
the candidates of both re-ordered by a cross-encoder ("rerank", the default).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache

import chromadb

from . import config


@dataclass
class Hit:
    text: str
    source: str
    page: int
    similarity: float  # cosine similarity to the question, 1.0 = identical

    @property
    def citation(self) -> str:
        return f"[{self.source}, p. {self.page}]"


@lru_cache(maxsize=1)
def _model():
    # Imported lazily: loading torch takes a few seconds.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    # Normalized vectors make cosine distance well-behaved (range 0..2).
    return _model().encode(texts, normalize_embeddings=True, batch_size=32).tolist()


@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=str(config.DB_DIR))


def get_collection():
    return _client().get_or_create_collection(config.COLLECTION, metadata={"hnsw:space": "cosine"})


def reset_collection() -> None:
    try:
        _client().delete_collection(config.COLLECTION)
    except Exception:  # collection doesn't exist yet
        pass


# ---------- keyword search (BM25) ----------

STOPWORDS = set(
    "a an and are as at be by does do for from has have how in is it its of on or that the this to "
    "was were what when where which who why with use uses used according".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


class BM25:
    """Okapi BM25 over a fixed list of documents (k1=1.5, b=0.75)."""

    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lengths = [sum(tf.values()) for tf in self.tfs]
        self.avg_len = sum(self.lengths) / max(len(docs), 1)
        df = Counter(term for tf in self.tfs for term in tf)
        n = len(docs)
        self.idf = {term: math.log(1 + (n - f + 0.5) / (f + 0.5)) for term, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, length in zip(self.tfs, self.lengths):
            norm = self.k1 * (1 - self.b + self.b * length / self.avg_len)
            out.append(sum(self.idf.get(t, 0) * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf))
        return out


@lru_cache(maxsize=1)
def _keyword_index() -> tuple[list[str], BM25]:
    # Built once per process from the chunks already in Chroma (a few hundred ms for
    # thousands of chunks). Restart long-running processes after re-ingesting.
    data = get_collection().get(include=["documents"])
    return data["ids"], BM25(data["documents"])


def keyword_ranking(question: str, n: int) -> list[str]:
    ids, bm25 = _keyword_index()
    scored = sorted(zip(bm25.scores(question), ids), reverse=True)
    return [chunk_id for score, chunk_id in scored[:n] if score > 0]


# ---------- search ----------


def _hits(ids: list[str], query_vec: list[float]) -> list[Hit]:
    """Load chunks by id, with their cosine similarity to the query, in the given order."""
    data = get_collection().get(ids=ids, include=["documents", "metadatas", "embeddings"])
    by_id = {
        i: Hit(text=doc, source=meta["source"], page=int(meta["page"]),
               similarity=float(sum(a * b for a, b in zip(vec, query_vec))))
        for i, doc, meta, vec in zip(data["ids"], data["documents"], data["metadatas"], data["embeddings"])
    }
    return [by_id[i] for i in ids]


def search(question: str, k: int = config.TOP_K, mode: str | None = None) -> list[Hit]:
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("The vector store is empty. Run `python ingest.py` first.")
    mode = mode or config.SEARCH_MODE

    if mode not in ("vector", "hybrid", "rerank"):
        raise SystemExit(f"Unknown SEARCH_MODE={mode!r}; use vector, hybrid or rerank.")
    query_vec = embed([config.QUERY_PREFIX + question])[0]

    pool = max(k * 4, 20)  # candidates each method contributes
    res = collection.query(query_embeddings=[query_vec], n_results=min(pool, collection.count()))
    vector_ids = res["ids"][0]
    if mode == "vector":
        return _hits(vector_ids[:k], query_vec)

    keyword_ids = keyword_ranking(question, pool)
    if mode == "rerank":
        candidates = _hits(list(dict.fromkeys(vector_ids + keyword_ids)), query_vec)
        scores = _reranker().predict([(question, h.text) for h in candidates])
        ranked = sorted(zip(scores, range(len(candidates))), reverse=True)
        return [candidates[i] for _, i in ranked[:k]]

    # Reciprocal rank fusion: a chunk ranked high by either method rises to the top;
    # 60 is the usual damping constant, so no single rank dominates.
    fused: Counter[str] = Counter()
    for ranking in (vector_ids, keyword_ids):
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] += 1 / (60 + rank)
    return _hits([chunk_id for chunk_id, _ in fused.most_common(k)], query_vec)


@lru_cache(maxsize=1)
def _reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(config.RERANK_MODEL)
