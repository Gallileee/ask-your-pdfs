"""Embedding model + Chroma vector store, and top-k semantic search over them."""

from dataclasses import dataclass
from functools import lru_cache

import chromadb

from . import config


@dataclass
class Hit:
    text: str
    source: str
    page: int
    similarity: float  # cosine similarity, 1.0 = identical

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


def search(question: str, k: int = config.TOP_K) -> list[Hit]:
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("The vector store is empty. Run `python ingest.py` first.")
    res = collection.query(query_embeddings=embed([config.QUERY_PREFIX + question]), n_results=k)
    return [
        Hit(text=doc, source=meta["source"], page=int(meta["page"]), similarity=1 - dist)
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
