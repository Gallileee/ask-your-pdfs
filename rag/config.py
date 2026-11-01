"""Shared settings. Everything can be overridden from .env or the environment."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

PDF_DIR = Path(os.getenv("PDF_DIR", ROOT / "pdfs"))
DB_DIR = Path(os.getenv("DB_DIR", ROOT / "chroma_db"))
COLLECTION = "pdf_chunks"

# Small (33M params, 384-dim), CPU-friendly retrieval model. Noticeably better than
# all-MiniLM-L6-v2 on acronyms like "SBERT". Changing it requires re-running ingest.py.
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models expect this instruction in front of queries (not in front of passages).
QUERY_PREFIX = os.getenv("QUERY_PREFIX", "Represent this sentence for searching relevant passages: ")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 800))  # characters
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 150))  # characters

TOP_K = int(os.getenv("TOP_K", 5))
# Cosine similarity below this for the best hit means "nothing relevant in the corpus",
# so we answer "I don't know" without calling the LLM.
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", 0.5))

NOT_FOUND = "I don't know based on the provided documents."
