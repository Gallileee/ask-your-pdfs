"""Index every PDF in ./pdfs (or PDF_DIR) into the local Chroma store.

    python ingest.py
"""

import sys

from rag.ingest import ingest

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ingest()
