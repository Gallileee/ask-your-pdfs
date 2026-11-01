"""PDF -> text (per page) -> overlapping chunks -> embeddings -> Chroma."""

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from . import config
from .store import embed, get_collection, reset_collection


@dataclass
class Chunk:
    id: str
    text: str
    source: str  # PDF filename
    page: int  # 1-based, as printed in a PDF viewer


def clean(text: str) -> str:
    # NFKC turns ligatures like "ﬁ" into "fi"; then re-join words hyphenated at line ends
    # and collapse all whitespace so chunk sizes are meaningful.
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    return re.sub(r"\s+", " ", text).strip()


def split_text(text: str, size: int = config.CHUNK_SIZE, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Pack sentences into chunks of ~`size` chars; each chunk starts with the last
    ~`overlap` chars of whole sentences from the previous one."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # A "sentence" longer than a chunk (tables, references) is cut on word boundaries.
    pieces: list[str] = []
    for s in sentences:
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > 0 else size
            pieces.append(s[:cut])
            s = s[cut:].lstrip()
        if s:
            pieces.append(s)

    chunks: list[str] = []
    current: list[str] = []
    for piece in pieces:
        if current and len(" ".join(current + [piece])) > size:
            chunks.append(" ".join(current))
            tail: list[str] = []
            for prev in reversed(current):
                if len(" ".join([prev] + tail)) > overlap:
                    break
                tail.insert(0, prev)
            if not tail:  # last sentence alone is longer than the overlap: take its last words
                last = current[-1]
                tail = [last[last.find(" ", len(last) - overlap) + 1 :]]
            current = tail
        current.append(piece)
    if current:
        chunks.append(" ".join(current))
    return chunks


def chunk_pdf(path: Path) -> list[Chunk]:
    reader = PdfReader(path)
    chunks: list[Chunk] = []
    # Chunks never cross a page boundary, so every chunk has exactly one page to cite.
    for page_no, page in enumerate(reader.pages, start=1):
        text = clean(page.extract_text() or "")
        if len(text) < 30:  # blank or image-only page
            continue
        for i, part in enumerate(split_text(text)):
            chunks.append(Chunk(f"{path.name}:p{page_no}:c{i}", part, path.name, page_no))
    return chunks


def ingest(pdf_dir: Path = config.PDF_DIR) -> None:
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {pdf_dir}")

    # Rebuilding from scratch keeps the index in sync when PDFs are removed or changed.
    reset_collection()
    collection = get_collection()

    total = 0
    for pdf in pdfs:
        chunks = chunk_pdf(pdf)
        if not chunks:
            print(f"  {pdf.name}: no extractable text (scanned PDF?), skipped")
            continue
        collection.add(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embed([c.text for c in chunks]),
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
        )
        pages = len({c.page for c in chunks})
        print(f"  {pdf.name}: {pages} pages -> {len(chunks)} chunks")
        total += len(chunks)

    print(f"Indexed {total} chunks from {len(pdfs)} PDFs into {config.DB_DIR}")
