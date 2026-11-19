"""MCP server: lets an AI agent (Claude Desktop, Claude Code, Cursor, ...) search the
indexed PDFs and get answers with page-level citations.

    python mcp_server.py          # speaks MCP over stdio; normally started by the client

Tools:
  list_documents    which PDFs are indexed
  search_documents  the most relevant excerpts for a query, with citations and scores
  ask_documents     a grounded answer with [file, p. N] citations, or "I don't know"

`search_documents` needs no API key: the agent can read the excerpts and answer itself.
`ask_documents` runs the full pipeline, including the LLM call and the citation check.
"""

from typing import TypedDict

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from rag import config

server = MCPServer(
    name="ask-your-pdfs",
    instructions=(
        "Search a local collection of PDF documents. Prefer search_documents and cite excerpts as "
        "[file, p. N]. Use ask_documents for a ready-made grounded answer. Never cite a page that "
        "was not returned by a tool."
    ),
    log_level="WARNING",
)
READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


class Excerpt(TypedDict):
    citation: str
    source: str
    page: int
    similarity: float
    text: str


class Document(TypedDict):
    file: str
    chunks: int


class GroundedAnswer(TypedDict):
    answer: str
    citations: list[str]
    unsupported_citations: list[str]
    answered_from_documents: bool
    excerpts: list[Excerpt]


def _excerpt(hit) -> Excerpt:
    return Excerpt(
        citation=hit.citation, source=hit.source, page=hit.page, similarity=round(hit.similarity, 3), text=hit.text
    )


@server.tool(annotations=READ_ONLY)
def list_documents() -> list[Document]:
    """List the indexed PDF files and how many text chunks each one has."""
    from collections import Counter

    from rag.store import get_collection

    metas = get_collection().get(include=["metadatas"])["metadatas"]
    counts = Counter(m["source"] for m in metas)
    return [Document(file=name, chunks=n) for name, n in sorted(counts.items())]


@server.tool(annotations=READ_ONLY)
def search_documents(query: str, k: int = config.TOP_K) -> list[Excerpt]:
    """Find the k excerpts most relevant to `query` (hybrid keyword + semantic search,
    re-ranked). Each excerpt has a citation like [file.pdf, p. 3] to quote in answers.
    Similarity below ~0.5 means the documents probably don't cover the topic."""
    from rag.store import search

    return [_excerpt(h) for h in search(query, max(1, min(k, 20)))]


@server.tool(annotations=READ_ONLY)
def ask_documents(question: str, k: int = config.TOP_K) -> GroundedAnswer:
    """Answer `question` using only the indexed PDFs. Every factual sentence ends with a
    [file, p. N] citation; if the documents don't contain the answer, the answer is
    "I don't know based on the provided documents.". Calls the configured LLM."""
    from rag.answer import ask

    result = ask(question, max(1, min(k, 20)))
    return GroundedAnswer(
        answer=result.text,
        citations=result.citations,
        unsupported_citations=result.unsupported_citations,
        answered_from_documents=result.text != config.NOT_FOUND,
        excerpts=[_excerpt(h) for h in result.hits],
    )


if __name__ == "__main__":
    server.run("stdio")
