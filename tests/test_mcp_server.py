"""The MCP server, exercised through a real MCP client over stdio.

Needs the index (`python ingest.py`); skipped otherwise. Doesn't call the LLM.
"""

import json
import sys
from pathlib import Path

import anyio
import pytest
from mcp.client.client import Client
from mcp.client.stdio import StdioServerParameters

ROOT = Path(__file__).resolve().parent.parent
pytestmark = pytest.mark.skipif(not (ROOT / "chroma_db").exists(), reason="run `python ingest.py` first")

SERVER = StdioServerParameters(command=sys.executable, args=[str(ROOT / "mcp_server.py")], cwd=str(ROOT))


def call(*steps):
    """Start the server, run `steps` (async functions taking the client) and return their results."""

    async def run():
        async with Client(SERVER, read_timeout_seconds=180) as client:
            return [await step(client) for step in steps]

    return anyio.run(run)


def payload(result):
    assert not result.is_error, result.content
    if result.structured_content is not None:
        content = result.structured_content
        return content.get("result", content) if isinstance(content, dict) else content
    return json.loads(result.content[0].text)


def test_server_exposes_the_three_read_only_tools():
    async def tools(client):
        return (await client.list_tools()).tools

    (listed,) = call(tools)
    by_name = {t.name: t for t in listed}
    assert set(by_name) == {"list_documents", "search_documents", "ask_documents"}
    assert all(t.annotations.read_only_hint for t in listed)
    assert "query" in by_name["search_documents"].input_schema["properties"]


def test_search_returns_cited_excerpts_and_lists_documents():
    async def search(client):
        return payload(await client.call_tool("search_documents", {"query": "default pooling strategy of SBERT", "k": 3}))

    async def documents(client):
        return payload(await client.call_tool("list_documents", {}))

    excerpts, docs = call(search, documents)
    assert len(excerpts) == 3
    first = excerpts[0]
    assert first["citation"] == f"[{first['source']}, p. {first['page']}]"
    assert any("default configuration is MEAN" in e["text"] for e in excerpts)
    assert {d["file"] for d in docs} >= {"sentence-bert.pdf", "self-rag.pdf"}
