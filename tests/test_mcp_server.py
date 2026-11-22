"""Milestone 6 tests for the local afm-rag MCPServer tools."""

import asyncio
import hashlib
import json
import re
from pathlib import Path

import httpx
import numpy as np

from afm_rag.embeddings import OllamaEmbedder
from afm_rag.mcp_server import create_mcp_server
from afm_rag.pipeline import ingest_path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "manuals"
FAKE_DIMS = 32


def _hash_embed(text: str, dims: int = FAKE_DIMS) -> list[float]:
    """Compute a deterministic normalized bag-of-tokens embedding for tests."""
    vec = np.zeros(dims, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:2], "big") % dims
        vec[slot] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.tolist()


def _make_fake_embedder() -> tuple[httpx.Client, OllamaEmbedder]:
    """Build an OllamaEmbedder backed by MockTransport for offline MCP tests."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        vectors = [_hash_embed(text) for text in payload["input"]]
        return httpx.Response(200, json={"embeddings": vectors})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    return client, OllamaEmbedder(batch_size=8, dimensions=FAKE_DIMS, client=client)


def test_mcp_server_registers_and_executes_all_tools(tmp_path: Path) -> None:
    """Verify MCPServer exposes all four tools and returns structured retrieval results."""
    db_path = tmp_path / "mcp_test.db"
    client, embedder = _make_fake_embedder()

    with client:
        ingest_path(FIXTURES_DIR, db_path, embedder=embedder, embedding_dims=FAKE_DIMS)
        server = create_mcp_server(db_path=db_path, embedder=embedder, embedding_dims=FAKE_DIMS)

        async def run_checks() -> None:
            tools = await server.list_tools()
            assert {tool.name for tool in tools} == {
                "index_stats",
                "list_manuals",
                "search_manuals",
                "read_manual_section",
            }

            stats_res = await server.call_tool("index_stats", {})
            assert stats_res.is_error is False
            assert stats_res.structured_content is not None
            assert stats_res.structured_content["exists"] is True
            assert stats_res.structured_content["manual_count"] == 3
            assert stats_res.structured_content["chunk_count"] == 15

            list_res = await server.call_tool("list_manuals", {"subcategory": "Land Navigation"})
            assert list_res.is_error is False
            assert list_res.structured_content is not None
            assert list_res.structured_content["count"] == 1
            nav_doc_id = list_res.structured_content["manuals"][0]["doc_id"]
            assert "fm_3-25.26" in nav_doc_id

            search_res = await server.call_tool(
                "search_manuals",
                {"query": "eight-digit grid coordinate", "top_k": 3},
            )
            assert search_res.is_error is False
            assert search_res.structured_content is not None
            assert search_res.structured_content["count"] >= 1
            top_hit = search_res.structured_content["results"][0]
            assert "fm_3-25.26" in top_hit["doc_id"]
            assert top_hit["status"] == "unknown"

            active_search = await server.call_tool(
                "search_manuals",
                {"query": "eight-digit grid coordinate", "top_k": 3, "status": "active"},
            )
            assert active_search.is_error is False
            assert active_search.structured_content is not None
            assert active_search.structured_content["count"] == 0

            section_res = await server.call_tool(
                "read_manual_section",
                {"doc_id": nav_doc_id, "section_query": "Precision Levels"},
            )
            assert section_res.is_error is False
            assert section_res.structured_content is not None
            assert section_res.structured_content["count"] == 3
            assert all(
                "Precision Levels" in chunk["section_path"]
                for chunk in section_res.structured_content["chunks"]
            )

        asyncio.run(run_checks())


def test_mcp_server_handles_missing_database_gracefully(tmp_path: Path) -> None:
    """Verify MCPServer reports missing database status without creating an empty file."""
    missing_db = tmp_path / "missing.db"
    server = create_mcp_server(db_path=missing_db, embedding_dims=FAKE_DIMS)

    async def check_missing() -> None:
        for tool_name, args in (
            ("index_stats", {}),
            ("list_manuals", {}),
            ("read_manual_section", {"doc_id": "any"}),
        ):
            res = await server.call_tool(tool_name, args)
            assert res.is_error is False
            assert res.structured_content is not None
            assert res.structured_content["exists"] is False
            assert "afm-rag ingest" in res.structured_content["error"]

    asyncio.run(check_missing())
    assert not missing_db.exists()
