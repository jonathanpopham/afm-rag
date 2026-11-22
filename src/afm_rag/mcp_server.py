"""Local stdio MCP server exposing afm-rag search and manual inspection tools."""

import os
from contextlib import ExitStack
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from afm_rag.embeddings import (
    DEFAULT_EMBED_DIMS,
    DEFAULT_EMBED_MODEL,
    DEFAULT_OLLAMA_URL,
    OllamaEmbedder,
)
from afm_rag.store import HybridStore


def _missing_db_payload(path: Path) -> dict[str, object]:
    """Return a structured error payload when the SQLite index file is missing."""
    return {
        "exists": False,
        "error": (
            f"Database file does not exist: {path}. "
            f"Run 'uv run afm-rag ingest <source> --db {path}' first."
        ),
    }


def create_mcp_server(
    db_path: Path | None = None,
    embedder: OllamaEmbedder | None = None,
    embedding_dims: int | None = None,
) -> MCPServer:
    """Create and configure the local MCPServer with afm-rag manual tools."""
    resolved_db = (
        db_path if db_path is not None else Path(os.environ.get("AFM_RAG_DB", "afm_rag.db"))
    )
    resolved_url = os.environ.get("AFM_RAG_OLLAMA_URL", DEFAULT_OLLAMA_URL)
    resolved_model = os.environ.get("AFM_RAG_EMBED_MODEL", DEFAULT_EMBED_MODEL)
    resolved_dims = (
        embedding_dims
        if embedding_dims is not None
        else int(os.environ.get("AFM_RAG_EMBED_DIMS", str(DEFAULT_EMBED_DIMS)))
    )

    server = MCPServer("afm-rag")

    @server.tool()
    def index_stats() -> dict[str, object]:
        """Return counts of indexed manuals, chunks, categories, and statuses."""
        if not resolved_db.is_file():
            return _missing_db_payload(resolved_db)

        with HybridStore(resolved_db, embedding_dims=resolved_dims) as store:
            return {"exists": True, "db_path": str(resolved_db), **store.stats()}

    @server.tool()
    def list_manuals(
        category: str | None = None,
        subcategory: str | None = None,
        status: str | None = None,
        limit: int = 50,
    ) -> dict[str, object]:
        """List indexed field manuals filtered by category, subcategory, and status."""
        if not resolved_db.is_file():
            return _missing_db_payload(resolved_db)

        with HybridStore(resolved_db, embedding_dims=resolved_dims) as store:
            manuals = store.list_manuals(
                category=category,
                subcategory=subcategory,
                status=status,
                limit=limit,
            )
        return {"manuals": manuals, "count": len(manuals)}

    @server.tool()
    def search_manuals(
        query: str,
        top_k: int = 5,
        category: str | None = None,
        subcategory: str | None = None,
        status: str | None = None,
    ) -> dict[str, object]:
        """Run hybrid vector and FTS5 keyword search across indexed field manuals."""
        if not resolved_db.is_file():
            return _missing_db_payload(resolved_db)

        with ExitStack() as stack:
            store = stack.enter_context(HybridStore(resolved_db, embedding_dims=resolved_dims))
            active_embedder = (
                embedder
                if embedder is not None
                else stack.enter_context(
                    OllamaEmbedder(
                        model=resolved_model,
                        base_url=resolved_url,
                        dimensions=resolved_dims,
                    )
                )
            )
            query_vec = active_embedder.embed_query(query)
            hits = store.search_hybrid(
                query=query,
                query_embedding=query_vec,
                top_k=top_k,
                category=category,
                subcategory=subcategory,
                status=status,
            )

        items = [
            {
                "chunk_id": hit.chunk.chunk_id,
                "doc_id": hit.chunk.doc_id,
                "title": hit.chunk.title,
                "designation": hit.chunk.designation,
                "status": hit.chunk.status,
                "edition_date": hit.chunk.edition_date,
                "category": hit.chunk.category,
                "subcategory": hit.chunk.subcategory,
                "section_path": hit.chunk.section_path,
                "score": hit.score,
                "vector_rank": hit.vector_rank,
                "lexical_rank": hit.lexical_rank,
                "text": hit.chunk.text,
            }
            for hit in hits
        ]
        return {"query": query, "count": len(items), "results": items}

    @server.tool()
    def read_manual_section(
        doc_id: str,
        section_query: str | None = None,
        max_chunks: int = 12,
    ) -> dict[str, object]:
        """Read ordered chunks from one manual, filtered optionally by section title."""
        if not resolved_db.is_file():
            return _missing_db_payload(resolved_db)

        with HybridStore(resolved_db, embedding_dims=resolved_dims) as store:
            chunks = store.get_section(
                doc_id=doc_id,
                section_query=section_query,
                max_chunks=max_chunks,
            )

        items = [
            {
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
                "section_path": chunk.section_path,
                "status": chunk.status,
                "designation": chunk.designation,
                "edition_date": chunk.edition_date,
                "text": chunk.text,
            }
            for chunk in chunks
        ]
        return {"doc_id": doc_id, "count": len(items), "chunks": items}

    return server


def main() -> None:
    """Start the local afm-rag MCP server over stdio transport."""
    create_mcp_server().run(transport="stdio")


if __name__ == "__main__":
    main()
