"""End-to-end ingestion, hybrid search, and grounded QA pipelines."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack
from itertools import islice
from pathlib import Path

import numpy as np

from afm_rag.chunker import chunk_document
from afm_rag.embeddings import DEFAULT_EMBED_DIMS, OllamaEmbedder
from afm_rag.generator import OllamaGenerator, RagAnswer
from afm_rag.loader import iter_vault_zip_manuals, load_markdown_dir
from afm_rag.models import ManualDocument, SearchResult, format_chunk_for_embedding
from afm_rag.store import HybridStore, compute_sha256


def _iter_source_manuals(source: Path, limit: int | None) -> Iterator[ManualDocument]:
    """Yield manuals from zip archives or Markdown directories up to limit."""
    if any(source.rglob("*.zip")):
        yield from iter_vault_zip_manuals(source, limit=limit)
        return

    docs = load_markdown_dir(source)
    if limit is not None:
        yield from islice(docs, max(0, limit))
        return
    yield from docs


def ingest_path(
    source: Path,
    db_path: Path,
    embedder: OllamaEmbedder | None = None,
    max_chars: int = 800,
    overlap_chars: int = 120,
    limit: int | None = None,
    force: bool = False,
    embedding_dims: int = DEFAULT_EMBED_DIMS,
    progress_callback: Callable[[int, str], None] | None = None,
) -> dict[str, int]:
    """Chunk, embed, and upsert manuals from a directory or vault zip tree."""
    manuals_indexed = 0
    manuals_skipped = 0
    chunks_indexed = 0

    with ExitStack() as stack:
        store = stack.enter_context(HybridStore(db_path, embedding_dims=embedding_dims))
        active_embedder = (
            embedder if embedder is not None else stack.enter_context(OllamaEmbedder())
        )

        for doc in _iter_source_manuals(source, limit=limit):
            content_sha256 = compute_sha256(doc.text)
            if not force and store.has_manual_hash(doc.doc_id, content_sha256):
                manuals_skipped += 1
                continue

            chunks = chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars)
            if chunks:
                texts = [format_chunk_for_embedding(chunk) for chunk in chunks]
                embeddings = active_embedder.embed_texts(texts)
            else:
                embeddings = np.empty((0, 0), dtype=np.float32)

            store.upsert_manual(doc, chunks, embeddings, content_sha256=content_sha256)
            manuals_indexed += 1
            chunks_indexed += len(chunks)
            if progress_callback is not None:
                progress_callback(manuals_indexed, doc.title)

    return {
        "manuals_indexed": manuals_indexed,
        "manuals_skipped": manuals_skipped,
        "chunks_indexed": chunks_indexed,
    }


def search_index(
    query: str,
    db_path: Path,
    top_k: int = 5,
    category: str | None = None,
    subcategory: str | None = None,
    embedder: OllamaEmbedder | None = None,
    embedding_dims: int = DEFAULT_EMBED_DIMS,
) -> list[SearchResult]:
    """Run hybrid vector and lexical retrieval against the local SQLite index."""
    with ExitStack() as stack:
        store = stack.enter_context(HybridStore(db_path, embedding_dims=embedding_dims))
        active_embedder = (
            embedder if embedder is not None else stack.enter_context(OllamaEmbedder())
        )
        query_vec = active_embedder.embed_query(query)
        return store.search_hybrid(
            query=query,
            query_embedding=query_vec,
            top_k=top_k,
            category=category,
            subcategory=subcategory,
        )


def ask_manuals(
    question: str,
    db_path: Path,
    top_k: int = 5,
    category: str | None = None,
    subcategory: str | None = None,
    think: str | bool = "low",
    embedder: OllamaEmbedder | None = None,
    generator: OllamaGenerator | None = None,
    embedding_dims: int = DEFAULT_EMBED_DIMS,
) -> RagAnswer:
    """Retrieve matching manual excerpts and generate a cited local answer."""
    with ExitStack() as stack:
        active_embedder = (
            embedder if embedder is not None else stack.enter_context(OllamaEmbedder())
        )
        active_generator = (
            generator
            if generator is not None
            else stack.enter_context(OllamaGenerator(think=think))
        )
        results = search_index(
            query=question,
            db_path=db_path,
            top_k=top_k,
            category=category,
            subcategory=subcategory,
            embedder=active_embedder,
            embedding_dims=embedding_dims,
        )
        return active_generator.answer(question, results)
