"""End-to-end ingestion, hybrid search, and grounded QA pipelines."""

import sqlite3
import sys
import zipfile
from collections.abc import Callable, Iterator
from contextlib import ExitStack
from pathlib import Path

import httpx
import numpy as np

from afm_rag.chunker import chunk_document
from afm_rag.embeddings import DEFAULT_EMBED_DIMS, DEFAULT_EMBED_MODEL, OllamaEmbedder
from afm_rag.generator import OllamaGenerator, RagAnswer
from afm_rag.loader import iter_vault_zip_manuals, load_markdown_dir
from afm_rag.models import ManualDocument, SearchResult, format_chunk_for_embedding
from afm_rag.store import HybridStore, compute_sha256


def _iter_source_manuals(
    source: Path,
    limit: int | None = None,
    status_filter: str | None = None,
    category_filter: str | None = None,
    active_first: bool = True,
) -> Iterator[ManualDocument]:
    """Yield manuals from zip archives or directories with optional active-first ordering."""
    if any(source.rglob("*.zip")):
        if active_first and status_filter is None:
            yielded = 0
            seen_ids: set[str] = set()

            for doc in iter_vault_zip_manuals(
                source,
                limit=limit,
                status_filter="active",
                category_filter=category_filter,
            ):
                seen_ids.add(doc.doc_id)
                yield doc
                yielded += 1

            if limit is not None and yielded >= limit:
                return

            for doc in iter_vault_zip_manuals(
                source,
                limit=None,
                status_filter=None,
                category_filter=category_filter,
            ):
                if doc.doc_id in seen_ids:
                    continue
                yield doc
                yielded += 1
                if limit is not None and yielded >= limit:
                    return
            return

        yield from iter_vault_zip_manuals(
            source,
            limit=limit,
            status_filter=status_filter,
            category_filter=category_filter,
        )
        return

    docs = load_markdown_dir(source)
    target_status = status_filter.strip().lower() if status_filter else None
    target_category = category_filter.strip().lower() if category_filter else None
    yielded = 0

    for doc in docs:
        if target_status is not None and doc.status.lower() != target_status:
            continue
        if target_category is not None and doc.category.lower() != target_category:
            continue
        yield doc
        yielded += 1
        if limit is not None and yielded >= limit:
            return


def ingest_path(
    source: Path,
    db_path: Path,
    embedder: OllamaEmbedder | None = None,
    max_chars: int = 800,
    overlap_chars: int = 120,
    limit: int | None = None,
    force: bool = False,
    embedding_dims: int = DEFAULT_EMBED_DIMS,
    embed_model: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
    status_filter: str | None = None,
    category_filter: str | None = None,
    active_first: bool = True,
    progress_callback: Callable[[dict[str, int], str], None] | None = None,
) -> dict[str, int]:
    """Chunk, embed, and upsert manuals from a directory or vault zip tree."""
    manuals_indexed = 0
    manuals_skipped = 0
    manuals_failed = 0
    chunks_indexed = 0

    with ExitStack() as stack:
        store = stack.enter_context(HybridStore(db_path, embedding_dims=embedding_dims))
        active_embedder = (
            embedder
            if embedder is not None
            else stack.enter_context(
                OllamaEmbedder(
                    model=embed_model,
                    batch_size=batch_size,
                    dimensions=embedding_dims,
                )
            )
        )

        for doc in _iter_source_manuals(
            source,
            limit=limit,
            status_filter=status_filter,
            category_filter=category_filter,
            active_first=active_first,
        ):
            try:
                content_sha256 = compute_sha256(doc.text)
                if not force and store.has_manual_hash(doc.doc_id, content_sha256):
                    manuals_skipped += 1
                    if progress_callback is not None and manuals_skipped % 250 == 0:
                        progress_callback(
                            {
                                "indexed": manuals_indexed,
                                "skipped": manuals_skipped,
                                "failed": manuals_failed,
                                "chunks": chunks_indexed,
                            },
                            f"Skipped {manuals_skipped} unchanged manuals",
                        )
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
                    progress_callback(
                        {
                            "indexed": manuals_indexed,
                            "skipped": manuals_skipped,
                            "failed": manuals_failed,
                            "chunks": chunks_indexed,
                        },
                        doc.title,
                    )
            except (httpx.HTTPError, sqlite3.Error, ValueError, OSError, zipfile.BadZipFile) as exc:
                manuals_failed += 1
                print(
                    f"WARN: Failed to ingest {doc.source_path} ({doc.doc_id}): {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    return {
        "manuals_indexed": manuals_indexed,
        "manuals_skipped": manuals_skipped,
        "manuals_failed": manuals_failed,
        "chunks_indexed": chunks_indexed,
    }


def search_index(
    query: str,
    db_path: Path,
    top_k: int = 5,
    category: str | None = None,
    subcategory: str | None = None,
    status: str | None = None,
    embedder: OllamaEmbedder | None = None,
    embedding_dims: int = DEFAULT_EMBED_DIMS,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> list[SearchResult]:
    """Run hybrid vector and lexical retrieval against the local SQLite index."""
    with ExitStack() as stack:
        store = stack.enter_context(HybridStore(db_path, embedding_dims=embedding_dims))
        active_embedder = (
            embedder
            if embedder is not None
            else stack.enter_context(OllamaEmbedder(model=embed_model, dimensions=embedding_dims))
        )
        query_vec = active_embedder.embed_query(query)
        return store.search_hybrid(
            query=query,
            query_embedding=query_vec,
            top_k=top_k,
            category=category,
            subcategory=subcategory,
            status=status,
        )


def ask_manuals(
    question: str,
    db_path: Path,
    top_k: int = 5,
    category: str | None = None,
    subcategory: str | None = None,
    status: str | None = None,
    think: str | bool = "low",
    embedder: OllamaEmbedder | None = None,
    generator: OllamaGenerator | None = None,
    embedding_dims: int = DEFAULT_EMBED_DIMS,
    embed_model: str = DEFAULT_EMBED_MODEL,
) -> RagAnswer:
    """Retrieve matching manual excerpts and generate a cited local answer."""
    with ExitStack() as stack:
        active_embedder = (
            embedder
            if embedder is not None
            else stack.enter_context(OllamaEmbedder(model=embed_model, dimensions=embedding_dims))
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
            status=status,
            embedder=active_embedder,
            embedding_dims=embedding_dims,
            embed_model=embed_model,
        )
        return active_generator.answer(question, results)
