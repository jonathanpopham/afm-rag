"""Hybrid SQLite index combining sqlite-vec vector search, FTS5 BM25, and RRF."""

import hashlib
import re
import sqlite3
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

import numpy as np
import sqlite_vec

from afm_rag.models import ManualDocument, SearchResult, TextChunk

STOP_WORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "is",
        "are",
        "to",
        "of",
        "in",
        "on",
        "for",
        "and",
        "or",
        "how",
        "what",
        "when",
        "where",
        "do",
        "does",
        "i",
        "you",
        "we",
        "my",
        "your",
        "with",
        "from",
        "by",
        "at",
        "as",
        "be",
        "it",
        "this",
        "that",
    }
)

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def compute_sha256(text: str) -> str:
    """Return the hexadecimal SHA-256 digest for UTF-8 text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_fts5_query(query: str) -> str:
    """Convert raw user query text into a safe FTS5 OR query string."""
    raw_tokens = _TOKEN_RE.findall(query.lower())
    if not raw_tokens:
        return ""

    # Drop common stop words only when meaningful search terms remain.
    filtered = [token for token in raw_tokens if token not in STOP_WORDS]
    selected = filtered if filtered else raw_tokens

    unique_tokens = list(dict.fromkeys(selected))
    return " OR ".join(f'"{token}"' for token in unique_tokens)


def reciprocal_rank_fusion(
    vector_ids: Sequence[str],
    lexical_ids: Sequence[str],
    k: int = 60,
    vector_weight: float = 1.0,
    lexical_weight: float = 1.0,
) -> list[tuple[str, float]]:
    """Fuse vector and lexical ranked chunk IDs with Reciprocal Rank Fusion."""
    if k <= 0:
        raise ValueError("k must be positive.")

    scores: dict[str, float] = {}
    for rank, chunk_id in enumerate(vector_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + vector_weight / (k + rank)

    for rank, chunk_id in enumerate(lexical_ids, start=1):
        scores[chunk_id] = scores.get(chunk_id, 0.0) + lexical_weight / (k + rank)

    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))


class HybridStore:
    """SQLite storage engine supporting vector KNN, FTS5 keyword search, and RRF."""

    def __init__(self, db_path: Path | str, embedding_dims: int = 1024) -> None:
        """Open or initialize the hybrid SQLite database and virtual tables."""
        if embedding_dims <= 0:
            raise ValueError("embedding_dims must be positive.")

        self.db_path = str(db_path)
        self.embedding_dims = embedding_dims

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

        if self.db_path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self.conn.execute("PRAGMA foreign_keys = ON")

        self.conn.enable_load_extension(True)
        try:
            sqlite_vec.load(self.conn)
        finally:
            self.conn.enable_load_extension(False)

        self._create_schema()

    def _create_schema(self) -> None:
        """Create core tables, indexes, and virtual tables when missing."""
        with self.conn:
            self.conn.executescript(
                f"""
                CREATE TABLE IF NOT EXISTS manuals (
                    doc_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    content_sha256 TEXT NOT NULL,
                    chunk_count INTEGER NOT NULL,
                    indexed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                    chunk_id TEXT UNIQUE NOT NULL,
                    doc_id TEXT NOT NULL REFERENCES manuals(doc_id) ON DELETE CASCADE,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    subcategory TEXT NOT NULL,
                    section_path TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
                CREATE INDEX IF NOT EXISTS idx_chunks_category ON chunks(category, subcategory);

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                    chunk_rowid INTEGER PRIMARY KEY,
                    embedding float[{self.embedding_dims}] distance_metric=cosine
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    title,
                    section_path,
                    text,
                    tokenize='porter unicode61'
                );
                """
            )

    def __enter__(self) -> Self:
        """Enter context manager and return the HybridStore instance."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the SQLite connection when exiting a context block."""
        self.close()

    def close(self) -> None:
        """Close the SQLite connection."""
        self.conn.close()

    def has_manual_hash(self, doc_id: str, content_sha256: str) -> bool:
        """Return True when doc_id already matches content_sha256 in the database."""
        row = self.conn.execute(
            "SELECT content_sha256 FROM manuals WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return row is not None and row["content_sha256"] == content_sha256

    def _delete_manual_rows(self, doc_id: str) -> None:
        """Delete manual metadata, chunks, vector rows, and FTS rows inside a transaction."""
        rowid_rows = self.conn.execute(
            "SELECT rowid FROM chunks WHERE doc_id = ?",
            (doc_id,),
        ).fetchall()
        rowids = [(int(row["rowid"]),) for row in rowid_rows]
        if rowids:
            self.conn.executemany("DELETE FROM chunks_vec WHERE chunk_rowid = ?", rowids)
            self.conn.executemany("DELETE FROM chunks_fts WHERE rowid = ?", rowids)

        self.conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        self.conn.execute("DELETE FROM manuals WHERE doc_id = ?", (doc_id,))

    def delete_manual(self, doc_id: str) -> None:
        """Remove a manual and all associated index rows from the store."""
        with self.conn:
            self._delete_manual_rows(doc_id)

    def upsert_manual(
        self,
        doc: ManualDocument,
        chunks: Sequence[TextChunk],
        embeddings: np.ndarray,
        content_sha256: str | None = None,
    ) -> int:
        """Insert or replace one manual and its chunks across all SQLite index tables."""
        emb_arr = np.asarray(embeddings, dtype=np.float32)
        if chunks:
            if emb_arr.ndim != 2 or emb_arr.shape[0] != len(chunks):
                raise ValueError("embeddings row count must match chunks count.")
            if emb_arr.shape[1] != self.embedding_dims:
                raise ValueError(
                    f"Expected {self.embedding_dims} embedding dims, got {emb_arr.shape[1]}."
                )
        elif emb_arr.size != 0:
            raise ValueError("embeddings must be empty when chunks is empty.")

        digest = content_sha256 if content_sha256 is not None else compute_sha256(doc.text)
        indexed_at = datetime.now(UTC).isoformat()

        with self.conn:
            self._delete_manual_rows(doc.doc_id)
            self.conn.execute(
                """
                INSERT INTO manuals (
                    doc_id, title, category, subcategory, source_path,
                    content_sha256, chunk_count, indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.doc_id,
                    doc.title,
                    doc.category,
                    doc.subcategory,
                    doc.source_path,
                    digest,
                    len(chunks),
                    indexed_at,
                ),
            )

            for chunk, vector in zip(chunks, emb_arr, strict=True):
                cursor = self.conn.execute(
                    """
                    INSERT INTO chunks (
                        chunk_id, doc_id, title, category, subcategory,
                        section_path, chunk_index, text
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.title,
                        chunk.category,
                        chunk.subcategory,
                        chunk.section_path,
                        chunk.chunk_index,
                        chunk.text,
                    ),
                )
                rowid = int(cursor.lastrowid)
                self.conn.execute(
                    "INSERT INTO chunks_vec(chunk_rowid, embedding) VALUES (?, ?)",
                    (rowid, sqlite_vec.serialize_float32(vector.tolist())),
                )
                self.conn.execute(
                    "INSERT INTO chunks_fts(rowid, title, section_path, text) VALUES (?, ?, ?, ?)",
                    (rowid, chunk.title, chunk.section_path, chunk.text),
                )

        return len(chunks)

    def search_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        category: str | None = None,
        subcategory: str | None = None,
    ) -> list[tuple[str, float]]:
        """Run cosine KNN vector search via sqlite-vec and optional metadata filters."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        query_vec = np.asarray(query_embedding, dtype=np.float32)
        if query_vec.ndim != 1 or query_vec.shape[0] != self.embedding_dims:
            raise ValueError(f"Expected 1D query vector with {self.embedding_dims} dims.")

        total_chunks = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        if total_chunks == 0:
            return []

        # Over-fetch KNN candidates before metadata filtering so filtered results stay complete.
        has_filter = category is not None or subcategory is not None
        candidate_k = (
            min(total_chunks, max(top_k * 8, 100)) if has_filter else min(total_chunks, top_k)
        )

        where_clauses: list[str] = []
        params: list[object] = [sqlite_vec.serialize_float32(query_vec.tolist()), candidate_k]
        if category is not None:
            where_clauses.append("c.category = ?")
            params.append(category)
        if subcategory is not None:
            where_clauses.append("c.subcategory = ?")
            params.append(subcategory)

        filter_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(top_k)

        rows = self.conn.execute(
            f"""
            WITH vec_matches AS (
                SELECT chunk_rowid, distance
                FROM chunks_vec
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT c.chunk_id, v.distance
            FROM vec_matches AS v
            JOIN chunks AS c ON c.rowid = v.chunk_rowid
            {filter_sql}
            ORDER BY v.distance ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        results: list[tuple[str, float]] = []
        for row in rows:
            distance = row["distance"]
            similarity = 0.0 if distance is None else 1.0 - float(distance)
            results.append((str(row["chunk_id"]), similarity))
        return results

    def search_lexical(
        self,
        query: str,
        top_k: int = 10,
        category: str | None = None,
        subcategory: str | None = None,
    ) -> list[tuple[str, float]]:
        """Run BM25 full-text search via SQLite FTS5 and optional metadata filters."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        fts_query = build_fts5_query(query)
        if not fts_query:
            return []

        where_clauses = ["chunks_fts MATCH ?"]
        params: list[object] = [fts_query]
        if category is not None:
            where_clauses.append("c.category = ?")
            params.append(category)
        if subcategory is not None:
            where_clauses.append("c.subcategory = ?")
            params.append(subcategory)
        params.append(top_k)

        rows = self.conn.execute(
            f"""
            SELECT c.chunk_id, bm25(chunks_fts, 2.0, 1.5, 1.0) AS bm25_score
            FROM chunks_fts
            JOIN chunks AS c ON c.rowid = chunks_fts.rowid
            WHERE {" AND ".join(where_clauses)}
            ORDER BY bm25_score ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        # Negate FTS5 bm25() because SQLite returns more negative values for stronger matches.
        return [(str(row["chunk_id"]), float(-row["bm25_score"])) for row in rows]

    def search_hybrid(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = 5,
        category: str | None = None,
        subcategory: str | None = None,
        rrf_k: int = 60,
    ) -> list[SearchResult]:
        """Run vector and lexical searches, fuse via RRF, and return ranked SearchResults."""
        if top_k <= 0:
            raise ValueError("top_k must be positive.")

        candidate_k = max(top_k * 3, 20)
        vec_hits = self.search_vector(
            query_embedding,
            top_k=candidate_k,
            category=category,
            subcategory=subcategory,
        )
        lex_hits = self.search_lexical(
            query,
            top_k=candidate_k,
            category=category,
            subcategory=subcategory,
        )

        vec_ids = [chunk_id for chunk_id, _ in vec_hits]
        lex_ids = [chunk_id for chunk_id, _ in lex_hits]
        fused = reciprocal_rank_fusion(vec_ids, lex_ids, k=rrf_k)[:top_k]
        if not fused:
            return []

        vector_ranks = {chunk_id: rank for rank, chunk_id in enumerate(vec_ids, start=1)}
        lexical_ranks = {chunk_id: rank for rank, chunk_id in enumerate(lex_ids, start=1)}

        target_ids = [chunk_id for chunk_id, _ in fused]
        placeholders = ", ".join("?" for _ in target_ids)
        rows = self.conn.execute(
            f"""
            SELECT chunk_id, doc_id, title, category, subcategory,
                   section_path, chunk_index, text
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            target_ids,
        ).fetchall()

        chunks_by_id = {
            str(row["chunk_id"]): TextChunk(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                category=str(row["category"]),
                subcategory=str(row["subcategory"]),
                section_path=str(row["section_path"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
            )
            for row in rows
        }

        return [
            SearchResult(
                chunk=chunks_by_id[chunk_id],
                score=float(score),
                vector_rank=vector_ranks.get(chunk_id),
                lexical_rank=lexical_ranks.get(chunk_id),
            )
            for chunk_id, score in fused
            if chunk_id in chunks_by_id
        ]

    def list_manuals(
        self,
        category: str | None = None,
        subcategory: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, object]]:
        """List indexed manuals filtered optionally by category and subcategory."""
        if limit <= 0:
            raise ValueError("limit must be positive.")

        where_clauses: list[str] = []
        params: list[object] = []
        if category is not None:
            where_clauses.append("category = ?")
            params.append(category)
        if subcategory is not None:
            where_clauses.append("subcategory = ?")
            params.append(subcategory)

        where_sql = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.append(limit)

        rows = self.conn.execute(
            f"""
            SELECT doc_id, title, category, subcategory, source_path, chunk_count, indexed_at
            FROM manuals
            {where_sql}
            ORDER BY category ASC, subcategory ASC, title ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            {
                "doc_id": str(row["doc_id"]),
                "title": str(row["title"]),
                "category": str(row["category"]),
                "subcategory": str(row["subcategory"]),
                "source_path": str(row["source_path"]),
                "chunk_count": int(row["chunk_count"]),
                "indexed_at": str(row["indexed_at"]),
            }
            for row in rows
        ]

    def get_section(
        self,
        doc_id: str,
        section_query: str | None = None,
        max_chunks: int = 12,
    ) -> list[TextChunk]:
        """Return ordered chunks for one manual, filtered optionally by section substring."""
        if max_chunks <= 0:
            raise ValueError("max_chunks must be positive.")

        where_clauses = ["doc_id = ?"]
        params: list[object] = [doc_id]
        if section_query is not None and section_query.strip():
            where_clauses.append("LOWER(section_path) LIKE ?")
            params.append(f"%{section_query.strip().lower()}%")
        params.append(max_chunks)

        rows = self.conn.execute(
            f"""
            SELECT chunk_id, doc_id, title, category, subcategory,
                   section_path, chunk_index, text
            FROM chunks
            WHERE {" AND ".join(where_clauses)}
            ORDER BY chunk_index ASC
            LIMIT ?
            """,
            params,
        ).fetchall()

        return [
            TextChunk(
                chunk_id=str(row["chunk_id"]),
                doc_id=str(row["doc_id"]),
                title=str(row["title"]),
                category=str(row["category"]),
                subcategory=str(row["subcategory"]),
                section_path=str(row["section_path"]),
                chunk_index=int(row["chunk_index"]),
                text=str(row["text"]),
            )
            for row in rows
        ]

    def stats(self) -> dict[str, object]:
        """Return counts of indexed manuals, chunks, and categories."""
        manual_count = int(self.conn.execute("SELECT COUNT(*) FROM manuals").fetchone()[0])
        chunk_count = int(self.conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
        category_rows = self.conn.execute(
            "SELECT category, COUNT(*) AS count FROM manuals GROUP BY category ORDER BY category ASC"
        ).fetchall()
        categories = {str(row["category"]): int(row["count"]) for row in category_rows}
        return {
            "manual_count": manual_count,
            "chunk_count": chunk_count,
            "categories": categories,
        }
