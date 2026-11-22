"""Immutable data models for manuals and text chunks."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManualDocument:
    """Represent one loaded field manual document.

    Attributes:
        doc_id: Unique deterministic identifier for the manual.
        title: Human-readable title extracted from the manual.
        category: Top-level vault folder name.
        subcategory: Topic folder or archive stem name.
        source_path: Relative file path or archive member path.
        text: Complete Markdown text content of the manual.
    """

    # frozen=True prevents accidental mutation after ingestion.
    # slots=True eliminates per-instance __dict__ allocations across thousands of records.
    doc_id: str
    title: str
    category: str
    subcategory: str
    source_path: str
    text: str


@dataclass(frozen=True, slots=True)
class TextChunk:
    """Represent one retrievable chunk from a manual section.

    Attributes:
        chunk_id: Unique identifier formatted as ``{doc_id}::chunk_{index:04d}``.
        doc_id: Parent manual identifier.
        title: Parent manual title.
        category: Top-level vault folder name.
        subcategory: Topic folder or archive stem name.
        section_path: Breadcrumb string built from Markdown headings.
        chunk_index: Zero-based sequential index inside the parent manual.
        text: Chunk body text without extra heading wrapper.
    """

    # frozen=True guarantees thread-safe reads and hashable records.
    # slots=True reduces RAM overhead when indexing large manual collections.
    chunk_id: str
    doc_id: str
    title: str
    category: str
    subcategory: str
    section_path: str
    chunk_index: int
    text: str


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Represent one ranked retrieval hit from hybrid search.

    Attributes:
        chunk: Retrieved TextChunk record.
        score: Combined ranking score where higher values indicate stronger matches.
        vector_rank: One-based rank from vector search or ``None`` when absent.
        lexical_rank: One-based rank from FTS5 search or ``None`` when absent.
    """

    chunk: TextChunk
    score: float
    vector_rank: int | None = None
    lexical_rank: int | None = None


def format_chunk_for_embedding(chunk: TextChunk) -> str:
    """Format chunk metadata and body text for the embedding model."""
    return f"Manual: {chunk.title}\nSection: {chunk.section_path}\n\n{chunk.text}"
