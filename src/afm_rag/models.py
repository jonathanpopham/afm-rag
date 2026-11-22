"""Immutable data models for manuals, text chunks, and search results."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ManualDocument:
    """Represent one loaded field manual document.

    Attributes:
        doc_id: Unique deterministic identifier for the manual.
        title: Human-readable title extracted from frontmatter or headings.
        category: Top-level vault folder name.
        subcategory: Topic folder or archive stem name.
        source_path: Relative file path or archive member path.
        text: Markdown body text without YAML frontmatter.
        status: Publication lifecycle status such as active or superseded.
        branch: Service branch such as army or joint.
        designation: Publication code such as FM 3-25.26.
        edition_date: Publication date string from frontmatter.
    """

    # frozen=True prevents accidental mutation after ingestion.
    # slots=True eliminates per-instance __dict__ allocations across thousands of records.
    doc_id: str
    title: str
    category: str
    subcategory: str
    source_path: str
    text: str
    status: str = "unknown"
    branch: str = ""
    designation: str = ""
    edition_date: str = ""


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
        status: Publication lifecycle status copied from ManualDocument.
        branch: Service branch copied from ManualDocument.
        designation: Publication code copied from ManualDocument.
        edition_date: Publication date copied from ManualDocument.
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
    status: str = "unknown"
    branch: str = ""
    designation: str = ""
    edition_date: str = ""


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
    lines = [f"Manual: {chunk.title}"]
    if chunk.designation and chunk.designation.lower() not in chunk.title.lower():
        lines.append(f"Designation: {chunk.designation}")
    lines.append(f"Section: {chunk.section_path}")
    return "\n".join(lines) + f"\n\n{chunk.text}"
