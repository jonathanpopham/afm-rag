"""Structure-aware Markdown chunking skeleton for Milestone 2."""

from afm_rag.models import ManualDocument, TextChunk


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split Markdown text into ordered ``(section_path, body)`` pairs.

    Contract:
        - Recognize ATX headings starting with 1 to 6 ``#`` characters plus one space.
        - Track active headings with a stack of ``(level, heading_text)`` tuples.
        - Pop stack entries where ``entry_level >= level`` before pushing a heading.
        - Join active stack titles with ``" > "`` to form ``section_path``.
        - Label non-whitespace text before the first heading as ``"Preamble"``.
        - Emit ``(section_path, body.strip())`` only when ``body.strip()`` is non-empty.
    """
    # TODO(1): Create a heading stack `list[tuple[int, str]]` and a line buffer `list[str]`.
    # TODO(2): Write a helper or closure to flush buffered lines if `body.strip()` is non-empty.
    # TODO(3): Scan `markdown_text.splitlines()` and check each line for an ATX heading (`#`..`######` + ` `).
    # TODO(4): Flush the current section before updating the heading stack.
    # TODO(5): Pop entries from the stack while `stack[-1][0] >= level`, then push `(level, heading)`.
    # TODO(6): Flush any remaining buffered lines after the loop ends.
    raise NotImplementedError("Implement split_markdown_sections for Milestone 2.")


def chunk_document(
    doc: ManualDocument,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    """Convert one ManualDocument into bounded, section-isolated TextChunk objects.

    Contract:
        - Raise ``ValueError`` when ``max_chars <= 0`` or ``overlap_chars < 0``.
        - Raise ``ValueError`` when ``overlap_chars >= max_chars``.
        - Call ``split_markdown_sections(doc.text)``.
        - Keep ``section_body`` as one chunk when ``len(section_body) <= max_chars``.
        - Split longer ``section_body`` values into slices of length ``<= max_chars``.
        - Prefer breaking slices on ``\\n\\n`` or whitespace boundaries rather than mid-word.
        - Overlap consecutive slices inside one section by roughly ``overlap_chars`` characters.
        - Advance the slice start index on every step to prevent infinite loops.
        - Never merge text from two different sections into one ``TextChunk``.
        - Number chunks sequentially across the manual starting at ``0``.
        - Set ``chunk_id`` to ``f"{doc.doc_id}::chunk_{chunk_index:04d}"``.
        - Copy ``doc_id``, ``title``, ``category``, and ``subcategory`` from ``doc``.
    """
    # TODO(1): Validate `max_chars` and `overlap_chars` with guard clauses raising `ValueError`.
    # TODO(2): Call `split_markdown_sections(doc.text)` and iterate through each `(section_path, body)`.
    # TODO(3): Split long `body` strings using a window loop that prefers `\n\n` or whitespace cuts.
    # TODO(4): Compute the next `start` offset from `end - overlap_chars` while guaranteeing forward progress.
    # TODO(5): Build `TextChunk` records using sequential `chunk_index` values across all sections.
    raise NotImplementedError("Implement chunk_document for Milestone 2.")
