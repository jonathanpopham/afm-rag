"""Structure-aware Markdown chunking for Army Field Manuals."""

import re

from afm_rag.models import ManualDocument, TextChunk


def _parse_heading(line: str) -> tuple[int, str] | None:
    """Parse an ATX heading line into ``(level, heading_text)`` or ``None``."""
    # Remove trailing whitespace while keeping leading indentation.
    clean_line = line.rstrip()

    # Reject lines without a leading hash or space.
    if not clean_line.startswith("#") or " " not in clean_line:
        return None

    # Count leading hash characters.
    level = 0
    while level < len(clean_line) and clean_line[level] == "#":
        level += 1

    # Reject headings deeper than six levels or missing a space separator.
    if level > 6 or level == len(clean_line) or clean_line[level] != " ":
        return None

    # Extract heading text and strip optional closing hash markers.
    heading_text = re.sub(r"\s+#+$", "", clean_line[level:].strip())
    if not heading_text:
        return None

    return level, heading_text


def split_markdown_sections(markdown_text: str) -> list[tuple[str, str]]:
    """Split Markdown text into ordered ``(section_path, body)`` pairs."""
    sections: list[tuple[str, str]] = []
    stack: list[tuple[int, str]] = []
    body_lines: list[str] = []

    def flush() -> None:
        # Flush buffered lines when non-empty body text exists.
        body = "\n".join(body_lines).strip()
        if body:
            section_path = " > ".join(title for _, title in stack) if stack else "Preamble"
            sections.append((section_path, body))
        body_lines.clear()

    for line in markdown_text.splitlines():
        parsed = _parse_heading(line)
        if parsed is not None:
            flush()
            level, heading_text = parsed
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading_text))
        else:
            body_lines.append(line)

    flush()
    return sections


def _split_section_body(body: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Split one section body into bounded slices with word-boundary overlap."""
    if len(body) <= max_chars:
        return [body]

    slices: list[str] = []
    start = 0
    text_len = len(body)

    while start < text_len:
        raw_end = min(start + max_chars, text_len)
        end = raw_end

        if raw_end < text_len:
            # Prefer paragraph, line, or space breaks in the upper half of the window.
            min_cut = start + max(1, max_chars // 2)
            for sep in ("\n\n", "\n", " "):
                boundary = body.rfind(sep, min_cut, raw_end)
                if boundary != -1:
                    end = boundary
                    break

        piece = body[start:end].strip()
        if piece:
            slices.append(piece)

        if end >= text_len:
            break

        # Start the next slice at an overlapping word boundary.
        next_start = max(start + 1, end - overlap_chars)
        if overlap_chars > 0:
            while next_start < end and not body[next_start - 1].isspace():
                next_start += 1
        while next_start < text_len and body[next_start].isspace():
            next_start += 1

        # Advance start when overlap adjustment makes zero forward progress.
        if next_start <= start:
            next_start = max(start + 1, end)
        start = next_start

    return slices


def chunk_document(
    doc: ManualDocument,
    max_chars: int = 800,
    overlap_chars: int = 120,
) -> list[TextChunk]:
    """Convert one ManualDocument into bounded, section-isolated TextChunk objects."""
    if max_chars <= 0:
        raise ValueError("max_chars must be positive.")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be non-negative.")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be less than max_chars.")

    chunks: list[TextChunk] = []
    # Keep chunks isolated per section and number chunks sequentially.
    for section_path, section_body in split_markdown_sections(doc.text):
        for piece in _split_section_body(section_body, max_chars, overlap_chars):
            chunk_index = len(chunks)
            chunks.append(
                TextChunk(
                    chunk_id=f"{doc.doc_id}::chunk_{chunk_index:04d}",
                    doc_id=doc.doc_id,
                    title=doc.title,
                    category=doc.category,
                    subcategory=doc.subcategory,
                    section_path=section_path,
                    chunk_index=chunk_index,
                    text=piece,
                )
            )

    return chunks
