"""Milestone 2 verification tests for structure-aware Markdown chunking."""

from pathlib import Path

import pytest

from afm_rag.chunker import chunk_document, split_markdown_sections
from afm_rag.loader import load_markdown_dir
from afm_rag.models import ManualDocument

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "manuals"


def _longest_suffix_prefix_overlap(left: str, right: str) -> int:
    """Return length of the longest suffix of left that matches a prefix of right."""
    max_len = min(len(left), len(right))
    for size in range(max_len, 0, -1):
        if left.endswith(right[:size]):
            return size
    return 0


def test_split_sections_tracks_nested_heading_hierarchy() -> None:
    """Verify nested headings maintain correct breadcrumbs across sibling pops."""
    markdown = (
        "# FM 3-25.26\n\n"
        "## Grid Coordinates\n\n"
        "### Four-Digit Grid\n"
        "Locates a 1,000-meter square.\n\n"
        "### Six-Digit Grid\n"
        "Locates a 100-meter square.\n\n"
        "## Terrain Features\n"
        "Hilltop, ridge, valley, and saddle.\n"
    )

    sections = split_markdown_sections(markdown)

    assert sections == [
        (
            "FM 3-25.26 > Grid Coordinates > Four-Digit Grid",
            "Locates a 1,000-meter square.",
        ),
        (
            "FM 3-25.26 > Grid Coordinates > Six-Digit Grid",
            "Locates a 100-meter square.",
        ),
        (
            "FM 3-25.26 > Terrain Features",
            "Hilltop, ridge, valley, and saddle.",
        ),
    ]


def test_split_sections_handles_preamble_before_first_heading() -> None:
    """Verify text before the first Markdown heading receives the Preamble path."""
    markdown = (
        "Distribution statement: Approved for public release.\n\n"
        "# Chapter 1\n"
        "Primary doctrine body.\n"
    )

    sections = split_markdown_sections(markdown)

    assert sections == [
        ("Preamble", "Distribution statement: Approved for public release."),
        ("Chapter 1", "Primary doctrine body."),
    ]


def test_split_sections_skips_empty_container_headings() -> None:
    """Verify empty parent headings do not produce empty section records."""
    markdown = (
        "# Chapter 1\n"
        "## Section 1.1\n"
        "Body text for section 1.1.\n"
    )

    sections = split_markdown_sections(markdown)

    assert sections == [
        ("Chapter 1 > Section 1.1", "Body text for section 1.1."),
    ]


@pytest.mark.parametrize(
    ("max_chars", "overlap_chars"),
    [
        (0, 0),
        (-100, 0),
        (200, -1),
        (200, 200),
        (200, 250),
    ],
)
def test_chunk_document_validates_arguments(max_chars: int, overlap_chars: int) -> None:
    """Verify invalid max_chars or overlap_chars parameters raise ValueError."""
    doc = ManualDocument(
        doc_id="test/sample",
        title="Sample Manual",
        category="Doctrine",
        subcategory="General",
        source_path="Doctrine/General/sample.md",
        text="# Sample\n\nValid body text.",
    )

    with pytest.raises(ValueError):
        chunk_document(doc, max_chars=max_chars, overlap_chars=overlap_chars)


def test_chunk_document_preserves_metadata_and_deterministic_ids() -> None:
    """Verify chunk_document copies parent metadata and generates sequential chunk IDs."""
    doc = ManualDocument(
        doc_id="doctrine/medical/tc_4-02.1",
        title="TC 4-02.1 First Aid",
        category="Doctrine & Field Manuals",
        subcategory="Medical & TCCC",
        source_path="Doctrine & Field Manuals/Medical & TCCC/TC_4-02.1.md",
        text=(
            "# TC 4-02.1\n\n"
            "## MARCH\n"
            "Massive hemorrhage comes first.\n\n"
            "## Airway\n"
            "Insert a nasopharyngeal airway.\n"
        ),
    )

    chunks = chunk_document(doc, max_chars=800, overlap_chars=120)

    assert len(chunks) == 2
    assert chunks[0].chunk_id == "doctrine/medical/tc_4-02.1::chunk_0000"
    assert chunks[0].chunk_index == 0
    assert chunks[0].section_path == "TC 4-02.1 > MARCH"
    assert chunks[0].doc_id == doc.doc_id
    assert chunks[0].title == doc.title
    assert chunks[0].category == doc.category
    assert chunks[0].subcategory == doc.subcategory

    assert chunks[1].chunk_id == "doctrine/medical/tc_4-02.1::chunk_0001"
    assert chunks[1].chunk_index == 1
    assert chunks[1].section_path == "TC 4-02.1 > Airway"


def test_chunk_document_splits_long_section_with_overlap_and_section_isolation() -> None:
    """Verify long sections split with bounded length, overlap, and strict section isolation."""
    para_one = (
        "Alpha paragraph one describes coordinate grid alignment on military topographic maps. "
        "Operators locate the lower-left grid corner before reading easting values left to right."
    )
    para_two = (
        "Alpha paragraph two explains northing measurements from bottom to top on the grid square. "
        "Ten-meter precision requires eight digits recorded in balanced easting and northing pairs."
    )
    para_three = (
        "Alpha paragraph three covers magnetic declination adjustments before azimuth plotting. "
        "Verify every plotted azimuth against visible terrain features and contour intersections."
    )
    long_alpha_body = f"{para_one}\n\n{para_two}\n\n{para_three}"

    doc = ManualDocument(
        doc_id="doctrine/nav/fm_3-25.26",
        title="FM 3-25.26 Land Navigation",
        category="Doctrine",
        subcategory="Land Navigation",
        source_path="Doctrine/Land Navigation/FM_3-25.26.md",
        text=(
            "# FM 3-25.26\n\n"
            "## Alpha Section\n\n"
            f"{long_alpha_body}\n\n"
            "## Beta Section\n\n"
            "Beta unique token describes hilltop concentric contour loops.\n"
        ),
    )

    chunks = chunk_document(doc, max_chars=260, overlap_chars=60)

    alpha_chunks = [c for c in chunks if c.section_path == "FM 3-25.26 > Alpha Section"]
    beta_chunks = [c for c in chunks if c.section_path == "FM 3-25.26 > Beta Section"]

    assert len(alpha_chunks) >= 2
    assert len(beta_chunks) == 1

    for chunk in chunks:
        assert 0 < len(chunk.text) <= 260

    # Verify consecutive chunks inside Alpha Section share overlapping text.
    overlap_len = _longest_suffix_prefix_overlap(alpha_chunks[0].text, alpha_chunks[1].text)
    assert overlap_len >= 15

    # Verify Alpha and Beta sections never mix into one chunk.
    for alpha_chunk in alpha_chunks:
        assert "Beta unique token" not in alpha_chunk.text
    for beta_chunk in beta_chunks:
        assert "Alpha paragraph" not in beta_chunk.text


def test_chunk_document_on_all_fixture_manuals() -> None:
    """Verify chunk_document processes all fixture manuals without errors or oversize chunks."""
    docs = load_markdown_dir(FIXTURES_DIR)
    assert len(docs) == 3

    all_chunk_ids: set[str] = set()
    for doc in docs:
        chunks = chunk_document(doc, max_chars=800, overlap_chars=120)
        assert len(chunks) >= 3
        for idx, chunk in enumerate(chunks):
            assert chunk.chunk_index == idx
            assert chunk.text.strip() != ""
            assert len(chunk.text) <= 800
            assert chunk.section_path.strip() != ""
            assert chunk.chunk_id not in all_chunk_ids
            all_chunk_ids.add(chunk.chunk_id)
