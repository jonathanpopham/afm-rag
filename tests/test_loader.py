"""Tests for YAML frontmatter parsing, directory loading, and vault zip archive loaders."""

import zipfile
from pathlib import Path

from afm_rag.loader import iter_vault_zip_manuals, load_markdown_dir, parse_frontmatter

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "manuals"


def test_parse_frontmatter_strips_yaml_entities_and_boilerplate() -> None:
    """Verify parse_frontmatter extracts metadata, unescapes entities, and removes boilerplate."""
    raw = (
        "---\r\n"
        'id: "adp_3-19_2019-07"\r\n'
        'designation: "ADP 3-19 &amp; 3-09"\r\n'
        'title: "FIRES"\r\n'
        'edition_date: "2019-07-31"\r\n'
        'status: "active"\r\n'
        'branch: "army"\r\n'
        "---\r\n\r\n"
        "> Text-only Markdown of *ADP 3-19 FIRES*, from the T.REX ARMS Field Manual Vault.\r\n\r\n"
        "# ADP 3-19 — FIRES\r\n\r\n"
        "Fires doctrine body.\r\n"
    )

    meta, body = parse_frontmatter(raw)

    assert meta["designation"] == "ADP 3-19 & 3-09"
    assert meta["status"] == "active"
    assert meta["branch"] == "army"
    assert meta["edition_date"] == "2019-07-31"
    assert body.startswith("# ADP 3-19 — FIRES")
    assert "Text-only Markdown of" not in body
    assert "edition_date:" not in body


def test_parse_frontmatter_handles_plain_markdown_without_header() -> None:
    """Verify parse_frontmatter preserves plain Markdown when YAML fences are absent."""
    plain = "# Plain Manual\n\nBody text without frontmatter.\n"
    meta, body = parse_frontmatter(plain)
    assert meta == {}
    assert body == plain


def test_load_markdown_dir_loads_fixtures_with_metadata() -> None:
    """Verify load_markdown_dir loads all three fixture manuals with exact metadata."""
    docs = load_markdown_dir(FIXTURES_DIR)

    assert len(docs) == 3

    by_subcategory = {doc.subcategory: doc for doc in docs}
    assert set(by_subcategory.keys()) == {
        "Land Navigation",
        "Medical & TCCC",
        "Small-Unit Tactics",
    }

    land_nav = by_subcategory["Land Navigation"]
    assert land_nav.category == "Doctrine & Field Manuals"
    assert land_nav.title == "FM 3-25.26 Map Reading and Land Navigation (Excerpt)"
    assert land_nav.status == "unknown"
    assert "eight-digit grid coordinate" in land_nav.text
    assert land_nav.doc_id

    medical = by_subcategory["Medical & TCCC"]
    assert medical.category == "Doctrine & Field Manuals"
    assert medical.title == "TC 4-02.1 First Aid (Excerpt)"
    assert "MARCH sequence" in medical.text

    tactics = by_subcategory["Small-Unit Tactics"]
    assert tactics.category == "Doctrine & Field Manuals"
    assert tactics.title == "ATP 3-21.8 Infantry Platoon and Squad (Excerpt)"
    assert "Bounding Overwatch" in tactics.text


def test_iter_vault_zip_manuals_reads_md_filters_status_and_honors_limit(tmp_path: Path) -> None:
    """Verify iter_vault_zip_manuals parses frontmatter, skips parity files, and filters status."""
    category_dir = tmp_path / "Doctrine & Field Manuals"
    category_dir.mkdir(parents=True)

    # Create a dummy parity file to confirm the loader skips .par2 archives.
    (category_dir / "Small-Unit Tactics.zip.par2").write_bytes(b"PAR2_DUMMY_DATA")

    zip_path = category_dir / "Small-Unit Tactics.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "FM_1-01_Alpha.md",
            '---\ndesignation: "FM 1-01"\ntitle: "Alpha Manual"\nstatus: "superseded"\n---\n\n'
            "# FM 1-01 Alpha Manual\n\nAlpha doctrine text.\n",
        )
        archive.writestr(
            "nested/FM_1-02_Bravo.md",
            '---\ndesignation: "FM 1-02"\nstatus: "active"\nbranch: "army"\n---\n\n'
            "Bravo doctrine text without H1 heading.\n",
        )
        archive.writestr(
            "README.txt",
            "Non-markdown companion file that loader must ignore.\n",
        )

    docs = list(iter_vault_zip_manuals(tmp_path))
    assert len(docs) == 2

    assert docs[0].category == "Doctrine & Field Manuals"
    assert docs[0].subcategory == "Small-Unit Tactics"
    assert docs[0].title == "FM 1-01 — Alpha Manual"
    assert docs[0].status == "superseded"
    assert docs[0].designation == "FM 1-01"
    assert "status:" not in docs[0].text
    assert "Alpha doctrine text." in docs[0].text

    # Without a frontmatter title or H1 heading, loader converts underscore stem to readable title.
    assert docs[1].category == "Doctrine & Field Manuals"
    assert docs[1].subcategory == "Small-Unit Tactics"
    assert docs[1].title == "FM 1-02 Bravo"
    assert docs[1].status == "active"
    assert docs[1].branch == "army"
    assert "Bravo doctrine text" in docs[1].text
    assert docs[0].doc_id != docs[1].doc_id

    active_only = list(iter_vault_zip_manuals(tmp_path, status_filter="active"))
    assert len(active_only) == 1
    assert active_only[0].doc_id == docs[1].doc_id

    limited_docs = list(iter_vault_zip_manuals(tmp_path, limit=1))
    assert len(limited_docs) == 1
    assert limited_docs[0].doc_id == docs[0].doc_id
