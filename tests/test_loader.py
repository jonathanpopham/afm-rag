"""Tests for Markdown directory and vault zip archive manual loaders."""

import zipfile
from pathlib import Path

from afm_rag.loader import iter_vault_zip_manuals, load_markdown_dir

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "manuals"


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


def test_iter_vault_zip_manuals_reads_md_skips_par2_and_honors_limit(tmp_path: Path) -> None:
    """Verify iter_vault_zip_manuals filters non-Markdown files and obeys limit."""
    category_dir = tmp_path / "Doctrine & Field Manuals"
    category_dir.mkdir(parents=True)

    # Create a dummy parity file to confirm the loader skips .par2 archives.
    (category_dir / "Small-Unit Tactics.zip.par2").write_bytes(b"PAR2_DUMMY_DATA")

    zip_path = category_dir / "Small-Unit Tactics.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr(
            "FM_1-01_Alpha.md",
            "# FM 1-01 Alpha Manual\n\nAlpha doctrine text.\n",
        )
        archive.writestr(
            "nested/FM_1-02_Bravo.md",
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
    assert docs[0].title == "FM 1-01 Alpha Manual"
    assert "Alpha doctrine text." in docs[0].text

    # Without an H1 heading, loader converts underscore stem to readable title.
    assert docs[1].category == "Doctrine & Field Manuals"
    assert docs[1].subcategory == "Small-Unit Tactics"
    assert docs[1].title == "FM 1-02 Bravo"
    assert "Bravo doctrine text" in docs[1].text
    assert docs[0].doc_id != docs[1].doc_id

    limited_docs = list(iter_vault_zip_manuals(tmp_path, limit=1))
    assert len(limited_docs) == 1
    assert limited_docs[0].doc_id == docs[0].doc_id
