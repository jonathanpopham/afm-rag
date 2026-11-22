"""Milestone 4 tests for HybridStore, sqlite-vec KNN, FTS5, and RRF."""

from pathlib import Path

import numpy as np
import pytest

from afm_rag.models import ManualDocument, TextChunk
from afm_rag.store import HybridStore, build_fts5_query, compute_sha256, reciprocal_rank_fusion


def test_build_fts5_query_normalizes_punctuation_and_stop_words() -> None:
    """Verify build_fts5_query handles publication codes, stop words, and empty queries."""
    assert build_fts5_query("FM 3-25.26") == '"fm" OR "3" OR "25" OR "26"'
    assert build_fts5_query("How do I apply a tourniquet?") == '"apply" OR "tourniquet"'
    assert build_fts5_query("to be or is") == '"to" OR "be" OR "or" OR "is"'
    assert build_fts5_query("??? --- !!!") == ""


def test_reciprocal_rank_fusion_combines_and_validates() -> None:
    """Verify reciprocal_rank_fusion ranks overlapping hits highest and rejects invalid k."""
    fused = reciprocal_rank_fusion(
        vector_ids=["c2", "c1", "c3"],
        lexical_ids=["c1", "c4", "c2"],
        k=60,
    )
    ids = [chunk_id for chunk_id, _ in fused]
    # c1 has ranks (2, 1) -> highest; c2 has ranks (1, 3) -> second.
    assert ids[0] == "c1"
    assert ids[1] == "c2"
    assert set(ids) == {"c1", "c2", "c3", "c4"}

    with pytest.raises(ValueError):
        reciprocal_rank_fusion(["c1"], ["c1"], k=0)


def test_hybrid_store_upsert_search_filter_and_replace(tmp_path: Path) -> None:
    """Verify HybridStore indexing, vector/lexical/hybrid searches, filters, and replacement."""
    db_path = tmp_path / "test_hybrid.db"

    nav_doc = ManualDocument(
        doc_id="doctrine/nav/fm_3-25.26",
        title="FM 3-25.26 Land Navigation",
        category="Doctrine",
        subcategory="Land Navigation",
        source_path="Doctrine/Land Navigation/FM_3-25.26.md",
        text="Eight-digit grid coordinate text.",
    )
    nav_chunks = [
        TextChunk(
            chunk_id="doctrine/nav/fm_3-25.26::chunk_0000",
            doc_id=nav_doc.doc_id,
            title=nav_doc.title,
            category=nav_doc.category,
            subcategory=nav_doc.subcategory,
            section_path="FM 3-25.26 > Grid Coordinates",
            chunk_index=0,
            text="An eight-digit grid coordinate locates a point within 10 meters.",
        ),
        TextChunk(
            chunk_id="doctrine/nav/fm_3-25.26::chunk_0001",
            doc_id=nav_doc.doc_id,
            title=nav_doc.title,
            category=nav_doc.category,
            subcategory=nav_doc.subcategory,
            section_path="FM 3-25.26 > Back Azimuth",
            chunk_index=1,
            text="Add 180 degrees when the azimuth measures under 180 degrees.",
        ),
    ]
    nav_emb = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    med_doc = ManualDocument(
        doc_id="doctrine/med/tc_4-02.1",
        title="TC 4-02.1 First Aid",
        category="Medical",
        subcategory="TCCC",
        source_path="Medical/TCCC/TC_4-02.1.md",
        text="MARCH and tourniquet text.",
    )
    med_chunks = [
        TextChunk(
            chunk_id="doctrine/med/tc_4-02.1::chunk_0000",
            doc_id=med_doc.doc_id,
            title=med_doc.title,
            category=med_doc.category,
            subcategory=med_doc.subcategory,
            section_path="TC 4-02.1 > MARCH",
            chunk_index=0,
            text="Massive hemorrhage control precedes airway management.",
        ),
    ]
    med_emb = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

    with HybridStore(db_path, embedding_dims=4) as store:
        assert store.upsert_manual(nav_doc, nav_chunks, nav_emb) == 2
        assert store.upsert_manual(med_doc, med_chunks, med_emb) == 1

        stats = store.stats()
        assert stats == {
            "manual_count": 2,
            "chunk_count": 3,
            "categories": {"Doctrine": 1, "Medical": 1},
        }
        assert store.has_manual_hash(nav_doc.doc_id, compute_sha256(nav_doc.text))
        assert not store.has_manual_hash(nav_doc.doc_id, "wrong_sha")

        manuals = store.list_manuals(category="Medical")
        assert len(manuals) == 1
        assert manuals[0]["doc_id"] == med_doc.doc_id

        section_chunks = store.get_section(nav_doc.doc_id, section_query="azimuth")
        assert len(section_chunks) == 1
        assert section_chunks[0].chunk_id == nav_chunks[1].chunk_id

        vec_hits = store.search_vector(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), top_k=2)
        assert vec_hits[0][0] == nav_chunks[0].chunk_id
        assert vec_hits[0][1] == pytest.approx(1.0, abs=1e-5)

        lex_hits = store.search_lexical("hemorrhage tourniquet", top_k=2)
        assert lex_hits[0][0] == med_chunks[0].chunk_id

        hybrid_hits = store.search_hybrid(
            "eight-digit grid coordinate",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            top_k=3,
        )
        assert hybrid_hits[0].chunk.chunk_id == nav_chunks[0].chunk_id
        assert hybrid_hits[0].vector_rank == 1
        assert hybrid_hits[0].lexical_rank == 1

        filtered_hits = store.search_hybrid(
            "eight-digit grid",
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            top_k=3,
            category="Medical",
        )
        assert len(filtered_hits) == 1
        assert filtered_hits[0].chunk.category == "Medical"

        # Re-upserting the same doc_id must cleanly replace existing rows across all tables.
        updated_doc = ManualDocument(
            doc_id=nav_doc.doc_id,
            title=nav_doc.title,
            category=nav_doc.category,
            subcategory=nav_doc.subcategory,
            source_path=nav_doc.source_path,
            text="Updated navigation text.",
        )
        updated_chunk = [
            TextChunk(
                chunk_id="doctrine/nav/fm_3-25.26::chunk_0000",
                doc_id=nav_doc.doc_id,
                title=nav_doc.title,
                category=nav_doc.category,
                subcategory=nav_doc.subcategory,
                section_path="FM 3-25.26 > Updated",
                chunk_index=0,
                text="Updated ridge contour reference.",
            )
        ]
        updated_emb = np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
        assert store.upsert_manual(updated_doc, updated_chunk, updated_emb) == 1

        assert store.stats()["chunk_count"] == 2
        vec_count = store.conn.execute("SELECT COUNT(*) FROM chunks_vec").fetchone()[0]
        fts_count = store.conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
        assert vec_count == 2
        assert fts_count == 2
