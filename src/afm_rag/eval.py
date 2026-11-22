"""Retrieval evaluation metrics (Hit@K and MRR) on doctrine test cases."""

from collections.abc import Sequence
from dataclasses import dataclass

from afm_rag.embeddings import OllamaEmbedder
from afm_rag.store import HybridStore


@dataclass(frozen=True, slots=True)
class EvalCase:
    """Represent one labeled retrieval evaluation question."""

    question: str
    expected_doc_substr: str
    expected_section_substr: str


FIXTURE_EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        question="What ground area does an eight-digit grid coordinate locate?",
        expected_doc_substr="fm_3-25.26",
        expected_section_substr="Precision Levels",
    ),
    EvalCase(
        question="When do you add 180 degrees to convert an azimuth to a back azimuth?",
        expected_doc_substr="fm_3-25.26",
        expected_section_substr="Plotting Back Azimuths",
    ),
    EvalCase(
        question="In the MARCH sequence, what priority comes first during tactical field care?",
        expected_doc_substr="tc_4-02.1",
        expected_section_substr="MARCH Assessment Sequence",
    ),
    EvalCase(
        question="Where should a tourniquet never be placed on an injured limb?",
        expected_doc_substr="tc_4-02.1",
        expected_section_substr="Extremity Tourniquet Application",
    ),
    EvalCase(
        question="How do soldiers inside the kill zone react to a near ambush?",
        expected_doc_substr="atp_3-21.8",
        expected_section_substr="React to Near Ambush",
    ),
)


def evaluate_retrieval(
    store: HybridStore,
    embedder: OllamaEmbedder,
    cases: Sequence[EvalCase] = FIXTURE_EVAL_CASES,
    top_k: int = 3,
) -> dict[str, object]:
    """Compute Hit@K and Mean Reciprocal Rank across labeled retrieval cases."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not cases:
        return {
            "cases_total": 0,
            "hits": 0,
            "hit_at_k": 0.0,
            "mrr": 0.0,
            "details": [],
        }

    hits = 0
    total_rr = 0.0
    details: list[dict[str, object]] = []

    for case in cases:
        query_vec = embedder.embed_query(case.question)
        results = store.search_hybrid(case.question, query_vec, top_k=top_k)

        matched_rank: int | None = None
        doc_target = case.expected_doc_substr.lower()
        sec_target = case.expected_section_substr.lower()

        for rank, result in enumerate(results, start=1):
            if (
                doc_target in result.chunk.doc_id.lower()
                and sec_target in result.chunk.section_path.lower()
            ):
                matched_rank = rank
                break

        hit = matched_rank is not None
        reciprocal_rank = 1.0 / matched_rank if matched_rank is not None else 0.0
        if hit:
            hits += 1
        total_rr += reciprocal_rank

        top_section = results[0].chunk.section_path if results else ""
        details.append(
            {
                "question": case.question,
                "hit": hit,
                "reciprocal_rank": reciprocal_rank,
                "top_section": top_section,
            }
        )

    total = len(cases)
    return {
        "cases_total": total,
        "hits": hits,
        "hit_at_k": hits / total,
        "mrr": total_rr / total,
        "details": details,
    }
