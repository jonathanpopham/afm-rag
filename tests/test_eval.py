"""Milestone 5 tests for ingestion deduplication and retrieval evaluation."""

import hashlib
import json
import re
from pathlib import Path

import httpx
import numpy as np

from afm_rag.embeddings import OllamaEmbedder
from afm_rag.eval import FIXTURE_EVAL_CASES, evaluate_retrieval
from afm_rag.pipeline import ingest_path
from afm_rag.store import HybridStore

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "manuals"
FAKE_DIMS = 32


def _hash_embed(text: str, dims: int = FAKE_DIMS) -> list[float]:
    """Compute a deterministic normalized bag-of-tokens embedding for tests."""
    vec = np.zeros(dims, dtype=np.float32)
    for token in re.findall(r"[a-z0-9]+", text.lower()):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:2], "big") % dims
        vec[slot] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec /= norm
    return vec.tolist()


def _make_fake_embedder() -> tuple[httpx.Client, OllamaEmbedder]:
    """Build an OllamaEmbedder backed by MockTransport for offline pipeline tests."""

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content.decode("utf-8"))
        vectors = [_hash_embed(text) for text in payload["input"]]
        return httpx.Response(200, json={"embeddings": vectors})

    client = httpx.Client(transport=httpx.MockTransport(handler), base_url="http://localhost:11434")
    return client, OllamaEmbedder(batch_size=8, client=client)


def test_ingest_path_skips_unchanged_and_evaluates_fixtures(tmp_path: Path) -> None:
    """Verify ingest_path deduplicates unchanged manuals, supports force, and passes eval."""
    db_path = tmp_path / "pipeline_eval.db"
    client, embedder = _make_fake_embedder()

    with client:
        first_run = ingest_path(
            FIXTURES_DIR,
            db_path,
            embedder=embedder,
            embedding_dims=FAKE_DIMS,
        )
        assert first_run["manuals_indexed"] == 3
        assert first_run["manuals_skipped"] == 0
        assert first_run["chunks_indexed"] == 15

        second_run = ingest_path(
            FIXTURES_DIR,
            db_path,
            embedder=embedder,
            embedding_dims=FAKE_DIMS,
        )
        assert second_run == {
            "manuals_indexed": 0,
            "manuals_skipped": 3,
            "chunks_indexed": 0,
        }

        forced_run = ingest_path(
            FIXTURES_DIR,
            db_path,
            embedder=embedder,
            force=True,
            embedding_dims=FAKE_DIMS,
        )
        assert forced_run["manuals_indexed"] == 3
        assert forced_run["manuals_skipped"] == 0
        assert forced_run["chunks_indexed"] == 15

        with HybridStore(db_path, embedding_dims=FAKE_DIMS) as store:
            report = evaluate_retrieval(
                store=store,
                embedder=embedder,
                cases=FIXTURE_EVAL_CASES,
                top_k=3,
            )

        assert report["cases_total"] == 5
        assert report["hits"] == 5
        assert report["hit_at_k"] == 1.0
        assert report["mrr"] > 0.8
