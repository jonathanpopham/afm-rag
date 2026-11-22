"""Milestone 3 tests for NumPy cosine similarity and OllamaEmbedder batching."""

import json

import httpx
import numpy as np
import pytest

from afm_rag.embeddings import OllamaEmbedder, cosine_similarity, rank_top_k


def test_cosine_similarity_handles_standard_and_zero_vectors() -> None:
    """Verify cosine_similarity computes exact angles and returns zero without NaN for zero norms."""
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    docs = np.array(
        [
            [2.0, 0.0, 0.0],  # Identical direction -> 1.0
            [0.0, 3.0, 0.0],  # Orthogonal -> 0.0
            [-5.0, 0.0, 0.0],  # Opposite -> -1.0
            [0.0, 0.0, 0.0],  # Zero vector -> 0.0 (no NaN)
        ],
        dtype=np.float32,
    )

    scores = cosine_similarity(query, docs)

    assert scores.dtype == np.float32
    assert scores.shape == (4,)
    assert not np.isnan(scores).any()
    assert scores == pytest.approx([1.0, 0.0, -1.0, 0.0], abs=1e-6)


def test_cosine_similarity_handles_zero_query_and_empty_matrix() -> None:
    """Verify cosine_similarity handles zero query vectors and empty matrices safely."""
    zero_query = np.zeros(3, dtype=np.float32)
    docs = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)

    zero_scores = cosine_similarity(zero_query, docs)
    assert zero_scores == pytest.approx([0.0, 0.0])

    empty_docs = np.empty((0, 3), dtype=np.float32)
    empty_scores = cosine_similarity(np.array([1.0, 0.0, 0.0], dtype=np.float32), empty_docs)
    assert empty_scores.shape == (0,)
    assert empty_scores.dtype == np.float32


def test_cosine_similarity_validates_dimensions() -> None:
    """Verify cosine_similarity raises ValueError on invalid dimensions or column mismatches."""
    with pytest.raises(ValueError):
        cosine_similarity(np.ones((2, 2)), np.ones((2, 2)))
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones(3))
    with pytest.raises(ValueError):
        cosine_similarity(np.ones(3), np.ones((2, 4)))


def test_rank_top_k_sorts_clamps_and_validates() -> None:
    """Verify rank_top_k orders indices descending and clamps top_k to row count."""
    query = np.array([1.0, 1.0], dtype=np.float32)
    docs = np.array(
        [
            [0.0, 1.0],  # 0.707
            [-1.0, -1.0],  # -1.0
            [2.0, 2.0],  # 1.0
        ],
        dtype=np.float32,
    )

    ranked = rank_top_k(query, docs, top_k=2)
    assert len(ranked) == 2
    assert ranked[0][0] == 2
    assert ranked[0][1] == pytest.approx(1.0, abs=1e-6)
    assert ranked[1][0] == 0
    assert ranked[1][1] == pytest.approx(np.sqrt(0.5), abs=1e-6)

    clamped = rank_top_k(query, docs, top_k=10)
    assert [idx for idx, _ in clamped] == [2, 0, 1]

    empty_ranked = rank_top_k(query, np.empty((0, 2), dtype=np.float32), top_k=5)
    assert empty_ranked == []

    with pytest.raises(ValueError):
        rank_top_k(query, docs, top_k=0)


def test_ollama_embedder_batches_requests_and_retries_via_mock_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify OllamaEmbedder sends dimensions, batches inputs, and retries transient 500 errors."""
    monkeypatch.setattr("afm_rag.embeddings.time.sleep", lambda _: None)
    recorded_batches: list[list[str]] = []
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        assert request.url.path == "/api/embed"
        body = json.loads(request.content.decode("utf-8"))
        assert body["model"] == "qwen3-embedding:0.6b"
        assert body["dimensions"] == 512

        if attempts == 1:
            return httpx.Response(503, json={"error": "loading model"})

        batch_inputs = body["input"]
        recorded_batches.append(batch_inputs)

        embeddings = [
            [float(len(text)), 1.0, 0.5, float(idx)] for idx, text in enumerate(batch_inputs)
        ]
        return httpx.Response(200, json={"embeddings": embeddings})

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://localhost:11434") as client:
        embedder = OllamaEmbedder(batch_size=2, dimensions=512, max_retries=3, client=client)
        matrix = embedder.embed_texts(["a", "bb", "ccc", "dddd", "eeeee"])
        query_vec = embedder.embed_query("grid")

    assert attempts == 5
    assert recorded_batches == [["a", "bb"], ["ccc", "dddd"], ["eeeee"], ["grid"]]
    assert matrix.shape == (5, 4)
    assert matrix.dtype == np.float32
    assert matrix[0].tolist() == pytest.approx([1.0, 1.0, 0.5, 0.0])
    assert matrix[4].tolist() == pytest.approx([5.0, 1.0, 0.5, 0.0])
    assert query_vec.shape == (4,)
    assert query_vec.dtype == np.float32
    assert embedder.embed_texts([]).shape == (0, 0)
