"""Local Ollama embedding client and NumPy cosine similarity utilities."""

import time
from collections.abc import Sequence
from types import TracebackType
from typing import Self

import httpx
import numpy as np

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "qwen3-embedding:0.6b"
DEFAULT_EMBED_DIMS = 512


def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between one query vector and a matrix of document vectors."""
    query_arr = np.asarray(query_vec)
    doc_arr = np.asarray(doc_matrix)

    if query_arr.ndim != 1:
        raise ValueError("query_vec must be a 1D array.")
    if doc_arr.ndim != 2:
        raise ValueError("doc_matrix must be a 2D array.")
    if doc_arr.shape[1] != query_arr.shape[0]:
        raise ValueError(
            f"Dimension mismatch: query_vec has {query_arr.shape[0]} dims, "
            f"doc_matrix has {doc_arr.shape[1]} dims."
        )
    if doc_arr.shape[0] == 0:
        return np.empty((0,), dtype=np.float32)

    # Calculate norms and dot products in float64 to avoid precision loss and NaN values.
    q = query_arr.astype(np.float64, copy=False)
    docs = doc_arr.astype(np.float64, copy=False)
    query_norm = float(np.linalg.norm(q))
    if query_norm == 0.0:
        return np.zeros(docs.shape[0], dtype=np.float32)

    doc_norms = np.linalg.norm(docs, axis=1)
    dots = docs @ q
    sims = np.zeros(docs.shape[0], dtype=np.float64)
    nonzero = doc_norms > 0.0
    sims[nonzero] = dots[nonzero] / (doc_norms[nonzero] * query_norm)
    return np.clip(sims, -1.0, 1.0).astype(np.float32)


def rank_top_k(
    query_vec: np.ndarray,
    doc_matrix: np.ndarray,
    top_k: int = 5,
) -> list[tuple[int, float]]:
    """Return the top-k ``(row_index, score)`` pairs sorted by descending cosine similarity."""
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    scores = cosine_similarity(query_vec, doc_matrix)
    if len(scores) == 0:
        return []

    limit = min(top_k, len(scores))
    ranked_indices = np.argsort(-scores, kind="stable")[:limit]
    return [(int(idx), float(scores[idx])) for idx in ranked_indices]


class OllamaEmbedder:
    """Batch client for generating dense text embeddings from local Ollama."""

    def __init__(
        self,
        model: str = DEFAULT_EMBED_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        batch_size: int = 64,
        dimensions: int | None = DEFAULT_EMBED_DIMS,
        max_retries: int = 3,
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Ollama embedder with batching, MRL dims, and retry settings."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if dimensions is not None and dimensions <= 0:
            raise ValueError("dimensions must be positive.")
        if max_retries <= 0:
            raise ValueError("max_retries must be positive.")

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size
        self.dimensions = dimensions
        self.max_retries = max_retries
        self._owns_client = client is None
        self._client = (
            client if client is not None else httpx.Client(base_url=self.base_url, timeout=timeout)
        )

    def __enter__(self) -> Self:
        """Enter context manager and return the embedder instance."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the internal HTTP client when exiting a context block."""
        self.close()

    def close(self) -> None:
        """Close the HTTP client when owned by OllamaEmbedder."""
        if self._owns_client:
            self._client.close()

    def _post_embed_with_retry(self, endpoint: str, batch: list[str]) -> list[list[float]]:
        """Send one /api/embed request with retries on transient transport or 5xx errors."""
        payload: dict[str, object] = {"model": self.model, "input": batch}
        if self.dimensions is not None:
            payload["dimensions"] = self.dimensions

        for attempt in range(self.max_retries):
            try:
                response = self._client.post(endpoint, json=payload)
                if response.status_code >= 500 and attempt < self.max_retries - 1:
                    time.sleep(float(2**attempt))
                    continue
                response.raise_for_status()
                body = response.json()
                vectors = body.get("embeddings")
                if not isinstance(vectors, list) or len(vectors) != len(batch):
                    raise ValueError("Ollama /api/embed returned an unexpected embeddings count.")
                return vectors
            except httpx.TransportError:
                if attempt >= self.max_retries - 1:
                    raise
                time.sleep(float(2**attempt))

        raise RuntimeError("Unreachable retry state in _post_embed_with_retry.")

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Embed a sequence of text strings in batches and return a float32 matrix."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        endpoint = "/api/embed" if self._client.base_url else f"{self.base_url}/api/embed"
        batches: list[np.ndarray] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            vectors = self._post_embed_with_retry(endpoint, batch)
            batches.append(np.asarray(vectors, dtype=np.float32))

        return np.vstack(batches).astype(np.float32, copy=False)

    def embed_query(self, query: str) -> np.ndarray:
        """Embed one search query string into a 1D float32 vector."""
        return self.embed_texts([query])[0]
