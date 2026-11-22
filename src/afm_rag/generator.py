"""Grounded answer generation using local Ollama chat models."""

from collections.abc import Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Self

import httpx

from afm_rag.embeddings import DEFAULT_OLLAMA_URL
from afm_rag.models import SearchResult

DEFAULT_CHAT_MODEL = "inkling-2-mini:latest"

SYSTEM_PROMPT = (
    "Answer the question using only the provided field manual excerpts. "
    "Cite supporting excerpts inline with bracket numbers such as [1] or [2]. "
    "Do not invent procedures, numbers, or doctrine absent from the excerpts. "
    "State clearly when the excerpts lack information needed for an answer."
)


@dataclass(frozen=True, slots=True)
class RagAnswer:
    """Represent a grounded answer and its supporting manual citations."""

    question: str
    answer: str
    citations: list[SearchResult]
    model: str


def build_grounded_messages(
    question: str,
    results: Sequence[SearchResult],
) -> list[dict[str, str]]:
    """Format retrieved SearchResult chunks into grounded chat messages."""
    blocks: list[str] = []
    for index, result in enumerate(results, start=1):
        chunk = result.chunk
        header = (
            f"[{index}] Manual: {chunk.title} | "
            f"Section: {chunk.section_path} | "
            f"Chunk: {chunk.chunk_id}"
        )
        blocks.append(f"{header}\n{chunk.text}")

    context_text = "\n\n---\n\n".join(blocks)
    user_content = f"Excerpts:\n\n{context_text}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


class OllamaGenerator:
    """Local Ollama chat client for citation-grounded RAG answers."""

    def __init__(
        self,
        model: str = DEFAULT_CHAT_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        think: str | bool = "low",
        timeout: float = 180.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Initialize the Ollama chat client with model and thinking settings."""
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.think = think
        self._owns_client = client is None
        self._client = (
            client if client is not None else httpx.Client(base_url=self.base_url, timeout=timeout)
        )

    def __enter__(self) -> Self:
        """Enter context manager and return the generator instance."""
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
        """Close the HTTP client when owned by OllamaGenerator."""
        if self._owns_client:
            self._client.close()

    def answer(self, question: str, results: Sequence[SearchResult]) -> RagAnswer:
        """Generate a grounded RagAnswer from retrieved SearchResult excerpts."""
        if not results:
            return RagAnswer(
                question=question,
                answer="The local index returned no matching manual excerpts.",
                citations=[],
                model=self.model,
            )

        endpoint = "/api/chat" if self._client.base_url else f"{self.base_url}/api/chat"
        messages = build_grounded_messages(question, results)
        response = self._client.post(
            endpoint,
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
                "think": self.think,
            },
        )
        response.raise_for_status()
        payload = response.json()
        content = str(payload.get("message", {}).get("content", "")).strip()

        return RagAnswer(
            question=question,
            answer=content,
            citations=list(results),
            model=self.model,
        )
