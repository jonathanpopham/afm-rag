"""Milestone 5 tests for grounded prompt formatting and OllamaGenerator."""

import json

import httpx

from afm_rag.generator import SYSTEM_PROMPT, OllamaGenerator, build_grounded_messages
from afm_rag.models import SearchResult, TextChunk


def _sample_results() -> list[SearchResult]:
    """Create sample SearchResult fixtures for generator tests."""
    chunk = TextChunk(
        chunk_id="doctrine/nav/fm_3-25.26::chunk_0001",
        doc_id="doctrine/nav/fm_3-25.26",
        title="FM 3-25.26 Map Reading and Land Navigation",
        category="Doctrine & Field Manuals",
        subcategory="Land Navigation",
        section_path="FM 3-25.26 > Grid Coordinates > Precision Levels",
        chunk_index=1,
        text="An eight-digit grid coordinate locates a point within a 10-meter square.",
    )
    return [SearchResult(chunk=chunk, score=0.032, vector_rank=1, lexical_rank=1)]


def test_build_grounded_messages_formats_numbered_citations() -> None:
    """Verify build_grounded_messages includes system grounding prompt and numbered excerpts."""
    results = _sample_results()
    messages = build_grounded_messages(
        "What ground area does an eight-digit grid coordinate locate?",
        results,
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[1]["role"] == "user"
    assert "[1] Manual: FM 3-25.26 Map Reading and Land Navigation" in messages[1]["content"]
    assert "Section: FM 3-25.26 > Grid Coordinates > Precision Levels" in messages[1]["content"]
    assert "10-meter square" in messages[1]["content"]
    assert (
        "Question: What ground area does an eight-digit grid coordinate locate?"
        in (messages[1]["content"])
    )


def test_ollama_generator_handles_empty_results_without_http_call() -> None:
    """Verify empty SearchResult input returns refusal message without calling Ollama."""

    def fail_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"Unexpected HTTP request: {request.url}")

    transport = httpx.MockTransport(fail_handler)
    with httpx.Client(transport=transport, base_url="http://localhost:11434") as client:
        generator = OllamaGenerator(client=client)
        rag_answer = generator.answer("Any question?", [])

    assert "no matching manual excerpts" in rag_answer.answer.lower()
    assert rag_answer.citations == []
    assert rag_answer.model == "inkling-2-mini:latest"


def test_ollama_generator_posts_chat_request_via_mock_transport() -> None:
    """Verify OllamaGenerator sends /api/chat payload and attaches citations."""
    captured_payloads: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/chat"
        payload = json.loads(request.content.decode("utf-8"))
        captured_payloads.append(payload)
        return httpx.Response(
            200,
            json={"message": {"role": "assistant", "content": "It locates a 10-meter square [1]."}},
        )

    results = _sample_results()
    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport, base_url="http://localhost:11434") as client:
        generator = OllamaGenerator(think="low", client=client)
        rag_answer = generator.answer(
            "What ground area does an eight-digit grid coordinate locate?",
            results,
        )

    assert len(captured_payloads) == 1
    assert captured_payloads[0]["model"] == "inkling-2-mini:latest"
    assert captured_payloads[0]["stream"] is False
    assert captured_payloads[0]["think"] == "low"
    assert rag_answer.answer == "It locates a 10-meter square [1]."
    assert rag_answer.citations == results
