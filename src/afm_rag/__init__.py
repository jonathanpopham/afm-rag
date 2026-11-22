"""Core data models, loaders, chunkers, embeddings, store, and RAG pipelines."""

from afm_rag.chunker import chunk_document, split_markdown_sections
from afm_rag.embeddings import OllamaEmbedder, cosine_similarity, rank_top_k
from afm_rag.eval import EvalCase, evaluate_retrieval
from afm_rag.generator import OllamaGenerator, RagAnswer, build_grounded_messages
from afm_rag.loader import iter_vault_zip_manuals, load_markdown_dir
from afm_rag.models import (
    ManualDocument,
    SearchResult,
    TextChunk,
    format_chunk_for_embedding,
)
from afm_rag.pipeline import ask_manuals, ingest_path, search_index
from afm_rag.store import HybridStore, reciprocal_rank_fusion

__all__ = [
    "EvalCase",
    "HybridStore",
    "ManualDocument",
    "OllamaEmbedder",
    "OllamaGenerator",
    "RagAnswer",
    "SearchResult",
    "TextChunk",
    "ask_manuals",
    "build_grounded_messages",
    "chunk_document",
    "cosine_similarity",
    "evaluate_retrieval",
    "format_chunk_for_embedding",
    "ingest_path",
    "iter_vault_zip_manuals",
    "load_markdown_dir",
    "rank_top_k",
    "reciprocal_rank_fusion",
    "search_index",
    "split_markdown_sections",
]
