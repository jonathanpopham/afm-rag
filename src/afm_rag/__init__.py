"""Core data models, loaders, and chunkers for the local AFM RAG system."""

from afm_rag.chunker import chunk_document, split_markdown_sections
from afm_rag.loader import iter_vault_zip_manuals, load_markdown_dir
from afm_rag.models import ManualDocument, TextChunk, format_chunk_for_embedding

__all__ = [
    "ManualDocument",
    "TextChunk",
    "chunk_document",
    "format_chunk_for_embedding",
    "iter_vault_zip_manuals",
    "load_markdown_dir",
    "split_markdown_sections",
]
