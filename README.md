# afm-rag

`afm-rag` builds a local Retrieval-Augmented Generation (RAG) engine for U.S. Army Field Manuals.
The system runs locally on Ollama (`inkling-2-mini:latest` and `qwen3-embedding:0.6b`) and SQLite (`sqlite-vec` and `FTS5`).

## Learning Roadmap

1. **Milestone 1 (Complete):** Configure Python 3.13, `uv`, sample manual fixtures, and data loaders.
2. **Milestone 2 (Active):** Implement structure-aware Markdown chunking inside `src/afm_rag/chunker.py`.
3. **Milestone 3 (Upcoming):** Generate local embeddings and calculate cosine similarity with NumPy.
4. **Milestone 4 (Upcoming):** Build a hybrid SQLite index combining `sqlite-vec`, `FTS5`, and Reciprocal Rank Fusion.
5. **Milestone 5 (Upcoming):** Generate cited answers with `inkling-2-mini` and measure retrieval accuracy.
6. **Milestone 6 (Upcoming):** Expose retrieval tools over a local MCP server for Codex CLI and skills.

## Local Commands

Run these commands inside `/Users/jag/antigravity/afm-rag`:

1. Sync dependencies into `.venv`:

```bash
uv sync
```

2. Run the test suite:

```bash
uv run pytest
```

3. Run the linter:

```bash
uv run ruff check src tests
```

## License

`afm-rag` uses the MIT License. Read [LICENSE](LICENSE) for complete terms.
