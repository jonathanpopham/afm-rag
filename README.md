# afm-rag

`afm-rag` is a local Retrieval-Augmented Generation (RAG) engine for U.S. Army Field Manuals.
The system runs locally on Ollama (`inkling-2-mini:latest` and `qwen3-embedding:0.6b`) and SQLite (`sqlite-vec` and `FTS5`).

## Three-Layer Architecture

1. **Layer 1 — Core Python RAG Engine (`src/afm_rag/`):** Strips YAML frontmatter, builds section chunks, embeds 512-dim MRL vectors, and runs hybrid SQLite search.
2. **Layer 2 — Local MCP Server (`afm-rag-mcp`):** Exposes index statistics, manual listing, hybrid search with `status` filters, and section reading over stdio.
3. **Layer 3 — Codex CLI + Skills (`.agents/skills/`):** Runs `inkling-2-mini:latest` in Codex CLI with local doctrine research skills.

## Module Study Guide

Read the source files in the numbered order below to learn the complete RAG pipeline:

1. `src/afm_rag/models.py` — Immutable dataclasses (`ManualDocument`, `TextChunk`, `SearchResult`) carrying publication status, branch, designation, and edition date.
2. `src/afm_rag/loader.py` — YAML frontmatter parser, boilerplate stripper, directory loader, and `.zip` vault reader skipping `.par2` files.
3. `src/afm_rag/chunker.py` — Heading-stack Markdown parser and overlapping word-boundary section splitter.
4. `src/afm_rag/embeddings.py` — NumPy cosine similarity math and batched Ollama `/api/embed` client (512-dim MRL, automatic retries).
5. `src/afm_rag/store.py` — SQLite WAL store combining native `sqlite-vec` metadata filtering, `FTS5` BM25 search, and Reciprocal Rank Fusion.
6. `src/afm_rag/generator.py` — Grounded citation prompt builder and Ollama `/api/chat` client for `inkling-2-mini:latest`.
7. `src/afm_rag/pipeline.py` — Active-first vault ingestion with SHA-256 deduplication, search, and cited question answering.
8. `src/afm_rag/eval.py` — Retrieval benchmark calculating `Hit@K` and Mean Reciprocal Rank (`MRR`).
9. `src/afm_rag/cli.py` — `afm-rag` command-line tool with `ingest`, `search`, `ask`, `stats`, and `eval` subcommands.
10. `src/afm_rag/mcp_server.py` — `MCPServer` exposing `index_stats`, `list_manuals`, `search_manuals`, and `read_manual_section`.
11. `.agents/skills/afm-doctrine-research/SKILL.md` — Codex CLI skill defining doctrine research and citation workflow.

## Quickstart Commands

Run all commands inside `/Users/jag/antigravity/afm-rag`.

### 1. Sync and Verify Tests

1. Sync dependencies into `.venv`:

```bash
uv sync
```

2. Run the unit test suite:

```bash
uv run pytest -v
```

3. Run code lint and format checks:

```bash
uv run ruff check src tests && uv run ruff format --check src tests
```

### 2. Ingest the Downloaded Vault

The vault contains `1,355` active manuals and `5,117` superseded manuals inside `.zip` archives.
Default ingestion indexes `active` manuals first, then indexes `superseded` manuals while SQLite WAL mode supports live queries.

1. Test ingestion on 25 active manuals:

```bash
uv run afm-rag ingest "data/T.Rex Arms Field Manual Vault/MD" --db afm_rag.db --status active --limit 25
```

2. Ingest the complete Markdown archive (SHA-256 hashing skips unchanged manuals):

```bash
uv run afm-rag ingest "data/T.Rex Arms Field Manual Vault/MD" --db afm_rag.db
```

3. Check indexed manual, status, and category counts:

```bash
uv run afm-rag stats --db afm_rag.db
```

4. Ask a question restricted to current active doctrine:

```bash
uv run afm-rag ask "What ground area does an eight-digit grid coordinate locate?" --db afm_rag.db --status active --think low
```

### 3. Connect Codex CLI and Local Skills

1. Register the local MCP server with Codex CLI:

```bash
codex mcp add afm-rag --env AFM_RAG_DB=/Users/jag/antigravity/afm-rag/afm_rag.db -- uv --directory /Users/jag/antigravity/afm-rag run afm-rag-mcp
```

2. Launch Codex CLI with `inkling-2-mini:latest` inside `/Users/jag/antigravity/afm-rag`:

```bash
ollama launch codex --model inkling-2-mini:latest
```

3. Ask Codex CLI a doctrine question so Codex loads `afm-doctrine-research` and calls `search_manuals`.

## License

`afm-rag` uses the MIT License. Read [LICENSE](LICENSE) for complete terms.
