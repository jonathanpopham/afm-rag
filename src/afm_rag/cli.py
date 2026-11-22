"""Command-line interface for ingesting, searching, evaluating, and querying manuals."""

import argparse
import time
from pathlib import Path

from afm_rag.embeddings import DEFAULT_EMBED_DIMS, DEFAULT_EMBED_MODEL, OllamaEmbedder
from afm_rag.eval import evaluate_retrieval
from afm_rag.generator import DEFAULT_CHAT_MODEL, OllamaGenerator
from afm_rag.pipeline import ask_manuals, ingest_path, search_index
from afm_rag.store import HybridStore

DEFAULT_DB_PATH = "afm_rag.db"


def _parse_think_level(value: str) -> str | bool:
    """Convert CLI think option string into an Ollama API think setting."""
    return False if value == "off" else value


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level argparse parser with afm-rag subcommands."""
    parser = argparse.ArgumentParser(
        prog="afm-rag",
        description="Local RAG CLI for U.S. Army Field Manuals.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Chunk, embed, and store manuals into SQLite.",
    )
    ingest_parser.add_argument("source", type=Path, help="Manual folder or vault MD root path.")
    ingest_parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH))
    ingest_parser.add_argument("--max-chars", type=int, default=1800)
    ingest_parser.add_argument("--overlap-chars", type=int, default=200)
    ingest_parser.add_argument("--limit", type=int, default=None)
    ingest_parser.add_argument("--force", action="store_true")
    ingest_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ingest_parser.add_argument("--dims", type=int, default=DEFAULT_EMBED_DIMS)
    ingest_parser.add_argument("--batch-size", type=int, default=64)
    ingest_parser.add_argument("--status", default=None)
    ingest_parser.add_argument("--category", default=None)
    ingest_parser.add_argument("--no-active-first", action="store_true")

    search_parser = subparsers.add_parser(
        "search",
        help="Run hybrid vector and FTS5 retrieval.",
    )
    search_parser.add_argument("query", help="Search query text.")
    search_parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH))
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--category", default=None)
    search_parser.add_argument("--subcategory", default=None)
    search_parser.add_argument("--status", default=None)
    search_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    search_parser.add_argument("--dims", type=int, default=DEFAULT_EMBED_DIMS)

    ask_parser = subparsers.add_parser(
        "ask",
        help="Retrieve matching chunks and generate a cited answer.",
    )
    ask_parser.add_argument("question", help="User question for the field manuals.")
    ask_parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH))
    ask_parser.add_argument("--top-k", type=int, default=5)
    ask_parser.add_argument("--category", default=None)
    ask_parser.add_argument("--subcategory", default=None)
    ask_parser.add_argument("--status", default=None)
    ask_parser.add_argument(
        "--think",
        choices=("off", "low", "medium", "high"),
        default="low",
    )
    ask_parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    ask_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    ask_parser.add_argument("--dims", type=int, default=DEFAULT_EMBED_DIMS)

    stats_parser = subparsers.add_parser("stats", help="Display SQLite index statistics.")
    stats_parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH))

    eval_parser = subparsers.add_parser(
        "eval",
        help="Evaluate Hit@K and MRR against fixture doctrine questions.",
    )
    eval_parser.add_argument("--db", type=Path, default=Path(DEFAULT_DB_PATH))
    eval_parser.add_argument("--top-k", type=int, default=3)
    eval_parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    eval_parser.add_argument("--dims", type=int, default=DEFAULT_EMBED_DIMS)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute the selected afm-rag CLI subcommand."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "ingest":
        start_time = time.perf_counter()

        def report_progress(stats: dict[str, int], title: str) -> None:
            elapsed = max(time.perf_counter() - start_time, 1e-6)
            rate = stats["chunks"] / elapsed
            print(
                f"[{stats['indexed']}] "
                f"(skipped={stats['skipped']}, failed={stats['failed']}, "
                f"chunks={stats['chunks']}, rate={rate:.1f} chunks/s) {title}",
                flush=True,
            )

        with OllamaEmbedder(
            model=args.embed_model,
            batch_size=args.batch_size,
            dimensions=args.dims,
        ) as embedder:
            summary = ingest_path(
                source=args.source,
                db_path=args.db,
                embedder=embedder,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                limit=args.limit,
                force=args.force,
                embedding_dims=args.dims,
                embed_model=args.embed_model,
                batch_size=args.batch_size,
                status_filter=args.status,
                category_filter=args.category,
                active_first=not args.no_active_first,
                progress_callback=report_progress,
            )
        print(
            f"Ingest complete: {summary['manuals_indexed']} indexed, "
            f"{summary['manuals_skipped']} skipped, "
            f"{summary['manuals_failed']} failed, "
            f"{summary['chunks_indexed']} chunks written.",
            flush=True,
        )
        return 0

    if args.command == "search":
        with OllamaEmbedder(model=args.embed_model, dimensions=args.dims) as embedder:
            results = search_index(
                query=args.query,
                db_path=args.db,
                top_k=args.top_k,
                category=args.category,
                subcategory=args.subcategory,
                status=args.status,
                embedder=embedder,
                embedding_dims=args.dims,
                embed_model=args.embed_model,
            )
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            print(
                f"[{index}] {chunk.title} [{chunk.status}] — {chunk.section_path} "
                f"({chunk.doc_id}) | score={result.score:.4f} "
                f"vec={result.vector_rank} lex={result.lexical_rank}"
            )
            print(chunk.text)
            print()
        return 0

    if args.command == "ask":
        think_mode = _parse_think_level(args.think)
        with (
            OllamaEmbedder(model=args.embed_model, dimensions=args.dims) as embedder,
            OllamaGenerator(model=args.model, think=think_mode) as generator,
        ):
            rag_answer = ask_manuals(
                question=args.question,
                db_path=args.db,
                top_k=args.top_k,
                category=args.category,
                subcategory=args.subcategory,
                status=args.status,
                think=think_mode,
                embedder=embedder,
                generator=generator,
                embedding_dims=args.dims,
                embed_model=args.embed_model,
            )
        print(rag_answer.answer)
        print("\nSources:")
        for index, result in enumerate(rag_answer.citations, start=1):
            chunk = result.chunk
            print(
                f"[{index}] {chunk.title} [{chunk.status}] — {chunk.section_path} ({chunk.doc_id})"
            )
        return 0

    if args.command == "stats":
        with HybridStore(args.db, embedding_dims=None) as store:
            data = store.stats()
        print(f"Manuals: {data['manual_count']}")
        print(f"Chunks: {data['chunk_count']}")
        print(f"Statuses: {data['statuses']}")
        print(f"Categories: {data['categories']}")
        return 0

    if args.command == "eval":
        with (
            HybridStore(args.db, embedding_dims=args.dims) as store,
            OllamaEmbedder(model=args.embed_model, dimensions=args.dims) as embedder,
        ):
            report = evaluate_retrieval(store=store, embedder=embedder, top_k=args.top_k)
        print(
            f"Cases: {report['cases_total']} | Hits: {report['hits']} | "
            f"Hit@{args.top_k}: {report['hit_at_k']:.2f} | MRR: {report['mrr']:.2f}"
        )
        for item in report["details"]:
            status = "PASS" if item["hit"] else "MISS"
            print(
                f"  [{status}] rr={item['reciprocal_rank']:.2f} "
                f"| top={item['top_section']} | {item['question']}"
            )
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
