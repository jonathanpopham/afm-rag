"""Load Markdown field manuals from directories and vault zip archives."""

import re
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from afm_rag.models import ManualDocument

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")


def _extract_title(text: str, fallback_stem: str) -> str:
    """Return the first H1 heading from Markdown text or a clean filename."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            heading = stripped[2:].strip()
            if heading:
                return heading
    return fallback_stem.replace("_", " ").strip()


def _make_doc_id(relative_key: str) -> str:
    """Build a stable lowercase identifier from a relative document path."""
    cleaned = relative_key.strip("/")
    if cleaned.lower().endswith(".md"):
        cleaned = cleaned[:-3]

    normalized_parts: list[str] = []
    for part in PurePosixPath(cleaned).parts:
        slug = _SLUG_RE.sub("-", part.lower()).strip("-")
        if slug:
            normalized_parts.append(slug)

    return "/".join(normalized_parts) or "manual"


def load_markdown_dir(root: Path) -> list[ManualDocument]:
    """Load all Markdown files recursively from a directory tree.

    Args:
        root: Base directory that contains categorized manual folders.

    Returns:
        Sorted list of ManualDocument objects.
    """
    if not root.is_dir():
        raise FileNotFoundError(f"Manual directory does not exist: {root}")

    documents: list[ManualDocument] = []
    for md_path in sorted(path for path in root.rglob("*.md") if path.is_file()):
        rel_path = md_path.relative_to(root)
        parent_parts = rel_path.parent.parts

        # Derive category and subcategory from the top two directory levels.
        category = parent_parts[0] if len(parent_parts) >= 1 else "Uncategorized"
        subcategory = parent_parts[1] if len(parent_parts) >= 2 else "General"

        text = md_path.read_text(encoding="utf-8", errors="replace")
        title = _extract_title(text, md_path.stem)
        rel_key = rel_path.as_posix()

        documents.append(
            ManualDocument(
                doc_id=_make_doc_id(rel_key),
                title=title,
                category=category,
                subcategory=subcategory,
                source_path=rel_key,
                text=text,
            )
        )

    return documents


def iter_vault_zip_manuals(
    md_root: Path,
    limit: int | None = None,
) -> Iterator[ManualDocument]:
    """Stream Markdown manuals from vault zip archives while skipping parity files.

    Args:
        md_root: Vault MD root directory that contains category folders and zip files.
        limit: Optional maximum count of manuals to yield.

    Yields:
        ManualDocument objects decoded from zip archive entries.
    """
    if not md_root.is_dir():
        raise FileNotFoundError(f"Vault MD directory does not exist: {md_root}")
    if limit is not None and limit <= 0:
        return

    yielded = 0
    zip_paths = sorted(path for path in md_root.rglob("*.zip") if path.is_file())

    for zip_path in zip_paths:
        category = zip_path.parent.name if zip_path.parent != md_root else "Uncategorized"
        subcategory = zip_path.stem
        rel_zip = zip_path.relative_to(md_root).as_posix()

        with zipfile.ZipFile(zip_path, "r") as archive:
            for member_name in sorted(archive.namelist()):
                if member_name.endswith("/") or member_name.startswith("__MACOSX/"):
                    continue

                member_posix = PurePosixPath(member_name)
                if member_posix.name.startswith("._"):
                    continue
                if member_posix.suffix.lower() != ".md":
                    continue

                raw_bytes = archive.read(member_name)
                text = raw_bytes.decode("utf-8", errors="replace")
                title = _extract_title(text, member_posix.stem)

                # Include category and subcategory in the key so identical filenames stay unique.
                doc_id = _make_doc_id(f"{category}/{subcategory}/{member_name}")
                yield ManualDocument(
                    doc_id=doc_id,
                    title=title,
                    category=category,
                    subcategory=subcategory,
                    source_path=f"{rel_zip}::{member_name}",
                    text=text,
                )

                yielded += 1
                if limit is not None and yielded >= limit:
                    return
