"""Load Markdown field manuals from directories and vault zip archives."""

import html
import re
import zipfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath

from afm_rag.models import ManualDocument

_SLUG_RE = re.compile(r"[^a-z0-9._-]+")
_BOILERPLATE_PREFIX = "> Text-only Markdown of "


def parse_frontmatter(raw_text: str) -> tuple[dict[str, str], str]:
    """Extract YAML frontmatter pairs and clean Markdown body text."""
    lines = raw_text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw_text

    closing_idx: int | None = None
    for idx in range(1, len(lines)):
        if lines[idx].strip() == "---":
            closing_idx = idx
            break

    if closing_idx is None:
        return {}, raw_text

    metadata: dict[str, str] = {}
    for raw_line in lines[1:closing_idx]:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw_val = stripped.split(":", 1)
        val = raw_val.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
            val = val[1:-1]
        metadata[key.strip().lower()] = html.unescape(val.strip())

    body_lines = lines[closing_idx + 1 :]
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    while body_lines and body_lines[0].lstrip().startswith(_BOILERPLATE_PREFIX):
        body_lines.pop(0)
        while body_lines and not body_lines[0].strip():
            body_lines.pop(0)

    clean_body = "\n".join(body_lines)
    if clean_body and raw_text.endswith("\n"):
        clean_body += "\n"
    return metadata, clean_body


def _extract_title(
    text: str,
    fallback_stem: str,
    frontmatter: dict[str, str] | None = None,
) -> str:
    """Return a clean manual title from frontmatter, H1 heading, or filename."""
    if frontmatter:
        fm_title = frontmatter.get("title", "").strip()
        fm_desig = frontmatter.get("designation", "").strip()
        if fm_title:
            if fm_desig and fm_desig.lower() not in fm_title.lower():
                return f"{fm_desig} — {fm_title}"
            return fm_title

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
    """Load all Markdown files recursively from a directory tree."""
    if not root.is_dir():
        raise FileNotFoundError(f"Manual directory does not exist: {root}")

    documents: list[ManualDocument] = []
    for md_path in sorted(path for path in root.rglob("*.md") if path.is_file()):
        rel_path = md_path.relative_to(root)
        parent_parts = rel_path.parent.parts

        # Derive category and subcategory from the top two directory levels.
        category = parent_parts[0] if len(parent_parts) >= 1 else "Uncategorized"
        subcategory = parent_parts[1] if len(parent_parts) >= 2 else "General"

        raw_text = md_path.read_text(encoding="utf-8", errors="replace")
        meta, clean_body = parse_frontmatter(raw_text)
        title = _extract_title(clean_body, md_path.stem, frontmatter=meta)
        rel_key = rel_path.as_posix()

        documents.append(
            ManualDocument(
                doc_id=_make_doc_id(rel_key),
                title=title,
                category=category,
                subcategory=subcategory,
                source_path=rel_key,
                text=clean_body,
                status=meta.get("status", "unknown") or "unknown",
                branch=meta.get("branch", ""),
                designation=meta.get("designation", ""),
                edition_date=meta.get("edition_date", ""),
            )
        )

    return documents


def iter_vault_zip_manuals(
    md_root: Path,
    limit: int | None = None,
    status_filter: str | None = None,
    category_filter: str | None = None,
) -> Iterator[ManualDocument]:
    """Stream Markdown manuals from vault zip archives while skipping parity files."""
    if not md_root.is_dir():
        raise FileNotFoundError(f"Vault MD directory does not exist: {md_root}")
    if limit is not None and limit <= 0:
        return

    yielded = 0
    zip_paths = sorted(path for path in md_root.rglob("*.zip") if path.is_file())
    target_status = status_filter.strip().lower() if status_filter else None
    target_category = category_filter.strip().lower() if category_filter else None

    for zip_path in zip_paths:
        category = zip_path.parent.name if zip_path.parent != md_root else "Uncategorized"
        if target_category is not None and category.lower() != target_category:
            continue

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
                raw_text = raw_bytes.decode("utf-8", errors="replace")
                meta, clean_body = parse_frontmatter(raw_text)
                doc_status = meta.get("status", "unknown") or "unknown"

                if target_status is not None and doc_status.lower() != target_status:
                    continue

                title = _extract_title(clean_body, member_posix.stem, frontmatter=meta)

                # Include category and subcategory in the key so identical filenames stay unique.
                doc_id = _make_doc_id(f"{category}/{subcategory}/{member_name}")
                yield ManualDocument(
                    doc_id=doc_id,
                    title=title,
                    category=category,
                    subcategory=subcategory,
                    source_path=f"{rel_zip}::{member_name}",
                    text=clean_body,
                    status=doc_status,
                    branch=meta.get("branch", ""),
                    designation=meta.get("designation", ""),
                    edition_date=meta.get("edition_date", ""),
                )

                yielded += 1
                if limit is not None and yielded >= limit:
                    return
