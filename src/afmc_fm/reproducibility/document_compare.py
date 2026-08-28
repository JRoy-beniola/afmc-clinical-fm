from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .report_source import ImageBlock, ParagraphBlock, ReportSnapshot, TableBlock, extract_docx

_HEADING_RE = re.compile(r"^heading\s*([1-6])$", re.IGNORECASE)
_IMAGE_PLACEHOLDER_RE = re.compile(r"^\[image\s+rel=([^\]]+)\]$", re.IGNORECASE)
_CAPTION_RE = re.compile(r"^(?:figure|table)\s+\d+\b", re.IGNORECASE)


@dataclass(frozen=True)
class DocumentComparison:
    ok: bool
    paragraph_order_match: bool
    table_order_match: bool
    heading_order_match: bool
    caption_order_match: bool
    image_placeholder_match: bool
    reference_image_count: int
    candidate_placeholder_count: int
    differences: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "comparison_kind": "structural-content",
            "byte_identical": False,
            "ok": self.ok,
            "paragraph_order_match": self.paragraph_order_match,
            "table_order_match": self.table_order_match,
            "heading_order_match": self.heading_order_match,
            "caption_order_match": self.caption_order_match,
            "image_placeholder_match": self.image_placeholder_match,
            "reference_image_count": self.reference_image_count,
            "candidate_placeholder_count": self.candidate_placeholder_count,
            "differences": list(self.differences),
        }


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFC", value)
    return " ".join(normalized.split())


def _heading_level(style: str | None) -> int | None:
    if style is None:
        return None
    match = _HEADING_RE.fullmatch(style.strip())
    return int(match.group(1)) if match else None


def _is_caption(block: ParagraphBlock) -> bool:
    if (block.style or "").strip().casefold() == "caption":
        return True
    return bool(_CAPTION_RE.match(_normalize_text(block.text)))


def _placeholder_relationship(block: ParagraphBlock) -> str | None:
    match = _IMAGE_PLACEHOLDER_RE.fullmatch(_normalize_text(block.text))
    return match.group(1) if match else None


def _narrative(snapshot: ReportSnapshot) -> tuple[str, ...]:
    values: list[str] = []
    for block in snapshot.blocks:
        if not isinstance(block, ParagraphBlock):
            continue
        if _heading_level(block.style) is not None or _is_caption(block):
            continue
        if _placeholder_relationship(block) is not None:
            continue
        values.append(_normalize_text(block.text))
    return tuple(values)


def _headings(snapshot: ReportSnapshot) -> tuple[tuple[int, str], ...]:
    values: list[tuple[int, str]] = []
    for block in snapshot.blocks:
        if not isinstance(block, ParagraphBlock):
            continue
        level = _heading_level(block.style)
        if level is not None:
            values.append((level, _normalize_text(block.text)))
    return tuple(values)


def _captions(snapshot: ReportSnapshot) -> tuple[str, ...]:
    return tuple(
        _normalize_text(block.text)
        for block in snapshot.blocks
        if isinstance(block, ParagraphBlock) and _is_caption(block)
    )


def _tables(snapshot: ReportSnapshot) -> tuple[tuple[tuple[str, ...], ...], ...]:
    return tuple(
        tuple(tuple(_normalize_text(cell) for cell in row) for row in block.rows)
        for block in snapshot.blocks
        if isinstance(block, TableBlock)
    )


def _reference_images(snapshot: ReportSnapshot) -> tuple[str, ...]:
    return tuple(
        block.relationship_id
        for block in snapshot.blocks
        if isinstance(block, ImageBlock)
    )


def _candidate_placeholders(snapshot: ReportSnapshot) -> tuple[str, ...]:
    values: list[str] = []
    for block in snapshot.blocks:
        if not isinstance(block, ParagraphBlock):
            continue
        relationship = _placeholder_relationship(block)
        if relationship is not None:
            values.append(relationship)
    return tuple(values)


def compare_snapshots(
    reference: ReportSnapshot,
    candidate: ReportSnapshot,
) -> DocumentComparison:
    """Compare documentary structure/content without claiming byte identity."""

    paragraph_order_match = _narrative(reference) == _narrative(candidate)
    heading_order_match = _headings(reference) == _headings(candidate)
    table_order_match = _tables(reference) == _tables(candidate)
    caption_order_match = _captions(reference) == _captions(candidate)
    reference_images = _reference_images(reference)
    candidate_placeholders = _candidate_placeholders(candidate)
    image_placeholder_match = reference_images == candidate_placeholders

    checks = (
        (paragraph_order_match, "paragraph_order"),
        (heading_order_match, "heading_order"),
        (table_order_match, "table_order"),
        (caption_order_match, "caption_order"),
        (image_placeholder_match, "image_placeholders"),
    )
    differences = tuple(label for matches, label in checks if not matches)
    ok = not differences

    return DocumentComparison(
        ok=ok,
        paragraph_order_match=paragraph_order_match,
        table_order_match=table_order_match,
        heading_order_match=heading_order_match,
        caption_order_match=caption_order_match,
        image_placeholder_match=image_placeholder_match,
        reference_image_count=len(reference_images),
        candidate_placeholder_count=len(candidate_placeholders),
        differences=differences,
    )


def compare_documents(reference: Path, candidate: Path) -> DocumentComparison:
    """Extract and compare immutable reference and rebuilt candidate DOCX files."""

    return compare_snapshots(extract_docx(reference), extract_docx(candidate))
