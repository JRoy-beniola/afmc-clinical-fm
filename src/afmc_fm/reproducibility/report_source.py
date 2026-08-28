from __future__ import annotations

import hashlib
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_W = f"{{{_W_NS}}}"
_R = f"{{{_R_NS}}}"
_HEADING_RE = re.compile(r"^Heading ([1-6])$", re.IGNORECASE)


@dataclass(frozen=True)
class ParagraphBlock:
    text: str
    style: str | None = None


@dataclass(frozen=True)
class TableBlock:
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class ImageBlock:
    relationship_id: str


ReportBlock = ParagraphBlock | TableBlock | ImageBlock


@dataclass(frozen=True)
class ReportSnapshot:
    reference_sha256: str
    blocks: tuple[ReportBlock, ...]

    @property
    def paragraph_count(self) -> int:
        return sum(isinstance(block, ParagraphBlock) for block in self.blocks)

    @property
    def table_count(self) -> int:
        return sum(isinstance(block, TableBlock) for block in self.blocks)

    @property
    def image_count(self) -> int:
        return sum(isinstance(block, ImageBlock) for block in self.blocks)

    def to_json_dict(self) -> dict[str, Any]:
        serialized: list[dict[str, Any]] = []
        for block in self.blocks:
            if isinstance(block, ParagraphBlock):
                serialized.append(
                    {"type": "paragraph", "style": block.style, "text": block.text}
                )
            elif isinstance(block, TableBlock):
                serialized.append(
                    {"type": "table", "rows": [list(row) for row in block.rows]}
                )
            else:
                serialized.append(
                    {"type": "image", "relationship_id": block.relationship_id}
                )
        return {
            "schema_version": 1,
            "reference_sha256": self.reference_sha256,
            "block_count": len(self.blocks),
            "paragraph_count": self.paragraph_count,
            "table_count": self.table_count,
            "image_count": self.image_count,
            "blocks": serialized,
        }


def _parse_xml(archive: zipfile.ZipFile, member: str) -> ElementTree.Element:
    try:
        payload = archive.read(member)
    except KeyError as exc:
        raise ValueError(f"DOCX is missing {member}") from exc
    try:
        return ElementTree.fromstring(payload)
    except ElementTree.ParseError as exc:
        raise ValueError(f"DOCX contains malformed XML in {member}") from exc


def _style_names(archive: zipfile.ZipFile) -> dict[str, str]:
    if "word/styles.xml" not in archive.namelist():
        return {}
    root = _parse_xml(archive, "word/styles.xml")
    result: dict[str, str] = {}
    for style in root.findall(f"{_W}style"):
        style_id = style.get(f"{_W}styleId")
        name = style.find(f"{_W}name")
        value = name.get(f"{_W}val") if name is not None else None
        if style_id and value:
            result[style_id] = value
    return result


def _text_from_element(element: ElementTree.Element) -> str:
    parts: list[str] = []
    for child in element.iter():
        if child.tag == f"{_W}t":
            parts.append(child.text or "")
        elif child.tag == f"{_W}tab":
            parts.append("\t")
        elif child.tag in {f"{_W}br", f"{_W}cr"}:
            parts.append("\n")
    return "".join(parts)


def _paragraph_block(
    paragraph: ElementTree.Element,
    styles: dict[str, str],
) -> ParagraphBlock:
    style: str | None = None
    properties = paragraph.find(f"{_W}pPr")
    if properties is not None:
        style_node = properties.find(f"{_W}pStyle")
        if style_node is not None:
            style_id = style_node.get(f"{_W}val")
            if style_id:
                style = styles.get(style_id, style_id)
    return ParagraphBlock(text=_text_from_element(paragraph), style=style)


def _table_block(table: ElementTree.Element) -> TableBlock:
    rows: list[tuple[str, ...]] = []
    for row in table.findall(f"{_W}tr"):
        cells: list[str] = []
        for cell in row.findall(f"{_W}tc"):
            paragraphs = [
                _text_from_element(paragraph)
                for paragraph in cell.findall(f"{_W}p")
            ]
            cells.append("\n".join(paragraphs))
        rows.append(tuple(cells))
    return TableBlock(rows=tuple(rows))


def _image_relationships(element: ElementTree.Element) -> tuple[str, ...]:
    relationships: list[str] = []
    for descendant in element.iter():
        for attribute, value in descendant.attrib.items():
            if attribute in {f"{_R}embed", f"{_R}link"} and value not in relationships:
                relationships.append(value)
    return tuple(relationships)


def extract_docx(path: Path) -> ReportSnapshot:
    """Extract a deterministic read-only snapshot from a DOCX package."""

    source = Path(path)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    try:
        archive = zipfile.ZipFile(source, "r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"invalid DOCX archive: {source}") from exc

    with archive:
        document = _parse_xml(archive, "word/document.xml")
        styles = _style_names(archive)
        body = document.find(f"{_W}body")
        if body is None:
            raise ValueError("DOCX document.xml has no body")

        blocks: list[ReportBlock] = []
        for child in body:
            if child.tag == f"{_W}p":
                blocks.append(_paragraph_block(child, styles))
                blocks.extend(
                    ImageBlock(relationship_id=relationship_id)
                    for relationship_id in _image_relationships(child)
                )
            elif child.tag == f"{_W}tbl":
                blocks.append(_table_block(child))
                blocks.extend(
                    ImageBlock(relationship_id=relationship_id)
                    for relationship_id in _image_relationships(child)
                )

    return ReportSnapshot(reference_sha256=digest, blocks=tuple(blocks))


def _escape_markdown_cell(value: str) -> str:
    return value.replace("\\", "\\\\").replace("|", "\\|").replace("\n", "<br>")


def _render_table(block: TableBlock) -> str:
    if not block.rows:
        return "<!-- empty-table -->"
    width = max(len(row) for row in block.rows)
    normalized = [row + ("",) * (width - len(row)) for row in block.rows]
    header = normalized[0]
    lines = ["| " + " | ".join(_escape_markdown_cell(cell) for cell in header) + " |"]
    lines.append("| " + " | ".join("---" for _ in range(width)) + " |")
    for row in normalized[1:]:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def render_markdown(snapshot: ReportSnapshot) -> str:
    """Render a deterministic, non-generative canonical Markdown representation."""

    rendered: list[str] = []
    for block in snapshot.blocks:
        if isinstance(block, ParagraphBlock):
            if not block.text.strip():
                rendered.append("<!-- blank -->")
                continue
            match = _HEADING_RE.fullmatch(block.style or "")
            if match:
                rendered.append(f"{'#' * int(match.group(1))} {block.text}")
            else:
                rendered.append(block.text)
        elif isinstance(block, TableBlock):
            rendered.append(_render_table(block))
        else:
            rendered.append(f"<!-- image rel={block.relationship_id} -->")
    return "\n\n".join(rendered) + "\n"
