from afmc_fm.reproducibility.document_compare import compare_snapshots

from afmc_fm.reproducibility.report_source import (
    ImageBlock,
    ParagraphBlock,
    ReportSnapshot,
    TableBlock,
)


def _snapshot(
    *,
    narrative: str = "Recovered narrative.",
    heading: str = "Report title",
    table_value: str = "1.25",
    caption: str = "Figure 1. Result",
    candidate: bool = False,
) -> ReportSnapshot:
    heading_style = "Heading1" if candidate else "Heading 1"
    blocks = [
        ParagraphBlock(heading, heading_style),
        ParagraphBlock(narrative),
        ParagraphBlock(caption, None if candidate else "Caption"),
        TableBlock((("Metric", "Value"), ("MAE", table_value))),
    ]
    if candidate:
        blocks.append(ParagraphBlock("[image rel=rId7]"))
    else:
        blocks.append(ImageBlock("rId7"))
    return ReportSnapshot(reference_sha256="a" * 64, blocks=tuple(blocks))


def test_compare_snapshots_accepts_semantically_equivalent_structure():
    comparison = compare_snapshots(_snapshot(), _snapshot(candidate=True))

    assert comparison.ok
    assert comparison.paragraph_order_match
    assert comparison.heading_order_match
    assert comparison.table_order_match
    assert comparison.caption_order_match
    assert comparison.image_placeholder_match


def test_compare_snapshots_normalizes_unicode_whitespace_only():
    reference = _snapshot(narrative="A\u00a0  B\nC")
    candidate = _snapshot(narrative="A B C", candidate=True)

    comparison = compare_snapshots(reference, candidate)

    assert comparison.paragraph_order_match


def test_compare_snapshots_detects_numeric_table_change():
    comparison = compare_snapshots(_snapshot(), _snapshot(table_value="1.26", candidate=True))

    assert not comparison.table_order_match
    assert not comparison.ok


def test_compare_snapshots_detects_paragraph_wording_change():
    comparison = compare_snapshots(
        _snapshot(),
        _snapshot(narrative="Different narrative.", candidate=True),
    )

    assert not comparison.paragraph_order_match
    assert not comparison.ok


def test_compare_snapshots_detects_heading_change():
    comparison = compare_snapshots(_snapshot(), _snapshot(heading="Different title", candidate=True))

    assert not comparison.heading_order_match
    assert not comparison.ok


def test_compare_snapshots_compares_caption_text_without_requiring_candidate_style():
    comparison = compare_snapshots(_snapshot(), _snapshot(candidate=True))
    changed = compare_snapshots(_snapshot(), _snapshot(caption="Figure 1. Changed", candidate=True))

    assert comparison.caption_order_match
    assert not changed.caption_order_match


def test_compare_snapshots_does_not_count_image_placeholder_as_narrative():
    comparison = compare_snapshots(_snapshot(), _snapshot(candidate=True))

    assert comparison.paragraph_order_match
    assert comparison.image_placeholder_match
    assert comparison.reference_image_count == 1
    assert comparison.candidate_placeholder_count == 1


def test_comparison_serialization_is_stable():
    comparison = compare_snapshots(_snapshot(), _snapshot(candidate=True))

    payload = comparison.to_json_dict()

    assert payload["comparison_kind"] == "structural-content"
    assert payload["byte_identical"] is False
    assert payload["ok"] is True
