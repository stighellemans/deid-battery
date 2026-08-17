from __future__ import annotations

import io
import json
from pathlib import Path

from deid_battery.human_annotation_missingness import (
    scan_human_annotation_missingness,
    write_report,
)


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def test_reports_all_missingness_without_source_text_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"doc_id": "doc-a", "text": "Date: 16/06/2025"})
        + "\n"
        + json.dumps({"doc_id": "doc-b", "text": "No annotations"})
        + "\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "legacy"
    _write_json(
        source_root / "stig1" / "spans" / "doc-a.json",
        [
            {"begin": 6, "end": 16, "text": "16/06/2025", "Category": "Date"},
            {"begin": 0, "label": "Date", "Category": "Date"},
        ],
    )
    _write_json(source_root / "stig2" / "spans" / "doc-a.json", [])

    findings = scan_human_annotation_missingness(
        source_root, input_path, annotators=["stig"]
    )

    kinds = [finding.kind for finding in findings]
    assert kinds.count("missing_label") == 1
    assert kinds.count("missing_required_fields") == 1
    assert kinds.count("duplicate_document_file") == 1
    assert kinds.count("empty_annotation_file") == 1
    assert kinds.count("missing_document") == 1
    missing_label = next(
        finding for finding in findings if finding.kind == "missing_label"
    )
    assert missing_label.text is None
    assert missing_label.category == "Date"
    assert missing_label.span_index == 0


def test_can_include_text_and_write_jsonl(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"doc_id": "doc-a", "text": "Date: 16/06/2025"}) + "\n",
        encoding="utf-8",
    )
    source_root = tmp_path / "legacy"
    _write_json(
        source_root / "stig1" / "spans" / "doc-a.json",
        [{"begin": 6, "end": 16, "text": "16/06/2025", "Category": "Date"}],
    )

    findings = scan_human_annotation_missingness(
        source_root,
        input_path,
        annotators=["stig"],
        include_text=True,
    )
    stream = io.StringIO()
    write_report(findings, stream, output_format="jsonl")
    row = json.loads(stream.getvalue())
    assert row["kind"] == "missing_label"
    assert row["text"] == "16/06/2025"
