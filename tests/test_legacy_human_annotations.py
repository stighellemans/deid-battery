from __future__ import annotations

import json
from pathlib import Path

import pytest

from deid_battery._vendor.data_readers import read_annotator_level_jsonl
from deid_battery.legacy_human_annotations import (
    LegacyAnnotationError,
    convert_legacy_human_annotations,
)
from deid_battery.schema import read_jsonl


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _write_input(path: Path) -> None:
    rows = [
        {"doc_id": "doc-a", "text": "Date: 16/06/2025"},
        {"doc_id": "doc-b", "text": "Patient: Ada Lovelace"},
    ]
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_merges_numbered_batches_and_normalizes_spans(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    output_dir = tmp_path / "out"
    _write_input(input_path)
    _write_json(
        source_root / "stig1" / "spans" / "doc-a.json",
        [{"begin": 6, "end": 16, "label": "Date", "text": "16/06/2025", "Category": "Date"}],
    )
    _write_json(
        source_root / "stig2" / "spans" / "doc-b.json",
        [
            {
                "begin": 9,
                "end": 21,
                "label": "Name:Patient",
                "text": "Ada Lovelace",
                "Category": "Name",
            }
        ],
    )

    summaries = convert_legacy_human_annotations(
        source_root,
        input_path,
        output_dir,
        annotators=["stig"],
    )

    assert summaries[0].documents == 2
    assert summaries[0].spans == 2
    assert summaries[0].incomplete_documents == 0
    assert summaries[0].missing_documents == 0
    assert summaries[0].batches == ("stig1", "stig2")
    rows = read_jsonl(output_dir / "stig.jsonl")
    assert [row["doc_id"] for row in rows] == ["doc-a", "doc-b"]
    assert rows[0]["entities"][0] == {
        "begin": 6,
        "end": 16,
        "label": "Date",
        "text": "16/06/2025",
        "category": "Date",
        "subtype": None,
    }
    assert rows[1]["entities"][0]["category"] == "Name"
    assert rows[1]["entities"][0]["subtype"] == "Patient"
    manifest = json.loads((output_dir / "coverage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["partial_import"] is False
    assert manifest["common_complete_documents"] == 2


def test_rejects_span_text_that_does_not_match_input(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    _write_input(input_path)
    _write_json(
        source_root / "stig1" / "spans" / "doc-a.json",
        [{"begin": 6, "end": 16, "label": "Date", "text": "wrong", "Category": "Date"}],
    )

    with pytest.raises(LegacyAnnotationError, match="text does not match"):
        convert_legacy_human_annotations(
            source_root,
            input_path,
            tmp_path / "out",
            annotators=["stig"],
            require_complete=False,
        )


def test_complete_mode_rejects_missing_documents(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    _write_input(input_path)
    _write_json(source_root / "stig1" / "spans" / "doc-a.json", [])

    with pytest.raises(LegacyAnnotationError, match="missing 1 of 2"):
        convert_legacy_human_annotations(
            source_root,
            input_path,
            tmp_path / "out",
            annotators=["stig"],
        )


def test_rejects_conflicting_duplicate_documents(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    _write_input(input_path)
    _write_json(source_root / "stig1" / "spans" / "doc-a.json", [])
    _write_json(
        source_root / "stig2" / "spans" / "doc-a.json",
        [{"begin": 6, "end": 16, "label": "Date", "text": "16/06/2025", "Category": "Date"}],
    )

    with pytest.raises(LegacyAnnotationError, match="conflicting duplicate"):
        convert_legacy_human_annotations(
            source_root,
            input_path,
            tmp_path / "out",
            annotators=["stig"],
            require_complete=False,
        )


def test_validates_all_annotators_before_replacing_outputs(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    output_dir = tmp_path / "out"
    _write_input(input_path)
    _write_json(source_root / "stig1" / "spans" / "doc-a.json", [])
    _write_json(source_root / "stig1" / "spans" / "doc-b.json", [])
    _write_json(source_root / "tomstroobants1" / "spans" / "doc-a.json", [])
    output_dir.mkdir()
    existing = output_dir / "stig.jsonl"
    existing.write_text("existing output\n", encoding="utf-8")

    with pytest.raises(LegacyAnnotationError, match="tomstroobants: missing 1 of 2"):
        convert_legacy_human_annotations(source_root, input_path, output_dir)

    assert existing.read_text(encoding="utf-8") == "existing output\n"
    assert not (output_dir / "tomstroobants.jsonl").exists()


def test_partial_mode_retains_unlabeled_geometry_and_records_coverage(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    output_dir = tmp_path / "out"
    _write_input(input_path)
    _write_json(
        source_root / "stig1" / "spans" / "doc-a.json",
        [{"begin": 6, "end": 16, "text": "16/06/2025", "Category": "Date"}],
    )
    _write_json(
        source_root / "stig1" / "spans" / "doc-b.json",
        [
            {
                "begin": 9,
                "end": 21,
                "label": "Name:Patient",
                "text": "Ada Lovelace",
                "Category": "Name",
            }
        ],
    )

    summaries = convert_legacy_human_annotations(
        source_root,
        input_path,
        output_dir,
        annotators=["stig"],
        require_complete=False,
    )

    assert summaries[0].documents == 2
    assert summaries[0].incomplete_documents == 1
    assert summaries[0].missing_documents == 0
    rows = read_jsonl(output_dir / "stig.jsonl")
    assert [row["doc_id"] for row in rows] == ["doc-a", "doc-b"]
    assert rows[0]["entities"][0]["label"] == "(missing label)"
    assert rows[0]["entities"][0]["category"] == "(missing label)"
    assert rows[0]["entities"][0]["text"] == "16/06/2025"
    normalized_records, issues = read_annotator_level_jsonl(output_dir / "stig.jsonl")
    assert issues == []
    assert normalized_records[0]["annotations"][0]["label"] == "(missing label)"
    manifest = json.loads((output_dir / "coverage_manifest.json").read_text(encoding="utf-8"))
    incomplete = manifest["annotators"]["stig"]["incomplete_documents_scored"]
    assert incomplete == [
        {
            "doc_id": "doc-a",
            "source_file": "stig1/spans/doc-a.json",
            "unlabeled_span_indices": [0],
        }
    ]
    assert manifest["annotators"]["stig"]["unlabeled_spans_assigned_missing_label"] == 1
    assert (output_dir / "common_complete_doc_ids.txt").read_text(encoding="utf-8") == "doc-b\n"


def test_partial_mode_writes_missing_documents_as_empty_submissions(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    output_dir = tmp_path / "out"
    _write_input(input_path)
    _write_json(source_root / "stig1" / "spans" / "doc-a.json", [])

    summaries = convert_legacy_human_annotations(
        source_root,
        input_path,
        output_dir,
        annotators=["stig"],
        require_complete=False,
    )

    assert summaries[0].documents == 2
    assert summaries[0].missing_documents == 1
    rows = read_jsonl(output_dir / "stig.jsonl")
    assert rows[1] == {"doc_id": "doc-b", "num_entities": 0, "entities": []}
    manifest = json.loads((output_dir / "coverage_manifest.json").read_text(encoding="utf-8"))
    assert manifest["annotators"]["stig"]["missing_input_documents_scored_as_empty"] == [
        "doc-b"
    ]


def test_strict_mode_explains_how_to_handle_unlabeled_span(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    source_root = tmp_path / "legacy"
    _write_input(input_path)
    _write_json(
        source_root / "stig1" / "spans" / "doc-a.json",
        [{"begin": 6, "end": 16, "text": "16/06/2025", "Category": "Date"}],
    )

    with pytest.raises(LegacyAnnotationError, match="Rerun with --allow-partial"):
        convert_legacy_human_annotations(
            source_root,
            input_path,
            tmp_path / "out",
            annotators=["stig"],
        )
