from __future__ import annotations

from deid_battery.runners.phase_timing import (
    combine_reports,
    read_report,
    warmup_docs,
    write_report,
)


def test_report_round_trip_and_chunk_aggregation(tmp_path):
    path = tmp_path / "timing.json"
    params = {"_timing_path": str(path)}
    write_report(
        params,
        setup_seconds=2.0,
        warmup_seconds=0.5,
        inference_seconds=4.0,
        warmup_documents=1,
    )
    first = read_report(path)
    combined = combine_reports(
        [first, {**first, "service_setup": "local_measured"}]
    )

    assert first["setup_seconds"] == 2.0
    assert combined == {
        "setup_seconds": 4.0,
        "warmup_seconds": 1.0,
        "inference_seconds": 8.0,
        "warmup_documents": 2,
        "service_setup": "local_measured",
    }


def test_warmup_selects_only_non_empty_docs():
    docs = [
        {"doc_id": "empty", "text": ""},
        {"doc_id": "one", "text": "one"},
        {"doc_id": "two", "text": "two"},
    ]

    assert [doc["doc_id"] for doc in warmup_docs({}, docs, default=1)] == ["one"]
    assert [doc["doc_id"] for doc in warmup_docs({"warmup_docs": 2}, docs, default=0)] == [
        "one",
        "two",
    ]
    assert warmup_docs({"warmup_docs": 0}, docs, default=1) == []
