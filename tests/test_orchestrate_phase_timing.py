from __future__ import annotations

import json

import yaml

from deid_battery import orchestrate
from deid_battery.runners.phase_timing import write_report
from deid_battery.timing import load


def test_orchestrator_records_warm_end_to_end_after_postprocessing(tmp_path, monkeypatch):
    input_path = tmp_path / "input.jsonl"
    input_path.write_text(
        json.dumps({"doc_id": "doc-1", "text": "Jan"}) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "out"
    timings_path = tmp_path / "timings.yaml"
    config_path = tmp_path / "battery.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "input": str(input_path),
                "output_dir": str(output_dir),
                "timings": str(timings_path),
                "metadata": {"source": "none"},
                "models": [{"id": "neural", "runner": "fake", "params": {}}],
                "postprocess": {"enabled": False},
                "evaluate": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )

    def fake_runner(docs, params):
        write_report(
            params,
            setup_seconds=3.0,
            warmup_seconds=1.0,
            inference_seconds=2.0,
            warmup_documents=1,
        )
        return {doc["doc_id"]: [] for doc in docs}

    monkeypatch.setattr(orchestrate, "get_runner", lambda _: fake_runner)
    orchestrate.run(config_path)

    row = load(timings_path)["neural"][0]
    assert row["timing_scope"] == "warm_end_to_end"
    assert row["setup_seconds"] == 3.0
    assert row["warmup_seconds"] == 1.0
    assert row["inference_seconds"] == 2.0
    assert row["seconds"] == row["warm_end_to_end_seconds"]
    assert row["cold_end_to_end_seconds"] >= row["warm_end_to_end_seconds"] + 3.0
    assert (output_dir / "neural" / "raw.jsonl").exists()
    assert (output_dir / "neural" / "by_doc.jsonl").exists()
