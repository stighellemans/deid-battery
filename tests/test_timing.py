from __future__ import annotations

import yaml

from deid_battery.timing import (
    compose_phase_metrics,
    load,
    measured_value,
    record_measured,
)


def test_compose_phase_metrics_separates_warm_cold_and_full_run():
    details = compose_phase_metrics(
        runner_seconds=16.0,
        raw_write_seconds=1.0,
        postprocess_seconds=2.0,
        runner_report={
            "setup_seconds": 5.0,
            "warmup_seconds": 3.0,
            "inference_seconds": 7.0,
            "warmup_documents": 1,
            "service_setup": "local_measured",
        },
    )

    assert details["cold_overhead_seconds"] == 1.0
    assert details["warm_end_to_end_seconds"] == 10.0
    assert details["cold_end_to_end_seconds"] == 16.0
    assert details["measured_full_run_seconds"] == 19.0
    assert details["timing_scope"] == "warm_end_to_end"


def test_record_measured_preserves_manual_rows_and_writes_phase_details(tmp_path):
    path = tmp_path / "timings.yaml"
    path.write_text(
        yaml.safe_dump(
            {"model": [{"device": "gpu", "seconds": 99, "source": "manual"}]},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    details = compose_phase_metrics(
        16.0,
        1.0,
        2.0,
        {"setup_seconds": 5.0, "warmup_seconds": 3.0, "inference_seconds": 7.0},
    )

    record_measured(path, "model", "gpu", 10.0, n_docs=12, details=details)
    rows = load(path)["model"]

    assert rows[0] == {"device": "gpu", "seconds": 99, "source": "manual"}
    assert rows[1]["seconds"] == 10.0
    assert rows[1]["timing_scope"] == "warm_end_to_end"
    assert rows[1]["cold_end_to_end_seconds"] == 16.0
    assert measured_value(rows, "setup_seconds") == 5.0
