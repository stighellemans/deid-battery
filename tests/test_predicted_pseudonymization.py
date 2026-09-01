import csv
import json
from dataclasses import dataclass
from pathlib import Path

from deid_battery.pseudonymization_eval import (
    PredictedSpan,
    build_predicted_aggregate_tables,
    score_predicted_spans,
    validate_predicted_safe_export,
    write_predicted_safe_export,
)


@dataclass(frozen=True)
class Gold:
    document_id: str
    label: str
    begin: int
    end: int


@dataclass(frozen=True)
class Settings:
    document_creation_date: str = "2025-01-15"
    date_shift_days: int = 371
    birthdate_replacement_mode: str = "age"


def test_predicted_outcomes_separate_protocol_and_residual_exposure():
    gold = [
        Gold("d", "Date", 0, 10),
        Gold("d", "Date", 20, 30),
        Gold("d", "Age_Birthdate", 40, 50),
        Gold("d", "Date", 60, 70),
        Gold("d", "Date", 80, 90),
        Gold("d", "Age_Birthdate", 100, 110),
    ]
    predictions = [
        PredictedSpan("d", 0, "Date", 0, 10),
        PredictedSpan("d", 1, "Name:Patient", 20, 30),
        PredictedSpan("d", 2, "Age_Birthdate", 40, 45),
        PredictedSpan("d", 3, "Date", 60, 65),
        PredictedSpan("d", 4, "Date", 65, 70),
        PredictedSpan("d", 5, "Date", 80, 90),
    ]
    transformations = {
        ("d", 0): (True, None),
        ("d", 2): (True, None),
        ("d", 3): (True, None),
        ("d", 4): (True, None),
        ("d", 5): (False, "unsupported_or_invalid_format"),
    }

    outcomes = score_predicted_spans(gold, predictions, transformations)

    assert [outcome.failure_reason for outcome in outcomes] == [
        None,
        "incorrect_label",
        "incomplete_prediction_coverage",
        "fragmented_prediction_coverage",
        "unsupported_or_invalid_format",
        "no_prediction_overlap",
    ]
    assert [outcome.fully_redacted for outcome in outcomes] == [
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    summary, failures = build_predicted_aggregate_tables(outcomes)
    overall = summary[0]
    assert overall == {
        "label": "Overall",
        "gold_spans": 6,
        "end_to_end_valid": 1,
        "end_to_end_failed": 5,
        "end_to_end_valid_rate": 0.166667,
        "end_to_end_failure_rate": 0.833333,
        "fully_redacted": 4,
        "residual_exposure": 2,
        "fully_redacted_rate": 0.666667,
        "residual_exposure_rate": 0.333333,
    }
    assert sum(row["count"] for row in failures if row["label"] == "Overall") == 5


def test_predicted_export_is_aggregate_only_and_validated(tmp_path: Path):
    outcomes = [
        # Deliberately use no document id or source surface in the public outcome.
        # The export contract has no place where either could be serialized.
        score_predicted_spans(
            [Gold("sensitive-document", "Date", 0, 10)],
            [PredictedSpan("sensitive-document", 0, "Date", 0, 10)],
            {("sensitive-document", 0): (True, None)},
        )[0]
    ]
    export_dir = tmp_path / "export"
    manifest = write_predicted_safe_export(
        export_dir,
        outcomes,
        Settings(),
        prediction_source="synthetic@meta",
    )

    assert manifest["privacy_check"] == "passed"
    assert validate_predicted_safe_export(export_dir)["files"] == [
        "failure_reasons.csv",
        "methodology.json",
        "summary.csv",
    ]
    combined = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(export_dir.iterdir())
    )
    assert "sensitive-document" not in combined
    with (export_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["end_to_end_failure_rate"] == "0.0"
    assert json.loads((export_dir / "methodology.json").read_text())["evaluation_scope"] == (
        "predicted_spans_end_to_end"
    )


def test_predicted_export_rejects_uncontrolled_source_id(tmp_path: Path):
    try:
        write_predicted_safe_export(
            tmp_path,
            [],
            Settings(),
            prediction_source="patient name",
        )
    except ValueError as error:
        assert "source ids" in str(error)
    else:
        raise AssertionError("unsafe source id was accepted")
