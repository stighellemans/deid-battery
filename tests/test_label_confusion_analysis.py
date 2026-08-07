from __future__ import annotations

from deid_battery._vendor import evaluate_quantity_deid as eq
from deid_battery._vendor import evaluation_plots as ep


def _gold_item(label: str, begin: int, end: int, index: int) -> dict:
    return {
        "document_id": "doc-1",
        "item_id": f"gold-{index}",
        "item_kind": "gold",
        "gold": {"label": label},
        "review_range": {"begin": begin, "end": end},
        "segments": [{"begin": begin, "end": end, "category": "core"}],
    }


def _prediction(label: str, begin: int, end: int) -> dict:
    return {
        "document_id": "doc-1",
        "begin": begin,
        "end": end,
        "label": label,
        "source_id": "model",
    }


def test_span_confusion_uses_geometry_only_and_one_to_one_matching() -> None:
    assert not hasattr(eq, "build_core_pii_label_confusion")
    gold = [
        _gold_item("Name:Patient", 0, 5, 0),
        _gold_item("Date", 6, 10, 1),
        _gold_item("ID:Patient", 12, 15, 2),
    ]
    predictions = {
        "doc-1": [
            _prediction("Name:Caregiver", 0, 4),
            _prediction("Name:Patient", 1, 3),
            _prediction("Date", 6, 10),
        ]
    }

    result = eq.build_core_pii_span_label_confusion(gold, predictions, set())

    assert result["total_core_pii_spans"] == 3
    assert result["matched_core_pii_spans"] == 2
    assert result["missed_core_pii_spans"] == 1
    assert result["metrics"]["exact_correct_core_pii_spans"] == 1
    assert result["metrics"]["coarse_correct_core_pii_spans"] == 2
    name_row = next(
        row
        for row in result["by_annotation_label"]
        if row["annotation_label"] == "Name:Patient"
    )
    assert name_row["assigned_labels"] == [
        {
            "prediction_label": "Name:Caregiver",
            "spans": 1,
            "fraction_of_matched_row_spans": 1.0,
            "fraction_of_total_row_spans": 1.0,
        }
    ]


def test_dedicated_analysis_writes_span_raw_data_and_plots(tmp_path) -> None:
    gold = [_gold_item("Name:Patient", 0, 5, 0), _gold_item("Date", 6, 10, 1)]
    predictions = {
        "doc-1": [
            _prediction("Name:Caregiver", 0, 5),
            _prediction("Date", 6, 10),
        ]
    }
    span_confusion = eq.build_core_pii_span_label_confusion(gold, predictions, set())
    payload = {
        "results": [
            {
                "display_annotation_id": "model",
                "source_name": "Example model",
                "result": {
                    "core_pii_span_label_confusion": span_confusion,
                },
            }
        ],
        "source_names": {"model": "Example model"},
        "source_order": ["model"],
    }

    ep.save_label_confusion_analysis(payload, tmp_path)

    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "confusion_long.csv").is_file()
    assert (tmp_path / "plots" / "model.png").is_file()
    assert not (tmp_path / "character").exists()
    assert not (tmp_path / "span").exists()
    assert (tmp_path / "definitions.json").is_file()
    assert (tmp_path / "README.md").is_file()
    assert '"character_matching"' not in (tmp_path / "definitions.json").read_text()


def test_dedicated_analysis_can_split_raw_data_and_plots(tmp_path) -> None:
    gold = [_gold_item("Name:Patient", 0, 5, 0)]
    predictions = {"doc-1": [_prediction("Name:Patient", 0, 5)]}
    payload = {
        "results": [
            {
                "display_annotation_id": "model",
                "source_name": "Example model",
                "result": {
                    "core_pii_span_label_confusion":
                        eq.build_core_pii_span_label_confusion(gold, predictions, set()),
                },
            }
        ],
        "source_names": {"model": "Example model"},
        "source_order": ["model"],
    }
    raw_dir = tmp_path / "raw" / "label_confusion"
    plots_dir = tmp_path / "plots" / "label_confusion"

    ep.save_label_confusion_analysis(payload, raw_dir, plots_dir)

    assert (raw_dir / "summary.csv").is_file()
    assert (raw_dir / "confusion_long.csv").is_file()
    assert not (raw_dir / "plots").exists()
    assert not (raw_dir / "character").exists()
    assert (plots_dir / "model.png").is_file()
    assert not (plots_dir / "character").exists()
