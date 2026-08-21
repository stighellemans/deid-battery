from deid_battery._vendor import evaluate_quantity_deid as quantity
from deid_battery._vendor import evaluation_plots


def test_hard_negative_documents_count_in_non_pii_denominator():
    item = {
        "document_id": "positive",
        "item_kind": "gold",
        "gold": {"begin": 1, "end": 3},
        "review_range": {"begin": 1, "end": 3},
        "segments": [{"begin": 1, "end": 3, "category": "annotation"}],
    }

    summary = quantity.summarize_gold_characters(
        [item], document_lengths={"positive": 5, "hard-negative": 10}
    )

    assert summary["total_document_chars"] == 15
    assert summary["pii_chars"] == 2
    assert summary["non_pii_chars"] == 13


def test_primary_plot_can_use_plain_annotation_recall():
    import pandas as pd

    summary = pd.DataFrame(
        [
            {
                "source": "model",
                "overall_recall": 0.75,
                "machine_only_redaction_rate": 0.01,
                "boundary_overflow_redaction_rate": 0.02,
                "non_pii_redaction_rate": 0.03,
            }
        ]
    )

    figure = evaluation_plots.plot_recall_and_non_pii_redaction(
        summary,
        recall_key="overall_recall",
        recall_label="Annotation recall",
    )

    assert figure.axes[0].get_xlabel() == "Annotation recall"
    assert figure.axes[0].get_title() == "Annotation Recall"
