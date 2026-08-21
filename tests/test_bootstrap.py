from __future__ import annotations

import math

from deid_battery.bootstrap import bootstrap_payload


def _payload():
    def source(source_id, covered, redacted):
        details = []
        for doc_id, doc_covered, doc_redacted in zip(
            ("positive-a", "positive-b", "hard-negative"), covered, redacted
        ):
            if doc_id != "hard-negative":
                details.append(
                    {
                        "row_kind": "gold_span",
                        "document_id": doc_id,
                        "gold_range": {"begin": 1, "end": 3},
                        "core_pii_recall": {
                            "total_chars": 2,
                            "covered_chars": doc_covered,
                        },
                        "overall_recall": {
                            "total_chars": 2,
                            "covered_chars": doc_covered,
                        },
                    }
                )
            if doc_redacted:
                details.append(
                    {
                        "row_kind": "non_pii_redaction",
                        "document_id": doc_id,
                        "non_pii_redacted_chars": doc_redacted,
                    }
                )
        return {
            "annotation_id": source_id,
            "source_name": source_id.upper(),
            "result": {"details": details},
        }

    return {
        "gold_character_summary": {"non_pii_chars": 16},
        "results": [
            source("model-a", [2, 1, 0], [1, 0, 2]),
            source("model-b", [1, 0, 0], [0, 2, 4]),
        ],
    }


def test_bootstrap_point_estimates_include_hard_negative_documents():
    estimates, differences, methodology = bootstrap_payload(
        _payload(),
        {"positive-a": 5, "positive-b": 5, "hard-negative": 10},
        replicates=500,
        seed=17,
    )

    indexed = estimates.set_index(["annotation_id", "metric"])
    assert math.isclose(indexed.loc[("model-a", "core_pii_recall"), "estimate"], 3 / 4)
    assert math.isclose(
        indexed.loc[("model-a", "non_pii_redaction_rate"), "estimate"], 3 / 16
    )
    assert math.isclose(
        indexed.loc[("model-b", "non_pii_redaction_rate"), "estimate"], 6 / 16
    )
    recall_difference = differences[
        differences["metric"] == "core_pii_recall"
    ].iloc[0]
    assert math.isclose(recall_difference["difference_a_minus_b"], 0.5)
    assert methodology["sampling_unit"] == "document"
    assert methodology["paired_resampling"] is True


def test_bootstrap_is_deterministic_for_a_fixed_seed():
    args = (
        _payload(),
        {"positive-a": 5, "positive-b": 5, "hard-negative": 10},
    )
    first = bootstrap_payload(*args, replicates=200, seed=7)
    second = bootstrap_payload(*args, replicates=200, seed=7)

    assert first[0].equals(second[0])
    assert first[1].equals(second[1])
    assert first[2] == second[2]
