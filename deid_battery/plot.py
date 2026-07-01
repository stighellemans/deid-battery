"""Essential-recall + false-positive-burden plot, from an evaluation payload.

Wraps the vendored ``evaluation_plots`` (matplotlib). Sources with zero
predictions are dropped from the plot. Returns the summary DataFrame.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "_vendor"
sys.path.insert(0, str(_VENDOR))


def plot(payload: dict, out_path):
    import evaluation_plots as ep

    summary = ep.build_summary_frame(payload)
    if not summary.empty:
        summary = summary[summary["prediction_span_count"] > 0].reset_index(drop=True)
    ep.plot_recall_and_fp_burden(summary, output_path=out_path)
    return summary


def plot_recall_by_gold_label(payload: dict, out_path):
    """Essential-recall heatmap: gold label (row) x source (col)."""
    import evaluation_plots as ep
    matrix, counts = ep.build_recall_matrix(
        payload, group_name="by_gold_label", row_key="gold_label",
        metric_key="essential_recall", count_key="gold_span_count")
    if matrix.empty:
        return None
    return ep.plot_recall_heatmap(matrix, counts, "Essential recall by gold label",
                                  "spans", output_path=out_path)


def plot_recall_by_subannotation_category(payload: dict, out_path):
    """Recall heatmap: subannotation category (row) x source (col)."""
    import evaluation_plots as ep
    matrix, counts = ep.build_recall_matrix(
        payload, group_name="by_subannotation_category", row_key="category",
        metric_key=None, count_key="total_chars",
        row_filter=ep._is_plottable_subannotation_category)
    if matrix.empty:
        return None
    return ep.plot_recall_heatmap(matrix, counts, "Recall by subannotation category",
                                  "chars", output_path=out_path)
