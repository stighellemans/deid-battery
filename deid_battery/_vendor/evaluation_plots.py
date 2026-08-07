from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib import ticker
from matplotlib.patches import Rectangle


SOURCE_LABELS = {
    "stig": "Annotator 1",
    "tomstroobants": "Annotator 2",
    "robert_2023_dutch_base": "RobBERT-2023 Dutch Base",
    "meddeid-500": "MedDeID-500",
    "med_roberta.nl": "MedRoBERTa.nl",
    "openai-pii-filter": "OpenAI PII Filter",
    "llm": "LLM",
    "custom_uza_belgian_deduce_annotations": "Custom UZA Belgian Deduce",
    "belgian_deduce_annotations": "Belgian Deduce",
    "deduce_annotations": "Deduce",
}

SOURCE_ORDER = [
    "Annotator 1",
    "Annotator 2",
    "RobBERT-2023",
    "RobBERT-2023 Dutch Base",
    "MedDeID-500",
    "MedRoBERTa.nl",
    "OpenAI PII Filter",
    "OpenAI Privacy Filter",
    "OpenMed Multilingual Privacy Filter",
    "LLM",
    "Qwen3-8B-AWQ",
    "Custom UZA Belgian Deduce",
    "Belgian Deduce",
    "Deduce",
]

NON_PII_REDACTION_RATE_LABEL = "Non-PII redaction rate"
FALSE_POSITIVE_SPANS_LABEL = "False positives"
PII_BOUNDARY_EXTENSIONS_LABEL = "Boundary extensions"
HEATMAP_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "deid_accessible_red_green",
    ["#CC3311", "#F5F1E6", "#009E73"],
)
CONFUSION_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "deid_confusion_blue", ["#f5f8fa", "#b9d7e5", "#457b9d"]
)


def source_display_name(annotation_id: str) -> str:
    return SOURCE_LABELS.get(annotation_id, annotation_id.replace("_", " "))


def source_display_names(payload: dict[str, Any]) -> dict[str, str]:
    annotation_ids = [str(entry["display_annotation_id"]) for entry in payload["results"]]
    configured_source_names = {
        str(annotation_id): str(source_name)
        for annotation_id, source_name in payload.get("source_names", {}).items()
        if source_name
    }
    entry_source_names = {
        str(entry["display_annotation_id"]): str(entry["source_name"])
        for entry in payload["results"]
        if entry.get("source_name")
    }
    base_names = {
        annotation_id: entry_source_names.get(
            annotation_id,
            configured_source_names.get(annotation_id, source_display_name(annotation_id)),
        )
        for annotation_id in annotation_ids
    }
    counts = Counter(base_names.values())
    return {
        annotation_id: (
            base_name
            if counts[base_name] == 1
            else f"{base_name} ({annotation_id})"
        )
        for annotation_id, base_name in base_names.items()
    }


def source_sort_key(source: str, source_order: list[str] | None = None) -> tuple[int, int, str]:
    if source_order and source in source_order:
        return (0, source_order.index(source), source)
    if source in SOURCE_ORDER:
        return (1, SOURCE_ORDER.index(source), source)
    return (2, len(SOURCE_ORDER), source)


def source_order_from_payload(
    payload: dict[str, Any],
    display_names: dict[str, str] | None = None,
) -> list[str]:
    configured_annotation_order = payload.get("source_order") or []
    if not configured_annotation_order:
        return []

    display_names = display_names or source_display_names(payload)
    ordered_sources: list[str] = []
    seen: set[str] = set()

    for annotation_id in configured_annotation_order:
        source = display_names.get(str(annotation_id))
        if source and source not in seen:
            ordered_sources.append(source)
            seen.add(source)

    for entry in payload.get("results", []):
        annotation_id = str(entry["display_annotation_id"])
        source = display_names.get(annotation_id)
        if source and source not in seen:
            ordered_sources.append(source)
            seen.add(source)

    return ordered_sources


def _source_sorter(source_order: list[str] | None = None):
    return lambda values: values.map(lambda source: source_sort_key(str(source), source_order))


def _with_source_order(frame: pd.DataFrame, source_order: list[str]) -> pd.DataFrame:
    frame.attrs["source_order"] = source_order
    return frame


def _sort_frame_by_source(frame: pd.DataFrame) -> pd.DataFrame:
    source_order = frame.attrs.get("source_order")
    sorted_frame = frame.sort_values("source", key=_source_sorter(source_order)).reset_index(drop=True)
    return _with_source_order(sorted_frame, source_order or [])


def _ordered_source_names(payload: dict[str, Any], display_names: dict[str, str]) -> list[str]:
    source_order = source_order_from_payload(payload, display_names)
    source_names = list(dict.fromkeys(display_names.values()))
    return sorted(source_names, key=lambda source: source_sort_key(source, source_order))


def _ordered_unique_sources(frame: pd.DataFrame) -> list[str]:
    source_order = frame.attrs.get("source_order") or []
    sources = [str(source) for source in frame["source"].dropna().unique()]
    return sorted(sources, key=lambda source: source_sort_key(source, source_order))


def build_summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    display_names = source_display_names(payload)
    source_order = source_order_from_payload(payload, display_names)
    for entry in payload["results"]:
        annotation_id = entry["display_annotation_id"]
        result = entry["result"]
        label_metrics = result.get("core_pii_span_label_confusion", {}).get("metrics", {})
        machine_only_redaction = result["non_pii_redaction_summaries"]["machine_only_redaction"]["non_pii_redacted_chars"]
        boundary_overflow_redaction = result["non_pii_redaction_summaries"]["boundary_overflow_redaction"]["non_pii_redacted_chars"]
        machine_only_redaction_fraction = _non_pii_redaction_rate(result, "machine_only_redaction", machine_only_redaction)
        boundary_overflow_redaction_fraction = _non_pii_redaction_rate(result, "boundary_overflow_redaction", boundary_overflow_redaction)
        rows.append(
            {
                "annotation_id": annotation_id,
                "source": display_names[str(annotation_id)],
                "core_pii_recall": _recall_value(result, "core_pii_recall"),
                "overall_recall": result["overall_recall"]["recall"],
                "machine_only_redaction_chars": machine_only_redaction,
                "boundary_overflow_redaction_chars": boundary_overflow_redaction,
                "non_pii_redacted_chars": machine_only_redaction + boundary_overflow_redaction,
                "machine_only_redaction_rate": machine_only_redaction_fraction,
                "boundary_overflow_redaction_rate": boundary_overflow_redaction_fraction,
                "non_pii_redaction_rate": _sum_optional(machine_only_redaction_fraction, boundary_overflow_redaction_fraction),
                "prediction_span_count": result["summary"]["prediction_span_count"],
                "exact_label_recall": label_metrics.get("exact_label_recall", np.nan),
                "coarse_label_recall": label_metrics.get("coarse_label_recall", np.nan),
                "exact_label_accuracy_matched": label_metrics.get("exact_label_accuracy_matched", np.nan),
                "coarse_label_accuracy_matched": label_metrics.get("coarse_label_accuracy_matched", np.nan),
            }
        )

    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df

    summary_df = summary_df.sort_values(
        "source",
        key=_source_sorter(source_order),
    ).reset_index(drop=True)
    return _with_source_order(summary_df, source_order)


def build_label_quality_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    display_names = source_display_names(payload)
    source_order = source_order_from_payload(payload, display_names)
    for entry in payload["results"]:
        annotation_id = str(entry["display_annotation_id"])
        confusion = entry["result"].get("core_pii_span_label_confusion", {})
        metrics = confusion.get("metrics", {})
        rows.append(
            {
                "annotation_id": annotation_id,
                "source": display_names[annotation_id],
                "total_core_pii_spans": confusion.get("total_core_pii_spans", np.nan),
                "matched_core_pii_spans": confusion.get("matched_core_pii_spans", np.nan),
                "missed_core_pii_spans": confusion.get("missed_core_pii_spans", np.nan),
                "span_detection_recall": confusion.get("span_detection_recall", np.nan),
                "exact_correct_core_pii_spans": metrics.get("exact_correct_core_pii_spans", np.nan),
                "coarse_correct_core_pii_spans": metrics.get("coarse_correct_core_pii_spans", np.nan),
                "exact_label_recall": metrics.get("exact_label_recall", np.nan),
                "coarse_label_recall": metrics.get("coarse_label_recall", np.nan),
                "exact_label_accuracy_matched": metrics.get("exact_label_accuracy_matched", np.nan),
                "coarse_label_accuracy_matched": metrics.get("coarse_label_accuracy_matched", np.nan),
            }
        )

    columns = [
        "annotation_id",
        "source",
        "total_core_pii_spans",
        "matched_core_pii_spans",
        "missed_core_pii_spans",
        "span_detection_recall",
        "exact_correct_core_pii_spans",
        "coarse_correct_core_pii_spans",
        "exact_label_recall",
        "coarse_label_recall",
        "exact_label_accuracy_matched",
        "coarse_label_accuracy_matched",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame = frame.sort_values("source", key=_source_sorter(source_order)).reset_index(drop=True)
    return _with_source_order(frame, source_order)


def build_label_confusion_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    display_names = source_display_names(payload)
    source_order = source_order_from_payload(payload, display_names)
    for entry in payload["results"]:
        annotation_id = str(entry["display_annotation_id"])
        source = display_names[annotation_id]
        confusion = entry["result"].get("core_pii_span_label_confusion", {})
        for row in confusion.get("by_annotation_label", []):
            row_label = str(row.get("annotation_label", ""))
            for assigned in row.get("assigned_labels") or []:
                rows.append(
                    {
                        "annotation_id": annotation_id,
                        "source": source,
                        "annotation_label": row_label,
                        "prediction_label": _display_prediction_label(assigned.get("prediction_label")),
                        "spans": int(assigned.get("spans") or 0),
                        "total_core_pii_spans": int(row.get("total_core_pii_spans") or 0),
                        "matched_core_pii_spans": int(row.get("matched_core_pii_spans") or 0),
                        "missed_core_pii_spans": int(row.get("missed_core_pii_spans") or 0),
                        "span_detection_recall": row.get("span_detection_recall", np.nan),
                        "fraction_of_matched_row_spans": assigned.get("fraction_of_matched_row_spans", np.nan),
                        "fraction_of_total_row_spans": assigned.get("fraction_of_total_row_spans", np.nan),
                    }
                )

    columns = [
        "annotation_id",
        "source",
        "annotation_label",
        "prediction_label",
        "spans",
        "total_core_pii_spans",
        "matched_core_pii_spans",
        "missed_core_pii_spans",
        "span_detection_recall",
        "fraction_of_matched_row_spans",
        "fraction_of_total_row_spans",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["source", "annotation_label", "spans", "prediction_label"],
        ascending=[True, True, False, True],
        key=lambda values: _source_sorter(source_order)(values) if values.name == "source" else values,
    ).reset_index(drop=True)
    return _with_source_order(frame, source_order)


def build_gold_character_summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    summary = gold_character_summary(payload)
    components = summary.get("components", {})
    rows = [
        {
            "character_type": "PII",
            "chars": summary.get("pii_chars", 0),
            "fraction": summary.get("pii_fraction", np.nan),
            "component": "gold_annotation_ranges",
        },
        {
            "character_type": "non-PII",
            "chars": summary.get("non_pii_chars", 0),
            "fraction": summary.get("non_pii_fraction", np.nan),
            "component": "outside_gold_annotation_ranges",
        },
    ]
    frame = pd.DataFrame(rows)
    frame.attrs["components"] = components
    return frame


def build_non_pii_redaction_label_frame(
    payload: dict[str, Any],
    redaction_bucket: str | None = None,
) -> pd.DataFrame:
    """Summarize non-PII redactions by the prediction label assigned by the model."""
    rows = []
    display_names = source_display_names(payload)
    source_order = source_order_from_payload(payload, display_names)
    for entry in payload["results"]:
        annotation_id = str(entry["display_annotation_id"])
        source = display_names[annotation_id]
        grouped: dict[tuple[str, str], dict[str, Any]] = {}

        for detail in entry["result"].get("details", []):
            if detail.get("row_kind") != "non_pii_redaction":
                continue
            bucket = str(detail.get("redaction_bucket", ""))
            if redaction_bucket is not None and bucket != redaction_bucket:
                continue

            prediction_label = _display_prediction_label(detail.get("prediction_label"))
            key = (bucket, prediction_label)
            group = grouped.setdefault(
                key,
                {
                    "annotation_id": annotation_id,
                    "source": source,
                    "redaction_bucket": bucket,
                    "prediction_label": prediction_label,
                    "redaction_prediction_count": 0,
                    "redacted_chars": 0,
                },
            )
            group["redaction_prediction_count"] += 1
            group["redacted_chars"] += int(detail.get("non_pii_redacted_chars", 0))

        rows.extend(grouped.values())

    columns = [
        "annotation_id",
        "source",
        "redaction_bucket",
        "prediction_label",
        "redaction_prediction_count",
        "redacted_chars",
        "fraction_of_source_bucket",
        "fraction_of_source_redaction",
        "mean_redacted_chars_per_prediction",
    ]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(columns=columns)

    source_bucket_totals = frame.groupby(["source", "redaction_bucket"])["redacted_chars"].transform("sum")
    source_totals = frame.groupby("source")["redacted_chars"].transform("sum")
    frame["fraction_of_source_bucket"] = np.where(
        source_bucket_totals == 0,
        np.nan,
        frame["redacted_chars"] / source_bucket_totals,
    )
    frame["fraction_of_source_redaction"] = np.where(source_totals == 0, np.nan, frame["redacted_chars"] / source_totals)
    frame["mean_redacted_chars_per_prediction"] = np.where(
        frame["redaction_prediction_count"] == 0,
        np.nan,
        frame["redacted_chars"] / frame["redaction_prediction_count"],
    )

    frame = frame.sort_values(
        ["source", "redaction_bucket", "redacted_chars", "prediction_label"],
        ascending=[True, True, False, True],
        key=lambda values: _source_sorter(source_order)(values) if values.name == "source" else values,
    ).reset_index(drop=True)[columns]
    return _with_source_order(frame, source_order)


def gold_character_summary(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("gold_character_summary")
    if isinstance(summary, dict):
        return summary
    for entry in payload.get("results", []):
        result_summary = entry.get("result", {}).get("gold_character_summary")
        if isinstance(result_summary, dict):
            return result_summary
    result_summary = payload.get("result", {}).get("gold_character_summary")
    if isinstance(result_summary, dict):
        return result_summary
    return {
        "pii_chars": 0,
        "non_pii_chars": 0,
        "total_document_chars": 0,
        "pii_fraction": np.nan,
        "non_pii_fraction": np.nan,
        "components": {},
        "missing_document_lengths": [],
        "estimated_document_lengths": [],
    }


def build_postprocess_comparison_frame(
    raw_payload: dict[str, Any],
    postprocessed_payload: dict[str, Any],
) -> pd.DataFrame:
    raw_summary = build_summary_frame(raw_payload)
    postprocessed_summary = build_summary_frame(postprocessed_payload)
    raw = raw_summary.set_index("annotation_id")
    postprocessed = postprocessed_summary.set_index("annotation_id")

    annotation_ids: list[str] = []
    seen_annotation_ids: set[str] = set()
    for frame in (raw_summary, postprocessed_summary):
        for annotation_id in frame.get("annotation_id", []):
            annotation_id = str(annotation_id)
            if annotation_id not in seen_annotation_ids:
                annotation_ids.append(annotation_id)
                seen_annotation_ids.add(annotation_id)

    source_order: list[str] = []
    seen_sources: set[str] = set()
    for frame in (raw_summary, postprocessed_summary):
        for source in frame.attrs.get("source_order", []):
            if source not in seen_sources:
                source_order.append(source)
                seen_sources.add(source)

    rows = []
    for annotation_id in annotation_ids:
        raw_row = raw.loc[annotation_id] if annotation_id in raw.index else None
        post_row = postprocessed.loc[annotation_id] if annotation_id in postprocessed.index else None
        source = _optional_value(post_row, "source")
        if pd.isna(source):
            source = _optional_value(raw_row, "source")

        raw_recall = _optional_value(raw_row, "core_pii_recall")
        post_recall = _optional_value(post_row, "core_pii_recall")
        raw_exact_label_recall = _optional_value(raw_row, "exact_label_recall")
        post_exact_label_recall = _optional_value(post_row, "exact_label_recall")
        raw_coarse_label_recall = _optional_value(raw_row, "coarse_label_recall")
        post_coarse_label_recall = _optional_value(post_row, "coarse_label_recall")
        raw_fp = _optional_value(raw_row, "non_pii_redacted_chars")
        post_fp = _optional_value(post_row, "non_pii_redacted_chars")
        raw_fp_fraction = _optional_value(raw_row, "non_pii_redaction_rate")
        post_fp_fraction = _optional_value(post_row, "non_pii_redaction_rate")
        raw_spans = _optional_value(raw_row, "prediction_span_count")
        post_spans = _optional_value(post_row, "prediction_span_count")

        rows.append(
            {
                "annotation_id": annotation_id,
                "source": source,
                "raw_core_pii_recall": raw_recall,
                "postprocessed_core_pii_recall": post_recall,
                "core_pii_recall_delta": _delta(post_recall, raw_recall),
                "core_pii_recall_delta_pp": _scale(_delta(post_recall, raw_recall), 100),
                "raw_exact_label_recall": raw_exact_label_recall,
                "postprocessed_exact_label_recall": post_exact_label_recall,
                "exact_label_recall_delta": _delta(post_exact_label_recall, raw_exact_label_recall),
                "exact_label_recall_delta_pp": _scale(
                    _delta(post_exact_label_recall, raw_exact_label_recall),
                    100,
                ),
                "raw_coarse_label_recall": raw_coarse_label_recall,
                "postprocessed_coarse_label_recall": post_coarse_label_recall,
                "coarse_label_recall_delta": _delta(post_coarse_label_recall, raw_coarse_label_recall),
                "coarse_label_recall_delta_pp": _scale(
                    _delta(post_coarse_label_recall, raw_coarse_label_recall),
                    100,
                ),
                "raw_non_pii_redacted_chars": raw_fp,
                "postprocessed_non_pii_redacted_chars": post_fp,
                "non_pii_redacted_chars_delta": _delta(post_fp, raw_fp),
                "raw_non_pii_redaction_rate": raw_fp_fraction,
                "postprocessed_non_pii_redaction_rate": post_fp_fraction,
                "non_pii_redaction_rate_delta": _delta(post_fp_fraction, raw_fp_fraction),
                "non_pii_redaction_rate_delta_pp": _scale(_delta(post_fp_fraction, raw_fp_fraction), 100),
                "raw_prediction_span_count": raw_spans,
                "postprocessed_prediction_span_count": post_spans,
                "prediction_span_count_delta": _delta(post_spans, raw_spans),
            }
        )

    return _with_source_order(pd.DataFrame(rows), source_order)


def plot_postprocess_value(
    comparison_df: pd.DataFrame,
    output_path: Path | None = None,
):
    if comparison_df.empty:
        raise ValueError("No post-processing comparison rows available to plot")

    df = _sort_frame_by_source(comparison_df)
    y = np.arange(len(df))
    fig_height = max(4.5, 0.58 * len(df) + 1.5)
    fig, (ax_recall, ax_fp) = plt.subplots(
        ncols=2,
        figsize=(15, fig_height),
        gridspec_kw={"width_ratios": [1.1, 1]},
        constrained_layout=True,
    )

    recall_delta = df["core_pii_recall_delta_pp"].astype(float)
    recall_limit = _symmetric_delta_limit(recall_delta, minimum=0.1)
    recall_text_offset = recall_limit * 0.015
    recall_colors = np.where(recall_delta >= 0, "#26734d", "#b23a32")
    ax_recall.barh(y, recall_delta, height=0.6, color=recall_colors, edgecolor="#222222", linewidth=0.8)
    ax_recall.axvline(0, color="#222222", linewidth=1)
    ax_recall.set_xlim(-recall_limit, recall_limit)
    ax_recall.set_yticks(y, df["source"])
    ax_recall.invert_yaxis()
    ax_recall.set_xlabel("Core PII recall change, percentage points")
    ax_recall.set_title("Recall Lift", fontsize=16, weight="bold", pad=14)
    ax_recall.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax_recall.set_axisbelow(True)

    for idx, value in enumerate(recall_delta):
        if pd.isna(value):
            continue
        x_offset = recall_text_offset if value >= 0 else -recall_text_offset
        ax_recall.text(
            value + x_offset,
            idx,
            f"{value:+.2f} pp",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    fp_delta = df["non_pii_redaction_rate_delta"].astype(float)
    fp_limit = _symmetric_delta_limit(fp_delta, minimum=0.0001)
    fp_text_offset = fp_limit * 0.015
    fp_colors = np.where(fp_delta <= 0, "#26734d", "#b23a32")
    ax_fp.barh(y, fp_delta, height=0.6, color=fp_colors, edgecolor="#222222", linewidth=0.8)
    ax_fp.axvline(0, color="#222222", linewidth=1)
    ax_fp.set_xlim(-fp_limit, fp_limit)
    ax_fp.set_yticks(y, [])
    ax_fp.invert_yaxis()
    ax_fp.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=3))
    ax_fp.set_xlabel(f"{NON_PII_REDACTION_RATE_LABEL} change")
    ax_fp.set_title("Non-PII Redaction Rate Change", fontsize=16, weight="bold", pad=14)
    ax_fp.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax_fp.set_axisbelow(True)

    for idx, value in enumerate(fp_delta):
        if pd.isna(value):
            continue
        x_offset = fp_text_offset
        ax_fp.text(
            value + (x_offset if value >= 0 else -x_offset),
            idx,
            f"{value * 100:+.3f}%",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9,
        )

    for axis in (ax_recall, ax_fp):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_recall_and_non_pii_redaction(summary_df: pd.DataFrame, output_path: Path | None = None):
    if summary_df.empty:
        raise ValueError("No evaluation results available to plot")

    df = summary_df.copy()
    y = np.arange(len(df))
    row_height = 0.6
    fig_height = max(4.5, 0.58 * len(df) + 1.4)
    fig, (ax_recall, ax_fp) = plt.subplots(
        ncols=2,
        figsize=(15, fig_height),
        gridspec_kw={"width_ratios": [1.55, 1]},
        constrained_layout=True,
    )

    palette = plt.get_cmap("tab10")
    bar_colors = [palette(index % 10) for index in range(len(df))]
    ax_recall.barh(
        y,
        df["core_pii_recall"],
        height=row_height,
        color=bar_colors,
        edgecolor="#222222",
        linewidth=0.8,
    )
    ax_recall.set_xlim(0, 1.0)
    ax_recall.set_yticks(y, df["source"])
    ax_recall.invert_yaxis()
    ax_recall.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax_recall.set_xlabel("Core PII recall")
    ax_recall.set_title("Core PII Recall", fontsize=18, weight="bold", pad=16)
    ax_recall.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax_recall.set_axisbelow(True)

    for idx, value in enumerate(df["core_pii_recall"]):
        if pd.isna(value):
            continue
        x = min(float(value) - 0.015, 0.985)
        ha = "right"
        color = "white"
        if value < 0.12:
            x = float(value) + 0.015
            ha = "left"
            color = "#222222"
        ax_recall.text(x, idx, f"{value * 100:.1f}%", va="center", ha=ha, color=color, fontsize=9)

    ax_fp.barh(
        y,
        df["machine_only_redaction_rate"],
        height=row_height,
        color="#879393",
        edgecolor="#222222",
        linewidth=0.8,
        label=FALSE_POSITIVE_SPANS_LABEL,
    )
    ax_fp.barh(
        y,
        df["boundary_overflow_redaction_rate"],
        height=row_height,
        left=df["machine_only_redaction_rate"],
        color="#c7392f",
        edgecolor="#222222",
        linewidth=0.8,
        label=PII_BOUNDARY_EXTENSIONS_LABEL,
    )
    ax_fp.set_yticks(y, [])
    ax_fp.invert_yaxis()
    ax_fp.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=3))
    ax_fp.set_xlabel(NON_PII_REDACTION_RATE_LABEL)
    ax_fp.set_title("Non-PII Redaction Rate", fontsize=18, weight="bold", pad=16)
    ax_fp.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax_fp.set_axisbelow(True)
    ax_fp.legend(frameon=False, loc="upper right")

    fp_fraction = df["non_pii_redaction_rate"]
    fp_fraction_max = fp_fraction.max(skipna=True)
    max_total = max(float(0 if pd.isna(fp_fraction_max) else fp_fraction_max), 0.0001)
    ax_fp.set_xlim(0, max_total * 1.18)
    for idx, total in enumerate(fp_fraction):
        if pd.isna(total):
            continue
        ax_fp.text(total + max_total * 0.01, idx, f"{total * 100:.3f}%", va="center", ha="left", fontsize=9)

    for axis in (ax_recall, ax_fp):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_label_quality_metrics(
    label_quality_df: pd.DataFrame,
    output_path: Path | None = None,
):
    if label_quality_df.empty:
        raise ValueError("No label-quality metrics available to plot")

    df = _sort_frame_by_source(label_quality_df)
    y = np.arange(len(df))
    row_height = 0.24
    fig_height = max(4.8, 0.86 * len(df) + 1.7)
    fig, (ax_recall, ax_accuracy) = plt.subplots(
        ncols=2,
        figsize=(16, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1]},
        constrained_layout=True,
    )

    detected_recall = df["span_detection_recall"].astype(float)
    exact_recall = df["exact_label_recall"].astype(float)
    coarse_recall = df["coarse_label_recall"].astype(float)
    exact_accuracy = df["exact_label_accuracy_matched"].astype(float)
    coarse_accuracy = df["coarse_label_accuracy_matched"].astype(float)

    ax_recall.barh(
        y - row_height,
        detected_recall,
        height=row_height,
        color="#b8bec4",
        edgecolor="#222222",
        linewidth=0.7,
        label="Detected as entity",
    )
    ax_recall.barh(
        y,
        exact_recall,
        height=row_height,
        color="#496d89",
        edgecolor="#222222",
        linewidth=0.7,
        label="Exact",
    )
    ax_recall.barh(
        y + row_height,
        coarse_recall,
        height=row_height,
        color="#72a276",
        edgecolor="#222222",
        linewidth=0.7,
        label="Coarse",
    )
    ax_recall.set_title("Label-Correct Recall", fontsize=16, weight="bold", pad=14)
    ax_recall.set_xlabel("Matched with correct label / all core PII spans")
    ax_recall.set_yticks(y, df["source"])

    ax_accuracy.barh(
        y - row_height / 2,
        exact_accuracy,
        height=row_height,
        color="#496d89",
        edgecolor="#222222",
        linewidth=0.7,
        label="Exact",
    )
    ax_accuracy.barh(
        y + row_height / 2,
        coarse_accuracy,
        height=row_height,
        color="#72a276",
        edgecolor="#222222",
        linewidth=0.7,
        label="Coarse",
    )
    ax_accuracy.set_title("Matched-Entity Label Accuracy", fontsize=16, weight="bold", pad=14)
    ax_accuracy.set_xlabel("Correct label / matched core PII spans")
    ax_accuracy.set_yticks(y, [])

    for axis in (ax_recall, ax_accuracy):
        axis.set_xlim(0, 1)
        axis.invert_yaxis()
        axis.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
        axis.grid(axis="x", color="#d0d0d0", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.legend(frameon=False, loc="lower right")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    _annotate_grouped_percent_bars(ax_recall, y - row_height, detected_recall)
    _annotate_grouped_percent_bars(ax_recall, y, exact_recall)
    _annotate_grouped_percent_bars(ax_recall, y + row_height, coarse_recall)
    _annotate_grouped_percent_bars(ax_accuracy, y - row_height / 2, exact_accuracy)
    _annotate_grouped_percent_bars(ax_accuracy, y + row_height / 2, coarse_accuracy)
    _annotate_detected_pool(ax_accuracy, y, df)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def build_recall_matrix(
    payload: dict[str, Any],
    group_name: str,
    row_key: str,
    metric_key: str | None,
    count_key: str,
    row_filter: Callable[[dict[str, Any]], bool] | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    display_names = source_display_names(payload)
    source_names = _ordered_source_names(payload, display_names)

    values: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for entry in payload["results"]:
        source = display_names[str(entry["display_annotation_id"])]
        for group in entry["result"].get(group_name, []):
            if row_filter is not None and not row_filter(group):
                continue
            row_name = str(group[row_key])
            metric = group if metric_key is None else group[metric_key]
            recall = metric.get("recall")
            values.setdefault(row_name, {})[source] = np.nan if recall is None else recall
            counts[row_name] = int(group.get(count_key, metric.get(count_key, 0)))

    matrix = pd.DataFrame.from_dict(values, orient="index")
    matrix = matrix.reindex(columns=source_names)
    if not matrix.empty:
        matrix = matrix.loc[matrix.mean(axis=1, skipna=True).sort_values().index]
    matrix.attrs["source_order"] = source_names
    return matrix, counts


def build_label_match_matrix(
    payload: dict[str, Any],
    *,
    match_level: str = "exact",
    denominator: str = "total",
) -> tuple[pd.DataFrame, dict[str, int]]:
    display_names = source_display_names(payload)
    source_names = _ordered_source_names(payload, display_names)

    values: dict[str, dict[str, float]] = {}
    counts: dict[str, int] = {}
    for entry in payload["results"]:
        source = display_names[str(entry["display_annotation_id"])]
        confusion = entry["result"].get("core_pii_span_label_confusion", {})
        for row in confusion.get("by_annotation_label", []):
            gold_label = str(row.get("annotation_label") or "")
            total_spans = int(row.get("total_core_pii_spans") or 0)
            matched_spans = int(row.get("matched_core_pii_spans") or 0)
            denominator_spans = matched_spans if denominator == "matched" else total_spans
            correct_spans = 0
            for assigned in row.get("assigned_labels") or []:
                prediction_label = str(assigned.get("prediction_label") or "")
                spans = int(assigned.get("spans") or 0)
                if _prediction_label_matches_gold(prediction_label, gold_label, match_level):
                    correct_spans += spans
            values.setdefault(gold_label, {})[source] = (
                np.nan if denominator_spans == 0 else correct_spans / denominator_spans
            )
            counts[gold_label] = max(counts.get(gold_label, 0), total_spans)

    matrix = pd.DataFrame.from_dict(values, orient="index")
    matrix = matrix.reindex(columns=source_names)
    if not matrix.empty:
        matrix = matrix.loc[matrix.mean(axis=1, skipna=True).sort_values().index]
    matrix.attrs["source_order"] = source_names
    return matrix, counts


def build_label_aware_recall_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    display_names = source_display_names(payload)
    source_order = source_order_from_payload(payload, display_names)

    for entry in payload["results"]:
        annotation_id = str(entry["display_annotation_id"])
        source = display_names[annotation_id]
        confusion = entry["result"].get("core_pii_span_label_confusion", {})
        metrics = confusion.get("metrics", {})
        total_spans = int(confusion.get("total_core_pii_spans") or 0)
        matched_spans = int(confusion.get("matched_core_pii_spans") or 0)
        exact_spans = int(metrics.get("exact_correct_core_pii_spans") or 0)
        coarse_spans = int(metrics.get("coarse_correct_core_pii_spans") or 0)
        category_only_spans = max(coarse_spans - exact_spans, 0)
        incorrect_category_spans = max(matched_spans - coarse_spans, 0)
        missed_spans = max(total_spans - matched_spans, 0)

        rows.append(
            {
                "annotation_id": annotation_id,
                "source": source,
                "total_core_pii_spans": total_spans,
                "matched_core_pii_spans": matched_spans,
                "exact_category_subtype_spans": exact_spans,
                "correct_category_only_spans": category_only_spans,
                "incorrect_category_spans": incorrect_category_spans,
                "missed_spans": missed_spans,
                "entity_detection_recall": _fraction_float(matched_spans, total_spans),
                "exact_category_subtype_recall": _fraction_float(exact_spans, total_spans),
                "correct_category_only_recall": _fraction_float(category_only_spans, total_spans),
                "incorrect_category_recall": _fraction_float(incorrect_category_spans, total_spans),
                "missed_fraction": _fraction_float(missed_spans, total_spans),
                "category_accuracy_matched": _fraction_float(coarse_spans, matched_spans),
                "exact_accuracy_matched": _fraction_float(exact_spans, matched_spans),
            }
        )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values("source", key=_source_sorter(source_order)).reset_index(drop=True)
    return _with_source_order(frame, source_order)


def build_label_aware_gold_label_frame(payload: dict[str, Any]) -> pd.DataFrame:
    rows = []
    display_names = source_display_names(payload)
    source_order = source_order_from_payload(payload, display_names)

    for entry in payload["results"]:
        annotation_id = str(entry["display_annotation_id"])
        source = display_names[annotation_id]
        confusion = entry["result"].get("core_pii_span_label_confusion", {})
        for group in confusion.get("by_annotation_label", []):
            gold_label = str(group.get("annotation_label") or "")
            total_spans = int(group.get("total_core_pii_spans") or 0)
            matched_spans = int(group.get("matched_core_pii_spans") or 0)
            exact_spans = 0
            coarse_spans = 0
            for assigned in group.get("assigned_labels") or []:
                prediction_label = str(assigned.get("prediction_label") or "")
                spans = int(assigned.get("spans") or 0)
                if _prediction_label_matches_gold(prediction_label, gold_label, "exact"):
                    exact_spans += spans
                if _prediction_label_matches_gold(prediction_label, gold_label, "coarse"):
                    coarse_spans += spans

            rows.append(
                {
                    "annotation_id": annotation_id,
                    "source": source,
                    "annotation_label": gold_label,
                    "total_core_pii_spans": total_spans,
                    "matched_core_pii_spans": matched_spans,
                    "exact_category_subtype_spans": exact_spans,
                    "correct_category_spans": coarse_spans,
                    "incorrect_category_spans": max(matched_spans - coarse_spans, 0),
                    "missed_spans": max(total_spans - matched_spans, 0),
                    "entity_detection_recall": _fraction_float(matched_spans, total_spans),
                    "exact_category_subtype_recall": _fraction_float(exact_spans, total_spans),
                    "correct_category_recall": _fraction_float(coarse_spans, total_spans),
                    "category_accuracy_matched": _fraction_float(coarse_spans, matched_spans),
                    "exact_accuracy_matched": _fraction_float(exact_spans, matched_spans),
                }
            )

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame = frame.sort_values(
        ["annotation_label", "source"],
        key=lambda values: _source_sorter(source_order)(values) if values.name == "source" else values,
    ).reset_index(drop=True)
    return _with_source_order(frame, source_order)


def _is_plottable_subannotation_category(group: dict[str, Any]) -> bool:
    if group.get("ignored_in_headline") is True:
        return False
    if group.get("is_aggregate") is True or group.get("aggregate") is True:
        return False

    category = str(group.get("category", "")).strip().casefold().replace("-", "_").replace(" ", "_")
    aggregate_category_names = {
        "all_non_ignored",
        "non_ignored",
        "all_but_ignored",
        "all_except_ignored",
        "all_except_ignored_categories",
        "everything_but_ignored",
        "everything_but_ignored_categories",
        "everything_except_ignored",
        "everything_except_ignored_categories",
        "core_pii_recall",
    }
    return category not in aggregate_category_names


def plot_recall_heatmap(
    matrix: pd.DataFrame,
    counts: dict[str, int],
    title: str,
    count_unit: str,
    output_path: Path | None = None,
    colorbar_label: str = "Recall",
):
    if matrix.empty:
        raise ValueError(f"No data available for {title}")

    fig_width = max(9, 1.25 * len(matrix.columns) + 4)
    fig_height = max(5, 0.42 * len(matrix.index) + 1.5)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap=cmap)

    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    y_labels = [f"{row} ({counts.get(row, 0):,} {count_unit})" for row in matrix.index]
    ax.set_yticks(np.arange(len(matrix.index)), y_labels)

    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, row_name in enumerate(matrix.index):
        for col_index, source in enumerate(matrix.columns):
            value = matrix.loc[row_name, source]
            if pd.isna(value):
                continue
            text_color = "white" if _relative_luminance(cmap(float(value))) < 0.179 else "#111111"
            ax.text(col_index, row_index, f"{value:.2f}", ha="center", va="center", color=text_color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label(colorbar_label)
    colorbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    for spine in ax.spines.values():
        spine.set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def build_span_label_confusion_matrices(
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Build row-normalized one-to-one span label-confusion matrices."""
    display_names = source_display_names(payload)
    matrices: dict[str, dict[str, Any]] = {}
    for entry in payload["results"]:
        annotation_id = str(entry["display_annotation_id"])
        source = display_names[annotation_id]
        confusion = entry["result"].get("core_pii_span_label_confusion", {})
        rows = confusion.get("by_annotation_label", [])
        if not rows:
            continue

        raw_values: dict[str, dict[str, int]] = {}
        totals: dict[str, int] = {}
        matched: dict[str, int] = {}
        prediction_totals: Counter[str] = Counter()
        for row in rows:
            gold_label = str(row.get("annotation_label") or "")
            totals[gold_label] = int(row.get("total_core_pii_spans") or 0)
            matched[gold_label] = int(row.get("matched_core_pii_spans") or 0)
            raw_values.setdefault(gold_label, {})
            for assigned in row.get("assigned_labels") or []:
                prediction_label = _display_prediction_label(assigned.get("prediction_label"))
                value = int(assigned.get("spans") or 0)
                if value <= 0:
                    continue
                raw_values[gold_label][prediction_label] = (
                    raw_values[gold_label].get(prediction_label, 0) + value
                )
                prediction_totals[prediction_label] += value

        row_order = sorted(raw_values)
        prediction_labels = [label for label in row_order if label in prediction_totals]
        prediction_labels.extend(
            sorted(
                set(prediction_totals).difference(row_order),
                key=lambda label: (-prediction_totals[label], label),
            )
        )
        if not prediction_labels:
            continue

        count_matrix = pd.DataFrame(0, index=row_order, columns=prediction_labels, dtype=int)
        fraction_matrix = pd.DataFrame(np.nan, index=row_order, columns=prediction_labels, dtype=float)
        for gold_label in row_order:
            for prediction_label, value in raw_values[gold_label].items():
                count_matrix.loc[gold_label, prediction_label] = value
                fraction_matrix.loc[gold_label, prediction_label] = value / matched[gold_label]

        matrices[annotation_id] = {
            "annotation_id": annotation_id,
            "source": source,
            "fraction_matrix": fraction_matrix,
            "count_matrix": count_matrix,
            "totals": totals,
            "matched": matched,
            "metrics": confusion.get("metrics", {}),
        }

    return matrices


def build_span_label_confusion_long_frame(payload: dict[str, Any]) -> pd.DataFrame:
    matrices = build_span_label_confusion_matrices(payload)
    rows: list[dict[str, Any]] = []
    for matrix_payload in matrices.values():
        counts = matrix_payload["count_matrix"]
        for gold_label in counts.index:
            total = int(matrix_payload["totals"][gold_label])
            matched = int(matrix_payload["matched"][gold_label])
            for prediction_label in counts.columns:
                value = int(counts.loc[gold_label, prediction_label])
                if value <= 0:
                    continue
                rows.append(
                    {
                        "annotation_id": matrix_payload["annotation_id"],
                        "source": matrix_payload["source"],
                        "gold_label": gold_label,
                        "prediction_label": prediction_label,
                        "spans": value,
                        "total_core_pii_spans": total,
                        "matched_core_pii_spans": matched,
                        "missed_core_pii_spans": total - matched,
                        "detection_recall": _fraction_float(matched, total),
                        "fraction_of_matched_row_spans": _fraction_float(value, matched),
                        "fraction_of_total_row_spans": _fraction_float(value, total),
                    }
                )
    return pd.DataFrame(rows)


def build_span_label_confusion_summary_frame(payload: dict[str, Any]) -> pd.DataFrame:
    matrices = build_span_label_confusion_matrices(payload)
    rows = []
    for matrix_payload in matrices.values():
        metrics = matrix_payload["metrics"]
        totals = matrix_payload["totals"]
        matched = matrix_payload["matched"]
        total = sum(totals.values())
        matched_total = sum(matched.values())
        rows.append(
            {
                "annotation_id": matrix_payload["annotation_id"],
                "source": matrix_payload["source"],
                "total_core_pii_spans": total,
                "matched_core_pii_spans": matched_total,
                "missed_core_pii_spans": total - matched_total,
                "detection_recall": _fraction_float(matched_total, total),
                "exact_label_accuracy_among_matches": metrics.get("exact_label_accuracy_matched"),
                "coarse_label_accuracy_among_matches": metrics.get("coarse_label_accuracy_matched"),
                "exact_label_recall": metrics.get("exact_label_recall"),
                "coarse_label_recall": metrics.get("coarse_label_recall"),
            }
        )
    return pd.DataFrame(rows)


def plot_span_label_confusion_heatmap(
    matrix_payload: dict[str, Any],
    output_path: Path | None = None,
):
    fraction_matrix = matrix_payload["fraction_matrix"]
    count_matrix = matrix_payload["count_matrix"]
    if fraction_matrix.empty:
        raise ValueError("No matched label-confusion data available")

    fig_width = max(9, 0.95 * len(fraction_matrix.columns) + 4)
    fig_height = max(5, 0.5 * len(fraction_matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    values = fraction_matrix.to_numpy(dtype=float)
    cell_colors = np.empty((*values.shape, 4), dtype=float)
    cell_colors[:] = mcolors.to_rgba("#f2f2f2")
    for row_index, gold_label in enumerate(fraction_matrix.index):
        for column_index, prediction_label in enumerate(fraction_matrix.columns):
            value = values[row_index, column_index]
            if np.isnan(value):
                continue
            cell_colors[row_index, column_index] = CONFUSION_CMAP(float(value))
    ax.imshow(cell_colors, aspect="auto")

    ax.set_title(
        f"Span-level Label Confusion Among Matched Detections\n{matrix_payload['source']}",
        fontsize=14,
        weight="bold",
        pad=12,
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Gold label")
    ax.set_xticks(np.arange(len(fraction_matrix.columns)), fraction_matrix.columns, rotation=35, ha="right")
    y_labels = [
        f"{label} ({matrix_payload['matched'][label]:,}/{matrix_payload['totals'][label]:,} spans)"
        for label in fraction_matrix.index
    ]
    ax.set_yticks(np.arange(len(fraction_matrix.index)), y_labels)
    ax.set_xticks(np.arange(-0.5, len(fraction_matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(fraction_matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    # The outlined cells form the preferred exact-label path. This remains
    # unambiguous without adding colour-scale legends to every matrix.
    prediction_positions = {
        label: position for position, label in enumerate(fraction_matrix.columns)
    }
    for row_index, gold_label in enumerate(fraction_matrix.index):
        column_index = prediction_positions.get(gold_label)
        if column_index is None:
            continue
        ax.add_patch(
            Rectangle(
                (column_index - 0.5, row_index - 0.5),
                1.0,
                1.0,
                fill=False,
                edgecolor="#4c5963",
                linewidth=1.35,
                zorder=5,
            )
        )

    for row_index, row_label in enumerate(fraction_matrix.index):
        for column_index, prediction_label in enumerate(fraction_matrix.columns):
            value = fraction_matrix.loc[row_label, prediction_label]
            count = int(count_matrix.loc[row_label, prediction_label])
            if pd.isna(value) or count == 0:
                continue
            text_color = (
                "white"
                if _relative_luminance(CONFUSION_CMAP(float(value))) < 0.179
                else "#111111"
            )
            ax.text(
                column_index,
                row_index,
                f"{float(value) * 100:.1f}%\n{count:,}",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)
    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.2)
    return fig


def save_label_confusion_analysis(
    payload: dict[str, Any], raw_dir: Path | str, plots_dir: Path | str | None = None
) -> dict[str, Any]:
    """Persist span-only confusion tables and plots in separate directory trees."""
    raw_root = Path(raw_dir)
    split_outputs = plots_dir is not None
    plot_root = Path(plots_dir) if split_outputs else raw_root
    raw_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    for stale_dir in {
        raw_root / "span",
        raw_root / "character",
        plot_root / "span",
        plot_root / "character",
    }:
        if stale_dir.is_dir():
            shutil.rmtree(stale_dir)
    artifacts: dict[str, Any] = {}
    definitions = {
        "purpose": "Label assignment among matched core-PII detections; separate from detection recall.",
        "span_matching": (
            "Positive overlap with non-ignored/core-PII gold characters; deterministic one-to-one "
            "greedy assignment by overlap characters, gold coverage, prediction coverage, then stable input order."
        ),
        "confusion_denominator": "Matched spans within each gold-label row.",
        "detection_denominator": "All core-PII gold spans within each gold-label row.",
        "label_normalization": (
            "BIO prefixes are stripped; known Deduce, OpenAI Privacy Filter, and OpenMed labels are "
            "mapped to workflow labels; unknown labels are retained verbatim."
        ),
        "plot_encoding": (
            "All cells use one neutral light-to-dark blue 0–100% magnitude scale; "
            "a bold outline marks every preferred exact-label diagonal cell."
        ),
    }
    (raw_root / "definitions.json").write_text(
        json.dumps(definitions, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (raw_root / "README.md").write_text(
        "# Label confusion analysis\n\n"
        "Confusion matrices use one-to-one span matches and are row-normalized among matched detections; "
        "detection misses are reported in the raw tables and summaries, not as a predicted label.\n\n"
        "- Raw tables: `summary.csv` and `confusion_long.csv` in this directory.\n"
        "- Rendered matrices: the matching `analysis/plots/label_confusion/` directory.\n"
        "- `definitions.json`\n",
        encoding="utf-8",
    )

    matrix_plots_dir = plot_root if split_outputs else raw_root / "plots"
    matrix_plots_dir.mkdir(parents=True, exist_ok=True)
    matrices = build_span_label_confusion_matrices(payload)
    long_frame = build_span_label_confusion_long_frame(payload)
    summary_frame = build_span_label_confusion_summary_frame(payload)
    long_frame.to_csv(raw_root / "confusion_long.csv", index=False)
    summary_frame.to_csv(raw_root / "summary.csv", index=False)
    figures = {}
    for annotation_id, matrix_payload in matrices.items():
        output_path = matrix_plots_dir / f"{_safe_filename(annotation_id)}.png"
        figure = plot_span_label_confusion_heatmap(matrix_payload, output_path)
        figures[annotation_id] = figure
        plt.close(figure)
    artifacts = {
        "matrices": matrices,
        "long_frame": long_frame,
        "summary_frame": summary_frame,
        "figures": figures,
    }
    return artifacts


def plot_label_aware_recall_stack(
    label_aware_df: pd.DataFrame,
    output_path: Path | None = None,
):
    if label_aware_df.empty:
        raise ValueError("No label-aware recall rows available to plot")

    df = _sort_frame_by_source(label_aware_df)
    y = np.arange(len(df))
    fig_height = max(4.5, 0.58 * len(df) + 1.5)
    fig, ax = plt.subplots(figsize=(13, fig_height), constrained_layout=True)

    segments = [
        ("exact_category_subtype_recall", "Correct category+subtype", "#2f6f9f"),
        ("correct_category_only_recall", "Correct category only", "#72a276"),
        ("incorrect_category_recall", "Wrong category", "#c76d4f"),
    ]
    left = np.zeros(len(df), dtype=float)
    for column, label, color in segments:
        values = df[column].astype(float).fillna(0).to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            height=0.58,
            color=color,
            edgecolor="#222222",
            linewidth=0.7,
            label=label,
        )
        left += values

    label_gutter = 0.08
    ax.set_xlim(0, 1 + label_gutter)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(y, df["source"])
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_xlabel("Share of all core PII spans")
    ax.set_title("Label-Aware Core PII Recall", fontsize=18, weight="bold", pad=16)
    ax.grid(axis="x", color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        ncol=len(segments),
        borderaxespad=0,
    )

    for idx, row in df.iterrows():
        detected = row.get("entity_detection_recall")
        if not pd.isna(detected):
            ax.text(
                min(float(detected) + 0.012, 1 + label_gutter - 0.005),
                idx,
                f"detected {float(detected) * 100:.1f}%",
                va="center",
                ha="left",
                fontsize=8,
                color="#222222",
                clip_on=False,
            )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def plot_label_aware_detection_heatmap(
    gold_label_df: pd.DataFrame,
    output_path: Path | None = None,
):
    if gold_label_df.empty:
        raise ValueError("No gold-label label-aware rows available to plot")

    source_names = _ordered_unique_sources(gold_label_df)
    label_counts = (
        gold_label_df.groupby("annotation_label")["total_core_pii_spans"].max().astype(int).to_dict()
    )
    gold_labels = sorted(
        label_counts,
        key=lambda label: (
            gold_label_df.loc[gold_label_df["annotation_label"] == label, "entity_detection_recall"].mean(),
            label,
        ),
    )

    fig_width = max(10, 1.35 * len(source_names) + 4.2)
    fig_height = max(5.2, 0.5 * len(gold_labels) + 1.7)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = HEATMAP_CMAP
    norm = mcolors.Normalize(vmin=0, vmax=1)
    row_lookup = {
        (row["annotation_label"], row["source"]): row
        for row in gold_label_df.to_dict(orient="records")
    }

    ax.set_xlim(-0.5, len(source_names) - 0.5)
    ax.set_ylim(len(gold_labels) - 0.5, -0.5)
    ax.set_title("Entity Detection Coverage + Category Label Accuracy", fontsize=15, weight="bold", pad=14)
    ax.set_xlabel("Annotation source")
    ax.set_ylabel("Gold label")
    ax.set_xticks(np.arange(len(source_names)), source_names, rotation=35, ha="right")
    y_labels = [f"{label} ({label_counts.get(label, 0):,} chars)" for label in gold_labels]
    ax.set_yticks(np.arange(len(gold_labels)), y_labels)
    ax.set_xticks(np.arange(-0.5, len(source_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(gold_labels), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, gold_label in enumerate(gold_labels):
        for col_index, source in enumerate(source_names):
            row = row_lookup.get((gold_label, source))
            if row is None:
                continue
            detection_recall = row.get("entity_detection_recall")
            category_accuracy = row.get("category_accuracy_matched")
            if pd.isna(detection_recall):
                continue

            cell_left = col_index - 0.42
            cell_width = 0.84
            bar_width = cell_width * max(0.0, min(float(detection_recall), 1.0))
            color_value = 0.0 if pd.isna(category_accuracy) else max(0.0, min(float(category_accuracy), 1.0))
            ax.add_patch(
                plt.Rectangle(
                    (cell_left, row_index - 0.28),
                    cell_width,
                    0.56,
                    facecolor="white",
                    edgecolor="#222222",
                    linewidth=0.6,
                )
            )
            if bar_width > 0:
                bar_color = cmap(norm(color_value))
                ax.add_patch(
                    plt.Rectangle(
                        (cell_left, row_index - 0.28),
                        bar_width,
                        0.56,
                        facecolor=bar_color,
                        edgecolor="none",
                    )
                )
            label = f"{float(detection_recall) * 100:.0f}%"
            if not pd.isna(category_accuracy):
                label = f"{float(detection_recall) * 100:.0f}%\n{float(category_accuracy) * 100:.0f}% label"
            text_inside_bar = bar_width >= cell_width * 0.34
            if text_inside_bar:
                text_x = cell_left + bar_width / 2
                text_color = "white" if _relative_luminance(bar_color) < 0.42 else "#111111"
            else:
                text_x = min(cell_left + bar_width + 0.04, cell_left + cell_width - 0.03)
                text_color = "#111111"
            ax.text(
                text_x,
                row_index,
                label,
                ha="center" if text_inside_bar else "left",
                va="center",
                fontsize=7,
                color=text_color,
            )

    colorbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        fraction=0.035,
        pad=0.02,
    )
    colorbar.set_label("Label accuracy among matched spans")
    colorbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    for spine in ax.spines.values():
        spine.set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def build_non_pii_redaction_label_matrix(
    payload: dict[str, Any],
    redaction_bucket: str | None = None,
) -> tuple[pd.DataFrame, dict[str, int]]:
    label_df = build_non_pii_redaction_label_frame(payload, redaction_bucket=redaction_bucket)
    display_names = source_display_names(payload)
    source_names = _ordered_source_names(payload, display_names)

    if label_df.empty:
        matrix = pd.DataFrame(columns=source_names)
        matrix.attrs["source_order"] = source_names
        return matrix, {}

    grouped = (
        label_df.groupby(["prediction_label", "source"], as_index=False)
        .agg(redacted_chars=("redacted_chars", "sum"), redaction_prediction_count=("redaction_prediction_count", "sum"))
    )
    source_totals = grouped.groupby("source")["redacted_chars"].transform("sum")
    grouped["fraction_of_source_redaction"] = np.where(source_totals == 0, np.nan, grouped["redacted_chars"] / source_totals)

    matrix = grouped.pivot(index="prediction_label", columns="source", values="fraction_of_source_redaction")
    matrix = matrix.reindex(columns=source_names)
    counts = grouped.groupby("prediction_label")["redacted_chars"].sum().astype(int).to_dict()
    if not matrix.empty:
        matrix = matrix.loc[sorted(matrix.index, key=lambda label: (-counts.get(label, 0), str(label)))]
    matrix.attrs["source_order"] = source_names
    return matrix, counts


def plot_non_pii_redaction_label_distribution(
    matrix: pd.DataFrame,
    counts: dict[str, int],
    title: str,
    output_path: Path | None = None,
):
    if matrix.empty:
        raise ValueError(f"No data available for {title}")

    fig_width = max(9, 1.25 * len(matrix.columns) + 4)
    fig_height = max(4.5, 0.48 * len(matrix.index) + 1.6)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)

    cmap = HEATMAP_CMAP.copy()
    cmap.set_bad(color="#f2f2f2")
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", vmin=0, vmax=1, cmap=cmap)

    ax.set_title(title, fontsize=15, weight="bold", pad=14)
    ax.set_xticks(np.arange(len(matrix.columns)), matrix.columns, rotation=35, ha="right")
    y_labels = [f"{row} ({counts.get(row, 0):,} chars)" for row in matrix.index]
    ax.set_yticks(np.arange(len(matrix.index)), y_labels)

    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="#d8d8d8", linewidth=0.7)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row_index, row_name in enumerate(matrix.index):
        for col_index, source in enumerate(matrix.columns):
            value = matrix.loc[row_name, source]
            if pd.isna(value):
                continue
            text_color = "white" if _relative_luminance(cmap(float(value))) < 0.179 else "#111111"
            ax.text(col_index, row_index, f"{value * 100:.1f}%", ha="center", va="center", color=text_color, fontsize=8)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("Share of source non-PII redacted characters")
    colorbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

    for spine in ax.spines.values():
        spine.set_visible(False)

    if output_path is not None:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return fig


def save_quantity_evaluation_plots(payload: dict[str, Any], figures_dir: Path | str) -> dict[str, Any]:
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)
    plt.close("all")

    summary_df = build_summary_frame(payload)
    annotation_label_matrix, annotation_label_counts = build_recall_matrix(
        payload=payload,
        group_name="by_annotation_label",
        row_key="annotation_label",
        metric_key="core_pii_recall",
        count_key="gold_span_count",
    )
    subannotation_matrix, subannotation_counts = build_recall_matrix(
        payload=payload,
        group_name="by_subannotation_category",
        row_key="category",
        metric_key=None,
        count_key="total_chars",
        row_filter=_is_plottable_subannotation_category,
    )
    fp_label_df = build_non_pii_redaction_label_frame(payload)
    fp_label_matrix, fp_label_counts = build_non_pii_redaction_label_matrix(payload)
    label_quality_df = build_label_quality_frame(payload)
    label_aware_recall_df = build_label_aware_recall_frame(payload)
    label_aware_gold_label_df = build_label_aware_gold_label_frame(payload)
    exact_label_match_matrix, label_match_counts = build_label_match_matrix(
        payload,
        match_level="exact",
        denominator="total",
    )
    coarse_label_match_matrix, _ = build_label_match_matrix(
        payload,
        match_level="coarse",
        denominator="total",
    )
    artifacts = {
        "summary_df": summary_df,
        "annotation_label_matrix": annotation_label_matrix,
        "annotation_label_counts": annotation_label_counts,
        "subannotation_matrix": subannotation_matrix,
        "subannotation_counts": subannotation_counts,
        "non_pii_redaction_label_df": fp_label_df,
        "non_pii_redaction_label_matrix": fp_label_matrix,
        "non_pii_redaction_label_counts": fp_label_counts,
        "label_quality_df": label_quality_df,
        "label_aware_recall_df": label_aware_recall_df,
        "label_aware_gold_label_df": label_aware_gold_label_df,
        "exact_label_match_matrix": exact_label_match_matrix,
        "coarse_label_match_matrix": coarse_label_match_matrix,
        "label_match_counts": label_match_counts,
        "core_pii_recall_non_pii_redaction_figure": plot_recall_and_non_pii_redaction(
            summary_df,
            figures_path / "core_pii_recall_non_pii_redaction.png",
        ),
        "annotation_label_figure": plot_recall_heatmap(
            annotation_label_matrix,
            annotation_label_counts,
            title="Core PII Recall By Gold Label",
            count_unit="spans",
            output_path=figures_path / "recall_by_gold_label.png",
            colorbar_label="Core PII recall",
        ),
        "subannotation_figure": plot_recall_heatmap(
            subannotation_matrix,
            subannotation_counts,
            title="Core PII Recall By Subannotation Category",
            count_unit="chars",
            output_path=figures_path / "recall_by_subannotation_category.png",
            colorbar_label="Core PII recall",
        ),
    }
    if not label_quality_df.empty and label_quality_df["exact_label_recall"].notna().any():
        artifacts["label_quality_figure"] = plot_label_quality_metrics(
            label_quality_df,
            figures_path / "label_quality_metrics.png",
        )
    else:
        artifacts["label_quality_figure"] = None
    if not label_aware_recall_df.empty:
        artifacts["label_aware_recall_figure"] = plot_label_aware_recall_stack(
            label_aware_recall_df,
            figures_path / "label_aware_core_pii_recall.png",
        )
    else:
        artifacts["label_aware_recall_figure"] = None
    if not label_aware_gold_label_df.empty:
        artifacts["label_aware_detection_heatmap_figure"] = plot_label_aware_detection_heatmap(
            label_aware_gold_label_df,
            figures_path / "label_aware_detection_heatmap.png",
        )
    else:
        artifacts["label_aware_detection_heatmap_figure"] = None
    if not exact_label_match_matrix.empty:
        artifacts["exact_label_match_figure"] = plot_recall_heatmap(
            exact_label_match_matrix,
            label_match_counts,
            title="Exact Label-Correct Recall By Gold Label",
            count_unit="spans",
            output_path=figures_path / "exact_label_recall_by_gold_label.png",
            colorbar_label="Exact label-correct recall",
        )
    else:
        artifacts["exact_label_match_figure"] = None
    if not coarse_label_match_matrix.empty:
        artifacts["coarse_label_match_figure"] = plot_recall_heatmap(
            coarse_label_match_matrix,
            label_match_counts,
            title="Coarse Label-Correct Recall By Gold Label",
            count_unit="spans",
            output_path=figures_path / "coarse_label_recall_by_gold_label.png",
            colorbar_label="Coarse label-correct recall",
        )
    else:
        artifacts["coarse_label_match_figure"] = None
    artifacts["label_confusion_analysis"] = save_label_confusion_analysis(
        payload, figures_path / "label_confusion"
    )
    if not fp_label_matrix.empty:
        artifacts["non_pii_redaction_label_figure"] = plot_non_pii_redaction_label_distribution(
            fp_label_matrix,
            fp_label_counts,
            title="Non-PII Redaction Labels",
            output_path=figures_path / "non_pii_redaction_labels.png",
        )
    else:
        artifacts["non_pii_redaction_label_figure"] = None
    return artifacts


def save_postprocess_comparison_analysis(
    raw_payload: dict[str, Any],
    postprocessed_payload: dict[str, Any],
    figures_dir: Path | str,
    output_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    comparison_df = build_postprocess_comparison_frame(raw_payload, postprocessed_payload)
    if output_csv_path is not None:
        output_path = Path(output_csv_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(output_path, index=False)

    return {
        "comparison_df": comparison_df,
        "postprocess_value_figure": plot_postprocess_value(
            comparison_df,
            figures_path / "postprocess_value.png",
        ),
    }


def _optional_value(row: pd.Series | None, key: str):
    if row is None:
        return np.nan
    return row.get(key, np.nan)


def _recall_value(result: dict[str, Any], preferred_key: str):
    metric = result.get(preferred_key) or {}
    return metric.get("recall", np.nan)


def _non_pii_redaction_rate(result: dict[str, Any], bucket_key: str, fallback_chars: int):
    bucket = result.get("non_pii_redaction_summaries", {}).get(bucket_key, {})
    fraction = bucket.get("rate")
    if fraction is not None:
        return fraction

    denominator = result.get("gold_character_summary", {}).get("non_pii_chars")
    try:
        denominator = int(denominator)
    except (TypeError, ValueError):
        return np.nan
    return np.nan if denominator == 0 else fallback_chars / denominator


def _display_prediction_label(value: Any) -> str:
    label = str(value or "").strip()
    return label if label else "(missing label)"


def _fraction_float(numerator: int | float | None, denominator: int | float | None) -> float:
    if numerator is None or denominator is None or denominator == 0:
        return np.nan
    return float(numerator) / float(denominator)


def _coarse_label(value: Any) -> str:
    label = str(value or "").strip()
    if not label or label in {"(multiple labels)", "(missing label)", "(missing gold label)"}:
        return ""
    return label.split(":", maxsplit=1)[0]


def _prediction_label_matches_gold(prediction_label: str, gold_label: str, match_level: str) -> bool:
    if match_level == "exact":
        return prediction_label == gold_label
    if match_level == "coarse":
        gold_coarse = _coarse_label(gold_label)
        return bool(gold_coarse and _coarse_label(prediction_label) == gold_coarse)
    raise ValueError(f"Unsupported match level: {match_level}")


def _safe_filename(value: Any) -> str:
    text = str(value or "").strip()
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in text)
    return safe or "annotation"


def _relative_luminance(color: Any) -> float:
    red, green, blue = mcolors.to_rgb(color)

    def channel_luminance(channel: float) -> float:
        return channel / 12.92 if channel <= 0.03928 else ((channel + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel_luminance(red)
        + 0.7152 * channel_luminance(green)
        + 0.0722 * channel_luminance(blue)
    )


def _annotate_grouped_percent_bars(axis, y_positions: np.ndarray, values: pd.Series) -> None:
    for y, value in zip(y_positions, values):
        if pd.isna(value):
            continue
        x = min(float(value) + 0.012, 0.985)
        ha = "left"
        color = "#222222"
        if value > 0.9:
            x = float(value) - 0.012
            ha = "right"
            color = "white"
        axis.text(x, y, f"{value * 100:.1f}%", va="center", ha=ha, color=color, fontsize=8)


def _annotate_detected_pool(axis, y_positions: np.ndarray, frame: pd.DataFrame) -> None:
    for y, row in zip(y_positions, frame.to_dict(orient="records")):
        matched_spans = row.get("matched_core_pii_spans")
        total_spans = row.get("total_core_pii_spans")
        detection_recall = row.get("span_detection_recall")
        if pd.isna(matched_spans) or pd.isna(total_spans) or pd.isna(detection_recall):
            continue
        axis.text(
            1.01,
            y,
            f"pool {int(matched_spans):,}/{int(total_spans):,} ({float(detection_recall) * 100:.1f}%)",
            va="center",
            ha="left",
            fontsize=8,
            color="#333333",
            clip_on=False,
        )


def _sum_optional(*values: Any):
    if any(pd.isna(value) for value in values):
        return np.nan
    return sum(values)


def _delta(after: Any, before: Any):
    if pd.isna(after) or pd.isna(before):
        return np.nan
    return after - before


def _scale(value: Any, factor: float):
    if pd.isna(value):
        return np.nan
    return value * factor


def _symmetric_delta_limit(values: pd.Series, minimum: float) -> float:
    finite_values = values.replace([np.inf, -np.inf], np.nan).dropna()
    if finite_values.empty:
        return minimum

    max_abs = float(finite_values.abs().max())
    return max(max_abs * 1.18, minimum)
