"""Core PII recall + non-PII redaction rate plot, from an evaluation payload.

Wraps the vendored ``evaluation_plots`` (matplotlib). Sources with zero
predictions are dropped from the plot. Returns the summary DataFrame.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "_vendor"
sys.path.insert(0, str(_VENDOR))


def _csv_beside(out_path):
    """The numerical twin of a figure path: same name, ``.csv`` for ``.png`` (so
    every plot lands next to the exact numbers it draws). ``None`` -> ``None``."""
    return None if out_path is None else Path(out_path).with_suffix(".csv")


def _recall_matrix_to_frame(matrix, counts, row_name, count_name):
    """Flatten the (row-label x source) recall matrix behind a heatmap into a
    saveable frame: the row label, its gold count (the number shown in the
    heatmap's y-tick), then one recall column per source -- rows kept in the
    plotted order (worst mean recall first)."""
    frame = matrix.reset_index()
    frame.columns = [row_name, *frame.columns[1:]]
    frame.insert(1, count_name, [int(counts.get(r, 0)) for r in matrix.index])
    return frame


def plot(payload: dict, out_path):
    import evaluation_plots as ep

    summary = ep.build_summary_frame(payload)
    if not summary.empty:
        summary = summary[summary["prediction_span_count"] > 0].reset_index(drop=True)
    ep.plot_recall_and_non_pii_redaction(summary, output_path=out_path)
    return summary


def plot_time_vs_recall(payload: dict, timings: dict, out_path,
                        include_sids=None, sid_to_model=None, sid_to_label=None,
                        recall_key: str = "core_pii_recall", csv_path=None):
    """Scatter of warm end-to-end time vs recall, one dot per timings row,
    coloured by device (cpu/gpu). ``include_sids`` restricts to the with-metadata
    sources; ``sid_to_model`` maps a source id (e.g. ``uza@meta``) to its model id
    (``uza``), the key under which times live in ``timings``; ``sid_to_label`` maps
    it to the bare method name for the point label (``UZA``) -- the condition is the
    same for every point (with metadata), so that belongs in the figure caption,
    not on each dot. Two rows for one method (e.g. a measured CPU time + a
    hand-added GPU time) become two dots at the same recall, linked by a faint
    line. Returns the figure, or None if there is nothing to plot."""
    import evaluation_plots as ep
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib import ticker
    from matplotlib.lines import Line2D

    from .timing import is_measured, normalize_device

    summary = ep.build_summary_frame(payload)
    if summary.empty:
        return None
    sid_to_model = sid_to_model or {}
    sid_to_label = sid_to_label or {}

    rows = []
    for _, r in summary.iterrows():
        sid = str(r["annotation_id"])
        if include_sids is not None and sid not in include_sids:
            continue
        recall = r.get(recall_key)
        if recall is None or (isinstance(recall, float) and np.isnan(recall)):
            continue
        model = sid_to_model.get(sid, sid)
        name = sid_to_label.get(sid, str(r["source"]))  # bare method name (no condition suffix)
        for e in timings.get(model, []):
            # Never mix legacy cold-stage or scope-unknown manual measurements
            # into the fair neural comparison. Manual values can opt in by adding
            # `timing_scope: warm_end_to_end` to their YAML row.
            if e.get("timing_scope") != "warm_end_to_end":
                continue
            secs = e.get("seconds")
            if secs is None:
                continue
            rows.append({"name": name, "seconds": float(secs),
                         "recall": float(recall), "device": normalize_device(e.get("device")),
                         "source": "measured" if is_measured(e) else "manual",
                         "timing_scope": e.get("timing_scope"),
                         "service_setup": e.get("service_setup", "unspecified")})
    if not rows:
        return None
    df = pd.DataFrame(rows)

    # Numerical twin of the scatter: one row per plotted dot (method, its run time,
    # recall, device, and whether the time was measured or hand-added).
    csv_path = Path(csv_path) if csv_path is not None else _csv_beside(out_path)
    if csv_path is not None:
        (df.rename(columns={"name": "method", "source": "timing_source"})
           [["method", "seconds", "recall", "device", "timing_source",
             "timing_scope", "service_setup"]]
           .sort_values(["recall", "seconds"], ascending=[False, True])
           .to_csv(csv_path, index=False))

    devices = list(dict.fromkeys(df["device"]))
    base = {"cpu": "#4c72b0", "gpu": "#dd8452"}
    spare = [c for c in plt.get_cmap("tab10").colors]
    color_for, si = {}, 0
    for d in devices:
        if d in base:
            color_for[d] = base[d]
        else:
            color_for[d] = spare[si % len(spare)]
            si += 1
    marker_for = {"measured": "o", "manual": "D"}

    fig, ax = plt.subplots(figsize=(11, 7), constrained_layout=True)

    # link the dots of one method (e.g. its cpu vs gpu points) so the pairing reads
    for name, g in df.groupby("name"):
        if len(g) > 1:
            gg = g.sort_values("seconds")
            ax.plot(gg["seconds"], gg["recall"], color="#c8c8c8", linewidth=1.0, zorder=1)

    for (device, source), g in df.groupby(["device", "source"]):
        ax.scatter(g["seconds"], g["recall"], s=95, color=color_for[device],
                   marker=marker_for.get(source, "o"), edgecolor="#222222",
                   linewidth=0.8, zorder=3)
    for _, r in df.iterrows():
        ax.annotate(r["name"], (r["seconds"], r["recall"]), xytext=(6, 4),
                    textcoords="offset points", fontsize=8, color="#222222")

    smin, smax = float(df["seconds"].min()), float(df["seconds"].max())
    if smin > 0 and smax / smin > 30:  # wide spread (deduce seconds vs LLM minutes) -> log
        ax.set_xscale("log")
        ax.set_xlabel("Warm end-to-end time (seconds, log scale)")
    else:
        ax.set_xlabel("Warm end-to-end time (seconds)")
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.set_ylabel("Core PII recall (with metadata)")
    ax.set_title("Warm End-to-End Time vs. Recall", fontsize=18, weight="bold", pad=14)
    ax.grid(color="#d0d0d0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    device_handles = [Line2D([0], [0], marker="o", linestyle="", markersize=10,
                             markerfacecolor=color_for[d], markeredgecolor="#222222",
                             label=d.upper()) for d in devices]
    # Keep both legends along the bottom so the high-recall region (top, where the
    # good systems and their labels sit) stays clear.
    device_legend = ax.legend(handles=device_handles, title="Device", frameon=False,
                              loc="lower right")
    ax.add_artist(device_legend)
    sources = [s for s in ("measured", "manual") if s in set(df["source"])]
    if len(sources) > 1:
        source_handles = [Line2D([0], [0], marker=marker_for[s], linestyle="", markersize=10,
                                 markerfacecolor="#999999", markeredgecolor="#222222", label=s)
                          for s in sources]
        ax.legend(handles=source_handles, title="Source", frameon=False, loc="lower left")

    if out_path is not None:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
    return fig


def plot_recall_by_gold_label(payload: dict, out_path, csv_path=None):
    """Core PII recall heatmap: gold label (row) x source (col)."""
    import evaluation_plots as ep
    matrix, counts = ep.build_recall_matrix(
        payload, group_name="by_gold_label", row_key="gold_label",
        metric_key="core_pii_recall", count_key="gold_span_count")
    if matrix.empty:
        return None
    csv_path = Path(csv_path) if csv_path is not None else _csv_beside(out_path)
    if csv_path is not None:
        _recall_matrix_to_frame(matrix, counts, "gold_label", "gold_span_count").to_csv(
            csv_path, index=False)
    return ep.plot_recall_heatmap(matrix, counts, "Core PII recall by gold label",
                                  "spans", output_path=out_path)


def plot_recall_by_subannotation_category(payload: dict, out_path, csv_path=None):
    """Recall heatmap: subannotation category (row) x source (col)."""
    import evaluation_plots as ep
    matrix, counts = ep.build_recall_matrix(
        payload, group_name="by_subannotation_category", row_key="category",
        metric_key=None, count_key="total_chars",
        row_filter=ep._is_plottable_subannotation_category)
    if matrix.empty:
        return None
    csv_path = Path(csv_path) if csv_path is not None else _csv_beside(out_path)
    if csv_path is not None:
        _recall_matrix_to_frame(matrix, counts, "category", "total_chars").to_csv(
            csv_path, index=False)
    return ep.plot_recall_heatmap(matrix, counts, "Recall by subannotation category",
                                  "chars", output_path=out_path)


def save_label_confusion_analysis(payload: dict, raw_dir, plots_dir=None):
    """Write span-level label confusion data and plots."""
    import evaluation_plots as ep

    return ep.save_label_confusion_analysis(payload, raw_dir, plots_dir)
