"""Document-clustered bootstrap intervals for de-identification metrics.

The sampling unit is the document.  All gold spans and all predictions from a
sampled document therefore travel together, and the reported rates remain
ratio-of-sums estimates rather than averages of per-document percentages.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


METRIC_LABELS = {
    "core_pii_recall": "Core PII recall",
    "overall_recall": "Annotation-character recall",
    "non_pii_redaction_rate": "Non-PII redaction rate",
}


def _payload_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    if isinstance(payload.get("results"), list):
        return payload["results"]
    if isinstance(payload.get("result"), dict):
        return [
            {
                "annotation_id": payload.get("annotation_id"),
                "display_annotation_id": payload.get("annotation_id"),
                "source_name": payload.get("source_name"),
                "result": payload["result"],
            }
        ]
    raise ValueError("Evaluation payload contains neither `results` nor `result`")


def _union_length(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    total = 0
    current_begin, current_end = sorted(intervals)[0]
    for begin, end in sorted(intervals)[1:]:
        if begin <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_begin
            current_begin, current_end = begin, end
    return total + current_end - current_begin


def _source_name(entry: dict[str, Any]) -> str:
    return str(
        entry.get("source_name")
        or entry.get("display_annotation_id")
        or entry.get("annotation_id")
        or "(missing annotation_id)"
    )


def _document_arrays(
    payload: dict[str, Any],
    document_lengths: Mapping[str, int],
    recall_key: str,
) -> tuple[list[str], list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if recall_key not in {"core_pii_recall", "overall_recall"}:
        raise ValueError("recall_key must be core_pii_recall or overall_recall")

    lengths = {str(key): int(value) for key, value in document_lengths.items()}
    if not lengths:
        raise ValueError("Document-clustered bootstrap requires document lengths")
    if any(value < 0 for value in lengths.values()):
        raise ValueError("Document lengths must be non-negative")

    document_ids = sorted(lengths)
    doc_index = {document_id: index for index, document_id in enumerate(document_ids)}
    entries = _payload_results(payload)
    if not entries:
        raise ValueError("Evaluation payload contains no result sources")

    n_sources = len(entries)
    n_documents = len(document_ids)
    recall_covered = np.zeros((n_sources, n_documents), dtype=np.int64)
    recall_total = np.zeros((n_sources, n_documents), dtype=np.int64)
    non_pii_redacted = np.zeros((n_sources, n_documents), dtype=np.int64)

    gold_ranges: dict[str, list[tuple[int, int]]] = {document_id: [] for document_id in document_ids}
    for source_index, entry in enumerate(entries):
        details = entry.get("result", {}).get("details", [])
        for row in details:
            document_id = str(row.get("document_id"))
            if document_id not in doc_index:
                raise ValueError(
                    f"Evaluation detail references document {document_id!r} without a document length"
                )
            index = doc_index[document_id]
            if row.get("row_kind") == "gold_span":
                metric = row.get(recall_key) or {}
                recall_covered[source_index, index] += int(metric.get("covered_chars") or 0)
                recall_total[source_index, index] += int(metric.get("total_chars") or 0)
                if source_index == 0:
                    gold_range = row.get("gold_range") or {}
                    try:
                        begin, end = int(gold_range["begin"]), int(gold_range["end"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if begin < end:
                        gold_ranges[document_id].append((begin, end))
            elif row.get("row_kind") == "non_pii_redaction":
                non_pii_redacted[source_index, index] += int(
                    row.get("non_pii_redacted_chars") or 0
                )

    non_pii_total = np.asarray(
        [
            lengths[document_id] - _union_length(gold_ranges[document_id])
            for document_id in document_ids
        ],
        dtype=np.int64,
    )
    if np.any(non_pii_total < 0):
        raise ValueError("Gold ranges extend beyond their document length")

    expected_non_pii = payload.get("gold_character_summary", {}).get("non_pii_chars")
    if expected_non_pii is None:
        expected_non_pii = entries[0].get("result", {}).get("gold_character_summary", {}).get(
            "non_pii_chars"
        )
    if expected_non_pii is not None and int(expected_non_pii) != int(non_pii_total.sum()):
        raise ValueError(
            "Per-document gold ranges do not reproduce the evaluator's non-PII denominator "
            f"({int(non_pii_total.sum())} != {int(expected_non_pii)})"
        )

    return (
        document_ids,
        entries,
        recall_covered,
        recall_total,
        non_pii_redacted,
        non_pii_total,
    )


def bootstrap_payload(
    payload: dict[str, Any],
    document_lengths: Mapping[str, int],
    *,
    recall_key: str = "core_pii_recall",
    replicates: int = 10_000,
    seed: int = 20_260_821,
    confidence_level: float = 0.95,
    pairs: list[list[str]] | None = None,
    chunk_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Return estimates, paired differences, and reproducibility metadata."""
    if replicates < 1:
        raise ValueError("bootstrap replicates must be at least 1")
    if not 0 < confidence_level < 1:
        raise ValueError("confidence_level must lie strictly between 0 and 1")

    (
        document_ids,
        entries,
        recall_covered,
        recall_total,
        non_pii_redacted,
        non_pii_total,
    ) = _document_arrays(payload, document_lengths, recall_key)

    n_sources, n_documents = recall_covered.shape
    source_ids = [str(entry.get("annotation_id")) for entry in entries]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("Bootstrap requires unique annotation_id values")

    if pairs is None:
        pair_indices = list(combinations(range(n_sources), 2))
    else:
        index_by_id = {source_id: index for index, source_id in enumerate(source_ids)}
        pair_indices = []
        for pair in pairs:
            if len(pair) != 2 or pair[0] not in index_by_id or pair[1] not in index_by_id:
                raise ValueError(f"Invalid bootstrap pair {pair!r}")
            pair_indices.append((index_by_id[pair[0]], index_by_id[pair[1]]))

    recall_samples = np.empty((replicates, n_sources), dtype=np.float64)
    redaction_samples = np.empty((replicates, n_sources), dtype=np.float64)
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_documents, 1.0 / n_documents)
    if chunk_size is None:
        # Bound the largest temporary counts matrix to roughly 32 MiB.
        chunk_size = max(1, min(256, (32 * 1024 * 1024) // (8 * n_documents)))

    for start in range(0, replicates, chunk_size):
        stop = min(start + chunk_size, replicates)
        counts = rng.multinomial(
            n_documents, probabilities, size=stop - start
        ).astype(np.int64, copy=False)
        sampled_recall_total = counts @ recall_total.T
        sampled_non_pii_total = counts @ non_pii_total
        with np.errstate(divide="ignore", invalid="ignore"):
            recall_samples[start:stop] = (counts @ recall_covered.T) / sampled_recall_total
            redaction_samples[start:stop] = (
                counts @ non_pii_redacted.T
            ) / sampled_non_pii_total[:, None]

    alpha = (1.0 - confidence_level) / 2.0
    quantiles = [alpha, 1.0 - alpha]
    observed_recall = recall_covered.sum(axis=1) / recall_total.sum(axis=1)
    observed_redaction = non_pii_redacted.sum(axis=1) / non_pii_total.sum()

    estimate_rows: list[dict[str, Any]] = []
    for source_index, entry in enumerate(entries):
        for metric, estimate, samples in (
            (recall_key, observed_recall[source_index], recall_samples[:, source_index]),
            (
                "non_pii_redaction_rate",
                observed_redaction[source_index],
                redaction_samples[:, source_index],
            ),
        ):
            lower, upper = np.nanquantile(samples, quantiles)
            estimate_rows.append(
                {
                    "annotation_id": source_ids[source_index],
                    "source": _source_name(entry),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "estimate": float(estimate),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "n_documents": n_documents,
                    "replicates": replicates,
                    "seed": seed,
                    "confidence_level": confidence_level,
                }
            )

    difference_rows: list[dict[str, Any]] = []
    for source_a, source_b in pair_indices:
        for metric, estimates, samples in (
            (recall_key, observed_recall, recall_samples),
            ("non_pii_redaction_rate", observed_redaction, redaction_samples),
        ):
            differences = samples[:, source_a] - samples[:, source_b]
            lower, upper = np.nanquantile(differences, quantiles)
            difference_rows.append(
                {
                    "source_a_id": source_ids[source_a],
                    "source_a": _source_name(entries[source_a]),
                    "source_b_id": source_ids[source_b],
                    "source_b": _source_name(entries[source_b]),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "difference_a_minus_b": float(estimates[source_a] - estimates[source_b]),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "interval_excludes_zero": bool(lower > 0 or upper < 0),
                    "n_documents": n_documents,
                    "replicates": replicates,
                    "seed": seed,
                    "confidence_level": confidence_level,
                }
            )

    methodology = {
        "method": "paired nonparametric bootstrap clustered by document",
        "sampling_unit": "document",
        "rate_estimator": "ratio of character-count sums",
        "confidence_interval": "percentile",
        "confidence_level": confidence_level,
        "replicates": replicates,
        "seed": seed,
        "n_documents": n_documents,
        "document_ids_sorted": True,
        "recall_metric": recall_key,
        "paired_resampling": True,
        "pairing_note": (
            "Every replicate uses the same sampled document multiplicities for all sources, "
            "preserving gold annotations and system outputs within each document."
        ),
    }
    return pd.DataFrame(estimate_rows), pd.DataFrame(difference_rows), methodology


def save_bootstrap(
    payload: dict[str, Any],
    document_lengths: Mapping[str, int],
    output_dir: str | Path,
    **kwargs: Any,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    estimates, differences, methodology = bootstrap_payload(
        payload, document_lengths, **kwargs
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    estimates.to_csv(destination / "estimates.csv", index=False)
    differences.to_csv(destination / "paired_differences.csv", index=False)
    (destination / "methodology.json").write_text(
        json.dumps(methodology, indent=2) + "\n", encoding="utf-8"
    )
    return estimates, differences, methodology
