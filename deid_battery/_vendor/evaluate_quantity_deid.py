#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_IGNORE_CATEGORIES = ["formatting", "additional_info", "medical_info", "time", "title"]
MULTIPLE_LABELS = "(multiple labels)"
MISSING_LABEL = "(missing label)"

DEDUCE_TO_WORKFLOW_LABEL = {
    "patient": "Name:Patient",
    "person": "Name:Caregiver",
    "persoon": "Name:Caregiver",
    "locatie": "Address_Location:Other",
    "date": "Date",
    "datum": "Date",
    "leeftijd": "Age_Birthdate",
    "telefoonnummer": "Contactdetails",
    "emailadres": "Contactdetails",
    "url": "Contactdetails",
    "id": "ID:Patient",
    "bsn": "ID:Patient",
    "healthcare_institution": "Organization:Healthcare",
    "organization": "Organization:Healthcare",
    "zorginstelling": "Organization:Healthcare",
    "ziekenhuis": "Organization:Healthcare",
}

OPENAI_PRIVACY_FILTER_TO_WORKFLOW_LABEL = {
    "account_number": "ID",
    "private_address": "Address_Location",
    "private_date": "Date",
    "private_email": "Contactdetails",
    "private_person": "Name",
    "private_phone": "Contactdetails",
    "private_url": "Contactdetails",
    "secret": "Anonymize_Other",
}

OPENMED_PRIVACY_FILTER_TO_WORKFLOW_LABEL = {
    "FIRSTNAME": "Name",
    "MIDDLENAME": "Name",
    "LASTNAME": "Name",
    "PREFIX": "Name",
    "USERNAME": "ID",
    "ACCOUNTNAME": "ID",
    "AGE": "Age_Birthdate",
    "DATEOFBIRTH": "Age_Birthdate",
    "DATE": "Date",
    "TIME": "Date",
    "EMAIL": "Contactdetails",
    "PHONE": "Contactdetails",
    "URL": "Contactdetails",
    "STREET": "Address_Location",
    "BUILDINGNUMBER": "Address_Location",
    "SECONDARYADDRESS": "Address_Location",
    "CITY": "Address_Location",
    "COUNTY": "Address_Location",
    "STATE": "Address_Location",
    "ZIPCODE": "Address_Location",
    "GPSCOORDINATES": "Address_Location",
    "ORDINALDIRECTION": "Address_Location",
    "OCCUPATION": "Profession",
    "JOBTITLE": "Profession",
    "ORGANIZATION": "Organization",
    "BANKACCOUNT": "ID",
    "IBAN": "ID",
    "BIC": "ID",
    "BITCOINADDRESS": "ID",
    "ETHEREUMADDRESS": "ID",
    "LITECOINADDRESS": "ID",
    "CREDITCARD": "ID",
    "CVV": "ID",
    "PIN": "ID",
    "MASKEDNUMBER": "ID",
    "SSN": "ID",
    "IMEI": "ID",
    "IPADDRESS": "ID",
    "MACADDRESS": "ID",
    "VIN": "ID",
    "VRM": "ID",
    "PASSWORD": "Anonymize_Other",
    "USERAGENT": "Anonymize_Other",
}

BIO_PREFIXES = ("B-", "I-", "E-", "S-")


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _load_manifest(bundle_path: Path) -> dict[str, Any] | None:
    manifest_path = bundle_path / "manifest.json"
    if not manifest_path.exists():
        return None
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def ensure_bundle_gold_coverage_validated(bundle_path: Path) -> None:
    manifest = _load_manifest(bundle_path)
    summary = manifest.get("summary") if isinstance(manifest, dict) else None
    if isinstance(summary, dict) and summary.get("goldCoverageValidated") is True:
        return

    failure_count = None
    if isinstance(summary, dict) and isinstance(summary.get("goldCoverageFailures"), list):
        failure_count = len(summary["goldCoverageFailures"])

    detail = (
        f" ({failure_count} known incomplete gold item(s))"
        if failure_count is not None and failure_count > 0
        else ""
    )
    raise ValueError(
        "Bundle gold coverage has not been validated"
        f"{detail}. Run `npm run evaluation:prepare` and finish every gold span's "
        "subannotations before evaluating."
    )


def _range(raw: dict[str, Any] | None) -> dict[str, int] | None:
    if not raw:
        return None
    try:
        begin = int(raw.get("begin"))
        end = int(raw.get("end"))
    except (TypeError, ValueError):
        return None
    if begin >= end:
        return None
    return {"begin": begin, "end": end}


def _category(raw: dict[str, Any]) -> str:
    for key in ("category", "subannotation_category", "subcategory", "subCategory", "label", "Category"):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _prediction_label(raw: dict[str, Any]) -> str:
    for key in (
        "label",
        "category",
        "Category",
        "tag",
        "type",
        "entity_label",
        "entity_type",
        "native_label",
        "pii_type",
    ):
        value = raw.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _strip_bio_prefix(label: str) -> str:
    for prefix in BIO_PREFIXES:
        if label.startswith(prefix):
            return label[len(prefix) :]
    return label


def _normalize_prediction_label(
    label: Any,
    source_id: str | None = None,
    source_name: str | None = None,
) -> str:
    raw_label = _strip_bio_prefix(str(label or "").strip())
    if not raw_label:
        return MISSING_LABEL

    source_hint = f"{source_id or ''} {source_name or ''}".lower()
    lower_label = raw_label.lower()
    upper_label = raw_label.upper()

    if "openmed" in source_hint and upper_label in OPENMED_PRIVACY_FILTER_TO_WORKFLOW_LABEL:
        return OPENMED_PRIVACY_FILTER_TO_WORKFLOW_LABEL[upper_label]
    if (
        ("privacy" in source_hint or "pii-filter" in source_hint or "openai" in source_hint)
        and lower_label in OPENAI_PRIVACY_FILTER_TO_WORKFLOW_LABEL
    ):
        return OPENAI_PRIVACY_FILTER_TO_WORKFLOW_LABEL[lower_label]
    if "deduce" in source_hint and lower_label in DEDUCE_TO_WORKFLOW_LABEL:
        return DEDUCE_TO_WORKFLOW_LABEL[lower_label]

    if lower_label in DEDUCE_TO_WORKFLOW_LABEL:
        return DEDUCE_TO_WORKFLOW_LABEL[lower_label]
    if lower_label in OPENAI_PRIVACY_FILTER_TO_WORKFLOW_LABEL:
        return OPENAI_PRIVACY_FILTER_TO_WORKFLOW_LABEL[lower_label]
    if upper_label in OPENMED_PRIVACY_FILTER_TO_WORKFLOW_LABEL:
        return OPENMED_PRIVACY_FILTER_TO_WORKFLOW_LABEL[upper_label]
    return raw_label


def _annotation_label(raw: dict[str, Any]) -> str:
    label = str(raw.get("label") or "").strip()
    if label:
        return label
    category = str(raw.get("Category") or raw.get("category") or "").strip()
    subtype = str(raw.get("Subtype") or raw.get("subtype") or "").strip()
    if category and subtype:
        return f"{category}:{subtype}"
    return category


def _merge_adjacent(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not segments:
        return []
    merged = [dict(segments[0])]
    for segment in segments[1:]:
        previous = merged[-1]
        if previous["end"] == segment["begin"] and previous["category"] == segment["category"]:
            previous["end"] = segment["end"]
        else:
            merged.append(dict(segment))
    return merged


def normalize_segments(raw_segments: list[dict[str, Any]], review_range: dict[str, int]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    for raw in raw_segments:
        segment_range = _range(raw)
        category = _category(raw)
        if not segment_range or not category:
            raise ValueError(f"Invalid segment: {raw!r}")
        if segment_range["begin"] < review_range["begin"] or segment_range["end"] > review_range["end"]:
            raise ValueError(f"Segment outside review range: {raw!r}")
        segments.append({**segment_range, "category": category})

    segments.sort(key=lambda segment: (segment["begin"], segment["end"], segment["category"]))
    for previous, current in zip(segments, segments[1:]):
        if current["begin"] < previous["end"]:
            raise ValueError(f"Overlapping segments: {previous!r} and {current!r}")
    return _merge_adjacent(segments)


def load_reference_items(bundle_dir: str | Path | None = None) -> list[dict[str, Any]]:
    bundle_path = Path(bundle_dir) if bundle_dir is not None else Path(__file__).resolve().parent
    reference_path = bundle_path / "reference_items.jsonl"
    items: list[dict[str, Any]] = []
    for raw in parse_jsonl(reference_path):
        review_range = _range(raw.get("review_range"))
        if not review_range:
            raise ValueError(f"Invalid reference review range in item {raw.get('item_id')!r}")
        items.append(
            {
                **raw,
                "review_range": review_range,
                "segments": normalize_segments(raw.get("segments", []), review_range),
            }
        )
    return items


def _prediction_segments(
    annotation: dict[str, Any],
    document_id: str,
    source_id: str | None,
    source_name: str | None = None,
) -> list[dict[str, Any]]:
    nested = annotation.get("subannotations") or annotation.get("subAnnotations") or annotation.get("segments")
    if isinstance(nested, list):
        parent_begin = annotation.get("begin")
        try:
            parent_begin_int = int(parent_begin)
        except (TypeError, ValueError):
            parent_begin_int = 0

        segments: list[dict[str, Any]] = []
        for raw in nested:
            if not isinstance(raw, dict):
                continue
            has_local = "begin" not in raw and ("localBegin" in raw or "local_begin" in raw)
            begin_value = raw.get("begin", raw.get("localBegin", raw.get("local_begin")))
            end_value = raw.get("end", raw.get("localEnd", raw.get("local_end")))
            try:
                begin = int(begin_value) + (parent_begin_int if has_local else 0)
                end = int(end_value) + (parent_begin_int if has_local else 0)
            except (TypeError, ValueError):
                continue
            if begin < end:
                segments.append(
                    {
                        "document_id": document_id,
                        "item_id": annotation.get("item_id") or annotation.get("itemId"),
                        "begin": begin,
                        "end": end,
                        "label": _prediction_label(raw),
                        "source_id": source_id,
                        **({"source_name": source_name} if source_name else {}),
                    }
                )
        return segments

    prediction_range = _range(annotation)
    if not prediction_range:
        return []
    return [
        {
            "document_id": document_id,
            "item_id": annotation.get("item_id") or annotation.get("itemId"),
            **prediction_range,
            "label": _prediction_label(annotation),
            "source_id": source_id,
            **({"source_name": source_name} if source_name else {}),
        }
    ]


def _prediction_files(predictions_path: Path) -> list[Path]:
    if predictions_path.is_dir():
        return sorted(predictions_path.glob("*.jsonl"), key=lambda path: path.name)
    if predictions_path.suffix != ".jsonl":
        return []
    return [predictions_path]


def _row_document_id(row: dict[str, Any]) -> str | None:
    explicit = row.get("doc_id") or row.get("document_id") or row.get("documentId")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()
    return None


def _row_source_id(row: dict[str, Any]) -> str | None:
    explicit = row.get("annotation_id") or row.get("source_id") or row.get("model_id")
    if explicit is not None and str(explicit).strip():
        return str(explicit).strip()

    return None


def _prediction_source_id(row: dict[str, Any], path: Path) -> str | None:
    if _row_document_id(row):
        return path.stem
    return _row_source_id(row)


def _prediction_source_name(row: dict[str, Any]) -> str | None:
    for key in ("source_name", "display_name", "name"):
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _prediction_document_id(row: dict[str, Any], path: Path) -> str:
    return _row_document_id(row) or path.stem


def load_predictions(
    predictions_path: str | Path,
    annotation_id: str | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    predictions: list[dict[str, Any]] = []
    skipped = {
        "rowsByAnnotationId": 0,
        "annotationsWithoutDocumentId": 0,
        "annotationsWithoutUsableRange": 0,
    }

    for path in _prediction_files(Path(predictions_path)):
        for row in parse_jsonl(path):
            source_id = _prediction_source_id(row, path)
            source_name = _prediction_source_name(row)
            if annotation_id and source_id != annotation_id:
                skipped["rowsByAnnotationId"] += 1
                continue

            document_id = _prediction_document_id(row, path)
            if isinstance(row.get("annotations"), list):
                if not document_id:
                    skipped["annotationsWithoutDocumentId"] += len(row["annotations"])
                    continue
                for annotation in row["annotations"]:
                    if not isinstance(annotation, dict):
                        skipped["annotationsWithoutUsableRange"] += 1
                        continue
                    normalized = _prediction_segments(annotation, document_id, source_id, source_name)
                    if not normalized:
                        skipped["annotationsWithoutUsableRange"] += 1
                    predictions.extend(normalized)
                continue

            if isinstance(row.get("entities"), list):
                if not document_id:
                    skipped["annotationsWithoutDocumentId"] += len(row["entities"])
                    continue
                for annotation in row["entities"]:
                    if not isinstance(annotation, dict):
                        skipped["annotationsWithoutUsableRange"] += 1
                        continue
                    normalized = _prediction_segments(annotation, document_id, source_id, source_name)
                    if not normalized:
                        skipped["annotationsWithoutUsableRange"] += 1
                    predictions.extend(normalized)
                continue

            if isinstance(row.get("segments"), list):
                for segment in row["segments"]:
                    if not isinstance(segment, dict):
                        skipped["annotationsWithoutUsableRange"] += 1
                        continue
                    normalized = _prediction_segments(
                        {
                            **segment,
                            "item_id": row.get("item_id")
                            or row.get("itemId")
                            or segment.get("item_id")
                            or segment.get("itemId"),
                        },
                        document_id,
                        source_id,
                        source_name,
                    )
                    if not normalized:
                        skipped["annotationsWithoutUsableRange"] += 1
                    predictions.extend(normalized)
                continue

            normalized = _prediction_segments(row, document_id, source_id, source_name)
            if not normalized:
                skipped["annotationsWithoutUsableRange"] += 1
            predictions.extend(normalized)

    predictions.sort(key=lambda pred: (pred["document_id"], pred["begin"], pred["end"], pred["label"]))
    return predictions, skipped


def _source_sort_key(source_id: Any) -> tuple[int, str]:
    if source_id is None:
        return (1, "")
    return (0, str(source_id))


def _display_source_id(source_id: Any) -> str:
    if source_id is None:
        return "(missing annotation_id)"
    return str(source_id)


def _overlap(a_begin: int, a_end: int, b_begin: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_begin, b_begin))


def _union_intervals(intervals: list[dict[str, int]]) -> list[dict[str, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals, key=lambda item: (item["begin"], item["end"]))
    merged = [dict(ordered[0])]
    for interval in ordered[1:]:
        previous = merged[-1]
        if interval["begin"] <= previous["end"]:
            previous["end"] = max(previous["end"], interval["end"])
        else:
            merged.append(dict(interval))
    return merged


def _covered_by_intervals(begin: int, end: int, intervals: list[dict[str, int]]) -> int:
    covered = 0
    for interval in intervals:
        if interval["end"] <= begin:
            continue
        if interval["begin"] >= end:
            break
        covered += _overlap(begin, end, interval["begin"], interval["end"])
    return covered


def _subtract_intervals(begin: int, end: int, blockers: list[dict[str, int]]) -> list[dict[str, int]]:
    pieces: list[dict[str, int]] = []
    cursor = begin
    for blocker in blockers:
        if blocker["end"] <= cursor:
            continue
        if blocker["begin"] >= end:
            break
        if blocker["begin"] > cursor:
            pieces.append({"begin": cursor, "end": min(blocker["begin"], end)})
        cursor = max(cursor, min(end, blocker["end"]))
        if cursor >= end:
            break
    if cursor < end:
        pieces.append({"begin": cursor, "end": end})
    return pieces


def _metric(total_chars: int, covered_chars: int) -> dict[str, Any]:
    missed_chars = total_chars - covered_chars
    return {
        "total_chars": total_chars,
        "covered_chars": covered_chars,
        "missed_chars": missed_chars,
        "recall": None if total_chars == 0 else round(covered_chars / total_chars, 6),
    }


def _fraction(numerator: int | None, denominator: int | None) -> float | None:
    return None if numerator is None or denominator is None or denominator == 0 else round(numerator / denominator, 6)


def _normalized_document_lengths(document_lengths: Mapping[str, int] | None) -> dict[str, int] | None:
    if document_lengths is None:
        return None
    return {str(document_id): int(length) for document_id, length in document_lengths.items()}


def summarize_gold_characters(
    reference_items: list[dict[str, Any]],
    document_lengths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    lengths_by_doc = _normalized_document_lengths(document_lengths)
    gold_annotation_ranges_by_doc: dict[str, list[dict[str, int]]] = defaultdict(list)
    reference_extent_by_doc: dict[str, int] = defaultdict(int)
    reference_doc_ids = set()

    for item in reference_items:
        doc_id = str(item["document_id"])
        reference_doc_ids.add(doc_id)
        for span in [item.get("review_range"), item.get("gold"), *(item.get("segments") or [])]:
            if isinstance(span, dict):
                span_range = _range(span)
                if span_range:
                    reference_extent_by_doc[doc_id] = max(reference_extent_by_doc[doc_id], span_range["end"])
        if item.get("item_kind") == "gold" and item.get("gold"):
            gold_range = _range(item.get("gold"))
            if gold_range:
                gold_annotation_ranges_by_doc[doc_id].append(gold_range)

    gold_annotation_ranges_by_doc = {
        document_id: _union_intervals(ranges)
        for document_id, ranges in gold_annotation_ranges_by_doc.items()
    }
    pii_chars = sum(
        interval["end"] - interval["begin"]
        for ranges in gold_annotation_ranges_by_doc.values()
        for interval in ranges
    )

    missing_document_lengths: list[str] = []
    estimated_document_lengths: list[str] = []
    total_document_chars = None
    non_pii_chars = None
    if lengths_by_doc is not None:
        total_document_chars = 0
        for doc_id in sorted(reference_doc_ids):
            doc_length = lengths_by_doc.get(doc_id)
            if doc_length is None:
                doc_length = reference_extent_by_doc.get(doc_id)
                if doc_length is None:
                    missing_document_lengths.append(doc_id)
                    continue
                estimated_document_lengths.append(doc_id)
            total_document_chars += doc_length
        non_pii_chars = max(total_document_chars - pii_chars, 0) if not missing_document_lengths else None

    return {
        "pii_chars": pii_chars,
        "non_pii_chars": non_pii_chars,
        "total_document_chars": total_document_chars,
        "pii_fraction": _fraction(pii_chars, total_document_chars),
        "non_pii_fraction": _fraction(non_pii_chars, total_document_chars),
        "definition": {
            "pii": "union of gold annotation ranges",
            "non_pii": "all evaluated document characters outside gold annotation ranges",
        },
        "components": {
            "gold_annotation_chars": pii_chars,
            "outside_gold_annotation_chars": non_pii_chars,
        },
        "missing_document_lengths": missing_document_lengths,
        "estimated_document_lengths": estimated_document_lengths,
    }


def _empty_group() -> dict[str, int]:
    return {
        "gold_span_count": 0,
        "essential_total_chars": 0,
        "essential_covered_chars": 0,
        "excluded_total_chars": 0,
        "excluded_covered_chars": 0,
        "overall_total_chars": 0,
        "overall_covered_chars": 0,
    }


def _add_group(
    group: dict[str, int],
    *,
    essential_total: int,
    essential_covered: int,
    excluded_total: int,
    excluded_covered: int,
    overall_total: int,
    overall_covered: int,
) -> None:
    group["gold_span_count"] += 1
    group["essential_total_chars"] += essential_total
    group["essential_covered_chars"] += essential_covered
    group["excluded_total_chars"] += excluded_total
    group["excluded_covered_chars"] += excluded_covered
    group["overall_total_chars"] += overall_total
    group["overall_covered_chars"] += overall_covered


def _group_summary(key_name: str, key_value: str, group: dict[str, int]) -> dict[str, Any]:
    return {
        key_name: key_value,
        "gold_span_count": group["gold_span_count"],
        "essential_recall": _metric(group["essential_total_chars"], group["essential_covered_chars"]),
        "excluded_recall": _metric(group["excluded_total_chars"], group["excluded_covered_chars"]),
        "overall_recall": _metric(group["overall_total_chars"], group["overall_covered_chars"]),
    }


def _annotation_label_summary(annotation_label: str, group: dict[str, int]) -> dict[str, Any]:
    return {
        "annotation_label": annotation_label,
        "gold_span_count": group["gold_span_count"],
        "essential_recall": _metric(group["essential_total_chars"], group["essential_covered_chars"]),
        "excluded_recall": _metric(group["excluded_total_chars"], group["excluded_covered_chars"]),
        "total_recall": _metric(group["overall_total_chars"], group["overall_covered_chars"]),
    }


def _empty_label_confusion() -> dict[str, Any]:
    return {
        "definition": (
            "Rows are gold annotation labels by default. Columns are normalized prediction labels assigned "
            "to detected essential gold characters. Missed essential characters are excluded from the "
            "matrix cells and reported in row/overall missed counts. A secondary "
            "by_subannotation_category view keeps the fine essential-character breakdown."
        ),
        "total_essential_chars": 0,
        "detected_essential_chars": 0,
        "missed_essential_chars": 0,
        "detected_recall": None,
        "matrix": [],
        "prediction_label_totals": [],
        "normalization": {
            "deduce": "Deduce/native tags are mapped to workflow labels when possible.",
            "privacy_filter": "OpenAI Privacy Filter/OpenMed labels are mapped to coarse workflow labels when possible.",
            "fallback": "Unknown labels are kept as emitted by the prediction source.",
        },
    }


def _prediction_labels_for_piece(
    begin: int,
    end: int,
    predictions: list[dict[str, Any]],
) -> set[str]:
    labels: set[str] = set()
    for prediction in predictions:
        pred_begin = int(prediction["begin"])
        pred_end = int(prediction["end"])
        if pred_end <= begin:
            continue
        if pred_begin >= end:
            break
        if _overlap(begin, end, pred_begin, pred_end) == 0:
            continue
        labels.add(
            _normalize_prediction_label(
                prediction.get("label"),
                source_id=prediction.get("source_id"),
                source_name=prediction.get("source_name"),
            )
        )
    return labels


def _assigned_prediction_label(labels: set[str]) -> str | None:
    if not labels:
        return None
    if len(labels) == 1:
        return next(iter(labels))
    return MULTIPLE_LABELS


def _add_confusion_count(
    matrix: dict[str, dict[str, int]],
    label_totals: dict[str, int],
    gold_category: str,
    prediction_label: str,
    chars: int,
) -> None:
    if chars <= 0:
        return
    matrix[gold_category][prediction_label] += chars
    label_totals[prediction_label] += chars


def _format_confusion_matrix(
    row_key: str,
    row_totals: dict[str, int],
    row_detected: dict[str, int],
    matrix: dict[str, dict[str, int]],
) -> list[dict[str, Any]]:
    return [
        {
            row_key: row_value,
            "total_essential_chars": row_totals[row_value],
            "detected_essential_chars": row_detected[row_value],
            "missed_essential_chars": row_totals[row_value] - row_detected[row_value],
            "detected_recall": None
            if row_totals[row_value] == 0
            else round(row_detected[row_value] / row_totals[row_value], 6),
            "assigned_labels": [
                {
                    "prediction_label": prediction_label,
                    "chars": chars,
                    "fraction_of_detected_row_chars": None
                    if row_detected[row_value] == 0
                    else round(chars / row_detected[row_value], 6),
                    "fraction_of_total_row_chars": None
                    if row_totals[row_value] == 0
                    else round(chars / row_totals[row_value], 6),
                }
                for prediction_label, chars in sorted(matrix[row_value].items())
            ],
        }
        for row_value in sorted(row_totals)
    ]


def _coarse_label(label: Any) -> str:
    raw_label = str(label or "").strip()
    if not raw_label or raw_label in {MULTIPLE_LABELS, MISSING_LABEL, "(missing gold label)"}:
        return ""
    return raw_label.split(":", maxsplit=1)[0]


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _label_confusion_metrics(
    annotation_matrix: list[dict[str, Any]],
    *,
    total_essential_chars: int,
    detected_essential_chars: int,
) -> dict[str, Any]:
    exact_correct_chars = 0
    coarse_correct_chars = 0
    macro_exact_detected_rates: list[float] = []
    macro_coarse_detected_rates: list[float] = []
    macro_exact_total_rates: list[float] = []
    macro_coarse_total_rates: list[float] = []

    for row in annotation_matrix:
        gold_label = str(row.get("annotation_label") or "")
        gold_coarse_label = _coarse_label(gold_label)
        row_exact_correct = 0
        row_coarse_correct = 0
        row_detected_chars = int(row.get("detected_essential_chars") or 0)
        row_total_chars = int(row.get("total_essential_chars") or 0)

        for assigned in row.get("assigned_labels") or []:
            prediction_label = str(assigned.get("prediction_label") or "")
            chars = int(assigned.get("chars") or 0)
            if prediction_label == gold_label:
                row_exact_correct += chars
            if gold_coarse_label and _coarse_label(prediction_label) == gold_coarse_label:
                row_coarse_correct += chars

        exact_correct_chars += row_exact_correct
        coarse_correct_chars += row_coarse_correct
        if row_detected_chars > 0:
            macro_exact_detected_rates.append(row_exact_correct / row_detected_chars)
            macro_coarse_detected_rates.append(row_coarse_correct / row_detected_chars)
        if row_total_chars > 0:
            macro_exact_total_rates.append(row_exact_correct / row_total_chars)
            macro_coarse_total_rates.append(row_coarse_correct / row_total_chars)

    return {
        "definition": {
            "exact_label_accuracy_detected": "exact label match among detected essential gold characters",
            "coarse_label_accuracy_detected": "label category before ':' matches among detected essential gold characters",
            "exact_label_recall": "exact-label-correct detected essential chars divided by all essential gold chars",
            "coarse_label_recall": "coarse-label-correct detected essential chars divided by all essential gold chars",
        },
        "exact_correct_essential_chars": exact_correct_chars,
        "coarse_correct_essential_chars": coarse_correct_chars,
        "exact_label_accuracy_detected": _fraction(exact_correct_chars, detected_essential_chars),
        "coarse_label_accuracy_detected": _fraction(coarse_correct_chars, detected_essential_chars),
        "exact_label_recall": _fraction(exact_correct_chars, total_essential_chars),
        "coarse_label_recall": _fraction(coarse_correct_chars, total_essential_chars),
        "macro_exact_label_accuracy_detected": _mean(macro_exact_detected_rates),
        "macro_coarse_label_accuracy_detected": _mean(macro_coarse_detected_rates),
        "macro_exact_label_recall": _mean(macro_exact_total_rates),
        "macro_coarse_label_recall": _mean(macro_coarse_total_rates),
    }


def build_essential_label_confusion(
    gold_items: list[dict[str, Any]],
    predictions_by_doc: Mapping[str, list[dict[str, Any]]],
    ignored_categories: set[str],
) -> dict[str, Any]:
    by_subannotation_category: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    subannotation_row_totals: dict[str, int] = defaultdict(int)
    subannotation_row_detected: dict[str, int] = defaultdict(int)
    by_annotation_label: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    annotation_row_totals: dict[str, int] = defaultdict(int)
    annotation_row_detected: dict[str, int] = defaultdict(int)
    prediction_label_totals: dict[str, int] = defaultdict(int)

    for item in gold_items:
        document_id = str(item["document_id"])
        annotation_label = _annotation_label(item.get("gold") or {}) or "(missing gold label)"
        doc_predictions = sorted(
            predictions_by_doc.get(document_id, []),
            key=lambda prediction: (int(prediction["begin"]), int(prediction["end"]), str(prediction.get("label", ""))),
        )
        if not doc_predictions:
            for segment in item["segments"]:
                category = str(segment["category"])
                if category not in ignored_categories:
                    chars = int(segment["end"]) - int(segment["begin"])
                    subannotation_row_totals[category] += chars
                    annotation_row_totals[annotation_label] += chars
            continue

        for segment in item["segments"]:
            category = str(segment["category"])
            if category in ignored_categories:
                continue

            segment_begin = int(segment["begin"])
            segment_end = int(segment["end"])
            segment_chars = segment_end - segment_begin
            subannotation_row_totals[category] += segment_chars
            annotation_row_totals[annotation_label] += segment_chars
            boundaries = {segment_begin, segment_end}
            for prediction in doc_predictions:
                pred_begin = int(prediction["begin"])
                pred_end = int(prediction["end"])
                if pred_end <= segment_begin:
                    continue
                if pred_begin >= segment_end:
                    break
                boundaries.add(max(segment_begin, pred_begin))
                boundaries.add(min(segment_end, pred_end))

            ordered_boundaries = sorted(boundaries)
            for begin, end in zip(ordered_boundaries, ordered_boundaries[1:]):
                if begin >= end:
                    continue
                assigned_label = _assigned_prediction_label(
                    _prediction_labels_for_piece(begin, end, doc_predictions)
                )
                if assigned_label is None:
                    continue
                chars = end - begin
                subannotation_row_detected[category] += chars
                annotation_row_detected[annotation_label] += chars
                _add_confusion_count(
                    by_subannotation_category,
                    prediction_label_totals,
                    category,
                    assigned_label,
                    chars,
                )
                by_annotation_label[annotation_label][assigned_label] += chars

    total_essential = sum(subannotation_row_totals.values())
    detected_essential = sum(subannotation_row_detected.values())
    subannotation_matrix = _format_confusion_matrix(
        "gold_category",
        subannotation_row_totals,
        subannotation_row_detected,
        by_subannotation_category,
    )
    annotation_matrix = _format_confusion_matrix(
        "annotation_label",
        annotation_row_totals,
        annotation_row_detected,
        by_annotation_label,
    )
    result = _empty_label_confusion()
    result.update(
        {
            "total_essential_chars": total_essential,
            "detected_essential_chars": detected_essential,
            "missed_essential_chars": total_essential - detected_essential,
            "detected_recall": None
            if total_essential == 0
            else round(detected_essential / total_essential, 6),
            "metrics": _label_confusion_metrics(
                annotation_matrix,
                total_essential_chars=total_essential,
                detected_essential_chars=detected_essential,
            ),
            "matrix": annotation_matrix,
            "by_subannotation_category": subannotation_matrix,
            "by_annotation_label": annotation_matrix,
            "prediction_label_totals": [
                {
                    "prediction_label": prediction_label,
                    "detected_essential_chars": chars,
                    "fraction_of_detected_essential_chars": None
                    if detected_essential == 0
                    else round(chars / detected_essential, 6),
                }
                for prediction_label, chars in sorted(prediction_label_totals.items())
            ],
        }
    )
    return result


def _empty_fp_bucket() -> dict[str, Any]:
    return {
        "prediction_count": 0,
        "total_fp_chars": 0,
        "labeled_fp_chars": 0,
        "unlabeled_fp_chars": 0,
        "by_subannotation_category": defaultdict(int),
    }


def _categorize_piece(begin: int, end: int, segments: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
    breakdown: dict[str, int] = defaultdict(int)
    labeled_chars = 0
    for segment in segments:
        if segment["end"] <= begin:
            continue
        if segment["begin"] >= end:
            break
        overlap = _overlap(begin, end, segment["begin"], segment["end"])
        if overlap == 0:
            continue
        breakdown[segment["category"]] += overlap
        labeled_chars += overlap
    return dict(breakdown), (end - begin) - labeled_chars


def _summarize_fp_bucket(bucket: dict[str, Any], non_pii_chars: int | None) -> dict[str, Any]:
    total = bucket["total_fp_chars"]
    return {
        "prediction_count": bucket["prediction_count"],
        "total_fp_chars": total,
        "fraction_of_non_pii": _fraction(total, non_pii_chars),
        "labeled_fp_chars": bucket["labeled_fp_chars"],
        "unlabeled_fp_chars": bucket["unlabeled_fp_chars"],
        "labeled_fraction": None if total == 0 else round(bucket["labeled_fp_chars"] / total, 6),
        "by_subannotation_category": [
            {
                "category": category,
                "fp_chars": chars,
                "fraction_of_bucket": None if total == 0 else round(chars / total, 6),
                "fraction_of_non_pii": _fraction(chars, non_pii_chars),
            }
            for category, chars in sorted(bucket["by_subannotation_category"].items())
        ],
    }


def evaluate_quantity_deid(
    reference_items: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    ignore_categories: list[str] | set[str] | None = None,
    document_lengths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    ignored = set(DEFAULT_IGNORE_CATEGORIES if ignore_categories is None else ignore_categories)
    gold_items = [
        item
        for item in reference_items
        if item.get("item_kind") == "gold" and item.get("gold") and item.get("segments")
    ]

    predictions_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_doc[str(prediction["document_id"])].append(prediction)

    predictions_by_doc = {
        document_id: sorted(
            doc_predictions,
            key=lambda pred: (int(pred["begin"]), int(pred["end"]), str(pred.get("label", ""))),
        )
        for document_id, doc_predictions in predictions_by_doc.items()
    }

    prediction_union_by_doc = {
        document_id: _union_intervals([
            {"begin": int(pred["begin"]), "end": int(pred["end"])}
            for pred in doc_predictions
        ])
        for document_id, doc_predictions in predictions_by_doc.items()
    }

    gold_ranges_by_doc: dict[str, list[dict[str, int]]] = defaultdict(list)
    all_segments_by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reference_items:
        doc_id = str(item["document_id"])
        all_segments_by_doc[doc_id].extend(item.get("segments", []))
        if item.get("item_kind") == "gold" and item.get("gold"):
            gold_ranges_by_doc[doc_id].append(dict(item["review_range"]))

    gold_ranges_by_doc = {
        document_id: _union_intervals(ranges)
        for document_id, ranges in gold_ranges_by_doc.items()
    }
    all_segments_by_doc = {
        document_id: sorted(segments, key=lambda segment: (segment["begin"], segment["end"], segment["category"]))
        for document_id, segments in all_segments_by_doc.items()
    }

    overall = _empty_group()
    by_gold_category: dict[str, dict[str, int]] = defaultdict(_empty_group)
    by_gold_label: dict[str, dict[str, int]] = defaultdict(_empty_group)
    by_annotation_label: dict[str, dict[str, int]] = defaultdict(_empty_group)
    by_subannotation_category: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total_chars": 0, "covered_chars": 0}
    )
    details: list[dict[str, Any]] = []

    fp_buckets = {
        "machine_only_fp": _empty_fp_bucket(),
        "overflow_fp": _empty_fp_bucket(),
    }

    for document_id, doc_predictions in predictions_by_doc.items():
        gold_ranges = gold_ranges_by_doc.get(document_id, [])
        doc_segments = all_segments_by_doc.get(document_id, [])

        for prediction_index, prediction in enumerate(doc_predictions):
            pred_begin = int(prediction["begin"])
            pred_end = int(prediction["end"])
            fp_pieces = _subtract_intervals(pred_begin, pred_end, gold_ranges)
            if not fp_pieces:
                continue

            overlaps_gold = any(
                _overlap(pred_begin, pred_end, gold_range["begin"], gold_range["end"]) > 0
                for gold_range in gold_ranges
            )
            bucket_key = "overflow_fp" if overlaps_gold else "machine_only_fp"
            bucket = fp_buckets[bucket_key]
            bucket["prediction_count"] += 1

            total_fp_chars = 0
            labeled_fp_chars = 0
            unlabeled_fp_chars = 0
            breakdown: dict[str, int] = defaultdict(int)

            for piece in fp_pieces:
                piece_begin = piece["begin"]
                piece_end = piece["end"]
                piece_total = piece_end - piece_begin
                piece_breakdown, piece_unlabeled = _categorize_piece(piece_begin, piece_end, doc_segments)
                total_fp_chars += piece_total
                unlabeled_fp_chars += piece_unlabeled
                labeled_fp_chars += piece_total - piece_unlabeled
                for category, chars in piece_breakdown.items():
                    breakdown[category] += chars

            bucket["total_fp_chars"] += total_fp_chars
            bucket["labeled_fp_chars"] += labeled_fp_chars
            bucket["unlabeled_fp_chars"] += unlabeled_fp_chars
            for category, chars in breakdown.items():
                bucket["by_subannotation_category"][category] += chars

            details.append(
                {
                    "row_kind": "prediction_fp",
                    "document_id": document_id,
                    "prediction_index": prediction_index,
                    "prediction_label": prediction.get("label", ""),
                    "prediction_range": {"begin": pred_begin, "end": pred_end},
                    "fp_bucket": bucket_key,
                    "total_fp_chars": total_fp_chars,
                    "labeled_fp_chars": labeled_fp_chars,
                    "unlabeled_fp_chars": unlabeled_fp_chars,
                    "subannotation_breakdown": [
                        {"category": category, "fp_chars": chars}
                        for category, chars in sorted(breakdown.items())
                    ],
                }
            )

    for item in gold_items:
        document_id = str(item["document_id"])
        pred_intervals = prediction_union_by_doc.get(document_id, [])
        gold = item["gold"]
        gold_category = str(gold.get("Category") or "")
        gold_label = str(gold.get("label") or "")
        annotation_label = _annotation_label(gold)

        essential_total = 0
        essential_covered = 0
        excluded_total = 0
        excluded_covered = 0
        overall_total = 0
        overall_covered = 0
        row_breakdown: list[dict[str, Any]] = []

        for segment in item["segments"]:
            total_chars = int(segment["end"]) - int(segment["begin"])
            covered_chars = _covered_by_intervals(int(segment["begin"]), int(segment["end"]), pred_intervals)
            category = str(segment["category"])

            overall_total += total_chars
            overall_covered += covered_chars
            if category in ignored:
                excluded_total += total_chars
                excluded_covered += covered_chars
            else:
                essential_total += total_chars
                essential_covered += covered_chars

            by_subannotation_category[category]["total_chars"] += total_chars
            by_subannotation_category[category]["covered_chars"] += covered_chars
            row_breakdown.append(
                {
                    "category": category,
                    "ignored_in_headline": category in ignored,
                    **_metric(total_chars, covered_chars),
                }
            )

        for group in (
            overall,
            by_gold_category[gold_category],
            by_gold_label[gold_label],
            by_annotation_label[annotation_label],
        ):
            _add_group(
                group,
                essential_total=essential_total,
                essential_covered=essential_covered,
                excluded_total=excluded_total,
                excluded_covered=excluded_covered,
                overall_total=overall_total,
                overall_covered=overall_covered,
            )

        details.append(
            {
                "row_kind": "gold_span",
                "document_id": document_id,
                "item_id": item.get("item_id"),
                "gold_index": item.get("gold_index"),
                "gold_label": gold_label,
                "gold_category": gold_category,
                "gold_subtype": gold.get("Subtype"),
                "gold_range": dict(item["review_range"]),
                "essential_recall": _metric(essential_total, essential_covered),
                "excluded_recall": _metric(excluded_total, excluded_covered),
                "overall_recall": _metric(overall_total, overall_covered),
                "subannotation_breakdown": row_breakdown,
            }
        )

    non_ignored_recall = _metric(overall["essential_total_chars"], overall["essential_covered_chars"])
    essential_label_confusion = build_essential_label_confusion(
        gold_items=gold_items,
        predictions_by_doc=predictions_by_doc,
        ignored_categories=ignored,
    )
    gold_character_summary = summarize_gold_characters(reference_items, document_lengths=document_lengths)
    non_pii_chars = gold_character_summary["non_pii_chars"]

    return {
        "summary": {
            "reference_items": len(reference_items),
            "gold_span_count": overall["gold_span_count"],
            "prediction_span_count": len(predictions),
            "ignore_categories": sorted(ignored),
        },
        "gold_character_summary": gold_character_summary,
        "non_ignored_recall": non_ignored_recall,
        "essential_recall": non_ignored_recall,
        "essential_label_confusion": essential_label_confusion,
        "excluded_recall": _metric(overall["excluded_total_chars"], overall["excluded_covered_chars"]),
        "overall_recall": _metric(overall["overall_total_chars"], overall["overall_covered_chars"]),
        "by_gold_category": [
            _group_summary("gold_category", key, by_gold_category[key])
            for key in sorted(by_gold_category)
        ],
        "by_gold_label": [
            _group_summary("gold_label", key, by_gold_label[key])
            for key in sorted(by_gold_label)
        ],
        "by_annotation_label": [
            _annotation_label_summary(key, by_annotation_label[key])
            for key in sorted(by_annotation_label)
        ],
        "by_subannotation_category": [
            {
                "category": category,
                "ignored_in_headline": category in ignored,
                **_metric(entry["total_chars"], entry["covered_chars"]),
            }
            for category, entry in sorted(by_subannotation_category.items())
        ],
        "false_positive_summaries": {
            bucket_key: _summarize_fp_bucket(bucket, non_pii_chars)
            for bucket_key, bucket in fp_buckets.items()
        },
        "details": sorted(
            details,
            key=lambda row: (
                0 if row.get("row_kind") == "gold_span" else 1,
                str(row.get("document_id", "")),
                int(row.get("gold_index") if row.get("gold_index") is not None else row.get("prediction_index", -1)),
            ),
        ),
    }


def evaluate_predictions(
    predictions_path: str | Path,
    bundle_dir: str | Path | None = None,
    annotation_id: str | None = None,
    output_path: str | Path | None = None,
    ignore_categories: list[str] | set[str] | None = None,
    require_complete_gold: bool = True,
    document_lengths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_dir) if bundle_dir is not None else Path(__file__).resolve().parent
    if require_complete_gold:
        ensure_bundle_gold_coverage_validated(bundle_path)
    reference_items = load_reference_items(bundle_path)
    predictions, skipped = load_predictions(predictions_path, annotation_id=annotation_id)
    source_name = _source_name_for_predictions(predictions)
    result = evaluate_quantity_deid(
        reference_items=reference_items,
        predictions=predictions,
        ignore_categories=ignore_categories,
        document_lengths=document_lengths,
    )
    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_path),
        "predictions_path": str(predictions_path),
        "annotation_id": annotation_id,
        **({"source_name": source_name} if source_name else {}),
        "gold_character_summary": result["gold_character_summary"],
        "skipped_predictions": skipped,
        "result": result,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def evaluate_all_prediction_sources(
    predictions_path: str | Path,
    bundle_dir: str | Path | None = None,
    output_path: str | Path | None = None,
    ignore_categories: list[str] | set[str] | None = None,
    require_complete_gold: bool = True,
    document_lengths: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    bundle_path = Path(bundle_dir) if bundle_dir is not None else Path(__file__).resolve().parent
    if require_complete_gold:
        ensure_bundle_gold_coverage_validated(bundle_path)
    reference_items = load_reference_items(bundle_path)
    predictions, skipped = load_predictions(predictions_path)

    predictions_by_source: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_source[prediction.get("source_id")].append(prediction)

    source_names_by_source = {
        source_id: _source_name_for_predictions(source_predictions)
        for source_id, source_predictions in predictions_by_source.items()
    }

    results = []
    for source_id in sorted(predictions_by_source, key=_source_sort_key):
        result = evaluate_quantity_deid(
            reference_items=reference_items,
            predictions=predictions_by_source[source_id],
            ignore_categories=ignore_categories,
            document_lengths=document_lengths,
        )
        results.append(
            {
                "annotation_id": source_id,
                "display_annotation_id": _display_source_id(source_id),
                **({"source_name": source_names_by_source[source_id]} if source_names_by_source.get(source_id) else {}),
                "result": result,
            }
        )

    payload = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "bundle_dir": str(bundle_path),
        "predictions_path": str(predictions_path),
        "annotation_ids": [entry["annotation_id"] for entry in results],
        "source_names": {
            str(source_id): source_name
            for source_id, source_name in source_names_by_source.items()
            if source_name
        },
        "gold_character_summary": (
            results[0]["result"]["gold_character_summary"]
            if results
            else summarize_gold_characters(reference_items, document_lengths=document_lengths)
        ),
        "skipped_predictions": skipped,
        "results": results,
    }
    if output_path is not None:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _source_name_for_predictions(predictions: list[dict[str, Any]]) -> str | None:
    names = {
        str(prediction["source_name"]).strip()
        for prediction in predictions
        if prediction.get("source_name") is not None and str(prediction["source_name"]).strip()
    }
    if len(names) == 1:
        return next(iter(names))
    return None


def _format_recall(value: dict[str, Any]) -> str:
    recall = value.get("recall")
    return "n/a" if recall is None else f"{recall * 100:.1f}%"


def _metric_summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall": _format_recall(value),
        "totalChars": value["total_chars"],
        "coveredChars": value["covered_chars"],
        "missedChars": value["missed_chars"],
    }


def _annotation_label_cli_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "annotationLabel": row["annotation_label"],
        "goldSpans": row["gold_span_count"],
        "essentialRecall": _metric_summary(row["essential_recall"]),
        "excludedRecall": _metric_summary(row["excluded_recall"]),
        "overallRecall": _metric_summary(row["total_recall"]),
    }


def _result_cli_summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "goldSpans": result["summary"]["gold_span_count"],
        "predictions": result["summary"]["prediction_span_count"],
        "ignoredSubannotationCategories": result["summary"]["ignore_categories"],
        "goldCharacters": result["gold_character_summary"],
        "essentialRecall": _metric_summary(result["essential_recall"]),
        "excludedRecall": _metric_summary(result["excluded_recall"]),
        "overallRecall": _metric_summary(result["overall_recall"]),
        "essentialLabelConfusion": result["essential_label_confusion"],
        "falsePositives": {
            "machineOnlyChars": result["false_positive_summaries"]["machine_only_fp"]["total_fp_chars"],
            "machineOnlyFractionOfNonPii": result["false_positive_summaries"]["machine_only_fp"]["fraction_of_non_pii"],
            "overflowChars": result["false_positive_summaries"]["overflow_fp"]["total_fp_chars"],
            "overflowFractionOfNonPii": result["false_positive_summaries"]["overflow_fp"]["fraction_of_non_pii"],
        },
        "byAnnotationLabel": [
            _annotation_label_cli_summary(row)
            for row in result["by_annotation_label"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate de-identification coverage against this bundle.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL file or directory.")
    parser.add_argument(
        "--bundle",
        default=Path(__file__).resolve().parent,
        help="Bundle directory. Defaults to this script's directory.",
    )
    parser.add_argument("--annotation-id", default=None, help="Only evaluate this source/model id.")
    parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Evaluate every loaded annotation_id/source_id/model_id separately.",
    )
    parser.add_argument(
        "--allow-incomplete-gold",
        action="store_true",
        help="Evaluate a diagnostic bundle even when gold coverage was not validated.",
    )
    parser.add_argument("--output", default=None, help="Optional result JSON output path.")
    parser.add_argument(
        "--ignore-categories",
        nargs="*",
        default=DEFAULT_IGNORE_CATEGORIES,
        help="Subannotation categories excluded from essential recall.",
    )
    args = parser.parse_args()

    if args.all_sources and args.annotation_id:
        raise SystemExit("--all-sources cannot be combined with --annotation-id")

    if args.all_sources:
        payload = evaluate_all_prediction_sources(
            predictions_path=args.predictions,
            bundle_dir=args.bundle,
            output_path=args.output,
            ignore_categories=args.ignore_categories,
            require_complete_gold=not args.allow_incomplete_gold,
        )
        summaries = []
        for entry in payload["results"]:
            result = entry["result"]
            summaries.append(
                {
                    "annotationId": entry["display_annotation_id"],
                    **_result_cli_summary(result),
                }
            )
        print(json.dumps(summaries, indent=2))
        return

    payload = evaluate_predictions(
        predictions_path=args.predictions,
        bundle_dir=args.bundle,
        annotation_id=args.annotation_id,
        output_path=args.output,
        ignore_categories=args.ignore_categories,
        require_complete_gold=not args.allow_incomplete_gold,
    )
    result = payload["result"]
    summary = _result_cli_summary(result)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        raise SystemExit(str(exc))
