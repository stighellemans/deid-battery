"""Run privacy-safe gold-span and end-to-end pseudonymization evaluations.

The gold-span analysis isolates the shared substitution layer.  The predicted-
span analysis answers the downstream question: how often does a gold date or
age/birthdate receive a protocol-valid transformation after model detection,
boundary selection, and label assignment have all taken effect?
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


EVALUATED_LABELS = ("Date", "Age_Birthdate")
PREDICTED_SUMMARY_LABELS = ("Overall", *EVALUATED_LABELS)
PREDICTED_FAILURE_REASONS = (
    "no_prediction_overlap",
    "incomplete_prediction_coverage",
    "fragmented_prediction_coverage",
    "incorrect_label",
    "unsupported_apostrophe_year",
    "unsupported_approximate_age",
    "unsupported_trailing_punctuation",
    "unsupported_mixed_format_range",
    "unsupported_or_invalid_format",
    "pseudonymizer_exception",
    "empty_substitution",
    "output_not_bracketed",
    "exact_date_shift_mismatch",
    "birthdate_not_reduced_to_age",
    "age_output_not_age_like",
    "missing_replacement",
    "placeholder_fallback",
    "unexpected_replacement",
    "invalid_transformation",
)
PREDICTED_SUMMARY_COLUMNS = (
    "label",
    "gold_spans",
    "end_to_end_valid",
    "end_to_end_failed",
    "end_to_end_valid_rate",
    "end_to_end_failure_rate",
    "fully_redacted",
    "residual_exposure",
    "fully_redacted_rate",
    "residual_exposure_rate",
)
PREDICTED_FAILURE_COLUMNS = (
    "label",
    "failure_reason",
    "count",
    "fraction_of_label",
)
PREDICTED_EXPORT_FILES = {
    "summary.csv",
    "failure_reasons.csv",
    "methodology.json",
}
PREDICTED_OPTIONAL_EXPORT_FILES = {"privacy_manifest.json"}
_SAFE_SOURCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@+-]*")


@dataclass(frozen=True)
class PredictedSpan:
    document_id: str
    prediction_index: int
    label: str
    begin: int
    end: int


@dataclass(frozen=True)
class PredictedOutcome:
    label: str
    end_to_end_valid: bool
    fully_redacted: bool
    failure_reason: str | None


def _resolve(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _load_evaluator():
    # Keep both sides of this evaluation pinned inside the battery: the
    # evaluator snapshot and the post-process pseudonymizer it exercises.
    vendor_root = Path(__file__).resolve().parent / "_vendor"
    sys.path.insert(0, str(vendor_root / "evaluation"))
    sys.path.insert(0, str(vendor_root / "post_process"))
    try:
        from pseudonymization_evaluation import (  # type: ignore
            EvaluationSettings,
            GoldSpan,
            evaluate_spans,
            load_document_texts,
            load_gold_date_spans,
            write_private_details,
            write_safe_export,
        )
    except ImportError as error:
        raise ImportError(
            "The bundled pseudonymization evaluator could not be loaded."
        ) from error
    return {
        "EvaluationSettings": EvaluationSettings,
        "GoldSpan": GoldSpan,
        "evaluate_spans": evaluate_spans,
        "load_document_texts": load_document_texts,
        "load_gold_date_spans": load_gold_date_spans,
        "write_private_details": write_private_details,
        "write_safe_export": write_safe_export,
    }


def _interval_is_covered(begin: int, end: int, spans: Sequence[PredictedSpan]) -> bool:
    """Return whether the union of overlapping prediction intervals covers a target."""
    cursor = begin
    for span in sorted(spans, key=lambda value: (value.begin, value.end)):
        if span.end <= cursor:
            continue
        if span.begin > cursor:
            return False
        cursor = max(cursor, span.end)
        if cursor >= end:
            return True
    return False


def score_predicted_spans(
    gold_spans: Sequence[Any],
    predictions: Sequence[PredictedSpan],
    transformation_results: Mapping[tuple[str, int], tuple[bool, str | None]],
) -> list[PredictedOutcome]:
    """Classify each gold Date/Age_Birthdate span after predicted-span processing.

    A target is end-to-end valid only when one prediction fully contains it,
    has the same label, and its generated substitution passes the transformation
    protocol.  ``fully_redacted`` is deliberately label-agnostic: it shows when
    the source identifier was removed even though the clinically useful
    pseudonym may be wrong.
    """
    predictions_by_document: dict[str, list[PredictedSpan]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_document[prediction.document_id].append(prediction)

    outcomes: list[PredictedOutcome] = []
    for gold in gold_spans:
        overlapping = [
            prediction
            for prediction in predictions_by_document.get(gold.document_id, [])
            if prediction.begin < gold.end and gold.begin < prediction.end
        ]
        fully_redacted = _interval_is_covered(gold.begin, gold.end, overlapping)
        containing = [
            prediction
            for prediction in overlapping
            if prediction.begin <= gold.begin and prediction.end >= gold.end
        ]
        same_label = [
            prediction for prediction in containing if prediction.label == gold.label
        ]
        valid_candidates = [
            prediction
            for prediction in same_label
            if transformation_results.get(
                (prediction.document_id, prediction.prediction_index), (False, None)
            )[0]
        ]
        if valid_candidates:
            outcomes.append(
                PredictedOutcome(
                    label=gold.label,
                    end_to_end_valid=True,
                    fully_redacted=True,
                    failure_reason=None,
                )
            )
            continue

        if not overlapping:
            reason = "no_prediction_overlap"
        elif not fully_redacted:
            reason = "incomplete_prediction_coverage"
        elif not containing:
            reason = "fragmented_prediction_coverage"
        elif not same_label:
            reason = "incorrect_label"
        else:
            reasons = [
                transformation_results.get(
                    (prediction.document_id, prediction.prediction_index),
                    (False, "invalid_transformation"),
                )[1]
                or "invalid_transformation"
                for prediction in same_label
            ]
            reason = reasons[0]
            if reason not in PREDICTED_FAILURE_REASONS:
                reason = "invalid_transformation"
        outcomes.append(
            PredictedOutcome(
                label=gold.label,
                end_to_end_valid=False,
                fully_redacted=fully_redacted,
                failure_reason=reason,
            )
        )
    return outcomes


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_predicted_aggregate_tables(
    outcomes: Sequence[PredictedOutcome],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[PredictedOutcome]] = defaultdict(list)
    by_label["Overall"].extend(outcomes)
    for outcome in outcomes:
        by_label[outcome.label].append(outcome)

    summary: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for label in PREDICTED_SUMMARY_LABELS:
        group = by_label.get(label, [])
        total = len(group)
        valid = sum(outcome.end_to_end_valid for outcome in group)
        fully_redacted = sum(outcome.fully_redacted for outcome in group)
        summary.append(
            {
                "label": label,
                "gold_spans": total,
                "end_to_end_valid": valid,
                "end_to_end_failed": total - valid,
                "end_to_end_valid_rate": _fraction(valid, total),
                "end_to_end_failure_rate": _fraction(total - valid, total),
                "fully_redacted": fully_redacted,
                "residual_exposure": total - fully_redacted,
                "fully_redacted_rate": _fraction(fully_redacted, total),
                "residual_exposure_rate": _fraction(total - fully_redacted, total),
            }
        )
        counts = Counter(
            outcome.failure_reason
            for outcome in group
            if outcome.failure_reason is not None
        )
        for reason in PREDICTED_FAILURE_REASONS:
            count = counts[reason]
            if count:
                failures.append(
                    {
                        "label": label,
                        "failure_reason": reason,
                        "count": count,
                        "fraction_of_label": _fraction(count, total),
                    }
                )
    return summary, failures


def _write_csv(
    path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _validate_numeric(value: str, *, integer: bool = False) -> None:
    try:
        number = int(value) if integer else float(value)
    except ValueError as error:
        raise ValueError(f"Expected numeric aggregate value, got {value!r}") from error
    if not integer and (math.isnan(number) or math.isinf(number)):
        raise ValueError("Aggregate values must be finite")


def validate_predicted_safe_export(export_dir: str | Path) -> dict[str, Any]:
    """Fail closed if a predicted-span export contains unsafe fields or strings."""
    output_dir = Path(export_dir)
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = actual_files - PREDICTED_EXPORT_FILES - PREDICTED_OPTIONAL_EXPORT_FILES
    missing = PREDICTED_EXPORT_FILES - actual_files
    if unexpected or missing:
        raise ValueError(
            f"Export file allowlist violation: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )

    with (output_dir / "summary.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTED_SUMMARY_COLUMNS:
            raise ValueError("Unsafe or unexpected predicted summary columns")
        for row in reader:
            if row["label"] not in PREDICTED_SUMMARY_LABELS:
                raise ValueError("Uncontrolled label in predicted summary")
            for field in (
                "gold_spans",
                "end_to_end_valid",
                "end_to_end_failed",
                "fully_redacted",
                "residual_exposure",
            ):
                _validate_numeric(row[field], integer=True)
            for field in (
                "end_to_end_valid_rate",
                "end_to_end_failure_rate",
                "fully_redacted_rate",
                "residual_exposure_rate",
            ):
                _validate_numeric(row[field])

    with (output_dir / "failure_reasons.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PREDICTED_FAILURE_COLUMNS:
            raise ValueError("Unsafe or unexpected predicted failure columns")
        for row in reader:
            if row["label"] not in PREDICTED_SUMMARY_LABELS:
                raise ValueError("Uncontrolled label in predicted failures")
            if row["failure_reason"] not in PREDICTED_FAILURE_REASONS:
                raise ValueError("Uncontrolled predicted failure reason")
            _validate_numeric(row["count"], integer=True)
            _validate_numeric(row["fraction_of_label"])

    methodology = json.loads(
        (output_dir / "methodology.json").read_text(encoding="utf-8")
    )
    expected_keys = {
        "schema_version",
        "generated_at",
        "evaluation_scope",
        "prediction_source",
        "document_creation_date",
        "date_shift_days",
        "birthdate_replacement_mode",
        "contains_source_text",
        "contains_document_identifiers",
    }
    if set(methodology) != expected_keys:
        raise ValueError("Unsafe or unexpected predicted methodology fields")
    if methodology["evaluation_scope"] != "predicted_spans_end_to_end":
        raise ValueError("Unexpected predicted evaluation scope")
    if not _SAFE_SOURCE_ID_RE.fullmatch(str(methodology["prediction_source"])):
        raise ValueError("Unsafe prediction source id")
    datetime.fromisoformat(str(methodology["generated_at"]))
    date.fromisoformat(str(methodology["document_creation_date"]))
    if not isinstance(methodology["date_shift_days"], int) or isinstance(
        methodology["date_shift_days"], bool
    ):
        raise ValueError("date_shift_days must be an integer")
    if methodology["birthdate_replacement_mode"] != "age":
        raise ValueError("Unexpected birthdate replacement mode")
    if methodology["contains_source_text"] is not False:
        raise ValueError("Export claims to contain source text")
    if methodology["contains_document_identifiers"] is not False:
        raise ValueError("Export claims to contain document identifiers")
    return {
        "files": sorted(PREDICTED_EXPORT_FILES),
        "controlled_string_fields": {
            "label": list(PREDICTED_SUMMARY_LABELS),
            "failure_reason": list(PREDICTED_FAILURE_REASONS),
        },
    }


def write_predicted_safe_export(
    export_dir: str | Path,
    outcomes: Sequence[PredictedOutcome],
    settings: Any,
    *,
    prediction_source: str,
) -> dict[str, Any]:
    """Write aggregate-only end-to-end results and validate their privacy schema."""
    if not _SAFE_SOURCE_ID_RE.fullmatch(prediction_source):
        raise ValueError(
            "prediction source ids may contain only letters, digits, . _ @ + and -"
        )
    output_dir = Path(export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = existing - PREDICTED_EXPORT_FILES - PREDICTED_OPTIONAL_EXPORT_FILES
    if unexpected:
        raise ValueError(
            "Refusing to clean an export directory with unrelated files: "
            f"{sorted(unexpected)}"
        )
    for filename in PREDICTED_EXPORT_FILES | PREDICTED_OPTIONAL_EXPORT_FILES:
        (output_dir / filename).unlink(missing_ok=True)

    summary, failures = build_predicted_aggregate_tables(outcomes)
    _write_csv(output_dir / "summary.csv", summary, PREDICTED_SUMMARY_COLUMNS)
    _write_csv(
        output_dir / "failure_reasons.csv", failures, PREDICTED_FAILURE_COLUMNS
    )
    methodology = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_scope": "predicted_spans_end_to_end",
        "prediction_source": prediction_source,
        "document_creation_date": settings.document_creation_date,
        "date_shift_days": settings.date_shift_days,
        "birthdate_replacement_mode": settings.birthdate_replacement_mode,
        "contains_source_text": False,
        "contains_document_identifiers": False,
    }
    (output_dir / "methodology.json").write_text(
        json.dumps(methodology, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = validate_predicted_safe_export(output_dir)
    manifest = {
        "schema_version": 1,
        "privacy_check": "passed",
        "contains_source_text": False,
        "contains_document_identifiers": False,
        "files": report["files"],
        "controlled_string_fields": report["controlled_string_fields"],
    }
    (output_dir / "privacy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _load_predicted_spans(
    path: Path,
    *,
    document_texts: Mapping[str, str],
    gold_span_factory: Any,
    context_chars: int = 80,
) -> tuple[list[PredictedSpan], list[Any]]:
    """Load battery or canonical prediction JSONL and prepare date/age inputs."""
    predictions: list[PredictedSpan] = []
    transformation_inputs: list[Any] = []
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    for row in rows:
        document_id = str(row.get("document_id") or row.get("doc_id") or "").strip()
        if document_id not in document_texts:
            raise ValueError(f"Prediction document {document_id!r} is absent from input")
        text = document_texts[document_id]
        raw_spans = row.get("spans")
        if raw_spans is None:
            raw_spans = row.get("entities") or []
        for prediction_index, raw in enumerate(raw_spans):
            begin, end = int(raw["begin"]), int(raw["end"])
            if not 0 <= begin < end <= len(text):
                raise ValueError(
                    f"Prediction range outside document {document_id!r}: {begin}:{end}"
                )
            label = str(raw.get("label") or "").strip()
            prediction = PredictedSpan(
                document_id=document_id,
                prediction_index=prediction_index,
                label=label,
                begin=begin,
                end=end,
            )
            predictions.append(prediction)
            if label in EVALUATED_LABELS:
                transformation_inputs.append(
                    gold_span_factory(
                        document_id=document_id,
                        item_id=str(prediction_index),
                        label=label,
                        begin=begin,
                        end=end,
                        source_text=text[begin:end],
                        context_before=text[max(0, begin - context_chars) : begin],
                        context_after=text[end : min(len(text), end + context_chars)],
                    )
                )
    return predictions, transformation_inputs


def _run_predicted_sources(
    *,
    spec: Mapping[str, Any],
    battery_output: Path,
    output_dir: Path,
    texts: Mapping[str, str],
    gold_spans: Sequence[Any],
    settings: Any,
    api: Mapping[str, Any],
) -> list[Path]:
    exports: list[Path] = []
    for source in spec.get("predicted_sources") or []:
        source_id = str(source.get("id") or "").strip()
        raw_path = Path(str(source.get("predictions") or "")).expanduser()
        prediction_path = raw_path if raw_path.is_absolute() else battery_output / raw_path
        if not prediction_path.exists():
            print(
                f"  [{source_id or 'predicted source'}] skipped: output not found -> "
                f"{prediction_path}",
                flush=True,
            )
            continue
        predictions, transformation_inputs = _load_predicted_spans(
            prediction_path,
            document_texts=texts,
            gold_span_factory=api["GoldSpan"],
        )
        transformation_rows = api["evaluate_spans"](transformation_inputs, settings)
        transformation_results = {
            (row.document_id, int(row.item_id)): (
                row.protocol_valid,
                row.failure_reason,
            )
            for row in transformation_rows
        }
        outcomes = score_predicted_spans(
            gold_spans, predictions, transformation_results
        )
        export_dir = output_dir / "predicted" / source_id / "export"
        manifest = write_predicted_safe_export(
            export_dir, outcomes, settings, prediction_source=source_id
        )
        summary, _ = build_predicted_aggregate_tables(outcomes)
        overall = summary[0]
        print(f"\npredicted-span pseudonymization: {source_id}")
        print(
            "  end-to-end protocol-valid: "
            f"{overall['end_to_end_valid']}/{overall['gold_spans']} "
            f"({overall['end_to_end_valid_rate']:.1%})"
        )
        print(
            "  downstream failure rate: "
            f"{overall['end_to_end_failure_rate']:.1%}"
        )
        print(
            "  residual-exposure rate: "
            f"{overall['residual_exposure_rate']:.1%}"
        )
        print(f"  privacy-safe export -> {export_dir}")
        print(f"  privacy check: {manifest['privacy_check']}")
        exports.append(export_dir)
    return exports


def run(
    config: dict[str, Any],
    *,
    base_dir: Path,
    include_private_details: bool = False,
) -> Path | None:
    """Run the configured evaluation and return its safe export directory."""
    evaluation = config.get("evaluate") or {}
    spec = evaluation.get("pseudonymization") or {}
    if not spec.get("enabled", False):
        return None

    input_path = _resolve(str(config["input"]), base_dir)
    bundle = _resolve(str(evaluation["bundle"]), base_dir)
    battery_output = _resolve(str(config.get("output_dir", "out")), base_dir)
    configured_output = Path(str(spec.get("output_dir", "pseudonymization"))).expanduser()
    output_dir = (
        configured_output
        if configured_output.is_absolute()
        else battery_output / configured_output
    )

    api = _load_evaluator()
    settings = api["EvaluationSettings"](
        document_creation_date=str(spec["document_creation_date"]),
        date_shift_days=int(spec["date_shift_days"]),
        birthdate_replacement_mode=str(
            spec.get("birthdate_replacement_mode", "age")
        ),
    )
    texts = api["load_document_texts"](input_path)
    spans = api["load_gold_date_spans"](bundle, document_texts=texts)
    rows = api["evaluate_spans"](spans, settings)
    export_dir = output_dir / "export"
    manifest = api["write_safe_export"](export_dir, rows, settings)
    if include_private_details:
        private_path = api["write_private_details"](
            battery_output / "work" / "private" / "pseudonymization" / "details.jsonl",
            rows,
        )
        print(f"private details (contains PII; do not export) -> {private_path}")

    valid = sum(row.protocol_valid for row in rows)
    fraction = valid / len(rows) if rows else 0.0
    print("\npseudonymization evaluation")
    print(f"  gold Date/Age_Birthdate spans: {len(rows)}")
    print(f"  protocol-valid transformations: {valid}/{len(rows)} ({fraction:.1%})")
    print(f"  privacy-safe export -> {export_dir}")
    print(f"  privacy check: {manifest['privacy_check']}")
    _run_predicted_sources(
        spec=spec,
        battery_output=battery_output,
        output_dir=output_dir,
        texts=texts,
        gold_spans=spans,
        settings=settings,
        api=api,
    )
    return export_dir


def run_from_path(
    config_path: str | Path,
    *,
    include_private_details: bool = False,
) -> Path | None:
    path = Path(config_path).expanduser().resolve()
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("battery configuration must be a YAML mapping")
    # Battery paths are intentionally relative to the checkout/run directory.
    base_dir = path.parent.parent if path.parent.name == "configs" else Path.cwd()
    return run(
        config,
        base_dir=base_dir,
        include_private_details=include_private_details,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/battery.yaml")
    parser.add_argument(
        "--include-private-details",
        action="store_true",
        help="write PII-bearing details locally; never export this file",
    )
    args = parser.parse_args()
    result = run_from_path(
        args.config,
        include_private_details=args.include_private_details,
    )
    if result is None:
        raise SystemExit("pseudonymization evaluation is disabled in battery.yaml")


if __name__ == "__main__":
    main()
