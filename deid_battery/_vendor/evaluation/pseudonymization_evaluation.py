"""Corpus-level evaluation of date and age pseudonymization.

This module deliberately evaluates the post-process pseudonymizer on *gold*
``Date`` and ``Age_Birthdate`` spans. Detection, boundary correction, and label
assignment are separate evaluations. A fixed document creation date and date
shift make the experiment deterministic and independent of source metadata.

Only aggregate, controlled-vocabulary results are written to ``export/``.
Optional span-level rows contain source clinical text and are confined to
``private/details.jsonl``.
"""
from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EVALUATED_LABELS = ("Date", "Age_Birthdate")

TRANSFORMATION_TYPES = (
    "exact_date_shift",
    "partial_day_month_shift",
    "month_interval_shift",
    "month_name_interval_shift",
    "month_phase_interval_shift",
    "month_range_shift",
    "season_interval_shift",
    "year_interval_shift",
    "date_range_shift",
    "contextual_date_component_transform",
    "explicit_age_to_band",
    "birthdate_to_age",
    "contextual_birth_component_scrub",
    "unclassified",
)

FAILURE_REASONS = (
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
)

SUMMARY_COLUMNS = (
    "label",
    "total",
    "transformed",
    "protocol_valid",
    "failed",
    "transformation_rate",
    "protocol_valid_rate",
)
TYPE_COLUMNS = (
    "label",
    "transformation_type",
    "total",
    "transformed",
    "protocol_valid",
    "failed",
    "protocol_valid_rate",
)
FAILURE_COLUMNS = (
    "label",
    "failure_reason",
    "count",
    "fraction_of_label",
)

ALLOWED_EXPORT_FILES = {
    "summary.csv",
    "transformation_by_type.csv",
    "failure_reasons.csv",
    "methodology.json",
}
OPTIONAL_EXPORT_FILES = {"privacy_manifest.json"}

AGE_UNIT_RE = re.compile(
    r"(?:dag|dagen|week|weken|maand|maanden|jaar|jaren|jarig|jarige|"
    r"day|days|week|weeks|month|months|year|years|"
    r"jr|yr|yrs|mnd|wk|wks|mo|mos|[jmwd])\b",
    re.IGNORECASE,
)
FOUR_DIGIT_YEAR_RE = re.compile(r"(?<!\d)[12]\d{3}(?!\d)")
APOSTROPHE_YEAR_RE = re.compile(r"['’]\s*\d{2}\b")
APPROXIMATE_AGE_RE = re.compile(
    r"(?:\b(?:ca\.?|circa|rond|bijna|ongeveer)\b|\+\s*/\s*-|±)",
    re.IGNORECASE,
)
TRAILING_DATE_PUNCTUATION_RE = re.compile(r"\d[/.]\s*$")
MIXED_RANGE_RE = re.compile(r"\b(?:tot|t/m)\b", re.IGNORECASE)


@dataclass(frozen=True)
class GoldSpan:
    document_id: str
    item_id: str
    label: str
    begin: int
    end: int
    source_text: str
    context_before: str = ""
    context_after: str = ""


@dataclass(frozen=True)
class EvaluationSettings:
    document_creation_date: str
    date_shift_days: int
    birthdate_replacement_mode: str = "age"

    def validated_document_date(self) -> date:
        try:
            parsed = date.fromisoformat(self.document_creation_date)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "document_creation_date must be an ISO date (YYYY-MM-DD)"
            ) from error
        if self.birthdate_replacement_mode != "age":
            raise ValueError(
                "Corpus evaluation requires birthdate_replacement_mode='age'"
            )
        if not isinstance(self.date_shift_days, int) or isinstance(
            self.date_shift_days, bool
        ):
            raise ValueError("date_shift_days must be an integer")
        if self.date_shift_days == 0:
            raise ValueError("date_shift_days must be non-zero")
        return parsed


@dataclass(frozen=True)
class EvaluationRow:
    document_id: str
    item_id: str
    label: str
    begin: int
    end: int
    source_text: str
    substitute: str | None
    transformation_type: str
    transformed: bool
    protocol_valid: bool
    failure_reason: str | None


def _load_pseudonymizer_module():
    try:
        import date_pseudonyms  # type: ignore
    except ImportError as error:
        raise ImportError(
            "The pseudonymization evaluation requires deid-post-process. "
            "Install it with `uv pip install -e ../post-process` or install a "
            "pinned git revision."
        ) from error
    return date_pseudonyms


def _iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_document_texts(path: str | Path) -> dict[str, str]:
    """Load only document id and text; metadata and embedded spans are ignored."""
    texts: dict[str, str] = {}
    for row in _iter_jsonl(path):
        raw_id = (
            row.get("document_id")
            or row.get("doc_id")
            or row.get("documentId")
            or row.get("id")
        )
        raw_text = (
            row.get("text")
            if isinstance(row.get("text"), str)
            else row.get("plain_text")
        )
        if raw_id is None or not isinstance(raw_text, str):
            continue
        document_id = str(raw_id).strip()
        if document_id:
            texts[document_id] = raw_text
    if not texts:
        raise ValueError(f"No usable document text found in {path}")
    return texts


def _range(value: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not value:
        return None
    try:
        begin, end = int(value.get("begin")), int(value.get("end"))
    except (TypeError, ValueError):
        return None
    return (begin, end) if begin < end else None


def load_gold_date_spans(
    bundle_dir: str | Path,
    *,
    document_texts: Mapping[str, str] | None = None,
    context_chars: int = 80,
) -> list[GoldSpan]:
    """Load gold date/age items from an exported evaluation bundle."""
    reference_path = Path(bundle_dir) / "reference_items.jsonl"
    if not reference_path.exists():
        raise FileNotFoundError(reference_path)

    spans: list[GoldSpan] = []
    for item in _iter_jsonl(reference_path):
        gold = item.get("gold")
        if not isinstance(gold, dict):
            continue
        label = str(gold.get("label") or "").strip()
        if label not in EVALUATED_LABELS:
            continue
        span_range = _range(gold)
        if span_range is None:
            raise ValueError(f"Invalid gold range in item {item.get('item_id')!r}")
        begin, end = span_range
        document_id = str(item.get("document_id") or "").strip()
        item_id = str(item.get("item_id") or "").strip()
        source_text = str(gold.get("text") or "")
        before = after = ""

        full_text = document_texts.get(document_id) if document_texts else None
        if full_text is not None:
            if 0 <= begin < end <= len(full_text):
                sliced = full_text[begin:end]
                if source_text and sliced != source_text:
                    raise ValueError(
                        f"Gold text does not match document text for item {item_id!r}"
                    )
                source_text = sliced
                before = full_text[max(0, begin - context_chars) : begin]
                after = full_text[end : min(len(full_text), end + context_chars)]
            else:
                raise ValueError(f"Gold range outside document for item {item_id!r}")
        else:
            review_range = _range(item.get("review_range"))
            review_text = item.get("review_text")
            if review_range and isinstance(review_text, str):
                local_begin = begin - review_range[0]
                local_end = end - review_range[0]
                if 0 <= local_begin < local_end <= len(review_text):
                    before = review_text[max(0, local_begin - context_chars) : local_begin]
                    after = review_text[
                        local_end : min(len(review_text), local_end + context_chars)
                    ]

        if not source_text:
            raise ValueError(f"Missing gold text for item {item_id!r}")
        spans.append(
            GoldSpan(
                document_id=document_id,
                item_id=item_id,
                label=label,
                begin=begin,
                end=end,
                source_text=source_text,
                context_before=before,
                context_after=after,
            )
        )
    return spans


def _strip_brackets(value: str) -> str:
    stripped = value.strip()
    return stripped[1:-1].strip() if stripped.startswith("[") and stripped.endswith("]") else stripped


def _is_bracketed(value: str) -> bool:
    stripped = value.strip()
    return len(stripped) >= 2 and stripped.startswith("[") and stripped.endswith("]")


def _classify_transformation(
    span: GoldSpan,
    *,
    module: Any,
    document_date: date,
    transformed: bool,
) -> tuple[str, Any | None, Any | None]:
    age_expression = None
    parsed_date = None
    if span.label == "Age_Birthdate":
        try:
            age_expression = module.parse_age_expression(span.source_text)
        except (TypeError, ValueError):
            age_expression = None
        if age_expression is not None:
            return "explicit_age_to_band", age_expression, None
        try:
            parsed_date = module.parse_date_text(
                span.source_text, document_date=document_date
            )
        except (TypeError, ValueError):
            parsed_date = None
        if parsed_date is not None:
            return "birthdate_to_age", None, parsed_date
        if transformed:
            return "contextual_birth_component_scrub", None, None
        return "unclassified", None, None

    try:
        parsed_date = module.parse_date_text(
            span.source_text, document_date=document_date
        )
    except (TypeError, ValueError):
        parsed_date = None
    if parsed_date is None:
        return (
            "contextual_date_component_transform" if transformed else "unclassified",
            None,
            None,
        )
    type_by_granularity = {
        "day": "exact_date_shift",
        "day_month": "partial_day_month_shift",
        "month": "month_interval_shift",
        "month_name": "month_name_interval_shift",
        "month_phase": "month_phase_interval_shift",
        "month_range": "month_range_shift",
        "season": "season_interval_shift",
        "year": "year_interval_shift",
        "range": "date_range_shift",
    }
    return type_by_granularity.get(parsed_date.granularity, "unclassified"), None, parsed_date


def _unsupported_reason(span: GoldSpan) -> str:
    value = span.source_text
    if APOSTROPHE_YEAR_RE.search(value):
        return "unsupported_apostrophe_year"
    if span.label == "Age_Birthdate" and APPROXIMATE_AGE_RE.search(value):
        return "unsupported_approximate_age"
    if TRAILING_DATE_PUNCTUATION_RE.search(value):
        return "unsupported_trailing_punctuation"
    if MIXED_RANGE_RE.search(value):
        return "unsupported_mixed_format_range"
    return "unsupported_or_invalid_format"


def _validate_substitution(
    span: GoldSpan,
    substitute: str | None,
    transformation_type: str,
    *,
    parsed_date: Any | None,
    document_date: date,
    date_shift_days: int,
    module: Any,
) -> tuple[bool, str | None]:
    if substitute is None:
        return False, _unsupported_reason(span)
    if not substitute.strip():
        return False, "empty_substitution"
    if not _is_bracketed(substitute):
        return False, "output_not_bracketed"

    body = _strip_brackets(substitute)
    if transformation_type == "exact_date_shift" and parsed_date is not None:
        try:
            reparsed = module.parse_date_text(body, document_date=document_date)
        except (TypeError, ValueError):
            return False, "exact_date_shift_mismatch"
        expected_start = parsed_date.start + timedelta(days=date_shift_days)
        expected_end = parsed_date.end + timedelta(days=date_shift_days)
        if reparsed.start != expected_start or reparsed.end != expected_end:
            return False, "exact_date_shift_mismatch"

    if transformation_type == "birthdate_to_age":
        if FOUR_DIGIT_YEAR_RE.search(body) or not AGE_UNIT_RE.search(body):
            return False, "birthdate_not_reduced_to_age"

    if transformation_type == "explicit_age_to_band":
        if not any(char.isdigit() for char in body) or not AGE_UNIT_RE.search(body):
            return False, "age_output_not_age_like"

    return True, None


def evaluate_spans(
    spans: Sequence[GoldSpan],
    settings: EvaluationSettings,
) -> list[EvaluationRow]:
    """Evaluate already-correct gold spans with fixed experimental metadata."""
    document_date = settings.validated_document_date()
    module = _load_pseudonymizer_module()
    rows: list[EvaluationRow] = []

    for span in spans:
        substitute: str | None = None
        exception = False
        try:
            substitute = module.pseudonymize_date_text(
                span.source_text,
                label=span.label,
                date_shift_days=settings.date_shift_days,
                context_before=span.context_before,
                context_after=span.context_after,
                document_creation_date=settings.document_creation_date,
                birthdate_replacement_mode=settings.birthdate_replacement_mode,
            )
        except Exception:  # noqa: BLE001 - an evaluation row must survive one bad span
            exception = True

        transformed = substitute is not None and bool(substitute.strip())
        transformation_type, _age_expression, parsed_date = _classify_transformation(
            span,
            module=module,
            document_date=document_date,
            transformed=transformed,
        )
        if exception:
            valid, failure_reason = False, "pseudonymizer_exception"
        else:
            valid, failure_reason = _validate_substitution(
                span,
                substitute,
                transformation_type,
                parsed_date=parsed_date,
                document_date=document_date,
                date_shift_days=settings.date_shift_days,
                module=module,
            )
        rows.append(
            EvaluationRow(
                document_id=span.document_id,
                item_id=span.item_id,
                label=span.label,
                begin=span.begin,
                end=span.end,
                source_text=span.source_text,
                substitute=substitute,
                transformation_type=transformation_type,
                transformed=transformed,
                protocol_valid=valid,
                failure_reason=failure_reason,
            )
        )
    return rows


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def build_aggregate_tables(
    rows: Sequence[EvaluationRow],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    by_label: dict[str, list[EvaluationRow]] = defaultdict(list)
    by_type: dict[tuple[str, str], list[EvaluationRow]] = defaultdict(list)
    failure_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        by_label[row.label].append(row)
        by_type[(row.label, row.transformation_type)].append(row)
        if row.failure_reason:
            failure_counts[(row.label, row.failure_reason)] += 1

    summary: list[dict[str, Any]] = []
    for label in EVALUATED_LABELS:
        group = by_label.get(label, [])
        transformed = sum(row.transformed for row in group)
        valid = sum(row.protocol_valid for row in group)
        total = len(group)
        summary.append(
            {
                "label": label,
                "total": total,
                "transformed": transformed,
                "protocol_valid": valid,
                "failed": total - valid,
                "transformation_rate": _fraction(transformed, total),
                "protocol_valid_rate": _fraction(valid, total),
            }
        )

    type_table: list[dict[str, Any]] = []
    for label in EVALUATED_LABELS:
        for transformation_type in TRANSFORMATION_TYPES:
            group = by_type.get((label, transformation_type), [])
            if not group:
                continue
            transformed = sum(row.transformed for row in group)
            valid = sum(row.protocol_valid for row in group)
            total = len(group)
            type_table.append(
                {
                    "label": label,
                    "transformation_type": transformation_type,
                    "total": total,
                    "transformed": transformed,
                    "protocol_valid": valid,
                    "failed": total - valid,
                    "protocol_valid_rate": _fraction(valid, total),
                }
            )

    failure_table: list[dict[str, Any]] = []
    for label in EVALUATED_LABELS:
        denominator = len(by_label.get(label, []))
        for reason in FAILURE_REASONS:
            count = failure_counts[(label, reason)]
            if count:
                failure_table.append(
                    {
                        "label": label,
                        "failure_reason": reason,
                        "count": count,
                        "fraction_of_label": _fraction(count, denominator),
                    }
                )
    return summary, type_table, failure_table


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def _post_process_version() -> str:
    try:
        value = importlib.metadata.version("deid-post-process")
    except importlib.metadata.PackageNotFoundError:
        module_path = Path(_load_pseudonymizer_module().__file__).resolve()
        pyproject_path = module_path.parent / "pyproject.toml"
        if not pyproject_path.exists():
            return "unavailable"
        match = re.search(
            r'^version\s*=\s*"(?P<version>[A-Za-z0-9_.+!-]+)"\s*$',
            pyproject_path.read_text(encoding="utf-8"),
            re.MULTILINE,
        )
        return match.group("version") if match else "unavailable"
    return value if re.fullmatch(r"[A-Za-z0-9_.+!-]+", value) else "unavailable"


def write_private_details(path: str | Path, rows: Sequence[EvaluationRow]) -> Path:
    """Write PII-bearing row details. This output must never be exported."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")
    return output_path


def write_safe_export(
    export_dir: str | Path,
    rows: Sequence[EvaluationRow],
    settings: EvaluationSettings,
) -> dict[str, Any]:
    """Write aggregate-only artifacts and verify their schema before returning."""
    output_dir = Path(export_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected_existing = existing_files - ALLOWED_EXPORT_FILES - OPTIONAL_EXPORT_FILES
    if unexpected_existing:
        raise ValueError(
            "Refusing to clean an export directory with unrelated files: "
            f"{sorted(unexpected_existing)}"
        )
    for filename in ALLOWED_EXPORT_FILES | OPTIONAL_EXPORT_FILES:
        (output_dir / filename).unlink(missing_ok=True)

    summary, type_table, failure_table = build_aggregate_tables(rows)
    _write_csv(output_dir / "summary.csv", summary, SUMMARY_COLUMNS)
    _write_csv(
        output_dir / "transformation_by_type.csv", type_table, TYPE_COLUMNS
    )
    _write_csv(
        output_dir / "failure_reasons.csv", failure_table, FAILURE_COLUMNS
    )
    methodology = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "evaluation_scope": "gold_date_and_age_spans_only",
        "document_creation_date": settings.document_creation_date,
        "date_shift_days": settings.date_shift_days,
        "birthdate_replacement_mode": settings.birthdate_replacement_mode,
        "post_process_version": _post_process_version(),
        "contains_source_text": False,
        "contains_document_identifiers": False,
    }
    (output_dir / "methodology.json").write_text(
        json.dumps(methodology, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = validate_safe_export(output_dir)
    manifest = {
        "schema_version": 1,
        "privacy_check": "passed",
        "contains_source_text": False,
        "contains_document_identifiers": False,
        "files": report["files"],
        "controlled_string_fields": report["controlled_string_fields"],
    }
    (output_dir / "privacy_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _validate_numeric(value: str, *, integer: bool = False) -> None:
    try:
        number = int(value) if integer else float(value)
    except ValueError as error:
        raise ValueError(f"Expected numeric aggregate value, got {value!r}") from error
    if not integer and (math.isnan(number) or math.isinf(number)):
        raise ValueError("Aggregate values must be finite")


def _read_and_validate_csv(
    path: Path,
    *,
    expected_columns: Sequence[str],
    controlled_fields: Mapping[str, set[str]],
    integer_fields: set[str],
    float_fields: set[str],
) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(
                f"Unsafe or unexpected columns in {path.name}: {reader.fieldnames}"
            )
        for row in reader:
            for field, allowed in controlled_fields.items():
                if row[field] not in allowed:
                    raise ValueError(
                        f"Uncontrolled string in {path.name}.{field}: {row[field]!r}"
                    )
            for field in integer_fields:
                _validate_numeric(row[field], integer=True)
            for field in float_fields:
                _validate_numeric(row[field])


def validate_safe_export(export_dir: str | Path) -> dict[str, Any]:
    """Fail closed if an export contains unexpected files, fields, or strings."""
    output_dir = Path(export_dir)
    actual_files = {path.name for path in output_dir.iterdir() if path.is_file()}
    unexpected = actual_files - ALLOWED_EXPORT_FILES - OPTIONAL_EXPORT_FILES
    missing = ALLOWED_EXPORT_FILES - actual_files
    if unexpected or missing:
        raise ValueError(
            f"Export file allowlist violation: unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )

    controlled = {
        "label": set(EVALUATED_LABELS),
        "transformation_type": set(TRANSFORMATION_TYPES),
        "failure_reason": set(FAILURE_REASONS),
    }
    _read_and_validate_csv(
        output_dir / "summary.csv",
        expected_columns=SUMMARY_COLUMNS,
        controlled_fields={"label": controlled["label"]},
        integer_fields={"total", "transformed", "protocol_valid", "failed"},
        float_fields={"transformation_rate", "protocol_valid_rate"},
    )
    _read_and_validate_csv(
        output_dir / "transformation_by_type.csv",
        expected_columns=TYPE_COLUMNS,
        controlled_fields={
            "label": controlled["label"],
            "transformation_type": controlled["transformation_type"],
        },
        integer_fields={"total", "transformed", "protocol_valid", "failed"},
        float_fields={"protocol_valid_rate"},
    )
    _read_and_validate_csv(
        output_dir / "failure_reasons.csv",
        expected_columns=FAILURE_COLUMNS,
        controlled_fields={
            "label": controlled["label"],
            "failure_reason": controlled["failure_reason"],
        },
        integer_fields={"count"},
        float_fields={"fraction_of_label"},
    )

    methodology = json.loads(
        (output_dir / "methodology.json").read_text(encoding="utf-8")
    )
    expected_methodology_keys = {
        "schema_version",
        "generated_at",
        "evaluation_scope",
        "document_creation_date",
        "date_shift_days",
        "birthdate_replacement_mode",
        "post_process_version",
        "contains_source_text",
        "contains_document_identifiers",
    }
    if set(methodology) != expected_methodology_keys:
        raise ValueError("Unsafe or unexpected methodology fields")
    if methodology["evaluation_scope"] != "gold_date_and_age_spans_only":
        raise ValueError("Unexpected evaluation scope")
    datetime.fromisoformat(str(methodology["generated_at"]))
    if methodology["birthdate_replacement_mode"] != "age":
        raise ValueError("Unexpected birthdate replacement mode")
    date.fromisoformat(str(methodology["document_creation_date"]))
    if not isinstance(methodology["date_shift_days"], int):
        raise ValueError("date_shift_days must be an integer")
    if methodology["contains_source_text"] is not False:
        raise ValueError("Export claims to contain source text")
    if methodology["contains_document_identifiers"] is not False:
        raise ValueError("Export claims to contain document identifiers")
    if not re.fullmatch(
        r"[A-Za-z0-9_.+!-]+", str(methodology["post_process_version"])
    ):
        raise ValueError("Unexpected post-process version string")

    manifest_path = output_dir / "privacy_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_manifest_keys = {
            "schema_version",
            "privacy_check",
            "contains_source_text",
            "contains_document_identifiers",
            "files",
            "controlled_string_fields",
        }
        if set(manifest) != expected_manifest_keys:
            raise ValueError("Unsafe or unexpected privacy-manifest fields")
        if manifest["privacy_check"] != "passed":
            raise ValueError("Privacy manifest did not pass")
        if manifest["contains_source_text"] is not False:
            raise ValueError("Privacy manifest permits source text")
        if manifest["contains_document_identifiers"] is not False:
            raise ValueError("Privacy manifest permits document identifiers")

    return {
        "files": sorted(ALLOWED_EXPORT_FILES),
        "controlled_string_fields": {
            key: sorted(values) for key, values in controlled.items()
        },
    }
