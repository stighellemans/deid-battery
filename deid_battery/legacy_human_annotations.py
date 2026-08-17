"""Convert legacy per-document human annotations into battery JSONL outputs.

The legacy UZA layout stores one JSON array of spans per document under numbered
annotator batches, for example::

    annotations/llm_experiment/stig1/spans/<doc_id>.json
    annotations/llm_experiment/stig2/spans/<doc_id>.json
    annotations/llm_experiment/tomstroobants1/spans/<doc_id>.json

This module merges the numbered batches into one canonical ``by_doc`` JSONL
file per annotator.  The resulting files can be consumed as
``annotator-level-jsonl`` sources by deid-evaluation.

Human annotation files contain sensitive document identifiers and span text.
Keep the generated outputs on the approved evaluation machine.
"""
from __future__ import annotations

import argparse
import json
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from deid_schema.taxonomy import split_label

from .schema import make_span, read_jsonl, write_by_doc


DEFAULT_ANNOTATORS = ("stig", "tomstroobants")


class LegacyAnnotationError(ValueError):
    """Raised when legacy annotations cannot be converted without ambiguity."""


@dataclass(frozen=True)
class ConversionSummary:
    annotator: str
    batches: tuple[str, ...]
    documents: int
    spans: int
    identical_duplicate_documents: int
    output_path: Path


def _load_input_documents(path: Path) -> dict[str, str]:
    documents: dict[str, str] = {}
    for line_number, row in enumerate(read_jsonl(path), start=1):
        raw_doc_id = row.get("doc_id", row.get("document_id"))
        if raw_doc_id is None or str(raw_doc_id) == "":
            raise LegacyAnnotationError(f"{path}:{line_number}: missing doc_id/document_id")
        doc_id = str(raw_doc_id)
        if doc_id in documents:
            raise LegacyAnnotationError(f"{path}:{line_number}: duplicate doc_id {doc_id!r}")
        text = row.get("text", row.get("plain_text"))
        if not isinstance(text, str):
            raise LegacyAnnotationError(f"{path}:{line_number}: document {doc_id!r} has no text")
        documents[doc_id] = text
    if not documents:
        raise LegacyAnnotationError(f"{path}: no input documents found")
    return documents


def _discover_batches(source_root: Path, annotator: str) -> list[Path]:
    pattern = re.compile(rf"^{re.escape(annotator)}(?P<number>[1-9][0-9]*)$")
    matched: list[tuple[int, Path]] = []
    for candidate in source_root.iterdir():
        if not candidate.is_dir():
            continue
        match = pattern.fullmatch(candidate.name)
        if match and (candidate / "spans").is_dir():
            matched.append((int(match.group("number")), candidate))
    matched.sort(key=lambda item: item[0])
    if not matched:
        raise LegacyAnnotationError(
            f"{source_root}: found no {annotator}<number>/spans directories"
        )
    return [path for _, path in matched]


def _raw_span_list(path: Path) -> list[Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise LegacyAnnotationError(f"{path}: invalid JSON: {error}") from error
    if isinstance(payload, dict) and isinstance(payload.get("spans"), list):
        payload = payload["spans"]
    if not isinstance(payload, list):
        raise LegacyAnnotationError(f"{path}: expected a JSON array of spans")
    return payload


def _required_integer(span: dict[str, Any], field: str, source: Path, index: int) -> int:
    value = span.get(field)
    if isinstance(value, bool):
        raise LegacyAnnotationError(f"{source}: span {index} has invalid {field}={value!r}")
    try:
        integer = int(value)
    except (TypeError, ValueError) as error:
        raise LegacyAnnotationError(
            f"{source}: span {index} has invalid {field}={value!r}"
        ) from error
    if isinstance(value, float) and not value.is_integer():
        raise LegacyAnnotationError(f"{source}: span {index} has non-integral {field}={value!r}")
    return integer


def _normalize_span(
    raw: Any,
    *,
    document_text: str,
    source: Path,
    index: int,
) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise LegacyAnnotationError(f"{source}: span {index} is not an object")

    begin = _required_integer(raw, "begin", source, index)
    end = _required_integer(raw, "end", source, index)
    if begin < 0 or end <= begin or end > len(document_text):
        raise LegacyAnnotationError(
            f"{source}: span {index} has invalid range [{begin}, {end}) "
            f"for document length {len(document_text)}"
        )

    label_value = raw.get("label")
    if not isinstance(label_value, str) or not label_value.strip():
        raise LegacyAnnotationError(f"{source}: span {index} has no label")
    label = label_value.strip()
    category, subtype = split_label(label)

    legacy_category = raw.get("category", raw.get("Category"))
    if legacy_category not in (None, "") and str(legacy_category) != category:
        raise LegacyAnnotationError(
            f"{source}: span {index} label {label!r} implies category {category!r}, "
            f"but legacy category is {legacy_category!r}"
        )
    legacy_subtype = raw.get("subtype", raw.get("Subtype"))
    if legacy_subtype not in (None, "") and str(legacy_subtype) != subtype:
        raise LegacyAnnotationError(
            f"{source}: span {index} label {label!r} implies subtype {subtype!r}, "
            f"but legacy subtype is {legacy_subtype!r}"
        )

    expected_text = document_text[begin:end]
    legacy_text = raw.get("text")
    if not isinstance(legacy_text, str):
        raise LegacyAnnotationError(f"{source}: span {index} has no text")
    if legacy_text != expected_text:
        raise LegacyAnnotationError(
            f"{source}: span {index} text does not match input.jsonl at "
            f"[{begin}, {end}); expected {expected_text!r}, got {legacy_text!r}"
        )

    return make_span(begin, end, label, expected_text)


def _load_annotator(
    source_root: Path,
    annotator: str,
    input_documents: dict[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], tuple[str, ...], int]:
    batches = _discover_batches(source_root, annotator)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    duplicate_count = 0

    for batch in batches:
        for source in sorted((batch / "spans").glob("*.json")):
            doc_id = source.stem
            if doc_id not in input_documents:
                raise LegacyAnnotationError(
                    f"{source}: document id {doc_id!r} is absent from input.jsonl"
                )
            normalized = [
                _normalize_span(
                    raw,
                    document_text=input_documents[doc_id],
                    source=source,
                    index=index,
                )
                for index, raw in enumerate(_raw_span_list(source))
            ]
            normalized.sort(key=lambda span: (span["begin"], span["end"], span["label"]))

            if doc_id in by_doc:
                if by_doc[doc_id] != normalized:
                    raise LegacyAnnotationError(
                        f"{source}: conflicting duplicate document {doc_id!r} for {annotator}"
                    )
                duplicate_count += 1
                continue
            by_doc[doc_id] = normalized

    if not by_doc:
        raise LegacyAnnotationError(f"{source_root}: no JSON span files found for {annotator}")
    return by_doc, tuple(batch.name for batch in batches), duplicate_count


def _write_by_doc_atomic(path: Path, by_doc: dict[str, list[dict[str, Any]]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        write_by_doc(temporary_path, by_doc)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return path


def convert_legacy_human_annotations(
    source_root: str | Path,
    battery_input: str | Path,
    output_dir: str | Path,
    *,
    annotators: Iterable[str] = DEFAULT_ANNOTATORS,
    require_complete: bool = True,
) -> list[ConversionSummary]:
    """Convert numbered legacy annotator batches into canonical JSONL files."""
    source_root = Path(source_root).expanduser().resolve()
    battery_input = Path(battery_input).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise LegacyAnnotationError(f"source root not found: {source_root}")
    if not battery_input.is_file():
        raise LegacyAnnotationError(f"battery input not found: {battery_input}")

    input_documents = _load_input_documents(battery_input)
    selected = tuple(dict.fromkeys(str(value).strip() for value in annotators if str(value).strip()))
    if not selected:
        raise LegacyAnnotationError("at least one annotator is required")

    prepared: list[
        tuple[str, dict[str, list[dict[str, Any]]], tuple[str, ...], int]
    ] = []
    for annotator in selected:
        by_doc, batches, duplicate_count = _load_annotator(
            source_root, annotator, input_documents
        )
        if require_complete:
            missing = sorted(set(input_documents).difference(by_doc))
            if missing:
                preview = ", ".join(missing[:5])
                suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
                raise LegacyAnnotationError(
                    f"{annotator}: missing {len(missing)} of {len(input_documents)} input "
                    f"documents: {preview}{suffix}"
                )

        prepared.append((annotator, by_doc, batches, duplicate_count))

    output_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[ConversionSummary] = []
    for annotator, by_doc, batches, duplicate_count in prepared:
        ordered = {doc_id: by_doc[doc_id] for doc_id in sorted(by_doc)}
        output_path = _write_by_doc_atomic(output_dir / f"{annotator}.jsonl", ordered)
        summaries.append(
            ConversionSummary(
                annotator=annotator,
                batches=batches,
                documents=len(ordered),
                spans=sum(len(spans) for spans in ordered.values()),
                identical_duplicate_documents=duplicate_count,
                output_path=output_path,
            )
        )
    return summaries


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Folder containing stig1/, ..., tomstroobants1/, ...",
    )
    parser.add_argument(
        "--battery-input",
        type=Path,
        default=Path("input.jsonl"),
        help="Battery input used to validate document ids, offsets and span text",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("out/human-annotators"),
        help="Destination for stig.jsonl and tomstroobants.jsonl",
    )
    parser.add_argument(
        "--annotator",
        action="append",
        dest="annotators",
        help="Annotator folder prefix to import; repeat as needed (default: stig, tomstroobants)",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow an annotator to cover only part of input.jsonl",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    summaries = convert_legacy_human_annotations(
        args.source_root,
        args.battery_input,
        args.output_dir,
        annotators=args.annotators or DEFAULT_ANNOTATORS,
        require_complete=not args.allow_partial,
    )
    for summary in summaries:
        print(
            f"{summary.annotator}: {summary.documents} documents, {summary.spans} spans "
            f"from {', '.join(summary.batches)} -> {summary.output_path}"
        )
        if summary.identical_duplicate_documents:
            print(
                f"  deduplicated {summary.identical_duplicate_documents} identical "
                "document copies"
            )


if __name__ == "__main__":
    main()
