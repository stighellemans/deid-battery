"""Report missingness in legacy UZA human annotations without modifying them.

The report covers missing labels, absent input documents, empty annotation
files, malformed span records, missing required span fields, and duplicate
document files. Span text is omitted by default because it may contain PHI.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, TextIO

from .legacy_human_annotations import (
    DEFAULT_ANNOTATORS,
    LegacyAnnotationError,
    _discover_batches,
    _load_input_documents,
    _raw_label,
    _raw_span_list,
)


@dataclass(frozen=True)
class MissingnessInstance:
    kind: str
    annotator: str
    doc_id: str
    source_file: str = ""
    span_index: int | None = None
    missing_fields: tuple[str, ...] = ()
    begin: Any = None
    end: Any = None
    category: Any = None
    subtype: Any = None
    text: str | None = None
    detail: str = ""


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def scan_human_annotation_missingness(
    source_root: str | Path,
    battery_input: str | Path,
    *,
    annotators: Iterable[str] = DEFAULT_ANNOTATORS,
    include_text: bool = False,
) -> list[MissingnessInstance]:
    """Return every detected missingness or structural-audit instance."""
    source_root = Path(source_root).expanduser().resolve()
    battery_input = Path(battery_input).expanduser().resolve()
    if not source_root.is_dir():
        raise LegacyAnnotationError(f"source root not found: {source_root}")
    if not battery_input.is_file():
        raise LegacyAnnotationError(f"battery input not found: {battery_input}")

    input_documents = _load_input_documents(battery_input)
    selected = tuple(
        dict.fromkeys(str(value).strip() for value in annotators if str(value).strip())
    )
    if not selected:
        raise LegacyAnnotationError("at least one annotator is required")

    findings: list[MissingnessInstance] = []
    for annotator in selected:
        try:
            batches = _discover_batches(source_root, annotator)
        except LegacyAnnotationError as error:
            findings.append(
                MissingnessInstance(
                    kind="missing_annotator_batches",
                    annotator=annotator,
                    doc_id="",
                    detail=str(error),
                )
            )
            findings.extend(
                MissingnessInstance(
                    kind="missing_document", annotator=annotator, doc_id=doc_id
                )
                for doc_id in sorted(input_documents)
            )
            continue

        seen_documents: dict[str, str] = {}
        for batch in batches:
            for source in sorted((batch / "spans").glob("*.json")):
                doc_id = source.stem
                source_file = _relative(source, source_root)
                if doc_id in seen_documents:
                    findings.append(
                        MissingnessInstance(
                            kind="duplicate_document_file",
                            annotator=annotator,
                            doc_id=doc_id,
                            source_file=source_file,
                            detail=f"also present at {seen_documents[doc_id]}",
                        )
                    )
                else:
                    seen_documents[doc_id] = source_file

                if doc_id not in input_documents:
                    findings.append(
                        MissingnessInstance(
                            kind="unknown_document",
                            annotator=annotator,
                            doc_id=doc_id,
                            source_file=source_file,
                            detail="document is absent from battery input",
                        )
                    )

                try:
                    raw_spans = _raw_span_list(source)
                except LegacyAnnotationError as error:
                    findings.append(
                        MissingnessInstance(
                            kind="invalid_annotation_file",
                            annotator=annotator,
                            doc_id=doc_id,
                            source_file=source_file,
                            detail=str(error),
                        )
                    )
                    continue

                if not raw_spans:
                    findings.append(
                        MissingnessInstance(
                            kind="empty_annotation_file",
                            annotator=annotator,
                            doc_id=doc_id,
                            source_file=source_file,
                            detail=(
                                "may be a valid reviewed-negative document; "
                                "verify completion status"
                            ),
                        )
                    )

                for index, raw in enumerate(raw_spans):
                    if not isinstance(raw, dict):
                        findings.append(
                            MissingnessInstance(
                                kind="invalid_span_record",
                                annotator=annotator,
                                doc_id=doc_id,
                                source_file=source_file,
                                span_index=index,
                                detail=f"expected object, got {type(raw).__name__}",
                            )
                        )
                        continue

                    common = {
                        "annotator": annotator,
                        "doc_id": doc_id,
                        "source_file": source_file,
                        "span_index": index,
                        "begin": raw.get("begin"),
                        "end": raw.get("end"),
                        "category": raw.get("category", raw.get("Category")),
                        "subtype": raw.get("subtype", raw.get("Subtype")),
                        "text": (
                            str(raw.get("text"))
                            if include_text and raw.get("text") is not None
                            else None
                        ),
                    }
                    missing_fields = tuple(
                        field
                        for field in ("begin", "end", "text")
                        if raw.get(field) is None
                    )
                    if missing_fields:
                        findings.append(
                            MissingnessInstance(
                                kind="missing_required_fields",
                                missing_fields=missing_fields,
                                **common,
                            )
                        )
                    label = _raw_label(raw)
                    if not isinstance(label, str) or not label.strip():
                        findings.append(
                            MissingnessInstance(kind="missing_label", **common)
                        )

        findings.extend(
            MissingnessInstance(
                kind="missing_document", annotator=annotator, doc_id=doc_id
            )
            for doc_id in sorted(set(input_documents).difference(seen_documents))
        )

    return sorted(
        findings,
        key=lambda item: (
            item.annotator,
            item.doc_id,
            item.kind,
            item.source_file,
            -1 if item.span_index is None else item.span_index,
        ),
    )


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (tuple, list)):
        return ",".join(str(item) for item in value)
    return str(value).replace("\t", "\\t").replace("\n", "\\n")


def write_report(
    findings: list[MissingnessInstance],
    stream: TextIO,
    *,
    output_format: str,
) -> None:
    if output_format == "jsonl":
        for finding in findings:
            stream.write(json.dumps(asdict(finding), ensure_ascii=False) + "\n")
        return

    columns = (
        "kind",
        "annotator",
        "doc_id",
        "source_file",
        "span_index",
        "missing_fields",
        "begin",
        "end",
        "category",
        "subtype",
        "text",
        "detail",
    )
    stream.write("\t".join(columns) + "\n")
    for finding in findings:
        row = asdict(finding)
        stream.write("\t".join(_cell(row[column]) for column in columns) + "\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--battery-input", type=Path, default=Path("input.jsonl"))
    parser.add_argument(
        "--annotator",
        action="append",
        dest="annotators",
        help="Annotator folder prefix; repeat as needed (default: stig, tomstroobants)",
    )
    parser.add_argument("--format", choices=("tsv", "jsonl"), default="tsv")
    parser.add_argument(
        "--output", type=Path, help="Write report to this file instead of stdout"
    )
    parser.add_argument(
        "--include-text",
        action="store_true",
        help="Include source span text (PHI-sensitive; approved evaluation machine only)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    findings = scan_human_annotation_missingness(
        args.source_root,
        args.battery_input,
        annotators=args.annotators or DEFAULT_ANNOTATORS,
        include_text=args.include_text,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as stream:
            write_report(findings, stream, output_format=args.format)
        destination = str(args.output.resolve())
    else:
        write_report(findings, sys.stdout, output_format=args.format)
        destination = "stdout"

    counts = Counter(finding.kind for finding in findings)
    summary = ", ".join(
        f"{kind}={count}" for kind, count in sorted(counts.items())
    )
    print(
        f"missingness report: {len(findings)} finding(s) -> {destination}"
        + (f" ({summary})" if summary else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
