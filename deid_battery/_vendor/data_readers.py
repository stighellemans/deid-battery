from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from deid_schema.normalize import normalize_span


REQUIRED_ANNOTATION_FIELDS = ("begin", "end", "label", "category", "subtype", "text")


@dataclass(frozen=True)
class ValidationIssue:
    source: str
    doc_id: str | None
    annotation_id: str | None
    annotation_index: int | None
    field: str | None
    message: str


def read_annotator_level_jsonl(
    path: str | Path,
    *,
    annotation_id: str | None = None,
    source_name: str | None = None,
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    """Read one JSONL file with records shaped like {doc_id, entities}.

    Returns records shaped like:
        {
            "doc_id": "...",
            "annotation_id": "source-file.jsonl",
            "annotations": [...]
        }
    """
    source_path = Path(path)
    source_annotation_id = annotation_id or source_path.name
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []

    for line_number, item in _iter_jsonl(source_path):
        doc_id = str(item.get("doc_id", ""))
        if not doc_id:
            issues.append(
                ValidationIssue(
                    source=str(source_path),
                    doc_id=None,
                    annotation_id=source_annotation_id,
                    annotation_index=None,
                    field="doc_id",
                    message=f"Missing doc_id on line {line_number}",
                )
            )

        raw_annotations = item.get("entities", [])
        if not isinstance(raw_annotations, list):
            issues.append(
                ValidationIssue(
                    source=str(source_path),
                    doc_id=doc_id or None,
                    annotation_id=source_annotation_id,
                    annotation_index=None,
                    field="entities",
                    message=f"Expected entities to be a list on line {line_number}",
                )
            )
            raw_annotations = []

        annotations, annotation_issues = normalize_annotations(
            raw_annotations,
            source=str(source_path),
            doc_id=doc_id or None,
            annotation_id=source_annotation_id,
        )
        issues.extend(annotation_issues)

        records.append(
            {
                "doc_id": doc_id,
                "annotation_id": source_annotation_id,
                **({"source_name": source_name} if source_name else {}),
                "annotations": annotations,
            }
        )

    return records, issues


def read_document_level_jsonl_folder(
    folder: str | Path,
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    """Read a folder of per-document JSONL files.

    Each file name stem is used as doc_id. Each JSONL row is expected to contain
    annotation_id and annotations.
    """
    folder_path = Path(folder)
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []

    for source_path in sorted(folder_path.glob("*.jsonl")):
        doc_id = source_path.stem

        for line_number, item in _iter_jsonl(source_path):
            annotation_id = item.get("annotation_id")
            if annotation_id is None:
                issues.append(
                    ValidationIssue(
                        source=str(source_path),
                        doc_id=doc_id,
                        annotation_id=None,
                        annotation_index=None,
                        field="annotation_id",
                        message=f"Missing annotation_id on line {line_number}",
                    )
                )
                annotation_id = source_path.name
            else:
                annotation_id = str(annotation_id)

            raw_annotations = item.get("annotations", [])
            if not isinstance(raw_annotations, list):
                issues.append(
                    ValidationIssue(
                        source=str(source_path),
                        doc_id=doc_id,
                        annotation_id=annotation_id,
                        annotation_index=None,
                        field="annotations",
                        message=f"Expected annotations to be a list on line {line_number}",
                    )
                )
                raw_annotations = []

            annotations, annotation_issues = normalize_annotations(
                raw_annotations,
                source=str(source_path),
                doc_id=doc_id,
                annotation_id=annotation_id,
            )
            issues.extend(annotation_issues)

            records.append(
                {
                    "doc_id": doc_id,
                    "annotation_id": annotation_id,
                    **({"source_name": str(item["source_name"])} if item.get("source_name") else {}),
                    "annotations": annotations,
                }
            )

    return records, issues


def read_annotation_id_jsonl_folder(
    folder: str | Path = "annotations",
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    """Read a folder where each annotation_id has its own JSONL file.

    File names are interpreted as annotation ids:
        annotations/stig.jsonl -> annotation_id == "stig"

    Each JSONL row should contain doc_id plus either annotations or entities.
    """
    folder_path = Path(folder)
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []

    for source_path in sorted(folder_path.glob("*.jsonl")):
        annotation_id = source_path.stem

        for line_number, item in _iter_jsonl(source_path):
            doc_id = str(item.get("doc_id", ""))
            if not doc_id:
                issues.append(
                    ValidationIssue(
                        source=str(source_path),
                        doc_id=None,
                        annotation_id=annotation_id,
                        annotation_index=None,
                        field="doc_id",
                        message=f"Missing doc_id on line {line_number}",
                    )
                )

            raw_annotations = item.get("annotations", item.get("entities", []))
            if not isinstance(raw_annotations, list):
                issues.append(
                    ValidationIssue(
                        source=str(source_path),
                        doc_id=doc_id or None,
                        annotation_id=annotation_id,
                        annotation_index=None,
                        field="annotations",
                        message=f"Expected annotations/entities to be a list on line {line_number}",
                    )
                )
                raw_annotations = []

            annotations, annotation_issues = normalize_annotations(
                raw_annotations,
                source=str(source_path),
                doc_id=doc_id or None,
                annotation_id=annotation_id,
            )
            issues.extend(annotation_issues)

            records.append(
                {
                    "doc_id": doc_id,
                    "annotation_id": annotation_id,
                    **({"source_name": str(item["source_name"])} if item.get("source_name") else {}),
                    "annotations": annotations,
                }
            )

    return records, issues


def write_annotation_id_jsonl_folder(
    records: list[dict[str, Any]],
    folder: str | Path = "annotations",
    *,
    append: bool = False,
) -> list[Path]:
    """Write normalized records into one JSONL file per annotation_id.

    Rows are written as:
        {"doc_id": "...", "annotations": [...]}

    The annotation_id itself is stored in the file name:
        annotations/stig.jsonl
    """
    folder_path = Path(folder)
    folder_path.mkdir(parents=True, exist_ok=True)

    grouped_records: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        annotation_id = record.get("annotation_id")
        if not annotation_id:
            raise ValueError(f"Record is missing annotation_id: {record!r}")

        output_record = {
            "doc_id": record.get("doc_id", ""),
            "annotations": record.get("annotations", []),
        }
        if record.get("source_name"):
            output_record["source_name"] = str(record["source_name"])
        grouped_records.setdefault(str(annotation_id), []).append(output_record)

    written_paths: list[Path] = []
    mode = "a" if append else "w"

    for annotation_id, annotation_records in sorted(grouped_records.items()):
        output_path = _annotation_id_jsonl_path(folder_path, annotation_id)
        with output_path.open(mode, encoding="utf-8") as file:
            for record in annotation_records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        written_paths.append(output_path)

    return written_paths


def convert_annotator_level_jsonl_to_annotation_folder(
    path: str | Path,
    folder: str | Path = "annotations",
    *,
    annotation_id: str | None = None,
    append: bool = False,
) -> tuple[list[Path], list[ValidationIssue]]:
    """Convert one annotator-level JSONL file into annotations/{annotation_id}.jsonl."""
    source_path = Path(path)
    records, issues = read_annotator_level_jsonl(
        source_path,
        annotation_id=annotation_id or source_path.stem,
    )
    written_paths = write_annotation_id_jsonl_folder(records, folder, append=append)
    return written_paths, issues


def convert_document_level_jsonl_folder_to_annotation_folder(
    source_folder: str | Path,
    folder: str | Path = "annotations",
    *,
    append: bool = False,
) -> tuple[list[Path], list[ValidationIssue]]:
    """Convert per-document JSONL files into one annotations/{annotation_id}.jsonl per id."""
    records, issues = read_document_level_jsonl_folder(source_folder)
    written_paths = write_annotation_id_jsonl_folder(records, folder, append=append)
    return written_paths, issues


def load_annotations_from_yaml(
    config_path: str | Path = "config.yaml",
    *,
    output_folder: str | Path | None = None,
    append: bool = False,
) -> tuple[list[Path], list[ValidationIssue]]:
    """Load all annotation sources from YAML and write annotations/{annotation_id}.jsonl.

    Expected YAML shape:
        output_folder: annotations
        clean_output: true
        annotations:
          stig:
            name: Annotator 1
            method: document-level-jsonl-folder
            path: /path/to/document-level-folder
          openai-pii-filter:
            method: annotator-level-jsonl
            path: /path/to/predicted_entities_by_doc.jsonl
          old-source:
            exclude: true
            method: document-level-jsonl-folder
            path: /path/to/document-level-folder
    """
    config = read_yaml_config(config_path)
    annotation_config = config.get("annotations", config)
    if not isinstance(annotation_config, dict):
        raise ValueError("YAML config must contain an 'annotations' mapping")

    target_folder = output_folder or config.get("output_folder", "annotations")
    records: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []
    document_folder_cache: dict[str, tuple[list[dict[str, Any]], list[ValidationIssue]]] = {}
    annotation_folder_cache: dict[str, tuple[list[dict[str, Any]], list[ValidationIssue]]] = {}

    for annotation_id, source_config in annotation_config.items():
        if not isinstance(source_config, dict):
            raise ValueError(f"Config for annotation_id {annotation_id!r} must be a mapping")
        if _source_excluded(source_config):
            continue

        method = source_config.get("method")
        source_path = source_config.get("path")
        if not method or not source_path:
            raise ValueError(f"Config for annotation_id {annotation_id!r} needs method and path")

        annotation_id = str(annotation_id)
        source_name = _source_name_from_config(source_config)
        source_path = str(source_path)
        found_records: list[dict[str, Any]]

        if method == "annotator-level-jsonl":
            source_records, source_issues = read_annotator_level_jsonl(
                source_path,
                annotation_id=annotation_id,
                source_name=source_name,
            )
            found_records = source_records
            issues.extend(source_issues)
        elif method == "document-level-jsonl-folder":
            if source_path not in document_folder_cache:
                document_folder_cache[source_path] = read_document_level_jsonl_folder(source_path)
                issues.extend(document_folder_cache[source_path][1])
            source_records = document_folder_cache[source_path][0]
            found_records = _records_for_annotation_id(source_records, annotation_id)
        elif method in {"annotation-folder", "annotation-id-jsonl-folder"}:
            if source_path not in annotation_folder_cache:
                annotation_folder_cache[source_path] = read_annotation_id_jsonl_folder(source_path)
                issues.extend(annotation_folder_cache[source_path][1])
            source_records = annotation_folder_cache[source_path][0]
            found_records = _records_for_annotation_id(source_records, annotation_id)
        elif method == "generate":
            raise NotImplementedError("The 'generate' import method is not implemented yet")
        else:
            raise ValueError(f"Unknown import method for {annotation_id!r}: {method!r}")

        records.extend(_records_with_source_name(found_records, source_name))
        if not found_records:
            issues.append(
                ValidationIssue(
                    source=source_path,
                    doc_id=None,
                    annotation_id=annotation_id,
                    annotation_index=None,
                    field="annotation_id",
                    message="No records found for configured annotation_id",
                )
            )

    if not append and config.get("clean_output", False):
        _clean_annotation_output_folder(target_folder)

    written_paths = write_annotation_id_jsonl_folder(records, target_folder, append=append)
    return written_paths, issues


def source_names_from_yaml(config_path: str | Path = "config.yaml") -> dict[str, str]:
    config = read_yaml_config(config_path)
    annotation_config = config.get("annotations", config)
    if not isinstance(annotation_config, dict):
        raise ValueError("YAML config must contain an 'annotations' mapping")

    source_names: dict[str, str] = {}
    for annotation_id, source_config in annotation_config.items():
        if isinstance(source_config, dict):
            if _source_excluded(source_config):
                continue
            source_name = _source_name_from_config(source_config)
            if source_name:
                source_names[str(annotation_id)] = source_name
    return source_names


def annotation_order_from_yaml(config_path: str | Path = "config.yaml") -> list[str]:
    config = read_yaml_config(config_path)
    annotation_config = config.get("annotations", config)
    if not isinstance(annotation_config, dict):
        raise ValueError("YAML config must contain an 'annotations' mapping")

    return [
        str(annotation_id)
        for annotation_id, source_config in annotation_config.items()
        if isinstance(source_config, dict) and not _source_excluded(source_config)
    ]


def attach_source_names_to_payload(
    payload: dict[str, Any],
    source_names: dict[str, str],
    source_order: list[str] | None = None,
) -> dict[str, Any]:
    if source_names:
        payload["source_names"] = dict(source_names)
    if source_order is not None:
        payload["source_order"] = [str(annotation_id) for annotation_id in source_order]
    for entry in payload.get("results", []):
        annotation_id = str(entry.get("display_annotation_id", entry.get("annotation_id", "")))
        source_name = source_names.get(annotation_id)
        if source_name:
            entry["source_name"] = source_name
    return payload


def read_yaml_config(config_path: str | Path = "config.yaml") -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise ImportError("Install PyYAML first: pip install pyyaml") from error

    with Path(config_path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    if not isinstance(config, dict):
        raise ValueError("YAML config must be a mapping")

    return config


def normalize_annotations(
    annotations: list[Any],
    *,
    source: str,
    doc_id: str | None,
    annotation_id: str | None,
) -> tuple[list[dict[str, Any]], list[ValidationIssue]]:
    normalized: list[dict[str, Any]] = []
    issues: list[ValidationIssue] = []

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, dict):
            issues.append(
                ValidationIssue(
                    source=source,
                    doc_id=doc_id,
                    annotation_id=annotation_id,
                    annotation_index=index,
                    field=None,
                    message="Annotation is not an object",
                )
            )
            continue

        fixed = normalize_span(annotation)
        _coerce_required_fields(fixed)
        issues.extend(
            _validate_annotation(
                fixed,
                source=source,
                doc_id=doc_id,
                annotation_id=annotation_id,
                annotation_index=index,
            )
        )
        normalized.append(fixed)

    return normalized, issues


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            yield line_number, json.loads(line)


def _coerce_required_fields(annotation: dict[str, Any]) -> None:
    # label/category/subtype are already canonicalized upstream via normalize_span;
    # just clean them and coerce offsets/text.
    for field in ("label", "category", "subtype"):
        annotation[field] = _clean_label_part(annotation.get(field))
    annotation["text"] = str(annotation.get("text", ""))

    for field in ("begin", "end"):
        value = annotation.get(field)
        if isinstance(value, bool):
            continue
        try:
            annotation[field] = int(value)
        except (TypeError, ValueError):
            pass


def _validate_annotation(
    annotation: dict[str, Any],
    *,
    source: str,
    doc_id: str | None,
    annotation_id: str | None,
    annotation_index: int,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    for field in REQUIRED_ANNOTATION_FIELDS:
        if field not in annotation:
            issues.append(
                _issue(source, doc_id, annotation_id, annotation_index, field, "Missing field")
            )

    expected_types = {
        "begin": int,
        "end": int,
        "label": str,
        "category": str,
        "subtype": str,
        "text": str,
    }
    for field, expected_type in expected_types.items():
        if field in annotation and not _is_exact_type(annotation[field], expected_type):
            issues.append(
                _issue(
                    source,
                    doc_id,
                    annotation_id,
                    annotation_index,
                    field,
                    f"Expected {expected_type.__name__}, got {type(annotation[field]).__name__}",
                )
            )

    category = annotation.get("category")
    subtype = annotation.get("subtype")
    label = annotation.get("label")
    if isinstance(category, str) and isinstance(subtype, str) and isinstance(label, str):
        expected_label = _compose_label(category, subtype)
        if label != expected_label:
            issues.append(
                _issue(
                    source,
                    doc_id,
                    annotation_id,
                    annotation_index,
                    "label",
                    f"Expected label to be {expected_label!r} from Category/Subtype, got {label!r}",
                )
            )

    begin = annotation.get("begin")
    end = annotation.get("end")
    text = annotation.get("text")
    if isinstance(begin, int) and isinstance(end, int) and isinstance(text, str):
        expected_length = end - begin
        if len(text) != expected_length:
            issues.append(
                _issue(
                    source,
                    doc_id,
                    annotation_id,
                    annotation_index,
                    "text",
                    f"Expected len(text) == end - begin ({expected_length}), got {len(text)}",
                )
            )

    return issues


def _is_exact_type(value: Any, expected_type: type) -> bool:
    if expected_type is int:
        return type(value) is int
    return isinstance(value, expected_type)


def _annotation_id_jsonl_path(folder: Path, annotation_id: str) -> Path:
    if "/" in annotation_id or "\\" in annotation_id:
        raise ValueError(f"annotation_id cannot contain path separators: {annotation_id!r}")

    if annotation_id.endswith(".jsonl"):
        return folder / annotation_id
    return folder / f"{annotation_id}.jsonl"


def _records_for_annotation_id(
    records: list[dict[str, Any]],
    annotation_id: str,
) -> list[dict[str, Any]]:
    return [record for record in records if record["annotation_id"] == annotation_id]


def _source_name_from_config(source_config: dict[str, Any]) -> str | None:
    for key in ("name", "source_name", "display_name"):
        value = source_config.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _source_excluded(source_config: dict[str, Any]) -> bool:
    return source_config.get("exclude") is True


def _records_with_source_name(records: list[dict[str, Any]], source_name: str | None) -> list[dict[str, Any]]:
    if not source_name:
        return records
    return [{**record, "source_name": source_name} for record in records]


def _clean_annotation_output_folder(folder: str | Path) -> None:
    folder_path = Path(folder)
    if not folder_path.exists():
        return
    for path in folder_path.glob("*.jsonl"):
        path.unlink()


def _issue(
    source: str,
    doc_id: str | None,
    annotation_id: str | None,
    annotation_index: int,
    field: str,
    message: str,
) -> ValidationIssue:
    return ValidationIssue(
        source=source,
        doc_id=doc_id,
        annotation_id=annotation_id,
        annotation_index=annotation_index,
        field=field,
        message=message,
    )


def _split_label(label: str) -> tuple[str, str]:
    if ":" not in label:
        return label, ""
    category, subtype = label.split(":", maxsplit=1)
    return category, subtype


def _compose_label(category: str, subtype: str) -> str:
    if not subtype:
        return category
    return f"{category}:{subtype}"


def _clean_label_part(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.casefold() in {"none", "nan", "null"}:
        return ""
    return text
