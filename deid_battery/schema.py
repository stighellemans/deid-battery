"""Unified span schema + by_doc I/O shared by every runner. Stdlib only.

Every runner emits one record per document::

    {"doc_id": str, "num_entities": int, "entities": [span, ...]}

where ``span = {begin, end, label, text, Category, Subtype}`` (Category/Subtype
are derived from ``label`` of the form ``Category:Subtype`` or just ``Category``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

SPAN_FIELDS = ("begin", "end", "label", "text", "Category", "Subtype")


def split_category_subtype(label: str) -> tuple[str, str | None]:
    if label and ":" in label:
        cat, sub = label.split(":", 1)
        return cat, sub
    return label, None


def make_span(begin: int, end: int, label: str, text: str, **extra: Any) -> dict:
    cat, sub = split_category_subtype(label)
    span = {"begin": int(begin), "end": int(end), "label": label,
            "text": text, "Category": cat, "Subtype": sub}
    span.update(extra)
    return span


def read_jsonl(path: str | Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_by_doc(path: str | Path, by_doc: dict[str, list[dict]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, ents in by_doc.items():
            f.write(json.dumps(
                {"doc_id": doc_id, "num_entities": len(ents), "entities": ents},
                ensure_ascii=False) + "\n")
    return path


def read_by_doc(path: str | Path) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for row in read_jsonl(path):
        out[str(row.get("doc_id"))] = row.get("entities", [])
    return out
