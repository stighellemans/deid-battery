from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/build_input_from_gold.py"
SPEC = importlib.util.spec_from_file_location("build_input_from_gold", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_embedded_metadata_is_used_instead_of_transformed_gold_surface(tmp_path: Path) -> None:
    source = tmp_path / "gold.jsonl"
    output = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "text": "NO werd gezien door Lila-Jane.",
                "annotations": [
                    {"begin": 0, "end": 2, "label": "Name:Patient", "text": "NO"},
                    {"begin": 18, "end": 27, "label": "Name:Caregiver", "text": "Lila-Jane"},
                ],
                "metadata": {
                    "patient_name": {"given_name": "Ny", "family_name": "Oruç"},
                    "caregiver_names": [
                        {"given_name": "Lila-Jane", "family_name": "Blommen"}
                    ],
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    MODULE.build(source, output, metadata_mode="embedded")
    row = json.loads(output.read_text(encoding="utf-8"))

    assert row["text"] == "NO werd gezien door Lila-Jane."
    assert row["annotations"][0]["text"] == "NO"
    assert row["metadata"] == {
        "patient": {
            "first_names": ["Ny"],
            "surname": "Oruç",
            "aliases": ["Ny Oruç"],
        },
        "caregivers": [
            {
                "first_names": ["Lila-Jane"],
                "surname": "Blommen",
                "aliases": ["Lila-Jane Blommen"],
            }
        ],
    }


def test_oracle_mode_remains_explicit(tmp_path: Path) -> None:
    source = tmp_path / "gold.jsonl"
    output = tmp_path / "input.jsonl"
    source.write_text(
        json.dumps(
            {
                "document_id": "doc-1",
                "text": "NO",
                "annotations": [
                    {"begin": 0, "end": 2, "label": "Name:Patient", "text": "NO"}
                ],
                "metadata": {
                    "patient_name": {"given_name": "Ny", "family_name": "Oruç"}
                },
            },
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    MODULE.build(source, output, metadata_mode="oracle")
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["metadata"]["patient"]["surname"] == "NO"
