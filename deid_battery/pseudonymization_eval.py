"""Run the privacy-safe date/age pseudonymization evaluation.

The normal battery calls this after its model evaluation. It can also be run
separately because it evaluates the shared pseudonymization layer on gold spans,
independently of model detection performance.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import yaml


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
        "evaluate_spans": evaluate_spans,
        "load_document_texts": load_document_texts,
        "load_gold_date_spans": load_gold_date_spans,
        "write_private_details": write_private_details,
        "write_safe_export": write_safe_export,
    }


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
