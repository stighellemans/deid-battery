"""Char-level evaluation against a deid-eval-annotator gold bundle.

Annotation readers (``data_readers``) come from the shared ``deid-eval`` package
(the deid-evaluation repo) instead of a vendored copy. The char-level evaluator
(``evaluate_quantity_deid``) is loaded from the gold bundle itself -- where it is
versioned together with the reference data and label taxonomy, and which the
``evaluate.bundle`` config already points at. Nothing is vendored.

Returns the evaluation payload (essential recall, FP buckets, label confusion,
...) keyed per model, with display names attached.
"""
from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path

import data_readers as dr


def _resolve_bundle(bundle_dir) -> Path:
    """The gold bundle (reference data + evaluator + manifest) is a placed
    artifact exported from deid-eval-annotator; point ``evaluate.bundle`` at it."""
    if bundle_dir:
        p = Path(bundle_dir)
        if p.exists():
            return p
    raise FileNotFoundError(
        f"evaluation bundle not found at {bundle_dir!r}. Place the deid-eval-annotator "
        f"gold bundle (reference_items.jsonl + evaluate_quantity_deid.py + manifest.json) "
        f"there, or point `evaluate.bundle` in the config at it."
    )


def _load_evaluator(bundle: Path):
    """Load the char-level evaluator that ships inside the bundle, so the metric
    definitions and label maps always match the reference format they were
    exported with."""
    path = bundle / "evaluate_quantity_deid.py"
    if not path.exists():
        raise FileNotFoundError(f"bundle {bundle} has no evaluate_quantity_deid.py")
    spec = importlib.util.spec_from_file_location("bundle_evaluate_quantity_deid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluate(by_doc_paths: dict[str, str], bundle_dir, document_lengths,
             ignore_categories=None, source_names=None, source_order=None,
             work_dir="out") -> dict:
    bundle = _resolve_bundle(bundle_dir)
    eq = _load_evaluator(bundle)

    pred_dir = Path(work_dir) / "_predictions"
    if pred_dir.exists():
        shutil.rmtree(pred_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # convert each model's by_doc.jsonl -> predictions/<id>.jsonl
    for model_id, path in by_doc_paths.items():
        dr.convert_annotator_level_jsonl_to_annotation_folder(
            path, pred_dir, annotation_id=model_id)

    payload = eq.evaluate_all_prediction_sources(
        predictions_path=pred_dir,
        bundle_dir=str(bundle),
        ignore_categories=ignore_categories,
        require_complete_gold=True,
        document_lengths=document_lengths,
    )
    if source_names:
        dr.attach_source_names_to_payload(payload, source_names, source_order)
    return payload
