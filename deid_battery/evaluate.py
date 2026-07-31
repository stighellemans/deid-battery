"""Char-level evaluation against a deid-eval-annotator gold bundle.

Reuses the vendored, self-contained ``evaluate_quantity_deid`` + ``data_readers``.
Returns the evaluation payload (core PII recall, non-PII redaction buckets, label confusion, ...)
keyed per model, with display names attached.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

_VENDOR = Path(__file__).resolve().parent / "_vendor"
sys.path.insert(0, str(_VENDOR))


def evaluate(by_doc_paths: dict[str, str], bundle_dir, document_lengths,
             ignore_categories=None, source_names=None, source_order=None,
             work_dir="out") -> dict:
    import data_readers as dr
    import evaluate_quantity_deid as eq

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
        bundle_dir=bundle_dir,
        ignore_categories=ignore_categories,
        require_complete_gold=True,
        document_lengths=document_lengths,
    )
    if source_names:
        dr.attach_source_names_to_payload(payload, source_names, source_order)
    return payload
