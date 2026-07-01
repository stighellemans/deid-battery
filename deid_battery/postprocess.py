"""Optional shared post-processing, applied uniformly to every model's spans.

Pipeline (all steps toggleable in battery.yaml ``postprocess:``):
  1. metadata-driven name injection  (patient via the vendored rule adder,
     caregiver via :func:`deid_battery.metadata.inject_name_spans`)
  2. vendored rule cleanup           (trim, regex date/URL extension, weekday
     extension, dedup) from deid_training/post_process
  3. optional INCEpTION-token normalization (needs jpype + a JVM)

Metadata (doc["_meta"]) flows in here too, so known patient/caregiver names can
be recovered at the post-processing stage regardless of the model.
"""
from __future__ import annotations

import sys
from pathlib import Path

from . import metadata as md_mod
from .schema import make_span

_VENDOR = Path(__file__).resolve().parent / "_vendor" / "post_process"


def _load_vendor():
    sys.path.insert(0, str(_VENDOR))
    from post_process import post_process_spans, set_current_doc_id
    return post_process_spans, set_current_doc_id


def canonical(spans, text):
    out = []
    for s in spans:
        b, e = int(s["begin"]), int(s["end"])
        b = max(0, min(b, len(text)))
        e = max(0, min(e, len(text)))
        if e <= b:
            continue
        label = s.get("label") or s.get("category") or s.get("Category")
        if not label:
            continue
        out.append(make_span(b, e, str(label), text[b:e]))
    out.sort(key=lambda s: (s["begin"], s["end"], s["label"]))
    return out


def _inception(spans, text, lang):
    from span_annotations.transform import (
        deduplicate_spans, normalize_spans, normalize_to_inception_tokens,
    )
    spans = normalize_to_inception_tokens(text=text, spans=spans, lang=lang)
    return deduplicate_spans(normalize_spans(spans, text))


def post_process(spans, text, meta=None, params=None, doc_id=None):
    params = params or {}
    use_meta = params.get("use_metadata", True) and bool(meta)
    spans = canonical(spans, text)

    if use_meta and meta.get("caregivers"):
        # vendored adder only handles the patient; inject caregivers ourselves
        spans = canonical(md_mod.inject_name_spans(spans, text, {"caregivers": meta["caregivers"]}), text)

    if params.get("rules", True):
        ppfn, set_doc = _load_vendor()
        pp_meta = md_mod.to_postprocess(meta) if use_meta else None
        set_doc(doc_id)
        try:
            spans = ppfn([dict(s) for s in spans], text, metadata=pp_meta)
        finally:
            set_doc(None)
        spans = canonical(spans, text)
    elif use_meta and meta.get("patient"):
        # rules off but metadata on: still inject the patient name
        spans = canonical(md_mod.inject_name_spans(spans, text, {"patient": meta["patient"]}), text)

    if params.get("inception", False):
        spans = canonical(_inception(spans, text, params.get("inception_lang", "nl")), text)
    return spans
