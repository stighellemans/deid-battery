"""Input loading + adapters.

Battery input is a JSONL of ``{doc_id, text, [metadata], [annotations]}``.
``metadata`` (optional) is an explicit key-value object (shape in
:mod:`deid_battery.metadata`), consumed by ``metadata.source: from_input``.
``annotations`` (optional) are gold spans carried through for convenience.

The ``adapt`` CLI converts another source (e.g. the review VM's ``results.jsonl``)
into battery input and can build ``metadata`` from per-document fields. Drive it
either with flags or from a battery.yaml ``adapter:`` section (keys in YAML):

    python -m deid_battery.inputs --config battery.yaml          # reads `adapter:`
    python -m deid_battery.inputs --from results.jsonl --field raw_text \
        --patient-first patient_first_name --patient-last patient_last_name \
        --doc-date date --out input.jsonl
"""
from __future__ import annotations

import argparse
import json
import os

from .schema import read_jsonl


def load_input(path) -> list[dict]:
    """Load the battery's own input JSONL of {doc_id, text, [metadata],
    [annotations]}. Converting a raw source (e.g. results.jsonl) into this shape
    is a separate pre-step -- see :func:`adapt` / the module CLI below -- so the
    battery core stays a pure consumer. The embedded ``metadata`` is fetched at
    run time by ``metadata.source: from_input``."""
    docs = []
    for r in read_jsonl(path):
        doc_id = str(r.get("doc_id") or r.get("id") or r.get("document_id") or "")
        docs.append({
            "doc_id": doc_id,
            "text": r.get("text", ""),
            "metadata": r.get("metadata"),
            "annotations": r.get("annotations"),
        })
    return docs


def _person(first, last, initials):
    p = {}
    if first:
        p["first_names"] = str(first).split()
    if last:
        p["surname"] = str(last)
    if initials:
        p["initials"] = str(initials)
    full = " ".join(t for t in [str(first or "").strip(), str(last or "").strip()] if t).strip()
    if full:
        p["aliases"] = [full]
    return p or None


def build_metadata(row: dict, mf: dict) -> dict:
    """Build canonical per-doc metadata from configured source field names."""
    meta: dict = {}
    patient = _person(row.get(mf.get("patient_first")), row.get(mf.get("patient_last")),
                      row.get(mf.get("patient_initials")))
    if patient:
        meta["patient"] = patient
    caregiver = _person(row.get(mf.get("caregiver_first")), row.get(mf.get("caregiver_last")), None)
    if caregiver:
        meta["caregivers"] = [caregiver]
    dd = row.get(mf.get("doc_date")) if mf.get("doc_date") else None
    if dd:
        meta["document_date"] = str(dd)
    return meta


def adapt(src, text_field="raw_text", id_field="id", out="input.jsonl",
          metadata_field=None, meta_fields=None) -> tuple[str, int]:
    meta_fields = meta_fields or {}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for r in read_jsonl(src):
            doc_id = str(r.get(id_field) or r.get("doc_id") or r.get("document_id") or "")
            rec = {"doc_id": doc_id, "text": r.get(text_field, "")}
            meta = dict(r.get(metadata_field) or {}) if metadata_field else {}
            meta.update({k: v for k, v in build_metadata(r, meta_fields).items()})
            if meta:
                rec["metadata"] = meta
            if r.get("annotations"):
                rec["annotations"] = r["annotations"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return out, n


def _from_config(config_path, out_override):
    """Read the pre-step's ``adapter:`` section from a battery.yaml -- so the
    field mapping lives in YAML rather than on the command line. The output path
    defaults to the battery's own ``input:`` so the two stay in sync."""
    import yaml
    cfg = yaml.safe_load(open(config_path, encoding="utf-8")) or {}
    ad = cfg.get("adapter") or {}
    src = ad.get("from") or ad.get("path")
    if not src:
        raise SystemExit(f"{config_path}: no `adapter.from` (raw source JSONL) configured")
    out = (out_override or ad.get("out")
           or (cfg.get("input") if isinstance(cfg.get("input"), str) else None) or "input.jsonl")
    return dict(src=os.path.expanduser(str(src)), field=ad.get("text_field", "raw_text"),
                id_field=ad.get("id_field", "id"), metadata_field=ad.get("metadata_field"),
                mf=ad.get("metadata_fields") or {}, out=os.path.expanduser(str(out)))


def main():
    ap = argparse.ArgumentParser(description="Adapt a source JSONL into battery input {doc_id,text,[metadata]}.")
    ap.add_argument("--config", default=None,
                    help="battery.yaml: read its `adapter:` section instead of the flags below")
    ap.add_argument("--from", dest="src", default=None)
    ap.add_argument("--field", default="raw_text", help="text field in the source JSONL")
    ap.add_argument("--id-field", default="id")
    ap.add_argument("--metadata-field", default=None, help="source field already holding a canonical metadata object")
    ap.add_argument("--patient-first", help="source field for the patient first name(s)")
    ap.add_argument("--patient-last", help="source field for the patient surname")
    ap.add_argument("--patient-initials", help="source field for the patient initials")
    ap.add_argument("--caregiver-first", help="source field for a caregiver first name(s)")
    ap.add_argument("--caregiver-last", help="source field for a caregiver surname")
    ap.add_argument("--doc-date", help="source field for the document date")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.config:
        p = _from_config(a.config, a.out)
        out, n = adapt(p["src"], p["field"], p["id_field"], p["out"], p["metadata_field"], p["mf"])
    else:
        if not a.src:
            ap.error("either --config or --from is required")
        mf = {"patient_first": a.patient_first, "patient_last": a.patient_last,
              "patient_initials": a.patient_initials, "caregiver_first": a.caregiver_first,
              "caregiver_last": a.caregiver_last, "doc_date": a.doc_date}
        out, n = adapt(a.src, a.field, a.id_field, a.out or "input.jsonl", a.metadata_field, mf)
    print(f"wrote {out} ({n} docs)")


if __name__ == "__main__":
    main()
