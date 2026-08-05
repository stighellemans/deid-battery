"""Build deid-battery input.jsonl from the validated gold set.

This is an INPUT PRE-STEP (not part of the battery run): it converts the
corrected working-copy gold file into the battery's input schema
``{doc_id, text, annotations, metadata}``.

The default ``embedded`` mode reads canonical ``patient_name`` and
``caregiver_names`` values already frozen in each document's metadata.  This is
the required mode for synthetic benchmark v2.2: name-surface transformations
must not change the underlying full-name metadata.

The legacy ``oracle`` mode derives names from the same file's gold name spans.
It is retained only for explicitly labelled upper-bound experiments and must
not be used for the v2.2 metadata-enabled benchmark condition.

Metadata shaping mirrors the original model_spans oracle builder:
  - patient: collapse all Name:Patient strings; surname = last token of the
    longest one; first_names = the remaining distinct tokens; aliases = the
    full strings.
  - caregivers: one person per distinct Name:Caregiver string;
    first_names = tokens[:-1], surname = tokens[-1], aliases = [the string].

  python scripts/build_input_from_gold.py --gold benchmark.v2.2.jsonl
  python scripts/build_input_from_gold.py --metadata-mode oracle \
      --limit 12 --require-caregiver \
      --out input.smoke.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_GOLD = Path(__file__).resolve().parents[2] / \
    "gold-testset-validation/platform/data/annotations.jsonl"


def _names(annotations, label):
    out, seen = [], set()
    for a in annotations:
        if (a.get("label") or "") == label:
            t = (a.get("text") or "").strip()
            if t and t not in seen:
                seen.add(t)
                out.append(t)
    return out


def _patient(texts):
    if not texts:
        return None
    canonical = max(texts, key=lambda s: len(s.split())).split()
    surname = canonical[-1] if canonical else None
    firsts = [t for t in dict.fromkeys(tok for s in texts for tok in s.split()) if t != surname]
    p = {}
    if firsts:
        p["first_names"] = firsts
    if surname:
        p["surname"] = surname
    p["aliases"] = list(texts)
    return p or None


def _caregivers(texts):
    out = []
    for name in texts:
        toks = name.split()
        if not toks:
            continue
        c = {"surname": toks[-1], "aliases": [name]}
        if toks[:-1]:
            c["first_names"] = toks[:-1]
        out.append(c)
    return out


def _embedded_metadata(row):
    source = row.get("metadata") or {}
    if source.get("patient"):
        # Already in battery shape; useful for adapted private inputs.
        return {
            "patient": source["patient"],
            **({"caregivers": source["caregivers"]} if source.get("caregivers") else {}),
        }

    patient_name = source.get("patient_name")
    caregivers = source.get("caregiver_names") or []
    meta = {}
    if patient_name:
        given = str(patient_name.get("given_name") or "").strip()
        family = str(patient_name.get("family_name") or "").strip()
        if not given or not family:
            raise ValueError(f"{row.get('document_id')}: incomplete embedded patient_name")
        full = f"{given} {family}"
        meta["patient"] = {
            "first_names": given.split(),
            "surname": family,
            "aliases": [full],
        }
    if caregivers:
        shaped = []
        for caregiver in caregivers:
            given = str(caregiver.get("given_name") or "").strip()
            family = str(caregiver.get("family_name") or "").strip()
            if not given or not family:
                raise ValueError(f"{row.get('document_id')}: incomplete embedded caregiver_name")
            shaped.append(
                {
                    "first_names": given.split(),
                    "surname": family,
                    "aliases": [f"{given} {family}"],
                }
            )
        meta["caregivers"] = shaped
    return meta


def build(gold_path, out_path, limit=None, require_caregiver=False, metadata_mode="embedded"):
    rows = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            anns = r.get("annotations") or []
            if metadata_mode == "embedded":
                meta = _embedded_metadata(r)
                patient = meta.get("patient")
                caregivers = meta.get("caregivers") or []
            elif metadata_mode == "oracle":
                patient = _patient(_names(anns, "Name:Patient"))
                caregivers = _caregivers(_names(anns, "Name:Caregiver"))
                meta = {}
                if patient:
                    meta["patient"] = patient
                if caregivers:
                    meta["caregivers"] = caregivers
            elif metadata_mode == "none":
                patient, caregivers, meta = None, [], {}
            else:
                raise ValueError(f"unknown metadata mode: {metadata_mode}")
            if require_caregiver and not (patient and caregivers):
                continue
            rec = {"doc_id": str(r.get("document_id") or r.get("doc_id") or ""),
                   "text": r.get("text", ""), "annotations": anns}
            if meta:
                rec["metadata"] = meta
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            written += 1
            if limit and written >= limit:
                break
    print(f"wrote {out_path} ({written} docs) from {gold_path}")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gold", default=str(DEFAULT_GOLD))
    ap.add_argument("--out", default="input.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--metadata-mode",
        choices=("embedded", "oracle", "none"),
        default="embedded",
        help="Use frozen embedded metadata (default), explicit oracle gold-span metadata, or none.",
    )
    ap.add_argument("--require-caregiver", action="store_true",
                    help="keep only docs with BOTH a patient and >=1 caregiver name (good for smoke tests)")
    a = ap.parse_args()
    build(a.gold, a.out, a.limit, a.require_caregiver, a.metadata_mode)


if __name__ == "__main__":
    main()
