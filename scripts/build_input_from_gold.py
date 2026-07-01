"""Build deid-battery input.jsonl from the validated gold set.

This is an INPUT PRE-STEP (not part of the battery run): it converts the
corrected working-copy gold file into the battery's input schema
``{doc_id, text, annotations, metadata}`` and injects ORACLE known-identifier
metadata (patient + caregiver names) derived from the gold name spans. Injecting
known identifiers at input-build time is the sanctioned place for it -- the
battery core never derives metadata from gold spans (see deid_battery/metadata.py).

Metadata shaping mirrors the original model_spans oracle builder:
  - patient: collapse all Name:Patient strings; surname = last token of the
    longest one; first_names = the remaining distinct tokens; aliases = the
    full strings.
  - caregivers: one person per distinct Name:Caregiver string;
    first_names = tokens[:-1], surname = tokens[-1], aliases = [the string].

  python scripts/build_input_from_gold.py                     # all 300 docs -> input.jsonl
  python scripts/build_input_from_gold.py --limit 12 --require-caregiver \
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


def build(gold_path, out_path, limit=None, require_caregiver=False):
    rows = [json.loads(l) for l in open(gold_path, encoding="utf-8") if l.strip()]
    written = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            anns = r.get("annotations") or []
            patient = _patient(_names(anns, "Name:Patient"))
            caregivers = _caregivers(_names(anns, "Name:Caregiver"))
            if require_caregiver and not (patient and caregivers):
                continue
            meta = {}
            if patient:
                meta["patient"] = patient
            if caregivers:
                meta["caregivers"] = caregivers
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
    ap.add_argument("--require-caregiver", action="store_true",
                    help="keep only docs with BOTH a patient and >=1 caregiver name (good for smoke tests)")
    a = ap.parse_args()
    build(a.gold, a.out, a.limit, a.require_caregiver)


if __name__ == "__main__":
    main()
