"""Deidentify runner (nedap/deidentify; Trienes et al. 2020) — the "Deidentify"
system from the MDPI 2025 Dutch replication study (Electronics 14(8):1636).

It is a 2020 stack (spaCy 2.x / flair 0.10 / torch 1.10) with no arm64/py3.12
wheels, so it runs in a DEDICATED py3.9 venv referenced via ``venv:`` in the
config (see requirements/deidentify.txt + scripts/setup_deidentify_venv.sh).
Runs on **CPU** by default (works on any amd64 Linux box; no GPU needed).

config::
    - id: deidentify
      runner: deidentify
      venv: /opt/deidentify-venv          # the py3.9 amd64 env
      params: {model: model_bilstmcrf_ons_large-v0.2.0, chunk: 50}

Memory: the bilstmcrf models use PooledFlairEmbeddings, whose footprint grows per
doc and is NOT reclaimed in-process even after the pool is cleared (glibc keeps
the high-water RSS). So a full run is split into ``chunk``-sized slices, each in a
FRESH subprocess (the OS reclaims memory on exit). The pool is also cleared per
doc, so each note is embedded independently and results are chunk-invariant —
``chunk`` is purely a memory bound (50 docs ~= 9 GB; lower it for less RAM).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from ..progress import track
from ..schema import make_span, read_by_doc, read_jsonl, write_by_doc

DEFAULT_MODEL = "model_bilstmcrf_ons_large-v0.2.0"

# nedap/deidentify NUT tagset -> project labels. deidentify is generic Dutch
# de-id: it cannot resolve patient/caregiver subtypes, so Name/Address/ID stay
# Category-level (like GLiNER). It separates care orgs (Hospital, Care_Institute)
# from companies (Organization_Company), so the Organization subtype is kept. Its
# catch-all "Other" tag has no canonical Category and is dropped. Override via
# params.label_map.
DEFAULT_LABEL_MAP = {
    "Name": "Name", "Initials": "Name",
    "Age": "Age_Birthdate", "Date": "Date",
    "Phone_fax": "Contactdetails", "Email": "Contactdetails", "URL_IP": "Contactdetails",
    "SSN": "ID", "ID": "ID",
    "Hospital": "Organization:Healthcare", "Care_Institute": "Organization:Healthcare",
    "Organization_Company": "Organization:Other",
    "Internal_Location": "Address_Location", "Address": "Address_Location",
    "Profession": "Profession",
}


def _clear_pool(tagger) -> None:
    """Reset every PooledFlairEmbeddings pool so each doc is embedded
    independently (keeps results chunk-invariant)."""
    embs = getattr(getattr(tagger, "tagger", None), "embeddings", None)
    members = getattr(embs, "embeddings", [embs]) if embs is not None else []
    for e in members or []:
        if e.__class__.__name__ == "PooledFlairEmbeddings":
            for attr in ("word_embeddings", "word_count"):
                if hasattr(e, attr):
                    setattr(e, attr, {})


def _windows(text, max_chars, overlap):
    """Yield (char_offset, chunk) windows of <= max_chars, preferring to cut at a
    newline/space so tokens aren't split. flair's char-LM memory scales with the
    length of a single annotate() call, so long docs (real corpora reach ~100 KB)
    must be windowed or they spike RAM into OOM regardless of `chunk`. Overlap
    catches boundary entities; duplicates are collapsed by (begin,end,label)."""
    n = len(text)
    if not max_chars or n <= max_chars:
        yield 0, text
        return
    start = 0
    while start < n:
        end = min(start + max_chars, n)
        if end < n:  # back up to a whitespace boundary within the last 400 chars
            cut = text.rfind("\n", end - 400, end)
            if cut <= start:
                cut = text.rfind(" ", end - 400, end)
            if cut > start:
                end = cut
        yield start, text[start:end]
        if end >= n:
            break
        start = max(end - overlap, start + 1)


def _annotate(docs, params):
    """In-process inference over (a slice of) docs. Run per chunk in a fresh
    subprocess by run()."""
    import flair
    import torch
    from deidentify.base import Document
    from deidentify.taggers import FlairTagger
    from deidentify.tokenizer import TokenizerFactory

    dev = str(params.get("device", "cpu")).lower()
    flair.device = torch.device("cuda" if dev == "cuda" and torch.cuda.is_available() else "cpu")

    model_name = params.get("model", DEFAULT_MODEL)
    label_map = {**DEFAULT_LABEL_MAP, **(params.get("label_map") or {})}
    tokenizer = TokenizerFactory().tokenizer(corpus="ons", disable=("tagger", "ner"))
    # PRIMARY memory lever. deidentify's FlairTagger defaults to mini_batch_size=256
    # (flair's own default is 32) — it feeds 256 sentences per char-LM batch, which
    # is what spikes RAM to ~12 GB on real clinical text with long sentences. 32 is
    # the safe default; drop to 8-16 on tight boxes.
    mbs = int(params.get("mini_batch_size", 32) or 32)
    tagger = FlairTagger(model=model_name, tokenizer=tokenizer, mini_batch_size=mbs, verbose=False)

    # Safety net for pathological single long sentences (a multi-KB line with no
    # break): cap the max chars in one annotate() call. Not the main lever — flair
    # mini-batches internally — so default is generous. 0 disables.
    max_chars = int(params.get("max_chars", 20000) or 0)
    overlap = int(params.get("overlap", 500))

    by_doc = {}
    for d in track(docs, params):
        text = d.get("text", "")
        best: dict[tuple, dict] = {}
        if text.strip():
            for off, chunk in _windows(text, max_chars, overlap):
                try:
                    doc = tagger.annotate([Document(name=str(d["doc_id"]), text=chunk)])[0]
                    for ann in doc.annotations:
                        label = label_map.get(str(ann.tag))
                        if not label:
                            continue
                        b, e = off + int(ann.start), off + int(ann.end)
                        if e <= b:
                            continue
                        best[(b, e, label)] = make_span(b, e, label, text[b:e])
                except Exception:  # noqa: BLE001 — one bad window must not drop the doc
                    pass
                finally:
                    _clear_pool(tagger)
        by_doc[d["doc_id"]] = sorted(best.values(), key=lambda s: (s["begin"], s["end"]))
    return by_doc


def _run_chunk_subprocess(chunk_docs, params):
    """Run one chunk in a fresh interpreter so its memory is reclaimed on exit.
    Reuses _worker.py with `_inproc` set so the child takes the in-process path."""
    tmp = Path(tempfile.mkdtemp(prefix="deid_chunk_"))
    try:
        docs_p, params_p, out_p = tmp / "docs.jsonl", tmp / "params.json", tmp / "out.jsonl"
        with open(docs_p, "w", encoding="utf-8") as f:
            for d in chunk_docs:
                f.write(json.dumps({"doc_id": d["doc_id"], "text": d.get("text", "")},
                                   ensure_ascii=False) + "\n")
        params_p.write_text(json.dumps({**params, "_inproc": True}), encoding="utf-8")
        subprocess.run(
            [sys.executable, "-m", "deid_battery.runners._worker", "deidentify",
             str(params_p), str(docs_p), str(out_p)],
            check=True, env={**os.environ},
        )
        return read_by_doc(out_p)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def run(docs, params):
    chunk = int(params.get("chunk", 50) or 0)
    # _inproc (set by the chunk subprocess) or no chunking requested -> run here.
    if params.get("_inproc") or not chunk or len(docs) <= chunk:
        return _annotate(docs, params)

    by_doc = {}
    n = len(docs)
    for start in range(0, n, chunk):
        sl = docs[start:start + chunk]
        print(f"[{params.get('_label', 'deidentify')}] chunk {start}-{start + len(sl)}/{n}",
              file=sys.stderr, flush=True)
        by_doc.update(_run_chunk_subprocess(sl, params))
    # preserve input order
    return {d["doc_id"]: by_doc.get(d["doc_id"], []) for d in docs}
