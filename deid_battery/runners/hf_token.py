"""Generic Hugging Face token-classification runner (BIO / BIOES), e.g. the
openai / OpenMed privacy filters.

The OpenAIPrivacyFilter architecture needs transformers>=5.7, which conflicts
with GLiNER (<5.7) -- run this via a `venv:` in the config when both are present.

HF fast tokenizers return CHARACTER offsets (even for byte-level BPE), so text
slicing is correct. Long docs are processed in overlapping char windows. Emits
Category-level project labels via params.label_map (default below covers the two
privacy filters).
"""
from __future__ import annotations

from ..progress import track
from ..schema import make_span

# native label (lowercased) -> project Category. Covers openai/privacy-filter and
# OpenMed/privacy-filter-multilingual; override via params.label_map.
DEFAULT_LABEL_MAP = {
    # openai/privacy-filter
    "account_number": "ID", "private_address": "Address_Location", "private_date": "Date",
    "private_email": "Contactdetails", "private_person": "Name", "private_phone": "Contactdetails",
    "private_url": "Contactdetails", "secret": "Anonymize_Other",
    # OpenMed/privacy-filter-multilingual
    "firstname": "Name", "middlename": "Name", "lastname": "Name", "prefix": "Name",
    "username": "ID", "accountname": "ID", "age": "Age_Birthdate", "dateofbirth": "Age_Birthdate",
    "date": "Date", "time": "Date", "email": "Contactdetails", "phone": "Contactdetails",
    "url": "Contactdetails", "street": "Address_Location", "buildingnumber": "Address_Location",
    "secondaryaddress": "Address_Location", "city": "Address_Location", "county": "Address_Location",
    "state": "Address_Location", "zipcode": "Address_Location", "gpscoordinates": "Address_Location",
    "ordinaldirection": "Address_Location", "occupation": "Profession", "jobtitle": "Profession",
    "organization": "Organization", "bankaccount": "ID", "iban": "ID", "bic": "ID",
    "bitcoinaddress": "ID", "ethereumaddress": "ID", "litecoinaddress": "ID", "creditcard": "ID",
    "cvv": "ID", "pin": "ID", "maskednumber": "ID", "ssn": "ID", "imei": "ID", "ipaddress": "ID",
    "macaddress": "ID", "vin": "ID", "vrm": "ID", "password": "Anonymize_Other", "useragent": "Anonymize_Other",
}
_BIOES = ("B-", "I-", "E-", "S-")


def _windows(text, size=1400, overlap=250):
    if len(text) <= size:
        yield 0, text
        return
    start, n = 0, len(text)
    while start < n:
        yield start, text[start:start + size]
        if start + size >= n:
            break
        start += size - overlap


def _split_bioes(label):
    for p in _BIOES:
        if label.startswith(p):
            return p[0], label[len(p):]
    return "S", label


def _trim(b, e, text):
    while b < e and text[b].isspace():
        b += 1
    while e > b and text[e - 1].isspace():
        e -= 1
    return b, e


def _predict_chunk(model, tok, chunk, torch):
    enc = tok(chunk, return_offsets_mapping=True, return_tensors="pt",
              truncation=True, max_length=512)
    offs = enc.pop("offset_mapping")[0].tolist()
    with torch.no_grad():
        probs = model(**enc).logits[0].softmax(-1)
    ids = probs.argmax(-1).tolist()
    scores = probs.max(-1).values.tolist()
    id2label = model.config.id2label

    toks = []
    for lid, sc, (b, e) in zip(ids, scores, offs):
        if e <= b:
            continue
        toks.append({"begin": int(b), "end": int(e), "label": str(id2label[int(lid)]), "score": float(sc)})

    spans, active = [], None
    for t in toks:
        if t["label"] == "O":
            if active:
                spans.append(active); active = None
            continue
        prefix, label = _split_bioes(t["label"])
        if prefix == "S":
            if active:
                spans.append(active); active = None
            spans.append({**t, "label": label}); continue
        if prefix == "B" or active is None or active["label"] != label:
            if active:
                spans.append(active)
            active = {**t, "label": label}
            if prefix == "E":
                spans.append(active); active = None
            continue
        active["end"] = t["end"]; active["score"] = min(active["score"], t["score"])
        if prefix == "E":
            spans.append(active); active = None
    if active:
        spans.append(active)
    return [s for s in spans if s["end"] > s["begin"]]


def run(docs, params):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_id = params["model"]
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id, trust_remote_code=params.get("trust_remote_code", True)).eval()
    label_map = {k.lower(): v for k, v in (params.get("label_map") or DEFAULT_LABEL_MAP).items()}

    by_doc = {}
    for d in track(docs, params):
        text = d["text"]
        best: dict[tuple, dict] = {}
        for offset, chunk in _windows(text):
            for s in _predict_chunk(model, tok, chunk, torch):
                label = label_map.get(s["label"].lower())
                if not label:
                    continue
                b, e = _trim(offset + s["begin"], offset + s["end"], text)
                if e <= b:
                    continue
                key = (b, e, label)
                if key not in best or s["score"] > best[key]["score"]:
                    best[key] = make_span(b, e, label, text[b:e], score=s["score"])
        by_doc[d["doc_id"]] = sorted(best.values(), key=lambda s: (s["begin"], s["end"]))
    return by_doc
