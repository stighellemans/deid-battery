"""GLiNER zero-shot PII runner. Needs `gliner` (transformers <5.7).

config::
    - id: gliner-pii
      runner: gliner
      params: {model: urchade/gliner_multi_pii-v1, threshold: 0.5}

GLiNER is generic, so it emits Category-level labels (no subtype). Override the
prompt-label -> project-label map via params.label_map.
"""
from __future__ import annotations

import re

from ..checkpoint import ordered, resume_split, track
from ..schema import make_span

# word/punctuation tokens, ~matching GLiNER's word splitter, for token-sized windows
_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def _device(name):
    # Mirror runners/robbert._device: explicit "cpu" stays CPU; None/"auto"/"" auto-
    # detects (cuda > mps > cpu); an explicit "mps"/"cuda" is honoured as given.
    import torch
    if name in (None, "auto", "", "cpu") and name != "cpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name or "cpu")

DEFAULT_LABEL_MAP = {
    "person": "Name", "name": "Name",
    "organization": "Organization", "hospital": "Organization", "healthcare institution": "Organization",
    "address": "Address_Location", "location": "Address_Location", "city": "Address_Location", "street": "Address_Location",
    "date": "Date", "date of birth": "Age_Birthdate", "age": "Age_Birthdate",
    "phone number": "Contactdetails", "email": "Contactdetails", "email address": "Contactdetails", "url": "Contactdetails",
    "identifier": "ID", "id number": "ID", "medical record number": "ID",
    "social security number": "ID", "national register number": "ID",
    "profession": "Profession", "occupation": "Profession",
}


def _windows(text, max_tokens, overlap_tokens):
    """Yield (char_offset, chunk) windows of at most ``max_tokens`` word/punct
    tokens, so GLiNER never silently truncates a long document. Sized in TOKENS
    (not characters) because GLiNER's limit is a token count; the budget must
    leave room for the label-prompt tokens, which also count toward that limit."""
    spans = [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]
    if len(spans) <= max_tokens:
        yield 0, text
        return
    step = max(1, max_tokens - overlap_tokens)
    i = 0
    while i < len(spans):
        win = spans[i:i + max_tokens]
        b, e = win[0][0], win[-1][1]
        yield b, text[b:e]
        if i + max_tokens >= len(spans):
            break
        i += step


def run(docs, params):
    ckpt, todo, by_doc = resume_split(docs, params)
    if not todo:  # everything already checkpointed -> don't load the model
        return ordered(docs, by_doc)
    from gliner import GLiNER

    load_kwargs = {}
    if params.get("revision"):
        load_kwargs["revision"] = params["revision"]
    model = GLiNER.from_pretrained(
        params.get("model", "urchade/gliner_multi_pii-v1"), **load_kwargs
    )
    model.eval()
    # GLiNER.predict_entities runs on the model's device, so moving the model is
    # enough (it tokenises + places tensors internally). Default None auto-detects.
    model.to(_device(params.get("device")))
    threshold = params.get("threshold", 0.5)
    label_map = {k.lower(): v for k, v in (params.get("label_map") or DEFAULT_LABEL_MAP).items()}
    labels = list(label_map)

    # GLiNER caps (text + label-prompt) tokens at config.max_len (e.g. 384) and
    # silently truncates longer input -> dropped entities. Window by tokens, with a
    # budget under max_len that also leaves room for the label-prompt tokens.
    model_max = int(getattr(getattr(model, "config", None), "max_len", 0) or 384)
    max_tokens = int(params.get("max_tokens") or max(64, model_max - 100))
    overlap_tokens = int(params.get("overlap_tokens", 50))

    batch_size = int(params.get("batch_size", 1) or 1)

    def _accumulate(best, text, offset, ents):
        for ent in ents:
            label = label_map.get(str(ent.get("label", "")).lower())
            if not label:
                continue
            b, e = offset + int(ent["start"]), offset + int(ent["end"])
            if e <= b:
                continue
            key, score = (b, e, label), float(ent.get("score", 0.0))
            if key not in best or score > best[key]["score"]:
                best[key] = make_span(b, e, label, text[b:e], score=score)

    def _finalize(best):
        return sorted(best.values(), key=lambda s: (s["begin"], s["end"]))

    if batch_size <= 1:
        # Default path: one window per predict call. Unchanged from pre-batching.
        for d in track(todo, params):
            text = d["text"]
            best: dict[tuple, dict] = {}
            for offset, chunk in _windows(text, max_tokens, overlap_tokens):
                _accumulate(best, text, offset,
                            model.predict_entities(chunk, labels, threshold=threshold))
            spans = _finalize(best)
            by_doc[d["doc_id"]] = spans
            ckpt.record(d["doc_id"], spans)
        return ordered(docs, by_doc)

    # Batched path (batch_size>1): pool windows ACROSS docs into GLiNER's own
    # batch_predict_entities (it pads+masks internally), so a fair GPU timing isn't
    # throttled by short single-window docs. Same model call -> same entities; only
    # the batch shape changes. Docs buffered `batch_docs` at a time, checkpointed as done.
    batch_docs = max(batch_size, int(params.get("batch_docs", 64)))

    def _results():
        for gi in range(0, len(todo), batch_docs):
            group = todo[gi:gi + batch_docs]
            texts = {d["doc_id"]: d["text"] for d in group}
            best_by = {d["doc_id"]: {} for d in group}
            work = [(d["doc_id"], off, chunk) for d in group
                    for off, chunk in _windows(d["text"], max_tokens, overlap_tokens)]
            for bi in range(0, len(work), batch_size):
                w = work[bi:bi + batch_size]
                preds = model.batch_predict_entities([c for _, _, c in w], labels, threshold=threshold)
                for (doc_id, off, _), ents in zip(w, preds):
                    _accumulate(best_by[doc_id], texts[doc_id], off, ents)
            for d in group:
                yield d["doc_id"], _finalize(best_by[d["doc_id"]])

    for doc_id, spans in track(_results(), params, total=len(todo)):
        by_doc[doc_id] = spans
        ckpt.record(doc_id, spans)
    return ordered(docs, by_doc)
