"""RobBERT-style dual-head token-classification runner (bring-your-own .pt).

Loads a checkpoint with ``bio_classifier`` + ``label_classifier`` heads on top of
a HF encoder; sliding-window inference with overlap; CPU/CUDA/MPS auto.

config::
    - id: my-robbert
      runner: robbert
      params:
        checkpoint: /path/to/model.pt
        base_model: DTAI-KULeuven/robbert-2023-dutch-base
        train_metrics: /path/to/train_metrics.json   # supplies bio_labels/entity_labels
        # or set explicitly: entity_labels: [...]   bio_labels: [O, B, I]
        max_length: 512
        overlap: 64
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

from ..checkpoint import ordered, resume_split, track
from ..schema import make_span
from .phase_timing import elapsed_since, measure, warmup_docs, write_report


def _device(name):
    import torch
    if name in (None, "auto", "", "cpu") and name != "cpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name or "cpu")


def _resolved_batch_size(requested, device):
    """Resolve 0/None to a conservative accelerator-specific batch size."""
    value = int(requested or 0)
    if value > 0:
        return value
    device_type = getattr(device, "type", str(device)).lower()
    if device_type == "cuda":
        return 16
    if device_type == "mps":
        return 8
    return 1


def _is_out_of_memory(exc):
    message = str(exc).lower()
    return "out of memory" in message or "mps backend out of memory" in message


def _empty_device_cache(torch, device):
    device_type = getattr(device, "type", str(device)).lower()
    if device_type == "cuda":
        torch.cuda.empty_cache()
    elif device_type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()


def _windows(n, usable, overlap):
    if n <= 0:
        return
    if n <= usable:
        yield 0, n
        return
    step = usable - overlap
    last = n - usable
    s = 0
    while s < last:
        yield s, s + usable
        s += step
    yield last, n


def _labels(params):
    bio = params.get("bio_labels")
    ent = params.get("entity_labels")
    if not ent and params.get("train_metrics"):
        m = json.loads(Path(params["train_metrics"]).read_text(encoding="utf-8"))
        bio = bio or m.get("bio_labels")
        ent = m.get("entity_labels")
    return bio or ["O", "B", "I"], ent


def _decode(pb, pe, offs, text, bio_labels, ent_labels, bio_o):
    spans, start, cur_ent, end = [], None, None, None

    def flush():
        nonlocal start, cur_ent, end
        if start is not None and end is not None:
            b, e = int(offs[start][0]), int(offs[end][1])
            if e > b:
                spans.append(make_span(b, e, cur_ent, text[b:e]))
        start = cur_ent = end = None

    for i, (bi, ei) in enumerate(zip(pb, pe)):
        if int(offs[i][1]) <= int(offs[i][0]):
            continue
        tag = bio_labels[int(bi)] if int(bi) < len(bio_labels) else "O"
        if int(bi) == bio_o or tag not in ("B", "I"):
            flush()
            continue
        ent = ent_labels[int(ei)]
        if tag == "B" or start is None or ent != cur_ent:
            flush()
            start, cur_ent = i, ent
        end = i
    flush()
    spans.sort(key=lambda s: (s["begin"], s["end"]))
    return spans


def _prepare_window(tok, ids_slice):
    """Pack a token-id slice with the model's special tokens, robustly across
    transformers versions (mirrors the azure-vm server's build_prepared_window):
    prefer prepare_for_model, then build_inputs_with_special_tokens, then a manual
    cls/sep (or bos/eos) wrap. Returns (input_ids, special_tokens_mask)."""
    ids_slice = list(ids_slice)
    try:
        prepared = tok.prepare_for_model(
            ids_slice, add_special_tokens=True, return_attention_mask=True,
            return_special_tokens_mask=True, truncation=False)
        return list(prepared["input_ids"]), list(prepared["special_tokens_mask"])
    except (AttributeError, AssertionError, NotImplementedError, TypeError, ValueError, KeyError):
        pass  # method missing (transformers 5.x removed it) or unsupported -> manual packing below
    build_inputs = getattr(tok, "build_inputs_with_special_tokens", None)
    if callable(build_inputs):
        packed = list(build_inputs(ids_slice))
    else:
        cls_id = getattr(tok, "cls_token_id", None)
        if cls_id is None:
            cls_id = getattr(tok, "bos_token_id", None)
        sep_id = getattr(tok, "sep_token_id", None)
        if sep_id is None:
            sep_id = getattr(tok, "eos_token_id", None)
        if cls_id is None or sep_id is None:
            raise AttributeError(f"{tok.__class__.__name__} cannot add special tokens")
        packed = [cls_id, *ids_slice, sep_id]
    get_special_mask = getattr(tok, "get_special_tokens_mask", None)
    special = None
    if callable(get_special_mask):
        try:
            special = list(get_special_mask(packed, already_has_special_tokens=True))
        except (AttributeError, AssertionError, NotImplementedError, TypeError, ValueError):
            special = None
    if special is None or len(special) != len(packed):
        special = [1] + [0] * (len(packed) - 2) + [1]
    return packed, special


def run(docs, params):
    ckpt, todo, by_doc = resume_split(docs, params)
    if not todo:  # everything already checkpointed -> don't load the model
        return ordered(docs, by_doc)
    setup_started = time.perf_counter()
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    device = _device(params.get("device"))
    payload = torch.load(str(Path(params["checkpoint"]).expanduser()),
                         map_location="cpu", weights_only=False)
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    num_bio = int(state["bio_classifier.weight"].shape[0])
    num_ent = int(state["label_classifier.weight"].shape[0])
    bio_labels, ent_labels = _labels(params)
    if not ent_labels or len(ent_labels) != num_ent:
        raise ValueError(f"entity_labels ({0 if not ent_labels else len(ent_labels)}) != head size "
                         f"{num_ent}; set params.entity_labels or params.train_metrics")

    base = params.get("base_model", "DTAI-KULeuven/robbert-2023-dutch-base")
    base_kwargs = {}
    # `revision` is Hugging Face's API name. Configs expose the clearer
    # `base_model_commit`; retain `base_revision` for older configs.
    base_model_commit = params.get("base_model_commit", params.get("base_revision"))
    if base_model_commit:
        base_kwargs["revision"] = base_model_commit
    tok = AutoTokenizer.from_pretrained(
        base, use_fast=True, add_prefix_space=True, **base_kwargs
    )
    # stable character offsets for byte-level BPE (otherwise spans are off by one)
    try:
        from tokenizers.pre_tokenizers import ByteLevel
        backend = getattr(tok, "backend_tokenizer", None)
        if backend is not None and "ByteLevel" in str(getattr(backend, "pre_tokenizer", "")):
            backend.pre_tokenizer = ByteLevel(add_prefix_space=True, use_regex=True)
    except Exception:  # noqa: BLE001
        pass

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            cfg = AutoConfig.from_pretrained(base, **base_kwargs)
            self.encoder = AutoModel.from_pretrained(
                base, config=cfg, **base_kwargs
            )
            h = int(cfg.hidden_size)
            self.dropout = torch.nn.Dropout(float(getattr(cfg, "hidden_dropout_prob", 0.1)))
            self.bio_classifier = torch.nn.Linear(h, num_bio)
            self.label_classifier = torch.nn.Linear(h, num_ent)

        def forward(self, input_ids, attention_mask):
            o = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            hid = self.dropout(o.last_hidden_state)
            return self.bio_classifier(hid), self.label_classifier(hid)

    model = Model()
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"checkpoint mismatch: missing={missing} unexpected={unexpected}")
    model.to(device).eval()

    max_len = int(params.get("max_length", 512))
    overlap = int(params.get("overlap", 64))
    try:
        n_special = int(tok.num_special_tokens_to_add(pair=False))
    except (AttributeError, NotImplementedError, TypeError):
        n_special = 2  # roberta/bert <s> ... </s>; matches the manual packing fallback
    usable = max_len - n_special
    bio_o = bio_labels.index("O") if "O" in bio_labels else 0

    batch_size = _resolved_batch_size(params.get("batch_size"), device)
    batch_docs = max(batch_size, int(params.get("batch_docs", 64)))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    setup_seconds = elapsed_since(setup_started)

    def infer_unbatched(run_docs, *, record):
        results = {}
        iterator = track(run_docs, params) if record else run_docs
        for d in iterator:
            text = d["text"]
            enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, verbose=False)
            ids, offs = enc["input_ids"], [tuple(x) for x in enc["offset_mapping"]]
            n = len(ids)
            if n == 0:
                results[d["doc_id"]] = []
                if record:
                    ckpt.record(d["doc_id"], [])
                continue
            bio_sum = np.zeros((n, num_bio)); ent_sum = np.zeros((n, num_ent)); cnt = np.zeros(n)
            with torch.no_grad():
                for s, e in _windows(n, usable, overlap):
                    sl = ids[s:e]
                    packed, special = _prepare_window(tok, sl)
                    inp = torch.tensor([packed], device=device)
                    att = torch.ones_like(inp)
                    bl, ll = model(input_ids=inp, attention_mask=att)
                    mask = np.array(special, dtype=bool)
                    bio_sum[s:e] += bl[0].cpu().numpy()[~mask]
                    ent_sum[s:e] += ll[0].cpu().numpy()[~mask]
                    cnt[s:e] += 1
            cnt[cnt == 0] = 1
            pb = (bio_sum / cnt[:, None]).argmax(1)
            pe = (ent_sum / cnt[:, None]).argmax(1)
            spans = _decode(pb, pe, offs, text, bio_labels, ent_labels, bio_o)
            results[d["doc_id"]] = spans
            if record:
                ckpt.record(d["doc_id"], spans)
        return results

    # Batched accelerator path: pad+mask `batch_size` windows -- pooled
    # ACROSS docs so short docs don't waste the GPU -- into one forward pass, then
    # scatter each window's real-token logits back. Padding is masked out (att=0) and
    # sliced off, so real-token logits (hence spans) match the per-window path; only
    # the number/size of GPU calls changes. Docs are buffered `batch_docs` at a time to
    # bound RAM, and still checkpointed per doc as each completes.
    def _forward(packed_list):
        maxlen = max(len(p) for p in packed_list)
        inp = torch.full((len(packed_list), maxlen), pad_id, dtype=torch.long)
        att = torch.zeros((len(packed_list), maxlen), dtype=torch.long)
        for j, p in enumerate(packed_list):
            inp[j, :len(p)] = torch.tensor(p, dtype=torch.long); att[j, :len(p)] = 1
        with torch.no_grad():
            bl, ll = model(input_ids=inp.to(device), attention_mask=att.to(device))
        return bl.cpu().numpy(), ll.cpu().numpy()

    def _results(run_docs):
        active_batch_size = batch_size
        for gi in range(0, len(run_docs), batch_docs):
            group = run_docs[gi:gi + batch_docs]
            prepared, acc, work = [], {}, []
            for d in group:
                enc = tok(d["text"], add_special_tokens=False, return_offsets_mapping=True, verbose=False)
                ids, offs = enc["input_ids"], [tuple(x) for x in enc["offset_mapping"]]
                n = len(ids)
                prepared.append((d, n, offs))
                acc[d["doc_id"]] = [np.zeros((n, num_bio)), np.zeros((n, num_ent)), np.zeros(n)]
                for s, e in _windows(n, usable, overlap):
                    packed, special = _prepare_window(tok, ids[s:e])
                    work.append((d["doc_id"], s, e, special, packed))
            bi = 0
            while bi < len(work):
                chunk = work[bi:bi + active_batch_size]
                try:
                    bl, ll = _forward([w[4] for w in chunk])
                except RuntimeError as exc:
                    if not _is_out_of_memory(exc) or len(chunk) <= 1:
                        raise
                    active_batch_size = max(1, len(chunk) // 2)
                    _empty_device_cache(torch, device)
                    print(f"  [{params.get('_label') or 'robbert'}] {device.type} memory limit; "
                          f"retrying with batch_size={active_batch_size}",
                          file=sys.stderr, flush=True)
                    continue
                for j, (doc_id, s, e, special, packed) in enumerate(chunk):
                    keep = ~np.array(special, dtype=bool)
                    acc[doc_id][0][s:e] += bl[j, :len(packed)][keep]
                    acc[doc_id][1][s:e] += ll[j, :len(packed)][keep]
                    acc[doc_id][2][s:e] += 1
                bi += len(chunk)
            for d, n, offs in prepared:
                bio_sum, ent_sum, cnt = acc[d["doc_id"]]
                if n == 0:
                    yield d["doc_id"], []
                    continue
                cnt[cnt == 0] = 1
                pb = (bio_sum / cnt[:, None]).argmax(1)
                pe = (ent_sum / cnt[:, None]).argmax(1)
                yield d["doc_id"], _decode(pb, pe, offs, d["text"], bio_labels, ent_labels, bio_o)

    def infer_batched(run_docs, *, record):
        results = {}
        iterator = _results(run_docs)
        if record:
            iterator = track(iterator, params, total=len(run_docs))
        for doc_id, spans in iterator:
            results[doc_id] = spans
            if record:
                ckpt.record(doc_id, spans)
        return results

    infer = infer_unbatched if batch_size <= 1 else infer_batched
    if batch_size > 1:
        print(f"  [{params.get('_label') or 'robbert'}] {device.type} inference: "
              f"batch_size={batch_size}, batch_docs={batch_docs}", file=sys.stderr, flush=True)

    warm_docs = warmup_docs(params, todo, default=max(1, batch_size))
    _, warmup_seconds = measure(
        lambda: infer(warm_docs, record=False), device=device
    )
    inferred, inference_seconds = measure(
        lambda: infer(todo, record=True), device=device
    )
    by_doc.update(inferred)
    write_report(
        params,
        setup_seconds=setup_seconds,
        warmup_seconds=warmup_seconds,
        inference_seconds=inference_seconds,
        warmup_documents=len(warm_docs),
    )
    return ordered(docs, by_doc)
