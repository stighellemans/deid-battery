"""LLM runner via any OpenAI-compatible chat endpoint.

Works with vLLM (GPU), llama.cpp (CPU), or any hosted server -- set base_url.
On the review VM (data must stay local) point base_url at a llama.cpp server on
the VM itself. Output is grammar-constrained JSON (response_format json_schema)
and spans are aligned to char offsets by EXACT literal matching (the prompt asks
for verbatim copies), with guards against repetition-loop artifacts.

config::
    - id: qwen
      runner: llm
      params:
        base_url: http://127.0.0.1:8089/v1
        model: qwen3-8b
        prompt_dir: prompts          # dict_prompt.txt, dict_example.txt, labels.csv
        temperature: 0.6             # NOT 0.0 -- greedy loops on some models
        thinking: true               # default; see note below
        reasoning: {effort: medium}  # optional, for OpenRouter-style reasoning enable
        workers: 2

IMPORTANT -- thinking matters a lot for Qwen DEID quality. With `thinking: true`
(default) the runner does NOT send a strict json-schema grammar and does NOT add
`/no_think` (a grammar would block the `<think>` block, and on llama.cpp the two
cannot coexist); the model reasons, then emits JSON, which is parsed leniently
(reasoning block stripped, truncated JSON salvaged). Empirically this cuts the
false-positive burden ~14x vs no-think. Set `thinking: false` only for endpoints
where you want grammar-constrained output and the model does not reason.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from ..progress import track
from ..schema import make_span

DELIM = "Originele tekst:"


def _load_assets(prompt_dir):
    p = Path(prompt_dir)
    template = (p / "dict_prompt.txt").read_text(encoding="utf-8")
    example = (p / "dict_example.txt").read_text(encoding="utf-8")
    labels = {}
    for line in (p / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        if "," in line:
            k = line.split(",", 1)[0].strip()
            if k:
                labels[k] = line.split(",", 1)[1].strip().strip('"')
    return template, example, labels


def _prompts(text, labels, template, example):
    system_tpl, user_tpl = template.split(DELIM)
    user_tpl = DELIM + user_tpl
    label_str = ",\n".join(f"* {k} - {v}" for k, v in labels.items())
    return system_tpl.format(labels=label_str.strip(), example=example), user_tpl.format(text=text)


def _schema(labels):
    return {"type": "json_schema", "json_schema": {"name": "annotation_response", "strict": True, "schema": {
        "type": "object", "additionalProperties": False, "required": ["spans"], "properties": {"spans": {
            "type": "array", "items": {"type": "object", "additionalProperties": False,
                "required": ["annotated_text", "label"], "properties": {
                    "annotated_text": {"type": "string"},
                    "label": {"type": "string", "enum": labels}}}}}}}}


def _annotate(base_url, system, user, rf, model, temperature, top_p, max_tokens,
              thinking, reasoning, attempts=4):
    # thinking: no /no_think, no grammar (a grammar blocks the <think> block).
    sys_content = system if thinking else system + "\n/no_think"
    payload = {"model": model,
               "messages": [{"role": "system", "content": sys_content}, {"role": "user", "content": user}],
               "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens}
    if not thinking:
        payload["response_format"] = rf            # strict grammar only when not thinking
    if reasoning is not None:
        payload["reasoning"] = reasoning           # OpenRouter-style reasoning enable
    last = None
    for i in range(attempts):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=payload, timeout=600)
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(2 ** i, 15))
    raise last


def _salvage_spans(s):
    """Decode the complete elements of a (possibly truncated) spans array."""
    start = s.find("[")
    if start < 0:
        return []
    dec = json.JSONDecoder()
    out, k, n = [], start + 1, len(s)
    while k < n:
        while k < n and s[k] in " \t\r\n,":
            k += 1
        if k < n and s[k] == "]":
            break
        try:
            obj, k = dec.raw_decode(s, k)
            out.append(obj)
        except json.JSONDecodeError:
            break
    return out


def _parse(content):
    s = content.strip()
    if "</think>" in s:                 # drop a leading reasoning block
        s = s.split("</think>", 1)[1].strip()
    if s.startswith("```"):
        s = s.split("\n", 1)[1].rsplit("```", 1)[0]
    i = s.find("{")
    if i > 0:
        s = s[i:]
    try:
        data = json.loads(s)
        spans = data.get("spans", [])
        return spans if isinstance(spans, list) else []
    except json.JSONDecodeError:        # truncated / soft JSON -> salvage
        return _salvage_spans(s)


def _extract(text, raw):
    out, used = [], set()
    for a in raw:
        phrase = (a.get("annotated_text") or "").strip()
        label = a.get("label") or ""
        if len(phrase) < 3 or not any(c.isalnum() for c in phrase) or not label:
            continue
        hits, start = [], 0
        while True:
            i = text.find(phrase, start)
            if i < 0:
                break
            hits.append((i, i + len(phrase)))
            start = i + len(phrase)
        if len(hits) > 10:  # repetition-loop / common-substring guard
            continue
        for b, e in hits:
            if (b, e) not in used:
                used.add((b, e))
                out.append(make_span(b, e, label, text[b:e]))
    return out


def run(docs, params):
    base_url = params["base_url"].rstrip("/")
    model = params.get("model", "model")
    template, example, labels = _load_assets(params["prompt_dir"])
    rf = _schema(list(labels))
    temperature = params.get("temperature", 0.6)
    top_p = params.get("top_p", 0.95)
    max_tokens = params.get("max_tokens", 8192 if params.get("thinking", True) else 6144)
    workers = params.get("workers", 2)
    thinking = params.get("thinking", True)
    reasoning = params.get("reasoning")

    def work(d):
        try:
            system, user = _prompts(d["text"], labels, template, example)
            content = _annotate(base_url, system, user, rf, model, temperature, top_p,
                                max_tokens, thinking, reasoning)
            return d["doc_id"], _extract(d["text"], _parse(content)), None
        except Exception as e:  # noqa: BLE001
            return d["doc_id"], [], f"{type(e).__name__}: {str(e)[:100]}"

    by_doc, fails = {}, []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(work, d) for d in docs]
        for fut in track(as_completed(futures), params, total=len(docs)):
            doc_id, spans, err = fut.result()
            by_doc[doc_id] = sorted(spans, key=lambda s: (s["begin"], s["end"]))
            if err:
                fails.append((doc_id, err))
    if fails:
        print(f"  [llm] {len(fails)} docs failed; first: {fails[0]}")
    if fails and len(fails) == len(docs):  # endpoint unreachable -> fail the model, don't emit phantom spans
        raise RuntimeError(f"all {len(docs)} docs failed (endpoint unreachable?): {fails[0][1]}")
    return by_doc
