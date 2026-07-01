"""Tiny per-runner progress bar, shared by the runners.

Uses tqdm when available (it ships as a transformers dependency, so it's normally
present) and degrades to a light stderr counter otherwise -- no hard dependency.
The bar is labelled with the model id, which the orchestrator injects into the
runner ``params`` as ``_label``.
"""
from __future__ import annotations

import sys


def _label_of(params) -> str:
    if isinstance(params, dict):
        return params.get("_label") or ""
    return params or ""


def track(iterable, params=None, total=None, unit="doc"):
    """Wrap an iterable with a per-item progress bar labelled by the model id.

    ``params`` may be the runner params dict (the label is read from
    ``params['_label']``) or a plain label string. ``total`` is inferred from
    ``len(iterable)`` when possible (pass it explicitly for generators, e.g.
    ``as_completed(...)``).
    """
    label = _label_of(params)
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=label or None, total=total, unit=unit,
                    leave=False, disable=None)  # disable=None -> auto-off on non-TTY
    except Exception:  # noqa: BLE001 -- tqdm absent: lightweight fallback
        return _counter(iterable, label, total)


def _counter(iterable, label, total):
    pre = f"[{label}] " if label else ""
    every = max(1, (total or 100) // 20)
    i = 0
    for i, item in enumerate(iterable, 1):
        if total and (i % every == 0 or i == total):
            print(f"\r{pre}{i}/{total}", end="", file=sys.stderr, flush=True)
        yield item
    if total and i:
        print("", file=sys.stderr)
