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
    # No TTY (journald / nohup logfile / subprocess pipe): tqdm would draw a
    # *disabled* (silent) bar, leaving long runs looking frozen. Fall back to
    # periodic newline-terminated counter lines that are readable in logs.
    if not sys.stderr.isatty():
        return _counter(iterable, label, total)
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=label or None, total=total, unit=unit,
                    leave=False, disable=None)  # disable=None -> auto-off on non-TTY
    except Exception:  # noqa: BLE001 -- tqdm absent: lightweight fallback
        return _counter(iterable, label, total)


def _counter(iterable, label, total):
    pre = f"[{label}] " if label else ""
    tty = sys.stderr.isatty()
    every = max(1, (total or 100) // 20)
    i = 0
    for i, item in enumerate(iterable, 1):
        if total and (i % every == 0 or i == total):
            # TTY: overwrite one line with \r. Non-TTY (logs): clean newline lines.
            lead, end = ("\r", "") if tty else ("", "\n")
            print(f"{lead}{pre}{i}/{total}", end=end, file=sys.stderr, flush=True)
        yield item
    if total and i and tty:
        print("", file=sys.stderr)
