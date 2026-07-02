"""Per-method run-time store for the time-vs-recall plot.

A single, human-editable YAML file (path from ``timings:`` in the battery config,
default ``timings.yaml`` at the repo root) maps each model id to a list of timing
entries::

    uza:
      - {device: cpu, seconds: 240.1, source: measured, n_docs: 300}
      - {device: gpu, seconds: 18.0, source: manual, note: "RTX 4090"}
    deduce:
      - {device: cpu, seconds: 12.3, source: measured, n_docs: 300}

- ``measured`` rows are written by the orchestrator after each successful
  inference -- one per device (``cpu``/``gpu``), set to that pass's wall-clock
  (re-running overwrites it). A crash-then-resume records only the completing
  session's time; re-run from scratch for an exact number, or hand-edit the row.
- ``manual`` rows are added by hand -- e.g. the same method timed on a GPU
  elsewhere -- and are NEVER touched by the orchestrator. This is how one method
  gets both a CPU and a GPU dot in the plot. A row is auto-managed ONLY if it
  explicitly says ``source: measured``; a hand-added row (``source: manual`` or no
  ``source:`` at all) is protected regardless of its device, so a CPU run never
  overwrites your hand-added GPU time.

The file lives OUTSIDE ``out/`` on purpose: ``out/`` is regenerated (and
gitignored), so hand-added rows would be lost there. Times are score-only (no
patient text), so the file is safe to keep/commit.
"""
from __future__ import annotations

from pathlib import Path

import yaml

# device strings that mean "an accelerator" -> collapsed to the "gpu" bucket.
# Anything else (e.g. a custom "a100" label on a manual row) is kept verbatim and
# gets its own colour in the plot.
_GPU_ALIASES = {"cuda", "gpu", "mps", "metal", "rocm", "xpu"}


def normalize_device(value: object) -> str:
    v = str(value or "cpu").strip().lower()
    if v in _GPU_ALIASES:
        return "gpu"
    return v or "cpu"


def measured_device(model_cfg: dict, params: dict) -> str:
    """Device label for an auto-measured row. An explicit ``device_label:`` on the
    model config wins (needed for the LLM runner, whose compute is a *remote* GPU
    the local ``device`` param can't describe); otherwise fall back to the
    resolved ``device`` param."""
    label = (model_cfg or {}).get("device_label")
    if label:
        return normalize_device(label)
    return normalize_device((params or {}).get("device", "cpu"))


def load(path: str | Path) -> dict[str, list[dict]]:
    """Read the timings file into ``{model_id: [entry, ...]}``. Missing file or
    malformed rows degrade to ``{}`` / are skipped rather than raising."""
    p = Path(path)
    if not p.exists():
        return {}
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    out: dict[str, list[dict]] = {}
    if isinstance(data, dict):
        for model_id, entries in data.items():
            if isinstance(entries, list):
                out[str(model_id)] = [dict(e) for e in entries if isinstance(e, dict)]
    return out


def save(path: str | Path, data: dict[str, list[dict]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True, default_flow_style=False),
        encoding="utf-8",
    )


def is_measured(entry: dict) -> bool:
    """True only for rows the orchestrator auto-manages. A row counts as measured
    iff it *explicitly* says ``source: measured``; anything else -- a hand-added
    row, whether it says ``source: manual`` or omits ``source:`` entirely -- is
    treated as manual and is never matched or overwritten."""
    return (entry or {}).get("source") == "measured"


def measured_seconds(entries: list[dict]) -> float | None:
    """The first auto-``measured`` row's seconds (what summary.csv reports), or None
    (e.g. a method whose only time was hand-added)."""
    for e in entries or []:
        if is_measured(e) and e.get("seconds") is not None:
            return float(e["seconds"])
    return None


def record_measured(path: str | Path, model_id: str, device: str, seconds: float,
                    n_docs: int | None = None) -> None:
    """Upsert the auto-``measured`` row for ``(model_id, device)`` to this pass's
    wall-clock (replace if it exists, else append).

    ONLY an explicit ``source: measured`` row for the SAME device is ever touched.
    Manual rows (``source: manual`` or no ``source:``) and measured rows for other
    devices are preserved verbatim -- so running on CPU never disturbs a hand-added
    GPU time, and vice versa.

    Called only after a *successful* full inference, so ``seconds`` is the time of
    the pass that produced the current raw.jsonl. A crash-then-resume records only
    the completing session's time (an undercount vs. one clean pass) -- re-run from
    scratch for an exact number, or hand-edit the row."""
    device = normalize_device(device)
    seconds = round(float(seconds), 2)
    data = load(path)
    rows = data.setdefault(str(model_id), [])
    for r in rows:
        if is_measured(r) and normalize_device(r.get("device")) == device:
            r["device"] = device
            r["seconds"] = seconds
            if n_docs is not None:
                r["n_docs"] = int(n_docs)
            break
    else:
        row: dict[str, object] = {"device": device, "seconds": seconds, "source": "measured"}
        if n_docs is not None:
            row["n_docs"] = int(n_docs)
        rows.append(row)
    save(path, data)
