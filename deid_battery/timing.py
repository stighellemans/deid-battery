"""Per-method phase-timing store for the time-vs-recall plot.

A single, human-editable YAML file (path from ``timings:`` in the battery config,
default ``timings.yaml`` at the repo root) maps each model id to a list of timing
entries::

    uza:
      - device: cpu
        seconds: 210.4
        source: measured
        timing_scope: warm_end_to_end
        setup_seconds: 29.7
        warmup_seconds: 0.8
        inference_seconds: 205.1
        postprocess_seconds: 4.6
        cold_end_to_end_seconds: 240.1
        n_docs: 300
      - {device: gpu, seconds: 18.0, source: manual, note: "RTX 4090"}
    deduce:
      - {device: cpu, seconds: 12.3, source: measured, n_docs: 300}

- ``measured`` rows are written after inference and shared post-processing. The
  primary ``seconds`` value is warm end-to-end time: resident inference, raw
  output, post-processing, and final output. Setup, warm-up, cold end-to-end, and
  the complete measured invocation are retained as separate fields.
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
_MEASURED_DETAIL_FIELDS = {
    "setup_seconds",
    "warmup_seconds",
    "warmup_documents",
    "inference_seconds",
    "raw_write_seconds",
    "postprocess_seconds",
    "cold_overhead_seconds",
    "warm_end_to_end_seconds",
    "cold_end_to_end_seconds",
    "measured_full_run_seconds",
    "timing_scope",
    "service_setup",
}


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


def measured_value(entries: list[dict], key: str) -> object | None:
    """Return one field from the first auto-measured row."""
    for entry in entries or []:
        if is_measured(entry) and entry.get(key) is not None:
            return entry[key]
    return None


def compose_phase_metrics(
    runner_seconds: float,
    raw_write_seconds: float,
    postprocess_seconds: float,
    runner_report: dict | None,
) -> dict:
    """Combine internal runner phases with orchestration and output phases."""
    report = dict(runner_report or {})
    if report:
        setup = max(0.0, float(report.get("setup_seconds", 0.0)))
        warmup = max(0.0, float(report.get("warmup_seconds", 0.0)))
        inference = max(0.0, float(report.get("inference_seconds", 0.0)))
        cold_overhead = max(0.0, float(runner_seconds) - setup - warmup - inference)
        service_setup = str(report.get("service_setup", "local_measured"))
        warmup_documents = max(0, int(report.get("warmup_documents", 0)))
    else:
        setup = warmup = cold_overhead = 0.0
        inference = max(0.0, float(runner_seconds))
        service_setup = "not_applicable"
        warmup_documents = 0

    raw_write = max(0.0, float(raw_write_seconds))
    postprocess = max(0.0, float(postprocess_seconds))
    warm_end_to_end = inference + raw_write + postprocess
    cold_end_to_end = setup + cold_overhead + warm_end_to_end
    measured_full = float(runner_seconds) + raw_write + postprocess
    return {
        "setup_seconds": setup,
        "warmup_seconds": warmup,
        "warmup_documents": warmup_documents,
        "inference_seconds": inference,
        "raw_write_seconds": raw_write,
        "postprocess_seconds": postprocess,
        "cold_overhead_seconds": cold_overhead,
        "warm_end_to_end_seconds": warm_end_to_end,
        "cold_end_to_end_seconds": cold_end_to_end,
        "measured_full_run_seconds": measured_full,
        "timing_scope": "warm_end_to_end",
        "service_setup": service_setup,
    }


def record_measured(path: str | Path, model_id: str, device: str, seconds: float,
                    n_docs: int | None = None, details: dict | None = None) -> None:
    """Upsert the auto-``measured`` row for ``(model_id, device)`` to this pass's
    wall-clock (replace if it exists, else append).

    ONLY an explicit ``source: measured`` row for the SAME device is ever touched.
    Manual rows (``source: manual`` or no ``source:``) and measured rows for other
    devices are preserved verbatim -- so running on CPU never disturbs a hand-added
    GPU time, and vice versa.

    Called only after successful inference and post-processing. A crash-then-resume
    still undercounts processing phases; re-run from scratch for exact numbers."""
    device = normalize_device(device)
    seconds = round(float(seconds), 2)
    data = load(path)
    rows = data.setdefault(str(model_id), [])
    target = None
    for r in rows:
        if is_measured(r) and normalize_device(r.get("device")) == device:
            target = r
            break
    else:
        target = {"source": "measured"}
        rows.append(target)

    target["device"] = device
    target["seconds"] = seconds
    if n_docs is not None:
        target["n_docs"] = int(n_docs)
    for key in _MEASURED_DETAIL_FIELDS:
        target.pop(key, None)
    for key, value in (details or {}).items():
        if key not in _MEASURED_DETAIL_FIELDS or value is None:
            continue
        if key.endswith("_seconds"):
            target[key] = round(float(value), 2)
        elif key == "warmup_documents":
            target[key] = int(value)
        else:
            target[key] = value
    save(path, data)
