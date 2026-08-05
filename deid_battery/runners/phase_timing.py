"""Small, dependency-free timing contract shared by every model runner.

Runners write their internal phases to the path injected as ``_timing_path`` by
the orchestrator.  A path instead of a mutable return object keeps the contract
working unchanged when a runner lives in another Python environment.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Iterable


def elapsed_since(started: float) -> float:
    return max(0.0, time.perf_counter() - started)


def warmup_docs(params: dict, docs: Iterable[dict], *, default: int) -> list[dict]:
    """Select non-empty documents for an unrecorded warm-up pass."""
    count = max(0, int(params.get("warmup_docs", default) or 0))
    if count == 0:
        return []
    selected = [doc for doc in docs if str(doc.get("text", "")).strip()]
    return selected[:count]


def synchronize(device=None) -> None:
    """Wait for accelerator work at timing boundaries; CPU is a no-op."""
    if device is None:
        return
    device_type = str(getattr(device, "type", device)).lower()
    try:
        import torch

        if device_type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
        elif device_type == "mps" and hasattr(torch, "mps"):
            torch.mps.synchronize()
    except (AttributeError, RuntimeError):
        return


def measure(call: Callable[[], object], *, device=None) -> tuple[object, float]:
    synchronize(device)
    started = time.perf_counter()
    result = call()
    synchronize(device)
    return result, elapsed_since(started)


def write_report(
    params: dict,
    *,
    setup_seconds: float,
    warmup_seconds: float,
    inference_seconds: float,
    warmup_documents: int,
    service_setup: str = "local_measured",
) -> None:
    """Atomically persist runner phases when orchestration requested them."""
    raw_path = params.get("_timing_path")
    if not raw_path:
        return
    path = Path(raw_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup_seconds": round(max(0.0, float(setup_seconds)), 6),
        "warmup_seconds": round(max(0.0, float(warmup_seconds)), 6),
        "inference_seconds": round(max(0.0, float(inference_seconds)), 6),
        "warmup_documents": max(0, int(warmup_documents)),
        "service_setup": str(service_setup),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_report(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def combine_reports(reports: Iterable[dict]) -> dict:
    """Aggregate repeated model loads, as used by chunked Deidentify."""
    rows = [dict(row) for row in reports if row]
    if not rows:
        return {}
    return {
        "setup_seconds": sum(float(row.get("setup_seconds", 0.0)) for row in rows),
        "warmup_seconds": sum(float(row.get("warmup_seconds", 0.0)) for row in rows),
        "inference_seconds": sum(float(row.get("inference_seconds", 0.0)) for row in rows),
        "warmup_documents": sum(int(row.get("warmup_documents", 0)) for row in rows),
        "service_setup": rows[0].get("service_setup", "local_measured"),
    }
