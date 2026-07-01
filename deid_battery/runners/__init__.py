"""Runner registry.

Each runner module exposes ``run(docs, params) -> {doc_id: [span, ...]}`` where
``docs`` is a list of ``{doc_id, text, _meta, annotations}`` (``_meta`` is the
resolved per-doc metadata). Runners are imported lazily, so a missing optional
dependency only breaks the runner that needs it.
"""
from __future__ import annotations

import importlib

_RUNNERS = {
    "deduce": "deid_battery.runners.deduce",
    "gliner": "deid_battery.runners.gliner",
    "hf_token": "deid_battery.runners.hf_token",
    "robbert": "deid_battery.runners.robbert",
    "llm": "deid_battery.runners.llm",
    "deidentify": "deid_battery.runners.deidentify",
}

# Runners that consume per-doc metadata at INFERENCE time (Channel 1), so their
# raw output changes with the metadata setting and must be re-run per metadata
# condition. Every other runner is metadata-independent at inference (metadata
# only affects shared post-processing, Channel 2), so its raw is reused across
# conditions. Kept as a plain name set so checking it needs no heavy import.
_METADATA_AT_INFERENCE = {"deduce"}


def get_runner(name: str):
    if name not in _RUNNERS:
        raise ValueError(f"unknown runner {name!r}; available: {sorted(_RUNNERS)}")
    return importlib.import_module(_RUNNERS[name]).run


def uses_metadata_at_inference(name: str) -> bool:
    return name in _METADATA_AT_INFERENCE
