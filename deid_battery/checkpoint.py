"""Per-document checkpoint for the expensive runners (deidentify, llm).

Each finished document's spans are appended to a sidecar JSONL the moment the
document completes, so an interrupted run (Ctrl-C, OOM, endpoint drop) resumes
without recomputing docs that already succeeded. The file uses the exact same
one-line ``{doc_id, num_entities, entities}`` shape as a ``raw.jsonl``, so it is
literally a partial of the final output; the orchestrator writes the finished
``raw.jsonl`` and deletes the checkpoint on success, and leaves it in place on
failure so the next run continues where it stopped.

The path is injected by the orchestrator as ``params['_checkpoint']`` (a stable
location under the model's output dir, e.g. ``out/<model>/raw.partial.jsonl``).
When it is absent every operation is a no-op, so a runner behaves exactly as it
did before checkpointing existed. Runners that don't stream per-doc simply never
construct one.
"""
from __future__ import annotations

import json
import threading
from pathlib import Path

from .progress import track


class Checkpoint:
    """Append-only, per-document resume log. Safe to share across threads
    (the llm runner records from its collector loop); reads are done once up
    front. Disabled (all methods no-op) when ``path`` is falsy."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.done: dict[str, list[dict]] = {}
        self._lock = threading.Lock()
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue  # tolerate a torn final line from a hard kill
                doc_id = row.get("doc_id")
                if doc_id is not None:
                    self.done[str(doc_id)] = row.get("entities", [])

    def has(self, doc_id) -> bool:
        return str(doc_id) in self.done

    def get(self, doc_id) -> list[dict]:
        return self.done.get(str(doc_id), [])

    def record(self, doc_id, entities: list[dict]) -> None:
        """Persist one finished document. Flushed immediately so it survives a
        killed process. No-op when checkpointing is disabled."""
        if self.path is None:
            return
        line = json.dumps({"doc_id": doc_id, "num_entities": len(entities),
                           "entities": entities}, ensure_ascii=False)
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
            self.done[str(doc_id)] = entities

    @property
    def n_done(self) -> int:
        return len(self.done)


def resume_split(docs, params, label=None):
    """Prepare a per-document runner for a checkpointed/resumable loop.

    Returns ``(ckpt, todo, by_doc)`` where ``by_doc`` is pre-seeded with the docs
    already in the checkpoint and ``todo`` is the docs still to process. The
    canonical use lets a runner skip loading its (expensive) model entirely on a
    full resume::

        ckpt, todo, by_doc = resume_split(docs, params)
        if not todo:
            return ordered(docs, by_doc)
        ... load model ...
        for d in track(todo, params):
            spans = ...
            by_doc[d["doc_id"]] = spans
            ckpt.record(d["doc_id"], spans)
        return ordered(docs, by_doc)
    """
    ckpt = Checkpoint(params.get("_checkpoint"))
    todo = [d for d in docs if not ckpt.has(d["doc_id"])]
    by_doc = {d["doc_id"]: ckpt.get(d["doc_id"]) for d in docs if ckpt.has(d["doc_id"])}
    if by_doc and not params.get("_inproc"):  # a chunk subprocess logs nothing; its parent does
        lbl = label or (params.get("_label") if isinstance(params, dict) else "") or ""
        print(f"  [{lbl}] resume: {len(by_doc)}/{len(docs)} docs already done", flush=True)
    return ckpt, todo, by_doc


def ordered(docs, by_doc):
    """Spans for every input doc, in input order (missing -> [])."""
    return {d["doc_id"]: by_doc.get(d["doc_id"], []) for d in docs}


__all__ = ["Checkpoint", "resume_split", "ordered", "track"]
