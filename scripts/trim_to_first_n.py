"""One-time helper: trim the whole run down to the first N documents.

If ``input.jsonl`` (and therefore the outputs generated from it) ended up with
more documents than you wanted, this rewrites every per-document file in place so
they all cover the *same* first N docs and stay aligned:

  - the raw source ``results.jsonl`` (matched on its ``id`` field),
  - the battery input ``input.jsonl`` (kept: first N lines),
  - every ``out/**/*.jsonl`` already generated (filtered to those N doc_ids):
    each model's ``raw*.jsonl`` / ``by_doc.*.jsonl`` and ``out/_predictions/*.jsonl``.

The doc set is defined by the first N lines of ``input.jsonl``; every other file
is filtered to that set of ``doc_id`` values, so partially-run / re-run models
stay consistent even if a file is missing rows. Non-JSONL eval artifacts
(summary.csv, *.png, quantity_payload.json) are left untouched -- re-run
evaluate to regenerate them.

Dry-run by default; pass --apply to actually rewrite. A ``.bak`` copy of each
changed file is kept unless you pass --no-backup.

  python scripts/trim_to_first_n.py                      # dry-run, defaults (n=100)
  python scripts/trim_to_first_n.py --apply              # do it
  python scripts/trim_to_first_n.py --n 100 --out out --results ~/results.jsonl --apply
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _read_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as f:
        return [ln for ln in f if ln.strip()]


def _doc_id(line: str) -> str | None:
    try:
        r = json.loads(line)
    except json.JSONDecodeError:
        return None
    v = r.get("doc_id") or r.get("id") or r.get("document_id")
    return str(v) if v is not None else None


def _write(path: Path, lines: list[str], *, apply: bool, backup: bool) -> None:
    if not apply:
        return
    if backup and path.exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln if ln.endswith("\n") else ln + "\n")
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=100, help="keep the first N documents (default 100)")
    ap.add_argument("--input", default="input.jsonl", help="battery input JSONL (defines the doc set)")
    ap.add_argument("--out", default="out", help="battery output_dir to trim (default: out)")
    ap.add_argument("--results", default="~/results.jsonl",
                    help="raw source JSONL to also trim; '' to skip (default: ~/results.jsonl)")
    ap.add_argument("--apply", action="store_true", help="actually rewrite files (default: dry-run)")
    ap.add_argument("--no-backup", dest="backup", action="store_false", help="don't keep .bak copies")
    a = ap.parse_args()

    inp = Path(a.input)
    if not inp.exists():
        raise SystemExit(f"{inp}: not found")

    # The doc set is the first N lines of input.jsonl.
    in_lines = _read_lines(inp)
    keep_lines = in_lines[: a.n]
    keep_ids = {i for ln in keep_lines if (i := _doc_id(ln)) is not None}
    print(f"[def] {inp}: {len(in_lines)} -> {len(keep_lines)} docs "
          f"({len(keep_ids)} distinct doc_ids define the keep-set)")

    # Collect every file to trim: input.jsonl (first N lines), the raw source
    # (matched by id), and every out/**/*.jsonl (filtered to keep_ids).
    plan: list[tuple[Path, list[str]]] = [(inp, keep_lines)]

    if a.results:
        res = Path(os.path.expanduser(a.results))
        if res.exists():
            rl = _read_lines(res)
            kept = [ln for ln in rl if _doc_id(ln) in keep_ids]
            if not kept and rl:  # id field doesn't line up -> fall back to first N lines
                kept = rl[: a.n]
                print(f"  ! {res}: no id overlap with keep-set; falling back to first {a.n} lines")
            plan.append((res, kept))
        else:
            print(f"  - {res}: not found, skipping")

    out = Path(a.out)
    for f in sorted(out.rglob("*.jsonl")) if out.exists() else []:
        lines = _read_lines(f)
        plan.append((f, [ln for ln in lines if _doc_id(ln) in keep_ids]))

    changed = 0
    for path, kept in plan:
        before = len(_read_lines(path)) if path.exists() else 0
        after = len(kept)
        mark = "" if before == after else "  <-- trim"
        if before != after:
            changed += 1
        print(f"  {before:>5} -> {after:>5}  {path}{mark}")
        _write(path, kept, apply=a.apply, backup=a.backup)

    if a.apply:
        print(f"\nDone. Rewrote {changed} file(s)"
              + ("" if not a.backup else " (.bak copies kept). Delete them once you've confirmed."))
        print("Re-run the remaining models + evaluate; new models will pick up the trimmed input.jsonl.")
    else:
        print(f"\nDry-run: {changed} file(s) would change. Re-run with --apply to do it.")


if __name__ == "__main__":
    main()
