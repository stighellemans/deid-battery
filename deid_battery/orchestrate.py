"""deid-battery orchestrator.

    python -m deid_battery.orchestrate run --config configs/battery.yaml

Three independent stages per model:
  1. inference   -> raw.jsonl   (expensive; gated by --only/--skip-existing/--no-run)
  2. post-process-> by_doc.jsonl (cheap; always re-derived from raw)
  3. evaluate + plot over every source on disk.

Optional `conditions:` runs every model under several metadata settings
(e.g. no-metadata vs with-metadata) as sources "<id>@<condition>". The
expensive Stage-1 raw is shared across conditions for every runner except the
metadata-at-inference ones (deduce), so the extra condition is nearly free.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import yaml

from . import metadata as md_mod
from .inputs import load_input
from .postprocess import post_process
from .runners import get_runner, uses_metadata_at_inference
from .schema import read_by_doc, write_by_doc

_PKG_PARENT = str(Path(__file__).resolve().parents[1])


def _run_in_venv(venv, runner, params, docs, work) -> dict:
    venv = Path(venv).expanduser()
    py = venv / "bin" / "python"
    py = py if py.exists() else venv  # allow passing a python path directly
    tmp = Path(tempfile.mkdtemp(dir=work))
    try:
        docs_p, params_p, out_p = tmp / "docs.jsonl", tmp / "params.json", tmp / "by_doc.jsonl"
        with open(docs_p, "w", encoding="utf-8") as f:
            for d in docs:
                f.write(json.dumps({"doc_id": d["doc_id"], "text": d["text"],
                                    "_meta": d.get("_meta") or {}}, ensure_ascii=False) + "\n")
        params_p.write_text(json.dumps(params), encoding="utf-8")
        env = {**os.environ, "PYTHONPATH": _PKG_PARENT}
        subprocess.run([str(py), "-m", "deid_battery.runners._worker", runner,
                        str(params_p), str(docs_p), str(out_p)], check=True, env=env)
        return read_by_doc(out_p)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _postprocess(raw_by_doc, texts, metas, pp_cfg):
    """Pure transform raw spans -> eval-facing spans. Cheap and deterministic,
    so it can be re-applied without re-running the (expensive) model."""
    if not pp_cfg.get("enabled", True):
        return raw_by_doc
    return {doc_id: post_process(spans, texts.get(doc_id, ""), metas.get(doc_id), pp_cfg, doc_id)
            for doc_id, spans in raw_by_doc.items()}


def _conditions(cfg):
    """Benchmark conditions = named metadata (and optional postprocess) settings.
    Each condition inherits the top-level `metadata:` as its base and overrides
    it per key, so the shared part (e.g. the `source`) is written once. With no
    `conditions:` block, a single unnamed condition from the top-level `metadata:`
    -> identical to single-condition behaviour (no filename suffix; source id =
    model id)."""
    base_md = cfg.get("metadata", {"source": "none"})
    conds = cfg.get("conditions")
    if not conds:
        return [{"id": "", "name": "", "metadata": base_md, "postprocess": None}]
    out = []
    for c in conds:
        cid = str(c.get("id") or c.get("name") or "")
        out.append({"id": cid, "name": c.get("name", cid),
                    "metadata": {**base_md, **(c.get("metadata") or {})},
                    "postprocess": c.get("postprocess")})
    return out


def _suffix(cid):
    return f".{cid}" if cid else ""


def _src_id(mid, cid):
    return f"{mid}@{cid}" if cid else mid


def _raw_path(out, mid, runner, cid):
    """A per-condition raw only for inference-metadata-sensitive runners (deduce);
    every other runner shares one condition-independent raw.jsonl."""
    if cid and uses_metadata_at_inference(runner):
        return out / mid / f"raw{_suffix(cid)}.jsonl"
    return out / mid / "raw.jsonl"


def _docs_with_meta(docs, metas):
    return [{**d, "_meta": (metas or {}).get(d["doc_id"])} for d in docs]


def run(config_path, only=None, skip_existing=False, no_run=False):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out = Path(cfg.get("output_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)

    docs = load_input(cfg["input"])
    conditions = _conditions(cfg)
    multi = len(conditions) > 1
    # resolve per-doc metadata once per condition
    metas_by_cond = {c["id"]: {d["doc_id"]: md_mod.resolve(d, c["metadata"]) for d in docs}
                     for c in conditions}

    device = cfg.get("device")
    pp_cfg = cfg.get("postprocess", {"enabled": True})
    texts = {d["doc_id"]: d["text"] for d in docs}
    only = set(only) if only else None
    if multi:
        print(f"conditions: {', '.join(c['id'] for c in conditions)}", flush=True)

    # --- Stage 1: inference -> raw (the expensive step; gated by flags). Raw is
    # shared across conditions, EXCEPT metadata-at-inference runners (deduce),
    # which get one raw per condition. Persisted *before* post-processing so
    # post-processing/metadata-Channel-2 can be re-derived (Stage 2) cheaply. ---
    failed = []
    for m in cfg["models"]:
        mid, runner = m["id"], m["runner"]
        sensitive = uses_metadata_at_inference(runner)
        per_cond = sensitive and multi
        for c in (conditions if per_cond else conditions[:1]):
            cid = c["id"]
            raw_p = _raw_path(out, mid, runner, cid)
            tag = _src_id(mid, cid) if per_cond else mid
            if no_run:
                continue
            if only is not None and mid not in only:
                continue
            if skip_existing and raw_p.exists():
                print(f"[{tag}] skip inference (raw exists)", flush=True)
                continue
            params = dict(m.get("params") or {})
            if device and "device" not in params:
                params["device"] = device
            params["_label"] = tag  # progress-bar label (see deid_battery.progress.track)
            # Per-document checkpoint: the expensive runners (deidentify, llm) append
            # each finished doc here and resume from it after an interruption. It is
            # promoted to raw.jsonl and removed on success, and left in place on
            # failure so the next run continues where it stopped. Runners that don't
            # stream per-doc ignore it (and never create the file).
            ckpt_p = raw_p.with_name(f"{raw_p.stem}.partial{raw_p.suffix}")
            raw_p.parent.mkdir(parents=True, exist_ok=True)
            params["_checkpoint"] = str(ckpt_p.resolve())
            metas = metas_by_cond[cid] if sensitive else None  # raw is meta-neutral otherwise
            print(f"[{tag}] running ({runner})...", flush=True)
            try:
                rdocs = _docs_with_meta(docs, metas)
                if m.get("venv"):
                    by = _run_in_venv(m["venv"], runner, params, rdocs, out)
                else:
                    by = get_runner(runner)(rdocs, params)
                write_by_doc(raw_p, by)
                ckpt_p.unlink(missing_ok=True)  # raw.jsonl now supersedes the partial
                print(f"[{tag}] {sum(len(v) for v in by.values())} raw spans -> {raw_p}", flush=True)
            except Exception as e:  # one model must not abort the whole battery (e.g. LLM endpoint down)
                failed.append((tag, f"{type(e).__name__}: {str(e)[:160]}"))
                print(f"[{tag}] FAILED, skipping: {failed[-1][1]}", flush=True)

    if failed:
        print("\nfailed this run:")
        for tag, err in failed:
            print(f"  {tag}: {err}")

    # --- Stage 2: post-process raw -> by_doc per (model, condition) (cheap;
    # always re-derived). A changed metadata/postprocess condition takes effect
    # with `--no-run` (no model re-run, except deduce which needs its own raw). ---
    for m in cfg["models"]:
        mid, runner = m["id"], m["runner"]
        for c in conditions:
            cid = c["id"]
            raw_p = _raw_path(out, mid, runner, cid)
            if not raw_p.exists():
                continue
            cpp = {**pp_cfg, **(c["postprocess"] or {})}
            by = _postprocess(read_by_doc(raw_p), texts, metas_by_cond[cid], cpp)
            write_by_doc(out / mid / f"by_doc{_suffix(cid)}.jsonl", by)

    # --- Stage 3: evaluate / plot over EVERY (model, condition) on disk -- not
    # just those run this invocation. So `--only X`, `--skip-existing`, and
    # `--no-run` all still produce the full combined plot. A source is included
    # only if its output covers ALL input docs; an INCOMPLETE one (an interrupted
    # run whose checkpoint was promoted, or a stale output from a smaller input)
    # is excluded with a warning so partial coverage never skews the scores. ---
    input_ids = {str(d["doc_id"]) for d in docs}
    by_doc_paths, names, order = {}, {}, []
    for m in cfg["models"]:
        nm = m.get("name", m["id"])
        for c in conditions:
            cid = c["id"]
            sid = _src_id(m["id"], cid)
            bp = out / m["id"] / f"by_doc{_suffix(cid)}.jsonl"
            if not bp.exists():
                continue
            missing = input_ids - set(read_by_doc(bp))
            if missing:
                print(f"WARNING: [{sid}] excluded from evaluation -- incomplete output: "
                      f"{len(input_ids) - len(missing)}/{len(input_ids)} docs "
                      f"({len(missing)} missing). Re-run to finish it.", flush=True)
                continue
            by_doc_paths[sid] = str(bp)
            names[sid] = f"{nm} ({c['name']})" if c["name"] else nm
            order.append(sid)

    # Single-condition only: optional raw-vs-postprocess overlay ("<id>__raw").
    if not multi and pp_cfg.get("enabled", True) and pp_cfg.get("compare"):
        for m in cfg["models"]:
            if m["id"] not in order:  # main source absent/excluded -> no raw overlay
                continue
            raw_p = out / m["id"] / "raw.jsonl"
            if not raw_p.exists():
                continue
            rid = m["id"] + "__raw"
            by_doc_paths[rid] = str(raw_p)
            names[rid] = names.get(m["id"], m["id"]) + " (raw)"
            order.insert(order.index(m["id"]) + 1, rid)
    elif multi and pp_cfg.get("compare"):
        print("note: postprocess.compare is ignored when multiple conditions are defined.", flush=True)

    if not by_doc_paths:
        print("no model outputs on disk; nothing to evaluate.")
        return
    print(f"\nevaluating {len(by_doc_paths)} source(s): {', '.join(by_doc_paths)}", flush=True)

    ev = cfg.get("evaluate", {})
    if ev.get("enabled"):
        from .evaluate import evaluate as run_eval
        from .plot import plot as run_plot
        doclens = {d["doc_id"]: len(d["text"]) for d in docs}
        payload = run_eval(by_doc_paths, ev["bundle"], doclens,
                           ev.get("ignore_categories"), names, order, out)
        (out / "quantity_payload.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        summary = run_plot(payload, str(out / ev.get("plot", "recall_fp_burden.png")))
        summary.to_csv(out / "summary.csv", index=False)
        cols = ["source", "essential_recall", "total_fp_fraction_of_non_pii", "prediction_span_count"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        print(f"\nplot -> {out / ev.get('plot', 'recall_fp_burden.png')}")

        from .plot import plot_recall_by_gold_label, plot_recall_by_subannotation_category
        for key, default, fn in (
            ("plot_by_gold_label", "recall_by_gold_label.png", plot_recall_by_gold_label),
            ("plot_by_subannotation_category", "recall_by_subannotation_category.png",
             plot_recall_by_subannotation_category),
        ):
            name = ev.get(key, default)
            if not name:
                continue
            try:
                if fn(payload, str(out / name)) is not None:
                    print(f"plot -> {out / name}")
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] skipped: {type(e).__name__}: {str(e)[:120]}")


def main():
    ap = argparse.ArgumentParser(description="Run the deid-battery.")
    ap.add_argument("command", choices=["run"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None,
                    help="comma-separated model ids to (re)run; others reuse their existing output")
    ap.add_argument("--skip-existing", action="store_true",
                    help="don't re-run a model whose raw.jsonl already exists (resume after an error)")
    ap.add_argument("--no-run", action="store_true",
                    help="run no models; re-apply post-processing from raw, then re-evaluate + re-plot. "
                         "Tune `postprocess:`/`conditions:` (Channel-2 metadata) + --no-run to compare without "
                         "re-running -- except deduce, whose metadata enters at inference (re-run with --only deduce).")
    a = ap.parse_args()
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    run(a.config, only=only, skip_existing=a.skip_existing, no_run=a.no_run)


if __name__ == "__main__":
    main()
