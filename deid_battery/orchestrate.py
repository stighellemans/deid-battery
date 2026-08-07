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
import time
from pathlib import Path

import yaml

from . import metadata as md_mod
from . import timing as timing_mod
from .inputs import load_input
from .postprocess import post_process, prepare as prepare_postprocess
from .runners import get_runner, uses_metadata_at_inference
from .runners.phase_timing import read_report
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


def _raw_path(runs_dir, mid, runner, cid):
    """A per-condition raw only for inference-metadata-sensitive runners (deduce);
    every other runner shares one condition-independent raw.jsonl."""
    if cid and uses_metadata_at_inference(runner):
        return runs_dir / mid / f"raw{_suffix(cid)}.jsonl"
    return runs_dir / mid / "raw.jsonl"


def _output_dirs(out: Path) -> dict[str, Path]:
    """Return the stable, deliberately small top-level output layout."""
    return {
        "runs": out / "runs",
        "work": out / "work",
        "analysis_raw": out / "analysis" / "raw",
        "analysis_plots": out / "analysis" / "plots",
    }


def _write_output_readme(out: Path) -> None:
    (out / "README.md").write_text(
        "# DEID battery outputs\n\n"
        "- `runs/`: persisted raw and post-processed outputs, grouped by model.\n"
        "- `analysis/raw/`: machine-readable evaluation payloads and tables.\n"
        "- `analysis/plots/`: rendered evaluation figures.\n"
        "- `work/`: disposable evaluator intermediates; safe to regenerate.\n",
        encoding="utf-8",
    )


def _migrate_legacy_model_outputs(out: Path, runs_dir: Path, model_ids) -> None:
    """Move pre-layout model folders once so upgrades never trigger costly reruns."""
    for model_id in model_ids:
        legacy = out / model_id
        destination = runs_dir / model_id
        if legacy.is_dir() and not destination.exists():
            legacy.rename(destination)
            print(f"migrated legacy model output -> {destination}", flush=True)


def _docs_with_meta(docs, metas):
    return [{**d, "_meta": (metas or {}).get(d["doc_id"])} for d in docs]


def _runtime_config(cfg, *, exclude=None, input_path=None, output_dir=None,
                    evaluation_bundle=None, timings=None, llm_base_url=None,
                    llm_model=None, llm_device_label=None, deidentify_venv=None):
    """Apply machine/run-specific choices without creating another YAML file."""
    cfg = dict(cfg)
    if input_path:
        cfg["input"] = input_path
    if output_dir:
        cfg["output_dir"] = output_dir
    if timings:
        cfg["timings"] = timings
    if evaluation_bundle:
        cfg["evaluate"] = {**(cfg.get("evaluate") or {}), "bundle": evaluation_bundle}

    excluded = set(exclude or ())
    models = []
    for original in cfg.get("models", []):
        if original["id"] in excluded:
            continue
        model = dict(original)
        params = dict(model.get("params") or {})
        if model.get("runner") == "llm":
            if llm_base_url:
                params["base_url"] = llm_base_url
            if llm_model:
                params["model"] = llm_model
            if llm_device_label:
                model["device_label"] = llm_device_label
        if model.get("runner") == "deidentify" and deidentify_venv:
            model["venv"] = deidentify_venv
        model["params"] = params
        models.append(model)
    cfg["models"] = models
    return cfg


def run(config_path, only=None, skip_existing=False, no_run=False, device=None, batch_size=None,
        exclude=None, input_path=None, output_dir=None, evaluation_bundle=None, timings=None,
        llm_base_url=None, llm_model=None, llm_device_label=None, deidentify_venv=None,
        warmup_docs=None):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cfg = _runtime_config(
        cfg,
        exclude=exclude,
        input_path=input_path,
        output_dir=output_dir,
        evaluation_bundle=evaluation_bundle,
        timings=timings,
        llm_base_url=llm_base_url,
        llm_model=llm_model,
        llm_device_label=llm_device_label,
        deidentify_venv=deidentify_venv,
    )
    out = Path(cfg.get("output_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)
    output_dirs = _output_dirs(out)
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    runs_dir = output_dirs["runs"]
    work_dir = output_dirs["work"]
    analysis_raw_dir = output_dirs["analysis_raw"]
    analysis_plots_dir = output_dirs["analysis_plots"]
    _migrate_legacy_model_outputs(
        out, runs_dir, (model["id"] for model in cfg.get("models", []))
    )
    _write_output_readme(out)
    # Where per-method run times are recorded (see deid_battery.timing). Kept OUT of
    # out/ by default so hand-added `manual` rows survive an out/ wipe.
    timings_path = cfg.get("timings", "timings.yaml")

    docs = load_input(cfg["input"])
    conditions = _conditions(cfg)
    multi = len(conditions) > 1
    # resolve per-doc metadata once per condition
    metas_by_cond = {c["id"]: {d["doc_id"]: md_mod.resolve(d, c["metadata"]) for d in docs}
                     for c in conditions}

    device = device or cfg.get("device")  # CLI --device overrides the config's device
    pp_cfg = cfg.get("postprocess", {"enabled": True})
    for condition in conditions:
        prepare_postprocess({**pp_cfg, **(condition["postprocess"] or {})})
    texts = {d["doc_id"]: d["text"] for d in docs}
    only = set(only) if only else None
    if multi:
        print(f"conditions: {', '.join(c['id'] for c in conditions)}", flush=True)
    primary_timing_cid = next(
        (c["id"] for c in conditions
         if (c["metadata"] or {}).get("source", "none") != "none"),
        conditions[0]["id"],
    )
    phase_runs = {}

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
            raw_p = _raw_path(runs_dir, mid, runner, cid)
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
            if batch_size is not None and "batch_size" not in params:
                params["batch_size"] = batch_size  # CLI --batch-size (neural runners; ignored by rule-based)
            if warmup_docs is not None and "warmup_docs" not in params:
                params["warmup_docs"] = warmup_docs
            params["_label"] = tag  # progress-bar label (see deid_battery.progress.track)
            # Per-document checkpoint: the expensive runners (deidentify, llm) append
            # each finished doc here and resume from it after an interruption. It is
            # promoted to raw.jsonl and removed on success, and left in place on
            # failure so the next run continues where it stopped. Runners that don't
            # stream per-doc ignore it (and never create the file).
            ckpt_p = raw_p.with_name(f"{raw_p.stem}.partial{raw_p.suffix}")
            raw_p.parent.mkdir(parents=True, exist_ok=True)
            params["_checkpoint"] = str(ckpt_p.resolve())
            timing_p = raw_p.with_name(f"runner_timing{_suffix(cid) if per_cond else ''}.json")
            timing_p.unlink(missing_ok=True)
            params["_timing_path"] = str(timing_p.resolve())
            metas = metas_by_cond[cid] if sensitive else None  # raw is meta-neutral otherwise
            print(f"[{tag}] running ({runner})...", flush=True)
            t0 = time.perf_counter()
            try:
                rdocs = _docs_with_meta(docs, metas)
                runner_started = time.perf_counter()
                if m.get("venv"):
                    by = _run_in_venv(m["venv"], runner, params, rdocs, work_dir)
                else:
                    by = get_runner(runner)(rdocs, params)
                runner_seconds = time.perf_counter() - runner_started
                raw_write_started = time.perf_counter()
                write_by_doc(raw_p, by)
                ckpt_p.unlink(missing_ok=True)  # raw.jsonl now supersedes the partial
                raw_write_seconds = time.perf_counter() - raw_write_started
                elapsed = time.perf_counter() - t0
                phase_runs[(mid, cid if per_cond else conditions[0]["id"])] = {
                    "runner_seconds": runner_seconds,
                    "raw_write_seconds": raw_write_seconds,
                    "runner_report": read_report(timing_p),
                    "device": timing_mod.measured_device(m, params),
                }
                print(f"[{tag}] {sum(len(v) for v in by.values())} raw spans "
                      f"in {elapsed:.1f}s -> {raw_p}", flush=True)
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
            raw_p = _raw_path(runs_dir, mid, runner, cid)
            if not raw_p.exists():
                continue
            cpp = {**pp_cfg, **(c["postprocess"] or {})}
            postprocess_started = time.perf_counter()
            by = _postprocess(read_by_doc(raw_p), texts, metas_by_cond[cid], cpp)
            write_by_doc(runs_dir / mid / f"by_doc{_suffix(cid)}.jsonl", by)
            postprocess_seconds = time.perf_counter() - postprocess_started

            if cid == primary_timing_cid:
                phase_key = (mid, cid if uses_metadata_at_inference(runner)
                             else conditions[0]["id"])
                phase = phase_runs.get(phase_key)
                if phase:
                    details = timing_mod.compose_phase_metrics(
                        phase["runner_seconds"],
                        phase["raw_write_seconds"],
                        postprocess_seconds,
                        phase["runner_report"],
                    )
                    timing_mod.record_measured(
                        timings_path,
                        mid,
                        phase["device"],
                        details["warm_end_to_end_seconds"],
                        n_docs=len(docs),
                        details=details,
                    )
                    print(
                        f"[{mid}] timing: setup={details['setup_seconds']:.1f}s, "
                        f"warmup={details['warmup_seconds']:.1f}s, "
                        f"inference={details['inference_seconds']:.1f}s, "
                        f"postprocess={details['postprocess_seconds']:.1f}s, "
                        f"warm-e2e={details['warm_end_to_end_seconds']:.1f}s, "
                        f"cold-e2e={details['cold_end_to_end_seconds']:.1f}s",
                        flush=True,
                    )

    # --- Stage 3: evaluate / plot over EVERY (model, condition) on disk -- not
    # just those run this invocation. So `--only X`, `--skip-existing`, and
    # `--no-run` all still produce the full combined plot. A source is included
    # only if its output covers ALL input docs; an INCOMPLETE one (an interrupted
    # run whose checkpoint was promoted, or a stale output from a smaller input)
    # is excluded with a warning so partial coverage never skews the scores. ---
    input_ids = {str(d["doc_id"]) for d in docs}
    by_doc_paths, names, order = {}, {}, []
    # source id -> model id / condition id / bare method name (for the time-vs-recall plot,
    # whose point labels drop the condition suffix -- every point is "with metadata").
    sid_model, sid_cid, sid_label = {}, {}, {}
    for m in cfg["models"]:
        nm = m.get("name", m["id"])
        for c in conditions:
            cid = c["id"]
            sid = _src_id(m["id"], cid)
            bp = runs_dir / m["id"] / f"by_doc{_suffix(cid)}.jsonl"
            if not bp.exists():
                print(
                    f"WARNING: [{sid}] excluded from evaluation -- output not found: {bp}",
                    flush=True,
                )
                continue
            missing = input_ids - set(read_by_doc(bp))
            if missing:
                print(f"WARNING: [{sid}] excluded from evaluation -- incomplete output: "
                      f"{len(input_ids) - len(missing)}/{len(input_ids)} docs "
                      f"({len(missing)} missing). Re-run to finish it.", flush=True)
                continue
            by_doc_paths[sid] = str(bp)
            names[sid] = f"{nm} ({c['name']})" if c["name"] else nm
            sid_model[sid], sid_cid[sid], sid_label[sid] = m["id"], cid, nm
            order.append(sid)

    # "With metadata" = conditions whose metadata source isn't 'none' (the time-vs-
    # recall plot uses these). Fall back to every condition if none qualify.
    meta_cids = {c["id"] for c in conditions
                 if (c["metadata"] or {}).get("source", "none") != "none"} or {c["id"] for c in conditions}

    # Single-condition only: optional raw-vs-postprocess overlay ("<id>__raw").
    if not multi and pp_cfg.get("enabled", True) and pp_cfg.get("compare"):
        for m in cfg["models"]:
            if m["id"] not in order:  # main source absent/excluded -> no raw overlay
                continue
            raw_p = runs_dir / m["id"] / "raw.jsonl"
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
        timings = timing_mod.load(timings_path)
        doclens = {d["doc_id"]: len(d["text"]) for d in docs}
        payload = run_eval(by_doc_paths, ev["bundle"], doclens,
                           ev.get("ignore_categories"), names, order, work_dir)
        (analysis_raw_dir / "quantity_payload.json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        primary_plot = analysis_plots_dir / ev.get(
            "plot", "core_pii_recall_non_pii_redaction.png"
        )
        summary = run_plot(payload, str(primary_plot))
        # Record the primary warm end-to-end value and its component phases in
        # summary.csv. The fields are repeated across metadata conditions because
        # raw inference is shared; the plot selects the with-metadata condition.
        timing_fields = {
            "measured_seconds": "seconds",
            "setup_seconds": "setup_seconds",
            "warmup_seconds": "warmup_seconds",
            "inference_seconds": "inference_seconds",
            "raw_write_seconds": "raw_write_seconds",
            "postprocess_seconds": "postprocess_seconds",
            "warm_end_to_end_seconds": "warm_end_to_end_seconds",
            "cold_end_to_end_seconds": "cold_end_to_end_seconds",
            "timing_scope": "timing_scope",
            "service_setup": "service_setup",
        }
        for column, field in timing_fields.items():
            summary[column] = summary["annotation_id"].map(
                lambda s, key=field: timing_mod.measured_value(
                    timings.get(sid_model.get(str(s), str(s)), []), key
                )
            )
        summary.to_csv(analysis_raw_dir / "summary.csv", index=False)
        cols = ["source", "core_pii_recall", "non_pii_redaction_rate",
                "prediction_span_count", "measured_seconds", "setup_seconds",
                "cold_end_to_end_seconds"]
        print(summary[[c for c in cols if c in summary.columns]].to_string(index=False))
        print(f"\nplot -> {primary_plot}")

        # Time vs. recall (with-metadata condition), one dot per timings row, coloured
        # by device (cpu/gpu). Needs at least one (measured or manual) row in timings.yaml.
        tv_name = ev.get("plot_time_vs_recall", "time_vs_recall.png")
        if tv_name and timings:
            from .plot import plot_time_vs_recall
            meta_sids = {sid for sid in order if sid_cid.get(sid) in meta_cids}
            try:
                plot_path = analysis_plots_dir / tv_name
                csv_path = analysis_raw_dir / Path(tv_name).with_suffix(".csv")
                if plot_time_vs_recall(payload, timings, str(plot_path),
                                       meta_sids, sid_model, sid_label,
                                       csv_path=csv_path) is not None:
                    print(f"plot -> {plot_path}  (data -> {csv_path})")
                else:
                    print(f"  [{tv_name}] skipped: no (time, recall) pairs -- "
                          f"add run times to {timings_path}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [{tv_name}] skipped: {type(e).__name__}: {str(e)[:120]}")

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
                plot_path = analysis_plots_dir / name
                csv_path = analysis_raw_dir / Path(name).with_suffix(".csv")
                if fn(payload, str(plot_path), csv_path=csv_path) is not None:
                    print(f"plot -> {plot_path}  (data -> {csv_path})")
            except Exception as e:  # noqa: BLE001
                print(f"  [{name}] skipped: {type(e).__name__}: {str(e)[:120]}")

        # Keep the per-source, one-to-one span-confusion matrices together. The
        # parallel raw/plots trees keep tables auditable without mixing file types.
        from .plot import save_label_confusion_analysis
        confusion_raw_dir = analysis_raw_dir / "label_confusion"
        confusion_plots_dir = analysis_plots_dir / "label_confusion"
        try:
            save_label_confusion_analysis(
                payload, confusion_raw_dir, confusion_plots_dir
            )
            print(
                "label confusion analysis -> "
                f"{confusion_raw_dir} (raw), {confusion_plots_dir} (plots)"
            )
        except Exception as e:  # noqa: BLE001
            print(
                f"  [label_confusion] skipped: {type(e).__name__}: {str(e)[:120]}",
                flush=True,
            )

        # Extra, model-independent evidence for the shared substitution layer.
        # This is part of a normal battery run but also has a standalone entry
        # point (`python -m deid_battery.pseudonymization_eval`).
        from .pseudonymization_eval import run as run_pseudonymization_eval
        run_pseudonymization_eval(cfg, base_dir=Path.cwd())


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
    ap.add_argument("--device", default=None,
                    help="override the config's device for this run (cpu | mps | cuda). "
                         "Measured times land under the matching cpu/gpu row in timings.yaml.")
    ap.add_argument("--batch-size", type=int, default=None, dest="batch_size",
                    help="windows/docs per forward pass for the neural runners (default 1). "
                         ">1 pools work across docs for a fair GPU timing; verified span-identical.")
    ap.add_argument("--warmup-docs", type=int, default=None,
                    help="override unrecorded warm-up documents for neural runners "
                         "(batched local default fills one batch; unbatched local 1; "
                         "remote LLM 0)")
    ap.add_argument("--exclude", default=None,
                    help="comma-separated model ids to omit entirely (for example deidentify on macOS)")
    ap.add_argument("--input", dest="input_path", default=None,
                    help="override input JSONL (useful for a smoke subset)")
    ap.add_argument("--output-dir", default=None,
                    help="override output directory")
    ap.add_argument("--evaluation-bundle", default=None,
                    help="override evaluate.bundle")
    ap.add_argument("--timings", default=None,
                    help="override timings YAML path")
    ap.add_argument("--llm-base-url", default=None,
                    help="override the OpenAI-compatible URL for every LLM runner")
    ap.add_argument("--llm-model", default=None,
                    help="override the served model name for every LLM runner")
    ap.add_argument("--llm-device-label", choices=["cpu", "gpu"], default=None,
                    help="override how LLM timing is labelled (the endpoint owns its compute)")
    ap.add_argument("--deidentify-venv", default=None,
                    help="override the dedicated Deidentify environment path")
    a = ap.parse_args()
    only = [s.strip() for s in a.only.split(",") if s.strip()] if a.only else None
    exclude = [s.strip() for s in a.exclude.split(",") if s.strip()] if a.exclude else None
    run(a.config, only=only, skip_existing=a.skip_existing, no_run=a.no_run,
        device=a.device, batch_size=a.batch_size, exclude=exclude,
        input_path=a.input_path, output_dir=a.output_dir,
        evaluation_bundle=a.evaluation_bundle, timings=a.timings,
        llm_base_url=a.llm_base_url, llm_model=a.llm_model,
        llm_device_label=a.llm_device_label,
        deidentify_venv=a.deidentify_venv,
        warmup_docs=a.warmup_docs)


if __name__ == "__main__":
    main()
