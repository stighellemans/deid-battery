# Timing methodology

`deid-battery` separates cold initialization from resident processing so neural
models with very different loading costs can be compared without mixing timing
scopes.

## Primary comparison

The `seconds` field in a newly measured `timings.yaml` row is
`warm_end_to_end_seconds`. It contains:

1. resident inference, including tokenization, model calls, decoding, progress,
   and per-document checkpoint writes;
2. the `raw.jsonl` write;
3. shared rules/metadata post-processing for the with-metadata condition; and
4. the final `by_doc.meta.jsonl` write.

It excludes model setup, cold-only process/import overhead, and the unrecorded
warm-up pass. This is the primary time-versus-recall comparison because it
represents a loaded service processing the same corpus through the same output
boundary. Optional shared post-processing backends are imported once before any
model timer starts, preventing model order from deciding who pays that common
one-time initialization.

The plot only accepts rows with `timing_scope: warm_end_to_end`. Older or manual
rows without an explicit scope are retained in YAML but excluded from the plot.
To use a manually measured value, label its scope explicitly.

## Recorded fields

Every measured row retains:

- `setup_seconds`: dependencies, tokenizer, checkpoint, and model loading inside
  the runner;
- `warmup_seconds` and `warmup_documents`: the unrecorded warm-up pass;
- `inference_seconds`: the full measured corpus with the model resident;
- `raw_write_seconds`;
- `postprocess_seconds`: shared post-processing plus final output writing;
- `cold_overhead_seconds`: cold process/import overhead not attributed inside the
  runner;
- `warm_end_to_end_seconds`: the primary comparison;
- `cold_end_to_end_seconds`: setup + cold overhead + warm end-to-end, excluding
  the deliberately separate warm-up pass; and
- `measured_full_run_seconds`: the literal measured invocation, including warm-up.

`summary.csv` includes the most useful phase columns alongside recall.

## Warm-up policy

Local neural runners perform an unrecorded warm-up by default. RobBERT and
GLiNER use enough non-empty documents to fill the configured batch; unbatched
Hugging Face token classifiers and Deidentify use one. Configure a runner with
`params.warmup_docs`, or apply one command-line override:

```bash
python -m deid_battery.orchestrate run \
  --config configs/battery.yaml \
  --warmup-docs 1
```

Remote LLM runners default to no warm-up because a duplicate request has cost and
the endpoint owns model initialization. Their rows use
`service_setup: external_preloaded`: client/prompt setup is measured locally,
but server startup and model loading are not. Do not interpret their
`setup_seconds` as remote model-load time.

## Reproducibility requirements

For timing comparisons, keep the input documents and order, model revisions,
hardware, device, batch size, windowing parameters, metadata condition,
post-processing configuration, and software environment identical. A resumed
partial run is operationally useful but is not a valid exact benchmark; rerun
from a clean partial/output state for publication timings.
