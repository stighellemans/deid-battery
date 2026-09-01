# Configuration reference

`configs/battery.yaml` is the single canonical, full-battery definition. Keep
machine-, dataset-, and one-off execution guidance here rather than expanding
the YAML with operational comments.

## Standard commands

Hospital defaults:

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml
```

Apple Silicon local run (the legacy Deidentify stack is unavailable):

```bash
.venv/bin/python -m deid_battery.orchestrate run \
  --config configs/battery.yaml \
  --device mps \
  --exclude deidentify \
  --llm-base-url http://127.0.0.1:11434/v1 \
  --llm-model qwen3:8b
```

Smoke subset using the same model definition:

```bash
.venv/bin/python -m deid_battery.orchestrate run \
  --config configs/battery.yaml \
  --input input.smoke.jsonl \
  --output-dir out_smoke \
  --evaluation-bundle evaluation_bundle.smoke \
  --timings timings.smoke.yaml
```

The benchmark input can be built from the validated synthetic source with
`scripts/build_input_from_gold.py`, or from a raw export with
`python -m deid_battery.inputs --config configs/battery.yaml`. The `adapter`
block is used only by the latter and does not affect an existing `input.jsonl`.

## Pseudonymization evidence

A normal battery evaluation generates aggregate evidence for the shared
date/age substitution layer under `out/analysis/raw/pseudonymization/export/`.
That gold-span result isolates transformation validity. Saved outputs configured
under `predicted_sources` are evaluated end to end under
`out/analysis/raw/pseudonymization/predicted/<source>/export/`: detection,
boundary coverage, label assignment, and transformation validity all enter the
headline failure rate. This second analysis reuses `by_doc` output and does not
invoke a model runner.

Run only this inexpensive evaluation when model outputs do not need to be
regenerated:

```bash
bash scripts/evaluate_pseudonymization.sh
```

This standalone entry point reads the same input, gold bundle, output directory,
and `evaluate.pseudonymization` settings from `configs/battery.yaml`. Prediction
paths are relative to the configured battery `output_dir`, for example:

```yaml
predicted_sources:
  - id: synthetic@meta
    predictions: runs/synthetic/by_doc.meta.jsonl
```

Only aggregate controlled-vocabulary tables may leave a clinical evaluation
machine. The private inputs, document identifiers, text, and span-level outcomes
remain local.

## Label-assignment confusion analysis

Every enabled evaluation writes label-confusion tables under
`out/analysis/raw/label_confusion/` and the corresponding matrices under
`out/analysis/plots/label_confusion/`. Results use deterministic one-to-one matching
between gold core-PII entity spans and predictions with positive core-PII
character overlap. Candidate pairs are ordered by overlap characters, gold
coverage, prediction coverage, and then stable input order; labels never affect
matching.

Confusion cells are row-normalized among matched spans,
so they measure label assignment rather than detection recall. Unmatched gold
spans are reported separately in `summary.csv` and `confusion_long.csv`. The
raw directory contains those CSV files directly, and the plots directory contains
one rendered matrix per annotation source. All confusion cells use one
neutral light-to-dark blue scale for percentage magnitude. The plots omit
repeated colourbar legends and use a bold outline around diagonal exact-label
cells to make the preferred path explicit.

For consumers of `quantity_payload.json`, the label-confusion result is now only
`core_pii_span_label_confusion`; the former character-level
`core_pii_label_confusion` key is intentionally absent. This breaking payload
change increments `version` from 1 to 2; regenerate cached payloads after
upgrading.

## Output layout and ordering upgrades

The orchestrator migrates a legacy `out/<model>/` directory to
`out/runs/<model>/` when the destination does not already exist. Custom scripts
must update their aggregate paths from `out/summary.csv` and `out/<plot>.png` to
`out/analysis/raw/summary.csv` and `out/analysis/plots/<plot>.png`. Custom
pseudonymization configuration should use
`output_dir: analysis/raw/pseudonymization`.

Source display order follows the `models:` list, with each model's conditions in
the order of the `conditions:` list. Reorder those lists to change report order,
then run the orchestrator's `evaluate` command to regenerate analysis from
existing raw outputs without model inference (`run --no-run` is retained as an
alias). This is separate from a RobBERT model's `entity_labels:` list: that list
is the checkpoint's load-bearing classifier index map and must never be reordered
for presentation.

## Document-clustered bootstrap

Set `evaluate.bootstrap.enabled: true` to generate 95% confidence intervals for
the configured recall metric and non-PII redaction rate. The default manuscript
contract uses 10,000 replicates, seed `20260821`, and percentile intervals.
Documents—not individual spans or characters—are sampled with replacement. All
gold spans and every source's outputs within a selected document stay together,
so source differences are paired and hard-negative documents remain in the
non-PII denominator.

```yaml
evaluate:
  bootstrap:
    enabled: true
    replicates: 10000
    seed: 20260821
    confidence_level: 0.95
```

By default all available sources are contrasted pairwise. Set `pairs` to a list
of annotation-ID pairs to limit the output, for example
`pairs: [[model, annotator-1], [model, annotator-2]]`.

## Device behavior

- `device: cpu` is the portable, reproducible default. `--device mps` and
  `--device cuda` override it for neural runners.
- Deduce is rule-based and ignores the device.
- RobBERT, GLiNER, and the privacy filters can use CPU/MPS/CUDA when their
  environment provides the corresponding PyTorch build.
- Qwen owns its compute behind an OpenAI-compatible endpoint. The global device
  setting does not control it; `device_label` records its timing category.
- Deidentify stays on CPU in its isolated legacy environment.

## Fields whose rationale is not obvious

- `timings` lives outside `out/` so hand-entered comparison timings survive an
  output rebuild. Measured rows are updated per CPU/GPU category.
- `metadata.source: from_input` reads explicit metadata from each input row. It
  never derives metadata from gold spans.
- The `nometa` and `meta` conditions share expensive inference except for
  metadata-sensitive Deduce runners.
- `model_commit` is the exact 40-character Git commit for a Hugging Face model
  snapshot. `base_model_commit` pins the base encoder used by a local RobBERT
  checkpoint. The runners translate these names to Hugging Face's `revision`
  argument internally.
- Both RobBERT checkpoints have 14-label heads; the retired subtype slot was
  removed. Their ordered labels are specified explicitly in the config and are
  not loaded from training metrics.
- `postprocess.inception` requires JVM-based INCEpTION token normalization and
  therefore remains disabled in the standard run.
- `evaluate.bundle` must cover exactly the same document IDs as `input`.

## Deidentify

Deidentify is a 2020 spaCy/Flair/Torch stack requiring a dedicated Python 3.9
environment on amd64 Linux. The established hospital path is
`/opt/.venv-deidentify`. Its model baseline is about 8.7 GB RAM.

- `model_bilstmcrf_ons_large-v0.2.0` is the released Nedap model name, not a
  Hugging Face revision.
- `max_chars: 20000` windows unusually long documents to prevent nonlinear RAM
  spikes.
- `overlap: 500` protects entities crossing window boundaries.
- `chunk: 50` provides progress, checkpoint boundaries, and fault isolation; it
  does not reduce the model's baseline memory.

## Qwen

The hospital endpoint is loopback-only at `http://127.0.0.1:11500/v1`, serving
the API model tag `qwen3:8b`. Before each benchmark run, manually verify that
the model behind that tag uses `Q4_K_M` quantization; the tag alone does not
prove the quantization. The validated generation settings are temperature
`0.6`, top-p `0.95`, thinking enabled, 8,000 output tokens, and two workers.
The paper's Ollama deployment used a context length of 16,384 tokens.
Local Ollama commonly uses `http://127.0.0.1:11434/v1`; override the endpoint
at runtime rather than editing or copying the canonical YAML.
