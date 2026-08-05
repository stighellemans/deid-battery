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

A normal battery evaluation also generates aggregate evidence for the shared
date/age substitution layer under `out/pseudonymization/export/`. It evaluates
gold `Date` and `Age_Birthdate` spans, so its transformation-validity results
are intentionally separate from the per-model detection results.

Run only this inexpensive evaluation when model outputs do not need to be
regenerated:

```bash
bash scripts/evaluate_pseudonymization.sh
```

This standalone entry point reads the same input, gold bundle, output directory,
and `evaluate.pseudonymization` settings from `configs/battery.yaml`.

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
Local Ollama commonly uses `http://127.0.0.1:11434/v1`; override the endpoint
at runtime rather than editing or copying the canonical YAML.
