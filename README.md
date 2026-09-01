# deid-battery

Research code accompanying the MedDeID manuscript. This repository is maintained
separately from the production MedDeID suite and is intended to reproduce the
paper's comparator runs and analyses.

Run a **battery of de-identification systems** over the same documents, emit one
**unified span schema**, optionally inject **known metadata** (patient / caregiver
names) into the rule engines and the post-processor, and evaluate **core PII
recall + non-PII redaction rate** against a gold reference.

Config-driven and path-independent: the same code runs on a CPU VM or a GPU box.
Designed so sensitive text can be processed **fully offline** (no model calls an
external API).

For the exact Qwen prompt assets, paper configuration, and repository dependency
boundaries, see [Paper reproducibility](docs/paper_reproducibility.md).

## Systems (runners)

| runner | system(s) | notes |
|---|---|---|
| `robbert` | dual-head RobBERT/BERT token classifier | bring-your-own `.pt` checkpoint |
| `deduce` | `deduce` / `belgian-deduce` | rule engines; consume metadata |
| `gliner` | GLiNER zero-shot PII | needs `transformers<5.7` |
| `hf_token` | HF token classifiers (e.g. openai / OpenMed privacy filters) | `OpenAIPrivacyFilter` needs `transformers>=5.7` |
| `llm` | any OpenAI-compatible chat endpoint (Ollama / vLLM / llama.cpp) | prompted JSON; schema-constrained when thinking is disabled |
| `deidentify` | `deidentify` BiLSTM-CRF (nedap/Trienes 2020) | 2020 stack: py3.9 + **amd64 Linux** only; runs on **CPU**; dedicated venv |

> **Dependency note.** GLiNER (`transformers<5.7`) and the privacy filters
> (`transformers>=5.7`) can't share one environment. Put the privacy filters in a
> second venv and point the model at it with `venv:` in the config — the
> orchestrator runs that runner there in a subprocess.

> **`deidentify` runner.** It needs a 2020 stack (spaCy 2.x / flair 0.10 /
> torch 1.10) with no arm64/py3.12 wheels, so it only builds on an **amd64 Linux**
> host (CPU is fine — no GPU). Build its venv once and reference it with `venv:`:
> ```bash
> bash scripts/setup.sh --deidentify --deid-schema ../deid-schema
> # into a shared/root-owned dir like /opt instead? create it for your user first,
> # then run the dedicated script WITHOUT sudo (sudo hides your uv + misplaces the model):
> #   sudo install -d -o "$USER" -g "$USER" /opt/.venv-deidentify
> #   bash scripts/setup_deidentify_venv.sh --deid-schema ../deid-schema /opt/.venv-deidentify
> ```
> ```yaml
> - id: deidentify
>   runner: deidentify
>   venv: /opt/.venv-deidentify
>   params: {model: model_bilstmcrf_ons_large-v0.2.0, device: cpu, chunk: 50, max_chars: 20000, overlap: 500}
> ```
> `max_chars` is the memory guard; `chunk` adds fault-isolation/checkpoints.
> Does **not** run on Apple Silicon — use `--exclude deidentify` locally.

## Install

One command builds whichever environments you want. It is uv-first: it
auto-installs [uv](https://astral.sh/uv) and fetches its own Python, so you need
no `apt` packages and no system `python3.x`.

Setup installs the committed lock files under `requirements/`; a fresh checkout
therefore gets the validated dependency versions rather than whatever happens
to be newest on the package index that day. The Linux locks use CPU-only
PyTorch. See `docs/hospital_rerun.md` before changing them for a GPU host.

```bash
bash scripts/setup.sh --deid-schema ../deid-schema       # main env
bash scripts/setup.sh --pf --deid-schema ../deid-schema  # + privacy filters
bash scripts/setup.sh --all --deid-schema ../deid-schema # + Deidentify (amd64 Linux)
```

`--deid-schema` must point to a local checkout. It defaults to the sibling
`../deid-schema` directory and setup fails before model installation when that
checkout is unavailable or its worker imports do not validate.

Then:

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml
```

There is deliberately only one YAML. Machine-specific choices are runtime
flags. For Apple Silicon, for example:

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml \
  --device mps --exclude deidentify \
  --llm-base-url http://127.0.0.1:11434/v1 --llm-model qwen3:8b
```

Use `--device cuda` on a CUDA host. Rule-based runners ignore the device; Qwen
runs behind its own endpoint; and Deidentify stays CPU-only in its legacy venv.

<details><summary>Manual install (no setup script / no uv)</summary>

```bash
pip install -e ../deid-schema
pip install -e . -r requirements/robbert.txt -r requirements/deduce.txt \
                 -r requirements/gliner.txt  -r requirements/llm.txt
# privacy filters need a SEPARATE env (transformers>=5.7 conflicts with gliner):
python -m venv .venv-pf
.venv-pf/bin/pip install -e ../deid-schema
.venv-pf/bin/pip install -e . -r requirements/privacy-filters.txt
```
</details>

## Use

Detailed field rationale and machine-specific commands are in
[docs/configuration.md](docs/configuration.md). Timing scopes and the fair neural
comparison are documented in [docs/timing.md](docs/timing.md).

```bash
# 1. turn your source JSONL into battery input {doc_id, text, [metadata]}.
#    Either pass flags, or put the field mapping in the config's `adapter:` block:
python -m deid_battery.inputs --config configs/battery.yaml
#    (equivalently: --from results.jsonl --field raw_text --patient-first ... --out input.jsonl)

# 2. review configs/battery.yaml (the one benchmark definition)
# 3. run: models -> post-process -> evaluate -> plot
python -m deid_battery.orchestrate run --config configs/battery.yaml
```

Outputs in `output_dir/` use a fixed, uncluttered layout:

- `runs/<model>/`: persisted raw and post-processed model spans;
- `analysis/raw/`: machine-readable evaluation payloads and tables;
- `analysis/plots/`: rendered figures; and
- `work/`: disposable evaluator intermediates.

The main outputs are `analysis/raw/summary.csv`,
`analysis/raw/quantity_payload.json`, and
`analysis/plots/core_pii_recall_non_pii_redaction.png`. Span-only label-confusion
tables and per-source matrices live under `analysis/raw/label_confusion/` and
`analysis/plots/label_confusion/`.

When `evaluate.bootstrap.enabled` is true, `analysis/raw/bootstrap/` also
contains document-clustered percentile intervals in `estimates.csv`, paired
source contrasts in `paired_differences.csv`, and the exact sampling contract
in `methodology.json`.

To regenerate post-processing, evaluation, plots, and bootstrap outputs from
persisted `raw.jsonl` files without invoking any model runner, use the explicit
evaluation-only command:

```bash
.venv/bin/python -m deid_battery.orchestrate evaluate --config configs/battery.yaml
```

`run --no-run` remains a backwards-compatible spelling of the same workflow.

Neural runners warm up non-empty documents by default (enough to fill a configured
RobBERT/GLiNER batch; one for unbatched local runners; zero for remote LLM
endpoints), then record model setup, warm-up, resident inference, shared
post-processing, warm end-to-end, and cold end-to-end separately. Override the
warm-up consistently with `--warmup-docs N`.


## Resuming an interrupted run

Inference is checkpointed **per document**, so a run that stops partway
(Ctrl-C, OOM, an LLM endpoint drop) never loses finished work. To continue,
re-run the same command with `--skip-existing`:

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml --skip-existing
```

Completed models are skipped (their `raw.jsonl` exists); any model that was
mid-flight **resumes from where it stopped** — you'll see e.g.
`[deidentify] resume: 240/300 docs already done`, and a runner whose docs are
all done won't even load its model.

How it works: each finished document is appended to `output_dir/runs/<model>/raw.partial.jsonl`
the moment it completes. On success that file is promoted to `raw.jsonl` and
removed; if the run dies it's left in place to resume from. Failed docs (e.g. a
dropped LLM call) are **not** checkpointed, so they retry on the next run.

**Incomplete outputs are excluded from evaluation, with a warning** — partial
coverage can never skew the scores:

```
WARNING: [uza@meta] excluded from evaluation -- incomplete output: 240/300 docs (60 missing). Re-run to finish it.
```

This also fires if you grow `input.jsonl` and re-run with `--skip-existing`: the
older, now-incomplete model outputs are flagged and dropped until re-run.

> Resume is for a **stopped** run — don't point two processes at the same
> `output_dir` at once (they'd both append to the same partial). For the
> `deidentify` runner two concurrent processes would also exceed its ~9 GB
> footprint.

## Metadata (patient / caregiver names, addresses, document date)

Known identifiers are a first-class, optional input that feeds **both** the
deduce runners **and** the post-processor. Set `metadata.source`:

- `none` — no metadata
- `from_input` — each doc carries an explicit `metadata` key-value object (shape
  in `deid_battery/metadata.py`). Metadata is never derived from the gold
  annotation spans; convert a raw source into `input.jsonl` with a pre-step
  (`python -m deid_battery.inputs …`).

```yaml
metadata:
  source: from_input   # each doc's explicit `metadata` object (built by the pre-step)
postprocess:
  use_metadata: true   # inject known names at the post-processing stage too
```

`belgian-deduce` consumes patient + caregiver names (+ addresses, birth date,
document date); plain `deduce` consumes the patient only — the post-processor
recovers caregiver names for it.

## Unified span schema

```json
{"begin": 35, "end": 42, "label": "Name:Patient", "text": "Ny Oruç",
 "Category": "Name", "Subtype": "Patient"}
```

### Importing the legacy UZA human annotations

The old human annotations are stored as per-document JSON arrays under numbered
folders such as `stig1/spans/` and `tomstroobants1/spans/`. Convert all numbered
batches into one battery JSONL file per annotator with:

```bash
python -m deid_battery.legacy_human_annotations \
  --source-root /path/to/annotations/llm_experiment \
  --battery-input input.jsonl \
  --output-dir out/human-annotators
```

This writes `out/human-annotators/stig.jsonl` and
`out/human-annotators/tomstroobants.jsonl`. The converter checks complete
document coverage, validates every offset and span text against `input.jsonl`,
derives canonical `category`/`subtype` values from each label, and rejects
conflicting duplicate documents.

If an annotator is known to be partial, add `--allow-partial`. The converter
does not guess a semantic label. It retains each explicitly marked span's
geometry under the sentinel label `(missing label)` and writes an empty record
for every input document absent from that annotator's folders. This permits
full-corpus character recall: omitted spans/documents count as false negatives,
while known span boundaries still count as redacted. The converter also writes
`coverage_manifest.json` and `common_complete_doc_ids.txt`. The originals remain
unchanged.

Inspect every missing-label span, absent document, empty annotation file and
malformed record without modifying the source data:

```bash
python -m deid_battery.human_annotation_missingness \
  --source-root /path/to/annotations/llm_experiment \
  --battery-input input.jsonl \
  --output out/human-annotators/missingness.tsv
```

The TSV omits span text by default. Add `--include-text` only on the approved
evaluation machine, or use `--format jsonl` for a machine-readable report.

For `deid-evaluation`, configure these files as `annotator-level-jsonl` sources:

```yaml
annotations:
  stig:
    name: Annotator 1
    method: annotator-level-jsonl
    path: /path/to/deid-battery/out/human-annotators/annotator-1.jsonl
  tomstroobants:
    name: Annotator 2
    method: annotator-level-jsonl
    path: /path/to/deid-battery/out/human-annotators/annotator-2.jsonl
```

These files contain sensitive document identifiers and span text and must remain
on the approved evaluation machine; export only aggregate evaluation results.
For full-corpus human metrics, evaluate the generated JSONLs against the full
gold bundle and clearly identify them as performance of the partial submission:
absent annotations count as false negatives, and `(missing label)` is incorrect
for semantic label metrics. As a sensitivity analysis, compare humans and
systems on `common_complete_doc_ids.txt`; full-corpus system results remain
separate and unchanged.

## Evaluation

Character-level core PII recall (with excluded categories) and a non-PII
redaction rate split into false-positive spans and PII boundary extensions,
scored against a `deid-eval-annotator` gold bundle
(`reference_items.jsonl`). The evaluator and
plot are vendored under `deid_battery/_vendor/` so the package is self-contained.

In addition to model-detection metrics, a normal battery run performs two
date/age pseudonymization analyses. The gold-span analysis isolates the shared
substitution layer's validity, transformation types, and failure reasons. The
predicted-span analysis measures the complete detection, boundary, label, and
transformation chain using saved model outputs, so it requires no new inference.

The predicted-span headline is gold-span based: a target succeeds only when one
prediction fully covers it, assigns the correct `Date` or `Age_Birthdate` label,
and produces a protocol-valid transformation. The export also separates
residual exposure from privacy-safe but clinically invalid generic redaction.
Privacy-safe aggregates are written to
`out/analysis/raw/pseudonymization/export/` and
`out/analysis/raw/pseudonymization/predicted/<source>/export/`.

Regenerate only that additional evidence, without running any models:

```bash
bash scripts/evaluate_pseudonymization.sh
```

The standalone command and the normal battery run use the same
`evaluate.pseudonymization` settings in `configs/battery.yaml`, including each
saved output listed under `predicted_sources`.

## Running on a CPU VM

See [docs/run_on_vm.md](docs/run_on_vm.md) for a full walkthrough on a Google
Cloud review VM (data stays on the VM; LLM via a local llama.cpp server).

For a hospital rerun or upgrade, use
[docs/hospital_rerun.md](docs/hospital_rerun.md). It uses a parallel checkout,
locked Git/model revisions, frozen Python dependencies, an offline-only LLM,
and a fail-fast preflight.
