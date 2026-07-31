# deid-battery

Run a **battery of de-identification systems** over the same documents, emit one
**unified span schema**, optionally inject **known metadata** (patient / caregiver
names) into the rule engines and the post-processor, and evaluate **core PII
recall + non-PII redaction rate** against a gold reference.

Config-driven and path-independent: the same code runs on a CPU VM or a GPU box.
Designed so sensitive text can be processed **fully offline** (no model calls an
external API).

## Systems (runners)

| runner | system(s) | notes |
|---|---|---|
| `robbert` | dual-head RobBERT/BERT token classifier | bring-your-own `.pt` checkpoint |
| `deduce` | `deduce` / `belgian-deduce` | rule engines; consume metadata |
| `gliner` | GLiNER zero-shot PII | needs `transformers<5.7` |
| `hf_token` | HF token classifiers (e.g. openai / OpenMed privacy filters) | `OpenAIPrivacyFilter` needs `transformers>=5.7` |
| `llm` | any OpenAI-compatible chat endpoint (vLLM / llama.cpp) | grammar-constrained JSON |
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
>   params: {model: model_bilstmcrf_ons_large-v0.2.0, chunk: 50}
> ```
> `chunk` runs the corpus in fresh-process slices to bound memory (50 ≈ 9 GB;
> lower it for smaller boxes). Does **not** run on Apple Silicon — use a Linux box.

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
cp configs/battery.example.yaml configs/battery.yaml    # edit models/paths
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml
```

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

```bash
# 1. turn your source JSONL into battery input {doc_id, text, [metadata]}.
#    Either pass flags, or put the field mapping in the config's `adapter:` block:
python -m deid_battery.inputs --config configs/battery.yaml
#    (equivalently: --from results.jsonl --field raw_text --patient-first ... --out input.jsonl)

# 2. edit configs/battery.yaml (models, device, metadata source, gold bundle)
# 3. run: models -> post-process -> evaluate -> plot
python -m deid_battery.orchestrate run --config configs/battery.yaml
```

Outputs in `output_dir/`: `<model>/by_doc.jsonl` (unified spans per model),
`summary.csv`, `quantity_payload.json`, and `core_pii_recall_non_pii_redaction.png`.

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

How it works: each finished document is appended to `output_dir/<model>/raw.partial.jsonl`
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

## Evaluation

Character-level core PII recall (with excluded categories) and a non-PII
redaction rate split into machine-only and boundary-overflow redactions, scored against a
`deid-eval-annotator` gold bundle (`reference_items.jsonl`). The evaluator and
plot are vendored under `deid_battery/_vendor/` so the package is self-contained.

The vendored post-processor also carries date and birthdate pseudonymization
(shifted dates, `Age_Birthdate` age text), but the battery never passes
`date_shift_days`, so that path stays inactive here and spans keep their generic
placeholders. See `post-process/README.md` for the age-rendering bands if you
enable it.

## Running on a CPU VM

See [docs/run_on_vm.md](docs/run_on_vm.md) for a full walkthrough on a Google
Cloud review VM (data stays on the VM; LLM via a local llama.cpp server).

For a hospital rerun or upgrade, use
[docs/hospital_rerun.md](docs/hospital_rerun.md). It uses a parallel checkout,
locked Git/model revisions, frozen Python dependencies, an offline-only LLM,
and a fail-fast preflight.
