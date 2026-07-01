# deid-battery

Run a **battery of de-identification systems** over the same documents, emit one
**unified span schema**, optionally inject **known metadata** (patient / caregiver
names) into the rule engines and the post-processor, and evaluate **essential
recall + false-positive burden** against a gold reference.

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
> bash scripts/setup.sh --deidentify     # builds ./.venv-deidentify (py3.9 + model)
> # into a shared/root-owned dir like /opt instead? create it for your user first,
> # then run the dedicated script WITHOUT sudo (sudo hides your uv + misplaces the model):
> #   sudo install -d -o "$USER" -g "$USER" /opt/.venv-deidentify
> #   bash scripts/setup_deidentify_venv.sh /opt/.venv-deidentify
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

```bash
bash scripts/setup.sh          # main env .venv: core + robbert + deduce + gliner + llm
bash scripts/setup.sh --pf     # + privacy-filters env (.venv-pf, transformers>=5.7)
bash scripts/setup.sh --all    # + deidentify env (.venv-deidentify, amd64 Linux only)
```

Then:

```bash
cp configs/battery.example.yaml configs/battery.yaml    # edit models/paths
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml
```

<details><summary>Manual install (no setup script / no uv)</summary>

```bash
pip install -e . -r requirements/robbert.txt -r requirements/deduce.txt \
                 -r requirements/gliner.txt  -r requirements/llm.txt
# privacy filters need a SEPARATE env (transformers>=5.7 conflicts with gliner):
python -m venv .venv-pf && .venv-pf/bin/pip install -e . -r requirements/privacy-filters.txt
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
`summary.csv`, `quantity_payload.json`, and `recall_fp_burden.png`.

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

Char-level essential recall (with ignorable categories) and a false-positive
burden split into machine-only vs overflow FP, scored against a
`deid-eval-annotator` gold bundle (`reference_items.jsonl`). The evaluator and
plot are vendored under `deid_battery/_vendor/` so the package is self-contained.

## Running on a CPU VM

See [docs/run_on_vm.md](docs/run_on_vm.md) for a full walkthrough on a Google
Cloud review VM (data stays on the VM; LLM via a local llama.cpp server).
