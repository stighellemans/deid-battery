# Running deid-battery on the review VM (CPU, data stays on the VM)

Walkthrough for `deid-review-vm` (Google Cloud). The documents never leave the
VM; everything — including the LLM — runs locally on the VM's CPU. Only the
aggregate outputs (`summary.csv`, `recall_fp_burden.png`) contain no patient
text and may be copied off.

## 0. Start + SSH (from your laptop)

```bash
gcloud compute instances start deid-review-vm --zone=europe-north1-b
gcloud compute ssh deid-review-vm --zone=europe-north1-b --tunnel-through-iap
# ... everything below runs ON the VM ...
```

## 1. Data

```bash
gsutil cp gs://co-medic-dashboard-407614-deid-research-bucket/deid-export.zip .
sudo apt-get update && sudo apt-get install -y unzip python3-venv build-essential
unzip deid-export.zip          # -> results.jsonl  (id, raw_text, spans, ...)
```

## 2. Code + environments

```bash
git clone <your deid-battery repo> && cd deid-battery

# main env: core + the runners that share transformers<5.7
python3 -m venv .venv && . .venv/bin/activate
pip install -e . \
  -r requirements/robbert.txt -r requirements/deduce.txt \
  -r requirements/gliner.txt  -r requirements/llm.txt
# belgian-deduce is not on PyPI -- install from its repo:
pip install /path/to/belgian-deduce
deactivate

# second env for the privacy filters (transformers>=5.7)
python3 -m venv .venv-pf
.venv-pf/bin/pip install -e . -r requirements/privacy-filters.txt
```

CPU note: `pip install torch` pulls the CPU build automatically on a VM without
CUDA. On a GPU VM, install the CUDA torch wheel instead and set `device: cuda`.

## 3. Input + gold bundle

```bash
# results.jsonl -> battery input {doc_id, text, [metadata]}.
# Put the column mapping in the config's `adapter:` block, then:
.venv/bin/python -m deid_battery.inputs --config configs/battery.yaml
# (or pass flags: --from results.jsonl --field raw_text --patient-first ... --out input.jsonl)

# the gold bundle (human-validated reference) from deid-eval-annotator:
#   evaluation_bundle/reference_items.jsonl + manifest.json
# copy it next to the config (it must cover the same doc_ids as input.jsonl).
```

## 4. LLM server on the VM (for the LLM comparison)

Data can't leave the VM, so the LLM endpoint must be local. Easiest is the
OpenAI-compatible server from `llama-cpp-python`:

```bash
.venv/bin/pip install "llama-cpp-python[server]"
# download a small GGUF into the VM (e.g. a 7-8B Q4_K_M), then:
.venv/bin/python -m llama_cpp.server --model qwen.gguf --host 127.0.0.1 --port 8089 \
    --n_ctx 8192 --chat_format chatml &
# point the llm runner's base_url at http://127.0.0.1:8089/v1
```

CPU inference is slow — use a small model and/or a document subset for the LLM
row. The non-LLM runners are fast on CPU.

## 5. Configure + run

Copy `configs/battery.example.yaml` to `configs/battery.yaml` and set:
`input: input.jsonl`, `device: cpu`, your RobBERT `checkpoint:`/`train_metrics:`,
the privacy-filter `venv: ../.venv-pf`, the llm `base_url:`, `metadata.source`,
and `evaluate.bundle: evaluation_bundle`.

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml
```

This runs every model → shared post-processing (with metadata) → evaluation →
`out/recall_fp_burden.png` + `out/summary.csv`. Watch the log; each model prints
its span count.

## 6. Take only the aggregates off the VM, then stop it

```bash
# summary.csv + the plot are pure scores (no patient text) -> safe to copy out
# (e.g. via the RDP desktop, or gsutil to a results bucket)
gcloud compute instances stop deid-review-vm --zone=europe-north1-b   # from your laptop
```

## Tips

- **Subset first.** Run a 10-doc `input.jsonl` end-to-end before the full set to
  shake out paths/labels, especially the RobBERT label order and the gold bundle
  doc-id coverage.
- **Metadata.** `metadata.source: from_input` uses the explicit `metadata` object
  each doc carries, built by the `deid_battery.inputs` pre-step (step 3). Metadata
  is never derived from the gold annotation spans.
- **One model at a time.** Comment models in/out of the config; outputs are
  per-model, so you can add the slow LLM row later without rerunning the rest.
