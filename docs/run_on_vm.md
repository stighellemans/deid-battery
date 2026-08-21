# Running deid-battery on the review VM (CPU, data stays on the VM)

Walkthrough for `deid-review-vm` (Google Cloud). The documents never leave the
VM; everything — including the LLM — runs locally on the VM's CPU. Only the
aggregate outputs in `out/analysis/raw/` and `out/analysis/plots/` contain no
patient text and may be copied off.

For a hospital upgrade, follow [hospital_rerun.md](hospital_rerun.md) instead;
it preserves the old deployment, uses the committed source/model locks, and
adds a production preflight.

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
git clone https://github.com/stighellemans/deid-schema.git deid-schema
git -C deid-schema checkout cfa99eb04e7884a13e05df27f942fa03855f4209
git clone https://github.com/stighellemans/deid-battery.git deid-battery && cd deid-battery

# one command builds .venv (main) + .venv-pf (privacy filters). uv fetches its
# own Python, so the python3-venv/build-essential from step 1 are optional.
bash scripts/setup.sh --pf --deid-schema ../deid-schema \
  --belgian-deduce "git+https://github.com/stighellemans/belgian-deduce.git@aeed6f27aef40bcdf4d6ddfa000cbfeb17bd6224"

sudo install -d -o "$USER" -g "$USER" /opt/.venv-deidentify
bash scripts/setup_deidentify_venv.sh --deid-schema ../deid-schema /opt/.venv-deidentify

# Confirm the checkout and all remote model commits are locked before adding data:
.venv/bin/python scripts/preflight_hospital.py --config configs/battery.yaml \
  --code-only
```

The committed VM lock installs CPU-only PyTorch on Linux. A GPU deployment needs
separately generated and reviewed GPU lock files plus `device: cuda`; do not
replace Torch ad hoc inside the frozen environment.

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
# pass this endpoint with --llm-base-url at run time
```

CPU inference is slow — use a small model and/or a document subset for the LLM
row. The non-LLM runners are fast on CPU.

## 5. Run the canonical config

There is one configuration, `configs/battery.yaml`. The review VM's CPU Qwen
endpoint is a runtime difference, so override it without copying the YAML:

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml \
  --llm-base-url http://127.0.0.1:8089/v1 \
  --llm-model qwen3-8b \
  --llm-device-label cpu
```

This runs every model → shared post-processing (with metadata) → evaluation →
`out/analysis/plots/core_pii_recall_non_pii_redaction.png` +
`out/analysis/raw/summary.csv`. Watch the log; each model prints
its span count.

**Interrupted? Just re-run with `--skip-existing`.** CPU inference is slow, so a
run may be stopped (Ctrl-C, an OOM, an overnight disconnect). Inference is
checkpointed per document, so nothing finished is lost:

```bash
.venv/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml \
  --skip-existing \
  --llm-base-url http://127.0.0.1:8089/v1 \
  --llm-model qwen3-8b \
  --llm-device-label cpu
```

Finished models are skipped; a model that was mid-flight continues from
`out/runs/<model>/raw.partial.jsonl` (`[deidentify] resume: 240/300 docs already done`).
An incomplete model is **excluded from the evaluation/plot with a warning** until
it's finished, so a partial run never skews the scores.

To refresh evaluation after installing a new deid-battery revision, use the
inference-free command on each VM/result directory:

```bash
.venv/bin/python -m deid_battery.orchestrate evaluate --config configs/battery.yaml
```

This command never imports a configured runner. It re-applies deterministic
post-processing to each completed `raw.jsonl`, then regenerates the evaluation,
plots, and configured document-clustered bootstrap intervals. Do not point it at
an output directory while an inference process is still writing there.

## 6. Take only the aggregates off the VM, then stop it

```bash
# analysis/raw/summary.csv + analysis/plots/ are aggregate outputs -> safe to copy out
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
- **One model at a time.** Use `--only model-id` or `--exclude model-id`; do not
  make another YAML. Outputs are per-model, so the slow LLM can run later.
- **Safe to interrupt.** Every runner checkpoints per document; re-run with
  `--skip-existing` to resume an interrupted run (see step 5). Don't run two
  processes against the same `out/` at once — resume is for a *stopped* run.
