# Hospital rerun: clean, pinned, and offline

Use this procedure for a new run on approved hospital infrastructure. It keeps
the previous deployment intact as a rollback and prevents old environments or
partial outputs from contaminating the new result.

## Safety invariants

- Patient text, `input*.jsonl`, the evaluation bundle, model outputs, and logs
  stay on approved hospital infrastructure.
- The LLM endpoint is loopback-only by default. Do not use a reverse tunnel,
  laptop endpoint, OpenRouter, or another public API.
- Start in a new checkout. Do not copy `.venv*`, `out*`, caches, or logs from the
  previous run.
- Copy model weights and PHI-bearing inputs out of band; Git intentionally
  ignores them.

## 1. Create a parallel deployment

Run in an approved parent directory next to the old deployment:

```bash
mkdir deid-rerun-2026-07-31
cd deid-rerun-2026-07-31
git clone https://github.com/stighellemans/deid-battery.git deid-battery
git clone https://github.com/stighellemans/deid-schema.git deid-schema

git -C deid-battery switch main
git -C deid-battery pull --ff-only origin main
git -C deid-schema checkout cfa99eb04e7884a13e05df27f942fa03855f4209
```

The source lock is `deployment/hospital-source-lock.json`; checkpoint hashes are
in `deployment/hospital-models.sha256`. The preflight checks the required
checkout, both checkpoints, and every Hugging Face model revision.

## 2. Build fresh frozen environments

```bash
cd deid-battery
bash scripts/setup.sh --all \
  --deid-schema ../deid-schema \
  --belgian-deduce "git+https://github.com/stighellemans/belgian-deduce.git@aeed6f27aef40bcdf4d6ddfa000cbfeb17bd6224"
```

The setup uses `requirements/main.lock.txt`,
`requirements/privacy-filters.lock.txt`, and
`requirements/deidentify.lock.txt`. The main and privacy-filter locks select
CPU-only PyTorch on Linux from PyTorch's official package index, with all other
packages from PyPI. Create separate GPU lock files before changing this for a
GPU deployment; do not silently replace Torch inside these environments.

## 3. Place approved assets

Create the local layout expected by `configs/battery.vm.yaml`:

```text
deid-battery/
├── input.jsonl
├── evaluation_bundle/
│   ├── manifest.json
│   └── reference_items.jsonl
└── models/
    ├── uza/model.pt
    ├── synthetic/best.pt
    └── qwen3-8b.gguf
```

Copy only these approved assets from their authoritative location. Do not copy
old `out/` directories into the new checkout. Verify the committed UZA and
synthetic checkpoint hashes after copying:

```bash
sha256sum -c deployment/hospital-models.sha256
```

The expected synthetic artifact is the selected 14-label v2.2 checkpoint
produced as `open-deid/models/selection/best.pt` (SHA-256
`2f3601625462fccdad833707f7e10787fad6f180eeee93b3cb2ec22bdee97bee`).
Record the GGUF identity separately before the run:

```bash
sha256sum models/qwen3-8b.gguf > models/qwen3-8b.gguf.sha256
```

Start an OpenAI-compatible Qwen server on the same approved host, listening on
`127.0.0.1:8089`. The committed config deliberately refuses an external LLM
hostname during preflight. The full preflight also recomputes the GGUF SHA-256
and compares it with `models/qwen3-8b.gguf.sha256`.

## 4. Validate before processing PHI

First validate only code and pins:

```bash
.venv/bin/python scripts/preflight_hospital.py \
  --config configs/battery.vm.yaml --code-only
```

After placing all assets and starting the local LLM, run the full file
preflight:

```bash
.venv/bin/python scripts/preflight_hospital.py \
  --config configs/battery.vm.yaml
```

Both commands must finish with `hospital preflight OK`. Do not override a dirty
checkout for a production run.

## 5. Smoke test, then full run

Build a 12-document input from the same authoritative gold source, or create an
equivalent approved subset, and use `configs/battery.smoke.yaml`. After that
passes end to end, start the full run:

```bash
.venv/bin/python -m deid_battery.orchestrate run \
  --config configs/battery.vm.yaml
```

If the process stops, resume the new run with:

```bash
.venv/bin/python -m deid_battery.orchestrate run \
  --config configs/battery.vm.yaml --skip-existing
```

Use `--skip-existing` only within this new checkout and output directory. Never
resume from outputs produced by an older code or model revision.

## 6. Export and retain evidence

Only export aggregate, inspected artifacts such as `summary.csv` and the plots.
Per-document JSONL, logs, checkpoints, inputs, and the evaluation bundle remain
inside the approved environment. Retain these alongside the run record:

- `git rev-parse HEAD` for `deid-battery`;
- `deployment/hospital-source-lock.json`;
- the three Python lock files;
- the GGUF SHA-256 file;
- the final config and preflight output;
- hardware details and `timings.yaml`.
