#!/bin/bash
# Build the dedicated py3.9 venv for the `deidentify` runner on an amd64 Linux
# host (CPU is fine — no GPU needed), and download the model. Reference the
# resulting venv via `venv:` in the battery config.
#
#   bash scripts/setup_deidentify_venv.sh [VENV_DIR] [MODEL_TAG]
#
# Defaults: VENV_DIR=./.venv-deidentify  MODEL_TAG=model_bilstmcrf_ons_large-v0.2.0
# Requires: uv (https://astral.sh/uv) and an amd64 Linux host. py3.9 is fetched
# by uv; you do NOT need a system python3.9.
set -euo pipefail

VENV_DIR="${1:-.venv-deidentify}"
MODEL_TAG="${2:-model_bilstmcrf_ons_large-v0.2.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

command -v uv >/dev/null || { echo "uv not found — install: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
[ "$(uname -m)" = "x86_64" ] || echo "WARNING: $(uname -m) is not x86_64; the 2020 wheels are amd64-only and will likely fail."

echo "[1/3] create py3.9 venv at $VENV_DIR"
uv venv --python 3.9 "$VENV_DIR"

echo "[2/3] install pinned deidentify stack"
uv pip install --python "$VENV_DIR/bin/python" -r "$HERE/requirements/deidentify.txt"

echo "[3/3] download model $MODEL_TAG (into ~/.deidentify)"
"$VENV_DIR/bin/python" -m deidentify.util.download_model "$MODEL_TAG"

echo
echo "done. Wire it into your config:"
echo "  - id: deidentify"
echo "    runner: deidentify"
echo "    venv: $(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"
echo "    params: {model: $MODEL_TAG, chunk: 50}"
