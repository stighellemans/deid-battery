#!/usr/bin/env bash
# Build the dedicated py3.9 venv for the `deidentify` runner on an amd64 Linux
# host (CPU is fine — no GPU needed), and download the model. Reference the
# resulting venv via `venv:` in the battery config.
#
#   bash scripts/setup_deidentify_venv.sh [VENV_DIR] [MODEL_TAG]
#
# Defaults: VENV_DIR=./.venv-deidentify  MODEL_TAG=model_bilstmcrf_ons_large-v0.2.0
# Needs an amd64 Linux host. py3.9 is fetched by uv — no system python3.9 needed.
# uv is auto-installed if missing (set NO_AUTO_INSTALL_UV=1 to opt out).
#
# Avoid `sudo`: it resets PATH (secure_path) so a per-user `uv` isn't found, and
# the model would download into root's ~/.deidentify. To install into a
# root-owned dir like /opt, pre-create it for your user, then run without sudo:
#   sudo install -d -o "$USER" -g "$USER" /opt/.venv-deidentify
#   bash scripts/setup_deidentify_venv.sh /opt/.venv-deidentify
set -euo pipefail

VENV_DIR="${1:-.venv-deidentify}"
MODEL_TAG="${2:-model_bilstmcrf_ons_large-v0.2.0}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Locate/install uv (survives sudo's PATH reset — see the note above).
. "$HERE/scripts/_uv.sh"
ensure_uv

[ "$(uname -m)" = "x86_64" ] || echo "WARNING: $(uname -m) is not x86_64; the 2020 wheels are amd64-only and will likely fail."

echo "[1/3] create py3.9 venv at $VENV_DIR"
"$UV" venv --python 3.9 "$VENV_DIR"

echo "[2/3] install pinned deidentify stack"
"$UV" pip install --python "$VENV_DIR/bin/python" -r "$HERE/requirements/deidentify.txt"

echo "[3/3] download model $MODEL_TAG (into ~/.deidentify)"
"$VENV_DIR/bin/python" -m deidentify.util.download_model "$MODEL_TAG"

echo
echo "done. Wire it into your config:"
echo "  - id: deidentify"
echo "    runner: deidentify"
echo "    venv: $(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"
echo "    params: {model: $MODEL_TAG, chunk: 50}"
