#!/usr/bin/env bash
# Build the dedicated py3.9 venv for the `deidentify` runner on an amd64 Linux
# host (CPU is fine — no GPU needed), and download the model. Reference the
# resulting venv via `venv:` in the battery config.
#
#   bash scripts/setup_deidentify_venv.sh [--deid-schema DIR] [VENV_DIR] [MODEL_TAG]
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

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() { cat <<'EOF'
setup_deidentify_venv.sh — build the isolated Python 3.9 Deidentify runtime.

Usage:
  bash scripts/setup_deidentify_venv.sh [options] [VENV_DIR] [MODEL_TAG]

Options:
  --deid-schema DIR  local deid-schema checkout (default: ../deid-schema)
  -h, --help         show this help
EOF
}

VENV_DIR=""
MODEL_TAG=""
DEID_SCHEMA="${DEID_SCHEMA_DIR:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --deid-schema) DEID_SCHEMA="${2:?--deid-schema needs a path}"; shift ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
    *)
      if [ -z "$VENV_DIR" ]; then VENV_DIR="$1"
      elif [ -z "$MODEL_TAG" ]; then MODEL_TAG="$1"
      else echo "unexpected argument: $1" >&2; exit 2
      fi
      ;;
  esac
  shift
done
VENV_DIR="${VENV_DIR:-.venv-deidentify}"
MODEL_TAG="${MODEL_TAG:-model_bilstmcrf_ons_large-v0.2.0}"

# Locate/install uv (survives sudo's PATH reset — see the note above).
. "$HERE/scripts/_uv.sh"
. "$HERE/scripts/_deid_schema.sh"
DEID_SCHEMA="$(resolve_deid_schema_dir "$HERE" "$DEID_SCHEMA")"
ensure_uv

[ "$(uname -m)" = "x86_64" ] || echo "WARNING: $(uname -m) is not x86_64; the 2020 wheels are amd64-only and will likely fail."

echo "[1/5] create py3.9 venv at $VENV_DIR"
if [ -x "$VENV_DIR/bin/python" ]; then
  echo "reusing existing environment: $VENV_DIR"
else
  "$UV" venv --python 3.9 "$VENV_DIR"
fi

echo "[2/5] install shared deid-schema"
"$UV" pip install --python "$VENV_DIR/bin/python" -e "$DEID_SCHEMA"

echo "[3/5] install pinned deidentify stack"
"$UV" pip install --python "$VENV_DIR/bin/python" -r "$HERE/requirements/deidentify.lock.txt"

echo "[4/5] verify worker runtime imports"
verify_deid_schema_runtime "$VENV_DIR/bin/python" "$HERE" "$VENV_DIR"

echo "[5/5] download model $MODEL_TAG (into ~/.deidentify)"
"$VENV_DIR/bin/python" -m deidentify.util.download_model "$MODEL_TAG"

echo
echo "done. Wire it into your config:"
echo "  - id: deidentify"
echo "    runner: deidentify"
echo "    venv: $(cd "$(dirname "$VENV_DIR")" && pwd)/$(basename "$VENV_DIR")"
echo "    params: {model: $MODEL_TAG, chunk: 50}"
