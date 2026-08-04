#!/usr/bin/env bash
set -euo pipefail

# Standalone entry point for the extra date/age pseudonymization evidence that
# is also generated automatically by a normal deid-battery run.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
CONFIG="${DEID_BATTERY_CONFIG:-$ROOT/configs/battery.yaml}"

find_python() {
  local candidate
  if [[ -n "${DEID_BATTERY_PYTHON:-}" ]]; then
    printf '%s\n' "$DEID_BATTERY_PYTHON"
    return
  fi
  for candidate in \
    "$ROOT/.venv/bin/python" \
    "$HOME/.uv/base/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"; do
    if [[ -n "$candidate" && -x "$candidate" ]] \
      && "$candidate" -c 'import deid_battery' \
        >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  echo "No Python runtime has deid-battery installed." >&2
  echo "Set DEID_BATTERY_PYTHON=/path/to/python." >&2
  exit 2
}

cd "$ROOT"
exec "$(find_python)" -m deid_battery.pseudonymization_eval \
  --config "$CONFIG" "$@"
