#!/usr/bin/env bash
# Shared deid-schema discovery and runtime verification for setup scripts.
# Source this file; do not execute it directly.

resolve_deid_schema_dir() {
  local battery_root="$1"
  local requested="${2:-${DEID_SCHEMA_DIR:-}}"
  local candidate

  if [ -n "$requested" ]; then
    candidate="$requested"
  elif [ -d "$battery_root/../deid-schema" ]; then
    candidate="$battery_root/../deid-schema"
  else
    echo "ERROR: deid-schema source not found." >&2
    echo "Pass --deid-schema /path/to/deid-schema or set DEID_SCHEMA_DIR." >&2
    return 2
  fi

  if [ ! -d "$candidate" ]; then
    echo "ERROR: deid-schema directory does not exist: $candidate" >&2
    return 2
  fi
  candidate="$(cd "$candidate" && pwd)"
  if [ ! -f "$candidate/pyproject.toml" ] || [ ! -d "$candidate/src/deid_schema" ]; then
    echo "ERROR: not a deid-schema source checkout: $candidate" >&2
    echo "Expected pyproject.toml and src/deid_schema/." >&2
    return 2
  fi
  printf '%s\n' "$candidate"
}

verify_deid_schema_runtime() {
  local python="$1"
  local battery_root="$2"
  local environment_name="$3"

  PYTHONPATH="$battery_root${PYTHONPATH:+:$PYTHONPATH}" "$python" -c \
    'from deid_schema.taxonomy import split_label; from deid_battery.schema import make_span'
  echo "verified deid-schema + battery worker imports in $environment_name"
}
