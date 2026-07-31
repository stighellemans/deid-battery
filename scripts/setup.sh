#!/usr/bin/env bash
# One-command setup for deid-battery. Builds the environment(s) you ask for and
# prints how to wire them into the config.
#
#   bash scripts/setup.sh                 # main env (.venv): core + robbert + deduce + gliner + llm
#   bash scripts/setup.sh --pf            # + privacy-filters env (.venv-pf, transformers>=5.7)
#   bash scripts/setup.sh --deidentify    # + deidentify env (.venv-deidentify, 2020 amd64 stack)
#   bash scripts/setup.sh --all           # everything above
#
# uv is used by default (fast; fetches Python itself, so no system python3.x and
# no apt packages are needed) and auto-installed if missing. Pass --no-uv to use
# the stdlib `python3 -m venv` + pip instead.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
. "$HERE/scripts/_uv.sh"
. "$HERE/scripts/_deid_schema.sh"

usage() { cat <<'EOF'
setup.sh — build the deid-battery environment(s) you ask for.

  bash scripts/setup.sh                 main env (.venv): core + robbert + deduce + gliner + llm
  bash scripts/setup.sh --pf            + privacy-filters env (.venv-pf, transformers>=5.7)
  bash scripts/setup.sh --deidentify    + deidentify env (.venv-deidentify, 2020 amd64 stack)
  bash scripts/setup.sh --all           everything above

Options:
  --pf                  also build the privacy-filters venv (.venv-pf)
  --deidentify          also build the deidentify venv (amd64 Linux only)
  --all                 shorthand for --pf --deidentify
  --deid-schema DIR     local deid-schema checkout (default: ../deid-schema)
  --belgian-deduce DIR  pip-install belgian-deduce from DIR into the main env
  --venv DIR            main venv dir           (default: .venv)
  --python VERSION      python for the main env (default: 3.11)
  --no-uv               use `python3 -m venv` + pip instead of uv
  -h, --help            show this help
EOF
}

VENV=".venv"; PYVER="3.11"; DO_PF=0; DO_DEID=0; USE_UV=1; BELGIAN=""
DEID_SCHEMA="${DEID_SCHEMA_DIR:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --pf)             DO_PF=1 ;;
    --deidentify)     DO_DEID=1 ;;
    --all)            DO_PF=1; DO_DEID=1 ;;
    --deid-schema)    DEID_SCHEMA="${2:?--deid-schema needs a path}"; shift ;;
    --belgian-deduce) BELGIAN="${2:?--belgian-deduce needs a path}"; shift ;;
    --venv)           VENV="${2:?--venv needs a dir}"; shift ;;
    --python)         PYVER="${2:?--python needs a version}"; shift ;;
    --no-uv)          USE_UV=0 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "unknown option: $1 (try --help)" >&2; exit 2 ;;
  esac
  shift
done

DEID_SCHEMA="$(resolve_deid_schema_dir "$HERE" "$DEID_SCHEMA")"
cd "$HERE"
[ "$USE_UV" = 1 ] && ensure_uv

mkenv() {  # mkenv DIR PYVER
  if [ -x "$1/bin/python" ]; then
    echo "reusing existing environment: $1"
    return 0
  fi
  if [ "$USE_UV" = 1 ]; then "$UV" venv --python "$2" "$1"
  else python3 -m venv "$1"; fi
}
pipi() {   # pipi DIR PIP_ARGS...
  local d="$1"; shift
  if [ "$USE_UV" = 1 ]; then "$UV" pip install --python "$d/bin/python" "$@"
  else "$d/bin/python" -m pip install --upgrade pip >/dev/null && "$d/bin/python" -m pip install "$@"; fi
}
pipi_lock() {  # exact locks use the trusted PyPI + official PyTorch indexes
  local d="$1"; shift
  if [ "$USE_UV" = 1 ]; then
    "$UV" pip install --index-strategy unsafe-best-match --python "$d/bin/python" "$@"
  else
    "$d/bin/python" -m pip install --upgrade pip >/dev/null
    "$d/bin/python" -m pip install "$@"
  fi
}

echo "==> main env ($VENV): frozen core + robbert + deduce + gliner + llm"
mkenv "$VENV" "$PYVER"
pipi "$VENV" -e "$DEID_SCHEMA"
pipi_lock "$VENV" -r requirements/main.lock.txt
pipi "$VENV" -e . --no-deps
[ -n "$BELGIAN" ] && { echo "==> belgian-deduce from $BELGIAN"; pipi "$VENV" "$BELGIAN"; }
verify_deid_schema_runtime "$VENV/bin/python" "$HERE" "$VENV"

if [ "$DO_PF" = 1 ]; then
  echo "==> privacy-filters env (.venv-pf): frozen transformers>=5.7 stack"
  mkenv ".venv-pf" "$PYVER"
  pipi ".venv-pf" -e "$DEID_SCHEMA"
  pipi_lock ".venv-pf" -r requirements/privacy-filters.lock.txt
  pipi ".venv-pf" -e . --no-deps
  verify_deid_schema_runtime ".venv-pf/bin/python" "$HERE" ".venv-pf"
fi

if [ "$DO_DEID" = 1 ]; then
  echo "==> deidentify env (.venv-deidentify): 2020 amd64 stack"
  bash "$HERE/scripts/setup_deidentify_venv.sh" --deid-schema "$DEID_SCHEMA" .venv-deidentify
fi

echo
echo "done. Next:"
echo "  1) review the one canonical config: configs/battery.yaml"
echo "  2) input: $VENV/bin/python -m deid_battery.inputs --config configs/battery.yaml"
echo "  3) run:   $VENV/bin/python -m deid_battery.orchestrate run --config configs/battery.yaml"
[ "$DO_PF" = 1 ]   && echo "  privacy filters: set  venv: ../.venv-pf  on those models in the config"
[ "$DO_DEID" = 1 ] && echo "  deidentify:      set  venv: $(pwd)/.venv-deidentify  on that model"
