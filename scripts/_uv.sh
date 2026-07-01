# Shared helpers to locate or install uv. Source this (don't execute it); it sets
# the global $UV to an absolute uv path via ensure_uv.
#
# sudo note: sudo resets PATH (secure_path), hiding a per-user uv in ~/.local/bin.
# find_uv also checks the invoking user's home ($SUDO_USER) so it survives sudo.

find_uv() {
  local p h
  p="$(command -v uv 2>/dev/null)" && { echo "$p"; return 0; }
  local homes=("$HOME")
  [ -n "${SUDO_USER:-}" ] && homes+=("$(eval echo "~$SUDO_USER")")
  for h in "${homes[@]}"; do
    for p in "$h/.local/bin/uv" "$h/.cargo/bin/uv"; do
      [ -x "$p" ] && { echo "$p"; return 0; }
    done
  done
  return 1
}

# Sets $UV; installs uv if missing (opt out with NO_AUTO_INSTALL_UV=1). Returns 1
# if uv still can't be found.
ensure_uv() {
  UV="$(find_uv || true)"
  if [ -z "$UV" ] && [ "${NO_AUTO_INSTALL_UV:-0}" != "1" ]; then
    echo "uv not found — installing it (set NO_AUTO_INSTALL_UV=1 to skip) ..." >&2
    curl -LsSf https://astral.sh/uv/install.sh | sh
    UV="$(find_uv || true)"
  fi
  [ -n "$UV" ] || { echo "uv not found. Install it:  curl -LsSf https://astral.sh/uv/install.sh | sh" >&2; return 1; }
  echo "using uv: $UV" >&2
}
