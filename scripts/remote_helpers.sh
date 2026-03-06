#!/usr/bin/env zsh

load_remote_cmd() {
  local remote_script="$1"
  local raw

  if [[ ! -f "$remote_script" ]]; then
    printf 'remote script not found: %s\n' "$remote_script" >&2
    return 1
  fi

  raw="$(tr -d '\n' < "$remote_script")"
  read -r -A REMOTE_CMD <<<"$raw"
  if [[ "${#REMOTE_CMD[@]}" -lt 4 ]]; then
    printf 'unsupported remote script format: %s\n' "$raw" >&2
    return 1
  fi
}

run_remote_script() {
  local script="$1"
  local quoted
  quoted="${script//\'/\'\"\'\"\'}"
  quoted="'$quoted'"
  "${REMOTE_CMD[@]}" "bash -lc $quoted"
}
