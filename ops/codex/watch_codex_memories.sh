#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ops/codex/watch_codex_memories.sh [--debounce SECONDS] [--poll SECONDS]

Watches .codex/memories for Markdown changes and commits them with:

  refresh memories

The watcher only stages .codex/memories/**/*.md and skips when unrelated
changes are already staged or when a merge/rebase operation is active.
EOF
}

debounce_seconds="${CODEX_MEMORY_WATCH_DEBOUNCE_SECONDS:-15}"
poll_seconds="${CODEX_MEMORY_WATCH_POLL_SECONDS:-30}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --debounce)
      if [[ $# -lt 2 ]]; then
        echo "error: --debounce requires a value" >&2
        exit 2
      fi
      debounce_seconds="$2"
      shift 2
      ;;
    --poll)
      if [[ $# -lt 2 ]]; then
        echo "error: --poll requires a value" >&2
        exit 2
      fi
      poll_seconds="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

repo_root="$(git rev-parse --show-toplevel)"
cd "$repo_root"

commit_script="$repo_root/ops/codex/commit_codex_memories.sh"

commit_after_debounce() {
  sleep "$debounce_seconds"
  "$commit_script" --message "refresh memories"
}

has_memory_status() {
  [[ -n "$(git status --porcelain -- .codex/memories)" ]]
}

echo "watching .codex/memories Markdown changes in $repo_root"
echo "debounce=${debounce_seconds}s poll=${poll_seconds}s"

"$commit_script" --message "refresh memories"

if command -v inotifywait >/dev/null 2>&1; then
  while true; do
    inotifywait -q -r \
      -e close_write,create,delete,move \
      --exclude '(^|/)\.git(/|$)' \
      .codex/memories >/dev/null
    commit_after_debounce
  done
fi

echo "inotifywait not found; falling back to polling" >&2

while true; do
  if has_memory_status; then
    commit_after_debounce
  fi
  sleep "$poll_seconds"
done
