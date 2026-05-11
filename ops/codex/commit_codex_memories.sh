#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ops/codex/commit_codex_memories.sh [--dry-run] [--message MESSAGE]

Stages and commits only tracked or untracked Markdown files under
.codex/memories/. Non-Markdown runtime files are ignored.
EOF
}

dry_run=0
message="refresh memories"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      dry_run=1
      shift
      ;;
    --message)
      if [[ $# -lt 2 ]]; then
        echo "error: --message requires a value" >&2
        exit 2
      fi
      message="$2"
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

if [[ "$dry_run" -eq 1 ]]; then
  real_index="$(git rev-parse --git-path index)"
  tmp_index="$(mktemp)"
  if [[ -e "$real_index" ]]; then
    cp "$real_index" "$tmp_index"
  fi
  export GIT_INDEX_FILE="$tmp_index"
  trap 'rm -f "$tmp_index"' EXIT
fi

is_memory_markdown() {
  [[ "$1" =~ ^\.codex/memories/.*\.md$ ]]
}

has_git_operation_in_progress() {
  local git_dir
  git_dir="$(git rev-parse --git-dir)"

  [[ -e "$git_dir/MERGE_HEAD" ]] && return 0
  [[ -e "$git_dir/CHERRY_PICK_HEAD" ]] && return 0
  [[ -e "$git_dir/REVERT_HEAD" ]] && return 0
  [[ -d "$git_dir/rebase-merge" ]] && return 0
  [[ -d "$git_dir/rebase-apply" ]] && return 0

  return 1
}

assert_no_staged_non_memory_changes() {
  local path

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    if ! is_memory_markdown "$path"; then
      echo "skip: staged non-memory change exists: $path" >&2
      exit 0
    fi
  done < <(git diff --cached --name-only)
}

stage_memory_markdown_changes() {
  local path

  while IFS= read -r -d '' path; do
    git add -- "$path"
  done < <(find .codex/memories -type f -name '*.md' -print0 2>/dev/null)

  while IFS= read -r -d '' path; do
    if is_memory_markdown "$path"; then
      git add -- "$path"
    fi
  done < <(git ls-files -z --deleted -- .codex/memories)
}

assert_only_memory_markdown_staged() {
  local path
  local has_staged=0

  while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    has_staged=1
    if ! is_memory_markdown "$path"; then
      echo "error: staged non-memory change after staging: $path" >&2
      exit 1
    fi
  done < <(git diff --cached --name-only)

  [[ "$has_staged" -eq 1 ]]
}

if has_git_operation_in_progress; then
  echo "skip: git merge/rebase/cherry-pick/revert in progress" >&2
  exit 0
fi

assert_no_staged_non_memory_changes
stage_memory_markdown_changes

if ! assert_only_memory_markdown_staged; then
  echo "skip: no .codex/memories Markdown changes to commit"
  exit 0
fi

if [[ "$dry_run" -eq 1 ]]; then
  echo "dry-run: would commit the following memory Markdown changes:"
  git diff --cached --name-status
  exit 0
fi

git commit --no-verify -m "$message"
