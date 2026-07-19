#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_path="${repo_root}/.codex/serena/contexts/coordexp-codex.yml"
target_path="${HOME}/.serena/contexts/coordexp-codex.yml"

if [[ "${1:-}" == "--check" ]]; then
    cmp --silent "${source_path}" "${target_path}"
    context_list="$(serena context list)"
    grep --quiet '^coordexp-codex ' <<<"${context_list}"
    echo "Serena context is installed and matches ${source_path}"
    exit 0
fi

install -D -m 0644 "${source_path}" "${target_path}"
cmp --silent "${source_path}" "${target_path}"
context_list="$(serena context list)"
grep --quiet '^coordexp-codex ' <<<"${context_list}"
echo "Installed ${target_path}"
