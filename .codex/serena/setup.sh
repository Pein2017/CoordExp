#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
source_root="${repo_root}/.codex/serena"
serena_home="${SERENA_HOME:-/data/CoordExp/.codex/runtime/serena}"
serena_command="${SERENA_COMMAND:-serena}"

verify() {
    cmp --silent \
        "${source_root}/contexts/coordexp-codex.yml" \
        "${serena_home}/contexts/coordexp-codex.yml"
    cmp --silent \
        "${source_root}/prompt_templates/system_prompt.yml" \
        "${serena_home}/prompt_templates/system_prompt.yml"

    context_list="$(SERENA_HOME="${serena_home}" SERENA_USAGE_REPORTING=false "${serena_command}" context list)"
    grep --quiet '^coordexp-codex ' <<<"${context_list}"

    instructions="$(
        SERENA_HOME="${serena_home}" SERENA_USAGE_REPORTING=false \
            "${serena_command}" print-system-prompt \
            --only-instructions \
            --context coordexp-codex \
            --mode no-memories \
            "${repo_root}"
    )"
    instruction_words="$(wc -w <<<"${instructions}")"
    if (( instruction_words > 160 )); then
        echo "Serena initial instructions are too long: ${instruction_words} words" >&2
        exit 1
    fi

    echo "Serena setup matches ${source_root}; initial instructions: ${instruction_words} words"
}

if [[ "${1:-}" == "--check" ]]; then
    verify
    exit 0
fi

install -D -m 0644 \
    "${source_root}/contexts/coordexp-codex.yml" \
    "${serena_home}/contexts/coordexp-codex.yml"
install -D -m 0644 \
    "${source_root}/prompt_templates/system_prompt.yml" \
    "${serena_home}/prompt_templates/system_prompt.yml"
verify
