#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly serena_bin="/root/.local/bin/serena"
readonly language_setup="${repo_root}/.codex/serena/setup_language_servers.sh"

check_official_serena() {
    local version
    [[ -x "${serena_bin}" ]] || {
        echo "official Serena is not installed at ${serena_bin}" >&2
        return 1
    }
    version="$("${serena_bin}" --version)"
    [[ "${version}" == "Serena 1.7.0" ]] || {
        echo "unexpected official Serena version: ${version}" >&2
        return 1
    }
    printf '%s\n' "${version}"
}

check_all() {
    check_official_serena
    "${language_setup}" --check
    printf 'official Serena stdio runtime is ready\n'
}

if [[ "${1:-}" == "--check" ]]; then
    check_all
    exit
fi

check_official_serena
"${language_setup}"
check_all
