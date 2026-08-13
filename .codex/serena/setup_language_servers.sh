#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly lock_file="${repo_root}/.codex/serena/language-servers.lock"
readonly runtime_root="${repo_root}/.codex/serena/runtime"
readonly pyright_dir="${runtime_root}/language-servers/pyright"
readonly pyright_bin="${pyright_dir}/bin/pyright-langserver"
readonly pyright_complete="${pyright_dir}/.complete"
readonly node_dir="${runtime_root}/node"
readonly node_bin="${node_dir}/bin/node"
readonly node_complete="${node_dir}/.complete"
readonly typescript_dir="${repo_root}/.codex/serena/language_servers/static/TypeScriptLanguageServer/ts-lsp"
readonly typescript_bin="${typescript_dir}/node_modules/.bin/typescript-language-server"
readonly typescript_complete="${typescript_dir}/.complete"
readonly bash_dir="${repo_root}/.codex/serena/language_servers/static/BashLanguageServer/bash-lsp"
readonly bash_bin="${bash_dir}/node_modules/.bin/bash-language-server"
readonly bash_complete="${bash_dir}/.complete"
readonly shellcheck_version="$(awk -F= '$1 == "shellcheck.version" {print $2}' "${lock_file}")"
readonly shellcheck_bin="${bash_dir}/shellcheck/shellcheck-v${shellcheck_version}/shellcheck"

read_lock() {
    local key="$1"
    local value
    value="$(awk -F= -v wanted="${key}" '$1 == wanted {print substr($0, index($0, "=") + 1)}' "${lock_file}")"
    if [[ -z "${value}" ]]; then
        echo "missing ${key} in ${lock_file}" >&2
        return 1
    fi
    printf '%s\n' "${value}"
}

check_pyright() {
    local expected observed
    expected="$(read_lock pyright.version)"
    [[ -f "${pyright_complete}" && -x "${pyright_bin}" ]] || {
        echo "Pyright runtime is incomplete at ${pyright_dir}" >&2
        return 1
    }
    observed="$("${pyright_dir}/bin/python" -c 'from importlib.metadata import version; print(version("pyright"))')"
    [[ "${observed}" == "${expected}" ]] || {
        echo "Pyright version mismatch: ${observed} != ${expected}" >&2
        return 1
    }
    "${pyright_dir}/bin/pyright" --version
}

check_node() {
    local expected_hash expected_version observed_hash observed_version
    expected_hash="$(read_lock node.sha256)"
    expected_version="v$(read_lock node.version)"
    [[ -f "${node_complete}" && -x "${node_bin}" ]] || {
        echo "Node runtime is incomplete at ${node_dir}" >&2
        return 1
    }
    observed_hash="$(sha256sum "${node_bin}" | awk '{print $1}')"
    observed_version="$("${node_bin}" --version)"
    [[ "${observed_hash}" == "${expected_hash}" ]] || {
        echo "Node digest mismatch: ${observed_hash} != ${expected_hash}" >&2
        return 1
    }
    [[ "${observed_version}" == "${expected_version}" ]] || {
        echo "Node version mismatch: ${observed_version} != ${expected_version}" >&2
        return 1
    }
    printf 'node %s\n' "${observed_version}"
}

check_typescript() {
    local expected_server expected_typescript observed_server observed_typescript
    expected_server="$(read_lock typescript-language-server.version)"
    expected_typescript="$(read_lock typescript.version)"
    [[ -f "${typescript_complete}" && -x "${typescript_bin}" ]] || {
        echo "TypeScript runtime is incomplete at ${typescript_dir}" >&2
        return 1
    }
    observed_server="$("${node_bin}" -p "require('${typescript_dir}/node_modules/typescript-language-server/package.json').version")"
    observed_typescript="$("${node_bin}" -p "require('${typescript_dir}/node_modules/typescript/package.json').version")"
    [[ "${observed_server}" == "${expected_server}" ]] || {
        echo "typescript-language-server mismatch: ${observed_server} != ${expected_server}" >&2
        return 1
    }
    [[ "${observed_typescript}" == "${expected_typescript}" ]] || {
        echo "TypeScript mismatch: ${observed_typescript} != ${expected_typescript}" >&2
        return 1
    }
    printf 'typescript-language-server %s (typescript %s)\n' "${observed_server}" "${observed_typescript}"
}

check_bash() {
    local expected_bash expected_shellcheck observed_bash observed_shellcheck
    expected_bash="$(read_lock bash-language-server.version)"
    expected_shellcheck="$(read_lock shellcheck.version)"
    [[ -f "${bash_complete}" && -x "${bash_bin}" && -x "${shellcheck_bin}" ]] || {
        echo "Bash runtime is incomplete at ${bash_dir}" >&2
        return 1
    }
    observed_bash="$("${node_bin}" -p "require('${bash_dir}/node_modules/bash-language-server/package.json').version")"
    observed_shellcheck="$("${shellcheck_bin}" --version | awk '$1 == "version:" {print $2}')"
    [[ "${observed_bash}" == "${expected_bash}" ]] || {
        echo "bash-language-server mismatch: ${observed_bash} != ${expected_bash}" >&2
        return 1
    }
    [[ "${observed_shellcheck}" == "${expected_shellcheck}" ]] || {
        echo "ShellCheck mismatch: ${observed_shellcheck} != ${expected_shellcheck}" >&2
        return 1
    }
    printf 'bash-language-server %s (ShellCheck %s)\n' "${observed_bash}" "${observed_shellcheck}"
}

check_runtime() {
    check_pyright
    check_node
    check_typescript
    check_bash
}

if [[ "${1:-}" == "--check" ]]; then
    check_runtime
    exit
fi

mkdir -p "${runtime_root}/language-servers"
chmod 700 "${runtime_root}" "${runtime_root}/language-servers"

if [[ ! -e "${pyright_dir}" ]]; then
    readonly pyright_version="$(read_lock pyright.version)"
    readonly python_version="$(read_lock python.version)"
    /root/.local/bin/uv venv --python "${python_version}" "${pyright_dir}"
    /root/.local/bin/uv pip install --python "${pyright_dir}/bin/python" "pyright==${pyright_version}"
    printf 'pyright.version=%s\npython.version=%s\n' "${pyright_version}" "${python_version}" \
        > "${pyright_dir}/SERENA_LANGUAGE_SERVER_PROVENANCE"
    chmod 600 "${pyright_dir}/SERENA_LANGUAGE_SERVER_PROVENANCE"
    : > "${pyright_complete}"
    chmod 600 "${pyright_complete}"
fi
check_pyright

if [[ ! -e "${node_dir}" ]]; then
    readonly node_version="$(read_lock node.version)"
    readonly node_source="/root/.nvm/versions/node/v${node_version}/bin/node"
    readonly expected_node_hash="$(read_lock node.sha256)"
    readonly observed_node_hash="$(sha256sum "${node_source}" | awk '{print $1}')"
    [[ "${observed_node_hash}" == "${expected_node_hash}" ]] || {
        echo "Node source digest mismatch: ${observed_node_hash} != ${expected_node_hash}" >&2
        exit 1
    }
    mkdir -p "${node_dir}/bin"
    chmod 700 "${node_dir}" "${node_dir}/bin"
    install -m 0755 "${node_source}" "${node_bin}"
    : > "${node_complete}"
    chmod 600 "${node_complete}"
fi
check_node

if [[ ! -x "${typescript_bin}" || ! -f "${typescript_complete}" ]]; then
    readonly npm_bin="/root/.nvm/versions/node/v$(read_lock node.version)/bin/npm"
    readonly typescript_version="$(read_lock typescript.version)"
    readonly typescript_server_version="$(read_lock typescript-language-server.version)"
    mkdir -p "${typescript_dir}"
    chmod 700 "${repo_root}/.codex/serena/language_servers" \
        "${repo_root}/.codex/serena/language_servers/static" \
        "${repo_root}/.codex/serena/language_servers/static/TypeScriptLanguageServer" \
        "${typescript_dir}"
    "${npm_bin}" install \
        --prefix "${typescript_dir}" \
        --no-audit \
        --no-fund \
        --no-save \
        "typescript@${typescript_version}" \
        "typescript-language-server@${typescript_server_version}"
    : > "${typescript_complete}"
    chmod 600 "${typescript_complete}"
fi
check_typescript

if [[ ! -x "${bash_bin}" || ! -x "${shellcheck_bin}" || ! -f "${bash_complete}" ]]; then
    readonly npm_bin="/root/.nvm/versions/node/v$(read_lock node.version)/bin/npm"
    readonly bash_version="$(read_lock bash-language-server.version)"
    readonly shellcheck_hash="$(read_lock shellcheck.linux-x64.sha256)"
    readonly shellcheck_url="https://github.com/koalaman/shellcheck/releases/download/v${shellcheck_version}/shellcheck-v${shellcheck_version}.linux.x86_64.tar.xz"
    readonly shellcheck_parent="${bash_dir}/shellcheck"
    readonly shellcheck_archive="$(mktemp /tmp/serena-shellcheck.XXXXXX.tar.xz)"
    cleanup_shellcheck() {
        rm -f -- "${shellcheck_archive}"
    }
    trap cleanup_shellcheck EXIT
    mkdir -p "${bash_dir}" "${shellcheck_parent}"
    chmod 700 "${repo_root}/.codex/serena/language_servers/static/BashLanguageServer" \
        "${bash_dir}" "${shellcheck_parent}"
    PATH="${node_dir}/bin:/root/.nvm/versions/node/v$(read_lock node.version)/bin:/usr/local/bin:/usr/bin:/bin" \
        "${npm_bin}" install \
        --prefix "${bash_dir}" \
        --no-audit \
        --no-fund \
        --no-save \
        "bash-language-server@${bash_version}"
    curl --fail --location --silent --show-error "${shellcheck_url}" --output "${shellcheck_archive}"
    printf '%s  %s\n' "${shellcheck_hash}" "${shellcheck_archive}" | sha256sum --check --status
    tar -xJf "${shellcheck_archive}" -C "${shellcheck_parent}"
    chmod 755 "${shellcheck_bin}"
    : > "${bash_complete}"
    chmod 600 "${bash_complete}"
    cleanup_shellcheck
    trap - EXIT
fi
check_bash
