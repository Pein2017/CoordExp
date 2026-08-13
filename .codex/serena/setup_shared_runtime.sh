#!/usr/bin/env bash
set -euo pipefail

readonly repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd -P)"
readonly lock_file="${repo_root}/.codex/serena/bridge.lock"
readonly runtime_dir="${repo_root}/.codex/serena/runtime/bridge"
readonly bridge_bin="${runtime_dir}/bin/mcp-proxy"
readonly complete_marker="${runtime_dir}/.complete"

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

check_runtime() {
    local observed_proxy observed_mcp expected_proxy expected_mcp
    expected_proxy="$(read_lock mcp-proxy.version)"
    expected_mcp="$(read_lock mcp.version)"
    [[ -f "${complete_marker}" ]] || {
        echo "bridge runtime is incomplete at ${runtime_dir}" >&2
        return 1
    }
    [[ -x "${bridge_bin}" ]] || {
        echo "bridge runtime is not installed at ${bridge_bin}" >&2
        return 1
    }
    observed_proxy="$("${runtime_dir}/bin/python" -c 'from importlib.metadata import version; print(version("mcp-proxy"))')"
    observed_mcp="$("${runtime_dir}/bin/python" -c 'from importlib.metadata import version; print(version("mcp"))')"
    [[ "${observed_proxy}" == "${expected_proxy}" ]] || {
        echo "mcp-proxy version mismatch: ${observed_proxy} != ${expected_proxy}" >&2
        return 1
    }
    [[ "${observed_mcp}" == "${expected_mcp}" ]] || {
        echo "mcp version mismatch: ${observed_mcp} != ${expected_mcp}" >&2
        return 1
    }
    "${bridge_bin}" --version
}

if [[ "${1:-}" == "--check" ]]; then
    check_runtime
    exit
fi

if [[ -e "${runtime_dir}" ]]; then
    check_runtime
    exit
fi

readonly proxy_revision="$(read_lock mcp-proxy.git)"
readonly proxy_version="$(read_lock mcp-proxy.version)"
readonly mcp_version="$(read_lock mcp.version)"
readonly runtime_parent="$(dirname "${runtime_dir}")"
mkdir -p "${runtime_parent}"
chmod 700 "${runtime_parent}"
installing=1
cleanup() {
    if [[ "${installing:-0}" == "1" && -d "${runtime_dir}" ]]; then
        rm -rf -- "${runtime_dir}"
    fi
}
trap cleanup EXIT

/root/.local/bin/uv venv --python /root/miniconda3/envs/ms/bin/python "${runtime_dir}"
/root/.local/bin/uv pip install \
    --python "${runtime_dir}/bin/python" \
    "mcp==${mcp_version}" \
    "mcp-proxy @ git+https://github.com/sparfenyuk/mcp-proxy.git@${proxy_revision}"
cat > "${runtime_dir}/SERENA_BRIDGE_PROVENANCE" <<EOF
mcp-proxy.git=${proxy_revision}
mcp-proxy.version=${proxy_version}
mcp.version=${mcp_version}
EOF
chmod 600 "${runtime_dir}/SERENA_BRIDGE_PROVENANCE"
: > "${complete_marker}"
chmod 600 "${complete_marker}"
installing=0
check_runtime
