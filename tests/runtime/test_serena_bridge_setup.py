from __future__ import annotations

import subprocess
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPO_ROOT / ".codex" / "serena" / "setup_shared_runtime.sh"
LANGUAGE_SETUP = REPO_ROOT / ".codex" / "serena" / "setup_language_servers.sh"
FULL_SETUP = REPO_ROOT / ".codex" / "serena" / "setup.sh"


def test_bridge_check_accepts_exact_installed_runtime() -> None:
    result = subprocess.run(
        [str(SETUP), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "mcp-proxy" in result.stdout


def test_installed_bridge_is_bound_to_exact_git_revision() -> None:
    runtime = REPO_ROOT / ".codex" / "serena" / "runtime" / "bridge"
    direct_url = next(
        runtime.glob("lib/python3.12/site-packages/mcp_proxy-*.dist-info/direct_url.json")
    )
    payload = json.loads(direct_url.read_text())

    assert payload["url"] == "https://github.com/sparfenyuk/mcp-proxy.git"
    assert payload["vcs_info"] == {
        "commit_id": "153a96a61fde2bf5a23961c64a3dd96b5e385108",
        "requested_revision": "153a96a61fde2bf5a23961c64a3dd96b5e385108",
        "vcs": "git",
    }


def test_language_server_check_accepts_exact_installed_pyright() -> None:
    result = subprocess.run(
        [str(LANGUAGE_SETUP), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "pyright 1.1.403" in result.stdout
    assert "node v22.22.0" in result.stdout
    assert "typescript-language-server 5.1.3" in result.stdout
    assert "bash-language-server 5.6.0" in result.stdout
    assert "ShellCheck 0.10.0" in result.stdout


def test_full_setup_checks_official_serena_and_all_runtime_components() -> None:
    result = subprocess.run(
        [str(FULL_SETUP), "--check"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Serena 1.7.0" in result.stdout
    assert "mcp-proxy 0.12.0" in result.stdout
    assert "pyright 1.1.403" in result.stdout
    assert "typescript-language-server 5.1.3" in result.stdout
    assert "bash-language-server 5.6.0" in result.stdout
    assert "official Serena shared runtime is ready" in result.stdout
