from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
LANGUAGE_SETUP = REPO_ROOT / ".codex" / "serena" / "setup_language_servers.sh"
FULL_SETUP = REPO_ROOT / ".codex" / "serena" / "setup.sh"


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
    assert "mcp-proxy" not in result.stdout
    assert "pyright 1.1.403" in result.stdout
    assert "typescript-language-server 5.1.3" in result.stdout
    assert "bash-language-server 5.6.0" in result.stdout
    assert "official Serena stdio runtime is ready" in result.stdout
