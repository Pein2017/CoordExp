from __future__ import annotations

import subprocess
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SETUP = REPO_ROOT / ".codex" / "serena" / "setup_shared_runtime.sh"


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
