from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml


ROOT = Path("/data/CoordExp")
EXPECTED_COMMAND = "/root/.local/bin/serena"
EXPECTED_ARGS = [
    "start-mcp-server",
    "--transport",
    "stdio",
    "--project-from-cwd",
    "--context",
    "coordexp-minimal",
    "--enable-web-dashboard",
    "false",
    "--enable-gui-log-window",
    "false",
    "--open-web-dashboard",
    "false",
    "--log-level",
    "ERROR",
]
EXPECTED_ENV = {
    "CONDA_PREFIX": "/root/miniconda3/envs/ms",
    "PATH": (
        "/data/CoordExp/.codex/serena/runtime/node/bin:"
        "/root/miniconda3/envs/ms/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin"
    ),
    "SERENA_HOME": "/data/CoordExp/.codex/serena",
    "VIRTUAL_ENV": "/root/miniconda3/envs/ms",
}
EXPECTED_TOOLS = [
    "initial_instructions",
    "activate_project",
    "get_current_config",
    "search_for_pattern",
    "get_symbols_overview",
    "find_symbol",
    "find_referencing_symbols",
    "find_implementations",
    "find_declaration",
    "get_diagnostics_for_file",
    "get_diagnostics_for_symbol",
    "rename_symbol",
    "replace_symbol_body",
    "insert_before_symbol",
    "insert_after_symbol",
]


def _assert_registration(registration: dict[str, object]) -> None:
    assert registration["command"] == EXPECTED_COMMAND
    assert registration["args"] == EXPECTED_ARGS
    assert registration["env"] == EXPECTED_ENV


def test_all_codex_and_claude_registrations_use_independent_official_stdio() -> None:
    codex = tomllib.loads((ROOT / ".codex" / "config.toml").read_text())["mcp_servers"]["serena"]
    root_mcp = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["serena"]
    light_mcp = json.loads((ROOT / "serena-light" / ".mcp.json").read_text())["mcpServers"]["serena"]
    claude = json.loads((ROOT / ".claude" / ".claude.json").read_text())["mcpServers"]["serena"]

    for registration in (codex, root_mcp, light_mcp, claude):
        _assert_registration(registration)


def test_retired_shared_transport_is_absent() -> None:
    retired_paths = (
        ROOT / ".codex" / "serena" / "serena_worktree_mcp.py",
        ROOT / ".codex" / "serena" / "setup_shared_runtime.sh",
        ROOT / ".codex" / "serena" / "bridge.lock",
        ROOT / "tests" / "runtime" / "test_serena_worktree_mcp.py",
    )

    assert all(not path.exists() for path in retired_paths)


def test_context_is_worktree_bound_with_exact_tool_surface() -> None:
    context = yaml.safe_load(
        (ROOT / ".codex" / "serena" / "contexts" / "coordexp-minimal.yml").read_text()
    )

    # Serena omits activate_project from the live tool list in single-project mode.
    # The user-approved surface requires the tool for explicit root confirmation.
    assert context["single_project"] is False
    assert context["fixed_tools"] == EXPECTED_TOOLS
    assert "private to this Agent session" in context["prompt"]
    assert "activate_project" in context["prompt"]


def test_blocking_serena_reminders_remain_registered() -> None:
    codex_hooks = json.loads((ROOT / ".codex" / "hooks.json").read_text())
    claude_settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    codex_text = json.dumps(codex_hooks, sort_keys=True)
    claude_text = json.dumps(claude_settings, sort_keys=True)

    assert codex_text.count("serena-hooks remind --client=codex") == 3
    assert claude_text.count("serena-hooks remind --client=claude-code") == 1


def test_serena_uses_service_owned_pyright_without_uvx() -> None:
    config = yaml.safe_load((ROOT / ".codex" / "serena" / "serena_config.yml").read_text())

    assert config["ls_specific_settings"]["python"]["ls_path"] == (
        "/data/CoordExp/.codex/serena/runtime/language-servers/pyright/bin/pyright-langserver"
    )
