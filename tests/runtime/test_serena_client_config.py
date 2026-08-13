from __future__ import annotations

import json
import tomllib
from pathlib import Path

import yaml


ROOT = Path("/data/CoordExp")
EXPECTED_COMMAND = "/usr/bin/setpriv"
EXPECTED_ARGS = [
    "--pdeathsig",
    "TERM",
    "/root/miniconda3/envs/ms/bin/python",
    "/data/CoordExp/.codex/serena/serena_worktree_mcp.py",
    "serve",
]
EXPECTED_ENV = {
    "CONDA_PREFIX": "/root/miniconda3/envs/ms",
    "PATH": "/root/miniconda3/envs/ms/bin:/root/.local/bin:/usr/local/bin:/usr/bin:/bin",
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


def test_all_codex_and_claude_registrations_use_shared_wrapper() -> None:
    codex = tomllib.loads((ROOT / ".codex" / "config.toml").read_text())["mcp_servers"]["serena"]
    root_mcp = json.loads((ROOT / ".mcp.json").read_text())["mcpServers"]["serena"]
    light_mcp = json.loads((ROOT / "serena-light" / ".mcp.json").read_text())["mcpServers"]["serena"]
    claude = json.loads((ROOT / ".claude" / ".claude.json").read_text())["mcpServers"]["serena"]

    for registration in (codex, root_mcp, light_mcp, claude):
        _assert_registration(registration)


def test_context_is_worktree_bound_with_exact_tool_surface() -> None:
    context = yaml.safe_load(
        (ROOT / ".codex" / "serena" / "contexts" / "coordexp-minimal.yml").read_text()
    )

    assert context["single_project"] is True
    assert context["fixed_tools"] == EXPECTED_TOOLS
    assert "shared by Agents started in the same Git worktree" in context["prompt"]
    assert "never switch this shared process to another project" in context["prompt"]


def test_blocking_serena_reminders_remain_registered() -> None:
    codex_hooks = json.loads((ROOT / ".codex" / "hooks.json").read_text())
    claude_settings = json.loads((ROOT / ".claude" / "settings.json").read_text())
    codex_text = json.dumps(codex_hooks, sort_keys=True)
    claude_text = json.dumps(claude_settings, sort_keys=True)

    assert codex_text.count("serena-hooks remind --client=codex") == 3
    assert claude_text.count("serena-hooks remind --client=claude-code") == 1
