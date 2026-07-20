#!/usr/bin/env python3
"""Wrap noisy Codex shell commands in RTK before execution."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


PYTHON_NAMES = {"python", "python3", "python3.10", "python3.11", "python3.12"}
MACHINE_OUTPUT_FLAGS = {"--json", "-json", "--porcelain", "-z"}
SHELL_NAMES = {"bash", "dash", "sh", "zsh"}
SUPPORTED_TOOL_NAMES = {"Bash", "shell", "exec_command", "functions.exec_command"}
CONDA_RUN_FLAGS = {
    "--debug-wrapper-scripts",
    "--dev",
    "--live-stream",
    "--no-capture-output",
    "-v",
    "--verbose",
}
CONDA_RUN_OPTIONS_WITH_VALUES = {"-n", "--name", "-p", "--prefix", "--cwd"}
NOISY_COMMANDS = {
    "bun",
    "cargo",
    "docker",
    "eslint",
    "go",
    "goctl",
    "grep",
    "kubectl",
    "make",
    "mypy",
    "npm",
    "npx",
    "pnpm",
    "prettier",
    "pytest",
    "rg",
    "ruff",
    "tsc",
    "vite",
    "vue-tsc",
    "yarn",
}
INFO_COMMAND_ARGS = {"--help", "-h", "--version", "version", "-version"}
SHELL_OPERATOR_TOKENS = {"|", "||", "&&", ";", "&", ">", ">>", "<", "2>", "2>>"}


def main() -> int:
    if os.environ.get("RTK_HOOK_DISABLE") == "1":
        return 0

    raw = sys.stdin.read()
    if not raw.strip() or shutil.which("rtk") is None:
        return 0

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return 0

    tool_name = str(payload.get("tool_name") or "")
    if tool_name and tool_name not in SUPPORTED_TOOL_NAMES:
        return 0

    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0

    command_key = "command" if "command" in tool_input else "cmd"
    command = tool_input.get(command_key)
    if not isinstance(command, str) or not command.strip():
        return 0

    try:
        rewritten = rewrite_command(command)
    except Exception:
        # A token-saving hook must never prevent the underlying Codex tool
        # invocation.  Unknown command syntax falls back to the raw command.
        return 0
    if rewritten is None or rewritten == command:
        return 0

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "allow",
                    "updatedInput": {command_key: rewritten},
                }
            },
            separators=(",", ":"),
        )
    )
    return 0


def rewrite_command(command: str) -> str | None:
    tokens = split_command(command)
    if not tokens:
        return None
    if command_requests_hook_disable(tokens):
        return None

    shell_wrapped = unwrap_shell(tokens)
    if shell_wrapped is not None:
        prefix, inner = shell_wrapped
        rewritten_inner = rewrite_simple_command(inner)
        if rewritten_inner is None:
            return None
        return shlex.join([*prefix, rewritten_inner])

    conda_wrapped = unwrap_conda_run(tokens)
    if conda_wrapped is not None:
        prefix, inner = conda_wrapped
        rewritten_inner = rewrite_simple_command(shlex.join(inner))
        if rewritten_inner is None:
            return None
        rewritten_inner_tokens = split_command(rewritten_inner)
        if not rewritten_inner_tokens:
            return None
        # Unlike ``bash -c``, conda receives the executable and its arguments
        # as separate argv entries.  Do not quote the rewritten command as one
        # argument, or conda would look for an executable named ``rtk pytest``.
        return shlex.join([*prefix, *rewritten_inner_tokens])

    return rewrite_simple_command(command)


def rewrite_simple_command(command: str) -> str | None:
    stripped = command.strip()
    if stripped.startswith("rtk ") or stripped == "rtk":
        return None

    tokens = split_command(stripped)
    if not tokens:
        return None
    if any(token in SHELL_OPERATOR_TOKENS for token in tokens):
        return None

    candidate_index = find_candidate_index(tokens)
    if candidate_index is None:
        return None

    candidate = Path(tokens[candidate_index]).name
    if candidate == "rtk":
        return None

    if should_skip_exact_output(tokens, candidate_index):
        return None

    rewritten = rewrite_with_rtk(stripped)
    if rewritten is not None:
        safe_rewrite = repair_or_reject_rtk_rewrite(tokens, candidate_index, rewritten)
        if safe_rewrite is None:
            return wrap_command_with_rtk(stripped, tokens, candidate_index)
        rewritten = safe_rewrite
    if rewritten is None or rewritten == stripped:
        if candidate in NOISY_COMMANDS and not is_tiny_info_command(tokens, candidate_index):
            return wrap_command_with_rtk(stripped, tokens, candidate_index)
        if os.environ.get("RTK_HOOK_AGGRESSIVE") == "1":
            return shlex.join(["rtk", "proxy", *tokens])
        return None

    return rewritten


def wrap_command_with_rtk(command: str, tokens: list[str], candidate_index: int) -> str | None:
    """Insert RTK while preserving shell operators when the command starts there."""

    if candidate_index == 0:
        return f"rtk {command}"
    if any(token in SHELL_OPERATOR_TOKENS for token in tokens):
        return None
    return wrap_with_rtk(tokens, candidate_index)


def wrap_with_rtk(tokens: list[str], candidate_index: int) -> str:
    """Insert RTK before known noisy commands when `rtk rewrite` has no mapping."""

    return shlex.join([*tokens[:candidate_index], "rtk", *tokens[candidate_index:]])


def is_tiny_info_command(tokens: list[str], candidate_index: int) -> bool:
    args = tokens[candidate_index + 1 :]
    return bool(args) and args[0] in INFO_COMMAND_ARGS


def split_command(command: str) -> list[str] | None:
    try:
        return shlex.split(command.strip(), posix=True)
    except ValueError:
        return None


def unwrap_shell(tokens: list[str]) -> tuple[list[str], str] | None:
    candidate_index = find_candidate_index(tokens)
    if candidate_index is None:
        return None

    shell = Path(tokens[candidate_index]).name
    if shell not in SHELL_NAMES:
        return None

    idx = candidate_index + 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-c", "-lc"} and idx + 1 < len(tokens):
            return tokens[: idx + 1], tokens[idx + 1]
        if token in {"-l", "--login"}:
            idx += 1
            continue
        break
    return None


def unwrap_conda_run(tokens: list[str]) -> tuple[list[str], list[str]] | None:
    """Find the executable inside a ``conda run`` invocation.

    RTK must be inserted after the conda environment-selection options so the
    wrapped command still resolves tools and plugins from that environment.
    Unknown options deliberately fail open instead of risking a misplaced
    insertion.
    """

    candidate_index = find_candidate_index(tokens)
    if candidate_index is None:
        return None
    if Path(tokens[candidate_index]).name != "conda":
        return None
    if candidate_index + 1 >= len(tokens) or tokens[candidate_index + 1] != "run":
        return None

    idx = candidate_index + 2
    while idx < len(tokens):
        token = tokens[idx]
        if token == "--":
            idx += 1
            break
        if token in CONDA_RUN_FLAGS:
            idx += 1
            continue
        if token in CONDA_RUN_OPTIONS_WITH_VALUES:
            if idx + 1 >= len(tokens):
                return None
            idx += 2
            continue
        if any(
            token.startswith(f"{option}=")
            for option in CONDA_RUN_OPTIONS_WITH_VALUES
            if option.startswith("--")
        ):
            idx += 1
            continue
        if token.startswith("-"):
            return None
        return tokens[:idx], tokens[idx:]
    return None


def rewrite_with_rtk(command: str) -> str | None:
    try:
        result = subprocess.run(
            ["rtk", "rewrite", command],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    rewritten = result.stdout.strip()
    return rewritten or None


def repair_or_reject_rtk_rewrite(
    original_tokens: list[str], candidate_index: int, rewritten: str
) -> str | None:
    """Keep RTK rewrites only when they preserve the command argv.

    `rtk rewrite` prints a shell command, so quoted arguments can come back
    flattened (for example pytest `-k 'a or b'`).  Rebuild known-safe rewrites
    from the original argv; reject known-bad ones so the caller can fall back.
    """

    candidate = Path(original_tokens[candidate_index]).name
    original_args = original_tokens[candidate_index + 1 :]

    rewritten_tokens = split_command(rewritten)
    if not rewritten_tokens:
        return None

    prefix = original_tokens[:candidate_index]
    rtk_index = rewritten_tokens.index("rtk") if "rtk" in rewritten_tokens else None
    if rtk_index is None:
        return None
    # RTK may place itself after a transparent runner prefix (for example,
    # ``uv run rtk pytest``).  This is safe whenever removing the inserted
    # token restores the original argv byte-for-byte at the shell-token level.
    if rewritten_tokens[:rtk_index] + rewritten_tokens[rtk_index + 1 :] == original_tokens:
        return shlex.join(rewritten_tokens)

    target = rtk_target_after(rewritten_tokens, rtk_index)
    if candidate == "rg" and "--files" in original_args and target == "grep":
        return None

    rewritten_payload = rewritten_tokens[rtk_index + 1 :]
    original_payload = original_tokens[candidate_index:]
    if rewritten_payload == original_payload:
        return shlex.join([*prefix, "rtk", *original_payload])

    # RTK 0.43.0 treats ``uv run`` as a transparent prefix and places ``rtk``
    # between it and pytest.  The generic candidate finder sees ``uv`` first,
    # so retain this rewrite only for the two pytest forms whose argv we can
    # reconstruct exactly from the original command.
    if candidate == "uv" and original_args[:1] == ["run"] and target == "pytest":
        inner_args = original_args[1:]
        if inner_args[:1] == ["pytest"]:
            return shlex.join([*prefix, "uv", "run", "rtk", "pytest", *inner_args[1:]])
        if (
            inner_args[:1]
            and Path(inner_args[0]).name in PYTHON_NAMES
            and inner_args[1:3] == ["-m", "pytest"]
        ):
            return shlex.join([*prefix, "uv", "run", "rtk", "pytest", *inner_args[3:]])

    if target == "pytest":
        if candidate == "pytest":
            return shlex.join([*prefix, "rtk", *original_payload])
        if candidate in PYTHON_NAMES and original_args[:2] == ["-m", "pytest"]:
            return shlex.join([*prefix, "rtk", "pytest", *original_args[2:]])

    return None


def find_rtk_target_name(tokens: list[str]) -> str | None:
    idx = find_candidate_index(tokens)
    if idx is None or Path(tokens[idx]).name != "rtk" or idx + 1 >= len(tokens):
        return None
    if tokens[idx + 1] == "proxy" and idx + 2 < len(tokens):
        return Path(tokens[idx + 2]).name
    return Path(tokens[idx + 1]).name


def rtk_target_after(tokens: list[str], rtk_index: int) -> str | None:
    """Return the RTK target when RTK is embedded after a transparent prefix."""

    if rtk_index + 1 >= len(tokens):
        return None
    if tokens[rtk_index + 1] == "proxy" and rtk_index + 2 < len(tokens):
        return Path(tokens[rtk_index + 2]).name
    return Path(tokens[rtk_index + 1]).name


def find_candidate_index(tokens: list[str]) -> int | None:
    idx = 0
    while idx < len(tokens):
        token = tokens[idx]
        if is_shell_assignment(token):
            idx += 1
            continue
        if token == "env":
            idx += 1
            while idx < len(tokens):
                current = tokens[idx]
                if current in {"-u", "--unset"} and idx + 1 < len(tokens):
                    idx += 2
                    continue
                if current.startswith("-"):
                    idx += 1
                    continue
                if "=" in current and not current.startswith("="):
                    idx += 1
                    continue
                break
            continue
        if token == "timeout":
            idx += 1
            while idx < len(tokens) and tokens[idx].startswith("-"):
                idx += 1
            if idx < len(tokens):
                idx += 1
            continue
        return idx
    return None


def command_requests_hook_disable(tokens: list[str]) -> bool:
    idx = 0
    while idx < len(tokens) and is_shell_assignment(tokens[idx]):
        if tokens[idx] == "RTK_HOOK_DISABLE=1":
            return True
        idx += 1

    if idx >= len(tokens) or tokens[idx] != "env":
        return False

    idx += 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-u", "--unset"} and idx + 1 < len(tokens):
            idx += 2
            continue
        if token.startswith("-"):
            idx += 1
            continue
        if is_shell_assignment(token):
            if token == "RTK_HOOK_DISABLE=1":
                return True
            idx += 1
            continue
        break
    return False


def is_shell_assignment(token: str) -> bool:
    name, separator, _ = token.partition("=")
    if separator != "=" or not name:
        return False
    if not (name[0].isalpha() or name[0] == "_"):
        return False
    return all(character.isalnum() or character == "_" for character in name)


def should_skip_exact_output(tokens: list[str], idx: int) -> bool:
    candidate = Path(tokens[idx]).name
    if candidate in {"awk", "cat", "date", "jq", "nl", "pwd", "sed", "wc"}:
        return True
    if candidate == "find" and not is_simple_rtk_find_shape(tokens[idx + 1 :]):
        return True
    if any(flag in MACHINE_OUTPUT_FLAGS for flag in tokens[idx + 1 :]):
        return True
    if "-o" in tokens and "json" in tokens:
        return True
    if candidate in PYTHON_NAMES and idx + 1 < len(tokens) and tokens[idx + 1] == "-c":
        return True
    if candidate == "node" and idx + 1 < len(tokens) and tokens[idx + 1] == "-e":
        return True
    return False


def is_simple_rtk_find_shape(args: list[str]) -> bool:
    """Return True only for simple searches that RTK can safely rewrite.

    Native find predicates/actions are intentionally skipped because RTK's find
    shorthand does not support the full find expression language.
    """

    find_expression_tokens = {
        "!",
        "(",
        ")",
        "-a",
        "-and",
        "-delete",
        "-exec",
        "-execdir",
        "-false",
        "-fls",
        "-fprint",
        "-fprint0",
        "-fprintf",
        "-ls",
        "-not",
        "-o",
        "-ok",
        "-okdir",
        "-or",
        "-print",
        "-print0",
        "-printf",
        "-prune",
        "-quit",
        "-true",
    }
    return not any(arg.startswith("-") or arg in find_expression_tokens for arg in args)


if __name__ == "__main__":
    raise SystemExit(main())
