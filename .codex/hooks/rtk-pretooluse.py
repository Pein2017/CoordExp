#!/usr/bin/env python3
"""Conservative, shared RTK rewrite owner for Codex and Pi.

Only human-facing git/search/direct-pytest commands are eligible. A None result
means execute the original bytes. RTK output is lossy, never research evidence.
"""
from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SUPPORTED_TOOL_NAMES = {"Bash", "shell", "exec_command", "functions.exec_command"}
DISABLE_KEYS = ("RTK_HOOK_DISABLE", "RTK_DISABLED")
MAX_COMMAND_BYTES = 32 * 1024
MACHINE_FLAGS = {
    "--json", "-json", "--porcelain", "-z", "-0", "--null", "--null-data",
    "--name-only", "--name-status", "--raw", "--numstat", "--shortstat",
    "--format", "--pretty", "--files", "--files-with-matches",
    "--files-without-match", "--count", "--count-matches", "--stats",
    "--only-matching", "--quiet", "--output", "--junitxml", "--junit-xml",
    "--json-report", "--report-log", "--log-file", "--xml", "--csv",
}
SCIENTIFIC_DIRS = {"outputs", "output", "research", "results", "artifacts", "logs", "data", "datasets", "checkpoints"}
SCIENTIFIC_SUFFIXES = {".json", ".jsonl", ".csv", ".tsv", ".npy", ".npz", ".parquet", ".safetensors", ".pt", ".log"}
EXPANSIONS = ("$", "`", "*", "?", "[", "{", "~")


def disabled() -> bool:
    return any(os.environ.get(key) == "1" for key in DISABLE_KEYS)


def split_command(command: str) -> list[str] | None:
    try:
        return shlex.split(command, posix=True)
    except ValueError:
        return None


def has_shell_operator(command: str) -> bool:
    """Detect unquoted shell operators without rebuilding shell syntax."""
    quote = None
    escaped = False
    for char in command:
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char in ";&|<>()":
            return True
    return False


def requests_exact_output(tokens: list[str]) -> bool:
    for token in tokens:
        flag = token.split("=", 1)[0]
        if flag in MACHINE_FLAGS:
            return True
        # pytest accepts option abbreviations; collection defines a denominator.
        if flag.startswith(("--co", "--junit", "--json", "--format", "--pretty")):
            return True
    if tokens[0] in {"rg", "grep"}:
        # Include bundled short machine-output switches (e.g. -lc, -n0).
        if any(t.startswith("-") and not t.startswith("--") and
               any(c in t[1:] for c in "lcLoqzZ0") for t in tokens[1:]):
            return True
    return False


def references_scientific_artifact(tokens: list[str]) -> bool:
    for token in tokens[1:]:
        path = Path(token.split("=", 1)[-1])
        if SCIENTIFIC_DIRS.intersection(path.parts) or path.suffix.lower() in SCIENTIFIC_SUFFIXES:
            return True
    return False


def rewrite_command(command: str) -> str | None:
    if disabled() or not command.strip() or len(command.encode()) > MAX_COMMAND_BYTES:
        return None
    # shlex is not a shell parser. Fail to raw rather than recreate assignments,
    # comments, continuations, substitutions, or control flow from flattened argv.
    if any(c in command for c in ("\n", "\r", "#", "\\")):
        return None
    tokens = split_command(command)
    if not tokens or any("=" in token for token in tokens[:1]):
        return None
    if requests_exact_output(tokens) or references_scientific_artifact(tokens):
        return None

    if has_shell_operator(command):
        # Only unquoted logical chains whose every segment is eligible. Pipes
        # and redirections are consumers of exact output, not human summaries.
        if any(c in command for c in "\"'$"):
            return None
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        parts = list(lexer)
        segments: list[str] = []
        segment: list[str] = []
        for part in [*parts, None]:
            if part is None or part in {"&&", "||", ";"}:
                if not segment:
                    return None
                rewritten = rewrite_command(shlex.join(segment))
                if rewritten is None:
                    return None
                segments.append(rewritten)
                segment = []
                if part is not None:
                    segments.append(part)
            elif any(c in part for c in ";&|<>()"):
                return None
            else:
                segment.append(part)
        return " ".join(segments)

    # Explicit shell $0/$1 argv must not be dropped. Only the exact three-argv
    # form is admitted; all environment/interpreter/runner prefixes stay raw.
    if tokens[0] in {"bash", "sh", "dash", "zsh"}:
        # Requoting can move parameter/command expansion from the outer shell
        # into the inner one (even inside quotes retained in the inner text).
        if "$" in command or "`" in command:
            return None
        if len(tokens) != 3 or tokens[1] not in {"-c", "-lc"}:
            return None
        inner = rewrite_command(tokens[2])
        return shlex.join([*tokens[:2], inner]) if inner else None

    candidate = tokens[0]
    if candidate == "git":
        if len(tokens) < 2 or tokens[1] not in {"status", "diff", "log"}:
            return None
    elif candidate not in {"rg", "grep", "pytest"}:
        return None

    if candidate == "pytest":
        # RTK 0.48.0 can erase pytest usage-error diagnostics. Admit only a
        # small known option set; plugin/config/report/unknown options stay raw.
        idx = 1
        while idx < len(tokens):
            token = tokens[idx]
            if token in {"-k", "-m"}:
                if idx + 1 >= len(tokens) or tokens[idx + 1].startswith("-"):
                    return None
                idx += 2
                continue
            if token.startswith("-") and not (
                token in {"-s", "-x", "-h", "-V", "--help", "--version"}
                or (token.startswith("-") and token[1:] and set(token[1:]) <= {"q", "v"})
            ):
                return None
            idx += 1

    has_expansion = any(any(marker in t for marker in EXPANSIONS) for t in tokens[1:])
    if has_expansion:
        # Only direct search can be inserted textually without changing the
        # caller's quoting/expansion. Never flatten pytest selectors/globs.
        if candidate not in {"rg", "grep"}:
            return None
        if rewrite_with_rtk(command) is None:
            return None
        return "rtk " + command.strip()

    rewritten = rewrite_with_rtk(command)
    if rewritten is None:
        return None
    rewritten_tokens = split_command(rewritten)
    # Reconstruct only a known direct insertion. pytest's upstream rewrite can
    # flatten -k 'a or b'; restore the original argv, never change interpreter.
    if rewritten_tokens == ["rtk", *tokens] or (
        candidate == "pytest" and rewritten_tokens and rewritten_tokens[:2] == ["rtk", "pytest"]
    ):
        return shlex.join(["rtk", *tokens])
    return None


def rewrite_with_rtk(command: str) -> str | None:
    executable = shutil.which("rtk")
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, "rewrite", command], check=False, shell=False,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0 or len(result.stdout.encode()) > MAX_COMMAND_BYTES * 2:
        return None
    return result.stdout.strip() or None


def main() -> int:
    if disabled():
        return 0
    raw = sys.stdin.read(MAX_COMMAND_BYTES * 4 + 1)
    if len(raw.encode()) > MAX_COMMAND_BYTES * 4:
        return 0
    try:
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            return 0
        tool_name = payload.get("tool_name")
        if tool_name and tool_name not in SUPPORTED_TOOL_NAMES:
            return 0
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return 0
        key = "command" if "command" in tool_input else "cmd"
        command = tool_input.get(key)
        if not isinstance(command, str):
            return 0
        rewritten = rewrite_command(command)
    except Exception:
        # A token-saving lookup must never block the underlying invocation.
        return 0
    if rewritten and rewritten != command:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "PreToolUse", "updatedInput": {key: rewritten},
        }}, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
