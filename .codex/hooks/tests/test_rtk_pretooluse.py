from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest


HOOK_PATH = Path(__file__).parents[1] / "rtk-pretooluse.py"
SPEC = importlib.util.spec_from_file_location("rtk_pretooluse", HOOK_PATH)
assert SPEC is not None and SPEC.loader is not None
rtk_pretooluse = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rtk_pretooluse)


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        (
            "conda run -n ms python -m pytest -q",
            "conda run -n ms rtk pytest -q",
        ),
        (
            "pytest tests/test_service.py -q",
            "rtk pytest tests/test_service.py -q",
        ),
        (
            "python -m pytest --collect-only -q",
            "rtk test python -m pytest --collect-only -q",
        ),
        (
            "uv run pytest tests/test_service.py -q",
            "uv run rtk pytest tests/test_service.py -q",
        ),
        (
            "uv run python -m pytest tests/test_service.py -q",
            "uv run rtk pytest tests/test_service.py -q",
        ),
    ],
)
def test_pytest_rewrite_preserves_argv_without_stale_workaround(
    command: str, expected: str
) -> None:
    """Only collection keeps the generic filter with RTK 0.45."""

    assert rtk_pretooluse.rewrite_command(command) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("git status --short", "rtk git status --short"),
        ("rg TODO src", "rtk rg TODO src"),
        ("RTK_HOOK_DISABLE=1 python -m pytest -q", None),
    ],
)
def test_pytest_workaround_does_not_change_other_rewrite_rules(
    command: str, expected: str | None
) -> None:
    assert rtk_pretooluse.rewrite_command(command) == expected


@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("rg -n TODO src/*.py", "rtk rg -n TODO src/*.py"),
        (
            "python -m pytest tests/test_*.py -q",
            None,
        ),
        (
            "conda run -n ms python -m pytest tests/test_*.py -q",
            None,
        ),
    ],
)
def test_shell_expansions_are_never_requoted_by_the_hook(
    command: str, expected: str | None
) -> None:
    assert rtk_pretooluse.rewrite_command(command) == expected


@pytest.mark.parametrize(
    "command",
    [
        "find src -type f -print0",
        "rg --null TODO src",
        "git diff --name-only",
        "git log --format=%H -1",
    ],
)
def test_exact_or_machine_output_is_left_raw(command: str) -> None:
    assert rtk_pretooluse.rewrite_command(command) is None


def test_pytest_collection_keeps_generic_filter() -> None:
    assert rtk_pretooluse.rewrite_command(
        "python -m pytest --collect-only -q"
    ) == "rtk test python -m pytest --collect-only -q"


def test_pytest_version_uses_upstream_filter() -> None:
    assert rtk_pretooluse.rewrite_command("pytest --version") == "rtk pytest --version"


@pytest.mark.parametrize(
    "suffix",
    [
        'printf "%s\\n" "$RTK_PROBE_VALUE"',
        'printf "%s\\n" *.py',
        'printf "%s\\n" ~/probe',
        'printf "%s\\n" {first,second}',
        'printf "%s\\n" "$(printf substituted)"',
        'for f in *.py; do printf "%s\\n" "$f"; done',
    ],
)
def test_compound_rewrite_preserves_executed_shell_behavior(tmp_path: Path, suffix: str) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "one.py").write_text("")
    (tmp_path / "two.py").write_text("")
    marker = "---PROBE---\n"
    command = f"git status --short && printf '\\n---PROBE---\\n' && {suffix}"
    env = {**os.environ, "RTK_PROBE_VALUE": "expanded-value"}

    def run(value: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["bash", "-c", value], cwd=tmp_path, env=env, text=True, capture_output=True
        )

    raw = run(command)
    rewritten = rtk_pretooluse.rewrite_command(command)
    actual = run(rewritten or command)
    # The Git status formatter may change presentation; the following shell
    # command must still execute identically.
    assert (actual.returncode, actual.stdout.split(marker, 1)[-1], actual.stderr) == (
        raw.returncode, raw.stdout.split(marker, 1)[-1], raw.stderr
    )


def test_plain_logical_chain_still_uses_rtk() -> None:
    assert rtk_pretooluse.rewrite_command("git status --short && git diff --stat") == (
        "rtk git status --short && rtk git diff --stat"
    )


@pytest.mark.parametrize("command", ["rg --json needle *.py", "rg --null needle *.py"])
def test_machine_output_with_glob_remains_raw(command: str) -> None:
    assert rtk_pretooluse.rewrite_command(command) is None
