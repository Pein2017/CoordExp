from __future__ import annotations

import importlib.util
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
