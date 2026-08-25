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
            "conda run -n ms rtk test python -m pytest -q",
        ),
        (
            "pytest tests/test_service.py -q",
            "rtk test pytest tests/test_service.py -q",
        ),
        (
            "python -m pytest --collect-only -q",
            "rtk test python -m pytest --collect-only -q",
        ),
        (
            "uv run pytest tests/test_service.py -q",
            "uv run rtk test pytest tests/test_service.py -q",
        ),
        (
            "uv run python -m pytest tests/test_service.py -q",
            "uv run rtk test python -m pytest tests/test_service.py -q",
        ),
    ],
)
def test_pytest_rewrite_uses_generic_test_filter_without_changing_argv(
    command: str, expected: str
) -> None:
    """Returning to `rtk pytest` would restore its false no-tests summary."""

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
            "rtk test python -m pytest tests/test_*.py -q",
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
