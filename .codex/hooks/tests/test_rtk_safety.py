"""Counterexamples use the nearest bad RTK rewrite, not RTK version luck."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).parents[1] / "rtk-pretooluse.py"
SPEC = importlib.util.spec_from_file_location("rtk_safety_owner", HOOK_PATH)
assert SPEC and SPEC.loader
hook = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(hook)

PYTHON = "/root/miniconda3/envs/ms/bin/python"


@pytest.fixture
def eager_rtk(monkeypatch):
    def rewrite(command):
        if "pytest" in command:
            return "rtk pytest -q"
        return "rtk " + command
    monkeypatch.setattr(hook, "rewrite_with_rtk", rewrite)


@pytest.mark.parametrize("command", [
    "bash -c 'pytest -q' sentinel extra",
    "FOO=$BAR git status",
    "FOO=literal git status",
    f"{PYTHON} -m pytest -q",
    "python -m pytest -q",
    "uv run python -m pytest -q",
    "conda run -n ms python -m pytest -q",
    "pytest --collect-only -q",
    "pytest --collect-only -k 'a or b'",
    "pytest --co -q",
    "pytest --not-a-real-option",
    "uv run pytest --collect-only",
    "git status && pytest --collect-only",
    "git status | wc -l",
    "git status > result.txt",
    "git ls-files",
    "rg -l needle src",
    "rg -c needle src",
    "rg accuracy outputs/metrics.jsonl",
    "cat outputs/metrics.jsonl",
    "make evaluate",
    "RTK_DISABLED=1 git status",
    "env RTK_DISABLED=1 git status",
])
def test_unsafe_commands_stay_raw(eager_rtk, command):
    assert hook.rewrite_command(command) is None


@pytest.mark.parametrize("command", [
    'bash -c "rg \'$PAT\' sample.txt"',
    'bash -c "rg \'`printf needle`\' sample.txt"',
])
def test_shell_wrapper_outer_expansion_is_not_moved(eager_rtk, tmp_path, command):
    (tmp_path / "sample.txt").write_text("needle\n$PAT\n")
    env = {**os.environ, "PAT": "needle"}
    def run(value):
        return subprocess.run(["/bin/bash", "-c", value], cwd=tmp_path, env=env,
                              text=True, capture_output=True, timeout=10)
    raw = run(command)
    assert (raw.returncode, raw.stdout, raw.stderr) == (0, "needle\n", "")
    rewritten = hook.rewrite_command(command)
    actual = run(rewritten or command)
    assert "needle" in actual.stdout, (rewritten, actual)
    assert "$PAT" not in actual.stdout, (rewritten, actual)
    assert rewritten is None


def test_rewrite_lookup_failure_stays_raw(monkeypatch):
    monkeypatch.setattr(hook, "rewrite_with_rtk", lambda command: None)
    assert hook.rewrite_command("pytest -q") is None


def test_hook_does_not_grant_permission(tmp_path):
    env = {**os.environ, "PATH": "/root/.local/bin:" + os.environ["PATH"]}
    result = subprocess.run([PYTHON, "-B", str(HOOK_PATH)], input=json.dumps({
        "tool_name": "Bash", "tool_input": {"command": "git status", "timeout": 17}
    }), text=True, capture_output=True, env=env, check=True)
    output = json.loads(result.stdout)["hookSpecificOutput"]
    assert "permissionDecision" not in output
    assert output["updatedInput"] == {"command": "rtk git status"}


@pytest.fixture
def actual_fixture(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "needle file.txt").write_text("needle two\nother\n")
    (tmp_path / "pytest.ini").write_text("[pytest]\n")
    (tmp_path / "test_probe.py").write_text(
        "def test_a(): assert True\n"
        "def test_b(): assert True\n"
        "def test_failure(): assert False\n"
    )
    (tmp_path / "conftest.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "def pytest_sessionstart(session):\n"
        "    Path('interpreter.txt').write_text(sys.executable)\n"
    )
    env = {**os.environ, "PATH": "/root/miniconda3/envs/ms/bin:/root/.local/bin:" + os.environ["PATH"],
           "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"}
    def run(command):
        return subprocess.run(["/bin/bash", "-c", command], cwd=tmp_path, env=env,
                              capture_output=True, text=True, timeout=15)
    return tmp_path, run


@pytest.mark.parametrize(("command", "exit_code"), [
    ("git status --short", 0),
    ("git diff --exit-code", 0),
    ("rg 'needle two' 'needle file.txt'", 0),
    ("rg absent 'needle file.txt'", 1),
    ("rg needle missing.txt", 2),
    ("pytest -q -k 'test_a or test_b'", 0),
    ("pytest -q -k failure", 1),
    ("pytest -q -k absent", 5),
    ("pytest --not-a-real-option", 4),
])
def test_actual_rtk_fixture_exit_codes_and_quoting(actual_fixture, command, exit_code):
    path, run = actual_fixture
    raw = run(command)
    rewritten = hook.rewrite_command(command)
    if exit_code == 4:
        assert rewritten is None
    else:
        assert rewritten is not None
    actual = run(rewritten or command)
    assert raw.returncode == actual.returncode == exit_code, (raw, actual, rewritten)
    print(json.dumps({
        "fixture_command": command,
        "raw_exit": raw.returncode, "actual_exit": actual.returncode,
        "rewritten": rewritten is not None,
        "raw_stdout_bytes": len(raw.stdout.encode()),
        "raw_stderr_bytes": len(raw.stderr.encode()),
        "actual_stdout_bytes": len(actual.stdout.encode()),
        "actual_stderr_bytes": len(actual.stderr.encode()),
    }, sort_keys=True))
    if command.startswith("pytest") and exit_code != 4:
        assert Path((path / "interpreter.txt").read_text()).resolve() == Path(PYTHON).resolve()
    if "needle two" in command:
        assert "needle two" in actual.stdout
        assert raw.stderr == actual.stderr == ""
    if command == "pytest -q -k 'test_a or test_b'":
        assert "2 passed" in raw.stdout and "2 passed" in actual.stdout
        assert raw.stderr == actual.stderr == ""
    if "missing.txt" in command or exit_code == 4:
        # RTK can move diagnostics across streams; don't promise stream identity.
        assert raw.stderr
        assert actual.stdout.strip() or actual.stderr.strip()
        if exit_code == 4:
            assert actual.stdout == raw.stdout
            assert actual.stderr == raw.stderr


@pytest.mark.parametrize("command", [
    f"{PYTHON} -m pytest -q -k 'test_a or test_b'",
    "pytest --collect-only -q -k 'test_a or test_b'",
    "FOO=$BAR git status --short",
    "bash -c 'printf \"%s|%s\\n\" \"$0\" \"$1\"; pytest -q -k absent' sentinel extra",
])
def test_actual_sensitive_commands_preserve_interpreter_argv_and_denominator(actual_fixture, command):
    path, run = actual_fixture
    assert hook.rewrite_command(command) is None
    result = run(command)
    assert result.stderr == ""
    if "--collect-only" in command:
        assert result.returncode == 0
        assert "test_probe.py::test_a" in result.stdout
        assert "test_probe.py::test_b" in result.stdout
        assert "2/3 tests collected" in result.stdout
    elif "sentinel extra" in command:
        assert result.returncode == 5
        assert "sentinel|extra" in result.stdout
    elif "-m pytest" in command:
        assert result.returncode == 0
        assert (path / "interpreter.txt").read_text() == PYTHON
    else:
        assert result.returncode == 0
