from __future__ import annotations

import dataclasses
import hashlib
import importlib.util
import json
import stat
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / ".codex" / "serena" / "serena_worktree_mcp.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("serena_worktree_mcp", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_resolve_worktree_uses_exact_git_root(tmp_path: Path) -> None:
    subprocess.run(["/usr/bin/git", "init", "-q", str(tmp_path)], check=True)
    nested = tmp_path / "src" / "package"
    nested.mkdir(parents=True)

    module = _load_module()

    assert module.resolve_worktree(nested) == tmp_path.resolve()


def test_slot_records_full_root_and_stable_key(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    runtime_base = tmp_path / "runtime"
    expected_key = hashlib.sha256(str(root.resolve()).encode()).hexdigest()[:24]

    module = _load_module()
    slot = module.slot_for(root, runtime_base)

    assert slot.root == root.resolve()
    assert slot.key == expected_key
    assert slot.path == runtime_base / expected_key


def test_process_identity_reads_exact_project_and_rejects_pid_reuse(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--project",
            str(root),
        ]
    )
    try:
        module = _load_module()
        identity = module.read_process_identity(process.pid)

        assert identity is not None
        assert identity.pid == process.pid
        assert identity.start_ticks > 0
        assert identity.executable == str(Path(sys.executable).resolve())
        assert identity.argv[-2:] == ("--project", str(root))
        assert identity.project_root == root.resolve()
        assert module.identity_matches(identity)
        assert not module.identity_matches(dataclasses.replace(identity, start_ticks=identity.start_ticks + 1))
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_process_identity_requires_exactly_one_project_argument(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            "--project",
            str(root),
            "--project",
            str(root),
        ]
    )
    try:
        module = _load_module()
        assert module.read_process_identity(process.pid) is None
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_atomic_json_metadata_is_complete_and_private(tmp_path: Path) -> None:
    module = _load_module()
    target = tmp_path / "slot" / "backend.json"

    module.write_json_atomic(target, {"root": "/repo", "pid": 123})

    assert json.loads(target.read_text()) == {"pid": 123, "root": "/repo"}
    assert stat.S_IMODE(target.stat().st_mode) == 0o600
    assert list(target.parent.glob("*.tmp")) == []
