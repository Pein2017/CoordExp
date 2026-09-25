import copy
import json
import subprocess
from pathlib import Path

import pytest

from src.artifacts.git_identity import SourceIdentityError, capture_source_identity, verify_source_identity


@pytest.fixture
def clean_repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    git("init", "-q")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "user.name", "Fixture")
    (root / "source.py").write_text("VALUE = 1\n")
    (root / ".gitignore").write_text(".local/\n")
    git("add", "source.py", ".gitignore")
    git("commit", "-qm", "Fixture")
    return root, git


def test_exact_committed_source_round_trip(clean_repo):
    root, _ = clean_repo
    identity = capture_source_identity(["source.py"], root=root)
    verify_source_identity(json.loads(json.dumps(identity)), required_paths=["source.py"], root=root)
    assert len(identity["commit"]) == len(identity["tree"]) == 40
    assert identity["files"][0]["size_bytes"] == 10


@pytest.mark.parametrize("mutation", ["dirty", "untracked", "missing", "staged", "commit", "assume_unchanged"])
def test_changed_checkout_fails_before_continuation(clean_repo, mutation):
    root, git = clean_repo
    identity = capture_source_identity(["source.py"], root=root)
    if mutation == "missing":
        (root / "source.py").unlink()
    elif mutation == "untracked":
        (root / "other.py").write_text("pass\n")
    else:
        if mutation == "assume_unchanged":
            git("update-index", "--assume-unchanged", "source.py")
        (root / "source.py").write_text("VALUE = 2\n")
        if mutation in {"staged", "commit"}:
            git("add", "source.py")
        if mutation == "commit":
            git("commit", "-qm", "Changed")
    called = []
    with pytest.raises(SourceIdentityError, match="unsupported for continuation"):
        verify_source_identity(identity, required_paths=["source.py"], root=root)
        called.append("execute")
    assert called == []


@pytest.mark.parametrize("value", [{}, {"path": "source.py", "sha256": "0" * 64}, {"schema": "old.receipt.v1"}])
def test_legacy_receipts_never_get_a_fallback(clean_repo, value):
    root, _ = clean_repo
    with pytest.raises(SourceIdentityError):
        verify_source_identity(value, required_paths=["source.py"], root=root)


@pytest.mark.parametrize("path", ["../source.py", "/source.py", "./source.py", ".git/config", "source.py\n"])
def test_unsafe_source_paths_reject(clean_repo, path):
    root, _ = clean_repo
    with pytest.raises(SourceIdentityError):
        capture_source_identity([path], root=root)


def test_identity_cannot_omit_required_files_or_change_digests(clean_repo):
    root, _ = clean_repo
    identity = capture_source_identity(["source.py"], root=root)
    with pytest.raises(SourceIdentityError):
        verify_source_identity(identity, required_paths=["another.py"], root=root)
    changed = copy.deepcopy(identity)
    changed["files"][0]["sha256"] = "0" * 64
    with pytest.raises(SourceIdentityError):
        verify_source_identity(changed, required_paths=["source.py"], root=root)


def test_committed_symlink_is_not_source_identity(clean_repo):
    root, git = clean_repo
    (root / "alias.py").symlink_to("source.py")
    git("add", "alias.py")
    git("commit", "-qm", "Alias")
    with pytest.raises(SourceIdentityError):
        capture_source_identity(["alias.py"], root=root)


def test_unrelated_commit_also_invalidates_continuation(clean_repo):
    root, git = clean_repo
    identity = capture_source_identity(["source.py"], root=root)
    (root / "README.md").write_text("New contract\n")
    git("add", "README.md")
    git("commit", "-qm", "Contract change")
    with pytest.raises(SourceIdentityError):
        verify_source_identity(identity, required_paths=["source.py"], root=root)
