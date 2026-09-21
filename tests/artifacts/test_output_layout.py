import pytest

from src.artifacts.output_layout import scan_output_root


def test_artifacts_are_allowed_but_code_prose_environments_and_bytecode_are_not(tmp_path):
    (tmp_path / "result.json").write_text("{}")
    assert scan_output_root(tmp_path)["passed"]
    for name in ["worker.py", "result.md", "launch.sh", "worker.pyc"]:
        (tmp_path / name).write_text("content")
    (tmp_path / ".venv").mkdir()
    result = scan_output_root(tmp_path)
    assert len(result["findings"]) == 5
    assert not result["passed"]


def test_links_are_reported_without_following_external_trees(tmp_path):
    root = tmp_path / "outputs"
    root.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "not_scanned.py").write_text("content")
    (root / "images").symlink_to(external, target_is_directory=True)
    assert scan_output_root(root)["passed"]
    (root / "missing").symlink_to(tmp_path / "gone")
    result = scan_output_root(root)
    assert result["symlinks_not_followed"] == 2
    assert result["findings"][0]["reason"] == "broken_symlink"


def test_missing_root_cannot_pass(tmp_path):
    with pytest.raises(FileNotFoundError):
        scan_output_root(tmp_path / "missing")
