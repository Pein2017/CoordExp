from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research import capture_natural_boundary_support_source_bindings as capture
from src.artifacts.json_values import canonical_json_bytes, json_sha256, load_canonical_json


EXPECTED_CONSUMER_SHA256 = "9ade7ad861e7c822458f65fbdb39ddda702648d62339533a3faf5e0dd37a52b3"
EXPECTED_MERGER_SHA256 = "9eb7534641be4c87768698719ed84c2f84119959d9fa324b0fc5a72f7cd5edf1"


def test_required_inventory_and_consumer_merger_digests() -> None:
    receipt = capture.capture_source_bindings()

    assert set(receipt["sources"]) == {
        "consumer",
        "merger",
        "direct_analyzers",
        "tests",
        "sealed_plan",
        "planner_receipt",
        "census",
    }
    assert set(receipt["sources"]["direct_analyzers"]) == {"materializer", "analyzer"}
    assert set(receipt["sources"]["tests"]) == {"consumer", "merger", "materializer", "analyzer"}
    assert receipt["sources"]["consumer"]["sha256"] == EXPECTED_CONSUMER_SHA256
    assert receipt["sources"]["merger"]["sha256"] == EXPECTED_MERGER_SHA256


def test_active_consumer_and_merger_are_tracked_in_target_snapshot() -> None:
    receipt = capture.capture_source_bindings()
    head_commit = receipt["git"]["head"]["commit"]
    for role in ("consumer", "merger"):
        provenance = receipt["sources"][role]["git"]
        assert provenance["state"] == "tracked_clean"
        assert provenance["tracked"] is True
        assert provenance["owner_commit"] == head_commit
        assert provenance["commit"] == head_commit
        assert provenance["proof"] == "git_ls_files_and_status"


def test_symlink_and_non_file_sources_are_rejected(tmp_path: Path) -> None:
    target = tmp_path / "source.py"
    target.write_text("x = 1\n", encoding="utf-8")
    symlink = tmp_path / "source-link.py"
    symlink.symlink_to(target)
    with pytest.raises(capture.SourceBindingError, match="symlink"):
        capture._read_regular_file(symlink, label="fixture")
    with pytest.raises(capture.SourceBindingError, match="regular file"):
        capture._read_regular_file(tmp_path, label="fixture")


def test_write_once_refuses_differing_bytes_and_accepts_identical_retry(tmp_path: Path) -> None:
    receipt = capture.capture_source_bindings()
    output = tmp_path / "source-bindings.json"

    first = capture.write_source_bindings(output, document=receipt)
    assert first["self_sha256"] == receipt["self_sha256"]
    assert capture.write_source_bindings(output, document=receipt) == receipt

    output.write_bytes(output.read_bytes() + b"\n")
    with pytest.raises(capture.SourceBindingError, match="differing"):
        capture.write_source_bindings(output, document=receipt)


def test_receipt_is_strict_canonical_and_hashes_recompute(tmp_path: Path) -> None:
    output = tmp_path / "source-bindings.json"
    receipt = capture.write_source_bindings(output)
    loaded = load_canonical_json(output)
    assert output.read_bytes() == canonical_json_bytes(loaded)
    assert capture.validate_source_bindings(loaded) == receipt
    body = dict(loaded)
    self_sha256 = body.pop("self_sha256")
    assert json_sha256(body) == self_sha256
    assert json_sha256(capture.stable_bindings(loaded)) == loaded["stable_bindings_sha256"]


def test_git_status_is_explicitly_volatile_and_excluded_from_stable_binding() -> None:
    receipt = capture.capture_source_bindings()
    assert receipt["git"]["status"]["volatile"] is True
    altered = dict(receipt)
    altered["git"] = dict(receipt["git"])
    altered["git"]["status"] = {"volatile": True, "format": "porcelain-v1", "entries": ["volatile"], "entry_count": 1}
    assert capture.stable_bindings(altered) == capture.stable_bindings(receipt)


def test_cli_regeneration_has_same_stable_bindings(tmp_path: Path) -> None:
    # The CLI itself is exercised by the repository acceptance command; this
    # check keeps the comparison contract explicit without invoking a child
    # process in every focused test.
    expected = capture.capture_source_bindings()
    output = tmp_path / "source-bindings.json"
    actual = capture.write_source_bindings(output)
    assert capture.stable_bindings(actual) == capture.stable_bindings(expected)
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "captured"
