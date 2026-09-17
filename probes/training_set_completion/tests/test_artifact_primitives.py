from __future__ import annotations

import hashlib

import pytest

from probes.training_set_completion import artifacts, source256_data, training


def test_shared_canonical_bytes_preserve_unicode_sorting_and_terminal_newline():
    value = {"z": "物体", "a": [0, 999]}
    expected = '{"a":[0,999],"z":"物体"}\n'.encode()
    for encoder in (artifacts.canonical, training.canonical, source256_data.canonical):
        assert encoder(value) == expected
    expected_hash = hashlib.sha256(expected).hexdigest()
    assert (
        artifacts.digest(value)
        == training.digest(value)
        == source256_data.digest(value)
        == expected_hash
    )
    assert (
        source256_data.digest(value, newline=False)
        == hashlib.sha256(expected[:-1]).hexdigest()
    )
    assert source256_data.digest(value, newline=False) != expected_hash


def test_binding_preserves_absolute_path_hash_and_size(tmp_path):
    path = tmp_path / "payload"
    path.write_bytes(b"literal\x00bytes")
    expected = {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "size_bytes": path.stat().st_size,
    }
    assert (
        artifacts.binding(path)
        == training.binding(path)
        == source256_data.binding(path)
        == expected
    )


def test_publication_policies_remain_distinct(tmp_path):
    exclusive = tmp_path / "exclusive.json"
    idempotent = tmp_path / "idempotent.json"
    value = {"owner": "物体"}
    training.publish(exclusive, value)
    source256_data.publish(idempotent, value)
    assert (
        exclusive.read_bytes() == idempotent.read_bytes() == artifacts.canonical(value)
    )
    with pytest.raises(ValueError, match="overwrite"):
        training.publish(exclusive, value)
    source256_data.publish(idempotent, value)
    with pytest.raises(ValueError, match="collision"):
        source256_data.publish(idempotent, {"owner": "changed"})
    assert idempotent.read_bytes() == artifacts.canonical(value)
