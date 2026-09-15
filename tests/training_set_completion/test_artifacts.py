import hashlib
from pathlib import Path

from probes.training_set_completion import acquisition, artifacts, readback_selectors, review_packets


def test_canonical_bytes_digest_and_seed_match_frozen_prechange_values():
    value = {"ascii": "x", "non_ascii": "深圳", "nested": [1, True, None, 3.25]}
    expected = b'{"ascii":"x","nested":[1,true,null,3.25],"non_ascii":"\xe6\xb7\xb1\xe5\x9c\xb3"}\n'
    assert artifacts.canonical(value) == expected
    assert artifacts.digest(value) == "4e98318c51ae2ec239feb255ebb4a2c1499ac48bc4007377887c7da3296d43b4"
    assert acquisition._canonical(value) == expected
    assert readback_selectors.canonical(value) == expected
    assert review_packets._canonical(value) == expected
    assert [acquisition.sample_seed(acquisition.IMAGE_IDS[0], temperature) for temperature in acquisition.TEMPERATURES] == [1799917875, 408415027, 1097602001]


def test_file_hash_and_binding_are_byte_identity_only(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abc\x00def")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert artifacts.file_hash(path) == expected
    assert artifacts.binding(path) == {
        "path": str(path.resolve()),
        "sha256": expected,
        "size_bytes": path.stat().st_size,
    }

    changed = tmp_path / "changed.bin"
    changed.write_bytes(path.read_bytes() + b"!")
    assert artifacts.binding(changed)["sha256"] != artifacts.binding(path)["sha256"]
