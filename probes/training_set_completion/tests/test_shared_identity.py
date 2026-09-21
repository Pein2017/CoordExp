import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from probes.training_set_completion import artifacts
from src.qwen.input_identity import input_identity, tensor_hash


def test_literal_and_resolved_paths_remain_distinct(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    path = Path("sample.json")
    path.write_bytes(b"example\n")
    observed = artifacts.literal_binding(path)
    assert observed == {
        "path": "sample.json", "size_bytes": 8,
        "sha256": hashlib.sha256(b"example\n").hexdigest(),
    }
    assert artifacts.binding(path)["path"] == str(path.resolve())


def test_historical_json_bytes_are_not_canonicalized(tmp_path):
    value = {"label": "坐标", "nan": float("nan")}
    path = tmp_path / "value.json"
    artifacts.write_pretty_json(path, value)
    assert path.read_text() == json.dumps(value, indent=2) + "\n"
    expected = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    assert artifacts.ascii_json_digest(value) == hashlib.sha256(expected).hexdigest()
    assert artifacts.digest(value) != artifacts.ascii_json_digest(value)


def test_native_identity_preserves_literal_values_and_noncontiguous_tensor_bytes():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
    expected = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
    assert tensor_hash(value) == expected
    batch = SimpleNamespace(
        inputs={"pixels": value, "not_a_tensor": "ignored"},
        request_ids=("one",), prompt_token_ids=((1, 2),),
        media_sha256=None, image_grids=(None, (1, 2, 3)),
    )
    assert input_identity(batch) == {
        "request_ids": ["one"], "prompt_token_ids": [[1, 2]],
        "media_sha256": None, "image_grids": [None, [1, 2, 3]],
        "tensor_sha256": {"pixels": expected},
    }
