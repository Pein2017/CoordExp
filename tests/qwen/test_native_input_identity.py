import hashlib
from types import SimpleNamespace
import torch
from src.qwen.input_identity import input_identity, tensor_hash


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
