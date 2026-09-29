from types import SimpleNamespace
from unittest.mock import patch

import pytest

from probes import online_row_credit as online
from src.qwen.native import NativeRequest
from src.qwen.vllm_rollout import VllmDoraRollout, _generate


def test_rollout_backend_requires_explicit_fresh_qualification():
    with patch.object(online.p, "load", return_value={}):
        online.rollout_binding("unused", "hf")
        with pytest.raises(ValueError, match="qualification"):
            online.rollout_binding("unused", "vllm")
    with patch.object(online.p, "load", return_value={"rollout_backend": "vllm"}):
        online.rollout_binding("unused", "vllm")
        with pytest.raises(ValueError, match="qualification"):
            online.rollout_binding("unused", "hf")


def test_rollout_never_accepts_changed_prompt_or_false_stop():
    from PIL import Image
    request = NativeRequest("image", "prompt", Image.new("RGB", (32, 32)),
                            expected_token_ids=(10, 11))
    completion = SimpleNamespace(token_ids=[5, 99], finish_reason="stop",
                                 logprobs=[{5: SimpleNamespace(logprob=-0.2)},
                                           {99: SimpleNamespace(logprob=-0.1)}])
    output = SimpleNamespace(prompt_token_ids=[10, 11], outputs=[completion])
    engine = SimpleNamespace(generate=lambda *a, **kw: [output])
    result = _generate(engine, [request], [8], 99, 0, True)[0]
    assert result.token_ids == (5, 99) and result.raw_logprobs == (-0.2, -0.1)
    output.prompt_token_ids = [10, 12]
    with pytest.raises(RuntimeError, match="prompt tokens"):
        _generate(engine, [request], [8], 99, 0, False)
    output.prompt_token_ids = [10, 11]
    completion.finish_reason = "length"
    with pytest.raises(RuntimeError, match="stop reason"):
        _generate(engine, [request], [8], 99, 0, False)


def test_refresh_590_tensors_uses_bytes_not_per_tensor_file_descriptors():
    import torch
    from safetensors.torch import load
    adapter = {str(i): torch.tensor([float(i)]) for i in range(588)}
    embeddings = {k: torch.ones(2, 3) for k in ('input_embed_delta', 'output_embed_delta')}
    runtime = object.__new__(VllmDoraRollout)
    calls = []
    runtime._call = lambda operation, payload: calls.append((operation, payload))
    with patch('peft.get_peft_model_state_dict', return_value=adapter):
        runtime.refresh(None, SimpleNamespace(delta_tensors=lambda: embeddings), identity='new')
    operation, (a, e, identity) = calls[0]
    assert operation == 'refresh' and isinstance(a, bytes) and isinstance(e, bytes)
    assert len(load(a)) == 588 and load(a)['587'].item() == 587
    assert set(load(e)) == set(embeddings) and identity == runtime.identity == 'new'
