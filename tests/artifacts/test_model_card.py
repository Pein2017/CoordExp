import hashlib
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from src.artifacts.model_card import package_model_card


def test_generated_card_preserves_exact_utf8_and_newlines(tmp_path):
    raw = "# 模型\r\n原始文本\r\n".encode()
    path = tmp_path / "README.md"
    path.write_bytes(raw)
    result = package_model_card(tmp_path)
    payload = json.loads(result.read_text())
    assert payload["content_utf8"].encode() == raw
    assert payload["sha256"] == hashlib.sha256(raw).hexdigest()
    assert not path.exists()
    assert package_model_card(tmp_path) is None


def test_conflict_retains_source_card(tmp_path):
    (tmp_path / "README.md").write_bytes(b"source")
    (tmp_path / "model_card.json").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        package_model_card(tmp_path)
    assert (tmp_path / "README.md").read_bytes() == b"source"
    assert (tmp_path / "model_card.json").read_bytes() == b"existing"


def test_iterative_probe_checkpoint_packages_card_before_identity(monkeypatch, tmp_path):
    import torch
    from probes import iterative_positive as probe

    card = "# 模型\r\n原始文本\r\n".encode()
    weight = torch.tensor([1.0])

    class Model:
        def save_pretrained(self, output, **kwargs):
            output.mkdir(parents=True)
            (output / "adapter_model.safetensors").write_bytes(b"weights")
            (output / "adapter_config.json").write_bytes(b"{}\n")
            (output / "README.md").write_bytes(card)

    peft = ModuleType("peft")
    peft.get_peft_model_state_dict = lambda model, adapter_name: {"weight": weight}
    safetensors = ModuleType("safetensors")
    safetensors.__path__ = []
    safetensors_torch = ModuleType("safetensors.torch")
    safetensors_torch.load_file = lambda path: {"weight": weight.clone()}
    untied = ModuleType("src.qwen.untied_embeddings")
    untied.save_special_token_embedding_deltas = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "peft", peft)
    monkeypatch.setitem(sys.modules, "safetensors", safetensors)
    monkeypatch.setitem(sys.modules, "safetensors.torch", safetensors_torch)
    monkeypatch.setitem(sys.modules, "src.qwen.untied_embeddings", untied)

    checkpoint = tmp_path / "checkpoint-1"
    probe.save_checkpoint(
        SimpleNamespace(model=Model(), base_model_path="base", base_config_sha256="base", tokenizer_sha256="tok"),
        object(),
        checkpoint,
    )

    payload = json.loads((checkpoint / "adapter/model_card.json").read_text())
    identity = json.loads((checkpoint / "identity.json").read_text())
    assert payload["content_utf8"].encode() == card
    assert payload["sha256"] == hashlib.sha256(card).hexdigest()
    assert "adapter/model_card.json" in identity
    assert "adapter/README.md" not in identity
    assert (checkpoint / "adapter/adapter_model.safetensors").read_bytes() == b"weights"
