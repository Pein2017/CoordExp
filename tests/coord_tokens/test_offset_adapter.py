import torch
import torch.nn as nn

from src.coord_tokens.offset_adapter import install_coord_offset_adapter
from src.config.schema import CoordOffsetConfig


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_size: int = 6) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        return self.lm_head(hidden)


def test_forward_backward_affects_only_coord_offsets():
    model = TinyLM()
    adapter = install_coord_offset_adapter(model, coord_ids=[2, 5], dtype="float32")

    # Set deterministic offsets
    adapter.embed_offset.data[:] = torch.tensor([[1.0] * 6, [0.5] * 6])
    adapter.head_offset.data[:] = torch.tensor([[0.1] * 6, [0.2] * 6])

    # Baseline copies for comparison
    base_embed_2 = model.embed_tokens.weight[2].detach().clone()
    base_embed_1 = model.embed_tokens.weight[1].detach().clone()

    input_ids = torch.tensor([[2, 1, 5]])
    logits = model(input_ids)

    # Embedding offsets applied only to coord ids
    hidden = model.embed_tokens(input_ids)
    assert torch.allclose(
        hidden[0, 0], base_embed_2 + adapter.embed_offset[0], atol=1e-5
    )
    assert torch.allclose(hidden[0, 1], base_embed_1, atol=1e-5)

    # Head offsets contribute only to coord token logits
    flat_hidden = hidden.view(-1, hidden.size(-1))
    extra_logits = flat_hidden @ adapter.head_offset.T
    flat_logits = logits.view(-1, logits.size(-1))
    coord_ids = adapter.coord_ids.tolist()
    # Check that coord logit equals base + extra
    base_logits = (flat_hidden @ model.lm_head.weight.t()).detach()
    for idx, token_id in enumerate(coord_ids):
        assert torch.allclose(
            flat_logits[:, token_id], base_logits[:, token_id] + extra_logits[:, idx]
        )

    loss = logits.sum()
    loss.backward()

    # Only offsets get gradients; base weights stay frozen
    assert adapter.embed_offset.grad is not None
    assert adapter.head_offset.grad is not None
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_coord_offset_config_parsing_on_off():
    cfg_default = CoordOffsetConfig.from_mapping(None)
    assert cfg_default.enabled is False
    assert cfg_default.ids == ()
    assert cfg_default.weight_decay == 0.0

    cfg = CoordOffsetConfig.from_mapping(
        {
            "enabled": True,
            "ids": {"start": 10, "end": 12},
            "embed_lr": 1e-4,
            "head_lr": 2e-4,
            "weight_decay": 0.1,
            "dtype": "bf16",
        }
    )
    assert cfg.enabled is True
    assert list(cfg.ids) == [10, 11, 12]
    assert cfg.embed_lr == 1e-4
    assert cfg.head_lr == 2e-4
    assert cfg.weight_decay == 0.1
    assert cfg.dtype == "bf16"

