import torch
import torch.nn as nn
import pytest

from src.coord_tokens.offset_adapter import install_coord_offset_adapter
from src.coord_tokens.codec import get_coord_token_ids
from src.config.schema import CoordOffsetConfig, TrainableTokenRowsConfig


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
    adapter = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )

    # Set deterministic offsets (shared for embed + head)
    adapter.embed_offset.data[:] = torch.tensor([[1.0] * 6, [0.5] * 6])
    assert adapter.head_offset is None

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

    # Tie-head semantics: head uses the same offset table
    flat_hidden = hidden.view(-1, hidden.size(-1))
    extra_logits = flat_hidden @ adapter.embed_offset.T
    flat_logits = logits.view(-1, logits.size(-1))
    coord_ids = adapter.coord_ids.tolist()
    base_logits = (flat_hidden @ model.lm_head.weight.t()).detach()
    for idx, token_id in enumerate(coord_ids):
        assert torch.allclose(
            flat_logits[:, token_id], base_logits[:, token_id] + extra_logits[:, idx]
        )

    loss = logits.sum()
    loss.backward()

    # Only offsets get gradients; base weights stay frozen
    assert adapter.embed_offset.grad is not None
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_forward_backward_affects_only_coord_offsets_untied():
    model = TinyLM()
    adapter = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=False, dtype="float32"
    )

    # Set deterministic offsets (embed and head are trained separately)
    adapter.embed_offset.data[:] = torch.tensor([[1.0] * 6, [0.5] * 6])
    assert adapter.head_offset is not None
    adapter.head_offset.data[:] = torch.tensor([[0.1] * 6, [0.2] * 6])

    base_embed_2 = model.embed_tokens.weight[2].detach().clone()
    base_embed_1 = model.embed_tokens.weight[1].detach().clone()

    input_ids = torch.tensor([[2, 1, 5]])
    logits = model(input_ids)

    hidden = model.embed_tokens(input_ids)
    assert torch.allclose(
        hidden[0, 0], base_embed_2 + adapter.embed_offset[0], atol=1e-5
    )
    assert torch.allclose(hidden[0, 1], base_embed_1, atol=1e-5)

    flat_hidden = hidden.view(-1, hidden.size(-1))
    extra_logits = flat_hidden @ adapter.head_offset.T
    flat_logits = logits.view(-1, logits.size(-1))
    coord_ids = adapter.coord_ids.tolist()
    base_logits = (flat_hidden @ model.lm_head.weight.t()).detach()
    for idx, token_id in enumerate(coord_ids):
        assert torch.allclose(
            flat_logits[:, token_id], base_logits[:, token_id] + extra_logits[:, idx]
        )

    loss = logits.sum()
    loss.backward()

    assert adapter.embed_offset.grad is not None
    assert adapter.head_offset.grad is not None
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_repeated_forward_graphs_backprop_without_inplace_version_error():
    model = TinyLM(vocab_size=1010, hidden_size=8)
    adapter = install_coord_offset_adapter(
        model, coord_ids=list(range(10, 1010)), tie_head=True, dtype="float32"
    )

    loss = torch.zeros(())
    for seq_len in (128, 97, 53):
        input_ids = torch.randint(0, 1010, (1, seq_len), dtype=torch.long)
        logits = model(input_ids)
        loss = loss + logits[..., 10:20].log_softmax(dim=-1).sum()

    loss.backward()

    assert adapter.embed_offset.grad is not None
    assert torch.isfinite(adapter.embed_offset.grad).all()
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_coord_offset_config_parsing_on_off():
    cfg_default = CoordOffsetConfig.from_mapping(None)
    assert cfg_default.enabled is False
    assert cfg_default.tie_head is True
    assert cfg_default.ids == ()
    assert cfg_default.weight_decay == 0.0

    cfg = CoordOffsetConfig.from_mapping(
        {
            "enabled": True,
            "tie_head": False,
            "ids": {"start": 10, "end": 12},
            "embed_lr": 1e-4,
            "head_lr": 2e-4,
            "weight_decay": 0.1,
            "dtype": "bf16",
        }
    )
    assert cfg.enabled is True
    assert cfg.tie_head is False
    assert list(cfg.ids) == [10, 11, 12]
    assert cfg.embed_lr == 1e-4
    assert cfg.head_lr == 2e-4
    assert cfg.weight_decay == 0.1
    assert cfg.dtype == "bf16"


class _FakeTokenizer:
    def __init__(self, token_to_id: dict[str, int]) -> None:
        self.token_to_id = dict(token_to_id)

    def convert_tokens_to_ids(self, token: str | list[str]) -> int | list[int]:
        if isinstance(token, list):
            return [self.token_to_id[item] for item in token]
        return self.token_to_id[token]


def test_trainable_token_rows_resolves_coord_range_and_sparse_compact_markers():
    cfg = TrainableTokenRowsConfig.from_mapping(
        {
            "enabled": True,
            "tie_head": True,
            "groups": {
                "coord": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_2|>",
                    "expected_start": 151670,
                    "expected_end": 151672,
                },
                "compact_markers": {
                    "role": "structural_ce_only",
                    "tokens": ["<|object_ref_start|>", "<|box_start|>"],
                    "expected_ids": {
                        "<|object_ref_start|>": 151646,
                        "<|box_start|>": 151648,
                    },
                },
            },
        }
    )

    tokenizer = _FakeTokenizer(
        {
            "<|object_ref_start|>": 151646,
            "<|box_start|>": 151648,
            "<|coord_0|>": 151670,
            "<|coord_2|>": 151672,
        }
    )

    role_sets = cfg.resolve_role_sets(tokenizer)
    assert role_sets.coord_geometry_ids == (151670, 151671, 151672)
    assert role_sets.structural_ce_only_ids == (151646, 151648)
    assert role_sets.trainable_row_ids == (
        151670,
        151671,
        151672,
        151646,
        151648,
    )
    assert role_sets.coord_loss_ids == (151670, 151671, 151672)


def test_trainable_token_rows_rejects_expected_id_mismatch():
    cfg = TrainableTokenRowsConfig.from_mapping(
        {
            "enabled": True,
            "groups": {
                "compact_markers": {
                    "role": "structural_ce_only",
                    "tokens": ["<|object_ref_start|>"],
                    "expected_ids": {"<|object_ref_start|>": 151646},
                },
            },
        }
    )

    tokenizer = _FakeTokenizer({"<|object_ref_start|>": 42})

    with pytest.raises(ValueError, match="expected id 151646"):
        cfg.resolve_ids(tokenizer)


def test_compact_markers_are_trainable_offsets_but_not_coord_loss_ids():
    token_to_id = {
        "<|object_ref_start|>": 151646,
        "<|box_start|>": 151648,
    }
    token_to_id.update({f"<|coord_{idx}|>": 151670 + idx for idx in range(1000)})
    tokenizer = _FakeTokenizer(token_to_id)
    cfg = TrainableTokenRowsConfig.from_mapping(
        {
            "enabled": True,
            "groups": {
                "coord": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_999|>",
                    "expected_start": 151670,
                    "expected_end": 152669,
                },
                "compact_markers": {
                    "role": "structural_ce_only",
                    "tokens": ["<|object_ref_start|>", "<|box_start|>"],
                    "expected_ids": {
                        "<|object_ref_start|>": 151646,
                        "<|box_start|>": 151648,
                    },
                },
            },
        }
    )

    role_sets = cfg.resolve_role_sets(tokenizer)
    trainable_offset_ids = set(role_sets.trainable_row_ids)
    coord_loss_ids = set(get_coord_token_ids(tokenizer, validate=True))

    assert {151646, 151648}.issubset(trainable_offset_ids)
    assert {151646, 151648}.isdisjoint(coord_loss_ids)
    assert coord_loss_ids == set(range(151670, 152670))
    assert set(role_sets.coord_loss_ids) == coord_loss_ids
