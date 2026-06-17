import torch
import torch.nn as nn
import pytest

from src.tokens.row_offsets import (
    TokenEmbeddingsAdapter,
    install_token_embeddings_adapter,
)
from src.coord_tokens.codec import get_coord_token_ids
from src.config.schema import TokenEmbeddingsAdapterConfig


class TinyLM(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_size: int = 6) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_tokens(input_ids)
        return self.lm_head(hidden)


class CrossDeviceEmbedding(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_size: int = 6) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab_size, hidden_size, device=torch.device("cuda:1")),
            requires_grad=False,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        target_ids = input_ids.to(self.weight.device)
        flat = target_ids.reshape(-1)
        hidden = self.weight.index_select(0, flat)
        return hidden.reshape(*target_ids.shape, self.weight.size(-1))


class CrossDeviceHead(nn.Module):
    def __init__(self, vocab_size: int = 10, hidden_size: int = 6) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.randn(vocab_size, hidden_size, device=torch.device("cuda:1")),
            requires_grad=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        shape = (*hidden_states.shape[:-1], self.weight.size(0))
        return torch.zeros(shape, device=self.weight.device, dtype=hidden_states.dtype)


def test_forward_backward_affects_only_token_embeddings_adapter_offsets():
    model = TinyLM()
    adapter = install_token_embeddings_adapter(
        model, token_ids=[2, 5], tie_head=True, dtype="float32"
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
    token_ids = adapter.token_ids.tolist()
    base_logits = (flat_hidden @ model.lm_head.weight.t()).detach()
    for idx, token_id in enumerate(token_ids):
        assert torch.allclose(
            flat_logits[:, token_id], base_logits[:, token_id] + extra_logits[:, idx]
        )

    loss = logits.sum()
    loss.backward()

    # Only offsets get gradients; base weights stay frozen
    assert adapter.embed_offset.grad is not None
    assert model.embed_tokens.weight.grad is None
    assert model.lm_head.weight.grad is None


def test_token_embeddings_adapter_installs_under_canonical_module_name():
    model = TinyLM()

    adapter = install_token_embeddings_adapter(
        model, token_ids=[2, 5], tie_head=True, dtype="float32"
    )

    assert adapter.module_name == "token_embeddings_adapter"
    assert getattr(model, "token_embeddings_adapter") is adapter
    assert adapter.token_ids.tolist() == [2, 5]


def test_forward_backward_affects_only_token_embeddings_adapter_offsets_untied():
    model = TinyLM()
    adapter = install_token_embeddings_adapter(
        model, token_ids=[2, 5], tie_head=False, dtype="float32"
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
    token_ids = adapter.token_ids.tolist()
    base_logits = (flat_hidden @ model.lm_head.weight.t()).detach()
    for idx, token_id in enumerate(token_ids):
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
    adapter = install_token_embeddings_adapter(
        model, token_ids=list(range(10, 1010)), tie_head=True, dtype="float32"
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_embedding_hook_accepts_sharded_input_and_output_devices():
    embed = CrossDeviceEmbedding()
    head = nn.Linear(6, 10, bias=False).to(torch.device("cuda:1"))
    adapter = TokenEmbeddingsAdapter(
        token_ids=[2, 5],
        tie_head=True,
        embed_dim=6,
        head_dim=6,
        base_dtype=torch.float32,
        device=torch.device("cuda:1"),
    )
    adapter.attach(embed, head)
    adapter.embed_offset.data[:] = torch.tensor(
        [[1.0] * 6, [0.5] * 6],
        device=adapter.embed_offset.device,
    )

    base_embed_2 = embed.weight[2].detach().clone()
    input_ids = torch.tensor([[2, 1, 5]], device=torch.device("cuda:0"))

    hidden = embed(input_ids)

    assert hidden.device == torch.device("cuda:1")
    assert torch.allclose(
        hidden[0, 0], base_embed_2 + adapter.embed_offset[0], atol=1e-5
    )


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_head_hook_accepts_sharded_hidden_and_logits_devices():
    embed = CrossDeviceEmbedding()
    head = CrossDeviceHead()
    adapter = TokenEmbeddingsAdapter(
        token_ids=[2, 5],
        tie_head=True,
        embed_dim=6,
        head_dim=6,
        base_dtype=torch.float32,
        device=torch.device("cuda:1"),
    )
    adapter.attach(embed, head)
    adapter.embed_offset.data[:] = torch.tensor(
        [[1.0] * 6, [0.5] * 6],
        device=adapter.embed_offset.device,
    )
    hidden_states = torch.ones((1, 3, 6), device=torch.device("cuda:0"))

    logits = head(hidden_states)

    assert logits.device == torch.device("cuda:1")
    assert torch.all(logits[..., 2] > 0)
    assert torch.all(logits[..., 5] > 0)


def test_token_embeddings_adapter_config_parsing_on_off():
    cfg_default = TokenEmbeddingsAdapterConfig.from_mapping(None)
    assert cfg_default.enabled is False
    assert cfg_default.tie_head is True
    assert cfg_default.groups == {}
    assert cfg_default.weight_decay == 0.0

    cfg_disabled = TokenEmbeddingsAdapterConfig.from_mapping({"enabled": False})
    assert cfg_disabled.enabled is False

    cfg = TokenEmbeddingsAdapterConfig.from_mapping(
        {
            "enabled": True,
            "tie_head": False,
            "groups": {
                "coords": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_2|>",
                    "expected_start": 151670,
                    "expected_end": 151672,
                }
            },
            "embed_lr": 1e-4,
            "head_lr": 2e-4,
            "weight_decay": 0.1,
            "dtype": "bf16",
        }
    )
    assert cfg.enabled is True
    assert cfg.tie_head is False
    assert tuple(cfg.groups) == ("coords",)
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


def test_token_embeddings_adapter_resolves_coord_range_and_sparse_compact_markers():
    cfg = TokenEmbeddingsAdapterConfig.from_mapping(
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


def test_token_embeddings_adapter_rejects_expected_id_mismatch():
    cfg = TokenEmbeddingsAdapterConfig.from_mapping(
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
    cfg = TokenEmbeddingsAdapterConfig.from_mapping(
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


def test_compact_object_box_closed_adapter_resolves_1004_trainable_rows():
    token_to_id = {
        "<|object_ref_start|>": 151646,
        "<|object_ref_end|>": 151647,
        "<|box_start|>": 151648,
        "<|box_end|>": 151649,
    }
    token_to_id.update({f"<|coord_{idx}|>": 151670 + idx for idx in range(1000)})
    tokenizer = _FakeTokenizer(token_to_id)

    cfg = TokenEmbeddingsAdapterConfig.from_mapping(
        {
            "enabled": True,
            "tie_head": True,
            "groups": {
                "coord_geometry": {
                    "role": "coord_geometry",
                    "start_token": "<|coord_0|>",
                    "end_token": "<|coord_999|>",
                    "expected_start": 151670,
                    "expected_end": 152669,
                },
                "schema_tokens": {
                    "role": "structural_ce_only",
                    "tokens": [
                        "<|object_ref_start|>",
                        "<|object_ref_end|>",
                        "<|box_start|>",
                        "<|box_end|>",
                    ],
                    "expected_ids": {
                        "<|object_ref_start|>": 151646,
                        "<|object_ref_end|>": 151647,
                        "<|box_start|>": 151648,
                        "<|box_end|>": 151649,
                    },
                },
            },
            "embed_lr": 1.0e-4,
            "head_lr": 1.0e-4,
            "weight_decay": 0.0,
        }
    )

    role_sets = cfg.resolve_role_sets(tokenizer)
    assert len(role_sets.trainable_row_ids) == 1004
    assert len(role_sets.coord_loss_ids) == 1000
    assert set(role_sets.structural_ce_only_ids) == {151646, 151647, 151648, 151649}
    assert set(role_sets.structural_ce_only_ids).isdisjoint(role_sets.coord_loss_ids)
