from __future__ import annotations

import types
from collections import OrderedDict

import pytest
import torch
import torch.nn as nn

from src.tokens.row_offsets import CoordOffsetAdapter, install_coord_offset_adapter
from src.trainers.rollout_runtime.vllm_sync_materialization import (
    materialize_state_dict_for_vllm_full_sync,
)


class TinyTiedModel(nn.Module):
    def __init__(self, *, tie_config: bool | None = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        if tie_config is not None:
            self.config = types.SimpleNamespace(tie_word_embeddings=bool(tie_config))

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


class TinyUntiedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.config = types.SimpleNamespace(tie_word_embeddings=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.embed_tokens(input_ids))


class TinyTiedModelWithoutTieEvidence(nn.Module):
    def __init__(self, *, tie_config: bool | None = None) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 4)
        self.lm_head = nn.Linear(4, 8, bias=False)
        self.lm_head.weight = self.embed_tokens.weight
        if tie_config is not None:
            self.config = types.SimpleNamespace(tie_word_embeddings=bool(tie_config))


class FakeModulesToSaveWrapper(nn.Module):
    def __init__(
        self,
        *,
        original_module: nn.Module,
        active_adapter: str,
        active_module: nn.Module,
    ) -> None:
        super().__init__()
        self.original_module = original_module
        self.active_adapters = [active_adapter]
        self.modules_to_save = nn.ModuleDict({active_adapter: active_module})


def _coord_ids(adapter: CoordOffsetAdapter) -> torch.Tensor:
    return adapter.coord_ids.detach().clone()


def _non_coord_ids(coord_ids: torch.Tensor, vocab_size: int) -> torch.Tensor:
    coord = {int(x) for x in coord_ids.tolist()}
    return torch.tensor(
        [idx for idx in range(vocab_size) if idx not in coord],
        dtype=torch.long,
    )


def _cloned_state_dict(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (key, tensor.detach().clone()) for key, tensor in model.state_dict().items()
    )


def _assert_mapping_tensors_unchanged(
    actual: OrderedDict[str, torch.Tensor],
    expected: OrderedDict[str, torch.Tensor],
) -> None:
    assert list(actual.keys()) == list(expected.keys())
    for key, expected_tensor in expected.items():
        assert torch.equal(actual[key], expected_tensor), key


def _assert_no_forbidden_keys(state_dict: dict[str, object]) -> None:
    forbidden_fragments = (
        "coord_offset_adapter",
        "modules_to_save",
        "original_module",
        "lora_",
    )
    assert not [
        key
        for key in state_dict
        if any(fragment in key for fragment in forbidden_fragments)
    ]


def _plain_tied_loaded_from(materialized: dict[str, torch.Tensor]) -> TinyTiedModel:
    plain = TinyTiedModel(tie_config=True)
    with torch.no_grad():
        plain.embed_tokens.weight.copy_(materialized["embed_tokens.weight"])
        plain.lm_head.weight = plain.embed_tokens.weight
        if "lm_head.weight" in materialized:
            plain.lm_head.weight.copy_(materialized["lm_head.weight"])
    return plain


def _plain_untied_loaded_from(materialized: dict[str, torch.Tensor]) -> TinyUntiedModel:
    plain = TinyUntiedModel()
    with torch.no_grad():
        plain.embed_tokens.weight.copy_(materialized["embed_tokens.weight"])
        plain.lm_head.weight.copy_(materialized["lm_head.weight"])
    return plain


def test_materialized_tied_coord_offset_matches_hook_logits_and_rows() -> None:
    torch.manual_seed(11)
    model = TinyTiedModel(tie_config=True)
    adapter = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    with torch.no_grad():
        adapter.embed_offset.copy_(
            torch.tensor(
                [
                    [0.25, -0.5, 0.75, 1.0],
                    [-1.25, 0.5, 0.125, -0.75],
                ],
                dtype=torch.float32,
            )
        )
    assert adapter.head_offset is None

    state_dict = _cloned_state_dict(model)
    before = _cloned_state_dict(model)
    base_embed = state_dict["embed_tokens.weight"].clone()
    base_head = state_dict["lm_head.weight"].clone()
    coord_ids = _coord_ids(adapter)
    non_coord_ids = _non_coord_ids(coord_ids, vocab_size=base_embed.size(0))

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    expected_embed = base_embed[coord_ids] + adapter.embed_offset.detach()
    expected_head = base_head[coord_ids] + adapter.embed_offset.detach()
    assert torch.allclose(materialized["embed_tokens.weight"][coord_ids], expected_embed)
    assert torch.allclose(materialized["lm_head.weight"][coord_ids], expected_head)
    assert torch.equal(
        materialized["embed_tokens.weight"][non_coord_ids], base_embed[non_coord_ids]
    )
    assert torch.equal(
        materialized["lm_head.weight"][non_coord_ids], base_head[non_coord_ids]
    )

    input_ids = torch.tensor([[2, 1, 5], [0, 5, 3]])
    plain = _plain_tied_loaded_from(materialized)
    assert torch.allclose(model(input_ids), plain(input_ids), atol=1e-5)
    _assert_mapping_tensors_unchanged(state_dict, before)
    assert materialized["embed_tokens.weight"] is not state_dict["embed_tokens.weight"]
    assert materialized["lm_head.weight"] is not state_dict["lm_head.weight"]
    _assert_no_forbidden_keys(materialized)


def test_materialized_untied_coord_offset_patches_distinct_head_rows() -> None:
    torch.manual_seed(13)
    model = TinyUntiedModel()
    adapter = install_coord_offset_adapter(
        model, coord_ids=[1, 6], tie_head=False, dtype="float32"
    )
    with torch.no_grad():
        adapter.embed_offset.copy_(
            torch.tensor(
                [
                    [0.5, -0.25, 1.0, -1.5],
                    [1.25, 0.75, -0.5, 0.25],
                ],
                dtype=torch.float32,
            )
        )
        assert adapter.head_offset is not None
        adapter.head_offset.copy_(
            torch.tensor(
                [
                    [-1.0, 0.5, 0.125, 0.75],
                    [0.375, -0.625, 1.5, -0.25],
                ],
                dtype=torch.float32,
            )
        )

    state_dict = _cloned_state_dict(model)
    before = _cloned_state_dict(model)
    base_embed = state_dict["embed_tokens.weight"].clone()
    base_head = state_dict["lm_head.weight"].clone()
    coord_ids = _coord_ids(adapter)
    non_coord_ids = _non_coord_ids(coord_ids, vocab_size=base_embed.size(0))

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    assert adapter.head_offset is not None
    assert torch.allclose(
        materialized["embed_tokens.weight"][coord_ids],
        base_embed[coord_ids] + adapter.embed_offset.detach(),
    )
    assert torch.allclose(
        materialized["lm_head.weight"][coord_ids],
        base_head[coord_ids] + adapter.head_offset.detach(),
    )
    assert not torch.allclose(
        materialized["lm_head.weight"][coord_ids],
        base_head[coord_ids] + adapter.embed_offset.detach(),
    )
    assert torch.equal(
        materialized["embed_tokens.weight"][non_coord_ids], base_embed[non_coord_ids]
    )
    assert torch.equal(
        materialized["lm_head.weight"][non_coord_ids], base_head[non_coord_ids]
    )

    input_ids = torch.tensor([[1, 0, 6], [3, 6, 2]])
    plain = _plain_untied_loaded_from(materialized)
    assert torch.allclose(model(input_ids), plain(input_ids), atol=1e-5)
    _assert_mapping_tensors_unchanged(state_dict, before)
    _assert_no_forbidden_keys(materialized)


def test_fake_modules_to_save_wrapper_active_adapter_is_discovered_beyond_stale_instance() -> None:
    model = TinyTiedModel(tie_config=True)
    stale = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    active = CoordOffsetAdapter(
        coord_ids=[2, 5],
        tie_head=True,
        embed_dim=4,
        head_dim=4,
        base_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        stale.embed_offset.fill_(0.0)
        active.embed_offset.copy_(
            torch.tensor(
                [
                    [2.0, 0.0, -1.0, 0.5],
                    [-0.5, 1.5, 0.25, -2.0],
                ],
                dtype=torch.float32,
            )
        )
    model.coord_offset_adapter = FakeModulesToSaveWrapper(
        original_module=stale,
        active_adapter="alt",
        active_module=active,
    )
    state_dict = OrderedDict(
        [
            ("embed_tokens.weight", model.embed_tokens.weight.detach().clone()),
            ("lm_head.weight", model.lm_head.weight.detach().clone()),
            (
                "coord_offset_adapter.modules_to_save.alt.coord_ids",
                active.coord_ids.detach().clone(),
            ),
            (
                "coord_offset_adapter.modules_to_save.alt.embed_offset",
                active.embed_offset.detach().clone(),
            ),
            (
                "coord_offset_adapter.original_module.embed_offset",
                stale.embed_offset.detach().clone(),
            ),
        ]
    )

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    coord_ids = _coord_ids(active)
    assert torch.allclose(
        materialized["embed_tokens.weight"][coord_ids],
        state_dict["embed_tokens.weight"][coord_ids] + active.embed_offset.detach(),
    )
    assert torch.allclose(
        materialized["lm_head.weight"][coord_ids],
        state_dict["lm_head.weight"][coord_ids] + active.embed_offset.detach(),
    )
    _assert_no_forbidden_keys(materialized)


@pytest.mark.parametrize("active_module", [None, nn.Linear(4, 4, bias=False)])
def test_modules_to_save_wrapper_with_unresolved_active_adapter_rejects_coord_keys(
    active_module: nn.Module | None,
) -> None:
    model = TinyTiedModel(tie_config=True)
    stale = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    wrapper = nn.Module()
    wrapper.original_module = stale
    wrapper.active_adapters = ["active"]
    wrapper.modules_to_save = nn.ModuleDict()
    if active_module is not None:
        wrapper.modules_to_save["active"] = active_module
    wrapper.modules_to_save["inactive"] = CoordOffsetAdapter(
        coord_ids=[2, 5],
        tie_head=True,
        embed_dim=4,
        head_dim=4,
        base_dtype=torch.float32,
        device=torch.device("cpu"),
    )
    model.coord_offset_adapter = wrapper
    state_dict = OrderedDict(
        [
            ("embed_tokens.weight", model.embed_tokens.weight.detach().clone()),
            ("lm_head.weight", model.lm_head.weight.detach().clone()),
            (
                "coord_offset_adapter.modules_to_save.active.embed_offset",
                torch.ones(2, 4),
            ),
        ]
    )

    with pytest.raises(ValueError, match="coord_offset_adapter|active adapter"):
        materialize_state_dict_for_vllm_full_sync(model, state_dict)


def test_peft_modules_to_save_wrapper_active_adapter_is_discovered() -> None:
    peft_other = pytest.importorskip("peft.utils.other")
    ModulesToSaveWrapper = peft_other.ModulesToSaveWrapper

    model = TinyTiedModel(tie_config=True)
    stale = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    wrapper = ModulesToSaveWrapper(stale, "default")
    active = wrapper.modules_to_save["default"]
    with torch.no_grad():
        wrapper.original_module.embed_offset.fill_(0.0)
        active.embed_offset.copy_(
            torch.tensor(
                [
                    [1.5, -0.25, 0.5, -1.0],
                    [-2.0, 0.25, 1.25, 0.75],
                ],
                dtype=torch.float32,
            )
        )
    model.coord_offset_adapter = wrapper
    state_dict = OrderedDict(
        [
            ("embed_tokens.weight", model.embed_tokens.weight.detach().clone()),
            ("lm_head.weight", model.lm_head.weight.detach().clone()),
            (
                "coord_offset_adapter.modules_to_save.default.coord_ids",
                active.coord_ids.detach().clone(),
            ),
            (
                "coord_offset_adapter.modules_to_save.default.embed_offset",
                active.embed_offset.detach().clone(),
            ),
            (
                "coord_offset_adapter.original_module.embed_offset",
                wrapper.original_module.embed_offset.detach().clone(),
            ),
        ]
    )

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    coord_ids = _coord_ids(active)
    assert torch.allclose(
        materialized["embed_tokens.weight"][coord_ids],
        state_dict["embed_tokens.weight"][coord_ids] + active.embed_offset.detach(),
    )
    assert torch.allclose(
        materialized["lm_head.weight"][coord_ids],
        state_dict["lm_head.weight"][coord_ids] + active.embed_offset.detach(),
    )
    _assert_no_forbidden_keys(materialized)


def test_coord_keys_without_discoverable_adapter_fail_fast() -> None:
    state_dict = OrderedDict(
        [
            ("embed_tokens.weight", torch.zeros(8, 4)),
            ("lm_head.weight", torch.zeros(8, 4)),
            ("modules_to_save.alt.coord_offset_adapter.embed_offset", torch.ones(2, 4)),
        ]
    )

    with pytest.raises(ValueError, match="coord_offset_adapter|active adapter"):
        materialize_state_dict_for_vllm_full_sync(TinyUntiedModel(), state_dict)


def test_no_adapter_noops_without_mutating_input_mapping() -> None:
    ordinary = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    state_dict = OrderedDict(
        [
            ("model.layers.0.weight", ordinary.clone()),
            ("base_model.model.layers.0.lora_A.default.weight", torch.ones(1, 4)),
            ("base_model.model.layers.0.lora_B.alt.weight", torch.ones(4, 1)),
            ("base_model.model.layers.0.lora_magnitude_vector.default", torch.ones(4)),
            ("base_model.model.layers.0.original_module.weight", torch.ones(3, 4)),
            ("base_model.model.modules_to_save.alt.weight", torch.ones(3, 4)),
        ]
    )
    before_keys = list(state_dict.keys())
    before_ordinary = state_dict["model.layers.0.weight"].clone()

    materialized = materialize_state_dict_for_vllm_full_sync(TinyUntiedModel(), state_dict)

    assert list(state_dict.keys()) == before_keys
    assert torch.equal(state_dict["model.layers.0.weight"], before_ordinary)
    assert materialized == {"model.layers.0.weight": state_dict["model.layers.0.weight"]}


def test_forbidden_key_matching_is_not_default_adapter_specific() -> None:
    model = TinyTiedModel(tie_config=True)
    adapter = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    state_dict = _cloned_state_dict(model)
    state_dict.update(
        {
            "base_model.model.modules_to_save.default.weight": torch.ones(1),
            "base_model.model.modules_to_save.alt.weight": torch.ones(1),
            "coord_offset_adapter.modules_to_save.default.embed_offset": (
                adapter.embed_offset.detach().clone()
            ),
            "base_model.model.layers.0.original_module.weight": torch.ones(1),
            "base_model.model.layers.0.lora_A.default.weight": torch.ones(1),
            "base_model.model.layers.0.lora_B.alt.weight": torch.ones(1),
            "base_model.model.layers.0.lora_embedding_A.default": torch.ones(1),
            "base_model.model.layers.0.lora_magnitude_vector.default": torch.ones(1),
        }
    )

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    _assert_no_forbidden_keys(materialized)


def test_active_coord_adapter_missing_embed_tokens_weight_fails_fast() -> None:
    model = TinyTiedModel(tie_config=True)
    install_coord_offset_adapter(model, coord_ids=[2, 5], tie_head=True, dtype="float32")
    state_dict = _cloned_state_dict(model)
    state_dict.pop("embed_tokens.weight")

    with pytest.raises(ValueError, match="embed_tokens.weight"):
        materialize_state_dict_for_vllm_full_sync(model, state_dict)


def test_untied_coord_adapter_missing_lm_head_weight_fails_fast() -> None:
    model = TinyUntiedModel()
    install_coord_offset_adapter(model, coord_ids=[2, 5], tie_head=False, dtype="float32")
    state_dict = _cloned_state_dict(model)
    state_dict.pop("lm_head.weight")

    with pytest.raises(ValueError, match="lm_head.weight|untied"):
        materialize_state_dict_for_vllm_full_sync(model, state_dict)


def test_tied_missing_lm_head_allowed_when_config_and_storage_confirm_tying() -> None:
    model = TinyTiedModel(tie_config=True)
    adapter = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    with torch.no_grad():
        adapter.embed_offset.fill_(0.25)
    state_dict = _cloned_state_dict(model)
    state_dict.pop("lm_head.weight")

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    coord_ids = _coord_ids(adapter)
    assert "lm_head.weight" not in materialized
    assert torch.allclose(
        materialized["embed_tokens.weight"][coord_ids],
        state_dict["embed_tokens.weight"][coord_ids] + adapter.embed_offset.detach(),
    )
    _assert_no_forbidden_keys(materialized)


def test_tied_missing_lm_head_allowed_when_storage_confirms_tying_without_config() -> None:
    model = TinyTiedModel(tie_config=None)
    install_coord_offset_adapter(model, coord_ids=[2, 5], tie_head=True, dtype="float32")
    state_dict = _cloned_state_dict(model)
    state_dict.pop("lm_head.weight")

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    assert "lm_head.weight" not in materialized
    _assert_no_forbidden_keys(materialized)


def test_tied_missing_lm_head_rejected_when_tying_is_unknown() -> None:
    model = TinyTiedModelWithoutTieEvidence()
    install_coord_offset_adapter(model, coord_ids=[2, 5], tie_head=True, dtype="float32")
    state_dict = _cloned_state_dict(model)
    state_dict.pop("lm_head.weight")

    with pytest.raises(ValueError, match="tie_word_embeddings|lm_head.weight"):
        materialize_state_dict_for_vllm_full_sync(model, state_dict)


def test_tied_missing_lm_head_allowed_when_config_says_untied_but_storage_confirms_tying() -> None:
    model = TinyTiedModel(tie_config=False)
    install_coord_offset_adapter(model, coord_ids=[2, 5], tie_head=True, dtype="float32")
    state_dict = _cloned_state_dict(model)
    state_dict.pop("lm_head.weight")

    materialized = materialize_state_dict_for_vllm_full_sync(model, state_dict)

    assert "lm_head.weight" not in materialized
    _assert_no_forbidden_keys(materialized)


def test_tied_missing_lm_head_rejected_when_config_says_untied_without_storage_evidence() -> None:
    model = TinyTiedModelWithoutTieEvidence(tie_config=False)
    install_coord_offset_adapter(model, coord_ids=[2, 5], tie_head=True, dtype="float32")
    state_dict = _cloned_state_dict(model)
    state_dict.pop("lm_head.weight")

    with pytest.raises(ValueError, match="tie_word_embeddings|lm_head.weight"):
        materialize_state_dict_for_vllm_full_sync(model, state_dict)


@pytest.mark.parametrize(
    ("case_name", "mutate_adapter", "match"),
    [
        (
            "non_1d_coord_ids",
            lambda adapter: setattr(
                adapter, "coord_ids", torch.tensor([[2, 5]], dtype=torch.long)
            ),
            "coord_ids.*1D",
        ),
        (
            "row_count_mismatch",
            lambda adapter: setattr(
                adapter,
                "embed_offset",
                nn.Parameter(torch.zeros(3, 4, dtype=torch.float32)),
            ),
            "row count|coord_ids",
        ),
        (
            "hidden_size_mismatch",
            lambda adapter: setattr(
                adapter,
                "embed_offset",
                nn.Parameter(torch.zeros(2, 5, dtype=torch.float32)),
            ),
            "hidden",
        ),
        (
            "negative_coord_id",
            lambda adapter: setattr(
                adapter, "coord_ids", torch.tensor([-1, 5], dtype=torch.long)
            ),
            "bounds|negative|coord_ids",
        ),
        (
            "coord_id_exceeds_vocab",
            lambda adapter: setattr(
                adapter, "coord_ids", torch.tensor([2, 8], dtype=torch.long)
            ),
            "bounds|vocab|coord_ids",
        ),
    ],
)
def test_shape_and_bounds_failures(
    case_name: str,
    mutate_adapter,
    match: str,
) -> None:
    model = TinyTiedModel(tie_config=True)
    adapter = install_coord_offset_adapter(
        model, coord_ids=[2, 5], tie_head=True, dtype="float32"
    )
    mutate_adapter(adapter)
    state_dict = _cloned_state_dict(model)

    with pytest.raises(ValueError, match=match):
        materialize_state_dict_for_vllm_full_sync(model, state_dict)
