from __future__ import annotations

import pytest

from src.config.schema import TrainableTokenRowsConfig
from src.coord_tokens.codec import get_coord_token_ids
from src.tokens.roles import TokenRole, TokenRoleSets


class FakeTokenizer:
    def __init__(self) -> None:
        self.token_to_id = {
            "<|object_ref_start|>": 151646,
            "<|box_start|>": 151648,
        }
        self.token_to_id.update(
            {f"<|coord_{idx}|>": 151670 + idx for idx in range(1000)}
        )

    def convert_tokens_to_ids(self, token: str | list[str]) -> int | list[int]:
        if isinstance(token, list):
            return [self.token_to_id[item] for item in token]
        return self.token_to_id[token]


def _compact_trainable_rows_config() -> TrainableTokenRowsConfig:
    return TrainableTokenRowsConfig.from_mapping(
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
                "compact_structure": {
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


def test_trainable_token_rows_resolves_structural_markers_separately_from_coord_loss_ids():
    tokenizer = FakeTokenizer()
    role_sets = _compact_trainable_rows_config().resolve_role_sets(tokenizer)

    assert isinstance(role_sets, TokenRoleSets)
    assert role_sets.coord_geometry_ids == tuple(range(151670, 152670))
    assert role_sets.structural_ce_only_ids == (151646, 151648)
    assert len(role_sets.trainable_row_ids) == 1002
    assert set(role_sets.trainable_row_ids) == set(range(151670, 152670)) | {
        151646,
        151648,
    }
    assert role_sets.coord_loss_ids == tuple(range(151670, 152670))
    assert {151646, 151648}.isdisjoint(role_sets.coord_loss_ids)


def test_structural_ce_only_ids_are_not_coord_codec_ids():
    tokenizer = FakeTokenizer()

    coord_ids = set(get_coord_token_ids(tokenizer, validate=True))

    assert coord_ids == set(range(151670, 152670))
    assert {151646, 151648}.isdisjoint(coord_ids)


def test_trainable_token_rows_rejects_unknown_role():
    with pytest.raises(ValueError, match="custom.trainable_token_rows.groups.bad.role"):
        TrainableTokenRowsConfig.from_mapping(
            {
                "enabled": True,
                "groups": {
                    "bad": {
                        "role": "coord_or_structural_maybe",
                        "tokens": ["<|object_ref_start|>"],
                    }
                },
            }
        )


def test_trainable_token_rows_rejects_expected_id_mismatch():
    tokenizer = FakeTokenizer()
    cfg = TrainableTokenRowsConfig.from_mapping(
        {
            "enabled": True,
            "groups": {
                "compact_structure": {
                    "role": TokenRole.STRUCTURAL_CE_ONLY.value,
                    "tokens": ["<|object_ref_start|>"],
                    "expected_ids": {"<|object_ref_start|>": 42},
                }
            },
        }
    )

    with pytest.raises(ValueError, match="expected id 42"):
        cfg.resolve_role_sets(tokenizer)
