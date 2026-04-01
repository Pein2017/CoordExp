from __future__ import annotations

import re

import torch

from src.coord_tokens.codec import int_to_token
from src.data_collators.enrichers import _build_proxy_weight_tensors_for_sample


_COORD_RE = re.compile(r"^<\|coord_(\d{1,4})\|>$")


class _FakeTokenizer:
    def __init__(self, pieces_by_id: dict[int, str]):
        self._pieces_by_id = dict(pieces_by_id)
        self._ids_by_piece = {piece: idx for idx, piece in self._pieces_by_id.items()}

    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = False,
    ) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return "".join(self._pieces_by_id[int(tok)] for tok in token_ids)

    def convert_tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            return self._convert_token_to_id(tokens)
        return [self._convert_token_to_id(token) for token in tokens]

    def _convert_token_to_id(self, token: str) -> int:
        match = _COORD_RE.match(str(token))
        if match:
            return 1000 + int(match.group(1))
        return int(self._ids_by_piece[token])


def _build_supervised_ids() -> list[int]:
    ids = {
        "{": 1,
        '"objects"': 2,
        ":": 3,
        "[": 4,
        "]": 5,
        "}": 6,
        ",": 7,
        '"desc"': 8,
        '"bbox_2d"': 9,
        '"': 10,
        "mug": 11,
        "tablecloth": 12,
    }
    pieces_by_id = {idx: piece for piece, idx in ids.items()}
    for value in range(8):
        pieces_by_id[1000 + value] = int_to_token(value)
    tokenizer = _FakeTokenizer(pieces_by_id)

    supervised_ids = [
        ids["{"],
        ids['"objects"'],
        ids[":"],
        ids["["],
        ids["{"],
        ids['"desc"'],
        ids[":"],
        ids['"'],
        ids["mug"],
        ids['"'],
        ids[","],
        ids['"bbox_2d"'],
        ids[":"],
        ids["["],
        1000,
        ids[","],
        1001,
        ids[","],
        1002,
        ids[","],
        1003,
        ids["]"],
        ids["}"],
        ids[","],
        ids["{"],
        ids['"desc"'],
        ids[":"],
        ids['"'],
        ids["tablecloth"],
        ids['"'],
        ids[","],
        ids['"bbox_2d"'],
        ids[":"],
        ids["["],
        1004,
        ids[","],
        1005,
        ids[","],
        1006,
        ids[","],
        1007,
        ids["]"],
        ids["}"],
        ids["]"],
        ids["}"],
    ]
    return tokenizer, supervised_ids


def test_build_proxy_weight_tensors_for_sample_uses_object_supervision_weights():
    tokenizer, supervised_ids = _build_supervised_ids()
    labels = torch.tensor([-100, -100] + supervised_ids, dtype=torch.long)
    payload = {
        "objects": [
            {
                "desc": "mug",
                "bbox_2d": [int_to_token(v) for v in range(4)],
            },
            {
                "desc": "tablecloth",
                "bbox_2d": [int_to_token(v) for v in range(4, 8)],
            },
        ]
    }
    metadata = {
        "coordexp_proxy_supervision": {
            "object_supervision": [
                {"desc_ce_weight": 1.0, "coord_weight": 1.0},
                {"desc_ce_weight": 0.25, "coord_weight": 0.0},
            ]
        }
    }

    weights = _build_proxy_weight_tensors_for_sample(
        tokenizer=tokenizer,
        payload=payload,
        labels=labels,
        metadata=metadata,
        namespace="coordexp_proxy_supervision",
    )

    desc_nonzero = torch.nonzero(weights.desc > 0, as_tuple=False).view(-1).tolist()
    coord_nonzero = torch.nonzero(weights.coord > 0, as_tuple=False).view(-1).tolist()
    assert desc_nonzero == [10, 30]
    assert coord_nonzero == [16, 18, 20, 22]
    assert float(weights.desc[10].item()) == 1.0
    assert float(weights.desc[30].item()) == 0.25
    assert all(float(weights.coord[idx].item()) == 1.0 for idx in [16, 18, 20, 22])
    assert all(float(weights.coord[idx].item()) == 0.0 for idx in [36, 38, 40, 42])


def test_build_proxy_weight_tensors_for_sample_defaults_to_real_weights_on_metadata_mismatch():
    tokenizer, supervised_ids = _build_supervised_ids()
    labels = torch.tensor(supervised_ids, dtype=torch.long)
    payload = {
        "objects": [
            {
                "desc": "mug",
                "bbox_2d": [int_to_token(v) for v in range(4)],
            },
            {
                "desc": "tablecloth",
                "bbox_2d": [int_to_token(v) for v in range(4, 8)],
            },
        ]
    }
    metadata = {
        "coordexp_proxy_supervision": {
            "object_supervision": [
                {"desc_ce_weight": 0.5, "coord_weight": 0.5},
            ]
        }
    }

    weights = _build_proxy_weight_tensors_for_sample(
        tokenizer=tokenizer,
        payload=payload,
        labels=labels,
        metadata=metadata,
        namespace="coordexp_proxy_supervision",
    )

    assert float(weights.desc[8].item()) == 1.0
    assert float(weights.desc[28].item()) == 1.0
    assert all(float(weights.coord[idx].item()) == 1.0 for idx in [14, 16, 18, 20, 34, 36, 38, 40])
