from types import SimpleNamespace

import pytest
import torch

from src.config.schema import StandardCECoordGaussianRPSAuxiliaryConfig
from src.data_collators.token_types import TokenType
from src.trainers.losses import coord_gaussian_rps as loss_mod
from src.trainers.losses.coord_gaussian_rps import (
    _compute_type_gate_contrib,
    build_coord_id_map,
    compute_coord_gaussian_rps_loss,
    resolve_standard_ce_type_gate_groups,
)


class TypeGateTokenizer:
    def __init__(self) -> None:
        vocab = {
            "<pad>": 0,
            "desc-a": 5,
            "desc-b": 6,
            "<|im_end|>": 2,
            "<|object_ref_start|>": 10,
            "<|box_start|>": 11,
            "\n": 12,
            "leak": 42,
        }
        for coord_index in range(1000):
            vocab[f"<|coord_{coord_index}|>"] = 100 + coord_index
        self._vocab = vocab
        self.pad_token_id = 0
        self.unk_token_id = 99999
        self.eos_token_id = 2

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    def convert_tokens_to_ids(self, tokens):
        def _one(token: str) -> int:
            return int(self._vocab.get(str(token), self.unk_token_id))

        if isinstance(tokens, str):
            return _one(tokens)
        return [_one(token) for token in tokens]


def _cfg(**updates):
    payload = {
        "enabled": True,
        "ce_weight": 1.0,
        "gaussian_weight": 1.0,
        "rps_weight": 1.0,
        "temperature": 1.0,
        "gaussian_r95_axis_fraction": 0.10,
        "gaussian_r95_cap_bins": 99,
        "gaussian_r95_min_bins": 1,
        "gaussian_r95_fallback_bins": 8,
        "type_gate": {
            "enabled": True,
            "mode": "allowed_type_mass",
            "weights": {
                "struct": 1.0,
                "coord": 1.0,
                "desc": 1.0,
                "eos": 0.5,
            },
        },
    }
    payload.update(updates)
    return StandardCECoordGaussianRPSAuxiliaryConfig.from_mapping(
        payload,
        path="objective.auxiliaries.coord_gaussian_rps",
    )


def _batch():
    # Shifted targets are:
    # desc-a, object_ref_start, x1, y1, x2, y2, im_end.
    labels = torch.tensor([[0, 5, 10, 110, 120, 130, 180, 2]], dtype=torch.long)
    masked_labels = labels.clone()
    masked_labels[0, 3:7] = -100
    token_types = torch.tensor(
        [
            [
                TokenType.IGNORE,
                TokenType.DESC,
                TokenType.FORMAT,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.COORD,
                TokenType.FORMAT,
            ]
        ],
        dtype=torch.long,
    )
    vocab_size = 1200
    logits = torch.full((1, labels.shape[1] - 1, vocab_size), -20.0)
    for position, token_id in enumerate(labels[0, 1:].tolist()):
        logits[0, position, int(token_id)] = 20.0
    coord_token_ids = list(range(100, 1100))
    coord_id_map = build_coord_id_map(
        vocab_size=vocab_size,
        device=logits.device,
        coord_token_ids=coord_token_ids,
    )
    return SimpleNamespace(
        labels=labels,
        masked_labels=masked_labels,
        token_types=token_types,
        logits=logits,
        coord_token_ids=coord_token_ids,
        coord_id_map=coord_id_map,
        tokenizer=TypeGateTokenizer(),
    )


def test_standard_ce_type_gate_resolves_coord_desc_schema_and_eos_groups() -> None:
    batch = _batch()
    labels_next = batch.labels[:, 1:]
    token_types_next = batch.token_types[:, 1:]

    group_names = resolve_standard_ce_type_gate_groups(
        labels_next=labels_next,
        token_types_next=token_types_next,
        tokenizer=batch.tokenizer,
        coord_id_map=batch.coord_id_map,
    )

    assert group_names == (
        "desc",
        "struct",
        "coord",
        "coord",
        "coord",
        "coord",
        "eos",
    )


def test_coord_gaussian_rps_loss_has_rps_not_w1_and_type_gate_penalty() -> None:
    batch = _batch()

    baseline = compute_coord_gaussian_rps_loss(
        logits=batch.logits,
        labels=batch.labels,
        masked_labels=batch.masked_labels,
        coord_token_weights=None,
        coord_token_ids=batch.coord_token_ids,
        coord_id_map=batch.coord_id_map,
        tokenizer=batch.tokenizer,
        token_types=batch.token_types,
        cfg=_cfg(),
        average_tokens_across_devices=False,
        model_accepts_loss_kwargs=False,
        accelerator_num_processes=None,
    )
    assert baseline is not None
    assert baseline.coord_tokens == 4
    assert baseline.rps_contrib.item() >= 0.0
    assert baseline.type_gate_contrib.item() >= 0.0
    assert not hasattr(baseline, "w1_contrib")

    with_non_coord_spike = batch.logits.clone()
    # Spike a desc token at coord positions. Coord-only CE/Gaussian/RPS are
    # unchanged; compact type-gate allowed-mass should catch the leakage.
    with_non_coord_spike[0, 2:6, 42] = 80.0
    changed = compute_coord_gaussian_rps_loss(
        logits=with_non_coord_spike,
        labels=batch.labels,
        masked_labels=batch.masked_labels,
        coord_token_weights=None,
        coord_token_ids=batch.coord_token_ids,
        coord_id_map=batch.coord_id_map,
        tokenizer=batch.tokenizer,
        token_types=batch.token_types,
        cfg=_cfg(),
        average_tokens_across_devices=False,
        model_accepts_loss_kwargs=False,
        accelerator_num_processes=None,
    )

    assert changed is not None
    assert changed.type_gate_contrib.item() > baseline.type_gate_contrib.item() + 1.0
    assert changed.coord_loss.item() > baseline.coord_loss.item() + 1.0


def test_type_gate_loss_bounds_full_vocab_log_softmax_rows(monkeypatch) -> None:
    batch = _batch()
    labels_next = batch.labels[:, 1:]
    token_types_next = batch.token_types[:, 1:]

    reference_loss, reference_count, reference_mass = _compute_type_gate_contrib(
        logits_next=batch.logits,
        labels_next=labels_next,
        token_types_next=token_types_next,
        tokenizer=batch.tokenizer,
        coord_id_map=batch.coord_id_map,
        cfg=_cfg().type_gate,
        average_tokens_across_devices=False,
        model_accepts_loss_kwargs=False,
        accelerator_num_processes=None,
    )

    calls: list[tuple[int, ...]] = []
    original_log_softmax = loss_mod.F.log_softmax

    def _recording_log_softmax(input, dim, *args, **kwargs):
        calls.append(tuple(input.shape))
        return original_log_softmax(input, dim, *args, **kwargs)

    monkeypatch.setattr(loss_mod, "_TYPE_GATE_CHUNK_TOKENS", 2, raising=False)
    monkeypatch.setattr(loss_mod.F, "log_softmax", _recording_log_softmax)

    chunked_loss, chunked_count, chunked_mass = _compute_type_gate_contrib(
        logits_next=batch.logits,
        labels_next=labels_next,
        token_types_next=token_types_next,
        tokenizer=batch.tokenizer,
        coord_id_map=batch.coord_id_map,
        cfg=_cfg().type_gate,
        average_tokens_across_devices=False,
        model_accepts_loss_kwargs=False,
        accelerator_num_processes=None,
    )

    assert chunked_count == reference_count
    assert chunked_loss == pytest.approx(reference_loss)
    assert chunked_mass is not None
    assert reference_mass is not None
    assert chunked_mass == pytest.approx(reference_mass)
    assert calls
    assert max(shape[0] for shape in calls) <= 2


def test_coord_gaussian_rps_rejects_legacy_w1_key() -> None:
    with pytest.raises(ValueError, match="w1_weight"):
        _cfg(w1_weight=1.0)
