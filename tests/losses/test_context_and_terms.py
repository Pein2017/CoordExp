from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F

from src.common.errors import LossContractError
from src.losses import (
    BaseTokenCE,
    LossContext,
    TokenTypeGateLoss,
    TokenVocabularyGroups,
    build_token_vocabulary_groups,
)
from src.packing.planner import PackedSegment
from src.qwen.tokens import QwenTokenIdentity
from src.supervision import TokenAtom, TokenSequence


def test_loss_context_selects_causal_logits_and_upcasts_to_fp32() -> None:
    sequence = _sequence(
        (
            _atom(target_position=1, token_id=7, token_type="desc_text"),
            _atom(target_position=2, token_id=3, token_type="coordinate"),
        )
    )
    logits = torch.arange(32, dtype=torch.bfloat16).reshape(1, 4, 8)
    context = LossContext(
        logits=logits,
        token_sequence=sequence,
        vocab_groups=_groups(),
    )

    selected_logits, target_ids, atoms = context.select_logits_fp32()

    assert context.logits_positions.tolist() == [0, 1]
    assert context.target_ids.tolist() == [7, 3]
    assert context.segment_indices.tolist() == [0, 0]
    assert context.token_types == ("desc_text", "coordinate")
    assert selected_logits.dtype == torch.float32
    assert selected_logits.shape == (2, 8)
    assert torch.equal(selected_logits[0], logits[0, 0].float())
    assert torch.equal(selected_logits[1], logits[0, 1].float())
    assert target_ids.tolist() == [7, 3]
    assert tuple(atom.causal_logits_position for atom in atoms) == (0, 1)


def test_loss_context_maps_compact_logits_position_ids() -> None:
    sequence = _sequence(
        (
            _atom(target_position=2, token_id=7, token_type="desc_text"),
            _atom(target_position=3, token_id=3, token_type="coordinate"),
        )
    )
    compact_logits = torch.arange(16, dtype=torch.bfloat16).reshape(1, 2, 8)
    context = LossContext(
        logits=compact_logits,
        token_sequence=sequence,
        vocab_groups=_groups(),
        logits_position_ids=(1, 2),
    )

    selected_logits, target_ids, atoms = context.select_logits_fp32()

    assert context.physical_logits_positions.tolist() == [1, 2]
    assert context.logits_positions.tolist() == [0, 1]
    assert selected_logits.dtype == torch.float32
    assert torch.equal(selected_logits[0], compact_logits[0, 0].float())
    assert torch.equal(selected_logits[1], compact_logits[0, 1].float())
    assert target_ids.tolist() == [7, 3]
    assert tuple(atom.causal_logits_position for atom in atoms) == (1, 2)


def test_base_token_ce_matches_full_vocab_cross_entropy_for_selected_atoms() -> None:
    sequence = _sequence(
        (
            _atom(target_position=1, token_id=7, token_type="desc_text"),
            _atom(target_position=2, token_id=3, token_type="coordinate"),
            _atom(target_position=3, token_id=5, token_type="eos"),
        )
    )
    logits = torch.tensor(
        [
            [
                [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 4.0],
                [0.0, 0.1, 0.2, 3.0, 0.4, 0.5, 0.6, 0.7],
                [0.0, 0.1, 0.2, 0.3, 0.4, 3.0, 0.6, 0.7],
                [9.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ]
        ],
        dtype=torch.bfloat16,
    )
    context = LossContext(logits=logits, token_sequence=sequence, vocab_groups=_groups())

    losses = BaseTokenCE().per_atom_loss(context)

    expected = F.cross_entropy(
        logits[0, (0, 1, 2)].float(),
        torch.tensor((7, 3, 5)),
        reduction="none",
    )
    assert losses.dtype == torch.float32
    assert torch.allclose(losses, expected)


def test_token_type_gate_loss_uses_exact_logsumexp_group_mass() -> None:
    sequence = _sequence(
        (
            _atom(target_position=1, token_id=3, token_type="coordinate"),
            _atom(target_position=2, token_id=1, token_type="schema"),
            _atom(target_position=3, token_id=5, token_type="eos"),
        )
    )
    logits = torch.tensor(
        [
            [
                [0.0, 0.5, 1.0, 3.0, 2.0, -1.0, -2.0, 0.25],
                [0.1, 4.0, 3.5, 0.0, 0.2, -1.0, -2.0, 0.5],
                [0.2, 0.1, 0.0, -1.0, -2.0, 4.0, 3.0, 0.5],
                [9.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            ]
        ],
        dtype=torch.float16,
    )
    context = LossContext(logits=logits, token_sequence=sequence, vocab_groups=_groups())

    losses = TokenTypeGateLoss().per_atom_loss(context)

    rows = logits[0, (0, 1, 2)].float()
    expected = torch.stack(
        (
            torch.logsumexp(rows[0], dim=0) - torch.logsumexp(rows[0, (3, 4)], dim=0),
            torch.logsumexp(rows[1], dim=0) - torch.logsumexp(rows[1, (1, 2)], dim=0),
            torch.logsumexp(rows[2], dim=0) - torch.logsumexp(rows[2, (5,)], dim=0),
        )
    )
    assert losses.dtype == torch.float32
    assert torch.allclose(losses, expected)


def test_token_type_gate_loss_matches_scalar_formula_for_mixed_groups() -> None:
    sequence = _sequence(
        (
            _atom(target_position=1, token_id=7, token_type="desc_text"),
            _atom(target_position=2, token_id=3, token_type="coordinate"),
            _atom(target_position=3, token_id=1, token_type="schema"),
        )
    )
    logits = torch.randn((1, 4, 8), dtype=torch.float32)
    context = LossContext(logits=logits, token_sequence=sequence, vocab_groups=_groups())

    losses = TokenTypeGateLoss().per_atom_loss(context)

    selected_logits, _target_ids, atoms = context.select_logits_fp32()
    scalar_expected = []
    for row, atom in zip(selected_logits, atoms, strict=True):
        group = torch.tensor(context.vocab_groups.allowed_ids(atom.token_type))
        scalar_expected.append(
            torch.logsumexp(row, dim=0)
            - torch.logsumexp(row.index_select(0, group), dim=0)
        )
    assert torch.allclose(losses, torch.stack(scalar_expected))


def test_loss_context_rejects_unknown_token_type() -> None:
    sequence = _sequence((_atom(target_position=1, token_id=7, token_type="mystery"),))
    logits = torch.zeros((1, 4, 8), dtype=torch.bfloat16)

    with pytest.raises(LossContractError) as exc_info:
        LossContext(logits=logits, token_sequence=sequence, vocab_groups=_groups())

    assert exc_info.value.code == "loss.token_type_unknown"
    assert exc_info.value.context["target_position"] == 1
    assert exc_info.value.context["example_id"] == "ex-0"
    assert exc_info.value.context["source"] == "unit"
    assert exc_info.value.context["logical_target_position"] == 1


def test_loss_context_rejects_target_id_outside_declared_group() -> None:
    sequence = _sequence((_atom(target_position=1, token_id=7, token_type="coordinate"),))
    logits = torch.zeros((1, 4, 8), dtype=torch.bfloat16)

    with pytest.raises(LossContractError) as exc_info:
        LossContext(logits=logits, token_sequence=sequence, vocab_groups=_groups())

    assert exc_info.value.code == "loss.token_id_outside_group"


def test_build_token_vocabulary_groups_excludes_qwen_specials_from_desc_text() -> None:
    identity = QwenTokenIdentity(
        required_tokens=(),
        wrapper_token_ids={
            "<|object_ref_start|>": 1,
            "<|object_ref_end|>": 2,
        },
        coordinate_token_ids=(3, 4),
        im_end_newline_text="<|im_end|>\n",
        im_end_token_ids=(5,),
        newline_token_ids=(6,),
        im_end_newline_token_ids=(5, 6),
        tokenizer_vocab_size=12,
    )
    tokenizer = FakeTokenizer(
        token_ids={
            "<|image_pad|>": 7,
            "<|coord_*|>": 8,
            "<tool_call>": 9,
        },
        all_special_ids=(0, 5, 6, 7),
    )

    groups = build_token_vocabulary_groups(identity, tokenizer=tokenizer)

    assert groups.schema == (1, 2)
    assert groups.coordinate == (3, 4)
    assert groups.eos == (5,)
    for token_id in (0, 1, 2, 3, 4, 5, 6, 7, 8, 9):
        assert token_id not in groups.desc_text
    assert 10 in groups.desc_text
    assert 11 in groups.desc_text


def test_build_token_vocabulary_groups_requires_tokenizer() -> None:
    identity = QwenTokenIdentity(
        required_tokens=(),
        wrapper_token_ids={"<|object_ref_start|>": 1},
        coordinate_token_ids=(3,),
        im_end_newline_text="<|im_end|>\n",
        im_end_token_ids=(5,),
        newline_token_ids=(6,),
        im_end_newline_token_ids=(5, 6),
        tokenizer_vocab_size=12,
    )

    with pytest.raises(LossContractError) as exc_info:
        build_token_vocabulary_groups(identity, tokenizer=None)

    assert exc_info.value.code == "loss.vocab_tokenizer_required"


def test_large_desc_text_group_membership_is_checked_without_linear_tuple_scan() -> None:
    groups = TokenVocabularyGroups(
        vocab_size=20_000,
        desc_text=tuple(range(10, 20_000)),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6, 7, 8, 9),
    )
    atom = _atom(target_position=1, token_id=19_999, token_type="desc_text")

    groups.validate_atom(atom)
    assert 19_999 in groups._membership["desc_text"]


def test_token_vocabulary_groups_reject_blocked_target_overlap() -> None:
    with pytest.raises(LossContractError) as exc_info:
        TokenVocabularyGroups(
            vocab_size=8,
            desc_text=(7,),
            schema=(1, 2),
            coordinate=(3, 4),
            eos=(5,),
            blocked=(3, 6),
        )

    assert exc_info.value.code == "loss.vocab_group_overlap"


def _groups() -> TokenVocabularyGroups:
    return TokenVocabularyGroups(
        vocab_size=8,
        desc_text=(7,),
        schema=(1, 2),
        coordinate=(3, 4),
        eos=(5,),
        blocked=(0, 6),
    )


def _sequence(atoms: tuple[TokenAtom, ...]) -> TokenSequence:
    return TokenSequence(
        pack_index=0,
        input_ids=(0, 7, 3, 5),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=4,
            ),
        ),
        atoms=atoms,
        spans=(),
    )


def _atom(*, target_position: int, token_id: int, token_type: str) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        target_position=target_position,
        token_id=token_id,
        token_type=token_type,
        text="x",
        logical_target_position=target_position,
        source="unit",
    )


@dataclass(frozen=True)
class FakeTokenizer:
    token_ids: dict[str, int]
    all_special_ids: tuple[int, ...]

    def convert_tokens_to_ids(self, token: str) -> int | None:
        return self.token_ids.get(token)
