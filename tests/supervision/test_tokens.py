from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.common.errors import LossContractError
from src.packing.planner import PackedSegment, PackedSequence, plan_packed_sequences
from src.packing.supervision import build_packed_supervision
from src.supervision import (
    TokenAtom,
    TokenSequence,
    TokenSpan,
    build_token_sequence_from_packed_supervision,
    dense_labels_from_token_sequence,
    index_token_atoms_by_pack,
    validate_dense_labels_match_token_sequence,
)


def test_token_sequence_builds_canonical_atoms_spans_and_dense_labels() -> None:
    examples = (
        FakeEncodedExample(
            "ex-0",
            (10, 11, 12, 13),
            (
                FakeTokenSpan(
                    "schema",
                    physical=1,
                    token_ids=(11, 12),
                    text="<box>",
                    source="template",
                ),
                FakeTokenSpan("eos", physical=3, token_ids=(13,), text="<|im_end|>"),
            ),
        ),
        FakeEncodedExample(
            "ex-1",
            (20, 21, 22),
            (
                FakeTokenSpan(
                    "coordinate",
                    physical=1,
                    token_ids=(21,),
                    text="<|coord_10|>",
                    object_id="obj-1",
                    field="x1",
                ),
            ),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=7)
    packed_supervision = build_packed_supervision(packs, examples)

    sequence = build_token_sequence_from_packed_supervision(packs[0], packed_supervision)

    assert sequence.pack_index == 0
    assert sequence.pack_length == 7
    assert [atom.target_position for atom in sequence.atoms] == [1, 2, 3, 5]
    assert [atom.causal_logits_position for atom in sequence.atoms] == [0, 1, 2, 4]
    assert [atom.token_id for atom in sequence.atoms] == [11, 12, 13, 21]
    assert [span.token_type for span in sequence.spans] == ["schema", "eos", "coordinate"]
    assert [span.target_range for span in sequence.spans] == [(1, 3), (3, 4), (5, 6)]
    assert sequence.spans[0].atoms == sequence.atoms[:2]
    assert dense_labels_from_token_sequence(sequence) == (-100, 11, 12, 13, -100, 21, -100)
    assert validate_dense_labels_match_token_sequence(
        sequence,
        (-100, 11, 12, 13, -100, 21, -100),
    ) is None


def test_token_atom_index_builds_equivalent_pack_sequences_without_full_scan() -> None:
    examples = (
        FakeEncodedExample(
            "ex-0",
            (10, 11, 12),
            (FakeTokenSpan("schema", physical=1, token_ids=(11,), text="<box>"),),
        ),
        FakeEncodedExample(
            "ex-1",
            (20, 21, 22),
            (FakeTokenSpan("coordinate", physical=1, token_ids=(21,), text="<|coord_10|>"),),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=3)
    packed_supervision = build_packed_supervision(packs, examples)
    atoms_by_pack = index_token_atoms_by_pack(packed_supervision)

    full_scan_sequence = build_token_sequence_from_packed_supervision(
        packs[1],
        packed_supervision,
    )
    indexed_sequence = build_token_sequence_from_packed_supervision(
        packs[1],
        atoms_by_pack[packs[1].pack_index],
    )

    assert tuple(atom.to_artifact_dict() for atom in indexed_sequence.atoms) == tuple(
        atom.to_artifact_dict() for atom in full_scan_sequence.atoms
    )
    assert indexed_sequence.to_dense_labels() == full_scan_sequence.to_dense_labels()


def test_token_sequence_rejects_target_position_zero_for_causal_loss() -> None:
    pack = PackedSequence(
        pack_index=0,
        input_ids=(10, 11),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=2,
            ),
        ),
        global_max_length=8,
    )
    atom = _atom(target_position=0, segment_index=0, example_id="ex-0", token_id=10)

    with pytest.raises(LossContractError) as exc_info:
        build_token_sequence_from_packed_supervision(pack, (atom,))

    assert exc_info.value.code == "supervision.causal_target_position"


def test_token_sequence_rejects_logits_position_crossing_segment_boundary() -> None:
    pack = PackedSequence(
        pack_index=0,
        input_ids=(10, 11, 20, 21),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=2,
            ),
            PackedSegment(
                pack_index=0,
                segment_index=1,
                example_index=1,
                example_id="ex-1",
                start=2,
                end=4,
            ),
        ),
        global_max_length=8,
    )
    atom = _atom(target_position=2, segment_index=1, example_id="ex-1", token_id=20)

    with pytest.raises(LossContractError) as exc_info:
        build_token_sequence_from_packed_supervision(pack, (atom,))

    assert exc_info.value.code == "supervision.logits_cross_segment_boundary"


def test_dense_label_parity_rejects_independent_label_semantics() -> None:
    examples = (
        FakeEncodedExample(
            "ex-0",
            (10, 11, 12),
            (FakeTokenSpan("schema", physical=1, token_ids=(11, 12)),),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=8)
    packed_supervision = build_packed_supervision(packs, examples)
    sequence = build_token_sequence_from_packed_supervision(packs[0], packed_supervision)

    with pytest.raises(LossContractError) as exc_info:
        validate_dense_labels_match_token_sequence(sequence, (-100, 11, 999))

    assert exc_info.value.code == "supervision.dense_label_mismatch"


def test_token_sequence_rejects_duplicate_target_positions() -> None:
    pack = PackedSequence(
        pack_index=0,
        input_ids=(10, 11, 12),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=3,
            ),
        ),
        global_max_length=8,
    )
    atoms = (
        _atom(target_position=1, segment_index=0, example_id="ex-0", token_id=11),
        _atom(target_position=1, segment_index=0, example_id="ex-0", token_id=12),
    )

    with pytest.raises(LossContractError) as exc_info:
        build_token_sequence_from_packed_supervision(pack, atoms)

    assert exc_info.value.code == "supervision.duplicate_target_position"


def test_token_sequence_rejects_non_sequential_segment_indices() -> None:
    pack = PackedSequence(
        pack_index=0,
        input_ids=(10, 11, 20, 21),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=2,
            ),
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=1,
                example_id="ex-1",
                start=2,
                end=4,
            ),
        ),
        global_max_length=8,
    )

    with pytest.raises(LossContractError) as exc_info:
        build_token_sequence_from_packed_supervision(pack, ())

    assert exc_info.value.code == "supervision.segment_index_sequence"


def test_direct_atom_sequence_rejects_wrong_pack_atoms() -> None:
    pack = PackedSequence(
        pack_index=0,
        input_ids=(10, 11),
        segments=(
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=2,
            ),
        ),
        global_max_length=8,
    )
    wrong_pack_atom = TokenAtom(
        pack_index=1,
        segment_index=0,
        example_index=0,
        example_id="ex-0",
        target_position=1,
        token_id=11,
        token_type="schema",
        text="x",
        logical_target_position=1,
    )

    with pytest.raises(LossContractError) as exc_info:
        build_token_sequence_from_packed_supervision(pack, (wrong_pack_atom,))

    assert exc_info.value.code == "supervision.atom_pack_index"


def test_packed_supervision_can_filter_atoms_for_requested_pack() -> None:
    examples = (
        FakeEncodedExample(
            "ex-0",
            (10, 11),
            (FakeTokenSpan("schema", physical=1, token_ids=(11,)),),
        ),
        FakeEncodedExample(
            "ex-1",
            (20, 21),
            (FakeTokenSpan("coordinate", physical=1, token_ids=(21,)),),
        ),
    )
    packs = plan_packed_sequences(examples, global_max_length=2)
    packed_supervision = build_packed_supervision(packs, examples)

    sequence = build_token_sequence_from_packed_supervision(packs[1], packed_supervision)

    assert [atom.pack_index for atom in sequence.atoms] == [1]
    assert [atom.example_id for atom in sequence.atoms] == ["ex-1"]


def test_token_span_rejects_mixed_provenance_atoms() -> None:
    first = _atom(
        target_position=1,
        segment_index=0,
        example_id="ex-0",
        token_id=11,
        object_id="obj-0",
        field="x1",
    )
    second = _atom(
        target_position=2,
        segment_index=0,
        example_id="ex-0",
        token_id=12,
        object_id="obj-1",
        field="x2",
    )

    with pytest.raises(LossContractError) as exc_info:
        TokenSpan(
            pack_index=0,
            segment_index=0,
            example_index=0,
            example_id="ex-0",
            token_type="schema",
            atoms=(first, second),
            text="x",
            object_id="obj-0",
            field="x1",
        )

    assert exc_info.value.code == "supervision.span_identity"


# ---------------------------------------------------------------------------
# Canonical causal-logit position selection (design decision 6)
# ---------------------------------------------------------------------------


def _single_segment_pack_fields(*, pack_length: int) -> dict[str, object]:
    return {
        "pack_index": 0,
        "input_ids": tuple(range(10, 10 + pack_length)),
        "segments": (
            PackedSegment(
                pack_index=0,
                segment_index=0,
                example_index=0,
                example_id="ex-0",
                start=0,
                end=pack_length,
            ),
        ),
    }


def test_causal_logits_positions_are_sorted_and_causally_shifted() -> None:
    atoms = tuple(
        _atom(
            target_position=position,
            segment_index=0,
            example_id="ex-0",
            token_id=100 + position,
        )
        # Deliberately unsorted: selection owns the ordering, not the caller.
        for position in (5, 1, 3)
    )
    sequence = TokenSequence(
        **_single_segment_pack_fields(pack_length=8), atoms=atoms, spans=()
    )

    positions = sequence.causal_logits_positions()

    assert positions == (0, 2, 4)
    assert positions == tuple(
        sorted(atom.target_position - 1 for atom in sequence.atoms)
    )
    assert isinstance(positions, tuple)
    assert all(type(position) is int for position in positions)


def test_causal_logits_positions_are_none_for_an_empty_atom_set() -> None:
    sequence = TokenSequence(
        **_single_segment_pack_fields(pack_length=4), atoms=(), spans=()
    )

    assert sequence.causal_logits_positions() is None


def test_causal_logits_positions_deduplicate_repeated_positions() -> None:
    """Duplicate selections must collapse before they reach Qwen forward."""

    repeated = _atom(
        target_position=3, segment_index=0, example_id="ex-0", token_id=13
    )

    positions = TokenSequence.causal_logits_positions(
        SimpleNamespace(atoms=(repeated, repeated))
    )

    assert positions == (2,)


def test_causal_logits_positions_select_nothing_without_atoms() -> None:
    """A token-sequence stand-in that reports no atoms selects no positions.

    This is the contract the deleted ``_logits_positions_to_keep`` trainer
    helper had, and the frozen ``causal_logits.json`` characterization keeps it
    (manifest ``harness_seam_repoints``); the selector must not start raising.
    """

    assert TokenSequence.causal_logits_positions(SimpleNamespace()) is None
    assert TokenSequence.causal_logits_positions(SimpleNamespace(atoms=None)) is None
    assert TokenSequence.causal_logits_positions(SimpleNamespace(atoms=())) is None


def _atom(
    *,
    target_position: int,
    segment_index: int,
    example_id: str,
    token_id: int,
    object_id: str | None = None,
    field: str | None = None,
) -> TokenAtom:
    return TokenAtom(
        pack_index=0,
        segment_index=segment_index,
        example_index=segment_index,
        example_id=example_id,
        target_position=target_position,
        token_id=token_id,
        token_type="schema",
        text="x",
        logical_target_position=target_position,
        object_id=object_id,
        field=field,
        source="unit",
    )


@dataclass(frozen=True)
class FakeEncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    supervised_token_spans: tuple["FakeTokenSpan", ...]


@dataclass(frozen=True)
class FakeTokenSpan:
    token_type: str
    physical: int
    token_ids: tuple[int, ...]
    text: str = "x"
    object_id: str | None = None
    field: str | None = None
    source: str | None = None

    @property
    def physical_token_start(self) -> int:
        return self.physical

    @property
    def physical_token_end(self) -> int:
        return self.physical + len(self.token_ids)

    @property
    def base_token_start(self) -> int:
        return self.physical

    @property
    def base_token_end(self) -> int:
        return self.physical + len(self.token_ids)
