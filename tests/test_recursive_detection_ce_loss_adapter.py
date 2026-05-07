from __future__ import annotations

import math

import pytest
import torch

import src.detection.loss as loss_module
from src.detection.loss import (
    RecursiveDetectionLossWeights,
    compute_recursive_detection_ce_batch_loss,
)
from src.detection.objective import (
    LossAtom,
    RecursiveDetectionTargets,
    SemanticRole,
    StateWeightingDiagnostics,
    TokenTarget,
    TrieBranchTarget,
)
from src.detection.tokenization import TokenRole
from src.metrics.events import flatten_metric_events, reduce_metric_events


def _state_weighting(profile_id: str = "uniform_permutation") -> StateWeightingDiagnostics:
    return StateWeightingDiagnostics(
        profile_id=profile_id,
        prefix_length_probabilities=(1.0,),
        supervised_token_counts_by_prefix_length=(1,),
        entry_exposures=(),
        separator_exposures=(),
        terminal_exposure=1.0,
    )


def _hard_target(
    *,
    position: int,
    teacher_token_id: int,
    semantic_role: SemanticRole | str | None = SemanticRole.OBJECT_CONTROL,
    state_weight: float = 1.0,
    loss_atom_id: str | None = None,
    object_instance_id: str | None = None,
) -> TokenTarget:
    return TokenTarget(
        position=position,
        teacher_token_id=teacher_token_id,
        kind="hard_ce",
        trie_branch_targets=(),
        object_instance_id=object_instance_id,
        token_role=TokenRole.ASSISTANT,
        state_weight=state_weight,
        semantic_role=semantic_role,
        loss_atom_id=loss_atom_id,
    )


def _branch_target(
    *,
    position: int,
    teacher_token_id: int,
    branches: tuple[tuple[int, int], ...],
    semantic_role: SemanticRole = SemanticRole.ENTRY_TRIE_DECISION,
    state_weight: float = 1.0,
    loss_atom_id: str | None = None,
) -> TokenTarget:
    active_count = sum(multiplicity for _, multiplicity in branches)
    return TokenTarget(
        position=position,
        teacher_token_id=teacher_token_id,
        kind="trie_multi_positive",
        trie_branch_targets=tuple(
            TrieBranchTarget(
                token_id=token_id,
                multiplicity=multiplicity,
                probability=float(multiplicity / active_count),
            )
            for token_id, multiplicity in branches
        ),
        object_instance_id="obj-0",
        token_role=TokenRole.OBJECT_ENTRY,
        state_weight=state_weight,
        semantic_role=semantic_role,
        loss_atom_id=loss_atom_id,
    )


def _targets(
    *,
    token_targets: tuple[TokenTarget, ...],
    normalization: str = "legacy_row_mean_equivalence",
    state_weighting: str = "uniform_permutation",
    loss_atoms: tuple[LossAtom, ...] | None = None,
) -> RecursiveDetectionTargets:
    atoms = loss_atoms
    if atoms is None:
        atoms = tuple(
            LossAtom(
                atom_id=target.loss_atom_id or f"atom:{target.position}",
                semantic_role=target.semantic_role,
                token_positions=(target.position,),
            )
            for target in token_targets
        )
    return RecursiveDetectionTargets(
        token_targets=token_targets,
        state_weighting=state_weighting,
        normalization=normalization,
        loss_atoms=atoms,
        state_weighting_diagnostics=_state_weighting(state_weighting),
    )


def test_hard_singleton_ce_matches_standard_log_softmax() -> None:
    logits = torch.tensor(
        [
            [0.2, -0.5, 1.1, 0.3],
            [1.4, -0.2, 0.0, -1.3],
        ],
        dtype=torch.float32,
    )
    targets = _targets(
        token_targets=(
            _hard_target(position=1, teacher_token_id=2),
            _hard_target(position=2, teacher_token_id=0),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    expected_first = -torch.log_softmax(logits[0], dim=-1)[2]
    expected_second = -torch.log_softmax(logits[1], dim=-1)[0]
    expected = (expected_first + expected_second) / 2.0

    assert result.loss.item() == pytest.approx(expected.item())
    assert result.per_position_losses[0][1].item() == pytest.approx(expected_first.item())
    assert result.per_position_losses[0][2].item() == pytest.approx(expected_second.item())


def test_trie_branch_with_unit_weights_matches_object_uniform_soft_ce() -> None:
    logits = torch.tensor([[2.5, -0.1, 1.0, 0.0]], dtype=torch.float32)
    targets = _targets(
        token_targets=(
            _branch_target(position=1, teacher_token_id=0, branches=((0, 1), (2, 1))),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    log_probs = torch.log_softmax(logits[0], dim=-1)
    expected = -0.5 * (log_probs[0] + log_probs[2])

    assert result.loss.item() == pytest.approx(expected.item())
    assert result.per_position_losses[0][1].item() == pytest.approx(expected.item())


def test_support_reweight_changes_only_branch_loss_when_valid_mass_is_low() -> None:
    logits = torch.tensor(
        [[-2.0, 4.5, -1.5, -3.0], [0.1, 2.4, -0.5, 0.0]],
        dtype=torch.float32,
    )
    targets = _targets(
        token_targets=(
            _branch_target(position=1, teacher_token_id=0, branches=((0, 1), (2, 1))),
            _hard_target(position=2, teacher_token_id=1),
        )
    )

    baseline = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))
    reweighted = compute_recursive_detection_ce_batch_loss(
        logits=logits,
        targets=(targets,),
        weights=RecursiveDetectionLossWeights(
            support_weight=2.0,
            balance_weight=1.0,
        ),
    )

    assert reweighted.per_position_losses[0][1].item() > baseline.per_position_losses[0][1].item()
    assert reweighted.per_position_losses[0][2].item() == pytest.approx(
        baseline.per_position_losses[0][2].item()
    )


def test_duplicate_child_counts_define_balance_distribution_from_multiplicity() -> None:
    logits = torch.tensor([[0.8, 1.1, -0.4, 2.0]], dtype=torch.float32)
    targets = _targets(
        token_targets=(
            _branch_target(
                position=1,
                teacher_token_id=1,
                branches=((1, 2), (3, 1)),
            ),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    log_probs = torch.log_softmax(logits[0], dim=-1)
    expected = -(2.0 / 3.0) * log_probs[1] - (1.0 / 3.0) * log_probs[3]

    assert result.loss.item() == pytest.approx(expected.item())


def test_bfloat16_logits_are_upcast_for_stable_loss_computation() -> None:
    logits = torch.tensor(
        [[0.25, -0.75, 1.5], [1.25, -3.0, 0.0]],
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    targets = _targets(
        token_targets=(
            _hard_target(position=1, teacher_token_id=2),
            _branch_target(position=2, teacher_token_id=0, branches=((0, 1), (2, 1))),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    assert torch.isfinite(result.loss)
    assert result.loss.dtype == torch.float32
    result.loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_recursive_detection_metrics_map_semantic_roles_to_public_span_categories() -> None:
    roles_and_segments = (
        (SemanticRole.SCHEMA_CONTROL, "schema"),
        (SemanticRole.DESC_IDENTITY, "description"),
        (SemanticRole.BBOX_COORD, "coordinate"),
        (SemanticRole.ENTRY_TRIE_DECISION, "object_control"),
        (SemanticRole.OBJECT_CONTROL, "object_control"),
        (SemanticRole.SEPARATOR_CONTINUE, "separator"),
        (SemanticRole.TERMINAL_STOP, "stop"),
        (SemanticRole.CHAT_STOP, "stop"),
        (None, "other"),
        ("future_role", "other"),
    )
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0, -1.0, -2.0, -3.0],
            [0.0, 5.0, 1.0, -1.0, -2.0, -3.0],
            [0.0, 1.0, 6.0, -1.0, -2.0, -3.0],
            [0.0, 1.0, 2.0, 7.0, -2.0, -3.0],
            [0.0, 1.0, 2.0, -1.0, 8.0, -3.0],
            [0.0, 1.0, 2.0, -1.0, -2.0, 9.0],
            [10.0, 1.0, 2.0, -1.0, -2.0, -3.0],
            [0.0, 11.0, 2.0, -1.0, -2.0, -3.0],
            [0.0, 1.0, 12.0, -1.0, -2.0, -3.0],
            [0.0, 1.0, 2.0, 13.0, -2.0, -3.0],
        ],
        dtype=torch.float32,
    )
    token_targets = tuple(
        _hard_target(
            position=index,
            teacher_token_id=(index - 1) % logits.shape[-1],
            semantic_role=role,
            loss_atom_id=f"atom-{index}",
        )
        for index, (role, _) in enumerate(roles_and_segments, start=1)
    )
    targets = _targets(token_targets=token_targets)

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    denominators = {
        event.key: event.denominator
        for event in result.metric_events
        if event.key.endswith("/token_ce/full_vocab")
    }
    assert denominators["detection_sequence/schema/token_ce/full_vocab"] == pytest.approx(1.0)
    assert denominators["detection_sequence/description/token_ce/full_vocab"] == pytest.approx(1.0)
    assert denominators["detection_sequence/coordinate/token_ce/full_vocab"] == pytest.approx(1.0)
    assert denominators["detection_sequence/object_control/token_ce/full_vocab"] == pytest.approx(2.0)
    assert denominators["detection_sequence/separator/token_ce/full_vocab"] == pytest.approx(1.0)
    assert denominators["detection_sequence/stop/token_ce/full_vocab"] == pytest.approx(2.0)
    assert denominators["detection_sequence/other/token_ce/full_vocab"] == pytest.approx(2.0)
    assert all("/desc_text/" not in event.key for event in result.metric_events)
    assert all("/coord/" not in event.key for event in result.metric_events)


def test_recursive_detection_metrics_split_schema_desc_coord_and_object_spans() -> None:
    logits = torch.tensor(
        [
            [0.0, 0.2, 5.0, -1.0, -2.0, -3.0, -4.0],
            [6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0],
            [6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.0],
            [0.0, 1.0, 2.0, 6.0, -1.0, -2.0, -3.0],
        ],
        dtype=torch.float32,
    )
    targets = _targets(
        token_targets=(
            _hard_target(
                position=1,
                teacher_token_id=2,
                semantic_role=SemanticRole.SCHEMA_CONTROL,
            ),
            _hard_target(
                position=2,
                teacher_token_id=6,
                semantic_role=SemanticRole.DESC_IDENTITY,
            ),
            _hard_target(
                position=3,
                teacher_token_id=4,
                semantic_role=SemanticRole.BBOX_COORD,
            ),
            _hard_target(
                position=4,
                teacher_token_id=3,
                semantic_role=SemanticRole.OBJECT_CONTROL,
            ),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))
    reduced = reduce_metric_events(result.metric_events)

    assert reduced["detection_sequence/schema/token_acc/full_vocab/top1"] == pytest.approx(1.0)
    assert reduced["detection_sequence/description/token_acc/full_vocab/top1"] == pytest.approx(0.0)
    assert reduced["detection_sequence/description/token_acc/full_vocab/top5"] == pytest.approx(0.0)
    assert reduced["detection_sequence/coordinate/token_acc/full_vocab/top1"] == pytest.approx(0.0)
    assert reduced["detection_sequence/coordinate/token_acc/full_vocab/top5"] == pytest.approx(1.0)
    assert reduced["detection_sequence/object_control/token_acc/full_vocab/top1"] == pytest.approx(1.0)
    expected_description_ce = -torch.log_softmax(logits[1], dim=-1)[6]
    assert reduced["detection_sequence/description/token_ce/full_vocab"] == pytest.approx(
        expected_description_ce.item()
    )


def test_recursive_detection_object_exact_metrics_use_object_instance_id() -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    token_targets = (
        _hard_target(
            position=1,
            teacher_token_id=0,
            semantic_role=SemanticRole.DESC_IDENTITY,
            loss_atom_id="a-desc",
        ),
        _hard_target(
            position=2,
            teacher_token_id=1,
            semantic_role=SemanticRole.BBOX_COORD,
            loss_atom_id="a-box",
        ),
        _hard_target(
            position=3,
            teacher_token_id=2,
            semantic_role=SemanticRole.DESC_IDENTITY,
            loss_atom_id="b-desc",
        ),
        _hard_target(
            position=4,
            teacher_token_id=2,
            semantic_role=SemanticRole.BBOX_COORD,
            loss_atom_id="b-box",
        ),
    )
    targets = _targets(
        token_targets=token_targets,
        loss_atoms=(
            LossAtom(
                atom_id="a-desc",
                semantic_role=SemanticRole.DESC_IDENTITY,
                token_positions=(1,),
                object_instance_id="object-a",
                object_index=0,
            ),
            LossAtom(
                atom_id="a-box",
                semantic_role=SemanticRole.BBOX_COORD,
                token_positions=(2,),
                object_instance_id="object-a",
                object_index=0,
            ),
            LossAtom(
                atom_id="b-desc",
                semantic_role=SemanticRole.DESC_IDENTITY,
                token_positions=(3,),
                object_instance_id="object-b",
                object_index=0,
            ),
            LossAtom(
                atom_id="b-box",
                semantic_role=SemanticRole.BBOX_COORD,
                token_positions=(4,),
                object_instance_id="object-b",
                object_index=0,
            ),
        ),
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))
    reduced = reduce_metric_events(result.metric_events)

    assert reduced["detection_sequence/object_entry/exact_sequence_match/object_entry"] == pytest.approx(0.5)


def test_recursive_detection_object_exact_metrics_distinguish_token_from_entry_correctness() -> None:
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
            [0.0, 1.0, 4.0],
            [4.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = _targets(
        token_targets=(
            _hard_target(
                position=1,
                teacher_token_id=0,
                semantic_role=SemanticRole.DESC_IDENTITY,
                object_instance_id="object-a",
            ),
            _hard_target(
                position=2,
                teacher_token_id=1,
                semantic_role=SemanticRole.DESC_IDENTITY,
                object_instance_id="object-a",
            ),
            _hard_target(
                position=3,
                teacher_token_id=2,
                semantic_role=SemanticRole.DESC_IDENTITY,
                object_instance_id="object-b",
            ),
            _hard_target(
                position=4,
                teacher_token_id=2,
                semantic_role=SemanticRole.DESC_IDENTITY,
                object_instance_id="object-b",
            ),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))
    reduced = reduce_metric_events(result.metric_events)

    assert reduced["detection_sequence/description/token_acc/full_vocab/top1"] == pytest.approx(0.75)
    assert reduced["detection_sequence/object_entry/exact_sequence_match/object_entry"] == pytest.approx(0.5)


def test_recursive_detection_object_exact_metric_omits_flat_rate_when_no_object_ids() -> None:
    logits = torch.tensor([[4.0, 1.0, 0.0]], dtype=torch.float32)
    targets = _targets(
        token_targets=(
            _hard_target(
                position=1,
                teacher_token_id=0,
                semantic_role=SemanticRole.SCHEMA_CONTROL,
            ),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))
    object_events = [
        event
        for event in result.metric_events
        if event.key == "detection_sequence/object_entry/exact_sequence_match/object_entry"
    ]
    flat = flatten_metric_events(result.metric_events)

    assert object_events
    assert object_events[-1].denominator == pytest.approx(0.0)
    assert "detection_sequence/object_entry/exact_sequence_match/object_entry" not in flat
    assert not any(key.startswith("compact/") for key in flat)


def test_recursive_detection_metric_summarization_runs_without_grad(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grad_enabled_flags: list[bool] = []
    input_requires_grad_flags: list[bool] = []
    original_log_softmax = loss_module.F.log_softmax

    def _recording_log_softmax(input_tensor: torch.Tensor, *args, **kwargs):
        grad_enabled_flags.append(torch.is_grad_enabled())
        input_requires_grad_flags.append(bool(input_tensor.requires_grad))
        return original_log_softmax(input_tensor, *args, **kwargs)

    monkeypatch.setattr(loss_module.F, "log_softmax", _recording_log_softmax)
    logits = torch.tensor(
        [
            [4.0, 1.0, 0.0],
            [0.0, 4.0, 1.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = _targets(
        token_targets=(
            _hard_target(
                position=1,
                teacher_token_id=0,
                semantic_role=SemanticRole.DESC_IDENTITY,
                object_instance_id="object-a",
            ),
            _hard_target(
                position=2,
                teacher_token_id=1,
                semantic_role=SemanticRole.BBOX_COORD,
                object_instance_id="object-a",
            ),
        )
    )

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    assert result.loss.requires_grad
    assert any(input_requires_grad_flags)
    assert grad_enabled_flags
    assert all(flag is False for flag in grad_enabled_flags[2:])
    for event in result.metric_events:
        for value in (event.numerator, event.denominator, event.value):
            assert not isinstance(value, torch.Tensor)


@pytest.mark.parametrize(
    ("token_targets", "match"),
    [
        ((_hard_target(position=0, teacher_token_id=1),), "position"),
        ((_branch_target(position=1, teacher_token_id=0, branches=((5, 1), (1, 1))),), "token id"),
    ],
)
def test_invalid_positions_and_child_token_ids_fail_fast(
    token_targets: tuple[TokenTarget, ...],
    match: str,
) -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    targets = _targets(token_targets=token_targets)

    with pytest.raises(ValueError, match=match):
        compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))


def test_trie_teacher_token_must_be_valid_child() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    targets = _targets(
        token_targets=(
            _branch_target(position=1, teacher_token_id=2, branches=((0, 1), (1, 1))),
        )
    )

    with pytest.raises(ValueError, match="teacher token"):
        compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))


def test_trie_child_multiplicities_must_be_positive() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    targets = _targets(
        token_targets=(
            _branch_target(position=1, teacher_token_id=1, branches=((0, 0), (1, 1))),
        )
    )

    with pytest.raises(ValueError, match="multiplicity"):
        compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))


def test_duplicate_target_positions_fail_fast() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32)
    targets = _targets(
        token_targets=(
            _hard_target(position=1, teacher_token_id=0),
            _hard_target(position=1, teacher_token_id=1),
        )
    )

    with pytest.raises(ValueError, match="Duplicate TokenTarget.position"):
        compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))


def test_semantic_normalization_requires_at_least_one_loss_component() -> None:
    logits = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
    targets = _targets(
        token_targets=(
            _hard_target(
                position=1,
                teacher_token_id=0,
                semantic_role=SemanticRole.DESC_IDENTITY,
                loss_atom_id="unowned-desc",
            ),
        ),
        normalization="semantic_image_bucket_balanced",
        loss_atoms=(
            LossAtom(
                atom_id="unowned-desc",
                semantic_role=SemanticRole.DESC_IDENTITY,
                token_positions=(1,),
            ),
        ),
    )

    with pytest.raises(ValueError, match="semantic_image_bucket_balanced"):
        compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))


def test_semantic_image_bucket_balanced_normalization_is_length_insensitive() -> None:
    logits = torch.zeros((12, 2), dtype=torch.float32)
    loss_atoms = (
        LossAtom(
            atom_id="object:0:desc_identity",
            semantic_role=SemanticRole.DESC_IDENTITY,
            token_positions=(1, 2),
            object_instance_id="obj-0",
            object_index=0,
        ),
        LossAtom(
            atom_id="object:0:bbox_coord",
            semantic_role=SemanticRole.BBOX_COORD,
            token_positions=(3,),
            object_instance_id="obj-0",
            object_index=0,
        ),
        LossAtom(
            atom_id="object:1:desc_identity",
            semantic_role=SemanticRole.DESC_IDENTITY,
            token_positions=(4,),
            object_instance_id="obj-1",
            object_index=1,
        ),
        LossAtom(
            atom_id="object:1:bbox_coord",
            semantic_role=SemanticRole.BBOX_COORD,
            token_positions=(5, 6, 7),
            object_instance_id="obj-1",
            object_index=1,
        ),
        LossAtom(
            atom_id="separator",
            semantic_role=SemanticRole.SEPARATOR_CONTINUE,
            token_positions=(8, 9),
        ),
        LossAtom(
            atom_id="terminal",
            semantic_role=SemanticRole.TERMINAL_STOP,
            token_positions=(10,),
        ),
        LossAtom(
            atom_id="schema",
            semantic_role=SemanticRole.SCHEMA_CONTROL,
            token_positions=(11, 12),
        ),
    )
    targets = _targets(
        token_targets=(
            _hard_target(
                position=1,
                teacher_token_id=0,
                semantic_role=SemanticRole.DESC_IDENTITY,
                loss_atom_id="object:0:desc_identity",
            ),
            _hard_target(
                position=2,
                teacher_token_id=0,
                semantic_role=SemanticRole.DESC_IDENTITY,
                loss_atom_id="object:0:desc_identity",
            ),
            _hard_target(
                position=3,
                teacher_token_id=0,
                semantic_role=SemanticRole.BBOX_COORD,
                loss_atom_id="object:0:bbox_coord",
            ),
            _hard_target(
                position=4,
                teacher_token_id=0,
                semantic_role=SemanticRole.DESC_IDENTITY,
                loss_atom_id="object:1:desc_identity",
            ),
            _hard_target(
                position=5,
                teacher_token_id=0,
                semantic_role=SemanticRole.BBOX_COORD,
                loss_atom_id="object:1:bbox_coord",
            ),
            _hard_target(
                position=6,
                teacher_token_id=0,
                semantic_role=SemanticRole.BBOX_COORD,
                loss_atom_id="object:1:bbox_coord",
            ),
            _hard_target(
                position=7,
                teacher_token_id=0,
                semantic_role=SemanticRole.BBOX_COORD,
                loss_atom_id="object:1:bbox_coord",
            ),
            _hard_target(
                position=8,
                teacher_token_id=0,
                semantic_role=SemanticRole.SEPARATOR_CONTINUE,
                loss_atom_id="separator",
            ),
            _hard_target(
                position=9,
                teacher_token_id=0,
                semantic_role=SemanticRole.SEPARATOR_CONTINUE,
                loss_atom_id="separator",
            ),
            _hard_target(
                position=10,
                teacher_token_id=0,
                semantic_role=SemanticRole.TERMINAL_STOP,
                loss_atom_id="terminal",
            ),
            _hard_target(
                position=11,
                teacher_token_id=0,
                semantic_role=SemanticRole.SCHEMA_CONTROL,
                loss_atom_id="schema",
            ),
            _hard_target(
                position=12,
                teacher_token_id=0,
                semantic_role=SemanticRole.SCHEMA_CONTROL,
                loss_atom_id="schema",
            ),
        ),
        normalization="semantic_image_bucket_balanced",
        loss_atoms=loss_atoms,
    )

    desired_losses = {
        1: 2.0,
        2: 4.0,
        3: 10.0,
        4: 1.0,
        5: 7.0,
        6: 7.0,
        7: 7.0,
        8: 5.0,
        9: 5.0,
        10: 9.0,
        11: 11.0,
        12: 11.0,
    }
    for position, desired_loss in desired_losses.items():
        teacher_logit = -math.log(math.exp(desired_loss) - 1.0)
        logits[position - 1, 0] = teacher_logit

    result = compute_recursive_detection_ce_batch_loss(logits=logits, targets=(targets,))

    object0 = (0.35 * 3.0 + 0.45 * 10.0) / (0.35 + 0.45)
    object1 = (0.35 * 1.0 + 0.45 * 7.0) / (0.35 + 0.45)
    object_component = (object0 + object1) / 2.0
    boundary_component = (0.5 * 5.0 + 0.5 * 9.0) / (0.5 + 0.5)
    schema_component = 11.0
    expected = (
        1.0 * object_component + 0.3 * boundary_component + 0.1 * schema_component
    ) / (1.0 + 0.3 + 0.1)

    assert result.loss.item() == pytest.approx(expected)
