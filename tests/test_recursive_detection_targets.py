from __future__ import annotations

from types import SimpleNamespace

from src.detection.dataset import DetectionTrainingDataset
from src.detection.objective import (
    LossAtom,
    RecursiveDetectionTargets,
    SemanticRole,
    StateWeightingDiagnostics,
    TokenTarget,
)
from src.detection.tokenization import TokenRole


def test_recursive_target_positions_and_loss_atoms_shift_together() -> None:
    targets = RecursiveDetectionTargets(
        token_targets=(
            TokenTarget(
                position=1,
                teacher_token_id=11,
                kind="hard_ce",
                trie_branch_targets=(),
                object_instance_id="obj-1",
                token_role=TokenRole.DESC,
                semantic_role=SemanticRole.DESC_IDENTITY,
                loss_atom_id="desc:obj-1",
            ),
            TokenTarget(
                position=3,
                teacher_token_id=22,
                kind="hard_ce",
                trie_branch_targets=(),
                object_instance_id="obj-1",
                token_role=TokenRole.COORD,
                semantic_role=SemanticRole.BBOX_COORD,
                loss_atom_id="bbox:obj-1",
            ),
        ),
        state_weighting="legacy_row_mean_prefix_mixture_equivalence",
        normalization="legacy_row_mean_equivalence",
        loss_atoms=(
            LossAtom(
                atom_id="desc:obj-1",
                semantic_role=SemanticRole.DESC_IDENTITY,
                token_positions=(1,),
                object_instance_id="obj-1",
            ),
            LossAtom(
                atom_id="bbox:obj-1",
                semantic_role=SemanticRole.BBOX_COORD,
                token_positions=(3,),
                object_instance_id="obj-1",
            ),
        ),
        state_weighting_diagnostics=StateWeightingDiagnostics(
            profile_id="legacy_row_mean_prefix_mixture_equivalence",
            prefix_length_probabilities=(1.0,),
            supervised_token_counts_by_prefix_length=(2,),
            entry_exposures=(),
            separator_exposures=(),
            terminal_exposure=0.0,
        ),
    )
    prepared = SimpleNamespace(
        input_ids=(99, 11, 88, 22),
        labels=(-100, 11, -100, 22),
        recursive_detection_targets=targets,
    )
    encoded = {
        "input_ids": (77, 99, 11, 88, 22),
        "labels": (-100, -100, 11, -100, 22),
    }

    dataset = object.__new__(DetectionTrainingDataset)
    shifted = DetectionTrainingDataset._align_prepared_targets_to_encoded(
        dataset,
        encoded,
        prepared,
    )

    assert shifted is not None
    target_positions = {target.position for target in shifted.token_targets}
    atom_positions = {
        position
        for atom in shifted.loss_atoms
        for position in atom.token_positions
    }
    assert target_positions == {2, 4}
    assert target_positions == atom_positions
    assert shifted.token_position_origin == "DetectionTrainingDataset.encoded"
