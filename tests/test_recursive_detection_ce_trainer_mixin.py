from __future__ import annotations

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

from src.detection.objective import (
    LossAtom,
    RecursiveDetectionTargets,
    SemanticRole,
    StateWeightingDiagnostics,
    TokenTarget,
    TrieBranchTarget,
)
from src.detection.tokenization import TokenRole
from src.trainers.batch_extras import RECURSIVE_DETECTION_TARGETS_KEY
from src.trainers.metrics.mixins import RecursiveDetectionCEMixin


class _Metric:
    def __init__(self) -> None:
        self.values: list[float] = []

    def update(self, value: float) -> None:
        self.values.append(float(value))


class _DummyModel:
    training = True

    def __init__(self, logits: torch.Tensor) -> None:
        self.logits = logits
        self.forward_inputs: dict[str, object] | None = None

    def __call__(self, **inputs):
        self.forward_inputs = dict(inputs)
        return SimpleNamespace(logits=self.logits)


class _BaseTrainer:
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        raise AssertionError("recursive CE must own the model forward and loss")


class _Trainer(RecursiveDetectionCEMixin, _BaseTrainer):
    def __init__(self, cfg: object) -> None:
        self.recursive_detection_ce_cfg = cfg
        self.model = _DummyModel(torch.zeros((1, 1, 2), dtype=torch.float32))
        self.custom_metrics = {
            "train": defaultdict(_Metric),
            "eval": defaultdict(_Metric),
        }


def _state_weighting() -> StateWeightingDiagnostics:
    return StateWeightingDiagnostics(
        profile_id="uniform_permutation",
        prefix_length_probabilities=(1.0,),
        supervised_token_counts_by_prefix_length=(1,),
        entry_exposures=(),
        separator_exposures=(),
        terminal_exposure=1.0,
    )


def _targets() -> RecursiveDetectionTargets:
    return RecursiveDetectionTargets(
        token_targets=(
            TokenTarget(
                position=1,
                teacher_token_id=1,
                kind="hard_ce",
                trie_branch_targets=(),
                object_instance_id=None,
                token_role=TokenRole.ASSISTANT,
                semantic_role=SemanticRole.SCHEMA_CONTROL,
                loss_atom_id="schema",
            ),
        ),
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        loss_atoms=(
            LossAtom(
                atom_id="schema",
                semantic_role=SemanticRole.SCHEMA_CONTROL,
                token_positions=(1,),
            ),
        ),
        state_weighting_diagnostics=_state_weighting(),
    )


def test_recursive_detection_ce_mixin_owns_forward_and_strips_sidecar() -> None:
    logits = torch.tensor([[[0.0, 2.0], [0.0, 0.0]]], dtype=torch.float32, requires_grad=True)
    model = _DummyModel(logits)
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )

    loss, outputs = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[4, 1]], dtype=torch.long),
            "labels": torch.tensor([[-100, 1]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            RECURSIVE_DETECTION_TARGETS_KEY: (_targets(),),
        },
        return_outputs=True,
    )

    expected = -torch.log_softmax(logits[0, 0].float(), dim=-1)[1]
    assert loss.item() == pytest.approx(expected.item())
    assert outputs.logits is logits
    assert model.forward_inputs is not None
    assert RECURSIVE_DETECTION_TARGETS_KEY not in model.forward_inputs
    assert "labels" not in model.forward_inputs
    assert trainer.custom_metrics["train"]["loss/recursive_detection_ce"].values[-1] == pytest.approx(
        expected.item()
    )
    logged_keys = set(trainer.custom_metrics["train"].keys())
    assert "batch_loss" not in logged_keys
    assert "batch_size" not in logged_keys
    assert not any(key.startswith("compact/") for key in logged_keys)
    assert "detection_sequence/schema/token_acc/full_vocab/top1" in logged_keys
    assert "detection_sequence/objective/recursive_detection_ce/loss_per_sample" in logged_keys
    assert "detection_sequence/objective/recursive_detection_ce/batch_size" in logged_keys


def test_recursive_detection_ce_mixin_uses_public_trie_weight_names() -> None:
    logits = torch.tensor([[[-3.0, 4.0, -2.5], [0.0, 0.0, 0.0]]], dtype=torch.float32)
    model = _DummyModel(logits)
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=2.0,
            trie_balance_weight=1.0,
        )
    )
    branch_targets = RecursiveDetectionTargets(
        token_targets=(
            TokenTarget(
                position=1,
                teacher_token_id=0,
                kind="trie_multi_positive",
                trie_branch_targets=(
                    TrieBranchTarget(token_id=0, multiplicity=1, probability=0.5),
                    TrieBranchTarget(token_id=2, multiplicity=1, probability=0.5),
                ),
                object_instance_id="obj-0",
                token_role=TokenRole.OBJECT_ENTRY,
                semantic_role=SemanticRole.SCHEMA_CONTROL,
                loss_atom_id="schema",
            ),
        ),
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        loss_atoms=(
            LossAtom(
                atom_id="schema",
                semantic_role=SemanticRole.SCHEMA_CONTROL,
                token_positions=(1,),
            ),
        ),
        state_weighting_diagnostics=_state_weighting(),
    )

    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[4, 0]], dtype=torch.long),
            "labels": torch.tensor([[-100, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            RECURSIVE_DETECTION_TARGETS_KEY: (branch_targets,),
        },
    )

    log_probs = torch.log_softmax(logits[0, 0].float(), dim=-1)
    valid_log_mass = torch.logsumexp(log_probs[torch.tensor([0, 2])], dim=0)
    support = -valid_log_mass
    balance = -0.5 * (log_probs[0] - valid_log_mass) - 0.5 * (
        log_probs[2] - valid_log_mass
    )
    expected = 2.0 * support + balance
    assert loss.item() == pytest.approx(expected.item())
    assert trainer.custom_metrics["train"]["recursive_detection_ce/trie_support_weight"].values[-1] == pytest.approx(2.0)


def test_recursive_detection_ce_mixin_supervises_boundary_tokens_as_ordinary_ce() -> None:
    logits = torch.tensor(
        [[[-2.0, 1.0, 3.0], [-2.0, 0.5, 2.0], [0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    model = _DummyModel(logits)
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    targets = RecursiveDetectionTargets(
        token_targets=(
            TokenTarget(
                position=1,
                teacher_token_id=1,
                kind="hard_ce",
                trie_branch_targets=(),
                object_instance_id=None,
                token_role=TokenRole.ASSISTANT,
                semantic_role=SemanticRole.SEPARATOR_CONTINUE,
                loss_atom_id="separator",
            ),
            TokenTarget(
                position=2,
                teacher_token_id=2,
                kind="hard_ce",
                trie_branch_targets=(),
                object_instance_id=None,
                token_role=TokenRole.ASSISTANT,
                semantic_role=SemanticRole.CHAT_STOP,
                loss_atom_id="eos",
            ),
        ),
        state_weighting="uniform_permutation",
        normalization="semantic_image_bucket_balanced",
        loss_atoms=(
            LossAtom(
                atom_id="separator",
                semantic_role=SemanticRole.SEPARATOR_CONTINUE,
                token_positions=(1,),
            ),
            LossAtom(
                atom_id="eos",
                semantic_role=SemanticRole.CHAT_STOP,
                token_positions=(2,),
            ),
        ),
        state_weighting_diagnostics=_state_weighting(),
    )

    loss = trainer.compute_loss(
        model,
        {
            "input_ids": torch.tensor([[0, 1, 2]], dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1]], dtype=torch.long),
            RECURSIVE_DETECTION_TARGETS_KEY: (targets,),
        },
    )

    log_probs = torch.log_softmax(logits[0, :2].float(), dim=-1)
    separator_ce = -log_probs[0, 1]
    eos_ce = -log_probs[1, 2]
    expected = (separator_ce + eos_ce) / 2.0
    assert loss.item() == pytest.approx(expected.item())


def test_recursive_detection_ce_mixin_requires_sidecar_when_enabled() -> None:
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    model = _DummyModel(torch.zeros((1, 2, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="recursive_detection_targets"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[4]], dtype=torch.long),
                "labels": torch.tensor([[1]], dtype=torch.long),
            },
        )


def test_recursive_detection_ce_mixin_rejects_logits_to_keep_before_forward() -> None:
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    model = _DummyModel(torch.zeros((1, 2, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="logits_to_keep.*unsupported"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[4]], dtype=torch.long),
                "labels": torch.tensor([[1]], dtype=torch.long),
                "logits_to_keep": 1,
                RECURSIVE_DETECTION_TARGETS_KEY: (_targets(),),
            },
        )
    assert model.forward_inputs is None


def test_recursive_detection_ce_mixin_requires_full_time_logits() -> None:
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    model = _DummyModel(torch.zeros((1, 1, 2), dtype=torch.float32))

    with pytest.raises(RuntimeError, match="full unsliced logits"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[4, 1]], dtype=torch.long),
                "labels": torch.tensor([[-100, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                RECURSIVE_DETECTION_TARGETS_KEY: (_targets(),),
            },
        )


def test_recursive_detection_ce_mixin_rejects_sidecar_label_mismatch_before_forward() -> None:
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    model = _DummyModel(torch.zeros((1, 2, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="labels.*teacher_token_id"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[4, 1]], dtype=torch.long),
                "labels": torch.tensor([[-100, 0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                RECURSIVE_DETECTION_TARGETS_KEY: (_targets(),),
            },
        )
    assert model.forward_inputs is None


def test_recursive_detection_ce_mixin_rejects_sidecar_input_id_mismatch_before_forward() -> None:
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    model = _DummyModel(torch.zeros((1, 2, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="input_ids.*teacher_token_id"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[4, 0]], dtype=torch.long),
                "labels": torch.tensor([[-100, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
                RECURSIVE_DETECTION_TARGETS_KEY: (_targets(),),
            },
        )
    assert model.forward_inputs is None


def test_recursive_detection_ce_mixin_rejects_sidecar_padding_position_before_forward() -> None:
    trainer = _Trainer(
        SimpleNamespace(
            enabled=True,
            trie_support_weight=1.0,
            trie_balance_weight=1.0,
        )
    )
    model = _DummyModel(torch.zeros((1, 2, 2), dtype=torch.float32))

    with pytest.raises(ValueError, match="attention_mask"):
        trainer.compute_loss(
            model,
            {
                "input_ids": torch.tensor([[4, 1]], dtype=torch.long),
                "labels": torch.tensor([[-100, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 0]], dtype=torch.long),
                RECURSIVE_DETECTION_TARGETS_KEY: (_targets(),),
            },
        )
    assert model.forward_inputs is None
