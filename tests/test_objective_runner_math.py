from __future__ import annotations

import pytest
import torch

from src.metrics.events import MetricEvent
from src.training.objectives.runner import ObjectiveRunner
from src.training.objectives.trie_ce import support_balance_loss
from src.training.objectives.types import LabelLogitRowMap, ObjectiveSpec
from src.training.supervision.batch import SupervisionBatch
from src.training.supervision.distributions import (
    BoxRegressionDistribution,
    CoordinateSoftTokenDistribution,
    HardTokenDistribution,
    MultiPositiveTokenDistribution,
)
from src.training.supervision.spans import SupervisionSpan
from src.trainers.teacher_forcing.geometry import (
    bbox_smoothl1_ciou_loss,
    compute_bbox_regression_loss,
    expectation_decode_coords,
)


def _batch(*spans: SupervisionSpan) -> SupervisionBatch:
    return SupervisionBatch(spans=spans, batch_id="batch-1")


def _span(
    *,
    sample_id: str = "sample-1",
    label_positions: tuple[int, ...] = (1,),
    distribution: object,
    role: str = "schema",
    metadata: dict[str, object] | None = None,
) -> SupervisionSpan:
    return SupervisionSpan(
        sample_id=sample_id,
        role=role,
        label_positions=label_positions,
        distribution=distribution,
        metadata=metadata or {},
    )


def _run(
    logits: torch.Tensor,
    supervision: SupervisionBatch,
    *objectives: ObjectiveSpec,
):
    return ObjectiveRunner().run(
        logits=logits,
        supervision=supervision,
        objectives=objectives,
    )


def _coord_vocab(size: int = 1000, *, start: int = 5) -> list[int]:
    return list(range(start, start + size))


def test_hard_token_ce_uses_default_causal_label_position_to_logit_row() -> None:
    logits = torch.tensor(
        [
            [9.0, -2.0, -2.0],
            [-2.0, 9.0, -2.0],
            [9.0, -2.0, -2.0],
        ],
        dtype=torch.float32,
    )
    supervision = _batch(
        _span(label_positions=(2,), distribution=HardTokenDistribution(token_id=1))
    )

    result = _run(logits, supervision, ObjectiveSpec("token_ce"))

    expected = -torch.log_softmax(logits[1], dim=-1)[1]
    assert result.objectives["token_ce"].loss.item() == pytest.approx(expected.item())
    assert result.loss.item() == pytest.approx(expected.item())


def test_hard_token_ce_matches_manual_log_softmax_with_state_denominator() -> None:
    logits = torch.tensor(
        [
            [1.0, -0.5, 2.0, 0.25],
            [0.0, 1.5, -1.0, 0.5],
        ],
        dtype=torch.float32,
    )
    spans = (
        _span(
            label_positions=(1,),
            distribution=HardTokenDistribution(token_id=2),
            metadata={"loss_weight": 3.0, "state_weight": 2.0},
        ),
        _span(
            label_positions=(2,),
            distribution=HardTokenDistribution(token_id=1),
            metadata={"loss_weight": 0.5, "state_weight": 4.0},
        ),
    )

    result = _run(logits, _batch(*spans), ObjectiveSpec("token_ce"))

    ce0 = -torch.log_softmax(logits[0], dim=-1)[2]
    ce1 = -torch.log_softmax(logits[1], dim=-1)[1]
    expected_num = 2.0 * 3.0 * ce0 + 4.0 * 0.5 * ce1
    expected = expected_num / 6.0
    objective = result.objectives["token_ce"]

    assert objective.numerator.item() == pytest.approx(expected_num.item())
    assert objective.denominator.item() == pytest.approx(6.0)
    assert objective.loss.item() == pytest.approx(expected.item())


def test_multi_positive_support_balance_unit_weights_matches_sparse_soft_ce() -> None:
    logits_row = torch.tensor([2.5, -0.1, 1.0, 0.0], dtype=torch.float32)

    actual = support_balance_loss(
        logits_row,
        positive_token_ids=(0, 2),
        support_weight=1.0,
        balance_weight=1.0,
    )

    log_probs = torch.log_softmax(logits_row, dim=-1)
    expected = -0.5 * (log_probs[0] + log_probs[2])
    assert actual.item() == pytest.approx(expected.item())


def test_weighted_multi_positive_trie_ce_matches_sparse_soft_ce() -> None:
    logits = torch.tensor([[2.5, -0.1, 1.0, 0.0]], dtype=torch.float32)
    distribution = MultiPositiveTokenDistribution(
        token_ids=(0, 2),
        token_weights=(2.0, 1.0),
    )

    result = _run(
        logits,
        _batch(_span(distribution=distribution)),
        ObjectiveSpec("trie_ce"),
    )

    log_probs = torch.log_softmax(logits[0], dim=-1)
    expected = -(2.0 / 3.0) * log_probs[0] - (1.0 / 3.0) * log_probs[2]
    assert result.objectives["trie_ce"].loss.item() == pytest.approx(expected.item())


def test_support_weight_distinguishes_low_valid_mass() -> None:
    logits_row = torch.tensor([-5.0, 5.0, -4.5, 4.0], dtype=torch.float32)

    baseline = support_balance_loss(
        logits_row,
        positive_token_ids=(0, 2),
        support_weight=1.0,
        balance_weight=1.0,
    )
    reweighted = support_balance_loss(
        logits_row,
        positive_token_ids=(0, 2),
        support_weight=2.0,
        balance_weight=1.0,
    )

    assert reweighted.item() > baseline.item()


def test_runner_sums_objective_local_weighted_losses_without_mixed_denominator() -> None:
    logits = torch.tensor(
        [
            [3.0, -1.0, 0.5, 0.0],
            [0.0, -0.5, 2.0, 1.0],
            [-2.0, 2.5, 0.0, 0.25],
        ],
        dtype=torch.float32,
    )
    token_span = _span(
        label_positions=(1,),
        distribution=HardTokenDistribution(token_id=0),
        metadata={"state_weight": 10.0, "loss_weight": 1.0},
    )
    trie_span = _span(
        label_positions=(3,),
        distribution=MultiPositiveTokenDistribution(token_ids=(1, 3)),
        metadata={"state_weight": 1.0, "loss_weight": 1.0},
    )

    result = _run(
        logits,
        _batch(token_span, trie_span),
        ObjectiveSpec("token_ce", weight=2.0),
        ObjectiveSpec("trie_ce", weight=3.0),
    )

    token = result.objectives["token_ce"]
    trie = result.objectives["trie_ce"]
    expected = 2.0 * token.loss + 3.0 * trie.loss
    mixed_denominator = (2.0 * token.numerator + 3.0 * trie.numerator) / (
        token.denominator + trie.denominator
    )

    assert result.loss.item() == pytest.approx(expected.item())
    assert result.loss.item() != pytest.approx(mixed_denominator.item())


def test_type_gate_metadata_fails_fast() -> None:
    supervision = _batch(
        _span(
            distribution=HardTokenDistribution(token_id=0),
            metadata={"type_gate": "coord"},
        )
    )

    with pytest.raises(NotImplementedError, match="type_gate.*deferred|unsupported"):
        _run(torch.zeros((1, 3), dtype=torch.float32), supervision, ObjectiveSpec("token_ce"))


def test_objective_runner_rejects_missing_distribution_objective_coverage() -> None:
    logits = torch.zeros((2, 8), dtype=torch.float32)
    supervision = _batch(
        _span(
            label_positions=(1,),
            distribution=HardTokenDistribution(token_id=2),
        ),
        _span(
            label_positions=(2,),
            role="coordinate",
            distribution=CoordinateSoftTokenDistribution(
                token_weights=((4, 1.0),),
                loss_mode="full_vocab_ce",
            ),
        ),
    )

    with pytest.raises(ValueError, match="coord_soft_ce.*coordinate_soft_token"):
        _run(logits, supervision, ObjectiveSpec("token_ce"))


@pytest.mark.parametrize(
    ("distribution", "objective_id"),
    (
        (HardTokenDistribution(token_id=2), "token_ce"),
        (MultiPositiveTokenDistribution(token_ids=(1, 3)), "trie_ce"),
        (
            CoordinateSoftTokenDistribution(token_weights=((4, 1.0),)),
            "coord_soft_ce",
        ),
    ),
)
def test_token_objectives_reject_multi_position_spans(
    distribution: object,
    objective_id: str,
) -> None:
    logits = torch.zeros((4, 8), dtype=torch.float32)
    supervision = _batch(
        _span(
            label_positions=(1, 2),
            distribution=distribution,
            role="coordinate",
        )
    )

    with pytest.raises(ValueError, match="exactly one label position"):
        _run(logits, supervision, ObjectiveSpec(objective_id))


def test_box_regression_rejects_non_four_position_spans() -> None:
    coord_token_ids = _coord_vocab()
    logits = torch.zeros((3, max(coord_token_ids) + 1), dtype=torch.float32)
    supervision = _batch(
        _span(
            label_positions=(1, 2, 3),
            role="coordinate",
            distribution=BoxRegressionDistribution(target_bbox=(0.1, 0.2, 0.3, 0.4)),
        )
    )

    with pytest.raises(ValueError, match="exactly four label positions"):
        _run(
            logits,
            supervision,
            ObjectiveSpec("box_regression", config={"coord_token_ids": coord_token_ids}),
        )


@pytest.mark.parametrize(
    "kwargs,error_type,error_match",
    (
        (
            {"positive_token_ids": (True, 2)},
            TypeError,
            "positive_token_ids",
        ),
        (
            {"positive_token_ids": torch.tensor([1.0, 2.0])},
            TypeError,
            "positive_token_ids",
        ),
        (
            {"positive_token_ids": (1, 2), "positive_weights": (True, 1.0)},
            TypeError,
            "positive_weights",
        ),
        (
            {"positive_token_ids": (1, 2), "positive_weights": (1.0, float("nan"))},
            ValueError,
            "finite",
        ),
        (
            {"positive_token_ids": (1, 2), "support_weight": True},
            TypeError,
            "support_weight",
        ),
        (
            {"positive_token_ids": (1, 2), "balance_weight": float("inf")},
            ValueError,
            "balance_weight",
        ),
    ),
)
def test_support_balance_loss_rejects_invalid_ids_and_weights(
    kwargs: dict[str, object],
    error_type: type[Exception],
    error_match: str,
) -> None:
    with pytest.raises(error_type, match=error_match):
        support_balance_loss(torch.zeros((4,), dtype=torch.float32), **kwargs)


def test_coordinate_soft_ce_full_vocab_matches_manual_soft_ce() -> None:
    logits = torch.tensor([[0.25, 1.5, -0.75, 0.5]], dtype=torch.float32)
    distribution = CoordinateSoftTokenDistribution(
        token_weights=((1, 2.0), (3, 1.0)),
        loss_mode="full_vocab_ce",
    )

    result = _run(
        logits,
        _batch(_span(distribution=distribution, role="coordinate")),
        ObjectiveSpec("coord_soft_ce"),
    )

    log_probs = torch.log_softmax(logits[0], dim=-1)
    probs = torch.tensor([2.0 / 3.0, 1.0 / 3.0], dtype=torch.float32)
    expected = -(probs * log_probs[torch.tensor([1, 3])]).sum()
    objective = result.objectives["coord_soft_ce"]

    assert objective.loss.item() == pytest.approx(expected.item())
    assert objective.state["support_token_count"] == 2
    assert objective.state["target_entropy"] > 0.0


def test_coord_vocab_ce_uses_full_coordinate_vocab_not_target_support_only() -> None:
    coord_token_ids = [3, 4, 5, 6]
    logits = torch.zeros((1, 7), dtype=torch.float32)
    logits[0, 6] = 10.0
    distribution = CoordinateSoftTokenDistribution(
        token_weights=((4, 1.0),),
        loss_mode="coord_vocab_ce",
    )

    result = _run(
        logits,
        _batch(_span(distribution=distribution, role="coordinate")),
        ObjectiveSpec("coord_soft_ce", config={"coord_token_ids": coord_token_ids}),
    )

    coord_logits = logits[0, torch.tensor(coord_token_ids)]
    expected = -torch.log_softmax(coord_logits, dim=-1)[1]
    assert result.objectives["coord_soft_ce"].loss.item() == pytest.approx(
        expected.item()
    )
    assert result.objectives["coord_soft_ce"].loss.item() > 5.0


def test_w1_distance_uses_coord_vocab_bins_not_raw_token_id_distance() -> None:
    coord_token_ids = [10, 11]
    logits = torch.zeros((1, 12), dtype=torch.float32)
    logits[0, 11] = 12.0
    distribution = CoordinateSoftTokenDistribution(
        token_weights=((10, 1.0),),
        loss_mode="w1_distance",
    )

    result = _run(
        logits,
        _batch(_span(distribution=distribution, role="coordinate")),
        ObjectiveSpec("coord_soft_ce", config={"coord_token_ids": coord_token_ids}),
    )

    objective = result.objectives["coord_soft_ce"]
    assert objective.loss.item() == pytest.approx(1.0, abs=1e-4)
    assert objective.loss.item() < 2.0


def test_coord_vocab_modes_reject_missing_target_token_in_coord_vocab() -> None:
    distribution = CoordinateSoftTokenDistribution(
        token_weights=((7, 1.0),),
        loss_mode="coord_vocab_ce",
    )

    with pytest.raises(ValueError, match="absent from coord_token_ids"):
        _run(
            torch.zeros((1, 8), dtype=torch.float32),
            _batch(_span(distribution=distribution, role="coordinate")),
            ObjectiveSpec("coord_soft_ce", config={"coord_token_ids": [3, 4, 5]}),
        )


@pytest.mark.parametrize(
    "objective_id,config",
    (
        ("coord_soft_ce", {"coord_gate_weight": 1.0}),
        ("coord_soft_ce", {"target_sigma": 2.0}),
        ("box_regression", {"log_wh_weight": 1.0}),
        ("box_regression", {"bbox_size_aux_weight": 1.0}),
    ),
)
def test_deferred_objective_config_keys_fail_explicitly(
    objective_id: str,
    config: dict[str, object],
) -> None:
    with pytest.raises((NotImplementedError, ValueError), match="deferred/unsupported"):
        _run(
            torch.zeros((1, 12), dtype=torch.float32),
            SupervisionBatch(),
            ObjectiveSpec(objective_id, config=config),
        )


def test_objective_runner_rejects_duplicate_objective_ids() -> None:
    with pytest.raises(ValueError, match="unique"):
        _run(
            torch.zeros((1, 3), dtype=torch.float32),
            SupervisionBatch(),
            ObjectiveSpec("token_ce"),
            ObjectiveSpec("token_ce", weight=2.0),
        )


def test_objective_runner_allows_empty_3d_batch_without_sample_row_map() -> None:
    logits = torch.zeros((2, 3, 4), dtype=torch.float32, requires_grad=True)

    result = ObjectiveRunner().run(
        logits=logits,
        supervision=SupervisionBatch(),
        objectives=(ObjectiveSpec("token_ce"),),
    )

    assert result.loss.item() == pytest.approx(0.0)
    assert result.objectives["token_ce"].loss.item() == pytest.approx(0.0)
    assert result.objectives["token_ce"].span_count == 0
    result.loss.backward()
    assert logits.grad is not None


def test_objective_runner_allows_empty_3d_batch_with_no_objectives() -> None:
    logits = torch.zeros((2, 3, 4), dtype=torch.float32, requires_grad=True)

    result = ObjectiveRunner().run(
        logits=logits,
        supervision=SupervisionBatch(),
        objectives=(),
    )

    assert result.loss.item() == pytest.approx(0.0)
    assert result.objectives == {}
    result.loss.backward()
    assert logits.grad is not None


def test_objective_runner_routes_3d_logits_by_sample_id() -> None:
    logits = torch.tensor(
        [
            [[0.0, -1.0, 3.0, 0.5], [0.0, 2.0, -1.0, 0.5]],
            [[1.0, -1.0, 0.0, 0.5], [-2.0, 3.0, 0.0, 0.5]],
        ],
        dtype=torch.float32,
    )
    supervision = _batch(
        _span(
            sample_id="sample-a",
            label_positions=(1,),
            distribution=HardTokenDistribution(token_id=2),
        ),
        _span(
            sample_id="sample-b",
            label_positions=(2,),
            distribution=HardTokenDistribution(token_id=1),
        ),
    )

    result = ObjectiveRunner().run(
        logits=logits,
        supervision=supervision,
        objectives=(ObjectiveSpec("token_ce"),),
        sample_id_to_batch_index={"sample-a": 0, "sample-b": 1},
    )

    expected = (
        -torch.log_softmax(logits[0, 0], dim=-1)[2]
        - torch.log_softmax(logits[1, 1], dim=-1)[1]
    ) / 2.0
    assert result.objectives["token_ce"].loss.item() == pytest.approx(expected.item())


@pytest.mark.parametrize(
    "label_rows,logits,error_match",
    (
        (
            LabelLogitRowMap.from_logits(torch.zeros((2, 4), dtype=torch.float32)),
            torch.zeros((1, 2, 4), dtype=torch.float32),
            "2D label row map",
        ),
        (
            LabelLogitRowMap.from_logits(
                torch.zeros((1, 2, 4), dtype=torch.float32),
                sample_id_to_batch_index={"sample-1": 0},
            ),
            torch.zeros((2, 4), dtype=torch.float32),
            "3D label row map",
        ),
        (
            LabelLogitRowMap(time_steps=3, vocab_size=4),
            torch.zeros((2, 4), dtype=torch.float32),
            "time_steps",
        ),
        (
            LabelLogitRowMap(time_steps=2, vocab_size=5),
            torch.zeros((2, 4), dtype=torch.float32),
            "vocab_size",
        ),
        (
            LabelLogitRowMap(
                time_steps=2,
                vocab_size=4,
                batch_size=2,
                sample_id_to_batch_index={"sample-1": 0},
            ),
            torch.zeros((1, 2, 4), dtype=torch.float32),
            "batch_size",
        ),
    ),
)
def test_objective_runner_rejects_stale_label_row_maps(
    label_rows: LabelLogitRowMap,
    logits: torch.Tensor,
    error_match: str,
) -> None:
    with pytest.raises(ValueError, match=error_match):
        ObjectiveRunner().run(
            logits=logits,
            supervision=SupervisionBatch(),
            objectives=(ObjectiveSpec("token_ce"),),
            label_rows=label_rows,
        )


@pytest.mark.parametrize("batch_index", (-1, 2))
def test_label_logit_row_map_direct_construction_rejects_invalid_batch_indices(
    batch_index: int,
) -> None:
    with pytest.raises(ValueError, match="sample_id_to_batch_index|batch index"):
        LabelLogitRowMap(
            time_steps=2,
            vocab_size=4,
            batch_size=2,
            sample_id_to_batch_index={"sample-1": batch_index},
        )


def test_label_logit_row_map_direct_construction_requires_mapping_for_3d() -> None:
    with pytest.raises(ValueError, match="sample_id_to_batch_index.*3D"):
        LabelLogitRowMap(time_steps=2, vocab_size=4, batch_size=2)


def test_label_logit_row_map_direct_construction_rejects_mapping_for_2d() -> None:
    with pytest.raises(ValueError, match="2D.*sample_id_to_batch_index"):
        LabelLogitRowMap(
            time_steps=2,
            vocab_size=4,
            sample_id_to_batch_index={"sample-1": 0},
        )


def test_label_logit_row_map_freezes_direct_mapping() -> None:
    source = {"sample-1": 0}
    row_map = LabelLogitRowMap(
        time_steps=2,
        vocab_size=4,
        batch_size=2,
        sample_id_to_batch_index=source,
    )

    source["sample-1"] = 1

    assert row_map.sample_id_to_batch_index["sample-1"] == 0
    with pytest.raises(TypeError):
        row_map.sample_id_to_batch_index["sample-2"] = 1  # type: ignore[index]


def test_token_ce_rejects_non_finite_logits() -> None:
    logits = torch.tensor([[0.0, float("nan"), 1.0]], dtype=torch.float32)
    supervision = _batch(
        _span(label_positions=(1,), distribution=HardTokenDistribution(token_id=2))
    )

    with pytest.raises((FloatingPointError, ValueError), match="non-finite|finite"):
        _run(logits, supervision, ObjectiveSpec("token_ce"))


def test_trie_ce_rejects_non_finite_logits() -> None:
    logits = torch.tensor([[0.0, float("nan"), 1.0]], dtype=torch.float32)
    supervision = _batch(
        _span(
            label_positions=(1,),
            distribution=MultiPositiveTokenDistribution(token_ids=(0, 2)),
        )
    )

    with pytest.raises((FloatingPointError, ValueError), match="non-finite|finite"):
        _run(logits, supervision, ObjectiveSpec("trie_ce"))


def test_box_regression_returns_finite_float32_loss_and_decoded_metrics() -> None:
    coord_token_ids = _coord_vocab()
    vocab = max(coord_token_ids) + 1
    logits = torch.zeros((4, vocab), dtype=torch.float32)
    target_bins = torch.tensor([100, 150, 500, 650], dtype=torch.long)
    for slot_index, target_bin in enumerate(target_bins.tolist()):
        logits[slot_index, coord_token_ids[target_bin]] = 8.0

    target_box = tuple((target_bins.float() / 999.0).tolist())
    supervision = _batch(
        _span(
            label_positions=(1, 2, 3, 4),
            role="coordinate",
            distribution=BoxRegressionDistribution(target_bbox=target_box),
            metadata={"group_weight": 2.5},
        )
    )

    result = _run(
        logits,
        supervision,
        ObjectiveSpec(
            "box_regression",
            config={
                "coord_token_ids": coord_token_ids,
                "temperature": 0.25,
                "smoothl1_weight": 1.25,
                "ciou_weight": 0.75,
                "parameterization": "xyxy",
            },
        ),
    )
    objective = result.objectives["box_regression"]

    coord_logits = logits.index_select(
        dim=-1,
        index=torch.tensor(coord_token_ids, dtype=torch.long),
    )
    pred = expectation_decode_coords(
        coord_logits=coord_logits,
        temperature=0.25,
        mode="exp",
    ).reshape(1, 4)
    target = torch.tensor([target_box], dtype=torch.float32)
    regression = compute_bbox_regression_loss(
        pred_boxes_xyxy=pred,
        target_boxes_xyxy=target,
        parameterization="xyxy",
    )
    ciou = bbox_smoothl1_ciou_loss(pred_xyxy=pred, gt_xyxy=target)
    expected = 1.25 * regression.per_box.mean() + 0.75 * ciou.ciou

    assert torch.isfinite(objective.loss)
    assert objective.loss.dtype == torch.float32
    assert objective.denominator.item() == pytest.approx(2.5)
    assert objective.loss.item() == pytest.approx(expected.item(), abs=1e-5)
    assert objective.state["decoded_boxes_xyxy"].shape == (1, 4)
    assert objective.state["target_boxes_xyxy"].shape == (1, 4)
    assert objective.state["smoothl1"].shape == (1,)
    assert objective.state["ciou"].shape == (1,)
    assert all(isinstance(event, MetricEvent) for event in result.metric_events)
