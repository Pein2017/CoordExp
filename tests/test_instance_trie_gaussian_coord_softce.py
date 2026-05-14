from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from src.detection.coord_soft_targets import (
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    build_coord_soft_target,
    full_vocab_coord_soft_ce,
)


def _cfg() -> CoordSoftTargetRuntimeConfig:
    return CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
    )


def _candidate(
    object_id: str,
    slot_name: str,
    bbox_xyxy: tuple[int, int, int, int],
    probability: float = 1.0,
) -> CoordSoftTargetCandidate:
    return CoordSoftTargetCandidate(object_id, slot_name, bbox_xyxy, probability)


def _prob_at(target, coord_value: int) -> float:
    return float(target.probs[coord_value].item())


def test_large_bbox_gaussian_is_broader_than_tiny_bbox() -> None:
    tiny = _candidate("tiny", "x1", (100, 100, 110, 200))
    large = _candidate("large", "x1", (100, 100, 700, 200))

    tiny_target = build_coord_soft_target((tiny,), _cfg())
    large_target = build_coord_soft_target((large,), _cfg())

    assert large_target.std.item() > tiny_target.std.item()
    assert large_target.entropy.item() > tiny_target.entropy.item()
    assert large_target.peak_prob.item() < tiny_target.peak_prob.item()


def test_x1_mixture_has_separate_peaks_and_union_midpoint_valley() -> None:
    left = _candidate("left", "x1", (100, 100, 150, 200))
    right = _candidate("right", "x1", (800, 100, 850, 200))

    target = build_coord_soft_target((left, right), _cfg())

    assert _prob_at(target, 100) > _prob_at(target, 450) * 1000.0
    assert _prob_at(target, 800) > _prob_at(target, 450) * 1000.0
    assert target.probs[95:106].argmax().item() + 95 == 100
    assert target.probs[795:806].argmax().item() + 795 == 800


def test_uniform_candidate_priors_ignore_candidate_probability() -> None:
    left = _candidate("left", "x1", (100, 100, 150, 200), probability=100.0)
    right = _candidate("right", "x1", (800, 100, 850, 200), probability=1.0)

    target = build_coord_soft_target((left, right), _cfg(), return_components=True)

    assert target.posterior["left"].item() == pytest.approx(0.5)
    assert target.posterior["right"].item() == pytest.approx(0.5)
    assert _prob_at(target, 100) == pytest.approx(_prob_at(target, 800), rel=0.10)


def test_components_are_normalized_before_mixing_and_exposed() -> None:
    narrow = _candidate("narrow", "x1", (100, 100, 110, 200))
    wide = _candidate("wide", "x1", (800, 100, 950, 200))

    target = build_coord_soft_target(
        (narrow, wide),
        _cfg(),
        return_components=True,
    )

    assert set(target.component_probs_by_id) == {"narrow", "wide"}
    assert target.component_probs_by_id["narrow"].sum().item() == pytest.approx(1.0)
    assert target.component_probs_by_id["wide"].sum().item() == pytest.approx(1.0)
    assert target.posterior["narrow"].item() == pytest.approx(0.5)
    assert target.posterior["wide"].item() == pytest.approx(0.5)
    assert target.probs.sum().item() == pytest.approx(1.0)


def test_coordinate_domain_resolves_1000_bins_and_token_ids() -> None:
    cfg = _cfg()

    assert cfg.coord_bins == 1000
    assert cfg.coord_value_to_token_id(0) == 10
    assert cfg.coord_value_to_token_id(999) == 1009
    assert cfg.coord_token_ids().shape == (1000,)
    assert cfg.coord_token_ids()[0].item() == 10
    assert cfg.coord_token_ids()[-1].item() == 1009


def test_instance_trie_gaussian_runtime_rejects_stale_tau() -> None:
    with pytest.raises(ValueError, match="tau.*not supported.*instance_trie_gaussian"):
        CoordSoftTargetRuntimeConfig(
            target_distribution="instance_trie_gaussian",
            tau=0.1,
            coord_token_start=10,
            coord_token_end=1009,
        )


def test_previous_teacher_prefix_softly_downweights_incompatible_candidates() -> None:
    aligned = _candidate("aligned", "x2", (100, 100, 210, 210))
    shifted = _candidate("shifted", "x2", (120, 100, 230, 210))

    aligned_prefix = build_coord_soft_target(
        (aligned, shifted),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 100},
        return_components=True,
    )
    shifted_prefix = build_coord_soft_target(
        (aligned, shifted),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 120},
        return_components=True,
    )

    assert aligned_prefix.posterior["aligned"].item() > aligned_prefix.posterior[
        "shifted"
    ].item()
    assert shifted_prefix.posterior["shifted"].item() > shifted_prefix.posterior[
        "aligned"
    ].item()
    for object_id in ("aligned", "shifted"):
        assert torch.allclose(
            aligned_prefix.component_probs_by_id[object_id],
            shifted_prefix.component_probs_by_id[object_id],
        )
        assert torch.equal(
            aligned_prefix.component_probs_by_id[object_id] > 0,
            shifted_prefix.component_probs_by_id[object_id] > 0,
        )


def test_shared_previous_coordinate_keeps_uniform_posterior_across_bbox_sizes() -> None:
    tiny = _candidate("tiny", "x2", (100, 100, 110, 210))
    large = _candidate("large", "x2", (100, 100, 900, 210))

    target = build_coord_soft_target(
        (tiny, large),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 100},
        return_components=True,
    )

    assert target.posterior["tiny"].item() == pytest.approx(0.5)
    assert target.posterior["large"].item() == pytest.approx(0.5)


def test_near_shared_top_left_ambiguity_carries_to_x2() -> None:
    first = _candidate("first", "x2", (100, 100, 210, 220))
    second = _candidate("second", "x2", (102, 101, 820, 920))

    target = build_coord_soft_target(
        (first, second),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 101, "y1": 100},
        return_components=True,
    )

    assert target.posterior["first"].item() > 0.05
    assert target.posterior["second"].item() > 0.05
    assert _prob_at(target, 210) > _prob_at(target, 500)
    assert _prob_at(target, 820) > _prob_at(target, 500)


def test_prefix_causality_by_current_slot() -> None:
    candidates = (
        _candidate("a", "y2", (100, 100, 210, 220)),
        _candidate("b", "y2", (700, 700, 820, 920)),
    )

    x1_base = build_coord_soft_target(
        tuple(_candidate(c.object_instance_id, "x1", c.bbox_xyxy) for c in candidates),
        _cfg(),
        current_slot="x1",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 210, "y2": 220},
    )
    x1_future_changed = build_coord_soft_target(
        tuple(_candidate(c.object_instance_id, "x1", c.bbox_xyxy) for c in candidates),
        _cfg(),
        current_slot="x1",
        teacher_prefix_values={"x1": 700, "y1": 700, "x2": 820, "y2": 920},
    )
    assert torch.allclose(x1_base.probs, x1_future_changed.probs)

    y1_base = build_coord_soft_target(
        tuple(_candidate(c.object_instance_id, "y1", c.bbox_xyxy) for c in candidates),
        _cfg(),
        current_slot="y1",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 210, "y2": 220},
    )
    y1_future_changed = build_coord_soft_target(
        tuple(_candidate(c.object_instance_id, "y1", c.bbox_xyxy) for c in candidates),
        _cfg(),
        current_slot="y1",
        teacher_prefix_values={"x1": 100, "y1": 700, "x2": 820, "y2": 920},
    )
    assert torch.allclose(y1_base.probs, y1_future_changed.probs)

    x2_base = build_coord_soft_target(
        tuple(_candidate(c.object_instance_id, "x2", c.bbox_xyxy) for c in candidates),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 210, "y2": 220},
    )
    x2_future_changed = build_coord_soft_target(
        tuple(_candidate(c.object_instance_id, "x2", c.bbox_xyxy) for c in candidates),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 820, "y2": 920},
    )
    assert torch.allclose(x2_base.probs, x2_future_changed.probs)

    y2_candidates = (
        _candidate("a", "y2", (100, 100, 210, 220)),
        _candidate("b", "y2", (100, 100, 820, 920)),
    )
    y2_base = build_coord_soft_target(
        y2_candidates,
        _cfg(),
        current_slot="y2",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 210, "y2": 220},
    )
    y2_uses_x2_changed = build_coord_soft_target(
        y2_candidates,
        _cfg(),
        current_slot="y2",
        teacher_prefix_values={"x1": 100, "y1": 100, "x2": 820, "y2": 920},
    )
    assert not torch.allclose(y2_base.probs, y2_uses_x2_changed.probs)


def test_union_box_coordinate_is_not_rewarded_after_prefix_disambiguates() -> None:
    selected = _candidate("selected", "x2", (100, 100, 210, 220))
    distractor = _candidate("distractor", "x2", (700, 700, 820, 920))

    target = build_coord_soft_target(
        (selected, distractor),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 100, "y1": 100},
        return_components=True,
    )

    assert target.posterior["selected"].item() > 0.999
    assert _prob_at(target, 210) > _prob_at(target, 820) * 1000.0


def test_structural_legality_masks_invalid_bins_and_validation_errors() -> None:
    candidate = _candidate("box", "x2", (3, 1, 8, 9))

    target = build_coord_soft_target(
        (candidate,),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 5},
    )

    assert target.probs[:4].sum().item() == pytest.approx(0.0)
    assert target.support_mask[:4].sum().item() == 0
    assert target.support_mask[4].item() is True
    assert target.support_mask[5].item() is True
    assert target.probs[4].item() > 0.0
    assert target.probs[5].item() > 0.0
    assert target.probs.sum().item() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="valid token-space xyxy"):
        _candidate("bad-order", "x1", (8, 1, 3, 9))
    with pytest.raises(ValueError, match="integer"):
        _candidate("bad-type", "x1", (3.0, 1, 8, 9))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid token-space xyxy"):
        _candidate("bad-domain", "x1", (-1, 1, 8, 9))


def test_full_vocab_coord_soft_ce_ignores_weights_and_uses_full_vocab_pressure() -> None:
    candidate = _candidate("box", "x1", (3, 1, 8, 9))
    cfg = _cfg()
    logits = torch.zeros((1020,), dtype=torch.float32)
    logits[0] = 20.0
    logits[13] = 1.0

    dist = build_coord_soft_target((candidate,), cfg)
    result = full_vocab_coord_soft_ce(
        logits,
        (candidate,),
        cfg,
        support_weight=100.0,
        balance_weight=0.0,
    )
    differently_weighted = full_vocab_coord_soft_ce(
        logits,
        (candidate,),
        cfg,
        support_weight=0.0,
        balance_weight=100.0,
    )
    manual = -(
        dist.probs * F.log_softmax(logits, dim=-1).index_select(0, dist.token_ids)
    ).sum()
    coord_only_wrong = -(
        dist.probs
        * F.log_softmax(logits.index_select(0, dist.token_ids), dim=-1)
    ).sum()

    assert result.weighted_loss.item() == pytest.approx(manual.item())
    assert differently_weighted.weighted_loss.item() == pytest.approx(manual.item())
    assert result.weighted_loss.item() > coord_only_wrong.item() + 10.0
