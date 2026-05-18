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


def test_focused_policy_makes_tiny_axis_exact_one_hot() -> None:
    tiny = _candidate("tiny", "x1", (100, 100, 110, 200))

    target = build_coord_soft_target((tiny,), _cfg())

    assert target.target_r95_radius.item() == pytest.approx(0.0)
    assert target.std.item() == pytest.approx(0.0)
    assert target.entropy.item() == pytest.approx(0.0)
    assert target.peak_prob.item() == pytest.approx(1.0)
    assert _prob_at(target, 100) == pytest.approx(1.0)
    assert target.probs.sum().item() == pytest.approx(1.0)


def test_focused_policy_fraction_changes_axis_100_target_shape() -> None:
    candidate = _candidate("box", "x1", (100, 100, 200, 240))
    frac4_cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
        gaussian_r95_axis_fraction=0.04,
        gaussian_r95_cap_bins=8,
    )
    frac6_cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
        gaussian_r95_axis_fraction=0.06,
        gaussian_r95_cap_bins=8,
    )

    frac4 = build_coord_soft_target((candidate,), frac4_cfg)
    frac6 = build_coord_soft_target((candidate,), frac6_cfg)

    assert frac4.target_r95_radius.item() == pytest.approx(4.0)
    assert frac6.target_r95_radius.item() == pytest.approx(6.0)
    assert frac6.std.item() > frac4.std.item()
    assert frac6.entropy.item() > frac4.entropy.item()
    assert frac6.peak_prob.item() < frac4.peak_prob.item()


def test_focused_policy_caps_large_axis_at_r95_eight_bins() -> None:
    large = _candidate("large", "x1", (100, 100, 900, 240))

    target = build_coord_soft_target((large,), _cfg())

    assert target.target_r95_radius.item() == pytest.approx(8.0)
    assert target.std.item() == pytest.approx(8.0 / 1.96, rel=0.03)
    assert target.std.item() < (801.0**0.5) / 3.0


def test_focused_cap8_distribution_is_sharper_than_previous_wide_span() -> None:
    medium = _candidate("medium", "x1", (100, 100, 356, 240))

    target = build_coord_soft_target((medium,), _cfg())
    previous_wide_std = ((356 - 100) + 1) ** 0.5

    assert target.target_r95_radius.item() == pytest.approx(8.0)
    assert target.std.item() < previous_wide_std / 3.0
    assert target.entropy.item() < 3.0


def test_prefix_compatibility_uses_focused_exact_radius_for_tiny_axis() -> None:
    aligned = _candidate("aligned", "x2", (100, 100, 110, 200))
    shifted = _candidate("shifted", "x2", (101, 100, 111, 200))

    target = build_coord_soft_target(
        (aligned, shifted),
        _cfg(),
        current_slot="x2",
        teacher_prefix_values={"x1": 100},
        return_components=True,
    )

    assert target.posterior["aligned"].item() == pytest.approx(1.0)
    assert target.posterior["shifted"].item() == pytest.approx(0.0)
    assert target.target_r95_radius.item() == pytest.approx(0.0)


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
        teacher_prefix_values={"x1": 3},
    )

    assert target.probs[:4].sum().item() == pytest.approx(0.0)
    assert target.support_mask[:4].sum().item() == 0
    assert target.support_mask[4].item() is True
    assert target.support_mask[5].item() is True
    assert target.probs[8].item() == pytest.approx(1.0)
    assert target.probs.sum().item() == pytest.approx(1.0)

    with pytest.raises(ValueError, match="valid token-space xyxy"):
        _candidate("bad-order", "x1", (8, 1, 3, 9))
    with pytest.raises(ValueError, match="integer"):
        _candidate("bad-type", "x1", (3.0, 1, 8, 9))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="valid token-space xyxy"):
        _candidate("bad-domain", "x1", (-1, 1, 8, 9))


def test_full_vocab_coord_soft_ce_ignores_weights_and_uses_coord_vocab_pressure() -> None:
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
    coord_vocab_manual = -(
        dist.probs
        * F.log_softmax(logits.index_select(0, dist.token_ids), dim=-1)
    ).sum()

    assert result.weighted_loss.item() == pytest.approx(coord_vocab_manual.item())
    assert differently_weighted.weighted_loss.item() == pytest.approx(
        coord_vocab_manual.item()
    )


def test_instance_trie_gaussian_softce_ignores_non_coord_logits() -> None:
    cfg = _cfg()
    candidate = _candidate("box", "x1", (100, 100, 160, 220))
    logits = torch.zeros(1200, dtype=torch.float32)
    logits[10:1010] = torch.linspace(-1.0, 1.0, steps=1000)
    baseline = full_vocab_coord_soft_ce(
        logits,
        (candidate,),
        cfg,
        current_slot="x1",
        teacher_coord_value=100,
    )

    changed = logits.clone()
    changed[1100] = 50.0
    with_non_coord_spike = full_vocab_coord_soft_ce(
        changed,
        (candidate,),
        cfg,
        current_slot="x1",
        teacher_coord_value=100,
    )

    assert with_non_coord_spike.weighted_loss.item() == pytest.approx(
        baseline.weighted_loss.item()
    )


def test_instance_trie_gaussian_support_mass_uses_structural_support_mask() -> None:
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
        gaussian_mixture_weight=0.1,
    )
    candidate = _candidate("box", "x1", (100, 100, 160, 220))
    logits = torch.zeros(1200, dtype=torch.float32)
    logits[cfg.coord_value_to_token_id(900)] = 20.0

    result = full_vocab_coord_soft_ce(
        logits,
        (candidate,),
        cfg,
        current_slot="x1",
        teacher_coord_value=100,
    )

    assert result.support_mass.item() < 0.01
    assert result.outside_support_mass.item() > 0.99


def test_full_vocab_coord_soft_ce_can_anchor_gaussian_with_exact_ce_mass() -> None:
    candidate = _candidate("box", "x1", (100, 1, 200, 9))
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
        gaussian_mixture_weight=0.2,
    )
    logits = torch.zeros((1020,), dtype=torch.float32)
    logits[110] = 2.0
    logits[111] = 1.0

    gaussian_dist = build_coord_soft_target((candidate,), _cfg())
    result = full_vocab_coord_soft_ce(
        logits,
        (candidate,),
        cfg,
        teacher_coord_value=100,
    )
    coord_log_probs = F.log_softmax(
        logits.index_select(0, gaussian_dist.token_ids),
        dim=-1,
    )
    expected_probs = gaussian_dist.probs * 0.2
    expected_probs[100] += 0.8
    expected_loss = -(expected_probs * coord_log_probs).sum()

    assert expected_probs.sum().item() == pytest.approx(1.0)
    assert result.peak_prob.item() == pytest.approx(expected_probs[100].item())
    assert result.weighted_loss.item() == pytest.approx(expected_loss.item())
    assert result.target_entropy.item() < gaussian_dist.entropy.item()


def test_full_vocab_coord_soft_ce_requires_teacher_value_for_ce_anchor() -> None:
    candidate = _candidate("box", "x1", (3, 1, 8, 9))
    cfg = CoordSoftTargetRuntimeConfig(
        target_distribution="instance_trie_gaussian",
        coord_token_start=10,
        coord_token_end=1009,
        gaussian_mixture_weight=0.2,
    )
    logits = torch.zeros((1020,), dtype=torch.float32)

    with pytest.raises(ValueError, match="teacher_coord_value"):
        full_vocab_coord_soft_ce(logits, (candidate,), cfg)
