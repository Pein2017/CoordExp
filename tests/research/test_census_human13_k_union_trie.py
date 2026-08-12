from __future__ import annotations

import copy

import pytest
import torch

from scripts.research.census_human13_k_union_trie import (
    CoherentChainSite,
    SelectedNativeRow,
    build_exact_token_trie,
    run_no_update_census,
)


def _site(
    owner_id: str,
    token_offset: int,
    target: int,
    role: str,
    packed: tuple[float, ...],
    hf: tuple[float, ...],
) -> CoherentChainSite:
    return CoherentChainSite(
        image_id="image-1",
        owner_id=owner_id,
        token_offset=token_offset,
        target_token_id=target,
        token_role=role,
        packed_logits=torch.tensor(packed),
        hf_logits=torch.tensor(hf),
    )


def test_exact_token_trie_deduplicates_rows_and_projects_one_viable_leaf() -> None:
    rows = (
        SelectedNativeRow("image-1", "owner-a", (1, 2)),
        SelectedNativeRow("image-1", "owner-a-alias", (1, 2)),
        SelectedNativeRow("image-1", "owner-b", (1, 3)),
    )

    trie = build_exact_token_trie(rows)
    result = run_no_update_census(
        selected_rows=rows,
        trie_logits={
            "image-1": {
                (): torch.tensor((0.0, 3.0, 1.0, 0.0)),
                (1,): torch.tensor((0.0, 0.0, 2.0, 1.0)),
            },
        },
        coherent_sites=(
            _site("owner-a", 0, 1, "boundary", (0, 3, 1, 0), (0, 3, 1, 0)),
            _site("owner-a", 1, 2, "row_terminator", (0, 0, 2, 1), (0, 0, 2, 1)),
            _site("owner-a-alias", 0, 1, "boundary", (0, 3, 1, 0), (0, 3, 1, 0)),
            _site("owner-a-alias", 1, 2, "row_terminator", (0, 0, 2, 1), (0, 0, 2, 1)),
            _site("owner-b", 0, 1, "boundary", (0, 3, 1, 0), (0, 3, 1, 0)),
            _site("owner-b", 1, 3, "row_terminator", (0, 0, 1, 2), (0, 0, 1, 2)),
        ),
        frozen_targets={"selected_rows": [[1, 2], [1, 2], [1, 3]]},
    )

    assert trie.original_row_count == 3
    assert trie.unique_row_count == 2
    assert trie.exact_duplicate_count == 1
    image = result["trie"]["images"][0]
    assert image["image_id"] == "image-1"
    assert image["projected_token_ids"] == [1, 2]
    assert image["projected_owner_ids"] == ["owner-a", "owner-a-alias"]
    assert image["reached_native_leaf"] is True
    assert all(node["actual_top1_is_viable_child"] for node in image["nodes"])


def test_trie_keeps_same_prefixes_and_logits_scoped_to_each_image() -> None:
    rows = (
        SelectedNativeRow("image-1", "owner-a", (1,)),
        SelectedNativeRow("image-2", "owner-b", (2,)),
    )

    result = run_no_update_census(
        selected_rows=rows,
        trie_logits={
            "image-1": {(): torch.tensor((0.0, 2.0, 0.0))},
            "image-2": {(): torch.tensor((0.0, 0.0, 3.0))},
        },
        coherent_sites=(
            CoherentChainSite(
                "image-1",
                "owner-a",
                0,
                1,
                "row_terminator",
                torch.tensor((0.0, 2.0, 0.0)),
                torch.tensor((0.0, 2.0, 0.0)),
            ),
            CoherentChainSite(
                "image-2",
                "owner-b",
                0,
                2,
                "row_terminator",
                torch.tensor((0.0, 0.0, 3.0)),
                torch.tensor((0.0, 0.0, 3.0)),
            ),
        ),
        frozen_targets={"selected_rows": {"image-1": [[1]], "image-2": [[2]]}},
    )

    assert result["trie"]["unique_row_count"] == 2
    assert [image["projected_token_ids"] for image in result["trie"]["images"]] == [
        [1],
        [2],
    ]
    assert [image["projected_owner_ids"] for image in result["trie"]["images"]] == [
        ["owner-a"],
        ["owner-b"],
    ]


def test_trie_scores_every_frozen_branch_before_projecting_greedy_row() -> None:
    rows = (
        SelectedNativeRow("image-1", "owner-a", (1, 3)),
        SelectedNativeRow("image-1", "owner-b", (2, 4)),
    )
    sites = (
        _site("owner-a", 0, 1, "boundary", (0, 3, 1, 0, 0), (0, 3, 1, 0, 0)),
        _site(
            "owner-a",
            1,
            3,
            "row_terminator",
            (0, 0, 0, 3, 0),
            (0, 0, 0, 3, 0),
        ),
        _site("owner-b", 0, 2, "boundary", (0, 1, 3, 0, 0), (0, 1, 3, 0, 0)),
        _site(
            "owner-b",
            1,
            4,
            "row_terminator",
            (0, 0, 0, 0, 3),
            (0, 0, 0, 0, 3),
        ),
    )
    root_and_projected_only = {
        (): torch.tensor((0.0, 3.0, 2.0, 0.0, 0.0)),
        (1,): torch.tensor((0.0, 0.0, 0.0, 3.0, 0.0)),
    }

    with pytest.raises(ValueError, match="every frozen trie prefix"):
        run_no_update_census(
            selected_rows=rows,
            trie_logits={"image-1": root_and_projected_only},
            coherent_sites=sites,
            frozen_targets={"selected_rows": [[1, 3], [2, 4]]},
        )

    result = run_no_update_census(
        selected_rows=rows,
        trie_logits={
            "image-1": {
                **root_and_projected_only,
                (2,): torch.tensor((0.0, 0.0, 0.0, 0.0, 4.0)),
            }
        },
        coherent_sites=sites,
        frozen_targets={"selected_rows": [[1, 3], [2, 4]]},
    )

    image = result["trie"]["images"][0]
    assert [node["prefix_token_ids"] for node in image["nodes"]] == [[], [1], [2]]
    assert image["projected_token_ids"] == [1, 3]
    assert image["projected_owner_ids"] == ["owner-a"]


def test_census_traverses_full_chain_and_seals_margin_ties_roles_and_drift() -> None:
    rows = (
        SelectedNativeRow("image-1", "owner-a", (1, 2, 3)),
        SelectedNativeRow("image-1", "owner-b", (2, 1)),
    )
    sites = (
        _site("owner-a", 0, 1, "boundary", (0, 2, 1, 0), (0, 1.9, 1, 0)),
        _site("owner-a", 1, 2, "schema", (0, 1, 1, 0), (0, 0.8, 1, 0)),
        _site("owner-a", 2, 3, "description", (0.2, 0, 0, 0), (0.0, 0, 0, 0.2)),
        _site("owner-b", 0, 2, "coordinate", (0, 0, 0.3, 0), (0, 0, 0.2, 0)),
        _site(
            "owner-b",
            1,
            1,
            "row_terminator",
            (0, -1, -0.2, -0.3),
            (0, -0.6, -0.2, -0.3),
        ),
    )
    frozen_targets = {
        "G": ["source-owner"],
        "H": ["owner-a", "owner-b"],
        "M": ["unknown-owner"],
        "selected_rows": [[1, 2, 3], [2, 1]],
        "residual_order": ["owner-a", "owner-b"],
        "arm_weights": {"A8-prime": 1.0},
    }
    original = copy.deepcopy(frozen_targets)

    result = run_no_update_census(
        selected_rows=rows,
        trie_logits={
            "image-1": {
                (): torch.tensor((0.0, 2.0, 1.0, 0.0)),
                (1,): torch.tensor((0, 0, 2, 1)),
                (1, 2): torch.tensor((0, 0, 0, 2)),
                (2,): torch.tensor((0, 2, 0, 0)),
            }
        },
        coherent_sites=sites,
        frozen_targets=frozen_targets,
    )

    assert [site["target_token_id"] for site in result["coherent_chain"]["sites"]] == [
        1,
        2,
        3,
        2,
        1,
    ]
    assert result["coherent_chain"]["first_non_argmax_site"]["site_index"] == 1
    assert (
        result["coherent_chain"]["first_non_argmax_site"]["packed_top_tie_count"] == 2
    )
    assert result["coherent_chain"]["minimum_strict_margin"] == pytest.approx(-1.0)
    assert result["coherent_chain"]["tie_site_count"] == 1
    assert result["coherent_chain"]["token_role_counts"] == {
        "boundary": 1,
        "schema": 1,
        "description": 1,
        "coordinate": 1,
        "row_terminator": 1,
    }
    assert result["aligned_surface"]["maximum_absolute_margin_drift"] == pytest.approx(
        0.4
    )
    assert result["a8_prime"] == {
        "applicable": True,
        "blocked": False,
        "block_reason": None,
        "required_margin": pytest.approx(0.4001),
        "violating_site_count": 4,
    }
    assert result["frozen_targets"]["byte_identical"] is True
    assert (
        result["frozen_targets"]["sha256_before"]
        == result["frozen_targets"]["sha256_after"]
    )
    assert frozen_targets == original


@pytest.mark.parametrize(
    ("packed", "hf", "reason"),
    (
        ((0.0, 1.0), (0.0, 1.0), "no_margin_violations"),
        ((0.0, 1.0), (0.0, 0.4), "required_margin_exceeds_0_5"),
        ((0.0, 1.0), (0.0, float("nan")), "aligned_finite_scores_unavailable"),
    ),
)
def test_census_blocks_a8_prime_for_noop_large_drift_or_nonfinite_alignment(
    packed: tuple[float, ...],
    hf: tuple[float, ...],
    reason: str,
) -> None:
    row = SelectedNativeRow("image-1", "owner-a", (1,))
    result = run_no_update_census(
        selected_rows=(row,),
        trie_logits={"image-1": {(): torch.tensor((0.0, 1.0))}},
        coherent_sites=(_site("owner-a", 0, 1, "row_terminator", packed, hf),),
        frozen_targets={"selected_rows": [[1]]},
    )

    assert result["a8_prime"]["blocked"] is True
    assert result["a8_prime"]["applicable"] is False
    assert result["a8_prime"]["block_reason"] == reason


def test_nonfinite_hf_preserves_finite_packed_chain_diagnostics() -> None:
    row = SelectedNativeRow("image-1", "owner-a", (1,))

    result = run_no_update_census(
        selected_rows=(row,),
        trie_logits={"image-1": {(): torch.tensor((0.0, 2.0, 0.0))}},
        coherent_sites=(
            _site(
                "owner-a",
                0,
                1,
                "row_terminator",
                (1.0, 1.0, 0.0),
                (0.0, float("nan"), 0.0),
            ),
        ),
        frozen_targets={"selected_rows": [[1]]},
    )

    chain = result["coherent_chain"]
    site = chain["sites"][0]
    assert site["aligned_finite"] is False
    assert site["packed_competitor_token_id"] == 0
    assert site["packed_target_margin"] == 0.0
    assert site["packed_top_tie_count"] == 2
    assert chain["first_non_argmax_site"]["site_index"] == 0
    assert chain["minimum_strict_margin"] == 0.0
    assert chain["tie_site_count"] == 1
    assert result["aligned_surface"] == {
        "all_finite": False,
        "maximum_absolute_margin_drift": None,
        "site_count": 0,
    }
    assert result["a8_prime"]["block_reason"] == "aligned_finite_scores_unavailable"
