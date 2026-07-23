"""Model-free contracts for the matched Source-route assembler."""

from __future__ import annotations

from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts/research/assemble_source_preservation_multi_route_state_banks.py"
)
SPEC = importlib.util.spec_from_file_location("source_preservation_multi_route", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _row(index: int, owner: str | None, *, iou: float = 0.9) -> dict[str, object]:
    return {
        "generated_row_index": index,
        "owner_id": owner,
        "entity_status": "verified_owner" if owner is not None else "unresolved_pending_crop_review",
        "category": "person",
        "intersection_over_union": iou,
        "raw": {"coord_bins": [10, 20, 30, 40], "raw_span_text": "exact"},
    }


def _assignment(rows: list[dict[str, object]], *, unresolved: int = 0) -> dict[str, object]:
    return {
        "row_assignment_receipts": rows,
        "row_counts": {"duplicate": 0, "malformed": 0, "unresolved": unresolved},
    }


def _route_image() -> dict[str, object]:
    return {
        "image_id": "42",
        "greedy_trajectory_id": "greedy",
        "sampled_trajectory_ids": ["seed-7", "seed-8", "seed-9"],
        "trajectory_evidence": {
            "greedy": {"seed": 0, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-7": {"seed": 7, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-8": {"seed": 8, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
            "seed-9": {"seed": 9, "stop_reason": "im_end", "parser": {"parse_status": "accepted"}},
        },
        "budgets": [
            {
                "budget": 16,
                "owner_sets": {
                    "greedy": ["42:a"],
                    "seed-7": ["42:a", "42:b", "42:c"],
                    "seed-8": ["42:a", "42:c", "42:d"],
                    "seed-9": ["42:a", "42:b"],
                },
                "trajectory_assignments": {
                    "greedy": _assignment([_row(0, "42:a")]),
                    "seed-7": _assignment([_row(0, "42:a"), _row(1, "42:b"), _row(2, "42:c")]),
                    "seed-8": _assignment([_row(0, "42:a"), _row(1, "42:c"), _row(2, "42:d")]),
                    "seed-9": _assignment([_row(0, "42:a"), _row(1, "42:b")]),
                },
            }
        ],
    }


def test_multi_route_selection_is_deterministic_and_marginal() -> None:
    first = MODULE.select_complementary_routes(_route_image())
    second = MODULE.select_multi_route_routes(_route_image())
    assert first == second
    assert first["selected_route_ids"] == ["seed-7", "seed-8"]
    assert first["selected_added_owner_union"] == ["42:b", "42:c", "42:d"]
    assert first["selected_routes"][0]["last_marginal_owner_row_index"] == 2
    assert first["selected_routes"][1]["marginal_added_owner_ids"] == ["42:d"]


def test_exact_integer_slices_are_preserved_without_retokenization() -> None:
    row_a = [151646, 987654, 151647, 151648, 151670, 151671, 151672, 151673, 151649]
    row_b = [151646, 123, 151647, 151648, 151700, 151800, 151900, 152000, 151649]
    prefix, row = MODULE.exact_row_slices(row_a + row_b, 1)
    assert prefix == row_a
    assert row == row_b
    assert hashlib.sha256(bytes(str(row), "utf-8")).hexdigest() == hashlib.sha256(bytes(str(row_b), "utf-8")).hexdigest()


def test_source_first_owner_rows_are_trusted_and_deduplicated() -> None:
    owners = [
        {"owner_id": "42:a", "category": "person"},
        {"owner_id": "42:b", "category": "person"},
    ]
    rows = MODULE.select_source_anchor_rows(
        _assignment([_row(0, "42:a"), _row(1, "42:a"), _row(2, "42:b"), _row(3, None)]),
        owners,
    )
    assert [item["owner_id"] for item in rows] == ["42:a", "42:b"]
    assert [item["generated_row_index"] for item in rows] == [0, 2]


def test_identical_source_anchor_candidates_have_identical_hashes_across_arms() -> None:
    source = [
        {
            "image_id": "1",
            "prefix_token_ids_sha256": "p1",
            "candidate_token_ids_sha256": "r1",
            "owner_id": "1:a",
            "event_id": "source-a",
        },
        {
            "image_id": "2",
            "prefix_token_ids_sha256": "p2",
            "candidate_token_ids_sha256": "r2",
            "owner_id": "2:a",
            "event_id": "source-b",
        },
    ]
    arm_a = MODULE.deduplicate_route_events(source)
    arm_b = MODULE.deduplicate_route_events(source)
    hashes_a = {(item["image_id"], item["prefix_token_ids_sha256"], item["candidate_token_ids_sha256"], item["owner_id"]) for item in arm_a}
    hashes_b = {(item["image_id"], item["prefix_token_ids_sha256"], item["candidate_token_ids_sha256"], item["owner_id"]) for item in arm_b}
    assert hashes_a == hashes_b


def test_sampled_source_collisions_are_excluded_before_fixed_selection() -> None:
    source_identity = ("1", "prefix-a", "row-a", "1:a")
    candidates = {
        "1": [
            {
                "image_id": "1",
                "prefix_token_ids_sha256": "prefix-a",
                "candidate_token_ids_sha256": "row-a",
                "owner_id": "1:a",
                "route_id": "seed-1",
                "generated_row_index": 0,
            },
            {
                "image_id": "1",
                "prefix_token_ids_sha256": "prefix-b",
                "candidate_token_ids_sha256": "row-b",
                "owner_id": "1:b",
                "route_id": "seed-1",
                "generated_row_index": 1,
            },
        ]
    }
    filtered, receipt = MODULE.exclude_sampled_source_collisions(candidates, {source_identity})
    assert len(filtered["1"]) == 1
    assert filtered["1"][0]["owner_id"] == "1:b"
    assert receipt["collision_event_count"] == 1
    with pytest.raises(ValueError, match="cannot select exactly"):
        MODULE.select_image_diverse_events(filtered, image_ids=["1"], event_count=2)


def _semantic_event(
    image_id: str, prefix_hash: str, row_hash: str, owner_id: str
) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {
            "image": {"image_id": image_id},
            "prefix_token_ids_sha256": prefix_hash,
            "candidates": [{"token_ids_sha256": row_hash}],
        },
        {"candidates": [{"physical_owner_id": owner_id}]},
    )


def test_final_combined_identity_validation_rejects_cross_family_clone() -> None:
    treatment_rollout, treatment_review = _semantic_event("1", "p", "r", "1:a")
    source_rollout, source_review = _semantic_event("1", "p", "r", "1:a")
    with pytest.raises(ValueError, match="cross_family_collision_count=1"):
        MODULE._validate_route_event_identity_uniqueness(
            treatment_rollouts=[treatment_rollout],
            treatment_reviews=[treatment_review],
            source_rollouts=[source_rollout],
            source_reviews=[source_review],
        )


def test_final_combined_identity_validation_receipts_unique_events() -> None:
    treatment_rollout, treatment_review = _semantic_event("1", "p", "r", "1:a")
    source_rollout, source_review = _semantic_event("1", "p", "r2", "1:a")
    receipt = MODULE._validate_route_event_identity_uniqueness(
        treatment_rollouts=[treatment_rollout],
        treatment_reviews=[treatment_review],
        source_rollouts=[source_rollout],
        source_reviews=[source_review],
    )
    assert receipt["combined_unique_identity_count"] == 2
    assert receipt["combined_duplicate_identity_count"] == 0
    assert receipt["combined_identity_unique"] is True


def test_single_route_purity_rejects_second_sampled_route() -> None:
    reviews = [
        {
            "event_id": "single-route-treatment-image-1-route-seed-1-row-0",
            "review_provenance": {"route_id": "seed-1"},
        },
        {
            "event_id": "single-route-treatment-image-1-route-seed-2-row-0",
            "review_provenance": {"route_id": "seed-2"},
        },
    ]
    with pytest.raises(ValueError, match="multiple sampled routes"):
        MODULE._validate_single_route_purity(reviews, expected_event_count=2)


def _checkpoint_fields(*, adapter: str = "a") -> dict[str, str]:
    return {
        "adapter_fingerprint": adapter * 64,
        "embedding_delta_fingerprint": "b" * 64,
        "base_config_sha256": "c" * 64,
        "tokenizer_sha256": "d" * 64,
        "token_identity_sha256": "e" * 64,
        "special_token_identity_sha256": "f" * 64,
        "processor_identity_sha256": "0" * 64,
    }


def test_explicit_checkpoint_identity_cannot_replace_runtime_proof() -> None:
    with pytest.raises(ValueError, match="lacks model_identity checkpoint evidence"):
        MODULE.derive_checkpoint_identity_from_rollout_artifact(
            {"checkpoint_identity": _checkpoint_fields()}
        )


def test_explicit_checkpoint_identity_must_match_runtime_proof(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MODULE.CheckpointIdentity.from_mapping(_checkpoint_fields())
    monkeypatch.setattr(
        MODULE,
        "_derive_runtime_checkpoint_identity_from_rollout_artifact",
        lambda payload, artifact_path=None: runtime,
    )
    with pytest.raises(ValueError, match="contradicts independently derived runtime identity"):
        MODULE.derive_checkpoint_identity_from_rollout_artifact(
            {
                "model_identity": {"runtime": "present"},
                "checkpoint_identity": _checkpoint_fields(adapter="1"),
            }
        )


def test_matching_explicit_checkpoint_identity_is_only_a_cross_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = MODULE.CheckpointIdentity.from_mapping(_checkpoint_fields())
    monkeypatch.setattr(
        MODULE,
        "_derive_runtime_checkpoint_identity_from_rollout_artifact",
        lambda payload, artifact_path=None: runtime,
    )
    observed = MODULE.derive_checkpoint_identity_from_rollout_artifact(
        {
            "model_identity": {"runtime": "present"},
            "checkpoint_identity": runtime.to_artifact_dict(),
        }
    )
    assert observed == runtime


def test_runtime_checkpoint_identity_mismatch_is_rejected_against_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = MODULE.CheckpointIdentity.from_mapping(_checkpoint_fields())
    mismatch = MODULE.CheckpointIdentity.from_mapping(_checkpoint_fields(adapter="1"))
    artifact = tmp_path / "greedy.json"
    artifact.write_text(json.dumps({"model_identity": {"runtime": "present"}}) + "\n", encoding="utf-8")
    monkeypatch.setattr(
        MODULE,
        "_derive_runtime_checkpoint_identity_from_rollout_artifact",
        lambda payload, artifact_path=None: mismatch,
    )
    with pytest.raises(ValueError, match="differs from reference binding"):
        MODULE.verify_rollout_checkpoint_identities(
            [artifact], reference_checkpoint=reference
        )


def test_single_route_purity_requires_complete_provenance() -> None:
    with pytest.raises(ValueError, match="lacks review_provenance"):
        MODULE._validate_single_route_purity(
            [{"event_id": "single-route-treatment-image-1-route-seed-1-row-0"}],
            expected_event_count=1,
        )
    with pytest.raises(ValueError, match="lacks non-empty route_id"):
        MODULE._validate_single_route_purity(
            [
                {
                    "event_id": "single-route-treatment-image-1-route-seed-1-row-0",
                    "review_provenance": {"route_id": ""},
                }
            ],
            expected_event_count=1,
        )


def test_image_diverse_selector_reaches_496_and_keeps_every_image() -> None:
    image_ids = [str(index) for index in range(1, 119)]
    pools = {
        image: [
            {
                "image_id": image,
                "owner_id": f"{image}:owner-{offset}",
                "prefix_token_ids_sha256": f"{image}-prefix-{offset}",
                "candidate_token_ids_sha256": f"{image}-row-{offset}",
                "generated_row_index": offset,
                "event_id": f"{image}-{offset}",
                "marginal_owner_count": 1,
            }
            for offset in range(5)
        ]
        for image in image_ids
    }
    selected, receipt = MODULE.select_image_diverse_events(pools, image_ids=image_ids, event_count=496)
    assert len(selected) == 496
    assert receipt["image_count"] == 118
    assert {item["image_id"] for item in selected} == set(image_ids)


def test_family_weights_are_equal_per_image_and_mean_one() -> None:
    image_ids = [str(index) for index in range(1, 119)]
    counts = {image: {"source_preservation": 4, "treatment": 4} for image in image_ids}
    counts["1"] = {"source_preservation": 20, "treatment": 20}
    counts["2"] = {"source_preservation": 12, "treatment": 12}
    weights = MODULE.image_family_event_weights(counts, event_count=992)
    assert weights[("1", "source_preservation")] == weights[("1", "treatment")]
    assert weights[("1", "source_preservation")] != weights[("3", "source_preservation")]
    total = sum(weights[(image, family)] * counts[image][family] for image in image_ids for family in ("source_preservation", "treatment"))
    assert total / 992 == pytest.approx(1.0)
    image_totals = [sum(weights[(image, family)] * counts[image][family] for family in ("source_preservation", "treatment")) for image in image_ids]
    assert len(set(round(value, 12) for value in image_totals)) == 1


def test_blind_image_rejection_is_fail_fast() -> None:
    with pytest.raises(ValueError, match="blind"):
        MODULE.reject_blind_images(["1584", "42"])


def test_output_refusal_is_immutable_and_rerun_safe(tmp_path: Path) -> None:
    output = tmp_path / "receipt.json"
    MODULE._write_json(output, {"status": "first"})
    with pytest.raises(ValueError, match="already exists"):
        MODULE._write_json(output, {"status": "second"})


def _breadth_candidate(
    image_id: str, family: str, rank: int, *, geometry_trusted: bool = True
) -> dict[str, object]:
    return {
        "event_id": f"{family}-{image_id}-{rank}",
        "image_id": image_id,
        "route_id": f"{family}-route-{rank}",
        "prefix_token_ids_sha256": f"prefix-{family}-{image_id}-{rank}",
        "candidate_token_ids_sha256": f"row-{family}-{image_id}-{rank}",
        "owner_id": f"{image_id}:{family}:{rank}",
        "marginal_route_added_owner_count": 3 - (rank % 2),
        "route_added_owner_count": 4,
        "unresolved_row_count": rank % 2,
        "route_seed": 100 + rank,
        "trusted_source_anchor_order": rank + 1,
        "route_count": 2 if family == "sampled" else 1,
        "generated_row_index": rank,
        "geometry": {"geometry_trusted": geometry_trusted},
    }


def _breadth_inputs() -> tuple[dict[str, list[dict[str, object]]], dict[str, list[dict[str, object]]], dict[str, str]]:
    bands = (
        "sparse_1_to_3",
        "medium_4_to_7",
        "dense_8_to_15",
        "very_dense_16_plus",
    )
    sampled: dict[str, list[dict[str, object]]] = {}
    source: dict[str, list[dict[str, object]]] = {}
    image_bands: dict[str, str] = {}
    for index in range(496):
        image = str(10_000 + index)
        image_bands[image] = bands[index // 124]
        sampled[image] = [_breadth_candidate(image, "sampled", rank) for rank in range(5)]
        source[image] = [_breadth_candidate(image, "source", rank, geometry_trusted=rank % 2 == 0) for rank in range(5)]
    return sampled, source, image_bands


def _breadth_inputs_with_rank_feasibility(
    mode: str,
) -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
    dict[str, str],
]:
    """Create exact, coarse-only, or policy-only interval-capacity panels."""

    sampled, source, image_bands = _breadth_inputs()
    if mode == "exact":
        return sampled, source, image_bands
    if mode not in {"coarse", "policy_only"}:
        raise ValueError(mode)
    for band in MODULE._BREADTH_BANDS:
        images = sorted(
            (image for image, observed in image_bands.items() if observed == band),
            key=MODULE._breadth_image_order,
        )
        concentrated_count = MODULE._breadth_quota(118)[band]
        concentrated = set(images[:concentrated_count])
        high_capacity = set(images[:10])
        for image in images:
            if image in high_capacity:
                capacity = 20
            elif image in concentrated:
                capacity = 1
            else:
                capacity = 4 if mode == "coarse" else 1
            sampled[image] = [
                _breadth_candidate(image, "sampled", rank) for rank in range(capacity)
            ]
            source[image] = [
                _breadth_candidate(
                    image,
                    "source",
                    rank,
                    geometry_trusted=rank % 2 == 0,
                )
                for rank in range(capacity)
            ]
    return sampled, source, image_bands


def _batch1_request_major_execution_metadata() -> dict[str, object]:
    return {
        "physical_batch_size": 1,
        "sampling_order": "request_major",
        "rng_reset": "per_image_seed",
    }


def _v2_execution_metadata() -> dict[str, object]:
    return {
        "panel_schema_version": "coordexp_vllm_trajectory_panel.v2",
        "sampled_panel_mode": "sampled_only",
        "source_panel_mode": "source_b16",
        "sampling_order": "request_major",
        "sample_index_range": [0, 15],
        "sample_count": 16,
        "source_b16_row_budget": 16,
        "execution_model_identity_sha256": "a" * 64,
    }


def _v2_capacity_constrained_breadth_inputs() -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
    dict[str, str],
]:
    eligible_counts = {
        "sparse_1_to_3": 33,
        "medium_4_to_7": 158,
        "dense_8_to_15": 294,
        "very_dense_16_plus": 312,
    }
    broad_quotas = {
        "sparse_1_to_3": 33,
        "medium_4_to_7": 155,
        "dense_8_to_15": 154,
        "very_dense_16_plus": 154,
    }
    prefix_capacities = {
        "sparse_1_to_3": [2] * 6 + [1] * 19 + [2],
        "medium_4_to_7": [3] * 12 + [2] * 59 + [3],
        "dense_8_to_15": [5] * 13 + [4] * 22 + [3],
        "very_dense_16_plus": [7] * 9 + [6] * 15 + [10],
    }
    sampled: dict[str, list[dict[str, object]]] = {}
    source: dict[str, list[dict[str, object]]] = {}
    image_bands: dict[str, str] = {}
    next_image = 20_000
    for band in MODULE._BREADTH_BANDS:
        images = [str(next_image + index) for index in range(eligible_counts[band])]
        next_image += eligible_counts[band]
        ordered = sorted(images, key=MODULE._breadth_image_order)
        capacities = [
            *prefix_capacities[band],
            *([16] * (broad_quotas[band] - len(prefix_capacities[band]))),
            *([1] * (len(images) - broad_quotas[band])),
        ]
        for image, capacity in zip(ordered, capacities, strict=True):
            image_bands[image] = band
            sampled[image] = [
                _breadth_candidate(image, "sampled", rank) for rank in range(capacity)
            ]
            source[image] = [
                _breadth_candidate(image, "source", rank) for rank in range(capacity)
            ]
            for candidate in [*sampled[image], *source[image]]:
                candidate.pop("event_id")
    return sampled, source, image_bands


def test_capped_max_min_quota_caps_scarce_band_and_uses_canonical_remainder() -> None:
    eligible = {
        "sparse_1_to_3": 33,
        "medium_4_to_7": 158,
        "dense_8_to_15": 294,
        "very_dense_16_plus": 312,
    }
    assert MODULE._breadth_capped_max_min_quota(eligible, total=496) == {
        "sparse_1_to_3": 33,
        "medium_4_to_7": 155,
        "dense_8_to_15": 154,
        "very_dense_16_plus": 154,
    }
    with pytest.raises(ValueError, match="exact canonical band set"):
        MODULE._breadth_capped_max_min_quota(
            {key: value for key, value in eligible.items() if key != "sparse_1_to_3"},
            total=496,
        )
    with pytest.raises(ValueError, match="lacks total eligible supply"):
        MODULE._breadth_capped_max_min_quota(
            {band: 1 for band in MODULE._BREADTH_BANDS}, total=496
        )


def test_minimum_capacity_prefix_is_hash_ordered_and_never_skips() -> None:
    images = ["41", "42", "43", "44"]
    ordered = sorted(images, key=MODULE._breadth_image_order)
    capacities = [1, 1, 5, 100]
    pairs = {
        image: [{"pair": index} for index in range(capacity)]
        for image, capacity in zip(ordered, capacities, strict=True)
    }
    selected, receipt = MODULE._breadth_minimum_capacity_prefix(
        images=list(reversed(images)), pairs_by_image=pairs, required_pair_count=6
    )
    assert selected == ordered[:3]
    assert receipt["prefix_image_count"] == 3
    assert receipt["raw_distinct_pair_capacity_at_prefix_minus_one"] == 2
    assert receipt["raw_distinct_pair_capacity_at_prefix"] == 7
    assert receipt["usable_pair_capacity_at_prefix_minus_one"] == 2
    assert receipt["usable_pair_capacity_at_prefix"] == 7
    with pytest.raises(ValueError, match="lacks usable trusted pair supply"):
        MODULE._breadth_minimum_capacity_prefix(
            images=images, pairs_by_image=pairs, required_pair_count=108
        )


def test_minimum_capacity_prefix_caps_one_high_capacity_image() -> None:
    images = ["51", "52"]
    ordered = sorted(images, key=MODULE._breadth_image_order)
    pairs = {
        ordered[0]: [{"pair": index} for index in range(12)],
        ordered[1]: [{"pair": index} for index in range(3)],
    }
    selected, receipt = MODULE._breadth_minimum_capacity_prefix(
        images=images,
        pairs_by_image=pairs,
        required_pair_count=10,
        max_pairs_per_image=8,
    )
    assert selected == ordered
    assert receipt["raw_distinct_pair_capacity_at_prefix_minus_one"] == 12
    assert receipt["usable_pair_capacity_at_prefix_minus_one"] == 8
    assert receipt["raw_distinct_pair_capacity_at_prefix"] == 15
    assert receipt["usable_pair_capacity_at_prefix"] == 11


def test_v2_capacity_constrained_protocol_freezes_exact_prefix_proofs() -> None:
    sampled, source, bands = _v2_capacity_constrained_breadth_inputs()
    result = MODULE.select_constant_dose_breadth_arms(
        sampled_candidates=sampled,
        source_candidates=source,
        training_image_bands=bands,
        trajectory_panel_execution_metadata=_v2_execution_metadata(),
    )
    assert result["eligible_images_by_band"] == {
        "sparse_1_to_3": 33,
        "medium_4_to_7": 158,
        "dense_8_to_15": 294,
        "very_dense_16_plus": 312,
    }
    assert result["broad_band_quota"] == result["pair_quota_by_object_count_band"] == {
        "sparse_1_to_3": 33,
        "medium_4_to_7": 155,
        "dense_8_to_15": 154,
        "very_dense_16_plus": 154,
    }
    assert result["concentrated_band_quota"] == {
        "sparse_1_to_3": 26,
        "medium_4_to_7": 72,
        "dense_8_to_15": 36,
        "very_dense_16_plus": 25,
    }
    assert len(result["broad"]["image_ids"]) == 496
    assert len(result["concentrated"]["image_ids"]) == 159
    assert set(result["concentrated"]["image_ids"]) < set(result["broad"]["image_ids"])
    protocol = result["allocation_protocol_receipt"]
    assert protocol["protocol_amendment_name"] == MODULE.V2_BREADTH_PROTOCOL_AMENDMENT
    assert protocol["concentrated_image_count"] == 159
    assert protocol["breadth_ratio_pair_count_per_image_by_band"] == {
        "sparse_1_to_3": 33 / 26,
        "medium_4_to_7": 155 / 72,
        "dense_8_to_15": 154 / 36,
        "very_dense_16_plus": 154 / 25,
    }
    assert protocol["overall_breadth_ratio_pair_count_per_image"] == 496 / 159
    proofs = protocol["minimum_prefix_capacity_proof_by_band"]
    assert {
        band: (
            proof["prefix_image_count"],
            proof["usable_pair_capacity_at_prefix_minus_one"],
            proof["usable_pair_capacity_at_prefix"],
        )
        for band, proof in proofs.items()
    } == {
        "sparse_1_to_3": (26, 31, 33),
        "medium_4_to_7": (72, 154, 157),
        "dense_8_to_15": (36, 153, 156),
        "very_dense_16_plus": (25, 153, 161),
    }
    assert protocol["max_pairs_per_concentrated_image"] == 8
    assert result["rank_matching_receipt"]["matching_mode"] == (
        "exact_band_by_selection_rank"
    )
    for arm_name in ("broad", "concentrated"):
        arm = result[arm_name]
        assert arm["event_count"] == 992
        identities = [MODULE._breadth_identity(event) for event in arm["events"]]
        assert len(identities) == len(set(identities)) == 992
    concentrated_treatment_counts = Counter(
        str(event["image_id"])
        for event in result["concentrated"]["events"]
        if event["event_family"] == "treatment"
    )
    assert max(concentrated_treatment_counts.values()) <= 8


def test_constant_dose_breadth_selection_is_nested_matched_and_weighted() -> None:
    sampled, source, bands = _breadth_inputs_with_rank_feasibility("exact")
    result = MODULE.select_constant_dose_breadth_arms(
        sampled_candidates=sampled,
        source_candidates=source,
        training_image_bands=bands,
        trajectory_panel_execution_metadata=_batch1_request_major_execution_metadata(),
    )
    repeated = MODULE.select_constant_dose_breadth_arms(
        sampled_candidates=sampled,
        source_candidates=source,
        training_image_bands=bands,
        trajectory_panel_execution_metadata=_batch1_request_major_execution_metadata(),
    )
    broad = result["broad"]
    concentrated = result["concentrated"]
    assert "allocation_protocol_receipt" not in result
    assert result["eligible_unique_training_image_count"] == 496
    assert broad["event_count"] == concentrated["event_count"] == 992
    assert broad["sampled_event_count"] == concentrated["sampled_event_count"] == 496
    assert broad["source_event_count"] == concentrated["source_event_count"] == 496
    assert broad["mean_event_weight"] == pytest.approx(1.0)
    assert concentrated["mean_event_weight"] == pytest.approx(1.0)
    assert broad["total_event_weight"] == pytest.approx(992.0)
    assert concentrated["total_event_weight"] == pytest.approx(992.0)
    assert len(broad["image_ids"]) == 496
    assert len(concentrated["image_ids"]) == 118
    assert set(concentrated["image_ids"]) < set(broad["image_ids"])
    assert set(broad["selection_distributions"]) == {
        "selection_rank",
        "route_count",
        "row_depth",
        "complete_row_coordinate_token_supervision",
    }
    assert set(event["image_id"] for event in broad["events"]) == set(broad["image_ids"])
    assert [event["event_id"] for event in broad["events"]] == [
        event["event_id"] for event in repeated["broad"]["events"]
    ]
    receipt = result["rank_matching_receipt"]
    assert receipt["matching_mode"] == "exact_band_by_selection_rank"
    assert receipt["exact_rank_feasibility"]["feasible"] is True
    assert receipt["coarse_rank_feasibility"]["attempted"] is False
    assert (
        receipt["broad_exact_rank_histogram_by_band"]
        == receipt["concentrated_exact_rank_histogram_by_band"]
    )
    assert all(
        sum(histogram.values()) == 124
        for histogram in receipt["broad_exact_rank_histogram_by_band"].values()
    )
    identities = [
        (
            event["image_id"],
            event["prefix_token_ids_sha256"],
            event["candidate_token_ids_sha256"],
            event["owner_id"],
        )
        for event in broad["events"]
    ]
    assert len(identities) == len(set(identities)) == 992


def test_constant_dose_breadth_coarsens_only_to_declared_rank_groups() -> None:
    sampled, source, bands = _breadth_inputs_with_rank_feasibility("coarse")
    result = MODULE.select_constant_dose_breadth_arms(
        sampled_candidates=sampled,
        source_candidates=source,
        training_image_bands=bands,
        trajectory_panel_execution_metadata=_batch1_request_major_execution_metadata(),
    )
    receipt = result["rank_matching_receipt"]
    assert receipt["matching_mode"] == "coarse_band_by_rank_1_2_3_4_plus"
    assert receipt["exact_rank_feasibility"]["feasible"] is False
    assert receipt["coarse_rank_feasibility"]["attempted"] is True
    assert receipt["coarse_rank_feasibility"]["feasible"] is True
    assert (
        receipt["broad_coarse_rank_histogram_by_band"]
        == receipt["concentrated_coarse_rank_histogram_by_band"]
    )
    assert any(
        check["passed"] is False
        for band in receipt["exact_rank_feasibility"]["bands"].values()
        for check in band["threshold_checks"]
    )


def test_constant_dose_breadth_falls_back_to_explicit_policy_only_scope() -> None:
    sampled, source, bands = _breadth_inputs_with_rank_feasibility("policy_only")
    result = MODULE.select_constant_dose_breadth_arms(
        sampled_candidates=sampled,
        source_candidates=source,
        training_image_bands=bands,
        trajectory_panel_execution_metadata=_batch1_request_major_execution_metadata(),
    )
    receipt = result["rank_matching_receipt"]
    assert receipt["matching_mode"] == "policy_only_broad_rank_one"
    assert receipt["interpretation_scope"] == (
        "data_allocation_policy_comparison_only_not_an_image_breadth_alone_claim"
    )
    assert receipt["exact_rank_feasibility"]["feasible"] is False
    assert receipt["coarse_rank_feasibility"]["attempted"] is True
    assert receipt["coarse_rank_feasibility"]["feasible"] is False
    assert all(event["selection_rank"] == 1 for event in result["broad"]["events"])
    assert result["broad"]["event_count"] == result["concentrated"]["event_count"] == 992
    assert result["broad"]["total_event_weight"] == pytest.approx(992.0)
    assert result["concentrated"]["total_event_weight"] == pytest.approx(992.0)


def test_constant_dose_breadth_requires_496_unique_training_images() -> None:
    sampled, source, bands = _breadth_inputs()
    missing = next(iter(sampled))
    del sampled[missing]
    with pytest.raises(ValueError, match="below 496"):
        MODULE.select_constant_dose_breadth_arms(
            sampled_candidates=sampled,
            source_candidates=source,
            training_image_bands=bands,
            trajectory_panel_execution_metadata=_batch1_request_major_execution_metadata(),
        )


def test_constant_dose_reservoir_rejects_split_leakage_and_eval_admission() -> None:
    sampled, source, bands = _breadth_inputs()
    with pytest.raises(ValueError, match="split leakage"):
        MODULE.validate_constant_dose_training_reservoir(
            training_image_bands=bands,
            development_image_ids=[next(iter(bands))],
            heldout_image_ids=[],
            sampled_candidates=sampled,
            source_candidates=source,
        )
    with pytest.raises(ValueError, match="not confined"):
        MODULE.validate_constant_dose_training_reservoir(
            training_image_bands=bands,
            development_image_ids=[],
            heldout_image_ids=[],
            sampled_candidates={**sampled, "999999": []},
            source_candidates=source,
        )


def test_constant_dose_breadth_rejects_noncanonical_batch_execution_contract() -> None:
    sampled, source, bands = _breadth_inputs()
    contract = _batch1_request_major_execution_metadata()
    contract["physical_batch_size"] = 8
    with pytest.raises(ValueError, match="physical_batch_size=1"):
        MODULE.select_constant_dose_breadth_arms(
            sampled_candidates=sampled,
            source_candidates=source,
            training_image_bands=bands,
            trajectory_panel_execution_metadata=contract,
        )
