from __future__ import annotations

import copy
from pathlib import Path

import pytest

from scripts.research.assemble_constant_dose_breadth_state_banks import (
    ANALYZER_PATH,
    AssemblyError,
    GREEDY_SEED,
    SAMPLED_SEEDS,
    _selection_receipt,
    _validate_panel_receipt_match,
    _validate_trajectory_analysis_provenance,
    derive_reference_records,
)
from src.config.fingerprint import sha256_file
from src.inference.backend import token_ids_sha256


def _prompt_evidence(image_path: Path) -> dict[tuple[str, int], dict[str, object]]:
    prompt = [1, 151655, 151655, 2]
    result: dict[tuple[str, int], dict[str, object]] = {}
    for seed in (GREEDY_SEED, *sorted(SAMPLED_SEEDS)):
        result[("42", seed)] = {
            "prompt_token_ids": prompt,
            "prompt_token_ids_sha256": token_ids_sha256(prompt),
            "image_path": str(image_path),
            "image_sha256": "a" * 64,
            "width": 32,
            "height": 24,
        }
    return result


def _candidate(image_name: str) -> dict[str, object]:
    return {
        "image_id": 42,
        "images": [image_name],
        "metadata": {"split": "train"},
        "width": 32,
        "height": 24,
        "objects": [{"coco_ann_id": 1, "bbox_2d": ["<|coord_1|>"] * 4, "category_name": "cat"}],
    }


def _analysis_provenance(
    *, candidate_pool: Path, rollout_paths: list[Path]
) -> dict[str, object]:
    observed_panel = {
        "image_count": 2432,
        "greedy_seeds": [GREEDY_SEED],
        "sampled_seeds": sorted(SAMPLED_SEEDS),
        "greedy_trajectories_per_image": [1],
        "sampled_trajectories_per_image": [len(SAMPLED_SEEDS)],
    }
    return {
        "fixed_budgets": [16],
        "iou_threshold": 0.5,
        "require_full_panel": False,
        "observed_panel": observed_panel,
        "sources": {
            "annotations_sha256": sha256_file(candidate_pool),
            "rollout_artifact_sha256": [sha256_file(path) for path in rollout_paths],
            "analyzer_path": str(ANALYZER_PATH.resolve()),
            "analyzer_sha256": sha256_file(ANALYZER_PATH),
            "analysis_policy": {
                "fixed_budgets": [16],
                "iou_threshold": 0.5,
                "require_full_panel": False,
                "review_decisions_used": False,
                "observed_panel": observed_panel,
            },
        },
    }


def _provenance_inputs(tmp_path: Path) -> tuple[Path, list[Path], dict[str, object]]:
    candidate_pool = tmp_path / "candidate-pool.jsonl"
    candidate_pool.write_text("candidate pool\n", encoding="utf-8")
    rollout_paths = [tmp_path / "old-greedy.json", tmp_path / "new-sampled.json"]
    for index, path in enumerate(rollout_paths):
        path.write_text(f"rollout {index}\n", encoding="utf-8")
    return (
        candidate_pool,
        rollout_paths,
        _analysis_provenance(
            candidate_pool=candidate_pool, rollout_paths=rollout_paths
        ),
    )


def _validate_provenance(
    analysis: dict[str, object], candidate_pool: Path, rollout_paths: list[Path]
) -> None:
    _validate_trajectory_analysis_provenance(
        analysis,
        candidate_pool=candidate_pool,
        rollout_paths=rollout_paths,
        validated_union={"image_count": 2432},
    )


def test_derives_reference_from_exact_prompt_ids_and_candidate_metadata(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    candidate_rows = {"42": _candidate(image.name)}
    greedy_rows = {("42", GREEDY_SEED): {}}
    sampled_rows = {("42", seed): {} for seed in SAMPLED_SEEDS}

    references = derive_reference_records(
        candidate_rows=candidate_rows,
        candidate_pool_path=tmp_path / "candidate-pool.jsonl",
        greedy_rows=greedy_rows,
        sampled_rows=sampled_rows,
        prompt_evidence=_prompt_evidence(image),
    )

    assert references["42"]["split"] == "train"
    assert references["42"]["image"]["path"] == str(image)
    assert references["42"]["image_pad_interval"] == [1, 3]


def test_rejects_any_sampled_prompt_drift(tmp_path: Path) -> None:
    image = tmp_path / "image.jpg"
    evidence = _prompt_evidence(image)
    evidence[("42", min(SAMPLED_SEEDS))] = {
        **evidence[("42", min(SAMPLED_SEEDS))],
        "prompt_token_ids": [1, 151655, 2],
        "prompt_token_ids_sha256": token_ids_sha256([1, 151655, 2]),
    }

    with pytest.raises(AssemblyError, match="greedy/sampled exact prompt evidence differs"):
        derive_reference_records(
            candidate_rows={"42": _candidate(image.name)},
            candidate_pool_path=tmp_path / "candidate-pool.jsonl",
            greedy_rows={("42", GREEDY_SEED): {}},
            sampled_rows={("42", seed): {} for seed in SAMPLED_SEEDS},
            prompt_evidence=evidence,
        )


def test_saved_panel_receipt_must_match_recomputed_execution_metadata() -> None:
    recomputed = {
        "schema_version": "constant_dose_trajectory_panel_union.v1",
        "passed": True,
        "image_count": 2432,
        "old_image_count": 256,
        "new_image_count": 2176,
        "sampled_seed_count": 16,
        "prompt_policy_fingerprint": "prompt",
        "execution_metadata": {
            "physical_batch_size": 1,
            "sampling_order": "request_major",
            "rng_reset": "per_image_seed",
            "producer_script": "/producer.py",
            "producer_script_sha256": "a" * 64,
        },
    }
    saved = {
        **recomputed,
        "execution_metadata": {
            **recomputed["execution_metadata"],
            "producer_script_sha256": "b" * 64,
        },
    }

    with pytest.raises(AssemblyError, match="execution_metadata"):
        _validate_panel_receipt_match(saved, recomputed)


def test_selection_receipt_preserves_rank_matching_decision() -> None:
    execution = {
        "physical_batch_size": 1,
        "sampling_order": "request_major",
        "rng_reset": "per_image_seed",
    }
    matching = {
        "matching_mode": "coarse_band_by_rank_1_2_3_4_plus",
        "interpretation_scope": (
            "image_breadth_with_predeclared_coarse_band_by_rank_matching"
        ),
    }
    event = {
        "event_id": "event-1",
        "image_id": "42",
        "event_family": "treatment",
        "object_count_band": "sparse_1_to_3",
        "selection_rank": 4,
        "route_id": "route-1",
        "route_seed": 31001,
        "generated_row_index": 0,
        "owner_id": "owner-1",
        "prefix_token_ids_sha256": "a" * 64,
        "candidate_token_ids_sha256": "b" * 64,
        "route_count": 16,
        "row_depth": 1,
        "complete_row_coordinate_token_supervision": "enabled",
        "image_balanced_event_weight": 1.0,
    }
    arm = {
        "image_ids": ["42"],
        "event_count": 1,
        "sampled_event_count": 1,
        "source_event_count": 0,
        "mean_event_weight": 1.0,
        "total_event_weight": 1.0,
        "image_family_event_counts": {"42": {"treatment": 1}},
        "selection_distributions": {},
        "trajectory_panel_execution_metadata": execution,
        "rank_matching_receipt": matching,
        "events": [event],
    }
    selection = {
        "trajectory_panel_execution_metadata": execution,
        "eligible_unique_training_image_count": 496,
        "eligible_images_by_band": {"sparse_1_to_3": 124},
        "broad_band_quota": {"sparse_1_to_3": 124},
        "concentrated_band_quota": {"sparse_1_to_3": 30},
        "pair_quota_by_object_count_band": {"sparse_1_to_3": 124},
        "rank_matching_receipt": matching,
        "broad": arm,
        "concentrated": arm,
    }

    receipt = _selection_receipt(selection)

    assert receipt["rank_matching_receipt"] == matching
    assert receipt["pair_quota_by_object_count_band"] == {
        "sparse_1_to_3": 124
    }
    assert receipt["arms"]["broad"]["rank_matching_receipt"] == matching


def test_analysis_provenance_accepts_exact_incomplete_mode_observed_panel(
    tmp_path: Path,
) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)

    _validate_provenance(analysis, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_swapped_rollout_hash(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    sources = mutated["sources"]
    assert isinstance(sources, dict)
    hashes = sources["rollout_artifact_sha256"]
    assert isinstance(hashes, list)
    hashes[1] = hashes[0]

    with pytest.raises(AssemblyError, match="rollout artifact hashes"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_wrong_annotations_hash(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    sources = mutated["sources"]
    assert isinstance(sources, dict)
    sources["annotations_sha256"] = "f" * 64

    with pytest.raises(AssemblyError, match="annotations hash"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_review_overlay(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    sources = mutated["sources"]
    assert isinstance(sources, dict)
    policy = sources["analysis_policy"]
    assert isinstance(policy, dict)
    policy["review_decisions_used"] = True

    with pytest.raises(AssemblyError, match="must not use a review overlay"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_wrong_budget(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    mutated["fixed_budgets"] = [8, 16]

    with pytest.raises(AssemblyError, match=r"fixed budget \[16\]"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_wrong_iou_threshold(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    sources = mutated["sources"]
    assert isinstance(sources, dict)
    policy = sources["analysis_policy"]
    assert isinstance(policy, dict)
    policy["iou_threshold"] = 0.45

    with pytest.raises(AssemblyError, match="canonical IoU threshold 0.5"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_stale_analyzer_hash(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    sources = mutated["sources"]
    assert isinstance(sources, dict)
    sources["analyzer_sha256"] = "f" * 64

    with pytest.raises(AssemblyError, match="analyzer hash is stale or swapped"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)


def test_analysis_provenance_rejects_observed_panel_drift(tmp_path: Path) -> None:
    candidate_pool, rollout_paths, analysis = _provenance_inputs(tmp_path)
    mutated = copy.deepcopy(analysis)
    sources = mutated["sources"]
    assert isinstance(sources, dict)
    policy = sources["analysis_policy"]
    assert isinstance(policy, dict)
    observed = policy["observed_panel"]
    assert isinstance(observed, dict)
    observed["sampled_seeds"] = list(range(21001, 21017))

    with pytest.raises(AssemblyError, match="observed panel"):
        _validate_provenance(mutated, candidate_pool, rollout_paths)
