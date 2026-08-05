from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.research.build_sorted_owner_basin_census as census_builder
from scripts.research.build_sorted_owner_basin_census import (
    CensusContractError,
    EXPECTED_SAMPLED_SEEDS,
    _min_cost_max_cardinality_assignment,
    _pixel_box,
    _source_bins_to_pixel_box,
    build_census,
    write_census,
)
from src.data.geometry import coord_bins_to_pixel_xyxy


@pytest.fixture(autouse=True)
def _stable_repository_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep shared-worktree edits from racing an in-process build/write test."""

    snapshot = census_builder._repository_state()
    monkeypatch.setattr(census_builder, "_repository_state", lambda _repository_root=None: snapshot)


def _json_hash(value: Any) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _identity(root: Path) -> dict[str, object]:
    base = root / "base"
    adapter = root / "adapter"
    delta = root / "delta"
    for directory in (base, adapter, delta):
        directory.mkdir(parents=True)
        (directory / "fixture.bin").write_bytes(b"fixture")
    return {
        "base": {"path": str(base)},
        "adapter": {"adapter_path": str(adapter)},
        "embedding_delta": {"identity": {"delta_path": str(delta)}},
    }


def _rollout(image_id: str, seed: int, mode: str, predictions: list[tuple[str, list[float]]]) -> dict[str, object]:
    prompt_ids = [11, 12, 13]
    generated_ids = [100 + seed, 101 + seed]
    return {
        "image_id": image_id,
        "example_id": image_id,
        "seed": seed,
        "decode_mode": mode,
        "stop_reason": "im_end",
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": _json_hash(prompt_ids),
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": _json_hash(generated_ids),
        "executed_media_sha256": "media-sha",
        "observed_image_grid_thw": [1, 1, 1],
        "predictions": {
            "parse_status": "accepted",
            "metric_bearing": True,
            "dropped_prediction_count": 0,
            "dropped_predictions": [],
            "valid_prediction_count": len(predictions),
            "predictions": [
                {
                    "description": description,
                    "bbox": bbox,
                    "generated_order": index,
                    "object_span_id": f"{image_id}:{seed}:{index}",
                }
                for index, (description, bbox) in enumerate(predictions)
            ],
        },
    }


def _artifact(
    path: Path,
    *,
    identity: dict[str, object],
    mode: str,
    rollouts: list[dict[str, object]],
    horizon: int = 3084,
    extra: dict[str, object] | None = None,
) -> Path:
    image_ids = sorted({str(item["image_id"]) for item in rollouts})
    config = {
        "decode_mode": mode,
        "max_new_tokens": horizon,
        "model_dtype": "fp32",
        "repetition_penalty": 1.0,
        "temperature": 0.0 if mode == "greedy" else 0.4,
        "top_p": 1.0 if mode == "greedy" else 0.95,
        "seeds": [0] if mode == "greedy" else list(EXPECTED_SAMPLED_SEEDS),
        "image_ids": image_ids,
        "resolved_fingerprint": "fixture-fingerprint",
    }
    prompt_ids = [11, 12, 13]
    payload: dict[str, object] = {
        "schema_version": "current_seeded_sampled_rollouts.v1",
        "rollout_count": len(rollouts),
        "config": config,
        "model_identity": identity,
        "prompt_metadata": {
            image_id: {
                "prompt_token_ids": prompt_ids,
                "chat_text_sha256": "chat-sha",
                "image_sha256": "source-image-sha",
            }
            for image_id in image_ids
        },
        "rollouts": rollouts,
    }
    if extra:
        payload.update(extra)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, duplicate_owners: bool = False, greedy_description: str = "cat") -> tuple[Path, Path, tuple[Path, Path]]:
    identity = _identity(tmp_path / "components")
    panel = tmp_path / "panel.jsonl"
    objects = [
        {"bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_999|>"], "desc": "cat", "category_id": 17},
        {"bbox_2d": ["<|coord_500|>", "<|coord_0|>", "<|coord_999|>", "<|coord_999|>"], "desc": "dog", "category_id": 18},
    ]
    if duplicate_owners:
        objects = [
            {"bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_999|>"], "desc": "cat", "category_id": 17},
            {"bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_999|>"], "desc": "cat", "category_id": 17},
        ]
    panel.write_text(json.dumps({"image_id": "scene", "width": 100, "height": 100, "objects": objects}) + "\n", encoding="utf-8")
    greedy = _artifact(
        tmp_path / "greedy.json",
        identity=identity,
        mode="greedy",
        rollouts=[_rollout("scene", 0, "greedy", [(greedy_description, [0, 0, 50, 100])])],
    )
    sampled_rollouts = [
        _rollout(
            "scene",
            seed,
            "sampled",
            [("dog", [50, 0, 100, 100])] if seed == 21001 and not duplicate_owners else [("cat", [0, 0, 50, 100])],
        )
        for seed in EXPECTED_SAMPLED_SEEDS
    ]
    shard0 = _artifact(tmp_path / "shard0.json", identity=identity, mode="sampled", rollouts=sampled_rollouts[:8])
    shard1 = _artifact(tmp_path / "shard1.json", identity=identity, mode="sampled", rollouts=sampled_rollouts[8:])
    return panel, greedy, (shard0, shard1)


def _duplicate_prediction_exchange_fixture(tmp_path: Path) -> tuple[Path, Path, tuple[Path, Path]]:
    identity = _identity(tmp_path / "components")
    panel = tmp_path / "panel.jsonl"
    objects = [
        {
            "bbox_2d": ["<|coord_0|>", "<|coord_0|>", "<|coord_500|>", "<|coord_999|>"],
            "desc": "cat",
            "category_id": 17,
        },
        {
            "bbox_2d": ["<|coord_500|>", "<|coord_0|>", "<|coord_999|>", "<|coord_999|>"],
            "desc": "cat",
            "category_id": 17,
        },
    ]
    panel.write_text(
        json.dumps({"image_id": "scene", "width": 100, "height": 100, "objects": objects}) + "\n",
        encoding="utf-8",
    )
    greedy = _artifact(
        tmp_path / "greedy.json",
        identity=identity,
        mode="greedy",
        rollouts=[
            _rollout(
                "scene",
                0,
                "greedy",
                [("cat", [0, 0, 100, 100]), ("cat", [0, 0, 100, 100])],
            )
        ],
    )
    sampled_rollouts = [
        _rollout(
            "scene",
            seed,
            "sampled",
            [("cat", [0, 0, 50, 100]), ("cat", [50, 0, 100, 100])],
        )
        for seed in EXPECTED_SAMPLED_SEEDS
    ]
    shard0 = _artifact(tmp_path / "shard0.json", identity=identity, mode="sampled", rollouts=sampled_rollouts[:8])
    shard1 = _artifact(tmp_path / "shard1.json", identity=identity, mode="sampled", rollouts=sampled_rollouts[8:])
    return panel, greedy, (shard0, shard1)


def _build_fixture(tmp_path: Path, **kwargs: object) -> dict[str, Any]:
    panel, greedy, shards = _fixture(tmp_path, **kwargs)
    return build_census(
        panel_path=panel,
        greedy_path=greedy,
        sampled_paths=shards,
        require_full_panel=False,
        hash_model_components=False,
    )


def test_writes_stable_ledgers_and_primary_any_hit_owner_union(tmp_path: Path) -> None:
    census = _build_fixture(tmp_path)

    primary = census["primary_metrics"]
    assert primary["estimand"] == "per_trajectory_any_hit_physical_owner_union"
    assert primary["not_estimand"] == "cross_trajectory_medoid_detection_union"
    assert primary["matched_rp_1_0_greedy"]["owner_presence_count"] == 1
    assert primary["matched_rp_1_0_k16_any_hit"]["owner_presence_count"] == 2
    assert primary["matched_rp_1_0_k16_any_hit"]["strict_rescued_gt_owner_ids"] == ["gt:scene:1"]
    assert census["owner_ledger"][0]["gt_owner_id"] == "gt:scene:0"
    assert census["prediction_ledger"][0]["pred_row_id"].startswith("pred:sorted:greedy:0:scene:")

    output = tmp_path / "census"
    paths = write_census(output, census)
    assert {path.name for path in paths.values()} == {
        "artifact-manifest.json",
        "ambiguity-receipts.jsonl",
        "execution-receipt.json",
        "matcher-contract.json",
        "owner-ledger.jsonl",
        "prediction-row-ledger.jsonl",
        "owner-trajectory-matrix.jsonl",
        "native-replay.jsonl",
    }
    pred_row = json.loads((output / "prediction-row-ledger.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert pred_row["schema_version"] == "sorted-owner-basin-prediction-row-ledger.v2"
    assert pred_row["schema_compatibility"]["replaces_schema_version"] == (
        "sorted-owner-basin-prediction-row-ledger.v1"
    )
    assert pred_row["foreign_keys"]["trajectory_id"].startswith("trajectory:sorted:")
    assert pred_row["source_digests"]["rollout_artifact"]
    assert pred_row["execution_receipt_content_sha256"]
    receipt = json.loads((output / "execution-receipt.json").read_text(encoding="utf-8"))
    receipt_content_sha256 = receipt["execution_receipt_content_sha256"]
    assert receipt["execution_status"] == "completed"
    assert receipt["execution_surface"] == "deterministic_cpu_census_no_model_inference"
    assert receipt["runtime"]["execution_device"] == "cpu"
    assert receipt["runtime"]["cuda_inventory_role"] == (
        "observed_environment_metadata_not_used_for_census_execution"
    )
    assert receipt_content_sha256 == pred_row["execution_receipt_content_sha256"]
    assert receipt_content_sha256 == _json_hash(
        {key: value for key, value in receipt.items() if key != "execution_receipt_content_sha256"}
    )
    assert receipt["coordinate_contract"]["coordinate_bin_max"] == 999
    assert receipt["coordinate_contract"]["pixel_conversion"] == "round(value * extent / 1000)"
    relevant_untracked = {
        item["relative_path"] for item in receipt["repository"]["relevant_untracked_files"]
    }
    assert {
        "scripts/research/build_sorted_owner_basin_census.py",
        "tests/research/test_sorted_owner_basin_census.py",
        (
            "research/investigations/qwen3-vl-dense-enumeration/experiments/"
            "2026-08-01-sorted-owner-basin-landscape-and-repair/control-registry.json"
        ),
        (
            "research/investigations/qwen3-vl-dense-enumeration/experiments/"
            "2026-08-01-sorted-owner-basin-landscape-and-repair/sentinel-registry.json"
        ),
        (
            "research/investigations/qwen3-vl-dense-enumeration/experiments/"
            "2026-08-01-sorted-owner-basin-landscape-and-repair/unit.md"
        ),
    } <= relevant_untracked
    manifest = json.loads((output / "artifact-manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "sorted-owner-basin-census-artifact-manifest.v2"
    assert manifest["execution_receipt_content_sha256"] == receipt_content_sha256
    import hashlib

    receipt_file_sha256 = hashlib.sha256((output / "execution-receipt.json").read_bytes()).hexdigest()
    assert manifest["artifacts"]["execution_receipt"]["sha256"] == receipt_file_sha256
    assert receipt_file_sha256 != receipt_content_sha256
    matcher = json.loads((output / "matcher-contract.json").read_text(encoding="utf-8"))
    assert matcher["category_namespace"]["local_evaluator_category_id_join"] == "forbidden"

    with pytest.raises(CensusContractError, match="refusing to overwrite"):
        write_census(output, census)


def test_rejects_legacy_512_horizon_instead_of_inheriting_old_reader_limit(tmp_path: Path) -> None:
    panel, greedy, shards = _fixture(tmp_path)
    payload = json.loads(greedy.read_text(encoding="utf-8"))
    payload["config"]["max_new_tokens"] = 512
    greedy.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CensusContractError, match="max_new_tokens must be 3084"):
        build_census(panel_path=panel, greedy_path=greedy, sampled_paths=shards, require_full_panel=False, hash_model_components=False)


def test_source_coordinate_bins_use_canonical_rounding_and_pixel_rollouts_are_unchanged() -> None:
    bins = [1, 2, 501, 777]
    source_tokens = [f"<|coord_{value}|>" for value in bins]
    expected = coord_bins_to_pixel_xyxy(bins, image_width=101, image_height=103, field="expected")

    assert _source_bins_to_pixel_box(source_tokens, width=101, height=103, context="fixture.bbox_2d") == expected
    assert _pixel_box([0.25, 1.5, 50.75, 99.5], context="fixture.rollout.bbox") == (0.25, 1.5, 50.75, 99.5)


def test_coord_1000_is_rejected_by_the_canonical_source_coordinate_contract() -> None:
    with pytest.raises(CensusContractError, match="canonical source coordinate contract"):
        _source_bins_to_pixel_box(
            ["<|coord_0|>", "<|coord_0|>", "<|coord_999|>", "<|coord_1000|>"],
            width=100,
            height=100,
            context="fixture.bbox_2d",
        )


def test_aliases_are_opt_in_and_never_inferred(tmp_path: Path) -> None:
    panel, greedy, shards = _fixture(tmp_path, greedy_description="kitty")
    without_aliases = build_census(panel_path=panel, greedy_path=greedy, sampled_paths=shards, require_full_panel=False, hash_model_components=False)
    assert without_aliases["primary_metrics"]["matched_rp_1_0_greedy"]["owner_presence_count"] == 0

    aliases = tmp_path / "aliases.json"
    aliases.write_text(json.dumps({"schema_version": "sorted-owner-basin-aliases.v1", "aliases": [{"left": "kitty", "right": "cat"}]}), encoding="utf-8")
    with_aliases = build_census(panel_path=panel, greedy_path=greedy, sampled_paths=shards, alias_table_path=aliases, require_full_panel=False, hash_model_components=False)
    assert with_aliases["primary_metrics"]["matched_rp_1_0_greedy"]["owner_presence_count"] == 1
    assert with_aliases["matcher_contract"]["alias_table"]["aliases"] == [{"left": "cat", "right": "kitty"}]


def test_indistinguishable_same_description_assignment_is_neutral(tmp_path: Path) -> None:
    census = _build_fixture(tmp_path, duplicate_owners=True)
    automatic = census["primary_metrics"]["matched_rp_1_0_greedy"]["aggregate_detection_counts"][
        "automatic_deterministic_assignment"
    ]
    committed = census["primary_metrics"]["matched_rp_1_0_greedy"]["aggregate_detection_counts"]["neutral_ambiguity_excluded_owner_presence"]
    assert automatic["tp"] == 1
    assert committed["tp"] == 0
    assert committed["fp"] == 0
    assert committed["fn"] == 0
    assert committed["neutral_assignment_count"] == 1
    assert committed["neutral_prediction_count"] == 1
    assert committed["neutral_owner_count"] == 2
    assert committed["precision_denominator"] == 0
    assert committed["recall_denominator"] == 0
    greedy_row = next(item for item in census["prediction_ledger"] if item["decode_mode"] == "greedy")
    assert greedy_row["strict_match_status"] == "ambiguous_neutral"


def test_duplicate_prediction_exchange_is_first_class_global_ambiguity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = [
        {"pred_row_id": f"pred-{index}", "original_row_index": index, "normalized_description": "cat", "bbox_xyxy": (0, 0, 1, 1)}
        for index in range(2)
    ]
    owners = [
        {"gt_owner_id": "owner-0", "original_annotation_index": 0, "normalized_description": "cat", "bbox_xyxy": (10, 0, 11, 1)},
        {"gt_owner_id": "owner-1", "original_annotation_index": 1, "normalized_description": "cat", "bbox_xyxy": (11, 0, 12, 1)},
    ]
    monkeypatch.setattr(
        census_builder,
        "_iou",
        lambda _left, right: 0.9 if tuple(right) == (10, 0, 11, 1) else 0.6,
    )

    analysis = census_builder._global_assignment_analysis(predictions, owners, {})

    assert len(analysis["ambiguity_classes"]) == 1
    ambiguity = analysis["ambiguity_classes"][0]
    assert ambiguity["reason"] == "globally_indistinguishable_exact_duplicate_prediction_rows"
    assert ambiguity["pred_row_ids"] == ["pred-0", "pred-1"]
    assert ambiguity["gt_owner_ids"] == ["owner-0", "owner-1"]
    assert ambiguity["objective_equality"]["maximum_cardinality"] == 2
    assert ambiguity["objective_equality"]["maximum_total_iou"] == pytest.approx(1.5)
    assert ambiguity["objective_equality"]["all_forced_edges_preserve_global_objective"] is True
    assert len(ambiguity["globally_optimal_edge_receipts"]) == 4


def test_neutral_owners_are_excluded_from_paired_rescue_and_fn_denominators(tmp_path: Path) -> None:
    panel, greedy, shards = _duplicate_prediction_exchange_fixture(tmp_path)
    census = build_census(
        panel_path=panel,
        greedy_path=greedy,
        sampled_paths=shards,
        require_full_panel=False,
        hash_model_components=False,
    )

    greedy_metrics = census["primary_metrics"]["matched_rp_1_0_greedy"]
    assert greedy_metrics["owner_presence_count"] == 0
    assert greedy_metrics["effective_owner_denominator"] == 0
    assert greedy_metrics["automatic_deterministic_assignment"] == {
        "owner_presence_count": 2,
        "owner_denominator": 2,
        "decision_bearing": False,
        "reason": "deterministic representative of globally ambiguous assignments",
    }
    neutral_counts = greedy_metrics["aggregate_detection_counts"]["neutral_ambiguity_excluded_owner_presence"]
    assert neutral_counts["fn"] == 0
    assert neutral_counts["recall_denominator"] == 0

    k16_metrics = census["primary_metrics"]["matched_rp_1_0_k16_any_hit"]
    paired = k16_metrics["ambiguity_neutral_paired_set"]
    assert k16_metrics["owner_presence_count"] == 2
    assert paired["owner_denominator"] == 0
    assert paired["k16_owner_presence_count"] == 0
    assert paired["strict_rescued_owner_count"] == 0
    assert paired["excluded_neutral_gt_owner_ids"] == ["gt:scene:0", "gt:scene:1"]
    assert all(
        row["decision_eligibility"]["greedy_k16_paired"]["eligible"] is False
        for row in census["owner_ledger"]
    )


def test_crossed_equal_total_global_optima_are_neutral_even_when_edge_ious_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    predictions = [
        {"pred_row_id": "pred-0", "original_row_index": 0, "normalized_description": "cat", "bbox_xyxy": (0, 0, 1, 1)},
        {"pred_row_id": "pred-1", "original_row_index": 1, "normalized_description": "cat", "bbox_xyxy": (1, 0, 2, 1)},
    ]
    owners = [
        {"gt_owner_id": "owner-0", "original_annotation_index": 0, "normalized_description": "cat", "bbox_xyxy": (10, 0, 11, 1)},
        {"gt_owner_id": "owner-1", "original_annotation_index": 1, "normalized_description": "cat", "bbox_xyxy": (11, 0, 12, 1)},
    ]
    overlaps = {
        ((0, 0, 1, 1), (10, 0, 11, 1)): 0.9,
        ((0, 0, 1, 1), (11, 0, 12, 1)): 0.8,
        ((1, 0, 2, 1), (10, 0, 11, 1)): 0.7,
        ((1, 0, 2, 1), (11, 0, 12, 1)): 0.6,
    }
    monkeypatch.setattr(census_builder, "_iou", lambda left, right: overlaps[(tuple(left), tuple(right))])

    matches = _min_cost_max_cardinality_assignment(predictions, owners, {})

    assert [(item["gt_owner_id"], item["pred_row_id"]) for item in matches] == [
        ("owner-0", "pred-0"),
        ("owner-1", "pred-1"),
    ]
    assert {item["intersection_over_union"] for item in matches} == {0.9, 0.6}
    assert {item["strict_match_status"] for item in matches} == {"ambiguous_neutral"}
    assert all(set(item["globally_optimal_gt_owner_ids"]) == {"owner-0", "owner-1"} for item in matches)


def test_global_optimum_tie_break_output_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    predictions = [
        {"pred_row_id": "pred-0", "original_row_index": 0, "normalized_description": "cat", "bbox_xyxy": (0, 0, 1, 1)},
        {"pred_row_id": "pred-1", "original_row_index": 1, "normalized_description": "cat", "bbox_xyxy": (1, 0, 2, 1)},
    ]
    owners = [
        {"gt_owner_id": "owner-0", "original_annotation_index": 0, "normalized_description": "cat", "bbox_xyxy": (10, 0, 11, 1)},
        {"gt_owner_id": "owner-1", "original_annotation_index": 1, "normalized_description": "cat", "bbox_xyxy": (11, 0, 12, 1)},
    ]
    monkeypatch.setattr(census_builder, "_iou", lambda _left, _right: 0.75)

    forward = _min_cost_max_cardinality_assignment(predictions, owners, {})
    reversed_inputs = _min_cost_max_cardinality_assignment(list(reversed(predictions)), list(reversed(owners)), {})

    assert forward == reversed_inputs
    assert [(item["gt_owner_id"], item["pred_row_id"]) for item in forward] == [
        ("owner-0", "pred-0"),
        ("owner-1", "pred-1"),
    ]


def test_execution_receipt_binding_mismatch_fails_before_writing(tmp_path: Path) -> None:
    census = _build_fixture(tmp_path)
    census["owner_ledger"][0]["execution_receipt_content_sha256"] = "0" * 64

    with pytest.raises(CensusContractError, match="execution receipt binding mismatch"):
        write_census(tmp_path / "must-not-exist", census)
    assert not (tmp_path / "must-not-exist").exists()


def test_forced_continuation_marker_fails_before_cohort_building(tmp_path: Path) -> None:
    panel, greedy, shards = _fixture(tmp_path)
    payload = json.loads(shards[0].read_text(encoding="utf-8"))
    payload["forced_continuation"] = True
    shards[0].write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CensusContractError, match="forced-continuation markers"):
        build_census(panel_path=panel, greedy_path=greedy, sampled_paths=shards, require_full_panel=False, hash_model_components=False)
