from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

import scripts.research.assemble_constant_dose_breadth_state_banks as breadth_assembler
from scripts.research.assemble_constant_dose_breadth_state_banks import (
    ANALYZER_PATH,
    AssemblyError,
    GREEDY_SEED,
    SAMPLED_SEEDS,
    _selection_receipt,
    _deduplicate_v2_sampled_candidates,
    _validate_panel_receipt_match,
    _validate_trajectory_analysis_provenance,
    derive_reference_records,
    load_v2_b16_panel_adapter,
)
from scripts.research.assemble_source_preservation_multi_route_state_banks import (
    _build_source_candidates,
    _write_arm,
)
from scripts.research.run_greedy_prefix_forced_owner_path import (
    BOX_END,
    BOX_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
)
from src.config.fingerprint import sha256_file
from src.inference.backend import token_ids_sha256
from src.rollout_calibration import CheckpointIdentity


_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "rollout_calibration_test_conftest_for_breadth",
    Path(__file__).parents[1] / "rollout_calibration" / "conftest.py",
)
assert _FIXTURE_SPEC and _FIXTURE_SPEC.loader
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)


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
        "objects": [
            {"coco_ann_id": 1, "bbox_2d": ["<|coord_1|>"] * 4, "category_name": "cat"}
        ],
    }


def _row_tokens(description_token: int, offset: int = 0) -> list[int]:
    return [
        OBJECT_REF_START,
        description_token,
        OBJECT_REF_END,
        BOX_START,
        151670 + offset + 1,
        151670 + offset + 2,
        151670 + offset + 3,
        151670 + offset + 4,
        BOX_END,
    ]


def _parser_prediction(
    *, category: str, row_index: int, offset: int = 0
) -> dict[str, object]:
    return {
        "description": category,
        "bbox": [1 + offset, 1 + offset, 10 + offset, 10 + offset],
        "bbox_format": "xyxy",
        "coord_bins": [offset + 1, offset + 2, offset + 3, offset + 4],
        "generated_order": row_index,
        "object_span_id": f"example-42:span-{row_index}",
        "raw_span_text": category,
        "raw_span_sha256": str(row_index) * 64,
        "schema_spans": [],
        "coord_token_spans": [],
    }


def _parser(*predictions: dict[str, object]) -> dict[str, object]:
    return {
        "parse_status": "accepted",
        "valid_prediction_count": len(predictions),
        "dropped_prediction_count": 0,
        "dropped_predictions": [],
        "predictions": list(predictions),
    }


def _v2_metadata(image: Path) -> dict[str, object]:
    prompt = [1, 151655, 2]
    return {
        "example-42": {
            "prompt_token_ids": prompt,
            "prompt_token_ids_sha256": token_ids_sha256(prompt),
            "source_image_file_sha256": "a" * 64,
            "image_path": str(image),
            "width": 32,
            "height": 24,
        }
    }


def _v2_artifact(
    *, mode: str, rows: list[dict[str, object]], image: Path
) -> dict[str, object]:
    config: dict[str, object] = {
        "decode_mode": mode,
        "panel_mode": "sampled_only" if mode == "sampled" else "source_b16",
        "temperature": 0.4 if mode == "sampled" else 0.0,
        "top_p": 0.95 if mode == "sampled" else 1.0,
        "repetition_penalty": 1.0,
        "sampling_order": "request_major",
    }
    if mode == "sampled":
        config.update({"sample_count": 16, "sample_index_range": [0, 15]})
    else:
        config["source_b16_row_budget"] = 16
    return {
        "schema_version": "coordexp_vllm_trajectory_panel.v2",
        "config": config,
        "model_identity": {
            "execution_model_identity": {
                "algorithm_version": "test-v1",
                "composition_key": "c" * 64,
            }
        },
        "prompt_metadata": _v2_metadata(image),
        "rollout_count": len(rows),
        "rollouts": rows,
    }


def _v2_base_row(*, generated_ids: list[int]) -> dict[str, object]:
    prompt = [1, 151655, 2]
    return {
        "image_id": 42,
        "example_id": "example-42",
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": token_ids_sha256(prompt),
        "source_image_file_sha256": "a" * 64,
        "executed_rgb_sha256": "b" * 64,
        "generated_token_ids": generated_ids,
        "generated_token_ids_sha256": token_ids_sha256(generated_ids),
        "image_width": 32,
        "image_height": 24,
    }


def _write_v2_panel(
    tmp_path: Path,
    *,
    source_status: str = "accepted_natural_end",
    sampled_indices: range | list[int] = range(16),
    source_execution_key: str = "c" * 64,
    source_executed_rgb_sha256: str = "b" * 64,
    projected_row_count: int = 1,
) -> tuple[Path, Path, Path, Path]:
    image = tmp_path / "image.jpg"
    image.write_bytes(b"image")
    candidate_pool = tmp_path / "candidate-pool.jsonl"
    candidate_pool.write_text("synthetic\n", encoding="utf-8")
    sampled_root = tmp_path / "sampled"
    source_root = tmp_path / "source"
    sampled_root.mkdir()
    source_root.mkdir()
    sampled_rows: list[dict[str, object]] = []
    for sample_index in sampled_indices:
        generated = _row_tokens(50 + sample_index)
        sampled_rows.append(
            {
                **_v2_base_row(generated_ids=generated),
                "trajectory_id": f"sample-{sample_index:02d}",
                "decode_mode": "sampled",
                "sample_index": sample_index,
                "stop_reason": "im_end",
                "predictions": _parser(_parser_prediction(category="cat", row_index=0)),
            }
        )
    sampled = _v2_artifact(mode="sampled", rows=sampled_rows, image=image)
    (sampled_root / "sampled-batch-00000.json").write_text(
        json.dumps(sampled), encoding="utf-8"
    )

    projected_rows = [
        _row_tokens(70 + row_index) for row_index in range(projected_row_count)
    ]
    projected = [token for row in projected_rows for token in row]
    projected_predictions = [
        _parser_prediction(category="cat", row_index=row_index)
        for row_index in range(projected_row_count)
    ]
    raw = [*projected, *_row_tokens(90, offset=10)]
    source_row = {
        **_v2_base_row(generated_ids=raw),
        "trajectory_id": "source-b16",
        "decode_mode": "source_b16",
        "stop_reason": "im_end",
        "executed_rgb_sha256": source_executed_rgb_sha256,
        "predictions": _parser(
            *projected_predictions,
            _parser_prediction(
                category="dog", row_index=projected_row_count, offset=10
            ),
        ),
        "source_b16": {
            "status": source_status,
            "row_budget": 16,
            "projected_token_ids": projected,
            "projected_token_ids_sha256": token_ids_sha256(projected),
            "projected_text": "projected-cat-only",
            "projected_valid_complete_row_count": projected_row_count,
            "projected_parser_evidence": _parser(*projected_predictions),
        },
    }
    source = _v2_artifact(mode="source_b16", rows=[source_row], image=image)
    model_identity = source["model_identity"]
    assert isinstance(model_identity, dict)
    execution_identity = model_identity["execution_model_identity"]
    assert isinstance(execution_identity, dict)
    execution_identity["composition_key"] = source_execution_key
    (source_root / "source_b16-batch-00000.json").write_text(
        json.dumps(source), encoding="utf-8"
    )
    return candidate_pool, sampled_root, source_root, image


def _patch_synthetic_candidate_pool(
    monkeypatch: pytest.MonkeyPatch, image: Path
) -> None:
    candidate = _candidate(image.name)
    candidate["objects"] = [
        {
            "coco_ann_id": 1,
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
            "category_name": "cat",
        },
        {
            "coco_ann_id": 2,
            "bbox_2d": ["<|coord_11|>", "<|coord_12|>", "<|coord_13|>", "<|coord_14|>"],
            "category_name": "dog",
        },
    ]
    monkeypatch.setattr(
        breadth_assembler, "_candidate_pool", lambda _: {"42": candidate}
    )
    monkeypatch.setattr(
        breadth_assembler,
        "load_generation7_annotations",
        lambda _: {
            "42": [
                {
                    "owner_id": "42:1",
                    "category": "cat",
                    "bbox": (1.0, 1.0, 10.0, 10.0),
                    "image_id": "42",
                },
                {
                    "owner_id": "42:2",
                    "category": "dog",
                    "bbox": (11.0, 11.0, 20.0, 20.0),
                    "image_id": "42",
                },
            ]
        },
    )


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


def test_derives_reference_from_exact_prompt_ids_and_candidate_metadata(
    tmp_path: Path,
) -> None:
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

    with pytest.raises(
        AssemblyError, match="greedy/sampled exact prompt evidence differs"
    ):
        derive_reference_records(
            candidate_rows={"42": _candidate(image.name)},
            candidate_pool_path=tmp_path / "candidate-pool.jsonl",
            greedy_rows={("42", GREEDY_SEED): {}},
            sampled_rows={("42", seed): {} for seed in SAMPLED_SEEDS},
            prompt_evidence=evidence,
        )


def test_v2_source_raw_rows_after_projection_cannot_enter_semantics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_pool, sampled_root, source_root, image = _write_v2_panel(
        tmp_path, source_status="accepted_budget", projected_row_count=16
    )
    _patch_synthetic_candidate_pool(monkeypatch, image)

    adapted = load_v2_b16_panel_adapter(
        sampled_panel_root=sampled_root,
        source_b16_root=source_root,
        candidate_pool=candidate_pool,
    )

    source_row = adapted["source_rows"][("42", 0)]
    assert source_row["generated_token_ids"] == [
        token for row_index in range(16) for token in _row_tokens(70 + row_index)
    ]
    source_assignment = adapted["image_results"]["42"]["budgets"][0][
        "trajectory_assignments"
    ]["source-b16"]
    assert source_assignment["matched_owner_ids"] == ["42:1"]
    assert "42:2" not in source_assignment["matched_owner_ids"]
    provenance = source_row["_source_b16_provenance"]
    assert provenance["raw_generated_token_count"] > provenance["projected_token_count"]


def test_v2_projected_tokens_drive_exact_source_anchor_slicing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_pool, sampled_root, source_root, image = _write_v2_panel(tmp_path)
    _patch_synthetic_candidate_pool(monkeypatch, image)
    adapted = load_v2_b16_panel_adapter(
        sampled_panel_root=sampled_root,
        source_b16_root=source_root,
        candidate_pool=candidate_pool,
    )
    candidate = _candidate(image.name)
    candidate["objects"] = [
        {
            "coco_ann_id": 1,
            "bbox_2d": ["<|coord_1|>", "<|coord_2|>", "<|coord_3|>", "<|coord_4|>"],
            "category_name": "cat",
        }
    ]

    source_candidates, _ = _build_source_candidates(
        image_results=adapted["image_results"],
        greedy_rows=adapted["source_rows"],
        reference_records=adapted["reference_records"],
        annotations={"42": candidate},
        image_ids=["42"],
        manual_review=None,
    )

    assert source_candidates["42"][0]["candidate_token_ids"] == _row_tokens(70)
    assert source_candidates["42"][0]["prefix_token_ids"] == []


def test_v2_failed_source_is_counted_and_excluded_from_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_pool, sampled_root, source_root, image = _write_v2_panel(
        tmp_path, source_status="failed_token_limit_before_budget"
    )
    _patch_synthetic_candidate_pool(monkeypatch, image)

    adapted = load_v2_b16_panel_adapter(
        sampled_panel_root=sampled_root,
        source_b16_root=source_root,
        candidate_pool=candidate_pool,
    )

    assert adapted["source_rows"] == {}
    assert adapted["image_results"] == {}
    assert adapted["census"]["source_status_counts"] == {
        "failed_token_limit_before_budget": 1
    }
    assert adapted["census"]["source_ineligible_excluded_from_admission_count"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source_executed_rgb_sha256": "d" * 64}, "identity mismatch"),
        ({"source_execution_key": "e" * 64}, "execution-model identity mismatch"),
    ],
)
def test_v2_sampled_and_source_identity_mismatch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, str],
    message: str,
) -> None:
    candidate_pool, sampled_root, source_root, image = _write_v2_panel(
        tmp_path, **kwargs
    )
    _patch_synthetic_candidate_pool(monkeypatch, image)

    with pytest.raises(AssemblyError, match=message):
        load_v2_b16_panel_adapter(
            sampled_panel_root=sampled_root,
            source_b16_root=source_root,
            candidate_pool=candidate_pool,
        )


def test_v2_requires_all_sixteen_sample_indices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_pool, sampled_root, source_root, image = _write_v2_panel(
        tmp_path, sampled_indices=list(range(15))
    )
    _patch_synthetic_candidate_pool(monkeypatch, image)

    with pytest.raises(AssemblyError, match="sample_index 0..15"):
        load_v2_b16_panel_adapter(
            sampled_panel_root=sampled_root,
            source_b16_root=source_root,
            candidate_pool=candidate_pool,
        )


def _duplicate_sampled_candidate(
    *, route_id: str, route_seed: int, marginal: int, total: int, unresolved: int
) -> dict[str, object]:
    return {
        "image_id": "42",
        "route_id": route_id,
        "route_seed": route_seed,
        "generated_row_index": 2,
        "owner_id": "42:1",
        "prefix_token_ids_sha256": "a" * 64,
        "candidate_token_ids_sha256": "b" * 64,
        "marginal_route_added_owner_count": marginal,
        "route_added_owner_count": total,
        "unresolved_row_count": unresolved,
        "event_id": f"event-{route_id}",
    }


def test_v2_sampled_dedup_keeps_existing_rank_winner_and_records_provenance() -> None:
    weaker = _duplicate_sampled_candidate(
        route_id="sample-01", route_seed=1, marginal=1, total=2, unresolved=0
    )
    stronger = _duplicate_sampled_candidate(
        route_id="sample-09", route_seed=9, marginal=3, total=4, unresolved=2
    )

    result, receipt = _deduplicate_v2_sampled_candidates(
        {"42": [weaker, stronger]}
    )

    assert [item["route_id"] for item in result["42"]] == ["sample-09"]
    assert receipt["duplicate_candidate_count_removed"] == 1
    assert receipt["duplicate_identity_group_count"] == 1
    group = receipt["duplicate_groups"][0]
    assert group["retained"]["route_id"] == "sample-09"
    assert group["discarded"][0]["route_id"] == "sample-01"


def test_v2_sampled_dedup_choice_is_independent_of_input_order() -> None:
    later = _duplicate_sampled_candidate(
        route_id="sample-12", route_seed=12, marginal=2, total=3, unresolved=1
    )
    earlier = _duplicate_sampled_candidate(
        route_id="sample-03", route_seed=3, marginal=2, total=3, unresolved=1
    )

    forward = _deduplicate_v2_sampled_candidates({"42": [later, earlier]})
    reverse = _deduplicate_v2_sampled_candidates({"42": [earlier, later]})

    assert forward[0]["42"][0]["route_id"] == "sample-03"
    assert reverse[0]["42"][0]["route_id"] == "sample-03"
    assert forward[1] == reverse[1]


def test_v2_source_provenance_survives_canonical_992_event_arm_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture_root = tmp_path / "canonical-fixture"
    fixture_root.mkdir()
    base_rollouts, base_reviews, _ = _FIXTURE_MODULE.synthetic_inputs(fixture_root)
    base_rollout = base_rollouts[0]
    base_review = base_reviews[0]
    base_rollout["candidates"] = [base_rollout["candidates"][0]]
    base_review["candidates"] = [base_review["candidates"][0]]
    base_review["candidates"][0].update(
        {
            "owner_resolution_interval": [0, 3],
            "selected_sites": [
                {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                {"candidate_token_offset": 2, "intended_token_type": "schema"},
            ],
        }
    )
    image_variants: dict[int, tuple[Path, str]] = {}
    for group in range(62):
        image_path = tmp_path / f"canonical-image-{group:02d}.png"
        Image.new("RGB", (64, 64), color=(group, 34, 56)).save(image_path)
        image_variants[group] = (image_path, sha256_file(image_path))

    rollouts: list[dict[str, object]] = []
    reviews: list[dict[str, object]] = []
    receipts: list[dict[str, object]] = []
    source_events: list[dict[str, object]] = []
    for index in range(992):
        source_index = index - 496
        image_group = index % 62
        image_id = 10_000 + image_group
        image_path, image_hash = image_variants[image_group]
        event_id = (
            f"treatment-image-{image_id}-event-{index}"
            if index < 496
            else f"source-preservation-image-{image_id}-route-source-{source_index}-row-{source_index}"
        )
        rollout = copy.deepcopy(base_rollout)
        review = copy.deepcopy(base_review)
        rollout["event_id"] = review["event_id"] = event_id
        rollout["image"].update(
            {
                "image_id": image_id,
                "path": str(image_path),
                "content_sha256": image_hash,
            }
        )
        rollout["split_group_id"] = f"image:{image_id}"
        review.update(
            {
                "entity_transition_eligible": False,
                "coordinate_boundary_eligible": False,
                "positive_path_imitation_eligible": index < 496,
                "source_route_imitation_eligible": index >= 496,
                "image_balanced_event_weight": 1.0,
            }
        )
        review["review_provenance"].update(
            {
                "event_family": (
                    "multi_route_treatment" if index < 496 else "source_preservation"
                ),
                "route_id": f"fixture-route-{index}",
                "generated_row_index": index,
            }
        )
        if index >= 496:
            generation = rollout["candidates"][0]["generation_provenance"]
            generation.update(
                {"mode": "greedy", "seed": 0, "temperature": 0.0, "top_p": 1.0}
            )
            source_provenance = {
                "status": "accepted_budget",
                "raw_generated_token_ids_sha256": f"{source_index:064x}",
                "raw_generated_token_count": 64,
                "raw_stop_reason": "source_b16_row_budget",
                "projected_token_ids_sha256": f"{source_index + 1:064x}",
                "projected_token_count": 32,
                "projected_valid_complete_row_count": 16,
            }
            source_events.append(
                {
                    "event_family": "source_preservation",
                    "image_id": str(image_id),
                    "route_id": f"source-{source_index}",
                    "generated_row_index": source_index,
                    "_event_inputs": {
                        "rollout": {"_source_b16_provenance": source_provenance}
                    },
                }
            )
        rollouts.append(rollout)
        reviews.append(review)
        receipts.append({"event_id": event_id})

    arm_receipt = {
        "arm": "constant_dose_concentrated_plus_source_preservation",
        "event_count": 992,
        "image_count": 159,
    }
    monkeypatch.setattr(
        breadth_assembler,
        "materialize_constant_dose_breadth_arm",
        lambda _selection, checkpoint_id: (
            copy.deepcopy(rollouts),
            copy.deepcopy(reviews),
            copy.deepcopy(receipts),
            copy.deepcopy(arm_receipt),
        ),
    )
    materialized = breadth_assembler._materialize_v2_arm(
        {"events": source_events}, checkpoint_id="unused-by-patched-materializer"
    )
    materialized_rollouts, materialized_reviews, materialized_receipts, arm = materialized
    assert all(
        "source_b16" not in candidate["generation_provenance"]
        for rollout in materialized_rollouts
        for candidate in rollout["candidates"]
    )
    assert sum(
        "source_b16" in review["review_provenance"]
        for review in materialized_reviews
    ) == 496
    assert sum("source_b16" in receipt for receipt in materialized_receipts) == 496

    output = tmp_path / "disposable-v2-canonical-write"
    write_receipt = _write_arm(
        root=output,
        rollouts=materialized_rollouts,
        reviews=materialized_reviews,
        receipts=materialized_receipts,
        census={"purpose": "production-shaped-v2-schema-test"},
        arm_receipt=arm,
        source_checkpoint=CheckpointIdentity(
            **_FIXTURE_MODULE.SYNTHETIC_SOURCE_CHECKPOINT
        ),
        prompt_identity_sha256="8" * 64,
        source_artifacts=_FIXTURE_MODULE.source_artifacts(),
    )
    assert write_receipt["state_bank_manifest"]["record_count"] == 992
    assert write_receipt["state_bank_validation_receipt"]["record_count"] == 992
    assert write_receipt["counts"] == {
        "rollout_rows": 992,
        "review_rows": 992,
        "event_receipts": 992,
    }
    assert sum("source_b16" in item for item in write_receipt["event_receipts"]) == 496
    assert output.is_relative_to(tmp_path)


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
    assert receipt["pair_quota_by_object_count_band"] == {"sparse_1_to_3": 124}
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
