from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis.qwen3_vl_instance_binding import (
    REQUIRED_POSITION_ROLES,
    CaseSelectionConfig,
    RuntimePathConfig,
    _best_other_candidate_from_alignment,
    _fine_schema_relative_spans_from_inventory,
    _select_core_donor_spec,
    _stage_shard_jsonl_paths,
    _summarize_core_diagnosis_rows,
    _summarize_donor_patching_rows,
    audit_checkpoint_surface,
    candidate_alignment_from_distribution,
    classify_mechanism,
    distribute_items_by_shard,
    load_study_config,
    mine_repeated_desc_cases,
    resolve_runtime_paths,
    run_study_stage,
    select_hidden_state_layers,
    validate_position_inventory,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def test_resolve_runtime_paths_prefers_shared_root_for_heavy_artifacts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo" / ".worktrees" / "binding"
    shared_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    checkpoint = shared_root / "output_remote" / "ckpt"
    dataset = shared_root / "public_data" / "coco" / "val.coord.jsonl"
    artifact_root = shared_root / "output" / "analysis" / "study"
    checkpoint.mkdir(parents=True)
    dataset.parent.mkdir(parents=True)
    dataset.write_text("", encoding="utf-8")
    (repo_root / "public_data" / "coco").mkdir(parents=True)

    resolved = resolve_runtime_paths(
        RuntimePathConfig(
            checkpoint="output_remote/ckpt",
            dataset_jsonls=("public_data/coco/val.coord.jsonl",),
            artifact_root="output/analysis/study",
        ),
        repo_root=repo_root,
        shared_root=shared_root,
    )

    assert resolved.checkpoint == checkpoint
    assert resolved.dataset_jsonls == (dataset,)
    assert resolved.artifact_root == artifact_root


def test_resolve_runtime_paths_rejects_worktree_checkpoint_shadow(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo" / ".worktrees" / "binding"
    shared_root = tmp_path / "repo"
    repo_root.mkdir(parents=True)
    shadow = repo_root / "output_remote" / "ckpt"
    shadow.mkdir(parents=True)

    with pytest.raises(ValueError, match="worktree-local checkpoint"):
        resolve_runtime_paths(
            RuntimePathConfig(
                checkpoint="output_remote/ckpt",
                dataset_jsonls=(),
                artifact_root=str(shared_root / "output" / "analysis" / "study"),
            ),
            repo_root=repo_root,
            shared_root=shared_root,
        )


def test_audit_checkpoint_surface_requires_coord_token_full_checkpoint(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in (
        "coord_tokens.json",
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        (checkpoint / name).write_text("{}", encoding="utf-8")

    audit = audit_checkpoint_surface(checkpoint)

    assert audit["surface"] == "coord_tokens_full_model"
    assert audit["has_coord_tokens_json"] is True
    assert audit["has_model_index"] is True


def test_mine_repeated_desc_cases_prioritizes_requested_dense_classes(
    tmp_path: Path,
) -> None:
    image = tmp_path / "images" / "val2017" / "000000000139.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    dataset = tmp_path / "val.coord.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "file_name": "images/val2017/000000000139.jpg",
                "images": ["images/val2017/000000000139.jpg"],
                "image_id": 139,
                "width": 1000,
                "height": 1000,
                "objects": [
                    {
                        "desc": "book",
                        "bbox_2d": [
                            "<|coord_10|>",
                            "<|coord_10|>",
                            "<|coord_20|>",
                            "<|coord_20|>",
                        ],
                    },
                    {
                        "desc": "book",
                        "bbox_2d": [
                            "<|coord_40|>",
                            "<|coord_10|>",
                            "<|coord_50|>",
                            "<|coord_20|>",
                        ],
                    },
                    {
                        "desc": "chair",
                        "bbox_2d": [
                            "<|coord_60|>",
                            "<|coord_60|>",
                            "<|coord_90|>",
                            "<|coord_90|>",
                        ],
                    },
                ],
            },
            {
                "file_name": "images/val2017/missing.jpg",
                "image_id": 285,
                "width": 1000,
                "height": 1000,
                "objects": [
                    {
                        "desc": "bear",
                        "bbox_2d": [
                            "<|coord_1|>",
                            "<|coord_1|>",
                            "<|coord_2|>",
                            "<|coord_2|>",
                        ],
                    },
                ],
            },
        ],
    )

    cases = mine_repeated_desc_cases(
        dataset,
        CaseSelectionConfig(
            priority_descs=("book", "person"),
            max_cases=8,
            min_same_desc_candidates=2,
            include_sparse_controls=True,
            max_sparse_controls=1,
        ),
    )

    assert cases[0].cohort == "priority_same_desc"
    assert cases[0].desc == "book"
    assert cases[0].candidate_object_indices == (0, 1)
    assert cases[0].target_object_index == 0
    assert cases[0].image_path == image
    repeated_targets = [
        case.target_object_index
        for case in cases
        if case.cohort == "priority_same_desc"
    ]
    assert repeated_targets[:2] == [0, 1]
    assert any(case.cohort == "sparse_single_instance_control" for case in cases)


def test_required_position_roles_match_research_matrix() -> None:
    assert REQUIRED_POSITION_ROLES == (
        "desc_end",
        "desc_closing_quote",
        "field_delimiter",
        "bbox_key",
        "bbox_open_bracket",
        "pre_x1",
        "post_x1",
        "post_y1",
    )


def test_validate_position_inventory_rejects_missing_and_duplicate_roles() -> None:
    with pytest.raises(ValueError, match="pre_x1"):
        validate_position_inventory(
            [
                {"case_id": "case-1", "role": "desc_end", "token_index": 4},
                {"case_id": "case-1", "role": "post_x1", "token_index": 10},
            ]
        )

    rows = [
        {"case_id": "case-1", "role": role, "token_index": idx}
        for idx, role in enumerate(REQUIRED_POSITION_ROLES)
    ]
    rows.append({"case_id": "case-1", "role": "pre_x1", "token_index": 99})
    with pytest.raises(ValueError, match="duplicate"):
        validate_position_inventory(rows)


def test_candidate_alignment_from_distribution_separates_instance_and_boundary_mass() -> (
    None
):
    row = candidate_alignment_from_distribution(
        slot="x1",
        probs={10: 0.35, 11: 0.2, 40: 0.3, 80: 0.1, 500: 0.05},
        candidates=[
            {"label": "target", "bbox_norm1000": [10, 20, 30, 40]},
            {"label": "same_desc_other", "bbox_norm1000": [40, 20, 60, 40]},
        ],
        target_label="target",
        neighbor_radius=1,
    )

    assert row["target_neighborhood_mass"] == pytest.approx(0.55)
    assert row["same_desc_candidate_mass"] == pytest.approx(0.85)
    assert row["non_candidate_mass"] == pytest.approx(0.15)
    assert row["uncertainty_kind"] == "instance_multimodal"


def test_distribute_items_by_shard_is_stable() -> None:
    items = [{"case_id": f"case-{idx}"} for idx in range(10)]

    shard_0 = distribute_items_by_shard(items, shard_index=0, num_shards=3)
    shard_1 = distribute_items_by_shard(items, shard_index=1, num_shards=3)
    shard_2 = distribute_items_by_shard(items, shard_index=2, num_shards=3)

    assert [item["case_id"] for item in shard_0] == [
        "case-0",
        "case-3",
        "case-6",
        "case-9",
    ]
    assert [item["case_id"] for item in shard_1] == ["case-1", "case-4", "case-7"]
    assert [item["case_id"] for item in shard_2] == ["case-2", "case-5", "case-8"]


def test_classify_mechanism_requires_explicit_mixed_when_pre_x1_softens_after_x1() -> (
    None
):
    conclusion = classify_mechanism(
        pre_x1_accuracy=0.58,
        after_x1_accuracy=0.82,
        schema_patch_delta=0.03,
        geometry_patch_delta=0.27,
    )

    assert conclusion == "partial_pre_x1_binding_x1_y1_decisive"


def test_select_hidden_state_layers_maps_model_layers_to_tuple_indices() -> None:
    layers = select_hidden_state_layers(
        hidden_state_count=29,
        requested_model_layers=(0, 1, 12, -2, -1),
    )

    assert layers == (
        {"model_layer": 0, "hidden_state_index": 1},
        {"model_layer": 1, "hidden_state_index": 2},
        {"model_layer": 12, "hidden_state_index": 13},
        {"model_layer": 26, "hidden_state_index": 27},
        {"model_layer": 27, "hidden_state_index": 28},
    )


def test_load_study_config_records_batch8_and_absolute_paths(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "val.coord.jsonl"
    artifact_root = tmp_path / "output" / "analysis" / "study"
    checkpoint.mkdir()
    dataset.write_text("", encoding="utf-8")
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  artifact_root: {artifact_root}
model:
  checkpoint: {checkpoint}
data:
  dataset_jsonls:
    - {dataset}
execution:
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  shard_count: 8
  per_gpu_generation_batch_size: 8
  per_gpu_teacher_forced_batch_size: 8
""".lstrip(),
        encoding="utf-8",
    )

    config = load_study_config(config_path)

    assert config.execution.gpu_ids == (0, 1, 2, 3, 4, 5, 6, 7)
    assert config.execution.per_gpu_generation_batch_size == 8
    assert config.multimodality.coordinate_slots == ("x1", "x2")
    assert config.patching.model_layers == (24, 25, 26, 27)
    assert config.rollout.batch_size == 8
    assert config.paths.checkpoint == checkpoint
    assert config.paths.dataset_jsonls == (dataset,)


def test_run_audit_stage_writes_checkpoint_and_resolved_config(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    dataset = tmp_path / "val.coord.jsonl"
    artifact_root = tmp_path / "artifact"
    checkpoint.mkdir()
    for name in (
        "coord_tokens.json",
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    dataset.write_text("", encoding="utf-8")
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  artifact_root: {artifact_root}
model:
  checkpoint: {checkpoint}
data:
  dataset_jsonls: [{dataset}]
execution:
  gpu_ids: []
  shard_count: 1
  per_gpu_generation_batch_size: 1
  per_gpu_teacher_forced_batch_size: 1
""".lstrip(),
        encoding="utf-8",
    )

    summary = run_study_stage(
        config_path=config_path,
        stage="audit",
        shard_index=None,
        num_shards=None,
    )

    assert summary["stage"] == "audit"
    audit = json.loads((artifact_root / "audit" / "checkpoint_audit.json").read_text())
    assert audit["surface"] == "coord_tokens_full_model"
    resolved = (artifact_root / "config.resolved.yaml").read_text(encoding="utf-8")
    assert str(checkpoint) in resolved
    assert "per_gpu_generation_batch_size: 1" in resolved


def test_run_select_cases_stage_writes_shardable_case_artifacts(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    for name in (
        "coord_tokens.json",
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    image = tmp_path / "images" / "val2017" / "000000000139.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"fake-jpeg")
    dataset = tmp_path / "val.coord.jsonl"
    _write_jsonl(
        dataset,
        [
            {
                "file_name": "images/val2017/000000000139.jpg",
                "image_id": 139,
                "width": 1000,
                "height": 1000,
                "objects": [
                    {
                        "desc": "book",
                        "bbox_2d": [
                            "<|coord_10|>",
                            "<|coord_10|>",
                            "<|coord_20|>",
                            "<|coord_20|>",
                        ],
                    },
                    {
                        "desc": "book",
                        "bbox_2d": [
                            "<|coord_40|>",
                            "<|coord_10|>",
                            "<|coord_50|>",
                            "<|coord_20|>",
                        ],
                    },
                ],
            }
        ],
    )
    artifact_root = tmp_path / "artifact"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  artifact_root: {artifact_root}
model:
  checkpoint: {checkpoint}
data:
  dataset_jsonls: [{dataset}]
execution:
  gpu_ids: []
  shard_count: 1
  per_gpu_generation_batch_size: 1
  per_gpu_teacher_forced_batch_size: 1
case_selection:
  priority_descs: [book]
  max_cases: 4
""".lstrip(),
        encoding="utf-8",
    )

    summary = run_study_stage(
        config_path=config_path,
        stage="select_cases",
        shard_index=0,
        num_shards=1,
    )

    assert summary["stage"] == "select_cases"
    assert summary["num_cases"] == 2
    rows = (
        (artifact_root / "cases" / "selected_cases.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    assert len(rows) == 2
    row = json.loads(rows[0])
    assert row["cohort"] == "priority_same_desc"
    assert row["candidate_object_indices"] == [0, 1]
    assert json.loads(rows[1])["target_object_index"] == 1
    shard_manifest = json.loads(
        (artifact_root / "cases" / "shard_000-of-001.json").read_text()
    )
    assert shard_manifest["num_cases"] == 2


def test_probe_merge_and_report_stages_summarize_existing_artifacts(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = tmp_path / "val.coord.jsonl"
    dataset.write_text("", encoding="utf-8")
    artifact_root = tmp_path / "artifact"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  artifact_root: {artifact_root}
model:
  checkpoint: {checkpoint}
data:
  dataset_jsonls: [{dataset}]
execution:
  gpu_ids: []
  shard_count: 1
  per_gpu_generation_batch_size: 1
  per_gpu_teacher_forced_batch_size: 1
""".lstrip(),
        encoding="utf-8",
    )
    cases = []
    vector_rows = []
    vectors = []
    for idx, target_idx in enumerate([0, 1, 0, 1]):
        objects = [
            {
                "desc": "book",
                "bbox_2d": [
                    "<|coord_100|>",
                    "<|coord_100|>",
                    "<|coord_200|>",
                    "<|coord_200|>",
                ],
            },
            {
                "desc": "book",
                "bbox_2d": [
                    "<|coord_700|>",
                    "<|coord_100|>",
                    "<|coord_800|>",
                    "<|coord_200|>",
                ],
            },
        ]
        case_id = f"case-{idx}"
        cases.append(
            {
                "case_id": case_id,
                "source_jsonl": str(dataset),
                "record_index": idx,
                "image_id": f"image-{idx}",
                "image_path": str(tmp_path / f"image-{idx}.jpg"),
                "width": 1000,
                "height": 1000,
                "desc": "book",
                "target_object_index": target_idx,
                "candidate_object_indices": [0, 1],
                "cohort": "priority_same_desc",
                "selection_reason": "test",
                "objects": objects,
            }
        )
        vector_rows.append(
            {
                "case_id": case_id,
                "role": "pre_x1",
                "model_layer": 0,
                "hidden_state_index": 1,
                "token_index": idx,
            }
        )
        vectors.append([float(target_idx), float(idx), 1.0, 0.0])
    _write_jsonl(artifact_root / "cases" / "selected_cases.jsonl", cases)
    (artifact_root / "cases" / "summary.json").write_text(
        json.dumps({"num_cases": 4, "cohort_counts": {"priority_same_desc": 4}}),
        encoding="utf-8",
    )
    _write_jsonl(
        artifact_root / "hidden_states" / "hidden_vector_rows_shard_000-of-001.jsonl",
        vector_rows,
    )
    np.savez_compressed(
        artifact_root / "hidden_states" / "hidden_vectors_shard_000-of-001.npz",
        vectors=np.asarray(vectors, dtype=np.float32),
    )
    _write_jsonl(
        artifact_root / "multimodality" / "coord_multimodality_shard_000-of-001.jsonl",
        [
            {
                "case_id": "case-0",
                "slot": "x1",
                "entropy": 1.0,
                "target_neighborhood_mass": 0.7,
                "other_candidate_mass": 0.1,
                "same_desc_candidate_mass": 0.8,
                "non_candidate_mass": 0.2,
                "uncertainty_kind": "boundary_style_uncertainty",
                "top_coord_bins": [{"bin": 100, "prob": 0.5}],
                "candidate_rows": [
                    {"label": "target", "center_bin": 100, "neighborhood_mass": 0.7},
                    {
                        "label": "candidate_1",
                        "center_bin": 700,
                        "neighborhood_mass": 0.1,
                    },
                ],
            }
        ],
    )
    _write_jsonl(
        artifact_root / "multimodality" / "pre_x1_multimodality_shard_000-of-001.jsonl",
        [
            {
                "case_id": "legacy-duplicate",
                "slot": "x1",
                "entropy": 9.0,
                "target_neighborhood_mass": 0.0,
                "other_candidate_mass": 0.0,
                "same_desc_candidate_mass": 0.0,
                "non_candidate_mass": 1.0,
                "uncertainty_kind": "diffuse_or_unassigned",
                "top_coord_bins": [],
                "candidate_rows": [],
            }
        ],
    )

    probe_summary = run_study_stage(config_path=config_path, stage="binding_probe")
    merge_summary = run_study_stage(
        config_path=config_path, stage="merge_multimodality"
    )
    report_summary = run_study_stage(config_path=config_path, stage="report")

    assert probe_summary["status"] == "ok"
    assert probe_summary["num_prediction_rows"] == 4
    assert merge_summary["status"] == "ok"
    assert merge_summary["num_rows"] == 1
    assert merge_summary["slots"]["x1"]["target_strict_best_rate"] == 1.0
    assert report_summary["status"] == "ok"
    assert (artifact_root / "report" / "report.md").is_file()


def test_prepare_and_merge_rollout_shards_use_yaml_first_infer_configs(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = tmp_path / "val.coord.jsonl"
    dataset.write_text("", encoding="utf-8")
    artifact_root = tmp_path / "artifact"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  artifact_root: {artifact_root}
model:
  checkpoint: {checkpoint}
data:
  dataset_jsonls: [{dataset}]
execution:
  gpu_ids: [0, 1]
  shard_count: 2
  per_gpu_generation_batch_size: 8
  per_gpu_teacher_forced_batch_size: 1
rollout:
  batch_size: 8
  max_new_tokens: 128
""".lstrip(),
        encoding="utf-8",
    )
    cases = []
    for idx in range(4):
        cases.append(
            {
                "case_id": f"case-{idx}",
                "source_jsonl": str(dataset),
                "record_index": idx,
                "image_id": f"image-{idx}",
                "image_path": str(tmp_path / f"image-{idx}.jpg"),
                "width": 1000,
                "height": 1000,
                "desc": "person",
                "target_object_index": 0,
                "candidate_object_indices": [0, 1],
                "cohort": "priority_same_desc",
                "selection_reason": "test",
                "objects": [
                    {
                        "desc": "person",
                        "bbox_2d": [
                            "<|coord_100|>",
                            "<|coord_100|>",
                            "<|coord_200|>",
                            "<|coord_200|>",
                        ],
                    },
                    {
                        "desc": "person",
                        "bbox_2d": [
                            "<|coord_700|>",
                            "<|coord_100|>",
                            "<|coord_800|>",
                            "<|coord_200|>",
                        ],
                    },
                ],
            }
        )
    _write_jsonl(artifact_root / "cases" / "selected_cases.jsonl", cases)

    prep = run_study_stage(config_path=config_path, stage="prepare_rollout_shards")

    assert prep["status"] == "ok"
    assert prep["num_shards"] == 2
    infer_config = (
        artifact_root / "rollout" / "configs" / "shard_000-of-002.yaml"
    ).read_text(encoding="utf-8")
    assert "batch_size: 8" in infer_config
    assert "max_new_tokens: 128" in infer_config
    shard_run = artifact_root / "rollout" / "runs" / "shard_000-of-002"
    _write_jsonl(
        shard_run / "gt_vs_pred.jsonl",
        [
            {
                "image": "a.jpg",
                "width": 1000,
                "height": 1000,
                "coord_mode": "pixel",
                "metadata": {"instance_binding_case_id": "case-0", "desc": "person"},
                "pred": [
                    {"desc": "person", "points": [100, 100, 200, 200]},
                    {"desc": "person", "points": [102, 102, 202, 202]},
                ],
                "errors": [],
            }
        ],
    )
    shard_run = artifact_root / "rollout" / "runs" / "shard_001-of-002"
    _write_jsonl(
        shard_run / "gt_vs_pred.jsonl",
        [
            {
                "image": "b.jpg",
                "width": 1000,
                "height": 1000,
                "coord_mode": "pixel",
                "metadata": {"instance_binding_case_id": "case-1", "desc": "person"},
                "pred": [{"desc": "person", "points": [700, 100, 800, 200]}],
                "errors": [],
            }
        ],
    )

    merged = run_study_stage(config_path=config_path, stage="merge_rollout_shards")

    assert merged["status"] == "ok"
    assert merged["label_counts"]["duplicate_collapse_like"] == 1
    assert merged["label_counts"]["healthy_single_same_desc"] == 1


def test_report_marks_first_pass_converged_with_rollout_contrast(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    dataset = tmp_path / "val.coord.jsonl"
    dataset.write_text("", encoding="utf-8")
    artifact_root = tmp_path / "artifact"
    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        f"""
run:
  artifact_root: {artifact_root}
model:
  checkpoint: {checkpoint}
data:
  dataset_jsonls: [{dataset}]
execution:
  gpu_ids: []
  shard_count: 1
  per_gpu_generation_batch_size: 1
  per_gpu_teacher_forced_batch_size: 1
""".lstrip(),
        encoding="utf-8",
    )
    _write_json(
        artifact_root / "cases" / "summary.json",
        {
            "num_cases": 4,
            "cohort_counts": {"priority_same_desc": 4},
            "target_ordinal_hist": {"0": 2, "1": 2},
        },
    )
    _write_json(
        artifact_root / "binding_probe" / "summary.json",
        {"best_pre_x1_accuracy": 0.421875, "best_after_x1_accuracy": 0.59375},
    )
    _write_json(
        artifact_root / "merge_multimodality" / "summary.json",
        {
            "slots": {
                "x1": {
                    "target_strict_best_rate": 0.546875,
                    "mean_target_neighborhood_mass": 0.167873,
                    "mean_other_candidate_mass": 0.145206,
                }
            }
        },
    )
    _write_json(
        artifact_root / "merge_patching" / "summary.json",
        {
            "intervention_summaries": {
                "schema_context": {
                    "mean_abs_margin_delta": 0.067651,
                    "top_candidate_flip_rate": 0.375,
                },
                "previous_geometry": {"mean_abs_margin_delta": 0.001629},
                "current_desc": {"mean_abs_margin_delta": 0.001123},
            }
        },
    )
    _write_json(
        artifact_root / "good_bad_panels" / "summary.json",
        {
            "label_counts": {
                "good_basin_proxy": 2,
                "bad_basin_proxy": 2,
            }
        },
    )
    _write_jsonl(
        artifact_root / "good_bad_panels" / "good_bad_case_panels.jsonl",
        [
            {"case_id": "case-0", "basin_label": "good_basin_proxy"},
            {"case_id": "case-1", "basin_label": "bad_basin_proxy"},
        ],
    )
    rollout_labels_path = (
        artifact_root / "rollout_failure_split" / "rollout_labels.jsonl"
    )
    _write_jsonl(
        rollout_labels_path,
        [
            {"case_id": "case-0", "rollout_label": "healthy_same_desc_multi"},
            {"case_id": "case-1", "rollout_label": "duplicate_collapse_like"},
        ],
    )
    _write_json(
        artifact_root / "rollout_failure_split" / "summary.json",
        {
            "status": "ok",
            "label_counts": {
                "healthy_same_desc_multi": 1,
                "duplicate_collapse_like": 1,
            },
            "rollout_labels": str(rollout_labels_path),
        },
    )

    report_summary = run_study_stage(config_path=config_path, stage="report")

    assert report_summary["convergence_status"] == (
        "converged_first_pass_mixed_soft_pre_x1_coordinate_hardening"
    )
    assert report_summary["evidence"]["rollout_healthy_count"] == 1
    assert report_summary["evidence"]["rollout_failure_like_count"] == 1
    assert report_summary["evidence"]["rollout_basin_crosstab"] == {
        "bad_basin_proxy": {"duplicate_collapse_like": 1},
        "good_basin_proxy": {"healthy_same_desc_multi": 1},
    }


def test_best_other_candidate_from_alignment_prefers_highest_non_target() -> None:
    donor = _best_other_candidate_from_alignment(
        {
            "candidate_rows": [
                {"label": "target", "object_index": 0, "neighborhood_mass": 0.42},
                {
                    "label": "candidate_3",
                    "object_index": 3,
                    "neighborhood_mass": 0.71,
                },
                {
                    "label": "candidate_5",
                    "object_index": 5,
                    "neighborhood_mass": 0.63,
                },
            ]
        }
    )

    assert donor == {"label": "candidate_3", "object_index": 3, "mass": 0.71}


def test_best_other_candidate_from_alignment_parses_candidate_label() -> None:
    donor = _best_other_candidate_from_alignment(
        {
            "candidate_rows": [
                {"label": "target", "neighborhood_mass": 0.42},
                {"label": "candidate_18", "neighborhood_mass": 0.71},
            ]
        }
    )

    assert donor == {"label": "candidate_18", "object_index": 18, "mass": 0.71}


def test_summarize_donor_patching_rows_reports_span_effects() -> None:
    summary = _summarize_donor_patching_rows(
        [
            {
                "span": "schema_context",
                "donor_mass_delta": 0.20,
                "target_mass_delta": -0.10,
                "top_candidate_flipped_to_donor": True,
                "patched_top_is_donor": True,
                "num_position_pairs": 4,
            },
            {
                "span": "schema_context",
                "donor_mass_delta": 0.10,
                "target_mass_delta": -0.05,
                "top_candidate_flipped_to_donor": False,
                "patched_top_is_donor": True,
                "num_position_pairs": 4,
            },
            {
                "span": "current_desc",
                "donor_mass_delta": 0.01,
                "target_mass_delta": 0.0,
                "top_candidate_flipped_to_donor": False,
                "patched_top_is_donor": False,
                "num_position_pairs": 2,
            },
        ]
    )

    schema = summary["span_summaries"]["schema_context"]
    assert schema["num_rows"] == 2
    assert schema["mean_donor_mass_delta"] == pytest.approx(0.15)
    assert schema["mean_target_mass_delta"] == pytest.approx(-0.075)
    assert schema["flip_to_donor_rate"] == pytest.approx(0.5)
    assert schema["patched_top_is_donor_rate"] == pytest.approx(1.0)
    assert schema["mean_num_position_pairs"] == pytest.approx(4.0)


def test_fine_schema_relative_spans_from_inventory_splits_transition() -> None:
    spans = _fine_schema_relative_spans_from_inventory(
        {
            "roles": {
                "desc_end": 10,
                "desc_closing_quote": 11,
                "field_delimiter": 12,
                "bbox_key": 15,
                "bbox_open_bracket": 18,
                "pre_x1": 18,
            }
        }
    )

    assert spans["schema_context"] == [11, 12, 13, 14, 15, 16, 17]
    assert spans["desc_closing_quote"] == [11]
    assert spans["field_delimiter"] == [12]
    assert spans["bbox_key_region"] == [15, 16]
    assert spans["bbox_colon_region"] == [17]
    assert spans["bbox_open_bracket"] == [18]
    assert spans["immediate_pre_x1"] == [18]


def test_select_core_donor_spec_covers_control_policies() -> None:
    objects = [
        {"desc": "book", "bbox_2d": ["<|coord_1|>"] * 4},
        {"desc": "book", "bbox_2d": ["<|coord_2|>"] * 4},
        {"desc": "book", "bbox_2d": ["<|coord_3|>"] * 4},
    ]
    case = {
        "case_id": "case-0",
        "image_id": "image-0",
        "desc": "book",
        "target_object_index": 0,
        "candidate_object_indices": [0, 1, 2],
        "objects": objects,
    }
    same_desc_wrong_image = {
        "case_id": "case-1",
        "image_id": "image-1",
        "desc": "book",
        "target_object_index": 0,
        "candidate_object_indices": [0],
        "objects": [{"desc": "book", "bbox_2d": ["<|coord_4|>"] * 4}],
    }
    any_desc_wrong_image = {
        "case_id": "case-2",
        "image_id": "image-2",
        "desc": "person",
        "target_object_index": 0,
        "candidate_object_indices": [0],
        "objects": [{"desc": "person", "bbox_2d": ["<|coord_5|>"] * 4}],
    }
    alignment = {
        "candidate_rows": [
            {"label": "target", "object_index": 0, "neighborhood_mass": 0.4},
            {"label": "candidate_1", "object_index": 1, "neighborhood_mass": 0.5},
            {"label": "candidate_2", "object_index": 2, "neighborhood_mass": 0.9},
        ]
    }
    all_cases = [case, same_desc_wrong_image, any_desc_wrong_image]

    best = _select_core_donor_spec(
        policy="same_image_best_competitor",
        case=case,
        all_cases=all_cases,
        baseline_alignment=alignment,
        seed=7,
    )
    self_noop = _select_core_donor_spec(
        policy="self_noop",
        case=case,
        all_cases=all_cases,
        baseline_alignment=alignment,
        seed=7,
    )
    wrong_same = _select_core_donor_spec(
        policy="wrong_image_same_desc",
        case=case,
        all_cases=all_cases,
        baseline_alignment=alignment,
        seed=7,
    )
    wrong_any = _select_core_donor_spec(
        policy="wrong_image_any_desc",
        case=case,
        all_cases=[case, any_desc_wrong_image],
        baseline_alignment=alignment,
        seed=7,
    )

    assert best is not None
    assert best["donor_case_id"] == "case-0"
    assert best["donor_object_index"] == 2
    assert best["donor_label"] == "candidate_2"
    assert best["donor_in_target_candidates"] is True
    assert self_noop is not None
    assert self_noop["donor_label"] == "target"
    assert wrong_same is not None
    assert wrong_same["donor_case_id"] == "case-1"
    assert wrong_same["donor_in_target_candidates"] is False
    assert wrong_any is not None
    assert wrong_any["donor_case_id"] == "case-2"
    assert wrong_any["donor_object_index"] == 0


def test_summarize_core_diagnosis_rows_ignores_missing_donor_delta() -> None:
    summary = _summarize_core_diagnosis_rows(
        [
            {
                "donor_policy": "same_image_best_competitor",
                "span": "bbox_open_bracket",
                "layer_set": "late",
                "target_mass_delta": -0.10,
                "donor_mass_delta": 0.05,
                "coord_kl_from_baseline": 0.20,
                "top_candidate_flipped_to_donor": True,
            },
            {
                "donor_policy": "same_image_best_competitor",
                "span": "bbox_open_bracket",
                "layer_set": "late",
                "target_mass_delta": -0.20,
                "donor_mass_delta": 0.01,
                "coord_kl_from_baseline": 0.10,
                "top_candidate_flipped_to_donor": False,
            },
            {
                "donor_policy": "wrong_image_same_desc",
                "span": "bbox_open_bracket",
                "layer_set": "late",
                "target_mass_delta": -0.30,
                "donor_mass_delta": None,
                "coord_kl_from_baseline": 0.50,
                "top_candidate_flipped_to_donor": False,
            },
        ]
    )

    same = summary["policy_span_layer_summaries"][
        "same_image_best_competitor|bbox_open_bracket|late"
    ]
    wrong = summary["policy_span_layer_summaries"][
        "wrong_image_same_desc|bbox_open_bracket|late"
    ]
    assert same["num_rows"] == 2
    assert same["mean_target_mass_delta"] == pytest.approx(-0.15)
    assert same["mean_donor_mass_delta"] == pytest.approx(0.03)
    assert same["flip_to_donor_rate"] == pytest.approx(0.5)
    assert same["mean_coord_kl_from_baseline"] == pytest.approx(0.15)
    assert wrong["num_rows"] == 1
    assert wrong["mean_donor_mass_delta"] is None
    assert wrong["mean_target_mass_delta"] == pytest.approx(-0.30)


def test_stage_shard_jsonl_paths_exclude_prior_merged_outputs(tmp_path: Path) -> None:
    stage_dir = tmp_path / "core_diagnosis"
    stage_dir.mkdir()
    shard = stage_dir / "core_patch_results_shard_000-of-008.jsonl"
    merged = stage_dir / "core_patch_results_merged.jsonl"
    token_edit = stage_dir / "core_token_edit_results_shard_000-of-008.jsonl"
    shard.write_text("{}", encoding="utf-8")
    merged.write_text("{}", encoding="utf-8")
    token_edit.write_text("{}", encoding="utf-8")

    assert _stage_shard_jsonl_paths(stage_dir, "core_patch_results") == [shard]
