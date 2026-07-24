from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import scripts.research.analyze_trajectory_owner_set_admission_census as census
import scripts.research.assemble_constant_dose_breadth_state_banks as adapter_module
from scripts.research.analyze_trajectory_owner_set_admission_census import (
    _analyze_image_candidates,
    _candidate_receipt,
    _finalize_staging,
    _materialize_source_snapshot,
    _summarize_records,
    _validate_frozen_panel_bindings,
    _validate_panel_manifests,
)
from scripts.research.assemble_positive_path_imitation_state_bank import (
    GEOMETRY_IOU_THRESHOLD,
)
from src.config.fingerprint import sha256_file


def _candidate(
    candidate_id: str,
    token: str,
    owners: list[str],
    sequence: list[str],
    *,
    duplicate: int = 0,
) -> dict[str, Any]:
    safety = {
        "duplicate": duplicate,
        "malformed": 0,
        "confirmed_false": 0,
        "semantic_error": 0,
        "unknown": 0,
        "premature_stop": 0,
    }
    return {
        "candidate_id": candidate_id,
        "trajectory_id": candidate_id,
        "generated_token_ids_sha256": token * 64,
        "final_owner_ids": sorted(owners),
        "ordered_unique_owner_sequence": sequence,
        "first_owner_id": sequence[0] if sequence else None,
        "safety_counts": safety,
        "geometry": {"all_evaluated_rows_trusted": True},
        "eligible": True,
        "exclusion_reasons": [],
        "exact_token_representative": True,
        "exact_token_duplicate_of": None,
    }


def _row(
    index: int,
    owner: str,
    category: str,
    iou: float,
    *,
    duplicate: bool = False,
) -> dict[str, object]:
    result: dict[str, object] = {
        "generated_row_index": index,
        "prediction_id": f"row-{index}",
        "category": category,
        "entity_status": "duplicate" if duplicate else "verified_owner",
    }
    if duplicate:
        result.update({"candidate_owner_id": owner, "candidate_owner_iou": iou})
    else:
        result.update({"owner_id": owner, "intersection_over_union": iou})
    return result


def _receipt(
    *rows: dict[str, object],
    route_id: str = "sample-00",
    decode_mode: str = "sampled",
    stop_reason: str = "im_end",
    source_status: str | None = None,
    sampled_status: str = "accepted_natural_end",
    raw_hash: str = "b" * 64,
    raw_count: int = 4,
) -> dict[str, Any]:
    owners = [
        {"owner_id": "1:A", "category": "cat", "bbox": [0, 0, 10, 10]},
        {"owner_id": "1:B", "category": "dog", "bbox": [10, 0, 20, 10]},
    ]
    assignment = {
        "row_assignment_receipts": list(rows),
        "malformed_row_count": 0,
        "row_counts": {
            "duplicate": sum(row["entity_status"] == "duplicate" for row in rows),
            "malformed": 0,
            "unsupported_hallucination": 0,
            "semantic_error": 0,
            "unresolved": 0,
        },
    }
    evidence = {
        "decode_mode": decode_mode,
        "seed": 0,
        "stop_reason": stop_reason,
        "parser": {
            "parse_status": "accepted",
            "valid_prediction_count": len(rows),
            "dropped_prediction_count": 0,
        },
    }
    projected_ids = [1, 2, 3]
    projected_hash = census.token_ids_sha256(projected_ids)
    route: dict[str, object] = {
        "decode_mode": decode_mode,
        "sample_index": 0,
        "stop_reason": stop_reason,
        "generated_token_ids": projected_ids,
        "generated_token_ids_sha256": projected_hash,
    }
    projection_identity = {
        "raw_generated_token_ids_sha256": raw_hash,
        "raw_generated_token_count": raw_count,
        "projected_token_ids_sha256": projected_hash,
        "projected_token_count": len(projected_ids),
    }
    if source_status is not None:
        route["_source_b16_provenance"] = {
            "status": source_status,
            **projection_identity,
        }
    elif decode_mode == "sampled":
        route["_sampled_b16_provenance"] = {
            "status": sampled_status,
            **projection_identity,
        }
    return _candidate_receipt(
        route_id=route_id,
        route_row=route,
        route_evidence=evidence,
        assignment=assignment,
        owners=owners,
    )


def test_primary_alias_predicate_uses_same_frontier_class_and_universal_edge() -> None:
    candidates = [
        _candidate("sample-00", "a", ["A"], ["A"], duplicate=2),
        _candidate("sample-01", "b", ["A", "B"], ["A", "B"], duplicate=0),
        _candidate("sample-02", "c", ["A", "B"], ["B", "A"], duplicate=1),
        # Exact-token alias is retained diagnostically but not counted as a
        # third exact serialization.
        _candidate("sample-03", "b", ["A", "B"], ["A", "B"], duplicate=0),
    ]

    result = _analyze_image_candidates("1", candidates)

    assert result["strict_semantic_edge_count"] == 1
    assert result["semantic_edges"][0]["universal_safety"] is True
    assert result["exact_token_duplicate_count"] == 1
    assert result["admission"]["image_level_frontier_natural_alias"] is True
    assert result["admission"]["primary_natural_alias_admitted"] is True
    frontier = result["positive_frontier_alias_metrics"]["per_class"][0]
    assert frontier["exact_serialization_count"] == 2
    assert frontier["distinct_first_owner_count"] == 2

    reversed_result = _analyze_image_candidates("1", list(reversed(candidates)))
    assert result == reversed_result


def test_universal_edge_does_not_borrow_one_favorable_alias() -> None:
    candidates = [
        _candidate("sample-00", "a", ["A"], ["A"], duplicate=2),
        _candidate("sample-01", "b", ["A", "B"], ["A", "B"], duplicate=0),
        _candidate("sample-02", "c", ["A", "B"], ["B", "A"], duplicate=3),
    ]

    result = _analyze_image_candidates("1", candidates)

    assert result["strict_semantic_edge_count"] == 0
    diagnostic = result["strict_owner_set_inclusion_diagnostics"][0]
    assert diagnostic == {
        "higher_class_id": diagnostic["higher_class_id"],
        "lower_class_id": diagnostic["lower_class_id"],
        "alias_pair_count": 2,
        "existential_safe_alias_pair_count": 1,
        "universal_safety": False,
    }
    high = next(item for item in result["outcome_classes"] if item["owner_count"] == 2)
    assert high["minimum_alias_safety_counts"]["duplicate"] == 0
    assert high["maximum_alias_safety_counts"]["duplicate"] == 3
    assert len(high["alias_safety_vectors"]) == 2
    assert result["admission"]["primary_natural_alias_admitted"] is False


def test_exact_token_signature_ignores_route_local_prediction_ids() -> None:
    left = _candidate("sample-00", "a", ["A"], ["A"])
    right = _candidate("sample-01", "a", ["A"], ["A"])
    common = {
        "generated_row_index": 0,
        "entity_status": "verified_owner",
        "owner_id": "A",
        "intersection_over_union": 0.9,
        "entity_owner_iou_at_least_0_50": True,
        "trusted_exact_geometry": True,
        "geometry_reason": "multi_instance_iou_at_least_0.75",
    }
    left["geometry"] = {
        "trusted_exact_geometry_iou_threshold": 0.75,
        "all_evaluated_rows_trusted": True,
        "untrusted_row_count": 0,
        "rows": [{**common, "prediction_id": "sample-00:row-0"}],
    }
    right["geometry"] = {
        "trusted_exact_geometry_iou_threshold": 0.75,
        "all_evaluated_rows_trusted": True,
        "untrusted_row_count": 0,
        "rows": [{**common, "prediction_id": "sample-01:row-0"}],
    }

    result = _analyze_image_candidates("1", [left, right])

    assert result["exact_token_duplicate_count"] == 1
    assert result["outcome_classes"][0]["exact_serialization_count"] == 1


def test_projected_dedup_preserves_distinct_raw_completion_identities() -> None:
    left = _receipt(
        _row(0, "1:A", "cat", 0.90),
        route_id="sample-00",
        raw_hash="b" * 64,
    )
    right = _receipt(
        _row(0, "1:A", "cat", 0.90),
        route_id="sample-01",
        raw_hash="c" * 64,
    )

    result = _analyze_image_candidates("1", [left, right])

    assert result["exact_token_duplicate_count"] == 1
    assert {
        item["raw_generated_token_ids_sha256"] for item in result["candidates"]
    } == {"b" * 64, "c" * 64}
    assert len(
        {
            item["projected_generated_token_ids_sha256"]
            for item in result["candidates"]
        }
    ) == 1

    with pytest.raises(adapter_module.AssemblyError, match="raw/projected token provenance"):
        _receipt(_row(0, "1:A", "cat", 0.90), raw_count=2)


def test_all_transitive_edges_and_first_owner_orphan_are_visible() -> None:
    transitive = _analyze_image_candidates(
        "1",
        [
            _candidate("sample-00", "a", ["A"], ["A"]),
            _candidate("sample-01", "b", ["A", "B"], ["A", "B"]),
            _candidate("sample-02", "c", ["A", "B", "C"], ["A", "B", "C"]),
        ],
    )
    assert transitive["strict_semantic_edge_count"] == 3

    orphan = _analyze_image_candidates(
        "1",
        [
            _candidate("sample-00", "a", ["B"], ["B"]),
            _candidate("sample-01", "b", ["A", "B"], ["A", "B"]),
        ],
    )
    assert orphan["first_owner_orphaned_edge_count"] == 1
    assert orphan["semantic_edges"][0]["orphaned_lower_first_owner_ids"] == ["B"]
    assert orphan["admission"]["has_admissible_strict_edge"] is False


def test_geometry_gate_excludes_entity_only_match_and_checks_duplicate_geometry() -> None:
    entity_only = _receipt(_row(0, "1:A", "cat", 0.60))
    assert entity_only["final_owner_ids"] == ["1:A"]
    assert entity_only["eligible"] is False
    assert entity_only["geometry"]["rows"][0]["entity_owner_iou_at_least_0_50"] is True
    assert entity_only["geometry"]["rows"][0]["trusted_exact_geometry"] is False
    assert "geometry_untrusted" in entity_only["exclusion_reasons"]

    trusted_duplicate = _receipt(
        _row(0, "1:A", "cat", GEOMETRY_IOU_THRESHOLD, duplicate=True),
        _row(1, "1:A", "cat", 0.90),
        _row(2, "1:B", "dog", 0.90),
    )
    assert trusted_duplicate["eligible"] is True
    assert trusted_duplicate["duplicate_count"] == 1
    assert trusted_duplicate["ordered_unique_owner_sequence"] == ["1:A", "1:B"]

    weak_duplicate = _receipt(
        _row(0, "1:A", "cat", 0.90),
        _row(1, "1:A", "cat", 0.70, duplicate=True),
    )
    assert weak_duplicate["eligible"] is False
    assert weak_duplicate["geometry"]["untrusted_row_count"] == 1


def test_premature_stop_is_natural_semantic_completion_with_uncovered_owner() -> None:
    source_natural_uncovered = _receipt(
        _row(0, "1:A", "cat", 0.90),
        route_id="source-b16",
        decode_mode="greedy",
        source_status="accepted_natural_end",
    )
    assert source_natural_uncovered["natural_semantic_stop"] is True
    assert source_natural_uncovered["uncovered_trusted_owner_ids_at_stop"] == ["1:B"]
    assert source_natural_uncovered["premature_stop_count"] == 1
    assert source_natural_uncovered["eligible"] is True

    source_natural_complete = _receipt(
        _row(0, "1:A", "cat", 0.90),
        _row(1, "1:B", "dog", 0.90),
        route_id="source-b16",
        decode_mode="greedy",
        source_status="accepted_natural_end",
    )
    assert source_natural_complete["uncovered_trusted_owner_ids_at_stop"] == []
    assert source_natural_complete["premature_stop_count"] == 0

    source_budget = _receipt(
        _row(0, "1:A", "cat", 0.90),
        route_id="source-b16",
        decode_mode="greedy",
        source_status="accepted_budget",
    )
    assert source_budget["natural_semantic_stop"] is False
    assert source_budget["premature_stop_count"] == 0

    sampled_natural = _receipt(_row(0, "1:A", "cat", 0.90))
    assert sampled_natural["premature_stop_count"] == 1

    sampled_budget = _receipt(
        _row(0, "1:A", "cat", 0.90), sampled_status="accepted_budget"
    )
    assert sampled_budget["natural_semantic_stop"] is False
    assert sampled_budget["premature_stop_count"] == 0

    sampled_non_natural = _receipt(
        _row(0, "1:A", "cat", 0.90),
        stop_reason="length",
        sampled_status="failed_invalid_before_budget",
    )
    assert sampled_non_natural["natural_semantic_stop"] is False
    assert sampled_non_natural["premature_stop_count"] == 0
    assert sampled_non_natural["eligible"] is False
    assert "sampled_b16_projection_status:failed_invalid_before_budget" in sampled_non_natural[
        "exclusion_reasons"
    ]


def test_summary_separates_censoring_from_structural_populations() -> None:
    fully_candidates = [
        _candidate("sample-00", "a", ["A"], ["A"], duplicate=2),
        _candidate("sample-01", "b", ["A", "B"], ["A", "B"]),
        _candidate("sample-02", "c", ["A", "B"], ["B", "A"]),
    ]
    for index, token in enumerate("defghijklmnopq", 3):
        fully_candidates.append(
            _candidate(f"sample-{index:02d}", token, ["A", "B"], ["A", "B"])
        )
    assert len(fully_candidates) == 17
    fully = _analyze_image_candidates("1", fully_candidates)

    excluded = _candidate("sample-01", "z", ["A"], ["A"])
    excluded["eligible"] = False
    excluded["exclusion_reasons"] = ["unknown_row"]
    excluded["safety_counts"]["unknown"] = 1
    censored = _analyze_image_candidates(
        "2",
        [
            _candidate("sample-00", "y", ["A"], ["A"]),
            _candidate("sample-02", "x", ["A", "B"], ["A", "B"]),
            excluded,
        ],
    )
    summary = _summarize_records([fully, censored])

    assert summary["fully_adjudicable_image_count"] == 1
    assert summary["censored_image_count"] == 1
    assert summary["failure_cause_split"]["censoring"][
        "candidate_exclusion_reason_image_counts"
    ] == {"unknown_row": 1}
    structural = summary["failure_cause_split"]["structural_scarcity"]
    assert structural["fully_adjudicable_images"]["image_count"] == 1
    assert structural["images_with_at_least_two_eligible_candidates"]["image_count"] == 2
    assert summary["feasibility"]["bounded_conclusion"] == (
        "does_not_support_256_image_screen_under_frozen_admission"
    )
    assert summary["feasibility"]["causal_failure_diagnosis_chosen"] is False


def test_manifest_binding_checks_manifest_and_batch_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(census, "EXPECTED_MANIFEST_COUNT_PER_ROOT", 1)
    monkeypatch.setattr(census, "EXPECTED_BATCH_COUNT_PER_ROOT", 1)
    monkeypatch.setattr(census, "EXPECTED_POOL_IMAGE_COUNT", 16)
    worker = tmp_path / "worker-00-of-01"
    worker.mkdir()
    artifact = worker / "sampled-batch-00000.json"
    artifact.write_text("{}\n", encoding="utf-8")
    manifest = {
        "status": "completed",
        "worker_index": 0,
        "worker_count": 1,
        "batches": [
            {
                "batch_index": 0,
                "image_ids": list(range(16)),
                "artifacts": {
                    "sampled": {"path": artifact.name, "sha256": sha256_file(artifact)}
                },
            }
        ],
    }
    (worker / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    expected_ids = {str(index) for index in range(16)}
    receipt = _validate_panel_manifests(
        tmp_path, mode="sampled", expected_pool_ids=expected_ids
    )
    assert receipt["manifest_count"] == receipt["batch_count"] == 1
    assert receipt["ordered_image_inventory_sha256"] == census.sha256_json(
        sorted(expected_ids, key=int)
    )

    artifact.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(adapter_module.AssemblyError, match="batch hash mismatch"):
        _validate_panel_manifests(
            tmp_path, mode="sampled", expected_pool_ids=expected_ids
        )


def test_manifest_binding_rejects_same_count_alien_image_id(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(census, "EXPECTED_MANIFEST_COUNT_PER_ROOT", 1)
    monkeypatch.setattr(census, "EXPECTED_BATCH_COUNT_PER_ROOT", 1)
    worker = tmp_path / "worker-00-of-01"
    worker.mkdir()
    artifact = worker / "sampled-batch-00000.json"
    artifact.write_text("{}\n", encoding="utf-8")
    alien_ids = [*range(15), 999]
    manifest = {
        "status": "completed",
        "worker_index": 0,
        "worker_count": 1,
        "batches": [
            {
                "batch_index": 0,
                "image_ids": alien_ids,
                "artifacts": {
                    "sampled": {"path": artifact.name, "sha256": sha256_file(artifact)}
                },
            }
        ],
    }
    (worker / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(adapter_module.AssemblyError, match=r"alien=\['999'\]"):
        _validate_panel_manifests(
            tmp_path,
            mode="sampled",
            expected_pool_ids={str(index) for index in range(16)},
        )


def test_frozen_panel_binding_rejects_alternate_root_and_joint_model_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sampled = tmp_path / "sampled"
    source = tmp_path / "source"
    alternate = tmp_path / "alternate"
    sampled.mkdir()
    source.mkdir()
    alternate.mkdir()
    monkeypatch.setattr(census, "FROZEN_SAMPLED_ROOT", sampled)
    monkeypatch.setattr(census, "FROZEN_SOURCE_ROOT", source)
    monkeypatch.setattr(census, "FROZEN_SAMPLED_MANIFEST_SET_SHA256", "sampled-set")
    monkeypatch.setattr(census, "FROZEN_SOURCE_MANIFEST_SET_SHA256", "source-set")
    monkeypatch.setattr(census, "FROZEN_EXECUTION_MODEL_IDENTITY_SHA256", "execution")
    monkeypatch.setattr(census, "FROZEN_TOKENIZER_IDENTITY_SHA256", "tokenizer")
    manifests = {
        "sampled": {"manifest_set_sha256": "sampled-set"},
        "source": {"manifest_set_sha256": "source-set"},
    }
    adapter = {
        "execution_model_identity_sha256": "execution",
        "tokenizer_identity_sha256": "tokenizer",
    }

    _validate_frozen_panel_bindings(
        sampled_root=sampled.resolve(),
        source_root=source.resolve(),
        sampled_manifests=manifests["sampled"],
        source_manifests=manifests["source"],
        adapter=adapter,
    )
    with pytest.raises(adapter_module.AssemblyError, match="sampled root"):
        _validate_frozen_panel_bindings(
            sampled_root=alternate.resolve(),
            source_root=source.resolve(),
            sampled_manifests=manifests["sampled"],
            source_manifests=manifests["source"],
            adapter=adapter,
        )
    jointly_changed = {
        "execution_model_identity_sha256": "jointly-changed",
        "tokenizer_identity_sha256": "tokenizer",
    }
    with pytest.raises(adapter_module.AssemblyError, match="execution-model identity"):
        _validate_frozen_panel_bindings(
            sampled_root=sampled.resolve(),
            source_root=source.resolve(),
            sampled_manifests=manifests["sampled"],
            source_manifests=manifests["source"],
            adapter=jointly_changed,
        )


def test_frozen_paths_reject_byte_identical_alternate_candidate_pool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate_pool = tmp_path / "candidate-pool.jsonl"
    alternate_pool = tmp_path / "alternate-candidate-pool.jsonl"
    split_receipt = tmp_path / "split-receipt.json"
    sampled_root = tmp_path / "sampled"
    source_root = tmp_path / "source"
    output_root = tmp_path / "output"
    split_paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("train_candidate", "development", "heldout")
    }
    candidate_pool.write_text("same bytes\n", encoding="utf-8")
    alternate_pool.write_bytes(candidate_pool.read_bytes())
    split_receipt.write_text("{}\n", encoding="utf-8")
    sampled_root.mkdir()
    source_root.mkdir()
    for path in split_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(census, "FROZEN_CANDIDATE_POOL_PATH", candidate_pool)
    monkeypatch.setattr(census, "FROZEN_SPLIT_RECEIPT_PATH", split_receipt)
    monkeypatch.setattr(census, "FROZEN_SPLIT_OUTPUT_PATHS", split_paths)
    monkeypatch.setattr(census, "FROZEN_SAMPLED_ROOT", sampled_root)
    monkeypatch.setattr(census, "FROZEN_SOURCE_ROOT", source_root)
    monkeypatch.setattr(census, "FROZEN_OUTPUT_ROOT", output_root)

    census._validate_frozen_paths(
        candidate_pool=candidate_pool.resolve(),
        split_receipt=split_receipt.resolve(),
        split_output_paths={name: path.resolve() for name, path in split_paths.items()},
        sampled_root=sampled_root.resolve(),
        source_root=source_root.resolve(),
        output_root=output_root.resolve(),
    )
    with pytest.raises(adapter_module.AssemblyError, match="candidate_pool"):
        census._validate_frozen_paths(
            candidate_pool=alternate_pool.resolve(),
            split_receipt=split_receipt.resolve(),
            split_output_paths={
                name: path.resolve() for name, path in split_paths.items()
            },
            sampled_root=sampled_root.resolve(),
            source_root=source_root.resolve(),
            output_root=output_root.resolve(),
        )

    alternate_receipt = tmp_path / "alternate-split-receipt.json"
    alternate_receipt.write_bytes(split_receipt.read_bytes())
    with pytest.raises(adapter_module.AssemblyError, match="split_receipt"):
        census._validate_frozen_paths(
            candidate_pool=candidate_pool.resolve(),
            split_receipt=alternate_receipt.resolve(),
            split_output_paths={
                name: path.resolve() for name, path in split_paths.items()
            },
            sampled_root=sampled_root.resolve(),
            source_root=source_root.resolve(),
            output_root=output_root.resolve(),
        )

    alternate_train = tmp_path / "alternate-train.jsonl"
    alternate_train.write_bytes(split_paths["train_candidate"].read_bytes())
    alternate_split_paths = {
        **{name: path.resolve() for name, path in split_paths.items()},
        "train_candidate": alternate_train.resolve(),
    }
    with pytest.raises(adapter_module.AssemblyError, match="train_candidate"):
        census._validate_frozen_paths(
            candidate_pool=candidate_pool.resolve(),
            split_receipt=split_receipt.resolve(),
            split_output_paths=alternate_split_paths,
            sampled_root=sampled_root.resolve(),
            source_root=source_root.resolve(),
            output_root=output_root.resolve(),
        )

    with pytest.raises(adapter_module.AssemblyError, match="output_root"):
        census._validate_frozen_paths(
            candidate_pool=candidate_pool.resolve(),
            split_receipt=split_receipt.resolve(),
            split_output_paths={
                name: path.resolve() for name, path in split_paths.items()
            },
            sampled_root=sampled_root.resolve(),
            source_root=source_root.resolve(),
            output_root=(tmp_path / "alternate-output").resolve(),
        )


def test_input_identity_receipt_binds_three_way_inventory_and_identity_objects() -> None:
    canonical_ids = ["1", "2"]
    inventory_hash = census.sha256_json(canonical_ids)
    execution = {"checkpoint": "frozen"}
    tokenizer = {"im_end_token_ids": [151645]}
    adapter = {
        "execution_model_identity": execution,
        "execution_model_identity_sha256": census.sha256_json(execution),
        "tokenizer_identity": tokenizer,
        "tokenizer_identity_sha256": census.sha256_json(tokenizer),
    }
    manifests = {
        "image_inventory_count": 2,
        "ordered_image_inventory_sha256": inventory_hash,
    }

    receipt = census._input_identity_receipt(
        candidate_pool_ids=["2", "1"],
        sampled_manifests=manifests,
        source_manifests=manifests,
        adapter=adapter,
    )

    assert receipt["candidate_pool_inventory"] == {
        "count": 2,
        "ordered_image_ids_sha256": census.sha256_json(["2", "1"]),
        "canonical_image_ids_sha256": inventory_hash,
    }
    assert receipt["three_way_inventory_equality"] == {
        "candidate_pool_equals_sampled_equals_source": True,
        "canonical_image_ids_sha256": inventory_hash,
    }
    assert receipt["shared_execution_model_identity"] == {
        "sha256": census.sha256_json(execution),
        "identity": execution,
    }
    assert receipt["shared_tokenizer_identity"] == {
        "sha256": census.sha256_json(tokenizer),
        "identity": tokenizer,
    }

    mismatched_adapter = dict(adapter)
    mismatched_adapter["tokenizer_identity_sha256"] = "0" * 64
    with pytest.raises(adapter_module.AssemblyError, match="tokenizer identity digest"):
        census._input_identity_receipt(
            candidate_pool_ids=canonical_ids,
            sampled_manifests=manifests,
            source_manifests=manifests,
            adapter=mismatched_adapter,
        )


def test_source_snapshot_is_task_scoped_and_hash_bound(tmp_path: Path) -> None:
    repo = Path(__file__).resolve().parents[2]
    receipt = _materialize_source_snapshot(repo, tmp_path)
    git_identity = census._git_identity(repo, receipt["snapshotted_repo_paths"])
    manifest = json.loads(
        (tmp_path / receipt["manifest_relative_path"]).read_text(encoding="utf-8")
    )
    assert sha256_file(tmp_path / receipt["manifest_relative_path"]) == receipt[
        "manifest_sha256"
    ]
    snapshotted = {item["repo_relative_path"] for item in manifest["entries"]}
    assert set(census.SOURCE_SNAPSHOT_PATHS) <= snapshotted
    assert receipt["loaded_repo_source_paths"] == census._repo_loaded_source_paths(repo)
    assert snapshotted == set(receipt["snapshotted_repo_paths"])
    assert all("checkpoint_reload_probe.py" not in item["repo_relative_path"] for item in manifest["entries"])
    assert git_identity["scope"] == receipt["snapshotted_repo_paths"]
    assert all("checkpoint_reload_probe.py" not in line for line in git_identity["scoped_status_lines"])


def test_source_snapshot_revalidation_rejects_live_or_snapshot_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    output = tmp_path / "output"
    repo.mkdir()
    output.mkdir()
    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    (repo / "contract.md").write_text("frozen\n", encoding="utf-8")
    monkeypatch.setattr(census, "SOURCE_SNAPSHOT_PATHS", ("contract.md", "a.py"))
    monkeypatch.setattr(census, "_repo_loaded_source_paths", lambda _: ["a.py"])
    receipt = _materialize_source_snapshot(repo, output)
    census._verify_source_snapshot(repo, output, receipt)

    (repo / "a.py").write_text("a = 2\n", encoding="utf-8")
    with pytest.raises(adapter_module.AssemblyError, match="source or snapshot drift"):
        census._verify_source_snapshot(repo, output, receipt)

    (repo / "a.py").write_text("a = 1\n", encoding="utf-8")
    snapshot_a = next(
        item for item in receipt["entries"] if item["repo_relative_path"] == "a.py"
    )
    (output / snapshot_a["snapshot_relative_path"]).write_text(
        "a = 3\n", encoding="utf-8"
    )
    with pytest.raises(adapter_module.AssemblyError, match="source or snapshot drift"):
        census._verify_source_snapshot(repo, output, receipt)


def test_atomic_finalization_failure_publishes_only_failed_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staging = tmp_path / ".output.staging-test"
    output = tmp_path / "output"
    staging.mkdir()
    (staging / "image-census.jsonl").write_text("{}\n", encoding="utf-8")
    (staging / "receipt.json").write_text(
        json.dumps({"terminal_status": "completed"}), encoding="utf-8"
    )

    real_chmod = census.os.chmod
    calls = 0

    def fail_chmod_once(path: Path, mode: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected chmod failure")
        real_chmod(path, mode)

    monkeypatch.setattr(census.os, "chmod", fail_chmod_once)
    with pytest.raises(OSError, match="injected chmod failure"):
        _finalize_staging(staging, output, analyzer=Path(census.__file__).resolve())

    assert not staging.exists()
    assert sorted(path.name for path in output.iterdir()) == ["receipt.json"]
    failed = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert failed["terminal_status"] == "failed"
    assert failed["failed_validation_stage"] == "finalize_staging"
    assert failed["failure_class"] == "OSError"
    assert failed["failure_message"] == "injected chmod failure"
    assert failed["failure_output_immutable"] is True
    assert failed["immutability_error"] is None
    assert failed["failures"] == ["OSError: injected chmod failure"]
    assert "completed" not in (output / "receipt.json").read_text(encoding="utf-8")
    assert (output.stat().st_mode & 0o777) == 0o555
    assert ((output / "receipt.json").stat().st_mode & 0o777) == 0o444


def test_failure_publication_reports_immutability_chmod_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed-output"

    def fail_chmod(path: Path, mode: int) -> None:
        raise OSError("failure tree chmod denied")

    monkeypatch.setattr(census.os, "chmod", fail_chmod)
    failure = RuntimeError("validation stopped")
    census._publish_failed_output(
        output,
        analyzer=Path(census.__file__).resolve(),
        failed_validation_stage="readback_outputs",
        failure=failure,
        failures=["RuntimeError: validation stopped"],
    )

    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == "failed"
    assert receipt["failure_output_immutable"] is False
    assert receipt["immutability_error"] == "OSError: failure tree chmod denied"


def _assert_failure_only(output: Path, *, failure_class: str, message: str) -> None:
    assert sorted(path.name for path in output.iterdir()) == ["receipt.json"]
    receipt = json.loads((output / "receipt.json").read_text(encoding="utf-8"))
    assert receipt["terminal_status"] == "failed"
    assert receipt["failed_validation_stage"] == "resolve_and_validate_inputs"
    assert receipt["failure_class"] == failure_class
    assert message in receipt["failure_message"]
    assert receipt["failure_output_immutable"] is True


def test_materialize_publishes_failure_only_for_missing_frozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "production-v1"
    monkeypatch.setattr(census, "FROZEN_OUTPUT_ROOT", output)

    with pytest.raises(FileNotFoundError):
        census._materialize(
            candidate_pool=tmp_path / "missing-candidate.jsonl",
            split_receipt=tmp_path / "missing-split.json",
            sampled_root=tmp_path / "missing-sampled",
            source_root=tmp_path / "missing-source",
            output_dir=output,
        )

    _assert_failure_only(output, failure_class="FileNotFoundError", message="missing-candidate")
    assert not list(tmp_path.glob(".production-v1.staging-*"))


def test_materialize_publishes_failure_only_for_malformed_split_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "production-v1"
    candidate = tmp_path / "candidate.jsonl"
    split_receipt = tmp_path / "split-receipt.json"
    sampled_root = tmp_path / "sampled"
    source_root = tmp_path / "source"
    candidate.write_text("{}\n", encoding="utf-8")
    split_receipt.write_text("{malformed", encoding="utf-8")
    sampled_root.mkdir()
    source_root.mkdir()
    monkeypatch.setattr(census, "FROZEN_OUTPUT_ROOT", output)

    with pytest.raises(json.JSONDecodeError):
        census._materialize(
            candidate_pool=candidate,
            split_receipt=split_receipt,
            sampled_root=sampled_root,
            source_root=source_root,
            output_dir=output,
        )

    _assert_failure_only(output, failure_class="JSONDecodeError", message="Expecting")


def test_materialize_publishes_failure_only_for_corrupt_or_alternate_frozen_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corrupt_output = tmp_path / "production-corrupt"
    path_output = tmp_path / "production-alternate-path"
    candidate = tmp_path / "candidate.jsonl"
    alternate_candidate = tmp_path / "alternate-candidate.jsonl"
    split_receipt = tmp_path / "split-receipt.json"
    sampled_root = tmp_path / "sampled"
    source_root = tmp_path / "source"
    split_paths = {
        name: tmp_path / f"{name}.jsonl"
        for name in ("train_candidate", "development", "heldout")
    }
    candidate.write_text("same bytes\n", encoding="utf-8")
    alternate_candidate.write_bytes(candidate.read_bytes())
    for path in split_paths.values():
        path.write_text("{}\n", encoding="utf-8")
    split_receipt.write_text(
        json.dumps(
            {
                "outputs": {
                    name: {"path": str(path)} for name, path in split_paths.items()
                }
            }
        ),
        encoding="utf-8",
    )
    sampled_root.mkdir()
    source_root.mkdir()
    monkeypatch.setattr(census, "FROZEN_CANDIDATE_POOL_PATH", candidate)
    monkeypatch.setattr(census, "FROZEN_SPLIT_RECEIPT_PATH", split_receipt)
    monkeypatch.setattr(census, "FROZEN_SPLIT_OUTPUT_PATHS", split_paths)
    monkeypatch.setattr(census, "FROZEN_SAMPLED_ROOT", sampled_root)
    monkeypatch.setattr(census, "FROZEN_SOURCE_ROOT", source_root)
    monkeypatch.setattr(census, "FROZEN_OUTPUT_ROOT", corrupt_output)

    with pytest.raises(adapter_module.AssemblyError, match="frozen SHA-256"):
        census._materialize(
            candidate_pool=candidate,
            split_receipt=split_receipt,
            sampled_root=sampled_root,
            source_root=source_root,
            output_dir=corrupt_output,
        )

    _assert_failure_only(
        corrupt_output, failure_class="AssemblyError", message="frozen SHA-256"
    )

    monkeypatch.setattr(census, "FROZEN_OUTPUT_ROOT", path_output)

    with pytest.raises(adapter_module.AssemblyError, match="candidate_pool"):
        census._materialize(
            candidate_pool=alternate_candidate,
            split_receipt=split_receipt,
            sampled_root=sampled_root,
            source_root=source_root,
            output_dir=path_output,
        )

    _assert_failure_only(
        path_output, failure_class="AssemblyError", message="candidate_pool"
    )


def test_success_receipt_readback_rejects_receipt_and_artifact_tampering(
    tmp_path: Path,
) -> None:
    record = _analyze_image_candidates(
        "1", [_candidate("sample-00", "a", ["A"], ["A"])]
    )
    records = [record]
    summary = _summarize_records(records)
    census_path = tmp_path / "image-census.jsonl"
    summary_path = tmp_path / "summary.json"
    manifest_path = tmp_path / "source-snapshot" / "manifest.json"
    manifest_path.parent.mkdir()
    census_path.write_text(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    census._write_json(summary_path, summary)
    census._write_json(manifest_path, {"entries": []})
    records_hash = census.sha256_json(records)
    summary_hash = census.sha256_json(summary)
    manifest_hash = sha256_file(manifest_path)
    receipt: dict[str, Any] = {
        "schema_version": census.SCHEMA_VERSION,
        "terminal_status": "completed",
        "immutable_output_directory": str(tmp_path / "production-v1"),
        "source_snapshot": {"manifest_sha256": manifest_hash},
        "row_counts": {
            "image_census": 1,
            "candidate": summary["candidate_count"],
            "eligible_candidate": summary["eligible_candidate_count"],
            "excluded_candidate": summary["excluded_candidate_count"],
        },
        "artifact_integrity_checks": {
            "serialized_success_receipt_readback": True
        },
        "determinism": {
            "forward_records_sha256": records_hash,
            "reversed_records_sha256": records_hash,
            "forward_summary_sha256": summary_hash,
            "reversed_summary_sha256": summary_hash,
        },
        "output_hashes": {
            "image-census.jsonl": sha256_file(census_path),
            "summary.json": sha256_file(summary_path),
            "source-snapshot/manifest.json": manifest_hash,
        },
    }
    receipt_path = tmp_path / "receipt.json"
    census._write_json(receipt_path, receipt)

    receipt_hash = census._validate_success_receipt_readback(
        receipt_path=receipt_path,
        expected_receipt=receipt,
        output_root=tmp_path,
    )
    assert receipt_hash == sha256_file(receipt_path)

    tampered_receipt = {**receipt, "terminal_status": "failed"}
    census._write_json(receipt_path, tampered_receipt)
    with pytest.raises(adapter_module.AssemblyError, match="differs from memory"):
        census._validate_success_receipt_readback(
            receipt_path=receipt_path,
            expected_receipt=receipt,
            output_root=tmp_path,
        )

    census._write_json(receipt_path, receipt)
    summary_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(adapter_module.AssemblyError, match="hash does not reproduce"):
        census._validate_success_receipt_readback(
            receipt_path=receipt_path,
            expected_receipt=receipt,
            output_root=tmp_path,
        )


def test_summary_reconciliation_and_serialized_readback_reject_tampering(
    tmp_path: Path,
) -> None:
    record = _analyze_image_candidates(
        "1", [_candidate("sample-00", "a", ["A"], ["A"])]
    )
    records = [record]
    summary = _summarize_records(records)
    census._validate_summary_reconciliation(records, summary)

    census_path = tmp_path / "image-census.jsonl"
    summary_path = tmp_path / "summary.json"
    census_path.write_text(
        json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    census._validate_serialized_output_readback(
        census_path=census_path,
        summary_path=summary_path,
        records=records,
        summary=summary,
    )

    mutated_summary = dict(summary)
    mutated_summary["fully_adjudicable_image_count"] = 999
    summary_path.write_text(
        json.dumps(mutated_summary, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(adapter_module.AssemblyError, match="readback differs"):
        census._validate_serialized_output_readback(
            census_path=census_path,
            summary_path=summary_path,
            records=records,
            summary=summary,
        )

    with pytest.raises(adapter_module.AssemblyError, match="exactly reconcile"):
        census._validate_summary_reconciliation(records, mutated_summary)
