from __future__ import annotations

import json
from pathlib import Path

from scripts.research.build_matched_random_sorted_candidate_score_manifest import (
    SCHEMA_VERSION,
    build_manifest,
    sha256_file,
    sha256_json,
)
from scripts.research.run_complete_candidate_row_scoring import validate_manifest


def _row() -> list[int]:
    return [151646, 8987, 151647, 151648, 151670, 151671, 151672, 151673, 151649]


def _artifact(path: Path, *, image_id: str = "123") -> None:
    row = _row()
    rows = []
    for entity_id, description in (("gt_1", "person"), ("gt_2", "person"), ("gt_3", "person"), ("gt_cov", "chair")):
        rows.append({
            "entity_id": entity_id,
            "description": description,
            "row_token_ids": row,
            "row_token_ids_sha256": sha256_json(row),
        })
    arms = {
        "current_sorted_rollout_order": {
            "covered_entity_ids": ["gt_cov"],
            "runs": [],
        }
    }
    for index in range(6):
        alt = f"alt_{index}"
        comparison = f"current_sorted_rollout_order_vs_{alt}"
        prefix = [100 + index]
        arms["current_sorted_rollout_order"]["runs"].append({
            "comparison_id": comparison,
            "mode": "greedy",
            "seed": None,
            "initial_prefix_token_ids": prefix,
            "initial_prefix_token_ids_sha256": sha256_json(prefix),
            "rows": [{"prefix_token_ids": prefix}],
        })
        arms[alt] = {
            "covered_entity_ids": ["gt_cov"],
            "runs": [{
                "comparison_id": comparison,
                "mode": "greedy",
                "seed": None,
                "initial_prefix_token_ids": [200 + index],
                "initial_prefix_token_ids_sha256": sha256_json([200 + index]),
                "rows": [{"prefix_token_ids": [200 + index]}],
            }],
        }
    document = {
        "schema_version": "same_covered_set_prefix_order.v2",
        "image": {"image_id": image_id},
        "base_prompt": {
            "prompt_token_ids": [1, 2],
            "prompt_token_ids_sha256": sha256_json([1, 2]),
        },
        "cases": [{
            "case_id": "case-a",
            "arms": arms,
            "entity_ledger": rows,
        }],
    }
    path.write_text(json.dumps(document), encoding="utf-8")


def _selection(path: Path) -> dict:
    pairs = []
    for index in range(6):
        alt = f"alt_{index}"
        sorted_activated = index == 0
        random_activated = not sorted_activated
        def result(activated: bool, missing_right: bool = False) -> dict:
            return {
                "activated": activated,
                "arms": ["current_sorted_rollout_order", alt],
                "greedy": {
                    "left": {
                        "strict_owner": "gt_3" if missing_right else ("gt_1" if index else "gt_1"),
                        "covered_recurrence": missing_right,
                    },
                    "right": {"strict_owner": None if missing_right else "gt_2"},
                },
                "paired_runs": ([
                    {"mode": "sample", "left": {"strict_owner": "gt_1"}, "right": {"strict_owner": "gt_2"}},
                ] if missing_right else []),
            }
        pairs.append({
            "artifact_file": "case-a.json",
            "case_id": "case-a",
            "checkpoint_results": {
                "sorted": result(sorted_activated, missing_right=False),
                "random": result(random_activated, missing_right=index == 2),
            },
        })
    path.write_text(json.dumps({"pairs": pairs}), encoding="utf-8")


def test_builder_emits_two_prefix_boundaries_and_validates(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    (artifact_root / "sorted").mkdir(parents=True)
    (artifact_root / "random").mkdir(parents=True)
    _artifact(artifact_root / "sorted" / "case-a.json")
    _artifact(artifact_root / "random" / "case-a.json")
    selection_path = tmp_path / "selection.json"
    _selection(selection_path)

    manifest = build_manifest(
        selection=json.loads(selection_path.read_text()),
        selection_path=selection_path,
        artifact_root=artifact_root,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    normalized = validate_manifest(manifest, manifest_path=manifest_path)

    assert normalized["schema_version"] == SCHEMA_VERSION
    assert len(normalized["images"]) == 1
    boundaries = normalized["images"][0]["boundaries"]
    assert len(boundaries) == 12
    assert {boundary["prefix_arm"] for boundary in boundaries} == {
        "current_sorted_rollout_order",
        "alt_0",
        "alt_1",
        "alt_2",
        "alt_3",
        "alt_4",
        "alt_5",
    }
    assert all("alternate_prefix" not in boundary for boundary in boundaries)
    assert all(boundary["prefix_mode"] == "base_prompt_plus_generated" for boundary in boundaries)
    assert all(len(boundary["candidates"]) >= 3 for boundary in boundaries)
    assert any(
        candidate["owner"] == "gt_3" and candidate["covered"]
        for candidate in boundaries[4]["candidates"]
    )
    assert len(manifest["builder"]["ambiguities"]) == 1
    assert manifest["builder"]["ambiguities"][0]["fallback_owners"] == ["gt_1", "gt_2"]


def test_selection_receipt_hash_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "selection.json"
    path.write_text('{"pairs": []}\n', encoding="utf-8")
    assert sha256_file(path) == sha256_file(path)
