from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripts.research import materialize_transition_phase0_fixed_prefix_panel as panel
from scripts.research import run_transition_phase0_fixed_prefix_release as release


def _canonical_row(description: list[int]) -> list[int]:
    return [
        panel.OBJECT_REF_START,
        *description,
        151647,
        151648,
        151670,
        151671,
        151672,
        151673,
        panel.BOX_END,
    ]


def _token_reference(token_ids: list[int]) -> dict[str, Any]:
    return {
        "token_ids": token_ids,
        "token_ids_sha256": panel.token_ids_sha256(token_ids),
    }


def _release_manifest(*, candidate_row: list[int], owner: str | None = "owner-1") -> dict[str, Any]:
    prefix = _canonical_row([601])
    return {
        "images": [
            {
                "image_id": "3442",
                "row_id": "row-3442",
                "case_id": "case-3442",
                "case_role": "test case",
                "boundaries": [
                    {
                        "boundary_id": "boundary-3442",
                        "prefix_mode": "base_prompt_plus_generated",
                        "prefix": _token_reference(prefix),
                        "source_completed_row_count": 1,
                        "covered_owner_ids": ["covered-2", "covered-1"],
                        "candidates": [
                            {
                                "candidate_id": "candidate-1",
                                "row": _token_reference(candidate_row),
                                "owner": owner,
                                "category": "car",
                                "role": "gained_uncovered_owner",
                            }
                        ],
                    }
                ],
            }
        ]
    }


def test_frozen_case_contract_has_exact_seven_cases_and_required_roles() -> None:
    assert [case["image_id"] for case in panel.FROZEN_CASE_SPECS] == [
        "3442",
        "4129",
        "15379",
        "28058",
        "65891",
        "70558",
        "355385",
    ]
    candidates = [
        candidate
        for case in panel.FROZEN_CASE_SPECS
        for candidate in case["candidates"]
    ]
    assert len(candidates) == 22
    assert {candidate["role"] for candidate in candidates} == {
        "covered_owner_control",
        "future_retained_uncovered_owner",
        "gained_uncovered_owner",
        "lost_uncovered_owner",
        "unmatched_neutral_candidate",
    }


def test_trace_rows_and_literal_slice_hash_are_fail_closed() -> None:
    row_a = _canonical_row([701])
    row_b = _canonical_row([702])
    records = [
        {"generated_step_index": index, "token_id": token_id}
        for index, token_id in enumerate([*row_a, *row_b])
    ]
    rows = panel.split_complete_trace_rows(records, row_id="row-a")
    assert [[record["token_id"] for record in row] for row in rows] == [row_a, row_b]
    assert panel.validate_token_slice(
        rows[1],
        expected_step_span=[len(row_a), len(row_a) + len(row_b) - 1],
        expected_token_count=len(row_b),
        expected_sha256=panel.token_ids_sha256(row_b),
        label="row-b",
    ) == row_b
    with pytest.raises(ValueError, match="token hash mismatch"):
        panel.validate_token_slice(
            rows[1],
            expected_step_span=[len(row_a), len(row_a) + len(row_b) - 1],
            expected_token_count=len(row_b),
            expected_sha256="0" * 64,
            label="row-b",
        )


def test_release_plan_forces_opener_or_one_token_complete_description() -> None:
    manifest = _release_manifest(candidate_row=_canonical_row([703]))
    opener = release.build_release_plan(
        manifest,
        image_id="3442",
        force_mode="opener",
    )
    assert opener["forced_row_prefix_token_ids"] == [panel.OBJECT_REF_START]
    assert opener["covered_owner_ids"] == ["covered-1", "covered-2"]

    description = release.build_release_plan(
        manifest,
        image_id="3442",
        force_mode="complete-description",
        candidate_id="candidate-1",
    )
    assert description["forced_row_prefix_token_ids"] == [
        panel.OBJECT_REF_START,
        703,
        151647,
    ]
    assert description["expected_owner_id"] == "owner-1"


def test_complete_description_rejects_neutral_or_multi_token_candidate() -> None:
    neutral = _release_manifest(candidate_row=_canonical_row([703]), owner=None)
    with pytest.raises(ValueError, match="owner-bearing candidate"):
        release.build_release_plan(
            neutral,
            image_id="3442",
            force_mode="complete-description",
            candidate_id="candidate-1",
        )

    multi_token = _release_manifest(candidate_row=_canonical_row([703, 704]))
    with pytest.raises(ValueError, match="exactly one description token"):
        release.build_release_plan(
            multi_token,
            image_id="3442",
            force_mode="complete-description",
            candidate_id="candidate-1",
        )


def test_declared_runtime_requires_exact_paths_and_hashes(tmp_path: Path) -> None:
    config = tmp_path / "infer.yaml"
    source = tmp_path / "heldout.jsonl"
    config.write_text("backend: hf\n", encoding="utf-8")
    source.write_text("{}\n", encoding="utf-8")
    manifest = {
        "panel": {
            "schema_version": panel.PANEL_SCHEMA_VERSION,
            "checkpoint_configs": {
                "source": {
                    "path": str(config),
                    "sha256": panel.sha256_file(config),
                }
            },
            "inputs": {
                "heldout_source_jsonl": {
                    "path": str(source),
                    "sha256": panel.sha256_file(source),
                }
            },
        }
    }
    runtime = release.validate_declared_runtime(
        manifest,
        checkpoint_role="source",
        infer_config=config,
    )
    assert runtime["infer_config"]["path"] == str(config)
    assert runtime["source_jsonl"]["path"] == str(source)

    other = tmp_path / "other.yaml"
    other.write_text("backend: hf\n", encoding="utf-8")
    with pytest.raises(ValueError, match="path differs"):
        release.validate_declared_runtime(
            manifest,
            checkpoint_role="source",
            infer_config=other,
        )

    config.write_text("backend: vllm\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        release.validate_declared_runtime(
            manifest,
            checkpoint_role="source",
            infer_config=config,
        )
