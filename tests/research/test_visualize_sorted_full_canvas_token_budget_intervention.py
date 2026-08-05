"""Focused contract tests for the paired full-canvas intervention atlas."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import (  # noqa: E402
    visualize_sorted_full_canvas_token_budget_intervention as visual,
)


def _owner(owner_id: str, image_id: str, box: list[int], *, description: str = "person"):
    return {
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "split": "discovery",
        "normalized_description": description,
        "bbox_pixel_xyxy": box,
        "owner_sort_key": [box[1], box[0]],
    }


def _report() -> dict:
    primary = [
        {
            "gt_owner_id": "gt:1:0",
            "treatment_support_u": True,
            "treatment_support_context_ids_u": ["1:boundary-002", "1:boundary-001"],
        },
        {
            "gt_owner_id": "gt:1:1",
            "treatment_support_u": False,
            "treatment_support_context_ids_u": [],
        },
        {
            "gt_owner_id": "gt:2:0",
            "treatment_support_u": False,
            "treatment_support_context_ids_u": [],
        },
    ]
    admissions = {
        owner_id: {
            "allowed_context_ids": [f"{owner_id.split(':')[1]}:boundary-000"],
        }
        for owner_id in ("gt:1:0", "gt:1:1", "gt:2:0", "gt:3:0", "gt:4:0")
    }
    admissions["gt:1:0"]["allowed_context_ids"] = [
        "1:boundary-000",
        "1:boundary-001",
        "1:boundary-002",
    ]
    return {
        "primary": {"owners": primary},
        "retention": {
            "confirmation_true_positive": {"lost": ["gt:3:0"]},
            "resolved_support": {"lost": ["gt:4:0"]},
        },
        "owner_role_context_admission": {"owners": admissions},
    }


def test_report_provenance_fails_closed_on_overlay_mismatch():
    overlay = {
        "arm_id": "treatment",
        "baseline_arm_id": "baseline",
        "overlay_content_sha256": "sealed-overlay",
        "base": {"predecessor_run_root": "/pred"},
        "images": {"1": {}},
    }
    plan = SimpleNamespace(receipt={"receipt_content_sha256": "plan"})
    seals = {
        "schema_version": visual.analyzer.CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION,
        "role": "exact_receipt_and_score_bytes_consumed_by_this_analyzer",
        "image_count": 1,
        "images": {"1": {}},
    }
    seals["capture_artifact_seals_sha256"] = visual.sha256_json(seals)
    report = {
        "schema_version": visual.REPORT_SCHEMA_VERSION,
        "status": "complete_uniform_capture_identity",
        "intervention_unit_id": visual.UNIT_ID,
        "arm_id": "treatment",
        "baseline_arm_id": "baseline",
        "overlay_content_sha256": "wrong-overlay",
        "base_plan_receipt_content_sha256": "plan",
        "predecessor_run_root": "/pred",
        "captured_images": ["1"],
        "capture_identity": {
            "status": "complete_uniform_capture_identity",
            "image_count": 1,
        },
        "capture_artifact_seals": seals,
        "comparison_semantics": {
            "raw_logprob_compared_across_arms": False,
            "compared_quantity": "support_disposition_under_arm_local_calibration",
        },
        "owner_role_context_admission": {
            "selection_basis": "pre_treatment_overlay_and_predecessor_owner_summaries_only",
            "treatment_scores_used_for_context_selection": False,
            "owners": {},
        },
    }
    with pytest.raises(visual.VisualContractError, match="overlay_content_sha256"):
        visual.validate_report_provenance(report, overlay, plan)


def test_deterministic_report_selection_matches_area_and_preserves_owner_ids():
    owners = {
        "gt:1:0": _owner("gt:1:0", "1", [10, 10, 30, 30]),
        "gt:1:1": _owner("gt:1:1", "1", [40, 10, 62, 32]),
        "gt:2:0": _owner("gt:2:0", "2", [10, 10, 90, 90]),
        "gt:3:0": _owner("gt:3:0", "3", [10, 10, 20, 20]),
        "gt:4:0": _owner("gt:4:0", "4", [10, 10, 20, 20]),
    }
    first = visual.select_cases(_report(), owners, max_per_role=4)
    second = visual.select_cases(_report(), dict(reversed(list(owners.items()))), max_per_role=4)
    assert first == second
    assert first == [
        visual.Case("gt:1:0", "recovered_persistent", "1:boundary-001"),
        visual.Case(
            "gt:1:1",
            "matched_nonrecovered_control",
            "1:boundary-000",
            matched_recovered_owner_id="gt:1:0",
        ),
        visual.Case("gt:3:0", "confirmation_retention_loss", "3:boundary-000"),
        visual.Case("gt:4:0", "resolved_retention_loss", "4:boundary-000"),
    ]


def test_recovered_context_must_be_frozen_allowed():
    report = _report()
    report["primary"]["owners"][0]["treatment_support_context_ids_u"] = [
        "1:boundary-999"
    ]
    owners = {
        "gt:1:0": _owner("gt:1:0", "1", [10, 10, 30, 30]),
        "gt:1:1": _owner("gt:1:1", "1", [40, 10, 62, 32]),
        "gt:2:0": _owner("gt:2:0", "2", [10, 10, 90, 90]),
        "gt:3:0": _owner("gt:3:0", "3", [10, 10, 20, 20]),
        "gt:4:0": _owner("gt:4:0", "4", [10, 10, 20, 20]),
    }
    with pytest.raises(visual.VisualContractError, match="not frozen-allowed"):
        visual.select_cases(report, owners)


@pytest.mark.parametrize("mutation", ["duplicate", "non_boolean"])
def test_primary_owner_rows_require_unique_ids_and_boolean_support(mutation: str):
    report = _report()
    if mutation == "duplicate":
        report["primary"]["owners"].append(dict(report["primary"]["owners"][0]))
        message = "duplicate"
    else:
        report["primary"]["owners"][0]["treatment_support_u"] = 1
        message = "non-boolean"
    owners = {
        "gt:1:0": _owner("gt:1:0", "1", [10, 10, 30, 30]),
        "gt:1:1": _owner("gt:1:1", "1", [40, 10, 62, 32]),
        "gt:2:0": _owner("gt:2:0", "2", [10, 10, 90, 90]),
        "gt:3:0": _owner("gt:3:0", "3", [10, 10, 20, 20]),
        "gt:4:0": _owner("gt:4:0", "4", [10, 10, 20, 20]),
    }
    with pytest.raises(visual.VisualContractError, match=message):
        visual.select_cases(report, owners)


def test_within_arm_normalization_is_independent_and_not_joint():
    baseline = visual.within_arm_confidence({"a": -1.0, "b": -2.0})
    treatment = visual.within_arm_confidence({"a": -101.0, "b": -102.0})
    assert baseline == pytest.approx(treatment)
    assert math.fsum(baseline.values()) == pytest.approx(1.0)
    assert math.fsum(treatment.values()) == pytest.approx(1.0)


def test_shared_crop_is_identical_for_both_arms_and_score_independent():
    crop = visual.shared_local_crop(
        [[100, 100, 106, 108], [95, 98, 110, 112]], width=400, height=300
    )
    assert crop["shared_across_arms"] is True
    assert crop["derived_from"] == "gt_and_frozen_owner_local_candidate_geometry_never_scores"
    x1, y1, x2, y2 = crop["window_pixel_xyxy"]
    assert x2 - x1 >= visual.MIN_CROP_EXTENT_PIXELS
    assert y2 - y1 >= visual.MIN_CROP_EXTENT_PIXELS


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_tampered_score_bytes_are_rejected_before_row_use(tmp_path: Path):
    root = tmp_path / "treatment"
    shard = root / "1"
    shard.mkdir(parents=True)
    receipt_path = shard / visual.RECEIPT_NAME
    score_path = shard / visual.SCORES_NAME
    _write_json(receipt_path, {"status": "captured"})
    _write_jsonl(score_path, [{"score": 1}])
    seals = {
        "schema_version": visual.analyzer.CAPTURE_ARTIFACT_SEALS_SCHEMA_VERSION,
        "role": "exact_receipt_and_score_bytes_consumed_by_this_analyzer",
        "image_count": 1,
        "images": {
            "1": {
                "image_id": "1",
                "receipt_path": str(receipt_path.resolve()),
                "receipt_sha256": visual.planner.sha256_file(receipt_path),
                "census_scores_path": str(score_path.resolve()),
                "census_scores_sha256": visual.planner.sha256_file(score_path),
            }
        },
    }
    seals["capture_artifact_seals_sha256"] = visual.sha256_json(seals)
    report = {"capture_artifact_seals": seals}
    score_path.write_text('{"score":2}\n', encoding="utf-8")
    with pytest.raises(visual.VisualContractError, match="digest changed"):
        visual.verify_capture_artifact_seals(report, root)


def test_exact_score_join_rejects_coordinate_token_drift(tmp_path: Path):
    image_id = "1"
    group_id = "1:boundary-000|person"
    candidate_id = "cand:1"
    root = tmp_path / "baseline"
    shard = root / image_id
    shard.mkdir(parents=True)
    _write_json(
        shard / visual.RECEIPT_NAME,
                {
            "status": "captured",
            "image_id": image_id,
            "capture_completeness": "complete_shard",
            "plan": {"receipt_content_sha256": "plan"},
        },
    )
    _write_jsonl(
        shard / visual.SCORES_NAME,
        [
            {
                "request_id": f"{group_id}|{candidate_id}",
                "query_group_id": group_id,
                "image_id": image_id,
                "context_id": "1:boundary-000",
                    "candidate_id": candidate_id,
                    "normalized_description": "person",
                    "rank_key": {
                        "image_id": image_id,
                        "context_id": "1:boundary-000",
                        "normalized_description": "person",
                    },
                "plan_receipt_content_sha256": "plan",
                "coord_token_ids": [9, 2, 3, 4],
                "coord_token_ids_sha256": "coords",
                "complete_box_logprob_sum": -1.0,
            }
        ],
    )
    plan = SimpleNamespace(
        receipt={"receipt_content_sha256": "plan"},
        candidates={
            candidate_id: {
                "coord_token_ids": [1, 2, 3, 4],
                "coord_token_ids_sha256": "coords",
            }
        },
        query_groups={
            group_id: {
                "context_id": "1:boundary-000",
                "normalized_description": "person",
            }
        },
    )
    with pytest.raises(visual.VisualContractError, match="coordinate token IDs drifted"):
        visual._score_index_for_group(  # noqa: SLF001 - exact join is the contract under test
            shard_root=root,
            image_id=image_id,
            query_group_id=group_id,
            candidate_ids=[candidate_id],
            plan=plan,
            overlay={"arm_id": "t", "baseline_arm_id": "b", "overlay_content_sha256": "o"},
            treatment=False,
        )


def _treatment_join_fixture(tmp_path: Path):
    image_id = "1"
    group_id = "1:boundary-000|person"
    candidate_id = "cand:1"
    directory = tmp_path / "treatment" / image_id
    directory.mkdir(parents=True)
    stamp = {
        "intervention_unit_id": visual.UNIT_ID,
        "arm_id": "treatment",
        "baseline_arm_id": "baseline",
        "overlay_content_sha256": "overlay",
        "base_plan_receipt_content_sha256": "plan",
    }
    receipt = {
        "status": "captured",
        "image_id": image_id,
        "plan": {"receipt_content_sha256": "plan"},
        "intervention": stamp,
        "intervention_completeness": {
            "status": "complete_frozen_overlay_selection",
            "is_complete_frozen_overlay_selection": True,
            "frozen_expected_query_group_count": 1,
            "executed_query_group_count": 1,
            "missing_query_group_ids": [],
            "extra_query_group_ids": [],
        },
        "intervention_capture_mode": {
            "score_only": True,
            "behavior_sidecars_captured": False,
            "behavior_sidecars_intentionally_disabled": True,
        },
    }
    row = {
        "request_id": f"{group_id}|{candidate_id}",
        "query_group_id": group_id,
        "image_id": image_id,
        "context_id": "1:boundary-000",
        "normalized_description": "person",
        "rank_key": {
            "image_id": image_id,
            "context_id": "1:boundary-000",
            "normalized_description": "person",
        },
        "candidate_id": candidate_id,
        "plan_receipt_content_sha256": "plan",
        "coord_token_ids": [1, 2, 3, 4],
        "coord_token_ids_sha256": "coords",
        "complete_box_logprob_sum": -1.0,
        "intervention": stamp,
    }
    _write_json(directory / visual.RECEIPT_NAME, receipt)
    _write_jsonl(directory / visual.SCORES_NAME, [row])
    plan = SimpleNamespace(
        receipt={"receipt_content_sha256": "plan"},
        candidates={
            candidate_id: {
                "coord_token_ids": [1, 2, 3, 4],
                "coord_token_ids_sha256": "coords",
            }
        },
        query_groups={
            group_id: {
                "context_id": "1:boundary-000",
                "normalized_description": "person",
            }
        },
    )
    overlay = {
        "arm_id": "treatment",
        "baseline_arm_id": "baseline",
        "overlay_content_sha256": "overlay",
        "query_group_selection": {"query_group_ids_by_image": {image_id: [group_id]}},
    }
    shard = visual.analyzer.TreatmentShard(
        image_id=image_id,
        directory=directory,
        receipt=receipt,
        scores=[row],
        proposals=[],
        x1=[],
    )
    return plan, overlay, shard, group_id, candidate_id


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("request", "noncanonical request_id"),
        ("description", "foreign description"),
        ("rank", "foreign rank_key"),
        ("stamp", "stamp differs from its receipt"),
    ],
)
def test_treatment_exact_join_mutations_fail_closed(
    tmp_path: Path, mutation: str, message: str
):
    plan, overlay, shard, group_id, candidate_id = _treatment_join_fixture(tmp_path)
    row = shard.scores[0]
    if mutation == "request":
        row["request_id"] = "wrong"
    elif mutation == "description":
        row["normalized_description"] = "car"
    elif mutation == "rank":
        row["rank_key"] = {**row["rank_key"], "normalized_description": "car"}
    else:
        row["intervention"] = {"arm_id": "treatment"}
    with pytest.raises(visual.VisualContractError, match=message):
        visual._score_index_for_group(  # noqa: SLF001 - exact row identity under test
            shard_root=shard.directory.parent,
            image_id="1",
            query_group_id=group_id,
            candidate_ids=[candidate_id],
            plan=plan,
            overlay=overlay,
            treatment=True,
            validated_treatment_shard=shard,
        )


def test_inconsistent_treatment_completeness_fails_closed(tmp_path: Path):
    plan, overlay, shard, group_id, candidate_id = _treatment_join_fixture(tmp_path)
    shard.receipt["intervention_completeness"][
        "is_complete_frozen_overlay_selection"
    ] = False
    with pytest.raises(visual.VisualContractError, match="completeness boolean"):
        visual._score_index_for_group(  # noqa: SLF001 - completeness under test
            shard_root=shard.directory.parent,
            image_id="1",
            query_group_id=group_id,
            candidate_ids=[candidate_id],
            plan=plan,
            overlay=overlay,
            treatment=True,
            validated_treatment_shard=shard,
        )


def _specs_only_inputs(tmp_path: Path) -> visual.Inputs:
    plan, overlay, treatment, group_id, candidate_id = _treatment_join_fixture(tmp_path)
    baseline_dir = tmp_path / "baseline" / "1"
    baseline_dir.mkdir(parents=True)
    observed_prefix_sha256 = "observed-prefix"
    query_prefix_sha256 = "query-prefix"
    query_suffix_token_ids_sha256 = "query-suffix"
    admission_receipt_id = visual.planner.admission_receipt_id(
        context_id="1:boundary-000",
        channel=visual.planner.CHANNEL_QUERY_SUFFIX,
        prefix_sha256=query_prefix_sha256,
    )
    baseline_receipt = {
        "status": "captured",
        "image_id": "1",
        "capture_completeness": "complete_shard",
        "plan": {"receipt_content_sha256": "plan"},
        "admission": {
            "receipts": [
                {
                    "schema_version": visual.merge.EXPECTED_ADMISSION_SCHEMA,
                    "context_id": "1:boundary-000",
                    "channel": visual.planner.CHANNEL_QUERY_SUFFIX,
                    "query_prefix_sha256": query_prefix_sha256,
                    "admission_receipt_id": admission_receipt_id,
                    "admitted": True,
                    "inherited_from_another_prefix": False,
                    "inherited_from_another_category": False,
                }
            ]
        },
    }
    treatment.scores[0].update(
        {
            "schema_version": visual.merge.EXPECTED_SCORE_SCHEMA,
            "row_kind": "census_localization_score",
            "is_sidecar": False,
            "observed_prefix_sha256": observed_prefix_sha256,
            "query_prefix_sha256": query_prefix_sha256,
            "query_suffix_token_ids_sha256": query_suffix_token_ids_sha256,
            "admission_receipt_id": admission_receipt_id,
        }
    )
    baseline_row = {
        key: value
        for key, value in treatment.scores[0].items()
        if key != "intervention"
    }
    owner_id = "gt:1:0"
    plan.owners = {
        owner_id: {
            **_owner(owner_id, "1", [10, 10, 20, 20]),
            "candidate_bank": {
                "generator_local_landscape_candidate_ids": [candidate_id]
            },
        }
    }
    plan.contexts = {
        "1:boundary-000": {"image_id": "1", "context_id": "1:boundary-000"}
    }
    plan.query_groups[group_id].update(
        {
            "status": "admitted",
            "image_id": "1",
            "candidate_ids": [candidate_id],
            "observed_prefix_sha256": observed_prefix_sha256,
            "query_prefix_sha256": query_prefix_sha256,
            "query_suffix_token_ids_sha256": query_suffix_token_ids_sha256,
        }
    )
    plan.candidates[candidate_id].update(
        {
            "image_id": "1",
            "normalized_description": "person",
            "decoded_bbox_pixel_xyxy": [9, 9, 21, 21],
            "representative_role": "exact_gt_anchor",
            "generator_gt_owner_ids": [owner_id],
        }
    )
    plan.images = {
        "1": {
            "split": "discovery",
            "image_width": 100,
            "image_height": 100,
            "file_name": "unused.jpg",
        }
    }
    treatment_plan = copy.deepcopy(plan)
    treatment_observed_prefix_sha256 = "treatment-observed-prefix"
    treatment_query_prefix_sha256 = "treatment-query-prefix"
    treatment_admission_receipt_id = visual.planner.admission_receipt_id(
        context_id="1:boundary-000",
        channel=visual.planner.CHANNEL_QUERY_SUFFIX,
        prefix_sha256=treatment_query_prefix_sha256,
    )
    treatment_plan.query_groups[group_id].update(
        {
            "observed_prefix_sha256": treatment_observed_prefix_sha256,
            "query_prefix_sha256": treatment_query_prefix_sha256,
            "admission_receipt_id": treatment_admission_receipt_id,
        }
    )
    treatment.scores[0].update(
        {
            "observed_prefix_sha256": treatment_observed_prefix_sha256,
            "query_prefix_sha256": treatment_query_prefix_sha256,
            "admission_receipt_id": treatment_admission_receipt_id,
        }
    )
    _write_json(treatment.directory / visual.RECEIPT_NAME, treatment.receipt)
    _write_jsonl(treatment.directory / visual.SCORES_NAME, treatment.scores)
    _write_json(baseline_dir / visual.RECEIPT_NAME, baseline_receipt)
    _write_jsonl(baseline_dir / visual.SCORES_NAME, [baseline_row])
    report = {
        "primary": {
            "owners": [
                {
                    "gt_owner_id": owner_id,
                    "treatment_support_u": True,
                    "treatment_support_context_ids_u": ["1:boundary-000"],
                }
            ]
        },
        "retention": {
            "confirmation_true_positive": {"lost": []},
            "resolved_support": {"lost": []},
        },
        "owner_role_context_admission": {
            "owners": {owner_id: {"allowed_context_ids": ["1:boundary-000"]}}
        },
    }
    report_path = tmp_path / "intervention-report.json"
    overlay_path = tmp_path / "overlay.json"
    _write_json(report_path, report)
    _write_json(overlay_path, overlay)
    return visual.Inputs(
        report_path=report_path,
        overlay_path=overlay_path,
        plan_dir=tmp_path / "plan",
        baseline_shard_root=baseline_dir.parent,
        treatment_shard_root=treatment.directory.parent,
        report=report,
        overlay=overlay,
        baseline_plan=plan,
        treatment_plan=treatment_plan,
        treatment_shards={"1": treatment},
    )


def test_specs_only_end_to_end_emits_specs_and_manifest(tmp_path: Path, monkeypatch):
    inputs = _specs_only_inputs(tmp_path)
    monkeypatch.setattr(visual, "load_inputs", lambda **_kwargs: inputs)
    output = tmp_path / "output"
    result = visual.main(
        [
            "--report",
            str(inputs.report_path),
            "--overlay",
            str(inputs.overlay_path),
            "--plan-dir",
            str(inputs.plan_dir),
            "--baseline-shard-root",
            str(inputs.baseline_shard_root),
            "--treatment-shard-root",
            str(inputs.treatment_shard_root),
            "--output-dir",
            str(output),
            "--specs-only",
        ]
    )
    assert result == 0
    specs = [json.loads(line) for line in (output / visual.SPECS_NAME).read_text().splitlines()]
    manifest = json.loads((output / visual.MANIFEST_NAME).read_text())
    assert [row["gt_owner_id"] for row in specs] == ["gt:1:0"]
    assert specs[0]["crop"]["shared_across_arms"] is True
    assert len(specs[0]["arms"]) == 2
    assert manifest["figure_count"] == 1


def test_failed_specs_only_run_emits_no_specs_or_manifest(tmp_path: Path, monkeypatch):
    inputs = _specs_only_inputs(tmp_path)
    inputs.treatment_shards["1"].scores[0]["request_id"] = "wrong"
    monkeypatch.setattr(visual, "load_inputs", lambda **_kwargs: inputs)
    output = tmp_path / "failed-output"
    result = visual.main(
        [
            "--report",
            str(inputs.report_path),
            "--overlay",
            str(inputs.overlay_path),
            "--plan-dir",
            str(inputs.plan_dir),
            "--baseline-shard-root",
            str(inputs.baseline_shard_root),
            "--treatment-shard-root",
            str(inputs.treatment_shard_root),
            "--output-dir",
            str(output),
            "--specs-only",
        ]
    )
    assert result == 2
    assert not (output / visual.SPECS_NAME).exists()
    assert not (output / visual.MANIFEST_NAME).exists()


def _load_inputs_composition_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    treatment_uses_base_prompt_identity: bool = False,
    invoke_load: bool = True,
) -> tuple[visual.Inputs | None, dict]:
    seed = _specs_only_inputs(tmp_path)
    baseline_plan = seed.baseline_plan
    image_id = "1"
    context_id = "1:boundary-000"
    group_id = f"{context_id}|person"
    current_prompt = [10, 11]
    treatment_prompt = [10, 11, 12]
    generated_prefix = [20]
    query_suffix = [30]
    current_observed = visual.sha256_json([*current_prompt, *generated_prefix])
    current_query = visual.sha256_json(
        [*current_prompt, *generated_prefix, *query_suffix]
    )
    treatment_observed = visual.sha256_json([*treatment_prompt, *generated_prefix])
    treatment_query = visual.sha256_json(
        [*treatment_prompt, *generated_prefix, *query_suffix]
    )
    suffix_sha = visual.sha256_json(query_suffix)
    current_prompt_sha = visual.sha256_json(current_prompt)
    treatment_prompt_sha = visual.sha256_json(treatment_prompt)
    current_media_sha = "current-media"
    treatment_media_sha = "treatment-media"
    capture_rules_sha = "capture-rules"
    baseline_plan.receipt["capture_rules_sha256"] = capture_rules_sha
    baseline_plan.images[image_id].update(
        {
            "prompt_token_ids": current_prompt,
            "prompt_token_ids_sha256": current_prompt_sha,
            "executed_media_sha256": current_media_sha,
            "coordinate_token_ids": [1, 2, 3, 4],
            "wrapper_token_ids": [5, 6, 7, 8],
        }
    )
    baseline_plan.contexts[context_id].update(
        {
            "generated_prefix_token_ids": generated_prefix,
            "generated_prefix_token_ids_sha256": visual.sha256_json(generated_prefix),
            "prompt_token_ids_sha256": current_prompt_sha,
            "observed_self_prefix_token_ids_sha256": current_observed,
            "observed_self_prefix_token_count": len(current_prompt) + len(generated_prefix),
        }
    )
    baseline_plan.categories = {"category:person": {"category_token_ids": [29]}}
    baseline_plan.query_groups[group_id].update(
        {
            "category_query_id": "category:person",
            "query_suffix_token_ids": query_suffix,
            "query_suffix_token_ids_sha256": suffix_sha,
            "observed_prefix_sha256": current_observed,
            "observed_prefix_token_count": 3,
            "query_prefix_sha256": current_query,
            "query_prefix_token_count": 4,
            "proposal_route_token_ids": [],
            "proposal_route_digest": None,
            "singleton_group_key": {
                "observed_prefix_sha256": current_observed,
                "query_prefix_sha256": current_query,
            },
        }
    )
    current_admission_id = visual.planner.admission_receipt_id(
        context_id=context_id,
        channel=visual.planner.CHANNEL_QUERY_SUFFIX,
        prefix_sha256=current_query,
    )
    baseline_plan.query_groups[group_id]["admission_receipt_id"] = current_admission_id

    overlay = {
        "schema_version": visual.prepare.OVERLAY_SCHEMA_VERSION,
        "intervention_unit_id": visual.prepare.INTERVENTION_UNIT_ID,
        "arm_id": visual.prepare.ARM_ID,
        "baseline_arm_id": visual.prepare.BASELINE_ARM_ID,
        "max_pixels": 200,
        "base": {
            "plan_receipt_content_sha256": "plan",
            "capture_rules_sha256": capture_rules_sha,
            "predecessor_run_root": "/pred",
        },
        "images": {
            image_id: {
                "current": {
                    "prompt_token_ids_sha256": current_prompt_sha,
                    "executed_media_sha256": current_media_sha,
                    "prompt_token_count": len(current_prompt),
                },
                "treatment": {
                    "prompt_token_ids": treatment_prompt,
                    "prompt_token_ids_sha256": treatment_prompt_sha,
                    "executed_media_sha256": treatment_media_sha,
                    "width": 200,
                    "height": 200,
                },
            }
        },
        "query_group_selection": {
            "query_group_ids_by_image": {image_id: [group_id]}
        },
    }
    overlay["overlay_content_sha256"] = visual.prepare.overlay_content_sha256(overlay)
    overlay_path = tmp_path / "real-overlay.json"
    _write_json(overlay_path, overlay)

    baseline_receipt_path = seed.baseline_shard_root / image_id / visual.RECEIPT_NAME
    baseline_score_path = seed.baseline_shard_root / image_id / visual.SCORES_NAME
    baseline_receipt = json.loads(baseline_receipt_path.read_text(encoding="utf-8"))
    baseline_score = json.loads(baseline_score_path.read_text(encoding="utf-8"))
    baseline_receipt["admission"]["receipts"][0].update(
        {
            "query_prefix_sha256": current_query,
            "admission_receipt_id": current_admission_id,
        }
    )
    baseline_score.update(
        {
            "observed_prefix_sha256": current_observed,
            "query_prefix_sha256": current_query,
            "query_suffix_token_ids_sha256": suffix_sha,
            "admission_receipt_id": current_admission_id,
        }
    )
    _write_json(baseline_receipt_path, baseline_receipt)
    _write_jsonl(baseline_score_path, [baseline_score])

    treatment = seed.treatment_shards[image_id]
    treatment_admission_id = visual.planner.admission_receipt_id(
        context_id=context_id,
        channel=visual.planner.CHANNEL_QUERY_SUFFIX,
        prefix_sha256=treatment_query,
    )
    row_observed = current_observed if treatment_uses_base_prompt_identity else treatment_observed
    row_query = current_query if treatment_uses_base_prompt_identity else treatment_query
    row_admission = (
        current_admission_id
        if treatment_uses_base_prompt_identity
        else treatment_admission_id
    )
    treatment.receipt["intervention"] = {
        "intervention_unit_id": visual.UNIT_ID,
        "arm_id": overlay["arm_id"],
        "baseline_arm_id": overlay["baseline_arm_id"],
        "overlay_content_sha256": overlay["overlay_content_sha256"],
        "base_plan_receipt_content_sha256": "plan",
    }
    treatment.scores[0].update(
        {
            "observed_prefix_sha256": row_observed,
            "query_prefix_sha256": row_query,
            "query_suffix_token_ids_sha256": suffix_sha,
            "admission_receipt_id": row_admission,
            "intervention": dict(treatment.receipt["intervention"]),
        }
    )
    _write_json(treatment.directory / visual.RECEIPT_NAME, treatment.receipt)
    _write_jsonl(treatment.directory / visual.SCORES_NAME, treatment.scores)

    capture_identity = {
        "status": "complete_uniform_capture_identity",
        "image_count": 1,
        "common": {},
        "per_image": {image_id: {}},
    }
    seals = visual.analyzer.build_capture_artifact_seals({image_id: treatment})
    report = dict(seed.report)
    report.update(
        {
            "schema_version": visual.REPORT_SCHEMA_VERSION,
            "status": "complete_uniform_capture_identity",
            "intervention_unit_id": visual.UNIT_ID,
            "arm_id": overlay["arm_id"],
            "baseline_arm_id": overlay["baseline_arm_id"],
            "overlay_content_sha256": overlay["overlay_content_sha256"],
            "base_plan_receipt_content_sha256": "plan",
            "predecessor_run_root": "/pred",
            "captured_images": [image_id],
            "capture_identity": capture_identity,
            "capture_artifact_seals": seals,
            "comparison_semantics": {
                "raw_logprob_compared_across_arms": False,
                "compared_quantity": "support_disposition_under_arm_local_calibration",
            },
        }
    )
    report["owner_role_context_admission"].update(
        {
            "selection_basis": "pre_treatment_overlay_and_predecessor_owner_summaries_only",
            "treatment_scores_used_for_context_selection": False,
        }
    )
    report_path = tmp_path / "real-report.json"
    _write_json(report_path, report)

    monkeypatch.setattr(
        visual.merge, "load_plan", lambda _path: copy.deepcopy(baseline_plan)
    )
    monkeypatch.setattr(
        visual.analyzer,
        "load_treatment_shards",
        lambda _root, _overlay: {image_id: treatment},
    )

    def admit_treatment_view(plan, shards):
        visual.analyzer.validate_treatment_score_row_joins(plan, treatment)
        group = plan.query_groups[group_id]
        row = treatment.scores[0]
        if row["observed_prefix_sha256"] != group["observed_prefix_sha256"]:
            raise visual.analyzer.AnalysisContractError(
                "observed prefix digest does not match its plan query group"
            )
        if row["query_prefix_sha256"] != group["query_prefix_sha256"]:
            raise visual.analyzer.AnalysisContractError(
                "query prefix digest does not match its plan query group"
            )
        return []

    monkeypatch.setattr(
        visual.analyzer, "build_treatment_owner_contexts", admit_treatment_view
    )
    monkeypatch.setattr(
        visual.analyzer,
        "capture_identity_summary",
        lambda _shards, _overlay: capture_identity,
    )
    loaded = None
    if invoke_load:
        loaded = visual.load_inputs(
            report_path=report_path,
            overlay_path=overlay_path,
            plan_dir=seed.plan_dir,
            baseline_shard_root=seed.baseline_shard_root,
            treatment_shard_root=seed.treatment_shard_root,
        )
    return loaded, {
        "report_path": report_path,
        "overlay_path": overlay_path,
        "plan_dir": seed.plan_dir,
        "baseline_shard_root": seed.baseline_shard_root,
        "treatment_shard_root": seed.treatment_shard_root,
        "current_observed": current_observed,
        "treatment_observed": treatment_observed,
    }


def test_real_load_inputs_and_cli_apply_overlay_to_treatment_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inputs, paths = _load_inputs_composition_fixture(tmp_path, monkeypatch)
    assert inputs is not None
    group_id = "1:boundary-000|person"
    assert inputs.baseline_plan.query_groups[group_id]["observed_prefix_sha256"] == (
        paths["current_observed"]
    )
    assert inputs.treatment_plan.query_groups[group_id]["observed_prefix_sha256"] == (
        paths["treatment_observed"]
    )
    output = tmp_path / "real-output"
    result = visual.main(
        [
            "--report",
            str(paths["report_path"]),
            "--overlay",
            str(paths["overlay_path"]),
            "--plan-dir",
            str(inputs.plan_dir),
            "--baseline-shard-root",
            str(inputs.baseline_shard_root),
            "--treatment-shard-root",
            str(inputs.treatment_shard_root),
            "--output-dir",
            str(output),
            "--specs-only",
        ]
    )
    assert result == 0
    spec = json.loads((output / visual.SPECS_NAME).read_text(encoding="utf-8"))
    assert spec["provenance"]["prompt_identity_by_arm"] == {
        "baseline": {
            "observed_prefix_sha256": paths["current_observed"],
            "query_prefix_sha256": inputs.baseline_plan.query_groups[group_id][
                "query_prefix_sha256"
            ],
        },
        "treatment": {
            "observed_prefix_sha256": paths["treatment_observed"],
            "query_prefix_sha256": inputs.treatment_plan.query_groups[group_id][
                "query_prefix_sha256"
            ],
        },
    }


def test_real_cli_rejects_treatment_row_with_base_prompt_identity_before_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    inputs, paths = _load_inputs_composition_fixture(
        tmp_path,
        monkeypatch,
        treatment_uses_base_prompt_identity=True,
        invoke_load=False,
    )
    assert inputs is None
    output = tmp_path / "failed-real-output"
    result = visual.main(
        [
            "--report",
            str(paths["report_path"]),
            "--overlay",
            str(paths["overlay_path"]),
            "--plan-dir",
            str(paths["plan_dir"]),
            "--baseline-shard-root",
            str(paths["baseline_shard_root"]),
            "--treatment-shard-root",
            str(paths["treatment_shard_root"]),
            "--output-dir",
            str(output),
            "--specs-only",
        ]
    )
    assert result == 2
    assert not (output / visual.SPECS_NAME).exists()
    assert not (output / visual.MANIFEST_NAME).exists()


def test_spatial_peaks_are_deterministic_and_suppress_same_peak():
    confidence = {"a": 0.5, "b": 0.3, "c": 0.2}
    boxes = {
        "a": [0, 0, 10, 10],
        "b": [1, 1, 11, 11],
        "c": [30, 30, 40, 40],
    }
    assert visual.select_spatial_peaks(confidence, boxes) == ["a", "c"]
