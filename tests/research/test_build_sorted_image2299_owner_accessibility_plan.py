from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from scripts.research import build_sorted_image2299_owner_accessibility_plan as subject
from scripts.research import build_sorted_owner_accessibility_census_plan as legacy
from scripts.research import score_sorted_owner_accessibility_census_shard as scorer


def _write_json(path: Path, value: object) -> None:
    path.write_bytes(legacy.canonical_json_bytes(value) + b"\n")


def _coord_bins(obj: dict[str, object]) -> list[int]:
    return [int(str(value).removeprefix("<|coord_").removesuffix("|>")) for value in obj["bbox_2d"]]


def _fixture_sources(tmp_path: Path, *, prediction_count: int = 2) -> subject.SourcePaths:
    panel_rows = [json.loads(line) for line in subject.PANEL.read_text().splitlines()]
    panel = next(row for row in panel_rows if str(row["image_id"]) == subject.IMAGE_ID)
    canvas = legacy.Canvas(int(panel["width"]), int(panel["height"]))
    owner_rows: list[dict[str, object]] = []
    for index, obj in enumerate(panel["objects"]):
        box = list(canvas.bins_to_pixel(_coord_bins(obj)))
        owner_rows.append(
            {
                "schema_version": "task0-compatible-owner.v2",
                "gt_owner_id": f"gt:2299:{index}",
                "image_id": "2299",
                "normalized_description": obj["desc"],
                "description": obj["desc"],
                "official_coco_category_id": obj["category_id"],
                "original_annotation_index": index,
                "bbox_xyxy": box,
                "decision_eligibility": {
                    "greedy_natural": {"eligible": True, "status": "eligible"}
                },
            }
        )
    root = tmp_path / "s0"
    root.mkdir()
    (root / "owner-ledger.jsonl").write_bytes(
        b"".join(legacy.canonical_json_bytes(row) + b"\n" for row in owner_rows)
    )

    selected = ([0, 8] + [index for index in range(46) if index not in {0, 8}])[
        :prediction_count
    ]
    generated: list[int] = []
    predictions: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    category_tokens = {"person": 8987, "tie": 8201}
    for row_index, owner_index in enumerate(selected):
        owner = owner_rows[owner_index]
        bins = _coord_bins(panel["objects"][owner_index])
        span = [
            legacy.OBJECT_REF_START,
            category_tokens[str(owner["normalized_description"])],
            legacy.OBJECT_REF_END,
            legacy.BOX_START,
            *legacy.coord_token_ids(bins),
            legacy.BOX_END,
        ]
        generated.extend(span)
        raw_digest = legacy.sha256_json(span)
        predictions.append(
            {
                "description": owner["normalized_description"],
                "bbox": owner["bbox_xyxy"],
                "raw_span_sha256": raw_digest,
            }
        )
        prediction_rows.append(
            {
                "schema_version": "task0-compatible-prediction.v2",
                "pred_row_id": f"pred:2299:{row_index}",
                "image_id": "2299",
                "original_row_index": row_index,
                "decode_mode": "greedy",
                "normalized_description": owner["normalized_description"],
                "bbox_xyxy": owner["bbox_xyxy"],
                "raw_span_sha256": raw_digest,
                "strict_match_status": "matched",
                "strict_match_gt_owner_id": owner["gt_owner_id"],
            }
        )
    (root / "prediction-row-ledger.jsonl").write_bytes(
        b"".join(legacy.canonical_json_bytes(row) + b"\n" for row in prediction_rows)
    )
    prompt = [11, 12, 13]
    greedy = {
        "config": {
            "decode_mode": "greedy",
            "max_new_tokens": 3084,
            "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "model_identity": {"backend": "hf"},
        "rollouts": [
            {
                "image_id": 2299,
                "decode_mode": "greedy",
                "stop_reason": "im_end",
                "executed_media_sha256": subject.IMAGE_SHA256,
                "prompt_token_ids": prompt,
                "prompt_token_ids_sha256": legacy.sha256_json(prompt),
                "generated_token_ids": generated,
                "generated_token_ids_sha256": legacy.sha256_json(generated),
                "seed": 0,
                "predictions": {
                    "parse_status": "accepted",
                    "dropped_prediction_count": 0,
                    "predictions": predictions,
                },
            }
        ],
    }
    _write_json(root / "greedy.json", greedy)
    admission_contract = {
        "target_image_id": 2299,
        "runtime_identity": {
            "backend": "hf",
            "attention": "sdpa",
            "model": "/checkpoint/geo_sorted/step-4887/adapter",
            "dtype": "torch.float32",
        },
        "generated_identity": {
            "horizon_nonbinding": True,
            "prospective_horizon": 3084,
        },
    }
    receipt = {
        "schema_version": "image2299-s0-fixture.v1",
        "status": "completed",
        "admission_contract": admission_contract,
        "admission_contract_sha256": legacy.sha256_json(admission_contract),
        "bindings": {
            "panel": subject.PANEL_SHA256,
            "authority_row": subject.AUTHORITY_ROW_SHA256,
            "image": subject.IMAGE_SHA256,
            "infer_config": subject.INFER_CONFIG_SHA256,
        },
        "outputs": {
            name: {"sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
            for name in (
                "owner-ledger.jsonl",
                "prediction-row-ledger.jsonl",
                "greedy.json",
            )
        },
    }
    receipt["receipt_content_sha256"] = legacy.sha256_json(receipt)
    _write_json(root / "receipt.json", receipt)
    return subject.SourcePaths(
        panel=subject.PANEL,
        panel_receipt=subject.PANEL_RECEIPT,
        owner_ledger=root / "owner-ledger.jsonl",
        prediction_ledger=root / "prediction-row-ledger.jsonl",
        greedy=root / "greedy.json",
        s0_receipt=root / "receipt.json",
        infer_config=subject.INFER_CONFIG,
    )


def test_builds_scorer_compatible_one_image_plan(tmp_path: Path) -> None:
    plan = subject.build_plan(_fixture_sources(tmp_path))
    output = tmp_path / "plan"
    subject.commit_plan(plan, output)

    loaded = scorer.load_plan(output)
    first_group = next(
        group_id
        for group_id, row in loaded.query_groups.items()
        if row["status"] == "admitted"
    )
    item = scorer.resolve_work_item(loaded, first_group)

    assert item.image_id == "2299"
    assert plan.receipt["schema_version"] == legacy.PLAN_SCHEMA_VERSION
    assert plan.receipt["unit_id"] == legacy.UNIT_ID
    assert plan.receipt["extension_unit_id"] == subject.EXTENSION_UNIT_ID
    assert plan.receipt["census_shape"]["owner_count"] == 46
    assert plan.receipt["denominator_contract"] == {
        "prospective_image2299_owner_count": 46,
        "legacy_12_owner_count_unchanged": 346,
        "legacy_12_eligible_native_fn_denominator_unchanged": 202,
        "pooled_13_image_denominator_created": False,
        "report_slices_separately": True,
    }
    assert plan.receipt["candidate_bank_contract"]["logical_roles"] == list(
        legacy.LOGICAL_ROLES
    )
    assert all(
        row["query_suffix_token_ids"][-1] == legacy.BOX_START
        for row in plan.query_groups
        if row["status"] == "admitted"
    )
    # Create-or-identical is idempotent.
    assert subject.commit_plan(plan, output)["receipt.json"]


def test_builds_random_checkpoint_profile_without_changing_scorer_abi(
    tmp_path: Path,
) -> None:
    sources = _fixture_sources(tmp_path)
    random_config = Path(
        "configs/coordexp_swift/infer/"
        "qwen3_vl_2b_desc_first_random_step4887_human_refined13_hf_fp32_rp1p0.yaml"
    )
    receipt = json.loads(sources.s0_receipt.read_text())
    receipt["admission_contract"]["runtime_identity"]["model"] = (
        "/checkpoint/desc_first_random_pure_ce_typegate/step-4887/adapter"
    )
    receipt["admission_contract_sha256"] = legacy.sha256_json(
        receipt["admission_contract"]
    )
    receipt["bindings"]["infer_config"] = subject.RANDOM_INFER_CONFIG_SHA256
    receipt["receipt_content_sha256"] = legacy.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    _write_json(sources.s0_receipt, receipt)
    random_sources = subject.SourcePaths(
        **{**sources.__dict__, "infer_config": random_config}
    )

    plan = subject.build_plan(
        random_sources,
        checkpoint_profile="random_step4887",
        extension_unit_id=subject.RANDOM_EXTENSION_UNIT_ID,
    )

    assert plan.receipt["unit_id"] == legacy.UNIT_ID
    assert plan.receipt["extension_unit_id"] == subject.RANDOM_EXTENSION_UNIT_ID
    assert plan.receipt["checkpoint_profile"] == "random_step4887"
    assert plan.receipt["runtime_gate"]["target_order"] == (
        "random_training_adapter_native_decode"
    )
    assert {row["split"] for row in plan.owners} == {subject.RANDOM_SPLIT}


def test_fails_closed_on_panel_digest_drift(tmp_path: Path) -> None:
    sources = _fixture_sources(tmp_path)
    panel = tmp_path / "drifted.jsonl"
    panel.write_bytes(sources.panel.read_bytes() + b"\n")
    drifted = subject.SourcePaths(**{**sources.__dict__, "panel": panel})
    with pytest.raises(subject.PlanContractError, match="panel digest mismatch"):
        subject.build_plan(drifted)


def test_fails_closed_on_truncated_native_rollout(tmp_path: Path) -> None:
    sources = _fixture_sources(tmp_path)
    payload = json.loads(sources.greedy.read_text())
    payload["rollouts"][0]["stop_reason"] = "length"
    _write_json(sources.greedy, payload)
    receipt = json.loads(sources.s0_receipt.read_text())
    receipt["outputs"]["greedy.json"]["sha256"] = hashlib.sha256(
        sources.greedy.read_bytes()
    ).hexdigest()
    receipt["receipt_content_sha256"] = legacy.sha256_json(
        {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
    )
    _write_json(sources.s0_receipt, receipt)
    with pytest.raises(subject.PlanContractError, match="naturally stopped"):
        subject.build_plan(sources)


@pytest.mark.parametrize("contamination", ["missing_file", "foreign_subdir"])
def test_commit_plan_requires_exact_existing_artifact_set(
    tmp_path: Path, contamination: str
) -> None:
    plan = subject.build_plan(_fixture_sources(tmp_path))
    output = tmp_path / "plan"
    subject.commit_plan(plan, output)

    if contamination == "missing_file":
        (output / "receipt.json").unlink()
    else:
        (output / "foreign").mkdir()

    with pytest.raises(subject.PlanContractError, match="artifact set is not exact"):
        subject.commit_plan(plan, output)

    if contamination == "missing_file":
        assert not (output / "receipt.json").exists()
    else:
        assert (output / "foreign").is_dir()
