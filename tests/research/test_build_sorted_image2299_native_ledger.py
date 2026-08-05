from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

import scripts.research.build_sorted_image2299_native_ledger as ledger
from scripts.research.build_sorted_image2299_native_ledger import (
    DEFAULT_PANEL,
    DEFAULT_RUN_DIR,
    ExpectedContract,
    NativeLedgerContractError,
)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.fixture(scope="module")
def real_manifest() -> dict[str, Any]:
    return _json(DEFAULT_RUN_DIR / "run_manifest.json")


@pytest.fixture(scope="module")
def real_resolved() -> dict[str, Any]:
    return _json(DEFAULT_RUN_DIR / "configs/resolved.json")


@pytest.fixture(scope="module")
def real_gt_row() -> dict[str, Any]:
    return next(
        row for row in _jsonl(DEFAULT_RUN_DIR / "gt_vs_pred.jsonl") if row["row_id"] == ledger.TARGET_ROW_ID
    )


@pytest.fixture(scope="module")
def real_panel_row() -> dict[str, Any]:
    return next(
        row for row in _jsonl(DEFAULT_PANEL) if str(row["image_id"]) == ledger.TARGET_IMAGE_ID
    )


@pytest.fixture(scope="module")
def real_trace() -> list[dict[str, Any]]:
    return _jsonl(DEFAULT_RUN_DIR / "pred_token_trace.jsonl")


class _FixedDecoder:
    def __init__(self, text: str) -> None:
        self.text = text

    def decode(self, *_args: Any, **_kwargs: Any) -> str:
        return self.text


def test_rejects_exact_checkpoint_and_runtime_identity_drift(
    real_manifest: dict[str, Any], real_resolved: dict[str, Any]
) -> None:
    drifted = deepcopy(real_manifest)
    drifted["model_identity"]["adapter"]["adapter_path"] = "/wrong/step-4887/adapter"

    with pytest.raises(NativeLedgerContractError, match="adapter identity drift"):
        ledger._validate_runtime_contract(drifted, real_resolved, ExpectedContract())


def test_rejects_official_22_owner_gt(
    real_gt_row: dict[str, Any], real_panel_row: dict[str, Any]
) -> None:
    wrong = deepcopy(real_gt_row)
    wrong["gt"] = wrong["gt"][:22]

    with pytest.raises(NativeLedgerContractError, match="refined 46 owners"):
        ledger._validate_historical_gt(wrong, real_panel_row)


def test_rejects_non_natural_or_truncated_stop(
    real_trace: list[dict[str, Any]], real_gt_row: dict[str, Any]
) -> None:
    trace = deepcopy(real_trace)
    target = [
        row
        for row in trace
        if row.get("row_id") == ledger.TARGET_ROW_ID
        and row.get("trace_type") == "generated_token"
        and not row.get("is_pad")
    ]
    target[-1]["is_stop"] = False

    with pytest.raises(NativeLedgerContractError, match="non-natural/truncated stop"):
        ledger._reconstruct_generated_ids(
            trace,
            real_gt_row,
            _FixedDecoder(real_gt_row["raw_decode_text"]),
            ExpectedContract(),
        )


def test_rejects_generated_token_reconstruction_mismatch(
    real_trace: list[dict[str, Any]], real_gt_row: dict[str, Any]
) -> None:
    trace = deepcopy(real_trace)
    first = next(
        row
        for row in trace
        if row.get("row_id") == ledger.TARGET_ROW_ID
        and row.get("trace_type") == "generated_token"
        and row.get("generated_step_index") == 0
    )
    first["token_id"] += 1

    with pytest.raises(NativeLedgerContractError, match="ID digest drift"):
        ledger._reconstruct_generated_ids(
            trace,
            real_gt_row,
            _FixedDecoder(real_gt_row["raw_decode_text"]),
            ExpectedContract(),
        )


def test_rejects_reconstructed_prompt_hash_mismatch(
    real_manifest: dict[str, Any],
    real_resolved: dict[str, Any],
) -> None:
    image_plan = next(
        row
        for row in _jsonl(DEFAULT_RUN_DIR / "image_plan.jsonl")
        if row["row_id"] == ledger.TARGET_ROW_ID
    )
    example = next(
        row for row in ledger.load_raw_examples(DEFAULT_PANEL) if row.example_id == ledger.TARGET_ROW_ID
    )

    def wrong_prompt(*_args: Any) -> tuple[list[int], _FixedDecoder, dict[str, Any]]:
        return [1, 2, 3], _FixedDecoder(""), {}

    with pytest.raises(NativeLedgerContractError, match="reconstructed prompt token IDs"):
        ledger._validate_prompt(
            real_manifest,
            real_resolved,
            image_plan,
            example,
            ExpectedContract(),
            wrong_prompt,
        )


def test_rejects_inherited_matcher_ambiguity(tmp_path: Path) -> None:
    owners = [
        {
            "gt_owner_id": f"gt:2299:{index}",
            "original_annotation_index": index,
            "description": "person",
            "normalized_description": "person",
            "official_coco_category_id": 1,
            "bbox_xyxy": [0, 0, 10, 10],
        }
        for index in range(2)
    ]
    predictions = [
        {
            "description": "person",
            "bbox": [0, 0, 10, 10],
            "generated_order": index,
            "object_span_id": f"span-{index}",
            "raw_span_sha256": "0" * 64,
        }
        for index in range(2)
    ]

    with pytest.raises(NativeLedgerContractError, match="matcher ambiguity"):
        ledger._build_ledgers(
            owners,
            predictions,
            tmp_path / "source.jsonl",
            "0" * 64,
            "1" * 64,
            "2" * 64,
        )


def test_create_or_identical_publication(tmp_path: Path) -> None:
    payload = {"serialized": {"receipt.json": b"{}\n", "greedy.json": b"{}\n"}}
    output = tmp_path / "published"

    assert ledger.publish_create_or_identical(output, payload) == "created"
    assert ledger.publish_create_or_identical(output, payload) == "identical"
    (output / "greedy.json").write_bytes(b'{"drift":true}\n')
    with pytest.raises(NativeLedgerContractError, match="not identical"):
        ledger.publish_create_or_identical(output, payload)


def test_exact_real_artifact_validate_only(capsys: pytest.CaptureFixture[str]) -> None:
    assert ledger.main(["--validate-only"]) == 0
    result = json.loads(capsys.readouterr().out)

    assert result["status"] == "validated"
    assert result["counts"] == {
        "owners": 46,
        "predictions": 24,
        "matched": 19,
        "false_negatives": 27,
        "false_positives": 5,
        "matcher_ambiguity_classes": 0,
    }
    assert len(result["receipt_content_sha256"]) == 64


def test_receipt_is_self_sealed_and_task0_shapes_are_stable() -> None:
    payload = ledger.build_native_ledger()
    receipt = payload["receipt"]
    expected_seal = hashlib.sha256(
        ledger._canonical_json_bytes(
            {key: value for key, value in receipt.items() if key != "receipt_content_sha256"}
        )
    ).hexdigest()

    assert receipt["receipt_content_sha256"] == expected_seal
    assert set(payload["serialized"]) == {
        "owner-ledger.jsonl",
        "prediction-row-ledger.jsonl",
        "greedy.json",
        "receipt.json",
    }
    assert payload["greedy"]["schema_version"] == "current_seeded_sampled_rollouts.v1"
    assert payload["greedy"]["rollout_count"] == 1
    assert payload["greedy"]["rollouts"][0]["stop_reason"] == "im_end"
    assert all("decision_eligibility" in row for row in payload["owner_ledger"])
    assert all("original_row_index" in row for row in payload["prediction_ledger"])
