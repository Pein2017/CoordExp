#!/usr/bin/env python3
"""Build one-image S0 ledgers from a current exact greedy rollout artifact.

This is the checkpoint-neutral counterpart of the historical sorted admission
script. It consumes ``current_seeded_sampled_rollouts.v1`` evidence, applies
the frozen ambiguity-neutral one-to-one matcher to the refined image-2299 row,
and publishes only the four files required by the owner-accessibility planner.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_basin_census as census  # noqa: E402
from scripts.research import build_sorted_owner_accessibility_census_plan as legacy  # noqa: E402
from src.config.inference import load_infer_config  # noqa: E402


SCHEMA_VERSION = "image2299-current-native-ledger.v1"
RECEIPT_SCHEMA_VERSION = "image2299-current-native-ledger-receipt.v1"
TARGET_IMAGE_ID = "2299"
EXPECTED_OWNER_COUNT = 46
EXPECTED_CATEGORY_COUNTS = {"person": 38, "tie": 8}
EXPECTED_PANEL_SHA256 = "01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8"
EXPECTED_AUTHORITY_ROW_SHA256 = "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"
EXPECTED_IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"
EXPECTED_EXECUTED_MEDIA_SHA256 = "02e1d433f77eb2e4a7469c12aa0d795da42ad389614395698d1a4468b99a245e"
EXPECTED_HORIZON = 3084


class NativeLedgerContractError(ValueError):
    """Raised before incompatible current evidence can be published."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise NativeLedgerContractError(f"{label} is unreadable at {path}") from exc
    if not isinstance(value, Mapping):
        raise NativeLedgerContractError(f"{label} is not a JSON object")
    return dict(value)


def _jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(legacy.canonical_json_bytes(row) + b"\n" for row in rows)


def _output_files(
    owner_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    greedy_payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
) -> dict[str, bytes]:
    return {
        "owner-ledger.jsonl": _jsonl_bytes(owner_rows),
        "prediction-row-ledger.jsonl": _jsonl_bytes(prediction_rows),
        "greedy.json": legacy.canonical_json_bytes(greedy_payload) + b"\n",
        "receipt.json": legacy.canonical_json_bytes(receipt) + b"\n",
    }


def _panel_contract(panel_path: Path, panel_receipt_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if _sha256_file(panel_path) != EXPECTED_PANEL_SHA256:
        raise NativeLedgerContractError("refined-13 panel digest drifted")
    records, owners_by_image, _ = census._load_panel(panel_path)
    if TARGET_IMAGE_ID not in {str(row["image_id"]) for row in records} or TARGET_IMAGE_ID not in owners_by_image:
        raise NativeLedgerContractError("refined-13 panel lacks image 2299")
    owners = list(owners_by_image[TARGET_IMAGE_ID])
    counts = Counter(str(row["normalized_description"]) for row in owners)
    if len(owners) != EXPECTED_OWNER_COUNT or dict(sorted(counts.items())) != EXPECTED_CATEGORY_COUNTS:
        raise NativeLedgerContractError("image 2299 is not 46 = 38 person + 8 tie")
    receipt = _read_json(panel_receipt_path, "panel admission receipt")
    if receipt.get("output", {}).get("jsonl_sha256") != EXPECTED_PANEL_SHA256:
        raise NativeLedgerContractError("panel receipt does not bind the refined-13 panel")
    if receipt.get("inputs", {}).get("authority", {}).get("target_line_sha256") != EXPECTED_AUTHORITY_ROW_SHA256:
        raise NativeLedgerContractError("panel receipt authority row drifted")
    if receipt.get("inputs", {}).get("target_image", {}).get("sha256") != EXPECTED_IMAGE_SHA256:
        raise NativeLedgerContractError("panel receipt image bytes drifted")
    return owners, receipt


def _checkpoint_contract(
    infer_config_path: Path,
    rollout: Mapping[str, Any],
    checkpoint_role: str,
) -> tuple[dict[str, str], str]:
    resolved = load_infer_config(infer_config_path)
    config = resolved.config
    expected = {
        "base_model_path": str(config.model.base_model),
        "adapter_path": str(config.adapter.path),
        "embedding_delta_path": str(config.embedding_delta.path),
    }
    actual = dict(rollout["model_components"])
    if actual != expected:
        raise NativeLedgerContractError(
            f"rollout components disagree with infer config: {actual!r} != {expected!r}"
        )
    if config.backend.type != "hf" or config.backend.hf.attn_implementation != "sdpa":
        raise NativeLedgerContractError("infer config is not HF SDPA")
    if str(config.model.dtype) != "fp32":
        raise NativeLedgerContractError("infer config is not fp32")
    if config.generation.max_new_tokens != EXPECTED_HORIZON:
        raise NativeLedgerContractError("infer config horizon is not 3084")
    if float(config.generation.repetition_penalty) != 1.0:
        raise NativeLedgerContractError("infer config repetition penalty is not 1.0")
    if checkpoint_role == "random_step4887":
        joined = json.dumps(expected, sort_keys=True)
        if "desc_first_random_pure_ce_typegate" not in joined or "step-4887" not in joined:
            raise NativeLedgerContractError("random profile does not bind the original random step-4887 checkpoint")
        if str(config.template.object_ordering) != "random":
            raise NativeLedgerContractError("random profile infer config does not declare random training ordering")
    elif checkpoint_role == "geo_sorted_step4887":
        joined = json.dumps(expected, sort_keys=True)
        if "geo_sorted_pure_ce_typegate" not in joined or "step-4887" not in joined:
            raise NativeLedgerContractError("sorted profile does not bind geo-sorted step-4887")
    else:
        raise NativeLedgerContractError(f"unsupported checkpoint role {checkpoint_role!r}")
    return expected, resolved.fingerprint


def build_artifacts(
    *,
    panel_path: Path,
    panel_receipt_path: Path,
    rollout_path: Path,
    infer_config_path: Path,
    checkpoint_role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    owners, panel_receipt = _panel_contract(panel_path, panel_receipt_path)
    try:
        loaded = census._load_rollout_artifact(rollout_path, mode="greedy", aliases={})
    except census.CensusContractError as exc:
        raise NativeLedgerContractError(str(exc)) from exc
    if len(loaded["rows"]) != 1:
        raise NativeLedgerContractError("current rollout must contain exactly one trajectory")
    trajectory = dict(loaded["rows"][0])
    if trajectory["image_id"] != TARGET_IMAGE_ID or trajectory["seed"] != 0:
        raise NativeLedgerContractError("current rollout is not seed-0 image 2299")
    if trajectory["stop_reason"] != "im_end":
        raise NativeLedgerContractError("native rollout did not stop naturally with im_end")
    if trajectory["invalid_predictions"] or trajectory["dropped_predictions"]:
        raise NativeLedgerContractError("native rollout contains invalid or dropped prediction rows")
    components, resolved_fingerprint = _checkpoint_contract(
        infer_config_path, loaded, checkpoint_role
    )
    payload = _read_json(rollout_path, "current greedy rollout")
    raw_rollout = payload["rollouts"][0]
    generated_ids = raw_rollout.get("generated_token_ids")
    if not isinstance(generated_ids, list) or len(generated_ids) >= EXPECTED_HORIZON:
        raise NativeLedgerContractError("native generated IDs are absent or reach the horizon")
    if raw_rollout.get("executed_media_sha256") != EXPECTED_EXECUTED_MEDIA_SHA256:
        raise NativeLedgerContractError(
            "executed RGB media is not the frozen image-2299 materialization"
        )

    role_tag = "random" if checkpoint_role == "random_step4887" else "sorted"
    old_to_new: dict[str, str] = {}
    predictions: list[dict[str, Any]] = []
    for prediction in trajectory["predictions"]:
        item = dict(prediction)
        new_id = (
            f"pred:{role_tag}:greedy:0:{TARGET_IMAGE_ID}:"
            f"{int(item['original_row_index'])}"
        )
        old_to_new[str(item["pred_row_id"])] = new_id
        item["pred_row_id"] = new_id
        predictions.append(item)
    trajectory["trajectory_id"] = f"trajectory:{role_tag}:rp1.00:greedy:0:{TARGET_IMAGE_ID}"
    trajectory["predictions"] = predictions
    match = census._trajectory_receipt(trajectory, owners, {})
    receipts_by_prediction = {
        str(row["pred_row_id"]): row for row in match["prediction_receipts"]
    }

    owner_rows = [
        {
            "schema_version": census.OWNER_LEDGER_SCHEMA_VERSION,
            "gt_owner_id": owner["gt_owner_id"],
            "diagnostic_owner_id": owner["diagnostic_owner_id"],
            "image_id": owner["image_id"],
            "original_annotation_index": owner["original_annotation_index"],
            "description": owner["description"],
            "normalized_description": owner["normalized_description"],
            "official_coco_category_id": owner["official_coco_category_id"],
            "bbox_xyxy": list(owner["bbox_xyxy"]),
            "decision_eligibility": {
                "greedy_natural": {"eligible": True, "status": "eligible"}
            },
        }
        for owner in owners
    ]
    prediction_rows: list[dict[str, Any]] = []
    for prediction in predictions:
        evidence = receipts_by_prediction[str(prediction["pred_row_id"])]
        prediction_rows.append(
            {
                "schema_version": census.PREDICTION_LEDGER_SCHEMA_VERSION,
                "pred_row_id": prediction["pred_row_id"],
                "image_id": TARGET_IMAGE_ID,
                "original_row_index": int(prediction["original_row_index"]),
                "decode_mode": "greedy",
                "normalized_description": prediction["normalized_description"],
                "bbox_xyxy": list(prediction["bbox_xyxy"]),
                "raw_span_sha256": prediction["raw_span_sha256"],
                "strict_match_status": evidence["strict_match_status"],
                "strict_match_gt_owner_id": evidence["strict_match_gt_owner_id"],
                "strict_match_iou": evidence["intersection_over_union"],
                "max_any_owner_iou": evidence["max_any_owner_iou"],
                "ambiguity_receipt_ids": evidence["ambiguity_receipt_ids"],
            }
        )

    # Publish the exact current artifact payload; only canonical JSON formatting
    # changes. The complete token, prompt, parser, and runtime identities remain.
    greedy_payload = payload
    output_without_receipt = {
        "owner-ledger.jsonl": _jsonl_bytes(owner_rows),
        "prediction-row-ledger.jsonl": _jsonl_bytes(prediction_rows),
        "greedy.json": legacy.canonical_json_bytes(greedy_payload) + b"\n",
    }
    admission_contract = {
        "target_image_id": TARGET_IMAGE_ID,
        "checkpoint_role": checkpoint_role,
        "runtime_identity": {
            "backend": "hf",
            "attention": "sdpa",
            "dtype": "torch.float32",
            "model_components": components,
            "resolved_infer_config_fingerprint": resolved_fingerprint,
        },
        "generated_identity": {
            "stop_reason": "im_end",
            "generated_token_count_without_im_end": len(generated_ids),
            "horizon_nonbinding": True,
            "prospective_horizon": EXPECTED_HORIZON,
            "generated_token_ids_sha256": raw_rollout["generated_token_ids_sha256"],
            "prompt_token_ids_sha256": raw_rollout["prompt_token_ids_sha256"],
        },
        "matcher": {
            "schema_version": census.MATCHER_SCHEMA_VERSION,
            "iou_threshold": census.IOU_THRESHOLD,
            "ambiguity_neutral": True,
        },
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": "2026-08-04-random-image2299-matched-mechanism-contrast"
        if checkpoint_role == "random_step4887"
        else "2026-08-04-sorted-image2299-prospective-mechanism-extension",
        "status": "completed",
        "execution_status": "captured",
        "admission_contract": admission_contract,
        "admission_contract_sha256": legacy.sha256_json(admission_contract),
        "bindings": {
            "panel_sha256": EXPECTED_PANEL_SHA256,
            "panel_receipt_sha256": _sha256_file(panel_receipt_path),
            "panel_receipt_content_sha256": panel_receipt.get("receipt_content_sha256"),
            "authority_row_sha256": EXPECTED_AUTHORITY_ROW_SHA256,
            "image_sha256": EXPECTED_IMAGE_SHA256,
            "executed_rgb_media_sha256": EXPECTED_EXECUTED_MEDIA_SHA256,
            "rollout_source_sha256": _sha256_file(rollout_path),
            "infer_config_sha256": _sha256_file(infer_config_path),
        },
        "counts": {
            "owner_count": len(owner_rows),
            "prediction_row_count": len(prediction_rows),
            "strict_matched_owner_count": len(match["committed_owner_ids"]),
            "strict_false_negative_owner_count": len(owner_rows) - len(match["committed_owner_ids"]),
            "strict_unmatched_prediction_count": sum(
                row["strict_match_status"] == "unmatched" for row in prediction_rows
            ),
            "ambiguity_neutral_owner_count": len(match["neutral_owner_ids"]),
        },
        "outputs": {
            name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
            for name, content in sorted(output_without_receipt.items())
        },
    }
    receipt["receipt_content_sha256"] = legacy.sha256_json(receipt)
    return owner_rows, prediction_rows, greedy_payload, receipt


def commit_artifacts(
    owner_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    greedy_payload: Mapping[str, Any],
    receipt: Mapping[str, Any],
    output_dir: Path,
) -> None:
    files = _output_files(owner_rows, prediction_rows, greedy_payload, receipt)

    def validate_existing() -> None:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise NativeLedgerContractError(f"output is not a regular directory: {output_dir}")
        entries = {path.name: path for path in output_dir.iterdir()}
        if set(entries) != set(files):
            raise NativeLedgerContractError("existing S0 artifact set is not exact")
        mismatched = [name for name, content in files.items() if entries[name].read_bytes() != content]
        if mismatched:
            raise NativeLedgerContractError(f"existing S0 artifacts differ: {sorted(mismatched)}")

    if output_dir.exists() or output_dir.is_symlink():
        validate_existing()
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    published = False
    try:
        for name, content in files.items():
            (staging / name).write_bytes(content)
        os.replace(staging, output_dir)
        published = True
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--panel-receipt", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-role",
        choices=("random_step4887", "geo_sorted_step4887"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        product = build_artifacts(
            panel_path=args.panel,
            panel_receipt_path=args.panel_receipt,
            rollout_path=args.rollout,
            infer_config_path=args.infer_config,
            checkpoint_role=args.checkpoint_role,
        )
        commit_artifacts(*product, args.output_dir)
    except (NativeLedgerContractError, census.CensusContractError, OSError) as exc:
        print(f"native-ledger contract error: {exc}", file=sys.stderr)
        return 2
    receipt = product[-1]
    print(
        "image2299 native ledger: "
        f"tp={receipt['counts']['strict_matched_owner_count']} "
        f"fn={receipt['counts']['strict_false_negative_owner_count']} "
        f"rows={receipt['counts']['prediction_row_count']}"
    )
    print(f"receipt_content_sha256 {receipt['receipt_content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
