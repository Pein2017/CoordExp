#!/usr/bin/env python3
"""Build the prospective image-2299 owner-accessibility scoring plan.

The output intentionally uses the legacy census plan schema and unit identity:
that is the compatibility boundary accepted by
``score_sorted_owner_accessibility_census_shard.py``.  Prospective identity and
the separate-denominator contract are carried as additional receipt fields.
No legacy default, split, count, or source constant is modified.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterator

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.research import build_sorted_owner_accessibility_census_plan as legacy  # noqa: E402


EXTENSION_UNIT_ID = "2026-08-04-sorted-image2299-prospective-mechanism-extension"
RANDOM_EXTENSION_UNIT_ID = "2026-08-04-random-image2299-matched-mechanism-contrast"
IMAGE_ID = "2299"
SPLIT = "prospective_2299"
RANDOM_SPLIT = "prospective_2299_random"

ROOT = Path("/data/CoordExp")
OUTPUTS = ROOT / "outputs/research/qwen3-vl-dense-enumeration"
PANEL = (
    OUTPUTS
    / "2026-08-04-sorted-prospective-13-image-panel-admission"
    / "evaluation-inputs/human-refined-13.coord.jsonl"
)
PANEL_RECEIPT = PANEL.parents[1] / "receipt.json"
INFER_CONFIG = (
    REPO_ROOT
    / "configs/coordexp_swift/infer"
    / "qwen3_vl_2b_desc_first_geo_sorted_step4887_human_refined12_hf_fp32_rp1p0.yaml"
)

PANEL_SHA256 = "01086b139fa23983697492fdb535b5154429277803e8f12b243f9a031d1451f8"
AUTHORITY_ROW_SHA256 = "ce19853c74a595f22cc183ce450e561f2da3216e54a1e499cfbca1be7e1c425b"
IMAGE_SHA256 = "cd7199a37188c9ac6481520175866cbd78fa6fba35f4290bb0b60c78afcb2df3"
INFER_CONFIG_SHA256 = "e4095437844eeeae0de3472d7c20e016476e595e952b79b9610477157fb16adc"
RANDOM_INFER_CONFIG_SHA256 = "d12878e7e17bd76cadffca5afb95cd0f9e131073b1100048be94ad9f3fd175ae"
EXPECTED_OWNER_COUNT = 46
EXPECTED_CATEGORY_COUNTS = {"person": 38, "tie": 8}
LEGACY_OWNER_COUNT = 346
LEGACY_ELIGIBLE_FN_DENOMINATOR = 202
MAX_NEW_TOKENS = 3084


class PlanContractError(ValueError):
    """A frozen prospective input or inherited plan invariant changed."""


@dataclass(frozen=True)
class SourcePaths:
    panel: Path
    panel_receipt: Path
    owner_ledger: Path
    prediction_ledger: Path
    greedy: Path
    s0_receipt: Path
    infer_config: Path = INFER_CONFIG


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise PlanContractError(f"input is unreadable: {path}") from exc
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanContractError(f"{label} is unreadable at {path}") from exc
    if not isinstance(value, Mapping):
        raise PlanContractError(f"{label} is not a JSON object")
    return dict(value)


def _read_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise PlanContractError(f"{label} is unreadable at {path}") from exc
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PlanContractError(f"{label} line {number} is invalid JSON") from exc
        if not isinstance(value, Mapping):
            raise PlanContractError(f"{label} line {number} is not an object")
        rows.append(dict(value))
    return rows


def _verify_content_digest(document: Mapping[str, Any], field: str, label: str) -> str:
    declared = document.get(field)
    if not isinstance(declared, str):
        raise PlanContractError(f"{label} lacks {field}")
    reconstructed = legacy.sha256_json({k: v for k, v in document.items() if k != field})
    if reconstructed != declared:
        raise PlanContractError(f"{label} does not reconstruct its {field}")
    return declared


def _load_panel(sources: SourcePaths) -> tuple[dict[str, dict[str, Any]], str]:
    actual = _sha256_file(sources.panel)
    if actual != PANEL_SHA256:
        raise PlanContractError(f"13-image panel digest mismatch: {actual} != {PANEL_SHA256}")
    receipt = _read_json(sources.panel_receipt, "panel admission receipt")
    receipt_digest = _verify_content_digest(receipt, "receipt_content_sha256", "panel receipt")
    if receipt.get("output", {}).get("jsonl_sha256") != PANEL_SHA256:
        raise PlanContractError("panel receipt does not bind the frozen 13-image panel")
    if receipt.get("inputs", {}).get("authority", {}).get("target_line_sha256") != AUTHORITY_ROW_SHA256:
        raise PlanContractError("image-2299 authority-row digest drifted")
    if receipt.get("inputs", {}).get("target_image", {}).get("sha256") != IMAGE_SHA256:
        raise PlanContractError("panel receipt image-2299 byte digest drifted")
    if receipt.get("target_owner_counts") != {
        "by_desc": EXPECTED_CATEGORY_COUNTS,
        "person": 38,
        "tie": 8,
        "total": 46,
    }:
        raise PlanContractError("panel receipt no longer proves 46 = 38 person + 8 tie")

    matches = [row for row in _read_jsonl(sources.panel, "13-image panel") if str(row.get("image_id")) == IMAGE_ID]
    if len(matches) != 1:
        raise PlanContractError(f"panel contains {len(matches)} image-2299 rows, expected one")
    row = matches[0]
    objects = row.get("objects")
    if not isinstance(objects, list):
        raise PlanContractError("image-2299 panel row lacks objects")
    counts = Counter(str(obj.get("desc")) for obj in objects if isinstance(obj, Mapping))
    if len(objects) != EXPECTED_OWNER_COUNT or dict(sorted(counts.items())) != EXPECTED_CATEGORY_COUNTS:
        raise PlanContractError("image-2299 row is not 46 = 38 person + 8 tie")
    refs = row.get("images")
    if not isinstance(refs, list) or len(refs) != 1:
        raise PlanContractError("image-2299 row must name exactly one image")
    try:
        image_path = (sources.panel.parent / str(refs[0])).resolve(strict=True)
    except OSError as exc:
        raise PlanContractError("image-2299 media reference does not resolve") from exc
    if _sha256_file(image_path) != IMAGE_SHA256:
        raise PlanContractError("image-2299 bytes do not match the frozen digest")
    panel = {
        IMAGE_ID: {
            "image_id": IMAGE_ID,
            "width": int(row["width"]),
            "height": int(row["height"]),
            "file_name": str(row.get("file_name", "")),
            "images": list(refs),
            "objects": list(objects),
        }
    }
    return panel, receipt_digest


def _load_owners(
    path: Path,
    panel: Mapping[str, Mapping[str, Any]],
    *,
    split: str = SPLIT,
) -> list[dict[str, Any]]:
    rows = _read_jsonl(path, "S0 owner ledger")
    if len(rows) != EXPECTED_OWNER_COUNT:
        raise PlanContractError(f"S0 owner ledger has {len(rows)} rows, expected 46")
    owners: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for row in rows:
        required = {
            "schema_version", "gt_owner_id", "image_id", "normalized_description",
            "description", "bbox_xyxy", "decision_eligibility",
        }
        missing = sorted(required - set(row))
        if missing:
            raise PlanContractError(f"S0 owner row lacks fields {missing}")
        owner_id = str(row["gt_owner_id"])
        if owner_id in seen:
            raise PlanContractError(f"duplicate S0 owner ID {owner_id!r}")
        seen.add(owner_id)
        if str(row["image_id"]) != IMAGE_ID:
            raise PlanContractError("S0 owner ledger contains a non-2299 owner")
        description = str(row["normalized_description"])
        counts[description] += 1
        eligibility = row["decision_eligibility"]
        if not isinstance(eligibility, Mapping) or not isinstance(eligibility.get("greedy_natural"), Mapping):
            raise PlanContractError(f"owner {owner_id} lacks greedy_natural eligibility")
        greedy = eligibility["greedy_natural"]
        if not isinstance(greedy.get("eligible"), bool) or not isinstance(greedy.get("status"), str):
            raise PlanContractError(f"owner {owner_id} has malformed greedy eligibility")
        box = [int(round(float(value))) for value in row["bbox_xyxy"]]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= panel[IMAGE_ID]["width"] and 0 <= box[1] < box[3] <= panel[IMAGE_ID]["height"]):
            raise PlanContractError(f"owner {owner_id} has invalid pixel xyxy geometry")
        owners.append(
            {
                "gt_owner_id": owner_id,
                "image_id": IMAGE_ID,
                "normalized_description": description,
                "description": str(row["description"]),
                "official_coco_category_id": row.get("official_coco_category_id"),
                "original_annotation_index": row.get("original_annotation_index"),
                "bbox_pixel_xyxy": box,
                "owner_sort_key": [box[1], box[0]],
                "greedy_eligible": bool(greedy["eligible"]),
                "greedy_eligibility_status": str(greedy["status"]),
                "split": split,
            }
        )
    if dict(sorted(counts.items())) != EXPECTED_CATEGORY_COUNTS:
        raise PlanContractError("S0 owner ledger category composition drifted")
    owners.sort(key=lambda row: str(row["gt_owner_id"]))
    return owners


def _load_prediction_ledger(path: Path, owner_ids: set[str]) -> dict[tuple[str, int], Mapping[str, Any]]:
    rows = _read_jsonl(path, "S0 prediction-row ledger")
    indexed: dict[tuple[str, int], Mapping[str, Any]] = {}
    matched: set[str] = set()
    for row in rows:
        required = {
            "schema_version", "pred_row_id", "image_id", "original_row_index",
            "decode_mode", "normalized_description", "bbox_xyxy", "raw_span_sha256",
            "strict_match_status", "strict_match_gt_owner_id",
        }
        missing = sorted(required - set(row))
        if missing:
            raise PlanContractError(f"S0 prediction row lacks fields {missing}")
        if str(row["image_id"]) != IMAGE_ID or str(row["decode_mode"]) != "greedy":
            raise PlanContractError("S0 prediction ledger is not image-2299 native greedy only")
        index = int(row["original_row_index"])
        key = (IMAGE_ID, index)
        if key in indexed:
            raise PlanContractError(f"duplicate prediction row index {index}")
        status = str(row["strict_match_status"])
        owner_id = row.get("strict_match_gt_owner_id")
        if status not in {"matched", "unmatched", "ambiguous_neutral"}:
            raise PlanContractError(f"unknown strict match status {status!r}")
        if status == "matched":
            owner_id = str(owner_id)
            if owner_id not in owner_ids:
                raise PlanContractError(f"prediction row matches unknown owner {owner_id!r}")
            if owner_id in matched:
                raise PlanContractError(f"owner {owner_id!r} is matched by more than one native row")
            matched.add(owner_id)
        elif owner_id is not None:
            raise PlanContractError("non-matched prediction row carries strict_match_gt_owner_id")
        indexed[key] = row
    if sorted(index for _, index in indexed) != list(range(len(indexed))):
        raise PlanContractError("S0 prediction row indices are not contiguous from zero")
    return indexed


def _load_rollout(path: Path, row_ledger: Mapping[tuple[str, int], Mapping[str, Any]]) -> tuple[dict[str, legacy.NativeRollout], Mapping[str, Any]]:
    payload = _read_json(path, "S0 greedy rollout")
    rows = payload.get("rollouts")
    if not isinstance(rows, list) or len(rows) != 1 or str(rows[0].get("image_id")) != IMAGE_ID:
        raise PlanContractError("S0 greedy artifact must contain exactly image 2299")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise PlanContractError("S0 greedy artifact lacks effective config identity")
    expected_config = {
        "decode_mode": "greedy",
        "model_dtype": "fp32", "repetition_penalty": 1.0,
        "temperature": 0.0, "top_p": 1.0,
    }
    for name, expected in expected_config.items():
        if config.get(name) != expected:
            raise PlanContractError(f"S0 effective config {name}={config.get(name)!r}, expected {expected!r}")
    model_identity = payload.get("model_identity")
    if not isinstance(model_identity, Mapping) or model_identity.get("backend") != "hf":
        raise PlanContractError("S0 runtime is not the frozen HF backend")
    configured_horizon = int(config.get("max_new_tokens", -1))
    if configured_horizon != MAX_NEW_TOKENS:
        if (
            config.get("historical_horizon_admitted_as_nonbinding") is not True
            or int(config.get("prospective_horizon", -1)) != MAX_NEW_TOKENS
        ):
            raise PlanContractError(
                "shorter S0 horizon lacks explicit nonbinding admission to the prospective horizon"
            )
    rollout = rows[0]
    if str(rollout.get("decode_mode")) != "greedy" or str(rollout.get("stop_reason")) != "im_end":
        raise PlanContractError("S0 rollout is not a naturally stopped greedy rollout")
    executed_media_sha256 = rollout.get("executed_media_sha256")
    if not isinstance(executed_media_sha256, str) or len(executed_media_sha256) != 64:
        raise PlanContractError("S0 rollout lacks its executed-media digest")
    generated = [int(value) for value in rollout.get("generated_token_ids", ())]
    prompt = [int(value) for value in rollout.get("prompt_token_ids", ())]
    if legacy.sha256_json(generated) != rollout.get("generated_token_ids_sha256"):
        raise PlanContractError("S0 generated token digest does not reconstruct")
    if legacy.sha256_json(prompt) != rollout.get("prompt_token_ids_sha256"):
        raise PlanContractError("S0 prompt token digest does not reconstruct")
    if len(generated) >= configured_horizon:
        raise PlanContractError("S0 rollout reached the configured horizon; it is truncated")
    if configured_horizon != MAX_NEW_TOKENS and configured_horizon - len(generated) < 32:
        raise PlanContractError("shorter S0 horizon was too close to natural stop to be nonbinding")
    spans = legacy.split_generated_rows(generated)
    envelope = rollout.get("predictions")
    if not isinstance(envelope, Mapping) or envelope.get("parse_status") != "accepted" or int(envelope.get("dropped_prediction_count", 0)) != 0:
        raise PlanContractError("S0 rollout parser was not accepted without dropped rows")
    predictions = envelope.get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(spans) or len(spans) != len(row_ledger):
        raise PlanContractError("S0 rollout, parser, and prediction ledger row counts disagree")
    for index, (prediction, span) in enumerate(zip(predictions, spans, strict=True)):
        ledger = row_ledger[(IMAGE_ID, index)]
        if str(prediction.get("description")) != str(ledger["normalized_description"]):
            raise PlanContractError(f"native row {index} category disagrees with the prediction ledger")
        if str(prediction.get("raw_span_sha256")) != str(ledger["raw_span_sha256"]):
            raise PlanContractError(f"native row {index} raw span digest disagrees with the prediction ledger")
        if legacy.sha256_json(list(span)) == "":  # pragma: no cover - documents token binding
            raise AssertionError
    native = legacy.NativeRollout(
        image_id=IMAGE_ID,
        prompt_token_ids=prompt,
        prompt_token_ids_sha256=str(rollout["prompt_token_ids_sha256"]),
        generated_token_ids=generated,
        generated_token_ids_sha256=str(rollout["generated_token_ids_sha256"]),
        executed_media_sha256=str(rollout["executed_media_sha256"]),
        row_token_spans=spans,
        predictions=list(predictions),
        stop_reason="im_end",
        seed=int(rollout.get("seed", 0)),
        decode_mode="greedy",
    )
    return {IMAGE_ID: native}, config


def _verify_s0_receipt(
    path: Path,
    sources: SourcePaths,
    *,
    checkpoint_profile: str = "geo_sorted_step4887",
) -> tuple[dict[str, Any], str]:
    receipt = _read_json(path, "S0 receipt")
    digest_field = next(
        (name for name in ("receipt_content_sha256", "execution_receipt_content_sha256") if name in receipt),
        None,
    )
    if digest_field is None:
        raise PlanContractError("S0 receipt lacks a reconstructible content digest")
    digest = _verify_content_digest(receipt, digest_field, "S0 receipt")
    if str(receipt.get("execution_status", receipt.get("status", ""))) not in {"completed", "accepted", "captured", "admitted"}:
        raise PlanContractError("S0 receipt is not terminal-success evidence")
    flattened = json.dumps(receipt, sort_keys=True)
    for expected in (PANEL_SHA256, AUTHORITY_ROW_SHA256):
        if expected not in flattened:
            raise PlanContractError(f"S0 receipt does not bind required frozen identity {expected}")
    contract = receipt.get("admission_contract")
    if not isinstance(contract, Mapping):
        raise PlanContractError("S0 receipt lacks its admission contract")
    if legacy.sha256_json(contract) != receipt.get("admission_contract_sha256"):
        raise PlanContractError("S0 admission contract does not reconstruct its digest")
    if str(contract.get("target_image_id")) != IMAGE_ID:
        raise PlanContractError("S0 admission contract targets a foreign image")
    runtime = contract.get("runtime_identity")
    if not isinstance(runtime, Mapping) or runtime.get("backend") != "hf" or runtime.get("attention") != "sdpa":
        raise PlanContractError("S0 admission contract runtime is not exact HF fp32 SDPA")
    runtime_text = json.dumps(runtime, sort_keys=True)
    required_fragment = {
        "geo_sorted_step4887": "geo_sorted",
        "random_step4887": "desc_first_random_pure_ce_typegate",
    }.get(checkpoint_profile)
    if required_fragment is None:
        raise PlanContractError(f"unknown checkpoint profile {checkpoint_profile!r}")
    if (
        "step-4887" not in runtime_text
        or required_fragment not in runtime_text
        or "torch.float32" not in runtime_text
    ):
        raise PlanContractError(
            f"S0 admission contract does not bind the {checkpoint_profile} fp32 runtime"
        )
    generated = contract.get("generated_identity")
    if not isinstance(generated, Mapping) or generated.get("horizon_nonbinding") is not True:
        raise PlanContractError("S0 receipt does not prove a nonbinding natural horizon")
    if int(generated.get("prospective_horizon", -1)) != MAX_NEW_TOKENS:
        raise PlanContractError("S0 receipt prospective horizon drifted")
    outputs = receipt.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PlanContractError("S0 receipt lacks output seals")
    for name, source in (
        ("owner-ledger.jsonl", sources.owner_ledger),
        ("prediction-row-ledger.jsonl", sources.prediction_ledger),
        ("greedy.json", sources.greedy),
    ):
        block = outputs.get(name)
        if not isinstance(block, Mapping) or block.get("sha256") != _sha256_file(source):
            raise PlanContractError(f"S0 receipt does not seal {name}")
    return receipt, digest


@contextmanager
def _one_image_legacy_scope(context_count: int, *, split: str = SPLIT) -> Iterator[None]:
    """Temporarily parameterize pure legacy builders without changing defaults."""

    old_split = legacy.SPLIT_BY_IMAGE_ID
    old_count = legacy.EXPECTED_CONTEXT_COUNT
    try:
        legacy.SPLIT_BY_IMAGE_ID = {**old_split, IMAGE_ID: split}
        legacy.EXPECTED_CONTEXT_COUNT = context_count
        yield
    finally:
        legacy.SPLIT_BY_IMAGE_ID = old_split
        legacy.EXPECTED_CONTEXT_COUNT = old_count


def build_plan(
    sources: SourcePaths,
    *,
    tokenizer_path: str | None = None,
    checkpoint_profile: str = "geo_sorted_step4887",
    extension_unit_id: str = EXTENSION_UNIT_ID,
) -> legacy.CensusPlan:
    panel, panel_receipt_digest = _load_panel(sources)
    profile = {
        "geo_sorted_step4887": {
            "infer_config_sha256": INFER_CONFIG_SHA256,
            "split": SPLIT,
            "target_order": "geo_sorted_(y1,x1)",
        },
        "random_step4887": {
            "infer_config_sha256": RANDOM_INFER_CONFIG_SHA256,
            "split": RANDOM_SPLIT,
            "target_order": "random_training_adapter_native_decode",
        },
    }.get(checkpoint_profile)
    if profile is None:
        raise PlanContractError(f"unknown checkpoint profile {checkpoint_profile!r}")
    if _sha256_file(sources.infer_config) != profile["infer_config_sha256"]:
        raise PlanContractError(f"frozen {checkpoint_profile} infer config digest drifted")
    _, s0_receipt_digest = _verify_s0_receipt(
        sources.s0_receipt, sources, checkpoint_profile=checkpoint_profile
    )
    owners = _load_owners(sources.owner_ledger, panel, split=str(profile["split"]))
    row_ledger = _load_prediction_ledger(sources.prediction_ledger, {str(row["gt_owner_id"]) for row in owners})
    rollouts, effective_config = _load_rollout(sources.greedy, row_ledger)

    owners_by_image = {IMAGE_ID: owners}
    matched_owner_ids: dict[str, list[str]] = {}
    for ledger in row_ledger.values():
        owner_id = ledger.get("strict_match_gt_owner_id")
        if ledger.get("strict_match_status") == "matched" and owner_id:
            matched_owner_ids.setdefault(str(owner_id), []).append(str(ledger["pred_row_id"]))

    observed_tokens = legacy.build_observed_category_tokens(rollouts)
    fallback = legacy.build_tokenizer_resolver(tokenizer_path, observed_tokens) if tokenizer_path else None
    context_count = len(row_ledger) + 1
    with _one_image_legacy_scope(context_count, split=str(profile["split"])):
        images = legacy.build_image_registry(panel, rollouts)
        categories = legacy.build_category_registry(owners_by_image, observed_tokens, fallback)
        contexts = legacy.build_context_registry(rollouts, row_ledger, owners_by_image, panel)
        candidates, bank_accounting = legacy.build_candidate_bank(owners_by_image, panel)
        bank_by_category: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for candidate in candidates:
            bank_by_category.setdefault((IMAGE_ID, str(candidate["normalized_description"])), []).append(candidate)
        owner_rows: list[dict[str, Any]] = []
        for owner in owners:
            native_rows = sorted(matched_owner_ids.get(str(owner["gt_owner_id"]), []))
            bank = bank_accounting[str(owner["gt_owner_id"])]
            owner_rows.append(
                {
                    "schema_version": legacy.PLAN_SCHEMA_VERSION,
                    "row_kind": "census_owner",
                    **owner,
                    "native_strict_match_pred_row_ids": native_rows,
                    "native_true_positive": bool(native_rows),
                    "calibration_role": "prospective_native_true_positive_transfer_control" if native_rows else "prospective_native_false_negative",
                    "excluded_from_census": False,
                    "candidate_bank": bank,
                    "disposition_eligibility": legacy.owner_disposition_eligibility(
                        greedy_eligible=bool(owner["greedy_eligible"]),
                        greedy_eligibility_status=str(owner["greedy_eligibility_status"]),
                        bank_coverage_status=str(bank["bank_coverage_status"]),
                        native_true_positive=bool(native_rows),
                    ),
                }
            )
        bank_token_index: dict[tuple[str, str], dict[tuple[int, ...], str]] = {}
        for candidate in candidates:
            bank_token_index.setdefault((IMAGE_ID, str(candidate["normalized_description"])), {})[
                tuple(candidate["coord_token_ids"])
            ] = str(candidate["candidate_id"])
        query_groups = legacy.build_query_group_registry(contexts, categories, rollouts, bank_by_category)
        sidecars = legacy.build_native_sidecar_registry(rollouts, row_ledger, bank_token_index)
        shards = legacy.build_shard_manifest(query_groups, contexts, categories, owners_by_image)
        rules = legacy.build_capture_rules()

    source_digests = {
        "panel": _sha256_file(sources.panel),
        "panel_receipt": _sha256_file(sources.panel_receipt),
        "owner_ledger": _sha256_file(sources.owner_ledger),
        "prediction_ledger": _sha256_file(sources.prediction_ledger),
        "greedy_rollout": _sha256_file(sources.greedy),
        "s0_receipt": _sha256_file(sources.s0_receipt),
        "infer_config": _sha256_file(sources.infer_config),
    }
    native_tp = sum(bool(row["native_true_positive"]) for row in owner_rows)
    receipt: dict[str, Any] = {
        # These two values are intentionally the legacy scorer ABI.
        "schema_version": legacy.PLAN_SCHEMA_VERSION,
        "unit_id": legacy.UNIT_ID,
        "extension_unit_id": extension_unit_id,
        "checkpoint_profile": checkpoint_profile,
        "plan_strategy": "image2299_only_frozen_candidate_and_context_accessibility_contrast",
        "scorer_compatibility": {
            "consumer": "score_sorted_owner_accessibility_census_shard.py",
            "legacy_plan_schema_and_unit_identity_retained": True,
            "prospective_scope_declared_by_extension_unit_id": True,
        },
        "source_paths": {
            "panel": str(sources.panel), "panel_receipt": str(sources.panel_receipt),
            "owner_ledger": str(sources.owner_ledger), "prediction_ledger": str(sources.prediction_ledger),
            "greedy_rollout": str(sources.greedy), "s0_receipt": str(sources.s0_receipt),
            "infer_config": str(sources.infer_config),
        },
        "source_digests": source_digests,
        "source_content_digests": {
            "panel_receipt_content_sha256": panel_receipt_digest,
            "s0_receipt_content_sha256": s0_receipt_digest,
            "image2299_authority_row_sha256": AUTHORITY_ROW_SHA256,
            "image2299_bytes_sha256": IMAGE_SHA256,
        },
        "runtime_gate": {
            "effective_config": dict(effective_config),
            "decode_mode": "greedy", "backend": "hf", "model_dtype": "fp32",
            "repetition_penalty": 1.0,
            "configured_native_horizon": int(effective_config["max_new_tokens"]),
            "prospective_nonbinding_horizon": MAX_NEW_TOKENS,
            "native_stop_reason": "im_end", "natural_nontruncated": True,
            "target_order": profile["target_order"],
        },
        "census_shape": {
            "image_count": 1, "image_ids": [IMAGE_ID], "owner_count": EXPECTED_OWNER_COUNT,
            "owner_category_counts": EXPECTED_CATEGORY_COUNTS,
            "native_true_positive_owner_count": native_tp,
            "native_false_negative_owner_count": EXPECTED_OWNER_COUNT - native_tp,
            "context_count": len(contexts), "native_row_count": len(sidecars),
            "category_count": len(categories), "query_group_count": len(query_groups),
            "physical_candidate_count": len(candidates),
        },
        "denominator_contract": {
            "prospective_image2299_owner_count": EXPECTED_OWNER_COUNT,
            "legacy_12_owner_count_unchanged": LEGACY_OWNER_COUNT,
            "legacy_12_eligible_native_fn_denominator_unchanged": LEGACY_ELIGIBLE_FN_DENOMINATOR,
            "pooled_13_image_denominator_created": False,
            "report_slices_separately": True,
        },
        "candidate_bank_contract": {
            "logical_roles": list(legacy.LOGICAL_ROLES), "logical_role_count": 17,
            "score_independent": True, "mid_run_growth": "forbidden",
            "proposal_and_localization_separate": True,
        },
        "query_suffix_contract": {
            "shape": ["object_ref_start", "category_token_ids", "object_ref_end", "box_start"],
            "wrapper_token_ids": dict(legacy.WRAPPER_TOKEN_IDS),
            "x1_distribution_read_point": "immediately_after_box_start", "on_mismatch": "fail_closed",
        },
        "estimand": {
            "name": "category_field_support_at_owner_geometry",
            "is_per_owner_proposal_probability": False,
            "proposal_and_localization_separate": True,
        },
        "score_input_policy": {
            "reads_any_score_artifact": False, "candidate_selection_uses_scores": False,
            "frozen_thresholds_loaded_by_downstream_analyzer_only": True,
        },
        "capture_rules_sha256": rules["capture_rules_sha256"],
        "output_file_digests": {},
    }
    plan = legacy.CensusPlan(
        images=images, owners=owner_rows, categories=categories, contexts=contexts,
        candidates=candidates, query_groups=query_groups, native_sidecars=sidecars,
        shards=shards, capture_rules=rules, receipt=receipt,
    )
    receipt["output_file_digests"] = {
        name: hashlib.sha256(content).hexdigest() for name, content in sorted(plan.files().items())
    }
    receipt["receipt_content_sha256"] = legacy.sha256_json(receipt)
    return plan


def commit_plan(plan: legacy.CensusPlan, output_dir: Path) -> dict[str, str]:
    files = dict(plan.files())
    files["receipt.json"] = legacy.canonical_json_bytes(plan.receipt) + b"\n"
    output_dir = Path(output_dir)

    def validate_existing() -> None:
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise PlanContractError(
                f"plan output exists but is not a regular directory: {output_dir}"
            )
        entries = {path.name: path for path in output_dir.iterdir()}
        expected = set(files)
        observed = set(entries)
        non_files = sorted(
            name
            for name, path in entries.items()
            if path.is_symlink() or not path.is_file()
        )
        if observed != expected or non_files:
            raise PlanContractError(
                "existing plan artifact set is not exact: "
                f"missing={sorted(expected - observed)} "
                f"foreign={sorted(observed - expected)} non_files={non_files}"
            )
        mismatched = sorted(
            name for name, content in files.items() if entries[name].read_bytes() != content
        )
        if mismatched:
            raise PlanContractError(
                f"existing plan artifacts are not byte-identical: {mismatched}"
            )

    result = {
        name: hashlib.sha256(content).hexdigest()
        for name, content in sorted(files.items())
    }
    if output_dir.exists() or output_dir.is_symlink():
        validate_existing()
        return result

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    published = False
    try:
        for name, content in sorted(files.items()):
            (staging / name).write_bytes(content)
        try:
            os.replace(staging, output_dir)
            published = True
        except OSError as exc:
            # A concurrent byte-identical publisher is admissible; every
            # other race or filesystem failure stays fail-closed.
            if not output_dir.exists() and not output_dir.is_symlink():
                raise PlanContractError(
                    f"atomic plan publication failed for {output_dir}"
                ) from exc
            validate_existing()
        return result
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--s0-root", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=PANEL)
    parser.add_argument("--panel-receipt", type=Path, default=PANEL_RECEIPT)
    parser.add_argument("--infer-config", type=Path, default=INFER_CONFIG)
    parser.add_argument("--owner-ledger", type=Path)
    parser.add_argument("--prediction-ledger", type=Path)
    parser.add_argument("--greedy", type=Path)
    parser.add_argument("--s0-receipt", type=Path)
    parser.add_argument("--tokenizer-path")
    parser.add_argument(
        "--checkpoint-profile",
        choices=("geo_sorted_step4887", "random_step4887"),
        default="geo_sorted_step4887",
    )
    parser.add_argument("--extension-unit-id", default=EXTENSION_UNIT_ID)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    root = args.s0_root
    sources = SourcePaths(
        panel=args.panel, panel_receipt=args.panel_receipt,
        owner_ledger=args.owner_ledger or root / "owner-ledger.jsonl",
        prediction_ledger=args.prediction_ledger or root / "prediction-row-ledger.jsonl",
        greedy=args.greedy or root / "greedy.json",
        s0_receipt=args.s0_receipt or root / "receipt.json",
        infer_config=args.infer_config,
    )
    try:
        plan = build_plan(
            sources,
            tokenizer_path=args.tokenizer_path,
            checkpoint_profile=args.checkpoint_profile,
            extension_unit_id=args.extension_unit_id,
        )
        commit_plan(plan, args.output_dir)
    except PlanContractError as exc:
        print(f"plan contract error: {exc}", file=sys.stderr)
        return 2
    shape = plan.receipt["census_shape"]
    print(
        f"wrote scorer-compatible image-2299 plan: owners={shape['owner_count']} "
        f"native_tp={shape['native_true_positive_owner_count']} "
        f"native_fn={shape['native_false_negative_owner_count']} contexts={shape['context_count']}"
    )
    print(f"receipt_content_sha256 {plan.receipt['receipt_content_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
