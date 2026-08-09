#!/usr/bin/env python3
"""Seal the CPU-only S native-FN support-completion plan.

The natural-boundary unit has a deliberately narrow missing denominator: the
220 S native false negatives in the 392-owner H0 ledger.  Twenty of those FN
owners were already measured by the old support probe and are retained as
lineage.  This module plans only the remaining 200 contexts.  The 172 native
true positives are calibration controls and are reused verbatim; they are not
re-scored and the other 160 TP owners are never converted to support
negatives.

Planning is CPU-only.  It validates the frozen panel/H0/support identities,
reuses the old physical candidate-bank construction (without reading or
changing ``FROZEN_CANDIDATES``), binds every owner/prefix/target-row/candidate
identity, and emits content-stable eight-shard accounting.  A batched count is
reported as an estimate only: the exact-history scorer remains scalar and no
GPU work is admitted by this module.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_static_dynamic_owner_support_probe as support_probe  # noqa: E402


UNIT_ID = "2026-08-06-natural-boundary-routing-history-replication"
SCHEMA_VERSION = "natural_boundary_owner_support_completion_plan.v1"
RECEIPT_SCHEMA_VERSION = f"{SCHEMA_VERSION}.receipt.v1"
CHECKPOINT = "S"
CHECKPOINT_WRAPPER = "object_box_closed"
CHECKPOINT_PARSER = "compact_object_box_closed_only"

# These are contract values, not tunable thresholds.  They are checked against
# live lineage before a production-shaped plan is sealed.
EXPECTED_PANEL_IMAGES = 13
EXPECTED_PANEL_OWNERS = 392
EXPECTED_NATIVE_TP = 172
EXPECTED_NATIVE_FN = 220
EXPECTED_MEASURED_FN = 20
EXPECTED_COMPLETION_CONTEXTS = 200
EXPECTED_UNSCORED_TP = 160
EXPECTED_SHARDS = 8
EXPECTED_SHARD_CONTEXTS = (21, 22, 25, 18, 25, 24, 33, 32)
EXPECTED_SHARD_FORWARDS = (6757, 7012, 11819, 8208, 9845, 8511, 13867, 11409)
EXPECTED_SCALAR_FORWARDS = 77428
DEFAULT_CANDIDATE_BATCH_SIZE = 16

DEFAULT_SOURCE_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-04-sorted-prospective-13-image-panel-admission/"
    "evaluation-inputs/human-refined-13.coord.jsonl"
)
DEFAULT_DERIVED_PANEL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
DEFAULT_DERIVED_RECEIPT = DEFAULT_DERIVED_PANEL.with_suffix(".receipt.json")
DEFAULT_H0_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/"
    "s-step2444-native-h0.json"
)
DEFAULT_SUPPORT_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/ledgers/"
    "s-step2444-final-support.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-06-natural-boundary-routing-history-replication/"
    "support-completion-plan-v1"
)


class SupportCompletionPlanError(ValueError):
    """Raised when a frozen support-completion identity or scope is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SupportCompletionPlanError("value is not canonical JSON serializable") from exc


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path) -> str:
    resolved = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_source(source: str | Path | Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser().resolve(strict=True)
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise SupportCompletionPlanError(f"invalid JSON input: {path}") from exc
        return value, {"path": str(path), "sha256": sha256_bytes(raw), "byte_count": len(raw)}
    if isinstance(source, Mapping):
        value = dict(source)
        return value, {"inline": True, "sha256": sha256_json(value), "byte_count": None}
    raise SupportCompletionPlanError("JSON input must be a path or mapping")


def _strict_hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SupportCompletionPlanError(f"{label} must be a 64-character SHA-256")
    lowered = value.lower()
    if any(ch not in "0123456789abcdef" for ch in lowered):
        raise SupportCompletionPlanError(f"{label} must be a lowercase SHA-256")
    return lowered


def _strict_int(value: Any, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SupportCompletionPlanError(f"{label} must be an integer >= {minimum}")
    return int(value)


def _owner_sort_key(row: Mapping[str, Any]) -> tuple[int, str, str]:
    return (
        _strict_int(row.get("image_id"), "owner image_id"),
        str(row.get("gt_owner_id")),
        str(row.get("exact_prefix_sha256") or "<no-prefix>"),
    )


def _stable_key(row: Mapping[str, Any]) -> str:
    """Use the old capture key so shard assignment is content-stable."""

    prefix = row.get("exact_prefix_sha256")
    if not isinstance(prefix, str) or not prefix:
        raise SupportCompletionPlanError("support-completion owner lacks exact prefix hash")
    return "|".join(
        (
            "candidate",  # old support probe key; context kind is sealed below
            CHECKPOINT,
            str(_strict_int(row.get("image_id"), "owner image_id")),
            str(row.get("gt_owner_id")),
            prefix,
        )
    )


def context_id_for_stable_key(stable_key: str) -> str:
    if not isinstance(stable_key, str) or not stable_key:
        raise SupportCompletionPlanError("stable key must be a non-empty string")
    return f"ctx:{sha256_json(stable_key)[:24]}"


def shard_index_for_stable_key(stable_key: str, num_shards: int = EXPECTED_SHARDS) -> int:
    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards <= 0:
        raise SupportCompletionPlanError("num_shards must be a positive integer")
    return int(sha256_json(stable_key)[:16], 16) % num_shards


def _source_rows_by_image(panel: support_probe.PanelInputs) -> dict[int, Mapping[str, Any]]:
    values = panel.source if isinstance(panel.source, list) else [panel.source]
    result: dict[int, Mapping[str, Any]] = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise SupportCompletionPlanError("source panel row is not an object")
        try:
            image = int(row["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupportCompletionPlanError("source panel row has no integer image_id") from exc
        result[image] = row
    return result


def _derived_rows_by_image(panel: support_probe.PanelInputs) -> dict[int, Mapping[str, Any]]:
    values = panel.derived if isinstance(panel.derived, list) else [panel.derived]
    result: dict[int, Mapping[str, Any]] = {}
    for row in values:
        if not isinstance(row, Mapping):
            raise SupportCompletionPlanError("derived panel row is not an object")
        try:
            image = int(row["image_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SupportCompletionPlanError("derived panel row has no integer image_id") from exc
        result[image] = row
    return result


def _owner_row_identity(
    *,
    panel: support_probe.PanelInputs,
    owner: Any,
    h0_row: Mapping[str, Any],
) -> dict[str, Any]:
    source_rows = _source_rows_by_image(panel)
    derived_rows = _derived_rows_by_image(panel)
    image = _strict_int(owner.image_id, "owner image_id")
    source_index = _strict_int(owner.source_index, "owner source_index")
    source_objects = source_rows[image].get("objects")
    derived_objects = derived_rows[image].get("objects")
    if not isinstance(source_objects, list) or not isinstance(derived_objects, list):
        raise SupportCompletionPlanError(f"image {image} lacks source/derived objects")
    derived_index = owner.derived_index
    if derived_index is None:
        raise SupportCompletionPlanError(f"owner {owner.gt_owner_id} has no derived index")
    derived_index = _strict_int(derived_index, "owner derived_index")
    if source_index >= len(source_objects) or derived_index >= len(derived_objects):
        raise SupportCompletionPlanError(f"owner {owner.gt_owner_id} index is outside panel")
    source_object = source_objects[source_index]
    derived_object = derived_objects[derived_index]
    source_object_sha256 = sha256_json(source_object)
    derived_object_sha256 = sha256_json(derived_object)
    identity = {
        "gt_owner_id": str(owner.gt_owner_id),
        "image_id": image,
        "source_panel_object_index": source_index,
        "derived_panel_object_index": derived_index,
        "category_name": str(owner.category or "").lower(),
        "bbox_pixel_xyxy": list(owner.bbox or ()),
        "coco_ann_id": owner.coco_ann_id,
        "source_object_sha256": source_object_sha256,
        "derived_object_sha256": derived_object_sha256,
        "row_identity_rule": "panel-object-canonical-json-sha256-with-source-derived-binding",
    }
    identity["target_owner_row_identity_sha256"] = sha256_json(identity)
    # H0 and panel must identify the same physical owner.  The loader performs
    # geometry checks; these explicit checks make the plan receipt readable.
    for key, expected in (
        ("gt_owner_id", str(owner.gt_owner_id)),
        ("image_id", image),
        ("source_panel_object_index", source_index),
    ):
        if h0_row.get(key) != expected:
            raise SupportCompletionPlanError(
                f"H0 owner identity mismatch for {owner.gt_owner_id}: {key}"
            )
    return identity


def _candidate_identity(candidate: Mapping[str, Any]) -> dict[str, Any]:
    raw_image = candidate.get("image_id")
    try:
        image = int(raw_image)
    except (TypeError, ValueError) as exc:
        raise SupportCompletionPlanError("candidate image_id must be an integer >= 0") from exc
    if image < 0:
        raise SupportCompletionPlanError("candidate image_id must be an integer >= 0")
    category = str(candidate.get("normalized_description", "")).lower()
    if not category:
        raise SupportCompletionPlanError("physical candidate has no category")
    tokens = candidate.get("coord_token_ids")
    if not isinstance(tokens, list) or not tokens or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in tokens
    ):
        raise SupportCompletionPlanError("physical candidate coordinate token IDs are malformed")
    bins = candidate.get("coord_bins")
    if not isinstance(bins, list) or len(bins) != 4:
        raise SupportCompletionPlanError("physical candidate coordinate bins are malformed")
    expected_id = support_probe.legacy.physical_candidate_id(
        image_id=str(image), normalized_description=category, tokens=tokens
    )
    if str(candidate.get("candidate_id")) != expected_id:
        raise SupportCompletionPlanError(
            f"physical candidate identity mismatch for {candidate.get('candidate_id')!r}"
        )
    core = {
        "candidate_id": expected_id,
        "image_id": image,
        "normalized_description": category,
        "coord_bins": [int(item) for item in bins],
        "coord_token_ids": [int(item) for item in tokens],
        "decoded_bbox_pixel_xyxy": list(candidate.get("decoded_bbox_pixel_xyxy") or []),
        "identity_rule": "digest_image_category_coord_tokens",
    }
    core["coord_token_ids_sha256"] = sha256_json(core["coord_token_ids"])
    core["candidate_identity_sha256"] = sha256_json(core)
    # This is the row identity used by the eventual teacher-forced scorer.  A
    # tokenizer is intentionally not loaded here; category tokenization is
    # bound at execution by the immutable category text plus coordinate span.
    target_row = {
        "candidate_id": expected_id,
        "category_name": category,
        "coord_token_ids": core["coord_token_ids"],
        "coord_token_ids_sha256": core["coord_token_ids_sha256"],
        "row_encoding": (
            "object_ref_start+tokenizer(category)+object_ref_end+box_start+"
            "coord_token_ids+box_end"
        ),
        "tokenizer_binding": "checkpoint-native-tokenizer-at-execution",
    }
    target_row["target_row_identity_sha256"] = sha256_json(target_row)
    return {**core, "target_row": target_row}


def _validate_h0_scope(
    *,
    panel: support_probe.PanelInputs,
    h0: support_probe.H0Inputs,
    strict: bool = True,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    if h0.envelope.get("checkpoint") != CHECKPOINT:
        raise SupportCompletionPlanError("support-completion plan requires S H0")
    if h0.envelope.get("run_kind") != "native_h0":
        raise SupportCompletionPlanError("H0 ledger is not native_h0")
    if h0.envelope.get("history_complete") is not True:
        raise SupportCompletionPlanError("H0 ledger lacks complete-history attestation")
    records = list(h0.records)
    by_owner: dict[str, Mapping[str, Any]] = {}
    for row in records:
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in by_owner:
            raise SupportCompletionPlanError(f"duplicate H0 owner {owner_id}")
        if row.get("checkpoint") != CHECKPOINT:
            raise SupportCompletionPlanError(f"foreign checkpoint H0 owner {owner_id}")
        if row.get("unit_id") != "2026-08-05-static-dynamic-owner-interface-crossover":
            raise SupportCompletionPlanError("H0 lineage unit identity drifted")
        if row.get("source_panel_sha256") != panel.source_info["sha256"]:
            raise SupportCompletionPlanError(f"H0 source panel hash mismatch for {owner_id}")
        if row.get("derived_panel_sha256") != panel.derived_info["sha256"]:
            raise SupportCompletionPlanError(f"H0 derived panel hash mismatch for {owner_id}")
        if row.get("history_complete") is not True or row.get("excludes_stop") is not True:
            raise SupportCompletionPlanError(f"H0 history/STOP attestation missing for {owner_id}")
        prefix = row.get("exact_prefix_token_ids")
        prefix_hash = row.get("exact_prefix_sha256")
        if not isinstance(prefix, list) or not isinstance(prefix_hash, str):
            raise SupportCompletionPlanError(f"H0 exact prefix missing for {owner_id}")
        if sha256_json(prefix) != prefix_hash:
            raise SupportCompletionPlanError(f"H0 exact prefix hash mismatch for {owner_id}")
        if row.get("natural_boundary_valid") is not True:
            raise SupportCompletionPlanError(
                f"S native-FN support completion requires valid natural boundary: {owner_id}"
            )
        if row.get("native_fn") is True:
            if row.get("native_tp") is not False or row.get("strict_complete_row") is not False:
                raise SupportCompletionPlanError(f"H0 FN native outcome is malformed for {owner_id}")
            if row.get("prefix_semantics") != "after_strict_covered_row_pre_stop":
                raise SupportCompletionPlanError(f"H0 FN prefix semantics drifted for {owner_id}")
        elif row.get("native_tp") is True:
            if row.get("strict_complete_row") is not True:
                raise SupportCompletionPlanError(f"H0 TP strict row flag is malformed for {owner_id}")
            if row.get("prefix_semantics") != "before_queried_owner_row":
                raise SupportCompletionPlanError(f"H0 TP prefix semantics drifted for {owner_id}")
        else:
            raise SupportCompletionPlanError(f"H0 owner is neither native TP nor FN: {owner_id}")
        by_owner[owner_id] = row
    tps = [row for row in records if row.get("native_tp") is True]
    fns = [row for row in records if row.get("native_fn") is True]
    if strict and len(records) != EXPECTED_PANEL_OWNERS:
        raise SupportCompletionPlanError(
            f"S H0 must contain {EXPECTED_PANEL_OWNERS} owners, got {len(records)}"
        )
    if strict and (len(tps) != EXPECTED_NATIVE_TP or len(fns) != EXPECTED_NATIVE_FN):
        raise SupportCompletionPlanError(
            "S H0 native TP/FN counts do not match 172/220"
        )
    if strict and (len(panel.owners_by_image) != EXPECTED_PANEL_IMAGES or sum(
        len(items) for items in panel.owners_by_image.values()
    ) != EXPECTED_PANEL_OWNERS):
        raise SupportCompletionPlanError("admitted panel is not the frozen 13-image/392-owner panel")
    return sorted(tps, key=_owner_sort_key), sorted(fns, key=_owner_sort_key), by_owner


def _validate_support_lineage(
    *,
    panel: support_probe.PanelInputs,
    h0: support_probe.H0Inputs,
    support_envelope: Mapping[str, Any],
    support_source_info: Mapping[str, Any],
    h0_by_owner: Mapping[str, Mapping[str, Any]],
    strict: bool = True,
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]], dict[str, Any]]:
    if support_envelope.get("checkpoint") != CHECKPOINT:
        raise SupportCompletionPlanError("support lineage is not S")
    if support_envelope.get("run_kind") != "native_h0":
        raise SupportCompletionPlanError("support lineage is not native_h0")
    if support_envelope.get("source_panel_sha256") != panel.source_info["sha256"]:
        raise SupportCompletionPlanError("support lineage source panel hash mismatch")
    if support_envelope.get("derived_panel_sha256") != panel.derived_info["sha256"]:
        raise SupportCompletionPlanError("support lineage derived panel hash mismatch")
    if support_envelope.get("h0_source_sha256") != h0.source_info["sha256"]:
        raise SupportCompletionPlanError("support lineage is detached from S H0")
    calibration = support_envelope.get("calibration")
    support_rule = support_envelope.get("support_rule")
    if not isinstance(calibration, Mapping) or not isinstance(support_rule, Mapping):
        raise SupportCompletionPlanError("support lineage lacks calibration/rule")
    declared_calibration_hash = _strict_hash(
        calibration.get("calibration_sha256"), "calibration_sha256"
    )
    if support_rule.get("calibration_sha256") != declared_calibration_hash:
        raise SupportCompletionPlanError("support rule calibration hash is detached from calibration")
    calibration_without_hash = {
        key: value for key, value in calibration.items() if key != "calibration_sha256"
    }
    if sha256_json(calibration_without_hash) != declared_calibration_hash:
        raise SupportCompletionPlanError("frozen calibration hash does not match calibration body")
    if calibration.get("checkpoint") != CHECKPOINT:
        raise SupportCompletionPlanError("calibration is not checkpoint-local S")
    if calibration.get("population") != "checkpoint_native_true_positive_controls":
        raise SupportCompletionPlanError("calibration population is not native TP controls")
    if strict and calibration.get("observation_count") != EXPECTED_NATIVE_TP:
        raise SupportCompletionPlanError("S calibration observation count is not 172")
    if calibration.get("excluded_context_count") != 0:
        raise SupportCompletionPlanError("S calibration excludes contexts")
    if calibration.get("quantile") != 0.1 or calibration.get("epsilon") != 0.002:
        raise SupportCompletionPlanError("S calibration quantile/epsilon drifted")
    # The old receipt uses ``teacher_forced_diagnostic_only`` in the support
    # rule and ``teacher_forced_is_diagnostic_only`` in the calibration body;
    # keep that explicit aliasing instead of normalizing away lineage fields.
    frozen_rule = {
        "criterion_id": (support_rule, "criterion_id", "local_peak_lift_and_local_concentration_under_both_ambiguity_bounds"),
        "calibration_population": (support_rule, "calibration_population", "checkpoint_native_true_positive_controls"),
        "rule": (calibration, "rule", "peak_lift >= theta_peak_lift + epsilon AND local_concentration >= theta_local_concentration + epsilon"),
        "epsilon": (calibration, "epsilon", 0.002),
        "quantile": (calibration, "quantile", 0.1),
        "teacher_forced_diagnostic_only": (support_rule, "teacher_forced_diagnostic_only", True),
        "teacher_forced_is_diagnostic_only": (calibration, "teacher_forced_is_diagnostic_only", True),
        "behavioral_transfer_claim": (support_rule, "behavioral_transfer_claim", False),
    }
    for key, (container, observed_key, expected) in frozen_rule.items():
        observed = container.get(observed_key)
        if observed != expected:
            raise SupportCompletionPlanError(f"frozen support rule field {key} drifted")
    records = support_envelope.get("records")
    if not isinstance(records, list):
        raise SupportCompletionPlanError("support lineage records are missing")
    by_owner: dict[str, Mapping[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            raise SupportCompletionPlanError("support lineage record is not an object")
        owner_id = str(row.get("gt_owner_id"))
        if owner_id in by_owner:
            raise SupportCompletionPlanError(f"duplicate support lineage owner {owner_id}")
        h0_row = h0_by_owner.get(owner_id)
        if h0_row is None:
            raise SupportCompletionPlanError(f"support lineage owner absent from H0: {owner_id}")
        if row.get("checkpoint") != CHECKPOINT or row.get("run_kind") != "native_h0":
            raise SupportCompletionPlanError(f"support lineage checkpoint/run kind drifted for {owner_id}")
        if row.get("support_status") != "measured" or row.get("verified_support_claim") is not True:
            raise SupportCompletionPlanError(f"support lineage measurement status drifted for {owner_id}")
        if row.get("source_panel_sha256") != panel.source_info["sha256"] or row.get("derived_panel_sha256") != panel.derived_info["sha256"]:
            raise SupportCompletionPlanError(f"support lineage panel hash mismatch for {owner_id}")
        for key in ("exact_prefix_sha256", "natural_boundary", "due_boundary_index", "covered_owner_ids"):
            if row.get(key) != h0_row.get(key):
                raise SupportCompletionPlanError(f"support lineage {key} mismatch for {owner_id}")
        if row.get("exact_prefix_token_ids") != h0_row.get("exact_prefix_token_ids"):
            raise SupportCompletionPlanError(f"support lineage exact prefix IDs mismatch for {owner_id}")
        if row.get("support_calibration_sha256") != declared_calibration_hash:
            raise SupportCompletionPlanError(f"support lineage calibration hash mismatch for {owner_id}")
        by_owner[owner_id] = row
    measured_fns = sorted(
        [row for row in records if row.get("native_fn") is True], key=_owner_sort_key
    )
    measured_tps = sorted(
        [row for row in records if row.get("native_tp") is True], key=_owner_sort_key
    )
    if strict and len(measured_fns) != EXPECTED_MEASURED_FN:
        raise SupportCompletionPlanError(
            f"support lineage must retain exactly {EXPECTED_MEASURED_FN} measured FN owners"
        )
    if strict and len(measured_tps) != 12:
        raise SupportCompletionPlanError("support lineage TP control count is not 12")
    return measured_fns, measured_tps, {
        "file_sha256": support_source_info["sha256"],
        "envelope_content_sha256": sha256_json(support_envelope),
        "calibration": dict(calibration),
        "support_rule": dict(support_rule),
        "calibration_sha256": declared_calibration_hash,
        "records_sha256": sha256_json(records),
        "record_count": len(records),
    }


def _context_target_rows(
    *,
    physical: Sequence[Mapping[str, Any]],
    image_id: int,
    category: str,
) -> list[dict[str, Any]]:
    selected = [
        candidate
        for candidate in physical
        if int(candidate.get("image_id")) == image_id
        and str(candidate.get("normalized_description", "")).lower() == category.lower()
    ]
    selected.sort(key=lambda row: str(row.get("candidate_id")))
    result = [_candidate_identity(row) for row in selected]
    ids = [row["candidate_id"] for row in result]
    if not ids or len(ids) != len(set(ids)):
        raise SupportCompletionPlanError(
            f"support-completion owner has no unique candidate bank: {image_id}/{category}"
        )
    return result


def _make_context(
    *,
    row: Mapping[str, Any],
    owner: Any,
    owner_identity: Mapping[str, Any],
    target_rows: Sequence[Mapping[str, Any]],
    batch_size: int,
    shard_count: int,
) -> dict[str, Any]:
    stable_key = _stable_key(row)
    context_id = context_id_for_stable_key(stable_key)
    shard_index = shard_index_for_stable_key(stable_key, shard_count)
    candidate_ids = [str(candidate["candidate_id"]) for candidate in target_rows]
    candidate_identity_hashes = [str(candidate["candidate_identity_sha256"]) for candidate in target_rows]
    target_row_hashes = [str(candidate["target_row"]["target_row_identity_sha256"]) for candidate in target_rows]
    target_rows_digest = sha256_json(list(target_rows))
    prefix_ids = list(row["exact_prefix_token_ids"])
    covered = list(row.get("covered_owner_ids") or [])
    due_evidence = dict(row.get("due_boundary_evidence") or {})
    target_row = {
        **dict(owner_identity),
        "target_owner_row_identity_sha256": owner_identity["target_owner_row_identity_sha256"],
    }
    context = {
        "context_id": context_id,
        "stable_key": stable_key,
        "stable_key_rule": "candidate|checkpoint|image_id|gt_owner_id|exact_prefix_sha256",
        "shard_index": shard_index,
        "context_kind": "support_completion_candidate",
        "checkpoint": CHECKPOINT,
        "image_id": int(row["image_id"]),
        "gt_owner_id": str(row["gt_owner_id"]),
        "source_panel_object_index": int(row["source_panel_object_index"]),
        "derived_panel_object_index": int(owner.derived_index),
        "coco_ann_id": row.get("coco_ann_id"),
        "category_name": str(row.get("category_name", "")).lower(),
        "bbox_pixel_xyxy": list(row.get("bbox_pixel_xyxy") or []),
        "native_tp": False,
        "native_fn": True,
        "strict_complete_row": False,
        "natural_boundary": row.get("natural_boundary"),
        "natural_boundary_valid": True,
        "prefix_semantics": row.get("prefix_semantics"),
        "covered_owner_ids": covered,
        "covered_owner_ids_sha256": sha256_json(covered),
        "latest_covered_owner_id": row.get("latest_covered_owner_id"),
        "due_boundary_evidence": due_evidence,
        "due_boundary_evidence_sha256": sha256_json(due_evidence),
        "exact_prefix_token_ids": prefix_ids,
        "exact_prefix_token_count": len(prefix_ids),
        "exact_prefix_sha256": row["exact_prefix_sha256"],
        "exact_prefix_identity_sha256": sha256_json(prefix_ids),
        "target_owner_row": target_row,
        "target_owner_row_identity_sha256": target_row["target_owner_row_identity_sha256"],
        "candidate_ids": candidate_ids,
        "candidate_ids_sha256": sha256_json(candidate_ids),
        "candidate_identity_hashes": candidate_identity_hashes,
        "candidate_identity_hashes_sha256": sha256_json(candidate_identity_hashes),
        "target_rows": [dict(candidate) for candidate in target_rows],
        "target_rows_sha256": target_rows_digest,
        "target_row_identity_hashes": target_row_hashes,
        "target_row_identity_hashes_sha256": sha256_json(target_row_hashes),
        "candidate_count": len(candidate_ids),
        "scalar_equivalent_forward_count": len(candidate_ids),
        "safe_batched_forward_estimate": math.ceil(len(candidate_ids) / batch_size),
        "batching_admitted": False,
        "status": "planned",
        "measured_wall_time_seconds": None,
        "realized_scalar_forward_count": None,
        "no_future_or_intervention_leakage": True,
    }
    context["context_identity_sha256"] = sha256_json(
        {
            "stable_key": stable_key,
            "owner_identity": target_row,
            "exact_prefix_sha256": row["exact_prefix_sha256"],
            "target_rows_sha256": target_rows_digest,
        }
    )
    return context


def _work_accounting(
    contexts: Sequence[Mapping[str, Any]],
    *,
    batch_size: int,
    shard_count: int,
) -> dict[str, Any]:
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise SupportCompletionPlanError("candidate_batch_size must be a positive integer")
    per_shard: list[dict[str, Any]] = []
    for shard in range(shard_count):
        assigned = [row for row in contexts if int(row["shard_index"]) == shard]
        scalar = sum(int(row["scalar_equivalent_forward_count"]) for row in assigned)
        per_shard.append(
            {
                "shard_index": shard,
                "context_count": len(assigned),
                "scalar_equivalent_forward_count": scalar,
                "safe_batched_forward_estimate": math.ceil(scalar / batch_size) if scalar else 0,
                "candidate_batch_size": batch_size,
                "batching_admitted": False,
                "measured_wall_time_seconds": None,
                "realized_scalar_forward_count": None,
            }
        )
    scalar_total = sum(row["scalar_equivalent_forward_count"] for row in per_shard)
    return {
        "shard_count": shard_count,
        "candidate_batch_size": batch_size,
        "batching_admitted": False,
        "batching_status": "estimate_only_exact_history_api_scalar_only",
        "scalar_equivalent_forward_count": scalar_total,
        "safe_batched_forward_estimate": math.ceil(scalar_total / batch_size) if scalar_total else 0,
        "per_shard": per_shard,
        "measured_wall_time_seconds": None,
        "realized_scalar_forward_count": None,
    }


def _enforce_exact_contract(
    *,
    panel: support_probe.PanelInputs,
    tps: Sequence[Mapping[str, Any]],
    fns: Sequence[Mapping[str, Any]],
    measured_fns: Sequence[Mapping[str, Any]],
    contexts: Sequence[Mapping[str, Any]],
    work: Mapping[str, Any],
    strict: bool,
) -> None:
    if not strict:
        return
    if len(panel.owners_by_image) != EXPECTED_PANEL_IMAGES:
        raise SupportCompletionPlanError("strict plan requires 13 panel images")
    if len(tps) != EXPECTED_NATIVE_TP or len(fns) != EXPECTED_NATIVE_FN:
        raise SupportCompletionPlanError("strict plan requires 172 TP and 220 FN owners")
    if len(measured_fns) != EXPECTED_MEASURED_FN:
        raise SupportCompletionPlanError("strict plan requires 20 retained measured FNs")
    if len(contexts) != EXPECTED_COMPLETION_CONTEXTS:
        raise SupportCompletionPlanError("strict plan requires exactly 200 completion contexts")
    if len({str(row["gt_owner_id"]) for row in contexts}) != EXPECTED_COMPLETION_CONTEXTS:
        raise SupportCompletionPlanError("completion contexts do not cover 200 unique owners")
    if int(work["shard_count"]) != EXPECTED_SHARDS:
        raise SupportCompletionPlanError("strict plan requires eight shards")
    observed_contexts = tuple(int(row["context_count"]) for row in work["per_shard"])
    observed_forwards = tuple(int(row["scalar_equivalent_forward_count"]) for row in work["per_shard"])
    if observed_contexts != EXPECTED_SHARD_CONTEXTS:
        raise SupportCompletionPlanError(
            f"completion shard context counts drifted: {observed_contexts!r}"
        )
    if observed_forwards != EXPECTED_SHARD_FORWARDS:
        raise SupportCompletionPlanError(
            f"completion shard scalar forwards drifted: {observed_forwards!r}"
        )
    if int(work["scalar_equivalent_forward_count"]) != EXPECTED_SCALAR_FORWARDS:
        raise SupportCompletionPlanError("completion scalar forward total drifted")


def build_support_completion_plan(
    *,
    source_panel: str | Path | Mapping[str, Any],
    derived_panel: str | Path | Mapping[str, Any],
    derived_receipt: str | Path | Mapping[str, Any],
    h0_ledger: str | Path | Mapping[str, Any],
    support_ledger: str | Path | Mapping[str, Any],
    candidate_batch_size: int = DEFAULT_CANDIDATE_BATCH_SIZE,
    num_shards: int = EXPECTED_SHARDS,
    strict_contract: bool = True,
) -> dict[str, Any]:
    """Build a deterministic, CPU-only support completion plan."""

    if isinstance(num_shards, bool) or not isinstance(num_shards, int) or num_shards != EXPECTED_SHARDS:
        raise SupportCompletionPlanError("natural-boundary support completion requires exactly eight shards")
    panel = support_probe.load_panel_inputs(source_panel, derived_panel, derived_receipt)
    h0 = support_probe.load_h0_inputs(h0_ledger, panel=panel, checkpoint=CHECKPOINT)
    h0_source_info = h0.source_info
    h0_tps, h0_fns, h0_by_owner = _validate_h0_scope(
        panel=panel, h0=h0, strict=strict_contract
    )
    support_envelope, support_source_info = _read_json_source(support_ledger)
    if not isinstance(support_envelope, Mapping):
        raise SupportCompletionPlanError("support ledger must be an envelope object")
    measured_fns, measured_tps, support_meta = _validate_support_lineage(
        panel=panel,
        h0=h0,
        support_envelope=support_envelope,
        support_source_info=support_source_info,
        h0_by_owner=h0_by_owner,
        strict=strict_contract,
    )
    measured_fn_ids = {str(row["gt_owner_id"]) for row in measured_fns}
    completion_rows = [row for row in h0_fns if str(row["gt_owner_id"]) not in measured_fn_ids]
    if strict_contract and len(completion_rows) != EXPECTED_COMPLETION_CONTEXTS:
        raise SupportCompletionPlanError(
            f"missing native-FN scope is {len(completion_rows)}, expected 200"
        )

    physical, accounting = support_probe.build_physical_owner_bank(panel)
    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for candidate in physical:
        groups.setdefault(
            (int(candidate["image_id"]), str(candidate["normalized_description"]).lower()), []
        ).append(candidate)
    panel_owner_by_id = {
        owner.gt_owner_id: owner
        for owners in panel.owners_by_image.values()
        for owner in owners
    }
    contexts: list[dict[str, Any]] = []
    for row in completion_rows:
        owner_id = str(row["gt_owner_id"])
        owner = panel_owner_by_id.get(owner_id)
        if owner is None:
            raise SupportCompletionPlanError(f"completion owner absent from panel: {owner_id}")
        if str(owner.category or "").lower() != str(row.get("category_name", "")).lower():
            raise SupportCompletionPlanError(f"completion category mapping mismatch: {owner_id}")
        if tuple(owner.bbox or ()) != tuple(row.get("bbox_pixel_xyxy") or ()):
            raise SupportCompletionPlanError(f"completion geometry mapping mismatch: {owner_id}")
        owner_identity = _owner_row_identity(panel=panel, owner=owner, h0_row=row)
        target_rows = _context_target_rows(
            physical=groups.get(
                (int(row["image_id"]), str(row.get("category_name", "")).lower()), []
            ),
            image_id=int(row["image_id"]),
            category=str(row.get("category_name", "")),
        )
        contexts.append(
            _make_context(
                row=row,
                owner=owner,
                owner_identity=owner_identity,
                target_rows=target_rows,
                batch_size=candidate_batch_size,
                shard_count=num_shards,
            )
        )
    contexts.sort(key=lambda item: str(item["stable_key"]))
    for position, context in enumerate(contexts):
        context["context_plan_position"] = position
    work = _work_accounting(
        contexts, batch_size=candidate_batch_size, shard_count=num_shards
    )
    _enforce_exact_contract(
        panel=panel,
        tps=h0_tps,
        fns=h0_fns,
        measured_fns=measured_fns,
        contexts=contexts,
        work=work,
        strict=strict_contract,
    )

    tp_ids = [str(row["gt_owner_id"]) for row in h0_tps]
    measured_tp_ids = [str(row["gt_owner_id"]) for row in measured_tps]
    unscored_tp_ids = sorted(set(tp_ids) - set(measured_tp_ids))
    if strict_contract and len(unscored_tp_ids) != EXPECTED_UNSCORED_TP:
        raise SupportCompletionPlanError("other native TP count is not 160")
    completion_ids = [str(row["gt_owner_id"]) for row in contexts]
    context_ids = [str(row["context_id"]) for row in contexts]
    candidate_index: dict[str, str] = {}
    for context in contexts:
        for candidate, identity_hash in zip(
            context["candidate_ids"], context["candidate_identity_hashes"], strict=True
        ):
            existing = candidate_index.get(str(candidate))
            if existing is not None and existing != str(identity_hash):
                raise SupportCompletionPlanError(f"candidate identity hash drifted: {candidate}")
            candidate_index[str(candidate)] = str(identity_hash)
    plan: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "sealed_cpu_plan",
        "checkpoint": CHECKPOINT,
        "wrapper": CHECKPOINT_WRAPPER,
        "parser": CHECKPOINT_PARSER,
        "execution_contract": {
            "cpu_only_planner": True,
            "model_loaded": False,
            "gpu_used": False,
            "training": False,
            "support_capture_held_until_s_gate": True,
            "native_h0_prefix_only": True,
            "teacher_forced_diagnostic_only": True,
            "batching_admitted": False,
            "source_frozen_candidate_registry_mutated": False,
        },
        "panel": {
            "source": dict(panel.source_info),
            "derived": dict(panel.derived_info),
            "derived_receipt": dict(panel.receipt_info),
            "source_sha256": panel.source_info["sha256"],
            "derived_sha256": panel.derived_info["sha256"],
            "derived_receipt_sha256": panel.receipt_info["sha256"],
            "image_count": len(panel.owners_by_image),
            "owner_count": sum(len(items) for items in panel.owners_by_image.values()),
            "image_ids": sorted(int(image) for image in panel.owners_by_image),
        },
        "h0_lineage": {
            "source": dict(h0_source_info),
            "source_sha256": h0_source_info["sha256"],
            "envelope_content_sha256": sha256_json(h0.envelope),
            "schema_version": h0.envelope.get("schema_version"),
            "unit_id": h0.envelope.get("unit_id"),
            "checkpoint": CHECKPOINT,
            "config_fingerprint": h0.envelope.get("config_fingerprint"),
            "record_count": len(h0.records),
            "native_tp_count": len(h0_tps),
            "native_fn_count": len(h0_fns),
            "native_tp_calibration_control_assessed": len(h0_tps),
            "native_tp_calibration_excluded": 0,
            "records_sha256": sha256_json(h0.records),
        },
        "support_lineage": support_meta,
        "calibration_reuse": {
            "reused": True,
            "calibration_sha256": support_meta["calibration_sha256"],
            "rule": support_meta["support_rule"],
            "calibration": support_meta["calibration"],
            "observation_count": support_meta["calibration"].get("observation_count"),
            "excluded": 0,
            "source": "frozen_S_native_tp_calibration",
            "recomputed": False,
        },
        "scope": {
            "native_fn_denominator": len(h0_fns),
            "support_measured_fn_retained": len(measured_fns),
            "support_unassessed_fn": len(completion_rows),
            "support_completion_candidates": len(contexts),
            "native_tp_calibration_complete": len(h0_tps),
            "native_tp_other_not_scored": len(unscored_tp_ids),
            "native_tp_other_not_scored_owner_ids_sha256": sha256_json(unscored_tp_ids),
            "native_tp_other_not_scored_policy": "native_already_covered_never_support_negative",
            "completion_owner_ids_sha256": sha256_json(sorted(completion_ids)),
            "retained_measured_fn_owner_ids_sha256": sha256_json(sorted(measured_fn_ids)),
        },
        "work": work,
        "contexts": contexts,
        "context_ids_sha256": sha256_json(context_ids),
        "context_owner_ids_sha256": sha256_json(completion_ids),
        "candidate_identity_index_count": len(candidate_index),
        "candidate_identity_index_sha256": sha256_json(candidate_index),
        "future_measurement_fields": {
            "measured_wall_time_seconds": None,
            "realized_scalar_forward_count": None,
            "realized_batched_forward_count": None,
            "execution_receipt_sha256": None,
        },
    }
    plan["plan_content_sha256"] = sha256_json(plan)
    return plan


def validate_plan(plan: Mapping[str, Any], *, strict_contract: bool = True) -> dict[str, Any]:
    """Validate an already materialized plan without loading a model."""

    if plan.get("schema_version") != SCHEMA_VERSION or plan.get("unit_id") != UNIT_ID:
        raise SupportCompletionPlanError("plan schema/unit identity mismatch")
    declared_content_hash = plan.get("plan_content_sha256")
    if not isinstance(declared_content_hash, str):
        raise SupportCompletionPlanError("plan content hash is missing")
    content_without_hash = dict(plan)
    content_without_hash.pop("plan_content_sha256", None)
    if sha256_json(content_without_hash) != declared_content_hash:
        raise SupportCompletionPlanError("plan content hash mismatch")
    contexts = plan.get("contexts")
    if not isinstance(contexts, list):
        raise SupportCompletionPlanError("plan contexts are missing")
    seen_contexts: set[str] = set()
    for context in contexts:
        if not isinstance(context, Mapping):
            raise SupportCompletionPlanError("plan context is not an object")
        stable = context.get("stable_key")
        context_id = context.get("context_id")
        if not isinstance(stable, str) or context_id != context_id_for_stable_key(stable):
            raise SupportCompletionPlanError("plan context stable identity mismatch")
        if context_id in seen_contexts or stable in seen_contexts:
            raise SupportCompletionPlanError("plan context identity is duplicated")
        seen_contexts.update((str(context_id), stable))
        if int(context.get("shard_index")) != shard_index_for_stable_key(stable, EXPECTED_SHARDS):
            raise SupportCompletionPlanError("plan context shard assignment drifted")
        ids = context.get("candidate_ids")
        rows = context.get("target_rows")
        if not isinstance(ids, list) or not isinstance(rows, list) or ids != [r.get("candidate_id") for r in rows]:
            raise SupportCompletionPlanError("plan candidate/target-row identity mismatch")
        if context.get("candidate_ids_sha256") != sha256_json(ids):
            raise SupportCompletionPlanError("plan candidate IDs hash mismatch")
        if context.get("target_rows_sha256") != sha256_json(rows):
            raise SupportCompletionPlanError("plan target rows hash mismatch")
        if context.get("scalar_equivalent_forward_count") != len(ids):
            raise SupportCompletionPlanError("plan scalar forward count mismatch")
        if context.get("measured_wall_time_seconds") is not None or context.get("realized_scalar_forward_count") is not None:
            raise SupportCompletionPlanError("sealed plan contains realized execution fields")
    work = plan.get("work")
    if not isinstance(work, Mapping):
        raise SupportCompletionPlanError("plan work accounting is missing")
    expected_work = _work_accounting(
        contexts,
        batch_size=int(work.get("candidate_batch_size", DEFAULT_CANDIDATE_BATCH_SIZE)),
        shard_count=int(work.get("shard_count", EXPECTED_SHARDS)),
    )
    if expected_work != work:
        raise SupportCompletionPlanError("plan work accounting is not reproducible")
    if strict_contract:
        observed_contexts = tuple(int(row["context_count"]) for row in work["per_shard"])
        observed_forwards = tuple(int(row["scalar_equivalent_forward_count"]) for row in work["per_shard"])
        if len(contexts) != EXPECTED_COMPLETION_CONTEXTS or observed_contexts != EXPECTED_SHARD_CONTEXTS or observed_forwards != EXPECTED_SHARD_FORWARDS:
            raise SupportCompletionPlanError("plan does not satisfy the sealed 200-context/8-shard contract")
    return dict(plan)


def _write_immutable(path: str | Path, payload: bytes) -> str:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        existing = destination.read_bytes()
        if existing != payload:
            raise SupportCompletionPlanError(f"existing output is not identical: {destination}")
    else:
        destination.write_bytes(payload)
    return sha256_bytes(payload)


def materialize_plan_and_receipt(
    plan: Mapping[str, Any],
    *,
    output_path: str | Path,
    receipt_path: str | Path,
    strict_contract: bool = True,
) -> dict[str, Any]:
    """Write canonical immutable plan and receipt; never overwrite drift."""

    validate_plan(plan, strict_contract=strict_contract)
    plan_payload = canonical_json_bytes(plan) + b"\n"
    plan_sha256 = _write_immutable(output_path, plan_payload)
    contexts = plan["contexts"]
    work = plan["work"]
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "status": "sealed_cpu_plan",
        "plan_path": str(Path(output_path).expanduser().resolve()),
        "plan_sha256": plan_sha256,
        "plan_content_sha256": plan.get("plan_content_sha256"),
        "context_count": len(contexts),
        "context_ids_sha256": plan.get("context_ids_sha256"),
        "context_owner_ids_sha256": plan.get("context_owner_ids_sha256"),
        "candidate_identity_index_count": plan.get("candidate_identity_index_count"),
        "candidate_identity_index_sha256": plan.get("candidate_identity_index_sha256"),
        "shard_count": work["shard_count"],
        "per_shard": work["per_shard"],
        "scalar_equivalent_forward_count": work["scalar_equivalent_forward_count"],
        "safe_batched_forward_estimate": work["safe_batched_forward_estimate"],
        "batching_admitted": False,
        "calibration_sha256": plan["calibration_reuse"]["calibration_sha256"],
        "native_tp_calibration_observation_count": plan["calibration_reuse"]["observation_count"],
        "native_tp_calibration_excluded": plan["calibration_reuse"]["excluded"],
        "native_fn_denominator": plan["scope"]["native_fn_denominator"],
        "retained_measured_fn_count": plan["scope"]["support_measured_fn_retained"],
        "support_completion_candidate_count": plan["scope"]["support_completion_candidates"],
        "native_tp_other_not_scored": plan["scope"]["native_tp_other_not_scored"],
        "gpu_used": False,
        "model_loaded": False,
        "measured_wall_time_seconds": None,
        "realized_scalar_forward_count": None,
        "realized_batched_forward_count": None,
        "execution_receipt_sha256": None,
    }
    receipt_payload = canonical_json_bytes(receipt) + b"\n"
    receipt_sha256 = _write_immutable(receipt_path, receipt_payload)
    return {
        "plan_path": str(Path(output_path).expanduser().resolve()),
        "receipt_path": str(Path(receipt_path).expanduser().resolve()),
        "plan_sha256": plan_sha256,
        "receipt_sha256": receipt_sha256,
        "plan": plan,
        "receipt": receipt,
    }


# Short aliases keep the CPU planner convenient for small audit/test callers
# while the descriptive names remain the public documentation surface.
build_plan = build_support_completion_plan
materialize_plan = materialize_plan_and_receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-panel", type=Path, default=DEFAULT_SOURCE_PANEL)
    parser.add_argument("--derived-panel", type=Path, default=DEFAULT_DERIVED_PANEL)
    parser.add_argument("--derived-receipt", type=Path, default=DEFAULT_DERIVED_RECEIPT)
    parser.add_argument("--h0-ledger", type=Path, default=DEFAULT_H0_LEDGER)
    parser.add_argument("--support-ledger", type=Path, default=DEFAULT_SUPPORT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "plan.json")
    parser.add_argument("--receipt", type=Path, default=DEFAULT_OUTPUT_ROOT / "receipt.json")
    parser.add_argument("--candidate-batch-size", type=int, default=DEFAULT_CANDIDATE_BATCH_SIZE)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    plan = build_support_completion_plan(
        source_panel=args.source_panel,
        derived_panel=args.derived_panel,
        derived_receipt=args.derived_receipt,
        h0_ledger=args.h0_ledger,
        support_ledger=args.support_ledger,
        candidate_batch_size=args.candidate_batch_size,
        num_shards=EXPECTED_SHARDS,
        strict_contract=True,
    )
    result = materialize_plan_and_receipt(
        plan,
        output_path=args.output,
        receipt_path=args.receipt,
        strict_contract=True,
    )
    print(
        json.dumps(
            {
                "status": "sealed_cpu_plan",
                "plan_path": result["plan_path"],
                "plan_sha256": result["plan_sha256"],
                "receipt_path": result["receipt_path"],
                "receipt_sha256": result["receipt_sha256"],
                "context_count": len(plan["contexts"]),
                "scalar_equivalent_forward_count": plan["work"]["scalar_equivalent_forward_count"],
                "safe_batched_forward_estimate": plan["work"]["safe_batched_forward_estimate"],
                "per_shard": plan["work"]["per_shard"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
