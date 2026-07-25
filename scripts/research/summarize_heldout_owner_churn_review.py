#!/usr/bin/env python3
"""Reduce the arm-blind held-out owner-churn review after integrity checks.

The blind manifest and completed review shards are validated before private
unblinding data is read.  The reducer preserves the official geometry ledger
as provenance and reports the human review as a separate interpretation; it
does not relabel ground truth or treat unmatched predictions as hallucinations.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NoReturn


SUMMARY_SCHEMA_VERSION = "heldout_owner_churn_audit_summary.v1"
BLIND_MANIFEST_SCHEMA_VERSION = (
    "heldout_owner_change_review_packet.v1.blind_manifest"
)
BLIND_CASE_SCHEMA_VERSION = "heldout_owner_change_review_packet.v1.blind_case"
PRIVATE_UNBLINDING_SCHEMA_VERSION = (
    "heldout_owner_change_review_packet.v1.private_unblinding"
)
EXPECTED_CASE_COUNT = 85
EXPECTED_TREATMENT_ONLY_COUNT = 46
EXPECTED_SOURCE_ONLY_COUNT = 39
EXPECTED_REVIEW_SHARD_COUNT = 5
EXPECTED_CHECKPOINT_ROLES = {
    "arm_a": "Source",
    "arm_b": "transition step 36",
}
ENTITY_CATEGORY_DISPOSITIONS = (
    "real_owner_change",
    "duplicate",
    "category_alias_or_disagreement",
    "unsupported_candidate",
    "uncertain",
)
GEOMETRY_DISPOSITIONS = (
    "acceptable",
    "localization_error",
    "neighboring_instance_or_mixed_extent_error",
    "uncertain",
)
REFINED_INTERPRETATIONS = (
    "genuine_real_owner_change",
    "geometry",
    "alias",
    "duplicate",
    "unsupported",
    "uncertain",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


class ReviewSummaryContractError(ValueError):
    """Raised when review inputs cannot support a trustworthy reduction."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _contract_error(message: str) -> NoReturn:
    raise ReviewSummaryContractError(message)


def _reject_constant(value: str) -> NoReturn:
    _contract_error(f"non-finite JSON number is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _contract_error(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _parse_json(text: str, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_constant,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ReviewSummaryContractError(f"{label} is not valid JSON: {exc}") from exc


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        value = _parse_json(path.read_text(encoding="utf-8"), label)
    except OSError as exc:
        raise ReviewSummaryContractError(f"cannot read {label}: {path}: {exc}") from exc
    return _mapping(value, label)


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ReviewSummaryContractError(f"cannot read {label}: {path}: {exc}") from exc
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            _contract_error(f"{label} contains a blank line at {line_number}")
        rows.append(_mapping(_parse_json(line, f"{label}:{line_number}"), label))
    return rows


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _contract_error(f"{label} must be a JSON object")
    return value


def _list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        _contract_error(f"{label} must be a JSON list")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        _contract_error(f"{label} must be non-empty text")
    return value


def _integer(value: Any, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        _contract_error(f"{label} must be an integer >= {minimum}")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        _contract_error(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )


def _declared_sha256(value: Any, label: str) -> str:
    digest = _text(value, label)
    if _SHA256_PATTERN.fullmatch(digest) is None:
        _contract_error(f"{label} must be a lowercase SHA-256 digest")
    return digest


def _validate_hash(path: Path, declared: Any, label: str) -> str:
    expected = _declared_sha256(declared, f"{label}.sha256")
    actual = sha256_file(path)
    if actual != expected:
        _contract_error(
            f"{label} hash mismatch for {path}: expected {expected}, got {actual}"
        )
    return actual


def _resolve_relative_file(root: Path, declared: Any, label: str) -> Path:
    relative = Path(_text(declared, label))
    if relative.is_absolute():
        _contract_error(f"{label} must be relative to {root}")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError:
        _contract_error(f"{label} escapes its declared root: {relative}")
    if not resolved.is_file():
        _contract_error(f"{label} is not a file: {resolved}")
    return resolved


def _resolve_expected_relative_file(
    base: Path, declared: Any, expected: Path, label: str
) -> Path:
    """Resolve a declared relative pointer only when it names the expected file."""
    relative = Path(_text(declared, label))
    if relative.is_absolute():
        _contract_error(f"{label} must be relative to {base}")
    resolved = (base.resolve() / relative).resolve()
    if resolved != expected.resolve():
        _contract_error(f"{label} is not the supplied file: {resolved}")
    if not resolved.is_file():
        _contract_error(f"{label} is not a file: {resolved}")
    return resolved


def _expected_case_ids(case_count: int) -> tuple[str, ...]:
    return tuple(f"case_{index:04d}" for index in range(1, case_count + 1))


def _validate_review_schema(manifest: Mapping[str, Any]) -> None:
    schema = _mapping(manifest.get("review_schema"), "manifest.review_schema")
    _exact_keys(
        schema,
        {"entity_category", "geometry", "notes"},
        "manifest.review_schema",
    )
    entity = _mapping(schema.get("entity_category"), "review_schema.entity_category")
    geometry = _mapping(schema.get("geometry"), "review_schema.geometry")
    notes = _mapping(schema.get("notes"), "review_schema.notes")
    if entity != {
        "required": True,
        "allowed_values": list(ENTITY_CATEGORY_DISPOSITIONS),
    }:
        _contract_error("manifest entity/category review enum is not the official enum")
    if geometry != {
        "required": True,
        "allowed_values": list(GEOMETRY_DISPOSITIONS),
    }:
        _contract_error("manifest geometry review enum is not the official enum")
    if notes != {"required": False, "type": "string_or_null"}:
        _contract_error("manifest notes schema is not the official optional string schema")


def _validate_template(path: Path, case_ids: Sequence[str]) -> None:
    rows = _read_jsonl(path, "manifest disposition template")
    if len(rows) != len(case_ids):
        _contract_error(
            f"disposition template count differs: expected {len(case_ids)}, got {len(rows)}"
        )
    for expected_id, row in zip(case_ids, rows, strict=True):
        _exact_keys(
            row,
            {"case_id", "entity_category", "geometry", "notes"},
            f"template row {expected_id}",
        )
        if row != {
            "case_id": expected_id,
            "entity_category": "pending",
            "geometry": "pending",
            "notes": None,
        }:
            _contract_error(f"disposition template row differs for {expected_id}")


def _validate_blind_manifest(
    manifest_path: Path,
    *,
    expected_case_count: int,
) -> tuple[Mapping[str, Any], tuple[str, ...], str]:
    manifest_path = manifest_path.resolve()
    manifest = _read_json(manifest_path, "reviewer manifest")
    _exact_keys(
        manifest,
        {
            "schema_version",
            "blinded",
            "case_count",
            "hidden_fields",
            "official_matching",
            "review_schema",
            "instructions",
            "instructions_sha256",
            "disposition_template",
            "disposition_template_sha256",
            "cases",
        },
        "reviewer manifest",
    )
    if manifest.get("schema_version") != BLIND_MANIFEST_SCHEMA_VERSION:
        _contract_error("reviewer manifest schema_version is not supported")
    if manifest.get("blinded") is not True:
        _contract_error("reviewer manifest must declare blinded=true")
    if manifest.get("hidden_fields") != [
        "checkpoint_identity",
        "physical_owner_change_direction",
    ]:
        _contract_error("reviewer manifest hidden_fields contract differs")
    case_count = _integer(manifest.get("case_count"), "manifest.case_count")
    if case_count != expected_case_count:
        _contract_error(
            f"manifest case_count differs: expected {expected_case_count}, got {case_count}"
        )
    official_matching = _mapping(
        manifest.get("official_matching"), "manifest.official_matching"
    )
    if official_matching != {
        "method": "cardinality_first_maximum_total_intersection_over_union",
        "threshold": 0.5,
        "unmatched_prediction_semantics": "unresolved_pending_human_review",
    }:
        _contract_error("manifest official matching contract differs")
    _validate_review_schema(manifest)

    reviewer_root = manifest_path.parent
    instructions_path = _resolve_relative_file(
        reviewer_root, manifest.get("instructions"), "manifest.instructions"
    )
    _validate_hash(
        instructions_path,
        manifest.get("instructions_sha256"),
        "manifest instructions",
    )
    template_path = _resolve_relative_file(
        reviewer_root,
        manifest.get("disposition_template"),
        "manifest.disposition_template",
    )
    _validate_hash(
        template_path,
        manifest.get("disposition_template_sha256"),
        "manifest disposition template",
    )

    case_ids = _expected_case_ids(case_count)
    case_entries = _list(manifest.get("cases"), "manifest.cases")
    if len(case_entries) != case_count:
        _contract_error(
            f"manifest cases length differs: expected {case_count}, got {len(case_entries)}"
        )
    observed_ids: list[str] = []
    observed_paths: set[Path] = set()
    for expected_id, raw_entry in zip(case_ids, case_entries, strict=True):
        entry = _mapping(raw_entry, f"manifest case {expected_id}")
        _exact_keys(
            entry,
            {
                "case_id",
                "case_file",
                "case_file_sha256",
                "full_image",
                "full_image_sha256",
                "enlarged_crop",
                "enlarged_crop_sha256",
            },
            f"manifest case {expected_id}",
        )
        case_id = _text(entry.get("case_id"), "manifest case_id")
        observed_ids.append(case_id)
        if case_id != expected_id:
            _contract_error(
                f"manifest case order/id differs: expected {expected_id}, got {case_id}"
            )
        paths: dict[str, Path] = {}
        for path_field, hash_field in (
            ("case_file", "case_file_sha256"),
            ("full_image", "full_image_sha256"),
            ("enlarged_crop", "enlarged_crop_sha256"),
        ):
            path = _resolve_relative_file(
                reviewer_root,
                entry.get(path_field),
                f"manifest case {case_id}.{path_field}",
            )
            if path in observed_paths:
                _contract_error(f"manifest file is reused across cases: {path}")
            observed_paths.add(path)
            _validate_hash(path, entry.get(hash_field), f"manifest case {case_id}.{path_field}")
            paths[path_field] = path

        blind_case = _read_json(paths["case_file"], f"blind case {case_id}")
        if blind_case.get("schema_version") != BLIND_CASE_SCHEMA_VERSION:
            _contract_error(f"blind case {case_id} schema_version differs")
        if blind_case.get("blinded") is not True or blind_case.get("case_id") != case_id:
            _contract_error(f"blind case {case_id} identity/blinding contract differs")
        artifacts = _mapping(blind_case.get("artifacts"), f"blind case {case_id}.artifacts")
        for path_field, hash_field in (
            ("full_image", "full_image_sha256"),
            ("enlarged_crop", "enlarged_crop_sha256"),
        ):
            if artifacts.get(path_field) != entry.get(path_field):
                _contract_error(f"blind case {case_id} {path_field} path differs")
            if artifacts.get(hash_field) != entry.get(hash_field):
                _contract_error(f"blind case {case_id} {path_field} hash differs")
        views = _mapping(blind_case.get("views"), f"blind case {case_id}.views")
        if set(views) != {"view_a", "view_b"}:
            _contract_error(f"blind case {case_id} must contain view_a and view_b")

    if tuple(observed_ids) != case_ids or len(set(observed_ids)) != case_count:
        _contract_error("manifest does not contain the exact expected case IDs once")
    _validate_template(template_path, case_ids)
    return manifest, case_ids, sha256_file(manifest_path)


def _validate_blind_reviews(
    review_shard_paths: Sequence[Path],
    *,
    reviewer_root: Path,
    case_ids: Sequence[str],
    expected_shard_count: int,
) -> tuple[dict[str, Mapping[str, Any]], list[dict[str, str]]]:
    if len(review_shard_paths) != expected_shard_count:
        _contract_error(
            f"review shard count differs: expected {expected_shard_count}, "
            f"got {len(review_shard_paths)}"
        )
    paths = sorted(path.resolve() for path in review_shard_paths)
    if len(set(paths)) != len(paths):
        _contract_error("review shard paths contain duplicates")
    expected_parent = (reviewer_root / "review-shards").resolve()
    records: list[Mapping[str, Any]] = []
    shard_inputs: list[dict[str, str]] = []
    for path in paths:
        if path.parent != expected_parent or not path.is_file():
            _contract_error(f"review shard is not a file under {expected_parent}: {path}")
        records.extend(_read_jsonl(path, f"review shard {path.name}"))
        shard_inputs.append({"path": str(path), "sha256": sha256_file(path)})

    review_by_id: dict[str, Mapping[str, Any]] = {}
    observed_ids: list[str] = []
    for row_number, record in enumerate(records, start=1):
        _exact_keys(
            record,
            {"case_id", "entity_category", "geometry", "notes"},
            f"review record {row_number}",
        )
        case_id = _text(record.get("case_id"), f"review record {row_number}.case_id")
        entity = record.get("entity_category")
        geometry = record.get("geometry")
        notes = record.get("notes")
        if entity not in ENTITY_CATEGORY_DISPOSITIONS:
            _contract_error(
                f"review record {case_id} has invalid entity/category disposition: {entity!r}"
            )
        if geometry not in GEOMETRY_DISPOSITIONS:
            _contract_error(
                f"review record {case_id} has invalid geometry disposition: {geometry!r}"
            )
        if notes is not None and not isinstance(notes, str):
            _contract_error(f"review record {case_id}.notes must be text or null")
        observed_ids.append(case_id)
        if case_id in review_by_id:
            _contract_error(f"review case ID occurs more than once: {case_id}")
        review_by_id[case_id] = record

    expected_counter = Counter(case_ids)
    observed_counter = Counter(observed_ids)
    missing = sorted((expected_counter - observed_counter).elements())
    unexpected = sorted((observed_counter - expected_counter).elements())
    if missing or unexpected or len(records) != len(case_ids):
        _contract_error(
            "review case IDs are not the exact manifest set once: "
            f"missing={missing}, unexpected={unexpected}, count={len(records)}"
        )
    return review_by_id, shard_inputs


def _parse_ledger_refs(
    geometry: Mapping[str, Any],
    *,
    side: str,
    expected_count: int,
) -> tuple[tuple[str, int], ...]:
    count_field = f"{side}_owner_count"
    refs_field = f"{side}_owner_refs"
    count = _integer(geometry.get(count_field), f"common_owner_geometry.{count_field}")
    if count != expected_count:
        _contract_error(
            f"official {side} count differs: expected {expected_count}, got {count}"
        )
    raw_refs = _list(geometry.get(refs_field), f"common_owner_geometry.{refs_field}")
    refs: list[tuple[str, int]] = []
    for index, raw_ref in enumerate(raw_refs):
        ref = _mapping(raw_ref, f"{refs_field}[{index}]")
        _exact_keys(ref, {"row_id", "owner_index"}, f"{refs_field}[{index}]")
        refs.append(
            (
                _text(ref.get("row_id"), f"{refs_field}[{index}].row_id"),
                _integer(ref.get("owner_index"), f"{refs_field}[{index}].owner_index"),
            )
        )
    if len(refs) != count or len(set(refs)) != count:
        _contract_error(f"official {side} references are not exactly {count} unique items")
    return tuple(refs)


def _validate_target_attribution(
    attribution: Mapping[str, Any],
    *,
    case_id: str,
    matched_arm: str,
) -> None:
    if set(attribution) != {"arm_a", "arm_b"}:
        _contract_error(f"unblinding case {case_id} attribution arms differ")
    for arm in ("arm_a", "arm_b"):
        arm_value = _mapping(attribution.get(arm), f"unblinding case {case_id}.{arm}")
        target = _mapping(
            arm_value.get("target"), f"unblinding case {case_id}.{arm}.target"
        )
        expected_status = "matched" if arm == matched_arm else "unmatched_unresolved"
        if target.get("status") != expected_status:
            _contract_error(
                f"unblinding case {case_id} {arm} target status differs: "
                f"expected {expected_status!r}"
            )
        if expected_status == "matched":
            _integer(target.get("prediction_index"), f"{case_id}.{arm}.prediction_index")
            overlap = target.get("intersection_over_union")
            if not isinstance(overlap, (int, float)) or isinstance(overlap, bool):
                _contract_error(f"unblinding case {case_id} matched overlap is invalid")
            if not 0.5 <= float(overlap) <= 1.0:
                _contract_error(
                    f"unblinding case {case_id} matched overlap is below official threshold"
                )


def _validate_unblinding_and_ledger(
    unblinding_path: Path,
    comparison_ledger_path: Path,
    *,
    manifest_path: Path,
    manifest_sha256: str,
    case_ids: Sequence[str],
    expected_treatment_only_count: int,
    expected_source_only_count: int,
) -> tuple[
    dict[str, Mapping[str, Any]],
    Mapping[str, Any],
    Mapping[str, Any],
    str,
    str,
]:
    unblinding_path = unblinding_path.resolve()
    comparison_ledger_path = comparison_ledger_path.resolve()
    unblinding = _read_json(unblinding_path, "private unblinding")
    if unblinding.get("schema_version") != PRIVATE_UNBLINDING_SCHEMA_VERSION:
        _contract_error("private unblinding schema_version is not supported")
    roles = _mapping(unblinding.get("checkpoint_roles"), "unblinding.checkpoint_roles")
    if roles != EXPECTED_CHECKPOINT_ROLES:
        _contract_error("checkpoint roles differ from Source and transition step 36")

    manifest_ref = _mapping(
        unblinding.get("reviewer_manifest"), "unblinding.reviewer_manifest"
    )
    linked_manifest = _resolve_expected_relative_file(
        unblinding_path.parent,
        manifest_ref.get("path"),
        manifest_path,
        "unblinding.reviewer_manifest.path",
    )
    linked_manifest_sha = _validate_hash(
        linked_manifest,
        manifest_ref.get("sha256"),
        "unblinding reviewer manifest",
    )
    if linked_manifest_sha != manifest_sha256:
        _contract_error("reviewer manifest changed between blind and private phases")

    ledger_ref = _mapping(
        unblinding.get("comparison_ledger"), "unblinding.comparison_ledger"
    )
    linked_ledger = _resolve_relative_file(
        unblinding_path.parent,
        ledger_ref.get("private_copy"),
        "unblinding.comparison_ledger.private_copy",
    )
    if linked_ledger != comparison_ledger_path:
        _contract_error("unblinding private ledger path is not the supplied ledger")
    ledger_sha = _validate_hash(
        linked_ledger,
        ledger_ref.get("private_copy_sha256"),
        "unblinding private comparison ledger",
    )
    source_declared_sha = _declared_sha256(
        ledger_ref.get("sha256"), "unblinding.comparison_ledger.sha256"
    )
    if source_declared_sha != ledger_sha:
        _contract_error("private comparison ledger hash differs from source ledger hash")

    ledger = _read_json(comparison_ledger_path, "original comparison ledger")
    policy = _mapping(ledger.get("policy"), "comparison ledger.policy")
    if policy.get("delta_convention") != "arm_b_minus_arm_a":
        _contract_error("comparison ledger delta convention differs")
    if policy.get("match_iou_threshold") != 0.5:
        _contract_error("comparison ledger match threshold differs from 0.5")
    if policy.get("unmatched_predictions_are_not_hallucinations") is not True:
        _contract_error("comparison ledger unmatched-prediction semantics differ")
    ledger_inputs = _mapping(ledger.get("inputs"), "comparison ledger.inputs")
    unblinding_inputs = _mapping(unblinding.get("inputs"), "unblinding.inputs")
    for arm in ("arm_a", "arm_b"):
        ledger_arm = _mapping(ledger_inputs.get(arm), f"comparison ledger.inputs.{arm}")
        unblinding_arm = _mapping(unblinding_inputs.get(arm), f"unblinding.inputs.{arm}")
        if unblinding_arm != ledger_arm:
            _contract_error(f"unblinding and comparison ledger inputs differ for {arm}")
        _text(ledger_arm.get("path"), f"comparison ledger.inputs.{arm}.path")
        _declared_sha256(
            ledger_arm.get("sha256"), f"comparison ledger.inputs.{arm}.sha256"
        )

    geometry = _mapping(
        ledger.get("common_owner_geometry"), "comparison ledger.common_owner_geometry"
    )
    source_refs = _parse_ledger_refs(
        geometry,
        side="arm_a_only",
        expected_count=expected_source_only_count,
    )
    treatment_refs = _parse_ledger_refs(
        geometry,
        side="arm_b_only",
        expected_count=expected_treatment_only_count,
    )
    if set(source_refs) & set(treatment_refs):
        _contract_error("official source-only and treatment-only owner references overlap")

    raw_cases = _list(unblinding.get("cases"), "unblinding.cases")
    if len(raw_cases) != len(case_ids):
        _contract_error(
            f"unblinding case count differs: expected {len(case_ids)}, got {len(raw_cases)}"
        )
    cases: dict[str, Mapping[str, Any]] = {}
    refs_by_side: dict[str, list[tuple[str, int]]] = {
        "arm_a_only": [],
        "arm_b_only": [],
    }
    direction_counts: Counter[str] = Counter()
    for raw_case in raw_cases:
        case = _mapping(raw_case, "unblinding case")
        case_id = _text(case.get("case_id"), "unblinding case.case_id")
        if case_id in cases:
            _contract_error(f"unblinding case ID occurs more than once: {case_id}")
        if case_id not in case_ids:
            _contract_error(f"unblinding contains unexpected case ID: {case_id}")
        row_id = _text(case.get("row_id"), f"unblinding case {case_id}.row_id")
        owner_index = _integer(
            case.get("owner_index"), f"unblinding case {case_id}.owner_index"
        )
        side = case.get("ledger_reference_side")
        if side not in refs_by_side:
            _contract_error(f"unblinding case {case_id} has invalid ledger side: {side!r}")
        expected_direction = "loss" if side == "arm_a_only" else "gain"
        if case.get("physical_owner_change_direction") != expected_direction:
            _contract_error(f"unblinding case {case_id} direction contradicts ledger side")
        views = _mapping(case.get("views"), f"unblinding case {case_id}.views")
        if set(views) != {"view_a", "view_b"} or set(views.values()) != {
            "arm_a",
            "arm_b",
        }:
            _contract_error(f"unblinding case {case_id} views are not an arm bijection")
        _validate_target_attribution(
            _mapping(
                case.get("authoritative_attribution"),
                f"unblinding case {case_id}.authoritative_attribution",
            ),
            case_id=case_id,
            matched_arm="arm_a" if side == "arm_a_only" else "arm_b",
        )
        refs_by_side[str(side)].append((row_id, owner_index))
        direction_counts[expected_direction] += 1
        cases[case_id] = case

    if set(cases) != set(case_ids):
        _contract_error("unblinding case IDs are not a bijection with manifest case IDs")
    if len(refs_by_side["arm_a_only"]) != len(set(refs_by_side["arm_a_only"])):
        _contract_error("unblinding repeats a source-only owner reference")
    if len(refs_by_side["arm_b_only"]) != len(set(refs_by_side["arm_b_only"])):
        _contract_error("unblinding repeats a treatment-only owner reference")
    if set(refs_by_side["arm_a_only"]) != set(source_refs):
        _contract_error("unblinding source-only references are not bijective with the ledger")
    if set(refs_by_side["arm_b_only"]) != set(treatment_refs):
        _contract_error("unblinding treatment-only references are not bijective with the ledger")
    if direction_counts != Counter(
        {"gain": expected_treatment_only_count, "loss": expected_source_only_count}
    ):
        _contract_error("unblinding direction counts differ from official counts")
    return (
        cases,
        ledger,
        geometry,
        sha256_file(unblinding_path),
        ledger_sha,
    )


def refined_interpretation(entity: str, geometry: str) -> str:
    """Combine the two blind axes without promoting geometry churn to genuine."""
    if entity == "real_owner_change":
        if geometry == "acceptable":
            return "genuine_real_owner_change"
        if geometry == "uncertain":
            return "uncertain"
        return "geometry"
    if entity == "category_alias_or_disagreement":
        return "alias"
    if entity == "duplicate":
        return "duplicate"
    if entity == "unsupported_candidate":
        return "unsupported"
    return "uncertain"


def _counts_by_direction(
    rows: Sequence[Mapping[str, Any]],
    *,
    key: str,
    values: Sequence[str],
) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    for value in values:
        gain_count = sum(
            row[key] == value and row["official_direction"] == "gain" for row in rows
        )
        loss_count = sum(
            row[key] == value and row["official_direction"] == "loss" for row in rows
        )
        result[value] = {
            "gain_count": gain_count,
            "loss_count": loss_count,
            "net_gain_minus_loss": gain_count - loss_count,
            "total_count": gain_count + loss_count,
        }
    return result


def _json_bytes(value: Any, *, pretty: bool) -> bytes:
    if pretty:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
    else:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    return (text + "\n").encode("utf-8")


def _per_case_jsonl_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row, pretty=False) for row in rows)


def _per_case_tsv_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    columns = (
        "case_id",
        "row_id",
        "owner_index",
        "official_direction",
        "official_side",
        "official_ledger_reference_side",
        "blind_entity_category_disposition",
        "blind_geometry_disposition",
        "refined_interpretation",
        "view_a_checkpoint",
        "view_b_checkpoint",
        "notes",
    )
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=columns,
        delimiter="\t",
        lineterminator="\n",
        extrasaction="ignore",
    )
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)


def summarize_review(
    manifest_path: Path,
    unblinding_path: Path,
    comparison_ledger_path: Path,
    review_shard_paths: Sequence[Path],
    output_dir: Path,
    *,
    expected_case_count: int = EXPECTED_CASE_COUNT,
    expected_treatment_only_count: int = EXPECTED_TREATMENT_ONLY_COUNT,
    expected_source_only_count: int = EXPECTED_SOURCE_ONLY_COUNT,
    expected_review_shard_count: int = EXPECTED_REVIEW_SHARD_COUNT,
) -> Mapping[str, Any]:
    """Validate, unblind, reduce, and write an immutable deterministic bundle."""
    output_dir = output_dir.resolve()
    if output_dir.exists() or output_dir.is_symlink():
        _contract_error(f"immutable output directory already exists: {output_dir}")
    if expected_case_count != expected_treatment_only_count + expected_source_only_count:
        _contract_error("expected gain/loss counts do not sum to expected case count")

    # Phase one is intentionally arm blind.  No private artifact is read until
    # the manifest and completed blind reviews are fully accepted.
    manifest, case_ids, manifest_sha = _validate_blind_manifest(
        manifest_path,
        expected_case_count=expected_case_count,
    )
    reviews, review_shard_inputs = _validate_blind_reviews(
        review_shard_paths,
        reviewer_root=manifest_path.resolve().parent,
        case_ids=case_ids,
        expected_shard_count=expected_review_shard_count,
    )

    # Phase two unblinds only the already-frozen review records.
    unblinded, ledger, geometry, unblinding_sha, ledger_sha = (
        _validate_unblinding_and_ledger(
            unblinding_path,
            comparison_ledger_path,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha,
            case_ids=case_ids,
            expected_treatment_only_count=expected_treatment_only_count,
            expected_source_only_count=expected_source_only_count,
        )
    )

    per_case: list[dict[str, Any]] = []
    for case_id in case_ids:
        review = reviews[case_id]
        private_case = unblinded[case_id]
        direction = str(private_case["physical_owner_change_direction"])
        side = str(private_case["ledger_reference_side"])
        views = _mapping(private_case["views"], f"unblinding case {case_id}.views")
        entity = str(review["entity_category"])
        geometry_disposition = str(review["geometry"])
        per_case.append(
            {
                "case_id": case_id,
                "row_id": str(private_case["row_id"]),
                "owner_index": int(private_case["owner_index"]),
                "official_direction": direction,
                "official_side": (
                    "treatment_only" if direction == "gain" else "source_only"
                ),
                "official_ledger_reference_side": side,
                "blind_entity_category_disposition": entity,
                "blind_geometry_disposition": geometry_disposition,
                "refined_interpretation": refined_interpretation(
                    entity, geometry_disposition
                ),
                "view_a_checkpoint": EXPECTED_CHECKPOINT_ROLES[str(views["view_a"])],
                "view_b_checkpoint": EXPECTED_CHECKPOINT_ROLES[str(views["view_b"])],
                "notes": review["notes"],
            }
        )

    jsonl_payload = _per_case_jsonl_bytes(per_case)
    tsv_payload = _per_case_tsv_bytes(per_case)
    official_direction_counts = {
        "gain_count": expected_treatment_only_count,
        "loss_count": expected_source_only_count,
        "net_gain_minus_loss": (
            expected_treatment_only_count - expected_source_only_count
        ),
        "total_count": expected_case_count,
    }
    summary: dict[str, Any] = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "status": "complete",
        "case_count": expected_case_count,
        "checkpoint_roles": copy.deepcopy(EXPECTED_CHECKPOINT_ROLES),
        "official_geometry_ledger": {
            "comparison_ledger_sha256": ledger_sha,
            "common_owner_geometry": copy.deepcopy(geometry),
            "direction_counts": official_direction_counts,
            "preservation": (
                "verbatim common_owner_geometry copy; original comparison ledger is "
                "read-only and remains the official geometry-derived record"
            ),
        },
        "blind_review": {
            "entity_category_counts_by_official_direction": _counts_by_direction(
                per_case,
                key="blind_entity_category_disposition",
                values=ENTITY_CATEGORY_DISPOSITIONS,
            ),
            "geometry_counts_by_official_direction": _counts_by_direction(
                per_case,
                key="blind_geometry_disposition",
                values=GEOMETRY_DISPOSITIONS,
            ),
        },
        "human_refined_review": {
            "interpretation_counts_by_official_direction": _counts_by_direction(
                per_case,
                key="refined_interpretation",
                values=REFINED_INTERPRETATIONS,
            ),
            "interpretation_rule": {
                "genuine_real_owner_change": (
                    "entity/category is real_owner_change and geometry is acceptable"
                ),
                "geometry": (
                    "entity/category is real_owner_change and geometry is localization_error "
                    "or neighboring_instance_or_mixed_extent_error"
                ),
                "alias": "entity/category is category_alias_or_disagreement",
                "duplicate": "entity/category is duplicate",
                "unsupported": "entity/category is unsupported_candidate",
                "uncertain": (
                    "entity/category is uncertain, or entity/category is real_owner_change "
                    "and geometry is uncertain"
                ),
            },
        },
        "claim_boundary": {
            "ground_truth": "Reviewer dispositions do not modify or replace ground truth.",
            "official_geometry": (
                "The original geometry-derived ledger remains unchanged and is reported "
                "separately from the human-refined interpretation."
            ),
            "unmatched_predictions": (
                "Officially unmatched predictions remain unresolved unless the blind review "
                "supports a narrower interpretation; they are not automatically hallucinations."
            ),
        },
        "inputs": {
            "reviewer_manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": manifest_sha,
            },
            "review_shards": review_shard_inputs,
            "private_unblinding": {
                "path": str(unblinding_path.resolve()),
                "sha256": unblinding_sha,
            },
            "original_comparison_ledger": {
                "path": str(comparison_ledger_path.resolve()),
                "sha256": ledger_sha,
            },
            "comparison_inputs": copy.deepcopy(
                _mapping(ledger.get("inputs"), "comparison ledger.inputs")
            ),
        },
        "outputs": {
            "per_case_jsonl": {
                "path": "per-case.jsonl",
                "sha256": _sha256_bytes(jsonl_payload),
            },
            "per_case_tsv": {
                "path": "per-case.tsv",
                "sha256": _sha256_bytes(tsv_payload),
            },
        },
    }
    summary_payload = _json_bytes(summary, pretty=True)

    output_dir.mkdir(parents=True, exist_ok=False)
    _write_new(output_dir / "per-case.jsonl", jsonl_payload)
    _write_new(output_dir / "per-case.tsv", tsv_payload)
    _write_new(output_dir / "summary.json", summary_payload)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer-manifest", type=Path, required=True)
    parser.add_argument("--private-unblinding", type=Path, required=True)
    parser.add_argument("--original-comparison-ledger", type=Path, required=True)
    parser.add_argument(
        "--review-shard",
        action="append",
        type=Path,
        required=True,
        help="Completed arm-blind review shard; repeat exactly five times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = summarize_review(
            args.reviewer_manifest,
            args.private_unblinding,
            args.original_comparison_ledger,
            args.review_shard,
            args.output_dir,
        )
    except (ReviewSummaryContractError, OSError) as exc:
        raise SystemExit(f"held-out owner-churn review reduction failed: {exc}") from exc
    print(
        json.dumps(
            {
                "case_count": summary["case_count"],
                "output_dir": str(args.output_dir.resolve()),
                "status": summary["status"],
                "summary": str((args.output_dir / "summary.json").resolve()),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
