#!/usr/bin/env python3
"""Build immutable matched-dose Source-preservation repeat controls.

This experiment-local builder repeats whole accepted events from the canonical
``source_preservation_only`` arm.  It does not retokenize, reweight, relabel,
or regenerate anything: each copy differs from its Source event only by a
unique ``event_id`` shared by the rollout and review inputs.  The StateBank
core remains the only writer of canonical records and manifests.

The default invocation materializes two controls from the 24-event Source arm:

* repeat factor 2: 48 events / 6 effective-batch-size-8 updates;
* repeat factor 3: 72 events / 9 effective-batch-size-8 updates.

Every new StateBank manifest retains the parent source artifacts and adds
hashes for the parent manifest, parent records, parent pre-StateBank rollout
and review files, and the immutable repetition policy written beside it.
"""

from __future__ import annotations

import argparse
import copy
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.common.errors import ArtifactContractError  # noqa: E402
from src.config.fingerprint import sha256_file  # noqa: E402
from src.rollout_calibration import (  # noqa: E402
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
)


SCHEMA_VERSION = "source_preservation_matched_dose_control_state_banks.v1"
REPEAT_FACTORS = (2, 3)
SOURCE_ARM_NAME = "source_preservation_only"
DEFAULT_REFERENCE_MANIFEST = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-physical-owner-duplication-causality-and-training-treatment/"
    "state-banks-screen-24-v6/state-banks/source_preservation_only/"
    "state-bank/manifest.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-physical-owner-duplication-causality-and-training-treatment/"
    "state-banks-screen-24-dose-control-v1"
)


class MatchedDoseControlError(ValueError):
    """Raised when a Source arm cannot safely become a repeat control."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _clone(value: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(value))


def _event_id(row: Mapping[str, Any], *, field: str) -> str:
    event_id = row.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise MatchedDoseControlError(f"{field}.event_id must be a non-empty string")
    return event_id


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    raise MatchedDoseControlError(
                        f"blank JSONL row at {path}:{line_number}"
                    )
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MatchedDoseControlError(
                        f"invalid JSONL at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise MatchedDoseControlError(
                        f"JSONL row at {path}:{line_number} must be an object"
                    )
                rows.append(value)
    except OSError as exc:
        raise MatchedDoseControlError(f"cannot read {path}: {exc}") from exc
    if not rows:
        raise MatchedDoseControlError(f"required input is empty: {path}")
    return rows


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise MatchedDoseControlError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(dict(value)) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise MatchedDoseControlError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        for row in rows:
            handle.write(_canonical(dict(row)) + b"\n")


def _event_id_for_copy(
    *, source_event_id: str, repeat_factor: int, repeat_index: int
) -> str:
    return (
        f"source-preservation-repeat-factor-{repeat_factor}-"
        f"copy-{repeat_index:02d}--{source_event_id}"
    )


def _source_artifacts(
    *,
    parent_manifest: Mapping[str, Any],
    parent_manifest_sha256: str,
    parent_records_sha256: str,
    parent_rollout_rows_sha256: str,
    parent_review_rows_sha256: str,
    repetition_policy_sha256: str,
) -> list[dict[str, str]]:
    raw_artifacts = parent_manifest.get("source_artifacts")
    if not isinstance(raw_artifacts, list):
        raise MatchedDoseControlError("parent manifest.source_artifacts must be a list")
    artifacts: list[dict[str, str]] = []
    for index, item in enumerate(raw_artifacts):
        if not isinstance(item, Mapping):
            raise MatchedDoseControlError(
                f"parent manifest.source_artifacts[{index}] must be an object"
            )
        artifact_id = item.get("artifact_id")
        digest = item.get("sha256")
        if not isinstance(artifact_id, str) or not artifact_id:
            raise MatchedDoseControlError(
                f"parent manifest.source_artifacts[{index}].artifact_id is invalid"
            )
        if not isinstance(digest, str) or len(digest) != 64:
            raise MatchedDoseControlError(
                f"parent manifest.source_artifacts[{index}].sha256 is invalid"
            )
        artifacts.append({"artifact_id": artifact_id, "sha256": digest})
    artifacts.extend(
        (
            {
                "artifact_id": "source-preservation-control-parent-manifest",
                "sha256": parent_manifest_sha256,
            },
            {
                "artifact_id": "source-preservation-control-parent-records",
                "sha256": parent_records_sha256,
            },
            {
                "artifact_id": "source-preservation-control-parent-pre-state-bank-rollout-rows",
                "sha256": parent_rollout_rows_sha256,
            },
            {
                "artifact_id": "source-preservation-control-parent-pre-state-bank-review-rows",
                "sha256": parent_review_rows_sha256,
            },
            {
                "artifact_id": "source-preservation-control-repetition-policy",
                "sha256": repetition_policy_sha256,
            },
        )
    )
    identifiers = [item["artifact_id"] for item in artifacts]
    if len(identifiers) != len(set(identifiers)):
        raise MatchedDoseControlError(
            "parent source_artifacts collide with matched-dose provenance identifiers"
        )
    return artifacts


def _validate_parent_input(
    reference_manifest: Path,
) -> tuple[
    Mapping[str, Any],
    Any,
    list[dict[str, Any]],
    list[dict[str, Any]],
    Path,
    Path,
]:
    if reference_manifest.name != "manifest.json" or reference_manifest.parent.name != "state-bank":
        raise MatchedDoseControlError(
            "reference manifest must use the canonical state-bank/manifest.json location"
        )
    if not reference_manifest.is_file():
        raise MatchedDoseControlError(f"reference manifest does not exist: {reference_manifest}")
    arm_root = reference_manifest.parent.parent
    if arm_root.name != SOURCE_ARM_NAME:
        raise MatchedDoseControlError(
            f"reference manifest must belong to {SOURCE_ARM_NAME!r}, not {arm_root.name!r}"
        )
    rollout_path = arm_root / "pre-state-bank" / "rollout_rows.jsonl"
    review_path = arm_root / "pre-state-bank" / "review_rows.jsonl"
    if not rollout_path.is_file() or not review_path.is_file():
        raise MatchedDoseControlError(
            "reference Source arm must contain canonical pre-state-bank rollout_rows.jsonl "
            "and review_rows.jsonl"
        )
    try:
        binding = load_state_bank_manifest_binding(reference_manifest)
        loaded = load_state_bank(
            reference_manifest,
            expected_source_checkpoint=binding.source_checkpoint,
            expected_prompt_identity_sha256=binding.prompt_identity_sha256,
        )
    except ArtifactContractError as exc:
        raise MatchedDoseControlError(
            f"reference StateBank failed validation: {exc.code}: {exc}"
        ) from exc
    parent_manifest = json.loads(reference_manifest.read_text(encoding="utf-8"))
    if not isinstance(parent_manifest, Mapping):
        raise MatchedDoseControlError("reference manifest must be an object")
    rollouts = _read_jsonl(rollout_path)
    reviews = _read_jsonl(review_path)
    rollout_ids = [_event_id(row, field="parent rollout row") for row in rollouts]
    review_ids = [_event_id(row, field="parent review row") for row in reviews]
    if len(rollout_ids) != len(set(rollout_ids)):
        raise MatchedDoseControlError("parent pre-state-bank rollout event IDs are not unique")
    if len(review_ids) != len(set(review_ids)):
        raise MatchedDoseControlError("parent pre-state-bank review event IDs are not unique")
    loaded_ids = {event.event_id for event in loaded.records}
    if set(rollout_ids) != loaded_ids or set(review_ids) != loaded_ids:
        raise MatchedDoseControlError(
            "parent pre-state-bank event IDs must exactly equal validated StateBank records"
        )
    review_by_id = {event_id: row for event_id, row in zip(review_ids, reviews)}
    for event_id in rollout_ids:
        review = review_by_id[event_id]
        if review.get("admission_status") != "accepted":
            raise MatchedDoseControlError(
                f"parent Source event {event_id!r} must be accepted"
            )
        if review.get("source_route_imitation_eligible") is not True:
            raise MatchedDoseControlError(
                f"parent event {event_id!r} is not Source-route imitation eligible"
            )
        if review.get("positive_path_imitation_eligible") is True:
            raise MatchedDoseControlError(
                f"parent event {event_id!r} unexpectedly enables positive-path imitation"
            )
    return parent_manifest, loaded, rollouts, reviews, rollout_path, review_path


def _repeated_rows(
    *,
    parent_rollouts: Sequence[Mapping[str, Any]],
    parent_reviews: Sequence[Mapping[str, Any]],
    repeat_factor: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    review_by_id = {
        _event_id(row, field="parent review row"): row for row in parent_reviews
    }
    rollouts: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    mapping: list[dict[str, Any]] = []
    for repeat_index in range(1, repeat_factor + 1):
        for parent_rollout in parent_rollouts:
            source_event_id = _event_id(parent_rollout, field="parent rollout row")
            new_event_id = _event_id_for_copy(
                source_event_id=source_event_id,
                repeat_factor=repeat_factor,
                repeat_index=repeat_index,
            )
            rollout = _clone(parent_rollout)
            review = _clone(review_by_id[source_event_id])
            rollout["event_id"] = new_event_id
            review["event_id"] = new_event_id
            rollouts.append(rollout)
            reviews.append(review)
            mapping.append(
                {
                    "repeat_factor": repeat_factor,
                    "repeat_index": repeat_index,
                    "source_event_id": source_event_id,
                    "event_id": new_event_id,
                }
            )
    event_ids = [row["event_id"] for row in rollouts]
    if len(event_ids) != len(set(event_ids)):
        raise MatchedDoseControlError("generated repeat-control event IDs are not unique")
    if event_ids != [row["event_id"] for row in reviews]:
        raise MatchedDoseControlError("generated rollout and review event IDs differ")
    return rollouts, reviews, mapping


def _same_except_event_id(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_without_id = _clone(left)
    right_without_id = _clone(right)
    left_without_id.pop("event_id", None)
    right_without_id.pop("event_id", None)
    return _canonical(left_without_id) == _canonical(right_without_id)


def _validate_repeated_records(
    *,
    parent_loaded: Any,
    repeated_loaded: Any,
    mapping: Sequence[Mapping[str, Any]],
) -> None:
    parent_by_id = {
        record.event_id: record.to_artifact_dict() for record in parent_loaded.records
    }
    repeated_by_id = {
        record.event_id: record.to_artifact_dict() for record in repeated_loaded.records
    }
    if len(repeated_by_id) != len(mapping):
        raise MatchedDoseControlError("validated repeat StateBank has an unexpected record count")
    for item in mapping:
        source_event_id = item["source_event_id"]
        event_id = item["event_id"]
        if source_event_id not in parent_by_id or event_id not in repeated_by_id:
            raise MatchedDoseControlError(
                "validated repeat StateBank lost a parent or generated event identifier"
            )
        if not _same_except_event_id(parent_by_id[source_event_id], repeated_by_id[event_id]):
            raise MatchedDoseControlError(
                "repeated StateBank record changed semantics beyond event_id: "
                f"{source_event_id!r} -> {event_id!r}"
            )


def build_matched_dose_source_preservation_controls(
    *,
    reference_manifest: str | Path = DEFAULT_REFERENCE_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    """Materialize the immutable 2x and 3x Source-preservation controls."""

    reference_manifest = Path(reference_manifest).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise MatchedDoseControlError(f"immutable output already exists: {output_dir}")
    (
        parent_manifest,
        parent_loaded,
        parent_rollouts,
        parent_reviews,
        parent_rollout_path,
        parent_review_path,
    ) = _validate_parent_input(reference_manifest)
    source_event_count = len(parent_rollouts)
    if source_event_count != parent_loaded.manifest.record_count:
        raise MatchedDoseControlError("parent source record count disagrees with pre-state-bank rows")
    parent_records_path = reference_manifest.parent / "records.jsonl"
    if not parent_records_path.is_file():
        raise MatchedDoseControlError(f"reference records do not exist: {parent_records_path}")
    parent_manifest_sha256 = sha256_file(reference_manifest)
    parent_records_sha256 = sha256_file(parent_records_path)
    if parent_records_sha256 != parent_loaded.manifest.records_sha256:
        raise MatchedDoseControlError("reference records SHA-256 disagrees with its manifest")
    parent_rollout_rows_sha256 = sha256_file(parent_rollout_path)
    parent_review_rows_sha256 = sha256_file(parent_review_path)
    policy = {
        "schema_version": SCHEMA_VERSION,
        "control_name": "matched_dose_source_preservation_repeated_exposure",
        "source_arm_name": SOURCE_ARM_NAME,
        "source_state_bank": {
            "manifest_path": str(reference_manifest),
            "manifest_sha256": parent_manifest_sha256,
            "records_path": str(parent_records_path),
            "records_sha256": parent_records_sha256,
            "record_count": source_event_count,
            "pre_state_bank_rollout_rows_path": str(parent_rollout_path),
            "pre_state_bank_rollout_rows_sha256": parent_rollout_rows_sha256,
            "pre_state_bank_review_rows_path": str(parent_review_path),
            "pre_state_bank_review_rows_sha256": parent_review_rows_sha256,
        },
        "repetition_policy": {
            "kind": "whole_accepted_event_repetition",
            "repeat_factors": list(REPEAT_FACTORS),
            "copy_order": "repeat-index-major_then_parent-pre-state-bank-order",
            "event_id_template": (
                "source-preservation-repeat-factor-{repeat_factor}-"
                "copy-{repeat_index:02d}--{source_event_id}"
            ),
            "mutated_fields": ["event_id"],
            "preserved_semantics": [
                "image identity",
                "executed prompt token identifiers",
                "prefix token identifiers",
                "candidate target token identifiers",
                "review decisions",
                "image-balanced event weight",
            ],
            "expected_event_counts": {
                str(factor): source_event_count * factor for factor in REPEAT_FACTORS
            },
        },
    }
    output_dir.mkdir(parents=True)
    policy_path = output_dir / "repetition-policy.json"
    _write_json(policy_path, policy)
    source_artifacts = _source_artifacts(
        parent_manifest=parent_manifest,
        parent_manifest_sha256=parent_manifest_sha256,
        parent_records_sha256=parent_records_sha256,
        parent_rollout_rows_sha256=parent_rollout_rows_sha256,
        parent_review_rows_sha256=parent_review_rows_sha256,
        repetition_policy_sha256=sha256_file(policy_path),
    )
    bank_receipts: dict[str, Any] = {}
    for repeat_factor in REPEAT_FACTORS:
        rollouts, reviews, mapping = _repeated_rows(
            parent_rollouts=parent_rollouts,
            parent_reviews=parent_reviews,
            repeat_factor=repeat_factor,
        )
        expected_count = source_event_count * repeat_factor
        if len(rollouts) != expected_count or len(reviews) != expected_count:
            raise MatchedDoseControlError(
                f"repeat factor {repeat_factor} did not produce {expected_count} event pairs"
            )
        bank_name = f"source_preservation_repeat_factor_{repeat_factor}"
        bank_root = output_dir / "state-banks" / bank_name
        _write_jsonl(bank_root / "pre-state-bank" / "rollout_rows.jsonl", rollouts)
        _write_jsonl(bank_root / "pre-state-bank" / "review_rows.jsonl", reviews)
        try:
            manifest = assemble_state_bank(
                output_dir=bank_root / "state-bank",
                rollout_rows=rollouts,
                review_rows=reviews,
                source_checkpoint=parent_loaded.manifest.source_checkpoint,
                prompt_identity_sha256=parent_loaded.manifest.prompt_identity_sha256,
                source_artifacts=source_artifacts,
            )
            loaded = load_state_bank(
                bank_root / "state-bank" / "manifest.json",
                expected_source_checkpoint=parent_loaded.manifest.source_checkpoint,
                expected_prompt_identity_sha256=parent_loaded.manifest.prompt_identity_sha256,
            )
        except ArtifactContractError as exc:
            raise MatchedDoseControlError(
                f"repeat-factor-{repeat_factor} StateBank failed validation: {exc.code}: {exc}"
            ) from exc
        if loaded.manifest.record_count != expected_count:
            raise MatchedDoseControlError(
                f"repeat-factor-{repeat_factor} StateBank record count is incorrect"
            )
        if dict(loaded.manifest.event_family_counts) != {
            "source_route_imitation": expected_count
        }:
            raise MatchedDoseControlError(
                f"repeat-factor-{repeat_factor} StateBank changed the event family"
            )
        _validate_repeated_records(
            parent_loaded=parent_loaded,
            repeated_loaded=loaded,
            mapping=mapping,
        )
        bank_receipts[bank_name] = {
            "repeat_factor": repeat_factor,
            "expected_event_count": expected_count,
            "rollout_rows_sha256": sha256_file(
                bank_root / "pre-state-bank" / "rollout_rows.jsonl"
            ),
            "review_rows_sha256": sha256_file(
                bank_root / "pre-state-bank" / "review_rows.jsonl"
            ),
            "event_id_mapping": mapping,
            "state_bank_manifest_path": str(
                (bank_root / "state-bank" / "manifest.json").resolve()
            ),
            "state_bank_manifest": manifest.to_artifact_dict(),
            "state_bank_validation_receipt": loaded.validation_receipt.to_artifact_dict(),
        }
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "repetition_policy_path": str(policy_path.resolve()),
        "repetition_policy_sha256": sha256_file(policy_path),
        "source_state_bank": policy["source_state_bank"],
        "banks": bank_receipts,
    }
    _write_json(output_dir / "materialization-receipt.json", receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=DEFAULT_REFERENCE_MANIFEST,
        help="canonical source_preservation_only state-bank manifest",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="new immutable root for the 2x and 3x matched-dose controls",
    )
    args = parser.parse_args(argv)
    try:
        receipt = build_matched_dose_source_preservation_controls(
            reference_manifest=args.reference_manifest,
            output_dir=args.output_dir,
        )
    except MatchedDoseControlError as exc:
        parser.error(str(exc))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
