#!/usr/bin/env python3
"""Make a small, exact-token StateBank subset from an assembled parent arm.

The parent arm is the authority for rows, review decisions, checkpoint identity,
and prompt identity.  This utility only selects whole event pairs and delegates
the actual StateBank write and validation to :func:`assemble_state_bank`.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file  # noqa: E402
from src.common.errors import ArtifactContractError  # noqa: E402
from src.rollout_calibration import (  # noqa: E402
    assemble_state_bank,
    load_state_bank,
    load_state_bank_manifest_binding,
)


FAMILIES = ("positive_path_imitation", "source_route_imitation")
SCHEMA_VERSION = "coordexp.rollout_calibration.state_bank_subset.v1"


class SubsetError(ValueError):
    """Raised when a parent arm or selection does not satisfy the contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SubsetError(f"invalid JSON: {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SubsetError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise SubsetError(f"JSONL row at {path}:{line_number} must be an object")
            rows.append(value)
    return rows


def _write_json(path: Path, value: Any) -> None:
    if path.exists():
        raise SubsetError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value) + b"\n")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if path.exists():
        raise SubsetError(f"immutable output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(_canonical(dict(row)) + b"\n")


def _rows_digest(rows: Sequence[Mapping[str, Any]]) -> str:
    payload = b"".join(_canonical(dict(row)) + b"\n" for row in rows)
    return hashlib.sha256(payload).hexdigest()


def _image_key(row: Mapping[str, Any]) -> tuple[int, str]:
    image = row.get("image", {})
    value = image.get("image_id") if isinstance(image, Mapping) else None
    text = str(value)
    try:
        return (0, f"{int(text):020d}")
    except ValueError:
        return (1, text)


def _image_id_text(row: Mapping[str, Any]) -> str:
    image = row.get("image", {})
    value = image.get("image_id") if isinstance(image, Mapping) else None
    return str(value)


def _event_id(row: Mapping[str, Any], field: str) -> str:
    value = row.get("event_id")
    if not isinstance(value, str) or not value:
        raise SubsetError(f"{field}.event_id must be a non-empty string")
    return value


def _family(review: Mapping[str, Any]) -> str:
    positive = review.get("positive_path_imitation_eligible") is True
    source = review.get("source_route_imitation_eligible") is True
    if positive == source:
        raise SubsetError(
            f"event {review.get('event_id', '<unknown>')} must enable exactly one imitation family"
        )
    return FAMILIES[0] if positive else FAMILIES[1]


def _balanced_candidates(
    rollout_by_id: Mapping[str, Mapping[str, Any]],
    review_by_id: Mapping[str, Mapping[str, Any]],
    family_by_id: Mapping[str, str],
    family: str,
    count: int,
) -> list[dict[str, Any]]:
    """Prefer equal parent image credit while maximizing image diversity.

    ``assemble_state_bank`` intentionally rejects unequal complete-row family
    credit between images.  Parent rows normally carry one constant weight per
    image, so selecting one row from a common-weight image cohort preserves the
    exact review rows and satisfies that invariant without reweighting.
    """

    by_weight: dict[float, dict[tuple[int, str], list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for event_id, family_value in family_by_id.items():
        if family_value != family:
            continue
        review = review_by_id[event_id]
        raw_weight = review.get("image_balanced_event_weight", 1.0)
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise SubsetError(f"event {event_id} has an invalid image_balanced_event_weight")
        image = _image_key(rollout_by_id[event_id])
        by_weight[round(float(raw_weight), 12)][image].append(copy.deepcopy(dict(rollout_by_id[event_id])))
    # A common weight cohort with one row per image is the most diverse choice.
    choices: list[tuple[int, int, float, list[dict[str, Any]]]] = []
    for weight, groups in by_weight.items():
        for per_image in range(1, count + 1):
            image_keys = sorted(key for key, rows in groups.items() if len(rows) >= per_image)
            if len(image_keys) * per_image < count or count % per_image:
                continue
            chosen: list[dict[str, Any]] = []
            for key in image_keys[: count // per_image]:
                chosen.extend(sorted(groups[key], key=lambda item: _event_id(item, "row"))[:per_image])
            choices.append((len(image_keys[: count // per_image]), -per_image, weight, chosen))
    if choices:
        return max(choices, key=lambda item: (item[0], item[1], -item[2]))[3]
    rows = [rollout_by_id[event_id] for event_id, value in family_by_id.items() if value == family]
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in sorted(rows, key=lambda item: (_image_key(item), _event_id(item, "row"))):
        groups[_image_key(row)].append(copy.deepcopy(dict(row)))
    selected: list[dict[str, Any]] = []
    image_keys = sorted(groups)
    while len(selected) < count:
        progressed = False
        for key in image_keys:
            if groups[key]:
                selected.append(groups[key].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise SubsetError(f"requested {count} events but only {len(selected)} candidates exist")
    return selected


def select_event_pairs(
    rollout_rows: Sequence[Mapping[str, Any]],
    review_rows: Sequence[Mapping[str, Any]],
    *,
    event_ids: Sequence[str] | None = None,
    family_counts: Mapping[str, int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Select whole rollout/review pairs, deterministically and image-diversely."""

    if (event_ids is None) == (family_counts is None):
        raise SubsetError("provide exactly one of event_ids or family_counts")
    rollout_by_id = {_event_id(row, "rollout row"): row for row in rollout_rows}
    review_by_id = {_event_id(row, "review row"): row for row in review_rows}
    if len(rollout_by_id) != len(rollout_rows) or len(review_by_id) != len(review_rows):
        raise SubsetError("rollout and review event IDs must be unique")
    if set(rollout_by_id) != set(review_by_id):
        raise SubsetError("rollout and review event IDs must match exactly")
    families = {event_id: _family(review) for event_id, review in review_by_id.items()}
    if event_ids is not None:
        selected_ids = sorted(str(item) for item in event_ids)
        if not selected_ids:
            raise SubsetError("explicit event IDs must not be empty")
        if len(set(selected_ids)) != len(selected_ids):
            raise SubsetError("explicit event IDs must be unique")
        unknown = sorted(set(selected_ids) - set(rollout_by_id))
        if unknown:
            raise SubsetError(f"unknown event IDs: {unknown}")
        mode: dict[str, Any] = {"kind": "explicit_event_ids"}
    else:
        requested = {family: 0 for family in FAMILIES}
        for family, value in (family_counts or {}).items():
            if family not in FAMILIES:
                raise SubsetError(f"unknown family: {family}")
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise SubsetError(f"family count for {family} must be a non-negative integer")
            requested[family] = value
        if not any(requested.values()):
            raise SubsetError("family counts must request at least one event")
        chosen: list[dict[str, Any]] = []
        for family in FAMILIES:
            chosen.extend(
                _balanced_candidates(rollout_by_id, review_by_id, families, family, requested[family])
                if requested[family]
                else []
            )
        selected_ids = sorted(_event_id(row, "selected row") for row in chosen)
        mode = {"kind": "family_counts", "requested": requested}
    selected_rollouts = [copy.deepcopy(dict(rollout_by_id[event_id])) for event_id in selected_ids]
    selected_reviews = [copy.deepcopy(dict(review_by_id[event_id])) for event_id in selected_ids]
    counts = {family: sum(families[event_id] == family for event_id in selected_ids) for family in FAMILIES}
    mode["selected"] = counts
    return selected_rollouts, selected_reviews, {
        "mode": mode,
        "event_ids": selected_ids,
        "family_counts": counts,
        "image_ids": sorted({_image_id_text(row) for row in selected_rollouts}),
        "rollout_rows_sha256": _rows_digest(selected_rollouts),
        "review_rows_sha256": _rows_digest(selected_reviews),
    }


def build_subset(
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    event_ids: Sequence[str] | None = None,
    family_counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Build, validate, and receipt an immutable subset output directory."""

    manifest_path = Path(manifest_path).expanduser().resolve(strict=True)
    output_dir = Path(output_dir).expanduser().resolve()
    if output_dir.exists():
        raise SubsetError(f"output already exists: {output_dir}")
    if manifest_path.name != "manifest.json" or manifest_path.parent.name != "state-bank":
        raise SubsetError("manifest must be the canonical state-bank/manifest.json")
    arm_root = manifest_path.parent.parent
    pre_root = arm_root / "pre-state-bank"
    rollout_path = pre_root / "rollout_rows.jsonl"
    review_path = pre_root / "review_rows.jsonl"
    for path in (rollout_path, review_path):
        if not path.is_file():
            raise SubsetError(f"parent arm is missing required artifact: {path}")
    receipt_path = arm_root / "assembly-receipt.json"
    if not receipt_path.is_file():
        receipt_path = arm_root / "assembly-validation-receipt.json"
    if not receipt_path.is_file():
        raise SubsetError(f"parent arm is missing assembly receipt: {arm_root}")
    parent_manifest = _read_json(manifest_path)
    parent_receipt = _read_json(receipt_path)
    if not isinstance(parent_receipt, Mapping) or parent_receipt.get("status") not in {"assembled", "validated"}:
        raise SubsetError("parent assembly receipt is not validated/assembled")
    if parent_receipt.get("bank_id") and parent_receipt["bank_id"] != parent_manifest.get("bank_id"):
        raise SubsetError("assembly receipt bank_id does not match manifest")
    binding = load_state_bank_manifest_binding(manifest_path)
    parent_loaded = load_state_bank(
        manifest_path,
        expected_source_checkpoint=binding.source_checkpoint,
        expected_prompt_identity_sha256=binding.prompt_identity_sha256,
    )
    rollouts = _read_jsonl(rollout_path)
    reviews = _read_jsonl(review_path)
    parent_record_ids = {event.event_id for event in parent_loaded.records}
    if {_event_id(row, "rollout row") for row in rollouts} != parent_record_ids:
        raise SubsetError("parent pre-state-bank rollout IDs do not match canonical records")
    if {_event_id(row, "review row") for row in reviews} != parent_record_ids:
        raise SubsetError("parent pre-state-bank review IDs do not match canonical records")
    parent_pre_counts = {
        family: sum(_family(review) == family for review in reviews) for family in FAMILIES
    }
    if parent_pre_counts != dict(parent_loaded.manifest.event_family_counts):
        raise SubsetError("parent pre-state-bank family counts do not match canonical manifest")
    selected_rollouts, selected_reviews, selection = select_event_pairs(
        rollouts, reviews, event_ids=event_ids, family_counts=family_counts
    )
    source_artifacts = parent_manifest.get("source_artifacts")
    if not isinstance(source_artifacts, list):
        raise SubsetError("parent manifest source_artifacts must be a list")
    _write_jsonl(output_dir / "pre-state-bank" / "rollout_rows.jsonl", selected_rollouts)
    _write_jsonl(output_dir / "pre-state-bank" / "review_rows.jsonl", selected_reviews)
    try:
        manifest = assemble_state_bank(
            output_dir=output_dir / "state-bank",
            rollout_rows=selected_rollouts,
            review_rows=selected_reviews,
            source_checkpoint=binding.source_checkpoint,
            prompt_identity_sha256=binding.prompt_identity_sha256,
            source_artifacts=source_artifacts,
        )
    except ArtifactContractError as exc:
        raise SubsetError(f"subset assembly rejected by StateBank contract: {exc.code}: {exc}") from exc
    loaded = load_state_bank(
        output_dir / "state-bank" / "manifest.json",
        expected_source_checkpoint=binding.source_checkpoint,
        expected_prompt_identity_sha256=binding.prompt_identity_sha256,
    )
    loaded_ids = [event.event_id for event in loaded.records]
    if loaded_ids != selection["event_ids"]:
        raise SubsetError("validated StateBank event IDs differ from selected IDs")
    loaded_family_counts = {
        family: int(loaded.manifest.event_family_counts.get(family, 0)) for family in FAMILIES
    }
    if loaded_family_counts != selection["family_counts"]:
        raise SubsetError("validated StateBank family counts differ from selection")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "validated",
        "parent_manifest_path": str(manifest_path),
        "parent_manifest_sha256": sha256_file(manifest_path),
        "parent_assembly_receipt_path": str(receipt_path),
        "parent_assembly_receipt_sha256": sha256_file(receipt_path),
        "parent_pre_state_bank": {
            "rollout_rows_sha256": sha256_file(rollout_path),
            "review_rows_sha256": sha256_file(review_path),
        },
        "parent_bank_id": parent_loaded.manifest.bank_id,
        "source_checkpoint_id": binding.source_checkpoint_id,
        "prompt_identity_sha256": binding.prompt_identity_sha256,
        "selection": selection,
        "state_bank_manifest": manifest.to_artifact_dict(),
        "state_bank_manifest_path": str((output_dir / "state-bank" / "manifest.json").resolve()),
        "state_bank_validation_receipt": loaded.validation_receipt.to_artifact_dict(),
    }
    _write_json(output_dir / "selection-receipt.json", receipt)
    return receipt


def _parse_family_counts(values: Sequence[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise SubsetError(f"family count must use FAMILY=COUNT: {value}")
        family, raw_count = value.split("=", 1)
        try:
            count = int(raw_count)
        except ValueError as exc:
            raise SubsetError(f"invalid family count: {value}") from exc
        if family in counts:
            raise SubsetError(f"duplicate family count: {family}")
        counts[family] = count
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", "--reference-manifest", dest="manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--event-id", action="append", default=[])
    parser.add_argument("--family-count", action="append", default=[], metavar="FAMILY=COUNT")
    args = parser.parse_args(argv)
    if args.event_id and args.family_count:
        parser.error("use either --event-id or --family-count, not both")
    if not args.event_id and not args.family_count:
        parser.error("provide at least one --event-id or --family-count")
    try:
        build_subset(
            args.manifest,
            args.output_dir,
            event_ids=args.event_id or None,
            family_counts=_parse_family_counts(args.family_count) if args.family_count else None,
        )
    except SubsetError as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
