#!/usr/bin/env python3
"""Attribute paired gradient-cohort owner changes to each arm's StateBank.

This is a read-only post-hoc analyzer.  It joins the immutable StateBank
selection receipt and records with the paired Source@B16 owner ledgers.  An
owner is *selected* only when it occurs in the current arm's selection events;
an owner selected by the other arm remains non-selected for this attribution.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "gradient_cohort_treatment_owner_attribution.v2"
IOU_KEYS = ("0.30", "0.50")
EVENT_FAMILY_LABELS = {
    "treatment": "sampled_route_treatment",
    "source_preservation": "source_preservation",
}


class OwnerAttributionError(ValueError):
    """Raised when receipt, StateBank, or ledger joins are not exact."""


def _object(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OwnerAttributionError(f"{context} must be an object")
    return value


def _list(value: Any, context: str) -> list[Any]:
    if not isinstance(value, list):
        raise OwnerAttributionError(f"{context} must be a list")
    return value


def _text(value: Any, context: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value):
        raise OwnerAttributionError(f"{context} must be a non-empty string")
    return str(value)


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")), str(path))
    except json.JSONDecodeError as exc:
        raise OwnerAttributionError(f"invalid JSON: {path}") from exc


def _owner_image_id(owner_id: str, context: str) -> str:
    image_id, separator, suffix = owner_id.partition(":")
    if not separator or not image_id or not suffix:
        raise OwnerAttributionError(f"{context} is not an image-qualified owner ID")
    return image_id


def _owner_set(value: Any, *, image_id: str, context: str) -> set[str]:
    owners = {_text(item, context) for item in _list(value, context)}
    if len(owners) != len(_list(value, context)):
        raise OwnerAttributionError(f"{context} contains duplicate owner IDs")
    bad = sorted(owner for owner in owners if _owner_image_id(owner, context) != image_id)
    if bad:
        raise OwnerAttributionError(f"{context} crosses image IDs: {bad[:3]}")
    return owners


def _rate(numerator: int, denominator: int) -> dict[str, int | float | None]:
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": None if denominator == 0 else numerator / denominator,
    }


def _object_count_band(object_count: int) -> str:
    if 1 <= object_count <= 3:
        return "one_to_three_annotated_objects"
    if 4 <= object_count <= 7:
        return "four_to_seven_annotated_objects"
    if 8 <= object_count <= 15:
        return "eight_to_fifteen_annotated_objects"
    if object_count >= 16:
        return "sixteen_or_more_annotated_objects"
    raise OwnerAttributionError("annotated object count must be positive")


def _empty_counts() -> dict[str, int]:
    return {
        "complete_case_image_count": 0,
        "annotated_owner_count": 0,
        "selected_owner_count": 0,
        "selected_source_found_count": 0,
        "selected_source_missed_count": 0,
        "selected_treatment_found_count": 0,
        "selected_recovery_count": 0,
        "selected_loss_count": 0,
        "non_selected_owner_count": 0,
        "non_selected_source_found_count": 0,
        "non_selected_source_missed_count": 0,
        "non_selected_treatment_found_count": 0,
        "non_selected_recovery_count": 0,
        "non_selected_loss_count": 0,
    }


def _summary(counts: Mapping[str, int], *, selection_scope: str) -> dict[str, Any]:
    selected_delta = (
        counts["selected_treatment_found_count"]
        - counts["selected_source_found_count"]
    )
    non_selected_delta = (
        counts["non_selected_treatment_found_count"]
        - counts["non_selected_source_found_count"]
    )
    return {
        "selection_scope": selection_scope,
        "complete_case_denominator": {
            "image_count": counts["complete_case_image_count"],
            "annotated_owner_count": counts["annotated_owner_count"],
        },
        "selected_owners": {
            "count": counts["selected_owner_count"],
            "source_found": _rate(
                counts["selected_source_found_count"], counts["selected_owner_count"]
            ),
            "source_missed": _rate(
                counts["selected_source_missed_count"], counts["selected_owner_count"]
            ),
            "treatment_found": _rate(
                counts["selected_treatment_found_count"], counts["selected_owner_count"]
            ),
            "exact_recovery_among_source_missed": _rate(
                counts["selected_recovery_count"],
                counts["selected_source_missed_count"],
            ),
            "loss_among_source_found": _rate(
                counts["selected_loss_count"], counts["selected_source_found_count"]
            ),
            "net_found_owner_delta": selected_delta,
        },
        "non_selected_annotated_owners": {
            "count": counts["non_selected_owner_count"],
            "source_found": _rate(
                counts["non_selected_source_found_count"],
                counts["non_selected_owner_count"],
            ),
            "source_missed": _rate(
                counts["non_selected_source_missed_count"],
                counts["non_selected_owner_count"],
            ),
            "treatment_found": _rate(
                counts["non_selected_treatment_found_count"],
                counts["non_selected_owner_count"],
            ),
            "exact_recovery_among_source_missed": _rate(
                counts["non_selected_recovery_count"],
                counts["non_selected_source_missed_count"],
            ),
            "loss_among_source_found": _rate(
                counts["non_selected_loss_count"],
                counts["non_selected_source_found_count"],
            ),
            "net_found_owner_delta": non_selected_delta,
        },
        "net_found_owner_delta": selected_delta + non_selected_delta,
    }


def _load_state_banks(
    state_banks_root: Path, receipt: Mapping[str, Any]
) -> tuple[dict[str, dict[str, set[str]]], dict[str, set[str]]]:
    arms = _object(receipt.get("arms"), "selection receipt arms")
    selected_by_arm: dict[str, dict[str, set[str]]] = {}
    annotations_by_image: dict[str, set[str]] = {}
    for arm, raw_arm in arms.items():
        arm_data = _object(raw_arm, f"selection receipt arm {arm}")
        receipt_events = _list(arm_data.get("events"), f"selection receipt arm {arm}.events")
        receipt_by_event: dict[str, Mapping[str, Any]] = {}
        selected_by_family = {family: set() for family in EVENT_FAMILY_LABELS}
        for index, raw_event in enumerate(receipt_events):
            event = _object(raw_event, f"selection receipt arm {arm} event {index}")
            event_id = _text(event.get("event_id"), f"selection receipt arm {arm} event_id")
            image_id = _text(event.get("image_id"), f"selection receipt arm {arm} image_id")
            owner_id = _text(event.get("owner_id"), f"selection receipt arm {arm} owner_id")
            event_family = _text(
                event.get("event_family"),
                f"selection receipt arm {arm} event {event_id}.event_family",
            )
            if event_family not in selected_by_family:
                raise OwnerAttributionError(
                    f"selection receipt arm {arm} has unsupported event family {event_family}"
                )
            if event_id in receipt_by_event:
                raise OwnerAttributionError(f"selection receipt arm {arm} duplicates {event_id}")
            if _owner_image_id(owner_id, f"selection receipt arm {arm} owner_id") != image_id:
                raise OwnerAttributionError(f"selection receipt arm {arm} event joins owner to wrong image")
            receipt_by_event[event_id] = event
            selected_by_family[event_family].add(owner_id)
        receipt_images = {
            _text(image_id, f"selection receipt arm {arm}.image_ids")
            for image_id in _list(arm_data.get("image_ids"), f"selection receipt arm {arm}.image_ids")
        }
        if receipt_images != {
            _text(event.get("image_id"), f"selection receipt arm {arm} event image_id")
            for event in receipt_by_event.values()
        }:
            raise OwnerAttributionError(f"selection receipt arm {arm} image/event mismatch")
        missing_families = sorted(
            family for family, owners in selected_by_family.items() if not owners
        )
        if missing_families:
            raise OwnerAttributionError(
                f"selection receipt arm {arm} lacks required event families: {missing_families}"
            )
        records_path = state_banks_root / f"{arm}-plus-source-preservation/state-bank/records.jsonl"
        if not records_path.is_file():
            raise OwnerAttributionError(f"missing StateBank records for arm {arm}: {records_path}")
        records_by_event: dict[str, Mapping[str, Any]] = {}
        for line_number, raw_line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line:
                raise OwnerAttributionError(f"blank StateBank record at {records_path}:{line_number}")
            try:
                record = _object(json.loads(raw_line), f"StateBank record {line_number}")
            except json.JSONDecodeError as exc:
                raise OwnerAttributionError(f"invalid StateBank JSONL {records_path}:{line_number}") from exc
            event_id = _text(record.get("event_id"), f"StateBank record {line_number}.event_id")
            if event_id in records_by_event:
                raise OwnerAttributionError(f"StateBank arm {arm} duplicates {event_id}")
            records_by_event[event_id] = record
        if set(records_by_event) != set(receipt_by_event):
            raise OwnerAttributionError(f"StateBank/receipt event identity mismatch for arm {arm}")
        for event_id, event in receipt_by_event.items():
            record = records_by_event[event_id]
            image = _object(record.get("image"), f"StateBank record {event_id}.image")
            image_id = _text(image.get("image_id"), f"StateBank record {event_id}.image_id")
            if image_id != _text(event.get("image_id"), f"selection receipt event {event_id}.image_id"):
                raise OwnerAttributionError(f"StateBank/receipt image mismatch for event {event_id}")
            entities = _owner_set(
                [
                    _object(entity, f"StateBank record {event_id}.physical_entities").get("entity_id")
                    for entity in _list(record.get("physical_entities"), f"StateBank record {event_id}.physical_entities")
                ],
                image_id=image_id,
                context=f"StateBank record {event_id}.physical_entities",
            )
            if not entities:
                raise OwnerAttributionError(f"StateBank record {event_id} has no annotated owners")
            previous = annotations_by_image.setdefault(image_id, entities)
            if previous != entities:
                raise OwnerAttributionError(f"inconsistent annotated owners across StateBanks for image {image_id}")
            candidate_owners = {
                _text(
                    _object(candidate, f"StateBank record {event_id}.candidates").get("physical_owner_id"),
                    f"StateBank record {event_id}.candidate owner",
                )
                for candidate in _list(record.get("candidates"), f"StateBank record {event_id}.candidates")
            }
            owner_id = _text(event.get("owner_id"), f"selection receipt event {event_id}.owner_id")
            if owner_id not in entities or candidate_owners != {owner_id}:
                raise OwnerAttributionError(f"StateBank selected-owner mismatch for event {event_id}")
        selected_by_arm[str(arm)] = selected_by_family
    return selected_by_arm, annotations_by_image


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_gradient_cohorts(
    root_or_receipt: Path,
) -> tuple[Path, Path, dict[str, dict[str, Any]]]:
    receipt_path = (
        root_or_receipt / "receipt.json"
        if root_or_receipt.is_dir()
        else root_or_receipt
    ).resolve(strict=True)
    receipt = _read_json(receipt_path)
    if receipt.get("schema_version") != "constant_dose_breadth_gradient_cohorts.v1":
        raise OwnerAttributionError(
            f"gradient cohort receipt has unsupported schema: {receipt_path}"
        )
    cohorts: dict[str, dict[str, Any]] = {}
    for cohort_name, raw_cohort in _object(receipt.get("cohorts"), "gradient cohort receipt cohorts").items():
        cohort = _object(raw_cohort, f"gradient cohort receipt cohort {cohort_name}")
        cohort_path = Path(
            _text(cohort.get("path"), f"gradient cohort {cohort_name}.path")
        ).expanduser().resolve(strict=True)
        expected_sha256 = _text(
            cohort.get("sha256"), f"gradient cohort {cohort_name}.sha256"
        )
        expected_count = cohort.get("image_count")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int) or expected_count <= 0:
            raise OwnerAttributionError(
                f"gradient cohort {cohort_name}.image_count must be positive"
            )
        actual_sha256 = _sha256(cohort_path)
        if actual_sha256 != expected_sha256:
            raise OwnerAttributionError(
                f"gradient cohort {cohort_name} SHA-256 mismatch: {cohort_path}"
            )
        image_ids: set[str] = set()
        for line_number, raw_line in enumerate(cohort_path.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line:
                raise OwnerAttributionError(f"blank gradient cohort row at {cohort_path}:{line_number}")
            try:
                row = _object(json.loads(raw_line), f"gradient cohort row {line_number}")
            except json.JSONDecodeError as exc:
                raise OwnerAttributionError(
                    f"invalid gradient cohort JSONL {cohort_path}:{line_number}"
                ) from exc
            image_id = _text(row.get("image_id"), f"gradient cohort row {line_number}.image_id")
            if image_id in image_ids:
                raise OwnerAttributionError(
                    f"gradient cohort {cohort_name} duplicates image {image_id}"
                )
            image_ids.add(image_id)
        if len(image_ids) != expected_count:
            raise OwnerAttributionError(
                f"gradient cohort {cohort_name} image_count mismatch: expected {expected_count}, got {len(image_ids)}"
            )
        cohorts[_text(cohort_name, "gradient cohort name")] = {
            "path": cohort_path,
            "sha256": expected_sha256,
            "image_ids": image_ids,
        }
    if not cohorts:
        raise OwnerAttributionError("gradient cohort receipt has no cohorts")
    return receipt_path.parent, receipt_path, cohorts


def _validate_ledger_cohort(
    ledger: Mapping[str, Any], *, ledger_path: Path, cohort_name: str, cohort: Mapping[str, Any]
) -> None:
    candidate = _object(
        _object(ledger.get("inputs"), str(ledger_path)).get("candidate_jsonl"),
        f"{ledger_path}.inputs.candidate_jsonl",
    )
    candidate_path = Path(
        _text(candidate.get("path"), f"{ledger_path}.inputs.candidate_jsonl.path")
    ).expanduser().resolve(strict=True)
    if candidate_path != cohort["path"]:
        raise OwnerAttributionError(
            f"ledger {ledger_path} does not declare gradient cohort {cohort_name}"
        )
    if _text(candidate.get("sha256"), f"{ledger_path}.inputs.candidate_jsonl.sha256") != cohort["sha256"]:
        raise OwnerAttributionError(
            f"ledger {ledger_path} gradient cohort SHA-256 mismatch"
        )
    if candidate.get("image_count") != len(cohort["image_ids"]):
        raise OwnerAttributionError(
            f"ledger {ledger_path} gradient cohort image_count mismatch"
        )


def _ledger_treatment_arm(name: str, receipt_arms: set[str]) -> str:
    arm, marker, _ = name.partition("-seed")
    if not marker or arm not in receipt_arms:
        raise OwnerAttributionError(f"ledger treatment name is not a receipt arm: {name}")
    return arm


def _accumulate(
    counts: dict[str, int], *, annotations: set[str], selected: set[str], source: set[str], treatment: set[str]
) -> None:
    selected = selected & annotations
    non_selected = annotations - selected
    for prefix, owners in (("selected", selected), ("non_selected", non_selected)):
        source_found = owners & source
        source_missed = owners - source
        treatment_found = owners & treatment
        counts[f"{prefix}_owner_count"] += len(owners)
        counts[f"{prefix}_source_found_count"] += len(source_found)
        counts[f"{prefix}_source_missed_count"] += len(source_missed)
        counts[f"{prefix}_treatment_found_count"] += len(treatment_found)
        counts[f"{prefix}_recovery_count"] += len(source_missed & treatment)
        counts[f"{prefix}_loss_count"] += len(source_found - treatment)


def analyze_gradient_cohort_owner_attribution(
    *,
    selection_receipt: str | Path,
    state_banks_root: str | Path,
    ledger_root: str | Path,
    gradient_cohort_root_or_receipt: str | Path,
) -> dict[str, Any]:
    """Return strict attribution summaries for all declared gradient cohorts."""

    receipt_path = Path(selection_receipt).expanduser().resolve(strict=True)
    state_root = Path(state_banks_root).expanduser().resolve(strict=True)
    ledgers = Path(ledger_root).expanduser().resolve(strict=True)
    cohort_root, cohort_receipt_path, declared_cohorts = _load_gradient_cohorts(
        Path(gradient_cohort_root_or_receipt).expanduser().resolve(strict=True)
    )
    receipt = _read_json(receipt_path)
    selected_by_arm, annotations_by_image = _load_state_banks(state_root, receipt)
    receipt_arms = set(selected_by_arm)
    cohorts: dict[str, Any] = {}
    for cohort_dir in sorted(path for path in ledgers.iterdir() if path.is_dir()):
        cohort = declared_cohorts.get(cohort_dir.name)
        if cohort is None:
            raise OwnerAttributionError(
                f"ledger cohort {cohort_dir.name} is absent from the gradient cohort receipt"
            )
        treatments: dict[str, Any] = {}
        for ledger_path in sorted(cohort_dir.glob("*.json")):
            ledger = _read_json(ledger_path)
            raw_treatments = _object(ledger.get("treatments", {}), str(ledger_path))
            if not raw_treatments:
                continue
            _validate_ledger_cohort(
                ledger,
                ledger_path=ledger_path,
                cohort_name=cohort_dir.name,
                cohort=cohort,
            )
            for treatment_name, raw_treatment in raw_treatments.items():
                if treatment_name in treatments:
                    raise OwnerAttributionError(f"cohort {cohort_dir.name} duplicates treatment {treatment_name}")
                arm = _ledger_treatment_arm(str(treatment_name), receipt_arms)
                all_counts = {
                    family: {threshold: _empty_counts() for threshold in IOU_KEYS}
                    for family in (*EVENT_FAMILY_LABELS, "all_event_families")
                }
                by_band: dict[str, dict[str, dict[str, dict[str, int]]]] = {}
                seen_images: set[str] = set()
                per_image_records = _list(
                    _object(raw_treatment, str(ledger_path)).get("per_image"),
                    f"{ledger_path}.per_image",
                )
                declared_image_ids = {
                    _text(
                        _object(raw_record, f"{ledger_path} per_image {index}").get(
                            "image_id"
                        ),
                        f"{ledger_path} per_image image_id",
                    )
                    for index, raw_record in enumerate(per_image_records)
                }
                if len(declared_image_ids) != len(per_image_records):
                    raise OwnerAttributionError(
                        f"ledger {ledger_path} treatment {treatment_name} duplicates per_image image IDs"
                    )
                if declared_image_ids != cohort["image_ids"]:
                    missing = sorted(cohort["image_ids"] - declared_image_ids)
                    unexpected = sorted(declared_image_ids - cohort["image_ids"])
                    raise OwnerAttributionError(
                        f"ledger {ledger_path} treatment {treatment_name} per_image image set does not equal "
                        f"gradient cohort {cohort_dir.name}; missing={missing[:3]}, unexpected={unexpected[:3]}"
                    )
                for index, raw_record in enumerate(per_image_records):
                    record = _object(raw_record, f"{ledger_path} per_image {index}")
                    image_id = _text(record.get("image_id"), f"{ledger_path} per_image image_id")
                    if image_id in seen_images:
                        raise OwnerAttributionError(f"ledger {ledger_path} duplicates image {image_id}")
                    seen_images.add(image_id)
                    annotations = annotations_by_image.get(image_id)
                    if annotations is None:
                        raise OwnerAttributionError(f"ledger image {image_id} lacks StateBank annotation evidence")
                    band = _text(record.get("object_count_band"), f"{ledger_path} image {image_id} band")
                    object_count = record.get("annotated_object_count")
                    if isinstance(object_count, bool) or not isinstance(object_count, int):
                        raise OwnerAttributionError(f"ledger image {image_id} lacks annotated object count")
                    if object_count != len(annotations) or band != _object_count_band(object_count):
                        raise OwnerAttributionError(f"ledger image {image_id} has inconsistent object-count band")
                    comparison = _object(record.get("owner_comparison"), f"{ledger_path} image {image_id} comparison")
                    if comparison.get("eligible") is not True:
                        continue
                    matched = _object(comparison.get("by_intersection_over_union"), f"{ledger_path} image {image_id} matching")
                    for threshold in IOU_KEYS:
                        row = _object(matched.get(threshold), f"{ledger_path} image {image_id} IoU {threshold}")
                        source = _owner_set(row.get("source_owner_ids"), image_id=image_id, context="source owners")
                        treatment = _owner_set(row.get("treatment_owner_ids"), image_id=image_id, context="treatment owners")
                        if not source <= annotations or not treatment <= annotations:
                            raise OwnerAttributionError(f"ledger owner is absent from annotated StateBank image {image_id}")
                        for family, selected in (
                            *selected_by_arm[arm].items(),
                            (
                                "all_event_families",
                                set().union(*selected_by_arm[arm].values()),
                            ),
                        ):
                            counts = all_counts[family][threshold]
                            counts["complete_case_image_count"] += 1
                            counts["annotated_owner_count"] += len(annotations)
                            _accumulate(
                                counts,
                                annotations=annotations,
                                selected=selected,
                                source=source,
                                treatment=treatment,
                            )
                            band_counts = by_band.setdefault(
                                band,
                                {
                                    event_family: {
                                        key: _empty_counts() for key in IOU_KEYS
                                    }
                                    for event_family in all_counts
                                },
                            )[family][threshold]
                            band_counts["complete_case_image_count"] += 1
                            band_counts["annotated_owner_count"] += len(annotations)
                            _accumulate(
                                band_counts,
                                annotations=annotations,
                                selected=selected,
                                source=source,
                                treatment=treatment,
                            )
                treatments[str(treatment_name)] = {
                    "arm": arm,
                    "selected_owner_set_sizes_by_event_family": {
                        EVENT_FAMILY_LABELS[family]: len(selected_by_arm[arm][family])
                        for family in EVENT_FAMILY_LABELS
                    },
                    "selected_owner_set_size_all_event_families": len(
                        set().union(*selected_by_arm[arm].values())
                    ),
                    "by_intersection_over_union": {
                        threshold: {
                            "selected_owners_by_event_family": {
                                EVENT_FAMILY_LABELS[family]: _summary(
                                    all_counts[family][threshold],
                                    selection_scope=f"StateBank event_family={family}",
                                )
                                for family in EVENT_FAMILY_LABELS
                            },
                            "selected_owners_all_event_families": _summary(
                                all_counts["all_event_families"][threshold],
                                selection_scope="union of all StateBank event families",
                            ),
                        }
                        for threshold in IOU_KEYS
                    },
                    "object_count_band_breakdown": {
                        band: {
                            threshold: {
                                "selected_owners_by_event_family": {
                                    EVENT_FAMILY_LABELS[family]: _summary(
                                        by_band[band][family][threshold],
                                        selection_scope=f"StateBank event_family={family}",
                                    )
                                    for family in EVENT_FAMILY_LABELS
                                },
                                "selected_owners_all_event_families": _summary(
                                    by_band[band]["all_event_families"][threshold],
                                    selection_scope="union of all StateBank event families",
                                ),
                            }
                            for threshold in IOU_KEYS
                        }
                        for band in sorted(by_band)
                    },
                }
        if not treatments:
            raise OwnerAttributionError(f"cohort has no treatment ledgers: {cohort_dir}")
        cohorts[cohort_dir.name] = {"treatments": treatments}
    if not cohorts:
        raise OwnerAttributionError(f"ledger root has no cohort directories: {ledgers}")
    return {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "selection_receipt": str(receipt_path),
            "state_banks_root": str(state_root),
            "ledger_root": str(ledgers),
            "gradient_cohort_root": str(cohort_root),
            "gradient_cohort_receipt": str(cohort_receipt_path),
        },
        "cohorts": cohorts,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-receipt", type=Path, required=True)
    parser.add_argument("--state-banks-root", type=Path, required=True)
    parser.add_argument("--ledger-root", type=Path, required=True)
    parser.add_argument(
        "--gradient-cohort-root-or-receipt",
        type=Path,
        required=True,
        help="Gradient cohort directory containing receipt.json, or that receipt itself.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = analyze_gradient_cohort_owner_attribution(
        selection_receipt=args.selection_receipt,
        state_banks_root=args.state_banks_root,
        ledger_root=args.ledger_root,
        gradient_cohort_root_or_receipt=args.gradient_cohort_root_or_receipt,
    )
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
