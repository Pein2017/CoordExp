#!/usr/bin/env python3
"""Analyze the bounded image-2299 successor-transition sampling panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


IMAGE_ID = "2299"
EMITTED_PERSON_RANKS = {
    "0003": 3,
    "0004": 2,
    "0006": 4,
    "0012": 14,
}
PARENT_PERSON_RANKS = {0, 1}
COORD_TOKEN_PATTERN = re.compile(r"^<\|coord_(\d+)\|>$")
EXPECTED_GENERATION_POLICY = {
    "max_new_tokens": 9,
    "mode": "sampled",
    "repetition_penalty": 1,
    "temperature": 0.4,
    "top_p": 0.95,
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def coord_token_value(token: str) -> int:
    match = COORD_TOKEN_PATTERN.match(token)
    if match is None:
        raise ValueError(f"invalid coordinate token: {token!r}")
    value = int(match.group(1))
    if not 0 <= value <= 999:
        raise ValueError(f"coordinate outside 0..999: {value}")
    return value


def coord_bins_to_pixel_box(
    coord_bins: Sequence[int], *, width: int, height: int
) -> list[int]:
    if len(coord_bins) != 4:
        raise ValueError("a box requires four coordinate bins")
    scales = (width, height, width, height)
    return [int(round(int(value) * scale / 1000.0)) for value, scale in zip(coord_bins, scales, strict=True)]


def intersection_over_union(first: Sequence[float], second: Sequence[float]) -> float:
    left = max(float(first[0]), float(second[0]))
    top = max(float(first[1]), float(second[1]))
    right = min(float(first[2]), float(second[2]))
    bottom = min(float(first[3]), float(second[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(
        0.0, float(first[3]) - float(first[1])
    )
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(
        0.0, float(second[3]) - float(second[1])
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def load_image_record(path: Path) -> tuple[dict[str, Any], str]:
    with path.open("rb") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            record = json.loads(raw_line)
            if str(record.get("image_id")) == IMAGE_ID:
                return record, sha256_bytes(raw_line)
    raise ValueError(f"image {IMAGE_ID} is absent from {path}")


def build_relabel_objects(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    width = int(record["width"])
    height = int(record["height"])
    category_ranks: Counter[str] = Counter()
    objects: list[dict[str, Any]] = []
    for serialized_index, source in enumerate(record["objects"]):
        description = str(source["desc"])
        category_rank = int(category_ranks[description])
        category_ranks[description] += 1
        coord_bins = [coord_token_value(str(token)) for token in source["bbox_2d"]]
        objects.append(
            {
                "serialized_index": serialized_index,
                "description": description,
                "category_rank": category_rank,
                "annotation_id": source.get("coco_ann_id"),
                "coord_bins": coord_bins,
                "pixel_box": coord_bins_to_pixel_box(
                    coord_bins, width=width, height=height
                ),
            }
        )
    return objects


def match_prediction(
    *,
    description: str,
    pixel_box: Sequence[float],
    relabel_objects: Sequence[Mapping[str, Any]],
    minimum_iou: float = 0.5,
    minimum_margin: float = 0.05,
) -> dict[str, Any]:
    candidates = [
        (intersection_over_union(pixel_box, obj["pixel_box"]), obj)
        for obj in relabel_objects
        if obj["description"] == description
    ]
    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if not candidates:
        return {
            "matched": False,
            "best_iou": 0.0,
            "second_iou": 0.0,
            "iou_margin": 0.0,
            "reason": "no_same_description_relabel_object",
        }
    best_iou, best_object = candidates[0]
    second_iou = candidates[1][0] if len(candidates) > 1 else 0.0
    margin = best_iou - second_iou
    if best_iou < minimum_iou:
        reason = "best_iou_below_threshold"
    elif margin < minimum_margin:
        reason = "top_match_margin_below_threshold"
    else:
        return {
            "matched": True,
            "best_iou": best_iou,
            "second_iou": second_iou,
            "iou_margin": margin,
            "reason": "unique_match",
            "matched_description": best_object["description"],
            "matched_category_rank": best_object["category_rank"],
            "matched_annotation_id": best_object["annotation_id"],
            "matched_pixel_box": best_object["pixel_box"],
        }
    return {
        "matched": False,
        "best_iou": best_iou,
        "second_iou": second_iou,
        "iou_margin": margin,
        "reason": reason,
        "best_candidate_description": best_object["description"],
        "best_candidate_category_rank": best_object["category_rank"],
        "best_candidate_annotation_id": best_object["annotation_id"],
    }


def validate_receipt(receipt: Mapping[str, Any], expected_seeds: Sequence[int]) -> None:
    seeds = [int(seed) for seed in receipt["sampling_seeds"]]
    if seeds != list(expected_seeds):
        raise ValueError(f"seed vector mismatch: {seeds} != {list(expected_seeds)}")
    contract = receipt["execution_contract"]
    if int(contract["physical_batch_size"]) != 1:
        raise ValueError("physical batch size must be one")
    if contract["sampling_attestation_mode"] != "local-direct-sampling-context":
        raise ValueError("unexpected sampling context")
    policy = contract["decode_generation_policy"]
    for key, expected in EXPECTED_GENERATION_POLICY.items():
        if policy.get(key) != expected:
            raise ValueError(f"generation policy mismatch for {key}: {policy.get(key)!r}")
    prompt = receipt["prompt"]
    if prompt.get("prompt_construction_mode") != "exact_donor_token_ids":
        raise ValueError("receipt did not execute exact donor token identifiers")
    if len(receipt["calls"]) != len(expected_seeds):
        raise ValueError("receipt call count does not match seed count")


def classify_call(
    *,
    bundle: Mapping[str, Any],
    bundle_path: Path,
    owner_id: str,
    emitted_rank: int,
    variant_label: str,
    relabel_objects: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    seed = int(bundle["sampling_seed"])
    record: dict[str, Any] = {
        "owner_id": owner_id,
        "emitted_person_rank": emitted_rank,
        "variant_label": variant_label,
        "sampling_seed": seed,
        "bundle_path": str(bundle_path),
        "immediate_self_repeat": False,
        "earlier_parent_repeat": False,
        "backward_uncovered_person": False,
        "forward_uncovered_person": False,
    }
    prompt = bundle.get("prompt", {})
    if prompt.get("prompt_construction_mode") != "exact_donor_token_ids":
        raise ValueError(f"call does not use exact donor token prompt: {bundle_path}")
    runtime = bundle.get("runtime", {})
    if runtime.get("physical_batch_size") != 1:
        raise ValueError(f"call does not use physical batch size one: {bundle_path}")
    if runtime.get("sampling_attestation_mode") != "local-direct-sampling-context":
        raise ValueError(f"call does not use local sampling context: {bundle_path}")

    predictions = bundle.get("parse_result", {}).get("predictions", [])
    if len(predictions) != 1:
        dropped = bundle.get("parse_result", {}).get("dropped_predictions", [])
        generated_ids = bundle.get("raw_generated_token_ids", [])
        if dropped:
            outcome = "malformed"
        elif generated_ids and int(generated_ids[0]) == 151645:
            outcome = "terminal"
        else:
            outcome = "invalid"
        record.update(
            {
                "outcome": outcome,
                "raw_generated_text": bundle.get("raw_generated_text", ""),
                "parser_status": bundle.get("parser_status"),
            }
        )
        return record

    prediction = predictions[0]
    description = str(prediction["description"])
    pixel_box = [int(value) for value in prediction["bbox"]]
    match = match_prediction(
        description=description,
        pixel_box=pixel_box,
        relabel_objects=relabel_objects,
    )
    record.update(
        {
            "predicted_description": description,
            "predicted_pixel_box": pixel_box,
            "match": match,
        }
    )
    if not match["matched"]:
        record["outcome"] = "unresolved"
        return record

    matched_description = str(match["matched_description"])
    matched_rank = int(match["matched_category_rank"])
    record["matched_description"] = matched_description
    record["matched_category_rank"] = matched_rank
    if matched_description == "person":
        record["outcome"] = "matched_person"
        record["immediate_self_repeat"] = matched_rank == emitted_rank
        record["earlier_parent_repeat"] = matched_rank in PARENT_PERSON_RANKS
        if not record["immediate_self_repeat"] and not record["earlier_parent_repeat"]:
            record["backward_uncovered_person"] = matched_rank < emitted_rank
            record["forward_uncovered_person"] = matched_rank > emitted_rank
    elif matched_description == "tie":
        record["outcome"] = "matched_tie"
    else:
        record["outcome"] = "matched_other_category"
    return record


def summarize_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    totals = Counter(str(record["outcome"]) for record in records)
    totals.update(
        {
            "call_count": len(records),
            "immediate_self_repeat_count": sum(bool(r["immediate_self_repeat"]) for r in records),
            "earlier_parent_repeat_count": sum(bool(r["earlier_parent_repeat"]) for r in records),
            "backward_uncovered_person_count": sum(bool(r["backward_uncovered_person"]) for r in records),
            "forward_uncovered_person_count": sum(bool(r["forward_uncovered_person"]) for r in records),
        }
    )

    def group_summary(group: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        rows = list(group)
        successor_ranks = Counter(
            str(row["matched_category_rank"])
            for row in rows
            if row.get("outcome") == "matched_person"
        )
        outcomes = Counter(str(row["outcome"]) for row in rows)
        return {
            "call_count": len(rows),
            "outcome_counts": dict(sorted(outcomes.items())),
            "person_successor_rank_counts": dict(
                sorted(successor_ranks.items(), key=lambda pair: int(pair[0]))
            ),
            "immediate_self_repeat_count": sum(bool(r["immediate_self_repeat"]) for r in rows),
            "earlier_parent_repeat_count": sum(bool(r["earlier_parent_repeat"]) for r in rows),
            "backward_uncovered_person_count": sum(bool(r["backward_uncovered_person"]) for r in rows),
            "forward_uncovered_person_count": sum(bool(r["forward_uncovered_person"]) for r in rows),
        }

    by_owner_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_variant_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        owner_id = str(record["owner_id"])
        by_owner_rows[owner_id].append(record)
        by_variant_rows[f'{owner_id}:{record["variant_label"]}'].append(record)

    counterfactual_appearances: dict[str, Any] = {}
    for owner_id, rank in EMITTED_PERSON_RANKS.items():
        self_arm = [r for r in records if r["owner_id"] == owner_id]
        other_arms = [r for r in records if r["owner_id"] != owner_id]

        def appears(row: Mapping[str, Any]) -> bool:
            return (
                row.get("outcome") == "matched_person"
                and int(row["matched_category_rank"]) == rank
            )

        counterfactual_appearances[owner_id] = {
            "person_rank": rank,
            "after_self_arm_count": sum(appears(row) for row in self_arm),
            "after_self_arm_total": len(self_arm),
            "after_other_owner_arms_count": sum(appears(row) for row in other_arms),
            "after_other_owner_arms_total": len(other_arms),
        }

    paired_variant_agreement: dict[str, Any] = {}
    for owner_id in EMITTED_PERSON_RANKS:
        owner_rows = [r for r in records if r["owner_id"] == owner_id]
        by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
        for row in owner_rows:
            by_seed[int(row["sampling_seed"])].append(row)
        agreement_count = 0
        seed_records: list[dict[str, Any]] = []
        for seed, rows in sorted(by_seed.items()):
            labels = [
                (
                    f'person:{row["matched_category_rank"]}'
                    if row.get("outcome") == "matched_person"
                    else str(row["outcome"])
                )
                for row in sorted(rows, key=lambda item: str(item["variant_label"]))
            ]
            all_equal = len(rows) == 3 and len(set(labels)) == 1
            agreement_count += int(all_equal)
            seed_records.append(
                {"sampling_seed": seed, "successor_labels": labels, "all_three_agree": all_equal}
            )
        paired_variant_agreement[owner_id] = {
            "agreeing_seed_count": agreement_count,
            "seed_count": len(by_seed),
            "seeds": seed_records,
        }

    canonical_rows = [r for r in records if r["variant_label"] == "variant-01"]
    canonical_by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in canonical_rows:
        canonical_by_seed[int(row["sampling_seed"])].append(row)
    differentiation_records: list[dict[str, Any]] = []
    distinct_counts: Counter[int] = Counter()
    for seed, rows in sorted(canonical_by_seed.items()):
        labels_by_owner = {
            str(row["owner_id"]): (
                f'person:{row["matched_category_rank"]}'
                if row.get("outcome") == "matched_person"
                else str(row["outcome"])
            )
            for row in rows
        }
        distinct_count = len(set(labels_by_owner.values()))
        distinct_counts[distinct_count] += 1
        differentiation_records.append(
            {
                "sampling_seed": seed,
                "successor_label_by_owner": dict(sorted(labels_by_owner.items())),
                "distinct_successor_count": distinct_count,
            }
        )

    return {
        "totals": dict(sorted(totals.items())),
        "by_owner": {
            owner_id: group_summary(rows)
            for owner_id, rows in sorted(by_owner_rows.items())
        },
        "by_variant": {
            key: group_summary(rows)
            for key, rows in sorted(by_variant_rows.items())
        },
        "owner_counterfactual_appearances": counterfactual_appearances,
        "paired_exact_variant_agreement": paired_variant_agreement,
        "canonical_owner_differentiation": {
            "distinct_successor_count_histogram": {
                str(key): value for key, value in sorted(distinct_counts.items())
            },
            "seeds": differentiation_records,
        },
    }


def collect_records(
    *,
    canonical_root: Path,
    variant_root: Path,
    relabel_objects: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[int], list[dict[str, Any]]]:
    directories: list[tuple[str, str, Path]] = []
    for owner_id in EMITTED_PERSON_RANKS:
        directories.append((owner_id, "variant-01", canonical_root / f"owner-{owner_id}"))
        directories.append((owner_id, "variant-02", variant_root / f"owner-{owner_id}-variant-02"))
        directories.append((owner_id, "variant-03", variant_root / f"owner-{owner_id}-variant-03"))

    first_receipt = json.loads((directories[0][2] / "receipt.json").read_text())
    expected_seeds = [int(seed) for seed in first_receipt["sampling_seeds"]]
    records: list[dict[str, Any]] = []
    receipt_records: list[dict[str, Any]] = []
    for owner_id, variant_label, directory in directories:
        receipt_path = directory / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        validate_receipt(receipt, expected_seeds)
        receipt_records.append(
            {
                "owner_id": owner_id,
                "variant_label": variant_label,
                "path": str(receipt_path),
                "branch_label": receipt["branch_label"],
                "prompt_token_ids_sha256": receipt["prompt"]["prompt_token_ids_sha256"],
                "branch_row_token_ids_sha256": receipt["donor"]["branch_row"]["token_ids_sha256"],
            }
        )
        calls_by_seed = {int(call["sampling_seed"]): call for call in receipt["calls"]}
        if sorted(calls_by_seed) != sorted(expected_seeds):
            raise ValueError(f"receipt calls do not cover expected seeds: {receipt_path}")
        for seed in expected_seeds:
            bundle_path = Path(calls_by_seed[seed]["bundle_path"])
            bundle = json.loads(bundle_path.read_text())
            if int(bundle["sampling_seed"]) != seed:
                raise ValueError(f"bundle seed mismatch: {bundle_path}")
            records.append(
                classify_call(
                    bundle=bundle,
                    bundle_path=bundle_path,
                    owner_id=owner_id,
                    emitted_rank=EMITTED_PERSON_RANKS[owner_id],
                    variant_label=variant_label,
                    relabel_objects=relabel_objects,
                )
            )
    records.sort(
        key=lambda row: (
            str(row["owner_id"]),
            str(row["variant_label"]),
            int(row["sampling_seed"]),
        )
    )
    return records, expected_seeds, receipt_records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--relabel-jsonl", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--variant-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    relabel_path = args.relabel_jsonl.expanduser().resolve(strict=True)
    canonical_root = args.canonical_root.expanduser().resolve(strict=True)
    variant_root = args.variant_root.expanduser().resolve(strict=True)
    record, relabel_record_sha256 = load_image_record(relabel_path)
    relabel_objects = build_relabel_objects(record)
    person_count = sum(obj["description"] == "person" for obj in relabel_objects)
    tie_count = sum(obj["description"] == "tie" for obj in relabel_objects)
    if person_count != 38 or tie_count != 8:
        raise ValueError(f"unexpected image-2299 relabel counts: person={person_count}, tie={tie_count}")
    records, seeds, receipts = collect_records(
        canonical_root=canonical_root,
        variant_root=variant_root,
        relabel_objects=relabel_objects,
    )
    if len(records) != 96:
        raise ValueError(f"expected 96 calls, found {len(records)}")
    payload = {
        "schema_version": "image2299_near_complete_relabel_successor_transition.analysis.v1",
        "image_id": IMAGE_ID,
        "relabel": {
            "path": str(relabel_path),
            "record_sha256": relabel_record_sha256,
            "width": int(record["width"]),
            "height": int(record["height"]),
            "person_count": person_count,
            "tie_count": tie_count,
        },
        "frozen_contract": {
            "emitted_person_rank_by_owner_id": EMITTED_PERSON_RANKS,
            "earlier_parent_person_ranks": sorted(PARENT_PERSON_RANKS),
            "sampling_seeds": seeds,
            "matching_minimum_intersection_over_union": 0.5,
            "matching_minimum_top_minus_second_margin": 0.05,
            "generation_policy": EXPECTED_GENERATION_POLICY,
            "physical_batch_size": 1,
            "prompt_construction_mode": "exact_donor_token_ids",
            "sampling_context": "local-direct-sampling-context",
        },
        "receipts": receipts,
        "records": records,
        "summary": summarize_records(records),
    }
    output_path = args.output_json.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["summary"]["totals"], sort_keys=True))
    print(output_path)


if __name__ == "__main__":
    main()
