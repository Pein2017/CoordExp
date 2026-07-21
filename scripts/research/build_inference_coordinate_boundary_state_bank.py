#!/usr/bin/env python3
"""Mine a conservative coordinate-boundary StateBank from source inference.

Only already-labelled entities with a unique same-category match are eligible.
Unmatched predictions, duplicate matches, ambiguous crowded matches,
border-truncated boxes, and rows without a clear first wrong
coordinate are ignored.  One event at most is admitted per physical image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file, sha256_json
from src.config.inference import load_infer_config
from src.data import load_raw_examples
from src.inference.backend import token_ids_sha256
from src.inference.runtime import assemble_frontend
from src.rollout_calibration import CheckpointIdentity, assemble_state_bank, load_state_bank
from scripts.research.run_current_seeded_sampled_rollouts import _build_requests


SCHEMA_VERSION = "inference_coordinate_boundary_state_bank_builder.v1"
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END = 152670
OBJECT_START = 151646
OBJECT_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATES = ("x1", "y1", "x2", "y2")
AXES = {"x1": "horizontal", "x2": "horizontal", "y1": "vertical", "y2": "vertical"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    with path.resolve(strict=True).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _iou(left: Sequence[int], right: Sequence[int]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    left_area = max(0, left[2] - left[0]) * max(0, left[3] - left[1])
    right_area = max(0, right[2] - right[0]) * max(0, right[3] - right[1])
    union = left_area + right_area - intersection
    return 0.0 if union <= 0 else intersection / union


def split_complete_rows(token_ids: Sequence[int]) -> list[list[int]]:
    """Return exact complete object rows, refusing nested or partial starts."""
    rows: list[list[int]] = []
    index = 0
    values = [int(value) for value in token_ids]
    while index < len(values):
        if values[index] != OBJECT_START:
            index += 1
            continue
        try:
            end = values.index(BOX_END, index + 1)
        except ValueError:
            break
        row = values[index : end + 1]
        if row.count(OBJECT_START) == 1 and row.count(OBJECT_END) == 1 and row.count(BOX_START) == 1:
            coords = [value for value in row if COORDINATE_TOKEN_START <= value < COORDINATE_TOKEN_END]
            if len(coords) == 4:
                rows.append(row)
        index = end + 1
    return rows


def first_wrong_coordinate(
    predicted: Sequence[int], reference: Sequence[int], *, tolerance: int
) -> tuple[int, list[dict[str, Any]]] | None:
    observations: list[dict[str, Any]] = []
    for index, (actual, expected) in enumerate(zip(predicted, reference, strict=True)):
        acceptable = list(range(max(0, expected - tolerance), min(999, expected + tolerance) + 1))
        observations.append(
            {
                "coordinate": COORDINATES[index],
                "tolerance_axis": AXES[COORDINATES[index]],
                "actual_coordinate_value": int(actual),
                "acceptable_coordinate_values": acceptable,
            }
        )
        if actual not in acceptable:
            return index, observations
    return None


def unique_owner_match(
    prediction: Mapping[str, Any],
    ground_truth: Sequence[Mapping[str, Any]],
    *,
    minimum_iou: float,
    minimum_margin: float,
) -> tuple[Mapping[str, Any], float, float] | None:
    category = str(prediction.get("description"))
    coordinate_bins = prediction.get("coord_bins")
    if (
        not isinstance(coordinate_bins, list)
        or len(coordinate_bins) != 4
        or not all(
            isinstance(value, int) and 0 <= value <= 999
            for value in coordinate_bins
        )
    ):
        return None
    candidates = [item for item in ground_truth if str(item.get("description")) == category]
    scored = sorted(
        ((_iou(coordinate_bins, item["bbox"]), item) for item in candidates),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not scored or scored[0][0] < minimum_iou:
        return None
    second = scored[1][0] if len(scored) > 1 else 0.0
    if scored[0][0] - second < minimum_margin:
        return None
    return scored[0][1], scored[0][0], second


def _image_id(example_id: str) -> int:
    return int(example_id.rsplit("_", 1)[-1])


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_pad_interval(ids: Sequence[int], image_pad_id: int) -> list[int]:
    positions = [index for index, value in enumerate(ids) if int(value) == image_pad_id]
    if not positions or positions != list(range(positions[0], positions[-1] + 1)):
        raise ValueError("executed prompt must contain one contiguous image-pad interval")
    return [positions[0], positions[-1] + 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference-dir", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--reference-bank-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=256)
    parser.add_argument("--eval-count", type=int, default=64)
    parser.add_argument("--coordinate-tolerance", type=int, default=10)
    parser.add_argument("--minimum-iou", type=float, default=0.55)
    parser.add_argument("--minimum-match-margin", type=float, default=0.30)
    args = parser.parse_args()

    infer_dir = args.inference_dir.resolve(strict=True)
    resolved = load_infer_config(args.infer_config.resolve(strict=True))
    config = resolved.config
    examples = list(load_raw_examples(config.data.input_jsonl))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    requests, _ = _build_requests(config, frontend, examples)
    request_by_row = {request.request_id: request for request in requests}
    image_pad_id = int(frontend.qwen.processor.tokenizer.convert_tokens_to_ids("<|image_pad|>"))

    gt_rows = {str(row["row_id"]): row for row in _jsonl(infer_dir / "gt_vs_pred.jsonl")}
    traces: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in _jsonl(infer_dir / "pred_token_trace.jsonl"):
        if row.get("trace_type") != "generated_token" or row.get("is_pad") or row.get("is_stop"):
            continue
        traces[str(row["row_id"])].append((int(row["generated_step_index"]), int(row["token_id"])))

    reference = json.loads(args.reference_bank_manifest.resolve(strict=True).read_text())
    checkpoint = CheckpointIdentity.from_mapping(reference["source_checkpoint"])
    checkpoint_id = str(reference["source_checkpoint_id"])
    prompt_identity = str(reference["prompt_identity_sha256"])
    rollout_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    exclusions: dict[str, int] = defaultdict(int)

    for row_id in sorted(gt_rows):
        artifact = gt_rows[row_id]
        exact_ids = [value for _, value in sorted(traces.get(row_id, []))]
        exact_rows = split_complete_rows(exact_ids)
        predictions = artifact.get("pred") or []
        if len(exact_rows) != len(predictions):
            exclusions["row_parse_mismatch"] += 1
            continue
        request = request_by_row[row_id]
        eligible_rows: list[
            tuple[
                int,
                list[int],
                Mapping[str, Any],
                Mapping[str, Any],
                list[dict[str, Any]],
                float,
                float,
            ]
        ] = []
        prefix: list[int] = []
        for row_index, (tokens, prediction) in enumerate(zip(exact_rows, predictions, strict=True)):
            match = unique_owner_match(
                prediction,
                artifact.get("gt") or [],
                minimum_iou=args.minimum_iou,
                minimum_margin=args.minimum_match_margin,
            )
            if match is None:
                prefix.extend(tokens)
                continue
            owner, best_iou, second_iou = match
            source = ((owner.get("metadata") or {}).get("source") or {})
            annotation_id = source.get("coco_ann_id")
            reference_box = [int(value) for value in owner["bbox"]]
            predicted_box = [int(value) for value in prediction["coord_bins"]]
            if not isinstance(annotation_id, int):
                exclusions["missing_annotation_identity"] += 1
                prefix.extend(tokens)
                continue
            if min(reference_box[2] - reference_box[0], reference_box[3] - reference_box[1]) < 20:
                exclusions["small_reference_extent"] += 1
                prefix.extend(tokens)
                continue
            if any(value in {0, 999} for value in reference_box):
                exclusions["border_truncated_reference"] += 1
                prefix.extend(tokens)
                continue
            wrong = first_wrong_coordinate(
                predicted_box, reference_box, tolerance=args.coordinate_tolerance
            )
            if wrong is None:
                exclusions["all_coordinates_accepted"] += 1
                prefix.extend(tokens)
                continue
            coordinate_index, observations = wrong
            coordinate_offsets = [
                index for index, value in enumerate(tokens)
                if COORDINATE_TOKEN_START <= value < COORDINATE_TOKEN_END
            ]
            for observation_index, observation in enumerate(observations):
                observation["candidate_token_offset"] = coordinate_offsets[observation_index]
                observation["review_provenance"] = {
                    "source": str(infer_dir / "gt_vs_pred.jsonl"),
                    "reviewer": "conservative-automatic-training-annotation-match",
                    "confidence": "high",
                    "comment": (
                        f"Training annotation owner; same-category unique match IoU={best_iou:.4f}, "
                        f"second-best IoU={second_iou:.4f}; tolerance={args.coordinate_tolerance}."
                    ),
                }
            eligible_rows.append((
                row_index, list(prefix), prediction, owner, observations,
                best_iou, second_iou,
            ))
            prefix.extend(tokens)
        if not eligible_rows:
            exclusions["no_eligible_event"] += 1
            continue
        selected_index = int.from_bytes(
            hashlib.sha256(row_id.encode()).digest()[:8], "big"
        ) % len(eligible_rows)
        selected = eligible_rows[selected_index]
        row_index, prefix, prediction, owner, observations, best_iou, second_iou = selected
        tokens = exact_rows[row_index]
        event_id = f"coordinate-boundary-image-{_image_id(row_id)}-row-{row_index}"
        candidate_id = f"source-greedy-image-{_image_id(row_id)}-row-{row_index}"
        prompt_ids = list(request.expected_executed_prompt_token_ids)
        physical_owner_id = str(owner["object_id"])
        image_path = Path(str(artifact["image_path"])).resolve(strict=True)
        rollout_rows.append(
            {
                "event_id": event_id,
                "image": {
                    "image_id": _image_id(row_id),
                    "path": str(image_path),
                    "width": int(artifact["image_width"]),
                    "height": int(artifact["image_height"]),
                    "content_sha256": _content_sha256(image_path),
                },
                "split": "train",
                "split_group_id": f"image:{_image_id(row_id)}",
                "executed_prompt_token_ids": prompt_ids,
                "executed_prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
                "image_pad_interval": _image_pad_interval(prompt_ids, image_pad_id),
                "prefix_token_ids": prefix,
                "prefix_token_ids_sha256": token_ids_sha256(prefix),
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "token_ids": tokens,
                        "token_ids_sha256": token_ids_sha256(tokens),
                        "generation_provenance": {
                            "mode": "greedy",
                            "seed": 0,
                            "temperature": 0.0,
                            "top_p": 1.0,
                            "repetition_penalty": 1.0,
                            "checkpoint_id": checkpoint_id,
                            "prompt_token_ids_sha256": token_ids_sha256(prompt_ids),
                            "prefix_token_ids_sha256": token_ids_sha256(prefix),
                        },
                        "evidence_text": frontend.qwen.processor.tokenizer.decode(tokens, skip_special_tokens=False),
                    }
                ],
            }
        )
        review_rows.append(
            {
                "event_id": event_id,
                "admission_status": "accepted",
                "rejection_reason": None,
                "physical_entities": [
                    {
                        "entity_id": physical_owner_id,
                        "category": str(owner["description"]),
                        "entity_trusted": True,
                        "geometry_trusted": True,
                        "reference_bbox": [int(value) for value in owner["bbox"]],
                        "review_source": str(infer_dir / "gt_vs_pred.jsonl"),
                        "reviewer": "conservative-automatic-training-annotation-match",
                        "review_confidence": "high",
                        "comment": f"Unique current training-annotation match; IoU={best_iou:.4f}; margin={best_iou-second_iou:.4f}.",
                    }
                ],
                "prefix_object_row_count": row_index,
                "prefix_coverage_status": "empty" if row_index == 0 else "unresolved",
                "prefix_covered_owner_proofs": [],
                "entity_transition_eligible": False,
                "coordinate_boundary_eligible": True,
                "candidates": [
                    {
                        "candidate_id": candidate_id,
                        "role": "diagnostic",
                        "harmful_kind": None,
                        "physical_owner_id": physical_owner_id,
                        "coverage_status": "unknown",
                        "entity_review_status": "trusted",
                        "geometry_review_status": "trusted",
                        "entity_eligible": False,
                        "geometry_eligible": True,
                        "owner_resolution_interval": None,
                        "coordinate_decision": {
                            "owner_id": physical_owner_id,
                            "observations": observations,
                        },
                        "selected_sites": [
                            {
                                "candidate_token_offset": observations[-1]["candidate_token_offset"],
                                "intended_token_type": "coordinate",
                            }
                        ],
                    }
                ],
                "review_provenance": {
                    "schema_version": SCHEMA_VERSION,
                    "policy": "current-training-annotation-positive-unique-match-only",
                    "best_iou": best_iou,
                    "second_best_iou": second_iou,
                    "coordinate_tolerance": args.coordinate_tolerance,
                },
            }
        )

    total = args.train_count + args.eval_count
    eligible_before_limit = len(rollout_rows)
    paired = sorted(
        zip(rollout_rows, review_rows, strict=True),
        key=lambda pair: hashlib.sha256(pair[0]["event_id"].encode()).hexdigest(),
    )
    if len(paired) < total:
        raise SystemExit(f"only {len(paired)} eligible image events; need {total}")
    paired = paired[:total]
    for index, (rollout, _) in enumerate(paired):
        rollout["split"] = "train" if index < args.train_count else "eval"
    rollout_rows = [pair[0] for pair in paired]
    review_rows = [pair[1] for pair in paired]
    output = args.output_dir.resolve()
    manifest = assemble_state_bank(
        output_dir=output / "state-bank",
        rollout_rows=rollout_rows,
        review_rows=review_rows,
        source_checkpoint=checkpoint,
        prompt_identity_sha256=prompt_identity,
        source_artifacts=[
            {"artifact_id": "source-inference-run-manifest", "sha256": sha256_file(infer_dir / "run_manifest.json")},
            {"artifact_id": "source-inference-predictions", "sha256": sha256_file(infer_dir / "gt_vs_pred.jsonl")},
            {"artifact_id": "source-inference-token-trace", "sha256": sha256_file(infer_dir / "pred_token_trace.jsonl")},
            {"artifact_id": "candidate-inference-config", "sha256": sha256_file(args.infer_config)},
            {"artifact_id": "reference-state-bank-manifest", "sha256": sha256_file(args.reference_bank_manifest)},
        ],
    )
    loaded = load_state_bank(
        output / "state-bank" / "manifest.json",
        expected_source_checkpoint=checkpoint,
        expected_prompt_identity_sha256=prompt_identity,
    )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "eligible_before_limit": eligible_before_limit,
        "train_count": args.train_count,
        "eval_count": args.eval_count,
        "exclusions": dict(sorted(exclusions.items())),
        "manifest": manifest.to_artifact_dict(),
        "validation": loaded.validation_receipt.to_artifact_dict(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "build-receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt["validation"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
