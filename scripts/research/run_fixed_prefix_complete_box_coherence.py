#!/usr/bin/env python3
"""Run the fixed-prefix complete-box coherence research experiment.

The entrypoint materializes immutable target fixtures, reconstructs exact
pre-``x1`` prefixes, scores coherent and binary-hybrid boxes, and progressively
releases coordinate suffixes under greedy or attested paired sampling.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import itertools
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_batch_coordinate_logit_invariance import (  # noqa: E402
    COORDINATE_TOKEN_END_EXCLUSIVE,
    COORDINATE_TOKEN_START,
)


DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
DEFAULT_BUNDLE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/executions/"
    "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls"
)
DEFAULT_BUNDLES = {
    "A": DEFAULT_BUNDLE_ROOT
    / "1db840709abcf5e20a5bd475fa1acfbeca2bcf64060374b856218444337f6762"
    / "terminal-output-bundle.json",
    "C": DEFAULT_BUNDLE_ROOT
    / "2b4c85bbd63fe017d8c9b67028e46c4f60006c93cf88974cff010c47cf0a200f"
    / "terminal-output-bundle.json",
}
TARGET_ROWS = {"A": 11, "C": 13}
TOKEN_NAME_RE = re.compile(r"<\|coord_(\d+)\|>")
MAX_COORDINATE_BIN = 999
CLASSIFICATION_RADIUS = 0.35
CLASSIFICATION_MARGIN = 0.05
RELEASE_MAX_NEW_TOKENS = 5
DEFAULT_RELEASE_TEMPERATURE = 0.4
DEFAULT_RELEASE_TOP_P = 0.95


def _json_hash(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(value) for value in values], separators=(",", ":")).encode()
    ).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def coordinate_token_id(coordinate_bin: int) -> int:
    if not isinstance(coordinate_bin, int) or not 0 <= coordinate_bin <= MAX_COORDINATE_BIN:
        raise ValueError(f"coordinate bin must be in [0, 999], got {coordinate_bin!r}")
    return COORDINATE_TOKEN_START + coordinate_bin


def coordinate_bin(token_id: int) -> int:
    if not COORDINATE_TOKEN_START <= int(token_id) < COORDINATE_TOKEN_END_EXCLUSIVE:
        raise ValueError(f"token {token_id!r} is not a coordinate token")
    return int(token_id) - COORDINATE_TOKEN_START


def _token_coordinate(token: Mapping[str, Any]) -> int | None:
    match = TOKEN_NAME_RE.fullmatch(str(token.get("token_text", "")))
    return None if match is None else int(match.group(1))


def _row_from_bundle(bundle: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    receipts = bundle.get("parse_score_receipts")
    if not isinstance(receipts, list) or not 0 <= row_index < len(receipts):
        raise ValueError("source bundle lacks the requested parse-score row")
    receipt = receipts[row_index]
    selected_steps = receipt.get("selected_generated_step_indices")
    selected_tokens = receipt.get("selected_token_ids")
    if not isinstance(selected_steps, list) or len(selected_steps) != 8:
        raise ValueError("source row must expose the frozen eight selected token steps")
    if not isinstance(selected_tokens, list) or len(selected_tokens) != 8:
        raise ValueError("source row must expose the frozen eight selected tokens")
    decode = bundle.get("decode_result")
    if not isinstance(decode, Mapping):
        raise ValueError("source bundle lacks decode_result")
    generated = [int(value) for value in decode.get("generated_token_ids", [])]
    prompt = [int(value) for value in decode.get("prompt_token_ids", [])]
    trace = decode.get("token_trace")
    if not isinstance(trace, list):
        raise ValueError("source bundle lacks token_trace")
    # The eight selected score steps include structural tokens but exclude the
    # free-form description token. The contiguous slice still contains the
    # complete nine-token one-word-description row.
    row_start, row_end = int(selected_steps[0]), int(selected_steps[-1])
    x1_step = int(selected_steps[3])
    if row_start < 0 or row_end >= len(generated) or x1_step >= len(generated):
        raise ValueError("source selected steps are outside generated token ids")
    if generated[row_start] != 151646:
        raise ValueError("source row does not begin with the object-reference opener")
    token_texts = [str(trace[int(step)].get("token_text", "")) for step in selected_steps]
    if token_texts[3] != f"<|coord_{coordinate_bin(int(selected_tokens[3]))}|>":
        raise ValueError("source token id/name coordinate semantics disagree")
    row_tokens = generated[row_start : row_end + 1]
    prefix_tokens = [*prompt, *generated[:x1_step]]
    return {
        "row_index": int(row_index),
        "category": str(receipt.get("normalized_category_name") or receipt.get("category_text") or ""),
        "row_span_generated_steps": [row_start, row_end],
        "selected_generated_step_indices": [int(value) for value in selected_steps],
        "selected_token_ids": [int(value) for value in selected_tokens],
        "selected_token_text": token_texts,
        "row_token_ids": row_tokens,
        "row_token_ids_sha256": _json_hash(row_tokens),
        "pre_x1_generated_token_count": x1_step,
        "pre_x1_generated_token_ids": generated[:x1_step],
        "pre_x1_prompt_token_ids": prefix_tokens,
        "pre_x1_prompt_token_ids_sha256": _json_hash(prefix_tokens),
        "source_prompt_token_ids": prompt,
        "source_prompt_token_ids_sha256": _json_hash(prompt),
        "source_generated_token_ids_sha256": _json_hash(generated),
        "source_row_parse_receipt_sha256": str(receipt.get("receipt_sha256", "")),
        "source_row_span_sha256": str(receipt.get("raw_span_sha256", "")),
        "source_row_coordinates": [coordinate_bin(int(value)) for value in selected_tokens[3:7]],
        "source_raw_span_text_sha256": str(receipt.get("raw_span_sha256", "")),
        "source_object_span_id": str(receipt.get("object_span_id", "")),
    }


def _bins_to_pixels(box: Sequence[int], width: int, height: int) -> list[int]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("box and image dimensions are invalid")
    x1, y1, x2, y2 = [int(value) for value in box]
    return [round(x1 * width / 1000), round(y1 * height / 1000), round(x2 * width / 1000), round(y2 * height / 1000)]


def _pixels_to_bins(box: Sequence[int | float], width: int, height: int) -> list[int]:
    if len(box) != 4 or width <= 0 or height <= 0:
        raise ValueError("box and image dimensions are invalid")
    x1, y1, x2, y2 = [float(value) for value in box]
    return [round(x1 * 1000 / width), round(y1 * 1000 / height), round(x2 * 1000 / width), round(y2 * 1000 / height)]


def assert_one_bin_round_trip(box: Sequence[int], width: int, height: int) -> dict[str, Any]:
    pixels = _bins_to_pixels(box, width, height)
    recovered = _pixels_to_bins(pixels, width, height)
    errors = [abs(int(a) - int(b)) for a, b in zip(box, recovered)]
    if max(errors, default=0) > 1:
        raise ValueError(f"coordinate round trip exceeds one bin: {box} -> {pixels} -> {recovered}")
    return {"source_bins": list(map(int, box)), "pixels": pixels, "recovered_bins": recovered, "absolute_bin_errors": errors}


def _reference(
    label: str,
    box: Sequence[int],
    *,
    kind: str = "visible",
    owner_id: str | None = None,
) -> dict[str, Any]:
    values = [int(value) for value in box]
    if len(values) != 4 or values[0] >= values[2] or values[1] >= values[3]:
        raise ValueError(f"invalid reference box {label}: {values}")
    result: dict[str, Any] = {"label": label, "kind": kind, "box": values}
    if owner_id is not None:
        result["owner_id"] = owner_id
    return result


def _target_references(target: str) -> tuple[dict[str, Any], ...]:
    if target == "A":
        return (
            _reference("part", [658, 450, 780, 487], kind="part"),
            _reference("visible_whole", [660, 448, 999, 488], owner_id="coco-ann:686666"),
        )
    if target == "C":
        return (
            _reference("target_owner", [537, 121, 651, 349], owner_id="coco-ann:378536"),
            _reference("adjacent_owner", [643, 126, 730, 350], owner_id="coco-ann:387701"),
            _reference("upper_row_owner", [557, 30, 656, 134], owner_id="coco-ann:385840"),
            _reference("multi_instance_union", [519, 33, 657, 347], kind="union", owner_id="source-row-4"),
        )
    raise ValueError(f"unsupported target {target!r}")


target_references = _target_references


def box_shape_metadata(box: Sequence[int], *, sort_rank: int | None = None) -> dict[str, Any]:
    """Return auditable normalized-bin shape metadata for one valid box."""

    if len(box) != 4:
        raise ValueError("box shape metadata requires four coordinates")
    x1, y1, x2, y2 = [int(value) for value in box]
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("box must be non-degenerate")
    result: dict[str, Any] = {
        "width": width,
        "height": height,
        "area": width * height,
        "aspect_ratio": float(width / height),
        "center": [float((x1 + x2) / 2), float((y1 + y2) / 2)],
    }
    if sort_rank is not None:
        result["geometry_sort_rank"] = int(sort_rank)
    return result


def enumerate_binary_hybrids(
    left: Sequence[int],
    right: Sequence[int],
    *,
    prefix: str = "hybrid",
    left_label: str = "left_endpoint",
    right_label: str = "right_endpoint",
    left_kind: str = "visible",
    right_kind: str = "visible",
) -> list[dict[str, Any]]:
    """Enumerate all unique valid boundary combinations from two parent boxes."""

    parents = ([int(value) for value in left], [int(value) for value in right])
    if any(len(box) != 4 for box in parents):
        raise ValueError("binary hybrid parents must be four-coordinate boxes")
    output: list[dict[str, Any]] = []
    seen: set[tuple[int, ...]] = set()
    for choices in itertools.product((0, 1), repeat=4):
        box = tuple(parents[choice][slot] for slot, choice in enumerate(choices))
        if box in seen:
            continue
        seen.add(box)
        valid = box[0] < box[2] and box[1] < box[3]
        is_left = choices == (0, 0, 0, 0)
        is_right = choices == (1, 1, 1, 1)
        output.append({
            "label": left_label if is_left else right_label if is_right else f"{prefix}:{''.join(map(str, choices))}",
            "kind": left_kind if is_left else right_kind if is_right else "hybrid",
            "box": list(box),
            "parent_choices": list(choices),
            "valid": bool(valid),
            "is_endpoint": is_left or is_right,
            "shape": box_shape_metadata(box) if valid else None,
        })
    return output


def enumerate_target_panels(target: str) -> list[dict[str, Any]]:
    references = _target_references(target)
    panels: list[dict[str, Any]] = []
    if target == "A":
        panels.append({
            "panel": "part_vs_visible_whole",
            "references": list(references),
            "hybrids": enumerate_binary_hybrids(
                references[0]["box"],
                references[1]["box"],
                prefix="part_whole",
                left_label=references[0]["label"],
                right_label=references[1]["label"],
                left_kind=references[0]["kind"],
                right_kind=references[1]["kind"],
            ),
            "claim_contract": {
                "exact_earliest_differing_slot": "x1",
                "earliest_semantically_meaningful_differing_slot": "x2",
                "owner_claim_admitted": False,
                "extent_claim_admitted": True,
                "owner_discriminative_slots": [],
                "extent_discriminative_slots": ["x2"],
                "admissible_claim_types": [
                    "part-versus-visible-whole configuration preference",
                    "late x2 extent-and-closure behavior",
                ],
            },
        })
    else:
        for index in (1, 2, 3):
            comparison = references[index]
            owner_admitted = comparison["kind"] != "union"
            owner_slots = {
                "adjacent_owner": ["x1", "x2"],
                "upper_row_owner": ["y1", "y2"],
                "multi_instance_union": [],
            }[comparison["label"]]
            region_slots = (
                ["x1", "y1"] if comparison["kind"] == "union" else []
            )
            semantic_slots = owner_slots if owner_admitted else region_slots
            semantic_entry_slot = semantic_slots[0]
            panels.append({
                "panel": f"target_vs_{comparison['label']}",
                "references": [references[0], comparison],
                "hybrids": enumerate_binary_hybrids(
                    references[0]["box"],
                    comparison["box"],
                    prefix=f"target_{comparison['label']}",
                    left_label=references[0]["label"],
                    right_label=comparison["label"],
                    left_kind=references[0]["kind"],
                    right_kind=comparison["kind"],
                ),
                "claim_contract": {
                    "exact_earliest_differing_slot": "x1",
                    "earliest_semantically_meaningful_differing_slot": semantic_entry_slot,
                    "owner_claim_admitted": owner_admitted,
                    "extent_claim_admitted": False,
                    "region_aggregation_claim_admitted": comparison["kind"] == "union",
                    "owner_discriminative_slots": owner_slots,
                    "region_discriminative_slots": region_slots,
                    "admissible_claim_types": (
                        [
                            "same-class instance-owner preservation after an "
                            f"{semantic_entry_slot} cue"
                        ]
                        if owner_admitted
                        else ["single-owner versus multi-instance-region aggregation"]
                    ),
                },
            })
    return panels


def _image_metadata(source_jsonl: Path, image_id: str) -> dict[str, Any]:
    from src.data import load_raw_examples

    rows = load_raw_examples(source_jsonl)
    row = next((item for item in rows if str(item.metadata.get("source", {}).get("image_id")) == str(image_id)), None)
    if row is None:
        raise ValueError(f"image {image_id} is absent from source JSONL")
    image_path = Path(row.image.path).resolve(strict=True)
    return {
        "example_id": row.example_id,
        "image_id": str(image_id),
        "image_path": str(image_path),
        "image_sha256": _file_hash(image_path),
        "width": int(row.image.width),
        "height": int(row.image.height),
        "source_objects": [obj.to_artifact_dict() for obj in row.objects],
    }


def materialize_fixture(
    *,
    source_jsonl: Path = DEFAULT_SOURCE_JSONL,
    bundles: Mapping[str, Path] = DEFAULT_BUNDLES,
) -> dict[str, Any]:
    """Resolve the immutable artifact-only target fixture and assert its anchors."""

    targets: dict[str, Any] = {}
    for target in ("A", "C"):
        path = Path(bundles[target]).expanduser().resolve(strict=True)
        bundle = json.loads(path.read_text(encoding="utf-8"))
        expected_request = {
            "A": "spatial-scope-history-request:18404482ab0b688758cca6759b5f83bbe5e5959d3689641fdc53bf857977d8a2",
            "C": "spatial-scope-history-request:ff5660928aa3d9401fb9679d0112b3a5477f1bf3b13a25b9339c2911e02fa0d9",
        }[target]
        if str(bundle.get("request_id")) != expected_request:
            raise ValueError(f"{target}: source request id does not match frozen target")
        row = _row_from_bundle(bundle, TARGET_ROWS[target])
        image_id = str(bundle["scheduled_request"]["image_id"])
        image = _image_metadata(source_jsonl, image_id)
        expected_digest = str(bundle["execution_evidence"].get("source_image_sha256", ""))
        if expected_digest and image["image_sha256"] != expected_digest:
            raise ValueError(f"{target}: source image digest differs from executed bundle")
        references = list(_target_references(target))
        for reference in references:
            reference["round_trip"] = assert_one_bin_round_trip(reference["box"], image["width"], image["height"])
            reference["shape"] = box_shape_metadata(reference["box"])
        if target == "A":
            expected_source = [658, 450, 780, 487]
            if row["source_row_coordinates"] != expected_source:
                raise ValueError(f"A: unexpected exact source row coordinates {row['source_row_coordinates']}")
        else:
            expected_source = [552, 123, 660, 357]
            if row["source_row_coordinates"] != expected_source:
                raise ValueError(f"C: unexpected exact source row coordinates {row['source_row_coordinates']}")
            row4 = _row_from_bundle(bundle, 4)
            if row4["source_row_coordinates"] != [519, 33, 657, 347]:
                raise ValueError(
                    f"C: unexpected exact source row-4 coordinates {row4['source_row_coordinates']}"
                )
            diagnostics = {"source_row4_union_diagnostic": row4}
        if target == "A":
            diagnostics = {}
        targets[target] = {
            "target": target,
            "image": image,
            "source_bundle_path": str(path),
            "source_bundle_sha256": _file_hash(path),
            "request_id": expected_request,
            "source_scheduled_request": bundle["scheduled_request"],
            "source_execution_evidence": {
                "source_image_sha256": expected_digest,
                "source_width": bundle["execution_evidence"].get("source_width"),
                "source_height": bundle["execution_evidence"].get("source_height"),
                "decode_receipt_fingerprint": bundle["execution_evidence"].get("decode_receipt_fingerprint"),
                "execution_evidence_fingerprint": bundle["execution_evidence"].get("envelope_fingerprint"),
                "token_trace_sha256": bundle["execution_evidence"].get("token_trace_sha256"),
            },
            "row": row,
            "diagnostics": diagnostics,
            "references": references,
            "diagnostic_paths": [
                {
                    "label": "source_native_row",
                    "kind": "native_diagnostic",
                    "box": list(row["source_row_coordinates"]),
                    "valid": True,
                    "is_endpoint": True,
                    "shape": box_shape_metadata(row["source_row_coordinates"]),
                    "claim_contract": {
                        "owner_claim_admitted": False,
                        "extent_claim_admitted": False,
                        "admissible_claim_types": [
                            "native-path calibration only"
                        ],
                    },
                }
            ],
            "panels": enumerate_target_panels(target),
            "manual_adjudication_note": (
                "The reviewed fork row covers the visible fork head while the shaft remains visible; the COCO annotation supplies the reportable whole-object reference."
                if target == "A"
                else "The target, adjacent, and upper-row chair boxes are accepted individual COCO annotations inspected on the enlarged source image; source row 4 is retained exactly as a multi-row union control."
            ),
            "admissible_claim_types": (
                ["part-versus-visible-whole configuration preference", "late x2 extent-and-closure behavior"]
                if target == "A"
                else ["same-class owner preservation", "cross-row region aggregation"]
            ),
            "terminology": {"coordinate_order": "x1,y1,x2,y2", "x_axis": "horizontal", "y_axis": "vertical"},
        }
    return {
        "schema_version": "fixed_prefix_complete_box_coherence.fixture.v1",
        "source_jsonl": str(Path(source_jsonl).expanduser().resolve()),
        "coordinate_token_start": COORDINATE_TOKEN_START,
        "coordinate_token_end_exclusive": COORDINATE_TOKEN_END_EXCLUSIVE,
        "targets": targets,
    }


def _log_probability_summary(full_vocabulary_logits: torch.Tensor, token_id: int) -> dict[str, Any]:
    full = full_vocabulary_logits.detach().to(device="cpu", dtype=torch.float32).flatten()
    if full.ndim != 1 or full.numel() < COORDINATE_TOKEN_END_EXCLUSIVE or not bool(torch.isfinite(full).all()):
        raise ValueError("full-vocabulary logits must be finite and cover coordinates")
    full_logp = torch.log_softmax(full, dim=0)
    coordinate_logits = full[COORDINATE_TOKEN_START:COORDINATE_TOKEN_END_EXCLUSIVE]
    coordinate_logp = torch.log_softmax(coordinate_logits, dim=0)
    token_id = int(token_id)
    result = {
        "selected_token_id": token_id,
        "selected_coordinate_bin": coordinate_bin(token_id),
        "selected_full_vocabulary_logprob_float32": float(full_logp[token_id]),
        "selected_coordinate_normalized_logprob_float32": float(coordinate_logp[coordinate_bin(token_id)]),
        "coordinate_logits_float32": [float(value) for value in coordinate_logits.tolist()],
        "coordinate_logits_float32_sha256": hashlib.sha256(coordinate_logits.numpy().tobytes()).hexdigest(),
    }
    top_values, top_indices = torch.topk(coordinate_logits, k=min(20, coordinate_logits.numel()))
    result["top20"] = [{"coordinate_bin": int(index), "logit_float32": float(value)} for value, index in zip(top_values, top_indices)]
    result["top1_margin_float32"] = float(top_values[0] - top_values[1]) if len(top_values) > 1 else None
    return result


def score_primary_coordinate_path(slot_summaries: Sequence[Mapping[str, Any]]) -> float:
    if len(slot_summaries) != 4:
        raise ValueError("a complete box requires exactly four coordinate slots")
    return float(sum(float(slot["selected_full_vocabulary_logprob_float32"]) for slot in slot_summaries))


def score_coordinate_path(logits_by_slot: Sequence[torch.Tensor], selected_bins: Sequence[int], *, box_close_logit: torch.Tensor | None = None, box_end_token_id: int | None = None) -> dict[str, Any]:
    if len(logits_by_slot) != 4 or len(selected_bins) != 4:
        raise ValueError("coordinate path requires four logits and four selected bins")
    summaries = [_log_probability_summary(logits, coordinate_token_id(int(bin_value))) for logits, bin_value in zip(logits_by_slot, selected_bins)]
    result: dict[str, Any] = {
        "selected_coordinate_bins": [int(value) for value in selected_bins],
        "slot_summaries": summaries,
        "joint_primary_score_float32": score_primary_coordinate_path(summaries),
    }
    if box_close_logit is not None and box_end_token_id is not None:
        full = box_close_logit.detach().to(device="cpu", dtype=torch.float32).flatten()
        result["box_close_logprob_float32"] = float(torch.log_softmax(full, dim=0)[int(box_end_token_id)])
        result["box_end_token_id"] = int(box_end_token_id)
    return result


def _distance(prediction: Sequence[float], reference: Sequence[int]) -> float:
    px1, py1, px2, py2 = [float(value) for value in prediction]
    rx1, ry1, rx2, ry2 = [float(value) for value in reference]
    width, height = rx2 - rx1, ry2 - ry1
    if width <= 0 or height <= 0:
        raise ValueError("reference box must be non-degenerate")
    return 0.25 * (abs(px1 - rx1) / width + abs(px2 - rx2) / width + abs(py1 - ry1) / height + abs(py2 - ry2) / height)


def classify_released_box(
    prediction: Sequence[float],
    references: Sequence[Mapping[str, Any]],
    *,
    parser_valid: bool = True,
    forced_coordinate_count: int = 0,
    radius: float = CLASSIFICATION_RADIUS,
    margin: float = CLASSIFICATION_MARGIN,
) -> dict[str, Any]:
    """Attribute one box on separate configuration and endpoint-family axes."""

    if not isinstance(forced_coordinate_count, int) or not 0 <= forced_coordinate_count <= 4:
        raise ValueError("forced_coordinate_count must be in [0, 4]")
    endpoint_references = [
        ref
        for ref in references
        if bool(ref.get("is_endpoint", str(ref.get("kind", "visible")) != "hybrid"))
    ]
    if len(endpoint_references) != 2:
        raise ValueError("release attribution requires exactly two coherent endpoint references")
    if not parser_valid:
        return {
            "boundary_configuration_attribution": _invalid_attribution(
                "invalid_or_no_closure", radius=radius, margin=margin
            ),
            "endpoint_family_attribution": {
                **_invalid_attribution(
                    "invalid_or_no_closure", radius=radius, margin=margin
                ),
                "ambiguous": False,
                "outside": False,
            },
            "slot_edge_comparisons": [],
        }
    values = [float(value) for value in prediction]
    if len(values) != 4:
        raise ValueError("release attribution requires four predicted coordinates")
    boundary = _boundary_configuration_attribution(
        values,
        references,
        radius=radius,
        margin=margin,
    )
    endpoint = _endpoint_family_attribution(
        values,
        endpoint_references,
        radius=radius,
        margin=margin,
    )
    return {
        "boundary_configuration_attribution": boundary,
        "endpoint_family_attribution": endpoint,
        "slot_edge_comparisons": compare_released_edges_to_endpoints(
            values,
            endpoint_references,
            forced_coordinate_count=forced_coordinate_count,
        ),
    }


def _invalid_attribution(label: str, *, radius: float, margin: float) -> dict[str, Any]:
    return {
        "label": label,
        "accepted": False,
        "precedence": "parser_invalid",
        "nearest_reference": None,
        "nearest_kind": None,
        "nearest_distance": None,
        "second_distance": None,
        "margin_over_second": None,
        "all_distances": [],
        "radius": float(radius),
        "required_margin": float(margin),
    }


def _reference_distances(
    prediction: Sequence[float], references: Sequence[Mapping[str, Any]]
) -> list[tuple[float, str, str, bool]]:
    return sorted(
        (
            (
                _distance(prediction, ref["box"]),
                str(ref["label"]),
                str(ref.get("kind", "visible")),
                bool(ref.get("is_endpoint", str(ref.get("kind", "visible")) != "hybrid")),
            )
            for ref in references
        ),
        key=lambda item: (item[0], item[1]),
    )


def _distance_receipt(
    distances: Sequence[tuple[float, str, str, bool]],
    *,
    label: str,
    accepted: bool,
    precedence: str,
    radius: float,
    margin: float,
) -> dict[str, Any]:
    if not distances:
        raise ValueError("classification requires at least one frozen reference")
    nearest = distances[0]
    second = distances[1][0] if len(distances) > 1 else math.inf
    return {
        "label": label,
        "accepted": bool(accepted),
        "precedence": precedence,
        "nearest_reference": nearest[1],
        "nearest_kind": nearest[2],
        "nearest_distance": float(nearest[0]),
        "second_distance": None if math.isinf(second) else float(second),
        "margin_over_second": None if math.isinf(second) else float(second - nearest[0]),
        "all_distances": [
            {
                "label": item[1],
                "kind": item[2],
                "is_endpoint": item[3],
                "distance": float(item[0]),
            }
            for item in distances
        ],
        "radius": float(radius),
        "required_margin": float(margin),
    }


def _boundary_configuration_attribution(
    prediction: Sequence[float],
    references: Sequence[Mapping[str, Any]],
    *,
    radius: float,
    margin: float,
) -> dict[str, Any]:
    distances = _reference_distances(prediction, references)
    exact_endpoints = [item for item in distances if item[0] == 0.0 and item[3]]
    if exact_endpoints:
        selected = exact_endpoints[0]
        distances = [selected, *[item for item in distances if item is not selected]]
        return _distance_receipt(
            distances,
            label=selected[1],
            accepted=True,
            precedence="exact_coherent_endpoint",
            radius=radius,
            margin=margin,
        )
    exact_hybrids = [
        item for item in distances if item[0] == 0.0 and not item[3] and item[2] == "hybrid"
    ]
    if exact_hybrids:
        selected = exact_hybrids[0]
        distances = [selected, *[item for item in distances if item is not selected]]
        return _distance_receipt(
            distances,
            label="boundary_hybrid",
            accepted=True,
            precedence="exact_nonendpoint_hybrid",
            radius=radius,
            margin=margin,
        )
    if not distances:
        raise ValueError("classification requires at least one frozen reference")
    nearest = distances[0]
    second = distances[1][0] if len(distances) > 1 else math.inf
    accepted = nearest[0] <= float(radius) and second - nearest[0] >= float(margin)
    if accepted:
        label = "boundary_hybrid" if nearest[2] == "hybrid" else nearest[1]
    elif nearest[0] <= float(radius) and nearest[2] == "hybrid":
        label = "boundary_hybrid"
    else:
        label = "other_same_category_region"
    return _distance_receipt(
        distances,
        label=label,
        accepted=accepted,
        precedence="full_boundary_lattice_distance",
        radius=radius,
        margin=margin,
    )


def _endpoint_family_attribution(
    prediction: Sequence[float],
    endpoints: Sequence[Mapping[str, Any]],
    *,
    radius: float,
    margin: float,
) -> dict[str, Any]:
    distances = _reference_distances(prediction, endpoints)
    exact = [item for item in distances if item[0] == 0.0]
    if exact:
        selected = exact[0]
        distances = [selected, *[item for item in distances if item is not selected]]
        result = _distance_receipt(
            distances,
            label=selected[1],
            accepted=True,
            precedence="exact_coherent_endpoint",
            radius=radius,
            margin=margin,
        )
        return {**result, "ambiguous": False, "outside": False}
    nearest = distances[0]
    second = distances[1][0]
    accepted = nearest[0] <= float(radius) and second - nearest[0] >= float(margin)
    ambiguous = nearest[0] <= float(radius) and not accepted
    outside = nearest[0] > float(radius)
    label = (
        nearest[1]
        if accepted
        else "ambiguous_between_endpoint_families"
        if ambiguous
        else "outside_endpoint_families"
    )
    result = _distance_receipt(
        distances,
        label=label,
        accepted=accepted,
        precedence="endpoint_only_distance",
        radius=radius,
        margin=margin,
    )
    return {**result, "ambiguous": bool(ambiguous), "outside": bool(outside)}


def compare_released_edges_to_endpoints(
    prediction: Sequence[float],
    endpoints: Sequence[Mapping[str, Any]],
    *,
    forced_coordinate_count: int,
) -> list[dict[str, Any]]:
    """Report raw per-edge errors without converting them into owner claims."""

    if len(prediction) != 4 or len(endpoints) != 2:
        raise ValueError("edge comparison requires one box and two coherent endpoints")
    if not 0 <= int(forced_coordinate_count) <= 4:
        raise ValueError("forced_coordinate_count must be in [0, 4]")
    left, right = endpoints
    left_box = [float(value) for value in left["box"]]
    right_box = [float(value) for value in right["box"]]
    output: list[dict[str, Any]] = []
    for slot, slot_name in enumerate(("x1", "y1", "x2", "y2")):
        value = float(prediction[slot])
        left_error = abs(value - left_box[slot])
        right_error = abs(value - right_box[slot])
        closer = (
            str(left["label"])
            if left_error < right_error
            else str(right["label"])
            if right_error < left_error
            else "tie"
        )
        output.append(
            {
                "slot_index": slot,
                "slot": slot_name,
                "generation_role": "forced" if slot < forced_coordinate_count else "released",
                "predicted_coordinate_bin": value,
                "left_endpoint_label": str(left["label"]),
                "left_endpoint_coordinate_bin": left_box[slot],
                "absolute_error_to_left_endpoint": float(left_error),
                "right_endpoint_label": str(right["label"]),
                "right_endpoint_coordinate_bin": right_box[slot],
                "absolute_error_to_right_endpoint": float(right_error),
                "closer_endpoint": closer,
                "endpoint_separation": float(abs(left_box[slot] - right_box[slot])),
            }
        )
    return output


def summarize_released_owner_crossover(
    records: Sequence[Mapping[str, Any]],
    *,
    owner_discriminative_slots: Sequence[str],
) -> dict[str, Any]:
    """Count causal owner crossover only on predeclared semantic owner slots."""

    slot_names = [str(value) for value in owner_discriminative_slots]
    unknown = sorted(set(slot_names) - {"x1", "y1", "x2", "y2"})
    if unknown:
        raise ValueError(f"unknown owner-discriminative coordinate slots: {unknown}")
    if len(set(slot_names)) != len(slot_names):
        raise ValueError("owner-discriminative coordinate slots must be unique")
    summary: dict[str, Any] = {
        "admission_status": "admitted" if slot_names else "not_admitted",
        "owner_discriminative_slots": slot_names,
        "eligible_released_owner_discriminative_edge_count": 0,
        "closer_to_cued_endpoint_count": 0,
        "crossed_to_other_endpoint_count": 0,
        "tie_count": 0,
        "by_forced_coordinate_count": {},
    }
    if not slot_names:
        return summary
    admitted = set(slot_names)
    for record in records:
        arm = record.get("arm")
        path = record.get("path")
        classification = record.get("classification")
        if not isinstance(arm, Mapping) or not isinstance(path, Mapping) or not isinstance(
            classification, Mapping
        ):
            raise ValueError("owner crossover record lacks arm, path, or classification")
        forced_count = int(arm.get("forced_coordinate_count", -1))
        if forced_count <= 0:
            continue
        force_key = str(forced_count)
        force_summary = summary["by_forced_coordinate_count"].setdefault(
            force_key,
            {
                "eligible_released_owner_discriminative_edge_count": 0,
                "closer_to_cued_endpoint_count": 0,
                "crossed_to_other_endpoint_count": 0,
                "tie_count": 0,
            },
        )
        edges = classification.get("slot_edge_comparisons", [])
        if not isinstance(edges, list):
            raise ValueError("owner crossover classification lacks slot edge comparisons")
        for edge in edges:
            if not isinstance(edge, Mapping):
                raise ValueError("owner crossover edge comparison must be an object")
            if (
                str(edge.get("slot")) not in admitted
                or edge.get("generation_role") != "released"
                or float(edge.get("endpoint_separation", 0.0)) == 0.0
            ):
                continue
            summary["eligible_released_owner_discriminative_edge_count"] += 1
            force_summary["eligible_released_owner_discriminative_edge_count"] += 1
            closer = str(edge.get("closer_endpoint"))
            if closer == str(path.get("label")):
                key = "closer_to_cued_endpoint_count"
            elif closer == "tie":
                key = "tie_count"
            else:
                key = "crossed_to_other_endpoint_count"
            summary[key] += 1
            force_summary[key] += 1
    return summary


def build_coordinate_release_arms(
    pre_x1_token_ids: Sequence[int],
    path_bins: Sequence[int],
    *,
    claim_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    if len(path_bins) != 4:
        raise ValueError("release path requires four coordinate bins")
    coordinate_ids = [coordinate_token_id(int(value)) for value in path_bins]
    arms = []
    for forced_count in range(5):
        arms.append({"arm": "free" if forced_count == 0 else f"force_first_{forced_count}_coordinates", "forced_coordinate_count": forced_count, "prompt_token_ids": [int(value) for value in pre_x1_token_ids] + coordinate_ids[:forced_count], "forced_coordinate_token_ids": coordinate_ids[:forced_count], "released_coordinate_slots": list(range(forced_count, 4)), "repetition_penalty": 1.0, "claim_contract": dict(claim_contract or {})})
    return arms


def select_release_panel(
    target_payload: Mapping[str, Any],
    panel_name: str,
    *,
    path_labels: Sequence[str] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Select one frozen pair and its coherent endpoint paths."""

    panels = target_payload.get("panels")
    if not isinstance(panels, list):
        raise ValueError("target fixture must contain panel records")
    matches = [panel for panel in panels if str(panel.get("panel")) == panel_name]
    if len(matches) != 1:
        available = sorted(str(panel.get("panel")) for panel in panels)
        raise ValueError(
            f"release panel {panel_name!r} must resolve exactly once; available={available}"
        )
    panel = dict(matches[0])
    hybrids = panel.get("hybrids")
    if not isinstance(hybrids, list):
        raise ValueError(f"release panel {panel_name!r} lacks frozen hybrid paths")
    endpoints = [
        dict(path)
        for path in hybrids
        if isinstance(path, Mapping)
        and bool(path.get("valid", False))
        and bool(path.get("is_endpoint", False))
    ]
    if path_labels:
        requested = {str(value) for value in path_labels}
        endpoints = [path for path in endpoints if str(path.get("label")) in requested]
        found = {str(path.get("label")) for path in endpoints}
        missing = sorted(requested - found)
        if missing:
            raise ValueError(
                f"release path labels are not coherent endpoints of {panel_name!r}: {missing}"
            )
    if not endpoints:
        raise ValueError(f"release panel {panel_name!r} has no selected coherent endpoints")
    endpoints.sort(key=lambda path: tuple(int(value) for value in path["parent_choices"]))
    return panel, endpoints


def classifier_references_for_panel(panel: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Freeze endpoint and non-endpoint hybrid families used for release labels."""

    hybrids = panel.get("hybrids")
    if not isinstance(hybrids, list):
        raise ValueError("release classification panel lacks hybrids")
    references: list[dict[str, Any]] = []
    for path in hybrids:
        if not isinstance(path, Mapping) or not bool(path.get("valid", False)):
            continue
        box = [int(value) for value in path.get("box", [])]
        if len(box) != 4:
            raise ValueError("release classification reference must have four coordinates")
        references.append(
            {
                "label": str(path.get("label", "")),
                "kind": str(path.get("kind", "visible")),
                "box": box,
                "is_endpoint": bool(path.get("is_endpoint", False)),
            }
        )
    if not references:
        raise ValueError("release classification requires at least one valid reference")
    return references


def parse_coordinate_release_suffix(
    generated_token_ids: Sequence[int],
    *,
    forced_coordinate_bins: Sequence[int],
    box_end_token_id: int,
) -> dict[str, Any]:
    """Parse the exact remaining coordinates followed by a natural box close."""

    forced = [int(value) for value in forced_coordinate_bins]
    if len(forced) > 4:
        raise ValueError("at most four coordinate bins may be forced")
    for value in forced:
        coordinate_token_id(value)
    generated = [int(value) for value in generated_token_ids]
    required_coordinate_count = 4 - len(forced)
    released_bins: list[int] = []
    failure: str | None = None
    for offset in range(required_coordinate_count):
        if offset >= len(generated):
            failure = f"missing_released_coordinate_{len(forced) + offset}"
            break
        try:
            released_bins.append(coordinate_bin(generated[offset]))
        except ValueError:
            failure = f"non_coordinate_at_released_slot_{len(forced) + offset}"
            break
    closure_index = required_coordinate_count
    natural_box_closure = (
        failure is None
        and closure_index < len(generated)
        and generated[closure_index] == int(box_end_token_id)
    )
    if failure is None and not natural_box_closure:
        failure = "missing_natural_box_closure"
    completed = [*forced, *released_bins]
    valid_geometry = (
        len(completed) == 4
        and completed[0] < completed[2]
        and completed[1] < completed[3]
    )
    if failure is None and not valid_geometry:
        failure = "invalid_completed_geometry"
    parser_valid = failure is None and natural_box_closure and valid_geometry
    return {
        "parser_valid": bool(parser_valid),
        "failure": failure,
        "forced_coordinate_bins": forced,
        "released_coordinate_bins": released_bins,
        "completed_coordinate_bins": completed if len(completed) == 4 else None,
        "required_released_coordinate_count": required_coordinate_count,
        "natural_box_closure": bool(natural_box_closure),
        "box_end_token_id": int(box_end_token_id),
        "box_end_generated_index": closure_index if natural_box_closure else None,
        "generated_token_ids_through_expected_closure": generated[: closure_index + 1],
        "trailing_generated_token_ids": generated[closure_index + 1 :] if natural_box_closure else [],
    }


def paired_release_sampling_seeds(seed_root: int, sample_count: int) -> list[int]:
    """Derive request-owned seeds shared by every path and force-count cell."""

    if not isinstance(sample_count, int) or isinstance(sample_count, bool) or sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")
    seeds: list[int] = []
    for sample_index in range(sample_count):
        payload = f"fixed-prefix-coordinate-release:{int(seed_root)}:{sample_index}".encode()
        seeds.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % ((1 << 63) - 1))
    return seeds


def load_verified_fixture(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Load a fixture only after verifying its exact on-disk digest."""

    fixture_path = Path(path).expanduser().resolve(strict=True)
    expected = str(expected_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("fixture-sha256 must be a 64-character lowercase hexadecimal digest")
    actual = _file_hash(fixture_path)
    if actual != expected:
        raise ValueError(
            f"fixture sha256 mismatch: expected {expected}, observed {actual}"
        )
    payload = json.loads(fixture_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or not str(payload.get("schema_version", "")).startswith(
        "fixed_prefix_complete_box_coherence.fixture."
    ):
        raise ValueError("fixture has an unsupported fixed-prefix coherence schema")
    if not isinstance(payload.get("targets"), Mapping):
        raise ValueError("fixture targets must be an object")
    return dict(payload)


def deduplicate_panel_paths(target_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Deduplicate identical boxes while preserving every panel provenance."""

    by_box: dict[tuple[int, ...], dict[str, Any]] = {}
    panels = target_payload.get("panels")
    if not isinstance(panels, list):
        raise ValueError("target fixture must contain panel records")
    for panel in panels:
        if not isinstance(panel, Mapping):
            raise ValueError("panel record must be an object")
        panel_name = str(panel.get("panel", ""))
        paths = panel.get("hybrids")
        if not isinstance(paths, list):
            raise ValueError(f"panel {panel_name!r} lacks hybrids")
        for path in paths:
            if not isinstance(path, Mapping):
                raise ValueError(f"panel {panel_name!r} has a non-object path")
            box = tuple(int(value) for value in path.get("box", []))
            if len(box) != 4:
                raise ValueError(f"panel {panel_name!r} has a non-four-coordinate path")
            entry = by_box.setdefault(
                box,
                {
                    "box": list(box),
                    "valid": bool(path.get("valid", False)),
                    "is_endpoint": bool(path.get("is_endpoint", False)),
                    "shape": path.get("shape"),
                    "labels": [],
                    "panel_provenance": [],
                },
            )
            entry["valid"] = bool(entry["valid"] or path.get("valid", False))
            entry["is_endpoint"] = bool(entry["is_endpoint"] or path.get("is_endpoint", False))
            label = str(path.get("label", ""))
            if label and label not in entry["labels"]:
                entry["labels"].append(label)
            provenance = {"panel": panel_name, "label": label}
            if provenance not in entry["panel_provenance"]:
                entry["panel_provenance"].append(provenance)
    diagnostics = target_payload.get("diagnostic_paths", [])
    if not isinstance(diagnostics, list):
        raise ValueError("target fixture diagnostic_paths must be a list")
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, Mapping):
            raise ValueError("diagnostic path must be an object")
        box = tuple(int(value) for value in diagnostic.get("box", []))
        if len(box) != 4:
            raise ValueError("diagnostic path must have four coordinates")
        entry = by_box.setdefault(
            box,
            {
                "box": list(box),
                "valid": bool(diagnostic.get("valid", False)),
                "is_endpoint": bool(diagnostic.get("is_endpoint", True)),
                "shape": diagnostic.get("shape"),
                "labels": [],
                "panel_provenance": [],
            },
        )
        label = str(diagnostic.get("label", "source_native_row"))
        if label not in entry["labels"]:
            entry["labels"].append(label)
        provenance = {"panel": "native_path_diagnostic", "label": label}
        if provenance not in entry["panel_provenance"]:
            entry["panel_provenance"].append(provenance)
    return [by_box[key] for key in sorted(by_box)]


@contextlib.contextmanager
def _temporary_cwd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def run_teacher_forced_scores(
    fixture: Mapping[str, Any],
    *,
    target: str,
    config_path: Path = DEFAULT_CONFIG,
    source_jsonl: Path = DEFAULT_SOURCE_JSONL,
) -> dict[str, Any]:
    """Score every unique configured complete box at physical batch size one."""

    from scripts.research.run_batch_coordinate_logit_invariance import (
        _direct_full_prefix_logits,
    )
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeGenerationPolicy, DecodeRequest, HFGenerateBackend
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.runtime import assemble_runtime

    targets = fixture.get("targets")
    if not isinstance(targets, Mapping) or target not in targets:
        raise ValueError(f"fixture does not contain target {target!r}")
    target_payload = targets[target]
    if not isinstance(target_payload, Mapping):
        raise ValueError("target fixture payload must be an object")
    config_path = Path(config_path).expanduser().resolve(strict=True)
    source_jsonl = Path(source_jsonl).expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    image_id = str(target_payload["image"]["image_id"])
    raw = next(
        (
            item
            for item in load_raw_examples(source_jsonl)
            if str(item.metadata.get("source", {}).get("image_id")) == image_id
        ),
        None,
    )
    if raw is None:
        raise ValueError(f"source JSONL lacks image {image_id}")
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    row_payload = target_payload["row"]
    generated_prefix = [int(value) for value in row_payload["pre_x1_generated_token_ids"]]
    continuation_text = qwen.tokenizer.decode(
        generated_prefix,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    prompt = build_prompt_record(
        raw,
        _template_config(resolved.config),
        processor=qwen.processor,
        row_index=0,
        assistant_continuation=AssistantContinuation(text=continuation_text),
    )
    frozen_prompt = [int(value) for value in row_payload["pre_x1_prompt_token_ids"]]
    if list(prompt.prompt_token_ids) != frozen_prompt:
        raise RuntimeError("rebuilt exact pre-x1 prompt differs from frozen artifact prefix")
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    model_inputs = image_plan.model_inputs_by_row_id[prompt.row_id]
    policy = DecodeGenerationPolicy.greedy(max_new_tokens=8, repetition_penalty=1.0)
    box_end_token_id = int(qwen.tokenizer.convert_tokens_to_ids("<|box_end|>"))
    paths = deduplicate_panel_paths(target_payload)
    records: list[dict[str, Any]] = []
    for path_index, path in enumerate(paths):
        if not path["valid"]:
            records.append(
                {
                    "path_index": path_index,
                    "path": path,
                    "score": None,
                    "skipped_reason": "invalid_geometry",
                }
            )
            continue
        prefix = list(frozen_prompt)
        logits_by_slot: list[torch.Tensor] = []
        slot_runtime_receipts: list[dict[str, Any]] = []
        for slot, coordinate in enumerate(path["box"]):
            request = DecodeRequest(
                request_id=f"fixed-box-coherence:{target}:{path_index}:slot-{slot}",
                prompt_token_ids=prefix,
                model_inputs=model_inputs,
                generation_policy=policy,
            )
            logits, _, runtime_receipt = _direct_full_prefix_logits(
                backend=backend,
                requests=[request],
            )
            logits_by_slot.append(logits[0])
            slot_runtime_receipts.append(runtime_receipt)
            prefix.append(coordinate_token_id(int(coordinate)))
        close_request = DecodeRequest(
            request_id=f"fixed-box-coherence:{target}:{path_index}:box-close",
            prompt_token_ids=prefix,
            model_inputs=model_inputs,
            generation_policy=policy,
        )
        close_logits, _, close_runtime_receipt = _direct_full_prefix_logits(
            backend=backend,
            requests=[close_request],
        )
        score = score_coordinate_path(
            logits_by_slot,
            path["box"],
            box_close_logit=close_logits[0],
            box_end_token_id=box_end_token_id,
        )
        records.append(
            {
                "path_index": path_index,
                "path": path,
                "score": score,
                "slot_runtime_receipts": slot_runtime_receipts,
                "box_close_runtime_receipt": close_runtime_receipt,
            }
        )
    valid_scores = [
        float(record["score"]["joint_primary_score_float32"])
        for record in records
        if record["score"] is not None
    ]
    return {
        "schema_version": "fixed_prefix_complete_box_coherence.teacher_forced_scores.v1",
        "target": target,
        "image_id": image_id,
        "config_path": str(config_path),
        "source_jsonl": str(source_jsonl),
        "physical_batch_size": 1,
        "repetition_penalty": 1.0,
        "prompt_token_ids_sha256": _json_hash(frozen_prompt),
        "runtime_model_identity": model_identity,
        "runtime_tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": generation_fingerprint,
        "unique_path_count": len(paths),
        "valid_path_count": len(valid_scores),
        "summary": {
            "joint_primary_score_min_float32": min(valid_scores) if valid_scores else None,
            "joint_primary_score_max_float32": max(valid_scores) if valid_scores else None,
        },
        "records": records,
    }


def _request_identifier_fragment(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(value)).strip("-") or "unnamed"


def run_progressive_coordinate_release(
    fixture: Mapping[str, Any],
    *,
    target: str,
    panel_name: str,
    path_labels: Sequence[str] | None = None,
    forced_coordinate_counts: Sequence[int] | None = None,
    decode_mode: str = "greedy",
    sample_count: int = 16,
    seed_root: int = 20260716,
    sampling_temperature: float = DEFAULT_RELEASE_TEMPERATURE,
    top_p: float = DEFAULT_RELEASE_TOP_P,
    max_new_tokens: int = RELEASE_MAX_NEW_TOKENS,
    physical_batch_size: int = 1,
    sampled_runtime_attestation: Path | None = None,
    config_path: Path = DEFAULT_CONFIG,
    source_jsonl: Path = DEFAULT_SOURCE_JSONL,
) -> dict[str, Any]:
    """Generate the native suffix after forcing zero through four coordinates."""

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        DecodeRequest,
        HFGenerateBackend,
        load_and_rebind_sampled_runtime_attestation_aggregate,
    )
    from src.inference.image_plan import (
        materialize_image_plan_batch,
        verify_processor_model_vision_parity,
    )
    from src.inference.pipeline import (
        _processor_config,
        _template_config,
        _tokenizer_identity,
    )
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.runtime import assemble_runtime

    targets = fixture.get("targets")
    if not isinstance(targets, Mapping) or target not in targets:
        raise ValueError(f"fixture does not contain target {target!r}")
    target_payload = targets[target]
    if not isinstance(target_payload, Mapping):
        raise ValueError("target fixture payload must be an object")
    panel, endpoint_paths = select_release_panel(
        target_payload,
        panel_name,
        path_labels=path_labels,
    )
    classifier_references = classifier_references_for_panel(panel)
    selected_force_counts = (
        {int(value) for value in forced_coordinate_counts}
        if forced_coordinate_counts
        else set(range(5))
    )
    if not selected_force_counts or not selected_force_counts <= set(range(5)):
        raise ValueError("forced_coordinate_counts must be a nonempty subset of 0..4")
    if decode_mode not in {"greedy", "sampled"}:
        raise ValueError("decode_mode must be greedy or sampled")
    if not isinstance(physical_batch_size, int) or physical_batch_size <= 0:
        raise ValueError("physical_batch_size must be a positive integer")
    if not isinstance(max_new_tokens, int) or max_new_tokens < RELEASE_MAX_NEW_TOKENS:
        raise ValueError(
            f"max_new_tokens must be at least {RELEASE_MAX_NEW_TOKENS} so a fully free box can close"
        )
    if decode_mode == "sampled":
        if sampled_runtime_attestation is None:
            raise ValueError("sampled release requires sampled_runtime_attestation")
        attestation_path = Path(sampled_runtime_attestation).expanduser().resolve(strict=True)
        seeds: list[int | None] = paired_release_sampling_seeds(seed_root, sample_count)
        policy = DecodeGenerationPolicy.sampled(
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
            temperature=float(sampling_temperature),
            top_p=float(top_p),
        )
    else:
        attestation_path = None
        seeds = [None]
        policy = DecodeGenerationPolicy.greedy(
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.0,
        )

    config_path = Path(config_path).expanduser().resolve(strict=True)
    source_jsonl = Path(source_jsonl).expanduser().resolve(strict=True)
    frozen_source_jsonl = Path(str(fixture.get("source_jsonl", ""))).expanduser().resolve()
    if frozen_source_jsonl != source_jsonl:
        raise ValueError(
            f"release source JSONL differs from fixture: {source_jsonl} != {frozen_source_jsonl}"
        )
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    image_id = str(target_payload["image"]["image_id"])
    raw = next(
        (
            item
            for item in load_raw_examples(source_jsonl)
            if str(item.metadata.get("source", {}).get("image_id")) == image_id
        ),
        None,
    )
    if raw is None:
        raise ValueError(f"source JSONL lacks image {image_id}")
    if _file_hash(Path(raw.image.path).resolve(strict=True)) != str(
        target_payload["image"]["image_sha256"]
    ):
        raise ValueError("release source image differs from the frozen target fixture")

    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    template_config = _template_config(resolved.config)
    row_payload = target_payload["row"]
    generated_prefix = [int(value) for value in row_payload["pre_x1_generated_token_ids"]]
    frozen_prompt = [int(value) for value in row_payload["pre_x1_prompt_token_ids"]]
    base_continuation_text = qwen.tokenizer.decode(
        generated_prefix,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    base_prompt = build_prompt_record(
        raw,
        template_config,
        processor=qwen.processor,
        row_index=0,
        assistant_continuation=AssistantContinuation(text=base_continuation_text),
    )
    if list(base_prompt.prompt_token_ids) != frozen_prompt:
        raise RuntimeError("rebuilt exact pre-x1 prompt differs from frozen artifact prefix")
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = image_plan.model_inputs_by_row_id[base_prompt.row_id]
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    capability = None
    if decode_mode == "sampled":
        assert attestation_path is not None
        capability = load_and_rebind_sampled_runtime_attestation_aggregate(
            attestation_path,
            decode_generation_policy_fingerprint=policy.fingerprint,
            backend=backend,
        )
    box_end_token_id = int(qwen.tokenizer.convert_tokens_to_ids("<|box_end|>"))

    request_specs: list[dict[str, Any]] = []
    requests: list[Any] = []
    for path in endpoint_paths:
        path_label = str(path["label"])
        arms = build_coordinate_release_arms(
            frozen_prompt,
            path["box"],
            claim_contract=panel.get("claim_contract"),
        )
        arms = [
            arm
            for arm in arms
            if int(arm["forced_coordinate_count"]) in selected_force_counts
        ]
        for arm in arms:
            forced_ids = [int(value) for value in arm["forced_coordinate_token_ids"]]
            continuation_ids = [*generated_prefix, *forced_ids]
            continuation_text = qwen.tokenizer.decode(
                continuation_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            prompt = build_prompt_record(
                raw,
                template_config,
                processor=qwen.processor,
                row_index=0,
                assistant_continuation=AssistantContinuation(text=continuation_text),
            )
            expected_prompt = [*frozen_prompt, *forced_ids]
            if list(prompt.prompt_token_ids) != expected_prompt:
                raise RuntimeError(
                    f"canonical prompt reconstruction failed for {path_label} force-count "
                    f"{arm['forced_coordinate_count']}"
                )
            for sample_index, seed in enumerate(seeds):
                request_id = (
                    "fixed-box-release:"
                    f"{target}:{_request_identifier_fragment(panel_name)}:"
                    f"{_request_identifier_fragment(path_label)}:"
                    f"force-{arm['forced_coordinate_count']}:"
                    f"{decode_mode}-{sample_index}"
                )
                request = DecodeRequest(
                    request_id=request_id,
                    prompt_token_ids=expected_prompt,
                    model_inputs=model_inputs,
                    generation_policy=policy,
                    sampling_seed=seed,
                )
                requests.append(request)
                request_specs.append(
                    {
                        "request_id": request_id,
                        "path": {
                            "label": path_label,
                            "kind": str(path.get("kind", "visible")),
                            "box": [int(value) for value in path["box"]],
                            "parent_choices": [int(value) for value in path["parent_choices"]],
                        },
                        "arm": {
                            "arm": str(arm["arm"]),
                            "forced_coordinate_count": int(arm["forced_coordinate_count"]),
                            "forced_coordinate_bins": [
                                coordinate_bin(token_id) for token_id in forced_ids
                            ],
                            "forced_coordinate_token_ids": forced_ids,
                            "released_coordinate_slots": list(arm["released_coordinate_slots"]),
                            "claim_contract": dict(arm["claim_contract"]),
                        },
                        "sample_index": sample_index,
                        "sampling_seed": seed,
                        "prompt": {
                            "prompt_token_count": len(expected_prompt),
                            "prompt_token_ids_sha256": _json_hash(expected_prompt),
                            "full_prompt_fingerprint": prompt.full_prompt_fingerprint,
                            "continuation_text_sha256": prompt.continuation_text_sha256,
                        },
                    }
                )

    results: list[Any] = []
    for offset in range(0, len(requests), physical_batch_size):
        batch = requests[offset : offset + physical_batch_size]
        if decode_mode == "sampled":
            assert capability is not None
            results.extend(
                backend.generate_batch_with_verified_runtime_attestation(
                    batch,
                    model_identity=model_identity,
                    tokenizer_identity=tokenizer_identity,
                    generation_config_fingerprint=generation_fingerprint,
                    verified_runtime_attestation=capability,
                )
            )
        else:
            results.extend(
                backend.generate_batch(
                    batch,
                    model_identity=model_identity,
                    tokenizer_identity=tokenizer_identity,
                    generation_config_fingerprint=generation_fingerprint,
                )
            )
    if len(results) != len(request_specs):
        raise RuntimeError("release backend result count differs from request plan")

    records: list[dict[str, Any]] = []
    boundary_counts: dict[str, int] = {}
    endpoint_counts: dict[str, int] = {}
    boundary_counts_by_force: dict[str, dict[str, int]] = {}
    endpoint_counts_by_force: dict[str, dict[str, int]] = {}
    for spec, result in zip(request_specs, results):
        generated_ids = [int(value) for value in result.generated_token_ids]
        parsed = parse_coordinate_release_suffix(
            generated_ids,
            forced_coordinate_bins=spec["arm"]["forced_coordinate_bins"],
            box_end_token_id=box_end_token_id,
        )
        prediction = parsed["completed_coordinate_bins"] or [0, 0, 1, 1]
        classification = classify_released_box(
            prediction,
            classifier_references,
            parser_valid=bool(parsed["parser_valid"]),
            forced_coordinate_count=int(spec["arm"]["forced_coordinate_count"]),
        )
        boundary_label = str(
            classification["boundary_configuration_attribution"]["label"]
        )
        endpoint_label = str(classification["endpoint_family_attribution"]["label"])
        boundary_counts[boundary_label] = boundary_counts.get(boundary_label, 0) + 1
        endpoint_counts[endpoint_label] = endpoint_counts.get(endpoint_label, 0) + 1
        force_key = str(spec["arm"]["forced_coordinate_count"])
        boundary_force = boundary_counts_by_force.setdefault(force_key, {})
        boundary_force[boundary_label] = boundary_force.get(boundary_label, 0) + 1
        endpoint_force = endpoint_counts_by_force.setdefault(force_key, {})
        endpoint_force[endpoint_label] = endpoint_force.get(endpoint_label, 0) + 1
        decode_artifact = result.to_artifact_dict()
        records.append(
            {
                **spec,
                "raw_suffix_token_ids": generated_ids,
                "raw_suffix_token_text": [
                    str(value)
                    for value in qwen.tokenizer.convert_ids_to_tokens(generated_ids)
                ],
                "raw_suffix_text": qwen.tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                "suffix_parse": parsed,
                "classification": classification,
                "decode_result": decode_artifact,
            }
        )

    panel_claim_contract = dict(panel.get("claim_contract", {}))
    released_owner_crossover = summarize_released_owner_crossover(
        records,
        owner_discriminative_slots=panel_claim_contract.get(
            "owner_discriminative_slots", []
        ),
    )

    return {
        "schema_version": "fixed_prefix_complete_box_coherence.progressive_release.v2",
        "target": target,
        "image_id": image_id,
        "panel": panel_name,
        "selected_path_labels": [str(path["label"]) for path in endpoint_paths],
        "selected_forced_coordinate_counts": sorted(selected_force_counts),
        "panel_claim_contract": panel_claim_contract,
        "classifier_references": classifier_references,
        "config_path": str(config_path),
        "source_jsonl": str(source_jsonl),
        "physical_batch_size": physical_batch_size,
        "decode_policy": policy.to_artifact_dict(),
        "decode_generation_policy_fingerprint": policy.fingerprint,
        "sample_count_per_cell": len(seeds),
        "seed_root": int(seed_root) if decode_mode == "sampled" else None,
        "paired_sampling_seeds": seeds if decode_mode == "sampled" else [],
        "sampled_runtime_attestation_path": (
            str(attestation_path) if attestation_path is not None else None
        ),
        "base_prompt": {
            "prompt_token_count": len(frozen_prompt),
            "prompt_token_ids_sha256": _json_hash(frozen_prompt),
            "frozen_prompt_token_ids_sha256": row_payload["pre_x1_prompt_token_ids_sha256"],
            "full_prompt_fingerprint": base_prompt.full_prompt_fingerprint,
            "continuation_text_sha256": base_prompt.continuation_text_sha256,
            "exact_frozen_prefix_verified": True,
        },
        "runtime_model_identity": model_identity,
        "runtime_tokenizer_identity": tokenizer_identity,
        "generation_config_fingerprint": generation_fingerprint,
        "request_count": len(records),
        "summary": {
            "boundary_configuration_attribution_counts": boundary_counts,
            "boundary_configuration_attribution_counts_by_forced_coordinate_count": (
                boundary_counts_by_force
            ),
            "endpoint_family_attribution_counts": endpoint_counts,
            "endpoint_family_attribution_counts_by_forced_coordinate_count": (
                endpoint_counts_by_force
            ),
            "released_owner_discriminative_edge_crossover": released_owner_crossover,
            "valid_natural_closure_count": sum(
                bool(record["suffix_parse"]["parser_valid"]) for record in records
            ),
        },
        "records": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("fixture", "score", "release"), default="fixture"
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--fixture-sha256")
    parser.add_argument("--target", choices=("A", "C"), default="A")
    parser.add_argument("--panel")
    parser.add_argument("--path-label", action="append", default=[])
    parser.add_argument("--forced-coordinate-count", action="append", type=int, default=[])
    parser.add_argument("--decode-mode", choices=("greedy", "sampled"), default="greedy")
    parser.add_argument("--sample-count", type=int, default=16)
    parser.add_argument("--seed-root", type=int, default=20260716)
    parser.add_argument("--sampling-temperature", type=float, default=DEFAULT_RELEASE_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_RELEASE_TOP_P)
    parser.add_argument("--max-new-tokens", type=int, default=RELEASE_MAX_NEW_TOKENS)
    parser.add_argument("--physical-batch-size", type=int, default=1)
    parser.add_argument("--sampled-runtime-attestation", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.phase == "fixture":
        result = materialize_fixture(source_jsonl=args.source_jsonl)
    elif args.phase == "score":
        if args.fixture is None or args.fixture_sha256 is None:
            raise SystemExit("--phase score requires --fixture and --fixture-sha256")
        # This verification happens before run_teacher_forced_scores imports or
        # constructs the model runtime.
        fixture = load_verified_fixture(args.fixture, args.fixture_sha256)
        result = run_teacher_forced_scores(
            fixture,
            target=args.target,
            config_path=args.infer_config,
            source_jsonl=args.source_jsonl,
        )
        if args.output is None:
            raise SystemExit("--phase score requires --output")
    else:
        if args.fixture is None or args.fixture_sha256 is None:
            raise SystemExit("--phase release requires --fixture and --fixture-sha256")
        if args.panel is None:
            raise SystemExit("--phase release requires --panel")
        if args.output is None:
            raise SystemExit("--phase release requires --output")
        # Keep the fixture gate above every model-loading import and runtime
        # construction performed by the release runner.
        fixture = load_verified_fixture(args.fixture, args.fixture_sha256)
        result = run_progressive_coordinate_release(
            fixture,
            target=args.target,
            panel_name=args.panel,
            path_labels=args.path_label,
            forced_coordinate_counts=args.forced_coordinate_count,
            decode_mode=args.decode_mode,
            sample_count=args.sample_count,
            seed_root=args.seed_root,
            sampling_temperature=args.sampling_temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            physical_batch_size=args.physical_batch_size,
            sampled_runtime_attestation=args.sampled_runtime_attestation,
            config_path=args.infer_config,
            source_jsonl=args.source_jsonl,
        )
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        if args.phase == "fixture":
            receipt = {
                "schema_version": "fixed_prefix_complete_box_coherence.fixture_receipt.v1",
                "fixture_path": str(output),
                "fixture_sha256": _file_hash(output),
                "source_fixture_schema_version": result["schema_version"],
                "admitted_targets": sorted(result["targets"]),
            }
        elif args.phase == "score":
            receipt = {
                "schema_version": "fixed_prefix_complete_box_coherence.score_receipt.v1",
                "score_path": str(output),
                "score_sha256": _file_hash(output),
                "source_fixture_path": str(args.fixture.expanduser().resolve()),
                "source_fixture_sha256": args.fixture_sha256,
                "target": args.target,
                "source_score_schema_version": result["schema_version"],
            }
        else:
            receipt = {
                "schema_version": "fixed_prefix_complete_box_coherence.release_receipt.v1",
                "release_path": str(output),
                "release_sha256": _file_hash(output),
                "source_fixture_path": str(args.fixture.expanduser().resolve()),
                "source_fixture_sha256": args.fixture_sha256,
                "target": args.target,
                "panel": args.panel,
                "decode_mode": args.decode_mode,
                "source_release_schema_version": result["schema_version"],
            }
        output.with_suffix(".receipt.json").write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
