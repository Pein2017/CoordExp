#!/usr/bin/env python3
"""Run the bounded image-2299 person-25 commit closeout.

This one-time probe reuses the verified historical checkpoint-3668 runtime. It
appends raw token IDs from naturally sampled person-25 continuations to the
same forced canonical prefix, then observes one free successor row. It also
supports a forced same-depth non-25 control and the common-final-row history
comparison. It is not a general inference entry point.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_historical_random_sorted_image2299_screen as historical


SCHEMA_VERSION = "person25_commit_closeout.v1"
PERSON25_RANK = 25
DEFAULT_DONOR_SEEDS = (21, 0, 9)
DEFAULT_SOURCE_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-18-historical-random-versus-geometry-sorted-image2299-screen"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-18-person25-dominant-owner-commit-and-persistence-closeout"
)
HISTORY_OWNERS = (2, 3, 4, 14)


def _sha256_ids(values: Sequence[int]) -> str:
    return hashlib.sha256(
        json.dumps([int(value) for value in values], separators=(",", ":")).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object at {path}")
    return value


def source_arm_path(source_root: Path, owner_rank: int) -> Path:
    return source_root / "sample-arms" / f"random-owner-person-{int(owner_rank)}.json"


def source_sample(source_root: Path, *, owner_rank: int, seed: int) -> dict[str, Any]:
    path = source_arm_path(source_root, owner_rank)
    payload = _load_json(path)
    if payload.get("adapter_role") != "random":
        raise ValueError(f"source arm is not random-order: {path}")
    samples = [sample for sample in payload.get("samples", []) if int(sample.get("seed", -1)) == int(seed)]
    if len(samples) != 1:
        raise ValueError(f"expected exactly one seed {seed} sample in {path}, found {len(samples)}")
    sample = dict(samples[0])
    match = sample.get("match", {})
    if match.get("status") != "matched" or int(match.get("matched_person_only_rank", -1)) != PERSON25_RANK:
        raise ValueError(f"source seed {seed} does not strictly match person 25 in {path}")
    token_ids = sample.get("generated_token_ids")
    if not isinstance(token_ids, list) or len(token_ids) != 7:
        raise ValueError(f"source seed {seed} does not contain exactly seven raw row tokens")
    sample["source_path"] = str(path.resolve())
    return sample


def common_donor_attestation(source_root: Path, *, seed: int) -> dict[str, Any]:
    samples = [source_sample(source_root, owner_rank=owner, seed=seed) for owner in HISTORY_OWNERS]
    token_rows = [[int(value) for value in sample["generated_token_ids"]] for sample in samples]
    common = all(row == token_rows[0] for row in token_rows[1:])
    return {
        "seed": int(seed),
        "common_across_history_owners": common,
        "history_owners": list(HISTORY_OWNERS),
        "source_paths": [sample["source_path"] for sample in samples],
        "raw_token_ids": token_rows[0],
        "raw_token_ids_sha256": _sha256_ids(token_rows[0]),
    }


def candidates_and_history_rows(
    packet: Mapping[str, Any], *, earlier_owner: int
) -> tuple[list[dict[str, Any]], list[str]]:
    """Build the forced history without inheriting the old four-owner CLI limit."""

    by_person = packet["by_person_rank"]
    if earlier_owner not in by_person:
        raise ValueError(f"unknown person-only rank {earlier_owner}")
    history_ranks = [0, 1, int(earlier_owner)]
    history_objects = [by_person[rank] for rank in history_ranks]
    history_rows = [
        historical.render_legacy_row(
            str(obj["desc"]),
            [historical._coord_value(str(token)) for token in obj["bbox_2d"]],
        )
        for obj in history_objects
    ]
    emitted_global = {int(obj["global_object_rank"]) for obj in history_objects}
    candidates: list[dict[str, Any]] = []
    for obj in packet["objects"]:
        bbox = [historical._coord_value(str(token)) for token in obj["bbox_2d"]]
        global_rank = int(obj["global_object_rank"])
        candidates.append({
            "global_object_rank": global_rank,
            "person_only_rank": obj.get("person_only_rank"),
            "category_rank": int(obj["category_rank"]),
            "desc": str(obj["desc"]),
            "category_id": obj.get("category_id"),
            "coco_ann_id": obj.get("coco_ann_id"),
            "bbox": bbox,
            "row_text": historical.render_legacy_row(str(obj["desc"]), bbox),
            "emitted": global_rank in emitted_global,
            "uncovered": global_rank not in emitted_global,
            "emitted_person": str(obj["desc"]) == "person" and global_rank in emitted_global,
            "uncovered_person": str(obj["desc"]) == "person" and global_rank not in emitted_global,
        })
    return candidates, history_rows


def classify_outcome(
    parsed: Mapping[str, Any],
    match: Mapping[str, Any],
    *,
    person25_iou: float,
    committed_person_ranks: set[int],
    final_person_rank: int,
) -> str:
    status = str(parsed.get("status"))
    if status == "terminal":
        return "terminal"
    if status != "row":
        return "malformed"
    match_status = str(match.get("status"))
    if match_status != "matched":
        return "ambiguous" if match_status == "ambiguous" else "unresolved_geometry"
    rank = match.get("matched_person_only_rank")
    description = str(parsed.get("description"))
    if description == "person" and rank is not None:
        person_rank = int(rank)
        if person_rank == PERSON25_RANK:
            return "person25_repeat"
        if person_rank in committed_person_ranks:
            return "earlier_committed_person_repeat"
        return "other_uncovered_person"
    if description == "tie":
        return "tie"
    del person25_iou, final_person_rank
    return "other_matched_entity"


def summarize_samples(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stochastic = [sample for sample in samples if sample.get("decode_mode") == "sample"]
    counts = Counter(str(sample.get("outcome")) for sample in stochastic)
    owner_counts = Counter(
        str(sample.get("match", {}).get("matched_person_only_rank"))
        for sample in stochastic
        if sample.get("match", {}).get("status") == "matched"
        and sample.get("match", {}).get("matched_person_only_rank") is not None
    )
    nearest_counts = Counter(
        str(sample.get("nearest_person_rank"))
        for sample in stochastic
        if sample.get("nearest_person_rank") is not None
    )
    valid = sum(str(sample.get("parsed", {}).get("status")) == "row" for sample in stochastic)
    person25_ious = [float(sample.get("person25_iou", 0.0)) for sample in stochastic]
    greedy = next((sample for sample in samples if sample.get("decode_mode") == "greedy"), None)
    return {
        "sample_count": len(stochastic),
        "valid_row_count": int(valid),
        "strict_person25_recurrence_count": int(counts.get("person25_repeat", 0)),
        "outcome_counts": dict(sorted(counts.items())),
        "matched_person_owner_counts": dict(sorted(owner_counts.items(), key=lambda item: int(item[0]))),
        "nearest_person_owner_counts": dict(sorted(nearest_counts.items(), key=lambda item: int(item[0]))),
        "mean_iou_to_person25": sum(person25_ious) / len(person25_ious) if person25_ious else None,
        "maximum_iou_to_person25": max(person25_ious) if person25_ious else None,
        "greedy": greedy,
    }


def evaluate_immediate_gate(arms: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_name = {str(arm["arm_name"]): arm for arm in arms}
    control = by_name.get("control-person3")
    treatment_names = [f"donor-seed-{seed}" for seed in DEFAULT_DONOR_SEEDS]
    missing = [name for name in ["control-person3", *treatment_names] if name not in by_name]
    if missing:
        return {"status": "incomplete", "missing_arms": missing}
    control_summary = control["summary"]
    control_recurrence = int(control_summary["strict_person25_recurrence_count"])
    control_passes = control_recurrence >= 20
    treatment_rows = []
    for name in treatment_names:
        summary = by_name[name]["summary"]
        recurrence = int(summary["strict_person25_recurrence_count"])
        valid = int(summary["valid_row_count"])
        treatment_rows.append({
            "arm_name": name,
            "strict_person25_recurrence_count": recurrence,
            "valid_row_count": valid,
            "reduction_from_control": control_recurrence - recurrence,
        })
    immediate_suppression = bool(
        control_passes
        and all(row["reduction_from_control"] >= 12 and row["valid_row_count"] >= 22 for row in treatment_rows)
    )
    no_reliable_commit = bool(
        control_passes
        and all(row["strict_person25_recurrence_count"] >= 20 for row in treatment_rows)
    )
    if immediate_suppression:
        status = "immediate_suppression_count_gate_passed_pending_overlap_review"
    elif no_reliable_commit:
        status = "no_reliable_dominant_owner_commit"
    else:
        status = "unresolved_variant_or_depth_sensitive"
    return {
        "status": status,
        "control_passes_depth_gate": control_passes,
        "control_strict_person25_recurrence_count": control_recurrence,
        "treatments": treatment_rows,
        "conditional_common_final_history_panel_authorized": immediate_suppression,
    }


def _arm_name(final_kind: str, donor_seed: int | None) -> str:
    return "control-person3" if final_kind == "control-person3" else f"donor-seed-{int(donor_seed)}"


def _run_arm(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    source_root = Path(args.source_root).expanduser().resolve(strict=True)
    output_root = Path(args.output_root).expanduser().resolve()
    checkpoint = Path(args.checkpoint or historical.DEFAULT_CHECKPOINTS["random"]).expanduser().resolve(strict=True)
    base_model = Path(args.base_model).expanduser().resolve(strict=True)
    image_path = Path(args.image).expanduser().resolve(strict=True)
    jsonl_path = Path(args.jsonl).expanduser().resolve(strict=True)
    earlier_owner = int(args.earlier_owner)
    donor_seed = None if args.final_kind == "control-person3" else int(args.donor_seed)
    arm_name = _arm_name(args.final_kind, donor_seed)
    output_path = output_root / "arms" / f"history-{earlier_owner}-{arm_name}.json"
    if output_path.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output_path}; pass --force")

    packet = historical.select_image2299_rows(jsonl_path, image_path)
    prompt = historical.verify_historical_prompt("sorted")
    candidates, history_rows = candidates_and_history_rows(
        packet, earlier_owner=earlier_owner
    )
    device = str(args.device)
    model, processor, coord_receipt = historical._load_historical_model(base_model, checkpoint, device)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": prompt["system"]}]},
        {"role": "user", "content": [
            {"type": "image", "image": str(image_path)},
            {"type": "text", "text": prompt["user"]},
        ]},
    ]
    base_inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt"
    )
    tokenizer = processor.tokenizer
    base_prompt_ids = [int(value) for value in base_inputs["input_ids"][0].tolist()]
    history_ids: list[int] = []
    for row in history_rows:
        ids = historical._tokenize_row(tokenizer, row)
        if len(ids) != 7:
            raise RuntimeError("historical prefix row did not tokenize to seven tokens")
        history_ids.extend(ids)

    donor_receipt: dict[str, Any]
    if args.final_kind == "control-person3":
        person3 = next(item for item in candidates if item.get("person_only_rank") == 3)
        final_ids = historical._tokenize_row(tokenizer, str(person3["row_text"]))
        donor_receipt = {
            "kind": "forced_same_depth_control",
            "final_person_rank": 3,
            "row_text": person3["row_text"],
            "raw_token_ids": final_ids,
            "raw_token_ids_sha256": _sha256_ids(final_ids),
        }
        final_person_rank = 3
    else:
        donor = source_sample(source_root, owner_rank=2, seed=int(donor_seed))
        final_ids = [int(value) for value in donor["generated_token_ids"]]
        common = common_donor_attestation(source_root, seed=int(donor_seed))
        donor_receipt = {
            "kind": "naturally_sampled_after_forced_prefix",
            "source_path": donor["source_path"],
            "source_seed": int(donor_seed),
            "raw_token_ids": final_ids,
            "raw_token_ids_sha256": _sha256_ids(final_ids),
            "raw_token_ids_exactly_equal_source": final_ids == [int(value) for value in donor["generated_token_ids"]],
            "source_generated_text": donor["generated_text"],
            "source_match": donor["match"],
            "common_history_attestation": common,
        }
        final_person_rank = PERSON25_RANK
    if len(final_ids) != 7:
        raise RuntimeError("final row must contain exactly seven raw tokens")

    prefix_ids = history_ids + final_ids
    prefix_sequence = base_prompt_ids + prefix_ids
    terminal_id = int(tokenizer.convert_tokens_to_ids(historical.TERMINAL_TOKEN))
    object_ref_start_id = int(tokenizer.convert_tokens_to_ids("<|object_ref_start|>"))
    box_start_id = int(tokenizer.convert_tokens_to_ids("<|box_start|>"))
    coordinate_ids = {int(tokenizer.convert_tokens_to_ids(historical.coord_token(value))) for value in range(1000)}

    class _OneRowStop(StoppingCriteria):
        def __init__(self, start_length: int) -> None:
            self.start_length = int(start_length)

        def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> Any:
            del scores, kwargs
            return torch.tensor([
                historical.legacy_sampling_suffix_is_complete(
                    row[self.start_length :].tolist(),
                    object_ref_start_token_id=object_ref_start_id,
                    box_start_token_id=box_start_id,
                    coordinate_token_ids=coordinate_ids,
                    terminal_token_id=terminal_id,
                )
                for row in input_ids
            ], dtype=torch.bool, device=input_ids.device)

    person25 = next(item for item in candidates if item.get("person_only_rank") == PERSON25_RANK)
    initially_committed = {0, 1, earlier_owner, final_person_rank}
    horizon_rows = int(args.horizon_rows)
    if horizon_rows < 1:
        raise ValueError("--horizon-rows must be positive")

    def generate_once(*, mode: str, seed: int | None) -> dict[str, Any]:
        if seed is not None:
            historical._seed_sampling(seed)
        current_prefix = list(prefix_sequence)
        committed = set(initially_committed)
        trajectory: list[dict[str, Any]] = []
        for row_index in range(horizon_rows):
            generation_inputs: dict[str, Any] = {
                key: value.to(device=device) if isinstance(value, torch.Tensor) else value
                for key, value in base_inputs.items()
            }
            generation_inputs["input_ids"] = torch.tensor(
                [current_prefix], dtype=torch.long, device=device
            )
            generation_inputs["attention_mask"] = torch.ones(
                (1, len(current_prefix)), dtype=torch.long, device=device
            )
            kwargs: dict[str, Any] = {
                "max_new_tokens": historical.SAMPLING_MAX_NEW_TOKENS,
                "num_return_sequences": 1,
                "use_cache": True,
                "stopping_criteria": StoppingCriteriaList(
                    [_OneRowStop(len(current_prefix))]
                ),
                "repetition_penalty": historical.SAMPLING_REPETITION_PENALTY,
            }
            if mode == "sample":
                kwargs.update({
                    "do_sample": True,
                    "temperature": historical.SAMPLING_TEMPERATURE,
                    "top_p": historical.SAMPLING_TOP_P,
                })
            else:
                kwargs["do_sample"] = False
            with torch.inference_mode():
                output = model.generate(**generation_inputs, **kwargs)
            new_ids = [
                int(value)
                for value in output[0, len(current_prefix):].detach().cpu().tolist()
            ]
            text = tokenizer.decode(
                new_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            parsed = historical.parse_legacy_generation(text)
            match = historical.match_parsed_row_to_candidates(parsed, candidates)
            parsed_bbox = parsed.get("bbox")
            iou25 = (
                historical.bbox_iou(parsed_bbox, person25["bbox"])
                if parsed.get("status") == "row"
                else 0.0
            )
            rankings = match.get("candidate_rankings", [])
            nearest_person_rank = (
                rankings[0].get("person_only_rank") if rankings else None
            )
            outcome = classify_outcome(
                parsed,
                match,
                person25_iou=iou25,
                committed_person_ranks=committed,
                final_person_rank=final_person_rank,
            )
            row_record = {
                "row_index": row_index,
                "input_prefix_token_ids_sha256": _sha256_ids(current_prefix),
                "generated_token_ids": new_ids,
                "generated_text": text,
                "parsed": parsed,
                "match": match,
                "person25_iou": iou25,
                "nearest_person_rank": nearest_person_rank,
                "exact_final_row_token_repeat": new_ids == final_ids,
                "outcome": outcome,
            }
            trajectory.append(row_record)
            if parsed.get("status") != "row" or parsed.get("terminal_after_row"):
                break
            current_prefix.extend(new_ids)
            matched_rank = match.get("matched_person_only_rank")
            if match.get("status") == "matched" and matched_rank is not None:
                committed.add(int(matched_rank))

        first = dict(trajectory[0])
        first.update({
            "decode_mode": mode,
            "seed": seed,
            "trajectory": trajectory,
            "horizon_rows_requested": horizon_rows,
            "horizon_rows_generated": len(trajectory),
        })
        return first

    samples = [generate_once(mode="greedy", seed=None)]
    samples.extend(generate_once(mode="sample", seed=seed) for seed in range(int(args.sample_count)))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "arm_name": arm_name,
        "earlier_owner_person_rank": earlier_owner,
        "final_person_rank": final_person_rank,
        "model": {
            "base_model": str(base_model),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": historical.sha256_file(checkpoint / "adapter_model.safetensors"),
            "runtime_dtype": "torch.float32",
            "attention_implementation": "eager",
            "device": device,
        },
        "image": {"image_id": historical.EXPECTED_IMAGE_ID, "path": str(image_path), "sha256": packet["image_sha256"]},
        "prompt_hash": prompt["hash"],
        "prefix": {
            "history_person_ranks": [0, 1, earlier_owner],
            "history_rows": history_rows,
            "history_token_ids": history_ids,
            "final_row": donor_receipt,
            "full_prefix_token_ids": prefix_ids,
            "full_prefix_token_ids_sha256": _sha256_ids(prefix_ids),
            "full_model_input_ids_sha256": _sha256_ids(prefix_sequence),
        },
        "coord_offset_load_receipt": coord_receipt,
        "decode": {
            "sample_seeds": list(range(int(args.sample_count))),
            "temperature": historical.SAMPLING_TEMPERATURE,
            "top_p": historical.SAMPLING_TOP_P,
            "repetition_penalty": historical.SAMPLING_REPETITION_PENALTY,
            "max_new_tokens": historical.SAMPLING_MAX_NEW_TOKENS,
            "horizon_rows": horizon_rows,
        },
        "samples": samples,
    }
    payload["summary"] = summarize_samples(samples)
    historical._atomic_json_dump(output_path, payload, force=bool(args.force))
    return payload


def _summarize(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.output_root).expanduser().resolve(strict=True)
    paths = sorted((root / "arms").glob("history-*.json"))
    arms = [_load_json(path) for path in paths]
    immediate_names = {"control-person3", *(f"donor-seed-{seed}" for seed in DEFAULT_DONOR_SEEDS)}
    immediate_arms = [
        arm
        for arm in arms
        if int(arm.get("earlier_owner_person_rank", -1)) == 2
        and str(arm.get("arm_name")) in immediate_names
    ]
    has_complete_immediate_panel = {
        str(arm.get("arm_name")) for arm in immediate_arms
    } == immediate_names
    summary = {
        "schema_version": SCHEMA_VERSION,
        "arm_paths": [str(path) for path in paths],
        "arms": [
            {
                "arm_name": arm["arm_name"],
                "earlier_owner_person_rank": arm["earlier_owner_person_rank"],
                "summary": arm["summary"],
            }
            for arm in arms
        ],
        "gate": (
            evaluate_immediate_gate(immediate_arms)
            if has_complete_immediate_panel
            else {
                "status": "not_applicable_in_incomplete_or_conditional_panel",
                "complete_immediate_panel_present": False,
            }
        ),
    }
    path = root / "summary.json"
    historical._atomic_json_dump(path, summary, force=bool(args.force))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run-arm")
    run.add_argument("--final-kind", choices=("donor", "control-person3"), required=True)
    run.add_argument("--donor-seed", type=int, choices=DEFAULT_DONOR_SEEDS)
    run.add_argument("--earlier-owner", type=int, default=2)
    run.add_argument("--sample-count", type=int, default=24)
    run.add_argument("--horizon-rows", type=int, default=1)
    run.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--base-model", type=Path, default=historical.DEFAULT_BASE_MODEL)
    run.add_argument("--image", type=Path, default=historical.DEFAULT_IMAGE)
    run.add_argument("--jsonl", type=Path, default=historical.DEFAULT_JSONL)
    run.add_argument("--device", default="cuda")
    run.add_argument("--force", action="store_true")
    summary = sub.add_parser("summarize")
    summary.add_argument("--output-root", type=Path, required=True)
    summary.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "run-arm":
        if args.final_kind == "donor" and args.donor_seed is None:
            raise SystemExit("--donor-seed is required for --final-kind donor")
        if args.final_kind == "control-person3" and args.donor_seed is not None:
            raise SystemExit("--donor-seed is invalid for the control arm")
        _run_arm(args)
    elif args.command == "summarize":
        _summarize(args)
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
