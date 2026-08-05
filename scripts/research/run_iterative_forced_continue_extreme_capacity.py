#!/usr/bin/env python3
"""Measure repeated forced continuation until GT coverage or token exhaustion.

The model always receives its own clean generated rows.  Whenever its greedy
top-1 action at a row boundary is ``<|im_end|>`` while GT owners remain
uncovered, the terminal is recorded as a counterfactual decision and replaced
with one canonical ``<object_ref_start>`` token.  GT is never exposed to the
model; it is used only by the external stopping controller.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.analyze_individual_trajectory_union_support import (
    _min_cost_max_cardinality_assignment,
    load_generation7_annotations,
)
from scripts.research.run_current_seeded_sampled_rollouts import physical_image_id
from scripts.research.run_greedy_prefix_forced_owner_path import OBJECT_REF_START
from scripts.research.run_local_branch_causal_value import (
    _generate_after_forced_partial_row,
    _generate_row,
    _single_native_inputs,
    append_row_if_complete,
    hash_prefix_token_ids,
)


SCHEMA_VERSION = "iterative_forced_continue_extreme_capacity.v1"
CLAIM_BOUNDARY = (
    "GT-aware extreme-capacity diagnostic; not native rollout performance, "
    "GT-guided owner selection, or evidence that unrestricted continuation is safe"
)


def _normalise_category(value: Any) -> str:
    return " ".join(str(value).strip().lower().replace("_", " ").split())


def _prediction(
    raw: Mapping[str, Any],
    *,
    image_id: str,
    row_index: int,
    prediction_index: int,
) -> dict[str, Any]:
    return {
        "prediction_id": f"{image_id}:row-{row_index}:prediction-{prediction_index}",
        "generated_row_index": int(row_index),
        "category": _normalise_category(raw["description"]),
        "bbox": tuple(float(value) for value in raw["bbox"]),
    }


def _coverage_snapshot(
    predictions: Sequence[Mapping[str, Any]],
    owners: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    matches = _min_cost_max_cardinality_assignment(predictions, owners)
    matched_owner_ids = sorted({str(item["owner_id"]) for item in matches})
    matched_prediction_ids = sorted({str(item["prediction_id"]) for item in matches})
    return {
        "coverage": len(matched_owner_ids),
        "gt_owner_count": len(owners),
        "prediction_count": len(predictions),
        "false_positive_count": len(predictions) - len(matched_prediction_ids),
        "false_negative_count": len(owners) - len(matched_owner_ids),
        "matched_owner_ids": matched_owner_ids,
        "matched_prediction_ids": matched_prediction_ids,
        "matches": matches,
    }


def _is_clean_terminal(row: Mapping[str, Any], *, stop_token_id: int) -> bool:
    raw_stop = row.get("row_stop")
    stop: Mapping[str, Any] = raw_stop if isinstance(raw_stop, Mapping) else {}
    raw = [int(value) for value in row.get("raw_generated_token_ids", [])]
    return (
        row.get("status") == "success"
        and stop.get("stop_reason") == "terminal"
        and raw == [int(stop_token_id)]
    )


def _slim_row(row: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(row)
    for key in ("prefix_token_ids", "parent_prefix_token_ids", "input_prefix_token_ids"):
        values = result.pop(key, None)
        if values is not None:
            result[f"{key}_count"] = len(values)
            result[f"{key}_sha256"] = hash_prefix_token_ids(values)
    return result


def _coverage_point(
    *,
    executed_token_count: int,
    force_count: int,
    row_count: int,
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "executed_token_count": int(executed_token_count),
        "force_count": int(force_count),
        "complete_row_count": int(row_count),
        "coverage": int(snapshot["coverage"]),
        "prediction_count": int(snapshot["prediction_count"]),
        "matched_owner_ids": list(snapshot["matched_owner_ids"]),
    }


def _probe_complete_boundary(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    prefix_token_ids: Sequence[int],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    repetition_penalty: float,
    malformed_limit: int,
    row_index: int,
) -> dict[str, Any]:
    probe = _generate_row(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=prefix_token_ids,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        mode="greedy",
        seed=None,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=repetition_penalty,
        max_new_tokens=1,
        malformed_limit=malformed_limit,
        row_index=row_index,
    )
    raw = [int(value) for value in probe.get("raw_generated_token_ids", [])]
    return {
        "counted_in_completion_budget": False,
        "would_naturally_stop": raw == [int(session._im_end_token_id())],
        "probe": _slim_row(probe),
    }


def _run_case(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    image_id: str,
    owners: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    repetition_penalty: float,
    max_new_tokens: int,
    malformed_limit: int,
) -> dict[str, Any]:
    current_prefix: list[int] = []
    executed_ids: list[int] = []
    predictions: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    forces: list[dict[str, Any]] = []
    coverage = _coverage_snapshot(predictions, owners)
    curve = [
        _coverage_point(
            executed_token_count=0,
            force_count=0,
            row_count=0,
            snapshot=coverage,
        )
    ]
    native_snapshot: dict[str, Any] | None = None
    native_boundary_observed = False
    row_index = 0
    terminal_reason: str | None = None
    post_complete_probe: dict[str, Any] | None = None
    stop_token_id = int(session._im_end_token_id())

    while len(executed_ids) < int(max_new_tokens):
        remaining = int(max_new_tokens) - len(executed_ids)
        natural = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=current_prefix,
            tokenizer=tokenizer,
            image_width=image_width,
            image_height=image_height,
            mode="greedy",
            seed=None,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=repetition_penalty,
            max_new_tokens=remaining,
            malformed_limit=malformed_limit,
            row_index=row_index,
        )
        natural_reason = str(natural.get("row_stop", {}).get("stop_reason", "unknown"))

        if natural_reason == "complete_row" and natural.get("status") == "success":
            successor, append_receipt = append_row_if_complete(current_prefix, natural)
            if not append_receipt.get("appended"):
                raise RuntimeError("clean natural row was unexpectedly refused")
            raw = [int(value) for value in natural.get("raw_generated_token_ids", [])]
            if len(raw) > remaining:
                raise RuntimeError("natural row exceeded the remaining completion budget")
            token_start = len(executed_ids)
            executed_ids.extend(raw)
            current_prefix = successor
            parsed = list(natural.get("parsed_predictions") or [])
            for prediction_index, raw_prediction in enumerate(parsed):
                predictions.append(
                    _prediction(
                        raw_prediction,
                        image_id=image_id,
                        row_index=row_index,
                        prediction_index=prediction_index,
                    )
                )
            previous_coverage = int(coverage["coverage"])
            coverage = _coverage_snapshot(predictions, owners)
            events.append(
                {
                    "event_index": len(events),
                    "event_type": "natural_complete_row",
                    "row_index": row_index,
                    "token_start": token_start,
                    "token_end": len(executed_ids),
                    "coverage_before": previous_coverage,
                    "coverage_after": int(coverage["coverage"]),
                    "row": _slim_row({**natural, "append_receipt": append_receipt}),
                }
            )
            row_index += 1
            curve.append(
                _coverage_point(
                    executed_token_count=len(executed_ids),
                    force_count=len(forces),
                    row_count=row_index,
                    snapshot=coverage,
                )
            )
            if int(coverage["coverage"]) == len(owners):
                terminal_reason = "all_gt_matched"
                post_complete_probe = _probe_complete_boundary(
                    session=session,
                    native_inputs=native_inputs,
                    prefix_token_ids=current_prefix,
                    tokenizer=tokenizer,
                    image_width=image_width,
                    image_height=image_height,
                    repetition_penalty=repetition_penalty,
                    malformed_limit=malformed_limit,
                    row_index=row_index,
                )
                break
            continue

        if _is_clean_terminal(natural, stop_token_id=stop_token_id):
            if native_snapshot is None:
                native_snapshot = {
                    **coverage,
                    "executed_token_count": len(executed_ids),
                    "complete_row_count": row_index,
                }
                native_boundary_observed = True
            stop_event_index = len(events)
            events.append(
                {
                    "event_index": stop_event_index,
                    "event_type": "counterfactual_natural_stop",
                    "row_index": row_index,
                    "token_position": len(executed_ids),
                    "coverage": int(coverage["coverage"]),
                    "counterfactual_token_id": stop_token_id,
                    "counterfactual_token_counted_in_budget": False,
                    "row": _slim_row(natural),
                }
            )
            if remaining <= 0:
                terminal_reason = "token_budget"
                break
            coverage_before = int(coverage["coverage"])
            force_index = len(forces)
            if remaining == 1:
                executed_ids.append(int(OBJECT_REF_START))
                force_receipt = {
                    "force_index": force_index,
                    "trigger_event_index": stop_event_index,
                    "executed_opener_token_id": int(OBJECT_REF_START),
                    "token_start": len(executed_ids) - 1,
                    "token_end": len(executed_ids),
                    "coverage_before": coverage_before,
                    "coverage_after": coverage_before,
                    "marginal_owner_gain": 0,
                    "accepted_complete_row": False,
                    "termination": "token_budget_after_forced_opener",
                }
                forces.append(force_receipt)
                events.append(
                    {
                        "event_index": len(events),
                        "event_type": "forced_opener_without_tail_budget",
                        **force_receipt,
                    }
                )
                terminal_reason = "token_budget_after_forced_opener"
                break

            forced = _generate_after_forced_partial_row(
                session=session,
                native_inputs=native_inputs,
                parent_prefix_token_ids=current_prefix,
                forced_row_prefix_token_ids=[OBJECT_REF_START],
                tokenizer=tokenizer,
                image_width=image_width,
                image_height=image_height,
                repetition_penalty=repetition_penalty,
                max_new_tokens=remaining - 1,
                malformed_limit=malformed_limit,
                row_index=row_index,
            )
            raw = [int(value) for value in forced.get("raw_generated_token_ids", [])]
            if not raw or raw[0] != int(OBJECT_REF_START):
                raise RuntimeError("forced row does not begin with the canonical opener")
            if len(raw) > remaining:
                raise RuntimeError("forced row exceeded the remaining completion budget")
            token_start = len(executed_ids)
            executed_ids.extend(raw)
            successor, append_receipt = append_row_if_complete(current_prefix, forced)
            accepted = bool(append_receipt.get("appended"))
            if accepted:
                current_prefix = successor
                parsed = list(forced.get("parsed_predictions") or [])
                for prediction_index, raw_prediction in enumerate(parsed):
                    predictions.append(
                        _prediction(
                            raw_prediction,
                            image_id=image_id,
                            row_index=row_index,
                            prediction_index=prediction_index,
                        )
                    )
                row_index += 1
                coverage = _coverage_snapshot(predictions, owners)
            force_receipt = {
                "force_index": force_index,
                "trigger_event_index": stop_event_index,
                "executed_opener_token_id": int(OBJECT_REF_START),
                "token_start": token_start,
                "token_end": len(executed_ids),
                "coverage_before": coverage_before,
                "coverage_after": int(coverage["coverage"]),
                "marginal_owner_gain": int(coverage["coverage"]) - coverage_before,
                "accepted_complete_row": accepted,
                "row_stop_reason": str(forced.get("row_stop", {}).get("stop_reason", "unknown")),
            }
            forces.append(force_receipt)
            events.append(
                {
                    "event_index": len(events),
                    "event_type": "forced_row",
                    **force_receipt,
                    "row_index": row_index - 1 if accepted else row_index,
                    "row": _slim_row({**forced, "append_receipt": append_receipt}),
                }
            )
            curve.append(
                _coverage_point(
                    executed_token_count=len(executed_ids),
                    force_count=len(forces),
                    row_count=row_index,
                    snapshot=coverage,
                )
            )
            if not accepted:
                forced_reason = str(forced.get("row_stop", {}).get("stop_reason", "unknown"))
                terminal_reason = "token_budget" if len(executed_ids) >= int(max_new_tokens) else f"malformed_or_incomplete_forced_row:{forced_reason}"
                break
            if int(coverage["coverage"]) == len(owners):
                terminal_reason = "all_gt_matched"
                post_complete_probe = _probe_complete_boundary(
                    session=session,
                    native_inputs=native_inputs,
                    prefix_token_ids=current_prefix,
                    tokenizer=tokenizer,
                    image_width=image_width,
                    image_height=image_height,
                    repetition_penalty=repetition_penalty,
                    malformed_limit=malformed_limit,
                    row_index=row_index,
                )
                break
            continue

        raw = [int(value) for value in natural.get("raw_generated_token_ids", [])]
        if len(raw) > remaining:
            raise RuntimeError("terminal natural row exceeded the remaining completion budget")
        token_start = len(executed_ids)
        executed_ids.extend(raw)
        events.append(
            {
                "event_index": len(events),
                "event_type": "natural_nonreplaceable_terminal",
                "row_index": row_index,
                "token_start": token_start,
                "token_end": len(executed_ids),
                "row": _slim_row(natural),
            }
        )
        terminal_reason = "token_budget" if len(executed_ids) >= int(max_new_tokens) else f"malformed_or_incomplete_natural_row:{natural_reason}"
        break

    if terminal_reason is None:
        terminal_reason = "token_budget"
    if native_snapshot is None:
        native_snapshot = {
            **coverage,
            "executed_token_count": len(executed_ids),
            "complete_row_count": row_index,
        }
    return {
        "image_id": image_id,
        "repetition_penalty": float(repetition_penalty),
        "max_new_tokens": int(max_new_tokens),
        "native_boundary_observed": native_boundary_observed,
        "native_snapshot": native_snapshot,
        "final_snapshot": coverage,
        "terminal_reason": terminal_reason,
        "executed_completion_token_count": len(executed_ids),
        "executed_completion_token_ids": executed_ids,
        "executed_completion_token_ids_sha256": hash_prefix_token_ids(executed_ids),
        "valid_successor_prefix_token_count": len(current_prefix),
        "valid_successor_prefix_token_ids_sha256": hash_prefix_token_ids(current_prefix),
        "complete_row_count": row_index,
        "force_count": len(forces),
        "forces": forces,
        "coverage_curve": curve,
        "events": events,
        "post_complete_boundary_probe": post_complete_probe,
    }


def run(args: argparse.Namespace) -> Path:
    import torch

    from src.config.fingerprint import sha256_file, sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
    from src.inference.hf_backend import HFBackendSession
    from src.inference.image_plan import plan_image_batch
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    config_path = args.infer_config.expanduser().resolve(strict=True)
    annotations_path = args.annotations.expanduser().resolve(strict=True)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    if int(args.max_new_tokens) <= 1:
        raise ValueError("max_new_tokens must exceed one forced opener token")
    if float(args.repetition_penalty) <= 0.0:
        raise ValueError("repetition_penalty must be positive")

    resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(
        update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})}
    )
    if config.backend.type != "hf":
        raise ValueError("iterative forced continuation requires backend.type: hf")
    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    requested_image_ids = {str(value) for value in (args.image_id or [])}
    if requested_image_ids:
        raw_examples = [
            row
            for row in raw_examples
            if str(physical_image_id(row)) in requested_image_ids or str(row.example_id) in requested_image_ids
        ]
    if not raw_examples:
        raise ValueError("no configured rows match the requested image IDs")
    physical_ids = [str(physical_image_id(row)) for row in raw_examples]
    owners_by_image = load_generation7_annotations(annotations_path, image_ids=physical_ids)
    missing = sorted(set(physical_ids) - set(owners_by_image))
    if missing:
        raise ValueError(f"annotations lack configured images: {missing}")
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    plans = plan_image_batch(
        raw_examples,
        components=frontend.qwen,
        processor_config=_processor_config(config),
        row_indices=list(range(len(raw_examples))),
    )
    plan_by_id = {row.row_id: row for row in plans.rows}
    cases: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("iterative forced continuation opened a non-HF session")
        tokenizer = opened._tokenizer
        if tokenizer is None:
            raise RuntimeError("HF session lacks tokenizer")
        for row_index, raw in enumerate(raw_examples):
            row_id = str(raw.example_id)
            image_id = str(physical_image_id(raw))
            plan = plan_by_id[row_id]
            record = build_prompt_record(
                raw,
                _template_config(config),
                processor=frontend.qwen.processor,
                row_index=row_index,
                merged_visual_tokens=plan.merged_visual_tokens,
                object_order_seed=config.template.object_order_seed,
            )
            request = DecodeRequest(
                request_id=row_id,
                chat_text=record.chat_text,
                input_prompt_token_ids=tuple(record.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(record.expected_executed_prompt_token_ids),
                image_path=plan.image_path,
                declared_image_width=plan.declared_width,
                declared_image_height=plan.declared_height,
                decoded_image_width=plan.decoded_width,
                decoded_image_height=plan.decoded_height,
                image_sha256=plan.image_content_sha256,
                expected_image_grid_thw=cast(
                    tuple[int, int, int], tuple(plan.expected_image_grid_thw)
                ),
                logical_transform_id=plan.logical_transform_id,
                generation_policy=GenerationPolicy(max_new_tokens=1),
            )
            native_inputs, executed_prompt_ids, _, _ = opened._materialize_native_inputs((request,))
            if tuple(executed_prompt_ids[0]) != request.expected_executed_prompt_token_ids:
                raise RuntimeError(f"{row_id} prompt token parity failed")
            case = _run_case(
                session=opened,
                native_inputs=_single_native_inputs(native_inputs),
                tokenizer=tokenizer,
                image_id=image_id,
                owners=owners_by_image[image_id],
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                repetition_penalty=float(args.repetition_penalty),
                max_new_tokens=int(args.max_new_tokens),
                malformed_limit=int(args.malformed_limit),
            )
            case.update(
                {
                    "row_id": row_id,
                    "prompt_token_count": len(request.expected_executed_prompt_token_ids),
                    "prompt_token_ids_sha256": hash_prefix_token_ids(request.expected_executed_prompt_token_ids),
                    "image_sha256": plan.image_content_sha256,
                    "gt_owner_count": len(owners_by_image[image_id]),
                }
            )
            cases.append(case)
        backend_receipt = opened.receipt.to_artifact_dict()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "claim_boundary": CLAIM_BOUNDARY,
        "config": {
            "infer_config": str(config_path),
            "infer_config_sha256": sha256_file(config_path),
            "annotations": str(annotations_path),
            "annotations_sha256": sha256_file(annotations_path),
            "device": str(args.device),
            "temperature": 0.0,
            "top_p": 1.0,
            "repetition_penalty": float(args.repetition_penalty),
            "max_new_tokens_total_executed_completion": int(args.max_new_tokens),
            "forced_opener_token_id": int(OBJECT_REF_START),
            "matching": "category-consistent global one-to-one maximum-cardinality at IoU >= 0.5",
            "terminal_replacement": "counterfactual im_end is not counted; executed opener is counted",
        },
        "backend_receipt": backend_receipt,
        "case_count": len(cases),
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repetition-penalty", type=float, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=3084)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--image-id", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    output = run(parse_args())
    print(output)


if __name__ == "__main__":
    main()
