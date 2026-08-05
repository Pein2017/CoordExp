#!/usr/bin/env python3
"""Force one canonical row opener after each model's own greedy terminal.

The frozen greedy artifact owns the native completion prefix.  The treatment
appends exactly one ``OBJECT_REF_START`` token, finishes that row greedily, and
then continues row by row until the model terminates or the original total
completion token budget is exhausted.  The receipt retains both the first-row
release and the full forced continuation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_current_seeded_sampled_rollouts import physical_image_id
from scripts.research.run_greedy_prefix_forced_owner_path import (
    OBJECT_REF_START,
    _greedy_suffix,
)
from scripts.research.run_local_branch_causal_value import (
    _generate_after_forced_partial_row,
    _single_native_inputs,
    append_row_if_complete,
    build_positive_entity_ledger,
)


SCHEMA_VERSION = "own_terminal_forced_opener_panel.v1"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _native_prefixes(artifact_dir: Path) -> dict[str, list[int]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _read_jsonl(artifact_dir / "pred_token_trace.jsonl"):
        if row.get("trace_type") == "generated_token" and not row.get("is_pad"):
            grouped[str(row["row_id"])].append(row)
    result: dict[str, list[int]] = {}
    for row_id, values in grouped.items():
        values.sort(key=lambda item: int(item["generated_step_index"]))
        stop_indices = [index for index, item in enumerate(values) if item.get("is_stop")]
        if stop_indices != [len(values) - 1]:
            raise ValueError(f"{row_id} lacks one terminal final stop token")
        result[row_id] = [int(item["token_id"]) for item in values[:-1]]
    return result


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

    config_path = args.infer_config.resolve(strict=True)
    artifact_dir = args.greedy_artifact_dir.resolve(strict=True)
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(f"refusing to overwrite {output}; pass --force")
    resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(
        update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})}
    )
    if config.backend.type != "hf":
        raise ValueError("forced-opener panel requires backend.type: hf")
    if int(args.max_new_tokens) <= 1:
        raise ValueError("max_new_tokens must exceed the one forced opener token")

    greedy_rows = {str(row["row_id"]): row for row in _read_jsonl(artifact_dir / "gt_vs_pred.jsonl")}
    prefixes = _native_prefixes(artifact_dir)
    if set(greedy_rows) != set(prefixes):
        raise ValueError("greedy rows and token traces cover different row IDs")
    if any(row.get("decode_stop_reason") != "im_end" for row in greedy_rows.values()):
        raise ValueError("every forced case must originate at a native im_end")

    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    requested_image_ids = set(args.image_id or [])
    if requested_image_ids:
        raw_examples = [
            row for row in raw_examples
            if str(physical_image_id(row)) in requested_image_ids
            or str(row.example_id) in requested_image_ids
        ]
        if not raw_examples:
            raise ValueError("no configured rows match --image-id")
        selected_row_ids = {str(row.example_id) for row in raw_examples}
        greedy_rows = {key: value for key, value in greedy_rows.items() if key in selected_row_ids}
        prefixes = {key: value for key, value in prefixes.items() if key in selected_row_ids}
    raw_by_row_id = {str(row.example_id): row for row in raw_examples}
    if set(raw_by_row_id) != set(prefixes):
        raise ValueError("configured input rows differ from the greedy artifact")
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
            raise RuntimeError("forced-opener panel opened a non-HF session")
        tokenizer = opened._tokenizer
        if tokenizer is None:
            raise RuntimeError("HF session lacks tokenizer")
        for row_index, raw in enumerate(raw_examples):
            row_id = str(raw.example_id)
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
                expected_image_grid_thw=tuple(plan.expected_image_grid_thw),
                logical_transform_id=plan.logical_transform_id,
                generation_policy=GenerationPolicy(max_new_tokens=1),
            )
            native_inputs, executed_ids, _, _ = opened._materialize_native_inputs((request,))
            if tuple(executed_ids[0]) != request.expected_executed_prompt_token_ids:
                raise RuntimeError(f"{row_id} prompt token parity failed")
            native_prefix = prefixes[row_id]
            remaining = int(args.max_new_tokens) - len(native_prefix) - 1
            if remaining <= 0:
                raise ValueError(f"{row_id} native prefix exhausts the total completion budget")
            one_native = _single_native_inputs(native_inputs)
            first = _generate_after_forced_partial_row(
                session=opened,
                native_inputs=one_native,
                parent_prefix_token_ids=native_prefix,
                forced_row_prefix_token_ids=[OBJECT_REF_START],
                tokenizer=tokenizer,
                image_width=int(plan.decoded_width),
                image_height=int(plan.decoded_height),
                repetition_penalty=float(args.repetition_penalty),
                max_new_tokens=remaining,
                malformed_limit=int(args.malformed_limit),
                row_index=len(greedy_rows[row_id].get("pred", [])),
            )
            current_prefix, append_receipt = append_row_if_complete(native_prefix, first)
            first["append_receipt"] = append_receipt
            first["accepted_complete_row"] = bool(append_receipt.get("appended"))
            suffix: list[dict[str, Any]] = []
            suffix_used = 0
            if first["accepted_complete_row"]:
                remaining_after_first = int(args.max_new_tokens) - len(native_prefix) - len(first["raw_generated_token_ids"])
                suffix, suffix_used, _ = _greedy_suffix(
                    session=opened,
                    native_inputs=one_native,
                    prefix=current_prefix,
                    tokenizer=tokenizer,
                    width=int(plan.decoded_width),
                    height=int(plan.decoded_height),
                    horizon_rows=int(args.max_rows),
                    remaining_tokens=max(0, remaining_after_first),
                    ledger=build_positive_entity_ledger(raw),
                    image_id=physical_image_id(raw),
                    covered=[],
                    repetition_penalty=float(args.repetition_penalty),
                    malformed_limit=int(args.malformed_limit),
                    start_row_index=len(greedy_rows[row_id].get("pred", [])) + 1,
                )
            cases.append({
                "row_id": row_id,
                "image_id": physical_image_id(raw),
                "native_prefix_token_count": len(native_prefix),
                "native_predictions": greedy_rows[row_id].get("pred", []),
                "forced_opener_token_id": OBJECT_REF_START,
                "first_row": first,
                "suffix_rows": suffix,
                "released_token_count": len(first.get("released_tail_token_ids", [])) + suffix_used,
            })
        backend_receipt = opened.receipt.to_artifact_dict()

    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "infer_config": str(config_path),
            "infer_config_sha256": sha256_file(config_path),
            "greedy_artifact_dir": str(artifact_dir),
            "max_new_tokens_total_completion": int(args.max_new_tokens),
            "repetition_penalty": float(args.repetition_penalty),
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "backend_receipt": backend_receipt,
        "case_count": len(cases),
        "cases": cases,
        "claim_boundary": "own-terminal forced continuation; first-row and full suffix are diagnostics, not native rollout metrics",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", required=True, type=Path)
    parser.add_argument("--greedy-artifact-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--image-id", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=3084)
    parser.add_argument("--max-rows", type=int, default=512)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    print(run(parse_args()))


if __name__ == "__main__":
    main()
