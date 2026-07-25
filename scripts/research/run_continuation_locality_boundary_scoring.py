#!/usr/bin/env python3
"""Score row-opener versus terminal at frozen exact-prefix control states."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.materialize_continuation_locality_owner_compositionality import (  # noqa: E402
    SCHEMA_VERSION as MANIFEST_SCHEMA_VERSION,
)
from scripts.research.run_complete_candidate_row_scoring import (  # noqa: E402
    OBJECT_REF_START,
    _forward_logits,
    _runtime_model_dtype_summary,
)
from scripts.research.run_native_sibling_branch_replay import (  # noqa: E402
    _attention_implementation,
)
from scripts.research.run_next_row_likelihood_change import (  # noqa: E402
    terminal_boundary_score,
)
from src.config.fingerprint import sha256_file  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402


RECEIPT_SCHEMA_VERSION = "continuation_locality_boundary_scoring.receipt.v1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stable_shard(value: str, count: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % count


def _ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(f"{label} must be a non-empty token-id list")
    return [int(item) for item in value]


def _validated_boundaries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected manifest schema {MANIFEST_SCHEMA_VERSION}")
    values = manifest.get("locality_boundaries")
    if not isinstance(values, list) or not values:
        raise ValueError("manifest has no locality boundaries")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ValueError(f"locality_boundaries[{index}] must be an object")
        item = dict(raw)
        boundary_id = str(item.get("boundary_id", ""))
        if not boundary_id or boundary_id in seen:
            raise ValueError(f"invalid or duplicate boundary_id={boundary_id!r}")
        seen.add(boundary_id)
        prompt = _ids(item.get("base_prompt_token_ids"), label=f"{boundary_id}.base_prompt")
        prefix = _ids(item.get("prefix_token_ids"), label=f"{boundary_id}.prefix")
        if token_ids_sha256(prompt) != item.get("base_prompt_token_ids_sha256"):
            raise ValueError(f"{boundary_id} base prompt hash mismatch")
        if token_ids_sha256(prefix) != item.get("prefix_token_ids_sha256"):
            raise ValueError(f"{boundary_id} prefix hash mismatch")
        if prefix[-1] != 151649:
            raise ValueError(f"{boundary_id} is not a complete-row boundary")
        results.append({**item, "base_prompt_token_ids": prompt, "prefix_token_ids": prefix})
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    boundaries = _validated_boundaries(manifest)
    cohort_filter = set(args.cohort or [])
    selected = [
        item
        for item in boundaries
        if _stable_shard(str(item["boundary_id"]), int(args.shard_count))
        == int(args.shard_index)
        and (not cohort_filter or str(item["cohort"]) in cohort_filter)
    ]
    selected.sort(key=lambda item: str(item["boundary_id"]))
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    if not selected:
        raise ValueError("selected locality shard is empty")

    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_jsonl = args.source_jsonl.expanduser().resolve(strict=True)
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {output_path}")

    import torch
    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
    from src.inference.hf_backend import HFBackendSession
    from src.inference.image_plan import plan_image_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend

    resolved = load_infer_config(config_path)
    config = resolved.config
    if args.runtime_dtype == "fp32":
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"dtype": "fp32"})}
        )
    if config.backend.type != "hf":
        raise ValueError("boundary scoring requires backend.type: hf")
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    raw_rows = load_raw_examples(source_jsonl)
    raw_by_id = {}
    for row in raw_rows:
        source_metadata = row.metadata.get("source")
        if not isinstance(source_metadata, Mapping) or source_metadata.get("image_id") is None:
            raise ValueError("source JSONL row lacks source.image_id metadata")
        raw_by_id[str(source_metadata["image_id"])] = row
    template = _template_config(config)
    output_records: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF launch opened an unexpected backend session")
        model = opened._model  # noqa: SLF001
        tokenizer = opened._tokenizer  # noqa: SLF001
        if model is None or tokenizer is None:
            raise RuntimeError("HF session did not expose its loaded model and tokenizer")
        model.eval()
        terminal_id = tokenizer.eos_token_id
        if terminal_id is None or int(terminal_id) < 0:
            raise ValueError("tokenizer does not expose eos_token_id")
        parity = verify_processor_model_vision_parity(
            processor_identity=frontend.qwen.processor_identity,
            model_config=model.config,
        )
        backend_receipt = opened.receipt.to_artifact_dict()
        model_dtype = _runtime_model_dtype_summary(model)
        attention = _attention_implementation(
            model, config.backend.hf.attn_implementation
        )
        for item in selected:
            image_id = str(item["image_id"])
            raw = raw_by_id.get(image_id)
            if raw is None:
                raise ValueError(f"image {image_id} is absent from source JSONL")
            image_plan = plan_image_batch(
                [raw],
                components=frontend.qwen,
                processor_config=_processor_config(config),
                row_indices=[0],
            ).rows[0]
            if image_plan.image_content_sha256 != item["image_content_sha256"]:
                raise ValueError(f"{item['boundary_id']} image hash mismatch")
            prompt_record = build_prompt_record(
                raw,
                template,
                processor=frontend.qwen.processor,
                row_index=0,
                merged_visual_tokens=image_plan.merged_visual_tokens,
            )
            base_prompt = [int(value) for value in prompt_record.prompt_token_ids]
            if base_prompt != item["base_prompt_token_ids"]:
                raise ValueError(f"{item['boundary_id']} active prompt mismatch")
            expected_grid = tuple(int(value) for value in image_plan.expected_image_grid_thw)
            if len(expected_grid) != 3:
                raise ValueError(f"{item['boundary_id']} expected image grid is not rank three")
            request = DecodeRequest(
                request_id=f"continuation-locality:{args.checkpoint_role}:{item['boundary_id']}",
                chat_text=prompt_record.chat_text,
                input_prompt_token_ids=tuple(prompt_record.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(
                    prompt_record.expected_executed_prompt_token_ids
                ),
                image_path=image_plan.image_path,
                declared_image_width=image_plan.declared_width,
                declared_image_height=image_plan.declared_height,
                decoded_image_width=image_plan.decoded_width,
                decoded_image_height=image_plan.decoded_height,
                image_sha256=image_plan.image_content_sha256,
                expected_image_grid_thw=(expected_grid[0], expected_grid[1], expected_grid[2]),
                logical_transform_id=image_plan.logical_transform_id,
                generation_policy=GenerationPolicy(
                    max_new_tokens=1,
                    repetition_penalty=1.0,
                    temperature=0.0,
                    top_p=1.0,
                    include_raw_model_logprob=True,
                ),
            )
            native_inputs, executed_prompt_ids, _, _ = opened._materialize_native_inputs(  # noqa: SLF001
                (request,)
            )
            if tuple(executed_prompt_ids[0]) != tuple(base_prompt):
                raise RuntimeError(f"{item['boundary_id']} materialized prompt mismatch")
            image_grid_thw = native_inputs.get("image_grid_thw")
            if not isinstance(image_grid_thw, torch.Tensor):
                raise ValueError("native inputs lack image_grid_thw")
            model_inputs = {
                key: value
                for key, value in native_inputs.items()
                if key
                not in {
                    "input_ids",
                    "attention_mask",
                    "position_ids",
                    "token_type_ids",
                }
            }
            actual_prefix = [*base_prompt, *item["prefix_token_ids"]]
            logits = _forward_logits(model, model_inputs, actual_prefix, image_grid_thw)
            terminal = terminal_boundary_score(
                logits,
                boundary_length=len(actual_prefix),
                row_entry_token_id=OBJECT_REF_START,
                terminal_token_id=int(terminal_id),
            )
            output_records.append(
                {
                    "boundary_id": item["boundary_id"],
                    "cohort": item["cohort"],
                    "image_id": image_id,
                    "prefix_depth": int(item["prefix_depth"]),
                    "object_count_band": item["object_count_band"],
                    "observed_next_action": item["observed_next_action"],
                    "remaining_annotation_owner_count": int(
                        item["remaining_annotation_owner_count"]
                    ),
                    "matched_training_event_id": item.get("matched_training_event_id"),
                    "prefix_token_ids_sha256": item["prefix_token_ids_sha256"],
                    "actual_prompt_plus_prefix_token_ids_sha256": token_ids_sha256(
                        actual_prefix
                    ),
                    "terminal_boundary": terminal,
                }
            )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": str(manifest["unit_id"]),
        "checkpoint_role": str(args.checkpoint_role),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "records": output_records,
        "runtime": {
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "limit": args.limit,
            "cohort_filter": sorted(cohort_filter),
            "record_count": len(output_records),
            "physical_batch_size": 1,
            "runtime_dtype_mode": str(args.runtime_dtype),
            "model_dtype": model_dtype,
            "config_path": str(config_path),
            "authored_config_sha256": sha256_file(config_path),
            "resolved_config_fingerprint": resolved.fingerprint,
            "effective_config_sha256": config_sha256_json(
                config.model_dump(mode="json")
            ),
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256": sha256_file(source_jsonl),
            "attention_implementation": attention,
            "processor_model_vision_parity": parity,
            "backend_session": backend_receipt,
            "eos_token_id": int(terminal_id),
            "row_entry_token_id": int(OBJECT_REF_START),
        },
        "claim_boundary": "fixed-prefix boundary scores are not free-rollout outcomes",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--checkpoint-role", required=True)
    parser.add_argument("--runtime-dtype", choices=("config", "fp32"), default="fp32")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--cohort", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "checkpoint_role": result["checkpoint_role"],
                "record_count": result["runtime"]["record_count"],
                "shard_index": result["runtime"]["shard_index"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
