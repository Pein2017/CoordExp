#!/usr/bin/env python3
"""Run exact-prefix remaining-owner recovery and one-step composition probes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
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
    _score_candidate,
)
from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _annotate_owner_matches,
    _generate_after_forced_partial_row,
    _generate_row,
    _single_native_inputs,
    build_positive_entity_ledger,
)
from scripts.research.run_native_sibling_branch_replay import (  # noqa: E402
    _attention_implementation,
)
from scripts.research.run_next_row_likelihood_change import (  # noqa: E402
    _canonical_row_phases,
    terminal_boundary_score,
)
from src.config.fingerprint import sha256_file  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402


RECEIPT_SCHEMA_VERSION = "exact_prefix_owner_compositionality.receipt.v1"
BOX_END_TOKEN_ID = 151649


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


def _target(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    result = dict(value)
    owner_id = str(result.get("owner_id", ""))
    category = str(result.get("category", ""))
    if not owner_id or not category:
        raise ValueError(f"{label} lacks owner/category")
    row = _ids(result.get("row_token_ids"), label=f"{label}.row_token_ids")
    if token_ids_sha256(row) != result.get("row_token_ids_sha256"):
        raise ValueError(f"{label} row hash mismatch")
    _canonical_row_phases(row)
    return {**result, "owner_id": owner_id, "category": category, "row_token_ids": row}


def _validated_cases(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected manifest schema {MANIFEST_SCHEMA_VERSION}")
    values = manifest.get("owner_cases")
    if not isinstance(values, list) or not values:
        raise ValueError("manifest has no owner cases")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(values):
        if not isinstance(raw, Mapping):
            raise ValueError(f"owner_cases[{index}] must be an object")
        item = dict(raw)
        case_id = str(item.get("case_id", ""))
        if not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate case_id={case_id!r}")
        seen.add(case_id)
        prompt = _ids(item.get("base_prompt_token_ids"), label=f"{case_id}.base_prompt")
        prefix = _ids(item.get("prefix_token_ids"), label=f"{case_id}.prefix")
        if token_ids_sha256(prompt) != item.get("base_prompt_token_ids_sha256"):
            raise ValueError(f"{case_id} base prompt hash mismatch")
        if token_ids_sha256(prefix) != item.get("prefix_token_ids_sha256"):
            raise ValueError(f"{case_id} prefix hash mismatch")
        if prefix[-1] != BOX_END_TOKEN_ID:
            raise ValueError(f"{case_id} is not a complete-row boundary")
        target = _target(item.get("target"), label=f"{case_id}.target")
        secondary_values = item.get("secondary_targets", [])
        if not isinstance(secondary_values, list):
            raise ValueError(f"{case_id}.secondary_targets must be a list")
        secondary = [
            _target(value, label=f"{case_id}.secondary_targets[{target_index}]")
            for target_index, value in enumerate(secondary_values)
        ]
        results.append(
            {
                **item,
                "base_prompt_token_ids": prompt,
                "prefix_token_ids": prefix,
                "target": target,
                "secondary_targets": secondary,
            }
        )
    return results


def _description_force(row_tokens: Sequence[int]) -> list[int]:
    row = [int(item) for item in row_tokens]
    phases = _canonical_row_phases(row)
    description = phases["description"]
    object_end_index = description[-1] + 1
    forced = row[: object_end_index + 1]
    if forced[0] != OBJECT_REF_START or forced[-1] != 151647:
        raise ValueError("description force does not end at object-ref close")
    return forced


def _annotated_release(
    *,
    kind: str,
    session: Any,
    native_inputs: Mapping[str, Any],
    prefix: Sequence[int],
    forced: Sequence[int] | None,
    tokenizer: Any,
    width: int,
    height: int,
    row_index: int,
    ledger: Sequence[Mapping[str, Any]],
    covered_owner_ids: Sequence[str],
    intended_owner_id: str | None,
    max_new_tokens: int,
    malformed_limit: int,
) -> dict[str, Any]:
    if forced is None:
        row = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=prefix,
            tokenizer=tokenizer,
            image_width=width,
            image_height=height,
            mode="greedy",
            seed=0,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            malformed_limit=malformed_limit,
            row_index=row_index,
        )
    else:
        row = _generate_after_forced_partial_row(
            session=session,
            native_inputs=native_inputs,
            parent_prefix_token_ids=prefix,
            forced_row_prefix_token_ids=forced,
            tokenizer=tokenizer,
            image_width=width,
            image_height=height,
            repetition_penalty=1.0,
            max_new_tokens=max_new_tokens,
            malformed_limit=malformed_limit,
            row_index=row_index,
        )
    covered_ledger_ids = [str(value).split(":", 1)[-1] for value in covered_owner_ids]
    _annotate_owner_matches(
        row,
        entity_ledger=ledger,
        image_width=width,
        image_height=height,
        covered_entity_ids=covered_ledger_ids,
    )
    matched = set(str(value) for value in row.get("strict_matched_owner_ids", []))
    intended_ledger_owner_id = (
        None if intended_owner_id is None else str(intended_owner_id).split(":", 1)[-1]
    )
    row["intervention_kind"] = kind
    row["intended_owner_id"] = intended_owner_id
    row["intended_ledger_owner_id"] = intended_ledger_owner_id
    row["covered_composite_owner_ids"] = sorted(str(value) for value in covered_owner_ids)
    row["intended_owner_realized"] = (
        None if intended_ledger_owner_id is None else intended_ledger_owner_id in matched
    )
    return row


def _score_state_target(
    *,
    model: Any,
    model_inputs: Mapping[str, Any],
    image_grid_thw: Any,
    base_prompt: Sequence[int],
    prefix: Sequence[int],
    target: Mapping[str, Any],
    terminal_id: int,
) -> dict[str, Any]:
    actual_prefix = [*base_prompt, *prefix]
    boundary_logits = _forward_logits(model, model_inputs, actual_prefix, image_grid_thw)
    terminal = terminal_boundary_score(
        boundary_logits,
        boundary_length=len(actual_prefix),
        row_entry_token_id=OBJECT_REF_START,
        terminal_token_id=terminal_id,
    )
    row_tokens = [int(item) for item in target["row_token_ids"]]
    row_logits = _forward_logits(
        model, model_inputs, [*actual_prefix, *row_tokens], image_grid_thw
    )
    candidate = _score_candidate(
        row_logits,
        boundary_length=len(actual_prefix),
        row_tokens=row_tokens,
        metadata={
            "candidate_id": f"owner-{target['owner_id']}",
            "owner": target["owner_id"],
            "category": target["category"],
            "role": "verified_uncovered_owner",
            "covered": False,
        },
    )
    return {
        "prefix_token_ids_sha256": token_ids_sha256(prefix),
        "actual_prompt_plus_prefix_token_ids_sha256": token_ids_sha256(actual_prefix),
        "terminal_boundary": terminal,
        "candidate_score": candidate,
    }


def _post_action_probe(
    *,
    mode: str,
    post_prefix: Sequence[int],
    target: Mapping[str, Any],
    session: Any,
    native_inputs: Mapping[str, Any],
    model: Any,
    model_inputs: Mapping[str, Any],
    image_grid_thw: Any,
    base_prompt: Sequence[int],
    tokenizer: Any,
    terminal_id: int,
    width: int,
    height: int,
    row_index: int,
    ledger: Sequence[Mapping[str, Any]],
    covered_owner_ids: Sequence[str],
    max_new_tokens: int,
    malformed_limit: int,
) -> dict[str, Any]:
    score = _score_state_target(
        model=model,
        model_inputs=model_inputs,
        image_grid_thw=image_grid_thw,
        base_prompt=base_prompt,
        prefix=post_prefix,
        target=target,
        terminal_id=terminal_id,
    )
    opener = _annotated_release(
        kind=f"{mode}:forced_opener",
        session=session,
        native_inputs=native_inputs,
        prefix=post_prefix,
        forced=[OBJECT_REF_START],
        tokenizer=tokenizer,
        width=width,
        height=height,
        row_index=row_index,
        ledger=ledger,
        covered_owner_ids=covered_owner_ids,
        intended_owner_id=str(target["owner_id"]),
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
    )
    description = _annotated_release(
        kind=f"{mode}:forced_description",
        session=session,
        native_inputs=native_inputs,
        prefix=post_prefix,
        forced=_description_force(target["row_token_ids"]),
        tokenizer=tokenizer,
        width=width,
        height=height,
        row_index=row_index,
        ledger=ledger,
        covered_owner_ids=covered_owner_ids,
        intended_owner_id=str(target["owner_id"]),
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
    )
    return {"mode": mode, "score": score, "forced_opener": opener, "forced_description": description}


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.shard_count) <= 0 or not 0 <= int(args.shard_index) < int(args.shard_count):
        raise ValueError("invalid shard index/count")
    if int(args.max_new_tokens) <= 0 or int(args.malformed_limit) <= 0:
        raise ValueError("generation limits must be positive")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    cases = _validated_cases(manifest)
    requested_case_ids = set(args.case_id or [])
    selected = [
        item
        for item in cases
        if _stable_shard(str(item["case_id"]), int(args.shard_count))
        == int(args.shard_index)
        and (not requested_case_ids or str(item["case_id"]) in requested_case_ids)
    ]
    selected.sort(key=lambda item: str(item["case_id"]))
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    if not selected:
        raise ValueError("selected owner-case shard is empty")

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
        raise ValueError("owner compositionality requires backend.type: hf")
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
    output_cases: list[dict[str, Any]] = []
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
        for case in selected:
            image_id = str(case["image_id"])
            raw = raw_by_id.get(image_id)
            if raw is None:
                raise ValueError(f"image {image_id} is absent from source JSONL")
            image_plan = plan_image_batch(
                [raw],
                components=frontend.qwen,
                processor_config=_processor_config(config),
                row_indices=[0],
            ).rows[0]
            if image_plan.image_content_sha256 != case["image_content_sha256"]:
                raise ValueError(f"{case['case_id']} image hash mismatch")
            prompt_record = build_prompt_record(
                raw,
                template,
                processor=frontend.qwen.processor,
                row_index=0,
                merged_visual_tokens=image_plan.merged_visual_tokens,
            )
            base_prompt = [int(value) for value in prompt_record.prompt_token_ids]
            if base_prompt != case["base_prompt_token_ids"]:
                raise ValueError(f"{case['case_id']} active prompt mismatch")
            expected_grid = tuple(int(value) for value in image_plan.expected_image_grid_thw)
            if len(expected_grid) != 3:
                raise ValueError(f"{case['case_id']} expected image grid is not rank three")
            request = DecodeRequest(
                request_id=f"owner-composition:{args.checkpoint_role}:{case['case_id']}",
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
                raise RuntimeError(f"{case['case_id']} materialized prompt mismatch")
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
            one_native = _single_native_inputs(native_inputs)
            ledger = build_positive_entity_ledger(raw)
            prefix = list(case["prefix_token_ids"])
            covered = [str(value) for value in case["covered_owner_ids"]]
            target = case["target"]
            state_score = _score_state_target(
                model=model,
                model_inputs=model_inputs,
                image_grid_thw=image_grid_thw,
                base_prompt=base_prompt,
                prefix=prefix,
                target=target,
                terminal_id=int(terminal_id),
            )
            native = _annotated_release(
                kind="native",
                session=opened,
                native_inputs=one_native,
                prefix=prefix,
                forced=None,
                tokenizer=tokenizer,
                width=int(image_plan.decoded_width),
                height=int(image_plan.decoded_height),
                row_index=int(case["prefix_depth"]),
                ledger=ledger,
                covered_owner_ids=covered,
                intended_owner_id=str(target["owner_id"]),
                max_new_tokens=int(args.max_new_tokens),
                malformed_limit=int(args.malformed_limit),
            )
            opener = _annotated_release(
                kind="forced_opener",
                session=opened,
                native_inputs=one_native,
                prefix=prefix,
                forced=[OBJECT_REF_START],
                tokenizer=tokenizer,
                width=int(image_plan.decoded_width),
                height=int(image_plan.decoded_height),
                row_index=int(case["prefix_depth"]),
                ledger=ledger,
                covered_owner_ids=covered,
                intended_owner_id=str(target["owner_id"]),
                max_new_tokens=int(args.max_new_tokens),
                malformed_limit=int(args.malformed_limit),
            )
            description = _annotated_release(
                kind="forced_description",
                session=opened,
                native_inputs=one_native,
                prefix=prefix,
                forced=_description_force(target["row_token_ids"]),
                tokenizer=tokenizer,
                width=int(image_plan.decoded_width),
                height=int(image_plan.decoded_height),
                row_index=int(case["prefix_depth"]),
                ledger=ledger,
                covered_owner_ids=covered,
                intended_owner_id=str(target["owner_id"]),
                max_new_tokens=int(args.max_new_tokens),
                malformed_limit=int(args.malformed_limit),
            )
            composition: list[dict[str, Any]] = []
            for secondary in case["secondary_targets"]:
                oracle_prefix = [*prefix, *target["row_token_ids"]]
                oracle_covered = sorted({*covered, str(target["owner_id"])})
                post = {
                    "secondary_owner_id": secondary["owner_id"],
                    "secondary_category": secondary["category"],
                    "oracle_sampled_target_row": _post_action_probe(
                        mode="oracle_sampled_target_row",
                        post_prefix=oracle_prefix,
                        target=secondary,
                        session=opened,
                        native_inputs=one_native,
                        model=model,
                        model_inputs=model_inputs,
                        image_grid_thw=image_grid_thw,
                        base_prompt=base_prompt,
                        tokenizer=tokenizer,
                        terminal_id=int(terminal_id),
                        width=int(image_plan.decoded_width),
                        height=int(image_plan.decoded_height),
                        row_index=int(case["prefix_depth"]) + 1,
                        ledger=ledger,
                        covered_owner_ids=oracle_covered,
                        max_new_tokens=int(args.max_new_tokens),
                        malformed_limit=int(args.malformed_limit),
                    ),
                    "self_generated_target_row": None,
                }
                raw_generated = description.get("raw_generated_token_ids", [])
                self_valid = (
                    description.get("intended_owner_realized") is True
                    and isinstance(raw_generated, list)
                    and bool(raw_generated)
                    and int(raw_generated[-1]) == BOX_END_TOKEN_ID
                )
                if self_valid:
                    self_prefix = [*prefix, *[int(value) for value in raw_generated]]
                    post["self_generated_target_row"] = _post_action_probe(
                        mode="self_generated_target_row",
                        post_prefix=self_prefix,
                        target=secondary,
                        session=opened,
                        native_inputs=one_native,
                        model=model,
                        model_inputs=model_inputs,
                        image_grid_thw=image_grid_thw,
                        base_prompt=base_prompt,
                        tokenizer=tokenizer,
                        terminal_id=int(terminal_id),
                        width=int(image_plan.decoded_width),
                        height=int(image_plan.decoded_height),
                        row_index=int(case["prefix_depth"]) + 1,
                        ledger=ledger,
                        covered_owner_ids=oracle_covered,
                        max_new_tokens=int(args.max_new_tokens),
                        malformed_limit=int(args.malformed_limit),
                    )
                composition.append(post)
            output_cases.append(
                {
                    "case_id": case["case_id"],
                    "image_id": image_id,
                    "prefix_depth": int(case["prefix_depth"]),
                    "object_count_band": case["object_count_band"],
                    "target": {
                        "owner_id": target["owner_id"],
                        "category": target["category"],
                        "same_category_annotation_count": int(
                            target["same_category_annotation_count"]
                        ),
                        "row_token_ids_sha256": target["row_token_ids_sha256"],
                    },
                    "covered_owner_ids": covered,
                    "state_score": state_score,
                    "releases": {
                        "native": native,
                        "forced_opener": opener,
                        "forced_description": description,
                    },
                    "composition": composition,
                }
            )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": str(manifest["unit_id"]),
        "checkpoint_role": str(args.checkpoint_role),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "cases": output_cases,
        "runtime": {
            "shard_index": int(args.shard_index),
            "shard_count": int(args.shard_count),
            "limit": args.limit,
            "requested_case_ids": sorted(requested_case_ids),
            "case_count": len(output_cases),
            "physical_batch_size": 1,
            "runtime_dtype_mode": str(args.runtime_dtype),
            "model_dtype": model_dtype,
            "max_new_tokens": int(args.max_new_tokens),
            "malformed_limit": int(args.malformed_limit),
            "repetition_penalty": 1.0,
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
        "claim_boundary": (
            "fixed-prefix scoring and forced-row recovery are diagnostics, not free-rollout final-set outcomes"
        ),
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
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    result = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "checkpoint_role": result["checkpoint_role"],
                "case_count": result["runtime"]["case_count"],
                "shard_index": result["runtime"]["shard_index"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
