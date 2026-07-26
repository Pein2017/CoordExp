#!/usr/bin/env python3
"""Run paired native and opener-forced one-row releases at Source natural stops."""

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
    _runtime_model_dtype_summary,
)
from scripts.research.run_exact_prefix_owner_compositionality import (  # noqa: E402
    _annotated_release,
)
from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _single_native_inputs,
    build_positive_entity_ledger,
)
from scripts.research.run_native_sibling_branch_replay import (  # noqa: E402
    _attention_implementation,
)
from src.config.fingerprint import sha256_file  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402


RECEIPT_SCHEMA_VERSION = "paired_terminal_forced_opener_release.receipt.v1"
UNIT_ID = "2026-07-25-paired-natural-terminal-forced-opener-release"
SOURCE_UNIT_ID = "2026-07-25-continuation-locality-and-exact-prefix-owner-compositionality"
COHORT = "untouched_terminal_with_remaining_owner"
BOX_END_TOKEN_ID = 151649


def _validated_checkpoint_contract(
    *,
    args: argparse.Namespace,
    config_path: Path,
    backend_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    role = str(args.checkpoint_role)
    expected_values = (
        args.expected_authored_config_sha256,
        args.expected_adapter_path,
        args.expected_adapter_sha256,
        args.expected_embedding_delta_path,
        args.expected_embedding_delta_sha256,
    )
    if role != "source" and not all(expected_values):
        raise ValueError(
            "non-Source checkpoint roles require expected config, adapter, and embedding identities"
        )
    config_hash = sha256_file(config_path)
    if (
        args.expected_authored_config_sha256
        and config_hash != args.expected_authored_config_sha256
    ):
        raise ValueError("authored inference-config hash differs from expected contract")

    model_identity = backend_receipt.get("model_identity")
    if not isinstance(model_identity, Mapping):
        raise ValueError("backend receipt lacks model identity")
    adapter = model_identity.get("adapter")
    embedding = model_identity.get("embedding_delta")
    if not isinstance(adapter, Mapping) or not isinstance(embedding, Mapping):
        raise ValueError("backend receipt lacks adapter or embedding-delta identity")
    embedding_identity = embedding.get("identity")
    if not isinstance(embedding_identity, Mapping):
        raise ValueError("backend receipt lacks embedding-delta path identity")

    actual_adapter = Path(str(adapter.get("adapter_path"))).resolve(strict=True)
    actual_embedding = Path(str(embedding_identity.get("delta_path"))).resolve(strict=True)
    if args.expected_adapter_path is not None:
        expected_adapter = args.expected_adapter_path.expanduser().resolve(strict=True)
        if actual_adapter != expected_adapter:
            raise ValueError("loaded adapter path differs from expected contract")
    if args.expected_embedding_delta_path is not None:
        expected_embedding = args.expected_embedding_delta_path.expanduser().resolve(strict=True)
        if actual_embedding != expected_embedding:
            raise ValueError("loaded embedding-delta path differs from expected contract")

    adapter_hash = sha256_file(actual_adapter / "adapter_model.safetensors")
    embedding_hash = sha256_file(
        actual_embedding / "special_token_embeddings.safetensors"
    )
    if args.expected_adapter_sha256 and adapter_hash != args.expected_adapter_sha256:
        raise ValueError("loaded adapter tensor hash differs from expected contract")
    if (
        args.expected_embedding_delta_sha256
        and embedding_hash != args.expected_embedding_delta_sha256
    ):
        raise ValueError("loaded embedding tensor hash differs from expected contract")
    base = model_identity.get("base")
    return {
        "checkpoint_role": role,
        "authored_config_sha256": config_hash,
        "adapter_path": str(actual_adapter),
        "adapter_tensor_sha256": adapter_hash,
        "embedding_delta_path": str(actual_embedding),
        "embedding_delta_tensor_sha256": embedding_hash,
        "model_family": model_identity.get("family"),
        "base_model_path": base.get("path") if isinstance(base, Mapping) else None,
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _stable_shard(value: str, count: int) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16) % count


def _token_ids(value: Any, *, label: str) -> list[int]:
    if not isinstance(value, list) or not value or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in value
    ):
        raise ValueError(f"{label} must be a non-empty token-id list")
    return [int(item) for item in value]


def _validated_boundaries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected manifest schema {MANIFEST_SCHEMA_VERSION}")
    if manifest.get("unit_id") != SOURCE_UNIT_ID:
        raise ValueError(f"unexpected source unit: {manifest.get('unit_id')}")
    values = manifest.get("locality_boundaries")
    if not isinstance(values, list):
        raise ValueError("manifest locality_boundaries must be a list")
    selected = [value for value in values if value.get("cohort") == COHORT]
    if len(selected) != 200:
        raise ValueError(f"expected 200 terminal boundaries, found {len(selected)}")
    results: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_images: set[str] = set()
    for index, raw in enumerate(selected):
        if not isinstance(raw, Mapping):
            raise ValueError(f"terminal boundary {index} must be an object")
        item = dict(raw)
        boundary_id = str(item.get("boundary_id", ""))
        image_id = str(item.get("image_id", ""))
        if not boundary_id or boundary_id in seen:
            raise ValueError(f"invalid or duplicate boundary_id={boundary_id!r}")
        if not image_id or image_id in seen_images:
            raise ValueError(f"invalid or duplicate image_id={image_id!r}")
        seen.add(boundary_id)
        seen_images.add(image_id)
        prompt = _token_ids(
            item.get("base_prompt_token_ids"), label=f"{boundary_id}.base_prompt"
        )
        prefix = _token_ids(item.get("prefix_token_ids"), label=f"{boundary_id}.prefix")
        if token_ids_sha256(prompt) != item.get("base_prompt_token_ids_sha256"):
            raise ValueError(f"{boundary_id} base prompt hash mismatch")
        if token_ids_sha256(prefix) != item.get("prefix_token_ids_sha256"):
            raise ValueError(f"{boundary_id} prefix hash mismatch")
        if prefix[-1] != BOX_END_TOKEN_ID:
            raise ValueError(f"{boundary_id} is not a complete-row boundary")
        if item.get("observed_next_action") != "terminal" or item.get("natural_end") is not True:
            raise ValueError(f"{boundary_id} is not an observed Source natural stop")
        remaining = item.get("census_uncovered_owner_ids_at_stop")
        covered = item.get("covered_owner_ids")
        if not isinstance(remaining, list) or not remaining:
            raise ValueError(f"{boundary_id} lacks remaining owners")
        if not isinstance(covered, list):
            raise ValueError(f"{boundary_id} lacks covered owners")
        if int(item.get("remaining_annotation_owner_count", -1)) != len(remaining):
            raise ValueError(f"{boundary_id} remaining-owner count mismatch")
        results.append(
            {
                **item,
                "boundary_id": boundary_id,
                "image_id": image_id,
                "base_prompt_token_ids": prompt,
                "prefix_token_ids": prefix,
                "covered_owner_ids": [str(value) for value in covered],
                "census_uncovered_owner_ids_at_stop": [str(value) for value in remaining],
            }
        )
    return results


def run(args: argparse.Namespace) -> dict[str, Any]:
    unit_id = str(args.unit_id)
    checkpoint_role = str(args.checkpoint_role)
    shard_count = int(args.shard_count)
    shard_index = int(args.shard_index)
    if shard_count <= 0 or not 0 <= shard_index < shard_count:
        raise ValueError("invalid shard index/count")
    if int(args.max_new_tokens) <= 0 or int(args.malformed_limit) <= 0:
        raise ValueError("generation limits must be positive")

    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    boundaries = _validated_boundaries(manifest)
    candidate_input = manifest.get("inputs", {}).get("candidate_pool", {})
    candidate_path = Path(str(candidate_input.get("path"))).resolve(strict=True)
    if sha256_file(candidate_path) != candidate_input.get("sha256"):
        raise ValueError("manifest candidate-pool hash mismatch")
    requested = set(args.boundary_id or [])
    selected = [
        boundary
        for boundary in boundaries
        if _stable_shard(str(boundary["boundary_id"]), shard_count) == shard_index
        and (not requested or str(boundary["boundary_id"]) in requested)
    ]
    selected.sort(key=lambda boundary: str(boundary["boundary_id"]))
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    if not selected:
        raise ValueError("selected terminal-boundary shard is empty")

    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_jsonl = args.source_jsonl.expanduser().resolve(strict=True)
    if source_jsonl != candidate_path:
        raise ValueError("declared source JSONL differs from frozen manifest candidate pool")
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
    config = resolved.config.model_copy(
        update={"model": resolved.config.model.model_copy(update={"dtype": "fp32"})}
    )
    if config.backend.type != "hf":
        raise ValueError("paired terminal release requires backend.type: hf")
    if float(config.generation.repetition_penalty) != 1.0:
        raise ValueError("paired terminal release requires repetition_penalty=1.0")

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    raw_by_id: dict[str, Any] = {}
    for row in load_raw_examples(source_jsonl):
        source = row.metadata.get("source")
        if not isinstance(source, Mapping) or source.get("image_id") is None:
            raise ValueError("source JSONL row lacks source.image_id metadata")
        raw_by_id[str(source["image_id"])] = row
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
        parity = verify_processor_model_vision_parity(
            processor_identity=frontend.qwen.processor_identity,
            model_config=model.config,
        )
        backend_receipt = opened.receipt.to_artifact_dict()
        checkpoint_contract = _validated_checkpoint_contract(
            args=args,
            config_path=config_path,
            backend_receipt=backend_receipt,
        )
        model_dtype = _runtime_model_dtype_summary(model)
        attention = _attention_implementation(model, config.backend.hf.attn_implementation)

        for boundary in selected:
            boundary_id = str(boundary["boundary_id"])
            image_id = str(boundary["image_id"])
            raw = raw_by_id.get(image_id)
            if raw is None:
                raise ValueError(f"image {image_id} is absent from source JSONL")
            image_plan = plan_image_batch(
                [raw],
                components=frontend.qwen,
                processor_config=_processor_config(config),
                row_indices=[0],
            ).rows[0]
            if image_plan.image_content_sha256 != boundary["image_content_sha256"]:
                raise ValueError(f"{boundary_id} image hash mismatch")
            prompt_record = build_prompt_record(
                raw,
                template,
                processor=frontend.qwen.processor,
                row_index=0,
                merged_visual_tokens=image_plan.merged_visual_tokens,
            )
            base_prompt = [int(value) for value in prompt_record.prompt_token_ids]
            if base_prompt != boundary["base_prompt_token_ids"]:
                raise ValueError(f"{boundary_id} active prompt mismatch")
            expected_grid = tuple(int(value) for value in image_plan.expected_image_grid_thw)
            if len(expected_grid) != 3:
                raise ValueError(f"{boundary_id} expected image grid is not rank three")
            request = DecodeRequest(
                request_id=f"paired-terminal-release:{checkpoint_role}:{boundary_id}",
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
                raise RuntimeError(f"{boundary_id} materialized prompt mismatch")
            image_grid_thw = native_inputs.get("image_grid_thw")
            if not isinstance(image_grid_thw, torch.Tensor):
                raise ValueError("native inputs lack image_grid_thw")
            one_native = _single_native_inputs(native_inputs)
            ledger = build_positive_entity_ledger(raw)
            prefix = list(boundary["prefix_token_ids"])
            covered = list(boundary["covered_owner_ids"])
            native = _annotated_release(
                kind="native",
                session=opened,
                native_inputs=one_native,
                prefix=prefix,
                forced=None,
                tokenizer=tokenizer,
                width=int(image_plan.decoded_width),
                height=int(image_plan.decoded_height),
                row_index=int(boundary["prefix_depth"]),
                ledger=ledger,
                covered_owner_ids=covered,
                intended_owner_id=None,
                max_new_tokens=int(args.max_new_tokens),
                malformed_limit=int(args.malformed_limit),
            )
            forced = _annotated_release(
                kind="forced_opener",
                session=opened,
                native_inputs=one_native,
                prefix=prefix,
                forced=[OBJECT_REF_START],
                tokenizer=tokenizer,
                width=int(image_plan.decoded_width),
                height=int(image_plan.decoded_height),
                row_index=int(boundary["prefix_depth"]),
                ledger=ledger,
                covered_owner_ids=covered,
                intended_owner_id=None,
                max_new_tokens=int(args.max_new_tokens),
                malformed_limit=int(args.malformed_limit),
            )
            output_cases.append(
                {
                    "boundary_id": boundary_id,
                    "image_id": image_id,
                    "prefix_depth": int(boundary["prefix_depth"]),
                    "object_count_band": str(boundary["object_count_band"]),
                    "annotation_object_count": int(boundary["annotation_object_count"]),
                    "covered_owner_ids": covered,
                    "remaining_owner_ids": list(
                        boundary["census_uncovered_owner_ids_at_stop"]
                    ),
                    "historical_observed_next_action": str(
                        boundary["observed_next_action"]
                    ),
                    "historical_natural_end": bool(boundary["natural_end"]),
                    "base_prompt_token_ids_sha256": token_ids_sha256(base_prompt),
                    "prefix_token_ids_sha256": token_ids_sha256(prefix),
                    "releases": {"native": native, "forced_opener": forced},
                }
            )

    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": unit_id,
        "checkpoint_role": checkpoint_role,
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "cases": output_cases,
        "runtime": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "limit": args.limit,
            "requested_boundary_ids": sorted(requested),
            "case_count": len(output_cases),
            "arm_order": ["native", "forced_opener"],
            "physical_batch_size": 1,
            "runtime_dtype_mode": "fp32",
            "model_dtype": model_dtype,
            "max_new_tokens": int(args.max_new_tokens),
            "malformed_limit": int(args.malformed_limit),
            "repetition_penalty": 1.0,
            "temperature": 0.0,
            "top_p": 1.0,
            "config_path": str(config_path),
            "authored_config_sha256": sha256_file(config_path),
            "resolved_config_fingerprint": resolved.fingerprint,
            "effective_config_sha256": config_sha256_json(
                config.model_dump(mode="json")
            ),
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256": sha256_file(source_jsonl),
            "authored_config_data_input_jsonl": str(config.data.input_jsonl),
            "attention_implementation": attention,
            "processor_model_vision_parity": parity,
            "backend_session": backend_receipt,
            "checkpoint_contract": checkpoint_contract,
            "forced_opener_token_id": OBJECT_REF_START,
        },
        "claim_boundary": (
            "paired one-row forced continuation is a causal diagnostic, not a free-rollout final-set outcome"
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
    parser.add_argument("--unit-id", default=UNIT_ID)
    parser.add_argument("--checkpoint-role", default="source")
    parser.add_argument("--expected-authored-config-sha256")
    parser.add_argument("--expected-adapter-path", type=Path)
    parser.add_argument("--expected-adapter-sha256")
    parser.add_argument("--expected-embedding-delta-path", type=Path)
    parser.add_argument("--expected-embedding-delta-sha256")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--boundary-id", action="append")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    receipt = run(build_parser().parse_args())
    print(
        json.dumps(
            {
                "case_count": receipt["runtime"]["case_count"],
                "output_schema": receipt["schema_version"],
                "shard_index": receipt["runtime"]["shard_index"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
