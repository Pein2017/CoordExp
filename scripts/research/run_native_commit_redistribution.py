#!/usr/bin/env python3
"""Run the bounded native row-commit redistribution panel.

This entrypoint is intentionally experiment-local.  It composes exact
model-native rows from one immutable source bundle, appends them to the exact
original prompt, and asks the existing decoder for only the next free action.
It does not add a model module, a decoder constraint, or a generic artifact
framework.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_sampled_rescue_transition import (  # noqa: E402
    _first_action,
    _request_id,
    _sha256_file,
    _write_bundle_once,
)


OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
BOX_END = 151649
COORDINATE_TOKEN_START = 151670
COORDINATE_TOKEN_END_EXCLUSIVE = 152670
IMAGE_ID = "7574"
TARGET_ANNOTATION_ID = "1535235"
OTHER_ANNOTATION_ID = "90913"
TARGET_DESCRIPTION = "Target Bowl Description"
OTHER_DESCRIPTION = "Other Bottle Description"
TARGET_GEOMETRY = "Target Bowl Geometry"
OTHER_GEOMETRY = "Other Bottle Geometry"
NO_APPENDED_ROW = "No Appended Row Baseline"
TARGET_ROW_WITH_TARGET_GEOMETRY = (
    "Target Bowl Description with Target Bowl Geometry"
)
OTHER_ROW_WITH_OTHER_GEOMETRY = (
    "Other Bottle Description with Other Bottle Geometry"
)
TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY = (
    "Target Bowl Description with Other Bottle Geometry"
)
OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY = (
    "Other Bottle Description with Target Bowl Geometry"
)
CONDITION_NAMES = (
    NO_APPENDED_ROW,
    TARGET_ROW_WITH_TARGET_GEOMETRY,
    OTHER_ROW_WITH_OTHER_GEOMETRY,
    TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY,
    OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY,
)
DEFAULT_SOURCE_BUNDLE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/executions/"
    "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls/"
    "6ea680bc021555440810a7eb43dfec3903993f0532d8c6d8d17daffc326e4b93/"
    "terminal-output-bundle.json"
)
DEFAULT_CONFIG = Path(
    "/data/CoordExp/.worktrees/research-probes/configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_gaussian_rps_dora_r16a32_step4887_val200.yaml"
)
DEFAULT_SOURCE_JSONL = Path(
    "/data/CoordExp/.worktrees/CoordExp-swift/outputs/coordexp_swift/infer/"
    "val200_inputs/coco_val200_len12000.rebased_images.coord.jsonl"
)
DEFAULT_BAGGING_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/executions/"
    "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls"
)
DEFAULT_ATTESTATION = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/runtime-attestation-v2/"
    "request-scoped-sampling-three-policy-cuda.json"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the image-7574 native commit-to-uncovered redistribution panel."
    )
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--bagging-artifact-root", type=Path, default=DEFAULT_BAGGING_ROOT)
    parser.add_argument("--sampled-runtime-attestation", type=Path, default=DEFAULT_ATTESTATION)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help=(
            "Generation horizon. The verified sampled runtime currently admits only "
            "the source policy value 512; interpretation remains first-action only."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        dest="seeds",
        help="Optional explicit replacement for canonical cell indices 0-7.",
    )
    parser.add_argument(
        "--greedy-only",
        action="store_true",
        help="Run only the five greedy requests for an execution-shape diagnostic.",
    )
    parser.add_argument(
        "--greedy-batch-size",
        type=int,
        default=4,
        help="Physical batch size for greedy requests; full panels use four.",
    )
    return parser


def _json_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=False).encode(
            "utf-8"
        )
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"expected a JSON object: {path}")
    return value


def _row_factors(tokens: Sequence[int], *, row_label: str) -> dict[str, Any]:
    """Validate a complete ten-token two-lexical-token row."""

    values = [int(token) for token in tokens]
    if len(values) != 10:
        raise SystemExit(f"{row_label} must contain exactly ten tokens")
    if values[0] != OBJECT_REF_START or values[-1] != BOX_END:
        raise SystemExit(f"{row_label} has non-canonical wrappers")
    try:
        object_end = values.index(OBJECT_REF_END, 1)
        box_start = values.index(BOX_START, object_end + 1)
    except ValueError as exc:
        raise SystemExit(f"{row_label} lacks canonical row markers") from exc
    description_tokens = values[1:object_end]
    coordinate_tokens = values[box_start + 1 : -1]
    if len(description_tokens) != 2:
        raise SystemExit(f"{row_label} must contain exactly two lexical description tokens")
    if any(
        token in {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}
        or COORDINATE_TOKEN_START <= token < COORDINATE_TOKEN_END_EXCLUSIVE
        for token in description_tokens
    ):
        raise SystemExit(f"{row_label} description contains a wrapper or coordinate token")
    if len(coordinate_tokens) != 4 or any(
        not (COORDINATE_TOKEN_START <= token < COORDINATE_TOKEN_END_EXCLUSIVE)
        for token in coordinate_tokens
    ):
        raise SystemExit(f"{row_label} must contain exactly four coordinate tokens")
    if object_end != 3 or box_start != 4:
        raise SystemExit(f"{row_label} does not have the expected ten-token layout")
    return {
        "description_token_ids": description_tokens,
        "coordinate_token_ids": coordinate_tokens,
        "token_count": len(values),
        "token_ids_sha256": _json_hash(values),
    }


def compose_row_tokens(
    description_tokens: Sequence[int],
    coordinate_tokens: Sequence[int],
    *,
    row_label: str = "composed row",
) -> list[int]:
    """Compose wrappers plus a description factor and geometry factor."""

    values = [
        OBJECT_REF_START,
        *[int(token) for token in description_tokens],
        OBJECT_REF_END,
        BOX_START,
        *[int(token) for token in coordinate_tokens],
        BOX_END,
    ]
    _row_factors(values, row_label=row_label)
    return values


def extract_source_rows(source_bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate the frozen target and other model-native rows."""

    decode = source_bundle.get("decode_result")
    if not isinstance(decode, Mapping):
        raise SystemExit("source bundle lacks decode_result")
    generated = decode.get("generated_token_ids")
    prompt = decode.get("prompt_token_ids")
    if not isinstance(generated, list) or not isinstance(prompt, list):
        raise SystemExit("source bundle lacks generated_token_ids or prompt_token_ids")
    target_tokens = [int(token) for token in generated[0:10]]
    other_tokens = [int(token) for token in generated[60:70]]
    target_factors = _row_factors(target_tokens, row_label="target bowl source row")
    other_factors = _row_factors(other_tokens, row_label="other bottle source row")
    execution = source_bundle.get("execution_evidence")
    if not isinstance(execution, Mapping) or str(execution.get("image_id")) != IMAGE_ID:
        raise SystemExit("source bundle is not the frozen image-7574 bundle")
    return {
        "prompt_token_ids": [int(token) for token in prompt],
        "prompt_token_ids_sha256": _json_hash(prompt),
        "target_row": {
            "owner": TARGET_DESCRIPTION,
            "annotation_id": TARGET_ANNOTATION_ID,
            "source_row_index": 0,
            "source_token_span": [0, 10],
            "token_ids": target_tokens,
            "factors": target_factors,
        },
        "other_row": {
            "owner": OTHER_DESCRIPTION,
            "annotation_id": OTHER_ANNOTATION_ID,
            "source_row_index": 6,
            "source_token_span": [60, 70],
            "token_ids": other_tokens,
            "factors": other_factors,
        },
    }


def build_condition_rows(source_rows: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    target = source_rows["target_row"]
    other = source_rows["other_row"]
    target_description = target["factors"]["description_token_ids"]
    other_description = other["factors"]["description_token_ids"]
    target_geometry = target["factors"]["coordinate_token_ids"]
    other_geometry = other["factors"]["coordinate_token_ids"]
    definitions = {
        NO_APPENDED_ROW: None,
        TARGET_ROW_WITH_TARGET_GEOMETRY: compose_row_tokens(
            target_description, target_geometry, row_label=TARGET_ROW_WITH_TARGET_GEOMETRY
        ),
        OTHER_ROW_WITH_OTHER_GEOMETRY: compose_row_tokens(
            other_description, other_geometry, row_label=OTHER_ROW_WITH_OTHER_GEOMETRY
        ),
        TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY: compose_row_tokens(
            target_description,
            other_geometry,
            row_label=TARGET_DESCRIPTION_WITH_OTHER_GEOMETRY,
        ),
        OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY: compose_row_tokens(
            other_description,
            target_geometry,
            row_label=OTHER_DESCRIPTION_WITH_TARGET_GEOMETRY,
        ),
    }
    result: dict[str, dict[str, Any]] = {}
    for condition_name, row_tokens in definitions.items():
        result[condition_name] = {
            "condition_name": condition_name,
            "appended_row_token_ids": None if row_tokens is None else list(row_tokens),
            "appended_row_token_ids_sha256": (
                None if row_tokens is None else _json_hash(row_tokens)
            ),
            "appended_row_token_count": 0 if row_tokens is None else len(row_tokens),
        }
    return result


def canonical_cell_seeds(
    bagging_artifact_root: Path, *, explicit_seeds: Sequence[int] | None = None
) -> dict[int, int]:
    """Read canonical cell-index seeds from immutable image-7574 bagging bundles."""

    if explicit_seeds is not None:
        if len(explicit_seeds) != 8:
            raise SystemExit("--seed must provide exactly eight values when supplied")
        return {index: int(seed) for index, seed in enumerate(explicit_seeds)}
    found: dict[int, int] = {}
    for path in sorted(bagging_artifact_root.rglob("terminal-output-bundle.json")):
        try:
            bundle = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        evidence = bundle.get("execution_evidence")
        scheduled = bundle.get("scheduled_request")
        if not isinstance(evidence, Mapping) or not isinstance(scheduled, Mapping):
            continue
        arm = evidence.get("arm")
        if not isinstance(arm, Mapping) or arm.get("arm_code") != "FULL_BAG_K":
            continue
        if str(evidence.get("image_id")) != IMAGE_ID:
            continue
        cell_index = scheduled.get("cell_index")
        seed = scheduled.get("sampling_seed")
        if isinstance(cell_index, int) and 0 <= cell_index < 8 and isinstance(seed, int):
            previous = found.setdefault(cell_index, seed)
            if previous != seed:
                raise SystemExit(f"canonical cell {cell_index} has conflicting seeds")
    if set(found) != set(range(8)):
        raise SystemExit(
            "image-7574 FULL_BAG_K artifacts do not contain canonical cell indices 0-7"
        )
    return found


def _condition_prompt(
    raw: Any,
    template_config: Any,
    processor: Any,
    *,
    condition: Mapping[str, Any],
    tokenizer: Any,
) -> Any:
    from src.inference.prompt import AssistantContinuation, build_prompt_record

    row_tokens = condition["appended_row_token_ids"]
    continuation = None
    if row_tokens is not None:
        continuation = AssistantContinuation(
            text=tokenizer.decode(
                row_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        )
    return build_prompt_record(
        raw,
        template_config,
        processor=processor,
        row_index=0,
        assistant_continuation=continuation,
    )


def _condition_receipt(
    *,
    result: Any,
    condition: Mapping[str, Any],
    prompt_record: Any,
    raw: Any,
    sampling_seed: int | None,
    source_bundle_path: Path,
) -> dict[str, Any]:
    from src.inference.parsing import parse_compact_object_box_closed

    parsed = parse_compact_object_box_closed(
        result.parser_text,
        row_id=result.request_id,
        row_index=0,
        image_width=raw.image.width,
        image_height=raw.image.height,
    )
    first_action = _first_action(
        parsed,
        result.parser_text,
        generated_token_ids=result.generated_token_ids,
        stop_reason=result.stop_reason,
    )
    decode = result.to_artifact_dict()
    bundle = {
        "schema_version": "native_commit_redistribution.call_bundle.v1",
        "request_id": result.request_id,
        "condition_name": condition["condition_name"],
        "sampling_seed": sampling_seed,
        "source_bundle_sha256": _sha256_file(source_bundle_path),
        "prompt_token_ids_sha256": _json_hash(prompt_record.prompt_token_ids),
        "appended_row_token_ids": condition["appended_row_token_ids"],
        "appended_row_token_ids_sha256": condition["appended_row_token_ids_sha256"],
        "decode_result": decode,
        "parse_result": parsed.to_artifact_dict(),
        "first_free_action": first_action,
    }
    return bundle


def _validate_identity_continuity(
    *,
    source_bundle: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    tokenizer_identity: Mapping[str, Any],
    generation_config_fingerprint: str,
    source_image_sha256: str,
) -> dict[str, bool]:
    """Fail before decoding if the frozen source lineage is not active."""

    source_decode = source_bundle.get("decode_result")
    source_execution = source_bundle.get("execution_evidence")
    if not isinstance(source_decode, Mapping) or not isinstance(source_execution, Mapping):
        raise SystemExit("source bundle lacks identity-bearing decode or execution evidence")
    checks = {
        "model_identity_matches_source": dict(model_identity)
        == source_decode.get("model_identity"),
        "tokenizer_identity_matches_source": dict(tokenizer_identity)
        == source_decode.get("tokenizer_identity"),
        "generation_config_fingerprint_matches_source": generation_config_fingerprint
        == source_decode.get("generation_config_fingerprint"),
        "source_image_sha256_matches_source": source_image_sha256
        == source_execution.get("source_image_sha256"),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise SystemExit(f"source lineage identity check failed: {failed}")
    return checks


def _validate_executed_receipt_continuity(
    *,
    source_bundle: Mapping[str, Any],
    greedy_results: Sequence[Any],
    sampled_results: Sequence[Any],
) -> dict[str, Any]:
    """Require executed model/runtime identities and sampled policy continuity."""

    source_decode = source_bundle.get("decode_result")
    if not isinstance(source_decode, Mapping):
        raise SystemExit("source bundle lacks decode_result")
    source_receipt = source_decode.get("execution_receipt")
    if not isinstance(source_receipt, Mapping):
        raise SystemExit("source bundle lacks decode execution receipt")
    identity_fields = (
        "model_identity_fingerprint",
        "tokenizer_identity_fingerprint",
        "installed_runtime_identity_fingerprint",
        "generation_config_fingerprint",
    )
    sampler_fields = (
        "custom_sampler_identity",
        "sampling_profile",
    )
    all_results = [*greedy_results, *sampled_results]
    for result in all_results:
        receipt = result.to_artifact_dict().get("execution_receipt")
        if not isinstance(receipt, Mapping):
            raise SystemExit(f"decode {result.request_id} lacks an execution receipt")
        mismatches = {
            field: {
                "source": source_receipt.get(field),
                "current": receipt.get(field),
            }
            for field in identity_fields
            if receipt.get(field) != source_receipt.get(field)
        }
        if mismatches:
            raise SystemExit(
                f"decode {result.request_id} violates source identity continuity: {mismatches}"
            )
    for result in sampled_results:
        receipt = result.to_artifact_dict()["execution_receipt"]
        policy = receipt.get("decode_generation_policy")
        expected_policy = {
            "max_new_tokens": 512,
            "mode": "sampled",
            "repetition_penalty": 1.0,
            "sampling_profile": "temperature_top_p_categorical_v1",
            "temperature": 0.4,
            "top_p": 0.95,
        }
        if policy != expected_policy:
            raise SystemExit(
                f"decode {result.request_id} violates the frozen sampled policy: {policy}"
            )
        mismatches = {
            field: {
                "source": source_receipt.get(field),
                "current": receipt.get(field),
            }
            for field in sampler_fields
            if receipt.get(field) != source_receipt.get(field)
        }
        if mismatches:
            raise SystemExit(
                f"decode {result.request_id} violates sampler identity continuity: {mismatches}"
            )
    return {
        "executed_result_count": len(all_results),
        "identity_fields_matching_source": list(identity_fields),
        "sampled_policy_matches_source": True,
        "sampled_sampler_fields_matching_source": list(sampler_fields),
    }


def run_panel(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the panel through the current inference backend."""

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
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.runtime import assemble_runtime

    source_bundle_path = args.source_bundle.expanduser().resolve(strict=True)
    source_bundle = _read_json(source_bundle_path)
    source_rows = extract_source_rows(source_bundle)
    conditions = build_condition_rows(source_rows)
    seeds = canonical_cell_seeds(
        args.bagging_artifact_root.expanduser().resolve(), explicit_seeds=args.seeds
    )
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    raw_rows = load_raw_examples(source_path)
    raw = next((row for row in raw_rows if str(row.metadata.get("source", {}).get("image_id")) == IMAGE_ID), None)
    if raw is None:
        raise SystemExit("image-7574 source row was not found")
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    processor = qwen.processor
    template_config = _template_config(resolved.config)
    prompt_records = {
        name: _condition_prompt(
            raw,
            template_config,
            processor,
            condition=condition,
            tokenizer=qwen.tokenizer,
        )
        for name, condition in conditions.items()
    }
    base_prompt_ids = source_rows["prompt_token_ids"]
    if prompt_records[NO_APPENDED_ROW].prompt_token_ids != base_prompt_ids:
        raise SystemExit("fresh no-row prompt does not equal the frozen source prompt")
    for name, prompt in prompt_records.items():
        row_tokens = conditions[name]["appended_row_token_ids"]
        expected = base_prompt_ids if row_tokens is None else [*base_prompt_ids, *row_tokens]
        if list(prompt.prompt_token_ids) != expected:
            raise SystemExit(f"{name} prompt does not equal frozen prompt plus exact row")
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = image_plan.model_inputs_by_row_id[prompt_records[NO_APPENDED_ROW].row_id]
    generation_fingerprint = sha256_json(resolved.config.generation.model_dump(mode="json"))
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    predecode_identity_checks = _validate_identity_continuity(
        source_bundle=source_bundle,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
        source_image_sha256=_sha256_file(raw.image.path),
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    boundary_hashes = {
        name: _json_hash(prompt.prompt_token_ids) for name, prompt in prompt_records.items()
    }
    greedy_policy = DecodeGenerationPolicy.greedy(
        max_new_tokens=int(args.max_new_tokens), repetition_penalty=1.0
    )
    sampled_policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=int(args.max_new_tokens),
        repetition_penalty=1.0,
        temperature=0.4,
        top_p=0.95,
    )
    greedy_requests = [
        DecodeRequest(
            request_id=_request_id(
                image_id=IMAGE_ID,
                boundary_role="native_commit_redistribution",
                prompt_token_hash=boundary_hashes[name],
                kind="greedy",
                index=index,
                condition_name=name,
            ),
            prompt_token_ids=list(prompt_records[name].prompt_token_ids),
            model_inputs=model_inputs,
            generation_policy=greedy_policy,
        )
        for index, name in enumerate(CONDITION_NAMES)
    ]
    greedy_results: list[Any] = []
    for start in range(0, len(greedy_requests), int(args.greedy_batch_size)):
        greedy_results.extend(
            backend.generate_batch(
                greedy_requests[start : start + int(args.greedy_batch_size)],
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_fingerprint,
            )
        )
    sampled_requests: list[DecodeRequest] = []
    sampled_results: list[Any] = []
    if not args.greedy_only:
        capability = load_and_rebind_sampled_runtime_attestation_aggregate(
            args.sampled_runtime_attestation.expanduser().resolve(strict=True),
            decode_generation_policy_fingerprint=sampled_policy.fingerprint,
            backend=backend,
        )
        for condition_index, name in enumerate(CONDITION_NAMES):
            for cell_index in range(8):
                sampled_requests.append(
                    DecodeRequest(
                        request_id=_request_id(
                            image_id=IMAGE_ID,
                            boundary_role="native_commit_redistribution",
                            prompt_token_hash=boundary_hashes[name],
                            kind="sample",
                            index=cell_index,
                            seed=seeds[cell_index],
                            sampling_temperature=0.4,
                            condition_name=name,
                        ),
                        prompt_token_ids=list(prompt_records[name].prompt_token_ids),
                        model_inputs=model_inputs,
                        generation_policy=sampled_policy,
                        sampling_seed=seeds[cell_index],
                    )
                )
        for start in range(0, len(sampled_requests), 4):
            sampled_results.extend(
                backend.generate_batch_with_verified_runtime_attestation(
                    sampled_requests[start : start + 4],
                    model_identity=model_identity,
                    tokenizer_identity=tokenizer_identity,
                    generation_config_fingerprint=generation_fingerprint,
                    verified_runtime_attestation=capability,
                )
            )
    executed_identity_checks = _validate_executed_receipt_continuity(
        source_bundle=source_bundle,
        greedy_results=greedy_results,
        sampled_results=sampled_results,
    )
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    source_hash = _sha256_file(source_bundle_path)
    output_conditions: list[dict[str, Any]] = []
    sampled_offset = 0
    for condition_index, name in enumerate(CONDITION_NAMES):
        condition = conditions[name]
        greedy_bundle = _condition_receipt(
            result=greedy_results[condition_index],
            condition=condition,
            prompt_record=prompt_records[name],
            raw=raw,
            sampling_seed=None,
            source_bundle_path=source_bundle_path,
        )
        condition_dir = output_root / f"condition-{condition_index:02d}"
        _write_bundle_once(condition_dir / "greedy-terminal-output-bundle.json", greedy_bundle)
        sample_bundles: list[dict[str, Any]] = []
        sample_count = 0 if args.greedy_only else 8
        for cell_index in range(sample_count):
            sample_bundle = _condition_receipt(
                result=sampled_results[sampled_offset],
                condition=condition,
                prompt_record=prompt_records[name],
                raw=raw,
                sampling_seed=seeds[cell_index],
                source_bundle_path=source_bundle_path,
            )
            sampled_offset += 1
            _write_bundle_once(
                condition_dir / f"sample-cell-{cell_index:02d}-terminal-output-bundle.json",
                sample_bundle,
            )
            sample_bundles.append(sample_bundle)
        output_conditions.append(
            {
                **condition,
                "prompt_token_count": len(prompt_records[name].prompt_token_ids),
                "prompt_token_ids_sha256": boundary_hashes[name],
                "prompt_equals_frozen_prompt_plus_row": True,
                "greedy_first_free_action": greedy_bundle["first_free_action"],
                "sampled_first_free_actions": [
                    bundle["first_free_action"] for bundle in sample_bundles
                ],
                "greedy_bundle": str((condition_dir / "greedy-terminal-output-bundle.json").resolve()),
                "sample_bundle_paths": [
                    str((condition_dir / f"sample-cell-{cell_index:02d}-terminal-output-bundle.json").resolve())
                    for cell_index in range(8)
                ],
            }
        )
    result = {
        "schema_version": "native_commit_redistribution.panel_receipt.v1",
        "experiment_name": "Native Commit-to-Uncovered Redistribution",
        "execution_mode": "greedy_only" if args.greedy_only else "full_panel",
        "image_id": IMAGE_ID,
        "source_bundle": {
            "path": str(source_bundle_path),
            "sha256": source_hash,
            "request_id": source_bundle.get("request_id"),
        },
        "frozen_original_prompt": {
            "token_count": len(base_prompt_ids),
            "token_ids_sha256": source_rows["prompt_token_ids_sha256"],
            "zero_generated_prefix_tokens": True,
        },
        "row_components": {
            "target_bowl": source_rows["target_row"],
            "other_bottle": source_rows["other_row"],
        },
        "paired_seed_schedule": {
            "canonical_cell_indices": list(range(8)),
            "sampling_seeds": {str(index): seed for index, seed in seeds.items()},
            "source": "image-7574 FULL_BAG_K terminal bundles",
        },
        "decode_policy": {
            "repetition_penalty": 1.0,
            "temperature": 0.4,
            "top_p": 0.95,
            "max_new_tokens": int(args.max_new_tokens),
            "greedy_max_new_tokens": int(args.max_new_tokens),
            "greedy_physical_batch_size": int(args.greedy_batch_size),
        },
        "identity_continuity": {
            "predecode": predecode_identity_checks,
            "executed_receipts": executed_identity_checks,
        },
        "conditions": output_conditions,
    }
    _write_bundle_once(
        output_root / "receipt.json",
        result,
    )
    return result


def _temporary_cwd(path: Path):
    import contextlib
    import os

    @contextlib.contextmanager
    def manager():
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)

    return manager()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.greedy_batch_size <= 0:
        raise SystemExit("--greedy-batch-size must be positive")
    result = run_panel(args)
    print(json.dumps({"output_root": str(args.output_root.resolve()), "conditions": len(result["conditions"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
