#!/usr/bin/env python3
"""Analyze sampled-rescue bundles and run bounded fixed-prefix causal replay.

Production inference remains the runtime owner. This research entrypoint adds
only exact-prefix and donor-span interventions around those existing seams.
"""

from __future__ import annotations

import argparse
import json
import hashlib
import math
import os
from pathlib import Path
import sys
from typing import Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.analysis.sampled_rescue_transition.artifacts import (
    load_case_table,
    load_call_records,
)
from src.analysis.sampled_rescue_transition.comparison import (
    BOX_END,
    BOX_START,
    OBJECT_REF_END,
    OBJECT_REF_START,
    trajectory_rows,
)
from src.analysis.spatial_scope_history.schedule import PRIMARY_ROOT_SEED


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze exact sampled-rescue spatial-scope terminal bundles."
    )
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--image-id", action="append", dest="image_ids")
    parser.add_argument("--arm", action="append", dest="arm_codes")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--bundle-root", type=Path)
    parser.add_argument("--audit-ledger", type=Path)
    parser.add_argument(
        "--mode",
        choices=(
            "case-table",
            "calls",
            "greedy-anchor",
            "fixed-prefix-smoke",
            "runtime-smoke",
            "forced-span-greedy",
            "forced-span-smoke",
            "composed-row-greedy",
            "composed-row-smoke",
        ),
        default="case-table",
        help="Read-only artifact projection; no model or decoder mutation.",
    )
    parser.add_argument("--infer-config", type=Path)
    parser.add_argument("--source-jsonl", type=Path)
    parser.add_argument("--sampled-runtime-attestation", type=Path)
    parser.add_argument(
        "--continuation-text",
        help="Canonical assistant continuation used by fixed-prefix-smoke.",
    )
    parser.add_argument(
        "--donor-bundle",
        type=Path,
        help="Terminal bundle whose prompt tokens must equal the fixed-prefix prompt.",
    )
    parser.add_argument(
        "--donor-prefix-token-count",
        type=int,
        help="Number of donor generated tokens forming the fixed boundary prefix.",
    )
    parser.add_argument(
        "--boundary-role",
        choices=("common_pre_row", "phase_divergence", "rescue_entry", "greedy_terminal"),
        help="Scientific role of the donor boundary for fixed-prefix-smoke.",
    )
    parser.add_argument("--seed", action="append", type=int, dest="seeds")
    parser.add_argument(
        "--seed-root",
        type=int,
        default=PRIMARY_ROOT_SEED,
        help="Immutable root seed accepted by the shared sampling schedule.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--recipient-bundle", type=Path)
    parser.add_argument("--recipient-prefix-token-count", type=int)
    parser.add_argument("--forced-donor-bundle", type=Path)
    parser.add_argument(
        "--description-row-receipt",
        type=Path,
        help="Verified forced-row receipt supplying one description token.",
    )
    parser.add_argument(
        "--geometry-row-receipt",
        type=Path,
        help="Verified forced-row receipt supplying four coordinate tokens.",
    )
    parser.add_argument("--forced-span-start", type=int)
    parser.add_argument("--forced-span-end", type=int)
    parser.add_argument(
        "--forced-span-role",
        choices=(
            "row_opener",
            "first_description_token",
            "complete_description",
            "first_coordinate",
            "complete_row",
        ),
    )
    parser.add_argument("--condition-name", default="forced-span")
    parser.add_argument("--incremental-predecessor", type=Path)
    parser.add_argument(
        "--forced-context-policy",
        choices=("exact", "syntax_only"),
        default="exact",
    )
    parser.add_argument(
        "--sampling-temperature",
        type=_positive_finite_temperature,
        default=0.4,
        help="Positive finite sampled temperature (runtime attestation admits 0.2, 0.4, and 0.6).",
    )
    return parser


def _positive_finite_temperature(value: str) -> float:
    """Parse a sampled temperature without allowing silent NaN/zero inputs."""

    try:
        temperature = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("temperature must be a finite positive float") from exc
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise argparse.ArgumentTypeError("temperature must be a finite positive float")
    return temperature


def _temperature_label(temperature: float) -> str:
    """Return a stable filesystem/request/seed label for one temperature."""

    if not math.isfinite(float(temperature)) or float(temperature) <= 0.0:
        raise ValueError("temperature must be finite and positive")
    return format(float(temperature), ".12g").replace("-", "m").replace(".", "p")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    images = tuple(args.image_ids or ())
    arms = tuple(args.arm_codes or ("FULL_BAG_K", "FULL_SINGLE"))
    if args.mode in {
        "forced-span-greedy",
        "forced-span-smoke",
        "composed-row-greedy",
        "composed-row-smoke",
    }:
        result = _run_forced_span_runtime(
            args,
            include_samples=args.mode in {"forced-span-smoke", "composed-row-smoke"},
        )
    elif args.mode in {"greedy-anchor", "fixed-prefix-smoke", "runtime-smoke"}:
        result = _run_runtime_smoke(
            args,
            include_samples=args.mode in {"fixed-prefix-smoke", "runtime-smoke"},
        )
    elif args.mode == "case-table":
        if args.artifact_root is None:
            raise SystemExit("--artifact-root is required for --mode case-table")
        if not images:
            raise SystemExit("--image-id is required for --mode case-table")
        result = load_case_table(
            args.artifact_root,
            image_ids=images,
            audit_ledger_path=args.audit_ledger,
        )
    else:
        if args.artifact_root is None:
            raise SystemExit("--artifact-root is required for --mode calls")
        records = load_call_records(
            args.artifact_root,
            image_ids=images or None,
            arm_codes=arms,
        )
        result = {
            "schema_version": "sampled_rescue_transition.call_projection.v1",
            "artifact_root": str(args.artifact_root.resolve()),
            "calls": [record.to_json_dict() for record in records],
        }
    text = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


def _run_runtime_smoke(
    args: argparse.Namespace, *, include_samples: bool = True
) -> dict[str, object]:
    """Run one greedy anchor plus eight sampled requests through current seams."""

    if args.infer_config is None or args.source_jsonl is None or len(args.image_ids or ()) != 1:
        raise SystemExit("runtime-smoke requires --infer-config, --source-jsonl, and exactly one --image-id")
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        DecodeRequest,
        HFGenerateBackend,
        load_and_rebind_sampled_runtime_attestation_aggregate,
    )
    from src.inference.image_plan import materialize_image_plan_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config, _tokenizer_identity
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.parsing import parse_compact_object_box_closed
    from src.inference.runtime import assemble_runtime
    from src.analysis.spatial_scope_history.schedule import derive_sampling_seed

    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    requested_image_id = str(args.image_ids[0])
    image_id = requested_image_id
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    raw_rows = load_raw_examples(source_path)
    raw_by_id = {row.example_id: row for row in raw_rows}
    raw_by_numeric_id = {}
    for row in raw_rows:
        source_meta = row.metadata.get("source")
        source_image_id = (
            source_meta.get("image_id")
            if hasattr(source_meta, "get")
            else None
        )
        if source_image_id is not None:
            raw_by_numeric_id[str(int(source_image_id))] = row
    raw = raw_by_id.get(image_id)
    if raw is None and requested_image_id.isdigit():
        raw = raw_by_numeric_id.get(str(int(requested_image_id)))
    if raw is None:
        raise SystemExit(f"image id not found in source JSONL: {image_id}")
    source_meta = raw.metadata.get("source")
    source_image_id = source_meta.get("image_id") if hasattr(source_meta, "get") else None
    if source_image_id is None:
        raise SystemExit(f"source JSONL row lacks numeric source image_id: {raw.example_id}")
    image_id = str(int(source_image_id))
    example_id = raw.example_id
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    processor = qwen.processor
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    continuation_text = getattr(args, "continuation_text", None)
    donor_prefix_ids: list[int] | None = None
    donor_prompt_ids: list[int] | None = None
    if args.mode == "fixed-prefix-smoke":
        if args.donor_bundle is None or args.donor_prefix_token_count is None:
            raise SystemExit(
                "fixed-prefix-smoke requires --donor-bundle and --donor-prefix-token-count"
            )
        if args.boundary_role is None:
            raise SystemExit("fixed-prefix-smoke requires --boundary-role")
        donor = json.loads(
            args.donor_bundle.expanduser().resolve(strict=True).read_text(encoding="utf-8")
        )
        donor_decode = donor.get("decode_result", donor)
        if not isinstance(donor_decode, dict):
            raise SystemExit("donor bundle decode result must be an object")
        donor_prompt_ids_raw = donor_decode.get("prompt_token_ids")
        donor_generated_ids = donor_decode.get("generated_token_ids")
        if not isinstance(donor_prompt_ids_raw, list) or not isinstance(donor_generated_ids, list):
            raise SystemExit(
                "donor bundle requires decode_result.prompt_token_ids and generated_token_ids"
            )
        donor_prompt_ids = [int(token) for token in donor_prompt_ids_raw]
        count = int(args.donor_prefix_token_count)
        if count <= 0 or count > len(donor_generated_ids):
            raise SystemExit("donor-prefix-token-count must be within donor generated token range")
        donor_prefix_ids = [int(token) for token in donor_generated_ids[:count]]
        if any(int(token) == OBJECT_REF_START for token in donor_prefix_ids) is False:
            raise SystemExit("donor prefix must contain at least one object-reference token")
        decoded_prefix = qwen.tokenizer.decode(
            donor_prefix_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if continuation_text is not None and continuation_text != decoded_prefix:
            raise SystemExit("continuation-text does not equal tokenizer-decoded donor prefix")
        continuation_text = decoded_prefix
    prompt = build_prompt_record(
        raw,
        _template_config(resolved.config),
        processor=processor,
        row_index=0,
        assistant_continuation=(
            AssistantContinuation(text=continuation_text)
            if continuation_text
            else None
        ),
    )
    prompt_token_hash = hashlib.sha256(
        json.dumps(prompt.prompt_token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    boundary_evidence: dict[str, object] = {
        "prompt_token_count": len(prompt.prompt_token_ids),
        "prompt_token_ids_sha256": hashlib.sha256(
            json.dumps(prompt.prompt_token_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "prompt_record_continuation_verified": bool(
            prompt.continuation_text_sha256 is not None
            and prompt.open_assistant_interval_verified is True
        ),
    }
    if args.mode == "fixed-prefix-smoke":
        assert donor_prompt_ids is not None
        assert donor_prefix_ids is not None
        expected_prompt_ids = [*donor_prompt_ids, *donor_prefix_ids]
        equal = expected_prompt_ids == list(prompt.prompt_token_ids)
        boundary_evidence.update(
            {
                "donor_bundle": str(args.donor_bundle.expanduser().resolve()),
                "donor_boundary_role": "generated_prefix_before_continuation",
                "scientific_boundary_role": args.boundary_role,
                "donor_prompt_token_count": len(donor_prompt_ids),
                "donor_prefix_token_count": len(donor_prefix_ids),
                "donor_prefix_token_ids_sha256": hashlib.sha256(
                    json.dumps(donor_prefix_ids, separators=(",", ":")).encode()
                ).hexdigest(),
                "donor_boundary_prompt_plus_prefix_token_ids_equal": equal,
            }
        )
        if not equal:
            raise SystemExit(
                "fixed-prefix prompt does not equal donor prompt tokens plus donor prefix tokens"
            )
        grammar = _donor_boundary_grammar(
            donor_prefix_ids,
            role=args.boundary_role,
        )
        boundary_evidence.update(grammar)
        if not grammar["boundary_grammar_verified"]:
            raise SystemExit(str(grammar["boundary_grammar_error"]))
        _verify_donor_runtime_identity(
            donor=donor,
            active_image_id=image_id,
            active_model_identity=runtime.model_identity,
            active_tokenizer_identity=_tokenizer_identity(qwen),
            active_generation_config_fingerprint=sha256_json(
                resolved.config.generation.model_dump(mode="json")
            ),
        )
        boundary_evidence["donor_bundle_sha256"] = _sha256_file(args.donor_bundle)
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    generation_fingerprint = sha256_json(resolved.config.generation.model_dump(mode="json"))
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    attestation_path = args.sampled_runtime_attestation
    if attestation_path is None:
        attestation_path = Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            "2026-07-13-spatial-scope-history-disentanglement/runtime-attestation-v2/"
            "request-scoped-sampling-three-policy-cuda.json"
        )
    inputs = image_plan.model_inputs_by_row_id[prompt.row_id]
    greedy = DecodeRequest(
        request_id=_request_id(
            image_id=image_id,
            boundary_role=args.boundary_role or "base",
            prompt_token_hash=prompt_token_hash,
            kind="greedy",
            index=0,
        ),
        prompt_token_ids=list(prompt.prompt_token_ids),
        model_inputs=inputs,
        generation_policy=DecodeGenerationPolicy.greedy(max_new_tokens=args.max_new_tokens, repetition_penalty=1.0),
    )
    numeric_image_id = int(image_id)
    seed_values = (
        tuple(args.seeds)
        if args.seeds
        else tuple(
            derive_sampling_seed(
                root_seed=args.seed_root,
                role="image-bootstrap",
                image_id=numeric_image_id,
                cell_or_call_label=(
                    f"sampled-rescue:{i}:temperature-{_temperature_label(args.sampling_temperature)}"
                ),
            )
            for i in range(8)
        )
    ) if include_samples else ()
    if include_samples and len(seed_values) != 8:
        raise SystemExit("runtime-smoke requires exactly eight seeds")
    sampled_policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        temperature=args.sampling_temperature,
        top_p=0.95,
    )
    capability = None
    sampled = []
    if include_samples:
        capability = load_and_rebind_sampled_runtime_attestation_aggregate(
            attestation_path.expanduser().resolve(strict=True),
            decode_generation_policy_fingerprint=sampled_policy.fingerprint,
            backend=backend,
        )
        sampled = [
            DecodeRequest(
                request_id=_request_id(
                    image_id=image_id,
                    boundary_role=args.boundary_role or "base",
                    prompt_token_hash=prompt_token_hash,
                    kind="sample",
                    index=i,
                    seed=int(seed),
                    sampling_temperature=args.sampling_temperature,
                ),
                prompt_token_ids=list(prompt.prompt_token_ids),
                model_inputs=inputs,
                generation_policy=sampled_policy,
                sampling_seed=int(seed),
            )
            for i, seed in enumerate(seed_values)
        ]
    results = [backend.generate_batch([greedy], model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint)[0]]
    for start in range(0, len(sampled), 4):
        results.extend(backend.generate_batch_with_verified_runtime_attestation(
            sampled[start:start + 4],
            model_identity=model_identity,
            tokenizer_identity=tokenizer_identity,
            generation_config_fingerprint=generation_fingerprint,
            verified_runtime_attestation=capability,
        ))
    bundles: list[dict[str, object]] = []
    bundle_root = args.bundle_root
    if bundle_root is not None:
        bundle_root = bundle_root.expanduser().resolve()
    first_row_projection: list[dict[str, object]] = []
    for result in results:
        parsed = parse_compact_object_box_closed(
            result.parser_text,
            row_id=result.request_id,
            row_index=0,
            image_width=raw.image.width,
            image_height=raw.image.height,
        )
        parse_rows = [
            {
                "normalized_category_name": row.get("description", ""),
                "parsed_bbox_xyxy": row.get("bbox", []),
                "generated_row_index": row.get("generated_order", index),
            }
            for index, row in enumerate(parsed.predictions)
        ]
        first_prediction = parse_rows[0] if parse_rows else None
        first_action = _first_action(
            parsed,
            result.parser_text,
            generated_token_ids=result.generated_token_ids,
            stop_reason=result.stop_reason,
        )
        bundle = {
            "schema_version": "sampled_rescue_transition.compact_call_bundle.v1",
            "request_id": result.request_id,
            "stop_reason": result.stop_reason,
            "execution_evidence": {
                "image_id": image_id,
                "example_id": example_id,
                "arm": {"arm_code": "GREEDY_ANCHOR" if result is results[0] else "FULL_BAG_K"},
                "prompt_record": prompt.to_artifact_dict(),
                "boundary_evidence": boundary_evidence,
            },
            "decode_result": result.to_artifact_dict(),
            "parse_score_receipts": parse_rows,
            "parse_result": parsed.to_artifact_dict(),
            "first_action": first_action,
        }
        bundles.append(bundle)
        first_row_projection.append(
            {
                "request_id": result.request_id,
                "rows": [list(span) for span in trajectory_rows(result.generated_token_ids)],
                "first_prediction": first_prediction,
                "first_action": first_action,
                "stop_reason": result.stop_reason,
                "decode": result.to_artifact_dict(),
            }
        )
        if bundle_root is not None:
            directory = bundle_root / result.request_id.replace("/", "_")
            bundle_path = directory / "terminal-output-bundle.json"
            _write_bundle_once(bundle_path, bundle)
    return {
        "schema_version": "sampled_rescue_transition.runtime_smoke.v1",
        "mode": args.mode,
        "image_id": image_id,
        "example_id": example_id,
        "config_path": str(config_path),
        "source_jsonl": str(source_path),
        "prompt_token_hash": prompt_token_hash,
        "repetition_penalty": 1.0,
        "sampling_temperature": float(args.sampling_temperature),
        "sampling_seeds": list(seed_values),
        "seed_root": args.seed_root,
        "continuation_text": continuation_text,
        "prompt_record": prompt.to_artifact_dict(),
        "boundary_evidence": boundary_evidence,
        "bundle_root": str(bundle_root) if bundle_root is not None else None,
        "call_bundles": bundles,
        "first_row_projection": first_row_projection,
    }


def _run_forced_span_runtime(
    args: argparse.Namespace,
    *,
    include_samples: bool,
) -> dict[str, object]:
    """Replay one donor span or one phrase-geometry-composed complete row."""

    composed_mode = args.mode in {"composed-row-greedy", "composed-row-smoke"}
    common_required = (
        args.infer_config,
        args.source_jsonl,
        args.recipient_bundle,
        args.recipient_prefix_token_count,
    )
    if len(args.image_ids or ()) != 1 or any(
        value is None for value in common_required
    ):
        raise SystemExit(
            "row replay modes require one image, runtime paths, a recipient bundle, "
            "and a recipient prefix count"
        )
    if composed_mode:
        if args.description_row_receipt is None or args.geometry_row_receipt is None:
            raise SystemExit(
                "composed-row modes require --description-row-receipt and "
                "--geometry-row-receipt"
            )
        if args.forced_context_policy != "exact":
            raise SystemExit("composed-row modes require --forced-context-policy exact")
        forced_span_role = "complete_row"
    else:
        forced_required = (
            args.forced_donor_bundle,
            args.forced_span_start,
            args.forced_span_end,
            args.forced_span_role,
        )
        if any(value is None for value in forced_required):
            raise SystemExit(
                "forced-span modes require a donor bundle, span bounds, and span role"
            )
        forced_span_role = str(args.forced_span_role)
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
    from src.inference.prompt import AssistantContinuation, build_prompt_record
    from src.inference.parsing import parse_compact_object_box_closed
    from src.inference.runtime import assemble_runtime
    from src.analysis.spatial_scope_history.schedule import derive_sampling_seed

    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    raw_rows = load_raw_examples(source_path)
    requested_image_id = str(args.image_ids[0])
    raw = next(
        (
            row for row in raw_rows
            if row.example_id == requested_image_id
            or str(row.metadata.get("source", {}).get("image_id")) == requested_image_id
        ),
        None,
    )
    if raw is None:
        raise SystemExit(f"image id not found in source JSONL: {requested_image_id}")
    source_meta = raw.metadata.get("source")
    image_id = str(int(source_meta.get("image_id")))
    example_id = raw.example_id
    recipient = _read_json_bundle(args.recipient_bundle)
    composed_row = (
        _compose_complete_row_from_receipts(
            args.description_row_receipt,
            args.geometry_row_receipt,
        )
        if composed_mode
        else None
    )
    donor = (
        None
        if composed_mode
        else _read_json_bundle(args.forced_donor_bundle)
    )
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    generation_fingerprint = sha256_json(resolved.config.generation.model_dump(mode="json"))
    _verify_donor_runtime_identity(
        donor=recipient,
        active_image_id=image_id,
        active_model_identity=model_identity,
        active_tokenizer_identity=tokenizer_identity,
        active_generation_config_fingerprint=generation_fingerprint,
    )
    if composed_mode:
        assert composed_row is not None
        for source_call_bundle in composed_row["source_call_bundles"]:
            _verify_donor_runtime_identity(
                donor=source_call_bundle,
                active_image_id=image_id,
                active_model_identity=model_identity,
                active_tokenizer_identity=tokenizer_identity,
                active_generation_config_fingerprint=generation_fingerprint,
            )
    else:
        assert donor is not None
        _verify_donor_runtime_identity(
            donor=donor,
            active_image_id=image_id,
            active_model_identity=model_identity,
            active_tokenizer_identity=tokenizer_identity,
            active_generation_config_fingerprint=generation_fingerprint,
        )
    recipient_decode = _bundle_decode(recipient)
    recipient_prompt_ids = _int_list(recipient_decode.get("prompt_token_ids"))
    recipient_generated_ids = _int_list(recipient_decode.get("generated_token_ids"))
    recipient_count = int(args.recipient_prefix_token_count)
    if not 0 <= recipient_count <= len(recipient_generated_ids):
        raise SystemExit("recipient-prefix-token-count is outside recipient generated tokens")
    if composed_mode:
        assert composed_row is not None
        span_ids = [int(token) for token in composed_row["token_ids"]]
        start = recipient_count
        end = recipient_count + len(span_ids)
        donor_prompt_ids = list(recipient_prompt_ids)
        donor_generated_ids = [*recipient_generated_ids[:recipient_count], *span_ids]
    else:
        assert donor is not None
        donor_decode = _bundle_decode(donor)
        donor_prompt_ids = _int_list(donor_decode.get("prompt_token_ids"))
        donor_generated_ids = _int_list(donor_decode.get("generated_token_ids"))
        start = int(args.forced_span_start)
        end = int(args.forced_span_end)
        if not 0 <= start < end <= len(donor_generated_ids):
            raise SystemExit("forced span bounds are outside donor generated tokens")
        span_ids = donor_generated_ids[start:end]
    span_grammar = _validate_forced_span_grammar(
        donor_generated_ids,
        start=start,
        end=end,
        role=forced_span_role,
    )
    if not span_grammar["verified"]:
        raise SystemExit(str(span_grammar["error"]))
    recipient_prefix_ids = recipient_generated_ids[:recipient_count]
    context_evidence = _forced_context_evidence(
        recipient_prompt_ids=recipient_prompt_ids,
        recipient_prefix_ids=recipient_prefix_ids,
        donor_prompt_ids=donor_prompt_ids,
        donor_prefix_ids=donor_generated_ids[:start],
        policy=args.forced_context_policy,
    )
    exact_context = bool(context_evidence["exact_context"])
    base_prompt = build_prompt_record(
        raw,
        _template_config(resolved.config),
        processor=qwen.processor,
        row_index=0,
    )
    if list(base_prompt.prompt_token_ids) != recipient_prompt_ids:
        raise SystemExit("recipient original prompt does not match active prompt")
    composed_ids = [*recipient_prefix_ids, *span_ids]
    continuation_text = qwen.tokenizer.decode(
        composed_ids,
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
    resulting_prompt_ids = list(prompt.prompt_token_ids)
    expected_prompt_ids = [*recipient_prompt_ids, *composed_ids]
    if resulting_prompt_ids != expected_prompt_ids:
        raise SystemExit("active prompt does not equal recipient prompt plus composed tokens")
    prompt_hash = hashlib.sha256(
        json.dumps(resulting_prompt_ids, separators=(",", ":")).encode()
    ).hexdigest()
    source_evidence = (
        {
            "span_source": "composed_row_receipts",
            **composed_row["evidence"],
        }
        if composed_mode
        else {
            "span_source": "contiguous_donor_bundle",
            "forced_donor_bundle_sha256": _sha256_file(args.forced_donor_bundle),
        }
    )
    boundary_evidence = {
        "condition_name": args.condition_name,
        "forced_span_role": forced_span_role,
        "forced_context_policy": args.forced_context_policy,
        "exact_context": exact_context,
        **context_evidence,
        **source_evidence,
        "recipient_bundle_sha256": _sha256_file(args.recipient_bundle),
        "recipient_prefix_token_count": recipient_count,
        "forced_span_start": start,
        "forced_span_end": end,
        "forced_span_token_count": len(span_ids),
        "forced_span_token_sha256": hashlib.sha256(
            json.dumps(span_ids, separators=(",", ":")).encode()
        ).hexdigest(),
        "resulting_prompt_token_hash": prompt_hash,
        "resulting_prompt_token_count": len(resulting_prompt_ids),
        "span_grammar": span_grammar,
        "incremental_predecessor": (
            str(args.incremental_predecessor.expanduser().resolve())
            if args.incremental_predecessor is not None
            else None
        ),
        "incremental_predecessor_sha256": (
            _sha256_file(args.incremental_predecessor)
            if args.incremental_predecessor is not None
            else None
        ),
    }
    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    inputs = image_plan.model_inputs_by_row_id[prompt.row_id]
    greedy = DecodeRequest(
        request_id=_request_id(
            image_id=image_id,
            boundary_role=forced_span_role,
            prompt_token_hash=prompt_hash,
            kind="greedy",
            index=0,
            condition_name=args.condition_name,
        ),
        prompt_token_ids=resulting_prompt_ids,
        model_inputs=inputs,
        generation_policy=DecodeGenerationPolicy.greedy(
            max_new_tokens=args.max_new_tokens,
            repetition_penalty=1.0,
        ),
    )
    seeds = (
        tuple(args.seeds)
        if args.seeds
        else tuple(
            derive_sampling_seed(
                root_seed=args.seed_root,
                role="image-bootstrap",
                image_id=int(image_id),
                cell_or_call_label=(
                    f"forced-span:{args.condition_name}:{i}:temperature-"
                    f"{_temperature_label(args.sampling_temperature)}"
                ),
            )
            for i in range(8)
        )
    ) if include_samples else ()
    if include_samples and len(seeds) != 8:
        raise SystemExit("forced-span-smoke requires exactly eight seeds")
    sampled_policy = DecodeGenerationPolicy.sampled(
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=1.0,
        temperature=args.sampling_temperature,
        top_p=0.95,
    )
    sampled: list[DecodeRequest] = []
    capability = None
    if include_samples:
        attestation_path = args.sampled_runtime_attestation or Path(
            "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
            "2026-07-13-spatial-scope-history-disentanglement/runtime-attestation-v2/"
            "request-scoped-sampling-three-policy-cuda.json"
        )
        capability = load_and_rebind_sampled_runtime_attestation_aggregate(
            attestation_path.expanduser().resolve(strict=True),
            decode_generation_policy_fingerprint=sampled_policy.fingerprint,
            backend=backend,
        )
        sampled = [
            DecodeRequest(
                request_id=_request_id(
                    image_id=image_id,
                    boundary_role=forced_span_role,
                    prompt_token_hash=prompt_hash,
                    kind="sample",
                    index=i,
                    seed=int(seed),
                    sampling_temperature=args.sampling_temperature,
                    condition_name=args.condition_name,
                ),
                prompt_token_ids=resulting_prompt_ids,
                model_inputs=inputs,
                generation_policy=sampled_policy,
                sampling_seed=int(seed),
            )
            for i, seed in enumerate(seeds)
        ]
    results = [backend.generate_batch([greedy], model_identity=model_identity, tokenizer_identity=tokenizer_identity, generation_config_fingerprint=generation_fingerprint)[0]]
    for offset in range(0, len(sampled), 4):
        results.extend(
            backend.generate_batch_with_verified_runtime_attestation(
                sampled[offset : offset + 4],
                model_identity=model_identity,
                tokenizer_identity=tokenizer_identity,
                generation_config_fingerprint=generation_fingerprint,
                verified_runtime_attestation=capability,
            )
        )
    bundles: list[dict[str, object]] = []
    first_rows: list[dict[str, object]] = []
    for result in results:
        suffix = _int_list(result.generated_token_ids)
        full_tokens = [*composed_ids, *suffix]
        reconstructed = _reconstruct_intervened_row(
            qwen.tokenizer,
            full_tokens,
            intervention_token_count=len(composed_ids),
            image_width=raw.image.width,
            image_height=raw.image.height,
            row_id=result.request_id,
        )
        free_text = qwen.tokenizer.decode(
            suffix,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        free_parse = parse_compact_object_box_closed(
            free_text,
            row_id=result.request_id,
            row_index=0,
            image_width=raw.image.width,
            image_height=raw.image.height,
        )
        first_free_action = _first_action(
            free_parse,
            free_text,
            generated_token_ids=suffix,
            stop_reason=result.stop_reason,
        )
        first_free_token = _first_free_token_evidence(result, suffix)
        bundle = {
            "schema_version": "sampled_rescue_transition.forced_span_call_bundle.v1",
            "request_id": result.request_id,
            "stop_reason": result.stop_reason,
            "execution_evidence": {
                "image_id": image_id,
                "example_id": example_id,
                "condition_name": args.condition_name,
                "boundary_evidence": boundary_evidence,
            },
            "decode_result": result.to_artifact_dict(),
            "reconstructed_intervened_row": reconstructed,
            "first_free_action": first_free_action,
            "first_free_token": first_free_token,
            "free_parse": free_parse.to_artifact_dict(),
            "raw_forced_span_bundle": {
                "recipient_bundle": str(args.recipient_bundle.expanduser().resolve()),
                "forced_donor_bundle": (
                    str(args.forced_donor_bundle.expanduser().resolve())
                    if args.forced_donor_bundle is not None
                    else None
                ),
                "factor_source_receipts": (
                    composed_row["source_receipts"] if composed_mode else None
                ),
                "recipient_raw_bundle": recipient,
                "forced_donor_raw_bundle": donor,
                "composed_token_ids": composed_ids,
                "suffix_token_ids": suffix,
            },
        }
        bundles.append(bundle)
        first_rows.append(
            {
                "request_id": result.request_id,
                "reconstructed_intervened_row": reconstructed,
                "first_free_action": first_free_action,
                "first_free_token": first_free_token,
            }
        )
        if args.bundle_root is not None:
            _write_bundle_once(
                args.bundle_root.expanduser().resolve()
                / result.request_id.replace("/", "_")
                / "terminal-output-bundle.json",
                bundle,
            )
    return {
        "schema_version": "sampled_rescue_transition.forced_span_runtime.v1",
        "mode": args.mode,
        "image_id": image_id,
        "example_id": example_id,
        "condition_name": args.condition_name,
        "forced_context_policy": args.forced_context_policy,
        "sampling_temperature": float(args.sampling_temperature),
        "sampling_seeds": list(seeds),
        "boundary_evidence": boundary_evidence,
        "first_rows": first_rows,
        "call_bundles": bundles,
    }


def _compose_complete_row_from_receipts(
    description_receipt_path: Path,
    geometry_receipt_path: Path,
) -> dict[str, object]:
    """Compose one canonical row from independently reviewed token factors."""

    def load_source(path: Path) -> dict[str, object]:
        resolved = path.expanduser().resolve(strict=True)
        receipt = _read_json_bundle(resolved)
        first_rows = receipt.get("first_rows")
        call_bundles = receipt.get("call_bundles")
        if not isinstance(first_rows, list) or not first_rows:
            raise SystemExit(f"factor receipt lacks first_rows: {resolved}")
        if not isinstance(call_bundles, list) or not call_bundles:
            raise SystemExit(f"factor receipt lacks call_bundles: {resolved}")
        first_row = first_rows[0]
        if not isinstance(first_row, dict):
            raise SystemExit(f"factor receipt first row is not an object: {resolved}")
        reconstructed = first_row.get("reconstructed_intervened_row")
        if not isinstance(reconstructed, dict) or reconstructed.get("status") != "complete":
            raise SystemExit(f"factor receipt lacks a completed intervened row: {resolved}")
        tokens = _int_list(reconstructed.get("token_ids"))
        factors = _canonical_complete_row_factors(tokens)
        source_call_bundle = call_bundles[0]
        if not isinstance(source_call_bundle, dict):
            raise SystemExit(f"factor receipt call bundle is not an object: {resolved}")
        return {
            "path": str(resolved),
            "sha256": _sha256_file(resolved),
            "condition_name": receipt.get("condition_name"),
            "row_text": reconstructed.get("text"),
            "tokens": tokens,
            "factors": factors,
            "source_call_bundle": source_call_bundle,
        }

    description_source = load_source(description_receipt_path)
    geometry_source = load_source(geometry_receipt_path)
    description_token = int(description_source["factors"]["description_token_id"])
    coordinate_tokens = [
        int(token)
        for token in geometry_source["factors"]["coordinate_token_ids"]
    ]
    token_ids = [
        OBJECT_REF_START,
        description_token,
        OBJECT_REF_END,
        BOX_START,
        *coordinate_tokens,
        BOX_END,
    ]
    _canonical_complete_row_factors(token_ids)
    token_sha256 = hashlib.sha256(
        json.dumps(token_ids, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "token_ids": token_ids,
        "source_call_bundles": [
            description_source["source_call_bundle"],
            geometry_source["source_call_bundle"],
        ],
        "source_receipts": [
            description_source["path"],
            geometry_source["path"],
        ],
        "evidence": {
            "description_row_receipt": description_source["path"],
            "description_row_receipt_sha256": description_source["sha256"],
            "description_source_condition": description_source["condition_name"],
            "description_source_row_text": description_source["row_text"],
            "description_token_id": description_token,
            "geometry_row_receipt": geometry_source["path"],
            "geometry_row_receipt_sha256": geometry_source["sha256"],
            "geometry_source_condition": geometry_source["condition_name"],
            "geometry_source_row_text": geometry_source["row_text"],
            "coordinate_token_ids": coordinate_tokens,
            "composed_row_token_ids": token_ids,
            "composed_row_token_sha256": token_sha256,
        },
    }


def _canonical_complete_row_factors(tokens: Sequence[int]) -> dict[str, object]:
    """Validate the exact one-description/four-coordinate row used by the factorial."""

    values = [int(token) for token in tokens]
    coordinate_vocabulary = set(range(151670, 152670))
    if len(values) != 9:
        raise SystemExit("factor row must contain exactly nine tokens")
    if values[0] != OBJECT_REF_START or values[2] != OBJECT_REF_END:
        raise SystemExit("factor row must contain exactly one description token")
    if values[3] != BOX_START or values[8] != BOX_END:
        raise SystemExit("factor row wrappers are not canonical")
    if values[1] in {
        OBJECT_REF_START,
        OBJECT_REF_END,
        BOX_START,
        BOX_END,
    } or values[1] in coordinate_vocabulary:
        raise SystemExit("factor row description token is not a lexical token")
    coordinate_tokens = values[4:8]
    if len(coordinate_tokens) != 4 or any(
        token not in coordinate_vocabulary for token in coordinate_tokens
    ):
        raise SystemExit("factor row must contain four canonical coordinate tokens")
    grammar = _validate_forced_span_grammar(
        values,
        start=0,
        end=len(values),
        role="complete_row",
    )
    if not grammar["verified"]:
        raise SystemExit(str(grammar["error"]))
    return {
        "description_token_id": values[1],
        "coordinate_token_ids": coordinate_tokens,
    }


def _read_json_bundle(path: Path) -> dict[str, object]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SystemExit(f"bundle must be an object: {path}")
    nested = value.get("payload")
    if isinstance(nested, dict):
        return nested
    return value


def _first_free_token_evidence(result: object, suffix: Sequence[int]) -> dict[str, object]:
    if not suffix:
        return {"token_id": None, "log_probability_float32": None}
    trace = getattr(result, "token_trace", None)
    if not isinstance(trace, Sequence) or not trace:
        raise SystemExit("first free token lacks an executed token trace")
    first = trace[0]
    if isinstance(first, Mapping):
        step_index = first.get("step_index")
        token_id = first.get("token_id")
        is_pad = first.get("is_pad", False)
        value = first.get("log_probability", first.get("logprob"))
    else:
        step_index = getattr(first, "step_index", None)
        token_id = getattr(first, "token_id", None)
        is_pad = getattr(first, "is_pad", False)
        value = getattr(first, "logprob", None)
    if step_index != 0 or int(token_id) != int(suffix[0]) or bool(is_pad):
        raise SystemExit("first free token trace does not match the first suffix token")
    if value is None:
        raise SystemExit("first free token lacks a selected-token logprob")
    try:
        from src.inference.backend import canonical_float32_logprob

        canonical = canonical_float32_logprob(
            value,
            context={"step_index": 0, "token_id": int(suffix[0])},
        )
    except Exception as exc:
        raise SystemExit(f"first free token logprob is not finite float32: {exc}") from exc
    return {
        "token_id": int(suffix[0]),
        "log_probability_float32": canonical,
    }


def _bundle_decode(bundle: Mapping[str, object]) -> dict[str, object]:
    decode = bundle.get("decode_result", bundle)
    if not isinstance(decode, dict):
        raise SystemExit("bundle decode_result must be an object")
    return decode


def _int_list(value: object) -> list[int]:
    if not isinstance(value, list):
        raise SystemExit("bundle token ids must be a list")
    return [int(token) for token in value]


def _validate_forced_span_grammar(
    tokens: Sequence[int],
    *,
    start: int,
    end: int,
    role: str,
) -> dict[str, object]:
    span = [int(token) for token in tokens[start:end]]
    next_box_end = next(
        (index for index in range(start, len(tokens)) if int(tokens[index]) == BOX_END),
        None,
    )
    markers = {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}
    coordinate_vocabulary = set(range(151670, 152670))
    error: str | None = None
    if start < 0 or end > len(tokens) or start >= end:
        error = "forced span bounds are invalid"
    elif not span or span[0] != OBJECT_REF_START:
        error = "forced span must start at OBJECT_REF_START"
    elif OBJECT_REF_START in span[1:]:
        error = "forced span crosses into another object row"
    elif role == "row_opener":
        if span != [OBJECT_REF_START]:
            error = "row_opener must contain exactly OBJECT_REF_START"
    elif role == "first_description_token":
        if len(span) != 2 or span[1] in markers:
            error = "first_description_token must contain exactly one description token"
    elif role == "complete_description":
        if (
            len(span) != 3
            or span[-1] != OBJECT_REF_END
            or any(token in {BOX_START, BOX_END} for token in span)
        ):
            error = "complete_description must contain exactly one description token"
    elif role == "first_coordinate":
        try:
            object_end = span.index(OBJECT_REF_END)
            box_start = span.index(BOX_START, object_end + 1)
        except ValueError:
            error = "first_coordinate requires OBJECT_REF_END followed by BOX_START"
        else:
            coordinate_tokens = span[box_start + 1 :]
            if len(coordinate_tokens) != 1 or any(
                token not in coordinate_vocabulary for token in coordinate_tokens
            ):
                error = "first_coordinate must contain exactly one canonical coordinate token"
    elif role == "complete_row":
        if not span or span[-1] != BOX_END:
            error = "complete_row must end at BOX_END"
        else:
            try:
                object_end = span.index(OBJECT_REF_END)
                box_start = span.index(BOX_START, object_end + 1)
            except ValueError:
                error = "complete_row requires OBJECT_REF_END followed by BOX_START"
            else:
                coordinate_tokens = span[box_start + 1 : -1]
                if len(coordinate_tokens) != 4 or any(
                    token not in coordinate_vocabulary for token in coordinate_tokens
                ):
                    error = "complete_row must contain exactly four canonical coordinate tokens"
    if end > len(tokens) or (
        role != "complete_row"
        and next_box_end is not None
        and end > next_box_end + 1
    ):
        error = "forced span crosses an incompatible row boundary"
    return {
        "verified": error is None,
        "role": role,
        "grammar_phase": (
            "complete_row" if role == "complete_row"
            else ("open_description" if "description" in role else "open_geometry")
        ),
        "error": error,
        "span_start": start,
        "span_end": end,
        "next_box_end": next_box_end,
    }


def _forced_context_evidence(
    *,
    recipient_prompt_ids: Sequence[int],
    recipient_prefix_ids: Sequence[int],
    donor_prompt_ids: Sequence[int],
    donor_prefix_ids: Sequence[int],
    policy: str,
) -> dict[str, object]:
    exact_context = (
        list(recipient_prompt_ids) + list(recipient_prefix_ids)
        == list(donor_prompt_ids) + list(donor_prefix_ids)
    )
    evidence = {
        "exact_context": exact_context,
        "context_mismatch": not exact_context,
        "recipient_context_token_hash": hashlib.sha256(
            json.dumps(
                [*recipient_prompt_ids, *recipient_prefix_ids],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "donor_context_token_hash": hashlib.sha256(
            json.dumps(
                [*donor_prompt_ids, *donor_prefix_ids],
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
        "forced_context_policy": policy,
    }
    if policy == "exact" and not exact_context:
        raise SystemExit("forced exact context mismatch between recipient and donor boundary")
    return evidence


def _reconstruct_intervened_row(
    tokenizer: object,
    tokens: Sequence[int],
    *,
    intervention_token_count: int,
    image_width: int,
    image_height: int,
    row_id: str,
) -> dict[str, object]:
    from src.inference.parsing import parse_compact_object_box_closed

    starts = [index for index, token in enumerate(tokens[:intervention_token_count]) if int(token) == OBJECT_REF_START]
    if not starts:
        return {"status": "no_open_object", "parse": None, "token_ids": []}
    start = starts[-1]
    end = next(
        (index + 1 for index in range(start, len(tokens)) if int(tokens[index]) == BOX_END),
        None,
    )
    row_tokens = list(tokens[start:end]) if end is not None else list(tokens[start:])
    text = tokenizer.decode(row_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    parsed = parse_compact_object_box_closed(
        text,
        row_id=row_id,
        row_index=0,
        image_width=image_width,
        image_height=image_height,
    )
    return {
        "status": "complete" if end is not None else "open",
        "token_start": start,
        "token_end": end,
        "token_ids": row_tokens,
        "text": text,
        "parse": parsed.to_artifact_dict(),
    }


def _first_action(
    parsed: object,
    parser_text: str,
    *,
    generated_token_ids: Sequence[int],
    stop_reason: str | None,
) -> dict[str, object]:
    """Classify the earliest object action without letting later rows hide it."""

    predictions = list(getattr(parsed, "predictions", ()))
    dropped = list(getattr(parsed, "dropped_predictions", ()))
    events: list[tuple[int, str, dict[str, object]]] = []
    token_spans = _object_token_spans(generated_token_ids)
    for row in predictions:
        order = row.get("generated_order")
        if order is not None:
            events.append((int(order), "valid_row", row))
    for row in dropped:
        order = row.get("generated_order")
        if order is not None:
            events.append((int(order), "malformed_row", row))
    ordered_rows = [*predictions, *[row for row in dropped if row.get("generated_order") is not None]]
    earliest_ordered_char = min(
        (int(row["char_start"]) for row in ordered_rows if row.get("char_start") is not None),
        default=None,
    )
    leading_drops = [
        row for row in dropped
        if row.get("generated_order") is None
        and row.get("char_start") is not None
        and (earliest_ordered_char is None or int(row["char_start"]) < earliest_ordered_char)
    ]
    if leading_drops:
        row = min(leading_drops, key=lambda item: int(item["char_start"]))
        return {
            "status": "leading_unmatched",
            "generated_order": None,
            "object_span_id": row.get("object_span_id"),
            "drop_reason": row.get("reason"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
            "token_start": None,
            "token_end": None,
            "token_ids": [],
        }
    if events:
        order, status, row = min(events, key=lambda item: item[0])
        token_start, token_end = token_spans.get(order, (None, None))
        return {
            "status": status,
            "generated_order": order,
            "object_span_id": row.get("object_span_id"),
            "description": row.get("description"),
            "bbox_xyxy": row.get("bbox"),
            "drop_reason": row.get("reason"),
            "char_start": row.get("char_start"),
            "char_end": row.get("char_end"),
            "token_start": token_start,
            "token_end": token_end,
            "token_ids": (
                list(generated_token_ids[token_start:token_end])
                if token_start is not None and token_end is not None
                else []
            ),
        }
    stripped = parser_text.strip()
    if (
        stop_reason == "im_end"
        and not any(int(token) == OBJECT_REF_START for token in generated_token_ids)
    ) or stripped.startswith("<|im_end|>"):
        return {
            "status": "immediate_terminal",
            "generated_order": None,
            "token_start": 0 if generated_token_ids else None,
            "token_end": 1 if generated_token_ids else None,
            "token_ids": list(generated_token_ids[:1]),
        }
    return {"status": "no_object_action", "generated_order": None}


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.expanduser().resolve(strict=True).read_bytes()).hexdigest()


def _request_id(
    *,
    image_id: str,
    boundary_role: str,
    prompt_token_hash: str,
    kind: str,
    index: int,
    seed: int | None = None,
    sampling_temperature: float | None = None,
    condition_name: str | None = None,
) -> str:
    suffix = f"{kind}-index-{index}"
    if condition_name:
        safe_condition = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in str(condition_name)
        )
        suffix = f"condition-{safe_condition}-" + suffix
    if sampling_temperature is not None:
        suffix += f"-temperature-{_temperature_label(sampling_temperature)}"
    if seed is not None:
        suffix += f"-seed-{seed}"
    return (
        f"sampled-rescue:{image_id}:boundary-{boundary_role}:"
        f"prompt-{prompt_token_hash[:12]}:{suffix}"
    )


def _write_bundle_once(path: Path, bundle: Mapping[str, object]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite existing call bundle: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(bundle, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _donor_boundary_grammar(
    prefix_tokens: Sequence[int],
    *,
    role: str,
) -> dict[str, object]:
    """Verify only token grammar; scientific boundary role remains caller asserted."""

    tokens = tuple(int(token) for token in prefix_tokens)
    starts = [index for index, token in enumerate(tokens) if token == OBJECT_REF_START]
    box_ends = [index for index, token in enumerate(tokens) if token == BOX_END]
    complete_row_boundary = bool(starts and box_ends and box_ends[-1] == len(tokens) - 1)
    open_object = bool(starts and (not box_ends or box_ends[-1] < starts[-1]))
    role_requires_complete = role in {"common_pre_row", "rescue_entry", "greedy_terminal"}
    role_requires_open = role == "phase_divergence"
    verified = (
        complete_row_boundary if role_requires_complete
        else (open_object and not complete_row_boundary if role_requires_open else False)
    )
    error = None
    if role_requires_complete and not complete_row_boundary:
        error = "donor prefix must end immediately after a complete BOX_END row boundary"
    elif role_requires_open and not (open_object and not complete_row_boundary):
        error = "phase_divergence donor prefix must end inside an open object span before BOX_END"
    return {
        "boundary_grammar_verified": verified,
        "boundary_grammar_role": role,
        "boundary_grammar_state": (
            "complete_row_boundary" if complete_row_boundary
            else ("open_object_span" if open_object else "outside_object_span")
        ),
        "boundary_grammar_error": error,
        "boundary_role_auto_verified": False,
    }


def _verify_donor_runtime_identity(
    *,
    donor: dict[str, object],
    active_image_id: str,
    active_model_identity: object,
    active_tokenizer_identity: object,
    active_generation_config_fingerprint: str,
) -> None:
    from src.inference.backend import _normalized_attested_model_identity_for_runtime_comparison

    evidence = donor.get("execution_evidence")
    if not isinstance(evidence, dict) or str(evidence.get("image_id")) != str(active_image_id):
        raise SystemExit("donor execution_evidence.image_id does not match active image_id")
    decode = donor.get("decode_result", donor)
    if not isinstance(decode, dict):
        raise SystemExit("donor decode result must be an object")
    mismatches: list[str] = []
    donor_model_identity = decode.get("model_identity")
    if donor_model_identity is None:
        mismatches.append("model_identity_missing")
    elif _normalized_attested_model_identity_for_runtime_comparison(
        donor_model_identity
    ) != _normalized_attested_model_identity_for_runtime_comparison(
        active_model_identity
    ):
        mismatches.append("model_identity")
    donor_tokenizer_identity = decode.get("tokenizer_identity")
    if donor_tokenizer_identity is None:
        mismatches.append("tokenizer_identity_missing")
    elif donor_tokenizer_identity != active_tokenizer_identity:
        mismatches.append("tokenizer_identity")
    donor_generation = decode.get("generation_config_fingerprint")
    if donor_generation is None:
        mismatches.append("generation_config_fingerprint_missing")
    elif str(donor_generation) != str(active_generation_config_fingerprint):
        mismatches.append("generation_config_fingerprint")
    if mismatches:
        raise SystemExit("donor runtime identity mismatch: " + ", ".join(mismatches))


def _object_token_spans(tokens: Sequence[int]) -> dict[int, tuple[int, int]]:
    """Map generated object order to token bounds using the canonical wrappers."""

    starts = [
        index for index, token in enumerate(tokens) if int(token) == OBJECT_REF_START
    ]
    spans: dict[int, tuple[int, int]] = {}
    for order, start in enumerate(starts):
        next_start = starts[order + 1] if order + 1 < len(starts) else len(tokens)
        end = next_start
        for index in range(start + 1, next_start):
            if int(tokens[index]) == BOX_END:
                end = index + 1
                break
        spans[order] = (start, end)
    return spans


def _temporary_cwd(path: Path):
    class _Cwd:
        def __enter__(self):
            self.previous = Path.cwd()
            os.chdir(path)
            return self
        def __exit__(self, *exc):
            os.chdir(self.previous)
    return _Cwd()


if __name__ == "__main__":
    raise SystemExit(main())
