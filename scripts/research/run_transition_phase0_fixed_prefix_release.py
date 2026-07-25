#!/usr/bin/env python3
"""Greedily release one row after a frozen heldout Source prefix intervention.

The runner consumes the exact literal-token manifest produced by
``materialize_transition_phase0_fixed_prefix_panel.py``.  It loads one of the
two checkpoint configs declared by that manifest, forces either the canonical
row opener or an owner-bearing candidate's one-token complete description,
then delegates row completion, parsing, and owner matching to the existing
local-branch research helpers.

This is a diagnostic fixed-prefix intervention.  It is not a production
decoding policy and its released row is not a free-rollout final-set result.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.materialize_transition_phase0_fixed_prefix_panel import (  # noqa: E402
    OBJECT_REF_START,
    PANEL_SCHEMA_VERSION,
    UNIT_ID,
    sha256_file,
    token_ids_sha256,
    validate_frozen_panel_manifest,
    verify_file_identity,
)
from scripts.research.run_complete_candidate_row_scoring import (  # noqa: E402
    _runtime_model_dtype_summary,
)
from scripts.research.run_local_branch_causal_value import (  # noqa: E402
    _annotate_owner_matches,
    _generate_after_forced_partial_row,
    _single_native_inputs,
    build_positive_entity_ledger,
)
from scripts.research.run_next_row_likelihood_change import (  # noqa: E402
    _canonical_row_phases,
)


RECEIPT_SCHEMA_VERSION = "transition_phase0_fixed_prefix_release.receipt.v1"
FORCE_MODES = ("opener", "complete-description")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _literal_tokens(reference: Mapping[str, Any], *, label: str) -> list[int]:
    values = reference.get("token_ids")
    if not isinstance(values, list) or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in values
    ):
        raise ValueError(f"{label} must contain literal non-negative token IDs")
    tokens = [int(value) for value in values]
    expected = reference.get("token_ids_sha256")
    observed = token_ids_sha256(tokens)
    if not isinstance(expected, str) or observed != expected:
        raise ValueError(f"{label} token hash mismatch")
    return tokens


def build_release_plan(
    manifest: Mapping[str, Any],
    *,
    image_id: str,
    force_mode: str,
    candidate_id: str | None = None,
    boundary_id: str | None = None,
) -> dict[str, Any]:
    if force_mode not in FORCE_MODES:
        raise ValueError(f"force_mode must be one of {FORCE_MODES}")
    images = manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("manifest images must be a list")
    matching_images = [
        image
        for image in images
        if isinstance(image, Mapping) and str(image.get("image_id")) == str(image_id)
    ]
    if len(matching_images) != 1:
        raise ValueError(f"expected exactly one manifest image_id={image_id!r}")
    image = matching_images[0]
    boundaries = image.get("boundaries")
    if not isinstance(boundaries, list):
        raise ValueError("selected image boundaries must be a list")
    matching_boundaries = [
        boundary
        for boundary in boundaries
        if isinstance(boundary, Mapping)
        and (boundary_id is None or str(boundary.get("boundary_id")) == boundary_id)
    ]
    if len(matching_boundaries) != 1:
        raise ValueError("expected exactly one selected boundary")
    boundary = matching_boundaries[0]
    if boundary.get("prefix_mode", "base_prompt_plus_generated") != "base_prompt_plus_generated":
        raise ValueError("forced release requires a generated-suffix prefix")
    prefix = _literal_tokens(boundary.get("prefix", {}), label="selected prefix")
    if not prefix or prefix[-1] != 151649:
        raise ValueError("selected prefix must end at a complete box boundary")

    candidates = boundary.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("selected boundary candidates must be a list")
    selected_candidate: Mapping[str, Any] | None = None
    if candidate_id is not None:
        candidate_matches = [
            candidate
            for candidate in candidates
            if isinstance(candidate, Mapping)
            and str(candidate.get("candidate_id")) == str(candidate_id)
        ]
        if len(candidate_matches) != 1:
            raise ValueError(f"expected exactly one candidate_id={candidate_id!r}")
        selected_candidate = candidate_matches[0]

    if force_mode == "opener":
        forced_tokens = [OBJECT_REF_START]
    else:
        if selected_candidate is None:
            raise ValueError("complete-description force requires --candidate-id")
        if selected_candidate.get("owner") is None:
            raise ValueError("complete-description force requires an owner-bearing candidate")
        row_tokens = _literal_tokens(
            selected_candidate.get("row", selected_candidate),
            label=f"candidate {candidate_id}",
        )
        phases = _canonical_row_phases(row_tokens)
        if len(phases["description"]) != 1:
            raise ValueError(
                "complete-description force currently requires exactly one description token"
            )
        object_end_index = phases["description"][0] + 1
        forced_tokens = row_tokens[: object_end_index + 1]
        if forced_tokens[0] != OBJECT_REF_START or forced_tokens[-1] != 151647:
            raise ValueError("candidate complete-description span is not canonical")

    covered_owner_ids = boundary.get("covered_owner_ids")
    if not isinstance(covered_owner_ids, list) or any(
        not isinstance(value, (str, int)) for value in covered_owner_ids
    ):
        raise ValueError("selected boundary lacks covered_owner_ids")
    return {
        "image_id": str(image["image_id"]),
        "row_id": str(image.get("row_id", image["image_id"])),
        "case_id": str(image.get("case_id", boundary.get("boundary_id", ""))),
        "case_role": image.get("case_role", boundary.get("case_role")),
        "boundary_id": str(boundary.get("boundary_id")),
        "source_completed_row_count": int(boundary.get("source_completed_row_count", 0)),
        "prefix_token_ids": prefix,
        "prefix_token_ids_sha256": token_ids_sha256(prefix),
        "covered_owner_ids": sorted({str(value) for value in covered_owner_ids}),
        "force_mode": force_mode,
        "forced_row_prefix_token_ids": forced_tokens,
        "forced_row_prefix_token_ids_sha256": token_ids_sha256(forced_tokens),
        "candidate": None if selected_candidate is None else dict(selected_candidate),
        "expected_owner_id": (
            None if selected_candidate is None or selected_candidate.get("owner") is None
            else str(selected_candidate["owner"])
        ),
        "expected_category": (
            None if selected_candidate is None else selected_candidate.get("category")
        ),
    }


def validate_declared_runtime(
    manifest: Mapping[str, Any],
    *,
    checkpoint_role: str,
    infer_config: Path,
    verify_files: bool = True,
) -> dict[str, Any]:
    panel = manifest.get("panel")
    if not isinstance(panel, Mapping) or panel.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValueError(f"manifest lacks {PANEL_SCHEMA_VERSION}")
    configs = panel.get("checkpoint_configs")
    inputs = panel.get("inputs")
    if not isinstance(configs, Mapping) or not isinstance(inputs, Mapping):
        raise ValueError("manifest lacks checkpoint configs or frozen inputs")
    reference = configs.get(checkpoint_role)
    if not isinstance(reference, Mapping):
        raise ValueError(f"undeclared checkpoint role: {checkpoint_role}")
    requested = infer_config.expanduser().resolve(strict=True)
    declared = Path(str(reference.get("path", ""))).expanduser().resolve(strict=True)
    if requested != declared:
        raise ValueError(
            f"infer config path differs from declared {checkpoint_role} config: {requested}"
        )
    if verify_files:
        verify_file_identity(reference, label=f"checkpoint_config:{checkpoint_role}")
    source_reference = inputs.get("heldout_source_jsonl")
    if not isinstance(source_reference, Mapping):
        raise ValueError("manifest lacks heldout_source_jsonl identity")
    source_path = Path(str(source_reference.get("path", ""))).expanduser().resolve(strict=True)
    if verify_files:
        verify_file_identity(source_reference, label="heldout_source_jsonl")
    return {
        "checkpoint_role": checkpoint_role,
        "infer_config": {"path": str(declared), "sha256": str(reference.get("sha256"))},
        "source_jsonl": {
            "path": str(source_path),
            "sha256": str(source_reference.get("sha256")),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint-role", choices=("source", "transition-step36"), required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--boundary-id")
    parser.add_argument("--force-mode", choices=FORCE_MODES, default="opener")
    parser.add_argument("--candidate-id")
    parser.add_argument("--runtime-dtype", choices=("config", "fp32"), default="fp32")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--malformed-limit", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.max_new_tokens) <= 0 or int(args.malformed_limit) <= 0:
        raise ValueError("generation and malformed limits must be positive")
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = _read_json(manifest_path)
    normalized = validate_frozen_panel_manifest(manifest, verify_files=True)
    release_plan = build_release_plan(
        normalized,
        image_id=str(args.image_id),
        boundary_id=args.boundary_id,
        force_mode=str(args.force_mode),
        candidate_id=args.candidate_id,
    )
    declared_runtime = validate_declared_runtime(
        manifest,
        checkpoint_role=str(args.checkpoint_role),
        infer_config=args.infer_config,
        verify_files=True,
    )

    from src.config.fingerprint import sha256_json as config_sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
    from src.inference.hf_backend import HFBackendSession
    from src.inference.image_plan import plan_image_batch, verify_processor_model_vision_parity
    from src.inference.pipeline import _processor_config, _template_config
    from src.inference.prompt import build_prompt_record
    from src.inference.runtime import assemble_frontend
    from scripts.research.run_native_sibling_branch_replay import _attention_implementation

    config_path = Path(declared_runtime["infer_config"]["path"])
    source_jsonl = Path(declared_runtime["source_jsonl"]["path"])
    resolved = load_infer_config(config_path)
    config = resolved.config
    if args.runtime_dtype == "fp32":
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"dtype": "fp32"})}
        )
    if config.backend.type != "hf":
        raise ValueError("fixed-prefix forced release requires backend.type: hf")
    configured_source = Path(config.data.input_jsonl).expanduser().resolve(strict=True)
    if configured_source != source_jsonl:
        raise ValueError("resolved infer config input_jsonl differs from frozen heldout source")
    if float(config.generation.repetition_penalty) != 1.0:
        raise ValueError("fixed-prefix forced release requires repetition_penalty=1.0")

    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    raw_rows = load_raw_examples(source_jsonl)
    raw = next(
        (
            row
            for row in raw_rows
            if str(row.metadata.get("source", {}).get("image_id"))
            == release_plan["image_id"]
        ),
        None,
    )
    if raw is None:
        raise ValueError(f"image {release_plan['image_id']} is absent from source JSONL")
    template = _template_config(config)

    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF launch opened an unexpected backend session")
        model = opened._model  # noqa: SLF001
        tokenizer = opened._tokenizer  # noqa: SLF001
        model.eval()
        parity = verify_processor_model_vision_parity(
            processor_identity=frontend.qwen.processor_identity,
            model_config=model.config,
        )
        image_plan = plan_image_batch(
            [raw],
            components=frontend.qwen,
            processor_config=_processor_config(config),
            row_indices=[0],
        ).rows[0]
        prompt_record = build_prompt_record(
            raw,
            template,
            processor=frontend.qwen.processor,
            row_index=0,
            merged_visual_tokens=image_plan.merged_visual_tokens,
        )
        request = DecodeRequest(
            request_id=(
                f"transition-phase0-release:{args.checkpoint_role}:"
                f"{release_plan['case_id']}:{release_plan['force_mode']}"
            ),
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
            expected_image_grid_thw=tuple(image_plan.expected_image_grid_thw),
            logical_transform_id=image_plan.logical_transform_id,
            generation_policy=GenerationPolicy(
                max_new_tokens=1,
                repetition_penalty=1.0,
                temperature=0.0,
                top_p=1.0,
                include_raw_model_logprob=True,
            ),
        )
        native_inputs, executed_prompt_ids, observed_grids, executed_media_sha256 = (
            opened._materialize_native_inputs((request,))  # noqa: SLF001
        )
        base_prompt = [int(value) for value in prompt_record.prompt_token_ids]
        if tuple(executed_prompt_ids[0]) != tuple(base_prompt):
            raise RuntimeError("materialized prompt differs from rebuilt heldout prompt")
        row = _generate_after_forced_partial_row(
            session=opened,
            native_inputs=_single_native_inputs(native_inputs),
            parent_prefix_token_ids=release_plan["prefix_token_ids"],
            forced_row_prefix_token_ids=release_plan["forced_row_prefix_token_ids"],
            tokenizer=tokenizer,
            image_width=int(image_plan.decoded_width),
            image_height=int(image_plan.decoded_height),
            repetition_penalty=1.0,
            max_new_tokens=int(args.max_new_tokens),
            malformed_limit=int(args.malformed_limit),
            row_index=int(release_plan["source_completed_row_count"]),
        )
        ledger = build_positive_entity_ledger(raw)
        _annotate_owner_matches(
            row,
            entity_ledger=ledger,
            image_width=int(image_plan.decoded_width),
            image_height=int(image_plan.decoded_height),
            covered_entity_ids=release_plan["covered_owner_ids"],
        )
        expected_owner_id = release_plan["expected_owner_id"]
        row["intended_owner_realized"] = (
            None
            if expected_owner_id is None
            else expected_owner_id in set(row.get("strict_matched_owner_ids", []))
        )
        actual_prefix = [*base_prompt, *release_plan["prefix_token_ids"]]
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "unit_id": UNIT_ID,
            "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            "checkpoint": declared_runtime,
            "release_plan": release_plan,
            "released_row": row,
            "runtime": {
                "runtime_dtype_mode": str(args.runtime_dtype),
                "model_dtype": _runtime_model_dtype_summary(model),
                "physical_batch_size": 1,
                "max_new_tokens": int(args.max_new_tokens),
                "malformed_limit": int(args.malformed_limit),
                "repetition_penalty": 1.0,
                "authored_config_sha256": sha256_file(config_path),
                "resolved_config_fingerprint": resolved.fingerprint,
                "effective_config_sha256": config_sha256_json(
                    config.model_dump(mode="json")
                ),
                "attention_implementation": _attention_implementation(
                    model, config.backend.hf.attn_implementation
                ),
                "processor_model_vision_parity": parity,
                "backend_session": opened.receipt.to_artifact_dict(),
                "base_prompt_token_ids_sha256": token_ids_sha256(base_prompt),
                "actual_prefix_prompt_plus_generated_sha256": token_ids_sha256(
                    actual_prefix
                ),
                "source_image": {
                    "path": image_plan.image_path,
                    "sha256": image_plan.image_content_sha256,
                    "width": int(image_plan.decoded_width),
                    "height": int(image_plan.decoded_height),
                    "executed_media_sha256": executed_media_sha256[0],
                    "observed_image_grid_thw": (
                        None if observed_grids[0] is None else list(observed_grids[0])
                    ),
                },
            },
            "claim_boundary": (
                "one fixed-prefix forced row is a diagnostic proxy and not a free-rollout "
                "final-set outcome"
            ),
        }
    return receipt


def main() -> None:
    args = build_parser().parse_args()
    output = args.output.expanduser().resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite immutable release receipt: {output}")
    receipt = run(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "case_id": receipt["release_plan"]["case_id"],
                "checkpoint_role": receipt["checkpoint"]["checkpoint_role"],
                "force_mode": receipt["release_plan"]["force_mode"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
