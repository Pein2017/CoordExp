#!/usr/bin/env python3
"""Score exact candidate rows at recorded prefix boundaries.

This is a small, experiment-local scorer for the 2026-07-19 candidate-row
research unit.  A manifest records exact prefix and candidate row token IDs,
or compact selectors into a Stage-2 artifact.  The scorer uses the native
Hugging Face inference path, raw language-model-head logits, and float32
log-softmax accumulation.  It does not apply repetition penalties and it
never turns candidate row scores into a shared probability distribution.

The terminal (stop) score is measured separately at the boundary.  A row
score is the teacher-forced score of the supplied complete row, with phase
summaries for the description, four coordinates, and closure.  Token ranks
are reported as ``1 + count(logit > selected_logit)`` in the float32 logits.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import sys
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_next_row_likelihood_change as _legacy_scorer  # noqa: E402
from scripts.research.run_next_row_likelihood_change import (  # noqa: E402
    OBJECT_REF_START,
    _canonical_row_phases,
    _forward_logits,
    _runtime_model_dtype_summary,
    score_token_logits,
    sha256_file,
    sha256_json,
    terminal_boundary_score,
)

# Re-export the wrapper constants for the experiment-local test helper and for
# manifests that want to construct canonical rows without importing the older
# scorer directly.
BOX_START = _legacy_scorer.BOX_START
BOX_END = _legacy_scorer.BOX_END
COORDINATE_TOKEN_START = _legacy_scorer.COORDINATE_TOKEN_START
COORDINATE_TOKEN_END_EXCLUSIVE = _legacy_scorer.COORDINATE_TOKEN_END_EXCLUSIVE
OBJECT_REF_END = _legacy_scorer.OBJECT_REF_END


MANIFEST_SCHEMA_VERSION = "complete_candidate_row_scoring.manifest.v1"
RECEIPT_SCHEMA_VERSION = "complete_candidate_row_scoring.receipt.v1"


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _resolve_path(raw: str | Path, *, relative_to: Path) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = relative_to / path
    return path.resolve(strict=True)


def _token_hash(tokens: Sequence[int]) -> str:
    return sha256_json([int(value) for value in tokens])


def _declared_hash(reference: Mapping[str, Any], context: str) -> str:
    value = reference.get("token_ids_sha256", reference.get("sha256"))
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires token_ids_sha256 (or sha256)")
    return value


def _stage2_case(document: Mapping[str, Any], case_id: str, context: str) -> Mapping[str, Any]:
    cases = document.get("cases")
    if not isinstance(cases, list):
        raise ValueError(f"{context} source artifact lacks cases list")
    matches = [case for case in cases if isinstance(case, Mapping) and str(case.get("case_id")) == case_id]
    if len(matches) != 1:
        raise ValueError(f"{context} expected one case_id={case_id!r}, found {len(matches)}")
    return matches[0]


def _stage2_run(arm: Mapping[str, Any], selector: Mapping[str, Any], context: str) -> Mapping[str, Any]:
    runs = arm.get("runs")
    if not isinstance(runs, list):
        raise ValueError(f"{context} arm lacks runs list")
    wanted_comparison = selector.get("comparison_id")
    wanted_mode = selector.get("mode")
    wanted_seed = selector.get("seed", "__unspecified__")
    matches = []
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        if wanted_comparison is not None and str(run.get("comparison_id")) != str(wanted_comparison):
            continue
        if wanted_mode is not None and str(run.get("mode")) != str(wanted_mode):
            continue
        if wanted_seed != "__unspecified__" and run.get("seed") != wanted_seed:
            continue
        matches.append(run)
    if len(matches) != 1:
        raise ValueError(f"{context} expected one Stage-2 run, found {len(matches)}")
    return matches[0]


def _resolve_stage2_selector(
    selector: Mapping[str, Any], *, manifest_path: Path, context: str
) -> tuple[list[int], dict[str, Any]]:
    artifact_raw = selector.get("source_artifact", selector.get("artifact"))
    if not isinstance(artifact_raw, str) or not artifact_raw:
        raise ValueError(f"{context} selector requires source_artifact")
    artifact = _resolve_path(artifact_raw, relative_to=manifest_path.parent)
    document = _read_json(artifact)
    token_field = str(selector.get("token_field", ""))
    case_id = selector.get("case_id")
    case: Mapping[str, Any] | None = None
    if case_id is not None:
        case = _stage2_case(document, str(case_id), context)

    value: Any
    if token_field in {"base_prompt.prompt_token_ids", "base_prompt_token_ids"}:
        base = document.get("base_prompt")
        if not isinstance(base, Mapping) or not isinstance(base.get("prompt_token_ids"), list):
            raise ValueError(f"{context} source lacks base_prompt.prompt_token_ids")
        value = base["prompt_token_ids"]
    else:
        if case is None:
            raise ValueError(f"{context} selector requires case_id for {token_field}")
        arm_name = selector.get("arm_name")
        arms = case.get("arms")
        if not isinstance(arms, Mapping) or not isinstance(arm_name, str) or arm_name not in arms:
            raise ValueError(f"{context} selector requires an existing arm_name")
        arm = arms[arm_name]
        if not isinstance(arm, Mapping):
            raise ValueError(f"{context} selected arm is not an object")
        if token_field == "arm_prefix_token_ids":
            value = arm.get("prefix_token_ids")
        elif token_field == "arm_row_token_ids":
            value = arm.get("row_token_ids")
        elif token_field == "entity_row_token_ids":
            entity_id = selector.get("entity_id")
            ledger = case.get("entity_ledger")
            if not isinstance(ledger, list) or entity_id is None:
                raise ValueError(f"{context} entity_row_token_ids requires entity_id and ledger")
            entities = [item for item in ledger if isinstance(item, Mapping) and str(item.get("entity_id")) == str(entity_id)]
            if len(entities) != 1:
                raise ValueError(f"{context} expected one entity_id={entity_id!r}")
            value = entities[0].get("row_token_ids")
        else:
            run = _stage2_run(arm, selector, context)
            row_index = selector.get("row_index")
            rows = run.get("rows")
            if not isinstance(row_index, int) or not isinstance(rows, list) or not 0 <= row_index < len(rows):
                raise ValueError(f"{context} row_index does not select a Stage-2 row")
            row = rows[row_index]
            if not isinstance(row, Mapping):
                raise ValueError(f"{context} selected row is not an object")
            if token_field in {"row_prefix_token_ids", "row_prefix"}:
                value = row.get("prefix_token_ids")
            elif token_field in {"row_generated_token_ids", "row_raw_generated_token_ids", "row_generated"}:
                value = row.get("raw_generated_token_ids")
            elif token_field == "final_prefix_token_ids":
                value = run.get("final_prefix_token_ids")
            elif token_field == "initial_prefix_token_ids":
                value = run.get("initial_prefix_token_ids")
            else:
                raise ValueError(f"{context} unsupported Stage-2 token_field={token_field!r}")
    if not isinstance(value, list) or any(not isinstance(item, int) for item in value):
        raise ValueError(f"{context} selector resolved to non-integer token IDs")
    tokens = [int(item) for item in value]
    expected = _declared_hash(selector, context)
    actual = _token_hash(tokens)
    if expected != actual:
        raise ValueError(f"{context} selector token hash mismatch")
    return tokens, {
        "kind": "stage2_selector",
        "source_artifact": str(artifact),
        "source_artifact_sha256": sha256_file(artifact),
        "token_field": token_field,
        "case_id": case_id,
        "arm_name": selector.get("arm_name"),
        "comparison_id": selector.get("comparison_id"),
        "mode": selector.get("mode"),
        "seed": selector.get("seed"),
        "row_index": selector.get("row_index"),
        "entity_id": selector.get("entity_id"),
        "token_ids_sha256": actual,
        "token_count": len(tokens),
    }


def resolve_token_reference(
    reference: Mapping[str, Any], *, manifest_path: Path, context: str
) -> tuple[list[int], dict[str, Any]]:
    """Resolve literal or compact Stage-2 token references and prove the hash."""

    if not isinstance(reference, Mapping):
        raise ValueError(f"{context} must be an object")
    if isinstance(reference.get("token_ids"), list):
        tokens = reference["token_ids"]
        if any(not isinstance(item, int) for item in tokens):
            raise ValueError(f"{context} literal token_ids must be integers")
        tokens = [int(item) for item in tokens]
        expected = _declared_hash(reference, context)
        actual = _token_hash(tokens)
        if expected != actual:
            raise ValueError(f"{context} literal token hash mismatch")
        return tokens, {"kind": "literal", "token_ids_sha256": actual, "token_count": len(tokens)}
    selector = reference.get("selector")
    if isinstance(selector, Mapping):
        return _resolve_stage2_selector({**dict(selector), **{k: v for k, v in reference.items() if k in {"sha256", "token_ids_sha256"}}}, manifest_path=manifest_path, context=context)
    raise ValueError(f"{context} requires token_ids or selector")


def validate_manifest(manifest: Mapping[str, Any], *, manifest_path: Path) -> dict[str, Any]:
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"expected {MANIFEST_SCHEMA_VERSION}")
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise ValueError("manifest requires non-empty images list")
    normalized_images: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    for image_index, image_item in enumerate(images):
        if not isinstance(image_item, Mapping):
            raise ValueError(f"images[{image_index}] must be an object")
        image = dict(image_item)
        image_id = str(image.get("image_id", ""))
        if not image_id or image_id in image_ids:
            raise ValueError(f"images[{image_index}] requires a unique image_id")
        image_ids.add(image_id)
        boundaries = image.get("boundaries")
        if not isinstance(boundaries, list) or not boundaries:
            raise ValueError(f"{image_id} requires non-empty boundaries")
        base_info: dict[str, Any] | None = None
        base_tokens: list[int] | None = None
        if image.get("base_prompt") is not None:
            base_tokens, base_info = resolve_token_reference(image["base_prompt"], manifest_path=manifest_path, context=f"{image_id}.base_prompt")
        normalized_boundaries: list[dict[str, Any]] = []
        boundary_ids: set[str] = set()
        for boundary_index, boundary_item in enumerate(boundaries):
            if not isinstance(boundary_item, Mapping):
                raise ValueError(f"{image_id}.boundaries[{boundary_index}] must be an object")
            boundary = dict(boundary_item)
            boundary_id = str(boundary.get("boundary_id", f"boundary-{boundary_index}"))
            if boundary_id in boundary_ids:
                raise ValueError(f"{image_id} has duplicate boundary_id={boundary_id}")
            boundary_ids.add(boundary_id)
            prefix_tokens, prefix_info = resolve_token_reference(boundary.get("prefix", {}), manifest_path=manifest_path, context=f"{image_id}.{boundary_id}.prefix")
            candidates = boundary.get("candidates")
            if not isinstance(candidates, list) or not candidates:
                raise ValueError(f"{image_id}.{boundary_id} requires non-empty candidates")
            normalized_candidates: list[dict[str, Any]] = []
            candidate_ids: set[str] = set()
            for candidate_index, candidate_item in enumerate(candidates):
                if not isinstance(candidate_item, Mapping):
                    raise ValueError(f"{image_id}.{boundary_id}.candidates[{candidate_index}] must be an object")
                candidate = dict(candidate_item)
                candidate_id = str(candidate.get("candidate_id", f"candidate-{candidate_index}"))
                if candidate_id in candidate_ids:
                    raise ValueError(f"{image_id}.{boundary_id} duplicate candidate_id={candidate_id}")
                candidate_ids.add(candidate_id)
                row_tokens, row_info = resolve_token_reference(candidate.get("row", candidate), manifest_path=manifest_path, context=f"{image_id}.{boundary_id}.{candidate_id}.row")
                try:
                    _canonical_row_phases(row_tokens)
                except ValueError as exc:
                    raise ValueError(f"{image_id}.{boundary_id}.{candidate_id} row is not canonical") from exc
                normalized_candidates.append({**candidate, "candidate_id": candidate_id, "row_token_ids": row_tokens, "row_reference": row_info})
            normalized_boundaries.append({
                **boundary,
                "boundary_id": boundary_id,
                "prefix_token_ids": prefix_tokens,
                "prefix_reference": prefix_info,
                "prefix_mode": str(boundary.get("prefix_mode", "base_prompt_plus_generated")),
                "candidates": normalized_candidates,
            })
        normalized_images.append({**image, "image_id": image_id, "base_prompt_token_ids": base_tokens, "base_prompt_reference": base_info, "boundaries": normalized_boundaries})
    return {**dict(manifest), "images": normalized_images}


def select_manifest_images(
    manifest: Mapping[str, Any], requested_image_ids: Sequence[str] | None = None
) -> list[Mapping[str, Any]]:
    """Select manifest images for one bounded worker process.

    The manifest order is retained.  An explicitly requested but absent image
    fails rather than silently producing a partial receipt.
    """

    images = manifest.get("images")
    if not isinstance(images, list):
        raise ValueError("manifest images must be a list")
    requested = [str(value) for value in (requested_image_ids or [])]
    available = {str(item.get("image_id")): item for item in images if isinstance(item, Mapping)}
    missing = [image_id for image_id in requested if image_id not in available]
    if missing:
        raise ValueError(f"requested image_id(s) absent from manifest: {', '.join(missing)}")
    if not requested:
        return list(images)
    wanted = set(requested)
    return [item for item in images if str(item.get("image_id")) in wanted]


def selected_token_ranks(logits: torch.Tensor, *, boundary_length: int, row_tokens: Sequence[int]) -> dict[str, Any]:
    """Return exact selected-token ranks from float32 logits for every row phase."""

    if logits.ndim != 2 or int(logits.shape[0]) < int(boundary_length) + len(row_tokens) - 1:
        raise ValueError("logits do not cover row-token rank positions")
    row = [int(item) for item in row_tokens]
    phases = _canonical_row_phases(row)
    source = logits.to(dtype=torch.float32)
    ranks_by_position: list[int] = []
    for index, token in enumerate(row):
        values = source[int(boundary_length) + index - 1]
        selected = values[int(token)]
        ranks_by_position.append(int((values > selected).sum().item()) + 1)
    return {
        "row_entry": [ranks_by_position[index] for index in phases["row_entry"]],
        "description": [ranks_by_position[index] for index in phases["description"]],
        "geometry": [ranks_by_position[index] for index in phases["geometry"]],
        "x1": [ranks_by_position[index] for index in phases["x1"]],
        "y1": [ranks_by_position[index] for index in phases["y1"]],
        "x2": [ranks_by_position[index] for index in phases["x2"]],
        "y2": [ranks_by_position[index] for index in phases["y2"]],
        "closure": [ranks_by_position[index] for index in phases["closure"]],
        "full_row": ranks_by_position,
    }


def _score_candidate(logits: torch.Tensor, *, boundary_length: int, row_tokens: Sequence[int], metadata: Mapping[str, Any]) -> dict[str, Any]:
    score = score_token_logits(logits, boundary_length=boundary_length, row_tokens=row_tokens)
    score["selected_token_ranks"] = selected_token_ranks(logits, boundary_length=boundary_length, row_tokens=row_tokens)
    result = {
        "candidate_id": str(metadata["candidate_id"]),
        "owner": metadata.get("owner"),
        "category": metadata.get("category"),
        "role": metadata.get("role"),
        "covered": metadata.get("covered"),
        "row_token_ids": [int(item) for item in row_tokens],
        **score,
    }
    for key in ("description", "geometry", "source", "truth_status", "notes"):
        if key in metadata:
            result[key] = metadata[key]
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--source-jsonl", type=Path, required=True)
    parser.add_argument("--runtime-dtype", choices=("config", "fp32"), default="fp32")
    parser.add_argument(
        "--image-id",
        action="append",
        default=None,
        help="score only this manifest image; repeat for multiple images",
    )
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = validate_manifest(_read_json(manifest_path), manifest_path=manifest_path)
    requested_image_ids = [str(value) for value in (args.image_id or [])]
    selected_images = select_manifest_images(manifest, requested_image_ids)
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_jsonl = args.source_jsonl.expanduser().resolve(strict=True)

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

    resolved = load_infer_config(config_path)
    config = resolved.config
    if args.runtime_dtype == "fp32":
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"dtype": "fp32"})}
        )
    if config.backend.type != "hf":
        raise ValueError("complete candidate-row scoring requires backend.type: hf")
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=config_sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    raw_rows = load_raw_examples(source_jsonl)
    raw_by_id = {str(row.metadata.get("source", {}).get("image_id")): row for row in raw_rows}
    template = _template_config(config)
    output_images: list[dict[str, Any]] = []
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
        terminal_id = tokenizer.eos_token_id
        if terminal_id is None or int(terminal_id) < 0:
            raise ValueError("tokenizer does not expose eos_token_id")
        model_dtype = _runtime_model_dtype_summary(model)
        attention_implementation = _attention_implementation(
            model,
            config.backend.hf.attn_implementation,
        )
        backend_receipt = opened.receipt.to_artifact_dict()

        for image in selected_images:
            image_id = str(image["image_id"])
            raw = raw_by_id.get(image_id)
            if raw is None:
                raise ValueError(f"image {image_id} is absent from source JSONL")
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
            base_prompt = image.get("base_prompt_token_ids")
            if base_prompt is None:
                base_prompt = [int(value) for value in prompt_record.prompt_token_ids]
            else:
                base_prompt = [int(value) for value in base_prompt]
                if list(prompt_record.prompt_token_ids) != base_prompt:
                    raise ValueError(
                        f"{image_id} base prompt does not equal active processor prompt"
                    )
            request = DecodeRequest(
                request_id=f"complete-candidate-row:{image_id}",
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
            (
                native_inputs,
                executed_prompt_ids,
                observed_grids,
                executed_media_sha256,
            ) = opened._materialize_native_inputs((request,))  # noqa: SLF001
            if tuple(executed_prompt_ids[0]) != tuple(base_prompt):
                raise RuntimeError(
                    f"{image_id} materialized prompt differs from recorded base prompt"
                )
            image_grid_thw = native_inputs.get("image_grid_thw")
            if not isinstance(image_grid_thw, torch.Tensor):
                raise ValueError(f"{image_id} native inputs lack image_grid_thw")
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
            output_boundaries: list[dict[str, Any]] = []
            for boundary in image["boundaries"]:
                suffix = [int(item) for item in boundary["prefix_token_ids"]]
                prefix_mode = boundary["prefix_mode"]
                if prefix_mode == "base_prompt_plus_generated":
                    actual_prefix = [*base_prompt, *suffix]
                elif prefix_mode == "full_model_prompt":
                    actual_prefix = suffix
                else:
                    raise ValueError(f"unsupported prefix_mode={prefix_mode!r}")
                boundary_logits = _forward_logits(
                    model,
                    model_inputs,
                    actual_prefix,
                    image_grid_thw,
                )
                terminal = terminal_boundary_score(
                    boundary_logits,
                    boundary_length=len(actual_prefix),
                    row_entry_token_id=OBJECT_REF_START,
                    terminal_token_id=int(terminal_id),
                )
                candidate_scores = []
                for candidate in boundary["candidates"]:
                    row_tokens = [int(item) for item in candidate["row_token_ids"]]
                    logits = _forward_logits(
                        model,
                        model_inputs,
                        [*actual_prefix, *row_tokens],
                        image_grid_thw,
                    )
                    candidate_scores.append(
                        _score_candidate(
                            logits,
                            boundary_length=len(actual_prefix),
                            row_tokens=row_tokens,
                            metadata=candidate,
                        )
                    )
                output_boundaries.append({
                    "boundary_id": str(boundary["boundary_id"]),
                    "prefix_mode": prefix_mode,
                    "prefix_token_ids_sha256": _token_hash(actual_prefix),
                    "prefix_token_count": len(actual_prefix),
                    "recorded_prefix_reference": boundary["prefix_reference"],
                    "terminal_boundary": terminal,
                    "candidate_scores": candidate_scores,
                })
            output_images.append({
                "image_id": image_id,
                "base_prompt_token_ids_sha256": _token_hash(base_prompt),
                "base_prompt_reference": image.get("base_prompt_reference"),
                "source_image": {
                    "path": image_plan.image_path,
                    "sha256": image_plan.image_content_sha256,
                    "width": image_plan.decoded_width,
                    "height": image_plan.decoded_height,
                    "executed_media_sha256": executed_media_sha256[0],
                    "observed_image_grid_thw": (
                        None
                        if observed_grids[0] is None
                        else list(observed_grids[0])
                    ),
                },
                "boundaries": output_boundaries,
            })
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "unit_id": str(manifest.get("unit_id", "2026-07-19-complete-candidate-row-score-divergence")),
        "manifest": {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
        "images": output_images,
        "runtime": {
            "requested_image_ids": requested_image_ids,
            "image_selection": "requested_subset" if requested_image_ids else "all_manifest_images",
            "physical_batch_size": 1,
            "runtime_dtype_mode": str(args.runtime_dtype),
            "score_accumulation_dtype": "torch.float32",
            "model_dtype": model_dtype,
            "config_path": str(config_path),
            "authored_config_sha256": sha256_file(config_path),
            "resolved_config_fingerprint": resolved.fingerprint,
            "effective_config_sha256": config_sha256_json(
                config.model_dump(mode="json")
            ),
            "source_jsonl": str(source_jsonl),
            "source_jsonl_sha256": sha256_file(source_jsonl),
            "repetition_penalty_processing": False,
            "cache": False,
            "feature_gating_intervention": False,
            "attention_implementation": attention_implementation,
            "processor_model_vision_parity": parity,
            "backend_session": backend_receipt,
            "eos_token_id": int(terminal_id),
            "row_entry_token_id": int(OBJECT_REF_START),
        },
        "forbidden_comparisons": [
            "candidate rows are not normalized into a common probability distribution",
            "terminal boundary margins are not compared numerically with full-row scores",
        ],
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "receipt.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite immutable receipt: {output_path}")
    output_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
