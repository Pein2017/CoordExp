#!/usr/bin/env python3
"""Score manually admitted coordinate branch pairs from native rollouts.

This is an experiment-local companion to ``score_fixed_prompt_coordinate_branches``.
The source rows are taken from the current sampled-rollout artifact rather than
from the historical terminal-bundle format.  Validation is intentionally strict:
the two rows must come from the same image and artifact, use the same prompt, and
share their generated token prefix until the first differing coordinate.

The script replays the exact prompt and image with the current inference runtime,
then evaluates every coordinate state under both branch prefixes.  Probability
reports are computed in CPU float32.  A source artifact may be rescored with a
different checkpoint only with an explicit counterfactual declaration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.score_fixed_prompt_coordinate_branches import (  # noqa: E402
    COORDINATE_SLOT_NAMES,
    _build_chat_text,
    _forward_last_logits,
    _sha256_json,
    _as_mapping,
    _reference_bins,
    expected_model_composition_from_config,
    summarize_coordinate_logits,
)


SCHEMA_VERSION = "native_coordinate_branch_pairs.v1"
NATIVE_ROLLOUT_SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
FULL_BAG_K = "FULL_BAG_K"
FRESH_BASE_PROMPT = "fresh_base_prompt_per_call"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read JSON {path}: {exc}") from exc


def _resolve_path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("path value must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=True)


def _as_int(value: Any, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be an integer") from exc


def _tokenizer_encode(tokenizer: Any, text: str) -> list[int]:
    """Encode one span without adding prompt/special wrapper tokens."""

    try:
        try:
            encoded = tokenizer.encode(text, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer.encode(text)
    except Exception as exc:  # tokenizer implementations commonly raise KeyError for unknown text
        raise ValueError(f"tokenizer could not encode raw row span: {exc}") from exc
    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if isinstance(encoded, Sequence) and not isinstance(encoded, (str, bytes)):
        if encoded and isinstance(encoded[0], Sequence):
            encoded = encoded[0]
        return [_as_int(value, "encoded token") for value in encoded]
    raise ValueError("tokenizer.encode did not return a token-id sequence")


def _token_id(tokenizer: Any, token: str) -> int:
    value = tokenizer.convert_tokens_to_ids(token)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 1:
            raise ValueError(f"tokenizer returned multiple ids for {token}")
        value = value[0]
    value = _as_int(value, f"token id for {token}")
    if value < 0:
        raise ValueError(f"tokenizer does not know special token {token}")
    return value


def coordinate_token_ids(tokenizer: Any) -> dict[int, int]:
    """Return the complete coordinate-bin to token-id map used by the model."""

    result: dict[int, int] = {}
    seen: set[int] = set()
    for coordinate in range(1000):
        token = _token_id(tokenizer, f"<|coord_{coordinate}|>")
        if token in seen:
            raise ValueError(f"coordinate token id {token} is reused")
        seen.add(token)
        result[coordinate] = token
    return result


def _find_subsequence(haystack: Sequence[int], needle: Sequence[int]) -> list[int]:
    if not needle:
        raise ValueError("raw row span tokenization is empty")
    return [
        start
        for start in range(0, len(haystack) - len(needle) + 1)
        if list(haystack[start : start + len(needle)]) == list(needle)
    ]


def _parse_native_row(
    prediction: Mapping[str, Any],
    rollout: Mapping[str, Any],
    *,
    tokenizer: Any,
    coordinate_ids: Mapping[int, int],
    category: str,
) -> dict[str, Any]:
    """Locate and structurally validate a parsed row in generated token ids."""

    description = str(prediction.get("description", "")).strip().lower()
    if description != category:
        raise ValueError(f"prediction category {description!r} does not match case category {category!r}")
    raw_span = prediction.get("raw_span_text")
    if not isinstance(raw_span, str) or not raw_span:
        raise ValueError("prediction raw_span_text must be non-empty")
    generated = rollout.get("generated_token_ids")
    if not isinstance(generated, list) or not generated:
        raise ValueError("rollout generated_token_ids must be a non-empty list")
    generated_ids = [_as_int(value, "generated token id") for value in generated]
    span_ids = _tokenizer_encode(tokenizer, raw_span)
    locations = _find_subsequence(generated_ids, span_ids)
    if len(locations) != 1:
        raise ValueError(f"raw_span_text token subsequence occurs {len(locations)} times; expected exactly once")
    start = locations[0]
    object_start = _token_id(tokenizer, "<|object_ref_start|>")
    object_end = _token_id(tokenizer, "<|object_ref_end|>")
    box_start = _token_id(tokenizer, "<|box_start|>")
    box_end = _token_id(tokenizer, "<|box_end|>")
    if span_ids[0] != object_start:
        raise ValueError("raw row span does not start with object_ref_start")
    try:
        object_end_index = span_ids.index(object_end, 1)
    except ValueError as exc:
        raise ValueError("raw row span lacks object_ref_end") from exc
    if object_end_index <= 1 or object_end_index + 1 >= len(span_ids) or span_ids[object_end_index + 1] != box_start:
        raise ValueError("raw row span has malformed object_ref_end/box_start structure")
    box_start_index = object_end_index + 1
    coord_start = box_start_index + 1
    coord_end = coord_start + 4
    if coord_end >= len(span_ids) or span_ids[coord_end] != box_end or len(span_ids) != coord_end + 1:
        raise ValueError("raw row span must contain exactly four coordinates followed by box_end")
    reverse_coordinates = {int(token): int(bin_value) for bin_value, token in coordinate_ids.items()}
    coord_token_ids = [int(value) for value in span_ids[coord_start:coord_end]]
    try:
        bins = [reverse_coordinates[token] for token in coord_token_ids]
    except KeyError as exc:
        raise ValueError(f"raw row span contains non-coordinate token {exc.args[0]}") from exc
    reported_bins = prediction.get("coord_bins")
    if not isinstance(reported_bins, Sequence) or isinstance(reported_bins, (str, bytes)) or len(reported_bins) != 4:
        raise ValueError("prediction coord_bins must contain four bins")
    reported = [_as_int(value, "prediction coordinate bin") for value in reported_bins]
    if reported != bins:
        raise ValueError(f"prediction coord_bins {reported} disagree with raw span token ids {bins}")
    generated_order = _as_int(prediction.get("generated_order"), "prediction generated_order")
    return {
        "description": description,
        "generated_order": generated_order,
        "coord_bins": bins,
        "coord_token_ids": coord_token_ids,
        "raw_span_text": raw_span,
        "raw_span_token_ids": span_ids,
        "generated_span_start": int(start),
        "coordinate_generated_steps": [int(start + coord_start + index) for index in range(4)],
        "generated_token_ids": generated_ids,
        "prediction": dict(prediction),
    }


def _artifact_composition(document: Mapping[str, Any]) -> dict[str, str | None]:
    identity = _as_mapping(document.get("model_identity"), "artifact model_identity")
    nested = identity.get("model_identity")
    if isinstance(nested, Mapping):
        identity = nested
    base = identity.get("base")
    adapter = identity.get("adapter")
    embedding = identity.get("embedding_delta")
    base_map = base if isinstance(base, Mapping) else {}
    adapter_map = adapter if isinstance(adapter, Mapping) else {}
    embedding_map = embedding if isinstance(embedding, Mapping) else {}
    nested_embedding = embedding_map.get("identity")
    if isinstance(nested_embedding, Mapping):
        embedding_map = nested_embedding
    base_path = base_map.get("path") or adapter_map.get("base_model_path")
    adapter_path = adapter_map.get("adapter_path")
    delta_path = embedding_map.get("delta_path")
    return {
        "base_model_path": None if not base_path else str(Path(str(base_path)).expanduser().resolve()),
        "adapter_path": None if not adapter_path else str(Path(str(adapter_path)).expanduser().resolve()),
        "adapter_type": None if not adapter_map.get("adapter_type") else str(adapter_map["adapter_type"]),
        "adapter_name": None if not adapter_map.get("adapter_name") else str(adapter_map["adapter_name"]),
        "embedding_delta_path": None if not delta_path else str(Path(str(delta_path)).expanduser().resolve()),
    }


def _native_rollout(document: Mapping[str, Any], *, seed: int, generated_order: int, image_id: str) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if document.get("schema_version") != NATIVE_ROLLOUT_SCHEMA_VERSION:
        raise ValueError(f"native artifact schema must be {NATIVE_ROLLOUT_SCHEMA_VERSION}")
    rollouts = document.get("rollouts")
    if not isinstance(rollouts, list) or not rollouts:
        raise ValueError("native artifact rollouts must be a non-empty list")
    seen_seeds: set[int] = set()
    for item in rollouts:
        row = _as_mapping(item, "native rollout")
        current_seed = _as_int(row.get("seed"), "rollout seed")
        if current_seed in seen_seeds:
            raise ValueError(f"native artifact contains duplicate seed {current_seed}")
        seen_seeds.add(current_seed)
    candidates = [row for row in rollouts if _as_int(row.get("seed"), "rollout seed") == int(seed)]
    if len(candidates) != 1:
        raise ValueError(f"seed {seed} resolves to {len(candidates)} rollouts")
    rollout = candidates[0]
    if str(rollout.get("image_id")) != str(image_id):
        raise ValueError(f"rollout image {rollout.get('image_id')!r} does not match case image {image_id!r}")
    prompt_ids = rollout.get("prompt_token_ids")
    if not isinstance(prompt_ids, list) or not prompt_ids:
        raise ValueError("native rollout prompt_token_ids must be a non-empty list")
    parsed = _as_mapping(rollout.get("predictions"), "native rollout predictions")
    predictions = parsed.get("predictions")
    if not isinstance(predictions, list):
        raise ValueError("native rollout predictions.predictions must be a list")
    selected = [
        _as_mapping(item, "native prediction")
        for item in predictions
        if _as_int(_as_mapping(item, "native prediction").get("generated_order"), "prediction generated_order") == int(generated_order)
    ]
    if len(selected) != 1:
        raise ValueError(f"generated_order {generated_order} resolves to {len(selected)} predictions")
    return rollout, selected[0]


def validate_native_case(
    case: Mapping[str, Any],
    document: Mapping[str, Any],
    *,
    tokenizer: Any,
    allow_counterfactual_model_composition: bool = False,
    expected_model_composition: Mapping[str, str | None] | None = None,
    counterfactual_purpose: str | None = None,
) -> dict[str, Any]:
    """Validate and decode one admitted native branch pair without model replay."""

    if allow_counterfactual_model_composition and not str(counterfactual_purpose or "").strip():
        raise ValueError("counterfactual model composition mode requires a non-empty purpose")
    if allow_counterfactual_model_composition and expected_model_composition is None:
        raise ValueError("counterfactual model composition mode requires the current inference composition")
    if counterfactual_purpose and not allow_counterfactual_model_composition:
        raise ValueError("counterfactual purpose requires explicit counterfactual mode")
    image_id = str(case.get("image_id"))
    category = str(case.get("category", "")).strip().lower()
    if not image_id or not category:
        raise ValueError("case requires non-empty image_id and category")
    clean_spec = _as_mapping(case.get("clean"), "case clean")
    degraded_spec = _as_mapping(case.get("degraded"), "case degraded")
    coordinate_ids = coordinate_token_ids(tokenizer)
    source_identity = _artifact_composition(document)
    composition_differs = expected_model_composition is not None and dict(expected_model_composition) != source_identity
    if composition_differs and not allow_counterfactual_model_composition:
        raise ValueError(
            "native artifact model composition does not match the current inference configuration: "
            f"observed={source_identity!r}, expected={dict(expected_model_composition)!r}"
        )
    clean_rollout, clean_prediction = _native_rollout(
        document,
        seed=_as_int(clean_spec.get("seed"), "clean seed"),
        generated_order=_as_int(clean_spec.get("generated_order"), "clean generated_order"),
        image_id=image_id,
    )
    degraded_rollout, degraded_prediction = _native_rollout(
        document,
        seed=_as_int(degraded_spec.get("seed"), "degraded seed"),
        generated_order=_as_int(degraded_spec.get("generated_order"), "degraded generated_order"),
        image_id=image_id,
    )
    clean_ids = [_as_int(value, "clean prompt token") for value in clean_rollout["prompt_token_ids"]]
    degraded_ids = [_as_int(value, "degraded prompt token") for value in degraded_rollout["prompt_token_ids"]]
    if clean_ids != degraded_ids:
        raise ValueError("clean and degraded rollouts do not share exact prompt token ids")
    clean = _parse_native_row(clean_prediction, clean_rollout, tokenizer=tokenizer, coordinate_ids=coordinate_ids, category=category)
    degraded = _parse_native_row(degraded_prediction, degraded_rollout, tokenizer=tokenizer, coordinate_ids=coordinate_ids, category=category)
    if clean["coordinate_generated_steps"] != degraded["coordinate_generated_steps"]:
        raise ValueError("clean and degraded coordinate generated steps differ")
    first_slot = next((index for index, (left, right) in enumerate(zip(clean["coord_token_ids"], degraded["coord_token_ids"])) if left != right), None)
    if first_slot is None:
        raise ValueError("clean and degraded rows have no differing coordinate slot")
    first_step = clean["coordinate_generated_steps"][first_slot]
    if clean["generated_token_ids"][:first_step] != degraded["generated_token_ids"][:first_step]:
        raise ValueError("clean and degraded generated prefixes differ before the first coordinate difference")
    return {
        "case_name": str(case.get("name", "")),
        "image_id": image_id,
        "category": category,
        "role": str(case.get("role", "primary")),
        "source_model_composition": source_identity,
        "executed_model_composition": None if expected_model_composition is None else dict(expected_model_composition),
        "counterfactual_model_composition": {
            "enabled": bool(allow_counterfactual_model_composition),
            "source_and_executed_composition_differ": bool(composition_differs),
            "purpose": None if counterfactual_purpose is None else str(counterfactual_purpose).strip(),
        },
        "prompt_token_ids": clean_ids,
        "prompt_token_ids_sha256": _sha256_json(clean_ids),
        "clean": {
            **clean,
            "seed": _as_int(clean_spec["seed"], "clean seed"),
            "executed_media_sha256": clean_rollout.get("executed_media_sha256"),
            "example_id": clean_rollout.get("example_id"),
        },
        "degraded": {
            **degraded,
            "seed": _as_int(degraded_spec["seed"], "degraded seed"),
            "executed_media_sha256": degraded_rollout.get("executed_media_sha256"),
            "example_id": degraded_rollout.get("example_id"),
        },
        "first_differing_coordinate_slot_index": int(first_slot),
        "first_differing_coordinate_slot": COORDINATE_SLOT_NAMES[first_slot],
        "first_differing_generated_step": int(first_step),
    }


def score_native_coordinate_states(
    validated: Mapping[str, Any],
    logits_by_branch_slot: Mapping[str, Sequence[Sequence[float]]],
    *,
    coordinate_ids: Mapping[int, int],
    reference_bins: Sequence[int],
) -> dict[str, Any]:
    """Build paired coordinate reports from four supplied logits per branch.

    Keeping this function model-free makes the accounting and log-probability
    aggregation unit-testable without loading Qwen3-VL.
    """

    if len(reference_bins) != 4:
        raise ValueError("reference_bins must contain four values")
    reports: dict[str, Any] = {}
    for branch in ("clean", "degraded"):
        branch_info = _as_mapping(validated.get(branch), f"validated {branch}")
        bins = [int(value) for value in branch_info["coord_bins"]]
        logits_rows = logits_by_branch_slot.get(branch)
        if not isinstance(logits_rows, Sequence) or len(logits_rows) != 4:
            raise ValueError(f"{branch} must provide four coordinate logit rows")
        branch_reports: list[dict[str, Any]] = []
        path_log_probability = 0.0
        path_finite = True
        for slot, logits in enumerate(logits_rows):
            summary = summarize_coordinate_logits(
                logits,
                coordinate_ids,
                clean_coordinate=int(validated["clean"]["coord_bins"][slot]),
                bad_coordinate=int(validated["degraded"]["coord_bins"][slot]),
                reference_coordinate=int(reference_bins[slot]),
            )
            source_token_id = int(branch_info["coord_token_ids"][slot])
            raw = summary["coordinate_argmax"]  # only to force a stable schema shape below
            del raw
            token_record = summary["clean"] if branch == "clean" else summary["bad"]
            raw_record = token_record["raw_temperature_1"]
            raw_probability = float(raw_record["probability_over_full_vocabulary"])
            log_probability = math.log(raw_probability) if raw_probability > 0.0 else float("-inf")
            if not math.isfinite(log_probability):
                path_finite = False
            else:
                path_log_probability += log_probability
            branch_reports.append(
                {
                    "slot_index": slot,
                    "slot": COORDINATE_SLOT_NAMES[slot],
                    "generation_step": int(branch_info["coordinate_generated_steps"][slot]),
                    "branch_coordinate_bin": bins[slot],
                    "branch_token_id": source_token_id,
                    "source_token_probability_raw_temperature_1": raw_probability,
                    "source_token_log_probability_raw_temperature_1": log_probability,
                    "source_token_rank_raw_temperature_1": int(raw_record["full_vocabulary_rank"]),
                    "source_token_probability_given_coordinate_vocabulary": float(raw_record["probability_given_coordinate_token"]),
                    "coordinate_report": summary,
                }
            )
        reports[branch] = {
            "coordinate_states": branch_reports,
            "coordinate_path_log_probability_raw_temperature_1": path_log_probability if path_finite else float("-inf"),
            "coordinate_path_log_probability_is_finite": path_finite,
        }
    return {
        "clean": reports["clean"],
        "degraded": reports["degraded"],
        "paired_by_slot": [
            {
                "slot_index": slot,
                "slot": COORDINATE_SLOT_NAMES[slot],
                "clean": reports["clean"]["coordinate_states"][slot],
                "degraded": reports["degraded"]["coordinate_states"][slot],
            }
            for slot in range(4)
        ],
    }


def load_cases(path: Path) -> tuple[list[dict[str, Any]], Path]:
    payload = _read_json(path)
    if isinstance(payload, list):
        values = payload
    else:
        values = _as_mapping(payload, "cases document").get("cases")
    if not isinstance(values, list) or not values:
        raise ValueError("cases document must contain a non-empty cases list")
    return [dict(_as_mapping(value, "case")) for value in values], path.parent.resolve()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="fp32")
    parser.add_argument("--allow-counterfactual-model-composition", action="store_true")
    parser.add_argument("--counterfactual-purpose")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.allow_counterfactual_model_composition and not str(args.counterfactual_purpose or "").strip():
        parser.error("--counterfactual-purpose is required with --allow-counterfactual-model-composition")
    if args.counterfactual_purpose and not args.allow_counterfactual_model_composition:
        parser.error("--counterfactual-purpose requires --allow-counterfactual-model-composition")
    return args


def main() -> int:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    if output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output}; pass --force")
    cases, cases_root = load_cases(args.cases.expanduser().resolve(strict=True))
    import torch
    from PIL import Image
    from src.config.fingerprint import sha256_file, sha256_json
    from src.config.inference import load_infer_config
    from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
    from src.inference.runtime import assemble_frontend

    infer_config_path = args.infer_config.expanduser().resolve(strict=True)
    resolved = load_infer_config(infer_config_path)
    config = resolved.config.model_copy(
        update={"model": resolved.config.model.model_copy(update={"dtype": "fp32" if args.dtype == "fp32" else "bf16"})}
    )
    expected_composition = expected_model_composition_from_config(config)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    output_cases: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as session:
        coordinate_ids = coordinate_token_ids(session._tokenizer)
        for case in cases:
            artifact_path = _resolve_path(case.get("source_rollout_artifact"), base=cases_root)
            document = _read_json(artifact_path)
            validated = validate_native_case(
                case,
                document,
                tokenizer=session._tokenizer,
                allow_counterfactual_model_composition=args.allow_counterfactual_model_composition,
                expected_model_composition=expected_composition,
                counterfactual_purpose=args.counterfactual_purpose,
            )
            image_path = _resolve_path(case.get("replay_image_path") or case.get("image_path"), base=cases_root)
            with Image.open(image_path) as image:
                width, height = (int(value) for value in image.size)
            image_sha = sha256_file(image_path)
            chat_text = _build_chat_text(session._tokenizer_processor if hasattr(session, "_tokenizer_processor") else frontend.qwen.processor, config, image_path)
            prompt_ids = tuple(int(value) for value in validated["prompt_token_ids"])
            request = DecodeRequest(
                request_id=f"native-coordinate-branch:{validated['case_name']}",
                chat_text=chat_text,
                input_prompt_token_ids=prompt_ids,
                expected_executed_prompt_token_ids=prompt_ids,
                image_path=str(image_path),
                declared_image_width=width,
                declared_image_height=height,
                decoded_image_width=width,
                decoded_image_height=height,
                image_sha256=image_sha,
                generation_policy=GenerationPolicy(max_new_tokens=8, repetition_penalty=1.0),
            )
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            if tuple(executed_ids[0]) != prompt_ids:
                raise RuntimeError("runtime prompt token ids differ from source rollout prompt ids")
            for branch in ("clean", "degraded"):
                expected_media = str(validated[branch].get("executed_media_sha256") or "")
                if expected_media and str(media_sha[0]) != expected_media:
                    raise RuntimeError("runtime executed image media hash differs from source rollout artifact")
            logits_by_branch_slot: dict[str, list[list[float]]] = {"clean": [], "degraded": []}
            for branch in ("clean", "degraded"):
                branch_info = validated[branch]
                for step in branch_info["coordinate_generated_steps"]:
                    logits_by_branch_slot[branch].append(_forward_last_logits(session, native_inputs, branch_info["generated_token_ids"][:step]))
            reference_box = case.get("reference_box_xyxy")
            if not isinstance(reference_box, Sequence) or len(reference_box) != 4:
                raise ValueError("case reference_box_xyxy must contain four values")
            reference_bins = _reference_bins(reference_box, width, height)
            reports = score_native_coordinate_states(validated, logits_by_branch_slot, coordinate_ids=coordinate_ids, reference_bins=reference_bins)
            output_cases.append({
                **validated,
                "source_rollout_artifact": str(artifact_path),
                "image_path": str(_resolve_path(case.get("image_path"), base=cases_root)),
                "replay_image_path": str(image_path),
                "image_dimensions": {"width": width, "height": height},
                "reference_box_xyxy_pixels": [float(value) for value in reference_box],
                "reference_box_xyxy_coordinate_bins": reference_bins,
                "observed_prompt_token_ids_sha256": _sha256_json(list(executed_ids[0])),
                "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                "executed_media_sha256": media_sha[0],
                "coordinate_reports": reports,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "schema_version": SCHEMA_VERSION,
        "config": {"infer_config_path": str(infer_config_path), "resolved_fingerprint": resolved.fingerprint, "dtype": config.model.dtype, "device": str(args.device)},
        "case_count": len(output_cases),
        "cases": output_cases,
    }, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
