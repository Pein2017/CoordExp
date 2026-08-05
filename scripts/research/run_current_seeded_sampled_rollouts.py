#!/usr/bin/env python3
"""Run small, request-scoped seeded rollouts with the current HF runtime.

This is intentionally an experiment-local seam.  The canonical CoordExp-Swift
backend remains deterministic; this script reuses its prompt/image
materialization and then calls the already-loaded Hugging Face model directly
for one request at a time.  Positive temperatures collect seeded samples;
temperature zero uses true greedy decoding.  It is for collecting on-policy
rollouts, not for producing benchmark inference artifacts.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
DEFAULT_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_dora_step4887_fixed_prompt_coordinate_branches.yaml"
)


def parse_seed_list(value: str | Iterable[int]) -> tuple[int, ...]:
    """Parse a non-empty, duplicate-free ordered seed list."""

    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
    else:
        pieces = list(value)
    try:
        seeds = tuple(int(piece) for piece in pieces)
    except (TypeError, ValueError) as exc:
        raise ValueError("seeds must be integers, for example 11,12,13") from exc
    if not seeds:
        raise ValueError("at least one seed is required")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be unique so each rollout is auditable")
    return seeds


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _parse_image_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    ids = {piece.strip() for piece in value.split(",") if piece.strip()}
    if not ids:
        raise ValueError("--image-ids must contain at least one non-empty id")
    return ids


def physical_image_id(example: Any) -> int | str:
    """Return the dataset image id, falling back to the internal example id."""

    metadata = getattr(example, "metadata", {})
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    if isinstance(source, Mapping) and source.get("image_id") is not None:
        value = source["image_id"]
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)
    return str(example.example_id)


def select_examples(examples: Sequence[Any], image_ids: set[str] | None) -> list[Any]:
    """Select by physical dataset image id or internal example id."""

    if image_ids is None:
        return list(examples)
    selected = [
        example
        for example in examples
        if str(example.example_id) in image_ids
        or str(physical_image_id(example)) in image_ids
    ]
    if not selected:
        raise ValueError("no selected images remain after --image-ids filtering")
    return selected


def _trim_generated_ids(
    generated_ids: Sequence[int],
    *,
    stop_token_id: int,
    pad_token_id: int,
) -> tuple[list[int], str]:
    """Remove padding after the first stop token and report the stop reason."""

    kept: list[int] = []
    for token_id in generated_ids:
        token_id = int(token_id)
        if token_id == stop_token_id:
            return kept, "im_end"
        if token_id == pad_token_id:
            raise ValueError("generation emitted pad before stop token")
        kept.append(token_id)
    return kept, "length"


def validate_artifact_payload(payload: Mapping[str, Any]) -> None:
    """Validate the compact, pre-write shape without requiring a model."""

    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unexpected sampled-rollout schema version")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("artifact config metadata is required")
    config_decode_mode = config.get("decode_mode")
    if config_decode_mode is not None and config_decode_mode not in {"greedy", "sampled"}:
        raise ValueError("unsupported decode_mode")
    rows = payload.get("rollouts")
    if not isinstance(rows, list) or not rows:
        raise ValueError("artifact must contain at least one rollout")
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("rollout rows must be objects")
        for field in ("image_id", "seed", "generated_token_ids", "generated_text", "stop_reason", "predictions"):
            if field not in row:
                raise ValueError(f"rollout is missing {field}")
        if not isinstance(row["generated_token_ids"], list):
            raise ValueError("generated_token_ids must be a list")
        if row["stop_reason"] not in {"im_end", "length"}:
            raise ValueError("unsupported stop_reason")
        if not isinstance(row["predictions"], Mapping):
            raise ValueError("predictions must contain parser evidence")
        row_decode_mode = row.get("decode_mode")
        if row_decode_mode is not None and row_decode_mode not in {"greedy", "sampled"}:
            raise ValueError("unsupported decode_mode")
        if (
            config_decode_mode is not None
            and row_decode_mode is not None
            and row_decode_mode != config_decode_mode
        ):
            raise ValueError("rollout decode_mode does not match artifact config")


def _processor_config(config: Any) -> Any:
    from src.config.models import ProcessorConfig

    return ProcessorConfig(
        do_resize=config.model.processor.do_resize,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )


def _template_config(config: Any) -> Any:
    from src.config.models import TemplateConfig, TemplatePromptConfig

    return TemplateConfig(
        object_field_order=config.template.object_field_order,
        object_ordering=config.template.object_ordering,
        assistant_format=config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=config.template.prompt.system,
            user=config.template.prompt.user,
        ),
    )


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def decode_mode_for_temperature(temperature: float) -> str:
    """Return the generation mode selected by the requested temperature."""

    if not 0.0 <= temperature:
        raise ValueError("temperature must be non-negative")
    return "greedy" if temperature == 0.0 else "sampled"


def _build_generate_kwargs(
    native_inputs: Mapping[str, Any],
    *,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    eos_token_id: int,
    pad_token_id: int,
) -> tuple[dict[str, Any], str]:
    """Build Hugging Face generation kwargs without sampling-only options for greedy mode."""

    decode_mode = decode_mode_for_temperature(temperature)
    generate_kwargs: dict[str, Any] = {
        **dict(native_inputs),
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": decode_mode == "sampled",
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "return_dict_in_generate": True,
        "output_scores": False,
    }
    if decode_mode == "sampled":
        generate_kwargs.update({
            "temperature": float(temperature),
            "top_p": float(top_p),
        })
    else:
        generate_kwargs.pop("temperature", None)
        generate_kwargs.pop("top_p", None)
    return generate_kwargs, decode_mode


def _sample_one(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    prompt_width: int,
    seed: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    tokenizer: Any,
) -> tuple[list[int], str, str]:
    import torch

    decode_mode = decode_mode_for_temperature(temperature)
    if decode_mode == "sampled":
        _seed_torch(seed)
    generate_kwargs, _ = _build_generate_kwargs(
        native_inputs,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        eos_token_id=session._im_end_token_id(),
        pad_token_id=session._pad_token_id(),
    )
    with torch.inference_mode():
        output = session._model.generate(**generate_kwargs)
    sequences = getattr(output, "sequences", None)
    if sequences is None or int(sequences.shape[0]) != 1:
        raise RuntimeError("sampled generation did not return one sequence")
    generated = [int(value) for value in sequences[0, prompt_width:].tolist()]
    kept, stop_reason = _trim_generated_ids(
        generated,
        stop_token_id=session._im_end_token_id(),
        pad_token_id=session._pad_token_id(),
    )
    text = tokenizer.decode(kept, skip_special_tokens=False)
    return kept, text, stop_reason


def _build_requests(config: Any, frontend: Any, raw_examples: Sequence[Any]) -> tuple[list[Any], dict[str, Any]]:
    from src.inference.image_plan import plan_image_batch
    from src.inference.prompt import build_prompt_record

    plans = plan_image_batch(
        list(raw_examples),
        components=frontend.qwen,
        processor_config=_processor_config(config),
        row_indices=list(range(len(raw_examples))),
    )
    by_id = {row.row_id: row for row in plans.rows}
    requests: list[Any] = []
    prompt_meta: dict[str, Any] = {}
    for index, example in enumerate(raw_examples):
        row = by_id[example.example_id]
        record = build_prompt_record(
            example,
            _template_config(config),
            processor=frontend.qwen.processor,
            row_index=index,
            merged_visual_tokens=row.merged_visual_tokens,
            object_order_seed=config.template.object_order_seed,
        )
        from src.inference.backend import DecodeRequest, GenerationPolicy

        requests.append(
            DecodeRequest(
                request_id=str(example.example_id),
                chat_text=record.chat_text,
                input_prompt_token_ids=tuple(record.input_prompt_token_ids),
                expected_executed_prompt_token_ids=tuple(record.expected_executed_prompt_token_ids),
                image_path=row.image_path,
                declared_image_width=row.declared_width,
                declared_image_height=row.declared_height,
                decoded_image_width=row.decoded_width,
                decoded_image_height=row.decoded_height,
                image_sha256=row.image_content_sha256,
                expected_image_grid_thw=tuple(row.expected_image_grid_thw),
                logical_transform_id=row.logical_transform_id,
                generation_policy=GenerationPolicy(max_new_tokens=1),
            )
        )
        prompt_meta[str(example.example_id)] = {
            "prompt_token_ids": list(record.expected_executed_prompt_token_ids),
            "prompt_token_ids_sha256": _sha256_json(record.expected_executed_prompt_token_ids),
            "chat_text_sha256": hashlib.sha256(record.chat_text.encode()).hexdigest(),
            "image_path": str(row.image_path),
            "image_sha256": row.image_content_sha256,
            "width": row.decoded_width,
            "height": row.decoded_height,
        }
    return requests, prompt_meta


def run_sampling(
    *,
    infer_config: Path,
    output: Path,
    image_ids: set[str] | None,
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    device: str,
    replay_first_seed: bool = False,
    force: bool = False,
) -> Path:
    import torch

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.parsing import parse_compact_object_box_closed
    from src.inference.runtime import assemble_frontend

    if output.exists() and not force:
        raise ValueError(f"refusing to overwrite {output}; pass --force")
    decode_mode = decode_mode_for_temperature(temperature)
    if not 0.0 < top_p <= 1.0:
        raise ValueError("top_p must be in (0, 1]")
    resolved = load_infer_config(infer_config.resolve(strict=True))
    config = resolved.config
    raw_examples = list(load_raw_examples(config.data.input_jsonl))
    raw_examples = select_examples(raw_examples, image_ids)
    if torch.cuda.is_available() and device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
    )
    requests, prompt_meta = _build_requests(config, frontend, raw_examples)
    output.parent.mkdir(parents=True, exist_ok=True)
    rollouts: list[dict[str, Any]] = []
    replay_check: dict[str, Any] = {
        "enabled": bool(replay_first_seed),
        "status": "not_requested",
        "checked_images": [],
    }
    with open_backend_session(frontend.launch) as session:
        for request, example in zip(requests, raw_examples, strict=True):
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            if tuple(executed_ids[0]) != request.expected_executed_prompt_token_ids:
                raise RuntimeError(f"prompt token parity failed for image {example.example_id}")
            prompt_width = int(native_inputs["input_ids"].shape[1])
            for seed in seeds:
                ids, text, stop_reason = _sample_one(
                    session=session,
                    native_inputs=native_inputs,
                    prompt_width=prompt_width,
                    seed=int(seed),
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_new_tokens=max_new_tokens,
                    tokenizer=session._tokenizer,
                )
                if replay_first_seed and int(seed) == int(seeds[0]):
                    replay_ids, replay_text, replay_stop_reason = _sample_one(
                        session=session,
                        native_inputs=native_inputs,
                        prompt_width=prompt_width,
                        seed=int(seed),
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        max_new_tokens=max_new_tokens,
                        tokenizer=session._tokenizer,
                    )
                    if (ids, text, stop_reason) != (replay_ids, replay_text, replay_stop_reason):
                        raise RuntimeError(
                            f"same-seed replay mismatch for image {example.example_id}"
                        )
                    replay_check["checked_images"].append(str(example.example_id))
                parsed = parse_compact_object_box_closed(
                    text,
                    row_id=f"{example.example_id}:seed-{seed}",
                    row_index=0,
                    image_width=int(example.image.width),
                    image_height=int(example.image.height),
                )
                rollouts.append(
                    {
                        "image_id": physical_image_id(example),
                        "example_id": str(example.example_id),
                        "seed": int(seed),
                        "decode_mode": decode_mode,
                        "generated_token_ids": ids,
                        "generated_token_ids_sha256": _sha256_json(ids),
                        "generated_text": text,
                        "stop_reason": stop_reason,
                        "prompt_token_ids": list(executed_ids[0]),
                        "prompt_token_ids_sha256": _sha256_json(executed_ids[0]),
                        "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                        "executed_media_sha256": media_sha[0],
                        "predictions": parsed.to_artifact_dict(),
                    }
                )
    if replay_first_seed:
        replay_check["status"] = "passed"
    receipt_artifact = session.receipt.to_artifact_dict()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": "experiment_local_sampled_counterfactual_to_canonical_deterministic_backend",
        "config": {
            "infer_config_path": str(infer_config.resolve()),
            "resolved_fingerprint": resolved.fingerprint,
            "model_dtype": str(config.model.dtype),
            "device": device,
            "temperature": float(temperature),
            "top_p": float(top_p),
            "decode_mode": decode_mode,
            "repetition_penalty": float(repetition_penalty),
            "max_new_tokens": int(max_new_tokens),
            "seeds": [int(seed) for seed in seeds],
            "image_ids": sorted(prompt_meta),
            "sampling_is_not_infer_config": True,
        },
        "replay_check": replay_check,
        "model_identity": receipt_artifact,
        "prompt_metadata": prompt_meta,
        "rollout_count": len(rollouts),
        "rollouts": rollouts,
    }
    validate_artifact_payload(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids", help="Comma-separated example IDs; default is every row in the config JSONL")
    parser.add_argument("--seeds", default="11,12,13,14,15,16,17,18")
    parser.add_argument("--temperature", type=float, default=0.4)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--replay-first-seed-check",
        action="store_true",
        help="Run the first seed twice per image and fail if generated ids/text differ.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        args.seeds = parse_seed_list(args.seeds)
        args.image_ids = _parse_image_ids(args.image_ids)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def main() -> int:
    args = _parse_args()
    run_sampling(
        infer_config=args.infer_config,
        output=args.output,
        image_ids=args.image_ids,
        seeds=args.seeds,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        replay_first_seed=args.replay_first_seed_check,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
