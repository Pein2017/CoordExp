"""Collect Source256 K4 raw-softmax banks through native Qwen generation."""
from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import torch

from src.config.inference import load_research_infer_config
from src.config.fingerprint import sha256_json
from src.data import load_raw_examples
from src.inference.parsing import parse_compact_object_box_closed
from src.inference.runtime import assemble_frontend
from src.qwen.generation import NativeGenerationPolicy, generate_continuations
from .runtime import DEFAULT_CONFIG, build_request, load_policy, materialize

SCHEMA_VERSION = "current_seeded_sampled_rollouts.v1"
RAW_SOFTMAX_POLICY = NativeGenerationPolicy(
    temperature=1.0, top_p=1.0, repetition_penalty=1.0,
    top_k=0, use_model_defaults=False,
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


def sample_one(model, tokenizer, batch, *, seed, max_new_tokens, eos_token_id, pad_token_id):
    """Keep legacy bank bodies without EOS, including sampled interior pad IDs."""
    result, = generate_continuations(
        model, batch, extensions=((),), budgets=(max_new_tokens,),
        eos_token_id=eos_token_id, pad_token_id=pad_token_id,
        policy=RAW_SOFTMAX_POLICY, trace="none", seed=seed, allow_pad_tokens=True,
    )
    ids = list(result.token_ids)
    if result.stop_reason == "im_end":
        if not ids or ids[-1] != eos_token_id:
            raise ValueError("native sampled EOS result lacks its terminal token")
        ids.pop()
    return ids, tokenizer.decode(ids, skip_special_tokens=False), result.stop_reason


def run_sampling(*, infer_config, output, image_ids, seeds, max_new_tokens, device,
                 replay_first_seed=False):
    if output.exists():
        raise ValueError(f"refusing to overwrite {output}")
    if max_new_tokens <= 0:
        raise ValueError("sampling max_new_tokens must be positive")
    resolved = load_research_infer_config(infer_config)
    config = resolved.config
    raw_examples = select_examples(list(load_raw_examples(config.data.input_jsonl)), image_ids)
    if device.startswith("cuda"):
        torch.cuda.set_device(torch.device(device))
    frontend = assemble_frontend(
        config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json"))
    )
    qwen, identity = load_policy(config, device=device)
    eos_token_id = qwen.token_identity.im_end_token_ids[0]
    pad_token_id = qwen.tokenizer.pad_token_id
    started = time.monotonic()
    prompt_meta, rollouts = {}, []
    checked = []
    for index, raw in enumerate(raw_examples):
        request, image, prompt = build_request(raw, config=config, qwen=frontend.qwen, row_index=index)
        batch = materialize(qwen, request)
        ids = list(batch.prompt_token_ids[0])
        prompt_meta[str(raw.example_id)] = {
            "prompt_token_ids": ids, "prompt_token_ids_sha256": _sha256_json(ids),
            "chat_text_sha256": hashlib.sha256(prompt.chat_text.encode()).hexdigest(),
            "image_path": str(image.image_path), "image_sha256": image.image_content_sha256,
            "width": image.decoded_width, "height": image.decoded_height,
        }
        for seed in seeds:
            sample = sample_one(qwen.model, qwen.tokenizer, batch, seed=seed,
                                max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
                                pad_token_id=pad_token_id)
            if replay_first_seed and seed == seeds[0]:
                replay = sample_one(qwen.model, qwen.tokenizer, batch, seed=seed,
                                    max_new_tokens=max_new_tokens, eos_token_id=eos_token_id,
                                    pad_token_id=pad_token_id)
                if sample != replay:
                    raise ValueError(f"same-seed replay mismatch: {raw.example_id}")
                checked.append(str(raw.example_id))
            action_ids, text, stop_reason = sample
            parsed = parse_compact_object_box_closed(
                text, row_id=f"{raw.example_id}:seed-{seed}", row_index=0,
                image_width=int(raw.image.width), image_height=int(raw.image.height),
            )
            rollouts.append({
                "image_id": physical_image_id(raw), "example_id": str(raw.example_id),
                "seed": int(seed), "decode_mode": "sampled",
                "generated_token_ids": action_ids, "generated_token_ids_sha256": _sha256_json(action_ids),
                "generated_text": text, "stop_reason": stop_reason,
                "prompt_token_ids": ids, "prompt_token_ids_sha256": _sha256_json(ids),
                "observed_image_grid_thw": list(batch.image_grids[0]),
                "executed_media_sha256": batch.media_sha256[0],
                "predictions": parsed.to_artifact_dict(),
            })
    payload = {
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": "source256_native_raw_softmax_sampling",
        "config": {
            "infer_config_path": str(infer_config.resolve()), "resolved_fingerprint": resolved.fingerprint,
            "model_dtype": str(config.model.dtype), "device": device,
            "temperature": 1.0, "top_p": 1.0, "decode_mode": "sampled",
            "repetition_penalty": 1.0, "max_new_tokens": max_new_tokens,
            "seeds": list(seeds), "image_ids": sorted(prompt_meta),
            "raw_softmax": True, "top_k": 0,
            "generation_config_source": "fresh_transformers_generation_config", "use_model_defaults": False,
            "sampling_is_not_infer_config": True,
        },
        "native_execution": {"operation": "generate_continuations", "trace": "none",
                             "output_scores": False, "output_logits": False,
                             "allow_pad_tokens": True, "bank_body_omits_eos": True},
        "model_identity": identity, "prompt_metadata": prompt_meta,
        "rollout_count": len(rollouts), "rollouts": rollouts,
        "replay_check": {"enabled": replay_first_seed, "status": "passed" if replay_first_seed else "not_requested",
                         "checked_images": checked},
        "performance": {"elapsed_seconds_before_publication": time.monotonic() - started},
    }
    validate_artifact_payload(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return output


def main():
    from .prepare import BASE_SEEDS, PRODUCTION_MAX_NEW_TOKENS

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-ids")
    parser.add_argument("--seeds", default=",".join(map(str, BASE_SEEDS)))
    parser.add_argument("--max-new-tokens", type=int, default=PRODUCTION_MAX_NEW_TOKENS)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--replay-first-seed-check", action="store_true")
    args = parser.parse_args()
    run_sampling(infer_config=args.infer_config, output=args.output,
                 image_ids=_parse_image_ids(args.image_ids), seeds=parse_seed_list(args.seeds),
                 max_new_tokens=args.max_new_tokens, device=args.device,
                 replay_first_seed=args.replay_first_seed_check)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
