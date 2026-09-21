#!/usr/bin/env python3
"""Build one immutable Source256 CE or fresh K4-RLOO update plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


from src.config.inference import load_research_infer_config
from src.eval.assignment import global_matches as _global_matches
from .reward_rows import _gt_objects, _pred_objects
from .runtime import DEFAULT_CONFIG
from src.inference.inputs import PlannedExample, plan_examples
from .sample import (  # noqa: E402
    SCHEMA_VERSION as ROLLOUT_SCHEMA,
    validate_artifact_payload,
)
from src.adapters.dora import inspect_dora_adapter_payload  # noqa: E402
from src.qwen.special_token_embeddings import (  # noqa: E402
    inspect_special_token_embedding_delta_payload,
)


PLAN_SCHEMA = "source256_ce_rloo.update_plan.v1"
UPDATE_SCHEMA = "source256_ce_rloo.update_receipt.v1"
OPTIMIZER_SCHEMA = "source256_ce_rloo.adamw_state.v1"
WORLD_SIZE = 8
K = 4
EOS_TOKEN_ID = 151645
PRODUCTION_MAX_NEW_TOKENS = 3084
BASE_MODEL = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
SOURCE_ADAPTER = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
    "checkpoints/step-2444/adapter"
)
SOURCE_ADAPTER_TENSOR_SHA256 = (
    "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
)
SOURCE_EMBEDDING = SOURCE_ADAPTER.parent / "special_token_embeddings"
TRAIN256 = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-05-sft256-dev128-baseline/inputs-v3/train.jsonl"
)
TRAIN256_SHA256 = "05d505764d473daf9d7580abddd402de1d68f37e1e1e9b0db8674cd56c6a5db5"
SOURCE_INFER_CONFIG = DEFAULT_CONFIG
BASE_SEEDS = (2026090601, 2026090602, 2026090603, 2026090604)


def require(condition: Any, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"{path} must contain one JSON object")
    return value


def _source(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve(strict=True)
    return {"path": str(path), "sha256": file_sha256(path)}


def _expected_seeds(round_index: int) -> tuple[int, ...]:
    require(1 <= round_index <= 4, "round must be in 1..4")
    return tuple(seed + 100 * (round_index - 1) for seed in BASE_SEEDS)


def _sampling_max_new_tokens(
    *, arm: str, image_count: int, qualification_max_new_tokens: Any
) -> int:
    if qualification_max_new_tokens is None:
        return PRODUCTION_MAX_NEW_TOKENS
    require(arm == "rloo", "qualification max_new_tokens is only valid for RLOO")
    require(image_count == 8, "qualification max_new_tokens requires exactly 8 images")
    require(
        isinstance(qualification_max_new_tokens, int)
        and not isinstance(qualification_max_new_tokens, bool)
        and 1 <= qualification_max_new_tokens < PRODUCTION_MAX_NEW_TOKENS,
        f"qualification max_new_tokens must be an integer in 1..{PRODUCTION_MAX_NEW_TOKENS - 1}",
    )
    return qualification_max_new_tokens


def _action_token_ids(generated_token_ids: Any, stop_reason: Any) -> list[int]:
    require(isinstance(generated_token_ids, list), "generated_token_ids must be a list")
    require(
        all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in generated_token_ids),
        "generated_token_ids must contain non-negative integers",
    )
    require(EOS_TOKEN_ID not in generated_token_ids, "generated body unexpectedly contains im_end")
    require(stop_reason in {"im_end", "length"}, "unsupported stop_reason")
    action = list(generated_token_ids)
    if stop_reason == "im_end":
        action.append(EOS_TOKEN_ID)
    require(action, "length-stopped action cannot be empty")
    return action


def _rloo_advantages(rewards: Sequence[float]) -> list[float]:
    require(len(rewards) == K, "RLOO requires exactly K=4 rewards")
    require(all(math.isfinite(float(reward)) for reward in rewards), "RLOO rewards must be finite")
    total = sum(float(reward) for reward in rewards)
    values = [float(reward) - (total - float(reward)) / (K - 1) for reward in rewards]
    require(math.isclose(sum(values), 0.0, abs_tol=1e-12), "RLOO advantages are not zero-sum")
    return values


def _physical_image_id(raw: Any) -> str:
    source = raw.metadata.get("source", {})
    value = source.get("image_id") if isinstance(source, Mapping) else None
    return str(raw.example_id if value is None else value)


def _adapter_from_backend_receipt(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    backend = payload.get("model_identity")
    require(isinstance(backend, Mapping), "rollout is missing backend receipt")
    identity = backend.get("model_identity")
    require(isinstance(identity, Mapping), "rollout backend receipt is missing model identity")
    adapter = identity.get("adapter")
    require(isinstance(adapter, Mapping), "rollout model identity is missing adapter")
    return adapter


def _validate_rollout_policy(
    artifact: Mapping[str, Any],
    *,
    round_index: int,
    config_path: Path,
    config_fingerprint: str,
    current_adapter: Mapping[str, Any],
    max_new_tokens: int = PRODUCTION_MAX_NEW_TOKENS,
) -> None:
    validate_artifact_payload(artifact)
    require(artifact.get("schema_version") == ROLLOUT_SCHEMA, "unsupported rollout schema")
    config = artifact.get("config", {})
    expected = {
        "decode_mode": "sampled",
        "temperature": 1.0,
        "top_p": 1.0,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
        "seeds": list(_expected_seeds(round_index)),
        "model_dtype": "fp32",
        "resolved_fingerprint": config_fingerprint,
    }
    require(all(config.get(key) == value for key, value in expected.items()), "rollout sampling policy changed")
    require(
        config.get("raw_softmax") is True
        and isinstance(config.get("top_k"), int)
        and not isinstance(config.get("top_k"), bool)
        and config.get("top_k") == 0
        and config.get("generation_config_source") == "fresh_transformers_generation_config"
        and config.get("use_model_defaults") is False,
        "rollout policy is not explicit raw-softmax",
    )
    require(
        Path(str(config.get("infer_config_path"))).expanduser().resolve() == config_path,
        "rollout used a different inference config",
    )
    adapter = _adapter_from_backend_receipt(artifact)
    require(
        Path(str(adapter.get("adapter_path"))).expanduser().resolve()
        == Path(str(current_adapter["root"])).resolve()
        and adapter.get("merged_adapters") == []
        and adapter.get("adapter_state_evidence", {}).get("state_checked") is True,
        "rollout bank is stale or lacks unmerged adapter-state evidence",
    )
    backend = artifact["model_identity"]
    require(
        backend.get("effective_settings", {}).get("observed_attn_implementation") == "sdpa"
        and set(backend.get("effective_settings", {}).get("observed_model_dtype", {}).get("parameter_dtype_names", []))
        == {"torch.float32"},
        "rollout backend was not FP32/SDPA",
    )
    for row in artifact["rollouts"]:
        body = row["generated_token_ids"]
        require(len(body) <= max_new_tokens, "rollout body exceeds sampling max_new_tokens")
        if row["stop_reason"] == "length":
            require(
                len(body) == max_new_tokens,
                "length-stopped rollout did not reach sampling max_new_tokens",
            )
        else:
            require(len(body) < max_new_tokens, "im_end rollout reached the sampling cap")


def _validate_previous_round(
    path: Path | None,
    *,
    arm: str,
    round_index: int,
    current_adapter: Mapping[str, Any],
    embedding: Mapping[str, Any],
) -> dict[str, Any] | None:
    if round_index == 1:
        require(path is None, "round 1 must not have a previous receipt")
        return None
    require(path is not None, "round 2+ requires --previous-receipt")
    assert path is not None
    path = path.expanduser().resolve(strict=True)
    receipt = _load_json(path)
    require(
        receipt.get("schema_version") == UPDATE_SCHEMA
        and receipt.get("mechanical_status") == "MECHANICALLY_VALID"
        and receipt.get("arm") == arm
        and receipt.get("round") == round_index - 1,
        "previous receipt is not the immediately preceding own-arm round",
    )
    saved = receipt.get("saved_adapter", {})
    require(
        Path(str(saved.get("root"))).resolve() == Path(str(current_adapter["root"])).resolve()
        and saved.get("fingerprint") == current_adapter.get("fingerprint"),
        "current adapter does not match the previous own-arm receipt",
    )
    require(
        receipt.get("source_embedding", {}).get("fingerprint") == embedding.get("fingerprint"),
        "previous round changed the frozen Source embedding",
    )
    state_ref = receipt.get("optimizer_state", {})
    state_path = Path(str(state_ref.get("path"))).expanduser().resolve(strict=True)
    require(file_sha256(state_path) == state_ref.get("sha256"), "previous optimizer state changed")
    import torch

    state = torch.load(state_path, map_location="cpu", weights_only=True)
    require(
        isinstance(state, dict)
        and state.get("schema_version") == OPTIMIZER_SCHEMA
        and state.get("arm") == arm
        and state.get("round") == round_index - 1
        and state.get("saved_adapter_fingerprint") == current_adapter.get("fingerprint")
        and state.get("source_embedding_fingerprint") == embedding.get("fingerprint")
        and isinstance(state.get("optimizer_state_dict"), dict),
        "previous optimizer state binding is wrong or incomplete",
    )
    return {
        "receipt": {"path": str(path), "sha256": file_sha256(path)},
        "optimizer_state": {"path": str(state_path), "sha256": state_ref["sha256"]},
        "parameter_layout_sha256": state.get("parameter_layout_sha256"),
        "optimizer_step": state.get("optimizer_step"),
    }


def _native_ce_group(raw: Any, *, index: int, config: Any, frontend: Any) -> dict[str, Any]:
    planned, = plan_examples(
        [raw], config=config, components=frontend.qwen, row_indices=[index], target_max_length=12000,
    )
    return _native_ce_group_from_plan(raw, index=index, planned=planned)


def _native_ce_group_from_plan(raw: Any, *, index: int, planned: PlannedExample) -> dict[str, Any]:
    encoded, prompt = planned.target, planned.prompt
    require(encoded is not None, "native CE plan is missing its annotated target")
    require(
        encoded.example_id == prompt.example_id == str(raw.example_id)
        and prompt.row_index == index
        and Path(planned.image.image_path).resolve() == raw.image.path.resolve(),
        "native CE planned row identity changed",
    )
    spans = encoded.supervised_token_spans
    ignored = encoded.ignored_token_spans
    require(spans and ignored, f"native CE spans are incomplete: {raw.example_id}")
    start, end = spans[0].physical_token_start, spans[-1].physical_token_end
    require(start == len(prompt.expected_executed_prompt_token_ids), "native CE prompt/action boundary changed")
    require(
        list(encoded.input_ids[:start]) == prompt.expected_executed_prompt_token_ids,
        "native CE prefix differs from build_prompt_record",
    )
    require(
        all(left.physical_token_end == right.physical_token_start for left, right in zip(spans, spans[1:])),
        "native CE supervised spans are not contiguous",
    )
    require(
        ignored[0].physical_token_start == end
        and all(span.physical_token_start >= end for span in ignored),
        "native ignored post-EOS tokens entered the CE target",
    )
    action = [int(token) for token in encoded.input_ids[start:end]]
    require(action and action[-1] == EOS_TOKEN_ID, "native CE action does not end at im_end")
    image = encoded.image_encoding.to_artifact_dict()
    return {
        "image_id": _physical_image_id(raw),
        "example_id": str(raw.example_id),
        "source_order": index,
        "rank": None,
        "prompt_token_ids": list(prompt.expected_executed_prompt_token_ids),
        "prompt_token_ids_sha256": json_sha256(prompt.expected_executed_prompt_token_ids),
        "chat_text_sha256": hashlib.sha256(prompt.chat_text.encode()).hexdigest(),
        "image_path": image["image_path"],
        "image_content_sha256": image["image_content_sha256"],
        "expected_image_grid_thw": image["image_grid_thw"],
        "media_identity_sha256": json_sha256({
            "image_content_sha256": image["image_content_sha256"],
            "image_grid_thw": image["image_grid_thw"],
            "logical_transform_id": image["logical_transform_id"],
        }),
        "actions": [{
            "kind": "native_ce",
            "action_token_ids": action,
            "action_token_ids_sha256": json_sha256(action),
            "action_token_count": len(action),
            "terminal_eos_included": True,
            "native_supervised_span_count": len(spans),
            "native_ignored_post_eos_token_count": sum(span.token_count for span in ignored),
        }],
    }


def _reward_for_rollout(raw: Any, rollout: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    predictions = rollout.get("predictions")
    require(isinstance(predictions, Mapping), "rollout predictions evidence is malformed")
    valid_rows = predictions.get("predictions")
    dropped_rows = predictions.get("dropped_predictions")
    require(
        isinstance(valid_rows, list)
        and isinstance(dropped_rows, list)
        and predictions.get("valid_prediction_count") == len(valid_rows)
        and predictions.get("dropped_prediction_count") == len(dropped_rows),
        "rollout parser counts or row lists are corrupt",
    )
    match_row = {
        "gt": [
            {"description": obj.description, "bbox": list(obj.bbox)}
            for obj in raw.objects
        ],
        "pred": valid_rows,
        "image_width": int(raw.image.width),
        "image_height": int(raw.image.height),
    }
    gt = _gt_objects(match_row, row_id=str(raw.example_id))
    pred, invalid = _pred_objects(match_row)
    require(gt, f"image has no annotated owners: {raw.example_id}")
    matches = _global_matches(gt, pred, 0.50)
    refs = [str(raw.objects[gt_index].object_id) for gt_index, _, _ in matches]
    return len(matches) / len(gt), {
        "annotated_owner_count": len(gt),
        "valid_prediction_count": len(pred),
        "invalid_prediction_count": invalid,
        "parser_dropped_prediction_count": len(dropped_rows),
        "matched_owner_count": len(matches),
        "matched_owner_refs": refs,
    }


def _rloo_groups(
    raw_examples: Sequence[Any],
    artifacts: Sequence[tuple[Path, Mapping[str, Any]]],
    *,
    round_index: int,
) -> list[dict[str, Any]]:
    seeds = _expected_seeds(round_index)
    raw_by_example = {str(raw.example_id): raw for raw in raw_examples}
    cells: dict[tuple[str, int], tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
    for _, artifact in artifacts:
        metadata = artifact.get("prompt_metadata", {})
        require(isinstance(metadata, Mapping), "rollout prompt_metadata is missing")
        for row in artifact.get("rollouts", []):
            example_id = str(row.get("example_id"))
            seed = row.get("seed")
            require(example_id in raw_by_example, f"rollout contains an unknown example: {example_id}")
            key = (example_id, seed)
            require(seed in seeds and key not in cells, f"duplicate or unexpected rollout cell: {key}")
            meta = metadata.get(example_id)
            require(isinstance(meta, Mapping), f"rollout prompt metadata missing: {key}")
            cells[key] = (row, meta)
    expected = {(str(raw.example_id), seed) for raw in raw_examples for seed in seeds}
    require(set(cells) == expected, "rollout bank has missing or extra image/seed cells")

    groups: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_examples):
        actions: list[dict[str, Any]] = []
        rewards: list[float] = []
        prompt_hashes: set[str] = set()
        media_hashes: set[str] = set()
        image_hashes: set[str] = set()
        grids: set[str] = set()
        prompt_ids: list[int] | None = None
        chat_hash: str | None = None
        observed_grid: Any = None
        for seed in seeds:
            row, meta = cells[(str(raw.example_id), seed)]
            body = row.get("generated_token_ids")
            action = _action_token_ids(body, row.get("stop_reason"))
            require(json_sha256(body) == row.get("generated_token_ids_sha256"), "generated body hash changed")
            ids = row.get("prompt_token_ids")
            require(
                isinstance(ids, list)
                and ids
                and all(isinstance(token, int) and not isinstance(token, bool) and token >= 0 for token in ids)
                and ids == meta.get("prompt_token_ids")
                and json_sha256(ids) == row.get("prompt_token_ids_sha256") == meta.get("prompt_token_ids_sha256"),
                "rollout prompt token identity changed",
            )
            require(
                str(Path(str(meta.get("image_path"))).resolve()) == str(raw.image.path.resolve())
                and file_sha256(raw.image.path) == meta.get("image_sha256"),
                "rollout image identity changed",
            )
            require(
                meta.get("width") == raw.image.width and meta.get("height") == raw.image.height,
                "rollout image dimensions changed",
            )
            media_sha = row.get("executed_media_sha256")
            observed_grid_value = row.get("observed_image_grid_thw")
            require(isinstance(media_sha, str) and len(media_sha) == 64, "rollout executed-media identity is missing")
            require(
                isinstance(observed_grid_value, list)
                and len(observed_grid_value) == 3
                and all(isinstance(value, int) and value > 0 for value in observed_grid_value),
                "rollout observed image grid is malformed",
            )
            reward, matching = _reward_for_rollout(raw, row)
            rewards.append(reward)
            prompt_hashes.add(str(row["prompt_token_ids_sha256"]))
            media_hashes.add(media_sha)
            image_hashes.add(str(meta["image_sha256"]))
            grids.add(json_sha256(observed_grid_value))
            observed_grid = observed_grid_value
            prompt_ids = list(ids)
            chat_hash = str(meta.get("chat_text_sha256"))
            actions.append({
                "kind": "sampled_trajectory",
                "seed": seed,
                "generated_token_ids": list(body),
                "generated_token_ids_sha256": row["generated_token_ids_sha256"],
                "generated_text_sha256": hashlib.sha256(str(row.get("generated_text", "")).encode()).hexdigest(),
                "stop_reason": row["stop_reason"],
                "action_token_ids": action,
                "action_token_ids_sha256": json_sha256(action),
                "action_token_count": len(action),
                "terminal_eos_included": row["stop_reason"] == "im_end",
                "reward": reward,
                "matching": matching,
                "parser_evidence_sha256": json_sha256(row["predictions"]),
            })
        require(
            len(prompt_hashes) == len(media_hashes) == len(image_hashes) == len(grids) == 1
            and all(media_hashes),
            f"within-image prompt/media identity varies: {raw.example_id}",
        )
        advantages = _rloo_advantages(rewards)
        for action, advantage in zip(actions, advantages, strict=True):
            action["advantage"] = advantage
        groups.append({
            "image_id": _physical_image_id(raw),
            "example_id": str(raw.example_id),
            "source_order": index,
            "rank": None,
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": next(iter(prompt_hashes)),
            "chat_text_sha256": chat_hash,
            "image_path": str(raw.image.path),
            "image_content_sha256": next(iter(image_hashes)),
            "observed_image_grid_thw": observed_grid,
            "executed_media_sha256": next(iter(media_hashes)),
            "annotated_owner_count": len(raw.objects),
            "reward_values": rewards,
            "semantic_flat": all(value == rewards[0] for value in rewards),
            "actions": actions,
        })
    return groups


def _assign(groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    require(len(groups) >= WORLD_SIZE and len(groups) % WORLD_SIZE == 0, "image count must be >=8 and divisible by 8")
    per_rank = len(groups) // WORLD_SIZE
    assignments = []
    for rank in range(WORLD_SIZE):
        selected = groups[rank * per_rank:(rank + 1) * per_rank]
        for group in selected:
            group["rank"] = rank
        assignments.append({
            "rank": rank,
            "image_ids": [group["image_id"] for group in selected],
            "example_ids": [group["example_id"] for group in selected],
            "image_count": len(selected),
            "action_count": sum(len(group["actions"]) for group in selected),
        })
    return assignments


def _plan_content(plan: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if key != "content_sha256"}


def validate_plan(plan_or_path: Mapping[str, Any] | Path, *, verify_sources: bool = True) -> dict[str, Any]:
    plan = _load_json(plan_or_path.expanduser().resolve(strict=True)) if isinstance(plan_or_path, Path) else dict(plan_or_path)
    require(plan.get("schema_version") == PLAN_SCHEMA, "unsupported update-plan schema")
    arm, round_index = plan.get("arm"), plan.get("round")
    require(arm in {"ce", "rloo"} and round_index in range(1, 5), "invalid arm or round")
    population = plan.get("population", {})
    groups = population.get("groups", [])
    count = population.get("image_count")
    require(isinstance(groups, list) and len(groups) == count, "plan population count changed")
    require(population.get("k") == K and population.get("world_size") == WORLD_SIZE, "plan K/world size changed")
    require(population.get("mode") == ("production" if count == 256 else "qualification"), "plan population mode changed")
    require(population.get("images_per_rank") == count // WORLD_SIZE, "images-per-rank changed")
    if count == 256:
        require(population.get("images_per_rank") == 32, "production plan is not 8 ranks x 32 images")
    if arm == "rloo":
        sampling = plan.get("sampling")
        require(isinstance(sampling, Mapping), "RLOO plan sampling metadata is missing")
        max_new_tokens = _sampling_max_new_tokens(
            arm=arm,
            image_count=count,
            qualification_max_new_tokens=sampling.get("qualification_max_new_tokens"),
        )
        require(
            sampling.get("max_new_tokens") == max_new_tokens,
            "RLOO sampling max_new_tokens is not bound to the plan contract",
        )
    require([group.get("source_order") for group in groups] == list(range(count)), "source order changed")
    expected_groups = [dict(group) for group in groups]
    expected_assignments = _assign(expected_groups)
    require(expected_assignments == population.get("assignments"), "rank assignments changed")
    require(
        [group.get("rank") for group in groups] == [group["rank"] for group in expected_groups],
        "group rank assignment changed",
    )
    expected_actions = 1 if arm == "ce" else K
    for group in groups:
        actions = group.get("actions", [])
        require(len(actions) == expected_actions, "plan action population changed")
        require(json_sha256(group.get("prompt_token_ids")) == group.get("prompt_token_ids_sha256"), "plan prompt hash changed")
        for action in actions:
            ids = action.get("action_token_ids")
            require(isinstance(ids, list) and ids and json_sha256(ids) == action.get("action_token_ids_sha256"), "plan action hash changed")
            require(action.get("action_token_count") == len(ids), "plan action count changed")
        if arm == "ce":
            require(actions[0].get("terminal_eos_included") is True and actions[0]["action_token_ids"][-1] == EOS_TOKEN_ID, "CE lost terminal EOS")
        else:
            require([action.get("seed") for action in actions] == list(_expected_seeds(round_index)), "RLOO seeds changed")
            expected_advantages = _rloo_advantages([float(action.get("reward")) for action in actions])
            require(
                all(math.isclose(float(action.get("advantage")), value, abs_tol=1e-15) for action, value in zip(actions, expected_advantages, strict=True)),
                "RLOO advantages changed",
            )
    require(json_sha256(_plan_content(plan)) == plan.get("content_sha256"), "plan content hash changed")
    if verify_sources:
        for name in ("train_jsonl", "infer_config"):
            source = plan.get("sources", {}).get(name, {})
            require(file_sha256(Path(source["path"])) == source.get("sha256"), f"plan source changed: {name}")
        for source in plan.get("sources", {}).get("rollout_artifacts", []):
            require(file_sha256(Path(source["path"])) == source.get("sha256"), "rollout bank changed")
        lineage = plan.get("lineage")
        if lineage:
            for name in ("receipt", "optimizer_state"):
                source = lineage[name]
                require(file_sha256(Path(source["path"])) == source.get("sha256"), f"previous {name} changed")
        adapter = inspect_dora_adapter_payload(plan["model"]["current_adapter"]["root"], BASE_MODEL)
        embedding = inspect_special_token_embedding_delta_payload(plan["model"]["source_embedding"]["root"], BASE_MODEL)
        require(
            plan["model"].get("base_model_path") == str(BASE_MODEL.resolve())
            and adapter["fingerprint"] == plan["model"]["current_adapter"]["fingerprint"],
            "current base or adapter changed",
        )
        require(
            Path(embedding["root"]).resolve() == SOURCE_EMBEDDING.resolve()
            and embedding["fingerprint"] == plan["model"]["source_embedding"]["fingerprint"],
            "Source embedding changed",
        )
        if round_index == 1:
            require(
                plan.get("lineage") is None
                and Path(adapter["root"]).resolve() == SOURCE_ADAPTER.resolve()
                and file_sha256(SOURCE_ADAPTER / "adapter_model.safetensors") == SOURCE_ADAPTER_TENSOR_SHA256,
                "round 1 is not Source-started",
            )
        else:
            lineage = plan.get("lineage")
            require(isinstance(lineage, Mapping), "round 2+ is missing lineage")
            expected_lineage = _validate_previous_round(
                Path(lineage["receipt"]["path"]),
                arm=arm,
                round_index=round_index,
                current_adapter=adapter,
                embedding=embedding,
            )
            require(expected_lineage == lineage, "round lineage changed")
        if arm == "rloo":
            artifacts = [_load_json(Path(source["path"])) for source in plan["sources"]["rollout_artifacts"]]
            for artifact in artifacts:
                _validate_rollout_policy(
                    artifact,
                    round_index=round_index,
                    config_path=Path(plan["sources"]["infer_config"]["path"]),
                    config_fingerprint=plan["sources"]["infer_config"]["resolved_fingerprint"],
                    current_adapter=adapter,
                    max_new_tokens=max_new_tokens,
                )
            require(
                {json_sha256(artifact["model_identity"]) for artifact in artifacts}
                == {plan["sampling"]["backend_receipt_sha256"]},
                "rollout backend snapshot changed",
            )
    return plan


def prepare_plan(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json as config_json_sha256
    from src.data import load_raw_examples
    from src.inference.runtime import assemble_frontend

    arm, round_index = args.arm, int(args.round)
    expected_count = int(args.expected_image_count)
    require(expected_count == 256 or expected_count >= 8, "qualification requires at least 8 images")
    require(expected_count % WORLD_SIZE == 0, "image count must be divisible by 8")
    qualification_max_new_tokens = getattr(args, "qualification_max_new_tokens", None)
    sampling_max_new_tokens = _sampling_max_new_tokens(
        arm=arm,
        image_count=expected_count,
        qualification_max_new_tokens=qualification_max_new_tokens,
    )
    train_jsonl = args.train_jsonl.expanduser().resolve(strict=True)
    if expected_count == 256:
        require(train_jsonl == TRAIN256.resolve() and file_sha256(train_jsonl) == TRAIN256_SHA256, "production train256 identity changed")
    config_path = args.infer_config.expanduser().resolve(strict=True)
    resolved = load_research_infer_config(config_path)
    config = resolved.config
    require(Path(config.data.input_jsonl).resolve() == train_jsonl, "inference config uses a different train JSONL")
    require(
        config.backend.type == "hf"
        and config.backend.hf.attn_implementation == "sdpa"
        and config.model.dtype == "fp32"
        and Path(config.model.base_model).resolve() == BASE_MODEL.resolve(),
        "plan requires exact HF FP32/SDPA Source base",
    )
    require(config.adapter is not None and config.embedding_delta is not None, "plan requires adapter plus Source embedding")
    current_adapter = inspect_dora_adapter_payload(config.adapter.path, BASE_MODEL)
    embedding = inspect_special_token_embedding_delta_payload(config.embedding_delta.path, BASE_MODEL)
    require(Path(embedding["root"]).resolve() == SOURCE_EMBEDDING.resolve(), "Source embedding path changed")
    if round_index == 1:
        require(Path(current_adapter["root"]).resolve() == SOURCE_ADAPTER.resolve(), "round 1 must start from Source adapter")
        require(file_sha256(SOURCE_ADAPTER / "adapter_model.safetensors") == SOURCE_ADAPTER_TENSOR_SHA256, "Source adapter tensor changed")
    lineage = _validate_previous_round(
        args.previous_receipt,
        arm=arm,
        round_index=round_index,
        current_adapter=current_adapter,
        embedding=embedding,
    )
    raw_examples = list(load_raw_examples(train_jsonl))
    require(len(raw_examples) == expected_count, "input JSONL image count differs from --expected-image-count")
    require(len({_physical_image_id(raw) for raw in raw_examples}) == expected_count, "input contains duplicate physical image IDs")

    rollout_paths = [path.expanduser().resolve(strict=True) for path in args.rollout_artifact]
    sources = {
        "train_jsonl": _source(train_jsonl),
        "infer_config": {**_source(config_path), "resolved_fingerprint": resolved.fingerprint},
        "rollout_artifacts": [_source(path) for path in rollout_paths],
    }
    if arm == "ce":
        require(not rollout_paths, "CE plan must not consume rollout artifacts")
        frontend = assemble_frontend(
            config,
            generation_config_fingerprint=config_json_sha256(config.generation.model_dump(mode="json")),
        )
        groups = [_native_ce_group(raw, index=index, config=config, frontend=frontend) for index, raw in enumerate(raw_examples)]
    else:
        require(rollout_paths, "RLOO plan requires at least one --rollout-artifact")
        artifacts = [(path, _load_json(path)) for path in rollout_paths]
        for _, artifact in artifacts:
            _validate_rollout_policy(
                artifact,
                round_index=round_index,
                config_path=config_path,
                config_fingerprint=resolved.fingerprint,
                current_adapter=current_adapter,
                max_new_tokens=sampling_max_new_tokens,
            )
        backend_hashes = {json_sha256(artifact["model_identity"]) for _, artifact in artifacts}
        require(len(backend_hashes) == 1, "rollout shards used different backend identities")
        groups = _rloo_groups(raw_examples, artifacts, round_index=round_index)
    assignments = _assign(groups)
    objective = {
        "optimizer": {"name": "AdamW", "learning_rate": 2.5e-6, "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0.0, "max_grad_norm": 1.0, "scheduler": "constant"},
        "formula": (
            "(1/N) * sum_image mean_native_GT_token_NLL_including_EOS"
            if arm == "ce"
            else "-(1/(N*K)) * sum_image,k advantage * sum_action_token_logprob"
        ),
        "image_denominator": expected_count,
        "k": K,
        "token_length_normalization": arm == "ce",
        "rloo": {"reward": "TP50/GTcount", "advantage": "r_k-mean(other3)", "ppo": False, "kl": False, "critic": False, "whitening": False},
    }
    plan: dict[str, Any] = {
        "schema_version": PLAN_SCHEMA,
        "arm": arm,
        "round": round_index,
        "sources": sources,
        "model": {
            "base_model_path": str(BASE_MODEL.resolve()),
            "current_adapter": current_adapter,
            "source_embedding": embedding,
            "dtype": "fp32",
            "attention_implementation": "sdpa",
        },
        "lineage": lineage,
        "population": {
            "mode": "production" if expected_count == 256 else "qualification",
            "image_count": expected_count,
            "world_size": WORLD_SIZE,
            "images_per_rank": expected_count // WORLD_SIZE,
            "k": K,
            "actions_per_image": 1 if arm == "ce" else K,
            "groups": groups,
            "assignments": assignments,
        },
        "objective": objective,
        "sampling": None if arm == "ce" else {
            "seeds": list(_expected_seeds(round_index)),
            "temperature": 1.0,
            "top_p": 1.0,
            "repetition_penalty": 1.0,
            "max_new_tokens": sampling_max_new_tokens,
            "qualification_max_new_tokens": qualification_max_new_tokens,
            "terminal_token_id": EOS_TOKEN_ID,
            "terminal_policy": "append_only_for_observed_im_end",
            "raw_softmax": True,
            "top_k": 0,
            "generation_config_source": "fresh_transformers_generation_config",
            "use_model_defaults": False,
            "backend_receipt_sha256": next(iter(backend_hashes)),
        },
        "unsupported": ["PPO", "KL", "critic", "advantage_whitening", "RLOO_token_length_normalization", "merged_adapter", "mid_round_resume"],
    }
    plan["content_sha256"] = json_sha256(plan)
    validate_plan(plan)
    return plan


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--arm", choices=("ce", "rloo"), required=True)
    result.add_argument("--round", type=int, choices=range(1, 5), required=True)
    result.add_argument("--infer-config", type=Path, default=SOURCE_INFER_CONFIG)
    result.add_argument("--train-jsonl", type=Path, default=TRAIN256)
    result.add_argument("--expected-image-count", type=int, default=256)
    result.add_argument(
        "--qualification-max-new-tokens",
        type=int,
        help="Opt-in sampled RLOO cap; only valid for an exactly 8-image qualification.",
    )
    result.add_argument("--rollout-artifact", type=Path, action="append", default=[])
    result.add_argument("--previous-receipt", type=Path)
    result.add_argument("--output", type=Path, required=True)
    return result


def main() -> int:
    args = parser().parse_args()
    output = args.output.expanduser().resolve()
    plan = prepare_plan(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    require(not output.exists(), f"refusing to overwrite plan: {output}")
    temp = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    require(not temp.exists(), f"temporary plan path exists: {temp}")
    try:
        temp.write_text(json.dumps(plan, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
        validate_plan(temp)
        os.link(temp, output)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({"mechanical_status": "MECHANICALLY_VALID", "plan": str(output), "sha256": file_sha256(output), "content_sha256": plan["content_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
