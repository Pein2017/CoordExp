#!/usr/bin/env python3
"""Collect one greedy plus sixteen sampled trajectories with offline vLLM.

This is an experiment-local collector for the dense-enumeration trajectory
panel.  One process owns one visible GPU and one vLLM engine.  Images are
sharded across workers; vLLM owns scheduling within each worker.  Completed
image batches are written as immutable greedy/sampled artifact pairs so a
long panel does not live only in process memory.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.run_current_seeded_sampled_rollouts import (  # noqa: E402
    _build_requests,
    physical_image_id,
    select_examples,
)


DEFAULT_CONFIG = Path(
    "configs/coordexp_swift/infer/research/"
    "qwen3_vl_2b_step4887_candidate_pool_2432_vllm_panel.yaml"
)
SAMPLED_COUNT = 16
SCHEMA_VERSION = "coordexp_vllm_trajectory_panel.v2"


def shard_examples(
    examples: Sequence[Any], *, worker_index: int, worker_count: int
) -> list[Any]:
    """Return the stable image shard owned by one worker."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    if not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    return list(examples[worker_index::worker_count])


def _chunks(values: Sequence[Any], size: int) -> list[list[Any]]:
    if size <= 0:
        raise ValueError("image_batch_size must be positive")
    return [list(values[offset : offset + size]) for offset in range(0, len(values), size)]


def parse_image_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    image_ids = {piece.strip() for piece in value.split(",") if piece.strip()}
    if not image_ids:
        raise ValueError("image_ids must contain at least one non-empty id")
    return image_ids


def sampling_params_kwargs(
    *,
    decode_mode: str,
    sample_count: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    stop_token_id: int,
    seed: int | None,
) -> dict[str, Any]:
    """Build the experiment-local vLLM sampling policy."""

    if decode_mode not in {"greedy", "sampled"}:
        raise ValueError("decode_mode must be greedy or sampled")
    if sample_count != SAMPLED_COUNT:
        raise ValueError(f"sample_count must be exactly {SAMPLED_COUNT}")
    if repetition_penalty != 1.0:
        raise ValueError("trajectory panel requires repetition_penalty=1.0")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if decode_mode == "greedy":
        if temperature != 0.0:
            raise ValueError("greedy temperature must be zero")
        kwargs: dict[str, Any] = {
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
        }
    else:
        if temperature != 0.4 or top_p != 0.95:
            raise ValueError("sampled panel requires temperature=0.4 and top_p=0.95")
        kwargs = {
            "n": sample_count,
            "temperature": temperature,
            "top_p": top_p,
        }
    kwargs.update(
        {
            "top_k": 0,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_new_tokens,
            "stop_token_ids": [int(stop_token_id)],
            "ignore_eos": False,
            "detokenize": True,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": True,
        }
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    return kwargs


def _sampling_params(**kwargs: Any) -> Any:
    from vllm import SamplingParams

    return SamplingParams(**sampling_params_kwargs(**kwargs))


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ordered_completions(native_output: Any, *, expected_count: int) -> list[Any]:
    completions = getattr(native_output, "outputs", None)
    if not isinstance(completions, Sequence) or len(completions) != expected_count:
        raise RuntimeError(
            f"vLLM returned {0 if completions is None else len(completions)} completions; "
            f"expected {expected_count}"
        )
    try:
        ordered = sorted(completions, key=lambda item: int(item.index))
        indices = [int(item.index) for item in ordered]
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("vLLM completions lack integer sample indices") from exc
    if indices != list(range(expected_count)):
        raise RuntimeError(f"vLLM completion indices are not 0..{expected_count - 1}: {indices}")
    return ordered


def _materialize_completion(
    *,
    completion: Any,
    tokenizer: Any,
    stop_token_id: int,
) -> tuple[list[int], str, str]:
    token_ids = [int(value) for value in getattr(completion, "token_ids", ())]
    if not token_ids:
        raise RuntimeError("vLLM returned an empty completion")
    stop_positions = [index for index, value in enumerate(token_ids) if value == stop_token_id]
    finish_reason = str(getattr(completion, "finish_reason", "") or "")
    if stop_positions:
        if stop_positions != [len(token_ids) - 1] or finish_reason != "stop":
            raise RuntimeError("vLLM returned inconsistent terminal stop-token evidence")
        parser_ids = token_ids[:-1]
        stop_reason = "im_end"
    else:
        if finish_reason != "length":
            raise RuntimeError("vLLM completion has neither a terminal stop nor length finish")
        parser_ids = token_ids
        stop_reason = "length"
    text = str(
        tokenizer.decode(
            parser_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    )
    return token_ids, text, stop_reason


def _artifact_config(
    *,
    infer_config: Path,
    resolved_fingerprint: str,
    model_dtype: str,
    decode_mode: str,
    max_new_tokens: int,
    request_seed: int | None,
    worker_index: int,
    worker_count: int,
    image_batch_size: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "infer_config_path": str(infer_config.resolve()),
        "resolved_fingerprint": resolved_fingerprint,
        "model_dtype": model_dtype,
        "backend": "vllm",
        "decode_mode": decode_mode,
        "temperature": 0.0 if decode_mode == "greedy" else 0.4,
        "top_p": 1.0 if decode_mode == "greedy" else 0.95,
        "repetition_penalty": 1.0,
        "max_new_tokens": max_new_tokens,
        "sample_count": 1 if decode_mode == "greedy" else SAMPLED_COUNT,
        "sample_index_range": None if decode_mode == "greedy" else [0, SAMPLED_COUNT - 1],
        "sampling_is_not_infer_config": True,
        "sampling_order": "request_major",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "image_batch_size": image_batch_size,
        "max_num_seqs": 32,
    }
    if request_seed is not None:
        config["request_seed"] = int(request_seed)
    return config


def build_artifact(
    *,
    config: Mapping[str, Any],
    model_identity: Mapping[str, Any],
    prompt_metadata: Mapping[str, Any],
    rollouts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build and validate one immutable batch artifact."""

    mode = config.get("decode_mode")
    if mode not in {"greedy", "sampled"}:
        raise ValueError("artifact decode_mode is required")
    rows = [dict(row) for row in rollouts]
    if not rows:
        raise ValueError("artifact must contain rollout rows")
    by_example: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("decode_mode") != mode:
            raise ValueError("rollout mode differs from artifact mode")
        example_id = str(row.get("example_id", ""))
        if not example_id or example_id not in prompt_metadata:
            raise ValueError("rollout lacks matching prompt metadata")
        metadata = prompt_metadata[example_id]
        if not isinstance(metadata, Mapping):
            raise ValueError("rollout prompt metadata is not an object")
        if row.get("source_image_file_sha256") != metadata.get(
            "source_image_file_sha256"
        ):
            raise ValueError("rollout source image identity differs from prompt metadata")
        executed_rgb_sha256 = row.get("executed_rgb_sha256")
        if not isinstance(executed_rgb_sha256, str) or len(executed_rgb_sha256) != 64:
            raise ValueError("rollout lacks executed RGB image identity")
        by_example.setdefault(example_id, []).append(row)
    expected_per_image = 1 if mode == "greedy" else SAMPLED_COUNT
    for example_id, image_rows in by_example.items():
        if len(image_rows) != expected_per_image:
            raise ValueError(f"{example_id} has {len(image_rows)} {mode} rows")
        if mode == "sampled" and [row.get("sample_index") for row in image_rows] != list(
            range(SAMPLED_COUNT)
        ):
            raise ValueError(f"{example_id} sampled rows are not sample_index 0..15")
    if set(by_example) != set(prompt_metadata):
        raise ValueError("prompt metadata does not cover exact artifact images")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_mode": "experiment_local_vllm_trajectory_panel",
        "config": dict(config),
        "model_identity": dict(model_identity),
        "prompt_metadata": dict(prompt_metadata),
        "rollout_count": len(rows),
        "rollouts": rows,
    }


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _run_native_generation(
    *,
    session: Any,
    requests: Sequence[Any],
    decode_mode: str,
    max_new_tokens: int,
    request_seed: int | None,
) -> tuple[tuple[Any, ...], tuple[str, ...], float]:
    from src.inference.vllm_backend import (
        _close_prompt_images,
        _restore_native_request_order,
    )

    prompts, media_hashes = session._generation_prompts(requests)
    started_at = time.perf_counter()
    try:
        outputs = session._engine.generate(
            prompts,
            _sampling_params(
                decode_mode=decode_mode,
                sample_count=SAMPLED_COUNT,
                temperature=0.0 if decode_mode == "greedy" else 0.4,
                top_p=1.0 if decode_mode == "greedy" else 0.95,
                repetition_penalty=1.0,
                max_new_tokens=max_new_tokens,
                stop_token_id=session._im_end_token_id(),
                seed=request_seed,
            ),
            use_tqdm=False,
        )
    finally:
        _close_prompt_images(prompts)
    elapsed_seconds = time.perf_counter() - started_at
    return (
        _restore_native_request_order(outputs, len(requests)),
        media_hashes,
        elapsed_seconds,
    )


def _record_research_live_decode(
    *,
    session: Any,
    requests: Sequence[Any],
    rows: Sequence[Mapping[str, Any]],
    decode_mode: str,
) -> None:
    """Bind experiment-local multi-completion evidence to live preflight."""

    settings = dict(session.receipt.effective_settings)
    value = settings.get("runtime_preflight")
    if not isinstance(value, Mapping):
        raise RuntimeError("vLLM session lacks upstream runtime preflight evidence")
    preflight = dict(value)
    if preflight.get("status") not in {
        "ready_for_engine_construction",
        "passed_live_decode",
    }:
        raise RuntimeError(
            f"unexpected vLLM runtime preflight status: {preflight.get('status')}"
        )
    previous = preflight.get("research_panel_live_decode")
    evidence = (
        dict(previous)
        if isinstance(previous, Mapping)
        else {"scope": "experiment_local_multi_completion", "calls": []}
    )
    calls = list(evidence.get("calls") or [])
    calls.append(
        {
            "decode_mode": decode_mode,
            "request_count": len(requests),
            "completion_count": len(rows),
            "generated_token_count": sum(
                len(row.get("generated_token_ids", ())) for row in rows
            ),
            "length_finished_completion_count": sum(
                row.get("stop_reason") == "length" for row in rows
            ),
        }
    )
    evidence["calls"] = calls
    evidence["request_count"] = sum(int(call["request_count"]) for call in calls)
    evidence["completion_count"] = sum(
        int(call["completion_count"]) for call in calls
    )
    evidence["generated_token_count"] = sum(
        int(call["generated_token_count"]) for call in calls
    )
    preflight["status"] = "passed_live_decode"
    preflight["live_decode"] = {
        "scope": "experiment_local_multi_completion",
        "first_decode_mode": calls[0]["decode_mode"],
        "first_request_count": calls[0]["request_count"],
        "first_completion_count": calls[0]["completion_count"],
    }
    preflight["research_panel_live_decode"] = evidence
    settings["runtime_preflight"] = preflight
    session._receipt = replace(session.receipt, effective_settings=settings)


def _generation_health(
    rows: Sequence[Mapping[str, Any]], *, elapsed_seconds: float
) -> dict[str, Any]:
    generated_token_count = sum(len(row.get("generated_token_ids", ())) for row in rows)
    completion_count = len(rows)
    stop_counts = {
        reason: sum(row.get("stop_reason") == reason for row in rows)
        for reason in ("im_end", "length")
    }
    parser_status_counts: dict[str, int] = {}
    for row in rows:
        predictions = row.get("predictions")
        status = (
            str(predictions.get("parse_status", "unknown"))
            if isinstance(predictions, Mapping)
            else "unknown"
        )
        parser_status_counts[status] = parser_status_counts.get(status, 0) + 1
    return {
        "completion_count": completion_count,
        "generated_token_count": generated_token_count,
        "stop_reason_counts": stop_counts,
        "natural_closure_count": stop_counts["im_end"],
        "parser_status_counts": parser_status_counts,
        "elapsed_seconds": elapsed_seconds,
        "completions_per_second": completion_count / elapsed_seconds,
        "routes_per_second": completion_count / elapsed_seconds,
        "generated_tokens_per_second": generated_token_count / elapsed_seconds,
    }


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return dict(value)


def _stable_session_identity(receipt: Any) -> dict[str, Any]:
    """Remove rank-local observations from cross-process artifact identity."""

    payload = dict(receipt.to_artifact_dict())
    settings_value = payload.get("effective_settings")
    if not isinstance(settings_value, Mapping):
        raise RuntimeError("backend receipt lacks effective settings")
    settings = dict(settings_value)
    settings.pop("runtime_preflight", None)
    settings.pop("performance", None)
    payload["effective_settings"] = settings
    return payload


def _validated_resume_batch(
    *,
    worker_root: Path,
    entry: Mapping[str, Any],
    expected_batch_index: int,
    expected_examples: Sequence[Any],
    resolved_fingerprint: str,
    model_identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Return a fail-closed completed batch entry, otherwise regenerate it."""

    try:
        if entry.get("batch_index") != expected_batch_index:
            return None
        expected_ids = [physical_image_id(example) for example in expected_examples]
        if entry.get("image_ids") != expected_ids:
            return None
        parts = entry.get("artifacts")
        if not isinstance(parts, Mapping) or set(parts) != {"greedy", "sampled"}:
            return None
        recomputed_health: dict[str, Any] = {}
        for mode in ("greedy", "sampled"):
            part = parts[mode]
            if not isinstance(part, Mapping):
                return None
            name = part.get("path")
            expected_hash = part.get("sha256")
            if not isinstance(name, str) or not isinstance(expected_hash, str):
                return None
            path = worker_root / name
            if not path.is_file() or _sha256_file(path) != expected_hash:
                return None
            artifact = _load_json_object(path)
            config = artifact.get("config")
            rows = artifact.get("rollouts")
            if (
                artifact.get("schema_version") != SCHEMA_VERSION
                or artifact.get("model_identity") != dict(model_identity)
                or not isinstance(config, Mapping)
                or config.get("resolved_fingerprint") != resolved_fingerprint
                or config.get("decode_mode") != mode
                or not isinstance(rows, list)
            ):
                return None
            row_ids = list(dict.fromkeys(row.get("image_id") for row in rows))
            if row_ids != expected_ids:
                return None
            expected_count = len(expected_ids) * (1 if mode == "greedy" else SAMPLED_COUNT)
            if len(rows) != expected_count or any(row.get("stop_reason") == "length" for row in rows):
                return None
            if mode == "sampled":
                for offset in range(0, len(rows), SAMPLED_COUNT):
                    if [row.get("sample_index") for row in rows[offset : offset + SAMPLED_COUNT]] != list(
                        range(SAMPLED_COUNT)
                    ):
                        return None
            old_health = entry.get("generation_health", {}).get(mode)
            if not isinstance(old_health, Mapping):
                return None
            elapsed = float(old_health.get("elapsed_seconds", 0.0))
            if elapsed <= 0.0:
                return None
            health = _generation_health(rows, elapsed_seconds=elapsed)
            for field in (
                "completion_count",
                "generated_token_count",
                "stop_reason_counts",
                "natural_closure_count",
                "parser_status_counts",
            ):
                if old_health.get(field) != health[field]:
                    return None
            recomputed_health[mode] = health
        return {
            "batch_index": expected_batch_index,
            "image_ids": expected_ids,
            "artifacts": {mode: dict(parts[mode]) for mode in ("greedy", "sampled")},
            "generation_health": recomputed_health,
            "resumed": True,
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _open_research_vllm_session(launch: Any) -> Any:
    """Open max_num_seqs=32 through the upstream live-preflight contract."""

    from src.inference.vllm_backend import (
        open_vllm_backend_session,
    )

    session = open_vllm_backend_session(launch)
    settings = session.receipt.effective_settings
    engine_kwargs = settings.get("engine_kwargs")
    if not isinstance(engine_kwargs, Mapping) or engine_kwargs.get("max_num_seqs") != 32:
        session.close()
        raise ValueError("research panel opener requires max_num_seqs=32")
    preflight = settings.get("runtime_preflight")
    if not isinstance(preflight, Mapping) or preflight.get("status") != (
        "ready_for_engine_construction"
    ):
        session.close()
        raise RuntimeError("research panel opener lacks ready live preflight")
    return session


def resolve_panel_execution_model(
    resolved: Any,
    *,
    resolver: Callable[[Any], Mapping[str, Any] | None] | None = None,
    validator: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve and validate the composed checkpoint required by vLLM."""

    if resolver is None:
        from src.inference.pipeline import _resolve_execution_model_for_run

        resolver = _resolve_execution_model_for_run
    if validator is None:
        from src.inference.execution_model import validate_execution_model_receipt

        validator = validate_execution_model_receipt
    execution_model = resolver(resolved)
    if not isinstance(execution_model, Mapping):
        raise RuntimeError("vLLM panel requires a resolved execution-model identity")
    validated = dict(validator(execution_model))
    return validated


def _rollout_rows(
    *,
    native_outputs: Sequence[Any],
    requests: Sequence[Any],
    examples: Sequence[Any],
    media_hashes: Sequence[str],
    session: Any,
    decode_mode: str,
    request_seed: int | None,
) -> list[dict[str, Any]]:
    from src.inference.parsing import parse_compact_object_box_closed

    rows: list[dict[str, Any]] = []
    expected_count = 1 if decode_mode == "greedy" else SAMPLED_COUNT
    for native, request, example, media_hash in zip(
        native_outputs, requests, examples, media_hashes, strict=True
    ):
        prompt_ids = [int(value) for value in (native.prompt_token_ids or ())]
        if tuple(prompt_ids) != request.expected_executed_prompt_token_ids:
            raise RuntimeError(f"prompt token parity failed for {example.example_id}")
        for sample_index, completion in enumerate(
            _ordered_completions(native, expected_count=expected_count)
        ):
            token_ids, text, stop_reason = _materialize_completion(
                completion=completion,
                tokenizer=session._tokenizer,
                stop_token_id=session._im_end_token_id(),
            )
            trajectory_id = "greedy" if decode_mode == "greedy" else f"sample-{sample_index:02d}"
            parsed = parse_compact_object_box_closed(
                text,
                row_id=f"{example.example_id}:{trajectory_id}",
                row_index=0,
                image_width=int(example.image.width),
                image_height=int(example.image.height),
            )
            row: dict[str, Any] = {
                "image_id": physical_image_id(example),
                "example_id": str(example.example_id),
                "trajectory_id": trajectory_id,
                "decode_mode": decode_mode,
                "generated_token_ids": token_ids,
                "generated_token_ids_sha256": _sha256_json(token_ids),
                "generated_text": text,
                "stop_reason": stop_reason,
                "prompt_token_ids": prompt_ids,
                "prompt_token_ids_sha256": _sha256_json(prompt_ids),
                "observed_image_grid_thw": None,
                "source_image_file_sha256": request.image_sha256,
                "executed_rgb_sha256": media_hash,
                "predictions": parsed.to_artifact_dict(),
            }
            if decode_mode == "sampled":
                row["sample_index"] = sample_index
            if request_seed is not None:
                row["request_seed"] = int(request_seed)
            rows.append(row)
    return rows


def collect_panel(
    *,
    infer_config: Path,
    output_root: Path,
    worker_index: int,
    worker_count: int,
    image_batch_size: int,
    image_ids: set[str] | None,
    max_images: int | None,
    request_seed: int | None,
    resume: bool,
) -> Path:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import open_backend_session
    from src.inference.runtime import assemble_frontend

    resolved = load_infer_config(infer_config.resolve(strict=True))
    config = resolved.config
    if config.backend.type != "vllm":
        raise ValueError("collector requires a vLLM inference config")
    if config.generation.batch_size != 32:
        raise ValueError("collector requires generation.batch_size=32 (vLLM max_num_seqs)")
    if config.generation.max_new_tokens != 1024:
        raise ValueError("collector requires max_new_tokens=1024")
    selected_examples = select_examples(
        list(load_raw_examples(config.data.input_jsonl)), image_ids
    )
    examples = shard_examples(
        selected_examples,
        worker_index=worker_index,
        worker_count=worker_count,
    )
    if max_images is not None:
        if max_images <= 0:
            raise ValueError("max_images must be positive")
        examples = examples[:max_images]
    if not examples:
        raise ValueError("worker shard contains no images")

    worker_root = output_root / f"worker-{worker_index:02d}-of-{worker_count:02d}"
    if worker_root.exists() and any(worker_root.iterdir()) and not resume:
        raise ValueError(f"refusing non-empty worker output root: {worker_root}")
    worker_root.mkdir(parents=True, exist_ok=True)
    old_manifest: dict[str, Any] = {}
    manifest_path = worker_root / "manifest.json"
    if resume and manifest_path.is_file():
        try:
            candidate = _load_json_object(manifest_path)
            if (
                candidate.get("worker_index") == worker_index
                and candidate.get("worker_count") == worker_count
                and candidate.get("image_count") == len(examples)
            ):
                old_manifest = candidate
        except (OSError, ValueError, json.JSONDecodeError):
            old_manifest = {}
    execution_model = resolve_panel_execution_model(resolved)
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")),
        execution_model=execution_model,
    )
    requests, prompt_metadata = _build_requests(config, frontend, examples)
    for metadata in prompt_metadata.values():
        metadata["source_image_file_sha256"] = metadata.pop("image_sha256")
        metadata["source_image_hash_semantics"] = "source_file_bytes"
    chunks = _chunks(list(range(len(examples))), image_batch_size)
    manifest: dict[str, Any] = {
        "schema_version": "vllm_trajectory_panel_worker_manifest.v1",
        "worker_index": worker_index,
        "worker_count": worker_count,
        "image_count": len(examples),
        "image_batch_size": image_batch_size,
        "max_num_seqs": 32,
        "runtime_contract": "upstream_live_preflight_plus_research_multi_completion",
        "zero_truncation_required": True,
        "completed_image_count": 0,
        "status": "running",
        "batches": [],
    }
    old_batches = {
        entry.get("batch_index"): entry
        for entry in old_manifest.get("batches", [])
        if isinstance(entry, Mapping)
    }
    overall_started_at = time.perf_counter()
    generated_new_batch = False
    with open_backend_session(frontend.launch, opener=_open_research_vllm_session) as session:
        model_identity = _stable_session_identity(session.receipt)
        for batch_index, indices in enumerate(chunks):
            batch_examples = [examples[index] for index in indices]
            batch_requests = [requests[index] for index in indices]
            resumed_batch = _validated_resume_batch(
                worker_root=worker_root,
                entry=old_batches.get(batch_index, {}),
                expected_batch_index=batch_index,
                expected_examples=batch_examples,
                resolved_fingerprint=resolved.fingerprint,
                model_identity=model_identity,
            )
            if resumed_batch is not None:
                manifest["batches"].append(resumed_batch)
                manifest["completed_image_count"] += len(batch_examples)
                _atomic_write_json(manifest_path, manifest)
                continue
            batch_prompt_metadata = {
                str(example.example_id): prompt_metadata[str(example.example_id)]
                for example in batch_examples
            }
            written: dict[str, dict[str, str]] = {}
            mode_health: dict[str, Any] = {}
            batch_length_count = 0
            for decode_mode in ("greedy", "sampled"):
                generated_new_batch = True
                outputs, media_hashes, elapsed_seconds = _run_native_generation(
                    session=session,
                    requests=batch_requests,
                    decode_mode=decode_mode,
                    max_new_tokens=config.generation.max_new_tokens,
                    request_seed=request_seed if decode_mode == "sampled" else None,
                )
                rows = _rollout_rows(
                    native_outputs=outputs,
                    requests=batch_requests,
                    examples=batch_examples,
                    media_hashes=media_hashes,
                    session=session,
                    decode_mode=decode_mode,
                    request_seed=request_seed if decode_mode == "sampled" else None,
                )
                _record_research_live_decode(
                    session=session,
                    requests=batch_requests,
                    rows=rows,
                    decode_mode=decode_mode,
                )
                health = _generation_health(rows, elapsed_seconds=elapsed_seconds)
                mode_health[decode_mode] = health
                batch_length_count += int(health["stop_reason_counts"]["length"])
                artifact = build_artifact(
                    config=_artifact_config(
                        infer_config=infer_config,
                        resolved_fingerprint=resolved.fingerprint,
                        model_dtype=str(config.model.dtype),
                        decode_mode=decode_mode,
                        max_new_tokens=config.generation.max_new_tokens,
                        request_seed=request_seed if decode_mode == "sampled" else None,
                        worker_index=worker_index,
                        worker_count=worker_count,
                        image_batch_size=image_batch_size,
                    ),
                    model_identity=model_identity,
                    prompt_metadata=batch_prompt_metadata,
                    rollouts=rows,
                )
                path = worker_root / f"{decode_mode}-batch-{batch_index:05d}.json"
                _atomic_write_json(path, artifact)
                written[decode_mode] = {
                    "path": path.name,
                    "sha256": _sha256_file(path),
                }
            manifest["batches"].append(
                {
                    "batch_index": batch_index,
                    "image_ids": [physical_image_id(example) for example in batch_examples],
                    "artifacts": written,
                    "generation_health": mode_health,
                }
            )
            manifest["completed_image_count"] += len(batch_examples)
            manifest["wall_elapsed_seconds_current_process"] = (
                time.perf_counter() - overall_started_at
            )
            manifest["length_finished_completion_count"] = sum(
                int(mode["stop_reason_counts"]["length"])
                for batch in manifest["batches"]
                for mode in batch["generation_health"].values()
            )
            if batch_length_count:
                manifest["status"] = "failed_truncation"
            _atomic_write_json(manifest_path, manifest)
            if batch_length_count:
                session.close()
                manifest["runtime_receipt"] = session.receipt.to_artifact_dict()
                _atomic_write_json(manifest_path, manifest)
                raise RuntimeError(
                    f"batch {batch_index} produced {batch_length_count} length-finished "
                    "completions; zero-truncation panel collection stopped"
                )
    current_runtime_receipt = session.receipt.to_artifact_dict()
    previous_runtime_receipt = old_manifest.get("runtime_receipt")
    if not generated_new_batch and isinstance(previous_runtime_receipt, Mapping):
        manifest["runtime_receipt"] = dict(previous_runtime_receipt)
        manifest["resume_session_receipt"] = current_runtime_receipt
    else:
        manifest["runtime_receipt"] = current_runtime_receipt
    manifest["status"] = "completed"
    manifest["wall_elapsed_seconds_current_process"] = (
        time.perf_counter() - overall_started_at
    )
    total_completions = sum(
        int(mode["completion_count"])
        for batch in manifest["batches"]
        for mode in batch["generation_health"].values()
    )
    total_tokens = sum(
        int(mode["generated_token_count"])
        for batch in manifest["batches"]
        for mode in batch["generation_health"].values()
    )
    generation_elapsed_seconds = sum(
        float(mode["elapsed_seconds"])
        for batch in manifest["batches"]
        for mode in batch["generation_health"].values()
    )
    stop_reason_counts = {
        reason: sum(
            int(mode["stop_reason_counts"][reason])
            for batch in manifest["batches"]
            for mode in batch["generation_health"].values()
        )
        for reason in ("im_end", "length")
    }
    parser_status_counts: dict[str, int] = {}
    for batch in manifest["batches"]:
        for mode in batch["generation_health"].values():
            for status, count in mode["parser_status_counts"].items():
                parser_status_counts[status] = parser_status_counts.get(status, 0) + int(count)
    manifest["completion_health"] = {
        "stop_reason_counts": stop_reason_counts,
        "natural_closure_count": stop_reason_counts["im_end"],
        "parser_status_counts": parser_status_counts,
    }
    manifest["throughput"] = {
        "completion_count": total_completions,
        "generated_token_count": total_tokens,
        "generation_elapsed_seconds": generation_elapsed_seconds,
        "completions_per_second": total_completions / generation_elapsed_seconds,
        "routes_per_second": total_completions / generation_elapsed_seconds,
        "generated_tokens_per_second": total_tokens / generation_elapsed_seconds,
    }
    _atomic_write_json(manifest_path, manifest)
    return manifest_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--worker-count", type=int, default=8)
    parser.add_argument("--image-batch-size", type=int, default=16)
    parser.add_argument(
        "--image-ids",
        help="Optional comma-separated physical or example IDs selected before sharding.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        help="Bound this worker for an 8-16 image smoke; omit for the full shard.",
    )
    parser.add_argument(
        "--request-seed",
        type=int,
        help="Optional vLLM request seed metadata; sample_index remains trajectory identity.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip only completed artifact pairs that pass identity/hash/count validation.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    image_ids = parse_image_ids(args.image_ids)
    collect_panel(
        infer_config=args.infer_config,
        output_root=args.output_root,
        worker_index=args.worker_index,
        worker_count=args.worker_count,
        image_batch_size=args.image_batch_size,
        image_ids=image_ids,
        max_images=args.max_images,
        request_seed=args.request_seed,
        resume=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
