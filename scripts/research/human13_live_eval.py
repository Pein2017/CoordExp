#!/usr/bin/env python3
"""Live Human-13 clean-greedy evaluation helpers.

This experiment-local module only constructs analyzer-bound output records.
The live checkpoint decoder is deliberately kept separate so a training loss,
teacher-forced score, or trie diagnostic cannot be mistaken for deployment
evidence.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


def _digest(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _nonempty(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def build_analyzer_output(
    *,
    manifest: Any,
    manifest_sha256: str,
    image: Any,
    arm_id: str,
    milestone: int,
    checkpoint_path: str,
    checkpoint_payload_sha256: str,
    run_id: str,
    run_root: str,
    resolved_arm_plan_sha256: str,
    resolved_config_sha256: str,
    trajectory_id: str,
    generated_token_ids: Sequence[int],
    predictions: Sequence[Mapping[str, object]],
    stop_reason: str,
    malformed_row_count: int,
    runtime: Mapping[str, int | float],
) -> dict[str, object]:
    """Build one strict input row for ``analyze_human13_k_union.py``."""

    if isinstance(milestone, bool) or not isinstance(milestone, int) or milestone < 0:
        raise ValueError("milestone must be a non-negative integer")
    if int(getattr(image, "image_id")) <= 0:
        raise ValueError("image must carry one positive Human-13 image_id")
    if (
        isinstance(malformed_row_count, bool)
        or not isinstance(malformed_row_count, int)
        or malformed_row_count < 0
    ):
        raise ValueError("malformed_row_count must be a non-negative integer")
    ids = tuple(generated_token_ids)
    if any(isinstance(item, bool) or not isinstance(item, int) for item in ids):
        raise ValueError("generated_token_ids must contain integers")
    source = tuple(getattr(image, "trajectories"))[0]
    binding = manifest.binding
    source_identity = binding.source
    surface = binding.surface
    image_id = int(image.image_id)
    provenance = {
        "unit_id": binding.unit_id,
        "purpose": binding.purpose,
        "artifact_root": binding.artifact_root,
        "manifest_sha256": _digest(manifest_sha256, "manifest_sha256"),
        "panel_sha256": binding.panel.panel_sha256,
        "image_id": image_id,
        "panel_row_sha256": image.panel_row_sha256,
        "image_sha256": image.image_sha256,
        "arm_id": _nonempty(arm_id, "arm_id"),
        "milestone": milestone,
        "backend": "hf",
        "backend_version": source.request.backend_version,
        "physical_batch_size": 1,
        "do_sample": False,
        "max_new_tokens": source.request.max_new_tokens,
        "source_checkpoint_identity": {
            "checkpoint_path": source_identity.checkpoint_path,
            "base_model_path": source_identity.base_model_path,
            "adapter_sha256": source_identity.adapter_sha256,
            "special_embedding_sha256": source_identity.special_embedding_sha256,
        },
        "checkpoint_path": _nonempty(checkpoint_path, "checkpoint_path"),
        "checkpoint_payload_sha256": _digest(
            checkpoint_payload_sha256, "checkpoint_payload_sha256"
        ),
        "run_id": _nonempty(run_id, "run_id"),
        "run_root": _nonempty(run_root, "run_root"),
        "resolved_arm_plan_sha256": _digest(
            resolved_arm_plan_sha256, "resolved_arm_plan_sha256"
        ),
        "resolved_config_sha256": _digest(
            resolved_config_sha256, "resolved_config_sha256"
        ),
        "prompt_policy_fingerprint": surface.prompt_policy_fingerprint,
        "tokenizer_sha256": surface.tokenizer_sha256,
        "tokenizer_class": surface.tokenizer_class,
        "wrapper": surface.wrapper,
        "parser": surface.parser,
        "source_trajectory_id": source.trajectory_id,
        "trajectory_id": _nonempty(trajectory_id, "trajectory_id"),
    }
    return {
        "image_id": image_id,
        "arm_id": arm_id,
        "milestone": milestone,
        "decode_mode": "original_prompt_clean_greedy",
        "repetition_penalty": 1.0,
        "predictions": [dict(item) for item in predictions],
        "generated_token_ids": list(ids),
        "stop_reason": _nonempty(stop_reason, "stop_reason"),
        "malformed_row_count": malformed_row_count,
        "runtime": dict(runtime),
        "provenance": provenance,
    }


def checkpoint_payload_sha256(path: str | Path) -> str:
    """Hash one adapter-plus-embedding checkpoint tree deterministically."""

    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"checkpoint path is not a directory: {root}")
    files = tuple(sorted(item for item in root.rglob("*") if item.is_file()))
    if not files:
        raise ValueError("checkpoint payload is empty")
    digest = hashlib.sha256()
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        payload = item.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def source_outputs_from_manifest(
    *,
    manifest: Any,
    manifest_sha256: str,
    source_discovery_records: Mapping[int, Mapping[str, Any]],
    arm_id: str = "frozen_source",
    run_id: str = "frozen-source",
    run_root: str | None = None,
    resolved_arm_plan_sha256: str,
    resolved_config_sha256: str,
    checkpoint_path: str | None = None,
) -> tuple[dict[str, object], ...]:
    """Project the already frozen exact Source decode without re-running it."""

    binding = manifest.binding
    root = run_root or f"{binding.artifact_root}/{run_id}"
    source_checkpoint = binding.source.checkpoint_path
    evaluated_checkpoint = checkpoint_path or source_checkpoint
    payload_sha = checkpoint_payload_sha256(source_checkpoint)
    records: list[dict[str, object]] = []
    for image in manifest.images:
        raw = source_discovery_records.get(int(image.image_id))
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"missing Source discovery record for image {image.image_id}"
            )
        trajectory = raw.get("trajectory")
        parse = raw.get("parse")
        if not isinstance(trajectory, Mapping) or not isinstance(parse, Mapping):
            raise ValueError("Source discovery record lacks trajectory/parse evidence")
        rows = trajectory.get("rows")
        token_ids = trajectory.get("token_ids")
        dropped = parse.get("dropped_predictions")
        if (
            not isinstance(rows, list)
            or not isinstance(token_ids, list)
            or not isinstance(dropped, list)
        ):
            raise ValueError("Source discovery trajectory shape is invalid")
        predictions = tuple(
            {
                "generated_order": int(row["row_index"]),
                "description": str(row["category"]),
                "bbox": list(row["bbox"]),
            }
            for row in rows
        )
        trajectory_id = str(trajectory["trajectory_id"])
        if arm_id != "frozen_source":
            trajectory_id = f"eval:{arm_id}:0:{image.image_id}"
        records.append(
            build_analyzer_output(
                manifest=manifest,
                manifest_sha256=manifest_sha256,
                image=image,
                arm_id=arm_id,
                milestone=0,
                checkpoint_path=evaluated_checkpoint,
                checkpoint_payload_sha256=payload_sha,
                run_id=run_id,
                run_root=root,
                resolved_arm_plan_sha256=resolved_arm_plan_sha256,
                resolved_config_sha256=resolved_config_sha256,
                trajectory_id=trajectory_id,
                generated_token_ids=tuple(int(item) for item in token_ids),
                predictions=predictions,
                stop_reason=str(trajectory["stop_reason"]),
                malformed_row_count=len(dropped),
                runtime={
                    "decode_seconds": float(
                        raw.get("runtime_counters", {}).get(
                            "batch_elapsed_seconds", 0.0
                        )
                    )
                },
            )
        )
    return tuple(records)


def evaluate_hf_checkpoint(
    *,
    manifest: Any,
    manifest_sha256: str,
    checkpoint_path: str | Path,
    arm_id: str,
    milestone: int,
    run_id: str,
    run_root: str,
    resolved_arm_plan_sha256: str,
    resolved_config_sha256: str,
    source_config_path: str | Path,
) -> tuple[dict[str, object], ...]:
    """Run exact original-prompt HF batch-one greedy for one checkpoint.

    This is the only decision-owning model-quality path in the Human-13 probe.
    It intentionally uses the frozen Source inference surface (fp32/SDPA,
    no sampling, rp=1.0) while substituting only the evaluated adapter and
    special-token delta paths.
    """

    from dataclasses import replace

    from scripts.research.collect_human13_discovery import (
        _scientific_request,
        trajectory_input_from_decode_result,
    )
    from scripts.research.run_current_seeded_sampled_rollouts import (
        _build_requests,
        physical_image_id,
    )
    from src.config.fingerprint import sha256_json
    from src.config.inference import InferConfig, load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import GenerationPolicy, open_backend_session
    from src.inference.runtime import assemble_frontend

    checkpoint = Path(checkpoint_path).expanduser().resolve(strict=True)
    payload = load_infer_config(source_config_path).config.model_dump(mode="json")
    payload["adapter"]["path"] = str(checkpoint / "adapter")
    payload["embedding_delta"]["path"] = str(checkpoint / "special_token_embeddings")
    payload["generation"].update(
        {
            "batch_size": 1,
            "max_new_tokens": 3084,
            "temperature": 0.0,
            "top_p": 1.0,
            "n": 1,
            "repetition_penalty": 1.0,
        }
    )
    payload["debug"].update({"smoke": True, "dry_run": False})
    config = InferConfig.model_validate(payload)
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=sha256_json(
            config.generation.model_dump(mode="json")
        ),
    )
    examples = tuple(load_raw_examples(config.data.input_jsonl))
    requests, _prompt_metadata = _build_requests(config, frontend, examples)
    images = {int(image.image_id): image for image in manifest.images}
    if tuple(int(physical_image_id(item)) for item in examples) != tuple(images):
        raise ValueError("HF evaluation input order differs from the frozen panel")
    policy = GenerationPolicy(
        max_new_tokens=3084,
        repetition_penalty=1.0,
        temperature=0.0,
        top_p=1.0,
        include_raw_model_logprob=False,
    )
    payload_digest = checkpoint_payload_sha256(checkpoint)
    outputs: list[dict[str, object]] = []
    with open_backend_session(frontend.launch) as session:
        backend_version = str(session.receipt.backend_version)
        for example, base_request in zip(examples, requests, strict=True):
            image_id = int(physical_image_id(example))
            request = replace(
                base_request,
                request_id=f"eval:{arm_id}:{milestone}:{image_id}",
                generation_policy=policy,
            )
            started = time.perf_counter()
            result = tuple(session.decode((request,)))[0]
            elapsed = time.perf_counter() - started
            scientific = _scientific_request(
                mode="source", seed=None, backend_version=backend_version
            )
            trajectory, parse = trajectory_input_from_decode_result(
                image_id=image_id,
                trajectory_id=request.request_id,
                request=scientific,
                result=result,
                image_width=int(example.image.width),
                image_height=int(example.image.height),
            )
            predictions = tuple(
                {
                    "generated_order": row.row_index,
                    "description": row.category,
                    "bbox": list(row.bbox),
                }
                for row in trajectory.rows
            )
            outputs.append(
                build_analyzer_output(
                    manifest=manifest,
                    manifest_sha256=manifest_sha256,
                    image=images[image_id],
                    arm_id=arm_id,
                    milestone=milestone,
                    checkpoint_path=str(checkpoint),
                    checkpoint_payload_sha256=payload_digest,
                    run_id=run_id,
                    run_root=run_root,
                    resolved_arm_plan_sha256=resolved_arm_plan_sha256,
                    resolved_config_sha256=resolved_config_sha256,
                    trajectory_id=request.request_id,
                    generated_token_ids=trajectory.token_ids,
                    predictions=predictions,
                    stop_reason=trajectory.stop_reason,
                    malformed_row_count=len(parse["dropped_predictions"]),
                    runtime={"decode_seconds": elapsed},
                )
            )
    return tuple(outputs)


def write_outputs_jsonl(path: str | Path, outputs: Sequence[Mapping[str, Any]]) -> None:
    """Publish one immutable analyzer input JSONL."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite Human-13 outputs: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            dict(item),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for item in outputs
    ).encode("utf-8")
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(target)


__all__ = [
    "build_analyzer_output",
    "checkpoint_payload_sha256",
    "evaluate_hf_checkpoint",
    "source_outputs_from_manifest",
    "write_outputs_jsonl",
]
