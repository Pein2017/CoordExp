#!/usr/bin/env python
"""Qualify HF raw generation likelihood against a real teacher-forced forward."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.fingerprint import sha256_file, sha256_json
from src.config.inference import load_infer_config
from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
from src.data import load_raw_examples
from src.inference.backend import DecodeRequest, GenerationPolicy, open_backend_session
from src.inference.hf_backend import (
    HFBackendSession,
    compare_raw_generation_to_teacher_forced,
    teacher_forced_chosen_token_logprobs,
)
from src.inference.image_plan import plan_image_batch
from src.inference.model_assets import build_model_snapshot_manifest
from src.inference.prompt import build_prompt_record
from src.inference.runtime import assemble_frontend


DEFAULT_CONFIG = Path(
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_desc_first_geo_sorted_pure_ce_typegate_"
    "dora_r16a32_step4887_val200.yaml"
)
DEFAULT_OUTPUT = Path(
    "openspec/changes/add-coordexp-swift-vllm-inference-backend/"
    "source-studies/receipts/hf-wave1-raw-likelihood.json"
)


def main() -> int:
    args = _parse_args()
    repo_root = Path.cwd().resolve()
    config_path = Path(args.config).expanduser().resolve()
    resolved = load_infer_config(config_path)
    config = resolved.config.model_copy(
        update={
            "model": resolved.config.model.model_copy(
                update={"dtype": args.dtype or resolved.config.model.dtype}
            ),
            "backend": resolved.config.backend.model_copy(
                update={
                    "hf": resolved.config.backend.hf.model_copy(
                        update={
                            "attn_implementation": args.attn_implementation
                            or resolved.config.backend.hf.attn_implementation
                        }
                    )
                }
            ),
            "generation": resolved.config.generation.model_copy(
                update={
                    "batch_size": 1,
                    "max_new_tokens": args.max_new_tokens,
                    "repetition_penalty": 1.10,
                }
            ),
            "artifacts": resolved.config.artifacts.model_copy(
                update={"include_raw_model_logprob": True}
            ),
        }
    )
    if config.backend.type != "hf":
        raise SystemExit("HF raw-likelihood qualification requires backend.type: hf")

    effective_config = config.model_dump(mode="json")
    effective_config_fingerprint = sha256_json(effective_config)
    execution_payload_identity = _execution_payload_identity(config)
    generation_fingerprint = sha256_json(
        config.generation.model_dump(mode="json")
    )
    frontend = assemble_frontend(
        config,
        generation_config_fingerprint=generation_fingerprint,
    )
    example = load_raw_examples(config.data.input_jsonl, sample_limit=1)[0]
    image_batch = plan_image_batch(
        [example],
        components=frontend.qwen,
        processor_config=ProcessorConfig(
            do_resize=False,
            max_raw_pixels=1_000_000_000,
            max_merged_visual_tokens=1_000_000,
        ),
        row_indices=[0],
    )
    image_row = image_batch.rows[0]
    prompt = build_prompt_record(
        example,
        TemplateConfig(
            object_field_order=config.template.object_field_order,
            object_ordering=config.template.object_ordering,
            assistant_format=config.template.assistant_format,
            prompt=TemplatePromptConfig(
                system=config.template.prompt.system,
                user=config.template.prompt.user,
            ),
        ),
        processor=frontend.qwen.processor,
        row_index=0,
        merged_visual_tokens=image_row.merged_visual_tokens,
    )
    request = DecodeRequest(
        request_id=example.example_id,
        chat_text=prompt.chat_text,
        input_prompt_token_ids=tuple(prompt.input_prompt_token_ids),
        expected_executed_prompt_token_ids=tuple(
            prompt.expected_executed_prompt_token_ids
        ),
        image_path=image_row.image_path,
        declared_image_width=image_row.declared_width,
        declared_image_height=image_row.declared_height,
        decoded_image_width=image_row.decoded_width,
        decoded_image_height=image_row.decoded_height,
        image_sha256=image_row.image_content_sha256,
        expected_image_grid_thw=tuple(image_row.expected_image_grid_thw),
        logical_transform_id=image_row.logical_transform_id,
        generation_policy=GenerationPolicy(
            max_new_tokens=config.generation.max_new_tokens,
            repetition_penalty=config.generation.repetition_penalty,
            temperature=0.0,
            top_p=1.0,
            include_raw_model_logprob=True,
        ),
    )

    with open_backend_session(frontend.launch) as opened:
        if not isinstance(opened, HFBackendSession):
            raise RuntimeError("HF qualification opened an unexpected backend session")
        result = opened.decode((request,))[0]
        (
            native_inputs,
            executed_prompt_ids,
            _,
            executed_media_sha256,
        ) = opened._materialize_native_inputs((request,))  # noqa: SLF001
        reference = teacher_forced_chosen_token_logprobs(
            model=opened._model,  # noqa: SLF001
            native_prompt_inputs=native_inputs,
            generated_token_ids=result.generated_token_ids,
        )
        raw = torch.tensor(
            [
                float(trace.raw_model_logprob)
                for trace in result.token_trace
                if not trace.is_pad
            ],
            dtype=torch.float32,
        )
        comparison = compare_raw_generation_to_teacher_forced(
            raw,
            reference,
            atol=args.atol,
            rtol=0.0,
        )
        receipt = opened.receipt.to_artifact_dict()
    execution_payload_identity_after = _execution_payload_identity(config)
    if execution_payload_identity_after != execution_payload_identity:
        raise RuntimeError("execution payload changed while the probe was running")

    policy = torch.tensor(
        [
            float(trace.policy_logprob)
            for trace in result.token_trace
            if not trace.is_pad
        ],
        dtype=torch.float32,
    )
    absolute = torch.abs(raw - reference.detach().cpu().float())
    policy_raw_absolute = torch.abs(policy - raw)
    if tuple(executed_prompt_ids[0]) != request.expected_executed_prompt_token_ids:
        raise RuntimeError("teacher-forced prompt ids differ from the executed prompt")
    if result.stop_reason != "im_end" or not result.token_trace[-1].is_stop:
        raise RuntimeError("real HF qualification did not include terminal <|im_end|>")
    if not torch.any(policy_raw_absolute > 0):
        raise RuntimeError("repetition penalty did not distinguish policy and raw likelihood")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "ok": True,
        "probe": "coordexp-swift-hf-raw-likelihood-v1",
        "config": {
            "path": str(config_path),
            "authored_sha256": sha256_file(config_path),
            "resolved_fingerprint": resolved.fingerprint,
            "effective": effective_config,
            "effective_fingerprint": effective_config_fingerprint,
            "argv": list(sys.argv),
            "generation_fingerprint": generation_fingerprint,
            "repetition_penalty": 1.10,
            "dtype": config.model.dtype,
            "attn_implementation": config.backend.hf.attn_implementation,
            "include_raw_model_logprob": True,
        },
        "request": {
            "row_id": request.request_id,
            "image_path": request.image_path,
            "image_sha256": request.image_sha256,
            "logical_transform_id": request.logical_transform_id,
            "executed_media_sha256": executed_media_sha256[0],
            "input_prompt_token_count": len(request.input_prompt_token_ids),
            "executed_prompt_token_count": len(
                request.expected_executed_prompt_token_ids
            ),
            "executed_prompt_ids_sha256": sha256_json(
                list(request.expected_executed_prompt_token_ids)
            ),
        },
        "generation": {
            "generated_token_ids": list(result.generated_token_ids),
            "generated_token_text": [
                trace.token_text for trace in result.token_trace if not trace.is_pad
            ],
            "stop_reason": result.stop_reason,
            "terminal_stop_included": bool(result.token_trace[-1].is_stop),
            "policy_logprobs": policy.tolist(),
            "raw_model_logprobs": raw.tolist(),
            "teacher_forced_logprobs": reference.detach().cpu().float().tolist(),
        },
        "comparison": {
            "compared_steps": comparison.compared_steps,
            "atol": comparison.atol,
            "rtol": comparison.rtol,
            "max_absolute_difference": comparison.max_absolute_difference,
            "median_absolute_difference": float(torch.median(absolute).item()),
            "p99_absolute_difference": float(
                torch.quantile(absolute, 0.99).item()
            ),
            "policy_raw_distinct_step_count": int(
                torch.count_nonzero(policy_raw_absolute > 0).item()
            ),
            "max_policy_raw_absolute_difference": float(
                torch.max(policy_raw_absolute).item()
            ),
        },
        "runtime": {
            "backend_session": receipt,
            "packages": {
                name: metadata.version(name)
                for name in ("torch", "transformers", "peft")
            },
            "model_snapshot": build_model_snapshot_manifest(config.model.base_model),
            "execution_payload_identity": execution_payload_identity,
        },
        "source": {
            "git_head": _git_output(repo_root, "rev-parse", "HEAD"),
            "binding_scope": "listed_exercised_source_files",
            "files": {
                str(path.relative_to(repo_root)): sha256_file(path)
                for path in _source_files(repo_root)
            },
        },
    }
    output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)
    return 0


def _execution_payload_identity(config: Any) -> dict[str, Any]:
    adapter = _payload_component_identity(
        role="adapter",
        configured_path=None if config.adapter is None else config.adapter.path,
        required_files=("adapter_config.json", "adapter_model.safetensors"),
    )
    embedding_delta = _payload_component_identity(
        role="embedding_delta",
        configured_path=(
            None if config.embedding_delta is None else config.embedding_delta.path
        ),
        required_files=(
            "special_token_embeddings.json",
            "special_token_embeddings.safetensors",
        ),
    )
    determinants = {
        "adapter": adapter,
        "embedding_delta": embedding_delta,
    }
    return {
        **determinants,
        "fingerprint": sha256_json(determinants),
    }


def _payload_component_identity(
    *,
    role: str,
    configured_path: str | None,
    required_files: tuple[str, ...],
) -> dict[str, Any] | None:
    if configured_path is None:
        return None
    path = Path(configured_path).expanduser().resolve()
    root = path.parent if path.name in required_files else path
    files: dict[str, Any] = {}
    for filename in required_files:
        payload_path = root / filename
        if not payload_path.is_file():
            raise RuntimeError(
                f"{role} payload is missing required file: {payload_path}"
            )
        files[filename] = {
            "path": str(payload_path),
            "size_bytes": payload_path.stat().st_size,
            "sha256": sha256_file(payload_path),
        }
    component = {
        "role": role,
        "configured_path": str(path),
        "root": str(root),
        "files": files,
    }
    component["fingerprint"] = sha256_json(component)
    return component


def _source_files(repo_root: Path) -> tuple[Path, ...]:
    return tuple(
        repo_root / relative
        for relative in (
            "scripts/probes/coordexp_swift/hf_raw_likelihood.py",
            "src/inference/backend.py",
            "src/inference/hf_backend.py",
            "src/inference/image_plan.py",
            "src/inference/prompt.py",
            "src/inference/runtime.py",
            "src/qwen/images.py",
        )
    )


def _git_output(repo_root: Path, *args: str) -> str:
    return _git_bytes(repo_root, *args).decode("utf-8").strip()


def _git_bytes(repo_root: Path, *args: str) -> bytes:
    return subprocess.run(
        ("git", *args),
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="fp32")
    parser.add_argument(
        "--attn-implementation",
        choices=("flash_attention_2", "sdpa", "eager"),
        default="sdpa",
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
