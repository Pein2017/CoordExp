"""GPU child execution for the BF16 vLLM qualification producer."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, NoReturn, cast

import yaml

from src.common.errors import RuntimeContractError


@dataclass(frozen=True)
class RuntimeDependencies:
    run_composition: Callable[[Path, Path], Mapping[str, object]]
    run_inference: Callable[[str, Path, Path, int, bool], Mapping[str, object]]
    owned_child_pids: Callable[[], list[int]]


def run_child(
    *,
    kind: str,
    max_num_seqs: int,
    config_path: str | Path,
    evidence_dir: str | Path,
    dependencies: RuntimeDependencies | None = None,
) -> dict[str, object]:
    """Execute exactly one canonical mode inside an isolated worker process."""

    expected = {
        "composition": 1,
        "runtime": 1,
        "concurrency": 4,
        "forced_replay": 1,
    }
    if expected.get(kind) != max_num_seqs:
        _fail(
            "qualification child mode is not canonical",
            code="vllm_qualification.child_mode",
            context={"kind": kind, "max_num_seqs": max_num_seqs},
        )
    config = Path(config_path).expanduser().resolve()
    output = Path(evidence_dir).expanduser().resolve()
    if output.exists():
        _fail(
            "qualification child evidence directory must be absent",
            code="vllm_qualification.child_output_exists",
            context={"path": str(output)},
        )
    output.mkdir(parents=True, exist_ok=False)
    deps = dependencies or RuntimeDependencies(
        run_composition=_run_production_composition,
        run_inference=_run_production_inference,
        owned_child_pids=_owned_child_pids,
    )
    if kind == "composition":
        evidence = dict(deps.run_composition(config, output))
    else:
        evidence = dict(
            deps.run_inference(
                kind,
                config,
                output,
                max_num_seqs,
                kind == "forced_replay",
            )
        )
    children = deps.owned_child_pids()
    process = {
        "worker_pid": os.getpid(),
        "worker_returncode": 0,
        "worker_pid_alive_after_exit": False,
        "owned_children_after": children,
        # Only the parent can measure this after this process exits.
        "gpu_memory_returned_to_baseline": False,
    }
    return {"evidence": evidence, "process": process}


def _run_production_inference(
    kind: str,
    config_path: Path,
    evidence_dir: Path,
    max_num_seqs: int,
    include_raw_model_logprob: bool,
) -> dict[str, object]:
    """Run the retained shard/backend/artifact path with a producer-only qualifier."""

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.config.writer import write_resolved_config_artifacts
    from src.inference.artifacts import validate_scored_artifact_set
    from src.inference.execution_model import (
        bind_execution_model_composition,
        resolve_execution_model,
    )
    from src.inference.execution_model_composition import (
        load_execution_model_composition_receipt,
    )
    from src.inference.pipeline import run_shard
    from src.inference.vllm_backend import open_vllm_backend_session

    overlay_path = evidence_dir / "qualification-overlay.yaml"
    overlay = {
        "schema_version": 1,
        "extends": str(config_path),
        "run": {
            "name": f"vllm-qualification-{kind}",
            "artifact_root": str(evidence_dir),
            "output_dir": "run",
            "collision_policy": "fail",
        },
        "generation": {"batch_size": max_num_seqs},
        "artifacts": {"include_raw_model_logprob": include_raw_model_logprob},
        "debug": {"smoke": True, "dry_run": False},
    }
    overlay_path.write_text(
        yaml.safe_dump(overlay, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    resolved = load_infer_config(overlay_path)
    if resolved.config.backend.type != "vllm" or resolved.config.model.dtype != "bf16":
        _fail(
            "qualification runtime requires a BF16 vLLM config",
            code="vllm_qualification.child_config",
        )
    execution_model = resolve_execution_model(
        base_model_path=resolved.config.model.base_model,
        target_dtype="bf16",
        adapter_path=(None if resolved.config.adapter is None else resolved.config.adapter.path),
        adapter_name=("default" if resolved.config.adapter is None else resolved.config.adapter.name),
        embedding_delta_path=(
            None if resolved.config.embedding_delta is None else resolved.config.embedding_delta.path
        ),
        _skip_existing_composition_fidelity=True,
    )
    composition_path = evidence_dir.parent / "composition" / "composition-receipt.json"
    composition = load_execution_model_composition_receipt(composition_path)
    bound_execution_model = bind_execution_model_composition(
        execution_model,
        composition,
        composition_path=composition_path,
    )
    run_dir = evidence_dir / "run"
    write_resolved_config_artifacts(cast(Any, resolved), run_dir)
    (run_dir / "execution_model.json").write_text(
        json.dumps(bound_execution_model, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def launch_qualifier(
        launch: Any,
        engine_kwargs: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {
            "candidate_version": metadata.version("vllm"),
            "engine_settings": dict(engine_kwargs),
            "runtime_qualification": {
                "status": "producer_candidate",
                "kind": kind,
                "max_num_seqs": max_num_seqs,
                "model_identity_sha256": sha256_json(bound_execution_model),
            },
        }

    def raw_replay_qualifier(
        launch: Any,
        processor_identity: Mapping[str, object],
    ) -> Mapping[str, object]:
        return {
            "status": "producer_candidate",
            "kind": kind,
            "max_num_seqs": max_num_seqs,
            "processor_source_sha256": processor_identity.get("source_sha256"),
        }

    def session_opener(launch: Any) -> Any:
        return open_vllm_backend_session(
            launch,
            launch_qualifier=launch_qualifier,
            raw_replay_qualifier=raw_replay_qualifier,
        )

    run_shard(
        resolved=resolved,
        output_dir=run_dir,
        row_indices=tuple(range(max_num_seqs)),
        worker_metadata={
            "qualification_kind": kind,
            "qualification_max_num_seqs": max_num_seqs,
        },
        session_opener=session_opener,
        execution_model=bound_execution_model,
    )
    validate_scored_artifact_set(run_dir)
    return _summarize_inference_evidence(
        kind=kind,
        run_dir=run_dir,
        max_num_seqs=max_num_seqs,
    )


def _run_production_composition(
    config_path: Path,
    evidence_dir: Path,
) -> dict[str, object]:
    """Execute dynamic-HF versus executable-snapshot composition evidence."""

    import torch
    from PIL import Image

    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
    from src.data import load_raw_examples
    from src.inference.backend import BackendLaunch
    from src.inference.execution_model import (
        bind_execution_model_composition,
        resolve_execution_model,
    )
    from src.inference.execution_model_composition import (
        COMPOSITION_SOURCE_RELATIVE_PATH,
        build_execution_model_composition_receipt,
        compare_execution_models,
        write_execution_model_composition_receipt,
    )
    from src.inference.hf_backend import _load_hf_components
    from src.inference.image_plan import plan_image_batch
    from src.inference.prompt import build_prompt_record
    from src.qwen.images import apply_logical_image_transform, rgb_image_sha256
    from src.qwen.runtime_loading import QwenLoadOptions, load_qwen_components_from_options
    from src.qwen.tokens import DEFAULT_WRAPPER_TOKENS

    resolved = load_infer_config(config_path)
    config = resolved.config
    if config.backend.type != "vllm" or config.model.dtype != "bf16":
        _fail(
            "composition qualification requires a BF16 vLLM config",
            code="vllm_qualification.child_config",
        )
    generation_fingerprint = sha256_json(config.generation.model_dump(mode="json"))
    execution_model = resolve_execution_model(
        base_model_path=config.model.base_model,
        target_dtype="bf16",
        adapter_path=None if config.adapter is None else config.adapter.path,
        adapter_name="default" if config.adapter is None else config.adapter.name,
        embedding_delta_path=(
            None if config.embedding_delta is None else config.embedding_delta.path
        ),
        _skip_existing_composition_fidelity=True,
    )
    dynamic_launch = BackendLaunch(
        backend="hf",
        model_path=str(Path(config.model.base_model).resolve()),
        model_dtype="bf16",
        batch_size=1,
        generation_config_fingerprint=generation_fingerprint,
        backend_options={
            "hf": {
                "attn_implementation": "sdpa",
                "patch_embed_linearization": "enabled",
            }
        },
        adapter=(None if config.adapter is None else config.adapter.model_dump(mode="json")),
        embedding_delta=(
            None
            if config.embedding_delta is None
            else config.embedding_delta.model_dump(mode="json")
        ),
    )
    dynamic_loaded = _load_hf_components(dynamic_launch)
    dynamic_qwen = dynamic_loaded.qwen
    materialized_qwen = load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=str(execution_model["model_path"]),
            dtype="bf16",
            attn_implementation="sdpa",
            patch_embed_linearization="enabled",
            load_model=True,
        )
    )
    device = torch.device("cuda:0")
    if dynamic_qwen.model is None or materialized_qwen.model is None:
        _fail(
            "composition qualification failed to load both executable models",
            code="vllm_qualification.child_model",
        )
    dynamic_model = dynamic_qwen.model.to(device).eval()
    materialized_model = materialized_qwen.model.to(device).eval()
    examples = list(load_raw_examples(config.data.input_jsonl))
    if not examples:
        _fail("composition input is empty", code="vllm_qualification.child_input")
    raw_example = examples[0]
    image_batch = plan_image_batch(
        [raw_example],
        components=dynamic_qwen,
        processor_config=ProcessorConfig(
            do_resize=False,
            max_raw_pixels=1_000_000_000,
            max_merged_visual_tokens=1_000_000,
        ),
        row_indices=[0],
    )
    image_row = image_batch.rows[0]
    prompt_record = build_prompt_record(
        raw_example,
        TemplateConfig(
            object_field_order=config.template.object_field_order,
            object_ordering=config.template.object_ordering,
            assistant_format=config.template.assistant_format,
            prompt=TemplatePromptConfig(
                system=config.template.prompt.system,
                user=config.template.prompt.user,
            ),
        ),
        processor=dynamic_qwen.processor,
        row_index=0,
        merged_visual_tokens=image_row.merged_visual_tokens,
        object_order_seed=config.template.object_order_seed,
    )
    with Image.open(raw_example.image.path) as source_image:
        image = apply_logical_image_transform(
            source_image.convert("RGB"),
            image_row.logical_transform_id,
            example_id=raw_example.example_id,
            image_path=raw_example.image.path,
        )
    try:
        dynamic_encoded = dynamic_qwen.processor(
            text=[prompt_record.chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        materialized_encoded = materialized_qwen.processor(
            text=[prompt_record.chat_text],
            images=[image.copy()],
            padding=True,
            return_tensors="pt",
            do_resize=False,
        )
        dynamic_inputs = _to_device_mapping(dynamic_encoded, device=device)
        materialized_inputs = _to_device_mapping(materialized_encoded, device=device)
        selected_ids = [
            *(dynamic_qwen.token_identity.wrapper_token_ids[token] for token in DEFAULT_WRAPPER_TOKENS),
            *dynamic_qwen.token_identity.coordinate_token_ids,
        ]
        comparison = compare_execution_models(
            dynamic_model=dynamic_model,
            materialized_model=materialized_model,
            dynamic_native_inputs=dynamic_inputs,
            materialized_native_inputs=materialized_inputs,
            selected_token_ids=selected_ids,
            generation_kwargs={
                "max_new_tokens": config.generation.max_new_tokens,
                "do_sample": False,
                "repetition_penalty": config.generation.repetition_penalty,
                "eos_token_id": dynamic_qwen.token_identity.im_end_token_ids[0],
                "pad_token_id": dynamic_qwen.tokenizer.pad_token_id,
                "use_cache": True,
            },
            expected_merged_target_identity=_merged_target_identity(execution_model),
            expected_folded_selected_rows_sha256=_folded_selected_rows_sha256(
                execution_model
            ),
        )
        composition = build_execution_model_composition_receipt(
            execution_model=execution_model,
            fixture_identity={
                "row_id": raw_example.example_id,
                "row_index": 0,
                "input_jsonl_sha256": _sha256_file(Path(config.data.input_jsonl)),
                "prompt_ids_sha256": sha256_json(prompt_record.prompt_token_ids),
                "dynamic_executed_prompt_ids_sha256": sha256_json(
                    dynamic_inputs["input_ids"][0].detach().cpu().tolist()
                ),
                "materialized_executed_prompt_ids_sha256": sha256_json(
                    materialized_inputs["input_ids"][0].detach().cpu().tolist()
                ),
                "processor_fingerprint": sha256_json(
                    dynamic_qwen.processor_identity.to_artifact_dict()
                ),
                "image_file_sha256": _sha256_file(Path(raw_example.image.path)),
                "executed_media_sha256": rgb_image_sha256(image),
                "generation_fingerprint": generation_fingerprint,
                "max_new_tokens": config.generation.max_new_tokens,
            },
            composition_source_identity={
                "path": COMPOSITION_SOURCE_RELATIVE_PATH.as_posix(),
                "sha256": _sha256_file(
                    Path(__file__).resolve().parents[2] / COMPOSITION_SOURCE_RELATIVE_PATH
                ),
            },
            resolved_config_identity={
                "fingerprint": resolved.fingerprint,
                "entry_config_path": str(resolved.entry_config_path),
                "sources": [
                    {"path": str(source.path), "sha256": source.sha256}
                    for source in resolved.sources
                ],
            },
            comparison=comparison,
        )
        composition_path = evidence_dir / "composition-receipt.json"
        write_execution_model_composition_receipt(composition_path, composition)
        bound = bind_execution_model_composition(
            execution_model,
            composition,
            composition_path=composition_path,
        )
        (evidence_dir / "bound-execution-model.json").write_text(
            json.dumps(bound, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "status": "passed",
            "composition_digest": composition["digest"],
            "prompt_ids_equal": comparison["composition_checks"]["prompt_ids"],
            "selected_rows_equal": comparison["composition_checks"][
                "selected_rows_target_dtype"
            ],
            "greedy_ids_equal": comparison["behavior_checks"][
                "greedy_generated_ids_match"
            ],
            "full_vocab": _bounded_numeric_summary(
                comparison["full_vocab"], comparison["full_vocab_shape"]
            ),
            "selected_vocab": _bounded_numeric_summary(
                comparison["selected_vocab"], comparison["selected_vocab_shape"]
            ),
            "cleanup": _child_cleanup(shutdown_completed=True),
        }
    finally:
        image.close()
        dynamic_model = None
        materialized_model = None
        dynamic_qwen = None
        materialized_qwen = None
        dynamic_loaded = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _summarize_inference_evidence(
    *,
    kind: str,
    run_dir: Path,
    max_num_seqs: int,
) -> dict[str, object]:
    manifest = _read_json(run_dir / "run_manifest.json")
    summary = _read_json(run_dir / "summary.json")
    raw_rows = _read_jsonl(run_dir / "gt_vs_pred.jsonl")
    trace_rows = _read_jsonl(run_dir / "pred_token_trace.jsonl")
    if summary.get("terminal_status") != "completed" or len(raw_rows) != max_num_seqs:
        _fail(
            "qualification inference artifacts are incomplete",
            code="vllm_qualification.child_artifacts",
        )
    backend = manifest.get("backend_session")
    settings = backend.get("effective_settings") if isinstance(backend, Mapping) else None
    engine_kwargs = settings.get("engine_kwargs") if isinstance(settings, Mapping) else None
    if not isinstance(engine_kwargs, Mapping) or engine_kwargs.get("max_num_seqs") != max_num_seqs:
        _fail(
            "qualification engine concurrency differs from the requested mode",
            code="vllm_qualification.child_engine",
        )
    generated = [row for row in trace_rows if row.get("trace_type") == "generated_token"]
    policy = [row.get("policy_logprob") for row in generated]
    base: dict[str, object] = {
        "status": "passed",
        "max_num_seqs": max_num_seqs,
        "request_count": len(raw_rows),
        "completed_request_count": len(raw_rows),
        "cleanup": _child_cleanup(
            shutdown_completed=_cleanup_completed(settings)
        ),
    }
    if kind in ("runtime", "concurrency"):
        base.update(
            generated_token_count=len(generated),
            finite_non_positive_policy_logprobs=(
                bool(policy)
                and all(_finite_non_positive(value) for value in policy)
            ),
        )
        if kind == "concurrency":
            base["ordered_request_ids_sha256"] = _sha256_json(
                [row.get("row_id") for row in raw_rows]
            )
        return base
    raw_candidates = [row.get("raw_model_logprob") for row in generated]
    raw_values = [
        float(value)
        for value in raw_candidates
        if not isinstance(value, bool) and isinstance(value, (int, float))
    ]
    raw_replay = settings.get("raw_replay") if isinstance(settings, Mapping) else None
    processor = (
        raw_replay.get("forced_logits_processor")
        if isinstance(raw_replay, Mapping)
        else None
    )
    base.update(
        aligned_token_count=len(raw_values),
        processor_source_sha256=(
            processor.get("source_sha256") if isinstance(processor, Mapping) else None
        ),
        finite_non_positive_raw_logprobs=(
            bool(raw_values)
            and len(raw_values) == len(raw_candidates)
            and all(_finite_non_positive(value) for value in raw_values)
        ),
        raw_logprob_min=min(raw_values) if raw_values else None,
        raw_logprob_max=max(raw_values) if raw_values else None,
    )
    return base


def _cleanup_completed(settings: object) -> bool:
    preflight = settings.get("runtime_preflight") if isinstance(settings, Mapping) else None
    cleanup = preflight.get("cleanup") if isinstance(preflight, Mapping) else None
    return (
        isinstance(cleanup, Mapping)
        and cleanup.get("status") == "completed"
        and all(event.get("status") == "completed" for event in cleanup.get("events", ()))
    )


def _child_cleanup(*, shutdown_completed: bool) -> dict[str, object]:
    return {
        "shutdown_completed": shutdown_completed,
        "worker_pid_alive_after_exit": False,
        "owned_children_after": [],
        "gpu_memory_returned_to_baseline": False,
    }


def _owned_child_pids() -> list[int]:
    children_path = Path(f"/proc/{os.getpid()}/task/{os.getpid()}/children")
    try:
        value = children_path.read_text(encoding="utf-8").strip()
    except OSError:
        return []
    return [] if not value else [int(item) for item in value.split()]


def _merged_target_identity(execution_model: Mapping[str, object]) -> dict[str, Any] | None:
    materialization = execution_model.get("materialization")
    adapter_merge = materialization.get("adapter_merge") if isinstance(materialization, Mapping) else None
    if adapter_merge is None:
        return None
    identity = adapter_merge.get("merge", {}).get("target_weight_identity")
    if not isinstance(identity, dict):
        _fail("merged target identity is missing", code="vllm_qualification.child_model")
    return identity


def _folded_selected_rows_sha256(execution_model: Mapping[str, object]) -> str | None:
    materialization = execution_model.get("materialization")
    delta = materialization.get("embedding_delta_fold") if isinstance(materialization, Mapping) else None
    if delta is None:
        return None
    value = delta.get("selected_rows_after_sha256") if isinstance(delta, Mapping) else None
    if not isinstance(value, str) or len(value) != 64:
        _fail("folded row identity is missing", code="vllm_qualification.child_model")
    return value


def _bounded_numeric_summary(value: Mapping[str, object], shape: object) -> dict[str, object]:
    count = math.prod(int(item) for item in shape) if isinstance(shape, list) else 0
    return {
        "allclose": value.get("allclose"),
        "max_abs_diff": value.get("max_abs_diff"),
        "max_rel_diff": value.get("max_rel_diff"),
        "compared_value_count": count,
    }


def _to_device_mapping(value: Mapping[str, Any], *, device: Any) -> dict[str, Any]:
    import torch

    return {
        key: item.to(device) if isinstance(item, torch.Tensor) else item
        for key, item in dict(value).items()
    }


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        _fail("qualification JSON artifact is malformed", code="vllm_qualification.child_artifacts")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            value = json.loads(line)
            if not isinstance(value, dict):
                _fail("qualification JSONL artifact is malformed", code="vllm_qualification.child_artifacts")
            rows.append(value)
    return rows


def _finite_non_positive(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value <= 0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _fail(
    message: str,
    *,
    code: str,
    context: Mapping[str, object] | None = None,
) -> NoReturn:
    raise RuntimeContractError(message, code=code, context=context)


__all__ = ["RuntimeDependencies", "run_child"]
