#!/usr/bin/env python3
"""Replay one-image Qwen visual features into a homogeneous physical batch of four."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_batch_coordinate_logit_invariance import (  # noqa: E402
    CACHED_REPLAY_MAXIMUM_NEW_TOKENS,
    COMMON_COORDINATE_RECIPIENT_SUFFIX,
    DEFAULT_CONFIG,
    DEFAULT_SOURCE_BUNDLE,
    DEFAULT_SOURCE_JSONL,
    FROZEN_TARGET_PROMPT_SHA256,
    HOMOGENEOUS_TARGET_COPIES_LAYOUT,
    SINGLE_TARGET_LAYOUT,
    TARGET_ROW_WITH_TARGET_GEOMETRY,
    _cached_natural_suffix_logits,
    _direct_full_prefix_logits,
    _requests,
    _serialize_rows,
    compare_coordinate_summaries,
)
from scripts.research.run_native_commit_redistribution import (  # noqa: E402
    IMAGE_ID,
    NO_APPENDED_ROW,
    _condition_prompt,
    _json_hash,
    _read_json,
    _temporary_cwd,
    _validate_identity_continuity,
    build_condition_rows,
    extract_source_rows,
)
from scripts.research.run_sampled_rescue_transition import (  # noqa: E402
    _sha256_file,
    _write_bundle_once,
)
from src.analysis.visual_support_counterfactual.intervention import (  # noqa: E402
    FeatureBundle,
    FeatureReplayController,
    capture_feature_bundle,
    feature_bundle_fingerprint,
)


EXPECTED_CACHED_SINGLE_VECTOR_SHA256 = (
    "0dadea7e1b4b0835946d5b33fef2a28a83eee448ef153acc787d6c6aab6f3c42"
)
EXPECTED_CACHED_HOMOGENEOUS_VECTOR_SHA256 = (
    "a7fbe8f169ae67c6b912d9b110f70ecdb71a3392f1da8f1c25aa3b2a61c81fbe"
)

NATURAL_SINGLE_TARGET_ARM = "Natural Single Target"
NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM = "Natural Homogeneous Batch Four"
REPLAYED_SINGLE_TARGET_ARM = "Replayed Single Target"
SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM = (
    "Single-Target Features Replayed into Homogeneous Batch Four"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay exact single-target primary and DeepStack visual features "
            "into a homogeneous downstream batch of four."
        )
    )
    parser.add_argument("--source-bundle", type=Path, default=DEFAULT_SOURCE_BUNDLE)
    parser.add_argument("--infer-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def build_layouts() -> tuple[dict[str, Any], dict[str, Any]]:
    single = {
        "layout_name": SINGLE_TARGET_LAYOUT,
        "layout_instance": "single-target",
        "condition_names": [TARGET_ROW_WITH_TARGET_GEOMETRY],
        "target_positions": [0],
    }
    homogeneous = {
        "layout_name": HOMOGENEOUS_TARGET_COPIES_LAYOUT,
        "layout_instance": "homogeneous-target-copies",
        "condition_names": [TARGET_ROW_WITH_TARGET_GEOMETRY] * 4,
        "target_positions": [0, 1, 2, 3],
    }
    return single, homogeneous


def _float32_tensor_fingerprint(value: torch.Tensor) -> dict[str, Any]:
    cpu = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if not bool(torch.isfinite(cpu).all()):
        raise SystemExit("visual feature tensor contains non-finite values")
    payload = cpu.numpy().tobytes()
    return {
        "shape": [int(x) for x in value.shape],
        "executed_dtype": str(value.dtype),
        "float32_sha256": hashlib.sha256(payload).hexdigest(),
        "float32_root_mean_square": float(torch.sqrt(torch.mean(cpu.square()))),
        "float32_maximum_absolute_value": float(torch.max(torch.abs(cpu))),
    }


def float32_feature_bundle_fingerprint(bundle: FeatureBundle) -> dict[str, Any]:
    return {
        "primary": [_float32_tensor_fingerprint(value) for value in bundle.primary],
        "deepstack": [_float32_tensor_fingerprint(value) for value in bundle.deepstack],
    }


def split_homogeneous_feature_bundle(
    bundle: FeatureBundle, *, batch_size: int = 4
) -> tuple[list[FeatureBundle], dict[str, Any]]:
    if len(bundle.primary) != batch_size:
        raise SystemExit(
            f"expected {batch_size} split primary streams, received {len(bundle.primary)}"
        )
    token_counts = {int(value.shape[0]) for value in bundle.primary}
    hidden_sizes = {int(value.shape[1]) for value in bundle.primary if value.ndim == 2}
    if len(token_counts) != 1 or len(hidden_sizes) != 1:
        raise SystemExit("homogeneous primary streams do not share one feature layout")
    token_count = next(iter(token_counts))
    deepstack_chunks: list[tuple[torch.Tensor, ...]] = []
    for stream in bundle.deepstack:
        if stream.ndim != 2 or int(stream.shape[0]) != batch_size * token_count:
            raise SystemExit("homogeneous DeepStack stream cannot be split per image")
        deepstack_chunks.append(tuple(torch.split(stream, token_count, dim=0)))
    per_image = [
        FeatureBundle(
            primary=(bundle.primary[index].detach().clone(),),
            deepstack=tuple(chunks[index].detach().clone() for chunks in deepstack_chunks),
        )
        for index in range(batch_size)
    ]
    reference = per_image[0]
    primary_equal = [
        bool(torch.equal(reference.primary[0], item.primary[0])) for item in per_image
    ]
    deepstack_equal = [
        [
            bool(torch.equal(reference.deepstack[layer], item.deepstack[layer]))
            for item in per_image
        ]
        for layer in range(len(reference.deepstack))
    ]
    return per_image, {
        "batch_size": batch_size,
        "primary_token_count_per_image": token_count,
        "primary_all_equal": all(primary_equal),
        "primary_equal_to_first": primary_equal,
        "deepstack_all_equal": all(all(row) for row in deepstack_equal),
        "deepstack_equal_to_first_by_layer": deepstack_equal,
    }


def compare_feature_tensors(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    left = reference.detach().to(device="cpu", dtype=torch.float32).contiguous()
    right = candidate.detach().to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(left.shape) != tuple(right.shape):
        raise SystemExit("feature comparison shapes differ")
    difference = right - left
    reference_norm = torch.linalg.vector_norm(left)
    difference_norm = torch.linalg.vector_norm(difference)
    return {
        "root_mean_square_difference": float(torch.sqrt(torch.mean(difference.square()))),
        "maximum_absolute_difference": float(torch.max(torch.abs(difference))),
        "relative_l2_difference": float(
            difference_norm / torch.clamp(reference_norm, min=1e-12)
        ),
        "exact_equal": bool(torch.equal(reference, candidate)),
    }


def compare_feature_bundles(
    reference: FeatureBundle, candidate: FeatureBundle
) -> dict[str, Any]:
    if len(reference.primary) != len(candidate.primary) or len(reference.deepstack) != len(
        candidate.deepstack
    ):
        raise SystemExit("feature bundle stream counts differ")
    return {
        "primary": [
            compare_feature_tensors(left, right)
            for left, right in zip(reference.primary, candidate.primary, strict=True)
        ],
        "deepstack": [
            compare_feature_tensors(left, right)
            for left, right in zip(reference.deepstack, candidate.deepstack, strict=True)
        ],
    }


def _score_layout(
    *,
    backend: Any,
    layout: Mapping[str, Any],
    prompt_token_ids_by_condition: Mapping[str, Sequence[int]],
    model_inputs: Mapping[str, Any],
    policy: Any,
    replay_features: FeatureBundle | None,
    grid_thw: Sequence[int],
    merge_size: int,
) -> dict[str, Any]:
    def run_path(*, direct: bool) -> tuple[torch.Tensor, list[int], dict[str, Any], Any]:
        requests = _requests(
            layout=layout,
            repeat_index=0,
            prompt_token_ids_by_condition=prompt_token_ids_by_condition,
            model_inputs=model_inputs,
            policy=policy,
            direct=direct,
        )
        controller = None
        context: Any = contextlib.nullcontext()
        if replay_features is not None:
            controller = FeatureReplayController(
                model=backend.model,
                recipient=replay_features,
                donor=replay_features,
                grid_thw=grid_thw,
                merge_size=merge_size,
                selected_mask=None,
                mode="clean",
            )
            context = controller
        with context:
            if direct:
                logits, tokens, path_receipt = _direct_full_prefix_logits(
                    backend=backend,
                    requests=requests,
                )
            else:
                logits, tokens, path_receipt = _cached_natural_suffix_logits(
                    backend=backend,
                    requests=requests,
                    policy=policy,
                    target_positions=layout["target_positions"],
                )
        controller_receipt = (
            None
            if controller is None
            else controller.validate_completed(expected_feature_calls=1)
        )
        return logits, tokens, path_receipt, controller_receipt

    cached_logits, cached_tokens, cached_receipt, cached_controller = run_path(direct=False)
    direct_logits, direct_tokens, direct_receipt, direct_controller = run_path(direct=True)
    return {
        "layout": dict(layout),
        "cached_path": {
            **cached_receipt,
            "controller": cached_controller,
            "rows": _serialize_rows(
                layout=layout,
                logits=cached_logits,
                selected_tokens=cached_tokens,
                prompt_token_ids_by_condition=prompt_token_ids_by_condition,
                direct=False,
            ),
        },
        "direct_path": {
            **direct_receipt,
            "controller": direct_controller,
            "rows": _serialize_rows(
                layout=layout,
                logits=direct_logits,
                selected_tokens=direct_tokens,
                prompt_token_ids_by_condition=prompt_token_ids_by_condition,
                direct=True,
            ),
        },
    }


def _first_target_summary(arm: Mapping[str, Any], path: str) -> Mapping[str, Any]:
    row = next(row for row in arm[path]["rows"] if row["is_target_recipient"])
    return row["logit_summary"]


def _all_target_vector_hashes(arm: Mapping[str, Any], path: str) -> list[str]:
    return [
        str(row["logit_summary"]["coordinate_raw_logits_float32_sha256"])
        for row in arm[path]["rows"]
        if row["is_target_recipient"]
    ]


def classify_recovery(recovery_fraction: float, *, closer_to: str) -> str:
    if recovery_fraction >= 0.9 and closer_to == "Natural Single Target":
        return "vision-output ownership"
    if recovery_fraction <= 0.1 and closer_to == "Natural Homogeneous Batch Four":
        return "post-vision ownership"
    return "mixed or unresolved ownership"


def build_causal_summary(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for path in ("cached_path", "direct_path"):
        single = _first_target_summary(arms[NATURAL_SINGLE_TARGET_ARM], path)
        homogeneous = _first_target_summary(
            arms[NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM], path
        )
        replayed = _first_target_summary(
            arms[SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM],
            path,
        )
        natural_separation = compare_coordinate_summaries(single, homogeneous)
        replay_to_single = compare_coordinate_summaries(single, replayed)
        replay_to_homogeneous = compare_coordinate_summaries(homogeneous, replayed)
        denominator = float(
            natural_separation["centered_root_mean_square_difference_float32"]
        )
        replay_single_distance = float(
            replay_to_single["centered_root_mean_square_difference_float32"]
        )
        recovery_fraction = (
            float(1.0 - replay_single_distance / denominator)
            if denominator > 0.0
            else float("nan")
        )
        homogeneous_distance = float(
            replay_to_homogeneous["centered_root_mean_square_difference_float32"]
        )
        closer_to = (
            NATURAL_SINGLE_TARGET_ARM
            if replay_single_distance < homogeneous_distance
            else NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM
        )
        output[path] = {
            "natural_single_to_homogeneous": natural_separation,
            "replayed_batch_four_to_natural_single": replay_to_single,
            "replayed_batch_four_to_natural_homogeneous": replay_to_homogeneous,
            "centered_root_mean_square_recovery_fraction": recovery_fraction,
            "closer_to": closer_to,
            "ownership_classification": classify_recovery(
                recovery_fraction, closer_to=closer_to
            ),
        }
    cached_classification = output["cached_path"]["ownership_classification"]
    direct_classification = output["direct_path"]["ownership_classification"]
    output["cross_path_verdict"] = (
        cached_classification
        if cached_classification == direct_classification
        else "cached-direct ownership disagreement"
    )
    return output


def build_trust_gate(
    *,
    arms: Mapping[str, Mapping[str, Any]],
    homogeneous_feature_equality: Mapping[str, Any],
) -> dict[str, Any]:
    natural_single_hashes = _all_target_vector_hashes(
        arms[NATURAL_SINGLE_TARGET_ARM], "cached_path"
    )
    natural_homogeneous_hashes = _all_target_vector_hashes(
        arms[NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM], "cached_path"
    )
    checks = {
        "natural_single_matches_frozen_vector": natural_single_hashes
        == [EXPECTED_CACHED_SINGLE_VECTOR_SHA256],
        "natural_homogeneous_matches_frozen_vector": bool(
            natural_homogeneous_hashes
        )
        and set(natural_homogeneous_hashes)
        == {EXPECTED_CACHED_HOMOGENEOUS_VECTOR_SHA256},
        "homogeneous_primary_features_equal": bool(
            homogeneous_feature_equality["primary_all_equal"]
        ),
        "homogeneous_deepstack_features_equal": bool(
            homogeneous_feature_equality["deepstack_all_equal"]
        ),
    }
    for path in ("cached_path", "direct_path"):
        checks[f"replayed_single_noop_matches_natural_single_{path}"] = (
            _all_target_vector_hashes(arms[REPLAYED_SINGLE_TARGET_ARM], path)
            == _all_target_vector_hashes(arms[NATURAL_SINGLE_TARGET_ARM], path)
        )
        for arm_name in (
            REPLAYED_SINGLE_TARGET_ARM,
            SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM,
        ):
            controller = arms[arm_name][path]["controller"]
            checks[f"{arm_name}_{path}_controller_valid"] = bool(
                controller
                and controller["feature_call_count"] == 1
                and controller["grid_mismatch_count"] == 0
                and controller["hook_restored"]
            )
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "natural_single_cached_vector_hashes": natural_single_hashes,
        "natural_homogeneous_cached_vector_hashes": natural_homogeneous_hashes,
        "failure_disposition": (
            None if all(checks.values()) else "trust_gate_failed_stop_interpretation"
        ),
    }


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    from src.config.fingerprint import sha256_json
    from src.config.inference import load_infer_config
    from src.data import load_raw_examples
    from src.inference.backend import (
        DecodeGenerationPolicy,
        HFGenerateBackend,
        _attention_implementation,
        _execution_device_identity,
        _runtime_identity,
    )
    from src.inference.image_plan import (
        materialize_image_plan_batch,
        verify_processor_model_vision_parity,
    )
    from src.inference.pipeline import (
        _processor_config,
        _template_config,
        _tokenizer_identity,
    )
    from src.inference.runtime import assemble_runtime

    source_bundle_path = args.source_bundle.expanduser().resolve(strict=True)
    source_bundle = _read_json(source_bundle_path)
    source_rows = extract_source_rows(source_bundle)
    conditions = build_condition_rows(source_rows)
    config_path = args.infer_config.expanduser().resolve(strict=True)
    source_path = args.source_jsonl.expanduser().resolve(strict=True)
    with _temporary_cwd(config_path.parents[3]):
        resolved = load_infer_config(config_path)
    raw = next(
        (
            row
            for row in load_raw_examples(source_path)
            if str(row.metadata.get("source", {}).get("image_id")) == IMAGE_ID
        ),
        None,
    )
    if raw is None:
        raise SystemExit("image-7574 source row was not found")
    runtime = assemble_runtime(resolved.config, source_gate_root=config_path.parents[3])
    qwen = runtime.qwen
    qwen.model.eval()
    verify_processor_model_vision_parity(
        processor_identity=qwen.processor_identity,
        model_config=getattr(qwen.model, "config", qwen.model),
    )
    prompt_records = {
        name: _condition_prompt(
            raw,
            _template_config(resolved.config),
            qwen.processor,
            condition=condition,
            tokenizer=qwen.tokenizer,
        )
        for name, condition in conditions.items()
    }
    target_prompt_hash = _json_hash(
        prompt_records[TARGET_ROW_WITH_TARGET_GEOMETRY].prompt_token_ids
    )
    if target_prompt_hash != FROZEN_TARGET_PROMPT_SHA256:
        raise SystemExit("coherent target prompt hash differs from the frozen recipient")

    image_plan = materialize_image_plan_batch(
        [raw],
        components=qwen,
        processor_config=_processor_config(resolved.config),
        materialize=True,
        row_indices=[0],
    )
    model_inputs = image_plan.model_inputs_by_row_id[
        prompt_records[NO_APPENDED_ROW].row_id
    ]
    generation_fingerprint = sha256_json(
        resolved.config.generation.model_dump(mode="json")
    )
    model_identity = dict(runtime.model_identity)
    tokenizer_identity = _tokenizer_identity(qwen)
    identity_continuity = _validate_identity_continuity(
        source_bundle=source_bundle,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
        source_image_sha256=_sha256_file(raw.image.path),
    )
    backend = HFGenerateBackend(
        model=qwen.model,
        tokenizer=qwen.tokenizer,
        model_identity=model_identity,
        tokenizer_identity=tokenizer_identity,
        generation_config_fingerprint=generation_fingerprint,
    )
    model_dtype = str(next(iter(qwen.model.parameters())).dtype)
    if model_dtype != "torch.bfloat16":
        raise SystemExit(f"source-lineage replay requires torch.bfloat16, got {model_dtype}")
    policy = DecodeGenerationPolicy.greedy(
        max_new_tokens=CACHED_REPLAY_MAXIMUM_NEW_TOKENS,
        repetition_penalty=1.0,
    )
    prompt_token_ids_by_condition = {
        name: list(prompt.prompt_token_ids) for name, prompt in prompt_records.items()
    }
    single_layout, homogeneous_layout = build_layouts()
    grid_thw = [int(x) for x in model_inputs["image_grid_thw"].reshape(-1, 3)[0].tolist()]
    merge_size = int(qwen.processor.image_processor.merge_size)

    arms: dict[str, dict[str, Any]] = {}
    arms[NATURAL_SINGLE_TARGET_ARM] = _score_layout(
        backend=backend,
        layout=single_layout,
        prompt_token_ids_by_condition=prompt_token_ids_by_condition,
        model_inputs=model_inputs,
        policy=policy,
        replay_features=None,
        grid_thw=grid_thw,
        merge_size=merge_size,
    )
    arms[NATURAL_HOMOGENEOUS_BATCH_FOUR_ARM] = _score_layout(
        backend=backend,
        layout=homogeneous_layout,
        prompt_token_ids_by_condition=prompt_token_ids_by_condition,
        model_inputs=model_inputs,
        policy=policy,
        replay_features=None,
        grid_thw=grid_thw,
        merge_size=merge_size,
    )

    single_features = capture_feature_bundle(qwen.model, model_inputs)
    homogeneous_capture_requests = _requests(
        layout=homogeneous_layout,
        repeat_index=0,
        prompt_token_ids_by_condition=prompt_token_ids_by_condition,
        model_inputs=model_inputs,
        policy=policy,
        direct=False,
    )
    homogeneous_capture_inputs = backend._collate_generate_inputs(
        homogeneous_capture_requests,
        device=backend._target_device(homogeneous_capture_requests),
    )
    homogeneous_features = capture_feature_bundle(qwen.model, homogeneous_capture_inputs)
    homogeneous_per_image, homogeneous_feature_equality = split_homogeneous_feature_bundle(
        homogeneous_features
    )

    arms[REPLAYED_SINGLE_TARGET_ARM] = _score_layout(
        backend=backend,
        layout=single_layout,
        prompt_token_ids_by_condition=prompt_token_ids_by_condition,
        model_inputs=model_inputs,
        policy=policy,
        replay_features=single_features,
        grid_thw=grid_thw,
        merge_size=merge_size,
    )
    arms[SINGLE_TARGET_FEATURES_REPLAYED_INTO_HOMOGENEOUS_BATCH_FOUR_ARM] = (
        _score_layout(
            backend=backend,
            layout=homogeneous_layout,
            prompt_token_ids_by_condition=prompt_token_ids_by_condition,
            model_inputs=model_inputs,
            policy=policy,
            replay_features=single_features,
            grid_thw=grid_thw,
            merge_size=merge_size,
        )
    )

    trust_gate = build_trust_gate(
        arms=arms,
        homogeneous_feature_equality=homogeneous_feature_equality,
    )
    causal_summary = build_causal_summary(arms) if trust_gate["passed"] else None
    return {
        "schema_version": "single_target_visual_feature_replay_batch_four.receipt.v1",
        "experiment_name": "Single-Target Visual-Feature Replay into Homogeneous Batch Four",
        "image_id": IMAGE_ID,
        "source_bundle": {
            "path": str(source_bundle_path),
            "sha256": _sha256_file(source_bundle_path),
        },
        "source_image": {
            "path": str(raw.image.path),
            "sha256": _sha256_file(raw.image.path),
        },
        "infer_config_path": str(config_path),
        "target_prompt_token_ids_sha256": target_prompt_hash,
        "required_natural_suffix_token_ids": list(COMMON_COORDINATE_RECIPIENT_SUFFIX),
        "execution_identity": {
            "model_dtype": model_dtype,
            "model_evaluation_mode": not bool(qwen.model.training),
            "attention_implementation": _attention_implementation(qwen.model),
            "runtime_identity": _runtime_identity(),
            "device_identity": _execution_device_identity(backend._model_device()),
            "generation_config_fingerprint": generation_fingerprint,
            "model_identity": model_identity,
            "tokenizer_identity": tokenizer_identity,
            "identity_continuity": identity_continuity,
        },
        "visual_feature_capture": {
            "single_target_executed_fingerprint": feature_bundle_fingerprint(single_features),
            "single_target_float32_fingerprint": float32_feature_bundle_fingerprint(
                single_features
            ),
            "homogeneous_executed_fingerprint": feature_bundle_fingerprint(
                homogeneous_features
            ),
            "homogeneous_float32_fingerprint": float32_feature_bundle_fingerprint(
                homogeneous_features
            ),
            "homogeneous_per_image_equality": homogeneous_feature_equality,
            "single_to_homogeneous_first_image_comparison": compare_feature_bundles(
                single_features, homogeneous_per_image[0]
            ),
        },
        "arms": arms,
        "trust_gate": trust_gate,
        "causal_summary": causal_summary,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_probe(args)
    output_root = args.output_root.expanduser().resolve()
    receipt_path = output_root / "receipt.json"
    _write_bundle_once(receipt_path, result)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "trust_gate": result["trust_gate"],
                "causal_summary": result["causal_summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["trust_gate"]["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
