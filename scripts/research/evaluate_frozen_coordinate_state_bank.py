#!/usr/bin/env python3
"""Evaluate a frozen StateBank eval split under one coordinate-only objective.

This is intentionally a one-time research evaluator rather than a training or
inference-pipeline entry point.  It replays the exact stored prompt/prefix and
candidate token ids, runs a teacher-forced Qwen forward, and applies the
``RolloutCalibrationLossRunner`` with a forced ``coordinate_boundary_only``
profile for every requested model.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import replace
import gc
import json
from pathlib import Path
import sys
from typing import Any

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from src.adapters.dora import inspect_dora_adapter_payload, setup_dora_adapter  # noqa: E402
from src.adapters.source_gates import (  # noqa: E402
    build_adapter_setup_plan,
    load_default_adapter_source_gate_evidence,
)
from src.config.fingerprint import sha256_json  # noqa: E402
from src.config.loader import load_train_config  # noqa: E402
from src.losses import build_token_vocabulary_groups  # noqa: E402
from src.qwen import (  # noqa: E402
    build_default_special_token_selection,
    install_special_token_embedding_deltas,
    load_special_token_embedding_deltas,
)
from src.qwen.runtime_loading import (  # noqa: E402
    QwenLoadOptions,
    load_qwen_components_from_options,
)
from src.qwen.special_token_embeddings import (  # noqa: E402
    inspect_special_token_embedding_delta_payload,
)
from src.rollout_calibration import (  # noqa: E402
    CheckpointIdentity,
    load_state_bank,
    load_state_bank_manifest_binding,
    plan_calibration_micro_steps,
    validate_state_bank_token_identity,
)
from src.training.pipeline import (  # noqa: E402
    _attach_image_processors_to_micro_steps,
    _image_token_id,
)
from src.training.rollout_calibration import (  # noqa: E402
    CalibrationTokenSequence,
    RolloutCalibrationLossRunner,
    rollout_calibration_loss_context,
)
from src.training.supervised_trainer import (  # noqa: E402
    SupervisedMicroStep,
    _default_qwen_forward,
)


EVAL_SPLIT = "eval"
EVALUATOR_SCHEMA_VERSION = "coordexp.research.frozen_coordinate_state_bank_eval.v1"
COORDINATE_TERM = "rollout_coordinate_boundary"
GATE_TERM = "rollout_site_token_type_gate"
EVALUATION_DTYPE = "fp32"
EVALUATION_ATTENTION = "sdpa"


class EvaluationArgumentError(ValueError):
    """Raised when a frozen-bank evaluator argument is unsafe or incomplete."""


def _required_path(value: str | Path, *, name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise EvaluationArgumentError(f"{name} does not exist: {path}")
    return path


def resolve_bank_manifest(bank: str | Path) -> Path:
    """Resolve a bank directory or manifest path to the immutable manifest."""

    path = Path(bank).expanduser().resolve()
    candidates = (
        path,
        path / "manifest.json",
        path / "state-bank" / "manifest.json",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "manifest.json":
            return candidate
    raise EvaluationArgumentError(
        "bank must be a StateBank manifest or directory containing manifest.json: "
        f"{path}"
    )


def resolve_checkpoint_payloads(checkpoint: str | Path) -> tuple[Path, Path, Path]:
    """Resolve ``checkpoint/{adapter,special_token_embeddings}`` payloads."""

    root = _required_path(checkpoint, name="checkpoint")
    if root.is_file():
        raise EvaluationArgumentError(f"checkpoint must be a directory: {root}")
    adapter = root / "adapter"
    embedding_delta = root / "special_token_embeddings"
    if not adapter.is_dir() or not embedding_delta.is_dir():
        # Being permissive about a direct adapter payload is useful for small
        # replay fixtures while keeping the checkpoint identity explicit.
        if (root / "adapter_config.json").is_file():
            adapter = root
        if (root / "special_token_embeddings.json").is_file():
            embedding_delta = root
    if not adapter.is_dir() or not embedding_delta.is_dir():
        raise EvaluationArgumentError(
            "checkpoint must contain adapter/ and special_token_embeddings/ payloads: "
            f"{root}"
        )
    return root, adapter, embedding_delta


def validate_eval_event_set(
    event_ids: Sequence[str],
    *,
    expected_count: int | None = None,
    expected_event_ids: Sequence[str] | None = None,
    split: str = EVAL_SPLIT,
) -> tuple[str, ...]:
    """Fail fast unless an event list is exactly the frozen split set.

    The StateBank loader validates the immutable manifest/records join.  This
    small pure guard protects callers that pass a filtered event list after
    loading and gives tests a no-runtime validation seam.
    """

    if split != EVAL_SPLIT:
        raise EvaluationArgumentError(
            f"frozen coordinate evaluator only accepts split={EVAL_SPLIT!r}, got {split!r}"
        )
    if any(not isinstance(event_id, str) for event_id in event_ids):
        raise EvaluationArgumentError("frozen eval event ids must be strings")
    checked = tuple(event_ids)
    if any(not event_id for event_id in checked):
        raise EvaluationArgumentError("frozen eval event ids must be non-empty strings")
    if expected_count is None and expected_event_ids is None:
        raise EvaluationArgumentError(
            "frozen eval validation requires an expected count or event-id set"
        )
    if expected_count is not None and len(checked) != int(expected_count):
        raise EvaluationArgumentError(
            "event set does not exactly match the frozen eval split count: "
            f"expected {expected_count}, got {len(checked)}"
        )
    if len(set(checked)) != len(checked):
        raise EvaluationArgumentError("frozen eval event set contains duplicate event ids")
    canonical = tuple(sorted(checked))
    if expected_event_ids is not None:
        expected = tuple(sorted(expected_event_ids))
        if canonical != expected:
            raise EvaluationArgumentError(
                "event set does not exactly match the frozen eval event-id set"
            )
    return canonical


def aggregate_coordinate_evaluation(
    event_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate pure per-event observations without touching model/runtime code."""

    rows = tuple(dict(row) for row in event_rows)
    if not rows:
        raise EvaluationArgumentError("cannot aggregate an empty frozen eval result")
    event_ids = validate_eval_event_set(
        [row.get("event_id", "") for row in rows],
        expected_count=len(rows),
    )
    if tuple(sorted(str(row.get("event_id", "")) for row in rows)) != event_ids:
        raise EvaluationArgumentError("per-event event ids are not stable strings")

    def _number(row: Mapping[str, Any], key: str) -> float:
        value = row.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationArgumentError(f"event row {key!r} must be numeric")
        number = float(value)
        if not torch.isfinite(torch.tensor(number, dtype=torch.float32)).item():
            raise EvaluationArgumentError(f"event row {key!r} must be finite")
        return number

    margins = [_number(row, "coordinate_target_margin") for row in rows]
    legal_mass = [_number(row, "legal_coordinate_token_mass") for row in rows]
    raw_loss_names = {COORDINATE_TERM, GATE_TERM, "total"}
    raw_losses: dict[str, dict[str, float]] = {}
    for name in sorted(raw_loss_names):
        values: list[float] = []
        for row in rows:
            losses = row.get("raw_losses")
            if not isinstance(losses, Mapping):
                raise EvaluationArgumentError("event row raw_losses must be an object")
            value = losses.get(name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise EvaluationArgumentError(f"event row raw_losses[{name!r}] must be numeric")
            value_float = float(value)
            if not torch.isfinite(torch.tensor(value_float, dtype=torch.float32)).item():
                raise EvaluationArgumentError(f"event row raw_losses[{name!r}] must be finite")
            values.append(value_float)
        raw_losses[name] = {
            "sum": float(sum(values)),
            "mean": float(sum(values) / len(values)),
        }

    return {
        "event_count": len(rows),
        "event_ids": list(event_ids),
        "coordinate_target_margin": float(sum(margins) / len(margins)),
        "coordinate_target_margin_sum": float(sum(margins)),
        "legal_coordinate_token_mass": float(sum(legal_mass) / len(legal_mass)),
        "legal_coordinate_token_mass_sum": float(sum(legal_mass)),
        "raw_losses": raw_losses,
        "math_dtype": "float32",
    }


def validate_evaluator_arguments(
    *,
    bank: str | Path,
    source_checkpoint: str | Path,
    config: str | Path,
    output: str | Path,
    target_checkpoint: str | Path | None = None,
) -> dict[str, Path]:
    """Pure path/argument validation used by the CLI and focused tests."""

    manifest = resolve_bank_manifest(bank)
    source_root, _, _ = resolve_checkpoint_payloads(source_checkpoint)
    config_path = _required_path(config, name="config")
    output_path = Path(output).expanduser().resolve()
    if output_path.exists() and output_path.is_dir():
        raise EvaluationArgumentError(f"output must be a file path: {output_path}")
    result = {
        "bank_manifest": manifest,
        "source_checkpoint": source_root,
        "config": config_path,
        "output": output_path,
    }
    if target_checkpoint is not None:
        target_root, _, _ = resolve_checkpoint_payloads(target_checkpoint)
        result["target_checkpoint"] = target_root
    return result


def _checkpoint_identity(
    *,
    components: Any,
    special_token_selection: Any,
    adapter_path: Path,
    embedding_delta_path: Path,
) -> CheckpointIdentity:
    adapter_identity = inspect_dora_adapter_payload(
        adapter_path,
        expected_base_model_path=components.base_model_path,
    )
    embedding_identity = inspect_special_token_embedding_delta_payload(
        embedding_delta_path,
        expected_base_model_path=components.base_model_path,
        expected_base_config_sha256=components.base_config_sha256,
        expected_tokenizer_sha256=components.tokenizer_sha256,
    )
    return CheckpointIdentity(
        adapter_fingerprint=str(adapter_identity["fingerprint"]),
        embedding_delta_fingerprint=str(embedding_identity["fingerprint"]),
        base_config_sha256=str(components.base_config_sha256),
        tokenizer_sha256=str(components.tokenizer_sha256),
        token_identity_sha256=sha256_json(components.token_identity.to_artifact_dict()),
        special_token_identity_sha256=sha256_json(
            special_token_selection.to_artifact_dict()
        ),
        processor_identity_sha256=sha256_json(
            components.processor_identity.to_artifact_dict()
        ),
    )


def _load_trusted_bank(
    *,
    manifest_path: Path,
    source_identity: CheckpointIdentity,
    token_identity: Any,
) -> Any:
    binding = load_state_bank_manifest_binding(manifest_path)
    bank = load_state_bank(
        manifest_path,
        expected_source_checkpoint=source_identity,
        expected_prompt_identity_sha256=binding.prompt_identity_sha256,
    )
    validate_state_bank_token_identity(bank, token_identity)
    events = bank.records_for_split(EVAL_SPLIT)
    expected_count = int(bank.manifest.split_counts.get(EVAL_SPLIT, -1))
    expected_event_ids = tuple(
        event.event_id for event in bank.records if event.split == EVAL_SPLIT
    )
    validate_eval_event_set(
        [event.event_id for event in events],
        expected_count=expected_count,
        expected_event_ids=expected_event_ids,
    )
    if not events or any(not event.coordinate_boundary_eligible for event in events):
        raise EvaluationArgumentError(
            "frozen eval split must contain only coordinate-boundary-eligible events"
        )
    return bank


def _build_eval_micro_steps(*, config: Any, components: Any, bank: Any) -> tuple[SupervisedMicroStep, ...]:
    planned = plan_calibration_micro_steps(
        bank,
        split=EVAL_SPLIT,
        components=components,
        processor_config=config.model.processor,
        global_max_length=config.packing.global_max_length,
        image_token_id=_image_token_id(components),
    )
    coordinate_token_ids = tuple(int(value) for value in components.token_identity.coordinate_token_ids)
    steps = tuple(
        SupervisedMicroStep(
            pack=item.pack,
            encoded_examples=item.replay_segments,
            position_inputs=item.position_inputs,
            token_sequence=CalibrationTokenSequence(pack_index=item.pack.pack_index),
            vocab_groups=build_token_vocabulary_groups(
                components.token_identity,
                tokenizer=components.tokenizer,
            ),
            metadata={"coordinate_token_ids": coordinate_token_ids},
            expected_vocab_size=components.token_identity.tokenizer_vocab_size,
            calibration_metadata=item.calibration_metadata,
        )
        for item in planned
    )
    steps = _attach_image_processors_to_micro_steps(
        steps,
        image_processor=getattr(components.processor, "image_processor", None),
    )
    # This evaluator deliberately uses SDPA; FA2 branch evidence from a
    # training config must never leak into this replay.
    return tuple(
        replace(
            step,
            fa2_branch_evidence=None,
            capture_fa2_branch=False,
            require_fa2_branch_proof=False,
            fa2_branch_proof_policy="disabled",
        )
        for step in steps
    )


def _load_evaluation_components(config: Any, *, load_model: bool) -> Any:
    """Load every compared model with one explicit fp32/SDPA contract."""

    return load_qwen_components_from_options(
        QwenLoadOptions(
            base_model=config.model.base_model,
            dtype=EVALUATION_DTYPE,
            attn_implementation=EVALUATION_ATTENTION,
            patch_embed_linearization=config.model.runtime_patches.patch_embed_linearization,
            load_model=load_model,
        )
    )


def _coordinate_runner(config: Any) -> RolloutCalibrationLossRunner:
    calibration = config.rollout_calibration
    if calibration is None:
        raise EvaluationArgumentError("config must declare rollout_calibration")
    gate = config.losses.protected.rollout_site_token_type_gate
    # A gate-only training profile still needs to be scored against the same
    # coordinate-boundary objective for this evaluator.
    coordinate_weight = max(1.0, float(calibration.coordinate_boundary.weight))
    return RolloutCalibrationLossRunner(
        profile="coordinate_boundary_only",
        entity_weight=0.0,
        entity_margin=float(calibration.entity_transition.margin),
        entity_smooth_max_temperature=float(
            calibration.entity_transition.smooth_max_temperature
        ),
        coordinate_weight=coordinate_weight,
        coordinate_margin=float(calibration.coordinate_boundary.margin),
        gate_weight=float(gate.weight),
    )


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except (AttributeError, StopIteration) as exc:
        raise EvaluationArgumentError("target model exposes no parameters") from exc


def _load_target_model(
    *,
    config: Any,
    checkpoint_root: Path,
    components: Any,
    selection: Any,
    adapter_gate: Any,
    embedding_gate: Any,
    device: str | torch.device,
) -> tuple[Any, dict[str, Any]]:
    _, adapter_path, embedding_delta_path = resolve_checkpoint_payloads(checkpoint_root)
    adapter_config = config.adapter.model_copy(
        update={
            "seed_mode": "load_existing",
            "path": str(adapter_path),
            "source_adapter_path": None,
            "repaired_embedding_payload_path": None,
        }
    )
    plan = build_adapter_setup_plan(
        adapter_config,
        adapter_gate,
        base_model_path=components.base_model_path,
    )
    adapter_result = setup_dora_adapter(components.model, plan)
    special_result = install_special_token_embedding_deltas(
        adapter_result.model,
        selection,
        source_gate=embedding_gate,
    )
    load_special_token_embedding_deltas(
        special_result,
        embedding_delta_path,
        expected_base_model_path=components.base_model_path,
        expected_base_config_sha256=components.base_config_sha256,
        expected_tokenizer_sha256=components.tokenizer_sha256,
    )
    model = special_result.model
    model.to(device)
    model.eval()
    identity = _checkpoint_identity(
        components=components,
        special_token_selection=selection,
        adapter_path=adapter_path,
        embedding_delta_path=embedding_delta_path,
    )
    return model, {
        "checkpoint_path": str(checkpoint_root),
        "adapter_path": str(adapter_path),
        "special_token_embedding_delta_path": str(embedding_delta_path),
        "checkpoint": identity.to_artifact_dict(),
        "model_dtype": str(next(model.parameters()).dtype).replace("torch.", ""),
        "base_model_path": str(components.base_model_path),
        "device": str(device),
        "evaluation_dtype": EVALUATION_DTYPE,
        "evaluation_attention": EVALUATION_ATTENTION,
    }


def _evaluate_model(
    *,
    model: Any,
    model_provenance: Mapping[str, Any],
    micro_steps: Sequence[SupervisedMicroStep],
    runner: RolloutCalibrationLossRunner,
) -> dict[str, Any]:
    device = _model_device(model)
    steps = tuple(
        step if step.forward_device is not None else replace(step, forward_device=device)
        for step in micro_steps
    )
    plan = runner.prepare_planned_step(steps, world_size=1, rank=0)
    artifacts: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for index, micro_step in enumerate(steps):
            forward_result = _default_qwen_forward(model, micro_step)
            context = rollout_calibration_loss_context(micro_step, forward_result)
            bundle = runner.compute_micro_step(
                context,
                plan,
                local_micro_step_index=index,
            )
            artifact = bundle.to_artifact_dict()
            artifacts.append(artifact)
            coordinate = bundle.term_by_name(COORDINATE_TERM)
            gate = bundle.term_by_name(GATE_TERM)
            coordinate_raw = float(
                coordinate.raw_loss.detach().float().item()
            ) * float(plan.denominators[COORDINATE_TERM].eligible_segment_count)
            gate_raw = float(gate.raw_loss.detach().float().item()) * float(
                plan.denominators[GATE_TERM].eligible_segment_count
            )
            event_rows.append(
                {
                    "event_id": context.metadata.event_id,
                    "image_id": context.metadata.image_id,
                    "split": context.metadata.split,
                    "coordinate_target_margin": float(
                        coordinate.diagnostics["target_margin"]
                    ),
                    "legal_coordinate_token_mass": float(gate.diagnostics["legal_mass"]),
                    "raw_losses": {
                        # Unnormalized one-event raw losses are the stable
                        # quantities to compare across models.  The runner's
                        # planned-step denominator contribution is retained
                        # separately below.
                        COORDINATE_TERM: coordinate_raw,
                        GATE_TERM: gate_raw,
                        "total": coordinate_raw + gate_raw,
                    },
                    "normalized_raw_losses": {
                        COORDINATE_TERM: float(
                            coordinate.raw_loss.detach().float().item()
                        ),
                        GATE_TERM: float(gate.raw_loss.detach().float().item()),
                        "total": float(bundle.total_loss.detach().float().item()),
                    },
                    "loss_bundle": artifact,
                    "model": dict(model_provenance),
                    "math_dtype": "float32",
                    "forward_dtype": str(context.logits.dtype).replace("torch.", ""),
                }
            )
            del context, forward_result, bundle
    aggregate_loss = runner.finalize_planned_step(artifacts, plan)
    aggregate_observation = aggregate_coordinate_evaluation(event_rows)
    aggregate_observation["runner_loss"] = aggregate_loss
    return {
        "model": dict(model_provenance),
        "event_rows": event_rows,
        "aggregate": aggregate_observation,
    }


def run_evaluation(arguments: argparse.Namespace) -> dict[str, Any]:
    paths = validate_evaluator_arguments(
        bank=arguments.bank,
        source_checkpoint=arguments.source_checkpoint,
        config=arguments.config,
        output=arguments.output,
        target_checkpoint=arguments.target_checkpoint,
    )
    resolved_config = load_train_config(paths["config"])
    config = resolved_config.config
    device = torch.device(getattr(arguments, "device", "cuda:0"))
    if config.adapter is None:
        raise EvaluationArgumentError("config must declare a DoRA adapter")

    # Load processor/token identities without an executable model first.  This
    # lets us bind the trusted StateBank before allocating a target model.
    components = _load_evaluation_components(config, load_model=False)
    selection = build_default_special_token_selection(
        config.model.special_token_embeddings,
        components.token_identity,
    )
    _, source_adapter_path, source_embedding_path = resolve_checkpoint_payloads(
        paths["source_checkpoint"]
    )
    source_identity = _checkpoint_identity(
        components=components,
        special_token_selection=selection,
        adapter_path=source_adapter_path,
        embedding_delta_path=source_embedding_path,
    )
    bank = _load_trusted_bank(
        manifest_path=paths["bank_manifest"],
        source_identity=source_identity,
        token_identity=components.token_identity,
    )
    micro_steps = _build_eval_micro_steps(
        config=config,
        components=components,
        bank=bank,
    )
    adapter_gate = load_default_adapter_source_gate_evidence(REPOSITORY_ROOT)
    from src.qwen.special_token_embeddings import (  # noqa: PLC0415
        load_default_special_token_embedding_source_gate_evidence,
    )

    embedding_gate = load_default_special_token_embedding_source_gate_evidence(
        REPOSITORY_ROOT
    )
    runner = _coordinate_runner(config)
    checkpoint_roots: list[tuple[str, Path]] = [("source", paths["source_checkpoint"])]
    target_path = paths.get("target_checkpoint")
    if target_path is not None:
        # An explicit target invocation is one-model-only.  The source path is
        # still required above to bind and report the trusted bank provenance,
        # but is not redundantly evaluated in the same process.
        checkpoint_roots = [
            (str(getattr(arguments, "model_label", "target")), target_path)
        ]

    model_results: list[dict[str, Any]] = []
    for label, checkpoint_root in checkpoint_roots:
        model_components = _load_evaluation_components(config, load_model=True)
        model, provenance = _load_target_model(
            config=config,
            checkpoint_root=checkpoint_root,
            components=model_components,
            selection=selection,
            adapter_gate=adapter_gate,
            embedding_gate=embedding_gate,
            device=device,
        )
        provenance = {
            **provenance,
            "label": label,
            "config_path": str(paths["config"]),
            "config_fingerprint": resolved_config.fingerprint,
            "bank_source_checkpoint": bank.manifest.source_checkpoint.to_artifact_dict(),
            "bank_source_checkpoint_id": bank.manifest.source_checkpoint_id,
            "bank_prompt_identity_sha256": bank.manifest.prompt_identity_sha256,
            "evaluation_dtype": EVALUATION_DTYPE,
            "evaluation_attention": EVALUATION_ATTENTION,
            "evaluation_fa2_branch_proof": "disabled",
        }
        # The target model owns the same replay contract but must not mutate the
        # bank-source identity in the emitted provenance.
        result = _evaluate_model(
            model=model,
            model_provenance=provenance,
            micro_steps=micro_steps,
            runner=runner,
        )
        model_results.append(result)
        del model, model_components
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload = {
        "schema_version": EVALUATOR_SCHEMA_VERSION,
        "split": EVAL_SPLIT,
        "objective": {
            "profile": runner.profile,
            "coordinate_weight": runner.coordinate_weight,
            "coordinate_margin": runner.coordinate_margin,
            "gate_weight": runner.gate_weight,
            "math_dtype": "float32",
        },
        "evaluation_runtime": {
            "dtype": EVALUATION_DTYPE,
            "attention": EVALUATION_ATTENTION,
            "fa2_branch_proof": "disabled",
            "device": str(device),
        },
        "bank": {
            "manifest_path": str(paths["bank_manifest"]),
            "manifest": bank.manifest.to_artifact_dict(),
            "validation": bank.validation_receipt.to_artifact_dict(),
            "eval_event_ids": [event.event_id for event in bank.records_for_split(EVAL_SPLIT)],
        },
        "config": {
            "path": str(paths["config"]),
            "fingerprint": resolved_config.fingerprint,
        },
        "models": model_results,
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True)
    paths["output"].write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--source-checkpoint", required=True, type=Path)
    parser.add_argument(
        "--target-checkpoint",
        type=Path,
        default=None,
        help="optional Smoke-B step-2 checkpoint; defaults to the source checkpoint",
    )
    parser.add_argument(
        "--model-label",
        default="target",
        help="label for an explicit target checkpoint (default: target)",
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="device for one composed model replay (default: cuda:0)",
    )
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    payload = run_evaluation(arguments)
    print(
        json.dumps(
            {
                "output": str(Path(arguments.output).expanduser().resolve()),
                "schema_version": payload["schema_version"],
                "event_count": len(payload["bank"]["eval_event_ids"]),
                "model_count": len(payload["models"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
