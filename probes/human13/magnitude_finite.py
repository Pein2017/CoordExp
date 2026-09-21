#!/usr/bin/env python3
"""Direct finite N2/N4/N13 canonical overfit of exact Source DoRA magnitudes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time
from typing import Any

import torch
from torch import Tensor, nn
from src.qwen.inspection import CaptureInputs
from src.losses import aligned_token_logprobs


from probes.human13 import output_qp as same_panel
from probes.human13.magnitude_qp import (
    MagnitudeSurface,
    MechanicalInvalid,
    _language_module,
    _maximum_abs_difference,
    _tensor_sha256,
    bind_magnitude_surface,
    validate_trainable_surface,
)
from src.adapters.dora import inspect_dora_adapter_payload
from src.common.errors import RuntimeContractError
from src.qwen.special_token_embeddings import SelectedDeltaOutputHead


IMAGE_IDS = same_panel.N2_IMAGE_IDS
OWNER_COUNT = same_panel.N2_OWNER_COUNT
DECISION_COUNT = same_panel.N2_DECISION_STATE_COUNT
MARGIN = same_panel.MARGIN
CERTIFICATE_TOLERANCE = same_panel.CERTIFICATE_TOLERANCE
CANDIDATE_NAME = "human13-n2-dora-magnitude-finite.safetensors"
RECEIPT_NAME = "human13-n2-dora-magnitude-finite.json"
SOURCE_ADAPTER = same_panel.SOURCE_CHECKPOINT / "adapter"
ADAPTER_TENSOR_NAME = "adapter_model.safetensors"
MATERIALIZATION_RECEIPT_NAME = "human13-n2-materialization.json"
CANDIDATE_TENSOR_COUNT = 196
EXPECTED_CANDIDATE_SHA256 = (
    "9c58350fb6e755901dcd70c6ea68c54405e6f90f080d58874c216788ec314e3c"
)
STAGE_NAMES = ("n2", "n4", "n13")
CANDIDATE_SCHEMA = "human13_dora_magnitude_finite_candidate.v2"
PRODUCER_MODULE = "probes.human13.magnitude_finite"


def _stage(stage: str) -> tuple[str, same_panel.StageSpec]:
    if stage not in STAGE_NAMES:
        raise MechanicalInvalid(f"unsupported finite stage: {stage}")
    label = stage.upper()
    return label, same_panel.STAGES[label]


def _candidate_name(stage: str) -> str:
    return f"human13-{stage}-dora-magnitude-finite.safetensors"


def _receipt_name(stage: str) -> str:
    return f"human13-{stage}-dora-magnitude-finite.json"


def _materialization_receipt_name(stage: str) -> str:
    return f"human13-{stage}-materialization.json"


@dataclass(frozen=True)
class CapturedSpecimen:
    image_id: int
    language_args: tuple[Any, ...] = field(repr=False)
    language_kwargs: Mapping[str, Any] = field(repr=False)
    decision_positions: Tensor = field(repr=False)
    target_token_ids: Tensor = field(repr=False)
    route_receipt: Mapping[str, Any]

    @property
    def decision_count(self) -> int:
        return int(self.target_token_ids.numel())


def _forward_logits(
    language: nn.Module, head: nn.Module, specimen: CapturedSpecimen
) -> Tensor:
    output = language(*specimen.language_args, **specimen.language_kwargs)
    hidden = getattr(output, "last_hidden_state", None)
    if not isinstance(hidden, Tensor) or hidden.ndim != 3 or hidden.shape[0] != 1:
        raise MechanicalInvalid("language model has no canonical last hidden state")
    states = hidden[0].index_select(0, specimen.decision_positions)
    logits = head(states.unsqueeze(0))[0]
    head_weight = getattr(head, "weight", None)
    if not isinstance(head_weight, Tensor) or head_weight.ndim != 2:
        raise MechanicalInvalid("actual output head has no matrix weight")
    expected = (specimen.decision_count, int(head_weight.shape[0]))
    if tuple(logits.shape) != expected or not bool(torch.isfinite(logits).all()):
        raise MechanicalInvalid("actual output-head finite logit layout differs")
    return logits


def scan_full_vocabulary_margin(
    logits: Tensor, target_token_ids: Tensor
) -> dict[str, Any]:
    """Return the exact worst non-target competitor from a full logit matrix."""

    if logits.ndim != 2 or logits.shape[0] != target_token_ids.numel():
        raise MechanicalInvalid("full-vocabulary margin layout differs")
    if logits.shape[1] < 2 or not bool(torch.isfinite(logits).all()):
        raise MechanicalInvalid("full-vocabulary margin input is invalid")
    targets = target_token_ids.to(device=logits.device, dtype=torch.long)
    if bool(torch.any(targets < 0)) or bool(torch.any(targets >= logits.shape[1])):
        raise MechanicalInvalid("target token is outside the output vocabulary")
    top_values, top_ids = torch.topk(logits.float(), k=2, dim=-1)
    target_is_first = top_ids[:, 0] == targets
    competitor_ids = torch.where(target_is_first, top_ids[:, 1], top_ids[:, 0])
    competitor_logits = torch.where(target_is_first, top_values[:, 1], top_values[:, 0])
    target_logits = logits.float().gather(1, targets[:, None]).squeeze(1)
    margins = target_logits - competitor_logits
    worst_position = int(torch.argmin(margins).item())
    return {
        "minimum_margin": float(margins[worst_position].item()),
        "worst_position": worst_position,
        "worst_target_token_id": int(targets[worst_position].item()),
        "worst_competitor_token_id": int(competitor_ids[worst_position].item()),
        "decision_count": int(targets.numel()),
        "vocabulary_width": int(logits.shape[1]),
        "full_vocabulary_exhaustive": True,
    }


def _load_finite_candidate_receipt(
    path: Path, *, stage: str = "n2"
) -> tuple[dict[str, Any], Path]:
    label, spec = _stage(stage)
    try:
        receipt = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MechanicalInvalid(
            "finite candidate receipt is missing or invalid"
        ) from exc
    claimed_content_sha256 = receipt.get("content_sha256")
    content = {key: value for key, value in receipt.items() if key != "content_sha256"}
    if (
        not isinstance(claimed_content_sha256, str)
        or same_panel.sha256_json(content) != claimed_content_sha256
    ):
        raise MechanicalInvalid("finite candidate receipt content hash differs")
    if receipt.get("disposition") != "FINITE_MARGIN_PASS":
        raise MechanicalInvalid("finite candidate receipt is not FINITE_MARGIN_PASS")
    candidate_path = Path(str(receipt.get("candidate_path", ""))).resolve()
    try:
        candidate_sha256 = same_panel.sha256_file(candidate_path)
    except OSError as exc:
        raise MechanicalInvalid(
            "finite candidate payload is missing or unreadable"
        ) from exc
    schema = receipt.get("schema_version")
    new_producer = schema == CANDIDATE_SCHEMA
    if new_producer:
        producer = receipt.get("producer")
        digest = producer.get("source_sha256") if isinstance(producer, dict) else None
        if (not isinstance(producer, dict) or producer.get("module") != PRODUCER_MODULE
                or not isinstance(digest, str) or len(digest) != 64
                or any(char not in "0123456789abcdef" for char in digest)):
            raise MechanicalInvalid("new finite candidate producer identity differs")
    elif schema not in (None, "human13_dora_magnitude_finite_candidate.v1"):
        raise MechanicalInvalid("unsupported finite candidate schema")
    receipt_stage = receipt.get("stage")
    legacy_n2 = not new_producer and stage == "n2" and receipt_stage is None
    if (
        (receipt_stage != label and not legacy_n2)
        or path.name != _receipt_name(stage)
        or candidate_path.name != _candidate_name(stage)
        or tuple(receipt.get("image_ids", ())) != spec.image_ids
        or int(receipt.get("owner_count", -1)) != spec.owner_count
        or int(receipt.get("decision_count", -1)) != spec.decision_state_count
        or receipt.get("candidate_sha256") != candidate_sha256
        or (not new_producer and stage == "n2" and candidate_sha256 != EXPECTED_CANDIDATE_SHA256)
    ):
        raise MechanicalInvalid("finite candidate receipt binding differs")
    return receipt, candidate_path


def _materialized_adapter_tensors(
    source: Mapping[str, Tensor], candidate: Mapping[str, Tensor]
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Replace exactly the 196 saved magnitudes and preserve every A/B tensor."""

    lora_a = {key for key in source if ".lora_A." in key}
    lora_b = {key for key in source if ".lora_B." in key}
    magnitudes = {key for key in source if ".lora_magnitude_vector" in key}
    if (
        len(source) != 3 * CANDIDATE_TENSOR_COUNT
        or len(lora_a) != CANDIDATE_TENSOR_COUNT
        or len(lora_b) != CANDIDATE_TENSOR_COUNT
        or len(magnitudes) != CANDIDATE_TENSOR_COUNT
        or len(candidate) != CANDIDATE_TENSOR_COUNT
    ):
        raise MechanicalInvalid("source/candidate tensor counts differ from 196 A/B/M")
    mapped: dict[str, str] = {}
    for live_name in candidate:
        if not live_name.endswith(".lora_magnitude_vector.default.weight"):
            raise MechanicalInvalid("candidate contains a non-magnitude live key")
        saved_name = "base_model.model." + live_name.removesuffix(".default.weight")
        if saved_name in mapped.values():
            raise MechanicalInvalid("candidate magnitude mapping is not one-to-one")
        mapped[live_name] = saved_name
    if set(mapped.values()) != magnitudes:
        raise MechanicalInvalid("candidate magnitude keys do not exactly map to Source")

    result = {key: value.detach().cpu().contiguous() for key, value in source.items()}
    for live_name, saved_name in mapped.items():
        value = candidate[live_name].detach().cpu().contiguous()
        source_value = source[saved_name]
        if value.shape != source_value.shape or value.dtype != source_value.dtype:
            raise MechanicalInvalid(
                "candidate magnitude shape or dtype differs from Source"
            )
        result[saved_name] = value
    if any(not torch.equal(result[key], source[key]) for key in lora_a | lora_b):
        raise MechanicalInvalid("materialization changed a Source A/B tensor")
    return result, {
        "mapping": "base_model.model. + live_name without .default.weight",
        "mapped_magnitude_count": len(mapped),
        "source_lora_a_count": len(lora_a),
        "source_lora_b_count": len(lora_b),
        "source_magnitude_count": len(magnitudes),
        "mapping_sha256": same_panel.sha256_json(sorted(mapped.items())),
        "source_a_b_tensor_sha256": same_panel.sha256_json(
            {key: _tensor_sha256(source[key]) for key in sorted(lora_a | lora_b)}
        ),
        "materialized_a_b_tensor_sha256": same_panel.sha256_json(
            {key: _tensor_sha256(result[key]) for key in sorted(lora_a | lora_b)}
        ),
    }


def _verify_materialization_files(
    adapter_path: Path,
    materialization: Mapping[str, Any],
    *,
    stage: str,
    archive_manifest: Path | None = None,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Verify live model payloads separately from current or historical metadata."""
    payload_names = {"adapter_config.json", ADAPTER_TENSOR_NAME}
    declared = materialization.get("adapter_files")
    if not isinstance(declared, Mapping):
        raise MechanicalInvalid("materialized adapter file map is missing")
    observed = {name: same_panel.sha256_file(adapter_path / name) for name in sorted(payload_names)}
    if any(observed[name] != declared.get(name) for name in payload_names):
        raise MechanicalInvalid("materialized adapter file hashes differ")
    version = materialization.get("schema_version")
    if version == f"human13_{stage}_dora_magnitude_materialization.v2":
        if set(declared) != payload_names:
            raise MechanicalInvalid("materialized adapter payload names differ")
        metadata = materialization.get("metadata_files")
        if not isinstance(metadata, Mapping) or not set(metadata) <= {"model_card.json"}:
            raise MechanicalInvalid("materialized metadata file map differs")
        current = {name: same_panel.sha256_file(adapter_path / name) for name in metadata}
        if current != metadata:
            raise MechanicalInvalid("materialized metadata hashes differ")
        return observed, {"mode": "current_json", "files": current}
    if version == f"human13_{stage}_dora_magnitude_materialization.v1":
        if set(declared) != payload_names | {"README.md"}:
            raise MechanicalInvalid("legacy materialization file names differ")
        from src.artifacts.source_archive import SourceArchive

        manifest = archive_manifest or Path(
            "/data/CoordExp/docs/history/output-sources/2026-09-21/manifest.json"
        )
        try:
            metadata = SourceArchive(manifest).resolve(
                adapter_path / "README.md", declared["README.md"]
            )
        except (OSError, ValueError) as exc:
            raise MechanicalInvalid("retained materialization metadata is unavailable") from exc
        # This is explicitly historical metadata, not a claim that README is live.
        return observed, {"mode": "historical_markdown", "binding": metadata}
    raise MechanicalInvalid("unsupported materialization schema")


def materialize_finite_adapter(
    *,
    candidate_receipt_path: Path,
    output_adapter: Path,
    source_adapter: Path = SOURCE_ADAPTER,
    stage: str = "n2",
) -> dict[str, Any]:
    """Publish one standard unmerged adapter without loading a model."""

    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    from src.artifacts.model_card import package_model_card

    label, _spec = _stage(stage)
    candidate_receipt, candidate_path = _load_finite_candidate_receipt(
        candidate_receipt_path, stage=stage
    )
    if output_adapter.exists():
        raise FileExistsError(f"refusing to overwrite {output_adapter}")
    source_tensor_path = source_adapter / ADAPTER_TENSOR_NAME
    source_sha256 = same_panel.sha256_file(source_tensor_path)
    if (
        source_adapter.resolve() == SOURCE_ADAPTER.resolve()
        and source_sha256 != same_panel.EXPECTED_HASHES[source_tensor_path]
    ):
        raise MechanicalInvalid("Source adapter hash differs")
    source_tensors = load_file(str(source_tensor_path), device="cpu")
    candidate_tensors = load_file(str(candidate_path), device="cpu")
    tensors, mapping = _materialized_adapter_tensors(source_tensors, candidate_tensors)
    output_adapter.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_adapter.name}.", dir=output_adapter.parent)
    )
    try:
        shutil.copyfile(
            source_adapter / "adapter_config.json", temporary / "adapter_config.json"
        )
        # Cards describe a model but are not load-bearing adapter payloads.
        if (source_adapter / "model_card.json").is_file():
            shutil.copyfile(source_adapter / "model_card.json", temporary / "model_card.json")
        elif (source_adapter / "README.md").is_file():
            shutil.copyfile(source_adapter / "README.md", temporary / "README.md")
            package_model_card(temporary)
        tensor_path = temporary / ADAPTER_TENSOR_NAME
        save_file(tensors, str(tensor_path), metadata={"format": "pt"})
        with safe_open(tensor_path, framework="pt", device="cpu") as handle:
            if handle.metadata() != {"format": "pt"}:
                raise MechanicalInvalid("materialized safetensors metadata differs")
        identity = inspect_dora_adapter_payload(
            temporary, expected_base_model_path=same_panel.BASE_MODEL
        )
        saved = load_file(str(tensor_path), device="cpu")
        if set(saved) != set(tensors) or any(
            saved[key].dtype != tensors[key].dtype
            or saved[key].shape != tensors[key].shape
            or not torch.equal(saved[key], tensors[key])
            for key in tensors
        ):
            raise MechanicalInvalid("materialized adapter persistence differs")
        receipt = {
            "schema_version": f"human13_{stage}_dora_magnitude_materialization.v2",
            "status": "materialized_unmerged",
            "stage": label,
            "adapter_path": str(output_adapter.resolve()),
            "adapter_fingerprint": identity["fingerprint"],
            "adapter_files": {
                name: same_panel.sha256_file(temporary / name)
                for name in ("adapter_config.json", ADAPTER_TENSOR_NAME)
            },
            "metadata_files": {
                "model_card.json": same_panel.sha256_file(temporary / "model_card.json")
            } if (temporary / "model_card.json").is_file() else {},
            "candidate_receipt_path": str(candidate_receipt_path.resolve()),
            "candidate_receipt_sha256": same_panel.sha256_file(candidate_receipt_path),
            "candidate_sha256": candidate_receipt["candidate_sha256"],
            "source_adapter_path": str(source_adapter.resolve()),
            "source_adapter_sha256": source_sha256,
            "safetensors_metadata": {"format": "pt"},
            "unmerged": True,
            "model_load_count": 0,
            "process_id": os.getpid(),
            **mapping,
        }
        receipt_name = _materialization_receipt_name(stage)
        same_panel.immutable_json(temporary / receipt_name, receipt)
        if output_adapter.exists():
            raise FileExistsError(f"refusing to overwrite {output_adapter}")
        os.rename(temporary, output_adapter)
        return {
            **receipt,
            "receipt_path": str((output_adapter / receipt_name).resolve()),
        }
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _restore_source(surface: MagnitudeSurface, source: Sequence[Tensor]) -> None:
    with torch.no_grad():
        for parameter, value in zip(surface.parameters, source, strict=True):
            parameter.copy_(value)
    observed = tuple(_tensor_sha256(parameter) for parameter in surface.parameters)
    expected = tuple(_tensor_sha256(value) for value in source)
    if observed != expected:
        raise MechanicalInvalid("Source magnitude restoration differs")


def train_finite_candidate(
    *,
    language: nn.Module,
    head: nn.Module,
    surface: MagnitudeSurface,
    specimens: Sequence[CapturedSpecimen],
    output_dir: Path,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    check_every: int,
    seed: int,
    bindings: Mapping[str, Any] | None = None,
    stage: str = "n2",
) -> dict[str, Any]:
    """Optimize every specimen into one shared AdamW update stream, then restore Source."""

    label, spec = _stage(stage)
    if (
        steps <= 0
        or learning_rate <= 0
        or weight_decay < 0
        or not 0 <= adam_beta1 < 1
        or not 0 <= adam_beta2 < 1
        or adam_eps <= 0
        or check_every <= 0
    ):
        raise MechanicalInvalid("finite optimizer schedule is invalid")
    if tuple(item.image_id for item in specimens) != spec.image_ids:
        raise MechanicalInvalid(
            f"finite specimens differ from frozen {label} image order"
        )
    total_decisions = sum(item.decision_count for item in specimens)
    if total_decisions <= 0:
        raise MechanicalInvalid("finite specimens contain no decisions")
    candidate_path = output_dir / _candidate_name(stage)
    receipt_path = output_dir / _receipt_name(stage)
    if candidate_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to overwrite finite candidate or receipt")

    validate_trainable_surface(surface.model, surface)
    source = tuple(parameter.detach().clone() for parameter in surface.parameters)
    source_hashes = tuple(_tensor_sha256(value) for value in source)
    started = time.perf_counter()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    optimizer = torch.optim.AdamW(
        surface.parameters,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        eps=adam_eps,
        weight_decay=weight_decay,
    )
    optimizer_steps = 0
    training_forward_count = 0
    margin_forward_count = 0
    scans: list[dict[str, Any]] = []
    final_loss = float("nan")
    passed = False
    body: dict[str, Any] | None = None
    try:
        for step in range(1, steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for specimen in specimens:
                logits = _forward_logits(language, head, specimen)
                loss = (
                    -aligned_token_logprobs(
                        logits, specimen.target_token_ids.to(logits.device)
                    ).sum()
                    / total_decisions
                )
                loss_sum += float(loss.detach().item())
                loss.backward()
                training_forward_count += 1
                del logits, loss
            if any(
                parameter.grad is None or not bool(torch.isfinite(parameter.grad).all())
                for parameter in surface.parameters
            ):
                raise MechanicalInvalid("magnitude gradient is absent or nonfinite")
            optimizer.step()
            optimizer_steps += 1
            final_loss = loss_sum
            if any(
                not bool(torch.isfinite(parameter).all())
                for parameter in surface.parameters
            ):
                raise MechanicalInvalid(
                    "finite optimizer produced a nonfinite magnitude"
                )

            if step % check_every == 0 or step == steps:
                image_scans = []
                with torch.no_grad():
                    for specimen in specimens:
                        logits = _forward_logits(language, head, specimen)
                        image_scans.append(
                            {
                                "image_id": specimen.image_id,
                                **scan_full_vocabulary_margin(
                                    logits, specimen.target_token_ids
                                ),
                            }
                        )
                        margin_forward_count += 1
                        del logits
                minimum_margin = min(item["minimum_margin"] for item in image_scans)
                scans.append(
                    {
                        "step": step,
                        "minimum_margin": minimum_margin,
                        "images": image_scans,
                    }
                )
                if minimum_margin >= MARGIN - CERTIFICATE_TOLERANCE:
                    passed = True
                    break

        candidate = {
            name: parameter.detach().cpu().contiguous()
            for name, parameter in zip(surface.names, surface.parameters, strict=True)
        }
        same_panel._save_tensors_immutable(candidate_path, candidate)
        candidate_hashes = [_tensor_sha256(candidate[name]) for name in surface.names]
        deltas = [
            candidate[name].float() - source_value.detach().cpu().float()
            for name, source_value in zip(surface.names, source, strict=True)
        ]
        body = {
            "schema_version": CANDIDATE_SCHEMA,
            "producer": {"module": PRODUCER_MODULE, "source_path": str(Path(__file__).resolve()),
                         "source_sha256": same_panel.sha256_file(__file__)},
            "disposition": "FINITE_MARGIN_PASS" if passed else "FINITE_MARGIN_HOLD",
            "stage": label,
            "candidate_path": str(candidate_path.resolve()),
            "candidate_sha256": same_panel.sha256_file(candidate_path),
            "image_ids": [item.image_id for item in specimens],
            "owner_count": spec.owner_count,
            "decision_count": total_decisions,
            "route_token_counts": {
                str(item.image_id): item.decision_count for item in specimens
            },
            "routes": [dict(item.route_receipt) for item in specimens],
            "surface": surface.receipt(),
            "source_magnitude_sha256": list(source_hashes),
            "candidate_magnitude_sha256": candidate_hashes,
            "candidate_movement": {
                "magnitude_l2": sum(
                    float(torch.sum(delta.square()).item()) for delta in deltas
                )
                ** 0.5,
                "maximum_absolute_delta": max(
                    float(delta.abs().max().item()) for delta in deltas
                ),
                "moved_scalar_count": sum(
                    int(torch.count_nonzero(delta).item()) for delta in deltas
                ),
                "realized_weight_metric": ("direct_sum_frobenius_equals_magnitude_l2"),
            },
            "optimizer": {
                "name": "AdamW",
                "steps_requested": steps,
                "steps_executed": optimizer_steps,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "betas": [adam_beta1, adam_beta2],
                "eps": adam_eps,
                "check_every": check_every,
                "seed": seed,
                "shared_update": True,
                "loss_reduction": "sum_per_image_divided_by_total_decisions",
                "final_loss": final_loss,
            },
            "margin": MARGIN,
            "certificate_tolerance": CERTIFICATE_TOLERANCE,
            "early_stop_threshold": MARGIN - CERTIFICATE_TOLERANCE,
            "margin_scans": scans,
            "bindings": dict(bindings or {}),
            "measurements": {
                "elapsed_seconds_before_source_restore": time.perf_counter() - started,
                "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "peak_gpu_reserved_bytes": (
                    int(torch.cuda.max_memory_reserved())
                    if torch.cuda.is_available()
                    else 0
                ),
                "model_load_count": 1,
                "source_capture_forward_count": len(specimens),
                "source_language_replay_forward_count": len(specimens),
                "training_language_forward_count": training_forward_count,
                "margin_language_forward_count": margin_forward_count,
                "optimizer_step_count": optimizer_steps,
            },
            "source_restoration_contract": "exact hash restore in finally",
        }
    finally:
        _restore_source(surface, source)
    if body is None:
        raise MechanicalInvalid("finite candidate did not produce a result")
    body["source_restored"] = True
    body["measurements"]["elapsed_seconds_total"] = time.perf_counter() - started
    body["content_sha256"] = same_panel.sha256_json(body)
    same_panel.immutable_json(receipt_path, body)
    return {**body, "receipt_path": str(receipt_path.resolve())}


def _capture_stage(
    stage: str = "n2",
    adapter_path: Path | None = None,
) -> tuple[
    Any,
    Any,
    tuple[Any, ...],
    nn.Module,
    nn.Module,
    MagnitudeSurface,
    tuple[CapturedSpecimen, ...],
    dict[str, Any],
]:
    label, spec = _stage(stage)
    bindings = same_panel.check_bindings(load_tokenizer=False)
    context, session, selected, runtime_identity, components = same_panel._runtime_setup_images(
        spec.image_ids, adapter_path=adapter_path
    )
    try:
        panel = {int(row["image_id"]): row for row in same_panel.load_panel()}
        model = components.model
        language_name, language = _language_module(model)
        surface = bind_magnitude_surface(model, language_module_name=language_name)
        head = same_panel._output_head(model)
        if not isinstance(head, SelectedDeltaOutputHead):
            raise MechanicalInvalid("actual selected-token output wrapper is absent")
        if any(parameter.requires_grad for parameter in head.parameters()):
            raise MechanicalInvalid("actual output head is not frozen")
        specimens = []
        prompt_hashes: dict[str, str] = {}
        head_parity: dict[str, float] = {}
        language_parity: dict[str, float] = {}
        for image_id, (_example, request) in zip(spec.image_ids, selected, strict=True):
            route_ids, route_receipt = same_panel.tokenize_canonical_route(
                components.tokenizer, panel[image_id]
            )
            if len(route_ids) != spec.route_token_counts[image_id]:
                raise MechanicalInvalid(f"{label} canonical route count differs")
            model_inputs, executed_ids, positions = same_panel.prepare_decision_history(
                components, request, route_ids
            )
            with CaptureInputs(language) as language_capture, CaptureInputs(head) as head_capture:
                with torch.no_grad():
                    output = model(**model_inputs)
            if len(head_capture.args) != 1 or not isinstance(head_capture.args[0], Tensor):
                raise MechanicalInvalid("output head input capture differs")
            states = head_capture.args[0]
            if tuple(states.shape[:2]) != (1, len(route_ids)):
                raise MechanicalInvalid("captured decision-state layout differs")
            with torch.no_grad():
                replay = head(states)[0]
            difference = _maximum_abs_difference(replay, output.logits[0])
            if not torch.equal(replay, output.logits[0]):
                raise MechanicalInvalid(
                    f"actual output-head replay differs: {difference}"
                )
            targets = torch.tensor(route_ids, dtype=torch.long, device=positions.device)
            specimen = CapturedSpecimen(
                image_id=image_id,
                language_args=language_capture.args,
                language_kwargs=language_capture.kwargs,
                decision_positions=positions,
                target_token_ids=targets,
                route_receipt=route_receipt,
            )
            with torch.no_grad():
                direct_output = language(
                    *specimen.language_args, **specimen.language_kwargs
                )
                direct_hidden = getattr(direct_output, "last_hidden_state", None)
                if (
                    not isinstance(direct_hidden, Tensor)
                    or direct_hidden.ndim != 3
                    or direct_hidden.shape[0] != 1
                ):
                    raise MechanicalInvalid(
                        "language model has no canonical last hidden state"
                    )
                direct = direct_hidden[0].index_select(0, positions)
            direct_difference = _maximum_abs_difference(direct, states[0])
            if not torch.equal(direct, states[0]):
                raise MechanicalInvalid(
                    f"full-vs-language Source hidden parity differs: {direct_difference}"
                )
            specimens.append(specimen)
            prompt_hashes[str(image_id)] = same_panel.sha256_json(
                tuple(int(value) for value in executed_ids)
            )
            head_parity[str(image_id)] = difference
            language_parity[str(image_id)] = direct_difference
        if sum(item.decision_count for item in specimens) != spec.decision_state_count:
            raise MechanicalInvalid(
                f"combined {label} canonical decision count differs"
            )
        if (
            sum(len(panel[image_id]["objects"]) for image_id in spec.image_ids)
            != spec.owner_count
        ):
            raise MechanicalInvalid(f"combined {label} owner count differs")
        return (
            context,
            session,
            selected,
            language,
            head,
            surface,
            tuple(specimens),
            {
                "source_checkpoint": str(same_panel.SOURCE_CHECKPOINT),
                "stage": label,
                "image_ids": list(spec.image_ids),
                "loaded_adapter_path": str(
                    (SOURCE_ADAPTER if adapter_path is None else adapter_path).resolve()
                ),
                "source_checkpoint_adapter_sha256": same_panel.EXPECTED_HASHES[
                    same_panel.SOURCE_CHECKPOINT / "adapter/adapter_model.safetensors"
                ],
                "selected_token_delta_sha256": same_panel.EXPECTED_HASHES[
                    same_panel.SOURCE_CHECKPOINT
                    / "special_token_embeddings/special_token_embeddings.safetensors"
                ],
                "binding_receipt_sha256": same_panel.sha256_json(bindings),
                "runtime_identity": runtime_identity,
                "executed_prompt_token_ids_sha256": prompt_hashes,
                "actual_output_head_replay_max_abs_difference": head_parity,
                "full_language_source_max_abs_difference": language_parity,
                "selected_token_output_wrapper": {
                    "class": f"{type(head).__module__}.{type(head).__qualname__}",
                    "selection": head.selection.to_artifact_dict(),
                },
            },
        )
    except BaseException:
        context.__exit__(*sys.exc_info())
        raise


def _capture_n2(
    adapter_path: Path | None = None,
) -> tuple[
    Any,
    Any,
    tuple[Any, ...],
    nn.Module,
    nn.Module,
    MagnitudeSurface,
    tuple[CapturedSpecimen, ...],
    dict[str, Any],
]:
    return _capture_stage("n2", adapter_path)


def run_stage(
    *,
    stage: str = "n2",
    output_dir: Path,
    steps: int,
    learning_rate: float,
    weight_decay: float,
    adam_beta1: float,
    adam_beta2: float,
    adam_eps: float,
    check_every: int,
    seed: int,
) -> dict[str, Any]:
    context, _session, _selected, language, head, surface, specimens, bindings = (
        _capture_stage(stage)
    )
    try:
        return train_finite_candidate(
            language=language,
            head=head,
            surface=surface,
            specimens=specimens,
            output_dir=output_dir,
            steps=steps,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_eps=adam_eps,
            check_every=check_every,
            seed=seed,
            bindings=bindings,
            stage=stage,
        )
    finally:
        context.__exit__(None, None, None)


def run_n2(**kwargs: Any) -> dict[str, Any]:
    return run_stage(stage="n2", **kwargs)


def aggregate_natural_gate(
    primary: Sequence[Mapping[str, Any]],
    monitor: Sequence[Mapping[str, Any]],
    *,
    stage: str = "n2",
) -> dict[str, Any]:
    """Apply only the registered RP1.0 gate; order and RP1.10 are diagnostics."""

    label, spec = _stage(stage)
    if (
        tuple(int(item.get("image_id", -1)) for item in primary) != spec.image_ids
        or tuple(int(item.get("image_id", -1)) for item in monitor) != spec.image_ids
    ):
        raise MechanicalInvalid(
            f"natural verification differs from frozen {label} image order"
        )
    owner_counts = {
        threshold: sum(int(item["matched_owner_count"][threshold]) for item in primary)
        for threshold in ("50", "60", "80")
    }
    hard_debt = sum(
        int(item["duplicate_count"])
        + int(item["unmatched_prediction_count"])
        + int(item["malformed_count"])
        + int(bool(item["cap_debt"]))
        for item in primary
    )
    primary_order_rows = sum(
        int(item.get("natural_order_violation_row_count", 0)) for item in primary
    )
    monitor_order_rows = sum(
        int(item.get("natural_order_violation_row_count", 0)) for item in monitor
    )
    passed = (
        owner_counts["50"] == spec.owner_count
        and hard_debt == 0
        and all(bool(item["natural_eos"]) for item in primary)
    )
    return {
        "disposition": (
            f"{label}_PASS" if passed else "M_ONLY_REPLAY_PASS_NATURAL_FAIL"
        ),
        "stage": label,
        "image_ids": list(spec.image_ids),
        "owner_counts": owner_counts,
        "required_iou50_owner_count": spec.owner_count,
        "hard_debt": hard_debt,
        "all_natural_eos": all(bool(item["natural_eos"]) for item in primary),
        "natural_order_monitor": {
            "gating": False,
            "rp1_0_violation_image_count": sum(
                bool(item.get("natural_order_violation", False)) for item in primary
            ),
            "rp1_0_violation_row_count": primary_order_rows,
            "rp1_10_violation_image_count": sum(
                bool(item.get("natural_order_violation", False)) for item in monitor
            ),
            "rp1_10_violation_row_count": monitor_order_rows,
        },
        "rp1_10_monitor_gating": False,
    }


def aggregate_n2_natural_gate(
    primary: Sequence[Mapping[str, Any]],
    monitor: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return aggregate_natural_gate(primary, monitor, stage="n2")


def _validate_materialized_adapter(
    adapter_path: Path, *, stage: str = "n2"
) -> dict[str, Any]:
    from safetensors.torch import load_file

    label, _spec = _stage(stage)
    materialization_path = adapter_path / _materialization_receipt_name(stage)
    try:
        materialization = json.loads(materialization_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise MechanicalInvalid(
            "materialization receipt is missing or invalid"
        ) from exc
    if int(materialization.get("process_id", os.getpid())) == os.getpid():
        raise MechanicalInvalid("verification must run in a fresh process")
    candidate_receipt_path = Path(
        str(materialization.get("candidate_receipt_path", ""))
    )
    candidate_receipt, candidate_path = _load_finite_candidate_receipt(
        candidate_receipt_path, stage=stage
    )
    if (
        materialization.get("stage", "N2") != label
        or materialization.get("candidate_receipt_sha256")
        != same_panel.sha256_file(candidate_receipt_path)
        or materialization.get("candidate_sha256")
        != candidate_receipt["candidate_sha256"]
    ):
        raise MechanicalInvalid("materialization candidate receipt binding differs")
    identity = inspect_dora_adapter_payload(
        adapter_path, expected_base_model_path=same_panel.BASE_MODEL
    )
    if identity["fingerprint"] != materialization.get("adapter_fingerprint"):
        raise MechanicalInvalid("materialized adapter fingerprint differs")
    observed_files, metadata_evidence = _verify_materialization_files(
        adapter_path, materialization, stage=stage
    )
    source = load_file(str(SOURCE_ADAPTER / ADAPTER_TENSOR_NAME), device="cpu")
    candidate = load_file(str(candidate_path), device="cpu")
    expected, mapping = _materialized_adapter_tensors(source, candidate)
    observed = load_file(str(adapter_path / ADAPTER_TENSOR_NAME), device="cpu")
    if set(observed) != set(expected) or any(
        observed[key].dtype != expected[key].dtype
        or observed[key].shape != expected[key].shape
        or not torch.equal(observed[key], expected[key])
        for key in expected
    ):
        raise MechanicalInvalid(
            "materialized adapter differs from Source A/B plus candidate M"
        )
    return {
        "adapter_identity": identity,
        "adapter_files": observed_files,
        "metadata_verification": metadata_evidence,
        "materialization_receipt_path": str(materialization_path.resolve()),
        "materialization_receipt_sha256": same_panel.sha256_file(materialization_path),
        "materialization_process_id": int(materialization["process_id"]),
        "candidate_receipt_path": str(candidate_receipt_path.resolve()),
        "candidate_receipt_sha256": same_panel.sha256_file(candidate_receipt_path),
        "candidate_sha256": candidate_receipt["candidate_sha256"],
        "mapping": mapping,
    }


def _validate_live_adapter_readback(
    model: nn.Module, adapter_path: Path
) -> dict[str, Any]:
    from safetensors.torch import load_file

    saved = load_file(str(adapter_path / ADAPTER_TENSOR_NAME), device="cpu")
    live = {
        name: parameter
        for name, parameter in model.named_parameters()
        if any(
            marker in name
            for marker in (".lora_A.", ".lora_B.", ".lora_magnitude_vector.")
        )
    }

    def live_name(saved_name: str) -> str:
        normalized = saved_name.removeprefix("base_model.model.")
        for suffix in (
            ".lora_A.weight",
            ".lora_B.weight",
            ".lora_magnitude_vector",
        ):
            if normalized.endswith(suffix):
                return (
                    normalized.removesuffix(suffix)
                    + suffix.removesuffix(".weight")
                    + ".default.weight"
                )
        raise MechanicalInvalid("saved adapter contains an unsupported tensor key")

    mapped = {saved_name: live_name(saved_name) for saved_name in saved}
    if len(mapped) != 3 * CANDIDATE_TENSOR_COUNT or set(mapped.values()) != set(live):
        raise MechanicalInvalid(
            "cold-loaded adapter key mapping differs from saved payload"
        )
    live_hashes = {}
    for saved_name, parameter_name in mapped.items():
        saved_tensor = saved[saved_name]
        live_tensor = live[parameter_name].detach().cpu()
        if (
            live_tensor.shape != saved_tensor.shape
            or live_tensor.dtype != saved_tensor.dtype
            or not torch.equal(live_tensor, saved_tensor)
        ):
            raise MechanicalInvalid("cold-loaded adapter tensor persistence differs")
        live_hashes[parameter_name] = _tensor_sha256(live_tensor)
    return {
        "status": "exact_saved_payload_to_live_unmerged_state_match",
        "tensor_count": len(mapped),
        "live_tensor_sha256": same_panel.sha256_json(live_hashes),
        "unmerged": True,
    }


def _verify_adapter(adapter_path: Path, *, stage: str = "n2") -> dict[str, Any]:
    label, spec = _stage(stage)
    started = time.perf_counter()
    persistence = _validate_materialized_adapter(adapter_path, stage=stage)
    (
        context,
        session,
        selected,
        language,
        head,
        surface,
        specimens,
        bindings,
    ) = _capture_stage(stage, adapter_path)
    try:
        live_readback = _validate_live_adapter_readback(surface.model, adapter_path)
        canonical = []
        routes = []
        with torch.no_grad():
            for specimen in specimens:
                scan = scan_full_vocabulary_margin(
                    _forward_logits(language, head, specimen),
                    specimen.target_token_ids,
                )
                canonical.append({"image_id": specimen.image_id, **scan})
                routes.append(
                    (
                        specimen.image_id,
                        tuple(
                            int(value) for value in specimen.target_token_ids.tolist()
                        ),
                    )
                )
        if (
            min(item["minimum_margin"] for item in canonical)
            < MARGIN - CERTIFICATE_TOLERANCE
        ):
            raise MechanicalInvalid("fresh canonical readback misses registered margin")
        del specimens

        primary = []
        monitor = []
        for repetition_penalty, destination in ((1.0, primary), (1.10, monitor)):
            for (image_id, canonical_ids), (_example, request) in zip(
                routes, selected, strict=True
            ):
                result = same_panel._decode(
                    session, request, repetition_penalty=repetition_penalty
                )
                destination.append(
                    {
                        "image_id": image_id,
                        **same_panel._evaluate_result(
                            result=result,
                            image_id=image_id,
                            canonical_ids=canonical_ids,
                            backend_version=str(session.receipt.backend_version),
                            repetition_penalty=repetition_penalty,
                        ),
                    }
                )
        gate = aggregate_natural_gate(primary, monitor, stage=stage)
        return {
            "schema_version": f"human13_{stage}_dora_magnitude_acceptance.v1",
            "disposition": gate["disposition"],
            "stage": label,
            "image_ids": list(spec.image_ids),
            "owner_count": spec.owner_count,
            "decision_count": spec.decision_state_count,
            "margin": MARGIN,
            "certificate_tolerance": CERTIFICATE_TOLERANCE,
            "persistence": persistence,
            "live_readback": live_readback,
            "canonical_readback": canonical,
            "natural_rp1_0": primary,
            "natural_rp1_10_monitor": monitor,
            "natural_gate": gate,
            "bindings": bindings,
            "runtime_identity": bindings["runtime_identity"],
            "process_id": os.getpid(),
            "model_load_count": 1,
            "canonical_language_forward_count": len(routes),
            "natural_generation_count": len(primary) + len(monitor),
            "elapsed_seconds": time.perf_counter() - started,
            "peak_host_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_gpu_reserved_bytes": (
                int(torch.cuda.max_memory_reserved())
                if torch.cuda.is_available()
                else 0
            ),
        }
    finally:
        context.__exit__(None, None, None)


def verify_adapter(
    *, stage: str = "n2", adapter_path: Path, output_receipt: Path
) -> dict[str, Any]:
    label, _spec = _stage(stage)
    if output_receipt.exists():
        raise FileExistsError(f"refusing to overwrite {output_receipt}")
    try:
        receipt = _verify_adapter(adapter_path, stage=stage)
    except (
        MechanicalInvalid,
        same_panel.HoldError,
        RuntimeContractError,
        ValueError,
        KeyError,
        OSError,
    ) as exc:
        receipt = {
            "schema_version": f"human13_{stage}_dora_magnitude_acceptance.v1",
            "disposition": "MECHANICAL_INVALID",
            "stage": label,
            "adapter_path": str(adapter_path.resolve()),
            "reason": str(exc),
            "process_id": os.getpid(),
        }
    receipt["content_sha256"] = same_panel.sha256_json(receipt)
    same_panel.immutable_json(output_receipt, receipt)
    return {**receipt, "receipt_path": str(output_receipt.resolve())}


def verify_n2_adapter(*, adapter_path: Path, output_receipt: Path) -> dict[str, Any]:
    return verify_adapter(
        stage="n2", adapter_path=adapter_path, output_receipt=output_receipt
    )


def _training_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGE_NAMES, default="n2")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--adam-beta1", type=float, required=True)
    parser.add_argument("--adam-beta2", type=float, required=True)
    parser.add_argument("--adam-eps", type=float, required=True)
    parser.add_argument("--check-every", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["materialize"]:
        parser = argparse.ArgumentParser(
            description="materialize a finite N2/N4/N13 adapter"
        )
        parser.add_argument("--stage", choices=STAGE_NAMES, default="n2")
        parser.add_argument("--candidate-receipt", type=Path, required=True)
        parser.add_argument("--output-adapter", type=Path, required=True)
        args = parser.parse_args(arguments[1:])
        try:
            result = materialize_finite_adapter(
                candidate_receipt_path=args.candidate_receipt,
                output_adapter=args.output_adapter,
                stage=args.stage,
            )
        except (
            MechanicalInvalid,
            RuntimeContractError,
            FileExistsError,
            OSError,
        ) as exc:
            print(json.dumps({"disposition": "MECHANICAL_INVALID", "reason": str(exc)}))
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0
    if arguments[:1] in (["verify"], ["readback"]):
        parser = argparse.ArgumentParser(
            description="freshly verify a finite N2/N4/N13 adapter"
        )
        parser.add_argument("--stage", choices=STAGE_NAMES, default="n2")
        parser.add_argument("--adapter-path", type=Path, required=True)
        parser.add_argument("--output-receipt", type=Path, required=True)
        args = parser.parse_args(arguments[1:])
        try:
            result = verify_adapter(
                stage=args.stage,
                adapter_path=args.adapter_path,
                output_receipt=args.output_receipt,
            )
        except FileExistsError as exc:
            print(json.dumps({"disposition": "MECHANICAL_INVALID", "reason": str(exc)}))
            return 2
        print(json.dumps(result, sort_keys=True))
        return 0 if result["disposition"] == f"{args.stage.upper()}_PASS" else 2

    args = _training_parser().parse_args(arguments)
    try:
        result = run_stage(
            stage=args.stage,
            output_dir=args.output_dir,
            steps=args.steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            adam_eps=args.adam_eps,
            check_every=args.check_every,
            seed=args.seed,
        )
    except (MechanicalInvalid, same_panel.HoldError, FileExistsError) as exc:
        print(json.dumps({"disposition": "FINITE_MECHANICAL_HOLD", "reason": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["disposition"] == "FINITE_MARGIN_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
