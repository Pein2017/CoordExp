"""PEFT DoRA setup for approved adapter plans."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors.torch import safe_open
from torch import nn

from src.adapters.source_gates import AdapterSetupPlan
from src.common.errors import RuntimeContractError


DEFAULT_ADAPTER_NAME = "default"


@dataclass(frozen=True)
class DoraTargetDiscoveryReceipt:
    target_policy: str
    target_towers: tuple[str, ...]
    matched_modules: tuple[str, ...]
    counts_by_tower: dict[str, int]
    lm_head_seen: bool
    lm_head_excluded: bool

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "target_policy": self.target_policy,
            "target_towers": list(self.target_towers),
            "matched_modules": list(self.matched_modules),
            "counts_by_tower": dict(self.counts_by_tower),
            "matched_count": len(self.matched_modules),
            "lm_head_seen": self.lm_head_seen,
            "lm_head_excluded": self.lm_head_excluded,
        }


@dataclass(frozen=True)
class DoraAdapterSetupReceipt:
    mode: str
    adapter_type: str
    adapter_name: str
    adapter_path: Path | None
    base_model_path: Path | None
    target_discovery: DoraTargetDiscoveryReceipt
    peft_config: Mapping[str, Any]
    trainable_names: tuple[str, ...]
    trainable_counts: dict[str, int]
    package_versions: dict[str, str]
    warm_start: Mapping[str, Any] | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        artifact = {
            "mode": self.mode,
            "adapter_type": self.adapter_type,
            "adapter_name": self.adapter_name,
            "adapter_identity": None
            if self.adapter_path is None
            else {"path": str(self.adapter_path)},
            "base_model_identity": None
            if self.base_model_path is None
            else {"path": str(self.base_model_path)},
            "target_discovery": self.target_discovery.to_artifact_dict(),
            "peft_config": dict(self.peft_config),
            "trainable_names": list(self.trainable_names),
            "trainable_counts": dict(self.trainable_counts),
            "package_versions": dict(self.package_versions),
        }
        if self.warm_start is not None:
            artifact["warm_start"] = dict(self.warm_start)
        return artifact


@dataclass(frozen=True)
class DoraAdapterSetupResult:
    model: nn.Module
    receipt: DoraAdapterSetupReceipt


@dataclass(frozen=True)
class InferenceAdapterStatusReceipt:
    status: str
    adapter_name: str
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    enabled: bool
    active_adapters: tuple[str, ...]
    merged_adapters: tuple[str, ...]
    requires_grad: Any
    available_adapters: tuple[str, ...] = ()
    num_adapter_layers: int | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "adapter_name": self.adapter_name,
            "missing_keys": list(self.missing_keys),
            "unexpected_keys": list(self.unexpected_keys),
            "enabled": self.enabled,
            "active_adapters": list(self.active_adapters),
            "merged_adapters": list(self.merged_adapters),
            "requires_grad": self.requires_grad,
            "available_adapters": list(self.available_adapters),
            "num_adapter_layers": self.num_adapter_layers,
        }


def validate_inference_adapter_status(
    *,
    load_result: Any,
    status: Any,
    expected_adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> InferenceAdapterStatusReceipt:
    missing_keys = tuple(str(item) for item in getattr(load_result, "missing_keys", ()))
    unexpected_keys = tuple(
        str(item) for item in getattr(load_result, "unexpected_keys", ())
    )
    if missing_keys or unexpected_keys:
        raise RuntimeContractError(
            "inference adapter load_result contains missing or unexpected keys",
            code="adapter.inference_load_result_irregular",
            context={
                "missing_keys": list(missing_keys),
                "unexpected_keys": list(unexpected_keys),
            },
        )

    enabled = getattr(status, "enabled", None)
    active_adapters = tuple(str(item) for item in getattr(status, "active_adapters", ()))
    merged_adapters = tuple(str(item) for item in getattr(status, "merged_adapters", ()))
    requires_grad = getattr(status, "requires_grad", None)
    available_adapters = tuple(str(item) for item in getattr(status, "available_adapters", ()))
    num_adapter_layers = getattr(status, "num_adapter_layers", None)
    irregular_fields = [
        field
        for field, value in {
            "enabled": enabled,
            "active_adapters": active_adapters,
            "merged_adapters": merged_adapters,
            "requires_grad": requires_grad,
            "available_adapters": available_adapters,
            "num_adapter_layers": num_adapter_layers,
        }.items()
        if _status_value_irregular(value)
    ]
    if irregular_fields:
        raise RuntimeContractError(
            "inference adapter status contains irregular fields",
            code="adapter.inference_status_irregular",
            context={"irregular_fields": irregular_fields},
        )
    if enabled is not True:
        raise RuntimeContractError(
            "inference adapter is not enabled after load",
            code="adapter.inference_status_disabled",
            context={"enabled": enabled},
        )
    if active_adapters != (expected_adapter_name,):
        raise RuntimeContractError(
            "inference adapter active adapter list does not match expectation",
            code="adapter.inference_active_adapter_mismatch",
            context={
                "expected_active_adapters": [expected_adapter_name],
                "active_adapters": list(active_adapters),
            },
        )
    if merged_adapters:
        raise RuntimeContractError(
            "inference adapter must not be merged before generation",
            code="adapter.inference_merged_state",
            context={"merged_adapters": list(merged_adapters)},
        )
    if available_adapters and expected_adapter_name not in available_adapters:
        raise RuntimeContractError(
            "inference adapter is not listed as available after load",
            code="adapter.inference_available_adapter_mismatch",
            context={
                "expected_adapter": expected_adapter_name,
                "available_adapters": list(available_adapters),
            },
        )
    if num_adapter_layers is not None and int(num_adapter_layers) <= 0:
        raise RuntimeContractError(
            "inference adapter status reports no materialized adapter layers",
            code="adapter.inference_status_no_layers",
            context={"num_adapter_layers": num_adapter_layers},
        )
    if _requires_grad_truthy(requires_grad):
        raise RuntimeContractError(
            "inference adapter status must be frozen for inference",
            code="adapter.inference_requires_grad",
            context={"requires_grad": requires_grad},
        )
    return InferenceAdapterStatusReceipt(
        status="validated",
        adapter_name=expected_adapter_name,
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        enabled=True,
        active_adapters=active_adapters,
        merged_adapters=merged_adapters,
        requires_grad=requires_grad,
        available_adapters=available_adapters,
        num_adapter_layers=None if num_adapter_layers is None else int(num_adapter_layers),
    )


def _requires_grad_truthy(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_requires_grad_truthy(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_requires_grad_truthy(item) for item in value)
    return bool(value)


def _status_value_irregular(value: Any) -> bool:
    if value == "irregular":
        return True
    if isinstance(value, Mapping):
        return any(_status_value_irregular(item) for item in value.values())
    if isinstance(value, (list, tuple, set, frozenset)):
        return any(_status_value_irregular(item) for item in value)
    return False


def _freeze_model_for_inference(model: Any) -> None:
    requires_grad = getattr(model, "requires_grad_", None)
    if callable(requires_grad):
        requires_grad(False)
        return
    named_parameters = getattr(model, "named_parameters", None)
    if callable(named_parameters):
        for _, parameter in named_parameters():
            parameter.requires_grad_(False)


def load_inference_dora_adapter(
    *,
    config: Any,
    qwen: Any,
) -> dict[str, Any]:
    adapter = getattr(config, "adapter", None)
    if adapter is None:
        raise RuntimeContractError(
            "inference DoRA adapter load requires adapter config",
            code="adapter.inference_config_missing",
        )
    model = _qwen_model(qwen)
    if model is None:
        raise RuntimeContractError(
            "inference DoRA adapter load requires a loaded model",
            code="adapter.inference_model_required",
            context={"adapter_path": str(adapter.path)},
        )
    if not hasattr(model, "load_adapter"):
        raise RuntimeContractError(
            "inference DoRA adapter owner requires captured PEFT load_adapter result",
            code="adapter.inference_load_adapter_unavailable",
            context={"model_class": type(model).__name__, "adapter_path": str(adapter.path)},
        )
    load_result = model.load_adapter(
        adapter.path,
        adapter_name=adapter.name,
        is_trainable=False,
    )
    model.set_adapter(adapter.name)
    _freeze_model_for_inference(model)
    status = _get_inference_adapter_status(model)
    adapter_path = Path(adapter.path)
    load_result_available = load_result is not None
    payload_evidence = (
        {"payload_checked": False, "reason": "load_result_available"}
        if load_result_available
        else _validate_inference_adapter_payload(
            adapter_path,
            expected_base_model_path=_qwen_base_model_path(qwen),
        )
    )
    state_evidence = (
        {"state_checked": False, "reason": "load_result_available"}
        if load_result_available
        else _validate_transformers_mixin_adapter_state(
            model,
            adapter_path=adapter_path,
            adapter_name=adapter.name,
        )
    )
    status_receipt = validate_inference_adapter_status(
        load_result=load_result or SimpleLoadResult(),
        status=status,
        expected_adapter_name=adapter.name,
    )
    artifact = status_receipt.to_artifact_dict()
    artifact.update(
        {
            "adapter_type": adapter.type,
            "adapter_path": str(adapter_path),
            "base_model_path": _qwen_base_model_path(qwen),
            "load_result_available": load_result_available,
            "load_result_api": "peft.PeftModel.load_adapter"
            if load_result_available
            else "transformers.PeftAdapterMixin.load_adapter",
            "adapter_payload_evidence": payload_evidence,
            "adapter_state_evidence": state_evidence,
            "adapter_status_evidence": {
                "available_adapters": artifact["available_adapters"],
                "num_adapter_layers": artifact["num_adapter_layers"],
                "requires_grad": artifact["requires_grad"],
            },
        }
    )
    return artifact


@dataclass(frozen=True)
class SimpleLoadResult:
    missing_keys: tuple[str, ...] = ()
    unexpected_keys: tuple[str, ...] = ()


def _get_inference_adapter_status(model: Any) -> Any:
    try:
        from peft import get_model_status

        return get_model_status(model)
    except Exception:
        get_status = getattr(model, "get_model_status", None)
        if callable(get_status):
            return get_status()
    raise RuntimeContractError(
        "inference adapter status evidence is unavailable after load",
        code="adapter.inference_status_unavailable",
        context={"model_class": type(model).__name__},
    )


def _validate_inference_adapter_payload(
    adapter_path: Path,
    *,
    expected_base_model_path: str | None,
) -> dict[str, Any]:
    config_path = adapter_path / "adapter_config.json"
    tensor_path = adapter_path / "adapter_model.safetensors"
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeContractError(
            "inference adapter payload is missing adapter_config.json",
            code="adapter.inference_config_missing",
            context={"adapter_path": str(adapter_path)},
            cause=exc,
        ) from exc
    if config.get("use_dora") is not True or str(config.get("peft_type")) != "LORA":
        raise RuntimeContractError(
            "inference adapter config is not a DoRA LoRA payload",
            code="adapter.inference_config_incompatible",
            context={
                "use_dora": config.get("use_dora"),
                "peft_type": config.get("peft_type"),
            },
        )
    if expected_base_model_path is not None:
        declared_base = config.get("base_model_name_or_path")
        if declared_base is not None and Path(str(declared_base)).resolve() != Path(
            expected_base_model_path
        ).resolve():
            raise RuntimeContractError(
                "inference adapter base identity does not match runtime base",
                code="adapter.inference_base_mismatch",
                context={
                    "expected_base_model_path": expected_base_model_path,
                    "adapter_base_model_path": declared_base,
                },
            )
    target_modules = config.get("target_modules")
    if not isinstance(target_modules, list) or not target_modules:
        raise RuntimeContractError(
            "inference adapter config must record non-empty target modules",
            code="adapter.inference_config_incompatible",
            context={"target_modules": target_modules},
        )
    rank = int(config.get("r") or 0)
    alpha = int(config.get("lora_alpha") or 0)
    if rank <= 0 or alpha <= 0:
        raise RuntimeContractError(
            "inference adapter config must record positive rank and alpha",
            code="adapter.inference_config_incompatible",
            context={"r": config.get("r"), "lora_alpha": config.get("lora_alpha")},
        )
    if not tensor_path.is_file():
        raise RuntimeContractError(
            "inference adapter payload is missing adapter_model.safetensors",
            code="adapter.inference_payload_missing",
            context={"tensor_path": str(tensor_path)},
        )
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
    evidence = _adapter_tensor_evidence(keys)
    if (
        evidence["key_count"] <= 0
        or evidence["lora_A_count"] <= 0
        or evidence["lora_B_count"] <= 0
        or evidence["lora_magnitude_vector_count"] <= 0
    ):
        raise RuntimeContractError(
            "inference adapter tensor payload is missing expected DoRA/LoRA tensors",
            code="adapter.inference_payload_shape",
            context=evidence,
        )
    evidence.update(
        {
            "config_path": str(config_path),
            "tensor_path": str(tensor_path),
            "target_module_count": len(target_modules),
            "rank": rank,
            "alpha": alpha,
        }
    )
    return evidence


def _adapter_tensor_evidence(keys: list[str]) -> dict[str, Any]:
    return {
        "key_count": len(keys),
        "lora_A_count": sum(".lora_A." in key for key in keys),
        "lora_B_count": sum(".lora_B." in key for key in keys),
        "lora_magnitude_vector_count": sum("lora_magnitude_vector" in key for key in keys),
    }


def _validate_transformers_mixin_adapter_state(
    model: Any,
    *,
    adapter_path: Path,
    adapter_name: str,
) -> dict[str, Any]:
    get_state = getattr(model, "get_adapter_state_dict", None)
    if not callable(get_state):
        return {"state_checked": False, "reason": "get_adapter_state_dict_unavailable"}
    state = get_state(adapter_name)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeContractError(
            "inference adapter materialized state is empty",
            code="adapter.inference_state_empty",
            context={"adapter_name": adapter_name},
        )
    tensor_path = adapter_path / "adapter_model.safetensors"
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        saved_keys = list(handle.keys())
    normalized_saved = {_normalize_adapter_state_key(key, adapter_name=adapter_name) for key in saved_keys}
    normalized_state = {
        _normalize_adapter_state_key(str(key), adapter_name=adapter_name)
        for key in state
    }
    missing = sorted(normalized_saved - normalized_state)
    extra = sorted(normalized_state - normalized_saved)
    if missing or extra:
        raise RuntimeContractError(
            "inference adapter materialized state does not match saved payload",
            code="adapter.inference_state_mismatch",
            context={
                "adapter_name": adapter_name,
                "missing_materialized_keys": missing[:20],
                "extra_materialized_keys": extra[:20],
                "missing_count": len(missing),
                "extra_count": len(extra),
            },
        )
    return {
        "state_checked": True,
        "normalized_saved_key_count": len(normalized_saved),
        "normalized_materialized_key_count": len(normalized_state),
    }


def _normalize_adapter_state_key(key: str, *, adapter_name: str) -> str:
    parts = [part for part in key.split(".") if part != adapter_name]
    normalized = ".".join(parts)
    while normalized.startswith("base_model.model."):
        normalized = normalized.removeprefix("base_model.model.")
    return normalized


def discover_dora_targets(
    model: nn.Module,
    plan: AdapterSetupPlan,
) -> DoraTargetDiscoveryReceipt:
    if plan.target_policy != "all_linear":
        raise RuntimeContractError(
            "unsupported DoRA target policy",
            code="adapter.target_policy_unsupported",
            context={"target_policy": plan.target_policy},
        )

    matched_by_tower: dict[str, list[str]] = {tower: [] for tower in plan.target_towers}
    lm_head_seen = False
    for name, module in model.named_modules():
        if _is_lm_head(name):
            lm_head_seen = True
        if not isinstance(module, nn.Linear):
            continue
        if _is_excluded_linear_name(name):
            continue
        for tower in plan.target_towers:
            if _linear_belongs_to_tower(name, tower):
                matched_by_tower[tower].append(name)
                break

    for tower, names in matched_by_tower.items():
        if not names:
            raise RuntimeContractError(
                "DoRA target discovery found no linear modules for requested tower",
                code="adapter.target_discovery_empty",
                context={"target_tower": tower, "target_policy": plan.target_policy},
            )

    matched_modules = tuple(
        name
        for tower in plan.target_towers
        for name in matched_by_tower[tower]
    )
    return DoraTargetDiscoveryReceipt(
        target_policy=plan.target_policy,
        target_towers=plan.target_towers,
        matched_modules=matched_modules,
        counts_by_tower={tower: len(names) for tower, names in matched_by_tower.items()},
        lm_head_seen=lm_head_seen,
        lm_head_excluded=not any(_is_lm_head(name) for name in matched_modules),
    )


def setup_dora_adapter(
    model: nn.Module,
    plan: AdapterSetupPlan,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> DoraAdapterSetupResult:
    from peft import LoraConfig, PeftModel, get_peft_model

    target_receipt = discover_dora_targets(model, plan)
    warm_start_receipt: dict[str, Any] | None = None
    if plan.mode in {"initialize_new", "warm_start_expand_dora"}:
        peft_config = LoraConfig(
            r=plan.rank,
            lora_alpha=plan.alpha,
            lora_dropout=plan.dropout,
            bias=plan.bias,
            target_modules=list(target_receipt.matched_modules),
            use_dora=True,
        )
        adapted_model = get_peft_model(model, peft_config, adapter_name=adapter_name)
        active_config = peft_config
        if plan.mode == "warm_start_expand_dora":
            warm_start_receipt = _warm_start_expand_dora_adapter(
                adapted_model,
                plan,
                target_receipt=target_receipt,
                adapter_name=adapter_name,
            )
    elif plan.mode == "load_existing":
        if plan.adapter_path is None:
            raise RuntimeContractError(
                "existing DoRA adapter setup requires adapter path",
                code="adapter.path_required",
            )
        adapted_model = PeftModel.from_pretrained(
            model,
            plan.adapter_path,
            adapter_name=adapter_name,
            is_trainable=True,
        )
        active_config = adapted_model.peft_config[adapter_name]
        _validate_loaded_dora_config(active_config, plan)
        _validate_loaded_adapter_identity(active_config, plan, model)
        _validate_loaded_targets_match_discovery(active_config, target_receipt)
    else:
        raise RuntimeContractError(
            "unsupported adapter setup mode",
            code="adapter.mode_unsupported",
            context={"mode": plan.mode},
        )

    trainable_names = tuple(
        name
        for name, parameter in adapted_model.named_parameters()
        if parameter.requires_grad
    )
    trainable_counts = _count_dora_trainable_names(trainable_names)
    expected_target_count = len(target_receipt.matched_modules)
    _validate_trainable_dora_surface(
        trainable_counts=trainable_counts,
        expected_target_count=expected_target_count,
        trainable_names=trainable_names,
        adapter_name=adapter_name,
    )

    return DoraAdapterSetupResult(
        model=adapted_model,
        receipt=DoraAdapterSetupReceipt(
            mode=plan.mode,
            adapter_type=plan.adapter_type,
            adapter_name=adapter_name,
            adapter_path=plan.adapter_path,
            base_model_path=plan.base_model_path,
            target_discovery=target_receipt,
            peft_config=_peft_config_artifact(active_config),
            trainable_names=trainable_names,
            trainable_counts=trainable_counts,
            package_versions=_package_versions(("peft", "torch")),
            warm_start=warm_start_receipt,
        ),
    )


def _warm_start_expand_dora_adapter(
    adapted_model: nn.Module,
    plan: AdapterSetupPlan,
    *,
    target_receipt: DoraTargetDiscoveryReceipt,
    adapter_name: str,
) -> dict[str, Any]:
    if plan.source_adapter_path is None:
        raise RuntimeContractError(
            "warm_start_expand_dora requires source adapter path",
            code="adapter.warm_start_source_adapter_required",
        )
    source_adapter_path = plan.source_adapter_path
    tensor_path = source_adapter_path / "adapter_model.safetensors"
    if not tensor_path.is_file():
        raise RuntimeContractError(
            "warm_start_expand_dora source adapter is missing adapter_model.safetensors",
            code="adapter.warm_start_source_tensor_missing",
            context={"tensor_path": str(tensor_path)},
        )

    source_tensors = _load_source_adapter_tensors(tensor_path)
    target_groups = _group_dora_target_parameters(
        adapted_model,
        adapter_name=adapter_name,
    )
    copied_counts = {"lora_A": 0, "lora_B": 0, "dora_magnitude": 0}
    initialized_counts = {"language": 0, "vision": 0, "aligner": 0}
    copied_records: list[dict[str, Any]] = []
    initialized_target_tensors: list[str] = []
    missing_source_keys: list[str] = []
    shape_mismatches: list[dict[str, Any]] = []
    equality_failures: list[dict[str, str]] = []
    used_source_keys: set[str] = set()

    for group in target_groups:
        present_entries = [
            entry for entry in group["entries"] if entry["source_key"] in source_tensors
        ]
        if not present_entries:
            initialized_counts[str(group["tower"])] += len(group["entries"])
            initialized_target_tensors.extend(
                str(entry["target_key"]) for entry in group["entries"]
            )
            continue
        if len(present_entries) != len(group["entries"]):
            missing_source_keys.extend(
                str(entry["source_key"])
                for entry in group["entries"]
                if entry["source_key"] not in source_tensors
            )
            continue
        for entry in group["entries"]:
            source_key = str(entry["source_key"])
            target_key = str(entry["target_key"])
            parameter = entry["parameter"]
            source_tensor = source_tensors[source_key]
            if tuple(source_tensor.shape) != tuple(parameter.shape):
                shape_mismatches.append(
                    {
                        "source_key": source_key,
                        "target_key": target_key,
                        "source_shape": [int(item) for item in source_tensor.shape],
                        "target_shape": [int(item) for item in parameter.shape],
                    }
                )
                continue
            with torch.no_grad():
                parameter.copy_(
                    source_tensor.to(device=parameter.device, dtype=parameter.dtype)
                )
            if not torch.equal(
                parameter.detach().cpu(),
                source_tensor.to(dtype=parameter.dtype).cpu(),
            ):
                equality_failures.append(
                    {"source_key": source_key, "target_key": target_key}
                )
            used_source_keys.add(source_key)
            kind = str(entry["kind"])
            copied_counts[kind] += 1
            copied_records.append(
                {
                    "source_key": source_key,
                    "target_key": target_key,
                    "shape": [int(item) for item in parameter.shape],
                    "source_sha256": _tensor_sha256(source_tensor),
                    "target_sha256": _tensor_sha256(parameter.detach()),
                    "status": "copied",
                    "post_copy_equality": "pass",
                }
            )

    if missing_source_keys:
        raise RuntimeContractError(
            "warm_start_expand_dora source adapter has partial tensors for requested target modules",
            code="adapter.warm_start_partial_target_tensor",
            context={"missing_source_keys": sorted(missing_source_keys)},
        )
    if shape_mismatches:
        raise RuntimeContractError(
            "warm_start_expand_dora source tensor shapes do not match requested target modules",
            code="adapter.warm_start_shape_mismatch",
            context={"shape_mismatches": shape_mismatches},
        )
    if equality_failures:
        raise RuntimeContractError(
            "warm_start_expand_dora post-copy equality check failed",
            code="adapter.warm_start_post_copy_equality",
            context={"equality_failures": equality_failures},
        )

    ignored_source_keys = sorted(set(source_tensors) - used_source_keys)
    return {
        "seed_mode": "warm_start_expand_dora",
        "source_adapter_path": str(source_adapter_path),
        "source_adapter_tensor_path": str(tensor_path),
        "source_adapter_tensor_sha256": _sha256_file(tensor_path),
        "base_model_path": None if plan.base_model_path is None else str(plan.base_model_path),
        "source_gate": plan.source_gate.to_artifact_dict(),
        "target_towers": list(target_receipt.target_towers),
        "target_policy": target_receipt.target_policy,
        "target_discovery": target_receipt.to_artifact_dict(),
        "copied": copied_counts,
        "initialized": {
            tower: int(initialized_counts.get(tower, 0))
            for tower in target_receipt.target_towers
        },
        "copied_tensors": copied_records,
        "source_to_target_key_map": copied_records,
        "initialized_target_tensors": initialized_target_tensors,
        "ignored_source_tensors": ignored_source_keys,
        "post_copy_equality": "pass",
        "repaired_embedding_payload_path": None
        if plan.repaired_embedding_payload_path is None
        else str(plan.repaired_embedding_payload_path),
    }


def _load_source_adapter_tensors(tensor_path: Path) -> dict[str, torch.Tensor]:
    tensors: dict[str, torch.Tensor] = {}
    with safe_open(str(tensor_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            tensors[key] = handle.get_tensor(key).cpu()
    if not tensors:
        raise RuntimeContractError(
            "warm_start_expand_dora source adapter tensor payload is empty",
            code="adapter.warm_start_source_tensor_empty",
            context={"tensor_path": str(tensor_path)},
        )
    return tensors


def _group_dora_target_parameters(
    adapted_model: nn.Module,
    *,
    adapter_name: str,
) -> list[dict[str, Any]]:
    by_prefix: dict[str, dict[str, Any]] = {}
    for target_key, parameter in adapted_model.named_parameters():
        if not parameter.requires_grad or not _is_dora_adapter_trainable_name(
            target_key,
            adapter_name=adapter_name,
        ):
            continue
        source_key = _target_trainable_key_to_source_key(
            target_key,
            adapter_name=adapter_name,
        )
        prefix = _source_key_target_prefix(source_key)
        tower = _tower_for_trainable_name(target_key)
        if tower == "unknown":
            raise RuntimeContractError(
                "warm_start_expand_dora encountered unsupported trainable adapter tower",
                code="adapter.warm_start_unsupported_tower",
                context={"target_key": target_key, "tower": tower},
            )
        group = by_prefix.setdefault(prefix, {"tower": tower, "entries": []})
        if group["tower"] != tower:
            raise RuntimeContractError(
                "warm_start_expand_dora target prefix maps to multiple towers",
                code="adapter.warm_start_ambiguous_target",
                context={"target_prefix": prefix, "towers": [group["tower"], tower]},
            )
        group["entries"].append(
            {
                "target_key": target_key,
                "source_key": source_key,
                "kind": _dora_tensor_kind(target_key),
                "parameter": parameter,
            }
        )
    groups = list(by_prefix.values())
    for group in groups:
        kinds = sorted(str(entry["kind"]) for entry in group["entries"])
        if kinds != ["dora_magnitude", "lora_A", "lora_B"]:
            raise RuntimeContractError(
                "warm_start_expand_dora target module does not expose complete DoRA tensors",
                code="adapter.warm_start_target_incomplete",
                context={"tower": group["tower"], "kinds": kinds},
            )
    return groups


def _target_trainable_key_to_source_key(target_key: str, *, adapter_name: str) -> str:
    suffix_map = {
        f".lora_A.{adapter_name}.weight": ".lora_A.weight",
        f".lora_B.{adapter_name}.weight": ".lora_B.weight",
        f".lora_magnitude_vector.{adapter_name}.weight": ".lora_magnitude_vector",
    }
    for target_suffix, source_suffix in suffix_map.items():
        if target_key.endswith(target_suffix):
            return f"{target_key.removesuffix(target_suffix)}{source_suffix}"
    raise RuntimeContractError(
        "warm_start_expand_dora target key is not a supported DoRA tensor",
        code="adapter.warm_start_target_key_unsupported",
        context={"target_key": target_key},
    )


def _source_key_target_prefix(source_key: str) -> str:
    for suffix in (".lora_A.weight", ".lora_B.weight", ".lora_magnitude_vector"):
        if source_key.endswith(suffix):
            return source_key.removesuffix(suffix)
    raise RuntimeContractError(
        "warm_start_expand_dora source key is not a supported DoRA tensor",
        code="adapter.warm_start_source_key_unsupported",
        context={"source_key": source_key},
    )


def _dora_tensor_kind(target_key: str) -> str:
    if ".lora_A." in target_key:
        return "lora_A"
    if ".lora_B." in target_key:
        return "lora_B"
    if ".lora_magnitude_vector." in target_key:
        return "dora_magnitude"
    raise RuntimeContractError(
        "warm_start_expand_dora target key has unknown DoRA tensor kind",
        code="adapter.warm_start_target_key_unsupported",
        context={"target_key": target_key},
    )


def _tower_for_trainable_name(name: str) -> str:
    for tower in ("language", "aligner", "vision"):
        if _linear_belongs_to_tower(name, tower):
            return tower
    return "unknown"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    byte_view = cpu_tensor.view(torch.uint8)
    return hashlib.sha256(byte_view.numpy().tobytes()).hexdigest()


def _validate_loaded_dora_config(active_config: Any, plan: AdapterSetupPlan) -> None:
    if getattr(active_config, "use_dora", False) is not True:
        raise RuntimeContractError(
            "loaded adapter config is not PEFT DoRA",
            code="adapter.loaded_not_dora",
            context={"adapter_path": str(plan.adapter_path)},
        )


def _validate_loaded_adapter_identity(
    active_config: Any,
    plan: AdapterSetupPlan,
    model: nn.Module,
) -> None:
    loaded_base = _normalize_base_model_identity(
        getattr(active_config, "base_model_name_or_path", None)
    )
    requested_base = _normalize_base_model_identity(plan.base_model_path)
    if loaded_base is not None and requested_base is not None and loaded_base != requested_base:
        raise RuntimeContractError(
            "loaded DoRA adapter base model identity does not match requested base model",
            code="adapter.loaded_base_model_mismatch",
            context={
                "adapter_path": None if plan.adapter_path is None else str(plan.adapter_path),
                "loaded_base_model_name_or_path": loaded_base,
                "requested_base_model_path": requested_base,
            },
        )

    loaded_model_class = _loaded_base_model_class(active_config)
    current_model_class = type(model).__name__
    if loaded_model_class is not None and loaded_model_class != current_model_class:
        raise RuntimeContractError(
            "loaded DoRA adapter base model class does not match current model",
            code="adapter.loaded_base_model_class_mismatch",
            context={
                "adapter_path": None if plan.adapter_path is None else str(plan.adapter_path),
                "loaded_base_model_class": loaded_model_class,
                "current_base_model_class": current_model_class,
            },
        )


def _validate_loaded_targets_match_discovery(
    active_config: Any,
    target_receipt: DoraTargetDiscoveryReceipt,
) -> None:
    loaded_targets = _target_modules_from_peft_config(active_config)
    discovered_targets = _normalize_target_modules(target_receipt.matched_modules)
    if loaded_targets != discovered_targets and not _compact_targets_cover_discovery(
        loaded_targets=loaded_targets,
        discovered_targets=discovered_targets,
    ):
        raise RuntimeContractError(
            "loaded DoRA adapter targets do not match requested target discovery",
            code="adapter.loaded_target_mismatch",
            context={
                "requested_target_towers": list(target_receipt.target_towers),
                "loaded_target_modules": list(loaded_targets),
                "discovered_target_modules": list(discovered_targets),
                "discovered_target_modules_in_receipt_order": list(
                    target_receipt.matched_modules
                ),
            },
        )


def _peft_config_artifact(config: Any) -> dict[str, Any]:
    return {
        "config_class": type(config).__name__,
        "use_dora": bool(getattr(config, "use_dora", False)),
        "r": int(getattr(config, "r")),
        "lora_alpha": int(getattr(config, "lora_alpha")),
        "lora_dropout": float(getattr(config, "lora_dropout")),
        "bias": str(getattr(config, "bias")),
        "target_modules": list(_target_modules_from_peft_config(config)),
    }


def _count_dora_trainable_names(trainable_names: tuple[str, ...]) -> dict[str, int]:
    return {
        "total": len(trainable_names),
        "lora_A": sum("lora_A" in name for name in trainable_names),
        "lora_B": sum("lora_B" in name for name in trainable_names),
        "lora_magnitude_vector": sum(
            "lora_magnitude_vector" in name for name in trainable_names
        ),
    }


def _target_modules_from_peft_config(config: Any) -> tuple[str, ...]:
    target_modules = getattr(config, "target_modules", None)
    if target_modules is None:
        raise RuntimeContractError(
            "PEFT DoRA config does not record target modules",
            code="adapter.peft_target_modules_missing",
        )
    return _normalize_target_modules(target_modules)


def _normalize_target_modules(target_modules: Any) -> tuple[str, ...]:
    if isinstance(target_modules, str):
        return (target_modules,)
    return tuple(sorted(str(item) for item in target_modules))


def _compact_targets_cover_discovery(
    *,
    loaded_targets: tuple[str, ...],
    discovered_targets: tuple[str, ...],
) -> bool:
    if not loaded_targets or not discovered_targets:
        return False
    loaded_used = {target: False for target in loaded_targets}
    for discovered_target in discovered_targets:
        matched_loaded_targets = [
            loaded_target
            for loaded_target in loaded_targets
            if discovered_target == loaded_target
            or discovered_target.endswith(f".{loaded_target}")
        ]
        if len(matched_loaded_targets) != 1:
            return False
        loaded_used[matched_loaded_targets[0]] = True
    return all(loaded_used.values())


def _normalize_base_model_identity(value: Any) -> str | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw == "":
        return None
    path = Path(raw).expanduser()
    if path.is_absolute() or raw.startswith("~"):
        return str(path.resolve(strict=False))
    return raw


def _qwen_model(qwen: Any) -> Any | None:
    if isinstance(qwen, Mapping):
        return qwen.get("model")
    return getattr(qwen, "model", None)


def _qwen_base_model_path(qwen: Any) -> str | None:
    if isinstance(qwen, Mapping):
        value = qwen.get("base_model_path")
    else:
        value = getattr(qwen, "base_model_path", None)
    return None if value is None else str(value)


def _loaded_base_model_class(config: Any) -> str | None:
    auto_mapping = getattr(config, "auto_mapping", None)
    if not isinstance(auto_mapping, Mapping):
        return None
    value = auto_mapping.get("base_model_class")
    if value is None:
        return None
    loaded_class = str(value).strip()
    return loaded_class or None


def _validate_trainable_dora_surface(
    *,
    trainable_counts: Mapping[str, int],
    expected_target_count: int,
    trainable_names: tuple[str, ...],
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> None:
    unexpected_trainables = [
        name
        for name in trainable_names
        if not _is_dora_adapter_trainable_name(name, adapter_name=adapter_name)
    ]
    if unexpected_trainables:
        raise RuntimeContractError(
            "DoRA setup produced unsupported trainable parameters",
            code="adapter.dora_trainable_surface",
            context={"unexpected_trainable_names": unexpected_trainables},
        )
    expected = {
        "lora_A": expected_target_count,
        "lora_B": expected_target_count,
        "lora_magnitude_vector": expected_target_count,
    }
    mismatched = {
        name: {"expected": expected_count, "actual": trainable_counts.get(name, 0)}
        for name, expected_count in expected.items()
        if trainable_counts.get(name, 0) != expected_count
    }
    if mismatched:
        raise RuntimeContractError(
            "DoRA trainable surface does not match selected target count",
            code="adapter.dora_trainable_surface",
            context={
                "expected_target_count": expected_target_count,
                "mismatched_counts": mismatched,
                "trainable_counts": dict(trainable_counts),
            },
        )


def _is_dora_adapter_trainable_name(
    name: str,
    *,
    adapter_name: str = DEFAULT_ADAPTER_NAME,
) -> bool:
    allowed_suffixes = (
        f".lora_A.{adapter_name}.weight",
        f".lora_B.{adapter_name}.weight",
        f".lora_magnitude_vector.{adapter_name}.weight",
    )
    return name.endswith(allowed_suffixes)


def _linear_belongs_to_tower(name: str, tower: str) -> bool:
    if tower == "language":
        return "language_model." in name or name.startswith("language_model.")
    if tower == "aligner":
        return _is_aligner_name(name)
    if tower == "vision":
        return ("visual." in name or name.startswith("visual.")) and not _is_aligner_name(name)
    return False


def _is_excluded_linear_name(name: str) -> bool:
    if _is_lm_head(name):
        return True
    excluded_parts = ("score", "v_head", "classifier", "lora_A", "lora_B", "base_layer")
    return any(part in name.split(".") for part in excluded_parts)


def _is_lm_head(name: str) -> bool:
    return name == "lm_head" or name.endswith(".lm_head")


def _is_aligner_name(name: str) -> bool:
    return (
        ".visual.merger" in name
        or name.startswith("visual.merger")
        or "deepstack_merger" in name
    )


def _package_versions(package_names: tuple[str, ...]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for package_name in package_names:
        try:
            versions[package_name] = metadata.version(package_name)
        except metadata.PackageNotFoundError:
            versions[package_name] = "not-installed"
    return versions
