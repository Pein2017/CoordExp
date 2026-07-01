"""PEFT DoRA setup for approved adapter plans."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Mapping

import torch
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

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
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


@dataclass(frozen=True)
class DoraAdapterSetupResult:
    model: nn.Module
    receipt: DoraAdapterSetupReceipt


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
    if plan.mode == "initialize_new":
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
        ),
    )


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
