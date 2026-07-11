"""Source-study gates for adapter setup.

The objects in this module intentionally stop before PEFT model mutation. They
prove the approved adapter mechanism and return a setup plan that later DoRA
code can consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from src.common.errors import RuntimeContractError
from src.config.models import AdapterConfig


DEFAULT_DORA_SOURCE_STUDY_PATH = Path(
    "docs/history/architecture/proposals/2026-06-27-coordexp-swift/"
    "source-studies/dora.md"
)
DEFAULT_DORA_PROBE_RECEIPT_PATH = Path(
    "outputs/probes/coordexp_swift/dora_roundtrip/receipt.json"
)

AdapterSetupMode = Literal["initialize_new", "load_existing", "warm_start_expand_dora"]


@dataclass(frozen=True)
class AdapterSourceGateEvidence:
    dora_source_study_passed: bool
    dora_probe_passed: bool
    dora_probe_receipt: Mapping[str, Any] | None = None
    dora_source_study_path: Path | None = None
    dora_probe_receipt_path: Path | None = None


@dataclass(frozen=True)
class DoraSourceGateReceipt:
    status: Literal["passed"]
    public_adapter_type: Literal["dora"]
    peft_mechanism: str
    source_study_path: Path | None
    probe_receipt_path: Path | None
    lm_head_excluded: bool
    saved_lora_a_count: int
    saved_lora_b_count: int
    saved_magnitude_vector_count: int
    reloaded_magnitude_vector_count: int
    reload_success: bool
    logit_equivalence: bool
    finite_reloaded_eval_logits: bool
    finite_magnitude_vector_gradient: bool

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_adapter_type": self.public_adapter_type,
            "peft_mechanism": self.peft_mechanism,
            "source_study_path": _optional_path_text(self.source_study_path),
            "probe_receipt_path": _optional_path_text(self.probe_receipt_path),
            "lm_head_excluded": self.lm_head_excluded,
            "lora": {
                "saved_A_count": self.saved_lora_a_count,
                "saved_B_count": self.saved_lora_b_count,
            },
            "magnitude_vector": {
                "saved_count": self.saved_magnitude_vector_count,
                "reload_count": self.reloaded_magnitude_vector_count,
                "finite_gradient": self.finite_magnitude_vector_gradient,
            },
            "reload": {
                "success": self.reload_success,
                "logit_equivalence": self.logit_equivalence,
                "finite_reloaded_eval_logits": self.finite_reloaded_eval_logits,
            },
        }


@dataclass(frozen=True)
class AdapterSetupPlan:
    mode: AdapterSetupMode
    adapter_type: Literal["dora"]
    adapter_path: Path | None
    source_adapter_path: Path | None
    repaired_embedding_payload_path: Path | None
    base_model_path: Path | None
    target_towers: tuple[str, ...]
    target_policy: Literal["all_linear"]
    rank: int
    alpha: int
    dropout: float
    bias: str
    source_gate: DoraSourceGateReceipt

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "adapter_type": self.adapter_type,
            "adapter_identity": None
            if self.adapter_path is None
            else {"path": str(self.adapter_path)},
            "source_adapter_identity": None
            if self.source_adapter_path is None
            else {"path": str(self.source_adapter_path)},
            "repaired_embedding_payload_identity": None
            if self.repaired_embedding_payload_path is None
            else {"path": str(self.repaired_embedding_payload_path)},
            "base_model_identity": None
            if self.base_model_path is None
            else {"path": str(self.base_model_path)},
            "target_towers": list(self.target_towers),
            "target_policy": self.target_policy,
            "rank": self.rank,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "bias": self.bias,
            "source_gate": self.source_gate.to_artifact_dict(),
        }


def load_default_adapter_source_gate_evidence(
    repo_root: str | Path,
) -> AdapterSourceGateEvidence:
    root = Path(repo_root).expanduser().resolve()
    source_study_path = root / DEFAULT_DORA_SOURCE_STUDY_PATH
    probe_receipt_path = root / DEFAULT_DORA_PROBE_RECEIPT_PATH
    source_study_passed = _dora_source_study_is_passed(source_study_path)
    probe_receipt = (
        load_dora_probe_receipt(probe_receipt_path)
        if probe_receipt_path.exists()
        else None
    )
    return AdapterSourceGateEvidence(
        dora_source_study_passed=source_study_passed,
        dora_probe_passed=probe_receipt is not None,
        dora_probe_receipt=probe_receipt,
        dora_source_study_path=source_study_path,
        dora_probe_receipt_path=probe_receipt_path,
    )


def load_dora_probe_receipt(path: str | Path) -> dict[str, Any]:
    receipt_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeContractError(
            "DoRA probe receipt does not exist",
            code="adapter.dora_probe_receipt_missing",
            context={"path": str(receipt_path)},
            cause=exc,
        ) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeContractError(
            "DoRA probe receipt is not valid JSON",
            code="adapter.dora_probe_receipt_json",
            context={"path": str(receipt_path), "error": str(exc)},
            cause=exc,
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeContractError(
            "DoRA probe receipt must be a JSON object",
            code="adapter.dora_probe_receipt_shape",
            context={"path": str(receipt_path), "type": type(payload).__name__},
        )
    return payload


def build_adapter_setup_plan(
    adapter_config: AdapterConfig,
    evidence: AdapterSourceGateEvidence,
    *,
    base_model_path: str | Path | None = None,
) -> AdapterSetupPlan:
    _ensure_supported_adapter_type(adapter_config)
    source_gate = _build_dora_source_gate_receipt(adapter_config, evidence)
    adapter_path = None if adapter_config.path is None else Path(adapter_config.path)
    source_adapter_path = (
        None
        if adapter_config.source_adapter_path is None
        else Path(adapter_config.source_adapter_path)
    )
    repaired_embedding_payload_path = (
        None
        if adapter_config.repaired_embedding_payload_path is None
        else Path(adapter_config.repaired_embedding_payload_path)
    )
    resolved_base_model_path = None if base_model_path is None else Path(base_model_path)
    seed_mode = adapter_config.seed_mode
    if seed_mode is None:
        mode: AdapterSetupMode = "initialize_new" if adapter_path is None else "load_existing"
    else:
        mode = seed_mode
    if mode == "warm_start_expand_dora":
        if source_adapter_path is None or repaired_embedding_payload_path is None:
            raise RuntimeContractError(
                "warm_start_expand_dora requires source adapter and repaired embedding payload paths",
                code="adapter.warm_start_source_paths_required",
                context={
                    "source_adapter_path": None
                    if source_adapter_path is None
                    else str(source_adapter_path),
                    "repaired_embedding_payload_path": None
                    if repaired_embedding_payload_path is None
                    else str(repaired_embedding_payload_path),
                },
            )
    if mode in {"load_existing", "warm_start_expand_dora"} and resolved_base_model_path is None:
        raise RuntimeContractError(
            "adapter setup must record base model identity",
            code="adapter.base_model_identity_required",
            context={
                "mode": mode,
                "adapter_path": None if adapter_path is None else str(adapter_path),
                "source_adapter_path": None
                if source_adapter_path is None
                else str(source_adapter_path),
            },
        )
    return AdapterSetupPlan(
        mode=mode,
        adapter_type="dora",
        adapter_path=adapter_path,
        source_adapter_path=source_adapter_path,
        repaired_embedding_payload_path=repaired_embedding_payload_path,
        base_model_path=resolved_base_model_path,
        target_towers=tuple(adapter_config.target_towers),
        target_policy=adapter_config.target_modules,
        rank=adapter_config.rank,
        alpha=adapter_config.alpha,
        dropout=adapter_config.dropout,
        bias=adapter_config.bias,
        source_gate=source_gate,
    )


def _ensure_supported_adapter_type(adapter_config: AdapterConfig) -> None:
    if adapter_config.type != "dora":
        raise RuntimeContractError(
            "unsupported V1 adapter type; V1 uses adapter.type: dora",
            code="adapter.unsupported_type",
            context={"adapter_type": adapter_config.type},
        )


def _build_dora_source_gate_receipt(
    adapter_config: AdapterConfig,
    evidence: AdapterSourceGateEvidence,
) -> DoraSourceGateReceipt:
    missing_gates: list[str] = []
    if not evidence.dora_source_study_passed:
        missing_gates.append("dora_source_study")
    if not evidence.dora_probe_passed:
        missing_gates.append("dora_roundtrip_probe")
    if evidence.dora_probe_receipt is None:
        missing_gates.append("dora_probe_receipt")
    if missing_gates:
        raise RuntimeContractError(
            "DoRA source-study/probe gate is not passed",
            code="adapter.dora_source_gate_missing",
            context={
                "missing_gates": missing_gates,
                "source_study_path": _optional_path_text(evidence.dora_source_study_path),
                "probe_receipt_path": _optional_path_text(evidence.dora_probe_receipt_path),
            },
        )

    receipt = evidence.dora_probe_receipt
    assert receipt is not None
    selected_targets = (
        _validate_receipt_self_target_gate(receipt)
        if adapter_config.seed_mode == "warm_start_expand_dora"
        else _validate_requested_target_gate(adapter_config, receipt)
    )
    _validate_receipt_value(
        receipt.get("public_adapter_type") == "dora",
        "DoRA probe receipt must record public adapter type dora",
        {"field": "public_adapter_type", "value": receipt.get("public_adapter_type")},
    )
    peft = _mapping(receipt, "peft")
    _validate_receipt_value(
        peft.get("config_class") == "LoraConfig",
        "DoRA probe receipt must use PEFT LoraConfig",
        {"field": "peft.config_class", "value": peft.get("config_class")},
    )
    _validate_receipt_value(
        peft.get("use_dora") is True,
        "DoRA probe receipt must prove PEFT use_dora=True",
        {"field": "peft.use_dora", "value": peft.get("use_dora")},
    )
    save = _mapping(receipt, "save")
    reload_receipt = _mapping(receipt, "reload")
    equivalence = _mapping(receipt, "equivalence")
    gradient_result = _mapping(receipt, "gradient_result")
    magnitude_vectors = _mapping(receipt, "magnitude_vectors")

    selected_target_count = _positive_int(
        receipt,
        "selected_target_count",
        label="selected target count",
    )
    saved_lora_a_count = _positive_int(
        save,
        "saved_lora_A_count",
        label="LoRA A count",
    )
    saved_lora_b_count = _positive_int(
        save,
        "saved_lora_B_count",
        label="LoRA B count",
    )
    saved_magnitude_count = _positive_int(
        save,
        "saved_lora_magnitude_vector_count",
        label="magnitude-vector save count",
    )
    reloaded_magnitude_count = _positive_int(
        reload_receipt,
        "reloaded_magnitude_vector_count",
        label="magnitude-vector reload count",
    )
    _validate_count_coverage(
        selected_target_count=selected_target_count,
        saved_lora_a_count=saved_lora_a_count,
        saved_lora_b_count=saved_lora_b_count,
        saved_magnitude_count=saved_magnitude_count,
        reloaded_magnitude_count=reloaded_magnitude_count,
    )
    _validate_name_evidence(
        save,
        "adapter_payload_keys",
        needle="lora_magnitude_vector",
        minimum_count=selected_target_count,
    )
    _validate_name_evidence(
        receipt,
        "trainable_names",
        needle="lora_magnitude_vector",
        minimum_count=selected_target_count,
    )
    _validate_name_evidence(
        magnitude_vectors,
        "reloaded_names",
        needle="lora_magnitude_vector",
        minimum_count=selected_target_count,
    )
    _validate_selected_target_name_coverage(
        selected_targets=selected_targets,
        trainable_names=_string_list(receipt, "trainable_names"),
        saved_payload_keys=_string_list(save, "adapter_payload_keys"),
        reloaded_magnitude_names=_string_list(magnitude_vectors, "reloaded_names"),
    )
    _validate_receipt_value(
        save.get("adapter_config_use_dora") is True,
        "DoRA adapter_config.json must preserve use_dora=True",
        {"field": "save.adapter_config_use_dora"},
    )
    reload_success = reload_receipt.get("success") is True
    _validate_receipt_value(
        reload_success,
        "DoRA probe receipt must prove reload success",
        {"field": "reload.success", "value": reload_receipt.get("success")},
    )
    logit_equivalence = equivalence.get("equivalent") is True
    _validate_receipt_value(
        logit_equivalence,
        "DoRA probe receipt must prove reload logit equivalence",
        {"field": "equivalence.equivalent", "value": equivalence.get("equivalent")},
    )
    finite_reloaded_logits = equivalence.get("finite_reloaded_eval_logits") is True
    _validate_receipt_value(
        finite_reloaded_logits,
        "DoRA probe receipt must prove finite reloaded eval logits",
        {
            "field": "equivalence.finite_reloaded_eval_logits",
            "value": equivalence.get("finite_reloaded_eval_logits"),
        },
    )
    finite_magnitude_gradient = (
        gradient_result.get("finite_magnitude_vector_gradient") is True
    )
    _validate_receipt_value(
        finite_magnitude_gradient,
        "DoRA probe receipt must prove finite magnitude-vector gradient",
        {
            "field": "gradient_result.finite_magnitude_vector_gradient",
            "value": gradient_result.get("finite_magnitude_vector_gradient"),
        },
    )
    lm_head_excluded = receipt.get("lm_head_excluded") is True
    _validate_receipt_value(
        lm_head_excluded,
        "DoRA target discovery receipt must prove lm_head exclusion",
        {"field": "lm_head_excluded", "value": receipt.get("lm_head_excluded")},
    )

    return DoraSourceGateReceipt(
        status="passed",
        public_adapter_type="dora",
        peft_mechanism="LoraConfig(use_dora=True)",
        source_study_path=evidence.dora_source_study_path,
        probe_receipt_path=evidence.dora_probe_receipt_path,
        lm_head_excluded=lm_head_excluded,
        saved_lora_a_count=saved_lora_a_count,
        saved_lora_b_count=saved_lora_b_count,
        saved_magnitude_vector_count=saved_magnitude_count,
        reloaded_magnitude_vector_count=reloaded_magnitude_count,
        reload_success=reload_success,
        logit_equivalence=logit_equivalence,
        finite_reloaded_eval_logits=finite_reloaded_logits,
        finite_magnitude_vector_gradient=finite_magnitude_gradient,
    )


def _dora_source_study_is_passed(path: Path) -> bool:
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    required_phrases = (
        "public V1 adapter schema uses `adapter.type: dora`",
        "Wave 1B probe evidence completes task 2.3",
        "LoraConfig(use_dora=True)",
    )
    return all(phrase in text for phrase in required_phrases)


def _validate_requested_target_gate(
    adapter_config: AdapterConfig,
    receipt: Mapping[str, Any],
) -> list[str]:
    receipt_towers = receipt.get("target_towers")
    if not isinstance(receipt_towers, list) or not all(
        isinstance(item, str) for item in receipt_towers
    ):
        raise RuntimeContractError(
            "DoRA probe receipt must list target towers",
            code="adapter.dora_target_gate_missing",
            context={"field": "target_towers", "value_type": type(receipt_towers).__name__},
        )
    requested_towers = set(adapter_config.target_towers)
    covered_towers = set(receipt_towers)
    missing_towers = sorted(requested_towers - covered_towers)
    if missing_towers:
        raise RuntimeContractError(
            "DoRA probe receipt does not cover requested target tower",
            code="adapter.dora_target_gate_missing",
            context={
                "requested_target_towers": sorted(requested_towers),
                "receipt_target_towers": sorted(covered_towers),
                "missing_target_towers": missing_towers,
            },
        )
    receipt_policy = receipt.get("target_policy")
    if receipt_policy != adapter_config.target_modules:
        raise RuntimeContractError(
            "DoRA probe receipt target policy does not match requested target policy",
            code="adapter.dora_target_gate_missing",
            context={
                "requested_target_policy": adapter_config.target_modules,
                "receipt_target_policy": receipt_policy,
            },
        )
    selected_targets = _string_list(receipt, "selected_target_modules")
    for tower in sorted(requested_towers):
        if not _targets_have_tower_evidence(selected_targets, tower):
            raise RuntimeContractError(
                "DoRA probe receipt target tower lacks matching module evidence",
                code="adapter.dora_target_gate_missing",
                context={
                    "target_tower": tower,
                    "selected_target_modules": selected_targets,
                },
            )
    return selected_targets


def _validate_receipt_self_target_gate(receipt: Mapping[str, Any]) -> list[str]:
    receipt_towers = receipt.get("target_towers")
    if not isinstance(receipt_towers, list) or not all(
        isinstance(item, str) for item in receipt_towers
    ):
        raise RuntimeContractError(
            "DoRA probe receipt must list target towers",
            code="adapter.dora_target_gate_missing",
            context={
                "field": "target_towers",
                "value_type": type(receipt_towers).__name__,
            },
        )
    receipt_policy = receipt.get("target_policy")
    if receipt_policy != "all_linear":
        raise RuntimeContractError(
            "DoRA probe receipt target policy must be all_linear",
            code="adapter.dora_target_gate_missing",
            context={"receipt_target_policy": receipt_policy},
        )
    selected_targets = _string_list(receipt, "selected_target_modules")
    for tower in sorted(set(receipt_towers)):
        if not _targets_have_tower_evidence(selected_targets, tower):
            raise RuntimeContractError(
                "DoRA probe receipt target tower lacks matching module evidence",
                code="adapter.dora_target_gate_missing",
                context={
                    "target_tower": tower,
                    "selected_target_modules": selected_targets,
                },
            )
    return selected_targets


def _mapping(receipt: Mapping[str, Any], field: str) -> Mapping[str, Any]:
    value = receipt.get(field)
    if not isinstance(value, Mapping):
        raise RuntimeContractError(
            f"DoRA probe receipt must include object field {field}",
            code="adapter.dora_probe_contract",
            context={"field": field, "value_type": type(value).__name__},
        )
    return value


def _string_list(receipt: Mapping[str, Any], field: str) -> list[str]:
    value = receipt.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeContractError(
            f"DoRA probe receipt must list {field}",
            code="adapter.dora_probe_contract",
            context={"field": field, "value_type": type(value).__name__},
        )
    if not value:
        raise RuntimeContractError(
            f"DoRA probe receipt must not have empty {field}",
            code="adapter.dora_probe_contract",
            context={"field": field},
        )
    return value


def _targets_have_tower_evidence(targets: list[str], tower: str) -> bool:
    if tower == "language":
        return any(
            "language_model." in target or target.startswith("language_model.")
            for target in targets
        )
    if tower == "vision":
        return any(target.startswith("visual.") or ".visual." in target for target in targets)
    if tower == "aligner":
        return any(
            ".merger" in target
            or target.startswith("visual.merger")
            or "deepstack_merger" in target
            for target in targets
        )
    return False


def _validate_selected_target_name_coverage(
    *,
    selected_targets: list[str],
    trainable_names: list[str],
    saved_payload_keys: list[str],
    reloaded_magnitude_names: list[str],
) -> None:
    for target in selected_targets:
        fragment = _target_name_fragment(target)
        _require_evidence_for_target(
            names=trainable_names,
            field="trainable_names",
            target=target,
            fragment=fragment,
            needle="lora_A",
        )
        _require_evidence_for_target(
            names=trainable_names,
            field="trainable_names",
            target=target,
            fragment=fragment,
            needle="lora_B",
        )
        _require_evidence_for_target(
            names=trainable_names,
            field="trainable_names",
            target=target,
            fragment=fragment,
            needle="lora_magnitude_vector",
        )
        _require_evidence_for_target(
            names=saved_payload_keys,
            field="save.adapter_payload_keys",
            target=target,
            fragment=fragment,
            needle="lora_A",
        )
        _require_evidence_for_target(
            names=saved_payload_keys,
            field="save.adapter_payload_keys",
            target=target,
            fragment=fragment,
            needle="lora_B",
        )
        _require_evidence_for_target(
            names=saved_payload_keys,
            field="save.adapter_payload_keys",
            target=target,
            fragment=fragment,
            needle="lora_magnitude_vector",
        )
        _require_evidence_for_target(
            names=reloaded_magnitude_names,
            field="magnitude_vectors.reloaded_names",
            target=target,
            fragment=fragment,
            needle="lora_magnitude_vector",
        )


def _target_name_fragment(target: str) -> str:
    return target.removeprefix("model.")


def _require_evidence_for_target(
    *,
    names: list[str],
    field: str,
    target: str,
    fragment: str,
    needle: str,
) -> None:
    if not any(fragment in name and needle in name for name in names):
        raise RuntimeContractError(
            "DoRA probe receipt must include evidence for each selected DoRA target",
            code="adapter.dora_probe_contract",
            context={
                "field": field,
                "target": target,
                "target_fragment": fragment,
                "needle": needle,
            },
        )


def _positive_int(receipt: Mapping[str, Any], field: str, *, label: str) -> int:
    value = receipt.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise RuntimeContractError(
            f"DoRA probe receipt must include positive {label}",
            code="adapter.dora_probe_contract",
            context={"field": field, "value": value},
        )
    return value


def _validate_count_coverage(
    *,
    selected_target_count: int,
    saved_lora_a_count: int,
    saved_lora_b_count: int,
    saved_magnitude_count: int,
    reloaded_magnitude_count: int,
) -> None:
    counts = {
        "selected_target_count": selected_target_count,
        "saved_lora_A_count": saved_lora_a_count,
        "saved_lora_B_count": saved_lora_b_count,
        "saved_lora_magnitude_vector_count": saved_magnitude_count,
        "reloaded_magnitude_vector_count": reloaded_magnitude_count,
    }
    if len(set(counts.values())) != 1:
        raise RuntimeContractError(
            "DoRA target count coverage must match across LoRA and magnitude-vector evidence",
            code="adapter.dora_probe_contract",
            context=counts,
        )


def _validate_name_evidence(
    receipt: Mapping[str, Any],
    field: str,
    *,
    needle: str,
    minimum_count: int,
) -> None:
    value = receipt.get(field)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise RuntimeContractError(
            f"DoRA probe receipt must list {field} with {needle} evidence",
            code="adapter.dora_probe_contract",
            context={"field": field, "value_type": type(value).__name__},
        )
    matching_count = sum(1 for item in value if needle in item)
    if matching_count < minimum_count:
        raise RuntimeContractError(
            f"DoRA probe receipt must include {needle} evidence in {field}",
            code="adapter.dora_probe_contract",
            context={
                "field": field,
                "needle": needle,
                "matching_count": matching_count,
                "minimum_count": minimum_count,
            },
        )


def _validate_receipt_value(
    predicate: bool,
    message: str,
    context: Mapping[str, Any],
) -> None:
    if not predicate:
        raise RuntimeContractError(
            message,
            code="adapter.dora_probe_contract",
            context=context,
        )


def _optional_path_text(path: Path | None) -> str | None:
    return None if path is None else str(path)
