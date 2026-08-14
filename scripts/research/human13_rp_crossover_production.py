#!/usr/bin/env python3
"""Lazy production composition boundary for the Human-13 RP crossover vertical.

Import and factory inspection are standard-library-only. Model, Torch, vLLM,
and experiment owners are imported only after the authorized node runner calls
the returned runtime. The module deliberately exposes one typed composition
boundary for the still-missing live sampled-trajectory pack/logit materializer;
it never substitutes a second sampler, trainer, parser, or matcher.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Protocol
import uuid


NODE_RUNTIME_FACTORY_CONTRACT = "human13_rp_crossover_node_runtime_factory.v1"
UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
QUALIFICATION_SEEDS = tuple(range(30001, 30017))
EVALUATION_RPS = (1.0, 1.10)
QUALIFICATION_LEARNING_RATE_RAY = (
    3.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
)
DEFAULT_QUALIFICATION_LEARNING_RATE = 3.0e-6
CANONICAL_IMAGE_IDS = (
    1584,
    2299,
    2685,
    4134,
    5001,
    6040,
    7511,
    10707,
    13348,
    13923,
    14038,
    14439,
    16228,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_CHECKPOINT_PATH = Path(
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
    "checkpoints/step-2444"
)
BASE_MODEL_PATH = Path(
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
ADAPTER_TENSOR_PATH = SOURCE_CHECKPOINT_PATH / "adapter/adapter_model.safetensors"
SPECIAL_EMBEDDING_TENSOR_PATH = (
    SOURCE_CHECKPOINT_PATH
    / "special_token_embeddings/special_token_embeddings.safetensors"
)
MANIFEST_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-12-human13-k-union-to-greedy-overfit-screen/manifest/"
    "human13-k-union-manifest.json"
)
PANEL_PATH = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
PROMPT_CONFIG_PATH = (
    REPO_ROOT / "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
)
C_LEAF_PATH_BY_RP = {
    1.0: REPO_ROOT
    / "configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/"
    "03_rp100_trajectory_compiler_preservation.yaml",
    1.10: REPO_ROOT
    / "configs/coordexp_swift/research/human13_k_trajectory_rp_crossover/"
    "06_rp110_trajectory_compiler_preservation.yaml",
}

SOURCE_CHECKPOINT_PAYLOAD_SHA256 = (
    "99678ea954c4b37abbf704432dbf43a8df5ce37cd07ebcef4e11f263782dca47"
)
ADAPTER_SHA256 = "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
SPECIAL_EMBEDDING_SHA256 = (
    "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
)
BASE_CONFIG_SHA256 = "c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de"
TOKENIZER_SHA256 = "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8"
PANEL_SHA256 = "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
MANIFEST_SHA256 = "a8f88716c1227054ab29dc698f89462c9369c47c8d6415de3783c0937f60a6fb"
PROMPT_CONFIG_SHA256 = (
    "d2217208bc3e419bc9d8c621b4842da262d216780eb1dfc88eade0edf1358f6b"
)
PROMPT_POLICY_FINGERPRINT = (
    "0b4fa411f289ccc29e3f6b59c65d32689ac04e6a014891cbe2dc2d976de634c1"
)
ALIAS_BANK_SHA256 = "3f64952f000380372f894443da5fe038265c1624d9449a5b0506b84cc57cc6ca"
C_LEAF_SHA256_BY_RP = {
    1.0: "b826fa9f70ec7a5772bdf5c023e31d48b5721d04bc92159c51f606c8e67d38e3",
    1.10: "d7c1a5de691a006a065e4b35ea212b7ea743b3748ae30d0ea739a4830aed0ddf",
}
ADAMW_CONFIG_SHA256 = "b938f99b3cb5145085eedd439e391e89348d58d96844180a73c0ff53e4dbf772"


class ProductionCompositionUnavailable(RuntimeError):
    """Raised when the existing owners cannot yet materialize a live node."""


@dataclass(frozen=True)
class FrozenProductionInputs:
    source_checkpoint_path: str
    source_checkpoint_payload_sha256: str
    base_model_path: str
    adapter_tensor_path: str
    adapter_sha256: str
    special_embedding_tensor_path: str
    special_embedding_sha256: str
    manifest_path: str
    manifest_sha256: str
    panel_path: str
    panel_sha256: str
    tokenizer_path: str
    tokenizer_sha256: str
    prompt_config_path: str
    prompt_config_sha256: str
    prompt_policy_fingerprint: str
    alias_bank_sha256: str
    c_leaf_path: str
    c_leaf_sha256: str
    qualification_learning_rate_ray: tuple[float, ...]
    default_qualification_learning_rate: float


_GLOBAL_LR_DECISION_MARKER = object()
_INJECTED_CPU_LR_DECISION_MARKER = object()


@dataclass(frozen=True, init=False)
class GlobalLearningRateDecision:
    """Content-addressed qualification choice applied globally to the matrix."""

    qualification_decision_sha256: str
    selected_learning_rate: float
    allowed_learning_rates: tuple[float, ...]
    training_rp_learning_rates: tuple[tuple[float, float], ...]
    arm_learning_rates: tuple[tuple[str, float], ...]
    seed_group_learning_rates: tuple[tuple[str, float], ...]
    selection_policy: str
    online_adaptation: bool
    grad_delta_norm_role: str
    qualification_receipt_sha256s: tuple[str, ...]
    selection_reason: str
    measurement_scope: str
    _factory_marker: object

    def __post_init__(self) -> None:
        digest = self.qualification_decision_sha256
        if not (
            isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "qualification learning-rate decision must bind a SHA-256 receipt"
            )
        selected = self.selected_learning_rate
        if (
            isinstance(selected, bool)
            or not isinstance(selected, (int, float))
            or selected not in QUALIFICATION_LEARNING_RATE_RAY
        ):
            raise ValueError(
                "selected learning rate must lie on the exact sealed qualification "
                "dose ray"
            )
        if self.allowed_learning_rates != QUALIFICATION_LEARNING_RATE_RAY:
            raise ValueError("qualification dose ray identity drifted")
        expected_rps = tuple((rp, selected) for rp in EVALUATION_RPS)
        if self.training_rp_learning_rates != expected_rps:
            raise ValueError(
                "one selected learning rate must govern both training RP contracts"
            )
        expected_arms = tuple((arm, selected) for arm in ("A", "B", "C"))
        if self.arm_learning_rates != expected_arms:
            raise ValueError("one selected learning rate must govern every A/B/C arm")
        expected_seed_groups = tuple(
            (group, selected) for group in ("matrix_a", "matrix_b", "matrix_c")
        )
        if self.seed_group_learning_rates != expected_seed_groups:
            raise ValueError(
                "one selected learning rate must govern every matrix seed group"
            )
        if (
            self.selection_policy != "sealed_qualification_global_before_matrix"
            or self.online_adaptation is not False
            or self.grad_delta_norm_role != "covariate_only"
        ):
            raise ValueError(
                "learning rate must be sealed before the matrix; grad/delta norms "
                "are covariates only"
            )
        receipt_sha256s = tuple(self.qualification_receipt_sha256s)
        if len(receipt_sha256s) != 10 or len(set(receipt_sha256s)) != 10:
            raise ValueError(
                "global decision must bind ten distinct mechanical receipts"
            )
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in receipt_sha256s
        ):
            raise ValueError("global decision mechanical receipt identity differs")
        expected_decision = hashlib.sha256(
            _canonical_json_bytes(
                {
                    "schema_version": (
                        "human13_rp_crossover_lr_qualification_bundle.v1"
                    ),
                    "receipt_sha256s": list(receipt_sha256s),
                }
            )
        ).hexdigest()
        if self.qualification_decision_sha256 != expected_decision:
            raise ValueError("global decision digest differs from mechanical receipts")
        if self.selection_reason not in {
            "default_passed_both_rps",
            "default_below_floor_smallest_larger_common_pass",
            "default_above_ceiling_largest_smaller_common_pass",
        }:
            raise ValueError("global decision selection reason differs")
        expected_marker = {
            "live": _GLOBAL_LR_DECISION_MARKER,
            "injected_cpu": _INJECTED_CPU_LR_DECISION_MARKER,
        }.get(self.measurement_scope)
        if expected_marker is None or self._factory_marker is not expected_marker:
            raise ValueError(
                "global decision measurement scope differs from its selector factory"
            )

    @classmethod
    def sealed(cls, *_args: Any, **_kwargs: Any) -> GlobalLearningRateDecision:
        raise ValueError(
            "global learning rate requires all ten mechanical qualification receipts"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "human13_rp_crossover_global_lr_decision.v1",
            "qualification_decision_sha256": self.qualification_decision_sha256,
            "selected_learning_rate": self.selected_learning_rate,
            "allowed_learning_rates": list(self.allowed_learning_rates),
            "training_rp_learning_rates": [
                [rp, learning_rate]
                for rp, learning_rate in self.training_rp_learning_rates
            ],
            "arm_learning_rates": [
                [arm, learning_rate] for arm, learning_rate in self.arm_learning_rates
            ],
            "seed_group_learning_rates": [
                [group, learning_rate]
                for group, learning_rate in self.seed_group_learning_rates
            ],
            "selection_policy": self.selection_policy,
            "online_adaptation": self.online_adaptation,
            "grad_delta_norm_role": self.grad_delta_norm_role,
            "qualification_receipt_sha256s": list(self.qualification_receipt_sha256s),
            "selection_reason": self.selection_reason,
            "measurement_scope": self.measurement_scope,
        }

    @property
    def production_admitted(self) -> bool:
        return (
            self.measurement_scope == "live"
            and self._factory_marker is _GLOBAL_LR_DECISION_MARKER
        )

    @property
    def content_sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def _construct_global_learning_rate_decision(
    *,
    qualification_decision_sha256: str,
    qualification_receipt_sha256s: tuple[str, ...],
    selected_learning_rate: float,
    selection_reason: str,
    measurement_scope: str,
) -> GlobalLearningRateDecision:
    marker = {
        "live": _GLOBAL_LR_DECISION_MARKER,
        "injected_cpu": _INJECTED_CPU_LR_DECISION_MARKER,
    }.get(measurement_scope)
    if marker is None:
        raise ValueError("global decision requires one supported measurement scope")
    result = object.__new__(GlobalLearningRateDecision)
    for field, value in (
        ("qualification_decision_sha256", qualification_decision_sha256),
        ("selected_learning_rate", selected_learning_rate),
        ("allowed_learning_rates", QUALIFICATION_LEARNING_RATE_RAY),
        (
            "training_rp_learning_rates",
            tuple((rp, selected_learning_rate) for rp in EVALUATION_RPS),
        ),
        (
            "arm_learning_rates",
            tuple((arm, selected_learning_rate) for arm in ("A", "B", "C")),
        ),
        (
            "seed_group_learning_rates",
            tuple(
                (group, selected_learning_rate)
                for group in ("matrix_a", "matrix_b", "matrix_c")
            ),
        ),
        ("selection_policy", "sealed_qualification_global_before_matrix"),
        ("online_adaptation", False),
        ("grad_delta_norm_role", "covariate_only"),
        ("qualification_receipt_sha256s", qualification_receipt_sha256s),
        ("selection_reason", selection_reason),
        ("measurement_scope", measurement_scope),
        ("_factory_marker", marker),
    ):
        object.__setattr__(result, field, value)
    result.__post_init__()
    if result._factory_marker is not marker:
        raise ValueError("global learning-rate decision factory identity differs")
    return result


def select_global_learning_rate(
    mechanical_receipts: Sequence[Any],
) -> GlobalLearningRateDecision:
    """Apply the authoritative default/floor/ceiling rule to ten receipts."""

    from scripts.research.human13_rp_crossover_matrix_contracts import (
        DoseMechanicalReceipt,
        QUALIFICATION_LEARNING_RATE_RAY as CONTRACT_RAY,
    )

    if CONTRACT_RAY != QUALIFICATION_LEARNING_RATE_RAY:
        raise RuntimeError("qualification dose ray drifted across production owners")
    receipts = tuple(mechanical_receipts)
    if len(receipts) != len(EVALUATION_RPS) * len(QUALIFICATION_LEARNING_RATE_RAY):
        raise ValueError("global selector requires exactly ten mechanical receipts")
    if any(type(receipt) is not DoseMechanicalReceipt for receipt in receipts):
        raise ValueError("global selector accepts only typed mechanical receipts")
    by_key = {
        (receipt.cell_key.acquisition_key.training_rp, receipt.learning_rate): receipt
        for receipt in receipts
    }
    expected_keys = {
        (rp, learning_rate)
        for rp in EVALUATION_RPS
        for learning_rate in QUALIFICATION_LEARNING_RATE_RAY
    }
    if set(by_key) != expected_keys or len(by_key) != len(receipts):
        raise ValueError("mechanical receipts must cover the exact two-RP dose ray")
    if len({receipt.cell_key.content_sha256 for receipt in receipts}) != len(receipts):
        raise ValueError("every qualification proposal must have a distinct cell key")

    canonical = tuple(
        by_key[(rp, learning_rate)]
        for rp in EVALUATION_RPS
        for learning_rate in QUALIFICATION_LEARNING_RATE_RAY
    )
    measurement_scopes = {receipt.resources.measurement_scope for receipt in canonical}
    if len(measurement_scopes) != 1:
        raise ValueError(
            "all ten qualification receipts require one resource measurement scope"
        )
    measurement_scope = next(iter(measurement_scopes))
    receipt_sha256s = tuple(receipt.content_sha256 for receipt in canonical)
    decision_preimage = {
        "schema_version": "human13_rp_crossover_lr_qualification_bundle.v1",
        "receipt_sha256s": list(receipt_sha256s),
    }
    decision_sha256 = hashlib.sha256(
        _canonical_json_bytes(decision_preimage)
    ).hexdigest()

    def common_pass(learning_rate: float) -> bool:
        return all(
            by_key[(rp, learning_rate)].mechanically_admissible for rp in EVALUATION_RPS
        )

    default_rows = tuple(
        by_key[(rp, DEFAULT_QUALIFICATION_LEARNING_RATE)] for rp in EVALUATION_RPS
    )
    if all(row.mechanically_admissible for row in default_rows):
        selected = DEFAULT_QUALIFICATION_LEARNING_RATE
        reason = "default_passed_both_rps"
    else:
        floor_failed = any(not row.floor_passed for row in default_rows)
        ceiling_failed = any(not row.ceiling_passed for row in default_rows)
        if floor_failed and ceiling_failed:
            raise ValueError("qualification evidence is mixed or non-monotone")
        default_index = QUALIFICATION_LEARNING_RATE_RAY.index(
            DEFAULT_QUALIFICATION_LEARNING_RATE
        )
        if floor_failed:
            if any(
                common_pass(value)
                for value in QUALIFICATION_LEARNING_RATE_RAY[:default_index]
            ):
                raise ValueError("qualification evidence is mixed or non-monotone")
            candidates = tuple(
                value
                for value in QUALIFICATION_LEARNING_RATE_RAY[default_index + 1 :]
                if common_pass(value)
            )
            if not candidates:
                raise ValueError("no common larger dose passes both RP contracts")
            selected = min(candidates)
            reason = "default_below_floor_smallest_larger_common_pass"
        elif ceiling_failed:
            if any(
                common_pass(value)
                for value in QUALIFICATION_LEARNING_RATE_RAY[default_index + 1 :]
            ):
                raise ValueError("qualification evidence is mixed or non-monotone")
            candidates = tuple(
                value
                for value in QUALIFICATION_LEARNING_RATE_RAY[:default_index]
                if common_pass(value)
            )
            if not candidates:
                raise ValueError("no common smaller dose passes both RP contracts")
            selected = max(candidates)
            reason = "default_above_ceiling_largest_smaller_common_pass"
        else:
            raise ValueError("qualification evidence is mixed or non-monotone")
    return _construct_global_learning_rate_decision(
        qualification_decision_sha256=decision_sha256,
        qualification_receipt_sha256s=receipt_sha256s,
        selected_learning_rate=selected,
        selection_reason=reason,
        measurement_scope=measurement_scope,
    )


@dataclass(frozen=True)
class QualificationAcquisition:
    """Typed result returned by the existing acquisition/semantic owners."""

    cell_specs: tuple[Any, ...]
    source_baselines: tuple[Any, ...]
    training_rp: float
    seeds: tuple[int, ...]
    image_ids: tuple[int, ...]
    native_request_count: int
    native_batch_count: int
    manifest_sha256: str
    tokenizer_sha256: str
    prompt_policy_fingerprint: str
    alias_bank_sha256: str
    nested_objective_hashes: Mapping[str, tuple[tuple[str, str], ...]]
    streaming_mode: str


@dataclass(frozen=True)
class AcquisitionReleaseReceipt:
    engine_closed: bool
    model_released: bool
    panel_wide_logits_retained: bool


class LiveNodeComposition(Protocol):
    """One injected bridge over existing Task2-5 and live-model owners.

    ``acquire_qualification`` owns native batch-four sampling, streamed packed
    replay, canonical Task3/4 materialization, and typed publication. It must
    not return until all A/B/C objective identities exist. ``close_acquisition``
    must release the vLLM engine/model before ``services_for_cell`` assembles a
    fresh FA2/MRoPE training model and Task5/HF services.
    """

    def acquire_qualification(
        self, node: Mapping[str, Any], frozen: FrozenProductionInputs
    ) -> QualificationAcquisition: ...

    def close_acquisition(self) -> AcquisitionReleaseReceipt: ...

    def services_for_cell(self, spec: Any) -> Any: ...


class QualificationAcquisitionOwner(Protocol):
    """Exact owner of native acquisition, packed replay, and release."""

    def acquire_qualification(
        self, node: Mapping[str, Any], frozen: FrozenProductionInputs
    ) -> QualificationAcquisition: ...

    def close_acquisition(self) -> AcquisitionReleaseReceipt: ...


class CellRuntimeServicesOwner(Protocol):
    """Exact owner of fresh-model per-cell runtime service construction."""

    def services_for_cell(self, spec: Any) -> Any: ...


class _RequiredQualificationAcquisitionOwner:
    def acquire_qualification(
        self, node: Mapping[str, Any], frozen: FrozenProductionInputs
    ) -> QualificationAcquisition:
        raise ProductionCompositionUnavailable(
            "exact native qualification acquisition owner must be injected; "
            "the production factory will not synthesize model evidence"
        )

    def close_acquisition(self) -> AcquisitionReleaseReceipt:
        return AcquisitionReleaseReceipt(
            engine_closed=True,
            model_released=True,
            panel_wide_logits_retained=False,
        )


class _RequiredCellRuntimeServicesOwner:
    def services_for_cell(self, spec: Any) -> Any:
        raise ProductionCompositionUnavailable(
            "exact fresh-model cell runtime services owner must be injected"
        )


@dataclass(frozen=True)
class ComposedLiveNodeComposition:
    """CPU-composable delegation over the two exact production service owners."""

    acquisition_owner: QualificationAcquisitionOwner
    cell_services_owner: CellRuntimeServicesOwner

    def acquire_qualification(
        self, node: Mapping[str, Any], frozen: FrozenProductionInputs
    ) -> QualificationAcquisition:
        return self.acquisition_owner.acquire_qualification(node, frozen)

    def close_acquisition(self) -> AcquisitionReleaseReceipt:
        return self.acquisition_owner.close_acquisition()

    def services_for_cell(self, spec: Any) -> Any:
        return self.cell_services_owner.services_for_cell(spec)


def _plan_contract() -> dict[str, Any]:
    return {
        "schema_version": NODE_RUNTIME_FACTORY_CONTRACT,
        "arms": ["A", "B", "C"],
        "evaluation_rps": list(EVALUATION_RPS),
        "max_updates": 1,
        "retry_policy": "none",
        "world_size": 1,
        "requires_user_model_gpu_authority": True,
        "actions": {
            "model_loads": 0,
            "gpu_allocations": 0,
            "subprocess_launches": 0,
            "output_roots_created": 0,
        },
    }


def _sha256_file(path: Path, *, label: str) -> str:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str, *, label: str) -> None:
    if _sha256_file(path, label=label) != expected:
        raise ValueError(f"{label} SHA-256 drifted")


def _require_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise FileNotFoundError(f"{label} is not a regular directory: {path}")


def _validate_node(node: Mapping[str, Any]) -> float:
    if not isinstance(node, Mapping):
        raise ValueError("production node must be a sealed mapping")
    rp = node.get("training_rp")
    if isinstance(rp, bool) or not isinstance(rp, (int, float)):
        raise ValueError("qualification training RP must be numeric")
    training_rp = float(rp)
    expected_node_id = {
        1.0: "rp100:qualification",
        1.10: "rp110:qualification",
    }.get(training_rp)
    if (
        expected_node_id is None
        or node.get("node_id") != expected_node_id
        or node.get("phase") != "qualification"
        or tuple(node.get("seeds", ())) != QUALIFICATION_SEEDS
    ):
        raise ValueError("production factory accepts only an exact qualification node")
    acquisition = node.get("acquisition_key")
    if not isinstance(acquisition, Mapping) or (
        float(acquisition.get("training_rp", -1.0)),
        acquisition.get("seed_group_id"),
        acquisition.get("phase"),
        tuple(acquisition.get("seeds", ())),
    ) != (training_rp, "qualification", "qualification", QUALIFICATION_SEEDS):
        raise ValueError("qualification acquisition identity or seeds drifted")
    cells = node.get("cells")
    if not isinstance(cells, Sequence) or isinstance(cells, (str, bytes)):
        raise ValueError("qualification node cells are malformed")
    if len(cells) != 5 or any(not isinstance(cell, Mapping) for cell in cells):
        raise ValueError("qualification node must contain the exact five C doses")
    if tuple(cell.get("learning_rate") for cell in cells) != (
        QUALIFICATION_LEARNING_RATE_RAY
    ):
        raise ValueError("qualification node dose ray drifted")
    roots: list[str] = []
    for cell in cells:
        cell_key = cell.get("cell_key")
        if not isinstance(cell_key, Mapping) or (
            cell_key.get("arm_id"),
            cell_key.get("qualification_learning_rate"),
        ) != ("C", cell.get("learning_rate")):
            raise ValueError("qualification node must execute preservation arm C only")
        if (
            cell_key.get("acquisition_key") != acquisition
            or tuple(cell.get("objective_components", ()))
            != ("trajectory", "compiler", "preservation")
            or tuple(cell.get("evaluation_rps", ())) != EVALUATION_RPS
            or cell.get("world_size") != 1
            or cell.get("max_updates") != 1
            or cell.get("retry_policy") != "none"
            or cell.get("source") != "fresh"
            or cell.get("optimizer") != "fresh_adamw"
            or cell.get("global_learning_rate_decision_sha256") is not None
            or not cell.get("resolved_leaf_config_sha256")
            or cell.get("adamw_config_sha256")
            != _resolved_adamw_config_sha256(float(cell["learning_rate"]))
        ):
            raise ValueError("qualification C cell execution contract drifted")
        root = str(cell.get("output_root", ""))
        roots.append(root)
        if Path(root).exists():
            raise FileExistsError("refusing to reuse a qualification cell output root")
    if len(set(roots)) != 5:
        raise ValueError("qualification C proposals require unique immutable roots")
    return training_rp


def _validate_frozen_inputs(training_rp: float) -> FrozenProductionInputs:
    _require_directory(SOURCE_CHECKPOINT_PATH, label="Source checkpoint")
    _require_directory(BASE_MODEL_PATH, label="Source base model")
    from scripts.research.human13_live_eval import checkpoint_payload_sha256

    if (
        checkpoint_payload_sha256(SOURCE_CHECKPOINT_PATH)
        != SOURCE_CHECKPOINT_PAYLOAD_SHA256
    ):
        raise ValueError("Source checkpoint payload SHA-256 drifted")
    _require_hash(ADAPTER_TENSOR_PATH, ADAPTER_SHA256, label="Source adapter")
    _require_hash(
        SPECIAL_EMBEDDING_TENSOR_PATH,
        SPECIAL_EMBEDDING_SHA256,
        label="Source special-token embeddings",
    )
    _require_hash(
        BASE_MODEL_PATH / "config.json", BASE_CONFIG_SHA256, label="base config"
    )
    _require_hash(
        BASE_MODEL_PATH / "tokenizer.json", TOKENIZER_SHA256, label="tokenizer"
    )
    _require_hash(MANIFEST_PATH, MANIFEST_SHA256, label="Human-13 manifest")
    _require_hash(PANEL_PATH, PANEL_SHA256, label="Human-13 panel")
    _require_hash(
        PROMPT_CONFIG_PATH, PROMPT_CONFIG_SHA256, label="Source prompt config"
    )
    leaf_path = C_LEAF_PATH_BY_RP[training_rp]
    leaf_sha256 = C_LEAF_SHA256_BY_RP[training_rp]
    _require_hash(leaf_path, leaf_sha256, label="qualification C leaf config")
    return FrozenProductionInputs(
        source_checkpoint_path=str(SOURCE_CHECKPOINT_PATH),
        source_checkpoint_payload_sha256=SOURCE_CHECKPOINT_PAYLOAD_SHA256,
        base_model_path=str(BASE_MODEL_PATH),
        adapter_tensor_path=str(ADAPTER_TENSOR_PATH),
        adapter_sha256=ADAPTER_SHA256,
        special_embedding_tensor_path=str(SPECIAL_EMBEDDING_TENSOR_PATH),
        special_embedding_sha256=SPECIAL_EMBEDDING_SHA256,
        manifest_path=str(MANIFEST_PATH),
        manifest_sha256=MANIFEST_SHA256,
        panel_path=str(PANEL_PATH),
        panel_sha256=PANEL_SHA256,
        tokenizer_path=str(BASE_MODEL_PATH / "tokenizer.json"),
        tokenizer_sha256=TOKENIZER_SHA256,
        prompt_config_path=str(PROMPT_CONFIG_PATH),
        prompt_config_sha256=PROMPT_CONFIG_SHA256,
        prompt_policy_fingerprint=PROMPT_POLICY_FINGERPRINT,
        alias_bank_sha256=ALIAS_BANK_SHA256,
        c_leaf_path=str(leaf_path),
        c_leaf_sha256=leaf_sha256,
        qualification_learning_rate_ray=QUALIFICATION_LEARNING_RATE_RAY,
        default_qualification_learning_rate=DEFAULT_QUALIFICATION_LEARNING_RATE,
    )


def _validate_release(receipt: AcquisitionReleaseReceipt) -> None:
    if not isinstance(receipt, AcquisitionReleaseReceipt) or (
        receipt.engine_closed,
        receipt.model_released,
        receipt.panel_wide_logits_retained,
    ) != (True, True, False):
        raise RuntimeError(
            "acquisition must release vLLM and retain no panel-wide logits before training"
        )


def _validate_acquisition(
    acquired: QualificationAcquisition,
    *,
    node: Mapping[str, Any],
    training_rp: float,
) -> tuple[Any, ...]:
    if not isinstance(acquired, QualificationAcquisition):
        raise TypeError("live composition must return QualificationAcquisition")
    if (
        acquired.training_rp != training_rp
        or acquired.seeds != QUALIFICATION_SEEDS
        or acquired.image_ids != CANONICAL_IMAGE_IDS
        or acquired.native_request_count != 208
        or acquired.native_batch_count != 52
        or acquired.manifest_sha256 != MANIFEST_SHA256
        or acquired.tokenizer_sha256 != TOKENIZER_SHA256
        or acquired.prompt_policy_fingerprint != PROMPT_POLICY_FINGERPRINT
        or acquired.alias_bank_sha256 != ALIAS_BANK_SHA256
        or acquired.streaming_mode != "per_image_or_pack"
    ):
        raise ValueError(
            "qualification acquisition or frozen semantic identity drifted"
        )
    nested = acquired.nested_objective_hashes
    if not isinstance(nested, Mapping) or set(nested) != {"A", "B", "C"}:
        raise ValueError(
            "qualification must materialize exact A/B/C objective identities"
        )
    a = tuple(nested["A"])
    b = tuple(nested["B"])
    c = tuple(nested["C"])
    if (
        tuple(name for name, _ in a) != ("trajectory",)
        or tuple(name for name, _ in b) != ("trajectory", "compiler")
        or tuple(name for name, _ in c) != ("trajectory", "compiler")
        or b[:1] != a
        or c != b
    ):
        raise ValueError("qualification A/B/C objective bytes are not exactly nested")
    from scripts.research.human13_rp_crossover_matrix_contracts import (
        AcquisitionKey,
        CellSpec,
        SourceBaselineRef,
    )

    baselines = tuple(acquired.source_baselines)
    if (
        len(baselines) != 2
        or any(not isinstance(item, SourceBaselineRef) for item in baselines)
        or tuple(item.evaluation_rp for item in baselines) != EVALUATION_RPS
        or any(
            item.checkpoint_sha256 != SOURCE_CHECKPOINT_PAYLOAD_SHA256
            or item.image_ids != CANONICAL_IMAGE_IDS
            for item in baselines
        )
    ):
        raise ValueError(
            "qualification requires two fresh deterministic Source baselines per RP"
        )
    specs = tuple(acquired.cell_specs)
    if len(specs) != 5 or any(not isinstance(spec, CellSpec) for spec in specs):
        raise ValueError("qualification acquisition must return five C CellSpecs")
    if tuple(spec.learning_rate for spec in specs) != QUALIFICATION_LEARNING_RATE_RAY:
        raise ValueError("qualification CellSpecs differ from the exact dose ray")
    acquisition_key = AcquisitionKey.from_dict(node["acquisition_key"])
    for spec, planned in zip(specs, node["cells"], strict=True):
        if (
            spec.cell_key.acquisition_key != acquisition_key
            or spec.cell_key.arm_id != "C"
            or spec.cell_key.qualification_learning_rate != spec.learning_rate
            or spec.objective_component_hashes != c
            or spec.leaf_config_sha256 != C_LEAF_SHA256_BY_RP[training_rp]
            or spec.source_checkpoint_sha256 != SOURCE_CHECKPOINT_PAYLOAD_SHA256
            or spec.shared_evidence.manifest_sha256 != MANIFEST_SHA256
            or spec.output_root != planned["output_root"]
            or spec.adamw_config_sha256 != planned["adamw_config_sha256"]
            or spec.resolved_leaf_config_sha256
            != planned["resolved_leaf_config_sha256"]
            or spec.global_learning_rate_decision_sha256 is not None
        ):
            raise ValueError(
                "qualification C CellSpec differs from frozen live evidence"
            )
    return specs


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _resolved_adamw_config_sha256(learning_rate: float) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "schema_version": "human13_rp_crossover_adamw_config.v1",
                "name": "adamw_torch",
                "learning_rate": learning_rate,
                "betas": [0.9, 0.999],
                "epsilon": 1.0e-8,
                "weight_decay": 0.0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    if target.exists():
        raise FileExistsError(f"refusing to overwrite production receipt: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{uuid.uuid4().hex}.tmp"
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        directory = os.open(target.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists():
            temporary.unlink()


class ProductionNodeRuntime:
    """Node-level phase owner used by the final launcher/runner contract."""

    def __init__(
        self, node: Mapping[str, Any], *, composition: LiveNodeComposition
    ) -> None:
        self._node = node
        self._composition = composition
        self._acquisition_released = False
        self._spec_output_roots: dict[str, str] = {}

    def acquire_cell_specs(self, node: Mapping[str, Any]) -> tuple[Any, ...]:
        if node is not self._node and dict(node) != dict(self._node):
            raise ValueError("node runtime received a different sealed node")
        training_rp = _validate_node(node)
        frozen = _validate_frozen_inputs(training_rp)
        acquired: QualificationAcquisition | None = None
        try:
            acquired = self._composition.acquire_qualification(node, frozen)
        finally:
            release = self._composition.close_acquisition()
            _validate_release(release)
            self._acquisition_released = True
        specs = _validate_acquisition(acquired, node=node, training_rp=training_rp)
        self._spec_output_roots = {
            spec.cell_key.content_sha256: spec.output_root for spec in specs
        }
        return specs

    def services_for_cell(self, spec: Any) -> Any:
        if not self._acquisition_released:
            raise RuntimeError("training model cannot load before acquisition release")
        if spec.cell_key.content_sha256 not in self._spec_output_roots:
            raise ValueError("cell service requested for an unacquired CellSpec")
        return self._composition.services_for_cell(spec)

    def write_cell_receipt(self, receipt: Any) -> None:
        key = receipt.cell_key.content_sha256
        try:
            output_root = self._spec_output_roots[key]
        except KeyError as exc:
            raise ValueError("cell receipt is not bound to this node") from exc
        _atomic_write_json(Path(output_root) / "cell-receipt.json", receipt.to_dict())

    def write_node_terminal_receipt(
        self, path: str, payload: Mapping[str, Any]
    ) -> None:
        _atomic_write_json(path, payload)


def _declare_runtime_factory(factory: Any) -> Any:
    """Declare the runner contract without importing its Torch-bearing module."""

    factory.node_runtime_factory_contract = NODE_RUNTIME_FACTORY_CONTRACT
    factory.plan_contract = _plan_contract
    return factory


@_declare_runtime_factory
def create_node_runtime(
    node: Mapping[str, Any],
    *,
    _composition: LiveNodeComposition | None = None,
    _acquisition_owner: QualificationAcquisitionOwner | None = None,
    _cell_services_owner: CellRuntimeServicesOwner | None = None,
) -> ProductionNodeRuntime:
    """Create one node runtime; all validation/live action remains execute-time."""

    if _composition is not None and (
        _acquisition_owner is not None or _cell_services_owner is not None
    ):
        raise ValueError("inject either one composition or its exact owners, not both")
    composition = _composition or ComposedLiveNodeComposition(
        acquisition_owner=(
            _acquisition_owner or _RequiredQualificationAcquisitionOwner()
        ),
        cell_services_owner=(
            _cell_services_owner or _RequiredCellRuntimeServicesOwner()
        ),
    )
    return ProductionNodeRuntime(node, composition=composition)


__all__ = [
    "ALIAS_BANK_SHA256",
    "ADAMW_CONFIG_SHA256",
    "AcquisitionReleaseReceipt",
    "C_LEAF_SHA256_BY_RP",
    "CellRuntimeServicesOwner",
    "ComposedLiveNodeComposition",
    "DEFAULT_QUALIFICATION_LEARNING_RATE",
    "FrozenProductionInputs",
    "GlobalLearningRateDecision",
    "MANIFEST_SHA256",
    "ProductionCompositionUnavailable",
    "ProductionNodeRuntime",
    "PROMPT_POLICY_FINGERPRINT",
    "QUALIFICATION_LEARNING_RATE_RAY",
    "QualificationAcquisition",
    "QualificationAcquisitionOwner",
    "SOURCE_CHECKPOINT_PAYLOAD_SHA256",
    "TOKENIZER_SHA256",
    "create_node_runtime",
    "select_global_learning_rate",
]
