#!/usr/bin/env python3
"""Exact live-model assembly boundary for the Human-13 overfit probe.

``build_human13_live_model_plan`` and
``validate_human13_live_model_plan`` are intentionally standard-library-only
operations.  They load no model/runtime package and allocate no accelerator.
Only ``assemble_human13_live_model`` crosses the live boundary.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Literal, Protocol, cast
from weakref import ReferenceType, ref


LIVE_PLAN_SCHEMA_VERSION = "human13_live_model_plan.v1"
VALIDATION_SCHEMA_VERSION = "human13_live_model_validation.v1"
UNIT_ID = "2026-08-12-human13-k-union-to-greedy-overfit-screen"
SUCCESSOR_UNIT_ID = "2026-08-13-human13-row-contrast-geometry-preservation-successor"
ON_POLICY_UNIT_ID = "2026-08-13-human13-on-policy-first-bottleneck-successor"
RP_CROSSOVER_UNIT_ID = "2026-08-14-human13-k-trajectory-rp-crossover-screen"
SOURCE_CHECKPOINT_PATH = (
    "/data/CoordExp/outputs/research/eight-coordinate-bbox-supervision/"
    "2026-08-05-closeout/artifacts/training/four-coordinate-xy/"
    "checkpoints/step-2444"
)
SOURCE_BASE_MODEL_PATH = (
    "/data/Qwen3-VL/model_cache/models/Qwen/"
    "Qwen3-VL-2B-Instruct-coordexp-natural-adjacent"
)
SOURCE_ADAPTER_PATH = f"{SOURCE_CHECKPOINT_PATH}/adapter"
SOURCE_SPECIAL_EMBEDDING_PATH = f"{SOURCE_CHECKPOINT_PATH}/special_token_embeddings"
SOURCE_ADAPTER_SHA256 = (
    "49aa206cb43ebc61bf0413e6de6d81cea725549c71d38b596826fda5f523b5da"
)
SOURCE_SPECIAL_EMBEDDING_SHA256 = (
    "a41cbb2fd05e3f6b7ad43f28f9fc5ce973477812435acf8d79d9b2b61f19e2f2"
)
SOURCE_BASE_CONFIG_SHA256 = (
    "c7d172360d0ff881db59a6f34865c379bbef40d976ad79cfe5fbbf50483655de"
)
SOURCE_TOKENIZER_SHA256 = (
    "ca7e80dee65c629af3b314e76a7587490db3f4e6412df4af9f3b690a9e9916f8"
)
HUMAN13_PANEL_SHA256 = (
    "5c6cc95965c6dd24d7f61f09a0c56edb71eb5a9a05664fa7d26269718f741f23"
)
HUMAN13_PROMPT_POLICY_FINGERPRINT = (
    "0b4fa411f289ccc29e3f6b59c65d32689ac04e6a014891cbe2dc2d976de634c1"
)
HUMAN13_PANEL_PATH = (
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-08-05-static-dynamic-owner-interface-crossover/inputs/"
    "human-refined-13.geo_sorted_xy.coord.jsonl"
)
HUMAN13_SOURCE_INFER_CONFIG = (
    "configs/coordexp_swift/infer/"
    "qwen3_vl_2b_static_dynamic_owner_interface_s_step2444_h0.yaml"
)
HUMAN13_IMAGE_IDS = (
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
MILESTONES = (0, 1, 2, 4, 8, 16)
SUCCESSOR_MILESTONES = (0, 1, 2)
ON_POLICY_MILESTONES = tuple(range(9))
RP_CROSSOVER_MILESTONES = (0, 1)
RP_CROSSOVER_LEARNING_RATE_RAY = (
    3.0e-7,
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
)
CHECKPOINT_STEPS = (1, 2, 4, 8, 16)
ZERO_MODEL_ACTIONS = {
    "model_imports": 0,
    "model_loads": 0,
    "forwards": 0,
    "backwards": 0,
    "optimizer_constructions": 0,
    "optimizer_steps": 0,
    "checkpoint_writes": 0,
    "gpu_allocations": 0,
}
_SOURCE_ADAPTER_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)
_UPDATED_ARM_IDS = frozenset(
    {
        "full_gt_capacity",
        "A0",
        "A1",
        "A3",
        "A4",
        "A6",
        "A7",
        "A8-prime",
        "R1",
        "R2",
        "O-Full-Safe",
        "O-First-Safe",
        "A",
        "B",
        "C",
    }
)


class Human13LiveModelError(RuntimeError):
    """Raised before or during an invalid Human-13 live assembly."""


@dataclass(frozen=True)
class Human13SourceContract:
    checkpoint_path: str
    base_model_path: str
    adapter_path: str
    special_embedding_path: str
    adapter_sha256: str
    special_embedding_sha256: str


@dataclass(frozen=True)
class Human13LiveModelPlan:
    schema_version: str
    unit_id: str
    arm_id: str
    source: Human13SourceContract
    mixed_precision: Literal["bf16", "fp32"]
    attn_implementation: Literal["flash_attention_2", "sdpa"]
    patch_embed_linearization: Literal["enabled"]
    adapter_seed_mode: Literal["warm_start_expand_dora"]
    adapter_target_towers: tuple[Literal["language"], ...]
    adapter_target_modules: Literal["all_linear"]
    adapter_rank: int
    adapter_alpha: int
    adapter_dropout: float
    adapter_bias: Literal["none"]
    freeze_special_token_delta: bool
    optimizer_name: Literal["adamw_torch"]
    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float
    scheduler_name: Literal["cosine_with_warmup"]
    scheduler_warmup_steps: int
    scheduler_horizon_updates: int
    max_grad_norm: float
    world_size: int
    milestones: tuple[int, ...]
    learning_rate_resolution: str = "not_applicable"
    global_learning_rate_decision_sha256: str | None = None
    resolved_plan_sha256: str | None = None

    def to_artifact_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["model_actions"] = dict(ZERO_MODEL_ACTIONS)
        return payload


def _resolved_plan_sha256(plan: Human13LiveModelPlan) -> str:
    payload = plan.to_artifact_dict()
    payload.pop("model_actions")
    payload["resolved_plan_sha256"] = None
    return hashlib.sha256(
        (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True)
class Human13PlanValidationReceipt:
    schema_version: str
    arm_id: str
    adapter_tensor_sha256: str
    special_embedding_tensor_sha256: str
    base_config_sha256: str
    tokenizer_sha256: str
    model_actions: Mapping[str, int]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "arm_id": self.arm_id,
            "adapter_tensor_sha256": self.adapter_tensor_sha256,
            "special_embedding_tensor_sha256": self.special_embedding_tensor_sha256,
            "base_config_sha256": self.base_config_sha256,
            "tokenizer_sha256": self.tokenizer_sha256,
            "model_actions": dict(self.model_actions),
        }


@dataclass(frozen=True)
class Human13LiveAssembly:
    plan: Human13LiveModelPlan
    validation: Human13PlanValidationReceipt
    components: Any
    accelerator: Any
    model: Any
    adapter_result: Any
    special_token_result: Any
    optimizer: Any
    scheduler: Any
    optimizer_group_plan: Any
    trainable_surface_receipt: Any
    runtime: Any
    memory_saver_receipt: Any


@dataclass(frozen=True)
class Human13LoadedSpecialTokenResult:
    """Special-token install result plus the exact loaded tensor identity."""

    model: Any
    shared_embed_delta: Any
    receipt: Any
    load_receipt: Any
    loaded_tensor_sha256: str


_ADMITTED_LIVE_ASSEMBLIES: dict[
    int, tuple[ReferenceType[object], str]
] = {}


def _loaded_adapter_tensor_sha256(adapter_result: Any) -> str | None:
    warm_start = getattr(getattr(adapter_result, "receipt", None), "warm_start", None)
    if not isinstance(warm_start, Mapping):
        return None
    value = warm_start.get("source_adapter_tensor_sha256")
    return value if isinstance(value, str) else None


def _loaded_special_embedding_tensor_sha256(special_result: Any) -> str | None:
    value = getattr(special_result, "loaded_tensor_sha256", None)
    return value if isinstance(value, str) else None


def _live_assembly_fingerprint(assembly: Human13LiveAssembly) -> str:
    try:
        surface = assembly.trainable_surface_receipt.to_artifact_dict()
    except AttributeError as exc:
        raise Human13LiveModelError(
            "trainable-surface receipt is unavailable"
        ) from exc
    payload = {
        "plan": assembly.plan.to_artifact_dict(),
        "validation": assembly.validation.to_artifact_dict(),
        "surface": surface,
        "loaded_adapter_tensor_sha256": _loaded_adapter_tensor_sha256(
            assembly.adapter_result
        ),
        "loaded_special_embedding_tensor_sha256": (
            _loaded_special_embedding_tensor_sha256(assembly.special_token_result)
        ),
        "model_object_id": id(assembly.model),
        "runtime_model_object_id": id(getattr(assembly.runtime, "model", None)),
        "adapter_model_object_id": id(getattr(assembly.adapter_result, "model", None)),
        "special_model_object_id": id(
            getattr(assembly.special_token_result, "model", None)
        ),
        "runtime_optimizer_object_id": id(
            getattr(assembly.runtime, "optimizer", None)
        ),
        "runtime_scheduler_object_id": id(
            getattr(assembly.runtime, "scheduler", None)
        ),
        "delta_requires_grad": getattr(
            getattr(assembly.special_token_result, "shared_embed_delta", None),
            "requires_grad",
            None,
        ),
    }
    return hashlib.sha256(
        (json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()
    ).hexdigest()


def _seal_human13_live_assembly(
    assembly: Human13LiveAssembly,
) -> Human13LiveAssembly:
    fingerprint = _live_assembly_fingerprint(assembly)
    identity = id(assembly)

    def cleanup(_: ReferenceType[object], *, key: int = identity) -> None:
        _ADMITTED_LIVE_ASSEMBLIES.pop(key, None)

    _ADMITTED_LIVE_ASSEMBLIES[identity] = (ref(assembly, cleanup), fingerprint)
    return assembly


def require_admitted_human13_live_assembly(
    assembly: Human13LiveAssembly,
) -> None:
    """Reject assemblies not issued by the exact live-model builder."""

    if type(assembly) is not Human13LiveAssembly:
        raise Human13LiveModelError("live assembly has the wrong value type")
    entry = _ADMITTED_LIVE_ASSEMBLIES.get(id(assembly))
    if (
        entry is None
        or entry[0]() is not assembly
        or entry[1] != _live_assembly_fingerprint(assembly)
    ):
        raise Human13LiveModelError(
            "live assembly was not issued by the admitted Human-13 builder"
        )


def validate_human13_live_assembly_values(
    assembly: Human13LiveAssembly,
) -> None:
    """Revalidate the sealed builder surface at a downstream live boundary."""

    require_admitted_human13_live_assembly(assembly)
    _require_frozen_plan(assembly.plan)
    _require_language_only_surface(assembly.trainable_surface_receipt)
    validation = assembly.validation
    if (
        type(validation) is not Human13PlanValidationReceipt
        or validation.arm_id != assembly.plan.arm_id
        or validation.schema_version != VALIDATION_SCHEMA_VERSION
        or validation.adapter_tensor_sha256 != assembly.plan.source.adapter_sha256
        or validation.special_embedding_tensor_sha256
        != assembly.plan.source.special_embedding_sha256
        or _loaded_adapter_tensor_sha256(assembly.adapter_result)
        != validation.adapter_tensor_sha256
        or _loaded_special_embedding_tensor_sha256(assembly.special_token_result)
        != validation.special_embedding_tensor_sha256
        or getattr(
            getattr(assembly.special_token_result, "shared_embed_delta", None),
            "requires_grad",
            None,
        )
        is not False
    ):
        raise Human13LiveModelError(
            "live assembly validation and loaded Source identities differ"
        )


@dataclass(frozen=True)
class Human13ParityRuntime:
    """Inference-only runtime surface required by exact-history replay."""

    accelerator: Any
    world_size: int = 1


@dataclass(frozen=True)
class Human13ParityAssembly:
    """One fp32/SDPA Source model without any training construction."""

    plan: Human13LiveModelPlan
    validation: Human13PlanValidationReceipt
    components: Any
    accelerator: Any
    model: Any
    adapter_result: Any
    special_token_result: Any
    runtime: Human13ParityRuntime


@dataclass(frozen=True)
class Human13CheckpointReadback:
    step: int
    checkpoint_dir: Path
    adapter_fingerprint: str
    special_embedding_fingerprint: str
    adapter_identity: Mapping[str, Any]
    special_embedding_identity: Mapping[str, Any]

    def to_artifact_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "checkpoint_dir": str(self.checkpoint_dir),
            "adapter_fingerprint": self.adapter_fingerprint,
            "special_embedding_fingerprint": self.special_embedding_fingerprint,
            "adapter_identity": dict(self.adapter_identity),
            "special_embedding_identity": dict(self.special_embedding_identity),
        }


class Human13AssemblyBackend(Protocol):
    """Injectable live operations; tests provide CPU-only doubles."""

    def create_accelerator(self, plan: Human13LiveModelPlan) -> Any: ...

    def validate_accelerator(
        self, accelerator: Any, plan: Human13LiveModelPlan
    ) -> None: ...

    def load_qwen(self, plan: Human13LiveModelPlan) -> Any: ...

    def warm_start_language_dora(
        self,
        model: Any,
        components: Any,
        plan: Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any: ...

    def load_and_freeze_special_token_delta(
        self,
        model: Any,
        components: Any,
        plan: Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any: ...

    def enable_memory_savers(self, model: Any) -> Any: ...

    def build_optimizer(
        self,
        model: Any,
        adapter_result: Any,
        plan: Human13LiveModelPlan,
    ) -> tuple[Any, Any, Any]: ...

    def build_trainable_surface_receipt(
        self,
        model: Any,
        adapter_result: Any,
        special_result: Any,
        optimizer_group_plan: Any,
    ) -> Any: ...

    def build_runtime(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        accelerator: Any,
        plan: Human13LiveModelPlan,
        pack_count: int,
    ) -> Any: ...


class Human13ParityAssemblyBackend(Protocol):
    """Only the lower model-loading seams needed by parity replay."""

    def create_accelerator(self, plan: Human13LiveModelPlan) -> Any: ...

    def validate_accelerator(
        self, accelerator: Any, plan: Human13LiveModelPlan
    ) -> None: ...

    def load_qwen(self, plan: Human13LiveModelPlan) -> Any: ...

    def warm_start_language_dora(
        self,
        model: Any,
        components: Any,
        plan: Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any: ...

    def load_and_freeze_special_token_delta(
        self,
        model: Any,
        components: Any,
        plan: Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any: ...


def _accelerator_mixed_precision(plan: Human13LiveModelPlan) -> str:
    # Accelerate has no "fp32" token: full precision is mixed_precision="no".
    return "no" if plan.mixed_precision == "fp32" else plan.mixed_precision


class DefaultHuman13AssemblyBackend:
    """Thin adapter over the accepted CoordExp-Swift assembly primitives."""

    def create_accelerator(self, plan: Human13LiveModelPlan) -> Any:
        from accelerate import Accelerator

        return Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=_accelerator_mixed_precision(plan),
        )

    def validate_accelerator(
        self, accelerator: Any, plan: Human13LiveModelPlan
    ) -> None:
        from src.runtime import validate_accelerator_runtime

        validate_accelerator_runtime(
            accelerator,
            expected_mixed_precision=_accelerator_mixed_precision(plan),
        )

    def load_qwen(self, plan: Human13LiveModelPlan) -> Any:
        from src.qwen.runtime_loading import (
            QwenLoadOptions,
            load_qwen_components_from_options,
        )

        return load_qwen_components_from_options(
            QwenLoadOptions(
                base_model=plan.source.base_model_path,
                dtype=plan.mixed_precision,
                attn_implementation=plan.attn_implementation,
                patch_embed_linearization=plan.patch_embed_linearization,
                load_model=True,
            )
        )

    def warm_start_language_dora(
        self,
        model: Any,
        components: Any,
        plan: Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any:
        from src.adapters import (
            build_adapter_setup_plan,
            load_default_adapter_source_gate_evidence,
            setup_dora_adapter,
        )
        from src.config.models import AdapterConfig

        adapter_config = AdapterConfig(
            type="dora",
            seed_mode=plan.adapter_seed_mode,
            path=None,
            source_adapter_path=plan.source.adapter_path,
            repaired_embedding_payload_path=plan.source.special_embedding_path,
            target_towers=plan.adapter_target_towers,
            target_modules=plan.adapter_target_modules,
            rank=plan.adapter_rank,
            alpha=plan.adapter_alpha,
            dropout=plan.adapter_dropout,
            bias=plan.adapter_bias,
        )
        evidence = load_default_adapter_source_gate_evidence(repo_root)
        adapter_plan = build_adapter_setup_plan(
            adapter_config,
            evidence,
            base_model_path=components.base_model_path,
        )
        return setup_dora_adapter(model, adapter_plan)

    def load_and_freeze_special_token_delta(
        self,
        model: Any,
        components: Any,
        plan: Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any:
        from src.config.models import (
            SpecialTokenEmbeddingGroupsConfig,
            SpecialTokenEmbeddingsConfig,
        )
        from src.qwen import (
            build_default_special_token_selection,
            install_special_token_embedding_deltas,
            load_special_token_embedding_deltas,
        )
        from src.qwen.special_token_embeddings import (
            load_default_special_token_embedding_source_gate_evidence,
        )

        special_config = SpecialTokenEmbeddingsConfig(
            groups=SpecialTokenEmbeddingGroupsConfig(
                coordinate_tokens="default_coord_0_999",
                wrapper_tokens="default_object_box_wrappers",
            )
        )
        selection = build_default_special_token_selection(
            special_config,
            components.token_identity,
        )
        source_gate = load_default_special_token_embedding_source_gate_evidence(
            repo_root,
            selection=selection,
        )
        result = install_special_token_embedding_deltas(
            model,
            selection,
            source_gate=source_gate,
        )
        load_receipt = load_special_token_embedding_deltas(
            result,
            Path(plan.source.special_embedding_path),
            expected_base_model_path=components.base_model_path,
            expected_base_config_sha256=components.base_config_sha256,
            expected_tokenizer_sha256=components.tokenizer_sha256,
        )
        result.shared_embed_delta.requires_grad_(False)
        return Human13LoadedSpecialTokenResult(
            model=result.model,
            shared_embed_delta=result.shared_embed_delta,
            receipt=result.receipt,
            load_receipt=load_receipt,
            loaded_tensor_sha256=_sha256_file(
                Path(plan.source.special_embedding_path)
                / "special_token_embeddings.safetensors"
            ),
        )

    def enable_memory_savers(self, model: Any) -> Any:
        from src.training.pipeline import enable_training_memory_savers

        return enable_training_memory_savers(model)

    def build_optimizer(
        self,
        model: Any,
        adapter_result: Any,
        plan: Human13LiveModelPlan,
    ) -> tuple[Any, Any, Any]:
        from src.config.models import (
            AdapterOptimizerGroupsConfig,
            OptimizerConfig,
            OptimizerGroupConfig,
            OptimizerGroupsConfig,
            SchedulerConfig,
        )
        from src.optim import (
            build_optimizer_and_scheduler,
            build_optimizer_group_plan,
        )

        language_group = OptimizerGroupConfig(
            lr=plan.learning_rate,
            weight_decay=plan.weight_decay,
        )
        optimizer_config = OptimizerConfig(
            name=plan.optimizer_name,
            betas=plan.betas,
            epsilon=plan.epsilon,
            kwargs={},
            groups=OptimizerGroupsConfig(
                adapters=AdapterOptimizerGroupsConfig(
                    language=language_group,
                    aligner=None,
                    vision=None,
                ),
                # Required by the generic typed config, but excluded from the
                # group plan because the loaded source delta is frozen.
                token_embeddings=OptimizerGroupConfig(
                    lr=plan.learning_rate,
                    weight_decay=plan.weight_decay,
                ),
            ),
            scheduler=SchedulerConfig(
                name=plan.scheduler_name,
                warmup_ratio=None,
                warmup_steps=plan.scheduler_warmup_steps,
                kwargs={},
            ),
        )
        group_plan = build_optimizer_group_plan(
            model,
            optimizer_config,
            adapter_receipt=adapter_result.receipt,
            special_token_receipt=None,
        )
        optimizer, scheduler = build_optimizer_and_scheduler(
            optimizer_config,
            group_plan,
            total_training_steps=plan.scheduler_horizon_updates,
        )
        return optimizer, scheduler, group_plan

    def build_trainable_surface_receipt(
        self,
        model: Any,
        adapter_result: Any,
        special_result: Any,
        optimizer_group_plan: Any,
    ) -> Any:
        from src.optim import build_trainable_surface_receipt

        return build_trainable_surface_receipt(
            model,
            adapter_receipt=adapter_result.receipt,
            special_token_receipt=special_result.receipt,
            optimizer_group_plan=optimizer_group_plan,
        )

    def build_runtime(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        accelerator: Any,
        plan: Human13LiveModelPlan,
        pack_count: int,
    ) -> Any:
        from src.config.models import RuntimeBatchResolution, RuntimeConfig
        from src.runtime import TrainRuntime

        return TrainRuntime(
            runtime_config=RuntimeConfig(seed=17),
            runtime_batch=RuntimeBatchResolution(
                world_size=1,
                effective_batch_size=pack_count,
                resolved_grad_accum_steps=pack_count,
            ),
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_mixed_precision=_accelerator_mixed_precision(plan),
            max_grad_norm=plan.max_grad_norm,
            accelerator=accelerator,
            rank_report_gatherer=None,
        )


def build_human13_live_model_plan(
    arm_config: str | Path | Any,
    *,
    global_learning_rate_decision: Any | None = None,
) -> Human13LiveModelPlan:
    """Project one strict updated arm into the immutable live assembly plan."""

    if isinstance(arm_config, (str, Path)):
        try:
            from scripts.research.materialize_human13_k_union_configs import (
                load_arm_config,
            )

            config = load_arm_config(arm_config)
        except Exception as exc:
            raise Human13LiveModelError(f"invalid Human-13 arm config: {exc}") from exc
    else:
        config = arm_config
    if getattr(config, "unit_id", None) not in {
        UNIT_ID,
        SUCCESSOR_UNIT_ID,
        ON_POLICY_UNIT_ID,
        RP_CROSSOVER_UNIT_ID,
    }:
        raise Human13LiveModelError("arm config is not bound to the Human-13 unit")
    if getattr(config, "updates", None) is not True:
        raise Human13LiveModelError("live training assembly requires an updated arm")
    if getattr(config, "arm_id", None) not in _UPDATED_ARM_IDS:
        raise Human13LiveModelError("live training assembly requires an approved arm")
    source = getattr(config, "source", None)
    optimizer = getattr(config, "optimizer", None)
    scheduler = getattr(config, "scheduler", None)
    surface = getattr(config, "trainable_surface", None)
    if source is None or optimizer is None or scheduler is None or surface is None:
        raise Human13LiveModelError("updated arm is missing its live assembly contract")
    if optimizer.name != "adamw_torch" or scheduler.name != "cosine_with_warmup":
        raise Human13LiveModelError(
            "updated arm optimizer or scheduler identity drifted"
        )
    max_grad_norm = getattr(config, "max_grad_norm", None)
    if max_grad_norm is None:
        raise Human13LiveModelError("updated arm is missing max_grad_norm")
    if (
        getattr(surface, "language_tower_dora", None),
        getattr(surface, "vision_tower", None),
        getattr(surface, "multimodal_aligner", None),
        getattr(surface, "token_embeddings", None),
        getattr(surface, "base_language_weights", None),
    ) != (True, False, False, False, False):
        raise Human13LiveModelError("updated arm must declare language-only DoRA")
    unit_id = str(config.unit_id)
    # Task 6.2 execution-surface correction: RP-crossover score-function
    # forwards run on the exact fp32/SDPA history surface; legacy units keep
    # their sealed BF16/FA2 surface.
    mixed_precision, attn_implementation = (
        ("fp32", "sdpa")
        if unit_id == RP_CROSSOVER_UNIT_ID
        else ("bf16", "flash_attention_2")
    )
    decision_sha256 = None
    selected_learning_rate = float(optimizer.learning_rate)
    resolution = "not_applicable"
    if unit_id == RP_CROSSOVER_UNIT_ID:
        resolution = "provisional_qualification"
        if global_learning_rate_decision is not None:
            from scripts.research.human13_rp_crossover_production import (
                GlobalLearningRateDecision,
            )

            if type(global_learning_rate_decision) is not GlobalLearningRateDecision:
                raise Human13LiveModelError(
                    "RP-crossover live plan requires a selector-produced global decision"
                )
            typed_decision: GlobalLearningRateDecision = global_learning_rate_decision
            selected_learning_rate = typed_decision.selected_learning_rate
            decision_sha256 = typed_decision.content_sha256
            resolution = "global_selected"
    elif global_learning_rate_decision is not None:
        raise Human13LiveModelError(
            "global learning-rate decisions apply only to the RP-crossover unit"
        )
    plan = Human13LiveModelPlan(
        schema_version=LIVE_PLAN_SCHEMA_VERSION,
        unit_id=unit_id,
        arm_id=str(config.arm_id),
        source=Human13SourceContract(
            checkpoint_path=str(source.checkpoint_path),
            base_model_path=str(source.base_model_path),
            adapter_path=str(source.adapter_path),
            special_embedding_path=str(source.special_embedding_path),
            adapter_sha256=str(source.adapter_sha256),
            special_embedding_sha256=str(source.special_embedding_sha256),
        ),
        mixed_precision=mixed_precision,
        attn_implementation=attn_implementation,
        patch_embed_linearization="enabled",
        adapter_seed_mode="warm_start_expand_dora",
        adapter_target_towers=("language",),
        adapter_target_modules="all_linear",
        adapter_rank=16,
        adapter_alpha=32,
        adapter_dropout=0.0,
        adapter_bias="none",
        freeze_special_token_delta=True,
        optimizer_name="adamw_torch",
        learning_rate=selected_learning_rate,
        betas=(float(optimizer.betas[0]), float(optimizer.betas[1])),
        epsilon=float(optimizer.epsilon),
        weight_decay=float(optimizer.weight_decay),
        scheduler_name="cosine_with_warmup",
        scheduler_warmup_steps=int(scheduler.warmup_steps),
        scheduler_horizon_updates=int(scheduler.horizon_updates),
        max_grad_norm=float(max_grad_norm),
        world_size=1,
        milestones=tuple(int(value) for value in config.milestones),
        learning_rate_resolution=resolution,
        global_learning_rate_decision_sha256=decision_sha256,
    )
    if resolution == "global_selected":
        plan = replace(plan, resolved_plan_sha256=_resolved_plan_sha256(plan))
    _require_frozen_plan(plan)
    return plan


def validate_human13_live_model_plan(
    plan: Human13LiveModelPlan,
) -> Human13PlanValidationReceipt:
    """Validate the frozen Source bytes without importing model/runtime code."""

    _require_frozen_plan(plan)
    adapter_root = _regular_directory(plan.source.adapter_path, "Source adapter")
    special_root = _regular_directory(
        plan.source.special_embedding_path,
        "Source special-token delta",
    )
    adapter_tensor = _regular_file(
        adapter_root / "adapter_model.safetensors",
        "Source adapter tensor",
    )
    special_tensor = _regular_file(
        special_root / "special_token_embeddings.safetensors",
        "Source special-token tensor",
    )
    adapter_hash = _sha256_file(adapter_tensor)
    if adapter_hash != plan.source.adapter_sha256:
        raise Human13LiveModelError("Source adapter tensor SHA-256 drifted")
    special_hash = _sha256_file(special_tensor)
    if special_hash != plan.source.special_embedding_sha256:
        raise Human13LiveModelError("Source special-token tensor SHA-256 drifted")

    adapter_config = _read_json_object(
        _regular_file(adapter_root / "adapter_config.json", "Source adapter config"),
        "Source adapter config",
    )
    _validate_adapter_config(adapter_config, plan)
    special_metadata = _read_json_object(
        _regular_file(
            special_root / "special_token_embeddings.json",
            "Source special-token metadata",
        ),
        "Source special-token metadata",
    )
    _validate_special_metadata(special_metadata, plan)
    return Human13PlanValidationReceipt(
        schema_version=VALIDATION_SCHEMA_VERSION,
        arm_id=plan.arm_id,
        adapter_tensor_sha256=adapter_hash,
        special_embedding_tensor_sha256=special_hash,
        base_config_sha256=SOURCE_BASE_CONFIG_SHA256,
        tokenizer_sha256=SOURCE_TOKENIZER_SHA256,
        model_actions=dict(ZERO_MODEL_ACTIONS),
    )


def assemble_human13_live_model(
    plan: Human13LiveModelPlan,
    *,
    pack_count: int,
    repo_root: str | Path,
    backend: Human13AssemblyBackend | None = None,
    _qualification_admitted: bool = False,
) -> Human13LiveAssembly:
    """Cross the sole live boundary and assemble one fresh world-size-one arm."""

    if (
        isinstance(pack_count, bool)
        or not isinstance(pack_count, int)
        or pack_count <= 0
    ):
        raise Human13LiveModelError("pack_count must be a positive integer")
    if (
        plan.unit_id == RP_CROSSOVER_UNIT_ID
        and plan.learning_rate_resolution != "global_selected"
        and not _qualification_admitted
    ):
        raise Human13LiveModelError(
            "RP-crossover matrix assembly requires the global learning-rate decision"
        )
    root = Path(repo_root).expanduser().resolve()
    validation = validate_human13_live_model_plan(plan)
    live_backend = backend or DefaultHuman13AssemblyBackend()

    accelerator = live_backend.create_accelerator(plan)
    live_backend.validate_accelerator(accelerator, plan)
    _require_world_size_one_cuda(accelerator)
    components = live_backend.load_qwen(plan)
    base_model = getattr(components, "model", None)
    if base_model is None:
        raise Human13LiveModelError("live Qwen assembly did not load a model")
    _require_loaded_component_identity(components, plan)

    adapter_result = live_backend.warm_start_language_dora(
        base_model,
        components,
        plan,
        repo_root=root,
    )
    special_result = live_backend.load_and_freeze_special_token_delta(
        adapter_result.model,
        components,
        plan,
        repo_root=root,
    )
    if getattr(special_result.shared_embed_delta, "requires_grad", None) is not False:
        raise Human13LiveModelError(
            "loaded Source special-token delta must remain frozen"
        )
    if (
        _loaded_adapter_tensor_sha256(adapter_result)
        != validation.adapter_tensor_sha256
        or validation.adapter_tensor_sha256 != plan.source.adapter_sha256
    ):
        raise Human13LiveModelError(
            "loaded adapter receipt differs from the validated Source identity"
        )
    if (
        _loaded_special_embedding_tensor_sha256(special_result)
        != validation.special_embedding_tensor_sha256
        or validation.special_embedding_tensor_sha256
        != plan.source.special_embedding_sha256
    ):
        raise Human13LiveModelError(
            "loaded special-token receipt differs from the validated Source identity"
        )
    model = special_result.model
    memory_saver_receipt = live_backend.enable_memory_savers(model)
    optimizer, scheduler, optimizer_group_plan = live_backend.build_optimizer(
        model,
        adapter_result,
        plan,
    )
    _require_fresh_exact_optimizer(optimizer, plan)
    surface_receipt = live_backend.build_trainable_surface_receipt(
        model,
        adapter_result,
        special_result,
        optimizer_group_plan,
    )
    _require_language_only_surface(surface_receipt)
    runtime = live_backend.build_runtime(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        accelerator=accelerator,
        plan=plan,
        pack_count=pack_count,
    )
    if int(getattr(runtime, "world_size", -1)) != 1:
        raise Human13LiveModelError("TrainRuntime must remain world size one")
    return _seal_human13_live_assembly(Human13LiveAssembly(
        plan=plan,
        validation=validation,
        components=components,
        accelerator=accelerator,
        model=getattr(runtime, "model", model),
        adapter_result=adapter_result,
        special_token_result=special_result,
        optimizer=getattr(runtime, "optimizer", optimizer),
        scheduler=getattr(runtime, "scheduler", scheduler),
        optimizer_group_plan=optimizer_group_plan,
        trainable_surface_receipt=surface_receipt,
        runtime=runtime,
        memory_saver_receipt=memory_saver_receipt,
    ))


def assemble_human13_qualification_model(
    plan: Human13LiveModelPlan,
    *,
    pack_count: int,
    repo_root: str | Path,
    backend: Human13AssemblyBackend | None = None,
) -> Human13LiveAssembly:
    """Admit one sealed C dose without masquerading as a selected matrix LR."""

    if (
        plan.unit_id != RP_CROSSOVER_UNIT_ID
        or plan.arm_id != "C"
        or plan.learning_rate_resolution != "provisional_qualification"
        or plan.learning_rate not in RP_CROSSOVER_LEARNING_RATE_RAY
    ):
        raise Human13LiveModelError(
            "qualification assembly requires one provisional sealed C dose"
        )
    return assemble_human13_live_model(
        plan,
        pack_count=pack_count,
        repo_root=repo_root,
        backend=backend,
        _qualification_admitted=True,
    )


def assemble_human13_parity_model(
    plan: Human13LiveModelPlan,
    *,
    repo_root: str | Path,
    backend: Human13ParityAssemblyBackend | None = None,
) -> Human13ParityAssembly:
    """Load one provisional C Source surface for inference-only parity replay."""

    if (
        plan.unit_id != RP_CROSSOVER_UNIT_ID
        or plan.arm_id != "C"
        or plan.learning_rate_resolution != "provisional_qualification"
        or plan.global_learning_rate_decision_sha256 is not None
        or plan.resolved_plan_sha256 is not None
        or plan.learning_rate not in RP_CROSSOVER_LEARNING_RATE_RAY
        or plan.mixed_precision != "fp32"
        or plan.attn_implementation != "sdpa"
    ):
        raise Human13LiveModelError(
            "parity assembly requires one provisional fp32/SDPA sealed C dose"
        )
    _require_frozen_plan(plan)
    root = Path(repo_root).expanduser().resolve()
    validation = validate_human13_live_model_plan(plan)
    live_backend = backend or DefaultHuman13AssemblyBackend()

    accelerator = live_backend.create_accelerator(plan)
    live_backend.validate_accelerator(accelerator, plan)
    _require_world_size_one_cuda(accelerator)
    components = live_backend.load_qwen(plan)
    base_model = getattr(components, "model", None)
    if base_model is None:
        raise Human13LiveModelError("parity Qwen assembly did not load a model")
    _require_loaded_component_identity(components, plan)
    adapter_result = live_backend.warm_start_language_dora(
        base_model,
        components,
        plan,
        repo_root=root,
    )
    special_result = live_backend.load_and_freeze_special_token_delta(
        adapter_result.model,
        components,
        plan,
        repo_root=root,
    )
    if getattr(special_result.shared_embed_delta, "requires_grad", None) is not False:
        raise Human13LiveModelError(
            "loaded Source special-token delta must remain frozen"
        )
    model = special_result.model
    model.requires_grad_(False)
    model.eval()
    model = accelerator.prepare_model(model, evaluation_mode=True)
    return Human13ParityAssembly(
        plan=plan,
        validation=validation,
        components=components,
        accelerator=accelerator,
        model=model,
        adapter_result=adapter_result,
        special_token_result=special_result,
        runtime=Human13ParityRuntime(accelerator=accelerator),
    )


def bind_human13_selected_rp_crossover_plan(
    plan: Human13LiveModelPlan, *, decision_sha256: str
) -> Human13LiveModelPlan:
    """Bind an already-admitted matrix cell to its selector receipt digest."""

    if (
        type(plan) is not Human13LiveModelPlan
        or plan.unit_id != RP_CROSSOVER_UNIT_ID
        or plan.arm_id not in {"A", "B", "C"}
        or plan.learning_rate_resolution != "provisional_qualification"
        or plan.global_learning_rate_decision_sha256 is not None
        or plan.resolved_plan_sha256 is not None
        or plan.learning_rate not in RP_CROSSOVER_LEARNING_RATE_RAY
        or not _is_sha256(decision_sha256)
    ):
        raise Human13LiveModelError(
            "matrix plan binding requires one provisional RP cell and decision digest"
        )
    _require_frozen_plan(plan)
    resolved = replace(
        plan,
        learning_rate_resolution="global_selected",
        global_learning_rate_decision_sha256=decision_sha256,
    )
    resolved = replace(
        resolved,
        resolved_plan_sha256=_resolved_plan_sha256(resolved),
    )
    _require_frozen_plan(resolved)
    return resolved


def build_human13_update_schedule(*, pack_count: int) -> Any:
    """Build the frozen sixteen-update 1/2/4/8/16 checkpoint schedule."""

    if (
        isinstance(pack_count, bool)
        or not isinstance(pack_count, int)
        or pack_count <= 0
    ):
        raise Human13LiveModelError("pack_count must be a positive integer")
    from src.config.models import RuntimeBatchResolution
    from src.training.schedule import ResolvedStepSchedule, StepScheduleEvent

    checkpoint_events = tuple(
        StepScheduleEvent(
            planned_step_id=step,
            event="checkpoint",
            trigger_reasons=("explicit_step",),
            source_config_path=None,
            deduped_from=(),
            required=step == 16,
        )
        for step in CHECKPOINT_STEPS
    )
    final_event = StepScheduleEvent(
        planned_step_id=16,
        event="final",
        trigger_reasons=("final",),
        source_config_path=None,
        deduped_from=(),
        required=True,
    )
    presentations = 16 * pack_count
    return ResolvedStepSchedule(
        resolved_max_steps=16,
        packs_per_epoch=pack_count,
        requested_pack_presentations=presentations,
        actual_pack_presentations=presentations,
        tail_fill_pack_count=0,
        runtime_batch=RuntimeBatchResolution(
            world_size=1,
            effective_batch_size=pack_count,
            resolved_grad_accum_steps=pack_count,
        ),
        events={
            "checkpoint": checkpoint_events,
            "eval.forward": (),
            "final": (final_event,),
        },
    )


def build_human13_processor_skeletons(
    manifest: Any,
    components: Any,
    *,
    repo_root: str | Path,
) -> dict[int, Any]:
    """Encode the canonical full-GT panel and attach exact reusable row tokens.

    This is a live processor boundary: it tokenizes and materializes image pixels,
    but performs no model forward.  Each returned encoded example carries the
    physical prompt length and one body-row token sequence per canonical owner.
    """

    _require_processor_manifest(manifest, components)
    root = Path(repo_root).expanduser().resolve()
    from src.config.inference import load_infer_config
    from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
    from src.data import load_raw_examples
    from src.qwen import encode_rendered_example
    from src.templates import render_example
    from scripts.research.collect_human13_discovery import (
        _prompt_policy_fingerprint,
    )

    resolved = load_infer_config(root / HUMAN13_SOURCE_INFER_CONFIG)
    infer_config = resolved.config
    if (
        str(Path(infer_config.data.input_jsonl).resolve()) != HUMAN13_PANEL_PATH
        or str(Path(infer_config.model.base_model).resolve()) != SOURCE_BASE_MODEL_PATH
    ):
        raise Human13LiveModelError("Source inference config identity drifted")
    template = TemplateConfig(
        object_field_order=infer_config.template.object_field_order,
        object_ordering=infer_config.template.object_ordering,
        assistant_format=infer_config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=infer_config.template.prompt.system,
            user=infer_config.template.prompt.user,
        ),
    )
    if (
        template.object_field_order,
        template.object_ordering,
        template.assistant_format,
    ) != ("desc_first", "geo_sorted_xy", "object_box_closed"):
        raise Human13LiveModelError("Source prompt template structure drifted")
    if _prompt_policy_fingerprint(resolved) != HUMAN13_PROMPT_POLICY_FINGERPRINT:
        raise Human13LiveModelError("Source prompt-policy fingerprint drifted")
    processor_config = ProcessorConfig(
        do_resize=False,
        max_raw_pixels=1_000_000_000,
        max_merged_visual_tokens=1_000_000,
    )
    raw_by_image: dict[int, Any] = {}
    for raw in load_raw_examples(HUMAN13_PANEL_PATH):
        image_id = _raw_example_image_id(raw)
        if image_id in raw_by_image:
            raise Human13LiveModelError("canonical panel repeats an image identity")
        raw_by_image[image_id] = raw
    if tuple(raw_by_image) != HUMAN13_IMAGE_IDS:
        raise Human13LiveModelError("canonical panel image order drifted")

    skeletons: dict[int, Any] = {}
    for image in manifest.images:
        image_id = int(image.image_id)
        raw = raw_by_image.get(image_id)
        if raw is None:
            raise Human13LiveModelError(
                f"canonical panel lacks manifest image {image_id}"
            )
        rendered = render_example(raw, template)
        encoded = encode_rendered_example(
            raw,
            rendered,
            components=components,
            processor_config=processor_config,
            global_max_length=12_000,
            materialize_image_pixels=True,
        )
        prompt_count, owner_row_tokens = _derive_owner_row_tokens(
            image,
            raw,
            encoded,
        )
        object.__setattr__(encoded, "prompt_token_count", prompt_count)
        object.__setattr__(encoded, "owner_row_tokens", owner_row_tokens)
        skeletons[image_id] = encoded
    if (
        tuple(skeletons) != HUMAN13_IMAGE_IDS
        or sum(len(item.owner_row_tokens) for item in skeletons.values()) != 392
    ):
        raise Human13LiveModelError(
            "processor skeletons must cover exactly 13 images and 392 owners"
        )
    return skeletons


def build_human13_parity_skeleton(
    *,
    image_id: int,
    components: Any,
    repo_root: str | Path,
) -> Any:
    """Encode only image 1584 for exact-history inference replay."""

    if image_id != 1584:
        raise Human13LiveModelError("parity replay is reserved for image 1584")
    root = Path(repo_root).expanduser().resolve()
    from src.config.inference import load_infer_config
    from src.config.models import ProcessorConfig, TemplateConfig, TemplatePromptConfig
    from src.data import load_raw_examples
    from src.qwen import encode_rendered_example
    from src.templates import render_example
    from scripts.research.collect_human13_discovery import (
        _prompt_policy_fingerprint,
    )

    resolved = load_infer_config(root / HUMAN13_SOURCE_INFER_CONFIG)
    infer_config = resolved.config
    if (
        infer_config.backend.type != "hf"
        or infer_config.model.dtype != "fp32"
        or infer_config.backend.hf.attn_implementation != "sdpa"
        or str(Path(infer_config.data.input_jsonl).resolve()) != HUMAN13_PANEL_PATH
        or str(Path(infer_config.model.base_model).resolve())
        != SOURCE_BASE_MODEL_PATH
        or infer_config.adapter is None
        or str(Path(infer_config.adapter.path).resolve()) != SOURCE_ADAPTER_PATH
        or infer_config.embedding_delta is None
        or str(Path(infer_config.embedding_delta.path).resolve())
        != SOURCE_SPECIAL_EMBEDDING_PATH
    ):
        raise Human13LiveModelError("Source parity inference config identity drifted")
    template = TemplateConfig(
        object_field_order=infer_config.template.object_field_order,
        object_ordering=infer_config.template.object_ordering,
        assistant_format=infer_config.template.assistant_format,
        prompt=TemplatePromptConfig(
            system=infer_config.template.prompt.system,
            user=infer_config.template.prompt.user,
        ),
    )
    if (
        template.object_field_order,
        template.object_ordering,
        template.assistant_format,
    ) != ("desc_first", "geo_sorted_xy", "object_box_closed"):
        raise Human13LiveModelError("Source prompt template structure drifted")
    if _prompt_policy_fingerprint(resolved) != HUMAN13_PROMPT_POLICY_FINGERPRINT:
        raise Human13LiveModelError("Source prompt-policy fingerprint drifted")

    raw = next(
        (
            item
            for item in load_raw_examples(HUMAN13_PANEL_PATH)
            if _raw_example_image_id(item) == image_id
        ),
        None,
    )
    if raw is None:
        raise Human13LiveModelError("canonical panel lacks image 1584")
    rendered = render_example(raw, template)
    encoded = encode_rendered_example(
        raw,
        rendered,
        components=components,
        processor_config=ProcessorConfig(
            do_resize=False,
            max_raw_pixels=1_000_000_000,
            max_merged_visual_tokens=1_000_000,
        ),
        global_max_length=12_000,
        materialize_image_pixels=True,
    )
    starts = tuple(
        int(span.physical_token_start)
        for span in getattr(encoded, "supervised_token_spans", ())
    )
    if not starts or min(starts) <= 0 or min(starts) >= len(encoded.input_ids):
        raise Human13LiveModelError(
            "image 1584 exact-history prompt boundary is absent"
        )
    object.__setattr__(encoded, "prompt_token_count", min(starts))
    return encoded


def build_human13_checkpoint_kwargs(
    assembly: Human13LiveAssembly,
) -> dict[str, Any]:
    """Return the stable payload arguments consumed by ``CheckpointWriter``."""

    adapter_name = getattr(assembly.adapter_result.receipt, "adapter_name", None)
    if not isinstance(adapter_name, str) or not adapter_name:
        raise Human13LiveModelError("assembled DoRA receipt has no adapter name")
    return {
        "accelerator": assembly.accelerator,
        "adapter_name": adapter_name,
        "special_token_result": assembly.special_token_result,
        "base_model_path": Path(assembly.components.base_model_path),
        "base_config_sha256": str(assembly.components.base_config_sha256),
        "tokenizer_sha256": str(assembly.components.tokenizer_sha256),
    }


def build_human13_checkpoint_writer(run_dir: str | Path) -> Any:
    """Construct the accepted atomic writer without writing a checkpoint."""

    root = Path(run_dir).expanduser()
    if not root.is_absolute():
        raise Human13LiveModelError("checkpoint run_dir must be absolute")
    from src.artifacts import CheckpointWriter

    writer = cast(Callable[..., Any], CheckpointWriter)
    return writer(run_dir=root.resolve())


def readback_human13_checkpoint(
    checkpoint_dir: str | Path,
    *,
    expected_step: int,
    assembly: Human13LiveAssembly,
) -> Human13CheckpointReadback:
    """CPU-validate one written adapter+delta checkpoint against the assembly."""

    allowed_steps = (
        frozenset(range(1, 9))
        if assembly.plan.unit_id == ON_POLICY_UNIT_ID
        else frozenset(CHECKPOINT_STEPS)
    )
    if (
        isinstance(expected_step, bool)
        or not isinstance(expected_step, int)
        or expected_step not in allowed_steps
    ):
        raise Human13LiveModelError(
            "checkpoint readback step is outside the owning unit schedule"
        )
    root = _regular_directory(checkpoint_dir, "Human-13 checkpoint")
    if root.name != f"step-{expected_step}" or root.parent.name != "checkpoints":
        raise Human13LiveModelError("checkpoint directory does not match its step")
    children = {item.name for item in root.iterdir()}
    if children != {"adapter", "special_token_embeddings"}:
        raise Human13LiveModelError(
            "Human-13 checkpoint must contain only adapter and frozen delta payloads"
        )
    _require_language_only_surface(assembly.trainable_surface_receipt)
    from src.adapters.dora import inspect_dora_adapter_payload
    from src.qwen.special_token_embeddings import (
        inspect_special_token_embedding_delta_payload,
    )

    base_path = Path(assembly.components.base_model_path)
    adapter_identity = inspect_dora_adapter_payload(
        root / "adapter",
        expected_base_model_path=base_path,
    )
    special_identity = inspect_special_token_embedding_delta_payload(
        root / "special_token_embeddings",
        expected_base_model_path=base_path,
        expected_base_config_sha256=assembly.components.base_config_sha256,
        expected_tokenizer_sha256=assembly.components.tokenizer_sha256,
    )
    _require_checkpoint_payload_identity(adapter_identity, special_identity)
    return Human13CheckpointReadback(
        step=expected_step,
        checkpoint_dir=root,
        adapter_fingerprint=str(adapter_identity["fingerprint"]),
        special_embedding_fingerprint=str(special_identity["fingerprint"]),
        adapter_identity=adapter_identity,
        special_embedding_identity=special_identity,
    )


def _require_processor_manifest(manifest: Any, components: Any) -> None:
    binding = getattr(manifest, "binding", None)
    panel = getattr(binding, "panel", None)
    source = getattr(binding, "source", None)
    surface = getattr(binding, "surface", None)
    if (
        getattr(manifest, "full_panel", None) is not True
        or getattr(binding, "unit_id", None) != UNIT_ID
        or getattr(binding, "purpose", None) != "overfit_only"
        or getattr(panel, "panel_sha256", None) != HUMAN13_PANEL_SHA256
        or int(getattr(panel, "owner_count", -1)) != 392
        or getattr(source, "checkpoint_path", None) != SOURCE_CHECKPOINT_PATH
        or getattr(source, "base_model_path", None) != SOURCE_BASE_MODEL_PATH
        or getattr(surface, "prompt_policy_fingerprint", None)
        != HUMAN13_PROMPT_POLICY_FINGERPRINT
        or getattr(surface, "tokenizer_sha256", None) != SOURCE_TOKENIZER_SHA256
        or getattr(components, "tokenizer_sha256", None) != SOURCE_TOKENIZER_SHA256
        or tuple(int(image.image_id) for image in getattr(manifest, "images", ()))
        != HUMAN13_IMAGE_IDS
    ):
        raise Human13LiveModelError(
            "processor skeletons require the canonical full-panel identity"
        )


def _raw_example_image_id(raw: Any) -> int:
    metadata = getattr(raw, "metadata", None)
    source = metadata.get("source") if isinstance(metadata, Mapping) else None
    value = source.get("image_id") if isinstance(source, Mapping) else None
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise Human13LiveModelError("panel row lacks a canonical image identity")
    return value


def _derive_owner_row_tokens(
    image: Any,
    raw: Any,
    encoded: Any,
) -> tuple[int, dict[str, tuple[int, ...]]]:
    owners = tuple(sorted(image.owners, key=lambda item: item.source_object_index))
    objects = tuple(raw.objects)
    if len(owners) != len(objects) or tuple(
        int(owner.source_object_index) for owner in owners
    ) != tuple(range(len(objects))):
        raise Human13LiveModelError(
            f"manifest owner ordering drifted for image {image.image_id}"
        )
    spans_by_object: dict[str, list[Any]] = {obj.object_id: [] for obj in objects}
    for span in getattr(encoded, "supervised_token_spans", ()):
        object_id = getattr(span, "object_id", None)
        if object_id in spans_by_object:
            spans_by_object[object_id].append(span)
    all_starts = [
        int(span.physical_token_start)
        for spans in spans_by_object.values()
        for span in spans
    ]
    if not all_starts:
        raise Human13LiveModelError(
            f"encoded full-GT row spans are absent for image {image.image_id}"
        )
    prompt_count = min(all_starts)
    cursor = prompt_count
    owner_row_tokens: dict[str, tuple[int, ...]] = {}
    for owner, obj in zip(owners, objects, strict=True):
        if owner.category != obj.description:
            raise Human13LiveModelError(
                f"manifest owner category drifted for image {image.image_id}"
            )
        spans = tuple(
            sorted(
                spans_by_object[obj.object_id],
                key=lambda span: int(span.physical_token_start),
            )
        )
        if not spans or int(spans[0].physical_token_start) != cursor:
            raise Human13LiveModelError(
                f"encoded owner row is missing or noncontiguous for {owner.owner_id}"
            )
        tokens: list[int] = []
        for span in spans:
            start = int(span.physical_token_start)
            end = int(span.physical_token_end)
            span_tokens = tuple(int(token) for token in span.token_ids)
            if start != cursor or end <= start or len(span_tokens) != end - start:
                raise Human13LiveModelError(
                    f"encoded owner row span drifted for {owner.owner_id}"
                )
            if tuple(encoded.input_ids[start:end]) != span_tokens:
                raise Human13LiveModelError(
                    f"encoded owner row tokens drifted for {owner.owner_id}"
                )
            tokens.extend(span_tokens)
            cursor = end
        owner_row_tokens[str(owner.owner_id)] = tuple(tokens)
    concatenated = tuple(
        token for owner in owners for token in owner_row_tokens[str(owner.owner_id)]
    )
    if tuple(encoded.input_ids[prompt_count:cursor]) != concatenated:
        raise Human13LiveModelError(
            f"encoded full-GT body is noncanonical for image {image.image_id}"
        )
    return prompt_count, owner_row_tokens


def _require_checkpoint_payload_identity(
    adapter: Mapping[str, Any],
    special: Mapping[str, Any],
) -> None:
    adapter_semantics = adapter.get("semantic_identity")
    special_semantics = special.get("semantic_identity")
    adapter_fingerprint = adapter.get("fingerprint")
    special_fingerprint = special.get("fingerprint")
    if (
        adapter.get("kind") != "dora_adapter"
        or not isinstance(adapter_semantics, Mapping)
        or adapter_semantics.get("peft_type") != "LORA"
        or adapter_semantics.get("use_dora") is not True
        or adapter_semantics.get("base_model_name_or_path") != SOURCE_BASE_MODEL_PATH
        or frozenset(adapter_semantics.get("target_modules", ()))
        != _SOURCE_ADAPTER_TARGET_MODULES
        or int(adapter_semantics.get("r", -1)) != 16
        or float(adapter_semantics.get("lora_alpha", -1.0)) != 32.0
        or int(adapter_semantics.get("lora_A_count", 0)) <= 0
        or adapter_semantics.get("lora_A_count")
        != adapter_semantics.get("lora_B_count")
        or adapter_semantics.get("lora_A_count")
        != adapter_semantics.get("lora_magnitude_vector_count")
        or not _is_sha256(adapter_fingerprint)
    ):
        raise Human13LiveModelError("checkpoint DoRA payload identity drifted")
    if (
        special.get("kind") != "special_token_embedding_delta"
        or not isinstance(special_semantics, Mapping)
        or special_semantics.get("semantics") != "additive_delta"
        or special_semantics.get("tensor_key") != "shared_embed_delta"
        or special_semantics.get("tensor_shape") != [1004, 2048]
        or special_semantics.get("tensor_dtype") != "float32"
        or len(special_semantics.get("token_ids", ())) != 1004
        or special_semantics.get("base_model_path") != SOURCE_BASE_MODEL_PATH
        or special_semantics.get("base_config_sha256") != SOURCE_BASE_CONFIG_SHA256
        or special_semantics.get("tokenizer_sha256") != SOURCE_TOKENIZER_SHA256
        or special_semantics.get("tie_word_embeddings") is not True
        or not _is_sha256(special_fingerprint)
    ):
        raise Human13LiveModelError(
            "checkpoint special-token delta payload identity drifted"
        )


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_frozen_plan(plan: Human13LiveModelPlan) -> None:
    if plan.unit_id not in {
        UNIT_ID,
        SUCCESSOR_UNIT_ID,
        ON_POLICY_UNIT_ID,
        RP_CROSSOVER_UNIT_ID,
    }:
        raise Human13LiveModelError("plan has an unknown Human-13 unit identity")
    if (plan.unit_id == SUCCESSOR_UNIT_ID) != (plan.arm_id in {"R1", "R2"}):
        raise Human13LiveModelError("successor unit and R1/R2 arm identity differ")
    if (plan.unit_id == ON_POLICY_UNIT_ID) != (
        plan.arm_id in {"O-Full-Safe", "O-First-Safe"}
    ):
        raise Human13LiveModelError("on-policy unit and O-arm identity differ")
    if (plan.unit_id == RP_CROSSOVER_UNIT_ID) != (plan.arm_id in {"A", "B", "C"}):
        raise Human13LiveModelError("RP-crossover unit and A/B/C arm identity differ")
    if (
        plan.unit_id == RP_CROSSOVER_UNIT_ID
        and plan.learning_rate not in RP_CROSSOVER_LEARNING_RATE_RAY
    ):
        raise Human13LiveModelError(
            "RP-crossover learning_rate is outside the sealed qualification "
            "learning-rate ray"
        )
    if plan.unit_id == RP_CROSSOVER_UNIT_ID:
        if plan.learning_rate_resolution == "provisional_qualification":
            if (
                plan.global_learning_rate_decision_sha256 is not None
                or plan.resolved_plan_sha256 is not None
            ):
                raise Human13LiveModelError(
                    "provisional RP-crossover plan must not claim a global decision"
                )
        elif plan.learning_rate_resolution == "global_selected":
            if not _is_sha256(
                plan.global_learning_rate_decision_sha256
            ) or not _is_sha256(plan.resolved_plan_sha256):
                raise Human13LiveModelError(
                    "resolved RP-crossover plan must bind decision and plan hashes"
                )
            expected_resolved = _resolved_plan_sha256(plan)
            if plan.resolved_plan_sha256 != expected_resolved:
                raise Human13LiveModelError(
                    "resolved RP-crossover plan hash differs from selected LR"
                )
        else:
            raise Human13LiveModelError(
                "RP-crossover plan learning-rate resolution state differs"
            )
    elif (
        plan.learning_rate_resolution != "not_applicable"
        or plan.global_learning_rate_decision_sha256 is not None
        or plan.resolved_plan_sha256 is not None
    ):
        raise Human13LiveModelError(
            "non-crossover live plans must not bind an RP learning-rate decision"
        )
    expected = {
        "schema_version": LIVE_PLAN_SCHEMA_VERSION,
        "unit_id": plan.unit_id,
        "source": Human13SourceContract(
            checkpoint_path=SOURCE_CHECKPOINT_PATH,
            base_model_path=SOURCE_BASE_MODEL_PATH,
            adapter_path=SOURCE_ADAPTER_PATH,
            special_embedding_path=SOURCE_SPECIAL_EMBEDDING_PATH,
            adapter_sha256=SOURCE_ADAPTER_SHA256,
            special_embedding_sha256=SOURCE_SPECIAL_EMBEDDING_SHA256,
        ),
        "mixed_precision": (
            "fp32" if plan.unit_id == RP_CROSSOVER_UNIT_ID else "bf16"
        ),
        "attn_implementation": (
            "sdpa" if plan.unit_id == RP_CROSSOVER_UNIT_ID else "flash_attention_2"
        ),
        "patch_embed_linearization": "enabled",
        "adapter_seed_mode": "warm_start_expand_dora",
        "adapter_target_towers": ("language",),
        "adapter_target_modules": "all_linear",
        "adapter_rank": 16,
        "adapter_alpha": 32,
        "adapter_dropout": 0.0,
        "adapter_bias": "none",
        "freeze_special_token_delta": True,
        "optimizer_name": "adamw_torch",
        "learning_rate": (
            plan.learning_rate if plan.unit_id == RP_CROSSOVER_UNIT_ID else 1.0e-5
        ),
        "betas": (0.9, 0.999),
        "epsilon": 1.0e-8,
        "weight_decay": 0.0,
        "scheduler_name": "cosine_with_warmup",
        "scheduler_warmup_steps": 0,
        "scheduler_horizon_updates": 16,
        "max_grad_norm": 1.0,
        "world_size": 1,
        "milestones": (
            SUCCESSOR_MILESTONES
            if plan.unit_id == SUCCESSOR_UNIT_ID
            else ON_POLICY_MILESTONES
            if plan.unit_id == ON_POLICY_UNIT_ID
            else RP_CROSSOVER_MILESTONES
            if plan.unit_id == RP_CROSSOVER_UNIT_ID
            else MILESTONES
        ),
        "learning_rate_resolution": plan.learning_rate_resolution,
        "global_learning_rate_decision_sha256": (
            plan.global_learning_rate_decision_sha256
        ),
        "resolved_plan_sha256": plan.resolved_plan_sha256,
    }
    drift = [name for name, value in expected.items() if getattr(plan, name) != value]
    if drift:
        raise Human13LiveModelError(
            f"plan differs from the frozen live-model contract: {sorted(drift)}"
        )


def _require_world_size_one_cuda(accelerator: Any) -> None:
    if (
        int(getattr(accelerator, "num_processes", -1)) != 1
        or int(getattr(accelerator, "process_index", -1)) != 0
    ):
        raise Human13LiveModelError("Human-13 live assembly requires world size one")
    if not str(getattr(accelerator, "device", "")).startswith("cuda"):
        raise Human13LiveModelError("Human-13 live assembly requires one CUDA device")


def _require_loaded_component_identity(
    components: Any,
    plan: Human13LiveModelPlan,
) -> None:
    observed_path = str(Path(components.base_model_path).expanduser().resolve())
    if observed_path != plan.source.base_model_path:
        raise Human13LiveModelError("loaded Qwen base-model identity drifted")
    if components.base_config_sha256 != SOURCE_BASE_CONFIG_SHA256:
        raise Human13LiveModelError("loaded Qwen base-config identity drifted")
    if components.tokenizer_sha256 != SOURCE_TOKENIZER_SHA256:
        raise Human13LiveModelError("loaded Qwen tokenizer identity drifted")


def _require_fresh_exact_optimizer(
    optimizer: Any,
    plan: Human13LiveModelPlan,
) -> None:
    if len(getattr(optimizer, "state", {})) != 0:
        raise Human13LiveModelError("Human-13 arm requires a fresh AdamW state")
    groups = list(getattr(optimizer, "param_groups", ()))
    if len(groups) != 1:
        raise Human13LiveModelError("optimizer must contain only adapter.language")
    group = groups[0]
    if (
        group.get("name") != "adapter.language"
        or float(group.get("lr", -1.0)) != plan.learning_rate
        or float(group.get("weight_decay", -1.0)) != plan.weight_decay
    ):
        raise Human13LiveModelError("optimizer group differs from adapter.language")
    defaults = getattr(optimizer, "defaults", {})
    if (
        tuple(defaults.get("betas", ())) != plan.betas
        or float(defaults.get("eps", -1.0)) != plan.epsilon
    ):
        raise Human13LiveModelError("AdamW betas or epsilon drifted")


def _require_language_only_surface(receipt: Any) -> None:
    try:
        artifact = receipt.to_artifact_dict()
    except AttributeError as exc:
        raise Human13LiveModelError("trainable-surface receipt is unavailable") from exc
    optimizer_groups = artifact.get("optimizer_groups")
    group_names = (
        tuple(item.get("group_name") for item in optimizer_groups)
        if isinstance(optimizer_groups, list)
        else ()
    )
    exact = artifact.get("exact_surface_groups")
    language = (
        exact.get("trainable_language_dora") if isinstance(exact, Mapping) else None
    )
    frozen_delta = (
        exact.get("frozen_selected_token_delta") if isinstance(exact, Mapping) else None
    )
    frozen_vision = exact.get("frozen_vision") if isinstance(exact, Mapping) else None
    frozen_aligner = exact.get("frozen_aligner") if isinstance(exact, Mapping) else None
    if (
        artifact.get("phase") != "before_first_backward"
        or tuple(artifact.get("frozen_towers", ())) != ("language", "vision", "aligner")
        or group_names != ("adapter.language",)
        or tuple(artifact.get("trainable_towers", ())) != ("adapter.language",)
        or artifact.get("unmatched_trainable_names") != []
        or not isinstance(language, Mapping)
        or int(language.get("parameter_count", 0)) <= 0
        or not isinstance(frozen_delta, Mapping)
        or int(frozen_delta.get("parameter_count", 0)) <= 0
        or not isinstance(frozen_vision, Mapping)
        or int(frozen_vision.get("parameter_count", 0)) <= 0
        or not isinstance(frozen_aligner, Mapping)
        or int(frozen_aligner.get("parameter_count", 0)) <= 0
    ):
        raise Human13LiveModelError(
            "trainable-surface receipt must contain only adapter.language with a frozen Source delta"
        )


def _validate_adapter_config(
    config: Mapping[str, Any],
    plan: Human13LiveModelPlan,
) -> None:
    targets = config.get("target_modules")
    if (
        config.get("peft_type") != "LORA"
        or config.get("use_dora") is not True
        or config.get("base_model_name_or_path") != plan.source.base_model_path
        or config.get("r") != plan.adapter_rank
        or float(config.get("lora_alpha", -1.0)) != plan.adapter_alpha
        or float(config.get("lora_dropout", -1.0)) != plan.adapter_dropout
        or config.get("bias") != plan.adapter_bias
        or not isinstance(targets, list)
        or frozenset(targets) != _SOURCE_ADAPTER_TARGET_MODULES
    ):
        raise Human13LiveModelError("Source adapter metadata drifted")


def _validate_special_metadata(
    metadata: Mapping[str, Any],
    plan: Human13LiveModelPlan,
) -> None:
    token_ids = metadata.get("token_ids")
    token_strings = metadata.get("token_strings")
    if (
        metadata.get("base_model_path") != plan.source.base_model_path
        or metadata.get("base_config_sha256") != SOURCE_BASE_CONFIG_SHA256
        or metadata.get("tokenizer_sha256") != SOURCE_TOKENIZER_SHA256
        or metadata.get("semantics") != "additive_delta"
        or metadata.get("tensor_dtype") != "float32"
        or metadata.get("tensor_key") != "shared_embed_delta"
        or metadata.get("tensor_shape") != [1004, 2048]
        or metadata.get("tie_word_embeddings") is not True
        or not isinstance(token_ids, list)
        or not isinstance(token_strings, list)
        or len(token_ids) != 1004
        or len(token_strings) != 1004
    ):
        raise Human13LiveModelError("Source special-token metadata drifted")


def _regular_directory(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_dir():
        raise Human13LiveModelError(f"{label} is not a regular directory: {path}")
    return path.resolve(strict=True)


def _regular_file(value: str | Path, label: str) -> Path:
    path = Path(value).expanduser()
    if path.is_symlink() or not path.is_file():
        raise Human13LiveModelError(f"{label} is not a regular file: {path}")
    return path.resolve(strict=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise Human13LiveModelError(f"cannot load {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise Human13LiveModelError(f"{label} must be a JSON object")
    return value


__all__ = [
    "CHECKPOINT_STEPS",
    "DefaultHuman13AssemblyBackend",
    "Human13AssemblyBackend",
    "Human13CheckpointReadback",
    "Human13LiveAssembly",
    "Human13LoadedSpecialTokenResult",
    "Human13LiveModelError",
    "Human13LiveModelPlan",
    "Human13ParityAssembly",
    "Human13ParityAssemblyBackend",
    "Human13ParityRuntime",
    "Human13PlanValidationReceipt",
    "Human13SourceContract",
    "HUMAN13_IMAGE_IDS",
    "HUMAN13_PANEL_PATH",
    "HUMAN13_PANEL_SHA256",
    "HUMAN13_PROMPT_POLICY_FINGERPRINT",
    "MILESTONES",
    "ON_POLICY_MILESTONES",
    "ON_POLICY_UNIT_ID",
    "RP_CROSSOVER_MILESTONES",
    "RP_CROSSOVER_LEARNING_RATE_RAY",
    "RP_CROSSOVER_UNIT_ID",
    "SOURCE_ADAPTER_SHA256",
    "SOURCE_BASE_CONFIG_SHA256",
    "SOURCE_BASE_MODEL_PATH",
    "SOURCE_SPECIAL_EMBEDDING_SHA256",
    "SOURCE_TOKENIZER_SHA256",
    "SUCCESSOR_MILESTONES",
    "SUCCESSOR_UNIT_ID",
    "ZERO_MODEL_ACTIONS",
    "assemble_human13_live_model",
    "assemble_human13_parity_model",
    "assemble_human13_qualification_model",
    "bind_human13_selected_rp_crossover_plan",
    "build_human13_checkpoint_kwargs",
    "build_human13_checkpoint_writer",
    "build_human13_live_model_plan",
    "build_human13_parity_skeleton",
    "build_human13_processor_skeletons",
    "build_human13_update_schedule",
    "readback_human13_checkpoint",
    "require_admitted_human13_live_assembly",
    "validate_human13_live_model_plan",
    "validate_human13_live_assembly_values",
]
