from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from scripts.research import human13_live_model as live
from scripts.research.human13_rp_crossover_matrix_contracts import (
    AcquisitionKey,
    AggregateResourceReceipt,
    CellKey,
    DoseMechanicalReceipt,
    PHASE_QUALIFICATION,
)
from scripts.research.human13_rp_crossover_production import (
    QUALIFICATION_LEARNING_RATE_RAY,
    select_global_learning_rate,
)


CONFIG_ROOT = Path("configs/coordexp_swift/research/human13_k_union")


def _plan() -> live.Human13LiveModelPlan:
    return live.build_human13_live_model_plan(CONFIG_ROOT / "03_a1.yaml")


def _selected_lr_decision(selected: float):
    resources = AggregateResourceReceipt(
        measurement_scope="live",
        wall_time_seconds=1.0,
        peak_host_rss_bytes=1,
        cuda_peak_allocated_bytes=None,
        cuda_peak_reserved_bytes=None,
        acquisition_request_count=1,
        acquisition_batch_count=1,
        acquisition_token_count=1,
        decode_request_count=2,
        decode_batch_count=2,
        decode_token_count=2,
        packed_token_count=1,
        logical_token_count=1,
        forward_count=1,
        backward_count=1,
        row_bytes=1,
        artifact_bytes=1,
        update_count=1,
        audit_count=2,
        rollback_count=1,
    )
    receipts = []
    for rp in (1.0, 1.10):
        acquisition = AcquisitionKey(rp, "qualification", PHASE_QUALIFICATION)
        for dose in QUALIFICATION_LEARNING_RATE_RAY:
            checkpoint = hashlib.sha256(f"checkpoint:{rp}:{dose}".encode()).hexdigest()
            receipts.append(
                DoseMechanicalReceipt(
                    cell_key=CellKey(acquisition, "C", dose),
                    proposal_sha256=hashlib.sha256(
                        f"proposal:{rp}:{dose}".encode()
                    ).hexdigest(),
                    private_checkpoint_sha256=checkpoint,
                    audit_checkpoint_sha256s=(checkpoint, checkpoint),
                    greedy_decision_change_count=1,
                    malformed_output_delta_count=int(dose > selected),
                    cap_terminated_output_delta_count=0,
                    unparseable_output_delta_count=0,
                    active_witness_count=1,
                    jvp_fd_max_abs_error=0.0,
                    jvp_fd_tolerance=1.0e-6,
                    median_abs_decision_margin_displacement=0.1,
                    median_abs_source_decision_margin=1.0,
                    rollback_reproduced=True,
                    resources=resources,
                )
            )
    return select_global_learning_rate(receipts)


def test_successor_r1_r2_reuse_the_exact_low_dose_live_surface() -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    for arm_id in ("R1", "R2"):
        successor = replace(
            base,
            unit_id=live.SUCCESSOR_UNIT_ID,
            arm_id=arm_id,
            milestones=live.SUCCESSOR_MILESTONES,
        )
        plan = live.build_human13_live_model_plan(successor)
        assert plan.unit_id == live.SUCCESSOR_UNIT_ID
        assert plan.arm_id == arm_id
        assert plan.milestones == (0, 1, 2)
        live.validate_human13_live_model_plan(plan)


def test_all_hf_vertical_has_its_own_one_update_source_plan() -> None:
    plan = live.build_human13_all_hf_vertical_source_plan()

    assert plan.unit_id == (
        "2026-08-15-human13-all-hf-shared-surface-trajectory-credit-vertical"
    )
    assert plan.arm_id == "C-One-Image"
    assert plan.learning_rate == 3.0e-6
    assert plan.scheduler_horizon_updates == 1
    assert plan.milestones == (0, 1)
    assert plan.mixed_precision == "bf16"
    assert plan.attn_implementation == "flash_attention_2"
    assert plan.adapter_target_towers == ("language",)
    assert plan.freeze_special_token_delta is True
    live.validate_human13_live_model_plan(plan)


def test_on_policy_arms_reuse_surface_with_eight_attempt_milestones() -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    for arm_id in ("O-Full-Safe", "O-First-Safe"):
        on_policy = replace(
            base,
            unit_id=live.ON_POLICY_UNIT_ID,
            arm_id=arm_id,
            milestones=live.ON_POLICY_MILESTONES,
        )
        plan = live.build_human13_live_model_plan(on_policy)
        assert plan.unit_id == live.ON_POLICY_UNIT_ID
        assert plan.arm_id == arm_id
        assert plan.milestones == tuple(range(9))
        live.validate_human13_live_model_plan(plan)


def test_rp_crossover_arms_accept_only_the_sealed_qualification_dose_ray() -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    assert base.optimizer is not None
    assert live.RP_CROSSOVER_LEARNING_RATE_RAY == (
        3.0e-7,
        1.0e-6,
        3.0e-6,
        1.0e-5,
        3.0e-5,
    )
    for learning_rate in live.RP_CROSSOVER_LEARNING_RATE_RAY:
        for arm_id in ("A", "B", "C"):
            crossover = replace(
                base,
                unit_id=live.RP_CROSSOVER_UNIT_ID,
                arm_id=arm_id,
                optimizer=replace(base.optimizer, learning_rate=learning_rate),
                milestones=live.RP_CROSSOVER_MILESTONES,
            )
            plan = live.build_human13_live_model_plan(crossover)
            assert plan.unit_id == live.RP_CROSSOVER_UNIT_ID
            assert plan.arm_id == arm_id
            assert plan.learning_rate == learning_rate
            assert plan.milestones == (0, 1)
            # task 6.2 execution-surface correction: score-function forwards
            # run on the exact fp32/SDPA history surface for this unit only
            assert (plan.mixed_precision, plan.attn_implementation) == (
                "fp32",
                "sdpa",
            )
            assert plan.source.checkpoint_path == live.SOURCE_CHECKPOINT_PATH
            assert plan.adapter_target_towers == ("language",)
            assert plan.world_size == 1
            live.validate_human13_live_model_plan(plan)


def test_rp_crossover_fp32_plan_maps_accelerator_precision_to_no(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    assert base.optimizer is not None
    crossover = replace(
        base,
        unit_id=live.RP_CROSSOVER_UNIT_ID,
        arm_id="C",
        milestones=live.RP_CROSSOVER_MILESTONES,
    )
    plan = live.build_human13_live_model_plan(crossover)
    assert plan.mixed_precision == "fp32"
    backend = live.DefaultHuman13AssemblyBackend()

    captured: dict[str, Any] = {}
    monkeypatch.setattr(
        "accelerate.Accelerator",
        lambda **kwargs: captured.update(accelerator_kwargs=kwargs) or "accelerator",
    )
    assert backend.create_accelerator(plan) == "accelerator"
    # Accelerate has no "fp32" token: full precision is mixed_precision="no"
    assert captured["accelerator_kwargs"]["mixed_precision"] == "no"

    monkeypatch.setattr(
        "src.runtime.validate_accelerator_runtime",
        lambda accelerator, *, expected_mixed_precision: captured.update(
            validated=expected_mixed_precision
        ),
    )
    backend.validate_accelerator("accelerator", plan)
    assert captured["validated"] == "no"

    monkeypatch.setattr("src.runtime.TrainRuntime", lambda **kwargs: kwargs)
    runtime = backend.build_runtime(
        model="model",
        optimizer="optimizer",
        scheduler="scheduler",
        accelerator="accelerator",
        plan=plan,
        pack_count=1,
    )
    assert runtime["expected_mixed_precision"] == "no"


def test_rp_crossover_rejects_an_off_grid_learning_rate() -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    assert base.optimizer is not None
    off_grid_dose = replace(
        base,
        unit_id=live.RP_CROSSOVER_UNIT_ID,
        arm_id="C",
        optimizer=replace(base.optimizer, learning_rate=2.0e-6),
        milestones=live.RP_CROSSOVER_MILESTONES,
    )

    with pytest.raises(
        live.Human13LiveModelError,
        match="sealed qualification learning-rate ray",
    ):
        live.build_human13_live_model_plan(off_grid_dose)


def test_rp_crossover_live_plan_consumes_one_selector_resolved_nondefault_lr() -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    crossover = replace(
        base,
        unit_id=live.RP_CROSSOVER_UNIT_ID,
        arm_id="C",
        milestones=live.RP_CROSSOVER_MILESTONES,
    )
    decision = _selected_lr_decision(1.0e-6)

    plan = live.build_human13_live_model_plan(
        crossover, global_learning_rate_decision=decision
    )

    assert plan.learning_rate == 1.0e-6
    assert plan.learning_rate_resolution == "global_selected"
    assert plan.global_learning_rate_decision_sha256 == decision.content_sha256
    assert plan.resolved_plan_sha256 is not None
    with pytest.raises(live.Human13LiveModelError, match="plan hash"):
        live.validate_human13_live_model_plan(replace(plan, learning_rate=3.0e-6))


def test_matrix_binding_resolves_an_admitted_cell_plan_without_a_second_selector() -> (
    None
):
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    provisional = live.build_human13_live_model_plan(
        replace(
            base,
            unit_id=live.RP_CROSSOVER_UNIT_ID,
            arm_id="B",
            milestones=live.RP_CROSSOVER_MILESTONES,
        )
    )
    decision_sha256 = "d" * 64

    resolved = live.bind_human13_selected_rp_crossover_plan(
        provisional, decision_sha256=decision_sha256
    )

    assert resolved.arm_id == "B"
    assert resolved.learning_rate == provisional.learning_rate
    assert resolved.learning_rate_resolution == "global_selected"
    assert resolved.global_learning_rate_decision_sha256 == decision_sha256
    live.validate_human13_live_model_plan(resolved)


def test_rp_crossover_live_assembly_rejects_a_provisional_leaf_before_backend() -> None:
    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    base = load_arm_config(CONFIG_ROOT / "05_a4.yaml")
    provisional = live.build_human13_live_model_plan(
        replace(
            base,
            unit_id=live.RP_CROSSOVER_UNIT_ID,
            arm_id="C",
            milestones=live.RP_CROSSOVER_MILESTONES,
        )
    )

    with pytest.raises(live.Human13LiveModelError, match="global learning-rate"):
        live.assemble_human13_live_model(
            provisional,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=cast(live.Human13AssemblyBackend, pytest.fail),
        )


def _validation(plan: live.Human13LiveModelPlan) -> live.Human13PlanValidationReceipt:
    return live.Human13PlanValidationReceipt(
        schema_version="human13_live_model_validation.v1",
        arm_id=plan.arm_id,
        adapter_tensor_sha256=plan.source.adapter_sha256,
        special_embedding_tensor_sha256=plan.source.special_embedding_sha256,
        base_config_sha256=live.SOURCE_BASE_CONFIG_SHA256,
        tokenizer_sha256=live.SOURCE_TOKENIZER_SHA256,
        model_actions=dict(live.ZERO_MODEL_ACTIONS),
    )


def _surface_receipt(
    *,
    groups: tuple[str, ...] = ("adapter.language",),
    frozen_vision_count: int = 2,
) -> Any:
    artifact = {
        "phase": "before_first_backward",
        "frozen_towers": ["language", "vision", "aligner"],
        "trainable_towers": list(groups),
        "optimizer_groups": [
            {
                "group_name": group,
                "lr": 1.0e-5,
                "weight_decay": 0.0,
                "parameter_count": 3,
                "parameter_names": [f"{group}.p{index}" for index in range(3)],
            }
            for group in groups
        ],
        "unmatched_trainable_names": [],
        "exact_surface_groups": {
            "trainable_language_dora": {
                "parameter_count": 3,
                "parameter_names": ["adapter.language.p0"],
            },
            "frozen_selected_token_delta": {
                "parameter_count": 1,
                "parameter_names": ["special_token_delta"],
            },
            "frozen_vision": {
                "parameter_count": frozen_vision_count,
                "parameter_names": ["visual.block.weight", "visual.block.bias"],
            },
            "frozen_aligner": {
                "parameter_count": 1,
                "parameter_names": ["visual.merger.weight"],
            },
        },
    }
    return SimpleNamespace(to_artifact_dict=lambda: artifact)


class _FakeLiveModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_dora = torch.nn.Parameter(torch.zeros(2))
        self.shared_embed_delta = torch.nn.Parameter(
            torch.zeros(2), requires_grad=False
        )
        self.config = SimpleNamespace(use_cache=None)


class _Optimizer:
    state: dict[str, object] = {}
    defaults = {
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": 0.0,
    }
    param_groups = [
        {
            "name": "adapter.language",
            "lr": 1.0e-5,
            "weight_decay": 0.0,
        }
    ]


class FakeBackend:
    def __init__(self, *, receipt: Any | None = None, world_size: int = 1) -> None:
        self.calls: list[tuple[str, object]] = []
        self.receipt = receipt or _surface_receipt()
        self.accelerator = SimpleNamespace(
            num_processes=world_size,
            process_index=0,
            mixed_precision="bf16",
            gradient_accumulation_steps=1,
            distributed_type=SimpleNamespace(name="NO"),
            device="cuda:0",
        )
        self.model = _FakeLiveModel()
        self.components = SimpleNamespace(
            model=self.model,
            base_model_path=Path(live.SOURCE_BASE_MODEL_PATH),
            base_config_sha256=live.SOURCE_BASE_CONFIG_SHA256,
            tokenizer_sha256=live.SOURCE_TOKENIZER_SHA256,
        )
        self.adapter_result = SimpleNamespace(
            model=self.model,
            receipt=SimpleNamespace(
                adapter_name="default",
                warm_start={
                    "source_adapter_tensor_sha256": live.SOURCE_ADAPTER_SHA256
                },
            ),
        )
        self.special_result = SimpleNamespace(
            model=self.model,
            shared_embed_delta=self.model.shared_embed_delta,
            receipt="special-receipt",
            loaded_tensor_sha256=live.SOURCE_SPECIAL_EMBEDDING_SHA256,
        )
        self.optimizer = _Optimizer()
        self.scheduler = object()
        self.group_plan = object()
        self.runtime = SimpleNamespace(
            world_size=world_size,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
        )

    def create_accelerator(self, plan: live.Human13LiveModelPlan) -> Any:
        self.calls.append(("accelerator", plan.mixed_precision))
        return self.accelerator

    def validate_accelerator(
        self, accelerator: Any, plan: live.Human13LiveModelPlan
    ) -> None:
        assert accelerator is self.accelerator
        self.calls.append(("validate_accelerator", plan.world_size))

    def load_qwen(self, plan: live.Human13LiveModelPlan) -> Any:
        self.calls.append(
            ("load_qwen", (plan.mixed_precision, plan.attn_implementation))
        )
        return self.components

    def warm_start_language_dora(
        self,
        model: Any,
        components: Any,
        plan: live.Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any:
        assert model is self.model
        assert components is self.components
        self.calls.append(
            ("warm_start_dora", (plan.adapter_rank, plan.adapter_alpha, repo_root))
        )
        return self.adapter_result

    def load_and_freeze_special_token_delta(
        self,
        model: Any,
        components: Any,
        plan: live.Human13LiveModelPlan,
        *,
        repo_root: Path,
    ) -> Any:
        assert model is self.model
        assert components is self.components
        self.calls.append(
            ("load_frozen_delta", (plan.freeze_special_token_delta, repo_root))
        )
        return self.special_result

    def enable_memory_savers(self, model: Any) -> object:
        assert model is self.model
        self.calls.append(("memory_savers", model))
        return {"enabled": True}

    def build_optimizer(
        self,
        model: Any,
        adapter_result: Any,
        plan: live.Human13LiveModelPlan,
    ) -> tuple[Any, Any, Any]:
        assert model is self.model
        assert adapter_result is self.adapter_result
        self.calls.append(
            ("optimizer", (plan.learning_rate, plan.scheduler_horizon_updates))
        )
        return self.optimizer, self.scheduler, self.group_plan

    def build_trainable_surface_receipt(
        self,
        model: Any,
        adapter_result: Any,
        special_result: Any,
        optimizer_group_plan: Any,
    ) -> Any:
        assert model is self.model
        assert adapter_result is self.adapter_result
        assert special_result is self.special_result
        assert optimizer_group_plan is self.group_plan
        self.calls.append(("surface_receipt", model))
        return self.receipt

    def build_runtime(
        self,
        *,
        model: Any,
        optimizer: Any,
        scheduler: Any,
        accelerator: Any,
        plan: live.Human13LiveModelPlan,
        pack_count: int,
    ) -> Any:
        assert model is self.model
        assert optimizer is self.optimizer
        assert scheduler is self.scheduler
        assert accelerator is self.accelerator
        self.calls.append(("runtime", (plan.max_grad_norm, pack_count)))
        return self.runtime


@dataclass(frozen=True)
class _FakeEncoded:
    example_id: str
    input_ids: tuple[int, ...]
    supervised_token_spans: tuple[Any, ...]


def _canonical_fake_manifest_and_rows() -> tuple[Any, tuple[Any, ...]]:
    image_ids = (
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
    images = []
    raws = []
    for image_index, image_id in enumerate(image_ids):
        owner_count = 32 if image_index == len(image_ids) - 1 else 30
        owners = tuple(
            SimpleNamespace(
                owner_id=f"gt:{image_id}:{owner_index}",
                source_object_index=owner_index,
                category=f"category-{owner_index}",
            )
            for owner_index in range(owner_count)
        )
        objects = tuple(
            SimpleNamespace(
                object_id=f"raw-{image_id}-{owner_index}",
                description=f"category-{owner_index}",
            )
            for owner_index in range(owner_count)
        )
        images.append(SimpleNamespace(image_id=image_id, owners=owners))
        raws.append(
            SimpleNamespace(
                example_id=f"example-{image_id}",
                objects=objects,
                metadata={"source": {"image_id": image_id}},
            )
        )
    binding = SimpleNamespace(
        unit_id=live.UNIT_ID,
        purpose="overfit_only",
        panel=SimpleNamespace(
            panel_sha256=live.HUMAN13_PANEL_SHA256,
            owner_count=392,
        ),
        source=SimpleNamespace(
            checkpoint_path=live.SOURCE_CHECKPOINT_PATH,
            base_model_path=live.SOURCE_BASE_MODEL_PATH,
        ),
        surface=SimpleNamespace(
            prompt_policy_fingerprint=live.HUMAN13_PROMPT_POLICY_FINGERPRINT,
            tokenizer_sha256=live.SOURCE_TOKENIZER_SHA256,
        ),
    )
    return SimpleNamespace(
        full_panel=True, binding=binding, images=tuple(images)
    ), tuple(raws)


def test_plan_binds_exact_source_bf16_fa2_language_dora_and_fresh_optimizer() -> None:
    plan = _plan()

    assert plan.arm_id == "A1"
    assert plan.source.checkpoint_path.endswith("checkpoints/step-2444")
    assert plan.source.base_model_path == live.SOURCE_BASE_MODEL_PATH
    assert (plan.mixed_precision, plan.attn_implementation) == (
        "bf16",
        "flash_attention_2",
    )
    assert (plan.adapter_seed_mode, plan.adapter_target_towers) == (
        "warm_start_expand_dora",
        ("language",),
    )
    assert (plan.adapter_rank, plan.adapter_alpha, plan.adapter_dropout) == (
        16,
        32,
        0.0,
    )
    assert plan.freeze_special_token_delta is True
    assert (
        plan.optimizer_name,
        plan.learning_rate,
        plan.betas,
        plan.epsilon,
        plan.weight_decay,
    ) == ("adamw_torch", 1.0e-5, (0.9, 0.999), 1.0e-8, 0.0)
    assert (
        plan.scheduler_name,
        plan.scheduler_warmup_steps,
        plan.scheduler_horizon_updates,
        plan.max_grad_norm,
        plan.world_size,
    ) == ("cosine_with_warmup", 0, 16, 1.0, 1)
    assert plan.milestones == (0, 1, 2, 4, 8, 16)
    assert plan.to_artifact_dict()["model_actions"] == live.ZERO_MODEL_ACTIONS


def test_frozen_source_no_update_cannot_build_live_training_plan() -> None:
    with pytest.raises(live.Human13LiveModelError, match="updated arm"):
        live.build_human13_live_model_plan(CONFIG_ROOT / "00_frozen_source.yaml")


def test_typed_caller_cannot_bypass_language_only_surface() -> None:
    from scripts.research.materialize_human13_k_union_configs import (
        TrainableSurface,
        load_arm_config,
    )

    config = load_arm_config(CONFIG_ROOT / "03_a1.yaml")
    drifted = replace(
        config,
        trainable_surface=TrainableSurface(
            language_tower_dora=True,
            vision_tower=True,
            multimodal_aligner=False,
            token_embeddings=False,
            base_language_weights=False,
        ),
    )
    with pytest.raises(live.Human13LiveModelError, match="language-only DoRA"):
        live.build_human13_live_model_plan(drifted)


def test_plan_and_validate_are_model_runtime_free_and_use_real_source_receipts() -> (
    None
):
    script = """
import json
import sys
from pathlib import Path
from scripts.research.human13_live_model import (
    build_human13_live_model_plan,
    validate_human13_live_model_plan,
)
before = set(sys.modules)
plan = build_human13_live_model_plan(
    Path('configs/coordexp_swift/research/human13_k_union/03_a1.yaml')
)
receipt = validate_human13_live_model_plan(plan)
loaded = sorted(
    name for name in ('torch', 'accelerate', 'transformers', 'peft')
    if name in set(sys.modules) - before
)
print(json.dumps({'loaded': loaded, 'receipt': receipt.to_artifact_dict()}, sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path.cwd(),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["loaded"] == []
    assert result["receipt"]["adapter_tensor_sha256"] == live.SOURCE_ADAPTER_SHA256
    assert (
        result["receipt"]["special_embedding_tensor_sha256"]
        == live.SOURCE_SPECIAL_EMBEDDING_SHA256
    )
    assert result["receipt"]["model_actions"] == live.ZERO_MODEL_ACTIONS


def test_validate_rejects_recomputed_source_tensor_drift_before_live_action(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    original = live._sha256_file

    def drift(path: Path) -> str:
        if path.name == "adapter_model.safetensors":
            return "0" * 64
        return original(path)

    monkeypatch.setattr(live, "_sha256_file", drift)
    with pytest.raises(live.Human13LiveModelError, match="adapter tensor"):
        live.validate_human13_live_model_plan(plan)


def test_live_assembly_is_the_only_action_boundary_and_returns_world_one_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    backend = FakeBackend()
    validation = _validation(plan)
    monkeypatch.setattr(live, "validate_human13_live_model_plan", lambda _: validation)

    assembly = live.assemble_human13_live_model(
        plan,
        pack_count=3,
        repo_root=Path.cwd(),
        backend=backend,
    )

    assert [name for name, _ in backend.calls] == [
        "accelerator",
        "validate_accelerator",
        "load_qwen",
        "warm_start_dora",
        "load_frozen_delta",
        "memory_savers",
        "optimizer",
        "surface_receipt",
        "runtime",
    ]
    assert assembly.validation is validation
    assert assembly.model is backend.model
    assert assembly.optimizer is backend.optimizer
    assert assembly.runtime is backend.runtime
    assert assembly.trainable_surface_receipt.to_artifact_dict()[
        "trainable_towers"
    ] == ["adapter.language"]
    parameter_receipt = assembly.parameter_state_receipt
    assert tuple(item.name for item in parameter_receipt.parameters) == tuple(
        name for name, _parameter in backend.model.named_parameters()
    )
    assert parameter_receipt.selected_delta_parameter_name == "shared_embed_delta"
    assert parameter_receipt.selected_delta_requires_grad is False
    assert len(parameter_receipt.selected_delta_tensor_sha256) == 64
    assert {str(parameter.dtype) for _name, parameter in backend.model.named_parameters()} == {
        "torch.bfloat16"
    }
    assert backend.model.config.use_cache is False


def test_live_assembly_fails_closed_on_world_size_or_surface_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    validation = _validation(plan)
    monkeypatch.setattr(live, "validate_human13_live_model_plan", lambda _: validation)

    with pytest.raises(live.Human13LiveModelError, match="world size one"):
        live.assemble_human13_live_model(
            plan,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=FakeBackend(world_size=2),
        )

    bad_backend = FakeBackend(
        receipt=_surface_receipt(groups=("adapter.language", "token_embeddings"))
    )
    with pytest.raises(live.Human13LiveModelError, match="adapter.language"):
        live.assemble_human13_live_model(
            plan,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=bad_backend,
        )

    unfrozen_vision = FakeBackend(receipt=_surface_receipt(frozen_vision_count=0))
    with pytest.raises(live.Human13LiveModelError, match="adapter.language"):
        live.assemble_human13_live_model(
            plan,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=unfrozen_vision,
        )


@pytest.mark.parametrize("payload", ("adapter", "special"))
def test_live_assembly_seals_exact_loaded_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
    payload: str,
) -> None:
    plan = _plan()
    backend = FakeBackend()
    monkeypatch.setattr(
        live,
        "validate_human13_live_model_plan",
        lambda _: _validation(plan),
    )
    if payload == "adapter":
        backend.adapter_result.receipt.warm_start[
            "source_adapter_tensor_sha256"
        ] = "0" * 64
    else:
        backend.special_result.loaded_tensor_sha256 = "0" * 64

    with pytest.raises(live.Human13LiveModelError, match="loaded .* receipt"):
        live.assemble_human13_live_model(
            plan,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=backend,
        )


def test_live_assembly_builder_seal_rejects_constructed_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setattr(
        live,
        "validate_human13_live_model_plan",
        lambda _: _validation(plan),
    )
    assembly = live.assemble_human13_live_model(
        plan,
        pack_count=1,
        repo_root=Path.cwd(),
        backend=FakeBackend(),
    )

    live.require_admitted_human13_live_assembly(assembly)
    with pytest.raises(live.Human13LiveModelError, match="not issued"):
        live.require_admitted_human13_live_assembly(replace(assembly))


def test_parity_skeleton_encodes_only_image_1584_without_owner_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.research import collect_human13_discovery as discovery
    from src.config import inference
    from src import data, qwen, templates

    calls: list[object] = []
    config = SimpleNamespace(
        backend=SimpleNamespace(
            type="hf", hf=SimpleNamespace(attn_implementation="sdpa")
        ),
        model=SimpleNamespace(
            dtype="fp32",
            base_model=live.SOURCE_BASE_MODEL_PATH,
        ),
        data=SimpleNamespace(input_jsonl=live.HUMAN13_PANEL_PATH),
        adapter=SimpleNamespace(path=live.SOURCE_ADAPTER_PATH),
        embedding_delta=SimpleNamespace(path=live.SOURCE_SPECIAL_EMBEDDING_PATH),
        template=SimpleNamespace(
            object_field_order="desc_first",
            object_ordering="geo_sorted_xy",
            assistant_format="object_box_closed",
            prompt=SimpleNamespace(system="system", user="user"),
        ),
    )
    resolved = SimpleNamespace(config=config)
    raw = SimpleNamespace(metadata={"source": {"image_id": 1584}})
    encoded = _FakeEncoded(
        example_id="image-1584",
        input_ids=(10, 11, 12, 13, 14),
        supervised_token_spans=(SimpleNamespace(physical_token_start=3),),
    )

    monkeypatch.setattr(inference, "load_infer_config", lambda path: resolved)
    monkeypatch.setattr(
        discovery,
        "_prompt_policy_fingerprint",
        lambda value: live.HUMAN13_PROMPT_POLICY_FINGERPRINT,
    )

    def raw_rows(path):
        calls.append(("load_rows", str(path)))
        yield raw
        raise AssertionError("parity skeleton must stop after image 1584")

    monkeypatch.setattr(data, "load_raw_examples", raw_rows)
    monkeypatch.setattr(
        templates,
        "render_example",
        lambda item, template: calls.append("render") or object(),
    )

    components = object()

    def encode(item, rendered, **kwargs):
        del rendered
        calls.append(("encode", item.metadata["source"]["image_id"]))
        assert kwargs["components"] is components
        assert kwargs["global_max_length"] == 12_000
        assert kwargs["materialize_image_pixels"] is True
        return encoded

    monkeypatch.setattr(qwen, "encode_rendered_example", encode)
    monkeypatch.setattr(
        live,
        "_derive_owner_row_tokens",
        lambda *args: (_ for _ in ()).throw(
            AssertionError("parity skeleton must not derive owner rows")
        ),
    )

    result = live.build_human13_parity_skeleton(
        image_id=1584,
        components=components,
        repo_root=Path.cwd(),
    )

    assert result is encoded
    assert result.prompt_token_count == 3
    assert not hasattr(result, "owner_row_tokens")
    assert calls == [
        ("load_rows", live.HUMAN13_PANEL_PATH),
        "render",
        ("encode", 1584),
    ]


def test_live_assembly_rejects_trainable_source_delta_or_nonfresh_optimizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    monkeypatch.setattr(
        live,
        "validate_human13_live_model_plan",
        lambda _: _validation(plan),
    )
    trainable_delta_backend = FakeBackend()
    trainable_delta_backend.special_result.shared_embed_delta.requires_grad_(True)
    with pytest.raises(live.Human13LiveModelError, match="must remain frozen"):
        live.assemble_human13_live_model(
            plan,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=trainable_delta_backend,
        )

    stale_backend = FakeBackend()
    stale_backend.optimizer.state = {"old": {"step": 9}}
    with pytest.raises(live.Human13LiveModelError, match="fresh AdamW"):
        live.assemble_human13_live_model(
            plan,
            pack_count=1,
            repo_root=Path.cwd(),
            backend=stale_backend,
        )


def test_processor_skeletons_bind_exact_prompt_counts_and_owner_row_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.config.models import TemplateConfig, TemplatePromptConfig

    manifest, raws = _canonical_fake_manifest_and_rows()
    components = SimpleNamespace(
        processor="processor",
        tokenizer="tokenizer",
        processor_identity="processor-identity",
        tokenizer_sha256=live.SOURCE_TOKENIZER_SHA256,
    )
    template = TemplateConfig(
        object_field_order="desc_first",
        object_ordering="geo_sorted_xy",
        assistant_format="object_box_closed",
        prompt=TemplatePromptConfig(system="system", user="user"),
    )
    resolved = SimpleNamespace(
        config=SimpleNamespace(
            data=SimpleNamespace(input_jsonl=live.HUMAN13_PANEL_PATH),
            model=SimpleNamespace(base_model=live.SOURCE_BASE_MODEL_PATH),
            template=template,
        )
    )
    monkeypatch.setattr("src.config.inference.load_infer_config", lambda _: resolved)
    monkeypatch.setattr("src.data.load_raw_examples", lambda _: raws)

    def fake_render(raw: Any, template_config: Any) -> Any:
        assert template_config.object_ordering == "geo_sorted_xy"
        return SimpleNamespace(
            example_id=raw.example_id,
            # Renderer identity and inference prompt-policy identity are two
            # intentionally distinct contracts.
            template_fingerprint="renderer-template-fingerprint",
        )

    monkeypatch.setattr("src.templates.render_example", fake_render)
    monkeypatch.setattr(
        "scripts.research.collect_human13_discovery._prompt_policy_fingerprint",
        lambda _: live.HUMAN13_PROMPT_POLICY_FINGERPRINT,
    )

    def fake_encode(raw: Any, rendered: Any, **kwargs: Any) -> _FakeEncoded:
        assert rendered.example_id == raw.example_id
        assert kwargs["components"] is components
        assert kwargs["processor_config"].model_dump() == {
            "do_resize": False,
            "max_raw_pixels": 1_000_000_000,
            "max_merged_visual_tokens": 1_000_000,
        }
        assert kwargs["global_max_length"] == 12_000
        assert kwargs["materialize_image_pixels"] is True
        prompt = (101, 102, 103)
        row_tokens = tuple(5000 + index for index in range(len(raw.objects)))
        spans = tuple(
            SimpleNamespace(
                object_id=obj.object_id,
                physical_token_start=len(prompt) + index,
                physical_token_end=len(prompt) + index + 1,
                token_ids=(row_tokens[index],),
            )
            for index, obj in enumerate(raw.objects)
        )
        terminal = (999,)
        return _FakeEncoded(
            raw.example_id,
            prompt + row_tokens + terminal,
            spans,
        )

    monkeypatch.setattr("src.qwen.encode_rendered_example", fake_encode)

    skeletons = live.build_human13_processor_skeletons(
        manifest,
        components,
        repo_root=Path.cwd(),
    )

    assert tuple(skeletons) == tuple(image.image_id for image in manifest.images)
    assert sum(len(item.owner_row_tokens) for item in skeletons.values()) == 392
    first = skeletons[1584]
    assert first.prompt_token_count == 3
    assert first.input_ids[: first.prompt_token_count] == (101, 102, 103)
    assert first.owner_row_tokens["gt:1584:0"] == (5000,)
    assert first.owner_row_tokens["gt:1584:29"] == (5029,)


def test_sixteen_update_schedule_uses_panel_packs_and_exact_checkpoints() -> None:
    schedule = live.build_human13_update_schedule(pack_count=3)

    assert schedule.resolved_max_steps == 16
    assert schedule.packs_per_epoch == 3
    assert schedule.requested_pack_presentations == 48
    assert schedule.actual_pack_presentations == 48
    assert schedule.tail_fill_pack_count == 0
    assert schedule.runtime_batch.to_artifact_dict() == {
        "world_size": 1,
        "effective_batch_size": 3,
        "resolved_grad_accum_steps": 3,
    }
    assert tuple(event.planned_step_id for event in schedule.events["checkpoint"]) == (
        1,
        2,
        4,
        8,
        16,
    )
    assert tuple(event.planned_step_id for event in schedule.events["final"]) == (16,)


def test_checkpoint_kwargs_bind_composed_payload_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    backend = FakeBackend()
    monkeypatch.setattr(
        live,
        "validate_human13_live_model_plan",
        lambda _: _validation(plan),
    )
    assembly = live.assemble_human13_live_model(
        plan,
        pack_count=1,
        repo_root=Path.cwd(),
        backend=backend,
    )

    assert live.build_human13_checkpoint_kwargs(assembly) == {
        "accelerator": backend.accelerator,
        "adapter_name": "default",
        "special_token_result": backend.special_result,
        "base_model_path": Path(live.SOURCE_BASE_MODEL_PATH),
        "base_config_sha256": live.SOURCE_BASE_CONFIG_SHA256,
        "tokenizer_sha256": live.SOURCE_TOKENIZER_SHA256,
    }


def test_checkpoint_writer_and_cpu_readback_bind_exact_composed_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    backend = FakeBackend()
    monkeypatch.setattr(
        live,
        "validate_human13_live_model_plan",
        lambda _: _validation(plan),
    )
    assembly = live.assemble_human13_live_model(
        plan,
        pack_count=1,
        repo_root=Path.cwd(),
        backend=backend,
    )
    run_dir = tmp_path.resolve() / "arm-a1"
    writer = object()
    monkeypatch.setattr(
        "src.artifacts.CheckpointWriter",
        lambda *, run_dir: writer if run_dir == run_dir.parent / "arm-a1" else None,
    )
    assert live.build_human13_checkpoint_writer(run_dir) is writer

    checkpoint_dir = run_dir / "checkpoints" / "step-4"
    (checkpoint_dir / "adapter").mkdir(parents=True)
    (checkpoint_dir / "special_token_embeddings").mkdir()
    adapter_identity = {
        "kind": "dora_adapter",
        "fingerprint": "a" * 64,
        "semantic_identity": {
            "peft_type": "LORA",
            "use_dora": True,
            "base_model_name_or_path": live.SOURCE_BASE_MODEL_PATH,
            "target_modules": [
                "down_proj",
                "gate_proj",
                "k_proj",
                "o_proj",
                "q_proj",
                "up_proj",
                "v_proj",
            ],
            "r": 16,
            "lora_alpha": 32.0,
            "lora_A_count": 3,
            "lora_B_count": 3,
            "lora_magnitude_vector_count": 3,
        },
    }
    special_identity = {
        "kind": "special_token_embedding_delta",
        "fingerprint": "b" * 64,
        "semantic_identity": {
            "semantics": "additive_delta",
            "tensor_key": "shared_embed_delta",
            "tensor_shape": [1004, 2048],
            "tensor_dtype": "float32",
            "token_ids": list(range(1004)),
            "base_model_path": live.SOURCE_BASE_MODEL_PATH,
            "base_config_sha256": live.SOURCE_BASE_CONFIG_SHA256,
            "tokenizer_sha256": live.SOURCE_TOKENIZER_SHA256,
            "tie_word_embeddings": True,
        },
    }
    captured: dict[str, Any] = {}

    def fake_adapter(path: Path, expected_base_model_path: Path) -> Any:
        captured["adapter"] = (path, expected_base_model_path)
        return adapter_identity

    def fake_special(path: Path, **kwargs: Any) -> Any:
        captured["special"] = (path, kwargs)
        return special_identity

    monkeypatch.setattr("src.adapters.dora.inspect_dora_adapter_payload", fake_adapter)
    monkeypatch.setattr(
        "src.qwen.special_token_embeddings.inspect_special_token_embedding_delta_payload",
        fake_special,
    )
    receipt = live.readback_human13_checkpoint(
        checkpoint_dir,
        expected_step=4,
        assembly=assembly,
    )

    assert receipt.step == 4
    assert receipt.adapter_fingerprint == "a" * 64
    assert receipt.special_embedding_fingerprint == "b" * 64
    assert captured["adapter"] == (
        checkpoint_dir / "adapter",
        Path(live.SOURCE_BASE_MODEL_PATH),
    )
    assert captured["special"][1] == {
        "expected_base_model_path": Path(live.SOURCE_BASE_MODEL_PATH),
        "expected_base_config_sha256": live.SOURCE_BASE_CONFIG_SHA256,
        "expected_tokenizer_sha256": live.SOURCE_TOKENIZER_SHA256,
    }

    from scripts.research.materialize_human13_k_union_configs import load_arm_config

    on_policy_config = replace(
        load_arm_config(CONFIG_ROOT / "05_a4.yaml"),
        unit_id=live.ON_POLICY_UNIT_ID,
        arm_id="O-First-Safe",
        milestones=live.ON_POLICY_MILESTONES,
    )
    on_policy_assembly = replace(
        assembly, plan=live.build_human13_live_model_plan(on_policy_config)
    )
    step_three = run_dir / "checkpoints" / "step-3"
    (step_three / "adapter").mkdir(parents=True)
    (step_three / "special_token_embeddings").mkdir()
    assert (
        live.readback_human13_checkpoint(
            step_three, expected_step=3, assembly=on_policy_assembly
        ).step
        == 3
    )


def test_default_backend_projects_exact_runtime_configs_with_cpu_mocks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    backend = live.DefaultHuman13AssemblyBackend()
    captured: dict[str, Any] = {}

    def fake_qwen_load(options: Any) -> Any:
        captured["qwen_options"] = options
        return "components"

    monkeypatch.setattr(
        "src.qwen.runtime_loading.load_qwen_components_from_options",
        fake_qwen_load,
    )
    assert backend.load_qwen(plan) == "components"
    options = captured["qwen_options"]
    assert (
        options.base_model,
        options.dtype,
        options.attn_implementation,
        options.patch_embed_linearization,
        options.load_model,
    ) == (
        live.SOURCE_BASE_MODEL_PATH,
        "bf16",
        "flash_attention_2",
        "enabled",
        True,
    )

    adapter_calls: dict[str, Any] = {}
    monkeypatch.setattr(
        "src.adapters.load_default_adapter_source_gate_evidence",
        lambda root: ("gate", root),
    )

    def fake_build_adapter(config: Any, evidence: Any, *, base_model_path: Any) -> str:
        adapter_calls.update(
            config=config, evidence=evidence, base_model_path=base_model_path
        )
        return "adapter-plan"

    monkeypatch.setattr("src.adapters.build_adapter_setup_plan", fake_build_adapter)
    monkeypatch.setattr(
        "src.adapters.setup_dora_adapter",
        lambda model, adapter_plan: (
            adapter_calls.update(model=model, plan=adapter_plan) or "adapter-result"
        ),
    )
    components = SimpleNamespace(base_model_path=Path(live.SOURCE_BASE_MODEL_PATH))
    assert (
        backend.warm_start_language_dora(
            "model", components, plan, repo_root=Path("/repo")
        )
        == "adapter-result"
    )
    config = adapter_calls["config"]
    assert config.model_dump() == {
        "type": "dora",
        "seed_mode": "warm_start_expand_dora",
        "path": None,
        "source_adapter_path": plan.source.adapter_path,
        "repaired_embedding_payload_path": plan.source.special_embedding_path,
        "target_towers": ("language",),
        "target_modules": "all_linear",
        "rank": 16,
        "alpha": 32,
        "dropout": 0.0,
        "bias": "none",
    }


def test_default_backend_freezes_delta_and_builds_exact_optimizer_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    backend = live.DefaultHuman13AssemblyBackend()
    delta = SimpleNamespace(requires_grad=True)

    def requires_grad_(value: bool) -> None:
        delta.requires_grad = value

    delta.requires_grad_ = requires_grad_
    special_result = SimpleNamespace(
        model="special-model", shared_embed_delta=delta, receipt="special"
    )
    components = SimpleNamespace(
        token_identity="tokens",
        base_model_path=Path(live.SOURCE_BASE_MODEL_PATH),
        base_config_sha256=live.SOURCE_BASE_CONFIG_SHA256,
        tokenizer_sha256=live.SOURCE_TOKENIZER_SHA256,
    )
    captured: dict[str, Any] = {}

    def fake_special_selection(config: Any, tokens: Any) -> tuple[Any, Any]:
        captured["special_config"] = config
        return (config, tokens)

    monkeypatch.setattr(
        "src.qwen.build_default_special_token_selection",
        fake_special_selection,
    )

    def fake_special_gate(root: Path, *, selection: Any = None) -> tuple[Any, ...]:
        captured["special_gate_selection"] = selection
        return ("special-gate", root, selection)

    monkeypatch.setattr(
        "src.qwen.special_token_embeddings.load_default_special_token_embedding_source_gate_evidence",
        fake_special_gate,
    )
    monkeypatch.setattr(
        "src.qwen.install_special_token_embedding_deltas",
        lambda model, selection, source_gate: special_result,
    )

    def fake_load_special(result: Any, payload: Path, **kwargs: Any) -> str:
        captured.update(result=result, payload=payload, load_kwargs=kwargs)
        return "load-receipt"

    monkeypatch.setattr(
        "src.qwen.load_special_token_embedding_deltas", fake_load_special
    )
    monkeypatch.setattr(
        live,
        "_sha256_file",
        lambda path: live.SOURCE_SPECIAL_EMBEDDING_SHA256,
    )
    loaded = backend.load_and_freeze_special_token_delta(
        "dora-model", components, plan, repo_root=Path("/repo")
    )
    assert loaded.model == special_result.model
    assert loaded.shared_embed_delta is special_result.shared_embed_delta
    assert loaded.receipt == special_result.receipt
    assert loaded.load_receipt == "load-receipt"
    assert loaded.loaded_tensor_sha256 == live.SOURCE_SPECIAL_EMBEDDING_SHA256
    assert delta.requires_grad is False
    assert captured["special_gate_selection"] == (captured["special_config"], "tokens")
    assert captured["payload"] == Path(plan.source.special_embedding_path)
    assert captured["load_kwargs"] == {
        "expected_base_model_path": Path(live.SOURCE_BASE_MODEL_PATH),
        "expected_base_config_sha256": live.SOURCE_BASE_CONFIG_SHA256,
        "expected_tokenizer_sha256": live.SOURCE_TOKENIZER_SHA256,
    }

    adapter_result = SimpleNamespace(receipt="adapter-receipt")
    monkeypatch.setattr(
        "src.optim.build_optimizer_group_plan",
        lambda model, config, **kwargs: (
            captured.update(optimizer_config=config, group_kwargs=kwargs) or "groups"
        ),
    )
    monkeypatch.setattr(
        "src.optim.build_optimizer_and_scheduler",
        lambda config, groups, **kwargs: (
            captured.update(build_kwargs=kwargs, build_groups=groups)
            or ("optimizer", "scheduler")
        ),
    )
    optimizer, scheduler, groups = backend.build_optimizer(
        "model", adapter_result, plan
    )
    assert (optimizer, scheduler, groups) == ("optimizer", "scheduler", "groups")
    optimizer_config = captured["optimizer_config"]
    assert optimizer_config.name == "adamw_torch"
    assert optimizer_config.betas == (0.9, 0.999)
    assert optimizer_config.epsilon == 1.0e-8
    assert optimizer_config.groups.adapters.language.model_dump() == {
        "lr": 1.0e-5,
        "weight_decay": 0.0,
    }
    assert optimizer_config.groups.adapters.vision is None
    assert optimizer_config.groups.adapters.aligner is None
    assert optimizer_config.scheduler.model_dump() == {
        "name": "cosine_with_warmup",
        "warmup_ratio": None,
        "warmup_steps": 0,
        "kwargs": {},
    }
    assert captured["group_kwargs"] == {
        "adapter_receipt": "adapter-receipt",
        "special_token_receipt": None,
    }
    assert captured["build_kwargs"] == {"total_training_steps": 16}

    monkeypatch.setattr("src.runtime.TrainRuntime", lambda **kwargs: kwargs)
    runtime = backend.build_runtime(
        model="model",
        optimizer="optimizer",
        scheduler="scheduler",
        accelerator="accelerator",
        plan=plan,
        pack_count=4,
    )
    assert runtime["runtime_batch"].to_artifact_dict() == {
        "world_size": 1,
        "effective_batch_size": 4,
        "resolved_grad_accum_steps": 4,
    }
    assert runtime["expected_mixed_precision"] == "bf16"
    assert runtime["max_grad_norm"] == 1.0


def test_plan_constants_are_not_caller_mutable() -> None:
    plan = _plan()
    with pytest.raises(live.Human13LiveModelError, match="frozen live-model contract"):
        live.validate_human13_live_model_plan(replace(plan, learning_rate=3.0e-5))


def _device_identity_accelerator(device: str = "cuda") -> Any:
    return SimpleNamespace(
        device=device,
        num_processes=1,
        process_index=0,
        local_process_index=0,
        distributed_type=SimpleNamespace(name="NO"),
    )


def _device_identity_parameter(device: str) -> Any:
    return SimpleNamespace(device=device, requires_grad=True, grad=None)


def test_cuda_logical_device_identity_resolves_indexless_accelerate_device() -> None:
    receipt = live.build_human13_cuda_logical_device_identity(
        _device_identity_accelerator("cuda"),
        (("adapter.language.weight", _device_identity_parameter("cuda:0")),),
        cuda_available=True,
        current_cuda_index=0,
        cuda_visible_devices="0,1",
    )

    assert receipt.canonical_device_type == "cuda"
    assert receipt.canonical_index == 0
    assert receipt.accelerator_device_index is None
    assert receipt.parameter_device_indices == (0,)
    assert receipt.visible_device_tokens == ("0", "1")
    assert receipt.content_sha256 == receipt.to_artifact_dict()["content_sha256"]


def test_cuda_logical_device_identity_rejects_missing_visibility_contract() -> None:
    with pytest.raises(live.Human13LiveModelError, match="visibility mapping"):
        live.build_human13_cuda_logical_device_identity(
            _device_identity_accelerator("cuda"),
            (("adapter.language.weight", _device_identity_parameter("cuda:0")),),
            cuda_available=True,
            current_cuda_index=0,
            cuda_visible_devices=None,
        )


@pytest.mark.parametrize(
    ("accelerator_device", "parameter_devices", "visible_devices", "message"),
    (
        ("cuda:1", ("cuda:0",), "0,1", "explicit accelerator device conflicts"),
        (
            "cuda",
            ("cuda:0", "cuda:1"),
            "0,1",
            "trainable parameters span multiple CUDA devices",
        ),
        ("cuda", ("cuda:0",), "0,,1", "CUDA visibility mapping is ambiguous"),
    ),
)
def test_cuda_logical_device_identity_rejects_ambiguous_mapping(
    accelerator_device: str,
    parameter_devices: tuple[str, ...],
    visible_devices: str,
    message: str,
) -> None:
    with pytest.raises(live.Human13LiveModelError, match=message):
        live.build_human13_cuda_logical_device_identity(
            _device_identity_accelerator(accelerator_device),
            tuple(
                (f"adapter.language.{index}", _device_identity_parameter(device))
                for index, device in enumerate(parameter_devices)
            ),
            cuda_available=True,
            current_cuda_index=0,
            cuda_visible_devices=visible_devices,
        )
