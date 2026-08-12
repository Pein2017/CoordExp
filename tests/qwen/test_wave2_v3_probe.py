from __future__ import annotations

from contextlib import contextmanager
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

from src.qwen.parity import ParityContractError


@pytest.fixture(autouse=True)
def _wave2_flash_attention_deterministic(monkeypatch) -> None:
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")


def _load_probe_module():
    name = "coordexp_wave2_v3_probe_test_module"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/probes/coordexp_swift/wave2_packed_parity.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _Receipt:
    def __init__(self, index: int) -> None:
        self._index = index

    def to_artifact_dict(self) -> dict[str, object]:
        return {"fa2_varlen": {"proof": None}, "microstep_index": self._index}


class _Bundle:
    def __init__(self, loss: torch.Tensor, index: int) -> None:
        self.total_loss = loss
        self._index = index

    def to_artifact_dict(self) -> dict[str, object]:
        return {
            "microstep_index": self._index,
            "loss": float(self.total_loss.detach()),
        }


class _LossRunner:
    def __init__(self, parameter: torch.nn.Parameter, events: list[str]) -> None:
        self._parameter = parameter
        self._events = events

    def compute_micro_step(self, _context, _plan, *, local_micro_step_index: int):
        self._events.append(f"loss:{local_micro_step_index}")
        coefficient = float(local_micro_step_index + 1)
        return _Bundle(self._parameter.sum() * coefficient, local_micro_step_index)

    def finalize_planned_step(self, artifacts, _plan):
        return {"terms": [], "microsteps": list(artifacts)}


class _Accelerator:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.backward_losses: list[torch.Tensor] = []

    @contextmanager
    def no_sync(self, _model):
        self._events.append("no_sync:enter")
        try:
            yield
        finally:
            self._events.append("no_sync:exit")

    def backward(self, loss: torch.Tensor) -> None:
        self._events.append(f"backward:{len(self.backward_losses)}")
        self.backward_losses.append(loss)
        loss.backward()


def test_production_fp32_loss_upcast_stays_in_graph_and_receipt_snapshot_detaches() -> (
    None
):
    from accelerate.utils.operations import convert_outputs_to_fp32

    source = torch.tensor([1.0], dtype=torch.bfloat16, requires_grad=True)

    @convert_outputs_to_fp32
    def production_forward(value: torch.Tensor) -> torch.Tensor:
        return value * 2

    graph_connected_logits = production_forward(source)
    assert graph_connected_logits.dtype == torch.float32
    assert graph_connected_logits.requires_grad is True
    assert graph_connected_logits.grad_fn is not None

    receipt_snapshot = graph_connected_logits.detach().float().cpu()
    assert receipt_snapshot.dtype == torch.float32
    assert receipt_snapshot.requires_grad is False
    assert receipt_snapshot.grad_fn is None

    graph_connected_logits.sum().backward()
    assert source.grad is not None
    assert source.grad.dtype == torch.bfloat16
    assert torch.equal(source.grad, torch.tensor([2.0], dtype=torch.bfloat16))


def test_separate_arm_streams_two_immediate_backwards_with_one_clear(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    events: list[str] = []
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 99.0)
    accelerator = _Accelerator(events)
    inputs = (
        SimpleNamespace(input_ids=torch.tensor([0])),
        SimpleNamespace(input_ids=torch.tensor([1])),
    )

    def fake_forward(_model, value, **_kwargs):
        index = int(value.input_ids.item())
        events.append(f"forward:{index}")
        return SimpleNamespace(
            logits=torch.tensor([[[float(index), 0.0]]], dtype=torch.float32),
            logits_position_ids=torch.tensor([0]),
            receipt=_Receipt(index),
        )

    clear_count = 0

    def fake_clear(current_model) -> None:
        nonlocal clear_count
        clear_count += 1
        current_model.zero_grad(set_to_none=True)

    @contextmanager
    def fake_autocast(_model):
        yield []

    monkeypatch.setattr(probe, "run_qwen_forward", fake_forward)
    monkeypatch.setattr(
        probe, "LossContext", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(probe, "_zero_grad", fake_clear)
    monkeypatch.setattr(probe, "_capture_inner_cuda_autocast", fake_autocast)
    monkeypatch.setattr(
        probe, "_validate_inner_autocast_observations", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        probe, "_assert_forward_output_dtype", lambda *_a, **_k: "torch.float32"
    )
    monkeypatch.setattr(probe, "selected_logits_by_semantic_key", lambda _context: {})
    monkeypatch.setattr(probe, "_timed_start", lambda _device: 1)
    monkeypatch.setattr(probe, "_timed_end", lambda _device, _start: 1)
    monkeypatch.setattr(probe, "snapshot_trainable_gradients", lambda *_a, **_k: ())

    arm = probe._execute_arm(
        name="separate_reference",
        model=model,
        forward_inputs=inputs,
        token_sequences=(object(), object()),
        loss_runner=_LossRunner(model.weight, events),
        loss_plan=object(),
        vocab_groups=object(),
        expected_vocab_size=2,
        capture_proof=False,
        accelerator=accelerator,
        expected_inventory={},
    )

    assert clear_count == 1
    assert events == [
        "no_sync:enter",
        "forward:0",
        "loss:0",
        "backward:0",
        "no_sync:exit",
        "forward:1",
        "loss:1",
        "backward:1",
    ]
    assert arm.backward_call_count == 2
    assert [event["microstep_index"] for event in arm.backward_events] == [0, 1]
    assert arm.backward_events[0]["accumulation_context"] == "accelerator.no_sync"
    assert arm.backward_events[1]["accumulation_context"] == "sync_gradients"
    assert accelerator.backward_losses[0] is not accelerator.backward_losses[1]
    assert torch.equal(model.weight.grad, torch.full_like(model.weight, 3.0))


def test_attempt_marker_precedes_cuda_rng_accelerator_and_model_gpu_setup() -> None:
    probe = _load_probe_module()
    caller_source = inspect.getsource(probe._execute_real_probe_with_sampler)
    publisher_source = inspect.getsource(probe._publish_attempt_start_marker)

    linked_callback = publisher_source.index("def record_linked_attempt()")
    loaded_source = publisher_source.index(
        'failure_evidence.record("source_identity", dict(source_identity))',
        linked_callback,
    )
    concrete_inventory = publisher_source.index('"trainable_inventory",', loaded_source)
    marker_evidence = publisher_source.index(
        'failure_evidence.record("attempt_marker", reference)',
        concrete_inventory,
    )
    attempt_reached = publisher_source.index(
        'failure_evidence.reach("attempt_started")', marker_evidence
    )
    persisted_write = publisher_source.index(
        "write_strict_json_atomic(target, marker, on_linked=record_linked_attempt)"
    )
    persisted_validation = publisher_source.index(
        "persisted = validate_attempt_marker(", attempt_reached
    )
    exact_persistence_check = publisher_source.index(
        "if persisted != marker or persisted_reference != reference:",
        persisted_validation,
    )
    injected_failure = publisher_source.index("inject_after()", exact_persistence_check)

    concrete_runtime_inventory = caller_source.index(
        "concrete_inventory = concrete_trainable_inventory(model)"
    )
    late_config_attestation = caller_source.index(
        "_attest_fresh_runtime_config_immediately_before_marker(plan)",
        concrete_runtime_inventory,
    )
    final_idle = caller_source.index(
        "_append_final_gpu_idle_check(", late_config_attestation
    )
    marker = caller_source.index("_publish_attempt_start_marker(")
    cuda_seed = caller_source.index("torch.cuda.manual_seed_all(")
    accelerator = caller_source.index("_build_exact_one_rank_accelerator(")
    model_transfer = caller_source.index("model.to(device)")

    assert (
        linked_callback
        < loaded_source
        < concrete_inventory
        < marker_evidence
        < attempt_reached
        < persisted_write
        < persisted_validation
        < exact_persistence_check
        < injected_failure
    )
    assert (
        concrete_runtime_inventory
        < late_config_attestation
        < final_idle
        < marker
        < cuda_seed
        < accelerator
        < model_transfer
    )


def test_prephase_failure_resources_create_a_complete_measurement_envelope(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda _device: 11)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda _device: 13)

    probe._record_failure_resources(
        evidence,
        device=torch.device("cuda:0"),
        device_sampler={"status": "completed"},
    )

    assert evidence.fields["measurement"] == {
        "schema": "coordexp-swift-wave2-failure-measurement-v1",
        "completed_phases": [],
        "phase_boundary_samples": [],
        "failure_resource_summary": {
            "device": "cuda:0",
            "torch_peak_allocated_bytes": 11,
            "torch_peak_reserved_bytes": 13,
            "device_sampler": {"status": "completed"},
        },
    }


def test_boundary_negative_arm_is_forward_only_and_clears_stale_gradients(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    events: list[str] = []
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.full_like(model.weight, 99.0)
    inputs = (SimpleNamespace(input_ids=torch.tensor([0])),)
    clear_count = 0

    def fake_clear(current_model) -> None:
        nonlocal clear_count
        clear_count += 1
        current_model.zero_grad(set_to_none=True)

    @contextmanager
    def fake_autocast(_model):
        yield []

    monkeypatch.setattr(probe, "_zero_grad", fake_clear)
    monkeypatch.setattr(probe, "_capture_inner_cuda_autocast", fake_autocast)
    monkeypatch.setattr(
        probe,
        "run_qwen_forward",
        lambda *_args, **_kwargs: SimpleNamespace(
            logits=torch.tensor([[[0.0, 1.0]]], dtype=torch.float32),
            logits_position_ids=torch.tensor([0]),
            receipt=_Receipt(0),
        ),
    )
    monkeypatch.setattr(
        probe, "LossContext", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        probe, "_validate_inner_autocast_observations", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        probe, "_assert_forward_output_dtype", lambda *_a, **_k: "torch.float32"
    )
    monkeypatch.setattr(probe, "selected_logits_by_semantic_key", lambda _context: {})
    monkeypatch.setattr(probe, "_timed_start", lambda _device: 1)
    monkeypatch.setattr(probe, "_timed_end", lambda _device, _start: 1)
    monkeypatch.setattr(
        probe,
        "snapshot_trainable_gradients",
        lambda *_a, **_k: (_ for _ in ()).throw(
            AssertionError("negative must not snapshot gradients")
        ),
    )
    accelerator = SimpleNamespace(
        backward=lambda _loss: (_ for _ in ()).throw(
            AssertionError("negative must not backward")
        )
    )

    arm = probe._execute_arm(
        name="packed_merged_boundary_negative",
        model=model,
        forward_inputs=inputs,
        token_sequences=(object(),),
        loss_runner=_LossRunner(model.weight, events),
        loss_plan=object(),
        vocab_groups=object(),
        expected_vocab_size=2,
        capture_proof=False,
        accelerator=accelerator,
        expected_inventory={},
        perform_backward=False,
    )

    assert clear_count == 1
    assert model.weight.grad is None
    assert arm.gradients == ()
    assert arm.backward_call_count == 0
    assert arm.backward_events == ()


def test_concrete_inventory_is_bound_to_adapter_and_delta_setup_receipts() -> None:
    probe = _load_probe_module()
    inventory = {
        "parameters": [
            {"name": "layer.lora_A.default.weight", "group": "lora_A"},
            {"name": "layer.lora_B.default.weight", "group": "lora_B"},
            {
                "name": "head.shared_embed_delta",
                "group": "special_token_delta",
            },
        ]
    }
    adapter = SimpleNamespace(
        trainable_names=(
            "layer.lora_A.default.weight",
            "layer.lora_B.default.weight",
        )
    )
    delta = SimpleNamespace(delta_parameter_names=("head.shared_embed_delta",))

    probe._assert_inventory_matches_setup_receipts(
        inventory,
        adapter_receipt=adapter,
        special_token_receipt=delta,
    )
    delta.delta_parameter_names = ("other.shared_embed_delta",)
    with pytest.raises(ParityContractError) as caught:
        probe._assert_inventory_matches_setup_receipts(
            inventory,
            adapter_receipt=adapter,
            special_token_receipt=delta,
        )
    assert caught.value.code == "qwen.parity.trainable_setup_receipt_binding"


def test_attempt_marker_failure_before_publication_leaves_target_absent(
    tmp_path: Path, monkeypatch
) -> None:
    probe = _load_probe_module()
    marker = tmp_path / "attempt.json"
    monkeypatch.setattr(probe, "finalize_attempt_marker", _fake_finalize_marker)
    monkeypatch.setattr(probe, "validate_attempt_marker", _fake_validate_marker)

    with pytest.raises(RuntimeError, match="before marker"):
        probe._publish_attempt_start_marker(
            marker,
            plan={"plan_sha256": "a" * 64},
            receipt_target=tmp_path / "receipt.json",
            command_identity={"argv_sha256": "b" * 64},
            source_identity={"repo": "frozen"},
            concrete_inventory={"inventory_sha256": "c" * 64},
            inject_before=lambda: (_ for _ in ()).throw(RuntimeError("before marker")),
        )

    assert not marker.exists()


@pytest.mark.parametrize("value", [None, "0", "true"])
def test_probe_rejects_unfrozen_flash_attention_determinism_before_plan_work(
    monkeypatch,
    value: str | None,
) -> None:
    probe = _load_probe_module()
    if value is None:
        monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    else:
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", value)
    monkeypatch.setattr(
        probe,
        "_materialize",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("plan materialization must not begin")
        ),
    )
    with pytest.raises(ParityContractError) as caught:
        probe.build_plan(
            config_path="ignored.yaml",
            parent_v2_plan_path="ignored.json",
        )
    assert caught.value.code == "qwen.parity.flash_attention_deterministic"


def test_probe_rejects_flash_attention_determinism_drift_before_model_work(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "0")
    with pytest.raises(ParityContractError) as caught:
        probe._execute_real_probe_with_sampler(
            {"determinism": {"flash_attention_deterministic": "1"}},
            object(),
            device=torch.device("cpu"),
            gpu_idle_preflight={},
            device_sampler=object(),
            receipt_path=Path("receipt.json"),
            attempt_marker_path=Path("attempt.json"),
            command_identity={},
            failure_evidence=probe._FailureEvidenceAccumulator(
                requested_device="cuda:0"
            ),
        )
    assert caught.value.code == "qwen.parity.flash_attention_deterministic"


def test_probe_reasserts_runtime_config_before_model_marker_or_gpu_work(
    monkeypatch,
) -> None:
    probe = _load_probe_module()

    def reject_runtime_config(*_args, **_kwargs):
        raise ParityContractError(
            "injected runtime config drift",
            code="qwen.parity.runtime_config_drift",
            context={},
        )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("model, marker, and GPU work must not begin")

    monkeypatch.setattr(probe, "attest_v3_runtime_config", reject_runtime_config)
    monkeypatch.setattr(probe, "load_qwen_components", forbidden)
    monkeypatch.setattr(probe, "_publish_attempt_start_marker", forbidden)
    monkeypatch.setattr(probe, "_append_final_gpu_idle_check", forbidden)
    materials = SimpleNamespace(resolved=SimpleNamespace(config_dict={}))
    with pytest.raises(ParityContractError) as caught:
        probe._execute_real_probe_with_sampler(
            {
                "determinism": {"flash_attention_deterministic": "1"},
                "config_identity": {},
            },
            materials,
            device=torch.device("cpu"),
            gpu_idle_preflight={},
            device_sampler=object(),
            receipt_path=Path("receipt.json"),
            attempt_marker_path=Path("attempt.json"),
            command_identity={},
            failure_evidence=probe._FailureEvidenceAccumulator(
                requested_device="cuda:0"
            ),
        )
    assert caught.value.code == "qwen.parity.runtime_config_drift"


def test_fresh_pre_marker_config_attestation_rejects_source_drift(
    monkeypatch,
) -> None:
    probe = _load_probe_module()
    expected_source = {"path": "/tmp/config.yaml", "sha256": "a" * 64}
    observed_source = {"path": "/tmp/config.yaml", "sha256": "b" * 64}
    identity = {
        "entry_path": "/tmp/config.yaml",
        "fingerprint": "c" * 64,
        "schema_version": 1,
        "loader_version": "loader-v1",
        "resolved_config_sha256": probe.sha256_json({}),
        "sources": [expected_source],
        "runtime_config_attestation": {"status": "passed"},
    }
    resolved = SimpleNamespace(
        entry_config_path=Path("/tmp/config.yaml"),
        fingerprint="c" * 64,
        schema_version=1,
        loader_version="loader-v1",
        config_dict={},
        sources=(
            SimpleNamespace(
                to_artifact_dict=lambda: dict(observed_source),
            ),
        ),
    )
    monkeypatch.setattr(probe, "load_train_config", lambda _path: resolved)
    monkeypatch.setattr(
        probe,
        "attest_v3_runtime_config",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity drift must fail before semantic attestation")
        ),
    )

    with pytest.raises(ParityContractError) as caught:
        probe._attest_fresh_runtime_config_immediately_before_marker(
            {"config_identity": identity}
        )
    assert caught.value.code == "qwen.parity.runtime_config_identity_drift"


def test_attempt_marker_failure_after_publication_consumes_attempt_and_rejects_second(
    tmp_path: Path, monkeypatch
) -> None:
    probe = _load_probe_module()
    marker = tmp_path / "attempt.json"
    monkeypatch.setattr(probe, "finalize_attempt_marker", _fake_finalize_marker)
    monkeypatch.setattr(probe, "validate_attempt_marker", _fake_validate_marker)
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.record("execution", {"requested_device": "cuda:0"})
    kwargs = {
        "plan": {"plan_sha256": "a" * 64},
        "receipt_target": tmp_path / "receipt.json",
        "command_identity": {"argv_sha256": "b" * 64},
        "source_identity": {"repo": "frozen"},
        "concrete_inventory": {"inventory_sha256": "c" * 64},
        "failure_evidence": evidence,
    }

    with pytest.raises(RuntimeError, match="after marker"):
        probe._publish_attempt_start_marker(
            marker,
            **kwargs,
            inject_after=lambda: (_ for _ in ()).throw(RuntimeError("after marker")),
        )

    persisted = json.loads(marker.read_text(encoding="utf-8"))
    assert persisted["status"] == "attempt_started"
    assert persisted["plan_sha256"] == "a" * 64
    assert evidence.stage_reached == "attempt_started"
    assert evidence.fields["source_identity"] == {"repo": "frozen"}
    assert evidence.fields["trainable_inventory"] == {
        "concrete": {"inventory_sha256": "c" * 64}
    }
    assert evidence.fields["attempt_marker"]["path"] == str(marker)
    with pytest.raises(ParityContractError) as caught:
        probe._publish_attempt_start_marker(marker, **kwargs)
    assert caught.value.code == "qwen.parity.artifact_collision"
    assert json.loads(marker.read_text(encoding="utf-8")) == persisted


@pytest.mark.parametrize("collision_kind", ["identical", "foreign"])
def test_attempt_marker_collision_never_claims_attempt_ownership(
    tmp_path: Path,
    monkeypatch,
    collision_kind: str,
) -> None:
    probe = _load_probe_module()
    marker = tmp_path / "attempt.json"
    monkeypatch.setattr(probe, "finalize_attempt_marker", _fake_finalize_marker)
    monkeypatch.setattr(probe, "validate_attempt_marker", _fake_validate_marker)
    kwargs = {
        "plan": {"plan_sha256": "a" * 64},
        "receipt_target": tmp_path / "receipt.json",
        "command_identity": {"argv_sha256": "b" * 64},
        "source_identity": {"repo": "frozen"},
        "concrete_inventory": {"inventory_sha256": "c" * 64},
    }
    intended = _fake_finalize_marker(
        {
            "schema": probe.PARITY_ATTEMPT_MARKER_SCHEMA,
            "status": "attempt_started",
            "plan_sha256": "a" * 64,
            "receipt_target": str((tmp_path / "receipt.json").resolve()),
            "command_identity": {"argv_sha256": "b" * 64},
            "source_identity": {"repo": "frozen"},
            "concrete_trainable_inventory": {"inventory_sha256": "c" * 64},
        }
    )
    existing = intended if collision_kind == "identical" else {"foreign": True}
    probe.write_strict_json_atomic(marker, existing)
    evidence = probe._FailureEvidenceAccumulator(requested_device="cuda:0")
    evidence.record("execution", {"requested_device": "cuda:0"})

    with pytest.raises(ParityContractError) as caught:
        probe._publish_attempt_start_marker(
            marker,
            **kwargs,
            failure_evidence=evidence,
        )
    assert caught.value.code == "qwen.parity.artifact_collision"
    assert evidence.stage_reached == "initialized"
    assert "attempt_marker" not in evidence.fields
    assert "trainable_inventory" not in evidence.fields
    assert "source_identity" not in evidence.fields


def _fake_finalize_marker(payload):
    return {**payload, "marker_sha256": "d" * 64}


def _fake_validate_marker(marker, **_kwargs):
    return dict(marker)
