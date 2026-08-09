from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.research import run_static_dynamic_owner_interface_experiment as legacy
from scripts.research import seal_natural_boundary_pre_gpu_receipt as pre_gpu
from scripts.research.run_natural_boundary_routing_history_probe import build_residual_request
from scripts.research.run_s_primary_natural_boundary_gate import (
    EVENT_ID,
    GateTechnicalInvalid,
    LiveRuntimeBinding,
    SPrimaryNaturalBoundaryGate,
    _load_pre_gpu_identity,
    _attest_postload_cuda_device,
    _require_preload_cuda_visibility,
    build_natural_event_context,
    build_live_attention_mask_actuators,
    persist_failure_receipt,
    remove_exact_preseeded_opener,
    run_s_primary_natural_boundary_gate,
    sha256_file,
    sha256_json,
    _event_regions,
    _validate_frozen_event_geometry,
    _forced_math_model_call,
    _sdpa_backend_receipt,
    resolve_s_primary_binding,
)


OPEN = 1
REF_END = 2
BOX_START = 3
BOX_END = 9
COORD = 10
STOP = 0


def _contract() -> legacy.static.WrapperContract:
    return legacy.static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=OPEN,
        object_ref_end_token_id=REF_END,
        box_start_token_id=BOX_START,
        box_end_token_id=BOX_END,
        coordinate_token_start_id=COORD,
        eos_token_id=STOP,
    )


def _row(description: int = 20) -> tuple[int, ...]:
    return (OPEN, description, REF_END, BOX_START, COORD, COORD + 1, COORD + 2, COORD + 3, BOX_END)


class TinyModel(torch.nn.Module):
    def __init__(self, script: dict[tuple[int, ...], int], *, vocab: int = 64) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        # The live gate's all-layer attestor resolves ``layers`` and observes
        # the exact model-facing mask at each decoder-layer call.
        self.layers = torch.nn.ModuleList([TinyDecoderLayer()])
        self.script = dict(script)
        self.vocab = vocab
        self.calls: list[dict[str, object]] = []

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        values = tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
        self.calls.append(
            {
                "input_ids": values,
                "input_device": str(input_ids.device),
                "attention_device": None if attention_mask is None else str(attention_mask.device),
                "position_device": None if position_ids is None else str(position_ids.device),
                "grid_device": None if image_grid_thw is None else str(image_grid_thw.device),
                "use_cache": use_cache,
                "return_dict": return_dict,
                "logits_to_keep": logits_to_keep,
            }
        )
        _ = self.layers[0](input_ids.to(dtype=torch.float32), attention_mask=attention_mask)
        token = self.script.get(values, STOP)
        logits = torch.full((1, input_ids.shape[1], self.vocab), -40.0, device=input_ids.device)
        logits[:, :, token] = 20.0
        return SimpleNamespace(logits=logits)


class TinyDecoderLayer(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        del attention_mask
        return hidden_states


class TinySDPA23Layer(torch.nn.Module):
    """Deterministic one-layer SDPA surface for the real K14 gate seam."""

    layer_idx = 23
    num_key_value_groups = 1

    def forward(self, hidden_states: torch.Tensor, *, attention_mask: torch.Tensor) -> torch.Tensor:
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

        batch, sequence_length, hidden_size = hidden_states.shape
        query = hidden_states.reshape(batch, 1, sequence_length, hidden_size)
        output, _weights = ALL_ATTENTION_FUNCTIONS["sdpa"](
            self,
            query,
            query,
            query,
            attention_mask,
        )
        return output.transpose(1, 2).reshape(batch, sequence_length, hidden_size)


class TinySDPA23Model(torch.nn.Module):
    """Scripted logits while exercising Transformers' real SDPA registry."""

    def __init__(self, script: dict[tuple[int, ...], int], *, vocab: int = 64) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.layers = torch.nn.ModuleList([TinySDPA23Layer()])
        self.script = dict(script)
        self.vocab = vocab

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        use_cache: bool = False,
        return_dict: bool = True,
        logits_to_keep: int = 0,
    ) -> SimpleNamespace:
        del position_ids, image_grid_thw, use_cache, return_dict, logits_to_keep
        if attention_mask is None:
            raise AssertionError("K14 regression requires a model-facing additive mask")
        values = tuple(int(value) for value in input_ids[0].detach().cpu().tolist())
        # Equal Q/K scores keep the declared +2 dose observable instead of
        # letting a scripted token value underflow the selected-key mass.
        hidden = torch.ones((input_ids.shape[0], input_ids.shape[1], 8), device=input_ids.device)
        _ = self.layers[0](hidden, attention_mask=attention_mask)
        token = self.script.get(values, STOP)
        logits = torch.full((1, input_ids.shape[1], self.vocab), -40.0, device=input_ids.device)
        logits[:, :, token] = 20.0
        return SimpleNamespace(logits=logits)


class TinyRuntimeAdapter:
    def __init__(self, *, script: dict[tuple[int, ...], int], ignored_kwarg: bool = False) -> None:
        self.model = TinyModel(script)
        self.wrapper_contract = _contract()
        self.ignored_kwarg = ignored_kwarg
        self.exact_calls: list[tuple[int, ...]] = []
        self.runtime = SimpleNamespace(
            prompt_ids=torch.tensor([[50]], dtype=torch.long),
            image_grid_thw=torch.tensor([1, 2, 2], dtype=torch.long),
            h0={"rows": [list(_row(description=19))]},
        )

    @property
    def model_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def exact_model_inputs(
        self,
        runtime: object,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, object], torch.Tensor, str]:
        del runtime
        ids = input_ids.to(device=self.model_device, dtype=torch.long)
        self.exact_calls.append(tuple(int(value) for value in ids[0].tolist()))
        positions = torch.arange(3 * ids.shape[1], dtype=torch.long, device=ids.device).reshape(3, 1, ids.shape[1])
        payload: dict[str, object] = {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids) if attention_mask is None else attention_mask.to(ids.device),
            "position_ids": positions,
            "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long, device=ids.device),
            "use_cache": False,
            "return_dict": True,
            "logits_to_keep": 0,
        }
        if self.ignored_kwarg:
            payload["ignored_kwarg"] = 1
        return payload, positions, sha256_json({"position_ids": positions.tolist(), "length": ids.shape[1]})

    def parse_row(self, tokens: list[int], runtime: object, *, row_index: int) -> dict[str, object]:
        del runtime, row_index
        return {
            "parse_status": "accepted",
            "owner_match": {
                "status": "unique",
                "owner_id": "gt:5001:15",
                "physical_match": True,
                "source_specific": True,
            },
            "row_token_ids": list(tokens),
        }


class TinySDPA23RuntimeAdapter(TinyRuntimeAdapter):
    def __init__(self, *, script: dict[tuple[int, ...], int], ignored_kwarg: bool = False) -> None:
        super().__init__(script=script, ignored_kwarg=ignored_kwarg)
        self.model = TinySDPA23Model(script)


def _script(prefix: tuple[int, ...], rows: list[tuple[int, ...]], *, stop_after: bool = True) -> dict[tuple[int, ...], int]:
    script: dict[tuple[int, ...], int] = {}
    current = prefix
    for row in rows:
        for token in row:
            script[current] = token
            current = current + (token,)
    if stop_after:
        script[current] = STOP
    return script


def _binding(
    *,
    rows: list[tuple[int, ...]] | None = None,
    ignored_kwarg: bool = False,
    adapter_type: type[TinyRuntimeAdapter] = TinyRuntimeAdapter,
) -> LiveRuntimeBinding:
    history = _row(description=19)
    seeded = (50, *history, OPEN)
    rows = rows or [_row()]
    script = _script(seeded[:-1], rows)
    adapter = adapter_type(script=script, ignored_kwarg=ignored_kwarg)
    seeded_context = SimpleNamespace(
        prefix_ids=torch.tensor([seeded], dtype=torch.long),
        latest_row_ids=list(history),
        covered_owner_ids=("gt:5001:2",),
        runtime=adapter.runtime,
    )
    adapter.runtime.wrapper_contract = adapter.wrapper_contract
    return LiveRuntimeBinding(
        adapter=adapter,
        runtime=adapter.runtime,
        event={"gt_owner_id": EVENT_ID, "image_id": 5001},
        seeded_context=seeded_context,
        identity={"fixture": "tiny-s", "checkpoint": "S"},
    )


def _launch_event(*, competitor: object = None, include_competitor: bool = False) -> dict[str, object]:
    regions: dict[str, object] = {
        "a_exclusive": [0],
        "b_exclusive": [1],
        "background": [2],
        "shared_core": [3],
    }
    if include_competitor:
        regions["same_class_competitor"] = competitor
    return {
        "gt_owner_id": EVENT_ID,
        "image_id": 5001,
        "source_panel_object_index": 15,
        "image_cell_regions": regions,
        "geometry_by_checkpoint": {
            "S": {
                "launch_eligible": True,
                "image_plan_identity": {"cell_count": 4},
            }
        },
    }


def _bind_launch_event(event: dict[str, object]) -> LiveRuntimeBinding:
    binding = _binding()
    binding.event = event
    binding.adapter.model.config = SimpleNamespace(num_hidden_layers=1, num_attention_heads=1)
    binding.runtime.image_span = legacy.static.ImageSpan(
        image_token_id=99,
        grid_thw=(1, 2, 2),
        merge_size=1,
        absolute_positions=(1, 2, 3, 4),
        grid_indices=((0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1)),
    )
    return binding


def test_exact_one_token_removal_is_fail_closed() -> None:
    boundary = remove_exact_preseeded_opener((50, 9, OPEN), opener_token_id=OPEN)
    assert boundary.natural_prefix_token_ids == (50, 9)
    assert boundary.removed_token_count == 1
    with pytest.raises(GateTechnicalInvalid, match="end"):
        remove_exact_preseeded_opener((50, 9), opener_token_id=OPEN)
    with pytest.raises(GateTechnicalInvalid, match="still ends"):
        remove_exact_preseeded_opener((50, OPEN, OPEN), opener_token_id=OPEN)


def test_missing_same_class_competitor_is_k13_not_applicable_but_other_geometry_stays_required() -> None:
    event = _launch_event()
    assert _event_regions(event)["same_class_competitor"] == ()
    _validate_frozen_event_geometry(event)

    missing_background = _launch_event()
    regions = missing_background["image_cell_regions"]
    assert isinstance(regions, dict)
    regions.pop("background")
    with pytest.raises(GateTechnicalInvalid, match="background"):
        _validate_frozen_event_geometry(missing_background)

    binding = _bind_launch_event(event)
    gate = SPrimaryNaturalBoundaryGate.from_binding(binding, max_rows=1)
    factories = build_live_attention_mask_actuators(binding, gate.context)
    result = gate.run_matrix(
        arms=("K13",),
        attention_mask_actuators={"K13": factories["K13"]},
    )
    arm = result["arms"]["K13"]
    assert arm["rows"][0]["status"] == "closure"
    receipts = arm["runtime_scalar_receipts"]
    assert receipts
    assert all(
        receipt["layer_consumption_attestation"]["passed"] is True
        and receipt["layer_consumption_attestation"]["status"] == "not_applicable"
        for receipt in receipts
    )


def test_live_gate_rebuilds_growing_multimodal_inputs_on_actual_device_and_no_opener_seed() -> None:
    binding = _binding()
    gate = SPrimaryNaturalBoundaryGate.from_binding(binding, max_rows=1)
    result = gate.run_matrix(arms=("N00",))
    arm = result["arms"]["N00"]
    row = arm["rows"][0]
    assert row["status"] == "closure"
    assert row["opener_generated_by_model"] is True
    assert row["opener_injected"] is False
    assert arm["synthetic_opener_injections"] == 0
    assert row["initial_prefix_last_token_id"] == BOX_END
    assert row["opener_token_id"] == OPEN
    assert row["first_generated_token_id"] == OPEN
    expected_lengths = list(range(len(gate.context.prefix_token_ids), len(gate.context.prefix_token_ids) + len(_row())))
    assert [len(prefix) for prefix in binding.adapter.exact_calls] == expected_lengths
    assert all(call["use_cache"] is False for call in binding.adapter.model.calls)
    assert all(call["input_device"] == "cpu" for call in binding.adapter.model.calls)
    assert all(call["position_device"] == "cpu" for call in binding.adapter.model.calls)
    assert all(call["grid_device"] == "cpu" for call in binding.adapter.model.calls)
    assert arm["full_logit_parity"]["status"] == "reference_captured"
    assert all(
        receipt["sdpa_backend"] == _sdpa_backend_receipt()
        for receipt in arm["runtime_scalar_receipts"]
    )


def test_forced_math_backend_wraps_every_strict_scalar_call(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch.nn.attention as attention_api

    observed: list[object] = []

    def fake_sdpa_kernel(backend: object):
        observed.append(backend)
        return nullcontext()

    monkeypatch.setattr(attention_api, "sdpa_kernel", fake_sdpa_kernel)
    monkeypatch.setattr(
        "scripts.research.run_s_primary_natural_boundary_gate._strict_model_call",
        lambda model, payload: (model, payload),
    )
    result = _forced_math_model_call("tiny-model", {"input_ids": torch.tensor([[1]])})
    assert result[0] == "tiny-model"
    assert observed == [attention_api.SDPBackend.MATH]


def test_n01_full_vocab_parity_and_attention_residual_seams() -> None:
    binding = _binding()
    gate = SPrimaryNaturalBoundaryGate.from_binding(binding, max_rows=1)
    residual_calls: list[str] = []
    attention_lengths: list[int] = []

    def residual(_model: object, *, request: object, **_kwargs: object):
        residual_calls.append(request.arm_id)
        return nullcontext()

    def attention(_context: object, *, input_ids: torch.Tensor, arm_id: str, **_kwargs: object):
        attention_lengths.append(int(input_ids.shape[-1]))
        return {
            "attention_mask": torch.ones((1, 1, input_ids.shape[-1], input_ids.shape[-1]), dtype=torch.bool),
            "receipt": {"arm": arm_id},
        }

    result = gate.run_matrix(
        arms=("N00", "N01"),
        residual_actuator=residual,
        residual_requests={"N01": build_residual_request("N01", positions=(1,))},
        attention_mask_actuators={"N01": attention},
    )
    parity = result["arms"]["N01"]["full_logit_parity"]
    assert parity["status"] == "measured"
    assert parity["passed"] is True
    assert parity["per_forward_max_abs_delta"] == 0.0
    assert residual_calls == ["N01"] * len(_row())
    assert attention_lengths == [len(gate.context.prefix_token_ids) + i for i in range(len(_row()))]


def test_keyword_only_attention_factory_seam_receives_frozen_arm() -> None:
    binding = _binding()
    gate = SPrimaryNaturalBoundaryGate.from_binding(binding, max_rows=1)
    observed: list[tuple[str, int, str]] = []

    def factory(*, sequence_length: int, query_position: int, device: torch.device, arm_id: str, context: object, **_kwargs: object):
        del context
        observed.append((arm_id, sequence_length, str(device)))
        assert query_position == sequence_length - 1
        attestation = {"passed": True, "status": "fake_layer_attestor", "layer_indices": [0]}
        return {
            "attention_mask": torch.ones((1, 1, sequence_length, sequence_length), dtype=torch.bool),
            "receipt": {
                "arm": arm_id,
                "layer_consumption_attestation": attestation,
                "all_layer_consumption_attestation": attestation,
            },
        }

    result = gate.run_matrix(
        arms=("N00", "H00"),
        attention_mask_actuators={"H00": factory},
    )
    assert result["arms"]["H00"]["full_logit_parity"]["passed"] is True
    assert observed and all(arm == "H00" for arm, _length, _device in observed)
    actual_attestations = [
        receipt["layer_consumption_attestation"]
        for receipt in result["arms"]["H00"]["runtime_scalar_receipts"]
    ]
    assert actual_attestations and all(
        attestation["passed"] is True
        and attestation["layer_indices"] == [0]
        and "expected_mask_sha256" in attestation
        for attestation in actual_attestations
    )


def test_ignored_model_kwargs_are_rejected_not_filtered() -> None:
    binding = _binding(ignored_kwarg=True)
    gate = SPrimaryNaturalBoundaryGate.from_binding(binding, max_rows=1)
    with pytest.raises(GateTechnicalInvalid, match="ignored"):
        gate.run_matrix(arms=("N00",))
    assert binding.adapter.model.calls == []


def test_deterministic_success_and_failure_receipts(tmp_path: Path) -> None:
    first = run_s_primary_natural_boundary_gate(binding=_binding(), arms=("N00",), output_root=tmp_path / "a")
    second = run_s_primary_natural_boundary_gate(binding=_binding(), arms=("N00",), output_root=tmp_path / "b")
    assert first["result"]["result_sha256"] == second["result"]["result_sha256"]
    assert (tmp_path / "a" / "result.json").read_bytes() == (tmp_path / "b" / "result.json").read_bytes()

    first_failure = persist_failure_receipt(tmp_path / "failure-a", "same technical failure")
    second_failure = persist_failure_receipt(tmp_path / "failure-b", "same technical failure")
    assert first_failure == second_failure
    assert (tmp_path / "failure-a" / "failure.stderr").read_bytes() == b"same technical failure"


def test_k14_factory_gate_persists_canonical_receipts_without_python_callbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pytest.importorskip("transformers")
    from scripts.research import natural_boundary_attention_actuators as attention

    def run_once(root: Path) -> dict[str, object]:
        binding = _binding(adapter_type=TinySDPA23RuntimeAdapter)
        factory = attention.build_scalar_step_factory(
            "K14T",
            image_key_positions=(1, 2, 3),
            b_exclusive_positions=(2,),
            layer_count=1,
            head_count=1,
            dtype=torch.float32,
        )
        callback = attention.make_natural_runner_callback(factory)
        monkeypatch.setattr(
            "scripts.research.run_s_primary_natural_boundary_gate._resolve_k14_reference_positions",
            lambda _binding: {"K14T": (2,)},
        )
        return run_s_primary_natural_boundary_gate(
            binding=binding,
            arms=("K14T",),
            attention_mask_actuators={"K14T": callback},
            output_root=root,
        )

    first = run_once(tmp_path / "first")
    second = run_once(tmp_path / "second")
    assert first["result"]["result_sha256"] == second["result"]["result_sha256"]
    for name in ("result.json", "runtime_identity.json", "terminal_summary.json"):
        assert (tmp_path / "first" / name).read_bytes() == (tmp_path / "second" / name).read_bytes()

    persisted_path = tmp_path / "first" / "result.json"
    persisted = persisted_path.read_text(encoding="utf-8")
    assert "FixedDoseScoreBias" not in persisted
    persisted_result = json.loads(persisted)
    persisted_hash = persisted_result.pop("result_sha256")
    assert persisted_hash == sha256_json(persisted_result)
    persisted_identity = json.loads(
        (tmp_path / "first" / "runtime_identity.json").read_text(encoding="utf-8")
    )
    identity_hash = persisted_identity.pop("identity_sha256")
    assert identity_hash == sha256_json(persisted_identity)
    terminal = json.loads(
        (tmp_path / "first" / "terminal_summary.json").read_text(encoding="utf-8")
    )
    assert terminal["result_sha256"] == persisted_hash
    result = first["result"]
    arm = result["arms"]["K14T"]
    receipts = arm["scalar_receipts"]
    runtime_receipts = arm["runtime_scalar_receipts"]
    assert receipts and len(receipts) == len(runtime_receipts)
    for scalar, runtime in zip(receipts, runtime_receipts, strict=True):
        receipt = scalar["attention_mask"]["receipt"]
        assert set(receipt["callback_metadata"]) == {"actuator_id", "protocol"}
        assert receipt["callback_metadata"]["actuator_id"] == "K14T"
        complete = runtime["attention_actuation_receipt"]
        assert complete["arm_id"] == "K14T"
        assert complete["selected_positions"] == [2]
        assert receipt["attention_mask"]["selected_positions"] == [2]
        assert receipt["attention_mask"]["exact_scope"] is True
        assert complete["block23_mass_requirement"]["passed"] is True
        assert runtime["layer_consumption_attestation"]["passed"] is True
    assert arm["rows"][0]["status"] == "closure"
    assert arm["owner_bookkeeping"]["raw_endpoint_owner_ids"] == ["gt:5001:15"]


def test_pre_gpu_loader_replays_full_sealer_and_failure_receipts_are_write_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    source_hash = sha256_file(source)
    document: dict[str, object] = {
        "unit_id": "2026-08-06-natural-boundary-routing-history-replication",
        "event_binding": {"event_id": EVENT_ID, "checkpoint": "S"},
        "model_identity": {"checkpoint": "S"},
        "source_files": {"runner": {"path": str(source), "sha256": source_hash}},
        "code_identity": {"sha256": {"runner": source_hash}},
    }
    document["self_sha256"] = sha256_json({key: value for key, value in document.items()})
    receipt_path = tmp_path / "pre-gpu.json"
    receipt_path.write_text(json.dumps(document), encoding="utf-8")
    calls: list[tuple[object, Path]] = []

    def replay(receipt: object, *, receipt_path: str | Path) -> None:
        calls.append((receipt, Path(receipt_path)))

    monkeypatch.setattr(pre_gpu, "validate_receipt", replay)
    identity = _load_pre_gpu_identity(receipt_path)
    assert calls == [(document, receipt_path.resolve())]
    assert identity["code_hashes"] == {"runner": source_hash}

    first = persist_failure_receipt(tmp_path / "failure", "first failure")
    with pytest.raises(FileExistsError, match="collision"):
        persist_failure_receipt(tmp_path / "failure", "different failure")
    assert (tmp_path / "failure" / "failure.json").read_text(encoding="utf-8").find(first["sha256"]) >= 0

    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "failure.stderr").write_text("partial\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="collision"):
        persist_failure_receipt(partial, "must not append")
    assert (partial / "failure.stderr").read_text(encoding="utf-8") == "partial\n"


def _gate_bound_receipt(tmp_path: Path, gate_root: Path) -> tuple[Path, dict[str, object]]:
    """Build the small v4 identity surface used by producer/consumer tests."""

    source = tmp_path / "source.py"
    source.write_text("source\n", encoding="utf-8")
    source_hash = sha256_file(source)
    root = str(gate_root.resolve())
    document: dict[str, object] = {
        "unit_id": "2026-08-06-natural-boundary-routing-history-replication",
        "event_binding": {"event_id": EVENT_ID, "checkpoint": "S"},
        "model_identity": {"checkpoint": "S"},
        "source_files": {"runner": {"path": str(source), "sha256": source_hash}},
        "code_identity": {"sha256": {"runner": source_hash}},
        "gate_output_root": root,
        "authorized_gate_output_root": root,
        "gate_output_binding": {
            "path": root,
            "suffix": "s-gt5001-live-gate-v3",
            "status": "reserved_absent_pre_gpu",
            "parent": str(gate_root.parent.resolve()),
            "parent_is_symlink": False,
        },
    }
    document["self_sha256"] = sha256_json(document)
    receipt_path = tmp_path / "pre-gpu-v4.json"
    receipt_path.write_text(json.dumps(document), encoding="utf-8")
    return receipt_path, document


def test_v4_gate_output_binding_exact_match_is_consumed_before_live_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_root = tmp_path / "s-gt5001-live-gate-v3"
    receipt_path, document = _gate_bound_receipt(tmp_path, gate_root)
    calls: list[tuple[object, Path]] = []

    def replay(receipt: object, *, receipt_path: str | Path) -> None:
        calls.append((receipt, Path(receipt_path)))

    monkeypatch.setattr(pre_gpu, "validate_receipt", replay)
    identity = _load_pre_gpu_identity(receipt_path, output_root=gate_root)
    assert calls == [(document, receipt_path.resolve())]
    assert identity["authorized_gate_output_root"] == str(gate_root.resolve())
    assert identity["gate_output_root"] == str(gate_root.resolve())
    assert not gate_root.exists()


def test_wrong_gate_output_root_fails_before_checkpoint_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_root = tmp_path / "s-gt5001-live-gate-v3"
    receipt_path, _document = _gate_bound_receipt(tmp_path, gate_root)
    monkeypatch.setattr(pre_gpu, "validate_receipt", lambda *_args, **_kwargs: None)
    preload_calls: list[str] = []
    monkeypatch.setattr(
        "scripts.research.run_s_primary_natural_boundary_gate._require_preload_cuda_visibility",
        lambda: preload_calls.append("preload") or {},
    )
    with pytest.raises(GateTechnicalInvalid, match="does not exactly match"):
        resolve_s_primary_binding(
            pre_gpu_receipt=receipt_path,
            output_root=tmp_path / "wrong-s-gt5001-live-gate-v3",
        )
    assert preload_calls == []


def test_reused_gate_output_root_fails_before_checkpoint_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_root = tmp_path / "s-gt5001-live-gate-v3"
    gate_root.mkdir()
    receipt_path, _document = _gate_bound_receipt(tmp_path, gate_root)
    monkeypatch.setattr(pre_gpu, "validate_receipt", lambda *_args, **_kwargs: None)
    with pytest.raises(GateTechnicalInvalid, match="absent and unused"):
        resolve_s_primary_binding(pre_gpu_receipt=receipt_path, output_root=gate_root)


def test_symlink_gate_output_root_fails_before_checkpoint_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    gate_root = tmp_path / "s-gt5001-live-gate-v3"
    receipt_path, _document = _gate_bound_receipt(tmp_path, gate_root)
    redirected = tmp_path / "redirected"
    redirected.mkdir()
    gate_root.symlink_to(redirected, target_is_directory=True)
    monkeypatch.setattr(pre_gpu, "validate_receipt", lambda *_args, **_kwargs: None)
    with pytest.raises(GateTechnicalInvalid, match="non-symlink"):
        resolve_s_primary_binding(pre_gpu_receipt=receipt_path, output_root=gate_root)


def test_programmatic_binding_none_threads_output_root_to_pre_gpu_resolver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requested = tmp_path / "s-gt5001-live-gate-v3"
    observed: dict[str, object] = {}

    def fake_resolve(**kwargs: object) -> LiveRuntimeBinding:
        observed.update(kwargs)
        return _binding()

    monkeypatch.setattr(
        "scripts.research.run_s_primary_natural_boundary_gate.resolve_s_primary_binding",
        fake_resolve,
    )
    run_s_primary_natural_boundary_gate(binding=None, arms=("N00",), output_root=requested)
    assert observed["output_root"] == requested
    assert (requested / "result.json").is_file()


def test_live_cuda_binding_requires_explicit_single_token_and_postload_cuda0(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(GateTechnicalInvalid, match="explicitly set"):
        _require_preload_cuda_visibility()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(GateTechnicalInvalid, match="exactly one"):
        _require_preload_cuda_visibility()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-abc")
    with pytest.raises(GateTechnicalInvalid, match="numeric"):
        _require_preload_cuda_visibility()
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    assert _require_preload_cuda_visibility()["selected_physical_device"] == "3"

    binding = _binding()
    with pytest.raises(GateTechnicalInvalid, match="logical cuda:0"):
        _attest_postload_cuda_device(binding)


def test_postload_cuda_attestation_rejects_later_parameter_or_buffer_drift() -> None:
    class FakeTensor:
        def __init__(self, device: str, dtype: torch.dtype = torch.float32) -> None:
            self.device = torch.device(device)
            self.dtype = dtype

        def numel(self) -> int:
            return 1

        def is_floating_point(self) -> bool:
            return bool(self.dtype.is_floating_point)

    class FakeModule:
        def __init__(
            self,
            parameter_devices: list[str],
            buffer_devices: list[str],
            *,
            parameter_dtypes: list[torch.dtype] | None = None,
            buffer_dtypes: list[torch.dtype] | None = None,
        ) -> None:
            parameter_dtypes = parameter_dtypes or [torch.float32] * len(parameter_devices)
            buffer_dtypes = buffer_dtypes or [torch.float32] * len(buffer_devices)
            self._parameters = [FakeTensor(device, dtype) for device, dtype in zip(parameter_devices, parameter_dtypes, strict=True)]
            self._buffers = {f"buffer_{index}": FakeTensor(device, dtype) for index, (device, dtype) in enumerate(zip(buffer_devices, buffer_dtypes, strict=True))}
            self._non_persistent_buffers_set: set[str] = set()

        def parameters(self) -> object:
            return iter(self._parameters)

        def named_modules(self) -> object:
            return iter((("", self),))

    def binding(
        parameter_devices: list[str],
        buffer_devices: list[str],
        *,
        parameter_dtypes: list[torch.dtype] | None = None,
        buffer_dtypes: list[torch.dtype] | None = None,
    ) -> object:
        model = FakeModule(
            parameter_devices,
            buffer_devices,
            parameter_dtypes=parameter_dtypes,
            buffer_dtypes=buffer_dtypes,
        )
        adapter = SimpleNamespace(model=model, model_device=torch.device("cuda:0"))
        return SimpleNamespace(model=model, adapter=adapter, device=torch.device("cuda:0"))

    passed = _attest_postload_cuda_device(binding(["cuda:0", "cuda:0"], ["cuda:0"]))
    assert passed["parameter_tensor_count"] == 2
    assert passed["persistent_buffer_tensor_count"] == 1
    assert passed["all_tensor_device_set"] == ["cuda:0"]
    assert passed["all_floating_dtype_set"] == ["torch.float32"]
    with pytest.raises(GateTechnicalInvalid, match="forbidden devices"):
        _attest_postload_cuda_device(binding(["cuda:0", "cpu"], ["cuda:0"]))
    with pytest.raises(GateTechnicalInvalid, match="forbidden devices"):
        _attest_postload_cuda_device(binding(["cuda:0"], ["meta"]))
    with pytest.raises(GateTechnicalInvalid, match="forbidden dtypes"):
        _attest_postload_cuda_device(
            binding(["cuda:0"], ["cuda:0"], parameter_dtypes=[torch.float16])
        )
    with pytest.raises(GateTechnicalInvalid, match="forbidden dtypes"):
        _attest_postload_cuda_device(
            binding(["cuda:0"], ["cuda:0"], buffer_dtypes=[torch.bfloat16])
        )
    integer_buffer = _attest_postload_cuda_device(
        binding(["cuda:0"], ["cuda:0"], buffer_dtypes=[torch.int64])
    )
    assert integer_buffer["nonfloating_persistent_buffer_dtype_set"] == ["torch.int64"]


def test_endpoint_bookkeeping_is_raw_and_matrix_delta_requires_own_baseline() -> None:
    binding = _binding()
    gate = SPrimaryNaturalBoundaryGate.from_binding(binding, max_rows=1)
    owner_by_row = {0: "gt:5001:2", 1: "gt:5001:2", 2: "gt:5001:99"}

    def parse_row(tokens: list[int], runtime: object, *, row_index: int) -> dict[str, object]:
        del tokens, runtime
        return {
            "parse_status": "accepted",
            "owner_match": {
                "status": "unique",
                "owner_id": owner_by_row[row_index],
                "physical_match": True,
                "source_specific": True,
            },
        }

    binding.adapter.parse_row = parse_row
    rows = [
        {
            "status": "closure",
            "row_index": 0,
            "prefix_before_row_token_ids": [50],
            "token_ids": list(_row()),
        },
        {
            "status": "closure",
            "row_index": 1,
            "prefix_before_row_token_ids": [50, *_row()],
            "token_ids": list(_row()),
        },
        {
            "status": "closure",
            "row_index": 2,
            "prefix_before_row_token_ids": [50, *_row(), *_row()],
            "token_ids": list(_row()),
        },
    ]
    normalized = {"rows": rows, "terminal_reason": "closure"}
    gate._normalize_endpoint_receipts(normalized)
    bookkeeping = normalized["owner_bookkeeping"]
    assert bookkeeping["raw_endpoint_owner_ids"] == ["gt:5001:2", "gt:5001:99"]
    assert bookkeeping["covered_repeat_owner_ids"] == ["gt:5001:2"]
    assert normalized["rows"][0]["owner_bookkeeping"]["covered_repeat"] is True
    assert normalized["rows"][0]["owner_bookkeeping"]["duplicate"] is False
    assert normalized["rows"][1]["owner_bookkeeping"]["covered_repeat"] is True
    assert normalized["rows"][1]["owner_bookkeeping"]["duplicate"] is True
    assert "G" not in bookkeeping and "K" not in bookkeeping and "L" not in bookkeeping
    assert "net" not in bookkeeping

    baseline = {
        "owner_bookkeeping": {
            "raw_endpoint_owner_ids": ["gt:5001:2"],
            "covered_repeat_owner_ids": ["gt:5001:2"],
            "row_entry": {
                "admission_mode": "pre_opener_natural",
                "first_generated_token_id": OPEN,
                "opener_generated_by_model": True,
                "opener_injected": False,
                "row_started": True,
            },
            "parse": {"unmatched_rows": 0, "duplicate_rows": 0, "invalid_rows": 0, "malformed_rows": 0},
            "stop": {"stopped": False, "stop_reason": "closure"},
        }
    }
    intervention = {
        "owner_bookkeeping": {
            "raw_endpoint_owner_ids": ["gt:5001:99"],
            "covered_repeat_owner_ids": [],
            "row_entry": {
                "admission_mode": "pre_opener_natural",
                "first_generated_token_id": STOP,
                "opener_generated_by_model": False,
                "opener_injected": False,
                "row_started": False,
            },
            "parse": {"unmatched_rows": 1, "duplicate_rows": 0, "invalid_rows": 1, "malformed_rows": 0},
            "stop": {"stopped": True, "stop_reason": "native_stop"},
        }
    }
    contrast = gate._build_matrix_contrast(
        intervention_arm="K10", baseline_arm="K01", outputs={"K01": baseline, "K10": intervention}
    )
    assert contrast["status"] == "measured"
    assert contrast["endpoint_delta"]["raw_endpoint_owner_ids_added"] == ["gt:5001:99"]
    assert contrast["endpoint_delta"]["raw_endpoint_owner_ids_removed"] == ["gt:5001:2"]
    missing = gate._build_matrix_contrast(
        intervention_arm="K10", baseline_arm="K01", outputs={"K10": intervention}
    )
    assert missing["status"] == "unqualified"
    assert missing["endpoint_delta"] is None


def test_natural_context_preserves_exact_prefix_identity() -> None:
    binding = _binding()
    context, boundary, identity = build_natural_event_context(binding, max_rows=1)
    assert context.prefix_token_ids == boundary.natural_prefix_token_ids
    assert identity["prefix"]["removed_token_count"] == 1
    assert identity["prefix"]["natural_prefix_sha256"] == context.prefix_sha256
