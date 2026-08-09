from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from types import SimpleNamespace

from scripts.research.run_static_dynamic_owner_interface_experiment import (
    FakeRuntimeAdapter,
    OrchestrationError,
    OwnerInterfaceOrchestrator,
    _capture_dynamic_replacement,
    _build_token_mass_contract,
    _build_source_derived_owner_mapping,
    _make_event_context,
    _geometry_actuator_gate,
    _native_arithmetic_position_view,
    _endpoint_row_evidence,
    _score_active_endpoint_candidates,
    _natural_segment_scores,
    _segment_token_indices,
    _verified_owner_regions,
    _row_from_owner,
    _run_gradient_audit,
    _build_independent_p4_candidate_inputs,
    _index_trace_rows_by_image,
    _ineligible_event_result,
    _partition_ineligible_materialization_shard,
    _validate_active_event_receipts,
    _select_event_panel_rows,
    source_specific_owner_match,
    _release_row_with_static,
    _release_dynamic_row,
    _release_dynamic_horizon,
    sha256_token_ids,
    _teacher_forced_row_offsets,
    build_parser,
    parse_event_shard,
)
from scripts.research import run_static_post_llm_image_field_probe as static
from scripts.research import run_dynamic_history_and_crossover_probe as dynamic
from scripts.research import run_static_dynamic_owner_interface_experiment as runner


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    config = tmp_path / "resolved.yaml"
    config.write_text("checkpoint: fake\n", encoding="utf-8")
    panel = tmp_path / "panel.jsonl"
    panel_payload = json.dumps({"image_id": 1, "objects": []}) + "\n"
    panel.write_text(panel_payload, encoding="utf-8")
    source_panel = tmp_path / "source-panel.jsonl"
    source_panel.write_text(panel_payload, encoding="utf-8")

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    ledger = tmp_path / "h0-ledger.json"
    exact_prefix_token_ids: list[int] = []
    exact_prefix_sha256 = hashlib.sha256(b"[]").hexdigest()
    ledger.write_text(
        json.dumps(
            {
                "unit_id": "2026-08-05-static-dynamic-owner-interface-crossover",
                "checkpoint": "S",
                "records": [
                    {
                        "image_id": 1,
                        "gt_owner_id": "gt:1:0",
                        "natural_boundary": 0,
                        "covered_owner_ids": [],
                        "latest_covered_owner_id": None,
                        "exact_prefix_token_ids": exact_prefix_token_ids,
                        "exact_prefix_sha256": exact_prefix_sha256,
                        "valid_prediction_count": 0,
                        "generated_row_boundaries": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps(
            {
                "unit_id": "2026-08-05-static-dynamic-owner-interface-crossover",
                "sources": {
                    "derived_panel": {"path": str(panel), "sha256": digest(panel)},
                    "source_panel": {"path": str(source_panel), "sha256": digest(source_panel)},
                    "h0_ledgers": [{"path": str(ledger), "sha256": digest(ledger)}],
                },
                "events": [{"gt_owner_id": "gt:1:0", "image_id": 1}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "cohort.manifest.json").write_text(
        json.dumps(
            {
                "unit_id": "2026-08-05-static-dynamic-owner-interface-crossover",
                "source_hashes": {
                    "derived_panel": digest(panel),
                    "source_panel": digest(source_panel),
                    "h0_ledgers": [digest(ledger)],
                },
            }
        ),
        encoding="utf-8",
    )
    return config, panel, cohort


def _actual_shaped_h0_record(*, checkpoint: str, owner_id: str = "gt:1:0") -> dict[str, object]:
    is_commit = checkpoint == "A"
    closure_id = 151669 if is_commit else 151649
    closure_text = "<|commit|>" if is_commit else "<|box_end|>"
    closure_offset = 1 if is_commit else 0
    boundaries = []
    for prediction_index, (generated_order, row_start, span_end) in enumerate(
        ((0, 0, 8), (2, 19, 27))
    ):
        closure_step = span_end + closure_offset
        matched_owner = "gt:1:9" if prediction_index == 0 else None
        boundaries.append({
            "prediction_index": prediction_index,
            "generated_order": generated_order,
            "object_span_id": f"image-1:span-{prediction_index}",
            "row_start_step": row_start,
            "span_end_step": span_end,
            "closure_step": closure_step,
            "closure_token_id": closure_id,
            "closure_token_text": closure_text,
            "commit_step": closure_step if is_commit else None,
            "match_status": "tp" if matched_owner is not None else "unmatched",
            "gt_owner_id": matched_owner,
        })
    return {
        "image_id": 1,
        "gt_owner_id": owner_id,
        "natural_boundary": 0,
        "covered_owner_ids": [],
        "latest_covered_owner_id": None,
        "exact_prefix_token_ids": [],
        "exact_prefix_sha256": sha256_token_ids([]),
        "strict_complete_row": False,
        "native_tp": False,
        "native_fn": True,
        "parse_status": "accepted_with_drops",
        "valid_prediction_count": len(boundaries),
        "generated_row_boundaries": boundaries,
    }


def _load_test_h0_ledger(
    tmp_path: Path,
    *,
    checkpoint: str,
    records: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    ledger = tmp_path / "ledger.json"
    ledger.write_text(
        json.dumps({
            "unit_id": runner.UNIT_ID,
            "checkpoint": checkpoint,
            "records": records,
        }) + "\n",
        encoding="utf-8",
    )
    digest = runner.sha256_file(ledger)
    loaded, _identity = runner._load_h0_ledger_records(
        sources={"h0_ledgers": [{"path": str(ledger), "sha256": digest}]},
        manifest_sources={"h0_ledgers": [digest]},
        checkpoint=checkpoint,
    )
    return loaded


def test_h0_loader_preserves_authoritative_generated_row_provenance(tmp_path: Path) -> None:
    source = _actual_shaped_h0_record(checkpoint="A")
    loaded = _load_test_h0_ledger(tmp_path, checkpoint="A", records=[source])
    record = loaded["1"]["gt:1:0"]
    assert record["valid_prediction_count"] == 2
    assert record["generated_row_boundaries"] == source["generated_row_boundaries"]
    assert record["generated_row_boundaries"][1]["generated_order"] == 2


@pytest.mark.parametrize(
    "tamper",
    (
        "count",
        "order",
        "steps",
        "closure",
        "commit",
        "match_owner",
        "prediction_index",
    ),
)
def test_h0_loader_rejects_tampered_generated_row_provenance(
    tmp_path: Path,
    tamper: str,
) -> None:
    record = _actual_shaped_h0_record(checkpoint="A")
    boundaries = record["generated_row_boundaries"]
    assert isinstance(boundaries, list)
    if tamper == "count":
        record["valid_prediction_count"] = 3
    elif tamper == "order":
        boundaries[1]["generated_order"] = 0
    elif tamper == "steps":
        boundaries[1]["row_start_step"] = boundaries[0]["closure_step"]
    elif tamper == "closure":
        boundaries[0]["closure_token_id"] = 151649
    elif tamper == "commit":
        boundaries[0]["commit_step"] = boundaries[0]["closure_step"] + 1
    elif tamper == "match_owner":
        boundaries[1]["gt_owner_id"] = "gt:1:8"
    else:
        boundaries[1]["prediction_index"] = 0
    with pytest.raises(OrchestrationError, match="H0 ledger record"):
        _load_test_h0_ledger(tmp_path, checkpoint="A", records=[record])


def test_h0_loader_rejects_cross_record_generated_row_provenance_drift(
    tmp_path: Path,
) -> None:
    first = _actual_shaped_h0_record(checkpoint="S", owner_id="gt:1:0")
    second = json.loads(json.dumps(first))
    second["gt_owner_id"] = "gt:1:1"
    second["generated_row_boundaries"][1]["generated_order"] = 3
    with pytest.raises(OrchestrationError, match="disagree on generated-row provenance"):
        _load_test_h0_ledger(
            tmp_path,
            checkpoint="S",
            records=[first, second],
        )


def test_fake_adapter_calls_parser_static_dynamic_gradient_and_persistent_receipts(tmp_path: Path) -> None:
    config, panel, cohort = _fixture(tmp_path)
    adapter = FakeRuntimeAdapter(events=[])
    output = tmp_path / "run"
    orchestrator = OwnerInterfaceOrchestrator(
        checkpoint="S",
        stage="all",
        output_dir=output,
        config_path=config,
        panel_path=panel,
        cohort_path=cohort,
        adapter=adapter,
    )
    summary = orchestrator.run()

    assert summary["status"] == "completed"
    assert adapter.parser_calls == 1
    assert {"static.generate_complete_row", "dynamic.forward_with_dynamic_arm", "gradient.run_static_dynamic_gradient_path_audit"} <= set(adapter.helper_calls)
    assert adapter.persistent_calls == [("Y11", 1), ("Y11", 3)]
    result = json.loads((output / "per_event_results.jsonl").read_text(encoding="utf-8"))
    assert result["same_forward_mask_and_hook"] is True
    assert (output / "exact_prefix_manifest.json").is_file()
    assert (output / "intervention_manifest.json").is_file()
    assert (output / "gradient_receipt.json").is_file()
    assert (output / "terminal_summary.json").is_file()


def test_fail_collision_is_hard_and_shard_parser_is_strict(tmp_path: Path) -> None:
    config, panel, cohort = _fixture(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        OwnerInterfaceOrchestrator(
            checkpoint="S",
            stage="p1",
            output_dir=output,
            config_path=config,
            panel_path=panel,
            cohort_path=cohort,
            adapter=FakeRuntimeAdapter(events=[]),
        ).run()
    assert parse_event_shard("0/2") == (0, 2)
    assert parse_event_shard("1:3") == (1, 3)
    with pytest.raises(ValueError):
        parse_event_shard("2/2")
    with pytest.raises(ValueError):
        parse_event_shard("bad")


def test_dry_run_emits_complete_schema_without_running_adapter(tmp_path: Path) -> None:
    config, panel, cohort = _fixture(tmp_path)
    adapter = FakeRuntimeAdapter(events=[])
    output = tmp_path / "dry-run"
    summary = OwnerInterfaceOrchestrator(
        checkpoint="S",
        stage="p4",
        output_dir=output,
        config_path=config,
        panel_path=panel,
        cohort_path=cohort,
        adapter=adapter,
        dry_run=True,
    ).run()
    assert summary["status"] == "dry_run_schema_validated"
    assert adapter.parser_calls == 0
    for name in (
        "runtime_identity.json",
        "exact_prefix_manifest.json",
        "intervention_manifest.json",
        "per_event_results.jsonl",
        "gradient_receipt.json",
        "terminal_summary.json",
    ):
        assert (output / name).is_file()


def test_source_panel_hash_is_required_and_not_read_from_legacy_panel_key(tmp_path: Path) -> None:
    config, panel, cohort = _fixture(tmp_path)
    payload = json.loads(cohort.read_text(encoding="utf-8"))
    del payload["sources"]["source_panel"]
    cohort.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(OrchestrationError, match="derived_panel and source_panel"):
        OwnerInterfaceOrchestrator(
            checkpoint="S",
            stage="p1",
            output_dir=tmp_path / "missing-source",
            config_path=config,
            panel_path=panel,
            cohort_path=cohort,
            adapter=FakeRuntimeAdapter(events=[]),
        ).run()


def test_teacher_forced_offsets_use_previous_input_position_for_next_token() -> None:
    offsets = _teacher_forced_row_offsets(prefix_length=5, row_lengths=(3, 2, 4))
    assert offsets == ((4, 7), (7, 9), (9, 13))
    # The logits at [4:7] predict target tokens at combined input positions
    # [5:8], including both the first and final token of the first row.
    combined = list(range(14))
    for (start, end), row_start in zip(offsets, (5, 8, 10), strict=True):
        assert [combined[index + 1] for index in range(start, end)] == list(range(row_start, row_start + end - start))
    with pytest.raises(OrchestrationError, match="non-empty prefix"):
        _teacher_forced_row_offsets(prefix_length=0, row_lengths=(1,))


def test_exact_model_inputs_normalizes_single_image_grid_for_qwen_position_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(runner.ExperimentRuntimeAdapter)
    adapter.model = torch.nn.Linear(2, 2)
    adapter.model.config = SimpleNamespace(
        image_token_id=9,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    # Frozen A ordinal4/H0 image-2299 grid.  The runtime stores one image as
    # [3], while the exact native model payload retains the batched [1,3]
    # form.  These values are identical but their receipt shapes are not.
    stored_grid = torch.tensor([1, 46, 76], dtype=torch.long)
    native_grid = stored_grid.reshape(1, 3).clone()
    runtime = SimpleNamespace(
        image_grid_thw=stored_grid,
        native_inputs={"image_grid_thw": native_grid},
    )
    observed: dict[str, torch.Tensor] = {}

    def fake_position_ids(
        _model: object,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image_grid_thw: torch.Tensor,
    ) -> torch.Tensor:
        assert torch.equal(attention_mask, torch.ones_like(input_ids))
        observed["grid"] = image_grid_thw.detach().clone()
        return torch.arange(3 * input_ids.shape[1], dtype=torch.long).reshape(
            3, 1, input_ids.shape[1]
        )

    monkeypatch.setattr(static.query, "derive_explicit_position_ids", fake_position_ids)
    input_ids = torch.tensor([[*([9] * 874), 7, 8]], dtype=torch.long)
    payload, position_ids, mrope_hash = adapter.exact_model_inputs(
        runtime,
        input_ids,
    )

    assert observed["grid"].shape == (1, 3)
    assert torch.equal(observed["grid"], native_grid)
    assert stored_grid.shape == (3,)
    assert payload["image_grid_thw"].shape == (1, 3)
    assert position_ids.shape == (3, 1, 876)
    contract = dynamic.ExactPrefixContract(
        prefix_token_ids=tuple(int(value) for value in input_ids[0]),
        position_ids=position_ids,
        mrope_hash=mrope_hash,
        wrapper="object_box_commit",
        image_grid_thw=payload["image_grid_thw"],
    )
    assert dynamic.validate_exact_prefix(
        contract,
        input_ids=input_ids,
        position_ids=position_ids,
        mrope_hash=mrope_hash,
        extra_inputs=payload,
    )["passed"] is True


def test_exact_model_inputs_rejects_native_grid_value_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter = object.__new__(runner.ExperimentRuntimeAdapter)
    adapter.model = torch.nn.Linear(2, 2)
    adapter.model.config = SimpleNamespace(
        image_token_id=9,
        vision_config=SimpleNamespace(spatial_merge_size=2),
    )
    runtime = SimpleNamespace(
        image_grid_thw=torch.tensor([1, 4, 6], dtype=torch.long),
        native_inputs={"image_grid_thw": torch.tensor([[1, 4, 8]], dtype=torch.long)},
    )
    monkeypatch.setattr(
        static.query,
        "derive_explicit_position_ids",
        lambda _model, *, input_ids, **_kwargs: torch.arange(
            3 * input_ids.shape[1], dtype=torch.long
        ).reshape(3, 1, input_ids.shape[1]),
    )

    with pytest.raises(OrchestrationError, match="native payload image_grid_thw differs"):
        adapter.exact_model_inputs(
            runtime,
            torch.tensor([[9, 9, 9, 9, 9, 9, 7, 8]], dtype=torch.long),
        )


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        (torch.tensor([1, 4]), r"shape \[3\] or \[N,3\]"),
        (
            torch.tensor([[1, 4, 6], [1, 2, 3]]),
            "position_ids require Qwen",
        ),
    ],
)
def test_exact_model_inputs_rejects_invalid_or_multi_image_runtime_grid(
    grid: torch.Tensor,
    message: str,
) -> None:
    adapter = object.__new__(runner.ExperimentRuntimeAdapter)
    adapter.model = torch.nn.Linear(2, 2)
    runtime = SimpleNamespace(image_grid_thw=grid, native_inputs={})
    with pytest.raises(OrchestrationError, match=message):
        adapter.exact_model_inputs(runtime, torch.tensor([[7, 8]], dtype=torch.long))


def test_p4_candidates_share_pre_opener_prefix_without_cross_conditioning() -> None:
    rows = {
        "target-B": (100, 41, 101, 102, 10, 11, 12, 13, 103),
        "uncovered-B": (100, 42, 101, 102, 14, 15, 16, 17, 103),
        "covered-A": (100, 43, 101, 102, 18, 19, 20, 21, 103),
    }
    base, inputs = _build_independent_p4_candidate_inputs(
        (7, 8, 100), rows, opener_token_id=100, device="cpu"
    )
    assert base == (7, 8)
    for role, row in rows.items():
        values = tuple(int(value) for value in inputs[role][0].tolist())
        assert values[: len(base)] == base
        assert values[len(base) :] == row[:-1]
        assert values[len(base)] == 100
    assert tuple(inputs["target-B"][0].tolist()) != tuple(inputs["covered-A"][0].tolist())
    assert rows["uncovered-B"][:-1] not in tuple(inputs["covered-A"][0].tolist())


def test_p4_native_position_view_requires_contiguous_hidden_output() -> None:
    contiguous = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4).requires_grad_()
    selected = _native_arithmetic_position_view(contiguous, (1, 3, 5), "test")
    assert selected.shape == (3, 4)
    assert selected.grad_fn is not None
    assert selected.detach().tolist() == contiguous[0, (1, 3, 5), :].tolist()

    noncontiguous = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6).transpose(1, 2)
    assert noncontiguous.shape == (1, 6, 4)
    with pytest.raises(OrchestrationError, match="contiguous"):
        _native_arithmetic_position_view(noncontiguous, (1, 3, 5), "test")


def _endpoint_contract() -> static.WrapperContract:
    return static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
        eos_token_id=105,
    )


def _endpoint_context(*, covered: tuple[str, ...] = ("gt:1:1",)) -> SimpleNamespace:
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    return SimpleNamespace(
        event={"gt_owner_id": "gt:1:2", "image_id": "1"},
        runtime=SimpleNamespace(image_id="1", h0={}),
        covered_owner_ids=covered,
        target_row_ids=list(row),
        prefix_receipt={
            "model_input": {"prefix_sha256": "p"},
            "h0": {"exact_generated_history_prefix_sha256": "h"},
            "support_contract": {
                "status": "measured",
                "source": "test-support",
                "owner_ids": ["gt:1:1", "gt:1:2", "gt:1:3"],
            },
        },
    )


def _accepted_endpoint_row(owner_id: str = "gt:1:2") -> dict[str, object]:
    return {
        "generated_token_ids": [41, 101, 102, 10, 11, 12, 13, 103],
        "native_parse": {
            "valid": True,
            "parse_status": "accepted",
            "row_token_ids": [100, 41, 101, 102, 10, 11, 12, 13, 103],
        },
        "owner_match": {
            "status": "unique",
            "owner_id": owner_id,
            "source_specific": True,
            "physical_match": True,
        },
        "selected_token_log_probabilities": [-1.0] * 8,
        "selected_token_ranks": [1] * 8,
        "stop_reason": "box_end",
        "operator_receipt": {"non_target_max_abs_delta": 0.0},
    }


def test_endpoint_segment_arithmetic_and_natural_diagnostic_boundary_are_explicit() -> None:
    contract = _endpoint_contract()
    phases = _segment_token_indices(
        [100, 41, 101, 102, 10, 11, 12, 13, 103], contract=contract
    )
    assert phases["description"] == (1,)
    assert phases["geometry"] == (3, 4, 5, 6, 7)
    assert phases["x1"] == (4,)
    assert phases["y2"] == (7,)
    assert phases["closure"] == (8,)
    evidence = _natural_segment_scores(
        [100, 41, 101, 102, 10, 11, 12, 13, 103],
        [-1.0] * 8,
        [1] * 8,
        contract=contract,
    )
    assert evidence["status"] == "measured"
    assert evidence["teacher_forced"] is False
    assert evidence["segments"]["description"]["token_count"] == 1
    assert evidence["segments"]["x1"]["sum_log_probability"] == -1.0
    assert evidence["margins"]["status"] == "not_measured"

    with pytest.raises(OrchestrationError, match="coordinate segment"):
        _segment_token_indices([100, 41, 101, 102, 10, 11, 12, 2000, 103], contract=contract)


def test_endpoint_receipt_binds_active_arm_and_distinguishes_duplicate_unmatched_stop_and_support() -> None:
    context = _endpoint_context()
    context.checkpoint = "S"
    result = _endpoint_row_evidence(
        context,
        _accepted_endpoint_row(),
        active_intervention="Y11",
        contract=_endpoint_contract(),
        stop_observed=True,
    )
    assert result["active_intervention"] == "Y11"
    assert result["strict_native_endpoint"]["status"] == "accepted"
    assert result["identity_binding"]["checkpoint"] == "S"
    assert result["outcome"]["duplicate"] is False
    assert result["newly_covered_unique_owner_ids"] == ["gt:1:2"]
    assert result["remaining_independently_verified_support_at_stop"]["owner_ids"] == ["gt:1:3"]
    assert result["scores"]["target_B_vs_covered_A"]["status"] == "not_measured"

    duplicate = _accepted_endpoint_row(owner_id="gt:1:1")
    duplicate_receipt = _endpoint_row_evidence(
        context, duplicate, active_intervention="K11", contract=_endpoint_contract()
    )
    assert duplicate_receipt["outcome"]["duplicate"] is True
    assert duplicate_receipt["covered_A_repeat"] is True

    malformed = _accepted_endpoint_row()
    malformed["native_parse"] = {
        "valid": False,
        "parse_status": "malformed",
        "row_token_ids": [100, 41, 101, 102, 10, 11, 12, 13, 103, 999],
    }
    malformed["owner_match"] = {"status": "unmatched"}
    malformed["generated_token_ids"] = [41, 101, 102, 10, 11, 12, 13, 103, 999]
    malformed["stop_reason"] = "max_new_tokens"
    malformed_receipt = _endpoint_row_evidence(
        context, malformed, active_intervention="K00", contract=_endpoint_contract()
    )
    assert malformed_receipt["outcome"]["malformed"] is True
    assert malformed_receipt["outcome"]["over_continuation"] is True
    assert malformed_receipt["outcome"]["unmatched"] is True
    assert malformed_receipt["outcome"]["premature_stop"] is False

    no_support_context = _endpoint_context()
    no_support_context.prefix_receipt = {}
    no_support = _endpoint_row_evidence(
        no_support_context, _accepted_endpoint_row(), active_intervention="K00", contract=_endpoint_contract()
    )
    assert no_support["remaining_independently_verified_support_at_stop"]["status"] == "not_measured"
    assert no_support["remaining_independently_verified_support_at_stop"]["reason"]


def test_active_endpoint_scorer_uses_intervention_and_emits_all_verified_uncovered_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = _endpoint_contract()
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    context = _endpoint_context()
    context.prefix_ids = torch.tensor([[7, 9, 100]], dtype=torch.long)
    context.latest_row_ids = list(row)
    context.runtime.owner_mapping = {"gt:1:3": {"derived_index": 0}}
    context.runtime.raw = {"objects": [{"description": "unused", "bbox_2d": [0, 0, 1, 1]}]}
    context.runtime.image_span = static.ImageSpan(9, (1, 1, 1), 1, (1,), ((0, 0, 0),))

    monkeypatch.setattr(
        runner,
        "_row_from_owner",
        lambda *_args, **_kwargs: list(row),
    )

    class Output:
        def __init__(self, sequence_length: int) -> None:
            self.logits = torch.zeros((1, sequence_length, 256), dtype=torch.float32)
            # Keep all selected tokens finite and make token 41 the local best.
            self.logits[..., 41] = 2.0

    calls: list[torch.Tensor] = []

    def active_forward(ids: torch.Tensor) -> tuple[object, Mapping[str, object]]:
        calls.append(ids.detach().clone())
        return Output(int(ids.shape[1])), {
            "active_intervention": "K10",
            "intervention_applied": True,
            "auxiliary_forward_count": 0,
        }

    receipt = _score_active_endpoint_candidates(
        SimpleNamespace(tokenizer=None),
        context,
        active_intervention="K10",
        forward_fn=active_forward,
        contract=contract,
    )
    assert receipt["status"] == "measured"
    assert receipt["source"] == "active_intervention_scalar_forward"
    assert "verified-uncovered:gt:1:3" in receipt["candidates"]
    assert receipt["candidates"]["target-B"]["segments"]["description"]["status"] == "measured"
    assert "target_deltas" in receipt["comparisons"]
    assert receipt["row_entry_vs_native_im_end"]["active_intervention"] == "K10"
    margins = receipt["candidates"]["target-B"]["selected_vs_best_token_margins"]
    assert margins[1] > 0.0  # token 41 is above every competing token
    assert margins[0] < 0.0  # the opener loses to token 41
    assert receipt["candidates"]["target-B"]["best_competing_token_ids"][0] == 41
    assert len(calls) == 1 + 3 * (len(row) - 1)

    def native_fallback(ids: torch.Tensor) -> tuple[object, Mapping[str, object]]:
        return Output(int(ids.shape[1])), {
            "active_intervention": "K10",
            "intervention_applied": False,
        }

    with pytest.raises(OrchestrationError, match="required active endpoint"):
        _score_active_endpoint_candidates(
            SimpleNamespace(tokenizer=None),
            context,
            active_intervention="K10",
            forward_fn=native_fallback,
            contract=contract,
        )


@pytest.mark.parametrize("failure", ["nonfinite", "hook_error"])
def test_required_active_endpoint_mechanical_failure_is_invalid(failure: str) -> None:
    context = _endpoint_context()
    contract = _endpoint_contract()

    def failed_forward(ids: torch.Tensor) -> tuple[object, Mapping[str, object]]:
        if failure == "hook_error":
            raise RuntimeError("hook application failed")
        output = SimpleNamespace(
            logits=torch.full((1, int(ids.shape[1]), 256), float("nan"))
        )
        return output, {
            "active_intervention": "K11",
            "intervention_applied": True,
        }

    with pytest.raises(OrchestrationError, match="required active endpoint"):
        _score_active_endpoint_candidates(
            SimpleNamespace(tokenizer=None),
            context,
            active_intervention="K11",
            forward_fn=failed_forward,
            contract=contract,
        )


def test_required_active_endpoint_rejects_missing_target_b_before_optional_comparators() -> None:
    context = _endpoint_context()
    context.prefix_ids = torch.tensor([[7, 9, 100]], dtype=torch.long)
    context.target_row_ids = []
    context.latest_row_ids = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    calls = 0

    def active_forward(_ids: torch.Tensor) -> tuple[object, Mapping[str, object]]:
        nonlocal calls
        calls += 1
        raise AssertionError("missing required target-B must fail before any scalar forward")

    with pytest.raises(OrchestrationError, match="required active endpoint.*target-B"):
        _score_active_endpoint_candidates(
            SimpleNamespace(tokenizer=None),
            context,
            active_intervention="K11",
            forward_fn=active_forward,
            contract=_endpoint_contract(),
        )
    assert calls == 0


def test_required_active_endpoint_allows_explicitly_absent_optional_comparators() -> None:
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    context = _endpoint_context(covered=())
    context.prefix_ids = torch.tensor([[7, 9, 100]], dtype=torch.long)
    context.target_row_ids = list(row)
    context.latest_row_ids = []
    context.prefix_receipt["support_contract"]["owner_ids"] = ["gt:1:2"]

    def active_forward(ids: torch.Tensor) -> tuple[object, Mapping[str, object]]:
        return SimpleNamespace(
            logits=torch.zeros((1, int(ids.shape[1]), 256), dtype=torch.float32)
        ), {
            "active_intervention": "K11",
            "intervention_applied": True,
        }

    result = _score_active_endpoint_candidates(
        SimpleNamespace(tokenizer=None),
        context,
        active_intervention="K11",
        forward_fn=active_forward,
        contract=_endpoint_contract(),
    )
    assert result["status"] == "measured"
    assert set(result["candidates"]) == {"target-B"}
    assert result["missing_candidates"]["covered-A"]
    assert result["missing_candidates"]["verified-uncovered"]
    assert result["comparisons"]["target_deltas"]["target-B"]["status"] == "measured"


def test_owner_region_contract_excludes_shared_cells_and_reports_missing_support() -> None:
    span = static.ImageSpan(
        image_token_id=9,
        grid_thw=(1, 1, 4),
        merge_size=1,
        absolute_positions=(2, 3, 4, 5),
        grid_indices=((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)),
    )
    context = _endpoint_context()
    context.runtime.image_span = span
    context.event["owner_regions"] = {
        "gt:1:2": [0],
        "gt:1:1": [1],
        "gt:1:3": [2],
    }
    regions = {"b_exclusive": [0], "shared_core": [3]}
    owner_positions, owner_receipts, reason = _verified_owner_regions(context, regions)
    assert reason is None
    assert owner_positions["gt:1:3"] == (4,)
    assert owner_receipts["gt:1:3"]["role"] == "non-target-owner"

    context.event["owner_regions"]["gt:1:3"] = [1]
    with pytest.raises(OrchestrationError, match="overlap another owner"):
        _verified_owner_regions(context, {"b_exclusive": [0], "shared_core": [3]})

    context.event["owner_regions"]["gt:1:3"] = [3]
    with pytest.raises(OrchestrationError, match="shared-core"):
        _verified_owner_regions(context, {"b_exclusive": [0], "shared_core": [3]})

    context.event.pop("owner_regions")
    positions, receipts, reason = _verified_owner_regions(context, {})
    assert positions == {}
    assert receipts == {}
    assert "owner-region" in str(reason)


class _MassTokenizer:
    def __init__(self, special_ids: list[int] | None = None) -> None:
        self.all_special_ids = special_ids
        self.vocab_size = 2048

    def __len__(self) -> int:
        return self.vocab_size


def _mass_adapter(*, commit: bool, special_ids: list[int] | None = None) -> SimpleNamespace:
    wrapper = static.WrapperContract(
        assistant_format="object_box_commit" if commit else "object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
        commit_token_id=1500 if commit else None,
        eos_token_id=105,
    )
    return SimpleNamespace(wrapper_contract=wrapper, tokenizer=_MassTokenizer(special_ids))


def test_token_mass_contract_closed_and_commit_sets_are_sorted_nonempty_and_hashed() -> None:
    closed = _build_token_mass_contract(_mass_adapter(commit=False, special_ids=[1500, 1500, 100, 103, 105]))
    commit = _build_token_mass_contract(_mass_adapter(commit=True, special_ids=[1600, 1601, 100, 1500, 105]))
    closed_grammar, closed_stop, closed_invalid, closed_receipt = closed
    commit_grammar, commit_stop, commit_invalid, commit_receipt = commit
    assert 1500 not in closed_grammar
    assert 1500 in commit_grammar
    assert closed_stop == (103, 105)
    assert commit_stop == (105, 1500)
    assert closed_invalid == (1500,)
    assert commit_invalid == (1600, 1601)
    assert closed_receipt["invalid"]["count"] == len(closed_invalid) > 0
    assert commit_receipt["grammar"]["sha256"]


def test_token_mass_contract_fails_closed_without_special_token_ids() -> None:
    with pytest.raises(OrchestrationError, match="all_special_ids"):
        _build_token_mass_contract(_mass_adapter(commit=False, special_ids=None))


def test_gradient_audit_builds_native_nonempty_invalid_mass_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
        eos_token_id=105,
    )
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    image_span = static.ImageSpan(9, (1, 1, 4), 1, (1, 2, 3, 4), ((0, 0, 0), (0, 0, 1), (0, 0, 2), (0, 0, 3)))
    runtime = SimpleNamespace(
        image_id="1",
        image_span=image_span,
        image_grid_thw=torch.tensor([1, 1, 4]),
        native_inputs={},
        h0={"trace_sha256": "trace"},
    )
    prefix = [7, 9, 9, 9, 9, *row, 100]
    context = SimpleNamespace(
        runtime=runtime,
        prefix_ids=torch.tensor([prefix], dtype=torch.long),
        event={"image_cell_regions": {"b_exclusive": [0, 1], "background": [2, 3]}},
        natural_boundary=1,
        latest_row_ids=list(row),
        covered_owner_ids=("gt:1:0",),
        target_row_ids=list(row),
        uncovered_row_ids=list(row),
    )

    class _P4Adapter(_StoredGridProductionAdapter):
        checkpoint = "S"
        h0_root = tmp_path
        model_device = torch.device("cpu")
        wrapper_contract = wrapper
        tokenizer = _MassTokenizer([1500, 103, 105])
        config = SimpleNamespace(model_dump=lambda mode="json": {"mode": mode})

        def parse_row(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"valid": True, "owner_match": {"status": "unique", "owner_id": "gt:1:0"}}

    adapter = _P4Adapter()
    seen: dict[str, object] = {}

    def fake_capture(**_kwargs: object) -> SimpleNamespace:
        states = {
            name: torch.ones((2, 4), dtype=torch.float32, requires_grad=True)
            for name in ("image_residual", "matched_background", "latest_terminal_carrier", "latest_row_span")
        }
        return SimpleNamespace(captured_states=states, forward_id="forward-1")

    def fake_gradient_audit(*, batch: object, **_kwargs: object) -> dict[str, object]:
        seen["batch"] = batch
        return {"status": "valid"}

    monkeypatch.setattr(runner.gradient, "capture_native_forward", fake_capture)
    monkeypatch.setattr(runner.gradient, "run_static_dynamic_gradient_path_audit", fake_gradient_audit)
    receipt = _run_gradient_audit(adapter, context)
    batch = seen["batch"]
    assert tuple(batch.invalid_token_ids) == (1500,)
    assert tuple(batch.stop_token_ids) == (103, 105)
    assert receipt["token_mass_contract"]["invalid"]["count"] == 1
    assert adapter.model.rope_grid_shapes == [(1, 3), (1, 3), (1, 3)]
    assert runtime.image_grid_thw.shape == (3,)


def test_p4_actual_fragmented_regions_keep_native_block23_capture_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression for A ordinal4's 23x38 image grid and fragmented owner boxes."""

    wrapper = static.WrapperContract(
        assistant_format="object_box_commit",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
        commit_token_id=1500,
        eos_token_id=105,
    )
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103, 1500]
    image_positions = tuple(range(88, 88 + 23 * 38))
    target_positions = tuple(
        position
        for image_row in range(8)
        for position in range(149 + 38 * image_row, 154 + 38 * image_row)
    )
    covered_positions = tuple(
        position
        for image_row in range(8)
        for position in range(142 + 38 * image_row, 146 + 38 * image_row)
    )
    background_positions = tuple(range(88, 128))
    image_span = static.ImageSpan(
        image_token_id=9,
        grid_thw=(1, 46, 76),
        merge_size=2,
        absolute_positions=image_positions,
        grid_indices=tuple(
            (0, y, x) for y in range(23) for x in range(38)
        ),
    )
    prefix = [7] * 1235
    for position in image_positions:
        prefix[position] = 9
    prefix[1224:1234] = row
    prefix[1234] = wrapper.object_ref_start_token_id
    runtime = SimpleNamespace(
        image_id="2299",
        image_span=image_span,
        image_grid_thw=torch.tensor([1, 46, 76]),
        native_inputs={"image_grid_thw": torch.tensor([[1, 46, 76]])},
        h0={"trace_sha256": "trace"},
    )
    context = SimpleNamespace(
        runtime=runtime,
        prefix_ids=torch.tensor([prefix], dtype=torch.long),
        event={
            "gt_owner_id": "gt:2299:2",
            "A_B": {
                "A": {
                    "A_latest_covered": {"gt_owner_id": "gt:2299:1"},
                }
            },
            "image_cell_regions": {
                "b_exclusive": [position - 88 for position in target_positions],
                "background": [position - 88 for position in background_positions],
            },
            "owner_regions": {
                "gt:2299:2": [position - 88 for position in target_positions],
                "gt:2299:1": [position - 88 for position in covered_positions],
            },
        },
        natural_boundary=1,
        latest_row_ids=list(row),
        covered_owner_ids=("gt:2299:1",),
        target_row_ids=list(row),
        uncovered_row_ids=list(row),
        prefix_receipt={
            "support_contract": {
                "status": "measured",
                "owner_ids": ["gt:2299:1", "gt:2299:2"],
            }
        },
    )

    class _ActualShapeP4Adapter(_StoredGridProductionAdapter):
        checkpoint = "A"
        h0_root = tmp_path
        model_device = torch.device("cpu")
        wrapper_contract = wrapper
        tokenizer = _MassTokenizer([1600, 103, 105, 1500])
        config = SimpleNamespace(model_dump=lambda mode="json": {"mode": mode})

        def __init__(self) -> None:
            super().__init__()
            self.model.config.vision_config.spatial_merge_size = 2

        def parse_row(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"valid": True, "owner_match": {"status": "unique", "owner_id": "gt:2299:1"}}

    adapter = _ActualShapeP4Adapter()
    captured: dict[str, object] = {}

    def fake_capture(**kwargs: object) -> SimpleNamespace:
        scale = torch.tensor(1.0, requires_grad=True)
        hook_outputs = tuple(
            torch.full((1, 1243, 4), float(index + 1)) * scale
            for index in range(3)
        )
        states = kwargs["state_selector"](hook_outputs)
        sources = kwargs["gradient_source_selector"](hook_outputs)
        captured["states"] = states
        captured["sources"] = sources
        assert all(state is hook_outputs[0] for state in states.values())
        assert tuple(sources["image_residual"][0].positions) == target_positions
        assert tuple(sources["non_target_owner:gt:2299:1"][0].positions) == covered_positions
        logits = torch.zeros((len(row), 2048), dtype=torch.float32, requires_grad=True)
        return SimpleNamespace(
            captured_states=states,
            gradient_sources=sources,
            forward_id="actual-shape-forward",
            outputs={
                "target_logits": logits,
                "uncovered_b_logits": logits,
                "covered_a_logits": logits,
                "row_entry_boundary_logits": logits[0],
            },
        )

    def fake_gradient_audit(*, batch: object, forward_fn: object, **_kwargs: object) -> dict[str, object]:
        capture = forward_fn(batch)
        assert batch.image_residual is capture.captured_states["image_residual"]
        assert batch.non_target_owner_states["gt:2299:1"] is capture.captured_states[
            "non_target_owner:gt:2299:1"
        ]
        return {"status": "valid"}

    monkeypatch.setattr(runner.gradient, "capture_native_forward", fake_capture)
    monkeypatch.setattr(runner.gradient, "run_static_dynamic_gradient_path_audit", fake_gradient_audit)
    receipt = _run_gradient_audit(adapter, context)
    assert receipt["status"] == "valid"
    assert captured["states"]
    assert captured["sources"]


def test_parser_requires_explicit_cohort_path(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--checkpoint", "S", "--output-dir", str(tmp_path / "run")])
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "S",
            "--output-dir",
            str(tmp_path / "run"),
            "--cohort",
            str(tmp_path / "cohort.json"),
        ]
    )
    assert args.cohort == tmp_path / "cohort.json"


def test_runtime_materialization_scopes_panel_rows_to_selected_events() -> None:
    rows = [{"image_id": 1584}, {"image_id": 2299}, {"image_id": 4134}]
    selected = _select_event_panel_rows(rows, [{"image_id": 2299}])
    assert selected == [{"image_id": 2299}]
    with pytest.raises(OrchestrationError, match="absent from the derived panel"):
        _select_event_panel_rows(rows, [{"image_id": 9999}])
    with pytest.raises(OrchestrationError, match="duplicate image identities"):
        _select_event_panel_rows([{"image_id": 2299}, {"image_id": 2299}], [{"image_id": 2299}])


def test_trace_rows_are_indexed_by_numeric_panel_image_id() -> None:
    row = {"row_id": "coco2017_val_000000002299"}
    indexed = _index_trace_rows_by_image({row["row_id"]: row}, {"2299"})
    assert indexed["2299"] is row

    with pytest.raises(OrchestrationError, match="collide"):
        _index_trace_rows_by_image(
            {"coco2017_val_000000002299": {}, "2299": {}},
            {"2299"},
        )


def _receipt_event_for_checkpoint(*, checkpoint: str = "S") -> dict[str, object]:
    other = "A" if checkpoint == "S" else "S"
    pair = {
        "pair_status": "verified_pair",
        "A_latest_covered": {
            "gt_owner_id": "gt:1:1",
            "source_panel_object_index": 1,
            "natural_boundary": 1,
            "strict_complete_row": True,
        },
        "B_verified_uncovered": {
            "gt_owner_id": "gt:1:2",
            "verified_support": True,
            "strict_complete_row": False,
            "natural_boundary": 2,
            "exact_prefix_sha256": "a" * 64,
        },
    }
    return {
        "image_id": 1,
        "gt_owner_id": "gt:1:2",
        "A_B": {checkpoint: pair, other: {"pair_status": "malformed", "B_verified_uncovered": "ignore-me"}},
        "checkpoint_status": {
            checkpoint: {"verified_support": True, "strict_complete_row": False},
            other: {"verified_support": None},
        },
    }


def test_active_receipt_validation_ignores_inactive_comparator_branch() -> None:
    owner, eligible, status, _pair = _validate_active_event_receipts(_receipt_event_for_checkpoint(), checkpoint="S")
    assert (owner, eligible, status) == ("gt:1:2", True, "verified_pair")


def test_production_s_ordinal11_identity_strings_are_not_boolean_receipts() -> None:
    cohort_path = Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-08-05-static-dynamic-owner-interface-crossover/cohort/"
        "s-step2444-final-support.json"
    )
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    event = next(item for item in cohort["events"] if item.get("ordinal") == 11)
    assert (
        event["geometry_by_checkpoint"]["S"]["same_class_competitor_owner_id"]
        == "gt:5001:0"
    )

    owner, eligible, status, _pair = _validate_active_event_receipts(
        event,
        checkpoint="S",
    )
    assert (owner, eligible, status) == ("gt:5001:15", True, "verified_pair")

    malformed = json.loads(json.dumps(event))
    malformed["geometry_by_checkpoint"]["S"]["same_class_route_verified"] = "true"
    with pytest.raises(
        OrchestrationError,
        match=r"same_class_route_verified must be a boolean receipt",
    ):
        _validate_active_event_receipts(malformed, checkpoint="S")


def test_active_null_or_unknown_receipts_fail_closed() -> None:
    event = _receipt_event_for_checkpoint()
    event["A_B"]["S"] = {"pair_status": "no_verified_B", "B_verified_uncovered": None}
    _owner, eligible, status, _pair = _validate_active_event_receipts(event, checkpoint="S")
    assert eligible is False
    assert status == "no_verified_B"

    event = _receipt_event_for_checkpoint()
    event["A_B"]["S"]["pair_status"] = "unknown"
    with pytest.raises(OrchestrationError, match="pair_status is unknown"):
        _validate_active_event_receipts(event, checkpoint="S")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("verified_support", "true", "verified_support must be true"),
        ("strict_complete_row", "false", "strict_complete_row must be false"),
        ("natural_boundary", True, "natural_boundary must be a non-negative integer"),
        ("exact_prefix_sha256", "A" * 64, "lowercase 64-hex"),
        ("gt_owner_id", "gt:1:3", "gt_owner_id differs"),
    ),
)
def test_active_receipt_validation_rejects_drift(field: str, value: object, message: str) -> None:
    event = _receipt_event_for_checkpoint()
    event["A_B"]["S"]["B_verified_uncovered"][field] = value
    with pytest.raises(OrchestrationError, match=message):
        _validate_active_event_receipts(event, checkpoint="S")


def test_active_receipt_validation_requires_a_shape_and_order() -> None:
    event = _receipt_event_for_checkpoint()
    del event["A_B"]["S"]["A_latest_covered"]["source_panel_object_index"]
    with pytest.raises(OrchestrationError, match="A_latest_covered is missing"):
        _validate_active_event_receipts(event, checkpoint="S")
    event = _receipt_event_for_checkpoint()
    event["A_B"]["S"]["A_latest_covered"]["natural_boundary"] = 2
    with pytest.raises(OrchestrationError, match="strictly earlier"):
        _validate_active_event_receipts(event, checkpoint="S")


def test_ineligible_event_emits_complete_non_scored_matrix_without_actuators() -> None:
    context = SimpleNamespace(
        event={"gt_owner_id": "gt:2299:1"},
        runtime=SimpleNamespace(image_id="2299"),
        pair_status="indeterminate_image2299_support_transfer",
        eligibility_reason="active pair is not actuator-eligible",
        prefix_receipt={"target_owner_id": "gt:2299:1"},
    )
    result = _ineligible_event_result(
        context,
        checkpoint="A",
        runtime_attestation={"status": "validated"},
    )
    assert result["eligibility"]["actuators_called"] is False
    assert set(result["p1"]["arms"]) == {
        "K00",
        "K01",
        "K10",
        "K11",
        "K12",
        "K13",
        *(f"{arm}_block{layer}" for layer in (13, 23) for arm in ("R00", "R10", "R11", "R12")),
        "R00_block27",
        "R10_block27",
    }
    assert set(result["p2"]["arms"]) == set(dynamic.DYNAMIC_ARM_IDS)
    assert set(result["p3"]["cells"]) == set(dynamic.P3_CELL_IDS)
    assert result["p4"]["status"] == "invalid/uninterpretable"


def test_geometry_gate_fails_closed_before_every_actuator(monkeypatch: pytest.MonkeyPatch) -> None:
    eligible, reason = _geometry_actuator_gate(
        {
            "geometry_status": "indeterminate",
            "geometry_launch_eligible": False,
            "geometry_mechanical_disposition": "indeterminate_required_regions",
            "image_cell_regions": {"a_exclusive": [], "b_exclusive": [], "background": []},
        }
    )
    assert eligible is False
    assert reason

    def actuator_called(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("an actuator was called after the geometry gate failed")

    monkeypatch.setattr(runner, "_release_row_with_static", actuator_called)
    monkeypatch.setattr(runner, "_release_dynamic_horizon", actuator_called)
    monkeypatch.setattr(runner, "_run_gradient_audit", actuator_called)
    orchestrator = object.__new__(OwnerInterfaceOrchestrator)
    orchestrator.stage = "all"
    orchestrator.checkpoint = "S"
    orchestrator._runtime_attestation = {"status": "validated"}
    context = SimpleNamespace(
        actuator_eligible=False,
        eligibility_reason=f"geometry is not actuator-eligible: {reason}",
        pair_status="verified_pair",
        event={"gt_owner_id": "gt:1:2"},
        runtime=SimpleNamespace(image_id="1"),
        prefix_receipt={"target_owner_id": "gt:1:2"},
    )
    result = orchestrator._run_event(SimpleNamespace(), context)
    assert result["eligibility"]["actuators_called"] is False
    assert set(result["p1"]["arms"]) == set(runner.P1_ARM_IDS)
    assert set(result["p2"]["arms"]) == set(dynamic.DYNAMIC_ARM_IDS)
    assert set(result["p3"]["cells"]) == set(dynamic.P3_CELL_IDS)
    assert result["p4"]["status"] == "invalid/uninterpretable"


def _materialization_event(*, checkpoint: str, ordinal: int, eligible: bool) -> dict[str, object]:
    event = _receipt_event_for_checkpoint(checkpoint=checkpoint)
    owner_id = f"gt:{checkpoint.lower()}:{ordinal}"
    event["ordinal"] = ordinal
    event["image_id"] = ordinal
    event["gt_owner_id"] = owner_id
    pair = event["A_B"][checkpoint]
    if eligible:
        pair["B_verified_uncovered"]["gt_owner_id"] = owner_id
        event.update({
            "geometry_launch_eligible": True,
            "geometry_status": "available",
            "geometry_mechanical_disposition": "eligible_verified_pair_regions",
            "image_cell_regions": {
                "a_exclusive": [1],
                "b_exclusive": [2],
                "background": [3],
            },
        })
    else:
        event["A_B"][checkpoint] = {
            "pair_status": "no_verified_B",
            "B_verified_uncovered": None,
        }
        event.update({
            "geometry_launch_eligible": False,
            "geometry_status": "not_applicable",
            "geometry_mechanical_disposition": "not_applicable",
            "image_cell_regions": {
                "a_exclusive": [1],
                "b_exclusive": [2],
                "background": [3],
            },
        })
    return event


@pytest.mark.parametrize(
    ("checkpoint", "eligible_ordinal", "expected_counts"),
    (("S", 11, [8, 8, 7, 8]), ("A", 4, [8, 8, 8, 7])),
)
def test_ineligible_materialization_preserves_original_modulo_four_shards(
    checkpoint: str,
    eligible_ordinal: int,
    expected_counts: list[int],
) -> None:
    events = [
        _materialization_event(
            checkpoint=checkpoint,
            ordinal=ordinal,
            eligible=ordinal == eligible_ordinal,
        )
        for ordinal in range(1, 33)
    ]
    counts: list[int] = []
    excluded: list[dict[str, object]] = []
    for shard in range(4):
        selected, receipt = _partition_ineligible_materialization_shard(
            events,
            checkpoint=checkpoint,
            event_shard=(shard, 4),
        )
        counts.append(len(selected))
        excluded.extend(receipt["excluded_actuator_eligible_events"])
        assert receipt["original_shard_cohort_indices"] == list(range(shard, 32, 4))
        assert receipt["materialized_cohort_indices"] == [
            index
            for index in range(shard, 32, 4)
            if index != eligible_ordinal - 1
        ]
    assert counts == expected_counts
    assert excluded == [{
        "cohort_index": eligible_ordinal - 1,
        "cohort_ordinal": eligible_ordinal,
        "event_id": f"gt:{checkpoint.lower()}:{eligible_ordinal}",
        "image_id": str(eligible_ordinal),
    }]


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda event: event["A_B"]["S"].update(pair_status="unknown"), "pair_status is unknown"),
        (lambda event: event.update(geometry_launch_eligible=None), "geometry_launch_eligible"),
        (lambda event: event.update(geometry_status="indeterminate"), "geometry receipt drift"),
        (lambda event: event.update(geometry_mechanical_disposition="unknown"), "geometry receipt drift"),
    ),
)
def test_ineligible_materialization_rejects_unknown_or_drifted_gates(
    mutator: object,
    message: str,
) -> None:
    event = _materialization_event(checkpoint="S", ordinal=1, eligible=False)
    mutator(event)
    with pytest.raises(OrchestrationError, match=message):
        _partition_ineligible_materialization_shard(
            [event], checkpoint="S", event_shard=(0, 4)
        )


def test_ineligible_materialization_cli_contract_rejects_ambiguous_modes(tmp_path: Path) -> None:
    parser = build_parser()
    parsed = parser.parse_args([
        "--checkpoint", "S",
        "--cohort", str(tmp_path / "cohort.json"),
        "--output-dir", str(tmp_path / "out"),
        "--event-shard", "0/4",
        "--materialize-ineligible-only",
    ])
    assert parsed.materialize_ineligible_only is True

    base = {
        "checkpoint": "S",
        "stage": "all",
        "output_dir": tmp_path / "out",
        "event_shard": (0, 4),
        "materialize_ineligible_only": True,
    }
    for overrides, message in (
        ({"stage": "p1"}, "requires stage=all"),
        ({"dry_run": True}, "rejects dry_run"),
        ({"event_limit": 1}, "rejects event_limit"),
        ({"event_shard": None}, "requires event_shard"),
        ({"event_shard": (0, 2)}, "requires modulo-4"),
    ):
        with pytest.raises(ValueError, match=message):
            OwnerInterfaceOrchestrator(**{**base, **overrides})


def test_ineligible_materialization_never_opens_backend_or_actuators_and_emits_six_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = OwnerInterfaceOrchestrator(
        checkpoint="S",
        stage="all",
        output_dir=tmp_path / "non-scored",
        event_shard=(0, 4),
        materialize_ineligible_only=True,
    )
    event = _materialization_event(checkpoint="S", ordinal=1, eligible=False)
    orchestrator.events = [event]
    identity = {
        "schema_version": runner.SCHEMA_VERSION,
        "checkpoint": "S",
        "stage": "all",
        "execution_mode": "ineligible_contract_materialization",
    }
    monkeypatch.setattr(orchestrator, "_load_cpu_contract", lambda: identity)

    class ProcessorOnlyAdapter:
        def close(self) -> None:
            return None

        @property
        def model(self) -> object:
            raise AssertionError("model was accessed")

    adapter = ProcessorOnlyAdapter()
    monkeypatch.setattr(orchestrator, "_load_ineligible_materialization_adapter", lambda _identity: adapter)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("backend/model/actuator path was called")

    monkeypatch.setattr(orchestrator, "_load_adapter", forbidden)
    monkeypatch.setattr(orchestrator, "_run_event", forbidden)
    monkeypatch.setattr(runner, "_release_row_with_static", forbidden)
    monkeypatch.setattr(runner, "_release_dynamic_horizon", forbidden)
    monkeypatch.setattr(runner, "_run_gradient_audit", forbidden)
    context = SimpleNamespace(
        actuator_eligible=False,
        eligibility_reason="active pair is not actuator-eligible: no_verified_B",
        pair_status="no_verified_B",
        event=event,
        runtime=SimpleNamespace(image_id="1"),
        prefix_receipt={"target_owner_id": event["gt_owner_id"]},
    )
    monkeypatch.setattr(runner, "_make_event_context", lambda _adapter, _event: context)

    summary = orchestrator.run()
    assert summary["status"] == "completed"
    assert summary["execution_mode"] == "ineligible_contract_materialization"
    assert summary["runtime_attestation"]["status"] == "not_applicable"
    assert summary["events_attempted"] == 1
    for name in orchestrator._artifact_paths:
        assert (orchestrator.output_dir / name).is_file()
    result = json.loads((orchestrator.output_dir / "per_event_results.jsonl").read_text())
    assert result["eligibility"]["actuators_called"] is False
    assert set(result["p1"]["arms"]) == set(runner.P1_ARM_IDS)
    assert set(result["p2"]["arms"]) == set(dynamic.DYNAMIC_ARM_IDS)
    assert set(result["p3"]["cells"]) == set(dynamic.P3_CELL_IDS)
    assert result["p4"]["status"] == "invalid/uninterpretable"


def test_p1_block27_sentinel_marks_event_invalid_on_r10_drift(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_release(
        _adapter: object,
        _context: object,
        *,
        arm: str,
        residual_layer: int | None = None,
        residual_arm: str | None = None,
        **_kwargs: object,
    ) -> dict[str, object]:
        return {
            "status": "valid",
            "arm": arm,
            "residual_layer": residual_layer,
            "residual_arm": residual_arm,
        }

    monkeypatch.setattr(runner, "_release_row_with_static", fake_release)
    monkeypatch.setattr(static, "compare_noop_receipts", lambda *_args, **_kwargs: {"passed": True})

    def fake_sentinel(_baseline: Mapping[str, object], candidate: Mapping[str, object]) -> dict[str, object]:
        passed = candidate.get("residual_arm") == "R00"
        return {"instrumentation_valid": passed, "technical_invalid": not passed}

    monkeypatch.setattr(static, "assess_block27_sentinel", fake_sentinel)
    orchestrator = object.__new__(OwnerInterfaceOrchestrator)
    orchestrator.stage = "p1"
    orchestrator.checkpoint = "S"
    orchestrator._runtime_attestation = {"status": "validated"}
    context = SimpleNamespace(
        actuator_eligible=True,
        event={
            "gt_owner_id": "gt:1:2",
            "image_cell_regions": {
                "a_exclusive": [0],
                "b_exclusive": [1],
                "background": [2],
            },
        },
        runtime=SimpleNamespace(image_id="1"),
        prefix_ids=torch.tensor([[7, 100]], dtype=torch.long),
        prefix_receipt={"model_input": {}},
    )
    result = orchestrator._run_event(SimpleNamespace(), context)
    assert result["eligibility"] == {
        "status": "eligible",
        "pair_status": None,
        "reason": None,
        "actuators_called": True,
    }
    assert result["p1"]["status"] == "invalid/uninterpretable"
    assert result["p1"]["block27_sentinels"]["R00"]["instrumentation_valid"] is True
    assert result["p1"]["block27_sentinels"]["R10"]["instrumentation_valid"] is False
    assert any("R10 block27 sentinel" in reason for reason in result["p1"]["invalid_reasons"])


class _RowTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [len(text)]


def _owner_panel_objects(*, count: int, image_id: int = 2299) -> dict[str, object]:
    objects = []
    for index in range(count):
        objects.append(
            {
                "desc": "person" if index in {1, 2} else f"thing-{index}",
                "category_name": "person" if index in {1, 2} else f"thing-{index}",
                "coco_ann_id": 1000 + index,
                "bbox_2d": [
                    f"<|coord_{index + 1}|>",
                    "<|coord_10|>",
                    f"<|coord_{index + 11}|>",
                    "<|coord_20|>",
                ],
            }
        )
    return {"image_id": image_id, "width": 1000, "height": 1000, "objects": objects}


def test_source_owner_mapping_uses_source_identity_after_geo_reorder() -> None:
    source = _owner_panel_objects(count=21)
    derived_objects = list(source["objects"])
    source_one = derived_objects.pop(1)
    derived_objects.insert(20, source_one)
    derived = {**source, "objects": derived_objects}
    owners, mapping, receipt = _build_source_derived_owner_mapping(source, derived)
    assert mapping["gt:2299:1"]["source_index"] == 1
    assert mapping["gt:2299:1"]["derived_index"] == 20
    assert mapping["gt:2299:1"]["coco_ann_id"] == 1001
    assert receipt["mapping_sha256"]
    assert owners[1]["owner_id"] == "gt:2299:1"
    runtime = SimpleNamespace(raw=derived, image_id="2299", owner_mapping=mapping)
    contract = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    row = _row_from_owner(runtime, "gt:2299:1", _RowTokenizer(), contract)
    assert row == [100, 6, 101, 102, 12, 20, 22, 30, 103]
    match = source_specific_owner_match(
        {"description": "person", "bbox": [2, 10, 12, 20]}, owners
    )
    assert match["status"] == "unique"
    assert match["owner_id"] == "gt:2299:1"


def test_source_owner_mapping_fails_closed_for_ambiguous_or_missing_identity() -> None:
    source = _owner_panel_objects(count=2)
    source["objects"][1]["desc"] = source["objects"][0]["desc"]
    source["objects"][1]["category_name"] = source["objects"][0]["category_name"]
    source["objects"][1]["bbox_2d"] = list(source["objects"][0]["bbox_2d"])
    derived = {**source, "objects": [dict(obj) for obj in source["objects"]]}
    for panel in (source, derived):
        for obj in panel["objects"]:
            obj.pop("coco_ann_id")
    with pytest.raises(OrchestrationError, match="ambiguous"):
        _build_source_derived_owner_mapping(source, derived)

    source = _owner_panel_objects(count=1)
    derived = {**source, "objects": [dict(source["objects"][0], coco_ann_id=2000, desc="other", category_name="other")]}
    with pytest.raises(OrchestrationError, match="missing"):
        _build_source_derived_owner_mapping(source, derived)


class _ProductionLayer(torch.nn.Module):
    _coordexp_decoder_layer = True

    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.ones(width))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.bias


class _ProductionDynamicModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = torch.nn.ModuleList(_ProductionLayer() for _ in range(24))
        self.config = SimpleNamespace(
            image_token_id=9,
            vision_config=SimpleNamespace(spatial_merge_size=1),
        )
        self.rope_grid_shapes: list[tuple[int, ...]] = []

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        _video_grid_thw: torch.Tensor | None,
        *,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        assert tuple(attention_mask.shape[-2:]) in {
            (1, int(input_ids.shape[1])),
            (int(input_ids.shape[1]), int(input_ids.shape[1])),
        }
        self.rope_grid_shapes.append(tuple(image_grid_thw.shape))
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).reshape(1, -1)
        return positions.repeat(3, 1).unsqueeze(1), None

    def forward(self, *, input_ids: torch.Tensor, position_ids: torch.Tensor, **_: object) -> torch.Tensor:
        del position_ids
        basis = torch.arange(1, 5, dtype=torch.float32, device=input_ids.device)
        hidden = input_ids.float().unsqueeze(-1) * basis
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class _ProductionDynamicAdapter:
    def __init__(self) -> None:
        self.model = _ProductionDynamicModel()

    def exact_model_inputs(self, _runtime: object, input_ids: torch.Tensor):
        positions = torch.arange(3 * input_ids.shape[1], dtype=torch.long).reshape(3, 1, input_ids.shape[1])
        return {"input_ids": input_ids, "position_ids": positions, "use_cache": False}, positions, "mrope"


class _StoredGridProductionAdapter(_ProductionDynamicAdapter):
    @property
    def model_device(self) -> torch.device:
        return torch.device("cpu")

    def exact_model_inputs(
        self,
        runtime: object,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[dict[str, object], torch.Tensor, str]:
        return runner.ExperimentRuntimeAdapter.exact_model_inputs(
            self,
            runtime,
            input_ids,
            attention_mask=attention_mask,
        )


def test_active_endpoint_scalar_scoring_normalizes_stored_runtime_grid() -> None:
    adapter = _StoredGridProductionAdapter()
    adapter.tokenizer = None
    context = _endpoint_context(covered=())
    context.prefix_ids = torch.tensor([[9, 7, 100]], dtype=torch.long)
    context.latest_row_ids = []
    context.prefix_receipt["support_contract"]["owner_ids"] = ["gt:1:2"]
    context.runtime.image_grid_thw = torch.tensor([1, 1, 1], dtype=torch.long)
    context.runtime.native_inputs = {
        "image_grid_thw": torch.tensor([[1, 1, 1]], dtype=torch.long)
    }

    def active_forward(ids: torch.Tensor) -> tuple[object, Mapping[str, object]]:
        payload, _positions, _mrope = adapter.exact_model_inputs(context.runtime, ids)
        assert payload["image_grid_thw"].shape == (1, 3)
        return SimpleNamespace(
            logits=torch.zeros((1, int(ids.shape[1]), 256), dtype=torch.float32)
        ), {
            "active_intervention": "K10",
            "intervention_applied": True,
        }

    receipt = _score_active_endpoint_candidates(
        adapter,
        context,
        active_intervention="K10",
        forward_fn=active_forward,
        contract=_endpoint_contract(),
    )
    assert receipt["status"] == "measured"
    assert adapter.model.rope_grid_shapes
    assert set(adapter.model.rope_grid_shapes) == {(1, 3)}
    assert context.runtime.image_grid_thw.shape == (3,)


def test_production_dynamic_replacement_distinguishes_d01_self_from_d10_mute() -> None:
    adapter = _ProductionDynamicAdapter()
    runtime = SimpleNamespace()
    ids = torch.tensor([[10, 11, 12, 13]], dtype=torch.long)
    d01, d01_receipt = _capture_dynamic_replacement(
        adapter,
        runtime,
        ids,
        (2,),
        background_positions=(0, 1),
        arm_id="D01",
    )
    d10, d10_receipt = _capture_dynamic_replacement(
        adapter,
        runtime,
        ids,
        (2,),
        background_positions=(0, 1),
        arm_id="D10",
    )
    assert d01_receipt["operation"] == "exact_self"
    assert d01_receipt["numeric_self"] is True
    assert d10_receipt["operation"] == "norm_matched_same_row_background"
    assert d10_receipt["numeric_self"] is False
    assert not torch.equal(d01, d10)


def test_event_context_binds_full_covered_owner_set_from_native_h0_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    runtime = SimpleNamespace(
        h0={"rows": [row, row]},
        owners=[{"owner_id": "gt:1:0"}, {"owner_id": "gt:1:1"}, {"owner_id": "gt:1:2"}],
        prompt_ids=torch.tensor([[7, 8]], dtype=torch.long),
    )
    adapter = SimpleNamespace(
        checkpoint="S",
        panel_rows={"1": runtime},
        h0_ledger_records={
            "1": {
                "gt:1:1": {
                    "natural_boundary": 1,
                    "covered_owner_ids": ("gt:1:0",),
                    "latest_covered_owner_id": "gt:1:0",
                    "exact_prefix_token_ids": tuple(row),
                    "exact_prefix_sha256": sha256_token_ids(row),
                    "strict_complete_row": True,
                },
                "gt:1:2": {
                    "natural_boundary": 2,
                    "covered_owner_ids": ("gt:1:0", "gt:1:1"),
                    "latest_covered_owner_id": "gt:1:1",
                    "exact_prefix_token_ids": tuple(row + row),
                    "exact_prefix_sha256": sha256_token_ids(row + row),
                },
            }
        },
        model_device=torch.device("cpu"),
        wrapper_contract=wrapper,
        tokenizer=None,
    )
    event = {
        "image_id": 1,
        "gt_owner_id": "gt:1:2",
        "A_B": {
            "S": {
                "pair_status": "verified_pair",
                "A_latest_covered": {
                    "gt_owner_id": "gt:1:1",
                    "source_panel_object_index": 1,
                    "natural_boundary": 1,
                    "strict_complete_row": True,
                },
                "B_verified_uncovered": {
                    "gt_owner_id": "gt:1:2",
                    "verified_support": True,
                    "strict_complete_row": False,
                    "natural_boundary": 2,
                    "exact_prefix_sha256": sha256_token_ids(row + row),
                },
            }
        },
        "target_row_token_ids": row,
    }
    context = _make_event_context(adapter, event)
    assert context.covered_owner_ids == ("gt:1:0", "gt:1:1")
    assert context.checkpoint == "S"
    # The injected resolver is opt-in.  Legacy consumers retain the exact
    # rows[:natural_boundary] projection and seeded prompt+history+opener input.
    assert context.prefix_ids.detach().cpu().tolist() == [[7, 8, *row, *row, 100]]
    assert context.latest_row_ids == row
    assert context.completed_row_ids == (tuple(row), tuple(row))
    mapping_rows = [
        {
            "owner_id": f"gt:1:{index}",
            "source_index": index,
            "derived_index": index,
            "coco_ann_id": 100 + index,
            "mapping_method": "coco_ann_id",
        }
        for index in range(3)
    ]
    mapped_runtime = SimpleNamespace(
        **runtime.__dict__,
        owner_mapping={row["owner_id"]: row for row in mapping_rows},
        mapping_receipt={
            "schema_version": "owner_interface.source_derived_mapping.v1",
            "image_id": "1",
            "source_owner_count": 3,
            "derived_owner_count": 3,
            "source_to_derived": mapping_rows,
            "mapping_sha256": runner.sha256_json(mapping_rows),
            "mapping_method_census": {"coco_ann_id": 3},
        },
    )
    mapped_adapter = SimpleNamespace(
        **{**adapter.__dict__, "panel_rows": {"1": mapped_runtime}}
    )
    mapped_event = {
        **event,
        "panel_identity": {
            "status": "matched",
            "source_panel_object_index": 2,
            "derived_panel_object_index": 2,
            "coco_ann_id": 102,
            "mapping_method": "coco_ann_id",
        },
    }
    monkeypatch.setattr(runner, "_row_from_owner", lambda *_args, **_kwargs: list(row))
    mapped_context = _make_event_context(mapped_adapter, mapped_event)
    assert mapped_context.prefix_receipt["owner_mapping"]["source_to_derived"] == mapping_rows
    missing_a = {"1": dict(adapter.h0_ledger_records["1"])}
    del missing_a["1"]["gt:1:1"]
    missing_adapter = SimpleNamespace(**{**adapter.__dict__, "h0_ledger_records": missing_a})
    with pytest.raises(OrchestrationError, match="active A owner .*no checkpoint-native H0 ledger"):
        _make_event_context(missing_adapter, event)
    bad_a = {"1": {owner: dict(record) for owner, record in adapter.h0_ledger_records["1"].items()}}
    bad_a["1"]["gt:1:1"]["exact_prefix_sha256"] = "0" * 64
    bad_a_adapter = SimpleNamespace(**{**adapter.__dict__, "h0_ledger_records": bad_a})
    with pytest.raises(OrchestrationError, match="active A exact generated-history prefix hash"):
        _make_event_context(bad_a_adapter, event)
    bad_boundary_event = json.loads(json.dumps(event))
    bad_boundary_event["A_B"]["S"]["A_latest_covered"]["natural_boundary"] = 0
    with pytest.raises(OrchestrationError, match="active A boundary differs"):
        _make_event_context(adapter, bad_boundary_event)


def test_event_context_rejects_wrong_boundary_or_exact_history_hash() -> None:
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    runtime = SimpleNamespace(
        h0={"rows": [row, row]},
        owners=[{"owner_id": "gt:1:0"}, {"owner_id": "gt:1:1"}, {"owner_id": "gt:1:2"}],
        prompt_ids=torch.tensor([[7, 8]], dtype=torch.long),
    )
    ledger_record = {
        "natural_boundary": 2,
        "covered_owner_ids": ("gt:1:0", "gt:1:1"),
        "latest_covered_owner_id": "gt:1:1",
        "exact_prefix_token_ids": tuple(row + row),
        "exact_prefix_sha256": sha256_token_ids(row + row),
    }
    adapter = SimpleNamespace(
        checkpoint="S",
        panel_rows={"1": runtime},
    h0_ledger_records={
        "1": {
            "gt:1:1": {
                "natural_boundary": 1,
                "covered_owner_ids": ("gt:1:0",),
                "latest_covered_owner_id": "gt:1:0",
                "exact_prefix_token_ids": tuple(row),
                "exact_prefix_sha256": sha256_token_ids(row),
                "strict_complete_row": True,
            },
            "gt:1:2": ledger_record,
        }
    },
        model_device=torch.device("cpu"),
        wrapper_contract=wrapper,
        tokenizer=None,
    )
    event = {
        "image_id": 1,
        "gt_owner_id": "gt:1:2",
        "A_B": {
            "S": {
                "pair_status": "verified_pair",
                "A_latest_covered": {
                    "gt_owner_id": "gt:1:1",
                    "source_panel_object_index": 1,
                    "natural_boundary": 1,
                    "strict_complete_row": True,
                },
                "B_verified_uncovered": {
                    "gt_owner_id": "gt:1:2",
                    "verified_support": True,
                    "strict_complete_row": False,
                    "natural_boundary": 2,
                    "exact_prefix_sha256": sha256_token_ids(row + row),
                },
            }
        },
        "target_row_token_ids": row,
    }
    wrong_boundary = {
        **event,
        "A_B": {
            "S": {
                **event["A_B"]["S"],
                "B_verified_uncovered": {
                    **event["A_B"]["S"]["B_verified_uncovered"],
                    "natural_boundary": 3,
                },
            }
        },
    }
    with pytest.raises(OrchestrationError, match="differs from checkpoint-native"):
        _make_event_context(adapter, wrong_boundary)
    bad_record = {**ledger_record, "exact_prefix_sha256": "0" * 64}
    bad_adapter = SimpleNamespace(**{**adapter.__dict__, "h0_ledger_records": {"1": {"gt:1:2": bad_record}}})
    with pytest.raises(OrchestrationError, match="exact generated-history prefix hash"):
        _make_event_context(bad_adapter, event)


def test_cpu_contract_keeps_cohort_manifest_identity_when_h0_is_discovered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config, panel, cohort = _fixture(tmp_path)
    source_panel = tmp_path / "source-panel.jsonl"
    derived_hash = hashlib.sha256(panel.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(source_panel.read_bytes()).hexdigest()
    monkeypatch.setattr(runner, "DERIVED_PANEL_SHA256", derived_hash)
    monkeypatch.setattr(runner, "PANEL_SOURCE_SHA256", source_hash)
    h0 = tmp_path / "h0"
    (h0 / "configs").mkdir(parents=True)
    (h0 / "summary.json").write_text(json.dumps({"terminal_status": "completed"}), encoding="utf-8")
    (h0 / "run_manifest.json").write_text(
        json.dumps(
            {
                "terminal_status": "completed",
                "backend": "hf",
                "backend_mode": "generate",
                "resolved_config_fingerprints": {"infer_config": "fp"},
            }
        ),
        encoding="utf-8",
    )
    (h0 / "configs" / "resolved.json").write_text(
        json.dumps({"resolution": {"fingerprint": "fp"}}), encoding="utf-8"
    )
    (h0 / "pred_token_trace.jsonl").write_text("", encoding="utf-8")
    (h0 / "image_plan.jsonl").write_text("", encoding="utf-8")
    orchestrator = runner.OwnerInterfaceOrchestrator(
        checkpoint="S",
        stage="p4",
        output_dir=tmp_path / "run",
        config_path=config,
        panel_path=panel,
        cohort_path=cohort,
        h0_dir=h0,
    )
    identity = orchestrator._load_cpu_contract()
    assert identity["panel_identity"]["cohort_manifest_path"] == str(
        cohort.with_name("cohort.manifest.json").resolve()
    )
    assert identity["panel_identity"]["cohort_manifest_path"] != str((h0 / "run_manifest.json").resolve())


def test_production_static_residual_factory_installs_and_cleans_declared_arm(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ProductionDynamicModel()
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    span = static.ImageSpan(9, (1, 1, 2), 1, (0, 1), ((0, 0, 0), (0, 0, 1)))
    runtime = SimpleNamespace(
        image_span=span,
        image_grid_thw=torch.tensor([1, 1, 2]),
        image_token_id=9,
        merge_size=1,
        native_inputs={},
    )
    context = SimpleNamespace(
        runtime=runtime,
        prefix_ids=torch.tensor([[9, 9]], dtype=torch.long),
    )
    adapter = SimpleNamespace(
        model=model,
        tokenizer=None,
        wrapper_contract=wrapper,
        parse_row=lambda *_args, **_kwargs: {"valid": True, "owner_match": {"status": "unmatched"}},
    )

    monkeypatch.setattr(
        static,
        "capture_post_block_image_field",
        lambda *_args, **_kwargs: {
            "state": torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
            "receipt": {"call_count": 1},
        },
    )
    observed: dict[str, object] = {}

    def fake_generate(_model: object, **kwargs: object) -> dict[str, object]:
        observed["residual_factory"] = kwargs["residual_factory"]
        observed["residual_arm"] = kwargs["residual_arm"]
        factory = kwargs["residual_factory"]
        assert callable(factory)
        replacement = factory(
            step=0,
            input_ids=torch.tensor([[9, 9]], dtype=torch.long),
            position_ids=torch.arange(6, dtype=torch.long).reshape(3, 1, 2),
            span=span,
        )
        with replacement:
            model.layers[23](torch.zeros((1, 2, 4)))
        observed["receipt"] = replacement.receipt()
        return {"parsed": {"row_token_ids": [100, 41, 101, 102, 10, 11, 12, 13, 103]}, "generated_token_ids": []}

    monkeypatch.setattr(static, "generate_complete_row", fake_generate)
    result = _release_row_with_static(
        adapter,
        context,
        arm="K00",
        regions={"b_exclusive": (0,), "background": (1,)},
        residual_layer=23,
        residual_arm="R00",
    )
    assert observed["residual_arm"] == "R00"
    assert observed["receipt"]["cleanup_complete"] is True
    assert observed["receipt"]["call_count"] == 1
    assert result["residual_factory_calls"][0]["replacement"]["operation"] == "byte_identical_self"
    assert result["status"] == "valid"


def test_r11_is_a_true_bidirectional_a_b_swap(monkeypatch: pytest.MonkeyPatch) -> None:
    model = _ProductionDynamicModel()
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    span = static.ImageSpan(9, (1, 1, 3), 1, (0, 1, 2), ((0, 0, 0), (0, 0, 1), (0, 0, 2)))
    state = torch.tensor([[1.0, 0.0], [0.0, 2.0], [9.0, 9.0]])
    context = SimpleNamespace(
        runtime=SimpleNamespace(
            image_span=span,
            image_grid_thw=torch.tensor([1, 1, 3]),
            image_token_id=9,
            merge_size=1,
            native_inputs={},
        ),
        prefix_ids=torch.tensor([[9, 9]], dtype=torch.long),
    )
    adapter = SimpleNamespace(
        model=model,
        tokenizer=None,
        wrapper_contract=wrapper,
        parse_row=lambda *_args, **_kwargs: {"valid": True, "owner_match": {"status": "unmatched"}},
    )
    monkeypatch.setattr(
        static,
        "capture_post_block_image_field",
        lambda *_args, **_kwargs: {"state": state, "receipt": {"call_count": 1}},
    )
    observed: dict[str, object] = {}

    def fake_generate(_model: object, **kwargs: object) -> dict[str, object]:
        factory = kwargs["residual_factory"]
        assert callable(factory)
        replacement = factory(
            step=0,
            input_ids=torch.tensor([[9, 9]], dtype=torch.long),
            position_ids=torch.arange(6, dtype=torch.long).reshape(3, 1, 2),
            span=span,
        )
        observed["positions"] = replacement.absolute_positions
        observed["replacement"] = replacement.replacement
        return {
            "parsed": {"row_token_ids": [100, 41, 101, 102, 10, 11, 12, 13, 103]},
            "generated_token_ids": [],
        }

    monkeypatch.setattr(static, "generate_complete_row", fake_generate)
    monkeypatch.setattr(runner, "_score_active_endpoint_candidates", lambda *_args, **_kwargs: {"status": "not_measured", "reason": "test"})
    monkeypatch.setattr(runner, "_endpoint_row_evidence", lambda *_args, **_kwargs: {"status": "test"})
    result = _release_row_with_static(
        adapter,
        context,
        arm="K00",
        regions={"a_exclusive": (0,), "b_exclusive": (1,), "background": (2,)},
        residual_layer=23,
        residual_arm="R11",
    )
    assert observed["positions"] == (0, 1)
    assert torch.equal(observed["replacement"], state[[1, 0]])
    receipt = result["residual_factory_calls"][0]
    assert receipt["selected_a_indices"] == [0]
    assert receipt["selected_b_indices"] == [1]
    assert receipt["replacement"]["operation"] == "equal_count_ab_swap"


def test_static_release_preserves_generation_status_for_unique_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    runtime = SimpleNamespace(
        image_token_id=9,
        image_grid_thw=torch.tensor([1, 1, 1]),
        merge_size=1,
        native_inputs={},
    )
    context = SimpleNamespace(runtime=runtime, prefix_ids=torch.tensor([[9, 100]], dtype=torch.long))
    adapter = SimpleNamespace(
        model=_ProductionDynamicModel(),
        tokenizer=None,
        wrapper_contract=wrapper,
        parse_row=lambda *_args, **_kwargs: {
            "valid": True,
            "owner_match": {"status": "unique", "owner_id": "gt:1:0"},
        },
    )
    monkeypatch.setattr(
        static,
        "generate_complete_row",
        lambda *_args, **_kwargs: {
            "status": "complete",
            "complete_row": True,
            "stop_reason": "box_end",
            "parsed": {"row_token_ids": [100, 41, 101, 102, 10, 11, 12, 13, 103]},
            "generated_token_ids": [41, 101, 102, 10, 11, 12, 13, 103],
        },
    )
    result = _release_row_with_static(adapter, context, arm="K00")
    assert result["status"] == "valid"
    assert result["generation_status"] == "complete"
    assert result["owner_match"]["owner_id"] == "gt:1:0"


def test_dynamic_d21_is_explicitly_not_applicable_without_same_parent_donor() -> None:
    result = _release_dynamic_horizon(
        SimpleNamespace(),
        SimpleNamespace(),
        arm_id="D21",
        horizons=(1, 3),
    )
    assert result["status"] == "not_applicable"
    assert "same-parent donor" in result["reason"]
    assert result["required_donor_contract"]


def test_d12_horizon_rebinds_to_current_accepted_rows_only_after_acceptance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opener = 100
    row0 = [opener, 30, 103]
    row1 = [opener, 31, 32, 103]
    emitted = [
        [40, 41, 42, 103],
        [50, 103],
        [60, 61, 103],
    ]
    context = SimpleNamespace(
        event={"gt_owner_id": "gt:1:2"},
        runtime=SimpleNamespace(h0={"rows": [row0, row1]}),
        prefix_ids=torch.tensor([[7, *row0, *row1, opener]], dtype=torch.long),
        natural_boundary=2,
        latest_row_ids=row1,
        covered_owner_ids=(),
        target_row_ids=row1,
        uncovered_row_ids=row1,
        prefix_receipt={},
        completed_row_ids=(tuple(row0), tuple(row1)),
    )
    adapter = SimpleNamespace(
        wrapper_contract=SimpleNamespace(object_ref_start_token_id=opener),
    )
    observed: list[tuple[tuple[int, ...], ...]] = []

    def fake_release(
        _adapter: object,
        local_context: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        observed.append(tuple(local_context.completed_row_ids))
        suffix = emitted[len(observed) - 1]
        return {
            "parsed": {"valid": True},
            "generated_token_ids": suffix,
            "stop_reason": "box_end",
            "owner_match": {"status": "unmatched"},
        }

    monkeypatch.setattr(runner, "_release_dynamic_row", fake_release)
    monkeypatch.setattr(runner, "_endpoint_row_evidence", lambda *_args, **_kwargs: {"outcome": {"valid_row": False}})
    monkeypatch.setattr(dynamic, "bookkeep_horizon", lambda *_args, **_kwargs: {"status": "test"})
    result = _release_dynamic_horizon(
        adapter,
        context,
        arm_id="D12",
        horizons=(3,),
    )
    row2 = (opener, *emitted[0])
    row3 = (opener, *emitted[1])
    assert observed[0][-2:] == (tuple(row0), tuple(row1))
    assert observed[1][-2:] == (tuple(row1), row2)
    assert observed[2][-2:] == (row2, row3)
    assert result["horizon_3"]["row_count"] == 3


def test_dynamic_horizons_keep_stored_grid_raw_and_use_shared_position_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    stored_grid = torch.tensor([1, 1, 1], dtype=torch.long)
    runtime = SimpleNamespace(
        image_grid_thw=stored_grid,
        native_inputs={"image_grid_thw": stored_grid.reshape(1, 3).clone()},
        h0={"rows": [row]},
    )
    context = SimpleNamespace(
        event={"gt_owner_id": "gt:1:2"},
        runtime=runtime,
        prefix_ids=torch.tensor([[9, *row, 100]], dtype=torch.long),
        natural_boundary=1,
        latest_row_ids=row,
        covered_owner_ids=(),
        target_row_ids=row,
        uncovered_row_ids=row,
        prefix_receipt={},
        completed_row_ids=(tuple(row),),
        checkpoint="A",
    )
    adapter = _StoredGridProductionAdapter()
    adapter.wrapper_contract = _endpoint_contract()
    adapter.tokenizer = None

    def fake_release(
        local_adapter: _StoredGridProductionAdapter,
        local_context: object,
        **_kwargs: object,
    ) -> dict[str, object]:
        payload, _positions, _mrope = local_adapter.exact_model_inputs(
            local_context.runtime,
            local_context.prefix_ids,
        )
        assert payload["image_grid_thw"].shape == (1, 3)
        return {
            "parsed": {"valid": True},
            "generated_token_ids": row[1:],
            "stop_reason": "box_end",
            "owner_match": {"status": "unmatched"},
        }

    monkeypatch.setattr(runner, "_release_dynamic_row", fake_release)
    monkeypatch.setattr(
        runner,
        "_endpoint_row_evidence",
        lambda *_args, **_kwargs: {"outcome": {"valid_row": False}},
    )
    monkeypatch.setattr(
        dynamic,
        "bookkeep_horizon",
        lambda *_args, **_kwargs: {"status": "test"},
    )
    result = _release_dynamic_horizon(
        adapter,
        context,
        arm_id="D00",
        horizons=(1, 3),
    )
    assert result["horizon_1"]["row_count"] == 1
    assert result["horizon_3"]["row_count"] == 3
    assert adapter.model.rope_grid_shapes == [(1, 3)] * 4
    assert stored_grid.shape == (3,)


def test_d12_uses_current_history_and_rejects_new_different_length_pair() -> None:
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    old_equal = [100, 30, 31, 103]
    current_earlier = [100, 40, 103]
    current_latest = [100, 50, 51, 103]
    context = SimpleNamespace(
        runtime=SimpleNamespace(h0={"rows": [old_equal, old_equal]}),
        prefix_ids=torch.tensor([[7, *current_earlier, *current_latest, 100]], dtype=torch.long),
        natural_boundary=2,
        completed_row_ids=(tuple(current_earlier), tuple(current_latest)),
    )
    adapter = SimpleNamespace(wrapper_contract=wrapper)
    with pytest.raises(OrchestrationError, match="equal token length"):
        _release_dynamic_row(
            adapter,
            context,
            arm_id="D12",
            latest_row_ids=current_latest,
            max_new_tokens=1,
        )


def test_dynamic_release_freezes_carrier_positions_across_growing_candidate_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = static.WrapperContract(
        assistant_format="object_box_closed",
        object_ref_start_token_id=100,
        object_ref_end_token_id=101,
        box_start_token_id=102,
        box_end_token_id=103,
        coordinate_token_start_id=10,
    )
    row = [100, 41, 101, 102, 10, 11, 12, 13, 103]
    prefix = torch.tensor([[9, *row, 100]], dtype=torch.long)
    runtime = SimpleNamespace(
        image_grid_thw=torch.tensor([1, 1, 1]),
        native_inputs={"image_grid_thw": torch.tensor([[1, 1, 1]])},
        h0={"rows": [row]},
        image_span=SimpleNamespace(absolute_positions=(0,)),
    )

    class Adapter(_StoredGridProductionAdapter):
        tokenizer = None
        wrapper_contract = wrapper

        def parse_row(self, row_tokens: object, _runtime: object, *, row_index: int):
            del row_index
            return {
                "valid": True,
                "owner_match": {
                    "status": "unique",
                    "owner_id": "gt:1:0",
                    "source_specific": True,
                    "physical_match": True,
                },
                "native": {"parse_status": "accepted"},
                "row_token_ids": list(row_tokens),
            }

    adapter = Adapter()
    context = SimpleNamespace(
        event={"gt_owner_id": "gt:1:0", "image_id": "1"},
        runtime=runtime,
        prefix_ids=prefix,
        natural_boundary=1,
        latest_row_ids=row,
        covered_owner_ids=(),
        target_row_ids=row,
        prefix_receipt={},
    )
    position_calls: list[tuple[int, ...]] = []
    arm_position_calls: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        runner,
        "_score_active_endpoint_candidates",
        lambda *_args, **_kwargs: {"status": "not_measured", "reason": "test"},
    )

    def fake_capture(*_args: object, **kwargs: object):
        positions = tuple(int(value) for value in kwargs.get("positions", ()))
        if not positions and len(_args) >= 4:
            positions = tuple(int(value) for value in _args[3])
        position_calls.append(positions)
        return torch.zeros((len(positions), 4)), {"arm": "D10"}

    monkeypatch.setattr(runner, "_capture_dynamic_replacement", fake_capture)

    tokens = [41, 101, 102, 10, 11, 12, 13, 103]
    step = {"index": 0}

    class Output:
        def __init__(self, sequence_length: int, token: int) -> None:
            self.logits = torch.full((1, sequence_length, 256), -10.0)
            self.logits[0, -1, token] = 10.0

    def fake_forward(_model: object, model_inputs: Mapping[str, object], *, arm: object, **_kwargs: object):
        arm_position_calls.append(tuple(int(value) for value in arm.positions))
        token = tokens[step["index"]]
        step["index"] += 1
        ids = model_inputs["input_ids"]
        assert isinstance(ids, torch.Tensor)
        return Output(int(ids.shape[1]), token), {"hook_clean": True}

    monkeypatch.setattr(dynamic, "forward_with_dynamic_arm", fake_forward)
    monkeypatch.setattr(
        dynamic,
        "_coerce_row",
        lambda *_args, **_kwargs: SimpleNamespace(
            as_dict=lambda duplicate=False: {"duplicate": bool(duplicate)},
        ),
    )

    result = _release_dynamic_row(
        adapter,
        context,
        arm_id="D10",
        latest_row_ids=row,
        max_new_tokens=len(tokens),
    )
    assert result["stop_reason"] == "box_end"
    assert len(position_calls) == len(tokens)
    assert len(set(position_calls)) == 1
    assert len(set(arm_position_calls)) == 1
    assert adapter.model.rope_grid_shapes
    assert set(adapter.model.rope_grid_shapes) == {(1, 3)}
    assert runtime.image_grid_thw.shape == (3,)
