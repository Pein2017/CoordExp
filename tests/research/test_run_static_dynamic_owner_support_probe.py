from __future__ import annotations

import copy
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.research import materialize_static_dynamic_owner_interface_cohort as cohort
from scripts.research import run_static_dynamic_owner_support_probe as probe


def _panel_inputs() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    source = [
        {
            "image_id": 4134,
            "width": 100,
            "height": 100,
            "objects": [
                {"category_name": "person", "bbox_2d": ["<|coord_10|>", "<|coord_10|>", "<|coord_30|>", "<|coord_30|>"], "coco_ann_id": "ann-a"},
                {"category_name": "person", "bbox_2d": ["<|coord_50|>", "<|coord_10|>", "<|coord_70|>", "<|coord_30|>"], "coco_ann_id": "ann-b"},
            ],
        }
    ]
    derived = copy.deepcopy(source)
    mappings = [
        {
            "row_index": 0,
            "image_id": 4134,
            "mapping": [
                {"source_index": 0, "derived_index": 0, "object_sha256": probe.sha256_json(source[0]["objects"][0])},
                {"source_index": 1, "derived_index": 1, "object_sha256": probe.sha256_json(source[0]["objects"][1])},
            ],
        }
    ]
    receipt: dict[str, object] = {
        "schema_version": 1,
        "unit_id": probe.UNIT_ID,
        "ordering": "geo_sorted_xy",
        "sort_key": ["decoded_x1", "decoded_y1", "source_index"],
        "source_sha256": probe.sha256_json(source),
        "derived_sha256": probe.sha256_json(derived),
        "coordinate_arity_verified": True,
        "owner_multiset_preserved": True,
        "stable_sort_verified": True,
        "row_count": 1,
        "owner_count": 2,
        "mapping_count": 2,
        "mapping_sha256": probe.sha256_json(mappings),
        "source_to_derived": mappings,
    }
    return source, derived, receipt


def _h0(source: list[dict[str, object]], derived: list[dict[str, object]], *, checkpoint: str = "S") -> dict[str, object]:
    prefix = [11, 12, 13]
    records = [
        {
            "image_id": 4134,
            "gt_owner_id": "gt:4134:0",
            "source_panel_object_index": 0,
            "coco_ann_id": "ann-a",
            "category_name": "person",
            "bbox_pixel_xyxy": [1, 1, 3, 3],
            "native_tp": True,
            "native_fn": False,
            "strict_complete_row": True,
            "natural_boundary": 1,
            "natural_boundary_valid": True,
            "due_boundary_index": 1,
            "boundary_disposition": "native_tp_before_queried_row",
            "is_earliest_eligible_boundary": None,
            "exact_prefix_token_ids": prefix,
            "exact_prefix_sha256": probe.sha256_json(prefix),
            "support_status": "not_measured",
            "verified_support_claim": False,
            "prefix_semantics": "before_queried_owner_row",
            "generated_history_start_step": 0,
            "generated_history_end_step": 2,
            "excludes_stop": True,
            "covered_owner_ids": ["gt:4134:1"],
            "latest_covered_owner_id": "gt:4134:1",
            "queried_owner_not_covered": True,
            "due_boundary_evidence": {},
        },
        {
            "image_id": 4134,
            "gt_owner_id": "gt:4134:1",
            "source_panel_object_index": 1,
            "coco_ann_id": "ann-b",
            "category_name": "person",
            "bbox_pixel_xyxy": [5, 1, 7, 3],
            "native_tp": False,
            "native_fn": True,
            "strict_complete_row": False,
            "natural_boundary": 2,
            "natural_boundary_valid": True,
            "due_boundary_index": 2,
            "boundary_disposition": "native_fn_after_first_covered_row",
            "is_earliest_eligible_boundary": True,
            "exact_prefix_token_ids": [21, 22, 23],
            "exact_prefix_sha256": probe.sha256_json([21, 22, 23]),
            "support_status": "not_measured",
            "verified_support_claim": False,
            "prefix_semantics": "after_strict_covered_row_pre_stop",
            "generated_history_start_step": 0,
            "generated_history_end_step": 2,
            "excludes_stop": True,
            "covered_owner_ids": ["gt:4134:0"],
            "latest_covered_owner_id": "gt:4134:0",
            "queried_owner_not_covered": True,
            "due_boundary_evidence": {},
        },
    ]
    for record in records:
        record.update(
            {
                "unit_id": probe.UNIT_ID,
                "checkpoint": checkpoint,
                "config_fingerprint": f"cfg-{checkpoint}",
                "source_panel_sha256": probe.sha256_json(source),
                "derived_panel_sha256": probe.sha256_json(derived),
                "run_kind": "native_h0",
                "history_complete": True,
                "due_boundary_evidence": {
                    "boundary_disposition": record["boundary_disposition"],
                    "queried_owner_id": record["gt_owner_id"],
                    "covered_owner_ids": list(record["covered_owner_ids"]),
                    "covered_row_count": len(record["covered_owner_ids"]),
                    "latest_covered_owner_id": record["latest_covered_owner_id"],
                    "queried_owner_not_covered": True,
                    "prefix_end_step": record["generated_history_end_step"],
                    "stop_step": record["generated_history_end_step"] + 1,
                },
            }
        )
    return {
        "schema_version": cohort.LEDGER_SCHEMA_VERSION,
        "unit_id": probe.UNIT_ID,
        "checkpoint": checkpoint,
        "config_fingerprint": "cfg-S",
        "source_panel_sha256": probe.sha256_json(source),
        "derived_panel_sha256": probe.sha256_json(derived),
        "run_kind": "native_h0",
        "history_complete": True,
        "native_outcome_only": True,
        "verified_support_claim": False,
        "records": records,
    }


def _panel_and_h0(tmp_path: Path) -> tuple[probe.PanelInputs, probe.H0Inputs]:
    source, derived, receipt = _panel_inputs()
    panel = probe.load_panel_inputs(source, derived, receipt)
    h0 = probe.load_h0_inputs(_h0(source, derived), panel=panel, checkpoint="S")
    return panel, h0


def test_owner_local_features_use_physical_bank_and_are_deterministic() -> None:
    candidates = [
        {
            "candidate_id": "a",
            "generators": [{"generator_gt_owner_id": "gt:1:0"}],
            "strict_assignment_status": "matched",
            "strict_assignment_gt_owner_id": "gt:1:0",
        },
        {
            "candidate_id": "b",
            "generators": [{"generator_gt_owner_id": "gt:1:0"}],
            "strict_assignment_status": "unmatched",
            "strict_assignment_gt_owner_id": None,
        },
        {
            "candidate_id": "c",
            "generators": [{"generator_gt_owner_id": "gt:1:1"}],
            "strict_assignment_status": "matched",
            "strict_assignment_gt_owner_id": "gt:1:1",
        },
    ]
    result = probe.support_features({"a": -1.0, "b": -2.0, "c": -3.0}, candidates, owner_id="gt:1:0")
    assert result["assessed"] is True
    assert result["local_bank_size"] == 2
    assert result["teacher_forced"] is True
    assert result["behavioral_transfer"] is False
    assert result["peak_lift"] == pytest.approx(
        -1.0 - probe._logsumexp([-1.0, -2.0, -3.0]) + math.log(3)
    )


def test_q10_calibration_is_checkpoint_local_and_rejects_foreign_or_missing_controls() -> None:
    base = {
        "checkpoint": "S",
        "native_tp": True,
        "strict_complete_row": True,
        "run_kind": "native_h0",
        "history_complete": True,
        "source_panel_sha256": "a" * 64,
        "derived_panel_sha256": "b" * 64,
        "image_id": 4134,
        "exact_prefix_sha256": "c" * 64,
        "support_features": {"assessed": True, "peak_lift": 2.0, "local_concentration": 1.0},
    }
    receipt = probe.calibrate_support([base], checkpoint="S", source_panel_sha256="a" * 64, derived_panel_sha256="b" * 64)
    assert receipt["quantile"] == 0.1
    assert receipt["theta_peak_lift"] == 2.0
    assert receipt["theta_local_concentration"] == 1.0
    assert receipt["image2299_control_count"] == 0
    with pytest.raises(probe.SupportProbeError, match="absent TP calibrators"):
        probe.calibrate_support([], checkpoint="S", source_panel_sha256="a" * 64, derived_panel_sha256="b" * 64)
    foreign = dict(base, checkpoint="A")
    with pytest.raises(probe.SupportProbeError, match="foreign checkpoint"):
        probe.calibrate_support([foreign], checkpoint="S", source_panel_sha256="a" * 64, derived_panel_sha256="b" * 64)
    intervention = dict(base, intervention="K11")
    with pytest.raises(probe.SupportProbeError, match="intervention"):
        probe.calibrate_support([intervention], checkpoint="S", source_panel_sha256="a" * 64, derived_panel_sha256="b" * 64)


def test_stale_prefix_and_owner_mapping_fail_closed() -> None:
    panel, h0 = _panel_and_h0(Path("."))
    stale = copy.deepcopy(dict(h0.envelope))
    stale["records"] = copy.deepcopy(stale["records"])
    stale["records"][0]["exact_prefix_sha256"] = "0" * 64  # type: ignore[index]
    with pytest.raises(probe.SupportProbeError, match="exact prefix hash"):
        probe.load_h0_inputs(stale, panel=panel, checkpoint="S")
    source, derived, receipt = _panel_inputs()
    bad = _h0(source, derived)
    bad["records"] = copy.deepcopy(bad["records"])
    bad["records"][0]["bbox_pixel_xyxy"] = [99, 99, 100, 100]  # type: ignore[index]
    with pytest.raises(probe.SupportProbeError, match="geometry mapping"):
        probe.load_h0_inputs(bad, panel=panel, checkpoint="S")


def test_panel_mapping_mismatch_and_malformed_wrapper_fail_closed(tmp_path: Path) -> None:
    source, derived, receipt = _panel_inputs()
    bad_receipt = copy.deepcopy(receipt)
    bad_receipt["source_to_derived"][0]["mapping"][0]["derived_index"] = 1  # type: ignore[index]
    with pytest.raises(probe.SupportProbeError, match="mapping"):
        probe.load_panel_inputs(source, derived, bad_receipt)
    panel = probe.load_panel_inputs(source, derived, receipt)
    h0 = probe.load_h0_inputs(_h0(source, derived), panel=panel, checkpoint="S")
    config = tmp_path / "bad.yaml"
    config.write_text("model: {}\nbackend: {}\ntemplate:\n  assistant_format: object_box_commit\n", encoding="utf-8")
    with pytest.raises(probe.SupportProbeError, match="wrapper mismatch"):
        probe.validate_checkpoint_config(config, checkpoint="S", panel=panel, h0=h0)


def test_resolved_raw_config_fingerprint_is_stable_across_loader_defaults(tmp_path: Path) -> None:
    panel, h0 = _panel_and_h0(tmp_path)
    raw_config = {"model": {"base_model": "/models/base"}, "optional_default": None}
    resolved = tmp_path / "resolved.json"
    resolved.write_text(
        probe.canonical_json_bytes(
            {
                "config": raw_config,
                "resolution": {"fingerprint": probe.sha256_json(raw_config)},
            }
        ).decode("utf-8"),
        encoding="utf-8",
    )
    authority, authority_path, fingerprint = probe._resolved_config_authority(h0, resolved)
    assert authority == raw_config
    assert authority_path == resolved.resolve()
    assert fingerprint == probe.sha256_json(raw_config)
    tampered = copy.deepcopy(raw_config)
    tampered["optional_default"] = {"injected": True}
    resolved.write_text(
        probe.canonical_json_bytes(
            {
                "config": tampered,
                "resolution": {"fingerprint": probe.sha256_json(raw_config)},
            }
        ).decode("utf-8"),
        encoding="utf-8",
    )
    with pytest.raises(probe.SupportProbeError, match="raw fingerprint"):
        probe._resolved_config_authority(h0, resolved)


def test_detached_or_wrong_runtime_is_rejected() -> None:
    class FakeModel:
        def parameters(self):
            yield SimpleNamespace(device="cpu", dtype="torch.float32")

    session = SimpleNamespace(
        _model=FakeModel(),
        receipt=SimpleNamespace(effective_settings={"observed_attn_implementation": "sdpa"}),
    )
    with pytest.raises(probe.SupportProbeError, match="device"):
        probe.validate_live_runtime_identity(session)


def test_support_envelope_is_measured_and_keeps_native_boundary_identity() -> None:
    panel, h0 = _panel_and_h0(Path("."))
    contract = probe.CheckpointContract(
        checkpoint="S",
        config_path=Path("config.yaml"),
        config_sha256="a" * 64,
        config_fingerprint="cfg-S",
        wrapper=probe.EXPECTED_WRAPPERS["S"],
        parser=probe.EXPECTED_PARSERS["S"],
        source_panel_sha256=probe.sha256_json(panel.source),
        derived_panel_sha256=probe.sha256_json(panel.derived),
    )
    measured = dict(h0.records[0])
    measured.update(
        {
            "support_status": "measured",
            "verified_support_claim": True,
            "verified_support": True,
            "support_features": {"assessed": True, "peak_lift": 1.0, "local_concentration": 1.0},
        }
    )
    envelope = probe.build_support_envelope(
        contract=contract,
        h0=h0,
        records=[measured],
        calibration={"checkpoint": "S", "calibration_sha256": "b" * 64},
        panel=panel,
    )
    record = envelope["records"][0]
    assert record["support_status"] == "measured"
    assert record["verified_support_claim"] is True
    assert record["verified_support"] is True
    assert record["prefix_semantics"] == "before_queried_owner_row"
    assert record["due_boundary_evidence"]["queried_owner_not_covered"] is True
    bad = dict(measured, verified_support=None)
    with pytest.raises(probe.SupportProbeError, match="JSON boolean"):
        probe.build_support_envelope(
            contract=contract,
            h0=h0,
            records=[bad],
            calibration={"checkpoint": "S", "calibration_sha256": "b" * 64},
            panel=panel,
        )


def _capture_contract_and_plan() -> tuple[probe.CheckpointContract, probe.H0Inputs, dict[str, object]]:
    panel, h0 = _panel_and_h0(Path("."))
    contract = probe.CheckpointContract(
        checkpoint="S",
        config_path=Path("config.yaml"),
        config_sha256="a" * 64,
        config_fingerprint="cfg-S",
        wrapper=probe.EXPECTED_WRAPPERS["S"],
        parser=probe.EXPECTED_PARSERS["S"],
        source_panel_sha256=probe.sha256_json(panel.source),
        derived_panel_sha256=probe.sha256_json(panel.derived),
    )
    context = {
        "stable_key": "calibration|S|4134|gt:4134:0|" + ("c" * 64),
        "context_id": probe._capture_context_id("calibration|S|4134|gt:4134:0|" + ("c" * 64)),
        "kind": "calibration",
        "image_id": 4134,
        "eligible_for_score": True,
        "candidate_ids": ["candidate-0"],
        "gt_owner_id": "gt:4134:0",
        "candidate_gt_owner_id": None,
        "category_name": "person",
    }
    plan: dict[str, object] = {
        "context_plan_sha256": "plan-hash",
        "context_ids_sha256": probe.sha256_json([context["context_id"]]),
        "event_limit": None,
        "smoke_calibrators": 0,
        "contexts": [context],
    }
    return contract, h0, plan


def _fake_runtime_identity(contract: probe.CheckpointContract) -> dict[str, object]:
    model_identity = {
        "family": "base-only",
        "base": {"path": "/models/fake"},
        "qwen": {"model_class": "FakeModel"},
        "adapter": None,
        "embedding_delta": None,
    }
    effective = {
        "batch_size": 1,
        "device": "cuda",
        "observed_model_dtype": {
            "parameter_dtype_counts": {"torch.float32": 123},
            "parameter_dtype_names": ["torch.float32"],
        },
        "observed_attn_implementation": "sdpa",
    }
    receipt = {
        "backend": "hf",
        "backend_mode": "generate",
        "response_family": "hf",
        "backend_version": "test-transformers",
        "model_identity": model_identity,
        "tokenizer_identity": {"tokenizer_sha256": "t" * 64},
        "processor_identity": {"processor_class": "FakeProcessor"},
        "generation_config_fingerprint": "g" * 64,
        "effective_settings": effective,
        "likelihood_semantics": {"score_owned_channel": "policy_logprob"},
        "execution_model_identity": None,
    }
    launch = {
        "backend": "hf",
        "model_path": "/models/fake",
        "model_dtype": "fp32",
        "batch_size": 1,
        "generation_config_fingerprint": "g" * 64,
        "backend_options": {"hf": {"attn_implementation": "sdpa"}},
        "expected_model_identity": {},
        "execution_model_identity": None,
        "adapter": None,
        "embedding_delta": None,
    }
    return {
        "device": "cuda:0",
        "effective_device": "cuda",
        "normalized_device": "cuda:0",
        "torch_current_device": "cuda:0",
        "cuda_visible_devices": {
            "raw": "0",
            "tokens": ["0"],
            "selected_physical_device": "0",
        },
        "physical_device_id": "0",
        "dtype": "torch.float32",
        "attn_implementation": "sdpa",
        "checkpoint": contract.checkpoint,
        "config_fingerprint": contract.config_fingerprint,
        "schema_version": probe.RUNTIME_IDENTITY_SCHEMA_VERSION,
        "status": "validated",
        "passed": True,
        "backend": "hf",
        "backend_mode": "generate",
        "response_family": "hf",
        "backend_version": "test-transformers",
        "model_path": "/models/fake",
        "model_dtype": "fp32",
        "batch_size": 1,
        "generation_config_fingerprint": "g" * 64,
        "model_identity": model_identity,
        "tokenizer_identity": receipt["tokenizer_identity"],
        "processor_identity": receipt["processor_identity"],
        "effective_settings": effective,
        "likelihood_semantics": receipt["likelihood_semantics"],
        "execution_model_identity": None,
        "adapter_identity": {"requested": None, "observed": None},
        "embedding_delta_identity": {"requested": None, "observed": None},
        "launch": launch,
        "receipt": receipt,
    }


def _refresh_envelope_hash(envelope: dict[str, object]) -> None:
    envelope["envelope_sha256"] = probe.sha256_json(
        {key: value for key, value in envelope.items() if key != "envelope_sha256"}
    )


def _capture_shard(
    contract: probe.CheckpointContract,
    h0: probe.H0Inputs,
    plan: dict[str, object],
    *,
    observations: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    contexts = plan["contexts"]
    assert isinstance(contexts, list)
    work = probe._capture_work_units(plan, num_shards=1, candidate_batch_size=16)["per_shard"][0]
    shard = {
        "schema_version": probe.CAPTURE_SCHEMA_VERSION,
        "unit_id": probe.UNIT_ID,
        "checkpoint": contract.checkpoint,
        "config_fingerprint": contract.config_fingerprint,
        "source_panel_sha256": contract.source_panel_sha256,
        "derived_panel_sha256": contract.derived_panel_sha256,
        "h0_source_sha256": str(h0.source_info["sha256"]),
        "status": "captured",
        "shard_index": 0,
        "num_shards": 1,
        "event_limit": None,
        "smoke_calibrators": 0,
        "context_plan_sha256": plan["context_plan_sha256"],
        "runtime_identity": _fake_runtime_identity(contract),
        "contexts": contexts,
        "observations": observations if observations is not None else [
            {
                "context_id": contexts[0]["context_id"],
                "stable_key": contexts[0]["stable_key"],
                "kind": contexts[0]["kind"],
                "image_id": contexts[0]["image_id"],
                "gt_owner_id": contexts[0]["gt_owner_id"],
                "candidate_gt_owner_id": contexts[0]["candidate_gt_owner_id"],
                "category_name": contexts[0]["category_name"],
                "candidate_ids": contexts[0]["candidate_ids"],
                "status": "measured",
                "support_features": {
                    "assessed": True,
                    "peak_lift": 0.0,
                    "local_concentration": 0.0,
                },
                "candidate_scores": {"candidate-0": -1.0},
                "candidate_score_count": 1,
                "candidate_scores_sha256": probe.sha256_json({"candidate-0": -1.0}),
            }
        ],
        "work_units": {
            **work,
            "candidate_batch_size": 16,
            "batching_status": "not_admitted_exact_history_api_scalar_only",
        },
    }
    _refresh_envelope_hash(shard)
    return shard


def _fake_physical_candidates(*, two: bool = False) -> list[dict[str, object]]:
    candidates = [
        {
            "candidate_id": "candidate-0",
            "image_id": 4134,
            "normalized_description": "person",
            "generators": [{"generator_gt_owner_id": "gt:4134:0"}],
            "strict_assignment_status": "matched",
            "strict_assignment_gt_owner_id": "gt:4134:0",
        }
    ]
    if two:
        candidates.append(
            {
                "candidate_id": "candidate-1",
                "image_id": 4134,
                "normalized_description": "person",
                "generators": [{"generator_gt_owner_id": "gt:4134:1"}],
                "strict_assignment_status": "matched",
                "strict_assignment_gt_owner_id": "gt:4134:1",
            }
        )
    return candidates


def test_capture_partition_is_stable_complete_and_non_overlapping() -> None:
    contexts = [
        {
            "stable_key": f"candidate|S|{image}|owner-{image}|{'a' * 64}",
            "context_id": probe._capture_context_id(
                f"candidate|S|{image}|owner-{image}|{'a' * 64}"
            ),
        }
        for image in (1, 2, 3, 4, 5, 6, 7)
    ]
    assignments = [
        probe.partition_capture_contexts(contexts, shard_index=index, num_shards=3)
        for index in range(3)
    ]
    ids = [{str(row["context_id"]) for row in shard} for shard in assignments]
    assert set.union(*ids) == {str(row["context_id"]) for row in contexts}
    assert sum(len(item) for item in ids) == len(contexts)
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    reordered = list(reversed(contexts))
    assert [
        [row["context_id"] for row in probe.partition_capture_contexts(reordered, shard_index=index, num_shards=3)]
        for index in range(3)
    ] == [[row["context_id"] for row in shard] for shard in assignments]


def test_capture_shard_duplicate_missing_and_foreign_identity_fail_closed() -> None:
    contract, h0, plan = _capture_contract_and_plan()
    probe._validate_runtime_identity_payload(_fake_runtime_identity(contract), contract=contract)
    cuda_one = copy.deepcopy(_fake_runtime_identity(contract))
    cuda_one["device"] = "cuda:1"
    with pytest.raises(probe.SupportProbeError, match="logical device mismatch"):
        probe._validate_runtime_identity_payload(cuda_one, contract=contract)
    missing_visible = copy.deepcopy(_fake_runtime_identity(contract))
    missing_visible["cuda_visible_devices"].pop("tokens")  # type: ignore[index]
    with pytest.raises(probe.SupportProbeError, match="CUDA_VISIBLE_DEVICES mapping is ambiguous"):
        probe._validate_runtime_identity_payload(missing_visible, contract=contract)
    multi_visible = copy.deepcopy(_fake_runtime_identity(contract))
    multi_visible["cuda_visible_devices"] = {
        "raw": "0,1",
        "tokens": ["0", "1"],
        "selected_physical_device": "0",
    }
    with pytest.raises(probe.SupportProbeError, match="CUDA_VISIBLE_DEVICES mapping is ambiguous"):
        probe._validate_runtime_identity_payload(multi_visible, contract=contract)
    context_id = str(plan["contexts"][0]["context_id"])  # type: ignore[index]
    observation = {
        "context_id": context_id,
        "stable_key": plan["contexts"][0]["stable_key"],  # type: ignore[index]
        "kind": "calibration",
        "image_id": 4134,
        "gt_owner_id": "gt:4134:0",
        "candidate_gt_owner_id": None,
        "category_name": "person",
        "candidate_ids": ["candidate-0"],
        "status": "measured",
        "support_features": {"assessed": True},
        "candidate_scores": {"candidate-0": -1.0},
        "candidate_score_count": 1,
        "candidate_scores_sha256": probe.sha256_json({"candidate-0": -1.0}),
    }
    duplicate = _capture_shard(contract, h0, plan, observations=[observation, dict(observation)])
    _refresh_envelope_hash(duplicate)
    with pytest.raises(probe.SupportProbeError, match="incomplete or duplicated"):
        probe._validate_capture_shard(
            duplicate,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    missing = _capture_shard(contract, h0, plan, observations=[])
    _refresh_envelope_hash(missing)
    with pytest.raises(probe.SupportProbeError, match="incomplete or duplicated"):
        probe._validate_capture_shard(
            missing,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    foreign = _capture_shard(contract, h0, plan)
    foreign["config_fingerprint"] = "foreign"
    _refresh_envelope_hash(foreign)
    with pytest.raises(probe.SupportProbeError, match="foreign config_fingerprint"):
        probe._validate_capture_shard(
            foreign,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    corrupted_scores = _capture_shard(contract, h0, plan)
    corrupted_scores["observations"][0]["candidate_scores"]["candidate-0"] = -2.0  # type: ignore[index]
    _refresh_envelope_hash(corrupted_scores)
    with pytest.raises(probe.SupportProbeError, match="score hash mismatch"):
        probe._validate_capture_shard(
            corrupted_scores,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    mismatched_observation = _capture_shard(contract, h0, plan)
    mismatched_observation["observations"][0]["gt_owner_id"] = "foreign-owner"  # type: ignore[index]
    _refresh_envelope_hash(mismatched_observation)
    with pytest.raises(probe.SupportProbeError, match="observation identity mismatch"):
        probe._validate_capture_shard(
            mismatched_observation,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    missing_runtime = _capture_shard(contract, h0, plan)
    missing_runtime.pop("runtime_identity")
    _refresh_envelope_hash(missing_runtime)
    with pytest.raises(probe.SupportProbeError, match="runtime_identity is missing"):
        probe._validate_capture_shard(
            missing_runtime,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    foreign_runtime = _capture_shard(contract, h0, plan)
    foreign_runtime["runtime_identity"]["config_fingerprint"] = "foreign"  # type: ignore[index]
    _refresh_envelope_hash(foreign_runtime)
    with pytest.raises(probe.SupportProbeError, match="runtime_identity checkpoint/config mismatch"):
        probe._validate_capture_shard(
            foreign_runtime,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    malformed_dtype = _capture_shard(contract, h0, plan)
    malformed_dtype["runtime_identity"]["effective_settings"]["observed_model_dtype"] = "torch.float32"  # type: ignore[index]
    _refresh_envelope_hash(malformed_dtype)
    with pytest.raises(probe.SupportProbeError, match="dtype structure"):
        probe._validate_capture_shard(
            malformed_dtype,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    malformed_attention = _capture_shard(contract, h0, plan)
    malformed_attention["runtime_identity"]["effective_settings"]["observed_attn_implementation"] = {"name": "sdpa"}  # type: ignore[index]
    _refresh_envelope_hash(malformed_attention)
    with pytest.raises(probe.SupportProbeError, match="attention structure"):
        probe._validate_capture_shard(
            malformed_attention,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )
    tampered_hash = _capture_shard(contract, h0, plan)
    tampered_hash["envelope_sha256"] = "0" * 64
    with pytest.raises(probe.SupportProbeError, match="envelope self-hash mismatch"):
        probe._validate_capture_shard(
            tampered_hash,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )


@pytest.mark.parametrize(
    ("message", "status"),
    [("CUDA out of memory", "quarantined_oom"), ("candidate batch parity mismatch", "quarantined_parity")],
)
def test_capture_observation_quarantines_oom_and_parity_without_fallback(message: str, status: str) -> None:
    class FailingScorer:
        def score(self, _raw: object, _candidate: object) -> float:
            raise RuntimeError(message)

    context = {
        "context_id": "ctx:test",
        "stable_key": "candidate|S|1|owner|hash",
        "kind": "candidate",
        "image_id": 1,
        "gt_owner_id": "owner",
        "candidate_gt_owner_id": "owner",
        "eligible_for_score": True,
        "ineligible_reason": None,
        "category_name": "person",
    }
    result = probe._capture_context_observation(
        context,
        h0_by_owner={"owner": {"gt_owner_id": "owner"}},
        groups={(1, "person"): [{"candidate_id": "candidate-0"}]},
        scorer=FailingScorer(),
    )
    assert result["status"] == status
    assert result["support_features"] is None


def test_capture_stops_entire_shard_after_quarantine() -> None:
    contexts = [
        {
            "context_id": f"ctx:{index}",
            "stable_key": f"candidate|S|{index}|owner-{index}|hash",
            "kind": "candidate",
            "image_id": index,
            "gt_owner_id": f"owner-{index}",
            "candidate_gt_owner_id": f"owner-{index}",
            "category_name": "person",
            "candidate_ids": [f"candidate-{index}"],
            "eligible_for_score": True,
            "ineligible_reason": None,
        }
        for index in (1, 2, 3)
    ]
    groups = {
        (index, "person"): [
            {
                "candidate_id": f"candidate-{index}",
                "generators": [{"generator_gt_owner_id": f"owner-{index}"}],
                "strict_assignment_status": "matched",
                "strict_assignment_gt_owner_id": f"owner-{index}",
            }
        ]
        for index in (1, 2, 3)
    }
    h0_by_owner = {f"owner-{index}": {"gt_owner_id": f"owner-{index}"} for index in (1, 2, 3)}

    class FailOnSecondScorer:
        def __init__(self) -> None:
            self.calls = 0

        def score(self, _raw: object, _candidate: object) -> float:
            self.calls += 1
            if self.calls == 2:
                raise RuntimeError("CUDA out of memory")
            return -1.0

    scorer = FailOnSecondScorer()
    observations = probe._capture_observations_until_quarantine(
        contexts,
        h0_by_owner=h0_by_owner,
        groups=groups,
        scorer=scorer,
    )
    assert [row["status"] for row in observations] == ["measured", "quarantined_oom"]
    assert scorer.calls == 2


def test_quarantined_capture_envelope_is_incomplete_and_merge_rejects_it() -> None:
    contract, h0, plan = _capture_contract_and_plan()
    contexts = plan["contexts"]
    assert isinstance(contexts, list)
    context = contexts[0]
    observation = {
        "context_id": context["context_id"],
        "status": "quarantined_parity",
        "reason": "candidate batch parity mismatch",
        "candidate_scores": None,
        "candidate_score_count": 0,
        "candidate_scores_sha256": None,
        "support_features": None,
    }
    envelope = probe._capture_envelope(
        contract=contract,
        h0=h0,
        plan=plan,
        assigned_contexts=contexts,
        observations=[observation],
        shard_index=0,
        num_shards=1,
        candidate_batch_size=16,
        runtime_identity=_fake_runtime_identity(contract),
    )
    assert envelope["status"] == "quarantined"
    assert envelope["complete_assigned_observations"] is False
    assert envelope["quarantine"]["policy"] == "stop_shard_no_fallback"
    with pytest.raises(probe.SupportProbeError, match="quarantined"):
        probe._validate_capture_shard(
            envelope,
            plan=plan,
            contract=contract,
            h0=h0,
            shard_index=0,
            num_shards=1,
            candidate_batch_size=16,
        )


def test_merge_calibration_threshold_requires_complete_raw_capture() -> None:
    contract, h0, plan = _capture_contract_and_plan()
    context = plan["contexts"][0]
    assert isinstance(context, dict)
    physical = _fake_physical_candidates()
    measured_features = probe.support_features(
        {"candidate-0": 2.0}, physical, owner_id="gt:4134:0"
    )
    observations = {
        str(context["context_id"]): {
            "context_id": context["context_id"],
            "status": "measured",
            "support_features": measured_features,
            "candidate_scores": {"candidate-0": 2.0},
            "candidate_score_count": 1,
            "candidate_scores_sha256": probe.sha256_json({"candidate-0": 2.0}),
        }
    }
    calibration = probe.merge_calibration_observations(
        observations,
        [context],
        contract=contract,
        h0=h0,
        physical=physical,
    )
    assert calibration["theta_peak_lift"] == pytest.approx(measured_features["peak_lift"])
    excluded_context = dict(
        context,
        context_id="ctx:excluded",
        stable_key="calibration|S|4134|gt:4134:0|excluded",
        eligible_for_score=False,
        ineligible_reason="undercovered_owner_bank",
    )
    with_exclusion = dict(observations)
    with_exclusion["ctx:excluded"] = {
        "context_id": "ctx:excluded",
        "status": "indeterminate",
    }
    calibration_with_exclusion = probe.merge_calibration_observations(
        with_exclusion,
        [context, excluded_context],
        contract=contract,
        h0=h0,
        physical=physical,
    )
    assert calibration_with_exclusion["excluded_context_count"] == 1
    assert calibration_with_exclusion["excluded_contexts"][0]["reason"] == "undercovered_owner_bank"
    with pytest.raises(probe.SupportProbeError, match="missing"):
        probe.merge_calibration_observations({}, [context], contract=contract, h0=h0, physical=physical)
    quarantined = copy.deepcopy(observations)
    quarantined[str(context["context_id"])] = {
        "context_id": context["context_id"],
        "status": "quarantined_oom",
        "support_features": None,
    }
    with pytest.raises(probe.SupportProbeError, match="incomplete/quarantined"):
        probe.merge_calibration_observations(
            quarantined, [context], contract=contract, h0=h0, physical=physical
        )
    owner_error = copy.deepcopy(observations)
    owner_error[str(context["context_id"])] = {
        "context_id": context["context_id"],
        "status": "indeterminate",
        "reason": "owner_local_support_error:bank mismatch",
        "support_features": None,
        "candidate_scores": None,
        "candidate_score_count": 0,
        "candidate_scores_sha256": None,
    }
    with pytest.raises(probe.SupportProbeError, match="incomplete/quarantined"):
        probe.merge_calibration_observations(
            owner_error, [context], contract=contract, h0=h0, physical=physical
        )


def test_merge_recomputes_features_and_rejects_swapped_raw_scores() -> None:
    contract, h0, plan = _capture_contract_and_plan()
    original_context = plan["contexts"][0]
    assert isinstance(original_context, dict)
    context = dict(original_context, candidate_ids=["candidate-0", "candidate-1"])
    physical = _fake_physical_candidates(two=True)
    original_scores = {"candidate-0": 2.0, "candidate-1": 0.0}
    original_features = probe.support_features(original_scores, physical, owner_id="gt:4134:0")
    swapped_scores = {"candidate-0": 0.0, "candidate-1": 2.0}
    swapped_observation = {
        "context_id": context["context_id"],
        "status": "measured",
        "support_features": original_features,
        "candidate_scores": swapped_scores,
        "candidate_score_count": len(swapped_scores),
        "candidate_scores_sha256": probe.sha256_json(swapped_scores),
    }
    with pytest.raises(probe.SupportProbeError, match="support feature integrity mismatch"):
        probe.merge_calibration_observations(
            {str(context["context_id"]): swapped_observation},
            [context],
            contract=contract,
            h0=h0,
            physical=physical,
        )
