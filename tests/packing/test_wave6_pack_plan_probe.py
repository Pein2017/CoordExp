from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest


def _load_probe_module():
    name = "coordexp_wave6_pack_plan_probe_test_module"
    if name in sys.modules:
        return sys.modules[name]
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/probes/coordexp_swift/wave6_pack_plan_comparison.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def probe():
    return _load_probe_module()


@dataclass(frozen=True)
class _EncodedExample:
    example_id: str
    input_ids: tuple[int, ...]
    row_ids: tuple[str, ...]

    @property
    def intra_image_order_identity(self) -> str:
        payload = json.dumps(self.row_ids, separators=(",", ":")).encode()
        return f"test-row-order-sha256:{hashlib.sha256(payload).hexdigest()}"


def _examples() -> tuple[_EncodedExample, ...]:
    lengths = (6, 6, 4, 4, 7, 3, 8, 2, 6, 4, 9, 1)
    next_token = 100
    result = []
    for ordinal, length in enumerate(lengths):
        result.append(
            _EncodedExample(
                example_id=f"fixture-{ordinal}",
                input_ids=tuple(range(next_token, next_token + length)),
                row_ids=(f"row-{ordinal}-a", f"row-{ordinal}-b"),
            )
        )
        next_token += length + 10
    return tuple(result)


def _published_fixture_plan(probe, tmp_path: Path):
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    plan = probe.build_fixture_plan(
        _examples(),
        plan_path=plan_path,
        receipt_path=receipt_path,
        global_max_length=10,
    )
    probe.publish_json_absent(plan_path, plan)
    return plan, plan_path, receipt_path


def _refinalize(probe, value: dict, hash_field: str) -> dict:
    changed = deepcopy(value)
    changed.pop(hash_field, None)
    return probe.finalize_artifact(changed, hash_field=hash_field)


def _set_path(value: dict, path: tuple[str | int, ...], replacement) -> None:
    current = value
    for part in path[:-1]:
        current = current[part]
    current[path[-1]] = replacement


def test_comparison_records_exact_cpu_semantics_and_bounded_metrics(
    probe, tmp_path: Path
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)

    receipt = probe.execute_comparison(plan, _examples())
    checked = probe.validate_receipt(receipt, expected_plan=plan)

    assert checked["status"] == "completed_cpu_only_training_required"
    assert [arm["arm_id"] for arm in checked["arms"]] == [
        "source_order_next_fit",
        "window_binpack_w8",
        "window_binpack_w32",
        "online_window_binpack_l8",
        "online_window_binpack_l32",
    ]
    for arm in checked["arms"]:
        assert arm["semantic_oracle"] == {
            "each_once_exact": True,
            "intra_image_row_identity_order_exact": True,
            "strict_pack_plan_roundtrip": True,
            "strict_replay_exact": True,
            "worker_count_1_vs_8_exact": True,
        }
        assert arm["metrics"]["used_tokens"] == 60
        assert arm["metrics"]["capacity_tokens"] == (arm["metrics"]["pack_count"] * 10)
        assert arm["metrics"]["tail_waste"] == (arm["metrics"]["capacity_tokens"] - 60)
        assert arm["metrics"]["planning_wall_seconds"] >= 0.0
        assert arm["metrics"]["process_rss_after_bytes"] > 0
        assert arm["boundedness"]["within_all_declared_bounds"] is True
        assert arm["worker_equality"]["compared_worker_counts"] == [1, 8]
        assert arm["worker_equality"]["exact_semantic_projection_equal"] is True

    reference = checked["arms"][0]
    assert reference["ordering_change"] == {
        "flattened_order_sha256": probe.sha256_json(list(range(12))),
        "moved_example_count": 0,
        "absolute_displacement_sum": 0,
        "absolute_displacement_max": 0,
        "co_present_pair_count": 5,
        "co_present_pairs_sha256": probe.sha256_json(
            [[1, 2], [4, 5], [6, 7], [8, 9], [10, 11]]
        ),
        "added_vs_reference_count": 0,
        "removed_vs_reference_count": 0,
        "retained_vs_reference_count": 5,
    }
    for arm in checked["arms"][3:]:
        assert arm["resume_cursor_oracle"]["applicable"] is True
        assert arm["resume_cursor_oracle"]["fragmented_resume_exact"] is True
        assert arm["resume_cursor_oracle"]["terminal_cursor_complete"] is True
        assert arm["resume_cursor_oracle"]["strict_fragment_chain_verified"] is True


def test_plan_binds_current_owners_config_grid_and_frozen_stream(
    probe, tmp_path: Path
) -> None:
    plan, _, receipt_path = _published_fixture_plan(probe, tmp_path)
    checked = probe.validate_plan(plan, require_receipt_absent=True)

    assert checked["scope"] == "temporary_fixture"
    assert checked["source_owners"]["planner"]["path"].endswith(
        "src/packing/planner.py"
    )
    assert checked["source_owners"]["config_models"]["path"].endswith(
        "src/config/models.py"
    )
    assert checked["pack_plan_contract"]["schema"] == "coordexp-swift-pack-plan"
    assert checked["pack_plan_contract"]["schema_version"] >= 1
    assert checked["config"]["resolved_contract"]["train_order"] == "source_order"
    assert checked["config"]["resolved_contract"]["object_ordering"] == "geo_sorted"
    assert checked["config"]["resolved_contract"]["runtime_seed"] == 17
    assert checked["w0_cpu_encoded_stream"]["input_count"] == 12
    assert checked["w0_cpu_encoded_stream"]["encoded_length_sum"] == 60
    assert checked["w0_cpu_encoded_stream"]["identity_status"] == (
        "fixture_not_historical_w0"
    )
    assert checked["candidate_grid"] == [dict(item) for item in probe.ARM_GRID]
    assert checked["execution_contract"]["adaptive_thresholds"] is False
    assert checked["execution_contract"]["retry_count"] == 0
    assert checked["execution_contract"]["training"] == "forbidden"
    assert checked["execution_contract"]["model_loading"] == "forbidden"
    assert checked["execution_contract"]["gpu"] == "forbidden"
    assert checked["execution_contract"]["cache_writes"] == "forbidden"
    assert Path(checked["artifact_targets"]["receipt"]) == receipt_path


def test_plan_binds_production_integration_private_v3_and_source_quiescence(
    probe, tmp_path: Path
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)
    checked = probe.validate_plan(plan, require_receipt_absent=True)

    integration = checked["production_integration"]
    assert integration["pipeline_symbols"]["_materialize_pack_plan"]["path"].endswith(
        "src/training/pipeline.py"
    )
    assert integration["pipeline_symbols"]["_build_encoded_examples_for_dataset"][
        "sha256"
    ]
    assert integration["supervision_owner"]["path"].endswith(
        "src/packing/supervision.py"
    )
    private_v3 = integration["private_v3_cache"]
    assert private_v3["cache_version"] == "coordexp-swift-pack-cache-v3"
    assert private_v3["registry_schema_version"] >= 1
    assert private_v3["determinant_owners"]["packing_config"] == (
        "src/packing/planner.py"
    )
    assert private_v3["manifest_symbols"]["_validate_manifest"]["sha256"]
    assert private_v3["payload_symbols"]["_load_validated_chunk"]["sha256"]
    assert private_v3["payload_schema"]["class"] == "SupervisedMicroStep"
    quiescence = checked["preparation_source_quiescence"]
    assert quiescence["required"] is True
    assert quiescence["before_sha256"] == quiescence["after_sha256"]
    assert quiescence["stable"] is True
    bindings = checked["arm_v3_determinant_bindings"]
    assert set(bindings) == {arm["arm_id"] for arm in probe.ARM_GRID}
    for arm_id, worker_bindings in bindings.items():
        assert set(worker_bindings) == {"1", "8"}
        for binding in worker_bindings.values():
            assert binding["cache_version"] == "coordexp-swift-pack-cache-v3"
            assert binding["registry_schema_version"] >= 1
            assert binding["projection_fingerprint"]
            assert binding["packing"]["policy_identity"][
                "worker_count_disposition"
            ] == ("semantic_pending_upstream_materialization_equality")
        assert worker_bindings["1"] != worker_bindings["8"], arm_id

    changed = deepcopy(plan)
    changed["production_integration"]["pipeline_symbols"]["_materialize_pack_plan"][
        "sha256"
    ] = "0" * 64
    changed = _refinalize(probe, changed, "plan_sha256")
    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe.validate_plan(changed, require_receipt_absent=True)
    assert exc_info.value.code == "wave6.production_integration"


def test_real_arm_v3_aggregate_is_required_and_exactly_recomputed(
    probe, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)
    expected = plan["arm_v3_determinant_bindings"]["window_binpack_w8"]["1"]
    aggregate = "a" * 64
    full = {
        "version": expected["cache_version"],
        "registry_schema_version": expected["registry_schema_version"],
        "packing": deepcopy(expected["packing"]),
        "aggregate_fingerprint": aggregate,
    }
    monkeypatch.setattr(
        probe.pack_cache,
        "packing_cache_fingerprint_from_determinants",
        lambda determinants: determinants["aggregate_fingerprint"],
    )
    valid = {
        **deepcopy(expected),
        "full_aggregate_fingerprint": aggregate,
        "full_determinants": full,
    }
    probe._validate_observed_arm_v3_determinant(
        valid, expected_projection=expected, real_scope=True
    )

    missing = deepcopy(valid)
    missing["full_aggregate_fingerprint"] = None
    swapped_aggregate = deepcopy(valid)
    swapped_aggregate["full_aggregate_fingerprint"] = "b" * 64
    swapped_arm = deepcopy(valid)
    other = plan["arm_v3_determinant_bindings"]["window_binpack_w32"]["1"]
    swapped_arm["full_aggregate_fingerprint"] = "c" * 64
    swapped_arm["full_determinants"] = {
        "version": other["cache_version"],
        "registry_schema_version": other["registry_schema_version"],
        "packing": deepcopy(other["packing"]),
        "aggregate_fingerprint": "c" * 64,
    }
    swapped_worker = deepcopy(valid)
    other_worker = plan["arm_v3_determinant_bindings"]["window_binpack_w8"]["8"]
    swapped_worker["full_aggregate_fingerprint"] = "d" * 64
    swapped_worker["full_determinants"] = {
        "version": other_worker["cache_version"],
        "registry_schema_version": other_worker["registry_schema_version"],
        "packing": deepcopy(other_worker["packing"]),
        "aggregate_fingerprint": "d" * 64,
    }
    for changed in (missing, swapped_aggregate, swapped_arm, swapped_worker):
        with pytest.raises(probe.Wave6ProbeError) as exc_info:
            probe._validate_observed_arm_v3_determinant(
                changed,
                expected_projection=expected,
                real_scope=True,
            )
        assert exc_info.value.code == "wave6.arm_v3_determinant"


def test_fixture_controller_uses_three_fresh_process_pairs_and_real_encoder_seam(
    probe, tmp_path: Path
) -> None:
    plan, _, receipt_path = _published_fixture_plan(probe, tmp_path)

    probe.run_controller(plan["artifact_targets"]["plan"])
    persisted = probe.validate_controller_receipt(
        probe.load_strict_json(receipt_path), expected_plan=plan
    )

    assert persisted["pair_orders"] == [
        ["source_order_next_fit", "candidate"],
        ["candidate", "source_order_next_fit"],
        ["source_order_next_fit", "candidate"],
    ]
    assert len(persisted["observations"]) == 24
    assert len({item["process"]["pid"] for item in persisted["observations"]}) == 24
    for observation in persisted["observations"]:
        assert observation["materialization"]["production_encoder_seam"] == (
            "src.training.pipeline._build_encoded_examples_for_dataset"
        )
        assert observation["materialization"]["worker_counts"] == [1, 8]
        assert observation["materialization"]["encoded_materialization_exact"] is True
        resources = observation["process"]
        assert resources["wall_seconds"] >= 0.0
        assert resources["rss_delta_bytes"] >= 0
        assert resources["rss_high_water_bytes"] > 0
        arm = observation["arm"]
        assert arm["worker_comparison"]["semantic_order_exact"] is True
        assert arm["worker_comparison"]["full_plan_sha256_worker_1"]
        assert arm["worker_comparison"]["full_plan_sha256_worker_8"]
        assert arm["semantic_oracle"]["strict_replay_exact"] is True
        assert arm["resume_cursor_oracle"]["fragmented_resume_exact"] is True
        assert arm["boundedness"]["within_all_declared_bounds"] is True
        assert set(arm["current_v3_determinants"]) == {"1", "8"}
        for determinant in arm["current_v3_determinants"].values():
            assert determinant["cache_version"] == "coordexp-swift-pack-cache-v3"
            assert determinant["registry_schema_version"] >= 1
            assert determinant["projection_fingerprint"]

    assert len(persisted["paired_observations"]) == 12
    for pair in persisted["paired_observations"]:
        assert pair["order"] in (
            ["source_order_next_fit", "candidate"],
            ["candidate", "source_order_next_fit"],
        )
        assert pair["source_observation_sha256"]
        assert pair["candidate_observation_sha256"]
    aggregate = persisted["aggregate"]
    assert aggregate["accepted_pair_count_per_candidate"] == 3
    assert set(aggregate["candidates"]) == {arm["arm_id"] for arm in probe.ARM_GRID[1:]}
    for candidate in aggregate["candidates"].values():
        for role in ("source", "candidate"):
            for metric in (
                "wall_seconds",
                "rss_delta_bytes",
                "rss_high_water_bytes",
            ):
                assert len(candidate[role][metric]["observations"]) == 3
                assert candidate[role][metric]["range"] >= 0
        for metric in (
            "wall_seconds_delta",
            "rss_delta_bytes_delta",
            "rss_hwm_bytes_delta",
        ):
            assert len(candidate[metric]["observations"]) == 3
            assert candidate[metric]["range"] >= 0
    disposition = persisted["research_disposition"]
    assert disposition["planner_utilization_can_promote"] is False
    assert disposition["promotion_authorized"] is False
    assert disposition["smallest_matched_training"]["step_count"] == 5
    assert disposition["smallest_matched_training"]["reference_arm"] == (
        "source_order_next_fit__matched5step_reference"
    )

    mutations = []
    changed = deepcopy(persisted)
    changed["paired_observations"][0]["source_observation_sha256"] = "0" * 64
    mutations.append(
        (_refinalize(probe, changed, "controller_receipt_sha256"), "wave6.aggregate")
    )
    changed = deepcopy(persisted)
    first_candidate = probe.ARM_GRID[1]["arm_id"]
    changed["aggregate"]["candidates"][first_candidate]["wall_seconds_delta"][
        "median"
    ] += 1.0
    mutations.append(
        (_refinalize(probe, changed, "controller_receipt_sha256"), "wave6.aggregate")
    )
    changed = deepcopy(persisted)
    changed["research_disposition"]["planner_utilization_can_promote"] = True
    mutations.append(
        (
            _refinalize(probe, changed, "controller_receipt_sha256"),
            "wave6.research_disposition",
        )
    )
    changed = deepcopy(persisted)
    changed["paired_observations"][0]["ordering_change"]["moved_example_count"] = 999
    mutations.append(
        (
            _refinalize(probe, changed, "controller_receipt_sha256"),
            "wave6.aggregate",
        )
    )
    changed = deepcopy(persisted)
    changed["observations"][1]["arm"]["current_v3_determinants"]["1"] = deepcopy(
        changed["observations"][0]["arm"]["current_v3_determinants"]["1"]
    )
    changed["observations"][1] = _refinalize(
        probe, changed["observations"][1], "observation_sha256"
    )
    mutations.append(
        (
            _refinalize(probe, changed, "controller_receipt_sha256"),
            "wave6.arm_v3_determinant",
        )
    )
    changed = deepcopy(persisted)
    changed["observations"][0]["position_index"] = 1
    changed["observations"][0] = _refinalize(
        probe, changed["observations"][0], "observation_sha256"
    )
    mutations.append(
        (
            _refinalize(probe, changed, "controller_receipt_sha256"),
            "wave6.observation_coordinate",
        )
    )
    changed = deepcopy(persisted)
    changed["observations"][1]["process"]["pid"] = changed["observations"][0][
        "process"
    ]["pid"]
    changed["observations"][1] = _refinalize(
        probe, changed["observations"][1], "observation_sha256"
    )
    mutations.append(
        (
            _refinalize(probe, changed, "controller_receipt_sha256"),
            "wave6.controller_receipt",
        )
    )
    for changed_receipt, code in mutations:
        with pytest.raises(probe.Wave6ProbeError) as mutation_exc:
            probe.validate_controller_receipt(changed_receipt, expected_plan=plan)
        assert mutation_exc.value.code == code


@pytest.mark.parametrize("failure_stage", ["child", "planner"])
def test_child_or_planner_failure_publishes_one_bounded_immutable_terminal_receipt(
    probe, tmp_path: Path, failure_stage: str
) -> None:
    plan, _, receipt_path = _published_fixture_plan(probe, tmp_path)
    failure_path = Path(plan["artifact_targets"]["failure_receipt"])

    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe.run_controller(
            plan["artifact_targets"]["plan"],
            injected_failure={
                "candidate_arm_id": probe.ARM_GRID[1]["arm_id"],
                "repetition_index": 0,
                "position_index": 0,
                "stage": failure_stage,
            },
        )
    assert exc_info.value.code == "wave6.child_failure"
    assert not receipt_path.exists()
    failure = probe.validate_failure_receipt(
        probe.load_strict_json(failure_path), expected_plan=plan
    )
    assert failure["status"] == "failed_terminal"
    assert failure["completed_observation_sha256"] == []
    assert failure["retry_count"] == 0
    assert len(failure["error"]["message"]) <= probe.MAX_ERROR_CHARS

    for path, replacement in (
        (("retry_count",), 1),
        (("error", "message"), "x" * (probe.MAX_ERROR_CHARS + 1)),
        (("failed_coordinate", "arm_id"), probe.ARM_GRID[1]["arm_id"]),
    ):
        changed = deepcopy(failure)
        _set_path(changed, path, replacement)
        changed = _refinalize(probe, changed, "failure_receipt_sha256")
        with pytest.raises(probe.Wave6ProbeError) as mutation_exc:
            probe.validate_failure_receipt(changed, expected_plan=plan)
        assert mutation_exc.value.code == "wave6.failure_receipt"

    with pytest.raises(probe.Wave6ProbeError) as second_exc:
        probe.run_controller(plan["artifact_targets"]["plan"])
    assert second_exc.value.code == "wave6.immutable_collision"


@pytest.mark.parametrize(
    ("path", "replacement", "code"),
    [
        (("source_owners", "planner", "sha256"), "0" * 64, "wave6.source_owner"),
        (
            ("pack_plan_contract", "schema_version"),
            999,
            "wave6.pack_plan_contract",
        ),
        (
            ("config", "resolved_contract", "runtime_seed"),
            999,
            "wave6.config",
        ),
        (
            ("w0_cpu_encoded_stream", "encoded_lengths_sha256"),
            "0" * 64,
            "wave6.encoded_stream",
        ),
        (("candidate_grid", 1, "window_size"), 9, "wave6.candidate_grid"),
        (
            (
                "arm_v3_determinant_bindings",
                "window_binpack_w8",
                "1",
                "packing",
                "policy_identity",
                "window_size",
            ),
            9,
            "wave6.arm_v3_determinant",
        ),
        (
            ("execution_contract", "adaptive_thresholds"),
            True,
            "wave6.execution_contract",
        ),
        (
            ("execution_contract", "retry_count"),
            1,
            "wave6.execution_contract",
        ),
    ],
)
def test_plan_deep_mutations_fail_closed(
    probe,
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement,
    code: str,
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)
    changed = deepcopy(plan)
    _set_path(changed, path, replacement)
    changed = _refinalize(probe, changed, "plan_sha256")

    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe.validate_plan(changed, require_receipt_absent=True)

    assert exc_info.value.code == code


@pytest.mark.parametrize(
    ("path", "replacement", "code"),
    [
        (("arms", 0, "metrics", "utilization"), 0.125, "wave6.metrics"),
        (
            ("arms", 1, "semantic_oracle", "each_once_exact"),
            False,
            "wave6.semantic_oracle",
        ),
        (
            ("arms", 2, "ordering_change", "moved_example_count"),
            999,
            "wave6.ordering_metrics",
        ),
        (
            ("arms", 3, "resume_cursor_oracle", "fragmented_resume_exact"),
            False,
            "wave6.resume_cursor",
        ),
        (
            (
                "arms",
                4,
                "worker_equality",
                "exact_semantic_projection_equal",
            ),
            False,
            "wave6.worker_equality",
        ),
        (
            ("research_disposition", "planner_utilization_can_promote"),
            True,
            "wave6.research_disposition",
        ),
        (
            ("research_disposition", "smallest_matched_training", "step_count"),
            6,
            "wave6.research_disposition",
        ),
    ],
)
def test_receipt_deep_mutations_fail_closed(
    probe,
    tmp_path: Path,
    path: tuple[str | int, ...],
    replacement,
    code: str,
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)
    receipt = probe.execute_comparison(plan, _examples())
    changed = deepcopy(receipt)
    _set_path(changed, path, replacement)
    changed = _refinalize(probe, changed, "receipt_sha256")

    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe.validate_receipt(changed, expected_plan=plan)

    assert exc_info.value.code == code


def test_stream_or_row_mutation_is_rejected_before_comparison(
    probe, tmp_path: Path
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)
    changed_tokens = list(_examples())
    changed_tokens[0] = replace(changed_tokens[0], input_ids=(999,) * 6)
    changed_rows = list(_examples())
    changed_rows[0] = replace(changed_rows[0], row_ids=("changed",))

    for changed in (changed_tokens, changed_rows):
        with pytest.raises(probe.Wave6ProbeError) as exc_info:
            probe.execute_comparison(plan, changed)
        assert exc_info.value.code == "wave6.encoded_stream_identity"


def test_plan_and_receipt_targets_must_both_be_absent(probe, tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    receipt_path = tmp_path / "receipt.json"
    plan_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe.build_fixture_plan(
            _examples(),
            plan_path=plan_path,
            receipt_path=receipt_path,
            global_max_length=10,
        )
    assert exc_info.value.code == "wave6.immutable_collision"

    plan_path.unlink()
    receipt_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe.build_fixture_plan(
            _examples(),
            plan_path=plan_path,
            receipt_path=receipt_path,
            global_max_length=10,
        )
    assert exc_info.value.code == "wave6.immutable_collision"


def test_receipt_names_only_surviving_changed_order_matched_five_step_arms(
    probe, tmp_path: Path
) -> None:
    plan, _, _ = _published_fixture_plan(probe, tmp_path)
    receipt = probe.execute_comparison(plan, _examples())

    disposition = receipt["research_disposition"]
    assert disposition["planner_utilization_can_promote"] is False
    assert disposition["promotion_authorized"] is False
    assert disposition["optimization_dynamics_unmeasured"] is True
    followup = disposition["smallest_matched_training"]
    assert followup["step_count"] == 5
    assert followup["reference_arm"] == "source_order_next_fit__matched5step_reference"
    expected = {
        f"{arm['arm_id']}__matched5step_candidate"
        for arm in receipt["arms"]
        if arm["arm_id"] != "source_order_next_fit"
        and arm["ordering_change"]["moved_example_count"] > 0
        and arm["cpu_disposition"] == "survives_cpu_gate_requires_matched_training"
    }
    assert set(followup["candidate_arms"]) == expected
    assert followup["only_changed_factor"] == (
        "packing_policy_and_its_declared_window_or_lookahead"
    )
    assert followup["identical_fields"] == [
        "images",
        "intra_image_rows_and_order",
        "encoded_tokens_and_supervision",
        "optimizer_budget",
        "five_step_schedule",
        "evaluation",
        "checkpoint_policy",
        "artifact_semantics",
    ]


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values(
    probe, tmp_path: Path
) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema":"a","schema":"b"}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")

    for path in (duplicate, nonfinite):
        with pytest.raises(probe.Wave6ProbeError) as exc_info:
            probe.load_strict_json(path)
        assert exc_info.value.code == "wave6.json_read"


def test_encoded_materialization_canonicalizes_qwen_serialization_state(probe) -> None:
    plan = probe.qwen_images.QwenNoResizeImagePlan(
        example_id="canonical-fixture",
        image_path=Path("fixture.png"),
        width=28,
        height=28,
        patch_size=14,
        merge_size=2,
        temporal_patch_size=2,
        required_spatial_factor=28,
        raw_pixels=784,
        raw_patch_rows=4,
        expected_pixel_values_width=1176,
        image_grid_thw=(1, 2, 2),
        merged_visual_tokens=1,
        max_raw_pixels=784,
        max_merged_visual_tokens=1,
    )
    encoding = probe.qwen_images.QwenImageEncoding(
        plan=plan,
        pixel_values=None,
        image_grid_thw_tensor=None,
        image_processor=object(),
    )

    canonical = probe._canonicalize_encoded_materialization((encoding,))

    assert encoding.image_processor is not None
    assert canonical[0].image_processor is None
    assert canonical[0].plan == encoding.plan


def test_child_failure_diagnostic_keeps_typed_stderr_tail(probe) -> None:
    stderr = (
        "tokenizers fork warning\n" * 1_000
        + "Traceback (most recent call last):\n"
        + '  File "probe.py", line 1, in <module>\n'
        + "Wave6ProbeError: independent encoded materialization differs\n"
    )

    error = probe._child_process_error(returncode=1, stdout="", stderr=stderr)
    bounded = probe._bounded_error(error)

    assert type(error).__name__ == "Wave6ChildProcessError"
    assert error.code == "wave6.child_failure"
    assert "child_exception_type=Wave6ProbeError" in str(error)
    assert "independent encoded materialization differs" in str(error)
    assert len(str(error)) <= probe.MAX_ERROR_CHARS
    assert bounded["type"] == "Wave6ChildProcessError"
    assert "independent encoded materialization differs" in bounded["message"]


def _research_meaning_sources(probe, tmp_path: Path):
    plan_path = tmp_path / "immutable-plan.json"
    receipt_path = tmp_path / "immutable-controller-receipt.json"
    plan = probe.finalize_artifact(
        {"schema": probe.PLAN_SCHEMA, "status": "prepared"},
        hash_field="plan_sha256",
    )
    probe.publish_json_absent(plan_path, plan)

    observations = []
    pairs = []
    for candidate_index, spec in enumerate(probe.CANDIDATE_GRID):
        candidate_arm_id = str(spec["arm_id"])
        for repetition_index, order in enumerate(probe.POLICY_PAIR_ORDERS):
            source_sha = probe.sha256_json(
                [candidate_arm_id, repetition_index, "source"]
            )
            candidate_sha = probe.sha256_json(
                [candidate_arm_id, repetition_index, "candidate"]
            )
            observations.extend(
                [
                    {"observation_sha256": source_sha},
                    {"observation_sha256": candidate_sha},
                ]
            )
            moved = 0 if candidate_arm_id == "window_binpack_w8" else 12
            pairs.append(
                {
                    "candidate_arm_id": candidate_arm_id,
                    "repetition_index": repetition_index,
                    "order": list(order),
                    "source_pid": 10_000 + candidate_index * 10 + repetition_index,
                    "candidate_pid": (20_000 + candidate_index * 10 + repetition_index),
                    "source_observation_sha256": source_sha,
                    "candidate_observation_sha256": candidate_sha,
                    "ordering_change": {
                        "flattened_order_sha256": probe.sha256_json(
                            [candidate_arm_id, repetition_index]
                        ),
                        "moved_example_count": moved,
                        "absolute_displacement_sum": moved,
                        "absolute_displacement_max": 0 if moved == 0 else 2,
                        "co_present_pair_count": 3,
                        "co_present_pairs_sha256": probe.sha256_json(
                            [candidate_arm_id, "pairs"]
                        ),
                        "added_vs_reference_count": 0 if moved == 0 else 1,
                        "removed_vs_reference_count": 0 if moved == 0 else 1,
                        "retained_vs_reference_count": 3 if moved == 0 else 2,
                    },
                    "resource_deltas": {
                        "wall_seconds": (1.0, 10.0, 2.0)[repetition_index],
                        "rss_delta_bytes": (10, 100, 20)[repetition_index],
                        "rss_high_water_bytes": (20, 200, 40)[repetition_index],
                    },
                    "planner_deltas": {
                        "utilization": (0.0, -0.1, 0.2)[repetition_index],
                        "pack_count": (0, 1, 2)[repetition_index],
                        "tail_waste": (0, 12_000, 24_000)[repetition_index],
                    },
                }
            )
    receipt = probe.finalize_artifact(
        {
            "schema": probe.CONTROLLER_RECEIPT_SCHEMA,
            "status": "completed_cpu_only_training_required",
            "finished_at": "2026-08-10T00:00:00+00:00",
            "plan_path": str(plan_path),
            "plan_sha256": plan["plan_sha256"],
            "plan_file_sha256": probe.sha256_file(plan_path),
            "pair_orders": [list(order) for order in probe.POLICY_PAIR_ORDERS],
            "observations": observations,
            "paired_observations": pairs,
            "aggregate": {},
            "research_disposition": {},
        },
        hash_field="controller_receipt_sha256",
    )
    probe.publish_json_absent(receipt_path, receipt)
    binding = {
        "plan_sha256": plan["plan_sha256"],
        "plan_file_sha256": probe.sha256_file(plan_path),
        "controller_receipt_sha256": receipt["controller_receipt_sha256"],
        "controller_receipt_file_sha256": probe.sha256_file(receipt_path),
    }
    return plan_path, receipt_path, binding, pairs


def test_research_meaning_packet_recomputes_paired_mad_and_disposition(
    probe, tmp_path: Path
) -> None:
    plan_path, receipt_path, binding, pairs = _research_meaning_sources(probe, tmp_path)

    packet = probe._build_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=binding,
    )
    checked = probe._validate_research_meaning_packet_from_binding(
        packet,
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=binding,
    )

    assert checked["raw_paired_deltas"] == pairs
    w8_wall = checked["paired_metric_summaries"]["window_binpack_w8"][
        "wall_seconds_delta"
    ]
    assert w8_wall == {
        "observations": [1.0, 10.0, 2.0],
        "median": 2.0,
        "minimum": 1.0,
        "maximum": 10.0,
        "median_absolute_deviation": 1.0,
    }
    assert checked["research_disposition"]["semantic_survivors"] == [
        "window_binpack_w32",
        "online_window_binpack_l8",
        "online_window_binpack_l32",
    ]
    assert checked["research_disposition"]["non_survivors"] == ["window_binpack_w8"]
    assert (
        checked["research_disposition"]["smallest_matched_training"]["step_count"] == 5
    )
    assert checked["research_disposition"]["planner_utilization_can_promote"] is False


@pytest.mark.parametrize("mutation", ["missing_mad", "wrong_mad", "survivor", "hash"])
def test_research_meaning_packet_mutations_fail_closed(
    probe, tmp_path: Path, mutation: str
) -> None:
    plan_path, receipt_path, binding, _ = _research_meaning_sources(probe, tmp_path)
    packet = probe._build_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        expected_binding=binding,
    )
    changed = deepcopy(packet)
    summary = changed["paired_metric_summaries"]["window_binpack_w8"][
        "wall_seconds_delta"
    ]
    if mutation == "missing_mad":
        summary.pop("median_absolute_deviation")
    elif mutation == "wrong_mad":
        summary["median_absolute_deviation"] += 1.0
    elif mutation == "survivor":
        changed["research_disposition"]["semantic_survivors"].append(
            "window_binpack_w8"
        )
    else:
        changed["source_artifacts"]["controller_receipt"]["file_sha256"] = "0" * 64
    changed = _refinalize(probe, changed, "research_meaning_packet_sha256")

    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe._validate_research_meaning_packet_from_binding(
            changed,
            plan_path=plan_path,
            receipt_path=receipt_path,
            expected_binding=binding,
        )
    assert exc_info.value.code == "wave6.research_meaning"


def test_research_meaning_publication_is_absent_and_r2_bound(
    probe, tmp_path: Path
) -> None:
    plan_path, receipt_path, binding, _ = _research_meaning_sources(probe, tmp_path)
    target = tmp_path / "research-meaning.packet.json"

    packet = probe._publish_research_meaning_packet_from_binding(
        plan_path=plan_path,
        receipt_path=receipt_path,
        packet_path=target,
        expected_binding=binding,
    )
    assert probe.load_strict_json(target) == packet
    with pytest.raises(probe.Wave6ProbeError) as exc_info:
        probe._publish_research_meaning_packet_from_binding(
            plan_path=plan_path,
            receipt_path=receipt_path,
            packet_path=target,
            expected_binding=binding,
        )
    assert exc_info.value.code == "wave6.immutable_collision"
    assert probe.R2_RESEARCH_MEANING_BINDING == {
        "plan_sha256": (
            "e890e6a60ebf25e56f928fb1f7d96f602c2000787d1d4b7133acd1d4f78d1dab"
        ),
        "plan_file_sha256": (
            "6e358950ab5b14ef94e493ddda57d955dd5e4200d6466483529380b1e4eb1478"
        ),
        "controller_receipt_sha256": (
            "d243c0d3af1b490f6ed768fad42f733e89d03eb010969a3dc3dc92804a0ec587"
        ),
        "controller_receipt_file_sha256": (
            "e6b2bfc6c50a904b1f2c2910ac27c68d9d8fbda75b54440157b5b83febdc5888"
        ),
    }
