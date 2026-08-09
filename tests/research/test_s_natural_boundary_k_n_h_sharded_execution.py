from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.research.analyze_s_natural_boundary_k_n_h_evidence import (
    analyze_cohort,
    document_self_sha256,
)
from scripts.research import run_s_natural_boundary_k_n_h_shard as shard_runner
from scripts.research import s_natural_boundary_k_n_h_live_executor as canonical_live
from scripts.research.plan_s_natural_boundary_k_n_h_execution import (
    CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT,
    ExecutionPlanError,
    build_plan,
    validate_plan,
)
from scripts.research.run_s_natural_boundary_k_n_h_cohort import (
    ARM_ORDER,
    canonical_json_bytes,
    sha256_json,
    validate_manifest,
)
from scripts.research.run_s_natural_boundary_k_n_h_shard import (
    ShardExecutionError,
    _run_shard_test_only,
    _validate_existing_event,
)
from scripts.research.merge_s_natural_boundary_k_n_h_shards import (
    ShardMergeError,
    _merge_shards_test_only,
    merge_shards,
)


_FIXTURE_PATH = Path(__file__).with_name("test_run_s_natural_boundary_k_n_h_cohort.py")
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location("_s_cohort_fixture", _FIXTURE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_FIXTURE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_FIXTURE)
_SEALER_FIXTURE_PATH = Path(__file__).with_name("test_seal_s_natural_boundary_k_n_h_pre_gpu_receipt.py")
_SEALER_SPEC = importlib.util.spec_from_file_location("_s_cohort_sealer_fixture", _SEALER_FIXTURE_PATH)
assert _SEALER_SPEC is not None and _SEALER_SPEC.loader is not None
_SEALER_FIXTURE = importlib.util.module_from_spec(_SEALER_SPEC)
_SEALER_SPEC.loader.exec_module(_SEALER_FIXTURE)
_ANALYZER_FIXTURE_PATH = Path(__file__).with_name(
    "test_analyze_s_natural_boundary_k_n_h_evidence.py"
)
_ANALYZER_SPEC = importlib.util.spec_from_file_location(
    "_s_cohort_analyzer_fixture", _ANALYZER_FIXTURE_PATH
)
assert _ANALYZER_SPEC is not None and _ANALYZER_SPEC.loader is not None
_ANALYZER_FIXTURE = importlib.util.module_from_spec(_ANALYZER_SPEC)
_ANALYZER_SPEC.loader.exec_module(_ANALYZER_FIXTURE)


def _executor_identity_for_binding(binding: dict[str, object]) -> dict[str, object]:
    assignment = binding["device_assignment"]
    assert isinstance(assignment, dict)
    shard_index = int(assignment["shard_index"])
    identity = _FIXTURE._executor_identity(shard_index)
    identity["pre_gpu"] = copy.deepcopy(binding)
    observed = identity["observed"]
    assert isinstance(observed, dict)
    model = observed["model"]
    assert isinstance(model, dict)
    input_paths = binding["input_paths"]
    input_hashes = binding["input_hashes"]
    assert isinstance(input_paths, dict) and isinstance(input_hashes, dict)
    model.update(
        {
            "h0_dir": input_paths["h0_dir"],
            "base_model_path": input_paths["base_model_dir"],
            "h0_identity_files_sha256": input_hashes["h0_identity_files_sha256"],
            "base_model_files_sha256": input_hashes["base_model_files_sha256"],
            "base_model_inventory_sha256": input_hashes["base_model_inventory_sha256"],
        }
    )
    observed["config_sha256"] = input_hashes["config_sha256"]
    runtime = binding["runtime"]
    assert isinstance(runtime, dict)
    observed["runtime_versions"] = {
        key: runtime[key]
        for key in ("python_version", "torch_version", "transformers_version")
    }
    identity.pop("identity_sha256", None)
    identity["identity_sha256"] = sha256_json(identity)
    return identity


def _executor_with_identity(identity: dict[str, object]) -> object:
    def execute(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        result = _FIXTURE._executor(event, arm_order=arm_order)
        for arm in arm_order:
            result["arms"][arm]["executor_identity"] = copy.deepcopy(identity)
        return result

    return execute


def _analyzer_executor_with_identity(identity: dict[str, object]) -> object:
    def execute(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        kinds = {arm: "other" for arm in arm_order}
        kinds.update(
            {
                "K10": "target",
                "K11": "malformed",
                "K12": "unmatched",
                "K13": "duplicate",
                "K14T": "target",
                "K14B": "other",
                "N10": "invalid",
                "N20": "native_stop",
            }
        )
        arms = {
            arm: _ANALYZER_FIXTURE._arm_result(arm, event, kind=kinds[arm])
            for arm in arm_order
        }
        for result in arms.values():
            result["executor_identity"] = copy.deepcopy(identity)
        return {"arms": arms}

    return execute


def _configure_sibling_final_root(
    kwargs: dict[str, object],
    paths: dict[str, object],
) -> None:
    shards_root = Path(paths["shard_roots"][0]).parent
    final_root = shards_root.parent / "cohort-merged"
    kwargs["final_merge_root"] = final_root
    paths["final_root"] = final_root


def _rewrite_canonical_census_for_one_event(paths: dict[str, object]) -> None:
    rows: list[dict[str, object]] = [
        {
            "checkpoint": "S",
            "gt_owner_id": f"gt:5001:{owner_index}",
            "image_id": 5001,
            "source_panel_object_index": owner_index,
            "derived_panel_object_index": owner_index,
        }
        for owner_index in (14, 15, 19)
    ]
    filler = 0
    while len(rows) < 392:
        rows.append(
            {
                "checkpoint": "S",
                "gt_owner_id": f"gt:{9000 + filler // 100}:{100 + filler}",
                "image_id": 9000 + filler // 100,
                "source_panel_object_index": 100 + filler,
                "derived_panel_object_index": 100 + filler,
            }
        )
        filler += 1
    rows.extend(
        {
            "checkpoint": "A",
            "gt_owner_id": f"gt:{12000 + index // 100}:{index}",
            "image_id": 12000 + index // 100,
            "source_panel_object_index": index,
            "derived_panel_object_index": index,
        }
        for index in range(392)
    )
    census: dict[str, object] = {
        "schema_version": "natural_boundary_owner_admission_census.v3",
        "status": "sealed",
        "unit_id": _SEALER_FIXTURE.sealer.UNIT_ID,
        "census_revision": "census-v3",
        "rows": rows,
    }
    census["self_sha256"] = sha256_json(census)
    Path(paths["census"]).write_bytes(canonical_json_bytes(census) + b"\n")


def _upgrade_event_with_live_k13_geometry(event: dict[str, object]) -> None:
    """Bind the canonical one-event fixture to the production K13 contract."""

    image_id = event["image_id"]
    cells = [1]
    weights = [{"cell_index": 1, "overlap_fraction": 1.0}]
    event["same_class_competitor_owner_id"] = f"gt:{image_id}:20"
    event["image_cell_regions"] = {"same_class_competitor": cells}
    event["image_cell_region_receipts"] = {
        "same_class_competitor": {
            "status": "available",
            "available": True,
            "not_measured_reason": None,
            "cell_indices": cells,
            "visual_indices": cells,
            "fractional_weights": weights,
            "weight_sum": 1.0,
            "cell_count": 1,
            "cell_indices_sha256": sha256_json(cells),
            "weights_sha256": sha256_json(weights),
        }
    }


def _truncate_canonical_fixture_to_one_event(
    kwargs: dict[str, object],
    paths: dict[str, object],
) -> None:
    _rewrite_canonical_census_for_one_event(paths)
    manifest_path = Path(paths["manifest"])
    context_path = Path(paths["cohort"])
    companion_path = Path(paths["cohort_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["events"] = manifest["events"][:1]
    manifest["event_count"] = 1
    manifest["image_count"] = 1
    _upgrade_event_with_live_k13_geometry(manifest["events"][0])
    natural_boundary = manifest["events"][0]["natural_boundary"]
    natural_boundary["prefix_token_ids"] = [50, 9]
    natural_boundary["prefix_sha256"] = sha256_json([50, 9])
    natural_boundary["history_token_ids"] = [9]
    natural_boundary["history_sha256"] = sha256_json([9])
    manifest["events"][0].pop("event_sha256", None)
    manifest["events"][0]["event_sha256"] = sha256_json(manifest["events"][0])
    context = json.loads(context_path.read_text(encoding="utf-8"))
    owner_ids = [manifest["events"][0]["event_id"]]
    context["events"] = [
        event for event in context["events"] if event.get("gt_owner_id") in owner_ids
    ]
    context_path.write_bytes(canonical_json_bytes(context) + b"\n")
    companion = json.loads(companion_path.read_text(encoding="utf-8"))
    companion["cohort_sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    companion["event_count"] = 1
    companion["owner_ids"] = owner_ids
    companion["owner_ids_sha256"] = sha256_json(owner_ids)
    companion_path.write_bytes(canonical_json_bytes(companion) + b"\n")
    legacy = manifest["legacy_context_cohort"]
    legacy["sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()
    legacy["manifest_sha256"] = hashlib.sha256(companion_path.read_bytes()).hexdigest()
    legacy["event_count"] = 1
    manifest["source_census"] = {
        "revision": "census-v3",
        "path": str(paths["census"]),
        "sha256": hashlib.sha256(Path(paths["census"]).read_bytes()).hexdigest(),
        "hash_semantics": "canonical_json_document_with_trailing_newline",
    }
    manifest.pop("self_sha256", None)
    manifest["self_sha256"] = sha256_json(manifest)
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    plan_path = Path(paths["execution_plan"])
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    event_ref = {
        key: manifest["events"][0][key]
        for key in ("event_index", "event_id", "image_id", "event_sha256")
    }
    plan["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    plan["manifest_self_sha256"] = manifest["self_sha256"]
    plan["event_count"] = 1
    plan["distinct_image_count"] = 1
    plan["events"] = [event_ref]
    plan["scalar_forward_upper_bound_total"] = CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT
    plan["gate_empirical_scalar_forward_estimate_total"] = 301
    if "wall_time_estimate_seconds_total" in plan:
        plan["wall_time_estimate_seconds_total"] = None
    plan["claim_scope"] = {
        "execution_scope": "case_study",
        "event_count": 1,
        "image_count": 1,
        "minimum_checkpoint_event_count": 3,
        "minimum_checkpoint_image_count": 2,
        "checkpoint_claim_qualified": False,
        "static_direction_claim_qualified": False,
        "training_claim_qualified": False,
        "subfloor_execution_authorized": True,
    }
    for index, shard in enumerate(plan["shards"]):
        refs = [event_ref] if index == 0 else []
        shard["event_indices"] = [0] if index == 0 else []
        shard["events"] = refs
        shard["event_count"] = len(refs)
        shard["distinct_image_count"] = len(refs)
        shard["scalar_forward_upper_bound"] = len(refs) * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT
        if "wall_time_estimate_seconds" in shard:
            shard["wall_time_estimate_seconds"] = None
    plan.pop("plan_sha256", None)
    plan["plan_sha256"] = sha256_json(plan)
    plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")
    cohort_identity = kwargs["cohort"]
    assert isinstance(cohort_identity, dict)
    cohort_identity["sha256"] = hashlib.sha256(context_path.read_bytes()).hexdigest()


def _gate() -> dict[str, object]:
    counts = {arm: (27 if index < 11 else 1) for index, arm in enumerate(ARM_ORDER)}
    document: dict[str, object] = {
        "schema_version": "s_primary_natural_boundary_gate.v1",
        "unit_id": "2026-08-06-natural-boundary-routing-history-replication",
        "checkpoint": "S",
        "event_id": "gt:5001:15",
        "gpu_launch_authorized": False,
        "no_training": True,
        "arm_order": list(ARM_ORDER),
        "arms": {
            arm: {"scalar_forward_count": count, "runtime_scalar_forward_count": count}
            for arm, count in counts.items()
        },
    }
    document["result_sha256"] = sha256_json(document)
    return document


def _plan(tmp_path: Path, *, event_count: int = 16) -> tuple[dict[str, object], dict[str, object]]:
    manifest = _FIXTURE._manifest(tmp_path, event_count=event_count)
    return manifest, build_plan(manifest, _gate())


def _path_plan(tmp_path: Path, *, event_count: int = 16) -> tuple[Path, Path]:
    manifest = _FIXTURE._manifest(tmp_path, event_count=event_count)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    plan = build_plan(manifest_path, _gate())
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")
    return manifest_path, plan_path


def test_plan_is_deterministic_and_binds_full_manifest_order(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    manifest_info = validate_manifest(manifest)
    validated = validate_plan(plan, manifest_info=manifest_info)
    assert validated["plan_sha256"] == plan["plan_sha256"]
    assert len(plan["shards"]) == 8
    assert plan["events"] == [
        {
            "event_index": event["event_index"],
            "event_id": event["event_id"],
            "image_id": event["image_id"],
            "event_sha256": event["event_sha256"],
        }
        for event in manifest["events"]
    ]
    assert plan["gate_scalar_forward_count"] == 301
    assert plan["scalar_forward_upper_bound_total"] == 16 * CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT
    assert plan["gate_empirical_scalar_forward_estimate_per_event"] == 301
    assert plan["wall_time_estimate_status"] == "unavailable_in_gate_receipt"
    assert [ref["event_index"] for ref in plan["shards"][0]["events"]] == [0, 8]


def test_plan_rejects_tampered_binding(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    tampered = copy.deepcopy(plan)
    tampered["events"][0]["event_id"] = "gt:9999:1"
    tampered["plan_sha256"] = sha256_json({key: value for key, value in tampered.items() if key != "plan_sha256"})
    with pytest.raises(ExecutionPlanError):
        validate_plan(tampered, manifest_info=validate_manifest(manifest))


def test_eight_shards_resume_and_merge_with_actual_scalar_receipts(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    shards_root = tmp_path / "shards"
    for index in range(8):
        result = _run_shard_test_only(
            manifest,
            plan,
            shards_root / f"shard-{index:03d}",
            shard_id=index,
            executor=_FIXTURE._executor,
        )
        assert result["aggregate"]["event_count"] == 2
        assert result["aggregate"]["scalar_forward_count"] == 30

    merged = _merge_shards_test_only(manifest, plan, shards_root, tmp_path / "merged")
    assert merged["aggregate"]["event_count"] == 16
    assert merged["aggregate"]["scalar_forward_count"] == 16 * 15
    assert merged["aggregate"]["execution_qualification"]["checkpoint_claim_qualified"] is True

    def should_not_execute(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("completed event roots must resume without invoking executor")

    for index in range(8):
        _run_shard_test_only(
            manifest,
            plan,
            shards_root / f"shard-{index:03d}",
            shard_id=index,
            executor=should_not_execute,
        )


def test_actual_above_empirical_301_is_accepted_below_contract_max(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path, event_count=3)

    def executor(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        result = _FIXTURE._executor(event, arm_order=arm_order)
        arm = result["arms"]["K00"]
        count = 302
        arm["scalar_forward_count"] = count
        arm["scalar_receipts"] = [{"step": index, "finite": True, "use_cache": False} for index in range(count)]
        arm["runtime_scalar_forward_count"] = count
        arm["runtime_scalar_receipts"] = [
            {"step": index, "finite": True, "use_cache": False} for index in range(count)
        ]
        arm["full_logit_parity"]["reference_step_count"] = count
        return result

    result = _run_shard_test_only(manifest, plan, tmp_path / "shard-000", shard_id=0, executor=executor)
    assert result["aggregate"]["scalar_forward_count"] == 302 + len(ARM_ORDER) - 1
    assert result["aggregate"]["scalar_forward_count"] > 301
    assert result["aggregate"]["scalar_forward_count"] <= CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT


def test_contract_scalar_ceiling_is_11520() -> None:
    assert CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT == 15 * 3 * 256 == 11520
    event = {"event_index": 0, "event_id": "gt:5001:15", "image_id": 5001, "event_sha256": "e" * 64}
    arms = {
        arm: {"scalar_forward_count": 1, "runtime_scalar_forward_count": 1}
        for arm in ARM_ORDER
    }
    arms["K00"]["scalar_forward_count"] = CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT - len(ARM_ORDER) + 2
    arms["K00"]["runtime_scalar_forward_count"] = CONTRACT_MAX_SCALAR_FORWARD_PER_EVENT - len(ARM_ORDER) + 2
    document = {
        "plan_sha256": "p" * 64,
        "shard_id": "shard-000",
        "shard_index": 0,
        "result_sha256": "r" * 64,
        "arms": arms,
        "executor_identity": _FIXTURE._executor_identity(),
    }
    with pytest.raises(ShardExecutionError, match="contract scalar-forward bound"):
        _validate_existing_event(
            document,
            event=event,
            plan_sha256="p" * 64,
            shard="shard-000",
            shard_index=0,
        )


@pytest.mark.parametrize(
    ("script", "module"),
    (
        ("plan_s_natural_boundary_k_n_h_execution.py", "scripts.research.plan_s_natural_boundary_k_n_h_execution"),
        ("run_s_natural_boundary_k_n_h_shard.py", "scripts.research.run_s_natural_boundary_k_n_h_shard"),
        ("merge_s_natural_boundary_k_n_h_shards.py", "scripts.research.merge_s_natural_boundary_k_n_h_shards"),
    ),
)
def test_cli_help_works_as_direct_file_and_module(script: str, module: str) -> None:
    direct = subprocess.run(
        [sys.executable, str(_REPO_ROOT / "scripts" / "research" / script), "--help"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert direct.returncode == 0, direct.stderr
    module_run = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert module_run.returncode == 0, module_run.stderr


def test_shard_runner_import_is_cpu_only_before_prelaunch() -> None:
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import scripts.research.run_s_natural_boundary_k_n_h_shard; "
                "assert 'torch' not in sys.modules; "
                "assert 'scripts.research.s_natural_boundary_k_n_h_live_executor' not in sys.modules"
            ),
        ],
        cwd=_REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr


def test_prelaunch_rejects_preexisting_shard_root_before_identity_or_executor(tmp_path: Path) -> None:
    root = tmp_path / "shard-000"
    root.mkdir()
    with pytest.raises(ShardExecutionError, match="must be absent"):
        shard_runner.validate_prelaunch(
            tmp_path / "missing-manifest.json",
            tmp_path / "missing-plan.json",
            root,
            shard_id=0,
        )


def test_canonical_prelaunch_receipt_authorizes_exact_two_event_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, paths = _SEALER_FIXTURE._fixture(tmp_path)
    document = _SEALER_FIXTURE.sealer.build_receipt(**kwargs)
    _SEALER_FIXTURE.sealer.seal_receipt(document)
    environment = {
        shard_runner.ENV_MANIFEST: paths["manifest"],
        shard_runner.ENV_CENSUS: paths["census"],
        shard_runner.ENV_CONFIG: paths["config"],
        shard_runner.ENV_PANEL: paths["panel"],
        shard_runner.ENV_COHORT: paths["cohort"],
        shard_runner.ENV_COHORT_MANIFEST: paths["cohort_manifest"],
        shard_runner.ENV_H0_ROOT: paths["h0_root"],
        shard_runner.ENV_H0_DIR: paths["h0_dir"],
        shard_runner.ENV_PRE_GPU_RECEIPT: paths["output"],
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = shard_runner.validate_prelaunch(
        paths["manifest"],
        paths["execution_plan"],
        paths["shard_roots"][0],
        shard_id="shard-000",
    )
    assert result.shard_id == "shard-000"
    assert document["authorized_shards"][0]["event_indices"] == [0, 8]
    assert result.authorization_sha256 == document["authorized_shards"][0]["authorization_sha256"]
    assert not paths["shard_roots"][0].exists()


def test_prelaunch_rejects_wrong_physical_device_and_missing_companion_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, paths = _SEALER_FIXTURE._fixture(tmp_path)
    document = _SEALER_FIXTURE.sealer.build_receipt(**kwargs)
    _SEALER_FIXTURE.sealer.seal_receipt(document)
    environment = {
        shard_runner.ENV_MANIFEST: paths["manifest"],
        shard_runner.ENV_CENSUS: paths["census"],
        shard_runner.ENV_CONFIG: paths["config"],
        shard_runner.ENV_PANEL: paths["panel"],
        shard_runner.ENV_COHORT: paths["cohort"],
        shard_runner.ENV_H0_ROOT: paths["h0_root"],
        shard_runner.ENV_H0_DIR: paths["h0_dir"],
        shard_runner.ENV_PRE_GPU_RECEIPT: paths["output"],
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, str(value))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    with pytest.raises(ShardExecutionError, match=shard_runner.ENV_COHORT_MANIFEST):
        shard_runner.validate_prelaunch(
            paths["manifest"], paths["execution_plan"], paths["shard_roots"][0], shard_id=0
        )
    monkeypatch.setenv(shard_runner.ENV_COHORT_MANIFEST, str(paths["cohort_manifest"]))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(ShardExecutionError, match="CUDA_VISIBLE_DEVICES|authorization"):
        shard_runner.validate_prelaunch(
            paths["manifest"], paths["execution_plan"], paths["shard_roots"][0], shard_id=0
        )
    assert not paths["shard_roots"][0].exists()


def test_prelaunch_runs_once_before_executor_import_and_two_same_shard_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path, plan_path = _path_plan(tmp_path, event_count=16)
    output_root = tmp_path / "shard-000"
    calls: list[str] = []

    def prelaunch(
        manifest: str | Path,
        plan: str | Path,
        output: str | Path,
        *,
        shard_id: str | int,
    ) -> shard_runner._PrelaunchContext:
        assert not Path(output).exists()
        assert Path(manifest) == manifest_path
        assert Path(plan) == plan_path
        assert shard_id == 0
        calls.append("prelaunch")
        manifest_info = validate_manifest(manifest_path)
        plan_info = validate_plan(plan_path, manifest_info=manifest_info)
        return shard_runner._PrelaunchContext(
            manifest_path=str(manifest_path),
            manifest_sha256=manifest_info["manifest_sha256"],
            plan_path=str(plan_path),
            plan_file_sha256=plan_info["source"]["sha256"],
            plan_sha256=plan_info["plan_sha256"],
            output_root=str(output_root),
            shard_id="shard-000",
            shard_index=0,
            receipt_path=str(tmp_path / "receipt.json"),
            receipt_self_sha256="r" * 64,
            authorization_sha256="a" * 64,
            runtime_identity=_FIXTURE._executor_identity()["pre_gpu"],
            runtime_versions=_FIXTURE._executor_identity()["observed"]["runtime_versions"],
            observed_cuda_visible_devices="0",
        )

    def load(*, pre_gpu_identity: object, runtime_versions: object) -> object:
        assert pre_gpu_identity == _FIXTURE._executor_identity()["pre_gpu"]
        assert runtime_versions == _FIXTURE._executor_identity()["observed"]["runtime_versions"]
        assert calls == ["prelaunch"]
        calls.append("executor_import")
        return _FIXTURE._executor

    monkeypatch.setattr(shard_runner, "validate_prelaunch", prelaunch)
    monkeypatch.setattr(shard_runner, "load_executor", load)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    result = shard_runner.run_shard_from_spec(
        manifest_path,
        plan_path,
        output_root,
        shard_id=0,
    )
    assert calls == ["prelaunch", "executor_import"]
    assert result["aggregate"]["event_count"] == 2


def test_guarded_runner_has_no_public_injected_executor_bypass() -> None:
    assert not hasattr(shard_runner, "run_shard")
    assert "executor" not in {action.dest for action in shard_runner.build_parser()._actions}


def test_canonical_executor_hash_and_runtime_version_drift_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _FIXTURE._executor_identity()["pre_gpu"]
    runtime_versions = _FIXTURE._executor_identity()["observed"]["runtime_versions"]
    tampered = copy.deepcopy(identity)
    tampered["code_hashes"]["live_executor"] = "0" * 64
    with pytest.raises(ShardExecutionError, match="code hash"):
        shard_runner.load_executor(
            pre_gpu_identity=tampered,
            runtime_versions=runtime_versions,
        )
    monkeypatch.setattr(
        shard_runner.importlib.metadata,
        "version",
        lambda package: "drifted" if package == "torch" else runtime_versions[f"{package}_version"],
    )
    with pytest.raises(ShardExecutionError, match="runtime versions"):
        shard_runner._runtime_version_attestation(identity)


def test_public_merge_accepts_exact_per_shard_identities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, paths = _SEALER_FIXTURE._fixture(tmp_path)
    _configure_sibling_final_root(kwargs, paths)
    document = _SEALER_FIXTURE.sealer.build_receipt(**kwargs)
    _SEALER_FIXTURE.sealer.seal_receipt(document)
    receipt = json.loads(Path(paths["output"]).read_text(encoding="utf-8"))
    identities = [
        _SEALER_FIXTURE.sealer.runtime_identity_binding(
            receipt,
            receipt_path=paths["output"],
            shard_id=index,
            observed_cuda_visible_devices=str(index),
        )
        for index in range(8)
    ]
    for index, binding in enumerate(identities):
        shard_runner._run_shard_core(
            paths["manifest"],
            paths["execution_plan"],
            paths["shard_roots"][index],
            shard_id=index,
            executor=_executor_with_identity(_executor_identity_for_binding(binding)),
            expected_pre_gpu_identity=binding,
        )
    full_validations = 0
    original_binding = _SEALER_FIXTURE.sealer.runtime_identity_binding

    def counted_binding(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal full_validations
        full_validations += 1
        return original_binding(*args, **kwargs)

    monkeypatch.setattr(_SEALER_FIXTURE.sealer, "runtime_identity_binding", counted_binding)
    shards_root = Path(paths["shard_roots"][0]).parent
    nested_final_root = shards_root / "merged"
    with pytest.raises(ShardMergeError, match="outside the exact shards root"):
        merge_shards(
            paths["manifest"],
            paths["execution_plan"],
            shards_root,
            nested_final_root,
            pre_gpu_receipt=paths["output"],
        )
    assert not nested_final_root.exists()
    result = merge_shards(
        paths["manifest"],
        paths["execution_plan"],
        shards_root,
        paths["final_root"],
        pre_gpu_receipt=paths["output"],
    )
    assert full_validations == 1
    assert set(result["aggregate"]["executor_identities"]) == {
        f"shard-{index:03d}" for index in range(8)
    }
    for index in range(8):
        shard = f"shard-{index:03d}"
        assert result["aggregate"]["executor_identities"][shard]["pre_gpu"] == identities[index]


def test_canonical_receipt_public_runner_and_merge_one_event_case_study(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kwargs, paths = _SEALER_FIXTURE._fixture(tmp_path)
    _configure_sibling_final_root(kwargs, paths)
    _truncate_canonical_fixture_to_one_event(kwargs, paths)
    document = _SEALER_FIXTURE.sealer.build_receipt(**kwargs)
    _SEALER_FIXTURE.sealer.seal_receipt(document)
    current_identity: dict[str, object] = {}

    def configure(
        identity: dict[str, object],
        *,
        runtime_versions: dict[str, str],
    ) -> None:
        assert runtime_versions == {
            key: identity["runtime"][key]
            for key in ("python_version", "torch_version", "transformers_version")
        }
        current_identity.clear()
        current_identity.update(_executor_identity_for_binding(identity))

    def execute(event: dict[str, object], *, arm_order: tuple[str, ...]) -> dict[str, object]:
        assert current_identity
        return _analyzer_executor_with_identity(current_identity)(event, arm_order=arm_order)

    monkeypatch.setattr(canonical_live, "configure_pre_gpu_identity", configure)
    monkeypatch.setattr(canonical_live, "execute_event", execute)
    environment = {
        shard_runner.ENV_MANIFEST: paths["manifest"],
        shard_runner.ENV_CENSUS: paths["census"],
        shard_runner.ENV_CONFIG: paths["config"],
        shard_runner.ENV_PANEL: paths["panel"],
        shard_runner.ENV_COHORT: paths["cohort"],
        shard_runner.ENV_COHORT_MANIFEST: paths["cohort_manifest"],
        shard_runner.ENV_H0_ROOT: paths["h0_root"],
        shard_runner.ENV_H0_DIR: paths["h0_dir"],
        shard_runner.ENV_PRE_GPU_RECEIPT: paths["output"],
    }
    for name, value in environment.items():
        monkeypatch.setenv(name, str(value))
    empty_shards: list[str] = []
    for index in range(8):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(index))
        shard_result = shard_runner.run_shard_from_spec(
            paths["manifest"],
            paths["execution_plan"],
            paths["shard_roots"][index],
            shard_id=index,
        )
        if index == 0:
            assert shard_result["aggregate"]["executor_identity"] is not None
        else:
            assert shard_result["aggregate"]["event_count"] == 0
            assert shard_result["aggregate"]["executor_identity"] is None
            assert shard_result["aggregate"]["pre_gpu_identity"] is not None
            empty_shards.append(f"shard-{index:03d}")
    assert empty_shards == [f"shard-{index:03d}" for index in range(1, 8)]
    shards_root = Path(paths["shard_roots"][0]).parent
    assert Path(paths["final_root"]).parent == shards_root.parent
    assert Path(paths["final_root"]).parent != shards_root
    merged = merge_shards(
        paths["manifest"],
        paths["execution_plan"],
        shards_root,
        paths["final_root"],
        pre_gpu_receipt=paths["output"],
    )
    scope = merged["aggregate"]["claim_scope"]
    assert scope["execution_scope"] == "case_study"
    assert scope["event_count"] == 1
    assert scope["image_count"] == 1
    assert scope["checkpoint_claim_qualified"] is False
    assert scope["training_claim_qualified"] is False
    assert set(merged["aggregate"]["pre_gpu_identities"]) == {
        f"shard-{index:03d}" for index in range(8)
    }
    assert set(merged["aggregate"]["executor_identities"]) == {"shard-000"}
    evidence_path = tmp_path / "fixture" / "evidence" / "evidence.json"
    evidence_receipt_path = evidence_path.with_name("evidence.receipt.json")
    gate_artifacts = kwargs["gate_v3_artifacts"]
    assert isinstance(gate_artifacts, dict)
    analyzed = analyze_cohort(
        Path(paths["final_root"]) / "aggregate.json",
        paths["manifest"],
        paths["census"],
        plan=paths["execution_plan"],
        gate_receipt=gate_artifacts["result"],
        shards_root=shards_root,
        output=evidence_path,
        receipt_output=evidence_receipt_path,
    )
    evidence = analyzed["evidence"]
    receipt = analyzed["receipt"]
    assert evidence["status"] == "complete"
    assert evidence["claim_scope"] == scope
    assert evidence["claim_decision"] == {
        "execution_scope": "case_study",
        "analyzed_event_count": 1,
        "all_case_events_analyzed": True,
        "checkpoint_claim_qualified": False,
        "static_direction_claim_qualified": False,
        "training_claim_qualified": False,
        "checkpoint_claim_status": "hold",
        "static_direction_claim_status": "hold",
        "training_claim_status": "hold",
    }
    assert evidence["qualification"]["scope_gate"] == {
        "execution_scope": "case_study",
        "checkpoint_scope_open": False,
        "static_direction_scope_open": False,
        "training_scope_open": False,
    }
    assert evidence["self_sha256"] == document_self_sha256(evidence)
    assert receipt["status"] == "complete"
    assert receipt["claim_scope"] == scope
    assert receipt["checkpoint_level_status"] == "hold"
    assert receipt["self_sha256"] == document_self_sha256(receipt)
    assert receipt["evidence_raw_file_sha256"] == hashlib.sha256(
        evidence_path.read_bytes()
    ).hexdigest()
    assert all(
        receipt["next_step_flags"][key] is False
        for key in ("authorize_crossover", "authorize_a3", "authorize_p4", "authorize_training")
    )
    assert evidence_path.is_file()
    assert evidence_receipt_path.is_file()


def test_individual_shards_need_not_meet_full_execution_gate(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path, event_count=3)
    shards_root = tmp_path / "shards"
    for index in range(8):
        _run_shard_test_only(
            manifest,
            plan,
            shards_root / f"shard-{index:03d}",
            shard_id=index,
            executor=_FIXTURE._executor,
        )
    merged = _merge_shards_test_only(manifest, plan, shards_root, tmp_path / "merged")
    assert merged["aggregate"]["execution_qualification"]["event_count"] == 3


def test_shard_and_merge_reject_foreign_roots(tmp_path: Path) -> None:
    manifest, plan = _plan(tmp_path)
    shard_root = tmp_path / "shard-000"
    shard_root.mkdir()
    (shard_root / "foreign.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ShardExecutionError):
        _run_shard_test_only(manifest, plan, shard_root, shard_id=0, executor=_FIXTURE._executor)

    shards_root = tmp_path / "shards"
    shards_root.mkdir()
    for index in range(8):
        (shards_root / f"shard-{index:03d}").mkdir()
    (shards_root / "foreign").mkdir()
    with pytest.raises(ShardMergeError):
        _merge_shards_test_only(manifest, plan, shards_root, tmp_path / "merged")
