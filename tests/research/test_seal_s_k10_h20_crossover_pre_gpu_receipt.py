from __future__ import annotations

import copy
from importlib import metadata
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import materialize_s_k10_h20_crossover_plan as planner
from scripts.research import seal_s_k10_h20_crossover_pre_gpu_receipt as sealer

BASE = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/2026-08-06-natural-boundary-routing-history-replication"
)
EVIDENCE = BASE / "s-k-n-h-evidence-native-fn-supersession-v3/evidence.json"
RECEIPT = BASE / "s-k-n-h-evidence-native-fn-supersession-v3/evidence.receipt.json"
MANIFEST = BASE / "cpu-census-v3-native-fn-supersession-v1/admitted-event-manifest.json"
CENSUS = BASE / "cpu-census-v3-native-fn-supersession-v1/admission-census.json"
ORIGINAL_PLAN = BASE / "execution-plan-native-fn-supersession-v1/plan.json"
GATE = BASE / "s-gt5001-live-gate-v3/result.json"


def _plan(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    planner.build_plan(EVIDENCE, RECEIPT, MANIFEST, CENSUS, ORIGINAL_PLAN, GATE, path)
    return path


def _json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(sealer.canonical_json_bytes(value) + b"\n")
    return path


def _fixture(tmp_path: Path) -> dict[str, Any]:
    code_dir = tmp_path / "code"
    test_dir = tmp_path / "tests"
    code = {role: (code_dir / f"{role}.py") for role in sealer.CODE_ROLE_PATHS}
    tests = {role: (test_dir / f"{role}.py") for role in sealer.TEST_PATHS}
    for path in [*code.values(), *tests.values()]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# {path.name}\n", encoding="utf-8")
    old_code = sealer.CODE_ROLE_PATHS.copy()
    old_tests = sealer.TEST_PATHS.copy()
    sealer.CODE_ROLE_PATHS.clear()
    sealer.CODE_ROLE_PATHS.update(code)
    sealer.TEST_PATHS.clear()
    sealer.TEST_PATHS.update(tests)
    config = tmp_path / "config.yaml"
    panel = tmp_path / "panel.jsonl"
    cohort = tmp_path / "cohort.json"
    h0_root = tmp_path / "h0-root"
    h0 = h0_root / "h0-dir"
    h0.mkdir(parents=True)
    base_model_dir = tmp_path / "base-model"
    base_model_dir.mkdir()
    config.write_text("model: S\n", encoding="utf-8")
    panel.write_text("{}\n", encoding="utf-8")
    cohort.write_text("{}\n", encoding="utf-8")
    h0_identity_files = {}
    for role in sealer.H0_IDENTITY_FILE_ROLES:
        path = h0 / f"{role}.bin"
        path.write_text(f"{role}\n", encoding="utf-8")
        h0_identity_files[role] = path
    base_model_files = {}
    for role in sealer.BASE_MODEL_FILE_ROLES:
        path = base_model_dir / f"{role}.bin"
        path.write_text(f"{role}\n", encoding="utf-8")
        base_model_files[role] = path
    cohort_manifest = _json(
        tmp_path / "cohort-manifest.json",
        {"schema_version": "cohort_manifest.v1", "status": "sealed"},
    )
    focused = _json(
        tmp_path / "focused.json",
        {
            "schema_version": "s_k10_h20_crossover_focused_cpu_evidence.v1",
            "status": "passed",
            "cpu_only": True,
            "gpu_used": False,
            "model_loaded": False,
            "focused_tests": [],
        },
    )
    probe = _json(
        tmp_path / "probe.json",
        {
            "schema_version": sealer.PRE_GPU_PROBE_SCHEMA_VERSION,
            "status": "passed",
            "unit_id": sealer.UNIT_ID,
            "no_gpu_launch": True,
            "composition": {"status": "passed"},
            "installed_qwen_consumption": {
                "status": "passed",
                "all_layer_consumption": True,
                "layer_consumption": {"passed": True, "all_layers_identical": True},
                "block23_sdpa_mass_receipt": {"status": "passed"},
            },
        },
    )
    runtime = {
        "python_version": f"{__import__('sys').version_info.major}.{__import__('sys').version_info.minor}",
        "torch_version": metadata.version("torch"),
        "transformers_version": metadata.version("transformers"),
    }
    return {
        "old_code": old_code,
        "old_tests": old_tests,
        "code": code,
        "tests": tests,
        "config": config,
        "panel": panel,
        "cohort": cohort,
        "h0": h0,
        "h0_root": h0_root,
        "h0_dir": h0,
        "base_model_dir": base_model_dir,
        "h0_identity_files": h0_identity_files,
        "base_model_files": base_model_files,
        "manifest": MANIFEST,
        "census": CENSUS,
        "cohort_manifest": cohort_manifest,
        "focused": focused,
        "probe": probe,
        "runtime": runtime,
    }


def _consumption() -> dict[str, Any]:
    return {
        "required": True,
        "status": "unattested",
        "exact_same_tensor_all_layers_required": True,
        "declared_layer_count": 28,
    }


def _source_preflight(
    fixture: dict[str, Any],
    plan_path: Path,
    execution_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    source_values = {
        "manifest": fixture["manifest"],
        "census": fixture["census"],
        "execution_plan": plan_path,
        "config": fixture["config"],
        "panel": fixture["panel"],
        "cohort": fixture["cohort"],
        "cohort_manifest": fixture["cohort_manifest"],
        "h0_root": fixture["h0_root"],
        "h0_dir": fixture["h0_dir"],
        "base_model_dir": fixture["base_model_dir"],
    }
    source_bindings = {
        role: {
            "path": ref["path"],
            "sha256": ref["sha256"],
            "kind": ref["kind"],
        }
        for role, value in source_values.items()
        for ref in [sealer._path_ref(value, f"test source {role}")]
    }
    code_bindings = {
        role: {
            "path": str(Path(path).resolve()),
            "sha256": sealer.sha256_file(path),
        }
        for role, path in fixture["code"].items()
    }
    test_bindings = {
        role: {
            "path": str(Path(path).resolve()),
            "sha256": sealer.sha256_file(path),
        }
        for role, path in fixture["tests"].items()
    }
    manifest = json.loads(Path(fixture["manifest"]).read_text(encoding="utf-8"))
    manifest_events = manifest["events"]
    all_event_ids = [event["event_id"] for event in manifest_events]
    admitted_identities = [
        (
            "S",
            event["owner_refs"]["gt_owner_id"],
            event["image_id"],
            event["owner_refs"]["source_panel_object_index"],
            event["owner_refs"]["derived_panel_object_index"],
        )
        for event in manifest_events
    ]
    processor_bindings = [{"event_id": event_id} for event_id in all_event_ids]
    cohort_body = {
        "status": "passed",
        "event_count": 11,
        "event_identities_sha256": sealer.sha256_json(admitted_identities),
        "authoritative_bindings_sha256": "2" * 64,
        "processor_context_bindings": processor_bindings,
        "processor_context_bindings_sha256": sealer.sha256_json(processor_bindings),
        "cohort_path": str(Path(fixture["cohort"]).resolve()),
        "cohort_sha256": sealer.sha256_file(fixture["cohort"]),
        "cohort_manifest_path": str(Path(fixture["cohort_manifest"]).resolve()),
        "cohort_manifest_sha256": sealer.sha256_file(fixture["cohort_manifest"]),
    }
    full_runtime_cohort = cohort_body | {
        "receipt_sha256": sealer.sha256_json(cohort_body)
    }
    metadata_body = {
        "status": "passed",
        "source": "sealed_base_model_config_metadata_only",
        "base_model_dir": str(Path(fixture["base_model_dir"]).resolve()),
        "base_model_inventory_sha256": source_bindings["base_model_dir"]["sha256"],
        "config_path": str(Path(fixture["base_model_files"]["config"]).resolve()),
        "config_raw_sha256": sealer.sha256_file(
            fixture["base_model_files"]["config"]
        ),
        "layer_count": 28,
        "head_count": 16,
        "device": "cpu",
        "executable_model_present": False,
        "model_loader_called": False,
    }
    factory_metadata = metadata_body | {
        "receipt_sha256": sealer.sha256_json(metadata_body)
    }
    contexts = []
    for event in plan["events"]:
        k14 = {"K14T": [2], "K14B": [3]}
        contracts = {
            arm: {
                "protocol": "natural_step_actuator_factory.v1",
                "arm_id": arm,
                "image_key_positions_sha256": "3" * 64,
                "b_exclusive_positions_sha256": (
                    sealer.sha256_json(k14["K14T"])
                    if arm == "K14T"
                    else "4" * 64
                ),
                "latest_row_key_positions_sha256": "5" * 64,
                "layer_count": 28,
                "head_count": 16,
                "device": "cpu",
            }
            for arm in sealer.SOURCE_PREFLIGHT_ATTENTION_ARMS
        }
        built = {
            arm: {
                "status": "ready",
                "mask_sha256": "6" * 64,
                "selected_positions": [2],
                "layer_consumption_attestation": _consumption(),
                "all_layer_consumption_attestation": _consumption(),
                "no_op_parity": (
                    {"required": True, "status": "unassessed"}
                    if arm == "K01"
                    else {}
                ),
                "receipt_sha256": "7" * 64,
            }
            for arm in ("K01", "K10", "H20")
        }
        c11 = {
            "unit_id": sealer.UNIT_ID,
            "arm_id": "C11",
            "cell_id": "C11",
            "status": "ready",
            "component_order": ["K10", "H20"],
            "component_changed_cell_union": True,
            "exact_scope": True,
            "children": [{"arm_id": "K10"}, {"arm_id": "H20"}],
            "layer_consumption_attestation": _consumption() | {"passed": False},
            "all_layer_consumption_attestation": _consumption() | {"passed": False},
        }
        contexts.append(
            {
                "event_id": event["event_id"],
                "event_index": event["event_index"],
                "image_id": event["image_id"],
                "exact_history_sha256": event["prefix_sha256"],
                "seeded_prefix_sha256": "9" * 64,
                "natural_context_sha256": "a" * 64,
                "natural_identity_sha256": "b" * 64,
                "attention_factory_arms": list(sealer.SOURCE_PREFLIGHT_ATTENTION_ARMS),
                "attention_factory_contracts": contracts,
                "built_factory_receipts": built,
                "k14_reference_positions": k14,
                "c11_receipt": c11,
            }
        )
    production_body = {
        "status": "passed",
        "cpu_contract": {
            "status": "passed",
            "event_count": 11,
            "identity_sha256": "c" * 64,
        },
        "full_runtime_cohort": full_runtime_cohort,
        "processor_only": {
            "status": "passed",
            "load_model": False,
            "model_present": False,
            "backend_session_opened": False,
            "identity_sha256": "d" * 64,
        },
        "factory_model_metadata": factory_metadata,
        "selected_factory_contexts": contexts,
        "selected_event_count": 3,
        "gpu_used": False,
        "model_loaded": False,
        "model_loader_called": False,
        "output_root_created": False,
    }
    production = production_body | {
        "receipt_sha256": sealer.sha256_json(production_body)
    }
    body = {
        "schema_version": sealer.SOURCE_PREFLIGHT_SCHEMA_VERSION,
        "status": "passed",
        "phase": "preseal_model_free_production",
        "unit_id": sealer.UNIT_ID,
        "plan_binding": {
            "path": str(plan_path.resolve()),
            "raw_sha256": sealer.sha256_file(plan_path),
            "self_sha256": plan["self_sha256"],
        },
        "source_bindings": source_bindings,
        "code_bindings": code_bindings,
        "test_bindings": test_bindings,
        "reserved_roots": {
            "execution_root": {
                "path": str(execution_root.resolve()),
                "status": "reserved_absent_pre_gpu",
            },
            "final_root": {
                "path": str(final_root.resolve()),
                "status": "reserved_absent_pre_gpu",
            },
        },
        "model_free_production_path": production,
        "gpu_used": False,
        "model_loaded": False,
        "backend_session_opened": False,
        "output_root_created": False,
        "receipt_independent": True,
    }
    return body | {"self_sha256": sealer.sha256_json(body)}


def _probe_with_source_preflight(
    fixture: dict[str, Any],
    plan_path: Path,
    execution_root: Path,
    final_root: Path,
    *,
    name: str = "probe-with-source-preflight.json",
) -> Path:
    probe = json.loads(Path(fixture["probe"]).read_text(encoding="utf-8"))
    probe["source_preflight"] = _source_preflight(
        fixture, plan_path, execution_root, final_root
    )
    return _json(plan_path.parent / name, probe)


def _seal(
    fixture: dict[str, Any],
    plan: Path,
    probe: Path,
    execution_root: Path,
    final_root: Path,
    output: Path,
) -> dict[str, Any]:
    return sealer.seal_pre_gpu_receipt(
        plan,
        {
            "path": str(sealer.UNIT_PATH),
            "unit_id": sealer.UNIT_ID,
            "status": "active",
            "scope": "one fixed launch",
        },
        fixture["focused"],
        probe,
        execution_root,
        final_root,
        output,
        config=fixture["config"],
        panel=fixture["panel"],
        cohort=fixture["cohort"],
        h0=fixture["h0"],
        manifest=fixture["manifest"],
        census=fixture["census"],
        execution_plan=plan,
        cohort_manifest=fixture["cohort_manifest"],
        h0_root=fixture["h0_root"],
        h0_dir=fixture["h0_dir"],
        base_model_dir=fixture["base_model_dir"],
        h0_identity_files=fixture["h0_identity_files"],
        base_model_files=fixture["base_model_files"],
        runtime=fixture["runtime"],
        source_files=fixture["code"],
        test_files=fixture["tests"],
    )


def _rehash_source_preflight(
    source_preflight: dict[str, Any],
    *,
    cohort: bool = False,
    metadata: bool = False,
    production: bool = True,
) -> None:
    production_receipt = source_preflight.get("model_free_production_path")
    if isinstance(production_receipt, dict):
        if cohort:
            cohort_receipt = production_receipt["full_runtime_cohort"]
            cohort_receipt["processor_context_bindings_sha256"] = (
                sealer.sha256_json(cohort_receipt["processor_context_bindings"])
            )
            cohort_receipt["receipt_sha256"] = sealer.sha256_json(
                {
                    key: value
                    for key, value in cohort_receipt.items()
                    if key != "receipt_sha256"
                }
            )
        if metadata:
            metadata_receipt = production_receipt["factory_model_metadata"]
            metadata_receipt["receipt_sha256"] = sealer.sha256_json(
                {
                    key: value
                    for key, value in metadata_receipt.items()
                    if key != "receipt_sha256"
                }
            )
        if production:
            production_receipt["receipt_sha256"] = sealer.sha256_json(
                {
                    key: value
                    for key, value in production_receipt.items()
                    if key != "receipt_sha256"
                }
            )
    source_preflight["self_sha256"] = sealer.document_self_sha256(source_preflight)


def _deep_source_validation_kwargs(
    fixture: dict[str, Any],
    plan_path: Path,
    execution_root: Path,
    final_root: Path,
) -> dict[str, Any]:
    plan_ref, plan_doc = sealer._validate_plan(plan_path)
    source_values = {
        "manifest": fixture["manifest"],
        "census": fixture["census"],
        "execution_plan": plan_path,
        "config": fixture["config"],
        "panel": fixture["panel"],
        "cohort": fixture["cohort"],
        "cohort_manifest": fixture["cohort_manifest"],
        "h0_root": fixture["h0_root"],
        "h0_dir": fixture["h0_dir"],
        "base_model_dir": fixture["base_model_dir"],
    }
    return {
        "plan_ref": plan_ref,
        "plan_doc": plan_doc,
        "manifest_doc": json.loads(
            Path(fixture["manifest"]).read_text(encoding="utf-8")
        ),
        "input_refs": {
            role: sealer._path_ref(value, f"test source {role}")
            for role, value in source_values.items()
        },
        "code_refs": sealer._validate_code(fixture["code"]),
        "test_refs": sealer._validate_tests(fixture["tests"]),
        "roots": sealer._validate_roots(execution_root, final_root),
    }


def test_probe_rejects_not_exercised_source_preflight(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        probe = copy.deepcopy(__import__("json").loads(fixture["probe"].read_text(encoding="utf-8")))
        probe["source_preflight"] = {"status": "not_exercised"}
        probe_path = _json(tmp_path / "not-exercised-probe.json", probe)
        with pytest.raises(sealer.PreGpuReceiptError, match="source preflight"):
            sealer._validate_probe(probe_path)
    finally:
        sealer.CODE_ROLE_PATHS.clear()
        sealer.CODE_ROLE_PATHS.update(fixture["old_code"])
        sealer.TEST_PATHS.clear()
        sealer.TEST_PATHS.update(fixture["old_tests"])


def test_source_preflight_deep_validation_rejects_shallow_tampered_or_exercised_claims(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)
    try:
        plan = _plan(tmp_path)
        execution_root = tmp_path / "execution"
        final_root = tmp_path / "final"
        valid = _source_preflight(fixture, plan, execution_root, final_root)
        validation_kwargs = _deep_source_validation_kwargs(
            fixture, plan, execution_root, final_root
        )
        assert sealer._validate_source_preflight(valid, **validation_kwargs) == valid

        cases = (
            ("shallow", "fields"),
            ("status", "passing, preseal"),
            ("phase", "passing, preseal"),
            ("receipt_independent", "passing, preseal"),
            ("unit", "passing, preseal"),
            ("self_tamper", "self_sha256"),
            ("plan_binding", "plan binding"),
            ("source_binding", "input bindings"),
            ("code_binding", "code bindings"),
            ("test_binding", "test bindings"),
            ("reserved_root", "reserved roots"),
            ("production_hash", "production receipt_sha256"),
            ("cohort_binding", "cohort/manifest bindings"),
            ("cohort_membership", "event order/membership"),
            ("cohort_identity_hash", "event identity hash"),
            ("cohort_event_count", "eleven-event"),
            ("selected_order", "selected event order/count"),
            ("selected_history", "exact plan prefix"),
            ("outer_gpu", "gpu_used must be false"),
            ("production_model", "model_loaded must be false"),
            ("processor_session", "backend_session_opened must be false"),
            ("factory_model", "model-free boundary"),
            ("built_missing", "built K01/K10/H20"),
            ("built_fake_pass", "required and unattested"),
            ("k14t_mismatch", "K14T positions differ"),
            ("k14b_empty", "K14B positions are empty"),
            ("c11_order", r"exact K10\+H20"),
            ("c11_fake_pass", "required and unattested"),
            ("receipt_cycle", "receipt cycle"),
        )
        for case, expected_error in cases:
            candidate = copy.deepcopy(valid)
            rehash_cohort = False
            rehash_metadata = False
            rehash_production = True
            if case == "shallow":
                candidate = {
                    key: candidate[key]
                    for key in (
                        "schema_version",
                        "status",
                        "phase",
                        "unit_id",
                        "gpu_used",
                        "model_loaded",
                        "backend_session_opened",
                        "output_root_created",
                        "receipt_independent",
                    )
                }
                rehash_production = False
            elif case == "status":
                candidate["status"] = "not_exercised"
            elif case == "phase":
                candidate["phase"] = "postseal"
            elif case == "receipt_independent":
                candidate["receipt_independent"] = False
            elif case == "unit":
                candidate["unit_id"] = "wrong-unit"
            elif case == "self_tamper":
                candidate["self_sha256"] = "0" * 64
                rehash_production = False
            elif case == "plan_binding":
                candidate["plan_binding"]["raw_sha256"] = "0" * 64
            elif case == "source_binding":
                candidate["source_bindings"]["config"]["sha256"] = "0" * 64
            elif case == "code_binding":
                candidate["code_bindings"]["crossover_runner"]["sha256"] = (
                    "0" * 64
                )
            elif case == "test_binding":
                candidate["test_bindings"]["crossover_runner_test"]["sha256"] = (
                    "0" * 64
                )
            elif case == "reserved_root":
                candidate["reserved_roots"]["execution_root"]["path"] = str(
                    (tmp_path / "other-execution").resolve()
                )
            elif case == "production_hash":
                candidate["model_free_production_path"]["receipt_sha256"] = "0" * 64
                rehash_production = False
            elif case == "cohort_binding":
                candidate["model_free_production_path"]["full_runtime_cohort"][
                    "cohort_sha256"
                ] = "0" * 64
                rehash_cohort = True
            elif case == "cohort_membership":
                bindings = candidate["model_free_production_path"][
                    "full_runtime_cohort"
                ]["processor_context_bindings"]
                nonselected_index = next(
                    index
                    for index, binding in enumerate(bindings)
                    if binding["event_id"] not in planner.EVENT_IDS
                )
                bindings[nonselected_index]["event_id"] = "fake:unique:event"
                rehash_cohort = True
            elif case == "cohort_identity_hash":
                candidate["model_free_production_path"]["full_runtime_cohort"][
                    "event_identities_sha256"
                ] = "0" * 64
                rehash_cohort = True
            elif case == "cohort_event_count":
                candidate["model_free_production_path"]["full_runtime_cohort"][
                    "event_count"
                ] = 10
                rehash_cohort = True
            elif case == "selected_order":
                candidate["model_free_production_path"][
                    "selected_factory_contexts"
                ].reverse()
            elif case == "selected_history":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["exact_history_sha256"] = "0" * 64
            elif case == "outer_gpu":
                candidate["gpu_used"] = True
            elif case == "production_model":
                candidate["model_free_production_path"]["model_loaded"] = True
            elif case == "processor_session":
                candidate["model_free_production_path"]["processor_only"][
                    "backend_session_opened"
                ] = True
            elif case == "factory_model":
                candidate["model_free_production_path"]["factory_model_metadata"][
                    "executable_model_present"
                ] = True
                rehash_metadata = True
            elif case == "built_missing":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["built_factory_receipts"].pop("H20")
            elif case == "built_fake_pass":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["built_factory_receipts"]["K10"][
                    "layer_consumption_attestation"
                ]["passed"] = True
            elif case == "k14t_mismatch":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["k14_reference_positions"]["K14T"] = [4]
            elif case == "k14b_empty":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["k14_reference_positions"]["K14B"] = []
            elif case == "c11_order":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["c11_receipt"]["component_order"] = ["H20", "K10"]
            elif case == "c11_fake_pass":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["c11_receipt"]["all_layer_consumption_attestation"][
                    "passed"
                ] = True
            elif case == "receipt_cycle":
                candidate["model_free_production_path"]["selected_factory_contexts"][
                    0
                ]["pre_gpu_receipt_sha256"] = "0" * 64
            else:  # pragma: no cover - closed case table
                raise AssertionError(case)

            if case != "self_tamper":
                _rehash_source_preflight(
                    candidate,
                    cohort=rehash_cohort,
                    metadata=rehash_metadata,
                    production=rehash_production,
                )
            with pytest.raises(sealer.PreGpuReceiptError, match=expected_error):
                sealer._validate_source_preflight(candidate, **validation_kwargs)
    finally:
        sealer.CODE_ROLE_PATHS.clear()
        sealer.CODE_ROLE_PATHS.update(fixture["old_code"])
        sealer.TEST_PATHS.clear()
        sealer.TEST_PATHS.update(fixture["old_tests"])


def test_sealer_fails_cleanly_until_runner_and_finalizer_exist(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        plan = _plan(tmp_path)
        execution_root = tmp_path / "execution"
        final_root = tmp_path / "final"
        probe = _probe_with_source_preflight(
            fixture, plan, execution_root, final_root
        )
        fixture["code"]["crossover_runner"].unlink()
        with pytest.raises(sealer.PreGpuReceiptError, match="source file"):
            _seal(
                fixture,
                plan,
                probe,
                execution_root,
                final_root,
                tmp_path / "receipt.json",
            )
    finally:
        sealer.CODE_ROLE_PATHS.clear()
        sealer.CODE_ROLE_PATHS.update(fixture["old_code"])
        sealer.TEST_PATHS.clear()
        sealer.TEST_PATHS.update(fixture["old_tests"])


def test_sealer_binds_canonical_cpu_evidence_roots_devices_and_self(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = _fixture(tmp_path)
    try:
        plan = _plan(tmp_path)
        execution_root = tmp_path / "execution"
        final_root = tmp_path / "final"
        probe = _probe_with_source_preflight(
            fixture, plan, execution_root, final_root
        )
        receipt_path = tmp_path / "receipt.json"
        document = _seal(
            fixture,
            plan,
            probe,
            execution_root,
            final_root,
            receipt_path,
        )
        assert document["schema_version"] == sealer.SCHEMA_VERSION
        assert document["self_sha256"] == sealer.document_self_sha256(document)
        assert document["device_plan"] == sealer.DEVICE_PLAN
        assert document["test_files"]["attention_actuator_test"]["path"] == str(
            fixture["tests"]["attention_actuator_test"]
        )
        assert document["source_preflight_document"] == json.loads(
            probe.read_text(encoding="utf-8")
        )["source_preflight"]
        assert sealer.validate_pre_gpu_receipt(receipt_path)["self_sha256"] == document["self_sha256"]
        # Same bytes are idempotent; a symlink output is never followed.
        _seal(
            fixture,
            plan,
            probe,
            execution_root,
            final_root,
            receipt_path,
        )
        assert receipt_path.is_file()
        fixture["tests"]["attention_actuator_test"].write_text(
            "# drifted actuator test\n", encoding="utf-8"
        )
        with pytest.raises(sealer.PreGpuReceiptError, match="test file|source/test"):
            sealer.validate_pre_gpu_receipt(receipt_path)
    finally:
        sealer.CODE_ROLE_PATHS.clear()
        sealer.CODE_ROLE_PATHS.update(fixture["old_code"])
        sealer.TEST_PATHS.clear()
        sealer.TEST_PATHS.update(fixture["old_tests"])


def test_receipt_root_or_device_tamper_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        plan = _plan(tmp_path)
        execution_root = tmp_path / "execution"
        final_root = tmp_path / "final"
        probe = _probe_with_source_preflight(
            fixture, plan, execution_root, final_root
        )
        receipt = _seal(
            fixture,
            plan,
            probe,
            execution_root,
            final_root,
            tmp_path / "receipt.json",
        )
        tampered = copy.deepcopy(receipt)
        tampered["device_plan"]["shard-000"] = "7"
        tampered["pre_gpu_receipt_path"] = str((tmp_path / "tampered.json").resolve())
        tampered["self_sha256"] = sealer.document_self_sha256(tampered)
        tmp_path.joinpath("tampered.json").write_bytes(sealer.canonical_json_bytes(tampered) + b"\n")
        with pytest.raises(sealer.PreGpuReceiptError, match="device plan"):
            sealer.validate_pre_gpu_receipt(tmp_path / "tampered.json")
        (tmp_path / "execution").mkdir()
        with pytest.raises(sealer.PreGpuReceiptError, match="absent"):
            sealer.validate_pre_gpu_receipt(tmp_path / "receipt.json")
    finally:
        sealer.CODE_ROLE_PATHS.clear()
        sealer.CODE_ROLE_PATHS.update(fixture["old_code"])
        sealer.TEST_PATHS.clear()
        sealer.TEST_PATHS.update(fixture["old_tests"])


def test_runtime_identity_rehashes_receipt_and_allows_only_sibling_shards(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    try:
        plan = _plan(tmp_path)
        receipt_path = tmp_path / "receipt.json"
        execution_root = tmp_path / "execution"
        final_root = tmp_path / "final"
        probe = _probe_with_source_preflight(
            fixture, plan, execution_root, final_root
        )
        receipt = _seal(
            fixture,
            plan,
            probe,
            execution_root,
            final_root,
            receipt_path,
        )
        (tmp_path / "execution" / "shard-000").mkdir(parents=True)
        (tmp_path / "execution" / "shard-001").mkdir()
        identity = sealer.runtime_identity_binding(
            receipt,
            receipt_path=receipt_path,
            shard_id="shard-002",
            observed_cuda_visible_devices=sealer.DEVICE_PLAN["shard-002"],
        )
        checked = sealer.validate_runtime_identity(
            identity,
            receipt,
            receipt_path=receipt_path,
            shard_id="shard-002",
            observed_cuda_visible_devices=sealer.DEVICE_PLAN["shard-002"],
        )
        assert checked["identity"] == identity
        assert identity["pre_gpu_receipt_path"] == str(receipt_path.resolve())
        assert identity["pre_gpu_receipt_sha256"] == sealer.sha256_file(receipt_path)
        assert identity["code_hashes"]["live_executor"] == identity["code_hashes"]["base_live_executor"]
        fixture["config"].write_text("model: drifted\n", encoding="utf-8")
        with pytest.raises(sealer.PreGpuReceiptError, match="input config|input hashes"):
            sealer.runtime_identity_binding(
                receipt,
                receipt_path=receipt_path,
                shard_id="shard-002",
                observed_cuda_visible_devices=sealer.DEVICE_PLAN["shard-002"],
            )
    finally:
        sealer.CODE_ROLE_PATHS.clear()
        sealer.CODE_ROLE_PATHS.update(fixture["old_code"])
        sealer.TEST_PATHS.clear()
        sealer.TEST_PATHS.update(fixture["old_tests"])
