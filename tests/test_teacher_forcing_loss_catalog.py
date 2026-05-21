import pytest

from src.trainers.teacher_forcing.module_registry import (
    ALLOWED_OBJECTIVE_MODULES,
    DIAGNOSTIC_CONFIG_ALLOWLIST,
    DIAGNOSTIC_MODULE_CATALOG,
    OBJECTIVE_APPLICATION_PRESET_ALLOWLIST,
    OBJECTIVE_CONFIG_ALLOWLIST,
    OBJECTIVE_MODULE_CATALOG,
    OBJECTIVE_OPTIONAL_CONFIG_KEYS,
    objective_modules_for_family,
)
from src.trainers.teacher_forcing.objective_pipeline import (
    _validate_registry_coverage,
    _run_residual_set_correction_module,
)


def test_loss_catalog_drives_objective_registry_allowlists() -> None:
    assert ALLOWED_OBJECTIVE_MODULES == set(OBJECTIVE_MODULE_CATALOG)

    for name, definition in OBJECTIVE_MODULE_CATALOG.items():
        assert OBJECTIVE_CONFIG_ALLOWLIST[name] == set(definition.config_keys)
        assert OBJECTIVE_APPLICATION_PRESET_ALLOWLIST[name] == set(
            definition.application_presets
        )
        assert OBJECTIVE_OPTIONAL_CONFIG_KEYS.get(name, set()) == set(
            definition.optional_config_keys
        )


def test_bbox_modules_are_removed_from_objective_catalog() -> None:
    assert objective_modules_for_family("bbox") == ()
    assert "bbox_geo" not in OBJECTIVE_MODULE_CATALOG
    assert "bbox_size_aux" not in OBJECTIVE_MODULE_CATALOG
    assert "coord_reg" not in OBJECTIVE_MODULE_CATALOG
    assert "coord_gate" not in OBJECTIVE_MODULE_CATALOG
    assert "text_gate" not in OBJECTIVE_MODULE_CATALOG
    assert "loss_duplicate_burst_unlikelihood" not in OBJECTIVE_MODULE_CATALOG


def test_residual_set_module_catalog_uses_strict_v1_config_keys() -> None:
    definition = OBJECTIVE_MODULE_CATALOG["residual_set_correction"]

    assert set(definition.config_keys) == {
        "prepared_rollout_jsonl",
        "expected_num_rollouts",
        "base_seed",
        "lambda_type",
        "lambda_inner",
        "fallback_loss_weight",
        "lambda_ul_promoted",
        "label_conflict_weight",
        "commit_iou_threshold",
        "duplicate_burst_iou_threshold",
        "ul_cluster_iou_threshold",
        "ul_gray_iou_low",
        "ul_consensus_ratio",
        "min_ul_valid_rollouts",
        "clean_gt_sft_mix",
        "strict_prepared_rollout_tokens",
        "legacy_reencode_fallback",
        "strict_builder_invariants",
    }
    assert {
        "num_rollouts",
        "coord_span_policy",
        "coverage_strength",
        "ul_geometry",
        "artifact_policy",
    }.isdisjoint(definition.config_keys)


def test_loss_catalog_drives_diagnostic_registry_allowlists() -> None:
    for name, definition in DIAGNOSTIC_MODULE_CATALOG.items():
        assert DIAGNOSTIC_CONFIG_ALLOWLIST[name] == set(definition.config_keys)


def test_objective_registry_drift_fails_fast() -> None:
    missing_name = sorted(OBJECTIVE_MODULE_CATALOG)[0]
    registry = {name: object() for name in OBJECTIVE_MODULE_CATALOG if name != missing_name}
    registry["unexpected_objective"] = object()

    with pytest.raises(
        RuntimeError,
        match=(
            r"objective registry is out of sync with loss catalog: "
            rf"missing=\['{missing_name}'\] unexpected=\['unexpected_objective'\]"
        ),
    ):
        _validate_registry_coverage(
            registry,
            allowed=set(OBJECTIVE_MODULE_CATALOG),
            kind="objective",
        )


def test_diagnostic_registry_drift_fails_fast() -> None:
    registry = {"unexpected_diagnostic": object()}

    with pytest.raises(
        RuntimeError,
        match=(
            r"diagnostic registry is out of sync with loss catalog: "
            r"missing=\[\] unexpected=\['unexpected_diagnostic'\]"
        ),
    ):
        _validate_registry_coverage(
            registry,
            allowed=set(DIAGNOSTIC_MODULE_CATALOG),
            kind="diagnostic",
        )


def test_residual_set_module_rejects_missing_context_or_spec() -> None:
    from src.trainers.teacher_forcing.modules.residual_set_correction import (
        run_residual_set_correction_module,
    )

    with pytest.raises(TypeError, match="TeacherForcingContext"):
        run_residual_set_correction_module(context=None, spec=None)


def test_residual_set_lazy_import_missing_module_reports_task_5(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "src.trainers.teacher_forcing.modules.residual_set_correction"
            or (
                level == 1
                and name == "modules.residual_set_correction"
                and fromlist == ("run_residual_set_correction_module",)
            )
        ):
            raise ModuleNotFoundError(
                "No module named 'src.trainers.teacher_forcing.modules.residual_set_correction'",
                name="src.trainers.teacher_forcing.modules.residual_set_correction",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(NotImplementedError, match="Task 5"):
        _run_residual_set_correction_module(context=None, spec=None)


def test_residual_set_lazy_import_reraises_inner_module_not_found(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if (
            name == "src.trainers.teacher_forcing.modules.residual_set_correction"
            or (
                level == 1
                and name == "modules.residual_set_correction"
                and fromlist == ("run_residual_set_correction_module",)
            )
        ):
            raise ModuleNotFoundError(
                "No module named 'residual_dependency'",
                name="residual_dependency",
            )
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        _run_residual_set_correction_module(context=None, spec=None)

    assert exc_info.value.name == "residual_dependency"
