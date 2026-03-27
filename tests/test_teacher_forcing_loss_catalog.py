from src.trainers.teacher_forcing.module_registry import (
    ALLOWED_OBJECTIVE_MODULES,
    OBJECTIVE_APPLICATION_PRESET_ALLOWLIST,
    OBJECTIVE_CONFIG_ALLOWLIST,
    OBJECTIVE_MODULE_CATALOG,
    OBJECTIVE_OPTIONAL_CONFIG_KEYS,
    objective_modules_for_family,
)



import pytest

from src.trainers.teacher_forcing.module_registry import (
    DIAGNOSTIC_CONFIG_ALLOWLIST,
    DIAGNOSTIC_MODULE_CATALOG,
)
from src.trainers.teacher_forcing.objective_pipeline import _validate_registry_coverage

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



def test_bbox_modules_have_shared_family_but_distinct_roles() -> None:
    bbox_modules = objective_modules_for_family("bbox")

    assert bbox_modules == ("bbox_geo", "bbox_size_aux")
    assert OBJECTIVE_MODULE_CATALOG["bbox_geo"].semantic_role == "geometry"
    assert OBJECTIVE_MODULE_CATALOG["bbox_size_aux"].semantic_role == "size_aux"
    assert (
        OBJECTIVE_MODULE_CATALOG["bbox_geo"].projected_atoms[1].atom_name
        == "bbox_ciou"
    )


def test_loss_catalog_drives_diagnostic_registry_allowlists() -> None:
    for name, definition in DIAGNOSTIC_MODULE_CATALOG.items():
        assert DIAGNOSTIC_CONFIG_ALLOWLIST[name] == set(definition.config_keys)


def test_objective_registry_drift_fails_fast() -> None:
    registry = {
        name: object() for name in OBJECTIVE_MODULE_CATALOG if name != "coord_reg"
    }
    registry["unexpected_objective"] = object()

    with pytest.raises(
        RuntimeError,
        match=(
            r"objective registry is out of sync with loss catalog: "
            r"missing=\['coord_reg'\] unexpected=\['unexpected_objective'\]"
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
            r"missing=\['coord_diag'\] unexpected=\['unexpected_diagnostic'\]"
        ),
    ):
        _validate_registry_coverage(
            registry,
            allowed=set(DIAGNOSTIC_MODULE_CATALOG),
            kind="diagnostic",
        )
