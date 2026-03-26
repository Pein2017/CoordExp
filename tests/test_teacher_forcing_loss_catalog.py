from src.trainers.teacher_forcing.module_registry import (
    ALLOWED_OBJECTIVE_MODULES,
    OBJECTIVE_APPLICATION_PRESET_ALLOWLIST,
    OBJECTIVE_CONFIG_ALLOWLIST,
    OBJECTIVE_MODULE_CATALOG,
    OBJECTIVE_OPTIONAL_CONFIG_KEYS,
    objective_modules_for_family,
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



def test_bbox_modules_have_shared_family_but_distinct_roles() -> None:
    bbox_modules = objective_modules_for_family("bbox")

    assert bbox_modules == ("bbox_geo", "bbox_size_aux")
    assert OBJECTIVE_MODULE_CATALOG["bbox_geo"].semantic_role == "geometry"
    assert OBJECTIVE_MODULE_CATALOG["bbox_size_aux"].semantic_role == "size_aux"
    assert (
        OBJECTIVE_MODULE_CATALOG["bbox_geo"].projected_atoms[1].atom_name
        == "bbox_ciou"
    )
