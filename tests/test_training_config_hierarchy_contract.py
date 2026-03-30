from __future__ import annotations

from pathlib import Path

import yaml

from src.config.loader import ConfigLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_ROOT = REPO_ROOT / "configs" / "stage1"
STAGE2_ROOT = REPO_ROOT / "configs" / "stage2_two_channel"
STAGE_BASES = {
    "configs/stage1/sft_base.yaml",
    "configs/stage2_two_channel/base.yaml",
}


def _raw_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _normalize_extends(path: Path) -> list[Path]:
    payload = _raw_yaml(path)
    extends = payload.get("extends")
    if not extends:
        return []
    if isinstance(extends, str):
        extends_items = [extends]
    else:
        extends_items = list(extends)
    return [(path.parent / item).resolve() for item in extends_items]


def _is_stage1_non_smoke_leaf(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return (
        rel.startswith("configs/stage1/")
        and "/_shared/" not in rel
        and "/smoke/" not in rel
        and rel != "configs/stage1/sft_base.yaml"
    )


def _is_stage2_non_smoke_leaf(path: Path) -> bool:
    rel = path.relative_to(REPO_ROOT).as_posix()
    return (
        rel.startswith("configs/stage2_two_channel/")
        and "/_shared/" not in rel
        and "/smoke/" not in rel
        and rel != "configs/stage2_two_channel/base.yaml"
    )


def _stage1_non_smoke_leaves() -> list[Path]:
    return sorted(
        path for path in STAGE1_ROOT.rglob("*.yaml") if _is_stage1_non_smoke_leaf(path)
    )


def _stage2_non_smoke_leaves() -> list[Path]:
    return sorted(
        path for path in STAGE2_ROOT.rglob("*.yaml") if _is_stage2_non_smoke_leaf(path)
    )


def _all_non_smoke_training_configs() -> list[Path]:
    return _stage1_non_smoke_leaves() + _stage2_non_smoke_leaves()


def _representative_migrated_leaves() -> list[Path]:
    return [
        STAGE1_ROOT / "profiles/4b/coord_soft_ce_gate_coco80_desc_first.yaml",
        STAGE1_ROOT / "lvis_bbox_max60_1024.yaml",
        STAGE2_ROOT / "prod/a_only.yaml",
        STAGE2_ROOT / "prod/ab_mixed.yaml",
        STAGE2_ROOT / "lvis_bbox_max60_1024.yaml",
    ]


def _ancestor_closure(leaf: Path) -> set[Path]:
    closure: set[Path] = set()
    stack = [leaf]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        stack.extend(_normalize_extends(current))
    return closure


def _semantic_role(path: Path, *, current_leaf: Path) -> str:
    rel = path.relative_to(REPO_ROOT).as_posix()
    if path == current_leaf:
        return "specialized_leaf"
    if rel == "configs/base.yaml":
        return "universal_base"
    if rel in STAGE_BASES:
        return "stage_base"
    if "/_shared/" in rel:
        return "shared_common"
    return "intermediate_specialized"


def test_stage1_canonical_profiles_load_under_current_hierarchy() -> None:
    profiles = _stage1_non_smoke_leaves()
    assert profiles, "Expected canonical Stage-1 non-smoke leaves."

    for path in profiles:
        ConfigLoader.load_materialized_training_config(str(path))


def test_stage2_prod_common_no_longer_hides_prompt_identity() -> None:
    prod_common = ConfigLoader.load_yaml_with_extends(
        str(STAGE2_ROOT / "_shared/prod_common.yaml")
    )
    prod_common_custom = prod_common.get("custom", {}) or {}
    prod_common_extra = prod_common_custom.get("extra", {}) or {}

    assert "prompt_variant" not in prod_common_extra
    assert "object_field_order" not in prod_common_custom

    expected_prompt = "coco_80"
    for path in (
        STAGE2_ROOT / "_shared/prod_ab_mixed_vllm.yaml",
        STAGE2_ROOT / "_shared/prod_a_only_hf.yaml",
    ):
        merged = ConfigLoader.load_yaml_with_extends(str(path))
        custom = merged.get("custom", {}) or {}
        extra = custom.get("extra", {}) or {}
        assert extra["prompt_variant"] == expected_prompt
        assert custom["object_field_order"] == "desc_first"


def test_canonical_non_smoke_leaves_author_raw_run_identity_fields() -> None:
    leaves = _all_non_smoke_training_configs()
    assert leaves, "Expected canonical non-smoke training leaves across Stage-1 and Stage-2."

    for leaf in leaves:
        payload = _raw_yaml(leaf)
        model = payload.get("model", {}) or {}
        training = payload.get("training", {}) or {}
        assert model.get("model"), f"{leaf.relative_to(REPO_ROOT)} must author model.model"
        assert training.get("run_name"), f"{leaf.relative_to(REPO_ROOT)} must author training.run_name"
        assert training.get("artifact_subdir"), (
            f"{leaf.relative_to(REPO_ROOT)} must author training.artifact_subdir"
        )


def test_representative_migrated_leaves_stay_within_semantic_depth_budget() -> None:
    leaves = _representative_migrated_leaves()
    assert leaves, "Expected representative migrated leaves."

    required_roles = {"universal_base", "stage_base", "specialized_leaf"}
    allowed_roles = required_roles | {"shared_common"}

    for leaf in leaves:
        roles_by_path = {
            path.relative_to(REPO_ROOT).as_posix(): _semantic_role(path, current_leaf=leaf)
            for path in _ancestor_closure(leaf)
            if path.is_relative_to(REPO_ROOT)
        }
        unexpected = sorted(
            rel for rel, role in roles_by_path.items() if role == "intermediate_specialized"
        )
        assert not unexpected, (
            f"{leaf.relative_to(REPO_ROOT)} introduces an extra specialized layer via "
            f"{unexpected!r}; reusable intermediates must be shared/common package settings."
        )

        roles = set(roles_by_path.values())
        assert roles <= allowed_roles
        assert required_roles <= roles, (
            f"{leaf.relative_to(REPO_ROOT)} is missing one of the required semantic layers: "
            f"{sorted(required_roles - roles)!r}"
        )
        assert len(roles) <= 4, (
            f"{leaf.relative_to(REPO_ROOT)} exceeds the preferred semantic depth budget: "
            f"{sorted(roles)!r}"
        )
