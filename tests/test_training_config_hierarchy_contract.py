from __future__ import annotations

from pathlib import Path

import yaml

from src.config.loader import ConfigLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
STAGE1_ROOT = REPO_ROOT / "configs" / "stage1"
STAGE2_ROOT = REPO_ROOT / "configs" / "stage2_two_channel"
STAGE1_COORD_COMPONENTS_2B_ROOT = STAGE1_ROOT / "ablation/coord_components_2b"
STAGE_BASES = {
    "configs/stage1/sft_base.yaml",
    "configs/stage2_two_channel/base.yaml",
}
STAGE1_COORD_COMPONENTS_2B_BASE = (
    "configs/stage1/ablation/coord_components_2b/base.yaml"
)


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
        and "/archive/" not in rel
        and "/smoke/" not in rel
        and rel != "configs/stage1/sft_base.yaml"
        and rel != STAGE1_COORD_COMPONENTS_2B_BASE
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
        STAGE1_COORD_COMPONENTS_2B_ROOT / "ciou_hard_ce.yaml",
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
    if rel == STAGE1_COORD_COMPONENTS_2B_BASE:
        return "shared_common"
    if "/_shared/" in rel:
        return "shared_common"
    return "intermediate_specialized"


def test_stage1_canonical_profiles_load_under_current_hierarchy() -> None:
    profiles = _stage1_non_smoke_leaves()
    assert profiles, "Expected canonical Stage-1 non-smoke leaves."

    for path in profiles:
        ConfigLoader.load_materialized_training_config(str(path))


def test_stage1_coord_components_2b_ablation_matrix_is_clean() -> None:
    expected = {
        "coord_token_hard_ce": {
            "coord_enabled": False,
            "ce": None,
            "soft_ce": None,
            "smooth_l1": 0.0,
            "ciou": 0.0,
            "bbox_geo_enabled": False,
        },
        "soft_ce_only": {
            "coord_enabled": True,
            "ce": 0.0,
            "soft_ce": 1.0,
            "smooth_l1": 0.0,
            "ciou": 0.0,
            "bbox_geo_enabled": False,
        },
        "soft_ce_hard_ce": {
            "coord_enabled": True,
            "ce": 1.0,
            "soft_ce": 1.0,
            "smooth_l1": 0.0,
            "ciou": 0.0,
            "bbox_geo_enabled": False,
        },
        "smooth_l1_hard_ce": {
            "coord_enabled": True,
            "ce": 1.0,
            "soft_ce": 0.0,
            "smooth_l1": 0.01,
            "ciou": 0.0,
            "bbox_geo_enabled": True,
        },
        "ciou_hard_ce": {
            "coord_enabled": True,
            "ce": 1.0,
            "soft_ce": 0.0,
            "smooth_l1": 0.0,
            "ciou": 1.0,
            "bbox_geo_enabled": True,
        },
    }

    for group_name, expected_cfg in expected.items():
        path = STAGE1_COORD_COMPONENTS_2B_ROOT / f"{group_name}.yaml"
        cfg = ConfigLoader.load_yaml_with_extends(str(path))
        model = cfg["model"]
        training = cfg["training"]
        custom = cfg["custom"]
        coord_tokens = custom["coord_tokens"]
        coord_loss = custom["coord_soft_ce_w1"]
        bbox_geo = custom["bbox_geo"]
        bbox_size_aux = custom["bbox_size_aux"]

        assert model["model"] == "model_cache/models/Qwen/Qwen3-VL-2B-Instruct-coordexp"
        assert "Qwen3-VL-4B" not in model["model"]
        assert not model["model"].startswith("output/")

        assert custom["train_jsonl"] == (
            "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/train.coord.jsonl"
        )
        assert custom["val_jsonl"] == (
            "public_data/coco/rescale_32_1024_bbox_max60_lvis_proxy/val.coord.jsonl"
        )
        assert custom["object_ordering"] == "sorted"
        assert custom["object_field_order"] == "desc_first"
        assert custom["extra"]["prompt_variant"] == "coco_80"

        assert training["num_train_epochs"] == 4
        assert training["learning_rate"] == 2.0e-4
        assert training["vit_lr"] == 1.0e-4
        assert training["aligner_lr"] == 4.0e-4
        assert training["artifact_subdir"] == (
            f"stage1_2b/ablation/coord_components/{group_name}"
        )
        assert group_name in training["run_name"]

        assert coord_tokens == {"enabled": True, "skip_bbox_norm": True}
        assert coord_loss["enabled"] is expected_cfg["coord_enabled"]
        assert coord_loss["w1_weight"] == 0.0
        assert coord_loss["gate_weight"] == 1.0
        assert coord_loss["text_gate_weight"] == 1.0
        assert coord_loss["adjacent_repulsion_weight"] == 0.0
        if expected_cfg["ce"] is not None:
            assert coord_loss["ce_weight"] == expected_cfg["ce"]
        if expected_cfg["soft_ce"] is not None:
            assert coord_loss["soft_ce_weight"] == expected_cfg["soft_ce"]

        assert bbox_geo["enabled"] is expected_cfg["bbox_geo_enabled"]
        assert bbox_geo["smoothl1_weight"] == expected_cfg["smooth_l1"]
        assert bbox_geo["ciou_weight"] == expected_cfg["ciou"]
        assert bbox_geo["parameterization"] == "xyxy"
        assert bbox_size_aux["enabled"] is False


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
