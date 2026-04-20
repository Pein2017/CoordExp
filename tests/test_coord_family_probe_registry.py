from __future__ import annotations

import yaml

from src.analysis.coord_family_probe_registry import (
    canonical_xyxy_norm1000,
    get_family_probe_spec,
    native_slot_names,
)


def test_registry_covers_all_headline_2b_families_from_bundled_inventory() -> None:
    worktree_root = __import__("pathlib").Path(__file__).resolve().parents[1]
    base_cfg = yaml.safe_load(
        (
            worktree_root / "configs/analysis/coord_family_comparison/base.yaml"
        ).read_text(encoding="utf-8")
    )

    aliases = [
        str(row["alias"])
        for row in base_cfg["families"]
        if bool(row.get("is_headline_2b_family", base_cfg["defaults"]["is_headline_2b_family"]))
    ]

    assert aliases == [
        "base_xyxy_merged",
        "raw_text_xyxy_pure_ce",
        "cxcywh_pure_ce",
        "cxcy_logw_logh_pure_ce",
        "center_parameterization",
        "hard_soft_ce_2b",
    ]
    assert [get_family_probe_spec(alias).alias for alias in aliases] == aliases


def test_native_slot_names_cover_required_native_semantics() -> None:
    assert native_slot_names("base_xyxy_merged") == ("x1", "y1", "x2", "y2")
    assert native_slot_names("cxcywh_pure_ce") == ("cx", "cy", "w", "h")
    assert native_slot_names("cxcy_logw_logh_pure_ce") == ("cx", "cy", "logw", "logh")


def test_get_family_probe_spec_flags_projection_requirements() -> None:
    spec = get_family_probe_spec("center_parameterization")

    assert spec.requires_family_native_probe is True
    assert spec.requires_canonical_projection is True


def test_registry_records_runtime_contract_metadata() -> None:
    raw_text = get_family_probe_spec("raw_text_xyxy_pure_ce")
    cxcywh = get_family_probe_spec("cxcywh_pure_ce")
    center = get_family_probe_spec("center_parameterization")

    assert raw_text.infer_mode == "text"
    assert raw_text.pred_coord_mode == "norm1000"
    assert cxcywh.eval_compatibility_path == "constant_score_scored_jsonl"
    assert center.canonical_projection_kind == "family_specific"


def test_canonical_xyxy_norm1000_projects_noncanonical_families() -> None:
    assert canonical_xyxy_norm1000("base_xyxy_merged", [100, 200, 300, 400]) == [
        100.0,
        200.0,
        300.0,
        400.0,
    ]
    cxcywh_projected = canonical_xyxy_norm1000("cxcywh_pure_ce", [500, 500, 200, 100])
    assert len(cxcywh_projected) == 4
    assert cxcywh_projected[0] < cxcywh_projected[2]
    assert cxcywh_projected[1] < cxcywh_projected[3]
