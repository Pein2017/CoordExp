"""Keep frozen experiment policy out of stable inference surfaces."""

from __future__ import annotations

from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
STABLE_INFERENCE_SURFACES = (
    REPOSITORY_ROOT / "src" / "inference",
    REPOSITORY_ROOT / "src" / "config",
    REPOSITORY_ROOT / "configs" / "infer",
)
SCANNED_TEXT_SUFFIXES = frozenset({".json", ".py", ".toml", ".yaml", ".yml"})

# Frozen arm identifiers and their complete operational meanings:
# - FULL_SINGLE: full-image single-rollout reference.
# - FULL_BAG_K: full-image bag of K calls, where K is the selected grid-cell count.
# - TILE_RESET: native-scale tile calls with a fresh prompt for every tile.
# - MASK_RESET: full-size masked-canvas calls with a fresh prompt for every mask.
# - MASK_CUMULATIVE: full-size masked-canvas calls with cumulative accepted rows.
# - MASK_RESET_CORE_ONLY: masked-canvas reset calls without halo context.
# - TILE_RESET_CORE_ONLY: native-scale tile reset calls without halo context.
# - MASK_CUMULATIVE_ORDER_PANEL: cumulative-mask traversal-order sensitivity panel.
# - MASK_RESET_ORDER_PANEL: reset-mask traversal-order negative-control panel.
# - TILE_CUMULATIVE_EXPLORATORY: exploratory cumulative-prefix native-tile arm.
FROZEN_SPATIAL_SCOPE_ARM_IDENTIFIERS = (
    "full_single",
    "full_bag_k",
    "tile_reset",
    "mask_reset",
    "mask_cumulative",
    "mask_reset_core_only",
    "tile_reset_core_only",
    "mask_cumulative_order_panel",
    "mask_reset_order_panel",
    "tile_cumulative_exploratory",
)

# Dense-Union-51 is the frozen 51-image, annotation-derived dense audit cohort.
# The seed namespace belongs only to that experiment's request schedule.
FROZEN_COHORT_AND_SEED_IDENTIFIERS = (
    "dense-union-51",
    "dense_union_51",
    "coordexp-dense-enumeration-seeds-v1",
    "coordexp_dense_enumeration_seeds_v1",
)

# Core-plus-halo is the frozen spatial cell plus boundary-context policy.
# Paired-cell is the frozen seed role shared by corresponding spatial calls.
FROZEN_GRID_POLICY_IDENTIFIERS = (
    "core-plus-halo",
    "core_plus_halo",
    "paired-cell",
    "paired_cell",
)

# These are experiment-owned calibration, effect, and safety measurements, not
# generic inference controls.
FROZEN_CALIBRATION_AND_DECISION_IDENTIFIERS = (
    "prediction-set diversity",
    "prediction_set_diversity",
    "mask-harm retention",
    "mask_harm_retention",
    "local_rescue_rate_difference",
    "prediction_count_inflation",
    "audit_manual_precision",
    "post_merge_strict_duplicate_rate",
    "image-clustered bootstrap",
    "image_clustered_bootstrap",
)

# NMS means Non-Maximum Suppression. These exact setting names belong to the
# frozen research-side object merger and reference matcher, not shard merging or
# generic stable inference scoring.
FROZEN_OBJECT_MERGE_AND_REFERENCE_MATCHING_IDENTIFIERS = (
    "nms_iou_threshold",
    "reference_match_iou_threshold",
    "reference_matching_iou_threshold",
    "strict_duplicate_iou_threshold",
    "merge-created match",
    "merge_created_match",
    "merge-destroyed match",
    "merge_destroyed_match",
)

# This is the exact research investigation family and unit run-root vocabulary.
FROZEN_RESEARCH_OUTPUT_ROOT_IDENTIFIERS = (
    "outputs/research/qwen3-vl-dense-enumeration",
    "outputs/research/qwen3_vl_dense_enumeration",
    "2026-07-13-spatial-scope-history-disentanglement",
    "2026_07_13_spatial_scope_history_disentanglement",
)

FROZEN_RESEARCH_VOCABULARY = {
    "spatial-scope arm": FROZEN_SPATIAL_SCOPE_ARM_IDENTIFIERS,
    "cohort or seed namespace": FROZEN_COHORT_AND_SEED_IDENTIFIERS,
    "grid policy": FROZEN_GRID_POLICY_IDENTIFIERS,
    "calibration or decision threshold": FROZEN_CALIBRATION_AND_DECISION_IDENTIFIERS,
    "object merge or reference matching": (
        FROZEN_OBJECT_MERGE_AND_REFERENCE_MATCHING_IDENTIFIERS
    ),
    "research output root": FROZEN_RESEARCH_OUTPUT_ROOT_IDENTIFIERS,
}


def test_stable_inference_surfaces_exclude_frozen_research_vocabulary() -> None:
    """Reject experiment ownership leaking into stable source or configuration."""

    violations: list[str] = []
    for path in _stable_text_files():
        relative_path = path.relative_to(REPOSITORY_ROOT)
        path_text = relative_path.as_posix().casefold()
        lines = path.read_text(encoding="utf-8").splitlines()
        for ownership_class, identifiers in FROZEN_RESEARCH_VOCABULARY.items():
            for identifier in identifiers:
                normalized_identifier = identifier.casefold()
                if normalized_identifier in path_text:
                    violations.append(
                        f"{relative_path}: path contains {ownership_class} "
                        f"identifier {identifier!r}"
                    )
                for line_number, line in enumerate(lines, start=1):
                    if normalized_identifier in line.casefold():
                        violations.append(
                            f"{relative_path}:{line_number}: contains "
                            f"{ownership_class} identifier {identifier!r}"
                        )

    assert not violations, (
        "Frozen experiment vocabulary leaked into stable inference surfaces:\n"
        + "\n".join(violations)
    )


def _stable_text_files() -> list[Path]:
    return sorted(
        path
        for root in STABLE_INFERENCE_SURFACES
        for path in root.rglob("*")
        if path.is_file() and path.suffix in SCANNED_TEXT_SUFFIXES
    )
