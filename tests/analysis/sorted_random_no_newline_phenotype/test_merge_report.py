from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from src.analysis.sorted_random_no_newline_phenotype.merge_report import (
    BANNED_CAUSAL_PHRASES,
    LEGACY_A31_LABELS,
    RANDOM_ROLE,
    SORTED_ROLE,
    build_report_markdown,
    merge_prefix_readout_rows,
    write_jsonl,
    write_report,
)


def test_merge_uses_dynamic_a3_2_roles_and_explicit_sorted_minus_random_deltas(
    tmp_path: Path,
) -> None:
    rows = [
        _readout_row(
            "state-2",
            SORTED_ROLE,
            residual_vs_eos_margin=0.50,
            strict_r95_x1_hit_rate=0.80,
            boundary_residual_favored_rate=0.40,
        ),
        _readout_row(
            "state-1",
            RANDOM_ROLE,
            residual_vs_eos_margin=0.10,
            strict_r95_x1_hit_rate=0.25,
            boundary_residual_favored_rate=0.20,
        ),
        _readout_row(
            "state-1",
            SORTED_ROLE,
            residual_vs_eos_margin=0.35,
            strict_r95_x1_hit_rate=0.75,
            boundary_residual_favored_rate=0.70,
        ),
        _readout_row(
            "state-2",
            RANDOM_ROLE,
            residual_vs_eos_margin=0.20,
            strict_r95_x1_hit_rate=0.70,
            boundary_residual_favored_rate=0.35,
        ),
    ]

    merged = merge_prefix_readout_rows(rows, role_a=RANDOM_ROLE, role_b=SORTED_ROLE)

    assert [row["prefix_state_id"] for row in merged] == ["state-1", "state-2"]
    assert merged[0]["role_a"] == RANDOM_ROLE
    assert merged[0]["role_b"] == SORTED_ROLE
    assert merged[0]["delta_role"] == "sorted_minus_random"
    assert merged[0]["sorted_minus_random_residual_vs_eos_margin"] == pytest.approx(
        0.25
    )
    assert merged[0]["sorted_minus_random_strict_r95_x1_hit_rate"] == pytest.approx(
        0.50
    )
    assert merged[0][
        "sorted_minus_random_boundary_residual_favored_rate"
    ] == pytest.approx(0.50)

    rendered = json.dumps(merged, allow_nan=False, sort_keys=True)
    for legacy in LEGACY_A31_LABELS:
        assert legacy not in rendered

    out_path = tmp_path / "summary" / "prefix_readout_merged_rows.jsonl"
    write_jsonl(out_path, merged)
    assert out_path.read_text(encoding="utf-8").count("\n") == 2


def test_merge_rejects_legacy_a3_1_labels_before_materializing() -> None:
    row = _readout_row(
        "state-legacy",
        RANDOM_ROLE,
        residual_vs_eos_margin=0.0,
        strict_r95_x1_hit_rate=0.0,
        boundary_residual_favored_rate=0.0,
    )
    row["phase_id"] = "phase_a3_1"

    with pytest.raises(ValueError, match="legacy A3.1 label"):
        merge_prefix_readout_rows(
            [
                row,
                _readout_row(
                    "state-legacy",
                    SORTED_ROLE,
                    residual_vs_eos_margin=0.1,
                    strict_r95_x1_hit_rate=0.1,
                    boundary_residual_favored_rate=0.1,
                ),
            ]
        )


def test_report_has_required_sections_cautious_language_and_val200_context(
    tmp_path: Path,
) -> None:
    merged = merge_prefix_readout_rows(
        [
            _readout_row(
                "state-1",
                RANDOM_ROLE,
                residual_vs_eos_margin=0.10,
                strict_r95_x1_hit_rate=0.25,
                boundary_residual_favored_rate=0.20,
            ),
            _readout_row(
                "state-1",
                SORTED_ROLE,
                residual_vs_eos_margin=0.35,
                strict_r95_x1_hit_rate=0.75,
                boundary_residual_favored_rate=0.70,
            ),
        ]
    )

    report = build_report_markdown(
        evidence_labels={
            "prefix_readout": "a3_2_prefix4096_hardbiased_len12000_canonical_sorted",
            "native_rollout": "a3_2_rollout1024_greedy_len12000_native",
            "fn_probe": "a3_2_fn512_greedyfn_len12000",
        },
        checkpoint_provenance={
            RANDOM_ROLE: {
                "checkpoint_path": "/random/checkpoint-3668",
                "training_ordering": "random_permutation",
            },
            SORTED_ROLE: {
                "checkpoint_path": "/sorted/checkpoint-3668",
                "training_ordering": "sorted",
            },
        },
        data_roots={
            "train_jsonl": "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/train.coord.jsonl",
            "val_jsonl": "/data/CoordExp/public_data/coco/rescale_32_1024_bbox_len12000/val.coord.jsonl",
            "image_root": "/data/CoordExp/public_data/coco/rescale_32_1024_bbox",
        },
        template_contract={
            "detection_sequence_format": "compact_full",
            "coordinate_surface": "coord_token",
            "bbox_format": "xyxy",
            "row_separator": "none",
        },
        prefix_readout_rows=merged,
        rollout_summary={"random_fn": 729, "sorted_fn": 630},
        fn_universe_counts={
            "shared_fn": 5,
            "random_only_fn": 8,
            "sorted_only_fn": 2,
            "not_fn": 20,
        },
        fn_bucket_summary={
            "coord_binding_failure": 3,
            "prefix_suppression_flip": 2,
        },
        prefix_sensitivity={"rollout_prefix_loss": 2},
    )

    for heading in (
        "## Scope And Evidence Labels",
        "## Checkpoint Provenance",
        "## Data And Image Roots",
        "## External Offline Val-200 Eval Context",
        "## Template Contract",
        "## Prefix Readout",
        "## Native Rollout Phenotype",
        "## FN Universe Counts",
        "## FN Multi-Axis Bucket Counts",
        "## Prefix Sensitivity",
        "## A3.1 Compatibility And Caveats",
        "## Interpretive Caveats",
    ):
        assert heading in report

    assert "phenotype context, not mechanism proof" in report
    assert "AP@[.50:.95]" in report
    assert "FN@0.50" in report
    assert "sorted > random" in report
    assert "consistent with" in report
    assert "supports under this evidence scope" in report

    lower_report = report.lower()
    for phrase in BANNED_CAUSAL_PHRASES:
        assert not re.search(rf"\b{re.escape(phrase.lower())}\b", lower_report)
    for legacy in LEGACY_A31_LABELS:
        assert legacy not in report

    path = write_report(tmp_path / "summary" / "report.md", report)
    assert path.read_text(encoding="utf-8") == report


def test_report_rejects_unqualified_causal_language() -> None:
    with pytest.raises(ValueError, match="unqualified causal language"):
        build_report_markdown(
            evidence_labels={},
            checkpoint_provenance={},
            data_roots={},
            template_contract={},
            prefix_readout_rows=[],
            rollout_summary={},
            fn_universe_counts={},
            fn_bucket_summary={},
            prefix_sensitivity={},
            interpretive_caveats=["This proved the visual encoder failed."],
        )


def _readout_row(
    prefix_state_id: str,
    checkpoint_role: str,
    *,
    residual_vs_eos_margin: float,
    strict_r95_x1_hit_rate: float,
    boundary_residual_favored_rate: float,
) -> dict[str, object]:
    return {
        "project_id": "sorted_random_no_newline_phenotype",
        "phase_id": "phase_a3_2",
        "schema_version": "a3.2.v1",
        "run_id": "fullobj_purece_random_vs_sorted_ckpt3668_phase_a3_2",
        "prefix_state_id": prefix_state_id,
        "checkpoint_role": checkpoint_role,
        "checkpoint_roles": [RANDOM_ROLE, SORTED_ROLE],
        "boundary_summary": {
            "residual_vs_eos_margin": residual_vs_eos_margin,
            "strict_r95_x1_hit_rate": strict_r95_x1_hit_rate,
            "boundary_residual_favored_rate": boundary_residual_favored_rate,
        },
    }
