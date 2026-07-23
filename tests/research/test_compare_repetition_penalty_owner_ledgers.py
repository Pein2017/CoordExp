from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research.compare_repetition_penalty_owner_ledgers import (
    ComparatorError,
    compare_ledgers,
)


def _row(
    image_id: str,
    band: str,
    source: dict[str, list[str]],
    treatment: dict[str, list[str]],
    *,
    eligible: bool = True,
) -> dict[str, Any]:
    by_iou = {}
    for threshold in ("0.30", "0.50"):
        source_ids = source[threshold]
        treatment_ids = treatment[threshold]
        by_iou[threshold] = {
            "source_owner_ids": source_ids,
            "treatment_owner_ids": treatment_ids,
            "retained_source_owner_ids": sorted(set(source_ids) & set(treatment_ids)),
            "gained_annotated_owner_ids": sorted(set(treatment_ids) - set(source_ids)),
            "lost_source_owner_ids": sorted(set(source_ids) - set(treatment_ids)),
        }
    return {
        "image_id": image_id,
        "annotated_object_count": 2 if band == "small" else 9,
        "object_count_band": band,
        "owner_comparison": {
            "eligible": eligible,
            "exclusion_reason": None if eligible else "synthetic_failure",
            "by_intersection_over_union": by_iou if eligible else None,
        },
    }


def _write_ledger(
    path: Path,
    *,
    policy: str,
    arm: str,
    seed: int,
    candidate_sha: str = "candidate-v1",
    treatment_overrides: dict[str, list[str]] | None = None,
    exclude_image: str | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    repetition_penalty = {"rp1p00": 1.0, "rp1p10": 1.1}[policy]
    source = {
        "1": {"0.30": ["1:a"], "0.50": ["1:a"]},
        "2": {"0.30": ["2:a"], "0.50": []},
        "3": {"0.30": ["3:a"], "0.50": ["3:a"]},
    }
    rows = []
    for image_id, band in (("1", "small"), ("2", "small"), ("3", "large")):
        treatment = {threshold: list(values) for threshold, values in source[image_id].items()}
        for threshold in ("0.30", "0.50"):
            override_key = f"{image_id}:{threshold}"
            if treatment_overrides and override_key in treatment_overrides:
                treatment[threshold] = treatment_overrides[override_key]
        rows.append(
            _row(
                image_id,
                band,
                source[image_id],
                treatment,
                eligible=image_id != exclude_image,
            )
        )
    config = {"repetition_penalty": repetition_penalty}
    name = f"{arm}-seed{seed}-step30"
    payload = {
        "schema_version": "source_b16_treatment_owner_ledger.v1",
        "inputs": {
            "candidate_jsonl": {"sha256": candidate_sha},
            "source_panel": {"config_identity": config},
        },
        "policy": {"intersection_over_union_thresholds": [0.3, 0.5]},
        "treatments": {
            name: {
                "panel": {"config_identity": config},
                "per_image": rows,
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _matrix(tmp_path: Path) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    for policy in ("rp1p00", "rp1p10"):
        for arm in ("broad", "concentrated"):
            for seed in (19, 23):
                overrides = None
                if policy == "rp1p10" and arm == "broad":
                    overrides = {"1:0.30": ["1:a", "1:b"], "1:0.50": ["1:b"]}
                if policy == "rp1p10" and arm == "concentrated":
                    overrides = {"2:0.30": []}
                path = tmp_path / f"{policy}-{arm}-{seed}.json"
                specs.append(
                    (
                        policy,
                        _write_ledger(
                            path,
                            policy=policy,
                            arm=arm,
                            seed=seed,
                            treatment_overrides=overrides,
                            exclude_image="3" if policy == "rp1p10" and seed == 23 else None,
                        ),
                    )
                )
    return specs


def test_compare_exact_matrix_on_one_common_complete_case_cohort(tmp_path: Path) -> None:
    specs = _matrix(tmp_path)
    result = compare_ledgers(specs, bootstrap_iterations=100, bootstrap_seed=7)

    assert result["cohort"] == {
        "candidate_image_count": 3,
        "common_complete_case_image_count": 2,
        "excluded_image_count": 1,
        "image_ids": ["1", "2"],
        "object_count_band_counts": {"small": 2},
    }
    threshold = result["thresholds"]["0.30"]
    broad = threshold["absolute_by_policy_arm_seed"]["rp1p10"]["broad"]["19"]
    assert broad == {
        "source_matched_owner_count": 2,
        "treatment_matched_owner_count": 3,
        "retained_source_owner_count": 2,
        "gained_annotated_owner_count": 1,
        "lost_source_owner_count": 0,
        "net_owner_delta": 1,
    }
    assert threshold["broad_minus_concentrated_treatment_owner_count_by_seed"][
        "rp1p10"
    ] == {"19": 2, "23": 2}
    direct = threshold["direct_rp1p10_minus_rp1p00"]["broad"]["19"]
    assert direct == {
        "owner_count_delta": 1,
        "gained_owner_count": 1,
        "gained_owner_ids": ["1:b"],
        "lost_owner_count": 0,
        "lost_owner_ids": [],
    }
    source_direct = threshold["direct_rp1p10_minus_rp1p00"]["source"]
    assert source_direct["owner_count_delta"] == 0
    first_bootstrap = threshold["object_count_band_stratified_paired_bootstrap"]
    again = compare_ledgers(specs, bootstrap_iterations=100, bootstrap_seed=7)
    assert first_bootstrap == again["thresholds"]["0.30"][
        "object_count_band_stratified_paired_bootstrap"
    ]
    assert first_bootstrap["direct_rp_effect"]["broad-seed19"]["estimate"] == 1.0
    assert len(result["inputs"]) == 8
    assert all(len(item["sha256"]) == 64 for item in result["inputs"])


def test_rejects_incomplete_or_duplicate_matrix(tmp_path: Path) -> None:
    specs = _matrix(tmp_path)
    with pytest.raises(ComparatorError, match="exactly eight"):
        compare_ledgers(specs[:-1], bootstrap_iterations=10)
    with pytest.raises(ComparatorError, match="duplicate matrix cell"):
        compare_ledgers(specs[:-1] + [specs[0]], bootstrap_iterations=10)


def test_rejects_candidate_or_runtime_policy_mismatch(tmp_path: Path) -> None:
    specs = _matrix(tmp_path)
    policy, path = specs[-1]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inputs"]["candidate_jsonl"]["sha256"] = "different"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ComparatorError, match="candidate JSONL identity"):
        compare_ledgers(specs, bootstrap_iterations=10)

    specs = _matrix(tmp_path / "runtime")
    policy, path = specs[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["inputs"]["source_panel"]["config_identity"]["repetition_penalty"] = 1.1
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ComparatorError, match="is not rp1p00"):
        compare_ledgers(specs, bootstrap_iterations=10)
