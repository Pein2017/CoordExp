from __future__ import annotations

import json
from pathlib import Path

import pytest

import scripts.research.run_image_12576_row_mediation_crossover as stage7


ADMISSION = Path(
    "research/investigations/qwen3-vl-dense-enumeration/experiments/"
    "2026-07-19-sampled-history-target-reachability-and-complete-row-value/"
    "stage7-image-12576-row-three-row-four-mediation-admission.json"
)


def _classification(label: str) -> dict[str, object]:
    return {"class": label, "owner_id": None, "owner_ids": []}


def test_load_stage_seven_source_reconstructs_four_exact_prefixes() -> None:
    frozen = stage7.load_stage_seven_source(ADMISSION)
    assert {item["arm"] for item in frozen["prefixes"]} == stage7.EXPECTED_ARMS
    assert all(len(item["prefix_token_ids"]) == 65 for item in frozen["prefixes"])
    assert frozen["rows"]["native_row_three"]["strict_matched_owner_ids"] == ["1509406"]
    assert frozen["rows"]["sampled_row_four"]["strict_matched_owner_ids"] == ["1506295"]


def test_load_stage_seven_source_refuses_changed_prefix_hash(tmp_path: Path) -> None:
    document = json.loads(ADMISSION.read_text(encoding="utf-8"))
    document["factorial_prefixes"][1]["prefix_token_ids_sha256"] = "0" * 64
    changed = tmp_path / "admission.json"
    changed.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(stage7.StageSevenValidationError, match="prefix hash mismatch"):
        stage7.load_stage_seven_source(changed)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("sampled_target_owner", "native_owner", "row_three_direct_retention"),
        ("native_owner", "sampled_target_owner", "row_four_mediation_or_screening"),
        ("sampled_target_owner", "sampled_target_owner", "either_changed_row_sufficient"),
        ("native_owner", "native_owner", "joint_interaction_or_endpoint_dependency"),
        ("other_strict_owner", "native_owner", "unresolved_alternative"),
        ("native_owner", "unresolved", "unresolved_alternative"),
    ],
)
def test_classify_crossover_follows_frozen_decision_table(
    first: str, second: str, expected: str
) -> None:
    receipt = stage7.classify_crossover(
        {
            "sampled_row_three__native_row_four": _classification(first),
            "native_row_three__sampled_row_four": _classification(second),
        }
    )
    assert receipt["conclusion"] == expected
    assert receipt["primary_mediation_claim_allowed"] is (
        expected != "unresolved_alternative"
    )


def test_validate_endpoint_parity_refuses_owner_only_match() -> None:
    frozen = stage7.load_stage_seven_source(ADMISSION)
    native_expected = frozen["candidates"]["native_candidate"]
    sampled_expected = frozen["candidates"]["sampled_target_candidate"]
    arm_results = {
        "native_row_three__native_row_four": {
            "row": {
                "raw_generated_token_ids": [*native_expected["raw_generated_token_ids"][:-1], 0]
            },
            "classification": _classification("native_owner"),
        },
        "sampled_row_three__sampled_row_four": {
            "row": {"raw_generated_token_ids": sampled_expected["raw_generated_token_ids"]},
            "classification": _classification("sampled_target_owner"),
        },
    }
    receipt = stage7.validate_endpoint_parity(arm_results, frozen["candidates"])
    assert receipt["passed"] is False
    assert receipt["native"]["raw_token_ids_equal"] is False
    assert receipt["native"]["owner_class_equal"] is True


def test_main_refuses_existing_output_before_loading_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("already here", encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        [
            "runner",
            "--stage7-admission",
            str(ADMISSION),
            "--infer-config",
            "unused.yaml",
            "--output",
            str(output),
        ],
    )
    monkeypatch.setattr(
        stage7,
        "load_stage_seven_source",
        lambda _: pytest.fail("source loading must not run after overwrite refusal"),
    )
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        stage7.main()
