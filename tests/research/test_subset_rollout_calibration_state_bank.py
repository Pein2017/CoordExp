from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pytest
from PIL import Image

from src.config.fingerprint import sha256_json
from src.config.fingerprint import sha256_file
from src.rollout_calibration import CheckpointIdentity, assemble_state_bank
from scripts.research.subset_rollout_calibration_state_bank import (
    SubsetError,
    build_subset,
    select_event_pairs,
)


_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "rollout_calibration_test_conftest", Path(__file__).parents[1] / "rollout_calibration" / "conftest.py"
)
assert _FIXTURE_SPEC and _FIXTURE_SPEC.loader
_FIXTURE_MODULE = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURE_MODULE)
SYNTHETIC_SOURCE_CHECKPOINT = _FIXTURE_MODULE.SYNTHETIC_SOURCE_CHECKPOINT
synthetic_inputs = _FIXTURE_MODULE.synthetic_inputs


def _parent_arm(tmp_path: Path) -> Path:
    rollouts: list[dict] = []
    reviews: list[dict] = []
    for index, family in enumerate(("positive_path_imitation", "source_route_imitation") * 2):
        (tmp_path / f"fixture-{index}").mkdir(parents=True)
        source_rollouts, source_reviews, _ = synthetic_inputs(
            tmp_path / f"fixture-{index}", event_id=f"event-{index}", image_id=42 + index
        )
        rollout = source_rollouts[0]
        review = source_reviews[0]
        image_path = Path(rollout["image"]["path"])
        Image.new("RGB", (64, 64), color=(12 + index, 34, 56)).save(image_path)
        rollout["image"]["content_sha256"] = sha256_file(image_path)
        review["positive_path_imitation_eligible"] = family == "positive_path_imitation"
        review["source_route_imitation_eligible"] = family == "source_route_imitation"
        review["entity_transition_eligible"] = False
        review["coordinate_boundary_eligible"] = False
        review["image_balanced_event_weight"] = 1.0
        review_candidate = review["candidates"][0]
        review["candidates"] = [review_candidate]
        review_candidate.update(
            {
                "geometry_review_status": "unknown",
                "geometry_eligible": False,
                "coordinate_decision": None,
                "owner_resolution_interval": [0, 3],
                "selected_sites": [
                    {"candidate_token_offset": 0, "intended_token_type": "desc_text"},
                    {"candidate_token_offset": 1, "intended_token_type": "coordinate"},
                    {"candidate_token_offset": 2, "intended_token_type": "schema"},
                ],
            }
        )
        rollout["candidates"] = [rollout["candidates"][0]]
        if family == "source_route_imitation":
            rollout["candidates"][0]["generation_provenance"].update(
                {"mode": "greedy", "seed": 0, "temperature": 0.0, "top_p": 1.0}
            )
        rollouts.append(rollout)
        reviews.append(review)
    arm_root = tmp_path / "arm"
    manifest = assemble_state_bank(
        output_dir=arm_root / "state-bank",
        rollout_rows=rollouts,
        review_rows=reviews,
        source_checkpoint=CheckpointIdentity(**SYNTHETIC_SOURCE_CHECKPOINT),
        prompt_identity_sha256="8" * 64,
        source_artifacts=[{"artifact_id": "fixture", "sha256": sha256_json({"fixture": True})}],
    )
    pre = arm_root / "pre-state-bank"
    pre.mkdir(parents=True)
    for name, rows in (("rollout_rows", rollouts), ("review_rows", reviews)):
        (pre / f"{name}.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows),
            encoding="utf-8",
        )
    (arm_root / "assembly-receipt.json").write_text(
        json.dumps({"status": "assembled", "bank_id": manifest.bank_id}), encoding="utf-8"
    )
    return arm_root / "state-bank" / "manifest.json"


def test_family_selection_is_deterministic_and_image_diverse(tmp_path: Path) -> None:
    manifest = _parent_arm(tmp_path)
    parent = manifest.parent.parent / "pre-state-bank"
    rollouts = [json.loads(line) for line in (parent / "rollout_rows.jsonl").read_text().splitlines()]
    reviews = [json.loads(line) for line in (parent / "review_rows.jsonl").read_text().splitlines()]
    first = select_event_pairs(
        rollouts, reviews, family_counts={"positive_path_imitation": 1, "source_route_imitation": 1}
    )
    second = select_event_pairs(
        list(reversed(rollouts)), list(reversed(reviews)), family_counts={"positive_path_imitation": 1, "source_route_imitation": 1}
    )
    assert first[2]["event_ids"] == second[2]["event_ids"]
    assert first[2]["family_counts"] == {"positive_path_imitation": 1, "source_route_imitation": 1}
    assert len(first[2]["image_ids"]) == 2


def test_subset_preserves_rows_identity_and_validates_output(tmp_path: Path) -> None:
    manifest = _parent_arm(tmp_path)
    receipt = build_subset(
        manifest,
        tmp_path / "subset",
        event_ids=["event-1"],
    )
    assert receipt["selection"]["family_counts"] == {"positive_path_imitation": 0, "source_route_imitation": 1}
    assert receipt["source_checkpoint_id"]
    selected_review = json.loads((tmp_path / "subset/pre-state-bank/review_rows.jsonl").read_text())
    assert selected_review["event_id"] == "event-1"
    assert selected_review["image_balanced_event_weight"] == 1.0
    with pytest.raises(SubsetError, match="output already exists"):
        build_subset(manifest, tmp_path / "subset", event_ids=["event-1"])


def test_selection_rejects_mixed_modes_and_unknown_ids() -> None:
    rollout = {"event_id": "e", "image": {"image_id": 1}}
    review = {
        "event_id": "e",
        "positive_path_imitation_eligible": False,
        "source_route_imitation_eligible": True,
    }
    with pytest.raises(SubsetError, match="exactly one"):
        select_event_pairs([rollout], [review], event_ids=["e"], family_counts={"source_route_imitation": 1})
    with pytest.raises(SubsetError, match="unknown event IDs"):
        select_event_pairs([rollout], [review], event_ids=["missing"])
