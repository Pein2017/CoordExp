"""Contracts for the CPU-only sorted false-negative mechanism registry builder."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest

from scripts.research.build_sorted_fn_mechanism_registry import (
    BOUND_NON_TARGETS,
    DEFAULT_SMOKE_COLLISION_PAIRS,
    TARGET_REGISTRY,
    RegistryError,
    build_sorted_fn_mechanism_registry,
    canonical_json_bytes,
    sha256_json,
)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        for row in rows:
            handle.write(canonical_json_bytes(row))
            handle.write(b"\n")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


# Frozen geometry/description overrides for owners that participate in
# collision, null-pair, or first-skip checks. Every other registry owner
# gets an arbitrary, non-overlapping default box from a counter.
_OWNER_OVERRIDES: dict[str, dict[str, Any]] = {
    "gt:7511:26": {"description": "person", "bbox": [100, 100, 110, 110], "idx": 26},
    "gt:7511:22": {"description": "person", "bbox": [105, 105, 115, 115], "idx": 22},
    # gt:7511:700 is a synthetic covering owner distinct from gt:7511:22: the
    # natural greedy row at the baseline cut boundary (row 3) strict-matches
    # gt:7511:22, so gt:7511:22 can never be a *genuine counterfactual*
    # covering insertion against this baseline (it is already what the
    # natural pre-pass continuation produces). Null pairs and the synthetic
    # second collision pair use gt:7511:700 instead.
    "gt:7511:700": {"description": "person", "bbox": [103, 103, 109, 109], "idx": 700},
    "gt:7511:24": {"description": "person", "bbox": [200, 200, 210, 210], "idx": 24},
    "gt:7511:29": {"description": "person", "bbox": [300, 300, 310, 310], "idx": 29},
    "gt:7511:1": {"description": "kite", "bbox": [10, 10, 20, 20], "idx": 1},
    "gt:7511:2": {"description": "person", "bbox": [50, 60, 60, 70], "idx": 2},
    "gt:1584:3": {"description": "person", "bbox": [400, 400, 410, 410], "idx": 3},
    "gt:1584:11": {"description": "person", "bbox": [405, 405, 415, 415], "idx": 11},
    "gt:1584:9": {"description": "person", "bbox": [600, 600, 610, 610], "idx": 9},
}


def _owner_row(gt_owner_id: str, image_id: str, *, counter: int) -> dict[str, Any]:
    override = _OWNER_OVERRIDES.get(gt_owner_id)
    if override is not None:
        description, bbox, idx = override["description"], override["bbox"], override["idx"]
    else:
        base = 1000 + counter
        description, bbox, idx = "person", [base, base, base + 10, base + 10], base
    return {
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "normalized_description": description,
        "bbox_xyxy": bbox,
        "original_annotation_index": idx,
    }


def _cohort_row(gt_owner_id: str, image_id: str, cohort: str) -> dict[str, Any]:
    return {
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "cohort": cohort,
        "foreign_keys": {"supporting_pred_row_ids": []},
    }


def _pred_row(
    pred_row_id: str,
    *,
    decode_mode: str,
    seed: int,
    strict_match_gt_owner_id: str | None,
    description: str = "person",
) -> dict[str, Any]:
    return {
        "pred_row_id": pred_row_id,
        "decode_mode": decode_mode,
        "seed": seed,
        "strict_match_gt_owner_id": strict_match_gt_owner_id,
        "normalized_description": description,
    }


def _rollout_entry(
    *,
    image_id: str,
    decode_mode: str,
    seed: int,
    row_count: int,
    stop_reason: str = "im_end",
    row_bboxes: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    prompt_token_ids = [1, 2, 3]
    generated_token_ids: list[int] = []
    predictions = []
    row_bboxes = row_bboxes or {}
    for index in range(row_count):
        chunk = [151646, 900000 + seed + index, 151649]
        generated_token_ids.extend(chunk)
        # Default bbox is far from every registered owner used in these
        # fixtures; only rows actually used as a G/F insertion get an
        # explicit override below.
        default_bbox = [9000 + seed + index, 9000 + seed + index, 9005 + seed + index, 9005 + seed + index]
        predictions.append(
            {
                "generated_order": index,
                "raw_span_sha256": f"span-{image_id}-{decode_mode}-{seed}-{index}",
                "bbox": row_bboxes.get(index, default_bbox),
            }
        )
    return {
        "decode_mode": decode_mode,
        "example_id": f"example-{image_id}",
        "image_id": int(image_id),
        "seed": seed,
        "stop_reason": stop_reason,
        "prompt_token_ids": prompt_token_ids,
        "generated_token_ids": generated_token_ids,
        "predictions": {"predictions": predictions},
    }


# Inserted-row predicted bboxes (not owner GT bboxes) used to satisfy the
# covering-row positive-overlap / foil-row exact-zero-overlap contract
# against the frozen target bboxes declared in _OWNER_OVERRIDES.
_COVERING_ROW_BBOX_7511 = [105, 105, 115, 115]  # overlaps target gt:7511:26 [100,100,110,110]
_FOIL_ROW_BBOX_7511_24 = [200, 200, 210, 210]  # does not overlap
_FOIL_ROW_BBOX_7511_29 = [300, 300, 310, 310]  # does not overlap
_COVERING_ROW_BBOX_1584 = [405, 405, 415, 415]  # overlaps target gt:1584:3 [400,400,410,410]
_FOIL_ROW_BBOX_1584_9 = [600, 600, 610, 610]  # does not overlap


# An independently valid, explicitly reviewed collision-pair spec: target
# gt:7511:26, covering gt:7511:700 (overlaps target, not the natural row at
# the baseline cut boundary), foil gt:7511:24 (does not overlap target).
# Mirrors a null-pair spec exactly -- arbitrary target/image/baseline, not a
# hardcoded default.
_VALID_COLLISION_PAIR = {
    "pair_id": "seed-21001-valid",
    "image_id": "7511",
    "review_status": "reviewed",
    "target_gt_owner_id": "gt:7511:26",
    "baseline": {
        "decode_mode": "greedy",
        "seed": 0,
        "cut_before_pred_row_id": "pred:sorted:greedy:0:7511:3",
    },
    "covering": {"pred_row_id": "pred:sorted:sampled:21001:7511:0", "gt_owner_id": "gt:7511:700"},
    "foil": {"pred_row_id": "pred:sorted:sampled:21001:7511:1", "gt_owner_id": "gt:7511:24"},
}

# A second valid pair sharing the identical baseline (same execution tensor)
# but bound to a distinct pair_id/foil, for the seed-pair-distinctness test.
_VALID_COLLISION_PAIR_B = {
    "pair_id": "seed-21002-valid",
    "image_id": "7511",
    "review_status": "reviewed",
    "target_gt_owner_id": "gt:7511:26",
    "baseline": {
        "decode_mode": "greedy",
        "seed": 0,
        "cut_before_pred_row_id": "pred:sorted:greedy:0:7511:3",
    },
    "covering": {"pred_row_id": "pred:sorted:sampled:21002:7511:0", "gt_owner_id": "gt:7511:700"},
    "foil": {"pred_row_id": "pred:sorted:sampled:21002:7511:1", "gt_owner_id": "gt:7511:29"},
}

# The two frozen pairs proposed in earlier unit.md drafts, kept only here
# (not in production code). seed-21011 remains invalid (its foil overlaps
# the target). seed-21003's covering owner gt:7511:22 is the strict-matched
# owner of the natural row at the cut boundary (pred:sorted:greedy:0:7511:3)
# -- P's natural successor, not something strictly earlier in P -- so under
# the "novelty relative to literal P only" contract it is *admitted*, not
# rejected; see test_explicitly_supplied_seed_21003_pair_is_admitted.
_SEED_21003_COLLISION_PAIR = {
    "pair_id": "seed-21003",
    "image_id": "7511",
    "review_status": "reviewed",
    "target_gt_owner_id": "gt:7511:26",
    "baseline": {
        "decode_mode": "greedy",
        "seed": 0,
        "cut_before_pred_row_id": "pred:sorted:greedy:0:7511:3",
    },
    "covering": {"pred_row_id": "pred:sorted:sampled:21003:7511:4", "gt_owner_id": "gt:7511:22"},
    "foil": {"pred_row_id": "pred:sorted:sampled:21003:7511:6", "gt_owner_id": "gt:7511:24"},
}

_INVALID_SEED_21011_COLLISION_PAIR = {
    "pair_id": "seed-21011",
    "image_id": "7511",
    "review_status": "reviewed",
    "target_gt_owner_id": "gt:7511:26",
    "baseline": {
        "decode_mode": "greedy",
        "seed": 0,
        "cut_before_pred_row_id": "pred:sorted:greedy:0:7511:3",
    },
    # covering=22 is the same natural-successor case as seed-21003 above
    # (admitted on its own); foil=29 has a positive intersection with the
    # target under this fixture's synthetic geometry, which is what still
    # rejects this pair.
    "covering": {"pred_row_id": "pred:sorted:sampled:21011:7511:5", "gt_owner_id": "gt:7511:22"},
    "foil": {"pred_row_id": "pred:sorted:sampled:21011:7511:4", "gt_owner_id": "gt:7511:29"},
}


def _fixture(root: Path) -> dict[str, Any]:
    owner_rows = [
        _owner_row(entry["gt_owner_id"], entry["image_id"], counter=index)
        for index, entry in enumerate((*TARGET_REGISTRY, *BOUND_NON_TARGETS))
    ]
    # gt:7511:700 is a synthetic covering owner used by the null pairs and
    # the alternate collision pair below; it is not part of the 24/9-owner
    # registry, only referenced for its geometry/description.
    owner_rows.append(_owner_row("gt:7511:700", "7511", counter=700))
    cohort_rows = [
        _cohort_row(entry["gt_owner_id"], entry["image_id"], entry["expected_cohort"])
        for entry in (*TARGET_REGISTRY, *BOUND_NON_TARGETS)
    ]

    owner_ledger = root / "owner-ledger.jsonl"
    cohort_assignments = root / "cohort-assignments.jsonl"
    _write_jsonl(owner_ledger, owner_rows)
    _write_jsonl(cohort_assignments, cohort_rows)

    greedy_7511 = _rollout_entry(image_id="7511", decode_mode="greedy", seed=0, row_count=4)
    greedy_1584 = _rollout_entry(image_id="1584", decode_mode="greedy", seed=0, row_count=2)
    greedy_document = {
        "schema_version": "fixture-rollout.v1",
        "rollouts": [greedy_7511, greedy_1584],
    }
    greedy_path = root / "greedy.json"
    _write_json(greedy_path, greedy_document)

    sampled_document = {
        "schema_version": "fixture-rollout.v1",
        "rollouts": [
            # Mechanical/calibrating null-pair sealed-witness rows on 7511:
            # row 0 = G (covering, at the cut), row 1 = F (foil, novel in P),
            # row 2 = T's later natural release.
            _rollout_entry(
                image_id="7511",
                decode_mode="sampled",
                seed=21001,
                row_count=3,
                row_bboxes={0: _COVERING_ROW_BBOX_7511, 1: _FOIL_ROW_BBOX_7511_24},
            ),
            _rollout_entry(
                image_id="7511",
                decode_mode="sampled",
                seed=21002,
                row_count=3,
                row_bboxes={0: _COVERING_ROW_BBOX_7511, 1: _FOIL_ROW_BBOX_7511_29},
            ),
            # Calibrating null-pair sealed-witness rows on 1584.
            _rollout_entry(
                image_id="1584",
                decode_mode="sampled",
                seed=21001,
                row_count=3,
                row_bboxes={0: _COVERING_ROW_BBOX_1584, 1: _FOIL_ROW_BBOX_1584_9},
            ),
            # Collision seed pairs and the strict-rescue due-turn trajectory.
            _rollout_entry(
                image_id="7511",
                decode_mode="sampled",
                seed=21003,
                row_count=7,
                row_bboxes={4: _COVERING_ROW_BBOX_7511, 6: _FOIL_ROW_BBOX_7511_24},
            ),
            _rollout_entry(image_id="7511", decode_mode="sampled", seed=21010, row_count=3),
            # seed-21011 is excluded from the default collision-pair set (its
            # frozen foil overlaps the target); its row 4 bbox is left
            # intentionally overlapping the target -- despite a
            # non-overlapping owner GT box -- to exercise the explicit-supply
            # rejection path in a dedicated test below.
            _rollout_entry(
                image_id="7511",
                decode_mode="sampled",
                seed=21011,
                row_count=6,
                row_bboxes={5: _COVERING_ROW_BBOX_7511, 4: [100, 100, 110, 110]},
            ),
        ],
    }
    sampled_path = root / "sampled.json"
    _write_json(sampled_path, sampled_document)

    prediction_rows = [
        _pred_row("pred:sorted:greedy:0:7511:0", decode_mode="greedy", seed=0, strict_match_gt_owner_id=None, description="kite"),
        _pred_row("pred:sorted:greedy:0:7511:1", decode_mode="greedy", seed=0, strict_match_gt_owner_id=None),
        _pred_row("pred:sorted:greedy:0:7511:2", decode_mode="greedy", seed=0, strict_match_gt_owner_id="gt:7511:2"),
        _pred_row("pred:sorted:greedy:0:7511:3", decode_mode="greedy", seed=0, strict_match_gt_owner_id="gt:7511:22"),
        _pred_row("pred:sorted:greedy:0:1584:0", decode_mode="greedy", seed=0, strict_match_gt_owner_id=None),
        _pred_row("pred:sorted:greedy:0:1584:1", decode_mode="greedy", seed=0, strict_match_gt_owner_id=None),
        # Mechanical null pair (target gt:7511:26): G=700 at row0 (the cut),
        # F=24 at row1 (novel in P), T=26 released later at row2.
        _pred_row("pred:sorted:sampled:21001:7511:0", decode_mode="sampled", seed=21001, strict_match_gt_owner_id="gt:7511:700"),
        _pred_row("pred:sorted:sampled:21001:7511:1", decode_mode="sampled", seed=21001, strict_match_gt_owner_id="gt:7511:24"),
        _pred_row("pred:sorted:sampled:21001:7511:2", decode_mode="sampled", seed=21001, strict_match_gt_owner_id="gt:7511:26"),
        # Calibrating null pair on 7511 (target gt:7511:26): G=700, F=29, T=26.
        _pred_row("pred:sorted:sampled:21002:7511:0", decode_mode="sampled", seed=21002, strict_match_gt_owner_id="gt:7511:700"),
        _pred_row("pred:sorted:sampled:21002:7511:1", decode_mode="sampled", seed=21002, strict_match_gt_owner_id="gt:7511:29"),
        _pred_row("pred:sorted:sampled:21002:7511:2", decode_mode="sampled", seed=21002, strict_match_gt_owner_id="gt:7511:26"),
        # Calibrating null pair on 1584 (target gt:1584:3): G=11, F=9, T=3.
        _pred_row("pred:sorted:sampled:21001:1584:0", decode_mode="sampled", seed=21001, strict_match_gt_owner_id="gt:1584:11"),
        _pred_row("pred:sorted:sampled:21001:1584:1", decode_mode="sampled", seed=21001, strict_match_gt_owner_id="gt:1584:9"),
        _pred_row("pred:sorted:sampled:21001:1584:2", decode_mode="sampled", seed=21001, strict_match_gt_owner_id="gt:1584:3"),
        # Collision seed-21003 pair: covering=22, foil=24.
        _pred_row("pred:sorted:sampled:21003:7511:4", decode_mode="sampled", seed=21003, strict_match_gt_owner_id="gt:7511:22"),
        _pred_row("pred:sorted:sampled:21003:7511:6", decode_mode="sampled", seed=21003, strict_match_gt_owner_id="gt:7511:24"),
        # Strict-rescue due-turn row.
        _pred_row("pred:sorted:sampled:21010:7511:2", decode_mode="sampled", seed=21010, strict_match_gt_owner_id="gt:7511:17"),
        # Collision seed-21011 pair: covering=22, foil=29.
        _pred_row("pred:sorted:sampled:21011:7511:4", decode_mode="sampled", seed=21011, strict_match_gt_owner_id="gt:7511:29"),
        _pred_row("pred:sorted:sampled:21011:7511:5", decode_mode="sampled", seed=21011, strict_match_gt_owner_id="gt:7511:22"),
    ]
    prediction_row_ledger = root / "prediction-row-ledger.jsonl"
    _write_jsonl(prediction_row_ledger, prediction_rows)

    null_pairs_plan = [
        {
            "pair_id": "null-mechanical-7511",
            "image_id": "7511",
            "is_mechanical": True,
            "review_status": "reviewed",
            "target_gt_owner_id": "gt:7511:26",
            "baseline": {"decode_mode": "sampled", "seed": 21001},
            "covering": {"pred_row_id": "pred:sorted:sampled:21001:7511:0", "gt_owner_id": "gt:7511:700"},
            "foil": {"pred_row_id": "pred:sorted:sampled:21001:7511:1", "gt_owner_id": "gt:7511:24"},
            "target_release_pred_row_id": "pred:sorted:sampled:21001:7511:2",
        },
        {
            "pair_id": "null-calibrating-7511",
            "image_id": "7511",
            "is_mechanical": False,
            "review_status": "reviewed",
            "target_gt_owner_id": "gt:7511:26",
            "baseline": {"decode_mode": "sampled", "seed": 21002},
            "covering": {"pred_row_id": "pred:sorted:sampled:21002:7511:0", "gt_owner_id": "gt:7511:700"},
            "foil": {"pred_row_id": "pred:sorted:sampled:21002:7511:1", "gt_owner_id": "gt:7511:29"},
            "target_release_pred_row_id": "pred:sorted:sampled:21002:7511:2",
        },
        {
            "pair_id": "null-calibrating-1584",
            "image_id": "1584",
            "is_mechanical": False,
            "review_status": "reviewed",
            "target_gt_owner_id": "gt:1584:3",
            "baseline": {"decode_mode": "sampled", "seed": 21001},
            "covering": {"pred_row_id": "pred:sorted:sampled:21001:1584:0", "gt_owner_id": "gt:1584:11"},
            "foil": {"pred_row_id": "pred:sorted:sampled:21001:1584:1", "gt_owner_id": "gt:1584:9"},
            "target_release_pred_row_id": "pred:sorted:sampled:21001:1584:2",
        },
    ]

    return {
        "owner_ledger": owner_ledger,
        "prediction_row_ledger": prediction_row_ledger,
        "cohort_assignments": cohort_assignments,
        "rollouts": [greedy_path, sampled_path],
        "due_turn_trajectories": {"gt:7511:17": ("sampled", 21010)},
        # The production default is empty (see DEFAULT_SMOKE_COLLISION_PAIRS);
        # this test suite's own fixture default supplies one independently
        # valid, explicitly reviewed pair so most tests exercise real
        # collision-role behavior. test_default_collision_pairs_is_empty
        # covers the production default directly.
        "collision_pairs": [_VALID_COLLISION_PAIR],
        "null_pairs": null_pairs_plan,
    }


def _build(root: Path, output: Path, **overrides: Any) -> dict[str, Any]:
    fixture = _fixture(root)
    fixture.update(overrides)
    return build_sorted_fn_mechanism_registry(
        owner_ledger=fixture["owner_ledger"],
        prediction_row_ledger=fixture["prediction_row_ledger"],
        cohort_assignments=fixture["cohort_assignments"],
        rollouts=fixture["rollouts"],
        due_turn_trajectories=fixture["due_turn_trajectories"],
        collision_pairs=fixture["collision_pairs"],
        null_pairs=fixture["null_pairs"],
        output=output,
    )


def _mutate_owner_rows(fixture: dict[str, Any], mutator) -> None:
    lines = fixture["owner_ledger"].read_text(encoding="utf-8").splitlines()
    mutated = []
    for line in lines:
        row = json.loads(line)
        mutator(row)
        mutated.append(json.dumps(row))
    fixture["owner_ledger"].write_text("\n".join(mutated) + "\n", encoding="utf-8")


def _append_owner_row(fixture: dict[str, Any], row: dict[str, Any]) -> None:
    with fixture["owner_ledger"].open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")


def _append_prediction_rows(fixture: dict[str, Any], rows: list[dict[str, Any]]) -> None:
    with fixture["prediction_row_ledger"].open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _append_rollout(fixture: dict[str, Any], rollout: dict[str, Any]) -> None:
    for rollout_path in fixture["rollouts"]:
        document = json.loads(rollout_path.read_text(encoding="utf-8"))
        if rollout["decode_mode"] == "sampled" and rollout_path.name == "sampled.json":
            document["rollouts"].append(rollout)
            rollout_path.write_text(json.dumps(document), encoding="utf-8")
            return
    raise AssertionError("no matching rollout artifact found to append to")


def _mutate_prediction_rows(fixture: dict[str, Any], mutator) -> None:
    lines = fixture["prediction_row_ledger"].read_text(encoding="utf-8").splitlines()
    mutated = []
    for line in lines:
        row = json.loads(line)
        mutator(row)
        mutated.append(json.dumps(row))
    fixture["prediction_row_ledger"].write_text("\n".join(mutated) + "\n", encoding="utf-8")


def _build_with_fixture(fixture: dict[str, Any], output: Path) -> dict[str, Any]:
    return build_sorted_fn_mechanism_registry(
        owner_ledger=fixture["owner_ledger"],
        prediction_row_ledger=fixture["prediction_row_ledger"],
        cohort_assignments=fixture["cohort_assignments"],
        rollouts=fixture["rollouts"],
        due_turn_trajectories=fixture["due_turn_trajectories"],
        collision_pairs=fixture["collision_pairs"],
        null_pairs=fixture["null_pairs"],
        output=output,
    )


# ---------------------------------------------------------------------------
# Structural sanity
# ---------------------------------------------------------------------------


def test_frozen_tables_have_exactly_24_targets_and_no_overlap_with_non_targets() -> None:
    assert len(TARGET_REGISTRY) == 24
    target_ids = {entry["gt_owner_id"] for entry in TARGET_REGISTRY}
    non_target_ids = {entry["gt_owner_id"] for entry in BOUND_NON_TARGETS}
    assert len(target_ids) == 24
    assert target_ids.isdisjoint(non_target_ids)


def test_registry_builds_two_layer_cohort_and_smoke_roles(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")

    cohort = document["mechanism_cohort"]
    assert cohort["owner_count"] == 24
    assert len(cohort["targets"]) == 24
    assert cohort["prevalence_denominator_owner_ids"] == sorted(
        entry["gt_owner_id"] for entry in TARGET_REGISTRY
    )

    context = document["context_control_registry"]
    context_ids = {row["gt_owner_id"] for row in context["bound_non_targets"]}
    assert context_ids == {entry["gt_owner_id"] for entry in BOUND_NON_TARGETS}
    assert context_ids.isdisjoint(cohort["prevalence_denominator_owner_ids"])

    roles = document["smoke"]["roles"]
    role_kinds = {role["role_kind"] for role in roles}
    assert {
        "root_context",
        "due_turn_context",
        "collision_baseline",
        "collision_covering",
        "collision_foil",
        "first_skip_pre",
        "first_skip_post",
    } <= role_kinds

    envelope = document["smoke"]["null_pair_envelope"]
    assert envelope["status"] == "sufficient"
    assert envelope["pair_count"] == 3
    assert envelope["mechanical_pair_count"] == 1
    assert envelope["calibrating_pair_count"] == 2
    assert envelope["distinct_image_count"] == 2


# ---------------------------------------------------------------------------
# P0-1: actual token_ids, hashed directly (no nested digest payload)
# ---------------------------------------------------------------------------


def _all_prefix_bearing_roles(document: dict[str, Any]) -> list[dict[str, Any]]:
    roles = list(document["smoke"]["roles"])
    for pair in document["smoke"]["null_pair_envelope"]["pairs"]:
        roles.extend(pair["roles"])
    return roles


def test_every_prefix_digest_recomputes_from_its_own_stored_token_ids(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    roles = _all_prefix_bearing_roles(document)
    assert roles, "expected at least one prefix-bearing role"
    for role in roles:
        prefix = role["prefix"]
        assert isinstance(prefix["token_ids"], list) and prefix["token_ids"]
        assert prefix["token_ids_sha256"] == sha256_json(prefix["token_ids"])
        assert prefix["token_count"] == len(prefix["token_ids"])


def test_appended_row_prefix_is_literal_concatenation_of_tokens(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    roles_by_id = {role["role_id"]: role for role in document["smoke"]["roles"]}
    baseline = roles_by_id["collision:seed-21001-valid:P"]["prefix"]
    covering = roles_by_id["collision:seed-21001-valid:P+G"]["prefix"]

    assert covering["token_ids"][: len(baseline["token_ids"])] == baseline["token_ids"]
    appended = covering["token_ids"][len(baseline["token_ids"]) :]
    assert len(appended) == covering["appended_row_token_count"]
    assert covering["token_ids"] == [*baseline["token_ids"], *appended]
    assert covering["token_ids_sha256"] == sha256_json(covering["token_ids"])
    # This is exactly the case the earlier nested-digest bug could not catch:
    # the digest must depend on the literal appended tokens, not merely on
    # the baseline digest plus an opaque nested payload.
    assert covering["token_ids_sha256"] != sha256_json(
        {"baseline_token_ids_sha256": baseline["token_ids_sha256"], "appended_tokens": appended}
    )


# ---------------------------------------------------------------------------
# P0-2: collision target/inserted-owner semantics
# ---------------------------------------------------------------------------


def test_default_collision_pairs_is_empty(tmp_path: Path) -> None:
    """The production default ships with the collision contrast unresolved
    rather than a fabricated pair; a caller must explicitly opt in."""

    assert DEFAULT_SMOKE_COLLISION_PAIRS == ()
    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = None
    document = _build_with_fixture(fixture, tmp_path / "registry.json")
    roles = document["smoke"]["roles"]
    assert not any(role["role_kind"].startswith("collision_") for role in roles)
    # 2 root/due-turn for gt:7511:22, 1 root-only for gt:7511:26, 2
    # root/due-turn for gt:7511:17, 0 collision, 2 first-skip.
    assert len(roles) == 7
    assert document["smoke"]["collision_target_gt_owner_ids"] == []


def test_collision_roles_use_the_pair_supplied_target_owner(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    roles = document["smoke"]["roles"]
    collision_roles = [role for role in roles if role["role_kind"].startswith("collision_")]
    assert collision_roles, "expected collision roles"
    for role in collision_roles:
        assert role["gt_owner_id"] == "gt:7511:26"
    assert document["smoke"]["collision_target_gt_owner_ids"] == ["gt:7511:26"]
    covering = [role for role in collision_roles if role["role_kind"] == "collision_covering"]
    foil = [role for role in collision_roles if role["role_kind"] == "collision_foil"]
    assert {role["inserted_gt_owner_id"] for role in covering} == {"gt:7511:700"}
    assert {role["inserted_gt_owner_id"] for role in foil} == {"gt:7511:24"}
    # The inserted owner is never the tested owner.
    for role in covering + foil:
        assert role["inserted_gt_owner_id"] != role["gt_owner_id"]
    # The covering-owner overlap and foil exact-zero-overlap are both
    # recorded explicitly, at the owner-geometry and inserted-row levels.
    for role in covering:
        assert role["target_covering_owner_intersection_area"] > 0.0
        assert role["target_covering_row_intersection_area"] > 0.0
    for role in foil:
        assert role["target_foil_owner_intersection_area"] == 0.0
        assert role["target_foil_row_intersection_area"] == 0.0


def test_each_seed_pair_gets_its_own_distinct_baseline_p_role(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = [_VALID_COLLISION_PAIR, _VALID_COLLISION_PAIR_B]
    document = _build_with_fixture(fixture, tmp_path / "registry.json")
    roles_by_id = {role["role_id"]: role for role in document["smoke"]["roles"]}
    p_a = roles_by_id["collision:seed-21001-valid:P"]
    p_b = roles_by_id["collision:seed-21002-valid:P"]
    assert p_a["role_id"] != p_b["role_id"]
    # Same execution tensor (identical cut of the identical greedy trajectory)...
    assert p_a["prefix"]["token_ids_sha256"] == p_b["prefix"]["token_ids_sha256"]
    # ...but two logically distinct receipt rows, each bound to its own pair.
    assert {p_a["role_id"], p_b["role_id"]} == {"collision:seed-21001-valid:P", "collision:seed-21002-valid:P"}


def test_explicitly_supplied_seed_21003_pair_is_admitted(tmp_path: Path) -> None:
    """Novelty is judged relative to literal P only: gt:7511:22 is P's
    natural successor (the row exactly at the cut boundary), not something
    strictly earlier in P, so it is a legitimate witnessed continuation --
    the pair is admitted, and the covering role records exactly why."""

    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = [_SEED_21003_COLLISION_PAIR]
    document = _build_with_fixture(fixture, tmp_path / "registry.json")
    covering = next(
        role
        for role in document["smoke"]["roles"]
        if role["role_id"] == "collision:seed-21003:P+G"
    )
    assert covering["natural_successor_pred_row_id"] == "pred:sorted:greedy:0:7511:3"
    assert covering["natural_successor_gt_owner_id"] == "gt:7511:22"
    assert covering["covering_owner_equals_natural_successor_owner"] is True
    # The covering row itself is spliced in from a different (sampled)
    # trajectory, so it is not literally the baseline's own successor row.
    assert covering["covering_pred_row_id_equals_natural_successor_pred_row_id"] is False


def test_explicitly_supplied_seed_21011_pair_is_still_rejected(tmp_path: Path) -> None:
    """seed-21011 is excluded from the default set. Its covering owner is
    the same P-natural-successor case as seed-21003 above (admitted on its
    own), but its foil still has a positive intersection with the target, so
    the pair as a whole is still rejected."""

    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = [_INVALID_SEED_21011_COLLISION_PAIR]
    with pytest.raises(RegistryError, match="foil-row contract requires exact zero overlap"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_collision_covering_owner_present_strictly_inside_p_is_rejected(tmp_path: Path) -> None:
    """An owner already covered strictly *before* the cut (a real part of P,
    unlike the natural-successor row exactly at the cut) is still a genuine
    duplicate insertion and must be rejected."""

    fixture = _fixture(tmp_path)
    _append_owner_row(
        fixture,
        {
            "gt_owner_id": "gt:7511:800",
            "image_id": "7511",
            "normalized_description": "person",
            "bbox_xyxy": [104, 104, 108, 108],
            "original_annotation_index": 800,
        },
    )
    # Row 1 (index 1, strictly before the cut at index 3) is retargeted to
    # strict-match gt:7511:800: it is now genuinely inside P.
    _mutate_prediction_rows(
        fixture,
        lambda row: row.update({"strict_match_gt_owner_id": "gt:7511:800"})
        if row["pred_row_id"] == "pred:sorted:greedy:0:7511:1"
        else None,
    )
    _append_rollout(
        fixture,
        _rollout_entry(
            image_id="7511",
            decode_mode="sampled",
            seed=21031,
            row_count=1,
            row_bboxes={0: _COVERING_ROW_BBOX_7511},
        ),
    )
    _append_prediction_rows(
        fixture,
        [
            _pred_row(
                "pred:sorted:sampled:21031:7511:0",
                decode_mode="sampled",
                seed=21031,
                strict_match_gt_owner_id="gt:7511:800",
            )
        ],
    )
    fixture["collision_pairs"] = [
        {
            "pair_id": "owner-earlier-in-p",
            "image_id": "7511",
            "review_status": "reviewed",
            "target_gt_owner_id": "gt:7511:26",
            "baseline": {
                "decode_mode": "greedy",
                "seed": 0,
                "cut_before_pred_row_id": "pred:sorted:greedy:0:7511:3",
            },
            "covering": {"pred_row_id": "pred:sorted:sampled:21031:7511:0", "gt_owner_id": "gt:7511:800"},
            "foil": {"pred_row_id": "pred:sorted:sampled:21001:7511:1", "gt_owner_id": "gt:7511:24"},
        }
    ]
    with pytest.raises(RegistryError, match="is not a genuine counterfactual insertion"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_collision_covering_owner_non_overlapping_target_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _mutate_owner_rows(
        fixture,
        lambda row: row.update({"bbox_xyxy": [9000, 9000, 9010, 9010]})
        if row["gt_owner_id"] == "gt:7511:700"
        else None,
    )
    with pytest.raises(RegistryError, match="does not spatially overlap target"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_collision_covering_row_non_overlapping_target_is_rejected(tmp_path: Path) -> None:
    """The covering *owner*'s registered GT box still overlaps C, but the
    literal inserted prediction row's own bbox does not -- strict owner
    match alone must not be enough."""

    fixture = _fixture(tmp_path)
    sampled_path = tmp_path / "sampled.json"
    document = json.loads(sampled_path.read_text(encoding="utf-8"))
    for rollout in document["rollouts"]:
        if rollout["seed"] == 21001 and rollout["decode_mode"] == "sampled" and rollout["image_id"] == 7511:
            rollout["predictions"]["predictions"][0]["bbox"] = [9000, 9000, 9010, 9010]
    sampled_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError, match="predicted bbox.*does not spatially overlap target"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_covering_owner_non_overlapping_target_is_rejected(tmp_path: Path) -> None:
    # Disable the fixture's default collision pair (which also uses
    # gt:7511:700 as its covering owner) so this exercises null-pair
    # validation specifically, not the collision path.
    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = []
    _mutate_owner_rows(
        fixture,
        lambda row: row.update({"bbox_xyxy": [9000, 9000, 9010, 9010]})
        if row["gt_owner_id"] == "gt:7511:700"
        else None,
    )
    with pytest.raises(RegistryError, match="does not spatially overlap target"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_covering_row_non_overlapping_target_is_rejected(tmp_path: Path) -> None:
    """Same as above, but the owner-level GT geometry still overlaps C; only
    the literal inserted prediction row's own bbox does not."""

    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = []
    sampled_path = tmp_path / "sampled.json"
    document = json.loads(sampled_path.read_text(encoding="utf-8"))
    for rollout in document["rollouts"]:
        if rollout["seed"] == 21001 and rollout["decode_mode"] == "sampled" and rollout["image_id"] == 7511:
            rollout["predictions"]["predictions"][0]["bbox"] = [9000, 9000, 9010, 9010]
    sampled_path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(RegistryError, match="predicted bbox.*does not spatially overlap target"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_collision_foil_overlapping_target_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _mutate_owner_rows(
        fixture,
        lambda row: row.update({"bbox_xyxy": [100, 100, 110, 110]})
        if row["gt_owner_id"] == "gt:7511:24"
        else None,
    )
    with pytest.raises(RegistryError, match="spatially overlaps target"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_collision_pair_description_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _mutate_owner_rows(
        fixture,
        lambda row: row.update({"normalized_description": "cat"})
        if row["gt_owner_id"] == "gt:7511:24"
        else None,
    )
    with pytest.raises(RegistryError, match="normalized_description"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_collision_pair_without_explicit_reviewed_status_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = [{**_VALID_COLLISION_PAIR, "review_status": "unreviewed"}]
    with pytest.raises(RegistryError, match="review_status"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


# ---------------------------------------------------------------------------
# P0-3: null-pair target/inserted-owner and evidence requirements
# ---------------------------------------------------------------------------


def test_null_roles_keep_target_identity_and_expose_inserted_owner(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    pairs = document["smoke"]["null_pair_envelope"]["pairs"]
    by_id = {pair["pair_id"]: pair for pair in pairs}
    mechanical = by_id["null-mechanical-7511"]
    assert mechanical["target_gt_owner_id"] == "gt:7511:26"
    assert mechanical["target_release_pred_row_id"] == "pred:sorted:sampled:21001:7511:2"
    baseline, covering, foil, release = mechanical["roles"]
    assert (
        baseline["gt_owner_id"]
        == covering["gt_owner_id"]
        == foil["gt_owner_id"]
        == release["gt_owner_id"]
        == "gt:7511:26"
    )
    assert "inserted_gt_owner_id" not in baseline
    assert covering["inserted_gt_owner_id"] == "gt:7511:700"
    assert foil["inserted_gt_owner_id"] == "gt:7511:24"
    assert covering["pred_row_id"] == "pred:sorted:sampled:21001:7511:0"
    assert foil["pred_row_id"] == "pred:sorted:sampled:21001:7511:1"
    assert release["pred_row_id"] == "pred:sorted:sampled:21001:7511:2"
    assert covering["witnessed_history"] is True
    assert foil["witnessed_history"] is False
    assert covering["raw_span_sha256"] and foil["raw_span_sha256"]
    assert covering["provenance"]["source_artifact_sha256"]
    assert foil["provenance"]["source_artifact_sha256"]
    # For a null pair, G *is* the row at the cut by construction, so its
    # witnessed-successor facts are trivially, but explicitly, true.
    assert covering["natural_successor_pred_row_id"] == covering["pred_row_id"]
    assert covering["natural_successor_gt_owner_id"] == covering["inserted_gt_owner_id"]
    assert covering["covering_owner_equals_natural_successor_owner"] is True
    assert covering["covering_pred_row_id_equals_natural_successor_pred_row_id"] is True


def test_null_pair_wrong_declared_owner_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    # gt:7511:601 shares the "person" description, overlaps the target, and
    # is distinct from the target -- but the covering row actually
    # strict-matches gt:7511:700 in this trajectory, so declaring it as the
    # covering owner must be rejected rather than inferred.
    _append_owner_row(
        fixture,
        {
            "gt_owner_id": "gt:7511:601",
            "image_id": "7511",
            "normalized_description": "person",
            "bbox_xyxy": [102, 102, 108, 108],
            "original_annotation_index": 601,
        },
    )
    fixture["null_pairs"][0]["covering"]["gt_owner_id"] = "gt:7511:601"
    with pytest.raises(RegistryError, match="ambiguous binding is rejected"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_foil_overlapping_target_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    _mutate_owner_rows(
        fixture,
        lambda row: row.update({"bbox_xyxy": [100, 100, 110, 110]})
        if row["gt_owner_id"] == "gt:7511:24"
        else None,
    )
    with pytest.raises(RegistryError, match="spatially overlaps target"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_same_physical_owner_as_target_is_rejected(tmp_path: Path) -> None:
    # Disable the fixture's default collision pair, which also references
    # this same row, so only null-pair validation is exercised.
    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = []
    fixture["null_pairs"][0]["covering"]["gt_owner_id"] = "gt:7511:26"
    fixture["null_pairs"][0]["covering"]["pred_row_id"] = "pred:sorted:sampled:21001:7511:0"
    # Ledger still says this row strict-matches gt:7511:700, not the target;
    # patch the ledger too so the *only* failure exercised is the
    # distinct-owner check, not the strict-match check.
    _mutate_prediction_rows(
        fixture,
        lambda row: row.update({"strict_match_gt_owner_id": "gt:7511:26"})
        if row["pred_row_id"] == "pred:sorted:sampled:21001:7511:0"
        else None,
    )
    with pytest.raises(RegistryError, match="pairwise distinct"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


@pytest.mark.parametrize(
    "review_status",
    [None, "unreviewed", "pending"],
)
def test_null_pair_without_explicit_reviewed_status_is_rejected(
    tmp_path: Path, review_status: str | None
) -> None:
    fixture = _fixture(tmp_path)
    if review_status is None:
        del fixture["null_pairs"][0]["review_status"]
    else:
        fixture["null_pairs"][0]["review_status"] = review_status
    with pytest.raises(RegistryError, match="review_status"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


@pytest.mark.parametrize(
    "null_pairs_selector,expected_reason_snippet",
    [
        (lambda pairs: pairs[:2], "at least 3 required"),
        (
            lambda pairs: [
                {**pairs[0], "pair_id": "dup-a"},
                {**pairs[0], "pair_id": "dup-b", "is_mechanical": False},
                {**pairs[0], "pair_id": "dup-c", "is_mechanical": False},
            ],
            "at least 2 required",
        ),
    ],
)
def test_undersized_or_one_image_null_envelope_is_marked_insufficient(
    tmp_path: Path, null_pairs_selector, expected_reason_snippet: str
) -> None:
    fixture = _fixture(tmp_path)
    fixture["null_pairs"] = null_pairs_selector(fixture["null_pairs"])
    document = _build_with_fixture(fixture, tmp_path / "registry.json")
    envelope = document["smoke"]["null_pair_envelope"]
    assert envelope["status"] == "insufficient"
    assert any(expected_reason_snippet in reason for reason in envelope["reasons"])


# ---------------------------------------------------------------------------
# Sealed-witness null contract: single trajectory, G at the cut, F novel in
# P, T released strictly later -- reject an out-of-order witness rather than
# accepting an arbitrary baseline cut or a null lacking a later T release.
# ---------------------------------------------------------------------------


def _sealed_witness_order_fixture(tmp_path: Path, *, seed: int, row_owners: list[str]) -> dict[str, Any]:
    """A dedicated 7511 trajectory whose rows strict-match ``row_owners`` in
    order, for isolating cut/novelty/release ordering edge cases."""

    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = []
    _append_rollout(
        fixture,
        _rollout_entry(
            image_id="7511",
            decode_mode="sampled",
            seed=seed,
            row_count=len(row_owners),
            row_bboxes={
                index: (_COVERING_ROW_BBOX_7511 if owner == "gt:7511:700" else _FOIL_ROW_BBOX_7511_24)
                for index, owner in enumerate(row_owners)
            },
        ),
    )
    _append_prediction_rows(
        fixture,
        [
            _pred_row(
                f"pred:sorted:sampled:{seed}:7511:{index}",
                decode_mode="sampled",
                seed=seed,
                strict_match_gt_owner_id=owner,
            )
            for index, owner in enumerate(row_owners)
        ],
    )
    return fixture


def test_null_pair_target_release_not_later_than_cut_is_rejected(tmp_path: Path) -> None:
    # row0 = T (already released *before* G), row1 = G, row2 = F.
    seed = 21020
    fixture = _sealed_witness_order_fixture(
        tmp_path, seed=seed, row_owners=["gt:7511:26", "gt:7511:700", "gt:7511:24"]
    )
    fixture["null_pairs"] = [
        {
            "pair_id": "null-order-violation",
            "image_id": "7511",
            "is_mechanical": True,
            "review_status": "reviewed",
            "target_gt_owner_id": "gt:7511:26",
            "baseline": {"decode_mode": "sampled", "seed": seed},
            "covering": {"pred_row_id": f"pred:sorted:sampled:{seed}:7511:1", "gt_owner_id": "gt:7511:700"},
            "foil": {"pred_row_id": f"pred:sorted:sampled:{seed}:7511:2", "gt_owner_id": "gt:7511:24"},
            "target_release_pred_row_id": f"pred:sorted:sampled:{seed}:7511:0",
        }
    ]
    with pytest.raises(RegistryError, match="a null lacking a later T release is rejected"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_foil_not_novel_in_p_is_rejected(tmp_path: Path) -> None:
    # row0 = F (already present *before* G, so it is not novel in P),
    # row1 = G, row2 = T.
    seed = 21021
    fixture = _sealed_witness_order_fixture(
        tmp_path, seed=seed, row_owners=["gt:7511:24", "gt:7511:700", "gt:7511:26"]
    )
    fixture["null_pairs"] = [
        {
            "pair_id": "null-foil-not-novel",
            "image_id": "7511",
            "is_mechanical": True,
            "review_status": "reviewed",
            "target_gt_owner_id": "gt:7511:26",
            "baseline": {"decode_mode": "sampled", "seed": seed},
            "covering": {"pred_row_id": f"pred:sorted:sampled:{seed}:7511:1", "gt_owner_id": "gt:7511:700"},
            "foil": {"pred_row_id": f"pred:sorted:sampled:{seed}:7511:0", "gt_owner_id": "gt:7511:24"},
            "target_release_pred_row_id": f"pred:sorted:sampled:{seed}:7511:2",
        }
    ]
    with pytest.raises(RegistryError, match="is not novel in P"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_missing_target_release_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    del fixture["null_pairs"][0]["target_release_pred_row_id"]
    with pytest.raises(RegistryError, match="target_release_pred_row_id"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_null_pair_ambiguous_covering_owner_is_rejected(tmp_path: Path) -> None:
    # Row 2 (originally T's release) is retargeted to also strict-match the
    # covering owner: G is no longer the *unique* strict match in this
    # trajectory, so the witness binding is ambiguous and must be rejected.
    fixture = _fixture(tmp_path)
    fixture["collision_pairs"] = []
    _mutate_prediction_rows(
        fixture,
        lambda row: row.update({"strict_match_gt_owner_id": "gt:7511:700"})
        if row["pred_row_id"] == "pred:sorted:sampled:21001:7511:2"
        else None,
    )
    with pytest.raises(RegistryError, match="ambiguous binding is rejected"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


# ---------------------------------------------------------------------------
# P0-4: first-skip scan-order and contamination validation
# ---------------------------------------------------------------------------


def test_first_skip_roles_present_and_scan_order_holds(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    roles = {role["role_id"]: role for role in document["smoke"]["roles"]}
    assert roles["first_skip:P_pre"]["gt_owner_id"] == "gt:7511:1"
    assert roles["first_skip:P_post"]["successor_gt_owner_id"] == "gt:7511:2"


def test_first_skip_rejects_when_missed_owner_does_not_precede_successor(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    # Move gt:7511:1 after gt:7511:2 in the frozen scan order.
    _mutate_owner_rows(
        fixture,
        lambda row: row.update({"bbox_xyxy": [900, 900, 910, 910]})
        if row["gt_owner_id"] == "gt:7511:1"
        else None,
    )
    with pytest.raises(RegistryError, match="does not precede"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_first_skip_rejects_contaminated_duplicate_prefix_row(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    # gt:7511:2 (the successor) sits at row index 2; rows 0 and 1 are both
    # unmatched filler and precede it. Retarget *both* to strict-match one
    # extra owner with a scan key before the successor's -- a genuine
    # duplicate strictly before the successor, independent of successor
    # identification itself.
    _append_owner_row(
        fixture,
        {
            "gt_owner_id": "gt:7511:500",
            "image_id": "7511",
            "normalized_description": "person",
            "bbox_xyxy": [5, 5, 15, 15],
            "original_annotation_index": 500,
        },
    )
    _mutate_prediction_rows(
        fixture,
        lambda row: row.update({"strict_match_gt_owner_id": "gt:7511:500"})
        if row["pred_row_id"] in {"pred:sorted:greedy:0:7511:0", "pred:sorted:greedy:0:7511:1"}
        else None,
    )
    with pytest.raises(RegistryError, match="contaminated by a duplicate"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_first_skip_rejects_out_of_order_preceding_row(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    # Row 0 (currently unmatched) is retargeted to strict-match an owner
    # whose frozen scan key is *after* the successor's -- an out-of-order
    # preceding row makes the transition prefix unprovably clean.
    _mutate_prediction_rows(
        fixture,
        lambda row: row.update({"strict_match_gt_owner_id": "gt:7511:29"})
        if row["pred_row_id"] == "pred:sorted:greedy:0:7511:0"
        else None,
    )
    with pytest.raises(RegistryError, match="not provably clean"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


# ---------------------------------------------------------------------------
# Preserved behavior from the prior revision
# ---------------------------------------------------------------------------


def test_root_and_due_turn_roles_stay_distinct_under_execution_dedup(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    roles = document["smoke"]["roles"]
    root_roles = {role["gt_owner_id"]: role for role in roles if role["role_kind"] == "root_context"}
    assert {"gt:7511:22", "gt:7511:26", "gt:7511:17"} <= set(root_roles)

    role_22 = root_roles["gt:7511:22"]
    role_26 = root_roles["gt:7511:26"]
    assert role_22["prefix"]["token_ids_sha256"] == role_26["prefix"]["token_ids_sha256"]
    assert role_22["role_id"] != role_26["role_id"]
    assert role_22["gt_owner_id"] != role_26["gt_owner_id"]

    role_ids = [role["role_id"] for role in roles]
    assert len(role_ids) == len(set(role_ids))


def test_cohort_provenance_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    lines = fixture["cohort_assignments"].read_text(encoding="utf-8").splitlines()
    mutated_lines = []
    for line in lines:
        row = json.loads(line)
        if row["gt_owner_id"] == "gt:7511:22":
            row["cohort"] = "loose_only_b1"
        mutated_lines.append(json.dumps(row))
    fixture["cohort_assignments"].write_text("\n".join(mutated_lines) + "\n", encoding="utf-8")

    with pytest.raises(RegistryError, match="cohort is"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_strict_rescue_due_turn_context_requires_explicit_trajectory(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["due_turn_trajectories"] = {}
    with pytest.raises(RegistryError, match="due-turn"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_forced_continuation_rollout_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    greedy_path = tmp_path / "greedy.json"
    document = json.loads(greedy_path.read_text(encoding="utf-8"))
    document["rollouts"][0]["stop_reason"] = "forced_continue"
    greedy_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RegistryError, match="[Ff]orced-continuation"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_row_chunk_and_prediction_count_mismatch_is_rejected(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    greedy_path = tmp_path / "greedy.json"
    document = json.loads(greedy_path.read_text(encoding="utf-8"))
    document["rollouts"][0]["predictions"]["predictions"].pop()
    greedy_path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(RegistryError, match="row boundaries are ambiguous"):
        _build_with_fixture(fixture, tmp_path / "registry.json")


def test_foil_contrast_allowed_reflects_sampled_or_oracle_only_rule(tmp_path: Path) -> None:
    document = _build(tmp_path, tmp_path / "registry.json")
    targets_by_id = {row["gt_owner_id"]: row for row in document["mechanism_cohort"]["targets"]}
    assert targets_by_id["gt:7511:22"]["foil_contrast_allowed"] == "natural_greedy_allowed"
    context_by_id = {
        row["gt_owner_id"]: row for row in document["context_control_registry"]["bound_non_targets"]
    }
    assert context_by_id["gt:7511:24"]["foil_contrast_allowed"] == "sampled_or_oracle_only"


def test_output_is_idempotent_and_deterministic(tmp_path: Path) -> None:
    output = tmp_path / "registry.json"
    first = _build(tmp_path, output)
    second = _build(tmp_path, output)
    assert first["registry_digest"] == second["registry_digest"]


def test_cli_writes_registry(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    output = tmp_path / "registry.json"
    args = [
        sys.executable,
        "-m",
        "scripts.research.build_sorted_fn_mechanism_registry",
        "--owner-ledger",
        str(fixture["owner_ledger"]),
        "--prediction-row-ledger",
        str(fixture["prediction_row_ledger"]),
        "--cohort-assignments",
        str(fixture["cohort_assignments"]),
        "--due-turn-trajectory",
        "gt:7511:17=sampled:21010",
        "--output",
        str(output),
    ]
    for rollout in fixture["rollouts"]:
        args.extend(["--rollout", str(rollout)])
    null_pairs_plan = tmp_path / "null-pairs.json"
    null_pairs_plan.write_text(json.dumps(fixture["null_pairs"]), encoding="utf-8")
    args.extend(["--null-pairs-plan", str(null_pairs_plan)])

    result = subprocess.run(
        args, cwd="/data/CoordExp/.worktrees/research-probes", capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    summary = json.loads(result.stdout)
    assert summary["target_count"] == 24
    assert output.is_file()
