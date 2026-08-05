"""Contract tests for the neutral-row control merge.

The fixture writes a real sealed plan directory, captures every image through the
deterministic ``FakeCensusBackend``, and merges the published shards, so
completeness, duplicate, cross-owner, scored-token, replay, parity and quarantine
validation are all exercised against genuinely produced evidence.

The sealed gate references are taken from a first capture pass, exactly as the
real plan takes them from the sealed predecessor capture; a second pass then has
to replay them within the frozen tolerances.
"""

from __future__ import annotations

from collections.abc import Callable
import copy
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import merge_sorted_crossing_neutral_row_control as sut
from scripts.research import prepare_sorted_crossing_neutral_row_control as plan_builder
from scripts.research import score_sorted_crossing_boundary_owner_release as crossing_scorer
from scripts.research import score_sorted_crossing_neutral_row_control as scorer
from scripts.research import score_sorted_owner_accessibility_census_shard as shard

OBJ_START = crossing_scorer.OBJECT_REF_START
OBJ_END = crossing_scorer.OBJECT_REF_END
BOX_START = crossing_scorer.BOX_START
BOX_END = crossing_scorer.BOX_END
COORD_START = crossing_scorer.COORDINATE_TOKEN_ID_START

EXECUTED_PER_IMAGE: dict[str, int] = {
    "10707": 1,
    "13348": 1,
    "13923": 2,
    "14038": 3,
    "14439": 1,
    "1584": 1,
    "16228": 4,
    "2685": 1,
    "4134": 5,
    "5001": 1,
    "6040": 1,
    "7511": 0,
}
IMAGE_IDS: tuple[str, ...] = tuple(EXECUTED_PER_IMAGE)
#: Nine voting owners, four of them clearly separated, matching the frozen shape.
VOTING_SLOTS: tuple[tuple[str, int], ...] = (
    ("13348", 0),
    ("13923", 0),
    ("14038", 0),
    ("1584", 0),
    ("16228", 0),
    ("16228", 1),
    ("4134", 0),
    ("4134", 1),
    ("6040", 0),
)
SEPARATED_SLOTS: frozenset[tuple[str, int]] = frozenset(
    {("13348", 0), ("16228", 0), ("4134", 0), ("4134", 1)}
)
#: The one image whose executed owners cover all four sealed strata, exactly as
#: the real plan's image ``4134`` does, and therefore the only image the
#: representative smoke may open a session for.
SMOKE_IMAGE_ID = "4134"


def _sha256_json(value: Any) -> str:
    return crossing_scorer.sha256_json(value)


def _strata(image_id: str, slot: int) -> tuple[bool, bool]:
    """``(E strict-matched, E realized C's description)`` for one executed owner.

    Only the smoke image carries all four strata; every other image stays
    unmatched-``E``/different-description, exactly as the single-owner images of
    the real sealed plan do.
    """

    if image_id != SMOKE_IMAGE_ID:
        return False, False
    return slot % 2 == 0, slot in (1, 2)


def _coords(index: int) -> list[int]:
    base = (index * 7) % 900
    return [COORD_START + base + offset for offset in (0, 1, 4, 5)]


def _row(seed: int) -> list[int]:
    description = [20000 + (seed * 13) % 500] * (1 + seed % 2)
    return [OBJ_START, *description, OBJ_END, BOX_START, *_coords(seed), BOX_END]


class _StubCensus:
    def __init__(self, contexts: dict[str, Any]) -> None:
        self.contexts_by_id = contexts


class _StubInputs:
    def __init__(self, contexts: dict[str, Any], images: dict[str, Any]) -> None:
        self.census = _StubCensus(contexts)
        self.images_by_id = images


def _make_request(
    *,
    arm: str,
    cohort_role: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    context_role: str,
    boundary_index: int,
    base_prefix_token_ids: list[int],
    appended_token_ids: list[int],
    scored_target: dict[str, Any],
    baseline_context_id: str,
    sealed_reference: dict[str, Any],
) -> dict[str, Any]:
    base_digest = _sha256_json(base_prefix_token_ids)
    family = plan_builder.REQUEST_FAMILY_BY_ARM[arm]
    identity = {
        "unit_id": scorer.UNIT_ID,
        "request_family": family,
        "arm": arm,
        "cohort_role": cohort_role,
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "baseline_context_id": baseline_context_id,
        "appended_token_ids": list(appended_token_ids),
        "base_prefix_token_ids_sha256": base_digest,
        "scored_target": dict(scored_target),
    }
    digest = _sha256_json(identity)
    return {
        "schema_version": plan_builder.REQUEST_SCHEMA_VERSION,
        "row_kind": "neutral_row_control_request",
        "unit_id": scorer.UNIT_ID,
        "request_id": f"req:{digest[:32]}",
        "request_key": f"{family}|{cohort_role}|{gt_owner_id}|{context_id}|{arm}",
        "request_family": family,
        "arm": arm,
        "cohort_role": cohort_role,
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "context_id": context_id,
        "context_role": context_role,
        "boundary_index": boundary_index,
        "paired_roots": {
            "baseline_context_id": baseline_context_id,
            "modified_context_id": context_id,
            "orientation": "modified_minus_baseline",
        },
        "prefix": {
            "base_context_id": context_id,
            "base_prefix_token_count": len(base_prefix_token_ids),
            "base_prefix_token_ids_sha256": base_digest,
            "appended_token_ids": list(appended_token_ids),
            "appended_token_ids_sha256": _sha256_json(list(appended_token_ids)),
            "appended_token_count": len(appended_token_ids),
            "appended_role": plan_builder.APPENDED_ROLE_BY_ARM[arm],
            "retokenized": False,
        },
        "scored_target": dict(scored_target),
        "sealed_reference": dict(sealed_reference),
        "score_blind_plan": True,
        "inspects_new_model_logits": False,
        "identity_digest": digest,
    }


def _placeholder_reference() -> dict[str, Any]:
    return {
        "coordinate_delta": 0.0,
        "coordinate_delta_sign": 0,
        "baseline_coordinate_sum": 0.0,
        "modified_coordinate_sum": 0.0,
        "baseline_argmax_token_ids_sha256": "0" * 64,
        "modified_argmax_token_ids_sha256": "0" * 64,
        "baseline_argmax_reproduces_description_path": True,
        "baseline_argmax_reproduces_complete_row": True,
        "scored_token_ids_sha256": "0" * 64,
        "scored_token_count": 0,
        "request_id": "req:placeholder",
        "role": "gate_reference_only_never_the_estimand",
        "materiality_cutoff_nats": -1.0,
        "replay_max_coordinate_delta_abs_diff": 0.05,
        "replay_max_selected_logit_abs_diff": 0.001,
    }


def build_plan_rows(
    references: dict[str, dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """The whole synthetic sealed plan, plus the stub context/image surfaces."""

    references = references or {}
    contexts: dict[str, Any] = {}
    images: dict[str, Any] = {}
    selection_rows: list[dict[str, Any]] = []
    benign_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    for image_index, image_id in enumerate(IMAGE_IDS):
        images[image_id] = {"prompt_token_ids": [900 + image_index, 901 + image_index]}
        owner_count = EXECUTED_PER_IMAGE[image_id]
        row_count = 4 + 2 * max(owner_count, 1)
        rows = [_row(image_index * 31 + row_index) for row_index in range(row_count)]
        prefixes: list[list[int]] = [[]]
        for row in rows:
            prefixes.append(prefixes[-1] + row)
        for boundary_index, prefix in enumerate(prefixes):
            context_id = f"{image_id}:boundary-{boundary_index:03d}"
            contexts[context_id] = {
                "context_id": context_id,
                "image_id": image_id,
                "boundary_index": boundary_index,
                "context_role": "root" if boundary_index == 0 else "row_boundary",
                "generated_prefix_token_ids": list(prefix),
                "generated_prefix_token_ids_sha256": _sha256_json(list(prefix)),
            }

        def _context(index: int, _image_id: str = image_id) -> str:
            return f"{_image_id}:boundary-{index:03d}"

        for slot in range(owner_count):
            boundary = 3 + 2 * slot
            gt_owner_id = f"gt:{image_id}:{boundary}"
            votes = (image_id, slot) in VOTING_SLOTS
            cohort_role = (
                plan_builder.COHORT_VOTING if votes else plan_builder.COHORT_SPECIFICITY
            )
            e_tokens = rows[boundary]
            clean = _row(5000 + image_index * 7 + slot)
            neutral = _row(9000 + image_index * 7 + slot)
            reference = references.get(gt_owner_id, _placeholder_reference())
            e_matched, e_same_description = _strata(image_id, slot)
            c_description = f"targ-{boundary}"
            selection_rows.append(
                {
                    "schema_version": plan_builder.SELECTION_SCHEMA_VERSION,
                    "row_kind": "neutral_row_selection",
                    "unit_id": scorer.UNIT_ID,
                    "gt_owner_id": gt_owner_id,
                    "image_id": image_id,
                    "normalized_description": c_description,
                    "cohort_role": cohort_role,
                    "votes": votes,
                    "executed": True,
                    "material_negative": votes,
                    "clearly_separated": (image_id, slot) in SEPARATED_SLOTS,
                    "sentinel_owner": False,
                    "posthoc_support_extent_uncertain": False,
                    "crossing": {
                        "boundary_index_b": boundary,
                        "p_context_id": _context(boundary),
                    },
                    "inserted_clean_row_c": {
                        "token_ids": clean,
                        "token_ids_sha256": _sha256_json(clean),
                        "token_count": len(clean),
                    },
                    "scored_e_row": {
                        "row_index": boundary,
                        "token_ids": e_tokens,
                        "token_ids_sha256": _sha256_json(e_tokens),
                        "strict_match_status": "matched" if e_matched else "unmatched",
                        "stratum": "matched_e" if e_matched else "unmatched_e",
                        "normalized_description": (
                            c_description if e_same_description else "erow"
                        ),
                        "pre_row_context_id": _context(boundary),
                        "post_row_context_id": _context(boundary + 1),
                    },
                    "neutral_row_n": {
                        "gt_owner_id": f"gt:{image_id}:n{slot}",
                        "token_ids": neutral,
                        "token_ids_sha256": _sha256_json(neutral),
                        "token_count": len(neutral),
                        "row_length_delta_tokens": 0,
                        "rows_back_distance": 2,
                        "min_center_distance_normalized": 0.2,
                    },
                    "sealed_clean_reference": reference,
                    "request_ids": [],
                }
            )
            scored_target = {
                "kind": "exact_native_row",
                "token_ids": list(e_tokens),
                "token_ids_sha256": _sha256_json(e_tokens),
                "native_row_index": boundary,
                "baseline_context_id": _context(boundary),
                "successor_context_id": _context(boundary + 1),
                "compare_against": "the same exact E row scored at the unmodified native P",
                "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
                "primary_segment": "coordinates",
            }
            for arm, appended in (
                (scorer.ARM_NEUTRAL, neutral),
                (scorer.ARM_CLEAN_REPLAY, clean),
            ):
                requests.append(
                    _make_request(
                        arm=arm,
                        cohort_role=cohort_role,
                        gt_owner_id=gt_owner_id,
                        image_id=image_id,
                        context_id=_context(boundary),
                        context_role="row_boundary",
                        boundary_index=boundary,
                        base_prefix_token_ids=prefixes[boundary],
                        appended_token_ids=appended,
                        scored_target=scored_target,
                        baseline_context_id=_context(boundary),
                        sealed_reference=reference,
                    )
                )

        control_owner = f"gt:{image_id}:tp"
        twin = _row(7000 + image_index)
        following = rows[1]
        reference = references.get(control_owner, _placeholder_reference())
        benign_rows.append(
            {
                "schema_version": plan_builder.BENIGN_SCHEMA_VERSION,
                "row_kind": "benign_reference_control",
                "unit_id": scorer.UNIT_ID,
                "cohort_role": plan_builder.COHORT_BENIGN,
                "gt_owner_id": control_owner,
                "image_id": image_id,
                "normalized_description": "tp",
                "due_context_id": _context(0),
                "replaced_native_row_index": 0,
                "following_native_action": {
                    "context_id": _context(1),
                    "native_row_index": 1,
                    "token_ids": following,
                    "token_ids_sha256": _sha256_json(following),
                    "token_count": len(following),
                },
                "inserted_clean_row_c": {
                    "token_ids": twin,
                    "token_ids_sha256": _sha256_json(twin),
                    "token_count": len(twin),
                },
                "sealed_benign_reference": reference,
                "request_ids": [],
            }
        )
        requests.append(
            _make_request(
                arm=scorer.ARM_BENIGN,
                cohort_role=plan_builder.COHORT_BENIGN,
                gt_owner_id=control_owner,
                image_id=image_id,
                context_id=_context(0),
                context_role="root",
                boundary_index=0,
                base_prefix_token_ids=prefixes[0],
                appended_token_ids=twin,
                scored_target={
                    "kind": "exact_native_row",
                    "token_ids": list(following),
                    "token_ids_sha256": _sha256_json(following),
                    "native_row_index": 1,
                    "baseline_context_id": _context(1),
                    "successor_context_id": _context(2),
                    "compare_against": "the unmodified native successor context",
                    "report": ["description_delta", "coordinate_delta", "complete_row_delta"],
                    "primary_segment": "coordinates",
                },
                baseline_context_id=_context(1),
                sealed_reference=reference,
            )
        )

    manifest = {
        "schema_version": plan_builder.MANIFEST_SCHEMA_VERSION,
        "unit_id": scorer.UNIT_ID,
        "builder_source": {"path": "builder.py", "sha256": "b" * 64},
        "cohort": {
            "voting_owner_count": 9,
            "specificity_owner_count": 12,
            "infeasible_owner_count": 5,
            "executed_owner_count": 21,
            "benign_control_count": 12,
            "image_count": 12,
            "image_ids": list(IMAGE_IDS),
        },
        "gates": {"specificity_material_max": 4},
        "routes": {"order": list(plan_builder.ROUTE_ORDER)},
        "materiality": {"cutoff_nats": -1.0},
        "non_voting_sensitivities": {"materiality_cutoffs_nats": [-0.75, -1.25]},
        "lineage": {"crossing_plan_dir": "/nonexistent/crossing"},
    }
    return manifest, selection_rows, benign_rows, requests, contexts, images


def write_plan_dir(
    plan_dir: Path,
    manifest: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Publish the plan rows and a self-sealed manifest that describes them."""

    plan_dir.mkdir(parents=True, exist_ok=True)
    row_sets = {
        plan_builder.SELECTION_REGISTRY_NAME: selection_rows,
        plan_builder.BENIGN_REGISTRY_NAME: benign_rows,
        plan_builder.REQUEST_PLAN_NAME: requests,
    }
    files = {
        name: b"".join(crossing_scorer.canonical_json_bytes(row) + b"\n" for row in rows)
        for name, rows in row_sets.items()
    }
    for name, payload in files.items():
        (plan_dir / name).write_bytes(payload)
    sealed = dict(manifest)
    sealed["output_file_digests"] = {
        name: {
            "path": name,
            "byte_size": len(files[name]),
            "sha256": crossing_scorer.sha256_bytes(files[name]),
            "row_count": len(row_sets[name]),
        }
        for name in sorted(files)
    }
    sealed["manifest_content_sha256"] = _sha256_json(sealed)
    (plan_dir / plan_builder.MANIFEST_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(sealed) + b"\n"
    )
    return sealed


def sealed_plan_object(
    plan_dir: Path,
    manifest: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    contexts: dict[str, Any],
    images: dict[str, Any],
) -> scorer.SealedNeutralPlan:
    crossing = crossing_scorer.SealedPlan(
        plan_dir=Path("/nonexistent/crossing"),
        manifest={"manifest_content_sha256": "crossing-digest"},
        cohort_rows=[],
        control_rows=[],
        request_rows=[],
        plan_file_sha256={},
        inputs=_StubInputs(contexts, images),
        candidates_by_id={},
        owners_by_image={},
    )
    return scorer.SealedNeutralPlan(
        plan_dir=plan_dir,
        manifest=manifest,
        selection_rows=selection_rows,
        benign_rows=benign_rows,
        request_rows=requests,
        plan_file_sha256={
            name: crossing_scorer.sha256_file(plan_dir / name)
            for name in (
                plan_builder.SELECTION_REGISTRY_NAME,
                plan_builder.BENIGN_REGISTRY_NAME,
                plan_builder.REQUEST_PLAN_NAME,
            )
        },
        crossing=crossing,
    )


def _runtime_identity(image_id: str) -> dict[str, Any]:
    return {
        "model_identity": {"name": "fake"},
        "tokenizer_identity": {"name": "fake"},
        "source_identity": scorer.source_identity(),
        "session_image_id": image_id,
    }


def promote_to_evidence(shard_dir: Path) -> None:
    """Make a shard look like an evidence-bearing capture with a clean replay.

    The deterministic ``FakeCensusBackend`` declares itself non-evidence-bearing
    and its argmax is a hash of the token sequence, so a capture over it never
    enforces native replay and the merge rightly refuses it -- which is itself
    tested.  For the positive path the fixture rewrites only the *baseline*
    root's argmax stream to the forced tokens, leaving every selected log
    probability untouched, so the shard stays internally consistent and every
    merge-time reconstruction still has to hold.
    """

    rows = [
        json.loads(line) for line in (shard_dir / scorer.ROWS_NAME).read_text().splitlines()
    ]
    for row in rows:
        baseline = row["roots"][scorer.ROOT_BASELINE]
        baseline["argmax_token_ids"] = list(row["scored_token_ids"])
        baseline["argmax_logprobs"] = list(baseline["selected_logprobs"])
        baseline["selected_is_argmax"] = [True] * len(row["scored_token_ids"])
        baseline["argmax_reproduces_description_path"] = True
        baseline["argmax_reproduces_complete_row"] = True
        row["baseline_replay_admitted"] = True
        row["replay_admission_enforced"] = True
    receipt = json.loads((shard_dir / scorer.RECEIPT_NAME).read_text())
    receipt["policy"]["replay_admission_enforced"] = True
    (shard_dir / scorer.RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )
    _reseal_shard(shard_dir, rows)


def capture_all_images(
    plan: scorer.SealedNeutralPlan, shard_root: Path
) -> list[Path]:
    """One four-strata smoke plus one capture shard per covered image, on disk.

    The smoke opens a session for the one image that covers every sealed
    stratum, and every capture shard -- including the eleven images that could
    never carry the smoke themselves -- inherits that single admission.
    """

    backend = shard.FakeCensusBackend(seed="neutral-row-merge")
    smoke = scorer.run_smoke_shard(
        plan=plan,
        backend=backend,
        shard_id="smoke",
        image_id=SMOKE_IMAGE_ID,
        runtime_identity=_runtime_identity(SMOKE_IMAGE_ID),
    )
    assert smoke.admission is not None
    assert sorted(smoke.admission["smoke_strata"]["representatives"]) == sorted(
        scorer.REQUIRED_SMOKE_STRATA
    )
    shard_dirs: list[Path] = []
    for image_id in IMAGE_IDS:
        result = scorer.run_capture_shard(
            plan=plan,
            backend=backend,
            shard_id=f"shard-{image_id}",
            session_image_id=image_id,
            admission=smoke.admission,
            runtime_identity=_runtime_identity(image_id),
        )
        assert result.quarantine is None
        output_dir = shard_root / f"shard-{image_id}"
        crossing_scorer._publish(output_dir, scorer.shard_output_files(result))  # noqa: SLF001
        promote_to_evidence(output_dir)
        shard_dirs.append(output_dir)
    return shard_dirs


def raw_capture_shard(plan: scorer.SealedNeutralPlan, output_dir: Path) -> Path:
    """One capture shard published exactly as the fake backend produced it."""

    backend = shard.FakeCensusBackend(seed="neutral-row-merge")
    image_id = IMAGE_IDS[0]
    smoke = scorer.run_smoke_shard(
        plan=plan,
        backend=backend,
        shard_id="smoke",
        image_id=SMOKE_IMAGE_ID,
        runtime_identity=_runtime_identity(SMOKE_IMAGE_ID),
    )
    assert smoke.admission is not None
    result = scorer.run_capture_shard(
        plan=plan,
        backend=backend,
        shard_id=f"raw-{image_id}",
        session_image_id=image_id,
        admission=smoke.admission,
        runtime_identity=_runtime_identity(image_id),
    )
    crossing_scorer._publish(output_dir, scorer.shard_output_files(result))  # noqa: SLF001
    return output_dir


def references_from(shard_dirs: list[Path]) -> dict[str, dict[str, Any]]:
    """Turn a first capture pass into the sealed gate references of the plan."""

    references: dict[str, dict[str, Any]] = {}
    for shard_dir in shard_dirs:
        for line in (shard_dir / scorer.ROWS_NAME).read_text().splitlines():
            row = json.loads(line)
            if row["arm"] == scorer.ARM_NEUTRAL:
                continue
            baseline = row["roots"][scorer.ROOT_BASELINE]
            modified = row["roots"][scorer.ROOT_MODIFIED]
            delta = row["deltas"]["coordinates"]["delta"]
            references[row["gt_owner_id"]] = {
                "coordinate_delta": delta,
                "coordinate_delta_sign": row["deltas"]["coordinates"]["sign"],
                "baseline_coordinate_sum": baseline["segment_sums"]["coordinates"]["sum"],
                "modified_coordinate_sum": modified["segment_sums"]["coordinates"]["sum"],
                "baseline_argmax_token_ids_sha256": _sha256_json(
                    baseline["argmax_token_ids"]
                ),
                "modified_argmax_token_ids_sha256": _sha256_json(
                    modified["argmax_token_ids"]
                ),
                "baseline_argmax_reproduces_description_path": baseline[
                    "argmax_reproduces_description_path"
                ],
                "baseline_argmax_reproduces_complete_row": baseline[
                    "argmax_reproduces_complete_row"
                ],
                "scored_token_ids_sha256": row["scored_token_ids_sha256"],
                "scored_token_count": row["scored_token_count"],
                "request_id": row["request_id"],
                "role": "gate_reference_only_never_the_estimand",
                "materiality_cutoff_nats": -1.0,
                "replay_max_coordinate_delta_abs_diff": 0.05,
                "replay_max_selected_logit_abs_diff": 0.001,
            }
    return references


@pytest.fixture(scope="module")
def captured(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """A sealed plan whose references came from a first pass, plus its shards."""

    root = tmp_path_factory.mktemp("neutral-row-merge")
    probe_dir = root / "probe-plan"
    manifest, selection, benign, requests, contexts, images = build_plan_rows()
    sealed = write_plan_dir(probe_dir, manifest, selection, benign, requests)
    probe_plan = sealed_plan_object(
        probe_dir, sealed, selection, benign, requests, contexts, images
    )
    probe_shards = capture_all_images(probe_plan, root / "probe-shards")
    references = references_from(probe_shards)

    plan_dir = root / "plan"
    manifest, selection, benign, requests, contexts, images = build_plan_rows(references)
    sealed = write_plan_dir(plan_dir, manifest, selection, benign, requests)
    plan = sealed_plan_object(
        plan_dir, sealed, selection, benign, requests, contexts, images
    )
    shard_dirs = capture_all_images(plan, root / "shards")
    return {
        "root": root,
        "plan_dir": plan_dir,
        "plan": plan,
        "shard_dirs": shard_dirs,
        "references": references,
    }


@pytest.fixture()
def merged(captured: dict[str, Any], tmp_path: Path) -> dict[str, Any]:
    receipt = sut.merge_shards(
        plan_dir=captured["plan_dir"],
        shard_dirs=captured["shard_dirs"],
        output_dir=tmp_path / "merged",
    )
    return {"receipt": receipt, "merged_dir": tmp_path / "merged"}


def _copy_shard(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        (destination / child.name).write_bytes(child.read_bytes())
    return destination


def _rewrite_inherited_admission(
    shard_dir: Path, mutate: Callable[[dict[str, Any]], None]
) -> None:
    """Edit one shard's inherited admission and reseal its parity honestly.

    The receipt seals the parity file's digest, so a mutation that did not
    reseal would be caught by the digest check instead of by the gate under
    test.  Resealing leaves the inherited-admission contract as the only thing
    that can fail.
    """

    parity = json.loads((shard_dir / scorer.PARITY_NAME).read_text())
    mutate(parity["inherited_admission"])
    payload = crossing_scorer.canonical_json_bytes(parity) + b"\n"
    (shard_dir / scorer.PARITY_NAME).write_bytes(payload)
    receipt = json.loads((shard_dir / scorer.RECEIPT_NAME).read_text())
    receipt.pop("receipt_content_sha256")
    receipt["output_file_digests"][scorer.PARITY_NAME] = {
        "path": scorer.PARITY_NAME,
        "byte_size": len(payload),
        "sha256": crossing_scorer.sha256_bytes(payload),
    }
    receipt["receipt_content_sha256"] = _sha256_json(receipt)
    (shard_dir / scorer.RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )


def _reseal_shard(shard_dir: Path, rows: list[dict[str, Any]]) -> None:
    """Rewrite a shard's rows and its receipt so only the target check can fail."""

    payload = b"".join(
        crossing_scorer.canonical_json_bytes(row) + b"\n" for row in rows
    )
    (shard_dir / scorer.ROWS_NAME).write_bytes(payload)
    receipt = json.loads((shard_dir / scorer.RECEIPT_NAME).read_text())
    receipt.pop("receipt_content_sha256")
    receipt["output_file_digests"][scorer.ROWS_NAME] = {
        "path": scorer.ROWS_NAME,
        "byte_size": len(payload),
        "sha256": crossing_scorer.sha256_bytes(payload),
    }
    receipt["executed"]["row_count"] = len(rows)
    receipt["receipt_content_sha256"] = _sha256_json(receipt)
    (shard_dir / scorer.RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )


# ---------------------------------------------------------------------------
# 1. Frozen identities
# ---------------------------------------------------------------------------


def test_merge_artifact_names_and_gate_ownership_are_frozen() -> None:
    assert sut.MERGED_OUTPUT_NAMES == (
        "neutral-row-control-rows.jsonl",
        "neutral-row-control-parity.jsonl",
        "neutral-row-control-merge-receipt.json",
    )
    assert sut.MERGE_OWNED_GATES == (
        "input_and_token_identity",
        "runtime_replay",
        "cached_versus_uncached_parity",
    )
    assert sut.ANALYSIS_OWNED_GATES == (
        "same_run_positive_controls",
        "benign_reference_replay",
        "neutral_row_specificity",
    )
    assert sut.MAX_QUARANTINED_OWNERS == 2
    assert sut.REPLAY_MAX_SELECTED_LOGIT_ABS_DIFF == 1e-3


def test_the_frozen_plan_reproduces_its_own_denominators(captured: dict[str, Any]) -> None:
    plan = sut.load_frozen_plan(captured["plan_dir"])
    assert len(plan.request_rows) == 54
    assert len(plan.executed_owner_ids) == 21
    assert len(plan.benign_by_owner) == 12
    assert plan.manifest_content_sha256 == captured["plan"].manifest_content_sha256


def test_an_edited_plan_row_file_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    plan_dir = _copy_shard(captured["plan_dir"], tmp_path / "plan")
    rows = [
        json.loads(line)
        for line in (plan_dir / plan_builder.REQUEST_PLAN_NAME).read_text().splitlines()
    ]
    rows[0]["cohort_role"] = plan_builder.COHORT_BENIGN
    (plan_dir / plan_builder.REQUEST_PLAN_NAME).write_bytes(
        b"".join(crossing_scorer.canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(sut.NeutralRowMergeContractError, match="hashes to"):
        sut.load_frozen_plan(plan_dir)


# ---------------------------------------------------------------------------
# 2. Completeness and closure
# ---------------------------------------------------------------------------


def test_merge_folds_exactly_the_frozen_54_rows(merged: dict[str, Any]) -> None:
    closure = merged["receipt"]["closure"]
    assert closure["row_count"] == 54
    assert closure["row_count_by_arm"] == {
        scorer.ARM_NEUTRAL: 21,
        scorer.ARM_CLEAN_REPLAY: 21,
        scorer.ARM_BENIGN: 12,
    }
    assert closure["shard_count"] == 12
    assert len(closure["image_ids"]) == 12
    rows = [
        json.loads(line)
        for line in (merged["merged_dir"] / sut.MERGED_ROWS_NAME).read_text().splitlines()
    ]
    assert len(rows) == 54
    assert len({row["request_id"] for row in rows}) == 54


def test_merge_receipt_self_seals_and_declares_its_own_gate_boundary(
    merged: dict[str, Any],
) -> None:
    receipt = copy.deepcopy(merged["receipt"])
    declared = receipt.pop("receipt_content_sha256")
    assert _sha256_json(receipt) == declared
    for gate in sut.MERGE_OWNED_GATES:
        assert merged["receipt"]["gates"][gate]["passed"] is True
    assert merged["receipt"]["gates"]["deferred_to_analysis"] == list(
        sut.ANALYSIS_OWNED_GATES
    )
    policy = merged["receipt"]["policy"]
    assert policy["materiality_decided_here"] is False
    assert policy["route_decided_here"] is False
    assert policy["specificity_gate_decided_here"] is False
    assert policy["same_run_benign_replay_owns_both_relative_estimands"] is True
    for name, entry in merged["receipt"]["output_file_digests"].items():
        payload = (merged["merged_dir"] / name).read_bytes()
        assert entry["sha256"] == crossing_scorer.sha256_bytes(payload)


def test_a_missing_shard_fails_closed(captured: dict[str, Any], tmp_path: Path) -> None:
    with pytest.raises(sut.NeutralRowMergeContractError, match="covers image"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=captured["shard_dirs"][:-1],
            output_dir=tmp_path / "merged",
        )


def test_a_duplicated_shard_fails_closed(captured: dict[str, Any], tmp_path: Path) -> None:
    duplicate = _copy_shard(captured["shard_dirs"][0], tmp_path / "dup")
    with pytest.raises(sut.NeutralRowMergeContractError, match="more than one shard"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=[*captured["shard_dirs"], duplicate],
            output_dir=tmp_path / "merged",
        )


def test_a_partial_image_is_never_merged(captured: dict[str, Any], tmp_path: Path) -> None:
    victim = _copy_shard(captured["shard_dirs"][3], tmp_path / "partial")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    _reseal_shard(victim, rows[:-1])
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[3] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="sealed requests"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_duplicated_row_inside_one_shard_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "dupe-row")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    _reseal_shard(victim, [*rows, copy.deepcopy(rows[0])])
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="more than once"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_cross_owner_attribution_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "cross-owner")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    rows[0]["gt_owner_id"] = "gt:4134:3"
    _reseal_shard(victim, rows)
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="not the sealed"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_quarantined_shard_is_never_merged(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "quarantined")
    (victim / scorer.QUARANTINE_NAME).write_text("{}")
    with pytest.raises(sut.NeutralRowMergeContractError, match="quarantined"):
        sut.read_shard(victim)


def test_an_unknown_artifact_in_a_shard_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "unknown")
    (victim / "stray.json").write_text("{}")
    with pytest.raises(sut.NeutralRowMergeContractError, match="unknown artifact"):
        sut.read_shard(victim)


def test_a_capture_that_never_enforced_native_replay_is_never_merged(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    """A non-evidence-bearing backend produces rows the merge must refuse."""

    raw = raw_capture_shard(captured["plan"], tmp_path / "raw-shard")
    with pytest.raises(
        sut.NeutralRowMergeContractError, match="replay_admission_enforced"
    ):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=[raw],
            output_dir=tmp_path / "merged",
        )


# ---------------------------------------------------------------------------
# 3. Row-level re-proof
# ---------------------------------------------------------------------------


def test_a_tampered_delta_does_not_reconstruct_from_its_own_roots(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "tampered-delta")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    rows[0]["deltas"]["coordinates"]["delta"] -= 5.0
    _reseal_shard(victim, rows)
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="do not reconstruct"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_tampered_selected_logprob_does_not_reconstruct_its_segment_sums(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "tampered-logprob")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    rows[0]["roots"][scorer.ROOT_BASELINE]["selected_logprobs"][0] -= 3.0
    _reseal_shard(victim, rows)
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="per-segment sums"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_row_scoring_tokens_the_plan_does_not_seal_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "tokens")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    rows[0]["scored_token_ids"] = [*rows[0]["scored_token_ids"][:-1], BOX_END]
    rows[0]["scored_token_ids"][1] += 1
    _reseal_shard(victim, rows)
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="sealed target tokens"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_drifted_row_policy_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "row-policy")
    rows = [
        json.loads(line) for line in (victim / scorer.ROWS_NAME).read_text().splitlines()
    ]
    rows[0]["retokenized"] = True
    _reseal_shard(victim, rows)
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="drifted row policy"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_shard_captured_against_another_plan_fails_closed(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "foreign-plan")
    receipt = json.loads((victim / scorer.RECEIPT_NAME).read_text())
    receipt.pop("receipt_content_sha256")
    receipt["plan"]["manifest_content_sha256"] = "0" * 64
    receipt["receipt_content_sha256"] = _sha256_json(receipt)
    (victim / scorer.RECEIPT_NAME).write_bytes(
        crossing_scorer.canonical_json_bytes(receipt) + b"\n"
    )
    with pytest.raises(sut.NeutralRowMergeContractError, match="different CPU plan"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=[victim],
            output_dir=tmp_path / "merged",
        )


# ---------------------------------------------------------------------------
# 4. Merge-owned gates
# ---------------------------------------------------------------------------


def test_gate_one_proves_the_scored_e_tokens_match_across_arms(
    merged: dict[str, Any],
) -> None:
    gate = merged["receipt"]["gates"][sut.GATE_INPUT_AND_TOKEN_IDENTITY]
    assert gate["checked_owner_count"] == 21
    assert gate["identical_scored_e_tokens_across_arms"] is True
    assert gate["paired_roots_forced_identical_tokens"] is True


def test_gate_two_replays_every_owner_within_1e_minus_3(merged: dict[str, Any]) -> None:
    gate = merged["receipt"]["gates"][sut.GATE_RUNTIME_REPLAY]
    assert gate["checked_owner_count"] == 21
    assert gate["quarantined_owner_count"] == 0
    assert gate["max_abs_diff"] <= 1e-3
    assert gate["tolerance"] == 1e-3
    assert gate["max_quarantined_owners"] == 2
    for entry in gate["owner_rows"]:
        assert entry["compared_argmax_preserved"] is True
        assert entry["admitted"] is True


def test_gate_two_quarantines_a_drifted_baseline_and_stops_above_two(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    plan = sut.load_frozen_plan(captured["plan_dir"])
    rows = [
        json.loads(line)
        for shard_dir in captured["shard_dirs"]
        for line in (shard_dir / scorer.ROWS_NAME).read_text().splitlines()
    ]
    shards = [
        sut.validate_shard(shard_dir, plan=plan) for shard_dir in captured["shard_dirs"]
    ]

    def _drift(owner_ids: set[str]) -> list[dict[str, Any]]:
        drifted = copy.deepcopy(rows)
        for row in drifted:
            if row["gt_owner_id"] in owner_ids and row["arm"] in scorer.PAIRED_CROSSING_ARMS:
                row["sealed_reference"] = {
                    **row["sealed_reference"],
                    "baseline_coordinate_sum": (
                        row["sealed_reference"]["baseline_coordinate_sum"] - 4.0
                    ),
                }
        return drifted

    executed = plan.executed_owner_ids
    two = sut.evaluate_merge_gates(plan, _drift(set(executed[:2])), shards)
    replay = two[sut.GATE_RUNTIME_REPLAY]
    assert replay["quarantined_owner_count"] == 2
    assert replay["quarantined_owner_ids"] == sorted(executed[:2])
    with pytest.raises(sut.NeutralRowMergeContractError, match="more than the frozen 2"):
        sut.evaluate_merge_gates(plan, _drift(set(executed[:3])), shards)


def test_gate_two_quarantines_a_flipped_baseline_argmax(
    captured: dict[str, Any],
) -> None:
    plan = sut.load_frozen_plan(captured["plan_dir"])
    rows = [
        json.loads(line)
        for shard_dir in captured["shard_dirs"]
        for line in (shard_dir / scorer.ROWS_NAME).read_text().splitlines()
    ]
    shards = [
        sut.validate_shard(shard_dir, plan=plan) for shard_dir in captured["shard_dirs"]
    ]
    victim = plan.executed_owner_ids[0]
    for row in rows:
        if row["gt_owner_id"] == victim and row["arm"] in scorer.PAIRED_CROSSING_ARMS:
            row["sealed_reference"] = {
                **row["sealed_reference"],
                "baseline_argmax_token_ids_sha256": "d" * 64,
            }
    gates = sut.evaluate_merge_gates(plan, rows, shards)
    replay = gates[sut.GATE_RUNTIME_REPLAY]
    assert replay["quarantined_owner_ids"] == [victim]
    entry = next(item for item in replay["owner_rows"] if item["gt_owner_id"] == victim)
    assert entry["compared_argmax_preserved"] is False
    assert entry["abs_diff"] <= 1e-3


def test_gate_two_refuses_two_arms_that_did_not_share_one_baseline(
    captured: dict[str, Any],
) -> None:
    plan = sut.load_frozen_plan(captured["plan_dir"])
    rows = [
        json.loads(line)
        for shard_dir in captured["shard_dirs"]
        for line in (shard_dir / scorer.ROWS_NAME).read_text().splitlines()
    ]
    shards = [
        sut.validate_shard(shard_dir, plan=plan) for shard_dir in captured["shard_dirs"]
    ]
    victim = plan.executed_owner_ids[0]
    for row in rows:
        if row["gt_owner_id"] == victim and row["arm"] == scorer.ARM_NEUTRAL:
            row["roots"][scorer.ROOT_BASELINE]["segment_sums"]["coordinates"]["sum"] -= 2.0
    with pytest.raises(sut.NeutralRowMergeContractError, match="did not replay one"):
        sut.evaluate_merge_gates(plan, rows, shards)


def test_gate_five_carries_the_inherited_parity_of_every_shard(
    merged: dict[str, Any],
) -> None:
    gate = merged["receipt"]["gates"][sut.GATE_CACHE_PARITY]
    assert gate["tolerance"] == 1e-3
    assert gate["inherited_cache_admitted"] == ["True"]
    assert len(gate["shard_root_backends"]) == 12


def test_gate_five_seals_the_one_smoke_that_admitted_every_shard(
    merged: dict[str, Any],
) -> None:
    inherited = merged["receipt"]["gates"][sut.GATE_CACHE_PARITY]["inherited_smoke"]
    assert inherited["required_strata"] == list(sut.REQUIRED_SMOKE_STRATA)
    assert inherited["strata_proven_by_every_shard"] == list(sut.REQUIRED_SMOKE_STRATA)
    assert inherited["smoke_image_id"] == SMOKE_IMAGE_ID
    assert len(inherited["admission_content_sha256"]) == 64
    assert inherited["admitted_shard_count"] == 12


def _stub_shard(session_image_id: str, **inherited: Any) -> sut.ValidatedShard:
    """The smallest shard the one-smoke helper reads: its inherited admission."""

    return sut.ValidatedShard(
        shard_dir=Path(f"/nonexistent/{session_image_id}"),
        shard_id=f"shard-{session_image_id}",
        session_image_id=session_image_id,
        receipt={},
        receipt_content_sha256="",
        parity={
            "inherited_admission": {
                "smoke_image_id": SMOKE_IMAGE_ID,
                "admission_content_sha256": "a" * 64,
                "smoke_strata_proven": list(scorer.REQUIRED_SMOKE_STRATA),
                **inherited,
            }
        },
        rows=[],
        file_sha256={},
    )


def test_the_one_smoke_helper_reproves_the_strata_it_seals() -> None:
    """The sealed fact stands alone, not on the caller having validated first."""

    sealed = sut.assert_one_smoke_admission(
        [_stub_shard("4134"), _stub_shard("10707")]
    )
    assert sealed["strata_proven_by_every_shard"] == list(sut.REQUIRED_SMOKE_STRATA)
    assert sealed["admitted_shard_count"] == 2
    with pytest.raises(sut.NeutralRowMergeContractError, match="not one smoke proving"):
        sut.assert_one_smoke_admission(
            [
                _stub_shard("4134"),
                _stub_shard(
                    "10707", smoke_strata_proven=list(sut.REQUIRED_SMOKE_STRATA)[:2]
                ),
            ]
        )


@pytest.mark.parametrize(
    ("label", "mutate"),
    [
        (
            "dropped",
            lambda inherited: inherited.__setitem__(
                "smoke_strata_proven", list(sut.REQUIRED_SMOKE_STRATA)[:-1]
            ),
        ),
        (
            "reordered",
            lambda inherited: inherited.__setitem__(
                "smoke_strata_proven", list(reversed(sut.REQUIRED_SMOKE_STRATA))
            ),
        ),
        (
            "renamed",
            lambda inherited: inherited.__setitem__(
                "smoke_strata_proven",
                [*list(sut.REQUIRED_SMOKE_STRATA)[:-1], "any_description"],
            ),
        ),
    ],
)
def test_a_shard_whose_smoke_proved_the_wrong_strata_is_never_merged(
    captured: dict[str, Any],
    tmp_path: Path,
    label: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    """unit.md gate 5 owns the exact four strata, in their frozen order."""

    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / f"strata-{label}")
    _rewrite_inherited_admission(victim, mutate)
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="unit.md gate 5 requires"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_a_shard_whose_smoke_recorded_no_strata_is_never_merged(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    """An admission from before the strata contract admits nothing."""

    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / "strata-absent")
    _rewrite_inherited_admission(victim, lambda inherited: inherited.pop("smoke_strata_proven"))
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match="names no smoke_strata_proven"):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("smoke_image_id", "13923", "2 different smoke images"),
        ("admission_content_sha256", "f" * 64, "2 different admission receipts"),
        ("smoke_image_id", "", "names no smoke_image_id"),
        ("admission_content_sha256", "", "names no admission_content_sha256"),
    ],
)
def test_shards_that_did_not_inherit_one_identical_smoke_are_never_merged(
    captured: dict[str, Any], tmp_path: Path, field: str, value: str, message: str
) -> None:
    """A second admission means the evidence never stood behind one smoke."""

    victim = _copy_shard(captured["shard_dirs"][0], tmp_path / f"divergent-{field}-{len(value)}")
    _rewrite_inherited_admission(victim, lambda inherited: inherited.__setitem__(field, value))
    shard_dirs = [*captured["shard_dirs"]]
    shard_dirs[0] = victim
    with pytest.raises(sut.NeutralRowMergeContractError, match=message):
        sut.merge_shards(
            plan_dir=captured["plan_dir"],
            shard_dirs=shard_dirs,
            output_dir=tmp_path / "merged",
        )


def test_every_merged_shard_inherits_one_four_strata_smoke(
    captured: dict[str, Any],
) -> None:
    """unit.md's representative smoke, as the merged shards actually record it.

    All twelve captures -- including the eleven images that carry too few strata
    to smoke themselves -- inherit one admission, sealed on one image session
    that proved every stratum.
    """

    inherited = [
        json.loads((shard_dir / scorer.PARITY_NAME).read_text())["inherited_admission"]
        for shard_dir in captured["shard_dirs"]
    ]
    assert len(inherited) == 12
    assert {entry["smoke_image_id"] for entry in inherited} == {SMOKE_IMAGE_ID}
    assert len({entry["admission_content_sha256"] for entry in inherited}) == 1
    for entry in inherited:
        assert entry["smoke_strata_proven"] == list(scorer.REQUIRED_SMOKE_STRATA)


def test_merging_twice_is_create_or_identical(
    captured: dict[str, Any], tmp_path: Path
) -> None:
    output_dir = tmp_path / "merged"
    first = sut.merge_shards(
        plan_dir=captured["plan_dir"],
        shard_dirs=captured["shard_dirs"],
        output_dir=output_dir,
    )
    before = sorted((path.name, path.read_bytes()) for path in output_dir.iterdir())
    second = sut.merge_shards(
        plan_dir=captured["plan_dir"],
        shard_dirs=captured["shard_dirs"],
        output_dir=output_dir,
    )
    after = sorted((path.name, path.read_bytes()) for path in output_dir.iterdir())
    assert first["receipt_content_sha256"] == second["receipt_content_sha256"]
    assert before == after
