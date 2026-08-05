"""Focused tests for the immutable primary-only merge of the sorted
crossing-boundary owner release/realization capture shards.

Fixture policy
--------------
Every ladder payload below is produced by the *scorer's own*
``_ladder_payload``/``_support_payload`` builders from real
``LadderReadout``/``OwnerRankResult``/``TargetLocalSupportResult`` objects, so a
schema drift in the capture surface breaks these tests instead of silently
passing a stale hand-written dict.  Receipts, the plan manifest and the smoke
admission are sealed with the scorer's own ``sha256_json`` canonicalization for
the same reason.

The fixture is full-scale on purpose: all twelve frozen images, the real
per-image primary owner counts, 26 primary owners, 14 disjoint timing controls
and 12 native-TP replay controls.  The frozen denominators are the contract
under test, so they are never shrunk or monkeypatched here.

``build_capture`` and its helpers are imported by
``test_analyze_sorted_crossing_boundary_owner_release`` rather than duplicated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import pytest

from scripts.research import merge_sorted_crossing_boundary_owner_release as merge
from scripts.research import score_sorted_crossing_boundary_owner_release as scorer

# ---------------------------------------------------------------------------
# Frozen fixture constants
# ---------------------------------------------------------------------------

#: The sealed plan's real per-image primary owner counts; they sum to 26.
FROZEN_PER_IMAGE_PRIMARY: Mapping[str, int] = {
    "10707": 1,
    "13348": 1,
    "13923": 3,
    "14038": 3,
    "14439": 2,
    "1584": 1,
    "16228": 5,
    "2685": 1,
    "4134": 5,
    "5001": 2,
    "6040": 1,
    "7511": 1,
}
IMAGE_IDS: tuple[str, ...] = tuple(sorted(FROZEN_PER_IMAGE_PRIMARY))
SMOKE_IMAGE_ID = "4134"

CALIBRATION_SOURCE = "sealed_census_calibration_sha256:" + "9d" * 32
PLAN_BUILDER_SHA256 = "ab" * 32


def timing_control_owner_ids() -> list[str]:
    """14 disjoint timing controls, spread deterministically over the twelve images."""

    owner_ids: list[str] = []
    index = 0
    while len(owner_ids) < 14:
        image_id = IMAGE_IDS[index % len(IMAGE_IDS)]
        owner_ids.append(f"gt:{image_id}:t{index // len(IMAGE_IDS)}")
        index += 1
    return sorted(owner_ids)


def tp_replay_control_owner_ids() -> list[str]:
    """One due-boundary native true positive per image."""

    return sorted(f"gt:{image_id}:c0" for image_id in IMAGE_IDS)


def _image_of(owner_id: str) -> str:
    return owner_id.split(":")[1]


# ---------------------------------------------------------------------------
# Ladder payloads, built through the scorer's own serializers
# ---------------------------------------------------------------------------


def make_support(bound: str, disposition: str) -> scorer.TargetLocalSupportResult:
    return scorer.TargetLocalSupportResult(
        bound=bound,
        support_disposition=disposition,
        features=scorer.SupportBoundFeatures(
            bound=bound,
            owner_best=-46.5,
            owner_best_candidate_id="cand:" + "0" * 24,
            peak_lift=3.1,
            local_concentration=2.2,
            bank_median=-50.25,
            bank_size=11,
            unique_population_size=61,
        ),
        peak_lift_threshold=2.3884847780085985,
        local_concentration_threshold=1.63233060836792,
        epsilon=scorer.SUPPORT_EPSILON,
        calibration_source=CALIBRATION_SOURCE,
    )


def make_greedy(
    status: str, *, owner_match: str | None = None, nonunique: bool = False
) -> dict[str, Any]:
    valid = status != scorer.GREEDY_MALFORMED
    return {
        "greedy_status": status,
        "greedy_owner_match": owner_match,
        "greedy_nonunique_match": bool(nonunique),
        "malformed_reason": None if valid else "domain: step 0 decoded a non-coordinate token",
        "decoded_bbox_pixel_xyxy": [10, 20, 30, 40] if valid else None,
        "assignment": None,
        "grammar_status": "valid" if valid else "malformed",
        "coord_token_ids": [1, 2, 3, 4] if valid else [68147],
        "token_ids": [1, 2, 3, 4, 5] if valid else [68147],
        "complete_box_logprob_sum": -3.5 if valid else None,
    }


def make_release(
    *, boundary_label: str, same_description: bool, margin: float | None, argmax_follows: bool
) -> scorer.NaturalReleaseObservation:
    return scorer.NaturalReleaseObservation(
        boundary_label=boundary_label,
        same_description=same_description,
        first_divergence_index=None if same_description else 1,
        target_minus_native_margin=margin,
        description_sum=-6.5,
        description_token_mean=-1.3,
        description_token_count=5,
        argmax_follows_target=argmax_follows,
        target_first_divergent_token_id=None if same_description else 900,
        native_first_divergent_token_id=None if same_description else 901,
    )


def make_rank(
    *,
    owner_id: str,
    target_rank: int | None,
    competitor: str | None,
    margin: float | None,
    disposition: str,
) -> scorer.OwnerRankResult | None:
    if target_rank is None:
        return None
    return scorer.OwnerRankResult(
        target_owner_id=owner_id,
        target_rank=int(target_rank),
        best_competitor_owner_id=competitor,
        target_minus_competitor_margin=margin,
        family_rank_disposition=disposition,
    )


def make_ladder(
    *,
    boundary_label: str,
    context_id: str,
    rank: scorer.OwnerRankResult | None,
    release: scorer.NaturalReleaseObservation | None,
    greedy: Mapping[str, Any],
    support_u: str,
    support_l: str,
    native_action_kind: str = "native_row",
    replay_admitted: bool = True,
) -> dict[str, Any]:
    """One ladder payload, serialized by the scorer's own ``_ladder_payload``."""

    readout = scorer.LadderReadout(
        boundary_label=boundary_label,
        context_id=context_id,
        context_group_id=scorer.sha256_json([context_id, boundary_label]),
        release=release,
        native_action_kind=native_action_kind,
        native_replay_admitted=replay_admitted,
        native_replay_depth=7,
        owner_rank=rank,
        greedy=dict(greedy),
        candidate_count=61,
        unattributable_candidate_ids=(),
        release_scoring_backend=scorer.KV_CACHE_BACKEND,
        coordinate_scoring_backend=scorer.KV_CACHE_BACKEND,
        support_by_bound={
            scorer.SUPPORT_BOUND_U: make_support(scorer.SUPPORT_BOUND_U, support_u),
            scorer.SUPPORT_BOUND_L: make_support(scorer.SUPPORT_BOUND_L, support_l),
        },
    )
    return scorer._ladder_payload(readout)  # noqa: SLF001 - the frozen serializer


# ---------------------------------------------------------------------------
# Owner specs: the branch each synthetic owner is built to land in
# ---------------------------------------------------------------------------

OTHER_OWNER = "gt:other:99"


@dataclass(frozen=True)
class OwnerSpec:
    """One synthetic primary owner, described by the branch it should reach."""

    intent: str
    stratum: str
    same_description: bool = False
    quarantined: bool = False
    #: The ``P`` leg, which drives the paired access tags and transitions.
    p_release_margin: float | None = 1.0
    p_support_u: str = scorer.SUPPORT_SUPPORTED
    p_support_l: str = scorer.SUPPORT_SUPPORTED
    p_rank: int = 1
    p_margin: float | None = 1.5
    p_competitor: str | None = OTHER_OWNER
    p_greedy_status: str = scorer.GREEDY_TARGET_MATCH


#: ``P+E`` readouts per intent: (rank_rank, competitor, competitor_margin,
#: rank_disposition, release margin, greedy status, greedy owner match,
#: greedy nonunique, U support, L support).
_INTENTS: Mapping[str, dict[str, Any]] = {
    "displaced_likelihood": {
        "rank": 3,
        "competitor": OTHER_OWNER,
        "margin": -2.0,
        "disposition": scorer.RANK_TARGET_OUTRANKED,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_UNMATCHED,
        "greedy_owner_match": None,
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
    "displaced_greedy": {
        "rank": 1,
        "competitor": OTHER_OWNER,
        "margin": 2.0,
        "disposition": scorer.RANK_TARGET_FIRST,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_OTHER_OWNER_MATCH,
        "greedy_owner_match": OTHER_OWNER,
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
    "displaced_decoding_contradicted": {
        "rank": 3,
        "competitor": OTHER_OWNER,
        "margin": -2.0,
        "disposition": scorer.RANK_TARGET_OUTRANKED,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_TARGET_MATCH,
        "greedy_owner_match": None,  # filled with the target id at build time
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
    "release_lost": {
        "rank": 1,
        "competitor": OTHER_OWNER,
        "margin": 2.0,
        "disposition": scorer.RANK_TARGET_FIRST,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_TARGET_MATCH,
        "greedy_owner_match": None,
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
    "realization_fail": {
        "rank": 1,
        "competitor": OTHER_OWNER,
        "margin": 2.0,
        "disposition": scorer.RANK_TARGET_FIRST,
        "release_margin": 0.5,
        "greedy_status": scorer.GREEDY_MALFORMED,
        "greedy_owner_match": None,
        "support_u": scorer.SUPPORT_UNSUPPORTED,
        "support_l": scorer.SUPPORT_UNSUPPORTED,
    },
    "ambiguous_tie": {
        "rank": 1,
        "competitor": OTHER_OWNER,
        "margin": 0.0,
        "disposition": scorer.RANK_TIE,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_TARGET_MATCH,
        "greedy_owner_match": None,
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
    "ambiguous_nonunique": {
        "rank": 1,
        "competitor": OTHER_OWNER,
        "margin": 2.0,
        "disposition": scorer.RANK_TARGET_FIRST,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_AMBIGUOUS,
        "greedy_owner_match": OTHER_OWNER,
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
    "ambiguous_calibration": {
        "rank": 1,
        "competitor": OTHER_OWNER,
        "margin": 2.0,
        "disposition": scorer.RANK_TARGET_FIRST,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_UNMATCHED,
        "greedy_owner_match": None,
        "support_u": scorer.SUPPORT_CALIBRATION_UNAVAILABLE,
        "support_l": scorer.SUPPORT_CALIBRATION_UNAVAILABLE,
    },
    "ambiguous_missing": {
        "rank": None,
        "competitor": None,
        "margin": None,
        "disposition": scorer.RANK_TARGET_FIRST,
        "release_margin": -1.0,
        "greedy_status": scorer.GREEDY_UNMATCHED,
        "greedy_owner_match": None,
        "support_u": scorer.SUPPORT_SUPPORTED,
        "support_l": scorer.SUPPORT_SUPPORTED,
    },
}


def make_primary_owner_record(spec: OwnerSpec, *, owner_id: str, shard_id: str) -> dict[str, Any]:
    """One current-schema primary owner record shaped to reach ``spec.intent``."""

    intent = _INTENTS[spec.intent]
    image_id = _image_of(owner_id)
    greedy_match = intent["greedy_owner_match"]
    if intent["greedy_status"] == scorer.GREEDY_TARGET_MATCH:
        greedy_match = owner_id
    at_p_plus_e = make_ladder(
        boundary_label="P_plus_E",
        context_id=f"{image_id}:boundary-{owner_id[-3:]}-ppe",
        rank=make_rank(
            owner_id=owner_id,
            target_rank=intent["rank"],
            competitor=intent["competitor"],
            margin=intent["margin"],
            disposition=intent["disposition"],
        ),
        release=make_release(
            boundary_label="P_plus_E",
            same_description=spec.same_description,
            margin=None if spec.same_description else intent["release_margin"],
            argmax_follows=False,
        ),
        greedy=make_greedy(
            intent["greedy_status"],
            owner_match=greedy_match,
            nonunique=intent["greedy_status"] == scorer.GREEDY_AMBIGUOUS,
        ),
        support_u=intent["support_u"],
        support_l=intent["support_l"],
        replay_admitted=not spec.quarantined,
    )
    at_p = make_ladder(
        boundary_label="P",
        context_id=f"{image_id}:boundary-{owner_id[-3:]}-p",
        rank=make_rank(
            owner_id=owner_id,
            target_rank=spec.p_rank,
            competitor=spec.p_competitor,
            margin=spec.p_margin,
            disposition=(
                scorer.RANK_TARGET_FIRST
                if (spec.p_margin or 0.0) > 0
                else scorer.RANK_TARGET_OUTRANKED
            ),
        ),
        release=make_release(
            boundary_label="P",
            same_description=spec.same_description,
            margin=None if spec.same_description else spec.p_release_margin,
            argmax_follows=bool((spec.p_release_margin or 0.0) > 0),
        ),
        greedy=make_greedy(
            spec.p_greedy_status,
            owner_match=owner_id if spec.p_greedy_status == scorer.GREEDY_TARGET_MATCH else None,
        ),
        support_u=spec.p_support_u,
        support_l=spec.p_support_l,
        replay_admitted=not spec.quarantined,
    )
    return {
        "schema_version": scorer.OWNER_RECORD_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "row_kind": "crossing_boundary_owner_record",
        "shard_id": shard_id,
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "cohort": merge.PRIMARY_COHORT,
        "stratum": spec.stratum,
        "normalized_description": "person",
        "same_description_as_e": spec.same_description,
        "description_observability": (
            "same_description_coordinate_only"
            if spec.same_description
            else "different_description_release_observable"
        ),
        "quarantined": spec.quarantined,
        "replay_admitted": not spec.quarantined,
        "ladders": {"at_p": at_p, "at_p_plus_e": at_p_plus_e},
        "branch_assignment": merge.BRANCH_ASSIGNMENT_SENTINEL,
        "secondary_compatibility": merge.SECONDARY_COMPATIBILITY_SENTINEL,
    }


def make_control_owner_record(owner_id: str, *, cohort: str, shard_id: str) -> dict[str, Any]:
    image_id = _image_of(owner_id)
    variants = (
        ("at_due_boundary",)
        if cohort == merge.TP_REPLAY_CONTROL_COHORT
        else ("at_control_boundary", "at_control_boundary_plus_row")
    )
    ladders = {
        variant: make_ladder(
            boundary_label=scorer.BOUNDARY_LABEL_BY_VARIANT[variant],
            context_id=f"{image_id}:{variant}:{owner_id}",
            rank=make_rank(
                owner_id=owner_id,
                target_rank=1,
                competitor=OTHER_OWNER,
                margin=1.25,
                disposition=scorer.RANK_TARGET_FIRST,
            ),
            release=make_release(
                boundary_label=scorer.BOUNDARY_LABEL_BY_VARIANT[variant],
                same_description=False,
                margin=0.75,
                argmax_follows=True,
            ),
            greedy=make_greedy(scorer.GREEDY_TARGET_MATCH, owner_match=owner_id),
            support_u=scorer.SUPPORT_SUPPORTED,
            support_l=scorer.SUPPORT_SUPPORTED,
        )
        for variant in variants
    }
    return {
        "schema_version": scorer.OWNER_RECORD_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "row_kind": "crossing_boundary_owner_record",
        "shard_id": shard_id,
        "gt_owner_id": owner_id,
        "image_id": image_id,
        "cohort": cohort,
        "stratum": None,
        "normalized_description": "person",
        "same_description_as_e": None,
        "description_observability": None,
        "quarantined": False,
        "replay_admitted": True,
        "ladders": ladders,
        "branch_assignment": merge.BRANCH_ASSIGNMENT_SENTINEL,
        "secondary_compatibility": merge.SECONDARY_COMPATIBILITY_SENTINEL,
    }


def make_score_row(*, image_id: str, shard_id: str, owner_id: str, index: int) -> dict[str, Any]:
    return {
        "schema_version": scorer.SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "row_kind": "crossing_boundary_score",
        "shard_id": shard_id,
        "image_id": image_id,
        "gt_owner_id": owner_id,
        "request_id": f"req:{image_id}:{owner_id}:{index}",
        "request_family": scorer.REQUEST_COORDINATE_TARGET_LOCAL,
        "readout_tier": scorer.PRIMARY_READOUT_TIER,
        "variant": "at_p_plus_e",
        "boundary_label": "P_plus_E",
        "context_id": f"{image_id}:boundary-{index}",
        "cohort": merge.PRIMARY_COHORT,
        "likelihood_channel": scorer.LIKELIHOOD_CHANNEL,
        "repetition_penalty_stratum": 1.0,
        "retokenized": False,
        "sampling": "disabled_primary_deterministic_pass_only",
        "scoring_backend": scorer.KV_CACHE_BACKEND,
        "uses_model_generate": False,
        "selected_logprobs": [-1.5, -2.5],
        "scored_token_ids": [11, 12],
    }


# ---------------------------------------------------------------------------
# Runtime identity, plan, admission, shards
# ---------------------------------------------------------------------------


def make_runtime_identity(*, marker: str = "frozen") -> dict[str, Any]:
    return {
        "backend": "hf",
        "is_real_model": True,
        "usable_as_evidence": True,
        "model_identity": {"base": {"path": "/models/qwen3-vl-2b"}, "revision": marker},
        "tokenizer_identity": {"sha256": "7f" * 32},
        "adapter_identity": {"adapter_type": "dora", "rank": 16},
        "numerics": {
            "likelihood_channel": scorer.LIKELIHOOD_CHANNEL,
            "explicit_position_ids": True,
            "uses_model_generate": False,
            "repetition_penalty_stratum": 1.0,
            "vision_kwargs_at_prefill_only": True,
        },
        "source_identity": {
            merge.SCORER_SOURCE_IDENTITY_KEY: merge.FROZEN_SCORER_SOURCE_SHA256,
            "scripts.research.prepare_sorted_crossing_boundary_owner_release_realization": (
                PLAN_BUILDER_SHA256
            ),
        },
        "vocab_size": 152680,
        "layer_count": 28,
    }


def _seal(payload: dict[str, Any], key: str) -> dict[str, Any]:
    payload[key] = scorer.sha256_json({k: v for k, v in payload.items() if k != key})
    return payload


def build_plan(tmp_path: Path, *, marker: str = "frozen") -> tuple[Path, dict[str, Any]]:
    """A minimal but fully self-sealed stand-in for the frozen v3 plan.

    ``marker`` travels through the sealed lineage, so two plans built at
    different paths are genuinely different revisions rather than byte twins.
    """

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True)
    digests: dict[str, Any] = {}
    for name in merge.PLAN_FILE_NAMES:
        payload = json.dumps({"plan_file": name}, sort_keys=True).encode("utf-8") + b"\n"
        (plan_dir / name).write_bytes(payload)
        digests[name] = {
            "path": name,
            "byte_size": len(payload),
            "row_count": 1,
            "sha256": scorer.sha256_bytes(payload),
        }

    manifest: dict[str, Any] = {
        "schema_version": merge.FROZEN_PLAN_MANIFEST_SCHEMA_VERSION,
        "builder_source": {
            "path": "scripts/research/prepare_sorted_crossing_boundary_owner_release_realization.py",
            "byte_size": 127581,
            "sha256": PLAN_BUILDER_SHA256,
        },
        "cohort_counts": {
            "u_bound_crossing_count": scorer.PRIMARY_OWNER_COUNT_U,
            "l_bound_crossing_count": scorer.PRIMARY_OWNER_COUNT_L,
            "exact_same_context_u_and_l_count": scorer.PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL,
            "matched_e_count": scorer.MATCHED_E_OWNER_COUNT,
            "unmatched_e_count": scorer.UNMATCHED_E_OWNER_COUNT,
            "per_image_owner_counts": dict(FROZEN_PER_IMAGE_PRIMARY),
        },
        "control_counts": {
            "disjoint_from_primary_cohort": True,
            "timing_control_count": scorer.TIMING_CONTROL_OWNER_COUNT,
            "timing_control_owner_ids": timing_control_owner_ids(),
            "tp_replay_control_count": scorer.TP_CALIBRATION_OWNER_COUNT,
            "tp_replay_control_owner_ids": tp_replay_control_owner_ids(),
        },
        "lineage": {
            "census_unit_id": "2026-08-03-sorted-owner-accessibility-phenotype-census",
            "census_run_root": f"/frozen/{marker}",
        },
        "support_calibration": {
            "support_contract_id": scorer.SUPPORT_CALIBRATION_CONTRACT_ID,
            "calibration_sha256": CALIBRATION_SOURCE.split(":")[-1],
        },
        "output_file_digests": digests,
    }
    _seal(manifest, "manifest_content_sha256")
    (plan_dir / "manifest.json").write_bytes(
        scorer.canonical_json_bytes(manifest) + b"\n"
    )
    return plan_dir, manifest


def build_admission(
    tmp_path: Path, *, manifest: Mapping[str, Any], runtime_identity: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    admission: dict[str, Any] = {
        "schema_version": scorer.ADMISSION_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "scope": "one_image_smoke_admits_every_capture_shard_of_the_same_plan_and_runtime",
        "smoke_shard_id": f"smoke-{SMOKE_IMAGE_ID}",
        "smoke_image_id": SMOKE_IMAGE_ID,
        "cache_admitted": True,
        "admitted_batch_size": 16,
        "plan_manifest_content_sha256": manifest["manifest_content_sha256"],
        "surface_backend": {
            scorer.SURFACE_RELEASE: scorer.KV_CACHE_BACKEND,
            scorer.SURFACE_COORDINATE: scorer.KV_CACHE_BACKEND,
        },
        "smoke_rows": [
            {"role": role} for role in sorted(scorer.REQUIRED_SMOKE_MATRIX_ROLES)
        ],
        "admission_identity_fields": list(scorer.ADMISSION_IDENTITY_FIELDS),
    }
    admission.update({key: runtime_identity[key] for key in scorer.ADMISSION_IDENTITY_FIELDS})
    admission["runtime_identity_sha256"] = scorer.runtime_identity_digest(runtime_identity)
    _seal(admission, "admission_content_sha256")

    path = tmp_path / "smoke" / scorer.ADMISSION_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scorer.canonical_json_bytes(admission) + b"\n")
    return path, admission


def make_shard_receipt(
    *,
    shard_id: str,
    image_id: str,
    plan_dir: Path,
    manifest: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
    admission: Mapping[str, Any],
    owner_records: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    quarantine_entries: Sequence[Mapping[str, Any]] = (),
    stopped: bool = False,
) -> dict[str, Any]:
    plan_file_sha256 = {
        name: str(manifest["output_file_digests"][name]["sha256"])
        for name in merge.PLAN_FILE_NAMES
    }
    context_group_ids = sorted(
        {
            str(ladder["context_group_id"])
            for record in owner_records
            for ladder in record["ladders"].values()
        }
    )
    receipt: dict[str, Any] = {
        "schema_version": scorer.RECEIPT_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "mode": scorer.MODE_CAPTURE,
        "shard_id": shard_id,
        "scorer_source_sha256": merge.FROZEN_SCORER_SOURCE_SHA256,
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
        "plan": {
            "plan_dir": str(plan_dir),
            "manifest_schema_version": merge.FROZEN_PLAN_MANIFEST_SCHEMA_VERSION,
            "manifest_content_sha256": manifest["manifest_content_sha256"],
            "builder_source_sha256": PLAN_BUILDER_SHA256,
            "plan_file_sha256": plan_file_sha256,
            "lineage": manifest["lineage"],
            "cohort_counts": manifest["cohort_counts"],
            "control_counts": manifest["control_counts"],
            "support_calibration": manifest["support_calibration"],
        },
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_sha256": scorer.runtime_identity_digest(runtime_identity),
        "cohort_denominators": {
            "u_count": scorer.PRIMARY_OWNER_COUNT_U,
            "l_count": scorer.PRIMARY_OWNER_COUNT_L,
            "same_context_ul_count": scorer.PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL,
            "matched_e_count": scorer.MATCHED_E_OWNER_COUNT,
            "unmatched_e_count": scorer.UNMATCHED_E_OWNER_COUNT,
        },
        "timing_control_owner_count": scorer.TIMING_CONTROL_OWNER_COUNT,
        "tp_calibration_owner_count": scorer.TP_CALIBRATION_OWNER_COUNT,
        "executed": {
            "session_image_id": image_id,
            "owner_ids": sorted(str(record["gt_owner_id"]) for record in owner_records),
            "owner_count": len(owner_records),
            "score_row_count": len(score_rows),
            "logical_context_group_count": len(context_group_ids),
            "logical_context_groups_sha256": scorer.sha256_json(context_group_ids),
            "request_ids_sha256": scorer.sha256_json(
                sorted({str(row["request_id"]) for row in score_rows})
            ),
            "deferred_secondary_request_count": 64,
            "not_owner_identifiable_candidate_ids": [],
        },
        "quarantine": {
            "schema_version": scorer.QUARANTINE_SCHEMA_VERSION,
            "count": len(quarantine_entries),
            "maximum_primary_quarantines": scorer.MAX_PRIMARY_QUARANTINES,
            "stopped": bool(stopped),
            "entries": [dict(entry) for entry in quarantine_entries],
        },
        "policy": {
            **dict(merge.REQUIRED_POLICY),
            "owner_match_iou_threshold": scorer.OWNER_MATCH_IOU_THRESHOLD,
            "vision_kwargs_passed_at_prefill_only": True,
        },
        "admission": {
            "smoke_shard_id": admission["smoke_shard_id"],
            "smoke_image_id": admission["smoke_image_id"],
            "admission_content_sha256": admission["admission_content_sha256"],
            "surface_backend": dict(admission["surface_backend"]),
            "admitted_batch_size": admission["admitted_batch_size"],
        },
    }
    return _seal(receipt, "receipt_content_sha256")


def make_parity(*, shard_id: str, image_id: str, admission: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": scorer.PARITY_SCHEMA_VERSION,
        "unit_id": merge.UNIT_ID,
        "shard_id": shard_id,
        "session_image_id": image_id,
        "inherited_admission": {
            "smoke_shard_id": admission["smoke_shard_id"],
            "smoke_image_id": admission["smoke_image_id"],
            "admission_content_sha256": admission["admission_content_sha256"],
            "cache_admitted": admission["cache_admitted"],
            "admitted_batch_size": admission["admitted_batch_size"],
        },
        "surface_backend": dict(admission["surface_backend"]),
        "effective_batch_size": 16,
        "batched_candidate_rows": 775,
        "scalar_candidate_rows": 0,
        "branch_parity_scope": "owner records carry no branch",
    }


def write_shard(
    shard_dir: Path,
    *,
    receipt: Mapping[str, Any],
    owner_records: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / scorer.RECEIPT_NAME).write_bytes(
        scorer.canonical_json_bytes(receipt) + b"\n"
    )
    (shard_dir / scorer.PARITY_NAME).write_bytes(
        scorer.canonical_json_bytes(parity) + b"\n"
    )
    (shard_dir / scorer.OWNER_RECORDS_NAME).write_bytes(
        b"".join(scorer.canonical_json_bytes(row) + b"\n" for row in owner_records)
    )
    (shard_dir / scorer.SCORES_NAME).write_bytes(
        b"".join(scorer.canonical_json_bytes(row) + b"\n" for row in score_rows)
    )
    return shard_dir


@dataclass
class Capture:
    """A complete twelve-shard capture run, plus its plan and admission."""

    root: Path
    plan_dir: Path
    manifest: dict[str, Any]
    admission_path: Path
    admission: dict[str, Any]
    runtime_identity: dict[str, Any]
    shard_root: Path
    shard_dirs: dict[str, Path] = field(default_factory=dict)

    def merge_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "shard_dirs": [self.shard_dirs[image_id] for image_id in IMAGE_IDS],
            "plan_dir": self.plan_dir,
            "admission_path": self.admission_path,
        }
        kwargs.update(overrides)
        return kwargs

    def run_merge(self, output_dir: Path | None = None, **overrides: Any) -> dict[str, Any]:
        result = merge.merge_shards(**self.merge_kwargs(**overrides))
        if output_dir is not None:
            merge.publish_merge(output_dir, result["files"])
        return result

    def read_shard(self, image_id: str) -> dict[str, Any]:
        shard_dir = self.shard_dirs[image_id]
        return {
            "receipt": json.loads((shard_dir / scorer.RECEIPT_NAME).read_text()),
            "parity": json.loads((shard_dir / scorer.PARITY_NAME).read_text()),
            "owner_records": [
                json.loads(line)
                for line in (shard_dir / scorer.OWNER_RECORDS_NAME)
                .read_text()
                .splitlines()
                if line.strip()
            ],
            "score_rows": [
                json.loads(line)
                for line in (shard_dir / scorer.SCORES_NAME).read_text().splitlines()
                if line.strip()
            ],
        }

    def rewrite_shard(
        self,
        image_id: str,
        *,
        owner_records: Sequence[Mapping[str, Any]] | None = None,
        score_rows: Sequence[Mapping[str, Any]] | None = None,
        parity: Mapping[str, Any] | None = None,
        runtime_identity: Mapping[str, Any] | None = None,
        admission: Mapping[str, Any] | None = None,
        quarantine_entries: Sequence[Mapping[str, Any]] = (),
        stopped: bool = False,
        receipt_patch: Mapping[str, Any] | None = None,
        reseal: bool = True,
    ) -> Path:
        """Rebuild one shard, resealing its receipt unless a test wants tampering."""

        current = self.read_shard(image_id)
        records = list(owner_records if owner_records is not None else current["owner_records"])
        rows = list(score_rows if score_rows is not None else current["score_rows"])
        used_admission = dict(admission or self.admission)
        receipt = make_shard_receipt(
            shard_id=f"capture-{image_id}",
            image_id=image_id,
            plan_dir=self.plan_dir,
            manifest=self.manifest,
            runtime_identity=dict(runtime_identity or self.runtime_identity),
            admission=used_admission,
            owner_records=records,
            score_rows=rows,
            quarantine_entries=quarantine_entries,
            stopped=stopped,
        )
        if receipt_patch:
            receipt.update(receipt_patch)
            if reseal:
                _seal(receipt, "receipt_content_sha256")
        return write_shard(
            self.shard_dirs[image_id],
            receipt=receipt,
            owner_records=records,
            score_rows=rows,
            parity=dict(
                parity
                or make_parity(
                    shard_id=f"capture-{image_id}",
                    image_id=image_id,
                    admission=used_admission,
                )
            ),
        )


def routable_specs() -> list[OwnerSpec]:
    """26 primary owners: 17 displaced, 6 release_lost, 1 realization_fail, 2 ambiguous.

    Interpretable = 24, so the two-thirds rule is 17/24 (not 17/26); the strata
    split leaves 12 interpretable matched-E and 12 interpretable unmatched-E.
    """

    intents = (
        ["displaced_likelihood"] * 16
        + ["displaced_decoding_contradicted"]
        + ["release_lost"] * 6
        + ["realization_fail"]
        + ["ambiguous_tie"] * 2
    )
    assert len(intents) == scorer.PRIMARY_OWNER_COUNT_U
    specs: list[OwnerSpec] = []
    for index, intent in enumerate(intents):
        stratum = (
            merge.MATCHED_E_STRATUM
            if index < scorer.MATCHED_E_OWNER_COUNT
            else merge.UNMATCHED_E_STRATUM
        )
        specs.append(OwnerSpec(intent=intent, stratum=stratum))
    return specs


def build_capture(
    tmp_path: Path,
    *,
    primary_specs: Sequence[OwnerSpec] | None = None,
    runtime_identity: Mapping[str, Any] | None = None,
) -> Capture:
    """A complete, admissible twelve-shard capture run under ``tmp_path``."""

    specs = list(primary_specs if primary_specs is not None else routable_specs())
    if len(specs) != scorer.PRIMARY_OWNER_COUNT_U:
        raise AssertionError("the primary fixture must carry exactly 26 owners")

    plan_dir, manifest = build_plan(tmp_path)
    identity = dict(runtime_identity or make_runtime_identity())
    admission_path, admission = build_admission(
        tmp_path, manifest=manifest, runtime_identity=identity
    )

    timing_ids = timing_control_owner_ids()
    tp_ids = tp_replay_control_owner_ids()
    shard_root = tmp_path / "shards"
    capture = Capture(
        root=tmp_path,
        plan_dir=plan_dir,
        manifest=manifest,
        admission_path=admission_path,
        admission=admission,
        runtime_identity=identity,
        shard_root=shard_root,
    )

    cursor = 0
    for image_id in IMAGE_IDS:
        shard_id = f"capture-{image_id}"
        count = FROZEN_PER_IMAGE_PRIMARY[image_id]
        records: list[dict[str, Any]] = []
        for offset in range(count):
            spec = specs[cursor + offset]
            records.append(
                make_primary_owner_record(
                    spec, owner_id=f"gt:{image_id}:p{offset}", shard_id=shard_id
                )
            )
        cursor += count
        for owner_id in (value for value in timing_ids if _image_of(value) == image_id):
            records.append(
                make_control_owner_record(
                    owner_id, cohort=merge.TIMING_CONTROL_COHORT, shard_id=shard_id
                )
            )
        for owner_id in (value for value in tp_ids if _image_of(value) == image_id):
            records.append(
                make_control_owner_record(
                    owner_id, cohort=merge.TP_REPLAY_CONTROL_COHORT, shard_id=shard_id
                )
            )
        rows = [
            make_score_row(
                image_id=image_id,
                shard_id=shard_id,
                owner_id=str(record["gt_owner_id"]),
                index=index,
            )
            for index, record in enumerate(records)
        ]
        receipt = make_shard_receipt(
            shard_id=shard_id,
            image_id=image_id,
            plan_dir=plan_dir,
            manifest=manifest,
            runtime_identity=identity,
            admission=admission,
            owner_records=records,
            score_rows=rows,
        )
        capture.shard_dirs[image_id] = write_shard(
            shard_root / image_id,
            receipt=receipt,
            owner_records=records,
            score_rows=rows,
            parity=make_parity(shard_id=shard_id, image_id=image_id, admission=admission),
        )
    if cursor != scorer.PRIMARY_OWNER_COUNT_U:
        raise AssertionError("the per-image primary counts must consume all 26 specs")
    return capture


@pytest.fixture()
def capture(tmp_path: Path) -> Capture:
    return build_capture(tmp_path)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_merge_publishes_a_complete_self_sealed_family(capture: Capture, tmp_path: Path) -> None:
    result = capture.run_merge()
    receipt = result["receipt"]

    assert receipt["schema_version"] == merge.MERGE_SCHEMA_VERSION
    assert receipt["unit_id"] == merge.UNIT_ID
    assert receipt["scorer_source_sha256"] == merge.FROZEN_SCORER_SOURCE_SHA256
    merge.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="merge receipt"
    )

    assert len(receipt["shards"]) == merge.EXPECTED_SHARD_COUNT
    assert sorted(shard["session_image_id"] for shard in receipt["shards"]) == sorted(
        merge.FROZEN_IMAGE_IDS
    )
    observed = receipt["observed_cohorts"]
    assert observed["primary_owner_count"] == 26
    assert observed["matched_e_count"] == 12
    assert observed["unmatched_e_count"] == 14
    assert observed["timing_control_owner_count"] == 14
    assert observed["tp_replay_control_owner_count"] == 12
    assert receipt["cohort_denominators"]["same_context_ul_count"] == 24
    assert receipt["policy"]["branch_assignment_performed"] is False

    published = merge.publish_merge(tmp_path / "merged", result["files"])
    assert published["published"] is True
    for name, entry in receipt["output_file_digests"].items():
        payload = (tmp_path / "merged" / name).read_bytes()
        assert scorer.sha256_bytes(payload) == entry["sha256"]
        assert len(payload) == entry["byte_size"]


def test_merge_preserves_every_raw_row_and_record_verbatim(capture: Capture) -> None:
    result = capture.run_merge()
    merged_records = {
        str(record["gt_owner_id"]): record for record in result["owner_records"]
    }
    shard_records = {
        str(record["gt_owner_id"]): record
        for image_id in IMAGE_IDS
        for record in capture.read_shard(image_id)["owner_records"]
    }
    assert merged_records == shard_records
    assert len(result["score_rows"]) == sum(
        len(capture.read_shard(image_id)["score_rows"]) for image_id in IMAGE_IDS
    )
    assert all(
        record["branch_assignment"] == merge.BRANCH_ASSIGNMENT_SENTINEL
        for record in result["owner_records"]
    )
    assert all(
        record["secondary_compatibility"] == merge.SECONDARY_COMPATIBILITY_SENTINEL
        for record in result["owner_records"]
    )


def test_merge_is_deterministic_and_idempotent(capture: Capture, tmp_path: Path) -> None:
    first = capture.run_merge()
    second = capture.run_merge()
    assert first["files"] == second["files"]
    assert (
        first["receipt"]["receipt_content_sha256"]
        == second["receipt"]["receipt_content_sha256"]
    )

    output_dir = tmp_path / "merged"
    assert merge.publish_merge(output_dir, first["files"])["published"] is True
    rerun = merge.publish_merge(output_dir, second["files"])
    assert rerun["published"] is False
    assert rerun["publish_mode"] == "no_op_identical_rerun"


def test_merge_shard_order_does_not_change_the_bytes(capture: Capture) -> None:
    forward = capture.run_merge()
    reversed_dirs = [capture.shard_dirs[image_id] for image_id in reversed(IMAGE_IDS)]
    backward = merge.merge_shards(**capture.merge_kwargs(shard_dirs=reversed_dirs))
    assert forward["files"] == backward["files"]


def test_publish_refuses_a_drifted_existing_directory(capture: Capture, tmp_path: Path) -> None:
    result = capture.run_merge()
    output_dir = tmp_path / "merged"
    merge.publish_merge(output_dir, result["files"])
    (output_dir / merge.MERGED_PARITY_NAME).write_bytes(b"{}\n")
    with pytest.raises(merge.MergeContractError, match="not a byte-identical merge"):
        merge.publish_merge(output_dir, result["files"])


# ---------------------------------------------------------------------------
# Image closure: missing, duplicate, foreign
# ---------------------------------------------------------------------------


def test_merge_refuses_a_missing_image(capture: Capture) -> None:
    partial = [capture.shard_dirs[image_id] for image_id in IMAGE_IDS[1:]]
    with pytest.raises(merge.MergeContractError, match="no admitted shard was supplied"):
        merge.merge_shards(**capture.merge_kwargs(shard_dirs=partial))


def test_merge_refuses_two_shards_for_one_image(capture: Capture, tmp_path: Path) -> None:
    twin = tmp_path / "twin" / IMAGE_IDS[0]
    source = capture.shard_dirs[IMAGE_IDS[0]]
    twin.mkdir(parents=True)
    for name in merge.REQUIRED_SHARD_FILES:
        (twin / name).write_bytes((source / name).read_bytes())
    shard_dirs = [capture.shard_dirs[image_id] for image_id in IMAGE_IDS] + [twin]
    with pytest.raises(merge.MergeContractError, match="more than one shard"):
        merge.merge_shards(**capture.merge_kwargs(shard_dirs=shard_dirs))


def test_merge_refuses_the_same_shard_directory_twice(capture: Capture) -> None:
    shard_dirs = [capture.shard_dirs[image_id] for image_id in IMAGE_IDS]
    shard_dirs.append(shard_dirs[0])
    with pytest.raises(merge.MergeContractError, match="was supplied twice"):
        merge.merge_shards(**capture.merge_kwargs(shard_dirs=shard_dirs))


def test_merge_refuses_an_image_outside_the_frozen_twelve(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    records = capture.read_shard(image_id)["owner_records"]
    receipt = json.loads((capture.shard_dirs[image_id] / scorer.RECEIPT_NAME).read_text())
    executed = dict(receipt["executed"])
    executed["session_image_id"] = "999999"
    capture.rewrite_shard(
        image_id, owner_records=records, receipt_patch={"executed": executed}
    )
    with pytest.raises(merge.MergeContractError, match="outside this unit's\n?\\s*frozen twelve"):
        capture.run_merge()


def test_merge_refuses_a_wrong_per_image_primary_count(capture: Capture) -> None:
    image_id = "16228"  # the plan seals five primary owners for this image
    records = [
        record
        for record in capture.read_shard(image_id)["owner_records"]
        if record["cohort"] != merge.PRIMARY_COHORT
        or record["gt_owner_id"] != f"gt:{image_id}:p4"
    ]
    capture.rewrite_shard(image_id, owner_records=records)
    with pytest.raises(merge.MergeContractError, match="not the sealed plan's 5"):
        capture.run_merge()


# ---------------------------------------------------------------------------
# Quarantine
# ---------------------------------------------------------------------------


def test_merge_refuses_a_quarantine_stopped_shard_directory(capture: Capture) -> None:
    shard_dir = capture.shard_dirs[IMAGE_IDS[0]]
    (shard_dir / scorer.QUARANTINE_NAME).write_bytes(b"{}\n")
    with pytest.raises(merge.MergeContractError, match="stopped by the quarantine rule"):
        capture.run_merge()


def test_merge_refuses_a_receipt_that_declares_it_stopped(capture: Capture) -> None:
    capture.rewrite_shard(IMAGE_IDS[0], stopped=True)
    with pytest.raises(merge.MergeContractError, match="stopped by the quarantine rule"):
        capture.run_merge()


def test_merge_admits_up_to_two_quarantined_primary_owners(capture: Capture) -> None:
    specs = routable_specs()
    specs[0] = replace(specs[0], quarantined=True)
    specs[1] = replace(specs[1], quarantined=True)
    quarantined = build_capture(capture.root / "two", primary_specs=specs)
    receipt = quarantined.run_merge()["receipt"]
    assert len(receipt["observed_cohorts"]["quarantined_primary_owner_ids"]) == 2


def test_merge_refuses_more_than_two_quarantined_primary_owners(capture: Capture) -> None:
    specs = [replace(spec, quarantined=index < 3) for index, spec in enumerate(routable_specs())]
    stopped = build_capture(capture.root / "three", primary_specs=specs)
    with pytest.raises(merge.MergeContractError, match="stops this unit without interpretation"):
        stopped.run_merge()


# ---------------------------------------------------------------------------
# Mixed runtime, admission, source, plan
# ---------------------------------------------------------------------------


def test_merge_refuses_a_mixed_runtime_identity(capture: Capture) -> None:
    other = make_runtime_identity(marker="a-different-checkpoint")
    other_admission_path, other_admission = build_admission(
        capture.root / "other-admission", manifest=capture.manifest, runtime_identity=other
    )
    assert other_admission_path.is_file()
    capture.rewrite_shard(IMAGE_IDS[0], runtime_identity=other, admission=other_admission)
    with pytest.raises(merge.MergeContractError, match="inherited admission"):
        capture.run_merge()


def test_merge_refuses_a_runtime_the_admission_never_admitted(capture: Capture) -> None:
    drifted = make_runtime_identity(marker="drifted")
    patched_admission = dict(capture.admission)
    # Same admission digest, different runtime: the shard's own receipt still
    # reconstructs, so only the admission cross-check can catch this.
    capture.rewrite_shard(
        IMAGE_IDS[0], runtime_identity=drifted, admission=patched_admission
    )
    with pytest.raises(merge.MergeContractError, match="runtime the smoke admission never"):
        capture.run_merge()


def test_merge_refuses_a_simulated_backend(capture: Capture) -> None:
    fake = make_runtime_identity()
    fake["is_real_model"] = False
    fake["usable_as_evidence"] = False
    fake["backend"] = "fake"
    fake_admission_path, fake_admission = build_admission(
        capture.root / "fake-admission", manifest=capture.manifest, runtime_identity=fake
    )
    assert fake_admission_path.is_file()
    for image_id in IMAGE_IDS:
        capture.rewrite_shard(image_id, runtime_identity=fake, admission=fake_admission)
    with pytest.raises(merge.MergeContractError, match="not a real model"):
        merge.merge_shards(**capture.merge_kwargs(admission_path=fake_admission_path))


def test_merge_refuses_a_foreign_scorer_source(capture: Capture) -> None:
    capture.rewrite_shard(
        IMAGE_IDS[0], receipt_patch={"scorer_source_sha256": "ff" * 32}
    )
    with pytest.raises(merge.MergeContractError, match="not this unit's frozen"):
        capture.run_merge()


def test_merge_refuses_a_foreign_scorer_source_identity(capture: Capture) -> None:
    drifted = make_runtime_identity()
    drifted["source_identity"] = {
        **drifted["source_identity"],
        merge.SCORER_SOURCE_IDENTITY_KEY: "ee" * 32,
    }
    drifted_admission_path, drifted_admission = build_admission(
        capture.root / "drifted-source", manifest=capture.manifest, runtime_identity=drifted
    )
    for image_id in IMAGE_IDS:
        capture.rewrite_shard(image_id, runtime_identity=drifted, admission=drifted_admission)
    with pytest.raises(merge.MergeContractError, match="scorer source identity"):
        merge.merge_shards(**capture.merge_kwargs(admission_path=drifted_admission_path))


def test_merge_refuses_a_plan_from_another_revision(capture: Capture, tmp_path: Path) -> None:
    other_plan_dir, other_manifest = build_plan(
        tmp_path / "other-plan", marker="another-revision"
    )
    # The admission is re-sealed against the other plan so the refusal has to
    # come from the shards' own plan binding, not from the admission check.
    other_admission_path, _ = build_admission(
        tmp_path / "other-plan-admission",
        manifest=other_manifest,
        runtime_identity=capture.runtime_identity,
    )
    with pytest.raises(merge.MergeContractError, match="captured against plan manifest"):
        merge.merge_shards(
            **capture.merge_kwargs(
                plan_dir=other_plan_dir, admission_path=other_admission_path
            )
        )


def test_merge_refuses_a_tampered_plan_file(capture: Capture) -> None:
    target = capture.plan_dir / merge.PLAN_FILE_NAMES[0]
    target.write_bytes(target.read_bytes() + b"{}\n")
    with pytest.raises(merge.MergeContractError, match="does not match the digest its own"):
        capture.run_merge()


def test_merge_refuses_an_admission_sealed_against_another_plan(
    capture: Capture, tmp_path: Path
) -> None:
    other_plan_dir, other_manifest = build_plan(tmp_path / "other-plan", marker="another-revision")
    assert other_plan_dir.is_dir()
    other_path, _ = build_admission(
        tmp_path / "other-admission",
        manifest=other_manifest,
        runtime_identity=capture.runtime_identity,
    )
    with pytest.raises(merge.MergeContractError, match="sealed against a different plan"):
        merge.merge_shards(**capture.merge_kwargs(admission_path=other_path))


# ---------------------------------------------------------------------------
# Tampering, incompleteness, unknown artifacts
# ---------------------------------------------------------------------------


def test_merge_refuses_an_edited_receipt(capture: Capture) -> None:
    shard_dir = capture.shard_dirs[IMAGE_IDS[0]]
    receipt = json.loads((shard_dir / scorer.RECEIPT_NAME).read_text())
    receipt["shard_id"] = "renamed-after-sealing"
    (shard_dir / scorer.RECEIPT_NAME).write_bytes(
        scorer.canonical_json_bytes(receipt) + b"\n"
    )
    with pytest.raises(merge.MergeContractError, match="does not reconstruct its own"):
        capture.run_merge()


def test_merge_refuses_an_edited_plan_manifest(capture: Capture) -> None:
    manifest = json.loads((capture.plan_dir / "manifest.json").read_text())
    manifest["cohort_counts"]["u_bound_crossing_count"] = 27
    (capture.plan_dir / "manifest.json").write_bytes(
        scorer.canonical_json_bytes(manifest) + b"\n"
    )
    with pytest.raises(merge.MergeContractError, match="does not reconstruct its own"):
        capture.run_merge()


def test_merge_refuses_an_incomplete_shard(capture: Capture) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / scorer.OWNER_RECORDS_NAME).unlink()
    with pytest.raises(merge.MergeContractError, match="is incomplete; missing"):
        capture.run_merge()


def test_merge_refuses_an_unknown_artifact_inside_a_shard(capture: Capture) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / "notes.txt").write_text("hand edited\n")
    with pytest.raises(merge.MergeContractError, match="unknown artifact"):
        capture.run_merge()


def test_merge_refuses_a_withheld_owner_record(capture: Capture) -> None:
    image_id = "16228"
    shard_dir = capture.shard_dirs[image_id]
    records = capture.read_shard(image_id)["owner_records"][:-1]
    (shard_dir / scorer.OWNER_RECORDS_NAME).write_bytes(
        b"".join(scorer.canonical_json_bytes(row) + b"\n" for row in records)
    )
    with pytest.raises(merge.MergeContractError, match="but its receipt sealed"):
        capture.run_merge()


def test_merge_refuses_a_duplicated_owner_record(capture: Capture) -> None:
    image_id = "16228"
    records = capture.read_shard(image_id)["owner_records"]
    duplicated = [*records[:-1], dict(records[0])]
    shard_dir = capture.shard_dirs[image_id]
    (shard_dir / scorer.OWNER_RECORDS_NAME).write_bytes(
        b"".join(scorer.canonical_json_bytes(row) + b"\n" for row in duplicated)
    )
    with pytest.raises(merge.MergeContractError, match="do not match the set its receipt"):
        capture.run_merge()


def test_merge_refuses_a_swapped_request_id_set(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_shard(image_id)["score_rows"]
    rows[0] = {**rows[0], "request_id": "req:rewritten"}
    shard_dir = capture.shard_dirs[image_id]
    (shard_dir / scorer.SCORES_NAME).write_bytes(
        b"".join(scorer.canonical_json_bytes(row) + b"\n" for row in rows)
    )
    with pytest.raises(merge.MergeContractError, match="do not reproduce the\n?\\s*digest"):
        capture.run_merge()


def test_merge_refuses_a_deferred_secondary_row(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_shard(image_id)["score_rows"]
    rows.append(
        {
            **rows[0],
            "request_id": "req:secondary",
            "request_family": scorer.REQUEST_DOWNSTREAM_COMPATIBILITY,
            "readout_tier": "secondary",
        }
    )
    capture.rewrite_shard(image_id, score_rows=rows)
    with pytest.raises(merge.MergeContractError, match="this merge is primary-only"):
        capture.run_merge()


def test_merge_refuses_a_record_that_already_carries_a_branch(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    records = capture.read_shard(image_id)["owner_records"]
    records[0] = {**records[0], "branch_assignment": "displaced"}
    capture.rewrite_shard(image_id, owner_records=records)
    with pytest.raises(merge.MergeContractError, match="carries a branch assignment"):
        capture.run_merge()


def test_merge_refuses_a_record_that_carries_secondary_evidence(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    records = capture.read_shard(image_id)["owner_records"]
    records[0] = {
        **records[0],
        "secondary_compatibility": {"p_plus_c_to_e": {"description_delta": -0.5}},
    }
    capture.rewrite_shard(image_id, owner_records=records)
    with pytest.raises(merge.MergeContractError, match="carries secondary compatibility"):
        capture.run_merge()


def test_merge_refuses_a_record_from_another_image_session(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    records = capture.read_shard(image_id)["owner_records"]
    records[0] = {**records[0], "image_id": "99999"}
    capture.rewrite_shard(image_id, owner_records=records)
    with pytest.raises(merge.MergeContractError, match="one image session per shard"):
        capture.run_merge()


def test_merge_refuses_a_primary_record_missing_a_frozen_ladder(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    records = capture.read_shard(image_id)["owner_records"]
    primary = next(row for row in records if row["cohort"] == merge.PRIMARY_COHORT)
    primary["ladders"] = {"at_p_plus_e": primary["ladders"]["at_p_plus_e"]}
    capture.rewrite_shard(image_id, owner_records=records)
    with pytest.raises(merge.MergeContractError, match="not the frozen"):
        capture.run_merge()


def test_merge_refuses_a_parity_file_bound_to_another_admission(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    parity = make_parity(
        shard_id=f"capture-{image_id}", image_id=image_id, admission=capture.admission
    )
    parity["inherited_admission"] = {
        **parity["inherited_admission"],
        "admission_content_sha256": "cc" * 32,
    }
    capture.rewrite_shard(image_id, parity=parity)
    with pytest.raises(merge.MergeContractError, match="disagrees with itself"):
        capture.run_merge()


def test_merge_refuses_a_control_cohort_that_leaves_the_sealed_registry(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    records = capture.read_shard(image_id)["owner_records"]
    control = next(
        row for row in records if row["cohort"] == merge.TIMING_CONTROL_COHORT
    )
    control["gt_owner_id"] = f"gt:{image_id}:t9"
    capture.rewrite_shard(image_id, owner_records=records)
    with pytest.raises(merge.MergeContractError, match="sealed control registry"):
        capture.run_merge()


def test_discover_shard_dirs_finds_every_capture_directory(capture: Capture) -> None:
    found = merge.discover_shard_dirs(capture.shard_root)
    assert sorted(path.name for path in found) == sorted(IMAGE_IDS)
