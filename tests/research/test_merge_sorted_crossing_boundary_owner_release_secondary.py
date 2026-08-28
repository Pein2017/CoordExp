"""Focused tests for the immutable secondary downstream-compatibility merge of
the sorted crossing-boundary owner release/realization capture shards.

Fixture policy
--------------
The fixture is a *complete* sealed run: all twelve frozen images, the sealed
plan's real per-image primary owner distribution, 26 crossing owners, 14
disjoint timing controls, 12 native-TP replay controls and exactly the frozen
64 secondary requests (26 / 26 / 12).  The frozen denominators are the contract
under test, so they are never shrunk or monkeypatched here.

Every root payload, per-segment sum and paired delta below is produced by the
*producer's own* ``_root_payload`` / ``row_segments`` / ``segment_sums`` /
``paired_deltas`` helpers from real :class:`ScoredToken` objects, and every
receipt, manifest and admission is sealed with the producer's own
``sha256_json`` canonicalization -- so a schema drift in the capture surface
breaks these tests instead of silently passing a stale hand-written dict.

``build_secondary_capture`` and its helpers live in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import dataclass, field
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from scripts.research import (
    analyze_sorted_crossing_boundary_owner_release as primary_analyzer,
)
from scripts.research import merge_sorted_crossing_boundary_owner_release as primary_merge
from scripts.research import (
    merge_sorted_crossing_boundary_owner_release_secondary as sut,
)
from scripts.research import (
    prepare_sorted_crossing_boundary_owner_release_realization as plan_builder,
)
from scripts.research import score_sorted_crossing_boundary_owner_release as primary
from scripts.research import (
    score_sorted_crossing_boundary_owner_release_secondary as secondary,
)


def _load_primary_merge_fixtures():
    """Import the primary merge test module by path so its frozen constants are
    reused rather than copied a third time.

    ``tests/research`` is not a package, so a plain ``from tests.research...``
    import is not available; the module is registered in ``sys.modules`` before
    execution because its dataclasses need to resolve their own module.
    """

    name = "crossing_boundary_merge_fixtures"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name(
        "test_merge_sorted_crossing_boundary_owner_release.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_primary_fx = _load_primary_merge_fixtures()
FROZEN_PER_IMAGE_PRIMARY: Mapping[str, int] = _primary_fx.FROZEN_PER_IMAGE_PRIMARY
IMAGE_IDS: tuple[str, ...] = _primary_fx.IMAGE_IDS
timing_control_owner_ids = _primary_fx.timing_control_owner_ids

# ---------------------------------------------------------------------------
# Frozen fixture constants
# ---------------------------------------------------------------------------

SMOKE_IMAGE_ID = "4134"
PLAN_BUILDER_SHA256 = "ab" * 32
SECONDARY_SCORER_SHA256 = "5e" * 32
PRIMARY_SCORER_SHA256 = primary_merge.FROZEN_SCORER_SOURCE_SHA256
ANALYZER_SHA256 = "9a" * 32

OBJECT_REF_START = secondary.OBJECT_REF_START
OBJECT_REF_END = secondary.OBJECT_REF_END
BOX_START = secondary.BOX_START
BOX_END = secondary.BOX_END
COORD_START = secondary.COORDINATE_TOKEN_ID_START
COORD_SPAN = (
    secondary.COORDINATE_TOKEN_ID_END_EXCLUSIVE - secondary.COORDINATE_TOKEN_ID_START
)

#: The paired-root offsets the fixture walks, so the merged evidence carries
#: strictly positive, exactly zero and strictly negative segment deltas.
DELTA_OFFSETS: tuple[float, ...] = (0.5, -0.75, 0.0, 0.125, -0.25)


def _sha256_json(value: Any) -> str:
    return secondary.sha256_json(value)


def _seal(payload: dict[str, Any], key: str) -> dict[str, Any]:
    payload[key] = _sha256_json({k: v for k, v in payload.items() if k != key})
    return payload


def _coords(seed: int) -> list[int]:
    return [COORD_START + (seed * (index + 7) * 13) % COORD_SPAN for index in range(4)]


def _description(seed: int) -> list[int]:
    return [20000 + seed % 97] * (1 + seed % 3)


def _row_tokens(description: Sequence[int], coordinates: Sequence[int]) -> list[int]:
    return [
        OBJECT_REF_START,
        *description,
        OBJECT_REF_END,
        BOX_START,
        *coordinates,
        BOX_END,
    ]


def _context_id(image_id: str, boundary_index: int) -> str:
    return f"{image_id}:boundary-{boundary_index:03d}"


# ---------------------------------------------------------------------------
# 1. The sealed plan, on disk
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlanFixture:
    """A complete, self-sealed stand-in for the frozen CPU plan."""

    plan_dir: Path
    manifest: dict[str, Any]
    cohort_rows: list[dict[str, Any]]
    control_rows: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    context_tokens: dict[str, list[int]]
    prompt_tokens: dict[str, list[int]]
    registry_by_owner: dict[str, dict[str, Any]]

    def requests_for_image(self, image_id: str) -> list[dict[str, Any]]:
        return sorted(
            (row for row in self.requests if str(row["image_id"]) == image_id),
            key=lambda row: str(row["request_id"]),
        )

    def executed_prefix(self, context_id: str, image_id: str) -> list[int]:
        return [*self.prompt_tokens[image_id], *self.context_tokens[context_id]]


def _make_request(
    *,
    cohort: str,
    gt_owner_id: str,
    image_id: str,
    context_id: str,
    context_role: str,
    boundary_index: int,
    appended_token_ids: Sequence[int],
    base_prefix_token_ids: Sequence[int],
    scored_target: Mapping[str, Any],
    variant: str,
    optional: bool,
) -> dict[str, Any]:
    """Reproduce the sealed request identity the plan builder publishes."""

    base_digest = _sha256_json(list(base_prefix_token_ids))
    identity = {
        "unit_id": secondary.UNIT_ID,
        "request_family": secondary.REQUEST_FAMILY,
        "cohort": cohort,
        "gt_owner_id": gt_owner_id,
        "context_id": context_id,
        "variant": variant,
        "appended_token_ids": list(appended_token_ids),
        "base_prefix_token_ids_sha256": base_digest,
        "scored_target": dict(scored_target),
        "decode": None,
        "candidate_family": None,
    }
    digest = _sha256_json(identity)
    return {
        "schema_version": plan_builder.REQUEST_SCHEMA_VERSION,
        "row_kind": "crossing_boundary_request",
        "unit_id": secondary.UNIT_ID,
        "branch_schema_id": plan_builder.BRANCH_SCHEMA_ID,
        "branch_inputs": [],
        "boundary_index": boundary_index,
        "candidate_family": None,
        "cohort": cohort,
        "context_id": context_id,
        "context_role": context_role,
        "decode": None,
        "gt_owner_id": gt_owner_id,
        "identity_digest": digest,
        "image_id": image_id,
        "inspects_new_model_logits": False,
        "optional": optional,
        "predecessor_query_group_id": None,
        "prefix": {
            "appended_role": secondary.APPENDED_ROLE,
            "appended_token_count": len(appended_token_ids),
            "appended_token_ids": list(appended_token_ids),
            "appended_token_ids_sha256": _sha256_json(list(appended_token_ids)),
            "base_context_id": context_id,
            "base_prefix_token_count": len(base_prefix_token_ids),
            "base_prefix_token_ids_sha256": base_digest,
            "retokenized": False,
        },
        "readout_tier": secondary.SECONDARY_READOUT_TIER,
        "request_family": secondary.REQUEST_FAMILY,
        "request_id": f"req:{digest[:32]}",
        "request_key": (
            f"{secondary.REQUEST_FAMILY}|{cohort}|{gt_owner_id}|{context_id}|{variant}"
        ),
        "score_blind_plan": True,
        "scored_target": dict(scored_target),
        "variant": variant,
    }


def _sidecar(
    token_ids: Sequence[int], *, row_index: int, image_id: str
) -> dict[str, Any]:
    return {
        "full_row_token_ids": list(token_ids),
        "full_row_token_ids_sha256": _sha256_json(list(token_ids)),
        "full_row_token_count": len(token_ids),
        "full_row_token_source": (
            "literal_adjacent_context_prefix_suffix_tokens_post_boundary_minus_pre_boundary"
        ),
        "image_id": image_id,
        "row_index": row_index,
        "pre_row_context_id": _context_id(image_id, row_index),
        "post_row_context_id": _context_id(image_id, row_index + 1),
        "strict_match_status": "unmatched",
        "strict_match_gt_owner_id": None,
    }


def build_plan(tmp_path: Path, *, marker: str = "frozen") -> PlanFixture:
    """The whole sealed 26/26/12 secondary census, written to a plan directory.

    ``marker`` travels through the sealed lineage, so two plans built at
    different paths are genuinely different revisions rather than byte twins.
    """

    context_tokens: dict[str, list[int]] = {}
    prompt_tokens: dict[str, list[int]] = {}
    cohort_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    requests: list[dict[str, Any]] = []

    for image_index, image_id in enumerate(IMAGE_IDS):
        owner_count = FROZEN_PER_IMAGE_PRIMARY[image_id]
        prompt_tokens[image_id] = [900 + image_index, 901 + image_index, 902 + image_index]
        row_count = 3 * owner_count + 4
        rows = [
            _row_tokens(
                _description(image_index * 31 + row_index),
                _coords(image_index * 17 + row_index + 1),
            )
            for row_index in range(row_count)
        ]
        prefix: list[int] = []
        context_tokens[_context_id(image_id, 0)] = []
        for boundary_index, row in enumerate(rows, start=1):
            prefix = [*prefix, *row]
            context_tokens[_context_id(image_id, boundary_index)] = list(prefix)

        # --- the image's crossing owners --------------------------------
        for slot in range(owner_count):
            boundary = 2 + 3 * slot
            owner_id = f"gt:{image_id}:{boundary}"
            e_row = _sidecar(rows[boundary], row_index=boundary, image_id=image_id)
            f_row = _sidecar(rows[boundary + 1], row_index=boundary + 1, image_id=image_id)
            inserted = _row_tokens(
                _description(image_index * 31 + boundary),
                _coords(1000 + image_index * 7 + boundary),
            )
            cohort_rows.append(
                {
                    "schema_version": plan_builder.COHORT_SCHEMA_VERSION,
                    "row_kind": "crossing_cohort_owner",
                    "unit_id": secondary.UNIT_ID,
                    "cohort": plan_builder.PRIMARY_COHORT,
                    "gt_owner_id": owner_id,
                    "image_id": image_id,
                    "normalized_description": f"thing-{boundary}",
                    "crossing": {
                        "boundary_index_b": boundary,
                        "p_context_id": _context_id(image_id, boundary),
                        "pe_context_id": _context_id(image_id, boundary + 1),
                    },
                    "e_row": e_row,
                    "f_row": f_row,
                    "f_row_present": True,
                    "inserted_clean_row_c": {
                        "token_ids": list(inserted),
                        "token_ids_sha256": _sha256_json(list(inserted)),
                        "token_count": len(inserted),
                        "role": "oracle_intervention_never_a_natural_generation_claim",
                    },
                }
            )
            for variant, ctx_index, target_row, optional in (
                (secondary.VARIANT_P_PLUS_C_THEN_E, boundary, e_row, False),
                (secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F, boundary + 1, f_row, True),
            ):
                requests.append(
                    _make_request(
                        cohort=plan_builder.PRIMARY_COHORT,
                        gt_owner_id=owner_id,
                        image_id=image_id,
                        context_id=_context_id(image_id, ctx_index),
                        context_role="row_boundary",
                        boundary_index=ctx_index,
                        appended_token_ids=inserted,
                        base_prefix_token_ids=context_tokens[
                            _context_id(image_id, ctx_index)
                        ],
                        scored_target={
                            "kind": "exact_native_row",
                            "token_ids": list(target_row["full_row_token_ids"]),
                            "token_ids_sha256": target_row["full_row_token_ids_sha256"],
                            "native_row_index": target_row["row_index"],
                            "compare_against": (
                                "the same exact row scored at the unmodified native context"
                            ),
                            "report": [
                                "description_delta",
                                "coordinate_delta",
                                "complete_row_delta",
                            ],
                        },
                        variant=variant,
                        optional=optional,
                    )
                )

        # --- one native-TP replay control per image ----------------------
        control_owner = f"gt:{image_id}:c0"
        twin = _row_tokens(_description(image_index * 31), _coords(2000 + image_index))
        following = {
            "context_id": _context_id(image_id, 1),
            "kind": "native_row",
            "row_index": 1,
            "terminal_kind": None,
            "token_ids": list(rows[1]),
            "token_ids_sha256": _sha256_json(list(rows[1])),
        }
        control_rows.append(
            {
                "schema_version": plan_builder.CONTROL_SCHEMA_VERSION,
                "row_kind": "crossing_control_owner",
                "unit_id": secondary.UNIT_ID,
                "cohort": plan_builder.TP_REPLAY_CONTROL_COHORT,
                "gt_owner_id": control_owner,
                "image_id": image_id,
                "normalized_description": "tp",
                "due_context_id": _context_id(image_id, 0),
                "row_index": 0,
                "following_native_action": following,
                "inserted_clean_row_c": {
                    "token_ids": list(twin),
                    "token_ids_sha256": _sha256_json(list(twin)),
                    "token_count": len(twin),
                    "role": "oracle_intervention_never_a_natural_generation_claim",
                },
            }
        )
        requests.append(
            _make_request(
                cohort=plan_builder.TP_REPLAY_CONTROL_COHORT,
                gt_owner_id=control_owner,
                image_id=image_id,
                context_id=_context_id(image_id, 0),
                context_role="root",
                boundary_index=0,
                appended_token_ids=twin,
                base_prefix_token_ids=context_tokens[_context_id(image_id, 0)],
                scored_target={
                    "kind": "native_row",
                    "token_ids": list(rows[1]),
                    "token_ids_sha256": _sha256_json(list(rows[1])),
                    "native_row_index": 1,
                    "compare_against": (
                        "the same exact action scored at the unmodified native context"
                    ),
                    "report": [
                        "description_delta",
                        "coordinate_delta",
                        "complete_row_delta",
                    ],
                },
                variant=secondary.VARIANT_BENIGN_SUBSTITUTION,
                optional=False,
            )
        )

    # --- the 14 disjoint timing controls (no secondary request) -------------
    for owner_id in timing_control_owner_ids():
        control_rows.append(
            {
                "schema_version": plan_builder.CONTROL_SCHEMA_VERSION,
                "row_kind": "crossing_control_owner",
                "unit_id": secondary.UNIT_ID,
                "cohort": plan_builder.TIMING_CONTROL_COHORT,
                "gt_owner_id": owner_id,
                "image_id": owner_id.split(":")[1],
                "normalized_description": "timing",
            }
        )

    plan_dir = tmp_path / "plan"
    plan_dir.mkdir(parents=True)
    payloads = {
        plan_builder.COHORT_REGISTRY_NAME: cohort_rows,
        plan_builder.CONTROL_REGISTRY_NAME: control_rows,
        plan_builder.REQUEST_PLAN_NAME: requests,
    }
    digests: dict[str, Any] = {}
    for name, rows_payload in payloads.items():
        blob = b"".join(
            secondary.canonical_json_bytes(row) + b"\n" for row in rows_payload
        )
        (plan_dir / name).write_bytes(blob)
        digests[name] = {
            "path": name,
            "byte_size": len(blob),
            "row_count": len(rows_payload),
            "sha256": secondary.sha256_bytes(blob),
        }

    manifest: dict[str, Any] = {
        "schema_version": primary_merge.FROZEN_PLAN_MANIFEST_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "builder_source": {
            "path": (
                "scripts/research/"
                "prepare_sorted_crossing_boundary_owner_release_realization.py"
            ),
            "byte_size": 127581,
            "sha256": PLAN_BUILDER_SHA256,
        },
        "cohort_counts": {
            "u_bound_crossing_count": primary.PRIMARY_OWNER_COUNT_U,
            "l_bound_crossing_count": primary.PRIMARY_OWNER_COUNT_L,
            "exact_same_context_u_and_l_count": (
                primary.PRIMARY_OWNER_COUNT_SAME_CONTEXT_UL
            ),
            "matched_e_count": primary.MATCHED_E_OWNER_COUNT,
            "unmatched_e_count": primary.UNMATCHED_E_OWNER_COUNT,
            "f_row_present_count": primary.PRIMARY_OWNER_COUNT_U,
            "per_image_owner_counts": dict(FROZEN_PER_IMAGE_PRIMARY),
        },
        "control_counts": {
            "disjoint_from_primary_cohort": True,
            "timing_control_count": primary.TIMING_CONTROL_OWNER_COUNT,
            "timing_control_owner_ids": timing_control_owner_ids(),
            "tp_replay_control_count": primary.TP_CALIBRATION_OWNER_COUNT,
            "tp_replay_control_owner_ids": sorted(
                str(row["gt_owner_id"])
                for row in control_rows
                if str(row["cohort"]) == plan_builder.TP_REPLAY_CONTROL_COHORT
            ),
        },
        "lineage": {
            "census_unit_id": "2026-08-03-sorted-owner-accessibility-phenotype-census",
            "census_run_root": f"/frozen/{marker}",
        },
        "output_file_digests": digests,
    }
    _seal(manifest, "manifest_content_sha256")
    (plan_dir / plan_builder.MANIFEST_NAME).write_bytes(
        secondary.canonical_json_bytes(manifest) + b"\n"
    )

    return PlanFixture(
        plan_dir=plan_dir,
        manifest=manifest,
        cohort_rows=cohort_rows,
        control_rows=control_rows,
        requests=requests,
        context_tokens=context_tokens,
        prompt_tokens=prompt_tokens,
        registry_by_owner={
            str(row["gt_owner_id"]): row for row in (*cohort_rows, *control_rows)
        },
    )


# ---------------------------------------------------------------------------
# 2. The sealed primary analysis gate
# ---------------------------------------------------------------------------


def build_primary_analysis(
    tmp_path: Path, *, plan: PlanFixture, marker: str = "frozen"
) -> tuple[Path, dict[str, Any]]:
    """A sealed primary analysis directory plus the merged dir it names."""

    merged_dir = tmp_path / f"primary-merged-{marker}"
    merged_dir.mkdir(parents=True)
    merge_receipt: dict[str, Any] = {
        "schema_version": primary_merge.MERGE_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "plan": {"manifest_content_sha256": plan.manifest["manifest_content_sha256"]},
        "policy": {"secondary_compatibility_merged": False},
        "runtime_identity_sha256": "primary-runtime-digest",
    }
    _seal(merge_receipt, "receipt_content_sha256")
    (merged_dir / secondary.ANALYSIS_MERGE_RECEIPT_NAME).write_bytes(
        secondary.canonical_json_bytes(merge_receipt) + b"\n"
    )

    analysis_dir = tmp_path / f"primary-analysis-{marker}"
    analysis_dir.mkdir(parents=True)
    owner_rows = [
        {
            "schema_version": primary_analyzer.OWNER_ROW_SCHEMA_VERSION,
            "unit_id": secondary.UNIT_ID,
            "cohort": plan_builder.PRIMARY_COHORT,
            "gt_owner_id": str(row["gt_owner_id"]),
            "image_id": str(row["image_id"]),
            "primary_branch": primary.BRANCH_ORDER[index % len(primary.BRANCH_ORDER)],
        }
        for index, row in enumerate(plan.cohort_rows)
    ]
    owner_rows_bytes = b"".join(
        secondary.canonical_json_bytes(row) + b"\n" for row in owner_rows
    )
    report = {
        "schema_version": primary_analyzer.REPORT_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "decision": primary_analyzer.DECISION_ROUTE,
    }
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    report_md_bytes = b"# primary analysis\n"
    (analysis_dir / secondary.ANALYSIS_OWNER_ROWS_NAME).write_bytes(owner_rows_bytes)
    (analysis_dir / secondary.ANALYSIS_REPORT_JSON_NAME).write_bytes(report_bytes)
    (analysis_dir / secondary.ANALYSIS_REPORT_MD_NAME).write_bytes(report_md_bytes)

    receipt: dict[str, Any] = {
        "schema_version": primary_analyzer.RECEIPT_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "analyzer_source_sha256": ANALYZER_SHA256,
        "scorer_source_sha256": PRIMARY_SCORER_SHA256,
        "merger_source_sha256": "6a" * 32,
        "merged_dir": str(merged_dir),
        "merge_receipt_content_sha256": merge_receipt["receipt_content_sha256"],
        "runtime_identity_sha256": "primary-runtime-digest",
        "input_file_sha256": {
            secondary.ANALYSIS_MERGE_RECEIPT_NAME: secondary.sha256_file(
                merged_dir / secondary.ANALYSIS_MERGE_RECEIPT_NAME
            )
        },
        "decision": primary_analyzer.DECISION_ROUTE,
        "primary_denominator": primary.PRIMARY_OWNER_COUNT_U,
        "policy": {"secondary_compatibility_read": False, "threshold_fitting": False},
        "output_file_digests": {
            secondary.ANALYSIS_OWNER_ROWS_NAME: {
                "path": secondary.ANALYSIS_OWNER_ROWS_NAME,
                "byte_size": len(owner_rows_bytes),
                "row_count": len(owner_rows),
                "sha256": secondary.sha256_bytes(owner_rows_bytes),
            },
            secondary.ANALYSIS_REPORT_JSON_NAME: {
                "path": secondary.ANALYSIS_REPORT_JSON_NAME,
                "byte_size": len(report_bytes),
                "sha256": secondary.sha256_bytes(report_bytes),
            },
            secondary.ANALYSIS_REPORT_MD_NAME: {
                "path": secondary.ANALYSIS_REPORT_MD_NAME,
                "byte_size": len(report_md_bytes),
                "sha256": secondary.sha256_bytes(report_md_bytes),
            },
        },
    }
    _seal(receipt, "receipt_content_sha256")
    (analysis_dir / secondary.ANALYSIS_RECEIPT_NAME).write_bytes(
        secondary.canonical_json_bytes(receipt) + b"\n"
    )
    return analysis_dir, receipt


def build_analysis_binding(
    analysis_dir: Path, *, receipt: Mapping[str, Any], plan: PlanFixture
) -> dict[str, Any]:
    """The binding block a shard seals about the primary gate it ran under."""

    return {
        "unit_id": secondary.UNIT_ID,
        "analysis_dir": str(analysis_dir),
        "analysis_file_sha256": {
            name: secondary.sha256_file(analysis_dir / name)
            for name in sorted(secondary.ANALYSIS_REQUIRED_FILES)
        },
        "receipt_content_sha256": str(receipt["receipt_content_sha256"]),
        "analyzer_source_sha256": str(receipt["analyzer_source_sha256"]),
        "plan_manifest_content_sha256": str(plan.manifest["manifest_content_sha256"]),
        "secondary_fields_present": False,
        "branch_labels_used_to_select_requests": False,
    }


# ---------------------------------------------------------------------------
# 3. Runtime identity, admission and shard payloads
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
            "likelihood_channel": secondary.LIKELIHOOD_CHANNEL,
            "explicit_position_ids": True,
            "uses_model_generate": False,
            "repetition_penalty_stratum": secondary.NATIVE_REPETITION_PENALTY_STRATUM,
        },
        "source_identity": source_identity(),
        "vocab_size": 152680,
        "layer_count": 28,
    }


def source_identity(*, secondary_sha256: str = SECONDARY_SCORER_SHA256) -> dict[str, str]:
    digests = {
        name: f"{index:02x}" * 32
        for index, name in enumerate(sut.REQUIRED_SOURCE_IDENTITY_KEYS)
    }
    digests[sut.SECONDARY_SCORER_SOURCE_IDENTITY_KEY] = secondary_sha256
    digests[sut.PRIMARY_SCORER_SOURCE_IDENTITY_KEY] = PRIMARY_SCORER_SHA256
    digests[sut.PRIMARY_ANALYZER_SOURCE_IDENTITY_KEY] = ANALYZER_SHA256
    return dict(sorted(digests.items()))


def build_admission(
    tmp_path: Path,
    *,
    plan: PlanFixture,
    runtime_identity: Mapping[str, Any],
    analysis_binding_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    admission: dict[str, Any] = {
        "schema_version": secondary.ADMISSION_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "smoke_shard_id": f"secondary-smoke-{SMOKE_IMAGE_ID}",
        "smoke_image_id": SMOKE_IMAGE_ID,
        "plan_manifest_content_sha256": plan.manifest["manifest_content_sha256"],
        "primary_analysis_binding_sha256": analysis_binding_sha256,
        "admission_identity_fields": list(primary.ADMISSION_IDENTITY_FIELDS),
        "smoke_variants": secondary.counts_by_variant(
            plan.requests_for_image(SMOKE_IMAGE_ID)
        ),
        "cache_admitted": True,
        "root_backend": {root: secondary.KV_CACHE_BACKEND for root in sut.PAIRED_ROOTS},
        "max_selected_logit_abs_diff": 4.2e-05,
        "mismatched_fields": [],
        "replay_admission_enforced": True,
        "scope": "one image session carried every sealed secondary variant",
    }
    admission.update(
        {key: runtime_identity[key] for key in primary.ADMISSION_IDENTITY_FIELDS}
    )
    admission["runtime_identity_sha256"] = primary.runtime_identity_digest(
        runtime_identity
    )
    _seal(admission, "admission_content_sha256")
    path = tmp_path / "secondary-smoke" / secondary.ADMISSION_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(secondary.canonical_json_bytes(admission) + b"\n")
    return path, admission


def _scored_tokens(
    token_ids: Sequence[int],
    *,
    base: float,
    offset: float,
    diverge_at: int | None,
) -> list[primary.ScoredToken]:
    """One root's teacher-forced stream, as real :class:`ScoredToken` objects."""

    scored: list[primary.ScoredToken] = []
    for index, token_id in enumerate(token_ids):
        selected = round(base - 0.25 * index + offset, 6)
        if diverge_at is not None and index == diverge_at:
            argmax_id = int(token_id) + 1
            argmax_logprob = round(selected + 0.5, 6)
        else:
            argmax_id = int(token_id)
            argmax_logprob = selected
        scored.append(
            primary.ScoredToken(
                token_id=int(token_id),
                selected_logprob=selected,
                argmax_token_id=argmax_id,
                argmax_logprob=argmax_logprob,
            )
        )
    return scored


def build_secondary_row(
    *,
    request: Mapping[str, Any],
    plan: PlanFixture,
    shard_id: str,
    analysis_binding_sha256: str,
    index: int,
) -> dict[str, Any]:
    """One published secondary row, built through the producer's own helpers."""

    variant = str(request["variant"])
    gt_owner_id = str(request["gt_owner_id"])
    image_id = str(request["image_id"])
    registry_row = plan.registry_by_owner[gt_owner_id]
    tokens = [int(value) for value in request["scored_target"]["token_ids"]]
    segments = secondary.row_segments(tokens)
    appended = [int(value) for value in request["prefix"]["appended_token_ids"]]
    appended_sha256 = str(request["prefix"]["appended_token_ids_sha256"])

    modified_context_id = str(request["context_id"])
    baseline_context_id, baseline_source = secondary._sealed_baseline_context(  # noqa: SLF001
        registry_row, variant=variant, gt_owner_id=gt_owner_id
    )
    native_row_index = int(request["scored_target"]["native_row_index"])
    successor_context_id = _context_id(image_id, native_row_index + 1)

    offset = DELTA_OFFSETS[index % len(DELTA_OFFSETS)]
    # Any divergence stays *after* the description path, so the baseline root's
    # native replay is admitted exactly as a published shard requires.
    diverge_at = segments.complete_row[1] - 1 if index % 3 == 0 else None

    payloads: dict[str, dict[str, Any]] = {}
    for root, context_id, root_appended, root_offset in (
        (sut.ROOT_BASELINE, baseline_context_id, [], 0.0),
        (sut.ROOT_MODIFIED, modified_context_id, appended, offset),
    ):
        scored = _scored_tokens(
            tokens, base=-1.5 - 0.1 * index, offset=root_offset, diverge_at=diverge_at
        )
        executed_prefix = plan.executed_prefix(context_id, image_id)
        group_id = secondary._root_group_id(  # noqa: SLF001
            gt_owner_id=gt_owner_id,
            context_id=context_id,
            variant=variant,
            root=root,
            appended_digest=_sha256_json(list(root_appended)),
            scoring_backend=secondary.KV_CACHE_BACKEND,
        )
        payloads[root] = secondary._root_payload(  # noqa: SLF001
            root=root,
            context_id=context_id,
            executed_prefix_token_ids=executed_prefix,
            appended_token_ids=root_appended,
            context_group_id=group_id,
            scoring_backend=secondary.KV_CACHE_BACKEND,
            scored=scored,
            segments=segments,
            target_token_ids=tokens,
        )
    deltas = secondary.paired_deltas(
        payloads[sut.ROOT_BASELINE]["segment_sums"],
        payloads[sut.ROOT_MODIFIED]["segment_sums"],
    )
    return {
        "schema_version": secondary.SCHEMA_VERSION,
        "row_kind": "secondary_compatibility_row",
        "unit_id": secondary.UNIT_ID,
        "shard_id": shard_id,
        "request_id": str(request["request_id"]),
        "request_key": str(request["request_key"]),
        "request_family": secondary.REQUEST_FAMILY,
        "readout_tier": secondary.SECONDARY_READOUT_TIER,
        "cohort": str(request["cohort"]),
        "variant": variant,
        "plan_optional": bool(request["optional"]),
        "gt_owner_id": gt_owner_id,
        "image_id": image_id,
        "session_image_id": image_id,
        "plan_identity_digest": str(request["identity_digest"]),
        "primary_analysis_binding_sha256": analysis_binding_sha256,
        "native_row_index": native_row_index,
        "scored_target_kind": str(request["scored_target"]["kind"]),
        "baseline_context_id": baseline_context_id,
        "modified_context_id": modified_context_id,
        "successor_context_id": successor_context_id,
        "baseline_context_source": baseline_source,
        "inserted_clean_row_c_token_ids_sha256": appended_sha256,
        "inserted_clean_row_c_token_count": len(appended),
        "scored_token_ids": list(tokens),
        "scored_token_ids_sha256": _sha256_json(list(tokens)),
        "scored_token_count": len(tokens),
        "segments": {
            segment: {
                "token_index_start": segments.span(segment)[0],
                "token_index_stop": segments.span(segment)[1],
                "token_ids": tokens[segments.span(segment)[0] : segments.span(segment)[1]],
            }
            for segment in sut.SEGMENTS
        },
        "roots": payloads,
        "deltas": deltas,
        "baseline_replay_admitted": True,
        "replay_admission_enforced": True,
        "likelihood_channel": secondary.LIKELIHOOD_CHANNEL,
        "repetition_penalty_stratum": float(secondary.NATIVE_REPETITION_PENALTY_STRATUM),
        "uses_model_generate": False,
        "sampling": "disabled_secondary_deterministic_teacher_forcing_only",
        "retokenized": False,
        "delta_orientation": "modified_minus_baseline",
        "claim_boundary": secondary.CLAIM_BOUNDARY,
    }


def make_shard_receipt(
    *,
    shard_id: str,
    image_id: str,
    plan: PlanFixture,
    runtime_identity: Mapping[str, Any],
    admission: Mapping[str, Any],
    analysis_binding: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    expected_requests: Sequence[Mapping[str, Any]],
    quarantine_entries: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    expected_counts = secondary.counts_by_variant(expected_requests)
    all_secondary = [
        row
        for row in plan.requests
        if str(row["readout_tier"]) == secondary.SECONDARY_READOUT_TIER
    ]
    receipt: dict[str, Any] = {
        "schema_version": secondary.RECEIPT_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "mode": secondary.MODE_CAPTURE,
        "shard_id": shard_id,
        "scorer_source_sha256": str(
            runtime_identity["source_identity"][sut.SECONDARY_SCORER_SOURCE_IDENTITY_KEY]
        ),
        "source_identity": dict(runtime_identity["source_identity"]),
        "plan": {
            "plan_dir": str(plan.plan_dir),
            "manifest_schema_version": (
                primary_merge.FROZEN_PLAN_MANIFEST_SCHEMA_VERSION
            ),
            "manifest_content_sha256": plan.manifest["manifest_content_sha256"],
            "builder_source_sha256": PLAN_BUILDER_SHA256,
            "plan_file_sha256": {
                name: str(plan.manifest["output_file_digests"][name]["sha256"])
                for name in sut.PLAN_FILE_NAMES
            },
            "lineage": plan.manifest["lineage"],
            "cohort_counts": plan.manifest["cohort_counts"],
            "control_counts": plan.manifest["control_counts"],
        },
        "primary_analysis": dict(analysis_binding),
        "primary_analysis_binding_sha256": _sha256_json(dict(analysis_binding)),
        "runtime_identity": dict(runtime_identity),
        "runtime_identity_sha256": primary.runtime_identity_digest(runtime_identity),
        "secondary_request_counts": {
            "plan": {
                "expected_total": secondary.EXPECTED_SECONDARY_REQUEST_COUNT,
                "observed_total": len(all_secondary),
                "expected_by_variant": dict(
                    secondary.EXPECTED_REQUEST_COUNT_BY_VARIANT
                ),
                "observed_by_variant": secondary.counts_by_variant(all_secondary),
            },
            "image": {
                "image_id": image_id,
                "expected_by_variant": dict(expected_counts),
                "expected_total": len(expected_requests),
                "observed_by_variant": secondary.counts_by_variant(rows),
                "observed_total": len(rows),
            },
        },
        "executed": {
            "session_image_id": image_id,
            "row_count": len(rows),
            "gt_owner_ids": sorted({str(row["gt_owner_id"]) for row in rows}),
            "logical_context_group_count": 2 * len(rows),
            "logical_context_groups_sha256": _sha256_json(
                sorted(
                    str(payload["context_group_id"])
                    for row in rows
                    for payload in row["roots"].values()
                )
            ),
            "request_ids_sha256": _sha256_json(
                sorted(str(row["request_id"]) for row in rows)
            ),
        },
        "quarantine": {
            "schema_version": secondary.QUARANTINE_SCHEMA_VERSION,
            "count": len(quarantine_entries),
            "entries": [dict(entry) for entry in quarantine_entries],
            "policy": "any quarantined request withholds the whole shard's evidence",
        },
        "policy": {
            **dict(sut.REQUIRED_POLICY),
            "backend_kind": "hf",
            "requested_batch_size": 1,
            "effective_batch_size": 1,
            "batching_applicable": False,
            "batching_note": secondary.BATCH_POLICY_NOT_APPLICABLE,
        },
        "admission": {
            "smoke_shard_id": admission["smoke_shard_id"],
            "smoke_image_id": admission["smoke_image_id"],
            "admission_content_sha256": admission["admission_content_sha256"],
            "root_backend": dict(sorted(admission["root_backend"].items())),
        },
        "artifact_determinism": "no_wall_clock_or_host_specific_field_is_sealed",
    }
    return _seal(receipt, "receipt_content_sha256")


def make_parity(
    *, shard_id: str, image_id: str, admission: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": secondary.PARITY_SCHEMA_VERSION,
        "unit_id": secondary.UNIT_ID,
        "shard_id": shard_id,
        "session_image_id": image_id,
        "backend_kind": "hf",
        "backend_is_evidence_bearing": True,
        "root_backend": dict(sorted(admission["root_backend"].items())),
        "max_selected_logit_abs_diff_threshold": (
            secondary.CACHE_PARITY_MAX_SELECTED_LOGIT_ABS_DIFF
        ),
        "compared_request_ids": [],
        "cached_versus_uncached": None,
        "inherited_admission": {
            "smoke_shard_id": admission["smoke_shard_id"],
            "smoke_image_id": admission["smoke_image_id"],
            "admission_content_sha256": admission["admission_content_sha256"],
            "cache_admitted": admission["cache_admitted"],
            "primary_analysis_binding_sha256": admission[
                "primary_analysis_binding_sha256"
            ],
        },
        "batching": {
            "applicable": False,
            "effective_batch_size": 1,
            "reason": secondary.BATCH_POLICY_NOT_APPLICABLE,
        },
        "compared_fields": [
            "selected_logprob",
            "argmax_token_id",
            "selected_is_argmax",
            "segment_delta_sign",
        ],
    }


def write_shard(
    shard_dir: Path,
    *,
    receipt: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    parity: Mapping[str, Any],
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    (shard_dir / secondary.RECEIPT_NAME).write_bytes(
        secondary.canonical_json_bytes(receipt) + b"\n"
    )
    (shard_dir / secondary.PARITY_NAME).write_bytes(
        secondary.canonical_json_bytes(parity) + b"\n"
    )
    (shard_dir / secondary.ROWS_NAME).write_bytes(
        b"".join(secondary.canonical_json_bytes(row) + b"\n" for row in rows)
    )
    return shard_dir


# ---------------------------------------------------------------------------
# 4. The whole twelve-shard capture
# ---------------------------------------------------------------------------


@dataclass
class Capture:
    """A complete twelve-shard secondary capture, plus every sealed input."""

    root: Path
    plan: PlanFixture
    analysis_dir: Path
    analysis_receipt: dict[str, Any]
    analysis_binding: dict[str, Any]
    admission_path: Path
    admission: dict[str, Any]
    runtime_identity: dict[str, Any]
    shard_root: Path
    shard_dirs: dict[str, Path] = field(default_factory=dict)

    @property
    def analysis_binding_sha256(self) -> str:
        return _sha256_json(dict(self.analysis_binding))

    def merge_kwargs(self, **overrides: Any) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "shard_dirs": [self.shard_dirs[image_id] for image_id in IMAGE_IDS],
            "plan_dir": self.plan.plan_dir,
            "primary_analysis_dir": self.analysis_dir,
            "admission_path": self.admission_path,
        }
        kwargs.update(overrides)
        return kwargs

    def run_merge(self, output_dir: Path | None = None, **overrides: Any) -> dict[str, Any]:
        result = sut.merge_shards(**self.merge_kwargs(**overrides))
        if output_dir is not None:
            result["published"] = primary_merge.publish_merge(
                Path(output_dir), result["files"]
            )
        return result

    # -- shard mutation helpers ------------------------------------------
    def read_rows(self, image_id: str) -> list[dict[str, Any]]:
        path = self.shard_dirs[image_id] / secondary.ROWS_NAME
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def read_receipt(self, image_id: str) -> dict[str, Any]:
        return json.loads(
            (self.shard_dirs[image_id] / secondary.RECEIPT_NAME).read_text(
                encoding="utf-8"
            )
        )

    def rewrite_rows(self, image_id: str, rows: Sequence[Mapping[str, Any]]) -> None:
        """Rewrite one shard's rows and reseal its receipt around them."""

        receipt = self.read_receipt(image_id)
        expected = self.plan.requests_for_image(image_id)
        resealed = make_shard_receipt(
            shard_id=str(receipt["shard_id"]),
            image_id=image_id,
            plan=self.plan,
            runtime_identity=self.runtime_identity,
            admission=self.admission,
            analysis_binding=self.analysis_binding,
            rows=rows,
            expected_requests=expected,
        )
        write_shard(
            self.shard_dirs[image_id],
            receipt=resealed,
            rows=rows,
            parity=make_parity(
                shard_id=str(receipt["shard_id"]),
                image_id=image_id,
                admission=self.admission,
            ),
        )

    def patch_row(self, image_id: str, index: int, **fields: Any) -> None:
        rows = self.read_rows(image_id)
        rows[index].update(fields)
        self.rewrite_rows(image_id, rows)

    def write_receipt(self, image_id: str, receipt: Mapping[str, Any]) -> None:
        (self.shard_dirs[image_id] / secondary.RECEIPT_NAME).write_bytes(
            secondary.canonical_json_bytes(receipt) + b"\n"
        )

    def reseal_receipt(self, image_id: str, **fields: Any) -> None:
        receipt = self.read_receipt(image_id)
        receipt.update(fields)
        receipt.pop("receipt_content_sha256", None)
        self.write_receipt(image_id, _seal(receipt, "receipt_content_sha256"))


def build_secondary_capture(tmp_path: Path, *, marker: str = "frozen") -> Capture:
    """Twelve successful shards over the whole sealed 64-request census."""

    plan = build_plan(tmp_path, marker=marker)
    analysis_dir, analysis_receipt = build_primary_analysis(
        tmp_path, plan=plan, marker=marker
    )
    analysis_binding = build_analysis_binding(
        analysis_dir, receipt=analysis_receipt, plan=plan
    )
    binding_sha256 = _sha256_json(dict(analysis_binding))
    runtime_identity = make_runtime_identity(marker=marker)
    admission_path, admission = build_admission(
        tmp_path,
        plan=plan,
        runtime_identity=runtime_identity,
        analysis_binding_sha256=binding_sha256,
    )

    shard_root = tmp_path / "secondary-shards"
    shard_root.mkdir(parents=True)
    capture = Capture(
        root=tmp_path,
        plan=plan,
        analysis_dir=analysis_dir,
        analysis_receipt=analysis_receipt,
        analysis_binding=analysis_binding,
        admission_path=admission_path,
        admission=admission,
        runtime_identity=runtime_identity,
        shard_root=shard_root,
    )
    for image_index, image_id in enumerate(IMAGE_IDS):
        shard_id = f"secondary-capture-{image_id}"
        expected = plan.requests_for_image(image_id)
        rows = [
            build_secondary_row(
                request=request,
                plan=plan,
                shard_id=shard_id,
                analysis_binding_sha256=binding_sha256,
                index=image_index * 7 + offset,
            )
            for offset, request in enumerate(expected)
        ]
        receipt = make_shard_receipt(
            shard_id=shard_id,
            image_id=image_id,
            plan=plan,
            runtime_identity=runtime_identity,
            admission=admission,
            analysis_binding=analysis_binding,
            rows=rows,
            expected_requests=expected,
        )
        capture.shard_dirs[image_id] = write_shard(
            shard_root / shard_id,
            receipt=receipt,
            rows=rows,
            parity=make_parity(
                shard_id=shard_id, image_id=image_id, admission=admission
            ),
        )
    return capture


@pytest.fixture()
def capture(tmp_path: Path) -> Capture:
    return build_secondary_capture(tmp_path)


# ---------------------------------------------------------------------------
# 5. The fixture itself is the frozen census
# ---------------------------------------------------------------------------


def test_fixture_reproduces_the_frozen_26_26_12_census(capture: Capture) -> None:
    secondary_rows = [
        row
        for row in capture.plan.requests
        if str(row["readout_tier"]) == secondary.SECONDARY_READOUT_TIER
    ]
    assert len(secondary_rows) == secondary.EXPECTED_SECONDARY_REQUEST_COUNT == 64
    assert secondary.counts_by_variant(secondary_rows) == dict(
        secondary.EXPECTED_REQUEST_COUNT_BY_VARIANT
    )
    # The per-image distribution is genuinely uneven, so nothing downstream can
    # pass by assuming a uniform split.
    per_image = {
        image_id: len(capture.plan.requests_for_image(image_id))
        for image_id in IMAGE_IDS
    }
    assert len(set(per_image.values())) > 1
    assert sum(per_image.values()) == 64


# ---------------------------------------------------------------------------
# 6. The happy path
# ---------------------------------------------------------------------------


def test_merge_publishes_a_complete_self_sealed_family(
    capture: Capture, tmp_path: Path
) -> None:
    output_dir = tmp_path / "merged"
    result = capture.run_merge(output_dir)
    assert result["published"]["published"] is True
    assert sorted(entry.name for entry in output_dir.iterdir()) == sorted(
        sut.MERGED_OUTPUT_NAMES
    )

    receipt = json.loads((output_dir / sut.MERGE_RECEIPT_NAME).read_text("utf-8"))
    assert receipt["schema_version"] == sut.MERGE_SCHEMA_VERSION
    assert receipt["unit_id"] == sut.UNIT_ID
    sut.assert_self_sealed(
        receipt, digest_key="receipt_content_sha256", label="merge receipt"
    )
    for name in (sut.MERGED_ROWS_NAME, sut.MERGED_PARITY_NAME):
        entry = receipt["output_file_digests"][name]
        assert entry["sha256"] == secondary.sha256_file(output_dir / name)
        assert entry["byte_size"] == (output_dir / name).stat().st_size
    assert receipt["policy"]["primary_branch_labels_read"] is False
    assert receipt["policy"]["threshold_fitting"] is False
    assert receipt["policy"]["claim_boundary"] == secondary.CLAIM_BOUNDARY


def test_merge_covers_the_complete_twelve_image_sixty_four_id_union(
    capture: Capture,
) -> None:
    result = capture.run_merge()
    receipt = result["receipt"]
    plan_ids = sorted(
        str(row["request_id"])
        for row in capture.plan.requests
        if str(row["readout_tier"]) == secondary.SECONDARY_READOUT_TIER
    )
    assert receipt["secondary_requests"]["request_count"] == 64
    assert receipt["secondary_requests"]["request_ids_sha256"] == _sha256_json(plan_ids)
    assert receipt["secondary_requests"]["request_ids_sha256"] == (
        receipt["secondary_requests"]["plan_request_ids_sha256"]
    )
    assert receipt["secondary_requests"]["counts_by_variant"] == dict(
        secondary.EXPECTED_REQUEST_COUNT_BY_VARIANT
    )
    assert len({shard["session_image_id"] for shard in receipt["shards"]}) == 12
    cohorts = receipt["observed_cohorts"]
    assert cohorts["primary_owner_count"] == primary.PRIMARY_OWNER_COUNT_U == 26
    assert cohorts["tp_replay_control_owner_count"] == 12
    assert cohorts["optional_f_owner_count"] == 26
    assert not set(cohorts["primary_owner_ids"]) & set(
        cohorts["tp_replay_control_owner_ids"]
    )


def test_merge_preserves_every_row_verbatim(capture: Capture) -> None:
    result = capture.run_merge()
    published = {str(row["request_id"]): row for row in result["rows"]}
    for image_id in IMAGE_IDS:
        for row in capture.read_rows(image_id):
            assert published[str(row["request_id"])] == row


def test_merge_is_deterministic_and_idempotent(capture: Capture, tmp_path: Path) -> None:
    output_dir = tmp_path / "merged"
    first = capture.run_merge(output_dir)
    second = capture.run_merge(output_dir)
    assert first["files"] == second["files"]
    assert second["published"]["published"] is False
    assert second["published"]["publish_mode"] == "no_op_identical_rerun"


def test_merge_shard_order_does_not_change_the_bytes(capture: Capture) -> None:
    forward = capture.run_merge()
    reversed_dirs = [capture.shard_dirs[image_id] for image_id in reversed(IMAGE_IDS)]
    backward = capture.run_merge(shard_dirs=reversed_dirs)
    assert forward["files"] == backward["files"]


def test_publish_refuses_a_drifted_existing_directory(
    capture: Capture, tmp_path: Path
) -> None:
    output_dir = tmp_path / "merged"
    capture.run_merge(output_dir)
    (output_dir / sut.MERGED_ROWS_NAME).write_bytes(b"{}\n")
    with pytest.raises(primary_merge.MergeContractError, match="byte-identical"):
        capture.run_merge(output_dir)


# ---------------------------------------------------------------------------
# 7. Shard-set closure
# ---------------------------------------------------------------------------


def test_merge_refuses_a_missing_image(capture: Capture) -> None:
    kwargs = capture.merge_kwargs()
    kwargs["shard_dirs"] = kwargs["shard_dirs"][:-1]
    with pytest.raises(sut.SecondaryMergeContractError, match="no successful shard"):
        sut.merge_shards(**kwargs)


def test_merge_refuses_two_shards_for_one_image(capture: Capture, tmp_path: Path) -> None:
    image_id = IMAGE_IDS[0]
    twin = tmp_path / "twin"
    twin.mkdir()
    for name in sut.REQUIRED_SHARD_FILES:
        (twin / name).write_bytes((capture.shard_dirs[image_id] / name).read_bytes())
    kwargs = capture.merge_kwargs()
    kwargs["shard_dirs"] = [*kwargs["shard_dirs"], twin]
    with pytest.raises(sut.SecondaryMergeContractError, match="more than one shard"):
        sut.merge_shards(**kwargs)


def test_merge_refuses_the_same_shard_directory_twice(capture: Capture) -> None:
    kwargs = capture.merge_kwargs()
    kwargs["shard_dirs"] = [*kwargs["shard_dirs"], kwargs["shard_dirs"][0]]
    with pytest.raises(sut.SecondaryMergeContractError, match="supplied twice"):
        sut.merge_shards(**kwargs)


def test_merge_refuses_an_image_outside_the_frozen_twelve(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    capture.reseal_receipt(
        image_id,
        executed={**capture.read_receipt(image_id)["executed"], "session_image_id": "99"},
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="frozen twelve"):
        capture.run_merge()


def test_merge_refuses_a_quarantine_stopped_shard_directory(capture: Capture) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / secondary.QUARANTINE_NAME).write_text("{}\n")
    with pytest.raises(sut.SecondaryMergeContractError, match="quarantine rule"):
        capture.run_merge()


def test_merge_refuses_a_smoke_admission_shard(capture: Capture) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / secondary.ADMISSION_NAME).write_text("{}\n")
    with pytest.raises(sut.SecondaryMergeContractError, match="smoke admission shard"):
        capture.run_merge()


def test_merge_refuses_an_unknown_artifact_inside_a_shard(capture: Capture) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / "notes.txt").write_text("hello\n")
    with pytest.raises(sut.SecondaryMergeContractError, match="unknown artifact"):
        capture.run_merge()


def test_merge_refuses_an_incomplete_shard(capture: Capture) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / secondary.PARITY_NAME).unlink()
    with pytest.raises(sut.SecondaryMergeContractError, match="incomplete"):
        capture.run_merge()


def test_merge_refuses_a_shard_that_declares_a_quarantined_request(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    capture.reseal_receipt(
        image_id,
        quarantine={
            **receipt["quarantine"],
            "count": 1,
            "entries": [
                {
                    "request_id": "req:x",
                    "gt_owner_id": "gt:x",
                    "reason": "baseline_native_argmax_replay_mismatch",
                    "detail": "d",
                }
            ],
        },
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="withholds the whole"):
        capture.run_merge()


def test_discover_shard_dirs_finds_every_capture_directory(capture: Capture) -> None:
    found = sut.discover_shard_dirs(capture.shard_root)
    assert sorted(found) == sorted(capture.shard_dirs.values())


def test_cli_merges_a_discovered_shard_root(
    capture: Capture, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output_dir = tmp_path / "cli-merged"
    exit_code = sut.main(
        [
            "--shard-root",
            str(capture.shard_root),
            "--plan-dir",
            str(capture.plan.plan_dir),
            "--primary-analysis-dir",
            str(capture.analysis_dir),
            "--admission",
            str(capture.admission_path),
            "--output-dir",
            str(output_dir),
        ]
    )
    assert exit_code == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["row_count"] == 64
    assert summary["merged"]["published"] is True
    assert sorted(entry.name for entry in output_dir.iterdir()) == sorted(
        sut.MERGED_OUTPUT_NAMES
    )


def test_cli_fails_closed_on_a_contract_violation(
    capture: Capture, tmp_path: Path
) -> None:
    (capture.shard_dirs[IMAGE_IDS[0]] / secondary.QUARANTINE_NAME).write_text("{}\n")
    with pytest.raises(SystemExit, match="secondary merge contract violated"):
        sut.main(
            [
                "--shard-root",
                str(capture.shard_root),
                "--plan-dir",
                str(capture.plan.plan_dir),
                "--primary-analysis-dir",
                str(capture.analysis_dir),
                "--admission",
                str(capture.admission_path),
                "--output-dir",
                str(tmp_path / "cli-merged"),
            ]
        )


# ---------------------------------------------------------------------------
# 8. Request-set closure
# ---------------------------------------------------------------------------


def test_merge_refuses_a_withheld_request(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    capture.rewrite_rows(image_id, rows[:-1])
    with pytest.raises(sut.SecondaryMergeContractError, match="missing="):
        capture.run_merge()


def test_merge_refuses_a_duplicated_request_inside_one_shard(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    capture.rewrite_rows(image_id, [*rows, copy.deepcopy(rows[0])])
    with pytest.raises(sut.SecondaryMergeContractError, match="same secondary request"):
        capture.run_merge()


def test_merge_refuses_a_request_published_by_two_shards(capture: Capture) -> None:
    donor, host = IMAGE_IDS[0], IMAGE_IDS[1]
    stolen = copy.deepcopy(capture.read_rows(donor)[0])
    stolen["shard_id"] = f"secondary-capture-{host}"
    capture.rewrite_rows(host, [*capture.read_rows(host), stolen])
    with pytest.raises(sut.SecondaryMergeContractError, match="one image session"):
        capture.run_merge()


def test_merge_refuses_an_unknown_request_id(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["request_id"] = "req:" + "0" * 32
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="not one of the sealed"):
        capture.run_merge()


def test_merge_refuses_a_shard_whose_receipt_census_drifted(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    image_block = dict(receipt["secondary_request_counts"]["image"])
    image_block["expected_by_variant"] = {
        variant: 99 for variant in image_block["expected_by_variant"]
    }
    capture.reseal_receipt(
        image_id,
        secondary_request_counts={
            **receipt["secondary_request_counts"],
            "image": image_block,
        },
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="expected_by_variant"):
        capture.run_merge()


# ---------------------------------------------------------------------------
# 9. Row-level identity, arithmetic and variant routing
# ---------------------------------------------------------------------------


def _first_row_index(capture: Capture, image_id: str, variant: str) -> int:
    for index, row in enumerate(capture.read_rows(image_id)):
        if str(row["variant"]) == variant:
            return index
    raise AssertionError(f"no {variant!r} row on image {image_id!r}")


def test_merge_refuses_a_tampered_scored_token_sequence(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["scored_token_ids"] = list(rows[0]["scored_token_ids"])[:-1]
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="scores tokens"):
        capture.run_merge()


def test_merge_refuses_a_tampered_scored_token_digest(capture: Capture) -> None:
    capture.patch_row(IMAGE_IDS[0], 0, scored_token_ids_sha256="0" * 64)
    with pytest.raises(sut.SecondaryMergeContractError, match="digest does not reconstruct"):
        capture.run_merge()


def test_merge_refuses_a_drifted_segment_span(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["segments"]["coordinates"]["token_index_start"] += 1
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="declares span"):
        capture.run_merge()


def test_merge_refuses_a_rewritten_segment_token_list(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["segments"]["description"]["token_ids"] = [1, 2, 3]
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="literal span"):
        capture.run_merge()


def test_merge_refuses_a_baseline_root_that_appended_something(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["roots"][sut.ROOT_BASELINE]["appended_token_count"] = 3
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="appended 3 tokens"):
        capture.run_merge()


def test_merge_refuses_a_modified_root_with_a_foreign_appended_row(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["roots"][sut.ROOT_MODIFIED]["appended_token_ids_sha256"] = "0" * 64
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="appended tokens hash to"):
        capture.run_merge()


def test_merge_refuses_an_inserted_row_that_is_not_the_owners_clean_gt_row(
    capture: Capture,
) -> None:
    capture.patch_row(
        IMAGE_IDS[0], 0, inserted_clean_row_c_token_ids_sha256="0" * 64
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="another inserted clean row"):
        capture.run_merge()


def test_merge_refuses_a_drifted_logical_context_group(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["roots"][sut.ROOT_MODIFIED]["context_group_id"] = "ctx:tampered"
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="logical context group"):
        capture.run_merge()


def test_merge_refuses_a_selected_is_argmax_stream_that_does_not_reconstruct(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    stream = list(rows[0]["roots"][sut.ROOT_BASELINE]["selected_is_argmax"])
    stream[0] = not stream[0]
    rows[0]["roots"][sut.ROOT_BASELINE]["selected_is_argmax"] = stream
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="selected_is_argmax"):
        capture.run_merge()


def test_merge_refuses_a_nonfinite_selected_logprob(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    selected = list(rows[0]["roots"][sut.ROOT_MODIFIED]["selected_logprobs"])
    selected[0] = float("-inf")
    rows[0]["roots"][sut.ROOT_MODIFIED]["selected_logprobs"] = selected
    path = capture.shard_dirs[image_id] / secondary.ROWS_NAME
    path.write_bytes(
        b"".join(
            json.dumps(row, allow_nan=True).encode("utf-8") + b"\n" for row in rows
        )
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="non-finite"):
        capture.run_merge()


def test_merge_refuses_drifted_per_segment_sums(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["roots"][sut.ROOT_MODIFIED]["segment_sums"]["description"]["sum"] += 1.0
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="per-segment sums"):
        capture.run_merge()


def test_merge_refuses_delta_arithmetic_that_does_not_reconstruct(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["deltas"]["complete_row"]["delta"] += 0.5
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="modified-minus-baseline"):
        capture.run_merge()


def test_merge_refuses_a_flipped_delta_sign(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["deltas"]["coordinates"]["sign"] = -int(
        rows[0]["deltas"]["coordinates"]["sign"] or 1
    )
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="modified-minus-baseline"):
        capture.run_merge()


def test_merge_refuses_a_wrong_delta_orientation(capture: Capture) -> None:
    capture.patch_row(IMAGE_IDS[0], 0, delta_orientation="baseline_minus_modified")
    with pytest.raises(sut.SecondaryMergeContractError, match="delta_orientation"):
        capture.run_merge()


def test_merge_refuses_an_unadmitted_baseline_replay(capture: Capture) -> None:
    capture.patch_row(IMAGE_IDS[0], 0, baseline_replay_admitted=False)
    with pytest.raises(sut.SecondaryMergeContractError, match="baseline_replay_admitted"):
        capture.run_merge()


def test_merge_refuses_a_row_whose_replay_admission_was_not_enforced(
    capture: Capture,
) -> None:
    capture.patch_row(IMAGE_IDS[0], 0, replay_admission_enforced=False)
    with pytest.raises(
        sut.SecondaryMergeContractError, match="replay_admission_enforced"
    ):
        capture.run_merge()


def test_merge_refuses_a_row_that_claims_a_rollout_result(capture: Capture) -> None:
    capture.patch_row(IMAGE_IDS[0], 0, claim_boundary="final-set retention proven")
    with pytest.raises(sut.SecondaryMergeContractError, match="claim_boundary"):
        capture.run_merge()


def test_merge_refuses_a_variant_moved_into_another_cohort(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_BENIGN_SUBSTITUTION)
    capture.patch_row(image_id, index, cohort=sut.PRIMARY_COHORT)
    with pytest.raises(sut.SecondaryMergeContractError, match="cohort"):
        capture.run_merge()


def test_merge_refuses_a_relabelled_optional_flag(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_P_PLUS_C_THEN_E)
    capture.patch_row(image_id, index, plan_optional=True)
    with pytest.raises(sut.SecondaryMergeContractError, match="optionality"):
        capture.run_merge()


def test_merge_refuses_a_primary_variant_that_appends_away_from_its_baseline(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_P_PLUS_C_THEN_E)
    rows = capture.read_rows(image_id)
    rows[index]["baseline_context_id"] = _context_id(image_id, 0)
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="pairs against baseline"):
        capture.run_merge()


def test_merge_refuses_a_benign_substitution_that_appends_at_its_own_baseline(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_BENIGN_SUBSTITUTION)
    rows = capture.read_rows(image_id)
    baseline = rows[index]["baseline_context_id"]
    rows[index]["modified_context_id"] = baseline
    rows[index]["roots"][sut.ROOT_MODIFIED]["context_id"] = baseline
    capture.rewrite_rows(image_id, rows)
    # Moving the append onto the baseline boundary already breaks the immutable
    # request key, which is derived from the appended-at context, so the row is
    # refused before the variant-semantics check is even reached.
    with pytest.raises(sut.SecondaryMergeContractError, match="declares request key"):
        capture.run_merge()


def test_merge_refuses_two_roots_that_do_not_share_one_boundary_prefix(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_P_PLUS_C_THEN_E)
    rows = capture.read_rows(image_id)
    rows[index]["roots"][sut.ROOT_MODIFIED]["executed_prefix_token_ids_sha256"] = "0" * 64
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="two different executed prefixes"):
        capture.run_merge()


def test_merge_refuses_a_benign_substitution_with_one_shared_prefix(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_BENIGN_SUBSTITUTION)
    rows = capture.read_rows(image_id)
    shared = rows[index]["roots"][sut.ROOT_BASELINE]["executed_prefix_token_ids_sha256"]
    rows[index]["roots"][sut.ROOT_MODIFIED]["executed_prefix_token_ids_sha256"] = shared
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(
        sut.SecondaryMergeContractError, match="one executed prefix from two different"
    ):
        capture.run_merge()


def test_merge_refuses_a_successor_that_is_not_the_sealed_post_row_boundary(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F)
    capture.patch_row(
        image_id, index, successor_context_id=_context_id(image_id, 0)
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="post-row boundary"):
        capture.run_merge()


def test_merge_refuses_a_baseline_attributed_to_another_registry_row(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    index = _first_row_index(capture, image_id, secondary.VARIANT_P_PLUS_C_THEN_E)
    rows = capture.read_rows(image_id)
    assert any(
        str(row["variant"]) == secondary.VARIANT_P_PLUS_E_PLUS_C_THEN_F for row in rows
    )  # the fixture really does carry both primary variants
    rows[index]["baseline_context_source"] = "cohort_registry.f_row"
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(sut.SecondaryMergeContractError, match="attributes its baseline"):
        capture.run_merge()


def test_merge_refuses_a_row_missing_a_segment_delta(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    rows = capture.read_rows(image_id)
    rows[0]["deltas"].pop("coordinates")
    capture.rewrite_rows(image_id, rows)
    with pytest.raises(
        sut.SecondaryMergeContractError, match="exactly the three per-segment deltas"
    ):
        capture.run_merge()


def test_merge_refuses_a_row_bound_to_another_primary_analysis(capture: Capture) -> None:
    capture.patch_row(IMAGE_IDS[0], 0, primary_analysis_binding_sha256="0" * 64)
    with pytest.raises(sut.SecondaryMergeContractError, match="binds a primary analysis"):
        capture.run_merge()


def test_merge_refuses_a_binding_block_that_does_not_reconstruct_its_digest(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    capture.reseal_receipt(
        image_id,
        primary_analysis={
            **receipt["primary_analysis"],
            "analysis_dir": "/somewhere/else",
        },
    )
    with pytest.raises(
        sut.SecondaryMergeContractError, match="does not reconstruct from"
    ):
        capture.run_merge()


def test_merge_refuses_shards_bound_to_two_different_gates(
    capture: Capture, tmp_path: Path
) -> None:
    other_dir, other_receipt = build_primary_analysis(
        tmp_path / "other-gate", plan=capture.plan, marker="other"
    )
    other_binding = build_analysis_binding(
        other_dir, receipt=other_receipt, plan=capture.plan
    )
    image_id = IMAGE_IDS[0]
    capture.reseal_receipt(
        image_id,
        primary_analysis=other_binding,
        primary_analysis_binding_sha256=_sha256_json(other_binding),
    )
    with pytest.raises(
        sut.SecondaryMergeContractError, match="primary analysis bindings"
    ):
        capture.run_merge()


# ---------------------------------------------------------------------------
# 10. Cross-shard identity
# ---------------------------------------------------------------------------


def test_merge_refuses_a_mixed_runtime_identity(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    drifted = make_runtime_identity(marker="drifted")
    capture.reseal_receipt(
        image_id,
        runtime_identity=drifted,
        runtime_identity_sha256=primary.runtime_identity_digest(drifted),
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="never admitted"):
        capture.run_merge()


def test_merge_refuses_a_simulated_backend(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    fake = {**copy.deepcopy(capture.runtime_identity), "is_real_model": False}
    capture.reseal_receipt(
        image_id,
        runtime_identity=fake,
        runtime_identity_sha256=primary.runtime_identity_digest(fake),
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="not a real model"):
        capture.run_merge()


def test_merge_refuses_a_receipt_whose_runtime_digest_does_not_reconstruct(
    capture: Capture,
) -> None:
    capture.reseal_receipt(IMAGE_IDS[0], runtime_identity_sha256="0" * 64)
    with pytest.raises(sut.SecondaryMergeContractError, match="does not reconstruct"):
        capture.run_merge()


def test_merge_refuses_a_mixed_secondary_scorer_revision(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    drifted = copy.deepcopy(capture.runtime_identity)
    drifted["source_identity"] = source_identity(secondary_sha256="ff" * 32)
    receipt = capture.read_receipt(image_id)
    capture.reseal_receipt(
        image_id,
        runtime_identity=drifted,
        runtime_identity_sha256=primary.runtime_identity_digest(drifted),
        scorer_source_sha256="ff" * 32,
        source_identity=drifted["source_identity"],
        admission=receipt["admission"],
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="never admitted"):
        capture.run_merge()


def test_merge_refuses_a_receipt_that_disagrees_with_its_own_source_identity(
    capture: Capture,
) -> None:
    capture.reseal_receipt(IMAGE_IDS[0], scorer_source_sha256="ff" * 32)
    with pytest.raises(sut.SecondaryMergeContractError, match="disagrees with itself"):
        capture.run_merge()


def test_merge_honours_an_explicit_scorer_source_pin(capture: Capture) -> None:
    pinned = capture.run_merge(expect_scorer_source_sha256=SECONDARY_SCORER_SHA256)
    assert pinned["receipt"]["scorer_source_pinned"] is True
    assert pinned["receipt"]["secondary_scorer_source_sha256"] == SECONDARY_SCORER_SHA256
    with pytest.raises(sut.SecondaryMergeContractError, match="not the pinned"):
        capture.run_merge(expect_scorer_source_sha256="0" * 64)


def test_merge_refuses_a_shard_that_seals_no_source_digest_for_a_module(
    capture: Capture,
) -> None:
    image_id = IMAGE_IDS[0]
    trimmed = {
        key: value
        for key, value in source_identity().items()
        if key != sut.PRIMARY_ANALYZER_SOURCE_IDENTITY_KEY
    }
    capture.reseal_receipt(image_id, source_identity=trimmed)
    with pytest.raises(sut.SecondaryMergeContractError, match="seals no source digest"):
        capture.run_merge()


def test_merge_refuses_an_edited_receipt(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    receipt["mode"] = secondary.MODE_SMOKE
    capture.write_receipt(image_id, receipt)
    with pytest.raises(sut.SecondaryMergeContractError, match="not a capture receipt"):
        capture.run_merge()


def test_merge_refuses_a_receipt_that_did_not_reseal(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    receipt["shard_id"] = "renamed"
    capture.write_receipt(image_id, receipt)
    with pytest.raises(sut.SecondaryMergeContractError, match="edited after"):
        capture.run_merge()


def test_merge_refuses_a_policy_that_left_the_frozen_contract(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    capture.reseal_receipt(
        image_id,
        policy={**receipt["policy"], "primary_branch_assignment_performed": True},
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="policy"):
        capture.run_merge()


def test_merge_refuses_a_parity_file_bound_to_another_admission(capture: Capture) -> None:
    image_id = IMAGE_IDS[0]
    parity = make_parity(
        shard_id=f"secondary-capture-{image_id}",
        image_id=image_id,
        admission={**capture.admission, "admission_content_sha256": "0" * 64},
    )
    (capture.shard_dirs[image_id] / secondary.PARITY_NAME).write_bytes(
        secondary.canonical_json_bytes(parity) + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="disagrees with itself"):
        capture.run_merge()


# ---------------------------------------------------------------------------
# 11. Plan, gate and admission identity
# ---------------------------------------------------------------------------


def test_merge_refuses_a_tampered_plan_file(capture: Capture) -> None:
    (capture.plan.plan_dir / plan_builder.REQUEST_PLAN_NAME).write_bytes(b"{}\n")
    with pytest.raises(primary_merge.MergeContractError, match="frozen plan was modified"):
        capture.run_merge()


def test_merge_refuses_an_edited_plan_manifest(capture: Capture) -> None:
    path = capture.plan.plan_dir / plan_builder.MANIFEST_NAME
    manifest = json.loads(path.read_text("utf-8"))
    manifest["lineage"]["census_run_root"] = "/elsewhere"
    path.write_bytes(secondary.canonical_json_bytes(manifest) + b"\n")
    with pytest.raises(primary_merge.MergeContractError, match="reconstruct its own"):
        capture.run_merge()


def test_merge_refuses_a_shard_captured_against_another_plan(
    capture: Capture, tmp_path: Path
) -> None:
    other = build_plan(tmp_path / "other", marker="other")
    image_id = IMAGE_IDS[0]
    receipt = capture.read_receipt(image_id)
    capture.reseal_receipt(
        image_id,
        plan={
            **receipt["plan"],
            "manifest_content_sha256": other.manifest["manifest_content_sha256"],
        },
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="captured against plan manifest"):
        capture.run_merge()


def test_merge_refuses_an_analysis_directory_that_drifted(capture: Capture) -> None:
    (capture.analysis_dir / secondary.ANALYSIS_REPORT_MD_NAME).write_text("# edited\n")
    with pytest.raises(sut.SecondaryMergeContractError, match="do not hash to the bytes"):
        capture.run_merge()


def test_merge_refuses_an_incomplete_analysis_directory(capture: Capture) -> None:
    (capture.analysis_dir / secondary.ANALYSIS_REPORT_MD_NAME).unlink()
    with pytest.raises(secondary.SecondaryCompatibilityContractError):
        capture.run_merge()


def test_merge_refuses_an_analysis_that_read_secondary_fields(
    capture: Capture,
) -> None:
    path = capture.analysis_dir / secondary.ANALYSIS_RECEIPT_NAME
    receipt = json.loads(path.read_text("utf-8"))
    receipt["policy"]["secondary_compatibility_read"] = True
    receipt.pop("receipt_content_sha256")
    path.write_bytes(
        secondary.canonical_json_bytes(_seal(receipt, "receipt_content_sha256")) + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="secondary_compatibility_read"):
        capture.run_merge()


def test_merge_refuses_an_analysis_decided_over_another_plan(
    capture: Capture, tmp_path: Path
) -> None:
    other_plan = build_plan(tmp_path / "other", marker="other")
    merged_dir = Path(
        json.loads(
            (capture.analysis_dir / secondary.ANALYSIS_RECEIPT_NAME).read_text("utf-8")
        )["merged_dir"]
    )
    merge_receipt = json.loads(
        (merged_dir / secondary.ANALYSIS_MERGE_RECEIPT_NAME).read_text("utf-8")
    )
    merge_receipt["plan"]["manifest_content_sha256"] = other_plan.manifest[
        "manifest_content_sha256"
    ]
    merge_receipt.pop("receipt_content_sha256")
    (merged_dir / secondary.ANALYSIS_MERGE_RECEIPT_NAME).write_bytes(
        secondary.canonical_json_bytes(_seal(merge_receipt, "receipt_content_sha256"))
        + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="do not describe one merge"):
        capture.run_merge()


def test_merge_refuses_an_admission_sealed_against_another_plan(
    capture: Capture, tmp_path: Path
) -> None:
    other = build_plan(tmp_path / "other", marker="other")
    admission = dict(capture.admission)
    admission["plan_manifest_content_sha256"] = other.manifest["manifest_content_sha256"]
    admission.pop("admission_content_sha256")
    capture.admission_path.write_bytes(
        secondary.canonical_json_bytes(_seal(admission, "admission_content_sha256"))
        + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="different plan manifest"):
        capture.run_merge()


def test_merge_refuses_an_admission_sealed_against_another_gate(
    capture: Capture,
) -> None:
    admission = dict(capture.admission)
    admission["primary_analysis_binding_sha256"] = "0" * 64
    admission.pop("admission_content_sha256")
    capture.admission_path.write_bytes(
        secondary.canonical_json_bytes(_seal(admission, "admission_content_sha256"))
        + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="different primary analysis"):
        capture.run_merge()


def test_merge_refuses_an_admission_missing_a_variant(capture: Capture) -> None:
    admission = dict(capture.admission)
    admission["smoke_variants"] = {
        variant: 0 for variant in secondary.SECONDARY_VARIANTS
    }
    admission.pop("admission_content_sha256")
    capture.admission_path.write_bytes(
        secondary.canonical_json_bytes(_seal(admission, "admission_content_sha256"))
        + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="proved no parity"):
        capture.run_merge()


def test_merge_refuses_an_edited_admission(capture: Capture) -> None:
    admission = dict(capture.admission)
    admission["cache_admitted"] = False
    capture.admission_path.write_bytes(
        secondary.canonical_json_bytes(admission) + b"\n"
    )
    with pytest.raises(sut.SecondaryMergeContractError, match="edited after"):
        capture.run_merge()
