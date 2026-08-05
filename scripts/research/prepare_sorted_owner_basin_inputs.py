#!/usr/bin/env python3
"""Author deterministic, CPU-only inputs for sorted owner-basin scoring.

The command joins sealed Task-0, Task-2, and Task-6 evidence with the frozen
sentinel/control registries and an exact model/tokenizer/runtime receipt.  It
writes the three inputs consumed by ``build_sorted_owner_basin_candidates.py``:

* ``landscape-decision-rules.json``;
* ``owner-context-ledger.jsonl``; and
* ``candidate-bank-seeds.json``.

This module never imports torch, opens a model, performs a forward pass, or
reads candidate scores.  Canonical descriptions are encoded once with the
tokenizer named by the frozen identity receipt.  Exact-history token ids are
copied from Task 6 and are never decoded or re-tokenized.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Protocol

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.build_sorted_owner_basin_candidates import (  # noqa: E402
    BANK_SEEDS_SCHEMA_VERSION,
    COORDINATE_SPACE_NAME,
    MATERIALIZER_RULES_SCHEMA_VERSION,
    OFFICIAL_COCO_CATEGORY_IDS,
    OFFICIAL_COCO_NAMESPACE,
    OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
    canonical_json_bytes,
    sha256_file,
    sha256_json,
)
from scripts.research.build_sorted_owner_basin_cohorts import (  # noqa: E402
    COHORT_ASSIGNMENT_SCHEMA_VERSION,
    SAMPLING_SUPPORT_SCHEMA_VERSION,
)
from scripts.research.build_sorted_owner_basin_contexts import (  # noqa: E402
    CONTEXT_SCHEMA_VERSION,
)
from scripts.research.sorted_owner_basin_landscape import (  # noqa: E402
    RULES_SCHEMA_VERSION,
    SEMANTIC_CORE_OUTER_EXECUTION_FIELDS,
    SEMANTIC_CORE_SCHEMA_VERSION,
    build_semantic_core_payload,
)


RECEIPT_SCHEMA_VERSION = "sorted_owner_basin_input_plan_receipt.v1"
CENSUS_MANIFEST_SCHEMA_VERSION = "sorted-owner-basin-census-artifact-manifest.v2"
OWNER_LEDGER_SOURCE_SCHEMA_VERSION = "sorted-owner-basin-owner-ledger.v2"
SENTINEL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-sentinel-registry.v2"
CONTROL_REGISTRY_SCHEMA_VERSION = "sorted-owner-basin-control-registry.v2"
SENTINEL_CONFIRMATION_SCHEMA_VERSION = (
    "sorted-owner-basin-sentinel-selection-confirmation-receipt.v2"
)
OWNER_TRAJECTORY_MATRIX_SCHEMA_VERSION = "sorted-owner-basin-owner-trajectory-matrix.v2"
FOIL_REVIEW_SCHEMA_VERSION = "sorted-owner-basin-foil-geometry-review.v1"
FOIL_REVIEW_STATUS = "lead_reviewed_and_frozen_before_landscape_scoring"
ORIGINAL_SENTINEL_SELECTION_RECEIPT_SHA256 = (
    "2af9a0575e0d11ed4b04a78b7c217e8c5dfd9843a96f14974fa37fdaea8458a8"
)
IDENTITY_RECEIPT_SCHEMA_VERSION = "sorted-owner-basin-runtime-identity.v1"
NON_C_SMOKE_FREEZE_SCHEMA_VERSION = "sorted_owner_basin_non_c_smoke_freeze_receipt.v1"

STRUCTURAL_STATUSES = frozenset({"draft_pre_smoke", "sealed_non_c_smoke"})
CONTEXT_STATUSES = frozenset({"admitted", "admitted_reviewed_candidate_only"})
UNRESOLVED_CONTEXT_STATUS = "grounding_or_order_conditioned_accessibility_unresolved"
UNRESOLVED_CONTROL_ELIGIBILITY = "not_control_diagnostic_eligible"
CONTEXT_KINDS = frozenset(
    {
        "P_pre",
        "P_post",
        "root",
        "natural_stop",
        "reference_scan_position",
        "B2_before",
        "B2_after",
    }
)
COHORTS = frozenset(
    {
        "greedy_strict_present",
        "strict_rescued",
        "loose_only_b1",
        "no_free_spatial_support",
        "positive_overlap_neutral",
        "strict_ambiguity_neutral",
    }
)
NEUTRAL_COHORTS = frozenset({"positive_overlap_neutral", "strict_ambiguity_neutral"})
CONTROL_ROLES = frozenset(
    {
        "strict_visible_true_positive",
        "b1_loose_only",
        "strict_rescued",
        "b2_distinct_same_description_pair",
    }
)
TASK4_CONTROL_ROLES = frozenset(
    {
        "strict_visible_true_positive",
        "b1_loose_only",
        "b2_distinct_same_description_pair",
    }
)
TASK4_PRODUCTION_CONTROL_CONTEXT_IDS = frozenset(
    {
        "ctx:control-owner:gt:7511:22:root",
        "ctx:control-owner:gt:7511:26:root",
        "ctx:B2:control:smoke:b2:7511:22-to-26:B2_before",
        "ctx:B2:control:smoke:b2:7511:22-to-26:B2_after",
    }
)
TASK4_PRODUCTION_CONTEXT_CONTROL_IDS = {
    "ctx:control-owner:gt:7511:22:root": frozenset(
        {
            "control:smoke:strict-visible:7511:22",
            "control:smoke:b2:7511:22-to-26",
        }
    ),
    "ctx:control-owner:gt:7511:26:root": frozenset(
        {
            "control:smoke:b1:7511:26",
            "control:smoke:b2:7511:22-to-26",
        }
    ),
    "ctx:B2:control:smoke:b2:7511:22-to-26:B2_before": frozenset(
        {"control:smoke:b2:7511:22-to-26"}
    ),
    "ctx:B2:control:smoke:b2:7511:22-to-26:B2_after": frozenset(
        {"control:smoke:b2:7511:22-to-26"}
    ),
}
TASK4_PRODUCTION_CALIBRATION_CONTROLS = {
    "control:smoke:strict-visible:7511:22": "strict_visible_true_positive",
    "control:smoke:b1:7511:26": "b1_loose_only",
}
TASK4_PRODUCTION_SENTINEL_CONTEXT_ID = (
    "ctx:sentinel:sentinel:far-person:7511:10:root"
)
BANK_ORDER = (
    "target",
    "covered",
    "background",
    "scan",
    "part",
    "whole",
    "merged",
)
REQUIRED_FOIL_BANKS = frozenset({"background", "scan"})
REVIEW_ONLY_BANKS = frozenset({"part", "whole", "merged"})
FOIL_BANKS = frozenset(BANK_ORDER[1:])
BANK_ALIASES = {
    "covered_same_description": "covered",
    "equal_size_background": "background",
    "sorted_scan": "scan",
    "reviewed_part": "part",
    "reviewed_whole": "whole",
    "reviewed_merged": "merged",
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COORD_TOKEN_RE = re.compile(r"^<\|coord_(\d+)\|>$")
_GT_OWNER_RE = re.compile(r"^gt:[^:]+:\d+$")
_SCORE_KEY_RE = re.compile(
    r"(?:^|_)(?:score|scores|logit|logits|logprob|logprobs|likelihood)(?:_|$)"
)


class InputPlanError(ValueError):
    """Raised before any output when a sealed input cannot be admitted."""


class ExactTokenizer(Protocol):
    def encode(self, text: str, *, add_special_tokens: bool) -> Sequence[int]: ...


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputPlanError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InputPlanError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise InputPlanError(f"{label} must be a non-empty trimmed string")
    return value


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InputPlanError(f"{label} must be an integer")
    return int(value)


def _number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise InputPlanError(f"{label} must be a finite number")
    return float(value)


def _digest(value: Any, label: str) -> str:
    result = _string(value, label)
    if not _SHA256_RE.fullmatch(result):
        raise InputPlanError(f"{label} must be a lowercase SHA-256 digest")
    return result


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        return _mapping(json.loads(path.read_text(encoding="utf-8")), label)
    except json.JSONDecodeError as exc:
        raise InputPlanError(f"{label} is not valid JSON") from exc


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            raise InputPlanError(f"{label} line {line_number} is blank")
        try:
            rows.append(_mapping(json.loads(line), f"{label} line {line_number}"))
        except json.JSONDecodeError as exc:
            raise InputPlanError(
                f"{label} line {line_number} is not valid JSON"
            ) from exc
    if not rows:
        raise InputPlanError(f"{label} must not be empty")
    return rows


def _verify_expected(path: Path, expected: str, label: str) -> str:
    expected_digest = _digest(expected, f"expected {label} digest")
    observed = sha256_file(path)
    if observed != expected_digest:
        raise InputPlanError(
            f"{label} digest is stale or does not match the supplied file"
        )
    return observed


def _bound_digest(document: Mapping[str, Any], *, field: str, label: str) -> str:
    declared = _digest(document.get(field), f"{label}.{field}")
    content = {key: value for key, value in document.items() if key != field}
    if declared != sha256_json(content):
        raise InputPlanError(f"{label}.{field} is stale")
    return declared


def _validate_non_c_smoke_freeze(
    document: Mapping[str, Any],
    *,
    semantic_core_sha256: str,
) -> dict[str, Any]:
    if document.get("schema_version") != NON_C_SMOKE_FREEZE_SCHEMA_VERSION:
        raise InputPlanError("non-C smoke freeze receipt schema_version is not recognized")
    if document.get("status") != "passed":
        raise InputPlanError("non-C smoke freeze receipt is not passed")
    control_rules_sha256 = _digest(
        document.get("control_decision_rules_sha256"),
        "non-C smoke freeze receipt.control_decision_rules_sha256",
    )
    if (
        _digest(
            document.get("semantic_core_sha256"),
            "non-C smoke freeze receipt.semantic_core_sha256",
        )
        != semantic_core_sha256
    ):
        raise InputPlanError(
            "non-C smoke freeze receipt is bound to a stale or different semantic core"
        )
    for field in (
        "control_score_artifact_sha256",
        "control_score_receipt_sha256",
        "control_summary_sha256",
        "control_summary_receipt_sha256",
        "calibration_receipt_sha256",
    ):
        _digest(document.get(field), f"non-C smoke freeze receipt.{field}")
    if document.get("independent_reconstruction") != "passed":
        raise InputPlanError(
            "non-C smoke freeze receipt lacks independent reconstruction"
        )
    gates = _mapping(document.get("gates"), "non-C smoke freeze receipt.gates")
    if (
        gates.get("representative_positive_control") != "passed"
        or gates.get("mandatory_cache_parity") != "passed"
        or gates.get("b2_reviewed_pair") != "passed"
        or gates.get("free_surface_executed_raw_only") != "passed"
    ):
        raise InputPlanError(
            "non-C smoke freeze receipt lacks the positive-control, parity, or B2 gate"
        )
    if document.get("c_outcomes_read") is not False:
        raise InputPlanError("non-C smoke freeze receipt is not C-blind")
    return {
        "control_decision_rules_sha256": control_rules_sha256,
        "semantic_core_sha256": semantic_core_sha256,
    }


def _source_task0_digest(document: Mapping[str, Any], label: str) -> str:
    sources = _mapping(document.get("source_digests"), f"{label}.source_digests")
    value = sources.get("task0_census_artifact_manifest_sha256")
    field = "task0_census_artifact_manifest_sha256"
    if value is None:
        value = sources.get("task0_artifact_manifest")
        field = "task0_artifact_manifest"
    if value is None:
        value = sources.get("task0_census_artifact_manifest")
        field = "task0_census_artifact_manifest"
    return _digest(value, f"{label}.source_digests.{field}")


def _resolved_declared_path(
    declaration: Mapping[str, Any],
    *,
    field: str,
    relative_to: Path,
    label: str,
) -> Path:
    declared = Path(_string(declaration.get(field), f"{label}.{field}"))
    if declared.is_absolute():
        return declared.expanduser().resolve(strict=True)
    return (relative_to / declared).resolve(strict=True)


def _require_digest_fields(
    document: Mapping[str, Any],
    expected: Mapping[str, str],
    *,
    label: str,
) -> None:
    for field, expected_digest in expected.items():
        observed = _digest(document.get(field), f"{label}.{field}")
        if observed != expected_digest:
            raise InputPlanError(f"{label}.{field} disagrees with final Task-0 v2")


def _assert_no_score_fields(value: Any, label: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _SCORE_KEY_RE.search(str(key)):
                raise InputPlanError(
                    f"{label} contains forbidden score-dependent field {key!r}"
                )
            _assert_no_score_fields(item, f"{label}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _assert_no_score_fields(item, f"{label}[{index}]")


def _normalize_token_ids(value: Any, label: str, *, allow_empty: bool) -> list[int]:
    items = _sequence(value, label)
    if not allow_empty and not items:
        raise InputPlanError(f"{label} must not be empty")
    result = [_integer(item, f"{label} item") for item in items]
    if any(item < 0 for item in result):
        raise InputPlanError(f"{label} must contain non-negative token ids")
    return result


def _coord_bin(value: Any, label: str) -> int:
    if isinstance(value, str):
        match = _COORD_TOKEN_RE.fullmatch(value)
        if match is None:
            raise InputPlanError(f"{label} is not an official coordinate token")
        value = int(match.group(1))
    result = _integer(value, label)
    if not 0 <= result <= 999:
        raise InputPlanError(f"{label} is outside the official 0..999 coordinate bins")
    return result


def _box(value: Any, label: str) -> list[int]:
    items = _sequence(value, label)
    if len(items) != 4:
        raise InputPlanError(f"{label} must contain four coordinate bins")
    result = [_coord_bin(item, f"{label}[{index}]") for index, item in enumerate(items)]
    if result[2] <= result[0] or result[3] <= result[1]:
        raise InputPlanError(f"{label} must satisfy x2>x1 and y2>y1")
    return result


def _content_id(prefix: str, payload: Any) -> str:
    return f"{prefix}:sha256:{sha256_json(payload)}"


def _is_decision_neutral(owner: Mapping[str, Any]) -> bool:
    owner_id = str(owner.get("gt_owner_id"))
    eligibility = _mapping(
        owner.get("decision_eligibility"), f"owner {owner_id}.decision_eligibility"
    )
    paired = _mapping(
        eligibility.get("greedy_k16_paired"),
        f"owner {owner_id}.decision_eligibility.greedy_k16_paired",
    )
    eligible = paired.get("eligible")
    if not isinstance(eligible, bool):
        raise InputPlanError(
            f"owner {owner_id!r} lacks boolean paired decision eligibility"
        )
    return not eligible


def _load_tokenizer(identity: Mapping[str, Any]) -> ExactTokenizer:
    tokenizer_path = (
        Path(_string(identity.get("path"), "identity.tokenizer.path"))
        .expanduser()
        .resolve(strict=True)
    )
    try:
        from transformers import AutoTokenizer
    except (
        ImportError
    ) as exc:  # pragma: no cover - exercised only by direct CLI execution.
        raise InputPlanError(
            "transformers is required to load the frozen tokenizer"
        ) from exc
    return AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, trust_remote_code=True
    )


def _parse_identity(document: Mapping[str, Any]) -> dict[str, Any]:
    if document.get("schema_version") != IDENTITY_RECEIPT_SCHEMA_VERSION:
        raise InputPlanError("identity receipt schema_version is not recognized")
    if document.get("status") != "frozen":
        raise InputPlanError("identity receipt must have status 'frozen'")
    _bound_digest(document, field="receipt_digest", label="identity receipt")
    tokenizer = _mapping(document.get("tokenizer"), "identity receipt.tokenizer")
    model = _mapping(document.get("model"), "identity receipt.model")
    runtime = _mapping(document.get("runtime"), "identity receipt.runtime")
    vocabulary = _mapping(
        document.get("coordinate_vocabulary"), "identity receipt.coordinate_vocabulary"
    )
    if vocabulary.get("coordinate_min") != 0 or vocabulary.get("coordinate_max") != 999:
        raise InputPlanError("production coordinate vocabulary must be exactly 0..999")
    token_start = _integer(
        vocabulary.get("token_id_start"), "coordinate_vocabulary.token_id_start"
    )
    token_end = _integer(
        vocabulary.get("token_id_end_exclusive"),
        "coordinate_vocabulary.token_id_end_exclusive",
    )
    if token_end - token_start != 1000:
        raise InputPlanError("coordinate vocabulary must bind exactly 1000 token ids")
    model_vocab_size = _integer(
        document.get("model_vocab_size"), "identity receipt.model_vocab_size"
    )
    if not 0 <= token_start < token_end <= model_vocab_size:
        raise InputPlanError(
            "coordinate token interval is outside the model vocabulary"
        )
    schema_tokens_raw = _mapping(
        document.get("schema_tokens"), "identity receipt.schema_tokens"
    )
    schema_token_keys = {
        "object_ref_start_token_id",
        "object_ref_end_token_id",
        "box_start_token_id",
        "box_end_token_id",
    }
    if set(schema_tokens_raw) != schema_token_keys:
        raise InputPlanError("identity receipt.schema_tokens has unrecognized keys")
    schema_tokens = {
        key: _integer(schema_tokens_raw[key], f"schema_tokens.{key}")
        for key in sorted(schema_token_keys)
    }
    return {
        "tokenizer": dict(tokenizer),
        "vocabulary_attestation": {
            "tokenizer_identity_sha256": _digest(
                tokenizer.get("identity_sha256"), "tokenizer.identity_sha256"
            ),
            "model_identity_sha256": _digest(
                model.get("identity_sha256"), "model.identity_sha256"
            ),
            "runtime_identity_sha256": _digest(
                runtime.get("identity_sha256"), "runtime.identity_sha256"
            ),
        },
        "model_vocab_size": model_vocab_size,
        "schema_tokens": schema_tokens,
        "coordinate_token_id_start": token_start,
        "coordinate_token_id_end_exclusive": token_end,
        "identity_receipt_digest": _digest(
            document.get("receipt_digest"), "identity receipt.receipt_digest"
        ),
    }


def _parse_census(
    manifest_path: Path,
    owner_ledger_path: Path,
    manifest: Mapping[str, Any],
    owner_rows: Sequence[Mapping[str, Any]],
) -> tuple[
    dict[str, dict[str, Any]],
    dict[tuple[str, int], list[int]],
    dict[str, tuple[int, int]],
    dict[str, Any],
]:
    if manifest.get("schema_version") != CENSUS_MANIFEST_SCHEMA_VERSION:
        raise InputPlanError("Task-0 census manifest schema_version is not recognized")
    artifacts = _mapping(manifest.get("artifacts"), "Task-0 census manifest.artifacts")
    owner_artifact = _mapping(
        artifacts.get("owner_ledger"), "Task-0 census owner-ledger artifact"
    )
    if _digest(
        owner_artifact.get("sha256"), "Task-0 census owner-ledger digest"
    ) != sha256_file(owner_ledger_path):
        raise InputPlanError("Task-0 census manifest has a stale owner-ledger digest")
    declared_owner_path = (
        manifest_path.parent
        / _string(owner_artifact.get("path"), "Task-0 owner-ledger path")
    ).resolve(strict=True)
    if declared_owner_path != owner_ledger_path:
        raise InputPlanError(
            "Task-0 owner-ledger path disagrees with the census manifest"
        )
    resolved_artifacts: dict[str, tuple[Path, str]] = {}
    execution_content_digest: str | None = None
    for artifact_name in (
        "execution_receipt",
        "ambiguity_receipts",
        "owner_trajectory_matrix",
    ):
        artifact = _mapping(
            artifacts.get(artifact_name), f"Task-0 census {artifact_name} artifact"
        )
        artifact_path = (
            manifest_path.parent
            / _string(artifact.get("path"), f"Task-0 {artifact_name} path")
        ).resolve(strict=True)
        artifact_digest = _digest(
            artifact.get("sha256"), f"Task-0 {artifact_name} digest"
        )
        if sha256_file(artifact_path) != artifact_digest:
            raise InputPlanError(f"Task-0 {artifact_name} artifact digest is stale")
        resolved_artifacts[artifact_name] = (artifact_path, artifact_digest)
        if artifact_name == "execution_receipt":
            execution = _read_json(artifact_path, "Task-0 execution receipt")
            execution_content_digest = _bound_digest(
                execution,
                field="execution_receipt_content_sha256",
                label="Task-0 execution receipt",
            )
            if execution_content_digest != _digest(
                manifest.get("execution_receipt_content_sha256"),
                "Task-0 manifest.execution_receipt_content_sha256",
            ):
                raise InputPlanError(
                    "Task-0 execution receipt identity disagrees with the census manifest"
                )
    if execution_content_digest is None:
        raise AssertionError("Task-0 execution receipt was not validated")
    task0_digest = sha256_file(manifest_path)
    owners: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(owner_rows):
        if row.get("schema_version") != OWNER_LEDGER_SOURCE_SCHEMA_VERSION:
            raise InputPlanError(
                f"Task-0 owner-ledger row {index} has an unrecognized schema"
            )
        owner_id = _string(
            row.get("gt_owner_id"), f"Task-0 owner-ledger row {index}.gt_owner_id"
        )
        if not _GT_OWNER_RE.fullmatch(owner_id):
            raise InputPlanError(f"Task-0 owner identity {owner_id!r} is unrecognized")
        if owner_id in owners or row.get("mapped_gt_owner_id") != owner_id:
            raise InputPlanError(
                f"Task-0 owner identity {owner_id!r} is duplicate or not permanent"
            )
        category_id = _integer(
            row.get("official_coco_category_id"), f"owner {owner_id}.category_id"
        )
        if category_id not in OFFICIAL_COCO_CATEGORY_IDS:
            raise InputPlanError(
                f"owner {owner_id!r} is not in the official gapped COCO namespace"
            )
        if row.get("diagnostic_owner_id") != f"diagnostic:{owner_id}":
            raise InputPlanError(
                f"owner {owner_id!r} has an unrecognized diagnostic identity"
            )
        _is_decision_neutral(row)
        owners[owner_id] = dict(row)
    sources = _mapping(manifest.get("sources"), "Task-0 census manifest.sources")
    panel_source = _mapping(
        sources.get("panel"), "Task-0 census manifest.sources.panel"
    )
    panel_path = (
        Path(_string(panel_source.get("path"), "Task-0 panel path"))
        .expanduser()
        .resolve(strict=True)
    )
    panel_digest = _digest(panel_source.get("sha256"), "Task-0 panel digest")
    if sha256_file(panel_path) != panel_digest:
        raise InputPlanError("Task-0 panel source digest is stale")
    panel_boxes: dict[tuple[str, int], list[int]] = {}
    panel_sizes: dict[str, tuple[int, int]] = {}
    for line_number, line in enumerate(
        panel_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        row = _mapping(json.loads(line), f"panel line {line_number}")
        image_id = str(row.get("image_id"))
        width = _integer(row.get("width"), f"panel image {image_id}.width")
        height = _integer(row.get("height"), f"panel image {image_id}.height")
        if width <= 0 or height <= 0 or image_id in panel_sizes:
            raise InputPlanError(
                f"panel image {image_id!r} has invalid or duplicate dimensions"
            )
        panel_sizes[image_id] = (width, height)
        objects = _sequence(row.get("objects"), f"panel image {image_id}.objects")
        for annotation_index, raw_object in enumerate(objects):
            obj = _mapping(
                raw_object, f"panel image {image_id} object {annotation_index}"
            )
            if "bbox_2d" not in obj:
                raise InputPlanError(
                    "Task-0 panel owner lacks authoritative coordinate-bin bbox_2d"
                )
            panel_boxes[(image_id, annotation_index)] = _box(
                obj["bbox_2d"],
                f"panel image {image_id} object {annotation_index}.bbox_2d",
            )
    for owner_id, owner in owners.items():
        key = (str(owner["image_id"]), int(owner["original_annotation_index"]))
        if key not in panel_boxes:
            raise InputPlanError(
                f"owner {owner_id!r} cannot be joined to its frozen panel annotation"
            )
    paired_metrics = _mapping(
        _mapping(
            _mapping(manifest.get("primary_metrics"), "Task-0 primary_metrics").get(
                "matched_rp_1_0_k16_any_hit"
            ),
            "Task-0 K16 metrics",
        ).get("ambiguity_neutral_paired_set"),
        "Task-0 paired ambiguity-neutral metrics",
    )
    neutral_owner_ids = sorted(
        owner_id for owner_id, owner in owners.items() if _is_decision_neutral(owner)
    )
    if paired_metrics.get(
        "excluded_neutral_gt_owner_ids"
    ) != neutral_owner_ids or paired_metrics.get("owner_denominator") != len(
        owners
    ) - len(neutral_owner_ids):
        raise InputPlanError(
            "Task-0 manifest paired denominator disagrees with owner decision eligibility"
        )
    task0_chain = {
        "root": manifest_path.parent,
        "artifact_manifest_sha256": task0_digest,
        "execution_receipt_content_sha256": execution_content_digest,
        "execution_receipt_file_sha256": resolved_artifacts["execution_receipt"][1],
        "owner_ledger_sha256": _digest(
            owner_artifact.get("sha256"), "Task-0 census owner-ledger digest"
        ),
        "owner_trajectory_matrix_sha256": resolved_artifacts["owner_trajectory_matrix"][
            1
        ],
        "ambiguity_receipts_sha256": resolved_artifacts["ambiguity_receipts"][1],
        "owner_trajectory_matrix_path": resolved_artifacts["owner_trajectory_matrix"][
            0
        ],
    }
    return owners, panel_boxes, panel_sizes, task0_chain


def _parse_cohorts(
    rows: Sequence[Mapping[str, Any]], *, task0_digest: str
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("schema_version") != COHORT_ASSIGNMENT_SCHEMA_VERSION:
            raise InputPlanError(
                f"cohort assignment row {index} has an unrecognized schema"
            )
        if _source_task0_digest(row, f"cohort assignment row {index}") != task0_digest:
            raise InputPlanError("cohort assignment is bound to a stale Task-0 census")
        owner_id = _string(
            row.get("gt_owner_id"), f"cohort assignment row {index}.gt_owner_id"
        )
        if owner_id in result:
            raise InputPlanError(f"duplicate cohort assignment for {owner_id!r}")
        cohort = row.get("cohort")
        if cohort not in COHORTS:
            raise InputPlanError(
                f"cohort assignment for {owner_id!r} has an unrecognized cohort"
            )
        natural = _mapping(
            row.get("natural_spatial_support"),
            f"cohort assignment {owner_id}.natural_spatial_support",
        )
        if (
            cohort == "no_free_spatial_support"
            and natural.get("no_free_spatial_support") is not True
        ):
            raise InputPlanError(
                f"cohort assignment for {owner_id!r} contradicts its spatial-support receipt"
            )
        if cohort in NEUTRAL_COHORTS:
            eligibility = _mapping(
                row.get("primary_eligibility"),
                f"cohort assignment {owner_id}.primary_eligibility",
            )
            if (
                eligibility.get("included_in_b1_or_no_free") is not False
                or eligibility.get("included_in_repair_or_calibration") is not False
            ):
                raise InputPlanError(
                    f"neutral cohort assignment for {owner_id!r} claims decision eligibility"
                )
            if cohort == "positive_overlap_neutral":
                if (
                    natural.get("positive_overlap_neutral") is not True
                    or eligibility.get("status")
                    != "excluded_unreviewed_or_subthreshold_positive_overlap"
                ):
                    raise InputPlanError(
                        f"positive-overlap neutral cohort for {owner_id!r} lacks its exclusion contract"
                    )
            else:
                strict = _mapping(
                    row.get("primary_strict_support"),
                    f"cohort assignment {owner_id}.primary_strict_support",
                )
                if (
                    strict.get("global_ambiguity_present") is not True
                    or eligibility.get("status") != "excluded_global_strict_ambiguity"
                ):
                    raise InputPlanError(
                        f"strict-ambiguity neutral cohort for {owner_id!r} lacks its exclusion contract"
                    )
        result[owner_id] = dict(row)
    return result


def _parse_sampling(
    rows: Sequence[Mapping[str, Any]], *, task0_digest: str
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != SAMPLING_SUPPORT_SCHEMA_VERSION:
            raise InputPlanError(
                f"sampling-support row {index} has an unrecognized schema"
            )
        if _source_task0_digest(row, f"sampling-support row {index}") != task0_digest:
            raise InputPlanError("sampling support is bound to a stale Task-0 census")
        owner_id = _string(
            row.get("gt_owner_id"), f"sampling-support row {index}.gt_owner_id"
        )
        panel = _string(
            row.get("support_panel"), f"sampling-support row {index}.support_panel"
        )
        registration = _string(
            row.get("registration_id"), f"sampling-support row {index}.registration_id"
        )
        identity = (owner_id, panel, registration)
        if identity in seen:
            raise InputPlanError(
                f"duplicate sampling-support panel {panel!r} for {owner_id!r}"
            )
        seen.add(identity)
        stratum = row.get("policy_stratum")
        if stratum not in {
            "primary_rp_1.00",
            "production_rp_1.10_greedy",
            "not_supplied",
        }:
            raise InputPlanError(
                "sampling support has an unrecognized repetition-penalty stratum"
            )
        result[owner_id].append(dict(row))
    return {
        owner_id: sorted(
            items,
            key=lambda item: (str(item["support_panel"]), str(item["registration_id"])),
        )
        for owner_id, items in result.items()
    }


def _parse_contexts(
    rows: Sequence[Mapping[str, Any]], *, task0_chain: Mapping[str, Any]
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if row.get("schema_version") != CONTEXT_SCHEMA_VERSION:
            raise InputPlanError(
                f"Task-6 context row {index} has an unrecognized schema"
            )
        binding = _mapping(
            row.get("source_binding"), f"Task-6 row {index}.source_binding"
        )
        upstream = _mapping(
            binding.get("upstream_artifact_digests"),
            f"Task-6 row {index}.upstream digests",
        )
        _require_digest_fields(
            upstream,
            {
                "task0_census_artifact_manifest": task0_chain[
                    "artifact_manifest_sha256"
                ],
                "task0_execution_receipt": task0_chain["execution_receipt_file_sha256"],
                "task0_ambiguity_receipts": task0_chain["ambiguity_receipts_sha256"],
            },
            label=f"Task-6 row {index}.upstream digests",
        )
        record_type = row.get("record_type")
        if record_type != "exact_context":
            if record_type not in {
                "first_skip_candidate_receipt",
                "b2_candidate_receipt",
            }:
                raise InputPlanError(
                    f"Task-6 row {index} has an unrecognized record_type"
                )
            if row.get("exact_prefix") is not None:
                raise InputPlanError(
                    f"Task-6 audit receipt row {index} must not carry exact tokens"
                )
            continue
        context_id = _string(
            row.get("context_id"), f"Task-6 context row {index}.context_id"
        )
        if context_id in seen:
            raise InputPlanError(f"duplicate Task-6 context id {context_id!r}")
        seen.add(context_id)
        context_kind = row.get("context_kind")
        if context_kind not in CONTEXT_KINDS:
            raise InputPlanError(
                f"Task-6 context {context_id!r} has an unrecognized kind"
            )
        status = row.get("status")
        is_unresolved_control_candidate = (
            status == UNRESOLVED_CONTEXT_STATUS
            and row.get("context_kind") == "reference_scan_position"
            and row.get("eligibility") == UNRESOLVED_CONTROL_ELIGIBILITY
        )
        if status == UNRESOLVED_CONTEXT_STATUS and not is_unresolved_control_candidate:
            raise InputPlanError(
                f"Task-6 context {context_id!r} is unresolved outside the explicit control-only exception"
            )
        if status not in CONTEXT_STATUSES and not is_unresolved_control_candidate:
            raise InputPlanError(
                f"Task-6 context {context_id!r} has an unrecognized status"
            )
        source_pred_row_id = row.get("source_pred_row_id")
        if source_pred_row_id is None:
            if (
                context_kind not in {"root", "natural_stop"}
                and not is_unresolved_control_candidate
            ) or row.get("registry_id") is None:
                raise InputPlanError(
                    f"Task-6 context {context_id!r} lacks its required source prediction row"
                )
        else:
            _string(
                source_pred_row_id, f"Task-6 context {context_id}.source_pred_row_id"
            )
        if (
            row.get("policy_stratum") != "primary_rp_1.00"
            or row.get("decode_mode") != "greedy"
            or row.get("seed") != 0
        ):
            raise InputPlanError(
                f"Task-6 context {context_id!r} mixes or changes the frozen native policy"
            )
        if is_unresolved_control_candidate:
            if row.get("exact_prefix") is not None:
                raise InputPlanError(
                    f"unresolved control context {context_id!r} must not carry exact tokens"
                )
            details = _mapping(
                row.get("details"), f"unresolved control context {context_id}.details"
            )
            declared_control_ids = [
                _string(item, f"unresolved control context {context_id}.control_ids")
                for item in _sequence(
                    details.get("control_ids"),
                    f"unresolved control context {context_id}.control_ids",
                )
            ]
            if (
                not declared_control_ids
                or row.get("registry_id") not in declared_control_ids
                or details.get("control_owner_id") != row.get("gt_owner_id")
                or details.get("repair_eligible") is not False
            ):
                raise InputPlanError(
                    f"unresolved control context {context_id!r} has a stale control binding"
                )
            _string(
                details.get("unresolved_reason"),
                f"unresolved control context {context_id}.unresolved_reason",
            )
            result.append(dict(row))
            continue
        exact = _mapping(
            row.get("exact_prefix"), f"Task-6 context {context_id}.exact_prefix"
        )
        prompt = _normalize_token_ids(
            exact.get("prompt_token_ids"), "exact prompt token ids", allow_empty=False
        )
        self_prefix = _normalize_token_ids(
            exact.get("self_prefix_generated_token_ids"),
            "exact self-prefix token ids",
            allow_empty=True,
        )
        model_input = _normalize_token_ids(
            exact.get("model_input_token_ids"),
            "exact model input token ids",
            allow_empty=False,
        )
        if model_input != prompt + self_prefix:
            raise InputPlanError(
                f"Task-6 context {context_id!r} model input is not prompt+self-prefix"
            )
        digest_fields = {
            "prompt_token_ids_sha256": prompt,
            "self_prefix_generated_token_ids_sha256": self_prefix,
            "model_input_token_ids_sha256": model_input,
        }
        for field, token_ids in digest_fields.items():
            if _digest(
                exact.get(field), f"Task-6 context {context_id}.{field}"
            ) != sha256_json(token_ids):
                raise InputPlanError(
                    f"Task-6 context {context_id!r} has stale literal token ids"
                )
        result.append(
            {
                **dict(row),
                "_prompt_token_ids": prompt,
                "_self_prefix_token_ids": self_prefix,
                "_model_input_token_ids": model_input,
            }
        )
    return sorted(
        result, key=lambda item: (str(item["gt_owner_id"]), str(item["context_id"]))
    )


def _validate_candidate_review(
    *,
    control: Mapping[str, Any],
    control_path: Path,
    candidate_members: Sequence[Mapping[str, Any]],
    owners: Mapping[str, Any],
) -> str:
    binding = _mapping(
        control.get("candidate_bank_review"), "control registry.candidate_bank_review"
    )
    review_path = _resolved_declared_path(
        binding,
        field="path",
        relative_to=control_path.parent,
        label="control registry.candidate_bank_review",
    )
    review_digest = _digest(
        binding.get("sha256"), "control registry.candidate_bank_review.sha256"
    )
    if sha256_file(review_path) != review_digest:
        raise InputPlanError("candidate-bank foil review digest is stale")
    if binding.get("status") != FOIL_REVIEW_STATUS:
        raise InputPlanError(
            "candidate-bank foil review binding is not frozen pre-score"
        )
    sources = _mapping(control.get("source_digests"), "control registry.source_digests")
    if (
        _digest(
            sources.get("candidate_bank_foil_review_sha256"),
            "control registry.source_digests.candidate_bank_foil_review_sha256",
        )
        != review_digest
    ):
        raise InputPlanError(
            "control registry has a stale candidate-bank review binding"
        )

    review = _read_json(review_path, "candidate-bank foil review")
    if review.get("schema_version") != FOIL_REVIEW_SCHEMA_VERSION:
        raise InputPlanError(
            "candidate-bank foil review schema_version is not recognized"
        )
    if review.get("status") != FOIL_REVIEW_STATUS:
        raise InputPlanError("candidate-bank foil review is not frozen pre-score")
    anti_leakage = _mapping(
        review.get("anti_leakage_contract"),
        "candidate-bank foil review.anti_leakage_contract",
    )
    if (
        anti_leakage.get("landscape_scores_used_for_selection") is not False
        or anti_leakage.get("forced_continuation_scores_used_for_selection")
        is not False
    ):
        raise InputPlanError("candidate-bank foil review violates score anti-leakage")
    review_for_score_scan = dict(review)
    anti_leakage_for_score_scan = dict(anti_leakage)
    del anti_leakage_for_score_scan["landscape_scores_used_for_selection"]
    del anti_leakage_for_score_scan["forced_continuation_scores_used_for_selection"]
    review_for_score_scan["anti_leakage_contract"] = anti_leakage_for_score_scan
    _assert_no_score_fields(review_for_score_scan, "candidate-bank foil review")
    reviewed_by_id: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _sequence(
            review.get("reviewed_geometries"),
            "candidate-bank foil review.reviewed_geometries",
        )
    ):
        row = _mapping(raw, f"candidate-bank foil review row {index}")
        source_row_id = _string(
            row.get("source_row_id"),
            f"candidate-bank foil review row {index}.source_row_id",
        )
        if source_row_id in reviewed_by_id:
            raise InputPlanError(
                f"duplicate candidate-bank review row {source_row_id!r}"
            )
        reviewed_by_id[source_row_id] = row

    for index, member in enumerate(candidate_members):
        provenance = _mapping(
            member.get("provenance"), f"candidate-bank member {index}.provenance"
        )
        declared_review_path = _resolved_declared_path(
            provenance,
            field="artifact_path",
            relative_to=control_path.parent,
            label=f"candidate-bank member {index}.provenance",
        )
        if (
            declared_review_path != review_path
            or _digest(
                provenance.get("artifact_sha256"),
                f"candidate-bank member {index}.provenance.artifact_sha256",
            )
            != review_digest
        ):
            raise InputPlanError(
                f"candidate-bank member {index} is bound to a foreign review artifact"
            )
        source_row_id = _string(
            provenance.get("source_row_id"),
            f"candidate-bank member {index}.provenance.source_row_id",
        )
        try:
            reviewed = reviewed_by_id[source_row_id]
        except KeyError as exc:
            raise InputPlanError(
                f"candidate-bank member {index} references a missing review row"
            ) from exc
        owner_id = _string(
            member.get("owner_id"), f"candidate-bank member {index}.owner_id"
        )
        bank_name = BANK_ALIASES.get(
            str(member.get("bank_name")), str(member.get("bank_name"))
        )
        if (
            reviewed.get("gt_owner_id") != owner_id
            or owner_id not in owners
            or str(reviewed.get("image_id")) != str(owners[owner_id].get("image_id"))
            or reviewed.get("bank_name") != bank_name
            or _box(reviewed.get("box"), f"review row {source_row_id}.box")
            != _box(member.get("box"), f"candidate-bank member {index}.box")
            or reviewed.get("extent_submode") != member.get("extent_submode")
        ):
            raise InputPlanError(
                f"candidate-bank member {index} drifts from review row {source_row_id!r}"
            )
        reviewed_identity = reviewed.get("identity_gt_owner_id")
        if reviewed_identity is not None:
            if member.get("identity_id") != reviewed_identity or member.get(
                "identity_kind"
            ) not in {"gt_owner", "reviewed_physical_owner"}:
                raise InputPlanError(
                    f"candidate-bank member {index} identity drifts from review row {source_row_id!r}"
                )
        elif member.get("identity_kind") not in {
            "owner_neutral_foil_geometry",
            "registered_geometry",
        }:
            raise InputPlanError(
                f"candidate-bank member {index} is not owner-neutral as reviewed"
            )
    return review_digest


def _parse_registries(
    sentinel: Mapping[str, Any],
    control: Mapping[str, Any],
    confirmation: Mapping[str, Any],
    *,
    sentinel_path: Path,
    control_path: Path,
    confirmation_path: Path,
    input_digests: Mapping[str, str],
    task0_chain: Mapping[str, Any],
    owners: Mapping[str, Any],
    contract_mode: str,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    if contract_mode not in {"production", "test_fixture"}:
        raise InputPlanError("contract_mode must be 'production' or 'test_fixture'")
    if sentinel.get("schema_version") != SENTINEL_REGISTRY_SCHEMA_VERSION:
        raise InputPlanError("sentinel registry schema_version is not recognized")
    if (
        sentinel.get("selection_status")
        != "lead_frozen_before_scoring_and_reconfirmed_against_final_task0_v2"
    ):
        raise InputPlanError("sentinel registry is not sealed before scoring")
    if (
        sentinel.get("claim_scope")
        != "outcome_selected_case_studies_only_no_prevalence"
    ):
        raise InputPlanError("sentinel registry claim scope is unrecognized")
    sentinel_sources = _mapping(
        sentinel.get("source_digests"), "sentinel registry.source_digests"
    )
    if (
        Path(
            _string(
                sentinel_sources.get("task0_v2_root"),
                "sentinel registry.source_digests.task0_v2_root",
            )
        ).resolve(strict=True)
        != task0_chain["root"]
    ):
        raise InputPlanError("sentinel registry Task-0 root path is stale")
    _require_digest_fields(
        sentinel_sources,
        {
            "task0_census_artifact_manifest_sha256": task0_chain[
                "artifact_manifest_sha256"
            ],
            "task0_execution_receipt_content_sha256": task0_chain[
                "execution_receipt_content_sha256"
            ],
            "task0_execution_receipt_file_sha256": task0_chain[
                "execution_receipt_file_sha256"
            ],
            "task0_owner_ledger_sha256": task0_chain["owner_ledger_sha256"],
            "task0_owner_trajectory_matrix_sha256": task0_chain[
                "owner_trajectory_matrix_sha256"
            ],
        },
        label="sentinel registry.source_digests",
    )

    confirmation_declaration = _mapping(
        sentinel.get("selection_confirmation_receipt"),
        "sentinel registry.selection_confirmation_receipt",
    )
    if (
        _resolved_declared_path(
            confirmation_declaration,
            field="path",
            relative_to=sentinel_path.parent,
            label="sentinel registry.selection_confirmation_receipt",
        )
        != confirmation_path
    ):
        raise InputPlanError("sentinel registry confirmation receipt path is stale")
    if (
        _digest(
            confirmation_declaration.get("sha256"),
            "sentinel registry.selection_confirmation_receipt.sha256",
        )
        != input_digests["sentinel_confirmation_receipt"]
    ):
        raise InputPlanError("sentinel registry confirmation receipt digest is stale")

    original_declaration = _mapping(
        sentinel.get("selection_receipt"), "sentinel registry.selection_receipt"
    )
    original_path = _resolved_declared_path(
        original_declaration,
        field="path",
        relative_to=sentinel_path.parent,
        label="sentinel registry.selection_receipt",
    )
    original_digest = _digest(
        original_declaration.get("sha256"),
        "sentinel registry.selection_receipt.sha256",
    )
    if sha256_file(original_path) != original_digest:
        raise InputPlanError("original sentinel selection receipt digest is stale")
    if (
        contract_mode == "production"
        and original_digest != ORIGINAL_SENTINEL_SELECTION_RECEIPT_SHA256
    ):
        raise InputPlanError("original sentinel selection receipt identity changed")

    sentinel_ids: set[str] = set()
    sentinel_registry_ids: set[str] = set()
    for index, entry in enumerate(
        _sequence(sentinel.get("sentinels"), "sentinel registry.sentinels")
    ):
        item = _mapping(entry, f"sentinel {index}")
        owner_id = _string(item.get("gt_owner_id"), f"sentinel {index}.gt_owner_id")
        registry_id = _string(item.get("sentinel_id"), f"sentinel {index}.sentinel_id")
        if (
            owner_id not in owners
            or owner_id in sentinel_ids
            or registry_id in sentinel_registry_ids
        ):
            raise InputPlanError(
                f"sentinel owner {owner_id!r} is unknown or duplicated"
            )
        if _is_decision_neutral(_mapping(owners[owner_id], f"owner {owner_id}")):
            raise InputPlanError(
                f"decision-neutral ambiguity owner {owner_id!r} cannot be a sentinel"
            )
        if (
            item.get("prior_non_recovery_status")
            != "verified_primary_natural_zero_spatial_support"
        ):
            raise InputPlanError(
                f"sentinel owner {owner_id!r} lacks the frozen non-recovery status"
            )
        owner = _mapping(owners[owner_id], f"owner {owner_id}")
        if owner.get("ambiguity_receipt_ids") != []:
            raise InputPlanError(
                f"sentinel owner {owner_id!r} carries an ambiguity receipt"
            )
        sentinel_ids.add(owner_id)
        sentinel_registry_ids.add(registry_id)
    if contract_mode == "production" and len(sentinel_ids) != 6:
        raise InputPlanError(
            "production sentinel registry must contain exactly six owners"
        )

    if confirmation.get("schema_version") != SENTINEL_CONFIRMATION_SCHEMA_VERSION:
        raise InputPlanError(
            "sentinel confirmation receipt schema_version is not recognized"
        )
    if (
        confirmation.get("confirmation_status")
        != "lead_reconfirmed_against_final_task0_v2_before_landscape_scoring"
    ):
        raise InputPlanError(
            "sentinel confirmation receipt is not sealed before scoring"
        )
    if confirmation.get("claim_scope") != sentinel.get("claim_scope"):
        raise InputPlanError(
            "sentinel confirmation claim scope disagrees with registry"
        )
    anti_leakage = _mapping(
        confirmation.get("anti_leakage_contract"),
        "sentinel confirmation.anti_leakage_contract",
    )
    if (
        anti_leakage.get("landscape_scores_used_for_original_selection") is not False
        or anti_leakage.get("landscape_scores_used_for_v2_confirmation") is not False
        or anti_leakage.get("selection_membership_changed") is not False
    ):
        raise InputPlanError("sentinel confirmation violates the pre-score contract")
    confirmation_original = _mapping(
        confirmation.get("original_selection_receipt"),
        "sentinel confirmation.original_selection_receipt",
    )
    if (
        _resolved_declared_path(
            confirmation_original,
            field="path",
            relative_to=confirmation_path.parent,
            label="sentinel confirmation.original_selection_receipt",
        )
        != original_path
        or _digest(
            confirmation_original.get("sha256"),
            "sentinel confirmation.original_selection_receipt.sha256",
        )
        != original_digest
    ):
        raise InputPlanError(
            "sentinel confirmation does not preserve the original selection receipt"
        )
    confirmed_task0 = _mapping(
        confirmation.get("final_task0_v2"), "sentinel confirmation.final_task0_v2"
    )
    if (
        Path(
            _string(
                confirmed_task0.get("root"),
                "sentinel confirmation.final_task0_v2.root",
            )
        ).resolve(strict=True)
        != task0_chain["root"]
    ):
        raise InputPlanError("sentinel confirmation Task-0 root path is stale")
    _require_digest_fields(
        confirmed_task0,
        {
            "artifact_manifest_sha256": task0_chain["artifact_manifest_sha256"],
            "execution_receipt_content_sha256": task0_chain[
                "execution_receipt_content_sha256"
            ],
            "execution_receipt_file_sha256": task0_chain[
                "execution_receipt_file_sha256"
            ],
            "owner_ledger_sha256": task0_chain["owner_ledger_sha256"],
            "owner_trajectory_matrix_sha256": task0_chain[
                "owner_trajectory_matrix_sha256"
            ],
        },
        label="sentinel confirmation.final_task0_v2",
    )
    confirmed_by_owner: dict[str, Mapping[str, Any]] = {}
    for index, raw in enumerate(
        _sequence(
            confirmation.get("confirmed_sentinels"),
            "sentinel confirmation.confirmed_sentinels",
        )
    ):
        item = _mapping(raw, f"confirmed sentinel {index}")
        owner_id = _string(
            item.get("gt_owner_id"), f"confirmed sentinel {index}.gt_owner_id"
        )
        if owner_id in confirmed_by_owner:
            raise InputPlanError(f"confirmed sentinel {owner_id!r} is duplicated")
        if (
            item.get("decision_eligible") is not True
            or _integer(item.get("strict_match_count"), "strict_match_count") != 0
            or _number(
                item.get("max_semantic_compatible_iou"),
                "max_semantic_compatible_iou",
            )
            != 0.0
        ):
            raise InputPlanError(
                f"confirmed sentinel {owner_id!r} violates eligibility/non-recovery"
            )
        confirmed_by_owner[owner_id] = item
    if set(confirmed_by_owner) != sentinel_ids:
        raise InputPlanError("confirmed sentinel membership differs from registry")

    matrix_by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for index, row in enumerate(
        _read_jsonl(
            task0_chain["owner_trajectory_matrix_path"],
            "Task-0 owner trajectory matrix",
        )
    ):
        owner_id = str(row.get("gt_owner_id"))
        if owner_id not in sentinel_ids:
            continue
        if row.get("schema_version") != OWNER_TRAJECTORY_MATRIX_SCHEMA_VERSION:
            raise InputPlanError(
                f"sentinel trajectory row {index} has an unrecognized schema"
            )
        if (
            row.get("policy_stratum") != "primary_rp_1.00"
            or row.get("strict_match_presence") is not False
            or _number(
                row.get("max_semantic_compatible_iou"),
                f"sentinel trajectory row {index}.max_semantic_compatible_iou",
            )
            != 0.0
            or row.get("ambiguity_receipt_ids") != []
            or row.get("global_ambiguity_presence") is not False
            or _digest(
                row.get("execution_receipt_content_sha256"),
                f"sentinel trajectory row {index}.execution_receipt_content_sha256",
            )
            != task0_chain["execution_receipt_content_sha256"]
        ):
            raise InputPlanError(
                f"sentinel trajectory row {index} violates final Task-0 non-recovery"
            )
        matrix_by_owner[owner_id].append(row)
    expected_trajectories = {("greedy", 0)} | {
        ("sampled", seed) for seed in range(21001, 21017)
    }
    for owner_id in sorted(sentinel_ids):
        rows = matrix_by_owner[owner_id]
        observed_trajectories = {
            (str(row.get("decode_mode")), _integer(row.get("seed"), "matrix seed"))
            for row in rows
        }
        receipt = confirmed_by_owner[owner_id]
        if len(rows) != _integer(
            receipt.get("trajectory_count"),
            f"confirmed sentinel {owner_id}.trajectory_count",
        ):
            raise InputPlanError(
                f"confirmed sentinel {owner_id!r} trajectory count is stale"
            )
        if (
            contract_mode == "production"
            and observed_trajectories != expected_trajectories
        ):
            raise InputPlanError(
                f"sentinel owner {owner_id!r} lacks the frozen greedy plus K16 trajectories"
            )

    if control.get("schema_version") != CONTROL_REGISTRY_SCHEMA_VERSION:
        raise InputPlanError("control registry schema_version is not recognized")
    if (
        control.get("status")
        != "lead_frozen_before_scoring_and_resealed_to_final_task0_v2"
    ):
        raise InputPlanError("control registry is not frozen before scoring")
    control_sources = _mapping(
        control.get("source_digests"), "control registry.source_digests"
    )
    if (
        Path(
            _string(
                control_sources.get("task0_v2_root"),
                "control registry.source_digests.task0_v2_root",
            )
        ).resolve(strict=True)
        != task0_chain["root"]
    ):
        raise InputPlanError("control registry Task-0 root path is stale")
    _require_digest_fields(
        control_sources,
        {
            "task0_census_artifact_manifest_sha256": task0_chain[
                "artifact_manifest_sha256"
            ],
            "task0_execution_receipt_content_sha256": task0_chain[
                "execution_receipt_content_sha256"
            ],
            "task0_execution_receipt_file_sha256": task0_chain[
                "execution_receipt_file_sha256"
            ],
            "owner_ledger_sha256": task0_chain["owner_ledger_sha256"],
            "owner_trajectory_matrix_sha256": task0_chain[
                "owner_trajectory_matrix_sha256"
            ],
            "sentinel_selection_confirmation_receipt_sha256": input_digests[
                "sentinel_confirmation_receipt"
            ],
        },
        label="control registry.source_digests",
    )
    controls: list[dict[str, Any]] = []
    for index, entry in enumerate(
        _sequence(control.get("controls"), "control registry.controls")
    ):
        item = dict(_mapping(entry, f"control {index}"))
        if item.get("role") not in CONTROL_ROLES:
            raise InputPlanError(f"control {index} has an unrecognized role")
        owner_fields = [
            key
            for key in ("gt_owner_id", "covering_gt_owner_id", "target_gt_owner_id")
            if key in item
        ]
        if not owner_fields or any(item[key] not in owners for key in owner_fields):
            raise InputPlanError(f"control {index} references an unknown owner")
        if any(
            _is_decision_neutral(_mapping(owners[str(item[key])], f"owner {item[key]}"))
            for key in owner_fields
        ):
            raise InputPlanError(
                f"control {index} references a decision-neutral ambiguity owner"
            )
        controls.append(item)
    required_calibration_roles = {
        "strict_visible_true_positive",
        "b1_loose_only",
        "strict_rescued",
    }
    observed_calibration_roles = {str(item["role"]) for item in controls}
    missing_control_roles = sorted(
        required_calibration_roles - observed_calibration_roles
    )
    if missing_control_roles:
        raise InputPlanError(
            "control registry leaves owner-context calibration unresolved: missing roles "
            + ", ".join(missing_control_roles)
        )
    calibration_strata = sorted(
        {
            str(stratum)
            for item in controls
            for stratum in item.get("strata", [])
            if "calibration" in str(stratum)
        }
    )
    for stratum in calibration_strata:
        stratum_roles = {
            str(item["role"]) for item in controls if stratum in item.get("strata", [])
        }
        missing = sorted(required_calibration_roles - stratum_roles)
        if missing:
            raise InputPlanError(
                f"control registry calibration stratum {stratum!r} is unresolved: missing roles {missing}"
            )
    candidate_members = [
        dict(_mapping(item, f"control registry.candidate_bank_members[{index}]"))
        for index, item in enumerate(control.get("candidate_bank_members", []))
    ]
    _assert_no_score_fields(
        candidate_members, "control registry.candidate_bank_members"
    )
    _validate_candidate_review(
        control=control,
        control_path=control_path,
        candidate_members=candidate_members,
        owners=owners,
    )
    return sentinel_ids, controls, candidate_members


def _description_tokens(
    text: str, tokenizer: ExactTokenizer, *, vocab_size: int
) -> list[int]:
    token_ids = list(tokenizer.encode(text, add_special_tokens=False))
    if not token_ids or any(
        isinstance(item, bool) or not isinstance(item, int) for item in token_ids
    ):
        raise InputPlanError(
            "the frozen tokenizer returned invalid canonical-description token ids"
        )
    result = [int(item) for item in token_ids]
    if any(item < 0 or item >= vocab_size for item in result):
        raise InputPlanError(
            "canonical-description token id is outside the frozen model vocabulary"
        )
    return result


def _extent_grid(gt_box: Sequence[int]) -> list[dict[str, Any]]:
    width = int(gt_box[2]) - int(gt_box[0])
    height = int(gt_box[3]) - int(gt_box[1])
    specifications = (
        ("gt_whole", 1.0, 1.0),
        ("scale_0p80", 0.8, 1.0),
        ("scale_1p20", 1.2, 1.0),
        ("aspect_0p80", 1.0, 0.8),
        ("aspect_1p20", 1.0, 1.2),
    )
    result: list[dict[str, Any]] = []
    for extent_id, scale, aspect in specifications:
        candidate_width = max(1, min(999, round(width * scale * math.sqrt(aspect))))
        candidate_height = max(1, min(999, round(height * scale / math.sqrt(aspect))))
        result.append(
            {
                "extent_id": extent_id,
                "scale": scale,
                "aspect_ratio_multiplier": aspect,
                "width": candidate_width,
                "height": candidate_height,
                "rounding": "python_round_half_to_even_then_clamp_1_999",
            }
        )
    result.append(
        {
            "extent_id": "anchor_validity_floor",
            "scale": None,
            "aspect_ratio_multiplier": None,
            "width": 1,
            "height": 1,
            "rounding": "exact",
            "decision_use": "coverage_only_not_a_gt_whole_perturbation",
        }
    )
    return result


def _task4_calibration_control_ids(
    controls: Sequence[Mapping[str, Any]], *, contract_mode: str
) -> list[str]:
    if contract_mode == "production":
        controls_by_id = {
            _string(control.get("control_id"), "Task-4 calibration control_id"): control
            for control in controls
        }
        for control_id, expected_role in TASK4_PRODUCTION_CALIBRATION_CONTROLS.items():
            control = controls_by_id.get(control_id)
            if control is None:
                raise InputPlanError(
                    f"production Task-4 calibration registry is missing {control_id!r}"
                )
            strata = {
                _string(item, f"control {control_id}.strata")
                for item in _sequence(control.get("strata"), f"control {control_id}.strata")
            }
            if control.get("role") != expected_role or "representative_smoke" not in strata:
                raise InputPlanError(
                    f"production Task-4 calibration control {control_id!r} has a stale role or stratum"
                )
        return sorted(TASK4_PRODUCTION_CALIBRATION_CONTROLS)
    return sorted(
        str(item["control_id"])
        for item in controls
        if item.get("role") in {"strict_visible_true_positive", "b1_loose_only"}
    )


def _structural_policy(
    status: str,
    *,
    controls: Sequence[Mapping[str, Any]],
    contract_mode: str,
) -> dict[str, Any]:
    if status not in STRUCTURAL_STATUSES:
        raise InputPlanError(
            f"structural_status must be one of {sorted(STRUCTURAL_STATUSES)}"
        )
    hierarchy = {
        stratum: sorted(
            str(item["control_id"])
            for item in controls
            if stratum in item.get("strata", [])
        )
        for stratum in sorted(
            {str(value) for item in controls for value in item.get("strata", [])}
        )
    }
    calibration_control_ids = _task4_calibration_control_ids(
        controls, contract_mode=contract_mode
    )
    return {
        "structural_status": status,
        "coordinate_bins": {"min": 0, "max": 999},
        "target_anchor": {
            "margin_fraction": 0.10,
            "min_margin_bins": 2,
            "max_margin_bins": 32,
        },
        "free_search": {
            "x1_branch_budget": 64,
            "y1_branch_budget_per_x1": 32,
            "extent_branch_budget_per_anchor": 16,
            "spatial_diversification": {
                "algorithm": "deterministic_farthest_point_xy_anchor_selection",
                "tie_break": "ascending_x1_then_y1",
                "minimum_center_distance_bins": 24,
            },
        },
        "free_search_selector": {
            "x1_selection": "top_64_by_raw_x1_log_probability",
            "y1_selection": "top_32_per_x1_by_joint_raw_x1_y1_log_probability",
            "anchor_pool_selection": {
                "algorithm": "global_farthest_point_xy_over_top64_top32_pool",
                "initial_anchor": "highest_joint_raw_x1_y1_log_probability",
                "distance": "euclidean_xy_coordinate_bins",
                "iteration": "maximize_minimum_distance_to_selected_anchors",
                "tie_break": "higher_joint_raw_then_ascending_x1_then_y1",
                "stop_when_max_min_distance_below_bins": 24,
            },
            "extent_selection": {
                "x2": "top_16_valid_x2_by_raw_conditional_log_probability",
                "y2": "best_valid_y2_by_raw_conditional_log_probability_per_selected_x2",
                "validity": "x2_greater_than_x1_and_y2_greater_than_y1_within_0_999",
            },
            "null_semantics": "bounded_search_null_is_non_evidence",
        },
        "p_x1_y1_pruning": {
            "mode": "disabled",
            "threshold": None,
            "declaration": "absence-neutral: no anchor is pruned before complete conditional-y1 scoring",
        },
        "score_channels": {
            "raw_fp32": {
                "role": "primary_model_likelihood",
                "repetition_penalty": None,
            },
            "rp_1_00": {"role": "auxiliary_policy_score", "repetition_penalty": 1.0},
            "rp_1_10": {"role": "auxiliary_policy_score", "repetition_penalty": 1.10},
            "shared_forward_rule": "derive both policy views from one raw forward only at byte-identical prefixes",
        },
        "registered_sampling_policy": {
            "repetition_penalty": 1.0,
            "temperature": 0.4,
            "top_p": 0.95,
            "seeds": list(range(21001, 21017)),
            "max_new_tokens": 3084,
            "null_semantics": "finite_sampling_null_is_absence_neutral",
        },
        "calibration": {
            "algorithm": "empirical_quantile_lower.v1",
            "quantile": 0.10,
            "minimum_roles": [
                "strict_visible_true_positive",
                "b1_loose_only",
            ],
            "control_ids": calibration_control_ids,
            "metric_reducers": {
                "peak_height": "maximum_target_peak",
                "peak_prominence": "minimum_registered_foil_prominence",
            },
            "numeric_decision_threshold": None,
            "status": "algorithm_frozen_threshold_unpopulated",
        },
        "loose_support": {
            "rule_owner": "sealed_task2_cohort_assignments",
            "threshold_perturbation_required": True,
            "sample_hit_semantics": "positive_support_blocks_C",
            "sample_null_semantics": "no_evidence_for_C",
        },
        "ablation": {
            "operator": "solid_mean_rgb_target_box_replace_full_image_reencode",
            "equal_area_background_operator": "same_operator_on_reviewed_equal_area_background",
            "trial_status": "operator_and_trial_log_frozen_before_control_scoring",
            "trials": [],
        },
        "invalidation_rule": (
            "any rule, tokenizer, model, runtime, control, foil, context, cohort, sampling, or upstream "
            "digest change voids all downstream scores, calibration, and C-eligibility receipts"
        ),
        "control_hierarchy": hierarchy,
        "candidate_weights": {bank: 1.0 for bank in BANK_ORDER},
    }


def _normalize_member(
    raw: Mapping[str, Any],
    *,
    owner_id: str,
    context_id: str,
    gt_box: Sequence[int],
    owners: Mapping[str, Any],
) -> dict[str, Any]:
    if raw.get("owner_id") != owner_id or raw.get("context_id") != context_id:
        raise InputPlanError(
            "reviewed candidate-bank member is attached to a different owner/context"
        )
    bank_name = BANK_ALIASES.get(str(raw.get("bank_name")), raw.get("bank_name"))
    if bank_name not in FOIL_BANKS:
        raise InputPlanError(
            "reviewed candidate-bank member has an unrecognized foil bank"
        )
    if (
        bank_name in REVIEW_ONLY_BANKS
        and raw.get("review_status") != "reviewed_explicit_geometry"
    ):
        raise InputPlanError(
            f"{bank_name} candidates require explicit reviewed geometry"
        )
    box = _box(raw.get("box"), "reviewed candidate-bank member.box")
    if bank_name == "background":
        if (box[2] - box[0], box[3] - box[1]) != (
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
        ):
            raise InputPlanError(
                "equal-size background foil does not match target width and height"
            )
    identity_kind = raw.get("identity_kind")
    identity_id = _string(
        raw.get("identity_id"), "reviewed candidate-bank member.identity_id"
    )
    if bank_name == "covered":
        if (
            identity_kind not in {"gt_owner", "reviewed_physical_owner"}
            or identity_id not in owners
            or identity_id == owner_id
        ):
            raise InputPlanError(
                "covered-owner foil must name a distinct registered GT owner"
            )
        if (
            owners[identity_id]["normalized_description"]
            != owners[owner_id]["normalized_description"]
        ):
            raise InputPlanError(
                "covered-owner foil must have the exact canonical description"
            )
        identity_kind = "reviewed_physical_owner"
    elif bank_name in {"background", "scan"}:
        if identity_kind not in {"owner_neutral_foil_geometry", "registered_geometry"}:
            raise InputPlanError(
                "background and scan foils must remain owner-neutral geometry"
            )
        identity_kind = "registered_geometry"
    elif identity_kind not in {
        "gt_owner",
        "reviewed_extent_geometry",
        "reviewed_physical_owner",
    }:
        raise InputPlanError("reviewed extent foil has an unrecognized identity kind")
    else:
        identity_kind = "reviewed_physical_owner"
    provenance = _mapping(
        raw.get("provenance"), "reviewed candidate-bank member.provenance"
    )
    normalized_provenance = {
        "artifact_path": _string(
            provenance.get("artifact_path"), "foil provenance.artifact_path"
        ),
        "artifact_sha256": _digest(
            provenance.get("artifact_sha256"), "foil provenance.artifact_sha256"
        ),
        "source_row_id": _string(
            provenance.get("source_row_id"), "foil provenance.source_row_id"
        ),
    }
    membership_payload = {
        "bank_name": bank_name,
        "box": box,
        "context_id": context_id,
        "identity_id": identity_id,
        "identity_kind": identity_kind,
        "owner_id": owner_id,
        "provenance": normalized_provenance,
    }
    return {
        **membership_payload,
        "extent_submode": _string(
            raw.get("extent_submode"), "reviewed candidate-bank member.extent_submode"
        ),
        "source_id": _content_id("candidate-source", membership_payload),
        "foil_member_id": _content_id("foil-member", membership_payload),
        "provenance": normalized_provenance,
    }


def _validate_task4_production_control_contexts(
    contexts: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
) -> None:
    contexts_by_id = {
        _string(context.get("context_id"), "Task-4 control context.context_id"): context
        for context in contexts
    }
    if set(contexts_by_id) != TASK4_PRODUCTION_CONTROL_CONTEXT_IDS:
        raise InputPlanError(
            "production draft_pre_smoke must contain the exact authorized four image-7511 context IDs"
        )
    roles_by_control_id = {
        _string(control.get("control_id"), "Task-4 control.control_id"): _string(
            control.get("role"), "Task-4 control.role"
        )
        for control in controls
    }
    observed_control_ids: set[str] = set()
    observed_roles: set[str] = set()
    for context_id, context in sorted(contexts_by_id.items()):
        details = _mapping(
            context.get("details"), f"Task-4 control context {context_id}.details"
        )
        declared_control_ids = {
            _string(item, f"Task-4 control context {context_id}.details.control_ids")
            for item in _sequence(
                details.get("control_ids"),
                f"Task-4 control context {context_id}.details.control_ids",
            )
        }
        registry_id = _string(
            context.get("registry_id"), f"Task-4 control context {context_id}.registry_id"
        )
        bound_control_ids = declared_control_ids | {registry_id}
        expected_control_ids = TASK4_PRODUCTION_CONTEXT_CONTROL_IDS[context_id]
        if bound_control_ids != expected_control_ids:
            raise InputPlanError(
                f"production Task-4 context {context_id!r} has stale or foreign sealed control bindings"
            )
        try:
            bound_roles = {roles_by_control_id[control_id] for control_id in bound_control_ids}
        except KeyError as exc:
            raise InputPlanError(
                f"production Task-4 context {context_id!r} references an unknown sealed control"
            ) from exc
        declared_roles_raw = details.get("control_roles")
        if declared_roles_raw is not None:
            declared_roles = {
                _string(item, f"Task-4 control context {context_id}.details.control_roles")
                for item in _sequence(
                    declared_roles_raw,
                    f"Task-4 control context {context_id}.details.control_roles",
                )
            }
            if declared_roles != bound_roles:
                raise InputPlanError(
                    f"production Task-4 context {context_id!r} role bindings drift from its sealed control IDs"
                )
        observed_control_ids.update(bound_control_ids)
        observed_roles.update(bound_roles)
    expected_control_ids = {
        "control:smoke:strict-visible:7511:22",
        "control:smoke:b1:7511:26",
        "control:smoke:b2:7511:22-to-26",
    }
    if observed_control_ids != expected_control_ids or observed_roles != TASK4_CONTROL_ROLES:
        raise InputPlanError(
            "production Task-4 four-context plan does not reconstruct the exact smoke control registry"
        )


def _validate_task4_production_sentinel_contexts(
    contexts: Sequence[Mapping[str, Any]],
) -> None:
    if len(contexts) != 1:
        raise InputPlanError(
            "production sealed_non_c_smoke must contain exactly one sentinel context"
        )
    context = contexts[0]
    if (
        context.get("context_id") != TASK4_PRODUCTION_SENTINEL_CONTEXT_ID
        or context.get("gt_owner_id") != "gt:7511:10"
        or context.get("registry_id") != "sentinel:far-person:7511:10"
        or context.get("context_kind") != "root"
        or context.get("eligibility") != "sentinel_diagnostic_only"
    ):
        raise InputPlanError(
            "production sealed_non_c_smoke must pin the exact far-person 7511:10 root sentinel"
        )


def _build_documents(
    *,
    owners: Mapping[str, Mapping[str, Any]],
    panel_boxes: Mapping[tuple[str, int], list[int]],
    panel_sizes: Mapping[str, tuple[int, int]],
    cohorts: Mapping[str, Mapping[str, Any]],
    sampling: Mapping[str, Sequence[Mapping[str, Any]]],
    contexts: Sequence[Mapping[str, Any]],
    sentinel_ids: set[str],
    controls: Sequence[Mapping[str, Any]],
    candidate_members: Sequence[Mapping[str, Any]],
    identity: Mapping[str, Any],
    tokenizer: ExactTokenizer,
    structural_status: str,
    contract_mode: str,
    upstream_digests: Mapping[str, str],
    sealed_inputs: Mapping[str, str],
    include_context_ids: Sequence[str],
    non_c_smoke_freeze_receipt: Mapping[str, Any] | None,
    non_c_smoke_freeze_receipt_sha256: str | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    control_owner_ids = {
        str(item[key])
        for item in controls
        for key in ("gt_owner_id", "covering_gt_owner_id", "target_gt_owner_id")
        if key in item
    }
    control_ids = {str(item["control_id"]) for item in controls}
    task4_control_ids = {
        str(item["control_id"])
        for item in controls
        if item.get("role") in TASK4_CONTROL_ROLES
    }
    requested_context_ids = [
        _string(item, "include_context_ids item") for item in include_context_ids
    ]
    if len(requested_context_ids) != len(set(requested_context_ids)):
        raise InputPlanError("include_context_ids contains a duplicate context id")
    all_context_ids = {str(context.get("context_id")) for context in contexts}
    resolved_contexts: list[Mapping[str, Any]] = []
    skipped_repair_context_ids: list[str] = []
    skipped_unresolved_control_context_ids: list[str] = []
    for context in contexts:
        owner_id = _string(context.get("gt_owner_id"), "Task-6 context.gt_owner_id")
        if owner_id not in owners:
            raise InputPlanError(
                f"Task-6 context references unknown owner {owner_id!r}"
            )
        if _is_decision_neutral(owners[owner_id]):
            continue
        if owner_id not in cohorts or owner_id not in sampling:
            raise InputPlanError(
                f"owner-context {owner_id!r}/{context['context_id']} lacks Task-2 cohort or sampling support"
            )
        owner = owners[owner_id]
        if (
            str(context.get("image_id")) != str(owner["image_id"])
            or str(cohorts[owner_id].get("image_id")) != str(owner["image_id"])
            or cohorts[owner_id].get("diagnostic_owner_id")
            != owner["diagnostic_owner_id"]
            or any(
                str(row.get("image_id")) != str(owner["image_id"])
                for row in sampling[owner_id]
            )
        ):
            raise InputPlanError(
                f"owner-context {owner_id!r}/{context['context_id']} has mixed stable identities"
            )
        eligibility = _string(
            context.get("eligibility"), f"context {context['context_id']}.eligibility"
        )
        if context.get("status") == UNRESOLVED_CONTEXT_STATUS:
            details = _mapping(
                context.get("details"),
                f"unresolved control context {context['context_id']}.details",
            )
            declared_control_ids = {
                str(item)
                for item in _sequence(
                    details.get("control_ids"),
                    f"unresolved control context {context['context_id']}.control_ids",
                )
            }
            if (
                eligibility != UNRESOLVED_CONTROL_ELIGIBILITY
                or owner_id not in control_owner_ids
                or str(context.get("registry_id")) not in control_ids
                or not declared_control_ids
                or not declared_control_ids <= control_ids
            ):
                raise InputPlanError(
                    f"unresolved context {context['context_id']!r} is not bound to sealed controls"
                )
            skipped_unresolved_control_context_ids.append(str(context["context_id"]))
            continue
        if eligibility.startswith("not_"):
            continue
        if context.get("context_kind") in {"P_pre", "P_post"}:
            if (
                context.get("status") != "admitted"
                or eligibility
                != "clean_first_skip_repair_candidate_pending_native_replay"
                or context.get("registry_id") is not None
            ):
                raise InputPlanError(
                    f"owner-context {owner_id!r}/{context['context_id']} violates the clean Task7/8 repair-context contract"
                )
            skipped_repair_context_ids.append(str(context["context_id"]))
            continue
        if owner_id not in sentinel_ids | control_owner_ids:
            raise InputPlanError(
                f"owner-context {owner_id!r}/{context['context_id']} is not in a sealed sentinel/control role"
            )
        registry_id = context.get("registry_id")
        if registry_id is None or (
            (
                context.get("context_kind") in {"B2_before", "B2_after"}
                or eligibility == "control_diagnostic_only"
            )
            and str(registry_id) not in control_ids
        ):
            raise InputPlanError(
                f"owner-context {owner_id!r}/{context['context_id']} lacks its sealed registry binding"
            )
        registry_id_string = str(registry_id)
        if structural_status == "draft_pre_smoke":
            if contract_mode == "production":
                if str(context["context_id"]) not in TASK4_PRODUCTION_CONTROL_CONTEXT_IDS:
                    continue
            elif registry_id_string not in task4_control_ids:
                continue
        else:
            if (
                contract_mode == "production"
                and str(context["context_id"])
                != TASK4_PRODUCTION_SENTINEL_CONTEXT_ID
            ):
                continue
            if (
                owner_id not in sentinel_ids
                or registry_id_string in control_ids
                or eligibility != "sentinel_diagnostic_only"
            ):
                continue
        resolved_contexts.append(context)
    resolved_by_id = {
        str(context["context_id"]): context for context in resolved_contexts
    }
    if requested_context_ids:
        unknown = sorted(set(requested_context_ids) - all_context_ids)
        if unknown:
            raise InputPlanError(
                "include_context_ids contains unknown context ids: "
                + ", ".join(unknown)
            )
        requested_repairs = sorted(
            set(requested_context_ids) & set(skipped_repair_context_ids)
        )
        if requested_repairs:
            raise InputPlanError(
                "include_context_ids contains Task7/8 repair-only contexts: "
                + ", ".join(requested_repairs)
            )
        requested_unresolved = sorted(
            set(requested_context_ids) & set(skipped_unresolved_control_context_ids)
        )
        if requested_unresolved:
            raise InputPlanError(
                "include_context_ids contains unresolved control contexts: "
                + ", ".join(requested_unresolved)
            )
        ineligible = sorted(set(requested_context_ids) - set(resolved_by_id))
        if ineligible:
            raise InputPlanError(
                "include_context_ids contains registry-ineligible contexts: "
                + ", ".join(ineligible)
            )
        selected_contexts = [
            resolved_by_id[context_id] for context_id in sorted(requested_context_ids)
        ]
    else:
        selected_contexts = [
            resolved_by_id[context_id] for context_id in sorted(resolved_by_id)
        ]
    if not selected_contexts:
        raise InputPlanError(
            f"no resolved {structural_status} owner-context is eligible for input preparation"
        )
    selected_registry_ids = {str(context["registry_id"]) for context in selected_contexts}
    if structural_status == "draft_pre_smoke":
        if contract_mode == "production":
            _validate_task4_production_control_contexts(selected_contexts, controls)
        else:
            observed_roles = {
                str(item["role"])
                for item in controls
                if str(item["control_id"]) in selected_registry_ids
            }
            if not observed_roles <= TASK4_CONTROL_ROLES:
                raise InputPlanError(
                    "test-fixture draft_pre_smoke contains a non-Task4 control role"
                )
    else:
        selected_owner_ids = {str(context["gt_owner_id"]) for context in selected_contexts}
        if len(selected_owner_ids) != 1:
            raise InputPlanError(
                "sealed_non_c_smoke must contain exactly one blinded sentinel owner"
            )
        if contract_mode == "production":
            _validate_task4_production_sentinel_contexts(selected_contexts)
    emitted_context_ids = [str(context["context_id"]) for context in selected_contexts]
    skipped_resolved_context_ids = sorted(
        set(resolved_by_id) - set(emitted_context_ids)
    )
    context_selection_payload = {
        "mode": "explicit_context_ids"
        if requested_context_ids
        else "all_resolved_sealed",
        "requested_context_ids": sorted(requested_context_ids),
        "emitted_context_ids": emitted_context_ids,
        "skipped_resolved_context_ids": skipped_resolved_context_ids,
        "plan_membership": (
            "task4_control_only"
            if structural_status == "draft_pre_smoke"
            else "task4_sentinel_only"
        ),
    }
    context_selection_sha256 = sha256_json(context_selection_payload)
    bound_upstream_digests = {
        **upstream_digests,
        "context_selection": context_selection_sha256,
    }

    members_by_key: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in candidate_members:
        owner_id = _string(raw.get("owner_id"), "candidate-bank member.owner_id")
        context_id = _string(raw.get("context_id"), "candidate-bank member.context_id")
        if owner_id not in owners:
            raise InputPlanError(
                "candidate-bank member references an unknown target owner"
            )
        owner = owners[owner_id]
        gt_box = panel_boxes[
            (str(owner["image_id"]), int(owner["original_annotation_index"]))
        ]
        members_by_key[(owner_id, context_id)].append(
            _normalize_member(
                raw,
                owner_id=owner_id,
                context_id=context_id,
                gt_box=gt_box,
                owners=owners,
            )
        )
    all_normalized_members = sorted(
        (member for members in members_by_key.values() for member in members),
        key=lambda member: str(member["foil_member_id"]),
    )
    if contract_mode == "production":
        registered_foil_owner_ids = {
            str(member["owner_id"]) for member in all_normalized_members
        }
        missing_sentinel_foil_registries = sorted(
            sentinel_ids - registered_foil_owner_ids
        )
        if missing_sentinel_foil_registries:
            raise InputPlanError(
                "global all-six sentinel foil registry is incomplete for owners: "
                + ", ".join(missing_sentinel_foil_registries)
            )

    ledger_rows: list[dict[str, Any]] = []
    description_by_owner: dict[str, dict[str, Any]] = {}
    coordinate_bin_token_ids = [
        int(identity["coordinate_token_id_start"]) + coordinate_bin
        for coordinate_bin in range(1000)
    ]
    token_registry_payload = {
        "schema_tokens": dict(identity["schema_tokens"]),
        "coordinate_bin_to_token_id": {
            "bin_min": 0,
            "bin_max": 999,
            "token_id_start": identity["coordinate_token_id_start"],
            "token_id_end_exclusive": identity["coordinate_token_id_end_exclusive"],
            "mapping": "token_id=token_id_start+coordinate_bin",
            "coordinate_bin_token_ids": coordinate_bin_token_ids,
            "coordinate_bin_token_ids_sha256": sha256_json(coordinate_bin_token_ids),
        },
        "model_vocab_size": identity["model_vocab_size"],
        "vocabulary_attestation": dict(identity["vocabulary_attestation"]),
        "identity_receipt_digest": identity["identity_receipt_digest"],
    }
    token_registry: dict[str, Any] = {
        **token_registry_payload,
        "registry_sha256": sha256_json(token_registry_payload),
    }
    global_owner_ids = sorted(control_owner_ids | sentinel_ids)
    for owner_id in global_owner_ids:
        owner = owners[owner_id]
        text = _string(
            owner.get("normalized_description"),
            f"owner {owner_id}.normalized_description",
        )
        token_ids = _description_tokens(
            text, tokenizer, vocab_size=int(identity["model_vocab_size"])
        )
        forced_row_prefix_token_ids = [
            identity["schema_tokens"]["object_ref_start_token_id"],
            *token_ids,
            identity["schema_tokens"]["object_ref_end_token_id"],
            identity["schema_tokens"]["box_start_token_id"],
        ]
        description_by_owner[owner_id] = {
            "text": text,
            "text_sha256": sha256_json(text),
            "token_ids": token_ids,
            "token_ids_sha256": sha256_json(token_ids),
            "forced_row_prefix_through_box_start_token_ids": forced_row_prefix_token_ids,
            "forced_row_prefix_through_box_start_sha256": sha256_json(
                forced_row_prefix_token_ids
            ),
        }
    for context in selected_contexts:
        owner_id = str(context["gt_owner_id"])
        context_id = str(context["context_id"])
        owner = owners[owner_id]
        gt_box = panel_boxes[
            (str(owner["image_id"]), int(owner["original_annotation_index"]))
        ]
        members = members_by_key.get((owner_id, context_id), [])
        banks = {str(member["bank_name"]) for member in members}
        required_banks = set(REQUIRED_FOIL_BANKS)
        if context["context_kind"] in {"B2_before", "B2_after"}:
            required_banks.add("covered")
        missing = sorted(required_banks - banks)
        if missing:
            raise InputPlanError(
                f"owner-context {owner_id!r}/{context_id!r} is unresolved: missing reviewed foil banks {missing}"
            )
        description = description_by_owner[owner_id]
        exact_token_ids = list(context["_model_input_token_ids"])
        prompt_token_ids = list(context["_prompt_token_ids"])
        self_prefix_token_ids = list(context["_self_prefix_token_ids"])
        ledger_rows.append(
            {
                "schema_version": OWNER_CONTEXT_LEDGER_SCHEMA_VERSION,
                "owner_status": "resolved",
                "diagnostic_owner_id": str(owner["diagnostic_owner_id"]),
                "gt_owner_id": owner_id,
                "context_id": context_id,
                "prompt_prefix_token_count": len(prompt_token_ids),
                "native_repetition_penalty_stratum": 1.0,
                "source_pred_row_id": context.get("source_pred_row_id"),
                "image_id": str(owner["image_id"]),
                "image_identity": sha256_json(
                    {
                        "image_id": str(owner["image_id"]),
                        "panel_sha256": _mapping(
                            owner.get("source_digests"),
                            f"owner {owner_id}.source_digests",
                        ).get("panel"),
                    }
                ),
                "image_size": {
                    "width": panel_sizes[str(owner["image_id"])][0],
                    "height": panel_sizes[str(owner["image_id"])][1],
                },
                "ground_truth": {
                    "box": gt_box,
                    "category": {
                        "namespace": OFFICIAL_COCO_NAMESPACE,
                        "category_id": int(owner["official_coco_category_id"]),
                    },
                },
                "canonical_description": description,
                "context_tokens": {
                    "token_ids": exact_token_ids,
                    "token_ids_sha256": sha256_json(exact_token_ids),
                    "prompt_prefix_token_count": len(prompt_token_ids),
                    "prompt_token_ids_sha256": sha256_json(prompt_token_ids),
                    "self_prefix_generated_token_ids_sha256": sha256_json(
                        self_prefix_token_ids
                    ),
                    "split": {
                        "prompt": [0, len(prompt_token_ids)],
                        "self_prefix": [len(prompt_token_ids), len(exact_token_ids)],
                    },
                    "copy_semantics": "literal_task6_model_input_token_ids_no_decode_or_retokenize",
                },
                "context_provenance": {
                    "context_kind": context["context_kind"],
                    "context_status": context["status"],
                    "eligibility": context["eligibility"],
                    "source_pred_row_id": context.get("source_pred_row_id"),
                    "registry_id": context.get("registry_id"),
                    "foreign_keys": context.get("foreign_keys"),
                    "source_binding": context.get("source_binding"),
                    "review_status": (
                        "reviewed_candidate"
                        if context["status"] == "admitted_reviewed_candidate_only"
                        else "sealed_context"
                    ),
                },
                "source_review_foreign_key_lineage": {
                    "source_pred_row_id": context.get("source_pred_row_id"),
                    "review_status": (
                        "reviewed_candidate"
                        if context["status"] == "admitted_reviewed_candidate_only"
                        else "sealed_context"
                    ),
                    "registry_id": context.get("registry_id"),
                    "foreign_keys": context.get("foreign_keys"),
                    "source_binding": context.get("source_binding"),
                },
                "coordinate_space": {
                    "name": COORDINATE_SPACE_NAME,
                    "min": 0,
                    "max": 999,
                    "rounding": "round(value*extent/1000)",
                    "contract_kind": contract_mode,
                    "bin_to_extent_conversion": "round(value*extent/1000)",
                    **(
                        {
                            "non_production_reason": "deterministic CPU contract fixture"
                        }
                        if contract_mode == "test_fixture"
                        else {}
                    ),
                },
                "vocabulary_attestation": dict(identity["vocabulary_attestation"]),
                "token_registry": token_registry,
                "runtime_vocabulary_receipt": {
                    "model_vocab_size": identity["model_vocab_size"],
                    "token_registry_sha256": token_registry["registry_sha256"],
                    "identity_receipt_digest": identity["identity_receipt_digest"],
                    **dict(identity["vocabulary_attestation"]),
                },
                "context_selection_sha256": context_selection_sha256,
                "upstream_digests": dict(sorted(bound_upstream_digests.items())),
            }
        )
    ledger_rows.sort(key=lambda row: (row["diagnostic_owner_id"], row["context_id"]))

    role_payloads = {
        "target": {
            "kind": "target",
            "identity_kind": "reviewed_physical_owner",
            "allowed_bank_names": ["target"],
            "foil_set_scope": "all_pre_score_registered_foils",
        },
        **{
            bank: {
                "kind": "foil",
                "identity_kind": "registered_geometry"
                if bank in {"background", "scan"}
                else "reviewed_physical_owner",
                "allowed_bank_names": [bank],
                "foil_set_scope": "all_pre_score_registered_foils",
            }
            for bank in FOIL_BANKS
        },
    }
    foil_members = [
        {
            "foil_member_id": member["foil_member_id"],
            "diagnostic_owner_id": str(
                owners[member["owner_id"]]["diagnostic_owner_id"]
            ),
            "context_id": member["context_id"],
            "bank_name": member["bank_name"],
            "source_id": member["source_id"],
            "identity_kind": member["identity_kind"],
            "identity_id": member["identity_id"],
            "provenance": member["provenance"],
        }
        for member in all_normalized_members
    ]
    foil_members.sort(key=lambda item: item["foil_member_id"])
    foil_set_id = _content_id("foil-set", foil_members)
    role_documents = {
        bank: {
            "kind": role_payloads[bank]["kind"],
            "foil_set_id": foil_set_id,
            "identity_kind": role_payloads[bank]["identity_kind"],
            "allowed_bank_names": role_payloads[bank]["allowed_bank_names"],
        }
        for bank in BANK_ORDER
    }
    role_ids = {
        bank: _content_id("basin-role", document)
        for bank, document in role_documents.items()
    }
    proposal_weights = {bank: 1.0 for bank in BANK_ORDER}
    policy = _structural_policy(
        structural_status, controls=controls, contract_mode=contract_mode
    )
    loose_rules = {
        (
            diagnostic.get("metric"),
            diagnostic.get("lower_threshold"),
            diagnostic.get("predeclared_threshold"),
            diagnostic.get("upper_threshold"),
            diagnostic.get("cohort_rule_independent_of_diagnostic_thresholds"),
        )
        for row in cohorts.values()
        for diagnostic in [
            _mapping(
                _mapping(
                    row.get("natural_spatial_support"), "cohort natural_spatial_support"
                ).get("meaningful_loose_iou_diagnostic"),
                "cohort meaningful_loose_iou_diagnostic",
            )
        ]
    }
    if len(loose_rules) != 1:
        raise InputPlanError(
            "Task-2 loose-support rule is missing or mixed across owners"
        )
    metric, lower, predeclared, upper, independent = next(iter(loose_rules))
    policy["loose_support"].update(
        {
            "metric": metric,
            "lower_threshold": lower,
            "predeclared_threshold": predeclared,
            "upper_threshold": upper,
            "cohort_rule_independent_of_diagnostic_thresholds": independent,
        }
    )
    rules: dict[str, Any] = {
        "schema_version": RULES_SCHEMA_VERSION,
        "contract_mode": contract_mode,
        "geometry_identity": {
            "schema": "canonical_round_bin_times_extent_over_1000.v1",
            "coordinate_denominator": 1000,
        },
        "coordinate_space_contract": ledger_rows[0]["coordinate_space"],
        "structural_status": structural_status,
        "coordinate_bins": {"min": 0, "max": 999},
        "target_anchor": policy["target_anchor"],
        "extent_grid_policy": {
            "membership": "gt_whole_plus_bounded_scale_and_aspect_perturbations_plus_validity_floor",
            "scale_bounds": [0.8, 1.2],
            "aspect_ratio_multiplier_bounds": [0.8, 1.2],
            "validity": "x2>x1_and_y2>y1_within_0_999",
        },
        "bank_order": list(BANK_ORDER),
        "proposal_measures": {
            "matched_pre_score_candidate_measure": {
                "comparability_group": "matched_target_and_foil_proposal_measure",
                "normalization": "full_domain_normalized_weighted_sum",
                "bank_weights": proposal_weights,
            }
        },
        "bank_proposal_measure": {
            bank: "matched_pre_score_candidate_measure" for bank in BANK_ORDER
        },
        "spatial_clustering": {
            "owner_link_iou_min": 0.10,
            "owner_link_center_distance_max": 48.0,
            "extent_submode_iou_min": 0.75,
        },
        "shape": {
            "near_peak_logprob_delta": 0.30,
            "wide_ridge_min_candidates": 3,
            "multi_submode_min": 2,
            "merged_extent_submodes": ["reviewed_merged"],
            "scan_bank_names": ["scan"],
        },
        "declared_extent_submodes": [
            "anchor_validity_floor",
            "equal_size_background",
            "gt_whole",
            "reviewed_merged",
            "reviewed_part",
            "reviewed_whole",
            "scale_aspect_perturbation",
            "same_description_whole",
            "sorted_scan",
        ],
        "registered_basin_roles": {
            role_ids[bank]: role_documents[bank] for bank in BANK_ORDER
        },
        "bank_roles": role_ids,
        "target_bank_name": "target",
        "coco_namespace": {
            "name": OFFICIAL_COCO_NAMESPACE,
            "id_space": "official_gapped",
        },
        "p_x1_y1_pruning": policy["p_x1_y1_pruning"],
        "prominence": {"functional": "peak_height_difference"},
        "free_search": policy["free_search"],
        "free_coordinate_tree": {
            "owner": "runtime_landscape_scorer_free_tree_surface",
            "execution_status": "declared_not_executed_by_cpu_input_author",
            "candidate_membership_surface": "separate_from_restricted_candidate_bank",
            "budget": policy["free_search"],
            "selector": policy["free_search_selector"],
        },
        "task6_context_selection": {
            **context_selection_payload,
            "context_selection_sha256": context_selection_sha256,
            "selected_sentinel_control_context_ids": sorted(
                str(context["context_id"]) for context in selected_contexts
            ),
            "skipped_repair_context_ids": sorted(skipped_repair_context_ids),
            "skipped_repair_context_count": len(skipped_repair_context_ids),
            "skipped_unresolved_control_context_ids": sorted(
                skipped_unresolved_control_context_ids
            ),
            "skipped_unresolved_control_context_count": len(
                skipped_unresolved_control_context_ids
            ),
            "skipped_context_kinds": ["P_pre", "P_post"],
            "skip_reason": "clean_first_skip_repair_contexts_are_owned_by_task7_task8_not_task3_task4_landscape_scoring",
            "unresolved_control_skip_reason": "explicit_not_control_diagnostic_eligible_reference_scan_contexts_are_non_evidence",
            "unresolved_sentinel_contexts_skipped": False,
            "score_repair_contexts": False,
        },
        "score_channels": policy["score_channels"],
        "registered_sampling_policy": policy["registered_sampling_policy"],
        "calibration": policy["calibration"],
        "loose_support": policy["loose_support"],
        "ablation": policy["ablation"],
        "invalidation_rule": policy["invalidation_rule"],
        "control_hierarchy": policy["control_hierarchy"],
        "candidate_weights": policy["candidate_weights"],
        "canonical_description_source": "Task0 owner-ledger normalized_description encoded by exact frozen tokenizer",
        "canonical_alias_policy": "Task0 matcher immutable declared aliases only; no outcome-specific alias",
        "owner_canonical_descriptions": description_by_owner,
        "global_foil_role_and_description_registry": {
            "registry_scope": "all_frozen_sentinel_and_task4_control_owners_before_plan_selection",
            "owner_ids": global_owner_ids,
            "owner_canonical_descriptions": description_by_owner,
            "foil_set": {
                "foil_set_id": foil_set_id,
                "members": foil_members,
                "members_sha256": sha256_json(foil_members),
            },
            "registered_basin_roles": {
                role_ids[bank]: role_documents[bank] for bank in BANK_ORDER
            },
            "registry_source_digests": {
                key: sealed_inputs[key]
                for key in (
                    "control_registry_sha256",
                    "sentinel_registry_sha256",
                    "sentinel_selection_confirmation_receipt_sha256",
                )
            },
        },
        "model_vocab_size": identity["model_vocab_size"],
        "schema_tokens": {
            **identity["schema_tokens"],
            "coordinate_token_id_start": identity["coordinate_token_id_start"],
            "coordinate_token_id_end_exclusive": identity[
                "coordinate_token_id_end_exclusive"
            ],
        },
        "token_registry": token_registry,
        "model_tokenizer_runtime_vocabulary_identity": {
            **dict(identity["vocabulary_attestation"]),
            "identity_receipt_digest": identity["identity_receipt_digest"],
            "model_vocab_size": identity["model_vocab_size"],
            "coordinate_token_id_start": identity["coordinate_token_id_start"],
            "coordinate_token_id_end_exclusive": identity[
                "coordinate_token_id_end_exclusive"
            ],
            "schema_tokens": dict(identity["schema_tokens"]),
            "token_registry_sha256": token_registry["registry_sha256"],
        },
        "structural_row_wrapper": {
            "composition": [
                "object_ref_start_token_id",
                "canonical_description_token_ids",
                "object_ref_end_token_id",
                "box_start_token_id",
                "x1_coordinate_token_id",
                "y1_coordinate_token_id",
                "x2_coordinate_token_id",
                "y2_coordinate_token_id",
                "box_end_token_id",
            ],
            "schema_tokens": dict(identity["schema_tokens"]),
            "coordinate_bin_token_ids_sha256": token_registry[
                "coordinate_bin_to_token_id"
            ]["coordinate_bin_token_ids_sha256"],
        },
        "candidate_materializer": {
            "schema_version": MATERIALIZER_RULES_SCHEMA_VERSION,
            "status": "sealed",
            "seal_scope": "immutable_pre_score_membership_only",
            "structural_status": structural_status,
            "coordinate_space": ledger_rows[0]["coordinate_space"],
            "coco_namespace": {
                "name": OFFICIAL_COCO_NAMESPACE,
                "id_space": "official_gapped",
            },
            "target_bank_name": "target",
            "bank_roles": role_ids,
            "foil_set": {
                "foil_set_id": foil_set_id,
                "members": foil_members,
                "members_sha256": sha256_json(foil_members),
            },
            "p_x1_y1_pruning": policy["p_x1_y1_pruning"],
            "free_search_budget": policy["free_search"],
            "vocabulary_attestation": dict(identity["vocabulary_attestation"]),
            "token_registry": token_registry,
            "structural_row_wrapper": {
                "composition": [
                    "object_ref_start_token_id",
                    "canonical_description_token_ids",
                    "object_ref_end_token_id",
                    "box_start_token_id",
                    "x1_coordinate_token_id",
                    "y1_coordinate_token_id",
                    "x2_coordinate_token_id",
                    "y2_coordinate_token_id",
                    "box_end_token_id",
                ],
                "schema_tokens": dict(identity["schema_tokens"]),
                "coordinate_bin_token_ids_sha256": token_registry[
                    "coordinate_bin_to_token_id"
                ]["coordinate_bin_token_ids_sha256"],
            },
            "analysis_channels": {
                "restricted_candidate_bank": {
                    "owner": "cpu_candidate_materializer",
                    "status": "immutable_pre_score_plan_authored",
                },
                "free_coordinate_tree": {
                    "owner": "runtime_landscape_scorer_free_tree_surface",
                    "status": "declared_not_executed_by_cpu_input_author",
                },
            },
            "context_selection_sha256": context_selection_sha256,
            "upstream_digests": dict(sorted(bound_upstream_digests.items())),
        },
        "sealed_inputs": dict(sorted(sealed_inputs.items())),
        "calibration_contract": policy["calibration"],
        "scoring_contract": {
            "coordinate_token_id_start": identity["coordinate_token_id_start"],
            "coordinate_token_id_end_exclusive": identity[
                "coordinate_token_id_end_exclusive"
            ],
            "foil_set_digests": {foil_set_id: sha256_json(foil_members)},
        },
        "smoke_attestation": {
            "disposition_rules": [
                {
                    "when": {
                        "stop_rule_3_positive_control_peak": "clear",
                        "stop_rule_9_b2_matched_foils": "clear",
                    },
                    "disposition": "proceed_census_only",
                },
                {
                    "when": {"stop_rule_3_positive_control_peak": "triggered"},
                    "disposition": "hold",
                },
                {
                    "when": {
                        "stop_rule_3_positive_control_peak": "clear",
                        "stop_rule_9_b2_matched_foils": "triggered",
                    },
                    "disposition": "hold",
                },
            ]
        },
    }

    semantic_payload = build_semantic_core_payload(rules)
    semantic_core_sha256 = sha256_json(semantic_payload)
    rules["semantic_core"] = {
        "schema_version": SEMANTIC_CORE_SCHEMA_VERSION,
        "payload": semantic_payload,
        "sha256": semantic_core_sha256,
        "excluded_outer_execution_fields": list(
            SEMANTIC_CORE_OUTER_EXECUTION_FIELDS
        ),
    }
    if structural_status == "sealed_non_c_smoke":
        if (
            non_c_smoke_freeze_receipt is None
            or non_c_smoke_freeze_receipt_sha256 is None
        ):
            raise InputPlanError(
                "sealed_non_c_smoke requires --non-c-smoke-freeze-receipt"
            )
        freeze_lineage = _validate_non_c_smoke_freeze(
            non_c_smoke_freeze_receipt,
            semantic_core_sha256=semantic_core_sha256,
        )
        rules["non_c_smoke_freeze_receipt"] = {
            "sha256": non_c_smoke_freeze_receipt_sha256,
            **freeze_lineage,
        }
    elif non_c_smoke_freeze_receipt is not None:
        raise InputPlanError(
            "draft_pre_smoke must not bind a non-C smoke freeze receipt"
        )

    rules_sha256 = hashlib.sha256(canonical_json_bytes(rules) + b"\n").hexdigest()
    ledger_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in ledger_rows)
    ledger_sha256 = hashlib.sha256(ledger_bytes).hexdigest()
    seeds: list[dict[str, Any]] = []
    for ledger in ledger_rows:
        diagnostic_owner_id = ledger["diagnostic_owner_id"]
        owner_id = ledger["gt_owner_id"]
        context_id = ledger["context_id"]
        gt_box = list(ledger["ground_truth"]["box"])
        for extent in _extent_grid(gt_box):
            extent_payload = {"owner_id": owner_id, "extent": extent}
            extent_submode = (
                "gt_whole"
                if extent["extent_id"] == "gt_whole"
                else "anchor_validity_floor"
                if extent["extent_id"] == "anchor_validity_floor"
                else "scale_aspect_perturbation"
            )
            seeds.append(
                {
                    "diagnostic_owner_id": diagnostic_owner_id,
                    "context_id": context_id,
                    "bank_name": "target",
                    "role_id": role_ids["target"],
                    "source_id": _content_id("target-extent", extent_payload),
                    "box": [0, 0, extent["width"], extent["height"]],
                    "extent_submode": extent_submode,
                    "expansion": "anchor_translate",
                    "identity_kind": "reviewed_physical_owner",
                    "identity_id": owner_id,
                    "extent_grid_member": extent,
                }
            )
        for member in members_by_key[(owner_id, context_id)]:
            seeds.append(
                {
                    "diagnostic_owner_id": diagnostic_owner_id,
                    "context_id": context_id,
                    "bank_name": member["bank_name"],
                    "role_id": role_ids[member["bank_name"]],
                    "source_id": member["source_id"],
                    "box": member["box"],
                    "extent_submode": member["extent_submode"],
                    "expansion": "exact",
                    "identity_kind": member["identity_kind"],
                    "identity_id": member["identity_id"],
                    "foil_member_id": member["foil_member_id"],
                    "foil_provenance": member["provenance"],
                }
            )
    seeds.sort(
        key=lambda item: (
            item["diagnostic_owner_id"],
            item["context_id"],
            BANK_ORDER.index(item["bank_name"]),
            item["source_id"],
        )
    )
    seed_document = {
        "schema_version": BANK_SEEDS_SCHEMA_VERSION,
        "landscape_decision_rules_sha256": rules_sha256,
        "semantic_core_sha256": semantic_core_sha256,
        "owner_context_ledger_sha256": ledger_sha256,
        "foil_set_sha256": rules["candidate_materializer"]["foil_set"][
            "members_sha256"
        ],
        "membership_status": "frozen_pre_score",
        "score_dependent_fields": "forbidden",
        "context_selection_sha256": context_selection_sha256,
        "seeds": seeds,
    }
    if non_c_smoke_freeze_receipt_sha256 is not None:
        seed_document["non_c_smoke_freeze_receipt_sha256"] = (
            non_c_smoke_freeze_receipt_sha256
        )
    _assert_no_score_fields(seed_document["seeds"], "candidate-bank seeds")
    return rules, ledger_rows, seed_document


def prepare_sorted_owner_basin_inputs(
    *,
    census_manifest: str | Path,
    owner_ledger: str | Path,
    cohort_assignments: str | Path,
    sampling_support: str | Path,
    exact_contexts: str | Path,
    sentinel_registry: str | Path,
    sentinel_confirmation_receipt: str | Path,
    identity_receipt: str | Path,
    control_registry: str | Path,
    output_dir: str | Path,
    expected_sha256: Mapping[str, str],
    structural_status: str = "draft_pre_smoke",
    contract_mode: str = "production",
    include_context_ids: Sequence[str] = (),
    non_c_smoke_freeze_receipt: str | Path | None = None,
    tokenizer: ExactTokenizer | None = None,
) -> Mapping[str, Any]:
    """Validate sealed sources and write four immutable deterministic files."""

    paths = {
        "census_manifest": Path(census_manifest).expanduser().resolve(strict=True),
        "owner_ledger": Path(owner_ledger).expanduser().resolve(strict=True),
        "cohort_assignments": Path(cohort_assignments)
        .expanduser()
        .resolve(strict=True),
        "sampling_support": Path(sampling_support).expanduser().resolve(strict=True),
        "exact_contexts": Path(exact_contexts).expanduser().resolve(strict=True),
        "sentinel_registry": Path(sentinel_registry).expanduser().resolve(strict=True),
        "sentinel_confirmation_receipt": Path(sentinel_confirmation_receipt)
        .expanduser()
        .resolve(strict=True),
        "identity_receipt": Path(identity_receipt).expanduser().resolve(strict=True),
        "control_registry": Path(control_registry).expanduser().resolve(strict=True),
    }
    if structural_status == "sealed_non_c_smoke":
        if non_c_smoke_freeze_receipt is None:
            raise InputPlanError(
                "sealed_non_c_smoke requires --non-c-smoke-freeze-receipt"
            )
        paths["non_c_smoke_freeze_receipt"] = (
            Path(non_c_smoke_freeze_receipt).expanduser().resolve(strict=True)
        )
    elif non_c_smoke_freeze_receipt is not None:
        raise InputPlanError(
            "draft_pre_smoke must not receive --non-c-smoke-freeze-receipt"
        )
    if set(expected_sha256) != set(paths):
        raise InputPlanError(f"expected_sha256 must contain exactly {sorted(paths)}")
    input_digests = {
        name: _verify_expected(path, expected_sha256[name], name.replace("_", " "))
        for name, path in paths.items()
    }
    manifest = _read_json(paths["census_manifest"], "Task-0 census manifest")
    owner_rows = _read_jsonl(paths["owner_ledger"], "Task-0 owner ledger")
    owners, panel_boxes, panel_sizes, task0_chain = _parse_census(
        paths["census_manifest"], paths["owner_ledger"], manifest, owner_rows
    )
    task0_digest = task0_chain["artifact_manifest_sha256"]
    cohorts = _parse_cohorts(
        _read_jsonl(paths["cohort_assignments"], "Task-2 cohort assignments"),
        task0_digest=task0_digest,
    )
    sampling = _parse_sampling(
        _read_jsonl(paths["sampling_support"], "Task-2 sampling support"),
        task0_digest=task0_digest,
    )
    contexts = _parse_contexts(
        _read_jsonl(paths["exact_contexts"], "Task-6 exact contexts"),
        task0_chain=task0_chain,
    )
    sentinel_document = _read_json(paths["sentinel_registry"], "sentinel registry")
    confirmation_document = _read_json(
        paths["sentinel_confirmation_receipt"], "sentinel confirmation receipt"
    )
    control_document = _read_json(paths["control_registry"], "control registry")
    sentinel_ids, controls, candidate_members = _parse_registries(
        sentinel_document,
        control_document,
        confirmation_document,
        sentinel_path=paths["sentinel_registry"],
        control_path=paths["control_registry"],
        confirmation_path=paths["sentinel_confirmation_receipt"],
        input_digests=input_digests,
        task0_chain=task0_chain,
        owners=owners,
        contract_mode=contract_mode,
    )
    identity_document = _read_json(paths["identity_receipt"], "identity receipt")
    identity = _parse_identity(identity_document)
    exact_tokenizer = (
        tokenizer if tokenizer is not None else _load_tokenizer(identity["tokenizer"])
    )
    sealed_inputs = {
        **input_digests,
        "task0_census_artifact_manifest_sha256": task0_chain[
            "artifact_manifest_sha256"
        ],
        "task0_execution_receipt_content_sha256": task0_chain[
            "execution_receipt_content_sha256"
        ],
        "task0_execution_receipt_file_sha256": task0_chain[
            "execution_receipt_file_sha256"
        ],
        "owner_ledger_sha256": task0_chain["owner_ledger_sha256"],
        "owner_trajectory_matrix_sha256": task0_chain[
            "owner_trajectory_matrix_sha256"
        ],
        "sentinel_selection_confirmation_receipt_sha256": input_digests[
            "sentinel_confirmation_receipt"
        ],
        "sentinel_registry_sha256": input_digests["sentinel_registry"],
        "control_registry_sha256": input_digests["control_registry"],
        "identity_receipt_sha256": input_digests["identity_receipt"],
    }
    freeze_document = (
        None
        if "non_c_smoke_freeze_receipt" not in paths
        else _read_json(
            paths["non_c_smoke_freeze_receipt"], "non-C smoke freeze receipt"
        )
    )
    rules, ledger_rows, seeds = _build_documents(
        owners=owners,
        panel_boxes=panel_boxes,
        panel_sizes=panel_sizes,
        cohorts=cohorts,
        sampling=sampling,
        contexts=contexts,
        sentinel_ids=sentinel_ids,
        controls=controls,
        candidate_members=candidate_members,
        identity=identity,
        tokenizer=exact_tokenizer,
        structural_status=structural_status,
        contract_mode=contract_mode,
        upstream_digests=input_digests,
        sealed_inputs=sealed_inputs,
        include_context_ids=include_context_ids,
        non_c_smoke_freeze_receipt=freeze_document,
        non_c_smoke_freeze_receipt_sha256=input_digests.get(
            "non_c_smoke_freeze_receipt"
        ),
    )
    output_bytes = {
        "landscape-decision-rules.json": canonical_json_bytes(rules) + b"\n",
        "owner-context-ledger.jsonl": b"".join(
            canonical_json_bytes(row) + b"\n" for row in ledger_rows
        ),
        "candidate-bank-seeds.json": canonical_json_bytes(seeds) + b"\n",
    }
    output_receipts = {
        name: {"sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        for name, content in output_bytes.items()
    }
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "structural_status": structural_status,
        "landscape_decision_rules_sha256": output_receipts[
            "landscape-decision-rules.json"
        ]["sha256"],
        "semantic_core_sha256": rules["semantic_core"]["sha256"],
        "execution_surface": "deterministic_cpu_input_plan_no_model_forward_no_gpu",
        "inputs": dict(sorted(input_digests.items())),
        "outputs": output_receipts,
        "counts": {
            "owner_context_rows": len(ledger_rows),
            "skipped_task7_task8_repair_contexts": rules["task6_context_selection"][
                "skipped_repair_context_count"
            ],
            "skipped_unresolved_control_contexts": rules["task6_context_selection"][
                "skipped_unresolved_control_context_count"
            ],
            "candidate_bank_seeds": len(seeds["seeds"]),
            "registered_foil_members": len(
                rules["candidate_materializer"]["foil_set"]["members"]
            ),
        },
        "score_dependence": {
            "score_fields_read": False,
            "score_dependent_membership": False,
        },
        "free_coordinate_tree": {
            "owner": "runtime_landscape_scorer_free_tree_surface",
            "execution_status": "declared_not_executed_by_cpu_input_author",
            "executed": False,
            "selector": rules["free_coordinate_tree"]["selector"],
            "null_semantics": "bounded_search_null_is_non_evidence",
        },
        "task6_context_selection": rules["task6_context_selection"],
        "model_forward_executed": False,
        "gpu_used": False,
        "calibrated_numeric_decision_threshold": None,
        "c_label_emitted": False,
    }
    freeze_receipt_sha256 = input_digests.get("non_c_smoke_freeze_receipt")
    if freeze_receipt_sha256 is not None:
        receipt["non_c_smoke_freeze_receipt_sha256"] = freeze_receipt_sha256
    receipt_bytes = canonical_json_bytes(receipt) + b"\n"
    destination = Path(output_dir).expanduser().resolve()
    targets = {name: destination / name for name in output_bytes}
    targets["input-plan-receipt.json"] = destination / "input-plan-receipt.json"
    existing = [path for path in targets.values() if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite immutable output(s): "
            + ", ".join(map(str, existing))
        )
    destination.mkdir(parents=True, exist_ok=True)
    for name, content in output_bytes.items():
        with targets[name].open("xb") as handle:
            handle.write(content)
    with targets["input-plan-receipt.json"].open("xb") as handle:
        handle.write(receipt_bytes)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--census-manifest", required=True, type=Path)
    parser.add_argument("--owner-ledger", required=True, type=Path)
    parser.add_argument("--cohort-assignments", required=True, type=Path)
    parser.add_argument("--sampling-support", required=True, type=Path)
    parser.add_argument("--exact-contexts", required=True, type=Path)
    parser.add_argument("--sentinel-registry", required=True, type=Path)
    parser.add_argument("--sentinel-confirmation-receipt", required=True, type=Path)
    parser.add_argument("--identity-receipt", required=True, type=Path)
    parser.add_argument("--control-registry", required=True, type=Path)
    parser.add_argument("--non-c-smoke-freeze-receipt", type=Path)
    parser.add_argument("--expected-non-c-smoke-freeze-receipt-sha256")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--structural-status",
        choices=sorted(STRUCTURAL_STATUSES),
        default="draft_pre_smoke",
    )
    parser.add_argument(
        "--contract-mode",
        choices=("production", "test_fixture"),
        default="production",
    )
    parser.add_argument(
        "--include-context-id",
        action="append",
        default=[],
        help="emit only this exact resolved sealed Task-6 context; repeat as needed",
    )
    for name in (
        "census-manifest",
        "owner-ledger",
        "cohort-assignments",
        "sampling-support",
        "exact-contexts",
        "sentinel-registry",
        "sentinel-confirmation-receipt",
        "identity-receipt",
        "control-registry",
    ):
        parser.add_argument(f"--expected-{name}-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    expected = {
        name: getattr(args, f"expected_{name}_sha256")
        for name in (
            "census_manifest",
            "owner_ledger",
            "cohort_assignments",
            "sampling_support",
            "exact_contexts",
            "sentinel_registry",
            "sentinel_confirmation_receipt",
            "identity_receipt",
            "control_registry",
        )
    }
    if args.non_c_smoke_freeze_receipt is not None:
        if args.expected_non_c_smoke_freeze_receipt_sha256 is None:
            raise InputPlanError(
                "--non-c-smoke-freeze-receipt requires its expected SHA-256"
            )
        expected["non_c_smoke_freeze_receipt"] = (
            args.expected_non_c_smoke_freeze_receipt_sha256
        )
    receipt = prepare_sorted_owner_basin_inputs(
        census_manifest=args.census_manifest,
        owner_ledger=args.owner_ledger,
        cohort_assignments=args.cohort_assignments,
        sampling_support=args.sampling_support,
        exact_contexts=args.exact_contexts,
        sentinel_registry=args.sentinel_registry,
        sentinel_confirmation_receipt=args.sentinel_confirmation_receipt,
        identity_receipt=args.identity_receipt,
        control_registry=args.control_registry,
        output_dir=args.output_dir,
        expected_sha256=expected,
        structural_status=args.structural_status,
        contract_mode=args.contract_mode,
        include_context_ids=args.include_context_id,
        non_c_smoke_freeze_receipt=args.non_c_smoke_freeze_receipt,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
