#!/usr/bin/env python3
"""Materialize pre-score candidate and conditional-``y1`` plans on CPU only.

This is deliberately a *membership* builder, not a scorer.  It accepts a
sealed landscape-rule document, a versioned owner/context ledger, and an
explicit bank-seed document.  The resulting JSONL contains two immutable row
kinds:

* ``conditional_y1_score_plan`` records the complete conditional ``y1``
  vocabulary for each restricted ``x1`` anchor, including the terminal bin
  that cannot form a valid box; and
* ``complete_box_candidate`` records every pre-score complete-box proposal.

No model, tokenizer, torch, GPU, candidate score, or score-derived pruning is
used here.  A later scorer must honor the emitted plans and can apply only the
already-declared pruning policy.  This separation prevents a high-scoring
candidate from changing the candidate bank that is supposed to measure it.

The small input schemas below are intentionally strict.  They provide a
stable hand-off seam between the owner/context census and the later GPU
scorer, while the geometry and proposal-measure rules remain compatible with
``sorted_owner_basin_landscape.py``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, TypeGuard

if __package__ in {None, ""}:  # Allow ``python scripts/research/...py``.
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.sorted_owner_basin_landscape import (  # noqa: E402
    CoordinateBin,
    CoordinateBox,
    ExtentBankSeed,
    LandscapeRules,
    canonical_geometry_identity,
    enumerate_complete_conditional_y1,
    enumerate_target_anchor_pairs,
    expand_extent_banks,
    validate_rule_mapping,
)


SCHEMA_VERSION = "sorted_owner_basin_candidates.v2"
OWNER_CONTEXT_LEDGER_SCHEMA_VERSION = "sorted_owner_basin_owner_context_ledger.v2"
BANK_SEEDS_SCHEMA_VERSION = "sorted_owner_basin_candidate_bank_seeds.v2"
MATERIALIZER_RULES_SCHEMA_VERSION = "sorted_owner_basin_candidate_materializer_rules.v2"

COORDINATE_SPACE_NAME = "qwen_coordinate_bins"
OFFICIAL_COCO_NAMESPACE = "coco_2017_official_gapped"
PRODUCTION_BIN_TO_EXTENT_CONVERSION = "round(value*extent/1000)"

# The canonical COCO ids deliberately retain the gaps in the official
# annotation namespace.  Evaluator-local contiguous class indices are not an
# interchangeable category space.
OFFICIAL_COCO_CATEGORY_IDS = frozenset(
    (
        *range(1, 12),
        *range(13, 26),
        27,
        28,
        *range(31, 45),
        *range(46, 66),
        67,
        70,
        *range(72, 83),
        *range(84, 91),
    )
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json_bytes(value: Any) -> bytes:
    """Return the sole JSON representation used for semantic identities."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_int(value: Any) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be an array")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty, trimmed string")
    return value


def _sha256(value: Any, label: str) -> str:
    result = _string(value, label)
    if not _SHA256_RE.fullmatch(result):
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return result


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _box(value: Any, label: str) -> CoordinateBox:
    items = _sequence(value, label)
    if len(items) != 4 or not all(_is_int(item) for item in items):
        raise ValueError(f"{label} must be four integer coordinate bins")
    return CoordinateBox.from_values(*items)


def _image_id(value: Any, label: str) -> str:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{label} must be a stable string or integer image id")
    if _is_int(value):
        return str(value)
    return _string(value, label)


def _box_json(box: CoordinateBox) -> dict[str, int]:
    x1, y1, x2, y2 = box.as_tuple()
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}


def _read_json(path: Path, label: str) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not valid JSON") from exc
    return _mapping(document, label)


def _read_jsonl(path: Path, label: str) -> list[Mapping[str, Any]]:
    raw_lines = path.read_bytes().splitlines()
    if not raw_lines:
        raise ValueError(f"{label} must contain at least one JSONL row")
    rows: list[Mapping[str, Any]] = []
    for index, raw_line in enumerate(raw_lines, start=1):
        if not raw_line.strip():
            raise ValueError(f"{label} row {index} is blank")
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{label} row {index} is not valid JSON") from exc
        rows.append(_mapping(value, f"{label} row {index}"))
    return rows


def _normalized_coordinate_space(
    value: Any, *, rules: LandscapeRules, label: str
) -> dict[str, Any]:
    raw = _mapping(value, label)
    name = _string(raw.get("name"), f"{label}.name")
    if name != COORDINATE_SPACE_NAME:
        raise ValueError(
            f"{label}.name must be the declared coordinate-bin space "
            f"{COORDINATE_SPACE_NAME!r}"
        )
    coordinate_min = raw.get("min")
    coordinate_max = raw.get("max")
    if not _is_int(coordinate_min) or not _is_int(coordinate_max):
        raise ValueError(f"{label}.min and {label}.max must be integers")
    if (coordinate_min, coordinate_max) != (rules.coordinate_min, rules.coordinate_max):
        raise ValueError(f"{label} disagrees with the sealed rule coordinate range")
    contract_kind = raw.get("contract_kind")
    if contract_kind not in {"production", "test_fixture"}:
        raise ValueError(f"{label}.contract_kind must be production or test_fixture")
    if contract_kind != rules.contract_mode:
        raise ValueError(f"{label}.contract_kind disagrees with rules.contract_mode")
    conversion = _string(
        raw.get("bin_to_extent_conversion"), f"{label}.bin_to_extent_conversion"
    )
    result: dict[str, Any] = {
        "name": name,
        "min": coordinate_min,
        "max": coordinate_max,
        "contract_kind": contract_kind,
        "bin_to_extent_conversion": conversion,
    }
    if contract_kind == "production":
        if (coordinate_min, coordinate_max) != (
            0,
            999,
        ) or conversion != PRODUCTION_BIN_TO_EXTENT_CONVERSION:
            raise ValueError(
                "production coordinate contract requires bins 0..999 and "
                f"{PRODUCTION_BIN_TO_EXTENT_CONVERSION!r} conversion"
            )
        if "non_production_reason" in raw:
            raise ValueError(
                f"{label} production contract cannot claim a non-production reason"
            )
    else:
        result["non_production_reason"] = _string(
            raw.get("non_production_reason"), f"{label}.non_production_reason"
        )
    return result


def _normalized_description(value: Any, label: str) -> dict[str, Any]:
    raw = _mapping(value, label)
    text = _string(raw.get("text"), f"{label}.text")
    token_ids = _sequence(raw.get("token_ids"), f"{label}.token_ids")
    if not token_ids or not all(
        _is_int(token_id) and token_id >= 0 for token_id in token_ids
    ):
        raise ValueError(
            f"{label}.token_ids must be a non-empty list of non-negative integers"
        )
    normalized_ids = [int(token_id) for token_id in token_ids]
    if _sha256(raw.get("text_sha256"), f"{label}.text_sha256") != sha256_json(text):
        raise ValueError(f"{label}.text_sha256 does not bind the exact canonical text")
    if _sha256(raw.get("token_ids_sha256"), f"{label}.token_ids_sha256") != sha256_json(
        normalized_ids
    ):
        raise ValueError(f"{label}.token_ids_sha256 does not bind the token ids")
    forced_prefix_ids = _sequence(
        raw.get("forced_row_prefix_through_box_start_token_ids"),
        f"{label}.forced_row_prefix_through_box_start_token_ids",
    )
    if not forced_prefix_ids or not all(
        _is_int(token_id) and token_id >= 0 for token_id in forced_prefix_ids
    ):
        raise ValueError(
            f"{label}.forced_row_prefix_through_box_start_token_ids must be "
            "non-empty non-negative integers"
        )
    normalized_forced_prefix = [int(token_id) for token_id in forced_prefix_ids]
    forced_prefix_sha256 = _sha256(
        raw.get("forced_row_prefix_through_box_start_sha256"),
        f"{label}.forced_row_prefix_through_box_start_sha256",
    )
    if forced_prefix_sha256 != sha256_json(normalized_forced_prefix):
        raise ValueError(f"{label} forced row prefix digest is stale")
    result = {
        "text": text,
        "text_sha256": sha256_json(text),
        "token_ids": normalized_ids,
        "token_ids_sha256": sha256_json(normalized_ids),
        "forced_row_prefix_through_box_start_token_ids": normalized_forced_prefix,
        "forced_row_prefix_through_box_start_sha256": forced_prefix_sha256,
    }
    result["description_id"] = "canonical-description:sha256:" + sha256_json(result)
    return result


def _normalized_context_tokens(value: Any, label: str) -> dict[str, Any]:
    raw = _mapping(value, label)
    token_ids = _sequence(raw.get("token_ids"), f"{label}.token_ids")
    if not all(_is_int(token_id) and token_id >= 0 for token_id in token_ids):
        raise ValueError(f"{label}.token_ids must be non-negative integers")
    normalized_ids = [int(token_id) for token_id in token_ids]
    digest = _sha256(raw.get("token_ids_sha256"), f"{label}.token_ids_sha256")
    if digest != sha256_json(normalized_ids):
        raise ValueError(f"{label}.token_ids_sha256 does not bind the exact context")
    prompt_count = raw.get("prompt_prefix_token_count")
    if not _is_int(prompt_count) or not 0 < prompt_count <= len(normalized_ids):
        raise ValueError(
            f"{label}.prompt_prefix_token_count must split the exact context"
        )
    prompt_range = _sequence(
        _mapping(raw.get("split"), f"{label}.split").get("prompt"),
        f"{label}.split.prompt",
    )
    self_prefix_range = _sequence(
        _mapping(raw.get("split"), f"{label}.split").get("self_prefix"),
        f"{label}.split.self_prefix",
    )
    expected_prompt_range = [0, int(prompt_count)]
    expected_self_prefix_range = [int(prompt_count), len(normalized_ids)]
    if (
        list(prompt_range) != expected_prompt_range
        or list(self_prefix_range) != expected_self_prefix_range
    ):
        raise ValueError(
            f"{label}.split does not bind prompt plus self-prefix boundaries"
        )
    prompt_sha256 = _sha256(
        raw.get("prompt_token_ids_sha256"), f"{label}.prompt_token_ids_sha256"
    )
    if prompt_sha256 != sha256_json(normalized_ids[: int(prompt_count)]):
        raise ValueError(f"{label}.prompt_token_ids_sha256 is stale")
    self_prefix_sha256 = _sha256(
        raw.get("self_prefix_generated_token_ids_sha256"),
        f"{label}.self_prefix_generated_token_ids_sha256",
    )
    if self_prefix_sha256 != sha256_json(normalized_ids[int(prompt_count) :]):
        raise ValueError(f"{label}.self_prefix_generated_token_ids_sha256 is stale")
    copy_semantics = _string(raw.get("copy_semantics"), f"{label}.copy_semantics")
    if copy_semantics != "literal_task6_model_input_token_ids_no_decode_or_retokenize":
        raise ValueError(
            f"{label}.copy_semantics does not prohibit decode/re-tokenization"
        )
    return {
        "token_ids": normalized_ids,
        "token_ids_sha256": digest,
        "prompt_prefix_token_count": int(prompt_count),
        "prompt_token_ids_sha256": prompt_sha256,
        "self_prefix_generated_token_ids_sha256": self_prefix_sha256,
        "split": {
            "prompt": expected_prompt_range,
            "self_prefix": expected_self_prefix_range,
        },
        "copy_semantics": copy_semantics,
    }


def _normalized_token_registry(value: Any, label: str) -> dict[str, Any]:
    raw = _mapping(value, label)
    expected_registry_keys = {
        "schema_tokens",
        "coordinate_bin_to_token_id",
        "model_vocab_size",
        "vocabulary_attestation",
        "identity_receipt_digest",
        "registry_sha256",
    }
    if set(raw) != expected_registry_keys:
        raise ValueError(
            f"{label} must contain exactly {sorted(expected_registry_keys)}"
        )
    schema_tokens_raw = _mapping(raw.get("schema_tokens"), f"{label}.schema_tokens")
    required_schema_tokens = {
        "object_ref_start_token_id",
        "object_ref_end_token_id",
        "box_start_token_id",
        "box_end_token_id",
    }
    if set(schema_tokens_raw) != required_schema_tokens:
        raise ValueError(
            f"{label}.schema_tokens must contain exactly the frozen row-wrapper ids"
        )
    schema_tokens: dict[str, int] = {}
    for name in sorted(required_schema_tokens):
        token_id = schema_tokens_raw[name]
        if not _is_int(token_id) or token_id < 0:
            raise ValueError(
                f"{label}.schema_tokens.{name} must be a non-negative integer"
            )
        schema_tokens[name] = int(token_id)
    if len(set(schema_tokens.values())) != len(schema_tokens):
        raise ValueError(f"{label}.schema_tokens must be pairwise distinct")

    coordinate_raw = _mapping(
        raw.get("coordinate_bin_to_token_id"), f"{label}.coordinate_bin_to_token_id"
    )
    expected_coordinate_keys = {
        "bin_min",
        "bin_max",
        "token_id_start",
        "token_id_end_exclusive",
        "mapping",
        "coordinate_bin_token_ids",
        "coordinate_bin_token_ids_sha256",
    }
    if set(coordinate_raw) != expected_coordinate_keys:
        raise ValueError(
            f"{label}.coordinate_bin_to_token_id must contain exactly "
            f"{sorted(expected_coordinate_keys)}"
        )
    bin_min = coordinate_raw.get("bin_min")
    bin_max = coordinate_raw.get("bin_max")
    token_start = coordinate_raw.get("token_id_start")
    token_end = coordinate_raw.get("token_id_end_exclusive")
    if (bin_min, bin_max) != (0, 999):
        raise ValueError(
            f"{label} must freeze the full 1000-bin production coordinate registry"
        )
    if not _is_int(token_start) or not _is_int(token_end) or token_end <= token_start:
        raise ValueError(f"{label} coordinate token interval is invalid")
    if int(token_end) - int(token_start) != 1000:
        raise ValueError(
            f"{label} coordinate token interval must contain exactly 1000 ids"
        )
    explicit_ids = _sequence(
        coordinate_raw.get("coordinate_bin_token_ids"),
        f"{label}.coordinate_bin_to_token_id.coordinate_bin_token_ids",
    )
    if len(explicit_ids) != 1000 or not all(
        _is_int(token_id) for token_id in explicit_ids
    ):
        raise ValueError(
            f"{label} must carry exactly 1000 explicit coordinate token ids"
        )
    coordinate_token_ids = [int(token_id) for token_id in explicit_ids]
    if len(set(coordinate_token_ids)) != 1000:
        raise ValueError(f"{label} coordinate token ids must be unique")
    if any(
        not int(token_start) <= token_id < int(token_end)
        for token_id in coordinate_token_ids
    ):
        raise ValueError(
            f"{label} explicit coordinate token id falls outside its attested interval"
        )
    coordinate_ids_sha256 = _sha256(
        coordinate_raw.get("coordinate_bin_token_ids_sha256"),
        f"{label}.coordinate_bin_to_token_id.coordinate_bin_token_ids_sha256",
    )
    if coordinate_ids_sha256 != sha256_json(coordinate_token_ids):
        raise ValueError(f"{label} coordinate-bin token registry digest is stale")
    model_vocab_size = raw.get("model_vocab_size")
    if not _is_int(model_vocab_size) or model_vocab_size <= 0:
        raise ValueError(f"{label}.model_vocab_size must be a positive integer")
    if int(token_end) > int(model_vocab_size) or any(
        token_id >= int(model_vocab_size) for token_id in schema_tokens.values()
    ):
        raise ValueError(f"{label} token ids exceed the frozen model vocabulary")
    normalized_payload = {
        "schema_tokens": schema_tokens,
        "coordinate_bin_to_token_id": {
            "bin_min": 0,
            "bin_max": 999,
            "token_id_start": int(token_start),
            "token_id_end_exclusive": int(token_end),
            "mapping": _string(
                coordinate_raw.get("mapping"),
                f"{label}.coordinate_bin_to_token_id.mapping",
            ),
            "coordinate_bin_token_ids": coordinate_token_ids,
            "coordinate_bin_token_ids_sha256": coordinate_ids_sha256,
        },
        "model_vocab_size": int(model_vocab_size),
        "vocabulary_attestation": _normalized_vocabulary_attestation(
            raw.get("vocabulary_attestation"), f"{label}.vocabulary_attestation"
        ),
        "identity_receipt_digest": _sha256(
            raw.get("identity_receipt_digest"), f"{label}.identity_receipt_digest"
        ),
    }
    registry_sha256 = _sha256(raw.get("registry_sha256"), f"{label}.registry_sha256")
    if registry_sha256 != sha256_json(normalized_payload):
        raise ValueError(f"{label}.registry_sha256 does not bind the exact registry")
    return {**normalized_payload, "registry_sha256": registry_sha256}


def _normalized_structural_row_wrapper(
    value: Any, *, token_registry: Mapping[str, Any], label: str
) -> dict[str, Any]:
    raw = _mapping(value, label)
    expected_keys = {
        "composition",
        "schema_tokens",
        "coordinate_bin_token_ids_sha256",
    }
    if set(raw) != expected_keys:
        raise ValueError(f"{label} must contain exactly {sorted(expected_keys)}")
    expected_composition = [
        "object_ref_start_token_id",
        "canonical_description_token_ids",
        "object_ref_end_token_id",
        "box_start_token_id",
        "x1_coordinate_token_id",
        "y1_coordinate_token_id",
        "x2_coordinate_token_id",
        "y2_coordinate_token_id",
        "box_end_token_id",
    ]
    if (
        list(_sequence(raw.get("composition"), f"{label}.composition"))
        != expected_composition
    ):
        raise ValueError(
            f"{label}.composition is not the sealed complete-row token order"
        )
    schema_tokens = _mapping(raw.get("schema_tokens"), f"{label}.schema_tokens")
    if dict(schema_tokens) != token_registry["schema_tokens"]:
        raise ValueError(f"{label}.schema_tokens differs from the token registry")
    coordinate_ids_sha256 = _sha256(
        raw.get("coordinate_bin_token_ids_sha256"),
        f"{label}.coordinate_bin_token_ids_sha256",
    )
    if (
        coordinate_ids_sha256
        != token_registry["coordinate_bin_to_token_id"][
            "coordinate_bin_token_ids_sha256"
        ]
    ):
        raise ValueError(f"{label} binds a different coordinate-token registry")
    normalized = {
        "composition": expected_composition,
        "schema_tokens": dict(schema_tokens),
        "coordinate_bin_token_ids_sha256": coordinate_ids_sha256,
    }
    return {**normalized, "structural_row_wrapper_sha256": sha256_json(normalized)}


def _normalized_analysis_channels(value: Any, label: str) -> dict[str, Any]:
    raw = _mapping(value, label)
    expected_keys = {"raw_fp32", "rp_1_00", "rp_1_10", "shared_forward_rule"}
    if set(raw) != expected_keys:
        raise ValueError(f"{label} must contain exactly {sorted(expected_keys)}")
    raw_channel = _mapping(raw.get("raw_fp32"), f"{label}.raw_fp32")
    if (
        raw_channel.get("role") != "primary_model_likelihood"
        or raw_channel.get("repetition_penalty") is not None
    ):
        raise ValueError(
            f"{label}.raw_fp32 must be the unmodified model-likelihood channel"
        )
    policy_views: list[dict[str, Any]] = []
    for channel_id, expected_penalty in (("rp_1_00", 1.0), ("rp_1_10", 1.10)):
        channel = _mapping(raw.get(channel_id), f"{label}.{channel_id}")
        if channel.get("role") != "auxiliary_policy_score" or not math.isclose(
            _finite_number(
                channel.get("repetition_penalty"),
                f"{label}.{channel_id}.repetition_penalty",
            ),
            expected_penalty,
            abs_tol=1e-9,
        ):
            raise ValueError(f"{label}.{channel_id} is not the frozen policy view")
        policy_views.append(
            {
                "channel_id": channel_id,
                "role": "auxiliary_policy_score",
                "repetition_penalty": expected_penalty,
            }
        )
    shared_forward_rule = _string(
        raw.get("shared_forward_rule"), f"{label}.shared_forward_rule"
    )
    return {
        "raw_model_likelihood_channel": {
            "channel_id": "raw_fp32",
            "role": "primary_model_likelihood",
            "repetition_penalty": None,
        },
        "policy_views": policy_views,
        "shared_raw_forward": True,
        "shared_forward_rule": shared_forward_rule,
    }


def _normalized_free_search_budget(value: Any, label: str) -> dict[str, Any]:
    raw = _mapping(value, label)
    expected_counts = {
        "x1_branch_budget": 64,
        "y1_branch_budget_per_x1": 32,
        "extent_branch_budget_per_anchor": 16,
    }
    if set(raw) != {*expected_counts, "spatial_diversification"}:
        raise ValueError(f"{label} has unrecognized free-tree policy fields")
    for field, expected in expected_counts.items():
        if raw.get(field) != expected:
            raise ValueError(f"{label}.{field} must equal the frozen value {expected}")
    diversification = _mapping(
        raw.get("spatial_diversification"), f"{label}.spatial_diversification"
    )
    expected_diversification = {
        "algorithm": "deterministic_farthest_point_xy_anchor_selection",
        "tie_break": "ascending_x1_then_y1",
        "minimum_center_distance_bins": 24,
    }
    if dict(diversification) != expected_diversification:
        raise ValueError(
            f"{label}.spatial_diversification differs from the frozen policy"
        )
    return {**expected_counts, "spatial_diversification": expected_diversification}


def _normalized_digests(value: Any, label: str) -> dict[str, str]:
    raw = _mapping(value, label)
    if not raw:
        raise ValueError(f"{label} must not be empty")
    result = {
        _string(name, f"{label} key"): _sha256(digest, f"{label}.{name}")
        for name, digest in raw.items()
    }
    if len(result) != len(raw):
        raise AssertionError("mapping keys unexpectedly collapsed")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class LedgerRow:
    diagnostic_owner_id: str
    gt_owner_id: str
    context_id: str
    image_id: str
    image_identity: str
    image_width: int
    image_height: int
    gt_box: CoordinateBox
    category_id: int
    canonical_description: Mapping[str, Any]
    context_tokens: Mapping[str, Any]
    prompt_prefix_token_count: int
    native_repetition_penalty_stratum: float
    source_pred_row_id: str | None
    context_provenance: Mapping[str, Any]
    source_review_foreign_key_lineage: Mapping[str, Any]
    coordinate_space: Mapping[str, Any]
    vocabulary_attestation: Mapping[str, str]
    token_registry: Mapping[str, Any]
    runtime_vocabulary_receipt: Mapping[str, Any]
    upstream_digests: Mapping[str, str]

    @property
    def key(self) -> tuple[str, str]:
        return (self.diagnostic_owner_id, self.context_id)


@dataclass(frozen=True)
class FoilSetMember:
    foil_member_id: str
    diagnostic_owner_id: str
    context_id: str
    bank_name: str
    source_id: str
    identity_kind: str
    identity_id: str
    provenance: Mapping[str, str]


@dataclass(frozen=True)
class MaterializerRules:
    landscape: LandscapeRules
    coordinate_space: Mapping[str, Any]
    target_bank_name: str
    bank_roles: Mapping[str, tuple[str, str, str]]
    foil_members: Mapping[str, FoilSetMember]
    foil_set_id: str
    foil_set_sha256: str
    p_x1_y1_pruning: Mapping[str, Any]
    free_search_budget: Mapping[str, Any]
    vocabulary_attestation: Mapping[str, str]
    token_registry: Mapping[str, Any]
    structural_row_wrapper: Mapping[str, Any]
    analysis_channels: Mapping[str, Any]
    canonical_descriptions: Mapping[str, Mapping[str, Any]]
    upstream_digests: Mapping[str, str]


@dataclass(frozen=True)
class ParsedSeed:
    diagnostic_owner_id: str
    context_id: str
    bank_name: str
    role_id: str
    role_kind: str
    source_id: str
    box: CoordinateBox
    extent_submode: str
    expansion: Literal["anchor_translate", "exact"]
    identity_kind: str
    identity_id: str
    foil_member_id: str | None
    foil_provenance: Mapping[str, str] | None

    def as_core_seed(self) -> ExtentBankSeed:
        return ExtentBankSeed(
            bank_name=self.bank_name,
            source_id=self.source_id,
            box=self.box,
            extent_submode=self.extent_submode,
            expansion=self.expansion,
            physical_owner_hint=(
                self.identity_id
                if self.identity_kind == "reviewed_physical_owner"
                else None
            ),
        )


def _normalized_provenance(value: Any, label: str) -> dict[str, str]:
    raw = _mapping(value, label)
    required = {"artifact_path", "artifact_sha256", "source_row_id"}
    if set(raw) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    return {
        "artifact_path": _string(raw.get("artifact_path"), f"{label}.artifact_path"),
        "artifact_sha256": _sha256(
            raw.get("artifact_sha256"), f"{label}.artifact_sha256"
        ),
        "source_row_id": _string(raw.get("source_row_id"), f"{label}.source_row_id"),
    }


def _gt_owner_id(value: Any, label: str) -> str:
    owner_id = _string(value, label)
    parts = owner_id.split(":")
    if len(parts) != 3 or parts[0] != "gt" or not parts[1] or not parts[2].isdigit():
        raise ValueError(
            f"{label} must use stable gt:<image_id>:<annotation_index> form"
        )
    return owner_id


def _normalized_identity(kind_value: Any, id_value: Any, label: str) -> tuple[str, str]:
    kind = _string(kind_value, f"{label}.identity_kind")
    identity_id = _string(id_value, f"{label}.identity_id")
    if kind == "reviewed_physical_owner":
        _gt_owner_id(identity_id, f"{label}.identity_id")
    elif kind == "registered_geometry":
        if not identity_id.startswith("foil-geometry:"):
            raise ValueError(
                f"{label}.identity_id must use stable foil-geometry:... form for registered geometry"
            )
    else:
        raise ValueError(
            f"{label}.identity_kind must be reviewed_physical_owner or registered_geometry"
        )
    return kind, identity_id


def _normalized_vocabulary_attestation(value: Any, label: str) -> dict[str, str]:
    """Bind coordinate-token meaning to tokenizer, model, and runtime identity."""

    raw = _mapping(value, label)
    required = {
        "tokenizer_identity_sha256",
        "model_identity_sha256",
        "runtime_identity_sha256",
    }
    if set(raw) != required:
        raise ValueError(f"{label} must contain exactly {sorted(required)}")
    return {key: _sha256(raw.get(key), f"{label}.{key}") for key in sorted(required)}


def _normalized_source_pred_row_id(value: Any, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _normalized_context_lineage(
    context_value: Any,
    source_value: Any,
    *,
    source_pred_row_id: str | None,
    label: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context = _mapping(context_value, f"{label}.context_provenance")
    source = _mapping(source_value, f"{label}.source_review_foreign_key_lineage")
    required_context = {
        "context_kind",
        "context_status",
        "eligibility",
        "source_pred_row_id",
        "registry_id",
        "foreign_keys",
        "source_binding",
        "review_status",
    }
    required_source = {
        "source_pred_row_id",
        "review_status",
        "registry_id",
        "foreign_keys",
        "source_binding",
    }
    if set(context) != required_context:
        raise ValueError(
            f"{label}.context_provenance must contain exactly {sorted(required_context)}"
        )
    if set(source) != required_source:
        raise ValueError(
            f"{label}.source_review_foreign_key_lineage must contain exactly {sorted(required_source)}"
        )
    normalized_context = dict(context)
    normalized_source = dict(source)
    for field in ("context_kind", "context_status", "eligibility", "review_status"):
        _string(normalized_context.get(field), f"{label}.context_provenance.{field}")
    _string(
        normalized_source.get("review_status"),
        f"{label}.source_review_foreign_key_lineage.review_status",
    )
    for field in required_source:
        if normalized_source.get(field) != normalized_context.get(field):
            raise ValueError(f"{label} duplicated lineage field {field!r} disagrees")
    if normalized_context.get("source_pred_row_id") != source_pred_row_id:
        raise ValueError(
            f"{label} source_pred_row_id differs across explicit lineage bindings"
        )
    if normalized_context.get("registry_id") is not None:
        _string(
            normalized_context["registry_id"], f"{label}.context_provenance.registry_id"
        )
    if normalized_context.get("foreign_keys") is not None:
        _mapping(
            normalized_context["foreign_keys"],
            f"{label}.context_provenance.foreign_keys",
        )
    _mapping(
        normalized_context.get("source_binding"),
        f"{label}.context_provenance.source_binding",
    )
    return normalized_context, normalized_source


def _normalized_runtime_vocabulary_receipt(
    value: Any, *, token_registry: Mapping[str, Any], label: str
) -> dict[str, Any]:
    raw = _mapping(value, label)
    expected_keys = {
        "model_vocab_size",
        "token_registry_sha256",
        "identity_receipt_digest",
        "tokenizer_identity_sha256",
        "model_identity_sha256",
        "runtime_identity_sha256",
    }
    if set(raw) != expected_keys:
        raise ValueError(f"{label} must contain exactly {sorted(expected_keys)}")
    result: dict[str, Any] = {
        "model_vocab_size": raw.get("model_vocab_size"),
        "token_registry_sha256": _sha256(
            raw.get("token_registry_sha256"), f"{label}.token_registry_sha256"
        ),
        "identity_receipt_digest": _sha256(
            raw.get("identity_receipt_digest"), f"{label}.identity_receipt_digest"
        ),
        **_normalized_vocabulary_attestation(
            {
                key: raw.get(key)
                for key in (
                    "tokenizer_identity_sha256",
                    "model_identity_sha256",
                    "runtime_identity_sha256",
                )
            },
            label,
        ),
    }
    if result["model_vocab_size"] != token_registry["model_vocab_size"]:
        raise ValueError(f"{label}.model_vocab_size differs from the token registry")
    if result["token_registry_sha256"] != token_registry["registry_sha256"]:
        raise ValueError(
            f"{label}.token_registry_sha256 differs from the token registry"
        )
    if result["identity_receipt_digest"] != token_registry["identity_receipt_digest"]:
        raise ValueError(
            f"{label}.identity_receipt_digest differs from the token registry"
        )
    if {
        key: result[key]
        for key in (
            "tokenizer_identity_sha256",
            "model_identity_sha256",
            "runtime_identity_sha256",
        )
    } != token_registry["vocabulary_attestation"]:
        raise ValueError(f"{label} vocabulary identity differs from the token registry")
    return result


def _parse_rules(document: Mapping[str, Any]) -> MaterializerRules:
    landscape = validate_rule_mapping(document)
    token_registry = _normalized_token_registry(
        document.get("token_registry"), "rules.token_registry"
    )
    top_level_schema_tokens = _mapping(
        document.get("schema_tokens"), "rules.schema_tokens"
    )
    expected_top_level_schema_tokens = {
        **token_registry["schema_tokens"],
        "coordinate_token_id_start": token_registry["coordinate_bin_to_token_id"][
            "token_id_start"
        ],
        "coordinate_token_id_end_exclusive": token_registry[
            "coordinate_bin_to_token_id"
        ]["token_id_end_exclusive"],
    }
    if dict(top_level_schema_tokens) != expected_top_level_schema_tokens:
        raise ValueError("rules.schema_tokens differs from the sealed token registry")
    if document.get("model_vocab_size") != token_registry["model_vocab_size"]:
        raise ValueError(
            "rules.model_vocab_size differs from the sealed token registry"
        )
    structural_row_wrapper = _normalized_structural_row_wrapper(
        document.get("structural_row_wrapper"),
        token_registry=token_registry,
        label="rules.structural_row_wrapper",
    )
    analysis_channels = _normalized_analysis_channels(
        document.get("score_channels"), "rules.score_channels"
    )
    canonical_descriptions_raw = _mapping(
        document.get("owner_canonical_descriptions"),
        "rules.owner_canonical_descriptions",
    )
    if not canonical_descriptions_raw:
        raise ValueError("rules.owner_canonical_descriptions must not be empty")
    canonical_descriptions = {
        _gt_owner_id(
            owner_id, "rules.owner_canonical_descriptions key"
        ): _normalized_description(
            description,
            f"rules.owner_canonical_descriptions.{owner_id}",
        )
        for owner_id, description in canonical_descriptions_raw.items()
    }
    raw = _mapping(
        document.get("candidate_materializer"), "rules.candidate_materializer"
    )
    if raw.get("schema_version") != MATERIALIZER_RULES_SCHEMA_VERSION:
        raise ValueError(
            "rules.candidate_materializer.schema_version must be "
            f"{MATERIALIZER_RULES_SCHEMA_VERSION!r}"
        )
    if raw.get("status") != "sealed":
        raise ValueError("rules.candidate_materializer.status must be 'sealed'")
    coordinate_space = _normalized_coordinate_space(
        raw.get("coordinate_space"),
        rules=landscape,
        label="rules.candidate_materializer.coordinate_space",
    )
    namespace = _mapping(
        raw.get("coco_namespace"), "rules.candidate_materializer.coco_namespace"
    )
    if (
        namespace.get("name") != OFFICIAL_COCO_NAMESPACE
        or namespace.get("id_space") != "official_gapped"
    ):
        raise ValueError(
            "rules must declare the official gapped COCO category namespace"
        )

    target_bank_name = _string(
        raw.get("target_bank_name"), "rules.candidate_materializer.target_bank_name"
    )
    landscape.bank_rank(target_bank_name)
    bank_roles_raw = _mapping(
        raw.get("bank_roles"), "rules.candidate_materializer.bank_roles"
    )
    if set(bank_roles_raw) != set(landscape.bank_order):
        raise ValueError(
            "rules.candidate_materializer.bank_roles must name exactly the rule banks"
        )
    bank_roles: dict[str, tuple[str, str, str]] = {}
    for bank_name in landscape.bank_order:
        role_id = _string(
            bank_roles_raw[bank_name], f"registered role for bank {bank_name!r}"
        )
        registered_role = landscape.basin_role(role_id)
        if bank_name not in registered_role.allowed_bank_names:
            raise ValueError(
                f"registered role {role_id!r} does not allow bank {bank_name!r}"
            )
        bank_roles[bank_name] = (
            registered_role.role_id,
            registered_role.kind,
            registered_role.identity_kind,
        )
        # This asks the core contract to validate every bank-to-measure binding.
        landscape.proposal_measure_for_bank(bank_name)
    if bank_roles[target_bank_name][1:] != ("target", "reviewed_physical_owner"):
        raise ValueError("rules target bank must have the registered target role")
    neutral_foil_banks = {"background", *landscape.shape.scan_bank_names}.intersection(
        landscape.bank_order
    )
    for bank_name in sorted(neutral_foil_banks):
        if bank_roles[bank_name][1:] != ("foil", "registered_geometry"):
            raise ValueError(
                f"{bank_name!r} must use an owner-neutral registered-geometry foil role"
            )
    if "covered" in bank_roles and bank_roles["covered"][1:] != (
        "foil",
        "reviewed_physical_owner",
    ):
        raise ValueError("'covered' must use a reviewed physical-owner foil role")

    foil_set_raw = _mapping(
        raw.get("foil_set"), "rules.candidate_materializer.foil_set"
    )
    foil_set_id = _string(
        foil_set_raw.get("foil_set_id"),
        "rules.candidate_materializer.foil_set.foil_set_id",
    )
    member_values = _sequence(
        foil_set_raw.get("members"), "rules.candidate_materializer.foil_set.members"
    )
    if not member_values:
        raise ValueError(
            "rules.candidate_materializer.foil_set.members must not be empty"
        )
    normalized_members: list[dict[str, Any]] = []
    foil_members: dict[str, FoilSetMember] = {}
    for index, value in enumerate(member_values):
        member = _mapping(
            value, f"rules.candidate_materializer.foil_set.members[{index}]"
        )
        member_id = _string(member.get("foil_member_id"), "foil member id")
        bank_name = _string(
            member.get("bank_name"), f"foil member {member_id}.bank_name"
        )
        if bank_name not in bank_roles or bank_roles[bank_name][1] != "foil":
            raise ValueError(
                f"foil member {member_id!r} does not name a registered foil bank"
            )
        if member_id in foil_members:
            raise ValueError(f"duplicate frozen foil member id {member_id!r}")
        identity_kind, identity_id = _normalized_identity(
            member.get("identity_kind"),
            member.get("identity_id"),
            f"foil member {member_id}",
        )
        if identity_kind != bank_roles[bank_name][2]:
            raise ValueError(
                f"foil member {member_id!r} identity kind disagrees with registered role"
            )
        diagnostic_owner_id = _string(
            member.get("diagnostic_owner_id"),
            f"foil member {member_id}.diagnostic_owner_id",
        )
        context_id = _string(
            member.get("context_id"), f"foil member {member_id}.context_id"
        )
        source_id = _string(
            member.get("source_id"), f"foil member {member_id}.source_id"
        )
        provenance = _normalized_provenance(
            member.get("provenance"), f"foil member {member_id}.provenance"
        )
        normalized = {
            "foil_member_id": member_id,
            "diagnostic_owner_id": diagnostic_owner_id,
            "context_id": context_id,
            "bank_name": bank_name,
            "source_id": source_id,
            "identity_kind": identity_kind,
            "identity_id": identity_id,
            "provenance": provenance,
        }
        normalized_members.append(normalized)
        foil_members[member_id] = FoilSetMember(
            foil_member_id=member_id,
            diagnostic_owner_id=diagnostic_owner_id,
            context_id=context_id,
            bank_name=bank_name,
            source_id=source_id,
            identity_kind=identity_kind,
            identity_id=identity_id,
            provenance=provenance,
        )
    frozen_foil_digest = _sha256(
        foil_set_raw.get("members_sha256"),
        "rules.candidate_materializer.foil_set.members_sha256",
    )
    if frozen_foil_digest != sha256_json(
        sorted(normalized_members, key=lambda item: item["foil_member_id"])
    ):
        raise ValueError("rules foil-set digest does not bind its exact members")
    for bank_name, (role_id, _, _) in bank_roles.items():
        if landscape.basin_role(role_id).foil_set_id != foil_set_id:
            raise ValueError(
                f"registered role {role_id!r} for bank {bank_name!r} does not bind the frozen foil set"
            )

    pruning_raw = _mapping(
        raw.get("p_x1_y1_pruning"), "rules.candidate_materializer.p_x1_y1_pruning"
    )
    pruning_mode = pruning_raw.get("mode")
    if pruning_mode not in {"disabled", "upper_bound_pruning"}:
        raise ValueError(
            "rules p_x1_y1 pruning mode must be disabled or upper_bound_pruning"
        )
    pruning = {
        "declaration": _string(
            pruning_raw.get("declaration"), "rules p_x1_y1 pruning declaration"
        ),
        "mode": pruning_mode,
    }
    if pruning_mode == "upper_bound_pruning":
        pruning["threshold"] = _finite_number(
            pruning_raw.get("threshold"), "rules p_x1_y1 pruning threshold"
        )
    elif "threshold" in pruning_raw and pruning_raw["threshold"] is not None:
        raise ValueError("disabled p_x1_y1 pruning may not declare a threshold")

    free_search_budget = _normalized_free_search_budget(
        raw.get("free_search_budget"),
        "rules.candidate_materializer.free_search_budget",
    )
    vocabulary_attestation = _normalized_vocabulary_attestation(
        raw.get("vocabulary_attestation"),
        "rules.candidate_materializer.vocabulary_attestation",
    )
    if vocabulary_attestation != token_registry["vocabulary_attestation"]:
        raise ValueError(
            "rules candidate materializer vocabulary attestation differs from token registry"
        )
    return MaterializerRules(
        landscape=landscape,
        coordinate_space=coordinate_space,
        target_bank_name=target_bank_name,
        bank_roles=dict(bank_roles),
        foil_members=dict(foil_members),
        foil_set_id=foil_set_id,
        foil_set_sha256=frozen_foil_digest,
        p_x1_y1_pruning=pruning,
        free_search_budget=free_search_budget,
        vocabulary_attestation=vocabulary_attestation,
        token_registry=token_registry,
        structural_row_wrapper=structural_row_wrapper,
        analysis_channels=analysis_channels,
        canonical_descriptions=canonical_descriptions,
        upstream_digests=_normalized_digests(
            raw.get("upstream_digests"), "rules.candidate_materializer.upstream_digests"
        ),
    )


def _parse_ledger_rows(
    rows: Sequence[Mapping[str, Any]], rules: MaterializerRules
) -> tuple[LedgerRow, ...]:
    result: list[LedgerRow] = []
    seen_keys: set[tuple[str, str]] = set()
    for index, row in enumerate(rows, start=1):
        label = f"owner/context ledger row {index}"
        if row.get("schema_version") != OWNER_CONTEXT_LEDGER_SCHEMA_VERSION:
            raise ValueError(f"{label}.schema_version is not supported")
        if row.get("owner_status") != "resolved":
            raise ValueError(
                f"{label} has an unresolved owner and cannot enter the candidate bank"
            )
        diagnostic_owner_id = _string(
            row.get("diagnostic_owner_id"), f"{label}.diagnostic_owner_id"
        )
        gt_owner_id = _gt_owner_id(row.get("gt_owner_id"), f"{label}.gt_owner_id")
        context_id = _string(row.get("context_id"), f"{label}.context_id")
        key = (diagnostic_owner_id, context_id)
        if key in seen_keys:
            raise ValueError(
                "duplicate stable owner/context identity: "
                f"{diagnostic_owner_id!r}, {context_id!r}"
            )
        seen_keys.add(key)
        image_id = _image_id(row.get("image_id"), f"{label}.image_id")
        image_identity = _sha256(row.get("image_identity"), f"{label}.image_identity")
        image_size = _mapping(row.get("image_size"), f"{label}.image_size")
        image_width = image_size.get("width")
        image_height = image_size.get("height")
        if (
            not _is_int(image_width)
            or not _is_int(image_height)
            or image_width <= 0
            or image_height <= 0
        ):
            raise ValueError(
                f"{label}.image_size requires positive integer width and height"
            )
        if diagnostic_owner_id.startswith("unresolved:") or context_id.startswith(
            "unresolved:"
        ):
            raise ValueError(f"{label} has an unresolved stable identity")
        ground_truth = _mapping(row.get("ground_truth"), f"{label}.ground_truth")
        box = _box(ground_truth.get("box"), f"{label}.ground_truth.box")
        category = _mapping(
            ground_truth.get("category"), f"{label}.ground_truth.category"
        )
        if category.get("namespace") != OFFICIAL_COCO_NAMESPACE:
            raise ValueError(f"{label} does not use the official gapped COCO namespace")
        category_id = category.get("category_id")
        if not _is_int(category_id) or category_id not in OFFICIAL_COCO_CATEGORY_IDS:
            raise ValueError(f"{label} has an invalid official COCO category id")
        coordinate_space = _normalized_coordinate_space(
            row.get("coordinate_space"),
            rules=rules.landscape,
            label=f"{label}.coordinate_space",
        )
        if coordinate_space != rules.coordinate_space:
            raise ValueError(
                f"{label} coordinate space differs from the sealed materializer rules"
            )
        vocabulary_attestation = _normalized_vocabulary_attestation(
            row.get("vocabulary_attestation"), f"{label}.vocabulary_attestation"
        )
        if vocabulary_attestation != rules.vocabulary_attestation:
            raise ValueError(
                f"{label} vocabulary attestation differs from the sealed rules"
            )
        upstream_digests = _normalized_digests(
            row.get("upstream_digests"), f"{label}.upstream_digests"
        )
        if upstream_digests != rules.upstream_digests:
            raise ValueError(f"{label} upstream digests differ from the sealed rules")
        canonical_description = _normalized_description(
            row.get("canonical_description"), f"{label}.canonical_description"
        )
        try:
            sealed_description = rules.canonical_descriptions[gt_owner_id]
        except KeyError as exc:
            raise ValueError(
                f"{label} owner has no canonical description in the sealed rules"
            ) from exc
        if canonical_description != sealed_description:
            raise ValueError(
                f"{label} canonical description differs from the sealed rules"
            )
        context_tokens = _normalized_context_tokens(
            row.get("context_tokens"), f"{label}.context_tokens"
        )
        prompt_prefix_token_count = row.get("prompt_prefix_token_count")
        if (
            not _is_int(prompt_prefix_token_count)
            or prompt_prefix_token_count != context_tokens["prompt_prefix_token_count"]
        ):
            raise ValueError(
                f"{label}.prompt_prefix_token_count differs from the sealed context-token split"
            )
        native_stratum = _finite_number(
            row.get("native_repetition_penalty_stratum"),
            f"{label}.native_repetition_penalty_stratum",
        )
        if not any(
            math.isclose(native_stratum, allowed, abs_tol=1e-9)
            for allowed in (1.0, 1.10)
        ):
            raise ValueError(
                f"{label}.native_repetition_penalty_stratum must be 1.0 or 1.10"
            )
        source_pred_row_id = _normalized_source_pred_row_id(
            row.get("source_pred_row_id"), f"{label}.source_pred_row_id"
        )
        context_provenance, source_lineage = _normalized_context_lineage(
            row.get("context_provenance"),
            row.get("source_review_foreign_key_lineage"),
            source_pred_row_id=source_pred_row_id,
            label=label,
        )
        token_registry = _normalized_token_registry(
            row.get("token_registry"), f"{label}.token_registry"
        )
        if token_registry != rules.token_registry:
            raise ValueError(f"{label} token registry differs from the sealed rules")
        runtime_vocabulary_receipt = _normalized_runtime_vocabulary_receipt(
            row.get("runtime_vocabulary_receipt"),
            token_registry=token_registry,
            label=f"{label}.runtime_vocabulary_receipt",
        )
        schema_tokens = token_registry["schema_tokens"]
        expected_forced_prefix = [
            schema_tokens["object_ref_start_token_id"],
            *canonical_description["token_ids"],
            schema_tokens["object_ref_end_token_id"],
            schema_tokens["box_start_token_id"],
        ]
        if (
            canonical_description["forced_row_prefix_through_box_start_token_ids"]
            != expected_forced_prefix
        ):
            raise ValueError(
                f"{label}.canonical_description forced row prefix differs from the sealed wrapper"
            )
        model_vocab_size = token_registry["model_vocab_size"]
        exact_prefix_ids = [
            *context_tokens["token_ids"],
            *canonical_description["forced_row_prefix_through_box_start_token_ids"],
        ]
        if any(token_id >= model_vocab_size for token_id in exact_prefix_ids):
            raise ValueError(
                f"{label} exact executable prefix token exceeds model vocabulary"
            )
        result.append(
            LedgerRow(
                diagnostic_owner_id=diagnostic_owner_id,
                gt_owner_id=gt_owner_id,
                context_id=context_id,
                image_id=image_id,
                image_identity=image_identity,
                image_width=image_width,
                image_height=image_height,
                gt_box=box,
                category_id=category_id,
                canonical_description=canonical_description,
                context_tokens=context_tokens,
                prompt_prefix_token_count=int(prompt_prefix_token_count),
                native_repetition_penalty_stratum=native_stratum,
                source_pred_row_id=source_pred_row_id,
                context_provenance=context_provenance,
                source_review_foreign_key_lineage=source_lineage,
                coordinate_space=coordinate_space,
                vocabulary_attestation=vocabulary_attestation,
                token_registry=token_registry,
                runtime_vocabulary_receipt=runtime_vocabulary_receipt,
                upstream_digests=upstream_digests,
            )
        )
    if not result:
        raise ValueError("owner/context ledger must not be empty")
    return tuple(sorted(result, key=lambda item: item.key))


def _parse_seeds(
    document: Mapping[str, Any],
    *,
    rules: MaterializerRules,
    rules_sha256: str,
    ledger_sha256: str,
    ledger_rows: Sequence[LedgerRow],
) -> tuple[ParsedSeed, ...]:
    if document.get("schema_version") != BANK_SEEDS_SCHEMA_VERSION:
        raise ValueError("bank seeds document schema_version is not supported")
    if (
        _sha256(
            document.get("landscape_decision_rules_sha256"), "bank seeds rules digest"
        )
        != rules_sha256
    ):
        raise ValueError("bank seeds do not bind the supplied sealed rules digest")
    if (
        _sha256(document.get("owner_context_ledger_sha256"), "bank seeds ledger digest")
        != ledger_sha256
    ):
        raise ValueError(
            "bank seeds do not bind the supplied owner/context ledger digest"
        )
    if (
        _sha256(document.get("foil_set_sha256"), "bank seeds foil-set digest")
        != rules.foil_set_sha256
    ):
        raise ValueError("bank seeds do not bind the sealed foil-set digest")
    raw_seeds = _sequence(document.get("seeds"), "bank seeds.seeds")
    if not raw_seeds:
        raise ValueError("bank seeds.seeds must not be empty")
    ledger_by_key = {row.key: row for row in ledger_rows}
    result: list[ParsedSeed] = []
    used_foil_members: set[str] = set()
    seen_seed_ids: set[tuple[str, str, str, str]] = set()
    target_anchor_seed_keys: set[tuple[str, str]] = set()
    for index, raw_seed in enumerate(raw_seeds):
        label = f"bank seed {index}"
        seed = _mapping(raw_seed, label)
        diagnostic_owner_id = _string(
            seed.get("diagnostic_owner_id"), f"{label}.diagnostic_owner_id"
        )
        context_id = _string(seed.get("context_id"), f"{label}.context_id")
        key = (diagnostic_owner_id, context_id)
        if key not in ledger_by_key:
            raise ValueError(f"{label} references an unknown owner/context identity")
        ledger = ledger_by_key[key]
        bank_name = _string(seed.get("bank_name"), f"{label}.bank_name")
        try:
            rules.landscape.bank_rank(bank_name)
        except ValueError as exc:
            raise ValueError(f"{label} names an unknown deterministic bank") from exc
        expected_role_id, expected_role_kind, expected_identity_kind = rules.bank_roles[
            bank_name
        ]
        if seed.get("role_id") != expected_role_id:
            raise ValueError(
                f"{label} role_id is not the registered role for bank {bank_name!r}"
            )
        source_id = _string(seed.get("source_id"), f"{label}.source_id")
        seed_identity = (diagnostic_owner_id, context_id, bank_name, source_id)
        if seed_identity in seen_seed_ids:
            raise ValueError(f"duplicate explicit bank seed {seed_identity!r}")
        seen_seed_ids.add(seed_identity)
        expansion = seed.get("expansion")
        if expansion not in {"anchor_translate", "exact"}:
            raise ValueError(f"{label}.expansion must be anchor_translate or exact")
        identity_kind, identity_id = _normalized_identity(
            seed.get("identity_kind"), seed.get("identity_id"), label
        )
        if identity_kind != expected_identity_kind:
            raise ValueError(
                f"{label} identity kind disagrees with its registered role"
            )
        if expected_role_kind == "target" and (
            identity_kind != "reviewed_physical_owner"
            or identity_id != ledger.gt_owner_id
        ):
            raise ValueError(f"{label} target role must bind the target GT owner")
        if bank_name == "covered" and (
            identity_kind != "reviewed_physical_owner"
            or identity_id == ledger.gt_owner_id
        ):
            raise ValueError(
                f"{label} covered-owner foil must bind an actual distinct GT owner"
            )
        if bank_name in {"background", *rules.landscape.shape.scan_bank_names} and (
            identity_kind != "registered_geometry"
        ):
            raise ValueError(f"{label} must remain owner-neutral registered geometry")
        foil_member_id: str | None = None
        foil_provenance: Mapping[str, str] | None = None
        if expected_role_kind == "foil":
            foil_member_id = _string(
                seed.get("foil_member_id"), f"{label}.foil_member_id"
            )
            try:
                member = rules.foil_members[foil_member_id]
            except KeyError as exc:
                raise ValueError(f"{label} has no frozen foil provenance") from exc
            if foil_member_id in used_foil_members:
                raise ValueError(
                    f"frozen foil member {foil_member_id!r} is materialized more than once"
                )
            if (
                member.diagnostic_owner_id,
                member.context_id,
                member.bank_name,
                member.source_id,
                member.identity_kind,
                member.identity_id,
            ) != (
                diagnostic_owner_id,
                context_id,
                bank_name,
                source_id,
                identity_kind,
                identity_id,
            ):
                raise ValueError(f"{label} disagrees with frozen foil membership")
            foil_provenance = _normalized_provenance(
                seed.get("foil_provenance"), f"{label}.foil_provenance"
            )
            if foil_provenance != member.provenance:
                raise ValueError(
                    f"{label} foil provenance does not match the frozen foil set"
                )
            used_foil_members.add(foil_member_id)
        elif "foil_member_id" in seed or "foil_provenance" in seed:
            raise ValueError(f"{label} target-role seed must not claim foil provenance")
        parsed = ParsedSeed(
            diagnostic_owner_id=diagnostic_owner_id,
            context_id=context_id,
            bank_name=bank_name,
            role_id=expected_role_id,
            role_kind=expected_role_kind,
            source_id=source_id,
            box=_box(seed.get("box"), f"{label}.box"),
            extent_submode=_string(
                seed.get("extent_submode"), f"{label}.extent_submode"
            ),
            expansion=expansion,
            identity_kind=identity_kind,
            identity_id=identity_id,
            foil_member_id=foil_member_id,
            foil_provenance=foil_provenance,
        )
        # Core construction validates box range and the expansion semantics.
        parsed.as_core_seed()
        if expected_role_kind == "target" and expansion == "anchor_translate":
            target_anchor_seed_keys.add(key)
        result.append(parsed)
    required_foil_members = {
        member_id
        for member_id, member in rules.foil_members.items()
        if (member.diagnostic_owner_id, member.context_id) in ledger_by_key
    }
    missing_foils = sorted(required_foil_members - used_foil_members)
    if missing_foils:
        raise ValueError(
            "bank seeds omit frozen foil provenance: " + ", ".join(missing_foils)
        )
    missing_target_anchors = sorted(set(ledger_by_key) - target_anchor_seed_keys)
    if missing_target_anchors:
        raise ValueError(
            "every resolved owner/context needs an anchor_translate target seed for dense restricted coverage: "
            + ", ".join(
                f"{owner_id}/{context_id}"
                for owner_id, context_id in missing_target_anchors
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.diagnostic_owner_id,
                item.context_id,
                item.bank_name,
                item.source_id,
            ),
        )
    )


def _candidate_id(*, ledger: LedgerRow, candidate: Any) -> str:
    payload = {
        "anchor": None
        if candidate.anchor is None
        else [candidate.anchor.x1.value, candidate.anchor.y1.value],
        "bank_name": candidate.bank_name,
        "box": list(candidate.box.as_tuple()),
        "canonical_description": ledger.canonical_description,
        "context_id": ledger.context_id,
        "diagnostic_owner_id": ledger.diagnostic_owner_id,
        "extent_submode": candidate.extent_submode,
        "gt_owner_id": ledger.gt_owner_id,
        "source_id": candidate.source_id,
    }
    return "landscape-candidate:sha256:" + sha256_json(payload)


def _unique_box_id(*, ledger: LedgerRow, box: CoordinateBox) -> str:
    payload = {
        "box": list(box.as_tuple()),
        "canonical_description_id": ledger.canonical_description["description_id"],
        "context_id": ledger.context_id,
        "diagnostic_owner_id": ledger.diagnostic_owner_id,
        "gt_owner_id": ledger.gt_owner_id,
    }
    return "landscape-unique-box:sha256:" + sha256_json(payload)


def _root_prefix_token_ids(ledger: LedgerRow) -> list[int]:
    """Return the literal production prefix through ``box_start`` only."""

    return [
        *ledger.context_tokens["token_ids"],
        *ledger.canonical_description["forced_row_prefix_through_box_start_token_ids"],
    ]


def _coordinate_token_id(rules: MaterializerRules, coordinate_bin: int) -> int:
    registry = rules.token_registry["coordinate_bin_to_token_id"]
    if not 0 <= coordinate_bin < len(registry["coordinate_bin_token_ids"]):
        raise ValueError(
            f"coordinate bin {coordinate_bin} is absent from the frozen registry"
        )
    # The explicit registry is authoritative.  Never infer an offset from a
    # token interval or from decoded token text here.
    return int(registry["coordinate_bin_token_ids"][coordinate_bin])


def _request_analysis_channels(
    ledger: LedgerRow, rules: MaterializerRules
) -> dict[str, Any]:
    return {
        **dict(rules.analysis_channels),
        "native_repetition_penalty_stratum": ledger.native_repetition_penalty_stratum,
        "raw_forward_reuse_key_fields": [
            "image_identity",
            "prefix_token_ids_sha256",
            "runtime_vocabulary_receipt_sha256",
        ],
        "duplicate_raw_forward_per_policy_view_allowed": False,
    }


def _request_lineage(ledger: LedgerRow) -> dict[str, Any]:
    return {
        "source_pred_row_id": ledger.source_pred_row_id,
        "review_status": ledger.source_review_foreign_key_lineage["review_status"],
        "registry_id": ledger.source_review_foreign_key_lineage["registry_id"],
        "foreign_keys": ledger.source_review_foreign_key_lineage["foreign_keys"],
        "source_binding": ledger.source_review_foreign_key_lineage["source_binding"],
        "lineage_sha256": sha256_json(ledger.source_review_foreign_key_lineage),
    }


def _candidate_kind_for_bank(bank_name: str) -> str:
    return {
        "target": "target_anchor",
        "covered": "covered_owner",
        "background": "background",
        "scan": "scan",
        "part": "part",
        "whole": "whole",
        "merged": "merged",
    }[bank_name]


def _runtime_vocabulary_receipt_sha256(ledger: LedgerRow) -> str:
    return sha256_json(ledger.runtime_vocabulary_receipt)


def _core_request_attestation(
    *,
    ledger: LedgerRow,
    rules: MaterializerRules,
    rules_sha256: str,
    prefix_token_ids: Sequence[int],
    request_payload: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "canonical_description_token_ids_sha256": ledger.canonical_description[
            "token_ids_sha256"
        ],
        "context_token_ids_sha256": ledger.context_tokens["token_ids_sha256"],
        "coordinate_bin_token_ids_sha256": rules.token_registry[
            "coordinate_bin_to_token_id"
        ]["coordinate_bin_token_ids_sha256"],
        "core_rule_digest": rules.landscape.rule_digest,
        "foil_set_sha256": rules.foil_set_sha256,
        "landscape_decision_rules_sha256": rules_sha256,
        "prefix_token_ids_sha256": sha256_json(list(prefix_token_ids)),
        "request_payload_sha256": sha256_json(dict(request_payload)),
        "runtime_vocabulary_receipt_sha256": _runtime_vocabulary_receipt_sha256(ledger),
        "structural_row_wrapper_sha256": rules.structural_row_wrapper[
            "structural_row_wrapper_sha256"
        ],
        "token_registry_sha256": rules.token_registry["registry_sha256"],
        "vocabulary_attestation": dict(ledger.vocabulary_attestation),
    }
    return {**payload, "attestation_sha256": sha256_json(payload)}


def _proposal_digest(
    *,
    candidate_kind: str,
    diagnostic_owner_id: str,
    context_id: str,
    role: str,
    foil_set_id: str,
    request_kind: str,
    coord_token_ids: Sequence[int] | None = None,
    fixed_coord_token_ids: Sequence[int] | None = None,
    scan_slot: str | None = None,
) -> str:
    payload: dict[str, Any] = {
        "candidate_kind": candidate_kind,
        "diagnostic_owner_id": diagnostic_owner_id,
        "context_id": context_id,
        "role": role,
        "foil_set_id": foil_set_id,
        "request_kind": request_kind,
    }
    if request_kind == "complete_box":
        payload["coord_token_ids"] = list(coord_token_ids or ())
    else:
        payload["fixed_coord_token_ids"] = list(fixed_coord_token_ids or ())
        payload["scan_slot"] = scan_slot
    return sha256_json(payload)


def _free_coordinate_tree_root_record(
    *,
    ledger: LedgerRow,
    rules: MaterializerRules,
    rules_sha256: str,
) -> dict[str, Any]:
    prefix_token_ids = _root_prefix_token_ids(ledger)
    request_payload = {
        "request_kind": "free_coordinate_tree_root",
        "scan_slot": "x1",
        "budget": dict(rules.free_search_budget),
        "dynamic_traversal_owner": "runtime_landscape_scorer_free_tree_surface",
    }
    request_id = "free-coordinate-tree-root:sha256:" + sha256_json(
        {
            "diagnostic_owner_id": ledger.diagnostic_owner_id,
            "context_id": ledger.context_id,
            "prefix_token_ids_sha256": sha256_json(prefix_token_ids),
            **request_payload,
        }
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "free_coordinate_tree_root_request",
        "request_id": request_id,
        "candidate_id": request_id,
        "request_kind": "free_coordinate_tree_root",
        "diagnostic_owner_id": ledger.diagnostic_owner_id,
        "gt_owner_id": ledger.gt_owner_id,
        "context_id": ledger.context_id,
        "image_id": ledger.image_id,
        "image_identity": ledger.image_identity,
        "image_size": {"width": ledger.image_width, "height": ledger.image_height},
        "canonical_description": dict(ledger.canonical_description),
        "prompt_prefix_token_count": ledger.prompt_prefix_token_count,
        "prefix_token_ids": prefix_token_ids,
        "root_prefix_token_ids": prefix_token_ids,
        "prefix_token_ids_sha256": sha256_json(prefix_token_ids),
        "root_prefix_token_ids_sha256": sha256_json(prefix_token_ids),
        "scan_slot": "x1",
        "fixed_coord_bin_values": [],
        "fixed_coord_token_ids": [],
        "fixed_coord_tokens_materialized_in_prefix": False,
        "coordinate_space": dict(ledger.coordinate_space),
        "coordinate_token_registry_sha256": rules.token_registry["registry_sha256"],
        "coordinate_bin_token_ids_sha256": rules.token_registry[
            "coordinate_bin_to_token_id"
        ]["coordinate_bin_token_ids_sha256"],
        "native_repetition_penalty_stratum": ledger.native_repetition_penalty_stratum,
        "source_pred_row_id": ledger.source_pred_row_id,
        "review_status": ledger.source_review_foreign_key_lineage["review_status"],
        "source_review_foreign_key_lineage": _request_lineage(ledger),
        "context_provenance": dict(ledger.context_provenance),
        "analysis_channels": _request_analysis_channels(ledger, rules),
        "runtime_vocabulary_receipt": dict(ledger.runtime_vocabulary_receipt),
        "runtime_vocabulary_receipt_sha256": _runtime_vocabulary_receipt_sha256(ledger),
        "vocabulary_attestation": dict(ledger.vocabulary_attestation),
        "dynamic_traversal_seam": {
            "execution_owner": "runtime_landscape_scorer_free_tree_surface",
            "execution_status": "declared_not_executed_by_candidate_materializer",
            "root_scan_slot": "x1",
            "transition": (
                "append_each_selected_coordinate_token_exactly_once_to_the_parent_cache_branch"
            ),
            "coordinate_token_source": "sealed_explicit_coordinate_bin_token_ids_registry",
            "budget": dict(rules.free_search_budget),
            "candidate_membership_surface": "separate_from_restricted_candidate_bank",
        },
        "execution_status": "declared_not_executed_by_candidate_materializer",
        "core_rule_digest": rules.landscape.rule_digest,
        "landscape_decision_rules_sha256": rules_sha256,
        "foil_set_id": rules.foil_set_id,
        "foil_set_sha256": rules.foil_set_sha256,
        "core_request_attestation": _core_request_attestation(
            ledger=ledger,
            rules=rules,
            rules_sha256=rules_sha256,
            prefix_token_ids=prefix_token_ids,
            request_payload=request_payload,
        ),
        "upstream_digests": dict(ledger.upstream_digests),
    }


def _conditional_y1_plan_record(
    *,
    ledger: LedgerRow,
    x1: int,
    rules: MaterializerRules,
    rules_sha256: str,
) -> dict[str, Any]:
    complete_y1 = enumerate_complete_conditional_y1(
        CoordinateBin(x1), ledger.gt_box, rules.landscape
    )
    root_prefix_token_ids = _root_prefix_token_ids(ledger)
    x1_token_id = _coordinate_token_id(rules, x1)
    fixed_coord_token_ids = [x1_token_id]
    expected_prefix_after_fixed = [*root_prefix_token_ids, x1_token_id]
    entries = [
        {
            "y1": entry.y1.value,
            "is_target_anchor": entry.is_target_anchor,
            "can_form_valid_box": entry.can_form_valid_box,
            "invalid_box_reason": entry.invalid_box_reason,
        }
        for entry in complete_y1
    ]
    plan_payload = {
        "canonical_description_id": ledger.canonical_description["description_id"],
        "context_id": ledger.context_id,
        "coordinate_space": ledger.coordinate_space,
        "diagnostic_owner_id": ledger.diagnostic_owner_id,
        "gt_owner_id": ledger.gt_owner_id,
        "prefix_token_ids_sha256": sha256_json(root_prefix_token_ids),
        "prompt_prefix_token_count": ledger.prompt_prefix_token_count,
        "fixed_coord_bin_values": [x1],
        "fixed_coord_token_ids": fixed_coord_token_ids,
        "scan_slot": "y1",
        "source_review_foreign_key_lineage": ledger.source_review_foreign_key_lineage,
        "x1": x1,
    }
    plan_id = "conditional-y1-plan:sha256:" + sha256_json(plan_payload)
    conditional_y1_plan_digest = sha256_json(
        {
            **plan_payload,
            "complete_conditional_y1": entries,
            "core_rule_digest": rules.landscape.rule_digest,
            "landscape_decision_rules_sha256": rules_sha256,
            "runtime_vocabulary_receipt_sha256": _runtime_vocabulary_receipt_sha256(
                ledger
            ),
            "token_registry_sha256": rules.token_registry["registry_sha256"],
            "vocabulary_attestation": ledger.vocabulary_attestation,
        }
    )
    pruning = dict(rules.p_x1_y1_pruning)
    pruning["execution_status"] = "declared_not_executed_by_candidate_materializer"
    request_kind = "dense_scan"
    candidate_kind = "target_anchor"
    role = "target"
    request_payload = {
        "request_kind": request_kind,
        "fixed_coord_bin_values": [x1],
        "fixed_coord_token_ids": fixed_coord_token_ids,
        "scan_slot": "y1",
    }
    core_request_attestation = _core_request_attestation(
        ledger=ledger,
        rules=rules,
        rules_sha256=rules_sha256,
        prefix_token_ids=root_prefix_token_ids,
        request_payload=request_payload,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": "conditional_y1_score_plan",
        "plan_id": plan_id,
        "candidate_id": plan_id,
        "diagnostic_owner_id": ledger.diagnostic_owner_id,
        "gt_owner_id": ledger.gt_owner_id,
        "context_id": ledger.context_id,
        "image_id": ledger.image_id,
        "image_identity": ledger.image_identity,
        "image_size": {"width": ledger.image_width, "height": ledger.image_height},
        "canonical_description": dict(ledger.canonical_description),
        "prompt_prefix_token_count": ledger.prompt_prefix_token_count,
        "prefix_token_ids": root_prefix_token_ids,
        "root_prefix_token_ids": root_prefix_token_ids,
        "prefix_token_ids_sha256": sha256_json(root_prefix_token_ids),
        "root_prefix_token_ids_sha256": sha256_json(root_prefix_token_ids),
        "context_token_ids_sha256": ledger.context_tokens["token_ids_sha256"],
        "request_kind": request_kind,
        "fixed_coord_bin_values": [x1],
        "fixed_coord_token_ids": fixed_coord_token_ids,
        "scan_slot": "y1",
        "fixed_coord_tokens_materialized_in_prefix": False,
        "expected_prefix_after_fixed_token_ids": expected_prefix_after_fixed,
        "expected_prefix_after_fixed_token_ids_sha256": sha256_json(
            expected_prefix_after_fixed
        ),
        "expected_prefix_after_fixed_is_attestation_only": True,
        "proposal_digest": _proposal_digest(
            candidate_kind=candidate_kind,
            diagnostic_owner_id=ledger.diagnostic_owner_id,
            context_id=ledger.context_id,
            role=role,
            foil_set_id=rules.foil_set_id,
            request_kind=request_kind,
            fixed_coord_token_ids=fixed_coord_token_ids,
            scan_slot="y1",
        ),
        "candidate_kind": candidate_kind,
        "role": role,
        "physical_owner_hint": ledger.gt_owner_id,
        "review_status": ledger.source_review_foreign_key_lineage["review_status"],
        "native_repetition_penalty_stratum": ledger.native_repetition_penalty_stratum,
        "source_pred_row_id": ledger.source_pred_row_id,
        "source_review_foreign_key_lineage": _request_lineage(ledger),
        "context_provenance": dict(ledger.context_provenance),
        "analysis_channels": _request_analysis_channels(ledger, rules),
        "coordinate_space": dict(ledger.coordinate_space),
        "coordinate_token_registry_sha256": rules.token_registry["registry_sha256"],
        "coordinate_bin_token_ids_sha256": rules.token_registry[
            "coordinate_bin_to_token_id"
        ]["coordinate_bin_token_ids_sha256"],
        "contract_mode": rules.landscape.contract_mode,
        "geometry_identity_schema": rules.landscape.geometry_identity_schema,
        "vocabulary_attestation": dict(ledger.vocabulary_attestation),
        "runtime_vocabulary_receipt": dict(ledger.runtime_vocabulary_receipt),
        "runtime_vocabulary_receipt_sha256": _runtime_vocabulary_receipt_sha256(ledger),
        "core_request_attestation": core_request_attestation,
        "x1": x1,
        "complete_conditional_y1": entries,
        "conditional_y1_plan_sha256": conditional_y1_plan_digest,
        "score_plan_receipt": {
            "selected_token": "y1",
            "conditional": "P(y1 | image, exact_context, canonical_description, x1)",
            "complete_coordinate_bin_count": len(entries),
            "terminal_y1": entries[-1],
            "status": "planned_no_scores_materialized",
        },
        "p_x1_y1_pruning": pruning,
        "core_rule_digest": rules.landscape.rule_digest,
        "landscape_decision_rules_sha256": rules_sha256,
        "foil_set_id": rules.foil_set_id,
        "foil_set_sha256": rules.foil_set_sha256,
        "scored_completeness_receipt_sha256": None,
        "upstream_digests": dict(ledger.upstream_digests),
    }


def build_candidate_records(
    *,
    ledger_rows: Sequence[LedgerRow],
    seeds: Sequence[ParsedSeed],
    rules: MaterializerRules,
    rules_sha256: str,
) -> list[dict[str, Any]]:
    """Build all deterministic records without reading or accepting scores."""

    seeds_by_key: dict[tuple[str, str], list[ParsedSeed]] = defaultdict(list)
    for seed in seeds:
        seeds_by_key[(seed.diagnostic_owner_id, seed.context_id)].append(seed)
    records: list[dict[str, Any]] = []
    candidate_records: list[dict[str, Any]] = []
    for ledger in ledger_rows:
        records.append(
            _free_coordinate_tree_root_record(
                ledger=ledger,
                rules=rules,
                rules_sha256=rules_sha256,
            )
        )
        anchors = enumerate_target_anchor_pairs(ledger.gt_box, rules.landscape)
        x1_values = sorted({anchor.x1.value for anchor in anchors})
        plan_id_by_x1: dict[int, str] = {}
        for x1 in x1_values:
            plan = _conditional_y1_plan_record(
                ledger=ledger, x1=x1, rules=rules, rules_sha256=rules_sha256
            )
            records.append(plan)
            plan_id_by_x1[x1] = plan["plan_id"]
        completeness_payload = {
            "canonical_description_id": ledger.canonical_description["description_id"],
            "conditional_y1_plan_sha256": [
                next(
                    record["conditional_y1_plan_sha256"]
                    for record in records
                    if record["record_type"] == "conditional_y1_score_plan"
                    and record["diagnostic_owner_id"] == ledger.diagnostic_owner_id
                    and record["context_id"] == ledger.context_id
                    and record["x1"] == x1
                )
                for x1 in x1_values
            ],
            "context_id": ledger.context_id,
            "coordinate_space": ledger.coordinate_space,
            "core_rule_digest": rules.landscape.rule_digest,
            "declared_x1_anchor_bins": x1_values,
            "diagnostic_owner_id": ledger.diagnostic_owner_id,
            "gt_box": list(ledger.gt_box.as_tuple()),
            "gt_owner_id": ledger.gt_owner_id,
            "image_identity": ledger.image_identity,
            "landscape_decision_rules_sha256": rules_sha256,
            "attestation_kind": rules.landscape.contract_mode,
            "canonical_description_text": ledger.canonical_description["text"],
            "canonical_description_token_digest": ledger.canonical_description[
                "token_ids_sha256"
            ],
            "context_token_digest": ledger.context_tokens["token_ids_sha256"],
            "geometry_identity_schema": rules.landscape.geometry_identity_schema,
            "tokenizer_identity": ledger.vocabulary_attestation[
                "tokenizer_identity_sha256"
            ],
            "model_identity": ledger.vocabulary_attestation["model_identity_sha256"],
            "runtime_identity": ledger.vocabulary_attestation[
                "runtime_identity_sha256"
            ],
            "vocabulary_attestation": ledger.vocabulary_attestation,
        }
        completeness_plan_digest = "restricted-completeness-plan:sha256:" + sha256_json(
            completeness_payload
        )
        for record in records:
            if (
                record["record_type"] == "conditional_y1_score_plan"
                and record["diagnostic_owner_id"] == ledger.diagnostic_owner_id
                and record["context_id"] == ledger.context_id
            ):
                record["completeness_plan_digest"] = completeness_plan_digest
                record["completeness_plan_status"] = "unscored_complete_y1_plan"
        expansion = expand_extent_banks(
            anchors,
            tuple(seed.as_core_seed() for seed in seeds_by_key[ledger.key]),
            rules.landscape,
        )
        seed_by_origin = {
            (seed.bank_name, seed.source_id): seed for seed in seeds_by_key[ledger.key]
        }
        required_target_anchors = {
            (anchor.x1.value, anchor.y1.value) for anchor in anchors
        }
        created_target_anchors = {
            (candidate.anchor.x1.value, candidate.anchor.y1.value)
            for candidate in expansion.candidates
            if candidate.anchor is not None
            and seed_by_origin[(candidate.bank_name, candidate.source_id)].role_kind
            == "target"
        }
        missing_target_anchors = sorted(
            required_target_anchors - created_target_anchors
        )
        if missing_target_anchors:
            raise ValueError(
                "frozen target extent seeds do not cover every declared restricted anchor: "
                + ", ".join(f"({x1},{y1})" for x1, y1 in missing_target_anchors)
            )
        for candidate in expansion.candidates:
            seed = seed_by_origin[(candidate.bank_name, candidate.source_id)]
            measure = rules.landscape.proposal_measure_for_bank(candidate.bank_name)
            candidate_id = _candidate_id(ledger=ledger, candidate=candidate)
            unique_box_id = _unique_box_id(ledger=ledger, box=candidate.box)
            coordinate_bin_values = list(candidate.box.as_tuple())
            coordinate_token_ids = [
                _coordinate_token_id(rules, coordinate_bin)
                for coordinate_bin in coordinate_bin_values
            ]
            prefix_token_ids = _root_prefix_token_ids(ledger)
            scorer_role = "target" if seed.role_kind == "target" else "registered_foil"
            candidate_kind = _candidate_kind_for_bank(candidate.bank_name)
            request_payload = {
                "request_kind": "complete_box",
                "coordinate_bin_values": coordinate_bin_values,
                "coord_token_ids": coordinate_token_ids,
            }
            geometry = canonical_geometry_identity(
                candidate.box,
                image_width=ledger.image_width,
                image_height=ledger.image_height,
                rules=rules.landscape,
            )
            candidate_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_type": "complete_box_candidate",
                    "candidate_id": candidate_id,
                    "complete_box_candidate_id": candidate.candidate_id,
                    "unique_box_id": unique_box_id,
                    "unique_box_mass_semantics": {
                        "membership": "one_score_per_owner_context_description_box",
                        "selection_status": "pre_score_frozen",
                    },
                    "diagnostic_owner_id": ledger.diagnostic_owner_id,
                    "gt_owner_id": ledger.gt_owner_id,
                    "context_id": ledger.context_id,
                    "image_id": ledger.image_id,
                    "image_identity": ledger.image_identity,
                    "image_size": {
                        "width": ledger.image_width,
                        "height": ledger.image_height,
                    },
                    "category": {
                        "namespace": OFFICIAL_COCO_NAMESPACE,
                        "category_id": ledger.category_id,
                    },
                    "canonical_description": dict(ledger.canonical_description),
                    "prompt_prefix_token_count": ledger.prompt_prefix_token_count,
                    "prefix_token_ids": prefix_token_ids,
                    "root_prefix_token_ids": prefix_token_ids,
                    "prefix_token_ids_sha256": sha256_json(prefix_token_ids),
                    "root_prefix_token_ids_sha256": sha256_json(prefix_token_ids),
                    "context_token_ids_sha256": ledger.context_tokens[
                        "token_ids_sha256"
                    ],
                    "request_kind": "complete_box",
                    "coordinate_bin_values": coordinate_bin_values,
                    "coordinate_token_ids": coordinate_token_ids,
                    # Current runtime scorer compatibility alias.  Both names
                    # are validated from the explicit frozen registry above.
                    "coord_token_ids": coordinate_token_ids,
                    "coord_token_ids_sha256": sha256_json(coordinate_token_ids),
                    "proposal_digest": _proposal_digest(
                        candidate_kind=candidate_kind,
                        diagnostic_owner_id=ledger.diagnostic_owner_id,
                        context_id=ledger.context_id,
                        role=scorer_role,
                        foil_set_id=rules.foil_set_id,
                        request_kind="complete_box",
                        coord_token_ids=coordinate_token_ids,
                    ),
                    "candidate_kind": candidate_kind,
                    "role": scorer_role,
                    "review_status": ledger.source_review_foreign_key_lineage[
                        "review_status"
                    ],
                    "native_repetition_penalty_stratum": ledger.native_repetition_penalty_stratum,
                    "source_pred_row_id": ledger.source_pred_row_id,
                    "source_review_foreign_key_lineage": _request_lineage(ledger),
                    "context_provenance": dict(ledger.context_provenance),
                    "analysis_channels": _request_analysis_channels(ledger, rules),
                    "coordinate_space": dict(ledger.coordinate_space),
                    "coordinate_token_registry_sha256": rules.token_registry[
                        "registry_sha256"
                    ],
                    "coordinate_bin_token_ids_sha256": rules.token_registry[
                        "coordinate_bin_to_token_id"
                    ]["coordinate_bin_token_ids_sha256"],
                    "contract_mode": rules.landscape.contract_mode,
                    "geometry_identity": {
                        "schema": geometry.schema,
                        "coordinate_box": list(geometry.coordinate_box.as_tuple()),
                        "image_width": geometry.image_width,
                        "image_height": geometry.image_height,
                        "pixel_box_xyxy": list(geometry.pixel_box_xyxy),
                        "identity_digest": geometry.identity_digest,
                    },
                    "vocabulary_attestation": dict(ledger.vocabulary_attestation),
                    "runtime_vocabulary_receipt": dict(
                        ledger.runtime_vocabulary_receipt
                    ),
                    "runtime_vocabulary_receipt_sha256": _runtime_vocabulary_receipt_sha256(
                        ledger
                    ),
                    "core_request_attestation": _core_request_attestation(
                        ledger=ledger,
                        rules=rules,
                        rules_sha256=rules_sha256,
                        prefix_token_ids=prefix_token_ids,
                        request_payload=request_payload,
                    ),
                    "role_id": seed.role_id,
                    "role_kind": seed.role_kind,
                    "identity_kind": seed.identity_kind,
                    "identity_id": seed.identity_id,
                    "reviewed_physical_owner_id": (
                        seed.identity_id
                        if seed.identity_kind == "reviewed_physical_owner"
                        else None
                    ),
                    "registered_geometry_id": (
                        seed.identity_id
                        if seed.identity_kind == "registered_geometry"
                        else None
                    ),
                    "physical_owner_registration_allowed": (
                        seed.identity_kind == "reviewed_physical_owner"
                    ),
                    "prominence_eligibility": (
                        "registered_target"
                        if seed.role_kind == "target"
                        else (
                            "registered_physical_owner_foil"
                            if seed.identity_kind == "reviewed_physical_owner"
                            else "registered_owner_neutral_foil"
                        )
                    ),
                    "foil_member_id": seed.foil_member_id,
                    "foil_provenance": None
                    if seed.foil_provenance is None
                    else dict(seed.foil_provenance),
                    "bank_name": candidate.bank_name,
                    "source_id": candidate.source_id,
                    "extent_submode": candidate.extent_submode,
                    "expansion": seed.expansion,
                    # Legacy core compatibility only.  Registered geometry is
                    # never synthesized into a physical-owner identifier.
                    "physical_owner_hint": (
                        seed.identity_id
                        if seed.identity_kind == "reviewed_physical_owner"
                        else None
                    ),
                    "proposal_measure_id": measure.measure_id,
                    "proposal_comparability_group": measure.comparability_group,
                    "box": _box_json(candidate.box),
                    "anchor": None
                    if candidate.anchor is None
                    else {
                        "x1": candidate.anchor.x1.value,
                        "y1": candidate.anchor.y1.value,
                    },
                    "conditional_y1_plan_id": None
                    if candidate.anchor is None
                    else plan_id_by_x1[candidate.anchor.x1.value],
                    "membership_frozen_pre_score": True,
                    "completeness_plan_digest": completeness_plan_digest,
                    "completeness_plan_status": "unscored_complete_y1_plan",
                    "scored_completeness_receipt_sha256": None,
                    "core_rule_digest": rules.landscape.rule_digest,
                    "landscape_decision_rules_sha256": rules_sha256,
                    "foil_set_id": rules.foil_set_id,
                    "foil_set_sha256": rules.foil_set_sha256,
                    "upstream_digests": dict(ledger.upstream_digests),
                }
            )
    by_unique_box: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in candidate_records:
        by_unique_box[record["unique_box_id"]].append(record)
    for members in by_unique_box.values():
        ordered_ids = sorted(record["candidate_id"] for record in members)
        for record in members:
            record["unique_box_mass_semantics"]["candidate_count_for_same_geometry"] = (
                len(ordered_ids)
            )
            record["unique_box_mass_semantics"]["is_unique_box_representative"] = (
                record["candidate_id"] == ordered_ids[0]
            )
    records.extend(candidate_records)

    def sort_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
        if record["record_type"] == "free_coordinate_tree_root_request":
            return (
                record["diagnostic_owner_id"],
                record["context_id"],
                0,
                record["request_id"],
            )
        if record["record_type"] == "conditional_y1_score_plan":
            return (
                record["diagnostic_owner_id"],
                record["context_id"],
                1,
                record["x1"],
            )
        box = record["box"]
        return (
            record["diagnostic_owner_id"],
            record["context_id"],
            2,
            rules.landscape.bank_rank(record["bank_name"]),
            box["x1"],
            box["y1"],
            box["x2"],
            box["y2"],
            record["source_id"],
            record["candidate_id"],
        )

    return sorted(records, key=sort_key)


def _prepare_write_once(*paths: Path) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise FileExistsError(
            "refusing to overwrite immutable output(s): "
            + ", ".join(map(str, existing))
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def build_sorted_owner_basin_candidates(
    *,
    owner_context_ledger: str | Path,
    landscape_decision_rules: str | Path,
    bank_seeds: str | Path,
    output_jsonl: str | Path,
    receipt: str | Path,
    expected_owner_context_ledger_sha256: str,
    expected_landscape_decision_rules_sha256: str,
    expected_bank_seeds_sha256: str,
) -> Mapping[str, Any]:
    """Validate immutable inputs and write deterministic pre-score JSONL once."""

    ledger_path = Path(owner_context_ledger).expanduser().resolve(strict=True)
    rules_path = Path(landscape_decision_rules).expanduser().resolve(strict=True)
    seeds_path = Path(bank_seeds).expanduser().resolve(strict=True)
    output_path = Path(output_jsonl).expanduser().resolve()
    receipt_path = Path(receipt).expanduser().resolve()
    if output_path == receipt_path:
        raise ValueError("output_jsonl and receipt must be distinct write-once paths")
    expected_ledger_digest = _sha256(
        expected_owner_context_ledger_sha256, "expected owner/context ledger digest"
    )
    expected_rules_digest = _sha256(
        expected_landscape_decision_rules_sha256, "expected rules digest"
    )
    expected_seeds_digest = _sha256(
        expected_bank_seeds_sha256, "expected bank-seeds digest"
    )
    actual_ledger_digest = sha256_file(ledger_path)
    actual_rules_digest = sha256_file(rules_path)
    actual_seeds_digest = sha256_file(seeds_path)
    if actual_ledger_digest != expected_ledger_digest:
        raise ValueError(
            "owner/context ledger digest does not match the expected upstream digest"
        )
    if actual_rules_digest != expected_rules_digest:
        raise ValueError(
            "landscape decision rules digest does not match the sealed digest"
        )
    if actual_seeds_digest != expected_seeds_digest:
        raise ValueError(
            "bank-seeds digest does not match the expected upstream digest"
        )

    parsed_rules = _parse_rules(_read_json(rules_path, "landscape decision rules"))
    parsed_ledger = _parse_ledger_rows(
        _read_jsonl(ledger_path, "owner/context ledger"), parsed_rules
    )
    parsed_seeds = _parse_seeds(
        _read_json(seeds_path, "bank seeds"),
        rules=parsed_rules,
        rules_sha256=actual_rules_digest,
        ledger_sha256=actual_ledger_digest,
        ledger_rows=parsed_ledger,
    )
    records = build_candidate_records(
        ledger_rows=parsed_ledger,
        seeds=parsed_seeds,
        rules=parsed_rules,
        rules_sha256=actual_rules_digest,
    )
    candidate_rows = [
        row for row in records if row["record_type"] == "complete_box_candidate"
    ]
    plan_rows = [
        row for row in records if row["record_type"] == "conditional_y1_score_plan"
    ]
    free_tree_rows = [
        row
        for row in records
        if row["record_type"] == "free_coordinate_tree_root_request"
    ]
    candidate_ids = [row["candidate_id"] for row in candidate_rows]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AssertionError("content-addressed candidate IDs unexpectedly collided")
    output_bytes = b"".join(canonical_json_bytes(row) + b"\n" for row in records)
    output_digest = hashlib.sha256(output_bytes).hexdigest()
    unique_box_ids = {row["unique_box_id"] for row in candidate_rows}
    receipt_document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "materializer": "cpu_only_pre_score_candidate_membership",
        "inputs": {
            "owner_context_ledger_sha256": actual_ledger_digest,
            "landscape_decision_rules_sha256": actual_rules_digest,
            "core_rule_digest": parsed_rules.landscape.rule_digest,
            "bank_seeds_sha256": actual_seeds_digest,
            "upstream_digests": dict(parsed_rules.upstream_digests),
            "foil_set_id": parsed_rules.foil_set_id,
            "foil_set_sha256": parsed_rules.foil_set_sha256,
        },
        "counts": {
            "owner_context_rows": len(parsed_ledger),
            "conditional_y1_score_plans": len(plan_rows),
            "complete_box_candidates": len(candidate_rows),
            "free_coordinate_tree_root_requests": len(free_tree_rows),
            "unique_box_ids": len(unique_box_ids),
        },
        "candidate_membership": {
            "selection": "frozen_pre_score_only",
            "score_dependent_selection_executed": False,
            "unique_box_mass_semantics": "one_score_per_owner_context_description_box",
        },
        "production_admission": {
            "eligible": parsed_rules.landscape.can_emit_production_attestation,
            "status": (
                "production"
                if parsed_rules.landscape.can_emit_production_attestation
                else "non_production_test_fixture"
            ),
            "contract_kind": parsed_rules.landscape.contract_mode,
            "coordinate_space": dict(parsed_rules.coordinate_space),
            "geometry_identity_schema": parsed_rules.landscape.geometry_identity_schema,
        },
        "p_x1_y1_pruning": {
            **dict(parsed_rules.p_x1_y1_pruning),
            "execution_status": "declared_not_executed_by_candidate_materializer",
        },
        "free_search": {
            "declared_budget": dict(parsed_rules.free_search_budget),
            "executed": False,
            "execution_status": "declared_not_executed_by_candidate_materializer",
            "execution_owner": "runtime_landscape_scorer_free_tree_surface",
            "explicit_root_request_count": len(free_tree_rows),
            "candidate_membership_surface": "separate_from_restricted_candidate_bank",
        },
        "executable_request_contract": {
            "prefix_ends_at": "box_start_token_id",
            "fixed_coordinate_transition": "append_each_fixed_coord_token_exactly_once",
            "coordinate_token_source": "sealed_explicit_1000_entry_registry",
            "token_registry_sha256": parsed_rules.token_registry["registry_sha256"],
            "coordinate_bin_token_ids_sha256": parsed_rules.token_registry[
                "coordinate_bin_to_token_id"
            ]["coordinate_bin_token_ids_sha256"],
            "structural_row_wrapper_sha256": parsed_rules.structural_row_wrapper[
                "structural_row_wrapper_sha256"
            ],
            "analysis_channels": dict(parsed_rules.analysis_channels),
            "duplicate_raw_forward_per_policy_view_allowed": False,
        },
        "output_jsonl": {"sha256": output_digest, "row_count": len(records)},
    }
    _prepare_write_once(output_path, receipt_path)
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    with receipt_path.open("xb") as handle:
        handle.write(canonical_json_bytes(receipt_document) + b"\n")
    return receipt_document


# A readable verb for callers that frame this operation as an artifact write.
materialize_sorted_owner_basin_candidates = build_sorted_owner_basin_candidates


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-context-ledger", required=True, type=Path)
    parser.add_argument("--landscape-decision-rules", required=True, type=Path)
    parser.add_argument("--bank-seeds", required=True, type=Path)
    parser.add_argument("--output-jsonl", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    parser.add_argument("--expected-owner-context-ledger-sha256", required=True)
    parser.add_argument("--expected-landscape-decision-rules-sha256", required=True)
    parser.add_argument("--expected-bank-seeds-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = build_sorted_owner_basin_candidates(
        owner_context_ledger=args.owner_context_ledger,
        landscape_decision_rules=args.landscape_decision_rules,
        bank_seeds=args.bank_seeds,
        output_jsonl=args.output_jsonl,
        receipt=args.receipt,
        expected_owner_context_ledger_sha256=args.expected_owner_context_ledger_sha256,
        expected_landscape_decision_rules_sha256=args.expected_landscape_decision_rules_sha256,
        expected_bank_seeds_sha256=args.expected_bank_seeds_sha256,
    )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
