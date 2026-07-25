#!/usr/bin/env python3
"""Prepare and reduce the Phase Zero common-row normalization probe.

``prepare`` proves the exact singleton action shared by the pairwise and
owner-conditioned StateBanks, then writes a manifest for the existing
``run_complete_candidate_row_scoring.py`` scorer.  It never scores a model.

``reduce`` consumes receipts from that canonical scorer.  It reports raw
teacher-forced row log-probability evidence as sequence sums, target-token
means, schema/description means, coordinate means, and an equal-group
diagnostic.  STOP evidence remains a separate boundary margin.  No candidate
set is normalized into a probability distribution.
"""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research import run_complete_candidate_row_scoring as _canonical_scorer  # noqa: E402
from scripts.research.run_next_row_likelihood_change import (  # noqa: E402
    OBJECT_REF_START,
    _canonical_row_phases,
    sha256_json,
)


UNIT_ID = "2026-07-25-existing-checkpoint-transition-mechanism-decomposition"
AUDIT_SCHEMA_VERSION = "transition_phase0_candidate_scoring.common_projection_audit.v1"
REDUCTION_SCHEMA_VERSION = "transition_phase0_candidate_scoring.reduction.v1"
STATE_BANK_SCHEMA_VERSION = "coordexp.rollout_calibration.state_bank.v1"
STOP_TOKEN_ID = 151645
ROW_LENGTH_STRATA = (9, 10, 11)
SMOKE_EVENT_ID = (
    "row-local-owner-stop-image-100157-depth-1-owner-100157-567184-"
    "prefix-06e34262e295"
)

DEFAULT_PARENT_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-24-prefix-local-and-on-policy-owner-set-training"
)
DEFAULT_PAIRWISE_STATE_BANK = (
    DEFAULT_PARENT_ROOT / "event-bank-pairwise-1440-v1/state-bank"
)
DEFAULT_OWNER_CONDITIONED_STATE_BANK = (
    DEFAULT_PARENT_ROOT / "event-bank-owner-conditioned-1440-v1/state-bank"
)

CANONICAL_POPULATION = {
    "pairwise_event_count": 1440,
    "owner_conditioned_event_count": 1440,
    "common_singleton_action_count": 1440,
    "pairwise_positive_action_count": 1440,
    "owner_conditioned_positive_action_count": 1749,
    "owner_conditioned_extra_alias_action_count": 309,
    "owner_conditioned_extra_alias_event_count": 309,
}

PREPARED_MANIFEST_NAME = "candidate-scoring-manifest.json"
PROJECTION_AUDIT_NAME = "common-projection-audit.json"


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve(strict=True).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be an object")
    return value


def _tokens(value: Any, context: str) -> list[int]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise ValueError(f"{context} must be a list of integer token IDs")
    return [int(item) for item in value]


def _checked_tokens(
    value: Mapping[str, Any], token_key: str, hash_key: str, context: str
) -> tuple[list[int], str]:
    tokens = _tokens(value.get(token_key), f"{context}.{token_key}")
    declared = value.get(hash_key)
    actual = sha256_json(tokens)
    if not isinstance(declared, str) or declared != actual:
        raise ValueError(f"{context} hash mismatch for {token_key}")
    return tokens, actual


def _validate_event_record(
    record: Mapping[str, Any], *, label: str, line_number: int, checkpoint_id: str
) -> dict[str, Any]:
    context = f"{label} records line {line_number}"
    event_id = record.get("event_id")
    if not isinstance(event_id, str) or not event_id:
        raise ValueError(f"{context} lacks event_id")
    prompt_tokens, prompt_hash = _checked_tokens(
        record,
        "executed_prompt_token_ids",
        "executed_prompt_token_ids_sha256",
        f"{context} event {event_id}",
    )
    prefix_tokens, prefix_hash = _checked_tokens(
        record,
        "prefix_token_ids",
        "prefix_token_ids_sha256",
        f"{context} event {event_id}",
    )
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError(f"{context} event {event_id} requires candidates")
    candidate_ids: set[str] = set()
    normalized_candidates: list[dict[str, Any]] = []
    for candidate_index, raw_candidate in enumerate(candidates):
        candidate = dict(
            _mapping(raw_candidate, f"{context} event {event_id} candidate {candidate_index}")
        )
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id or candidate_id in candidate_ids:
            raise ValueError(
                f"{context} event {event_id} candidate IDs must be unique non-empty strings"
            )
        candidate_ids.add(candidate_id)
        candidate_tokens, candidate_hash = _checked_tokens(
            candidate,
            "token_ids",
            "token_ids_sha256",
            f"{context} event {event_id} candidate {candidate_id}",
        )
        provenance = _mapping(
            candidate.get("generation_provenance"),
            f"{context} event {event_id} candidate {candidate_id}.generation_provenance",
        )
        if provenance.get("prompt_token_ids_sha256") != prompt_hash:
            raise ValueError(
                f"{context} event {event_id} candidate {candidate_id} prompt mismatch"
            )
        if provenance.get("prefix_token_ids_sha256") != prefix_hash:
            raise ValueError(
                f"{context} event {event_id} candidate {candidate_id} prefix mismatch"
            )
        if provenance.get("checkpoint_id") != checkpoint_id:
            raise ValueError(
                f"{context} event {event_id} candidate {candidate_id} checkpoint mismatch"
            )
        candidate["token_ids"] = candidate_tokens
        candidate["token_ids_sha256"] = candidate_hash
        normalized_candidates.append(candidate)
    normalized = dict(record)
    normalized["executed_prompt_token_ids"] = prompt_tokens
    normalized["executed_prompt_token_ids_sha256"] = prompt_hash
    normalized["prefix_token_ids"] = prefix_tokens
    normalized["prefix_token_ids_sha256"] = prefix_hash
    normalized["candidates"] = normalized_candidates
    return normalized


def load_state_bank(state_bank: Path, *, label: str) -> dict[str, Any]:
    """Load a StateBank and prove its manifest, records, and token hashes."""

    root = state_bank.expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"{label} StateBank is not a directory: {root}")
    manifest_path = (root / "manifest.json").resolve(strict=True)
    manifest = _read_json(manifest_path)
    if manifest.get("schema_version") != STATE_BANK_SCHEMA_VERSION:
        raise ValueError(f"{label} expected {STATE_BANK_SCHEMA_VERSION}")
    records_name = manifest.get("records_file")
    if not isinstance(records_name, str) or not records_name:
        raise ValueError(f"{label} StateBank manifest lacks records_file")
    records_path = (root / records_name).resolve(strict=True)
    if records_path.parent != root:
        raise ValueError(f"{label} records_file must remain inside the StateBank")
    records_hash = _sha256_file(records_path)
    if manifest.get("records_sha256") != records_hash:
        raise ValueError(f"{label} StateBank records hash mismatch")
    checkpoint_id = manifest.get("source_checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise ValueError(f"{label} StateBank manifest lacks source_checkpoint_id")

    records: list[dict[str, Any]] = []
    records_by_id: dict[str, dict[str, Any]] = {}
    with records_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{label} records line {line_number} is blank")
            value = json.loads(line)
            record = _validate_event_record(
                _mapping(value, f"{label} records line {line_number}"),
                label=label,
                line_number=line_number,
                checkpoint_id=checkpoint_id,
            )
            event_id = str(record["event_id"])
            if event_id in records_by_id:
                raise ValueError(f"{label} duplicate event_id={event_id}")
            records.append(record)
            records_by_id[event_id] = record
    if manifest.get("record_count") != len(records):
        raise ValueError(
            f"{label} StateBank record-count mismatch: manifest={manifest.get('record_count')}, "
            f"observed={len(records)}"
        )
    family_counts = manifest.get("event_family_counts")
    if not isinstance(family_counts, Mapping) or family_counts.get("entity_transition") != len(records):
        raise ValueError(f"{label} StateBank event-family count mismatch")
    return {
        "label": label,
        "root": root,
        "manifest_path": manifest_path,
        "manifest_sha256": _sha256_file(manifest_path),
        "manifest": manifest,
        "records_path": records_path,
        "records_sha256": records_hash,
        "records": records,
        "records_by_id": records_by_id,
    }


def _positive_candidates(record: Mapping[str, Any], context: str) -> list[dict[str, Any]]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{context} lacks candidates")
    return [dict(item) for item in candidates if isinstance(item, Mapping) and item.get("role") == "positive"]


def _constructed_stop(record: Mapping[str, Any], context: str) -> dict[str, Any]:
    candidates = record.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError(f"{context} lacks candidates")
    matches = [
        dict(item)
        for item in candidates
        if isinstance(item, Mapping)
        and item.get("role") == "harmful"
        and item.get("harmful_kind") == "premature_terminal"
    ]
    if len(matches) != 1:
        raise ValueError(f"{context} constructed-stop mismatch: expected exactly one")
    stop = matches[0]
    if stop.get("token_ids") != [STOP_TOKEN_ID]:
        raise ValueError(f"{context} constructed-stop mismatch: token is not {STOP_TOKEN_ID}")
    if stop.get("token_ids_sha256") != sha256_json([STOP_TOKEN_ID]):
        raise ValueError(f"{context} constructed-stop hash mismatch")
    provenance = record.get("review_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{context} constructed-stop provenance is absent")
    if provenance.get("constructed_stop") is not True:
        raise ValueError(f"{context} constructed-stop flag is not true")
    if provenance.get("constructed_stop_is_not_sampled_or_on_policy") is not True:
        raise ValueError(f"{context} constructed-stop policy flag is not true")
    return stop


def _category_for_owner(record: Mapping[str, Any], owner_id: str, context: str) -> str:
    entities = record.get("physical_entities")
    if not isinstance(entities, list):
        raise ValueError(f"{context} lacks physical_entities")
    matches = [
        item
        for item in entities
        if isinstance(item, Mapping) and str(item.get("entity_id")) == owner_id
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("category"), str):
        raise ValueError(f"{context} cannot resolve one category for owner {owner_id}")
    return str(matches[0]["category"])


def _length_strata(lengths: Sequence[int]) -> dict[str, int]:
    observed = Counter(int(value) for value in lengths)
    unexpected = sorted(set(observed) - set(ROW_LENGTH_STRATA))
    if unexpected:
        raise ValueError(f"common-action row length outside 9/10/11 strata: {unexpected}")
    return {str(length): int(observed.get(length, 0)) for length in ROW_LENGTH_STRATA}


def _source_bank_audit(bank: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _mapping(bank["manifest"], "StateBank manifest")
    return {
        "state_bank": str(bank["root"]),
        "manifest": {
            "path": str(bank["manifest_path"]),
            "sha256": str(bank["manifest_sha256"]),
            "schema_version": str(manifest["schema_version"]),
        },
        "records": {
            "path": str(bank["records_path"]),
            "sha256": str(bank["records_sha256"]),
            "count": len(bank["records"]),
        },
        "bank_id": str(manifest.get("bank_id")),
        "prompt_identity_sha256": str(manifest.get("prompt_identity_sha256")),
        "source_checkpoint_id": str(manifest.get("source_checkpoint_id")),
    }


def build_common_projection(
    pairwise_state_bank: Path,
    owner_conditioned_state_bank: Path,
    *,
    selected_event_ids: Sequence[str] | None = None,
    expected_population: Mapping[str, int] | None = CANONICAL_POPULATION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the canonical scorer manifest and deterministic projection audit."""

    pairwise = load_state_bank(pairwise_state_bank, label="pairwise")
    owner_conditioned = load_state_bank(
        owner_conditioned_state_bank, label="owner-conditioned"
    )
    pair_manifest = _mapping(pairwise["manifest"], "pairwise manifest")
    owner_manifest = _mapping(owner_conditioned["manifest"], "owner-conditioned manifest")
    for key in (
        "prompt_identity_sha256",
        "source_checkpoint_id",
        "source_checkpoint",
        "split_assignments",
        "blind_image_ids",
    ):
        if pair_manifest.get(key) != owner_manifest.get(key):
            raise ValueError(f"StateBank manifest {key} mismatch")

    pair_records = _mapping(pairwise["records_by_id"], "pairwise records")
    owner_records = _mapping(owner_conditioned["records_by_id"], "owner records")
    pair_event_ids = set(pair_records)
    owner_event_ids = set(owner_records)
    if pair_event_ids != owner_event_ids:
        missing_owner = sorted(pair_event_ids - owner_event_ids)
        missing_pair = sorted(owner_event_ids - pair_event_ids)
        raise ValueError(
            "event mismatch between StateBanks: "
            f"missing_owner_conditioned={missing_owner[:3]}, missing_pairwise={missing_pair[:3]}"
        )

    projected: list[dict[str, Any]] = []
    pair_positive_count = 0
    owner_positive_count = 0
    extra_alias_count = 0
    extra_alias_event_count = 0
    for event_id in sorted(pair_event_ids):
        pair = _mapping(pair_records[event_id], f"pairwise event {event_id}")
        owner = _mapping(owner_records[event_id], f"owner-conditioned event {event_id}")
        if pair.get("event_id") != owner.get("event_id"):
            raise ValueError(f"event mismatch for {event_id}")
        if (
            pair.get("executed_prompt_token_ids") != owner.get("executed_prompt_token_ids")
            or pair.get("executed_prompt_token_ids_sha256")
            != owner.get("executed_prompt_token_ids_sha256")
        ):
            raise ValueError(f"prompt mismatch for event {event_id}")
        if (
            pair.get("prefix_token_ids") != owner.get("prefix_token_ids")
            or pair.get("prefix_token_ids_sha256") != owner.get("prefix_token_ids_sha256")
        ):
            raise ValueError(f"prefix mismatch for event {event_id}")
        for key in (
            "image",
            "split",
            "split_group_id",
            "prefix_object_row_count",
            "prefix_covered_owner_proofs",
            "physical_entities",
        ):
            if pair.get(key) != owner.get(key):
                raise ValueError(f"event lineage mismatch for {event_id}: {key}")

        pair_stop = _constructed_stop(pair, f"pairwise event {event_id}")
        owner_stop = _constructed_stop(owner, f"owner-conditioned event {event_id}")
        if pair_stop != owner_stop:
            raise ValueError(f"constructed-stop mismatch between StateBanks for event {event_id}")

        pair_positives = _positive_candidates(pair, f"pairwise event {event_id}")
        owner_positives = _positive_candidates(owner, f"owner-conditioned event {event_id}")
        pair_positive_count += len(pair_positives)
        owner_positive_count += len(owner_positives)
        if len(pair_positives) != 1:
            raise ValueError(
                f"common-action mismatch for event {event_id}: pairwise requires one positive"
            )
        common = pair_positives[0]
        try:
            _canonical_row_phases(common["token_ids"])
        except (KeyError, ValueError) as exc:
            raise ValueError(f"common-action row is non-canonical for event {event_id}") from exc
        if common.get("coverage_status") != "uncovered" or not isinstance(
            common.get("physical_owner_id"), str
        ):
            raise ValueError(
                f"common-action mismatch for event {event_id}: positive owner is unresolved"
            )
        token_matches = [
            candidate
            for candidate in owner_positives
            if candidate.get("token_ids") == common.get("token_ids")
            and candidate.get("physical_owner_id") == common.get("physical_owner_id")
        ]
        if len(token_matches) != 1 or token_matches[0] != common:
            raise ValueError(
                f"common-action mismatch for event {event_id}: exact singleton is absent or changed"
            )
        extra_aliases = [candidate for candidate in owner_positives if candidate is not token_matches[0]]
        if extra_aliases:
            extra_alias_event_count += 1
            extra_alias_count += len(extra_aliases)
        for alias in extra_aliases:
            try:
                _canonical_row_phases(alias["token_ids"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"owner-conditioned alias is non-canonical for event {event_id}") from exc
            if alias.get("physical_owner_id") != common.get("physical_owner_id"):
                raise ValueError(
                    f"owner-conditioned alias changes physical owner for event {event_id}"
                )

        image = _mapping(pair.get("image"), f"event {event_id}.image")
        image_id = image.get("image_id")
        if not isinstance(image_id, int):
            raise ValueError(f"event {event_id} image_id must be an integer")
        owner_id = str(common["physical_owner_id"])
        prompt_tokens = _tokens(
            pair["executed_prompt_token_ids"], f"event {event_id}.executed_prompt_token_ids"
        )
        prefix_tokens = _tokens(pair["prefix_token_ids"], f"event {event_id}.prefix_token_ids")
        full_prefix_hash = sha256_json([*prompt_tokens, *prefix_tokens])
        projected.append(
            {
                "event_id": event_id,
                "image_id": str(image_id),
                "image": dict(image),
                "prompt_token_ids": prompt_tokens,
                "prompt_token_ids_sha256": str(pair["executed_prompt_token_ids_sha256"]),
                "prefix_token_ids": prefix_tokens,
                "prefix_token_ids_sha256": str(pair["prefix_token_ids_sha256"]),
                "full_model_prefix_token_ids_sha256": full_prefix_hash,
                "candidate_id": str(common["candidate_id"]),
                "physical_owner_id": owner_id,
                "category": _category_for_owner(pair, owner_id, f"event {event_id}"),
                "row_token_ids": list(common["token_ids"]),
                "row_token_ids_sha256": str(common["token_ids_sha256"]),
                "row_token_count": len(common["token_ids"]),
                "constructed_stop_candidate_id": str(pair_stop["candidate_id"]),
                "constructed_stop_token_id": STOP_TOKEN_ID,
                "constructed_stop_token_ids_sha256": str(pair_stop["token_ids_sha256"]),
                "extra_owner_conditioned_aliases": [
                    {
                        "candidate_id": str(alias["candidate_id"]),
                        "physical_owner_id": str(alias["physical_owner_id"]),
                        "row_token_ids_sha256": str(alias["token_ids_sha256"]),
                        "row_token_count": len(alias["token_ids"]),
                    }
                    for alias in sorted(extra_aliases, key=lambda item: str(item["candidate_id"]))
                ],
            }
        )

    population = {
        "pairwise_event_count": len(pair_records),
        "owner_conditioned_event_count": len(owner_records),
        "common_singleton_action_count": len(projected),
        "pairwise_positive_action_count": pair_positive_count,
        "owner_conditioned_positive_action_count": owner_positive_count,
        "owner_conditioned_extra_alias_action_count": extra_alias_count,
        "owner_conditioned_extra_alias_event_count": extra_alias_event_count,
        "common_action_length_strata": _length_strata(
            [int(item["row_token_count"]) for item in projected]
        ),
    }
    if expected_population is not None:
        for key, expected in expected_population.items():
            if population.get(key) != int(expected):
                raise ValueError(
                    f"canonical population mismatch for {key}: "
                    f"expected={expected}, observed={population.get(key)}"
                )

    requested = [str(item) for item in (selected_event_ids or [])]
    if len(requested) != len(set(requested)):
        raise ValueError("selected event IDs must be unique")
    projected_by_id = {str(item["event_id"]): item for item in projected}
    missing = sorted(set(requested) - set(projected_by_id))
    if missing:
        raise ValueError(f"selected event ID is absent from common projection: {missing}")
    selected = projected if not requested else [projected_by_id[event_id] for event_id in sorted(requested)]

    images: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
    for item in selected:
        image_id = str(item["image_id"])
        image = images.get(image_id)
        if image is None:
            image = {
                "image_id": image_id,
                "base_prompt": {
                    "token_ids": list(item["prompt_token_ids"]),
                    "token_ids_sha256": str(item["prompt_token_ids_sha256"]),
                },
                "boundaries": [],
            }
            images[image_id] = image
        elif image["base_prompt"]["token_ids_sha256"] != item["prompt_token_ids_sha256"]:
            raise ValueError(f"prompt mismatch within image {image_id}")
        image["boundaries"].append(
            {
                "boundary_id": str(item["event_id"]),
                "prefix_mode": "base_prompt_plus_generated",
                "prefix": {
                    "token_ids": list(item["prefix_token_ids"]),
                    "token_ids_sha256": str(item["prefix_token_ids_sha256"]),
                },
                "candidates": [
                    {
                        "candidate_id": str(item["candidate_id"]),
                        "owner": str(item["physical_owner_id"]),
                        "category": str(item["category"]),
                        "role": "common_singleton_positive",
                        "covered": False,
                        "truth_status": "trusted_uncovered_physical_owner_common_projection",
                        "row": {
                            "token_ids": list(item["row_token_ids"]),
                            "token_ids_sha256": str(item["row_token_ids_sha256"]),
                        },
                        "source": {
                            "event_id": str(item["event_id"]),
                            "pairwise_state_bank": str(pairwise["root"]),
                            "owner_conditioned_state_bank": str(owner_conditioned["root"]),
                        },
                    }
                ],
            }
        )
    manifest = {
        "schema_version": _canonical_scorer.MANIFEST_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "projection_contract": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "selection": "requested_subset" if requested else "full_common_projection",
            "common_singleton_only": True,
            "owner_conditioned_aliases_excluded_from_scorer_manifest": True,
            "terminal_evidence_scored_separately": True,
        },
        "images": list(images.values()),
    }
    # Reuse the canonical scorer's validator so this adapter cannot drift into
    # a second row schema or scoring interface.
    _canonical_scorer.validate_manifest(
        manifest, manifest_path=Path(PREPARED_MANIFEST_NAME)
    )
    manifest_sha256 = _sha256_bytes(_json_bytes(manifest))

    selected_audit_events = [
        {
            "event_id": str(item["event_id"]),
            "image_id": str(item["image_id"]),
            "image_content_sha256": str(item["image"].get("content_sha256")),
            "prompt_token_ids_sha256": str(item["prompt_token_ids_sha256"]),
            "prefix_token_ids_sha256": str(item["prefix_token_ids_sha256"]),
            "full_model_prefix_token_ids_sha256": str(
                item["full_model_prefix_token_ids_sha256"]
            ),
            "candidate_id": str(item["candidate_id"]),
            "physical_owner_id": str(item["physical_owner_id"]),
            "category": str(item["category"]),
            "row_token_ids_sha256": str(item["row_token_ids_sha256"]),
            "row_token_count": int(item["row_token_count"]),
            "constructed_stop_candidate_id": str(
                item["constructed_stop_candidate_id"]
            ),
            "constructed_stop_token_id": int(item["constructed_stop_token_id"]),
            "constructed_stop_token_ids_sha256": str(
                item["constructed_stop_token_ids_sha256"]
            ),
            "owner_conditioned_extra_aliases": list(
                item["extra_owner_conditioned_aliases"]
            ),
        }
        for item in selected
    ]
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "source_banks": {
            "pairwise": _source_bank_audit(pairwise),
            "owner_conditioned": _source_bank_audit(owner_conditioned),
        },
        "population": population,
        "selection": {
            "mode": "requested_subset" if requested else "full_common_projection",
            "requested_event_ids": sorted(requested),
            "event_count": len(selected),
            "event_ids": [str(item["event_id"]) for item in selected],
            "common_action_length_strata": _length_strata(
                [int(item["row_token_count"]) for item in selected]
            ),
        },
        "events": selected_audit_events,
        "scorer_manifest": {
            "filename": PREPARED_MANIFEST_NAME,
            "schema_version": _canonical_scorer.MANIFEST_SCHEMA_VERSION,
            "sha256": manifest_sha256,
            "image_count": len(manifest["images"]),
            "boundary_count": len(selected),
            "candidate_count": len(selected),
        },
        "invariants": {
            "event_prompt_prefix_and_common_action_equal_across_banks": True,
            "constructed_stop_equal_and_excluded_from_candidate_rows": True,
            "owner_conditioned_aliases_are_audited_but_not_scored": True,
        },
        "forbidden_interpretations": [
            "independently scored candidate rows are not a normalized candidate distribution",
            "the STOP boundary margin is not a full-row normalized score",
            "fixed-prefix teacher-forced evidence is not final-set improvement",
        ],
    }
    return manifest, audit


def _write_immutable_json(path: Path, value: object) -> None:
    destination = path.expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_json_bytes(value))


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.expanduser().resolve()
    manifest_path = output_root / PREPARED_MANIFEST_NAME
    audit_path = output_root / PROJECTION_AUDIT_NAME
    if manifest_path.exists() or audit_path.exists():
        raise FileExistsError(f"refusing to overwrite artifacts under {output_root}")
    manifest, audit = build_common_projection(
        args.pairwise_state_bank,
        args.owner_conditioned_state_bank,
        selected_event_ids=args.event_id,
        expected_population=CANONICAL_POPULATION,
    )
    _write_immutable_json(manifest_path, manifest)
    if _sha256_file(manifest_path) != audit["scorer_manifest"]["sha256"]:
        raise RuntimeError("written scorer manifest hash differs from projection audit")
    _write_immutable_json(audit_path, audit)
    return audit


ROW_METRIC_KEYS = (
    "sequence_log_probability_sum",
    "target_token_mean_log_probability",
    "schema_description_mean_log_probability",
    "coordinate_mean_log_probability",
    "equal_group_mean_log_probability",
)


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    return result


def _assert_close(observed: Any, expected: float, context: str) -> None:
    value = _finite_float(observed, context)
    if not math.isclose(value, expected, rel_tol=1e-6, abs_tol=1e-5):
        raise ValueError(f"{context} mismatch: observed={value}, recomputed={expected}")


def reduce_candidate_score(score: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all row-normalization diagnostics from canonical token traces."""

    if "row_entry_vs_terminal" in score:
        raise ValueError("candidate row score illegally contains terminal boundary evidence")
    row_tokens = _tokens(score.get("row_token_ids"), "candidate.row_token_ids")
    declared_hash = score.get("token_ids_sha256")
    row_hash = sha256_json(row_tokens)
    if declared_hash != row_hash:
        raise ValueError("candidate row hash mismatch")
    if score.get("token_count") != len(row_tokens):
        raise ValueError("candidate row token-count mismatch")
    if len(row_tokens) not in ROW_LENGTH_STRATA:
        raise ValueError("candidate row length is outside the declared 9/10/11 strata")
    phases = _canonical_row_phases(row_tokens)
    token_log_probabilities = score.get("token_log_probabilities")
    if not isinstance(token_log_probabilities, list) or len(token_log_probabilities) != len(
        row_tokens
    ):
        raise ValueError("candidate token log-probability count mismatch")
    values = [
        _finite_float(value, f"candidate.token_log_probabilities[{index}]")
        for index, value in enumerate(token_log_probabilities)
    ]
    for phase_name, indices in phases.items():
        phase = _mapping(score.get(phase_name), f"candidate.{phase_name}")
        phase_values = [values[index] for index in indices]
        expected_sum = math.fsum(phase_values)
        expected_mean = expected_sum / len(phase_values)
        if phase.get("count") != len(indices):
            raise ValueError(f"candidate.{phase_name} phase-count mismatch")
        _assert_close(phase.get("sum"), expected_sum, f"candidate.{phase_name}.sum")
        _assert_close(phase.get("mean"), expected_mean, f"candidate.{phase_name}.mean")

    coordinate_indices = [
        phases[axis][0] for axis in ("x1", "y1", "x2", "y2")
    ]
    if len(set(coordinate_indices)) != 4:
        raise ValueError("candidate coordinate phase does not contain four unique tokens")
    schema_description_indices = [
        index for index in phases["full_row"] if index not in set(coordinate_indices)
    ]
    coordinate_values = [values[index] for index in coordinate_indices]
    schema_description_values = [values[index] for index in schema_description_indices]
    sequence_sum = math.fsum(values)
    token_mean = sequence_sum / len(values)
    schema_description_mean = math.fsum(schema_description_values) / len(
        schema_description_values
    )
    coordinate_mean = math.fsum(coordinate_values) / len(coordinate_values)
    equal_group = 0.5 * schema_description_mean + 0.5 * coordinate_mean
    return {
        "sequence_log_probability_sum": sequence_sum,
        "target_token_mean_log_probability": token_mean,
        "schema_description_mean_log_probability": schema_description_mean,
        "coordinate_mean_log_probability": coordinate_mean,
        "equal_group_mean_log_probability": equal_group,
        "denominators": {
            "target_token_count": len(values),
            "schema_description_token_count": len(schema_description_values),
            "coordinate_token_count": len(coordinate_values),
        },
        "phase_counts": {
            **{name: len(indices) for name, indices in phases.items()},
            "schema_description": len(schema_description_values),
            "coordinate": len(coordinate_values),
        },
    }


def aggregate_row_normalization(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate already reduced rows with exactly equal event weight."""

    event_count = len(rows)
    denominators = {
        "event_count": event_count,
        "target_token_count": sum(
            int(_mapping(row["denominators"], "row denominators")["target_token_count"])
            for row in rows
        ),
        "schema_description_token_count": sum(
            int(
                _mapping(row["denominators"], "row denominators")[
                    "schema_description_token_count"
                ]
            )
            for row in rows
        ),
        "coordinate_token_count": sum(
            int(_mapping(row["denominators"], "row denominators")["coordinate_token_count"])
            for row in rows
        ),
    }
    means = {
        key: (
            None
            if not rows
            else math.fsum(_finite_float(row[key], f"row.{key}") for row in rows)
            / event_count
        )
        for key in ROW_METRIC_KEYS
    }
    return {
        "aggregation": "equal_event_weighted_mean",
        "denominators": denominators,
        "equal_event_weighted_means": means,
    }


def _aggregate_stop(events: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "row_entry_log_probability",
        "stop_log_probability",
        "row_entry_minus_stop_log_probability",
    )
    count = len(events)
    return {
        "aggregation": "equal_event_weighted_mean",
        "denominators": {"event_count": count, "boundary_decision_count": count},
        "equal_event_weighted_means": {
            key: (
                None
                if not events
                else math.fsum(_finite_float(event[key], f"stop.{key}") for event in events)
                / count
            )
            for key in keys
        },
    }


def _audit_events(audit: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if audit.get("schema_version") != AUDIT_SCHEMA_VERSION:
        raise ValueError(f"expected {AUDIT_SCHEMA_VERSION}")
    events = audit.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("projection audit requires non-empty events")
    by_id: dict[str, Mapping[str, Any]] = {}
    for index, event in enumerate(events):
        item = _mapping(event, f"audit.events[{index}]")
        event_id = item.get("event_id")
        if not isinstance(event_id, str) or not event_id or event_id in by_id:
            raise ValueError("projection audit event IDs must be unique")
        by_id[event_id] = item
    selection = _mapping(audit.get("selection"), "audit.selection")
    if selection.get("event_count") != len(by_id):
        raise ValueError("projection audit selection count mismatch")
    return by_id


def _reduce_receipt_group(
    label: str,
    audit: Mapping[str, Any],
    receipts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not receipts:
        raise ValueError(f"score receipt group {label!r} is empty")
    expected_events = _audit_events(audit)
    scorer_manifest = _mapping(audit.get("scorer_manifest"), "audit.scorer_manifest")
    expected_manifest_hash = scorer_manifest.get("sha256")
    observed: dict[str, dict[str, Any]] = {}
    for receipt_index, receipt in enumerate(receipts):
        context = f"score receipt {label}[{receipt_index}]"
        if receipt.get("schema_version") != _canonical_scorer.RECEIPT_SCHEMA_VERSION:
            raise ValueError(
                f"{context} expected {_canonical_scorer.RECEIPT_SCHEMA_VERSION}"
            )
        manifest = _mapping(receipt.get("manifest"), f"{context}.manifest")
        if manifest.get("sha256") != expected_manifest_hash:
            raise ValueError(f"{context} scorer-manifest hash mismatch")
        runtime = _mapping(receipt.get("runtime"), f"{context}.runtime")
        if runtime.get("repetition_penalty_processing") is not False:
            raise ValueError(
                f"{context} is not raw canonical scorer evidence: repetition processing changed"
            )
        images = receipt.get("images")
        if not isinstance(images, list):
            raise ValueError(f"{context} lacks images")
        for image_index, raw_image in enumerate(images):
            image = _mapping(raw_image, f"{context}.images[{image_index}]")
            image_id = str(image.get("image_id"))
            boundaries = image.get("boundaries")
            if not isinstance(boundaries, list):
                raise ValueError(f"{context} image {image_id} lacks boundaries")
            for boundary_index, raw_boundary in enumerate(boundaries):
                boundary = _mapping(
                    raw_boundary, f"{context} image {image_id} boundary {boundary_index}"
                )
                event_id = boundary.get("boundary_id")
                if not isinstance(event_id, str) or event_id not in expected_events:
                    raise ValueError(f"{context} contains unexpected event {event_id!r}")
                if event_id in observed:
                    raise ValueError(f"duplicate score receipt event {event_id} in group {label}")
                expected = expected_events[event_id]
                if image_id != str(expected.get("image_id")):
                    raise ValueError(f"{context} event {event_id} image mismatch")
                if boundary.get("prefix_token_ids_sha256") != expected.get(
                    "full_model_prefix_token_ids_sha256"
                ):
                    raise ValueError(f"{context} event {event_id} prefix mismatch")
                scores = boundary.get("candidate_scores")
                if not isinstance(scores, list) or len(scores) != 1:
                    raise ValueError(
                        f"{context} event {event_id} requires one common candidate score"
                    )
                score = _mapping(scores[0], f"{context} event {event_id}.candidate")
                if score.get("candidate_id") != expected.get("candidate_id"):
                    raise ValueError(f"{context} event {event_id} common-action ID mismatch")
                if score.get("owner") != expected.get("physical_owner_id"):
                    raise ValueError(f"{context} event {event_id} physical-owner mismatch")
                if score.get("token_ids_sha256") != expected.get("row_token_ids_sha256"):
                    raise ValueError(f"{context} event {event_id} common-action hash mismatch")
                row = reduce_candidate_score(score)
                terminal = _mapping(
                    boundary.get("terminal_boundary"),
                    f"{context} event {event_id}.terminal_boundary",
                )
                if terminal.get("row_entry_token_id") != OBJECT_REF_START:
                    raise ValueError(f"{context} event {event_id} row-entry token mismatch")
                if terminal.get("terminal_token_id") != expected.get(
                    "constructed_stop_token_id"
                ):
                    raise ValueError(f"{context} event {event_id} constructed-stop mismatch")
                stop = {
                    "row_entry_log_probability": _finite_float(
                        terminal.get("row_entry_log_probability"),
                        f"{context} event {event_id}.row_entry_log_probability",
                    ),
                    "stop_log_probability": _finite_float(
                        terminal.get("terminal_log_probability"),
                        f"{context} event {event_id}.terminal_log_probability",
                    ),
                    "row_entry_minus_stop_log_probability": _finite_float(
                        terminal.get("row_entry_minus_terminal"),
                        f"{context} event {event_id}.row_entry_minus_terminal",
                    ),
                }
                _assert_close(
                    stop["row_entry_minus_stop_log_probability"],
                    stop["row_entry_log_probability"] - stop["stop_log_probability"],
                    f"{context} event {event_id}.terminal margin",
                )
                observed[event_id] = {
                    "event_id": event_id,
                    "image_id": image_id,
                    "candidate_id": str(score["candidate_id"]),
                    "physical_owner_id": str(score["owner"]),
                    "row_token_count": int(row["denominators"]["target_token_count"]),
                    "row_normalization": row,
                    "stop_boundary": stop,
                }
    missing = sorted(set(expected_events) - set(observed))
    if missing:
        raise ValueError(f"score receipt group {label!r} is missing events: {missing[:5]}")
    events = [observed[event_id] for event_id in sorted(observed)]
    row_values = [event["row_normalization"] for event in events]
    length_strata: dict[str, Any] = {}
    for length in ROW_LENGTH_STRATA:
        stratum = [
            event["row_normalization"]
            for event in events
            if event["row_token_count"] == length
        ]
        length_strata[str(length)] = aggregate_row_normalization(stratum)
    return {
        "receipt_count": len(receipts),
        "event_count": len(events),
        "events": events,
        "row_normalization": aggregate_row_normalization(row_values),
        "length_strata": length_strata,
        "stop_boundary": _aggregate_stop([event["stop_boundary"] for event in events]),
    }


def reduce_score_receipts(
    audit: Mapping[str, Any],
    receipt_groups: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    receipt_sources: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Reduce one or more complete score-receipt groups deterministically."""

    _audit_events(audit)
    if not receipt_groups:
        raise ValueError("at least one labeled score receipt is required")
    arms: dict[str, Any] = {}
    for label in sorted(receipt_groups):
        if not label:
            raise ValueError("score receipt label must not be empty")
        arm = _reduce_receipt_group(label, audit, receipt_groups[label])
        if receipt_sources is not None:
            arm["source_receipts"] = list(receipt_sources.get(label, []))
        arms[label] = arm
    return {
        "schema_version": REDUCTION_SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "projection_audit": {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "scorer_manifest_sha256": audit["scorer_manifest"]["sha256"],
            "event_count": audit["selection"]["event_count"],
        },
        "score_semantics": {
            "source": (
                "canonical complete-candidate-row scorer float32 log-softmax "
                "of raw language-model-head logits"
            ),
            "row_evidence": "teacher-forced autoregressive token log probabilities",
            "aggregation": "equal event weight",
            "equal_group_diagnostic": (
                "0.5 * schema/description mean + 0.5 * coordinate mean"
            ),
            "stop_evidence": "separate row-entry minus STOP boundary log-probability margin",
        },
        "arms": arms,
        "forbidden_interpretations": [
            "candidate rows are not normalized into a probability distribution",
            "STOP margins are not combined with complete-row normalized scores",
            "policy-processed generation log probabilities must not be relabeled as raw LM evidence",
            "fixed-prefix teacher-forced evidence is not a free-rollout final-set outcome",
        ],
    }


def _parse_receipt_spec(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError("--score-receipt must have LABEL=PATH form")
    label, path = raw.split("=", 1)
    if not label or not path:
        raise ValueError("--score-receipt must have non-empty LABEL=PATH values")
    return label, Path(path)


def reduce(args: argparse.Namespace) -> dict[str, Any]:
    audit_path = args.audit.expanduser().resolve(strict=True)
    audit = _read_json(audit_path)
    groups: dict[str, list[Mapping[str, Any]]] = {}
    sources: dict[str, list[Mapping[str, Any]]] = {}
    for raw in args.score_receipt:
        label, raw_path = _parse_receipt_spec(raw)
        path = raw_path.expanduser().resolve(strict=True)
        groups.setdefault(label, []).append(_read_json(path))
        sources.setdefault(label, []).append(
            {"path": str(path), "sha256": _sha256_file(path)}
        )
    result = reduce_score_receipts(audit, groups, receipt_sources=sources)
    result["projection_audit"]["path"] = str(audit_path)
    result["projection_audit"]["sha256"] = _sha256_file(audit_path)
    _write_immutable_json(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser(
        "prepare", help="prove the common projection and write the canonical scorer manifest"
    )
    prepare_parser.add_argument(
        "--pairwise-state-bank", type=Path, default=DEFAULT_PAIRWISE_STATE_BANK
    )
    prepare_parser.add_argument(
        "--owner-conditioned-state-bank",
        type=Path,
        default=DEFAULT_OWNER_CONDITIONED_STATE_BANK,
    )
    prepare_parser.add_argument("--output-root", type=Path, required=True)
    prepare_parser.add_argument(
        "--event-id",
        action="append",
        default=None,
        help=(
            "emit only this already-audited common event; repeat as needed. "
            f"The one-event schema smoke uses {SMOKE_EVENT_ID}."
        ),
    )
    prepare_parser.set_defaults(handler=prepare)

    reduce_parser = subparsers.add_parser(
        "reduce", help="reduce canonical complete-row score receipts without rescoring"
    )
    reduce_parser.add_argument("--audit", type=Path, required=True)
    reduce_parser.add_argument(
        "--score-receipt",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="canonical scorer receipt; repeat a label to join immutable worker shards",
    )
    reduce_parser.add_argument("--output", type=Path, required=True)
    reduce_parser.set_defaults(handler=reduce)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
