"""Private compiler from explicit adjudicator decisions to final ledger seal."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from typing import Any

from src.analysis.spatial_scope_history import review_ledger as _core
from src.analysis.spatial_scope_history.cohort_ledger import canonical_json_text
from src.eval.detection_categories import (
    COCO_80_CATEGORY_NAMESPACE_SHA256,
    COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME,
)

ADJUDICATOR_DECISION_SCHEMA_VERSION = "dense-union-51.adjudicator-decision.v1"
AUDIT_LEDGER_SCHEMA_VERSION = "dense-union-51.audit-augmented-ledger.v1"
FINAL_LEDGER_SEAL_SCHEMA_VERSION = "dense-union-51.final-ledger-seal.v1"
SUPERSEDED_V1_OFFICIAL_INDIVIDUAL_LEDGER_SHA256 = (
    "2b5b7356778c3383a499a80c7e383e86b239768f24b3031a3906f2e4e9e2f354"
)
_ACCEPTED_OUTCOMES = {"accept-reviewer-proposal", "accept-distinct-new-instance"}
# fmt: off
_OUTCOME_STATE = {
    "accept-reviewer-proposal": "accepted", "accept-official-only": "accepted",
    "accept-distinct-new-instance": "accepted", "ambiguous": "ambiguous",
    "partial": "partial", "crowd": "crowd", "out-of-scope": "out-of-scope",
    "reject": "rejected",
}
_DECISION_FIELDS = {
    "adjudication_group_sha256", "adjudication_identifier", "adjudication_queue_sha256",
    "adjudicator_identifier", "candidate_categories", "decision_outcome",
    "final_normalized_category_name", "final_official_coco_category_id",
    "final_source_canvas_box_xyxy", "final_state", "provenance_decision",
    "rationale", "reason_code", "schema_version", "source_digest_binding",
}
_BINDING_KEYS = (
    "category_namespace_sha256", "ontology_sha256", "official_crowd_ledger_sha256",
    "official_individual_ledger_sha256", "packet_sha256", "review_queue_sha256",
    "reviewer_label_sha256_by_role",
)
_PROVENANCE_DECISIONS = {
    "official_annotation", "official_crowd_region", "reviewer_agreement",
    "reviewer_one_only", "reviewer_two_only", "reviewer_disagreement",
    "adjudicator_override",
}
_REASON_CODES = {
    "none", "category_not_unique", "instance_not_separable", "boundary_not_reproducible",
    "occluded_but_boxable", "truncated_but_boxable", "non_coco80",
    "official_duplicate", "reviewer_disagreement", "adjudicator_override",
}
# fmt: on


@dataclass(frozen=True)
class FinalReviewLedgerArtifacts:
    adjudication_jsonl: bytes
    audit_augmented_ledger_jsonl: bytes
    ledger_seal_json: bytes


def _validated_context_and_queue(inputs: Mapping[str, Any]) -> tuple[Any, bytes, str]:
    context = _core._validate_review_inputs(
        **{
            key: inputs[key]
            for key in (
                "review_queue_jsonl",
                "reviewer_one_labels_jsonl",
                "reviewer_two_labels_jsonl",
                "official_individual_ledger_jsonl",
                "official_crowd_ledger_jsonl",
                "expected_review_queue_sha256",
                "expected_packet_sha256",
                "expected_ontology_sha256",
            )
        }
    )
    if context.official_individual_ledger_sha256 == (
        SUPERSEDED_V1_OFFICIAL_INDIVIDUAL_LEDGER_SHA256
    ):
        raise ValueError("superseded readiness-v1 official individual ledger")
    correction_digest = _validate_optional_correction_receipt(inputs, context)
    expected_queue = _core._jsonl_bytes(_core._adjudication_queue_rows(context))
    return context, expected_queue, correction_digest or ""


def build_bound_adjudication_queue(**inputs: Any) -> bytes:
    """Build a queue only after readiness-v2 sources and correction are bound."""

    _, queue, correction_digest = _validated_context_and_queue(inputs)
    if not correction_digest:
        raise ValueError("bound queue materialization requires a correction receipt")
    return queue


def assemble_final_review_ledger(**inputs: Any) -> FinalReviewLedgerArtifacts:
    """Compile only a complete, source-bound, explicit decision artifact."""

    context, expected_queue, correction_digest = _validated_context_and_queue(inputs)
    queue_payload = inputs["adjudication_queue_jsonl"]
    if queue_payload != expected_queue:
        raise ValueError("adjudication queue does not match the validated sources")
    queue_rows = _core._parse_canonical_jsonl(
        queue_payload, artifact_name="adjudication queue"
    )
    decision_payload = inputs["adjudicator_decisions_jsonl"]
    decisions = _validate_decisions(
        decision_payload,
        queue_rows=queue_rows,
        queue_sha256=_core._sha256_bytes(queue_payload),
        image_facts=context.image_facts,
    )
    ledger_version = _core._require_nonempty(
        inputs["ledger_version"], field="ledger_version"
    )
    created = _parse_timestamp(inputs["seal_created_at"], field="seal_created_at")
    earliest = _parse_timestamp(
        inputs["earliest_permitted_metric_run_start"],
        field="earliest_permitted_metric_run_start",
    )
    if earliest < created:
        raise ValueError("earliest metric-run start precedes final seal creation")
    official = _core._parse_canonical_jsonl(
        inputs["official_individual_ledger_jsonl"],
        artifact_name="official individual ledger",
    )
    crowd = _core._parse_canonical_jsonl(
        inputs["official_crowd_ledger_jsonl"], artifact_name="official crowd ledger"
    )
    ledger_rows = _ledger_rows(queue_rows, decisions, official, crowd, ledger_version)
    ledger_payload = _core._jsonl_bytes(ledger_rows)
    state_counts: dict[str, int] = defaultdict(int)
    for row in ledger_rows:
        state_counts[str(row["final_state"])] += 1
    # fmt: off
    source_digests = {
        "adjudication_queue_jsonl": _core._sha256_bytes(queue_payload),
        "official_crowd_ledger_jsonl": context.official_crowd_ledger_sha256,
        "official_individual_ledger_jsonl": context.official_individual_ledger_sha256,
        "review_queue_jsonl": context.review_queue_sha256,
        "reviewer_one_labels_jsonl": context.reviewer_label_sha256_by_role["reviewer-one"],
        "reviewer_two_labels_jsonl": context.reviewer_label_sha256_by_role["reviewer-two"],
    }
    # fmt: on
    if correction_digest:
        source_digests["pre_seal_ordering_correction_receipt_json"] = correction_digest
    seal = {
        "artifact_digests": {
            "adjudication.jsonl": _core._sha256_bytes(decision_payload),
            "audit-augmented-ledger.jsonl": _core._sha256_bytes(ledger_payload),
        },
        "category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "counts": {
            "adjudication_decisions": len(decisions),
            "audit_ledger_rows": len(ledger_rows),
            "final_state_counts": dict(sorted(state_counts.items())),
        },
        "earliest_permitted_metric_run_start": inputs[
            "earliest_permitted_metric_run_start"
        ],
        "ledger_version": ledger_version,
        "reviewer_role_identifiers": list(_core.REVIEWER_ROLES),
        "schema_version": FINAL_LEDGER_SEAL_SCHEMA_VERSION,
        "seal_created_at": inputs["seal_created_at"],
        "source_digests": source_digests,
    }
    return FinalReviewLedgerArtifacts(
        adjudication_jsonl=decision_payload,
        audit_augmented_ledger_jsonl=ledger_payload,
        ledger_seal_json=(canonical_json_text(seal) + "\n").encode(),
    )


def _validate_optional_correction_receipt(
    inputs: Mapping[str, Any], context: Any
) -> str | None:
    payload = inputs.get("pre_seal_ordering_correction_receipt_json")
    expected = inputs.get("expected_pre_seal_ordering_correction_receipt_sha256")
    if payload is None and expected is None:
        return None
    if not isinstance(payload, bytes) or not isinstance(expected, str):
        raise ValueError(
            "correction receipt bytes and expected digest are both required"
        )
    _core._require_sha256(expected, field="correction receipt expected digest")
    digest = _core._sha256_bytes(payload)
    if digest != expected:
        raise ValueError("correction receipt digest mismatch")
    try:
        receipt = json.loads(payload)
    except (UnicodeDecodeError, ValueError) as exc:
        raise ValueError("correction receipt is not valid JSON") from exc
    if (
        not isinstance(receipt, Mapping)
        or payload != (canonical_json_text(receipt) + "\n").encode()
    ):
        raise ValueError("correction receipt is not canonical JSON")
    if (
        receipt.get("schema_version")
        != "spatial_scope_history.pre_seal_ordering_correction.v1"
        or receipt.get("operational_status") != "pre-seal-ordering-correction-complete"
    ):
        raise ValueError("correction receipt contract mismatch")
    comparisons = {
        row.get("name"): row
        for row in receipt.get("artifact_comparisons", [])
        if isinstance(row, Mapping)
    }
    # fmt: off
    required = {
        "official-individual-ledger.jsonl": context.official_individual_ledger_sha256,
        "official-crowd-ignore-ledger.jsonl": context.official_crowd_ledger_sha256,
        "review-queue.jsonl": context.review_queue_sha256,
        "reviewer-one-labels.jsonl": context.reviewer_label_sha256_by_role["reviewer-one"],
        "reviewer-two-labels.jsonl": context.reviewer_label_sha256_by_role["reviewer-two"],
    }
    # fmt: on
    if any(
        comparisons.get(name, {}).get("new_sha256") != sha
        for name, sha in required.items()
    ):
        raise ValueError(
            "correction receipt does not bind the selected source artifacts"
        )
    official = comparisons["official-individual-ledger.jsonl"]
    if (
        official.get("old_sha256") != SUPERSEDED_V1_OFFICIAL_INDIVIDUAL_LEDGER_SHA256
        or official.get("classification") != "changed_expected"
    ):
        raise ValueError("correction receipt does not supersede readiness-v1")
    if (
        receipt.get("supersession_policy")
        != "readiness-v1 remains immutable provenance and must not become metric-bearing; readiness-v2 is the only candidate for final review ledger sealing."
    ):
        raise ValueError("correction receipt supersession policy mismatch")
    return digest


def _validate_decisions(
    payload: bytes,
    *,
    queue_rows: Sequence[Mapping[str, Any]],
    queue_sha256: str,
    image_facts: Mapping[int, Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    rows = tuple(
        _core._parse_canonical_jsonl(payload, artifact_name="adjudicator decisions")
    )
    queue = {row["adjudication_identifier"]: row for row in queue_rows}
    seen: set[str] = set()
    for decision in rows:
        _core._require_exact_fields(
            decision, _DECISION_FIELDS, record="adjudicator decision"
        )
        identifier = _core._require_nonempty(
            decision["adjudication_identifier"], field="adjudication_identifier"
        )
        if identifier in seen:
            raise ValueError("duplicate adjudicator decision")
        seen.add(identifier)
        group = queue.get(identifier)
        if group is None:
            raise ValueError("orphan adjudicator decision")
        if decision["schema_version"] != ADJUDICATOR_DECISION_SCHEMA_VERSION:
            raise ValueError("adjudicator decision schema version mismatch")
        if decision["adjudication_queue_sha256"] != queue_sha256:
            raise ValueError("adjudicator decision queue digest drift")
        expected_group = _core._sha256_bytes(canonical_json_text(group).encode())
        if decision["adjudication_group_sha256"] != expected_group:
            raise ValueError("adjudicator decision group digest drift")
        if decision["source_digest_binding"] != {
            key: group[key] for key in _BINDING_KEYS
        }:
            raise ValueError("adjudicator decision source digest binding drift")
        for field in (
            "adjudicator_identifier",
            "provenance_decision",
            "reason_code",
            "rationale",
        ):
            _core._require_nonempty(decision[field], field=field)
        if decision["provenance_decision"] not in _PROVENANCE_DECISIONS:
            raise ValueError("unknown adjudication provenance decision")
        if decision["reason_code"] not in _REASON_CODES:
            raise ValueError("unknown adjudication reason code")
        _validate_semantics(decision, group=group, image_facts=image_facts)
    if seen != set(queue):
        raise ValueError("missing adjudicator decision")
    if [row["adjudication_identifier"] for row in rows] != list(queue):
        raise ValueError("adjudicator decisions are not in canonical queue order")
    return rows


def _validate_semantics(
    decision: Mapping[str, Any],
    *,
    group: Mapping[str, Any],
    image_facts: Mapping[int, Mapping[str, Any]],
) -> None:
    outcome = decision["decision_outcome"]
    if outcome not in _OUTCOME_STATE:
        raise ValueError("unknown adjudicator decision outcome")
    if (
        group["group_kind"] in {"official-individual", "official-crowd-region"}
        and outcome != "accept-official-only"
    ):
        raise ValueError("official source group must use accept-official-only")
    linked = (
        group["linked_official_object_identifiers"]
        + group["linked_official_crowd_region_identifiers"]
    )
    if outcome in _ACCEPTED_OUTCOMES and group["group_kind"] != "reviewer-proposal":
        raise ValueError(
            "accepted audit addition must originate from a reviewer proposal"
        )
    if outcome == "accept-reviewer-proposal" and linked:
        raise ValueError("official-linked proposal requires distinct-instance outcome")
    if outcome == "accept-distinct-new-instance" and not linked:
        raise ValueError("distinct-instance outcome requires an official link")
    if outcome == "accept-official-only" and not linked:
        raise ValueError("official-only outcome requires an official link")
    expected_state = _OUTCOME_STATE[outcome]
    if (
        outcome == "accept-official-only"
        and group["group_kind"] == "official-crowd-region"
    ):
        expected_state = "crowd"
    if decision["final_state"] != expected_state:
        raise ValueError("decision outcome and final state disagree")
    candidates = _core._validate_candidate_categories(decision["candidate_categories"])
    name = decision["final_normalized_category_name"]
    category_id = decision["final_official_coco_category_id"]
    box = decision["final_source_canvas_box_xyxy"]
    if outcome in _ACCEPTED_OUTCOMES | {"partial", "crowd"}:
        _core._validate_category_pair(name, category_id)
        if candidates:
            raise ValueError("resolved decision cannot retain candidate categories")
    elif outcome == "ambiguous":
        if not candidates or name is not None or category_id is not None:
            raise ValueError(
                "ambiguous decision requires candidates and null final category"
            )
    elif name is not None or category_id is not None or candidates:
        raise ValueError("non-object decision must have null category fields")
    if outcome in {"accept-official-only", "reject"}:
        if box is not None:
            raise ValueError("non-ledger decision must have a null final box")
        return
    if box is None and outcome != "partial":
        raise ValueError("final decision requires a source-canvas box")
    if box is not None:
        facts = image_facts[int(group["image_id"])]
        _core._validate_box_for_image(
            box,
            state="partial" if outcome == "partial" else "accepted",
            reason="boundary_not_reproducible" if outcome == "partial" else "none",
            width=int(facts["source_image_width"]),
            height=int(facts["source_image_height"]),
        )


def _ledger_rows(
    queue_rows: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
    official: Sequence[Mapping[str, Any]],
    crowd: Sequence[Mapping[str, Any]],
    ledger_version: str,
) -> list[Mapping[str, Any]]:
    queue = {row["adjudication_identifier"]: row for row in queue_rows}
    source_adjudications = {
        identifier: row["adjudication_identifier"]
        for row in queue_rows
        for identifier in row["linked_official_object_identifiers"]
        + row["linked_official_crowd_region_identifiers"]
        if row["group_kind"] in {"official-individual", "official-crowd-region"}
    }
    rows = [
        _source_row(
            source,
            ledger_version,
            state,
            provenance,
            source_adjudications[source["object_or_region_identifier"]],
        )
        for sources, state, provenance in (
            (official, "accepted", "official_annotation"),
            (crowd, "crowd", "official_crowd_region"),
        )
        for source in sources
    ]
    additions = [
        decision
        for decision in decisions
        if decision["decision_outcome"] in _ACCEPTED_OUTCOMES
    ]
    additions.sort(key=lambda decision: _addition_key(decision, queue))
    ordinals: dict[int, int] = defaultdict(int)
    for decision in additions:
        group = queue[decision["adjudication_identifier"]]
        image_id = int(group["image_id"])
        ordinals[image_id] += 1
        rows.append(
            _decision_row(
                decision,
                group,
                ledger_version,
                f"audit:{ledger_version}:{image_id}:{ordinals[image_id]:04d}",
            )
        )
    ignored = _ACCEPTED_OUTCOMES | {"accept-official-only", "reject"}
    for decision in decisions:
        if decision["decision_outcome"] not in ignored:
            group = queue[decision["adjudication_identifier"]]
            identifier = (
                f"audit-decision:{ledger_version}:{decision['adjudication_identifier']}"
            )
            rows.append(_decision_row(decision, group, ledger_version, identifier))
    rows.sort(key=lambda row: (row["image_id"], row["object_identifier"]))
    return rows


def _addition_key(
    decision: Mapping[str, Any], queue: Mapping[str, Mapping[str, Any]]
) -> tuple[Any, ...]:
    box = decision["final_source_canvas_box_xyxy"]
    linked = queue[decision["adjudication_identifier"]][
        "linked_reviewer_label_identifiers"
    ]
    return (
        decision["final_official_coco_category_id"],
        box[1],
        box[0],
        box[3],
        box[2],
        tuple(linked),
    )


def _source_row(
    source: Mapping[str, Any],
    version: str,
    state: str,
    provenance: str,
    adjudication_id: str,
) -> dict[str, Any]:
    # fmt: off
    return _row(
        version=version, image_id=source["image_id"], image_sha=source["source_image_sha256"],
        identifier=source["object_or_region_identifier"], name=source["normalized_category_name"],
        category_id=source["official_coco_category_id"], box=source["geometry"]["clipped_source_corners_xyxy"],
        state=state, provenance=provenance, source_ids=[source["object_or_region_identifier"]],
        adjudication_id=adjudication_id, outcome="accept-official-only",
    )
    # fmt: on


def _decision_row(
    decision: Mapping[str, Any], group: Mapping[str, Any], version: str, identifier: str
) -> dict[str, Any]:
    sources = sorted(
        group["linked_reviewer_label_identifiers"]
        + group["linked_official_object_identifiers"]
        + group["linked_official_crowd_region_identifiers"]
    )
    # fmt: off
    return _row(
        version=version, image_id=group["image_id"], image_sha=group["image_sha256"],
        identifier=identifier, name=decision["final_normalized_category_name"],
        category_id=decision["final_official_coco_category_id"], box=decision["final_source_canvas_box_xyxy"],
        state=decision["final_state"], provenance=decision["provenance_decision"],
        source_ids=sources, adjudication_id=decision["adjudication_identifier"],
        outcome=decision["decision_outcome"],
    )
    # fmt: on


def _row(*, version: str, image_id: int, image_sha: str, identifier: str,
         name: str | None, category_id: int | None, box: Any, state: str,
         provenance: str, source_ids: list[str], adjudication_id: str | None,
         outcome: str) -> dict[str, Any]:  # fmt: skip
    # fmt: off
    return {
        "adjudication_identifier": adjudication_id, "category_namespace_sha256": COCO_80_CATEGORY_NAMESPACE_SHA256,
        "decision_outcome": outcome, "evaluator_local_category_id": None if name is None else COCO_80_EVALUATOR_LOCAL_CATEGORY_ID_BY_NAME[name],
        "final_state": state, "image_id": image_id, "image_sha256": image_sha,
        "ledger_version": version, "linked_source_record_identifiers": source_ids,
        "normalized_category_name": name, "object_identifier": identifier,
        "official_coco_category_id": category_id, "provenance": provenance,
        "schema_version": AUDIT_LEDGER_SCHEMA_VERSION, "source_canvas_box_xyxy": box,
    }
    # fmt: on


def _parse_timestamp(value: str, *, field: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"{field} must be a UTC RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise ValueError(f"{field} must be a UTC RFC 3339 timestamp") from exc
    if (
        parsed.tzinfo != timezone.utc
        or parsed.isoformat().replace("+00:00", "Z") != value
    ):
        raise ValueError(f"{field} must use canonical UTC RFC 3339 serialization")
    return parsed
