#!/usr/bin/env python3
"""Replay a frozen greedy-prefix donor path with exact Hugging Face tokens.

This is deliberately an experiment-local runner.  The raw rollout files are
the source of truth for the prompt and the two frozen trajectories; the HF
session is used only to replay the native row, finish forced partial rows, and
release a greedy suffix.  It does not modify the canonical inference path.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.run_local_branch_causal_value import (
    _annotate_owner_matches,
    _generate_after_forced_partial_row,
    _generate_row,
    _single_native_inputs,
    append_row_if_complete,
    build_positive_entity_ledger,
    extend_covered_set_if_unambiguous,
    hash_prefix_token_ids,
)


SCHEMA_VERSION = "greedy_prefix_forced_owner_path.v1"
BOX_END = 151649
OBJECT_REF_START = 151646
OBJECT_REF_END = 151647
BOX_START = 151648
COORDINATE_START = 151670
COORDINATE_END = 152670
MARKERS = {OBJECT_REF_START, OBJECT_REF_END, BOX_START, BOX_END}
DEFAULT_TOTAL_TOKEN_BUDGET = 512
DEFAULT_MALFORMED_LIMIT = 2


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def _ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a token-id list")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integer token ids")
        result.append(int(item))
    return result


def _path(value: Any, *, base: Path, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError(f"{label} path is required")
    candidate = Path(str(value)).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _validated_artifact_ref(value: Any, *, base: Path, label: str) -> Path:
    """Resolve one declared artifact and verify its digest when provided."""

    path_value = value.get("path") if isinstance(value, Mapping) else value
    path = _path(path_value, base=base, label=label)
    if isinstance(value, Mapping):
        digest = value.get("sha256")
        if not isinstance(digest, str) or not digest:
            raise ValueError(f"{label} must declare sha256")
        observed = hashlib.sha256(path.read_bytes()).hexdigest()
        if observed != digest:
            raise ValueError(f"{label} SHA-256 mismatch: observed {observed}, expected {digest}")
    return path


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _declared_hash(ids: Sequence[int], declaration: Any, *, label: str) -> str:
    expected = hash_prefix_token_ids(ids)
    if not isinstance(declaration, str) or declaration != expected:
        raise ValueError(f"{label} hash mismatch: observed {declaration!r}, expected {expected}")
    return expected


def _unwrap_record(value: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("decode_result", "rollout", "result", "payload"):
        nested = value.get(key)
        if isinstance(nested, Mapping) and any(
            field in nested for field in ("generated_token_ids", "prompt_token_ids")
        ):
            return nested
    return value


def _rollout_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("rollouts", "records", "trajectories", "runs"):
        rows = payload.get(key)
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, Mapping)]
    return [payload]


def _select_rollout(
    payload: Mapping[str, Any],
    *,
    image_id: Any,
    seed: int | None,
    mode: str,
) -> Mapping[str, Any]:
    records = _rollout_records(payload)
    image_text = str(image_id)
    candidates: list[Mapping[str, Any]] = []
    for raw in records:
        row = _unwrap_record(raw)
        row_image = row.get("image_id", row.get("physical_image_id"))
        if row_image is not None and str(row_image) != image_text:
            continue
        row_mode = str(row.get("decode_mode", row.get("mode", ""))).lower()
        if mode == "greedy":
            if row_mode and row_mode not in {"greedy", "native"}:
                continue
        else:
            if row_mode and row_mode not in {"sampled", "sample", "sampling"}:
                continue
            row_seed = row.get("seed", row.get("sampling_seed"))
            if seed is not None and (row_seed is None or int(row_seed) != int(seed)):
                continue
        candidates.append(row)
    if len(candidates) != 1:
        raise ValueError(
            f"{mode} rollout selection for image {image_id!r}, seed {seed!r} yielded {len(candidates)} records"
        )
    row = candidates[0]
    prompt = row.get("prompt") if isinstance(row.get("prompt"), Mapping) else row
    prompt_ids = prompt.get("prompt_token_ids")
    if prompt_ids is None:
        prompt_ids = row.get("input_prompt_token_ids")
    generated = row.get("generated_token_ids")
    if generated is None:
        generated = row.get("raw_generated_token_ids")
    if prompt_ids is None or generated is None:
        raise ValueError("rollout must retain prompt_token_ids and generated_token_ids")
    prompt_ids = _ids(prompt_ids, label="prompt_token_ids")
    generated = _ids(generated, label="generated_token_ids")
    prompt_hash = prompt.get("prompt_token_ids_sha256")
    if prompt_hash is None:
        prompt_hash = row.get("prompt_token_ids_sha256")
    _declared_hash(prompt_ids, prompt_hash, label="prompt_token_ids")
    generated_hash = row.get("generated_token_ids_sha256", row.get("raw_generated_token_ids_sha256"))
    if generated_hash is not None:
        _declared_hash(generated, generated_hash, label="generated_token_ids")
    return {
        **dict(row),
        "prompt_token_ids": prompt_ids,
        "prompt_token_ids_sha256": prompt_hash,
        "generated_token_ids": generated,
        "generated_token_ids_sha256": hash_prefix_token_ids(generated),
    }


def split_generated_rows(token_ids: Sequence[int], *, box_end_token_id: int = BOX_END) -> list[list[int]]:
    """Split only at BOX_END, retaining integer ids and rejecting crossings."""

    values = _ids(token_ids, label="generated_token_ids")
    rows: list[list[int]] = []
    start = 0
    for index, token in enumerate(values):
        if token != int(box_end_token_id):
            continue
        row = values[start : index + 1]
        if not row or row[0] != OBJECT_REF_START:
            raise ValueError("generated tokens contain a cross-row or non-row span")
        if OBJECT_REF_START in row[1:]:
            raise ValueError("generated tokens contain a nested/cross-row donor span")
        _validate_complete_row_tokens(row)
        rows.append(row)
        start = index + 1
    if start != len(values):
        raise ValueError("generated tokens end with an incomplete row")
    return rows


def _validate_complete_row_tokens(tokens: Sequence[int]) -> None:
    values = _ids(tokens, label="row_token_ids")
    if len(values) < 8 or values[0] != OBJECT_REF_START or values[-1] != BOX_END:
        raise ValueError("row is not a complete canonical object row")
    if values.count(OBJECT_REF_START) != 1 or values.count(OBJECT_REF_END) != 1 or values.count(BOX_START) != 1 or values.count(BOX_END) != 1:
        raise ValueError("row contains duplicate or cross-row grammar markers")
    try:
        object_end = values.index(OBJECT_REF_END)
        box_start = values.index(BOX_START, object_end + 1)
    except ValueError as exc:
        raise ValueError("row lacks OBJECT_REF_END followed by BOX_START") from exc
    description = values[1:object_end]
    coordinates = values[box_start + 1 : -1]
    if not description or any(token in MARKERS for token in description):
        raise ValueError("row description span is malformed")
    if len(coordinates) != 4 or any(not COORDINATE_START <= token < COORDINATE_END for token in coordinates):
        raise ValueError("row must contain exactly four canonical coordinate tokens")


def _validate_forced_span_grammar(tokens: Sequence[int], *, role: str = "complete_row") -> dict[str, Any]:
    """Small local grammar check used for endpoint receipts and tests."""

    values = _ids(tokens, label="forced_span")
    error: str | None = None
    if not values or values[0] != OBJECT_REF_START:
        error = "forced span must start at OBJECT_REF_START"
    elif OBJECT_REF_START in values[1:]:
        error = "forced span crosses into another row"
    elif role == "row_opener" and values != [OBJECT_REF_START]:
        error = "row_opener must contain exactly OBJECT_REF_START"
    elif role == "description_end" and (len(values) < 3 or values[-1] != OBJECT_REF_END):
        error = "description_end must end at OBJECT_REF_END"
    elif role == "box_start" and (BOX_START not in values or values[-1] != BOX_START):
        error = "box_start must end at BOX_START"
    elif role == "complete_row":
        try:
            _validate_complete_row_tokens(values)
        except ValueError as exc:
            error = str(exc)
    return {"verified": error is None, "role": role, "error": error, "span_start": 0, "span_end": len(values)}


def semantic_endpoint_role(prefix_token_ids: Sequence[int]) -> str:
    """Name only the token-grammar endpoint; this is not an owner claim."""

    values = _ids(prefix_token_ids, label="prefix_token_ids")
    if not values:
        return "native"
    if values[0] != OBJECT_REF_START:
        return "unknown"
    if len(values) == 1:
        return "row_opener"
    if OBJECT_REF_END not in values:
        return "description"
    object_end = values.index(OBJECT_REF_END)
    if len(values) == object_end + 1:
        return "description_end"
    if BOX_START not in values[object_end + 1 :]:
        return "description"
    box_start = values.index(BOX_START, object_end + 1)
    if len(values) == box_start + 1:
        return "box_start"
    if values[-1] == BOX_END:
        return "box_end"
    coordinate_index = len(values) - box_start - 2
    if coordinate_index in range(4):
        return ("x1", "y1", "x2", "y2")[coordinate_index]
    return "unknown"


def first_divergence_index(native_row_token_ids: Sequence[int], donor_row_token_ids: Sequence[int], *, native_stop: bool = False) -> int | None:
    """Return the first differing token position, or None for identical rows."""

    native = _ids(native_row_token_ids, label="native_row_token_ids")
    donor = _ids(donor_row_token_ids, label="donor_row_token_ids")
    if native_stop or not native:
        return 0
    for index, (left, right) in enumerate(zip(native, donor)):
        if left != right:
            return index
    return len(native) if len(native) != len(donor) else None


def derive_intervention_rungs(native_row_token_ids: Sequence[int], donor_row_token_ids: Sequence[int], *, native_stop: bool = False) -> list[dict[str, Any]]:
    """Build native, empirical donor-prefix, and full-row endpoints."""

    native = _ids(native_row_token_ids, label="native_row_token_ids")
    donor = _ids(donor_row_token_ids, label="donor_row_token_ids")
    if not donor:
        raise ValueError("donor row must be non-empty")
    _validate_complete_row_tokens(donor)
    divergence = first_divergence_index(native, donor, native_stop=native_stop)
    records: list[dict[str, Any]] = [{
        "rung_name": "native",
        "forced_row_prefix_token_ids": [],
        "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids([]),
        "endpoint_token_count": 0,
        "endpoint_role": "native",
        "first_divergence_index": divergence,
    }]
    seen = {tuple()}
    start = 0 if divergence is None else max(0, int(divergence))
    # The full donor row gets its own stable name; prefixes stop one token
    # before it so deduplication never creates two records for one endpoint.
    for end in range(start + 1, len(donor)):
        prefix = donor[:end]
        key = tuple(prefix)
        if key in seen:
            continue
        seen.add(key)
        records.append({
            "rung_name": f"prefix_{end}",
            "forced_row_prefix_token_ids": prefix,
            "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids(prefix),
            "endpoint_token_count": end,
            "endpoint_role": semantic_endpoint_role(prefix),
            "first_divergence_index": divergence,
        })
    key = tuple(donor)
    if key not in seen:
        records.append({
            "rung_name": "full_row",
            "forced_row_prefix_token_ids": donor,
            "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids(donor),
            "endpoint_token_count": len(donor),
            "endpoint_role": semantic_endpoint_role(donor),
            "first_divergence_index": divergence,
        })
    return records


def compare_exact_token_receipts(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Pure comparison used by native replay and the no-op parity gate."""

    left_ids = _ids(left.get("raw_generated_token_ids", []), label="left raw_generated_token_ids")
    right_ids = _ids(right.get("raw_generated_token_ids", []), label="right raw_generated_token_ids")
    return {"passed": left_ids == right_ids, "raw_token_ids_equal": left_ids == right_ids, "left_token_ids": left_ids, "right_token_ids": right_ids}


def compare_noop_parity(native_row: Mapping[str, Any], forced_rows: Sequence[Mapping[str, Any]], native_suffix: Sequence[Mapping[str, Any]] = (), forced_suffixes: Sequence[Sequence[Mapping[str, Any]]] = ()) -> dict[str, Any]:
    """Compare all forced native endpoints against one natural trajectory."""

    checks: list[bool] = []
    row_checks: list[dict[str, Any]] = []
    for row in forced_rows:
        result = compare_exact_token_receipts(native_row, row)
        row_checks.append(result)
        checks.append(bool(result["passed"]))
    suffix_checks: list[dict[str, Any]] = []
    if forced_suffixes:
        native_ids = [_ids(row.get("raw_generated_token_ids", []), label="native suffix") for row in native_suffix]
        for suffix in forced_suffixes:
            observed = [_ids(row.get("raw_generated_token_ids", []), label="forced suffix") for row in suffix]
            equal = observed == native_ids
            suffix_checks.append({"passed": equal, "native": native_ids, "forced": observed})
            checks.append(equal)
    return {"passed": bool(checks) and all(checks), "row_checks": row_checks, "suffix_checks": suffix_checks}


def owner_id_aliases(owner_id: Any, *, image_id: Any | None = None) -> set[str]:
    value = str(owner_id).strip()
    if not value:
        return set()
    aliases = {value}
    if ":" in value:
        aliases.add(value.split(":", 1)[1])
    elif image_id is not None:
        aliases.add(f"{image_id}:{value}")
    return aliases


def target_already_covered(target_owner_id: Any, covered_owner_ids: Sequence[Any], *, image_id: Any | None = None) -> bool:
    target = owner_id_aliases(target_owner_id, image_id=image_id)
    return any(target & owner_id_aliases(owner, image_id=image_id) for owner in covered_owner_ids)


def require_target_not_covered(target_owner_id: Any, covered_owner_ids: Sequence[Any], *, image_id: Any | None = None) -> None:
    if target_already_covered(target_owner_id, covered_owner_ids, image_id=image_id):
        raise ValueError("target owner is already covered by greedy parent rows")


def fixed_budget_plan(greedy_parent_row_count: int, complete_row_budget: int) -> dict[str, int]:
    """Return the fixed complete-row horizon shared by every intervention arm."""

    parent = int(greedy_parent_row_count)
    budget = int(complete_row_budget)
    if parent < 0 or budget <= parent:
        raise ValueError("complete-row budget must be greater than greedy parent row count")
    return {"greedy_parent_row_count": parent, "complete_row_budget": budget, "remaining_suffix_horizon": budget - parent - 1}


def _record_predictions(record: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    for key in ("predictions", "parse_evidence", "parse_result", "parser_evidence"):
        value = record.get(key)
        if isinstance(value, Mapping):
            nested = value.get("predictions")
            if isinstance(nested, list):
                return [item for item in nested if isinstance(item, Mapping)]
        elif isinstance(value, list):
            return [item for item in value if isinstance(item, Mapping)]
    return []


def _frozen_rows(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = split_generated_rows(record.get("generated_token_ids", []))
    predictions = _record_predictions(record)
    output: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        output.append({
            "row_index": index,
            "status": "success",
            "accepted_complete_row": True,
            "raw_generated_token_ids": row,
            "raw_generated_token_ids_sha256": hash_prefix_token_ids(row),
            "row_stop": {"stop_reason": "complete_row"},
            "parse_evidence": {"parse_status": "accepted"},
            "parsed_predictions": [predictions[index]] if index < len(predictions) else [],
        })
    return output


def validate_expected_donor(record: Mapping[str, Any], donor_row_index: int, expected: Mapping[str, Any]) -> dict[str, Any]:
    rows = _frozen_rows(record)
    index = int(donor_row_index)
    if index < 0 or index >= len(rows):
        raise ValueError(f"sampled donor row index {index} is outside complete rows ({len(rows)})")
    predictions = _record_predictions(record)
    if index >= len(predictions):
        raise ValueError("sampled donor parser evidence lacks the selected row")
    observed = predictions[index]
    for field in ("description", "coord_bins", "raw_span_sha256"):
        if field in expected and observed.get(field) != expected[field]:
            raise ValueError(f"sampled donor expected_{field} mismatch at row {index}")
    return {"row_index": index, "token_ids": rows[index]["raw_generated_token_ids"], "prediction": dict(observed)}


def _annotate_with_qualified_ids(row: dict[str, Any], *, ledger: Sequence[Mapping[str, Any]], image_id: Any, width: int, height: int, covered: Sequence[str]) -> dict[str, Any]:
    _annotate_owner_matches(row, entity_ledger=ledger, image_width=width, image_height=height, covered_entity_ids=covered)
    row["strict_matched_owner_ids_qualified"] = sorted({f"{image_id}:{owner}" for owner in row.get("strict_matched_owner_ids", [])})
    row["covered_prefix_owner_ids_qualified"] = sorted({f"{image_id}:{owner}" for owner in row.get("covered_prefix_owner_ids", [])})
    row["uncovered_ledger_owner_ids_qualified"] = sorted({f"{image_id}:{owner}" for owner in row.get("uncovered_ledger_owner_ids", [])})
    return row


def _attach_reviewed_owner(row: dict[str, Any], *, owner_id: str, image_id: Any) -> dict[str, Any]:
    """Attach a frozen human owner verdict without relabelling it as an IoU match."""

    aliases = owner_id_aliases(owner_id, image_id=image_id)
    qualified = next((value for value in aliases if ":" in value), f"{image_id}:{owner_id}")
    local = qualified.split(":", 1)[1]
    row["reviewed_owner_ids"] = [local]
    row["reviewed_owner_ids_qualified"] = [qualified]
    row["reviewed_owner_evidence"] = "frozen_human_crop_review"
    return row


def _annotate_exact_donor(row: dict[str, Any], *, donor_prediction: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]], image_id: Any, width: int, height: int, covered: Sequence[str]) -> dict[str, Any]:
    row["parsed_predictions"] = [dict(donor_prediction)]
    return _annotate_with_qualified_ids(row, ledger=ledger, image_id=image_id, width=width, height=height, covered=covered)


def _extract_case_values(case: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    image_id = case.get("image_id")
    if image_id is None:
        raise ValueError("case requires image_id")
    sampled_ref = case.get("sampled_artifact", case.get("sampled_rollout", case.get("sampled_rollout_json")))
    greedy_ref = case.get("greedy_artifact", case.get("greedy_rollout", case.get("greedy_rollout_json", manifest.get("greedy_artifact"))))
    if isinstance(sampled_ref, Mapping):
        sampled_path_ref = sampled_ref.get("path")
    else:
        sampled_path_ref = sampled_ref
    if isinstance(greedy_ref, Mapping):
        greedy_path_ref = greedy_ref.get("path")
    else:
        greedy_path_ref = greedy_ref
    if not sampled_path_ref or not greedy_path_ref:
        raise ValueError("case requires greedy and sampled rollout artifact paths")
    parent_count = int(case.get("greedy_parent_row_count", case.get("parent_row_count", -1)))
    donor_index = int(case.get("sampled_donor_row_index", case.get("donor_row_index", -1)))
    budget = int(case.get("complete_row_budget", case.get("fixed_complete_row_budget", case.get("budget", -1))))
    fixed_budget_plan(parent_count, budget)
    if donor_index < 0:
        raise ValueError("sampled donor row index must be non-negative")
    return {
        "image_id": image_id,
        "sampled_seed": int(case.get("sampled_seed")),
        "sampled_path_ref": sampled_ref,
        "greedy_path_ref": greedy_ref,
        "parent_count": parent_count,
        "donor_index": donor_index,
        "budget": budget,
        "target_owner_id": str(case.get("target_owner_id", "")).strip(),
        "geometry_trust": case.get("geometry_trust"),
        "expected_donor": case.get("expected_donor", {}),
        "reviewed_donor_owner_id": str(case.get("reviewed_donor_owner_id", "")).strip() or None,
    }


def _summary(rows: Sequence[Mapping[str, Any]], *, target_owner_id: str, image_id: Any, covered_parent: Sequence[str]) -> dict[str, Any]:
    owners: set[str] = set()
    owner_occurrences: list[str] = []
    unresolved = False
    automatic_unresolved = False
    malformed = False
    target_aliases = owner_id_aliases(target_owner_id, image_id=image_id)
    target_in: list[str] = []
    for row in rows:
        matched = {str(value) for value in row.get("strict_matched_owner_ids", [])}
        qualified = {f"{image_id}:{value}" for value in matched}
        qualified.update(
            str(value) for value in row.get("reviewed_owner_ids_qualified", [])
        )
        owners.update(qualified)
        owner_occurrences.extend(sorted(qualified))
        if target_aliases & (matched | qualified):
            target_in.append(str(row.get("row_index", len(target_in))))
        stop_reason = str((row.get("row_stop") or {}).get("stop_reason", "unknown"))
        if row.get("status") == "failed" or stop_reason not in {"complete_row", "terminal"}:
            malformed = True
        if row.get("unmatched_or_ambiguous_prediction_indices"):
            automatic_unresolved = True
            if not row.get("reviewed_owner_ids_qualified"):
                unresolved = True
    covered = {str(value) for value in covered_parent}
    qualified_covered = {value if ":" in value else f"{image_id}:{value}" for value in covered}
    repeated = {owner for owner in owner_occurrences if owner_occurrences.count(owner) > 1}
    return {
        "target_owner_acquired": bool(target_in),
        "target_owner_row_indices": target_in,
        "unique_owner_set": sorted(owners),
        "duplicate_owner_set": sorted((owners & qualified_covered) | repeated),
        "unresolved_or_malformed": bool(unresolved or malformed),
        "unresolved": unresolved,
        "automatic_unresolved_present": automatic_unresolved,
        "malformed": malformed,
    }


def _greedy_suffix(*, session: Any, native_inputs: Mapping[str, Any], prefix: Sequence[int], tokenizer: Any, width: int, height: int, horizon_rows: int, remaining_tokens: int, ledger: Sequence[Mapping[str, Any]], image_id: Any, covered: Sequence[str], repetition_penalty: float, malformed_limit: int, start_row_index: int = 0) -> tuple[list[dict[str, Any]], int, list[int]]:
    rows: list[dict[str, Any]] = []
    current = list(map(int, prefix))
    used = 0
    working_covered = set(str(value) for value in covered)
    for offset in range(int(horizon_rows)):
        left = int(remaining_tokens) - used
        if left <= 0:
            break
        row_index = int(start_row_index) + offset
        row = _generate_row(session=session, native_inputs=native_inputs, prefix_token_ids=current, tokenizer=tokenizer, image_width=width, image_height=height, mode="greedy", seed=None, temperature=0.0, top_p=1.0, repetition_penalty=repetition_penalty, max_new_tokens=left, malformed_limit=malformed_limit, row_index=row_index)
        row["row_index"] = row_index
        row["input_prefix_token_ids"] = list(current)
        _annotate_with_qualified_ids(row, ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(working_covered))
        current, append_receipt = append_row_if_complete(current, row)
        row["append_receipt"] = append_receipt
        row["accepted_complete_row"] = bool(append_receipt.get("appended"))
        used += len(row.get("raw_generated_token_ids", []))
        rows.append(row)
        if row.get("accepted_complete_row"):
            updated, coverage_receipt = extend_covered_set_if_unambiguous(working_covered, row)
            row["coverage_receipt"] = coverage_receipt
            if coverage_receipt.get("coverage_updated"):
                working_covered.update(updated)
        if not row["accepted_complete_row"]:
            break
    return rows, used, current


def _covered_after_row(covered: Sequence[str], row: Mapping[str, Any]) -> tuple[list[str], dict[str, Any]]:
    """Keep the executable coverage ledger in the ledger's local owner-id space."""

    reviewed = [str(value) for value in row.get("reviewed_owner_ids", [])]
    if row.get("accepted_complete_row") and len(reviewed) == 1:
        updated = sorted(set(map(str, covered)) | {reviewed[0]})
        return updated, {
            "coverage_updated": True,
            "owner_id": reviewed[0],
            "source": "frozen_human_crop_review",
            "refusal_reason": None,
        }
    updated, receipt = extend_covered_set_if_unambiguous(covered, row)
    return [str(value) for value in updated], dict(receipt)


def _control_donor_record(
    *,
    case: Mapping[str, Any],
    greedy_payload: Mapping[str, Any],
    sampled_payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], dict[str, Any]] | None:
    control = case.get("control_donor")
    if not isinstance(control, Mapping):
        return None
    source = str(control.get("source", "greedy_artifact"))
    if source == "greedy_artifact":
        record = _select_rollout(greedy_payload, image_id=case.get("image_id"), seed=None, mode="greedy")
    elif source == "sampled_artifact":
        seed = control.get("seed")
        record = _select_rollout(sampled_payload, image_id=case.get("image_id"), seed=None if seed is None else int(seed), mode="sampled")
    else:
        raise ValueError(f"unsupported control_donor source {source!r}")
    index = int(control.get("row_index", control.get("donor_row_index", -1)))
    expected = {
        key: control[key]
        for key in ("description", "coord_bins", "raw_span_sha256")
        if key in control
    }
    info = validate_expected_donor(record, index, expected)
    if control.get("owner_id") is not None:
        info["owner_id"] = str(control["owner_id"])
    info["source"] = source
    info["seed"] = control.get("seed")
    return record, info


def _run_control_ladder(
    *,
    control_record: Mapping[str, Any],
    control_info: Mapping[str, Any],
    native_ids: Sequence[int],
    native_stop: bool,
    native_replay: Mapping[str, Any],
    target_rungs: Sequence[Mapping[str, Any]],
    parent_ids: Sequence[int],
    parent_count: int,
    complete_row_budget: int,
    total_token_budget: int,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    ledger: Sequence[Mapping[str, Any]],
    image_id: Any,
    width: int,
    height: int,
    covered: Sequence[str],
    target_owner_id: str,
    malformed_limit: int,
) -> list[dict[str, Any]]:
    """Run a same-depth non-target donor ladder with the identical parent/cap."""

    control_rows = _frozen_rows(control_record)
    donor_index = int(control_info["row_index"])
    donor_row = control_rows[donor_index]
    control_ids = donor_row["raw_generated_token_ids"]
    if native_stop:
        native_row: Mapping[str, Any] = {"raw_generated_token_ids": []}
    else:
        native_row = {"raw_generated_token_ids": list(native_ids)}
    # A matched control must be evaluated at every target intervention depth.
    # Deriving a second staircase from the control's own divergence can omit a
    # target-positive depth and silently turn missing evidence into absence.
    rungs: list[dict[str, Any]] = []
    for target_rung in target_rungs:
        count = int(target_rung.get("endpoint_token_count", -1))
        if count < 0 or count > len(control_ids):
            raise ValueError("control donor cannot supply every target staircase depth")
        if str(target_rung.get("rung_name")) == "full_row" and count != len(control_ids):
            raise ValueError("target and control complete rows must have equal token length")
        control_prefix = control_ids[:count]
        rungs.append({
            "rung_name": str(target_rung.get("rung_name")),
            "forced_row_prefix_token_ids": control_prefix,
            "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids(control_prefix),
            "endpoint_token_count": count,
            "endpoint_role": semantic_endpoint_role(control_prefix) if control_prefix else "native",
            "target_endpoint_role": target_rung.get("endpoint_role"),
            "target_first_divergence_index": target_rung.get("first_divergence_index"),
        })
    initial_remaining = int(total_token_budget) - len(parent_ids)
    arms: list[dict[str, Any]] = []
    for rung in rungs:
        forced = list(rung["forced_row_prefix_token_ids"])
        forced_count = len(forced)
        if rung["rung_name"] == "native":
            # The native arm is shared with the target ladder.  Reuse the
            # already verified replay instead of fabricating a terminal row or
            # paying for another nominally identical generation.
            intervened = dict(native_replay)
            intervened["row_index"] = parent_count
            if not native_stop:
                _annotate_with_qualified_ids(intervened, ledger=ledger, image_id=image_id, width=width, height=height, covered=covered)
            released = len(intervened.get("raw_generated_token_ids", []))
            current_prefix, append_receipt = append_row_if_complete(parent_ids, intervened)
            intervened["append_receipt"] = append_receipt
            intervened["accepted_complete_row"] = bool(append_receipt.get("appended"))
        elif rung["rung_name"] == "full_row":
            if forced_count > initial_remaining:
                raise ValueError("control forced tokens exceed total_token_budget")
            intervened = dict(donor_row)
            _annotate_exact_donor(intervened, donor_prediction=control_info["prediction"], ledger=ledger, image_id=image_id, width=width, height=height, covered=covered)
            intervened["accepted_complete_row"] = True
            current_prefix = list(parent_ids) + forced
            append_receipt = {"appended": True, "appended_token_count": forced_count}
            released = 0
        else:
            available = initial_remaining - forced_count
            if available < 0:
                raise ValueError("control forced prefix exceeds total_token_budget")
            intervened = _generate_after_forced_partial_row(session=session, native_inputs=native_inputs, parent_prefix_token_ids=parent_ids, forced_row_prefix_token_ids=forced, tokenizer=tokenizer, image_width=width, image_height=height, repetition_penalty=1.0, max_new_tokens=available, malformed_limit=malformed_limit, row_index=parent_count)
            _annotate_with_qualified_ids(intervened, ledger=ledger, image_id=image_id, width=width, height=height, covered=covered)
            current_prefix, append_receipt = append_row_if_complete(parent_ids, intervened)
            intervened["append_receipt"] = append_receipt
            intervened["accepted_complete_row"] = bool(append_receipt.get("appended"))
            released = len(intervened.get("released_tail_token_ids", []))
        suffix: list[dict[str, Any]] = []
        suffix_used = 0
        covered_after_intervened = list(map(str, covered))
        coverage_receipt: dict[str, Any] = {"coverage_updated": False, "refusal_reason": "row_not_complete"}
        if intervened.get("accepted_complete_row"):
            covered_after_intervened, coverage_receipt = _covered_after_row(covered, intervened)
            suffix, suffix_used, current_prefix = _greedy_suffix(session=session, native_inputs=native_inputs, prefix=current_prefix, tokenizer=tokenizer, width=width, height=height, horizon_rows=complete_row_budget - parent_count - 1, remaining_tokens=initial_remaining - forced_count - released, ledger=ledger, image_id=image_id, covered=covered_after_intervened, repetition_penalty=1.0, malformed_limit=malformed_limit, start_row_index=parent_count + 1)
        intervened_summary = _summary(
            [intervened],
            target_owner_id=target_owner_id,
            image_id=image_id,
            covered_parent=covered,
        )
        suffix_summary = _summary(
            suffix,
            target_owner_id=target_owner_id,
            image_id=image_id,
            covered_parent=covered_after_intervened,
        )
        # Keep direct target flags alongside the nested summaries so a control
        # consumer cannot confuse acquisition in the current row with later
        # acquisition in the suffix.
        target_hits_in_intervened = bool(intervened_summary["target_owner_acquired"])
        target_hits_in_suffix = bool(suffix_summary["target_owner_acquired"])
        complete_rows = parent_count + int(bool(intervened.get("accepted_complete_row"))) + sum(
            bool(row.get("accepted_complete_row")) for row in suffix
        )
        total_used = forced_count + released + suffix_used
        arms.append({
            "rung": rung,
            "forced_token_count": forced_count,
            "intervened_row": intervened,
            "suffix_rows": suffix,
            "control_owner_id": control_info.get("owner_id"),
            "target_owner_acquired_by_decoder": bool(target_hits_in_intervened and rung["rung_name"] != "full_row" and released > 0),
            "target_owner_acquired_in_suffix": target_hits_in_suffix,
            "token_receipt": {"parent_token_count": len(parent_ids), "forced_token_count": forced_count, "released_intervened_token_count": released, "released_suffix_token_count": suffix_used, "total_post_prompt_token_count": len(parent_ids) + total_used, "total_token_budget": total_token_budget, "remaining_token_budget": total_token_budget - len(parent_ids) - total_used},
            "budget_comparability": {
                "row_budget_comparable": complete_rows <= complete_row_budget,
                "token_budget_comparable": len(parent_ids) + total_used <= total_token_budget,
                "complete_row_budget": complete_row_budget,
                "complete_rows_consumed": complete_rows,
            },
            "summary": {
                "intervened": intervened_summary,
                "suffix": suffix_summary,
                "target_owner_acquired": bool(target_hits_in_intervened or target_hits_in_suffix),
                "coverage_after_intervened": covered_after_intervened,
                "coverage_update_receipt": coverage_receipt,
            },
        })
    return arms


def _run_case(*, case: Mapping[str, Any], manifest: Mapping[str, Any], greedy_payload: Mapping[str, Any], sampled_payload: Mapping[str, Any], greedy_record: Mapping[str, Any], sampled_record: Mapping[str, Any], session: Any, native_inputs: Mapping[str, Any], tokenizer: Any, ledger: Sequence[Mapping[str, Any]], width: int, height: int, image_id: Any, malformed_limit: int, noop_gate: bool) -> dict[str, Any]:
    values = _extract_case_values(case, manifest)
    parent_count = values["parent_count"]
    budget = values["budget"]
    greedy_rows = _frozen_rows(greedy_record)
    sampled_rows = _frozen_rows(sampled_record)
    if len(greedy_rows) < parent_count:
        raise ValueError("greedy artifact has fewer complete rows than greedy_parent_row_count")
    donor_info = validate_expected_donor(sampled_record, values["donor_index"], values["expected_donor"])
    donor_row = sampled_rows[values["donor_index"]]
    parent_rows = [dict(row) for row in greedy_rows[:parent_count]]
    parent_ids = [token for row in parent_rows for token in row["raw_generated_token_ids"]]
    # Keep this set in the ledger's local entity-id namespace.  Qualified IDs
    # are presentation fields only; mixing the two namespaces makes every
    # previous object look uncovered to the matcher.
    covered: set[str] = set()
    for row in parent_rows:
        _annotate_with_qualified_ids(row, ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(covered))
        covered.update(str(value) for value in row.get("strict_matched_owner_ids", []))
    parent_owner_occurrences = [
        f"{image_id}:{owner}"
        for row in parent_rows
        for owner in row.get("strict_matched_owner_ids", [])
    ]
    parent_duplicate_owner_set = sorted({owner for owner in parent_owner_occurrences if parent_owner_occurrences.count(owner) > 1})
    parent_owner_set = sorted(set(parent_owner_occurrences))
    require_target_not_covered(values["target_owner_id"], sorted(covered), image_id=image_id)
    donor_check = dict(donor_row)
    _annotate_exact_donor(
        donor_check,
        donor_prediction=donor_info["prediction"],
        ledger=ledger,
        image_id=image_id,
        width=width,
        height=height,
        covered=sorted(covered),
    )
    if values["reviewed_donor_owner_id"]:
        _attach_reviewed_owner(
            donor_check,
            owner_id=values["reviewed_donor_owner_id"],
            image_id=image_id,
        )
    target_aliases = owner_id_aliases(values["target_owner_id"], image_id=image_id)
    donor_owners = {
        str(value) for value in donor_check.get("strict_matched_owner_ids_qualified", [])
    }
    donor_owners.update(
        str(value) for value in donor_check.get("reviewed_owner_ids_qualified", [])
    )
    if not any(target_aliases & owner_id_aliases(value, image_id=image_id) for value in donor_owners):
        raise ValueError("sampled donor does not resolve to the declared target owner")
    native_stop = len(greedy_rows) == parent_count
    native_row = greedy_rows[parent_count] if not native_stop else {"raw_generated_token_ids": [], "status": "terminal", "row_stop": {"stop_reason": "terminal"}, "parsed_predictions": []}
    native_ids = list(native_row.get("raw_generated_token_ids", []))
    rungs = derive_intervention_rungs(native_ids, donor_row["raw_generated_token_ids"], native_stop=native_stop)
    initial_remaining = int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET)) - len(parent_ids)
    if initial_remaining < 0:
        raise ValueError("parent token ids already exceed total_token_budget")
    natural_native: dict[str, Any] | None = None
    parity: dict[str, Any] | None = None
    natural_native = _generate_row(session=session, native_inputs=native_inputs, prefix_token_ids=parent_ids, tokenizer=tokenizer, image_width=width, image_height=height, mode="greedy", seed=None, temperature=0.0, top_p=1.0, repetition_penalty=1.0, max_new_tokens=initial_remaining, malformed_limit=malformed_limit, row_index=parent_count)
    if native_stop:
        terminal_ids = [int(session._im_end_token_id())]
        if list(map(int, natural_native.get("raw_generated_token_ids", []))) != terminal_ids or str((natural_native.get("row_stop") or {}).get("stop_reason")) != "terminal":
            raise ValueError("frozen native STOP was not reproduced by natural HF replay")
    else:
        if not compare_exact_token_receipts(natural_native, native_row)["passed"]:
            raise ValueError("full native row and natural native row differ; refusing causal interpretation")
    arms: list[dict[str, Any]] = []
    noop_forced_rows: list[dict[str, Any]] = []
    noop_suffixes: list[list[dict[str, Any]]] = []
    for rung in rungs:
        forced = list(rung["forced_row_prefix_token_ids"])
        forced_count = len(forced)
        if rung["rung_name"] == "native":
            if native_stop:
                intervened = {**natural_native, "row_index": parent_count, "accepted_complete_row": False}
                released = len(intervened.get("raw_generated_token_ids", []))
                current_prefix = list(parent_ids)
            else:
                intervened = dict(natural_native or {})
                intervened["row_index"] = parent_count
                _annotate_with_qualified_ids(intervened, ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(covered))
                released = len(intervened.get("raw_generated_token_ids", []))
                current_prefix, append_receipt = append_row_if_complete(parent_ids, intervened)
                intervened["append_receipt"] = append_receipt
                intervened["accepted_complete_row"] = bool(append_receipt.get("appended"))
            forced_ids = []
        elif rung["rung_name"] == "full_row":
            if forced_count > initial_remaining:
                raise ValueError("donor forced tokens exceed total_token_budget")
            intervened = dict(donor_row)
            _annotate_exact_donor(intervened, donor_prediction=donor_info["prediction"], ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(covered))
            intervened["row_index"] = parent_count
            intervened["forced_row_prefix_token_ids"] = forced
            intervened["accepted_complete_row"] = True
            current_prefix = parent_ids + forced
            released = 0
            forced_ids = forced
        else:
            available = initial_remaining - forced_count
            if available < 0:
                raise ValueError("forced donor prefix exceeds total_token_budget")
            intervened = _generate_after_forced_partial_row(session=session, native_inputs=native_inputs, parent_prefix_token_ids=parent_ids, forced_row_prefix_token_ids=forced, tokenizer=tokenizer, image_width=width, image_height=height, repetition_penalty=1.0, max_new_tokens=available, malformed_limit=malformed_limit, row_index=parent_count)
            intervened["row_index"] = parent_count
            _annotate_with_qualified_ids(intervened, ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(covered))
            current_prefix, append_receipt = append_row_if_complete(parent_ids, intervened)
            intervened["append_receipt"] = append_receipt
            intervened["accepted_complete_row"] = bool(append_receipt.get("appended"))
            released = len(intervened.get("released_tail_token_ids", []))
            forced_ids = forced
        if (
            values["reviewed_donor_owner_id"]
            and list(map(int, intervened.get("raw_generated_token_ids", [])))
            == list(map(int, donor_row["raw_generated_token_ids"]))
        ):
            _attach_reviewed_owner(
                intervened,
                owner_id=values["reviewed_donor_owner_id"],
                image_id=image_id,
            )
        suffix: list[dict[str, Any]] = []
        suffix_used = 0
        covered_after_intervened = sorted(covered)
        coverage_receipt: dict[str, Any] = {"coverage_updated": False, "refusal_reason": "row_not_complete"}
        if intervened.get("accepted_complete_row"):
            covered_after_intervened, coverage_receipt = _covered_after_row(sorted(covered), intervened)
            remaining = initial_remaining - forced_count - released
            suffix, suffix_used, current_prefix = _greedy_suffix(session=session, native_inputs=native_inputs, prefix=current_prefix, tokenizer=tokenizer, width=width, height=height, horizon_rows=budget - parent_count - 1, remaining_tokens=remaining, ledger=ledger, image_id=image_id, covered=covered_after_intervened, repetition_penalty=1.0, malformed_limit=malformed_limit, start_row_index=parent_count + 1)
            if values["reviewed_donor_owner_id"]:
                for suffix_row in suffix:
                    if list(map(int, suffix_row.get("raw_generated_token_ids", []))) == list(
                        map(int, donor_row["raw_generated_token_ids"])
                    ):
                        _attach_reviewed_owner(
                            suffix_row,
                            owner_id=values["reviewed_donor_owner_id"],
                            image_id=image_id,
                        )
        all_rows = [intervened] + suffix
        total_used = forced_count + released + suffix_used
        complete_rows = parent_count + sum(bool(row.get("accepted_complete_row")) for row in all_rows)
        intervened_summary = _summary([intervened], target_owner_id=values["target_owner_id"], image_id=image_id, covered_parent=sorted(covered))
        suffix_summary = _summary(suffix, target_owner_id=values["target_owner_id"], image_id=image_id, covered_parent=covered_after_intervened)
        decoder_acquired_target = bool(
            rung["rung_name"] != "full_row"
            and released > 0
            and intervened_summary["target_owner_acquired"]
        )
        coverage_including = sorted(
            set(parent_owner_set)
            | set(intervened_summary["unique_owner_set"])
            | set(suffix_summary["unique_owner_set"])
        )
        coverage_excluding_intervened = sorted(
            set(parent_owner_set) | set(suffix_summary["unique_owner_set"])
        )
        arm = {
            "rung": rung,
            "forced_token_ids": forced_ids,
            "forced_token_count": forced_count,
            "intervened_row": intervened,
            "suffix_rows": suffix,
            "summary": {
                **intervened_summary,
                "suffix": suffix_summary,
                # The exact donor row is an intervention input, not decoder
                # evidence.  Keep its factual owner match, but exclude it
                # from acquisition and promotion fields.
                "target_owner_acquired_by_decoder": decoder_acquired_target,
                "promotion_eligible_target_acquisition": decoder_acquired_target,
                "fixed_budget_owner_set_including_intervened_row": coverage_including,
                "fixed_budget_owner_set_excluding_intervened_row": coverage_excluding_intervened,
                "parent_duplicate_owner_set": parent_duplicate_owner_set,
                "coverage_after_intervened": covered_after_intervened,
                "coverage_update_receipt": coverage_receipt,
            },
            "token_receipt": {"parent_token_count": len(parent_ids), "forced_token_count": forced_count, "released_intervened_token_count": released, "released_suffix_token_count": suffix_used, "total_post_prompt_token_count": len(parent_ids) + total_used, "total_token_budget": int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET)), "remaining_token_budget": int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET)) - len(parent_ids) - total_used},
            "budget_comparability": {"row_budget_comparable": complete_rows <= budget, "token_budget_comparable": len(parent_ids) + total_used <= int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET)), "complete_row_budget": budget, "complete_rows_consumed": complete_rows},
        }
        arms.append(arm)

    native_arm_rows = [arms[0]["intervened_row"], *arms[0]["suffix_rows"]]
    actual_native_complete = [
        row for row in native_arm_rows if row.get("accepted_complete_row")
    ]
    expected_native_complete = greedy_rows[parent_count:budget]
    frozen_native_horizon_parity = {
        "passed": [row.get("raw_generated_token_ids", []) for row in actual_native_complete]
        == [row.get("raw_generated_token_ids", []) for row in expected_native_complete],
        "expected_complete_row_count": len(expected_native_complete),
        "observed_complete_row_count": len(actual_native_complete),
        "expected_terminal_within_budget": len(greedy_rows) < budget,
        "observed_terminal_within_budget": any(
            str((row.get("row_stop") or {}).get("stop_reason")) == "terminal"
            for row in native_arm_rows
        ),
    }
    if frozen_native_horizon_parity["expected_terminal_within_budget"]:
        frozen_native_horizon_parity["passed"] = bool(
            frozen_native_horizon_parity["passed"]
            and frozen_native_horizon_parity["observed_terminal_within_budget"]
        )
    if not frozen_native_horizon_parity["passed"]:
        raise ValueError("natural native horizon does not reproduce the frozen greedy artifact")

    native_coverage = set(
        arms[0]["summary"]["fixed_budget_owner_set_including_intervened_row"]
    )
    native_duplicate_owners = set(arms[0]["summary"]["duplicate_owner_set"]) | set(
        arms[0]["summary"]["suffix"]["duplicate_owner_set"]
    )
    native_non_target = {
        owner
        for owner in native_coverage
        if not owner_id_aliases(owner, image_id=image_id) & target_aliases
    }
    for arm in arms:
        current = set(arm["summary"]["fixed_budget_owner_set_including_intervened_row"])
        current_duplicate_owners = set(arm["summary"]["duplicate_owner_set"]) | set(
            arm["summary"]["suffix"]["duplicate_owner_set"]
        )
        retained = native_non_target & current
        arm["summary"]["fixed_budget_delta_vs_native"] = {
            "unique_owner_count_delta": len(current) - len(native_coverage),
            "added_owner_ids": sorted(current - native_coverage),
            "removed_owner_ids": sorted(native_coverage - current),
            "native_non_target_owner_count": len(native_non_target),
            "retained_native_non_target_owner_count": len(retained),
            "retained_native_non_target_owner_fraction": (
                len(retained) / len(native_non_target) if native_non_target else 1.0
            ),
            "new_duplicate_owner_ids": sorted(current_duplicate_owners - native_duplicate_owners),
            "removed_native_duplicate_owner_ids": sorted(native_duplicate_owners - current_duplicate_owners),
        }
    if noop_gate and not native_stop:
        native_forced_rows: list[dict[str, Any]] = []
        native_suffixes: list[list[dict[str, Any]]] = []
        for end in range(1, len(native_ids) + 1):
            prefix = native_ids[:end]
            if end == len(native_ids):
                forced_row = {**native_row, "raw_generated_token_ids": native_ids}
                suffix_prefix = parent_ids + native_ids
                suffix, _, _ = _greedy_suffix(session=session, native_inputs=native_inputs, prefix=suffix_prefix, tokenizer=tokenizer, width=width, height=height, horizon_rows=budget - parent_count - 1, remaining_tokens=initial_remaining - len(native_ids), ledger=ledger, image_id=image_id, covered=sorted(covered), repetition_penalty=1.0, malformed_limit=malformed_limit, start_row_index=parent_count + 1)
            else:
                forced_row = _generate_after_forced_partial_row(session=session, native_inputs=native_inputs, parent_prefix_token_ids=parent_ids, forced_row_prefix_token_ids=prefix, tokenizer=tokenizer, image_width=width, image_height=height, repetition_penalty=1.0, max_new_tokens=initial_remaining - len(prefix), malformed_limit=malformed_limit, row_index=parent_count)
                suffix_prefix, _ = append_row_if_complete(parent_ids, forced_row)
                suffix, _, _ = _greedy_suffix(session=session, native_inputs=native_inputs, prefix=suffix_prefix, tokenizer=tokenizer, width=width, height=height, horizon_rows=budget - parent_count - 1, remaining_tokens=initial_remaining - len(prefix) - len(forced_row.get("released_tail_token_ids", [])), ledger=ledger, image_id=image_id, covered=sorted(covered), repetition_penalty=1.0, malformed_limit=malformed_limit, start_row_index=parent_count + 1)
            native_forced_rows.append(forced_row)
            native_suffixes.append(suffix)
        parity = compare_noop_parity(natural_native or native_row, native_forced_rows, native_suffix=(_greedy_suffix(session=session, native_inputs=native_inputs, prefix=parent_ids + native_ids, tokenizer=tokenizer, width=width, height=height, horizon_rows=budget - parent_count - 1, remaining_tokens=initial_remaining - len(native_ids), ledger=ledger, image_id=image_id, covered=sorted(covered), repetition_penalty=1.0, malformed_limit=malformed_limit, start_row_index=parent_count + 1)[0]), forced_suffixes=native_suffixes)
        if not parity["passed"]:
            raise ValueError("native prefix no-op parity gate failed; refusing causal interpretation")
    control_result: dict[str, Any] | None = None
    control_selected = _control_donor_record(case=case, greedy_payload=greedy_payload, sampled_payload=sampled_payload)
    if control_selected is None:
        raise ValueError("every case requires a non-target control donor")
    if control_selected is not None:
        control_record, control_info = control_selected
        control_check = dict(_frozen_rows(control_record)[int(control_info["row_index"])])
        _annotate_exact_donor(control_check, donor_prediction=control_info["prediction"], ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(covered))
        expected_control_owner = owner_id_aliases(control_info.get("owner_id", ""), image_id=image_id)
        observed_control_owner = {
            str(value)
            for value in control_check.get("strict_matched_owner_ids_qualified", [])
        }
        if expected_control_owner and not any(expected_control_owner & owner_id_aliases(value, image_id=image_id) for value in observed_control_owner):
            raise ValueError("control_donor owner_id does not match its selected positive row")
        if expected_control_owner & owner_id_aliases(values["target_owner_id"], image_id=image_id):
            raise ValueError("control_donor owner_id must differ from target owner")
        control_arms = _run_control_ladder(control_record=control_record, control_info=control_info, native_ids=native_ids, native_stop=native_stop, native_replay=natural_native, target_rungs=rungs, parent_ids=parent_ids, parent_count=parent_count, complete_row_budget=budget, total_token_budget=int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET)), session=session, native_inputs=native_inputs, tokenizer=tokenizer, ledger=ledger, image_id=image_id, width=width, height=height, covered=sorted(covered), target_owner_id=values["target_owner_id"], malformed_limit=malformed_limit)
        target_by_depth = {
            int(arm["rung"]["endpoint_token_count"]): bool(arm["summary"].get("target_owner_acquired_by_decoder"))
            for arm in arms
        }
        control_current_by_depth = {
            int(arm["rung"]["endpoint_token_count"]): bool(arm.get("target_owner_acquired_by_decoder"))
            for arm in control_arms
        }
        control_suffix_by_depth = {
            int(arm["rung"]["endpoint_token_count"]): bool(arm.get("target_owner_acquired_in_suffix"))
            for arm in control_arms
        }
        if set(target_by_depth) != set(control_current_by_depth):
            raise ValueError("control ladder does not cover every target staircase depth")
        comparisons: list[dict[str, Any]] = []
        for depth in sorted(target_by_depth):
            control_current = control_current_by_depth[depth]
            control_suffix = control_suffix_by_depth[depth]
            control_any = bool(control_current or control_suffix)
            target_arm = next(
                arm for arm in arms
                if int(arm["rung"]["endpoint_token_count"]) == depth
            )
            target_arm["summary"]["control_target_absent_at_same_depth"] = not control_any
            delta = target_arm["summary"]["fixed_budget_delta_vs_native"]
            target_arm["summary"]["promotion_eligible_target_acquisition"] = bool(
                target_arm["summary"]["target_owner_acquired_by_decoder"]
                and not control_any
                and int(delta["unique_owner_count_delta"]) >= 0
                and not delta["new_duplicate_owner_ids"]
            )
            comparisons.append({
                "endpoint_token_count": depth,
                "target_arm_acquired_in_intervened_row": target_by_depth[depth],
                "control_arm_acquired_in_intervened_row": control_current,
                "control_arm_acquired_in_suffix": control_suffix,
                "control_target_absent": not control_any,
            })
        control_result = {
            "donor": control_info,
            "rungs": [arm["rung"] for arm in control_arms],
            "arms": control_arms,
            "same_depth_target_comparison": comparisons,
        }
    return {"case_id": str(case.get("case_id", image_id)), "image_id": image_id, "sampled_seed": values["sampled_seed"], "target_owner_id": values["target_owner_id"], "geometry_trust": values["geometry_trust"], "greedy_parent_row_count": parent_count, "parent_token_ids": parent_ids, "parent_token_ids_sha256": hash_prefix_token_ids(parent_ids), "parent_rows": parent_rows, "native_row": native_row, "donor_row": donor_row, "expected_donor": donor_info, "rungs": rungs, "arms": arms, "control_ladder": control_result, "native_noop_parity": parity, "frozen_native_horizon_parity": frozen_native_horizon_parity, "native_stop": native_stop}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--case-id", default=None)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = args.manifest.expanduser().resolve(strict=True)
    manifest = _load_json(manifest_path)
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise SystemExit("manifest requires a non-empty cases list")
    selected = [case for case in cases if isinstance(case, Mapping) and (args.case_id is None or str(case.get("case_id")) == str(args.case_id))]
    if not selected:
        raise SystemExit(f"no case matched --case-id={args.case_id!r}")
    total_budget = int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET))
    if total_budget <= 0:
        raise SystemExit("total_token_budget must be positive")
    base = manifest_path.parent
    greedy_ref = manifest.get("greedy_artifact")
    greedy_path = _validated_artifact_ref(greedy_ref, base=base, label="greedy_artifact")
    greedy_payload = _load_json(greedy_path)
    source_path_declared = _validated_artifact_ref(
        manifest.get("source_jsonl"), base=base, label="source_jsonl"
    )
    for label in ("reviewed_support_artifact", "human_review_decisions"):
        if label not in manifest:
            raise SystemExit(f"manifest requires {label}")
        _validated_artifact_ref(manifest[label], base=base, label=label)
    sampled_payloads: dict[str, Mapping[str, Any]] = {}
    for case in selected:
        ref = case.get("sampled_artifact", case.get("sampled_rollout", case.get("sampled_rollout_json")))
        ref_path = _validated_artifact_ref(
            ref, base=base, label=f"sampled_artifact:{case.get('case_id')}"
        )
        sampled_payloads[str(case.get("case_id"))] = _load_json(ref_path)
    try:
        import torch
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.runtime import assemble_frontend
        from scripts.research.run_same_covered_set_prefix_order_probe import _build_request, _select_example
    except Exception as exc:
        raise SystemExit(f"runtime import failed; no artifact was written: {type(exc).__name__}: {exc}") from exc
    resolved = load_infer_config(args.infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    source_path = Path(config.data.input_jsonl).expanduser().resolve(strict=True)
    if source_path != source_path_declared:
        raise SystemExit("manifest source_jsonl does not match infer-config source")
    examples = list(load_raw_examples(source_path))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    outputs: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as session:
        for index, case in enumerate(selected):
            values = _extract_case_values(case, manifest)
            image_id = values["image_id"]
            greedy_record = _select_rollout(greedy_payload, image_id=image_id, seed=None, mode="greedy")
            sampled_record = _select_rollout(sampled_payloads[str(case.get("case_id"))], image_id=image_id, seed=values["sampled_seed"], mode="sampled")
            if greedy_record["prompt_token_ids"] != sampled_record["prompt_token_ids"]:
                raise SystemExit(f"greedy/sampled prompt mismatch for case {case.get('case_id')}")
            example = _select_example(examples, str(image_id))
            request, plan, prompt_meta = _build_request(config, frontend, example)
            native_inputs, executed_ids, _, _ = session._materialize_native_inputs((request,))
            if list(map(int, executed_ids[0])) != list(greedy_record["prompt_token_ids"]):
                raise SystemExit(f"executed prompt mismatch for case {case.get('case_id')}")
            ledger = [dict(row) for row in case.get("entity_ledger", []) if isinstance(row, Mapping)] or build_positive_entity_ledger(example)
            result = _run_case(case=case, manifest=manifest, greedy_payload=greedy_payload, sampled_payload=sampled_payloads[str(case.get("case_id"))], greedy_record=greedy_record, sampled_record=sampled_record, session=session, native_inputs=_single_native_inputs(native_inputs), tokenizer=session._tokenizer, ledger=ledger, width=int(plan.decoded_width), height=int(plan.decoded_height), image_id=image_id, malformed_limit=int(manifest.get("malformed_limit", DEFAULT_MALFORMED_LIMIT)), noop_gate=True)
            result["prompt"] = {**dict(prompt_meta), "prompt_token_ids": greedy_record["prompt_token_ids"], "prompt_token_ids_sha256": greedy_record["prompt_token_ids_sha256"]}
            outputs.append(result)
        model_identity = session.receipt.to_artifact_dict()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    payload = {"schema_version": SCHEMA_VERSION, "manifest": str(manifest_path), "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(), "config": {"infer_config": str(args.infer_config.resolve()), "device": args.device, "total_token_budget": total_budget}, "model_identity": model_identity, "cases": outputs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
