#!/usr/bin/env python3
"""Run the bounded earliest shared-prefix branch pilot.

Level A forces the sampled route only from the exact first divergence in the
first row through that row's meaningful boundaries.  Level B, used to test a
real sampled-only owner, supplies the complete sampled history before that
owner's row and then releases greedy generation at successive boundaries of
the owner row.  Forced context, the partially supplied current row, and the
released suffix are reported separately so a set gain cannot be hidden by an
injected owner.
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

from scripts.research.build_earliest_shared_prefix_branch_cases import row_spans
from scripts.research.run_greedy_prefix_forced_owner_path import (
    DEFAULT_MALFORMED_LIMIT,
    _annotate_with_qualified_ids,
    _attach_reviewed_owner,
    _covered_after_row,
    _frozen_rows,
    _generate_after_forced_partial_row,
    _generate_row,
    _greedy_suffix,
    _load_json,
    _select_rollout,
    _single_native_inputs,
    append_row_if_complete,
    build_positive_entity_ledger,
    hash_prefix_token_ids,
)


SCHEMA_VERSION = "earliest_shared_prefix_branch_pilot.v1"
BOX_END = 151649
DEFAULT_TOTAL_TOKEN_BUDGET = 512


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(value: Any, *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a token-id sequence")
    return [int(token) for token in value]


def _owner_set(rows: Sequence[Mapping[str, Any]], image_id: Any) -> set[str]:
    owners: set[str] = set()
    for row in rows:
        for owner in row.get("strict_matched_owner_ids", []):
            owners.add(f"{image_id}:{owner}")
        for owner in row.get("reviewed_owner_ids_qualified", []):
            owners.add(str(owner))
    return owners


def _can_attach_reviewed_owner(
    *,
    row_index: int,
    target_row_index: int,
    current_row_is_complete: bool,
    current_row_token_ids: Sequence[int],
    sampled_row_token_ids: Sequence[int],
) -> bool:
    """Allow a reviewed owner only on a complete declared sampled target row.

    ``current_row_is_complete`` refers to the final row assembled after any
    forced prefix and released tail, not to whether the rung forced the whole
    row.  This lets a partial rung receive the reviewed owner when native
    continuation exactly reconstructs the sampled donor row.
    """

    return (
        bool(current_row_is_complete)
        and int(row_index) == int(target_row_index)
        and list(map(int, current_row_token_ids))
        == list(map(int, sampled_row_token_ids))
    )


def _decoder_acquisition_eligible(
    *,
    row_index: int,
    target_row_index: int,
    current_row_is_complete: bool,
    released_tail_token_count: int,
    target_owner_in_intervened_row: bool,
) -> bool:
    """Mark only decoder-completed target rows as acquisition evidence."""

    return (
        bool(current_row_is_complete)
        and int(row_index) == int(target_row_index)
        and int(released_tail_token_count) > 0
        and bool(target_owner_in_intervened_row)
    )


def _row_target_hit(rows: Sequence[Mapping[str, Any]], target_owner_id: str, image_id: Any) -> bool:
    target = str(target_owner_id)
    aliases = {target, target.split(":", 1)[1] if ":" in target else f"{image_id}:{target}"}
    for row in rows:
        observed = _owner_set([row], image_id)
        if observed & aliases:
            return True
    return False


def _harm_receipt(rows: Sequence[Mapping[str, Any]], image_id: Any) -> dict[str, Any]:
    """Count duplicate and unresolved rows without calling unknown rows hallucinations."""

    occurrences: list[str] = []
    unresolved_rows: list[int] = []
    malformed_rows: list[int] = []
    for index, row in enumerate(rows):
        owners = sorted(_owner_set([row], image_id))
        occurrences.extend(owners)
        stop_reason = str((row.get("row_stop") or {}).get("stop_reason", "unknown"))
        if row.get("status") == "failed" or row.get("unmatched_or_ambiguous_prediction_indices"):
            unresolved_rows.append(index)
        if row.get("status") == "failed" or stop_reason not in {"complete_row", "terminal"}:
            malformed_rows.append(index)
    duplicate_owner_ids = sorted({owner for owner in occurrences if occurrences.count(owner) > 1})
    return {
        "duplicate_owner_ids": duplicate_owner_ids,
        "duplicate_owner_count": len(duplicate_owner_ids),
        "unresolved_row_count": len(unresolved_rows),
        "unresolved_row_indices": unresolved_rows,
        "malformed_row_count": len(malformed_rows),
        "malformed_row_indices": malformed_rows,
        "row_count": len(rows),
    }


def _make_exact_row(record: Mapping[str, Any], row_index: int) -> dict[str, Any]:
    rows = _frozen_rows(record)
    if row_index < 0 or row_index >= len(rows):
        raise ValueError(f"row index {row_index} unavailable")
    return dict(rows[row_index])


def _context_rows(
    *,
    sampled_record: Mapping[str, Any],
    row_count: int,
    ledger: Sequence[Mapping[str, Any]],
    image_id: Any,
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    rows = _frozen_rows(sampled_record)[:row_count]
    covered: set[str] = set()
    output: list[dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        _annotate_with_qualified_ids(
            row,
            ledger=ledger,
            image_id=image_id,
            width=width,
            height=height,
            covered=sorted(covered),
        )
        output.append(row)
        covered.update(str(value) for value in row.get("strict_matched_owner_ids", []))
        covered.update(
            str(value).split(":", 1)[1]
            for value in row.get("reviewed_owner_ids_qualified", [])
            if ":" in str(value)
        )
    return output, sorted(covered)


def _native_reference(
    *,
    greedy_record: Mapping[str, Any],
    target_row_index: int,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    ledger: Sequence[Mapping[str, Any]],
    image_id: Any,
    width: int,
    height: int,
    total_token_budget: int,
    complete_row_budget: int,
    malformed_limit: int,
) -> dict[str, Any]:
    frozen_rows = _frozen_rows(greedy_record)
    spans = row_spans(greedy_record["generated_token_ids"])
    if target_row_index >= len(spans):
        raise ValueError("greedy target row is unavailable")
    parent_end = spans[target_row_index][0]
    parent_ids = list(map(int, greedy_record["generated_token_ids"][:parent_end]))
    parent_rows, covered = _context_rows(
        sampled_record=greedy_record,
        row_count=target_row_index,
        ledger=ledger,
        image_id=image_id,
        width=width,
        height=height,
    )
    remaining = total_token_budget - len(parent_ids)
    native_row = _generate_row(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=parent_ids,
        tokenizer=tokenizer,
        image_width=width,
        image_height=height,
        mode="greedy",
        seed=None,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        max_new_tokens=remaining,
        malformed_limit=malformed_limit,
        row_index=target_row_index,
    )
    expected = frozen_rows[target_row_index]["raw_generated_token_ids"]
    observed = list(map(int, native_row.get("raw_generated_token_ids", [])))
    if observed != list(map(int, expected)):
        raise ValueError("native target-row replay did not match frozen greedy tokens")
    _annotate_with_qualified_ids(
        native_row,
        ledger=ledger,
        image_id=image_id,
        width=width,
        height=height,
        covered=covered,
    )
    current_prefix, append_receipt = append_row_if_complete(parent_ids, native_row)
    native_row["append_receipt"] = append_receipt
    native_row["accepted_complete_row"] = bool(append_receipt.get("appended"))
    covered_after, coverage_receipt = _covered_after_row(covered, native_row)
    suffix, suffix_used, _ = _greedy_suffix(
        session=session,
        native_inputs=native_inputs,
        prefix=current_prefix,
        tokenizer=tokenizer,
        width=width,
        height=height,
        horizon_rows=complete_row_budget - target_row_index - 1,
        remaining_tokens=total_token_budget - len(parent_ids) - len(observed),
        ledger=ledger,
        image_id=image_id,
        covered=covered_after,
        repetition_penalty=1.0,
        malformed_limit=malformed_limit,
        start_row_index=target_row_index + 1,
    )
    # The row runner may append a final terminal/stop record after the last
    # complete object row.  Compare complete rows to the frozen artifact and
    # check that terminal state separately; do not mistake the terminal record
    # for an extra generated object row.
    observed_complete = [
        row for row in suffix if row.get("accepted_complete_row")
    ]
    horizon = int(complete_row_budget - target_row_index - 1)
    expected_complete = frozen_rows[
        target_row_index + 1 : target_row_index + 1 + horizon
    ]
    if len(observed_complete) != len(expected_complete):
        raise ValueError(
            "native suffix replay returned an incomplete row horizon: "
            f"observed_complete_rows={len(observed_complete)} "
            f"expected_complete_rows={len(expected_complete)}"
        )
    observed_suffix = [row.get("raw_generated_token_ids", []) for row in observed_complete]
    expected_suffix = [row["raw_generated_token_ids"] for row in expected_complete]
    if observed_suffix != expected_suffix:
        mismatch = next(
            (
                index,
                observed_row,
                expected_row,
            )
            for index, (observed_row, expected_row) in enumerate(
                zip(observed_suffix, expected_suffix)
            )
            if list(map(int, observed_row)) != list(map(int, expected_row))
        ) if len(observed_suffix) and len(expected_suffix) else None
        raise ValueError(
            "native suffix replay did not match frozen greedy tokens: "
            f"observed_rows={len(observed_suffix)} expected_rows={len(expected_suffix)} "
            f"first_mismatch={mismatch[0] if mismatch else None} "
            f"observed_len={len(mismatch[1]) if mismatch else None} "
            f"expected_len={len(mismatch[2]) if mismatch else None}"
        )
    expected_stopped = (
        len(frozen_rows) < target_row_index + 1 + horizon
        and str(greedy_record.get("stop_reason")) in {"im_end", "terminal", "eos"}
    )
    observed_terminal = [
        row for row in suffix if not row.get("accepted_complete_row")
    ]
    if expected_stopped != bool(observed_terminal):
        raise ValueError(
            "native suffix terminal parity did not match frozen greedy stop: "
            f"expected_stopped={expected_stopped} observed_terminal_rows={len(observed_terminal)}"
        )
    return {
        "parent_rows": parent_rows,
        "parent_token_ids": parent_ids,
        "parent_owner_ids": sorted(_owner_set(parent_rows, image_id)),
        "intervened_row": native_row,
        "suffix_rows": suffix,
        "suffix_used_token_count": suffix_used,
        "owner_ids": sorted(_owner_set([*parent_rows, native_row, *suffix], image_id)),
        "harm_receipt": _harm_receipt([*parent_rows, native_row, *suffix], image_id),
        "native_noop_parity": {"passed": True, "target_row_tokens_equal": True, "suffix_tokens_equal": True},
        "coverage_update_receipt": coverage_receipt,
    }


def _run_rung(
    *,
    rung: Mapping[str, Any],
    case: Mapping[str, Any],
    sampled_record: Mapping[str, Any],
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    ledger: Sequence[Mapping[str, Any]],
    image_id: Any,
    width: int,
    height: int,
    total_token_budget: int,
    complete_row_budget: int,
    malformed_limit: int,
) -> dict[str, Any]:
    sampled_ids = list(map(int, sampled_record["generated_token_ids"]))
    row_start = int(rung["row_start_token_index"])
    endpoint = row_start + int(rung["endpoint_token_count"])
    context_ids = sampled_ids[:row_start]
    forced_row_prefix = sampled_ids[row_start:endpoint]
    row_index = len(row_spans(context_ids))
    context_rows, covered = _context_rows(
        sampled_record=sampled_record,
        row_count=row_index,
        ledger=ledger,
        image_id=image_id,
        width=width,
        height=height,
    )
    if not forced_row_prefix:
        raise ValueError("staircase rung has empty forced current-row prefix")
    target_row_end = int(rung["row_end_token_index"])
    current_row_is_complete = endpoint == target_row_end
    if current_row_is_complete:
        current_row = _make_exact_row(sampled_record, row_index)
        current_row["forced_row_prefix_token_ids"] = forced_row_prefix
        current_row["released_tail_token_ids"] = []
        current_row["mode"] = "exact_donor_row"
        released_tail: list[int] = []
    else:
        released_budget = total_token_budget - len(context_ids) - len(forced_row_prefix)
        if released_budget <= 0:
            raise ValueError("no token budget remains after forced context")
        current_row = _generate_after_forced_partial_row(
            session=session,
            native_inputs=native_inputs,
            parent_prefix_token_ids=context_ids,
            forced_row_prefix_token_ids=forced_row_prefix,
            tokenizer=tokenizer,
            image_width=width,
            image_height=height,
            repetition_penalty=1.0,
            max_new_tokens=released_budget,
            malformed_limit=malformed_limit,
            row_index=row_index,
        )
        released_tail = list(map(int, current_row.get("released_tail_token_ids", [])))
    _annotate_with_qualified_ids(
        current_row,
        ledger=ledger,
        image_id=image_id,
        width=width,
        height=height,
        covered=covered,
    )
    # The reviewed owner belongs only to the declared sampled target row.  A
    # complete Level-A first-row donor is an arbitrary branch probe and must
    # never inherit that later row's human owner label merely because its
    # tokens equal the supplied prefix.
    current_prefix, append_receipt = append_row_if_complete(context_ids, current_row)
    current_row["append_receipt"] = append_receipt
    current_row["accepted_complete_row"] = bool(append_receipt.get("appended"))
    if _can_attach_reviewed_owner(
        row_index=row_index,
        target_row_index=int(case["sampled_donor_row_index"]),
        current_row_is_complete=bool(current_row["accepted_complete_row"]),
        current_row_token_ids=current_row.get("raw_generated_token_ids", []),
        sampled_row_token_ids=sampled_ids[row_start:target_row_end],
    ):
        reviewed = case.get("reviewed_donor_owner_id")
        if reviewed:
            _attach_reviewed_owner(current_row, owner_id=str(reviewed), image_id=image_id)
    if current_row.get("accepted_complete_row"):
        covered_after, coverage_receipt = _covered_after_row(covered, current_row)
    else:
        covered_after = list(covered)
        coverage_receipt = {"coverage_updated": False, "refusal_reason": "row_not_complete"}
    if current_row.get("accepted_complete_row"):
        suffix, suffix_used, _ = _greedy_suffix(
            session=session,
            native_inputs=native_inputs,
            prefix=current_prefix,
            tokenizer=tokenizer,
            width=width,
            height=height,
            horizon_rows=complete_row_budget - row_index - 1,
            remaining_tokens=total_token_budget - len(context_ids) - len(current_row.get("raw_generated_token_ids", [])),
            ledger=ledger,
            image_id=image_id,
            covered=covered_after,
            repetition_penalty=1.0,
            malformed_limit=malformed_limit,
            start_row_index=row_index + 1,
        )
    else:
        # A terminal or malformed current row has no valid next-row prefix.
        # Do not replay a suffix from the unchanged pre-row context.
        suffix, suffix_used = [], 0
    forced_context_owners = _owner_set(context_rows, image_id)
    intervened_owners = _owner_set([current_row], image_id)
    released_suffix_owners = _owner_set(suffix, image_id)
    all_owners = forced_context_owners | intervened_owners | released_suffix_owners
    harm_receipt = _harm_receipt([*context_rows, current_row, *suffix], image_id)
    target = str(case["target_owner_id"])
    target_aliases = {target, target.split(":", 1)[1] if ":" in target else target}
    def target_in(owners: set[str]) -> bool:
        return bool(owners & target_aliases)
    decoder_acquisition_eligible = _decoder_acquisition_eligible(
        row_index=row_index,
        target_row_index=int(case["sampled_donor_row_index"]),
        current_row_is_complete=bool(current_row.get("accepted_complete_row")),
        released_tail_token_count=len(released_tail),
        target_owner_in_intervened_row=target_in(intervened_owners),
    )
    return {
        "rung": dict(rung),
        "row_index": row_index,
        "context_token_count": len(context_ids),
        "forced_current_row_token_count": len(forced_row_prefix),
        "released_current_row_token_count": len(released_tail),
        "forced_context_owner_ids": sorted(forced_context_owners),
        "intervened_row_owner_ids": sorted(intervened_owners),
        "released_suffix_owner_ids": sorted(released_suffix_owners),
        "released_only_owner_ids": sorted(released_suffix_owners),
        "fixed_budget_owner_ids": sorted(all_owners),
        "fixed_budget_owner_ids_excluding_forced_context": sorted(intervened_owners | released_suffix_owners),
        "fixed_budget_owner_ids_excluding_intervened_row": sorted(forced_context_owners | released_suffix_owners),
        "harm_receipt": harm_receipt,
        "target_owner_in_forced_context": target_in(forced_context_owners),
        "target_owner_in_intervened_row": target_in(intervened_owners),
        "target_owner_in_released_suffix": target_in(released_suffix_owners),
        "target_owner_in_released_only": target_in(released_suffix_owners),
        "target_owner_decoder_acquisition_eligible": decoder_acquisition_eligible,
        "intervened_row": current_row,
        "suffix_rows": suffix,
        "coverage_update_receipt": coverage_receipt,
        "budget_receipt": {
            "total_token_budget": total_token_budget,
            "used_token_count": len(context_ids) + len(current_row.get("raw_generated_token_ids", [])) + suffix_used,
            "remaining_token_count": total_token_budget - len(context_ids) - len(current_row.get("raw_generated_token_ids", [])) - suffix_used,
            "complete_row_budget": complete_row_budget,
            "row_index": row_index,
        },
    }


def _run_case(*, case: Mapping[str, Any], greedy_payload: Mapping[str, Any], sampled_payload: Mapping[str, Any], session: Any, native_inputs: Mapping[str, Any], tokenizer: Any, ledger: Sequence[Mapping[str, Any]], width: int, height: int, total_token_budget: int, malformed_limit: int) -> dict[str, Any]:
    image_id = int(case["image_id"])
    seed = int(case["sampled_seed"])
    greedy = _select_rollout(greedy_payload, image_id=image_id, seed=None, mode="greedy")
    sampled = _select_rollout(sampled_payload, image_id=image_id, seed=seed, mode="sampled")
    if greedy["prompt_token_ids"] != sampled["prompt_token_ids"]:
        raise ValueError("greedy and sampled prompt ids differ")
    greedy_ids = list(map(int, greedy["generated_token_ids"]))
    sampled_ids = list(map(int, sampled["generated_token_ids"]))
    spans = row_spans(sampled_ids)
    target_index = int(case["sampled_donor_row_index"])
    target_start, target_end = spans[target_index]
    lcp = 0
    for left, right in zip(greedy_ids, sampled_ids):
        if left != right:
            break
        lcp += 1
    if lcp >= min(len(greedy_ids), len(sampled_ids)):
        raise ValueError("sampled and greedy routes do not diverge")
    native = _native_reference(
        greedy_record=greedy,
        target_row_index=target_index,
        session=session,
        native_inputs=native_inputs,
        tokenizer=tokenizer,
        ledger=ledger,
        image_id=image_id,
        width=width,
        height=height,
        total_token_budget=total_token_budget,
        complete_row_budget=int(case["complete_row_budget"]),
        malformed_limit=malformed_limit,
    )
    rungs: list[dict[str, Any]] = []
    for rung in case.get("staircase", []):
        rungs.append(
            _run_rung(
                rung=rung,
                case=case,
                sampled_record=sampled,
                session=session,
                native_inputs=native_inputs,
                tokenizer=tokenizer,
                ledger=ledger,
                image_id=image_id,
                width=width,
                height=height,
                total_token_budget=total_token_budget,
                complete_row_budget=int(case["complete_row_budget"]),
                malformed_limit=malformed_limit,
            )
        )
        # Level A is the proposed fast discriminator.  Once a rung retains or
        # improves the native set without a new harmful owner, Level B still
        # remains in the receipt but can be skipped by a future larger runner.
    native_owners = set(native["owner_ids"])
    native_harm = native["harm_receipt"]
    for result in rungs:
        current = set(result["fixed_budget_owner_ids"])
        released_only = set(result["released_only_owner_ids"])
        harm = result["harm_receipt"]
        result["fixed_budget_delta_vs_native"] = {
            "unique_owner_count_delta": len(current) - len(native_owners),
            "added_owner_ids": sorted(current - native_owners),
            "removed_owner_ids": sorted(native_owners - current),
            "target_owner_not_counted_as_forced_context": not result["target_owner_in_forced_context"],
            "decoder_acquisition_eligible": bool(result["target_owner_decoder_acquisition_eligible"]),
            "released_only_owner_count_delta": len(released_only) - len(native_owners),
            "released_only_added_owner_ids": sorted(released_only - native_owners),
            "duplicate_owner_count_delta": int(harm["duplicate_owner_count"]) - int(native_harm["duplicate_owner_count"]),
            "new_duplicate_owner_ids": sorted(set(harm["duplicate_owner_ids"]) - set(native_harm["duplicate_owner_ids"])),
            "unresolved_row_count_delta": int(harm["unresolved_row_count"]) - int(native_harm["unresolved_row_count"]),
            "malformed_row_count_delta": int(harm["malformed_row_count"]) - int(native_harm["malformed_row_count"]),
        }
    return {
        "case_id": case["case_id"],
        "image_id": image_id,
        "sampled_seed": seed,
        "target_owner_id": case["target_owner_id"],
        "target_row_index": target_index,
        "target_row_span_in_sampled": {"start": target_start, "end": target_end},
        "prompt_token_ids_sha256": greedy["prompt_token_ids_sha256"],
        "global_longest_common_prefix": {
            "token_count": lcp,
            "token_ids_sha256": hash_prefix_token_ids(greedy_ids[:lcp]),
            "first_divergence_index": lcp,
            "greedy_token": greedy_ids[lcp],
            "sampled_token": sampled_ids[lcp],
        },
        "native_reference": native,
        "rungs": rungs,
    }


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
        raise SystemExit("manifest requires cases")
    selected = [case for case in cases if isinstance(case, Mapping) and (args.case_id is None or str(case.get("case_id")) == str(args.case_id))]
    if not selected:
        raise SystemExit(f"no case matched --case-id={args.case_id!r}")
    try:
        import torch
        from src.config.fingerprint import sha256_json
        from src.config.inference import load_infer_config
        from src.data import load_raw_examples
        from src.inference.backend import open_backend_session
        from src.inference.runtime import assemble_frontend
        from scripts.research.run_same_covered_set_prefix_order_probe import _build_request, _select_example
    except Exception as exc:
        raise SystemExit(f"runtime import failed: {type(exc).__name__}: {exc}") from exc
    resolved = load_infer_config(args.infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    source_path = Path(config.data.input_jsonl).expanduser().resolve(strict=True)
    examples = list(load_raw_examples(source_path))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    outputs: list[dict[str, Any]] = []
    with open_backend_session(frontend.launch) as session:
        for case in selected:
            image_id = int(case["image_id"])
            request, plan, prompt_meta = _build_request(config, frontend, _select_example(examples, str(image_id)))
            native_inputs, executed_ids, _, _ = session._materialize_native_inputs((request,))
            sampled_payload = _load_json(Path(case["sampled_artifact"]["path"]))
            greedy_payload = _load_json(Path(manifest["greedy_artifact"]["path"]))
            greedy = _select_rollout(greedy_payload, image_id=image_id, seed=None, mode="greedy")
            if list(map(int, executed_ids[0])) != list(map(int, greedy["prompt_token_ids"])):
                raise SystemExit(f"executed prompt mismatch for {case['case_id']}")
            ledger = build_positive_entity_ledger(_select_example(examples, str(image_id)))
            result = _run_case(
                case=case,
                greedy_payload=greedy_payload,
                sampled_payload=sampled_payload,
                session=session,
                native_inputs=_single_native_inputs(native_inputs),
                tokenizer=session._tokenizer,
                ledger=ledger,
                width=int(plan.decoded_width),
                height=int(plan.decoded_height),
                total_token_budget=int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET)),
                malformed_limit=int(manifest.get("malformed_limit", DEFAULT_MALFORMED_LIMIT)),
            )
            result["prompt"] = {**dict(prompt_meta), "prompt_token_ids_sha256": greedy["prompt_token_ids_sha256"]}
            outputs.append(result)
        model_identity = session.receipt.to_artifact_dict()
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {output_path}; pass --force")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "config": {"infer_config": str(args.infer_config.resolve()), "device": args.device, "total_token_budget": int(manifest.get("total_token_budget", DEFAULT_TOTAL_TOKEN_BUDGET))},
        "model_identity": model_identity,
        "cases": outputs,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
