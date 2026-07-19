#!/usr/bin/env python3
"""Discover and replay local branch states for downstream unique-object value.

This file is intentionally an experiment-local Hugging Face (HF) seam.  It
reuses prompt construction, image materialisation, row parsing, and the
private loaded-model call from ``run_same_covered_set_prefix_order_probe``.
It is not a production inference backend and it does not decide whether an
unmatched prediction is real.  A supplied entity ledger must explicitly mark
an entity as verified before it can be used as a positive owner.

The scientific unit is a fixed natural prefix.  We first record native greedy
and sampled trajectories.  Only after a greedy row is classified as a verified
duplicate, terminal-with-remaining-entity, or malformed state do we sample
same-prefix alternatives.  Every intervention uses exact token ids appended
to the immutable prefix; generated text is never decoded and re-tokenized.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


SCHEMA_VERSION = "local_branch_causal_value.v2"
INTERVENTION_MODE = "exact_token_prefix_append"
DEFAULT_HORIZON_ROWS = 8
DEFAULT_DISCOVERY_SEEDS = tuple(range(11, 19))
INTERVENTION_ARMS = (
    "native",
    "force_native_token",
    "force_rescue_token",
    "force_native_row",
    "force_rescue_row",
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_frozen_file_identity(
    manifest: Mapping[str, Any],
    *,
    infer_config_path: Path,
    source_jsonl_path: Path,
) -> dict[str, Any]:
    """Fail before model execution when a frozen scientific input drifted."""

    primary = manifest.get("primary_checkpoint")
    inference = manifest.get("inference_contract")
    if not isinstance(primary, Mapping) or not isinstance(inference, Mapping):
        raise ValueError("manifest lacks frozen checkpoint or inference identity")
    checkpoint_json = Path(str(primary.get("checkpoint_json", ""))).expanduser().resolve(strict=True)
    expected_paths = {
        "infer_config": (infer_config_path.resolve(strict=True), Path(str(inference.get("config_path", ""))).expanduser().resolve(strict=True), str(inference.get("config_sha256", ""))),
        "source_jsonl": (source_jsonl_path.resolve(strict=True), Path(str(inference.get("source_jsonl", ""))).expanduser().resolve(strict=True), str(inference.get("source_jsonl_sha256", ""))),
        "checkpoint_json": (checkpoint_json, checkpoint_json, str(primary.get("checkpoint_json_sha256", ""))),
        "adapter_model": (checkpoint_json.parent / "adapter" / "adapter_model.safetensors", checkpoint_json.parent / "adapter" / "adapter_model.safetensors", str(primary.get("adapter_model_sha256", ""))),
        "special_token_embeddings": (checkpoint_json.parent / "special_token_embeddings" / "special_token_embeddings.safetensors", checkpoint_json.parent / "special_token_embeddings" / "special_token_embeddings.safetensors", str(primary.get("special_token_embeddings_tensor_sha256", ""))),
    }
    receipt: dict[str, Any] = {}
    for label, (observed_path, frozen_path, expected_sha) in expected_paths.items():
        observed_path = observed_path.resolve(strict=True)
        if observed_path != frozen_path:
            raise ValueError(f"{label} path drift: observed {observed_path}, frozen {frozen_path}")
        observed_sha = sha256_file(observed_path)
        if not expected_sha or observed_sha != expected_sha:
            raise ValueError(f"{label} SHA-256 drift: observed {observed_sha}, frozen {expected_sha}")
        receipt[label] = {"path": str(observed_path), "sha256": observed_sha}
    return receipt


def git_execution_identity(workdir: Path) -> dict[str, Any]:
    """Record source identity without mutating or staging the worktree."""

    def run(*args: str) -> str:
        return subprocess.check_output(
            ["git", *args], cwd=workdir, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        status = run("status", "--porcelain=v1", "--untracked-files=all")
        return {
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(status),
            "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
            "runner_sha256": sha256_file(Path(__file__).resolve()),
        }
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}", "runner_sha256": sha256_file(Path(__file__).resolve())}


def hash_prefix_token_ids(prefix_token_ids: Sequence[int]) -> str:
    """Hash an exact ordered token prefix for deduplication and receipts."""

    return _sha256_json([int(value) for value in prefix_token_ids])


def _normalise_ids(value: Sequence[int], *, label: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a token-id sequence")
    ids: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError(f"{label} must contain non-negative integers")
        ids.append(int(item))
    return ids


def deduplicate_prefix_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact prefixes while retaining every natural next action.

    The scientific comparison is defined at an exact prefix.  Provenance alone
    is insufficient: discovery must also retain the row that each root
    trajectory naturally emitted from that prefix.  Coverage disagreements at
    byte-identical prefixes are refused rather than silently taking whichever
    trajectory happened to be visited first.
    """

    by_hash: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        ids = _normalise_ids(record.get("prefix_token_ids", []), label="prefix_token_ids")
        prefix_hash = hash_prefix_token_ids(ids)
        existing = by_hash.get(prefix_hash)
        if existing is None:
            record["prefix_token_ids"] = ids
            record["prefix_token_ids_sha256"] = prefix_hash
            provenance = record.pop("trajectory_provenance", None)
            natural_action = record.pop("natural_action", None)
            existing = {**record, "trajectory_provenance": [], "natural_actions": []}
            if provenance is not None:
                existing["trajectory_provenance"].append(provenance)
            if natural_action is not None:
                existing["natural_actions"].append(natural_action)
            by_hash[prefix_hash] = existing
            continue
        incoming_coverage = sorted(str(value) for value in record.get("covered_entity_ids", []))
        existing_coverage = sorted(str(value) for value in existing.get("covered_entity_ids", []))
        incoming_valid = bool(record.get("coverage_valid", True))
        existing_valid = bool(existing.get("coverage_valid", True))
        if incoming_coverage != existing_coverage or incoming_valid != existing_valid:
            existing["coverage_valid"] = False
            existing["coverage_refusal_reason"] = "exact_prefix_coverage_state_disagreement"
        provenance = record.get("trajectory_provenance")
        if provenance is not None and provenance not in existing["trajectory_provenance"]:
            existing["trajectory_provenance"].append(provenance)
        natural_action = record.get("natural_action")
        if natural_action is not None:
            existing["natural_actions"].append(natural_action)
    return list(by_hash.values())


def natural_action_record(
    row: Mapping[str, Any],
    *,
    trajectory_index: int,
    mode: str,
    seed: int | None,
) -> dict[str, Any]:
    """Materialize the naturally emitted action needed for later case freeze."""

    return {
        "trajectory_index": int(trajectory_index),
        "row_index": int(row.get("row_index", 0)),
        "mode": str(mode),
        "seed": None if seed is None else int(seed),
        "status": row.get("status"),
        "accepted_complete_row": bool(row.get("accepted_complete_row")),
        "raw_generated_token_ids": list(map(int, row.get("raw_generated_token_ids", []))),
        "raw_generated_token_ids_sha256": row.get("raw_generated_token_ids_sha256"),
        "raw_generated_text": row.get("raw_generated_text"),
        "row_stop": row.get("row_stop"),
        "parse_evidence": row.get("parse_evidence"),
        "parsed_predictions": row.get("parsed_predictions", []),
        "entity_matches": row.get("entity_matches", []),
        "strict_matched_owner_ids": row.get("strict_matched_owner_ids", []),
        "covered_prefix_owner_ids": row.get("covered_prefix_owner_ids", []),
        "uncovered_ledger_owner_ids": row.get("uncovered_ledger_owner_ids", []),
        "unmatched_or_ambiguous_prediction_indices": row.get(
            "unmatched_or_ambiguous_prediction_indices", []
        ),
    }


def build_intervention_arms(
    *,
    parent_prefix_token_ids: Sequence[int],
    shared_row_prefix_token_ids: Sequence[int],
    native_token_ids: Sequence[int],
    rescue_token_ids: Sequence[int],
    native_row_token_ids: Sequence[int],
    rescue_row_token_ids: Sequence[int],
) -> dict[str, dict[str, Any]]:
    """Construct exposure-matched force-and-release arms.

    ``force_*_token`` appends the candidates' exact common row prefix and one
    differing token, then releases to native greedy generation.
    ``force_*_row`` appends the full exact row and then releases to future-row
    generation.  The native arm is a no-op reference.
    """

    parent = _normalise_ids(parent_prefix_token_ids, label="parent_prefix_token_ids")
    shared = _normalise_ids(shared_row_prefix_token_ids, label="shared_row_prefix_token_ids")
    native_token = _normalise_ids(native_token_ids, label="native_token_ids")
    rescue_token = _normalise_ids(rescue_token_ids, label="rescue_token_ids")
    native_row = _normalise_ids(native_row_token_ids, label="native_row_token_ids")
    rescue_row = _normalise_ids(rescue_row_token_ids, label="rescue_row_token_ids")
    if len(native_token) != 1 or len(rescue_token) != 1:
        raise ValueError("token arms require exactly one token id")
    if not native_row or not rescue_row:
        raise ValueError("row arms require non-empty complete-row token ids")
    if native_row[: len(shared) + 1] != shared + native_token:
        raise ValueError("native row does not contain shared prefix plus native branch token")
    if rescue_row[: len(shared) + 1] != shared + rescue_token:
        raise ValueError("rescue row does not contain shared prefix plus rescue branch token")
    branch_parent = parent + shared
    return {
        "native": {
            "arm_name": "native",
            "intervention_mode": INTERVENTION_MODE,
            "forced_token_ids": [],
            "forced_row_token_ids": [],
            "effective_prefix_token_ids": parent,
            "release_after": "none",
        },
        "force_native_token": {
            "arm_name": "force_native_token",
            "intervention_mode": INTERVENTION_MODE,
            "forced_token_ids": native_token,
            "shared_row_prefix_token_ids": shared,
            "forced_row_token_ids": [],
            "effective_prefix_token_ids": branch_parent + native_token,
            "release_after": "one_token",
        },
        "force_rescue_token": {
            "arm_name": "force_rescue_token",
            "intervention_mode": INTERVENTION_MODE,
            "forced_token_ids": rescue_token,
            "shared_row_prefix_token_ids": shared,
            "forced_row_token_ids": [],
            "effective_prefix_token_ids": branch_parent + rescue_token,
            "release_after": "one_token",
        },
        "force_native_row": {
            "arm_name": "force_native_row",
            "intervention_mode": INTERVENTION_MODE,
            "forced_token_ids": [],
            "forced_row_token_ids": native_row,
            "effective_prefix_token_ids": parent + native_row,
            "release_after": "complete_row",
        },
        "force_rescue_row": {
            "arm_name": "force_rescue_row",
            "intervention_mode": INTERVENTION_MODE,
            "forced_token_ids": [],
            "forced_row_token_ids": rescue_row,
            "effective_prefix_token_ids": parent + rescue_row,
            "release_after": "complete_row",
        },
    }


def validate_intervention_arm_symmetry(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Verify the parity contracts before any model call."""

    missing = [name for name in INTERVENTION_ARMS if name not in arms]
    if missing:
        raise ValueError(f"missing intervention arms: {missing}")
    native = list(map(int, arms["native"].get("effective_prefix_token_ids", [])))
    native_token = list(map(int, arms["force_native_token"].get("effective_prefix_token_ids", [])))
    native_row = list(map(int, arms["force_native_row"].get("effective_prefix_token_ids", [])))
    shared = list(map(int, arms["force_native_token"].get("shared_row_prefix_token_ids", [])))
    if native_token[:-1] != native + shared or native_row[: len(native)] != native:
        raise ValueError("forced-native arms do not expose the same parent prefix")
    if arms["native"].get("release_after") != "none":
        raise ValueError("native arm must not force or release a token")
    for name in INTERVENTION_ARMS:
        if arms[name].get("intervention_mode") != INTERVENTION_MODE:
            raise ValueError(f"arm {name} has an unsupported intervention mode")
    return {
        "parent_prefix_token_ids_sha256": hash_prefix_token_ids(native),
        "branch_prefix_token_ids_sha256": hash_prefix_token_ids(native + shared),
        "native_token_noop_expected": True,
        "native_row_noop_expected": True,
        "exposure_matched": True,
    }


def classify_greedy_state(
    greedy_row: Mapping[str, Any],
    *,
    covered_entity_ids: Sequence[str],
    entity_ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify a greedy row without treating unmatched boxes as negatives."""

    ledger = {str(row.get("entity_id")): row for row in entity_ledger if row.get("entity_id") is not None}
    covered = {str(value) for value in covered_entity_ids}
    verified_remaining = sorted(
        entity_id
        for entity_id, row in ledger.items()
        if entity_id not in covered and str(row.get("verification", row.get("status", ""))).lower() in {"verified", "approved", "human_verified"}
    )
    row_stop = greedy_row.get("row_stop") if isinstance(greedy_row.get("row_stop"), Mapping) else {}
    stop_reason = str(row_stop.get("stop_reason", "unknown"))
    owners = {str(value) for value in greedy_row.get("strict_matched_owner_ids", [])}
    duplicate_owners = sorted(owners & covered)
    uncovered_owners = sorted(owners - covered)
    result = {
        "stop_reason": stop_reason,
        "covered_owner_ids": duplicate_owners,
        "uncovered_owner_ids": uncovered_owners,
        "verified_remaining_owner_ids": verified_remaining,
        "eligible_for_same_prefix_sampling": False,
        "refusal_reason": None,
    }
    if not verified_remaining:
        result["refusal_reason"] = "no_verified_remaining_owner"
        return result
    if uncovered_owners:
        result["refusal_reason"] = "greedy_already_reached_verified_uncovered_owner"
        return result
    if duplicate_owners:
        result["classification"] = "verified_duplicate"
        result["eligible_for_same_prefix_sampling"] = True
        return result
    if stop_reason == "terminal":
        result["classification"] = "terminal_with_verified_remaining_owner"
        result["eligible_for_same_prefix_sampling"] = True
        return result
    if stop_reason in {"malformed_limit", "contaminated_complete_row"}:
        result["classification"] = "malformed"
        result["eligible_for_same_prefix_sampling"] = True
        return result
    result["refusal_reason"] = "unresolved_greedy_owner_or_stop_state"
    return result


def select_verified_uncovered_owners(
    row: Mapping[str, Any],
    *,
    covered_entity_ids: Sequence[str],
    entity_ledger: Sequence[Mapping[str, Any]],
) -> list[str]:
    """Return only explicitly verified, same-prefix uncovered owners."""

    covered = {str(value) for value in covered_entity_ids}
    verified = {
        str(item.get("entity_id"))
        for item in entity_ledger
        if item.get("entity_id") is not None
        and str(item.get("verification", item.get("status", ""))).lower() in {"verified", "approved", "human_verified"}
    }
    matches = row.get("strict_matched_owner_ids", [])
    return sorted({str(value) for value in matches} & verified - covered)


def build_positive_entity_ledger(example: Any, *, verification: str = "verified") -> list[dict[str, Any]]:
    """Convert a selected ``RawExample`` into a positive-only owner ledger.

    Every row proves that the listed entity exists; it does *not* assert that
    an unmatched prediction is false, nor that the listed objects are complete.
    This distinction is essential for incomplete Common Objects in Context
    (COCO) annotations and human-relabelled dense images.
    """

    objects = getattr(example, "objects", None)
    if not isinstance(objects, Sequence) or isinstance(objects, (str, bytes)) or not objects:
        raise ValueError("selected example has no positive objects")
    ledger: list[dict[str, Any]] = []
    for obj in objects:
        object_id = str(getattr(obj, "object_id", "")).strip()
        description = str(getattr(obj, "description", "")).strip()
        bbox = getattr(obj, "bbox", None)
        if not object_id or not description or not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise ValueError("RawExample object cannot form a positive owner ledger entry")
        ledger.append({
            "entity_id": object_id,
            "description": description,
            "bbox_norm1000": [float(value) for value in bbox],
            "verification": str(verification),
            "positive_only": True,
            "source": "selected_raw_example",
        })
    return ledger


def extend_covered_set_if_unambiguous(
    covered_entity_ids: Sequence[str],
    row: Mapping[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    """Update coverage only when one complete row has one resolved owner."""

    covered = {str(value) for value in covered_entity_ids}
    if row.get("status") != "success" or not row.get("accepted_complete_row"):
        return sorted(covered), {"coverage_updated": False, "refusal_reason": "row_not_complete"}
    matches = row.get("entity_matches")
    if not isinstance(matches, list) or not matches:
        return sorted(covered), {"coverage_updated": False, "refusal_reason": "no_owner_matches"}
    unresolved = [item for item in matches if item.get("status") in {"unmatched", "ambiguous"}]
    owners = sorted({str(item.get("matched_entity_id")) for item in matches if item.get("status") == "matched" and item.get("matched_entity_id") is not None})
    if unresolved:
        return sorted(covered), {"coverage_updated": False, "refusal_reason": "unmatched_or_ambiguous_prediction"}
    if len(owners) != 1:
        return sorted(covered), {"coverage_updated": False, "refusal_reason": "multiple_or_missing_owners"}
    covered.add(owners[0])
    return sorted(covered), {"coverage_updated": True, "owner_id": owners[0], "refusal_reason": None}


def downstream_outcome_bookkeeping(
    rows: Sequence[Mapping[str, Any]],
    *,
    covered_entity_ids: Sequence[str],
    branch_owner_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """Separate branch-row output from future rows and count unique owners."""

    covered = {str(value) for value in covered_entity_ids}
    branch = {str(value) for value in branch_owner_ids}
    future = rows[1:] if rows else []
    def owners(source: Sequence[Mapping[str, Any]]) -> set[str]:
        result: set[str] = set()
        for row in source:
            result.update(str(value) for value in row.get("strict_matched_owner_ids", []))
        return result
    branch_owners = owners(rows[:1])
    future_owners = owners(future)
    return {
        "branch_row_owner_ids": sorted(branch_owners),
        "future_row_owner_ids": sorted(future_owners),
        "future_new_owner_ids": sorted(future_owners - covered - branch),
        "future_duplicate_owner_ids": sorted(future_owners & covered),
        "future_branch_revisit_owner_ids": sorted(future_owners & branch),
        "future_row_count": len(future),
    }


def append_row_if_complete(
    prefix_token_ids: Sequence[int],
    row: Mapping[str, Any],
) -> tuple[list[int], dict[str, Any]]:
    """Append exact generated ids only for a clean complete row.

    Terminal, malformed, contaminated, failed, and unresolved rows are
    deliberately left out of the successor prefix.  Returning a refusal
    record instead of silently appending is important for exposure matching.
    """

    current = _normalise_ids(prefix_token_ids, label="prefix_token_ids")
    stop = row.get("row_stop") if isinstance(row.get("row_stop"), Mapping) else {}
    reason = str(stop.get("stop_reason", "unknown"))
    if reason != "complete_row" or row.get("status") != "success":
        return current, {"appended": False, "refusal_reason": f"row_not_clean_complete:{reason}"}
    generated = _normalise_ids(row.get("raw_generated_token_ids", []), label="raw_generated_token_ids")
    if not generated:
        return current, {"appended": False, "refusal_reason": "complete_row_without_token_ids"}
    return current + generated, {"appended": True, "appended_token_count": len(generated), "refusal_reason": None}


def _append_exact_prefix(native_inputs: Mapping[str, Any], token_ids: Sequence[int]) -> tuple[dict[str, Any], int]:
    """Append exact ids to native input tensors; no decode/re-tokenize."""

    import torch

    ids = _normalise_ids(token_ids, label="token_ids")
    input_ids = native_inputs["input_ids"]
    suffix = torch.tensor([ids], dtype=input_ids.dtype, device=input_ids.device)
    model_inputs = dict(native_inputs)
    model_inputs["input_ids"] = torch.cat((input_ids, suffix), dim=1)
    if "attention_mask" in model_inputs:
        model_inputs["attention_mask"] = torch.cat((model_inputs["attention_mask"], torch.ones_like(suffix)), dim=1)
    return model_inputs, int(model_inputs["input_ids"].shape[1])


def _seed_torch(seed: int) -> None:
    import torch

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _generate_row(*, session: Any, native_inputs: Mapping[str, Any], prefix_token_ids: Sequence[int], tokenizer: Any, image_width: int, image_height: int, mode: str, seed: int | None, temperature: float, top_p: float, repetition_penalty: float, max_new_tokens: int, malformed_limit: int, row_index: int = 0) -> dict[str, Any]:
    """Delegate one native row to the tested research helper."""

    from scripts.research.run_same_covered_set_prefix_order_probe import _generate_one

    return _generate_one(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=prefix_token_ids,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        mode=mode,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
        row_index=row_index,
    )


def _generate_after_forced_partial_row(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    parent_prefix_token_ids: Sequence[int],
    forced_row_prefix_token_ids: Sequence[int],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    repetition_penalty: float,
    max_new_tokens: int,
    malformed_limit: int,
    row_index: int = 0,
) -> dict[str, Any]:
    """Greedily finish one row while including forced tokens in row parsing.

    Ordinary one-row generation treats every supplied token as prompt and only
    inspects newly emitted tokens for row closure.  A branch intervention starts
    inside a row, so stopping and parsing must instead inspect the concatenation
    of the forced common prefix/token and the released tail.
    """

    import torch
    from transformers import StoppingCriteriaList
    from scripts.research.run_same_covered_set_prefix_order_probe import (
        _RowStoppingCriteria,
        extract_row_stop,
        validate_generated_row_boundary,
    )
    from src.inference.parsing import parse_compact_object_box_closed

    parent = _normalise_ids(parent_prefix_token_ids, label="parent_prefix_token_ids")
    forced = _normalise_ids(forced_row_prefix_token_ids, label="forced_row_prefix_token_ids")
    if not forced:
        raise ValueError("forced_row_prefix_token_ids must be non-empty")
    native_width = int(native_inputs["input_ids"].shape[1])
    model_inputs, model_input_width = _append_exact_prefix(native_inputs, parent + forced)
    row_start_width = native_width + len(parent)
    kwargs: dict[str, Any] = {
        **model_inputs,
        "max_new_tokens": int(max_new_tokens),
        "repetition_penalty": float(repetition_penalty),
        "do_sample": False,
        "eos_token_id": session._im_end_token_id(),
        "pad_token_id": session._pad_token_id(),
        "return_dict_in_generate": True,
        "output_scores": False,
        "stopping_criteria": StoppingCriteriaList([
            _RowStoppingCriteria(
                prompt_width=row_start_width,
                tokenizer=tokenizer,
                malformed_limit=malformed_limit,
            )
        ]),
    }
    with torch.inference_mode():
        output = session._model.generate(**kwargs)
    sequences = getattr(output, "sequences", None)
    if sequences is None or int(sequences.shape[0]) != 1:
        raise RuntimeError("forced partial-row generation did not return one sequence")
    full_row_ids = [int(value) for value in sequences[0, row_start_width:].tolist()]
    released_tail_ids = [int(value) for value in sequences[0, model_input_width:].tolist()]
    raw_text = tokenizer.decode(full_row_ids, skip_special_tokens=False)
    row_stop = validate_generated_row_boundary(
        raw_text,
        extract_row_stop(raw_text, malformed_limit=malformed_limit),
    )
    if row_stop.get("stop_reason") == "contaminated_complete_row":
        status = "failed"
        parse_evidence: dict[str, Any] = {
            "parse_status": "not_run",
            "predictions": [],
            "dropped_predictions": [],
        }
        parsed_predictions: list[dict[str, Any]] = []
    else:
        parser_text = row_stop.get("row_text") or raw_text
        parsed = parse_compact_object_box_closed(
            parser_text,
            row_id=f"local-branch:forced-greedy:row-{row_index}",
            row_index=int(row_index),
            image_width=int(image_width),
            image_height=int(image_height),
        )
        status = "success"
        parse_evidence = parsed.to_artifact_dict()
        parsed_predictions = parsed.predictions
    return {
        "mode": "forced_prefix_then_greedy",
        "status": status,
        "parent_prefix_token_ids": parent,
        "parent_prefix_token_ids_sha256": hash_prefix_token_ids(parent),
        "forced_row_prefix_token_ids": forced,
        "forced_row_prefix_token_ids_sha256": hash_prefix_token_ids(forced),
        "released_tail_token_ids": released_tail_ids,
        "raw_generated_token_ids": full_row_ids,
        "raw_generated_token_ids_sha256": hash_prefix_token_ids(full_row_ids),
        "raw_generated_text": raw_text,
        "row_stop": row_stop,
        "parse_evidence": parse_evidence,
        "parsed_predictions": parsed_predictions,
    }


def compare_exact_rows(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact deterministic parity fields used by the no-op gate."""

    left_ids = list(map(int, left.get("raw_generated_token_ids", [])))
    right_ids = list(map(int, right.get("raw_generated_token_ids", [])))
    left_stop = left.get("row_stop") if isinstance(left.get("row_stop"), Mapping) else {}
    right_stop = right.get("row_stop") if isinstance(right.get("row_stop"), Mapping) else {}
    left_parse = left.get("parse_evidence") if isinstance(left.get("parse_evidence"), Mapping) else {}
    right_parse = right.get("parse_evidence") if isinstance(right.get("parse_evidence"), Mapping) else {}
    checks = {
        "raw_token_ids_equal": left_ids == right_ids,
        "status_equal": left.get("status") == right.get("status"),
        "stop_reason_equal": left_stop.get("stop_reason") == right_stop.get("stop_reason"),
        "parse_status_equal": left_parse.get("parse_status") == right_parse.get("parse_status"),
    }
    return {"passed": all(checks.values()), "checks": checks}


def _generate_natural_horizon(*, session: Any, native_inputs: Mapping[str, Any], prefix_token_ids: Sequence[int], tokenizer: Any, image_width: int, image_height: int, mode: str, seed: int | None, temperature: float, top_p: float, repetition_penalty: float, max_new_tokens: int, malformed_limit: int, horizon_rows: int) -> dict[str, Any]:
    if not 1 <= int(horizon_rows) <= 8:
        raise ValueError("horizon_rows must be in [1,8]")
    if seed is not None:
        _seed_torch(seed)
    current = [int(value) for value in prefix_token_ids]
    rows: list[dict[str, Any]] = []
    for row_index in range(int(horizon_rows)):
        prefix_hash = hash_prefix_token_ids(current)
        try:
            row = _generate_row(session=session, native_inputs=native_inputs, prefix_token_ids=current, tokenizer=tokenizer, image_width=image_width, image_height=image_height, mode=mode, seed=None, temperature=temperature, top_p=top_p, repetition_penalty=repetition_penalty, max_new_tokens=max_new_tokens, malformed_limit=malformed_limit, row_index=row_index)
        except Exception as exc:
            row = {"status": "failed", "failure": {"type": type(exc).__name__, "message": str(exc)}, "raw_generated_token_ids": [], "row_stop": {"stop_reason": "failed"}}
        row["row_index"] = row_index
        row["input_prefix_token_ids"] = list(current)
        row["input_prefix_token_ids_sha256"] = prefix_hash
        current, append_receipt = append_row_if_complete(current, row)
        appendable = bool(append_receipt["appended"])
        row["append_receipt"] = append_receipt
        row["accepted_complete_row"] = appendable
        row["appended_to_prefix"] = appendable
        row["cumulative_prefix_token_ids"] = list(current)
        row["cumulative_prefix_token_ids_sha256"] = hash_prefix_token_ids(current)
        rows.append(row)
        if not appendable:
            break
    return {"mode": mode, "seed": seed, "rows": rows, "initial_prefix_token_ids": list(prefix_token_ids), "final_prefix_token_ids": current, "horizon_rows_requested": int(horizon_rows), "horizon_rows_generated": sum(bool(row.get("accepted_complete_row")) for row in rows), "horizon_complete": len(rows) == int(horizon_rows) and all(bool(row.get("accepted_complete_row")) for row in rows)}


def run_noop_smoke(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    repetition_penalty: float,
    max_new_tokens: int,
    malformed_limit: int,
) -> dict[str, Any]:
    """Execute model-level token and complete-row no-op parity checks."""

    native = _generate_natural_horizon(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=[],
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        mode="greedy",
        seed=None,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
        horizon_rows=2,
    )
    if len(native["rows"]) < 2 or not native["rows"][0].get("accepted_complete_row"):
        return {
            "passed": False,
            "refusal_reason": "native_root_did_not_supply_complete_branch_and_successor",
            "native": native,
        }
    native_branch = native["rows"][0]
    native_successor = native["rows"][1]
    native_row_ids = list(map(int, native_branch.get("raw_generated_token_ids", [])))
    if len(native_row_ids) < 2:
        return {"passed": False, "refusal_reason": "native_branch_row_too_short", "native": native}
    # A non-trivial mid-row prefix checks the exact implementation needed by
    # later candidate-distinguishing token interventions.
    forced_count = max(1, len(native_row_ids) // 2)
    forced_count = min(forced_count, len(native_row_ids) - 1)
    forced_branch = _generate_after_forced_partial_row(
        session=session,
        native_inputs=native_inputs,
        parent_prefix_token_ids=[],
        forced_row_prefix_token_ids=native_row_ids[:forced_count],
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
    )
    token_branch_parity = compare_exact_rows(native_branch, forced_branch)
    forced_branch_ids = list(map(int, forced_branch.get("raw_generated_token_ids", [])))
    token_successor_run = _generate_natural_horizon(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=forced_branch_ids,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        mode="greedy",
        seed=None,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
        horizon_rows=1,
    )
    row_successor_run = _generate_natural_horizon(
        session=session,
        native_inputs=native_inputs,
        prefix_token_ids=native_row_ids,
        tokenizer=tokenizer,
        image_width=image_width,
        image_height=image_height,
        mode="greedy",
        seed=None,
        temperature=0.4,
        top_p=0.95,
        repetition_penalty=repetition_penalty,
        max_new_tokens=max_new_tokens,
        malformed_limit=malformed_limit,
        horizon_rows=1,
    )
    token_successor = token_successor_run["rows"][0]
    row_successor = row_successor_run["rows"][0]
    token_successor_parity = compare_exact_rows(native_successor, token_successor)
    row_successor_parity = compare_exact_rows(native_successor, row_successor)
    return {
        "passed": bool(
            token_branch_parity["passed"]
            and token_successor_parity["passed"]
            and row_successor_parity["passed"]
        ),
        "forced_native_prefix_token_count": forced_count,
        "native": native,
        "forced_native_branch": forced_branch,
        "forced_native_token_successor": token_successor_run,
        "forced_native_row_successor": row_successor_run,
        "token_branch_parity": token_branch_parity,
        "token_successor_parity": token_successor_parity,
        "row_successor_parity": row_successor_parity,
    }


def _single_native_inputs(native_inputs: Any) -> Mapping[str, Any]:
    """Normalize the current one-request materialization result."""

    if isinstance(native_inputs, (list, tuple)):
        if len(native_inputs) != 1:
            raise ValueError("local branch runner requires one physical request")
        value = native_inputs[0]
    else:
        value = native_inputs
    if not isinstance(value, Mapping):
        raise ValueError("native input materialization did not return a mapping")
    return value


def _annotate_owner_matches(
    row: dict[str, Any],
    *,
    entity_ledger: Sequence[Mapping[str, Any]],
    image_width: int,
    image_height: int,
    covered_entity_ids: Sequence[str],
) -> dict[str, Any]:
    """Attach strict same-category owner evidence without calling unmatched FP."""

    from scripts.research.run_same_covered_set_prefix_order_probe import match_predictions_to_entities

    matches = match_predictions_to_entities(
        row.get("parsed_predictions", []),
        entity_ledger,
        image_width=int(image_width),
        image_height=int(image_height),
        restrict_to_person=False,
    )
    covered = {str(value) for value in covered_entity_ids}
    owners = sorted({str(item.get("matched_entity_id")) for item in matches if item.get("status") == "matched" and item.get("matched_entity_id") is not None})
    row["entity_matches"] = matches
    row["strict_matched_owner_ids"] = owners
    row["covered_prefix_owner_ids"] = sorted(set(owners) & covered)
    row["uncovered_ledger_owner_ids"] = sorted(set(owners) - covered)
    row["unmatched_or_ambiguous_prediction_indices"] = [int(item["prediction_index"]) for item in matches if item.get("status") in {"unmatched", "ambiguous"} and item.get("prediction_index") is not None]
    return row


def collect_same_prefix_rescues(
    *,
    session: Any,
    native_inputs: Mapping[str, Any],
    prefix_token_ids: Sequence[int],
    covered_entity_ids: Sequence[str],
    entity_ledger: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    image_width: int,
    image_height: int,
    seeds: Sequence[int],
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    max_new_tokens: int,
    malformed_limit: int,
    greedy_row: Mapping[str, Any],
) -> dict[str, Any]:
    """Sample alternatives only after explicit greedy-state eligibility."""

    classification = classify_greedy_state(
        greedy_row,
        covered_entity_ids=covered_entity_ids,
        entity_ledger=entity_ledger,
    )
    result: dict[str, Any] = {
        "classification": classification,
        "prefix_token_ids": list(map(int, prefix_token_ids)),
        "prefix_token_ids_sha256": hash_prefix_token_ids(prefix_token_ids),
        "alternatives": [],
    }
    if not classification.get("eligible_for_same_prefix_sampling"):
        result["refusal_reason"] = classification.get("refusal_reason", "state_not_eligible")
        return result
    for seed in seeds:
        row = _generate_row(
            session=session,
            native_inputs=native_inputs,
            prefix_token_ids=prefix_token_ids,
            tokenizer=tokenizer,
            image_width=image_width,
            image_height=image_height,
            mode="sample",
            seed=int(seed),
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            malformed_limit=malformed_limit,
        )
        row = _annotate_owner_matches(row, entity_ledger=entity_ledger, image_width=image_width, image_height=image_height, covered_entity_ids=covered_entity_ids)
        row["verified_uncovered_owner_ids"] = select_verified_uncovered_owners(row, covered_entity_ids=covered_entity_ids, entity_ledger=entity_ledger)
        result["alternatives"].append(row)
    result["rescue_owner_ids"] = sorted({owner for row in result["alternatives"] for owner in row.get("verified_uncovered_owner_ids", [])})
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Discovery manifest JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--infer-config", type=Path, required=True)
    parser.add_argument("--phase", choices=("noop-smoke", "discovery"), default="discovery")
    parser.add_argument("--image-ids", default="", help="Optional comma-separated frozen-pool subset for one-image smoke or parallel shards")
    parser.add_argument("--seeds", default="", help="Optional assertion; must equal the frozen manifest seeds")
    parser.add_argument("--temperature", type=float, default=None, help="Optional assertion; must equal the frozen manifest value")
    parser.add_argument("--top-p", type=float, default=None, help="Optional assertion; must equal the frozen manifest value")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Optional assertion; must equal the frozen manifest value")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Optional assertion; must equal the frozen manifest value")
    parser.add_argument("--horizon-rows", type=int, default=None, help="Optional assertion; must equal the frozen manifest value")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--collect-same-prefix-rescues", action="store_true", help="After discovery, sample eligible same-prefix alternatives; off by default")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {args.output}; pass --force")
    # The full runtime path is intentionally loaded lazily.  This keeps the
    # helper tests model-free while making missing runtime dependencies clear.
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
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    images = manifest.get("images")
    if not isinstance(images, list) or not images:
        raise SystemExit("manifest requires a non-empty images list")
    inference_contract = manifest.get("inference_contract")
    discovery_budget = manifest.get("discovery_budget")
    if not isinstance(inference_contract, Mapping) or not isinstance(discovery_budget, Mapping):
        raise SystemExit("manifest requires inference_contract and discovery_budget mappings")
    if args.collect_same_prefix_rescues and not bool(discovery_budget.get("additional_same_prefix_sampling", False)):
        raise SystemExit("supplementary same-prefix sampling is forbidden by the frozen discovery manifest")
    frozen_seeds = tuple(int(value) for value in discovery_budget.get("sampled_root_seeds", []))
    frozen_horizon = int(discovery_budget.get("maximum_complete_rows_per_trajectory", 0))
    frozen_max_new_tokens = int(discovery_budget.get("maximum_new_tokens_per_row", 0))
    frozen_malformed_limit = int(discovery_budget.get("malformed_row_limit", 0))
    frozen_temperature = float(inference_contract.get("temperature"))
    frozen_top_p = float(inference_contract.get("top_p"))
    frozen_repetition_penalty = float(inference_contract.get("repetition_penalty"))
    if frozen_seeds != DEFAULT_DISCOVERY_SEEDS:
        raise SystemExit(f"manifest discovery seeds drifted from the frozen contract: {frozen_seeds}")
    if not 1 <= frozen_horizon <= 8 or frozen_max_new_tokens <= 0 or frozen_malformed_limit <= 0:
        raise SystemExit("manifest has an invalid frozen row-generation budget")
    assertions = {
        "seeds": None if not args.seeds else tuple(int(value) for value in args.seeds.split(",") if value.strip()),
        "horizon_rows": args.horizon_rows,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
    }
    frozen_values = {
        "seeds": frozen_seeds,
        "horizon_rows": frozen_horizon,
        "max_new_tokens": frozen_max_new_tokens,
        "temperature": frozen_temperature,
        "top_p": frozen_top_p,
        "repetition_penalty": frozen_repetition_penalty,
    }
    for label, asserted in assertions.items():
        if asserted is not None and asserted != frozen_values[label]:
            raise SystemExit(f"--{label.replace('_', '-')}={asserted!r} disagrees with frozen manifest value {frozen_values[label]!r}")
    if frozen_repetition_penalty != 1.0:
        raise SystemExit("this research unit requires repetition_penalty=1.0")
    selected_image_ids = {value.strip() for value in str(args.image_ids).split(",") if value.strip()}
    frozen_image_ids = {str(image.get("image_id")) for image in images}
    unknown_image_ids = sorted(selected_image_ids - frozen_image_ids)
    if unknown_image_ids:
        raise SystemExit(f"--image-ids contains values outside the frozen pool: {unknown_image_ids}")
    if selected_image_ids:
        images = [image for image in images if str(image.get("image_id")) in selected_image_ids]
    if args.phase == "noop-smoke" and len(images) != 1:
        raise SystemExit("--phase noop-smoke requires exactly one selected frozen image")
    resolved = load_infer_config(args.infer_config.expanduser().resolve(strict=True))
    config = resolved.config
    if str(config.model.dtype) != "fp32":
        raise SystemExit(f"this research unit requires model.dtype=fp32, observed {config.model.dtype!r}")
    try:
        frozen_file_identity = validate_frozen_file_identity(
            manifest,
            infer_config_path=args.infer_config,
            source_jsonl_path=Path(config.data.input_jsonl),
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(f"frozen file identity check failed: {exc}") from exc
    examples = list(load_raw_examples(config.data.input_jsonl))
    frontend = assemble_frontend(config, generation_config_fingerprint=sha256_json(config.generation.model_dump(mode="json")))
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
    output_images: list[dict[str, Any]] = []
    seeds = frozen_seeds
    with open_backend_session(frontend.launch) as session:
        for image in images:
            image_id = str(image.get("image_id", "")).strip()
            if not image_id:
                raise SystemExit("manifest image requires image_id")
            example = _select_example(examples, image_id)
            request, plan, prompt_meta = _build_request(config, frontend, example)
            native_inputs, executed_ids, observed_grids, media_sha = session._materialize_native_inputs((request,))
            one_native = _single_native_inputs(native_inputs)
            # The manifest may override/add a human-reviewed positive ledger,
            # but the default is derived from the selected RawExample.  This
            # is positive-only evidence: omitted entities remain unresolved.
            entity_ledger = [dict(row) for row in image.get("entity_ledger", []) if isinstance(row, Mapping)]
            if not entity_ledger:
                entity_ledger = build_positive_entity_ledger(example)
            if args.phase == "noop-smoke":
                smoke = run_noop_smoke(
                    session=session,
                    native_inputs=one_native,
                    tokenizer=session._tokenizer,
                    image_width=int(plan.decoded_width),
                    image_height=int(plan.decoded_height),
                    repetition_penalty=frozen_repetition_penalty,
                    max_new_tokens=frozen_max_new_tokens,
                    malformed_limit=frozen_malformed_limit,
                )
                output_images.append({
                    "image_id": image_id,
                    "prompt": prompt_meta,
                    "no_op_smoke": smoke,
                    "runtime": {
                        "observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]),
                        "executed_media_sha256": media_sha[0],
                        "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0]),
                    },
                })
                continue
            trajectories: list[dict[str, Any]] = []
            trajectories.append(_generate_natural_horizon(session=session, native_inputs=one_native, prefix_token_ids=[], tokenizer=session._tokenizer, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), mode="greedy", seed=None, temperature=frozen_temperature, top_p=frozen_top_p, repetition_penalty=1.0, max_new_tokens=frozen_max_new_tokens, malformed_limit=frozen_malformed_limit, horizon_rows=frozen_horizon))
            for seed in seeds:
                trajectories.append(_generate_natural_horizon(session=session, native_inputs=one_native, prefix_token_ids=[], tokenizer=session._tokenizer, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), mode="sample", seed=seed, temperature=frozen_temperature, top_p=frozen_top_p, repetition_penalty=1.0, max_new_tokens=frozen_max_new_tokens, malformed_limit=frozen_malformed_limit, horizon_rows=frozen_horizon))
            prefix_records = []
            for trajectory_index, trajectory in enumerate(trajectories):
                current = []
                covered: set[str] = set()
                coverage_valid = True
                coverage_refusal_reason: str | None = None
                for row in trajectory["rows"]:
                    _annotate_owner_matches(row, entity_ledger=entity_ledger, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), covered_entity_ids=sorted(covered)) if entity_ledger else None
                    row["verified_uncovered_owner_ids"] = select_verified_uncovered_owners(
                        row,
                        covered_entity_ids=sorted(covered),
                        entity_ledger=entity_ledger,
                    )
                    provenance = {"trajectory_index": trajectory_index, "row_index": row["row_index"], "mode": trajectory["mode"], "seed": trajectory.get("seed")}
                    prefix_records.append({
                        "prefix_token_ids": list(current),
                        "covered_entity_ids": sorted(covered) if coverage_valid else [],
                        "coverage_valid": coverage_valid,
                        "coverage_refusal_reason": coverage_refusal_reason,
                        "trajectory_provenance": provenance,
                        "natural_action": natural_action_record(
                            row,
                            trajectory_index=trajectory_index,
                            mode=trajectory["mode"],
                            seed=trajectory.get("seed"),
                        ),
                    })
                    if row.get("appended_to_prefix"):
                        current.extend(row.get("raw_generated_token_ids", []))
                    if coverage_valid and entity_ledger:
                        covered_after, coverage_receipt = extend_covered_set_if_unambiguous(covered, row)
                        row["coverage_receipt"] = coverage_receipt
                        if coverage_receipt["coverage_updated"]:
                            covered = set(covered_after)
                        else:
                            coverage_valid = False
                            coverage_refusal_reason = str(coverage_receipt["refusal_reason"])
                            row["coverage_refusal_reason"] = coverage_refusal_reason
            unique_prefixes = deduplicate_prefix_records(prefix_records)
            prefix_evaluations: list[dict[str, Any]] = []
            for prefix_record in unique_prefixes:
                prefix = prefix_record["prefix_token_ids"]
                covered_ids = prefix_record.get("covered_entity_ids", [])
                greedy_row = _generate_row(session=session, native_inputs=one_native, prefix_token_ids=prefix, tokenizer=session._tokenizer, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), mode="greedy", seed=None, temperature=frozen_temperature, top_p=frozen_top_p, repetition_penalty=1.0, max_new_tokens=frozen_max_new_tokens, malformed_limit=frozen_malformed_limit)
                if entity_ledger:
                    _annotate_owner_matches(greedy_row, entity_ledger=entity_ledger, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), covered_entity_ids=covered_ids)
                if not prefix_record.get("coverage_valid", True):
                    classification = {
                        "eligible_for_same_prefix_sampling": False,
                        "refusal_reason": "prior_prefix_owner_unresolved",
                        "verified_remaining_owner_ids": [],
                    }
                else:
                    classification = classify_greedy_state(greedy_row, covered_entity_ids=covered_ids, entity_ledger=entity_ledger)
                evaluation = {"prefix": prefix_record, "native_greedy_row": greedy_row, "classification": classification}
                if args.collect_same_prefix_rescues and classification.get("eligible_for_same_prefix_sampling") and entity_ledger:
                    evaluation["same_prefix_rescues"] = collect_same_prefix_rescues(session=session, native_inputs=one_native, prefix_token_ids=prefix, covered_entity_ids=covered_ids, entity_ledger=entity_ledger, tokenizer=session._tokenizer, image_width=int(plan.decoded_width), image_height=int(plan.decoded_height), seeds=seeds, temperature=frozen_temperature, top_p=frozen_top_p, repetition_penalty=1.0, max_new_tokens=frozen_max_new_tokens, malformed_limit=frozen_malformed_limit, greedy_row=greedy_row)
                prefix_evaluations.append(evaluation)
            output_images.append({"image_id": image_id, "prompt": prompt_meta, "trajectories": trajectories, "unique_prefixes": unique_prefixes, "prefix_evaluations": prefix_evaluations, "entity_ledger": entity_ledger, "runtime": {"observed_image_grid_thw": None if observed_grids[0] is None else list(observed_grids[0]), "executed_media_sha256": media_sha[0], "executed_prompt_token_ids_sha256": hash_prefix_token_ids(executed_ids[0])}})
        model_receipt = session.receipt.to_artifact_dict()
    payload = {"schema_version": SCHEMA_VERSION, "experiment": "local_branch_causal_value", "phase": args.phase, "intervention_mode": INTERVENTION_MODE, "source_identity": git_execution_identity(Path(__file__).resolve().parents[2]), "frozen_file_identity": frozen_file_identity, "manifest_sha256": sha256_file(args.manifest.resolve()), "config": {"manifest": str(args.manifest.resolve()), "infer_config": str(args.infer_config.resolve()), "resolved_config_fingerprint": resolved.fingerprint, "selected_image_ids": sorted(selected_image_ids or frozen_image_ids), "device": args.device, "temperature": frozen_temperature, "top_p": frozen_top_p, "repetition_penalty": 1.0, "max_new_tokens": frozen_max_new_tokens, "malformed_limit": frozen_malformed_limit, "horizon_rows": frozen_horizon, "seeds": list(seeds), "physical_batch_size": 1, "model_dtype": "fp32"}, "model_identity": model_receipt, "images": output_images}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
