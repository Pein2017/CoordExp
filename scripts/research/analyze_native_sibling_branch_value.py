#!/usr/bin/env python3
"""Analyze native sibling branch value and commit crossover.

This is intentionally an experiment-local, offline reader.  It consumes the
immutable admission/variant files and one or more completed Wave-Two receipt
roots.  It never performs model inference, never reassigns a duplicate to an
unused ledger owner, and refuses owner-level conclusions when the exact
variant or paired-seed panel is incomplete.

The three modes are deliberately small:

``collect-review``
    Produce a blinded, deterministic list of unmatched COCO-80 candidates.
``freeze-check``
    Check that every candidate has a valid human decision and stable entity
    reference, then emit the frozen review mapping.
``analyze``
    Reconstruct parent coverage, classify the first four suffix rows in time
    order, and emit exact-row/owner branch values, crossover primitives, and
    paired bootstrap intervals.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.research.admit_native_sibling_first_rows import (  # noqa: E402
    DEFAULT_AMBIGUITY_MARGIN,
    DEFAULT_IOU_FLOOR,
    box_iou,
    match_first_row_to_ledger,
    read_accepted_ledger,
)


UNIT_ID = "2026-07-17-native-sibling-row-branch-value-and-commit-crossover"
SCHEMA_VERSION = "native_sibling_branch_value_analysis.v1"
HORIZON = 4
CONFIRMATION_SEED_ROOT = 2026071702000001
BOOTSTRAP_SEED_ROOT = 2026071703000001
BOOTSTRAP_REPLICATES = 100_000
EXPECTED_CONFIRMATION_SEEDS = 8
SUPPORTED_REVIEW_VERDICTS = frozenset({"approve", "reject", "unknown"})


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _write_once(path: Path, value: Mapping[str, Any]) -> None:
    path = path.expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _resolve(path: str | Path, *, base: Path | None = None) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute() and base is not None:
        candidate = base / candidate
    return candidate.resolve(strict=True)


def _unit_digest(unit_path: Path) -> tuple[str, str]:
    path = unit_path.expanduser().resolve(strict=True)
    text = path.read_text(encoding="utf-8")
    match = re.search(r"^unit_id:\s*(\S+)\s*$", text, flags=re.MULTILINE)
    if match is None or match.group(1) != UNIT_ID:
        raise ValueError(f"unit does not identify {UNIT_ID}: {path}")
    return str(path), _sha256_file(path)


def _tree_digest(root: Path) -> str:
    """Digest receipt inputs, excluding transient log files."""

    root = root.expanduser().resolve(strict=True)
    paths = [root] if root.is_file() else sorted(root.rglob("*.json"))
    records = []
    for path in paths:
        if path.is_file():
            rel = path.name if root.is_file() else str(path.relative_to(root))
            records.append((rel, _sha256_file(path)))
    return _sha256_json(records)


def _load_variants(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    sampled_counts: Counter[str] = Counter()
    sampled_rows = 0
    with path.expanduser().resolve(strict=True).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"variant row {line_no} is not an object")
            match = value.get("ledger_match") if isinstance(value.get("ledger_match"), Mapping) else {}
            owner_id = value.get("owner_id") or match.get("owner_id")
            row_hash = value.get("row_token_ids_sha256")
            if not owner_id or not row_hash:
                continue
            # The discovery manifest is an append-only per-sample file.  It
            # starts with a canonical greedy first row and then contains the
            # sampled discovery draws.  Greedy is not an exact sampled branch
            # variant and must not enter confirmation completeness or
            # discovery-frequency weighting.
            decode_mode = str(value.get("decode_mode") or "")
            sampling_seed = value.get("sampling_seed")
            is_sampled = decode_mode == "sampled" or sampling_seed is not None
            if not is_sampled:
                continue
            row = dict(value)
            row["owner_id"] = str(owner_id)
            row["row_token_ids_sha256"] = str(row_hash)
            result[str(row_hash)] = row
            sampled_counts[str(row_hash)] += 1
            sampled_rows += 1
    if not result:
        raise ValueError(f"no exact variants found in {path}")
    # Support counts are frequencies over discovery samples, not fields on
    # each per-sample row.  Preserve the exact decimal-free integer counts in
    # the in-memory contract for later owner weighting.
    for row_hash, row in result.items():
        row["discovery_support_count"] = int(sampled_counts[row_hash])
        row["discovery_denominator"] = int(sampled_rows)
    return result


def _load_admission(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if value.get("schema_version") != "native_sibling_first_row_owner_admission.v1":
        raise ValueError("unsupported owner-admission manifest schema")
    admitted = value.get("admitted_owner_ids")
    groups = value.get("owner_groups")
    if not isinstance(admitted, list) or not isinstance(groups, list):
        raise ValueError("owner-admission manifest lacks owner groups")
    return value


def _load_ledger(path: Path) -> list[dict[str, Any]]:
    return [dict(row) for row in read_accepted_ledger(path)]


def _load_receipt_paths(roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for item in roots:
        path = item.expanduser().resolve(strict=True)
        if path.is_file():
            if path.name != "receipt.json":
                raise ValueError(f"receipt root file must be receipt.json: {path}")
            paths.add(path)
        else:
            paths.update(path.rglob("receipt.json"))
    return sorted(paths, key=str)


def _expected_parent_prefix_hash(donor_bundle: Path, prefix_count: int) -> str:
    from scripts.research.run_native_sibling_branch_replay import reconstruct_donor_prefix

    donor = _read_json(donor_bundle)
    return str(reconstruct_donor_prefix(donor, int(prefix_count))["donor_prefix_token_ids_sha256"])


def _actual_model_parameter_dtype_names(contract: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the attested model-parameter dtype names from a receipt.

    The robustness replay is only meaningful when the receipt records the
    dtypes that were actually present on the loaded model.  Keep this check at
    the analyzer boundary so a mislabeled runtime cannot silently enter a
    scientific comparison.
    """

    actual = contract.get("actual_model_parameter_dtypes")
    if not isinstance(actual, Mapping):
        raise ValueError("Wave-Two receipt lacks actual model parameter dtype evidence")
    names = actual.get("parameter_dtype_names")
    if not isinstance(names, list) or not names or any(not isinstance(item, str) for item in names):
        raise ValueError("Wave-Two receipt has malformed actual model parameter dtype evidence")
    return tuple(sorted(str(item) for item in names))


def _execution_contract_ok(
    contract: Mapping[str, Any],
    *,
    policy_mode: str = "sampled",
    expected_runtime_dtype: str = "config",
) -> None:
    """Validate one receipt against the explicitly selected runtime lineage.

    ``config`` is the source-consistent Brain Floating Point 16-bit path.
    ``fp32`` is a narrow, opt-in full-model 32-bit floating-point robustness
    replay.  The latter must retain the Brain Floating Point 16-bit model
    configuration while attesting that every actual model parameter is
    32-bit floating point.  This prevents an accidental mixed root from being
    compared with the canonical source-consistent panel.
    """

    if expected_runtime_dtype not in {"config", "fp32"}:
        raise ValueError(f"unsupported expected runtime dtype: {expected_runtime_dtype}")
    if int(contract.get("physical_batch_size", 0)) != 1:
        raise ValueError("Wave-Two receipt is not physical batch size one")
    actual_names = _actual_model_parameter_dtype_names(contract)
    if str(contract.get("model_config_dtype")) != "bf16":
        raise ValueError("Wave-Two receipt model configuration dtype is not Brain Floating Point 16-bit")
    runtime_mode = str(contract.get("runtime_dtype_mode"))
    if expected_runtime_dtype == "config":
        if runtime_mode != "config":
            raise ValueError("Wave-Two receipt runtime dtype is not the expected source-consistent config mode")
        if "torch.bfloat16" not in actual_names:
            raise ValueError("source-consistent receipt lacks actual Brain Floating Point 16-bit parameters")
    else:
        if runtime_mode != "fp32":
            raise ValueError("Wave-Two receipt runtime dtype is not the explicitly requested full-model fp32 mode")
        if actual_names != ("torch.float32",):
            raise ValueError("fp32 robustness receipt does not attest all model parameters as torch.float32")
    if str(contract.get("attention_implementation")) != "sdpa":
        raise ValueError("Wave-Two receipt does not use Scaled Dot-Product Attention")
    policy = contract.get("decode_generation_policy")
    if not isinstance(policy, Mapping) or str(policy.get("mode")) != policy_mode:
        raise ValueError("Wave-Two receipt has an unexpected decode policy")
    if policy_mode == "sampled":
        if float(policy.get("temperature", -1)) != 0.4 or float(policy.get("top_p", -1)) != 0.95:
            raise ValueError("Wave-Two sampling policy drifted")
    if float(policy.get("repetition_penalty", -1)) != 1.0:
        raise ValueError("Wave-Two repetition penalty drifted")


def _validate_execution_contract_lineage(
    contracts: Sequence[Mapping[str, Any]],
    *,
    expected_runtime_dtype: str = "config",
    policy_mode: str = "sampled",
) -> str:
    """Validate and fingerprint a homogeneous receipt-contract panel."""

    if not contracts:
        raise ValueError("Wave-Two receipt execution contract panel is empty")
    fingerprints: set[str] = set()
    for contract in contracts:
        _execution_contract_ok(
            contract,
            policy_mode=policy_mode,
            expected_runtime_dtype=expected_runtime_dtype,
        )
        fingerprints.add(_sha256_json(_scientific_contract(contract)))
    if len(fingerprints) != 1:
        raise ValueError("Wave-Two receipt execution contracts disagree")
    return next(iter(fingerprints))


def _scientific_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must be shared across sibling variants.

    Prompt and branch-source digests are intentionally variant-specific.  The
    model, numerical, attention, and decode-policy lineage must be shared.
    """

    excluded = {"branch_source_donor_sha256", "prompt_token_ids_sha256"}
    return {str(key): value for key, value in contract.items() if str(key) not in excluded}


def _bundle_path(call: Mapping[str, Any], receipt_path: Path) -> Path:
    raw = call.get("bundle_path")
    if not raw:
        raise ValueError(f"call in {receipt_path} lacks bundle_path")
    return _resolve(str(raw), base=receipt_path.parent)


def _receipt_branch_key(receipt: Mapping[str, Any]) -> str:
    donor = receipt.get("donor")
    branch = donor.get("branch_row") if isinstance(donor, Mapping) else None
    key = branch.get("token_ids_sha256") if isinstance(branch, Mapping) else None
    if not key:
        raise ValueError("Wave-Two receipt lacks exact branch row hash")
    return str(key)


def _validate_call_bundle_identity(
    receipt: Mapping[str, Any],
    call: Mapping[str, Any],
    bundle: Mapping[str, Any],
    *,
    receipt_branch_hash: str,
    seed: int,
    receipt_path: Path,
) -> None:
    """Reject stale or cross-linked call bundles before parsing predictions."""

    if str(bundle.get("image_id")) != str(receipt.get("image_id")):
        raise ValueError(f"call bundle image identity mismatch: {receipt_path}")
    if int(bundle.get("sampling_seed", -1)) != int(seed):
        raise ValueError("call bundle seed disagrees with receipt")
    request_id = str(call.get("request_id") or "")
    if not request_id or str(bundle.get("request_id") or "") != request_id:
        raise ValueError("call bundle request identifier disagrees with receipt")
    attested = bundle.get("executed_call_attestation")
    if not isinstance(attested, Mapping) or str(attested.get("request_id") or "") != request_id:
        raise ValueError("call bundle attestation request identifier disagrees with receipt")
    donor = bundle.get("donor")
    branch = donor.get("branch_row") if isinstance(donor, Mapping) else None
    if not isinstance(branch, Mapping) or str(branch.get("token_ids_sha256") or "") != receipt_branch_hash:
        raise ValueError("call bundle branch hash disagrees with receipt")
    prompt = bundle.get("prompt")
    receipt_prompt = receipt.get("prompt")
    if not isinstance(prompt, Mapping) or not isinstance(receipt_prompt, Mapping):
        raise ValueError("call bundle prompt identity is absent")
    if str(prompt.get("prompt_token_ids_sha256") or "") != str(receipt_prompt.get("prompt_token_ids_sha256") or ""):
        raise ValueError("call bundle prompt hash disagrees with receipt")
    runtime = bundle.get("runtime")
    contract = receipt.get("execution_contract")
    if not isinstance(runtime, Mapping) or not isinstance(contract, Mapping):
        raise ValueError("call bundle runtime or receipt contract is absent")
    expected_policy = contract.get("decode_generation_policy")
    actual_policy = runtime.get("decode_generation_policy")
    if _sha256_json(expected_policy) != _sha256_json(actual_policy):
        raise ValueError("call bundle decode policy disagrees with receipt contract")
    if str(runtime.get("decode_mode") or "") != str(call.get("decode_mode") or ""):
        raise ValueError("call bundle decode mode disagrees with receipt")


def _call_prediction_trace(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Recover rows without silently compressing a malformed chronology.

    ``parse_result.predictions`` is a convenience view and may omit malformed
    spans.  For this unit, an omitted first action is an invalid row-zero
    event, not permission to promote a later parsed row to row zero.  The
    raw token trajectory and complete-row spans therefore remain the
    chronology authority.
    """

    parse = bundle.get("parse_result")
    if not isinstance(parse, Mapping):
        return {"rows": [], "chronology_valid": False, "chronology_reason": "missing_parse_result"}
    value = parse.get("predictions")
    if not isinstance(value, list):
        return {"rows": [], "chronology_valid": False, "chronology_reason": "missing_predictions"}
    rows = [dict(item) for item in value if isinstance(item, Mapping)]
    rows.sort(key=lambda item: int(item.get("generated_order", 10**9)))
    raw = bundle.get("raw_generated_token_ids")
    attested = bundle.get("executed_call_attestation")
    if not isinstance(raw, list) and isinstance(attested, Mapping):
        raw = attested.get("raw_generated_token_ids")
    complete_spans = bundle.get("complete_row_spans")
    drop_count = int(parse.get("dropped_prediction_count", 0) or 0)
    parser_status = str(bundle.get("parser_status") or parse.get("parser_status") or "")
    reasons: list[str] = []
    if drop_count:
        reasons.append("parser_dropped_prediction")
    status_lower = parser_status.lower()
    if any(token in status_lower for token in ("dropped", "gap", "malformed", "invalid")) and "optional_drops" not in status_lower:
        reasons.append(f"parser_status:{parser_status}")
    try:
        from src.analysis.sampled_rescue_transition.comparison import trajectory_rows

        raw_rows = trajectory_rows([int(token) for token in raw]) if isinstance(raw, list) else []
    except Exception as exc:  # pragma: no cover - defensive artifact boundary
        raw_rows = []
        reasons.append(f"raw_trajectory_parse:{type(exc).__name__}")
    if not raw_rows and rows:
        reasons.append("raw_trajectory_missing")
    if isinstance(complete_spans, list) and raw_rows and len(complete_spans) != len(raw_rows):
        reasons.append("complete_row_span_count_mismatch")
    if raw_rows and len(rows) != len(raw_rows):
        reasons.append("parsed_row_count_mismatch")
    orders = [int(row.get("generated_order", -1)) for row in rows]
    if orders and orders != list(range(len(orders))):
        reasons.append("generated_order_gap")
    chronology_valid = not reasons
    return {
        "rows": rows[:HORIZON] if chronology_valid else [],
        "chronology_valid": chronology_valid,
        "chronology_reason": ";".join(reasons) if reasons else "raw_and_parsed_rows_aligned",
        "raw_row_count": len(raw_rows),
        "parsed_row_count": len(rows),
    }


def _call_prediction_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compatibility wrapper returning only valid chronological rows."""

    return list(_call_prediction_trace(bundle)["rows"])


def _receipt_records(
    roots: Sequence[Path],
    *,
    variants: Mapping[str, Mapping[str, Any]],
    donor_prefix_hash: str,
    ledger_digest: str,
    expected_runtime_dtype: str = "config",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate every variant receipt and every paired call."""

    receipt_paths = _load_receipt_paths(roots)
    if not receipt_paths:
        raise ValueError("no Wave-Two receipt.json files found")
    expected_seeds: tuple[int, ...] | None = None
    records: list[dict[str, Any]] = []
    contracts: list[Mapping[str, Any]] = []
    for receipt_path in receipt_paths:
        receipt = _read_json(receipt_path)
        if receipt.get("schema_version") != "native_sibling_branch_replay.shard_receipt.v1":
            raise ValueError(f"unsupported Wave-Two receipt schema: {receipt_path}")
        contract = receipt.get("execution_contract")
        if not isinstance(contract, Mapping):
            raise ValueError(f"receipt lacks execution contract: {receipt_path}")
        _execution_contract_ok(contract, expected_runtime_dtype=expected_runtime_dtype)
        contracts.append(contract)
        if receipt.get("ledger_digest") not in {None, ledger_digest}:
            raise ValueError("Wave-Two receipt ledger digest disagrees with supplied ledger")
        donor = receipt.get("donor")
        prompt = receipt.get("prompt")
        if not isinstance(donor, Mapping) or str(donor.get("parent_prefix_token_ids_sha256")) != donor_prefix_hash:
            raise ValueError("Wave-Two parent prefix hash mismatch")
        branch_lineage = donor.get("branch_lineage")
        if not isinstance(branch_lineage, Mapping) or str(branch_lineage.get("image_id")) != str(receipt.get("image_id")):
            raise ValueError("Wave-Two receipt image identity mismatch")
        if not isinstance(prompt, Mapping):
            raise ValueError("Wave-Two receipt lacks prompt evidence")
        branch_hash = _receipt_branch_key(receipt)
        variant = variants.get(branch_hash)
        if variant is None:
            raise ValueError(f"Wave-Two branch variant is absent from frozen discovery variants: {branch_hash}")
        owner_id = str(variant["owner_id"])
        seeds = receipt.get("sampling_seeds")
        if not isinstance(seeds, list) or len(seeds) != EXPECTED_CONFIRMATION_SEEDS:
            raise ValueError(f"variant {branch_hash} does not have exactly eight confirmation seeds")
        seeds_int = tuple(int(seed) for seed in seeds)
        if len(set(seeds_int)) != len(seeds_int):
            raise ValueError("duplicate confirmation seed in a variant receipt")
        identity = receipt.get("seed_schedule_identity")
        if not isinstance(identity, Mapping) or int(identity.get("seed_root", -1)) != CONFIRMATION_SEED_ROOT:
            raise ValueError("confirmation seed root drifted")
        if expected_seeds is None:
            expected_seeds = seeds_int
        elif seeds_int != expected_seeds:
            raise ValueError("paired confirmation seed vector differs across exact variants")
        calls = receipt.get("calls")
        if not isinstance(calls, list) or len(calls) != EXPECTED_CONFIRMATION_SEEDS:
            raise ValueError("variant receipt call count is not eight")
        by_seed: dict[int, dict[str, Any]] = {}
        for call in calls:
            if not isinstance(call, Mapping) or call.get("decode_mode") != "sampled":
                raise ValueError("Wave-Two owner analysis requires sampled calls only")
            seed = int(call.get("sampling_seed"))
            if seed in by_seed:
                raise ValueError("duplicate call seed within a variant")
            bundle_path = _bundle_path(call, receipt_path)
            bundle = _read_json(bundle_path)
            if bundle.get("schema_version") != "native_sibling_branch_replay.call_bundle.v1":
                raise ValueError(f"unsupported call-bundle schema: {bundle_path}")
            if int(bundle.get("executed_call_attestation", {}).get("physical_batch_size", 0)) != 1:
                raise ValueError("call bundle is not physical batch size one")
            _validate_call_bundle_identity(
                receipt,
                call,
                bundle,
                receipt_branch_hash=branch_hash,
                seed=seed,
                receipt_path=receipt_path,
            )
            runtime = bundle.get("runtime")
            if not isinstance(runtime, Mapping) or runtime.get("decode_mode") != "sampled":
                raise ValueError("call bundle sampled runtime evidence is absent")
            trace = _call_prediction_trace(bundle)
            by_seed[seed] = {
                "seed": str(seed),
                "bundle_path": str(bundle_path),
                "bundle_sha256": _sha256_file(bundle_path),
                "rows": trace["rows"],
                "chronology_valid": bool(trace["chronology_valid"]),
                "chronology_reason": str(trace["chronology_reason"]),
                "raw_row_count": int(trace.get("raw_row_count", 0)),
                "parsed_row_count": int(trace.get("parsed_row_count", 0)),
                "termination": bundle.get("horizon_projection", {}),
                "request_id": str(bundle.get("request_id") or call.get("request_id") or ""),
            }
        if set(by_seed) != set(seeds_int):
            raise ValueError("variant call seeds do not exactly match receipt seed vector")
        records.append(
            {
                "receipt_path": str(receipt_path),
                "receipt_sha256": _sha256_file(receipt_path),
                "receipt_root": str(next((root for root in roots if receipt_path.is_relative_to(root.resolve()) if root.resolve().is_dir()), receipt_path.parent)),
                "branch_hash": branch_hash,
                "owner_id": owner_id,
                "discovery_support_count": int(
                    variant.get("discovery_support_count", variant.get("support_count", 0))
                ),
                "discovery_denominator": int(
                    variant.get("discovery_denominator", variant.get("support_denominator", 32))
                ),
                "variant": dict(variant),
                "calls_by_seed": by_seed,
            }
        )
    execution_contract_sha256 = _validate_execution_contract_lineage(
        contracts,
        expected_runtime_dtype=expected_runtime_dtype,
    )
    if expected_seeds is None:
        raise ValueError("no confirmation calls found")
    return records, {
        "receipt_paths": [item["receipt_path"] for item in records],
        "receipt_root_digests": {str(root.resolve()): _tree_digest(root) for root in roots},
        "confirmation_seeds": [str(seed) for seed in expected_seeds],
        "execution_contract_sha256": execution_contract_sha256,
        "expected_runtime_dtype": expected_runtime_dtype,
    }


def _parent_coverage_details(
    donor_bundle: Path,
    *,
    prefix_count: int,
    image_id: str,
    ledger: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    donor = _read_json(donor_bundle)
    decode = donor.get("decode_result")
    if not isinstance(decode, Mapping):
        raise ValueError("donor lacks decode_result")
    generated = decode.get("generated_token_ids")
    scores = donor.get("parse_score_receipts")
    if not isinstance(generated, list) or not isinstance(scores, list):
        raise ValueError("donor lacks token IDs or canonical parse score receipts")
    from src.analysis.sampled_rescue_transition.comparison import trajectory_rows

    spans = trajectory_rows([int(token) for token in generated[:prefix_count]])
    covered: set[str] = set()
    unresolved: list[dict[str, Any]] = []
    for row_index, span in enumerate(spans):
        score = next((item for item in scores if isinstance(item, Mapping) and int(item.get("generated_row_index", -1)) == row_index), None)
        if not isinstance(score, Mapping) or score.get("parse_status") != "accepted":
            unresolved.append({
                "image_id": image_id,
                "source": "parent_prefix",
                "row_index": row_index,
                "prediction": dict(score.get("prediction") or {}) if isinstance(score, Mapping) and isinstance(score.get("prediction"), Mapping) else {},
                "reason": "missing_or_unaccepted_parent_score",
            })
            continue
        prediction = {"description": score.get("normalized_category_name"), "bbox": score.get("parsed_bbox_xyxy")}
        match = match_first_row_to_ledger(prediction=prediction, image_id=image_id, ledger=ledger)
        if match.get("status") == "unique" and match.get("owner_id"):
            covered.add(str(match["owner_id"]))
        else:
            unresolved.append({
                "image_id": image_id,
                "source": "parent_prefix",
                "row_index": row_index,
                "prediction": prediction,
                "reason": str(match.get("status") or "unresolved_parent_match"),
                "match": dict(match),
            })
    return {"covered_owner_ids": covered, "unresolved_rows": unresolved, "prefix_row_count": len(spans)}


def _parent_covered_owners(
    donor_bundle: Path,
    *,
    prefix_count: int,
    image_id: str,
    ledger: Sequence[Mapping[str, Any]],
) -> set[str]:
    """Compatibility helper; strict analysis also consumes unresolved rows."""

    return set(_parent_coverage_details(donor_bundle, prefix_count=prefix_count, image_id=image_id, ledger=ledger)["covered_owner_ids"])


def _resolve_greedy_control(
    admission: Mapping[str, Any],
    *,
    image_id: str,
    ledger: Sequence[Mapping[str, Any]],
    parent_covered: set[str],
    candidates: Sequence[Mapping[str, Any]] = (),
    frozen_review: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    controls = admission.get("greedy_control")
    admitted = {str(item) for item in admission.get("admitted_owner_ids", [])}
    if not isinstance(controls, list) or len(controls) != 1:
        return {"status": "refused", "reason": "greedy_control_is_not_singleton"}
    control = controls[0]
    if not isinstance(control, Mapping):
        return {"status": "refused", "reason": "greedy_control_is_not_mapping"}
    advertised_owner = str(control.get("owner_id") or "")
    # Admission metadata may intentionally leave the greedy row unresolved.
    # In that case the blinded review is the first authority; only its stable
    # entity reference is subsequently checked against admitted owners.
    metadata_requires_review = (
        str(control.get("ledger_status") or "") != "unique"
        or not bool(control.get("owner_admitted"))
        or not advertised_owner
    )
    owner = advertised_owner
    source = control.get("source_bundle")
    if not source:
        return {"status": "refused", "reason": "greedy_control_source_bundle_missing", "owner_id": owner}
    try:
        bundle_path = Path(str(source)).expanduser().resolve(strict=True)
        bundle = _read_json(bundle_path)
        predictions = bundle.get("parse_result", {}).get("predictions") if isinstance(bundle.get("parse_result"), Mapping) else None
        first = predictions[0] if isinstance(predictions, list) and predictions and isinstance(predictions[0], Mapping) else None
        if str(bundle.get("image_id")) != str(image_id) or first is None:
            raise ValueError("greedy source bundle image or first row is missing")
        advertised_request = str(control.get("request_id") or "")
        if advertised_request and advertised_request != str(bundle.get("request_id") or ""):
            raise ValueError("greedy source bundle request identifier disagrees with admission")
        physical = _match_physical(first, image_id=image_id, ledger=ledger, candidates=candidates, frozen_review=frozen_review)
        if metadata_requires_review:
            candidate = _candidate_for_prediction(first, candidates, image_id)
            decision = (frozen_review or {}).get("mapping", {}).get(str(candidate.get("candidate_id"))) if candidate is not None else None
            if not isinstance(decision, Mapping) or decision.get("verdict") != "approve" or not str(decision.get("entity_ref") or "").strip():
                raise ValueError("unresolved greedy source row lacks frozen approval")
            owner = str(decision["entity_ref"])
            physical = {**physical, "physical_status": "review_approved", "owner_id": owner, "candidate_id": candidate["candidate_id"]}
        elif physical.get("physical_status") not in {"unique", "review_approved"}:
            raise ValueError("greedy source bundle lacks unique physical ownership")
        if advertised_owner and str(owner) != advertised_owner:
            raise ValueError("greedy source bundle owner disagrees with admission or frozen review")
        if owner not in admitted:
            raise ValueError("greedy source owner is not an admitted owner")
        if owner in parent_covered:
            return {"status": "refused", "reason": "greedy_owner_parent_covered", "owner_id": owner}
        return {
            "status": "validated",
            "owner_id": owner,
            "request_id": str(control.get("request_id") or bundle.get("request_id") or ""),
            "source_bundle": str(bundle_path),
            "source_bundle_sha256": _sha256_file(bundle_path),
            "physical_status": str(physical.get("physical_status")),
        }
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {"status": "refused", "reason": f"greedy_control_source_invalid:{exc}", "owner_id": owner}


def _greedy_control_review_rows(
    admission: Mapping[str, Any],
    *,
    image_id: str,
    ledger: Sequence[Mapping[str, Any]],
    parent_covered: set[str],
) -> list[dict[str, Any]]:
    """Return a blinded greedy row only when automatic identity is unresolved."""

    controls = admission.get("greedy_control")
    if not isinstance(controls, list) or len(controls) != 1 or not isinstance(controls[0], Mapping):
        return []
    control = controls[0]
    source = control.get("source_bundle")
    if not source:
        return []
    try:
        bundle = _read_json(Path(str(source)).expanduser().resolve(strict=True))
        predictions = bundle.get("parse_result", {}).get("predictions") if isinstance(bundle.get("parse_result"), Mapping) else None
        first = predictions[0] if isinstance(predictions, list) and predictions and isinstance(predictions[0], Mapping) else None
        if first is None:
            return []
        match = match_first_row_to_ledger(prediction=first, image_id=image_id, ledger=ledger)
        metadata_unresolved = (
            str(control.get("ledger_status") or "") != "unique"
            or not bool(control.get("owner_admitted"))
            or not str(control.get("owner_id") or "")
        )
        if not metadata_unresolved and match.get("status") == "unique" and str(match.get("owner_id")) not in parent_covered:
            return []
        return [{
            "image_id": image_id,
            "prediction": dict(first),
            "source": "greedy_control",
            "row_index": 0,
            "physical_match_status": str(match.get("status")),
            "metadata_unresolved": metadata_unresolved,
        }]
    except (OSError, ValueError, json.JSONDecodeError):
        return []


def _review_candidate_key(image_id: str, category: str, bbox: Sequence[Any]) -> str:
    normalized = [round(float(value), 3) for value in bbox]
    return "candidate:" + _sha256_json([str(image_id), str(category).strip().lower(), normalized])[:24]


def _prediction_signature(prediction: Mapping[str, Any]) -> str:
    """Stable physical-review signature independent of seed/request provenance."""

    category = str(prediction.get("description", "")).strip().lower()
    bbox = prediction.get("bbox")
    if not isinstance(bbox, Sequence) or len(bbox) != 4:
        return "prediction:" + _sha256_json([category, None])
    return "prediction:" + _sha256_json([category, [round(float(value), 3) for value in bbox]])


def _unmatched_rows(records: Sequence[Mapping[str, Any]], *, ledger: Sequence[Mapping[str, Any]], image_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        for seed, call in record["calls_by_seed"].items():
            for index, prediction in enumerate(call.get("rows", [])):
                if not isinstance(prediction, Mapping):
                    continue
                match = match_first_row_to_ledger(prediction=prediction, image_id=image_id, ledger=ledger)
                if match.get("status") in {"unmatched", "ambiguous"} and isinstance(prediction.get("bbox"), Sequence):
                    rows.append({
                        "image_id": image_id,
                        "prediction": dict(prediction),
                        "seed": str(seed),
                        "record": record,
                        "row_index": index,
                        "physical_match_status": str(match.get("status")),
                    })
    return rows


def _candidate_clusters(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Cluster only same-category unmatched boxes; no owner matching here."""

    # Connected components preserve replay membership for bridge chains such
    # as A~B, B~C, A!~C.  Assigning against only the first representative
    # would make a later replay move between review candidates.
    valid_rows = [row for row in rows if isinstance(row.get("prediction"), Mapping)]
    parent = list(range(len(valid_rows)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left, root_right = find(left), find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, left_row in enumerate(valid_rows):
        left_prediction = left_row["prediction"]
        left_category = str(left_prediction.get("description", "")).strip().lower()
        for right in range(left + 1, len(valid_rows)):
            right_prediction = valid_rows[right]["prediction"]
            right_category = str(right_prediction.get("description", "")).strip().lower()
            if left_category == right_category and box_iou(left_prediction.get("bbox"), right_prediction.get("bbox")) >= 0.50:
                union(left, right)
    grouped_indices: dict[int, list[int]] = defaultdict(list)
    for index in range(len(valid_rows)):
        grouped_indices[find(index)].append(index)
    groups = [[valid_rows[index] for index in indices] for indices in grouped_indices.values()]
    result: list[dict[str, Any]] = []
    for group in groups:
        sorted_rows = sorted(group, key=lambda row: (str(row["prediction"].get("description", "")), json.dumps(row["prediction"].get("bbox"))))
        representative = sorted_rows[0]["prediction"]
        category = str(representative.get("description", "")).strip().lower()
        bbox = [float(value) for value in representative["bbox"]]
        result.append(
            {
                "candidate_id": _review_candidate_key(str(group[0]["image_id"]), category, bbox),
                "image_id": str(group[0]["image_id"]),
                "category": category,
                "bbox_xyxy": bbox,
                "support_count": len(group),
                "cluster_geometry_rule": "same_category_and_iou_at_least_0.50",
                "member_prediction_signatures": sorted({_prediction_signature(row["prediction"]) for row in group}),
            }
        )
    return sorted(result, key=lambda item: (item["image_id"], item["category"], item["candidate_id"]))


def _load_review(path: Path) -> list[dict[str, Any]]:
    value = _read_json(path)
    candidates = value.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("review file lacks candidates")
    result = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("review candidate is not an object")
        verdict = str(candidate.get("verdict") or candidate.get("decision") or "").strip().lower()
        if verdict not in SUPPORTED_REVIEW_VERDICTS:
            raise ValueError(f"review candidate has invalid verdict: {verdict}")
        result.append({**dict(candidate), "verdict": verdict})
    return result


def _freeze_review(candidates: Sequence[Mapping[str, Any]], review: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {str(item["candidate_id"]): dict(item) for item in candidates}
    observed: dict[str, Mapping[str, Any]] = {}
    for item in review:
        key = str(item.get("candidate_id") or "")
        if key in observed or key not in expected:
            raise ValueError(f"review candidate ID is duplicate or not in current blind set: {key}")
        observed[key] = item
        if item["verdict"] == "approve" and not str(item.get("entity_ref") or "").strip():
            raise ValueError(f"approved review candidate lacks stable entity_ref: {key}")
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise ValueError(f"review is incomplete; missing candidate IDs: {missing[:8]}")
    mapping = {}
    for key, item in sorted(observed.items()):
        mapping[key] = {
            "verdict": item["verdict"],
            "entity_ref": str(item.get("entity_ref") or "") if item["verdict"] == "approve" else None,
            "comment": str(item.get("comment") or ""),
        }
    return {"schema_version": f"{SCHEMA_VERSION}.frozen_review.v1", "candidate_count": len(mapping), "mapping": mapping}


def _candidate_for_prediction(prediction: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]], image_id: str) -> Mapping[str, Any] | None:
    signature = _prediction_signature(prediction)
    exact = [
        candidate
        for candidate in candidates
        if str(candidate.get("image_id")) == str(image_id)
        and signature in set(str(item) for item in candidate.get("member_prediction_signatures", []))
    ]
    if exact:
        return sorted(exact, key=lambda item: str(item.get("candidate_id")))[0]
    matches = [
        candidate
        for candidate in candidates
        if str(candidate.get("image_id")) == str(image_id)
        and str(candidate.get("category", "")).strip().lower() == str(prediction.get("description", "")).strip().lower()
        and box_iou(candidate.get("bbox_xyxy"), prediction.get("bbox")) >= 0.50
    ]
    return sorted(matches, key=lambda item: (float(-box_iou(item.get("bbox_xyxy"), prediction.get("bbox"))), str(item.get("candidate_id"))))[0] if matches else None


def _match_physical(
    prediction: Mapping[str, Any],
    *,
    image_id: str,
    ledger: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    frozen_review: Mapping[str, Any] | None,
) -> dict[str, Any]:
    match = match_first_row_to_ledger(prediction=prediction, image_id=image_id, ledger=ledger, iou_floor=DEFAULT_IOU_FLOOR, ambiguity_margin=DEFAULT_AMBIGUITY_MARGIN)
    if match.get("status") == "unique":
        return {"physical_status": "unique", "owner_id": str(match["owner_id"]), "match": match}
    candidate = _candidate_for_prediction(prediction, candidates, image_id)
    candidate_id = candidate.get("candidate_id") if candidate is not None else None
    if candidate is None or frozen_review is None:
        return {"physical_status": "ambiguous" if match.get("status") == "ambiguous" else "unresolved", "owner_id": None, "candidate_id": candidate_id, "match": match}
    decision = frozen_review.get("mapping", {}).get(str(candidate["candidate_id"]))
    if not isinstance(decision, Mapping):
        return {"physical_status": "ambiguous" if match.get("status") == "ambiguous" else "unresolved", "owner_id": None, "candidate_id": candidate_id, "match": match}
    verdict = decision.get("verdict")
    if verdict == "approve":
        return {"physical_status": "review_approved", "owner_id": str(decision.get("entity_ref")), "candidate_id": candidate["candidate_id"], "match": match}
    if verdict == "reject":
        return {"physical_status": "review_rejected", "owner_id": None, "candidate_id": candidate["candidate_id"], "match": match}
    return {"physical_status": "review_unknown", "owner_id": None, "candidate_id": candidate["candidate_id"], "match": match}


def _classify_path(
    record: Mapping[str, Any],
    *,
    image_id: str,
    parent_covered: set[str],
    ledger: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    frozen_review: Mapping[str, Any],
    discovered_owner_ids: set[str],
    accepted_owner_ids: set[str] | None = None,
) -> dict[str, Any]:
    branch_owner = str(record["owner_id"])
    outcomes: list[dict[str, Any]] = [{"classification": "new_supported_predeclared", "owner_id": branch_owner, "source": "branch_row", "row_index": -1}]
    call = record["call"]
    if not bool(call.get("chronology_valid", True)):
        outcomes.append({
            "classification": "invalid",
            "owner_id": None,
            "row_index": 0,
            "reason": f"chronology:{call.get('chronology_reason', 'invalid')}",
        })
        return {"branch_hash": record["branch_hash"], "owner_id": branch_owner, "seed": str(call["seed"]), "outcomes": outcomes, "chronology_valid": False}
    seen_suffix: set[str] = set()
    for index, prediction in enumerate(call["rows"]):
        if not isinstance(prediction, Mapping):
            outcomes.append({"classification": "invalid", "owner_id": None, "row_index": index, "reason": "non_object_prediction"})
            continue
        if not isinstance(prediction.get("bbox"), Sequence) or len(prediction.get("bbox", [])) != 4:
            outcomes.append({"classification": "invalid", "owner_id": None, "row_index": index, "reason": "malformed_bbox"})
            continue
        physical = _match_physical(prediction, image_id=image_id, ledger=ledger, candidates=candidates, frozen_review=frozen_review)
        if physical["physical_status"] in {"ambiguous", "review_unknown", "unresolved"}:
            classification = "unknown"
            owner = None
            unknown_id = str(physical.get("candidate_id") or _review_candidate_key(image_id, str(prediction.get("description", "")), prediction.get("bbox", [])))
        elif physical["physical_status"] == "review_rejected":
            classification = "unsupported"
            owner = None
            unknown_id = None
        else:
            owner = str(physical["owner_id"])
            unknown_id = None
            if owner == branch_owner:
                classification = "duplicate_current"
            elif owner in seen_suffix:
                classification = "duplicate_within_suffix"
            elif owner in parent_covered:
                classification = "duplicate_parent_covered"
            elif owner in (accepted_owner_ids or discovered_owner_ids):
                # Review provenance does not change accepted-ledger
                # membership: an ambiguous row approved to an accepted owner
                # is still predeclared.  Only a truly absent-ledger human
                # entity is not-predeclared.
                classification = "new_supported_predeclared"
            elif physical["physical_status"] == "review_approved":
                classification = "new_supported_not_predeclared"
            elif owner in discovered_owner_ids:
                classification = "new_supported_predeclared"
            else:
                classification = "new_supported_not_predeclared"
        outcome = {"classification": classification, "owner_id": owner, "row_index": index, "prediction": dict(prediction), "physical": physical}
        if unknown_id is not None:
            outcome["unknown_id"] = unknown_id
            outcome["candidate_id"] = physical.get("candidate_id")
        outcomes.append(outcome)
        if owner is not None:
            seen_suffix.add(owner)
    termination = record["call"].get("termination")
    if isinstance(termination, Mapping):
        term_class = str(termination.get("termination_classification") or "")
        if term_class == "natural_termination_before_horizon":
            outcomes.append({"classification": "terminal", "owner_id": None, "row_index": len(call["rows"])})
        elif term_class in {"invalid_or_malformed", "token_limit_truncated"}:
            outcomes.append({"classification": "invalid", "owner_id": None, "row_index": len(call["rows"]), "reason": term_class})
    return {"branch_hash": record["branch_hash"], "owner_id": branch_owner, "seed": str(call["seed"]), "outcomes": outcomes, "chronology_valid": True}


def _outcome_summary(outcomes: Sequence[Mapping[str, Any]], *, unknown_as_supported: bool) -> dict[str, float]:
    labels = [str(item.get("classification")) for item in outcomes]
    unknown_ids = {
        str(item.get("unknown_id") or item.get("candidate_id"))
        for item in outcomes
        if str(item.get("classification")) == "unknown" and (item.get("unknown_id") or item.get("candidate_id"))
    }
    predeclared_owner_ids = {
        str(item.get("owner_id"))
        for item in outcomes
        if item.get("owner_id") is not None
        and str(item.get("classification")) == "new_supported_predeclared"
    }
    not_predeclared_owner_ids = {
        str(item.get("owner_id"))
        for item in outcomes
        if item.get("owner_id") is not None
        and str(item.get("classification")) == "new_supported_not_predeclared"
    }
    # The lower world trusts only accepted-ledger/predeclared entities.  A
    # human-approved entity absent from that ledger is deliberately censored
    # in the lower world and admitted once in the upper world.
    supported = len(predeclared_owner_ids) + (
        len(not_predeclared_owner_ids) + len(unknown_ids) if unknown_as_supported else 0
    )
    first_suffix = next((item for item in outcomes if int(item.get("row_index", -1)) == 0), None)
    good = bool(first_suffix and first_suffix.get("classification") == "new_supported_predeclared")
    bad_labels = {"terminal", "invalid", "unsupported", "duplicate_current", "duplicate_parent_covered", "duplicate_within_suffix", "unknown"}
    bad = bool(first_suffix and str(first_suffix.get("classification")) in bad_labels)
    return {
        "new_supported_count": float(supported),
        "new_supported_predeclared_count": float(len({str(item.get("owner_id")) for item in outcomes if str(item.get("classification")) == "new_supported_predeclared" and item.get("owner_id") is not None})),
        "new_supported_not_predeclared_count": float(len({str(item.get("owner_id")) for item in outcomes if str(item.get("classification")) == "new_supported_not_predeclared" and item.get("owner_id") is not None})),
        "unknown_unique_count": float(len(unknown_ids)),
        "unsupported_event": float("unsupported" in labels),
        "invalid_event": float("invalid" in labels),
        "duplicate_event": float(any(label.startswith("duplicate_") for label in labels)),
        "unknown_event": float(bool(unknown_ids)),
        "terminal_event": float("terminal" in labels),
        "duplicate_current_event": float("duplicate_current" in labels),
        "duplicate_covered_event": float(any(label in {"duplicate_parent_covered", "duplicate_within_suffix"} for label in labels)),
        "good_event": float(good),
        "bad_event": float(bad),
    }


def _quantile(values: Sequence[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return float("nan")
    index = (len(ordered) - 1) * q
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _bootstrap_differences(
    rows_by_variant_owner_seed: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    owner_a: str,
    owner_b: str,
    *,
    metric: str,
    seed_order: Sequence[str],
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap paired seed-index contrasts, weighted over exact variants."""

    variants_a = rows_by_variant_owner_seed.get(owner_a, {})
    variants_b = rows_by_variant_owner_seed.get(owner_b, {})
    if not variants_a or not variants_b:
        raise ValueError("paired bootstrap requires both owner arms")
    # Discovery weights are applied by the caller through repeated variant
    # values in the maps.  Keeping this primitive equal-weighted makes tests
    # and the exact-row report transparent.
    diffs: list[float] = []
    import random

    rng = random.Random(BOOTSTRAP_SEED_ROOT)
    for _ in range(int(replicates)):
        sampled = [seed_order[rng.randrange(len(seed_order))] for _ in seed_order]
        vals_a = [sum(variants_a[v][seed][metric] for v in variants_a) / len(variants_a) for seed in sampled]
        vals_b = [sum(variants_b[v][seed][metric] for v in variants_b) / len(variants_b) for seed in sampled]
        diffs.append(sum(vals_a) / len(vals_a) - sum(vals_b) / len(vals_b))
    point = sum(
        sum(variants_a[v][seed][metric] for v in variants_a) / len(variants_a)
        - sum(variants_b[v][seed][metric] for v in variants_b) / len(variants_b)
        for seed in seed_order
    ) / len(seed_order)
    return {"point": point, "lower_95": _quantile(diffs, 0.025), "upper_95": _quantile(diffs, 0.975), "replicates": int(replicates), "bootstrap_seed_root": BOOTSTRAP_SEED_ROOT}


def _paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    seed_order: Sequence[str],
    replicates: int,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Return a paired seed-index interval for ``left - right``."""

    if not seed_order or set(left) != set(seed_order) or set(right) != set(seed_order):
        raise ValueError("paired bootstrap requires the same complete seed index set")
    observed = sum(float(left[seed]) - float(right[seed]) for seed in seed_order) / len(seed_order)
    import random

    rng = random.Random(BOOTSTRAP_SEED_ROOT + int(seed_offset))
    values: list[float] = []
    n = len(seed_order)
    for _ in range(int(replicates)):
        draw = [seed_order[rng.randrange(n)] for _ in range(n)]
        values.append(sum(float(left[seed]) - float(right[seed]) for seed in draw) / n)
    return {
        "point": observed,
        "lower_95": _quantile(values, 0.025),
        "upper_95": _quantile(values, 0.975),
        "replicates": int(replicates),
        "bootstrap_seed_root": BOOTSTRAP_SEED_ROOT,
        "paired_unit": "confirmation_seed_index",
    }


def _joint_bootstrap_intervals(
    series: Mapping[str, Mapping[str, float]],
    *,
    seed_order: Sequence[str],
    replicates: int,
    family: str,
    seed_offset: int = 0,
) -> dict[str, Any]:
    """Compute simultaneous non-studentized max-deviation intervals.

    Every metric in ``series`` is resampled from the same confirmation-seed
    index draw on every replicate.  The maximum absolute deviation across the
    family is then used as one common radius, so the returned intervals are
    genuinely simultaneous rather than pointwise intervals with a misleading
    label.
    """

    if not seed_order or not series:
        raise ValueError("joint bootstrap requires a non-empty metric family")
    expected = set(seed_order)
    if any(set(values) != expected for values in series.values()):
        raise ValueError("joint bootstrap requires complete paired seed vectors")
    point = {name: sum(float(values[seed]) for seed in seed_order) / len(seed_order) for name, values in series.items()}
    import random

    rng = random.Random(BOOTSTRAP_SEED_ROOT + int(seed_offset))
    max_deviations: list[float] = []
    n = len(seed_order)
    for _ in range(int(replicates)):
        draw = [seed_order[rng.randrange(n)] for _ in range(n)]
        sample = {name: sum(float(values[seed]) for seed in draw) / n for name, values in series.items()}
        max_deviations.append(max(abs(sample[name] - point[name]) for name in point))
    radius = _quantile(max_deviations, 0.95)
    return {
        "family": str(family),
        "interval_method": "non_studentized_max_deviation",
        "simultaneous": True,
        "confidence_level": 0.95,
        "metrics": {
            name: {"point": value, "lower_95": value - radius, "upper_95": value + radius}
            for name, value in point.items()
        },
        "max_deviation_95": radius,
        "replicates": int(replicates),
        "bootstrap_seed_root": BOOTSTRAP_SEED_ROOT + int(seed_offset),
        "paired_unit": "confirmation_seed_index",
    }


def _gate_status(*, lower_bound: float, threshold: float, direction: str) -> str:
    if direction == "greater":
        return "pass" if lower_bound > threshold else "fail"
    if direction == "less_equal":
        return "pass" if lower_bound <= threshold else "fail"
    raise ValueError(f"unknown gate direction: {direction}")


def _owner_completeness(
    expected_variants: set[str],
    observed_variants: set[str],
    confirmation_counts: Mapping[str, int],
) -> dict[str, Any]:
    """Return a strict owner completeness record, including absent owners."""

    complete = bool(expected_variants) and expected_variants == observed_variants and all(
        int(confirmation_counts.get(item, 0)) == EXPECTED_CONFIRMATION_SEEDS for item in expected_variants
    )
    return {
        "expected_exact_variant_count": len(expected_variants),
        "observed_exact_variant_count": len(observed_variants),
        "complete": complete,
        "missing_exact_variants": sorted(expected_variants - observed_variants),
        "unexpected_exact_variants": sorted(observed_variants - expected_variants),
    }


def _owner_seed_metrics(
    records: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    *,
    unknown_as_supported: bool,
    owner_ids: Sequence[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Discovery-frequency-weight path primitives by owner and seed."""

    weight_by_branch = {str(record["branch_hash"]): float(record["discovery_support_count"]) for record in records}
    grouped: dict[tuple[str, str], list[tuple[float, Mapping[str, float]]]] = defaultdict(list)
    for path in paths:
        summary = _outcome_summary(path["outcomes"], unknown_as_supported=unknown_as_supported)
        grouped[(str(path["owner_id"]), str(path["seed"]))].append((weight_by_branch[str(path["branch_hash"])], summary))
    result: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    all_owner_ids = sorted(set(str(item) for item in (owner_ids or ())) | {str(path.get("owner_id")) for path in paths} | {
        str(item.get("owner_id"))
        for path in paths
        for item in path.get("outcomes", [])
        if item.get("row_index", -1) == 0 and item.get("owner_id") is not None
    })
    for (owner, seed), items in grouped.items():
        total = sum(weight for weight, _ in items)
        if total <= 0:
            raise ValueError("non-positive discovery frequency weight")
        metrics = {}
        for metric in ("new_supported_count", "new_supported_predeclared_count", "new_supported_not_predeclared_count", "unknown_unique_count", "unsupported_event", "invalid_event", "duplicate_event", "duplicate_current_event", "duplicate_covered_event", "unknown_event", "terminal_event", "good_event", "bad_event"):
            metrics[metric] = sum(weight * float(summary[metric]) for weight, summary in items) / total
        first_rows = [
            (weight, path)
            for path in paths
            if str(path["owner_id"]) == owner and str(path["seed"]) == seed
            for weight in [weight_by_branch[str(path["branch_hash"])]]
        ]
        for target in all_owner_ids:
            weighted = 0.0
            for weight, path in first_rows:
                first = next((item for item in path["outcomes"] if int(item.get("row_index", -1)) == 0), None)
                weighted += weight * float(bool(first and first.get("owner_id") == target))
            metrics[f"next_owner:{target}"] = weighted / total
        result[owner][seed] = metrics
    return result


def _crossover_primitives(
    owner_seed: Mapping[str, Mapping[str, Mapping[str, float]]],
    owners: Sequence[str],
    *,
    seed_order: Sequence[str],
    replicates: int,
    upper_owner_seed: Mapping[str, Mapping[str, Mapping[str, float]]] | None = None,
) -> dict[str, Any]:
    from itertools import combinations

    if len(owners) < 2:
        return {"status": "not_identified", "reason": "fewer_than_two_admitted_owner_arms"}
    chosen = sorted({str(item) for item in owners})
    if any(owner not in owner_seed or any(seed not in owner_seed[owner] for seed in seed_order) for owner in chosen):
        return {"status": "refused", "reason": "incomplete_owner_seed_panel"}
    pairs: dict[str, Any] = {}
    for left, right in combinations(chosen, 2):
        contrast_series = {
            f"C_{left}": {
                seed: float(owner_seed[right][seed].get(f"next_owner:{left}", 0.0)) - float(owner_seed[left][seed].get(f"next_owner:{left}", 0.0))
                for seed in seed_order
            },
            f"C_{right}": {
                seed: float(owner_seed[left][seed].get(f"next_owner:{right}", 0.0)) - float(owner_seed[right][seed].get(f"next_owner:{right}", 0.0))
                for seed in seed_order
            },
        }
        crossover_interval = _joint_bootstrap_intervals(contrast_series, seed_order=seed_order, replicates=replicates, family=f"reciprocal_commit_crossover:{left}:{right}", seed_offset=len(pairs))
        crossover_gates = {name: {**metric, "status": _gate_status(lower_bound=float(metric["lower_95"]), threshold=0.0, direction="greater")} for name, metric in crossover_interval["metrics"].items()}
        good_bad_by_world: dict[str, Any] = {}
        for world, world_owner_seed in (("lower", owner_seed), ("upper", upper_owner_seed or owner_seed)):
            good_bad_series = {
                f"good_minus_bad:{owner}": {
                    seed: float(world_owner_seed[owner][seed].get("good_event", 0.0)) - float(world_owner_seed[owner][seed].get("bad_event", 0.0))
                    for seed in seed_order
                }
                for owner in (left, right)
            }
            interval = _joint_bootstrap_intervals(good_bad_series, seed_order=seed_order, replicates=replicates, family=f"reciprocal_good_minus_bad:{left}:{right}:{world}", seed_offset=1000 + len(pairs) * 2 + (0 if world == "lower" else 1))
            good_bad_by_world[world] = {
                "intervals": interval,
                "gates": {name: {**metric, "status": _gate_status(lower_bound=float(metric["lower_95"]), threshold=-0.10, direction="greater")} for name, metric in interval["metrics"].items()},
            }
        # Confirmation seeds estimate the aggregate owner-pair contrasts. A
        # noisy negative seed is not itself a reversal; interval-level lower
        # bounds own that judgment, while strict positivity remains a separate
        # gate below.
        no_reversal = all(
            float(metric["lower_95"]) >= 0.0
            for metric in crossover_interval["metrics"].values()
        )
        strict = no_reversal and all(item["status"] == "pass" for item in crossover_gates.values()) and all(item["status"] == "pass" for world in good_bad_by_world.values() for item in world["gates"].values())
        pairs[f"{left}__vs__{right}"] = {
            "owners": [left, right],
            "status": "identified" if strict else "not_identified",
            "strict_verdict": "supported" if strict else "rejected_or_inconclusive",
            "contrasts": crossover_gates,
            "per_seed_contrasts": contrast_series,
            "no_reversal": no_reversal,
            "good_minus_bad": {world: value["gates"] for world, value in good_bad_by_world.items()},
            "bootstrap": {"crossover": crossover_interval, "good_minus_bad": {world: value["intervals"] for world, value in good_bad_by_world.items()}},
            "requirements": {
                "both_reciprocal_lower_bounds_positive": all(item["status"] == "pass" for item in crossover_gates.values()),
                "no_reversal": no_reversal,
                "both_good_minus_bad_gates_pass": all(item["status"] == "pass" for world in good_bad_by_world.values() for item in world["gates"].values()),
            },
        }
    strict_panel = bool(pairs) and all(value["status"] == "identified" for value in pairs.values())
    return {"status": "identified" if strict_panel else "not_identified", "strict_verdict": "supported" if strict_panel else "rejected_or_inconclusive", "pair_count": len(pairs), "pairs": pairs}


def _exact_variant_crossover_primitives(
    exact_rows: Sequence[Mapping[str, Any]],
    owners: Sequence[str],
    *,
    seed_order: Sequence[str],
    replicates: int,
) -> dict[str, Any]:
    """Compare every exact-variant pair with paired seed-index intervals.

    This is deliberately stricter than an aggregate owner comparison: every
    unordered owner pair must have a directionally consistent crossover for
    every exact variant combination.  A reversal in one exact combination is
    retained as evidence and prevents a strong panel claim.
    """

    from itertools import combinations

    chosen = sorted({str(item) for item in owners})
    by_owner: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in exact_rows:
        owner = str(row.get("owner_id") or "")
        if owner in chosen:
            by_owner[owner].append(row)
    if len(chosen) < 2:
        return {"status": "not_identified", "reason": "fewer_than_two_admitted_owner_arms", "pair_count": 0, "pairs": {}}
    if any(not by_owner[owner] for owner in chosen):
        return {"status": "refused", "reason": "missing_exact_variant_owner_rows", "pair_count": 0, "pairs": {}}

    pairs: dict[str, Any] = {}
    for pair_index, (left, right) in enumerate(combinations(chosen, 2)):
        variants: dict[str, Any] = {}
        all_no_reversal = True
        for left_row in by_owner[left]:
            for right_row in by_owner[right]:
                left_seed_rows = {str(item["seed"]): item for item in left_row.get("per_seed", []) if isinstance(item, Mapping)}
                right_seed_rows = {str(item["seed"]): item for item in right_row.get("per_seed", []) if isinstance(item, Mapping)}
                if set(left_seed_rows) != set(seed_order) or set(right_seed_rows) != set(seed_order):
                    return {"status": "refused", "reason": "incomplete_exact_variant_seed_panel", "pair_count": len(pairs), "pairs": pairs}
                series = {
                    f"C_{left}": {
                        seed: float(right_seed_rows[seed].get("next_owner") == left) - float(left_seed_rows[seed].get("next_owner") == left)
                        for seed in seed_order
                    },
                    f"C_{right}": {
                        seed: float(left_seed_rows[seed].get("next_owner") == right) - float(right_seed_rows[seed].get("next_owner") == right)
                        for seed in seed_order
                    },
                }
                interval = _joint_bootstrap_intervals(
                    series,
                    seed_order=seed_order,
                    replicates=replicates,
                    family=f"exact_variant_crossover:{left}:{right}:{left_row.get('branch_hash')}:{right_row.get('branch_hash')}",
                    seed_offset=pair_index,
                )
                # Paired seeds estimate the mean contrast for this exact
                # variant pair.  A single noisy negative seed is not a
                # reversal; use the simultaneous interval's lower bounds.
                no_reversal = all(
                    float(metric["lower_95"]) >= 0.0
                    for metric in interval["metrics"].values()
                )
                all_no_reversal = all_no_reversal and no_reversal
                variant_key = f"{left_row.get('branch_hash')}__vs__{right_row.get('branch_hash')}"
                variants[variant_key] = {
                    "owners": [left, right],
                    "branch_hashes": [str(left_row.get("branch_hash")), str(right_row.get("branch_hash"))],
                    "per_seed_contrasts": series,
                    "intervals": interval,
                    "no_reversal": no_reversal,
                    "directional_gates": {
                        name: _gate_status(lower_bound=float(metric["lower_95"]), threshold=0.0, direction="greater")
                        for name, metric in interval["metrics"].items()
                    },
                }
        pairs[f"{left}__vs__{right}"] = {
            "owners": [left, right],
            "variant_count": len(variants),
            "no_reversal_across_exact_variants": all_no_reversal,
            "variants": variants,
            "status": "identified" if all_no_reversal and all(
                all(item["directional_gates"][name] == "pass" for name in item["directional_gates"])
                for item in variants.values()
            ) else "not_identified",
        }
    strict = bool(pairs) and all(item["status"] == "identified" for item in pairs.values())
    return {
        "status": "identified" if strict else "not_identified",
        "strict_verdict": "supported" if strict else "rejected_or_inconclusive",
        "pair_count": len(pairs),
        "pairs": pairs,
    }


def _formal_greedy_gap(
    owner_comparisons: Mapping[str, Mapping[str, Any]],
    *,
    greedy_owner_id: str,
    value_key: str,
    comparison_evaluated: bool,
) -> dict[str, Any]:
    """Return formal G_H with the greedy owner's zero contrast included."""

    if not comparison_evaluated or not owner_comparisons or not str(greedy_owner_id):
        return {
            "formal_value": None,
            "formal_owner_id": None,
            "maximum_alternative_value": None,
            "maximum_alternative_owner_id": None,
            "greedy_zero_baseline_included": False,
            "status": "not_applicable",
        }

    alternative_owner = max(
        owner_comparisons,
        key=lambda owner: float(owner_comparisons[owner][value_key]),
    ) if owner_comparisons else None
    alternative_value = (
        float(owner_comparisons[alternative_owner][value_key])
        if alternative_owner is not None
        else None
    )
    if alternative_value is not None and alternative_value > 0.0:
        formal_value = alternative_value
        formal_owner = alternative_owner
    else:
        formal_value = 0.0
        formal_owner = str(greedy_owner_id)
    return {
        "formal_value": formal_value,
        "formal_owner_id": formal_owner,
        "maximum_alternative_value": alternative_value,
        "maximum_alternative_owner_id": alternative_owner,
        "greedy_zero_baseline_included": True,
        "status": "identified",
    }


def _all_exact_variant_pairs_nonreversing(exact_variant_crossover: Mapping[str, Any]) -> bool:
    """Aggregate pair-level no-reversal independently of strict status."""

    pairs = exact_variant_crossover.get("pairs", {})
    if not isinstance(pairs, Mapping) or not pairs:
        return False
    return all(
        isinstance(pair, Mapping) and bool(pair.get("no_reversal_across_exact_variants"))
        for pair in pairs.values()
    )


def _analyze(
    *,
    unit_path: Path,
    admission_path: Path,
    variants_path: Path,
    ledger_path: Path,
    donor_bundle: Path,
    donor_prefix_token_count: int,
    roots: Sequence[Path],
    review_path: Path,
    output: Path,
    bootstrap_replicates: int,
    expected_runtime_dtype: str = "config",
) -> dict[str, Any]:
    unit_path_str, unit_digest = _unit_digest(unit_path)
    admission = _load_admission(admission_path)
    variants = _load_variants(variants_path)
    ledger = _load_ledger(ledger_path)
    ledger_digest = _sha256_file(ledger_path)
    donor_data = _read_json(donor_bundle)
    image_id = str(donor_data.get("execution_evidence", {}).get("image_id", donor_data.get("image_id", "")))
    donor_prefix_hash = _expected_parent_prefix_hash(donor_bundle, donor_prefix_token_count)
    records, receipt_meta = _receipt_records(
        roots,
        variants=variants,
        donor_prefix_hash=donor_prefix_hash,
        ledger_digest=ledger_digest,
        expected_runtime_dtype=expected_runtime_dtype,
    )
    parent_details = _parent_coverage_details(donor_bundle, prefix_count=donor_prefix_token_count, image_id=image_id, ledger=ledger)
    parent_rows = [row for row in parent_details["unresolved_rows"] if isinstance(row.get("prediction"), Mapping) and isinstance(row["prediction"].get("bbox"), Sequence) and len(row["prediction"].get("bbox", [])) == 4]
    candidate_rows = _unmatched_rows(records, ledger=ledger, image_id=image_id) + parent_rows
    candidate_rows += _greedy_control_review_rows(admission, image_id=image_id, ledger=ledger, parent_covered=set(parent_details["covered_owner_ids"]))
    candidates = _candidate_clusters(candidate_rows)
    frozen = _freeze_review(candidates, _load_review(review_path))
    covered = set(parent_details["covered_owner_ids"])
    unresolved_parent_after_review: list[dict[str, Any]] = []
    for row in parent_details["unresolved_rows"]:
        prediction = row.get("prediction")
        if not isinstance(prediction, Mapping) or not isinstance(prediction.get("bbox"), Sequence) or len(prediction.get("bbox", [])) != 4:
            unresolved_parent_after_review.append(dict(row))
            continue
        physical = _match_physical(prediction, image_id=image_id, ledger=ledger, candidates=candidates, frozen_review=frozen)
        if physical.get("physical_status") in {"unique", "review_approved"} and physical.get("owner_id"):
            covered.add(str(physical["owner_id"]))
        else:
            unresolved_parent_after_review.append({**dict(row), "physical": physical})
    admitted_owner_ids = {str(x) for x in admission.get("admitted_owner_ids", [])}
    accepted_owner_ids = {
        str(item.get("object_identifier"))
        for item in ledger
        if item.get("object_identifier") and str(item.get("final_state", "accepted")) == "accepted"
    }
    predeclared_owner_ids = {
        str(item.get("object_identifier"))
        for item in ledger
        if item.get("object_identifier") and str(item.get("final_state", "accepted")) == "accepted" and str(item.get("object_identifier")) not in covered
    }
    paths: list[dict[str, Any]] = []
    for record in records:
        for _, call in sorted(record["calls_by_seed"].items(), key=lambda item: int(item[0])):
            enriched = {**record, "call": call}
            paths.append(_classify_path(enriched, image_id=image_id, parent_covered=covered, ledger=ledger, candidates=candidates, frozen_review=frozen, discovered_owner_ids=predeclared_owner_ids, accepted_owner_ids=accepted_owner_ids))
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        grouped[str(path["branch_hash"])].append(path)
    exact_rows: list[dict[str, Any]] = []
    for branch_hash, branch_paths in sorted(grouped.items()):
        owner = str(next(record["owner_id"] for record in records if record["branch_hash"] == branch_hash))
        metric_rows = []
        for path in sorted(branch_paths, key=lambda item: int(item["seed"])):
            lower = _outcome_summary(path["outcomes"], unknown_as_supported=False)
            upper = _outcome_summary(path["outcomes"], unknown_as_supported=True)
            first = next((item for item in path["outcomes"] if int(item.get("row_index", -1)) == 0), None)
            metric_rows.append({"seed": path["seed"], "lower": lower, "upper": upper, "outcomes": path["outcomes"], "next_owner": first.get("owner_id") if isinstance(first, Mapping) else None})
        q_lower = sum(row["lower"]["new_supported_count"] for row in metric_rows) / len(metric_rows)
        q_upper = sum(row["upper"]["new_supported_count"] for row in metric_rows) / len(metric_rows)
        exact_rows.append({"branch_hash": branch_hash, "owner_id": owner, "confirmation_count": len(metric_rows), "q_H_lower": q_lower, "q_H_upper": q_upper, "per_seed": metric_rows})
    owner_values: dict[str, dict[str, float]] = {}
    owner_variant_completeness: dict[str, dict[str, Any]] = {}
    expected_by_owner: dict[str, set[str]] = defaultdict(set)
    for variant in variants.values():
        owner = str(variant["owner_id"])
        if owner in admitted_owner_ids:
            expected_by_owner[owner].add(str(variant["row_token_ids_sha256"]))
    observed_by_owner: dict[str, set[str]] = defaultdict(set)
    weight_by_branch = {str(record["branch_hash"]): float(record["discovery_support_count"]) for record in records}
    for record in records:
        observed_by_owner[str(record["owner_id"])].add(str(record["branch_hash"]))
    for owner in sorted(admitted_owner_ids):
        owner_rows = [row for row in exact_rows if row["owner_id"] == owner]
        observed = observed_by_owner.get(owner, set())
        expected = expected_by_owner.get(owner, set())
        confirmation_counts = {str(row["branch_hash"]): int(row["confirmation_count"]) for row in owner_rows}
        owner_variant_completeness[owner] = _owner_completeness(expected, observed, confirmation_counts)
        complete = bool(owner_variant_completeness[owner]["complete"])
        total_weight = sum(weight_by_branch[row["branch_hash"]] for row in owner_rows)
        if complete and total_weight > 0:
            owner_values[owner] = {world: sum(float(row[f"q_H_{world}"]) * weight_by_branch[row["branch_hash"]] for row in owner_rows) / total_weight for world in ("lower", "upper")}
    owner_seed_lower = _owner_seed_metrics(records, paths, unknown_as_supported=False, owner_ids=sorted(admitted_owner_ids))
    owner_seed_upper = _owner_seed_metrics(records, paths, unknown_as_supported=True, owner_ids=sorted(admitted_owner_ids))
    seed_order = [str(seed) for seed in receipt_meta["confirmation_seeds"]]
    complete_owners = sorted(owner_values)
    greedy = _resolve_greedy_control(admission, image_id=image_id, ledger=ledger, parent_covered=covered, candidates=candidates, frozen_review=frozen)
    owner_vs_greedy: dict[str, Any] = {}
    owner_vs_greedy_families: dict[str, dict[str, Any]] = {}
    greedy_owner = str(greedy.get("owner_id") or "")
    if greedy.get("status") == "validated" and greedy_owner in complete_owners and not unresolved_parent_after_review and not any(owner in covered for owner in admitted_owner_ids):
        alternatives = [owner for owner in complete_owners if owner != greedy_owner]
        estimands = {
            "q_H_difference": "new_supported_count",
            "unsupported_difference": "unsupported_event",
            "invalid_difference": "invalid_event",
            "total_duplicate_difference": "duplicate_event",
            "duplicate_current_difference": "duplicate_current_event",
            "duplicate_covered_difference": "duplicate_covered_event",
            "unknown_difference": "unknown_event",
            "terminal_difference": "terminal_event",
        }
        safety_estimands = frozenset({"unsupported_difference", "invalid_difference", "total_duplicate_difference"})
        for world, owner_seed in (("lower", owner_seed_lower), ("upper", owner_seed_upper)):
            owner_vs_greedy_families[world] = {}
            for estimand, metric_name in estimands.items():
                series = {
                    owner: {seed: float(owner_seed[owner][seed][metric_name]) - float(owner_seed[greedy_owner][seed][metric_name]) for seed in seed_order}
                    for owner in alternatives
                }
                owner_vs_greedy_families[world][estimand] = _joint_bootstrap_intervals(
                    series,
                    seed_order=seed_order,
                    replicates=bootstrap_replicates,
                    family=f"owner-vs-greedy:{greedy_owner}:{world}:{estimand}",
                    seed_offset=(0 if world == "lower" else 100) + list(estimands).index(estimand),
                )
        for owner in alternatives:
            worlds: dict[str, Any] = {}
            for world in ("lower", "upper"):
                gates: dict[str, Any] = {}
                for estimand in estimands:
                    metric = owner_vs_greedy_families[world][estimand]["metrics"][owner]
                    if estimand == "q_H_difference":
                        status = _gate_status(lower_bound=float(metric["lower_95"]), threshold=0.0, direction="greater")
                    elif estimand in safety_estimands:
                        status = _gate_status(lower_bound=float(metric["upper_95"]), threshold=0.10, direction="less_equal")
                    else:
                        status = "descriptive"
                    gates[estimand] = {**metric, "status": status}
                worlds[world] = {"gates": gates, "estimand_families": {estimand: owner_vs_greedy_families[world][estimand]["family"] for estimand in estimands}}
            owner_vs_greedy[owner] = {
                "owner_id": owner,
                "greedy_owner_id": greedy_owner,
                "Q_H_lower": owner_values[owner]["lower"] - owner_values[greedy_owner]["lower"],
                "Q_H_upper": owner_values[owner]["upper"] - owner_values[greedy_owner]["upper"],
                "worlds": worlds,
                "positive_unique_value_both_worlds": all(worlds[world]["gates"]["q_H_difference"]["status"] == "pass" for world in worlds),
                "safety_gates_both_worlds": all(worlds[world]["gates"][name]["status"] == "pass" for world in worlds for name in safety_estimands),
            }
    crossover = _crossover_primitives(owner_seed_lower, complete_owners, seed_order=seed_order, replicates=bootstrap_replicates, upper_owner_seed=owner_seed_upper)
    # A failed greedy control only refuses the greedy branch-value handle.
    # Commit crossover remains a sibling-arm claim and must not be silently
    # discarded because the external control was unresolved.
    owner_claims_refused = bool(unresolved_parent_after_review or any(not value["complete"] for value in owner_variant_completeness.values()) or any(owner in covered for owner in admitted_owner_ids))
    if owner_claims_refused:
        owner_vs_greedy = {owner: {**value, "status": "refused_due_to_completeness_or_control"} for owner, value in owner_vs_greedy.items()}
        crossover = {"status": "refused", "reason": "owner_level_completeness_or_parent_control_failure", "underlying": crossover}
    exact_variant_crossover = _exact_variant_crossover_primitives(
        exact_rows,
        complete_owners,
        seed_order=seed_order,
        replicates=bootstrap_replicates,
    )
    crossover["exact_variant_crossover"] = exact_variant_crossover
    crossover["no_reversal_across_exact_variants"] = _all_exact_variant_pairs_nonreversing(exact_variant_crossover)
    if crossover.get("status") == "identified" and exact_variant_crossover.get("status") != "identified":
        crossover["status"] = "not_identified"
        crossover["strict_verdict"] = "rejected_or_inconclusive"
    greedy_evaluated = bool(owner_vs_greedy) and not owner_claims_refused
    positive_safe_owners = [
        owner
        for owner, value in owner_vs_greedy.items()
        if value.get("positive_unique_value_both_worlds") and value.get("safety_gates_both_worlds")
    ]
    formal_lower = _formal_greedy_gap(owner_vs_greedy, greedy_owner_id=greedy_owner, value_key="Q_H_lower", comparison_evaluated=greedy_evaluated)
    formal_upper = _formal_greedy_gap(owner_vs_greedy, greedy_owner_id=greedy_owner, value_key="Q_H_upper", comparison_evaluated=greedy_evaluated)
    max_safe_owner = max(positive_safe_owners, key=lambda owner: owner_vs_greedy[owner]["Q_H_lower"]) if positive_safe_owners else None
    greedy_positive = bool(positive_safe_owners)
    greedy_safe = bool(positive_safe_owners)
    result = {
        "schema_version": SCHEMA_VERSION,
        "unit_id": UNIT_ID,
        "unit": {"path": unit_path_str, "sha256": unit_digest},
        "inputs": {
            "admission_manifest": str(admission_path.resolve()),
            "admission_manifest_sha256": _sha256_file(admission_path),
            "variants_jsonl": str(variants_path.resolve()),
            "variants_jsonl_sha256": _sha256_file(variants_path),
            "ledger": str(ledger_path.resolve()),
            "ledger_sha256": ledger_digest,
            "donor_bundle": str(donor_bundle.resolve()),
            "donor_bundle_sha256": _sha256_file(donor_bundle),
            "donor_prefix_token_count": int(donor_prefix_token_count),
            "donor_prefix_token_ids_sha256": donor_prefix_hash,
            "review": str(review_path.resolve()),
            "review_sha256": _sha256_file(review_path),
        },
        "receipt_inputs": receipt_meta,
        "parent_covered_owner_ids": sorted(covered),
        "parent_unresolved_rows": unresolved_parent_after_review,
        "parent_coverage_reviewed_row_count": int(parent_details.get("prefix_row_count", 0)),
        "parent_covered_branch_owner_ids": sorted(owner for owner in admitted_owner_ids if owner in covered),
        "admitted_owner_ids": sorted(admitted_owner_ids),
        "frozen_review": frozen,
        "blind_candidate_count": len(candidates),
        "exact_row_q_H": exact_rows,
        "owner_Q_H": owner_values,
        "owner_variant_completeness": owner_variant_completeness,
        "owner_level_claims_refused": owner_claims_refused,
        "claims": {
            "greedy_branch_value_gap": {
                "status": "identified" if greedy_evaluated else "refused",
                "positive_safe_handle": bool(positive_safe_owners),
                "greedy_control": greedy,
                "owner_comparisons": owner_vs_greedy,
                "G_H_lower": formal_lower["formal_value"],
                "G_H_lower_owner_id": formal_lower["formal_owner_id"],
                "G_H_upper": formal_upper["formal_value"],
                "G_H_upper_owner_id": formal_upper["formal_owner_id"],
                "greedy_zero_baseline_included": bool(formal_lower["greedy_zero_baseline_included"] and formal_upper["greedy_zero_baseline_included"]),
                "max_raw_G_H_lower": formal_lower["formal_value"],
                "max_raw_G_H_lower_owner_id": formal_lower["formal_owner_id"],
                "max_raw_G_H_upper": formal_upper["formal_value"],
                "max_raw_G_H_upper_owner_id": formal_upper["formal_owner_id"],
                "max_alternative_G_H_lower": formal_lower["maximum_alternative_value"],
                "max_alternative_G_H_lower_owner_id": formal_lower["maximum_alternative_owner_id"],
                "max_alternative_G_H_upper": formal_upper["maximum_alternative_value"],
                "max_alternative_G_H_upper_owner_id": formal_upper["maximum_alternative_owner_id"],
                "max_eligible_safe_G_H_lower": owner_vs_greedy[max_safe_owner]["Q_H_lower"] if max_safe_owner else None,
                "max_eligible_safe_G_H_lower_owner_id": max_safe_owner,
                "unique_value_sign_stable": greedy_positive,
                "safety_gates_pass": greedy_safe,
            },
            "commit_crossover": crossover,
            "good_minus_bad": {
                "status": "identified" if crossover.get("status") == "identified" else "not_identified",
                "unknown_worlds": {"lower": True, "upper": True},
            },
        },
        "bootstrap": {
            "seed_root": BOOTSTRAP_SEED_ROOT,
            "replicates": int(bootstrap_replicates),
            "simultaneous_max_statistic": True,
            "interval_method": "non_studentized_max_deviation",
            "paired_seed_index_intervals": owner_vs_greedy_families,
        },
        "limitations": ["Owner-level claims are refused when an admitted owner is absent/incomplete, a parent row remains unresolved, or a branch owner is parent-covered; greedy-control failure only refuses G_H."],
    }
    _write_once(output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("collect-review", "freeze-check", "analyze"), required=True)
    parser.add_argument("--unit", type=Path, required=True)
    parser.add_argument("--admission-manifest", type=Path, required=True)
    parser.add_argument("--variants-jsonl", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--donor-bundle", type=Path, required=True)
    parser.add_argument("--donor-prefix-token-count", type=int, required=True)
    parser.add_argument("--wave2-root", action="append", type=Path, required=True)
    parser.add_argument(
        "--expected-runtime-dtype",
        choices=("config", "fp32"),
        default="config",
        help="Receipt runtime lineage to accept; fp32 is explicit opt-in for the full-model robustness replay.",
    )
    parser.add_argument("--review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.donor_prefix_token_count < 0:
        raise SystemExit("--donor-prefix-token-count must be non-negative")
    unit_path, _ = _unit_digest(args.unit)
    admission = _load_admission(args.admission_manifest)
    variants = _load_variants(args.variants_jsonl)
    ledger = _load_ledger(args.ledger)
    image_id = str(_read_json(args.donor_bundle).get("execution_evidence", {}).get("image_id", ""))
    prefix_hash = _expected_parent_prefix_hash(args.donor_bundle, args.donor_prefix_token_count)
    records, receipt_meta = _receipt_records(
        args.wave2_root,
        variants=variants,
        donor_prefix_hash=prefix_hash,
        ledger_digest=_sha256_file(args.ledger),
        expected_runtime_dtype=args.expected_runtime_dtype,
    )
    parent_details = _parent_coverage_details(args.donor_bundle, prefix_count=args.donor_prefix_token_count, image_id=image_id, ledger=ledger)
    parent_rows = [row for row in parent_details["unresolved_rows"] if isinstance(row.get("prediction"), Mapping) and isinstance(row["prediction"].get("bbox"), Sequence) and len(row["prediction"].get("bbox", [])) == 4]
    candidate_rows = _unmatched_rows(records, ledger=ledger, image_id=image_id) + parent_rows
    candidate_rows += _greedy_control_review_rows(admission, image_id=image_id, ledger=ledger, parent_covered=set(parent_details["covered_owner_ids"]))
    candidates = _candidate_clusters(candidate_rows)
    if args.mode == "collect-review":
        result = {"schema_version": f"{SCHEMA_VERSION}.blind_review.v1", "unit_id": UNIT_ID, "unit": {"path": unit_path, "sha256": _sha256_file(args.unit)}, "image_id": image_id, "candidate_count": len(candidates), "candidates": [{k: item[k] for k in ("candidate_id", "image_id", "category", "bbox_xyxy", "support_count")} for item in candidates], "receipt_inputs": receipt_meta, "blinded": True, "hidden_fields_excluded": ["arm", "sampling_seed", "branch_variant", "request_id"]}
        _write_once(args.output, result)
        print(json.dumps({"output": str(args.output.resolve()), "candidate_count": len(candidates)}, sort_keys=True))
        return 0
    if args.review is None:
        raise SystemExit(f"{args.mode} requires --review")
    frozen = _freeze_review(candidates, _load_review(args.review))
    if args.mode == "freeze-check":
        result = {"schema_version": f"{SCHEMA_VERSION}.freeze_check.v1", "unit_id": UNIT_ID, "unit": {"path": unit_path, "sha256": _sha256_file(args.unit)}, "image_id": image_id, "candidate_count": len(candidates), "frozen_review": frozen, "receipt_inputs": receipt_meta, "admitted_owner_ids": sorted(str(x) for x in admission.get("admitted_owner_ids", []))}
        _write_once(args.output, result)
        print(json.dumps({"output": str(args.output.resolve()), "candidate_count": len(candidates), "complete": True}, sort_keys=True))
        return 0
    result = _analyze(
        unit_path=args.unit,
        admission_path=args.admission_manifest,
        variants_path=args.variants_jsonl,
        ledger_path=args.ledger,
        donor_bundle=args.donor_bundle,
        donor_prefix_token_count=args.donor_prefix_token_count,
        roots=args.wave2_root,
        review_path=args.review,
        output=args.output,
        bootstrap_replicates=args.bootstrap_replicates,
        expected_runtime_dtype=args.expected_runtime_dtype,
    )
    print(json.dumps({"output": str(args.output.resolve()), "exact_row_count": len(result["exact_row_q_H"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
