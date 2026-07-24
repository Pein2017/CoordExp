#!/usr/bin/env python3
"""Assemble the fixed-dose broad and concentrated StateBanks.

This is the narrow final assembly step for the 2,432-image constant-dose
panel.  It consumes only frozen artifacts: the candidate pool and split,
the old/new greedy and sampled rollout artifacts, the trajectory-union
analysis, and the upstream panel-union receipt.  In particular, it does not
borrow a reference StateBank or accept a manually edited review overlay.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import copy
import glob
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.fingerprint import sha256_file, sha256_json  # noqa: E402
from src.inference.backend import token_ids_sha256  # noqa: E402
from src.qwen.special_token_embeddings import SpecialTokenSelection  # noqa: E402
from src.rollout_calibration import CheckpointIdentity  # noqa: E402

from scripts.research.assemble_positive_path_imitation_state_bank import (  # noqa: E402
    AssemblyError,
    _image_id,
    _image_pad_interval,
    _mapping,
    _norm_sha,
    _read_json,
    _token_ids,
)
from scripts.research.assemble_source_preservation_multi_route_state_banks import (  # noqa: E402
    ROUTE_ANALYSIS_SCHEMA_VERSION,
    _breadth_identity,
    _breadth_sampled_rank,
    _build_multi_candidates,
    _build_source_candidates,
    _load_rollout_rows,
    _source_artifacts,
    _write_arm,
    _write_json,
    derive_checkpoint_identity_from_rollout_artifact,
    materialize_constant_dose_breadth_arm,
    select_constant_dose_breadth_arms,
    validate_constant_dose_panel_execution_contract,
    validate_constant_dose_training_reservoir,
)
from scripts.research.analyze_individual_trajectory_union_support import (  # noqa: E402
    _malformed_before_budget,
    _parsed_rows,
    load_generation7_annotations,
    match_prefix,
)
from scripts.research.validate_constant_dose_trajectory_panel_union import (  # noqa: E402
    GREEDY_SEED,
    SAMPLED_SEEDS,
    validate_panel_union,
)


SCHEMA_VERSION = "constant_dose_breadth_state_bank_assembly.v1"
UNION_SCHEMA_VERSION = "constant_dose_trajectory_panel_union.v1"
CANONICAL_ANALYSIS_IOU_THRESHOLD = 0.5
V2_PANEL_SCHEMA_VERSION = "coordexp_vllm_trajectory_panel.v2"
SAMPLED_B16_ACCEPTED_STATUSES = frozenset(
    {"accepted_budget", "accepted_natural_end"}
)
SAMPLED_B16_FAILED_STATUS = "failed_invalid_before_budget"
SOURCE_B16_ACCEPTED_STATUSES = frozenset({"accepted_budget", "accepted_natural_end"})
SOURCE_B16_INELIGIBLE_STATUSES = frozenset(
    {"failed_invalid_before_budget", "failed_token_limit_before_budget"}
)
SOURCE_B16_STATUSES = SOURCE_B16_ACCEPTED_STATUSES | SOURCE_B16_INELIGIBLE_STATUSES
SOURCE_B16_ROW_BUDGET = 16
SAMPLED_INDEX_COUNT = 16
ANALYZER_PATH = Path(__file__).resolve().with_name(
    "analyze_individual_trajectory_union_support.py"
)


def _canonical_image_id(value: Any, *, field: str) -> str:
    if isinstance(value, bool) or value is None:
        raise AssemblyError(f"{field} has no usable image_id")
    if isinstance(value, (int, str)) and str(value).strip():
        return str(value).strip()
    raise AssemblyError(f"{field} has no usable image_id")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            raise AssemblyError(f"blank JSONL row {line_number}: {path}")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AssemblyError(f"invalid JSONL row {line_number}: {path}") from exc
        if not isinstance(value, Mapping):
            raise AssemblyError(f"non-object JSONL row {line_number}: {path}")
        rows.append(dict(value))
    return rows


def _candidate_pool(path: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(_read_jsonl(path), 1):
        image_id = _canonical_image_id(row.get("image_id"), field=f"candidate-pool row {index}")
        if image_id in result:
            raise AssemblyError(f"candidate pool duplicates image_id {image_id}")
        result[image_id] = row
    if len(result) != 2432:
        raise AssemblyError(f"candidate pool has {len(result)} images, expected 2432")
    return result


def _split_membership(*, candidate_pool: Path, split_receipt: Path) -> dict[str, set[str]]:
    receipt = _read_json(split_receipt)
    if not isinstance(receipt, Mapping):
        raise AssemblyError("split receipt must be an object")
    source = receipt.get("input")
    outputs = receipt.get("outputs")
    if not isinstance(source, Mapping) or source.get("sha256") != sha256_file(candidate_pool):
        raise AssemblyError("split receipt does not bind the supplied candidate pool")
    if not isinstance(outputs, Mapping) or set(outputs) != {"train_candidate", "development", "heldout"}:
        raise AssemblyError("split receipt must name exactly train_candidate/development/heldout")
    expected_counts = {"train_candidate": 2048, "development": 256, "heldout": 128}
    result: dict[str, set[str]] = {}
    for name, expected_count in expected_counts.items():
        entry = outputs[name]
        if not isinstance(entry, Mapping):
            raise AssemblyError(f"split receipt {name} entry is invalid")
        path = Path(str(entry.get("path", ""))).expanduser().resolve(strict=True)
        if entry.get("sha256") != sha256_file(path) or entry.get("count") != expected_count:
            raise AssemblyError(f"split receipt {name} does not bind its frozen JSONL")
        ids = {
            _canonical_image_id(row.get("image_id"), field=f"{name} split")
            for row in _read_jsonl(path)
        }
        if len(ids) != expected_count:
            raise AssemblyError(f"{name} split has {len(ids)} unique images, expected {expected_count}")
        result[name] = ids
    if any(left & right for index, left in enumerate(result.values()) for right in list(result.values())[index + 1:]):
        raise AssemblyError("frozen candidate-pool split memberships overlap")
    pool_ids = set(_candidate_pool(candidate_pool))
    if set().union(*result.values()) != pool_ids:
        raise AssemblyError("frozen split is not the exact candidate-pool union")
    return result


def _expand_paths(values: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        matches = [Path(item) for item in glob.glob(str(value))] or [value]
        for path in matches:
            paths.add(path.expanduser().resolve(strict=True))
    return sorted(paths)


def _load_rollouts(paths: Sequence[Path], *, sampled: bool) -> tuple[dict[tuple[str, int], dict[str, Any]], dict[str, Any]]:
    rows, configs = _load_rollout_rows(paths, sampled_only=sampled)
    for row in rows.values():
        config = configs[str(Path(str(row["_source_path"])).resolve())]
        row["_temperature"] = config["temperature"]
        row["_top_p"] = config["top_p"]
        row["_repetition_penalty"] = config["repetition_penalty"]
    return rows, configs


def _rollout_prompt_evidence(
    *, paths: Sequence[Path], rows: Mapping[tuple[str, int], Mapping[str, Any]]
) -> dict[tuple[str, int], dict[str, Any]]:
    """Read exact prompt/image evidence for every row without re-tokenizing."""

    metadata_by_path: dict[str, Mapping[str, Any]] = {}
    for path in paths:
        artifact = _read_json(path)
        if not isinstance(artifact, Mapping):
            raise AssemblyError(f"rollout artifact must be an object: {path}")
        metadata = artifact.get("prompt_metadata")
        if not isinstance(metadata, Mapping):
            raise AssemblyError(f"rollout artifact lacks prompt_metadata: {path}")
        metadata_by_path[str(path.resolve())] = metadata
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for key, row in rows.items():
        source_path = str(Path(str(row.get("_source_path", ""))).resolve())
        metadata = metadata_by_path.get(source_path)
        if metadata is None:
            raise AssemblyError(f"rollout row has unknown source artifact: {source_path}")
        example_id = str(row.get("example_id", ""))
        image_id = key[0]
        raw = metadata.get(example_id, metadata.get(image_id))
        if not isinstance(raw, Mapping):
            raise AssemblyError(f"rollout {image_id}:{key[1]} lacks prompt metadata keyed by example_id")
        prompt_ids = _token_ids(row.get("prompt_token_ids"), f"rollout {image_id}:{key[1]}.prompt_token_ids")
        prompt_hash = _norm_sha(row.get("prompt_token_ids_sha256"), f"rollout {image_id}:{key[1]}.prompt_token_ids_sha256")
        if prompt_hash != token_ids_sha256(prompt_ids):
            raise AssemblyError(f"rollout {image_id}:{key[1]} prompt-token hash mismatch")
        metadata_ids = _token_ids(raw.get("prompt_token_ids"), f"prompt metadata {image_id}.prompt_token_ids")
        metadata_hash = _norm_sha(raw.get("prompt_token_ids_sha256"), f"prompt metadata {image_id}.prompt_token_ids_sha256")
        if metadata_hash != token_ids_sha256(metadata_ids) or metadata_ids != prompt_ids or metadata_hash != prompt_hash:
            raise AssemblyError(f"rollout {image_id}:{key[1]} prompt IDs differ from its exact prompt metadata")
        image_path = raw.get("image_path")
        image_sha = raw.get("image_sha256")
        width, height = raw.get("width"), raw.get("height")
        if not isinstance(image_path, str) or not image_path or not isinstance(image_sha, str):
            raise AssemblyError(f"prompt metadata lacks image path/hash for {image_id}:{key[1]}")
        if isinstance(width, bool) or not isinstance(width, int) or isinstance(height, bool) or not isinstance(height, int):
            raise AssemblyError(f"prompt metadata lacks integer image dimensions for {image_id}:{key[1]}")
        if row.get("executed_media_sha256") != image_sha:
            raise AssemblyError(f"rollout {image_id}:{key[1]} media hash differs from prompt metadata")
        result[key] = {
            "prompt_token_ids": prompt_ids,
            "prompt_token_ids_sha256": prompt_hash,
            "image_path": str(Path(image_path).expanduser().resolve()),
            "image_sha256": image_sha,
            "width": width,
            "height": height,
        }
    return result


def derive_reference_records(
    *,
    candidate_rows: Mapping[str, Mapping[str, Any]],
    candidate_pool_path: Path,
    greedy_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    sampled_rows: Mapping[tuple[str, int], Mapping[str, Any]],
    prompt_evidence: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Derive per-image references from the frozen rollout and pool evidence."""

    references: dict[str, dict[str, Any]] = {}
    for image_id, candidate in candidate_rows.items():
        required_keys = [(image_id, GREEDY_SEED), *[(image_id, seed) for seed in sorted(SAMPLED_SEEDS)]]
        if any(key not in prompt_evidence for key in required_keys):
            raise AssemblyError(f"image {image_id} lacks greedy or all sixteen sampled prompt records")
        first = prompt_evidence[required_keys[0]]
        for key in required_keys[1:]:
            other = prompt_evidence[key]
            for field in ("prompt_token_ids", "prompt_token_ids_sha256", "image_path", "image_sha256", "width", "height"):
                if other[field] != first[field]:
                    raise AssemblyError(f"greedy/sampled exact prompt evidence differs for image {image_id}: {field}")
        images = candidate.get("images")
        metadata = candidate.get("metadata")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise AssemblyError(f"candidate-pool image {image_id} lacks one image path")
        if not isinstance(metadata, Mapping) or metadata.get("split") != "train":
            raise AssemblyError(f"candidate-pool image {image_id} lacks train split metadata")
        path = Path(images[0])
        candidate_path = (candidate_pool_path.parent / path).resolve() if not path.is_absolute() else path.resolve()
        if candidate_path != Path(str(first["image_path"])).resolve():
            raise AssemblyError(f"candidate-pool image path differs from rollout evidence for image {image_id}")
        if candidate.get("width") != first["width"] or candidate.get("height") != first["height"]:
            raise AssemblyError(f"candidate-pool dimensions differ from rollout evidence for image {image_id}")
        references[image_id] = {
            "image": {
                "image_id": int(image_id),
                "path": str(candidate_path),
                "width": int(first["width"]),
                "height": int(first["height"]),
                "content_sha256": str(first["image_sha256"]),
            },
            "split": "train",
            "executed_prompt_token_ids": list(first["prompt_token_ids"]),
            "executed_prompt_token_ids_sha256": str(first["prompt_token_ids_sha256"]),
            "image_pad_interval": list(_image_pad_interval(first["prompt_token_ids"])),
        }
    if set(greedy_rows) != {(image, GREEDY_SEED) for image in candidate_rows}:
        raise AssemblyError("greedy rows are not exactly one canonical seed per candidate-pool image")
    expected_sampled = {(image, seed) for image in candidate_rows for seed in SAMPLED_SEEDS}
    if set(sampled_rows) != expected_sampled:
        raise AssemblyError("sampled rows are not exactly the canonical sixteen seeds per candidate-pool image")
    return references


def _panel_receipt(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    if not isinstance(value, Mapping) or value.get("schema_version") != UNION_SCHEMA_VERSION or value.get("passed") is not True:
        raise AssemblyError("upstream trajectory-union receipt is not a passed canonical receipt")
    if value.get("image_count") != 2432:
        raise AssemblyError("upstream trajectory-union receipt does not cover 2,432 images")
    contract = value.get("execution_metadata")
    if not isinstance(contract, Mapping):
        raise AssemblyError("upstream trajectory-union receipt lacks request-scoped execution_metadata")
    result = dict(value)
    result["execution_metadata"] = validate_constant_dose_panel_execution_contract(contract)
    return result


def _validate_panel_receipt_match(
    saved: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> None:
    """Require the saved union receipt to describe the supplied raw artifacts.

    The assembler recomputes the panel union from the raw rollout artifacts.
    Comparing only the prompt fingerprint would leave the saved producer
    fingerprint and old/new panel counts unbound to those inputs.
    """

    fields = (
        "schema_version",
        "passed",
        "image_count",
        "old_image_count",
        "new_image_count",
        "sampled_seed_count",
        "prompt_policy_fingerprint",
        "execution_metadata",
    )
    mismatches = [field for field in fields if saved.get(field) != recomputed.get(field)]
    if mismatches:
        raise AssemblyError(
            "upstream trajectory-union receipt differs from the supplied raw artifacts: "
            + ", ".join(mismatches)
        )


def _validate_trajectory_analysis_provenance(
    analysis: Mapping[str, Any],
    *,
    candidate_pool: Path,
    rollout_paths: Sequence[Path],
    validated_union: Mapping[str, Any],
) -> None:
    """Bind trajectory analysis to the exact evidence selected for assembly."""

    sources = analysis.get("sources")
    if not isinstance(sources, Mapping):
        raise AssemblyError("trajectory analysis lacks source provenance")
    if sources.get("annotations_sha256") != sha256_file(candidate_pool):
        raise AssemblyError(
            "trajectory analysis annotations hash differs from the supplied candidate pool"
        )

    recorded_rollout_hashes = sources.get("rollout_artifact_sha256")
    if not isinstance(recorded_rollout_hashes, list) or not all(
        isinstance(value, str) for value in recorded_rollout_hashes
    ):
        raise AssemblyError("trajectory analysis lacks rollout artifact hash provenance")
    supplied_rollout_hashes = [sha256_file(path) for path in rollout_paths]
    if Counter(recorded_rollout_hashes) != Counter(supplied_rollout_hashes):
        raise AssemblyError(
            "trajectory analysis rollout artifact hashes differ from the supplied artifacts"
        )

    analyzer_path = ANALYZER_PATH.resolve(strict=True)
    if sources.get("analyzer_path") != str(analyzer_path):
        raise AssemblyError("trajectory analysis analyzer path is stale or swapped")
    if sources.get("analyzer_sha256") != sha256_file(analyzer_path):
        raise AssemblyError("trajectory analysis analyzer hash is stale or swapped")

    policy = sources.get("analysis_policy")
    if not isinstance(policy, Mapping):
        raise AssemblyError("trajectory analysis lacks analysis policy provenance")
    if policy.get("fixed_budgets") != [16] or analysis.get("fixed_budgets") != [16]:
        raise AssemblyError("trajectory analysis must use exactly fixed budget [16]")
    if (
        policy.get("iou_threshold") != CANONICAL_ANALYSIS_IOU_THRESHOLD
        or analysis.get("iou_threshold") != CANONICAL_ANALYSIS_IOU_THRESHOLD
    ):
        raise AssemblyError("trajectory analysis must use canonical IoU threshold 0.5")
    require_full_panel = policy.get("require_full_panel")
    if not isinstance(require_full_panel, bool) or analysis.get(
        "require_full_panel"
    ) is not require_full_panel:
        raise AssemblyError("trajectory analysis full/incomplete panel policy is missing or inconsistent")
    if policy.get("review_decisions_used") is not False:
        raise AssemblyError("trajectory analysis must not use a review overlay")
    if analysis.get("review_provenance") is not None or any(
        key in sources
        for key in ("review_decisions_path", "review_decisions_sha256")
    ):
        raise AssemblyError("trajectory analysis contains review overlay provenance")

    expected_observed_panel = {
        "image_count": validated_union.get("image_count"),
        "greedy_seeds": [GREEDY_SEED],
        "sampled_seeds": sorted(SAMPLED_SEEDS),
        "greedy_trajectories_per_image": [1],
        "sampled_trajectories_per_image": [len(SAMPLED_SEEDS)],
    }
    observed_panel = policy.get("observed_panel")
    if (
        observed_panel != expected_observed_panel
        or analysis.get("observed_panel") != expected_observed_panel
    ):
        raise AssemblyError(
            "trajectory analysis observed panel differs from the validated rollout union"
        )


def _discover_v2_artifacts(root: Path, *, mode: str) -> list[Path]:
    root = root.expanduser().resolve(strict=True)
    pattern = "sampled-batch-*.json" if mode == "sampled" else "source_b16-batch-*.json"
    paths = sorted(root.rglob(pattern), key=str) if root.is_dir() else []
    if not paths:
        raise AssemblyError(f"no {pattern} artifacts under {root}")
    return paths


def _read_json_exact_bytes(
    path: Path, *, expected_sha256: str | None = None
) -> tuple[Any, str]:
    """Hash one byte buffer before decoding JSON from that same buffer."""

    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and observed_sha256 != expected_sha256:
        raise AssemblyError(
            f"v2 artifact hash differs before JSON decoding: {path}"
        )
    try:
        text = raw.decode("utf-8")
        return json.loads(text), observed_sha256
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AssemblyError(f"invalid exact-byte JSON artifact: {path}") from exc


def _resolve_v2_artifact_inventory(
    root: Path,
    *,
    mode: str,
    strict_inventory: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[Path], dict[Path, str]]:
    root = root.expanduser().resolve(strict=True)
    discovered = _discover_v2_artifacts(root, mode=mode)
    if strict_inventory is None:
        return discovered, {}
    expected: dict[Path, str] = {}
    for index, raw in enumerate(strict_inventory):
        path = Path(str(raw.get("path", ""))).expanduser().resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise AssemblyError(
                f"strict {mode} artifact is outside its panel root: {path}"
            ) from exc
        digest = str(raw.get("sha256", ""))
        if len(digest) != 64:
            raise AssemblyError(
                f"strict {mode} artifact {index} has an invalid SHA-256"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise AssemblyError(
                f"strict {mode} artifact {index} has an invalid SHA-256"
            ) from exc
        if path in expected:
            raise AssemblyError(f"strict {mode} artifact inventory duplicates {path}")
        expected[path] = digest
    if set(discovered) != set(expected):
        raise AssemblyError(
            f"{mode} artifact discovery differs from strict inventory"
        )
    return discovered, expected


def _v2_execution_identity(payload: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], str]:
    model = _mapping(payload.get("model_identity"), f"{path}.model_identity")
    execution = dict(
        _mapping(
            model.get("execution_model_identity"),
            f"{path}.model_identity.execution_model_identity",
        )
    )
    return execution, sha256_json(execution)


def _v2_tokenizer_contract(
    payload: Mapping[str, Any], path: Path
) -> tuple[dict[str, Any], str, int, int]:
    model = _mapping(payload.get("model_identity"), f"{path}.model_identity")
    tokenizer = dict(
        _mapping(model.get("tokenizer_identity"), f"{path}.tokenizer_identity")
    )
    im_end_ids = tokenizer.get("im_end_token_ids")
    wrappers = _mapping(
        tokenizer.get("wrapper_token_ids"), f"{path}.tokenizer_identity.wrapper_token_ids"
    )
    box_end = wrappers.get("<|box_end|>")
    if (
        not isinstance(im_end_ids, list)
        or len(im_end_ids) != 1
        or isinstance(im_end_ids[0], bool)
        or not isinstance(im_end_ids[0], int)
        or isinstance(box_end, bool)
        or not isinstance(box_end, int)
        or im_end_ids[0] == box_end
    ):
        raise AssemblyError(f"invalid frozen tokenizer token contract in {path}")
    return tokenizer, sha256_json(tokenizer), int(im_end_ids[0]), int(box_end)


def _project_sampled_b16(
    row: Mapping[str, Any], *, im_end_token_id: int, box_end_token_id: int
) -> tuple[list[int], str, dict[str, Any]]:
    """Project one naturally closed sampled route to its exact B16 token prefix."""

    raw_ids = _token_ids(row.get("generated_token_ids"), "sampled.generated_token_ids")
    raw_hash = _norm_sha(
        row.get("generated_token_ids_sha256"), "sampled.generated_token_ids_sha256"
    )
    if token_ids_sha256(raw_ids) != raw_hash:
        raise AssemblyError("sampled raw generated-token hash mismatch")
    if (
        row.get("stop_reason") != "im_end"
        or not raw_ids
        or raw_ids[-1] != im_end_token_id
        or raw_ids.count(im_end_token_id) != 1
    ):
        raise AssemblyError("sampled route lacks exact terminal im_end token evidence")
    parser_ids = raw_ids[:-1]
    parsed, parser = _parsed_rows(row)
    malformed_before = _malformed_before_budget(
        parser, parsed, SOURCE_B16_ROW_BUDGET
    )
    box_ends = [index for index, value in enumerate(parser_ids) if value == box_end_token_id]
    if len(box_ends) < min(len(parsed), SOURCE_B16_ROW_BUDGET):
        raise AssemblyError("sampled parser rows exceed exact box-end token evidence")
    if malformed_before:
        projected = list(parser_ids)
        status = SAMPLED_B16_FAILED_STATUS
        projected_count = min(len(parsed), SOURCE_B16_ROW_BUDGET)
        projected_end = len(projected)
    elif len(parsed) >= SOURCE_B16_ROW_BUDGET:
        projected_end = box_ends[SOURCE_B16_ROW_BUDGET - 1] + 1
        projected = parser_ids[:projected_end]
        status = "accepted_budget"
        projected_count = SOURCE_B16_ROW_BUDGET
    else:
        projected = list(parser_ids)
        projected_end = len(projected)
        status = "accepted_natural_end"
        projected_count = len(parsed)
    projected_hash = token_ids_sha256(projected)
    provenance = {
        "status": status,
        "row_budget": SOURCE_B16_ROW_BUDGET,
        "raw_generated_token_ids_sha256": raw_hash,
        "raw_generated_token_count": len(raw_ids),
        "raw_stop_reason": "im_end",
        "terminal_im_end_token_id": im_end_token_id,
        "box_end_token_id": box_end_token_id,
        "raw_valid_complete_row_count": len(parsed),
        "malformed_or_dropped_before_b16_count": malformed_before,
        "projected_valid_complete_row_count": projected_count,
        "projected_token_end_offset_exclusive": projected_end,
        "projected_token_ids_sha256": projected_hash,
        "projected_token_count": len(projected),
        "natural_end_before_or_at_budget": status == "accepted_natural_end",
    }
    return projected, projected_hash, provenance


def _validate_accepted_source_b16(
    row: Mapping[str, Any],
    receipt: Mapping[str, Any],
    *,
    im_end_token_id: int,
    box_end_token_id: int,
) -> None:
    """Replay accepted Source projection from raw tokens/parser chronology."""

    status = str(receipt.get("status", ""))
    if status not in SOURCE_B16_ACCEPTED_STATUSES:
        raise AssemblyError("Source replay validator requires an accepted status")
    raw_ids = _token_ids(row.get("generated_token_ids"), "Source.raw_generated_token_ids")
    raw_hash = _norm_sha(
        row.get("generated_token_ids_sha256"), "Source.raw_generated_token_ids_sha256"
    )
    if token_ids_sha256(raw_ids) != raw_hash:
        raise AssemblyError("Source raw generated-token hash mismatch")
    raw_parser = _mapping(row.get("predictions"), "Source.predictions")
    stored_raw_parser = _mapping(
        receipt.get("raw_parser_evidence"), "Source.raw_parser_evidence"
    )
    if dict(raw_parser) != dict(stored_raw_parser):
        raise AssemblyError("Source stored raw parser evidence differs from top-level parser")
    projected = _token_ids(
        receipt.get("projected_token_ids"), "Source.projected_token_ids"
    )
    end = receipt.get("projected_token_end_offset_exclusive")
    if isinstance(end, bool) or not isinstance(end, int) or end != len(projected):
        raise AssemblyError("Source projected token end offset is not its exact length")
    if projected != raw_ids[:end]:
        raise AssemblyError("Source projected token IDs are not an exact raw prefix")
    parsed, parser_evidence = _parsed_rows(row)
    malformed_before = _malformed_before_budget(
        parser_evidence, parsed, SOURCE_B16_ROW_BUDGET
    )
    if malformed_before:
        raise AssemblyError("accepted Source has malformed/drop evidence before B16")
    count = receipt.get("projected_valid_complete_row_count")
    if isinstance(count, bool) or not isinstance(count, int):
        raise AssemblyError("Source projected row count is invalid")
    projected_parser = _mapping(
        receipt.get("projected_parser_evidence"), "Source.projected_parser_evidence"
    )
    raw_predictions = raw_parser.get("predictions")
    projected_predictions = projected_parser.get("predictions")
    if (
        not isinstance(raw_predictions, list)
        or not isinstance(projected_predictions, list)
        or projected_predictions != raw_predictions[:count]
    ):
        raise AssemblyError("Source projected parser is not the exact raw-parser prefix")
    if status == "accepted_budget":
        box_ends = [
            index for index, token_id in enumerate(raw_ids) if token_id == box_end_token_id
        ]
        expected_end = (
            box_ends[SOURCE_B16_ROW_BUDGET - 1] + 1
            if len(box_ends) >= SOURCE_B16_ROW_BUDGET
            else None
        )
        if (
            count != SOURCE_B16_ROW_BUDGET
            or len(parsed) < SOURCE_B16_ROW_BUDGET
            or end != expected_end
            or not projected
            or projected[-1] != box_end_token_id
        ):
            raise AssemblyError("Source accepted_budget does not end exactly at row 16")
    elif (
        count >= SOURCE_B16_ROW_BUDGET
        or count != len(parsed)
        or row.get("stop_reason") != "im_end"
        or not raw_ids
        or raw_ids[-1] != im_end_token_id
        or raw_ids.count(im_end_token_id) != 1
        or projected != raw_ids[:-1]
    ):
        raise AssemblyError("Source accepted_natural_end lacks exact terminal im_end projection")


def _derive_v2_checkpoint_identity(payload: Mapping[str, Any], *, path: Path) -> CheckpointIdentity:
    model = _mapping(payload.get("model_identity"), f"{path}.model_identity")
    execution = _mapping(
        model.get("execution_model_identity"), f"{path}.execution_model_identity"
    )
    source = _mapping(execution.get("source_identity"), f"{path}.source_identity")
    adapter = _mapping(source.get("adapter"), f"{path}.source_identity.adapter")
    embedding = _mapping(
        source.get("embedding_delta"), f"{path}.source_identity.embedding_delta"
    )
    semantic = _mapping(
        embedding.get("semantic_identity"), f"{path}.embedding_delta.semantic_identity"
    )
    strings, ids = semantic.get("token_strings"), semantic.get("token_ids")
    if not isinstance(strings, list) or not isinstance(ids, list):
        raise AssemblyError(f"v2 execution receipt lacks selected token identity: {path}")
    selection = SpecialTokenSelection(token_strings=strings, token_ids=ids)
    tokenizer = _mapping(model.get("tokenizer_identity"), f"{path}.tokenizer_identity")
    processor = _mapping(model.get("processor_identity"), f"{path}.processor_identity")
    return CheckpointIdentity(
        adapter_fingerprint=_norm_sha(adapter.get("fingerprint"), f"{path}.adapter.fingerprint"),
        embedding_delta_fingerprint=_norm_sha(
            embedding.get("fingerprint"), f"{path}.embedding_delta.fingerprint"
        ),
        base_config_sha256=_norm_sha(
            semantic.get("base_config_sha256"), f"{path}.base_config_sha256"
        ),
        tokenizer_sha256=_norm_sha(
            semantic.get("tokenizer_sha256"), f"{path}.tokenizer_sha256"
        ),
        token_identity_sha256=sha256_json(dict(tokenizer)),
        special_token_identity_sha256=sha256_json(selection.to_artifact_dict()),
        processor_identity_sha256=sha256_json(dict(processor)),
    )


def _v2_row_identity(
    payload: Mapping[str, Any], row: Mapping[str, Any], path: Path, model_hash: str
) -> dict[str, Any]:
    image = _canonical_image_id(row.get("image_id"), field=f"{path}.rollout.image_id")
    example = str(row.get("example_id", ""))
    metadata_map = _mapping(payload.get("prompt_metadata"), f"{path}.prompt_metadata")
    metadata = _mapping(
        metadata_map.get(example, metadata_map.get(image)), f"{path}.prompt_metadata[{example}]"
    )
    prompt = _token_ids(row.get("prompt_token_ids"), f"{path}:{image}.prompt_token_ids")
    prompt_hash = _norm_sha(
        row.get("prompt_token_ids_sha256"), f"{path}:{image}.prompt_token_ids_sha256"
    )
    metadata_prompt = _token_ids(
        metadata.get("prompt_token_ids"), f"{path}:{image}.metadata.prompt_token_ids"
    )
    metadata_hash = _norm_sha(
        metadata.get("prompt_token_ids_sha256"),
        f"{path}:{image}.metadata.prompt_token_ids_sha256",
    )
    if (
        token_ids_sha256(prompt) != prompt_hash
        or token_ids_sha256(metadata_prompt) != metadata_hash
        or prompt != metadata_prompt
        or prompt_hash != metadata_hash
    ):
        raise AssemblyError(f"v2 prompt-token identity mismatch for image {image}")
    source_hash = _norm_sha(
        row.get("source_image_file_sha256"), f"{path}:{image}.source_image_file_sha256"
    )
    if source_hash != _norm_sha(
        metadata.get("source_image_file_sha256"),
        f"{path}:{image}.metadata.source_image_file_sha256",
    ):
        raise AssemblyError(f"v2 source-image identity mismatch for image {image}")
    width, height = metadata.get("width"), metadata.get("height")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (width, height)):
        raise AssemblyError(f"v2 prompt metadata lacks dimensions for image {image}")
    if row.get("image_width", width) != width or row.get("image_height", height) != height:
        raise AssemblyError(f"v2 image dimensions mismatch for image {image}")
    image_path = metadata.get("image_path")
    if not isinstance(image_path, str) or not image_path:
        raise AssemblyError(f"v2 prompt metadata lacks image_path for image {image}")
    return {
        "image_id": image,
        "example_id": example,
        "prompt_token_ids": prompt,
        "prompt_token_ids_sha256": prompt_hash,
        "source_image_file_sha256": source_hash,
        "executed_rgb_sha256": _norm_sha(
            row.get("executed_rgb_sha256"), f"{path}:{image}.executed_rgb_sha256"
        ),
        "width": width,
        "height": height,
        "image_path": str(Path(image_path).expanduser().resolve()),
        "execution_model_identity_sha256": model_hash,
    }


_V2_IDENTITY_FIELDS = (
    "prompt_token_ids",
    "prompt_token_ids_sha256",
    "source_image_file_sha256",
    "executed_rgb_sha256",
    "width",
    "height",
    "image_path",
    "execution_model_identity_sha256",
)


def _register_v2_identity(
    identities: dict[str, dict[str, Any]], observed: Mapping[str, Any]
) -> dict[str, Any]:
    image = str(observed["image_id"])
    previous = identities.setdefault(image, copy.deepcopy(dict(observed)))
    mismatch = [field for field in _V2_IDENTITY_FIELDS if previous[field] != observed[field]]
    if mismatch:
        raise AssemblyError(
            f"sampled/Source v2 identity mismatch for image {image}: {', '.join(mismatch)}"
        )
    return previous


def _v2_assignment(
    row: Mapping[str, Any], owners: Sequence[Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed, parser = _parsed_rows(row)
    assignment = match_prefix(parsed, owners, SOURCE_B16_ROW_BUDGET)
    malformed = _malformed_before_budget(parser, parsed, SOURCE_B16_ROW_BUDGET)
    duplicate = sum(
        item.get("entity_status") in {"duplicate", "duplicate_owner"}
        for item in assignment["row_assignment_receipts"]
    )
    unresolved_statuses = {
        "semantic_mismatch_unresolved",
        "unresolved_pending_crop_review",
        "ambiguous_matched_review",
        "uncertain",
    }
    unresolved = sum(
        item.get("entity_status") in unresolved_statuses
        for item in assignment["row_assignment_receipts"]
    )
    assignment.update(
        {
            "malformed_row_count": malformed,
            "harmful_row_count": malformed + duplicate,
            "row_counts": {
                "duplicate": duplicate,
                "malformed": malformed,
                "unsupported_hallucination": 0,
                "semantic_error": 0,
                "unresolved": unresolved,
            },
        }
    )
    return assignment, parser


def _compact_v2_rollout(
    row: Mapping[str, Any], path: Path, config: Mapping[str, Any], identity: Mapping[str, Any], seed: int
) -> dict[str, Any]:
    tokens = _token_ids(row.get("generated_token_ids"), f"{path}.generated_token_ids")
    token_hash = _norm_sha(row.get("generated_token_ids_sha256"), f"{path}.generated_token_ids_sha256")
    if token_ids_sha256(tokens) != token_hash:
        raise AssemblyError(f"v2 generated-token hash mismatch for image {identity['image_id']}")
    return {
        "image_id": str(identity["image_id"]),
        "example_id": str(identity["example_id"]),
        "trajectory_id": str(row.get("trajectory_id", "")),
        "decode_mode": str(row.get("decode_mode", "")),
        "seed": seed,
        "sample_index": row.get("sample_index"),
        "stop_reason": str(row.get("stop_reason", "")),
        "prompt_token_ids": identity["prompt_token_ids"],
        "prompt_token_ids_sha256": str(identity["prompt_token_ids_sha256"]),
        "source_image_file_sha256": str(identity["source_image_file_sha256"]),
        "executed_rgb_sha256": str(identity["executed_rgb_sha256"]),
        "image_width": int(identity["width"]),
        "image_height": int(identity["height"]),
        "generated_token_ids": tokens,
        "generated_token_ids_sha256": token_hash,
        "_source_path": str(path),
        "_temperature": float(config.get("temperature", 0.0)),
        "_top_p": float(config.get("top_p", 1.0)),
        "_repetition_penalty": float(config.get("repetition_penalty", 1.0)),
    }


def _v2_references(
    pool: Mapping[str, Mapping[str, Any]], pool_path: Path, identities: Mapping[str, Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for image, candidate in pool.items():
        identity = _mapping(identities.get(image), f"v2 identity {image}")
        images, metadata = candidate.get("images"), candidate.get("metadata")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise AssemblyError(f"candidate-pool image {image} lacks one image path")
        if not isinstance(metadata, Mapping) or metadata.get("split") != "train":
            raise AssemblyError(f"candidate-pool image {image} lacks train split metadata")
        path = Path(images[0])
        path = path.resolve() if path.is_absolute() else (pool_path.parent / path).resolve()
        if path != Path(str(identity["image_path"])):
            raise AssemblyError(f"candidate-pool image path differs from v2 panel for image {image}")
        if candidate.get("width") != identity["width"] or candidate.get("height") != identity["height"]:
            raise AssemblyError(f"candidate-pool dimensions differ from v2 panel for image {image}")
        prompt = list(identity["prompt_token_ids"])
        result[image] = {
            "image": {
                "image_id": int(image),
                "path": str(path),
                "width": int(identity["width"]),
                "height": int(identity["height"]),
                "content_sha256": str(identity["source_image_file_sha256"]),
            },
            "split": "train",
            "executed_prompt_token_ids": prompt,
            "executed_prompt_token_ids_sha256": str(identity["prompt_token_ids_sha256"]),
            "image_pad_interval": list(_image_pad_interval(prompt)),
        }
    return result


def load_v2_b16_panel_adapter(
    *,
    sampled_panel_root: Path,
    source_b16_root: Path,
    candidate_pool: Path,
    semantic_image_ids: Sequence[str] | None = None,
    sampled_artifact_inventory: Sequence[Mapping[str, Any]] | None = None,
    source_artifact_inventory: Sequence[Mapping[str, Any]] | None = None,
    expected_candidate_pool_sha256: str | None = None,
) -> dict[str, Any]:
    """Adapt v2 sampled + Source@B16 artifacts to the existing candidate seam.

    ``semantic_image_ids`` is a split barrier: excluded rows are inspected only
    for ``image_id`` and are skipped before stop, parser, text, owner, or route
    semantics are read. Strict callers pass both manifest-bound artifact
    inventories; recursive discovery must then match those paths exactly, and
    each artifact hash is checked from the same bytes decoded as JSON.
    """

    pool_path = candidate_pool.expanduser().resolve(strict=True)
    candidate_pool_sha256 = sha256_file(pool_path)
    if (
        expected_candidate_pool_sha256 is not None
        and candidate_pool_sha256 != expected_candidate_pool_sha256
    ):
        raise AssemblyError("candidate pool differs before v2 adapter decoding")
    pool = _candidate_pool(pool_path)
    semantic_ids = (
        set(pool)
        if semantic_image_ids is None
        else {_canonical_image_id(item, field="semantic_image_ids") for item in semantic_image_ids}
    )
    if not semantic_ids <= set(pool):
        raise AssemblyError("semantic image filter is not a candidate-pool subset")
    owners = load_generation7_annotations(pool_path, image_ids=semantic_ids)
    if sha256_file(pool_path) != candidate_pool_sha256:
        raise AssemblyError("candidate pool changed while loading the v2 adapter")
    if set(owners) != semantic_ids:
        raise AssemblyError("candidate-pool owner records do not cover the exact v2 image cohort")
    sampled_paths, sampled_expected_hashes = _resolve_v2_artifact_inventory(
        sampled_panel_root,
        mode="sampled",
        strict_inventory=sampled_artifact_inventory,
    )
    source_paths, source_expected_hashes = _resolve_v2_artifact_inventory(
        source_b16_root,
        mode="source_b16",
        strict_inventory=source_artifact_inventory,
    )
    identities: dict[str, dict[str, Any]] = {}
    sampled_rows: dict[tuple[str, int], dict[str, Any]] = {}
    source_rows: dict[tuple[str, int], dict[str, Any]] = {}
    routes: dict[str, dict[str, dict[str, Any]]] = {}
    source_status: dict[str, str] = {}
    source_provenance: dict[str, dict[str, Any]] = {}
    execution: dict[str, Any] | None = None
    execution_hash: str | None = None
    tokenizer_identity: dict[str, Any] | None = None
    tokenizer_hash: str | None = None
    im_end_token_id: int | None = None
    box_end_token_id: int | None = None
    representative: dict[str, Any] | None = None
    artifact_provenance: list[dict[str, Any]] = []

    def artifact(path: Path, mode: str) -> tuple[dict[str, Any], Mapping[str, Any], str, str]:
        nonlocal execution, execution_hash, representative
        nonlocal tokenizer_identity, tokenizer_hash, im_end_token_id, box_end_token_id
        expected_hashes = (
            sampled_expected_hashes if mode == "sampled" else source_expected_hashes
        )
        decoded, digest = _read_json_exact_bytes(
            path, expected_sha256=expected_hashes.get(path)
        )
        payload = dict(_mapping(decoded, f"v2 artifact {path}"))
        if payload.get("schema_version") != V2_PANEL_SCHEMA_VERSION:
            raise AssemblyError(f"unsupported v2 panel schema in {path}")
        config = _mapping(payload.get("config"), f"{path}.config")
        if config.get("decode_mode") != mode:
            raise AssemblyError(f"v2 artifact has wrong decode mode in {path}")
        current, current_hash = _v2_execution_identity(payload, path)
        current_tokenizer, current_tokenizer_hash, current_im_end, current_box_end = (
            _v2_tokenizer_contract(payload, path)
        )
        if execution_hash is None:
            execution, execution_hash = current, current_hash
            representative = {"model_identity": copy.deepcopy(payload["model_identity"])}
        elif current_hash != execution_hash or current != execution:
            raise AssemblyError("sampled/Source v2 execution-model identity mismatch")
        if tokenizer_hash is None:
            tokenizer_identity = current_tokenizer
            tokenizer_hash = current_tokenizer_hash
            im_end_token_id = current_im_end
            box_end_token_id = current_box_end
        elif (
            current_tokenizer_hash != tokenizer_hash
            or current_tokenizer != tokenizer_identity
            or current_im_end != im_end_token_id
            or current_box_end != box_end_token_id
        ):
            raise AssemblyError("sampled/Source v2 tokenizer identity mismatch")
        rows = payload.get("rollouts")
        if not isinstance(rows, list) or not rows or payload.get("rollout_count") not in {None, len(rows)}:
            raise AssemblyError(f"v2 artifact lacks consistent rollout rows: {path}")
        artifact_provenance.append(
            {"mode": mode, "path": str(path), "sha256": digest, "execution_model_identity_sha256": current_hash}
        )
        return payload, config, current_hash, digest

    for path in sampled_paths:
        payload, config, model_hash, _ = artifact(path, "sampled")
        if (
            config.get("panel_mode") != "sampled_only"
            or config.get("sample_count") != 16
            or config.get("sample_index_range") != [0, 15]
            or (float(config.get("temperature", -1)), float(config.get("top_p", -1)), float(config.get("repetition_penalty", -1)))
            != (0.4, 0.95, 1.0)
        ):
            raise AssemblyError(f"sampled v2 artifact has non-canonical panel config: {path}")
        for raw in payload["rollouts"]:
            row = _mapping(raw, f"{path}.rollout")
            image = _canonical_image_id(
                row.get("image_id"), field=f"{path}.rollout.image_id"
            )
            if image not in semantic_ids:
                continue
            identity = _register_v2_identity(
                identities, _v2_row_identity(payload, row, path, model_hash)
            )
            image = str(identity["image_id"])
            index = row.get("sample_index")
            if isinstance(index, bool) or not isinstance(index, int) or index not in range(16):
                raise AssemblyError(f"sampled v2 row has invalid sample_index for image {image}")
            if row.get("decode_mode") != "sampled" or row.get("stop_reason") != "im_end":
                raise AssemblyError(f"sampled v2 row is not naturally closed for image {image}")
            if (image, index) in sampled_rows:
                raise AssemblyError(f"duplicate sampled v2 image/sample_index: {image}/{index}")
            normalized = _compact_v2_rollout(row, path, config, identity, index)
            normalized["trajectory_id"] = f"sample-{index:02d}"
            normalized["predictions"] = row.get("predictions")
            assert im_end_token_id is not None and box_end_token_id is not None
            projected, projected_hash, sampled_b16 = _project_sampled_b16(
                normalized,
                im_end_token_id=im_end_token_id,
                box_end_token_id=box_end_token_id,
            )
            normalized["generated_token_ids"] = projected
            normalized["generated_token_ids_sha256"] = projected_hash
            normalized["_sampled_b16_provenance"] = sampled_b16
            assignment, parser = _v2_assignment(normalized, owners[image])
            normalized.pop("predictions", None)
            sampled_rows[(image, index)] = normalized
            routes.setdefault(image, {})[normalized["trajectory_id"]] = {
                "assignment": assignment,
                "parser": parser,
                "stop_reason": "im_end",
                "seed": index,
                "decode_mode": "sampled",
                "sampled_b16": copy.deepcopy(sampled_b16),
            }
    observed: dict[str, set[int]] = {}
    for image, index in sampled_rows:
        observed.setdefault(image, set()).add(index)
    if set(observed) != semantic_ids:
        raise AssemblyError("sampled v2 panel image cohort differs from the semantic image filter")
    incomplete = {image: values for image, values in observed.items() if values != set(range(16))}
    if incomplete:
        image = min(incomplete, key=int)
        raise AssemblyError(f"sampled v2 image does not have sample_index 0..15: {image}")

    for path in source_paths:
        payload, config, model_hash, artifact_hash = artifact(path, "source_b16")
        if (
            config.get("panel_mode") != "source_b16"
            or config.get("source_b16_row_budget") != 16
            or (float(config.get("temperature", -1)), float(config.get("top_p", -1)), float(config.get("repetition_penalty", -1)))
            != (0.0, 1.0, 1.0)
        ):
            raise AssemblyError(f"Source@B16 artifact has non-canonical panel config: {path}")
        for raw in payload["rollouts"]:
            row = _mapping(raw, f"{path}.rollout")
            image = _canonical_image_id(
                row.get("image_id"), field=f"{path}.rollout.image_id"
            )
            if image not in semantic_ids:
                continue
            identity = _register_v2_identity(
                identities, _v2_row_identity(payload, row, path, model_hash)
            )
            image = str(identity["image_id"])
            if image in source_status:
                raise AssemblyError(f"duplicate Source@B16 image: {image}")
            receipt = _mapping(row.get("source_b16"), f"{path}:{image}.source_b16")
            status = str(receipt.get("status", ""))
            if (
                row.get("decode_mode") != "source_b16"
                or row.get("trajectory_id") != "source-b16"
                or status not in SOURCE_B16_STATUSES
                or receipt.get("row_budget") != 16
            ):
                raise AssemblyError(f"invalid Source@B16 contract for image {image}")
            raw_ids = _token_ids(row.get("generated_token_ids"), f"{path}:{image}.generated_token_ids")
            raw_hash = _norm_sha(
                row.get("generated_token_ids_sha256"), f"{path}:{image}.generated_token_ids_sha256"
            )
            if token_ids_sha256(raw_ids) != raw_hash:
                raise AssemblyError(f"raw Source token hash mismatch for image {image}")
            provenance = {
                "status": status,
                "source_artifact_path": str(path),
                "source_artifact_sha256": artifact_hash,
                "raw_generated_token_ids_sha256": raw_hash,
                "raw_generated_token_count": len(raw_ids),
                "raw_stop_reason": str(row.get("stop_reason", "")),
            }
            source_status[image], source_provenance[image] = status, provenance
            if status not in SOURCE_B16_ACCEPTED_STATUSES:
                continue
            assert im_end_token_id is not None and box_end_token_id is not None
            _validate_accepted_source_b16(
                row,
                receipt,
                im_end_token_id=im_end_token_id,
                box_end_token_id=box_end_token_id,
            )
            projected = _token_ids(
                receipt.get("projected_token_ids"), f"{path}:{image}.projected_token_ids"
            )
            projected_hash = _norm_sha(
                receipt.get("projected_token_ids_sha256"), f"{path}:{image}.projected_token_ids_sha256"
            )
            parser = receipt.get("projected_parser_evidence")
            text = receipt.get("projected_text")
            count = receipt.get("projected_valid_complete_row_count")
            if (
                token_ids_sha256(projected) != projected_hash
                or not isinstance(text, str)
                or not isinstance(parser, Mapping)
                or parser.get("parse_status") != "accepted"
                or parser.get("dropped_prediction_count") != 0
                or isinstance(count, bool)
                or not isinstance(count, int)
                or not 0 <= count <= 16
                or parser.get("valid_prediction_count") != count
                or (status == "accepted_budget" and count != 16)
            ):
                raise AssemblyError(f"Source@B16 projection is invalid for image {image}")
            normalized = _compact_v2_rollout(
                {**dict(row), "generated_token_ids": projected, "generated_token_ids_sha256": projected_hash},
                path,
                config,
                identity,
                0,
            )
            normalized.update(
                {
                    "trajectory_id": "source-b16",
                    "decode_mode": "greedy",
                    "generated_text": text,
                    "predictions": parser,
                    "_source_b16_provenance": {
                        **provenance,
                        "projected_token_ids_sha256": projected_hash,
                        "projected_token_count": len(projected),
                        "projected_valid_complete_row_count": count,
                    },
                }
            )
            assignment, parser_evidence = _v2_assignment(normalized, owners[image])
            normalized.pop("predictions", None)
            normalized.pop("generated_text", None)
            source_rows[(image, 0)] = normalized
            routes.setdefault(image, {})["source-b16"] = {
                "assignment": assignment,
                "parser": parser_evidence,
                "stop_reason": str(row.get("stop_reason", "")),
                "seed": 0,
                "decode_mode": "greedy",
            }
    if set(source_status) != semantic_ids or set(identities) != semantic_ids:
        raise AssemblyError("Source@B16 image cohort differs from the semantic image filter")

    image_results: dict[str, dict[str, Any]] = {}
    sampled_ids = [f"sample-{index:02d}" for index in range(16)]
    for image, _ in sorted(source_rows, key=lambda item: int(item[0])):
        route_map = routes[image]
        assignments = {
            route: copy.deepcopy(route_map[route]["assignment"])
            for route in ["source-b16", *sampled_ids]
        }
        image_results[image] = {
            "image_id": image,
            "greedy_trajectory_id": "source-b16",
            "sampled_trajectory_ids": sampled_ids,
            "trajectory_evidence": {
                route: {
                    "decode_mode": route_map[route]["decode_mode"],
                    "seed": route_map[route]["seed"],
                    "stop_reason": route_map[route]["stop_reason"],
                    "parser": copy.deepcopy(route_map[route]["parser"]),
                }
                for route in ["source-b16", *sampled_ids]
            },
            "owners": copy.deepcopy(owners[image]),
            "budgets": [
                {
                    "budget": 16,
                    "trajectory_assignments": assignments,
                    "owner_sets": {
                        route: list(value["matched_owner_ids"])
                        for route, value in assignments.items()
                    },
                }
            ],
        }
    references = _v2_references(
        {image: pool[image] for image in semantic_ids}, pool_path, identities
    )
    counts = Counter(source_status.values())
    assert execution is not None and execution_hash is not None and representative is not None
    assert tokenizer_identity is not None and tokenizer_hash is not None
    return {
        "sampled_paths": sampled_paths,
        "source_paths": source_paths,
        "candidate_pool_sha256": candidate_pool_sha256,
        "sampled_rows": sampled_rows,
        "source_rows": source_rows,
        "image_results": image_results,
        "reference_records": references,
        "execution_model_identity": execution,
        "execution_model_identity_sha256": execution_hash,
        "tokenizer_identity": tokenizer_identity,
        "tokenizer_identity_sha256": tokenizer_hash,
        "representative_payload": representative,
        "prompt_identity_sha256": sha256_json(
            [[image, identities[image]["prompt_token_ids_sha256"]] for image in sorted(identities, key=int)]
        ),
        "execution_metadata": {
            "panel_schema_version": V2_PANEL_SCHEMA_VERSION,
            "sampled_panel_mode": "sampled_only",
            "source_panel_mode": "source_b16",
            "sampling_order": "request_major",
            "sample_index_range": [0, 15],
            "sample_count": 16,
            "source_b16_row_budget": 16,
            "execution_model_identity_sha256": execution_hash,
        },
        "census": {
            "sampled_image_count": len(observed),
            "sampled_trajectory_count": len(sampled_rows),
            "source_image_count": len(source_status),
            "source_status_counts": dict(sorted(counts.items())),
            "source_accepted_image_count": sum(counts[item] for item in SOURCE_B16_ACCEPTED_STATUSES),
            "source_ineligible_image_count": sum(counts[item] for item in SOURCE_B16_INELIGIBLE_STATUSES),
            "source_ineligible_excluded_from_admission_count": sum(
                counts[item] for item in SOURCE_B16_INELIGIBLE_STATUSES
            ),
            "source_rows": [
                {"image_id": image, **source_provenance[image]}
                for image in sorted(source_provenance, key=int)
            ],
            "artifact_provenance": artifact_provenance,
        },
    }


def _selection_receipt(selection: Mapping[str, Any]) -> dict[str, Any]:
    """Project selection to immutable JSON without serializing internal inputs twice."""

    arms: dict[str, Any] = {}
    for name in ("broad", "concentrated"):
        arm = _mapping(selection.get(name), f"selection.{name}")
        arms[name] = {
            key: copy.deepcopy(arm[key])
            for key in (
                "image_ids", "event_count", "sampled_event_count", "source_event_count",
                "mean_event_weight", "total_event_weight", "image_family_event_counts",
                "selection_distributions", "trajectory_panel_execution_metadata",
                "rank_matching_receipt",
            )
        }
        if "allocation_protocol_receipt" in arm:
            arms[name]["allocation_protocol_receipt"] = copy.deepcopy(
                arm["allocation_protocol_receipt"]
            )
        arms[name]["events"] = [
            {
                key: copy.deepcopy(event[key])
                for key in (
                    "event_id", "image_id", "event_family", "object_count_band", "selection_rank",
                    "route_id", "route_seed", "generated_row_index", "owner_id",
                    "prefix_token_ids_sha256", "candidate_token_ids_sha256", "route_count", "row_depth",
                    "complete_row_coordinate_token_supervision",
                    "image_balanced_event_weight",
                )
            }
            for event in arm["events"]
        ]
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "frozen_selection",
        "trajectory_panel_execution_metadata": copy.deepcopy(selection["trajectory_panel_execution_metadata"]),
        "eligible_unique_training_image_count": selection["eligible_unique_training_image_count"],
        "eligible_images_by_band": copy.deepcopy(selection["eligible_images_by_band"]),
        "broad_band_quota": copy.deepcopy(selection["broad_band_quota"]),
        "concentrated_band_quota": copy.deepcopy(selection["concentrated_band_quota"]),
        "pair_quota_by_object_count_band": copy.deepcopy(
            selection["pair_quota_by_object_count_band"]
        ),
        "rank_matching_receipt": copy.deepcopy(selection["rank_matching_receipt"]),
        "arms": arms,
    }
    if "allocation_protocol_receipt" in selection:
        receipt["allocation_protocol_receipt"] = copy.deepcopy(
            selection["allocation_protocol_receipt"]
        )
    return receipt


def _deduplicate_v2_sampled_candidates(
    candidates_by_image: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Collapse v2 cross-route exact identities before shared selection.

    The shared selector remains fail-closed on clones.  This adapter-local
    normalization treats identical rows from different sampled trajectories
    as one available event and keeps the route preferred by the frozen unit
    ranking.
    """

    deduplicated: dict[str, list[dict[str, Any]]] = {}
    duplicate_groups: list[dict[str, Any]] = []

    def descriptor(item: Mapping[str, Any]) -> dict[str, Any]:
        return {
            field: copy.deepcopy(item.get(field))
            for field in (
                "route_id",
                "route_seed",
                "generated_row_index",
                "owner_id",
                "marginal_route_added_owner_count",
                "route_added_owner_count",
                "unresolved_row_count",
                "event_id",
                "prefix_token_ids_sha256",
                "candidate_token_ids_sha256",
            )
        }

    for image in sorted((str(value) for value in candidates_by_image), key=int):
        groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
        for raw in candidates_by_image[image]:
            item = dict(raw)
            groups.setdefault(_breadth_identity(item), []).append(item)
        retained: list[dict[str, Any]] = []
        for identity, group in sorted(groups.items()):
            ordered = sorted(group, key=_breadth_sampled_rank)
            retained.append(ordered[0])
            if len(ordered) > 1:
                duplicate_groups.append(
                    {
                        "image_id": image,
                        "identity": {
                            "prefix_token_ids_sha256": identity[1],
                            "candidate_token_ids_sha256": identity[2],
                            "owner_id": identity[3],
                        },
                        "candidate_count": len(ordered),
                        "retained": descriptor(ordered[0]),
                        "discarded": [descriptor(item) for item in ordered[1:]],
                    }
                )
        deduplicated[image] = sorted(retained, key=_breadth_sampled_rank)
    before = sum(len(values) for values in candidates_by_image.values())
    after = sum(len(values) for values in deduplicated.values())
    return deduplicated, {
        "policy": "v2_exact_identity_keep_existing_sampled_rank_winner",
        "candidate_count_before": before,
        "candidate_count_after": after,
        "duplicate_candidate_count_removed": before - after,
        "duplicate_identity_group_count": len(duplicate_groups),
        "affected_image_count": len({item["image_id"] for item in duplicate_groups}),
        "duplicate_groups": duplicate_groups,
    }


def _materialize_v2_arm(
    selection: Mapping[str, Any], *, checkpoint_id: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    assembled = materialize_constant_dose_breadth_arm(selection, checkpoint_id=checkpoint_id)
    provenance: dict[str, dict[str, Any]] = {}
    for event in selection.get("events", []):
        if not isinstance(event, Mapping) or event.get("event_family") != "source_preservation":
            continue
        inputs = event.get("_event_inputs")
        rollout = inputs.get("rollout") if isinstance(inputs, Mapping) else None
        source = rollout.get("_source_b16_provenance") if isinstance(rollout, Mapping) else None
        if not isinstance(source, Mapping):
            raise AssemblyError("selected Source@B16 event lacks raw completion provenance")
        event_id = (
            f"source-preservation-image-{event['image_id']}-route-{event['route_id']}-"
            f"row-{event['generated_row_index']}"
        )
        provenance[event_id] = copy.deepcopy(dict(source))
    rollouts, reviews, receipts, arm = assembled
    for rollout, review, receipt in zip(rollouts, reviews, receipts, strict=True):
        source = provenance.get(str(receipt.get("event_id")))
        if source is None:
            continue
        review_provenance = dict(_mapping(review.get("review_provenance"), "review_provenance"))
        review_provenance["source_b16"] = copy.deepcopy(source)
        review["review_provenance"], receipt["source_b16"] = review_provenance, copy.deepcopy(source)
    return rollouts, reviews, receipts, arm


def _training_surfaces(
    training_ids: set[str], candidate_rows: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    bands: dict[str, str] = {}
    annotations: dict[str, dict[str, Any]] = {}
    for image in training_ids:
        row = candidate_rows[image]
        objects = row.get("objects")
        if not isinstance(objects, list) or not objects:
            raise AssemblyError(f"candidate-pool training image {image} has no annotation objects")
        count = len(objects)
        bands[image] = (
            "sparse_1_to_3" if count <= 3 else "medium_4_to_7" if count <= 7
            else "dense_8_to_15" if count <= 15 else "very_dense_16_plus"
        )
        annotations[image] = dict(row)
    return bands, annotations


def assemble_v2_b16_constant_dose_breadth_state_banks(
    *,
    candidate_pool: Path,
    split_receipt: Path,
    sampled_panel_root: Path,
    source_b16_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    output = output_dir.expanduser().resolve()
    if output.exists():
        raise AssemblyError(f"output path already exists and will not be overwritten: {output}")
    pool_path = candidate_pool.expanduser().resolve(strict=True)
    split_path = split_receipt.expanduser().resolve(strict=True)
    split_ids = _split_membership(candidate_pool=pool_path, split_receipt=split_path)
    candidate_rows = _candidate_pool(pool_path)
    adapter = load_v2_b16_panel_adapter(
        sampled_panel_root=sampled_panel_root,
        source_b16_root=source_b16_root,
        candidate_pool=pool_path,
    )
    training_ids = split_ids["train_candidate"]
    bands, annotations = _training_surfaces(training_ids, candidate_rows)
    admitted = sorted(training_ids & set(adapter["image_results"]), key=int)
    status_by_image = {
        str(item["image_id"]): str(item["status"])
        for item in adapter["census"]["source_rows"]
    }
    training_status = Counter(status_by_image[image] for image in training_ids)
    adapter_census = copy.deepcopy(adapter["census"])
    adapter_census.update(
        {
            "training_source_status_counts": dict(sorted(training_status.items())),
            "training_source_accepted_image_count": len(admitted),
            "training_source_ineligible_excluded_from_admission_count": sum(
                training_status[item] for item in SOURCE_B16_INELIGIBLE_STATUSES
            ),
            "development_admission_count": 0,
            "heldout_admission_count": 0,
        }
    )
    representative_path = adapter["sampled_paths"][0]
    source_checkpoint = _derive_v2_checkpoint_identity(
        adapter["representative_payload"], path=representative_path
    )
    checkpoint_id = sha256_json(source_checkpoint.to_artifact_dict())
    sampled_candidates, sampled_census = _build_multi_candidates(
        image_results=adapter["image_results"],
        sampled_rows=adapter["sampled_rows"],
        reference_records=adapter["reference_records"],
        annotations=annotations,
        image_ids=admitted,
        checkpoint_id=checkpoint_id,
    )
    sampled_candidates, sampled_deduplication = _deduplicate_v2_sampled_candidates(
        sampled_candidates
    )
    sampled_census["v2_exact_identity_deduplication"] = copy.deepcopy(
        sampled_deduplication
    )
    adapter_census["sampled_exact_identity_deduplication"] = copy.deepcopy(
        sampled_deduplication
    )
    source_candidates, source_census = _build_source_candidates(
        image_results=adapter["image_results"],
        greedy_rows=adapter["source_rows"],
        reference_records=adapter["reference_records"],
        annotations=annotations,
        image_ids=admitted,
        manual_review=None,
    )
    validate_constant_dose_training_reservoir(
        training_image_bands=bands,
        development_image_ids=sorted(split_ids["development"], key=int),
        heldout_image_ids=sorted(split_ids["heldout"], key=int),
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
    )
    selection = select_constant_dose_breadth_arms(
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
        training_image_bands=bands,
        trajectory_panel_execution_metadata=adapter["execution_metadata"],
    )
    arms = {
        name: _materialize_v2_arm(selection[name], checkpoint_id=checkpoint_id)
        for name in ("broad", "concentrated")
    }
    source_paths = [
        ("candidate-pool", pool_path),
        ("split-receipt", split_path),
        *[(f"sampled-v2-{index:04d}", path) for index, path in enumerate(adapter["sampled_paths"])],
        *[(f"source-b16-v2-{index:04d}", path) for index, path in enumerate(adapter["source_paths"])],
    ]
    artifacts = _source_artifacts(source_paths)
    verified = [
        {
            "artifact_path": str(path),
            "checkpoint_id": checkpoint_id,
            "checkpoint_identity": source_checkpoint.to_artifact_dict(),
            "execution_model_identity_sha256": adapter["execution_model_identity_sha256"],
        }
        for path in [*adapter["sampled_paths"], *adapter["source_paths"]]
    ]
    output.mkdir(parents=True)
    selection_document = _selection_receipt(selection)
    selection_document.update({"source_artifacts": artifacts, "v2_panel_adapter": adapter_census})
    _write_json(output / "selection-receipt.json", selection_document)
    common_census = {
        "v2_panel_adapter": adapter_census,
        "sampled_candidates": sampled_census,
        "source_candidates": source_census,
        "selection": selection_document,
    }
    arm_receipts: dict[str, Any] = {}
    for name, assembled in arms.items():
        arm_receipts[name] = _write_arm(
            root=output / f"{name}-plus-source-preservation",
            rollouts=assembled[0],
            reviews=assembled[1],
            receipts=assembled[2],
            census={**common_census, "arm": assembled[3]},
            arm_receipt=assembled[3],
            source_checkpoint=source_checkpoint,
            prompt_identity_sha256=str(adapter["prompt_identity_sha256"]),
            source_artifacts=artifacts,
            verified_rollout_checkpoint_identities=verified,
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "assembled",
        "input_mode": "v2_sampled_panel_plus_source_b16",
        "source_checkpoint": source_checkpoint.to_artifact_dict(),
        "source_checkpoint_id": checkpoint_id,
        "prompt_identity_sha256": adapter["prompt_identity_sha256"],
        "execution_metadata": adapter["execution_metadata"],
        "reference_derivation": {
            "policy": "exact_v2_prompt_ids_plus_matched_source_file_and_executed_rgb_hashes",
            "image_count": len(adapter["reference_records"]),
            "sampled_and_source_b16_identity_cross_checked": True,
            "source_semantics": "projected_token_ids_text_and_parser_evidence_only",
        },
        "v2_panel_adapter": adapter_census,
        "selection_receipt": str((output / "selection-receipt.json").resolve()),
        "arms": {
            name: str((output / f"{name}-plus-source-preservation" / "assembly-receipt.json").resolve())
            for name in arm_receipts
        },
        "source_artifacts": artifacts,
    }
    _write_json(output / "assembly-receipt.json", receipt)
    return receipt


def assemble_constant_dose_breadth_state_banks(
    *,
    candidate_pool: Path,
    split_receipt: Path,
    trajectory_analysis: Path | None = None,
    panel_union_receipt: Path | None = None,
    old_greedy: Sequence[Path] | None = None,
    new_greedy: Sequence[Path] | None = None,
    old_sampled: Sequence[Path] | None = None,
    new_sampled: Sequence[Path] | None = None,
    sampled_panel_root: Path | None = None,
    source_b16_root: Path | None = None,
    output_dir: Path,
) -> dict[str, Any]:
    """Validate frozen evidence and write two immutable matched StateBanks."""

    legacy = (
        trajectory_analysis,
        panel_union_receipt,
        old_greedy,
        new_greedy,
        old_sampled,
        new_sampled,
    )
    if sampled_panel_root is not None or source_b16_root is not None:
        if sampled_panel_root is None or source_b16_root is None:
            raise AssemblyError("--sampled-panel-root and --source-b16-root must be supplied together")
        if any(value is not None for value in legacy):
            raise AssemblyError("v2 panel roots are mutually exclusive with legacy analysis/rollout inputs")
        return assemble_v2_b16_constant_dose_breadth_state_banks(
            candidate_pool=candidate_pool,
            split_receipt=split_receipt,
            sampled_panel_root=sampled_panel_root,
            source_b16_root=source_b16_root,
            output_dir=output_dir,
        )
    if any(value is None for value in legacy):
        raise AssemblyError(
            "legacy mode requires trajectory analysis, panel receipt, and all old/new rollout families"
        )
    assert trajectory_analysis is not None and panel_union_receipt is not None
    assert old_greedy is not None and new_greedy is not None
    assert old_sampled is not None and new_sampled is not None

    output = output_dir.expanduser().resolve()
    if output.exists():
        raise AssemblyError(f"output path already exists and will not be overwritten: {output}")
    pool_path = candidate_pool.expanduser().resolve(strict=True)
    split_path = split_receipt.expanduser().resolve(strict=True)
    analysis_path = trajectory_analysis.expanduser().resolve(strict=True)
    union_path = panel_union_receipt.expanduser().resolve(strict=True)
    old_greedy_paths, new_greedy_paths = _expand_paths(old_greedy), _expand_paths(new_greedy)
    old_sampled_paths, new_sampled_paths = _expand_paths(old_sampled), _expand_paths(new_sampled)
    if not all((old_greedy_paths, new_greedy_paths, old_sampled_paths, new_sampled_paths)):
        raise AssemblyError("old/new greedy and sampled rollout artifacts are all required")

    panel_receipt = _panel_receipt(union_path)
    recomputed_union = validate_panel_union(
        candidate_pool=pool_path,
        split_receipt=split_path,
        old_greedy=old_greedy_paths,
        old_sampled=old_sampled_paths,
        new_greedy=new_greedy_paths,
        new_sampled=new_sampled_paths,
    )
    _validate_panel_receipt_match(panel_receipt, recomputed_union)
    analysis = _read_json(analysis_path)
    if not isinstance(analysis, Mapping) or analysis.get("schema_version") != ROUTE_ANALYSIS_SCHEMA_VERSION:
        raise AssemblyError("unsupported trajectory union analysis schema")
    _validate_trajectory_analysis_provenance(
        analysis,
        candidate_pool=pool_path,
        rollout_paths=[
            *old_greedy_paths,
            *new_greedy_paths,
            *old_sampled_paths,
            *new_sampled_paths,
        ],
        validated_union=recomputed_union,
    )
    split_ids = _split_membership(candidate_pool=pool_path, split_receipt=split_path)
    candidate_rows = _candidate_pool(pool_path)
    all_greedy_paths = [*old_greedy_paths, *new_greedy_paths]
    all_sampled_paths = [*old_sampled_paths, *new_sampled_paths]
    greedy_rows, _ = _load_rollouts(all_greedy_paths, sampled=False)
    sampled_rows, _ = _load_rollouts(all_sampled_paths, sampled=True)
    prompt_evidence = {
        **_rollout_prompt_evidence(paths=all_greedy_paths, rows=greedy_rows),
        **_rollout_prompt_evidence(paths=all_sampled_paths, rows=sampled_rows),
    }
    reference_records = derive_reference_records(
        candidate_rows=candidate_rows,
        candidate_pool_path=pool_path,
        greedy_rows=greedy_rows,
        sampled_rows=sampled_rows,
        prompt_evidence=prompt_evidence,
    )
    identities = [
        derive_checkpoint_identity_from_rollout_artifact(_mapping(_read_json(path), f"rollout {path}"), artifact_path=path)
        for path in [*all_greedy_paths, *all_sampled_paths]
    ]
    source_checkpoint = identities[0]
    if any(identity != source_checkpoint for identity in identities[1:]):
        raise AssemblyError("old/new rollout artifacts do not share one exact checkpoint identity")
    checkpoint_id = sha256_json(source_checkpoint.to_artifact_dict())

    raw_results = analysis.get("image_results")
    if not isinstance(raw_results, list):
        raise AssemblyError("trajectory analysis lacks image_results")
    image_results = {
        _image_id(_mapping(item, "trajectory analysis image result").get("image_id")): _mapping(item, "trajectory analysis image result")
        for item in raw_results
    }
    if len(image_results) != len(raw_results):
        raise AssemblyError("trajectory analysis contains duplicate image IDs")
    training_ids = split_ids["train_candidate"]
    missing_analysis = sorted(training_ids - set(image_results), key=int)
    if missing_analysis:
        raise AssemblyError(f"trajectory analysis lacks training images: {missing_analysis[:8]}")
    training_bands: dict[str, str] = {}
    annotations: dict[str, dict[str, Any]] = {}
    for image in training_ids:
        row = candidate_rows[image]
        objects = row.get("objects")
        if not isinstance(objects, list) or not objects:
            raise AssemblyError(f"candidate-pool training image {image} has no annotation objects")
        count = len(objects)
        training_bands[image] = (
            "sparse_1_to_3" if count <= 3 else "medium_4_to_7" if count <= 7
            else "dense_8_to_15" if count <= 15 else "very_dense_16_plus"
        )
        annotations[image] = row
    sampled_candidates, sampled_census = _build_multi_candidates(
        image_results=image_results,
        sampled_rows=sampled_rows,
        reference_records=reference_records,
        annotations=annotations,
        image_ids=sorted(training_ids, key=int),
        checkpoint_id=checkpoint_id,
    )
    source_candidates, source_census = _build_source_candidates(
        image_results=image_results,
        greedy_rows=greedy_rows,
        reference_records=reference_records,
        annotations=annotations,
        image_ids=sorted(training_ids, key=int),
        manual_review=None,
    )
    validate_constant_dose_training_reservoir(
        training_image_bands=training_bands,
        development_image_ids=sorted(split_ids["development"], key=int),
        heldout_image_ids=sorted(split_ids["heldout"], key=int),
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
    )
    selection = select_constant_dose_breadth_arms(
        sampled_candidates=sampled_candidates,
        source_candidates=source_candidates,
        training_image_bands=training_bands,
        trajectory_panel_execution_metadata=panel_receipt["execution_metadata"],
    )
    broad = materialize_constant_dose_breadth_arm(selection["broad"], checkpoint_id=checkpoint_id)
    concentrated = materialize_constant_dose_breadth_arm(selection["concentrated"], checkpoint_id=checkpoint_id)

    source_paths = [
        ("candidate-pool", pool_path), ("split-receipt", split_path),
        ("trajectory-analysis", analysis_path), ("trajectory-union-receipt", union_path),
        *[(f"old-greedy-{index:02d}", path) for index, path in enumerate(old_greedy_paths)],
        *[(f"new-greedy-{index:02d}", path) for index, path in enumerate(new_greedy_paths)],
        *[(f"old-sampled-{index:02d}", path) for index, path in enumerate(old_sampled_paths)],
        *[(f"new-sampled-{index:02d}", path) for index, path in enumerate(new_sampled_paths)],
    ]
    artifacts = _source_artifacts(source_paths)
    verified_identities = [
        {"artifact_path": str(path), "checkpoint_id": checkpoint_id, "checkpoint_identity": source_checkpoint.to_artifact_dict()}
        for path in [*all_greedy_paths, *all_sampled_paths]
    ]
    output.mkdir(parents=True)
    selection_document = _selection_receipt(selection)
    selection_document["source_artifacts"] = artifacts
    _write_json(output / "selection-receipt.json", selection_document)
    common_census = {
        "sampled_candidates": sampled_census,
        "source_candidates": source_census,
        "selection": selection_document,
    }
    arm_receipts: dict[str, Any] = {}
    for name, assembled in (("broad", broad), ("concentrated", concentrated)):
        root = output / f"{name}-plus-source-preservation"
        arm_receipts[name] = _write_arm(
            root=root,
            rollouts=assembled[0], reviews=assembled[1], receipts=assembled[2],
            census={**common_census, "arm": assembled[3]}, arm_receipt=assembled[3],
            source_checkpoint=source_checkpoint,
            prompt_identity_sha256=str(panel_receipt["prompt_policy_fingerprint"]),
            source_artifacts=artifacts,
            verified_rollout_checkpoint_identities=verified_identities,
        )
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "status": "assembled",
        "source_checkpoint": source_checkpoint.to_artifact_dict(),
        "source_checkpoint_id": checkpoint_id,
        "prompt_identity_sha256": panel_receipt["prompt_policy_fingerprint"],
        "execution_metadata": panel_receipt["execution_metadata"],
        "reference_derivation": {
            "policy": "exact_rollout_prompt_ids_plus_candidate_pool_image_and_split_metadata",
            "image_count": len(reference_records),
            "greedy_and_all_sampled_prompt_ids_cross_checked": True,
            "image_pad_interval": "canonical_single_contiguous_image_pad_run",
        },
        "selection_receipt": str((output / "selection-receipt.json").resolve()),
        "arms": {name: str((output / f"{name}-plus-source-preservation" / "assembly-receipt.json").resolve()) for name in arm_receipts},
        "source_artifacts": artifacts,
    }
    _write_json(output / "assembly-receipt.json", receipt)
    return receipt


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--trajectory-analysis", type=Path)
    parser.add_argument("--panel-union-receipt", type=Path)
    for name in ("old-greedy", "new-greedy", "old-sampled", "new-sampled"):
        parser.add_argument(f"--{name}", type=Path, action="append")
    parser.add_argument("--sampled-panel-root", type=Path)
    parser.add_argument("--source-b16-root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        receipt = assemble_constant_dose_breadth_state_banks(
            candidate_pool=args.candidate_pool,
            split_receipt=args.split_receipt,
            trajectory_analysis=args.trajectory_analysis,
            panel_union_receipt=args.panel_union_receipt,
            old_greedy=args.old_greedy,
            new_greedy=args.new_greedy,
            old_sampled=args.old_sampled,
            new_sampled=args.new_sampled,
            sampled_panel_root=args.sampled_panel_root,
            source_b16_root=args.source_b16_root,
            output_dir=args.output_dir,
        )
    except (AssemblyError, FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"status": receipt["status"], "output_dir": str(args.output_dir.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
