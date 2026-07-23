#!/usr/bin/env python3
"""Measure per-owner support in the sampled-only vLLM trajectory panel.

This is an experiment-local reader for the authoritative
``coordexp_vllm_trajectory_panel.v2`` sampled-only panel.  It reads one
persisted batch at a time, deliberately never retaining the 1.8 GiB panel or
its generated token arrays in memory.  Matching and its conservative
ambiguity/unresolved receipts are owned by
``analyze_individual_trajectory_union_support``; this reader only aggregates
those receipts into per-annotation-owner support bounds.

``q_lower`` counts only unambiguous category-constrained IoU >= 0.50 matches.
``q_upper`` adds only explicit ambiguous-match candidate owners.  In
particular, an official-unmatched or otherwise unresolved prediction is not
treated as a candidate owner or as a hallucination.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from scripts.research.analyze_individual_trajectory_union_support import (  # noqa: E402
    _parsed_rows,
    _sha256_file,
    _sha256_json,
    load_generation7_annotations,
    match_prefix,
)


SCHEMA_VERSION = "sampled_owner_inclusion.v1"
PANEL_SCHEMA_VERSION = "coordexp_vllm_trajectory_panel.v2"
SAMPLE_COUNT = 16
SAMPLE_INDICES = frozenset(range(SAMPLE_COUNT))
IOU_THRESHOLD = 0.50
EXPECTED_IMAGE_COUNT = 2432
EXPECTED_TRAJECTORY_COUNT = 38912
DEFAULT_PANEL_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "trajectory-panel-2432-vllm/production-v2"
)
DEFAULT_CANDIDATE_POOL = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-22-constant-dose-image-breadth-treatment-screen/"
    "candidate-pool-v1/candidate-pool-2432.coord.jsonl"
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _read_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact is not an object: {path}")
    return value


def _numeric_equal(value: Any, expected: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and float(value) == expected
    )


def _natural_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def discover_artifact_paths(panel_root: str | Path) -> list[Path]:
    """Return authoritative sampled-batch artifacts in deterministic order."""

    root = Path(panel_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    artifacts = sorted(root.rglob("sampled-batch-*.json"), key=str)
    if not artifacts:
        raise ValueError(f"no sampled-batch artifacts under {root}")
    return artifacts


def _artifact_model_signature(model_identity: Any) -> str | None:
    if not isinstance(model_identity, Mapping):
        return None
    # Rank-local device settings are not semantic model identity.  The
    # collector provides this stable subdocument specifically for this use.
    semantic = model_identity.get("execution_model_identity", model_identity)
    return _sha256_json(semantic)


def _validate_artifact_contract(
    payload: Mapping[str, Any], *, path: Path, require_full_panel: bool
) -> tuple[Mapping[str, Any], str | None]:
    if payload.get("schema_version") != PANEL_SCHEMA_VERSION:
        raise ValueError(f"wrong panel schema_version: {path}")
    config = payload.get("config")
    if not isinstance(config, Mapping):
        raise ValueError(f"artifact lacks config: {path}")
    if config.get("decode_mode") != "sampled":
        raise ValueError(f"artifact is not sampled-only: {path}")
    if config.get("panel_mode") != "sampled_only":
        raise ValueError(f"artifact lacks sampled_only panel_mode: {path}")
    if not _numeric_equal(config.get("temperature"), 0.4):
        raise ValueError(f"artifact has non-canonical temperature: {path}")
    if not _numeric_equal(config.get("top_p"), 0.95):
        raise ValueError(f"artifact has non-canonical top_p: {path}")
    if not _numeric_equal(config.get("repetition_penalty"), 1.0):
        raise ValueError(f"artifact has non-canonical repetition_penalty: {path}")
    if config.get("max_new_tokens") != 1024:
        raise ValueError(f"artifact has non-canonical max_new_tokens: {path}")
    if config.get("sample_count") != SAMPLE_COUNT:
        raise ValueError(f"artifact has non-canonical sample_count: {path}")
    if config.get("sample_index_range") != [0, SAMPLE_COUNT - 1]:
        raise ValueError(f"artifact has non-canonical sample_index_range: {path}")
    rows = payload.get("rollouts")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"artifact lacks rollout rows: {path}")
    if payload.get("rollout_count") is not None and payload.get("rollout_count") != len(rows):
        raise ValueError(f"artifact rollout_count disagrees with rows: {path}")
    model_signature = _artifact_model_signature(payload.get("model_identity"))
    if require_full_panel and model_signature is None:
        raise ValueError(f"full panel artifact lacks model identity: {path}")
    return config, model_signature


def _row_identity(row: Mapping[str, Any], metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep rollout identity sufficient to reopen its immutable source row."""

    metadata = metadata or {}

    def value(name: str, *fallbacks: str) -> Any:
        for key in (name, *fallbacks):
            if row.get(key) is not None:
                return row[key]
            if metadata.get(key) is not None:
                return metadata[key]
        return None

    selected = {
        "example_id": row.get("example_id"),
        "prompt_token_ids_sha256": value("prompt_token_ids_sha256"),
        "source_image_file_sha256": value("source_image_file_sha256"),
        "executed_rgb_sha256": row.get("executed_rgb_sha256"),
        "image_width": value("image_width", "width"),
        "image_height": value("image_height", "height"),
    }
    return {key: item for key, item in selected.items() if item is not None}


def _trajectory_identity(identity: Mapping[str, Any], row: Mapping[str, Any]) -> dict[str, Any]:
    """Add per-completion identity without confusing it for prompt identity."""

    result = dict(identity)
    if row.get("generated_token_ids_sha256") is not None:
        result["generated_token_ids_sha256"] = row["generated_token_ids_sha256"]
    return result


def _validate_row_identity(
    *, image_id: str, identity: Mapping[str, Any], seen: dict[str, dict[str, Any]]
) -> None:
    previous = seen.setdefault(image_id, dict(identity))
    for field in sorted(set(previous) | set(identity)):
        if field not in previous or field not in identity:
            continue
        if previous[field] != identity[field]:
            raise ValueError(f"prompt/image identity mismatch for image {image_id}: {field}")


def _validate_row_against_prompt_metadata(
    *, row: Mapping[str, Any], metadata: Mapping[str, Any], image_id: str
) -> None:
    for field in ("prompt_token_ids_sha256", "source_image_file_sha256"):
        if row.get(field) is not None and metadata.get(field) is not None:
            if row[field] != metadata[field]:
                raise ValueError(
                    f"prompt/image identity mismatch for image {image_id}: {field}"
                )


def _stratum(q_lower: float, q_upper: float) -> str:
    if q_lower == 1.0:
        return "stable"
    if q_lower > 0.0 and q_upper < 1.0:
        return "occasional"
    if q_upper == 0.0:
        return "unseen"
    return "uncertain"


def _validate_row_budget(row_budget: int | None) -> int | None:
    if row_budget is None:
        return None
    if isinstance(row_budget, bool) or not isinstance(row_budget, int) or row_budget <= 0:
        raise ValueError("row_budget must be a positive integer or None")
    return row_budget


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("row budget must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("row budget must be a positive integer")
    return parsed


def _source_record(
    *, artifact_path: Path, artifact_sha256: str, config: Mapping[str, Any], model_signature: str | None
) -> dict[str, Any]:
    return {
        "path": str(artifact_path),
        "sha256": artifact_sha256,
        "resolved_fingerprint": config.get("resolved_fingerprint"),
        "model_identity_sha256": model_signature,
    }


def _occurrence(
    *,
    receipt: Mapping[str, Any],
    owner_id: str,
    sample_index: int,
    trajectory_id: str,
    artifact_path: Path,
    artifact_sha256: str,
    config: Mapping[str, Any],
    model_signature: str | None,
    rollout_identity: Mapping[str, Any],
    evidence_kind: str,
    intersection_over_union: float,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "image_id": str(receipt["image_id"]),
        "sample_index": sample_index,
        "trajectory_id": trajectory_id,
        "generated_row_index": int(receipt["generated_row_index"]),
        "prediction_id": str(receipt["prediction_id"]),
        "owner_id": owner_id,
        "prediction_category": str(receipt["category"]),
        "intersection_over_union": float(intersection_over_union),
        "entity_evidence_kind": evidence_kind,
        "geometry_evidence_status": str(receipt.get("geometry_status", "unknown")),
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "rollout_identity": {
            **dict(rollout_identity),
            "resolved_fingerprint": config.get("resolved_fingerprint"),
            "model_identity_sha256": model_signature,
        },
    }


def analyze_artifact_paths(
    artifact_paths: Iterable[str | Path],
    candidate_pool: str | Path,
    *,
    require_full_panel: bool = True,
    panel_root: str | Path | None = None,
    row_budget: int | None = None,
) -> dict[str, Any]:
    """Read batches sequentially and return compact support records.

    The returned records intentionally omit generated token arrays and text.
    Each occurrence instead binds an immutable artifact path/hash and the
    rollout hashes needed to retrieve an exact token prefix later.
    """

    row_budget = _validate_row_budget(row_budget)
    paths = sorted({Path(path).expanduser().resolve(strict=True) for path in artifact_paths}, key=str)
    if not paths:
        raise ValueError("no sampled batch artifacts")
    pool_path = Path(candidate_pool).expanduser().resolve(strict=True)
    annotations = load_generation7_annotations(pool_path)
    owners_by_id: dict[str, dict[str, Any]] = {}
    for image_owners in annotations.values():
        for owner in image_owners:
            owner_id = str(owner["owner_id"])
            if owner_id in owners_by_id:
                raise ValueError(f"duplicate annotation owner_id: {owner_id}")
            owners_by_id[owner_id] = dict(owner)

    image_samples: dict[str, set[int]] = defaultdict(set)
    image_identities: dict[str, dict[str, Any]] = {}
    seen_pairs: set[tuple[str, int]] = set()
    seen_trajectory_ids: set[tuple[str, str]] = set()
    owner_lower: dict[str, set[tuple[str, int]]] = defaultdict(set)
    owner_ambiguous: dict[str, set[tuple[str, int]]] = defaultdict(set)
    owner_direct_ious: dict[str, list[float]] = defaultdict(list)
    owner_ambiguous_ious: dict[str, list[float]] = defaultdict(list)
    trajectory_records: list[dict[str, Any]] = []
    candidate_occurrences: list[dict[str, Any]] = []
    artifact_sources: list[dict[str, Any]] = []
    model_signatures: set[str] = set()
    resolved_fingerprints: set[str] = set()
    parser_statuses: Counter[str] = Counter()
    unresolved_count = 0
    duplicate_count = 0

    for artifact_path in paths:
        # This is the streaming boundary: only one source batch is decoded at
        # a time.  The compact records accumulated below do not retain raw
        # generated text, token arrays, or prompt token arrays.
        artifact_sha256 = _sha256_file(artifact_path)
        payload = _read_json(artifact_path)
        config, model_signature = _validate_artifact_contract(
            payload, path=artifact_path, require_full_panel=require_full_panel
        )
        if model_signature is not None:
            model_signatures.add(model_signature)
        fingerprint = config.get("resolved_fingerprint")
        if fingerprint is not None:
            resolved_fingerprints.add(str(fingerprint))
        artifact_sources.append(
            _source_record(
                artifact_path=artifact_path,
                artifact_sha256=artifact_sha256,
                config=config,
                model_signature=model_signature,
            )
        )
        prompt_metadata = payload.get("prompt_metadata")
        if prompt_metadata is not None and not isinstance(prompt_metadata, Mapping):
            raise ValueError(f"artifact prompt_metadata is not an object: {artifact_path}")
        if require_full_panel and not isinstance(prompt_metadata, Mapping):
            raise ValueError(f"full panel artifact lacks prompt metadata: {artifact_path}")
        artifact_example_ids: set[str] = set()
        for raw_row in payload["rollouts"]:
            if not isinstance(raw_row, Mapping):
                raise ValueError(f"rollout row is not an object: {artifact_path}")
            row = dict(raw_row)
            image_id = str(row.get("image_id", ""))
            if not image_id or image_id not in annotations:
                raise ValueError(f"rollout image lacks candidate-pool annotations: {image_id}")
            if row.get("decode_mode") != "sampled":
                raise ValueError(f"rollout is not sampled: {artifact_path}")
            if row.get("stop_reason") != "im_end":
                raise ValueError(f"rollout does not end with im_end: {image_id}")
            sample_index = row.get("sample_index")
            if isinstance(sample_index, bool) or not isinstance(sample_index, int):
                raise ValueError(f"rollout sample_index is not an integer: {image_id}")
            if sample_index not in SAMPLE_INDICES:
                raise ValueError(f"rollout sample_index is outside 0..15: {image_id}")
            trajectory_id = str(row.get("trajectory_id", ""))
            if not trajectory_id:
                raise ValueError(f"rollout lacks trajectory_id: {image_id}/{sample_index}")
            pair = (image_id, sample_index)
            if pair in seen_pairs:
                raise ValueError(f"duplicate image/sample_index pair: {image_id}/{sample_index}")
            seen_pairs.add(pair)
            trajectory_key = (image_id, trajectory_id)
            if trajectory_key in seen_trajectory_ids:
                raise ValueError(f"duplicate trajectory identity: {image_id}/{trajectory_id}")
            seen_trajectory_ids.add(trajectory_key)
            image_samples[image_id].add(sample_index)

            example_id = str(row.get("example_id", ""))
            if example_id:
                artifact_example_ids.add(example_id)
            metadata = (
                prompt_metadata.get(example_id)
                if isinstance(prompt_metadata, Mapping) and example_id
                else None
            )
            if metadata is not None and not isinstance(metadata, Mapping):
                raise ValueError(f"prompt metadata is not an object: {artifact_path}/{example_id}")
            if isinstance(metadata, Mapping):
                _validate_row_against_prompt_metadata(
                    row=row, metadata=metadata, image_id=image_id
                )
            identity = _row_identity(row, metadata if isinstance(metadata, Mapping) else None)
            _validate_row_identity(image_id=image_id, identity=identity, seen=image_identities)
            trajectory_identity = _trajectory_identity(identity, row)
            row["image_id"] = image_id
            row["_artifact_config"] = dict(config)
            parsed_rows, parser_evidence = _parsed_rows(row)
            parser_statuses[str(parser_evidence["parse_status"])] += 1
            evaluated_row_count = min(len(parsed_rows), row_budget) if row_budget is not None else len(parsed_rows)
            assignment = match_prefix(parsed_rows, annotations[image_id], evaluated_row_count)

            unambiguous_ids = sorted(str(owner_id) for owner_id in assignment["matched_owner_ids"])
            ambiguous_ids: set[str] = set()
            unresolved_receipts: list[dict[str, Any]] = []
            duplicate_prediction_ids: list[str] = []
            for receipt in assignment["row_assignment_receipts"]:
                status = str(receipt.get("entity_status", "unknown"))
                if status == "verified_owner":
                    owner_id = str(receipt["owner_id"])
                    iou = float(receipt["intersection_over_union"])
                    owner_lower[owner_id].add(pair)
                    owner_direct_ious[owner_id].append(iou)
                    candidate_occurrences.append(
                        _occurrence(
                            receipt=receipt,
                            owner_id=owner_id,
                            sample_index=sample_index,
                            trajectory_id=trajectory_id,
                            artifact_path=artifact_path,
                            artifact_sha256=artifact_sha256,
                            config=config,
                            model_signature=model_signature,
                            rollout_identity=trajectory_identity,
                            evidence_kind="unambiguous_matched_owner",
                            intersection_over_union=iou,
                        )
                    )
                elif status == "ambiguous_matched_review":
                    candidate_ious = receipt.get("candidate_owner_ious", {})
                    if not isinstance(candidate_ious, Mapping):
                        raise ValueError("ambiguous match receipt lacks candidate owner IoUs")
                    for raw_owner_id in receipt.get("candidate_owner_ids", []):
                        owner_id = str(raw_owner_id)
                        if owner_id not in owners_by_id:
                            raise ValueError(f"ambiguity receipt names unknown owner_id: {owner_id}")
                        try:
                            iou = float(candidate_ious[owner_id])
                        except (KeyError, TypeError, ValueError) as exc:
                            raise ValueError(
                                f"ambiguity receipt lacks IoU for candidate owner {owner_id}"
                            ) from exc
                        ambiguous_ids.add(owner_id)
                        owner_ambiguous[owner_id].add(pair)
                        owner_ambiguous_ious[owner_id].append(iou)
                        candidate_occurrences.append(
                            _occurrence(
                                receipt=receipt,
                                owner_id=owner_id,
                                sample_index=sample_index,
                                trajectory_id=trajectory_id,
                                artifact_path=artifact_path,
                                artifact_sha256=artifact_sha256,
                                config=config,
                                model_signature=model_signature,
                                rollout_identity=trajectory_identity,
                                evidence_kind="ambiguous_candidate_owner",
                                intersection_over_union=iou,
                            )
                        )
                else:
                    # Unmatched rows retain the shared matcher’s neutral
                    # receipt; no inference here upgrades them to a candidate
                    # owner or a hallucination.
                    if status in {"duplicate", "duplicate_owner"}:
                        duplicate_prediction_ids.append(str(receipt["prediction_id"]))
                        duplicate_count += 1
                    else:
                        unresolved_count += 1
                    unresolved_receipts.append(
                        {
                            "generated_row_index": int(receipt["generated_row_index"]),
                            "prediction_id": str(receipt["prediction_id"]),
                            "entity_status": status,
                            "geometry_status": str(receipt.get("geometry_status", "unknown")),
                            "review_required": bool(receipt.get("review_required", False)),
                            "candidate_owner_id": receipt.get("candidate_owner_id"),
                            "candidate_owner_iou": receipt.get("candidate_owner_iou"),
                        }
                    )
            trajectory_records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "image_id": image_id,
                    "sample_index": sample_index,
                    "trajectory_id": trajectory_id,
                    "stop_reason": "im_end",
                    "complete_parsed_row_count": len(parsed_rows),
                    "evaluated_row_count": evaluated_row_count,
                    "parser_evidence": _jsonable(parser_evidence),
                    "unambiguous_owner_ids": unambiguous_ids,
                    "ambiguous_candidate_owner_ids": sorted(ambiguous_ids),
                    "duplicate_prediction_ids": sorted(duplicate_prediction_ids),
                    "unresolved_prediction_receipts": sorted(
                        unresolved_receipts,
                        key=lambda item: (item["generated_row_index"], item["prediction_id"]),
                    ),
                    "artifact_path": str(artifact_path),
                    "artifact_sha256": artifact_sha256,
                    "rollout_identity": {
                        **trajectory_identity,
                        "resolved_fingerprint": config.get("resolved_fingerprint"),
                        "model_identity_sha256": model_signature,
                    },
                }
            )
        if require_full_panel and isinstance(prompt_metadata, Mapping):
            if artifact_example_ids != {str(key) for key in prompt_metadata}:
                raise ValueError(
                    f"prompt metadata does not cover exact artifact images: {artifact_path}"
                )
        del payload

    if len(model_signatures) > 1:
        raise ValueError("sampled panel has multiple execution-model identities")
    if len(resolved_fingerprints) > 1:
        raise ValueError("sampled panel has multiple resolved config fingerprints")
    for image_id, samples in image_samples.items():
        if samples != SAMPLE_INDICES:
            raise ValueError(
                f"image {image_id} sample_index values are not exactly 0..15: {sorted(samples)}"
            )
    if require_full_panel:
        if len(annotations) != EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"candidate pool has {len(annotations)} annotated images, expected {EXPECTED_IMAGE_COUNT}"
            )
        if set(image_samples) != set(annotations):
            missing = sorted(set(annotations) - set(image_samples), key=_natural_key)
            extra = sorted(set(image_samples) - set(annotations), key=_natural_key)
            raise ValueError(
                "panel image identities differ from candidate pool: "
                f"missing={missing[:8]} extra={extra[:8]}"
            )
        if len(image_samples) != EXPECTED_IMAGE_COUNT:
            raise ValueError(f"panel has {len(image_samples)} images, expected {EXPECTED_IMAGE_COUNT}")
        if len(trajectory_records) != EXPECTED_TRAJECTORY_COUNT:
            raise ValueError(
                f"panel has {len(trajectory_records)} trajectories, expected {EXPECTED_TRAJECTORY_COUNT}"
            )

    owner_records: list[dict[str, Any]] = []
    stratum_counts: Counter[str] = Counter()
    for owner_id, owner in sorted(
        owners_by_id.items(), key=lambda item: (_natural_key(str(item[1]["image_id"])), item[0])
    ):
        lower_pairs = owner_lower[owner_id]
        ambiguous_pairs = owner_ambiguous[owner_id]
        upper_pairs = lower_pairs | ambiguous_pairs
        q_lower = len(lower_pairs) / SAMPLE_COUNT
        q_upper = len(upper_pairs) / SAMPLE_COUNT
        stratum = _stratum(q_lower, q_upper)
        stratum_counts[stratum] += 1
        direct_ious = owner_direct_ious[owner_id]
        ambiguity_ious = owner_ambiguous_ious[owner_id]
        owner_records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "image_id": str(owner["image_id"]),
                "owner_id": owner_id,
                "category": str(owner["category"]),
                "annotation_index": int(owner["annotation_index"]),
                "entity_support": {
                    "sample_count": SAMPLE_COUNT,
                    "unambiguous_matched_trajectory_count": len(lower_pairs),
                    "ambiguous_candidate_trajectory_count": len(ambiguous_pairs - lower_pairs),
                    "q_lower": q_lower,
                    "q_upper": q_upper,
                    "stratum": stratum,
                },
                # Geometry is deliberately evidence about the matched boxes,
                # not a substitute for entity support or human ambiguity
                # adjudication.
                "geometry_evidence": {
                    "unambiguous_occurrence_count": len(direct_ious),
                    "unambiguous_iou_min": min(direct_ious) if direct_ious else None,
                    "unambiguous_iou_max": max(direct_ious) if direct_ious else None,
                    "ambiguous_candidate_occurrence_count": len(ambiguity_ious),
                    "ambiguous_candidate_iou_min": min(ambiguity_ious) if ambiguity_ious else None,
                    "ambiguous_candidate_iou_max": max(ambiguity_ious) if ambiguity_ious else None,
                },
            }
        )

    trajectory_records.sort(
        key=lambda item: (_natural_key(str(item["image_id"])), int(item["sample_index"]))
    )
    candidate_occurrences.sort(
        key=lambda item: (
            _natural_key(str(item["image_id"])),
            int(item["sample_index"]),
            int(item["generated_row_index"]),
            str(item["prediction_id"]),
            str(item["owner_id"]),
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "panel": {
            "panel_root": str(Path(panel_root).expanduser().resolve()) if panel_root is not None else None,
            "require_full_panel": bool(require_full_panel),
            "image_count": len(image_samples),
            "trajectory_count": len(trajectory_records),
            "artifact_count": len(artifact_sources),
            "sample_count_per_image": SAMPLE_COUNT,
            "row_budget": row_budget,
            "all_stop_reasons": ["im_end"],
            "resolved_fingerprints": sorted(resolved_fingerprints),
            "model_identity_sha256": sorted(model_signatures),
            "parser_status_counts": dict(sorted(parser_statuses.items())),
        },
        "summary": {
            "annotated_owner_count": len(owner_records),
            "owner_stratum_counts": dict(sorted(stratum_counts.items())),
            "candidate_occurrence_count": len(candidate_occurrences),
            "unresolved_prediction_count": unresolved_count,
            "duplicate_prediction_count": duplicate_count,
            "row_budget": row_budget,
            "matching_policy": {
                "category_constrained": True,
                "iou_threshold": IOU_THRESHOLD,
                "unambiguous_matches_only_for_q_lower": True,
                "q_upper_uses_explicit_ambiguity_candidates_only": True,
                "unmatched_predictions_are_unresolved_not_hallucinations": True,
            },
        },
        "owner_records": owner_records,
        "trajectory_records": trajectory_records,
        "candidate_occurrences": candidate_occurrences,
        "sources": {
            "candidate_pool_path": str(pool_path),
            "candidate_pool_sha256": _sha256_file(pool_path),
            "artifacts": artifact_sources,
            "analyzer_path": str(Path(__file__).resolve()),
            "analyzer_sha256": _sha256_file(Path(__file__).resolve()),
            "matching_analyzer_path": str(
                Path(match_prefix.__code__.co_filename).resolve()
            ),
            "matching_analyzer_sha256": _sha256_file(
                Path(match_prefix.__code__.co_filename).resolve()
            ),
        },
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_jsonable(row), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_analysis(result: Mapping[str, Any], output_dir: str | Path) -> Path:
    """Publish a complete immutable directory by atomically renaming a sibling."""

    output = Path(output_dir).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        owner_path = temporary / "owner-support.jsonl"
        trajectory_path = temporary / "trajectory-owners.jsonl"
        occurrence_path = temporary / "candidate-occurrences.jsonl"
        _write_jsonl(owner_path, result["owner_records"])
        _write_jsonl(trajectory_path, result["trajectory_records"])
        _write_jsonl(occurrence_path, result["candidate_occurrences"])
        summary = {
            "schema_version": result["schema_version"],
            "panel": result["panel"],
            "summary": result["summary"],
            "sources": {
                "candidate_pool_path": result["sources"]["candidate_pool_path"],
                "candidate_pool_sha256": result["sources"]["candidate_pool_sha256"],
                "artifact_count": len(result["sources"]["artifacts"]),
                "analyzer_path": result["sources"]["analyzer_path"],
                "analyzer_sha256": result["sources"]["analyzer_sha256"],
                "matching_analyzer_path": result["sources"]["matching_analyzer_path"],
                "matching_analyzer_sha256": result["sources"]["matching_analyzer_sha256"],
            },
        }
        summary_path = temporary / "summary.json"
        _write_json(summary_path, summary)
        receipt = {
            "schema_version": SCHEMA_VERSION,
            "output_contract": {
                "immutable_directory": str(output),
                "files": {
                    path.name: _sha256_file(path)
                    for path in (summary_path, owner_path, trajectory_path, occurrence_path)
                },
            },
            "sources": result["sources"],
        }
        _write_json(temporary / "receipt.json", receipt)
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-root", type=Path, default=DEFAULT_PANEL_ROOT)
    parser.add_argument("--candidate-pool", type=Path, default=DEFAULT_CANDIDATE_POOL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--row-budget",
        type=_positive_int,
        help="Optional positive parsed-row prefix budget per trajectory; defaults to all parsed rows.",
    )
    parser.add_argument(
        "--allow-incomplete-panel",
        action="store_true",
        help="Permit a synthetic subset, while still requiring all 16 sample indices per image.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    paths = discover_artifact_paths(args.panel_root)
    result = analyze_artifact_paths(
        paths,
        args.candidate_pool,
        require_full_panel=not args.allow_incomplete_panel,
        panel_root=args.panel_root,
        row_budget=args.row_budget,
    )
    write_analysis(result, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
