from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

try:
    import yaml
except Exception:  # pragma: no cover - configuration runs require PyYAML.
    yaml = None

from src.infer.artifacts import load_comparable_artifact


REQUIRED_ARTIFACTS = (
    "gt_vs_pred.jsonl",
    "pred_token_trace.jsonl",
    "pred_confidence.jsonl",
    "gt_vs_pred_scored.jsonl",
    "gt_vs_pred_scored_guarded.jsonl",
    "summary.json",
    "resolved_config.json",
    "eval/metrics.json",
    "eval/metrics_guarded.json",
    "eval/matches.jsonl",
    "eval/matches_guarded.jsonl",
    "eval/duplicate_guard_report.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_idx, line in enumerate(handle):
            if not line.strip():
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_idx + 1} must contain a JSON object")
            rows.append(payload)
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_line_count(path: Path) -> int | None:
    if path.suffix != ".jsonl":
        return None
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


def _row_hash(row: Mapping[str, Any]) -> str:
    data = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(data).hexdigest()


def _first_image(row: Mapping[str, Any]) -> str | None:
    image = row.get("image") or row.get("file_name")
    if image is not None:
        return str(image)
    images = row.get("images")
    if isinstance(images, list) and images:
        return str(images[0])
    return None


def _objects(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key)
    return list(value) if isinstance(value, list) else []


def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to load analysis YAML configs")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return payload


def run_phase1_from_config(config_path: str | Path) -> dict[str, Any]:
    return run_phase1(_load_yaml(Path(config_path)))


def run_phase1(config: Mapping[str, Any]) -> dict[str, Any]:
    resolved = build_resolved_inputs(config)
    gate_report = build_gate_report(resolved)
    per_rows, per_images, summary = build_lane_a_rows(resolved)

    output_dir = Path(str(resolved["run"]["output_dir"]))
    rollout_dir = output_dir / "rollout_anatomy"
    _write_json(output_dir / "resolved_inputs.json", resolved)
    _write_json(output_dir / "gate_report.json", gate_report)
    _write_jsonl(rollout_dir / "per_row.jsonl", per_rows)
    _write_jsonl(rollout_dir / "per_image.jsonl", per_images)
    _write_coverage_survival(rollout_dir / "coverage_survival.csv", per_images)
    _write_json(rollout_dir / "summary.json", summary)
    _write_report(rollout_dir / "report.md", resolved, gate_report, summary)
    return {
        "resolved_inputs": resolved,
        "gate_report": gate_report,
        "per_rows": per_rows,
        "per_images": per_images,
        "summary": summary,
    }


def build_resolved_inputs(config: Mapping[str, Any]) -> dict[str, Any]:
    run_cfg = dict(config.get("run") or {})
    input_cfg = dict(config.get("inputs") or {})
    analysis_cfg = dict(config.get("analysis") or {})
    artifact_root = Path(str(input_cfg["artifact_root"])).resolve()
    dataset_jsonl = Path(str(input_cfg["dataset_jsonl"])).resolve()
    output_dir = Path(str(run_cfg["output_dir"])).resolve()

    source_artifacts: dict[str, str] = {}
    artifact_hashes: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for rel in REQUIRED_ARTIFACTS:
        path = artifact_root / rel
        source_artifacts[rel] = str(path)
        if not path.exists():
            missing.append(rel)
            continue
        artifact_hashes[rel] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "line_count": _jsonl_line_count(path),
        }

    dataset_slice = dict(input_cfg.get("dataset_slice") or {})
    if not dataset_slice:
        dataset_slice = {"kind": "all"}
    dataset_hash = {
        "path": str(dataset_jsonl),
        "size_bytes": dataset_jsonl.stat().st_size if dataset_jsonl.exists() else None,
        "sha256": _sha256_file(dataset_jsonl) if dataset_jsonl.exists() else None,
        "line_count": _jsonl_line_count(dataset_jsonl) if dataset_jsonl.exists() else None,
    }

    summary = _read_json(artifact_root / "summary.json") if (artifact_root / "summary.json").exists() else {}
    resolved_config = (
        _read_json(artifact_root / "resolved_config.json")
        if (artifact_root / "resolved_config.json").exists()
        else {}
    )

    return {
        "schema_version": 1,
        "analysis_name": "autoreg_object_rollout_phase1",
        "run": {
            "output_dir": str(output_dir),
            "scope_label": str(run_cfg.get("scope_label", "")),
            "checkpoint": str(run_cfg.get("checkpoint", "")),
        },
        "inputs": {
            "artifact_source_root_requested": str(input_cfg["artifact_root"]),
            "artifact_source_root_resolved": str(artifact_root),
            "dataset_jsonl": str(dataset_jsonl),
            "dataset_hash": dataset_hash,
            "dataset_slice": dataset_slice,
        },
        "analysis": {
            "truncate_at_first_im_end": bool(
                analysis_cfg.get("truncate_at_first_im_end", True)
            ),
            "require_scored_provenance": bool(
                analysis_cfg.get("require_scored_provenance", False)
            ),
            "metric_family": str(analysis_cfg.get("metric_family", "guarded")),
        },
        "source_artifacts": source_artifacts,
        "artifact_hashes": artifact_hashes,
        "missing_artifacts": missing,
        "decode": _decode_scope(summary, resolved_config),
    }


def _decode_scope(summary: Mapping[str, Any], resolved_config: Mapping[str, Any]) -> dict[str, Any]:
    generation = dict(summary.get("generation") or {})
    infer = dict(summary.get("infer") or {})
    resolved_infer = dict(resolved_config.get("infer") or {})
    return {
        "temperature": generation.get("temperature"),
        "repetition_penalty": generation.get("repetition_penalty"),
        "max_new_tokens": generation.get("max_new_tokens"),
        "prompt_template_hash": infer.get("prompt_template_hash")
        or resolved_infer.get("prompt_template_hash"),
        "object_ordering": infer.get("object_ordering") or resolved_infer.get("object_ordering"),
        "resolved_base_model_checkpoint": resolved_infer.get("resolved_base_model_checkpoint"),
    }


def build_gate_report(resolved: Mapping[str, Any]) -> dict[str, Any]:
    artifact_root = Path(str(resolved["inputs"]["artifact_source_root_resolved"]))
    dataset_jsonl = Path(str(resolved["inputs"]["dataset_jsonl"]))
    errors: list[str] = []
    warnings: list[str] = []

    missing = list(resolved.get("missing_artifacts") or [])
    if missing:
        errors.append(f"missing required artifacts: {missing}")

    base_rows = _read_jsonl(artifact_root / "gt_vs_pred.jsonl")
    trace_rows = _read_jsonl(artifact_root / "pred_token_trace.jsonl")
    dataset_rows = _slice_dataset_rows(dataset_jsonl, resolved["inputs"]["dataset_slice"])
    summary = _read_json(artifact_root / "summary.json")

    if len(base_rows) != len(trace_rows):
        errors.append(f"base/trace row count mismatch: {len(base_rows)} != {len(trace_rows)}")
    for idx, trace in enumerate(trace_rows):
        if int(trace.get("line_idx", -1)) != idx:
            errors.append(f"trace line_idx mismatch at row {idx}: {trace.get('line_idx')}")
            break
    if len(dataset_rows) != len(base_rows):
        errors.append(f"dataset/base row count mismatch: {len(dataset_rows)} != {len(base_rows)}")
    else:
        for idx, (dataset_row, base_row) in enumerate(zip(dataset_rows, base_rows)):
            if _first_image(dataset_row) != _first_image(base_row):
                errors.append(
                    f"dataset/base image mismatch at row {idx}: "
                    f"{_first_image(dataset_row)} != {_first_image(base_row)}"
                )
                break
            if dataset_row.get("image_id") != base_row.get("image_id"):
                errors.append(
                    f"dataset/base image_id mismatch at row {idx}: "
                    f"{dataset_row.get('image_id')} != {base_row.get('image_id')}"
                )
                break

    total_emitted = summary.get("total_emitted")
    if total_emitted is not None and int(total_emitted) != len(base_rows):
        errors.append(f"summary.total_emitted mismatch: {total_emitted} != {len(base_rows)}")
    total_read = summary.get("total_read")
    if total_read is not None and int(total_read) != len(base_rows):
        errors.append(f"summary.total_read mismatch: {total_read} != {len(base_rows)}")
    errors_total = int(summary.get("errors_total") or 0)
    row_errors_total = sum(len(_objects(row, "errors")) for row in base_rows)
    if errors_total != row_errors_total:
        warnings.append(
            f"summary.errors_total differs from row errors: {errors_total} != {row_errors_total}"
        )

    duplicate_guard_status = _validate_duplicate_guard_mapping(artifact_root)
    errors.extend(duplicate_guard_status["errors"])
    warnings.extend(duplicate_guard_status["warnings"])

    score_checks = {}
    score_errors = []
    for rel in ("gt_vs_pred_scored.jsonl", "gt_vs_pred_scored_guarded.jsonl"):
        try:
            loaded = load_comparable_artifact(artifact_root / rel, require_score=True)
            score_checks[rel] = {
                "status": "ok",
                "provenance_path": loaded.get("provenance_path"),
                "provenance_carrier": loaded.get("provenance_carrier"),
            }
        except Exception as exc:
            score_checks[rel] = {
                "status": "missing_or_invalid",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
            score_errors.append(rel)

    require_scored = bool(resolved["analysis"].get("require_scored_provenance", False))
    gate0_status = "ok" if not errors else "fail"
    if score_errors and require_scored:
        gate0b_status = "fail"
    elif score_errors:
        gate0b_status = "warning"
    else:
        gate0b_status = "ok"

    return {
        "gate0": {
            "status": gate0_status,
            "errors": errors,
            "warnings": warnings,
            "row_count": len(base_rows),
            "trace_count": len(trace_rows),
            "dataset_slice_count": len(dataset_rows),
        },
        "gate0b": {
            "status": gate0b_status,
            "score_checks": score_checks,
            "metric_family_effective": "debug-f1ish-only" if score_errors else resolved["analysis"].get("metric_family"),
            "raw_guarded_mapping": duplicate_guard_status,
        },
    }


def _slice_dataset_rows(path: Path, dataset_slice: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _read_jsonl(path)
    if dataset_slice.get("kind") == "first_n":
        return rows[: int(dataset_slice["n"])]
    return rows


def _validate_duplicate_guard_mapping(artifact_root: Path) -> dict[str, Any]:
    base_rows = _read_jsonl(artifact_root / "gt_vs_pred.jsonl")
    guarded_matches = _read_jsonl(artifact_root / "eval/matches_guarded.jsonl")
    duplicate_report = _read_json(artifact_root / "eval/duplicate_guard_report.json")
    guard_records = {
        int(rec.get("record_index")): rec
        for rec in list(duplicate_report.get("records") or [])
        if isinstance(rec, Mapping) and rec.get("record_index") is not None
    }
    errors: list[str] = []
    warnings: list[str] = []
    checked_guarded_indices = 0
    for idx, (base_row, match_row) in enumerate(zip(base_rows, guarded_matches)):
        raw_pred_count = len(_objects(base_row, "pred"))
        raw_to_guarded, guarded_to_raw, _suppressed = _guard_maps(
            guard_records.get(idx), raw_pred_count
        )
        guarded_indices = set(guarded_to_raw)
        for key in ("matches",):
            for item in list(match_row.get(key) or []):
                pred_idx = int(item.get("pred_idx", -1))
                checked_guarded_indices += 1
                if pred_idx not in guarded_indices:
                    errors.append(f"guarded pred_idx {pred_idx} at row {idx} has no raw mapping")
        for key in ("unmatched_pred_indices", "ignored_pred_indices"):
            for pred_idx in list(match_row.get(key) or []):
                checked_guarded_indices += 1
                if int(pred_idx) not in guarded_indices:
                    errors.append(f"guarded {key} pred_idx {pred_idx} at row {idx} has no raw mapping")
        for raw_idx, guarded_idx in raw_to_guarded.items():
            if guarded_to_raw.get(guarded_idx) != raw_idx:
                errors.append(f"non-bijective guard mapping at row {idx}, raw {raw_idx}")
                break
    if int(duplicate_report.get("total_records") or len(base_rows)) != len(base_rows):
        warnings.append(
            "duplicate_guard_report.total_records does not match base row count: "
            f"{duplicate_report.get('total_records')} != {len(base_rows)}"
        )
    return {
        "status": "ok" if not errors else "fail",
        "errors": errors,
        "warnings": warnings,
        "checked_guarded_indices": checked_guarded_indices,
        "total_predictions_inspected": duplicate_report.get("total_predictions_inspected"),
        "total_predictions_suppressed": duplicate_report.get("total_predictions_suppressed"),
    }


def _guard_maps(
    guard_record: Mapping[str, Any] | None,
    raw_pred_count: int,
) -> tuple[dict[int, int], dict[int, int], set[int]]:
    if guard_record is None:
        kept = list(range(raw_pred_count))
        suppressed: set[int] = set()
    else:
        kept = [int(idx) for idx in list(guard_record.get("kept_indices") or [])]
        suppressed = {int(idx) for idx in list(guard_record.get("suppressed_indices") or [])}
        if not kept and raw_pred_count and not suppressed:
            kept = list(range(raw_pred_count))
    raw_to_guarded = {raw_idx: guarded_idx for guarded_idx, raw_idx in enumerate(kept)}
    guarded_to_raw = {guarded_idx: raw_idx for raw_idx, guarded_idx in raw_to_guarded.items()}
    return raw_to_guarded, guarded_to_raw, suppressed


def build_lane_a_rows(
    resolved: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    artifact_root = Path(str(resolved["inputs"]["artifact_source_root_resolved"]))
    base_rows = _read_jsonl(artifact_root / "gt_vs_pred.jsonl")
    guarded_rows = _read_jsonl(artifact_root / "gt_vs_pred_scored_guarded.jsonl")
    trace_rows = _read_jsonl(artifact_root / "pred_token_trace.jsonl")
    raw_matches = _read_jsonl(artifact_root / "eval/matches.jsonl")
    guarded_matches = _read_jsonl(artifact_root / "eval/matches_guarded.jsonl")
    duplicate_report = _read_json(artifact_root / "eval/duplicate_guard_report.json")
    summary_json = _read_json(artifact_root / "summary.json")
    dataset_rows = _slice_dataset_rows(
        Path(str(resolved["inputs"]["dataset_jsonl"])),
        resolved["inputs"]["dataset_slice"],
    )
    guard_records = {
        int(rec.get("record_index")): rec
        for rec in list(duplicate_report.get("records") or [])
        if isinstance(rec, Mapping) and rec.get("record_index") is not None
    }

    max_new_tokens = int((summary_json.get("generation") or {}).get("max_new_tokens") or 0)
    per_rows: list[dict[str, Any]] = []
    per_images: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    token_summary: Counter[str] = Counter()
    under_equal_over: Counter[str] = Counter()
    total_gt = 0
    total_raw_pred = 0
    total_guarded_pred = 0

    for source_line_idx, base_row in enumerate(base_rows):
        raw_pred_count = len(_objects(base_row, "pred"))
        guarded_pred_count = len(_objects(guarded_rows[source_line_idx], "pred"))
        gt_count = len(_objects(base_row, "gt"))
        dataset_gt_count = len(_objects(dataset_rows[source_line_idx], "objects"))
        total_gt += gt_count
        total_raw_pred += raw_pred_count
        total_guarded_pred += guarded_pred_count
        if raw_pred_count < gt_count:
            under_equal_over["under"] += 1
        elif raw_pred_count > gt_count:
            under_equal_over["over"] += 1
        else:
            under_equal_over["equal"] += 1

        trace_info = _token_trace_summary(
            trace_rows[source_line_idx],
            max_new_tokens=max_new_tokens,
            truncate_at_first_im_end=bool(resolved["analysis"].get("truncate_at_first_im_end", True)),
        )
        token_summary["used_token_count"] += trace_info["used_token_count"]
        token_summary["tokens_after_first_im_end"] += trace_info["tokens_after_first_im_end"]
        token_summary["endoftext_after_im_end_count"] += trace_info[
            "endoftext_after_im_end_count"
        ]
        if trace_info["hit_max_new_tokens"]:
            token_summary["hit_max_new_tokens_count"] += 1
        if trace_info["first_im_end_index"] is not None:
            token_summary["trace_contains_im_end_count"] += 1
        if bool(base_row.get("raw_ends_with_im_end")):
            token_summary["raw_ends_with_im_end_count"] += 1

        raw_to_guarded, _guarded_to_raw, suppressed = _guard_maps(
            guard_records.get(source_line_idx),
            raw_pred_count,
        )
        raw_label_map, raw_gt_map = _match_label_maps(raw_matches[source_line_idx])
        guarded_label_map, guarded_gt_map = _match_label_maps(guarded_matches[source_line_idx])
        cumulative_matched_gt: set[int] = set()
        for raw_idx, pred in enumerate(_objects(base_row, "pred")):
            guarded_idx = raw_to_guarded.get(raw_idx)
            raw_match_label = raw_label_map.get(raw_idx, "unknown_pred")
            guarded_match_label = (
                guarded_label_map.get(guarded_idx, "unknown_pred")
                if guarded_idx is not None
                else None
            )
            if raw_idx in suppressed:
                row_label = "duplicate_suppressed"
            else:
                row_label = raw_match_label
            matched_gt_idx = raw_gt_map.get(raw_idx)
            if matched_gt_idx is not None:
                cumulative_matched_gt.add(int(matched_gt_idx))
            label_counts[row_label] += 1
            per_rows.append(
                {
                    "source_line_idx": source_line_idx,
                    "eval_record_idx": int(raw_matches[source_line_idx].get("image_id", source_line_idx)),
                    "coco_image_id": base_row.get("image_id"),
                    "image": _first_image(base_row),
                    "source_row_hash": _row_hash(base_row),
                    "raw_pred_idx": raw_idx,
                    "guarded_pred_idx": guarded_idx,
                    "suppressed_by_guard": raw_idx in suppressed,
                    "raw_match_label": raw_match_label,
                    "guarded_match_label": guarded_match_label,
                    "row_label": row_label,
                    "matched_gt_idx": matched_gt_idx,
                    "guarded_matched_gt_idx": guarded_gt_map.get(guarded_idx)
                    if guarded_idx is not None
                    else None,
                    "pred_desc": pred.get("desc") if isinstance(pred, Mapping) else None,
                    "pred_points": pred.get("points") if isinstance(pred, Mapping) else None,
                    "raw_pred_count": raw_pred_count,
                    "guarded_pred_count": guarded_pred_count,
                    "gt_count": gt_count,
                    "dataset_gt_count": dataset_gt_count,
                    "matched_prefix_gt_indices": sorted(cumulative_matched_gt),
                    "remaining_gt_indices_after_row": [
                        idx for idx in range(gt_count) if idx not in cumulative_matched_gt
                    ],
                }
            )

        per_images.append(
            {
                "source_line_idx": source_line_idx,
                "eval_record_idx": int(raw_matches[source_line_idx].get("image_id", source_line_idx)),
                "coco_image_id": base_row.get("image_id"),
                "image": _first_image(base_row),
                "source_row_hash": _row_hash(base_row),
                "gt_count": gt_count,
                "dataset_gt_count": dataset_gt_count,
                "raw_pred_count": raw_pred_count,
                "guarded_pred_count": guarded_pred_count,
                "under_equal_over": (
                    "under"
                    if raw_pred_count < gt_count
                    else "over"
                    if raw_pred_count > gt_count
                    else "equal"
                ),
                "raw_tp_like": sum(1 for label in raw_label_map.values() if label == "tp_like"),
                "raw_unmatched_fp": sum(
                    1 for label in raw_label_map.values() if label == "unmatched_fp"
                ),
                "suppressed_by_guard_count": len(suppressed),
                "raw_ends_with_im_end": bool(base_row.get("raw_ends_with_im_end")),
                "errors": list(base_row.get("errors") or []),
                "error_entries": list(base_row.get("error_entries") or []),
                **trace_info,
            }
        )

    scope_slice = resolved["inputs"].get("dataset_slice") or {}
    slice_label = "first_200" if scope_slice.get("kind") == "first_n" and int(scope_slice.get("n", 0)) == 200 else (
        f"first_{scope_slice.get('n')}" if scope_slice.get("kind") == "first_n" else str(scope_slice.get("kind", "all"))
    )
    lane_summary = {
        "scope": {
            "checkpoint": resolved["run"].get("checkpoint"),
            "artifact_root": resolved["inputs"].get("artifact_source_root_resolved"),
            "dataset_slice": slice_label,
            "metric_family": resolved["analysis"].get("metric_family"),
            "decode": resolved.get("decode"),
        },
        "counts": {
            "images": len(base_rows),
            "gt_objects": total_gt,
            "raw_predictions": total_raw_pred,
            "guarded_predictions": total_guarded_pred,
            "under_generated_images": under_equal_over["under"],
            "equal_count_images": under_equal_over["equal"],
            "over_generated_images": under_equal_over["over"],
        },
        "label_counts": dict(sorted(label_counts.items())),
        "token_summary": dict(sorted(token_summary.items())),
        "production_training_recommendation": "none",
    }
    return per_rows, per_images, lane_summary


def _match_label_maps(match_row: Mapping[str, Any]) -> tuple[dict[int, str], dict[int, int]]:
    labels: dict[int, str] = {}
    gt_by_pred: dict[int, int] = {}
    for match in list(match_row.get("matches") or []):
        pred_idx = int(match["pred_idx"])
        labels[pred_idx] = "tp_like"
        gt_by_pred[pred_idx] = int(match["gt_idx"])
    for pred_idx in list(match_row.get("unmatched_pred_indices") or []):
        labels[int(pred_idx)] = "unmatched_fp"
    for pred_idx in list(match_row.get("ignored_pred_indices") or []):
        labels[int(pred_idx)] = "ignored_pred"
    return labels, gt_by_pred


def _token_trace_summary(
    trace_row: Mapping[str, Any],
    *,
    max_new_tokens: int,
    truncate_at_first_im_end: bool,
) -> dict[str, Any]:
    tokens = [str(token) for token in list(trace_row.get("generated_token_text") or [])]
    logprobs = list(trace_row.get("token_logprobs") or [])
    if len(tokens) != len(logprobs):
        raise ValueError(
            f"trace token/logprob length mismatch at line_idx={trace_row.get('line_idx')}: "
            f"{len(tokens)} != {len(logprobs)}"
        )
    first_im_end_index = None
    for idx, token in enumerate(tokens):
        if token == "<|im_end|>":
            first_im_end_index = idx
            break
    if first_im_end_index is None:
        used_token_count = len(tokens)
        tokens_after = 0
        post_eos = []
    elif truncate_at_first_im_end:
        used_token_count = first_im_end_index + 1
        post_eos = tokens[first_im_end_index + 1 :]
        tokens_after = len(post_eos)
    else:
        used_token_count = len(tokens)
        post_eos = tokens[first_im_end_index + 1 :]
        tokens_after = len(post_eos)
    return {
        "trace_token_count": len(tokens),
        "used_token_count": used_token_count,
        "first_im_end_index": first_im_end_index,
        "tokens_after_first_im_end": tokens_after,
        "endoftext_after_im_end_count": sum(1 for token in post_eos if token == "<|endoftext|>"),
        "hit_max_new_tokens": bool(max_new_tokens and len(tokens) >= max_new_tokens and first_im_end_index is None),
    }


def _write_coverage_survival(path: Path, per_images: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    max_rows = max((int(row["raw_pred_count"]) for row in per_images), default=0)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_ordinal", "images_with_at_least_row", "gt_total"],
        )
        writer.writeheader()
        for row_ordinal in range(1, max_rows + 1):
            writer.writerow(
                {
                    "row_ordinal": row_ordinal,
                    "images_with_at_least_row": sum(
                        1 for row in per_images if int(row["raw_pred_count"]) >= row_ordinal
                    ),
                    "gt_total": sum(int(row["gt_count"]) for row in per_images),
                }
            )


def _write_report(
    path: Path,
    resolved: Mapping[str, Any],
    gate_report: Mapping[str, Any],
    summary: Mapping[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = summary["counts"]
    token_summary = summary["token_summary"]
    text = f"""# Autoregressive Object Rollout Anatomy Phase 1

## Scope

- checkpoint: `{resolved['run'].get('checkpoint')}`
- artifact_root: `{resolved['inputs'].get('artifact_source_root_resolved')}`
- dataset_slice: `{summary['scope'].get('dataset_slice')}`
- metric_family: `{summary['scope'].get('metric_family')}`

## Artifact Validity

- Gate 0: `{gate_report['gate0']['status']}`
- Gate 0b: `{gate_report['gate0b']['status']}`
- metric_family_effective: `{gate_report['gate0b'].get('metric_family_effective')}`

## Main Symptom

- images: `{counts['images']}`
- GT objects: `{counts['gt_objects']}`
- raw predictions: `{counts['raw_predictions']}`
- guarded predictions: `{counts['guarded_predictions']}`
- under/equal/over images: `{counts['under_generated_images']}` / `{counts['equal_count_images']}` / `{counts['over_generated_images']}`

## Token Health

- trace_contains_im_end_count: `{token_summary.get('trace_contains_im_end_count', 0)}`
- raw_ends_with_im_end_count: `{token_summary.get('raw_ends_with_im_end_count', 0)}`
- tokens_after_first_im_end: `{token_summary.get('tokens_after_first_im_end', 0)}`
- endoftext_after_im_end_count: `{token_summary.get('endoftext_after_im_end_count', 0)}`

## Training Implications

production_training_recommendation: none

This is a Gate 0/0b + Lane A report only. It must not be used to launch a
production training change or to claim H1-H5 convergence.
"""
    path.write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Analysis YAML config path.")
    args = parser.parse_args(argv)
    result = run_phase1_from_config(args.config)
    output_dir = result["resolved_inputs"]["run"]["output_dir"]
    print(f"wrote phase1 analysis to {output_dir}")
    print(f"gate0={result['gate_report']['gate0']['status']}")
    print(f"gate0b={result['gate_report']['gate0b']['status']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
