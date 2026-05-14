from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
while str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.config.loader import ConfigLoader
from src.detection.coord_soft_targets import (  # noqa: E402
    CoordSoftTargetCandidate,
    CoordSoftTargetRuntimeConfig,
    build_coord_soft_target,
)
from src.detection.runtime import resolve_recursive_detection_ce_runtime_cfg  # noqa: E402


SLOT_NAMES = ("x1", "y1", "x2", "y2")
SLOT_INDEX = {"x1": 0, "y1": 1, "x2": 2, "y2": 3}
REAL_PROBE_REPLACEMENT_SCAN_LIMIT = 200
REQUIRED_METRIC_KEYS = (
    "candidate_count",
    "effective_candidate_count",
    "posterior_entropy",
    "posterior_top1",
    "target_entropy",
    "target_peak_prob",
    "target_std",
    "union_coordinate_probability_ratio",
    "component_mass_by_candidate_id",
)


@dataclass(frozen=True)
class SyntheticFixture:
    fixture_id: str
    description: str
    candidate_bboxes: Mapping[str, tuple[int, int, int, int]]
    primary_slot: str
    teacher_prefix_by_slot: Mapping[str, Mapping[str, int]]
    excluded_candidate_ids: tuple[str, ...] = ()


def resolve_runtime_config(config_path: str | Path) -> CoordSoftTargetRuntimeConfig:
    training_config = ConfigLoader.load_materialized_training_config(str(config_path))
    runtime_cfg = resolve_recursive_detection_ce_runtime_cfg(training_config)
    if runtime_cfg is None or runtime_cfg.coord_soft_ce is None:
        raise ValueError("config does not resolve recursive_detection_ce coord_soft_ce")
    if runtime_cfg.coord_soft_ce.target_distribution != "instance_trie_gaussian":
        raise ValueError(
            "config coord_soft_ce target_distribution is not instance_trie_gaussian"
        )
    return runtime_cfg.coord_soft_ce


def audit_slot(
    *,
    candidates: Sequence[CoordSoftTargetCandidate],
    runtime_cfg: CoordSoftTargetRuntimeConfig,
    slot_name: str,
    teacher_prefix_values: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    target = build_coord_soft_target(
        candidates,
        runtime_cfg,
        current_slot=slot_name,
        teacher_prefix_values=teacher_prefix_values or {},
        return_components=True,
    )
    values = [candidate.bbox_xyxy[SLOT_INDEX[slot_name]] for candidate in candidates]
    union_ratio = _union_coordinate_probability_ratio(target.probs, values)
    component_mass = {
        object_id: _round_float(value)
        for object_id, value in sorted(target.posterior.items())
    }
    metrics = {
        "candidate_count": int(_tensor_float(target.candidate_count)),
        "effective_candidate_count": _round_float(target.effective_candidate_count),
        "posterior_entropy": _round_float(target.posterior_entropy),
        "posterior_top1": _round_float(target.posterior_top1),
        "target_entropy": _round_float(target.entropy),
        "target_peak_prob": _round_float(target.peak_prob),
        "target_std": _round_float(target.std),
        "union_coordinate_probability_ratio": _round_float(union_ratio),
        "component_mass_by_candidate_id": component_mass,
    }
    _assert_finite_metrics(metrics)
    return metrics


def build_audit_artifact(
    *,
    config_path: str | Path,
    include_record_idx: int | None,
) -> dict[str, Any]:
    resolved_config_path = Path(config_path).resolve()
    runtime_cfg = resolve_runtime_config(resolved_config_path)
    synthetic_fixtures = [
        _audit_synthetic_fixture(fixture, runtime_cfg)
        for fixture in _synthetic_fixtures()
    ]
    artifact = {
        "target_distribution": "instance_trie_gaussian",
        "config_path": str(resolved_config_path),
        "synthetic_fixtures": synthetic_fixtures,
        "real_probe_records": _build_real_probe_records(
            config_path=resolved_config_path,
            runtime_cfg=runtime_cfg,
            include_record_idx=include_record_idx,
        ),
        "summary": {
            "teacher_candidate_missing_count": 0,
            "candidate_leak_count": _candidate_leak_count(synthetic_fixtures),
            "nonfinite_target_count": _nonfinite_target_count(synthetic_fixtures),
        },
        "slots": _aggregate_slots(synthetic_fixtures),
    }
    _assert_synthetic_gates(artifact)
    return artifact


def write_audit_artifact(artifact: Mapping[str, Any], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit Instance-Trie Gaussian SoftCE target shapes without training."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--include-record-idx", type=int, default=None)
    args = parser.parse_args(argv)

    artifact = build_audit_artifact(
        config_path=args.config,
        include_record_idx=args.include_record_idx,
    )
    write_audit_artifact(artifact, args.output)
    print(f"wrote {args.output}")
    return 0


def _synthetic_fixtures() -> tuple[SyntheticFixture, ...]:
    return (
        SyntheticFixture(
            fixture_id="far_apart_same_desc",
            description="Two far-apart same-desc objects keep x1 multi-positive.",
            candidate_bboxes={
                "left": (100, 100, 180, 220),
                "right": (800, 100, 880, 220),
            },
            primary_slot="x1",
            teacher_prefix_by_slot={},
        ),
        SyntheticFixture(
            fixture_id="near_shared_top_left",
            description="Near-shared top-left objects remain ambiguous at x2.",
            candidate_bboxes={
                "compact": (100, 100, 210, 220),
                "large": (102, 101, 820, 920),
            },
            primary_slot="x2",
            teacher_prefix_by_slot={
                "x2": {"x1": 101, "y1": 100},
                "y2": {"x1": 101, "y1": 100, "x2": 210},
            },
        ),
        SyntheticFixture(
            fixture_id="same_previous_coord_tiny_large",
            description="Same x1/y1 prefix leaves tiny and large boxes tied.",
            candidate_bboxes={
                "tiny": (100, 100, 110, 210),
                "large": (100, 100, 900, 900),
            },
            primary_slot="x2",
            teacher_prefix_by_slot={
                "x2": {"x1": 100, "y1": 100},
                "y2": {"x1": 100, "y1": 100, "x2": 110},
            },
        ),
        SyntheticFixture(
            fixture_id="already_emitted_same_desc_excluded",
            description="Already-emitted same-desc object is absent from candidates.",
            candidate_bboxes={
                "remaining": (420, 120, 520, 260),
            },
            primary_slot="x1",
            teacher_prefix_by_slot={},
            excluded_candidate_ids=("emitted",),
        ),
        SyntheticFixture(
            fixture_id="structural_boundary_tiny_boxes",
            description="Tiny boxes on structural coordinate boundaries stay finite.",
            candidate_bboxes={
                "low": (0, 0, 1, 1),
                "high": (998, 998, 999, 999),
            },
            primary_slot="x1",
            teacher_prefix_by_slot={
                "x2": {"x1": 0, "y1": 0},
                "y2": {"x1": 0, "y1": 0, "x2": 1},
            },
        ),
        SyntheticFixture(
            fixture_id="prefix_disambiguates_far_candidate",
            description="Matching x1/y1 prefix downweights the far incompatible candidate.",
            candidate_bboxes={
                "matched": (100, 100, 220, 220),
                "far": (700, 700, 920, 920),
            },
            primary_slot="x2",
            teacher_prefix_by_slot={
                "x2": {"x1": 100, "y1": 100},
                "y2": {"x1": 100, "y1": 100, "x2": 220},
            },
        ),
    )


def _audit_synthetic_fixture(
    fixture: SyntheticFixture,
    runtime_cfg: CoordSoftTargetRuntimeConfig,
) -> dict[str, Any]:
    slots = {
        slot_name: audit_slot(
            candidates=_slot_candidates(
                fixture.candidate_bboxes,
                slot_name,
            ),
            runtime_cfg=runtime_cfg,
            slot_name=slot_name,
            teacher_prefix_values=fixture.teacher_prefix_by_slot.get(slot_name, {}),
        )
        for slot_name in SLOT_NAMES
    }
    primary = slots[fixture.primary_slot]
    return {
        "fixture_id": fixture.fixture_id,
        "description": fixture.description,
        "primary_slot": fixture.primary_slot,
        "excluded_candidate_ids": list(fixture.excluded_candidate_ids),
        "slots": slots,
        **{key: primary[key] for key in REQUIRED_METRIC_KEYS},
    }


def _slot_candidates(
    candidate_bboxes: Mapping[str, tuple[int, int, int, int]],
    slot_name: str,
) -> tuple[CoordSoftTargetCandidate, ...]:
    return tuple(
        CoordSoftTargetCandidate(
            object_instance_id=object_id,
            slot_name=slot_name,
            bbox_xyxy=bbox,
            probability=1.0,
        )
        for object_id, bbox in candidate_bboxes.items()
    )


def _build_real_probe_records(
    *,
    config_path: Path,
    runtime_cfg: CoordSoftTargetRuntimeConfig,
    include_record_idx: int | None,
) -> list[dict[str, Any]]:
    if include_record_idx is None:
        return []
    base = {
        "record_idx": int(include_record_idx),
        "status": "skipped",
    }
    try:
        training_config = ConfigLoader.load_materialized_training_config(str(config_path))
        raw_jsonl = getattr(training_config.data, "train_jsonl", None)
        if not raw_jsonl:
            return [{**base, "reason": "materialized config has no data.train_jsonl"}]
        jsonl_path = _resolve_repo_path(raw_jsonl)
        if not jsonl_path.exists():
            return [
                {
                    **base,
                    "jsonl_path": str(jsonl_path),
                    "reason": "training JSONL is unavailable in this workspace",
                }
            ]
        record = _read_jsonl_record(jsonl_path, include_record_idx)
        candidates, desc = _real_probe_candidates(record)
        if candidates:
            return [
                _real_probe_record_payload(
                    record_idx=include_record_idx,
                    requested_record_idx=include_record_idx,
                    record=record,
                    jsonl_path=jsonl_path,
                    desc=desc,
                    candidates=candidates,
                    runtime_cfg=runtime_cfg,
                )
            ]

        replacement = _find_replacement_real_probe_record(
            jsonl_path,
            requested_record_idx=include_record_idx,
        )
        if replacement is not None:
            replacement_idx, replacement_record, replacement_candidates, replacement_desc = (
                replacement
            )
            payload = _real_probe_record_payload(
                record_idx=replacement_idx,
                requested_record_idx=include_record_idx,
                record=replacement_record,
                jsonl_path=jsonl_path,
                desc=replacement_desc,
                candidates=replacement_candidates,
                runtime_cfg=runtime_cfg,
            )
            payload["selection_reason"] = "replacement_for_unusable_requested_record"
            payload["requested_record_skip_reason"] = _real_probe_skip_reason(record)
            return [payload]

        return [
            {
                **base,
                "jsonl_path": str(jsonl_path),
                "reason": _real_probe_skip_reason(record),
            }
        ]
    except Exception as exc:  # pragma: no cover - exercised only with local data quirks.
        return [{**base, "reason": f"real probe skipped after parse error: {exc}"}]


def _real_probe_record_payload(
    *,
    record_idx: int,
    requested_record_idx: int,
    record: Mapping[str, Any],
    jsonl_path: Path,
    desc: str | None,
    candidates: Sequence[tuple[str, tuple[int, int, int, int]]],
    runtime_cfg: CoordSoftTargetRuntimeConfig,
) -> dict[str, Any]:
    slots = {
        slot_name: audit_slot(
            candidates=tuple(
                CoordSoftTargetCandidate(
                    object_instance_id=object_id,
                    slot_name=slot_name,
                    bbox_xyxy=bbox,
                    probability=1.0,
                )
                for object_id, bbox in candidates
            ),
            runtime_cfg=runtime_cfg,
            slot_name=slot_name,
        )
        for slot_name in SLOT_NAMES
    }
    return {
        "record_idx": int(record_idx),
        "requested_record_idx": int(requested_record_idx),
        "status": "ok",
        "jsonl_path": str(jsonl_path),
        "image_id": _record_image_id(record),
        "repeated_desc": desc,
        "candidate_count": len(candidates),
        "slots": slots,
    }


def _find_replacement_real_probe_record(
    jsonl_path: Path,
    *,
    requested_record_idx: int,
) -> tuple[
    int,
    Mapping[str, Any],
    tuple[tuple[str, tuple[int, int, int, int]], ...],
    str | None,
] | None:
    for record_idx, record in _iter_jsonl_records(
        jsonl_path,
        limit=REAL_PROBE_REPLACEMENT_SCAN_LIMIT,
    ):
        if record_idx == requested_record_idx:
            continue
        candidates, desc = _real_probe_candidates(record)
        if candidates:
            return record_idx, record, candidates, desc
    return None


def _real_probe_candidates(
    record: Mapping[str, Any],
) -> tuple[tuple[tuple[str, tuple[int, int, int, int]], ...], str | None]:
    objects = record.get("objects")
    if not isinstance(objects, list):
        return (), None
    grouped: dict[str, list[tuple[str, tuple[int, int, int, int]]]] = defaultdict(list)
    for index, obj in enumerate(objects):
        if not isinstance(obj, Mapping):
            continue
        desc = _object_desc(obj)
        bbox = _object_bbox(obj)
        if desc is None or bbox is None:
            continue
        grouped[desc].append((f"record{index}", bbox))
    for desc, values in sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0])):
        if len(values) >= 2:
            return tuple(values[:4]), desc
    return (), None


def _real_probe_skip_reason(record: Mapping[str, Any]) -> str:
    objects = record.get("objects")
    if not isinstance(objects, list):
        return "record has no objects list"
    desc_counts: dict[str, int] = defaultdict(int)
    repeated_desc_seen = False
    unsupported_repeated_geometry = False
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        desc = _object_desc(obj)
        if desc is None:
            continue
        desc_counts[desc] += 1
    repeated_descs = {desc for desc, count in desc_counts.items() if count >= 2}
    if not repeated_descs:
        return "record has no repeated-desc object group"
    for obj in objects:
        if not isinstance(obj, Mapping):
            continue
        desc = _object_desc(obj)
        if desc not in repeated_descs:
            continue
        repeated_desc_seen = True
        if _object_bbox(obj) is None:
            unsupported_repeated_geometry = True
    if repeated_desc_seen and unsupported_repeated_geometry:
        return "record repeated-desc groups have unsupported geometry field/format"
    return "record has no repeated-desc valid token-space bbox group"


def _object_desc(obj: Mapping[str, Any]) -> str | None:
    for key in ("desc", "description", "label", "category", "name"):
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _object_bbox(obj: Mapping[str, Any]) -> tuple[int, int, int, int] | None:
    raw = obj.get("bbox_2d")
    if raw is None:
        raw = obj.get("bbox_xyxy", obj.get("bbox"))
    if not isinstance(raw, (list, tuple)) or len(raw) != 4:
        return None
    try:
        bbox = tuple(_coord_value_from_raw(value) for value in raw)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = bbox
    if 0 <= x1 < x2 <= 999 and 0 <= y1 < y2 <= 999:
        return bbox
    return None


def _coord_value_from_raw(value: Any) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        prefix = "<|coord_"
        suffix = "|>"
        if stripped.startswith(prefix) and stripped.endswith(suffix):
            return int(stripped[len(prefix) : -len(suffix)])
        return int(round(float(stripped)))
    if isinstance(value, float):
        return int(round(value))
    raise TypeError("unsupported coordinate value")


def _read_jsonl_record(path: Path, record_idx: int) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index == record_idx:
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ValueError("JSONL record is not an object")
                return value
    raise IndexError(f"record index {record_idx} is unavailable")


def _iter_jsonl_records(
    path: Path,
    *,
    limit: int,
) -> Sequence[tuple[int, Mapping[str, Any]]]:
    records: list[tuple[int, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= limit:
                break
            value = json.loads(line)
            if isinstance(value, Mapping):
                records.append((index, value))
    return tuple(records)


def _record_image_id(record: Mapping[str, Any]) -> str | None:
    for key in ("image_id", "id", "file_name", "image_path"):
        value = record.get(key)
        if isinstance(value, (str, int)):
            return str(value)
    return None


def _resolve_repo_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (REPO_ROOT / path).resolve()


def _aggregate_slots(synthetic_fixtures: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    by_slot: dict[str, list[Mapping[str, Any]]] = {slot_name: [] for slot_name in SLOT_NAMES}
    for fixture in synthetic_fixtures:
        slots = fixture.get("slots", {})
        for slot_name in SLOT_NAMES:
            slot_metrics = slots.get(slot_name)
            if isinstance(slot_metrics, Mapping):
                by_slot[slot_name].append(slot_metrics)

    return {
        slot_name: _aggregate_metric_entries(entries)
        for slot_name, entries in by_slot.items()
    }


def _aggregate_metric_entries(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not entries:
        return {key: 0.0 for key in REQUIRED_METRIC_KEYS if key != "component_mass_by_candidate_id"} | {
            "component_mass_by_candidate_id": {}
        }
    numeric_keys = [key for key in REQUIRED_METRIC_KEYS if key != "component_mass_by_candidate_id"]
    aggregated = {
        key: _round_float(sum(float(entry[key]) for entry in entries) / len(entries))
        for key in numeric_keys
    }
    component_mass: dict[str, float] = {}
    for index, entry in enumerate(entries):
        masses = entry.get("component_mass_by_candidate_id", {})
        if not isinstance(masses, Mapping):
            continue
        for candidate_id, value in masses.items():
            component_mass[f"fixture{index}:{candidate_id}"] = _round_float(value)
    aggregated["component_mass_by_candidate_id"] = component_mass
    return aggregated


def _candidate_leak_count(synthetic_fixtures: Sequence[Mapping[str, Any]]) -> int:
    leaks = 0
    for fixture in synthetic_fixtures:
        excluded = set(fixture.get("excluded_candidate_ids", ()))
        masses = fixture.get("component_mass_by_candidate_id", {})
        if not isinstance(masses, Mapping):
            continue
        leaks += len(excluded & set(masses))
    return leaks


def _nonfinite_target_count(synthetic_fixtures: Sequence[Mapping[str, Any]]) -> int:
    count = 0
    for fixture in synthetic_fixtures:
        for metrics in [fixture, *fixture.get("slots", {}).values()]:
            for key in REQUIRED_METRIC_KEYS:
                if key == "component_mass_by_candidate_id":
                    values = metrics.get(key, {}).values()
                else:
                    values = (metrics.get(key),)
                for value in values:
                    if not _is_finite_number(value):
                        count += 1
    return count


def _assert_synthetic_gates(artifact: Mapping[str, Any]) -> None:
    fixtures = {
        fixture["fixture_id"]: fixture
        for fixture in artifact["synthetic_fixtures"]
    }
    if fixtures["far_apart_same_desc"]["slots"]["x1"]["effective_candidate_count"] <= 1.0:
        raise AssertionError("far-apart repeated-desc x1 candidate ambiguity collapsed")
    if fixtures["near_shared_top_left"]["slots"]["x2"]["effective_candidate_count"] <= 1.0:
        raise AssertionError("near-shared top-left ambiguity did not survive to x2")
    prefix_x2 = fixtures["prefix_disambiguates_far_candidate"]["slots"]["x2"]
    if prefix_x2["component_mass_by_candidate_id"].get("far", 1.0) >= 0.01:
        raise AssertionError("far incompatible candidate was not downweighted by prefix")
    if prefix_x2["union_coordinate_probability_ratio"] >= 0.05:
        raise AssertionError("union-coordinate probability ratio is too high")
    if artifact["summary"]["candidate_leak_count"] != 0:
        raise AssertionError("excluded synthetic candidate leaked into audit candidates")
    if artifact["summary"]["nonfinite_target_count"] != 0:
        raise AssertionError("synthetic target metrics include non-finite values")


def _union_coordinate_probability_ratio(
    probs: torch.Tensor,
    coordinate_values: Sequence[int],
) -> float:
    if len(coordinate_values) <= 1:
        return 0.0
    low = min(coordinate_values)
    high = max(coordinate_values)
    if low == high:
        return 1.0
    midpoint = int(round((low + high) / 2.0))
    midpoint = max(0, min(int(probs.numel()) - 1, midpoint))
    peak = float(probs.max().item())
    if peak <= 0.0:
        return math.inf
    return float(probs[midpoint].item()) / peak


def _assert_finite_metrics(metrics: Mapping[str, Any]) -> None:
    for key, value in metrics.items():
        if key == "component_mass_by_candidate_id":
            if not all(_is_finite_number(item) for item in value.values()):
                raise ValueError("component candidate masses contain non-finite values")
        elif not _is_finite_number(value):
            raise ValueError(f"{key} is non-finite")


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _round_float(value: Any) -> float:
    return round(_tensor_float(value), 10)


def _tensor_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


if __name__ == "__main__":
    raise SystemExit(main())
