"""Small, read-only loaders for sampled-rescue call bundles.

The preceding spatial-scope run already stores exact request, prompt, seed,
token, parse, and score evidence.  This module intentionally reads those
artifacts directly instead of introducing a second runtime contract.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"artifact must be an object: {path}")
    nested = value.get("payload")
    if isinstance(nested, Mapping):
        return nested
    return value


@dataclass(frozen=True)
class PredictionRecord:
    """One parsed prediction with chronology preserved."""

    category: str
    box: tuple[float, float, float, float]
    row_index: int
    score: float | None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "box_xyxy": list(self.box),
            "row_index": self.row_index,
            "score_float32": self.score,
        }


@dataclass(frozen=True)
class CallRecord:
    """Exact per-request data needed by Wave 0 and Wave 1 analysis."""

    bundle_path: str
    request_id: str
    image_id: str
    arm_code: str
    sampling_seed: int | None
    temperature: float | None
    repetition_penalty: float | None
    prompt_token_ids: tuple[int, ...]
    prompt_token_hash: str | None
    generated_token_ids: tuple[int, ...]
    generated_text: str
    predictions: tuple[PredictionRecord, ...]
    stop_reason: str | None
    bundle_sha256: str
    example_id: str = ""

    @classmethod
    def from_bundle(cls, path: Path) -> "CallRecord":
        bundle = _payload(path)
        evidence = bundle.get("execution_evidence")
        decode = bundle.get("decode_result")
        if not isinstance(evidence, Mapping) or not isinstance(decode, Mapping):
            raise ValueError(f"call bundle lacks execution/decode evidence: {path}")
        arm = evidence.get("arm")
        if isinstance(arm, Mapping):
            arm_code = str(arm.get("arm_code", "unknown"))
        else:
            arm_code = str(arm or "unknown")
        receipt = decode.get("execution_receipt")
        if not isinstance(receipt, Mapping):
            receipt = {}
        policy = receipt.get("decode_generation_policy")
        if not isinstance(policy, Mapping):
            policy = {}
        prompt_ids = tuple(int(x) for x in (decode.get("prompt_token_ids") or ()))
        prompt_hash = receipt.get("prompt_token_identifiers_hash")
        if prompt_hash is None and prompt_ids:
            prompt_hash = hashlib.sha256(
                json.dumps(list(prompt_ids), separators=(",", ":")).encode()
            ).hexdigest()
        parsed = bundle.get("parse_score_receipts") or ()
        predictions: list[PredictionRecord] = []
        for row in parsed:
            if not isinstance(row, Mapping):
                continue
            box = row.get("parsed_bbox_xyxy")
            if not isinstance(box, Sequence) or len(box) != 4:
                continue
            try:
                box_tuple = tuple(float(x) for x in box)
            except (TypeError, ValueError):
                continue
            predictions.append(
                PredictionRecord(
                    category=str(row.get("normalized_category_name") or row.get("category_text") or ""),
                    box=box_tuple,  # type: ignore[arg-type]
                    row_index=int(row.get("generated_row_index", row.get("parse_row_index", len(predictions)))),
                    score=(float(row["score"]) if row.get("score") is not None else None),
                )
            )
        evidence_image_id = str(evidence.get("image_id", ""))
        example_id = str(
            bundle.get("example_id")
            or evidence.get("example_id")
            or evidence_image_id
        )
        return cls(
            bundle_path=str(path.resolve()),
            request_id=str(bundle.get("request_id", "")),
            image_id=evidence_image_id,
            arm_code=arm_code,
            sampling_seed=(int(receipt["sampling_seed"]) if receipt.get("sampling_seed") is not None else None),
            temperature=(float(policy["temperature"]) if policy.get("temperature") is not None else None),
            repetition_penalty=(float(policy["repetition_penalty"]) if policy.get("repetition_penalty") is not None else None),
            prompt_token_ids=prompt_ids,
            prompt_token_hash=(str(prompt_hash) if prompt_hash is not None else None),
            generated_token_ids=tuple(int(x) for x in (decode.get("generated_token_ids") or ())),
            generated_text=str(decode.get("raw_generated_text") or decode.get("parser_text") or ""),
            predictions=tuple(predictions),
            stop_reason=(str(bundle.get("stop_reason")) if bundle.get("stop_reason") is not None else None),
            bundle_sha256=_sha256(path),
            example_id=example_id,
        )

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "bundle_path": self.bundle_path,
            "request_id": self.request_id,
            "image_id": self.image_id,
            "example_id": self.example_id,
            "arm_code": self.arm_code,
            "sampling_seed": self.sampling_seed,
            "temperature": self.temperature,
            "repetition_penalty": self.repetition_penalty,
            "prompt_token_hash": self.prompt_token_hash,
            "prompt_token_count": len(self.prompt_token_ids),
            "generated_token_count": len(self.generated_token_ids),
            "predictions": [p.to_json_dict() for p in self.predictions],
            "stop_reason": self.stop_reason,
            "bundle_sha256": self.bundle_sha256,
        }


@dataclass(frozen=True)
class GeometryMode:
    """Class-agnostic geometry cluster with category disagreement retained."""

    mode_id: str
    image_id: str
    representative_box: tuple[float, float, float, float]
    calls: tuple[str, ...]
    hit_count: int
    prediction_count: int
    category_counts: Mapping[str, int]
    row_indices: tuple[int, ...]
    earliest_row_by_call: Mapping[str, int]

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "mode_id": self.mode_id,
            "image_id": self.image_id,
            "representative_box_xyxy": list(self.representative_box),
            "calls": list(self.calls),
            "hit_count": self.hit_count,
            "prediction_count": self.prediction_count,
            "category_counts": dict(self.category_counts),
            "row_indices": list(self.row_indices),
            "earliest_row_by_call": dict(self.earliest_row_by_call),
        }


def box_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def load_call_records(
    artifact_root: str | Path,
    *,
    image_ids: Iterable[str] | None = None,
    arm_codes: Iterable[str] = ("FULL_BAG_K", "FULL_SINGLE"),
) -> tuple[CallRecord, ...]:
    """Load exact terminal call bundles, filtering only after validation."""

    wanted_images = {str(x) for x in image_ids} if image_ids is not None else None
    wanted_arms = {str(x) for x in arm_codes}
    records: list[CallRecord] = []
    for path in sorted(Path(artifact_root).rglob("terminal-output-bundle.json")):
        record = CallRecord.from_bundle(path)
        if record.arm_code not in wanted_arms:
            continue
        if wanted_images is not None and record.image_id not in wanted_images:
            continue
        records.append(record)
    return tuple(records)


def cluster_geometry_modes(
    calls: Sequence[CallRecord], *, iou_threshold: float = 0.50
) -> tuple[GeometryMode, ...]:
    """Cluster boxes without using category labels, then report label variants."""

    entries: list[tuple[CallRecord, PredictionRecord]] = [
        (call, pred) for call in calls for pred in call.predictions
    ]
    clusters: list[list[tuple[CallRecord, PredictionRecord]]] = []
    for call, pred in entries:
        assigned = None
        for index, cluster in enumerate(clusters):
            representative = cluster[0][1].box
            if box_iou(representative, pred.box) >= iou_threshold:
                assigned = index
                break
        if assigned is None:
            clusters.append([(call, pred)])
        else:
            clusters[assigned].append((call, pred))
    modes: list[GeometryMode] = []
    for idx, cluster in enumerate(clusters):
        boxes = [pred.box for _, pred in cluster]
        representative = tuple(
            sum(float(box[j]) for box in boxes) / float(len(boxes)) for j in range(4)
        )
        counts = Counter(pred.category for _, pred in cluster)
        modes.append(
            GeometryMode(
                mode_id=f"{cluster[0][0].image_id}:geometry-{idx}",
                image_id=cluster[0][0].image_id,
                representative_box=representative,  # type: ignore[arg-type]
                calls=tuple(sorted({call.request_id for call, _ in cluster})),
                hit_count=len({call.request_id for call, _ in cluster}),
                prediction_count=len(cluster),
                category_counts=dict(sorted(counts.items())),
                row_indices=tuple(sorted(pred.row_index for _, pred in cluster)),
                earliest_row_by_call={
                    request_id: min(
                        pred.row_index
                        for call, pred in cluster
                        if call.request_id == request_id
                    )
                    for request_id in sorted({call.request_id for call, _ in cluster})
                },
            )
        )
    return tuple(sorted(modes, key=lambda mode: (-mode.hit_count, mode.mode_id)))


def load_case_table(
    artifact_root: str | Path,
    *,
    image_ids: Iterable[str],
    audit_ledger_path: str | Path | None = None,
) -> dict[str, Any]:
    """Return a compact deterministic Wave-0 case table."""

    calls = load_call_records(artifact_root, image_ids=image_ids)
    by_image: dict[str, list[CallRecord]] = defaultdict(list)
    for call in calls:
        by_image[call.image_id].append(call)
    cases = []
    for image_id in sorted(by_image):
        image_calls = tuple(by_image[image_id])
        modes = cluster_geometry_modes(image_calls)
        ledger_rows = (
            load_audit_ledger(audit_ledger_path, image_id=image_id)
            if audit_ledger_path is not None
            else ()
        )
        cases.append(
            {
                "image_id": image_id,
                "call_count": len(image_calls),
                "call_ids": [call.request_id for call in image_calls],
                "prompt_token_hashes": sorted({call.prompt_token_hash for call in image_calls if call.prompt_token_hash}),
                "temperatures": sorted({call.temperature for call in image_calls if call.temperature is not None}),
                "geometry_modes": [mode.to_json_dict() for mode in modes],
                "ledger_matches": [
                    match_mode_to_ledger(mode, ledger_rows) for mode in modes
                ],
                "ledger_object_inclusion": match_calls_to_ledger(
                    image_calls, ledger_rows
                ) if ledger_rows else [],
            }
        )
    return {
        "schema_version": "sampled_rescue_transition.case_table.v1",
        "artifact_root": str(Path(artifact_root).resolve()),
        "audit_ledger_path": (
            str(Path(audit_ledger_path).resolve())
            if audit_ledger_path is not None
            else None
        ),
        "image_ids": sorted(by_image),
        "cases": cases,
    }


def load_audit_ledger(
    path: str | Path, *, image_id: str | None = None
) -> tuple[Mapping[str, Any], ...]:
    """Load the audit-augmented ledger without treating COCO as completeness."""

    rows: list[Mapping[str, Any]] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, Mapping):
            raise ValueError(f"audit ledger row must be an object: {path}")
        if image_id is None or str(row.get("image_id")) == str(image_id):
            rows.append(row)
    return tuple(rows)


def match_mode_to_ledger(
    mode: GeometryMode,
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> dict[str, Any]:
    """Match geometry modes to reviewed rows, retaining unmatched uncertainty."""

    best: Mapping[str, Any] | None = None
    best_iou = 0.0
    for row in ledger_rows:
        if str(row.get("final_state", "")).lower() != "accepted":
            continue
        box = row.get("source_canvas_box_xyxy")
        if not isinstance(box, Sequence) or len(box) != 4:
            continue
        score = box_iou(mode.representative_box, box)
        if score > best_iou:
            best_iou, best = score, row
    if best is None or best_iou < iou_threshold:
        return {
            "mode_id": mode.mode_id,
            "match_status": "unmatched_or_uncertain",
            "iou": float(best_iou),
            "object_identifier": None,
            "final_state": "uncertain",
        }
    return {
        "mode_id": mode.mode_id,
        "match_status": "reviewed_match",
        "iou": float(best_iou),
        "object_identifier": best.get("object_identifier"),
        "final_state": best.get("final_state", "uncertain"),
        "normalized_category_name": best.get("normalized_category_name"),
    }


def match_calls_to_ledger(
    calls: Sequence[CallRecord],
    ledger_rows: Sequence[Mapping[str, Any]],
    *,
    iou_threshold: float = 0.50,
) -> list[dict[str, Any]]:
    """Perform per-call one-to-one matching before aggregating inclusion.

    This prevents a duplicate chain or adjacent dense objects from becoming a
    single geometry cluster.  Unmatched predictions remain explicit extras.
    """

    objects = [
        row for row in ledger_rows
        if str(row.get("final_state", "")).lower() == "accepted"
        if row.get("object_identifier") is not None
        and isinstance(row.get("source_canvas_box_xyxy"), Sequence)
    ]
    summary: dict[str, dict[str, Any]] = {
        str(row["object_identifier"]): {
            "object_identifier": str(row["object_identifier"]),
            "image_id": str(row.get("image_id", "")),
            "final_state": row.get("final_state", "uncertain"),
            "hit_calls": 0,
            "prediction_count": 0,
            "earliest_row_by_call": {},
            "category_counts": {},
            "category_confusion_count": 0,
        }
        for row in objects
    }
    extras: list[dict[str, Any]] = []
    from src.analysis.spatial_scope_history.metrics import _exact_maximum_flow_assignment

    for call in calls:
        candidate_rows: list[tuple[int, int, float]] = []
        for pi, pred in enumerate(call.predictions):
            for oi, row in enumerate(objects):
                iou = box_iou(pred.box, row["source_canvas_box_xyxy"])
                if iou >= iou_threshold:
                    candidate_rows.append((pi, oi, float(iou)))
        candidate_rows.sort(key=lambda edge: (edge[0], edge[1]))
        if candidate_rows:
            common_denominator = max(
                overlap.as_integer_ratio()[1] for _, _, overlap in candidate_rows
            )
            lexicographic_base = 1 << len(candidate_rows)
            benefits = []
            for edge_index, (_, _, overlap) in enumerate(candidate_rows):
                numerator, denominator = overlap.as_integer_ratio()
                exact_iou = numerator * (common_denominator // denominator)
                benefits.append(
                    exact_iou * lexicographic_base
                    + (1 << (len(candidate_rows) - edge_index - 1))
                )
            selected_indexes = _exact_maximum_flow_assignment(
                prediction_count=len(call.predictions),
                reference_count=len(objects),
                candidate_rows=candidate_rows,
                benefits=benefits,
            )
        else:
            selected_indexes = frozenset()
        used_predictions: set[int] = set()
        matched_objects: set[int] = set()
        for edge_index in sorted(selected_indexes):
            pi, oi, _ = candidate_rows[edge_index]
            used_predictions.add(pi)
            matched_objects.add(oi)
            pred = call.predictions[pi]
            object_id = str(objects[oi]["object_identifier"])
            item = summary[object_id]
            item["hit_calls"] += 1
            item["prediction_count"] += 1
            item["earliest_row_by_call"][call.request_id] = pred.row_index
            category = pred.category
            item["category_counts"][category] = item["category_counts"].get(category, 0) + 1
            expected_category = str(
                objects[oi].get("normalized_category_name")
                or objects[oi].get("category")
                or ""
            )
            if expected_category and category != expected_category:
                item["category_confusion_count"] += 1
        for pi, pred in enumerate(call.predictions):
            if pi not in used_predictions:
                duplicate_or_fragment = any(
                    oi in matched_objects
                    and box_iou(pred.box, objects[oi]["source_canvas_box_xyxy"]) >= iou_threshold
                    for oi in range(len(objects))
                )
                extras.append({
                    "request_id": call.request_id,
                    "row_index": pred.row_index,
                    "category": pred.category,
                    "box_xyxy": list(pred.box),
                    "classification": (
                        "duplicate_or_fragment"
                        if duplicate_or_fragment
                        else "unmatched_or_uncertain"
                    ),
                })
    return [*sorted(summary.values(), key=lambda item: item["object_identifier"]), {"extras": extras}]
