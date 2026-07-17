#!/usr/bin/env python3
"""Build a small, offline atlas of native full-image rollout row boundaries.

This is deliberately an experiment-local reader.  It consumes already sealed
``FULL_BAG_K`` terminal bundles and never replays the model.  Each complete row
is recorded together with the exact token prefix that preceded it.  Recurrent
children are grouped only by exact parent-prefix hash; no physical identity is
claimed when accepted-ledger matching is ambiguous.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.inference.parsing import parse_compact_object_box_closed  # noqa: E402


FULL_BAG_K = "FULL_BAG_K"
OBJECT_REF_START_ID = 151646
BOX_END_ID = 151649
IM_END_ID = 151645

DEFAULT_ROOTS = (
    Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-13-spatial-scope-history-disentanglement/executions/"
        "dense-union-51-primary-after-wave-local-tail-contract/artifacts/calls"
    ),
    Path(
        "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
        "2026-07-13-spatial-scope-history-disentanglement/executions/"
        "dense-union-51-second-root-2026071302-after-wave-local-tail-contract/"
        "artifacts/calls"
    ),
)
DEFAULT_LEDGER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-07-13-spatial-scope-history-disentanglement/readiness-v2/"
    "audit-augmented-ledger.jsonl"
)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _bundle_paths(roots: Sequence[Path]) -> list[Path]:
    paths: set[Path] = set()
    for root in roots:
        if root.is_file() and root.name == "terminal-output-bundle.json":
            paths.add(root)
        elif root.is_dir():
            paths.update(root.rglob("terminal-output-bundle.json"))
    return sorted(paths)


def _bundle_image_id(bundle: Mapping[str, Any]) -> str | None:
    evidence = bundle.get("execution_evidence")
    if isinstance(evidence, Mapping) and evidence.get("image_id") is not None:
        return str(evidence["image_id"])
    request = bundle.get("scheduled_request")
    if isinstance(request, Mapping) and request.get("image_id") is not None:
        return str(request["image_id"])
    return None


def _arm_code(bundle: Mapping[str, Any]) -> str | None:
    for parent in (bundle.get("scheduled_request"), bundle.get("execution_evidence")):
        if not isinstance(parent, Mapping):
            continue
        arm = parent.get("arm")
        if isinstance(arm, Mapping) and arm.get("arm_code") is not None:
            return str(arm["arm_code"])
    return None


def _dimensions(bundle: Mapping[str, Any]) -> tuple[int, int]:
    evidence = bundle.get("execution_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("terminal bundle lacks execution_evidence")
    width = _as_int(evidence.get("source_width"))
    height = _as_int(evidence.get("source_height"))
    if width is None or height is None or width <= 0 or height <= 0:
        receipts = bundle.get("parse_score_receipts")
        if isinstance(receipts, list) and receipts:
            width = _as_int(receipts[0].get("coordinate_extent_width"))
            height = _as_int(receipts[0].get("coordinate_extent_height"))
    if width is None or height is None or width <= 0 or height <= 0:
        raise ValueError("terminal bundle lacks positive source dimensions")
    return width, height


def _call_metadata(bundle: Mapping[str, Any], source_path: Path) -> dict[str, Any]:
    evidence = bundle.get("execution_evidence")
    scheduled = bundle.get("scheduled_request")
    if not isinstance(evidence, Mapping):
        evidence = {}
    if not isinstance(scheduled, Mapping):
        scheduled = {}
    decode = bundle.get("decode_result")
    if not isinstance(decode, Mapping):
        decode = {}
    receipt = decode.get("execution_receipt")
    if not isinstance(receipt, Mapping):
        receipt = {}
    return {
        "image_id": _bundle_image_id(bundle),
        "request_id": str(bundle.get("request_id") or scheduled.get("request_id") or ""),
        "call_label": scheduled.get("call_label"),
        "cell_index": _as_int(scheduled.get("cell_index")),
        "sampling_seed": _as_int(
            scheduled.get("sampling_seed")
            or evidence.get("sampling_seed")
            or receipt.get("sampling_seed")
        ),
        "source_bundle": str(source_path),
        "source_bundle_sha256": _sha256_file(source_path),
        "arm_code": _arm_code(bundle),
        "stop_reason": bundle.get("stop_reason") or decode.get("stop_reason"),
        "source_width": _dimensions(bundle)[0],
        "source_height": _dimensions(bundle)[1],
    }


def _token_texts(bundle: Mapping[str, Any], token_ids: Sequence[int]) -> list[str]:
    decode = bundle.get("decode_result")
    if not isinstance(decode, Mapping):
        return [""] * len(token_ids)
    trace = decode.get("token_trace")
    if not isinstance(trace, list):
        return [""] * len(token_ids)
    result: list[str] = []
    for index in range(len(token_ids)):
        item = trace[index] if index < len(trace) else {}
        result.append(str(item.get("token_text", "")) if isinstance(item, Mapping) else "")
    return result


def _row_spans(generated_token_ids: Sequence[int]) -> list[tuple[int, int]]:
    """Return complete object rows whose boundaries are legal and contiguous."""

    rows: list[tuple[int, int]] = []
    starts = [i for i, token in enumerate(generated_token_ids) if int(token) == OBJECT_REF_START_ID]
    for position, start in enumerate(starts):
        if start and int(generated_token_ids[start - 1]) != BOX_END_ID:
            continue
        end = next(
            (i for i in range(start + 1, len(generated_token_ids)) if int(generated_token_ids[i]) == BOX_END_ID),
            None,
        )
        if end is None:
            continue
        next_start = starts[position + 1] if position + 1 < len(starts) else None
        if next_start is not None and end >= next_start:
            continue
        if end + 1 < len(generated_token_ids):
            next_token = int(generated_token_ids[end + 1])
            if next_token not in {OBJECT_REF_START_ID, IM_END_ID}:
                continue
        rows.append((start, end))
    return rows


def _ledger_match(
    prediction: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    *,
    image_id: str,
    iou_threshold: float = 0.35,
    ambiguity_margin: float = 0.05,
) -> dict[str, Any]:
    category = str(prediction.get("description", "")).strip().lower()
    box = prediction.get("bbox")
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return {"status": "unavailable", "candidate_owner_ids": []}
    candidates: list[tuple[float, Mapping[str, Any]]] = []
    for item in ledger:
        if str(item.get("image_id")) != str(image_id):
            continue
        if str(item.get("normalized_category_name", "")).strip().lower() != category:
            continue
        candidate_box = item.get("source_canvas_box_xyxy")
        if not isinstance(candidate_box, Sequence) or len(candidate_box) != 4:
            continue
        score = _iou(box, candidate_box)
        if score >= iou_threshold:
            candidates.append((score, item))
    candidates.sort(key=lambda item: (-item[0], str(item[1].get("object_identifier", ""))))
    if not candidates:
        return {"status": "unmatched", "candidate_owner_ids": []}
    owners = [str(item.get("object_identifier", "")) for _, item in candidates]
    best = candidates[0][0]
    if len(candidates) > 1 and best - candidates[1][0] < ambiguity_margin:
        return {
            "status": "ambiguous",
            "candidate_owner_ids": owners[:5],
            "candidate_iou": [round(float(score), 6) for score, _ in candidates[:5]],
        }
    return {
        "status": "unique",
        "owner_id": owners[0],
        "candidate_owner_ids": owners[:5],
        "candidate_iou": [round(float(score), 6) for score, _ in candidates[:5]],
    }


def _iou(left: Sequence[Any], right: Sequence[Any]) -> float:
    lx1, ly1, lx2, ly2 = [float(value) for value in left]
    rx1, ry1, rx2, ry2 = [float(value) for value in right]
    ix1, iy1, ix2, iy2 = max(lx1, rx1), max(ly1, ry1), min(lx2, rx2), min(ly2, ry2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(0.0, lx2 - lx1) * max(0.0, ly2 - ly1)
    right_area = max(0.0, rx2 - rx1) * max(0.0, ry2 - ry1)
    union = left_area + right_area - inter
    return 0.0 if union <= 0 else inter / union


def _read_ledger(path: Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                value = json.loads(line)
                if isinstance(value, Mapping) and value.get("final_state", "accepted") == "accepted":
                    rows.append(dict(value))
    return rows


def _extract_call(bundle_path: Path, *, ledger: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    bundle = json.loads(bundle_path.read_text())
    if bundle.get("attempt_status") not in {None, "completed"}:
        return None
    if _arm_code(bundle) != FULL_BAG_K:
        return None
    decode = bundle.get("decode_result")
    if not isinstance(decode, Mapping):
        return None
    generated = decode.get("generated_token_ids")
    if not isinstance(generated, list) or not generated:
        return None
    generated = [int(value) for value in generated]
    prompt = decode.get("prompt_token_ids")
    if not isinstance(prompt, list):
        raise ValueError(f"bundle {bundle_path} lacks prompt_token_ids")
    prompt = [int(value) for value in prompt]
    meta = _call_metadata(bundle, bundle_path)
    image_id = str(meta["image_id"])
    token_texts = _token_texts(bundle, generated)
    rows: list[dict[str, Any]] = []
    for row_index, (start, end) in enumerate(_row_spans(generated)):
        row_ids = generated[start : end + 1]
        prefix_ids = [*prompt, *generated[:start]]
        row_text = "".join(token_texts[start : end + 1])
        parsed = parse_compact_object_box_closed(
            row_text,
            row_id=f"{meta['request_id']}:span-{row_index}",
            row_index=row_index,
            image_width=int(meta["source_width"]),
            image_height=int(meta["source_height"]),
        )
        prediction = parsed.predictions[0] if len(parsed.predictions) == 1 else None
        ledger_match = (
            _ledger_match(prediction, ledger, image_id=image_id)
            if prediction is not None
            else {"status": "unavailable", "candidate_owner_ids": []}
        )
        rows.append(
            {
                "row_index": row_index,
                "generated_token_span": [start, end],
                "generated_prefix_token_count": start,
                "full_prefix_token_count": len(prefix_ids),
                "generated_prefix_token_ids_sha256": _sha256_json(generated[:start]),
                "full_prefix_token_ids_sha256": _sha256_json(prefix_ids),
                "row_token_ids_sha256": _sha256_json(row_ids),
                "row_token_ids": row_ids,
                "row_text": row_text,
                "parse_status": parsed.parse_status,
                "prediction": prediction,
                "ledger_match": ledger_match,
                "source_call": {
                    "image_id": image_id,
                    "request_id": meta["request_id"],
                    "call_label": meta["call_label"],
                    "cell_index": meta["cell_index"],
                    "sampling_seed": meta["sampling_seed"],
                    "source_bundle": meta["source_bundle"],
                },
            }
        )
    return {**meta, "generated_token_count": len(generated), "prompt_token_count": len(prompt), "rows": rows}


def _aggregate(calls: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for call in calls:
        for row in call.get("rows", []):
            if row.get("parse_status") != "accepted" or not row.get("prediction"):
                continue
            key = (str(call["image_id"]), str(row["full_prefix_token_ids_sha256"]))
            groups[key].append({"call": call, "row": row})
    result: list[dict[str, Any]] = []
    for (image_id, prefix_hash), items in sorted(groups.items()):
        category_counts = Counter(str(item["row"]["prediction"].get("description", "")) for item in items)
        owner_counts = Counter(
            str(item["row"]["ledger_match"].get("owner_id"))
            for item in items
            if item["row"]["ledger_match"].get("status") == "unique"
        )
        variants = []
        for item in items:
            row = item["row"]
            match = row["ledger_match"]
            variants.append(
                {
                    "category": row["prediction"].get("description"),
                    "ledger_status": match.get("status"),
                    "owner_id": match.get("owner_id"),
                    "candidate_owner_ids": match.get("candidate_owner_ids", []),
                    "call_request_id": item["call"].get("request_id"),
                    "sampling_seed": item["call"].get("sampling_seed"),
                    "row_index": row.get("row_index"),
                }
            )
        result.append(
            {
                "image_id": image_id,
                "parent_prefix_sha256": prefix_hash,
                "support_count": len(items),
                "category_counts": dict(sorted(category_counts.items())),
                "unique_owner_counts": dict(sorted(owner_counts.items())),
                "variants": variants,
                "physical_owner_claim": "none_when_ambiguous",
            }
        )
    return result


def build_atlas(
    roots: Sequence[Path],
    *,
    ledger_path: Path | None = DEFAULT_LEDGER,
    image_ids: Iterable[str] | None = None,
    min_support: int = 1,
) -> dict[str, Any]:
    requested = {str(value) for value in image_ids} if image_ids is not None else None
    ledger = _read_ledger(ledger_path)
    calls: list[dict[str, Any]] = []
    skipped = Counter()
    for bundle_path in _bundle_paths(roots):
        try:
            bundle = json.loads(bundle_path.read_text())
            image_id = _bundle_image_id(bundle)
            if requested is not None and image_id not in requested:
                continue
            if _arm_code(bundle) != FULL_BAG_K:
                skipped["non_full_bag_arm"] += 1
                continue
            call = _extract_call(bundle_path, ledger=ledger)
            if call is not None:
                calls.append(call)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            skipped[type(exc).__name__] += 1
    groups = [group for group in _aggregate(calls) if int(group["support_count"]) >= min_support]
    by_image = Counter(str(call["image_id"]) for call in calls)
    return {
        "schema_version": "native_sibling_row_branch_atlas.v1",
        "full_name": "Native Sibling-Row Branch Atlas",
        "operational_meaning": "Offline exact-prefix grouping of complete natural Full-Image K-Rollout Independent Bagging rows.",
        "arm_code": FULL_BAG_K,
        "source_roots": [str(root) for root in roots],
        "ledger_path": str(ledger_path) if ledger_path else None,
        "image_ids": sorted(by_image),
        "call_count": len(calls),
        "calls_per_image": dict(sorted(by_image.items())),
        "skipped": dict(skipped),
        "calls": calls,
        "recurrent_sibling_groups": groups,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", type=Path, dest="roots")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--image-id", action="append", default=None)
    parser.add_argument("--min-support", type=int, default=1)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    roots = tuple(args.roots or DEFAULT_ROOTS)
    atlas = build_atlas(roots, ledger_path=args.ledger, image_ids=args.image_id, min_support=args.min_support)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(atlas, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "call_count": atlas["call_count"], "group_count": len(atlas["recurrent_sibling_groups"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
