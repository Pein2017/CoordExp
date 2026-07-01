from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

DEFAULT_METADATA_NAMESPACE = "coordexp_proxy_supervision"
VIEW_TO_TIERS: Dict[str, tuple[str, ...]] = {
    "coco_real": ("real",),
    "coco_real_strict": ("real", "strict"),
    "coco_real_strict_plausible": ("real", "strict", "plausible"),
}


def supported_proxy_views() -> tuple[str, ...]:
    return tuple(VIEW_TO_TIERS.keys())


def _normalize_tier(value: Any) -> str:
    return str(value or "").strip().lower()


def _gt_key(record: Mapping[str, Any]) -> str:
    if isinstance(record.get("gt"), list):
        return "gt"
    if isinstance(record.get("objects"), list):
        return "objects"
    raise ValueError("record must contain list-valued `gt` or `objects`")


def _proxy_supervision_entries(
    record: Mapping[str, Any],
    *,
    metadata_namespace: str,
) -> list[Mapping[str, Any]]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("record metadata must be a mapping")
    supervision = metadata.get(metadata_namespace)
    if not isinstance(supervision, Mapping):
        raise ValueError(
            f"record metadata is missing proxy namespace {metadata_namespace!r}"
        )
    object_supervision = supervision.get("object_supervision")
    if not isinstance(object_supervision, list):
        raise ValueError(
            f"metadata.{metadata_namespace}.object_supervision must be a list"
        )
    return object_supervision


def filter_proxy_record(
    record: Mapping[str, Any],
    *,
    view: str,
    metadata_namespace: str = DEFAULT_METADATA_NAMESPACE,
) -> Dict[str, Any]:
    if view not in VIEW_TO_TIERS:
        raise ValueError(
            f"unsupported proxy view {view!r}; expected one of {supported_proxy_views()}"
        )

    gt_key = _gt_key(record)
    gt_objects = record.get(gt_key)
    if not isinstance(gt_objects, list):
        raise ValueError(f"record {gt_key!r} must be a list")

    object_supervision = _proxy_supervision_entries(
        record, metadata_namespace=metadata_namespace
    )
    if len(object_supervision) != len(gt_objects):
        raise ValueError(
            "proxy supervision length mismatch: "
            f"{len(object_supervision)} supervision entries for {len(gt_objects)} GT objects"
        )

    keep_tiers = set(VIEW_TO_TIERS[view])
    filtered_gt = [
        dict(obj)
        for obj, supervision in zip(gt_objects, object_supervision)
        if _normalize_tier(supervision.get("proxy_tier")) in keep_tiers
    ]

    out = dict(record)
    out[gt_key] = filtered_gt

    metadata = dict(record.get("metadata") or {})
    metadata["coordexp_proxy_eval_view"] = {
        "view": view,
        "keep_proxy_tiers": list(VIEW_TO_TIERS[view]),
        "metadata_namespace": metadata_namespace,
        "gt_key": gt_key,
        "objects_in": len(gt_objects),
        "objects_out": len(filtered_gt),
    }
    out["metadata"] = metadata
    return out


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"expected JSON object at {path}:{line_no}, got {type(row).__name__}"
                )
            yield row


def _build_output_path(input_jsonl: Path, output_dir: Path, view: str) -> Path:
    stem = input_jsonl.name
    if stem.endswith(".jsonl"):
        stem = stem[: -len(".jsonl")]
    return output_dir / f"{stem}.{view}.jsonl"


def materialize_proxy_eval_views(
    input_jsonl: Path,
    *,
    output_dir: Path,
    views: Sequence[str] | None = None,
    metadata_namespace: str = DEFAULT_METADATA_NAMESPACE,
) -> Dict[str, Any]:
    selected_views = list(views or supported_proxy_views())
    if not selected_views:
        raise ValueError("at least one proxy eval view must be selected")
    for view in selected_views:
        if view not in VIEW_TO_TIERS:
            raise ValueError(
                f"unsupported proxy view {view!r}; expected one of {supported_proxy_views()}"
            )

    rows = list(_iter_jsonl(input_jsonl))
    if not rows:
        raise ValueError(f"input JSONL is empty: {input_jsonl}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tier_counts_in: Counter[str] = Counter()
    outputs: Dict[str, str] = {}
    summaries: Dict[str, Dict[str, Any]] = {}

    for view in selected_views:
        output_path = _build_output_path(input_jsonl, output_dir, view)
        objects_in = 0
        objects_out = 0
        tier_counts_out: Counter[str] = Counter()

        with output_path.open("w", encoding="utf-8") as handle:
            for row in rows:
                gt_key = _gt_key(row)
                gt_objects = row.get(gt_key)
                if not isinstance(gt_objects, list):
                    raise ValueError(f"record {gt_key!r} must be a list")
                supervision_entries = _proxy_supervision_entries(
                    row, metadata_namespace=metadata_namespace
                )
                if len(supervision_entries) != len(gt_objects):
                    raise ValueError(
                        "proxy supervision length mismatch: "
                        f"{len(supervision_entries)} supervision entries for {len(gt_objects)} GT objects"
                    )

                if view == selected_views[0]:
                    for supervision in supervision_entries:
                        tier_counts_in.update(
                            [_normalize_tier(supervision.get("proxy_tier"))]
                        )

                filtered = filter_proxy_record(
                    row,
                    view=view,
                    metadata_namespace=metadata_namespace,
                )
                objects_in += len(gt_objects)
                kept_objects = filtered[gt_key]
                objects_out += len(kept_objects)
                for supervision in supervision_entries:
                    tier = _normalize_tier(supervision.get("proxy_tier"))
                    if tier in VIEW_TO_TIERS[view]:
                        tier_counts_out.update([tier])
                handle.write(json.dumps(filtered, ensure_ascii=False) + "\n")

        outputs[view] = str(output_path)
        summaries[view] = {
            "path": str(output_path),
            "records": len(rows),
            "objects_in": objects_in,
            "objects_out": objects_out,
            "objects_dropped": objects_in - objects_out,
            "tiers_kept": list(VIEW_TO_TIERS[view]),
            "tier_counts_out": dict(sorted(tier_counts_out.items())),
        }

    summary = {
        "input_jsonl": str(input_jsonl),
        "metadata_namespace": metadata_namespace,
        "record_count": len(rows),
        "tier_counts_in": dict(sorted(tier_counts_in.items())),
        "views": summaries,
        "outputs": outputs,
    }

    summary_path = output_dir / "proxy_eval_views_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary

