#!/usr/bin/env python3
"""Prepare and, after C acceptance, render a bounded A-vs-C visual comparison.

Preparation is CPU/read-only with respect to model artifacts.  The only
derived files are four-row adapters for the shared ``src.vis`` contract.  The
``render`` command is fail-closed until endpoint-C/consumer.json exists and
passes the same source-row checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


UNIT_DIR = Path(__file__).resolve().parent
REPO_ROOT = UNIT_DIR.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RAW_ROOT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-margin-preserved-positive-branch"
)
VISUAL_ROOT = RAW_ROOT / "visual-preparation"
A_CONSUMER = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-branch-vs-repeat-event/endpoint-A/consumer.json"
)
A_PACKET = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-branch-vs-repeat-event/endpoint-preparation/packet.json"
)
A_RECEIPT = Path(
    "/data/CoordExp/outputs/research/qwen3-vl-dense-enumeration/"
    "2026-09-11-positive-branch-vs-repeat-event/full-A/receipt.json"
)
C_CONSUMER = RAW_ROOT / "endpoint-C/consumer.json"
A_ADAPTER = VISUAL_ROOT / "adapter-A"
C_ADAPTER = VISUAL_ROOT / "adapter-C"
LATER_VISUALS = VISUAL_ROOT / "later-visuals"
PREPARATION_RECEIPT = VISUAL_ROOT / "preparation-receipt.json"
RENDER_RECEIPT = VISUAL_ROOT / "render-receipt.json"
SHARED_RENDER_FILES = (
    REPO_ROOT / "src/vis/api.py",
    REPO_ROOT / "src/vis/normalization.py",
    REPO_ROOT / "src/vis/matching.py",
    REPO_ROOT / "src/vis/rendering.py",
)

SELECTED_IMAGE_IDS = ("39654", "351017", "417044", "477415")
SELECTED_ROW_IDS = tuple(f"coco2017_train_{int(image_id):012d}" for image_id in SELECTED_IMAGE_IDS)
EXPECTED_A_CONSUMER_SHA256 = "d61454f058793fd40338f19d1ebeac9023a3889df58912c3b960224dd8b91a37"
EXPECTED_A_PACKET_SHA256 = "560006e73f3f0fc416e7d58751fe96c936aba9478fb6b320ab6118db2bcd5053"
EXPECTED_A_RECEIPT_SHA256 = "dd14419bf11a07aa36c783b26207addf0e4294242d13e4d6219d10674dfeaa40"
A_SOURCE_SCHEMA = "positive_branch_vs_repeat_event.endpoint_natural.v1"
C_SOURCE_SCHEMA = "margin_preserved_endpoint.natural.v1"
SOURCE_SCHEMAS = {"A": A_SOURCE_SCHEMA, "C": C_SOURCE_SCHEMA}
DUPLICATE_IOU_THRESHOLD = 0.95


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path) -> str:
    require(path.is_file(), f"missing file: {path}")
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_immutable_json(path: Path, value: Any) -> None:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2).encode("utf-8") + b"\n"
    if path.exists():
        require(path.read_bytes() == encoded, f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def write_immutable_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    encoded = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
        for row in rows
    )
    if path.exists():
        require(path.read_bytes() == encoded, f"immutable output differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(encoded)


def _select_rows_from_value(value: Any, *, arm: str) -> list[dict[str, Any]]:
    require(isinstance(value, list), f"{arm} consumer must be a JSON row list")
    require(len(value) == 384, f"{arm} consumer row count must be 384, got {len(value)}")
    expected_schema = SOURCE_SCHEMAS[arm]
    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in value:
        require(isinstance(row, dict), f"{arm} consumer row must be an object")
        require(row.get("arm") == arm, f"{arm} source row arm mismatch")
        require(row.get("schema") == expected_schema, f"{arm} source schema mismatch: expected {expected_schema}")
        parsed = row.get("parsed")
        require(isinstance(parsed, dict), f"{arm} parsed payload missing")
        row_id = str(row.get("request_id", ""))
        require(row_id and row_id == str(row.get("example_id")) == str(parsed.get("row_id")), f"{arm} row identity mismatch")
        require(row_id not in rows_by_id, f"{arm} duplicate row id: {row_id}")
        rows_by_id[row_id] = row
    selected: list[dict[str, Any]] = []
    for row_id in SELECTED_ROW_IDS:
        require(row_id in rows_by_id, f"{arm} missing requested row: {row_id}")
        selected.append(rows_by_id[row_id])
    return selected


def _selected_source_rows(path: Path, *, arm: str, expected_sha256: str | None) -> tuple[list[dict[str, Any]], str]:
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None:
        require(actual_sha256 == expected_sha256, f"{arm} consumer hash changed: {actual_sha256}")
    return _select_rows_from_value(read_json(path), arm=arm), actual_sha256


def _schema_fixture_checks() -> dict[str, Any]:
    """Exercise arm-specific schema admission without any model result."""
    fixture_rows = []
    for index in range(384):
        row_id = SELECTED_ROW_IDS[index] if index < len(SELECTED_ROW_IDS) else f"fixture-{index:03d}"
        fixture_rows.append({
            "arm": "C",
            "schema": C_SOURCE_SCHEMA,
            "request_id": row_id,
            "example_id": row_id,
            "parsed": {"row_id": row_id},
        })
    accepted = _select_rows_from_value(fixture_rows, arm="C")
    wrong_format = [dict(row) for row in fixture_rows]
    wrong_format[0] = {**wrong_format[0], "schema": A_SOURCE_SCHEMA}
    rejected = False
    try:
        _select_rows_from_value(wrong_format, arm="C")
    except ValueError as exc:
        rejected = "schema mismatch" in str(exc)
    require(rejected, "C wrong-schema fixture was not rejected")
    return {
        "schema": "margin_preserved_positive_branch.endpoint_schema_fixture.v1",
        "fixture_is_not_model_result": True,
        "A_schema": A_SOURCE_SCHEMA,
        "C_schema": C_SOURCE_SCHEMA,
        "C_fixture_rows": 384,
        "C_fixture_selected_rows": len(accepted),
        "wrong_C_schema_rejected": rejected,
    }


def _validate_parsed_row(row: dict[str, Any], *, arm: str, source_consumer_sha256: str) -> dict[str, Any]:
    parsed = row["parsed"]
    row_id = str(parsed["row_id"])
    image_path = Path(str(parsed["image_path"]))
    require(image_path.is_file(), f"{arm} original image missing: {image_path}")
    width = parsed.get("image_width")
    height = parsed.get("image_height")
    require(type(width) is int and width > 0 and type(height) is int and height > 0, f"{arm} image dimensions invalid: {row_id}")
    image_sha256 = file_sha256(image_path)
    gt = parsed.get("gt")
    pred = parsed.get("pred")
    require(isinstance(gt, list) and isinstance(pred, list), f"{arm} parsed GT/pred lists missing: {row_id}")

    # GT values are the source norm1000 bins.  Do not rewrite them to pixels;
    # src.vis owns that conversion.  Predictions are already pixel xyxy.
    for index, obj in enumerate(gt):
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        require(isinstance(bbox, list) and len(bbox) == 4, f"{arm} GT bbox shape: {row_id}:{index}")
        require(all(type(value) is int and 0 <= value <= 999 for value in bbox), f"{arm} GT must remain norm1000: {row_id}:{index}")
        require(bbox[0] < bbox[2] and bbox[1] < bbox[3], f"{arm} invalid GT geometry: {row_id}:{index}")
    for index, obj in enumerate(pred):
        bbox = obj.get("bbox") if isinstance(obj, dict) else None
        require(isinstance(bbox, list) and len(bbox) == 4, f"{arm} prediction bbox shape: {row_id}:{index}")
        values = [float(value) for value in bbox]
        require(all(math.isfinite(value) for value in values), f"{arm} prediction bbox nonfinite: {row_id}:{index}")
        require(values[0] < values[2] and values[1] < values[3], f"{arm} invalid prediction geometry: {row_id}:{index}")

    return {
        "row_id": row_id,
        "image_id": str(row["image_id"]),
        "example_id": str(row["example_id"]),
        "split": row.get("split"),
        "image_path": str(image_path),
        "image_file_sha256": image_sha256,
        "source_executed_media_sha256": row.get("executed_media_sha256"),
        "source_prompt_token_ids_sha256": row.get("prompt_token_ids_sha256"),
        "observed_image_grid_thw": row.get("observed_image_grid_thw"),
        "image_width": width,
        "image_height": height,
        "gt_count": len(gt),
        "pred_count": len(pred),
        "source_row_sha256": digest(row),
        "source_consumer_sha256": source_consumer_sha256,
        "source_schema": row["schema"],
        "source_score_50": row.get("score", {}).get("50"),
    }


def _adapter_rows(row: dict[str, Any], *, source_consumer_sha256: str, arm: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parsed = row["parsed"]
    info = _validate_parsed_row(row, arm=arm, source_consumer_sha256=source_consumer_sha256)
    source_marker = {
        "source_arm": arm,
        "source_consumer_sha256": source_consumer_sha256,
        "source_row_sha256": info["source_row_sha256"],
        "source_packet_sha256": row.get("packet_sha256"),
        "source_adapter_receipt_sha256": row.get("adapter_receipt_sha256"),
        "source_split": row.get("split"),
        "coordinate_surfaces": {
            "gt": "parsed.gt.bbox norm1000 bins; shared src.vis converts to pixels",
            "pred": "parsed.pred.bbox pixel xyxy; coord_bins retained as metadata",
        },
        "no_gt_edit": True,
        "no_prediction_relabel": True,
    }
    base = {
        "row_id": info["row_id"],
        "row_index": parsed["row_index"],
        "image_path": info["image_path"],
        "image_width": info["image_width"],
        "image_height": info["image_height"],
        "gt": parsed["gt"],
        "_source": source_marker,
    }
    raw = dict(base)
    scored = {**base, "pred": parsed["pred"]}
    return raw, scored, info


def _validate_adapter(adapter_dir: Path, expected_rows: tuple[str, ...]) -> dict[str, Any]:
    from src.vis.normalization import load_visual_rows

    artifacts = load_visual_rows(adapter_dir)
    actual_ids = tuple(row.row_id for row in artifacts.rows)
    require(actual_ids == expected_rows, f"shared src.vis row contract mismatch: {actual_ids}")
    return {
        "artifact_dir": str(adapter_dir),
        "raw_jsonl": str(artifacts.raw_jsonl),
        "scored_jsonl": str(artifacts.scored_jsonl),
        "raw_sha256": file_sha256(artifacts.raw_jsonl),
        "scored_sha256": file_sha256(artifacts.scored_jsonl),
        "row_ids": list(actual_ids),
        "row_count": len(actual_ids),
        "renderer_loader": "src.vis.normalization.load_visual_rows",
        "coordinate_contract_verified": True,
    }


def _code_identity() -> dict[str, Any]:
    return {
        "producer": {"path": str(Path(__file__).resolve()), "sha256": file_sha256(Path(__file__))},
        "shared_renderer": [
            {"path": str(path), "sha256": file_sha256(path)} for path in SHARED_RENDER_FILES
        ],
    }


def prepare() -> dict[str, Any]:
    require(file_sha256(A_PACKET) == EXPECTED_A_PACKET_SHA256, "accepted A packet changed")
    require(file_sha256(A_RECEIPT) == EXPECTED_A_RECEIPT_SHA256, "accepted A receipt changed")
    a_rows, a_consumer_sha256 = _selected_source_rows(A_CONSUMER, arm="A", expected_sha256=EXPECTED_A_CONSUMER_SHA256)
    raw_rows: list[dict[str, Any]] = []
    scored_rows: list[dict[str, Any]] = []
    row_info: list[dict[str, Any]] = []
    for row in a_rows:
        raw, scored, info = _adapter_rows(row, source_consumer_sha256=a_consumer_sha256, arm="A")
        raw_rows.append(raw)
        scored_rows.append(scored)
        row_info.append(info)
    require(tuple(row["row_id"] for row in raw_rows) == SELECTED_ROW_IDS, "A adapter ordering changed")
    write_immutable_jsonl(A_ADAPTER / "gt_vs_pred.jsonl", raw_rows)
    write_immutable_jsonl(A_ADAPTER / "gt_vs_pred_scored.jsonl", scored_rows)
    adapter = _validate_adapter(A_ADAPTER, SELECTED_ROW_IDS)
    receipt = {
        "schema": "margin_preserved_positive_branch.endpoint_comparison_visual_preparation.v1",
        "status": "awaiting_accepted_C_consumer",
        "candidate_only": True,
        "code_identity": _code_identity(),
        "selected_image_ids": list(SELECTED_IMAGE_IDS),
        "selected_row_ids": list(SELECTED_ROW_IDS),
        "selection": "root-selected fixed diagnostic IDs; no metric/positive filtering",
        "schema_fixture": _schema_fixture_checks(),
        "source": {
            "A_consumer": {"path": str(A_CONSUMER), "sha256": a_consumer_sha256, "expected_sha256": EXPECTED_A_CONSUMER_SHA256},
            "A_packet": {"path": str(A_PACKET), "sha256": EXPECTED_A_PACKET_SHA256},
            "A_full_receipt": {"path": str(A_RECEIPT), "sha256": EXPECTED_A_RECEIPT_SHA256},
            "C_consumer": {"path": str(C_CONSUMER), "exists_at_prepare": C_CONSUMER.is_file(), "required_before_render": True},
            "rows": row_info,
        },
        "adapter_A": adapter,
        "coordinate_surfaces": {
            "original_image": "source parsed.image_path; byte hash recorded per row",
            "ground_truth": "source parsed.gt.bbox norm1000 bins; never rewritten; shared renderer converts to pixels",
            "prediction": "source parsed.pred.bbox pixel xyxy; coord_bins retained only as metadata",
            "matching": "shared src.vis class-aware IoU/missing/FP rendering; no relabeling or hallucination promotion",
        },
        "renderer": {
            "module": "src.vis.api.render_prediction_comparison",
            "required_artifacts": ["gt_vs_pred.jsonl", "gt_vs_pred_scored.jsonl"],
            "duplicate_iou_threshold": DUPLICATE_IOU_THRESHOLD,
            "raw_source_metrics_separate": True,
            "one_png_per_image": True,
            "collage": False,
        },
        "no_model_execution": True,
        "no_gt_edits": True,
        "no_prediction_relabel": True,
        "next_command": f"python {Path(__file__).resolve()} render",
    }
    write_immutable_json(PREPARATION_RECEIPT, receipt)
    return receipt


def render() -> dict[str, Any]:
    require(C_CONSUMER.is_file(), f"accepted C consumer not available; refusing render: {C_CONSUMER}")
    receipt = read_json(PREPARATION_RECEIPT)
    require(receipt.get("status") == "awaiting_accepted_C_consumer", "visual preparation receipt is not pending C")
    a_rows, a_sha256 = _selected_source_rows(A_CONSUMER, arm="A", expected_sha256=EXPECTED_A_CONSUMER_SHA256)
    c_rows, c_sha256 = _selected_source_rows(C_CONSUMER, arm="C", expected_sha256=None)
    a_by_id = {str(row["parsed"]["row_id"]): row for row in a_rows}
    c_by_id = {str(row["parsed"]["row_id"]): row for row in c_rows}
    for row_id in SELECTED_ROW_IDS:
        a = a_by_id[row_id]["parsed"]
        c = c_by_id[row_id]["parsed"]
        require(a["image_path"] == c["image_path"], f"A/C original image path mismatch: {row_id}")
        require(a["image_width"] == c["image_width"] and a["image_height"] == c["image_height"], f"A/C image dimensions mismatch: {row_id}")
        require(a["gt"] == c["gt"], f"A/C GT payload mismatch: {row_id}")
    c_raw: list[dict[str, Any]] = []
    c_scored: list[dict[str, Any]] = []
    c_info: list[dict[str, Any]] = []
    for row in c_rows:
        raw, scored, info = _adapter_rows(row, source_consumer_sha256=c_sha256, arm="C")
        c_raw.append(raw)
        c_scored.append(scored)
        c_info.append(info)
    write_immutable_jsonl(C_ADAPTER / "gt_vs_pred.jsonl", c_raw)
    write_immutable_jsonl(C_ADAPTER / "gt_vs_pred_scored.jsonl", c_scored)
    a_adapter_check = _validate_adapter(A_ADAPTER, SELECTED_ROW_IDS)
    c_adapter_check = _validate_adapter(C_ADAPTER, SELECTED_ROW_IDS)
    require(not LATER_VISUALS.exists() or not any(LATER_VISUALS.iterdir()), "later-visuals is occupied; choose a fresh output")
    from src.vis import render_prediction_comparison

    result = render_prediction_comparison(
        A_ADAPTER,
        C_ADAPTER,
        LATER_VISUALS,
        left_label="A",
        right_label="C",
        row_ids=list(SELECTED_ROW_IDS),
        duplicate_iou_threshold=DUPLICATE_IOU_THRESHOLD,
    )
    require(len(result.image_paths) == 4 and all(path.is_file() for path in result.image_paths), "renderer did not produce exactly four PNGs")
    render_receipt = {
        "schema": "margin_preserved_positive_branch.endpoint_comparison_visual_render.v1",
        "status": "rendered",
        "selected_image_ids": list(SELECTED_IMAGE_IDS),
        "selected_row_ids": list(SELECTED_ROW_IDS),
        "source": {"A_consumer": {"path": str(A_CONSUMER), "sha256": a_sha256}, "C_consumer": {"path": str(C_CONSUMER), "sha256": c_sha256}, "C_rows": c_info},
        "adapter_A": a_adapter_check,
        "adapter_C": c_adapter_check,
        "manifest": {"path": str(result.manifest_path), "sha256": file_sha256(result.manifest_path)},
        "readme": {"path": str(result.readme_path), "sha256": file_sha256(result.readme_path)},
        "pngs": [{"path": str(path), "sha256": file_sha256(path)} for path in result.image_paths],
        "shared_renderer": "src.vis.api.render_prediction_comparison",
        "coordinate_surfaces": receipt["coordinate_surfaces"],
        "no_gt_edits": True,
        "no_prediction_relabel": True,
    }
    write_immutable_json(RENDER_RECEIPT, render_receipt)
    return render_receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("prepare", "render"), nargs="?", default="prepare")
    args = parser.parse_args()
    result = prepare() if args.command == "prepare" else render()
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
