import io
import json
from contextlib import redirect_stdout
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


ARMS = {
    "dynamic_hf_bf16": Path(
        "/data/CoordExp/outputs/infra_base/optimization-20260912/verification/"
        "mature-geo-sorted-xy-consistency/lead-verification/evaluation/dynamic"
    ),
    "dense_hf_bf16": Path(
        "/data/CoordExp/outputs/infra_base/optimization-20260912/verification/"
        "mature-geo-sorted-xy-consistency/lead-verification/evaluation/materialized"
    ),
    "dense_vllm_bf16": Path(
        "/data/CoordExp/outputs/infra_base/optimization-20260912/verification/"
        "mature-geo-sorted-xy-systematic-debug/evaluation/vllm-bf16"
    ),
    "dense_hf_fp32": Path(
        "/data/CoordExp/outputs/infra_base/optimization-20260912/verification/"
        "mature-geo-sorted-xy-systematic-debug/evaluation/hf-fp32"
    ),
}

EXCLUDED_ROW_ID = "coco2017_val_000000076468"


def evaluate(root: Path) -> dict[str, float | int]:
    gt = json.loads((root / "coco_gt.json").read_text())
    predictions = json.loads((root / "coco_predictions.json").read_text())
    excluded_image_ids = {
        image["id"]
        for image in gt["images"]
        if image.get("row_id") == EXCLUDED_ROW_ID
    }
    if len(excluded_image_ids) != 1:
        raise RuntimeError(
            f"expected one truncated row, found {len(excluded_image_ids)}"
        )

    gt["images"] = [
        image for image in gt["images"] if image["id"] not in excluded_image_ids
    ]
    gt["annotations"] = [
        annotation
        for annotation in gt["annotations"]
        if annotation["image_id"] not in excluded_image_ids
    ]
    predictions = [
        prediction
        for prediction in predictions
        if prediction["image_id"] not in excluded_image_ids
    ]

    with redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_predictions = coco_gt.loadRes(predictions)
        evaluator = COCOeval(coco_gt, coco_predictions, "bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()

    return {
        "row_count": len(gt["images"]),
        "prediction_count": len(predictions),
        "bbox_AP": float(evaluator.stats[0]),
        "bbox_AP50": float(evaluator.stats[1]),
        "bbox_AP75": float(evaluator.stats[2]),
    }


def main() -> None:
    payload = {
        "status": "diagnostic_only_small_cohort",
        "excluded_row_id": EXCLUDED_ROW_ID,
        "reason": "shared max_new_tokens truncation across all four arms",
        "arms": {name: evaluate(root) for name, root in ARMS.items()},
    }
    output = Path(__file__).with_name("metrics-excluding-shared-truncation.json")
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
