import json
import unittest

from final_review_tools import (
    COCO_80_CLASS_NAMES,
    build_final_review_task,
    export_reviewed_objects_from_annotation,
    match_objects,
)


class FinalReviewToolsTest(unittest.TestCase):
    def test_coco80_is_canonical_and_unique(self):
        self.assertEqual(len(COCO_80_CLASS_NAMES), 80)
        self.assertEqual(len(set(COCO_80_CLASS_NAMES)), 80)
        self.assertEqual(COCO_80_CLASS_NAMES[0], "person")
        self.assertEqual(COCO_80_CLASS_NAMES[-1], "toothbrush")

    def test_class_exact_iou_matching(self):
        gt = [
            {"index": 0, "desc": "person", "bbox_2d": [0, 0, 100, 100]},
            {"index": 1, "desc": "chair", "bbox_2d": [200, 200, 260, 260]},
        ]
        pred = [
            {"index": 0, "desc": "person", "bbox_2d": [10, 10, 90, 90]},
            {"index": 1, "desc": "chair", "bbox_2d": [300, 300, 330, 330]},
            {"index": 2, "desc": "cat", "bbox_2d": [200, 200, 260, 260]},
        ]
        matching = match_objects(gt, pred, iou_threshold=0.5)
        self.assertEqual(matching.tp_pairs, [(0, 0)])
        self.assertEqual(matching.fp_pred_indices, [1, 2])
        self.assertEqual(matching.fn_gt_indices, [1])

    def test_task_contains_editable_final_and_readonly_evidence_layers(self):
        source_task = {
            "data": {
                "image_abs": "/data/CoordExp/public_data/coco/raw/images/val2017/000000000139.jpg",
                "image_rel": "images/val2017/000000000139.jpg",
                "line_idx": 0,
                "width": 640,
                "height": 426,
            }
        }
        gt = [
            {"index": 0, "desc": "person", "bbox_2d": [0, 0, 100, 100]},
            {"index": 1, "desc": "chair", "bbox_2d": [200, 200, 260, 260]},
        ]
        pred = [
            {"index": 0, "desc": "person", "bbox_2d": [10, 10, 90, 90]},
            {"index": 1, "desc": "chair", "bbox_2d": [300, 300, 330, 330]},
        ]
        task = build_final_review_task(source_task, gt, pred, "desc_first_t07")
        data = task["data"]
        self.assertTrue(data["final_image"].startswith("http://127.0.0.1:18080/coordexp-assets/"))
        self.assertEqual(data["gt_count"], 2)
        self.assertEqual(data["pred_count"], 2)
        self.assertEqual(data["tp_count"], 1)
        self.assertEqual(data["fp_count"], 1)
        self.assertEqual(data["fn_count"], 1)

        results = task["annotations"][0]["result"]
        final_boxes = [r for r in results if r["from_name"] == "final_bbox"]
        evidence_boxes = [r for r in results if r["from_name"] == "evidence_bbox"]
        self.assertEqual([r["value"]["rectanglelabels"][0] for r in final_boxes],
                         ["final_object", "final_object", "candidate_from_prediction"])
        self.assertEqual([r["value"]["rectanglelabels"][0] for r in evidence_boxes],
                         ["evidence_tp", "evidence_fp", "evidence_fn"])
        self.assertTrue(all(not r.get("readonly", False) for r in final_boxes))
        self.assertTrue(all(r.get("readonly", False) for r in evidence_boxes))

    def test_export_only_final_object_with_valid_coco_class(self):
        annotation = {
            "result": [
                {
                    "id": "keep",
                    "from_name": "final_bbox",
                    "to_name": "final_image",
                    "type": "rectanglelabels",
                    "original_width": 200,
                    "original_height": 100,
                    "value": {
                        "x": 10,
                        "y": 20,
                        "width": 30,
                        "height": 40,
                        "rectanglelabels": ["final_object"],
                    },
                },
                {
                    "id": "keep",
                    "from_name": "final_class",
                    "to_name": "final_image",
                    "type": "taxonomy",
                    "value": {"taxonomy": [["person"]]},
                },
                {
                    "id": "skip",
                    "from_name": "final_bbox",
                    "to_name": "final_image",
                    "type": "rectanglelabels",
                    "original_width": 200,
                    "original_height": 100,
                    "value": {
                        "x": 0,
                        "y": 0,
                        "width": 10,
                        "height": 10,
                        "rectanglelabels": ["candidate_from_prediction"],
                    },
                },
                {
                    "id": "skip",
                    "from_name": "final_class",
                    "to_name": "final_image",
                    "type": "taxonomy",
                    "value": {"taxonomy": [["cat"]]},
                },
            ]
        }
        objects, audit = export_reviewed_objects_from_annotation(annotation)
        self.assertEqual(objects, [{"bbox_2d": [20, 20, 80, 60], "desc": "person", "source": "reviewed_final"}])
        self.assertEqual(audit["exported_final_count"], 1)
        self.assertEqual(audit["skipped_candidate_count"], 1)
        self.assertEqual(audit["missing_class_count"], 0)


if __name__ == "__main__":
    unittest.main()
