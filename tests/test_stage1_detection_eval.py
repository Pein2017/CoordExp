from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch

from src.callbacks.stage1_detection_eval import Stage1DetectionEvalCallback


class _FakeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))


def test_stage1_detection_eval_backfills_lvis_metadata_and_logs_metrics(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gt_jsonl = tmp_path / "val.coord.jsonl"
    gt_jsonl.write_text(
        json.dumps(
            {
                "images": ["val2017/000000000001.jpg"],
                "width": 640,
                "height": 480,
                "objects": [
                    {
                        "desc": "cat",
                        "bbox_2d": [10, 20, 110, 120],
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    raw_lvis_json = tmp_path / "lvis_v1_val.json"
    raw_lvis_json.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "id": 1,
                        "coco_url": "http://images.cocodataset.org/val2017/000000000001.jpg",
                        "neg_category_ids": [2],
                        "not_exhaustive_category_ids": [3],
                    }
                ],
                "annotations": [],
                "categories": [
                    {"id": 1, "name": "cat", "frequency": "common"},
                    {"id": 2, "name": "dog", "frequency": "frequent"},
                    {"id": 3, "name": "bird", "frequency": "rare"},
                ],
            }
        ),
        encoding="utf-8",
    )

    callback = Stage1DetectionEvalCallback(
        gt_jsonl=str(gt_jsonl),
        output_root=str(tmp_path / "output"),
        model_checkpoint="dummy-model",
        prompt_variant="lvis_stage1_federated",
        bbox_format="cxcy_logw_logh",
        object_field_order="desc_first",
        object_ordering="sorted",
        metrics="lvis",
        use_segm=False,
        strict_parse=True,
        iou_thrs=None,
        f1ish_iou_thrs=[0.3, 0.5],
        f1ish_pred_scope="annotated",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.6,
        semantic_device="cpu",
        semantic_batch_size=8,
        lvis_max_dets=300,
        pred_score_source="stage1_eval_constant",
        pred_score_version=1,
        constant_score=1.0,
        batch_size=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        limit=1,
        seed=7,
        lvis_annotations_json=str(raw_lvis_json),
    )

    def _fake_infer(self):
        out_path = Path(self.cfg.out_path)
        summary_path = Path(
            self.cfg.summary_path or (out_path.parent / "infer_summary.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "val2017/000000000001.jpg",
            "width": 640,
            "height": 480,
            "mode": "coord",
            "coord_mode": "pixel",
            "gt": [
                {
                    "desc": "cat",
                    "type": "bbox_2d",
                    "points": [10, 20, 110, 120],
                    "bbox": [10, 20, 110, 120],
                }
            ],
            "pred": [
                {
                    "desc": "cat",
                    "bbox_2d": [10, 20, 110, 120],
                    "score": 0.17,
                }
            ],
            "raw_output_json": {"objects": []},
            "raw_special_tokens": [],
            "raw_ends_with_im_end": True,
            "errors": [],
            "error_entries": [],
        }
        out_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        summary_path.write_text("{}", encoding="utf-8")
        return out_path, summary_path

    def _fake_evaluate_and_save(pred_jsonl, options):
        rows = [
            json.loads(line)
            for line in Path(pred_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows[0]["metadata"]["dataset_policy"] == "lvis_federated"
        assert rows[0]["metadata"]["image_id"] == 1
        assert rows[0]["pred_score_source"] == "stage1_eval_constant"
        assert rows[0]["pred_score_version"] == 1
        assert rows[0]["pred"][0]["score"] == 1.0
        assert options.metrics == "lvis"
        return {"metrics": {"bbox_AP": 0.5}}

    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.InferenceEngine.infer",
        _fake_infer,
    )
    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.evaluate_and_save",
        _fake_evaluate_and_save,
    )

    model = _FakeModel()
    model.train()
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        args=SimpleNamespace(process_index=0, device=torch.device("cpu")),
        state=SimpleNamespace(global_step=12, is_world_process_zero=True),
        control=SimpleNamespace(),
        metrics=metrics,
        model=model,
    )

    assert model.training is True
    assert metrics["eval_det_bbox_AP"] == 0.5

    base_jsonl = (
        tmp_path / "output" / "eval_detection" / "step_0000012" / "gt_vs_pred.jsonl"
    )
    base_rows = [
        json.loads(line)
        for line in base_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert base_rows[0]["metadata"]["dataset_policy"] == "lvis_federated"


def test_stage1_detection_eval_nonzero_rank_shards_generation_without_scoring(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gt_jsonl = tmp_path / "val.coord.jsonl"
    gt_jsonl.write_text("", encoding="utf-8")

    callback = Stage1DetectionEvalCallback(
        gt_jsonl=str(gt_jsonl),
        output_root=str(tmp_path / "output"),
        model_checkpoint="dummy-model",
        prompt_variant="coco_80",
        bbox_format="xyxy",
        object_field_order="desc_first",
        object_ordering="sorted",
        metrics="f1ish",
        use_segm=False,
        strict_parse=True,
        iou_thrs=None,
        f1ish_iou_thrs=[0.5],
        f1ish_pred_scope="annotated",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.6,
        semantic_device="cpu",
        semantic_batch_size=8,
        lvis_max_dets=300,
        pred_score_source="stage1_eval_constant",
        pred_score_version=1,
        constant_score=1.0,
        batch_size=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        limit=8,
        seed=7,
        lvis_annotations_json=None,
    )

    captured: dict[str, object] = {}

    def _fake_infer(self):
        captured["rank"] = self.cfg.rank
        captured["local_rank"] = self.cfg.local_rank
        captured["world_size"] = self.cfg.world_size
        captured["distributed_enabled"] = self.cfg.distributed_enabled
        captured["out_path"] = self.cfg.out_path
        return Path(self.cfg.out_path), Path(self.cfg.summary_path or "")

    def _fail_evaluate_and_save(*_args, **_kwargs):
        raise AssertionError("nonzero eval ranks must not run final detection scoring")

    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.InferenceEngine.infer",
        _fake_infer,
    )
    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.evaluate_and_save",
        _fail_evaluate_and_save,
    )

    model = _FakeModel()
    model.train()
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        args=SimpleNamespace(
            process_index=1,
            local_process_index=1,
            local_rank=1,
            world_size=4,
            device=torch.device("cpu"),
        ),
        state=SimpleNamespace(global_step=12, is_world_process_zero=False),
        control=SimpleNamespace(),
        metrics=metrics,
        model=model,
    )

    assert model.training is True
    assert metrics == {}
    assert captured == {
        "rank": 1,
        "local_rank": 1,
        "world_size": 4,
        "distributed_enabled": True,
        "out_path": str(
            tmp_path / "output" / "eval_detection" / "step_0000012" / "gt_vs_pred.jsonl"
        ),
    }


def test_stage1_detection_eval_rank_zero_scores_after_distributed_generation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gt_jsonl = tmp_path / "val.coord.jsonl"
    gt_jsonl.write_text("", encoding="utf-8")

    callback = Stage1DetectionEvalCallback(
        gt_jsonl=str(gt_jsonl),
        output_root=str(tmp_path / "output"),
        model_checkpoint="dummy-model",
        prompt_variant="coco_80",
        bbox_format="xyxy",
        object_field_order="desc_first",
        object_ordering="sorted",
        metrics="f1ish",
        use_segm=False,
        strict_parse=True,
        iou_thrs=None,
        f1ish_iou_thrs=[0.5],
        f1ish_pred_scope="annotated",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.6,
        semantic_device="cpu",
        semantic_batch_size=8,
        lvis_max_dets=300,
        pred_score_source="stage1_eval_constant",
        pred_score_version=1,
        constant_score=1.0,
        batch_size=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        limit=8,
        seed=7,
        lvis_annotations_json=None,
    )

    captured: dict[str, object] = {}

    def _fake_infer(self):
        captured["rank"] = self.cfg.rank
        captured["local_rank"] = self.cfg.local_rank
        captured["world_size"] = self.cfg.world_size
        captured["distributed_enabled"] = self.cfg.distributed_enabled
        out_path = Path(self.cfg.out_path)
        summary_path = Path(
            self.cfg.summary_path or (out_path.parent / "infer_summary.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "val2017/000000000001.jpg",
            "width": 640,
            "height": 480,
            "mode": "coord",
            "coord_mode": "pixel",
            "gt": [],
            "pred": [],
            "raw_output_json": {"objects": []},
            "raw_special_tokens": [],
            "raw_ends_with_im_end": True,
            "errors": [],
            "error_entries": [],
        }
        out_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        summary_path.write_text("{}", encoding="utf-8")
        return out_path, summary_path

    def _fake_evaluate_and_save(pred_jsonl, options):
        rows = [
            json.loads(line)
            for line in Path(pred_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(rows) == 1
        assert options.metrics == "f1ish"
        return {"metrics": {"f1_full_micro": 0.25}}

    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.InferenceEngine.infer",
        _fake_infer,
    )
    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.evaluate_and_save",
        _fake_evaluate_and_save,
    )

    model = _FakeModel()
    model.train()
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        args=SimpleNamespace(
            process_index=0,
            local_process_index=0,
            local_rank=0,
            world_size=4,
            device=torch.device("cpu"),
        ),
        state=SimpleNamespace(global_step=12, is_world_process_zero=True),
        control=SimpleNamespace(),
        metrics=metrics,
        model=model,
    )

    assert model.training is True
    assert metrics["eval_det_f1_full_micro"] == 0.25
    assert captured == {
        "rank": 0,
        "local_rank": 0,
        "world_size": 4,
        "distributed_enabled": True,
    }


def test_stage1_detection_eval_can_score_map_with_confidence_postop(
    monkeypatch,
    tmp_path: Path,
) -> None:
    gt_jsonl = tmp_path / "val.coord.jsonl"
    gt_jsonl.write_text("", encoding="utf-8")

    callback = Stage1DetectionEvalCallback(
        gt_jsonl=str(gt_jsonl),
        output_root=str(tmp_path / "output"),
        model_checkpoint="dummy-model",
        prompt_variant="coco_80",
        bbox_format="xyxy",
        object_field_order="desc_first",
        object_ordering="sorted",
        metrics="both",
        use_segm=False,
        strict_parse=True,
        iou_thrs=None,
        f1ish_iou_thrs=[0.5],
        f1ish_pred_scope="annotated",
        semantic_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.6,
        semantic_device="cpu",
        semantic_batch_size=8,
        lvis_max_dets=300,
        score_mode="confidence_postop",
        pred_score_source="stage1_eval_constant",
        pred_score_version=1,
        constant_score=1.0,
        batch_size=1,
        max_new_tokens=128,
        temperature=0.0,
        top_p=1.0,
        repetition_penalty=1.0,
        limit=1,
        seed=7,
        lvis_annotations_json=None,
    )

    def _fake_infer(self):
        out_path = Path(self.cfg.out_path)
        summary_path = Path(
            self.cfg.summary_path or (out_path.parent / "infer_summary.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "image": "val2017/000000000001.jpg",
            "width": 640,
            "height": 480,
            "mode": "coord",
            "coord_mode": "pixel",
            "gt": [
                {
                    "desc": "cat",
                    "type": "bbox_2d",
                    "points": [10, 20, 110, 120],
                    "bbox": [10, 20, 110, 120],
                }
            ],
            "pred": [
                {
                    "desc": "cat",
                    "bbox_2d": [10, 20, 110, 120],
                }
            ],
            "raw_output_json": {"objects": []},
            "raw_special_tokens": [],
            "raw_ends_with_im_end": True,
            "errors": [],
            "error_entries": [],
        }
        out_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        (out_path.parent / "pred_token_trace.jsonl").write_text(
            json.dumps(
                {
                    "line_idx": 0,
                    "generated_token_text": ["cat"],
                    "token_logprobs": [-0.1],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        summary_path.write_text("{}", encoding="utf-8")
        return out_path, summary_path

    captured: dict[str, object] = {}

    def _fake_run_confidence_postop(paths, *, options=None):
        captured["confidence_gt_vs_pred_jsonl"] = str(paths.gt_vs_pred_jsonl)
        captured["confidence_trace_jsonl"] = str(paths.pred_token_trace_jsonl)
        scored_row = json.loads(
            Path(paths.gt_vs_pred_jsonl).read_text(encoding="utf-8").strip()
        )
        scored_row["pred_score_source"] = "confidence_postop"
        scored_row["pred_score_version"] = 2
        scored_row["pred"][0]["score"] = 0.73
        Path(paths.gt_vs_pred_scored_jsonl).write_text(
            json.dumps(scored_row) + "\n",
            encoding="utf-8",
        )
        Path(paths.pred_confidence_jsonl).write_text("", encoding="utf-8")
        Path(paths.confidence_postop_summary_json).write_text(
            json.dumps({"kept_fraction": 1.0}),
            encoding="utf-8",
        )
        return {"kept_fraction": 1.0}

    def _fake_evaluate_and_save(pred_jsonl, options):
        captured["eval_pred_jsonl"] = str(pred_jsonl)
        rows = [
            json.loads(line)
            for line in Path(pred_jsonl).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert rows[0]["pred_score_source"] == "confidence_postop"
        assert rows[0]["pred"][0]["score"] == 0.73
        assert options.metrics == "both"
        return {
            "metrics": {
                "bbox_AP": 0.44,
                "bbox_AP50": 0.55,
                "f1ish@0.50_f1_full_micro": 0.66,
            }
        }

    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.InferenceEngine.infer",
        _fake_infer,
    )
    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.run_confidence_postop",
        _fake_run_confidence_postop,
    )
    monkeypatch.setattr(
        "src.callbacks.stage1_detection_eval.evaluate_and_save",
        _fake_evaluate_and_save,
    )

    model = _FakeModel()
    model.train()
    metrics: dict[str, float] = {}

    callback.on_evaluate(
        args=SimpleNamespace(process_index=0, device=torch.device("cpu")),
        state=SimpleNamespace(global_step=12, is_world_process_zero=True),
        control=SimpleNamespace(),
        metrics=metrics,
        model=model,
    )

    eval_dir = tmp_path / "output" / "eval_detection" / "step_0000012"
    assert captured["confidence_gt_vs_pred_jsonl"] == str(
        eval_dir / "gt_vs_pred.jsonl"
    )
    assert captured["confidence_trace_jsonl"] == str(
        eval_dir / "pred_token_trace.jsonl"
    )
    assert captured["eval_pred_jsonl"] == str(eval_dir / "gt_vs_pred_scored.jsonl")
    assert metrics["eval_det_bbox_AP"] == 0.44
    assert metrics["eval_det_bbox_AP50"] == 0.55
    assert metrics["eval_det_f1ish@0.50_f1_full_micro"] == 0.66
