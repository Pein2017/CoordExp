from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.common.semantic_desc import normalize_desc
from src.eval.artifacts import with_constant_scores
from src.eval.detection import EvalOptions, evaluate_and_save
from src.infer.engine import GenerationConfig, InferenceConfig, InferenceEngine
from src.utils import get_logger

logger = get_logger(__name__)


def _is_world_process_zero(*, args: TrainingArguments, state: TrainerState) -> bool:
    state_flag = getattr(state, "is_world_process_zero", None)
    if isinstance(state_flag, bool):
        return state_flag
    process_index = getattr(args, "process_index", None)
    if isinstance(process_index, int):
        return int(process_index) == 0
    local_rank = getattr(args, "local_rank", None)
    if isinstance(local_rank, int):
        return int(local_rank) in {-1, 0}
    return True


def _unwrap_runtime_model(model: Any) -> Any:
    runtime_model = model
    while hasattr(runtime_model, "module"):
        next_model = getattr(runtime_model, "module")
        if next_model is runtime_model:
            break
        runtime_model = next_model
    return runtime_model


def _resolve_runtime_device(model: Any, args: TrainingArguments) -> str:
    try:
        first_param = next(model.parameters())
        return str(first_param.device)
    except (AttributeError, StopIteration, TypeError):
        device = getattr(args, "device", None)
        if device is not None:
            return str(device)
    return "cuda:0"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _image_key_variants(image_value: str) -> List[str]:
    pure = PurePosixPath(str(image_value).replace("\\", "/"))
    parts = [part for part in pure.parts if part not in {"", "."}]
    variants: List[str] = []
    if len(parts) >= 2:
        variants.append("/".join(parts[-2:]))
    if parts:
        variants.append(parts[-1])
    if not variants:
        variants.append(str(image_value))
    out: List[str] = []
    for item in variants:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return out


def _infer_lvis_split(gt_jsonl: Path) -> str:
    text = str(gt_jsonl).lower()
    name = gt_jsonl.name.lower()
    if "val" in name or "/val" in text:
        return "val"
    return "train"


def _default_lvis_annotations_json(gt_jsonl: Path) -> Path:
    split = _infer_lvis_split(gt_jsonl)
    return Path("public_data/lvis/raw/annotations") / f"lvis_v1_{split}.json"


class _LvisLegacyMetadataIndex:
    def __init__(self, annotations_json: Path) -> None:
        if not annotations_json.is_file():
            raise FileNotFoundError(
                "LVIS eval metadata backfill requires the raw annotations JSON at "
                f"{annotations_json}"
            )
        payload = json.loads(annotations_json.read_text(encoding="utf-8"))
        categories_raw = payload.get("categories")
        images_raw = payload.get("images")
        if not isinstance(categories_raw, list) or not isinstance(images_raw, list):
            raise ValueError(f"Malformed LVIS annotations JSON: {annotations_json}")

        self.categories_by_norm_name: Dict[str, Dict[str, Any]] = {}
        self.categories_by_id: Dict[int, Dict[str, Any]] = {}
        for category in categories_raw:
            if not isinstance(category, Mapping):
                continue
            try:
                category_id = int(category.get("id"))
            except (TypeError, ValueError):
                continue
            name = str(category.get("name") or "").strip()
            if not name:
                continue
            entry = {
                "category_id": int(category_id),
                "name": name,
                "frequency": str(category.get("frequency") or "unknown"),
            }
            self.categories_by_id[int(category_id)] = dict(entry)
            norm_name = normalize_desc(name)
            if norm_name and norm_name not in self.categories_by_norm_name:
                self.categories_by_norm_name[norm_name] = dict(entry)

        self.images_by_key: Dict[str, Dict[str, Any]] = {}
        for image in images_raw:
            if not isinstance(image, Mapping):
                continue
            try:
                image_id = int(image.get("id"))
            except (TypeError, ValueError):
                continue
            coco_url = str(image.get("coco_url") or "").strip()
            key_variants = _image_key_variants(coco_url) if coco_url else []
            image_meta = {
                "image_id": int(image_id),
                "neg_category_ids": [
                    int(cat_id) for cat_id in list(image.get("neg_category_ids") or [])
                ],
                "not_exhaustive_category_ids": [
                    int(cat_id)
                    for cat_id in list(image.get("not_exhaustive_category_ids") or [])
                ],
            }
            for key in key_variants:
                self.images_by_key.setdefault(key, dict(image_meta))

    def image_meta_for(self, image_value: str) -> Optional[Dict[str, Any]]:
        for key in _image_key_variants(image_value):
            image_meta = self.images_by_key.get(key)
            if image_meta is not None:
                return dict(image_meta)
        return None

    def category_entry_for_desc(self, desc: str) -> Optional[Dict[str, Any]]:
        norm_desc = normalize_desc(str(desc or ""))
        if not norm_desc:
            return None
        entry = self.categories_by_norm_name.get(norm_desc)
        return dict(entry) if entry is not None else None

    def category_entries_from_ids(
        self, category_ids: Iterable[int]
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen: set[int] = set()
        for category_id in category_ids:
            cat_id = int(category_id)
            if cat_id in seen:
                continue
            seen.add(cat_id)
            entry = self.categories_by_id.get(cat_id)
            if entry is not None:
                out.append(dict(entry))
        return out


def _maybe_backfill_lvis_metadata(
    *,
    records: Sequence[Mapping[str, Any]],
    metrics_mode: str,
    gt_jsonl: Path,
    lvis_annotations_json: Optional[str],
) -> List[Dict[str, Any]]:
    if str(metrics_mode).strip().lower() not in {"lvis", "both"}:
        return [dict(record) for record in records]

    rows = [dict(record) for record in records]
    if rows and all(
        isinstance(row.get("metadata"), Mapping)
        and str(row["metadata"].get("dataset_policy") or "").strip().lower()
        == "lvis_federated"
        for row in rows
    ):
        return rows

    annotations_path = (
        Path(str(lvis_annotations_json))
        if lvis_annotations_json is not None
        else _default_lvis_annotations_json(gt_jsonl)
    )
    index = _LvisLegacyMetadataIndex(annotations_path)

    out: List[Dict[str, Any]] = []
    for row in rows:
        metadata = row.get("metadata")
        if (
            isinstance(metadata, Mapping)
            and str(metadata.get("dataset_policy") or "").strip().lower()
            == "lvis_federated"
        ):
            out.append(dict(row))
            continue

        image_value = str(row.get("image") or "").strip()
        image_meta = index.image_meta_for(image_value)
        if image_meta is None:
            raise ValueError(
                "Unable to backfill LVIS metadata for image "
                f"{image_value!r} from {annotations_path}"
            )

        gt_objects_raw = row.get("gt")
        if not isinstance(gt_objects_raw, list):
            raise ValueError(
                "LVIS eval metadata backfill requires canonical `gt` objects in "
                f"gt_vs_pred rows for image {image_value!r}"
            )

        gt_categories: List[Dict[str, Any]] = []
        positive_category_ids: List[int] = []
        for obj in gt_objects_raw:
            if not isinstance(obj, Mapping):
                continue
            category_entry = index.category_entry_for_desc(str(obj.get("desc") or ""))
            if category_entry is None:
                raise ValueError(
                    "Unable to map GT desc to an LVIS category while backfilling "
                    f"metadata for image {image_value!r}: desc={obj.get('desc')!r}"
                )
            gt_categories.append(dict(category_entry))
            positive_category_ids.append(int(category_entry["category_id"]))

        enriched = dict(row)
        enriched["image_id"] = int(image_meta["image_id"])
        enriched["metadata"] = {
            "dataset": "lvis",
            "dataset_policy": "lvis_federated",
            "image_id": int(image_meta["image_id"]),
            "split": _infer_lvis_split(gt_jsonl),
            "lvis": {
                "gt_objects": list(gt_categories),
                "positive_categories": index.category_entries_from_ids(
                    positive_category_ids
                ),
                "neg_categories": index.category_entries_from_ids(
                    image_meta["neg_category_ids"]
                ),
                "not_exhaustive_categories": index.category_entries_from_ids(
                    image_meta["not_exhaustive_category_ids"]
                ),
            },
        }
        out.append(enriched)
    return out


class Stage1DetectionEvalCallback(TrainerCallback):
    """Run generation-backed detection eval during Stage-1 evaluation windows."""

    def __init__(
        self,
        *,
        gt_jsonl: str,
        output_root: str,
        model_checkpoint: str,
        prompt_variant: str,
        bbox_format: str = "xyxy",
        object_field_order: str,
        object_ordering: str,
        metrics: str,
        use_segm: bool,
        strict_parse: bool,
        iou_thrs: Optional[Sequence[float]],
        f1ish_iou_thrs: Sequence[float],
        f1ish_pred_scope: str,
        semantic_model: str,
        semantic_threshold: float,
        semantic_device: str,
        semantic_batch_size: int,
        lvis_max_dets: int,
        pred_score_source: str,
        pred_score_version: int,
        constant_score: float,
        batch_size: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        limit: Optional[int],
        seed: Optional[int],
        lvis_annotations_json: Optional[str],
    ) -> None:
        self.gt_jsonl = Path(gt_jsonl)
        self.output_root = Path(output_root)
        self.model_checkpoint = str(model_checkpoint)
        self.prompt_variant = str(prompt_variant)
        self.bbox_format = str(bbox_format)
        self.object_field_order = str(object_field_order)
        self.object_ordering = str(object_ordering)
        self.bbox_format = str(bbox_format)
        self.metrics_mode = str(metrics).strip().lower()
        self.eval_options = EvalOptions(
            metrics=self.metrics_mode,
            strict_parse=bool(strict_parse),
            use_segm=bool(use_segm),
            iou_thrs=[float(value) for value in list(iou_thrs or [])] or None,
            f1ish_iou_thrs=[float(value) for value in list(f1ish_iou_thrs)],
            f1ish_pred_scope=str(f1ish_pred_scope),
            semantic_model=str(semantic_model),
            semantic_threshold=float(semantic_threshold),
            semantic_device=str(semantic_device),
            semantic_batch_size=int(semantic_batch_size),
            lvis_max_dets=int(lvis_max_dets),
        )
        self.pred_score_source = str(pred_score_source)
        self.pred_score_version = int(pred_score_version)
        self.constant_score = float(constant_score)
        self.limit = int(limit) if limit is not None else None
        self.seed = int(seed) if seed is not None else None
        self.lvis_annotations_json = (
            str(lvis_annotations_json) if lvis_annotations_json is not None else None
        )
        self.gen_cfg = GenerationConfig(
            temperature=float(temperature),
            top_p=float(top_p),
            max_new_tokens=int(max_new_tokens),
            repetition_penalty=float(repetition_penalty),
            batch_size=int(batch_size),
            seed=self.seed,
        )
        self._processor: Any | None = None

    def _eval_dir(self, state: TrainerState) -> Path:
        step = int(getattr(state, "global_step", 0) or 0)
        return self.output_root / "eval_detection" / f"step_{step:07d}"

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not _is_world_process_zero(args=args, state=state):
            return

        model = kwargs.get("model")
        if model is None:
            raise RuntimeError(
                "Stage1DetectionEvalCallback requires the live trainer model during on_evaluate()."
            )

        runtime_model = _unwrap_runtime_model(model)
        device = _resolve_runtime_device(runtime_model, args)
        eval_dir = self._eval_dir(state)
        eval_dir.mkdir(parents=True, exist_ok=True)

        infer_cfg = InferenceConfig(
            gt_jsonl=str(self.gt_jsonl),
            model_checkpoint=self.model_checkpoint,
            mode="auto",
            prompt_variant=self.prompt_variant,
            bbox_format=self.bbox_format,
            object_field_order=self.object_field_order,
            object_ordering=self.object_ordering,
            pred_coord_mode="auto",
            out_path=str(eval_dir / "gt_vs_pred.jsonl"),
            summary_path=str(eval_dir / "infer_summary.json"),
            device=device,
            limit=int(self.limit or 0),
            backend_type="hf",
        )
        engine = InferenceEngine(infer_cfg, self.gen_cfg, logger=logger)
        engine.model = runtime_model
        if self._processor is not None:
            engine.processor = self._processor

        was_training = bool(getattr(runtime_model, "training", False))
        runtime_model.eval()
        try:
            base_jsonl_path, _summary_path = engine.infer()
        finally:
            if was_training:
                runtime_model.train()
        self._processor = engine.processor

        base_rows = _load_jsonl(base_jsonl_path)
        base_rows = _maybe_backfill_lvis_metadata(
            records=base_rows,
            metrics_mode=self.metrics_mode,
            gt_jsonl=self.gt_jsonl,
            lvis_annotations_json=self.lvis_annotations_json,
        )
        _write_jsonl(base_jsonl_path, base_rows)

        want_official = self.metrics_mode in {"coco", "lvis", "both"}
        pred_jsonl_path = base_jsonl_path
        if want_official:
            scored_rows = with_constant_scores(
                records=base_rows,
                pred_score_source=self.pred_score_source,
                pred_score_version=self.pred_score_version,
                constant_score=self.constant_score,
            )
            pred_jsonl_path = eval_dir / "gt_vs_pred_scored.jsonl"
            _write_jsonl(pred_jsonl_path, scored_rows)

        options = EvalOptions(
            metrics=self.eval_options.metrics,
            strict_parse=self.eval_options.strict_parse,
            use_segm=self.eval_options.use_segm,
            iou_thrs=list(self.eval_options.iou_thrs)
            if self.eval_options.iou_thrs is not None
            else None,
            f1ish_iou_thrs=list(self.eval_options.f1ish_iou_thrs),
            f1ish_pred_scope=str(self.eval_options.f1ish_pred_scope),
            output_dir=eval_dir,
            overlay=False,
            overlay_k=0,
            num_workers=0,
            semantic_model=str(self.eval_options.semantic_model),
            semantic_threshold=float(self.eval_options.semantic_threshold),
            semantic_device=str(self.eval_options.semantic_device),
            semantic_batch_size=int(self.eval_options.semantic_batch_size),
            lvis_max_dets=int(self.eval_options.lvis_max_dets),
        )
        summary = evaluate_and_save(pred_jsonl_path, options=options)
        det_metrics = dict(summary.get("metrics", {}))

        if metrics is not None:
            for key, value in det_metrics.items():
                metrics[f"eval_det_{key}"] = value

        logger.info(
            "Stage-1 detection eval finished at step=%s metrics=%s",
            int(getattr(state, "global_step", 0) or 0),
            det_metrics,
        )
