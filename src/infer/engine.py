"""Centralized inference engine for CoordExp.

Features
--------
- Single entrypoint that works for coord-token (normalized) and pure-text
  checkpoints via an explicit ``mode`` switch.
- Standardized ``pred.jsonl`` with pixel-space geometries for both GT and
  predictions; polygons and lines are preserved.
- Per-sample error reporting plus run-level counters and summary JSON.
- Deterministic generation when ``--seed`` is provided (torch + CUDA +
  generator passed into ``model.generate``).

Schema (per line of ``pred.jsonl``)
-----------------------------------
```
{
  "index": int,
  "image": str,              # relative or absolute path
  "width": int,
  "height": int,
  "mode": "coord" | "text",
  "coord_mode": "norm1000" | "pixel" | null,  # optional trace
  "gt": [ {"type","points","points_text","desc","score","_coord_mode"} ],
  "pred": [ {"type","points","points_text","desc","score","_coord_mode"} ],
  "raw_output": str,         # generation text (may be empty on early skip)
  "errors": ["..."]         # empty when none
}
```

Summary (``pred.summary.json``)
-------------------------------
```
{
  "mode": "coord" | "text",
  "total_read": int,
  "total_emitted": int,
  "counters": {"invalid_json": 0, ...},
  "error_codes": ["invalid_coord", ...]
}
```

Notes
-----
- ``gt`` and ``pred`` points are always pixel-space when emitted.
- ``points_text`` mirrors the pixel coords as a text string to unify downstream
  consumers that expect pure-text geometry.
- Polygons are kept verbatim (single ring) for COCO segmentation; lines are
  tolerated structurally but excluded from metrics downstream.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

from src.common.coord_standardizer import CoordinateStandardizer
from src.config.prompts import SYSTEM_PROMPT, USER_PROMPT
from src.utils import get_logger

# Map fine-grained error tags to canonical counter buckets.
ERROR_CANONICAL = {
    "geometry_keys": "invalid_geometry",
    "geometry_points": "invalid_geometry",
    "bbox_points": "invalid_geometry",
    "poly_points": "invalid_geometry",
    "line_points": "invalid_geometry",
    "degenerate": "invalid_geometry",
    "coord_parse": "invalid_coord",
    "coord_range": "invalid_coord",
    "mode_gt_mismatch": "mode_gt_mismatch",
    "size_mismatch": "size_mismatch",
    "empty_pred": "empty_pred",
    "generation_failed": "generation_failed",
    "image_load_failed": "image_load_failed",
    "multi_image_not_supported": "multi_image_not_supported",
}


@dataclass
class GenerationConfig:
    temperature: float = 0.01
    top_p: float = 0.95
    max_new_tokens: int = 1024
    repetition_penalty: float = 1.05
    seed: Optional[int] = None


@dataclass
class InferenceConfig:
    gt_jsonl: str
    model_checkpoint: str
    mode: Literal["coord", "text"]
    pred_coord_mode: Literal["auto", "norm1000", "pixel"] = "auto"
    out_path: str = "pred.jsonl"
    summary_path: Optional[str] = None
    device: str = "cuda:0"
    limit: int = 0


class RunCounters:
    """Aggregated counters for run-level summary."""

    def __init__(self) -> None:
        self.counts: Dict[str, int] = {}
        self.error_codes: set[str] = set()
        self.total_read: int = 0
        self.total_emitted: int = 0

    def add(self, code: str) -> None:
        self.counts[code] = self.counts.get(code, 0) + 1
        self.error_codes.add(code)

    def to_summary(self) -> Dict[str, Any]:
        return {
            "counters": self.counts,
            "error_codes": sorted(self.error_codes),
            "total_read": self.total_read,
            "total_emitted": self.total_emitted,
        }


class InferenceEngine:
    def __init__(
        self,
        cfg: InferenceConfig,
        gen_cfg: GenerationConfig,
        *,
        logger=None,
    ) -> None:
        self.cfg = cfg
        self.gen_cfg = gen_cfg
        self.logger = logger or get_logger(__name__)
        self.coord = CoordinateStandardizer(
            cfg.mode, pred_coord_mode=cfg.pred_coord_mode
        )
        self.processor: AutoProcessor | None = None
        self.model: Qwen3VLForConditionalGeneration | None = None
        self.generator: torch.Generator | None = None

    def _seed(self) -> None:
        if self.gen_cfg.seed is None:
            return
        torch.manual_seed(self.gen_cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.gen_cfg.seed)
        self.generator = torch.Generator(device=self.cfg.device)
        self.generator.manual_seed(self.gen_cfg.seed)

    def load_model(self) -> None:
        if self.model is not None:
            return
        self._seed()
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.cfg.model_checkpoint,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        ).to(self.cfg.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model_checkpoint, trust_remote_code=True
        )

    @staticmethod
    def _resolve_image_path(jsonl_path: Path, image_rel: str) -> Path:
        if os.path.isabs(image_rel):
            return Path(image_rel)
        root = os.environ.get("ROOT_IMAGE_DIR")
        base = Path(root) if root else jsonl_path.parent
        return (base / image_rel).resolve()

    def _build_messages(self, image: Image.Image) -> List[Dict[str, Any]]:
        user_prompt = USER_PROMPT
        system_prompt = SYSTEM_PROMPT
        return [
            {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image", "image": image},
                ],
            },
        ]

    def _generate(self, image: Image.Image) -> str:
        assert self.model is not None and self.processor is not None

        messages = self._build_messages(image)
        prompt_text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        model_inputs = self.processor(
            text=prompt_text, images=[image], return_tensors="pt"
        )
        model_inputs = {k: v.to(self.cfg.device) for k, v in model_inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=self.gen_cfg.max_new_tokens,
            do_sample=self.gen_cfg.temperature > 0,
            temperature=max(1e-4, self.gen_cfg.temperature),
            top_p=self.gen_cfg.top_p,
            use_cache=True,
        )
        if self.gen_cfg.repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = self.gen_cfg.repetition_penalty
        # NOTE: Do not pass `generator=` into `model.generate()`.
        #
        # Some upstream / remote-code model implementations (incl. some Qwen3-VL
        # checkpoints) treat unknown kwargs as `model_kwargs` and raise:
        #   "The following `model_kwargs` are not used by the model: ['generator']"
        #
        # We still seed torch/CUDA globally in `_seed()` for deterministic sampling.

        with torch.inference_mode():
            gen_ids = self.model.generate(**model_inputs, **gen_kwargs)

        prompt_len = model_inputs["input_ids"].shape[1]
        gen_only = gen_ids[:, prompt_len:]
        raw_text = self.processor.tokenizer.batch_decode(
            gen_only, skip_special_tokens=False, clean_up_tokenization_spaces=False
        )[0]
        return raw_text

    def _process_gt(
        self,
        record: Dict[str, Any],
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        return self.coord.process_record_gt(
            record, width=width, height=height, errors=errors
        )

    def _process_pred(
        self,
        raw_text: str,
        *,
        width: int,
        height: int,
        errors: List[str],
    ) -> List[Dict[str, Any]]:
        return self.coord.process_prediction_text(
            raw_text, width=width, height=height, errors=errors
        )

    def _prepare_image(
        self, jsonl_path: Path, record: Dict[str, Any]
    ) -> Tuple[Optional[Path], Optional[Image.Image]]:
        images = record.get("images") or []
        if len(images) != 1:
            return None, None
        img_path = self._resolve_image_path(jsonl_path, images[0])
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            return img_path, None
        return img_path, image

    def infer(self) -> Tuple[Path, Path]:
        jsonl_path = Path(self.cfg.gt_jsonl)
        out_path = Path(self.cfg.out_path)
        summary_path = Path(
            self.cfg.summary_path or out_path.with_suffix(".summary.json")
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.parent.mkdir(parents=True, exist_ok=True)

        if not os.environ.get("ROOT_IMAGE_DIR"):
            os.environ["ROOT_IMAGE_DIR"] = str(jsonl_path.parent.resolve())

        counters = RunCounters()
        self.load_model()

        pbar_total = self.cfg.limit if self.cfg.limit > 0 else None
        with (
            jsonl_path.open("r", encoding="utf-8") as fin,
            out_path.open("w", encoding="utf-8") as fout,
            tqdm(
                total=pbar_total,
                desc="Infer",
                unit="samples",
                dynamic_ncols=True,
                smoothing=0.1,
                mininterval=1.0,
                bar_format="{l_bar}{bar}| {n_fmt} {unit} [{elapsed}, {rate_fmt}]",
            ) as pbar,
        ):
            for line in fin:
                if self.cfg.limit and counters.total_emitted >= self.cfg.limit:
                    break
                pbar.update(1)
                line = line.strip()
                if not line:
                    continue
                counters.total_read += 1
                try:
                    record = json.loads(line)
                except Exception:
                    counters.add("invalid_json")
                    continue

                errors: List[str] = []
                width = record.get("width")
                height = record.get("height")
                try:
                    width = int(width)
                    height = int(height)
                except Exception:
                    width = None
                    height = None

                if not width or not height:
                    errors.append("size_mismatch")
                    output = {
                        "index": counters.total_read - 1,
                        "image": (record.get("images") or [None])[0],
                        "width": width,
                        "height": height,
                        "mode": self.cfg.mode,
                        "coord_mode": None,
                        "gt": [],
                        "pred": [],
                        "raw_output": "",
                        "errors": errors,
                    }
                    fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                    for code in errors:
                        counters.add(ERROR_CANONICAL.get(code, code))
                    counters.total_emitted += 1
                    continue

                # Process GT first to catch mode mismatches early.
                # Validate GT geometries early.
                gt_errors: List[str] = []
                gt = self._process_gt(
                    record, width=width, height=height, errors=gt_errors
                )
                errors.extend(gt_errors)

                # Mode/GT mismatch detected by processor -> skip generation but still emit record.
                run_generation = "mode_gt_mismatch" not in errors

                images = record.get("images") or []
                img_path: Optional[Path]
                image: Optional[Image.Image]
                if len(images) != 1:
                    errors.append("multi_image_not_supported")
                    img_path, image = None, None
                    run_generation = False
                else:
                    img_path, image = self._prepare_image(jsonl_path, record)
                    if image is None:
                        errors.append("image_load_failed")
                        run_generation = False

                raw_output = ""
                pred: List[Dict[str, Any]] = []
                if run_generation:
                    try:
                        raw_output = self._generate(image)
                        pred_errors: List[str] = []
                        pred = self._process_pred(
                            raw_output, width=width, height=height, errors=pred_errors
                        )
                        errors.extend(pred_errors)
                    except Exception as exc:  # noqa: BLE001
                        errors.append("generation_failed")
                        raw_output = str(exc)

                # Aggregate per-sample errors into counters.
                for code in errors:
                    counters.add(ERROR_CANONICAL.get(code, code))

                output = {
                    "index": counters.total_read - 1,
                    "image": (record.get("images") or [None])[0],
                    "width": width,
                    "height": height,
                    "mode": self.cfg.mode,
                    # Points are emitted in pixel space; hint downstream to skip re-denorm.
                    "coord_mode": "pixel",
                    "gt": gt,
                    "pred": pred,
                    "raw_output": raw_output,
                    "errors": errors,
                }
                fout.write(json.dumps(output, ensure_ascii=False) + "\n")
                counters.total_emitted += 1

        summary_payload = {
            "mode": self.cfg.mode,
            **counters.to_summary(),
        }
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        self.logger.info(
            "Inference finished: %s samples emitted, summary=%s",
            counters.total_emitted,
            summary_path,
        )
        return out_path, summary_path


__all__ = ["GenerationConfig", "InferenceConfig", "InferenceEngine"]
