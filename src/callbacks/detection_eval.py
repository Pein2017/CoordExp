"""
Training-time hook to run the offline detection evaluator and log metrics.

This callback assumes predictions JSONL are provided (e.g., generated separately
with vis_tools/vis_coordexp.py). It does not trigger generation itself.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from src.eval.detection import EvalOptions, evaluate_and_save
from src.utils import get_logger

logger = get_logger(__name__)


class DetectionEvalCallback(TrainerCallback):
    """Run detection evaluation during evaluation steps."""

    def __init__(
        self,
        *,
        pred_jsonl: str,
        out_dir: str = "eval_out",
        unknown_policy: str = "semantic",
        strict_parse: bool = False,
        use_segm: bool = True,
    ) -> None:
        self.pred_jsonl = Path(pred_jsonl)
        self.out_dir = Path(out_dir)
        if unknown_policy not in (None, "semantic"):
            logger.warning(
                "DetectionEvalCallback unknown_policy=%s is deprecated and ignored; description matching always uses sentence-transformers/all-MiniLM-L6-v2.",
                unknown_policy,
            )
        self.options = EvalOptions(
            strict_parse=strict_parse,
            use_segm=use_segm,
            output_dir=self.out_dir,
        )

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> None:
        if not torch.cuda.is_available():
            logger.warning("DetectionEvalCallback skipped: CUDA not available.")
            return

        summary = evaluate_and_save(self.pred_jsonl, options=self.options)
        det_metrics = summary.get("metrics", {})

        if metrics is not None:
            for key, val in det_metrics.items():
                metrics[f"eval_det_{key}"] = val
        logger.info("Detection evaluation metrics: %s", det_metrics)
