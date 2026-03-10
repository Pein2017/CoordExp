#!/usr/bin/env python
"""Export an official COCO detection submission JSON from scored CoordExp predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from src.eval.detection import EvalOptions, export_coco_submission
from src.utils import get_logger

logger = get_logger(__name__)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "YAML config requires PyYAML (import yaml). Install it in the ms env."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("submission export config must be a YAML mapping")
    return data


def _get(cfg: Mapping[str, Any], key: str, default: Any) -> Any:
    return cfg[key] if key in cfg else default


def _resolve_path(cli_value: Path | None, yaml_value: Any) -> Path | None:
    if cli_value is not None:
        return cli_value
    if yaml_value is None:
        return None
    text = str(yaml_value).strip()
    if not text:
        return None
    return Path(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export official COCO detection submission JSON from scored CoordExp predictions."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML config")
    parser.add_argument(
        "--pred_jsonl", type=Path, default=None, help="Scored artifact JSONL"
    )
    parser.add_argument(
        "--source_jsonl",
        type=Path,
        default=None,
        help="Source COCO JSONL with original image_id values (e.g. raw/test-dev.jsonl)",
    )
    parser.add_argument(
        "--categories_json",
        type=Path,
        default=None,
        help="COCO categories.json produced by the converter",
    )
    parser.add_argument(
        "--out_json",
        type=Path,
        default=None,
        help="Output submission JSON path",
    )
    parser.add_argument(
        "--strict-parse",
        action="store_true",
        help="Abort on malformed prediction rows instead of best-effort warnings.",
    )
    parser.add_argument(
        "--semantic-model",
        default=None,
        help="HF model id used for semantic label mapping.",
    )
    parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold for semantic category mapping.",
    )
    parser.add_argument(
        "--semantic-device",
        default=None,
        help="Device for semantic label mapping: auto|cpu|cuda[:N].",
    )
    parser.add_argument(
        "--semantic-batch-size",
        type=int,
        default=None,
        help="Batch size for semantic embedding encoding.",
    )
    return parser.parse_args()


def _resolve_from_yaml(
    ycfg: Mapping[str, Any], args: argparse.Namespace
) -> tuple[Path, Path, Path, Path, EvalOptions]:
    pred_jsonl = _resolve_path(args.pred_jsonl, _get(ycfg, "pred_jsonl", None))
    source_jsonl = _resolve_path(args.source_jsonl, _get(ycfg, "source_jsonl", None))
    categories_json = _resolve_path(
        args.categories_json, _get(ycfg, "categories_json", None)
    )
    out_json = _resolve_path(
        args.out_json, _get(ycfg, "out_json", "coco_submission.json")
    )

    for key, value in (
        ("pred_jsonl", pred_jsonl),
        ("source_jsonl", source_jsonl),
        ("categories_json", categories_json),
        ("out_json", out_json),
    ):
        if value is None:
            raise ValueError(f"{key} must be set (either in YAML or via CLI)")

    strict_parse = (
        bool(args.strict_parse)
        if args.strict_parse
        else bool(_get(ycfg, "strict_parse", True))
    )

    options = EvalOptions(
        metrics="coco",
        strict_parse=strict_parse,
        use_segm=False,
        output_dir=out_json.parent,
        overlay=False,
        num_workers=0,
        semantic_model=str(
            args.semantic_model
            or _get(ycfg, "semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        semantic_threshold=float(
            args.semantic_threshold or _get(ycfg, "semantic_threshold", 0.6)
        ),
        semantic_device=str(
            args.semantic_device or _get(ycfg, "semantic_device", "auto")
        ),
        semantic_batch_size=int(
            args.semantic_batch_size or _get(ycfg, "semantic_batch_size", 64)
        ),
    )
    return pred_jsonl, source_jsonl, categories_json, out_json, options


def _resolve_legacy(
    args: argparse.Namespace,
) -> tuple[Path, Path, Path, Path, EvalOptions]:
    required = {
        "pred_jsonl": args.pred_jsonl,
        "source_jsonl": args.source_jsonl,
        "categories_json": args.categories_json,
        "out_json": args.out_json,
    }
    missing = [key for key, value in required.items() if value is None]
    if missing:
        raise ValueError(
            "Without --config, the following flags are required: "
            + ", ".join(f"--{key}" for key in missing)
        )

    options = EvalOptions(
        metrics="coco",
        strict_parse=bool(args.strict_parse),
        use_segm=False,
        output_dir=args.out_json.parent,
        overlay=False,
        num_workers=0,
        semantic_model=str(
            args.semantic_model or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        semantic_threshold=float(args.semantic_threshold or 0.6),
        semantic_device=str(args.semantic_device or "auto"),
        semantic_batch_size=int(args.semantic_batch_size or 64),
    )
    return (
        args.pred_jsonl,
        args.source_jsonl,
        args.categories_json,
        args.out_json,
        options,
    )


def main() -> None:
    args = parse_args()

    if args.config is not None:
        cfg = _load_yaml(args.config)
        pred_jsonl, source_jsonl, categories_json, out_json, options = (
            _resolve_from_yaml(cfg, args)
        )
    else:
        pred_jsonl, source_jsonl, categories_json, out_json, options = _resolve_legacy(
            args
        )

    resolved = {
        "pred_jsonl": str(pred_jsonl),
        "source_jsonl": str(source_jsonl),
        "categories_json": str(categories_json),
        "out_json": str(out_json),
        "semantic_model": options.semantic_model,
        "semantic_threshold": float(options.semantic_threshold),
        "semantic_device": options.semantic_device,
        "semantic_batch_size": int(options.semantic_batch_size),
        "strict_parse": bool(options.strict_parse),
    }
    logger.info("Resolved COCO submission export config: %s", json.dumps(resolved))

    summary = export_coco_submission(
        pred_jsonl,
        source_jsonl=source_jsonl,
        categories_json=categories_json,
        out_json=out_json,
        options=options,
    )
    print(
        "COCO submission export complete. "
        f"submission_json={out_json} predictions_total={summary['predictions_total']}"
    )


if __name__ == "__main__":
    main()
