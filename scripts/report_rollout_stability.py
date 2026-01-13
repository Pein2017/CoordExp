#!/usr/bin/env python
"""Report rollout stability / parsability from CoordExp inference outputs.

This script is meant as a lightweight "are we EM-ish ready?" gate:
- reads `pred.jsonl` emitted by `scripts/run_infer.py`
- summarizes parse/geometry/coord error rates and basic object-count stats
- optionally reads inference `summary.json` and evaluator `eval/metrics.json`

Example:
  /root/miniconda3/envs/ms/bin/python scripts/report_rollout_stability.py \\
    --pred_jsonl output/infer/coord_loss_ckpt3106_val200/pred.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Tuple


GEOM_ERRORS = {
    "geometry_keys",
    "geometry_points",
    "bbox_points",
    "poly_points",
    "line_points",
    "degenerate",
}
COORD_ERRORS = {"coord_parse", "coord_range"}
GEN_SKIP_ERRORS = {"mode_gt_mismatch", "image_load_failed", "multi_image_not_supported", "size_mismatch"}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                yield obj


def _infer_paths(pred_jsonl: Path, *, summary_json: Optional[Path], eval_metrics_json: Optional[Path]) -> Tuple[Optional[Path], Optional[Path]]:
    inferred_summary = summary_json
    inferred_eval = eval_metrics_json

    if inferred_summary is None:
        # `scripts/run_infer_eval.sh` uses `summary.json` in the output dir.
        candidate = pred_jsonl.parent / "summary.json"
        if candidate.exists():
            inferred_summary = candidate
        else:
            # `scripts/run_infer.py` default is sibling ".summary.json"
            candidate = pred_jsonl.with_suffix(".summary.json")
            if candidate.exists():
                inferred_summary = candidate

    if inferred_eval is None:
        candidate = pred_jsonl.parent / "eval" / "metrics.json"
        if candidate.exists():
            inferred_eval = candidate

    return inferred_summary, inferred_eval


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize CoordExp rollout stability / parsability.")
    ap.add_argument("--pred_jsonl", type=Path, required=True, help="Path to pred.jsonl produced by scripts/run_infer.py")
    ap.add_argument("--summary_json", type=Path, default=None, help="Optional inference summary JSON (default: auto-detect)")
    ap.add_argument("--eval_metrics_json", type=Path, default=None, help="Optional evaluator metrics.json (default: auto-detect)")
    ap.add_argument("--max_examples", type=int, default=5, help="Max failure examples to print")
    ap.add_argument("--max_raw_chars", type=int, default=240, help="Max raw_output chars to print per example")
    args = ap.parse_args()

    if not args.pred_jsonl.exists():
        raise SystemExit(f"pred_jsonl not found: {args.pred_jsonl}")

    summary_path, eval_path = _infer_paths(
        args.pred_jsonl, summary_json=args.summary_json, eval_metrics_json=args.eval_metrics_json
    )

    summary = _read_json(summary_path) if summary_path else None
    eval_metrics = _read_json(eval_path) if eval_path else None

    records = list(_iter_jsonl(args.pred_jsonl))
    if not records:
        raise SystemExit(f"No valid JSON records found in: {args.pred_jsonl}")

    total = len(records)
    error_counter: Counter[str] = Counter()
    coord_err = 0
    geom_err = 0
    empty_pred = 0
    gen_failed = 0
    gen_skipped = 0

    pred_counts: List[int] = []
    gt_counts: List[int] = []

    failure_examples: List[Tuple[int, str, List[str], str]] = []

    attempted = 0
    has_any_pred = 0

    for rec in records:
        errors = rec.get("errors") or []
        if not isinstance(errors, list):
            errors = []
        errors = [str(e) for e in errors]
        for e in errors:
            error_counter[e] += 1

        gt = rec.get("gt") or []
        pred = rec.get("pred") or []
        if not isinstance(gt, list):
            gt = []
        if not isinstance(pred, list):
            pred = []

        gt_counts.append(len(gt))
        pred_counts.append(len(pred))

        has_coord = any(e in COORD_ERRORS for e in errors)
        has_geom = any(e in GEOM_ERRORS for e in errors)
        if has_coord:
            coord_err += 1
        if has_geom:
            geom_err += 1
        if "empty_pred" in errors:
            empty_pred += 1
        if "generation_failed" in errors:
            gen_failed += 1
        if any(e in GEN_SKIP_ERRORS for e in errors):
            gen_skipped += 1

        # Generation attempted if we didn't hard-skip it due to obvious preconditions.
        did_skip = any(e in GEN_SKIP_ERRORS for e in errors)
        if not did_skip:
            attempted += 1
            if len(pred) > 0:
                has_any_pred += 1

        # Capture a few examples for quick qualitative debugging.
        if args.max_examples > 0 and len(failure_examples) < args.max_examples:
            if ("empty_pred" in errors) or ("generation_failed" in errors) or has_coord or has_geom:
                idx = int(rec.get("index", -1))
                image = str(rec.get("image", ""))
                raw = str(rec.get("raw_output", "") or "")
                raw = raw.replace("\n", "\\n")
                if len(raw) > args.max_raw_chars:
                    raw = raw[: args.max_raw_chars] + "…"
                failure_examples.append((idx, image, errors, raw))

    avg_pred = sum(pred_counts) / len(pred_counts)
    avg_gt = sum(gt_counts) / len(gt_counts)

    attempted = max(attempted, 1)  # avoid div-by-zero if something went very wrong
    parse_rate = has_any_pred / attempted

    print("=== Rollout Stability Report ===")
    print(f"pred_jsonl: {args.pred_jsonl}")
    if summary_path:
        print(f"infer_summary: {summary_path} ({'loaded' if summary else 'unreadable'})")
    if eval_path:
        print(f"eval_metrics: {eval_path} ({'loaded' if eval_metrics else 'unreadable'})")

    print("")
    print("--- Inference-level stats (from pred.jsonl) ---")
    print(f"samples_emitted: {total}")
    print(f"generation_attempted: {attempted} ({_fmt_pct(attempted / total)})")
    print(f"generation_skipped: {gen_skipped} ({_fmt_pct(gen_skipped / total)})")
    print(f"has_nonempty_pred (attempted only): {has_any_pred}/{attempted} ({_fmt_pct(parse_rate)})")
    print(f"empty_pred (all samples): {empty_pred} ({_fmt_pct(empty_pred / total)})")
    print(f"coord_error_samples: {coord_err} ({_fmt_pct(coord_err / total)})")
    print(f"geom_error_samples: {geom_err} ({_fmt_pct(geom_err / total)})")
    print(f"generation_failed: {gen_failed} ({_fmt_pct(gen_failed / total)})")

    print("")
    print("--- Object count sanity ---")
    print(f"gt_count: mean={avg_gt:.2f} median={median(gt_counts):.0f} min={min(gt_counts)} max={max(gt_counts)}")
    print(f"pred_count: mean={avg_pred:.2f} median={median(pred_counts):.0f} min={min(pred_counts)} max={max(pred_counts)}")

    print("")
    print("--- Top error tags (from pred.jsonl errors[]) ---")
    for k, v in error_counter.most_common(12):
        print(f"{k}: {v} ({_fmt_pct(v / total)})")

    if summary:
        counters = summary.get("counters", {})
        if isinstance(counters, dict):
            print("")
            print("--- Inference summary.json counters (canonical buckets) ---")
            for k in sorted(counters.keys()):
                try:
                    v = int(counters[k])
                except Exception:
                    continue
                print(f"{k}: {v}")

    if eval_metrics:
        metrics = eval_metrics.get("metrics", {})
        counters = eval_metrics.get("counters", {})
        if isinstance(metrics, dict):
            print("")
            print("--- Eval metrics.json (COCO-style) ---")
            for k in ["bbox_AP", "bbox_AP50", "bbox_AP75", "bbox_AR100", "segm_AP", "segm_AP50", "segm_AP75", "segm_AR100"]:
                if k in metrics:
                    try:
                        print(f"{k}: {float(metrics[k]):.6g}")
                    except Exception:
                        pass
        if isinstance(counters, dict):
            print("")
            print("--- Eval counters (parsing/robustness) ---")
            for k in sorted(counters.keys()):
                try:
                    v = int(counters[k])
                except Exception:
                    continue
                print(f"{k}: {v}")

    if failure_examples:
        print("")
        print(f"--- Failure examples (up to {args.max_examples}) ---")
        for idx, image, errors, raw in failure_examples:
            print(f"[index={idx}] image={image}")
            print(f"  errors={errors}")
            print(f"  raw_output={raw}")


if __name__ == "__main__":
    main()

