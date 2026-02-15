#!/usr/bin/env python
"""Performance monitor for CoordExp rollout output.

This script provides reusable checks for any new checkpoint rollout directory or
`pred/gt_vs_pred.jsonl`, with:
- inference-level format reliability (empty prediction, parse/geometry/coord errors)
- object-miss trend from evaluator `f1ish` metrics
- optional persistent logging of failed rollout samples

Examples:
  conda run -n ms python scripts/analysis/report_rollout_stability.py \\
    --run_dir output/bench/a_only_ckpt_6064 \\
    --run_dir output/bench/a_only_iter_1_ckpt_6064
  conda run -n ms python scripts/analysis/report_rollout_stability.py \\
    --pred_jsonl output/bench/a_only_ckpt_6064/gt_vs_pred.jsonl \\
    --dump_failed_rollout \\
    --failed_rollout_dump /tmp/a_only_failed.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


GEOM_ERRORS = {
    "geometry_keys",
    "geometry_points",
    "bbox_points",
    "poly_points",
    "degenerate",
}
COORD_ERRORS = {"coord_parse", "coord_range"}
GEN_SKIP_ERRORS = {"mode_gt_mismatch", "image_load_failed", "multi_image_not_supported", "size_mismatch"}
DEFAULT_IGNORE_ERRORS = {"wrong"}
PREDICTION_FILES = ("gt_vs_pred.jsonl", "pred.jsonl")
IGNORED_ERROR_CANONICAL_MAP = {"wrong_hallucination_prediction": "wrong"}


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


def _coerce_error_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _normalize_ignore_tags(tags: Sequence[str]) -> Tuple[Set[str], List[str]]:
    normalized: Set[str] = set()
    deprecated: List[str] = []
    for tag in tags:
        tag_norm = str(tag).strip()
        if not tag_norm:
            continue
        canonical = IGNORED_ERROR_CANONICAL_MAP.get(tag_norm, tag_norm)
        if canonical != tag_norm:
            deprecated.append(f"{tag_norm}->{canonical}")
        normalized.add(canonical)
    return normalized, sorted(set(deprecated))


def _resolve_image_path(run_dir: Path, image_value: Any, image_root: Optional[Path]) -> str:
    if not image_value:
        return ""
    image_rel = str(image_value).strip()
    if not image_rel:
        return ""
    candidate = Path(image_rel)
    if candidate.is_absolute():
        return str(candidate)
    if image_root is not None:
        return str((image_root / candidate).resolve())
    return str((run_dir / candidate))


def _coerce_raw_output(rec: Dict[str, Any]) -> Tuple[Any, str]:
    raw_output = rec.get("raw_output_json")
    if raw_output is None:
        raw_output = rec.get("raw_output")
    if raw_output is None:
        raw_output = rec.get("raw_output_text")
    if raw_output is None:
        raw_output = ""
    if isinstance(raw_output, (dict, list)):
        raw_text = json.dumps(raw_output, ensure_ascii=False)
    else:
        raw_text = str(raw_output)
    return raw_output, raw_text

def _infer_paths(
    pred_jsonl: Path, *, summary_json: Optional[Path], eval_metrics_json: Optional[Path]
) -> Tuple[Optional[Path], Optional[Path]]:
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


def _collect_run_specs(
    pred_jsonls: Sequence[Path],
    run_dirs: Sequence[Path],
    summary_json: Optional[Path],
    eval_metrics_json: Optional[Path],
) -> List[Tuple[str, Path, Path, Optional[Path], Optional[Path]]]:
    specs: List[Tuple[str, Path, Path, Optional[Path], Optional[Path]]] = []
    seen: Set[Path] = set()

    for raw_pred in pred_jsonls:
        pred_jsonl = raw_pred
        if not pred_jsonl.exists():
            raise SystemExit(f"pred_jsonl not found: {pred_jsonl}")
        run_dir = pred_jsonl.parent
        summary_path, eval_path = _infer_paths(pred_jsonl, summary_json=summary_json, eval_metrics_json=eval_metrics_json)
        run_name = run_dir.name
        specs.append((run_name, run_dir, pred_jsonl, summary_path, eval_path))
        seen.add(pred_jsonl.resolve())

    for raw_dir in run_dirs:
        run_dir = raw_dir
        if not run_dir.exists():
            raise SystemExit(f"run_dir not found: {run_dir}")
        run_name = run_dir.name
        pred_jsonl = raw_dir / PREDICTION_FILES[0]
        if pred_jsonl.exists():
            pass
        elif (pred_jsonl := raw_dir / PREDICTION_FILES[1]).exists():
            pass
        else:
            pred_jsonl = raw_dir / PREDICTION_FILES[0]
            raise SystemExit(f"run_dir missing prediction jsonl: {run_dir}")

        if pred_jsonl.resolve() in seen:
            continue
        seen.add(pred_jsonl.resolve())

        summary_path, eval_path = _infer_paths(pred_jsonl, summary_json=summary_json, eval_metrics_json=eval_metrics_json)
        specs.append((run_name, run_dir, pred_jsonl, summary_path, eval_path))
    return specs


def _extract_object_miss(
    metrics: Dict[str, Any], threshold: Optional[float]
) -> Tuple[Optional[float], Optional[int], Optional[int], Optional[float]]:
    pattern = re.compile(r"^f1ish@([0-9]+(?:\.[0-9]+)?)_(tp_full|fn_full)$")
    bucket: Dict[float, Dict[str, float]] = {}
    for k, v in metrics.items():
        m = pattern.match(str(k))
        if not m:
            continue
        thr = float(m.group(1))
        stat = bucket.setdefault(thr, {})
        stat[m.group(2)] = float(v)

    if not bucket:
        return None, None, None, None

    if threshold is None:
        selected = max(bucket.keys())
    else:
        selected = min(bucket.keys(), key=lambda t: abs(t - threshold))

    selected_bucket = bucket[selected]
    tp = int(selected_bucket.get("tp_full", 0.0))
    fn = int(selected_bucket.get("fn_full", 0.0))
    denom = tp + fn
    miss_rate = float(fn) / float(denom) if denom > 0 else None
    return selected, tp, fn, miss_rate


def _dump_failed_rollout_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_failure_row(
    run_name: str,
    run_dir: Path,
    local_idx: int,
    rec: Dict[str, Any],
    errors: List[str],
    ignore_errors: Set[str],
    image_root: Optional[Path],
) -> Dict[str, Any]:
    effective_errors = [e for e in errors if e not in ignore_errors]
    image = str(rec.get("image", ""))
    raw_sample, raw_text = _coerce_raw_output(rec)
    gt = rec.get("gt") or []
    pred = rec.get("pred") or []
    return {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "index": int(rec.get("index", local_idx)),
        "image": image,
        "image_path": _resolve_image_path(run_dir, image, image_root),
        "errors": errors,
        "effective_errors": effective_errors,
        "width": int(rec.get("width")) if isinstance(rec.get("width"), int) else rec.get("width"),
        "height": int(rec.get("height")) if isinstance(rec.get("height"), int) else rec.get("height"),
        "mode": rec.get("mode", ""),
        "coord_mode": rec.get("coord_mode", ""),
        "gt_count": len(gt) if isinstance(gt, list) else 0,
        "pred_count": len(pred) if isinstance(pred, list) else 0,
        "gt": gt if isinstance(gt, list) else [],
        "pred": pred if isinstance(pred, list) else [],
        "raw_sample": raw_sample,
        "raw_output": raw_text,
        "raw_output_len": len(raw_text),
        "raw_output_preview": raw_text.replace("\\n", "\\\\n")[:240],
    }


def _analyze_rollout(
    run_name: str,
    run_dir: Path,
    pred_jsonl: Path,
    summary_path: Optional[Path],
    eval_path: Optional[Path],
    ignore_error_tags: Set[str],
    object_miss_threshold: float,
    max_examples: int,
    max_raw_chars: int,
    image_root: Optional[Path],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    records = list(_iter_jsonl(pred_jsonl))
    if not records:
        raise SystemExit(f"No valid JSON records found in: {pred_jsonl}")

    summary = _read_json(summary_path) if summary_path else None
    eval_metrics = _read_json(eval_path) if eval_path else None

    total = len(records)
    error_counter: Counter[str] = Counter()
    coord_err = 0
    geom_err = 0
    empty_pred = 0
    gen_failed = 0
    gen_skipped = 0

    pred_counts: List[int] = []
    gt_counts: List[int] = []
    failure_rows: List[Dict[str, Any]] = []
    plot_rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []
    object_row_errors: List[Tuple[int, str, List[str], str]] = []

    attempted = 0
    has_any_pred = 0

    for local_idx, rec in enumerate(records):
        errors = _coerce_error_list(rec.get("errors"))
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
        has_fail = bool(errors)
        has_any_ignored = any(e in ignore_error_tags for e in errors)
        has_effective_fail = any(e not in ignore_error_tags for e in errors)

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

        did_skip = any(e in GEN_SKIP_ERRORS for e in errors)
        if not did_skip:
            attempted += 1
            if len(pred) > 0 and (not has_effective_fail or has_any_ignored):
                has_any_pred += 1

        should_dump_failure = (not pred and not did_skip) or ("empty_pred" in errors) or has_effective_fail
        row = _build_failure_row(
            run_name=run_name,
            run_dir=run_dir,
            local_idx=local_idx,
            rec=rec,
            errors=errors,
            ignore_errors=ignore_error_tags,
            image_root=image_root,
        )
        plot_rows.append(row)
        if should_dump_failure:
            failure_rows.append(row)
            if len(examples) < max_examples:
                raw = row["raw_output"]
                raw = raw.replace("\n", "\\n")
                if len(raw) > max_raw_chars:
                    raw = raw[:max_raw_chars] + "…"
                object_row_errors.append((int(rec.get("index", local_idx)), str(rec.get("image", "")), errors, raw))
                examples.append(row)

        if has_fail:
            if (not did_skip) and has_any_ignored:
                # Keep this as a non-fatal warning-style event.
                pass

    avg_pred = sum(pred_counts) / len(pred_counts)
    avg_gt = sum(gt_counts) / len(gt_counts)

    attempted = max(attempted, 1)
    parse_rate = has_any_pred / attempted

    object_miss_thr = None
    object_miss_tp = None
    object_miss_fn = None
    object_miss_rate = None
    if isinstance(eval_metrics, dict):
        metrics = eval_metrics.get("metrics", {})
        if isinstance(metrics, dict):
            object_miss_thr, object_miss_tp, object_miss_fn, object_miss_rate = _extract_object_miss(
                metrics, threshold=object_miss_threshold
            )

    format_success_no_ignored = has_any_pred / attempted if attempted else 0.0

    run_payload = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "pred_jsonl": str(pred_jsonl),
        "summary_json": str(summary_path) if summary_path else None,
        "eval_metrics_json": str(eval_path) if eval_path else None,
        "samples": total,
        "generation_attempted": attempted,
        "generation_skipped": gen_skipped,
        "format_success_rate": parse_rate,
        "format_success_rate_no_ignored_errors": format_success_no_ignored,
        "empty_pred": empty_pred,
        "coord_error_samples": coord_err,
        "geom_error_samples": geom_err,
        "generation_failed": gen_failed,
        "gt_count_mean": avg_gt,
        "gt_count_median": median(gt_counts),
        "pred_count_mean": avg_pred,
        "pred_count_median": median(pred_counts),
        "pred_count_min": min(pred_counts),
        "pred_count_max": max(pred_counts),
        "gt_count_min": min(gt_counts),
        "gt_count_max": max(gt_counts),
        "object_miss_threshold": object_miss_thr,
        "object_miss_tp": object_miss_tp,
        "object_miss_fn": object_miss_fn,
        "object_miss_rate": object_miss_rate,
        "error_top": [list(pair) for pair in error_counter.most_common(12)],
        "failed_rows": len(failure_rows),
    }
    if summary:
        run_payload["summary_counters"] = summary.get("counters", {})
    if isinstance(eval_metrics, dict):
        run_payload["eval_counters"] = eval_metrics.get("counters", {})

    failure_examples = [
        {
            "index": idx,
            "image": image,
            "errors": errs,
            "raw_output": raw,
        }
        for idx, image, errs, raw in object_row_errors
    ]
    return run_payload, failure_rows, failure_examples, plot_rows


def _fmt_pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize and compare CoordExp rollout outputs.")
    ap.add_argument(
        "--pred_jsonl",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path to pred-like jsonl (gt_vs_pred.jsonl or pred.jsonl). "
            "Repeat for multiple checkpoints."
        ),
    )
    ap.add_argument(
        "--run_dir",
        type=Path,
        action="append",
        default=[],
        help="Run directory containing pred.jsonl and optional summary/eval artifacts.",
    )
    ap.add_argument("--summary_json", type=Path, default=None, help="Optional inference summary JSON (default: auto-detect)")
    ap.add_argument("--eval_metrics_json", type=Path, default=None, help="Optional evaluator metrics.json (default: auto-detect)")
    ap.add_argument("--max_examples", type=int, default=5, help="Max failure examples to print")
    ap.add_argument("--max_raw_chars", type=int, default=240, help="Max raw_output chars to print per example")
    ap.add_argument(
        "--ignore_error_tag",
        action="append",
        default=[*DEFAULT_IGNORE_ERRORS],
        help=(
            "Error tag to ignore in format checks (repeatable, default=wrong). "
            "Useful when a noisy label source marks a prediction as a non-fatal wrong prediction."
        ),
    )
    ap.add_argument(
        "--dump_failed_rollout",
        action="store_true",
        help="Write failed samples into JSONL for post-mortem analysis.",
    )
    ap.add_argument(
        "--failed_rollout_dump",
        type=Path,
        default=None,
        help=(
            "Path to write JSONL failed samples. Defaults to <run_dir>/eval/failed_rollout_dossier.jsonl "
            "for single-run mode."
        ),
    )
    ap.add_argument(
        "--plot_rollout_dump",
        type=Path,
        default=None,
        help=(
            "Path to write JSONL all rows (gt-vs-pred) for plotting and error triage. "
            "If omitted with a single run, defaults to <run_dir>/eval/gt_vs_pred_plot.jsonl."
        ),
    )
    ap.add_argument(
        "--image_root",
        type=Path,
        default=None,
        help="Optional base directory to resolve absolute image_path.",
    )
    ap.add_argument(
        "--comparison_out",
        type=Path,
        default=None,
        help="Optional JSON path for machine-readable comparison output.",
    )
    ap.add_argument(
        "--object_miss_threshold",
        type=float,
        default=0.5,
        help="F1-ish IoU threshold used for object-miss metric selection.",
    )
    args = ap.parse_args()

    ignore_error_tags = list(args.ignore_error_tag or [])
    ignore_error_tags, canonicalized = _normalize_ignore_tags(ignore_error_tags)
    if canonicalized:
        print(
            "[warn] normalized deprecated ignore tags: "
            + ", ".join(canonicalized)
            + "; use canonical names."
        )
    if not args.pred_jsonl and not args.run_dir:
        raise SystemExit("At least one of --pred_jsonl or --run_dir is required.")

    run_specs = _collect_run_specs(args.pred_jsonl, args.run_dir, args.summary_json, args.eval_metrics_json)
    if not run_specs:
        raise SystemExit("No valid rollout run specs discovered.")

    summaries = []
    all_failed_rows: List[Dict[str, Any]] = []
    all_plot_rows: List[Dict[str, Any]] = []
    print("=== Rollout Stability Report ===")

    for run_name, run_dir, pred_jsonl, summary_path, eval_path in run_specs:
        summary_payload, failed_rows, failure_examples, plot_rows = _analyze_rollout(
            run_name,
            run_dir,
            pred_jsonl,
            summary_path,
            eval_path,
            ignore_error_tags=ignore_error_tags,
            object_miss_threshold=args.object_miss_threshold,
            max_examples=args.max_examples,
            max_raw_chars=args.max_raw_chars,
            image_root=args.image_root,
        )
        summaries.append(summary_payload)
        all_failed_rows.extend(failed_rows)
        all_plot_rows.extend(plot_rows)

        print("")
        print(f"--- Run: {run_name} ---")
        print(f"pred_jsonl: {pred_jsonl}")
        if summary_path:
            print(f"infer_summary: {summary_path} (loaded)")
        if eval_path:
            print(f"eval_metrics: {eval_path} (loaded)")

        total = int(summary_payload["samples"])
        attempted = int(summary_payload["generation_attempted"])
        gen_skipped = int(summary_payload["generation_skipped"])
        parse_rate = float(summary_payload["format_success_rate"])
        obj_miss_rate = summary_payload.get("object_miss_rate")
        object_thr = summary_payload.get("object_miss_threshold")
        print("")
        print("--- Inference-level stats (from pred.jsonl) ---")
        print(f"samples_emitted: {total}")
        print(f"generation_attempted: {attempted} ({_fmt_pct(attempted / total)})")
        print(f"generation_skipped: {gen_skipped} ({_fmt_pct(gen_skipped / total)})")
        print(f"has_nonempty_pred (attempted only): {int((parse_rate * attempted) + 0.5)}/{attempted} ({_fmt_pct(parse_rate)})")
        print(f"empty_pred (all samples): {summary_payload['empty_pred']} ({_fmt_pct(int(summary_payload['empty_pred']) / total)})")
        print(f"coord_error_samples: {summary_payload['coord_error_samples']} ({_fmt_pct(int(summary_payload['coord_error_samples']) / total)})")
        print(f"geom_error_samples: {summary_payload['geom_error_samples']} ({_fmt_pct(int(summary_payload['geom_error_samples']) / total)})")
        print(f"generation_failed: {summary_payload['generation_failed']} ({_fmt_pct(int(summary_payload['generation_failed']) / total)})")
        if obj_miss_rate is None:
            print("object-miss@f1ish: unavailable (f1ish metrics missing)")
        else:
            print(
                f"object-miss@f1ish@{object_thr:.2f}: "
                f"{summary_payload['object_miss_fn']} misses / "
                f"{summary_payload['object_miss_tp'] + summary_payload['object_miss_fn']} gt "
                f"({_fmt_pct(float(obj_miss_rate))})"
            )

        print("")
        print("--- Object count sanity ---")
        print(
            f"gt_count: mean={summary_payload['gt_count_mean']:.2f} "
            f"median={summary_payload['gt_count_median']:.0f} "
            f"min={summary_payload['gt_count_min']} max={summary_payload['gt_count_max']}"
        )
        print(
            f"pred_count: mean={summary_payload['pred_count_mean']:.2f} "
            f"median={summary_payload['pred_count_median']:.0f} "
            f"min={summary_payload['pred_count_min']} max={summary_payload['pred_count_max']}"
        )

        print("")
        print("--- Top error tags (from pred.jsonl errors[]) ---")
        for k, v in summary_payload["error_top"]:
            print(f"{k}: {v} ({_fmt_pct(v / total)})")

        if summary_payload.get("summary_counters"):
            print("")
            print("--- Inference summary.json counters ---")
            for key in sorted(summary_payload["summary_counters"]):
                print(f"{key}: {summary_payload['summary_counters'][key]}")

        if summary_payload.get("eval_counters"):
            print("")
            print("--- Eval counters (parsing/robustness) ---")
            for key in sorted(summary_payload["eval_counters"]):
                print(f"{key}: {summary_payload['eval_counters'][key]}")

        if failure_examples:
            print("")
            print(f"--- Failure examples (up to {args.max_examples}) ---")
            for entry in failure_examples:
                print(f"[index={entry['index']}] image={entry['image']}")
                print(f"  errors={entry['errors']}")
                print(f"  raw_output={entry['raw_output']}")

    if len(summaries) > 1:
        print("")
        print("=== Cross-run comparison (rank by object miss) ===")
        print("lower is better for object-miss, ignoring --ignore_error_tag when computing format success")
        ranked = sorted(
            summaries,
            key=lambda item: float(item.get("object_miss_rate") if item.get("object_miss_rate") is not None else 1.0),
        )
        for idx, row in enumerate(ranked, 1):
            run_name = str(row["run_name"])
            miss = row["object_miss_rate"]
            parse_rate = row["format_success_rate"]
            obj_miss_text = "n/a"
            if miss is not None:
                obj_miss_text = _fmt_pct(float(miss))
            print(
                f"{idx:>2}. {run_name:<36} "
                f"object_miss={obj_miss_text:<7}  "
                f"format_success={_fmt_pct(float(parse_rate)):>7}  "
                f"empty_pred={int(row['empty_pred']):>4}/{int(row['samples'])}"
            )

    if args.dump_failed_rollout:
        if args.failed_rollout_dump is not None:
            failed_out = args.failed_rollout_dump
        elif len(summaries) == 1:
            failed_out = run_specs[0][1] / "eval" / "failed_rollout_dossier.jsonl"
        else:
            failed_out = Path("failed_rollout_dossier.jsonl")
        _dump_failed_rollout_rows(failed_out, all_failed_rows)
        print("")
        print(f"=== Failed rollout dossier written: {failed_out} ===")
        print(f"records: {len(all_failed_rows)}")

    plot_out: Optional[Path] = args.plot_rollout_dump
    if args.plot_rollout_dump is None and len(summaries) == 1:
        plot_out = run_specs[0][1] / "eval" / "gt_vs_pred_plot.jsonl"
    if plot_out is not None:
        _dump_failed_rollout_rows(plot_out, all_plot_rows)
        print("")
        print(f"=== Plot-ready gt-vs-pred rows written: {plot_out} ===")
        print(f"records: {len(all_plot_rows)}")

    if args.comparison_out:
        payload = {
            "runs": summaries,
            "failed_rollout_dump": str(args.failed_rollout_dump or ""),
            "plot_rollout_dump": str(plot_out or ""),
            "image_root": str(args.image_root or ""),
            "ignored_error_tags": sorted(ignore_error_tags),
            "object_miss_threshold": args.object_miss_threshold,
        }
        args.comparison_out.parent.mkdir(parents=True, exist_ok=True)
        args.comparison_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"=== Comparison payload written: {args.comparison_out} ===")


if __name__ == "__main__":
    main()
