#!/usr/bin/env python
"""Compare CoordExp detection runs (rollout + confidence + eval artifacts).

This is an *offline* analysis harness that reads existing run directories produced
by `scripts/run_infer.py` + `scripts/postop_confidence.py` + `scripts/evaluate_detection.py`.

It focuses on questions like:
- Why does one checkpoint decode better even if teacher-forcing metrics look worse?
- Are gains coming from recall (more objects), termination behavior, or confidence calibration?

Inputs (per run_dir):
- gt_vs_pred.jsonl
- pred_token_trace.jsonl
- pred_confidence.jsonl
- eval/metrics.json
- eval/per_image.json
- eval/per_class.csv

Example:
  conda run -n ms python scripts/analysis/compare_detection_runs.py \
    --run_dir output/bench/softce_w1_1832_merged_coco_val_limit200_res768_temp0 \
    --run_dir output/bench/pure_ce_1932_merged_coco_val_limit200_res768_temp0 \
    --run_dir output/bench/softce_hardce_mixed_1344_merged_coco_val_limit200_res768_temp0_v2 \
    --out_md progress/temp0_compare_3runs.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


_COORD_RE = re.compile(r"^<\|coord_\d+\|>$")


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(obj, dict):
        return obj
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


def _quantile(vals: Sequence[float], q: float) -> float:
    if not vals:
        return float("nan")
    if q <= 0:
        return float(min(vals))
    if q >= 1:
        return float(max(vals))
    s = sorted(float(v) for v in vals)
    idx = int(round((len(s) - 1) * q))
    idx = max(0, min(idx, len(s) - 1))
    return float(s[idx])


def _fmt_float(x: Any, *, nd: int = 4) -> str:
    if x is None:
        return "NA"
    try:
        xf = float(x)
    except Exception:
        return "NA"
    if math.isnan(xf):
        return "NA"
    return f"{xf:.{nd}f}"


def _fmt_pct(num: int, den: int) -> str:
    if den <= 0:
        return "NA"
    return f"{(100.0 * float(num) / float(den)):.2f}%"


@dataclass
class DistSummary:
    count: int
    mean: float
    p50: float
    p90: float
    min: float
    max: float


def _summarize_dist(vals: Sequence[float]) -> DistSummary:
    if not vals:
        return DistSummary(count=0, mean=float("nan"), p50=float("nan"), p90=float("nan"), min=float("nan"), max=float("nan"))
    v = [float(x) for x in vals]
    return DistSummary(
        count=len(v),
        mean=float(mean(v)),
        p50=_quantile(v, 0.50),
        p90=_quantile(v, 0.90),
        min=float(min(v)),
        max=float(max(v)),
    )


def _first_idx(tokens: Sequence[str], needle: str) -> Optional[int]:
    for i, t in enumerate(tokens):
        if t == needle:
            return i
    return None


def _effective_token_window(tokens: Sequence[str]) -> Tuple[int, int, bool]:
    """Return (start=0, end_exclusive, has_im_end).

    We consider the effective generated content to end at the first <|im_end|> if present,
    else at the first <|endoftext|>, else at the full sequence.
    """

    im_end_idx = _first_idx(tokens, "<|im_end|>")
    if im_end_idx is not None:
        return 0, im_end_idx + 1, True
    eos_idx = _first_idx(tokens, "<|endoftext|>")
    if eos_idx is not None:
        return 0, eos_idx + 1, False
    return 0, len(tokens), False


def summarize_token_trace(trace_path: Path) -> Dict[str, Any]:
    if not trace_path.exists():
        return {"trace_rows": 0, "missing": True}

    token_lens: List[int] = []
    eff_token_lens: List[int] = []
    coord_frac: List[float] = []
    im_end_present = 0
    im_end_pos: List[int] = []
    eos_tail: List[int] = []
    bad_rows = 0
    rows = 0

    for row in _iter_jsonl(trace_path):
        rows += 1
        toks = row.get("generated_token_text")
        if not isinstance(toks, list):
            bad_rows += 1
            continue
        clean = [str(t) for t in toks]
        token_lens.append(len(clean))

        start, end, has_im_end = _effective_token_window(clean)
        if has_im_end:
            im_end_present += 1
            im_end_pos.append(end - 1)

        eff = clean[start:end]
        eff_token_lens.append(len(eff))
        if eff:
            coord = sum(1 for t in eff if _COORD_RE.match(t))
            coord_frac.append(float(coord) / float(len(eff)))
            # tail eos/pad after the effective window
            tail = clean[end:]
            eos_tail.append(sum(1 for t in tail if t == "<|endoftext|>"))

    out: Dict[str, Any] = {
        "trace_rows": rows,
        "bad_rows": bad_rows,
        "token_len": _summarize_dist(token_lens).__dict__,
        "effective_token_len": _summarize_dist(eff_token_lens).__dict__,
        "coord_token_fraction": _summarize_dist(coord_frac).__dict__,
        "im_end_present": im_end_present,
        "im_end_present_rate": float(im_end_present) / float(rows) if rows else 0.0,
        "im_end_pos": _summarize_dist(im_end_pos).__dict__,
        "eos_tail_count": _summarize_dist(eos_tail).__dict__,
    }
    return out


def summarize_gt_vs_pred(pred_path: Path) -> Dict[str, Any]:
    if not pred_path.exists():
        return {"samples": 0, "missing": True}

    pred_counts: List[int] = []
    gt_counts: List[int] = []
    empty_pred = 0
    samples = 0

    for row in _iter_jsonl(pred_path):
        samples += 1
        gt = row.get("gt") or []
        pred = row.get("pred") or []
        if not isinstance(gt, list):
            gt = []
        if not isinstance(pred, list):
            pred = []
        gt_counts.append(len(gt))
        pred_counts.append(len(pred))
        if len(pred) == 0:
            empty_pred += 1

    out: Dict[str, Any] = {
        "samples": samples,
        "empty_pred": empty_pred,
        "empty_pred_rate": float(empty_pred) / float(samples) if samples else 0.0,
        "gt_count": _summarize_dist(gt_counts).__dict__,
        "pred_count": _summarize_dist(pred_counts).__dict__,
    }
    return out


def summarize_pred_confidence(conf_path: Path) -> Dict[str, Any]:
    if not conf_path.exists():
        return {"objects": 0, "missing": True}

    fusion: List[float] = []
    geom: List[float] = []
    desc: List[float] = []
    objs = 0
    kept = 0

    for row in _iter_jsonl(conf_path):
        for obj in row.get("objects") or []:
            if not isinstance(obj, dict):
                continue
            objs += 1
            if obj.get("kept") is True:
                kept += 1
            if "score_fusion" in obj:
                fusion.append(float(obj["score_fusion"]))
            if "score_geom" in obj:
                geom.append(float(obj["score_geom"]))
            if "score_desc" in obj:
                desc.append(float(obj["score_desc"]))

    out: Dict[str, Any] = {
        "objects": objs,
        "kept": kept,
        "kept_rate": float(kept) / float(objs) if objs else 0.0,
        "score_fusion": _summarize_dist(fusion).__dict__,
        "score_geom": _summarize_dist(geom).__dict__,
        "score_desc": _summarize_dist(desc).__dict__,
    }
    return out


def summarize_match_score_separation(run_dir: Path) -> Dict[str, Any]:
    """Summarize confidence calibration by comparing matched vs unmatched predictions.

    Uses evaluator artifacts:
    - pred_confidence.jsonl: per-prediction scores (fusion/geom/desc)
    - eval/matches.jsonl: which predictions matched a GT at IoU=0.50 (loc-only)

    This is helpful for explaining mAP gaps: COCO mAP is ranking-based, so if a model
    assigns higher scores to true positives than false positives, AP improves even
    at similar recall.
    """

    conf_path = run_dir / "pred_confidence.jsonl"
    matches_path = run_dir / "eval" / "matches.jsonl"
    if not conf_path.exists() or not matches_path.exists():
        return {"missing": True}

    conf_by_img: Dict[int, Dict[int, Dict[str, float]]] = {}
    for row in _iter_jsonl(conf_path):
        idx = row.get("line_idx")
        objs = row.get("objects")
        if not isinstance(idx, int) or not isinstance(objs, list):
            continue

        by_idx: Dict[int, Dict[str, float]] = {}
        for obj in objs:
            if not isinstance(obj, dict):
                continue
            obj_idx = obj.get("object_idx")
            if not isinstance(obj_idx, int):
                continue

            by_idx[obj_idx] = {
                "fusion": float(obj.get("score_fusion", obj.get("score", 0.0)) or 0.0),
                "geom": float(obj.get("score_geom", 0.0) or 0.0),
                "desc": float(obj.get("score_desc", 0.0) or 0.0),
            }

        conf_by_img[idx] = by_idx

    matched_f: List[float] = []
    unmatched_f: List[float] = []
    matched_g: List[float] = []
    unmatched_g: List[float] = []
    matched_d: List[float] = []
    unmatched_d: List[float] = []

    match_rows = 0
    for row in _iter_jsonl(matches_path):
        match_rows += 1
        img = row.get("image_id")
        if not isinstance(img, int):
            continue
        by_idx = conf_by_img.get(img)
        if not isinstance(by_idx, dict):
            continue

        matched: set[int] = set()
        for m in row.get("matches") or []:
            if isinstance(m, dict) and isinstance(m.get("pred_idx"), int):
                matched.add(int(m["pred_idx"]))

        unmatched: set[int] = set()
        for oi in row.get("unmatched_pred_indices") or []:
            if isinstance(oi, int):
                unmatched.add(int(oi))

        ignored: set[int] = set()
        for oi in row.get("ignored_pred_indices") or []:
            if isinstance(oi, int):
                ignored.add(int(oi))

        for oi in matched:
            if oi in ignored:
                continue
            s = by_idx.get(oi)
            if not s:
                continue
            matched_f.append(float(s.get("fusion", 0.0)))
            matched_g.append(float(s.get("geom", 0.0)))
            matched_d.append(float(s.get("desc", 0.0)))

        for oi in unmatched:
            if oi in ignored:
                continue
            s = by_idx.get(oi)
            if not s:
                continue
            unmatched_f.append(float(s.get("fusion", 0.0)))
            unmatched_g.append(float(s.get("geom", 0.0)))
            unmatched_d.append(float(s.get("desc", 0.0)))

    def _gap(a: Sequence[float], b: Sequence[float]) -> float:
        if not a or not b:
            return float("nan")
        return float(mean(a) - mean(b))

    return {
        "match_rows": match_rows,
        "matched_fusion": _summarize_dist(matched_f).__dict__,
        "unmatched_fusion": _summarize_dist(unmatched_f).__dict__,
        "fusion_gap_mean": _gap(matched_f, unmatched_f),
        "matched_geom": _summarize_dist(matched_g).__dict__,
        "unmatched_geom": _summarize_dist(unmatched_g).__dict__,
        "geom_gap_mean": _gap(matched_g, unmatched_g),
        "matched_desc": _summarize_dist(matched_d).__dict__,
        "unmatched_desc": _summarize_dist(unmatched_d).__dict__,
        "desc_gap_mean": _gap(matched_d, unmatched_d),
    }


def read_eval_metrics(metrics_path: Path) -> Dict[str, Any]:
    raw = _read_json(metrics_path) or {}
    metrics = raw.get("metrics") if isinstance(raw, dict) else None
    if not isinstance(metrics, dict):
        metrics = {}

    # Common keys we care about.
    keys = [
        "bbox_AP",
        "bbox_AP50",
        "bbox_AP75",
        "bbox_AR1",
        "bbox_AR10",
        "bbox_AR100",
        "f1ish@0.50_f1_full_micro",
        "f1ish@0.50_precision_full_micro",
        "f1ish@0.50_recall_full_micro",
        "f1ish@0.50_tp_full",
        "f1ish@0.50_fp_full",
        "f1ish@0.50_fn_full",
        "f1ish@0.50_pred_total",
        "f1ish@0.50_pred_eval",
        "f1ish@0.50_pred_ignored",
    ]

    out = {"metrics": {k: metrics.get(k) for k in keys if k in metrics}, "metrics_raw": metrics}
    # Also surface num_samples if present.
    counters = raw.get("counters") if isinstance(raw, dict) else None
    if isinstance(counters, dict):
        out["counters"] = counters
    return out


def read_per_class(per_class_csv: Path) -> Dict[str, Dict[str, Any]]:
    if not per_class_csv.exists():
        return {}

    rows: Dict[str, Dict[str, Any]] = {}
    with per_class_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not isinstance(r, dict):
                continue

            # Normalize column names so we can support different evaluator outputs:
            # - (category, AP)
            # - (class_name, ap)
            norm: Dict[str, Any] = {}
            for k, v in r.items():
                if k is None:
                    continue
                norm[str(k).strip().lower()] = v

            name = (
                str(
                    norm.get("class_name")
                    or norm.get("category")
                    or norm.get("class")
                    or norm.get("name")
                    or ""
                )
                .strip()
            )
            if not name:
                continue

            rows[name] = norm
    return rows


def read_per_image(per_image_json: Path) -> Dict[str, Dict[str, Any]]:
    if not per_image_json.exists():
        return {}

    try:
        obj = json.loads(per_image_json.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(obj, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for row in obj:
        if not isinstance(row, dict):
            continue
        fn = str(row.get("file_name") or "").strip()
        if not fn:
            continue
        out[fn] = row
    return out


def _get_f1ish_counts(per_image_row: Dict[str, Any], iou_thr: str = "0.50") -> Tuple[int, int, int]:
    f1ish = per_image_row.get("f1ish")
    if not isinstance(f1ish, dict):
        return 0, 0, 0
    bucket = f1ish.get(iou_thr)
    if not isinstance(bucket, dict):
        return 0, 0, 0
    tp = int(bucket.get("tp_full") or 0)
    fp = int(bucket.get("fp_full") or 0)
    fn = int(bucket.get("fn_full") or 0)
    return tp, fp, fn


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _diff_topk(
    a: Dict[str, Any],
    b: Dict[str, Any],
    *,
    key: str,
    k: int,
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """Return top (improved, degraded) keys by (b - a) for numeric dict values."""

    deltas: List[Tuple[str, float]] = []
    for name in sorted(set(a.keys()) | set(b.keys())):
        va = _as_float(a.get(name))
        vb = _as_float(b.get(name))
        if va is None or vb is None:
            continue
        deltas.append((name, float(vb - va)))

    improved = sorted(deltas, key=lambda x: x[1], reverse=True)[:k]
    degraded = sorted(deltas, key=lambda x: x[1])[:k]
    return improved, degraded


def _md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    h = "| " + " | ".join(headers) + " |"
    sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    out = [h, sep]
    for r in rows:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare detection runs from existing rollout artifacts.")
    ap.add_argument("--run_dir", type=Path, action="append", default=[], help="Run dir under output/bench/... (repeatable)")
    ap.add_argument("--baseline", type=str, default=None, help="Optional baseline run name (defaults to first run_dir)")
    ap.add_argument("--out_md", type=Path, default=None, help="Optional markdown path to write")
    ap.add_argument("--out_json", type=Path, default=None, help="Optional JSON path to write")
    ap.add_argument("--topk", type=int, default=12, help="Top-K per-class/per-image diffs to show")
    args = ap.parse_args()

    if not args.run_dir:
        raise SystemExit("--run_dir is required (repeatable)")

    runs: List[Dict[str, Any]] = []
    for run_dir in args.run_dir:
        run_dir = run_dir.resolve()
        name = run_dir.name

        gt_vs_pred = run_dir / "gt_vs_pred.jsonl"
        pred_trace = run_dir / "pred_token_trace.jsonl"
        pred_conf = run_dir / "pred_confidence.jsonl"
        conf_summary = _read_json(run_dir / "confidence_postop_summary.json")
        eval_metrics = read_eval_metrics(run_dir / "eval" / "metrics.json")
        per_image = read_per_image(run_dir / "eval" / "per_image.json")
        per_class = read_per_class(run_dir / "eval" / "per_class.csv")

        runs.append(
            {
                "name": name,
                "run_dir": str(run_dir),
                "gt_vs_pred": summarize_gt_vs_pred(gt_vs_pred),
                "token_trace": summarize_token_trace(pred_trace),
                "pred_confidence": summarize_pred_confidence(pred_conf),
                "score_separation": summarize_match_score_separation(run_dir),
                "confidence_postop_summary": conf_summary,
                "eval": eval_metrics,
                "per_image": per_image,
                "per_class": per_class,
            }
        )

    # Resolve baseline index.
    baseline_name = args.baseline or runs[0]["name"]
    baseline_idx = 0
    for i, r in enumerate(runs):
        if r["name"] == baseline_name:
            baseline_idx = i
            break

    payload = {"baseline": baseline_name, "runs": runs}

    # Build markdown.
    md: List[str] = []
    md.append(f"# Detection Run Comparison\n")
    md.append(f"Baseline: `{baseline_name}`\n")

    # Metrics table.
    metric_rows: List[List[str]] = []
    for r in runs:
        m = (r.get("eval") or {}).get("metrics") or {}
        gt_s = (r.get("gt_vs_pred") or {})
        pred_s = gt_s.get("pred_count") or {}
        metric_rows.append(
            [
                r["name"],
                str(gt_s.get("samples", "NA")),
                _fmt_float(m.get("bbox_AP")),
                _fmt_float(m.get("bbox_AR1")),
                _fmt_float(m.get("f1ish@0.50_f1_full_micro")),
                str(m.get("f1ish@0.50_pred_total", "NA")),
                _fmt_float((pred_s or {}).get("mean")),
                _fmt_float((pred_s or {}).get("p90"), nd=2),
                _fmt_pct(int(gt_s.get("empty_pred") or 0), int(gt_s.get("samples") or 0)),
            ]
        )

    md.append("## Summary Metrics\n")
    md.append(
        _md_table(
            [
                "run",
                "samples",
                "mAP",
                "AR1",
                "F1@0.50(full)",
                "pred_total",
                "pred_mean",
                "pred_p90",
                "empty_pred",
            ],
            metric_rows,
        )
    )
    md.append("")

    # Token trace / confidence table.
    diag_rows: List[List[str]] = []
    for r in runs:
        t = r.get("token_trace") or {}
        tc = (t.get("effective_token_len") or {})
        frac = (t.get("coord_token_fraction") or {})
        conf = r.get("pred_confidence") or {}
        geom = (conf.get("score_geom") or {})
        desc = (conf.get("score_desc") or {})
        fusion = (conf.get("score_fusion") or {})
        sep = r.get("score_separation") or {}
        diag_rows.append(
            [
                r["name"],
                str(t.get("trace_rows", "NA")),
                _fmt_float(tc.get("mean"), nd=1),
                str(int(tc.get("p90") or 0)) if tc.get("p90") == tc.get("p90") else "NA",
                _fmt_float(frac.get("mean")),
                _fmt_float(fusion.get("mean")),
                _fmt_float(geom.get("mean")),
                _fmt_float(desc.get("mean")),
                _fmt_float(sep.get("fusion_gap_mean")),
                _fmt_float(sep.get("desc_gap_mean")),
            ]
        )

    md.append("## Decode And Confidence Diagnostics\n")
    md.append(
        _md_table(
            [
                "run",
                "trace_rows",
                "eff_tok_mean",
                "eff_tok_p90",
                "coord_frac_mean",
                "score_fusion_mean",
                "score_geom_mean",
                "score_desc_mean",
                "tp_fp_gap_fusion",
                "tp_fp_gap_desc",
            ],
            diag_rows,
        )
    )
    md.append("")

    # Pairwise diffs vs baseline.
    base = runs[baseline_idx]
    base_per_image = base.get("per_image") or {}
    base_per_class = base.get("per_class") or {}

    md.append("## Diffs Vs Baseline\n")

    for r in runs:
        if r["name"] == baseline_name:
            continue
        md.append(f"### `{r['name']}` vs `{baseline_name}`\n")

        # Per-class AP deltas.
        other_per_class = r.get("per_class") or {}
        base_ap = {k: (base_per_class.get(k) or {}).get("ap") for k in base_per_class.keys()}
        other_ap = {k: (other_per_class.get(k) or {}).get("ap") for k in other_per_class.keys()}
        improved, degraded = _diff_topk(base_ap, other_ap, key="ap", k=args.topk)

        md.append("Per-class AP (delta = other - base), top improved:\n")
        md.append(_md_table(["class", "delta_ap"], [[c, _fmt_float(d)] for c, d in improved]))
        md.append("")
        md.append("Per-class AP (delta = other - base), top degraded:\n")
        md.append(_md_table(["class", "delta_ap"], [[c, _fmt_float(d)] for c, d in degraded]))
        md.append("")

        # Per-image F1-ish deltas at IoU 0.50.
        other_per_image = r.get("per_image") or {}
        image_deltas: List[Tuple[str, int]] = []
        for fn in sorted(set(base_per_image.keys()) & set(other_per_image.keys())):
            btp, bfp, bfn = _get_f1ish_counts(base_per_image[fn])
            otp, ofp, ofn = _get_f1ish_counts(other_per_image[fn])
            # Use delta TP as a simple proxy for "recall" improvements.
            image_deltas.append((fn, int(otp - btp)))

        improved_imgs = sorted(image_deltas, key=lambda x: x[1], reverse=True)[: args.topk]
        degraded_imgs = sorted(image_deltas, key=lambda x: x[1])[: args.topk]

        md.append("Per-image delta TP_full@0.50 (other - base), top improved:\n")
        md.append(_md_table(["file_name", "delta_tp"], [[fn, str(dt)] for fn, dt in improved_imgs]))
        md.append("")
        md.append("Per-image delta TP_full@0.50 (other - base), top degraded:\n")
        md.append(_md_table(["file_name", "delta_tp"], [[fn, str(dt)] for fn, dt in degraded_imgs]))
        md.append("")

    md_text = "\n".join(md).rstrip() + "\n"

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(md_text, encoding="utf-8")
        print(f"Wrote: {args.out_md}")
    else:
        print(md_text)

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote: {args.out_json}")


if __name__ == "__main__":
    main()
