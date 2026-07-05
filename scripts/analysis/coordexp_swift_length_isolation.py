#!/usr/bin/env python3
"""Summarize CoordExp-swift physical packed-length isolation runs."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


METRIC_KEYS = ("loss/total", "loss/base_ce", "acc_top1", "acc_top5")
COUNT_KEYS = (
    "count/examples",
    "count/packs",
    "count/supervised_atoms",
    "count/eligible_segments",
    "count/skipped_segments",
)


def main() -> int:
    args = _parse_args()
    run_dirs = tuple(Path(run).resolve() for run in args.runs)
    summaries = [_summarize_run(run_dir) for run_dir in run_dirs]
    comparisons = _compare_pairs(summaries)
    payload = {
        "evidence_scope": "tiered smoke",
        "runs": summaries,
        "comparisons": comparisons,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(_render_markdown(payload), encoding="utf-8")
    print(f"wrote {args.output_json}")
    print(f"wrote {args.output_md}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories to summarize and compare.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Path for the machine-readable summary.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        required=True,
        help="Path for the Markdown research summary.",
    )
    return parser.parse_args()


def _summarize_run(run_dir: Path) -> dict[str, Any]:
    _require_dir(run_dir)
    manifest = _load_json(run_dir / "run_manifest.json")
    resolved = _load_json(run_dir / "configs" / "resolved.json")
    config = resolved.get("config", {})
    resolution = resolved.get("resolution", {})
    training_result = _load_json(run_dir / "receipts" / "runtime" / "training_result.json")
    runtime_setup = _load_json(run_dir / "receipts" / "runtime" / "runtime_setup.json")
    schedule = _load_json(run_dir / "resolved_step_schedule.json")
    pack_plan = _load_json(run_dir / "receipts" / "packing" / "pack_plan.json")
    qwen = _summarize_qwen_receipts(run_dir)
    steps = _summarize_steps(training_result)

    return {
        "run_dir": str(run_dir),
        "config_path": resolution.get("entry_config_path"),
        "run_name": config.get("run", {}).get("name"),
        "status": manifest.get("status"),
        "warnings": manifest.get("warnings", []),
        "global_max_length": config.get("packing", {}).get("global_max_length"),
        "max_steps": config.get("training", {}).get("max_steps"),
        "effective_batch_size": config.get("training", {}).get("effective_batch_size"),
        "train_sample_limit": _nested_get(config, ("data", "train", "sample_limit")),
        "eval_sample_limit": _nested_get(config, ("data", "eval", "sample_limit")),
        "runtime": {
            "backend": runtime_setup.get("backend"),
            "world_size": runtime_setup.get("world_size"),
            "runtime_batch": runtime_setup.get("runtime_batch"),
            "seed": runtime_setup.get("seed"),
        },
        "schedule": {
            "resolved_max_steps": schedule.get("resolved_max_steps"),
            "requested_pack_presentations": schedule.get("requested_pack_presentations"),
            "actual_pack_presentations": schedule.get("actual_pack_presentations"),
            "packs_per_epoch": schedule.get("packs_per_epoch"),
            "event_counts": {
                key: len(value) for key, value in schedule.get("events", {}).items()
            },
        },
        "packing": _summarize_pack_plan(pack_plan),
        "qwen_forward": qwen,
        "training": {
            "completed_steps": training_result.get("completed_steps"),
            "consumed_micro_steps": training_result.get("consumed_micro_steps"),
            "scheduled_event_counts": training_result.get("scheduled_event_counts"),
            "steps": steps,
        },
        "eval_forward": _summarize_metric_stream(
            run_dir / "metrics" / "eval.forward.jsonl"
        ),
        "boundary_pass": qwen["boundary_pass"],
        "finite_pass": _finite_pass(manifest, training_result, steps),
    }


def _summarize_pack_plan(pack_plan: dict[str, Any]) -> dict[str, Any]:
    previews = pack_plan.get("rank_local_micro_step_preview") or ()
    lengths: list[int] = []
    segment_counts: list[int] = []
    max_segment_lengths: list[int] = []
    for item in previews:
        pack = item.get("pack", {})
        segments = pack.get("segments") or ()
        lengths.append(int(pack.get("length", 0)))
        segment_counts.append(int(pack.get("segment_count", 0)))
        if segments:
            max_segment_lengths.append(max(int(segment.get("length", 0)) for segment in segments))
    return {
        "actual_pack_presentations": pack_plan.get("actual_pack_presentations"),
        "packs_per_epoch": pack_plan.get("packs_per_epoch"),
        "tail_fill_pack_count": pack_plan.get("tail_fill_pack_count"),
        "cache": pack_plan.get("cache"),
        "preview_count": len(previews),
        "preview_length_stats": _stats(lengths),
        "preview_segment_count_stats": _stats(segment_counts),
        "preview_max_segment_length_stats": _stats(max_segment_lengths),
    }


def _summarize_qwen_receipts(run_dir: Path) -> dict[str, Any]:
    receipt_paths = sorted((run_dir / "receipts" / "qwen").glob("forward_step_*.json"))
    receipts: list[dict[str, Any]] = []
    failures: list[str] = []
    proof_present_count = 0
    pack_lengths: list[int] = []
    max_segment_lengths: list[int] = []
    segment_counts: list[int] = []
    for receipt_path in receipt_paths:
        doc = _load_json(receipt_path)
        for index, receipt in enumerate(doc.get("qwen_forward_receipts") or ()):
            summary, receipt_failures = _check_qwen_receipt(receipt)
            summary["receipt_path"] = str(receipt_path)
            summary["receipt_index"] = index
            receipts.append(summary)
            if summary["proof_present"]:
                proof_present_count += 1
            failures.extend(
                f"{receipt_path.name}[{index}]: {failure}" for failure in receipt_failures
            )
            if summary["pack_length"] is not None:
                pack_lengths.append(summary["pack_length"])
            if summary["max_segment_length"] is not None:
                max_segment_lengths.append(summary["max_segment_length"])
            if summary["segment_count"] is not None:
                segment_counts.append(summary["segment_count"])
    if receipts and proof_present_count == 0:
        failures.append("no FA2 branch proof receipt was captured")
    return {
        "receipt_count": len(receipts),
        "proof_present_count": proof_present_count,
        "boundary_pass": bool(receipts) and not failures,
        "boundary_failures": failures,
        "pack_length_stats": _stats(pack_lengths),
        "max_segment_length_stats": _stats(max_segment_lengths),
        "segment_count_stats": _stats(segment_counts),
        "receipts": receipts,
    }


def _check_qwen_receipt(receipt: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    fa2 = receipt.get("fa2_varlen") or {}
    proof = fa2.get("proof") or {}
    proof_present = bool(proof)
    boundaries = fa2.get("segment_boundaries") or ()
    segment_lengths = fa2.get("segment_lengths") or ()
    pack_length = boundaries[-1] if boundaries else receipt.get("pack_length")
    max_segment_length = max(segment_lengths) if segment_lengths else None
    structural_checks = {
        "position_ids_4row": _first(receipt.get("position_ids_shape")) == 4,
        "row_meaning": receipt.get("position_row_meaning")
        == ["text", "temporal", "height", "width"],
        "cu_q_matches_boundaries": fa2.get("cu_seq_lens_q") == boundaries,
        "cu_k_matches_boundaries": fa2.get("cu_seq_lens_k") == boundaries,
        "max_q_matches_segment": fa2.get("max_length_q") == max_segment_length,
        "max_k_matches_segment": fa2.get("max_length_k") == max_segment_length,
        "attention_mask_null": fa2.get("attention_mask") is None,
        "multi_segment": len(boundaries) > 2,
        "physical_not_used_as_max": pack_length != fa2.get("max_length_q"),
    }
    proof_checks = {
        "proof_pass": proof.get("status") == "pass",
        "varlen_branch": proof.get("observed_branch") == "padding_free_varlen",
        "varlen_called": proof.get("flash_varlen_fn_called") is True,
        "ordinary_flash_not_called": proof.get("flash_fn_called") is False,
        "pad_not_called": proof.get("pad_fn_called") is False,
        "unpad_not_called": proof.get("unpad_fn_called") is False,
    }
    failures = [key for key, value in structural_checks.items() if not value]
    if proof_present:
        failures.extend(key for key, value in proof_checks.items() if not value)
    return (
        {
            "pack_index": receipt.get("pack_index"),
            "pack_length": pack_length,
            "segment_count": fa2.get("segment_count") or receipt.get("segment_count"),
            "max_segment_length": max_segment_length,
            "fa2_max_length_q": fa2.get("max_length_q"),
            "fa2_max_length_k": fa2.get("max_length_k"),
            "proof_present": proof_present,
            "structural_checks": structural_checks,
            "proof_checks": proof_checks if proof_present else {},
        },
        failures,
    )


def _summarize_steps(training_result: dict[str, Any]) -> dict[str, Any]:
    steps: dict[str, Any] = {}
    for step in training_result.get("step_results") or ():
        planned_step_id = str(step.get("planned_step_id"))
        metrics = step.get("loss_bundle", {}).get("metrics", {})
        steps[planned_step_id] = {
            "finite_status": step.get("finite_status"),
            "optimizer_update_status": step.get("optimizer_update_status"),
            "micro_step_count": step.get("micro_step_count"),
            "post_backward_decision": step.get("post_backward_decision", {}),
            "metrics": {key: metrics.get(key) for key in METRIC_KEYS},
            "counts": {key: metrics.get(key) for key in COUNT_KEYS},
        }
    return steps


def _summarize_metric_stream(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "steps": {}}
    steps: dict[str, dict[str, float]] = {}
    for record in _read_jsonl(path):
        step = str(record.get("planned_step_id"))
        name = record.get("name")
        if name in (*METRIC_KEYS, *COUNT_KEYS):
            steps.setdefault(step, {})[name] = record.get("value")
    return {"exists": True, "steps": steps}


def _compare_pairs(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_step: dict[int, dict[int, dict[str, Any]]] = {}
    for summary in summaries:
        by_step.setdefault(int(summary["max_steps"]), {})[
            int(summary["global_max_length"])
        ] = summary

    comparisons = []
    for max_steps, by_length in sorted(by_step.items()):
        if 12000 not in by_length or 6000 not in by_length:
            continue
        twelve = by_length[12000]
        six = by_length[6000]
        comparison = _compare_pair(twelve, six, max_steps=max_steps)
        comparisons.append(comparison)
    return comparisons


def _compare_pair(
    twelve: dict[str, Any], six: dict[str, Any], *, max_steps: int
) -> dict[str, Any]:
    step_id = "1"
    twelve_step = twelve["training"]["steps"].get(step_id, {})
    six_step = six["training"]["steps"].get(step_id, {})
    metric_deltas = {
        key: _delta(
            _nested_get(twelve_step, ("metrics", key)),
            _nested_get(six_step, ("metrics", key)),
        )
        for key in METRIC_KEYS
    }
    count_deltas = {
        key: _delta(
            _nested_get(twelve_step, ("counts", key)),
            _nested_get(six_step, ("counts", key)),
        )
        for key in COUNT_KEYS
    }
    counts_match = all(delta == 0 for delta in count_deltas.values())
    physical_length_material = _mean(
        twelve["qwen_forward"]["pack_length_stats"]
    ) > _mean(six["qwen_forward"]["pack_length_stats"]) * 1.1
    loss_ref = abs(_nested_get(twelve_step, ("metrics", "loss/total")) or 0.0)
    loss_tolerance = max(0.02, 0.005 * loss_ref)
    metric_pass = (
        _abs_or_inf(metric_deltas["loss/total"]) <= loss_tolerance
        and _abs_or_inf(metric_deltas["acc_top1"]) <= 0.01
        and _abs_or_inf(metric_deltas["acc_top5"]) <= 0.01
    )
    runtime_pass = twelve["finite_pass"] and six["finite_pass"]
    boundary_pass = twelve["boundary_pass"] and six["boundary_pass"]

    if not boundary_pass:
        verdict = "failed_boundary_gate"
    elif not runtime_pass:
        verdict = "failed_runtime_gate"
    elif not physical_length_material:
        verdict = "failed_to_materially_change_physical_length"
    elif not counts_match:
        verdict = "unresolved_count_confounded"
    elif metric_pass:
        verdict = "pure_physical_length_eliminated_smoke_scope"
    else:
        verdict = "physical_length_implicated_smoke_scope"

    return {
        "max_steps": max_steps,
        "twelve_k_run": twelve["run_dir"],
        "six_k_run": six["run_dir"],
        "boundary_pass": boundary_pass,
        "runtime_pass": runtime_pass,
        "physical_length_material": physical_length_material,
        "counts_match": counts_match,
        "metric_pass": metric_pass,
        "loss_tolerance": loss_tolerance,
        "metric_deltas_12k_minus_6k": metric_deltas,
        "count_deltas_12k_minus_6k": count_deltas,
        "verdict": verdict,
    }


def _finite_pass(
    manifest: dict[str, Any], training_result: dict[str, Any], steps: dict[str, Any]
) -> bool:
    if manifest.get("status") != "completed":
        return False
    if manifest.get("warnings"):
        return False
    if training_result.get("completed_steps") is None:
        return False
    for step in steps.values():
        if step.get("finite_status") != "finite":
            return False
        if step.get("optimizer_update_status") != "applied":
            return False
        decision = step.get("post_backward_decision") or {}
        if decision and decision.get("finite_status") != "finite":
            return False
    return True


def _render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# CoordExp-Swift Physical-Length Isolation Summary",
        "",
        f"Evidence scope: `{payload['evidence_scope']}`.",
        "",
        "## Runs",
        "",
        "| run | length | steps | status | boundary | finite | step1 loss | step1 top1 | step1 examples | qwen mean pack | qwen mean max segment |",
        "| --- | ---: | ---: | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in payload["runs"]:
        step1 = run["training"]["steps"].get("1", {})
        metrics = step1.get("metrics", {})
        counts = step1.get("counts", {})
        lines.append(
            "| {name} | {length} | {steps} | {status} | {boundary} | {finite} | {loss} | {top1} | {examples} | {pack} | {max_segment} |".format(
                name=run["run_name"],
                length=run["global_max_length"],
                steps=run["max_steps"],
                status=run["status"],
                boundary=run["boundary_pass"],
                finite=run["finite_pass"],
                loss=_fmt(metrics.get("loss/total")),
                top1=_fmt(metrics.get("acc_top1")),
                examples=_fmt(counts.get("count/examples")),
                pack=_fmt(_mean(run["qwen_forward"]["pack_length_stats"])),
                max_segment=_fmt(
                    _mean(run["qwen_forward"]["max_segment_length_stats"])
                ),
            )
        )

    lines.extend(["", "## Comparisons", ""])
    if not payload["comparisons"]:
        lines.append("No matched 12k/6k pairs were supplied.")
    for comparison in payload["comparisons"]:
        lines.extend(
            [
                f"### {comparison['max_steps']}-step pair",
                "",
                f"Verdict: `{comparison['verdict']}`.",
                "",
                f"- Boundary pass: `{comparison['boundary_pass']}`",
                f"- Runtime pass: `{comparison['runtime_pass']}`",
                f"- Physical length materially changed: `{comparison['physical_length_material']}`",
                f"- Counts match: `{comparison['counts_match']}`",
                f"- Metric pass: `{comparison['metric_pass']}`",
                "",
                "| field | 12k - 6k |",
                "| --- | ---: |",
            ]
        )
        for key, value in comparison["metric_deltas_12k_minus_6k"].items():
            lines.append(f"| {key} | {_fmt(value)} |")
        for key, value in comparison["count_deltas_12k_minus_6k"].items():
            lines.append(f"| {key} | {_fmt(value)} |")
        lines.append("")

    lines.extend(
        [
            "## Handles",
            "",
        ]
    )
    for run in payload["runs"]:
        lines.extend(
            [
                f"- `{run['run_name']}`",
                f"  - config: `{run.get('config_path')}`",
                f"  - run: `{run['run_dir']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `pure_physical_length_eliminated_smoke_scope` means the pair changed physical packed-row length, preserved boundary/runtime/count gates, and stayed inside the metric tolerances.",
            "- `unresolved_count_confounded` means physical row length changed, but examples or supervised atoms per compared step also changed, so the run cannot isolate pure length by itself.",
            "- This report is smoke evidence only; it is not a full validation or benchmark claim.",
            "",
            "## Recommended Next Action",
            "",
            "- Do not run the planned 2-step confirmation for this pair, because the 1-step gate did not isolate pure physical length.",
            "- If a cleaner isolation is needed, add a separate count-matched follow-up arm, for example by increasing the 6k pack presentations so examples and supervised atoms match before comparing precision-sensitive metrics.",
            "",
        ]
    )
    return "\n".join(lines)


def _require_dir(path: Path) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"run directory does not exist: {path}")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "median": None, "max": None}
    return {
        "count": len(values),
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _mean(stats: dict[str, Any]) -> float:
    value = stats.get("mean")
    return float(value) if value is not None else math.nan


def _nested_get(value: dict[str, Any] | None, path: tuple[str, ...]) -> Any:
    current: Any = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first(value: Any) -> Any:
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return None


def _delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def _abs_or_inf(value: Any) -> float:
    if value is None:
        return math.inf
    number = abs(float(value))
    return math.inf if math.isnan(number) else number


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "nan"
    return f"{number:.6g}"


if __name__ == "__main__":
    raise SystemExit(main())
