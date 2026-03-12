"""Oracle-K repeated-sampling evaluation for detection artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from .detection import (
    EvalCounters,
    EvalOptions,
    Sample,
    _prepare_all,
    _select_primary_f1ish_iou_thr,
    evaluate_f1ish,
    load_jsonl,
)


@dataclass(frozen=True)
class OracleKRunSpec:
    label: str
    pred_jsonl: Path | None = None
    pred_token_trace_jsonl: Path | None = None
    resolved_config_json: Path | None = None
    pipeline_config: Path | None = None
    overrides: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OracleKResolvedRun:
    label: str
    pred_jsonl: Path
    pred_token_trace_jsonl: Path | None
    resolved_config_json: Path | None
    materialized: bool
    run_dir: Path | None


@dataclass(frozen=True)
class OracleKConfig:
    out_dir: Path
    eval_options: EvalOptions
    baseline_run: OracleKRunSpec
    oracle_runs: Tuple[OracleKRunSpec, ...]


@dataclass
class OracleKPreparedRun:
    resolved: OracleKResolvedRun
    gt_samples: List[Sample]
    pred_samples: List[Tuple[int, List[Dict[str, Any]]]]
    counters: Dict[str, int]
    per_image: List[Dict[str, Any]]
    metrics: Dict[str, float] = field(default_factory=dict)
    matches_by_thr: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "YAML config requires PyYAML (import yaml). Install it in the ms env."
        ) from exc

    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("Oracle-K config must be a YAML mapping")
    return data


def _slugify_label(label: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", label.strip())
    slug = slug.strip("._-")
    return slug or "run"


def _ensure_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _optional_path(value: Any, *, field_name: str) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text)


def _require_path(value: Any, *, field_name: str) -> Path:
    path = _optional_path(value, field_name=field_name)
    if path is None:
        raise ValueError(f"{field_name} must be a non-empty path")
    return path


def _require_label(value: Any, *, field_name: str) -> str:
    label = str(value or "").strip()
    if not label:
        raise ValueError(f"{field_name} must be a non-empty string")
    return label


def _parse_run_spec(cfg: Mapping[str, Any], *, field_name: str) -> OracleKRunSpec:
    label = _require_label(cfg.get("label"), field_name=f"{field_name}.label")
    pred_jsonl = _optional_path(cfg.get("pred_jsonl"), field_name=f"{field_name}.pred_jsonl")
    pipeline_config = _optional_path(
        cfg.get("pipeline_config"), field_name=f"{field_name}.pipeline_config"
    )
    if (pred_jsonl is None) == (pipeline_config is None):
        raise ValueError(
            f"{field_name} must set exactly one of pred_jsonl or pipeline_config"
        )
    overrides_raw = cfg.get("overrides") or {}
    if not isinstance(overrides_raw, Mapping):
        raise ValueError(f"{field_name}.overrides must be a mapping when provided")
    return OracleKRunSpec(
        label=label,
        pred_jsonl=pred_jsonl,
        pred_token_trace_jsonl=_optional_path(
            cfg.get("pred_token_trace_jsonl"),
            field_name=f"{field_name}.pred_token_trace_jsonl",
        ),
        resolved_config_json=_optional_path(
            cfg.get("resolved_config_json"),
            field_name=f"{field_name}.resolved_config_json",
        ),
        pipeline_config=pipeline_config,
        overrides=dict(overrides_raw),
    )


def load_oracle_k_config(config_path: Path) -> OracleKConfig:
    cfg = _load_yaml(config_path)
    out_dir = Path(str(cfg.get("out_dir") or "oracle_k_eval"))

    baseline_cfg = _ensure_mapping(cfg.get("baseline_run"), field_name="baseline_run")
    oracle_cfgs = cfg.get("oracle_runs")
    if not isinstance(oracle_cfgs, list) or not oracle_cfgs:
        raise ValueError("oracle_runs must be a non-empty list")

    baseline_run = _parse_run_spec(baseline_cfg, field_name="baseline_run")
    oracle_runs = tuple(
        _parse_run_spec(
            _ensure_mapping(run_cfg, field_name=f"oracle_runs[{idx}]"),
            field_name=f"oracle_runs[{idx}]",
        )
        for idx, run_cfg in enumerate(oracle_cfgs)
    )

    labels = [baseline_run.label, *[run.label for run in oracle_runs]]
    if len(labels) != len(set(labels)):
        raise ValueError("Oracle-K run labels must be unique across baseline_run and oracle_runs")

    options = EvalOptions(
        metrics="f1ish",
        strict_parse=bool(cfg.get("strict_parse", False)),
        use_segm=False,
        f1ish_iou_thrs=[
            float(x) for x in (cfg.get("f1ish_iou_thrs", [0.3, 0.5]) or [])
        ],
        f1ish_pred_scope=str(cfg.get("f1ish_pred_scope", "annotated")),
        output_dir=out_dir,
        overlay=False,
        num_workers=int(cfg.get("num_workers", 0)),
        semantic_model=str(
            cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
        ),
        semantic_threshold=float(cfg.get("semantic_threshold", 0.6)),
        semantic_device=str(cfg.get("semantic_device", "auto")),
        semantic_batch_size=int(cfg.get("semantic_batch_size", 64)),
    )
    return OracleKConfig(
        out_dir=out_dir,
        eval_options=options,
        baseline_run=baseline_run,
        oracle_runs=oracle_runs,
    )


def _path_or_none(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path)


def _require_existing_file(path: Path, *, field_name: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"{field_name} does not exist: {path}")
    return path


def _materialize_run(
    spec: OracleKRunSpec,
    *,
    out_dir: Path,
    role: str,
    ordinal: int,
) -> OracleKResolvedRun:
    from src.infer.pipeline import run_pipeline

    pipeline_config = _require_path(
        spec.pipeline_config, field_name=f"{role}.pipeline_config"
    )
    _require_existing_file(pipeline_config, field_name=f"{role}.pipeline_config")

    label_slug = _slugify_label(spec.label)
    default_run_name = f"{role}_{ordinal:02d}_{label_slug}"
    overrides: Dict[str, Any] = dict(spec.overrides)
    overrides.setdefault("run.output_dir", str(out_dir / "materialized"))
    overrides.setdefault("run.name", default_run_name)
    overrides["stages.infer"] = True
    overrides["stages.eval"] = False
    overrides["stages.vis"] = False

    artifacts = run_pipeline(config_path=pipeline_config, overrides=overrides)
    pred_jsonl = _require_existing_file(
        artifacts.gt_vs_pred_jsonl,
        field_name=f"{role}.materialized.pred_jsonl",
    )
    trace_path = spec.pred_token_trace_jsonl or artifacts.pred_token_trace_jsonl
    resolved_config = spec.resolved_config_json or (artifacts.run_dir / "resolved_config.json")
    return OracleKResolvedRun(
        label=spec.label,
        pred_jsonl=pred_jsonl,
        pred_token_trace_jsonl=trace_path if trace_path.is_file() else None,
        resolved_config_json=resolved_config if resolved_config.is_file() else None,
        materialized=True,
        run_dir=artifacts.run_dir,
    )


def _resolve_run(
    spec: OracleKRunSpec,
    *,
    out_dir: Path,
    role: str,
    ordinal: int,
) -> OracleKResolvedRun:
    if spec.pred_jsonl is not None:
        pred_jsonl = _require_existing_file(
            spec.pred_jsonl,
            field_name=f"{role}.pred_jsonl",
        )
        trace_path = spec.pred_token_trace_jsonl
        resolved_config = spec.resolved_config_json
        if trace_path is not None:
            _require_existing_file(
                trace_path,
                field_name=f"{role}.pred_token_trace_jsonl",
            )
        if resolved_config is not None:
            _require_existing_file(
                resolved_config,
                field_name=f"{role}.resolved_config_json",
            )
        return OracleKResolvedRun(
            label=spec.label,
            pred_jsonl=pred_jsonl,
            pred_token_trace_jsonl=trace_path,
            resolved_config_json=resolved_config,
            materialized=False,
            run_dir=pred_jsonl.parent,
        )
    return _materialize_run(spec, out_dir=out_dir, role=role, ordinal=ordinal)


def _run_meta(run: OracleKResolvedRun, *, eval_dir: Path | None = None) -> Dict[str, Any]:
    return {
        "label": run.label,
        "pred_jsonl": str(run.pred_jsonl),
        "pred_token_trace_jsonl": _path_or_none(run.pred_token_trace_jsonl),
        "resolved_config_json": _path_or_none(run.resolved_config_json),
        "materialized": bool(run.materialized),
        "run_dir": _path_or_none(run.run_dir),
        "eval_dir": _path_or_none(eval_dir),
    }


def _canonical_gt_object(obj: Mapping[str, Any]) -> Dict[str, Any]:
    points_in = obj.get("points") or []
    if points_in and isinstance(points_in[0], (list, tuple)):
        points_out: List[Any] = [
            [float(point[0]), float(point[1])] for point in points_in
        ]
    else:
        points_out = [float(point) for point in points_in]
    return {
        "type": str(obj.get("type", "")),
        "desc": str(obj.get("desc", "")),
        "bbox": [float(x) for x in (obj.get("bbox") or [])],
        "points": points_out,
    }


def _validate_alignment(
    baseline_gt: Sequence[Sample],
    run_gt: Sequence[Sample],
    *,
    run_label: str,
) -> None:
    if len(baseline_gt) != len(run_gt):
        raise ValueError(
            f"Alignment failure for run '{run_label}': "
            f"record count mismatch baseline={len(baseline_gt)} run={len(run_gt)}"
        )

    for record_idx, (base_sample, run_sample) in enumerate(zip(baseline_gt, run_gt)):
        base_record_idx = int(base_sample.image_id)
        if int(base_sample.width) != int(run_sample.width) or int(base_sample.height) != int(
            run_sample.height
        ):
            raise ValueError(
                f"Alignment failure for run '{run_label}' at record_idx={base_record_idx}: "
                f"size mismatch baseline=({base_sample.width}, {base_sample.height}) "
                f"run=({run_sample.width}, {run_sample.height})"
            )
        if str(base_sample.file_name) != str(run_sample.file_name):
            raise ValueError(
                f"Alignment failure for run '{run_label}' at record_idx={base_record_idx}: "
                "required file_name mismatch for visualization provenance "
                f"baseline={base_sample.file_name!r} run={run_sample.file_name!r}"
            )
        base_objects = [_canonical_gt_object(obj) for obj in base_sample.objects]
        run_objects = [_canonical_gt_object(obj) for obj in run_sample.objects]
        if base_objects != run_objects:
            raise ValueError(
                f"Alignment failure for run '{run_label}' at record_idx={base_record_idx}: "
                "GT object content mismatch"
            )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _prepare_run(
    resolved: OracleKResolvedRun,
    *,
    base_options: EvalOptions,
) -> OracleKPreparedRun:
    counters = EvalCounters()
    pred_records = load_jsonl(
        resolved.pred_jsonl,
        counters,
        strict=base_options.strict_parse,
    )
    gt_samples, pred_samples, _, _, _, _, per_image = _prepare_all(
        pred_records,
        base_options,
        counters,
        prepare_coco=False,
    )
    return OracleKPreparedRun(
        resolved=resolved,
        gt_samples=gt_samples,
        pred_samples=pred_samples,
        counters=counters.to_dict(),
        per_image=per_image,
    )


def _evaluate_prepared_run(
    prepared: OracleKPreparedRun,
    *,
    base_options: EvalOptions,
    eval_dir: Path,
) -> None:
    options = replace(base_options, output_dir=eval_dir)
    f1ish_summary = evaluate_f1ish(
        prepared.gt_samples,
        prepared.pred_samples,
        prepared.per_image,
        options=options,
    )
    eval_dir.mkdir(parents=True, exist_ok=True)
    metrics_payload = {
        "metrics": f1ish_summary.get("metrics", {}),
        "counters": prepared.counters,
    }
    (eval_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (eval_dir / "per_image.json").write_text(
        json.dumps(prepared.per_image, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prepared.metrics = dict(f1ish_summary.get("metrics", {}))
    prepared.matches_by_thr = {
        str(thr_key): list(rows)
        for thr_key, rows in (f1ish_summary.get("matches_by_thr", {}) or {}).items()
    }


def _build_match_index(
    rows: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[Tuple[int, int], Dict[str, Any]], Dict[Tuple[int, int], Dict[str, Any]]]:
    loc_by_gt: Dict[Tuple[int, int], Dict[str, Any]] = {}
    full_by_gt: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for row in rows:
        record_idx = int(row.get("image_id", 0))
        for match in row.get("matches") or []:
            gt_idx = int(match.get("gt_idx", -1))
            if gt_idx < 0:
                continue
            key = (record_idx, gt_idx)
            evidence = {
                "matched_pred_idx": match.get("pred_idx"),
                "matched_pred_desc": match.get("pred_desc"),
                "matched_pred_bbox": match.get("pred_bbox"),
                "iou": match.get("iou"),
                "sem_sim": match.get("sem_sim"),
                "sem_ok": bool(match.get("sem_ok", False)),
            }
            loc_by_gt[key] = evidence
            if evidence["sem_ok"]:
                full_by_gt[key] = evidence
    return loc_by_gt, full_by_gt


def _compute_recall(tp: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return float(tp) / float(total)


def _run_eval_dir(out_dir: Path, *, role: str, ordinal: int, label: str) -> Path:
    label_slug = _slugify_label(label)
    return out_dir / "runs" / f"{role}_{ordinal:02d}_{label_slug}"


def _nullable_fraction(count: int | None, total: int) -> float | None:
    if count is None:
        return None
    if total <= 0:
        return 0.0
    return float(count) / float(total)


def _sum_defined(values: Iterable[int | None]) -> int:
    return sum(int(value) for value in values if value is not None)


def evaluate_oracle_k(config: OracleKConfig) -> Dict[str, Any]:
    out_dir = config.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_resolved = _resolve_run(
        config.baseline_run,
        out_dir=out_dir,
        role="baseline",
        ordinal=0,
    )
    oracle_resolved = [
        _resolve_run(run, out_dir=out_dir, role="oracle", ordinal=idx + 1)
        for idx, run in enumerate(config.oracle_runs)
    ]

    baseline_eval_dir = _run_eval_dir(
        out_dir,
        role="baseline",
        ordinal=0,
        label=baseline_resolved.label,
    )
    baseline_prepared = _prepare_run(
        baseline_resolved,
        base_options=config.eval_options,
    )

    oracle_prepared: List[OracleKPreparedRun] = []
    for idx, resolved in enumerate(oracle_resolved, start=1):
        prepared = _prepare_run(
            resolved,
            base_options=config.eval_options,
        )
        _validate_alignment(
            baseline_prepared.gt_samples,
            prepared.gt_samples,
            run_label=resolved.label,
        )
        oracle_prepared.append(prepared)

    _evaluate_prepared_run(
        baseline_prepared,
        base_options=config.eval_options,
        eval_dir=baseline_eval_dir,
    )
    for idx, prepared in enumerate(oracle_prepared, start=1):
        _evaluate_prepared_run(
            prepared,
            base_options=config.eval_options,
            eval_dir=_run_eval_dir(
                out_dir,
                role="oracle",
                ordinal=idx,
                label=prepared.resolved.label,
            ),
        )

    iou_thrs = sorted({float(x) for x in (config.eval_options.f1ish_iou_thrs or [0.3, 0.5])})
    primary_thr = _select_primary_f1ish_iou_thr(iou_thrs)
    primary_key = f"{primary_thr:.2f}"

    sample_by_record = {
        int(sample.image_id): sample for sample in baseline_prepared.gt_samples
    }
    all_gt_keys = {
        (int(sample.image_id), gt_idx)
        for sample in baseline_prepared.gt_samples
        for gt_idx, _ in enumerate(sample.objects)
    }
    total_gt_count = len(all_gt_keys)

    baseline_indexes: Dict[str, Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]] = {}
    oracle_indexes: Dict[str, List[Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]]] = {}
    thresholds_summary: Dict[str, Dict[str, Any]] = {}

    for thr in iou_thrs:
        thr_key = f"{thr:.2f}"
        baseline_loc, baseline_full = _build_match_index(
            baseline_prepared.matches_by_thr.get(thr_key, [])
        )
        baseline_indexes[thr_key] = {"loc": baseline_loc, "full": baseline_full}

        run_indexes: List[Dict[str, Dict[Tuple[int, int], Dict[str, Any]]]] = []
        for prepared in oracle_prepared:
            loc_map, full_map = _build_match_index(prepared.matches_by_thr.get(thr_key, []))
            run_indexes.append({"loc": loc_map, "full": full_map})
        oracle_indexes[thr_key] = run_indexes

        oracle_loc_union = {
            key for run_index in run_indexes for key in run_index["loc"].keys()
        }
        oracle_full_union = {
            key for run_index in run_indexes for key in run_index["full"].keys()
        }

        baseline_tp_loc = len(baseline_loc)
        baseline_tp_full = len(baseline_full)
        oracle_tp_loc = len(oracle_loc_union)
        oracle_tp_full = len(oracle_full_union)
        thresholds_summary[thr_key] = {
            "baseline": {
                "tp_loc": int(baseline_tp_loc),
                "fn_loc": int(total_gt_count - baseline_tp_loc),
                "recall_loc": _compute_recall(baseline_tp_loc, total_gt_count),
                "tp_full": int(baseline_tp_full),
                "fn_full": int(total_gt_count - baseline_tp_full),
                "recall_full": _compute_recall(baseline_tp_full, total_gt_count),
            },
            "oracle_k": {
                "tp_loc": int(oracle_tp_loc),
                "fn_loc": int(total_gt_count - oracle_tp_loc),
                "recall_loc": _compute_recall(oracle_tp_loc, total_gt_count),
                "tp_full": int(oracle_tp_full),
                "fn_full": int(total_gt_count - oracle_tp_full),
                "recall_full": _compute_recall(oracle_tp_full, total_gt_count),
            },
        }

    baseline_primary = baseline_indexes[primary_key]
    baseline_fn_loc_keys = sorted(all_gt_keys - set(baseline_primary["loc"].keys()))
    baseline_fn_full_keys = sorted(all_gt_keys - set(baseline_primary["full"].keys()))
    fn_keys = sorted(set(baseline_fn_loc_keys) | set(baseline_fn_full_keys))

    fn_rows: List[Dict[str, Any]] = []
    for record_idx, gt_idx in fn_keys:
        sample = sample_by_record[record_idx]
        gt_object = sample.objects[gt_idx]
        baseline_loc_hit = (record_idx, gt_idx) in baseline_primary["loc"]
        baseline_full_hit = (record_idx, gt_idx) in baseline_primary["full"]

        per_run_rows: List[Dict[str, Any]] = []
        recovered_labels_loc: List[str] = []
        recovered_labels_full: List[str] = []

        for prepared, run_index in zip(oracle_prepared, oracle_indexes[primary_key]):
            loc_evidence = run_index["loc"].get((record_idx, gt_idx))
            full_evidence = run_index["full"].get((record_idx, gt_idx))
            loc_hit = loc_evidence is not None
            full_hit = full_evidence is not None
            if loc_hit:
                recovered_labels_loc.append(prepared.resolved.label)
            if full_hit:
                recovered_labels_full.append(prepared.resolved.label)
            evidence = loc_evidence or full_evidence or {}
            per_run_rows.append(
                {
                    "label": prepared.resolved.label,
                    "loc_hit": bool(loc_hit),
                    "full_hit": bool(full_hit),
                    "matched_pred_idx": evidence.get("matched_pred_idx"),
                    "matched_pred_desc": evidence.get("matched_pred_desc"),
                    "matched_pred_bbox": evidence.get("matched_pred_bbox"),
                    "iou": evidence.get("iou"),
                    "sem_sim": evidence.get("sem_sim"),
                    "sem_ok": evidence.get("sem_ok"),
                    "pred_jsonl": str(prepared.resolved.pred_jsonl),
                    "pred_token_trace_jsonl": _path_or_none(
                        prepared.resolved.pred_token_trace_jsonl
                    ),
                    "resolved_config_json": _path_or_none(
                        prepared.resolved.resolved_config_json
                    ),
                }
            )

        baseline_loc_evidence = baseline_primary["loc"].get((record_idx, gt_idx))
        baseline_full_evidence = baseline_primary["full"].get((record_idx, gt_idx))
        baseline_evidence = baseline_loc_evidence or baseline_full_evidence or {}

        baseline_fn_loc = not baseline_loc_hit
        baseline_fn_full = not baseline_full_hit
        recover_count_loc = len(recovered_labels_loc) if baseline_fn_loc else None
        recover_count_full = len(recovered_labels_full) if baseline_fn_full else None
        ever_recovered_loc = bool(recovered_labels_loc) if baseline_fn_loc else None
        ever_recovered_full = bool(recovered_labels_full) if baseline_fn_full else None

        fn_rows.append(
            {
                "record_idx": int(record_idx),
                "gt_idx": int(gt_idx),
                "image_id": int(record_idx),
                "file_name": sample.file_name,
                "gt_desc": gt_object.get("desc"),
                "gt_bbox": gt_object.get("bbox"),
                "baseline_fn_loc": bool(baseline_fn_loc),
                "baseline_fn_full": bool(baseline_fn_full),
                "baseline_loc_hit": bool(baseline_loc_hit),
                "baseline_full_hit": bool(baseline_full_hit),
                "baseline_matched_pred_idx": baseline_evidence.get("matched_pred_idx"),
                "baseline_matched_pred_desc": baseline_evidence.get("matched_pred_desc"),
                "baseline_matched_pred_bbox": baseline_evidence.get("matched_pred_bbox"),
                "baseline_iou": baseline_evidence.get("iou"),
                "baseline_sem_sim": baseline_evidence.get("sem_sim"),
                "baseline_sem_ok": baseline_evidence.get("sem_ok"),
                "ever_recovered_loc": ever_recovered_loc,
                "ever_recovered_full": ever_recovered_full,
                "systematic_loc": (not ever_recovered_loc) if baseline_fn_loc else None,
                "systematic_full": (not ever_recovered_full) if baseline_fn_full else None,
                "recover_count_loc": recover_count_loc,
                "recover_count_full": recover_count_full,
                "recover_fraction_loc": _nullable_fraction(
                    recover_count_loc, len(oracle_prepared)
                ),
                "recover_fraction_full": _nullable_fraction(
                    recover_count_full, len(oracle_prepared)
                ),
                "recovered_run_labels_loc": recovered_labels_loc,
                "recovered_run_labels_full": recovered_labels_full,
                "oracle_runs": per_run_rows,
            }
        )

    summary_primary = {
        "baseline_fn_count_loc": int(sum(1 for row in fn_rows if row["baseline_fn_loc"])),
        "baseline_fn_count_full": int(sum(1 for row in fn_rows if row["baseline_fn_full"])),
        "recoverable_fn_count_loc": int(
            sum(1 for row in fn_rows if row["ever_recovered_loc"] is True)
        ),
        "recoverable_fn_count_full": int(
            sum(1 for row in fn_rows if row["ever_recovered_full"] is True)
        ),
        "systematic_fn_count_loc": int(
            sum(1 for row in fn_rows if row["systematic_loc"] is True)
        ),
        "systematic_fn_count_full": int(
            sum(1 for row in fn_rows if row["systematic_full"] is True)
        ),
        "recover_fraction_loc": 0.0,
        "recover_fraction_full": 0.0,
    }
    if summary_primary["baseline_fn_count_loc"] > 0 and oracle_prepared:
        summary_primary["recover_fraction_loc"] = float(
            _sum_defined(row["recover_count_loc"] for row in fn_rows)
        ) / float(summary_primary["baseline_fn_count_loc"] * len(oracle_prepared))
    if summary_primary["baseline_fn_count_full"] > 0 and oracle_prepared:
        summary_primary["recover_fraction_full"] = float(
            _sum_defined(row["recover_count_full"] for row in fn_rows)
        ) / float(summary_primary["baseline_fn_count_full"] * len(oracle_prepared))

    per_image_rows: List[Dict[str, Any]] = []
    fn_rows_by_record: Dict[int, List[Dict[str, Any]]] = {}
    for row in fn_rows:
        fn_rows_by_record.setdefault(int(row["record_idx"]), []).append(row)
    for sample in baseline_prepared.gt_samples:
        record_idx = int(sample.image_id)
        rows = fn_rows_by_record.get(record_idx, [])
        per_image_rows.append(
            {
                "record_idx": int(record_idx),
                "image_id": int(record_idx),
                "file_name": sample.file_name,
                "gt_count": int(len(sample.objects)),
                "baseline_fn_count_loc": int(sum(1 for row in rows if row["baseline_fn_loc"])),
                "recoverable_fn_count_loc": int(
                    sum(1 for row in rows if row["ever_recovered_loc"] is True)
                ),
                "systematic_fn_count_loc": int(
                    sum(1 for row in rows if row["systematic_loc"] is True)
                ),
                "baseline_fn_count_full": int(sum(1 for row in rows if row["baseline_fn_full"])),
                "recoverable_fn_count_full": int(
                    sum(1 for row in rows if row["ever_recovered_full"] is True)
                ),
                "systematic_fn_count_full": int(
                    sum(1 for row in rows if row["systematic_full"] is True)
                ),
            }
        )

    summary = {
        "out_dir": str(out_dir),
        "baseline_run": _run_meta(baseline_prepared.resolved, eval_dir=baseline_eval_dir),
        "oracle_runs": [
            _run_meta(
                prepared.resolved,
                eval_dir=_run_eval_dir(
                    out_dir,
                    role="oracle",
                    ordinal=idx + 1,
                    label=prepared.resolved.label,
                ),
            )
            for idx, prepared in enumerate(oracle_prepared)
        ],
        "oracle_run_count": int(len(oracle_prepared)),
        "primary_iou_thr": float(primary_thr),
        "f1ish_pred_scope": str(config.eval_options.f1ish_pred_scope),
        "iou_thresholds": thresholds_summary,
        "primary_recovery": summary_primary,
    }

    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "per_image.json").write_text(
        json.dumps(per_image_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_jsonl(out_dir / "fn_objects.jsonl", fn_rows)
    return summary


def run_oracle_k_from_config(config_path: Path) -> Dict[str, Any]:
    config = load_oracle_k_config(config_path)
    return evaluate_oracle_k(config)
