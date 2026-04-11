from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

import torch

from src.eval.detection import EvalOptions, evaluate_and_save


def _write_jsonl(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def _stage2_eval_output_dir(*, owner: Any, global_step: int) -> Path:
    output_root = Path(str(getattr(getattr(owner, "args", None), "output_dir", ".")))
    return output_root / "eval_detection" / f"step_{int(global_step):07d}"


def _build_stage2_eval_infer_summary(
    *,
    owner: Any,
    eval_prompt_variant: str | None,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    eval_detection_score_mode: str,
    eval_detection_cfg: Mapping[str, Any],
    sample_count: int,
    trace_count: int,
) -> Dict[str, Any]:
    return {
        "mode": "stage2_eval_rollout",
        "backend": {
            "type": str(eval_rollout_backend),
            "vllm_mode": str(eval_vllm_mode),
        },
        "generation": {
            "decode_mode": str(owner._cfg("decode_mode", "greedy")),
            "max_new_tokens": int(owner._cfg("max_new_tokens", 0) or 0),
        },
        "infer": {
            "prompt_variant": str(eval_prompt_variant or ""),
            "object_field_order": str(owner._object_field_order()),
            "object_ordering": str(owner._object_ordering()),
            "limit": int(sample_count),
        },
        "eval_detection": {
            "score_mode": str(eval_detection_score_mode),
            "metrics": str(eval_detection_cfg.get("metrics", "coco") or "coco"),
            "trace_count": int(trace_count),
        },
        "counters": {
            "records": int(sample_count),
            "trace_records": int(trace_count),
        },
    }


def _build_eval_options(eval_cfg: Mapping[str, Any], *, output_dir: Path) -> EvalOptions:
    def _coerce_optional_float_list(raw: Any) -> list[float] | None:
        if raw is None:
            return None
        if isinstance(raw, (list, tuple)):
            out: list[float] = []
            for value in raw:
                out.append(float(value))
            return out
        return [float(raw)]

    iou_thrs = _coerce_optional_float_list(eval_cfg.get("iou_thrs", None))
    f1ish_iou_thrs = _coerce_optional_float_list(
        eval_cfg.get("f1ish_iou_thrs", [0.3, 0.5])
    )
    if f1ish_iou_thrs is None:
        f1ish_iou_thrs = [0.3, 0.5]
    return EvalOptions(
        metrics=str(eval_cfg.get("metrics", "coco") or "coco"),
        strict_parse=bool(eval_cfg.get("strict_parse", True)),
        use_segm=bool(eval_cfg.get("use_segm", False)),
        iou_thrs=iou_thrs,
        f1ish_iou_thrs=[float(x) for x in f1ish_iou_thrs],
        f1ish_pred_scope=str(eval_cfg.get("f1ish_pred_scope", "annotated") or "annotated"),
        output_dir=output_dir,
        overlay=False,
        overlay_k=0,
        num_workers=0,
        semantic_model=str(
            eval_cfg.get("semantic_model", "sentence-transformers/all-MiniLM-L6-v2")
            or "sentence-transformers/all-MiniLM-L6-v2"
        ),
        semantic_threshold=float(eval_cfg.get("semantic_threshold", 0.6) or 0.6),
        semantic_device=str(eval_cfg.get("semantic_device", "auto") or "auto"),
        semantic_batch_size=int(eval_cfg.get("semantic_batch_size", 64) or 64),
        lvis_max_dets=int(eval_cfg.get("lvis_max_dets", 300) or 300),
    )


def _materialize_stage2_eval_artifacts(
    *,
    owner: Any,
    global_step: int,
    eval_rollout_artifacts_all: List[Dict[str, Any]],
    eval_prompt_variant: str | None,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    eval_detection_score_mode: str,
    eval_detection_cfg: Mapping[str, Any],
) -> Dict[str, Any]:
    eval_dir = _stage2_eval_output_dir(owner=owner, global_step=global_step)
    eval_dir.mkdir(parents=True, exist_ok=True)

    base_rows: List[Dict[str, Any]] = []
    scored_rows: List[Dict[str, Any]] = []
    raw_rows: List[Dict[str, Any]] = []
    trace_rows: List[Dict[str, Any]] = []
    for record_idx, artifact in enumerate(eval_rollout_artifacts_all):
        base_record = dict(artifact.get("base_record", {}))
        scored_record = dict(artifact.get("scored_record", {}))
        base_record["index"] = int(record_idx)
        scored_record["index"] = int(record_idx)
        base_rows.append(base_record)
        scored_rows.append(scored_record)

        rollout = artifact.get("rollout", {})
        if isinstance(rollout, Mapping):
            generated_token_text = rollout.get("generated_token_text")
            token_logprobs = rollout.get("token_logprobs")
            if isinstance(generated_token_text, list) and isinstance(token_logprobs, list):
                trace_rows.append(
                    {
                        "line_idx": int(record_idx),
                        "generated_token_text": list(generated_token_text),
                        "token_logprobs": [float(x) for x in token_logprobs],
                    }
                )

        artifact_row = dict(artifact)
        artifact_row["index"] = int(record_idx)
        artifact_row["base_record"] = base_record
        artifact_row["scored_record"] = scored_record
        raw_rows.append(artifact_row)

    _write_jsonl(eval_dir / "gt_vs_pred.jsonl", base_rows)
    _write_jsonl(eval_dir / "gt_vs_pred_scored.jsonl", scored_rows)
    _write_jsonl(eval_dir / "raw_rollouts.jsonl", raw_rows)
    if trace_rows:
        _write_jsonl(eval_dir / "pred_token_trace.jsonl", trace_rows)

    resolved_config_path = (
        Path(str(getattr(getattr(owner, "args", None), "output_dir", "")))
        / "resolved_config.json"
    )
    if resolved_config_path.is_file():
        (eval_dir / "resolved_config.path").write_text(
            str(resolved_config_path.resolve()),
            encoding="utf-8",
        )

    infer_summary = _build_stage2_eval_infer_summary(
        owner=owner,
        eval_prompt_variant=eval_prompt_variant,
        eval_rollout_backend=eval_rollout_backend,
        eval_vllm_mode=eval_vllm_mode,
        eval_detection_score_mode=eval_detection_score_mode,
        eval_detection_cfg=eval_detection_cfg,
        sample_count=len(base_rows),
        trace_count=len(trace_rows),
    )
    (eval_dir / "infer_summary.json").write_text(
        json.dumps(infer_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    options = _build_eval_options(eval_detection_cfg, output_dir=eval_dir)
    return evaluate_and_save(eval_dir / "gt_vs_pred_scored.jsonl", options=options)


def finalize_rollout_aligned_evaluation(
    *,
    owner: Any,
    logger: Any,
    metric_key_prefix: str,
    eval_prompt_variant: str | None,
    eval_detection_enabled: bool,
    eval_detection_use_confidence_postop: bool,
    eval_detection_score_mode: str,
    eval_detection_score_version: int,
    eval_detection_score_source: str,
    eval_detection_cfg: Mapping[str, Any],
    desc_enabled: bool,
    n_samples: float,
    gt_total: float,
    pred_total: float,
    matched_total: float,
    fp_total: float,
    fn_total: float,
    gating_rejections_total: float,
    dropped_invalid_total: float,
    dropped_ambiguous_total: float,
    trunc_samples: float,
    matched_iou_sum: float,
    matched_iou_count: float,
    n_samples_valid_pred: float,
    n_samples_any_match: float,
    n_steps: float,
    desc_pairs_total: float,
    desc_exact_ok_total: float,
    desc_sem_ok_total: float,
    desc_sem_sim_sum_total: float,
    desc_sem_sim_count_total: float,
    sem_loaded_local: float,
    trace_fallback_count_local: float,
    vllm_decode_error_count_local: float,
    trace_fallback_window_active: bool,
    runtime_local_s: float,
    eval_detection_records_local: List[Dict[str, Any]],
    eval_rollout_artifacts_local: List[Dict[str, Any]],
    do_dump: bool,
    dump_fail_samples: List[Dict[str, Any]],
    dump_other_samples: List[Dict[str, Any]],
    dump_max_samples: int,
    gs: int,
    eval_rollout_backend: str,
    eval_vllm_mode: str,
    top_k: int,
    gate_thr: float,
    mask_res: int,
    fp_cost: float,
    fn_cost: float,
    was_training: bool,
    compute_eval_detection_coco_metrics_fn: Any,
    metric_name_matches_key_fn: Any,
    stage2_eval_metric_key_fn: Any,
) -> Dict[str, float]:
    try:
        import torch.distributed as dist
    except (TypeError, ValueError):
        dist = None  # type: ignore[assignment]

    world_size = 1
    rank = 0
    if dist is not None and dist.is_available() and dist.is_initialized():
        world_size = int(dist.get_world_size())
        rank = int(dist.get_rank())

    sums_t = torch.tensor(
        [
            n_samples,
            gt_total,
            pred_total,
            matched_total,
            fp_total,
            fn_total,
            gating_rejections_total,
            dropped_invalid_total,
            dropped_ambiguous_total,
            trunc_samples,
            matched_iou_sum,
            matched_iou_count,
            n_samples_valid_pred,
            n_samples_any_match,
            n_steps,
            desc_pairs_total,
            desc_exact_ok_total,
            desc_sem_ok_total,
            desc_sem_sim_sum_total,
            desc_sem_sim_count_total,
            sem_loaded_local,
            trace_fallback_count_local,
            vllm_decode_error_count_local,
            1.0 if trace_fallback_window_active else 0.0,
        ],
        device=owner.model.device,
        dtype=torch.float64,
    )
    rt_t = torch.tensor(
        [float(runtime_local_s)], device=owner.model.device, dtype=torch.float64
    )
    if dist is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(sums_t, op=dist.ReduceOp.SUM)
        dist.all_reduce(rt_t, op=dist.ReduceOp.MAX)

    (
        n_samples,
        gt_total,
        pred_total,
        matched_total,
        fp_total,
        fn_total,
        gating_rejections_total,
        dropped_invalid_total,
        dropped_ambiguous_total,
        trunc_samples,
        matched_iou_sum,
        matched_iou_count,
        n_samples_valid_pred,
        n_samples_any_match,
        n_steps,
        desc_pairs_total,
        desc_exact_ok_total,
        desc_sem_ok_total,
        desc_sem_sim_sum_total,
        desc_sem_sim_count_total,
        sem_loaded_sum,
        trace_fallback_count_local,
        vllm_decode_error_count_local,
        trace_fallback_window_active_sum,
    ) = [float(x.item()) for x in sums_t]
    runtime = float(rt_t.item())
    trace_fallback_window_active = bool(trace_fallback_window_active_sum > 0.0)

    precision = (matched_total / pred_total) if pred_total > 0 else 0.0
    recall = (matched_total / gt_total) if gt_total > 0 else 0.0
    f1 = (
        (2.0 * precision * recall / (precision + recall))
        if (precision + recall) > 0.0
        else 0.0
    )

    def _k(suffix: str) -> str:
        return stage2_eval_metric_key_fn(
            metric_key_prefix=str(metric_key_prefix), suffix=str(suffix)
        )

    metrics: Dict[str, float] = {}
    metrics[_k("time/runtime_s")] = float(runtime)
    if runtime > 0:
        metrics[_k("time/samples_per_s")] = float(n_samples / runtime)
        metrics[_k("time/steps_per_s")] = float(n_steps / runtime)

    metrics[_k("rollout/precision")] = float(precision)
    metrics[_k("rollout/recall")] = float(recall)
    metrics[_k("rollout/f1")] = float(f1)

    metrics[_k("rollout/pred_objects")] = float(pred_total)
    metrics[_k("rollout/gt_objects_total")] = float(gt_total)
    metrics[_k("rollout/matched")] = float(matched_total)
    metrics[_k("rollout/fp_total")] = float(fp_total)
    metrics[_k("rollout/fn_total")] = float(fn_total)
    metrics[_k("rollout/gating_rejections")] = float(gating_rejections_total)

    metrics[_k("rollout/parse_dropped_invalid")] = float(dropped_invalid_total)
    metrics[_k("rollout/parse_dropped_ambiguous")] = float(dropped_ambiguous_total)
    metrics[_k("rollout/parse_truncated_rate")] = (
        float(trunc_samples / n_samples) if n_samples > 0 else 0.0
    )

    metrics[_k("rollout/sample_valid_pred_rate")] = (
        float(n_samples_valid_pred / n_samples) if n_samples > 0 else 0.0
    )
    metrics[_k("rollout/sample_any_match_rate")] = (
        float(n_samples_any_match / n_samples) if n_samples > 0 else 0.0
    )

    metrics[_k("rollout/matched_maskiou_mean")] = (
        float(matched_iou_sum / matched_iou_count) if matched_iou_count > 0 else 0.0
    )
    metrics[_k("rollout/trace_fallback_count")] = float(trace_fallback_count_local)
    metrics[_k("rollout/vllm_decode_error_count")] = float(
        vllm_decode_error_count_local
    )

    if desc_enabled:
        metrics[_k("rollout/desc_pairs_total")] = float(desc_pairs_total)
        exact_acc = (
            float(desc_exact_ok_total / desc_pairs_total)
            if desc_pairs_total > 0
            else 1.0
        )
        metrics[_k("rollout/desc_exact_acc_on_matched")] = float(exact_acc)

        sem_enabled = bool(sem_loaded_sum >= float(world_size) - 0.5)
        metrics[_k("rollout/desc_sem_enabled")] = float(1.0 if sem_enabled else 0.0)
        if sem_enabled:
            sem_acc = (
                float(desc_sem_ok_total / desc_pairs_total)
                if desc_pairs_total > 0
                else 1.0
            )
            metrics[_k("rollout/desc_sem_acc_on_matched")] = float(sem_acc)
            if desc_sem_sim_count_total > 0:
                metrics[_k("rollout/desc_sem_sim_mean")] = float(
                    desc_sem_sim_sum_total / desc_sem_sim_count_total
                )
                metrics[_k("rollout/desc_sem_sim_count")] = float(
                    desc_sem_sim_count_total
                )

    if eval_prompt_variant is not None:
        metrics[_k("rollout/prompt_variant_is_coco_80")] = (
            1.0 if str(eval_prompt_variant).strip().lower() == "coco_80" else 0.0
        )

    if eval_detection_enabled:
        cfg_mode = str(eval_detection_score_mode or "constant").strip().lower()
        metrics[_k("rollout/config_score_mode_is_constant")] = float(
            1.0 if cfg_mode == "constant" else 0.0
        )
        metrics[_k("rollout/config_score_mode_is_confidence_postop")] = float(
            1.0 if cfg_mode in {"confidence_postop", "confidence"} else 0.0
        )

        effective_confidence_postop = bool(
            eval_detection_use_confidence_postop and not trace_fallback_window_active
        )
        eff_mode = "confidence_postop" if effective_confidence_postop else "constant"
        metrics[_k("rollout/effective_score_mode_is_constant")] = float(
            1.0 if eff_mode == "constant" else 0.0
        )
        metrics[_k("rollout/effective_score_mode_is_confidence_postop")] = float(
            1.0 if eff_mode == "confidence_postop" else 0.0
        )

        metrics[_k("rollout/config_pred_score_version")] = float(
            int(eval_detection_score_version)
        )
        eff_version = (
            2 if effective_confidence_postop else int(eval_detection_score_version)
        )
        metrics[_k("rollout/effective_pred_score_version")] = float(int(eff_version))

        cfg_source = str(eval_detection_score_source or "").strip()
        eff_source = "confidence_postop" if effective_confidence_postop else cfg_source
        metrics[_k("rollout/config_pred_score_source_is_eval_rollout_constant")] = float(
            1.0 if cfg_source == "eval_rollout_constant" else 0.0
        )
        metrics[_k("rollout/config_pred_score_source_is_confidence_postop")] = float(
            1.0 if cfg_source == "confidence_postop" else 0.0
        )
        metrics[
            _k("rollout/effective_pred_score_source_is_eval_rollout_constant")
        ] = float(1.0 if eff_source == "eval_rollout_constant" else 0.0)
        metrics[_k("rollout/effective_pred_score_source_is_confidence_postop")] = float(
            1.0 if eff_source == "confidence_postop" else 0.0
        )

    if eval_detection_enabled:
        eval_records_all: List[Dict[str, Any]] = [
            dict(record) for record in eval_detection_records_local
        ]
        eval_rollout_artifacts_all: List[Dict[str, Any]] = [
            dict(record) for record in eval_rollout_artifacts_local
        ]
        if dist is not None and dist.is_available() and dist.is_initialized():
            eval_records_all = []
            eval_rollout_artifacts_all = []
            gather_object = getattr(dist, "gather_object", None)
            if callable(gather_object):
                gathered_records = (
                    [None for _ in range(int(world_size))] if int(rank) == 0 else None
                )
                gathered_rollout_artifacts = (
                    [None for _ in range(int(world_size))] if int(rank) == 0 else None
                )
                try:
                    gather_object(
                        list(eval_detection_records_local),
                        object_gather_list=gathered_records,
                        dst=0,
                    )
                except TypeError:
                    gather_object(
                        list(eval_detection_records_local),
                        gathered_records,
                        0,
                    )
                try:
                    gather_object(
                        list(eval_rollout_artifacts_local),
                        object_gather_list=gathered_rollout_artifacts,
                        dst=0,
                    )
                except TypeError:
                    gather_object(
                        list(eval_rollout_artifacts_local),
                        gathered_rollout_artifacts,
                        0,
                    )
                if int(rank) == 0 and isinstance(gathered_records, list):
                    for src_rank, part in enumerate(gathered_records):
                        if not isinstance(part, list):
                            raise TypeError(
                                "eval_detection gather_object returned non-list part: "
                                f"src_rank={int(src_rank)} type={type(part).__name__}"
                            )
                        for rec in part:
                            if isinstance(rec, Mapping):
                                eval_records_all.append(dict(rec))
                if int(rank) == 0 and isinstance(gathered_rollout_artifacts, list):
                    for src_rank, part in enumerate(gathered_rollout_artifacts):
                        if not isinstance(part, list):
                            raise TypeError(
                                "eval_rollout_artifacts gather_object returned non-list part: "
                                f"src_rank={int(src_rank)} type={type(part).__name__}"
                            )
                        for rec in part:
                            if isinstance(rec, Mapping):
                                eval_rollout_artifacts_all.append(dict(rec))
            else:
                gathered_records = [None for _ in range(int(world_size))]
                gathered_rollout_artifacts = [None for _ in range(int(world_size))]
                dist.all_gather_object(
                    gathered_records, list(eval_detection_records_local)
                )
                dist.all_gather_object(
                    gathered_rollout_artifacts,
                    list(eval_rollout_artifacts_local),
                )
                for src_rank, part in enumerate(gathered_records):
                    if not isinstance(part, list):
                        raise TypeError(
                            "eval_detection all_gather_object returned non-list part: "
                            f"src_rank={int(src_rank)} type={type(part).__name__}"
                        )
                    for rec in part:
                        if isinstance(rec, Mapping):
                            eval_records_all.append(dict(rec))
                for src_rank, part in enumerate(gathered_rollout_artifacts):
                    if not isinstance(part, list):
                        raise TypeError(
                            "eval_rollout_artifacts all_gather_object returned non-list part: "
                            f"src_rank={int(src_rank)} type={type(part).__name__}"
                        )
                    for rec in part:
                        if isinstance(rec, Mapping):
                            eval_rollout_artifacts_all.append(dict(rec))

        for record_idx, record in enumerate(eval_records_all):
            record["index"] = int(record_idx)

        eval_det_payload: Dict[str, Any] = {
            "ok": 0.0,
            "runtime_s": 0.0,
            "metrics": {},
            "counters": {},
            "error": "",
        }
        eval_det_exc: Exception | None = None
        if int(rank) == 0:
            t_coco0 = time.perf_counter()
            try:
                output_dir_raw = str(
                    getattr(getattr(owner, "args", None), "output_dir", "") or ""
                ).strip()
                should_materialize_artifacts = bool(
                    eval_detection_cfg.get("materialize_artifacts", True)
                )
                if output_dir_raw and should_materialize_artifacts:
                    eval_summary = _materialize_stage2_eval_artifacts(
                        owner=owner,
                        global_step=int(gs),
                        eval_rollout_artifacts_all=eval_rollout_artifacts_all,
                        eval_prompt_variant=eval_prompt_variant,
                        eval_rollout_backend=eval_rollout_backend,
                        eval_vllm_mode=eval_vllm_mode,
                        eval_detection_score_mode=eval_detection_score_mode,
                        eval_detection_cfg=eval_detection_cfg,
                    )
                    coco_metrics = eval_summary.get("metrics", {})
                    coco_counters = eval_summary.get("counters", {})
                else:
                    coco_metrics, coco_counters = compute_eval_detection_coco_metrics_fn(
                        pred_records=eval_records_all,
                        eval_cfg=eval_detection_cfg,
                    )
                eval_det_payload["ok"] = 1.0
                eval_det_payload["metrics"] = {
                    str(k): float(v) for k, v in coco_metrics.items()
                }
                eval_det_payload["counters"] = {
                    str(k): int(v)
                    for k, v in coco_counters.items()
                    if isinstance(v, (int, float))
                }
            except Exception as exc:
                eval_det_exc = exc
                eval_det_payload["error"] = repr(exc)
                logger.exception("Eval-step COCO/mAP failed")
            finally:
                eval_det_payload["runtime_s"] = float(time.perf_counter() - t_coco0)

        if dist is not None and dist.is_available() and dist.is_initialized():
            payload_list: List[Any] = [eval_det_payload]

            backend = None
            try:
                backend = str(dist.get_backend())
            except Exception:
                backend = None

            broadcast_device = torch.device("cpu")
            if backend == "nccl":
                if not torch.cuda.is_available():
                    raise RuntimeError(
                        "torch.distributed backend is NCCL but CUDA is not available"
                    )
                broadcast_device = torch.device(
                    "cuda", int(torch.cuda.current_device())
                )

            try:
                dist.broadcast_object_list(
                    payload_list, src=0, device=broadcast_device
                )
            except TypeError as exc:
                if backend == "nccl":
                    raise RuntimeError(
                        "broadcast_object_list(..., device=...) is required for NCCL; "
                        "upgrade PyTorch to a version that supports the 'device' argument."
                    ) from exc
                dist.broadcast_object_list(payload_list, src=0)

            if not isinstance(payload_list[0], Mapping):
                raise TypeError(
                    "eval_detection broadcast payload is not a Mapping: "
                    f"type={type(payload_list[0]).__name__}"
                )
            recv_payload = dict(payload_list[0])
        else:
            recv_payload = eval_det_payload

        try:
            metrics[_k("time/coco_eval_runtime_s")] = float(
                recv_payload.get("runtime_s", 0.0) or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            metrics[_k("time/coco_eval_runtime_s")] = 0.0
        try:
            metrics[_k("rollout/coco_eval_ok")] = float(
                recv_payload.get("ok", 0.0) or 0.0
            )
        except (AttributeError, TypeError, ValueError):
            metrics[_k("rollout/coco_eval_ok")] = 0.0

        metrics[_k("rollout/mAP")] = 0.0
        coco_metrics_recv = recv_payload.get("metrics", {})
        if isinstance(coco_metrics_recv, Mapping) and "bbox_AP" in coco_metrics_recv:
            try:
                metrics[_k("rollout/mAP")] = float(coco_metrics_recv["bbox_AP"])
            except (TypeError, ValueError):
                metrics[_k("rollout/mAP")] = 0.0

        metric_for_best_model = str(
            getattr(getattr(owner, "args", None), "metric_for_best_model", "") or ""
        ).strip()
        coco_ok = float(metrics.get(_k("rollout/coco_eval_ok"), 0.0) or 0.0)
        if coco_ok <= 0.0:
            coco_err = str(recv_payload.get("error", "") or "").strip()
            if metric_name_matches_key_fn(
                metric_for_best_model,
                stage2_eval_metric_key_fn("eval", "rollout/mAP"),
            ):
                msg = (
                    "Eval-step COCO/mAP failed while metric_for_best_model targets eval/detection/mAP; "
                    "aborting to avoid invalid best-checkpoint selection. "
                    f"error={coco_err or 'unknown'}"
                )
            else:
                msg = (
                    "Eval-step COCO/mAP failed; aborting (fail-fast) to avoid silent metric corruption. "
                    f"error={coco_err or 'unknown'}"
                )
            if int(rank) == 0 and eval_det_exc is not None:
                raise RuntimeError(msg) from eval_det_exc
            raise RuntimeError(msg)

        coco_counters_recv = recv_payload.get("counters", {})
        if isinstance(coco_counters_recv, Mapping):
            for name in (
                "empty_pred",
                "invalid_geometry",
                "invalid_coord",
                "missing_size",
                "size_mismatch",
                "degenerate",
                "unknown_dropped",
                "semantic_mapped",
                "semantic_unmapped",
            ):
                if name not in coco_counters_recv:
                    continue
                try:
                    metrics[_k(f"rollout/coco_counter_{name}")] = float(
                        coco_counters_recv[name]
                    )
                except (TypeError, ValueError):
                    continue

    if do_dump:
        try:
            samples_out: List[Dict[str, Any]] = list(dump_fail_samples)
            if len(samples_out) < dump_max_samples:
                need = int(dump_max_samples) - int(len(samples_out))
                samples_out.extend(list(dump_other_samples)[:need])

            payload = {
                "kind": "eval_monitor_dump",
                "global_step": int(gs),
                "epoch": float(
                    getattr(getattr(owner, "state", None), "epoch", 0.0) or 0.0
                ),
                "time": float(time.time()),
                "meta": {
                    "phase": "eval",
                    "metric_key_prefix": str(metric_key_prefix),
                    "rollout_backend": str(eval_rollout_backend),
                    "vllm_mode": str(eval_vllm_mode),
                    "decode_mode": str(owner._cfg("decode_mode", "greedy")),
                    "max_new_tokens": int(owner._cfg("max_new_tokens", 0) or 0),
                    "candidate_top_k": int(top_k),
                    "maskiou_gate": float(gate_thr),
                    "maskiou_resolution": int(mask_res),
                    "fp_cost": float(fp_cost),
                    "fn_cost": float(fn_cost),
                },
                "metrics": metrics,
                "samples": samples_out,
            }
            owner._write_monitor_dump(global_step=int(gs), payload=payload)
            owner._eval_monitor_dump_count += 1
        except Exception as exc:
            logger.warning(
                "Failed to write eval monitor dump at global_step=%s: %r",
                int(gs),
                exc,
            )

    owner.log(metrics)
    owner.control = owner.callback_handler.on_evaluate(
        owner.args, owner.state, owner.control, metrics
    )

    if was_training:
        owner.model.train()

    return metrics
