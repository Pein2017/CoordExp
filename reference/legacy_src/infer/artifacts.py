from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _read_json_file(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_provenance: {path} must contain a JSON object")
    return payload


_CANONICAL_PROVENANCE_KEYS = (
    "prompt_policy_fingerprint",
    "decode_policy_fingerprint",
    "model_identity_fingerprint",
)
_SCORE_PROVENANCE_KEY = "score_policy_fingerprint"
_RAW_SCORE_POLICY_KEY = "score_policy"
_RAW_SCORE_POLICY_NONE = "none"
_TRANSITIONAL_FINGERPRINT_PREFIX = "transitional_"
_CANONICAL_SCORED_ARTIFACT_NAMES = frozenset(
    {
        "gt_vs_pred_scored.jsonl",
        "gt_vs_pred_scored_guarded.jsonl",
    }
)
_CANONICAL_RAW_ARTIFACT_NAMES = frozenset(
    {
        "gt_vs_pred.jsonl",
        "gt_vs_pred_guarded.jsonl",
    }
)


def confidence_options_from_eval_config(raw_confidence: Any) -> Any:
    """Build confidence-postop options from the Stage-2 eval config payload."""

    from src.eval.confidence_postop import options_from_config

    if raw_confidence is None:
        conf_cfg: Mapping[str, Any] = {}
    elif isinstance(raw_confidence, Mapping):
        conf_cfg = raw_confidence
    else:
        try:
            conf_cfg = asdict(raw_confidence)
        except Exception as exc:
            raise TypeError(
                "rollout_matching.eval_detection.confidence must be a mapping"
            ) from exc

    return options_from_config({"confidence": conf_cfg})


def score_stage2_confidence_eval_record(
    *,
    line_idx: int,
    base_eval_record: Mapping[str, Any],
    parse: Any,
    token_logprobs: Optional[Sequence[float]],
    generated_token_text: Optional[Sequence[str]],
    confidence_postop_opts: Any,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Validate generated-token traces and build a scored Stage-2 eval record."""

    from src.eval.confidence_postop import (
        TraceRecord,
        _build_scored_record,
        _compute_sample_confidence_objects,
    )

    trace_len = int(len(getattr(parse, "response_token_ids", []) or []))
    trace_invalid_reason: Optional[str] = None
    if confidence_postop_opts is None:
        trace_invalid_reason = "confidence_postop_opts missing for eval-step scoring"
    elif token_logprobs is None or generated_token_text is None:
        trace_invalid_reason = "eval-step confidence scoring requires token traces"
    elif len(token_logprobs) < trace_len:
        trace_invalid_reason = (
            "rollout trace shorter than parsed response_token_ids: "
            f"trace_len={len(token_logprobs)} parsed_len={trace_len}"
        )
    elif len(generated_token_text) < trace_len:
        trace_invalid_reason = (
            "generated_token_text shorter than parsed response_token_ids: "
            f"trace_text_len={len(generated_token_text)} parsed_len={trace_len}"
        )
    elif any(not math.isfinite(float(x)) for x in token_logprobs[:trace_len]):
        trace_invalid_reason = "rollout trace contains non-finite logprobs"

    if trace_invalid_reason is not None:
        raise RuntimeError(
            "Eval confidence trace invariant violation: "
            f"line_idx={int(line_idx)} reason={trace_invalid_reason}"
        )

    try:
        assert token_logprobs is not None
        assert generated_token_text is not None
        trace = TraceRecord(
            line_idx=int(line_idx),
            generated_token_text=list(generated_token_text[:trace_len]),
            token_logprobs=[float(x) for x in token_logprobs[:trace_len]],
        )
        confidence_objects = _compute_sample_confidence_objects(
            line_idx=int(line_idx),
            record=dict(base_eval_record),
            trace=trace,
            options=confidence_postop_opts,
        )
        confidence_objects_payload = [dict(obj) for obj in confidence_objects]
        scored_record = _build_scored_record(
            record=dict(base_eval_record),
            confidence_objects=confidence_objects,
        )
        return dict(scored_record), confidence_objects_payload
    except Exception as exc:
        raise RuntimeError(
            "Eval confidence scoring failed: "
            f"line_idx={int(line_idx)} error={exc.__class__.__name__}: {exc}"
        ) from exc


def _json_safe_exact_int_sequence(values: Sequence[Any]) -> list[Any]:
    out: list[Any] = []
    for value in list(values):
        if isinstance(value, bool):
            out.append(value)
        elif isinstance(value, Integral):
            out.append(int(value))
        else:
            out.append(value)
    return out


def build_stage2_rollout_eval_artifact_record(
    *,
    eval_record_index: int,
    sample: Mapping[str, Any],
    base_eval_record: Mapping[str, Any],
    scored_eval_record: Mapping[str, Any],
    response_token_ids: Sequence[int],
    prompt_token_ids: Sequence[int],
    decode_mode: str,
    response_text: str,
    generated_token_text: Optional[Sequence[str]],
    token_logprobs: Optional[Sequence[float]],
    parse: Any,
    pred_objects_dump: Sequence[Mapping[str, Any]],
    eval_error_codes: Sequence[str],
    eval_error_entries: Sequence[Mapping[str, Any]],
    match: Any,
    confidence_objects_payload: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Build the offline-compatible Stage-2 eval rollout artifact payload."""

    return {
        "index": int(eval_record_index),
        "sample_id": sample.get("sample_id"),
        "base_idx": sample.get("base_idx"),
        "image": base_eval_record.get("image"),
        "images": list(base_eval_record.get("images", [])),
        "width": base_eval_record.get("width"),
        "height": base_eval_record.get("height"),
        "image_id": sample.get("image_id"),
        "metadata": (
            dict(sample.get("metadata"))
            if isinstance(sample.get("metadata"), Mapping)
            else None
        ),
        "base_record": dict(base_eval_record),
        "scored_record": dict(scored_eval_record),
        "rollout": {
            "decode_mode": str(decode_mode),
            "response_token_ids": [int(x) for x in list(response_token_ids)],
            "prompt_token_ids": _json_safe_exact_int_sequence(
                list(prompt_token_ids),
            ),
            "response_text": str(response_text or ""),
            "generated_token_text": (
                list(generated_token_text) if generated_token_text is not None else None
            ),
            "token_logprobs": (
                [float(x) for x in list(token_logprobs)]
                if token_logprobs is not None
                else None
            ),
            "trace_invalid_reason": None,
        },
        "parse": {
            "invalid_rollout": bool(getattr(parse, "invalid_rollout", False)),
            "dropped_invalid": int(getattr(parse, "dropped_invalid", 0) or 0),
            "dropped_ambiguous": int(getattr(parse, "dropped_ambiguous", 0) or 0),
            "truncated": bool(getattr(parse, "truncated", False)),
            "response_token_ids": [
                int(x) for x in list(getattr(parse, "response_token_ids", []))
            ],
            "response_text": str(getattr(parse, "response_text", "") or ""),
            "prefix_text": str(getattr(parse, "prefix_text", "") or ""),
            "valid_objects": [dict(obj) for obj in list(pred_objects_dump)],
            "errors": list(eval_error_codes),
            "error_entries": [dict(entry) for entry in list(eval_error_entries)],
        },
        "match": {
            "matched_pairs": [[int(a), int(b)] for a, b in list(match.matched_pairs)],
            "fp_pred_indices": [int(x) for x in list(match.fp_pred_indices)],
            "fn_gt_indices": [int(x) for x in list(match.fn_gt_indices)],
            "gating_rejections": int(match.gating_rejections),
            "matched_maskiou_sum": float(getattr(match, "matched_maskiou_sum", 0.0)),
            "matched_maskiou_count": int(getattr(match, "matched_maskiou_count", 0)),
        },
        "confidence_objects": [
            dict(obj) for obj in list(confidence_objects_payload)
        ],
    }


def build_score_policy_fingerprint(
    *,
    policy_name: str,
    score_source: str,
    aggregation_rule: str,
    token_span_rule: str,
    constant_score_value: Any,
    source_raw_artifact_identity: Any,
    parser_policy: str,
    metric_bearing: bool,
) -> str:
    """Build a stable fingerprint for score-bearing inference artifacts."""

    if not isinstance(metric_bearing, bool):
        raise TypeError("metric_bearing must be a bool")
    payload = {
        "aggregation_rule": aggregation_rule,
        "constant_score_value": constant_score_value,
        "metric_bearing": metric_bearing,
        "parser_policy": parser_policy,
        "policy_name": policy_name,
        "score_source": score_source,
        "source_raw_artifact_identity": _score_source_identity_for_fingerprint(
            source_raw_artifact_identity
        ),
        "token_span_rule": token_span_rule,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return "score_policy:" + hashlib.sha256(encoded).hexdigest()


def _score_source_identity_for_fingerprint(identity: Any) -> Any:
    if isinstance(identity, Mapping) and "sha256" in identity:
        # Raw paths remain in sidecars for human lineage, but the score-policy
        # identity should survive equivalent artifact moves across machines.
        return {"sha256": identity.get("sha256")}
    return identity


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_score_provenance_sidecar(
    *,
    scored_path: Path,
    prompt_policy_fingerprint: str,
    decode_policy_fingerprint: str,
    model_identity_fingerprint: str,
    policy_name: str,
    score_source: str,
    aggregation_rule: str,
    token_span_rule: str,
    constant_score_value: Any,
    source_raw_artifact_path: Optional[Path],
    parser_policy: str,
    metric_bearing: bool = True,
    extra: Optional[Mapping[str, Any]] = None,
) -> Path:
    source_raw_artifact_identity: Any = None
    if source_raw_artifact_path is not None:
        raw_path = Path(source_raw_artifact_path)
        source_raw_artifact_identity = {
            "path": str(raw_path),
            "sha256": file_sha256(raw_path),
        }
    score_policy_fingerprint = build_score_policy_fingerprint(
        policy_name=policy_name,
        score_source=score_source,
        aggregation_rule=aggregation_rule,
        token_span_rule=token_span_rule,
        constant_score_value=constant_score_value,
        source_raw_artifact_identity=source_raw_artifact_identity,
        parser_policy=parser_policy,
        metric_bearing=metric_bearing,
    )
    sidecar_payload: Dict[str, Any] = {
        "prompt_policy_fingerprint": prompt_policy_fingerprint,
        "decode_policy_fingerprint": decode_policy_fingerprint,
        "model_identity_fingerprint": model_identity_fingerprint,
        "score_policy_fingerprint": score_policy_fingerprint,
        "metric_bearing": bool(metric_bearing),
        "artifact_path": str(scored_path),
        "parser_policy": parser_policy,
        "source_raw_artifact_identity": source_raw_artifact_identity,
        "score_policy": {
            "policy_name": policy_name,
            "score_source": score_source,
            "aggregation_rule": aggregation_rule,
            "token_span_rule": token_span_rule,
            "constant_score_value": constant_score_value,
        },
    }
    if source_raw_artifact_path is not None:
        sidecar_payload["source_raw_artifact"] = str(source_raw_artifact_path)
    if extra:
        sidecar_payload.update(dict(extra))
    scored = Path(scored_path)
    sidecar_path = scored.with_suffix(scored.suffix + ".provenance.json")
    sidecar_path.write_text(
        json.dumps(sidecar_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return sidecar_path


def _candidate_provenance_payloads(payload: Dict[str, Any]) -> tuple[Dict[str, Any], ...]:
    candidates = [payload]
    for key in ("provenance", "inference_provenance"):
        value = payload.get(key)
        if isinstance(value, dict):
            candidates.append(value)
    return tuple(candidates)


def _is_canonical_fingerprint(value: Any) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and not value.startswith(_TRANSITIONAL_FINGERPRINT_PREFIX)
    )


def _has_explicit_scored_role(payload: Dict[str, Any]) -> bool:
    for candidate in _candidate_provenance_payloads(payload):
        if _is_canonical_fingerprint(candidate.get(_SCORE_PROVENANCE_KEY)):
            return True
        score_policy = candidate.get(_RAW_SCORE_POLICY_KEY)
        if isinstance(score_policy, str) and score_policy.strip():
            return score_policy != _RAW_SCORE_POLICY_NONE
    return False


def _has_explicit_raw_role(payload: Dict[str, Any]) -> bool:
    for candidate in _candidate_provenance_payloads(payload):
        if candidate.get(_RAW_SCORE_POLICY_KEY) == _RAW_SCORE_POLICY_NONE:
            return True
    return False


def _is_score_bearing_artifact(
    path: Path,
    provenance: Dict[str, Any],
    *,
    carrier: Path,
) -> bool:
    if path.name in _CANONICAL_SCORED_ARTIFACT_NAMES:
        return True
    if path.name in _CANONICAL_RAW_ARTIFACT_NAMES:
        return False
    if _has_explicit_scored_role(provenance):
        return True
    if _has_explicit_raw_role(provenance):
        return False
    raise ValueError(
        f"missing_provenance: {carrier} lacks explicit score role metadata "
        f"for non-canonical artifact {path.name}"
    )


def _has_canonical_string_provenance(
    payload: Dict[str, Any],
    *,
    require_score: bool,
) -> bool:
    if payload.get("comparable") is False:
        return False
    if payload.get("metric_bearing") is False:
        return False
    for candidate in _candidate_provenance_payloads(payload):
        values = [candidate.get(key) for key in _CANONICAL_PROVENANCE_KEYS]
        if not all(_is_canonical_fingerprint(value) for value in values):
            continue
        if require_score and not _is_canonical_fingerprint(
            candidate.get(_SCORE_PROVENANCE_KEY)
        ):
            continue
        if not require_score and candidate.get(_RAW_SCORE_POLICY_KEY) != (
            _RAW_SCORE_POLICY_NONE
        ):
            continue
        if candidate.get("comparable") is False:
            continue
        if candidate.get("metric_bearing") is False:
            continue
        return True
    return False


def _detection_template_id_from_payload(payload: Mapping[str, Any]) -> str | None:
    from src.detection.template_contracts import resolve_detection_template_contract

    candidates: list[Mapping[str, Any]] = [payload]
    for key in ("infer", "provenance", "inference_provenance"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            candidates.append(value)

    for candidate in candidates:
        raw_id = candidate.get("detection_template_id")
        if isinstance(raw_id, str) and raw_id.strip():
            return resolve_detection_template_contract(raw_id).template_id
        template = candidate.get("detection_template")
        if isinstance(template, Mapping):
            nested_id = template.get("id")
            if isinstance(nested_id, str) and nested_id.strip():
                return resolve_detection_template_contract(nested_id).template_id
    return None


def _artifact_path_matches(value: Any, *, carrier: Path, artifact_path: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    candidate = Path(value).expanduser()
    candidate_paths = [candidate]
    if not candidate.is_absolute():
        candidate_paths.append(carrier.parent / candidate)
    resolved_artifact_path = artifact_path.resolve()
    return any(path.resolve() == resolved_artifact_path for path in candidate_paths)


def _is_provenance_bound_to_artifact(
    provenance: Dict[str, Any],
    *,
    carrier: Path,
    artifact_path: Path,
) -> bool:
    for candidate in _candidate_provenance_payloads(provenance):
        for key in ("artifact_path", "artifact_jsonl"):
            if _artifact_path_matches(
                candidate.get(key),
                carrier=carrier,
                artifact_path=artifact_path,
            ):
                return True
    artifacts = provenance.get("artifacts")
    if isinstance(artifacts, dict):
        for key in (
            "gt_vs_pred_jsonl",
            "gt_vs_pred_scored_jsonl",
            "gt_vs_pred_guarded_jsonl",
            "gt_vs_pred_scored_guarded_jsonl",
        ):
            if _artifact_path_matches(
                artifacts.get(key),
                carrier=carrier,
                artifact_path=artifact_path,
            ):
                return True
    return False


def _validate_comparable_provenance(
    provenance: Dict[str, Any],
    *,
    carrier: Path,
    artifact_path: Path,
    require_artifact_binding: bool,
    force_score_bearing: bool = False,
) -> None:
    if force_score_bearing and artifact_path.name in _CANONICAL_RAW_ARTIFACT_NAMES:
        raise ValueError(
            f"missing_provenance: {artifact_path} is a raw artifact family name "
            "and cannot satisfy score-bearing provenance"
        )
    if require_artifact_binding and not _is_provenance_bound_to_artifact(
        provenance,
        carrier=carrier,
        artifact_path=artifact_path,
    ):
        raise ValueError(
            f"missing_provenance: {carrier} is not bound to artifact {artifact_path}"
        )
    require_score = force_score_bearing or _is_score_bearing_artifact(
        artifact_path,
        provenance,
        carrier=carrier,
    )
    if _detection_template_id_from_payload(provenance) is None:
        raise ValueError(
            f"missing_provenance: {carrier} lacks detection_template.id"
        )
    if not _has_canonical_string_provenance(
        provenance,
        require_score=require_score,
    ):
        required_fields = list(_CANONICAL_PROVENANCE_KEYS)
        if require_score:
            required_fields.append(_SCORE_PROVENANCE_KEY)
        else:
            required_fields.append(_RAW_SCORE_POLICY_KEY)
        missing_fields = ",".join(required_fields)
        raise ValueError(
            f"missing_provenance: {carrier} lacks canonical string fields "
            f"{missing_fields}"
        )


def _owner_cfg_value(owner: Any, name: str) -> Any:
    value = getattr(owner, name, None)
    if value is not None:
        return value
    cfg = getattr(owner, "cfg", None)
    if cfg is not None:
        return getattr(cfg, name, None)
    return None


def _build_inference_provenance(
    *,
    owner: Any,
    backend: str,
    batch_size: int,
    checkpoint_meta: Dict[str, Any],
    generation_meta: Dict[str, Any],
) -> Dict[str, Any]:
    missing: list[str] = []
    invalid: list[str] = []
    prompt_policy_fingerprint = _owner_cfg_value(owner, "prompt_policy_fingerprint")
    if prompt_policy_fingerprint is None:
        missing.append("prompt_policy_fingerprint")
    elif not _is_canonical_fingerprint(prompt_policy_fingerprint):
        invalid.append("prompt_policy_fingerprint")
        prompt_policy_fingerprint = None

    decode_policy_fingerprint = _owner_cfg_value(owner, "decode_policy_fingerprint")
    if decode_policy_fingerprint is None:
        missing.append("decode_policy_fingerprint")
    elif not _is_canonical_fingerprint(decode_policy_fingerprint):
        invalid.append("decode_policy_fingerprint")
        decode_policy_fingerprint = None

    model_identity_fingerprint = _owner_cfg_value(owner, "model_identity_fingerprint")
    if model_identity_fingerprint is None:
        missing.append("model_identity_fingerprint")
    elif not _is_canonical_fingerprint(model_identity_fingerprint):
        invalid.append("model_identity_fingerprint")
        model_identity_fingerprint = None

    provenance: Dict[str, Any] = {
        "comparable": not missing and not invalid,
        "missing_provenance_fields": tuple(missing),
        "invalid_provenance_fields": tuple(invalid),
        "score_policy": _RAW_SCORE_POLICY_NONE,
    }
    if prompt_policy_fingerprint is not None:
        provenance["prompt_policy_fingerprint"] = prompt_policy_fingerprint
    if decode_policy_fingerprint is not None:
        provenance["decode_policy_fingerprint"] = decode_policy_fingerprint
    if model_identity_fingerprint is not None:
        provenance["model_identity_fingerprint"] = model_identity_fingerprint
    return provenance


def _resolve_config_pointer(pointer_path: Path) -> Optional[Path]:
    raw = pointer_path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    resolved = Path(raw).expanduser()
    if not resolved.is_absolute():
        resolved = pointer_path.parent / resolved
    if resolved.is_dir():
        resolved = resolved / "resolved_config.json"
    return resolved


def _provenance_carriers(path: Path) -> tuple[Path, ...]:
    return (
        path.with_suffix(path.suffix + ".provenance.json"),
        path.parent / "resolved_config.json",
        path.parent / "summary.json",
        path.parent / "infer_summary.json",
        path.parent / "resolved_config.path",
    )


def load_comparable_artifact(
    path: Path,
    *,
    require_score: bool = False,
) -> Dict[str, Any]:
    artifact_path = Path(path)
    diagnostics: list[str] = []
    for carrier in _provenance_carriers(artifact_path):
        if not carrier.exists():
            continue
        if carrier.name == "resolved_config.path":
            resolved = _resolve_config_pointer(carrier)
            if resolved is None or not resolved.exists():
                continue
            provenance = _read_json_file(resolved)
            try:
                _validate_comparable_provenance(
                    provenance,
                    carrier=resolved,
                    artifact_path=artifact_path,
                    require_artifact_binding=True,
                    force_score_bearing=require_score,
                )
            except ValueError as exc:
                diagnostics.append(str(exc))
                continue
            return {
                "artifact_path": str(artifact_path),
                "provenance_path": str(resolved),
                "provenance_carrier": str(carrier),
                "provenance": provenance,
            }
        provenance = _read_json_file(carrier)
        try:
            _validate_comparable_provenance(
                provenance,
                carrier=carrier,
                artifact_path=artifact_path,
                require_artifact_binding=(
                    require_score
                    or carrier
                    != artifact_path.with_suffix(
                        artifact_path.suffix + ".provenance.json"
                    )
                ),
                force_score_bearing=require_score,
            )
        except ValueError as exc:
            diagnostics.append(str(exc))
            continue
        return {
            "artifact_path": str(artifact_path),
            "provenance_path": str(carrier),
            "provenance_carrier": str(carrier),
            "provenance": provenance,
        }
    checked = ", ".join(str(carrier) for carrier in _provenance_carriers(artifact_path))
    detail = ""
    if diagnostics:
        detail = "; diagnostics: " + " | ".join(diagnostics)
    raise ValueError(
        f"missing_provenance: no provenance carrier found for {artifact_path}; "
        f"checked {checked}{detail}"
    )


def _checkpoint_meta(owner: Any) -> Dict[str, Any]:
    return {
        "checkpoint_mode": getattr(owner.cfg, "checkpoint_mode", "full_model"),
        "requested_model_checkpoint": getattr(
            owner.cfg, "requested_model_checkpoint", owner.cfg.model_checkpoint
        ),
        "requested_adapter_checkpoint": getattr(
            owner.cfg, "requested_adapter_checkpoint", owner.cfg.adapter_checkpoint
        ),
        "resolved_base_model_checkpoint": getattr(
            owner.cfg, "resolved_base_model_checkpoint", owner.cfg.model_checkpoint
        ),
        "resolved_adapter_checkpoint": getattr(
            owner.cfg, "resolved_adapter_checkpoint", owner.cfg.adapter_checkpoint
        ),
    }


def build_eval_artifact_paths(*, run_dir: Path, eval_dir: Path) -> Dict[str, Path]:
    return {
        "gt_vs_pred_guarded_jsonl": run_dir / "gt_vs_pred_guarded.jsonl",
        "gt_vs_pred_scored_guarded_jsonl": run_dir / "gt_vs_pred_scored_guarded.jsonl",
        "metrics_json": eval_dir / "metrics.json",
        "metrics_guarded_json": eval_dir / "metrics_guarded.json",
        "duplicate_guard_report_json": eval_dir / "duplicate_guard_report.json",
    }


def resolve_infer_artifact_paths(
    *,
    cfg: Any,
    backend: str,
) -> tuple[Path, Path, Optional[Path]]:
    out_path = Path(str(cfg.out_path))
    summary_path = Path(str(cfg.summary_path or (out_path.parent / "summary.json")))
    trace_path_raw = str(cfg.pred_token_trace_path or "").strip()
    if trace_path_raw:
        trace_path: Optional[Path] = Path(trace_path_raw)
    elif backend == "hf":
        trace_path = out_path.parent / "pred_token_trace.jsonl"
    else:
        trace_path = None
    return out_path, summary_path, trace_path


def ensure_infer_artifact_dirs(
    *,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path is not None:
        trace_path.parent.mkdir(parents=True, exist_ok=True)


def _generation_meta(owner: Any, *, backend: str, batch_size: int) -> Dict[str, Any]:
    return {
        "temperature": owner.gen_cfg.temperature,
        "top_p": owner.gen_cfg.top_p,
        "max_new_tokens": owner.gen_cfg.max_new_tokens,
        "repetition_penalty": owner.gen_cfg.repetition_penalty,
        "batch_size": batch_size,
        "seed": owner.gen_cfg.seed,
        "qwen_chat_generation": _qwen_chat_generation_meta(owner),
    }


def _qwen_chat_generation_meta(owner: Any) -> Dict[str, Any]:
    from src.common.detection_sequence import END_OF_TEXT_TOKEN, IM_END_TOKEN
    from src.common.qwen_generation import resolve_qwen_chat_generation_token_ids

    token_ids = getattr(owner, "qwen_generation_token_ids", None)
    if token_ids is None:
        tokenizer = getattr(owner, "tokenizer", None)
        processor = getattr(owner, "processor", None)
        if tokenizer is None and processor is not None:
            tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None:
            try:
                token_ids = resolve_qwen_chat_generation_token_ids(tokenizer)
            except ValueError:
                token_ids = None

    eos_token_id = getattr(token_ids, "eos_token_id", None)
    pad_token_id = getattr(token_ids, "pad_token_id", None)
    return {
        "eos_token": IM_END_TOKEN,
        "eos_token_id": eos_token_id,
        "pad_token": END_OF_TEXT_TOKEN,
        "pad_token_id": pad_token_id,
        "stop_tokens": [IM_END_TOKEN],
        "processor_do_resize": False,
    }


@dataclass(frozen=True)
class InferArtifactFacts:
    """Resolved offline inference facts used for artifact metadata payloads."""

    mode: str
    requested_mode: str
    mode_reason: str
    backend: str
    model_checkpoint: Any
    adapter_checkpoint: Any
    checkpoint_meta: Dict[str, Any]
    gt_jsonl: Any
    pred_coord_mode: Any
    prompt_variant: Any
    bbox_format: Any
    detection_template_id: str
    detection_sequence_format: str
    object_field_order: Any
    object_ordering: Any
    parsing: Dict[str, Any]
    prompt_template_hash: Any
    device: Any
    limit: Any
    generation: Dict[str, Any]
    inference_provenance: Dict[str, Any]
    distributed: Optional[Dict[str, Any]]
    backend_cfg: Optional[Dict[str, Any]]
    attn_implementation_requested: Any
    attn_implementation_selected: Any


def resolve_infer_artifact_facts_from_owner(
    *,
    owner: Any,
    backend: str,
    batch_size: int,
) -> InferArtifactFacts:
    """Translate an offline inference owner into artifact metadata facts."""

    checkpoint_meta = _checkpoint_meta(owner)
    generation_meta = _generation_meta(owner, backend=backend, batch_size=batch_size)
    detection_template_id = str(
        getattr(owner, "detection_template_id", "stage1_json_pretty")
    )
    parsing = {
        "mode": str(getattr(owner, "parser_mode", "strict_expected")),
    }
    distributed: Optional[Dict[str, Any]] = None
    if bool(getattr(owner.cfg, "distributed_enabled", False)):
        distributed = {
            "enabled": True,
            "rank": int(getattr(owner.cfg, "rank", 0) or 0),
            "local_rank": int(getattr(owner.cfg, "local_rank", 0) or 0),
            "world_size": max(int(getattr(owner.cfg, "world_size", 1) or 1), 1),
            "merge_strategy": "ordinal_restore",
        }
    backend_cfg: Optional[Dict[str, Any]] = None
    if backend == "vllm":
        public_fields = {
            "mode",
            "base_url",
            "model",
            "timeout_s",
            "client_concurrency",
        }
        backend_raw = getattr(owner.cfg, "backend", None) or {}
        backend_cfg = {
            k: v
            for k, v in backend_raw.items()
            if str(k) in public_fields
        }
    return InferArtifactFacts(
        mode=owner.resolved_mode,
        requested_mode=owner.requested_mode,
        mode_reason=owner.mode_reason,
        backend=backend,
        model_checkpoint=owner.cfg.model_checkpoint,
        adapter_checkpoint=owner.cfg.adapter_checkpoint,
        checkpoint_meta=checkpoint_meta,
        gt_jsonl=owner.cfg.gt_jsonl,
        pred_coord_mode=owner.cfg.pred_coord_mode,
        prompt_variant=owner.prompt_variant,
        bbox_format=owner.bbox_format,
        detection_template_id=detection_template_id,
        detection_sequence_format=getattr(owner, "detection_sequence_format", "coordjson"),
        object_field_order=owner.object_field_order,
        object_ordering=owner.object_ordering,
        parsing=parsing,
        prompt_template_hash=owner.prompt_template_hash,
        device=owner.cfg.device,
        limit=owner.cfg.limit,
        generation=generation_meta,
        inference_provenance=_build_inference_provenance(
            owner=owner,
            backend=backend,
            batch_size=batch_size,
            checkpoint_meta=checkpoint_meta,
            generation_meta=generation_meta,
        ),
        distributed=distributed,
        backend_cfg=backend_cfg,
        attn_implementation_requested=getattr(
            owner,
            "attn_implementation_requested",
            None,
        ),
        attn_implementation_selected=getattr(
            owner,
            "attn_implementation_selected",
            None,
        ),
    )


def build_infer_resolved_meta_from_facts(
    *,
    facts: InferArtifactFacts,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> Dict[str, Any]:
    """Build resolved-config metadata from already-resolved artifact facts."""

    resolved_meta = {
        "mode": facts.mode,
        "mode_resolution_reason": facts.mode_reason,
        "backend": facts.backend,
        "model_checkpoint": facts.model_checkpoint,
        "adapter_checkpoint": facts.adapter_checkpoint,
        **facts.checkpoint_meta,
        "gt_jsonl": facts.gt_jsonl,
        "pred_coord_mode": facts.pred_coord_mode,
        "prompt_variant": facts.prompt_variant,
        "bbox_format": facts.bbox_format,
        "detection_template": {
            "id": facts.detection_template_id,
        },
        "detection_template_id": facts.detection_template_id,
        "detection_sequence_format": facts.detection_sequence_format,
        "object_field_order": facts.object_field_order,
        "object_ordering": facts.object_ordering,
        "parsing": dict(facts.parsing),
        "prompt_template_hash": facts.prompt_template_hash,
        "device": facts.device,
        "limit": facts.limit,
        "generation": dict(facts.generation),
        "inference_provenance": dict(facts.inference_provenance),
        "artifacts": {
            "gt_vs_pred_jsonl": str(out_path),
            "pred_token_trace_jsonl": str(trace_path) if trace_path is not None else None,
            "summary_json": str(summary_path),
        },
    }
    if facts.distributed is not None:
        resolved_meta["distributed"] = dict(facts.distributed)
    if facts.backend_cfg is not None:
        resolved_meta["backend_cfg"] = dict(facts.backend_cfg)
    return resolved_meta


def build_infer_resolved_meta(
    *,
    owner: Any,
    backend: str,
    batch_size: int,
    out_path: Path,
    summary_path: Path,
    trace_path: Optional[Path],
) -> Dict[str, Any]:
    return build_infer_resolved_meta_from_facts(
        facts=resolve_infer_artifact_facts_from_owner(
            owner=owner,
            backend=backend,
            batch_size=batch_size,
        ),
        out_path=out_path,
        summary_path=summary_path,
        trace_path=trace_path,
    )


def build_infer_summary_payload_from_facts(
    *,
    facts: InferArtifactFacts,
    counters: Any,
    determinism: str,
) -> Dict[str, Any]:
    """Build offline inference summary from already-resolved artifact facts."""

    summary_payload: Dict[str, Any] = {
        "mode": facts.mode,
        "determinism": determinism,
        **counters.to_summary(),
        "backend": {
            "type": facts.backend,
            "model_checkpoint": facts.model_checkpoint,
            "adapter_checkpoint": facts.adapter_checkpoint,
            **facts.checkpoint_meta,
        },
        "generation": dict(facts.generation),
        "inference_provenance": dict(facts.inference_provenance),
        "infer": {
            "gt_jsonl": facts.gt_jsonl,
            "pred_coord_mode": facts.pred_coord_mode,
            "prompt_variant": facts.prompt_variant,
            "bbox_format": facts.bbox_format,
            "detection_template": {
                "id": facts.detection_template_id,
            },
            "detection_template_id": facts.detection_template_id,
            "detection_sequence_format": facts.detection_sequence_format,
            "object_field_order": facts.object_field_order,
            "object_ordering": facts.object_ordering,
            "parsing": dict(facts.parsing),
            "prompt_template_hash": facts.prompt_template_hash,
            "device": facts.device,
            "limit": facts.limit,
        },
    }
    if facts.distributed is not None:
        summary_payload["distributed"] = dict(facts.distributed)

    if facts.requested_mode == "auto":
        summary_payload["mode_resolution_reason"] = facts.mode_reason

    if facts.backend == "hf":
        summary_payload["backend"]["attn_implementation_requested"] = (
            facts.attn_implementation_requested
        )
        summary_payload["backend"]["attn_implementation_selected"] = (
            facts.attn_implementation_selected
        )
    return summary_payload


def build_infer_summary_payload(
    *,
    owner: Any,
    counters: Any,
    backend: str,
    determinism: str,
    batch_size: int,
) -> Dict[str, Any]:
    return build_infer_summary_payload_from_facts(
        facts=resolve_infer_artifact_facts_from_owner(
            owner=owner,
            backend=backend,
            batch_size=batch_size,
        ),
        counters=counters,
        determinism=determinism,
    )


def write_infer_summary(*, summary_path: Path, summary_payload: Dict[str, Any]) -> None:
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
