from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from . import PHASE_ID, PROJECT_ID, RUN_ID, SCHEMA_VERSION
from .status import (
    CONSTRAINT_POLICY,
    DECODE_POLICY,
    REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
)

if TYPE_CHECKING:
    from .config import A32Config


ROLE_INFER_CONFIG_NAMES = {
    "fullobj_random_pure_ce_ckpt3668": (
        "fullobj_random_purece_ckpt3668_a3_2_rollout1024_greedy.yaml"
    ),
    "fullobj_sorted_pure_ce_ckpt3668": (
        "fullobj_sorted_purece_ckpt3668_a3_2_rollout1024_greedy.yaml"
    ),
}


def run_real_native_rollout(
    config: A32Config,
    *,
    allow_overwrite: bool = False,
    gpu_id: str | None = None,
) -> dict[str, Any]:
    """Run native free-text greedy rollout for configured A3.2 checkpoints."""

    roles = _checkpoint_roles(config)
    written: dict[str, dict[str, str]] = {}
    for role in roles:
        paths = _role_artifact_paths(config, role)
        _ensure_paths_can_write(
            (
                paths["gt_vs_pred_jsonl"],
                paths["pred_token_trace_jsonl"],
                paths["summary_json"],
            ),
            allow_overwrite=allow_overwrite,
        )
    for role in roles:
        role_config_path = materialize_role_infer_config(
            config,
            checkpoint_role=role,
            gpu_id=gpu_id,
        )
        from src.infer.pipeline import run_pipeline

        run_pipeline(config_path=role_config_path)
        paths = _role_artifact_paths(config, role)
        decorate_native_rollout_artifacts(
            paths["rollout_dir"],
            checkpoint_role=role,
            gpu_id=_gpu_id(gpu_id),
            native_prompt_ordering=str(config.checkpoints[role].training_ordering),
            template_contract={
                "detection_sequence_format": config.template_contract.detection_sequence_format,
                "coordinate_surface": config.template_contract.coordinate_surface,
                "bbox_format": config.template_contract.bbox_format,
                "row_separator": config.template_contract.row_separator,
            },
        )
        written[role] = {key: str(path) for key, path in paths.items()}
    return {
        "stage": "native_rollout",
        "runtime_kind": REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
        "checkpoint_roles": list(roles),
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
        "written": written,
    }


def materialize_role_infer_config(
    config: A32Config,
    *,
    checkpoint_role: str,
    gpu_id: str | None = None,
) -> Path:
    import yaml

    if checkpoint_role not in config.checkpoints:
        raise ValueError(f"unknown checkpoint role: {checkpoint_role}")
    raw = yaml.safe_load(_role_template_path(config, checkpoint_role).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("role infer template must be a mapping")
    cfg = deepcopy(dict(raw))
    rollout_dir = config.artifact_root / "rollout" / checkpoint_role
    cfg.setdefault("run", {})
    cfg["run"]["name"] = f"{config.run_id}_{checkpoint_role}"
    cfg["run"]["output_dir"] = str(config.artifact_root / "rollout")
    cfg.setdefault("artifacts", {})
    cfg["artifacts"].update(
        {
            "run_dir": str(rollout_dir),
            "gt_vs_pred_jsonl": str(rollout_dir / "gt_vs_pred.jsonl"),
            "pred_token_trace_jsonl": str(rollout_dir / "pred_token_trace.jsonl"),
            "summary_json": str(rollout_dir / "summary.json"),
        }
    )
    cfg.setdefault("a3_2_launch_spec", {})
    cfg["a3_2_launch_spec"].update(
        {
            "checkpoint_role": checkpoint_role,
            "checkpoint_path": str(config.checkpoints[checkpoint_role].checkpoint_path),
            "output_root": str(rollout_dir),
            "decode_policy": DECODE_POLICY,
            "constraint_policy": CONSTRAINT_POLICY,
            "native_prompt_ordering": str(config.checkpoints[checkpoint_role].training_ordering),
            "image_limit": int(config.rollout.limit_images),
            "source_jsonl": str(config.val_jsonl),
            "image_root": str(config.image_root),
            "template_contract": {
                "detection_sequence_format": config.template_contract.detection_sequence_format,
                "coordinate_surface": config.template_contract.coordinate_surface,
                "bbox_format": config.template_contract.bbox_format,
                "row_separator": config.template_contract.row_separator,
                "compact_full_parse_mode": "marker_delimited_strict",
            },
        }
    )
    infer = cfg.setdefault("infer", {})
    infer.update(
        {
            "gt_jsonl": str(config.val_jsonl),
            "model_checkpoint": str(config.checkpoints[checkpoint_role].checkpoint_path),
            "detection_sequence_format": "compact_full",
            "row_separator": "none",
            "allow_diagnostic_gt_vs_pred": True,
            "bbox_format": "xyxy",
            "object_field_order": "desc_first",
            "object_ordering": (
                "random"
                if config.checkpoints[checkpoint_role].training_ordering.startswith("random")
                else "sorted"
            ),
            "limit": int(config.rollout.limit_images),
            "device": "cuda:0" if _gpu_id(gpu_id) != "unknown" else infer.get("device", "cuda:0"),
        }
    )
    infer.setdefault("generation", {})
    infer["generation"].update(
        {
            "decode_mode": "greedy",
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "num_beams": 1,
            "max_new_tokens": int(infer["generation"].get("max_new_tokens", 3084)),
            "repetition_penalty": 1.0,
            "trace_logprobs": True,
        }
    )
    infer["generation"].pop("compact_grammar", None)
    cfg.setdefault("metadata", {})
    cfg["metadata"].update(
        {
            "project_id": PROJECT_ID,
            "phase_id": PHASE_ID,
            "schema_version": SCHEMA_VERSION,
            "run_id": str(config.run_id),
            "runtime_kind": REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
        }
    )
    out_path = config.artifact_root / "runtime_configs" / f"{checkpoint_role}_infer.yaml"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return out_path


def decorate_native_rollout_artifacts(
    rollout_dir: str | Path,
    *,
    checkpoint_role: str,
    gpu_id: str,
    native_prompt_ordering: str,
    template_contract: Mapping[str, Any],
) -> dict[str, str]:
    rollout_path = Path(rollout_dir)
    summary_path = rollout_path / "summary.json"
    gt_path = rollout_path / "gt_vs_pred.jsonl"
    trace_path = rollout_path / "pred_token_trace.jsonl"
    summary = _read_json(summary_path)
    checkpoint_fingerprint = _checkpoint_fingerprint_from_summary(summary)
    provenance = {
        "runtime_kind": REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
        "checkpoint_role": checkpoint_role,
        "gpu_id": str(gpu_id),
        "checkpoint_fingerprint": checkpoint_fingerprint,
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
        "native_prompt_ordering": str(native_prompt_ordering),
        "template_contract": dict(template_contract),
    }
    summary.update(provenance)
    _write_json(summary_path, summary)
    _rewrite_jsonl_with_provenance(
        gt_path,
        provenance,
        add_source_line_idx=True,
        require_trace_hash=False,
    )
    _rewrite_jsonl_with_provenance(
        trace_path,
        provenance,
        add_source_line_idx=True,
        require_trace_hash=True,
    )
    return {
        "summary_json": str(summary_path),
        "gt_vs_pred_jsonl": str(gt_path),
        "pred_token_trace_jsonl": str(trace_path),
    }


def _rewrite_jsonl_with_provenance(
    path: Path,
    provenance: Mapping[str, Any],
    *,
    add_source_line_idx: bool,
    require_trace_hash: bool,
) -> None:
    rows = _read_jsonl(path)
    out_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        updated = {**row, **provenance}
        if add_source_line_idx and updated.get("source_line_idx") is None:
            updated["source_line_idx"] = int(row.get("line_idx", index))
        if require_trace_hash:
            _ensure_trace_hashes(updated)
        out_rows.append(updated)
    _write_jsonl(path, out_rows)


def _ensure_trace_hashes(row: dict[str, Any]) -> None:
    if not row.get("token_trace_sha256"):
        payload = {
            "generated_token_text": row.get("generated_token_text", []),
            "token_logprobs": row.get("token_logprobs", []),
        }
        row["token_trace_sha256"] = _sha256_json(payload)
    if not row.get("raw_output_sha256"):
        row["raw_output_sha256"] = row["token_trace_sha256"]


def _checkpoint_fingerprint_from_summary(summary: Mapping[str, Any]) -> str:
    provenance = summary.get("inference_provenance")
    if isinstance(provenance, Mapping):
        value = provenance.get("model_identity_fingerprint")
        if isinstance(value, str) and value:
            return value
    value = summary.get("model_identity_fingerprint")
    if isinstance(value, str) and value:
        return value
    return "checkpoint:" + _sha256_json(summary)


def _role_artifact_paths(config: A32Config, role: str) -> dict[str, Path]:
    rollout_dir = config.artifact_root / "rollout" / role
    return {
        "rollout_dir": rollout_dir,
        "gt_vs_pred_jsonl": rollout_dir / "gt_vs_pred.jsonl",
        "pred_token_trace_jsonl": rollout_dir / "pred_token_trace.jsonl",
        "summary_json": rollout_dir / "summary.json",
    }


def _role_template_path(config: A32Config, role: str) -> Path:
    template_role = _template_role(config, role)
    return (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "infer"
        / "recursive_detection_ce"
        / ROLE_INFER_CONFIG_NAMES[template_role]
    )


def _template_role(config: A32Config, role: str) -> str:
    if role in ROLE_INFER_CONFIG_NAMES:
        return role
    if role not in config.checkpoints:
        raise ValueError(f"unknown checkpoint role: {role}")
    ordering = str(config.checkpoints[role].training_ordering)
    if ordering.startswith("random"):
        return "fullobj_random_pure_ce_ckpt3668"
    if ordering.startswith("sorted"):
        return "fullobj_sorted_pure_ce_ckpt3668"
    raise ValueError(
        f"no native rollout infer template for role {role!r} "
        f"with training_ordering={ordering!r}"
    )


def _checkpoint_roles(config: A32Config) -> tuple[str, ...]:
    return tuple(str(role) for role in config.checkpoints)


def _ensure_paths_can_write(paths: Sequence[Path], *, allow_overwrite: bool) -> None:
    for path in paths:
        if Path(path).exists() and not allow_overwrite:
            raise FileExistsError(f"refusing to overwrite existing artifact: {path}")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"JSONL row must be an object: {path}")
                rows.append(row)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True))
            handle.write("\n")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _gpu_id(gpu_id: str | None) -> str:
    if gpu_id is not None and str(gpu_id).strip():
        return str(gpu_id)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is not None and visible.strip():
        return visible.strip()
    return "unknown"


__all__ = [
    "ROLE_INFER_CONFIG_NAMES",
    "decorate_native_rollout_artifacts",
    "materialize_role_infer_config",
    "run_real_native_rollout",
]
