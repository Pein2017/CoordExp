#!/usr/bin/env python3
"""Reusable CoordExp inference/evaluation helpers.

The script intentionally lives inside the skill bundle: it captures repeated
operator work such as creating infer/eval configs, writing tmux-safe launchers,
checking completion, and summarizing comparable metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_OUTPUT_ROOT = Path("/data/CoordExp/outputs/infer/recursive_detection_ce_latest")
DEFAULT_TEMP_DIR = Path("/data/CoordExp/temp/infer/recursive_detection_ce_latest")
DEFAULT_GT_JSONL = Path("public_data/coco/rescale_32_1024_bbox_max60/val.coord.jsonl")
DEFAULT_SEMANTIC_MODEL = Path("model_cache/all-MiniLM-L6-v2-local")
DEFAULT_PYTHON = Path("/root/miniconda3/envs/ms/bin/python")
DEFAULT_TEMPERATURE = 0.0
DEFAULT_REPETITION_PENALTY = "1.10"


def _rp_slug(value: str) -> str:
    """filesystem-safe repetition-penalty label."""

    text = str(value).strip()
    return text.replace(".", "p").replace("-", "m")


def _sanitize_slug(value: str) -> str:
    """stable slug for names used in files and tmux sessions."""

    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        raise ValueError("slug cannot be empty")
    return cleaned


def _checkpoint_label(checkpoint: Path) -> str:
    """compact checkpoint label from a checkpoint path."""

    name = checkpoint.name
    match = re.fullmatch(r"checkpoint-(\d+)", name)
    if match:
        return f"ckpt{match.group(1)}"
    return _sanitize_slug(name)


def _gpu_count(gpus: str) -> int:
    """number of visible CUDA devices in a CUDA_VISIBLE_DEVICES string."""

    parts = [item.strip() for item in gpus.split(",") if item.strip()]
    if not parts:
        raise ValueError("--gpus must contain at least one device id")
    return len(parts)


def _gpu_compact(gpus: str) -> str:
    """compact GPU label for launcher file names."""

    return "".join(item.strip() for item in gpus.split(",") if item.strip())


def _json_load(path: Path) -> dict[str, Any]:
    """JSON object loaded from a path."""

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"expected JSON object in {path}")
    return data


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """YAML file written without external dependencies."""

    text = "\n".join(_dump_yaml_lines(payload, indent=0)) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dump_yaml_lines(value: Any, *, indent: int) -> list[str]:
    """minimal YAML emitter for the config shapes produced by this helper."""

    space = " " * indent
    if isinstance(value, dict):
        lines: list[str] = []
        for key, child in value.items():
            if isinstance(child, (dict, list)):
                lines.append(f"{space}{key}:")
                lines.extend(_dump_yaml_lines(child, indent=indent + 2))
            else:
                lines.append(f"{space}{key}: {_yaml_scalar(child)}")
        return lines
    if isinstance(value, list):
        if not value:
            return [f"{space}[]"]
        lines = []
        for child in value:
            if isinstance(child, (dict, list)):
                lines.append(f"{space}-")
                lines.extend(_dump_yaml_lines(child, indent=indent + 2))
            else:
                lines.append(f"{space}- {_yaml_scalar(child)}")
        return lines
    return [f"{space}{_yaml_scalar(value)}"]


def _yaml_scalar(value: Any) -> str:
    """YAML scalar for primitive Python values."""

    if value is True:
        return "true"
    if value is False:
        return "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def _line_count(path: Path) -> int | None:
    """line count for an artifact, or None when missing."""

    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _format_float(value: Any) -> str:
    """human-friendly metric float."""

    if isinstance(value, (int, float)):
        return f"{float(value):.4f}"
    return "NA"


def _extract_nested(mapping: dict[str, Any], path: Sequence[str]) -> Any:
    """nested dictionary lookup with None for missing keys."""

    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


@dataclass(frozen=True)
class RecursiveInferEvalSpec:
    """single recursive-detection infer/eval launch specification."""

    repo_root: Path
    checkpoint: Path
    run_tag: str
    gpus: str
    master_port: int
    repetition_penalty: str = DEFAULT_REPETITION_PENALTY
    output_root: Path = DEFAULT_OUTPUT_ROOT
    temp_dir: Path = DEFAULT_TEMP_DIR
    gt_jsonl: Path = DEFAULT_GT_JSONL
    python: Path = DEFAULT_PYTHON
    run_prefix: str = "compact_full_prefix_rollin_balance2"
    prompt_variant: str = "coco_80"
    detection_sequence_format: str = "compact_full"
    bbox_format: str = "xyxy"
    object_field_order: str = "desc_first"
    object_ordering: str = "random"
    mode: str = "coord"
    pred_coord_mode: str = "auto"
    temperature: float = DEFAULT_TEMPERATURE
    max_new_tokens: int = 3084
    batch_size: int = 8
    seed: int = 42
    limit: int = 200
    detect_samples: int = 128
    semantic_model: Path = DEFAULT_SEMANTIC_MODEL
    semantic_device: str = "cuda:0"
    num_workers: int = 8
    force: bool = False

    @property
    def ngpu(self) -> int:
        """number of visible GPUs."""

        return _gpu_count(self.gpus)

    @property
    def rp_slug(self) -> str:
        """filesystem-safe repetition-penalty label."""

        return _rp_slug(self.repetition_penalty)

    @property
    def checkpoint_label(self) -> str:
        """compact checkpoint label."""

        return _checkpoint_label(self.checkpoint)

    @property
    def run_name(self) -> str:
        """pipeline run name."""

        tag = _sanitize_slug(self.run_tag)
        return (
            f"{self.run_prefix}_{tag}_{self.checkpoint_label}_val{self.limit}"
            f"_bsz{self.batch_size}_temp0_rp{self.rp_slug}"
            f"_max{self.max_new_tokens}_chatfix_{self.ngpu}gpu"
        )

    @property
    def output_dir(self) -> Path:
        """expected pipeline output directory."""

        return self.output_root / self.run_name

    @property
    def config_path(self) -> Path:
        """generated infer/eval config path."""

        return self.temp_dir / f"{self.run_name}.yaml"

    @property
    def launcher_path(self) -> Path:
        """generated shell launcher path."""

        gpu_label = _gpu_compact(self.gpus)
        return self.temp_dir / f"run_{_sanitize_slug(self.run_tag)}_{self.checkpoint_label}_rp{self.rp_slug}_{gpu_label}.sh"

    @property
    def tmux_session(self) -> str:
        """default tmux session name."""

        gpu_label = _gpu_compact(self.gpus)
        return f"{_sanitize_slug(self.run_tag)}_eval_rp{self.rp_slug}_{gpu_label}"

    @property
    def log_path(self) -> Path:
        """launcher log path."""

        return self.output_root / f"{_sanitize_slug(self.run_tag)}_{self.checkpoint_label}_rp{self.rp_slug}_{self.ngpu}gpu.log"

    def validate(self) -> None:
        """fail fast on unsafe or likely-wrong launch inputs."""

        if not self.repo_root.exists():
            raise FileNotFoundError(f"repo root does not exist: {self.repo_root}")
        if not self.checkpoint.exists():
            raise FileNotFoundError(f"checkpoint does not exist: {self.checkpoint}")
        if not (self.repo_root / self.gt_jsonl).exists() and not self.gt_jsonl.is_absolute():
            raise FileNotFoundError(f"gt_jsonl does not resolve from repo root: {self.gt_jsonl}")
        if self.output_dir.exists() and any(self.output_dir.iterdir()) and not self.force:
            raise FileExistsError(f"output directory exists; pass --force to reuse: {self.output_dir}")

    def build_config(self) -> dict[str, Any]:
        """pipeline YAML payload."""

        return {
            "run": {
                "name": self.run_name,
                "output_dir": str(self.output_root),
            },
            "stages": {
                "infer": True,
                "eval": True,
                "vis": False,
            },
            "infer": {
                "gt_jsonl": str(self.gt_jsonl),
                "model_checkpoint": str(self.checkpoint),
                "prompt_variant": self.prompt_variant,
                "detection_sequence_format": self.detection_sequence_format,
                "bbox_format": self.bbox_format,
                "object_field_order": self.object_field_order,
                "object_ordering": self.object_ordering,
                "mode": self.mode,
                "pred_coord_mode": self.pred_coord_mode,
                "backend": {
                    "type": "hf",
                    "attn_implementation": "flash_attention_2",
                },
                "generation": {
                    "temperature": float(self.temperature),
                    "top_p": 0.9,
                    "max_new_tokens": self.max_new_tokens,
                    "repetition_penalty": float(self.repetition_penalty),
                    "batch_size": self.batch_size,
                    "seed": self.seed,
                    "compact_grammar": {
                        "enabled": True,
                        "force_row_start": True,
                    },
                },
                "device": "cuda:0",
                "limit": self.limit,
                "detect_samples": self.detect_samples,
            },
            "confidence": {
                "fusion_w_geom": 1.0,
                "fusion_w_desc": 0.0,
                "desc_span_policy": "best_effort",
                "empty_desc_policy": "geom_only",
            },
            "eval": {
                "metrics": "both",
                "strict_parse": False,
                "use_segm": False,
                "lvis_max_dets": 300,
                "overlay": False,
                "overlay_k": 12,
                "num_workers": self.num_workers,
                "semantic_model": str(self.semantic_model),
                "semantic_threshold": 0.5,
                "semantic_device": self.semantic_device,
                "semantic_batch_size": 64,
                "f1ish_iou_thrs": [0.3, 0.5],
                "f1ish_pred_scope": "annotated",
                "duplicate_control": {
                    "enabled": True,
                },
            },
        }

    def build_launcher(self) -> str:
        """tmux-safe shell launcher."""

        repo_root = shlex.quote(str(self.repo_root))
        log_path = shlex.quote(str(self.log_path))
        python = shlex.quote(str(self.python))
        config = shlex.quote(str(self.config_path))
        gpus = shlex.quote(self.gpus)
        master_port = shlex.quote(str(self.master_port))
        run_tag = shlex.quote(self.run_tag)
        rp = shlex.quote(self.repetition_penalty)

        return f"""#!/usr/bin/env bash
set -euo pipefail

cd {repo_root}

export CUDA_VISIBLE_DEVICES={gpus}
export MASTER_ADDR=127.0.0.1
export MASTER_PORT={master_port}
export PYTHONPATH={repo_root}
export TOKENIZERS_PARALLELISM=false

LOG={log_path}
mkdir -p "$(dirname "$LOG")"

echo "[$(date -Is)] {run_tag} rp={rp} infer/eval start" | tee -a "$LOG"
{python} -m torch.distributed.run \\
  --nproc_per_node={self.ngpu} \\
  --master_addr=127.0.0.1 \\
  --master_port={master_port} \\
  scripts/run_infer.py \\
  --config {config} \\
  2>&1 | tee -a "$LOG"
echo "[$(date -Is)] {run_tag} rp={rp} infer/eval finished" | tee -a "$LOG"
"""


class RecursiveRunPreparer:
    """writer for recursive-detection infer/eval configs and launchers."""

    def __init__(self, spec: RecursiveInferEvalSpec) -> None:
        self._spec = spec

    def prepare(self, *, launch: bool = False, dry_run: bool = False) -> dict[str, str]:
        """write files and optionally launch the generated tmux session."""

        self._spec.validate()
        plan = {
            "config": str(self._spec.config_path),
            "launcher": str(self._spec.launcher_path),
            "output_dir": str(self._spec.output_dir),
            "log": str(self._spec.log_path),
            "tmux_session": self._spec.tmux_session,
            "tmux_command": f"tmux new-session -d -s {shlex.quote(self._spec.tmux_session)} bash {shlex.quote(str(self._spec.launcher_path))}",
        }

        if dry_run:
            return plan

        _write_yaml(self._spec.config_path, self._spec.build_config())
        self._spec.launcher_path.parent.mkdir(parents=True, exist_ok=True)
        self._spec.launcher_path.write_text(self._spec.build_launcher(), encoding="utf-8")
        self._spec.launcher_path.chmod(0o755)

        if launch:
            subprocess.run(
                ["tmux", "new-session", "-d", "-s", self._spec.tmux_session, "bash", str(self._spec.launcher_path)],
                check=True,
            )

        return plan


class RunArtifactSummary:
    """summarizer for completed CoordExp infer/eval run directories."""

    def __init__(self, run_dir: Path) -> None:
        self._run_dir = run_dir

    def as_row(self) -> dict[str, Any]:
        """compact metric/status row from artifacts."""

        metrics = self._read_metrics("metrics.json")
        guarded = self._read_metrics("metrics_guarded.json")
        confidence = self._read_optional_json("confidence_postop_summary.json")
        duplicate = self._read_optional_json("eval/duplicate_guard_report.json")

        return {
            "run": self._run_dir.name,
            "status": self._status(metrics, guarded),
            "rows": _line_count(self._run_dir / "gt_vs_pred.jsonl"),
            "raw_AP": _extract_nested(metrics, ["metrics", "bbox_AP"]),
            "raw_AP50": _extract_nested(metrics, ["metrics", "bbox_AP50"]),
            "raw_AP75": _extract_nested(metrics, ["metrics", "bbox_AP75"]),
            "raw_f1_50": _extract_nested(metrics, ["metrics", "f1ish@0.50_f1_full_micro"]),
            "guard_AP": _extract_nested(guarded, ["metrics", "bbox_AP"]),
            "guard_AP50": _extract_nested(guarded, ["metrics", "bbox_AP50"]),
            "guard_AP75": _extract_nested(guarded, ["metrics", "bbox_AP75"]),
            "guard_f1_50": _extract_nested(guarded, ["metrics", "f1ish@0.50_f1_full_micro"]),
            "pred_total": confidence.get("total_pred_objects"),
            "pred_kept": confidence.get("kept_pred_objects"),
            "pred_dropped": confidence.get("dropped_pred_objects"),
            "eval_pred": _extract_nested(metrics, ["metrics", "f1ish@0.50_pred_total"]),
            "degenerate": _extract_nested(metrics, ["counters", "degenerate"]),
            "invalid_geometry": _extract_nested(metrics, ["counters", "invalid_geometry"]),
            "dup_suppressed": duplicate.get("total_predictions_suppressed"),
            "guard_records": duplicate.get("total_guarded_records_affected"),
            "path": str(self._run_dir),
        }

    def _read_optional_json(self, relative_path: str) -> dict[str, Any]:
        """optional JSON artifact."""

        path = self._run_dir / relative_path
        if not path.exists():
            return {}
        return _json_load(path)

    def _read_metrics(self, name: str) -> dict[str, Any]:
        """optional metric artifact under eval/."""

        return self._read_optional_json(f"eval/{name}")

    def _status(self, metrics: dict[str, Any], guarded: dict[str, Any]) -> str:
        """coarse artifact completion status."""

        if metrics and guarded:
            return "evaluated"
        if (self._run_dir / "gt_vs_pred_scored.jsonl").exists():
            return "scored"
        if (self._run_dir / "gt_vs_pred.jsonl").exists():
            return "inferred"
        return "missing"


def _print_markdown(rows: Iterable[dict[str, Any]]) -> None:
    """markdown table output."""

    fields = [
        ("run", "run"),
        ("status", "status"),
        ("rows", "rows"),
        ("raw_AP", "raw AP"),
        ("raw_AP50", "raw AP50"),
        ("raw_AP75", "raw AP75"),
        ("raw_f1_50", "raw F1@0.50"),
        ("guard_AP", "guard AP"),
        ("guard_AP50", "guard AP50"),
        ("guard_f1_50", "guard F1@0.50"),
        ("pred_total", "pred"),
        ("pred_kept", "kept"),
        ("pred_dropped", "drop"),
        ("degenerate", "degen"),
        ("dup_suppressed", "dup supp"),
    ]
    rows = list(rows)
    print("| " + " | ".join(title for _, title in fields) + " |")
    print("|" + "|".join("---" for _ in fields) + "|")
    for row in rows:
        values: list[str] = []
        for key, _ in fields:
            value = row.get(key)
            if key.endswith("AP") or key.endswith("AP50") or key.endswith("AP75") or key.endswith("f1_50"):
                values.append(_format_float(value))
            elif value is None:
                values.append("NA")
            else:
                values.append(str(value))
        print("| " + " | ".join(values) + " |")


def _collect_run_dirs(paths: Sequence[str], *, glob_pattern: str | None) -> list[Path]:
    """run directories from explicit paths and optional glob pattern."""

    run_dirs = [Path(item).expanduser().resolve() for item in paths]
    if glob_pattern:
        run_dirs.extend(path for path in sorted(Path("/").glob(glob_pattern.lstrip("/"))) if path.is_dir())
    unique: dict[str, Path] = {}
    for path in run_dirs:
        unique[str(path)] = path
    return list(unique.values())


def _cmd_prepare_recursive(args: argparse.Namespace) -> int:
    """prepare recursive-detection config and launcher."""

    spec = RecursiveInferEvalSpec(
        repo_root=Path(args.repo_root).resolve(),
        checkpoint=Path(args.checkpoint).resolve(),
        run_tag=args.run_tag,
        gpus=args.gpus,
        master_port=args.master_port,
        output_root=Path(args.output_root).resolve(),
        temp_dir=Path(args.temp_dir).resolve(),
        gt_jsonl=Path(args.gt_jsonl),
        python=Path(args.python).resolve(),
        repetition_penalty=args.rp,
        run_prefix=args.run_prefix,
        object_ordering=args.object_ordering,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
        detect_samples=args.detect_samples,
        semantic_model=Path(args.semantic_model),
        semantic_device=args.semantic_device,
        force=args.force,
    )
    plan = RecursiveRunPreparer(spec).prepare(launch=args.launch, dry_run=args.dry_run)
    print(json.dumps(plan, indent=2))
    return 0


def _cmd_summarize(args: argparse.Namespace) -> int:
    """summarize completed or partial run directories."""

    run_dirs = _collect_run_dirs(args.run_dirs, glob_pattern=args.glob)
    if not run_dirs:
        raise SystemExit("no run directories supplied")
    rows = [RunArtifactSummary(path).as_row() for path in run_dirs]

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        _print_markdown(rows)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    """command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare-recursive",
        help="write a recursive-detection infer/eval YAML and tmux-safe launcher",
    )
    prepare.add_argument("--repo-root", required=True, help="CoordExp repo/worktree root to launch from")
    prepare.add_argument("--checkpoint", required=True, help="checkpoint or adapter-shorthand path")
    prepare.add_argument("--run-tag", required=True, help="short run tag, e.g. a3 or a4_eos")
    prepare.add_argument(
        "--rp",
        default=DEFAULT_REPETITION_PENALTY,
        help=f"generation repetition penalty; default: {DEFAULT_REPETITION_PENALTY}",
    )
    prepare.add_argument("--gpus", required=True, help="CUDA_VISIBLE_DEVICES list, e.g. 0,1,2,3")
    prepare.add_argument("--master-port", required=True, type=int, help="distributed launch port")
    prepare.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    prepare.add_argument("--temp-dir", default=str(DEFAULT_TEMP_DIR))
    prepare.add_argument("--gt-jsonl", default=str(DEFAULT_GT_JSONL))
    prepare.add_argument("--python", default=str(DEFAULT_PYTHON), help="python executable for launcher")
    prepare.add_argument("--run-prefix", default="compact_full_prefix_rollin_balance2")
    prepare.add_argument("--object-ordering", default="random")
    prepare.add_argument("--max-new-tokens", default=3084, type=int)
    prepare.add_argument("--batch-size", default=8, type=int)
    prepare.add_argument("--seed", default=42, type=int)
    prepare.add_argument("--limit", default=200, type=int)
    prepare.add_argument("--detect-samples", default=128, type=int)
    prepare.add_argument("--semantic-model", default=str(DEFAULT_SEMANTIC_MODEL))
    prepare.add_argument("--semantic-device", default="cuda:0")
    prepare.add_argument("--force", action="store_true", help="allow existing nonempty output directory")
    prepare.add_argument("--dry-run", action="store_true", help="print planned paths without writing files")
    prepare.add_argument("--launch", action="store_true", help="start tmux session after writing files")
    prepare.set_defaults(func=_cmd_prepare_recursive)

    summarize = subparsers.add_parser(
        "summarize",
        help="summarize CoordExp infer/eval run directories from artifacts",
    )
    summarize.add_argument("run_dirs", nargs="*", help="run directories to summarize")
    summarize.add_argument("--glob", help="absolute glob pattern for run directories")
    summarize.add_argument("--format", choices=["markdown", "json"], default="markdown")
    summarize.set_defaults(func=_cmd_summarize)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """entrypoint."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
