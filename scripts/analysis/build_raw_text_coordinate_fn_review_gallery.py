from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.raw_text_coordinate_fn_review import materialize_fn_review_gallery


@dataclass(frozen=True)
class InputConfig:
    selected_cases_jsonl: str
    per_case_margins_jsonl: str
    baseline_gt_vs_pred_jsonl: str


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str
    title: str


@dataclass(frozen=True)
class Config:
    input: InputConfig
    output: OutputConfig


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError("fn review config root must be a mapping")
    return payload


def _shared_repo_root(anchor: Path) -> Path:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=anchor.parent,
            check=True,
            capture_output=True,
            text=True,
        )
        common_dir = Path(completed.stdout.strip())
        return common_dir.parent if common_dir.name == ".git" else common_dir
    except (OSError, subprocess.CalledProcessError, ValueError):
        return anchor.parent


def _resolve_path(raw_path: str, *, anchor: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = anchor.parent / path
    if candidate.exists():
        return candidate
    return _shared_repo_root(anchor) / path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise TypeError(f"expected object rows in {path}")
            rows.append(payload)
    return rows


def load_config(config_path: Path) -> Config:
    raw = _load_yaml(config_path)
    input_raw = raw["input"]
    output_raw = raw["output"]
    if not isinstance(input_raw, dict) or not isinstance(output_raw, dict):
        raise TypeError("input/output must be mappings")
    return Config(
        input=InputConfig(
            selected_cases_jsonl=str(input_raw["selected_cases_jsonl"]),
            per_case_margins_jsonl=str(input_raw["per_case_margins_jsonl"]),
            baseline_gt_vs_pred_jsonl=str(input_raw["baseline_gt_vs_pred_jsonl"]),
        ),
        output=OutputConfig(
            output_dir=str(output_raw["output_dir"]),
            title=str(output_raw.get("title", "Raw-Text FN Review")),
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    cfg = load_config(config_path)
    selected_cases = _read_jsonl(_resolve_path(cfg.input.selected_cases_jsonl, anchor=config_path))
    margin_rows = _read_jsonl(_resolve_path(cfg.input.per_case_margins_jsonl, anchor=config_path))
    output_dir = _resolve_path(cfg.output.output_dir, anchor=config_path)
    baseline_gt_vs_pred_path = _resolve_path(
        cfg.input.baseline_gt_vs_pred_jsonl,
        anchor=config_path,
    )
    rows = materialize_fn_review_gallery(
        selected_cases=selected_cases,
        margin_rows=margin_rows,
        baseline_gt_vs_pred_path=baseline_gt_vs_pred_path,
        output_dir=output_dir,
        title=cfg.output.title,
    )
    print(
        json.dumps(
            {
                "review_dir": str(output_dir),
                "review_html": str(output_dir / "review.html"),
                "case_count": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
