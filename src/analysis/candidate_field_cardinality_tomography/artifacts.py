from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CORE_JSONL = (
    "case_index.jsonl",
    "probe_plan.jsonl",
    "x1_candidate_field_rows.jsonl",
    "residual_row_score_rows.jsonl",
    "basin_attraction_rows.jsonl",
    "attention_component_rows.jsonl",
    "phase_a_case_taxonomy_rows.jsonl",
)
SUPPORT_FILES = (
    "case_index_summary.json",
    "controls_summary.json",
    "summary.json",
    "report.md",
    "resolved_config.yaml",
    "manifest.json",
    "plots/plot_manifest.json",
    "gallery/gallery_rows.jsonl",
)

BASE_REQUIRED_FIELDS = (
    "schema_version",
    "project_id",
    "phase_id",
    "run_id",
    "checkpoint_id",
    "case_id",
    "case_index_row_id",
    "split",
    "pool_role",
    "source_dataset_jsonl",
    "dataset_manifest_id",
    "dataset_manifest_sha256",
    "fn_rescue_overlay_membership",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_required_fields(row: Mapping[str, Any], *, extra_fields: Sequence[str] = ()) -> None:
    for key in (*BASE_REQUIRED_FIELDS, *tuple(extra_fields)):
        if key not in row:
            raise ValueError(f"missing required field: {key}")


def validate_probe_row_against_plan(row: Mapping[str, Any], plan_row: Mapping[str, Any]) -> None:
    required = (
        "probe_plan_row_id",
        "sampling_policy_id",
        "sampling_policy_sha256",
        "strata_key",
        "shard_id",
    )
    validate_required_fields(row, extra_fields=required)
    if row["probe_plan_row_id"] != plan_row.get("probe_plan_row_id"):
        raise ValueError("probe_plan_row_id mismatch")
    if plan_row.get("probe_sampled") is not True:
        raise ValueError("probe row references unsampled plan row")
    comparisons = (
        ("sampling_policy_id", "sampling_policy_id"),
        ("sampling_policy_sha256", "sampling_policy_sha256"),
        ("strata_key", "strata_key"),
        ("shard_id", "planned_shard_id"),
    )
    for row_key, plan_key in comparisons:
        if row.get(row_key) != plan_row.get(plan_key):
            raise ValueError(f"{row_key} mismatch")


def write_manifest(root: Path, files: Sequence[Path], metadata: Mapping[str, Any]) -> dict[str, Any]:
    entries = []
    for path in files:
        rel = path.relative_to(root)
        entries.append({"path": str(rel), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    manifest = {"metadata": dict(metadata), "files": entries}
    write_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise ValueError("manifest files must be a list")
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ValueError("manifest file entry must be a mapping")
        rel_path = str(entry.get("path") or "")
        if not rel_path:
            raise ValueError("manifest file entry missing path")
        path = root / rel_path
        if not path.exists():
            raise FileNotFoundError(f"missing manifest file: {rel_path}")
        expected_bytes = int(entry.get("bytes", -1))
        if path.stat().st_size != expected_bytes:
            raise ValueError(f"bytes mismatch: {rel_path}")
        expected_sha = str(entry.get("sha256") or "")
        if sha256_file(path) != expected_sha:
            raise ValueError(f"sha256 mismatch: {rel_path}")
    return {
        "validation_status": "ok",
        "file_count": len(entries),
        "metadata": dict(manifest.get("metadata") or {}),
    }
