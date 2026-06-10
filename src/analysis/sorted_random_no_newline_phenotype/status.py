from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Mapping

from .merge_report import (
    BANNED_CAUSAL_PHRASES,
    LEGACY_A31_LABELS,
    RANDOM_ROLE,
    SORTED_ROLE,
)


FINAL_STATUS = "final_artifacts_present"
INDEX_READY_STATUS = "index_ready_pending_gpu"
INCOMPLETE_STATUS = "incomplete"
DECODE_POLICY = "free_text_unconstrained_greedy_temp0"
CONSTRAINT_POLICY = "none"
REAL_PREFIX_RUNTIME_KIND = "real_gpu_prefix_readout_v1"
REAL_NATIVE_ROLLOUT_RUNTIME_KIND = "real_gpu_native_rollout_v1"
REAL_FN_HINT_RUNTIME_KIND = "real_gpu_fn_hint_probe_v1"

EXPECTED_ROLES = (RANDOM_ROLE, SORTED_ROLE)
EXPECTED_TEMPLATE_CONTRACT = {
    "detection_sequence_format": "compact_full",
    "coordinate_surface": "coord_token",
    "bbox_format": "xyxy",
    "row_separator": "none",
}

INDEX_READY_ARTIFACTS = (
    "data_root_audit.json",
    "prefix_state_index.jsonl",
    "prefix_state_sampled_rows.jsonl",
    "prefix_state_index_summary.json",
    "sample_manifest.json",
)
PREFIX_SHARD_ARTIFACTS = tuple(
    f"prefix_readout_shards/shard_{shard_id}.jsonl" for shard_id in range(8)
)
NATIVE_ROLLOUT_FILENAMES = (
    "gt_vs_pred.jsonl",
    "pred_token_trace.jsonl",
    "summary.json",
)
FINAL_ONLY_ARTIFACTS = (
    "prefix_state_shard_summaries.jsonl",
    "summary/prefix_readout_merged_rows.jsonl",
    "summary/prefix_readout_summary.json",
    "summary/report.md",
    "rollout/rollout_summary.json",
    "rollout/rollout_phenotype_rows.jsonl",
    "fn_probe/fn_case_universe.jsonl",
    "fn_probe/fn_cases.jsonl",
    "fn_probe/fn_probe_rows.jsonl",
    "fn_probe/fn_candidate_scores.jsonl",
    "fn_probe/fn_slot_evidence.jsonl",
    "fn_probe/fn_bucket_summary.json",
    "fn_probe/fn_prefix_sensitivity.json",
    "fn_probe/fn_slot_rescue_summary.json",
    "gallery/index.md",
    "gallery/metadata.json",
    "fn_probe/gallery/index.md",
    "fn_probe/gallery/metadata.json",
) + PREFIX_SHARD_ARTIFACTS
NATIVE_ROLLOUT_ARTIFACTS = tuple(
    f"rollout/{role}/{name}"
    for role in EXPECTED_ROLES
    for name in NATIVE_ROLLOUT_FILENAMES
)
REQUIRED_FINAL_ARTIFACTS = (
    INDEX_READY_ARTIFACTS + FINAL_ONLY_ARTIFACTS + NATIVE_ROLLOUT_ARTIFACTS
)
FN_SUMMARY_ARTIFACTS = (
    "fn_probe/fn_bucket_summary.json",
    "fn_probe/fn_prefix_sensitivity.json",
    "fn_probe/fn_slot_rescue_summary.json",
)
PLACEHOLDER_RUNTIME_STATUSES = {
    "planned_or_external_rollout_required",
    "rollout_inputs_missing",
}
PLACEHOLDER_RUNTIME_MARKERS = {
    "mocked_cpu_status_case",
}


def evaluate_status(artifact_root: str | Path) -> dict[str, Any]:
    root = Path(artifact_root)
    expected_roles = _expected_roles(root)
    final_only_artifacts = _final_only_artifacts(expected_roles)
    required_final_artifacts = INDEX_READY_ARTIFACTS + final_only_artifacts
    missing_index = _missing(root, INDEX_READY_ARTIFACTS)
    missing_final = _missing(root, required_final_artifacts)
    existing_final_only = [
        rel_path for rel_path in final_only_artifacts if (root / rel_path).exists()
    ]
    post_index_started = bool(existing_final_only)

    failed_gates: list[str] = []
    if missing_index:
        _append(failed_gates, "index_ready_artifacts_present")
    if not _data_root_audit_ok(root):
        _append(failed_gates, "data_root_audit_present")
    if not _template_contract_ok(root):
        _append(failed_gates, "template_contract_present")
    if _inprogress_leftovers(root):
        _append(failed_gates, "no_inprogress_leftovers")
    if not _no_legacy_labels(root):
        _append(failed_gates, "no_legacy_a31_labels")
    if not _checkpoint_roles_ok(root, expected_roles):
        _append(failed_gates, "checkpoint_roles_are_a3_2")
    if not _shard_and_merged_row_counts_match(root):
        _append(failed_gates, "shard_and_merged_row_counts_match")

    if post_index_started:
        if not _real_runtime_artifacts_ok(root):
            _append(failed_gates, "real_runtime_artifacts_present")
        if not _report_language_ok(root):
            _append(failed_gates, "report_language_is_cautious")
        if not _prefix_shard_surface_ok(root):
            _append(failed_gates, "prefix_shard_surface_complete")
        if not (root / "fn_probe" / "fn_case_universe.jsonl").is_file():
            _append(failed_gates, "fn_universe_present")
        if any(not (root / rel_path).is_file() for rel_path in FN_SUMMARY_ARTIFACTS):
            _append(failed_gates, "fn_probe_summaries_present")
        if not _gallery_images_present(root):
            _append(failed_gates, "gallery_images_present")

    final_artifacts_present = not missing_final and not failed_gates
    index_ready_pending_gpu = (
        not final_artifacts_present
        and not failed_gates
        and not missing_index
        and not post_index_started
    )

    if (
        missing_final
        and not index_ready_pending_gpu
        and post_index_started
    ):
        _append(failed_gates, "required_final_artifacts_present")
    if final_artifacts_present:
        status = FINAL_STATUS
    elif index_ready_pending_gpu:
        status = INDEX_READY_STATUS
    else:
        status = INCOMPLETE_STATUS

    return {
        "status": status,
        "final_artifacts_present": final_artifacts_present,
        "index_ready_pending_gpu": index_ready_pending_gpu,
        "failed_gates": failed_gates,
        "missing_index_artifacts": missing_index,
        "missing_final_artifacts": missing_final,
        "pending_final_artifacts": (
            missing_final if index_ready_pending_gpu else []
        ),
        "required_final_artifacts": list(required_final_artifacts),
    }


def _expected_roles(root: Path) -> tuple[str, ...]:
    manifest = _read_json(root / "sample_manifest.json")
    if isinstance(manifest, Mapping):
        roles = manifest.get("checkpoint_roles")
        parsed = _parse_role_sequence(roles)
        if parsed:
            return parsed
    return EXPECTED_ROLES


def _parse_role_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    roles = tuple(str(role) for role in value if str(role))
    if len(roles) != len(value) or len(set(roles)) != len(roles):
        return ()
    return roles


def _final_only_artifacts(expected_roles: tuple[str, ...]) -> tuple[str, ...]:
    native_rollout_artifacts = tuple(
        f"rollout/{role}/{name}"
        for role in expected_roles
        for name in NATIVE_ROLLOUT_FILENAMES
    )
    return FINAL_ONLY_ARTIFACTS + native_rollout_artifacts


def _missing(root: Path, rel_paths: tuple[str, ...]) -> list[str]:
    return [rel_path for rel_path in rel_paths if not (root / rel_path).exists()]


def _append(items: list[str], item: str) -> None:
    if item not in items:
        items.append(item)


def _data_root_audit_ok(root: Path) -> bool:
    path = root / "data_root_audit.json"
    if not path.is_file():
        return False
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return False
    return (
        payload.get("status") == "ok"
        and bool(payload.get("actual_train_jsonl"))
        and bool(payload.get("actual_val_jsonl"))
        and bool(payload.get("image_root"))
    )


def _template_contract_ok(root: Path) -> bool:
    manifest = _read_json(root / "sample_manifest.json")
    if not isinstance(manifest, Mapping):
        return False
    contract = manifest.get("template_contract")
    if not isinstance(contract, Mapping):
        return False
    return {
        str(key): str(value) for key, value in contract.items()
    } == EXPECTED_TEMPLATE_CONTRACT


def _inprogress_leftovers(root: Path) -> list[str]:
    if not root.exists():
        return []
    return [str(path.relative_to(root)) for path in root.rglob("*.inprogress")]


def _no_legacy_labels(root: Path) -> bool:
    if not root.exists():
        return True
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md", ".txt", ".yaml", ".yml"}:
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            for line in handle:
                if any(label in line for label in LEGACY_A31_LABELS):
                    return False
    return True


def _checkpoint_roles_ok(root: Path, expected_roles: tuple[str, ...]) -> bool:
    for payload in _iter_structured_payloads(root):
        if not _payload_roles_ok(payload, expected_roles):
            return False
    return True


def _payload_roles_ok(payload: Any, expected_roles: tuple[str, ...]) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_str = str(key)
            if key_str == "checkpoint_roles":
                roles = _parse_role_sequence(value)
                if not roles or any(role not in expected_roles for role in roles):
                    return False
            elif key_str == "checkpoint_role":
                if value is not None and str(value) not in expected_roles:
                    return False
            elif key_str == "role_a":
                if str(value) not in expected_roles:
                    return False
            elif key_str == "role_b":
                if str(value) not in expected_roles:
                    return False
            if not _payload_roles_ok(value, expected_roles):
                return False
    elif isinstance(payload, list | tuple):
        for item in payload:
            if not _payload_roles_ok(item, expected_roles):
                return False
    return True


def _shard_and_merged_row_counts_match(root: Path) -> bool:
    shard_path = root / "prefix_state_shard_summaries.jsonl"
    merged_path = root / "summary" / "prefix_readout_merged_rows.jsonl"
    if not shard_path.exists() and not merged_path.exists():
        return True
    if not shard_path.is_file() or not merged_path.is_file():
        return False
    expected = 0
    try:
        for row in _iter_jsonl_payloads(shard_path):
            if not isinstance(row, Mapping):
                return False
            try:
                expected += int(row["prefix_state_count"])
            except (KeyError, TypeError, ValueError):
                return False
        merged_count = 0
        for row in _iter_jsonl_payloads(merged_path):
            if not isinstance(row, Mapping):
                return False
            merged_count += 1
    except json.JSONDecodeError:
        return False
    return expected == merged_count


def _prefix_shard_surface_ok(root: Path) -> bool:
    summaries_path = root / "prefix_state_shard_summaries.jsonl"
    if not summaries_path.is_file():
        return False
    summary_shards: set[int] = set()
    try:
        for row in _iter_jsonl_payloads(summaries_path):
            if not isinstance(row, Mapping):
                return False
            try:
                shard_id = int(row["shard_id"])
                prefix_state_count = int(row["prefix_state_count"])
            except (KeyError, TypeError, ValueError):
                return False
            if shard_id < 0 or shard_id >= 8 or prefix_state_count < 0:
                return False
            summary_shards.add(shard_id)
    except json.JSONDecodeError:
        return False
    if summary_shards != set(range(8)):
        return False
    for shard_id in range(8):
        shard_path = root / "prefix_readout_shards" / f"shard_{shard_id}.jsonl"
        if not shard_path.is_file() or shard_path.stat().st_size <= 0:
            return False
        try:
            shard_rows = sum(1 for _ in _iter_jsonl_payloads(shard_path))
        except json.JSONDecodeError:
            return False
        if shard_rows <= 0:
            return False
    return True


def _report_language_ok(root: Path) -> bool:
    report_path = root / "summary" / "report.md"
    if not report_path.is_file():
        return False
    banned = tuple(phrase.lower() for phrase in BANNED_CAUSAL_PHRASES)
    with report_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            lowered = line.lower()
            if any(phrase in lowered for phrase in banned):
                return False
    return True


def _gallery_images_present(root: Path) -> bool:
    for gallery_root in (root / "gallery", root / "fn_probe" / "gallery"):
        if not (gallery_root / "index.md").is_file():
            return False
        metadata = _read_json(gallery_root / "metadata.json")
        if not isinstance(metadata, list) or not metadata:
            return False
        referenced_paths: set[Path] = set()
        for item in metadata:
            if not isinstance(item, Mapping):
                return False
            rel_image = item.get("relative_image_path")
            if not isinstance(rel_image, str) or not rel_image:
                return False
            image_path = _safe_gallery_relative_path(gallery_root, rel_image)
            if image_path is None:
                return False
            if not _valid_jpeg_file(image_path):
                return False
            referenced_paths.add(image_path)
        image_root = gallery_root / "images"
        if not image_root.is_dir():
            return False
        actual_images = {path for path in image_root.glob("*.jpg") if path.is_file()}
        if actual_images != referenced_paths:
            return False
    return True


def _real_runtime_artifacts_ok(root: Path) -> bool:
    if _structured_payload_has_placeholder_runtime(root):
        return False
    if not _real_prefix_runtime_ok(root):
        return False
    if not _real_native_rollout_runtime_ok(root):
        return False
    if not _real_fn_hint_runtime_ok(root):
        return False
    return True


def _real_prefix_runtime_ok(root: Path) -> bool:
    summaries_path = root / "prefix_state_shard_summaries.jsonl"
    if not summaries_path.is_file():
        return False
    seen_shards: set[int] = set()
    try:
        for row in _iter_jsonl_payloads(summaries_path):
            if not isinstance(row, Mapping):
                return False
            if str(row.get("runtime_kind")) != REAL_PREFIX_RUNTIME_KIND:
                return False
            shard_id = _coerce_shard_id(row)
            if shard_id is None:
                return False
            if not _nonempty(row.get("gpu_id")):
                return False
            seen_shards.add(shard_id)
    except json.JSONDecodeError:
        return False
    if seen_shards != set(range(8)):
        return False
    for shard_id in range(8):
        shard_path = root / "prefix_readout_shards" / f"shard_{shard_id}.jsonl"
        if not _jsonl_has_runtime_kind(
            shard_path,
            REAL_PREFIX_RUNTIME_KIND,
            shard_id=shard_id,
            require_gpu_id=True,
        ):
            return False
    return True


def _real_native_rollout_runtime_ok(root: Path) -> bool:
    for role in _expected_roles(root):
        rollout_dir = root / "rollout" / role
        if not _json_has_runtime_kind(
            rollout_dir / "summary.json",
            REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
            checkpoint_role=role,
            require_decode_policy=True,
            require_gpu_id=True,
            require_checkpoint_fingerprint=True,
        ):
            return False
        if not _jsonl_has_runtime_kind(
            rollout_dir / "gt_vs_pred.jsonl",
            REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
            checkpoint_role=role,
            require_decode_policy=True,
        ):
            return False
        if not _jsonl_has_runtime_kind(
            rollout_dir / "pred_token_trace.jsonl",
            REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
            checkpoint_role=role,
            require_decode_policy=True,
            require_trace_hash=True,
        ):
            return False
    rollout_rows = root / "rollout" / "rollout_phenotype_rows.jsonl"
    return _jsonl_has_runtime_kind(
        rollout_rows,
        REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
        runtime_kind_keys=("runtime_kind", "source_runtime_kind"),
    )


def _real_fn_hint_runtime_ok(root: Path) -> bool:
    fn_dir = root / "fn_probe"
    for rel_name in (
        "fn_probe_rows.jsonl",
        "fn_candidate_scores.jsonl",
        "fn_slot_evidence.jsonl",
    ):
        if not _jsonl_has_runtime_kind(
            fn_dir / rel_name,
            REAL_FN_HINT_RUNTIME_KIND,
            require_probe_runtime=True,
        ):
            return False
    for rel_name in (
        "fn_bucket_summary.json",
        "fn_prefix_sensitivity.json",
        "fn_slot_rescue_summary.json",
    ):
        if not _json_has_runtime_kind(
            fn_dir / rel_name,
            REAL_FN_HINT_RUNTIME_KIND,
            require_probe_runtime=True,
        ):
            return False
    return True


def _structured_payload_has_placeholder_runtime(root: Path) -> bool:
    for payload in _iter_structured_payloads(root):
        if _payload_has_placeholder_runtime(payload):
            return True
    return False


def _payload_has_placeholder_runtime(payload: Any) -> bool:
    if isinstance(payload, Mapping):
        for key, value in payload.items():
            key_str = str(key)
            if key_str == "mocked_runtime" and value is True:
                return True
            if key_str in {"runtime_status", "status", "sampling_reason"}:
                value_str = str(value)
                if value_str in PLACEHOLDER_RUNTIME_STATUSES:
                    return True
                if value_str in PLACEHOLDER_RUNTIME_MARKERS:
                    return True
            if _payload_has_placeholder_runtime(value):
                return True
    elif isinstance(payload, list | tuple):
        for item in payload:
            if _payload_has_placeholder_runtime(item):
                return True
    return False


def _json_has_runtime_kind(
    path: Path,
    runtime_kind: str,
    *,
    checkpoint_role: str | None = None,
    require_decode_policy: bool = False,
    require_gpu_id: bool = False,
    require_checkpoint_fingerprint: bool = False,
    require_probe_runtime: bool = False,
) -> bool:
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        return False
    return _runtime_payload_ok(
        payload,
        runtime_kind,
        checkpoint_role=checkpoint_role,
        require_decode_policy=require_decode_policy,
        require_gpu_id=require_gpu_id,
        require_checkpoint_fingerprint=require_checkpoint_fingerprint,
        require_probe_runtime=require_probe_runtime,
    )


def _jsonl_has_runtime_kind(
    path: Path,
    runtime_kind: str,
    *,
    runtime_kind_keys: tuple[str, ...] = ("runtime_kind",),
    shard_id: int | None = None,
    checkpoint_role: str | None = None,
    require_decode_policy: bool = False,
    require_gpu_id: bool = False,
    require_trace_hash: bool = False,
    require_probe_runtime: bool = False,
) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    saw_data_row = False
    try:
        for row in _iter_jsonl_payloads(path):
            if not isinstance(row, Mapping):
                return False
            if _is_runtime_header_row(row):
                continue
            saw_data_row = True
            if not _runtime_payload_ok(
                row,
                runtime_kind,
                runtime_kind_keys=runtime_kind_keys,
                shard_id=shard_id,
                checkpoint_role=checkpoint_role,
                require_decode_policy=require_decode_policy,
                require_gpu_id=require_gpu_id,
                require_trace_hash=require_trace_hash,
                require_probe_runtime=require_probe_runtime,
            ):
                return False
    except json.JSONDecodeError:
        return False
    return saw_data_row


def _runtime_payload_ok(
    payload: Mapping[str, Any],
    runtime_kind: str,
    *,
    runtime_kind_keys: tuple[str, ...] = ("runtime_kind",),
    shard_id: int | None = None,
    checkpoint_role: str | None = None,
    require_decode_policy: bool = False,
    require_gpu_id: bool = False,
    require_trace_hash: bool = False,
    require_checkpoint_fingerprint: bool = False,
    require_probe_runtime: bool = False,
) -> bool:
    if not any(str(payload.get(key)) == runtime_kind for key in runtime_kind_keys):
        return False
    if shard_id is not None and _coerce_shard_id(payload) != shard_id:
        return False
    if checkpoint_role is not None and str(payload.get("checkpoint_role")) != checkpoint_role:
        return False
    if require_decode_policy:
        if str(payload.get("decode_policy")) != DECODE_POLICY:
            return False
        if str(payload.get("constraint_policy")) != CONSTRAINT_POLICY:
            return False
    if require_gpu_id and not _nonempty(payload.get("gpu_id")):
        return False
    if require_trace_hash and not _nonempty(
        payload.get("token_trace_sha256", payload.get("raw_output_sha256"))
    ):
        return False
    if require_checkpoint_fingerprint and not _nonempty(
        payload.get("checkpoint_fingerprint")
    ):
        return False
    if require_probe_runtime and not _nonempty(
        payload.get("probe_runtime_id", payload.get("runtime_id"))
    ):
        return False
    return True


def _coerce_shard_id(payload: Mapping[str, Any]) -> int | None:
    try:
        shard_id = int(payload["shard_id"])
    except (KeyError, TypeError, ValueError):
        return None
    if shard_id < 0 or shard_id >= 8:
        return None
    return shard_id


def _nonempty(value: Any) -> bool:
    return value is not None and str(value) != ""


def _is_runtime_header_row(payload: Mapping[str, Any]) -> bool:
    return str(payload.get("row_type")) in {"shard_manifest", "manifest", "header"}


def _safe_gallery_relative_path(gallery_root: Path, rel_image: str) -> Path | None:
    candidate = Path(rel_image)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    path = gallery_root / candidate
    try:
        path.resolve().relative_to(gallery_root.resolve())
    except ValueError:
        return None
    return path


def _valid_jpeg_file(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size <= 0:
        return False
    if path.suffix.lower() not in {".jpg", ".jpeg"}:
        return False
    try:
        from PIL import Image
    except ImportError:
        return True
    try:
        with Image.open(path) as image:
            image.verify()
            return image.format == "JPEG"
    except Exception:
        return False


def _iter_structured_payloads(root: Path) -> Iterator[Any]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix == ".json":
            payload = _read_json(path)
            if payload is not None:
                yield payload
        elif path.suffix == ".jsonl":
            try:
                yield from _iter_jsonl_payloads(path)
            except json.JSONDecodeError:
                yield {"checkpoint_roles": ["<invalid-jsonl>"]}


def _read_json(path: Path) -> Any:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError:
        return None


def _iter_jsonl_payloads(path: Path) -> Iterator[Any]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


__all__ = [
    "FINAL_STATUS",
    "INDEX_READY_STATUS",
    "INCOMPLETE_STATUS",
    "RANDOM_ROLE",
    "REQUIRED_FINAL_ARTIFACTS",
    "SORTED_ROLE",
    "evaluate_status",
]
