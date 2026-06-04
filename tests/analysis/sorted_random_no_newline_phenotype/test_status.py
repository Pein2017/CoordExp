from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from src.analysis.sorted_random_no_newline_phenotype.status import (
    CONSTRAINT_POLICY,
    DECODE_POLICY,
    FINAL_STATUS,
    INDEX_READY_STATUS,
    INCOMPLETE_STATUS,
    RANDOM_ROLE,
    REAL_FN_HINT_RUNTIME_KIND,
    REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
    REAL_PREFIX_RUNTIME_KIND,
    SORTED_ROLE,
    evaluate_status,
)


def test_final_artifacts_present_only_when_all_required_surfaces_and_gates_pass(
    tmp_path: Path,
) -> None:
    _write_final_artifacts(tmp_path)

    result = evaluate_status(tmp_path)

    assert result["status"] == FINAL_STATUS
    assert result["final_artifacts_present"] is True
    assert result["failed_gates"] == []


def test_missing_required_artifacts_are_rejected(tmp_path: Path) -> None:
    _write_final_artifacts(tmp_path)
    (tmp_path / "rollout" / "rollout_summary.json").unlink()

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["final_artifacts_present"] is False
    assert "required_final_artifacts_present" in result["failed_gates"]
    assert "rollout/rollout_summary.json" in result["missing_final_artifacts"]


def test_missing_data_root_audit_fn_universe_and_fn_summaries_are_rejected(
    tmp_path: Path,
) -> None:
    _write_final_artifacts(tmp_path)
    (tmp_path / "data_root_audit.json").unlink()
    (tmp_path / "fn_probe" / "fn_case_universe.jsonl").unlink()
    (tmp_path / "fn_probe" / "fn_bucket_summary.json").unlink()

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert "data_root_audit_present" in result["failed_gates"]
    assert "fn_universe_present" in result["failed_gates"]
    assert "fn_probe_summaries_present" in result["failed_gates"]


def test_inprogress_legacy_labels_wrong_roles_and_row_count_mismatch_are_rejected(
    tmp_path: Path,
) -> None:
    _write_final_artifacts(tmp_path)
    (tmp_path / "worker-0.inprogress").write_text("still running", encoding="utf-8")
    with (tmp_path / "summary" / "report.md").open("a", encoding="utf-8") as handle:
        handle.write("\npure_minus_et\n")
    _write_json(
        tmp_path / "sample_manifest.json",
        {
            "checkpoint_roles": [RANDOM_ROLE, "fullobj_sorted_pure_ce_ckpt3664"],
            "template_contract": _template_contract(),
        },
    )
    _write_jsonl(
        tmp_path / "prefix_state_shard_summaries.jsonl",
        [
            {"shard_id": 0, "prefix_state_count": 2},
            {"shard_id": 1, "prefix_state_count": 1},
        ],
    )

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert "no_inprogress_leftovers" in result["failed_gates"]
    assert "no_legacy_a31_labels" in result["failed_gates"]
    assert "checkpoint_roles_are_a3_2" in result["failed_gates"]
    assert "shard_and_merged_row_counts_match" in result["failed_gates"]


def test_final_report_with_banned_causal_language_is_rejected(tmp_path: Path) -> None:
    _write_final_artifacts(tmp_path)
    (tmp_path / "summary" / "report.md").write_text(
        "## Scope And Evidence Labels\n"
        "This proved the root cause of the FN cases.\n",
        encoding="utf-8",
    )

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["final_artifacts_present"] is False
    assert "report_language_is_cautious" in result["failed_gates"]


def test_gallery_metadata_referenced_missing_images_are_rejected(
    tmp_path: Path,
) -> None:
    _write_final_artifacts(tmp_path)
    (tmp_path / "gallery" / "images" / "case-1.jpg").unlink()

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert "gallery_images_present" in result["failed_gates"]


def test_gallery_unreferenced_stale_images_are_rejected(tmp_path: Path) -> None:
    _write_final_artifacts(tmp_path)
    _write_jpeg(tmp_path / "gallery" / "images" / "stale-placeholder.jpg")

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert "gallery_images_present" in result["failed_gates"]


def test_status_streams_jsonl_without_read_text(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write_final_artifacts(tmp_path)
    original_read_text = Path.read_text

    def fail_jsonl_read_text(self: Path, *args, **kwargs) -> str:
        if self.suffix == ".jsonl":
            raise AssertionError(f"read_text used for JSONL: {self}")
        return original_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_jsonl_read_text)

    result = evaluate_status(tmp_path)

    assert result["status"] == FINAL_STATUS
    assert result["failed_gates"] == []


def test_index_ready_pending_gpu_for_cpu_index_ready_state(tmp_path: Path) -> None:
    _write_index_ready_artifacts(tmp_path)

    result = evaluate_status(tmp_path)

    assert result["status"] == INDEX_READY_STATUS
    assert result["index_ready_pending_gpu"] is True
    assert result["final_artifacts_present"] is False
    assert result["failed_gates"] == []
    assert "rollout/rollout_summary.json" in result["pending_final_artifacts"]


def test_missing_prefix_index_or_sampled_rows_prevents_index_ready(
    tmp_path: Path,
) -> None:
    _write_index_ready_artifacts(tmp_path)
    (tmp_path / "prefix_state_index.jsonl").unlink()

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["index_ready_pending_gpu"] is False
    assert "index_ready_artifacts_present" in result["failed_gates"]

    _write_index_ready_artifacts(tmp_path)
    (tmp_path / "prefix_state_sampled_rows.jsonl").unlink()

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["index_ready_pending_gpu"] is False
    assert "index_ready_artifacts_present" in result["failed_gates"]


def test_missing_expanded_final_artifacts_are_rejected(tmp_path: Path) -> None:
    cases = (
        "rollout/rollout_phenotype_rows.jsonl",
        "summary/prefix_readout_summary.json",
        f"rollout/{RANDOM_ROLE}/gt_vs_pred.jsonl",
        f"rollout/{SORTED_ROLE}/pred_token_trace.jsonl",
    )
    for rel_path in cases:
        case_root = tmp_path / rel_path.replace("/", "_")
        _write_final_artifacts(case_root)
        (case_root / rel_path).unlink()

        result = evaluate_status(case_root)

        assert result["status"] == INCOMPLETE_STATUS
        assert result["final_artifacts_present"] is False
        assert "required_final_artifacts_present" in result["failed_gates"]
        assert rel_path in result["missing_final_artifacts"]


def test_nonempty_synthetic_runtime_rows_do_not_satisfy_final_readiness(
    tmp_path: Path,
) -> None:
    _write_final_artifacts(tmp_path, real_runtime=False)

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["final_artifacts_present"] is False
    assert "real_runtime_artifacts_present" in result["failed_gates"]


def test_mixed_unproven_runtime_rows_do_not_satisfy_final_readiness(
    tmp_path: Path,
) -> None:
    _write_final_artifacts(tmp_path)
    _append_jsonl(
        tmp_path / "prefix_readout_shards" / "shard_0.jsonl",
        {"prefix_state_id": "unproven-prefix-row", "shard_id": 0},
    )
    _append_jsonl(
        tmp_path / "rollout" / RANDOM_ROLE / "gt_vs_pred.jsonl",
        {"role": RANDOM_ROLE, "prediction": "unproven-rollout-row"},
    )
    _append_jsonl(
        tmp_path / "rollout" / "rollout_phenotype_rows.jsonl",
        {"case": "unproven-rollout-phenotype-row"},
    )
    _append_jsonl(
        tmp_path / "fn_probe" / "fn_probe_rows.jsonl",
        {"case": "unproven-fn-probe-row"},
    )

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["final_artifacts_present"] is False
    assert "real_runtime_artifacts_present" in result["failed_gates"]


def test_missing_prefix_shard_surface_prevents_final_readiness(tmp_path: Path) -> None:
    _write_final_artifacts(tmp_path)
    (tmp_path / "prefix_readout_shards" / "shard_7.jsonl").unlink()

    result = evaluate_status(tmp_path)

    assert result["status"] == INCOMPLETE_STATUS
    assert result["final_artifacts_present"] is False
    assert "prefix_shard_surface_complete" in result["failed_gates"]
    assert "required_final_artifacts_present" in result["failed_gates"]


def _write_final_artifacts(root: Path, *, real_runtime: bool = True) -> None:
    _write_index_ready_artifacts(root)
    _write_jsonl(
        root / "prefix_state_shard_summaries.jsonl",
        [
            {
                "shard_id": shard_id,
                "prefix_state_count": 1,
                **_prefix_runtime_marker(shard_id, enabled=real_runtime),
            }
            for shard_id in range(8)
        ],
    )
    for shard_id in range(8):
        _write_jsonl(
            root / "prefix_readout_shards" / f"shard_{shard_id}.jsonl",
            [
                {
                    "prefix_state_id": f"state-{shard_id}",
                    "shard_id": shard_id,
                    **_prefix_runtime_marker(shard_id, enabled=real_runtime),
                }
            ],
        )
    _write_jsonl(
        root / "summary" / "prefix_readout_merged_rows.jsonl",
        [
            {"prefix_state_id": f"state-{shard_id}", "role_a": RANDOM_ROLE, "role_b": SORTED_ROLE}
            for shard_id in range(8)
        ],
    )
    _write_json(
        root / "summary" / "prefix_readout_summary.json",
        {"merged_prefix_state_count": 8},
    )
    (root / "summary").mkdir(parents=True, exist_ok=True)
    (root / "summary" / "report.md").write_text(
        "## Scope And Evidence Labels\n"
        "A3.2 evidence supports under this evidence scope and is consistent with "
        "phenotype context, not mechanism proof.\n",
        encoding="utf-8",
    )
    _write_json(root / "rollout" / "rollout_summary.json", {"status": "ok"})
    _write_jsonl(
        root / "rollout" / "rollout_phenotype_rows.jsonl",
        [
            {
                "case": "r",
                **_native_runtime_marker(RANDOM_ROLE, enabled=real_runtime),
            }
        ],
    )
    for role in (RANDOM_ROLE, SORTED_ROLE):
        _write_jsonl(
            root / "rollout" / role / "gt_vs_pred.jsonl",
            [
                {
                    "role": role,
                    **_native_runtime_marker(role, enabled=real_runtime),
                }
            ],
        )
        _write_jsonl(
            root / "rollout" / role / "pred_token_trace.jsonl",
            [
                {
                    "role": role,
                    "token_trace_sha256": "trace-sha256",
                    **_native_runtime_marker(role, enabled=real_runtime),
                }
            ],
        )
        _write_json(
            root / "rollout" / role / "summary.json",
            {
                "role": role,
                "checkpoint_fingerprint": f"{role}-fingerprint",
                **_native_runtime_marker(role, enabled=real_runtime),
            },
        )
    _write_jsonl(root / "fn_probe" / "fn_case_universe.jsonl", [{"case": "u"}])
    _write_jsonl(root / "fn_probe" / "fn_cases.jsonl", [{"case": "c"}])
    _write_jsonl(
        root / "fn_probe" / "fn_probe_rows.jsonl",
        [{"case": "p", **_fn_runtime_marker(enabled=real_runtime)}],
    )
    _write_jsonl(
        root / "fn_probe" / "fn_candidate_scores.jsonl",
        [{"case": "s", **_fn_runtime_marker(enabled=real_runtime)}],
    )
    _write_jsonl(
        root / "fn_probe" / "fn_slot_evidence.jsonl",
        [{"case": "e", **_fn_runtime_marker(enabled=real_runtime)}],
    )
    _write_json(
        root / "fn_probe" / "fn_bucket_summary.json",
        {"coord_binding_failure": 1, **_fn_runtime_marker(enabled=real_runtime)},
    )
    _write_json(
        root / "fn_probe" / "fn_prefix_sensitivity.json",
        {"rollout_prefix_loss": 1, **_fn_runtime_marker(enabled=real_runtime)},
    )
    _write_json(
        root / "fn_probe" / "fn_slot_rescue_summary.json",
        {"strict_r95_x1_hit_rate": 0.5, **_fn_runtime_marker(enabled=real_runtime)},
    )
    _write_gallery(root / "gallery")
    _write_gallery(root / "fn_probe" / "gallery")


def _write_index_ready_artifacts(root: Path) -> None:
    _write_jsonl(root / "prefix_state_index.jsonl", [{"prefix_state_id": "state-0"}])
    _write_jsonl(
        root / "prefix_state_sampled_rows.jsonl",
        [{"prefix_state_id": "state-0", "shard_id": 0}],
    )
    _write_json(
        root / "data_root_audit.json",
        {
            "status": "ok",
            "actual_train_jsonl": "/data/train.coord.jsonl",
            "actual_val_jsonl": "/data/val.coord.jsonl",
            "image_root": "/data/images",
        },
    )
    _write_json(
        root / "prefix_state_index_summary.json",
        {
            "status": "ok",
            "checkpoint_roles": [RANDOM_ROLE, SORTED_ROLE],
            "prefix_state_count": 2,
            "template_contract": _template_contract(),
        },
    )
    _write_json(
        root / "sample_manifest.json",
        {
            "checkpoint_roles": [RANDOM_ROLE, SORTED_ROLE],
            "template_contract": _template_contract(),
            "data_root_audit": "data_root_audit.json",
        },
    )


def _prefix_runtime_marker(shard_id: int, *, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {}
    return {
        "runtime_kind": REAL_PREFIX_RUNTIME_KIND,
        "gpu_id": str(shard_id),
    }


def _native_runtime_marker(role: str, *, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {}
    return {
        "runtime_kind": REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
        "source_runtime_kind": REAL_NATIVE_ROLLOUT_RUNTIME_KIND,
        "checkpoint_role": role,
        "gpu_id": "0",
        "decode_policy": DECODE_POLICY,
        "constraint_policy": CONSTRAINT_POLICY,
    }


def _fn_runtime_marker(*, enabled: bool) -> dict[str, object]:
    if not enabled:
        return {}
    return {
        "runtime_kind": REAL_FN_HINT_RUNTIME_KIND,
        "probe_runtime_id": "fn-hint-real-gpu-smoke",
    }


def _write_gallery(path: Path) -> None:
    (path / "images").mkdir(parents=True, exist_ok=True)
    (path / "index.md").write_text("gallery index\n", encoding="utf-8")
    _write_jpeg(path / "images" / "case-1.jpg")
    _write_json(
        path / "metadata.json",
        [
            {
                "case_id": "case-1",
                "relative_image_path": "images/case-1.jpg",
                "legend_placement": "right_panel",
                "legend_overlaps_image": False,
            }
        ],
    )


def _write_jpeg(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color=(240, 240, 240)).save(path, format="JPEG")


def _template_contract() -> dict[str, str]:
    return {
        "detection_sequence_format": "compact_full",
        "coordinate_surface": "coord_token",
        "bbox_format": "xyxy",
        "row_separator": "none",
    }


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, allow_nan=False, sort_keys=True) + "\n" for row in rows
        ),
        encoding="utf-8",
    )


def _append_jsonl(path: Path, row: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
