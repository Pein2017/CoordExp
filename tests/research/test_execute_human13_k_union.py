from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research import execute_human13_k_union as entry


def _manifest(path: Path) -> tuple[Path, str]:
    payload = {
        "schema_version": "human13_k_union_manifest.v1",
        "binding": {
            "unit_id": entry.UNIT_ID,
            "purpose": "overfit_only",
            "panel": {"panel_sha256": entry.PANEL_SHA256},
            "source": {"checkpoint_path": entry.FROZEN_SOURCE.checkpoint_path},
        },
        "full_panel": True,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"
    path.write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    path.with_name(path.name + ".sha256").write_text(
        f"{digest}  {path.name}\n", encoding="ascii"
    )
    return path, digest


def _plan(tmp_path: Path, manifest_sha: str, *, arm_id: str = "A1") -> dict[str, object]:
    source = {
        "checkpoint_path": entry.FROZEN_SOURCE.checkpoint_path,
        "base_model_path": entry.FROZEN_SOURCE.base_model_path,
        "adapter_path": entry.FROZEN_SOURCE.adapter_path,
        "special_embedding_path": entry.FROZEN_SOURCE.special_embedding_path,
        "adapter_sha256": entry.FROZEN_SOURCE.adapter_sha256,
        "special_embedding_sha256": entry.FROZEN_SOURCE.special_embedding_sha256,
    }
    return {
        "schema_version": entry.PLAN_SCHEMA_VERSION,
        "unit_id": entry.UNIT_ID,
        "arm_id": arm_id,
        "updates": arm_id != "frozen_source",
        "source": source,
        "manifest_identity": {
            "schema_version": "human13_k_union_manifest.v1",
            "unit_id": entry.UNIT_ID,
            "panel_sha256": entry.PANEL_SHA256,
            "manifest_sha256": manifest_sha,
        },
        "global_max_length": 12_000,
        "milestones": [0, 1, 2, 4, 8, 16],
        "output_root": str(tmp_path / "arm"),
        "optimizer_state_root": str(tmp_path / "arm" / "optimizer_state"),
        "fresh_state_id": "fresh-state",
    }


def test_resolved_plan_requires_manifest_content_binding(tmp_path: Path) -> None:
    manifest, digest = _manifest(tmp_path / "manifest.json")
    path = tmp_path / "plan.json"
    path.write_text(json.dumps(_plan(tmp_path, digest)), encoding="utf-8")

    checked = entry.validate_resolved_plan(path, manifest)
    assert checked.arm_id == "A1"
    assert checked.manifest_sha256 == digest

    raw = json.loads(path.read_text(encoding="utf-8"))
    raw["manifest_identity"]["manifest_sha256"] = "0" * 64
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(entry.ExecutionContractError, match="manifest SHA"):
        entry.validate_resolved_plan(path, manifest)


def test_source_prompt_prefix_parity_uses_literal_ids_and_all_images() -> None:
    rows = {
        1584: {
            "image_sha256": entry.EXPECTED_IMAGE_SHA256[1584],
            "input_prompt_token_ids": [10, 11],
            "expected_executed_prompt_token_ids": [10, 11, 12],
            "executed_prompt_token_ids": [10, 11, 12],
            "generated_token_ids": [21, 22],
        }
    }
    result = entry.verify_source_prompt_prefix_parity(
        rows,
        expected_image_ids=(1584,),
        expected_prefix_token_ids={1584: (10, 11, 12)},
    )
    assert result[1584].prompt_token_ids == (10, 11, 12)
    assert result[1584].generated_token_ids == (21, 22)

    bad = dict(rows)
    bad[1584] = {**rows[1584], "executed_prompt_token_ids": [10, 99, 12]}
    with pytest.raises(entry.ExecutionContractError, match="prompt-prefix parity"):
        entry.verify_source_prompt_prefix_parity(
            bad,
            expected_image_ids=(1584,),
            expected_prefix_token_ids={1584: (10, 11, 12)},
        )


def test_exposure_schedule_has_fresh_roots_and_frozen_source_no_updates(tmp_path: Path) -> None:
    schedule = entry.plan_exposures(
        arm_id="A1", output_root=tmp_path / "a1", optimizer_state_root=tmp_path / "state"
    )
    assert tuple(item.milestone for item in schedule) == (0, 1, 2, 4, 8, 16)
    assert len({item.run_root for item in schedule}) == 6
    assert all(item.updates == (item.milestone > 0) for item in schedule)

    frozen = entry.plan_exposures(arm_id="frozen_source", output_root=tmp_path / "source")
    assert all(not item.updates for item in frozen)
    assert all(item.optimizer_state_root is None for item in frozen)


def test_runtime_entry_is_not_execution_ready_without_real_runtime(tmp_path: Path) -> None:
    manifest, digest = _manifest(tmp_path / "manifest.json")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(tmp_path, digest)), encoding="utf-8")

    receipt = entry.dry_run_resolved_plan(plan_path, manifest)
    assert receipt["actions"] == entry.ZERO_ACTIONS
    assert receipt["execution_ready"] is False
    with pytest.raises(entry.ExecutionContractError, match="real runtime"):
        entry.execute_resolved_plan(plan_path, manifest, execution_authorized=True)


def test_runtime_entry_rejects_forged_ready_bit_before_any_callback(tmp_path: Path) -> None:
    manifest, digest = _manifest(tmp_path / "manifest.json")
    plan_path = tmp_path / "plan.json"
    raw = _plan(tmp_path, digest)
    raw["execution_ready"] = True
    plan_path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(entry.ExecutionContractError, match="unknown fields"):
        entry.validate_resolved_plan(plan_path, manifest)


def test_runtime_entry_rejects_unbound_callback_receipt(tmp_path: Path) -> None:
    manifest, digest = _manifest(tmp_path / "manifest.json")
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan(tmp_path, digest)), encoding="utf-8")

    with pytest.raises(entry.ExecutionContractError, match="bound runtime receipt"):
        entry.execute_resolved_plan(
            plan_path,
            manifest,
            execution_authorized=True,
            runtime_factory=lambda *_: {"execution_ready": True},
        )
